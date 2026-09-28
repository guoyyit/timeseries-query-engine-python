# 本地时序指标查询引擎（Python）

一个无外部时序数据库依赖的本地时序查询能力，提供 PromQL 风格的表达式语言、
不可变快照的并发控制、协作式超时/取消以及基于 WAL 的崩溃恢复。

## 能力总览

| 需求 | 实现 |
| --- | --- |
| 表达式解析与校验 | 手写词法/递归下降解析器；非法语法返回带 `line/col/offset` 的错误，且在校验通过前**不接触任何数据**，不会部分执行 |
| 指标选择器与标签匹配 | 名称选择、`{label="...", !=, =~, !~}` 匹配；正则在解析期编译校验 |
| 范围向量与窗口函数 | `metric[5m]` / `offset`；窗口为**闭区间**；缺失采样点一律不当作零；空窗口/样本不足的序列直接丢弃 |
| 二元运算与向量匹配 | 标量↔标量、标量↔向量、向量↔向量；`on(...)`、`ignoring(...)`、`group_left/group_right` 均可配置；歧义匹配（一对多、输出标签重复）直接报错 |
| 聚合与分组 | `sum/avg/count/min/max/topk/bottomk`，`by(...)` / `without(...)`；每个分组只输出一次，标签集变化规则固定（见下） |
| 并发一致性 | 写入串行化 + 不可变快照（`append_batch` 单 WAL 记录、单版本），查询始终运行在某一个已提交版本上 |
| 超时与取消 | `CancellationToken` 在所有循环边界检查；超时/取消抛错且无部分结果，工作线程随之停止；HTTP 层在客户端断连时取消 |
| 重启恢复 | 每条/每批写入 fsync 落 WAL；启动回放，支持 checkpoint；损坏记录显式报错而非静默忽略 |

## 表达式语言（示例）

```text
# 选择器
cpu
cpu{host="a", code=~"2..", zone!="edge"}
{__name__=~"cpu_.*"}

# 窗口函数（闭区间，缺失点不补零）
rate(cpu[5m])
increase(cpu[1h] offset 10m)
avg_over_time / sum_over_time / max_over_time / min_over_time / count_over_time

# 二元运算
requests / on(job) capacity
a * ignoring(instance) group_left(zone) b
capacity > bool 10            # bool 修饰：输出 0/1 而非过滤
a + on(job) group_right b

# 聚合
sum by (host) (rate(requests[1m]))
avg without (instance) (cpu)
topk(5, cpu)
```

## 关键语义（确定性约定）

- **窗口边界**：`[range]` 在求值时刻 `at` 上给出闭区间 `[at-offset-range, at-offset]`，
  两端均含。`rate/increase` 只使用窗口内首末两个点；末值小于首值视为计数器复位，
  增量取末值。
- **缺失采样**：瞬时选择器在陈旧窗口（默认 300s，闭区间）内无点 → 该序列不输出；
  范围函数空窗口 → 不输出。任何地方都不会把缺失补成 0。
- **对齐规则**：
  - 默认按两侧完整标签集对齐（`__name__` 隐式忽略，可用 `on(__name__)` 显式纳入）；
  - `on(l1,l2)`：仅以列出的标签连接；
  - `ignoring(l1,...)`：连接前剔除这些标签；
  - 一对一要求两侧键唯一；`group_left` 表示左多右一，`group_right` 表示右多左一，
    “一”侧重复直接报错；任何输出标签集重复也报错。
- **结果标签集**：
  - 一对一默认：左方标签减去 `ignoring`；右方独有标签丢弃；
  - 一对一 `on`：连接标签 + 左方独有标签；双方共有但不在连接列表中的标签丢弃（取值无歧义才保留）；
  - `group_left/right`：以“多”方标签为基础，`(...)` 中列出的标签从“一”方复制；
  - 任何算术/比较结果都不再保留指标名 `__name__`。
- **聚合标签变化**：
  - `by(x,y)`：输出标签集合恰为 `{x,y}`；
  - `without(x)`：输出保留除 `x`（及 `__name__`）外的全部标签；
  - 无分组：全部序列聚为一个无标签序列。
- **除零**：产生 `NaN`；比较默认按真假过滤，`bool` 修饰时输出 0/1。

## 运行

```bash
uv sync --group dev
# 设置 TSDB_WAL_PATH 后进程启动会自动回放 WAL；不设置则为纯内存模式
TSDB_WAL_PATH=./data/wal.log uv run uvicorn main:app --reload

# 写入（批量写入为原子提交）
curl -s localhost:8000/write/batch -H 'content-type: application/json' -d '{
  "writes": [
    {"metric":"cpu","labels":{"host":"a"},"samples":[{"t":100,"v":1.5}]},
    {"metric":"cpu","labels":{"host":"b"},"samples":[{"t":100,"v":2.5}]}
  ]
}'

# 查询
curl -s 'localhost:8000/query?expr=sum%20by%20(host)%20(cpu)&at=100&timeout=2'
```

错误响应示例（语法错误带位置）：

```json
{"error":"syntax_error","message":"...","pos":3,"line":1,"col":4}
```

## 代码结构

```
tsdb/
  errors.py    错误类型（语法/校验/超时/取消/存储）
  model.py     标签、采样点、向量、矩阵、匹配配置与结果标签推导
  lexer.py     词法分析（token 带行列位置）
  parser.py    AST + 递归下降解析 + 语义校验
  functions.py 窗口函数与聚合算子
  storage.py  不可变快照存储、原子批量写、WAL/checkpoint
  engine.py   执行器（选择/运算/聚合）、取消令牌
  api.py      FastAPI：/query、/write、/write/batch、/health
tests/         语法、窗口、向量匹配、并发一致性、重启恢复、HTTP 测试
```

## 测试

```bash
uv run pytest -q
```
