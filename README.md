# 本地时序指标查询引擎

一个受 PromQL 启发的本地时序数据库查询引擎（Python 3.14，零第三方运行时依赖）。
通过表达式完成指标选择、时间窗口聚合、多序列运算与结果分组，并在并发写入、
超时取消、重启恢复下保证一致性。

## 能力概览

| 需求 | 能力 | 关键语义 |
| --- | --- | --- |
| 1 | 表达式解析与校验 | 递归下降解析 + 全量静态类型校验；非法语法抛出带 `line/column/offset` 的 `QuerySyntaxError`，**解析失败不会触碰任何数据**，绝不返回空结果冒充成功 |
| 2 | 指标选择器与标签匹配 | 支持 `=`、`!=`、`=~`、`!~`；按名称和标签筛选单/多条序列；`{label=""}` 表示标签缺失 |
| 3 | 范围向量与窗口函数 | 窗口为**半开区间** `(t-offset-range, t-offset]`；缺失采样点**绝不补零**，点不足的序列不产出 |
| 4 | 二元运算与向量匹配 | 标量/向量四则与比较，`on`/`ignoring`/`group_left`/`group_right`/`bool` 对齐规则全部可配置；多对多未声明分组时报错 |
| 5 | 聚合与分组 | `sum/avg/count/min/max/stddev/stdvar/topk/bottomk/group/count_values`，`by(...)`/`without(...)`；每个分组只输出一次，标签增减可解释 |
| 6 | 并发一致性 | 每次写入原子替换不可变快照；查询开始时锁定**单一快照**，全程只读该版本，绝不跨提交点读到中间状态 |
| 7 | 超时与取消 | 每个求值步骤协作检查 deadline/取消信号；超时抛 `QueryTimeoutError`、取消抛 `QueryCancelledError`，不返回部分结果 |
| 8 | 重启恢复 | 写入先落 WAL(fsync) 再可见；定期 checkpoint 压缩；重启回放 checkpoint + WAL，撕裂的末尾记录被丢弃，已确认提交不丢 |

## 快速开始

```python
from tsengine import QueryEngine, Storage

store = Storage(data_dir="./data")          # data_dir=None 表示纯内存
store.append({"__name__": "http_requests", "job": "api", "code": "200"},
             timestamp_ms=1000, value=12.0)

engine = QueryEngine(store)

# 即时查询：窗口速率按 job 聚合
result = engine.execute('sum(rate(http_requests[5m])) by (job)',
                        at_ms=60_000, timeout_ms=1000)
for sample in result:
    print(dict(sample.labels), sample.value)
```

### 写入

```python
store.append(labels, timestamp_ms, value)                 # 单点（默认 fsync）
store.append_batch([(labels, ts, value), ...])            # 一批原子可见
```

标签必须包含 `__name__`；同一序列点按时间升序保存，相同时间戳覆盖旧值。

### 查询表达式示例

```promql
m                                          # 按名称选择
m{job="api", code=~"5..", path!~"/health"} # 标签选择
rate(m[5m])                                # 窗口速率
avg_over_time(m[1h] offset 1h)             # 偏移窗口均值
a + on(job) b                              # 按 job 对齐
a / ignoring(code) group_left(instance) b # 多对一
m > bool 10                                # 比较（不滤除，返回 0/1）
sum(rate(m[5m])) by (job, region)          # 分组聚合
topk(3, m)                                 # 参数化聚合
a and on(job) b                            # 集合运算
```

## 窗口边界语义（重点）

对于范围向量 `m[RANGE] offset OFFSET`，求值时刻 `t`：

```
window_start = t - OFFSET - RANGE   (排他，不含)
window_end   = t - OFFSET           (包含)
取点集合 = { p | window_start < p.timestamp <= window_end }
```

- 边界点落在 `window_start` 上时**被排除**；落在 `window_end` 上时**被包含**。
- 窗口内没有实际采样点的序列不会出现在结果中，而不是作为 0 参与计算。
- `rate`/`irate`/`increase`/`delta` 至少需要 2 个点，否则该序列无输出；
  `rate` 基于窗口内首末点并做计数器归零（reset）补偿。

## 二元运算对齐规则（可配置）

- **标量-标量**：确定的单值结果。
- **向量-标量**：标量对向量每个序列独立运算；比较默认过滤不满足的序列，
  加 `bool` 则保留全部并返回 `0/1`；除以 0 的序列被丢弃。
- **向量-向量**：
  - 默认要求**完整标签集一对一**；一对多且未声明 `group_*` 时抛
    `QueryTypeError`，不产生歧义结果。
  - `on(l1,l2)`：仅按列出标签对齐；`ignoring(l1,l2)`：忽略列出标签。
  - `group_left`/`group_right` 显式声明多对一；运算顺序始终是 `左 OP 右`，
    输出标签取自"多"侧并合并对侧的连接标签。
  - `and`/`or`/`unless` 按同样的匹配键求交/并/差。

## 聚合标签变化（可解释）

- `sum(...) by (job)`：输出标签**只有** `job`。
- `sum(...) without (code)`：输出标签为**除 `code` 外**的全部标签。
- `sum(...)`（无分组）：输出**空标签集**的单条结果。
- 每个分组键在结果中恰好出现一次。

## 超时与取消

```python
engine.execute(expr, at_ms=..., timeout_ms=100)   # 100ms 截止
# timeout_ms <= 0  => 立即超时；None => 不设截止

import threading
event = threading.Event()
ctx = QueryContext(cancel_event=event)
event.set()                  # 外部取消
engine.execute(expr, ctx=ctx)
```

## 持久化与恢复

- WAL：`data/wal.log`，每次提交追加并 `fsync`，一行为一条原子提交。
- Checkpoint：`data/checkpoint.json`，原子 `os.replace` 发布后清空 WAL。
- 重启：先加载 checkpoint，再顺序回放其后的 WAL；崩溃留下的**撕裂尾记录**
  （只可能是最后一行）被丢弃，之前已 fsync 的提交完整恢复；WAL 中间损坏
  会抛 `StorageError`，不静默吞掉。

## 测试

```bash
uv run pytest -q
```

覆盖：语法错误位置、窗口半开边界/缺失点不补零、向量匹配各模式、
并发快照一致性（读写多线程）、超时/取消及时退出、WAL/checkpoint 重启恢复。
