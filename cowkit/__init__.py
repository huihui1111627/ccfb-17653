"""cowkit —— Redis 类键值库 fork()/COW 内存与延迟模拟工具包。

模块划分
--------
pages      : 键值如何落到内存页上（小键挤页、大键跨页、删除空洞、THP 大页）
workloads  : 四种真实业务写入模式
analytic   : 被复制页数的闭式公式（与仿真互相校验）
engine     : 持久化快照期间的 COW 逐拍仿真
real_fork  : 真的 fork() 派生子进程的本机实测
sweep      : 数据量/写入速率/大页开关/写入模式组合推演
recommend  : 基于推演结果生成建议与调整后效果
report     : 汇总成 dashboard 使用的 JSON
"""

__version__ = "1.0.0"
