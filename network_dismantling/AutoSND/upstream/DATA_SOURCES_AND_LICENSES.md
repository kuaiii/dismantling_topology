# 数据来源与许可证状态

本文档记录源研究快照中能够确认的数据来源信息，并严格区分学术来源与法律授权：
论文引用或下载地址本身不构成再发布许可证。

随仓库提供的文件均已规范化为无向、简单、整数节点边列表。规范化过程会移除自环，
将每条边统一为 `min(u,v) max(u,v)`，去除重复边，并对结果排序。网络规模和规范化
边集的 SHA-256 值记录在 `src/autosnd/data/manifest.json`。

| 数据集 | 已记录来源 | 本发布版中的处理 | 源快照中是否有许可证文本 |
|---|---|---|---|
| CEnew | 与已发表网络拆解方法配套的数据 | 规范化为无向简单图 | 无 |
| Collaboration | 本地研究归档；记录为与归档的 GrQC 图同构 | 规范化为无向简单图 | 无 |
| condmat | SNAP `ca-CondMat` | 移除自环；取最大连通分量；规范化 | 未随文件提供独立许可证文本 |
| crime | KONECT crime 二部图数据 | 折叠标识符空间；规范化为无向简单图 | 无 |
| email | 论文中归于 Guimerà/KONECT 的公开电子邮件网络 | 规范化为无向简单图 | 无 |
| Grid | SuiteSparse Matrix Collection `Newman/power` | 转换为规范化无向边列表；属于修改版本 | SuiteSparse 将其矩阵标记为 CC BY 4.0；必须保留署名和修改说明 |
| hamster | KONECT hamsterster 网络 | 取最大连通分量；规范化 | 无 |
| HepPh | SNAP `ca-HepPh` | 移除自环；取最大连通分量；规范化 | 未随文件提供独立许可证文本 |
| Yeast | WalkPooling 使用的 Yeast PPI 快照 | 规范化为无向简单图 | 无 |
| HI-II-14 | 本地研究归档 | 规范化为无向简单图 | 无 |
| Digg | 本地研究归档 | 规范化为无向简单图 | 无 |
| Gnutella31 | SNAP `p2p-Gnutella31` 的最大弱连通分量 | 转换为无向规范化简单图 | 未随文件提供独立许可证文本 |
| Powerlaw_500 | 本地生成的合成代理（`n=500`、`gamma=2.5`、`seed=42`、`k_min=2`） | 固定边列表，用于跨版本检查 | 项目生成；不属于 12 个主要网络 |

已知上游页面：

- SNAP ca-CondMat：<https://snap.stanford.edu/data/ca-CondMat.html>
- SNAP ca-HepPh：<https://snap.stanford.edu/data/ca-HepPh.html>
- SNAP p2p-Gnutella31：<https://snap.stanford.edu/data/p2p-Gnutella31.html>
- SuiteSparse Newman/power：<https://sparse.tamu.edu/Newman/power>
- Creative Commons Attribution 4.0：<https://creativecommons.org/licenses/by/4.0/>
- KONECT 项目：<https://konect.cc/>

## 权利声明

源研究快照没有仓库级 `LICENSE`、完整的数据集 `NOTICE`，也没有完整的逐数据集
许可证文本。因此：

1. 本文档不推定、创建或授予任何代码许可证。
2. 任何单一代码许可证都不会自动适用于图数据。
3. 本仓库不会授予超出相应权利人或上游来源已经授予的权利。
4. 再发布或商业使用前，必须逐一核查每个网络当时有效的上游条款；仅有本地归档来源
   时，应在需要的情况下替换为经过授权的来源。

CEnew、Collaboration、HI-II-14 和 Digg 的来源与许可证状态尤其不明确。

## Grid.edgelist 的署名与修改说明

`Grid.edgelist` 是 SuiteSparse Matrix Collection 中 `Newman/power` 的规范化修改版本。
原始网络由 D. J. Watts 和 S. H. Strogatz 编制，并可引用：

> D. J. Watts and S. H. Strogatz. Collective dynamics of small-world networks.
> Nature 393, 440--442 (1998).

AutoSND 将该矩阵转换为经过排序、去重的无向整数边列表。再发布此文件时，必须保留
SuiteSparse Matrix Collection、`Newman/power`、原作者和上述修改说明，链接
CC BY 4.0：<https://creativecommons.org/licenses/by/4.0/>。
