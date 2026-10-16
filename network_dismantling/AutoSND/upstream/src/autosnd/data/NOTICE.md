# 随包提供的图数据声明

AutoSND wheel 包含 12 个经过规范化的真实网络边列表和 1 个固定的合成搜索代理。
学术来源汇总在仓库根目录的 `DATA_SOURCES_AND_LICENSES.md` 和
`manifest.json` 中。

源研究快照没有仓库级代码许可证，也没有完整的逐数据集许可证文本。任何代码
许可证都不会自动适用于这些图文件；本软件包也不会授予超出相应权利人或上游来源
已经授予的权利。

再发布或商业使用前，必须逐一核查每个网络当时有效的上游条款。CEnew、
Collaboration、HI-II-14 和 Digg 的来源与许可证状态尤其不明确。

已知上游页面：

- SNAP ca-CondMat：https://snap.stanford.edu/data/ca-CondMat.html
- SNAP ca-HepPh：https://snap.stanford.edu/data/ca-HepPh.html
- SNAP p2p-Gnutella31：https://snap.stanford.edu/data/p2p-Gnutella31.html
- SuiteSparse Newman/power：https://sparse.tamu.edu/Newman/power
- KONECT 项目：https://konect.cc/

`Grid.edgelist` 是 SuiteSparse `Newman/power` 的规范化修改版本。原始网络由
D. J. Watts 和 S. H. Strogatz 编制，可引用其 1998 年发表于 Nature 的论文
“Collective dynamics of small-world networks”。AutoSND 将该矩阵转换为经过
排序、去重的无向整数边列表。SuiteSparse 将其矩阵标记为 CC BY 4.0；再发布该
文件时，必须保留原作者与 SuiteSparse 署名、许可证链接和本修改说明：
https://creativecommons.org/licenses/by/4.0/
