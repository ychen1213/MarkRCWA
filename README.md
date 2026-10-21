# MarkRCWA

用于一维周期结构 RCWA 仿真的 Windows x64 便携程序。

## 下载与运行

1. 打开 [Releases](https://github.com/ychen1213/MarkRCWA/releases/latest)，下载 `MarkRCWA-Windows-x64.zip`。
2. 完整解压压缩包。
3. 双击 `MarkRCWA/MarkRCWA.exe`，保留同目录的 `_internal` 文件夹。

支持 Windows 10/11 x64。程序已包含 Python 和运行依赖，无需另行安装 Python。

## 功能

- 导入 GDS，定义 Mark 图案，编辑材料和工艺步骤。
- 计算一维结构的 TE/TM 反射、透射、各衍射阶效率与复振幅。
- 扫描波长、入射角及几何参数，检查傅里叶阶数收敛。
- 导出反射矩阵 `R_matrix.xlsx`、JSON、CSV、HDF5 和图表。
- 保存和载入 `.mrcwa.json` 工程。

完整操作说明及本版计算范围见 [使用说明](使用说明.md)。

## 完整性校验

Releases 同时提供 `MarkRCWA-Windows-x64.zip.sha256`。在 PowerShell 中可计算下载文件的 SHA-256，并与校验文件核对：

```powershell
Get-FileHash .\MarkRCWA-Windows-x64.zip -Algorithm SHA256
```

## 许可

项目采用 [MIT License](LICENSE)。随程序分发的第三方组件及其许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和程序包内的许可证文件。
