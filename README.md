# MarkRCWA 源码版

用于 Windows 上一维周期结构 RCWA 仿真的可编辑 Python 程序。仓库只包含源码、启动脚本、示例和说明，使用已有的 Python 与依赖环境；不包含打包 EXE、Python 运行时或第三方二进制。运行时离线，无需管理员权限，也无需安装本项目。

## 下载与启动

1. 点击仓库上方 **Code → Download ZIP**，完整解压到当前账户可写的目录；也可以使用 `git clone`。
2. 双击 **`检查环境.bat`**，确认实际使用的 Python、依赖版本及源码路径。
3. 检查通过后，双击 **`启动RCWA.bat`**。

需要 Python **3.11 或 3.12**（含 Tk）和 **Meent 0.13.2**。其他依赖及参考版本见 [requirements-source.txt](requirements-source.txt)。脚本只检查环境，不安装或下载组件。

如果有多个 Python 环境，用记事本打开 `启动RCWA.bat`，按开头注释指定 `MARKRCWA_PYTHON` 为已有环境中 `python.exe` 的完整路径。也可以直接运行：

```powershell
python run_rcwa.py --check-env
python run_rcwa.py
```

## 修改代码

直接编辑 `src/mark_rcwa` 中的 `.py` 文件，保存后关闭软件并重新启动即可生效，无需打包或重新安装。启动器使用本目录的源码。

| 修改内容 | 文件 |
| --- | --- |
| 界面、按钮与交互 | [desktop.py](src/mark_rcwa/desktop.py) |
| 工程保存、任务与扫描 | [desktop_project.py](src/mark_rcwa/desktop_project.py) |
| 材料文件读取 | [materials.py](src/mark_rcwa/modeling/materials.py) |
| RCWA 数值适配 | [meent_adapter.py](src/mark_rcwa/solvers/meent_adapter.py) |
| R 矩阵与 Excel 输出 | [reflection_matrix.py](src/mark_rcwa/reflection_matrix.py) |

请保留自行修改的源码副本，更新时先比较差异，避免覆盖自己的改动。

## 功能

- 导入 GDS，定义 Mark 图案，编辑材料和工艺，查看截面与本地 3D 预览。
- 一维 TE/TM RCWA 仿真、波长/角度/几何扫描、傅里叶阶数收敛检查。
- 输出 `R_matrix.xlsx`，包含 `Rss/Rsp/Rps/Rpp`、复数实部和虚部、幅值、相位及与衍射级的关系。
- 保存和载入 `.mrcwa.json` 工程，导出 JSON、CSV、HDF5 和图表。
- 自建材料 TXT/CSV 支持 UTF-8、GBK/GB18030 及带 BOM 的 UTF-16/UTF-32；导入后显示读取编码。

材料示例见 [materials/示例材料.txt](materials/示例材料.txt)，其中数值为合成示例。输入时需要明确波长单位和列号；没有 k 时，需要明确选择无损近似或补充真实 k 数据。

## 说明与计算范围

入门操作见 [源码版使用说明](使用说明_源码版.txt)，详细建模、矩阵约定和计算范围见 [桌面程序说明](docs/DESKTOP.md)。

桌面界面目前支持一维、各向同性、非锥形入射，交叉偏振项按对称性为零；锥形计算接口尚未接入界面。二维 GDS 可以预览，但沿 y 变化的结构不能直接作为一维模型求解。默认设置尚不能保证全部小振幅复数矩阵元素达到 1% 相对精度，需要针对实际结构检查收敛。修改计算代码后，应重新验证对应数值结果。

本次源码交付已验证移动目录后启动、修改源码后重启生效，以及材料编码读取；没有因更换交付形式重跑整套 RCWA 基准。源码中的独立基准辅助模块需要额外的参考工具和数据，完整开发测试与报告未随此运行目录发布。

## 许可

项目采用 [MIT License](LICENSE)。第三方组件说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
