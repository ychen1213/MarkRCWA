"""Reflection Jones matrices and a dependency-free desktop Excel export.

This module only postprocesses existing solver records.  In particular it does
not infer electric-field amplitude by taking a square root of power efficiency.
The existing fixed laboratory electric-field basis and phase are retained.
"""
from __future__ import annotations

import cmath
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape, quoteattr


ELEMENTS = ("Rss", "Rsp", "Rps", "Rpp")
_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def build_reflection_matrices(records):
    """Pair TE/TM records into per-order matrices without inventing missing data.

    Rows are output s/p; columns are unit incident s/p.  Off-diagonal zeros
    apply only to the current isotropic, one-dimensional, nonconical solver.
    A missing polarization or an order outside its Fourier truncation remains
    None.  Group order follows the supplied records; diffraction orders sort.
    """
    grouped = {}
    for record in records:
        pol = record["polarization"]
        if pol not in ("TE", "TM"):
            raise ValueError(f"不支持的偏振：{pol}")
        key = tuple(record.get(k) for k in ("wavelength_nm", "theta_deg", "geometry_value_um"))
        group = grouped.setdefault(key, {})
        if pol in group:
            raise ValueError("同一波长、角度和几何点存在重复偏振结果，无法唯一组成 R 矩阵")
        group[pol] = record

    points = []
    for (wavelength, theta, geometry), sources in grouped.items():
        first = next(iter(sources.values()))
        for field in ("phase_reference_incident_z_um", "phase_reference_reflection_z_um"):
            values = [r.get(field) for r in sources.values()]
            if any(value != values[0] for value in values):
                raise ValueError("TE/TM 相位参考面不一致，不能直接合并为 R 矩阵")
        by_pol = {pol: {int(row["order"]): row for row in source["orders"]}
                  for pol, source in sources.items()}
        orders = sorted({m for rows in by_pol.values() for m in rows})
        rows = []
        for m in orders:
            available = [pol_rows[m] for pol_rows in by_pol.values() if m in pol_rows]
            row = {"order": m, "Rss": None, "Rsp": 0j, "Rps": 0j, "Rpp": None,
                   "propagating": bool(available[0].get("propagating", False)),
                   "phase_reliable": {element: False for element in ELEMENTS},
                   "element_status": {"Rss": "not_computed", "Rsp": "symmetry_zero",
                                      "Rps": "symmetry_zero", "Rpp": "not_computed"}}
            for pol, element in (("TE", "Rss"), ("TM", "Rpp")):
                source = by_pol.get(pol, {}).get(m)
                if source is not None:
                    row[element] = complex(source["amplitude_real"], source["amplitude_imag"])
                    row["phase_reliable"][element] = bool(source.get("phase_reliable", False))
                    row["element_status"][element] = "computed"
            rows.append(row)
        points.append({"wavelength_nm": wavelength, "theta_deg": theta,
                       "geometry_value_um": geometry,
                       "phase_reference_incident_z_um": first.get("phase_reference_incident_z_um"),
                       "phase_reference_reflection_z_um": first.get("phase_reference_reflection_z_um"),
                       "fourier_order": {pol: sources.get(pol, {}).get("fourier_order") for pol in ("TE", "TM")},
                       "convergence_status": {pol: sources.get(pol, {}).get("convergence_status", "not_checked") for pol in ("TE", "TM")},
                       "polarizations": [pol for pol in ("TE", "TM") if pol in sources],
                       "rows": rows})
    return points


def _incident_pair(incident):
    if len(incident) != 2:
        raise ValueError("入射复振幅需包含 Es_in、Ep_in 两个值")
    pair = tuple(complex(value) for value in incident)
    if not all(math.isfinite(v.real) and math.isfinite(v.imag) for v in pair):
        raise ValueError("入射复振幅必须为有限数值")
    return pair


def amplitude_for_row(row, incident=(1 + 0j, 0j)):
    """Return Es_out/Ep_out = R @ incident, respecting uncomputed entries.

    A missing coefficient matters only when its incident component is nonzero.
    Zero incident weight does not fabricate a missing matrix coefficient.
    """
    es, ep = _incident_pair(incident)

    def combine(left, right):
        total = 0j
        for element, weight in ((left, es), (right, ep)):
            if weight != 0:
                if row[element] is None:
                    return None
                total += row[element] * weight
        return total

    return {"Es_out": combine("Rss", "Rsp"), "Ep_out": combine("Rps", "Rpp")}


def complex_text(value):
    """Excel-readable a+bi text; unknown remains blank."""
    if value is None:
        return None
    return f"{value.real:.16g}{value.imag:+.16g}i"


def _phase_deg(value):
    return math.degrees(cmath.phase(value)) if value is not None and abs(value) > 0 else None


def _derived_phase_reliable(row, output, incident, value):
    if value is None or abs(value) == 0 or not row["propagating"]:
        return False
    elements = ("Rss", "Rsp") if output == "Es_out" else ("Rps", "Rpp")
    return all(weight == 0 or row[element] == 0 or row["phase_reliable"][element]
               for element, weight in zip(elements, incident))


def amplitude_phase_reliable(row, incident=(1 + 0j, 0j)):
    """Reliable-phase flags for the two postprocessed output components."""
    incident = _incident_pair(incident)
    return {name: _derived_phase_reliable(row, name, incident, value)
            for name, value in amplitude_for_row(row, incident).items()}


def export_reflection_excel(records, path, incident=(1 + 0j, 0j)):
    """Write a real .xlsx containing R, complex amplitudes and conventions.

    Input Jones amplitudes affect only derived output columns, never RCWA or
    the matrix.  Exported values are a snapshot of the selected input pair.
    """
    incident = _incident_pair(incident)
    points = build_reflection_matrices(records)
    path = Path(path)
    if path.suffix.lower() != ".xlsx":
        path = path.with_suffix(".xlsx")
    path.parent.mkdir(parents=True, exist_ok=True)

    common_headers = ["组", "波长_nm", "入射角_deg", "几何扫描_um", "衍射级_m", "传播级"]
    matrix_headers = common_headers + ["入射参考_z_um", "反射参考_z_um", "TE_半阶数_N", "TM_半阶数_N", "TE_收敛状态", "TM_收敛状态"]
    amplitude_headers = list(common_headers)
    for element in ELEMENTS:
        matrix_headers += [element, element + "_实部", element + "_虚部", element + "_状态", element + "_相位可靠"]
        amplitude_headers += ["|" + element + "|", "arg(" + element + ")_deg_原始", element + "_相位可靠"]
    for output in ("Es_out", "Ep_out"):
        amplitude_headers += [output, output + "_实部", output + "_虚部", "|" + output + "|",
                              "arg(" + output + ")_deg_原始", output + "_相位可靠"]
    matrix_rows, amplitude_rows = [matrix_headers], [amplitude_headers]
    status_text = {"computed": "已计算", "not_computed": "未计算", "symmetry_zero": "对称性为零"}
    for index, point in enumerate(points, 1):
        for row in point["rows"]:
            common = [index, point["wavelength_nm"], point["theta_deg"], point["geometry_value_um"],
                      row["order"], row["propagating"]]
            matrix = common + [point["phase_reference_incident_z_um"], point["phase_reference_reflection_z_um"],
                               point["fourier_order"]["TE"], point["fourier_order"]["TM"],
                               point["convergence_status"]["TE"], point["convergence_status"]["TM"]]
            amplitudes = list(common)
            for element in ELEMENTS:
                value = row[element]
                matrix += [complex_text(value), value.real if value is not None else None,
                           value.imag if value is not None else None, status_text[row["element_status"][element]],
                           row["phase_reliable"][element]]
                amplitudes += [abs(value) if value is not None else None, _phase_deg(value),
                               row["phase_reliable"][element]]
            for output, value in amplitude_for_row(row, incident).items():
                amplitudes += [complex_text(value), value.real if value is not None else None,
                               value.imag if value is not None else None, abs(value) if value is not None else None,
                               _phase_deg(value), _derived_phase_reliable(row, output, incident, value)]
            matrix_rows.append(matrix)
            amplitude_rows.append(amplitudes)

    notes = [["项目", "定义与说明"],
             ["R 矩阵", "[Es_out, Ep_out]^T = [[Rss, Rsp], [Rps, Rpp]] · [Es_in, Ep_in]^T；行是输出偏振，列是输入偏振。"],
             ["单位入射", "Rss/Rps 对应单位 s 入射，Rsp/Rpp 对应单位 p 入射；矩阵元素本身就是复电场振幅系数 a+bi，不是功率，也不是功率开方。"],
             ["偏振基", "s=TE、p=TM，保留当前 PH 固定实验室电场基。若改用反射 p=k×s 的局部基，Rpp 的符号需取反；本文件未作此转换。"],
             ["交叉偏振", "当前模型为一维、各向同性、非圆锥入射，Rsp=Rps=0 是该模型对称性结果；不代表已支持圆锥入射或各向异性。"],
             ["缺失数据", "仅计算 TE 时 Rpp 为空，仅计算 TM 时 Rss 为空；不同截断阶数取阶次并集，未算阶次保持空白并标为未计算。"],
             ["相位参考", "exp(-iωt)；物理 x 原点；物理 z 向上。入射和反射参考面见 R矩阵 表的对应 z 列；沿用求解记录的复系数与相位。"],
             ["原始相位", "arg 的范围为 [-180,180] 度；非零复数均列出原始相位。零振幅相位为空；传播级及求解记录中的相位可靠标记单独保留。"],
             ["入射 Es_in", complex_text(incident[0])], ["入射 Ep_in", complex_text(incident[1])],
             ["输出复振幅", "Es_out=Rss*Es_in+Rsp*Ep_in；Ep_out=Rps*Es_in+Rpp*Ep_in。默认单位 s 输入(1,0)，所选输入仅用于结果后处理。"],
             ["导出快照", "复振幅按上述入射值在导出时计算。更改输入请在结果页重新导出；Excel 中的复值字符串便于阅读，实部/虚部列为数值。"],
             ["输出缺失", "仅当非零入射分量需要的系数缺失时，该输出为空。输入权重为零不需要该列系数；不据此补造矩阵值。"],
             ["收敛", "此导出不新增计算或验证；传播级和相位可靠标志不等于已通过数值收敛检查。"],
             ["幅值与衍射级图", "图中显示第一组条件的输出 s/p 复振幅绝对值，包含传播阶与倏逝阶；多组完整数值按组号筛选。"]]
    chart = _amplitude_chart(points, amplitude_headers, amplitude_rows) if points and points[0]["rows"] else None
    _write_xlsx(path, [("R矩阵", matrix_rows), ("复振幅", amplitude_rows), ("说明", notes)], chart=chart)
    return path


def _column(index):
    value, result = index + 1, ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _xml(body):
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + body


def _cell(value, reference, *, header=False):
    if value is None:
        return ""
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return ""
        return f'<c r="{reference}" s="{0 if isinstance(value, int) else 2}"><v>{value!r}</v></c>'
    text = escape(str(value))
    return f'<c r="{reference}" s="{1 if header else 0}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'


def _worksheet(rows, *, notes=False, drawing=False):
    count = max(map(len, rows), default=1)
    last = f"{_column(count - 1)}{max(1, len(rows))}"
    split = 0 if notes else 6
    freeze = f'<pane xSplit="{split}" ySplit="1" topLeftCell="{_column(split)}2" activePane="{"bottomLeft" if notes else "bottomRight"}" state="frozen"/>'
    width = ('<col min="1" max="1" width="24" customWidth="1"/><col min="2" max="2" width="120" customWidth="1"/>' if notes else
             f'<col min="1" max="6" width="16" customWidth="1"/><col min="7" max="{count}" width="24" customWidth="1"/>')
    data = []
    for number, row in enumerate(rows, 1):
        cells = "".join(_cell(value, f"{_column(index)}{number}", header=number == 1) for index, value in enumerate(row))
        data.append(f'<row r="{number}" ht="{48 if notes else 22}" customHeight="1">{cells}</row>')
    return _xml(f'<worksheet xmlns="{_NS}" xmlns:r="{_REL}"><dimension ref="A1:{last}"/>'
                f'<sheetViews><sheetView workbookViewId="0">{freeze}</sheetView></sheetViews>'
                f'<sheetFormatPr defaultRowHeight="22"/><cols>{width}</cols><sheetData>{"".join(data)}</sheetData>'
                + (f'<autoFilter ref="A1:{last}"/>' if not notes else '')
                + ('<drawing r:id="rId1"/>' if drawing else '') + '</worksheet>')


def _amplitude_chart(points, headers, data):
    count = len(points[0]["rows"])
    series = []
    for number, (name, color) in enumerate((("Es_out", "1769AA"), ("Ep_out", "A23E96"))):
        column = headers.index("|" + name + "|")
        values = [row[column] for row in data[1:count + 1]]
        if all(value is None for value in values):
            continue
        x_values = [row[4] for row in data[1:count + 1]]

        def numeric_ref(formula, values):
            cache = ''.join(f'<c:pt idx="{i}"><c:v>{v!r}</c:v></c:pt>' for i, v in enumerate(values) if v is not None)
            return f'<c:numRef><c:f>{formula}</c:f><c:numCache><c:formatCode>General</c:formatCode><c:ptCount val="{count}"/>{cache}</c:numCache></c:numRef>'

        x_formula = "'复振幅'!$E$2:$E$" + str(count + 1)
        y_formula = "'复振幅'!$" + _column(column) + "$2:$" + _column(column) + "$" + str(count + 1)
        series.append(f'<c:ser><c:idx val="{number}"/><c:order val="{number}"/><c:tx><c:v>|{name}|</c:v></c:tx>'
                      f'<c:spPr><a:ln w="25400"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:ln></c:spPr>'
                      '<c:marker><c:symbol val="circle"/><c:size val="4"/></c:marker>'
                      f'<c:xVal>{numeric_ref(x_formula, x_values)}</c:xVal>'
                      f'<c:yVal>{numeric_ref(y_formula, values)}</c:yVal></c:ser>')
    if not series:
        return None
    title = f'组1：{points[0]["wavelength_nm"]:g} nm，{points[0]["theta_deg"]:g}°；|E| vs 衍射级 m'
    axes = ''.join(f'<c:valAx><c:axId val="{axis}"/><c:scaling><c:orientation val="minMax"/></c:scaling>'
                   f'<c:axPos val="{position}"/><c:numFmt formatCode="General" sourceLinked="1"/>'
                   f'<c:crossAx val="{other}"/><c:crosses val="autoZero"/><c:crossBetween val="midCat"/></c:valAx>'
                   for axis, other, position in ((1, 2, "b"), (2, 1, "l")))
    return _xml('<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                f'<c:chart><c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>{escape(title)}</a:t></a:r></a:p></c:rich></c:tx></c:title>'
                f'<c:plotArea><c:layout/><c:scatterChart><c:scatterStyle val="lineMarker"/>{"".join(series)}'
                f'<c:axId val="1"/><c:axId val="2"/></c:scatterChart>{axes}</c:plotArea>'
                '<c:legend><c:legendPos val="b"/><c:layout/></c:legend><c:plotVisOnly val="1"/>'
                '<c:dispBlanksAs val="gap"/></c:chart></c:chartSpace>')


def _write_xlsx(path, sheets, *, chart=None):
    """Small standards-based OOXML writer; no Excel/Python add-ons on user PC."""
    content = ['<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
               '<Default Extension="xml" ContentType="application/xml"/>',
               '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
               '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
    workbook_sheets, relationships = [], []
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for index, (name, rows) in enumerate(sheets, 1):
            content.append(f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
            workbook_sheets.append(f'<sheet name={quoteattr(name)} sheetId="{index}" r:id="rId{index}"/>')
            relationships.append(f'<Relationship Id="rId{index}" Type="{_REL}/worksheet" Target="worksheets/sheet{index}.xml"/>')
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _worksheet(rows, notes=name == "说明", drawing=bool(chart and index == 2)))
        relationships.append(f'<Relationship Id="rId4" Type="{_REL}/styles" Target="styles.xml"/>')
        archive.writestr("xl/workbook.xml", _xml(f'<workbook xmlns="{_NS}" xmlns:r="{_REL}"><sheets>{"".join(workbook_sheets)}</sheets></workbook>'))
        archive.writestr("xl/_rels/workbook.xml.rels", _xml(f'<Relationships xmlns="{_PKG_REL}">{"".join(relationships)}</Relationships>'))
        archive.writestr("_rels/.rels", _xml(f'<Relationships xmlns="{_PKG_REL}"><Relationship Id="rId1" Type="{_REL}/officeDocument" Target="xl/workbook.xml"/></Relationships>'))
        archive.writestr("xl/styles.xml", _xml(f'<styleSheet xmlns="{_NS}">'
            '<numFmts count="1"><numFmt numFmtId="164" formatCode="0.00000000000000E+00"/></numFmts>'
            '<fonts count="2"><font><sz val="11"/><name val="Microsoft YaHei"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Microsoft YaHei"/></font></fonts>'
            '<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF17365D"/><bgColor indexed="64"/></patternFill></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf>'
            '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf>'
            '<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'))
        if chart:
            content += ['<Override PartName="/xl/drawings/drawing1.xml" ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>',
                        '<Override PartName="/xl/charts/chart1.xml" ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>']
            archive.writestr("xl/worksheets/_rels/sheet2.xml.rels", _xml(f'<Relationships xmlns="{_PKG_REL}"><Relationship Id="rId1" Type="{_REL}/drawing" Target="../drawings/drawing1.xml"/></Relationships>'))
            archive.writestr("xl/drawings/_rels/drawing1.xml.rels", _xml(f'<Relationships xmlns="{_PKG_REL}"><Relationship Id="rId1" Type="{_REL}/chart" Target="../charts/chart1.xml"/></Relationships>'))
            archive.writestr("xl/drawings/drawing1.xml", _xml('<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                '<xdr:twoCellAnchor><xdr:from><xdr:col>31</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>1</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:from>'
                '<xdr:to><xdr:col>45</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>21</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:to>'
                '<xdr:graphicFrame macro=""><xdr:nvGraphicFramePr><xdr:cNvPr id="2" name="复振幅与衍射级"/><xdr:cNvGraphicFramePr/></xdr:nvGraphicFramePr>'
                '<xdr:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/></xdr:xfrm><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart">'
                f'<c:chart xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:r="{_REL}" r:id="rId1"/>'
                '</a:graphicData></a:graphic></xdr:graphicFrame><xdr:clientData/></xdr:twoCellAnchor></xdr:wsDr>'))
            archive.writestr("xl/charts/chart1.xml", chart)
        archive.writestr("[Content_Types].xml", _xml('<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' + ''.join(content) + '</Types>'))
