"""Visible GDS import and one-dimensional Mark design workflows."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import uuid


class PatternTools:
    def _pattern_candidate(self):
        candidate = copy.deepcopy(self.project)
        if self.step_dirty and self.step_index is not None:
            candidate["process"]["steps"][self.step_index] = self._step_from_form()
        return candidate

    def _pattern_target(self, index=None):
        steps = self.project["process"]["steps"]
        if index == -1:
            return None
        if index is not None:
            if not 0 <= index < len(steps):
                raise ValueError("目标工艺步骤不存在")
            if steps[index]["operation"] == "create_substrate":
                raise ValueError("不能用图案替换衬底；请选择图案步骤或新增步骤")
            return index
        current = self.step_index
        if current is not None and steps[current].get("mask") and steps[current]["operation"] in ("add","etch","delete"):
            return current
        return next((i for i,s in enumerate(steps) if s.get("mask") and s["operation"] in ("add","etch","delete")),None)

    def _install_pattern(self, candidate, index, *, build):
        self.project = candidate
        self.step_dirty = False
        self.step_index = None
        self.refresh_steps()
        self.step_tree.selection_set(str(index))
        self.step_tree.see(str(index))
        self.select_step()
        self.preview_dir = None
        self.preview_image.configure(image="",text="图案已更新，正在生成几何。" if build else "图案已更新，点击生成几何查看。")
        self.layer_tree.delete(*self.layer_tree.get_children())
        self.changed()
        self.tabs.select(self.pages["process"])
        if build:
            self.build_preview()
        return index

    def apply_mark_definition(self, spec, *, target_index=None, update_cell=False, build=True):
        from .modeling.patterns import make_pattern
        from .modeling import build_geometry
        self._require_process()
        if self.busy:
            raise ValueError("请等待当前任务完成")
        candidate = self._pattern_candidate()
        recipe = candidate["process"]
        result = make_pattern(spec,y_bounds=recipe["cell_bounds_um"][1])
        period = result["period_um"]
        if not math.isclose(period,recipe["period_x_um"],rel_tol=1e-12,abs_tol=1e-12) and not update_cell:
            raise ValueError("Mark 母周期与当前单元不同；勾选‘同步计算单元’后应用，或保持原周期。")
        if update_cell:
            recipe["period_x_um"] = period
            recipe["cell_bounds_um"][0] = [-period/2,period/2]
        index = self._pattern_target(target_index)
        old = recipe["steps"][index] if index is not None else {}
        z = float(spec.get("z_min_um",old.get("z_min_um",0))) if "z_min_um" in spec else float(old.get("z_min_um",0))
        height = result["spec"]["height_um"]
        material = result["spec"]["material"]
        if material not in candidate["materials"]["materials"]:
            raise ValueError(f"没有材料 {material}；请先在材料页添加")
        # Pattern polygons are centred at x=0. Account for a cell whose origin
        # was chosen elsewhere, without changing its period unless requested.
        center_of_cell = sum(recipe["cell_bounds_um"][0])/2
        polygons = [[[x+center_of_cell,y] for x,y in p] for p in result["polygons_xy_um"]]
        step = {"id":old.get("id","mark_"+uuid.uuid4().hex[:10]),"name":"Mark · "+result["spec"]["kind"],
                "operation":"add","material":material,"enabled":True,"z_min_um":z,"z_max_um":z+height,
                "sidewall_angle_deg":90,"cd_bias_um":0.,"delta_x_um":0.,"delta_y_um":0.,"rotation_deg":0.,
                "mask":{"polygons_xy_um":polygons},"mark_pattern":result["spec"]}
        recipe["cell_bounds_um"][2][0] = min(recipe["cell_bounds_um"][2][0],z)
        recipe["cell_bounds_um"][2][1] = max(recipe["cell_bounds_um"][2][1],z+height+.1)
        if index is None:
            index = len(recipe["steps"])
            recipe["steps"].append(step)
        else:
            recipe["steps"][index] = step
        build_geometry(recipe,base_dir=self.base_dir)
        self._install_pattern(candidate,index,build=build)
        self.log("已定义 Mark："+result["summary"])
        return index

    def apply_gds_definition(self, step, *, target_index=None, cell_bounds_um=None, update_cell=False, build=True):
        from .modeling import build_geometry
        self._require_process()
        if self.busy:
            raise ValueError("请等待当前任务完成")
        candidate = self._pattern_candidate()
        recipe = candidate["process"]
        if cell_bounds_um is not None:
            bounds = [[float(v) for v in pair] for pair in cell_bounds_um]
            if len(bounds)!=3 or any(len(pair)!=2 or not all(map(math.isfinite,pair)) or pair[0]>=pair[1] for pair in bounds):
                raise ValueError("计算单元须包含三个有限且递增的区间")
            changed = any(not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for p,q in zip(bounds,recipe["cell_bounds_um"]) for a,b in zip(p,q))
            if changed and not update_cell:
                raise ValueError("单元范围已更改；勾选‘同步计算单元’后应用，或使用当前单元。")
            if update_cell:
                recipe["cell_bounds_um"] = bounds
                recipe["period_x_um"] = bounds[0][1]-bounds[0][0]
        index = self._pattern_target(target_index)
        selected = copy.deepcopy(step)
        materials = candidate["materials"]["materials"]
        used = selected.get("target_materials",[]) if selected["operation"] in ("etch","delete") else [selected["material"]]
        if any(name not in materials for name in used):
            raise ValueError("请先添加选中的材料或刻蚀目标材料")
        if index is None:
            index = len(recipe["steps"])
            recipe["steps"].append(selected)
        else:
            selected["id"] = recipe["steps"][index]["id"]
            recipe["steps"][index] = selected
        low = selected.get("z_min_um",selected.get("start_z_um",0)-selected.get("etch_depth_um",0))
        high = selected.get("z_max_um",selected.get("z_min_um",0)+selected.get("thickness_um",0))
        recipe["cell_bounds_um"][2][0] = min(recipe["cell_bounds_um"][2][0],low)
        recipe["cell_bounds_um"][2][1] = max(recipe["cell_bounds_um"][2][1],high+.1)
        build_geometry(recipe,base_dir=self.base_dir)
        self._install_pattern(candidate,index,build=build)
        self.log(f"已导入 GDS：{selected['mask']['gds_path']} · {selected['mask']['cell']} · Layer {selected['mask']['layer']}/{selected['mask']['datatype']}")
        return index

    def define_mark_pattern(self):
        self._require_process()
        dialog = MarkPatternDialog(self)
        self._active_pattern_dialog = dialog
        return dialog.show()

    def import_gds_pattern(self):
        from tkinter import filedialog
        self._require_process()
        selected = filedialog.askopenfilename(parent=self.root,title="导入 GDS 图案",filetypes=[("GDS 版图","*.gds *.gdsii"),("全部文件","*.*")])
        if not selected:
            return
        dialog = GdsImportDialog(self,selected)
        self._active_pattern_dialog = dialog
        return dialog.show()


class _PatternDialog:
    def __init__(self,app,title):
        self.app,self.tk,self.ttk = app,app.tk,app.ttk
        self.top = self.tk.Toplevel(app.root)
        self.top.title(title)
        self.top.transient(app.root)
        self.top.geometry(f"{min(1150,app.root.winfo_screenwidth()-90)}x{min(810,app.root.winfo_screenheight()-100)}")
        self.top.minsize(930,650)
        self.accepted = False
        self.top.protocol("WM_DELETE_WINDOW",self.cancel)
        outer = self.ttk.Frame(self.top,padding=16)
        outer.pack(fill="both",expand=True)
        outer.columnconfigure(0,weight=1)
        outer.rowconfigure(1,weight=1,minsize=120)
        self.ttk.Label(outer,text=title,font=("Microsoft YaHei UI",14,"bold")).grid(row=0,column=0,sticky="w",pady=(0,10))
        self.body = self.ttk.Frame(outer)
        self.body.grid(row=1,column=0,sticky="nsew")
        self.body.columnconfigure(0,weight=1)
        self.body.columnconfigure(1,weight=1)
        self.body.rowconfigure(0,weight=1)
        left_parent = self.ttk.Frame(self.body)
        left_parent.grid(row=0,column=0,sticky="nsew")
        self.left = app._scroll_panel(left_parent)
        self.right = self.ttk.Frame(self.body,padding=(14,0,0,0))
        self.right.grid(row=0,column=1,sticky="nsew")
        self.right.columnconfigure(0,weight=1)
        self.right.rowconfigure(0,weight=1,minsize=60)
        self.canvas = self.tk.Canvas(self.right,background="white",height=290,highlightthickness=1,highlightbackground="#d6d6d6")
        self.canvas.grid(row=0,column=0,sticky="nsew")
        self._redraw_timer = None
        self.canvas.bind("<Configure>",self._resize_preview)
        self.summary = self.tk.StringVar(value="")
        summary_label = self.ttk.Label(self.right,textvariable=self.summary,wraplength=490,justify="left")
        summary_label.grid(row=1,column=0,sticky="ew",pady=12)
        self.right.bind("<Configure>",lambda event:summary_label.configure(wraplength=max(180,event.width-25)))
        self.options = self.ttk.Frame(outer)
        self.options.grid(row=2,column=0,sticky="ew",pady=6)
        self.message = self.tk.StringVar()
        self.ttk.Label(outer,textvariable=self.message,foreground="#da1e28",wraplength=860).grid(row=3,column=0,sticky="ew",pady=6)
        bottom = self.ttk.Frame(outer)
        bottom.grid(row=4,column=0,sticky="ew")
        self.ttk.Button(bottom,text="取消",command=self.cancel).pack(side="right",padx=5)
        self.ttk.Button(bottom,text="应用并生成几何",command=self.apply,style="Primary.TButton").pack(side="right",padx=5)
        self.ttk.Button(bottom,text="刷新图案预览",command=self.preview).pack(side="left")
        self.target = self.tk.StringVar()
        self.targets = {"新增图案步骤":-1}
        for i,step in enumerate(app.project["process"]["steps"]):
            if step["operation"]!="create_substrate":
                self.targets[f"替换 {i+1:02d} · {step.get('name',step['id'])}"] = i
        index = app._pattern_target()
        self.target.set(next((key for key,value in self.targets.items() if value==index),"新增图案步骤"))
        self.update_cell = self.tk.BooleanVar(value=False)

    def _resize_preview(self,_event=None):
        if self._redraw_timer is not None:
            self.top.after_cancel(self._redraw_timer)
        def redraw():
            self._redraw_timer = None
            if self.top.winfo_exists():
                self.preview()
        self._redraw_timer = self.top.after(70,redraw)

    def target_controls(self,parent):
        frame = self.ttk.LabelFrame(self.options,text="应用到工程",padding=(10,6))
        frame.pack(fill="x")
        row = self.ttk.Frame(frame)
        row.pack(fill="x")
        self.ttk.Combobox(row,textvariable=self.target,values=list(self.targets),state="readonly",width=36).pack(side="left")
        self.ttk.Checkbutton(row,text="同步计算单元（周期或边界变化时勾选）",variable=self.update_cell).pack(side="left",padx=12)
        self.ttk.Label(frame,text="替换只修改目标步骤；其它膜层和工艺保留。改变单元不会缩放其它图案。",wraplength=1000,foreground="#6f6f6f").pack(anchor="w",pady=(4,0))

    def draw(self,polygons,bounds):
        self.top.update_idletasks()
        canvas = self.canvas
        canvas.delete("all")
        width,height = max(100,canvas.winfo_width()),max(80,canvas.winfo_height())
        x0,x1 = bounds[0]
        y0,y1 = bounds[1]
        scale = min((width-70)/max(x1-x0,1e-9),(height-70)/max(y1-y0,1e-9))
        ox,oy = width/2,height/2
        def point(x,y):
            return ox+(x-(x0+x1)/2)*scale,oy-(y-(y0+y1)/2)*scale
        for polygon in polygons[:600]:
            coords = [v for x,y in polygon for v in point(x,y)]
            if len(coords)>=6:
                canvas.create_polygon(coords,fill="#0f62fe",outline="#0043ce")
        a,b = point(x0,y0),point(x1,y1)
        canvas.create_rectangle(a[0],b[1],b[0],a[1],outline="#161616",dash=(5,3),width=2)
        canvas.create_text(width/2,height-14,text=f"x: {x0:g} → {x1:g} µm",fill="#525252")
        canvas.create_text(width/2,14,text="图案俯视图 / 选中计算区域",fill="#525252")

    def show(self):
        self.top.grab_set()
        self.app.root.wait_window(self.top)
        return self.accepted

    def cancel(self):
        if self._redraw_timer is not None:
            self.top.after_cancel(self._redraw_timer)
            self._redraw_timer = None
        self.top.destroy()


class MarkPatternDialog(_PatternDialog):
    kinds = {"普通周期光栅":"grating","AH32（2 条 / 3 阶）":"AH32","AH53（3 条 / 5 阶）":"AH53","AH74（4 条 / 7 阶）":"AH74","自定义多条纹":"custom"}

    def __init__(self,app):
        from .modeling.patterns import pattern_defaults
        super().__init__(app,"定义 Mark 图案")
        self.kind = self.tk.StringVar(value="普通周期光栅")
        form = self.ttk.LabelFrame(self.left,text="Mark 尺寸",padding=10)
        form.pack(fill="x")
        self.ttk.Label(form,text="图案类型").grid(row=0,column=0,sticky="w",pady=6)
        combo = self.ttk.Combobox(form,textvariable=self.kind,values=list(self.kinds),state="readonly")
        combo.grid(row=0,column=1,sticky="ew")
        combo.bind("<<ComboboxSelected>>",lambda _e:self.load_kind())
        self.fields = {}
        fields = (("period_um","母周期 · µm",8),("cd_nm","条纹 CD · nm",4000),("height_nm","高度 · nm",120),
                  ("center_um","整体横移 · µm",0),("material","图案材料","ridge_demo"))
        for row,(key,label,value) in enumerate(fields,start=1):
            values = list(app.project["materials"]["materials"]) if key=="material" else None
            self.fields[key] = app._field(form,row,label,value,values=values)
        self.lines = []
        custom = self.ttk.LabelFrame(self.left,text="条纹列表（自定义模式可增删）",padding=10)
        custom.pack(fill="x",pady=10)
        self.tree = self.ttk.Treeview(custom,columns=("center","width"),show="headings",height=4)
        self.tree.heading("center",text="中心 x · µm")
        self.tree.heading("width",text="宽度 · nm")
        self.tree.column("center",width=140)
        self.tree.column("width",width=140)
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>",self.select_line)
        self.line_center = self.tk.StringVar(value="0")
        self.line_width = self.tk.StringVar(value="500")
        row = self.ttk.Frame(custom)
        row.pack(fill="x",pady=7)
        self.ttk.Label(row,text="中心 µm").pack(side="left")
        self.ttk.Entry(row,textvariable=self.line_center,width=9).pack(side="left",padx=4)
        self.ttk.Label(row,text="宽度 nm").pack(side="left")
        self.ttk.Entry(row,textvariable=self.line_width,width=9).pack(side="left",padx=4)
        row = self.ttk.Frame(custom)
        row.pack(fill="x")
        for text,fn in (("新增条纹",self.add_line),("修改选中",self.update_line),("删除选中",self.delete_line)):
            self.ttk.Button(row,text=text,command=fn).pack(side="left",padx=2)
        self.target_controls(self.left)
        index = app._pattern_target()
        step = app.project["process"]["steps"][index] if index is not None else {}
        saved = step.get("mark_pattern")
        if saved:
            self.set_spec(saved)
        else:
            spec = pattern_defaults("grating")
            spec.update(period_um=app.project["process"]["period_x_um"],cd_um=app.project["process"]["period_x_um"]*.5,
                        height_um=step.get("z_max_um",step.get("z_min_um",0)+step.get("thickness_um",.12))-step.get("z_min_um",0),
                        material=step.get("material",next((m for m in app.project["materials"]["materials"] if m!=app.project["process"]["background_material"]))))
            self.set_spec(spec)
        self.top.after_idle(self.preview)

    def set_spec(self,spec):
        self.kind.set(next(label for label,key in self.kinds.items() if key==spec["kind"]))
        for key,value in (("period_um",spec["period_um"]),("cd_nm",spec["cd_um"]*1000),("height_nm",spec["height_um"]*1000),("center_um",spec.get("center_um",0)),("material",spec["material"])):
            self.fields[key].set(str(value))
        self.lines = copy.deepcopy(spec.get("lines",[{"center_um":0.,"width_um":spec["cd_um"]}]))
        self.refresh_lines()

    def load_kind(self):
        from .modeling.patterns import pattern_defaults
        key = self.kinds[self.kind.get()]
        spec = pattern_defaults(key)
        materials = self.app.project["materials"]["materials"]
        spec["material"] = self.fields["material"].get() if self.fields["material"].get() in materials else next(iter(materials))
        self.set_spec(spec)
        self.preview()

    def spec(self):
        result = {"kind":self.kinds[self.kind.get()],"period_um":float(self.fields["period_um"].get()),
                  "cd_um":float(self.fields["cd_nm"].get())/1000,"height_um":float(self.fields["height_nm"].get())/1000,
                  "center_um":float(self.fields["center_um"].get()),"material":self.fields["material"].get()}
        if result["kind"]=="custom":
            result["lines"] = copy.deepcopy(self.lines)
        return result

    def refresh_lines(self):
        self.tree.delete(*self.tree.get_children())
        for i,line in enumerate(self.lines):
            self.tree.insert("","end",iid=str(i),values=(f"{line['center_um']:g}",f"{line['width_um']*1000:g}"))

    def select_line(self,_event=None):
        selected = self.tree.selection()
        if selected:
            line = self.lines[int(selected[0])]
            self.line_center.set(str(line["center_um"]))
            self.line_width.set(str(line["width_um"]*1000))

    def custom_mode(self):
        if self.kinds[self.kind.get()]!="custom":
            from .modeling.patterns import make_pattern
            result = make_pattern(self.spec())
            shift = result["spec"]["center_um"]
            self.lines = [{"center_um":l["center_um"]-shift,"width_um":l["width_um"]} for l in result["lines"]]
            self.kind.set("自定义多条纹")

    def add_line(self):
        try:
            self.custom_mode()
            self.lines.append({"center_um":float(self.line_center.get()),"width_um":float(self.line_width.get())/1000})
            self.refresh_lines()
            self.preview()
        except Exception as exc:
            self.message.set(str(exc))

    def update_line(self):
        selected = self.tree.selection()
        if not selected:
            return
        try:
            index = int(selected[0])
            self.custom_mode()
            self.lines[index] = {"center_um":float(self.line_center.get()),"width_um":float(self.line_width.get())/1000}
            self.refresh_lines()
            self.preview()
        except Exception as exc:
            self.message.set(str(exc))

    def delete_line(self):
        selected = self.tree.selection()
        if selected:
            self.custom_mode()
            self.lines.pop(int(selected[0]))
            self.refresh_lines()
            self.preview()

    def preview(self):
        from .modeling.patterns import make_pattern
        try:
            result = make_pattern(self.spec(),y_bounds=self.app.project["process"]["cell_bounds_um"][1])
            if result["spec"]["kind"]!="custom":
                self.lines = [{"center_um":line["center_um"]-result["spec"]["center_um"],"width_um":line["width_um"]} for line in result["lines"]]
                self.refresh_lines()
            p = result["period_um"]
            self.draw(result["polygons_xy_um"],[[-p/2,p/2],self.app.project["process"]["cell_bounds_um"][1]])
            self.summary.set(result["summary"]+"\n蓝色为实际条纹，虚线为完整母周期。AH 模板的内部子间距由母周期派生。")
            self.message.set("")
            return True
        except Exception as exc:
            self.message.set(str(exc))
            return False

    def apply(self):
        try:
            self.app.apply_mark_definition(self.spec(),target_index=self.targets[self.target.get()],update_cell=self.update_cell.get())
            self.accepted = True
            self.cancel()
        except Exception as exc:
            self.message.set(str(exc))


class GdsImportDialog(_PatternDialog):
    def __init__(self,app,path):
        from .modeling.gds_import import inspect_layout
        info = inspect_layout(path)
        if not any(cell["layers"] for cell in info["cells"]):
            raise ValueError("GDS 中没有可选择的图案")
        self.info,self.path = info,Path(path).resolve()
        super().__init__(app,"导入 GDS 图案")
        self.ttk.Label(self.left,text=self.path.name+f"\n文件单位：{info['unit_m']:g} m，已换算为 µm",wraplength=430).pack(anchor="w",pady=5)
        form = self.ttk.LabelFrame(self.left,text="选择版图",padding=10)
        form.pack(fill="x")
        self.cell = self.tk.StringVar()
        self.layer = self.tk.StringVar()
        self.fields = {}
        self.ttk.Label(form,text="Cell").grid(row=0,column=0,sticky="w",pady=5)
        self.cells = {c["name"]:c for c in info["cells"]}
        available = [name for name,c in self.cells.items() if c["layers"]]
        self.cell_combo = self.ttk.Combobox(form,textvariable=self.cell,values=available,state="readonly")
        self.cell_combo.grid(row=0,column=1,sticky="ew")
        self.cell_combo.bind("<<ComboboxSelected>>",lambda _e:self.cell_changed())
        self.ttk.Label(form,text="Layer / Datatype").grid(row=1,column=0,sticky="w",pady=5)
        self.layer_combo = self.ttk.Combobox(form,textvariable=self.layer,state="readonly")
        self.layer_combo.grid(row=1,column=1,sticky="ew")
        self.layer_combo.bind("<<ComboboxSelected>>",lambda _e:self.preview())
        process = app.project["process"]
        index = app._pattern_target()
        step = process["steps"][index] if index is not None else {}
        material = step.get("material",next((m for m in app.project["materials"]["materials"] if m!=process["background_material"])))
        for row,(key,label,value,values) in enumerate((("material","材料 / 刻蚀目标",material,list(app.project["materials"]["materials"])),
                 ("operation","应用方式","添加材料",("添加材料","刻蚀指定材料")),("z_nm","底面 / 刻蚀起点 · nm",step.get("z_min_um",0)*1000,None),
                 ("height_nm","厚度 / 刻蚀深度 · nm",(step.get("z_max_um",step.get("z_min_um",0)+step.get("thickness_um",.12))-step.get("z_min_um",0))*1000,None)),start=2):
            self.fields[key] = app._field(form,row,label,value,values=values)
        region = self.ttk.LabelFrame(self.left,text="周期单元与版图选区",padding=10)
        region.pack(fill="x",pady=10)
        bounds = process["cell_bounds_um"]
        values = (("period_um","真实母周期 · µm",process["period_x_um"]),("x_min","单元 x 起点 · µm",bounds[0][0]),
                  ("y_min","单元 y 起点 · µm",bounds[1][0]),("y_max","单元 y 终点 · µm",bounds[1][1]),
                  ("dx","导入图形横移 · µm",0),("dy","导入图形纵移 · µm",0))
        for row,(key,label,value) in enumerate(values):
            self.fields[key] = app._field(region,row,label,value)
        self.ttk.Button(region,text="将图形中心移到当前单元中心",command=self.center_layout).grid(row=len(values),column=0,columnspan=2,sticky="ew",pady=8)
        self.ttk.Label(region,text="包围盒宽度不一定是母周期，请按设计值确认。只读取所选单元对应的版图区域，不自动提取任意截面。",wraplength=430,foreground="#6f6f6f").grid(row=len(values)+1,column=0,columnspan=2,sticky="w")
        self.target_controls(self.left)
        selected = next((name for name in info["top_cells"] if name in available),available[0])
        self.cell.set(selected)
        self.cell_changed()
        self.top.after_idle(self.preview)

    def cell_changed(self):
        self.layers = {f"{item['layer']} / {item['datatype']}":item for item in self.cells[self.cell.get()]["layers"]}
        self.layer_combo.configure(values=list(self.layers))
        self.layer.set(next(iter(self.layers)))
        self.preview()

    def region(self):
        p,x0,y0,y1,dx,dy = [float(self.fields[k].get()) for k in ("period_um","x_min","y_min","y_max","dx","dy")]
        if not all(map(math.isfinite,(p,x0,y0,y1,dx,dy))) or p<=0 or y0>=y1:
            raise ValueError("周期须为正，y 边界须递增，所有坐标须为有限数值")
        roi = [[x0-dx,x0+p-dx],[y0-dy,y1-dy]]
        bounds = [[x0,x0+p],[y0,y1],copy.deepcopy(self.app.project["process"]["cell_bounds_um"][2])]
        return roi,bounds,dx,dy

    def center_layout(self):
        item = self.layers[self.layer.get()]
        _,bounds,_,_ = self.region()
        self.fields["dx"].set(str(sum(bounds[0])/2-sum(item["bounds_um"][0])/2))
        self.fields["dy"].set(str(sum(bounds[1])/2-sum(item["bounds_um"][1])/2))
        self.preview()

    def preview(self):
        from .modeling.geometry import read_gds_polygons
        try:
            item = self.layers[self.layer.get()]
            roi,bounds,dx,dy = self.region()
            polygons,_ = read_gds_polygons(self.path,cell=self.cell.get(),layer=item["layer"],datatype=item["datatype"],roi_um=roi)
            points = [[[float(x)+dx,float(y)+dy] for x,y in polygon.points] for polygon in polygons]
            self.draw(points,bounds[:2])
            bbox = item["bounds_um"]
            self.summary.set(f"Cell：{self.cell.get()}\nLayer / Datatype：{self.layer.get()}\n该层总图形数：{item['polygon_count']}，选区内：{len(points)}\n原始边界：x {bbox[0][0]:g} → {bbox[0][1]:g} µm，y {bbox[1][0]:g} → {bbox[1][1]:g} µm\n\n蓝色为所选单元内的真实图案。一维 RCWA 要求选区内结构沿 y 不变；二维图案可查看，但会在求解时拒绝。")
            self.message.set("选区中没有图案；请调整单元范围或移动图形中心。" if not points else "")
            return bool(points)
        except Exception as exc:
            self.message.set(str(exc))
            return False

    def apply(self):
        from .modeling.gds_import import make_gds_step
        try:
            item = self.layers[self.layer.get()]
            roi,bounds,dx,dy = self.region()
            material = self.fields["material"].get()
            operation = "etch" if self.fields["operation"].get()=="刻蚀指定材料" else "add"
            step = make_gds_step(self.path,cell=self.cell.get(),layer=item["layer"],datatype=item["datatype"],material=material,
                z_min_um=float(self.fields["z_nm"].get())/1000,thickness_um=float(self.fields["height_nm"].get())/1000,
                roi_um=roi,delta_x_um=dx,delta_y_um=dy,operation=operation,target_materials=[material] if operation=="etch" else None)
            self.app.apply_gds_definition(step,target_index=self.targets[self.target.get()],cell_bounds_um=bounds,update_cell=self.update_cell.get())
            self.accepted = True
            self.cancel()
        except Exception as exc:
            self.message.set(str(exc))
