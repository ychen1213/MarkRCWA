"""Offline Windows desktop for materials, process geometry and RCWA results."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import queue
import threading
import time
import uuid
import webbrowser
from pathlib import Path

from .desktop_patterns import PatternTools


def user_directory():
    import tempfile
    custom = os.environ.get("MARKRCWA_USER_DATA")
    root = Path(custom) if custom else Path(os.environ.get("LOCALAPPDATA", Path.home())) / "MarkRCWA"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        root = Path(tempfile.gettempdir()) / "MarkRCWA"
        root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _configure_runtime():
    # Executables may live in a read-only directory; caches belong to the user.
    root = user_directory()
    os.environ.setdefault("MPLCONFIGDIR", str(root / "cache" / "matplotlib"))
    os.environ.setdefault("PLOTLY_RENDERER", "browser")
    import matplotlib
    matplotlib.use("Agg")
    if os.name == "nt":
        import ctypes
        # Source launches and the frozen manifest should use matching pixels.
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError, OSError):
            pass


class DesktopApp(PatternTools):
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        from .desktop_project import default_project
        self.tk, self.ttk, self.root = tk, ttk, root
        self.project = default_project()
        self.workspace = user_directory() / "workspaces" / uuid.uuid4().hex[:10]
        self.workspace.mkdir(parents=True)
        self.base_dir, self.project_file = self.workspace, None
        self.messages = queue.Queue()
        self.cancel_event = threading.Event()
        self.busy, self.pending_close, self.loading = False, False, True
        self.dirty, self.step_dirty, self.step_index = False, False, None
        self.preview_dir, self.result, self.last_error = None, None, None
        self.result_stale = False
        self.photo = None
        self._buttons = []
        self.root.title("MarkRCWA · GDS 导入 / Mark 图案 / 一维周期仿真")
        width = min(1400,max(1000,root.winfo_screenwidth()-100))
        height = min(900,max(680,root.winfo_screenheight()-100))
        self.root.geometry(f"{width}x{height}+30+30")
        self.root.minsize(min(1080,width),min(720,height))
        self.status = tk.StringVar(value="就绪 · 合成示例可直接生成几何并运行；实际分析请导入真实材料。")
        self.project_caption = tk.StringVar()
        self._style()
        self._build()
        self._load_forms()
        self.tabs.select(self.pages["process"])
        self.loading = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Control-s>", lambda _e:self.safe(self.save))
        self.root.bind("<Control-o>", lambda _e:self.safe(self.open_project))
        self.root.bind("<F5>", lambda _e:self.safe(self.build_preview))
        self.root.bind("<Escape>", lambda _e:self.cancel())
        self.root.after(100, self._poll)

    def _style(self):
        style = self.ttk.Style(self.root)
        style.theme_use("clam")
        self.root.configure(background="#f4f4f4")
        self.root.option_add("*Font", "{Microsoft YaHei UI} 10")
        style.configure("TFrame", background="#f4f4f4")
        style.configure("TLabel", background="#f4f4f4", foreground="#161616")
        style.configure("TLabelframe", background="#f4f4f4", bordercolor="#d6d6d6")
        style.configure("TLabelframe.Label", background="#f4f4f4", foreground="#525252")
        style.configure("TButton", padding=(12,7), borderwidth=0)
        style.configure("Primary.TButton", background="#0f62fe", foreground="white")
        style.map("Primary.TButton", background=[("active","#0043ce"),("disabled","#c6c6c6")])
        style.configure("Treeview", rowheight=29, background="white", fieldbackground="white", borderwidth=0)
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI",10,"bold"), background="#e0e0e0")
        style.configure("TNotebook.Tab", padding=(20,10))
        style.map("TNotebook.Tab", background=[("selected","white")], foreground=[("selected","#0f62fe")])

    def _button(self, parent, text, command, primary=False):
        button = self.ttk.Button(parent, text=text, command=lambda:self.safe(command),
                                 style="Primary.TButton" if primary else "TButton")
        button.pack(side="left", padx=(0,7), pady=3)
        self._buttons.append(button)
        return button

    def _build(self):
        tk, ttk = self.tk, self.ttk
        header = tk.Frame(self.root, background="#161616", height=62)
        header.pack(fill="x")
        tk.Label(header,text="MARK / RCWA",background="#161616",foreground="white",font=("Segoe UI",20,"bold"),padx=22,pady=13).pack(side="left")
        tk.Label(header,text="一维周期 · Windows 本地计算",background="#161616",foreground="#c6c6c6").pack(side="left",padx=14)
        tk.Label(header,textvariable=self.project_caption,background="#161616",foreground="white").pack(side="right",padx=22)
        bar = ttk.Frame(self.root,padding=(16,7))
        bar.pack(fill="x")
        for text, fn in (("新建",self.new_project),("打开",self.open_project),("保存",self.save),("另存为",lambda:self.save(save_as=True)),
                         ("导入 MarkSim",self.import_marksim),("导入 PH 截面",self.import_ph),("使用说明",self.help)):
            self._button(bar,text,fn)
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both",expand=True,padx=16,pady=(0,8))
        self.pages = {}
        for key,title in (("materials","材料"),("process","工艺与结构"),("simulation","RCWA 设置"),("results","仿真结果"),("logs","任务日志")):
            page = ttk.Frame(self.tabs,padding=14)
            self.tabs.add(page,text=title)
            self.pages[key] = page
        self._materials_page()
        self._process_page()
        self._simulation_page()
        self._results_page()
        self.log_text = tk.Text(self.pages["logs"],wrap="word",font=("Consolas",10),background="#161616",foreground="#e0e0e0",state="disabled")
        self.log_text.pack(fill="both",expand=True)
        bottom = ttk.Frame(self.root,padding=(16,5))
        bottom.pack(side="bottom",fill="x",before=self.tabs)
        self.progress = ttk.Progressbar(bottom,length=180,maximum=1)
        self.progress.pack(side="right",padx=8)
        self.cancel_button = ttk.Button(bottom,text="取消任务",command=self.cancel,state="disabled")
        self.cancel_button.pack(side="right")
        ttk.Label(bottom,textvariable=self.status,wraplength=950).pack(side="left",fill="x",expand=True)

    def _field(self, parent, row, label, value="", *, values=None, changed=None, width=23):
        ttk,tk = self.ttk,self.tk
        ttk.Label(parent,text=label).grid(row=row,column=0,sticky="w",padx=(0,12),pady=5)
        var = tk.StringVar(value=str(value))
        entry = ttk.Combobox(parent,textvariable=var,values=values,width=width,state="readonly") if values is not None else ttk.Entry(parent,textvariable=var,width=width)
        entry.grid(row=row,column=1,sticky="ew",pady=5)
        parent.columnconfigure(1,weight=1)
        if changed:
            var.trace_add("write",lambda *_:changed())
        return var

    def _scroll_panel(self,parent):
        canvas = self.tk.Canvas(parent,background="#f4f4f4",highlightthickness=0,width=450)
        scroll = self.ttk.Scrollbar(parent,orient="vertical",command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right",fill="y")
        canvas.pack(side="left",fill="both",expand=True)
        body = self.ttk.Frame(canvas,padding=(0,0,12,10))
        window = canvas.create_window((0,0),window=body,anchor="nw")
        body.bind("<Configure>",lambda _e:canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",lambda e:canvas.itemconfigure(window,width=e.width))
        def wheel(event):
            canvas.yview_scroll(int(-event.delta/120),"units")
        canvas.bind("<MouseWheel>",wheel)
        body.bind("<MouseWheel>",wheel)
        return body

    def _materials_page(self):
        ttk,tk,page = self.ttk,self.tk,self.pages["materials"]
        row = ttk.Frame(page)
        row.grid(row=0,column=0,sticky="ew")
        self._button(row,"导入 TXT / CSV",lambda:self.import_materials(False),True)
        self._button(row,"导入 TXT 文件夹",lambda:self.import_materials(True))
        self._button(row,"添加常数材料",self.add_material)
        self._button(row,"删除选中材料",self.delete_material)
        settings = ttk.Frame(page)
        settings.grid(row=1,column=0,sticky="ew",pady=8)
        ttk.Label(settings,text="导入波长单位").pack(side="left",padx=(0,8))
        self.material_unit = tk.StringVar(value="nm")
        ttk.Combobox(settings,textvariable=self.material_unit,values=("nm","um","m"),state="readonly",width=7).pack(side="left")
        ttk.Label(settings,text="列号 λ,n,k（从 1 开始；无 k 填两列）").pack(side="left",padx=(16,8))
        self.material_columns = tk.StringVar(value="1,2,3")
        ttk.Entry(settings,textvariable=self.material_columns,width=10).pack(side="left")
        self.assume_lossless = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings,text="缺少 k 时明确采用无损近似",variable=self.assume_lossless).pack(side="left",padx=15)
        self.material_tree = ttk.Treeview(page,columns=("range","loss","source"),show="tree headings",height=7)
        for key,title,width in (("#0","材料",180),("range","有效波段 / nm",180),("loss","吸收数据",160),("source","数据来源",520)):
            self.material_tree.heading(key,text=title)
            self.material_tree.column(key,width=width,minwidth=90)
        self.material_tree.grid(row=2,column=0,sticky="ew",pady=8)
        self.material_tree.bind("<<TreeviewSelect>>",lambda _e:self.safe(self.plot_material))
        self.material_plot = ttk.Frame(page)
        self.material_plot.grid(row=3,column=0,sticky="nsew")
        page.rowconfigure(3,weight=1,minsize=150)
        page.columnconfigure(0,weight=1)
        self.material_canvas = None
        ttk.Label(page,text="示例材料是假设常数。导入数据按 n、k 分别线性插值；超出有效波段会停止计算。",foreground="#6f6f6f").grid(row=4,column=0,sticky="w",pady=5)

    def _process_page(self):
        ttk,tk,page = self.ttk,self.tk,self.pages["process"]
        create = ttk.Frame(page)
        create.pack(fill="x",pady=(0,5))
        self._button(create,"导入 GDS",self.import_gds_pattern,True)
        self._button(create,"定义 Mark 图案",self.define_mark_pattern,True)
        ttk.Label(create,text="选择版图图层，或定义光栅 / AH32 / AH53 / AH74 / 自定义多条纹",foreground="#525252").pack(side="left",padx=12)
        toolbar = ttk.Frame(page)
        toolbar.pack(fill="x")
        for text,fn,primary in (("生成几何 / 验证",self.build_preview,True),("旋转 3D 视图",self.open_3d,False),
                                ("计算单元",self.edit_cell,False),("高级工艺参数",self.edit_process_json,False)):
            self._button(toolbar,text,fn,primary)
        self.model_note = tk.StringVar()
        ttk.Label(page,textvariable=self.model_note,foreground="#525252",wraplength=1200).pack(anchor="w",pady=(8,10))
        panes = ttk.Panedwindow(page,orient="horizontal")
        panes.pack(fill="both",expand=True)
        left,right = ttk.Frame(panes),ttk.Frame(panes,padding=(14,0,0,0))
        panes.add(left,weight=2)
        panes.add(right,weight=3)
        self.process_body = body = self._scroll_panel(left)
        buttons = ttk.Frame(body)
        buttons.pack(fill="x")
        for text,fn in (("新增",self.add_step),("删除",self.delete_step),("上移",lambda:self.move_step(-1)),("下移",lambda:self.move_step(1))):
            button = self._button(buttons,text,fn)
            button.configure(width=5,padding=(4,5))
        self.step_tree = ttk.Treeview(body,columns=("op","mat"),show="tree headings",height=5)
        for key,title,width in (("#0","工艺步骤",175),("op","操作",100),("mat","材料",140)):
            self.step_tree.heading(key,text=title)
            self.step_tree.column(key,width=width,minwidth=70)
        self.step_tree.pack(fill="x",pady=(5,12))
        self.step_tree.bind("<<TreeviewSelect>>",lambda _e:self.safe(self.select_step))
        self.operations = {"创建衬底":"create_substrate","表面薄膜":"create_film","按掩模添加":"add","垂直刻蚀":"etch","删除指定材料":"delete"}
        self.film_modes = {"等厚包覆":"conformal","填平覆盖":"planar_fill","指定高度":"absolute"}
        form = ttk.LabelFrame(body,text="选中步骤",padding=10)
        form.pack(fill="x")
        self.step_vars = {}
        fields = [("name","步骤名称","",None),("operation","几何操作","按掩模添加",tuple(self.operations)),
                  ("material","材料 / 刻蚀目标（逗号分隔）","",None),("film_mode","薄膜形貌","等厚包覆",tuple(self.film_modes)),
                  ("z","底面 / 刻蚀起点 · nm",0,None),("height","厚度 · nm",100,None),
                  ("sidewall","侧壁包覆厚度 · nm",100,None),("depth","刻蚀深度 · nm",100,None),
                  ("cd","CD 总宽度偏置 · nm",0,None),("dx","横向偏移 · nm",0,None),
                  ("mask_mode","掩模","保留原掩模",("保留原掩模","整个单元","矩形条纹","GDS 图层")),
                  ("left","矩形左边界 · µm",-2,None),("right","矩形右边界 · µm",2,None),
                  ("gds","GDS 文件","",None),("cell","GDS Cell","",None),
                  ("layer","GDS Layer",1,None),("datatype","GDS Datatype",0,None)]
        for row,(key,label,val,values) in enumerate(fields):
            self.step_vars[key] = self._field(form,row,label,val,values=values,changed=self._step_changed)
        self.step_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(form,text="启用此步骤",variable=self.step_enabled,command=self._step_changed).grid(row=len(fields),column=0,sticky="w",pady=7)
        ttk.Button(form,text="选择 / 检查 GDS",command=lambda:self.safe(self.choose_gds)).grid(row=len(fields),column=1,sticky="e")
        self._button(body,"应用步骤",self.apply_step,True)
        ttk.Label(body,text="工艺采用几何规则，侧壁为 90°。包覆和填平沿用 MarkSim 的定义。",wraplength=450,foreground="#6f6f6f").pack(anchor="w",pady=7)
        self.preview_note = tk.StringVar(value="点击生成几何，查看实际结构与 RCWA 层栈。")
        ttk.Label(right,textvariable=self.preview_note,wraplength=650).pack(anchor="w",pady=(0,10))
        self.preview_image = ttk.Label(right,anchor="center")
        self.preview_image.pack(fill="both",expand=True)
        self.preview_image.bind("<Configure>",lambda _e:self._resize_preview())
        self.layer_tree = ttk.Treeview(right,columns=("z","materials"),show="headings",height=6)
        self.layer_tree.heading("z",text="RCWA 深度 / 厚度 · µm")
        self.layer_tree.heading("materials",text="各横向区间的材料")
        self.layer_tree.column("z",width=230)
        self.layer_tree.column("materials",width=400)
        self.layer_tree.pack(fill="x",pady=(10,0))

    def _simulation_page(self):
        ttk,tk,page = self.ttk,self.tk,self.pages["simulation"]
        page = self._scroll_panel(page)
        ttk.Label(page,text="平面波照明 · x 周期、y 方向结构不变 · 入射方向从模型上方指向衬底",font=("Microsoft YaHei UI",13,"bold")).pack(anchor="w",pady=(0,16))
        form = ttk.LabelFrame(page,text="入射与计算精度",padding=16)
        form.pack(anchor="nw",fill="x")
        self.sim_vars = {}
        for row,(key,label,val,values) in enumerate((("wavelength_nm","真空波长 · nm",633,None),("theta_deg","入射角 · °",0,None),
                                                    ("polarization","偏振","TE + TM",("TE + TM","TE","TM")),
                                                    ("fourier_order","傅里叶半阶数 N（共 2N+1 阶）",31,None),
                                                    ("convergence_max_order","自动收敛的最大半阶数 N",127,None),
                                                    ("reference_z_um","相位参考面 z · µm（留空取单元上边界）","",None))):
            self.sim_vars[key] = self._field(form,row,label,val,values=values,changed=self.changed)
        ttk.Label(form,text="N 为求解基底半阶数。入射/反射相位使用固定物理 z 平面；透射平面为输出端界面，位置随结果保存。",foreground="#6f6f6f").grid(row=6,column=0,columnspan=2,sticky="w",pady=10)
        sweep = ttk.LabelFrame(page,text="参数扫描",padding=16)
        sweep.pack(fill="x",pady=14)
        self.sweep_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(sweep,text="启用扫描",variable=self.sweep_enabled,command=self.changed).grid(row=0,column=0,sticky="w")
        self.sweep_vars = {}
        self.sweep_vars["parameter"] = self._field(sweep,1,"扫描参数","波长 / nm",values=("波长 / nm","入射角 / °","步骤厚度 / nm","步骤横向偏移 / nm","步骤 CD 偏置 / nm","步骤刻蚀深度 / nm","侧壁包覆厚度 / nm"),changed=self.changed)
        for row,(key,label,val) in enumerate((("start","起点",500),("stop","终点",750),("samples","点数",11),("step","工艺步骤序号（几何扫描时，从 1 开始）",2)),start=2):
            self.sweep_vars[key] = self._field(sweep,row,label,val,changed=self.changed)
        actions = ttk.Frame(page)
        actions.pack(fill="x",pady=10)
        self._button(actions,"运行 RCWA",self.run,True)
        self._button(actions,"检查阶数收敛",lambda:self.run(converge=True))
        self._button(actions,"打开结果目录",self.open_result_folder)
        ttk.Label(page,text="收敛检查：关闭扫描后逐步增加 N，要求连续两档各传播反射阶复振幅相对变化≤1%，同时满足 R/T、各阶效率与有效相位条件。\n数值近零项单列、不做相对验收；变化达标不等于真解精度达到1%。反射矩阵输出请选择 TE + TM。\n当前一维各向同性常规入射下 Rsp/Rps 为对称性零，入射半空间需无损。圆锥入射、偏振耦合及内部近场尚未接入界面。",foreground="#6f6f6f",wraplength=1150,justify="left").pack(anchor="w",pady=14)

    def _results_page(self):
        ttk,tk,page = self.ttk,self.tk,self.pages["results"]
        row = ttk.Frame(page)
        row.grid(row=0,column=0,sticky="ew")
        self._button(row,"加载已有结果",self.load_result)
        self._button(row,"导出 Excel",self.export_excel,True)
        self._button(row,"打开结果目录",self.open_result_folder)
        self._button(row,"导出当前图",self.export_plot)
        self._button(row,"扫描曲线",self.show_scan)
        controls = ttk.Frame(page)
        controls.grid(row=1,column=0,sticky="ew",pady=(8,0))
        self.point_var = tk.StringVar()
        self.point_combo = ttk.Combobox(controls,textvariable=self.point_var,state="readonly",width=48)
        self.point_combo.pack(side="left",fill="x",expand=True,padx=(0,15))
        self.point_combo.bind("<<ComboboxSelected>>",lambda _e:self.safe(self.show_point))
        self.incident_s = tk.StringVar(value="1+0j")
        self.incident_p = tk.StringVar(value="0+0j")
        for label,variable in (("Es_in",self.incident_s),("Ep_in",self.incident_p)):
            ttk.Label(controls,text=label).pack(side="left",padx=(5,4))
            entry = ttk.Entry(controls,textvariable=variable,width=11)
            entry.pack(side="left")
            entry.bind("<Return>",lambda _e:self.safe(self.show_point))
        self._button(controls,"更新复振幅",self.show_point)
        self.result_note = tk.StringVar(value="尚未计算。默认输出 R_matrix.xlsx；复振幅按所填入射分量计算，不自动归一化。")
        ttk.Label(page,textvariable=self.result_note,wraplength=1200,font=("Microsoft YaHei UI",10)).grid(row=2,column=0,sticky="w",pady=10)
        self.result_plot = ttk.Frame(page)
        self.result_plot.grid(row=3,column=0,sticky="nsew")
        page.rowconfigure(3,weight=1,minsize=180)
        page.columnconfigure(0,weight=1)
        self.result_canvas = None
        frame = ttk.Frame(page)
        frame.grid(row=4,column=0,sticky="ew",pady=(10,0))
        self.result_columns = ("m","Rss","Rsp","Rps","Rpp","Es","Ep","abs_s","phase_s","abs_p","phase_p")
        table = ttk.Frame(frame)
        table.pack(fill="x")
        self.result_tree = ttk.Treeview(table,columns=self.result_columns,show="headings",height=5)
        for key,label in zip(self.result_columns,("阶数 m","Rss","Rsp","Rps","Rpp","Es_out 复振幅","Ep_out 复振幅","|Es_out|","Es 相位 / °","|Ep_out|","Ep 相位 / °")):
            self.result_tree.heading(key,text=label)
            self.result_tree.column(key,width=235 if key in ("Rss","Rsp","Rps","Rpp","Es","Ep") else 100,minwidth=65,stretch=True)
        scroll = ttk.Scrollbar(frame,orient="horizontal",command=self.result_tree.xview)
        yscroll = ttk.Scrollbar(table,orient="vertical",command=self.result_tree.yview)
        self.result_tree.configure(xscrollcommand=scroll.set,yscrollcommand=yscroll.set)
        yscroll.pack(side="right",fill="y")
        self.result_tree.pack(side="left",fill="both",expand=True)
        scroll.pack(fill="x")

    def safe(self, function):
        try:
            return function()
        except Exception as exc:
            from tkinter import messagebox
            self.last_error = str(exc)
            self.status.set(str(exc))
            self.log(f"错误：{type(exc).__name__}: {exc}")
            messagebox.showerror("MarkRCWA",str(exc),parent=self.root)
            return None

    def log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end",time.strftime("%H:%M:%S")+"  "+str(message)+"\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def changed(self):
        if self.loading:
            return
        self.dirty = True
        self._caption()
        if self.result:
            self.result_stale = True
            self.result_note.set("设置已更改；下方显示上一次计算结果，重新运行后更新。")

    def _step_changed(self):
        if not self.loading:
            self.step_dirty = True
            self.changed()

    def _caption(self):
        self.project_caption.set(self.project.get("name","工程")+(" · 未保存" if self.dirty else ""))

    def _load_forms(self):
        previous,self.loading = self.loading,True
        sim = self.project["simulation"]
        for key in ("wavelength_nm","theta_deg","fourier_order","convergence_max_order"):
            self.sim_vars[key].set(str(sim.get(key,127)))
        self.sim_vars["reference_z_um"].set("" if sim.get("reference_z_um") is None else str(sim["reference_z_um"]))
        self.sim_vars["polarization"].set("TE + TM" if len(sim.get("polarizations",[]))==2 else sim.get("polarizations",["TE"])[0])
        self.sweep_enabled.set(False)
        sweep = sim.get("sweep",{})
        if sweep.get("enabled"):
            self.sweep_enabled.set(True)
            reverse = {"wavelength_nm":"波长 / nm","theta_deg":"入射角 / °","thickness_um":"步骤厚度 / nm","delta_x_um":"步骤横向偏移 / nm","cd_bias_um":"步骤 CD 偏置 / nm","etch_depth_um":"步骤刻蚀深度 / nm","sidewall_thickness_um":"侧壁包覆厚度 / nm"}
            parameter = sweep["parameter"]
            geometric = parameter.startswith("steps.")
            self.sweep_vars["parameter"].set(reverse[parameter.split(".")[-1]])
            for key in ("start","stop"):
                self.sweep_vars[key].set(str(sweep[key]*(1000 if geometric else 1)))
            self.sweep_vars["samples"].set(str(sweep["samples"]))
            if geometric:
                self.sweep_vars["step"].set(str(int(parameter.split(".")[1])+1))
        self.step_index,self.step_dirty = None,False
        self.refresh_steps()
        self.refresh_materials()
        kind = self.project.get("model_kind","process")
        self.model_note.set("工艺步骤建模 · 复用 MarkSim 的材料、布尔几何、包覆及填平规则。" if kind=="process" else "PH 参数截面建模 · 保留梯形/圆角/多边形切片；用‘高级工艺参数’编辑截面参数。")
        self.preview_dir = None
        self.preview_image.configure(image="",text="尚未生成几何")
        self.layer_tree.delete(*self.layer_tree.get_children())
        self.loading = previous
        self._caption()
        if kind=="process":
            target = self._pattern_target()
            if target is not None:
                self.step_tree.selection_set(str(target))
                self.select_step()

    def refresh_materials(self):
        self.material_tree.delete(*self.material_tree.get_children())
        if self.project.get("model_kind","process") == "profile":
            from .materials import from_config
            registry = from_config(self._profile_material_specs())
            for name,mat in registry.items():
                spec = {"wavelength_nm":[w*1000 for w in mat.wavelength_um] or [100,100000],
                        "n":[z.real for z in mat.n_table] or [mat.n.real]*2,
                        "k":[z.imag for z in mat.n_table] or [mat.n.imag]*2,"source":mat.source}
                self.material_tree.insert("","end",iid=name,text=name,values=("常数" if mat.n is not None else str([spec['wavelength_nm'][0],spec['wavelength_nm'][-1]]),"已提供 k",mat.source))
            self._profile_registry = registry
        else:
            for name,spec in self.project["materials"]["materials"].items():
                waves = spec["wavelength_nm"]
                loss = "已提供 k" if spec.get("k") is not None else "明确无损近似" if spec.get("assume_lossless") else "k 未知 / 暂不可计算"
                self.material_tree.insert("","end",iid=name,text=name,values=(f"{waves[0]:g} – {waves[-1]:g}",loss,spec.get("source","")))

    def _profile_material_specs(self):
        specs = copy.deepcopy(self.project.get("profile_materials",{}))
        for spec in specs.values():
            if "table_csv" in spec and not Path(spec["table_csv"]).is_absolute():
                spec["table_csv"] = str(self.base_dir/spec["table_csv"])
        return specs

    def plot_material(self):
        selection = self.material_tree.selection()
        if not selection:
            return
        name = selection[0]
        if self.project.get("model_kind","process")=="profile":
            mat = self._profile_registry[name]
            spec = {"wavelength_nm":[w*1000 for w in mat.wavelength_um] or [100,100000],
                    "n":[z.real for z in mat.n_table] or [mat.n.real]*2,
                    "k":[z.imag for z in mat.n_table] or [mat.n.imag]*2}
        else:
            spec = self.project["materials"]["materials"][name]
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        if self.material_canvas:
            self.material_canvas.get_tk_widget().destroy()
        fig = Figure(figsize=(9,3),dpi=100)
        axes = fig.subplots(1,2)
        axes[0].plot(spec["wavelength_nm"],spec["n"],"o-",color="#0f62fe")
        axes[0].set(xlabel="Vacuum wavelength / nm",ylabel="n",title=name)
        if spec.get("k") is not None:
            axes[1].plot(spec["wavelength_nm"],spec["k"],"o-",color="#8a3ffc")
        else:
            axes[1].text(.5,.5,"k unknown",ha="center",transform=axes[1].transAxes)
        axes[1].set(xlabel="Vacuum wavelength / nm",ylabel="k")
        fig.tight_layout()
        fig.set_layout_engine("constrained")
        self.material_canvas = FigureCanvasTkAgg(fig,master=self.material_plot)
        self.material_canvas.draw()
        self.material_canvas.get_tk_widget().configure(width=1,height=1)
        self.material_canvas.get_tk_widget().pack(fill="both",expand=True)

    def import_materials(self, folder=False):
        from tkinter import filedialog
        from .modeling import import_material_txt
        if self.project.get("model_kind","process") != "process":
            raise ValueError("PH 截面材料可通过高级参数编辑；TXT 管理用于工艺建模工程。")
        if folder:
            selected = filedialog.askdirectory(parent=self.root,title="选择材料 TXT 文件夹")
            if not selected:
                return
            paths = sorted(Path(selected).glob("*.txt"))
        else:
            paths = [Path(p) for p in filedialog.askopenfilenames(parent=self.root,title="导入材料数据",filetypes=[("材料数据","*.txt *.csv"),("全部文件","*.*")])]
        if not paths:
            return
        columns = [int(v.strip())-1 for v in self.material_columns.get().split(",")]
        if len(columns) not in (2,3):
            raise ValueError("列号应为 λ,n 或 λ,n,k 的两列或三列")
        mapping = dict(zip(("wavelength","n","k"),columns))
        imported = {p.stem:import_material_txt(p,wavelength_unit=self.material_unit.get(),columns=mapping,assume_lossless=self.assume_lossless.get()) for p in paths}
        self.project["materials"]["materials"].update(imported)
        self.refresh_materials()
        self.changed()
        encodings = "、".join(sorted({spec["import"]["encoding"] for spec in imported.values()}))
        self.status.set(f"已导入 {len(imported)} 种材料（读取编码：{encodings}）；数据已存入工程快照，保存时复制原始文件。")

    def add_material(self):
        values = self.form_dialog("常数材料",[("name","名称","自定义材料"),("n","n",1.5),("k","k",0)])
        if values is None:
            return
        name = values["name"].strip()
        n,k = float(values["n"]),float(values["k"])
        if not name or not math.isfinite(n) or not math.isfinite(k) or n<=0 or k<0:
            raise ValueError("材料名称不能为空；n 应为正数，k 应为非负数")
        if self.project.get("model_kind","process")=="profile":
            self.project.setdefault("profile_materials",{})[name] = {"n":n,"k":k,"source":"用户指定常数"}
        else:
            self.project["materials"]["materials"][name] = {"wavelength_nm":[100.,100000.],"n":[n,n],"k":[k,k],"source":"用户指定常数（有效范围 100–100000 nm）","assume_lossless":False}
        self.refresh_materials()
        self.changed()

    def delete_material(self):
        selection = self.material_tree.selection()
        if not selection:
            return
        if self.project.get("model_kind","process")=="profile":
            self.project.get("profile_materials",{}).pop(selection[0],None)
        else:
            self.project["materials"]["materials"].pop(selection[0])
        self.refresh_materials()
        self.changed()

    def refresh_steps(self):
        self.step_tree.delete(*self.step_tree.get_children())
        if self.project.get("model_kind","process") != "process":
            return
        names = {v:k for k,v in self.operations.items()}
        for i,step in enumerate(self.project["process"]["steps"]):
            self.step_tree.insert("","end",iid=str(i),text=f"{i+1:02d}  {step.get('name',step['id'])}"+("（停用）" if not step.get("enabled",True) else ""),values=(names.get(step["operation"],step["operation"]),step.get("material",",".join(step.get("target_materials",[])))))

    def select_step(self):
        selection = self.step_tree.selection()
        if not selection:
            return
        index = int(selection[0])
        if self.step_index == index:
            return
        if self.step_dirty and self.step_index is not None:
            self._commit_step()
        self.step_index = index
        self._populate_step()

    def _populate_step(self):
        if self.step_index is None:
            return
        step = self.project["process"]["steps"][self.step_index]
        previous,self.loading = self.loading,True
        reverse = {v:k for k,v in self.operations.items()}
        mode = step.get("film_mode","absolute")
        values = {"name":step.get("name",step["id"]),"operation":reverse[step["operation"]],"material":step.get("material",",".join(step.get("target_materials",[]))),
                  "film_mode":{v:k for k,v in self.film_modes.items()}.get(mode,"指定高度"),"z":1000*step.get("start_z_um",step.get("z_min_um",0)),
                  "height":1000*(step["z_max_um"]-step.get("z_min_um",0) if "z_max_um" in step else step.get("thickness_um",.1)),
                  "depth":1000*step.get("etch_depth_um",.1),"sidewall":1000*step.get("sidewall_thickness_um",.1),
                  "cd":1000*step.get("cd_bias_um",0),"dx":1000*step.get("delta_x_um",0),"mask_mode":"保留原掩模",
                  "gds":step.get("mask",{}).get("gds_path",""),"cell":step.get("mask",{}).get("cell",""),
                  "layer":step.get("mask",{}).get("layer",1),"datatype":step.get("mask",{}).get("datatype",0)}
        polygons = step.get("mask",{}).get("polygons_xy_um",[])
        if len(polygons)==1:
            xs = [p[0] for p in polygons[0]]
            values.update(left=min(xs),right=max(xs))
        for key,value in values.items():
            self.step_vars[key].set(str(value))
        self.step_enabled.set(step.get("enabled",True))
        self.step_dirty = False
        self.loading = previous

    def _step_from_form(self):
        if self.step_index is None:
            return None
        step = copy.deepcopy(self.project["process"]["steps"][self.step_index])
        text = {k:v.get().strip() for k,v in self.step_vars.items()}
        op = self.operations[text["operation"]]
        material = text["material"]
        if not material:
            raise ValueError("请填写材料或目标材料")
        def length(key,positive=False,nonnegative=False):
            x = float(text[key])/1000
            if not math.isfinite(x) or (positive and x<=0) or (nonnegative and x<0):
                raise ValueError(f"{key} 需要有限且符合范围的数值")
            return x
        mode = self.film_modes[text["film_mode"]]
        for key in ("material","target_materials","z_min_um","z_max_um","thickness_um","start_z_um","etch_depth_um","film_mode","sidewall_thickness_um"):
            step.pop(key,None)
        step.update(name=text["name"] or step["id"],operation=op,enabled=self.step_enabled.get(),sidewall_angle_deg=90)
        surface = op=="create_film" and mode!="absolute"
        if op=="etch":
            step.update(target_materials=[m.strip() for m in material.split(",")],start_z_um=length("z"),etch_depth_um=length("depth",positive=True))
        elif op=="delete":
            step.update(target_materials=[m.strip() for m in material.split(",")],z_min_um=length("z"),z_max_um=length("z")+length("height",positive=True))
        else:
            step["material"] = material
            step["thickness_um"] = length("height",positive=True)
            if not surface:
                step["z_min_um"] = length("z")
            if op=="create_film":
                step["film_mode"] = mode
                if mode=="conformal":
                    step["sidewall_thickness_um"] = length("sidewall",nonnegative=True)
        if surface:
            for key in ("mask","mask_polarity","cd_bias_um","delta_x_um","delta_y_um","rotation_deg"):
                step.pop(key,None)
        else:
            step.update(cd_bias_um=length("cd"),delta_x_um=length("dx"))
            if text["mask_mode"]=="整个单元":
                step.pop("mask",None)
            elif text["mask_mode"]=="矩形条纹":
                left,right = float(text["left"]),float(text["right"])
                if not math.isfinite(left) or not math.isfinite(right) or left>=right:
                    raise ValueError("矩形左边界应小于右边界")
                y0,y1 = self.project["process"]["cell_bounds_um"][1]
                step["mask"] = {"polygons_xy_um":[[[left,y0],[right,y0],[right,y1],[left,y1]]]}
            elif text["mask_mode"]=="GDS 图层":
                step["mask"] = {"gds_path":text["gds"],"cell":text["cell"],"layer":int(text["layer"]),"datatype":int(text["datatype"])}
        if "mark_pattern" in step:
            step["mark_pattern"]["height_um"] = step.get("thickness_um",step.get("z_max_um",0)-step.get("z_min_um",0))
            if "material" in step:
                step["mark_pattern"]["material"] = step["material"]
        return step

    def _commit_step(self):
        if self.step_index is None or not self.step_dirty:
            return
        self.project["process"]["steps"][self.step_index] = self._step_from_form()
        self.step_dirty = False
        self.changed()

    def apply_step(self):
        self._require_process()
        self._commit_step()
        index = self.step_index
        self.refresh_steps()
        if index is not None:
            self.step_tree.selection_set(str(index))
        self.status.set("已应用步骤；生成几何查看实际结构。")

    def _require_process(self):
        if self.project.get("model_kind","process") != "process":
            raise ValueError("此工程为 PH 参数截面；请用高级参数编辑，或新建工艺工程。")

    def add_step(self):
        self._require_process()
        self._commit_step()
        materials = self.project["materials"]["materials"]
        background = self.project["process"]["background_material"]
        material = "coat_demo" if "coat_demo" in materials else next((m for m in materials if m!=background),background)
        self.project["process"]["steps"].append({"id":"step_"+uuid.uuid4().hex[:8],"name":"新增薄膜","operation":"create_film","material":material,"film_mode":"conformal","thickness_um":.05,"sidewall_thickness_um":.05,"enabled":True,"sidewall_angle_deg":90})
        index = len(self.project["process"]["steps"])-1
        self.step_index = None
        self.refresh_steps()
        self.step_tree.selection_set(str(index))
        self.select_step()
        self.changed()

    def delete_step(self):
        if self.step_index is None:
            return
        self.project["process"]["steps"].pop(self.step_index)
        self.step_index,self.step_dirty = None,False
        self.refresh_steps()
        self.changed()

    def move_step(self,delta):
        if self.step_index is None:
            return
        self._commit_step()
        steps = self.project["process"]["steps"]
        i,j = self.step_index,self.step_index+delta
        if not 0<=j<len(steps):
            return
        steps[i],steps[j] = steps[j],steps[i]
        self.step_index = None
        self.refresh_steps()
        self.step_tree.selection_set(str(j))
        self.select_step()
        self.changed()

    def choose_gds(self):
        from tkinter import filedialog
        from .modeling.geometry import inspect_gds
        selected = filedialog.askopenfilename(parent=self.root,title="选择 GDS 文件",filetypes=[("GDS","*.gds *.gdsii"),("全部文件","*.*")])
        if not selected:
            return
        info = inspect_gds(selected)
        self.log(json.dumps(info,ensure_ascii=False))
        self.step_vars["gds"].set(selected)
        self.step_vars["mask_mode"].set("GDS 图层")
        self.status.set("已读取 GDS；Cell 和图层信息见任务日志，请填写目标 Cell / Layer / Datatype。")

    def edit_cell(self):
        self._require_process()
        process = self.project["process"]
        bounds = process["cell_bounds_um"]
        fields = [("pitch","周期 · µm",process["period_x_um"]),("x0","x 起点 · µm",bounds[0][0]),("y0","y 起点 · µm",bounds[1][0]),("y1","y 终点 · µm",bounds[1][1]),("z0","z 起点 · µm",bounds[2][0]),("z1","z 终点 · µm",bounds[2][1])]
        values = self.form_dialog("计算单元（改变单元不会缩放掩模）",fields)
        if values is None:
            return
        n = {k:float(v) for k,v in values.items()}
        if not all(math.isfinite(v) for v in n.values()) or n["pitch"]<=0 or n["y0"]>=n["y1"] or n["z0"]>=n["z1"]:
            raise ValueError("单元边界应为有限且递增的区间")
        process.update(period_x_um=n["pitch"],cell_bounds_um=[[n["x0"],n["x0"]+n["pitch"]],[n["y0"],n["y1"]],[n["z0"],n["z1"]]])
        self.changed()

    def edit_process_json(self):
        self._commit_step()
        kind = self.project.get("model_kind","process")
        value = self.project["process"] if kind=="process" else {"geometry":self.project["profile"],"materials":self.project["profile_materials"]}
        edited = self.json_dialog("高级工艺参数" if kind=="process" else "PH 截面与材料参数",value)
        if edited is None:
            return
        if kind=="process":
            self.project["process"] = edited
        else:
            self.project["profile"],self.project["profile_materials"] = edited["geometry"],edited["materials"]
        self.step_index,self.step_dirty = None,False
        self.refresh_steps()
        self.refresh_materials()
        self.changed()

    def form_dialog(self,title,fields):
        top = self.tk.Toplevel(self.root)
        top.title(title)
        top.transient(self.root)
        body = self.ttk.Frame(top,padding=18)
        body.pack(fill="both",expand=True)
        values = {key:self._field(body,i,label,value,width=34) for i,(key,label,value) in enumerate(fields)}
        answer = []
        def accept():
            answer.append({k:v.get() for k,v in values.items()})
            top.destroy()
        self.ttk.Button(body,text="应用",command=accept).grid(row=len(fields),column=1,sticky="e",pady=14)
        top.bind("<Return>",lambda _e:accept())
        top.grab_set()
        self.root.wait_window(top)
        return answer[0] if answer else None

    def json_dialog(self,title,value):
        top = self.tk.Toplevel(self.root)
        top.title(title)
        top.geometry("850x650")
        top.transient(self.root)
        text = self.tk.Text(top,wrap="none",font=("Consolas",11))
        text.pack(fill="both",expand=True,padx=12,pady=12)
        text.insert("1.0",json.dumps(value,ensure_ascii=False,indent=2))
        answer = []
        def accept():
            from tkinter import messagebox
            try:
                data = json.loads(text.get("1.0","end"))
                if not isinstance(data,dict):
                    raise ValueError("参数应为对象")
                answer.append(data)
                top.destroy()
            except Exception as exc:
                messagebox.showerror(title,str(exc),parent=top)
        self.ttk.Button(top,text="应用参数",command=accept).pack(anchor="e",padx=12,pady=(0,12))
        top.grab_set()
        self.root.wait_window(top)
        return answer[0] if answer else None

    def _snapshot(self):
        from .desktop_project import simulation_points
        self._commit_step()
        spec = {"wavelength_nm":float(self.sim_vars["wavelength_nm"].get()),"theta_deg":float(self.sim_vars["theta_deg"].get()),
                "fourier_order":int(self.sim_vars["fourier_order"].get()),"convergence_max_order":int(self.sim_vars["convergence_max_order"].get()),
                "polarizations":["TE","TM"] if self.sim_vars["polarization"].get()=="TE + TM" else [self.sim_vars["polarization"].get()]}
        reference_text = self.sim_vars["reference_z_um"].get().strip()
        spec["reference_z_um"] = float(reference_text) if reference_text else None
        if self.sweep_enabled.get():
            name = self.sweep_vars["parameter"].get()
            mapping = {"波长 / nm":"wavelength_nm","入射角 / °":"theta_deg","步骤厚度 / nm":"thickness_um","步骤横向偏移 / nm":"delta_x_um","步骤 CD 偏置 / nm":"cd_bias_um","步骤刻蚀深度 / nm":"etch_depth_um","侧壁包覆厚度 / nm":"sidewall_thickness_um"}
            parameter = mapping[name]
            geometric = parameter.endswith("_um")
            if geometric:
                self._require_process()
                index = int(self.sweep_vars["step"].get())-1
                if index<0 or index>=len(self.project["process"]["steps"]):
                    raise ValueError("几何扫描的工艺步骤序号不存在")
                parameter = f"steps.{index}.{parameter}"
            spec["sweep"] = {"enabled":True,"parameter":parameter,"start":float(self.sweep_vars["start"].get())/(1000 if geometric else 1),
                             "stop":float(self.sweep_vars["stop"].get())/(1000 if geometric else 1),"samples":int(self.sweep_vars["samples"].get())}
        simulation_points(spec)
        self.project["simulation"] = spec
        return copy.deepcopy(self.project)

    def _set_project(self,data,base_dir,file=None):
        self.project,self.base_dir,self.project_file = data,Path(base_dir),Path(file) if file else None
        self.dirty = file is None
        self._load_forms()
        self.result = None
        self.result_stale = False
        self.point_combo.configure(values=[])
        self.point_var.set("")
        self.result_tree.delete(*self.result_tree.get_children())
        if self.result_canvas:
            self.result_canvas.get_tk_widget().destroy()
            self.result_canvas = None
        self.result_note.set("尚未计算当前工程。")
        self.status.set("已打开工程 · 先生成几何检查结构，再运行 RCWA。")

    def _preserve_draft(self):
        if not self.dirty:
            return
        from .desktop_project import save_project
        data = self._snapshot()
        save_project(data,self.workspace/"recovery.mrcwa.json",base_dir=self.base_dir)

    def new_project(self):
        from .desktop_project import default_project
        self._preserve_draft()
        self._set_project(default_project(),self.workspace)
        self.tabs.select(self.pages["process"])

    def open_project(self):
        from tkinter import filedialog
        from .desktop_project import load_project
        selected = filedialog.askopenfilename(parent=self.root,title="打开 MarkRCWA 工程",filetypes=[("MarkRCWA","*.mrcwa.json"),("JSON","*.json")])
        if selected:
            data = load_project(selected)
            self._preserve_draft()
            self._set_project(data,Path(selected).parent,selected)

    def save(self,save_as=False):
        from tkinter import filedialog
        from .desktop_project import save_project
        data = self._snapshot()
        target = self.project_file
        if save_as or target is None:
            chosen = filedialog.asksaveasfilename(parent=self.root,title="保存工程及输入快照",defaultextension=".mrcwa.json",initialfile=self.project.get("name","工程")+".mrcwa.json",filetypes=[("MarkRCWA","*.mrcwa.json")])
            if not chosen:
                return
            target = Path(chosen)
        self.project = save_project(data,target,base_dir=self.base_dir)
        self.project_file,self.base_dir = Path(target),Path(target).parent
        self._populate_step()
        self.dirty = False
        self._caption()
        self.status.set(f"已保存：{target}")

    def import_marksim(self):
        from tkinter import filedialog
        from .desktop_project import import_marksim
        chosen = filedialog.askdirectory(parent=self.root,title="选择包含 project.yaml 的 MarkSim 工程目录")
        if chosen:
            data,base = import_marksim(chosen)
            self._preserve_draft()
            self._set_project(data,base)
            self.tabs.select(self.pages["process"])

    def import_ph(self):
        from tkinter import filedialog
        from .desktop_project import import_ph_config
        chosen = filedialog.askopenfilename(parent=self.root,title="选择 PH 截面配置",filetypes=[("PH 配置","*.yaml *.yml *.json")])
        if chosen:
            data,base = import_ph_config(chosen)
            self._preserve_draft()
            self._set_project(data,base)
            self.tabs.select(self.pages["process"])

    def _start(self,kind,operation):
        if self.busy:
            raise ValueError("请等待当前任务完成")
        self.busy,self.last_error = True,None
        self.cancel_event.clear()
        self.cancel_button.configure(state="normal")
        self.progress["value"] = 0
        for button in self._buttons:
            button.configure(state="disabled")
        def work():
            try:
                payload = operation()
                self.messages.put(("done",(kind,payload)))
            except Exception as exc:
                import traceback
                self.messages.put(("failed",(str(exc),traceback.format_exc(),isinstance(exc,InterruptedError))))
        threading.Thread(target=work,daemon=True,name="MarkRCWA-job").start()

    def build_preview(self):
        from .desktop_project import prepare_model
        from .modeling import preview_geometry, build_geometry, geometry_to_stack
        data = self._snapshot()
        base = self.base_dir
        folder = self.workspace / "previews" / uuid.uuid4().hex[:10]
        self.status.set("正在构建几何并检查一维适用性…")
        def work():
            reason = None
            if data.get("model_kind","process")=="process":
                geometry = build_geometry(data["process"],base_dir=base)
                try:
                    stack = geometry_to_stack(geometry)
                except ValueError as exc:
                    stack,reason = None,str(exc)
            else:
                geometry,stack,_materials = prepare_model(data,base)
            preview_geometry(geometry,folder)
            return {"geometry":geometry,"stack":stack,"preview_dir":str(folder),"ineligible_reason":reason}
        self._start("preview",work)

    def run(self,converge=False):
        from .desktop_project import run_simulation
        data = self._snapshot()
        base = self.base_dir
        output = self.project_file.parent/(self.project_file.stem+"_results") if self.project_file else self.workspace/"runs"
        self.status.set("开始 RCWA 计算…")
        self.log("运行 RCWA；结果目录："+str(output))
        self._start("simulation",lambda:run_simulation(data,base,output,cancel_event=self.cancel_event,
                    progress=lambda row:self.messages.put(("progress",row)),converge=converge))

    def _poll(self):
        try:
            while True:
                kind,value = self.messages.get_nowait()
                if kind=="progress":
                    self.status.set(value["message"])
                    self.progress["value"] = value.get("fraction",0)
                    self.log(value["message"])
                elif kind=="failed":
                    self.last_error,trace,cancelled = value
                    self.log(trace)
                    self.status.set("任务已取消；不完整任务保留状态记录。" if cancelled else "计算失败："+self.last_error)
                    self._finish()
                elif kind=="done":
                    task,payload = value
                    if task=="preview":
                        self.preview_dir = Path(payload["preview_dir"])
                        if payload["stack"] is not None:
                            self._accept_preview(payload["stack"])
                            self.status.set("几何已生成；一维适用性检查通过。")
                        else:
                            self.layer_tree.delete(*self.layer_tree.get_children())
                            self.preview_note.set("图案已导入并生成 3D 预览；尚不满足一维 RCWA 条件：\n"+payload["ineligible_reason"])
                            self._resize_preview()
                            self.status.set("可查看几何；仿真前需调整选区，使结构沿 y 不变。")
                    else:
                        self.result = payload
                        self.preview_dir = Path(payload["run_dir"])/"geometry"
                        self._resize_preview()
                        self._refresh_result_choices()
                        from .desktop_project import fingerprint
                        try:
                            stale = fingerprint(self._snapshot()) != payload["input_sha256"]
                        except (ValueError,KeyError,TypeError):
                            stale = True
                        self.result_stale = stale
                        self.show_point()
                        self.tabs.select(self.pages["results"])
                        self.status.set("计算完成 · "+("阶数收敛检查通过" if payload["manifest"].get("convergence_verified") else "请查看各点收敛状态"))
                        self.log("结果已保存："+payload["run_dir"])
                    self.progress["value"] = 1
                    self._finish()
        except queue.Empty:
            pass
        except Exception as exc:
            import traceback
            self.last_error = str(exc)
            self.log(traceback.format_exc())
            self.status.set("显示结果失败："+str(exc))
            self._finish()
        if self.root.winfo_exists():
            self.root.after(100,self._poll)

    def _finish(self):
        self.busy = False
        self.cancel_button.configure(state="disabled")
        for button in self._buttons:
            button.configure(state="normal")
        if self.pending_close:
            self.root.after_idle(self.close)

    def _accept_preview(self,stack):
        self.preview_note.set(f"周期 {stack.pitch_um:g} µm · 有限结构厚度 {stack.thickness_um:g} µm · {len(stack.layers)} 个 RCWA 层\n入射介质：{stack.top}　半无限衬底：{stack.bottom}")
        self.layer_tree.delete(*self.layer_tree.get_children())
        z = 0.
        for s in stack.layers:
            self.layer_tree.insert("","end",values=(f"{z:g} → {z+s.thickness_um:g} / {s.thickness_um:g}"," | ".join(s.materials)))
            z += s.thickness_um
        self._resize_preview()

    def _resize_preview(self):
        if self.preview_dir is None:
            return
        path = self.preview_dir/"geometry_sections.png"
        if not path.is_file():
            return
        from PIL import Image, ImageTk
        with Image.open(path) as source:
            image = source.copy()
        width,height = max(300,self.preview_image.winfo_width()-10),max(220,self.preview_image.winfo_height()-10)
        image.thumbnail((width,height),Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.preview_image.configure(image=self.photo,text="")

    def open_3d(self):
        if self.preview_dir is None or not (self.preview_dir/"geometry_3d.html").is_file():
            raise ValueError("先生成几何")
        webbrowser.open((self.preview_dir/"geometry_3d.html").resolve().as_uri())

    def _refresh_result_choices(self):
        from .reflection_matrix import build_reflection_matrices
        self.matrix_points = build_reflection_matrices(self.result["records"])
        self.point_combo.configure(values=[f"{i+1:03d} · {r['wavelength_nm']:g} nm · {r['theta_deg']:g}°"+(f" · 几何值 {r['geometry_value_um']:g} µm" if r.get("geometry_value_um") is not None else "") for i,r in enumerate(self.matrix_points)])
        if self.matrix_points:
            self.point_combo.current(0)

    def _incident_vector(self):
        try:
            values = tuple(complex(v.get().strip().replace(" ","").replace("I","j").replace("i","j")) for v in (self.incident_s,self.incident_p))
        except ValueError as exc:
            raise ValueError("入射分量请填复数，例如 1、0.7071 或 0+0.7071j。") from exc
        if not all(math.isfinite(v.real) and math.isfinite(v.imag) for v in values) or not any(values):
            raise ValueError("入射分量必须为有限复数，且不能同时为零。")
        return values

    def show_point(self):
        if not self.result:
            return
        from .reflection_matrix import amplitude_for_row, amplitude_phase_reliable
        incident = self._incident_vector()
        i = max(0,self.point_combo.current())
        point = self.matrix_points[i]
        status_names = {"not_checked":"未检查收敛","PASS":"连续变化判据通过（非精度保证）","UNCONVERGED":"变化尚未达标"}
        status = " / ".join(f"{pol} N={point['fourier_order'][pol]} {status_names.get(point['convergence_status'][pol],point['convergence_status'][pol])}" for pol in point["polarizations"])
        self.result_note.set(f"{point['wavelength_nm']:g} nm · {point['theta_deg']:g}° · {status}\nR 的行/列为输出/输入 s,p（固定实验室 E 基）；Rsp/Rps 为一维常规入射对称性零；— 表示未计算或相位无效。")
        if self.result_stale:
            self.result_note.set("历史结果 / 当前设置已更改 · " + self.result_note.get())
        rows = point["rows"]
        amplitudes = [amplitude_for_row(r,incident) for r in rows]
        reliability = [amplitude_phase_reliable(r,incident) for r in rows]
        self.result_tree.delete(*self.result_tree.get_children())
        def fmt(value):
            return "—" if value is None else f"{value:.7g}"
        def complex_fmt(value):
            return "—" if value is None else f"{value.real:.6g}{value.imag:+.6g}j"
        for r,amplitude,reliable in zip(rows,amplitudes,reliability):
            es,ep = amplitude["Es_out"],amplitude["Ep_out"]
            def phase(value,key):
                return math.degrees(math.atan2(value.imag,value.real)) if value is not None and reliable[key] else None
            self.result_tree.insert("","end",values=(r["order"],*(complex_fmt(r[key]) for key in ("Rss","Rsp","Rps","Rpp")),complex_fmt(es),complex_fmt(ep),fmt(abs(es) if es is not None else None),fmt(phase(es,"Es_out")),fmt(abs(ep) if ep is not None else None),fmt(phase(ep,"Ep_out"))))
        from matplotlib.figure import Figure
        import numpy as np
        if self.result_canvas:
            self.result_canvas.get_tk_widget().destroy()
        fig = Figure(figsize=(11,3.6),dpi=100)
        axes = fig.subplots(1,2)
        orders = np.array([r["order"] for r in rows])
        for key,color,style in (("Rss","#0f62fe","-"),("Rsp","#009d9a",":"),("Rps","#da1e28","--"),("Rpp","#8a3ffc","-.")):
            axes[0].plot(orders,[abs(r[key]) if r[key] is not None else np.nan for r in rows],style,marker="o",markersize=3,label=key,color=color)
        axes[0].set(xlabel="Diffraction order m",ylabel="Matrix |Rij|")
        for key,color in (("Es_out","#0f62fe"),("Ep_out","#8a3ffc")):
            for part,style in (("real","-"),("imag","--")):
                axes[1].plot(orders,[getattr(a[key],part) if a[key] is not None else np.nan for a in amplitudes],style,marker="o",markersize=3,label=f"{key} {part}",color=color)
        axes[1].set(xlabel="Diffraction order m",ylabel="Complex E")
        for axis in axes:
            axis.tick_params(labelsize=8)
            axis.xaxis.label.set_size(8)
            axis.yaxis.label.set_size(8)
            axis.legend(fontsize=8,ncol=2)
            axis.grid(alpha=.2)
        fig.tight_layout()
        fig.set_layout_engine("constrained")
        self._embed_result_figure(fig)

    def _embed_result_figure(self,figure):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.result_figure = figure
        self.result_canvas = canvas = FigureCanvasTkAgg(figure,master=self.result_plot)
        widget = canvas.get_tk_widget()
        def fit_after_dpi_change(_event):
            # Tk may change figure DPI after the initial Configure event.
            figure.set_size_inches(widget.winfo_width()/figure.dpi,widget.winfo_height()/figure.dpi,forward=False)
            canvas.draw_idle()
        widget.bind("<Map>",fit_after_dpi_change,add="+")
        canvas.draw()
        widget.configure(width=1,height=1)
        widget.pack(fill="both",expand=True)

    def load_result(self):
        from tkinter import filedialog
        chosen = filedialog.askopenfilename(parent=self.root,title="选择 summary.json",filetypes=[("结果汇总","summary.json")])
        if not chosen:
            return
        folder = Path(chosen).parent
        records = json.loads(Path(chosen).read_text(encoding="utf-8-sig"))
        manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8-sig"))
        if manifest.get("status")!="succeeded" or not records:
            raise ValueError("该任务未完整完成，不能作为完整结果加载")
        self.result = {"run_dir":str(folder),"records":records,"manifest":manifest}
        from .desktop_project import fingerprint
        self.result_stale = fingerprint(self._snapshot()) != manifest.get("input_sha256")
        self._refresh_result_choices()
        self.show_point()
        self.status.set("已加载历史结果；当前建模设置保持不变。")

    def show_scan(self):
        from .desktop_project import scan_figure
        if not self.result:
            raise ValueError("先运行参数扫描或加载扫描结果")
        figure = scan_figure(self.result["records"])
        if figure is None:
            raise ValueError("当前任务是单点计算；在 RCWA 设置中启用参数扫描。")
        if self.result_canvas:
            self.result_canvas.get_tk_widget().destroy()
        self._embed_result_figure(figure)
        self.result_note.set(("历史结果 / 当前设置已更改 · " if self.result_stale else "")+f"扫描汇总 · {len(self.result['records'])} 个偏振/参数点；下表为所选单点的逐阶结果。")

    def open_result_folder(self):
        folder = Path(self.result["run_dir"]) if self.result else self.workspace
        if os.name=="nt":
            os.startfile(str(folder))
        else:
            webbrowser.open(folder.resolve().as_uri())

    def export_plot(self):
        from tkinter import filedialog
        if not self.result_canvas:
            raise ValueError("先计算或加载结果")
        selected = filedialog.asksaveasfilename(parent=self.root,title="导出当前图",defaultextension=".png",filetypes=[("PNG","*.png"),("PDF","*.pdf")])
        if selected:
            self.result_figure.savefig(selected,dpi=180)
            self.status.set("已导出："+selected)

    def export_excel(self):
        from tkinter import filedialog
        from .reflection_matrix import export_reflection_excel
        if not self.result:
            raise ValueError("先计算或加载结果")
        incident = self._incident_vector()
        selected = filedialog.asksaveasfilename(parent=self.root,title="导出反射矩阵与逐阶复振幅",initialdir=self.result["run_dir"],initialfile="R_matrix.xlsx",defaultextension=".xlsx",filetypes=[("Excel 工作簿","*.xlsx")])
        if selected:
            export_reflection_excel(self.result["records"],selected,incident=incident)
            self.show_point()
            self.status.set("已导出反射矩阵与复振幅："+selected)

    def cancel(self):
        if self.busy:
            self.cancel_event.set()
            self.status.set("正在取消；当前计算点结束后停止。")

    def help(self):
        from tkinter import messagebox
        messagebox.showinfo("MarkRCWA 使用说明","1. 在工艺与结构页点击‘导入 GDS’或‘定义 Mark 图案’。\nGDS 可选择 Cell / Layer / Datatype、选区和材料；Mark 可选光栅、AH32/AH53/AH74 或自定义多条纹。\n2. 导入真实 n/k 材料，编辑工艺并生成几何。\n3. 在 RCWA 设置中选择波长、角度、TE + TM 和阶数。\n4. 结果页查看 Rss/Rsp/Rps/Rpp，填写 Es_in/Ep_in，计算各阶输出复振幅并导出 Excel。\n5. 结果目录自动生成 R_matrix.xlsx（单位 s 入射），并保留 CSV、JSON、HDF5。\n\n快捷键：Ctrl+O 打开，Ctrl+S 保存，F5 生成几何，Esc 取消。\n3D 视图使用内置资源的本地 HTML，可断网查看。\n当前为一维各向同性常规入射，交叉反射项为零；不含光路/APD。",parent=self.root)

    def close(self):
        if self.busy:
            self.pending_close = True
            self.cancel()
            return
        try:
            self._preserve_draft()
        except Exception:
            pass
        self.root.destroy()


def _smoke_ui(output):
    import tkinter as tk
    from .desktop_project import write_json
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    root = tk.Tk()
    app = DesktopApp(root)
    app.project["simulation"]["fourier_order"] = 7
    app._load_forms()
    app.tabs.select(app.pages["process"])
    app.build_preview()
    start = time.monotonic()
    state = {"stage":"preview"}
    def capture(name,widget=None):
        from PIL import ImageGrab
        import ctypes
        root.update()
        widget = widget or root
        hwnd = ctypes.windll.user32.GetAncestor(widget.winfo_id(),2) if os.name=="nt" else widget.winfo_id()
        ImageGrab.grab(window=hwnd).save(output/name)
    def step():
        if app.last_error or time.monotonic()-start>120:
            write_json(output/"ui_smoke.json",{"ok":False,"error":app.last_error or "timeout"})
            root.destroy()
            return
        if not app.busy:
            if state["stage"]=="preview":
                capture("geometry_ui.png")
                from .desktop_patterns import MarkPatternDialog
                dialog = MarkPatternDialog(app)
                dialog.kind.set("AH53（3 条 / 5 阶）")
                dialog.load_kind()
                dialog.update_cell.set(True)
                dialog.preview()
                capture("mark_designer.png",dialog.top)
                dialog.apply()
                if not dialog.accepted:
                    raise AssertionError("Mark dialog: "+dialog.message.get())
                state["stage"] = "mark"
            elif state["stage"]=="mark":
                capture("mark_geometry.png")
                import gdstk
                from .desktop_patterns import GdsImportDialog
                library = gdstk.Library()
                cell = library.new_cell("DESKTOP_MARK")
                cell.add(gdstk.rectangle((-1.,-.5),(1.,.5),layer=7))
                for x in (-3.2,0.,3.2):
                    cell.add(gdstk.rectangle((x-.8,-.5),(x+.8,.5),layer=12,datatype=3))
                path = output/"workflow.gds"
                library.write_gds(path)
                dialog = GdsImportDialog(app,path)
                dialog.layer.set("12 / 3")
                dialog.fields["height_nm"].set("180")
                dialog.preview()
                capture("gds_importer.png",dialog.top)
                dialog.apply()
                if not dialog.accepted:
                    raise AssertionError("GDS dialog: "+dialog.message.get())
                state["stage"] = "gds"
            elif state["stage"]=="gds":
                capture("gds_geometry.png")
                masks = [s for s in app.project["process"]["steps"] if s.get("mask")]
                if len(masks)!=1 or masks[0]["mask"].get("layer")!=12 or masks[0]["mask"].get("datatype")!=3:
                    raise AssertionError("GDS layer selection or replacement failed")
                state["stage"] = "simulation"
                app.run()
            else:
                capture("results_ui.png")
                write_json(output/"ui_smoke.json",{"ok":True,"status":"PASS","points":len(app.result["records"]),"preview_dir":str(app.preview_dir),"run_dir":app.result["run_dir"],
                    "mark_designer_verified":True,"gds_importer_verified":True,"selected_gds_cell":"DESKTOP_MARK","selected_gds_layer":12,"selected_gds_datatype":3})
                root.destroy()
                return
        root.after(200,tick)
    def tick():
        try:
            step()
        except Exception as exc:
            import traceback
            write_json(output/"ui_smoke.json",{"ok":False,"error":str(exc),"traceback":traceback.format_exc()})
            root.destroy()
    root.after(300,tick)
    root.mainloop()
    report = json.loads((output/"ui_smoke.json").read_text(encoding="utf-8"))
    return 0 if report["ok"] else 1


def main(argv=None):
    import multiprocessing
    multiprocessing.freeze_support()
    _configure_runtime()
    parser = argparse.ArgumentParser(description="MarkRCWA offline desktop")
    parser.add_argument("--self-test",metavar="DIRECTORY")
    parser.add_argument("--smoke-ui",metavar="DIRECTORY")
    parser.add_argument("--project",metavar="FILE")
    parser.add_argument("--export-r-matrix",metavar="SUMMARY_JSON",help="export existing results without solving")
    parser.add_argument("--excel",metavar="OUTPUT_XLSX")
    args = parser.parse_args(argv)
    if bool(args.export_r_matrix) != bool(args.excel):
        parser.error("--export-r-matrix and --excel must be used together")
    if args.export_r_matrix:
        from .reflection_matrix import export_reflection_excel
        records = json.loads(Path(args.export_r_matrix).read_text(encoding="utf-8-sig"))
        export_reflection_excel(records,args.excel)
        return 0
    if args.self_test:
        from .desktop_project import self_test
        self_test(args.self_test)
        return 0
    if args.smoke_ui:
        return _smoke_ui(args.smoke_ui)
    import tkinter as tk
    root = tk.Tk()
    app = DesktopApp(root)
    if args.project:
        from .desktop_project import load_project
        path = Path(args.project).resolve()
        app.safe(lambda:app._set_project(load_project(path),path.parent,path))
    root.mainloop()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
