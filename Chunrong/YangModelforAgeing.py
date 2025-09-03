import pybamm as pb

# http://dx.doi.org/10.1016/j.jpowsour.2017.05.110
# https://doi.org/10.1016/j.jpowsour.2018.09.069

class YangPlatingSEI(pb.BaseSubModel):
    """
    Yang et al. 2017: SEI growth + Li plating (irreversible) + porosity drop
    with film resistance and EC diffusion-limited SEI kinetics;
    Yang & Wang 2018: Arrhenius T-dependence for key kinetic/transport params.
    """

    def __init__(self, param, domain="negative"):
        super().__init__(param, domain)
        self.domain      = domain
        self.phase       = "primary"
        self.domain_Dict = pb.standard_spatial_vars
        self.a           = param.a_n  if domain == "negative" else param.a_p  # specific surface area
        # ---- 参数名占位：按你参数表对接（建议放到自定义 ParameterValues.yaml 再注入）----
        self.k0_SEI      = param.k0_SEI
        self.alpha_c_SEI = param.alpha_c_SEI
        self.D_EC        = param.D_EC
        self.cEC_bulk    = param.cEC_bulk
        self.U_SEI       = param.U_SEI
        self.k_SEI_ion   = param.k_SEI_ion       # k_SEI (ionic conductivity in SEI)
        self.u_SEI       = param.u_SEI           # volume fraction of SEI in film
        self.M_SEI       = param.M_SEI
        self.rho_SEI     = param.rho_SEI
        self.M_Li        = param.M_Li
        self.rho_Li      = param.rho_Li

        self.alpha_a_int = param.alpha_a_int
        self.alpha_c_int = param.alpha_c_int
        self.i0_int      = param.i0_int          # exchange current density for intercalation

        self.alpha_c_Li  = param.alpha_c_Li
        self.i0_Li       = param.i0_Li_irrev     # irreversible part only (as in Yang 2017)

        # Arrhenius（随温度的）——建议在 ParameterValues 里用 functions 实现，这里仅示意
        self.arrhenius   = getattr(param, "arrhenius_wrapper", lambda x, Eact: x)

    # 变量注册
    def get_fundamental_variables(self):
        var = {}
        dom = self.domain.capitalize()  # "Negative"
        # 取 PyBaMM 标准变量
        phi_s   = pb.standard_variables.get(f"{dom} electrode potential")
        phi_e   = pb.standard_variables.get(f"{dom} electrolyte potential")
        T       = pb.standard_variables.get("Cell temperature")
        c_e     = pb.standard_variables.get("Electrolyte concentration")
        c_s_surf= pb.standard_variables.get(f"{dom} particle surface concentration")

        # 总膜厚 δ_film 作为状态变量（SEI + Li）
        delta   = pb.standard_variables.get(f"{dom} film thickness")  # 你可用 pb.Variable 新增状态
        # 局部 EC 表面浓度（通过稳态扩散平衡近似得到）
        cEC_s   = pb.Variable(f"{dom} EC surface concentration")

        var.update({
            f"{dom} electrode potential": phi_s,
            f"{dom} electrolyte potential": phi_e,
            "Cell temperature": T,
            "Electrolyte concentration": c_e,
            f"{dom} particle surface concentration": c_s_surf,
            f"{dom} film thickness": delta,
            f"{dom} EC surface concentration": cEC_s,
        })
        return var

    # 反应动力学与耦合变量
    def get_coupled_variables(self, variables):
        dom = self.domain.capitalize()
        F   = pb.constants.F
        R   = pb.constants.R

        phi_s    = variables[f"{dom} electrode potential"]
        phi_e    = variables[f"{dom} electrolyte potential"]
        T        = variables["Cell temperature"]
        c_s_surf = variables[f"{dom} particle surface concentration"]
        delta    = variables[f"{dom} film thickness"]
        cEC_s    = variables[f"{dom} EC surface concentration"]

        # --- 膜电阻：R_film = u_SEI * δ_film / k_SEI_ion  (仅 SEI 承担离子导通) ---
        R_film   = self.u_SEI * delta / self.k_SEI_ion

        # --- 交换电流（可用温度因子修饰） ---
        i0_int_T = self.arrhenius(self.i0_int,  self.param.Eact_i0_int)
        i0_Li_T  = self.arrhenius(self.i0_Li,   self.param.Eact_i0_Li)
        k0_SEI_T = self.arrhenius(self.k0_SEI,  self.param.Eact_k0_SEI)
        D_EC_T   = self.arrhenius(self.D_EC,    self.param.Eact_DEC)

        # --- 总反应电流是三者之和：j_tot = j_int + j_SEI + j_Li ---
        # 先占位（需要迭代一致性，PyBaMM 中常以 algebraic 方程联立解）
        j_tot = pb.Variable(f"{dom} total interfacial current density")

        # --- 主反应过电位：η_int = φ_s - φ_e - (j_tot/a) R_film - U_int(c_s_surf, T) ---
        U_int  = self.param.U_n(c_s_surf, T) if self.domain == "negative" else self.param.U_p(c_s_surf, T)
        eta_int= phi_s - phi_e - j_tot / self.a * R_film - U_int

        # --- 主反应电流：Butler-Volmer （Yang 2017 式(8)）---
        j_int  = self.a * i0_int_T * (
                    pb.exp(self.alpha_a_int * F * eta_int / (R * T))
                    - pb.exp(-self.alpha_c_int * F * eta_int / (R * T))
                 )

        # --- SEI 反应：阴极 Tafel（Yang 2017 式(10)），受 EC 扩散限制 ---
        # EC 跨膜通量平衡：-D_EC (cEC_s - cEC_bulk) / δ = - j_SEI / F  => cEC_s 代入 Tafel
        j_SEI   = - self.a * F * k0_SEI_T * cEC_s * pb.exp(
                    - self.alpha_c_SEI * F * (phi_s - phi_e - j_tot / self.a * R_film - self.U_SEI) / (R * T)
                 )

        # --- Li plating：阴极 Tafel（Yang 2017 式(12)，只取不可逆部分）---
        eta_Li  = phi_s - phi_e - j_tot / self.a * R_film   # 相对于 Li/Li+
        j_Li    = - self.a * i0_Li_T * pb.exp(
                    - self.alpha_c_Li * F * (eta_Li) / (R * T)
                 )

        # --- EC 表面浓度代数约束（Yang 2017 式(11)）---
        ec_balance = - D_EC_T * (cEC_s - self.cEC_bulk) / pb.maximum(delta, 1e-12) - ( - j_SEI / F )
        self.algebraic = getattr(self, "algebraic", {})
        self.algebraic[cEC_s] = ec_balance

        # --- 总电流一致性：j_tot - (j_int + j_SEI + j_Li) = 0 ---
        self.algebraic[j_tot] = j_tot - (j_int + j_SEI + j_Li)

        # --- 膜厚动力学（由 Faraday 定律与体积换算） & 孔隙率演化（Yang 2017 式(13)–(16)）---
        # dδ/dt = (V_SEI rate + V_Li rate) / a ； V_SEI = (|j_SEI|/(2F)) * (M_SEI/ρ_SEI)； V_Li = (|j_Li|/F)*(M_Li/ρ_Li)
        V_SEI_rate = (-j_SEI) / (2*F) * (self.M_SEI / self.rho_SEI)
        V_Li_rate  = (-j_Li)  / (1*F)  * (self.M_Li  / self.rho_Li )
        ddelta_dt  = (V_SEI_rate + V_Li_rate) / self.a

        # 负极孔隙率变量（PyBaMM 已有 epsilon_n，直接构造其源项）
        eps_n      = pb.standard_variables.get("Negative electrode porosity")
        deps_dt    = - self.a * ddelta_dt

        self.rhs = getattr(self, "rhs", {})
        self.rhs[delta] = ddelta_dt
        self.rhs[eps_n] = deps_dt

        # 导出变量
        variables.update({
            f"{dom} interfacial current density (int)": j_int,
            f"{dom} interfacial current density (SEI)": j_SEI,
            f"{dom} interfacial current density (Li)":  j_Li,
            f"{dom} total interfacial current density": j_tot,
            f"{dom} film resistance": R_film,
            f"{dom} film thickness rate": ddelta_dt,
            f"{dom} porosity rate": deps_dt,
        })
        return variables



class YangDFNWithPlating(pb.lithium_ion.DFN):
    """
    DFN 主模型 + Yang(2017) plating/SEI 子模型（负极）。
    使用方法：
      params = pb.ParameterValues("Chen2020")  # 或你的 NMC/石墨参数表
      # 追加 Yang 2017/2018 所需自定义参数（i0_Li_irrev, k0_SEI, D_EC, U_SEI, M/rho...与激活能）
      model = YangDFNWithPlating(options={"thermal": "lumped"})
      sim   = pb.Simulation(model, parameter_values=params)
      sim.solve([0, 3600])  # 例：一次充放
    """
    def set_interfacial_submodels(self):
        super().set_interfacial_submodels()
        # 用我们自定义的负极“副反应包”叠加（或替换）默认 SEI/析锂
        self.submodels["negative plating sei (yang)"] = YangPlatingSEI(self.param, domain="negative")

    def set_porosity_submodel(self):
        # 调用 DFN 默认（包含 eps 的 PDE），我们仅提供负极源项（rhs 已在子模型里写入）
        super().set_porosity_submodel()
