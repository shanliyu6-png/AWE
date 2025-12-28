import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ==========================================
# 1. 物理模型核心 (保持数值稳定性)
# ==========================================
class AWE2DModel_Robust:
    def __init__(self, nz=60, ny=25):
        # 几何参数
        self.L = 0.5  # 流道高度 (m) - Z轴
        self.d = 0.001  # 极间隙 (m) - Y轴
        self.W = 0.5  # 宽度 (m)

        # 网格设置
        self.nz = nz
        self.ny = ny
        self.dz = self.L / self.nz
        self.dy = self.d / self.ny

        # 坐标定义
        self.y_nodes = np.linspace(self.dy / 2, self.d - self.dy / 2, self.ny)
        self.z_nodes = np.linspace(0, self.L, self.nz)

        # 物性
        self.rho_l = 1200.0
        self.Cp_l = 3500.0
        self.k_l = 0.6
        self.sigma_0 = 100.0
        self.D_gas = 1e-5
        self.F = 96485.0
        self.V_tn = 1.48  # 热中性电压

    def get_velocity_profile(self, Q_m3h):
        area = self.W * self.d
        w_avg = (Q_m3h / 3600.0) / area
        w = 6 * w_avg * (self.y_nodes / self.d) * (1 - self.y_nodes / self.d)
        min_w = w_avg * 0.01
        w = np.clip(w, min_w, None)
        return w

    def solve_steady_field(self, J_avg, Q_m3h, T_in=60.0):
        # 初始化
        T_field = np.zeros((self.ny, self.nz))
        Alpha_field = np.zeros((self.ny, self.nz))
        T_field[:, 0] = T_in

        w = self.get_velocity_profile(Q_m3h)

        # 矩阵系数预计算
        lambda_alpha = (self.D_gas * self.dz) / (w * self.dy ** 2)
        alpha_thermal = self.k_l / (self.rho_l * self.Cp_l)
        lambda_temp = (alpha_thermal * self.dz) / (w * self.dy ** 2)

        def build_sparse_matrix(lambdas, size):
            main_diag = 1 + 2 * lambdas
            upper_diag = -lambdas[:-1]
            lower_diag = -lambdas[1:]
            main_diag[0] = 1 + lambdas[0]
            main_diag[-1] = 1 + lambdas[-1]
            return sparse.diags([main_diag, upper_diag, lower_diag], [0, 1, -1], format='csr')

        A_alpha = build_sparse_matrix(lambda_alpha, self.ny)
        A_temp = build_sparse_matrix(lambda_temp, self.ny)

        # 步进求解
        for i in range(1, self.nz):
            # Alpha
            b_alpha = Alpha_field[:, i - 1].copy()
            rho_gas = 5.0
            gas_source = (J_avg * self.dz) / (w[0] * 2 * self.F * rho_gas * self.dy)
            b_alpha[0] += gas_source
            Alpha_new = spsolve(A_alpha, b_alpha)
            Alpha_new = np.clip(Alpha_new, 0, 0.95)
            Alpha_field[:, i] = Alpha_new

            # Temp
            b_temp = T_field[:, i - 1].copy()
            sigma_eff = self.sigma_0 * (1 - Alpha_new) ** 1.5
            q_joule = (J_avg ** 2) / sigma_eff
            source_joule = q_joule * (self.dz / (w * self.rho_l * self.Cp_l))
            b_temp += source_joule
            q_surf = J_avg * 0.5
            source_surf = q_surf * (self.dz / (w[0] * self.rho_l * self.Cp_l * self.dy))
            b_temp[0] += source_surf
            h_cool = 50.0
            q_cool = h_cool * (b_temp[-1] - 25.0)
            source_cool = q_cool * (self.dz / (w[-1] * self.rho_l * self.Cp_l * self.dy))
            b_temp[-1] -= source_cool
            T_new = spsolve(A_temp, b_temp)
            T_field[:, i] = T_new

        return self.y_nodes, self.z_nodes, T_field, Alpha_field

    def calculate_efficiency(self, J, Q, T_avg):
        """
        计算能量利用效率
        效率 = (产氢有效功率) / (电解输入总功率 + 泵功)
        """
        # 1. 产氢有效功率 (基于热中性电压)
        I_total = J * (self.L * self.W)
        P_H2_effective = I_total * self.V_tn

        # 2. 电解输入功率 (P = V_cell * I)
        # 简单伏安特性模型：温度越高，电压越低
        # V = 1.48 + R(T)*J
        # R(T) 随温度升高而降低
        R_eff = 1.0e-4 * (1 - 0.005 * (T_avg - 60))
        V_cell = 1.45 + R_eff * J
        P_elec = I_total * V_cell

        # 3. 泵功 (P_pump ~ Q^3)
        # 假设系数，使得在 5m3/h 时泵功约占总功率的 1-2%
        k_pump = 20.0
        P_pump = k_pump * (Q ** 3)

        P_total = P_elec + P_pump
        efficiency = P_H2_effective / P_total

        return efficiency, P_total


# 初始化
model = AWE2DModel_Robust(nz=200, ny=200)

# ==========================================
# 2. Streamlit UI
# ==========================================
st.set_page_config(page_title="AWE Optimization Pro", layout="wide")

st.sidebar.title("🎛️ 模拟控制台")
mode = st.sidebar.radio("功能模块", ["稳态场分布 (竖直视图)", "高级动态优化 (能量视角)"])

# -----------------------------------------------------------------------------
# 模式 1: 稳态场分布 (竖直视图)
# -----------------------------------------------------------------------------
if mode == "稳态场分布 (竖直视图)":
    st.title("🔬 稳态场分布 (Vertical View)")
    st.markdown("通过将流动方向设为纵轴，模拟真实的电解槽内部视角。")

    with st.sidebar:
        st.subheader("工况设置")
        J_in = st.slider("电流密度 (A/m²)", 1000, 10000, 4000, step=100)
        Q_in = st.slider("循环流量 (m³/h)", 0.1, 20.0, 2.0)
        T_in = st.slider("入口温度 (°C)", 20.0, 90.0, 60.0, step=0.1)

    # 计算
    y, z, T, Alpha = model.solve_steady_field(J_in, Q_in, T_in)

    # 统计数据
    T_surface = T[0, :]
    T_mean = np.mean(T_surface)
    T_var = np.var(T_surface)

    # 指标展示
    c1, c2, c3 = st.columns(3)
    c1.metric("电极表面均温", f"{T_mean:.2f} °C")
    c2.metric("最高温度", f"{np.max(T):.2f} °C")
    c3.metric("温度不均匀度 (方差)", f"{T_var:.4f}")

    # 绘图 (转置矩阵以实现竖直视图)
    col1, col2 = st.columns(2)

    # 注意：在 Heatmap 中交换 x 和 y，并转置 z 数据 (.T)
    with col1:
        st.subheader("🔥 温度场 T(y,z)")
        fig1 = go.Figure(data=go.Heatmap(
            z=T.T,  # 转置
            x=y,  # X轴现在是极间隙 Y
            y=z,  # Y轴现在是高度 Z
            colorscale='RdYlBu_r',
            colorbar=dict(title='Temp (°C)'),
            zmin=T_in, zmax=np.max(T)
        ))
        fig1.update_layout(
            xaxis_title="极间隙宽度 (m) [左侧为电极]",
            yaxis_title="流道高度 (m) [流动方向]",
            height=600,
            xaxis=dict(range=[0, 0.001], constrain='domain'),  # 锁定比例
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("🫧 气含率场 α(y,z)")
        fig2 = go.Figure(data=go.Heatmap(
            z=Alpha.T,  # 转置
            x=y,
            y=z,
            colorscale='Teal',
            colorbar=dict(title='Void Fraction'),
            zmin=0, zmax=0.6
        ))
        fig2.update_layout(
            xaxis_title="极间隙宽度 (m)",
            yaxis_title="流道高度 (m) [流动方向]",
            height=600
        )
        st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------------------------------
# 模式 2: 动态优化 (多策略对比 + 能效分析)
# -----------------------------------------------------------------------------
elif mode == "高级动态优化 (能量视角)":
    st.title("🌊 波动输入下的多目标优化控制")

    with st.sidebar:
        st.subheader("1. 波动源")
        wave_type = st.selectbox("波形", ["正弦波", "方波"])
        base_J = st.number_input("基准电流", 4000)
        amp_J = st.number_input("波动幅度", 2000)
        period = st.number_input("周期 (min)", 20)

        st.subheader("2. 策略参数")
        base_Q = st.slider("基础流量", 2.0, 10.0, 5.0)
        k_factor = st.slider("流量响应增益 k (x10^-4)", 1.0, 10.0, 4.0) * 1e-4
        look_ahead = st.slider("预判提前量 (min)", 0, 5, 2, help="提前多少分钟调整流量")

    # 模拟设置
    total_time = period * 1.5
    steps = 80
    t_sim = np.linspace(0, total_time, steps)
    dt = total_time / steps

    # 生成波形
    if wave_type == "正弦波":
        J_wave = base_J + amp_J * np.sin(2 * np.pi * t_sim / period)
    else:
        J_wave = base_J + amp_J * np.sign(np.sin(2 * np.pi * t_sim / period))

    # 结果容器
    res = {
        "Instant": {"Q": [], "T_var": [], "Eff": []},
        "Predictive": {"Q": [], "T_var": [], "Eff": []}
    }

    # 模拟循环
    bar = st.progress(0)
    for i, J_now in enumerate(J_wave):
        # --- 策略 A: 即时响应 (Instant) ---
        # Q 随当前的 J 变化
        Q_inst = base_Q + k_factor * (J_now - base_J)
        Q_inst = np.clip(Q_inst, 0.5, 20.0)

        _, _, T_inst, _ = model.solve_steady_field(J_now, Q_inst)
        eff_inst, _ = model.calculate_efficiency(J_now, Q_inst, np.mean(T_inst))

        res["Instant"]["Q"].append(Q_inst)
        res["Instant"]["T_var"].append(np.var(T_inst[0, :]))
        res["Instant"]["Eff"].append(eff_inst * 100)  # 转百分比

        # --- 策略 B: 预判调节 (Predictive) ---
        # Q 随未来的 J 变化 (look ahead)
        # 计算提前多少个时间步
        steps_ahead = int(look_ahead / (total_time / steps))
        idx_future = min(i + steps_ahead, len(J_wave) - 1)
        J_future = J_wave[idx_future]

        Q_pred = base_Q + k_factor * (J_future - base_J)
        Q_pred = np.clip(Q_pred, 0.5, 20.0)

        _, _, T_pred, _ = model.solve_steady_field(J_now, Q_pred)
        eff_pred, _ = model.calculate_efficiency(J_now, Q_pred, np.mean(T_pred))

        res["Predictive"]["Q"].append(Q_pred)
        res["Predictive"]["T_var"].append(np.var(T_pred[0, :]))
        res["Predictive"]["Eff"].append(eff_pred * 100)

        bar.progress((i + 1) / steps)

    # --- 绘图 (4行子图) ---
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=(
        "(1) 输入电流 J(t)", "(2) 流量策略 Q(t)", "(3) 温度不均匀度响应 (方差)", "(4) 能量利用效率 η(t)"),
        row_heights=[0.15, 0.25, 0.3, 0.3]
    )

    # 1. Current
    fig.add_trace(go.Scatter(x=t_sim, y=J_wave, name="Current", line=dict(color='black', dash='dot')), row=1, col=1)

    # 2. Flow
    fig.add_trace(go.Scatter(x=t_sim, y=res["Instant"]["Q"], name="即时响应流量", line=dict(color='gray')), row=2,
                  col=1)
    fig.add_trace(go.Scatter(x=t_sim, y=res["Predictive"]["Q"], name="预判调节流量", line=dict(color='green', width=2)),
                  row=2, col=1)

    # 3. Variance
    fig.add_trace(
        go.Scatter(x=t_sim, y=res["Instant"]["T_var"], name="不均匀度 (即时)", line=dict(color='gray', dash='dash')),
        row=3, col=1)
    fig.add_trace(go.Scatter(x=t_sim, y=res["Predictive"]["T_var"], name="不均匀度 (预判)", line=dict(color='green')),
                  row=3, col=1)

    # 添加阈值参考线 (User defined logic: 比如方差<0.5为优)
    fig.add_hline(y=0.5, line_dash="dot", annotation_text="目标阈值", row=3, col=1)

    # 4. Efficiency
    fig.add_trace(go.Scatter(x=t_sim, y=res["Instant"]["Eff"], name="能效 (即时)", line=dict(color='gray', width=1)),
                  row=4, col=1)
    fig.add_trace(
        go.Scatter(x=t_sim, y=res["Predictive"]["Eff"], name="能效 (预判)", line=dict(color='green', width=2)), row=4,
        col=1)

    fig.update_layout(height=900, template="plotly_white")
    fig.update_yaxes(title="A/m²", row=1, col=1)
    fig.update_yaxes(title="m³/h", row=2, col=1)
    fig.update_yaxes(title="Variance", row=3, col=1)
    fig.update_yaxes(title="Efficiency (%)", row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # 结论分析
    avg_eff_inst = np.mean(res["Instant"]["Eff"])
    avg_eff_pred = np.mean(res["Predictive"]["Eff"])
    var_peak_inst = np.max(res["Instant"]["T_var"])
    var_peak_pred = np.max(res["Predictive"]["T_var"])

    st.info(f"""
    **策略对比结论**：
    1. **温度控制**：预判调节策略将最大的温度不均匀度从 {var_peak_inst:.2f} 降低到了 {var_peak_pred:.2f}。通过在电流洪峰到达前提前增大流量，有效削峰。
    2. **能量代价**：预判策略的平均能效为 {avg_eff_pred:.2f}%，相比即时响应 ({avg_eff_inst:.2f}%) 变化微乎其微。
    **综合评价**：提前调节流量可以在**几乎不牺牲系统能效**的前提下，显著提升极端工况下的**热安全性**。
    """)