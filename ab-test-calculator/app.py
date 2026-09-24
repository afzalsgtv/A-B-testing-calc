import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from ab_core import (
    ContinuousResult,
    ProportionResult,
    analyze_continuous,
    analyze_proportions,
    peeking_curve,
    sample_size_proportions,
    test_duration_days,
)

st.set_page_config(page_title="Калькулятор A/B-тестов", page_icon="📊", layout="wide")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
ALPHAS = [0.01, 0.05, 0.10]


def fmt_int(x: int) -> str:
    return f"{x:,}".replace(",", " ")


def style_axes(fig: go.Figure, x_title: str = "", y_title: str = "") -> go.Figure:
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK, size=13),
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    axis = dict(
        showgrid=True,
        gridcolor=GRID,
        linecolor=AXIS,
        tickfont=dict(color=INK_MUTED),
        title_font=dict(color=INK_MUTED, size=12),
        zeroline=False,
    )
    fig.update_xaxes(title_text=x_title, **axis)
    fig.update_yaxes(title_text=y_title, **axis)
    return fig


st.title("Калькулятор A/B-тестов")
tab_plan, tab_analyze, tab_peek = st.tabs(["Планирование", "Анализ", "Подглядывание"])


with tab_plan:
    left, right = st.columns([1, 1.4], gap="large")

    with left:
        p_base = st.number_input("Базовая конверсия, %", 0.01, 99.0, 5.0, step=0.1) / 100

        mde_mode = st.radio("MDE", ["Относительный, %", "Абсолютный, п.п."], horizontal=True)
        if mde_mode == "Относительный, %":
            mde_rel = st.number_input("MDE, %", 0.1, 500.0, 10.0, step=0.5) / 100
            mde_abs = p_base * mde_rel
        else:
            mde_abs = st.number_input("MDE, п.п.", 0.01, 50.0, 0.5, step=0.05) / 100
            mde_rel = mde_abs / p_base

        c1, c2 = st.columns(2)
        alpha = c1.selectbox("α", ALPHAS, index=1)
        power = c2.selectbox("Мощность", [0.80, 0.90, 0.95], index=0)
        two_sided = st.checkbox("Двусторонний тест", value=True)

        st.divider()
        daily_users = st.number_input("Пользователей в день", min_value=1, value=5000, step=100)
        share = st.slider("Доля трафика в тесте, %", 1, 100, 100) / 100

    with right:
        try:
            n_per_group = sample_size_proportions(p_base, mde_abs, alpha, power, two_sided)
        except ValueError as exc:
            st.error(str(exc))
        else:
            days = test_duration_days(n_per_group, daily_users, share)

            m1, m2, m3 = st.columns(3)
            m1.metric("На группу", fmt_int(n_per_group))
            m2.metric("Всего", fmt_int(2 * n_per_group))
            m3.metric("Длительность", f"{days:.1f} дн.")
            st.caption(f"{p_base:.2%} → {p_base + mde_abs:.2%} ({mde_rel:+.1%})")

            mde_grid = np.linspace(max(mde_abs * 0.3, 1e-4), mde_abs * 2.5, 60)
            mde_grid = mde_grid[(p_base + mde_grid) < 0.999]

            fig = go.Figure()
            for pw, color in [(0.80, SERIES_1), (0.90, SERIES_2)]:
                ys = [sample_size_proportions(p_base, m, alpha, pw, two_sided) for m in mde_grid]
                fig.add_trace(
                    go.Scatter(
                        x=mde_grid * 100,
                        y=ys,
                        mode="lines",
                        name=f"Мощность {pw:.0%}",
                        line=dict(color=color, width=2),
                        hovertemplate="MDE %{x:.2f} п.п.<br>%{y:,.0f} на группу<extra></extra>",
                    )
                )
            fig.add_vline(
                x=mde_abs * 100,
                line=dict(color=INK_MUTED, width=1, dash="dot"),
                annotation_text="ваш MDE",
                annotation_position="top",
                annotation_font=dict(color=INK_MUTED, size=11),
            )
            fig.update_yaxes(type="log", dtick=1, minor=dict(showgrid=False))
            style_axes(fig, "MDE, п.п.", "Пользователей на группу")
            st.plotly_chart(fig, width="stretch")


def verdict_block(p_value: float, alpha: float, significant: bool) -> None:
    if significant:
        st.success(f"**Значимо** · p-value = {p_value:.4f} < α = {alpha}")
    else:
        st.info(f"**Не значимо** · p-value = {p_value:.4f} ≥ α = {alpha}")


def render_proportions(res: ProportionResult, n_a: int, n_b: int) -> None:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Конверсия A", f"{res.p_a:.3%}")
    m2.metric("Конверсия B", f"{res.p_b:.3%}", f"{res.diff_abs * 100:+.3f} п.п.")
    m3.metric("Отн. эффект", f"{res.lift_rel:+.2%}")
    m4.metric("p-value", f"{res.p_value:.4f}")

    verdict_block(res.p_value, res.alpha, res.significant)

    conf = int(round((1 - res.alpha) * 100))
    st.markdown(
        f"ДИ {conf}%: `[{res.diff_ci[0] * 100:+.3f}; {res.diff_ci[1] * 100:+.3f}]` п.п. · "
        f"`[{res.lift_ci[0]:+.2%}; {res.lift_ci[1]:+.2%}]` отн."
    )

    z = stats.norm.ppf(1 - res.alpha / 2)
    groups = [
        ("Контроль (A)", res.p_a, np.sqrt(res.p_a * (1 - res.p_a) / n_a), n_a, SERIES_1),
        ("Вариант (B)", res.p_b, np.sqrt(res.p_b * (1 - res.p_b) / n_b), n_b, SERIES_2),
    ]

    fig = go.Figure()
    for name, val, se, _, color in groups:
        fig.add_trace(
            go.Scatter(
                x=[val * 100],
                y=[name],
                mode="markers+text",
                name=name,
                marker=dict(size=13, color=color, line=dict(width=2, color=SURFACE)),
                error_x=dict(type="data", array=[z * se * 100], color=color, thickness=2, width=8),
                text=[f"{val:.3%}"],
                textposition="top center",
                textfont=dict(color=INK, size=12),
                hovertemplate=f"{name}: %{{x:.3f}}%<extra></extra>",
            )
        )
    style_axes(fig, "Конверсия, %")
    fig.update_layout(height=260, hovermode="closest")
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        pd.DataFrame(
            {
                "Группа": [g[0] for g in groups],
                "Наблюдений": [g[3] for g in groups],
                "Конверсия": [f"{g[1]:.4%}" for g in groups],
                f"ДИ {conf}%": [f"[{g[1] - z * g[2]:.4%}; {g[1] + z * g[2]:.4%}]" for g in groups],
            }
        ),
        hide_index=True,
        width="stretch",
    )

    if not res.significant and not np.isnan(res.mde_achievable):
        st.warning(
            f"MDE при такой выборке (мощность 80%): "
            f"**{res.mde_achievable * 100:.3f} п.п.** ({res.mde_achievable / res.p_a:.1%} отн.)"
        )


def render_continuous(res: ContinuousResult) -> None:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Среднее A", f"{res.mean_a:,.2f}")
    m2.metric("Среднее B", f"{res.mean_b:,.2f}", f"{res.diff:+,.2f}")
    m3.metric("Отн. эффект", f"{res.lift_rel:+.2%}")
    m4.metric("p-value", f"{res.p_value:.4f}")

    verdict_block(res.p_value, res.alpha, res.significant)

    conf = int(round((1 - res.alpha) * 100))
    st.markdown(
        f"ДИ {conf}%: `[{res.diff_ci[0]:+,.3f}; {res.diff_ci[1]:+,.3f}]` · "
        f"t = {res.t_stat:.3f}, df = {res.df:.1f}"
    )


with tab_analyze:
    source = st.radio("Данные", ["Вручную", "CSV"], horizontal=True)
    alpha_a = st.selectbox("α", ALPHAS, index=1, key="alpha_a")

    if source == "Вручную":
        c1, c2 = st.columns(2)
        n_a = c1.number_input("Пользователей A", min_value=1, value=10000, step=100)
        conv_a = c1.number_input("Конверсий A", min_value=0, value=500, step=10)
        n_b = c2.number_input("Пользователей B", min_value=1, value=10000, step=100)
        conv_b = c2.number_input("Конверсий B", min_value=0, value=560, step=10)

        st.divider()
        try:
            render_proportions(analyze_proportions(n_a, conv_a, n_b, conv_b, alpha_a), n_a, n_b)
        except ValueError as exc:
            st.error(str(exc))

    else:
        uploaded = st.file_uploader("CSV: колонка группы + колонка метрики", type=["csv"])
        demo = st.checkbox("Демо-данные", value=uploaded is None)

        df = None
        if demo:
            rng = np.random.default_rng(7)
            n = 8000
            df = pd.DataFrame(
                {
                    "group": np.repeat(["A", "B"], n),
                    "converted": np.concatenate([rng.binomial(1, 0.050, n), rng.binomial(1, 0.057, n)]),
                }
            )
        elif uploaded is not None:
            df = pd.read_csv(uploaded)

        if df is not None:
            st.dataframe(df.head(8), hide_index=True, width="stretch")

            c1, c2 = st.columns(2)
            group_col = c1.selectbox("Группа", df.columns, index=0)
            metric_col = c2.selectbox("Метрика", [c for c in df.columns if c != group_col], index=0)

            groups = df[group_col].dropna().unique().tolist()
            if len(groups) != 2:
                st.error(f"В «{group_col}» {len(groups)} значений, нужно 2: {groups}")
            else:
                g_a, g_b = sorted(map(str, groups))
                vals_a = df.loc[df[group_col].astype(str) == g_a, metric_col].dropna()
                vals_b = df.loc[df[group_col].astype(str) == g_b, metric_col].dropna()
                is_binary = set(pd.unique(df[metric_col].dropna())).issubset({0, 1})

                st.divider()
                st.caption(f"{'z-тест долей' if is_binary else 'Тест Уэлча'} · A = `{g_a}`, B = `{g_b}`")
                try:
                    if is_binary:
                        res = analyze_proportions(
                            len(vals_a), int(vals_a.sum()), len(vals_b), int(vals_b.sum()), alpha_a
                        )
                        render_proportions(res, len(vals_a), len(vals_b))
                    else:
                        render_continuous(analyze_continuous(vals_a.values, vals_b.values, alpha_a))
                except ValueError as exc:
                    st.error(str(exc))


with tab_peek:
    st.caption("Симуляция A/A-теста: эффекта нет, считаем долю ложных срабатываний.")

    c1, c2, c3, c4 = st.columns(4)
    n_peek = c1.number_input("Пользователей на группу", 500, 200_000, 20_000, step=500)
    max_looks = c2.slider("Максимум взглядов", 2, 15, 10)
    p_peek = c3.number_input("Базовая конверсия, %", 0.5, 50.0, 10.0, step=0.5) / 100
    alpha_p = c4.selectbox("α", ALPHAS, index=1, key="alpha_p")
    n_sims = st.select_slider("Симуляций", [500, 1000, 2000, 5000], value=2000)

    if st.button("Запустить", type="primary"):
        with st.spinner("Считаем…"):
            curve = peeking_curve(n_peek, max_looks, p_peek, alpha_p, n_sims)

        looks = [c["n_looks"] for c in curve]
        naive = [c["naive"] for c in curve]
        bonf = [c["bonferroni"] for c in curve]

        fig = go.Figure()
        for name, vals, color in [("Без поправки", naive, SERIES_2), ("Бонферрони", bonf, SERIES_1)]:
            fig.add_trace(
                go.Scatter(
                    x=looks,
                    y=[v * 100 for v in vals],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color, width=2),
                    marker=dict(size=9, line=dict(width=2, color=SURFACE)),
                    hovertemplate="%{x} взглядов → %{y:.1f}%<extra></extra>",
                )
            )
        fig.add_hline(
            y=alpha_p * 100,
            line=dict(color=INK_MUTED, width=1, dash="dot"),
            annotation_text=f"α = {alpha_p:.0%}",
            annotation_position="bottom right",
            annotation_font=dict(color=INK_MUTED, size=11),
        )
        style_axes(fig, "Число взглядов", "Ложные срабатывания, %")
        st.plotly_chart(fig, width="stretch")

        final = curve[-1]
        c1, c2 = st.columns(2)
        for col, label, key in [(c1, f"Без поправки, {max_looks} взглядов", "naive"), (c2, "Бонферрони", "bonferroni")]:
            col.metric(
                label,
                f"{final[key]:.1%}",
                f"{(final[key] - alpha_p) * 100:+.1f} п.п. к α",
                delta_color="inverse",
            )

        st.dataframe(
            pd.DataFrame(
                {
                    "Взглядов": looks,
                    "Без поправки": [f"{v:.2%}" for v in naive],
                    "Бонферрони": [f"{v:.2%}" for v in bonf],
                    "Только финал": [f"{c['final_only']:.2%}" for c in curve],
                }
            ),
            hide_index=True,
            width="stretch",
        )
