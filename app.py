"""Run with: python -m streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from decision_model import ValidationError, evaluate, explain_comparison, validate_inputs
from sample_data import sample_criteria, sample_matrix


st.set_page_config(page_title="Pathway | Educational decision support", page_icon="🎓", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 4rem; padding-bottom: 2rem; max-width: 1450px;}
h1 {letter-spacing: -0.04em;}
[data-testid="stMetric"] {background: #f1f5f9; padding: 1rem; border-radius: .65rem;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
</style>
""", unsafe_allow_html=True)


def load_sample():
    st.session_state.criteria = sample_criteria()
    st.session_state.matrix = sample_matrix()
    st.session_state.result = evaluate(st.session_state.matrix, st.session_state.criteria)
    st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1
    st.session_state.calculation_error = None


def load_blank():
    names = [f"Criterion {i}" for i in range(1, 6)]
    st.session_state.criteria = pd.DataFrame({"Criterion": names, "Type": ["benefit"] * 5, "Weight": [.2] * 5, "Unit": ["value"] * 5})
    st.session_state.matrix = pd.DataFrame(np.nan, index=range(5), columns=names)
    st.session_state.matrix.insert(0, "Alternative", [f"Pathway {i}" for i in range(1, 6)])
    st.session_state.result = None
    st.session_state.calculation_error = None
    st.session_state.editor_version += 1
    st.session_state.section = "Edit data"


if "criteria" not in st.session_state:
    load_sample()

st.caption("COGNITIVE SYSTEMS · ASSIGNMENT 4")
st.title("Choose your next learning pathway")
st.write("Compare educational pathways with your priorities, using WSM and TOPSIS.")
st.caption("Illustrative classroom scenario · IT student exploring applied software/data work · All values are assumed, not market evidence.")

with st.sidebar:
    st.header("Pathway DSS")
    st.write("A transparent recommendation to support the student's decision.")
    st.markdown("**Scenario**  \nBudget: up to **300,000 KZT**  \nStudy time: up to **12 hours/week**")
    st.caption("The five examples meet these limits. Check feasibility, prerequisites and availability before entering other alternatives. The ranking engine does not enforce these scenario limits.")
    if st.button("Reset to sample data", use_container_width=True):
        load_sample()
        st.rerun()
    if st.button("Start blank manual entry", use_container_width=True):
        load_blank()
        st.rerun()
    st.divider()
    st.markdown("**Rating scale**  \n1 = very low fit/quality  \n5 = moderate  \n10 = very high")
    st.caption("Weights sum to 1. Higher scores rank better. A recommendation is not a guarantee of educational or employment outcomes.")

section = st.segmented_control("Workspace", ["Results", "Edit data", "Calculations", "About"], default="Results", key="section")
result = st.session_state.result

if section == "Edit data":
    st.subheader("1. Define criteria and priorities")
    st.write("Edit names, units, types and weights. Add or delete table rows to change criteria. Keep at least five criteria and five alternatives.")
    draft_criteria = st.data_editor(
        st.session_state.criteria, key=f"criteria_editor_{st.session_state.editor_version}",
        num_rows="dynamic", hide_index=True, width="stretch",
        column_config={
            "Criterion": st.column_config.TextColumn(required=True),
            "Type": st.column_config.SelectboxColumn(options=["benefit", "cost"], required=True),
            "Weight": st.column_config.NumberColumn(format="%.2f", required=True, step=.05),
            "Unit": st.column_config.TextColumn(required=True),
        },
    )
    draft_sum = pd.to_numeric(draft_criteria["Weight"], errors="coerce").sum()
    st.caption(f"Draft weight total: {draft_sum:.4f} · Required total: 1.0000")
    if st.button("Apply criteria", type="primary"):
        try:
            criterion_names = draft_criteria["Criterion"].fillna("").astype(str).str.strip().tolist()
            probe = pd.DataFrame(np.ones((5, len(criterion_names))), columns=criterion_names)
            probe.insert(0, "Alternative", [f"Validation {i}" for i in range(5)])
            _, cleaned = validate_inputs(probe, draft_criteria)
            old = st.session_state.matrix
            new = pd.DataFrame({"Alternative": old["Alternative"]})
            for name in cleaned["Criterion"]:
                new[name] = old[name] if name in old.columns else np.nan
            st.session_state.criteria = cleaned
            st.session_state.matrix = new
            st.session_state.result = None
            st.session_state.calculation_error = None
            st.session_state.editor_version += 1
            st.rerun()
        except (ValidationError, ValueError) as exc:
            st.error(str(exc))
    st.caption("Apply criteria before editing the matrix. New or renamed criteria create empty columns to fill. Existing columns with unchanged names keep their data.")
    st.divider()
    st.subheader("2. Enter the decision matrix")
    st.write("Change any value or pathway name. Use the final empty row to add an alternative; select rows to delete them.")
    column_config = {"Alternative": st.column_config.TextColumn(required=True)}
    for row in st.session_state.criteria.to_dict("records"):
        column_config[row["Criterion"]] = st.column_config.NumberColumn(help=f'{row["Unit"]} · {row["Type"]}', required=True)
    draft_matrix = st.data_editor(
        st.session_state.matrix, key=f"matrix_editor_{st.session_state.editor_version}",
        num_rows="dynamic", hide_index=True, width="stretch", column_config=column_config,
    )
    if st.button("Calculate rankings", type="primary"):
        st.session_state.matrix = draft_matrix.copy()
        try:
            st.session_state.result = evaluate(draft_matrix, st.session_state.criteria)
            st.session_state.calculation_error = None
            st.success("Rankings recalculated. Open Results to view the recommendation.")
        except ValidationError as exc:
            st.session_state.result = None
            st.session_state.calculation_error = str(exc)
            st.error(str(exc))
    st.info("Manual example: set the weights to 0.10, 0.10, 0.30, 0.30, 0.20 in the displayed order; Apply criteria, then Calculate rankings. Compare the winners in Results.")
    st.caption("Only applied criteria and the last successfully calculated matrix appear in Results. Finish edits by clicking the buttons above.")

elif section == "Results":
    if result is None:
        st.info("Open Edit data, fill all cells, and click Calculate rankings.")
        if st.session_state.calculation_error:
            st.error(st.session_state.calculation_error)
    else:
        table = result.comparison
        wsm_winners = table.loc[table["WSM rank"].eq(1), "Alternative"].tolist()
        topsis_winners = table.loc[table["TOPSIS rank"].eq(1), "Alternative"].tolist()
        col1, col2, col3 = st.columns([2, 2, 1])
        col1.metric("WSM recommendation", ", ".join(wsm_winners), f"Score {table['WSM score'].max():.4f}", delta_color="off")
        col2.metric("TOPSIS recommendation", ", ".join(topsis_winners), f"Closeness {table['TOPSIS score'].max():.4f}", delta_color="off")
        col3.metric("Compared", f"{len(result.matrix)} pathways", f"{len(result.criteria)} criteria", delta_color="off")
        st.write(explain_comparison(result))
        left, right = st.columns(2)
        score_data = table.melt(id_vars="Alternative", value_vars=["WSM score", "TOPSIS score"], var_name="Method", value_name="Score")
        score_data["Method"] = score_data["Method"].str.replace(" score", "")
        chart = px.bar(score_data, x="Score", y="Alternative", color="Method", barmode="group", orientation="h", color_discrete_map={"WSM": "#167d8d", "TOPSIS": "#e49b40"}, title="Scores by method")
        chart.update_layout(height=390, margin=dict(l=0, r=10, t=50, b=0), legend=dict(orientation="h", y=1.09, x=0), yaxis=dict(autorange="reversed", title=None), xaxis=dict(range=[0, 1]))
        left.plotly_chart(chart, width="stretch")
        heatmap = px.imshow(result.wsm_normalized, x=result.wsm_normalized.columns, y=result.wsm_normalized.index, zmin=0, zmax=1, color_continuous_scale=["#f0f4f6", "#167d8d"], text_auto=".2f", aspect="auto", title="Normalized performance · higher is better")
        heatmap.update_layout(height=390, margin=dict(l=0, r=5, t=50, b=0), xaxis_title=None, yaxis_title=None, coloraxis_showscale=False)
        right.plotly_chart(heatmap, width="stretch")
        st.caption("WSM uses the normalized performance shown at right. TOPSIS uses its own vector normalization. Compare ranks within each method; the two score scales have different meanings.")
        st.subheader("Ranking comparison")
        visible = table[["Alternative", "WSM score", "WSM rank", "TOPSIS score", "TOPSIS rank", "Rank difference"]].sort_values("WSM rank", kind="stable")
        st.dataframe(visible.style.format({"WSM score": "{:.6f}", "TOPSIS score": "{:.6f}"}), hide_index=True, width="stretch")
        st.caption("Rank difference = WSM rank minus TOPSIS rank. Ties share the same rank; zero means agreement for that pathway.")
        st.download_button("Download results CSV", data=table.to_csv(index=False), file_name="pathway_results.csv", mime="text/csv")

elif section == "Calculations":
    if result is None:
        st.info("Calculate valid inputs in Edit data first.")
    else:
        st.subheader("Follow the calculation")
        st.caption("All tables use the last successfully calculated inputs. Displayed values are rounded; calculations use full precision.")
        with st.expander("Input matrix and criterion weights", expanded=True):
            st.dataframe(result.matrix, hide_index=True, width="stretch")
            st.dataframe(result.criteria, hide_index=True, width="stretch")
        with st.expander("WSM: normalization, contributions and scores"):
            st.latex(r"r_{ij}=\frac{x_{ij}-\min_i x_{ij}}{\max_i x_{ij}-\min_i x_{ij}}\quad\text{(benefit)},\qquad r_{ij}=\frac{\max_i x_{ij}-x_{ij}}{\max_i x_{ij}-\min_i x_{ij}}\quad\text{(cost)}")
            st.latex(r"S_i=\sum_j w_jr_{ij}")
            st.write("A constant column receives 1 for every alternative, adding the same contribution to all scores and not affecting the ranking.")
            st.write("Normalized matrix")
            st.dataframe(result.wsm_normalized.style.format("{:.6f}"), width="stretch")
            st.write("Weighted contributions (sum each row for its WSM score)")
            st.dataframe(result.wsm_contributions.style.format("{:.6f}"), width="stretch")
            st.dataframe(result.comparison[["Alternative", "WSM score", "WSM rank"]], hide_index=True, width="stretch")
        with st.expander("TOPSIS: vector normalization and weighted matrix"):
            st.latex(r"r_{ij}=\frac{x_{ij}}{\sqrt{\sum_i x_{ij}^{2}}},\qquad v_{ij}=w_jr_{ij}")
            st.write("An all-zero column becomes zero; it contributes no distance. Cost directions are handled when selecting the ideals.")
            st.dataframe(result.topsis_normalized.style.format("{:.6f}"), width="stretch")
            st.dataframe(result.topsis_weighted.style.format("{:.6f}"), width="stretch")
        with st.expander("TOPSIS: ideal best/worst, distances and closeness"):
            st.write("Ideal best = maximum for benefits, minimum for costs. Ideal worst reverses those choices.")
            st.dataframe(result.ideals.style.format("{:.6f}"), width="stretch")
            st.latex(r"D_i^{+}=\sqrt{\sum_j(v_{ij}-v_j^{+})^2},\quad D_i^{-}=\sqrt{\sum_j(v_{ij}-v_j^{-})^2},\quad C_i=\frac{D_i^{-}}{D_i^{+}+D_i^{-}}")
            st.dataframe(result.comparison[["Alternative", "Distance to best", "Distance to worst", "TOPSIS score", "TOPSIS rank"]].style.format({"Distance to best": "{:.6f}", "Distance to worst": "{:.6f}", "TOPSIS score": "{:.6f}"}), hide_index=True, width="stretch")
            st.caption("If both distances are zero, every alternative is tied on the weighted criteria; the score is set to 0.5. Ranks use scores rounded to 12 decimals to avoid floating-point tie artifacts.")

elif section == "About":
    st.subheader("What decision does this system support?")
    st.write("An IT student chooses one additional 12-week learning pathway that fits their interests, career direction, time and budget. Each pathway is a hypothetical package of guided study and a practical project. Basic Python/math knowledge, entry-level suitability and availability are assumed satisfied.")
    st.write("The decision is semi-structured: numerical calculations are clear, while personal goals, ratings and trade-offs require judgment. This is a model-driven decision support system. The student is the decision maker; an adviser can help validate the inputs.")
    st.markdown("**Criterion meanings**\n\n- Total cost: the complete 12-week package cost; lower is preferable.\n- Weekly workload: time required each week; lower is easier to fit around existing studies.\n- Interest fit: personal enthusiasm for the pathway, assessed by the student.\n- Career fit: alignment with the student's applied software/data goal.\n- Project quality: expected depth and usefulness of the pathway's practical portfolio project.")
    st.write("All values and weights are assumptions for this assignment. Rating anchors: 1 = very low, 5 = moderate, 10 = very high. Decimal values are allowed. Actual recommendations need real programme data and careful student/adviser ratings.")
    st.markdown("**Possible consequences**\n\nA suitable choice can improve engagement and relevant practical skills. A poor fit can waste money or study time. Both methods compensate weaknesses with strengths, and rankings can change when weights or alternatives change.")
    st.write("WSM uses min-max normalization; TOPSIS uses vector normalization. They therefore express preferences differently. Neither method proves that one pathway is objectively best, and a change of winner is a useful prompt to revisit priorities.")
