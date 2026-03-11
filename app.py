import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ── configuración ──────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Tabular ML — Readmisión Hospitalaria",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── paleta ─────────────────────────────────────────────────
AZUL      = "#1A6B9A"
CELESTE   = "#4DACD6"
GRIS      = "#B0BEC5"
ROJO      = "#C0392B"
FONDO     = "#0E1117"

# ── CSS personalizado ──────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    h1 { color: #4DACD6; font-size: 2.2rem; font-weight: 700; }
    h2 { color: #4DACD6; font-size: 1.4rem; font-weight: 600; border-bottom: 1px solid #1A6B9A; padding-bottom: 4px; }
    h3 { color: #B0BEC5; font-size: 1.1rem; }
    .narrative {
        background-color: #111827;
        border-left: 3px solid #1A6B9A;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        color: #B0BEC5;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .metric-label { color: #4DACD6; font-size: 0.8rem; }
    .sidebar .sidebar-content { background-color: #111827; }
</style>
""", unsafe_allow_html=True)

# ── datos ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/features.parquet")

@st.cache_resource
def load_model():
    with open("models/xgb_readmission.pkl", "rb") as f:
        return pickle.load(f)

df    = load_data()
model = load_model()

# ── sidebar ────────────────────────────────────────────────
st.sidebar.markdown("## Clinical Tabular ML")
st.sidebar.markdown("**Readmisión Hospitalaria — 30 días**")
st.sidebar.markdown("---")

panel = st.sidebar.radio(
    "Navegacion",
    ["Introduccion",
     "La cohorte",
     "Patrones clinicos",
     "El modelo",
     "Interpretabilidad"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Dataset: Diabetes 130-US Hospitals (UCI)  \n"
    "Modelo: XGBoost + SHAP  \n"
    "Santiago Vega · 2025–2026"
)

# ══════════════════════════════════════════════════════════
# PANEL 1 — INTRODUCCION
# ══════════════════════════════════════════════════════════
if panel == "Introduccion":

    st.title("Readmision Hospitalaria a 30 Dias")
    st.markdown("---")

    st.markdown("""<div class="narrative">
    La readmision dentro de los 30 dias del alta es uno de los indicadores mas criticos
    de calidad asistencial. En pacientes diabeticos, cuya gestion involucra multiples
    comorbilidades y polifarmacia, la tasa de readmision temprana supera a la poblacion general.
    Este proyecto construye un modelo predictivo usando exclusivamente variables clinicas y
    administrativas disponibles al momento del alta.
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Encuentros analizados", "99,343")
    col2.metric("Hospitales cubiertos", "130")
    col3.metric("Periodo", "1999 – 2008")

    st.markdown("---")
    st.markdown("## Pregunta central")
    st.markdown("""<div class="narrative">
    Es posible identificar, al momento del alta hospitalaria, que pacientes diabeticos
    tienen mayor riesgo de ser readmitidos dentro de los 30 dias, usando exclusivamente
    variables clinicas y administrativas disponibles en el registro?
    </div>""", unsafe_allow_html=True)

    st.markdown("## Decisiones metodologicas clave")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("#### Split sin leakage")
        st.markdown("GroupShuffleSplit por paciente — ningun paciente comparte encuentros entre train y test.")

    with col_b:
        st.markdown("#### Desbalance de clases")
        st.markdown("Solo el 11.4% de los encuentros son readmisiones. Se usa scale_pos_weight en XGBoost.")

    with col_c:
        st.markdown("#### Umbral ajustado")
        st.markdown("El umbral por defecto (0.5) no es optimo. Se elige 0.42 por F2-score priorizando recall.")

    st.markdown("---")
    st.markdown("#### Navegacion sugerida")
    st.markdown("La cohorte  →  Patrones clinicos  →  El modelo  →  Interpretabilidad")

# ══════════════════════════════════════════════════════════
# PANEL 2 — LA COHORTE
# ══════════════════════════════════════════════════════════
elif panel == "La cohorte":

    st.title("La cohorte")
    st.markdown("""<div class="narrative">
    Antes de modelar, es necesario entender quienes son los pacientes.
    La distribucion demografica y clinica define el contexto del problema
    y condiciona las decisiones de preprocesamiento.
    </div>""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total encuentros", f"{len(df):,}")
    col2.metric("Readmision menos 30d", f"{df['readmitted_binary'].sum():,}")
    col3.metric("Tasa de readmision", f"{df['readmitted_binary'].mean()*100:.1f}%")
    col4.metric("Variables", f"{df.shape[1]}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("## Distribucion por edad")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0E1117")
        ax.set_facecolor("#0E1117")
        ax.hist(df["age_numeric"], bins=20, color=CELESTE, edgecolor="#0E1117")
        ax.set_xlabel("Edad", color=GRIS)
        ax.set_ylabel("Frecuencia", color=GRIS)
        ax.tick_params(colors=GRIS)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1A6B9A")
        st.pyplot(fig)
        plt.close()
        st.caption("La cohorte es predominantemente adulta mayor. La edad viene en rangos de 10 años — se convirtio al punto medio para uso en modelos.")

    with col_b:
        st.markdown("## Tiempo de internacion")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0E1117")
        ax.set_facecolor("#0E1117")
        ax.hist(df["time_in_hospital"], bins=14, color=AZUL, edgecolor="#0E1117")
        ax.set_xlabel("Dias", color=GRIS)
        ax.set_ylabel("Frecuencia", color=GRIS)
        ax.tick_params(colors=GRIS)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1A6B9A")
        st.pyplot(fig)
        plt.close()
        st.caption("La mayoria de las internaciones duran entre 2 y 5 dias. Las estadias largas son poco frecuentes pero concentran mayor complejidad clinica.")

    st.markdown("---")

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("## Polifarmacia")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0E1117")
        ax.set_facecolor("#0E1117")
        ax.hist(df["num_medications"], bins=20, color=CELESTE, edgecolor="#0E1117")
        ax.set_xlabel("N medicamentos", color=GRIS)
        ax.set_ylabel("Frecuencia", color=GRIS)
        ax.tick_params(colors=GRIS)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1A6B9A")
        st.pyplot(fig)
        plt.close()
        st.caption("Alta carga de medicacion es caracteristica de este tipo de paciente. La polifarmacia es un proxy de complejidad clinica.")

    with col_d:
        st.markdown("## Distribucion del target")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0E1117")
        ax.set_facecolor("#0E1117")
        counts = df["readmitted_binary"].value_counts()
        ax.bar(["No readmitido / mas 30d", "Readmitido menos 30d"],
               counts.values,
               color=[AZUL, ROJO], edgecolor="#0E1117")
        ax.set_ylabel("Frecuencia", color=GRIS)
        ax.tick_params(colors=GRIS)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1A6B9A")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 200, f"{v:,}", ha="center", color=GRIS, fontweight="bold")
        st.pyplot(fig)
        plt.close()
        st.caption("Desbalance severo: solo el 11.4% son readmisiones en menos de 30 dias. Accuracy es irrelevante como metrica en este contexto.")

# ══════════════════════════════════════════════════════════
# PANEL 3 — PATRONES CLINICOS
# ══════════════════════════════════════════════════════════
elif panel == "Patrones clinicos":

    st.title("Patrones clinicos")
    st.markdown("""<div class="narrative">
    El analisis exploratorio revela patrones que guian el feature engineering
    y anticipan que variables tendran señal en el modelo.
    Estos hallazgos son descriptivos — no causales.
    </div>""", unsafe_allow_html=True)

    st.markdown("## Utilizacion hospitalaria previa")
    st.markdown("""<div class="narrative">
    La historia de internaciones previas es el patron mas consistente en los datos.
    Pacientes con mayor utilizacion del sistema tienen tasas de readmision
    significativamente mas altas.
    </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        prior = df.groupby("number_inpatient")["readmitted_binary"].mean() * 100
        prior = prior[prior.index <= 10]
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0E1117")
        ax.set_facecolor("#0E1117")
        ax.bar(prior.index, prior.values, color=CELESTE, edgecolor="#0E1117")
        ax.set_xlabel("N internaciones previas", color=GRIS)
        ax.set_ylabel("Tasa de readmision (%)", color=GRIS)
        ax.tick_params(colors=GRIS)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1A6B9A")
        st.pyplot(fig)
        plt.close()
        st.caption("A mayor numero de internaciones previas, mayor tasa de readmision. El patron es monotono y consistente.")

    with col_b:
        vis = df.copy()
        vis["visits_group"] = pd.cut(vis["total_prior_visits"],
                                      bins=[-1, 0, 2, 5, 10, 100],
                                      labels=["0", "1-2", "3-5", "6-10", "11+"])
        vg = vis.groupby("visits_group", observed=True)["readmitted_binary"].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0E1117")
        ax.set_facecolor("#0E1117")
        ax.bar(vg.index, vg.values, color=AZUL, edgecolor="#0E1117")
        ax.set_xlabel("Visitas previas totales", color=GRIS)
        ax.set_ylabel("Tasa de readmision (%)", color=GRIS)
        ax.tick_params(colors=GRIS)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1A6B9A")
        st.pyplot(fig)
        plt.close()
        st.caption("total_prior_visits agrega outpatient + emergency + inpatient. Feature construido en el pipeline.")

    st.markdown("---")
    st.markdown("## Diagnosticos y riesgo")
    st.markdown("""<div class="narrative">
    Los diagnosticos primarios fueron agrupados en 9 categorias clinicas estandar (CCS/ICD-9).
    No se aplico One-Hot sobre las 900 categorias originales.
    </div>""", unsafe_allow_html=True)

    col_c, col_d = st.columns(2)

    with col_c:
        diag = df.groupby("diag_1_cat")["readmitted_binary"].mean() * 100
        diag = diag.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0E1117")
        ax.set_facecolor("#0E1117")
        ax.barh(diag.index, diag.values, color=CELESTE, edgecolor="#0E1117")
        ax.set_xlabel("Tasa de readmision (%)", color=GRIS)
        ax.tick_params(colors=GRIS)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1A6B9A")
        for i, v in enumerate(diag.values):
            ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=8, color=GRIS)
        st.pyplot(fig)
        plt.close()
        st.caption("Riesgo por categoria diagnostica primaria.")

    with col_d:
        vol = df["diag_1_cat"].value_counts().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0E1117")
        ax.set_facecolor("#0E1117")
        ax.barh(vol.index, vol.values, color=AZUL, edgecolor="#0E1117")
        ax.set_xlabel("N encuentros", color=GRIS)
        ax.tick_params(colors=GRIS)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1A6B9A")
        st.pyplot(fig)
        plt.close()
        st.caption("Volumen por categoria. Circulatorio domina en frecuencia.")

    st.markdown("---")
    st.markdown("## Medicacion")
    st.markdown("""<div class="narrative">
    Insulina muestra el mayor rango de variacion en tasa de readmision segun el tipo de cambio.
    Metformina tiene señal mas debil pero se conserva individualmente por plausibilidad clinica.
    El resto de medicaciones se agrega en n_meds_changed.
    </div>""", unsafe_allow_html=True)

    col_e, col_f = st.columns(2)

    with col_e:
        ins = df.groupby("insulin")["readmitted_binary"].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0E1117")
        ax.set_facecolor("#0E1117")
        ax.bar(ins.index, ins.values, color=CELESTE, edgecolor="#0E1117")
        ax.set_xlabel("Categoria de insulina", color=GRIS)
        ax.set_ylabel("Tasa de readmision (%)", color=GRIS)
        ax.tick_params(colors=GRIS)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1A6B9A")
        for i, v in enumerate(ins.values):
            ax.text(i, v + 0.1, f"{v:.1f}%", ha="center", color=GRIS, fontsize=9)
        st.pyplot(fig)
        plt.close()
        st.caption("Down (reduccion de dosis al alta) tiene la mayor tasa. Posible señal de descontrol glucemico.")

    with col_f:
        met = df.groupby("metformin")["readmitted_binary"].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0E1117")
        ax.set_facecolor("#0E1117")
        ax.bar(met.index, met.values, color=AZUL, edgecolor="#0E1117")
        ax.set_xlabel("Categoria de metformina", color=GRIS)
        ax.set_ylabel("Tasa de readmision (%)", color=GRIS)
        ax.tick_params(colors=GRIS)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1A6B9A")
        for i, v in enumerate(met.values):
            ax.text(i, v + 0.1, f"{v:.1f}%", ha="center", color=GRIS, fontsize=9)
        st.pyplot(fig)
        plt.close()
        st.caption("Señal inversa debil: aumento de metformina asociado a menor readmision. Diferencia de ~4pp.")

# ══════════════════════════════════════════════════════════
# PANEL 4 — EL MODELO
# ══════════════════════════════════════════════════════════
elif panel == "El modelo":

    st.title("El modelo")
    st.markdown("""<div class="narrative">
    Se entrenaron tres modelos con estrategia de desbalance explicita.
    XGBoost es el modelo principal. El umbral de decision se eligio
    por F2-score priorizando recall sobre precision — en readmision hospitalaria,
    no detectar a alguien que vuelve es mas costoso que generar una alerta innecesaria.
    </div>""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", "0.668")
    col2.metric("PR-AUC", "0.224")
    col3.metric("Recall", "0.738")
    col4.metric("Umbral optimo", "0.42")

    st.markdown("---")

    st.markdown("## Comparacion de modelos")
    st.markdown("""<div class="narrative">
    Logistic Regression como baseline interpretable.
    Random Forest como ensemble sin boosting.
    XGBoost como modelo principal con scale_pos_weight nativo.
    </div>""", unsafe_allow_html=True)
    st.image("reports/figures/roc_pr_curves.png", use_container_width=True)

    st.markdown("---")
    st.markdown("## Threshold tuning")
    st.markdown("""<div class="narrative">
    El umbral 0.42 maximiza F2-score. Con este umbral el modelo detecta
    el 73.8% de las readmisiones reales a costa de una tasa alta de falsos positivos.
    Este trade-off es una decision explicita, no una limitacion del modelo.
    </div>""", unsafe_allow_html=True)
    st.image("reports/figures/threshold_tuning.png", use_container_width=True)

    st.markdown("---")
    st.markdown("## Confusion Matrix (umbral=0.42)")
    st.image("reports/figures/confusion_matrix.png", use_container_width=True)

    st.markdown("---")
    st.markdown("## Calibracion")
    st.markdown("""<div class="narrative">
    El modelo esta subcalibrado: las probabilidades predichas son sistematicamente
    mas altas que la proporcion real de positivos. Causa esperada con scale_pos_weight alto.
    El modelo es util para rankear pacientes por riesgo relativo,
    no para estimar probabilidad absoluta de readmision.
    </div>""", unsafe_allow_html=True)
    st.image("reports/figures/calibration_curve.png", use_container_width=True)

# ══════════════════════════════════════════════════════════
# PANEL 5 — INTERPRETABILIDAD
# ══════════════════════════════════════════════════════════
elif panel == "Interpretabilidad":

    st.title("Interpretabilidad")
    st.markdown("""<div class="narrative">
    SHAP permite entender que variables explican cada prediccion.
    Las variables con mayor impacto no son necesariamente intervenibles —
    SHAP muestra asociacion estadistica, no causalidad.
    </div>""", unsafe_allow_html=True)

    st.markdown("## Importancia global — Top 15 features")
    st.markdown("""<div class="narrative">
    La utilizacion hospitalaria previa domina el modelo.
    number_inpatient es el predictor mas fuerte por amplio margen.
    Las variables construidas en el feature engineering
    (total_prior_visits, lab_intensity) aparecen en el top 15.
    </div>""", unsafe_allow_html=True)
    st.image("reports/figures/shap_beeswarm.png", use_container_width=True)
    st.markdown("**Rosa** = valor alto de la feature · **Azul** = valor bajo")

    st.markdown("---")
    st.markdown("## Perfiles de paciente — Waterfall plots")
    st.markdown("""<div class="narrative">
    Cada waterfall muestra como el modelo llega a su score para un paciente especifico.
    Las barras rojas suman riesgo, las azules lo reducen.
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "Alto riesgo (TP)",
        "Bajo riesgo (TN)",
        "Falso Negativo (FN)"
    ])

    with tab1:
        st.image("reports/figures/shap_waterfall_perfil_1.png", use_container_width=True)
        st.caption("discharge_disposition_id es el driver principal (+0.31). Detectado correctamente a pesar de pocas internaciones previas.")

    with tab2:
        st.image("reports/figures/shap_waterfall_perfil_2.png", use_container_width=True)
        st.caption("number_inpatient y discharge_disposition_id empujan hacia no readmision. insulin_Down aparece con señal positiva (+0.07).")

    with tab3:
        st.image("reports/figures/shap_waterfall_perfil_3.png", use_container_width=True)
        st.caption("Paciente joven con poco historial. El modelo no lo detecta porque el perfil parece protector. Limite explicito del modelo.")

    st.markdown("---")
    st.info("SHAP muestra asociacion estadistica, no causalidad. Las variables con mayor impacto no son necesariamente intervenibles.")