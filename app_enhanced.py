# app_enhanced.py
# Simulatore decisionale AI vs Clinico - Melanoma (VERSIONE AVANZATA CON GRAFICI KAPLAN-MEIER E ANALISI COSTO-EFFICACIA)
# Presidi: Busto Arsizio, Gallarate, Saronno
# Grafici: KM vere, CE plane, distribuzioni costi, tossicità, ICER per sottogruppi

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# ========================
# COSTANTI E PARAMETRI
# ========================

PRESIDI = ["Busto Arsizio", "Gallarate", "Saronno"]
SETTINGS = ["Adiuvante", "Metastatico"]

# Utilità QALY (health states)
UTILITY_NO_RECURRENCE = 0.88
UTILITY_STABLE = 0.80
UTILITY_PROGRESSIVE = 0.60
UTILITY_TOX_PENALTY = 0.10

# Soglia WTP
WTP_THRESHOLD = 30000  # €/QALY

# ========================
# FUNZIONI DI SIMULAZIONE (come prima)
# ========================

def simulate_patients(n_patients: int, seed: int = 42) -> pd.DataFrame:
    """Genera coorte sintetica di pazienti con melanoma."""
    np.random.seed(seed)
    df = pd.DataFrame()
    df["patient_id"] = np.arange(1, n_patients + 1)
    df["presidio"] = np.random.choice(PRESIDI, size=n_patients, p=[0.4, 0.3, 0.3])
    df["setting"] = np.random.choice(SETTINGS, size=n_patients, p=[0.5, 0.5])
    df["age"] = np.random.normal(loc=65, scale=10, size=n_patients).astype(int)
    df["age"] = df["age"].clip(30, 90)
    df["sex"] = np.random.choice(["M", "F"], size=n_patients, p=[0.6, 0.4])
    df["ecog"] = np.random.choice([0, 1, 2], size=n_patients, p=[0.5, 0.35, 0.15])
    df["braf_status"] = np.random.choice(["Mutato", "Wild-type"], size=n_patients, p=[0.45, 0.55])
    df["ldh"] = np.random.choice(["Normale", "Elevato"], size=n_patients, p=[0.7, 0.3])
    df["stage"] = df["setting"].apply(
        lambda x: np.random.choice(
            ["IIIA", "IIIB", "IIIC", "IV_resecato"] if x == "Adiuvante" else ["M1a", "M1b", "M1c"],
            p=[0.25, 0.35, 0.3, 0.1] if x == "Adiuvante" else [0.3, 0.3, 0.4]
        )
    )
    df["met_sites"] = np.where(
        df["setting"] == "Metastatico",
        np.random.choice(
            ["Linfonodale", "Viscerale_non_cerebrale", "Cerebrale"],
            size=n_patients, p=[0.3, 0.5, 0.2]
        ),
        "N/A"
    )
    return df


def ai_recommendation(row):
    """Algoritmo AI basato su linee guida."""
    if row["setting"] == "Adiuvante":
        if row["braf_status"] == "Mutato":
            return np.random.choice(["PD-1_adiuvante", "BRAFMEK_adiuvante"], p=[0.6, 0.4])
        else:
            return "PD-1_adiuvante"
    if row["braf_status"] == "Mutato":
        if row["ecog"] <= 1 and row["ldh"] == "Normale":
            return np.random.choice(["IO_combo", "BRAFMEK"], p=[0.6, 0.4])
        else:
            return "BRAFMEK"
    else:
        if row["ecog"] == 0 and row["ldh"] == "Normale":
            return "IO_combo"
        elif row["ecog"] <= 1:
            return "PD-1_mono"
        else:
            return "Best_supportive_or_chemo"


def clinician_choice(row):
    """Decisione clinico con override in base a presidio e fragilità."""
    ai = row["ai_treatment"]
    fragile = (row["age"] > 75) or (row["ecog"] >= 2) or (row["ldh"] == "Elevato")

    if row["presidio"] == "Busto Arsizio":
        if fragile and "IO_combo" in ai:
            return "PD-1_mono"
        else:
            return ai
    if row["presidio"] == "Gallarate":
        if "IO_combo" in ai:
            if fragile:
                return "PD-1_mono"
            else:
                return np.random.choice(["IO_combo", "PD-1_mono"], p=[0.7, 0.3])
        if ai == "BRAFMEK" and fragile:
            return "BRAFMEK"
        return ai
    if row["presidio"] == "Saronno":
        if "IO_combo" in ai:
            return np.random.choice(["PD-1_mono", "Best_supportive_or_chemo"], p=[0.7, 0.3])
        if ai == "BRAFMEK" and fragile:
            return np.random.choice(["BRAFMEK", "Best_supportive_or_chemo"], p=[0.7, 0.3])
        return ai
    return ai


def simulate_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Simula outcome clinici (RFS/PFS/OS, tossicità)."""
    rfs = []
    pfs = []
    os_ = []
    tox = []

    for _, row in df.iterrows():
        trt = row["clinician_treatment"]
        if row["setting"] == "Adiuvante":
            if trt == "PD-1_adiuvante":
                mean_rfs = 60
                tox_prob = 0.20
            elif trt == "BRAFMEK_adiuvante":
                mean_rfs = 54
                tox_prob = 0.30
            else:
                mean_rfs = 30
                tox_prob = 0.05
            rfs_time = np.random.exponential(scale=mean_rfs)
            rfs.append(rfs_time)
            pfs.append(np.nan)
            os_tail = np.random.exponential(scale=40)
            os_.append(rfs_time + os_tail)
            tox.append(np.random.rand() < tox_prob)
        else:
            if trt == "IO_combo":
                mean_pfs = 18
                mean_os = 60
                tox_prob = 0.40
            elif trt in ["PD-1_mono", "PD-1_adiuvante"]:
                mean_pfs = 12
                mean_os = 40
                tox_prob = 0.20
            elif trt == "BRAFMEK":
                mean_pfs = 10
                mean_os = 32
                tox_prob = 0.25
            else:
                mean_pfs = 6
                mean_os = 14
                tox_prob = 0.10
            pfs_time = np.random.exponential(scale=mean_pfs)
            extra = np.random.exponential(scale=max(mean_os - mean_pfs, 2))
            os_time = pfs_time + extra
            rfs.append(np.nan)
            pfs.append(pfs_time)
            os_.append(os_time)
            tox.append(np.random.rand() < tox_prob)

    df["rfs_months"] = rfs
    df["pfs_months"] = pfs
    df["os_months"] = os_
    df["tox_g3_4"] = tox
    return df


def add_costs_and_qalys(df: pd.DataFrame, horizon_years: float = 5.0) -> pd.DataFrame:
    """Calcola costi e QALY."""
    cost_drug_month = {
        "PD-1_adiuvante": 7000,
        "BRAFMEK_adiuvante": 8000,
        "IO_combo": 12000,
        "PD-1_mono": 7000,
        "BRAFMEK": 9000,
        "Best_supportive_or_chemo": 3000,
    }
    cost_tox_event = 5000
    durations = []
    total_costs = []
    qalys = []
    horizon_months = horizon_years * 12

    for _, row in df.iterrows():
        trt = row["clinician_treatment"]
        monthly_cost = cost_drug_month.get(trt, 0)
        tox_penalty = UTILITY_TOX_PENALTY if row["tox_g3_4"] else 0.0

        if row["setting"] == "Adiuvante":
            dur = min(row["rfs_months"], 12) if pd.notnull(row["rfs_months"]) else 12
            durations.append(dur)
            cost = dur * monthly_cost
            if row["tox_g3_4"]:
                cost += cost_tox_event
            total_costs.append(cost)
            time_no_rec = min(row["rfs_months"], horizon_months)
            qaly = (time_no_rec / 12.0) * (UTILITY_NO_RECURRENCE - tox_penalty)
            if row["rfs_months"] < horizon_months:
                time_after = horizon_months - row["rfs_months"]
                qaly += (time_after / 12.0) * (UTILITY_PROGRESSIVE - tox_penalty * 0.5)
            qalys.append(qaly)
        else:
            dur = min(row["pfs_months"], 24) if pd.notnull(row["pfs_months"]) else 6
            durations.append(dur)
            cost = dur * monthly_cost
            if row["tox_g3_4"]:
                cost += cost_tox_event
            total_costs.append(cost)
            time_pfs = min(row["pfs_months"], horizon_months) if pd.notnull(row["pfs_months"]) else 0
            qaly = (time_pfs / 12.0) * (UTILITY_STABLE - tox_penalty)
            if pd.notnull(row["os_months"]):
                time_total = min(row["os_months"], horizon_months)
                time_post = max(time_total - time_pfs, 0)
                qaly += (time_post / 12.0) * (UTILITY_PROGRESSIVE - tox_penalty * 0.5)
            qalys.append(qaly)

    df["treatment_duration_months"] = durations
    df["treatment_cost_euro"] = total_costs
    df["qaly"] = qalys
    return df


def add_concordance_and_reason(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola concordanza e motivazioni."""
    df["concordant"] = df["ai_treatment"] == df["clinician_treatment"]
    reasons = []
    for _, row in df.iterrows():
        if row["concordant"]:
            reasons.append("Aderenza a AI/linee guida")
        else:
            fragile = (row["age"] > 75) or (row["ecog"] >= 2) or (row["ldh"] == "Elevato")
            if fragile:
                reasons.append("Fragilità/comorbidità → de-intensificazione")
            elif row["presidio"] == "Saronno":
                reasons.append("Limitazioni risorse/accesso farmaci innovativi")
            else:
                reasons.append("Preferenze del paziente/valutazione clinica individuale")
    df["reason_clinician_diff"] = reasons
    return df


# ========================
# ANALISI: KAPPA, ICER
# ========================

def compute_kappa(df: pd.DataFrame) -> float:
    return cohen_kappa_score(df["ai_treatment"], df["clinician_treatment"])


def scenario_ai_vs_clinical(df: pd.DataFrame, horizon_years: float) -> pd.DataFrame:
    clin = df.copy()
    clin["scenario"] = "Clinico"
    ai = df.copy()
    ai["clinician_treatment"] = ai["ai_treatment"]
    ai = simulate_outcomes(ai)
    ai = add_costs_and_qalys(ai, horizon_years)
    ai["scenario"] = "AI"
    combo = pd.concat([clin, ai], ignore_index=True)
    return combo


def compute_icer(combo: pd.DataFrame) -> pd.DataFrame:
    """Calcola ICER per scenario AI vs Clinico."""
    results = []
    
    # GLOBALE
    ai = combo[combo["scenario"] == "AI"]
    cl = combo[combo["scenario"] == "Clinico"]
    if not ai.empty and not cl.empty:
        mean_cost_ai = ai["treatment_cost_euro"].mean()
        mean_cost_cl = cl["treatment_cost_euro"].mean()
        mean_qaly_ai = ai["qaly"].mean()
        mean_qaly_cl = cl["qaly"].mean()
        delta_cost = mean_cost_ai - mean_cost_cl
        delta_qaly = mean_qaly_ai - mean_qaly_cl
        icer = np.nan if abs(delta_qaly) < 1e-6 else delta_cost / delta_qaly
        results.append({
            "gruppo": "TOTALE",
            "mean_cost_AI": mean_cost_ai,
            "mean_cost_Clinico": mean_cost_cl,
            "mean_qaly_AI": mean_qaly_ai,
            "mean_qaly_Clinico": mean_qaly_cl,
            "delta_cost": delta_cost,
            "delta_qaly": delta_qaly,
            "ICER_euro_per_QALY": icer,
        })

    # PER PRESIDIO
    for presidio in combo["presidio"].unique():
        df_sub = combo[combo["presidio"] == presidio]
        ai = df_sub[df_sub["scenario"] == "AI"]
        cl = df_sub[df_sub["scenario"] == "Clinico"]
        if not ai.empty and not cl.empty:
            mean_cost_ai = ai["treatment_cost_euro"].mean()
            mean_cost_cl = cl["treatment_cost_euro"].mean()
            mean_qaly_ai = ai["qaly"].mean()
            mean_qaly_cl = cl["qaly"].mean()
            delta_cost = mean_cost_ai - mean_cost_cl
            delta_qaly = mean_qaly_ai - mean_qaly_cl
            icer = np.nan if abs(delta_qaly) < 1e-6 else delta_cost / delta_qaly
            results.append({
                "gruppo": f"Presidio: {presidio}",
                "mean_cost_AI": mean_cost_ai,
                "mean_cost_Clinico": mean_cost_cl,
                "mean_qaly_AI": mean_qaly_ai,
                "mean_qaly_Clinico": mean_qaly_cl,
                "delta_cost": delta_cost,
                "delta_qaly": delta_qaly,
                "ICER_euro_per_QALY": icer,
            })

    # PER SETTING
    for setting in combo["setting"].unique():
        df_sub = combo[combo["setting"] == setting]
        ai = df_sub[df_sub["scenario"] == "AI"]
        cl = df_sub[df_sub["scenario"] == "Clinico"]
        if not ai.empty and not cl.empty:
            mean_cost_ai = ai["treatment_cost_euro"].mean()
            mean_cost_cl = cl["treatment_cost_euro"].mean()
            mean_qaly_ai = ai["qaly"].mean()
            mean_qaly_cl = cl["qaly"].mean()
            delta_cost = mean_cost_ai - mean_cost_cl
            delta_qaly = mean_qaly_ai - mean_qaly_cl
            icer = np.nan if abs(delta_qaly) < 1e-6 else delta_cost / delta_qaly
            results.append({
                "gruppo": f"Setting: {setting}",
                "mean_cost_AI": mean_cost_ai,
                "mean_cost_Clinico": mean_cost_cl,
                "mean_qaly_AI": mean_qaly_ai,
                "mean_qaly_Clinico": mean_qaly_cl,
                "delta_cost": delta_cost,
                "delta_qaly": delta_qaly,
                "ICER_euro_per_QALY": icer,
            })

    res_df = pd.DataFrame(results).drop_duplicates(subset=["gruppo"])
    return res_df


# ========================
# GRAFICI AVANZATI
# ========================

def plot_km_pfs_os(df, figsize=(14, 5)):
    """Curve Kaplan-Meier per PFS e OS nel metastatico (trattamenti clinici)."""
    df_meta = df[df["setting"] == "Metastatico"].copy()
    
    if df_meta.empty:
        st.warning("Nessun paziente metastatico per KM.")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # PFS
    ax_pfs = axes[0]
    trts = df_meta["clinician_treatment"].unique()
    for trt in sorted(trts):
        df_trt = df_meta[df_meta["clinician_treatment"] == trt].copy()
        if len(df_trt) < 3:  # Troppo piccolo per KM
            continue
        
        kmf = KaplanMeierFitter()
        # Usa PFS; se NaN, ignora il paziente
        df_trt_pfs = df_trt[df_trt["pfs_months"].notna()].copy()
        if len(df_trt_pfs) < 3:
            continue
        
        # Crea colonna event (tutti i pazienti hanno "event" entro orizzonte)
        df_trt_pfs["event"] = 1  # Tutti hanno progressione entro l'orizzonte
        kmf.fit(durations=df_trt_pfs["pfs_months"], event_observed=df_trt_pfs["event"], label=trt)
        kmf.plot_survival_function(ax=ax_pfs, linewidth=2)
    
    ax_pfs.set_xlabel("Mesi", fontsize=11)
    ax_pfs.set_ylabel("Probabilità di sopravvivenza libera da progressione", fontsize=11)
    ax_pfs.set_title("Progression-Free Survival (metastatico, trattamenti clinici)", fontsize=12, fontweight="bold")
    ax_pfs.grid(alpha=0.3)
    ax_pfs.legend(fontsize=9, loc="best")
    
    # OS
    ax_os = axes[1]
    for trt in sorted(trts):
        df_trt = df_meta[df_meta["clinician_treatment"] == trt].copy()
        if len(df_trt) < 3:
            continue
        
        kmf = KaplanMeierFitter()
        df_trt_os = df_trt[df_trt["os_months"].notna()].copy()
        if len(df_trt_os) < 3:
            continue
        
        df_trt_os["event"] = 1
        kmf.fit(durations=df_trt_os["os_months"], event_observed=df_trt_os["event"], label=trt)
        kmf.plot_survival_function(ax=ax_os, linewidth=2)
    
    ax_os.set_xlabel("Mesi", fontsize=11)
    ax_os.set_ylabel("Probabilità di sopravvivenza globale", fontsize=11)
    ax_os.set_title("Overall Survival (metastatico, trattamenti clinici)", fontsize=12, fontweight="bold")
    ax_os.grid(alpha=0.3)
    ax_os.legend(fontsize=9, loc="best")
    
    plt.tight_layout()
    return fig


def plot_km_ai_vs_clinico(combo, figsize=(14, 5)):
    """KM PFS e OS confronto scenario AI vs Clinico (metastatico)."""
    combo_meta = combo[combo["setting"] == "Metastatico"].copy()
    
    if combo_meta.empty:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # PFS
    ax_pfs = axes[0]
    for scenario in ["AI", "Clinico"]:
        df_s = combo_meta[combo_meta["scenario"] == scenario].copy()
        if len(df_s) < 3:
            continue
        
        kmf = KaplanMeierFitter()
        df_s_pfs = df_s[df_s["pfs_months"].notna()].copy()
        if len(df_s_pfs) < 3:
            continue
        
        df_s_pfs["event"] = 1
        kmf.fit(durations=df_s_pfs["pfs_months"], event_observed=df_s_pfs["event"], label=scenario)
        kmf.plot_survival_function(ax=ax_pfs, linewidth=2.5)
    
    ax_pfs.set_xlabel("Mesi", fontsize=11)
    ax_pfs.set_ylabel("Probabilità PFS", fontsize=11)
    ax_pfs.set_title("PFS: Scenario AI vs Clinico (metastatico)", fontsize=12, fontweight="bold")
    ax_pfs.grid(alpha=0.3)
    ax_pfs.legend(fontsize=10)
    
    # OS
    ax_os = axes[1]
    for scenario in ["AI", "Clinico"]:
        df_s = combo_meta[combo_meta["scenario"] == scenario].copy()
        if len(df_s) < 3:
            continue
        
        kmf = KaplanMeierFitter()
        df_s_os = df_s[df_s["os_months"].notna()].copy()
        if len(df_s_os) < 3:
            continue
        
        df_s_os["event"] = 1
        kmf.fit(durations=df_s_os["os_months"], event_observed=df_s_os["event"], label=scenario)
        kmf.plot_survival_function(ax=ax_os, linewidth=2.5)
    
    ax_os.set_xlabel("Mesi", fontsize=11)
    ax_os.set_ylabel("Probabilità OS", fontsize=11)
    ax_os.set_title("OS: Scenario AI vs Clinico (metastatico)", fontsize=12, fontweight="bold")
    ax_os.grid(alpha=0.3)
    ax_os.legend(fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_cost_effectiveness_plane(icer_df):
    """Cost-effectiveness plane con WTP threshold."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot punti
    colors = {"TOTALE": "red", "Presidio: Busto Arsizio": "blue", "Presidio: Gallarate": "green", "Presidio: Saronno": "orange", "Setting: Adiuvante": "purple", "Setting: Metastatico": "brown"}
    
    for _, row in icer_df.iterrows():
        color = colors.get(row["gruppo"], "gray")
        ax.scatter(row["delta_qaly"], row["delta_cost"], s=150, alpha=0.6, color=color, edgecolors="black", linewidth=1)
        ax.annotate(row["gruppo"], (row["delta_qaly"], row["delta_cost"]), fontsize=8, ha="center", va="bottom")
    
    # Linee di riferimento
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=1)
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.3, linewidth=1)
    
    # Linea WTP
    qaly_range = np.linspace(icer_df["delta_qaly"].min() - 0.5, icer_df["delta_qaly"].max() + 0.5, 100)
    wtp_line = WTP_THRESHOLD * qaly_range
    ax.plot(qaly_range, wtp_line, "r-", linewidth=2, label=f"WTP = €{WTP_THRESHOLD:,.0f}/QALY", alpha=0.7)
    
    # Labels quadranti
    ax.text(0.98, 0.95, "NE: AI migliore\nma costoso", transform=ax.transAxes, fontsize=9, ha="right", va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.text(0.02, 0.95, "NW: AI peggiore\ne costoso", transform=ax.transAxes, fontsize=9, ha="left", va="top", bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5))
    ax.text(0.02, 0.05, "SW: Clinico meglio", transform=ax.transAxes, fontsize=9, ha="left", va="bottom", bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))
    
    ax.set_xlabel("ΔQALYs (AI − Clinico)", fontsize=12, fontweight="bold")
    ax.set_ylabel("ΔCosto (€)", fontsize=12, fontweight="bold")
    ax.set_title("Cost-Effectiveness Plane", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_cost_distribution_enhanced(combo):
    """Boxplot costi per scenario, con outliers e percentili."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    data_to_plot = [combo[combo["scenario"] == "AI"]["treatment_cost_euro"], combo[combo["scenario"] == "Clinico"]["treatment_cost_euro"]]
    bp = ax.boxplot(data_to_plot, labels=["AI", "Clinico"], patch_artist=True, widths=0.6)
    
    colors = ["lightblue", "lightgreen"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel("Costo totale (€)", fontsize=11, fontweight="bold")
    ax.set_title("Distribuzione dei costi per scenario", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    # Aggiungi mediane come testo
    for i, scenario in enumerate(["AI", "Clinico"]):
        median = combo[combo["scenario"] == scenario]["treatment_cost_euro"].median()
        ax.text(i + 1, median, f"€{median:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    return fig


def plot_toxicity_comparison(df):
    """Tasso di tossicità G3-4 per trattamento clinico."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    tox_summary = df.groupby("clinician_treatment")["tox_g3_4"].apply(lambda x: (x.sum() / len(x)) * 100).sort_values(ascending=False)
    
    bars = ax.bar(range(len(tox_summary)), tox_summary.values, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_xticks(range(len(tox_summary)))
    ax.set_xticklabels(tox_summary.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Tasso di tossicità G3-4 (%)", fontsize=11, fontweight="bold")
    ax.set_title("Incidenza di tossicità immunocorrelata per regime terapeutico", fontsize=12, fontweight="bold")
    ax.set_ylim([0, max(tox_summary.values) * 1.1])
    
    # Etichette su barre
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f"{height:.1f}%", ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_outcome_medians_table(df):
    """Tabella mediane PFS/OS per trattamento clinico."""
    df_meta = df[df["setting"] == "Metastatico"].copy()
    
    outcomes = []
    for trt in df_meta["clinician_treatment"].unique():
        df_trt = df_meta[df_meta["clinician_treatment"] == trt]
        if len(df_trt) < 5:
            continue
        
        pfs_median = df_trt["pfs_months"].median()
        os_median = df_trt["os_months"].median()
        n_patients = len(df_trt)
        tox_rate = (df_trt["tox_g3_4"].sum() / len(df_trt)) * 100
        
        outcomes.append({
            "Trattamento": trt,
            "N": n_patients,
            "Mediana PFS (mesi)": f"{pfs_median:.1f}",
            "Mediana OS (mesi)": f"{os_median:.1f}",
            "Tossicità G3-4 (%)": f"{tox_rate:.1f}"
        })
    
    return pd.DataFrame(outcomes)


def plot_icer_per_subgroup(combo):
    """ICER per sottogruppi clinici (BRAF, ECOG, LDH)."""
    results = []
    
    for subgroup_var in ["braf_status", "ecog", "ldh"]:
        for subgroup_val in combo[subgroup_var].unique():
            df_sub = combo[combo[subgroup_var] == subgroup_val]
            
            ai = df_sub[df_sub["scenario"] == "AI"]
            cl = df_sub[df_sub["scenario"] == "Clinico"]
            
            if ai.empty or cl.empty or len(ai) < 3:
                continue
            
            mean_cost_ai = ai["treatment_cost_euro"].mean()
            mean_cost_cl = cl["treatment_cost_euro"].mean()
            mean_qaly_ai = ai["qaly"].mean()
            mean_qaly_cl = cl["qaly"].mean()
            
            delta_cost = mean_cost_ai - mean_cost_cl
            delta_qaly = mean_qaly_ai - mean_qaly_cl
            icer = np.nan if abs(delta_qaly) < 1e-6 else delta_cost / delta_qaly
            
            results.append({
                "Sottogruppo": f"{subgroup_var}={subgroup_val}",
                "N": len(df_sub) // 2,  # Diviso per 2 perché ogni paziente ha AI e Clinico
                "ICER (€/QALY)": f"{icer:,.0f}" if not np.isnan(icer) else "N/A"
            })
    
    return pd.DataFrame(results) if results else None


def plot_motivation_barplot(df):
    """Barplot motivazioni di override."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Solo pazienti discordanti
    discordant = df[~df["concordant"]]
    reasons = discordant["reason_clinician_diff"].value_counts()
    
    bars = ax.barh(range(len(reasons)), reasons.values, color="coral", alpha=0.7, edgecolor="black")
    ax.set_yticks(range(len(reasons)))
    ax.set_yticklabels(reasons.index, fontsize=10)
    ax.set_xlabel("Numero di pazienti", fontsize=11, fontweight="bold")
    ax.set_title("Motivazioni di discordanza tra raccomandazione AI e scelta clinica", fontsize=12, fontweight="bold")
    
    # Etichette
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2., f"{int(width)}", ha="left", va="center", fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_qaly_distribution(combo):
    """Distribuzione QALY per scenario."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    data = [combo[combo["scenario"] == "AI"]["qaly"], combo[combo["scenario"] == "Clinico"]["qaly"]]
    bp = ax.boxplot(data, labels=["AI", "Clinico"], patch_artist=True, widths=0.6)
    
    colors = ["lightblue", "lightgreen"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel("QALY (5 anni)", fontsize=11, fontweight="bold")
    ax.set_title("Distribuzione QALY per scenario", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    # Mediane
    for i, scenario in enumerate(["AI", "Clinico"]):
        median = combo[combo["scenario"] == scenario]["qaly"].median()
        ax.text(i + 1, median, f"{median:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    return fig


# ========================
# INTERFACCIA STREAMLIT
# ========================

st.set_page_config(page_title="Melanoma AI-Clinico Simulator AVANZATO", layout="wide")
st.title("🔬 Simulatore Decisionale AI vs Clinico - Melanoma (VERSIONE AVANZATA)")
st.subheader("Kaplan–Meier, Cost-Effectiveness Plane, Analisi Sottogruppi")

st.sidebar.markdown("## ⚙️ Parametri")
n_patients = st.sidebar.slider("Numero pazienti", 100, 2000, 400, step=50)
seed = st.sidebar.number_input("Seed random", min_value=0, max_value=9999, value=42)
horizon_years = st.sidebar.slider("Orizzonte (anni)", 2.0, 10.0, 5.0, step=0.5)

if st.button("▶️ Esegui simulazione completa", key="run_sim"):
    with st.spinner("Simulazione in corso..."):
        df = simulate_patients(n_patients, seed)
        df["ai_treatment"] = df.apply(ai_recommendation, axis=1)
        df["clinician_treatment"] = df.apply(clinician_choice, axis=1)
        df = simulate_outcomes(df)
        df = add_costs_and_qalys(df, horizon_years)
        df = add_concordance_and_reason(df)

    st.success("✅ Simulazione completata.")

    # ===== SEZIONE 1: CONCORDANZA =====
    st.markdown("---")
    st.markdown("## 1️⃣ Concordanza AI–Clinico")
    kappa = compute_kappa(df)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cohen's Kappa", f"{kappa:.3f}", "eccellente" if kappa > 0.75 else "buona" if kappa > 0.6 else "moderata")
    with col2:
        conc_pct = (df["concordant"].sum() / len(df)) * 100
        st.metric("% Concordanti", f"{conc_pct:.1f}%")
    with col3:
        st.metric("% Discordanti", f"{100 - conc_pct:.1f}%")

    conc_summary = df.groupby(["presidio", "setting"]).agg(n_pazienti=("patient_id", "count"), concordance_rate=("concordant", "mean")).reset_index()
    conc_summary["concordance_rate"] = (conc_summary["concordance_rate"] * 100).round(1)
    st.dataframe(conc_summary, use_container_width=True)

    # ===== SEZIONE 2: OUTCOME CLINICI =====
    st.markdown("---")
    st.markdown("## 2️⃣ Outcome clinici (scenario clinico)")

    summary_outcomes = df.groupby(["presidio", "setting"]).agg(
        n_pazienti=("patient_id", "count"),
        mean_pfs=("pfs_months", lambda x: x.mean() if not x.isna().all() else np.nan),
        mean_os=("os_months", "mean"),
        tox_rate=("tox_g3_4", "mean"),
        mean_cost=("treatment_cost_euro", "mean"),
        mean_qaly=("qaly", "mean"),
    ).reset_index().round(2)
    st.dataframe(summary_outcomes, use_container_width=True)

    # ===== SEZIONE 3: KAPLAN-MEIER =====
    st.markdown("---")
    st.markdown("## 3️⃣ Curve Kaplan–Meier (metastatico, stratificate per trattamento)")
    st.markdown("**Figura 1:** PFS e OS per regime terapeutico scelto dal clinico. Le curve mostrano la gerarchia di efficacia: ICI-combo > PD-1 > BRAF/MEK > BSC.")
    
    fig_km = plot_km_pfs_os(df)
    if fig_km:
        st.pyplot(fig_km)
    else:
        st.info("Numero pazienti metastatici insufficiente per KM.")

    # ===== SEZIONE 3B: OUTCOME MEDIANI =====
    st.markdown("### Mediane di outcome per regime")
    outcome_table = plot_outcome_medians_table(df)
    if not outcome_table.empty:
        st.dataframe(outcome_table, use_container_width=True)
    
    # ===== SEZIONE 3C: TOSSICITÀ =====
    st.markdown("### Incidenza di tossicità G3-4 per regime")
    fig_tox = plot_toxicity_comparison(df)
    st.pyplot(fig_tox)

    # ===== SEZIONE 4: SCENARIO AI vs CLINICO =====
    st.markdown("---")
    st.markdown("## 4️⃣ Confronto scenario AI vs Clinico")
    st.markdown("**Figura 2:** Curve KM sovrapposte per scenario. Differenza visiva che evidenzia il potenziale beneficio (o trade-off) di uno scenario AI-driven.")
    
    combo = scenario_ai_vs_clinical(df, horizon_years)
    fig_km_ai = plot_km_ai_vs_clinico(combo)
    if fig_km_ai:
        st.pyplot(fig_km_ai)

    # ===== SEZIONE 5: COST-EFFECTIVENESS =====
    st.markdown("---")
    st.markdown("## 5️⃣ Analisi costo-efficacia (ICER/QALY)")

    icer_df = compute_icer(combo)
    st.dataframe(icer_df.round(2), use_container_width=True)

    st.markdown(f"**Soglia WTP:** €{WTP_THRESHOLD:,.0f}/QALY (standard europeo)")

    # ===== SEZIONE 5B: CE PLANE =====
    st.markdown("### Cost-Effectiveness Plane")
    st.markdown("**Figura 3:** Posizionamento dei diversi scenari nel CE plane. Quadrante NE (AI migliore ma costoso con ICER accettabile) è il più favorevole.")
    
    fig_ce = plot_cost_effectiveness_plane(icer_df)
    st.pyplot(fig_ce)

    # ===== SEZIONE 5C: DISTRIBUZIONI COSTI QALY =====
    st.markdown("### Distribuzioni di costi e QALY")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Figura 4a:** Distribuzione costi")
        fig_costs = plot_cost_distribution_enhanced(combo)
        st.pyplot(fig_costs)
    with col2:
        st.markdown("**Figura 4b:** Distribuzione QALY")
        fig_qaly = plot_qaly_distribution(combo)
        st.pyplot(fig_qaly)

    # ===== SEZIONE 5D: ICER SOTTOGRUPPI =====
    st.markdown("### ICER per sottogruppi clinici")
    st.markdown("**Figura 5:** Variabilità di ICER in funzione di BRAF, ECOG, LDH. Mostra dove l'AI è più cost-effective.")
    
    icer_subgroup = plot_icer_per_subgroup(combo)
    if icer_subgroup is not None:
        st.dataframe(icer_subgroup, use_container_width=True)

    # ===== SEZIONE 6: MOTIVAZIONI =====
    st.markdown("---")
    st.markdown("## 6️⃣ Motivazioni di discordanza")
    st.markdown("**Figura 6:** Frequenza dei motivi per cui i clinici deviavano dalle raccomandazioni AI.")
    
    fig_reasons = plot_motivation_barplot(df)
    st.pyplot(fig_reasons)

    # ===== SEZIONE 7: DOWNLOAD =====
    st.markdown("---")
    st.markdown("## 7️⃣ Download risultati")

    csv_df = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Dataset clinico completo (CSV)", csv_df, "melanoma_scenario_clinico_avanzato.csv", "text/csv")

    csv_combo = combo.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Dataset combinato AI vs Clinico (CSV)", csv_combo, "melanoma_ai_vs_clinico_avanzato.csv", "text/csv")

    csv_icer = icer_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Risultati ICER (CSV)", csv_icer, "melanoma_icer_avanzato.csv", "text/csv")

if __name__ == "__main__":
    pass
