from __future__ import annotations

from shiny import App, Inputs, Outputs, Session, reactive, render, ui, req
from shiny.types import FileInfo
from collections import defaultdict, Counter
import math, pandas as pd, io, matplotlib.pyplot as plt, re
from typing import Dict, List, Tuple, Optional

# Try seaborn, but plotting will be guarded anyway
try:
    import seaborn as sns  # type: ignore
except Exception:  # seaborn optional
    sns = None  # type: ignore

# ---------------- Codon table / utilities ----------------
CODON_TABLE: Dict[str, List[str]] = {
    'F': ['TTT', 'TTC'], 'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'I': ['ATT', 'ATC', 'ATA'], 'M': ['ATG'], 'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'], 'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'], 'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'Y': ['TAT', 'TAC'], 'H': ['CAT', 'CAC'], 'Q': ['CAA', 'CAG'],
    'N': ['AAT', 'AAC'], 'K': ['AAA', 'AAG'], 'D': ['GAT', 'GAC'],
    'E': ['GAA', 'GAG'], 'C': ['TGT', 'TGC'], 'W': ['TGG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'], 'G': ['GGT', 'GGC', 'GGA', 'GGG']
}
SENSE_CODONS = sorted({c for v in CODON_TABLE.values() for c in v})
AA_FOR = {c: aa for aa, codons in CODON_TABLE.items() for c in codons}
AA_TO_CODONS = {aa: set(codons) for aa, codons in CODON_TABLE.items()}

def normalize_seq(s: str) -> str:
    s = (s or "").upper().replace(" ", "").replace("\n", "").replace("U", "T")
    s = re.sub(r"[^ACGT]", "", s)
    n = len(s) - (len(s) % 3)
    return s[:n]

def normalize_codon(c: str) -> Optional[str]:
    c = (c or "").upper().replace("U", "T").strip()
    return c if len(c) == 3 and all(x in "ACGT" for x in c) else None

# ---------------- Built-in CAI/tAI weights ----------------
ECOLI_CAI = {
    'TTT': 1.0, 'TTC': 0.72565, 'TTA': 0.295593, 'TTG': 0.268044, 'CTT': 0.245184, 'CTC': 0.21075, 'CTA': 0.086953, 'CTG': 1.0, 'TCT': 0.684542, 'TCC': 0.601789, 'TCA': 0.588918, 'TCG': 0.560674, 'AGT': 0.649799, 'AGC': 1.0, 'TAT': 1.0, 'TAC': 0.698047, 'TGT': 0.862538, 'TGC': 1.0, 'TGG': 1.0, 'CCT': 0.356714, 'CCC': 0.256589, 'CCA': 0.411435, 'CCG': 1.0, 'CAT': 1.0, 'CAC': 0.74278, 'CAA': 0.512561, 'CAG': 1.0, 'CGT': 1.0, 'CGC': 0.986543, 'CGA': 0.189133, 'CGG': 0.294438, 'AGA': 0.179679, 'AGG': 0.103925, 'ATT': 1.0, 'ATC': 0.795768, 'ATA': 0.22906, 'ATG': 1.0, 'ACT': 0.469813, 'ACC': 1.0, 'ACA': 0.420996, 'ACG': 0.624095, 'AAT': 0.961666, 'AAC': 1.0, 'AAA': 1.0, 'AAG': 0.352021, 'GTT': 0.813129, 'GTC': 0.587453, 'GTA': 0.475214, 'GTG': 1.0, 'GCT': 0.566405, 'GCC': 0.804302, 'GCA': 0.703004, 'GCG': 1.0, 'GAT': 1.0, 'GAC': 0.586915, 'GAA': 1.0, 'GAG': 0.478602, 'GGT': 0.938978, 'GGC': 1.0, 'GGA': 0.350126, 'GGG': 0.415303
}
ECOLI_TAI = {
    'TTT': 0.5, 'TTC': 1.0, 'TTA': 0.222222, 'TTG': 0.333333, 'TCT': 0.5, 'TCC': 1.0, 'TCA': 0.5, 'TCG': 0.75, 'TAT': 0.5, 'TAC': 1.0, 'TGT': 0.5, 'TGC': 1.0, 'TGG': 1.0, 'CTT': 0.111111, 'CTC': 0.222222, 'CTA': 0.222222, 'CTG': 1.0, 'CCT': 0.333333, 'CCC': 0.666667, 'CCA': 0.666667, 'CCG': 1.0, 'CAT': 0.5, 'CAC': 1.0, 'CAA': 0.666667, 'CAG': 1.0, 'CGT': 1.0, 'CGC': 0.0, 'CGA': 0.0, 'CGG': 0.25, 'ATT': 0.5, 'ATC': 1.0, 'ATA': 0.0, 'ATG': 1.0, 'ACT': 0.4, 'ACC': 0.8, 'ACA': 0.4, 'ACG': 1.0, 'AAT': 0.5, 'AAC': 1.0, 'AAA': 1.0, 'AAG': 0.5, 'AGT': 0.25, 'AGC': 0.5, 'AGA': 0.25, 'AGG': 0.375, 'GTT': 0.2, 'GTC': 0.4, 'GTA': 1.0, 'GTG': 0.5, 'GCT': 0.333333, 'GCC': 0.666667, 'GCA': 1.0, 'GCG': 0.5, 'GAT': 0.5, 'GAC': 1.0, 'GAA': 1.0, 'GAG': 0.5, 'GGT': 0.5, 'GGC': 1.0, 'GGA': 0.25, 'GGG': 0.375
}

SCER_CAI = {
    'TTT': 1.0, 'TTC': 0.706116, 'TTA': 0.962331, 'TTG': 1.0, 'CTT': 0.450947, 'CTC': 0.200171, 'CTA': 0.493425, 'CTG': 0.385723, 'TCT': 1.0, 'TCC': 0.605137, 'TCA': 0.794676, 'TCG': 0.364366, 'AGT': 0.602161, 'AGC': 0.414999, 'TAT': 1.0, 'TAC': 0.787074, 'TGT': 1.0, 'TGC': 0.587774, 'TGG': 1.0, 'CCT': 0.737732, 'CCC': 0.37035, 'CCA': 1.0, 'CCG': 0.289173, 'CAT': 1.0, 'CAC': 0.570573, 'CAA': 1.0, 'CAG': 0.443874, 'CGT': 0.30048, 'CGC': 0.122181, 'CGA': 0.140652, 'CGG': 0.081614, 'AGA': 1.0, 'AGG': 0.433481, 'ATT': 1.0, 'ATC': 0.569731, 'ATA': 0.590443, 'ATG': 1.0, 'ACT': 1.0, 'ACC': 0.627873, 'ACA': 0.87596, 'ACG': 0.392727, 'AAT': 1.0, 'AAC': 0.695763, 'AAA': 1.0, 'AAG': 0.73592, 'GTT': 1.0, 'GTC': 0.533454, 'GTA': 0.533315, 'GTG': 0.487629, 'GCT': 1.0, 'GCC': 0.595246, 'GCA': 0.765478, 'GCG': 0.291693, 'GAT': 1.0, 'GAC': 0.537565, 'GAA': 1.0, 'GAG': 0.421948, 'GGT': 1.0, 'GGC': 0.409349, 'GGA': 0.456194, 'GGG': 0.252125
}
SCER_TAI = {
    'TTT': 0.5, 'TTC': 1.0, 'TTA': 0.518519, 'TTG': 1.0, 'TCT': 1.0, 'TCC': 0.0, 'TCA': 0.272727, 'TCG': 0.227273, 'TAT': 0.5, 'TAC': 1.0, 'TGT': 0.5, 'TGC': 1.0, 'TGG': 1.0, 'CTT': 0.037037, 'CTC': 0.074074, 'CTA': 0.222222, 'CTG': 0.111111, 'CCT': 0.2, 'CCC': 0.0, 'CCA': 1.0, 'CCG': 0.5, 'CAT': 0.5, 'CAC': 1.0, 'CAA': 1.0, 'CAG': 0.611111, 'CGT': 0.545455, 'CGC': 0.0, 'CGA': 0.0, 'CGG': 0.090909, 'ATT': 1.0, 'ATC': 0.0, 'ATA': 0.153846, 'ATG': 1.0, 'ACT': 1.0, 'ACC': 0.0, 'ACA': 0.363636, 'ACG': 0.272727, 'AAT': 0.5, 'AAC': 1.0, 'AAA': 0.4, 'AAG': 1.0, 'AGT': 0.090909, 'AGC': 0.181818, 'AGA': 1.0, 'AGG': 0.590909, 'GTT': 1.0, 'GTC': 0.0, 'GTA': 0.142857, 'GTG': 0.214286, 'GCT': 1.0, 'GCC': 0.0, 'GCA': 0.454545, 'GCG': 0.227273, 'GAT': 0.5, 'GAC': 1.0, 'GAA': 1.0, 'GAG': 0.642857, 'GGT': 0.5, 'GGC': 1.0, 'GGA': 0.1875, 'GGG': 0.21875
}

# COPT ratios
ECO_COPT_RATIO = {
    'GCT': 1.717, 'GCC': 0.639, 'GCA': 0.986, 'GCG': 0.900,
    'CGT': 2.934, 'CGC': 1.189, 'CGA': 0.276, 'CGG': 0.210, 'AGA': 0.144, 'AGG': 0.153,
    'AAT': 0.271, 'AAC': 3.693, 'GAT': 0.455, 'GAC': 2.198, 'TGT': 0.622, 'TGC': 1.608,
    'CAA': 0.507, 'CAG': 1.972, 'GAA': 1.331, 'GAG': 0.751, 'GGT': 1.840, 'GGC': 1.474,
    'GGA': 0.234, 'GGG': 0.327, 'CAT': 0.381, 'CAC': 2.622, 'ATT': 0.539, 'ATC': 3.406,
    'ATA': 0.130, 'CTT': 0.542, 'CTC': 0.744, 'CTA': 0.285, 'CTG': 3.759, 'TTA': 0.301,
    'TTG': 0.527, 'AAA': 1.136, 'AAG': 0.881, 'TTT': 0.285, 'TTC': 3.510, 'CCT': 0.526,
    'CCC': 0.278, 'CCA': 0.737, 'CCG': 2.883, 'TCT': 2.360, 'TCC': 2.142, 'TCA': 0.418,
    'TCG': 0.581, 'AGT': 0.343, 'AGC': 1.086, 'ACT': 1.963, 'ACC': 1.886, 'ACA': 0.293,
    'ACG': 0.413, 'TAT': 0.348, 'TAC': 2.876, 'GTT': 1.668, 'GTC': 0.579, 'GTA': 1.221,
    'GTG': 0.707, 'ATG': 1.000, 'TGG': 1.000,
}
SCER_COPT_RATIO = {
    'GCT': 3.919, 'GCC': 1.318, 'GCA': 0.150, 'GCG': 0.122,
    'CGT': 1.554, 'CGC': 0.157, 'CGA': 0.042, 'CGG': 0.040, 'AGA': 4.273, 'AGG': 0.122,
    'AAT': 0.193, 'AAC': 5.188, 'GAT': 0.485, 'GAC': 2.061, 'TGT': 3.671, 'TGC': 0.272,
    'CAA': 6.547, 'CAG': 0.153, 'GAA': 4.168, 'GAG': 0.240, 'GGT': 10.496, 'GGC': 0.304,
    'GGA': 0.120, 'GGG': 0.111, 'CAT': 0.281, 'CAC': 3.561, 'ATT': 1.261, 'ATC': 2.669,
    'ATA': 0.108, 'CTT': 0.542, 'CTC': 0.744, 'CTA': 0.285, 'CTG': 3.759, 'TTA': 0.301,
    'TTG': 0.527, 'AAA': 0.208, 'AAG': 4.811, 'TTT': 0.251, 'TTC': 3.985, 'CCT': 0.423,
    'CCC': 0.144, 'CCA': 6.153, 'CCG': 0.126, 'TCT': 2.819, 'TCC': 2.565, 'TCA': 0.331,
    'TCG': 0.191, 'AGT': 0.364, 'AGC': 0.413, 'ACT': 2.056, 'ACC': 2.640, 'ACA': 1.018,
    'ACG': 0.126, 'TAT': 0.230, 'TAC': 4.340, 'GTT': 1.950, 'GTC': 2.696, 'GTA': 0.135,
    'GTG': 0.221, 'ATG': 1.000, 'TGG': 1.000,
}

def copt_ratio_table_for_species(species_key: str) -> Dict[str, float]:
    if species_key == "scer":
        return {c: SCER_COPT_RATIO.get(c, 1.0) for c in SENSE_CODONS}
    return {c: ECO_COPT_RATIO.get(c, 1.0) for c in SENSE_CODONS}

# ---------------- Indices ----------------
def calculate_index(seq: str, weights: Dict[str, float]) -> float:
    seq = normalize_seq(seq)
    log_sum, count = 0.0, 0
    for i in range(0, len(seq) - 2, 3):
        w = weights.get(seq[i:i+3], 0.0)
        if w > 0.0:
            log_sum += math.log(w); count += 1
    return math.exp(log_sum / count) if count else 0.0

def calculate_cai(seq: str, cai_w: Dict[str, float]) -> float: return calculate_index(seq, cai_w)
def calculate_tai(seq: str, tai_w: Dict[str, float]) -> float: return calculate_index(seq, tai_w)

def calculate_enc(seq: str) -> float:
    seq = normalize_seq(seq)
    aa_codon_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        aa = AA_FOR.get(codon)
        if aa:
            aa_codon_counts[aa][codon] += 1
    Fk_list: List[Tuple[int, float]] = []
    for aa, counts in aa_codon_counts.items():
        family = AA_TO_CODONS[aa]
        k = len(family)
        if k <= 1:
            continue
        n = sum(counts.values())
        if n <= 1:
            continue
        pk = [counts.get(c, 0) / n for c in family]
        fk = sum(p * p for p in pk)
        Fk = (n * fk - 1) / (n - 1)
        if Fk > 0:
            Fk_list.append((k, Fk))
    if not Fk_list:
        return 61.0
    try:
        return float(2 + sum(k for k, _ in Fk_list) / sum(1 / Fk for _, Fk in Fk_list))
    except ZeroDivisionError:
        return 61.0

# ---------------- Copt (ratio/log2) + Binary % ----------------
def copt_ratio_for_gene(seq: str, codon_ratio: Dict[str, float]) -> Tuple[float, float]:
    seq = normalize_seq(seq)
    logs = []
    for i in range(0, len(seq) - 2, 3):
        c = seq[i:i+3]
        if c in AA_FOR:
            r = max(codon_ratio.get(c, 1.0), 1e-12)
            logs.append(math.log(r, 2))
    if not logs:
        return 1.0, 0.0
    mean_log2 = sum(logs) / len(logs)
    gm_ratio = float(2 ** mean_log2)
    return gm_ratio, float(mean_log2)

BINARY_THRESHOLD = 1.0  # codon considered "optimal" if ratio >= 1

def copt_percent_for_gene(seq: str, codon_ratio: Dict[str, float], threshold: float = BINARY_THRESHOLD) -> float:
    seq = normalize_seq(seq)
    total, optimal = 0, 0
    for i in range(0, len(seq) - 2, 3):
        c = seq[i:i+3]
        if c in AA_FOR:
            total += 1
            if codon_ratio.get(c, 1.0) >= threshold:
                optimal += 1
    return (optimal / total * 100.0) if total > 0 else 0.0

# ---------------- Sequence helpers ----------------
def parse_sequences_from_text(text: str) -> List[Tuple[str, str]]:
    text = (text or "").strip()
    if not text: return []
    if text.startswith(">"):
        entries, sid, buf = [], None, []
        for line in text.splitlines():
            if line.startswith(">"):
                if sid is not None and buf: entries.append((sid, "".join(buf)))
                sid, buf = line[1:].strip() or f"seq{len(entries)+1}", []
            else:
                buf.append(line.strip())
        if sid is not None: entries.append((sid, "".join(buf)))
        return [(sid, normalize_seq(seq)) for sid, seq in entries if normalize_seq(seq)]
    else:
        seqs = [normalize_seq(l) for l in text.splitlines() if l.strip()]
        return [(f"seq{i+1}", s) for i, s in enumerate(seqs) if s]

# ---------------- Custom weights (upload) ----------------
def parse_weights_file(path: str) -> Dict[str, float]:
    """Accepts CSV/TSV with columns like codon/CODON/triplet and weight/w/cai/tai/value/score."""
    df = pd.read_csv(path, sep=None, engine="python")
    cols_lower = {c.lower(): c for c in df.columns}

    codon_col = None
    for cand in ["codon", "triplet", "kmer", "seq", "sequence"]:
        if cand in cols_lower:
            codon_col = cols_lower[cand]; break
    if codon_col is None:
        raise ValueError("Weights file must contain a 'codon' column (or 'triplet', 'kmer', 'seq').")

    weight_col = None
    for cand in ["weight", "w", "cai", "tai", "value", "score"]:
        if cand in cols_lower:
            weight_col = cols_lower[cand]; break
    if weight_col is None:
        raise ValueError("Weights file must contain a 'weight' column (or 'w', 'cai', 'tai', 'value', 'score').")

    out: Dict[str, float] = {}
    for _, r in df[[codon_col, weight_col]].dropna().iterrows():
        c = normalize_codon(str(r[codon_col]))
        if not c or c not in SENSE_CODONS:
            continue
        try:
            w = float(r[weight_col])
        except Exception:
            continue
        if w > 0 and math.isfinite(w):
            out[c] = w
    if not out:
        raise ValueError("No valid (codon, weight) pairs found.")
    return out

# ---------------- UI ----------------
help_modal = ui.modal(
    ui.h3("Help & Notes"),
    ui.p("This app computes CAI, tAI, ENC, and Copt metrics for coding sequences."),
    ui.h4("Custom weights"),
    ui.p("Optionally upload a CSV/TSV with columns 'codon' and 'weight' to override built-in CAI or tAI weights."),
    easy_close=True,
    footer=ui.input_action_button("help_close", "Close"),
    size="l",
)

MAX_ROWS_FOR_TSV = 5000  # (5) limit textarea size

app_ui = ui.page_fillable(
    ui.navset_bar(
        ui.nav_panel(
            "App",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_action_button("help_open", "Help"),
                    ui.hr(),
                    ui.h4("Input mode"),
                    ui.input_radio_buttons(
                        "input_mode", None,
                        choices={"upload": "Upload FASTA", "paste": "Paste sequences", "table": "Upload table (CSV)"},
                        selected="upload"
                    ),
                    ui.panel_conditional("input.input_mode == 'upload'",
                        ui.input_file("fasta", "Upload FASTA", accept=[".fa",".fasta",".fna"])
                    ),
                    ui.panel_conditional("input.input_mode == 'paste'",
                        ui.input_text_area("seq_text", "Paste FASTA or one sequence per line", rows=8)
                    ),
                    ui.panel_conditional("input.input_mode == 'table'",
                        ui.TagList(
                            ui.input_file("table_file", "Upload CSV table", accept=[".csv",".tsv",".txt"]),
                            ui.output_ui("table_col_picker"),
                            ui.input_select("fixed_species", "Set species for ALL rows (optional; overrides column)",
                                            choices={"": "(none)", "ecoli": "E. coli", "scer": "S. cerevisiae"},
                                            selected="")
                        )
                    ),
                    ui.hr(),
                    ui.h4("Species (for CAI/tAI/Copt weights)"),
                    ui.input_radio_buttons("species", None,
                        choices={"ecoli": "E. coli (built-in)", "scer": "S. cerevisiae (built-in)"},
                        selected="ecoli"),

                    ui.hr(),
                    ui.h4("Custom CAI/tAI weights (optional)"),
                    ui.input_select("custom_kind", "Apply uploaded weights to", choices={"cai": "CAI", "tai": "tAI"}, selected="cai"),
                    ui.input_file("weights_file", "Upload weights CSV/TSV (codon, weight)", accept=[".csv", ".tsv", ".txt"]),
                    ui.input_action_button("clear_weights", "Clear custom weights"),
                    ui.output_ui("weights_status"),

                    ui.hr(),
                    ui.input_switch("combined", "Combine plots", value=False),
                    ui.download_button("download_csv", "Download metrics (CSV)"),
                    ui.download_button("download_plot", "Download plot (PNG)"),
                    ui.panel_conditional("input.input_mode == 'table'",
                        ui.download_button("download_augmented", "Download augmented table (CSV)")
                    ),
                    width=380
                ),
                ui.layout_columns(
                    ui.card(ui.card_header("Codon Metrics Table"), ui.output_data_frame("metrics_df")),
                    ui.card(ui.card_header("Copy-paste table (TSV)"), ui.output_ui("metrics_tsv_ui")),
                ),
                ui.card(ui.card_header("CAI / tAI / ENC / Copt"),
                        ui.output_plot("metrics_plot", height="460px"))
            ),
        ),
        ui.nav_spacer(),
        ui.nav_control(ui.a("CAI • tAI • ENC • Copt", href="#")),
        title="Codon Metrics (CAI, tAI, ENC, Copt)"
    )
)

# ---------------- Helpers updated per suggestions ----------------
def normalize_species(val: str) -> str:
    v = (val or "").strip().lower()
    if v in {"ecoli","e.coli","escherichia coli","e coli"} or "coli" in v:
        return "ecoli"
    if v in {"scer","s.cer","s cerevisiae","s. cerevisiae","saccharomyces cerevisiae","yeast"}:
        return "scer"
    return "ecoli"

def active_sets_for_species(
    species_key: str,
    cai_override: Optional[Dict[str, float]] = None,
    tai_override: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    if species_key == "scer":
        cai = dict(SCER_CAI)
        tai = dict(SCER_TAI)
    else:
        cai = dict(ECOLI_CAI)
        tai = dict(ECOLI_TAI)
    if cai_override:
        for k, v in cai_override.items():
            if k in SENSE_CODONS and v > 0:
                cai[k] = float(v)
    if tai_override:
        for k, v in tai_override.items():
            if k in SENSE_CODONS and v > 0:
                tai[k] = float(v)
    return cai, tai, copt_ratio_table_for_species(species_key)

def compute_all_metrics(
    seq: str,
    species_key: str,
    cai_override: Optional[Dict[str, float]] = None,
    tai_override: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    cai_w, tai_w, copt_ratio = active_sets_for_species(species_key, cai_override, tai_override)
    ratio_gm, log2_mean = copt_ratio_for_gene(seq, copt_ratio)
    copt_percent = copt_percent_for_gene(seq, copt_ratio, BINARY_THRESHOLD)
    return {
        "CAI": round(calculate_cai(seq, cai_w), 4),
        "tAI": round(calculate_tai(seq, tai_w), 4),
        "ENC": round(calculate_enc(seq), 2),
        "Copt_ratio": round(ratio_gm, 4),
        "Copt_log2": round(log2_mean, 4),
        "Copt (%)": round(copt_percent, 2),
    }

# ---------------- Server ----------------
def server(input: Inputs, output: Outputs, session: Session):
    # Help modal controls
    @reactive.effect
    @reactive.event(input.help_open)
    def _open_help(): ui.modal_show(help_modal)

    @reactive.effect
    @reactive.event(input.help_close)
    def _close_help(): ui.modal_remove()

    # ---- Custom weights state ----
    custom_cai = reactive.Value(dict())  # type: reactive.Value[Dict[str, float]]
    custom_tai = reactive.Value(dict())

    @reactive.effect
    @reactive.event(input.weights_file)
    def _load_weights():
        files = input.weights_file()
        if not files:
            return
        try:
            kind = input.custom_kind()
            path = files[0]["datapath"]
            parsed = parse_weights_file(path)
            if kind == "cai":
                custom_cai.set(parsed)
            else:
                custom_tai.set(parsed)
        except Exception as e:
            custom_cai.set({})
            custom_tai.set({})
            ui.notification_show(f"Failed to load weights: {e}", type="warning", duration=6)

    @reactive.effect
    @reactive.event(input.clear_weights)
    def _clear_weights():
        custom_cai.set({})
        custom_tai.set({})

    @output
    @render.ui
    def weights_status():
        nc, nt = len(custom_cai()), len(custom_tai())
        if nc == 0 and nt == 0:
            return ui.help_text("No custom weights loaded.")
        msgs = []
        if nc: msgs.append(f"Custom CAI weights active ({nc} codons).")
        if nt: msgs.append(f"Custom tAI weights active ({nt} codons).")
        return ui.div(*[ui.p(m) for m in msgs])

    # ===== FASTA / Paste modes (2: add req guards) =====
    @reactive.calc
    def metrics_df_calc_fp() -> pd.DataFrame:
        sp = input.species()
        seqs: List[Tuple[str, str]] = []
        mode = input.input_mode()

        if mode == "upload":
            files = input.fasta()
            if not files:
                return pd.DataFrame(columns=["ID","CAI","tAI","ENC","Copt_ratio","Copt_log2","Copt (%)"])
            for rec in SeqIO.parse(files[0]["datapath"], "fasta"):  # noqa: F821 (filled below)
                seqs.append((rec.id, str(rec.seq)))

        elif mode == "paste":
            txt = (input.seq_text() or "").strip()
            if not txt:
                return pd.DataFrame(columns=["ID","CAI","tAI","ENC","Copt_ratio","Copt_log2","Copt (%)"])
            seqs.extend(parse_sequences_from_text(txt))

        if not seqs:
            return pd.DataFrame(columns=["ID","CAI","tAI","ENC","Copt_ratio","Copt_log2","Copt (%)"])

        rows = [{"ID": sid, **compute_all_metrics(s, sp, custom_cai(), custom_tai())} for sid, s in seqs]
        return pd.DataFrame(rows)[["ID","CAI","tAI","ENC","Copt_ratio","Copt_log2","Copt (%)"]]

    # ===== TABLE mode =====
    @reactive.calc
    def table_df_raw() -> Optional[pd.DataFrame]:
        if input.input_mode() != "table":
            return None
        f = input.table_file()
        if not f:
            return None
        try:
            # auto-detect CSV/TSV
            return pd.read_csv(f[0]["datapath"], sep=None, engine="python")
        except Exception as e:
            return pd.DataFrame({"Error":[f"Failed to read table: {e}"]})

    # (1) Robust column guessing
    @output
    @render.ui
    def table_col_picker():
        df = table_df_raw()
        if df is None or df.empty or "Error" in (df.columns if df is not None else []):
            return ui.help_text("Upload a CSV to choose columns.")
        cols = list(df.columns)

        def pick(choices: List[str], default: str) -> str:
            low = {c.lower(): c for c in cols}
            for want in choices:
                if want in low: return low[want]
            return default

        seq_guess = pick(["cds","sequence","seq","cds_seq","coding_sequence","coding seq"], cols[0])
        id_guess  = pick(["id","tx_id","gene","name","locus","accession","transcript_id"], cols[0])
        species_guess = pick(["species","organism","host"], "")

        return ui.TagList(
            ui.input_select("id_col", "ID column", choices=cols, selected=id_guess),
            ui.input_select("seq_col", "Sequence column", choices=cols, selected=seq_guess),
            ui.input_select("species_col", "Species column (optional: 'ecoli' or 'scer')",
                            choices=["(none)"] + cols, selected=species_guess or "(none)")
        )

    @reactive.calc
    def metrics_df_calc_table() -> pd.DataFrame:
        df = table_df_raw()
        if df is None or df.empty:
            return pd.DataFrame(columns=["ID","CAI","tAI","ENC","Copt_ratio","Copt_log2","Copt (%)"])
        if "Error" in df.columns:
            return df

        # (2) req() guards so we don't spin
        req("id_col" in input and input.id_col() in df.columns)
        req("seq_col" in input and input.seq_col() in df.columns)

        id_col = input.id_col()
        seq_col = input.seq_col()
        species_col_sel = input.species_col() if "species_col" in input else "(none)"
        fixed_species = input.fixed_species() if "fixed_species" in input else ""

        rows = []
        for _, r in df.iterrows():
            sid = str(r[id_col])
            seq = normalize_seq(str(r[seq_col]))
            if not seq:
                rows.append({"ID": sid, "CAI": float("nan"), "tAI": float("nan"),
                             "ENC": float("nan"), "Copt_ratio": float("nan"),
                             "Copt_log2": float("nan"), "Copt (%)": float("nan")})
                continue

            # (3) explicit species parsing
            if fixed_species:
                sp = fixed_species
            else:
                if species_col_sel and species_col_sel != "(none)" and species_col_sel in df.columns:
                    sp = normalize_species(str(r[species_col_sel]))
                else:
                    sp = "ecoli"

            m = compute_all_metrics(seq, sp, custom_cai(), custom_tai())
            rows.append({"ID": sid, **m})

        return pd.DataFrame(rows)[["ID","CAI","tAI","ENC","Copt_ratio","Copt_log2","Copt (%)"]]

    # ===== Unified outputs =====
    def metrics_df_current() -> pd.DataFrame:
        mode = input.input_mode()
        if mode == "table":
            return metrics_df_calc_table()
        return metrics_df_calc_fp()

    # Interactive table
    @render.data_frame
    def metrics_df():
        return render.DataTable(metrics_df_current())

    # (5) TSV with soft cap
    @render.ui
    def metrics_tsv_ui():
        df = metrics_df_current()
        cols = ["ID","CAI","tAI","ENC","Copt_ratio","Copt_log2","Copt (%)"]
        if df.empty:
            tsv = "\t".join(cols)
            note = ""
        else:
            show = df if "Error" in df.columns else df[cols]
            note = ""
            if len(show) > MAX_ROWS_FOR_TSV:
                show = show.head(MAX_ROWS_FOR_TSV)
                note = f"(showing first {MAX_ROWS_FOR_TSV} rows)"
            tsv = show.to_csv(sep="\t", index=False, lineterminator="\n")
        return ui.TagList(
            ui.input_action_button("copy_btn", "Copy to clipboard"),
            ui.help_text(note) if note else None,
            ui.tags.textarea(
                tsv, id="tsv_area", readonly=True,
                style=("width:100%; height:260px; font-family: ui-monospace, SFMono-Regular, "
                       "Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;")
            ),
            ui.tags.script(
                """
                (function(){
                  const btn = document.getElementById("copy_btn");
                  const ta  = document.getElementById("tsv_area");
                  if (!btn || !ta) return;
                  btn.onclick = async function(){
                    ta.focus(); ta.select(); ta.setSelectionRange(0, ta.value.length);
                    try { await navigator.clipboard.writeText(ta.value); }
                    catch (e) { document.execCommand("copy"); }
                  };
                })();
                """
            )
        )

    # Plot (4) safe wrapper
    def make_boxplot_figure(df: pd.DataFrame, combined: bool):
        if df.empty or ("Error" in df.columns):
            fig = plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "Provide sequences", ha="center", va="center")
            plt.axis("off")
            return fig
        metrics = ["CAI","tAI","ENC","Copt_ratio","Copt_log2","Copt (%)"]
        if combined and sns is not None:
            dfm = df.melt(id_vars=["ID"], value_vars=metrics, var_name="Metric", value_name="Value")
            plt.figure(figsize=(10, 4))
            sns.boxplot(x="Metric", y="Value", data=dfm)
            plt.ylabel("Value"); plt.title("Codon Usage Metrics"); plt.tight_layout()
            return plt.gcf()
        # Fallback: individual subplots (works without seaborn)
        fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]
        for ax, col in zip(axes, metrics):
            try:
                if sns is not None:
                    sns.boxplot(y=df[col], ax=ax, width=0.3)
                else:
                    ax.boxplot(df[col].dropna().values, vert=True)
                ax.set_title(col); ax.set_ylabel(col); ax.set_xticks([])
            except Exception as e:
                ax.text(0.5, 0.5, f"{col}\n{e}", ha="center", va="center"); ax.axis("off")
        plt.tight_layout(); return fig

    def safe_boxplot(df: pd.DataFrame, combined: bool):
        try:
            return make_boxplot_figure(df, combined)
        except Exception as e:
            fig = plt.figure(figsize=(5, 2))
            plt.text(0.5, 0.5, f"Plot error: {e}", ha="center", va="center")
            plt.axis("off"); return fig

    @render.plot(alt="Boxplots of CAI, tAI, ENC, and Copt")
    def metrics_plot():
        return safe_boxplot(metrics_df_current(), combined=bool(input.combined()))

    # Downloads
    @render.download(filename="codon_metrics.csv")
    def download_csv():
        df = metrics_df_current()
        with io.StringIO() as s:
            df.to_csv(s, index=False); yield s.getvalue()

    @render.download(filename="codon_metrics_boxplot.png")
    def download_plot():
        fig = safe_boxplot(metrics_df_current(), combined=bool(input.combined()))
        with io.BytesIO() as buf:
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            plt.close(fig); yield buf.getvalue()

    @render.download(filename=lambda: augmented_name())
    def download_augmented():
        if input.input_mode() != "table":
            yield b""; return
        raw = table_df_raw()
        metrics = metrics_df_calc_table()
        if raw is None or raw.empty or metrics.empty or "Error" in metrics.columns:
            with io.StringIO() as s:
                s.write("Error: no augmented output available.\n")
                yield s.getvalue().encode("utf-8")
            return
        id_col = input.id_col() if "id_col" in input else None
        if id_col and id_col in raw.columns:
            merged = raw.merge(metrics, left_on=id_col, right_on="ID", how="left")
        else:
            merged = pd.concat([raw.reset_index(drop=True), metrics.reset_index(drop=True)], axis=1)
        with io.StringIO() as s:
            merged.to_csv(s, index=False)
            yield s.getvalue().encode("utf-8")

    def augmented_name() -> str:
        f = input.table_file()
        if f:
            name = f[0]["name"]
            base = name.rsplit(".", 1)[0]
            return f"{base}_with_metrics.csv"
        return "table_with_metrics.csv"

# ---- Biopython FASTA parsing (kept from original; simple fallback below) ----
try:
    from Bio import SeqIO  # type: ignore
except Exception:
    # Lightweight fallback if Biopython isn't present
    class _Rec:  # minimal
        def __init__(self, id, seq): self.id, self.seq = id, seq
    class SeqIO:  # type: ignore
        @staticmethod
        def parse(path, fmt):
            sid, buf = None, []
            with open(path, "r") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if line.startswith(">"):
                        if sid is not None and buf:
                            yield _Rec(sid, "".join(buf))
                        sid, buf = line[1:].strip() or "seq", []
                    else:
                        buf.append(line)
            if sid is not None and buf:
                yield _Rec(sid, "".join(buf))

app = App(app_ui, server)