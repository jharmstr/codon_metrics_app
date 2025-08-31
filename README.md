# Codon Metrics Shiny App

This repository contains a Python [Shiny](https://shiny.posit.co/py/) web application for computing codon usage metrics from coding sequence (CDS) data.  
The app allows you to score sequences for **CAI (Codon Adaptation Index)**, **tAI (tRNA Adaptation Index)**, **ENC (Effective Number of Codons)**, and **Copt (Codon Optimal)** metrics, visualize the results, and export them as tables or plots.

Authored by Jake Armstrong

https://jharmstr.github.io/codon_metrics_app/

---

## Features

- **Input options**
  - Upload a FASTA file of CDS sequences.
  - Paste sequences directly into a text area (FASTA format or one sequence per line).
  - Upload a CSV table with sequence data (select ID, sequence, and optional species columns).

- **Built-in weights**
  - *E. coli* (CAI, tAI, Copt ratios).
  - *S. cerevisiae* (CAI, tAI, Copt ratios).

- **Metrics**
  - **CAI** (Codon Adaptation Index)
  - **tAI** (tRNA Adaptation Index)
  - **ENC** (Effective Number of Codons)
  - **Copt**
    - Gene-level ratio (geometric mean of codon High/Low ratios)
    - Gene-level log2 mean
    - Binary % optimal codons (≥ 1.0 threshold)

- **Outputs**
  - Interactive metrics table
  - Copy-pasteable TSV export
  - Boxplot visualizations (per metric or combined)
  - CSV download of computed metrics
  - Augmented CSV download (for Table mode: merges metrics into the original input table)

---

## Installation

1. Clone or download this repository.
2. Install dependencies (Python ≥3.9 recommended):

   ```bash
   pip install shiny biopython pandas matplotlib seaborn scikit-learn
   
## Running the App

From the repository directory, run:
shiny run --reload app.py
This will start a local server. By default, the app is accessible at:  http://127.0.0.1:8000

## Usage
Choose input mode (sidebar):
  1. Upload FASTA: supply .fa, .fasta, or .fna file.
  2. Paste sequences: copy/paste directly into text box.
  3. Upload table (CSV): load a CSV containing CDS sequences.

If using CSV table mode:
  1. Select the ID, Sequence, and optional Species columns.
  2. Or set a fixed species for all rows.

Outputs:
  1. Codon Metrics Table: view results interactively.
  2. Copy-paste table (TSV): quickly grab results for spreadsheets or R/Python.
  3. Boxplots: visualize distributions of CAI, tAI, ENC, and Copt.

Downloads:
  1. Metrics table as CSV
  2. Boxplots as PNG
  3. Augmented CSV (Table mode only: original table + computed metrics)

## Example

Input CSV:
| tx_id  | species | CDS                     |
| ------- | ------- | ----------------------- |
| gfp     | ecoli   | ATGAGTAAAGGAGAAGAACTTT… |
| mcherry | scer    | ATGGTGAGCAAGGGCGAGGAG…  |


Output CSV (augmented):

| tx_id  | species | CDS | CAI  | tAI  | ENC  | Copt\_ratio (Zhou) | Copt\_log2 (Zhou) | Copt (%) |
| ------- | ------- | --- | ---- | ---- | ---- | ------------------ | ----------------- | -------- |
| gfp     | ecoli   | …   | 0.73 | 0.68 | 42.1 | 1.23               | 0.30              | 64.5     |
| mcherry | scer    | …   | 0.81 | 0.74 | 39.7 | 0.95               | -0.07             | 48.9     |


## Notes
Input sequences are automatically normalized (uppercase, U→T, trimmed to full codons).
Single-codon amino acids (Met, Trp) are treated as neutral (ratio = 1.0) in Copt calculations.
Exponential/log transforms require strictly positive values (e.g., when comparing metrics downstream).
Built-in weights currently cover E. coli and S. cerevisiae only, but additional species can be added by extending the weight dictionaries.
CAI weights determined by using original CAI definition as done by Sharp et al., and based on codon usage tables from Genscript (https://www.genscript.com/tools/codon-frequency-table)
tAI weights determined by using original tAI definition as done by dos Reis et al, and based on tRNA counts from gtRNAdb (https://gtrnadb.org/)

## License
MIT License - https://mit-license.org/

## References:
Sharp et al., The codon Adaptation Index--a measure of directional synonymous codon usage bias, and its potential applications, Nucleic Acids Research (1987).

Wright, The ‘effective number of codons’ used in a gene, Gene (1990).

dos Reis et al., Solving the riddle of codon usage preferences: a test for translational selection, Nucleic Acids Research (2004).

Zhou et al., Measuring codon usage bias in microbial genomes, Nucleic Acids Research (2016).
