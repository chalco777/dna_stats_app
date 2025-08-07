#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__version__ = "0.1.0"

"""
Genomics Dashboard – VERSION en_v0.1.0.py
"""

from __future__ import annotations
import os, re, shutil, subprocess, tempfile, pathlib
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from io import BytesIO
import base64
import textwrap


def parse_fasta(txt: str) -> dict[str, str]:
    seqs, head, buf = {}, None, []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith(">"):
            if head:
                seqs[head] = "".join(buf)
            head, buf = ln[1:], []
        else:
            buf.append(ln.upper())
    if head:
        seqs[head] = "".join(buf)
    return seqs

def kmer_counts(seqs: dict[str, str], k: int) -> Counter:
    cnt = Counter()
    for s in seqs.values():
        for i in range(len(s) - k + 1):
            cnt[s[i : i + k]] += 1
    return cnt

def find_motif_positions(seq: str, motif: str) -> list[int]:
    m = len(motif)
    return [i + 1 for i in range(len(seq) - m + 1) if seq[i : i + m] == motif]

def highlight_motif(seq: str, motif: str) -> str:
    motif, mlen = motif.upper(), len(motif)
    i, out = 0, ""
    while i <= len(seq) - mlen:
        if seq[i : i + mlen] == motif:
            out += f'<span style="background:#ffaeae;font-weight:bold">{motif}</span>'
            i += mlen
        else:
            out += seq[i]
            i += 1
    return out + seq[i:]

CPU_ALL = os.cpu_count() or 4

def _run(cmd: list[str], stdout_path: pathlib.Path | None = None, env: dict | None = None):
    if stdout_path:
        with stdout_path.open("w") as fh:
            subprocess.run(cmd, stdout=fh, check=True, env=env)
    else:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True, env=env)

def align_fasta(fa_txt: str, alg: str) -> str:
    tmp = pathlib.Path(tempfile.mkdtemp())
    inp, outp = tmp / "in.fa", tmp / "aln.fa"
    inp.write_text(fa_txt)

    if alg == "Clustal-Omega":
        _run(["clustalo", "-i", inp, "-o", outp, "--force", "--threads", str(CPU_ALL)], stdout_path=None)
    else:  # MAFFT
        _run(["mafft", "--auto", "--thread", str(CPU_ALL), "--quiet", inp], stdout_path=outp)
    return outp.read_text()

def compute_identity_matrix(aligned_seqs: dict[str, str]) -> pd.DataFrame:
    species = list(aligned_seqs.keys())
    n = len(species)
    L = len(next(iter(aligned_seqs.values())))
    mat = np.zeros((n, n))
    
    seq_list = [aligned_seqs[sp] for sp in species]
    
    for i in range(n):
        s1 = seq_list[i]
        for j in range(i, n):
            s2 = seq_list[j]
            matches = 0
            total = 0
            for k in range(L):
                if s1[k] == '-' and s2[k] == '-':
                    continue
                total += 1
                if s1[k] == s2[k]:
                    matches += 1
            identity = matches / total * 100 if total > 0 else 0
            mat[i, j] = identity
            mat[j, i] = identity
    
    return pd.DataFrame(mat, index=species, columns=species)

def conservation_profile(aligned_seqs: dict[str, str]) -> np.ndarray:
    seqs = list(aligned_seqs.values())
    L = len(seqs[0])
    prof = np.zeros(L)
    for i in range(L):
        col = [s[i] for s in seqs if s[i] != '-']
        if not col:
            continue
        mc = Counter(col).most_common(1)[0][1]
        prof[i] = mc / len(col)
    return prof

def conservation_profile_trimmed(aligned_seqs: dict[str, str]) -> tuple[np.ndarray, int]:
    """Profile that penalizes gaps (denominator = total number of sequences)
       and trims ends where only one sequence contributes a residue."""
    seqs   = list(aligned_seqs.values())
    n_seq  = len(seqs)
    L      = len(seqs[0])

    # how many sequences are NOT a gap at each position
    non_gap_counts = [sum(s[i] != '-' for s in seqs) for i in range(L)]

    # first / last indices with ≥2 non-gaps
    start = next((i for i,c in enumerate(non_gap_counts) if c > 1), 0)
    end   = next((L-1-i for i,c in enumerate(reversed(non_gap_counts)) if c > 1), L-1)

    prof = np.zeros(L)
    for i in range(L):
        col = [s[i] for s in seqs if s[i] != '-']
        if col:
            mc = Counter(col).most_common(1)[0][1]
            prof[i] = mc / n_seq         # ← penalizes gaps
    return prof[start:end+1], start      # trimmed profile + offset


def generate_story(df_stats: pd.DataFrame, identity: pd.DataFrame, prof: np.ndarray, nwk: str, modo: str) -> str:
    longest = df_stats.loc[df_stats['Length'].idxmax(), 'Species']
    shortest = df_stats.loc[df_stats['Length'].idxmin(), 'Species']
    max_length = df_stats['Length'].max()
    min_length = df_stats['Length'].min()
    most_gc = df_stats.loc[df_stats['%GC'].idxmax(), 'Species']
    least_gc = df_stats.loc[df_stats['%GC'].idxmin(), 'Species']

    tril = identity.where(np.tril(np.ones(identity.shape), k=-1).astype(bool))
    max_pair = tril.stack().idxmax()
    min_pair = tril.stack().idxmin()

    conserved_cols = (prof > 0.9).sum()
    var_cols = (prof < 0.5).sum()
    root_like = nwk.split(',')[0].replace('(', '')

    def fmt(txt):
        return textwrap.fill(txt, width=90)

    if modo == "General-audience":
        tpl = (
            f"**In summary:**  \n"
            f"- **{longest}** has the longest sequence ({max_length} bp), while **{shortest}** is the shortest ({min_length} bp).  \n"
            f"- GC content ranges from **{least_gc}** (minimum) to **{most_gc}** (maximum).  \n"
            f"- The most similar pair is **{max_pair[0]} – {max_pair[1]}** "
            f"({identity.loc[max_pair]:.1f}% identity); the least similar is "
            f"**{min_pair[0]} – {min_pair[1]}**.  \n"
            f"- We found **{conserved_cols} ultraconserved positions** (>90% identity) "
            f"and **{var_cols} highly variable ones** (<50%).  \n"
            f"- The tree suggests that **{root_like}** split off first within the group.  \n"
        )
    else:  # Technical
        tpl = (
            f"**In summary:**  \n"
            f"- **{longest}** has the longest sequence ({max_length} bp), while **{shortest}** is the shortest ({min_length} bp).  \n"
            f"- GC content ranges from **{least_gc}** (minimum) to **{most_gc}** (maximum).  \n"
            f"- The most similar pair is **{max_pair[0]} – {max_pair[1]}** "
            f"({identity.loc[max_pair]:.1f}% identity); the least similar is "
            f"**{min_pair[0]} – {min_pair[1]}**.  \n"
            f"- We found **{conserved_cols} ultraconserved positions** (>90% identity) "
            f"and **{var_cols} highly variable ones** (<50%).  \n"
            f"- The tree suggests that **{root_like}** split off first within the group.  \n"
        )
        tpl += (
            f"\n**Technical highlights:**  \n"
            f"- Extreme lengths: {longest} ({max_length} bp, max) vs {shortest} ({min_length} bp, min).  \n"
            f"- ΔGC = {df_stats['%GC'].max() - df_stats['%GC'].min():.2f} %.  \n"
            f"- Max/min identity: {identity.loc[max_pair]:.2f} / "
            f"{identity.loc[min_pair]:.2f}.  \n"
            f"- Conserved cols ≥0.9 = {conserved_cols}; cols ≤0.5 = {var_cols}.  \n"
            f"- Root-like taxon: {root_like}.  \n"
        )
        tpl += "\n" + fmt(
            "These patterns are consistent with the expected evolution of the nuclear gene "
            "COX4I1: it remains highly conserved in mammals and shows greater divergence "
            "when compared with the avian lineage."
        )
    return tpl


    

def _fasttree_exe():
    for e in ("FastTreeMP", "fasttreeMP", "FastTree", "fasttree"):
        if shutil.which(e):
            return e
    raise RuntimeError("FastTree not found")

def make_tree_fasttree(aln_txt):
    import tempfile, subprocess, shutil
    from pathlib import Path
    exe = _fasttree_exe()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fa = td/"aln.fa"; fa.write_text(aln_txt)
        out = td/"ft.tree"
        r = subprocess.run([exe, "-gtr", "-gamma", str(fa)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"FastTree failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
        out.write_text(r.stdout)
        return out.read_text()

def make_tree_iqtree(aln_txt, model="GTR+G", boot=1000):
    import tempfile, uuid, subprocess, os
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fa = td/"aln.fa"; fa.write_text(aln_txt)
        prefix = td/f"iq_{uuid.uuid4().hex}"
        cmd = ["iqtree2","-s",str(fa),"-m",model,
               "-B",str(boot),"-T","AUTO",
               "--prefix",str(prefix),"--redo","--quiet"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=td)
        if r.returncode not in (0,2):
            raise RuntimeError(f"IQ-TREE failed\nCMD: {' '.join(cmd)}\n"
                               f"STDOUT:\n{r.stdout}\n\nSTDERR:\n{r.stderr}")
        # localizar archivo
        treefile = prefix.with_suffix(".treefile")
        if not treefile.exists():
            treefile = prefix.with_suffix(".contree")
        if not treefile.exists():
            files = '\n'.join(str(p) for p in td.glob(prefix.name+"*"))
            raise FileNotFoundError(
                f"No tree file produced.\nCMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{r.stdout}\n\nSTDERR:\n{r.stderr}\n\nFound files:\n{files}"
            )
        return treefile.read_text()

def make_tree(aln_txt, model="GTR+G", boot=1000):
    import tempfile, uuid, subprocess, os
    from pathlib import Path

    CPU_ALL = os.cpu_count() or 1
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fa = td / "aln.fa"
        fa.write_text(aln_txt)

        prefix = td / f"iq_{uuid.uuid4().hex}"       # put outputs INSIDE tmp
        cmd = [
            "iqtree2", "-s", str(fa), "-m", model,
            "-B", str(boot), "-T", "AUTO",
            "--prefix", str(prefix), "--redo", "--quiet"
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=td)
        if r.returncode not in (0, 2):
            raise RuntimeError(
                f"IQ-TREE failed\nCMD: {' '.join(cmd)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )

        treefile = prefix.with_suffix(".treefile")
        if not treefile.exists():
            treefile = prefix.with_suffix(".contree")  # fallback
        if not treefile.exists():
            raise FileNotFoundError(
                "No tree file produced.\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )

        return treefile.read_text()   # or return path if you prefer


def phylocanvas_html(nwk: str) -> str:
    nk = nwk.replace("\n", "").strip().replace('"', r"\"")
    return f"""
<!doctype html><html><head><meta charset="utf-8">
<script src="https://unpkg.com/@phylocanvas/phylocanvas.gl@latest/dist/bundle.min.js"></script>
<style>html,body,#tree{{margin:0;height:100%;width:100%;background:#FFFFFF}}</style>
</head><body><div id="tree"></div>
<script>
const NK="{nk}", el=document.getElementById('tree');
function build(){{
  const t=new phylocanvas.PhylocanvasGL(el,{{
    source:NK,type:phylocanvas.TreeTypes.Rectangular,
    showLabels:true,showLeafLabels:true,interactive:true,antialias:true,
    pixelRatio:window.devicePixelRatio||2,
    styles:{{branch:{{color:'#000',width:4}},
            node:{{color:'#000',size:6}},
            label:{{color:'#000',font:'18px sans-serif',
                    backgroundColor:'#FFF',backgroundOpacity:1,padding:4}}}}
  }});
  const fit=()=>t.fit(); t.on('loaded',fit); window.addEventListener('resize',fit); fit();
}}
(function wait(){{ el.clientWidth?build():requestAnimationFrame(wait); }})();
</script></body></html>"""

COLOR_MAP = {
    "A": "#66c2a5", "T": "#fc8d62", "U": "#fc8d62",
    "G": "#ffd92f", "C": "#8da0cb", "-": "#bdbdbd", "N": "#d9d9d9",
}

def color_span(ch: str) -> str:
    return f'<span style="background:{COLOR_MAP.get(ch.upper(), "#fff")};color:#000">{ch}</span>'

def alignment_html(aln_fa: str) -> str:
    aln = parse_fasta(aln_fa)
    max_name = max(len(n) for n in aln)
    rows = []
    for name, seq in aln.items():
        colored = "".join(color_span(c) for c in seq)
        rows.append(
            f'<div style="font-family:monospace">'
            f'<span style="display:inline-block;width:{max_name}ch;'
            f'font-weight:bold">{name}</span> {colored}</div>'
        )
    return (
        '<div style="border:1px solid #e0e0e0;'
        'padding:8px;overflow-x:auto;white-space:nowrap;'
        'background:#ffffff">'
        + "\n".join(rows)
        + "</div>"
    )

# ═══════════════════════════════════════════════════════════════════
# MAIN APPLICATION WITH st.session_state FIX
# ═══════════════════════════════════════════════════════════════════

def main():
    st.set_page_config("Genomics Dashboard", layout="wide")
    st.title("📊 Interactive Genomics Dashboard")

    # ── INICIALIZAR SESSION STATE ──
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'seqs' not in st.session_state:
        st.session_state.seqs = {}
    if 'aln_txt' not in st.session_state:
        st.session_state.aln_txt = ""
    if 'nwk' not in st.session_state:
        st.session_state.nwk = ""
    if 'df_stats' not in st.session_state:
        st.session_state.df_stats = pd.DataFrame()
    if 'aligned_seqs' not in st.session_state:
        st.session_state.aligned_seqs = {}
    if 'identity_df' not in st.session_state:
        st.session_state.identity_df = pd.DataFrame()
    if 'prof_raw' not in st.session_state:
        st.session_state.prof_raw = np.array([])
    if 'prof_trim' not in st.session_state:
        st.session_state.prof_trim = np.array([])
    if 'offset' not in st.session_state:
        st.session_state.offset = 0

    sb = st.sidebar
    f_up = sb.file_uploader("Fasta File", ["fa", "fasta", "txt"])
    alg = sb.selectbox("Aligner", ["MAFFT", "Clustal-Omega"])
    tree_m = sb.selectbox("Tree", ["FastTree (rápido)", "IQ-TREE (ML)"])
    boot = sb.slider("Bootstraps IQ-TREE", 1000, 5000, 1000, 500, disabled=(tree_m != "IQ-TREE (ML)"))

    # ── MAIN PROCESSING (only when you click Run) ──
    if f_up and sb.button("Run"):
        # Parseo y renombrado de cabeceras
        seqs_raw = parse_fasta(f_up.getvalue().decode())
        if len(seqs_raw) < 2:
            st.error("The FASTA must contain ≥2 sequences")
            return

        org_rx = re.compile(r"\[organism=([^\]]+)\]", re.I)
        seqs = {(m.group(1).replace(" ", "_") if (m := org_rx.search(h)) else h.split()[0]): s
                for h, s in seqs_raw.items()}
        fasta_one = "\n".join(f">{sp}\n{seqs[sp]}" for sp in seqs)

        # Procesamiento pesado
        with st.spinner("Aligning..."):
            aln_txt = align_fasta(fasta_one, alg)
        with st.spinner('Computing similarities...'):
            aligned_seqs = parse_fasta(aln_txt)
            identity_df = compute_identity_matrix(aligned_seqs)
            prof_raw = conservation_profile(aligned_seqs)
            prof_trim, off = conservation_profile_trimmed(aligned_seqs) 
        with st.spinner("Inferring tree..."):
            if tree_m.startswith("FastTree"):
                nwk = make_tree_fasttree(aln_txt)     # usa FastTree
            else:
                nwk = make_tree_iqtree(aln_txt, model="GTR+G", boot=boot)
            

        # Basic statistics
        df_stats = pd.DataFrame([
            {"Species": sp, "Length": len(s), "%GC": round((Counter(s)["G"] + Counter(s)["C"]) * 100 / len(s), 2)}
            for sp, s in seqs.items()
        ])

        # ── SAVE ALL IN SESSION STATE ──
        st.session_state.seqs = seqs
        st.session_state.aln_txt = aln_txt
        st.session_state.nwk = nwk
        st.session_state.df_stats = df_stats
        st.session_state.aligned_seqs = aligned_seqs
        st.session_state.identity_df = identity_df
        st.session_state.prof_raw = prof_raw
        st.session_state.prof_trim = prof_trim
        st.session_state.offset = off

        st.session_state.data_processed = True

        st.success("✅ Processing completed!")

    # ── SHOW TABS ONLY IF DATA IS AVAILABLE ──
    if not st.session_state.data_processed:
        st.info("Upload a FASTA file and click **Run**")
        return

    # Retrieve data from session state

    seqs = st.session_state.seqs
    aln_txt = st.session_state.aln_txt
    nwk = st.session_state.nwk
    df_stats = st.session_state.df_stats
    aligned_seqs = st.session_state.aligned_seqs
    identity_df = st.session_state.identity_df
    prof_raw = st.session_state.prof_raw
    prof_trim = st.session_state.prof_trim
    offset = st.session_state.offset

    # ── TABS ──
    (tab_stats, tab_kmer, tab_motif, tab_aln, tab_tree, tab_story) = st.tabs(
        ["Statistics", "k-mers", "Motif", "Alignment", "Tree & Viewer", "Story"]
    )

    # 1️⃣ Statistics
    with tab_stats:
        st.markdown(
            "**Why does it matter?**  \n"
            "These basic metrics (length and %GC) are the first quality control step "
            "for any FASTA dataset. Large differences may reveal contaminant sequences, "
            "fragmented entries, or compositional biases."
        )
        st.dataframe(df_stats)

        fig, ax = plt.subplots()
        ax.hist(df_stats["Length"], bins=20, color="#5563DE", edgecolor="#fff")
        ax.set_xlabel("Length (bp)")
        ax.set_ylabel("Number of sequences")
        st.pyplot(fig)

    # 2️⃣ k-mers – NOW UPDATES WITHOUT RELOADING EVERYTHING
    with tab_kmer:
        st.markdown(
            "**What is it for?**  \n"
            "The k-mer distribution reveals genomic signatures: over- or under-represented "
            "sequences that may indicate restriction sites, codon usage preferences, or repetitive regions."
        )
        k = st.number_input("k-mer size", 1, 10, 3, key="kmer_len")
        
        # Here's the key! It recalculates only when k changes
        dfk = pd.DataFrame(kmer_counts(seqs, k).most_common(20), columns=["k-mer", "Count"])
        st.dataframe(dfk)

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.bar(dfk["k-mer"], dfk["Count"], color="#74ABE2")
        plt.xticks(rotation=45)
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)


    # 3️⃣ Motif – NOW UPDATES WITHOUT RELOADING EVERYTHING
    with tab_motif:
        st.markdown(
            "**Motif / pattern**  \n"
            "Quickly search for an oligonucleotide (e.g., a promoter site) and display "
            "where it appears in each sequence."
        )
        motif = st.text_input("Motif to search", key="motif_txt").upper()

        if motif:
            m = motif.upper()
            rows = [{"Species": sp, "Occurrences": len(pos := find_motif_positions(s, m)), "Positions": pos}
                    for sp, s in seqs.items()]
            st.dataframe(pd.DataFrame(rows))

            st.subheader("Highlighted in sequences")
            for sp, s in seqs.items():
                st.markdown(f"**{sp}**", unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-family:monospace;white-space:nowrap;overflow-x:auto;'
                    f'border:1px solid #eee;padding:6px">{highlight_motif(s, m)}</div>',
                    unsafe_allow_html=True)
        else:
            st.info("Enter a motif above.")


    # 4️⃣ Alignment
    with tab_aln:
        st.subheader("Sequence Identity Matrix")
        st.markdown(
            "**Identity calculation:** The matrix is computed after the multiple sequence alignment, "
            "where all sequences are of equal length due to the insertion of gaps (`-`). "
            "For each pair of sequences, a position-by-position comparison is performed:  \n"
            "- **Matches**: When both characters are identical (same base in both sequences)  \n"
            "- **Valid positions**: Columns with at least one nucleotide (double gaps are ignored)  \n"
            "- **% Identity** = (No. of matches / No. of valid positions) × 100  \n"
            "This standard method ensures fair comparisons between sequences of differing original lengths."
        )
        display_df = identity_df.copy()
        np.fill_diagonal(display_df.values, 100)
        formatted_df = display_df.applymap(lambda x: f"{x:.2f}%")
        
        def style_low_values(val):
            num = float(val.rstrip('%'))
            color = "#AA0909" if num < 50 else "#70e807" if num < 75 else "#029a49"
            return f'background-color: {color}; font-weight: bold'
        
        st.dataframe(formatted_df.style.applymap(style_low_values).format(precision=2),
                    height=min(800, 50 + 35 * len(identity_df)))
        
        csv = display_df.round(2).to_csv().encode('utf-8')
        st.download_button("Download full matrix as CSV", csv, "identity_matrix.csv", "text/csv")
        st.subheader("Multiple sequence alignment")

        st.markdown(
            "It allows comparison of homologous positions across all sequences. "
            "This is the foundation for phylogenetic inference and the detection of "
            "conserved sites or specific mutations."
        )
        st.markdown(
            "**MAFFT or Clustal Omega?**  \n"
            "* **MAFFT** uses FFT and iterative refinement; it is usually **faster** "
            "on large datasets (>1,000 sequences) and offers several modes that balance "
            "speed and accuracy.  \n"
            "* **Clustal Omega** uses progressive HMM profiles; it is deterministic and "
            "**very robust for global alignments** of similar sequences, although it can be "
            "slower on very large batches."
        )
        # Download button BEFORE the viewer
        st.download_button('📥 Download alignment', aln_txt, 'alignment.fa')

        # Alignment viewer
        components.html(alignment_html(aln_txt), height=400, scrolling=False)

        # ── Conservation profile ───────────────────────────
        st.subheader("📈 Conservation Profile")
        st.markdown(
            "**Interpretation:** High values (→1.0) = highly conserved positions; "
            "low values (→0.0) = highly variable positions among species."
        )

        modo_prof = st.radio(
            "Profile mode",
            ("Ignore gaps (original)", "Penalize gaps + trim"),
            horizontal=True,
            key="prof_mode"
        )
        
        if modo_prof.startswith("Ignore"):
            prof_plot = prof_raw
            x_vals = np.arange(len(prof_raw))
            title_tag = "original"
        else:
            prof_plot = prof_trim
            x_vals = np.arange(offset, offset + len(prof_trim))
            title_tag = f"trimmed (from {offset})"

        mostrar_total = st.checkbox("Show 100% conserved line", value=True, key="show_total")
        umbral = st.slider("Conservation threshold", 0.0, 1.0, 0.5, 0.01, key="conservation_threshold")
        ax.plot(x_vals, prof_plot, linewidth=0.25, label="Conservation")
        ax.fill_between(x_vals, prof_plot, alpha=0.3)
        ax.axhline(y=umbral, color='red', linestyle='--', linewidth=1, label=f"Threshold {umbral:.2f}")
        if mostrar_total:
            ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, label="100% conserved")

        ax.set_xlabel("Position in alignment")
        ax.set_ylabel("Conservation fraction")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right")

        # Quick stats
        ultra_conserved = (prof_plot > umbral).sum()
        highly_variable = (prof_plot < umbral).sum()
        ax.set_title(f'{title_tag.capitalize()} profile: {ultra_conserved} conserved positions, '
                    f'{highly_variable} variable positions')
        ax.grid(True, linestyle='--', alpha=0.5)

        st.pyplot(fig)
        

    # 5️⃣ Tree & Viewer
    with tab_tree:
        st.markdown("**Phylogenetic tree**  \nVisualize the inferred evolutionary relationships...")
        components.html(phylocanvas_html(nwk), height=650, scrolling=False)

    # 6️⃣ Story – NOW UPDATES WITHOUT RELOADING EVERYTHING
    with tab_story:
        st.header("📝 Sequence story")
        modo = st.radio("Story mode", ["General-audience", "Technical"], key="story_mode")

        # Recalculates only when the mode changes!
        story_md = generate_story(df_stats, identity_df, prof_raw, nwk, modo)
        st.markdown(story_md)

        md_bytes = story_md.encode('utf-8')
        st.download_button("Download story (.md)", md_bytes, "cox4i1_story.md", "text/markdown")
if __name__ == "__main__":
    main()