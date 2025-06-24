#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genomics Dashboard – pestañas con descripciones + visor interactivo de alineamiento
Funciona tanto con MAFFT como con Clustal-Omega.
"""

from __future__ import annotations
import os, re, shutil, subprocess, tempfile, pathlib
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

# ─────────────────────── utilidades FASTA, k-mers y motivos ──────────────────────
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


# ──────────────────────────── alineamiento y árbol ────────────────────────────
CPU_ALL = os.cpu_count() or 4


def _run(cmd: list[str], stdout_path: pathlib.Path | None = None, env: dict | None = None):
    """Ejecuta un comando externo con redirección opcional de stdout."""
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
        _run(
            ["clustalo", "-i", inp, "-o", outp, "--force", "--threads", str(CPU_ALL)],
            stdout_path=None,
        )
    else:  # MAFFT
        _run(
            ["mafft", "--auto", "--thread", str(CPU_ALL), "--quiet", inp],
            stdout_path=outp,
        )
    return outp.read_text()


def _fasttree_exe():
    for e in ("FastTreeMP", "fasttreeMP", "FastTree", "fasttree"):
        if shutil.which(e):
            return e
    raise RuntimeError("FastTree no encontrado")


def make_tree(aln_txt: str, method: str, boot: int) -> str:
    tmp = pathlib.Path(tempfile.mkdtemp())
    fa = tmp / "aln.fa"; fa.write_text(aln_txt)
    nwk = tmp / "tree.nwk"

    if method.startswith("FastTree"):
        env = os.environ.copy()
        exe = _fasttree_exe()
        if "MP" in exe.upper():
            env["OMP_NUM_THREADS"] = str(CPU_ALL)
        _run([exe, "-nt", "-quiet", fa], stdout_path=nwk, env=env)
    else:  # IQ-TREE
        _run(
            ["iqtree2", "-s", fa, "-m", "GTR+G",
             "-B", str(boot), "-T", str(CPU_ALL),
             "--prefix", "iq", "--quiet"]
        )
        shutil.copy2(tmp / "iq.treefile", nwk)
    return nwk.read_text()


# ─────────────────────── visor PhylocanvasGL (árbol) ─────────────────────────
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


# ─────────────────── visor de alineamiento coloreado ─────────────────────────
COLOR_MAP = {
    "A": "#66c2a5", "T": "#fc8d62", "U": "#fc8d62",
    "G": "#ffd92f", "C": "#8da0cb", "-": "#bdbdbd", "N": "#d9d9d9",
}

def color_span(ch: str) -> str:
    return (
        f'<span style="background:{COLOR_MAP.get(ch.upper(), "#fff")};'
        f'color:#000">{ch}</span>'
    )

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


# ───────────────────────────── STREAMLIT APP ────────────────────────────────
def main():
    st.set_page_config("Genomics Dashboard", layout="wide")
    st.title("📊 Panel Genómico Interactivo")

    sb = st.sidebar
    f_up = sb.file_uploader("Archivo FASTA", ["fa", "fasta", "txt"])
    k = sb.number_input("k-mer", 1, 10, 3)
    motif = sb.text_input("Motivo (ej: ATG)")
    alg = sb.selectbox("Alineador", ["MAFFT", "Clustal-Omega"])
    tree_m = sb.selectbox("Árbol", ["FastTree (rápido)", "IQ-TREE (ML)"])
    boot = sb.slider(
        "Bootstraps IQ-TREE", 1000, 5000, 1000, 500,
        disabled=(tree_m != "IQ-TREE (ML)")
    )

    if not (f_up and sb.button("Ejecutar")):
        st.info("Sube un FASTA y pulsa **Ejecutar**")
        return

    # ─── Parseo y renombrado de cabeceras ───
    seqs_raw = parse_fasta(f_up.getvalue().decode())
    if len(seqs_raw) < 2:
        st.error("El FASTA debe contener ≥2 secuencias"); return

    org_rx = re.compile(r"\[organism=([^\]]+)\]", re.I)
    seqs = { (m.group(1).replace(" ", "_") if (m := org_rx.search(h)) else h.split()[0]): s
             for h, s in seqs_raw.items() }

    fasta_one = "\n".join(f">{sp}\n{seqs[sp]}" for sp in seqs)

    with st.spinner("Alineando…"):
        aln_txt = align_fasta(fasta_one, alg)
    with st.spinner("Inferiendo árbol…"):
        nwk = make_tree(aln_txt, tree_m, boot)

    # ─── Pestañas ────────────────────────────
    tab_stats, tab_kmer, tab_motif, tab_aln, tab_tree = st.tabs(
        ["Estadísticas", "k-mers", "Motivo", "Alineamiento", "Árbol & Visor"]
    )

    # 1️⃣ Estadísticas
    with tab_stats:
        st.markdown(
            "**¿Por qué importa?**  \n"
            "Estas métricas básicas (longitud y %GC) son el primer control de calidad "
            "de un conjunto FASTA. Diferencias grandes pueden revelar secuencias "
            "contaminantes, fragmentadas o con sesgos de composición."
        )
        df_stats = pd.DataFrame(
            [{"Especie": sp,
              "Longitud": len(s),
              "%GC": round((Counter(s)["G"] + Counter(s)["C"]) * 100 / len(s), 2)}
             for sp, s in seqs.items()])
        st.dataframe(df_stats)

        fig, ax = plt.subplots()
        ax.hist(df_stats["Longitud"], bins=20, color="#5563DE", edgecolor="#fff")
        ax.set_xlabel("Longitud (bp)"); ax.set_ylabel("Nº de secuencias")
        st.pyplot(fig)

    # 2️⃣ k-mers
    with tab_kmer:
        st.markdown(
            "**¿Para qué sirve?**  \n"
            "La distribución de k-mers revela firmas genómicas: secuencias sobre- o "
            "sub-representadas que pueden indicar sitios de restricción, preferencias "
            "de codón o regiones repetitivas."
        )
        dfk = pd.DataFrame(kmer_counts(seqs, k).most_common(20),
                           columns=["k-mer", "Conteo"])
        st.dataframe(dfk)

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.bar(dfk["k-mer"], dfk["Conteo"], color="#74ABE2")
        plt.xticks(rotation=45); ax2.set_ylabel("Frecuencia")
        st.pyplot(fig2)

    # 3️⃣ Motivo
    with tab_motif:
        st.markdown(
            "**Motivo / patrón**  \n"
            "Busca rápidamente un oligonucleótido (p. ej. sitio promotor) y muestra "
            "dónde aparece en cada secuencia."
        )
        if motif:
            m = motif.upper()
            rows = [{"Especie": sp,
                     "Ocurrencias": len(pos := find_motif_positions(s, m)),
                     "Posiciones": pos}
                    for sp, s in seqs.items()]
            st.dataframe(pd.DataFrame(rows))

            st.subheader("Resaltado en secuencias")
            for sp, s in seqs.items():
                st.markdown(f"**{sp}**", unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-family:monospace;white-space:nowrap;overflow-x:auto;'
                    f'border:1px solid #eee;padding:6px">{highlight_motif(s, m)}</div>',
                    unsafe_allow_html=True)
        else:
            st.info("Escribe un motivo en la barra lateral.")

    # 4️⃣ Alineamiento (coloreado y scroll horizontal)
    with tab_aln:
        st.markdown(
            "**Alineamiento múltiple**  \n"
            "Permite comparar posiciones homólogas entre todas las secuencias. "
            "Es la base para la inferencia filogenética y para detectar sitios "
            "conservados o mutaciones específicas."
        )
        st.markdown(
            "**¿MAFFT o Clustal-Omega?**  \n"
            "* **MAFFT** emplea FFT y refinamiento iterativo; suele ser **más rápido** "
            "en conjuntos grandes (> 1 000 secuencias) y ofrece varios modos que "
            "equilibran velocidad y precisión.  \n"
            "* **Clustal-Omega** usa perfiles HMM progresivos; es determinista y "
            "**muy robusto para alineamientos globales** de secuencias similares, "
            "aunque puede tardar más en lotes muy grandes."
        )
        components.html(alignment_html(aln_txt), height=400, scrolling=False)
        st.download_button("Descargar alineamiento (FASTA)", aln_txt, "alignment.fa")

    # 5️⃣ Árbol & Visor
    with tab_tree:
        st.markdown(
            "**Árbol filogenético**  \n"
            "Visualiza las relaciones evolutivas inferidas. El visor interactivo "
            "permite acercar, desplazar y explorar etiquetas sin perder resolución."
        )
        components.html(phylocanvas_html(nwk), height=650, scrolling=False)


if __name__ == "__main__":
    main()
