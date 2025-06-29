#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genomics Dashboard – VERSIÓN CORREGIDA con st.session_state
Ya no se recarga todo cuando cambias k-mer, motivo o historia
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

# ═══════════════════════════════════════════════════════════════════
# TODO EL CÓDIGO DE UTILIDADES SE MANTIENE IGUAL
# ═══════════════════════════════════════════════════════════════════

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
    """Perfil que penaliza gaps (denominador = nº total de secuencias)
       y recorta extremos donde sólo 1 secuencia aporta residuo."""
    seqs   = list(aligned_seqs.values())
    n_seq  = len(seqs)
    L      = len(seqs[0])

    # cuántas secuencias NO son gap en cada posición
    non_gap_counts = [sum(s[i] != '-' for s in seqs) for i in range(L)]

    # primeros / últimos índices con ≥2 no-gaps
    start = next((i for i,c in enumerate(non_gap_counts) if c > 1), 0)
    end   = next((L-1-i for i,c in enumerate(reversed(non_gap_counts)) if c > 1), L-1)

    prof = np.zeros(L)
    for i in range(L):
        col = [s[i] for s in seqs if s[i] != '-']
        if col:
            mc = Counter(col).most_common(1)[0][1]
            prof[i] = mc / n_seq         # ← penaliza gaps
    return prof[start:end+1], start      # perfil recortado + offset


def generate_story(df_stats: pd.DataFrame, identity: pd.DataFrame, prof: np.ndarray, nwk: str, modo: str) -> str:
    longest = df_stats.loc[df_stats['Longitud'].idxmax(), 'Especie']
    shortest = df_stats.loc[df_stats['Longitud'].idxmin(), 'Especie']
    max_length = df_stats['Longitud'].max()
    min_length = df_stats['Longitud'].min()
    most_gc = df_stats.loc[df_stats['%GC'].idxmax(), 'Especie']
    least_gc = df_stats.loc[df_stats['%GC'].idxmin(), 'Especie']

    tril = identity.where(np.tril(np.ones(identity.shape), k=-1).astype(bool))
    max_pair = tril.stack().idxmax()
    min_pair = tril.stack().idxmin()

    conserved_cols = (prof > 0.9).sum()
    var_cols = (prof < 0.5).sum()
    root_like = nwk.split(',')[0].replace('(', '')

    def fmt(txt):
        return textwrap.fill(txt, width=90)

    if modo == "Divulgativo":
        tpl = (
            f"**En resumen:**  \n"
            f"- **{longest}** posee la secuencia más larga ({max_length} bp), mientras que **{shortest}** es la más corta ({min_length} bp).  \n"
            f"- El contenido GC varía de **{least_gc}** (mínimo) a **{most_gc}** (máximo).  \n"
            f"- El par más parecido es **{max_pair[0]} – {max_pair[1]}** "
            f"({identity.loc[max_pair]:.1f}% identidad); el menos parecido es "
            f"**{min_pair[0]} – {min_pair[1]}**.  \n"
            f"- Encontramos **{conserved_cols} posiciones ultraconservadas** (>90 % identidad) "
            f"y **{var_cols} muy variables** (<50 %).  \n"
            f"- El árbol sugiere que **{root_like}** se separó primero dentro del grupo.  \n"
        )
    else:  # Técnico
        tpl = (
            f"**En resumen:**  \n"
            f"- **{longest}** posee la secuencia más larga ({max_length} bp), mientras que **{shortest}** es la más corta ({min_length} bp).  \n"
            f"- El contenido GC varía de **{least_gc}** (mínimo) a **{most_gc}** (máximo).  \n"
            f"- El par más parecido es **{max_pair[0]} – {max_pair[1]}** "
            f"({identity.loc[max_pair]:.1f}% identidad); el menos parecido es "
            f"**{min_pair[0]} – {min_pair[1]}**.  \n"
            f"- Encontramos **{conserved_cols} posiciones ultraconservadas** (>90 % identidad) "
            f"y **{var_cols} muy variables** (<50 %).  \n"
            f"- El árbol sugiere que **{root_like}** se separó primero dentro del grupo.  \n"
        )
        tpl += (
            f"\n**Highlights técnicos:**  \n"
            f"- Longitudes extremas: {longest} ({max_length} bp, máx) vs {shortest} ({min_length} bp, mín).  \n"
            f"- ΔGC = {df_stats['%GC'].max() - df_stats['%GC'].min():.2f} %.  \n"
            f"- Identidad máx/min: {identity.loc[max_pair]:.2f} / "
            f"{identity.loc[min_pair]:.2f}.  \n"
            f"- Cols conservadas ≥0.9 = {conserved_cols}; cols ≤0.5 = {var_cols}.  \n"
            f"- Root-like taxon: {root_like}.  \n"
        )
        tpl += "\n" + fmt(
            "Estos patrones concuerdan con la evolución esperada del gen nuclear "
            "COX4I1: se mantiene altamente conservado en mamíferos y muestra mayor "
            "divergencia al compararlo con el linaje aviar."
        )
    return tpl

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
        _run(["iqtree2", "-s", fa, "-m", "GTR+G", "-B", str(boot), "-T", str(CPU_ALL), "--prefix", "iq", "--quiet"])
        shutil.copy2(tmp / "iq.treefile", nwk)
    return nwk.read_text()

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
# APLICACIÓN PRINCIPAL CON CORRECCIÓN DE st.session_state
# ═══════════════════════════════════════════════════════════════════

def main():
    st.set_page_config("Genomics Dashboard", layout="wide")
    st.title("📊 Panel Genómico Interactivo")

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
    f_up = sb.file_uploader("Archivo FASTA", ["fa", "fasta", "txt"])
    alg = sb.selectbox("Alineador", ["MAFFT", "Clustal-Omega"])
    tree_m = sb.selectbox("Árbol", ["FastTree (rápido)", "IQ-TREE (ML)"])
    boot = sb.slider("Bootstraps IQ-TREE", 1000, 5000, 1000, 500, disabled=(tree_m != "IQ-TREE (ML)"))

    # ── PROCESAMIENTO PRINCIPAL (solo cuando se hace clic en Ejecutar) ──
    if f_up and sb.button("Ejecutar"):
        # Parseo y renombrado de cabeceras
        seqs_raw = parse_fasta(f_up.getvalue().decode())
        if len(seqs_raw) < 2:
            st.error("El FASTA debe contener ≥2 secuencias")
            return

        org_rx = re.compile(r"\[organism=([^\]]+)\]", re.I)
        seqs = {(m.group(1).replace(" ", "_") if (m := org_rx.search(h)) else h.split()[0]): s
                for h, s in seqs_raw.items()}
        fasta_one = "\n".join(f">{sp}\n{seqs[sp]}" for sp in seqs)

        # Procesamiento pesado
        with st.spinner("Alineando…"):
            aln_txt = align_fasta(fasta_one, alg)
        with st.spinner('Calculando similitudes...'):
            aligned_seqs = parse_fasta(aln_txt)
            identity_df = compute_identity_matrix(aligned_seqs)
            prof_raw = conservation_profile(aligned_seqs)
            prof_trim, off = conservation_profile_trimmed(aligned_seqs) 
        with st.spinner("Inferiendo árbol…"):
            nwk = make_tree(aln_txt, tree_m, boot)

        # Estadísticas básicas
        df_stats = pd.DataFrame([
            {"Especie": sp, "Longitud": len(s), "%GC": round((Counter(s)["G"] + Counter(s)["C"]) * 100 / len(s), 2)}
            for sp, s in seqs.items()
        ])

        # ── GUARDAR TODO EN SESSION STATE ──
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

        st.success("✅ ¡Procesamiento completado!")

    # ── MOSTRAR PESTAÑAS SOLO SI HAY DATOS ──
    if not st.session_state.data_processed:
        st.info("Sube un FASTA y pulsa **Ejecutar**")
        return

    # Recuperar datos del session state
    seqs = st.session_state.seqs
    aln_txt = st.session_state.aln_txt
    nwk = st.session_state.nwk
    df_stats = st.session_state.df_stats
    aligned_seqs = st.session_state.aligned_seqs
    identity_df = st.session_state.identity_df
    prof_raw = st.session_state.prof_raw
    prof_trim = st.session_state.prof_trim
    offset = st.session_state.offset

    # ── PESTAÑAS ──
    (tab_stats, tab_kmer, tab_motif, tab_aln, tab_tree, tab_story) = st.tabs(
        ["Estadísticas", "k-mers", "Motivo", "Alineamiento", "Árbol & Visor", "Historia"]
    )

    # 1️⃣ Estadísticas
    with tab_stats:
        st.markdown(
            "**¿Por qué importa?**  \n"
            "Estas métricas básicas (longitud y %GC) son el primer control de calidad "
            "de un conjunto FASTA. Diferencias grandes pueden revelar secuencias "
            "contaminantes, fragmentadas o con sesgos de composición."
        )
        st.dataframe(df_stats)

        fig, ax = plt.subplots()
        ax.hist(df_stats["Longitud"], bins=20, color="#5563DE", edgecolor="#fff")
        ax.set_xlabel("Longitud (bp)")
        ax.set_ylabel("Nº de secuencias")
        st.pyplot(fig)

    # 2️⃣ k-mers - AHORA SE ACTUALIZA SIN RECARGAR TODO
    with tab_kmer:
        st.markdown(
            "**¿Para qué sirve?**  \n"
            "La distribución de k-mers revela firmas genómicas: secuencias sobre- o "
            "sub-representadas que pueden indicar sitios de restricción, preferencias "
            "de codón o regiones repetitivas."
        )
        k = st.number_input("Tamaño de k-mer", 1, 10, 3, key="kmer_len")
        
        # ¡Aquí está la clave! Se recalcula solo cuando cambia k
        dfk = pd.DataFrame(kmer_counts(seqs, k).most_common(20), columns=["k-mer", "Conteo"])
        st.dataframe(dfk)

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.bar(dfk["k-mer"], dfk["Conteo"], color="#74ABE2")
        plt.xticks(rotation=45)
        ax2.set_ylabel("Frecuencia")
        st.pyplot(fig2)

    # 3️⃣ Motivo - AHORA SE ACTUALIZA SIN RECARGAR TODO
    with tab_motif:
        st.markdown(
            "**Motivo / patrón**  \n"
            "Busca rápidamente un oligonucleótido (p. ej. sitio promotor) y muestra "
            "dónde aparece en cada secuencia."
        )
        motif = st.text_input("Motivo a buscar", key="motif_txt").upper()

        if motif:
            m = motif.upper()
            rows = [{"Especie": sp, "Ocurrencias": len(pos := find_motif_positions(s, m)), "Posiciones": pos}
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
            st.info("Escribe un motivo arriba.")

    # 4️⃣ Alineamiento
    with tab_aln:
        st.subheader("Matriz de Identidad entre Secuencias")
        st.markdown(
            "**Cálculo de identidad:** La matriz se calcula después del alineamiento múltiple, "
            "donde todas las secuencias tienen la misma longitud gracias a la inserción de gaps (`-`). "
            "Para cada par de secuencias, comparamos posición por posición:  \n"
            "- **Matches**: Cuando ambos caracteres son idénticos (misma base en ambas secuencias)  \n"
            "- **Posiciones válidas**: Columnas con al menos un nucleótido (se ignoran dobles gaps)  \n"
            "- **% Identidad** = (Nº matches / Nº posiciones válidas) × 100  \n"
            "Este método estándar garantiza comparaciones justas entre secuencias de diferentes longitudes originales."
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
        st.download_button("Descargar matriz completa como CSV", csv, "matriz_identidad.csv", "text/csv")
        st.subheader("Alineamiento múltiple")

        st.markdown(
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
        
        # Botón de descarga ANTES del visor
        st.download_button('📥 Descargar alineamiento', aln_txt, 'alignment.fa')
        
        # Visor del alineamiento
        components.html(alignment_html(aln_txt), height=400, scrolling=False)

        # ── Perfil de conservación ───────────────────────────
        st.subheader("📈 Perfil de Conservación")
        st.markdown(
            "**Interpretación:** Valores altos (→1.0) = posiciones muy conservadas; "
            "valores bajos (→0.0) = posiciones muy variables entre especies."
        )
        
        modo_prof = st.radio(
            "Modo de perfil",
            ("Ignorar gaps (original)", "Penalizar gaps + recorte"),
            horizontal=True,
            key="prof_mode"
        )
        if modo_prof.startswith("Ignorar"):
            prof_plot = prof_raw
            x_vals = np.arange(len(prof_raw))
            title_tag = "original"
        else:
            prof_plot = prof_trim
            x_vals = np.arange(offset, offset + len(prof_trim))
            title_tag = f"recortado (desde {offset})"

        mostrar_total = st.checkbox("Mostrar línea de 100% conservado", value=True, key="show_total")
        umbral = st.slider("Umbral de conservación", 0.0, 1.0, 0.5, 0.01, key="conservation_threshold")
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(x_vals, prof_plot, linewidth=0.25, label="Conservación")
        ax.fill_between(x_vals, prof_plot, alpha=0.3)
        ax.axhline(y=umbral, color='red', linestyle='--', linewidth=1, label=f"Umbral {umbral:.2f}")
        if mostrar_total:
            ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, label="100% conservado")

        ax.set_xlabel("Posición en el alineamiento")
        ax.set_ylabel("Fracción de conservación")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right")
        
        # Estadísticas rápidas
        ultra_conserved = (prof_plot > umbral).sum()
        highly_variable = (prof_plot < umbral).sum()
        ax.set_title(f'Perfil {title_tag}: {ultra_conserved} posiciones conservadas, '
                    f'{highly_variable} posiciones variables')
        ax.grid(True, linestyle='--', alpha=0.5)

        st.pyplot(fig)

        

    # 5️⃣ Árbol & Visor
    with tab_tree:
        st.markdown("**Árbol filogenético**  \n" "Visualiza las relaciones evolutivas inferidas...")
        components.html(phylocanvas_html(nwk), height=650, scrolling=False)

    # 6️⃣ Historia - AHORA SE ACTUALIZA SIN RECARGAR TODO
    with tab_story:
        st.header("📝 Historia de las secuencias")
        modo = st.radio("Modo de historia", ["Divulgativo", "Técnico"], key="story_mode")

        # ¡Se recalcula solo cuando cambia el modo!
        story_md = generate_story(df_stats, identity_df, prof_raw, nwk, modo)
        st.markdown(story_md)

        md_bytes = story_md.encode('utf-8')
        st.download_button("Descargar historia (.md)", md_bytes, "historia_cox4i1.md", "text/markdown")

if __name__ == "__main__":
    main()