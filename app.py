import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def parse_fasta(file_contents):
    sequences = {}
    header = None
    seq_lines = []
    for line in file_contents.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if header:
                sequences[header] = ''.join(seq_lines)
            header = line[1:]
            seq_lines = []
        else:
            seq_lines.append(line.upper())
    if header:
        sequences[header] = ''.join(seq_lines)
    return sequences

# Levenshtein distance implementation
def levenshtein(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    dp = np.zeros((len_s1 + 1, len_s2 + 1), dtype=int)
    for i in range(len_s1 + 1):
        dp[i][0] = i
    for j in range(len_s2 + 1):
        dp[0][j] = j
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,    # deletion
                dp[i][j-1] + 1,    # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[len_s1][len_s2]

# Hamming distance (requires equal length)
def hamming(s1, s2):
    if len(s1) != len(s2):
        return None
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

# Compute k-mer counts across all sequences
def compute_kmer_counts(sequences, k):
    kmer_counts = Counter()
    for seq in sequences.values():
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmer_counts[kmer] += 1
    return kmer_counts

# Motif search: returns list of positions (1-based)
def find_motif_positions(seq, motif):
    positions = []
    m = len(motif)
    for i in range(len(seq) - m + 1):
        if seq[i:i+m] == motif:
            positions.append(i+1)
    return positions

# Streamlit app
def main():
    st.set_page_config(page_title="Genomics Dashboard", layout="wide")
    # Aesthetic CSS
    st.markdown("""
        <style>
        body {
            background-color: #f0f0f5;
            color: #333333;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
        .stApp {
            background-image: linear-gradient(135deg, #74ABE2 0%, #5563DE 100%);
        }
        .sidebar .sidebar-content {
            background: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📊 Panel Genómico Interactivo")
    st.sidebar.header("Parámetros de Entrada")
    uploaded_file = st.sidebar.file_uploader("Sube un archivo FASTA", type=['fa', 'fasta', 'txt'])
    k = st.sidebar.number_input("Tamaño de k-mer (k)", min_value=1, max_value=10, value=3, step=1)
    motif = st.sidebar.text_input("Motivo a buscar (ej: ATG)")
    calcular = st.sidebar.button("Calcular Estadísticas")

    if uploaded_file and calcular:
        # Leer archivo
        file_contents = uploaded_file.getvalue().decode('utf-8')
        sequences = parse_fasta(file_contents)
        if not sequences:
            st.error("No se encontraron secuencias en el archivo.")
            return
        # Número de secuencias
        num_seqs = len(sequences)
        st.subheader(f"Número de secuencias: {num_seqs}")

        # Longitud, GC, conteo bases
        stats = []
        for header, seq in sequences.items():
            length = len(seq)
            counts = Counter(seq)
            gc_count = counts.get('G', 0) + counts.get('C', 0)
            gc_content = gc_count / length * 100 if length > 0 else 0
            stats.append({
                'ID': header,
                'Longitud': length,
                '%GC': round(gc_content, 2),
                'A': counts.get('A', 0),
                'T': counts.get('T', 0),
                'G': counts.get('G', 0),
                'C': counts.get('C', 0),
                'N': counts.get('N', 0)
            })
        df_stats = pd.DataFrame(stats)
        st.subheader("📈 Estadísticas de Secuencias")
        st.dataframe(df_stats)

        # Distribución de longitudes
        st.subheader("📊 Distribución de Longitudes de Secuencias")
        fig_len, ax_len = plt.subplots()
        ax_len.hist(df_stats['Longitud'], bins=20, color='#5563DE', edgecolor='#ffffff')
        ax_len.set_xlabel('Longitud (bp)')
        ax_len.set_ylabel('Número de Secuencias')
        st.pyplot(fig_len)

        # K-mer distribución
        st.subheader(f"🔍 Distribución de {k}-mers")
        kmer_counts = compute_kmer_counts(sequences, k)
        if kmer_counts:
            top_kmers = kmer_counts.most_common(20)
            df_kmers = pd.DataFrame(top_kmers, columns=['k-mer', 'Conteo'])
            st.dataframe(df_kmers)
            fig_kmer, ax_kmer = plt.subplots(figsize=(8,4))
            ax_kmer.bar(df_kmers['k-mer'], df_kmers['Conteo'], color='#74ABE2')
            ax_kmer.set_xlabel('k-mer')
            ax_kmer.set_ylabel('Frecuencia')
            plt.xticks(rotation=45)
            st.pyplot(fig_kmer)
        else:
            st.write("No se pudieron calcular k-mers con los parámetros dados.")

        # Matriz de distancias
        st.subheader("📐 Matriz de Distancias")
        headers = list(sequences.keys())
        n = len(headers)
        # Preparar matrices
        mat_hamming = np.zeros((n, n), dtype=float)
        mat_lev = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                seq1 = sequences[headers[i]]
                seq2 = sequences[headers[j]]
                # Hamming: si diferente longitud, asignar NaN
                if len(seq1) == len(seq2):
                    mat_hamming[i, j] = hamming(seq1, seq2)
                else:
                    mat_hamming[i, j] = np.nan
                mat_lev[i, j] = levenshtein(seq1, seq2)
        df_hamming = pd.DataFrame(mat_hamming, index=headers, columns=headers)
        df_lev = pd.DataFrame(mat_lev, index=headers, columns=headers)

        st.write("**Distancia de Hamming** (NaN para longitudes distintas)")
        st.dataframe(df_hamming.fillna('—'))
        st.write("**Distancia de Levenshtein** (Edit Distance)")
        st.dataframe(df_lev)

        # Heatmaps
        st.write("**Mapa de calor: Hamming**")
        fig_hm, ax_hm = plt.subplots(figsize=(6,5))
        im = ax_hm.imshow(df_hamming.fillna(0), cmap='viridis')
        ax_hm.set_xticks(range(n)); ax_hm.set_xticklabels(headers, rotation=45, ha='right')
        ax_hm.set_yticks(range(n)); ax_hm.set_yticklabels(headers)
        fig_hm.colorbar(im, ax=ax_hm, orientation='vertical', label='Distancia')
        st.pyplot(fig_hm)

        st.write("**Mapa de calor: Levenshtein**")
        fig_lev, ax_lev = plt.subplots(figsize=(6,5))
        im2 = ax_lev.imshow(df_lev, cmap='magma')
        ax_lev.set_xticks(range(n)); ax_lev.set_xticklabels(headers, rotation=45, ha='right')
        ax_lev.set_yticks(range(n)); ax_lev.set_yticklabels(headers)
        fig_lev.colorbar(im2, ax=ax_lev, orientation='vertical', label='Distancia')
        st.pyplot(fig_lev)

        # Búsqueda de motivo
        if motif:
            st.subheader(f"🔎 Búsqueda de Motivo: '{motif}'")
            motif_results = []
            for header, seq in sequences.items():
                positions = find_motif_positions(seq, motif.upper())
                motif_results.append({'ID': header, 'Posiciones': positions, 'Conteo': len(positions)})
            df_motif = pd.DataFrame(motif_results)
            st.dataframe(df_motif)
        else:
            st.info("Ingresa un motivo en la barra lateral para buscar en las secuencias.")

if __name__ == "__main__":
    main()
