import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network
import tempfile
import os

st.set_page_config(page_title="Real-Time Social Network Analyzer", layout="wide")

st.title("🕸️ Real-Time Social Network Analyzer & Structural Pattern Simulator")
st.write("Build and analyze social networks using pure graph theory — no datasets, no ML models.")

# -----------------------------------
# Initialize Graph
# -----------------------------------
if "G" not in st.session_state:
    st.session_state.G = nx.DiGraph()

G = st.session_state.G

# -----------------------------------
# Sidebar: Add Nodes
# -----------------------------------
st.sidebar.header("Add Actor (Node)")
node = st.sidebar.text_input("Actor Name")

if st.sidebar.button("Add Actor"):
    if node and node not in G.nodes:
        G.add_node(node)
        st.sidebar.success(f"Actor '{node}' added.")
    else:
        st.sidebar.error("Invalid or duplicate actor name.")

# -----------------------------------
# Sidebar: Add Edges
# -----------------------------------
st.sidebar.header("Add Relationship (Edge)")

source = st.sidebar.selectbox("Source Actor", [""] + list(G.nodes))
target = st.sidebar.selectbox("Target Actor", [""] + list(G.nodes))

weight = st.sidebar.number_input("Weight", value=1.0)
sign = st.sidebar.selectbox("Sign", ["Positive (+)", "Negative (-)", "Neutral (0)"])
directed = st.sidebar.checkbox("Directed Edge?", value=True)

if sign == "Positive (+)":
    s_val = 1
elif sign == "Negative (-)":
    s_val = -1
else:
    s_val = 0

if st.sidebar.button("Add Relationship"):
    if source and target and source != target:
        if directed:
            G.add_edge(source, target, weight=weight, sign=s_val)
        else:
            G.add_edge(source, target, weight=weight, sign=s_val)
            G.add_edge(target, source, weight=weight, sign=s_val)
        st.sidebar.success("Relationship added.")
    else:
        st.sidebar.error("Invalid source/target selection.")

# -----------------------------------
# Graph Visualization (FIXED VERSION)
# -----------------------------------
st.subheader("📌 Network Visualization")

net = Network(height="500px", width="100%", directed=True)
net.from_nx(G)

with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
    html_path = tmp_file.name
    net.show(html_path)

with open(html_path, "r", encoding="utf-8") as f:
    html_content = f.read()

st.components.v1.html(html_content, height=500, scrolling=True)

# -----------------------------------
# Adjacency Matrix
# -----------------------------------
st.subheader("📊 Adjacency Matrix")

if G.nodes:
    adj_df = nx.to_pandas_adjacency(G, dtype=int)
    st.dataframe(adj_df)

# -----------------------------------
# Centrality Measures
# -----------------------------------
st.subheader("📈 Centrality Measures")

if len(G.nodes) > 0:
    centrality_df = pd.DataFrame({
        "Degree": nx.degree_centrality(G),
        "Closeness": nx.closeness_centrality(G),
        "Betweenness": nx.betweenness_centrality(G),
        "Eigenvector": nx.eigenvector_centrality_numpy(G) if len(G) > 1 else {n: 0 for n in G.nodes}
    })

    st.dataframe(centrality_df)

# -----------------------------------
# Structural Equivalence (Correlation Matrix)
# -----------------------------------
st.subheader("🔍 Structural Equivalence Matrix")

if len(G.nodes) > 1:
    A = nx.to_numpy_array(G)
    sim = np.corrcoef(A)
    sim_df = pd.DataFrame(sim, index=G.nodes, columns=G.nodes)
    st.dataframe(sim_df)

# -----------------------------------
# Dyad Census
# -----------------------------------
st.subheader("👥 Dyad Census")

if nx.is_directed(G):
    mutual = sum(1 for u, v in G.edges if G.has_edge(v, u))
    asym = len(G.edges) - mutual
else:
    mutual = 0
    asym = len(G.edges)

null_dyads = len(G.nodes) * (len(G.nodes) - 1) - (mutual + asym)

st.write(f"**Mutual Dyads:** {mutual}")
st.write(f"**Asymmetric Dyads:** {asym}")
st.write(f"**Null Dyads:** {null_dyads}")

# -----------------------------------
# Triad Census
# -----------------------------------
st.subheader("🔺 Triad Census")

if len(G.nodes) >= 3 and nx.is_directed(G):
    triad = nx.triadic_census(G)
    st.json(triad)
else:
    st.info("Triad census works only for directed graphs with ≥ 3 nodes.")

# -----------------------------------
# Structural Balance Test
# -----------------------------------
st.subheader("⚖️ Structural Balance Test (Signed Graph)")

balanced = True
for triangle in nx.cycle_basis(G.to_undirected()):
    if len(triangle) == 3:
        edges = [(triangle[i], triangle[(i + 1) % 3]) for i in range(3)]
        product = 1
        for u, v in edges:
            if G.has_edge(u, v):
                product *= G[u][v].get("sign", 1)
        if product == -1:
            balanced = False

st.write("**Balanced:** ✔️ Yes" if balanced else "**Balanced:** ❌ No")

# -----------------------------------
# Simple Block Model
# -----------------------------------
st.subheader("🧩 Simple Block Model (Degree-Based Groups)")

degrees = dict(G.degree())
groups = {"High-Degree": [], "Low-Degree": []}

threshold = np.mean(list(degrees.values())) if degrees else 0

for node, deg in degrees.items():
    if deg >= threshold:
        groups["High-Degree"].append(node)
    else:
        groups["Low-Degree"].append(node)

st.write(groups)
