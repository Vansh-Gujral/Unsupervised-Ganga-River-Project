# """
# 🌊 AI + Satellite Enhanced River Pollution Detection Dashboard
# --------------------------------------------------------------
# Combines unsupervised ML on IoT river sensor data
# with satellite-derived environmental indicators (NDWI, NDVI, Temperature)
# to detect and localize pollution with higher accuracy.
# """

# # ======== IMPORTS ========
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import IsolationForest
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px

# # ======== APP CONFIG ========
# st.set_page_config(page_title="AI + Satellite River Pollution Detection", layout="wide")
# st.title("🛰️ AI + Satellite Enhanced River Pollution Detection System")
# st.markdown(
#     """
#     This system fuses **river sensor data** with **simulated satellite features**
#     (NDWI, NDVI, and Land Surface Temperature) to improve pollution detection accuracy.
#     The AI model uses **Unsupervised Learning (Isolation Forest + DBSCAN)** 
#     for real-time pollution identification and clustering.
#     """
# )

# # ======== FILE UPLOAD ========
# uploaded_file = st.file_uploader("📂 Upload your river sensor dataset (CSV)", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.success(f"✅ File uploaded successfully! Rows: {df.shape[0]} | Columns: {df.shape[1]}")
# else:
#     st.info("Using sample dataset (generated locally)...")
#     df = pd.read_csv("AI_Unsupervised_River_Pollution_Data_1000.csv")

# # ======== SIMULATE SATELLITE DATA INTEGRATION ========
# def simulate_satellite_features(df):
#     np.random.seed(42)
#     df["NDWI"] = np.random.uniform(0.2, 0.8, len(df))  # Water index
#     df["NDVI"] = np.random.uniform(0.1, 0.7, len(df))  # Vegetation index
#     df["Surface_Temperature"] = np.random.uniform(20, 35, len(df))  # °C
#     return df

# df = simulate_satellite_features(df)

# st.subheader("🌤️ Added Satellite-Derived Features:")
# st.write("**NDWI:** Water turbidity index | **NDVI:** Vegetation health | **Temperature:** Surface heating")
# st.dataframe(df.head(10))

# # ======== DATA PREPROCESSING ========
# features = df.drop(columns=["Date_Time", "Station_ID", "Latitude", "Longitude"], errors='ignore')
# features = features.fillna(features.mean())

# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)

# # ======== UNSUPERVISED ANOMALY DETECTION ========
# isolation_model = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
# isolation_model.fit(features_scaled)

# df["Anomaly_Score"] = isolation_model.decision_function(features_scaled)
# df["Anomaly_Flag"] = isolation_model.predict(features_scaled)
# df["Pollution_Event"] = df["Anomaly_Flag"].apply(lambda x: "Yes" if x == -1 else "No")

# num_anomalies = df[df["Pollution_Event"] == "Yes"].shape[0]
# st.metric(label="⚠️ Detected Pollution Events", value=num_anomalies)

# # ======== DBSCAN CLUSTERING ========
# anomalies = df[df["Pollution_Event"] == "Yes"].copy()
# anomaly_features = features_scaled[df["Pollution_Event"] == "Yes"]

# dbscan_model = DBSCAN(eps=2.5, min_samples=2)
# clusters = dbscan_model.fit_predict(anomaly_features)
# anomalies["Cluster_Label"] = clusters

# df = df.merge(
#     anomalies[["Date_Time", "Station_ID", "Cluster_Label"]],
#     on=["Date_Time", "Station_ID"], how="left"
# )
# df["Cluster_Label"] = df["Cluster_Label"].fillna(-1)

# # ======== TABS ========
# tab1, tab2, tab3, tab4 = st.tabs([
#     "📈 Pollution Timeline",
#     "🗺️ Cluster Map",
#     "🌤️ Satellite Analysis",
#     "📊 Correlation & Stats"
# ])

# # --- Timeline ---
# with tab1:
#     st.subheader("Pollution Timeline by Station")
#     station_list = df["Station_ID"].unique().tolist()
#     selected_station = st.selectbox("Select Station", station_list)
#     station_df = df[df["Station_ID"] == selected_station]

#     fig, ax = plt.subplots(figsize=(10, 4))
#     ax.plot(station_df["Date_Time"], station_df["pH"], color='blue', label="pH Level")
#     anomaly_points = station_df[station_df["Pollution_Event"] == "Yes"]
#     ax.scatter(anomaly_points["Date_Time"], anomaly_points["pH"], color='red', label="Pollution Event", s=60)
#     ax.set_title(f"Pollution Detection Timeline - {selected_station}")
#     ax.set_xlabel("Time")
#     ax.set_ylabel("pH Level")
#     ax.legend()
#     st.pyplot(fig)

# # --- Map ---
# with tab2:
#     st.subheader("🗺️ Pollution Clusters on Map (with Satellite Features)")
#     map_df = anomalies.copy()
#     map_df["Cluster_Label"] = map_df["Cluster_Label"].astype(int)
#     map_df["Tooltip"] = map_df["Station_ID"] + " | Cluster: " + map_df["Cluster_Label"].astype(str)

#     if not map_df.empty:
#         fig = px.scatter_mapbox(
#             map_df,
#             lat="Latitude",
#             lon="Longitude",
#             color="Cluster_Label",
#             size="Turbidity",
#             hover_name="Tooltip",
#             zoom=10,
#             height=500,
#             color_continuous_scale="rainbow"
#         )
#         fig.update_layout(mapbox_style="open-street-map")
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.warning("No clusters found yet.")

# # --- Satellite Feature Analysis ---
# with tab3:
#     st.subheader("🌤️ Satellite-Derived Feature Trends")
#     fig, ax = plt.subplots(1, 3, figsize=(15, 4))

#     sns.histplot(df["NDWI"], color="blue", ax=ax[0])
#     ax[0].set_title("NDWI Distribution (Water Index)")
#     sns.histplot(df["NDVI"], color="green", ax=ax[1])
#     ax[1].set_title("NDVI Distribution (Vegetation)")
#     sns.histplot(df["Surface_Temperature"], color="orange", ax=ax[2])
#     ax[2].set_title("Surface Temperature (°C)")

#     st.pyplot(fig)

#     st.markdown("**Observation:** Sudden drops in NDWI or spikes in temperature can indicate pollution or reduced water quality.")

# # --- Correlation ---
# with tab4:
#     st.subheader("📉 Correlation Heatmap (Including Satellite Features)")
#     numeric_df = df.select_dtypes(include=[np.number])
#     fig, ax = plt.subplots(figsize=(12, 6))
#     sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
#     st.pyplot(fig)

# # ======== DOWNLOAD RESULTS ========
# pollution_results = df[df["Pollution_Event"] == "Yes"]
# csv = pollution_results.to_csv(index=False).encode("utf-8")
# st.download_button(
#     label="💾 Download Detected Pollution Events (CSV)",
#     data=csv,
#     file_name="Detected_Pollution_Events_with_Satellite.csv",
#     mime="text/csv",
# )

# st.success("✅ Analysis Completed with Satellite Integration!")




# app.py
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import branca.colormap as cm
import streamlit.components.v1 as components

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="AI River Pollution — Maps & Animations", layout="wide")

# ------------------------------
# Utilities / Config
# ------------------------------
DATA_FILE = "RealisticSatelliteClusterData.csv"  # your saved dataset
OUT_FILE = "Clustered_River_Dataset.csv"

# Synthetic station -> coords mapping (update if you have real lat/lon)
STATION_COORDS = {
    "Station_1": (29.00, 78.00),
    "Station_2": (29.02, 78.02),
    "Station_3": (29.04, 78.04),
    "Station_4": (29.06, 78.06),
    "Station_5": (29.08, 78.08),
}

PALETTE = px.colors.qualitative.Safe  # for consistent coloring

# ------------------------------
# Load dataset
# ------------------------------
@st.cache_data
def load_data(path=DATA_FILE):
    df = pd.read_csv(path)
    # ensure Station_ID exists
    if "Station_ID" not in df.columns:
        df["Station_ID"] = np.random.choice(list(STATION_COORDS.keys()), size=len(df))
    # add Lat/Lon if missing
    if ("Latitude" not in df.columns) or ("Longitude" not in df.columns):
        lat = []
        lon = []
        for s in df["Station_ID"].astype(str):
            coords = STATION_COORDS.get(s, (29.05, 78.05))
            lat.append(coords[0] + np.random.normal(0, 0.0005))   # slight jitter
            lon.append(coords[1] + np.random.normal(0, 0.0005))
        df["Latitude"] = lat
        df["Longitude"] = lon
    return df

# ------------------------------
# DBSCAN (use your provided function)
# ------------------------------
def run_dbscan(df):
    features = df[["pH", "DO", "Temperature", "Turbidity", "Conductivity", "Nitrate"]]
    X = StandardScaler().fit_transform(features)

    db = DBSCAN(eps=0.9, min_samples=12)
    labels = db.fit_predict(X)

    df["Cluster_Label"] = labels
    st.write("🧩 Cluster Summary:")
    st.write(df["Cluster_Label"].value_counts())
    return df, X, labels

# ------------------------------
# Auto-labeling (simple rule-based)
# ------------------------------
def auto_label(df):
    cluster_labels = {}
    for c in df["Cluster_Label"].unique():
        if c == -1:
            cluster_labels[c] = "Noise / Outliers"
            continue
        sub = df[df["Cluster_Label"] == c]
        avg_turb = sub["Turbidity"].mean()
        avg_cond = sub["Conductivity"].mean()
        avg_nitrate = sub["Nitrate"].mean()

        if avg_cond > 800:
            cluster_labels[c] = "Chemical Pollution"
        elif avg_turb > 60:
            cluster_labels[c] = "Severe Contamination"
        elif avg_nitrate > 12:
            cluster_labels[c] = "Agricultural Runoff"
        elif avg_turb < 20 and avg_cond < 400:
            cluster_labels[c] = "Normal Water"
        else:
            cluster_labels[c] = "Unknown / Mixed Type"
    df["Auto_Label"] = df["Cluster_Label"].map(cluster_labels)
    st.write("🏷 Auto-label counts:")
    st.write(df["Auto_Label"].value_counts())
    return df

# ------------------------------
# Folium map generator
# ------------------------------
def folium_map(df):
    # center map on mean coords
    center = [df["Latitude"].mean(), df["Longitude"].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    # color map for labels
    unique_labels = df["Auto_Label"].unique().tolist()
    cmap = cm.linear.Set1_09.scale(0, max(1, len(unique_labels)))
    label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels)}

    marker_cluster = MarkerCluster().add_to(m)

    # add markers, size by turbidity, popup with info
    for _, row in df.iterrows():
        lab = row["Auto_Label"]
        color = label_to_color.get(lab, "#000000")
        popup_html = f"""
        <b>Station:</b> {row['Station_ID']}<br>
        <b>Cluster:</b> {row['Cluster_Label']}<br>
        <b>Auto Label:</b> {lab}<br>
        <b>pH:</b> {row['pH']:.2f}  <b>DO:</b> {row['DO']:.2f} <br>
        <b>Conductivity:</b> {row['Conductivity']:.1f}  <b>Turbidity:</b> {row['Turbidity']:.1f}
        """
        folium.CircleMarker(
            location=(row["Latitude"], row["Longitude"]),
            radius=max(4, min(12, row["Turbidity"] / 10)),  # size from turbidity
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(marker_cluster)

    # legend (basic)
    legend_html = "<div style='position: fixed; bottom: 50px; left: 50px; z-index:9999; background: white; padding: 10px; border-radius:5px;'>"
    for lab, col in label_to_color.items():
        legend_html += f"<div style='margin-bottom:4px;'><span style='background:{col};width:12px;height:12px;display:inline-block;margin-right:6px;'></span>{lab}</div>"
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

# ------------------------------
# Animated Plotly graphs
# ------------------------------
def prepare_animation_frames(df, n_per_frame=20):
    # create a 'frame' index so data progressively appears; preserves input order
    df2 = df.copy().reset_index(drop=True)
    df2["frame"] = (df2.index // n_per_frame).astype(int)
    return df2

def animated_conductivity_turbidity(df):
    df_anim = prepare_animation_frames(df, n_per_frame=15)
    fig = px.scatter(
        df_anim, x="Conductivity", y="Turbidity",
        color="Auto_Label",
        animation_frame="frame",
        hover_data=["Station_ID", "Cluster_Label"],
        title="Animated: Conductivity vs Turbidity (frames show data buildup)"
    )
    fig.update_traces(marker=dict(size=8))
    return fig

def animated_pca(df):
    # run PCA on the 6 core features
    feats = ["pH", "DO", "Temperature", "Turbidity", "Conductivity", "Nitrate"]
    X = StandardScaler().fit_transform(df[feats])
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    df_p = df.copy().reset_index(drop=True)
    df_p["PC1"] = pcs[:,0]
    df_p["PC2"] = pcs[:,1]
    df_p = prepare_animation_frames(df_p, n_per_frame=15)
    fig = px.scatter(
        df_p, x="PC1", y="PC2",
        color="Auto_Label",
        animation_frame="frame",
        hover_data=["Station_ID", "Cluster_Label"],
        title="Animated PCA (PC1 vs PC2)"
    )
    fig.update_traces(marker=dict(size=7))
    return fig

# ------------------------------
# Heatmap (numeric only)
# ------------------------------
def draw_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation (numeric features only)")
    return fig

# ------------------------------
# Main UI
# ------------------------------
st.title("🌊 AI River Pollution Detection — Maps & Animated Visuals")
st.markdown("DBSCAN clustering + satellite features + animated graphs + folium map")

df = load_data()
st.sidebar.header("Data & Clustering")
st.sidebar.write(f"Rows: {df.shape[0]}  Columns: {df.shape[1]}")

if st.sidebar.button("Run Clustering & Prepare Visuals"):
    with st.spinner("Running DBSCAN and labeling..."):
        df, X, labels = run_dbscan(df)
        df = auto_label(df)

    # show silhouette
    mask = labels != -1
    if mask.sum() > 0 and len(set(labels[mask])) > 1:
        sil = silhouette_score(X[mask], labels[mask])
        st.success(f"Silhouette Score (excl noise): {sil:.4f}")
    else:
        st.warning("Not enough clusters for silhouette score")

    # top row: cluster counts and download
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.subheader("Cluster Counts")
        st.write(df["Cluster_Label"].value_counts())
    with c2:
        st.subheader("Auto Labels")
        st.write(df["Auto_Label"].value_counts())
    with c3:
        st.subheader("Download")
        st.download_button("Download labeled dataset", data=df.to_csv(index=False), file_name=OUT_FILE, mime="text/csv")

    # folium map
    st.subheader("📍 Pollution Map (Satellite / Stations)")
    m = folium_map(df)
    # render folium map in streamlit
    folium_html = m._repr_html_()
    components.html(folium_html, height=600)

    # animated plotly
    st.subheader("🎞 Animated Conductivity vs Turbidity")
    fig_anim = animated_conductivity_turbidity(df)
    st.plotly_chart(fig_anim, use_container_width=True)

    st.subheader("🎞 Animated PCA (PC1 vs PC2)")
    fig_pca_anim = animated_pca(df)
    st.plotly_chart(fig_pca_anim, use_container_width=True)

    # static PCA + scatter + heatmap
    st.subheader("PCA 2D (static)")
    pca = PCA(n_components=2)
    pc = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(pc[:,0], pc[:,1], c=labels, cmap="tab10", s=50)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PCA 2D")
    st.pyplot(fig)

    st.subheader("Conductivity vs Turbidity (static)")
    fig2, ax2 = plt.subplots(figsize=(8,6))
    ax2.scatter(df["Conductivity"], df["Turbidity"], c=labels, cmap="tab10", s=40)
    ax2.set_xlabel("Conductivity"); ax2.set_ylabel("Turbidity")
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    st.pyplot(draw_heatmap(df))

    st.success("All visuals generated — scroll up!")

else:
    st.info("Click 'Run Clustering & Prepare Visuals' in the sidebar to compute clusters and generate maps/animations.")
    st.caption("The app will use 'RealisticSatelliteClusterData.csv' in the same folder. If missing, generate it first.")
