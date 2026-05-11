import streamlit as st          # Streamlit: Framework für die Web-App (Benutzeroberfläche)
import gspread                   # Kommunikation mit Google Sheets (Datenbank)
from google.oauth2.service_account import Credentials  # Sicherer Google-Login
import pandas as pd              # Datenanalyse: Tabellen einlesen, filtern, rechnen
import matplotlib.pyplot as plt  # Diagramme (für PDF-Export benötigt)
import matplotlib
matplotlib.use("Agg")            # Kein interaktives Fenster öffnen (wichtig auf Servern)
import plotly.express as px      # Interaktive Charts im Browser (hover, zoom, filter)
import plotly.graph_objects as go
from fpdf import FPDF            # PDF-Dateien generieren
import os                        # Betriebssystem-Interaktion (Dateien löschen)
import tempfile                  # Temporäre Dateien zwischenspeichern
import numpy as np               # Mathematische Operationen und Arrays

# ==========================================
# MACHINE LEARNING (ML) IMPORTS
# ==========================================
# scikit-learn ist die Standard-Bibliothek für ML in Python
from sklearn.neighbors import KNeighborsRegressor        # KNN: Vorhersage per Ähnlichkeit
from sklearn.tree import DecisionTreeClassifier, plot_tree  # Entscheidungsbaum + Visualisierung
from sklearn.linear_model import LinearRegression        # Lineares Regressionsmodell (Vergleich)
from sklearn.model_selection import train_test_split     # Daten in Training/Test aufteilen
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score  # Gütemasse
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Datenaufbereitung

# ==========================================
# 1. SETUP UND AUTHENTIFIZIERUNG (DATENBANK)
# ==========================================

# 'scopes' definiert, auf welche Google-Dienste wir zugreifen
scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# Beispieldaten für die automatische Sheet-Erstellung beim ersten Start
SAMPLE_DATA = [
    ["PlayerName","Age","Gender","Position","InjuryStatus","TrainingAttendanceRate","FitnessScore","Goals","PerformanceScore"],
    ["Luca Meier",24,"Male","Forward","Fit",92,88,18,84],
    ["Thomas Müller",28,"Male","Midfielder","Fit",85,79,9,76],
    ["Jonas Becker",31,"Male","Defender","Recovering",70,65,2,62],
    ["Kevin Wolf",19,"Male","Goalkeeper","Fit",95,91,0,80],
    ["Mario Huber",26,"Male","Forward","Injured",40,55,6,48],
    ["Nico Braun",22,"Male","Midfielder","Fit",88,82,12,79],
    ["Felix Schulz",29,"Male","Defender","Fit",78,74,3,71],
    ["Philipp Koch",33,"Male","Forward","Recovering",60,60,5,55],
    ["Stefan Vogel",21,"Male","Midfielder","Fit",90,85,10,81],
    ["Andreas Mayer",27,"Male","Goalkeeper","Fit",83,77,0,73],
    ["Leon Roth",23,"Male","Defender","Fit",87,83,4,78],
    ["Tobias Haas",30,"Male","Forward","Fit",76,72,14,74],
    ["Jan Krause",25,"Male","Midfielder","Injured",35,50,4,42],
    ["Sven Fischer",18,"Male","Forward","Fit",93,90,20,88],
    ["Markus Lorenz",32,"Male","Defender","Fit",72,68,1,64],
    ["Sarah König",24,"Female","Forward","Fit",91,87,16,83],
    ["Laura Bauer",27,"Female","Midfielder","Fit",84,80,8,77],
    ["Julia Hoffmann",22,"Female","Defender","Fit",89,85,3,80],
    ["Nina Richter",26,"Female","Goalkeeper","Recovering",65,62,0,58],
    ["Anna Werner",20,"Female","Forward","Fit",94,92,19,89],
    ["Lisa Schmitt",29,"Female","Midfielder","Injured",38,52,5,44],
    ["Marie Lange",23,"Female","Defender","Fit",86,81,2,77],
    ["Eva Lehmann",31,"Female","Forward","Fit",74,70,11,70],
    ["Sophie Neumann",25,"Female","Midfielder","Fit",88,84,9,80],
    ["Klara Zimmermann",28,"Female","Goalkeeper","Fit",80,76,0,72],
    ["Hanna Dietrich",21,"Female","Defender","Fit",92,89,4,84],
    ["Emma Schröder",33,"Female","Forward","Recovering",58,57,7,52],
    ["Mia Krüger",19,"Female","Midfielder","Fit",96,93,13,90],
    ["Clara Hartmann",30,"Female","Defender","Fit",71,67,1,63],
    ["Ida Baumann",26,"Female","Forward","Injured",42,54,8,46],
]

# try-except fängt Fehler ab, falls der Login zur Datenbank scheitert
try:
    import json, traceback as _tb
    # JSON-Roundtrip: wandelt tomlkit-Objekte (Streamlit Cloud) in echte Python-Strings um
    creds_info = json.loads(json.dumps(dict(st.secrets["google_credentials"])))
    client = gspread.service_account_from_dict(creds_info, scopes=scopes)

    # Sheet per ID öffnen – ID aus Secrets lesen (beide TOML-Positionen werden unterstützt)
    spreadsheet_id = (
        st.secrets.get("spreadsheet_id")
        or st.secrets.get("google_credentials", {}).get("spreadsheet_id")
    )
    sheet = client.open_by_key(spreadsheet_id).sheet1

    # Records lesen – wenn leer (auch bei leeren Dummy-Zeilen), Sheet bereinigen und Beispieldaten schreiben
    data = sheet.get_all_records()
    if not data:
        sheet.clear()                  # Leere Zeilen/Formatierung entfernen
        sheet.update("A1", SAMPLE_DATA)
        data = sheet.get_all_records() # Nach dem Schreiben neu einlesen

    # Pandas DataFrame: wie eine Excel-Tabelle in Python – ermöglicht schnelles Filtern und Rechnen
    df = pd.DataFrame(data)

    # Numerische Spalten explizit konvertieren (gspread liefert manchmal Strings zurück)
    for _col in ["Age", "TrainingAttendanceRate", "FitnessScore", "Goals", "PerformanceScore"]:
        if _col in df.columns:
            df[_col] = pd.to_numeric(df[_col], errors="coerce")
except Exception as e:
    st.error(f"Datenbankfehler [{type(e).__name__}]: {e}")
    with st.expander("🔍 Details (für Debugging)"):
        st.code(_tb.format_exc())
    st.stop()

# ==========================================
# SEITEN-NAVIGATION (SESSION STATE)
# ==========================================
# session_state merkt sich Variablen zwischen Streamlit-Neustarts (z.B. aktuelle Seite)
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ==========================================
# 2. HOME PAGE (STARTSEITE)
# ==========================================
if st.session_state.page == "Home":
    st.title("⚽ Football Club Performance Monitor")

    # --- Problemformulierung ---
    # Anforderung 1: Klares Business-Problem beschreiben
    st.markdown("""
    ### Das Problem
    Fussballvereine verwalten Dutzende von Spielerdaten – Fitness, Trainingsbeteiligung,
    Verletzungsstatus, Tore – oft in Excel-Tabellen ohne systematische Auswertung.
    Trainerentscheidungen (Aufstellung, Regeneration, Talentförderung) basieren dadurch
    häufig auf Bauchgefühl statt auf Daten.

    ### Die Lösung
    Diese App verbindet eine **live Google-Sheets-Datenbank** mit interaktiven Visualisierungen
    und **Machine Learning**, um Trainer bei datengetriebenen Entscheidungen zu unterstützen:

    - Wer ist leistungsschwach und braucht gezieltes Training?
    - Welche Spieler sind verletzungsgefährdet?
    - Wie wird sich ein Neuverpflichteter basierend auf seinem Profil entwickeln?
    """)

    st.write("---")
    st.write("### Navigiere durch die Anwendung:")

    # 5 Navigationsspalten für alle Seiten
    col1, col2, col3, col4, col5 = st.columns(5)

    # st.button() reagiert auf Klick, st.rerun() lädt die neue Seite sofort
    with col1:
        if st.button("📊 Dashboard"):
            st.session_state.page = "Dashboard"
            st.rerun()
    with col2:
        if st.button("🤖 Machine Learning"):
            st.session_state.page = "Machine Learning"
            st.rerun()
    with col3:
        if st.button("🗃️ Data Management"):
            st.session_state.page = "Data Management"
            st.rerun()
    with col4:
        if st.button("📄 Player Report"):
            st.session_state.page = "Player Report"
            st.rerun()
    with col5:
        if st.button("👥 Team"):
            st.session_state.page = "Contribution"
            st.rerun()

# ==========================================
# 3. DASHBOARD PAGE (DATENVISUALISIERUNG)
# ==========================================
elif st.session_state.page == "Dashboard":
    st.title("📊 Player Performance Dashboard")

    if st.button("⬅️ Homepage"):
        st.session_state.page = "Home"
        st.rerun()

    if df.empty:
        st.warning("Keine Daten vorhanden. Bitte füge zuerst Spieler unter 'Data Management' hinzu.")
        st.stop()

    # --- Club Overview: Kennzahlen auf einen Blick ---
    st.subheader("Club Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Players", len(df))
    if "PerformanceScore" in df.columns:
        col2.metric("Avg Performance", round(df["PerformanceScore"].mean(), 1))
    if "TrainingAttendanceRate" in df.columns:
        col3.metric("Avg Attendance", f"{round(df['TrainingAttendanceRate'].mean(), 1)}%")
    if "Goals" in df.columns:
        col4.metric("Total Goals", int(df["Goals"].sum()))

    st.write("---")

    # --- Diagramm 1: Altersverteilung (Histogramm) ---
    # px.histogram() erstellt ein interaktives Histogramm – Hover zeigt genaue Werte
    st.subheader("Altersverteilung")
    fig1 = px.histogram(df, x="Age", nbins=20, title="Age Distribution",
                        color_discrete_sequence=["#4A90D9"],
                        labels={"Age": "Alter", "count": "Anzahl Spieler"})
    fig1.update_layout(bargap=0.05)
    st.plotly_chart(fig1, use_container_width=True)

    # --- Diagramm 2: Spieler pro Position (Balkendiagramm) ---
    st.subheader("Spieler pro Position")
    pos_counts = df["Position"].value_counts().reset_index()
    pos_counts.columns = ["Position", "Anzahl"]
    fig2 = px.bar(pos_counts, x="Position", y="Anzahl", title="Players by Position",
                  color="Position", text="Anzahl")
    fig2.update_traces(textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    # --- Diagramm 3: Tore nach Position (Boxplot) ---
    # Boxplot zeigt Minimum, Maximum, Median und Ausreisser pro Position
    st.subheader("Tore nach Position")
    fig3 = px.box(df, x="Position", y="Goals", title="Goals Distribution by Position",
                  color="Position", points="all")
    st.plotly_chart(fig3, use_container_width=True)

    # --- Diagramm 4: Leistung vs. Trainingsbeteiligung (Streudiagramm + Trendlinie) ---
    st.subheader("Leistung vs. Trainingsbeteiligung")
    fig4 = px.scatter(df, x="TrainingAttendanceRate", y="PerformanceScore",
                      color="Position", hover_data=["Age"],
                      trendline="ols",  # OLS = Ordinary Least Squares (lineare Trendlinie)
                      title="Performance vs. Training Attendance",
                      labels={"TrainingAttendanceRate": "Training (%)", "PerformanceScore": "Performance"})
    st.plotly_chart(fig4, use_container_width=True)

    # --- Diagramm 5: Fitness vs. Alter ---
    st.subheader("Fitness vs. Alter")
    fig5 = px.scatter(df, x="Age", y="FitnessScore",
                      color="Position", hover_data=["Goals"],
                      trendline="ols",
                      title="Fitness Score vs. Age",
                      labels={"Age": "Alter", "FitnessScore": "Fitness Score"})
    st.plotly_chart(fig5, use_container_width=True)

    # --- Diagramm 6: Verletzungsstatus ---
    st.subheader("Verletzungsstatus im Team")
    inj_counts = df["InjuryStatus"].value_counts().reset_index()
    inj_counts.columns = ["Status", "Anzahl"]
    fig6 = px.pie(inj_counts, values="Anzahl", names="Status",
                  title="Injury Status Distribution",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig6, use_container_width=True)

# ==========================================
# 4. MACHINE LEARNING PAGE (KI-VORHERSAGEN)
# ==========================================
elif st.session_state.page == "Machine Learning":
    st.title("🤖 Player Performance Predictor")
    st.write(
        "Diese Seite nutzt **Machine Learning**, um die Leistung eines Spielers vorherzusagen "
        "und ihn in eine Leistungsklasse einzuteilen. Wir trainieren drei Modelle und vergleichen sie."
    )
    if st.button("⬅️ Homepage"):
        st.session_state.page = "Home"
        st.rerun()

    if df.empty:
        st.warning("Keine Daten vorhanden. Bitte füge zuerst Spieler unter 'Data Management' hinzu.")
        st.stop()

    # Spalten-Definitionen: Was gibt das Modell rein, was soll es lernen?
    CATEGORICAL = ["InjuryStatus", "Position", "Gender"]   # Text-Spalten
    NUMERICAL   = ["Age", "TrainingAttendanceRate", "FitnessScore", "Goals"]  # Zahlen-Spalten
    FEATURES    = NUMERICAL + CATEGORICAL
    TARGET      = "PerformanceScore"

    # Zeilen mit fehlenden Werten entfernen (ML-Modelle brauchen vollständige Daten)
    ml_df = df[FEATURES + [TARGET]].dropna().copy()

    if len(ml_df) < 10:
        st.warning("Zu wenig Daten vorhanden (min. 10 Spieler nötig).")
        st.stop()

    # Text in Zahlen umwandeln (LabelEncoder): z.B. "Torwart" → 0, "Stürmer" → 1
    encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col].astype(str))
        encoders[col] = le

    X = ml_df[FEATURES]
    y = ml_df[TARGET]

    # 80% Trainingsdaten / 20% Testdaten – Testdaten dienen zur unabhängigen Bewertung
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling: Alle Zahlenwerte auf ähnliche Grössenordnung bringen,
    # damit grosse Werte (z.B. Alter 30) nicht kleinere (z.B. Tore 5) dominieren
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # -------------------------------------------------------
    # MODELL 1: K-Nearest Neighbors (KNN) – Regression
    # -------------------------------------------------------
    # KNN sucht die 5 ähnlichsten Spieler und nimmt ihren Durchschnittswert als Vorhersage
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train_sc, y_train)
    knn_pred = knn.predict(X_test_sc)
    knn_r2  = r2_score(y_test, knn_pred)
    knn_mae = mean_absolute_error(y_test, knn_pred)

    # -------------------------------------------------------
    # MODELL 2: Lineare Regression – Vergleichsmodell (Baseline)
    # -------------------------------------------------------
    # Lineare Regression sucht die beste Gerade durch die Datenpunkte
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    lr_r2  = r2_score(y_test, lr_pred)
    lr_mae = mean_absolute_error(y_test, lr_pred)

    # -------------------------------------------------------
    # MODELL 3: Decision Tree – Klassifikation (High/Medium/Low)
    # -------------------------------------------------------
    def categorise(score):
        """Teilt einen numerischen Score in drei Leistungsklassen ein."""
        if score >= 75: return "High"
        elif score >= 50: return "Medium"
        else: return "Low"

    ml_df["PerformanceCategory"] = ml_df[TARGET].apply(categorise)
    y_dt = ml_df["PerformanceCategory"]

    # stratify stellt sicher, dass alle drei Klassen gleichmässig in Train/Test verteilt sind
    X_dt_train, X_dt_test, y_dt_train, y_dt_test = train_test_split(
        ml_df[FEATURES], y_dt, test_size=0.2, random_state=42, stratify=y_dt
    )
    # max_depth=4: Der Baum darf maximal 4 Fragen tief werden (verhindert Overfitting)
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_dt_train, y_dt_train)
    dt_acc = accuracy_score(y_dt_test, dt.predict(X_dt_test))

    # -------------------------------------------------------
    # MODELLVERGLEICH
    # -------------------------------------------------------
    st.subheader("📊 Modellvergleich")
    st.write("Drei Modelle im direkten Vergleich – R² misst die Erklärungskraft (1.0 = perfekt), MAE den durchschnittlichen Fehler in Punkten.")

    comp_col1, comp_col2, comp_col3 = st.columns(3)
    with comp_col1:
        st.markdown("**K-Nearest Neighbors**")
        st.metric("R² Score", f"{knn_r2:.2f}")
        st.metric("Ø Fehler (MAE)", f"{knn_mae:.1f} Pkt")
        st.metric("Trainings-Datensätze", len(X_train))
    with comp_col2:
        st.markdown("**Lineare Regression**")
        st.metric("R² Score", f"{lr_r2:.2f}")
        st.metric("Ø Fehler (MAE)", f"{lr_mae:.1f} Pkt")
        st.metric("Trainings-Datensätze", len(X_train))
    with comp_col3:
        st.markdown("**Decision Tree**")
        st.metric("Accuracy", f"{dt_acc:.0%}")
        st.metric("Klassen", "High / Medium / Low")
        st.metric("Max Tiefe", "4")

    # Balkendiagramm: R²-Vergleich KNN vs. Linear Regression
    fig_comp = go.Figure(data=[
        go.Bar(name="KNN",               x=["R²-Score"], y=[knn_r2], marker_color="#4A90D9"),
        go.Bar(name="Lineare Regression", x=["R²-Score"], y=[lr_r2],  marker_color="#E8A838"),
    ])
    fig_comp.update_layout(title="R²-Score Vergleich: KNN vs. Lineare Regression",
                           yaxis=dict(range=[0, 1]), barmode="group")
    st.plotly_chart(fig_comp, use_container_width=True)

    # -------------------------------------------------------
    # LIVE-VORHERSAGE
    # -------------------------------------------------------
    st.markdown("---")
    st.subheader("🎯 Teste es selbst: Spieler-Vorhersage")
    st.write("Stelle einen fiktiven Spieler zusammen – beide Modelle sagen sofort seine Leistung voraus.")

    col1, col2 = st.columns(2)
    with col1:
        input_age        = st.slider("Alter",                    15, 40, 25)
        input_attendance = st.slider("Trainingsbeteiligung (%)", 0.0, 100.0, 80.0)
        input_fitness    = st.slider("Fitness Score",            0.0, 100.0, 75.0)
        input_goals      = st.slider("Tore",                     0, 50, 5)
    with col2:
        input_injury_label   = st.selectbox("Verletzungsstatus", encoders["InjuryStatus"].classes_.tolist())
        input_position_label = st.selectbox("Position",          encoders["Position"].classes_.tolist())
        input_gender_label   = st.selectbox("Geschlecht",        encoders["Gender"].classes_.tolist())

    # Text-Auswahl zurück in Zahlen übersetzen, damit das Modell sie verarbeiten kann
    input_injury   = encoders["InjuryStatus"].transform([input_injury_label])[0]
    input_position = encoders["Position"].transform([input_position_label])[0]
    input_gender   = encoders["Gender"].transform([input_gender_label])[0]

    input_array  = np.array([[input_age, input_attendance, input_fitness, input_goals,
                               input_injury, input_position, input_gender]])
    input_scaled = scaler.transform(input_array)

    knn_result = knn.predict(input_scaled)[0]
    lr_result  = lr.predict(input_scaled)[0]
    dt_result  = dt.predict(input_array)[0]
    dt_proba   = dt.predict_proba(input_array)[0]

    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.success(f"**KNN:** {knn_result:.1f} / 100")
    res_col2.success(f"**Lin. Regression:** {lr_result:.1f} / 100")
    res_col3.success(f"**Decision Tree:** {dt_result} Performer")

    # Konfidenz-Balken: zeigt, wie sicher der Decision Tree bei seiner Klassifikation ist
    st.write("**Decision Tree Konfidenz:**")
    proba_df = pd.DataFrame({
        "Klasse": dt.classes_,
        "Wahrscheinlichkeit": dt_proba
    }).sort_values("Wahrscheinlichkeit", ascending=False)
    fig_proba = px.bar(proba_df, x="Klasse", y="Wahrscheinlichkeit",
                       color="Klasse", text=proba_df["Wahrscheinlichkeit"].apply(lambda x: f"{x:.0%}"),
                       range_y=[0, 1], title="Konfidenz der Klassifikation")
    fig_proba.update_traces(textposition="outside")
    st.plotly_chart(fig_proba, use_container_width=True)

    # -------------------------------------------------------
    # ENTSCHEIDUNGSBAUM-VISUALISIERUNG
    # -------------------------------------------------------
    st.markdown("---")
    st.subheader("🌳 Entscheidungsbaum (Decision Tree) visualisiert")
    st.write("Hier siehst du, wie die KI intern 'denkt' – welche Fragen sie stellt, um einen Spieler zu klassifizieren.")
    fig_tree, ax_tree = plt.subplots(figsize=(22, 8))
    plot_tree(dt, feature_names=FEATURES, class_names=dt.classes_,
              filled=True, rounded=True, fontsize=9, ax=ax_tree)
    st.pyplot(fig_tree)

# ==========================================
# 5. DATA MANAGEMENT PAGE (DATENBANK ÄNDERN)
# ==========================================
elif st.session_state.page == "Data Management":
    st.title("🗃️ Data Input Manager")
    st.write("Spieler hinzufügen, bearbeiten oder löschen – alle Änderungen sind sofort live im Google Sheet.")

    if "Backend_subpage" not in st.session_state:
        st.session_state.Backend_subpage = "main"

    headers     = sheet.row_values(1)       # Spalten-Namen aus Zeile 1
    sheet_data  = sheet.get_all_values()    # Alle Daten inkl. Header

    if st.session_state.Backend_subpage == "main":
        if st.button("⬅️ Homepage"):
            st.session_state.page = "Home"
            st.rerun()

        st.subheader("Wähle eine Aktion:")
        if st.button("➕ Neuen Spieler hinzufügen"):
            st.session_state.Backend_subpage = "add_player"
            st.rerun()
        if st.button("✏️ Spieler bearbeiten"):
            st.session_state.Backend_subpage = "edit_player"
            st.rerun()
        if st.button("🗑️ Spieler löschen"):
            st.session_state.Backend_subpage = "delete_player"
            st.rerun()

    # --- SPIELER HINZUFÜGEN ---
    elif st.session_state.Backend_subpage == "add_player":
        st.subheader("Neuen Spieler eintragen")

        # st.form() schickt alle Eingaben erst beim Klick auf den Submit-Button ab
        with st.form("add_player_form"):
            name       = st.text_input("Name (Optional)")
            age        = st.number_input("Alter",                       min_value=15,  max_value=50,    value=25)
            gender     = st.selectbox("Geschlecht",                     ["Male", "Female"])
            position   = st.selectbox("Position",                       ["Forward", "Midfielder", "Defender", "Goalkeeper"])
            injury_status = st.selectbox("Verletzungsstatus",           ["Fit", "Injured", "Recovering"])
            attendance = st.number_input("Trainingsbeteiligung (%)",    min_value=0.0, max_value=100.0, value=80.0)
            fitness    = st.number_input("Fitness Score",               min_value=0.0, max_value=100.0, value=75.0)
            goals      = st.number_input("Tore",                        min_value=0,                   value=0)
            performance = st.number_input("Performance Score",          min_value=0.0, max_value=100.0, value=50.0)

            if st.form_submit_button("Speichern in Datenbank"):
                # Neue Zeile in der Reihenfolge der Google-Sheet-Spalten aufbauen
                new_row = []
                for header in headers:
                    if   header == "Age":                    new_row.append(age)
                    elif header == "Gender":                 new_row.append(gender)
                    elif header == "Position":               new_row.append(position)
                    elif header == "InjuryStatus":           new_row.append(injury_status)
                    elif header == "TrainingAttendanceRate": new_row.append(attendance)
                    elif header == "FitnessScore":           new_row.append(fitness)
                    elif header == "Goals":                  new_row.append(goals)
                    elif header == "PerformanceScore":       new_row.append(performance)
                    elif header == "PlayerName":             new_row.append(name)
                    else:                                    new_row.append("")
                sheet.append_row(new_row)
                st.success("Spieler erfolgreich gespeichert!")

        if st.button("⬅️ Zurück"):
            st.session_state.Backend_subpage = "main"
            st.rerun()

    # --- SPIELER BEARBEITEN ---
    elif st.session_state.Backend_subpage == "edit_player":
        st.subheader("Spielerdaten anpassen")

        if "PlayerName" in headers:
            idx_name    = headers.index("PlayerName")
            player_list = [f"Reihe {i+2} - {row[idx_name]}" for i, row in enumerate(sheet_data[1:])]
        else:
            player_list = [f"Reihe {i+2}" for i in range(len(sheet_data) - 1)]

        selected_player = st.selectbox("Wähle einen Spieler", player_list)

        if selected_player:
            row_num      = int(selected_player.split(" ")[1])
            current_data = sheet_data[row_num - 1]

            with st.form("edit_player_form"):
                updated_data = []
                for i, header in enumerate(headers):
                    val = st.text_input(header, value=current_data[i])
                    updated_data.append(val)

                if st.form_submit_button("Update durchführen"):
                    # Gspread: Überschreibt genau diese Zeile mit den aktualisierten Werten
                    sheet.update(f"A{row_num}", [updated_data])
                    st.success("Spielerdaten erfolgreich aktualisiert!")

        if st.button("⬅️ Zurück"):
            st.session_state.Backend_subpage = "main"
            st.rerun()

    # --- SPIELER LÖSCHEN ---
    elif st.session_state.Backend_subpage == "delete_player":
        st.subheader("Spieler löschen")

        if "PlayerName" in headers:
            idx_name    = headers.index("PlayerName")
            player_list = [f"Reihe {i+2} - {row[idx_name]}" for i, row in enumerate(sheet_data[1:])]
        else:
            player_list = [f"Reihe {i+2}" for i in range(len(sheet_data) - 1)]

        selected_player = st.selectbox("Spieler zum Löschen wählen", player_list)

        if selected_player:
            row_num = int(selected_player.split(" ")[1])
            st.warning(f"Bist du sicher, dass du {selected_player} löschen willst? Das kann nicht rückgängig gemacht werden.")
            if st.button("🚨 Unwiderruflich löschen"):
                sheet.delete_rows(row_num)
                st.success("Spieler gelöscht!")

        if st.button("⬅️ Zurück"):
            st.session_state.Backend_subpage = "main"
            st.rerun()

# ==========================================
# 6. PLAYER REPORT PAGE (PDF GENERIERUNG)
# ==========================================
elif st.session_state.page == "Player Report":
    st.title("📄 Player Report")
    st.write("Erstelle einen professionellen Bericht für einen einzelnen Spieler und lade ihn als PDF herunter.")

    if st.button("⬅️ Homepage"):
        st.session_state.page = "Home"
        st.rerun()

    st.markdown("---")

    # Spieler-Auswahl
    if "PlayerName" in df.columns:
        player_labels = df["PlayerName"].astype(str).tolist()
    else:
        player_labels = [f"Player {i+1} | {df.iloc[i].get('Position','?')} | Age {df.iloc[i].get('Age','?')}" for i in range(len(df))]

    selected_label = st.selectbox("Spieler auswählen", player_labels)
    selected_idx   = player_labels.index(selected_label)
    player         = df.iloc[selected_idx]

    # Kerndaten als Metrics anzeigen
    st.subheader(f"Übersicht — {selected_label}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Performance Score",   f"{player.get('PerformanceScore', 'N/A')}")
    c2.metric("Fitness Score",       f"{player.get('FitnessScore', 'N/A')}")
    c3.metric("Training Attendance", f"{player.get('TrainingAttendanceRate', 'N/A')}%")
    c4.metric("Goals",               f"{player.get('Goals', 'N/A')}")

    # Vergleich Spieler vs. Team-Durchschnitt (interaktiv mit Plotly)
    st.subheader("Vergleich vs. Team-Durchschnitt")
    num_cols  = ["PerformanceScore", "FitnessScore", "TrainingAttendanceRate", "Goals"]
    available = [c for c in num_cols if c in df.columns]

    player_vals = [float(player.get(c, 0)) for c in available]
    team_avg    = [float(df[c].mean())      for c in available]

    fig_report = go.Figure(data=[
        go.Bar(name="Dieser Spieler",  x=available, y=player_vals, marker_color="#4A90D9"),
        go.Bar(name="Team Durchschnitt", x=available, y=team_avg,  marker_color="#D3D3D3"),
    ])
    fig_report.update_layout(barmode="group", title="Spieler vs. Team Durchschnitt")
    st.plotly_chart(fig_report, use_container_width=True)

    st.markdown("---")
    st.subheader("Als PDF exportieren")

    if st.button("PDF Bericht generieren"):
        try:
            # FPDF zeichnet das PDF wie mit einem Stempel auf Papier (x/y-Koordinaten)
            pdf = FPDF()
            pdf.add_page()

            # Header-Block (dunkelgrün)
            pdf.set_fill_color(20, 83, 45)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Arial", "B", 18)
            pdf.cell(0, 14, "Football Club -- Player Report", ln=True, fill=True, align="C")
            pdf.ln(4)

            # Spielerinfo
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", "B", 13)
            pdf.cell(0, 9, f"Player: {selected_label}", ln=True)
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 7, f"Position: {player.get('Position','N/A')} | Age: {player.get('Age','N/A')}", ln=True)
            pdf.ln(4)

            # Statistik-Tabelle
            pdf.set_font("Arial", "B", 12)
            pdf.set_fill_color(220, 240, 220)
            pdf.cell(0, 8, "Performance Metrics", ln=True, fill=True)
            pdf.set_font("Arial", "", 11)
            stats = {
                "Performance Score":    player.get("PerformanceScore", "N/A"),
                "Fitness Score":        player.get("FitnessScore", "N/A"),
                "Training Attendance":  f"{player.get('TrainingAttendanceRate','N/A')}%",
                "Goals":                player.get("Goals", "N/A"),
            }
            for label, val in stats.items():
                pdf.cell(80, 7, label, border=1)
                pdf.cell(0, 7, str(val), border=1, ln=True)
            pdf.ln(4)

            # Vergleichs-Diagramm als Bild ins PDF einfügen
            # (Plotly-Charts können nicht direkt eingebettet werden, daher matplotlib für den Export)
            fig_pdf, ax_pdf = plt.subplots(figsize=(7, 3))
            x = np.arange(len(available))
            ax_pdf.bar(x - 0.175, player_vals, 0.35, label="This Player",   color="steelblue")
            ax_pdf.bar(x + 0.175, team_avg,    0.35, label="Team Average",  color="lightgray")
            ax_pdf.set_xticks(x)
            ax_pdf.set_xticklabels(available, rotation=15)
            ax_pdf.legend()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                fig_pdf.savefig(tmp_path, bbox_inches="tight", dpi=120)
                plt.close(fig_pdf)

            pdf.set_font("Arial", "B", 12)
            pdf.set_fill_color(220, 240, 220)
            pdf.cell(0, 8, "Comparison Chart", ln=True, fill=True)
            pdf.image(tmp_path, x=15, w=170)
            os.unlink(tmp_path)  # Temporäre Bilddatei nach dem Einbetten löschen

            # Footer
            pdf.set_y(-15)
            pdf.set_font("Arial", "I", 8)
            pdf.set_text_color(120, 120, 120)
            pdf.cell(0, 10, "Generiert durch den Football Club Performance Monitor", align="C")

            pdf_bytes = pdf.output(dest="S").encode("latin-1")
            st.download_button(
                label="📥 Download PDF",
                data=pdf_bytes,
                file_name=f"player_report_{selected_label.replace(' ','_')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Fehler bei der PDF Generierung: {e}")

# ==========================================
# 7. CONTRIBUTION MATRIX (ANFORDERUNG 7)
# ==========================================
elif st.session_state.page == "Contribution":
    st.title("👥 Team & Contribution Matrix")
    st.write("Übersicht über die Beiträge der einzelnen Teammitglieder gemäss Projektanforderung.")

    if st.button("⬅️ Homepage"):
        st.session_state.page = "Home"
        st.rerun()

    st.markdown("---")

    # --- Teammitglieder ---
    st.subheader("Das Team")
    team_col1, team_col2, team_col3 = st.columns(3)
    with team_col1:
        st.markdown("### Teammitglied 1")
        st.write("Projektleitung, Google Sheets Integration, Authentifizierung, Deployment auf Streamlit Cloud")
    with team_col2:
        st.markdown("### Teammitglied 2")
        st.write("Machine Learning Modelle (KNN, Decision Tree), Feature Engineering, Modellbewertung")
    with team_col3:
        st.markdown("### Teammitglied 3")
        st.write("Dashboard Visualisierungen, Player Report, PDF-Export, UI-Design")

    st.markdown("---")

    # --- Contribution Matrix als Tabelle ---
    st.subheader("Contribution Matrix")
    st.write("Jede Anforderung des Professors und wer dafür verantwortlich war:")

    # DataFrame für die Tabelle: Zeilen = Anforderungen, Spalten = Teammitglieder
    matrix_data = {
        "Anforderung": [
            "1. Problemformulierung",
            "2. API / Datenbank (Google Sheets)",
            "3. Datenvisualisierung (Dashboard)",
            "4. Benutzerinteraktion (CRUD + ML-Slider)",
            "5. Machine Learning (KNN, LinReg, Decision Tree)",
            "6. Code-Kommentare",
            "7. Contribution Matrix",
            "8. Präsentationsvideo",
        ],
        "Teammitglied 1": ["✅", "✅", "",   "✅", "",   "✅", "✅", "✅"],
        "Teammitglied 2": ["",   "",   "",   "",   "✅", "✅", "",   "✅"],
        "Teammitglied 3": ["✅", "",   "✅", "✅", "",   "✅", "",   "✅"],
    }

    matrix_df = pd.DataFrame(matrix_data)
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- Aufwandsverteilung als Tortendiagramm ---
    st.subheader("Geschätzter Arbeitsaufwand")
    effort_fig = px.pie(
        values=[35, 35, 30],
        names=["Teammitglied 1", "Teammitglied 2", "Teammitglied 3"],
        title="Arbeitsaufwand-Verteilung (geschätzt)",
        color_discrete_sequence=["#4A90D9", "#E8A838", "#50C878"]
    )
    st.plotly_chart(effort_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Technologie-Stack")
    tech_col1, tech_col2 = st.columns(2)
    with tech_col1:
        st.markdown("""
        **Backend / Daten**
        - Python 3.12
        - Google Sheets API (gspread 6.x)
        - Pandas, NumPy

        **Machine Learning**
        - scikit-learn
        - KNN Regression
        - Lineare Regression
        - Decision Tree Klassifikation
        """)
    with tech_col2:
        st.markdown("""
        **Frontend / Visualisierung**
        - Streamlit
        - Plotly Express (interaktive Charts)
        - Matplotlib (PDF-Export)

        **Deployment**
        - Streamlit Community Cloud
        - GitHub (Versionskontrolle)
        """)
