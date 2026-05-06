import streamlit as st # Streamlit ist unser Framework für die Web-App (Benutzeroberfläche)
import gspread # Erlaubt uns, mit Google Sheets (Tabellen) zu kommunizieren
from google.oauth2.service_account import Credentials # Für den sicheren Login bei Google
import pandas as pd # Pandas ist das wichtigste Tool für Datenanalyse (Tabellenkalkulation in Python)
import matplotlib.pyplot as plt # Zum Zeichnen von Basis-Diagrammen
import seaborn as sns # Baut auf matplotlib auf und macht schönere Diagramme mit weniger Code
from fpdf import FPDF # Erlaubt uns, PDF-Dateien zu generieren
import io # Hilft beim Verarbeiten von Dateien im Arbeitsspeicher
import os # Erlaubt Interaktion mit dem Betriebssystem (z.B. Dateien löschen)
import tempfile # Um temporäre Dateien (wie Bilder für das PDF) kurz zwischenzuspeichern
import numpy as np # Für mathematische Operationen und Arrays (Listen aus Zahlen)

# ==========================================
# MACHINE LEARNING (ML) IMPORTS
# ==========================================
# Scikit-learn (sklearn) ist die Standard-Bibliothek für Machine Learning in Python
from sklearn.neighbors import KNeighborsRegressor # KNN-Algorithmus: Vorhersage von Zahlenwerten (Regression)
from sklearn.tree import DecisionTreeClassifier, plot_tree # Entscheidungsbaum für Kategorien (Klassifikation) + Visualisierung
from sklearn.model_selection import train_test_split # Teilt unsere Daten in Trainings- und Testdaten auf
from sklearn.metrics import mean_absolute_error, r2_score # Um zu messen, wie gut unser ML-Modell ist
from sklearn.preprocessing import StandardScaler, LabelEncoder # Um Daten für das Modell aufzubereiten (Skalieren und Text in Zahlen umwandeln)

# ==========================================
# 1. SETUP UND AUTHENTIFIZIERUNG (DATENBANK)
# ==========================================

# 'scopes' definiert, auf welche Google-Dienste wir zugreifen wollen
scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# Wir versuchen uns bei Google einzuloggen. Das "try-except" fängt Fehler ab, falls der Login scheitert.
try:
    # Holt die geheimen Login-Daten (API-Keys) aus den Streamlit Secrets. 
    # So stehen keine Passwörter direkt im Code!
    credentials = Credentials.from_service_account_info(
        st.secrets["google_credentials"],
        scopes=scopes
    )
    client = gspread.authorize(credentials) # Loggt uns bei Google Sheets ein
    
    # Wir öffnen das Google Sheet mit dem Namen "Dataset" und wählen das erste Tabellenblatt (sheet1)
    sheet = client.open("Dataset").sheet1
    data = sheet.get_all_records() # Holt alle Zeilen und Spalten als eine große Liste
    
    # Wir verwandeln diese Daten in einen "Pandas DataFrame" (df). 
    # Ein DataFrame ist wie eine Excel-Tabelle direkt in Python, mit der man extrem schnell rechnen und filtern kann.
    df = pd.DataFrame(data) 
except Exception as e:
    # Wenn der Login fehlschlägt, zeigen wir eine rote Fehlermeldung auf der Webseite an und stoppen das Skript.
    st.error(f"Fehler bei der Verbindung zur Datenbank: {e}")
    st.stop()

# ==========================================
# SEITEN-NAVIGATION (SESSION STATE)
# ==========================================
# Streamlit lädt das Skript jedes Mal neu, wenn man einen Knopf drückt.
# Der 'session_state' merkt sich Dinge zwischen diesen Neustarts (z.B. auf welcher Seite wir gerade sind).
if "page" not in st.session_state:
    st.session_state.page = "Home" # Startseite festlegen

# ==========================================
# 2. HOME PAGE (STARTSEITE)
# ==========================================
if st.session_state.page == "Home":
    st.title("⚽ Football Club Performance Monitor") # Titel der Webseite
    
    # st.write schreibt einfach Text auf die Webseite
    st.write(
        "Diese Anwendung bietet einen umfassenden Überblick über die Spielerleistung, "
        "Trainingsbeteiligung und Teamzusammensetzung."
    )
    st.write("### Navigiere durch die Anwendung:")
    st.write("- **Dashboard:** Interaktive Visualisierungen der Spielerdaten (Alter, Positionen, Verletzungen).")
    st.write("- **Machine Learning:** Vorhersage der Spielerleistung basierend auf historischen Daten.")
    st.write("- **Data Management:** Spieler hinzufügen, bearbeiten oder löschen (live in die Datenbank).")
    st.write("- **Player Report:** Individuelle Spielerberichte generieren und als PDF herunterladen.")
    st.write("---")
    
    # Wir erstellen 4 Spalten nebeneinander für unsere Navigations-Knöpfe
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    # st.button() erstellt einen Knopf. Wenn er gedrückt wird, ändert sich st.session_state.page
    # st.rerun() erzwingt ein sofortiges Neuladen der Seite, damit die neue Seite sofort angezeigt wird.
    with col1:
        if st.button("Dashboard"):
            st.session_state.page = "Dashboard"
            st.rerun()
    with col2:
        if st.button("Machine Learning"):
            st.session_state.page = "Machine Learning"
            st.rerun()
    with col3:
        if st.button("Data Management"):
            st.session_state.page = "Data Management"
            st.rerun()
    with col4:
        if st.button("Player Report"):
            st.session_state.page = "Player Report"
            st.rerun()

# ==========================================
# 3. DASHBOARD PAGE (DATENVISUALISIERUNG)
# ==========================================
elif st.session_state.page == "Dashboard":
    st.title("📊 Player Performance Dashboard")
    
    # Zurück-Knopf
    if st.button("⬅️ Homepage"):
        st.session_state.page = "Home"
        st.rerun()
        
    st.subheader("Club Overview") # Untertitel
    col1, col2, col3 = st.columns(3)
    
    # st.metric() zeigt eine Zahl groß und übersichtlich an (wie auf Dashboards üblich)
    col1.metric("Total Players", len(df)) # len(df) gibt die Anzahl der Zeilen (Spieler) in der Tabelle
    
    # Sicherheitsprüfung: Existiert die Spalte "PerformanceScore" überhaupt?
    if "PerformanceScore" in df.columns:
        # Berechnet den Durchschnitt (.mean()) und rundet ihn auf 1 Nachkommastelle
        col2.metric("Avg Performance", round(df["PerformanceScore"].mean(), 1))
    if "TrainingAttendanceRate" in df.columns:
        col3.metric("Avg Attendance", f"{round(df['TrainingAttendanceRate'].mean(),1)}%")

    # --- Diagramm 1: Altersverteilung (Histogramm) ---
    st.subheader("Altersverteilung")
    fig, ax = plt.subplots() # Erstellt eine "leere Leinwand" (fig) und die Achsen (ax)
    # Seaborn (sns) zeichnet ein Histogramm für die Spalte "Age" (Alter)
    sns.histplot(df['Age'], bins=20, kde=True, ax=ax, color='skyblue') 
    ax.set_title("Age Distribution")
    st.pyplot(fig) # Zeigt die fertige "Leinwand" auf der Streamlit Webseite an

    # --- Diagramm 2: Spieler pro Position (Balkendiagramm) ---
    st.subheader("Spieler pro Position")
    position_counts = df['Position'].value_counts() # Zählt, wie oft jede Position vorkommt
    fig, ax = plt.subplots()
    position_counts.plot(kind='bar', ax=ax, color='lightblue')
    ax.set_title("Players by Position")
    st.pyplot(fig)

    # --- Diagramm 3: Tore nach Position (Boxplot) ---
    st.subheader("Tore nach Position")
    fig, ax = plt.subplots()
    # Ein Boxplot zeigt an, wie die Tore für jede Position verteilt sind (Minimum, Maximum, Durchschnitt)
    sns.boxplot(x='Position', y='Goals', data=df, ax=ax)
    ax.set_title("Goals Distribution by Position")
    st.pyplot(fig)

    # --- Diagramm 4: Leistung vs. Trainingsbeteiligung (Streudiagramm) ---
    st.subheader("Leistung vs. Trainingsbeteiligung")
    fig, ax = plt.subplots()
    # Jeder Punkt ist ein Spieler. x-Achse = Training, y-Achse = Leistung
    sns.scatterplot(x='TrainingAttendanceRate', y='PerformanceScore', data=df, ax=ax)
    # sns.regplot zeichnet eine Trendlinie (rote Linie) ein
    sns.regplot(x='TrainingAttendanceRate', y='PerformanceScore', data=df, ax=ax, scatter=False, color='red')
    st.pyplot(fig)

    # --- Diagramm 5: Fitness vs Alter ---
    st.subheader("Fitness vs Alter")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Age', y='FitnessScore', data=df, ax=ax)
    sns.regplot(x='Age', y='FitnessScore', data=df, ax=ax, scatter=False, color='red')
    st.pyplot(fig)

    # --- Diagramm 6: Verletzungsstatus ---
    st.subheader("Verletzungsstatus")
    injury_counts = df['InjuryStatus'].value_counts()
    fig, ax = plt.subplots()
    injury_counts.plot(kind='bar', ax=ax, color='salmon')
    st.pyplot(fig)

# ==========================================
# 4. MACHINE LEARNING PAGE (KI-VORHERSAGEN)
# ==========================================
elif st.session_state.page == "Machine Learning":
    st.title("🤖 Player Performance Predictor")
    st.write(
        "Hier nutzen wir **Machine Learning (Künstliche Intelligenz)**, um die Leistung eines Spielers "
        "vorherzusagen. Das Modell hat aus unseren Datenbank-Daten gelernt und wendet dieses Wissen "
        "nun auf neue, ausgedachte Spieler an."
    )
    if st.button("⬅️ Homepage"):
        st.session_state.page = "Home"
        st.rerun()

    # Wir definieren, welche Spalten das Modell als Input (Features) und Output (Target) nutzt
    CATEGORICAL = ["InjuryStatus", "Position", "Gender"] # Text-Spalten (Kategorien)
    NUMERICAL = ["Age", "TrainingAttendanceRate", "FitnessScore", "Goals"] # Zahlen-Spalten
    FEATURES = NUMERICAL + CATEGORICAL # Die 7 Indikatoren, auf die das Modell schaut
    TARGET = "PerformanceScore" # Das wollen wir vorhersagen (Ergebnis)

    # Entfernt alle Zeilen aus der Tabelle, bei denen Daten fehlen (.dropna()), 
    # da ML-Modelle bei leeren Feldern abstürzen.
    ml_df = df[FEATURES + [TARGET]].dropna().copy()

    # Sicherheitsprüfung: Wenn zu wenig Daten in der Datenbank sind, stoppt das Modell.
    if len(ml_df) < 10:
        st.warning("Zu wenig Daten vorhanden, um das Modell zu trainieren (min. 10 Spieler nötig).")
        st.stop()

    # ML-Modelle verstehen keinen Text (wie "Torwart" oder "Verletzt").
    # Ein LabelEncoder wandelt jeden Text in eine Nummer um (z.B. Torwart=0, Stürmer=1, etc.)
    encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col].astype(str))
        encoders[col] = le # Wir speichern uns den Übersetzer für später

    X = ml_df[FEATURES] # X = Die 7 Input-Variablen
    y = ml_df[TARGET]   # y = Das Ergebnis, das das Modell lernen soll

    # ---------------------------------------------------------
    # SCHRITT 1: Train/Test Split (Daten aufteilen)
    # ---------------------------------------------------------
    # Wir trainieren das Modell nicht mit 100% der Daten, sonst könnten wir es nicht prüfen.
    # Wir nehmen 80% zum Lernen (Trainingsdaten) und verstecken 20% für den Test (Testdaten).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # SCHRITT 2: Feature Scaling (Werte angleichen)
    # ---------------------------------------------------------
    # Das Alter geht von z.B. 18-40, Tore von 0-30. Damit große Zahlen das Modell
    # nicht fälschlicherweise dominieren, "stauchen" wir alle Zahlenwerte auf eine ähnliche Skala.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------
    # SCHRITT 3: Modell 1 (K-Nearest Neighbors - KNN) trainieren
    # ---------------------------------------------------------
    # KNN sucht die 5 Spieler in der Datenbank, die dem neuen Spieler am ähnlichsten sind,
    # und nimmt deren Durchschnitt als Vorhersage.
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train_scaled, y_train) # Hier "lernt" das Modell!

    # Wir lassen das Modell Vorhersagen für die 20% versteckten Testdaten machen
    y_pred = model.predict(X_test_scaled)
    # Wir vergleichen die Vorhersage (y_pred) mit der Realität (y_test)
    r2 = r2_score(y_test, y_pred) # R²: Wie viel Prozent der Wahrheit hat das Modell verstanden? (1.0 = perfekt)
    mae = mean_absolute_error(y_test, y_pred) # MAE: Um wie viele Punkte liegt das Modell im Schnitt falsch?

    # --- Ergebnisse anzeigen ---
    st.subheader("📊 Modell Genauigkeit")
    col1, col2, col3 = st.columns(3)
    col1.metric("Trainings-Datensätze", len(X_train))
    col2.metric("R² Score (Qualität)", f"{r2:.2f}")
    col3.metric("Durchschn. Fehler", f"{mae:.1f} Punkte")

    # --- Live-Vorhersage durch den User (Interaktion) ---
    st.subheader("🎯 Teste es selbst: Spieler-Vorhersage")
    st.write("Baue dir deinen eigenen Spieler zusammen. Die KI sagt seinen Performance Score voraus.")
    
    col1, col2 = st.columns(2)
    with col1:
        # Streamlit Slider erlauben es dem User, Zahlenwerte intuitiv einzustellen
        input_age = st.slider("Alter", min_value=15, max_value=40, value=25)
        input_attendance = st.slider("Trainingsbeteiligung (%)", min_value=0.0, max_value=100.0, value=80.0)
        input_fitness = st.slider("Fitness Score", min_value=0.0, max_value=100.0, value=75.0)
        input_goals = st.slider("Tore", min_value=0, max_value=50, value=5)
    
    with col2:
        # Streamlit Dropdowns (Selectbox) für die Text-Kategorien
        # .classes_ greift auf die Originalwörter zurück, die der LabelEncoder gelernt hat.
        input_injury_label = st.selectbox("Verletzungsstatus", encoders["InjuryStatus"].classes_.tolist())
        input_position_label = st.selectbox("Position", encoders["Position"].classes_.tolist())
        input_gender_label = st.selectbox("Geschlecht", encoders["Gender"].classes_.tolist())

    # Wir übersetzen die Text-Auswahlen des Users wieder in Zahlen für das ML Modell
    input_injury = encoders["InjuryStatus"].transform([input_injury_label])[0]
    input_position = encoders["Position"].transform([input_position_label])[0]
    input_gender = encoders["Gender"].transform([input_gender_label])[0]

    # Wir fassen alle 7 Werte in ein Array zusammen und skalieren es wie die Trainingsdaten
    input_array = np.array([[input_age, input_attendance, input_fitness, input_goals, input_injury, input_position, input_gender]])
    input_scaled = scaler.transform(input_array)
    
    # Das Modell trifft seine Vorhersage basierend auf dem Input des Users
    prediction = model.predict(input_scaled)[0]

    st.success(f"**Vorhergesagter Performance Score: {prediction:.1f} / 100**")

    # ---------------------------------------------------------
    # SCHRITT 4: Modell 2 (Decision Tree - Entscheidungsbaum)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🌳 Spieler-Klassifizierung (Decision Tree)")
    st.write("Ein zweites KI-Modell (Entscheidungsbaum) teilt den Spieler in eine von drei Leistungsgruppen ein.")

    # Funktion: Wandelt den genauen Score in eine Text-Kategorie um (High/Medium/Low)
    def categorise(score):
        if score >= 75: return "High"
        elif score >= 50: return "Medium"
        else: return "Low"

    # Neue Spalte anlegen mit den Kategorien
    ml_df["PerformanceCategory"] = ml_df[TARGET].apply(categorise)
    y_dt = ml_df["PerformanceCategory"] # Das neue Target für den Decision Tree

    # Train/Test Split für den Baum (stratify stellt sicher, dass alle Kategorien gleichmäßig verteilt sind)
    X_dt_train, X_dt_test, y_dt_train, y_dt_test = train_test_split(
        ml_df[FEATURES], y_dt, test_size=0.2, random_state=42, stratify=y_dt
    )

    # Trainieren des Entscheidungsbaums (max_depth=4 heißt, er darf maximal 4 Fragen in Folge stellen)
    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_model.fit(X_dt_train, y_dt_train)

    st.write("#### Wie denkt die KI? (Der gelernte Entscheidungsbaum)")
    # Wir zeichnen den Baum grafisch, damit der User nachvollziehen kann, 
    # auf Basis welcher Kriterien die KI ihre Entscheidungen trifft.
    fig, ax = plt.subplots(figsize=(22, 8))
    plot_tree(dt_model, feature_names=FEATURES, class_names=dt_model.classes_, filled=True, rounded=True, fontsize=9, ax=ax)
    st.pyplot(fig)

    # Klassifizierung des vom User erstellten Spielers aus den Slidern oben
    dt_prediction = dt_model.predict(input_array)[0] # In welche Klasse fällt der User-Spieler?
    dt_proba = dt_model.predict_proba(input_array)[0] # Mit wie viel % Sicherheit ist sich das Modell sicher?
    
    st.success(f"**Vorhersage Kategorie: {dt_prediction} Performer**")

# ==========================================
# 5. DATA MANAGEMENT PAGE (DATENBANK ÄNDERN)
# ==========================================
elif st.session_state.page == "Data Management":
    st.title("🗃️ Data Input Manager")
    st.write("Hier kannst du Spieler zur Datenbank hinzufügen, bearbeiten oder löschen. Alle Änderungen sind sofort live im Google Sheet.")
    
    # Session State für die Unterseiten im Data Management (Add/Edit/Delete)
    if "Backend_subpage" not in st.session_state:
        st.session_state.Backend_subpage = "main"

    headers = sheet.row_values(1) # Liest die Spalten-Namen aus Zeile 1 im Google Sheet
    sheet_data = sheet.get_all_values() # Liest alle Daten

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
        
        # Ein Formular in Streamlit. Es schickt die Daten erst ab, wenn der "Submit"-Button gedrückt wird.
        with st.form("add_player_form"):
            name = st.text_input("Name (Optional)")
            age = st.number_input("Alter", min_value=15, max_value=50, value=25)
            gender = st.selectbox("Geschlecht", ["Male", "Female"])
            position = st.selectbox("Position", ["Forward", "Midfielder", "Defender", "Goalkeeper"])
            injury_status = st.selectbox("Verletzungsstatus", ["Fit", "Injured", "Recovering"])
            attendance = st.number_input("Trainingsbeteiligung (%)", min_value=0.0, max_value=100.0, value=80.0)
            fitness = st.number_input("Fitness Score", min_value=0.0, max_value=100.0, value=75.0)
            goals = st.number_input("Tore", min_value=0, value=0)
            performance = st.number_input("Performance Score", min_value=0.0, max_value=100.0, value=50.0)

            # Wenn der Benutzer auf "Speichern" klickt...
            if st.form_submit_button("Speichern in Datenbank"):
                # Wir bauen die neue Reihe so auf, wie die Spalten im Google Sheet sortiert sind.
                new_row_data = []
                for header in headers:
                    if header == "Age": new_row_data.append(age)
                    elif header == "Gender": new_row_data.append(gender)
                    elif header == "Position": new_row_data.append(position)
                    elif header == "InjuryStatus": new_row_data.append(injury_status)
                    elif header == "TrainingAttendanceRate": new_row_data.append(attendance)
                    elif header == "FitnessScore": new_row_data.append(fitness)
                    elif header == "Goals": new_row_data.append(goals)
                    elif header == "PerformanceScore": new_row_data.append(performance)
                    elif header == "PlayerName": new_row_data.append(name)
                    else: new_row_data.append("") # Wenn eine Spalte existiert, die wir nicht kennen, bleibt sie leer.
                
                # Befehl an gspread: Füge diese neue Reihe ganz unten ins Google Sheet ein!
                sheet.append_row(new_row_data)
                st.success("Spieler erfolgreich gespeichert!")

        if st.button("⬅️ Zurück"):
            st.session_state.Backend_subpage = "main"
            st.rerun()

    # --- SPIELER BEARBEITEN ---
    elif st.session_state.Backend_subpage == "edit_player":
        st.subheader("Spielerdaten anpassen")
        
        # Liste mit allen Spielern zum Auswählen erstellen (Zeile 2, 3, 4...)
        if "PlayerName" in headers:
            idx_name = headers.index("PlayerName")
            player_list = [f"Reihe {i+2} - {row[idx_name]}" for i, row in enumerate(sheet_data[1:])]
        else:
            player_list = [f"Reihe {i+2}" for i in range(len(sheet_data)-1)]
            
        selected_player = st.selectbox("Wähle einen Spieler", player_list)
        
        if selected_player:
            # Wir suchen die Excel-Zeilennummer des ausgewählten Spielers heraus
            row_num = int(selected_player.split(" ")[1])
            current_data = sheet_data[row_num - 1]
            
            with st.form("edit_player_form"):
                updated_data = []
                # Für jede Spalte zeigen wir ein Textfeld mit dem aktuellen Wert an
                for i, header in enumerate(headers):
                    val = st.text_input(header, value=current_data[i])
                    updated_data.append(val)
                    
                if st.form_submit_button("Update durchführen"):
                    # Befehl an gspread: Überschreibe diese genaue Reihe mit den neuen Werten
                    sheet.update(f"A{row_num}", [updated_data])
                    st.success("Spielerdaten erfolgreich aktualisiert!")

        if st.button("⬅️ Zurück"):
            st.session_state.Backend_subpage = "main"
            st.rerun()

    # --- SPIELER LÖSCHEN ---
    elif st.session_state.Backend_subpage == "delete_player":
        st.subheader("Spieler löschen")
        
        # Identisch zur Bearbeiten-Seite, wir erstellen eine Auswahlliste
        if "PlayerName" in headers:
            idx_name = headers.index("PlayerName")
            player_list = [f"Reihe {i+2} - {row[idx_name]}" for i, row in enumerate(sheet_data[1:])]
        else:
            player_list = [f"Reihe {i+2}" for i in range(len(sheet_data)-1)]
            
        selected_player = st.selectbox("Spieler zum Löschen wählen", player_list)
        
        if selected_player:
            row_num = int(selected_player.split(" ")[1])
            st.warning(f"Bist du sicher, dass du {selected_player} löschen willst? Das kann nicht rückgängig gemacht werden.")
            if st.button("🚨 Unwiderruflich löschen"):
                # Befehl an gspread: Lösche genau diese Excel-Reihe komplett
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
    
    # 1. Spieler zum Drucken auswählen
    if "PlayerName" in df.columns:
        player_labels = df["PlayerName"].astype(str).tolist()
    else:
        player_labels = [f"Player {i+1} | {df.iloc[i].get('Position','?')} | Age {df.iloc[i].get('Age','?')}" for i in range(len(df))]
        
    selected_label = st.selectbox("Spieler auswählen", player_labels)
    selected_idx = player_labels.index(selected_label)
    # Wähle die genaue Zeile aus Pandas (df.iloc = Index Location)
    player = df.iloc[selected_idx] 
    
    st.subheader(f"Übersicht -- {selected_label}")
    # Zeige die Kerndaten als Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Performance Score", f"{player.get('PerformanceScore', 'N/A')}")
    c2.metric("Fitness Score", f"{player.get('FitnessScore', 'N/A')}")
    c3.metric("Training Attendance", f"{player.get('TrainingAttendanceRate', 'N/A')}%")
    c4.metric("Goals", f"{player.get('Goals', 'N/A')}")

    # 2. Vergleich: Spieler vs Team-Durchschnitt
    st.subheader("Vergleich vs. Team-Durchschnitt")
    num_cols = ["PerformanceScore", "FitnessScore", "TrainingAttendanceRate", "Goals"]
    available = [c for c in num_cols if c in df.columns]
    
    player_vals = [float(player.get(c, 0)) for c in available]
    team_avg = [float(df[c].mean()) for c in available]
    
    # Wir erstellen ein Doppel-Balkendiagramm
    x = np.arange(len(available))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    bars1 = ax.bar(x - width/2, player_vals, width, label="Dieser Spieler", color="steelblue")
    bars2 = ax.bar(x + width/2, team_avg, width, label="Team Durchschnitt", color="lightgray")
    ax.set_xticks(x)
    ax.set_xticklabels(available, rotation=15)
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Als PDF exportieren")
    
    # FPDF ist eine Bibliothek, um programmatisch PDFs zu zeichnen.
    # Wir platzieren Text wie mit einem Stempel auf einem Blatt Papier (x, y Koordinaten).
    if st.button("PDF Bericht generieren"):
        try:
            pdf = FPDF() # Erstellt ein leeres PDF-Dokument
            pdf.add_page() # Fügt eine leere Seite hinzu
            
            # --- Header-Bereich im PDF ---
            pdf.set_fill_color(20, 83, 45) # Hintergrundfarbe dunkelgrün (RGB)
            pdf.set_text_color(255, 255, 255) # Schrift weiß
            pdf.set_font("Arial", "B", 18) # Schriftart Arial, Fett (Bold), Größe 18
            pdf.cell(0, 14, "Football Club -- Player Report", ln=True, fill=True, align="C") # Druckt ein Rechteck mit Text
            pdf.ln(4) # Fügt einen kleinen Zeilenabstand ein (LineBreak)
            
            # --- Spielerinfo ---
            pdf.set_text_color(0, 0, 0) # Schrift wieder schwarz
            pdf.set_font("Arial", "B", 13)
            pdf.cell(0, 9, f"Player: {selected_label}", ln=True)
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 7, f"Position: {player.get('Position','N/A')} | Age: {player.get('Age','N/A')}", ln=True)
            pdf.ln(4)
            
            # --- Tabelle mit Stats zeichnen ---
            pdf.set_font("Arial", "B", 12)
            pdf.set_fill_color(220, 240, 220) # Leichtes grün
            pdf.cell(0, 8, "Performance Metrics", ln=True, fill=True)
            pdf.set_font("Arial", "", 11)
            
            stats = {
                "Performance Score": player.get("PerformanceScore", "N/A"),
                "Fitness Score": player.get("FitnessScore", "N/A"),
                "Training Attendance": f"{player.get('TrainingAttendanceRate','N/A')}%",
                "Goals": player.get("Goals", "N/A"),
            }
            
            # Für jedes Stat zeichnen wir eine Zelle mit Rand (border=1)
            for label, val in stats.items():
                pdf.cell(80, 7, label, border=1) # Linke Zelle (Name)
                pdf.cell(0, 7, str(val), border=1, ln=True) # Rechte Zelle (Wert)
            pdf.ln(4)
            
            # --- Diagramm ins PDF einfügen ---
            # Wir generieren das Diagramm von oben nochmal neu...
            fig_pdf, ax_pdf = plt.subplots(figsize=(7, 3))
            ax_pdf.bar(np.arange(len(available)) - 0.175, player_vals, 0.35, label="This Player", color="steelblue")
            ax_pdf.bar(np.arange(len(available)) + 0.175, team_avg, 0.35, label="Team Average", color="lightgray")
            ax_pdf.set_xticks(np.arange(len(available)))
            ax_pdf.set_xticklabels(available, rotation=15)
            ax_pdf.legend()
            
            # ... speichern es kurz als Bild auf der Festplatte des Servers ...
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                fig_pdf.savefig(tmp_path, bbox_inches="tight", dpi=120)
                plt.close(fig_pdf)
                
            # ... und stempeln das Bild in das PDF-Dokument!
            pdf.set_font("Arial", "B", 12)
            pdf.set_fill_color(220, 240, 220)
            pdf.cell(0, 8, "Comparison Chart", ln=True, fill=True)
            pdf.image(tmp_path, x=15, w=170)
            os.unlink(tmp_path) # WICHTIG: Das temporäre Bild auf der Festplatte wieder löschen!
            
            # --- Footer ---
            pdf.set_y(-15) # 15 Einheiten vom unteren Rand entfernt
            pdf.set_font("Arial", "I", 8) # I = Italic (Kursiv)
            pdf.set_text_color(120, 120, 120) # Grau
            pdf.cell(0, 10, "Generiert durch den Football Club Performance Monitor", align="C")
            
            # PDF in Bytes umwandeln, damit es über das Internet heruntergeladen werden kann
            pdf_bytes = pdf.output(dest="S").encode("latin-1")
            
            # Streamlit Download Button für das fertige PDF
            st.download_button(
                label="📥 Download PDF",
                data=pdf_bytes,
                file_name=f"player_report_{selected_label.replace(' ','_')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Fehler bei der PDF Generierung: {e}")
