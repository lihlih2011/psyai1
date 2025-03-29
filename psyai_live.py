import cv2
import streamlit as st
import numpy as np
from datetime import datetime
from fpdf import FPDF
import mediapipe as mp
import json
import csv
import os

st.set_page_config(page_title="PsyAI Live", layout="centered")
st.title("🧠 PsyAI - Assistant Médical en Temps Réel")

# ---------------------------
# 📚 MODULE MANUELS CLINIQUES
# ---------------------------
MANUALS_PATH = "data/manuals.json"
if os.path.exists(MANUALS_PATH):
    with open(MANUALS_PATH, "r", encoding="utf-8") as f:
        manuals = json.load(f)
    with st.expander("📖 Référentiels Cliniques Intégrés"):
        for nom, contenu in manuals.items():
            st.markdown(f"### {nom}")
            st.markdown(f"*{contenu['description']}*")
            st.markdown("**Signes observables associés :**")
            for signe in contenu["signes_observables"]:
                st.markdown(f"- {signe}")
else:
    st.warning("Aucun manuel clinique trouvé dans data/manuals.json")

# ---------------------------
# 🧑‍⚕️ FORMULAIRE PATIENT
# ---------------------------
st.subheader("📝 Informations Patient")
with st.form("form_patient"):
    nom = st.text_input("Nom")
    age = st.number_input("Âge", min_value=0, max_value=120, step=1)
    sexe = st.selectbox("Sexe", ["", "Homme", "Femme", "Autre"])
    symptomes = st.text_area("Symptômes rapportés")
    historique = st.text_area("Historique médical", help="Antécédents, diagnostics précédents, traitements...")
    is_private = st.checkbox("Session privée (non partagée)")
    submitted = st.form_submit_button("Valider les infos patient")

if submitted:
    st.success(f"Patient {nom} ({age} ans, {sexe}) enregistré.")
    st.session_state["patient_info"] = {
        "nom": nom,
        "age": age,
        "sexe": sexe,
        "symptomes": symptomes,
        "historique": historique,
        "session_privee": is_private
    }

# ---------------------------
# 🧠 MEDIA PIPE SETUP
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
LEFT_EYE_LANDMARKS = [159, 145]
RIGHT_EYE_LANDMARKS = [386, 374]
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
LEFT_BROW = 70
RIGHT_BROW = 300


def get_eye_aspect_ratio(landmarks, image_shape):
    ih, iw = image_shape[:2]
    def dist(p1, p2):
        y1 = int(p1.y * ih)
        y2 = int(p2.y * ih)
        return abs(y1 - y2)
    left_ratio = dist(landmarks[LEFT_EYE_LANDMARKS[0]], landmarks[LEFT_EYE_LANDMARKS[1]])
    right_ratio = dist(landmarks[RIGHT_EYE_LANDMARKS[0]], landmarks[RIGHT_EYE_LANDMARKS[1]])
    return (left_ratio + right_ratio) / 2

def get_facial_asymmetry(landmarks):
    left = landmarks[LEFT_CHEEK].x
    right = landmarks[RIGHT_CHEEK].x
    return abs(left - (1 - right))

def get_brow_movement(landmarks):
    return abs(landmarks[LEFT_BROW].y - landmarks[RIGHT_BROW].y)

def enregistrer_session_csv(patient, stats):
    if patient.get("session_privee"):
        return  # Ne pas enregistrer dans le fichier partagé
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", "sessions.csv")
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["date", "nom", "age", "sexe"] + list(stats.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow({"date": datetime.now().isoformat(), "nom": patient['nom'], "age": patient['age'], "sexe": patient['sexe'], **stats})

def enregistrer_session_json(patient, stats):
    dossier = "data/mes_sessions" if patient.get("session_privee") else "data"
    os.makedirs(dossier, exist_ok=True)
    filename = f"session_{patient['nom']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(dossier, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"patient": patient, "stats": stats, "date": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)

def generer_rapport_pdf(patient, stats, score, niveau):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Rapport d'analyse PsyAI", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Nom: {patient['nom']}", ln=True)
    pdf.cell(200, 10, txt=f"Âge: {patient['age']} | Sexe: {patient['sexe']}", ln=True)
    pdf.cell(200, 10, txt=f"Symptômes: {patient['symptomes']}", ln=True)
    pdf.cell(200, 10, txt=f"Historique: {patient['historique']}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Indicateurs comportementaux:", ln=True)
    for label, value in stats.items():
        pdf.cell(200, 10, txt=f"- {label}: {value}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Score comportemental: {score} / 6", ln=True)
    pdf.cell(200, 10, txt=f"Risque clinique estimé: {niveau}", ln=True)

    filename = f"rapport_{patient['nom']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

# ---------------------------
# LIVE ANALYSE
# ---------------------------
st.subheader("📷 Analyse Visuelle en Direct")
run = st.checkbox("🎥 Activer la webcam")
FRAME_WINDOW = st.image([])
stats_placeholder = st.empty()
pdf_placeholder = st.empty()
cap = cv2.VideoCapture(0)

blink_count = 0
frame_count = 0
last_ear = 0
blink_threshold = 4
asymmetry_vals = []
brow_vals = []

while run:
    success, frame = cap.read()
    if not success:
        st.warning("Webcam non détectée")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    ih, iw = frame.shape[:2]

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ear = get_eye_aspect_ratio(face_landmarks.landmark, frame.shape)
            if last_ear and abs(last_ear - ear) > blink_threshold:
                blink_count += 1
            last_ear = ear

            asym = get_facial_asymmetry(face_landmarks.landmark)
            brow = get_brow_movement(face_landmarks.landmark)
            asymmetry_vals.append(asym)
            brow_vals.append(brow)

            cv2.putText(frame, f"Clignements: {blink_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Asymétrie: {asym:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Sourcils: {brow:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

    FRAME_WINDOW.image(frame, channels="BGR")
    frame_count += 1

# Enregistrement des données à la fin
if frame_count > 0 and "patient_info" in st.session_state:
    moy_asym = np.mean(asymmetry_vals)
    moy_brow = np.mean(brow_vals)
    stats = {
        "clignements": blink_count,
        "asymétrie_moyenne": round(moy_asym, 4),
        "mouvement_sourcils_moyen": round(moy_brow, 4)
    }
    enregistrer_session_csv(st.session_state["patient_info"], stats)
    enregistrer_session_json(st.session_state["patient_info"], stats)

cap.release()
face_mesh.close()
