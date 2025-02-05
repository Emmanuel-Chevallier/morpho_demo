# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:33:33 2025

@author: emman
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Fonctions utilitaires ---

def create_initial_image():
    """Crée une image binaire 300×300 avec deux triangles partageant un sommet."""
    img = np.zeros((300, 300), dtype=np.uint8)
    pts1 = np.array([[155, 150], [50, 250], [50, 50]], np.int32).reshape((-1, 1, 2))
    pts2 = np.array([[145, 150], [250, 250], [250, 50]], np.int32).reshape((-1, 1, 2))
    pts3 = np.array([[130, 50], [170, 50], [170, 70], [130,70]], np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts1], 255)
    cv2.fillPoly(img, [pts2], 255)
    cv2.fillPoly(img, [pts3], 255)
    return img

def get_structuring_element(elem_type, param):
    """Retourne l’élément structurant en fonction du type et du paramètre."""
    size = 2 * param + 1
    if elem_type == "Disque":
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    else:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return se

def perform_operation(op, img, se):
    """Effectue l'opération morphologique sur l'image."""
    if op == "Érosion":
        res = cv2.erode(img, se)
    elif op == "Dilatation":
        res = cv2.dilate(img, se)
    elif op == "Ouverture":
        res = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
    elif op == "Fermeture":
        res = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)
    else:
        res = img.copy()
    return res

def compute_contour_overlay(base_img, processed_img):
    """
    Calcule le contour de l'image traitée et dessine ce contour en rouge sur
    une copie de l'image de base.
    """
    color_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color_img, contours, -1, (0, 0, 255), 2)
    return color_img

def img_to_pil(img):
    """Convertit une image OpenCV (numpy array) en image PIL."""
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_color)

# --- Initialisation de l'état de session ---

if 'logical_image' not in st.session_state:
    st.session_state.logical_image = create_initial_image()
if 'history' not in st.session_state:
    st.session_state.history = []
if 'preview_image' not in st.session_state:
    st.session_state.preview_image = None

# --- Interface principale ---

st.title("Morphologie Mathématique")

# Panneau latéral pour les contrôles
st.sidebar.header("Paramètres")
operation = st.sidebar.selectbox("Opération", ["Érosion", "Dilatation", "Ouverture", "Fermeture"])
elem_type = st.sidebar.radio("Élément Structurant", ("Disque", "Carré"))
param = st.sidebar.number_input("Rayon / Demi-côté", min_value=1, max_value=50, value=5)

# Boutons d'action
if st.sidebar.button("Prévisualiser"):
    se = get_structuring_element(elem_type, param)
    processed = perform_operation(operation, st.session_state.logical_image, se)
    st.session_state.preview_image = compute_contour_overlay(st.session_state.logical_image, processed)
    
if st.sidebar.button("Appliquer"):
    if st.session_state.preview_image is not None:
        se = get_structuring_element(elem_type, param)
        processed = perform_operation(operation, st.session_state.logical_image, se)
        # Sauvegarder l'état actuel dans l'historique
        st.session_state.history.append(st.session_state.logical_image.copy())
        st.session_state.logical_image = processed.copy()
        st.session_state.preview_image = None
    else:
        st.warning("Veuillez d'abord prévisualiser l'opération.")

if st.sidebar.button("Annuler"):
    if st.session_state.history:
        st.session_state.logical_image = st.session_state.history.pop()
        st.session_state.preview_image = None
    else:
        st.warning("Aucune opération à annuler.")

# Affichage de l'image (prévisualisation ou image logique)
if st.session_state.preview_image is not None:
    img_disp = st.session_state.preview_image
else:
    img_disp = st.session_state.logical_image

st.image(img_to_pil(img_disp), caption="Résultat", use_column_width=True)
