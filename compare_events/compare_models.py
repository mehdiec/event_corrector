import napari
import zarr
import networkx as nx
import numpy as np
import pickle, sys, json, os, pandas as pd
from qtpy.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel, QPushButton, QComboBox, QDoubleSpinBox, QCheckBox, QFrame, QSpinBox
from qtpy.QtCore import Qt # type: ignore

# --- Données et chargement (inchangé) ---
ZARR_PATH = '/Volumes/u934/equipe_bellaiche/c_casabonne/vi_pupa_for_benchmark_cleo_no_modif'
GRAPH_PATH = '/Volumes/u934/equipe_bellaiche/c_casabonne/vi_pupa_for_benchmark_cleo_no_modif/CELL/graph.gpickle'
# Chemin vers votre nouveau fichier CSV de référence pour les divisions
DIVISION_REF_CSV_PATH = '/Volumes/u934/equipe_bellaiche/c_casabonne/vi_pupa_for_benchmark_cleo_no_modif/divisions_correction.csv'


# Modifiez ces chemins selon vos besoins
SAVE_DIR = os.path.expanduser("/Volumes/u934/equipe_bellaiche/c_casabonne/vi_pupa_for_benchmark_cleo_no_modif/annotations")
ANNOTATION_SAVE_PATH = os.path.join(SAVE_DIR, "annotation_state.json")
STATS_SAVE_PATH = os.path.join(SAVE_DIR, "annotation_stats.txt")

print("Chargement des données...")
# ... (code de chargement identique)
try:
    animal = zarr.open(ZARR_PATH, mode='r')
    labels_zarr = animal.IMAGE.D2.label[:]
    raw_image_zarr = animal.IMAGE.D2.raw[:, 0]
    with open(GRAPH_PATH, 'rb') as f:
        mon_graphe = pickle.load(f)
    print("✅ Données chargées avec succès !")
except Exception as e:
    print(f"❌ Erreur lors du chargement des fichiers : {e}")
    sys.exit()

print("Extraction des événements du graphe de référence...")
division_points_ref = []
apoptosis_points_ref = []
# ... (code d'extraction identique)
for node_id, data in mon_graphe.nodes(data=True):
    time, coords_yx = data.get('time'), data.get('coords')
    if time is not None and coords_yx is not None:
        point_coord = [time, coords_yx[0], coords_yx[1]]
        if data.get('is_dividing') is True:
            division_points_ref.append(point_coord)
        if data.get('delaminating_node') is True:
            apoptosis_points_ref.append(point_coord)
print(f"🔬 Événements trouvés : {len(division_points_ref)} divisions, {len(apoptosis_points_ref)} apoptoses.")

print("Chargement des divisions de référence (CSV)...")
division_points_csv = []
try:
    # Le séparateur est une virgule (',')
    df = pd.read_csv(DIVISION_REF_CSV_PATH, sep=',')
    # Assurez-vous que les colonnes sont dans le bon ordre pour napari [t, y, x]
    division_points_csv = df[['t', 'y', 'x']].to_numpy()
    print(f"✅ {len(division_points_csv)} divisions chargées depuis le CSV.")
except FileNotFoundError:
    print(f"⚠️ Fichier CSV de référence non trouvé à : {DIVISION_REF_CSV_PATH}. La couche ne sera pas ajoutée.")
except Exception as e:
    print(f"❌ Erreur lors du chargement du fichier CSV : {e}")


# --- La classe de notre Widget Qt ---

class ScoringWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.action_history = []
        self.layers = {"Divisions": {}, "Apoptoses": {}}
        self.ui_elements = {"Divisions": [], "Apoptoses": []}
        
        # --- Création des widgets et mise en page (inchangé) ---
        self.event_selector = QComboBox()
        self.event_selector.addItems(["Divisions", "Apoptoses"])

        # --- Labels pour les métriques dérivées (Modèle vs Matlab) ---
        self.tp_model_div_label = QLabel("TP (Modèle): 0")
        self.fp_model_div_label = QLabel("FP (Modèle): 0")
        self.fn_model_div_label = QLabel("FN (Modèle): 0")
        self.tp_matlab_div_label = QLabel("TP (Matlab): 0")
        self.fp_matlab_div_label = QLabel("FP (Matlab): 0")
        self.fn_matlab_div_label = QLabel("FN (Matlab): 0")
        self.tp_model_apo_label = QLabel("TP (Modèle): 0")
        self.fp_model_apo_label = QLabel("FP (Modèle): 0")
        self.fn_model_apo_label = QLabel("FN (Modèle): 0")
        self.tp_matlab_apo_label = QLabel("TP (Matlab): 0")
        self.fp_matlab_apo_label = QLabel("FP (Matlab): 0")
        self.fn_matlab_apo_label = QLabel("FN (Matlab): 0")

        # --- Nouveaux widgets pour les seuils ---
        self.div_threshold_label = QLabel("Seuil Divisions:")
        self.div_threshold_spinbox = QDoubleSpinBox()
        self.div_threshold_spinbox.setRange(0.0, 1.0); self.div_threshold_spinbox.setSingleStep(0.01); self.div_threshold_spinbox.setValue(0.7)
        self.apo_threshold_label = QLabel("Seuil Apoptoses:")
        self.apo_threshold_spinbox = QDoubleSpinBox()
        self.apo_threshold_spinbox.setRange(0.0, 1.0); self.apo_threshold_spinbox.setSingleStep(0.01); self.apo_threshold_spinbox.setValue(0.5)
        
        # --- Widgets pour le compteur de frames ---
        self.total_frames = self.viewer.dims.nsteps[0] if self.viewer.dims.ndim > 2 else 1
        self.frame_status = {"Divisions": [False] * self.total_frames, "Apoptoses": [False] * self.total_frames}
        self._current_annotation_frame = 0  # La frame que l'utilisateur est en train d'annoter
        self.frames_validated_div_label = QLabel(f"Frames validées: 0 / {self.total_frames}")

        self.frame_counter_label = QLabel("Frame en cours:")
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setRange(0, self.total_frames - 1)
        self.frame_done_checkbox = QCheckBox("Done")
        self.next_undone_btn = QPushButton("Prochaine non validée")
        self.frames_validated_label = QLabel(f"Frames validées: 0 / {self.total_frames}")
        self.frames_validated_apo_label = QLabel(f"Frames validées: 0 / {self.total_frames}")
        self.prev_frame_btn = QPushButton("Frame Précédente")
        self.next_frame_btn = QPushButton("Frame Suivante")
        self.sync_viewer_frame_btn = QPushButton("Sync avec Viewer")

        # --- Boutons de sauvegarde et chargement ---
        self.save_btn = QPushButton("Sauvegarder l'annotation")
        self.load_btn = QPushButton("Charger l'annotation")

        self.reset_btn = QPushButton("Reset Tous les Points")
        
        # --- Mise en page ---
        layout = QVBoxLayout()
        layout.addWidget(QLabel("<b>Événement à annoter :</b>")); layout.addWidget(self.event_selector)
        
        grid = QGridLayout()
        row = 0

        # Création des titres pour pouvoir les activer/désactiver
        div_title = QLabel("<b>Divisions</b>")
        div_model_title = QLabel("<b>Modèle</b>")
        div_matlab_title = QLabel("<b>Matlab</b>")
        apo_title = QLabel("<b>Apoptoses</b>")
        apo_model_title = QLabel("<b>Modèle</b>")
        apo_matlab_title = QLabel("<b>Matlab</b>")

        # --- Section Divisions ---
        grid.addWidget(div_title, row, 0, 1, 2); row += 1
        grid.addWidget(div_model_title, row, 0); grid.addWidget(div_matlab_title, row, 1); row += 1
        grid.addWidget(self.tp_model_div_label, row, 0); grid.addWidget(self.tp_matlab_div_label, row, 1); row += 1
        grid.addWidget(self.fp_model_div_label, row, 0); grid.addWidget(self.fp_matlab_div_label, row, 1); row += 1
        grid.addWidget(self.fn_model_div_label, row, 0); grid.addWidget(self.fn_matlab_div_label, row, 1); row += 1
        grid.addWidget(self.frames_validated_div_label, row, 0, 1, 2); row += 1 # Nouveau label pour Divisions
        self.ui_elements["Divisions"].extend([div_title, div_model_title, div_matlab_title, self.tp_model_div_label, self.tp_matlab_div_label, self.fp_model_div_label, self.fp_matlab_div_label, self.fn_model_div_label, self.fn_matlab_div_label, self.frames_validated_div_label])

        # --- Section Apoptoses ---
        grid.addWidget(apo_title, row, 0, 1, 2); row += 1
        grid.addWidget(apo_model_title, row, 0); grid.addWidget(apo_matlab_title, row, 1); row += 1
        grid.addWidget(self.tp_model_apo_label, row, 0); grid.addWidget(self.tp_matlab_apo_label, row, 1); row += 1
        grid.addWidget(self.fp_model_apo_label, row, 0); grid.addWidget(self.fp_matlab_apo_label, row, 1); row += 1
        grid.addWidget(self.fn_model_apo_label, row, 0); grid.addWidget(self.fn_matlab_apo_label, row, 1); row += 1
        grid.addWidget(self.frames_validated_apo_label, row, 0, 1, 2); row += 1 # Nouveau label pour Apoptoses
        self.ui_elements["Apoptoses"].extend([apo_title, apo_model_title, apo_matlab_title, self.tp_model_apo_label, self.tp_matlab_apo_label, self.fp_model_apo_label, self.fp_matlab_apo_label, self.fn_model_apo_label, self.fn_matlab_apo_label, self.frames_validated_apo_label])
        
        # --- Section Contrôles ---
        grid.addWidget(QLabel("<b>Contrôles</b>"), row, 0, 1, 3); row += 1
        grid.addWidget(self.div_threshold_label, row, 0); grid.addWidget(self.div_threshold_spinbox, row, 1, 1, 2); row += 1
        grid.addWidget(self.apo_threshold_label, row, 0); grid.addWidget(self.apo_threshold_spinbox, row, 1, 1, 2); row += 1
        grid.addWidget(self.frame_counter_label, row, 0); grid.addWidget(self.frame_spinbox, row, 1); grid.addWidget(self.frame_done_checkbox, row, 2); row += 1
        grid.addWidget(self.prev_frame_btn, row, 0); grid.addWidget(self.next_frame_btn, row, 1); grid.addWidget(self.sync_viewer_frame_btn, row, 2); row += 1
        grid.addWidget(self.next_undone_btn, row, 0, 1, 3); row += 1
        grid.addWidget(self.save_btn, row, 0, 1, 3); row += 1
        grid.addWidget(self.load_btn, row, 0, 1, 3); row += 1

        # --- Section Aide ---
        help_text = (
            "<b>Raccourcis souris :</b><br>"
            "<ul>" # True Positives (TP)
            "<li><b>Clic Droit :</b> TP (Accord)</li>"
            "<li><b>Shift + Clic Droit :</b> TP (Modèle seul)</li>"
            "<li><b>Ctrl + Clic Droit :</b> TP (Matlab seul)</li>"
            "</ul><ul>" # False Positives (FP)
            "<li><b>Shift + Clic Gauche :</b> FP (Modèle)</li>"
            "<li><b>Ctrl + Clic Gauche :</b> FP (Matlab)</li>"
            "</ul><ul>" # False Negatives (FN)
            "<li><b>Clic Molette :</b> FN (Accord)</li>"
            "</ul>"
        )
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        grid.addWidget(QLabel("<b>Aide</b>"), row, 0, 1, 3); row += 1
        grid.addWidget(help_label, row, 0, 1, 3); row += 1

        layout.addLayout(grid); layout.addWidget(self.reset_btn)
        self.setLayout(layout)
        
        # --- Connexions ---
        self.reset_btn.clicked.connect(self._reset_annotations)
        self.viewer.mouse_drag_callbacks.append(self._on_click)
        self.viewer.bind_key('Control-Z', self._undo)
        self.event_selector.currentTextChanged.connect(self._on_selection_change)
        self.viewer.bind_key('v', self._toggle_ref_visibility)
        self.div_threshold_spinbox.valueChanged.connect(self._update_thresholds)
        self.apo_threshold_spinbox.valueChanged.connect(self._update_thresholds)
        # Connexions pour le compteur de frames
        # La frame d'annotation est maintenant indépendante du slider du viewer
        # self.viewer.dims.events.current_step.connect(self._update_frame_display) # Supprimé
        self.frame_done_checkbox.toggled.connect(self._on_frame_done_toggled)
        self.prev_frame_btn.clicked.connect(self._prev_annotation_frame)
        self.next_frame_btn.clicked.connect(self._next_annotation_frame)
        self.next_undone_btn.clicked.connect(self._go_to_next_undone_frame)
        self.sync_viewer_frame_btn.clicked.connect(self._sync_annotation_frame_with_viewer)
        self.frame_spinbox.valueChanged.connect(self._on_spinbox_changed)
        self.save_btn.clicked.connect(self._save_annotations)
        self.load_btn.clicked.connect(self._load_annotations)

        # Initialisation à la frame 0
        # Assurez-vous que les labels des frames validées sont corrects pour tous les événements
        self._update_all_validated_frames_labels()
        # Puis mettez à jour l'affichage de la frame courante
        self.viewer.dims.current_step = (0,) + self.viewer.dims.current_step[1:]
        self._update_frame_display()

    def link_layers(self):
        # ... (inchangé)
        for event in ["Divisions", "Apoptoses"]:
            # Correction: Utiliser le nom 'Matlab' au lieu de 'Ref'
            matlab_ref_layer_name = f'{event} (Matlab)'
            if matlab_ref_layer_name in self.viewer.layers:
                self.layers[event]['ref_matlab'] = self.viewer.layers[matlab_ref_layer_name]
            # On lie aussi la nouvelle couche de référence CSV
            if event == "Divisions":
                if 'Divisions (CSV Ref)' in self.viewer.layers:
                    self.layers[event]['ref_csv'] = self.viewer.layers['Divisions (CSV Ref)']
            model_layer_name = f'{event} model'
            if model_layer_name in self.viewer.layers:
                self.layers[event]['model'] = self.viewer.layers[model_layer_name]
            
            # Liaison des catégories d'annotation de base
            for cat_key, cat_name in self.get_base_categories().items():
                layer_name = f'{cat_name} {event}'
                if layer_name in self.viewer.layers:
                    layer = self.viewer.layers[layer_name]
                    self.layers[event][cat_key] = layer
                    layer.events.data.connect(self._update_counts)
                
        self._update_counts()
        self._on_selection_change(self.event_selector.currentText())
        self._init_thresholds()

    def _init_thresholds(self):
        """Initialise les spinbox avec les seuils actuels des couches."""
        if self.layers["Divisions"].get("model"):
            self.div_threshold_spinbox.setValue(self.layers["Divisions"]["model"].contrast_limits[0])
        if self.layers["Apoptoses"].get("model"):
            self.apo_threshold_spinbox.setValue(self.layers["Apoptoses"]["model"].contrast_limits[0])

    def _update_thresholds(self):
        """Met à jour le seuil de la couche image correspondante."""
        if self.layers["Divisions"].get("model"):
            self.layers["Divisions"]["model"].contrast_limits = (self.div_threshold_spinbox.value(), 1.0)
        if self.layers["Apoptoses"].get("model"):
            self.layers["Apoptoses"]["model"].contrast_limits = (self.apo_threshold_spinbox.value(), 1.0)

    def _update_frame_display(self, event=None):
        """Met à jour l'affichage lié à la frame d'annotation actuelle (spinbox, checkbox)."""
        current_event_type = self.event_selector.currentText()
        current_annotation_frame = self._current_annotation_frame
        
        # Bloquer les signaux pour éviter les appels récursifs
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(current_annotation_frame)
        self.frame_spinbox.blockSignals(False)
        
        self.frame_done_checkbox.blockSignals(True)
        self.frame_done_checkbox.setChecked(self.frame_status[current_event_type][current_annotation_frame]) # Vérifie l'état pour l'événement sélectionné
        self.frame_done_checkbox.blockSignals(False)

        # Gérer l'état des boutons Précédent/Suivant
        self.prev_frame_btn.setEnabled(current_annotation_frame > 0)
        self.next_frame_btn.setEnabled(current_annotation_frame < self.total_frames - 1)
        
        # Mettre à jour le label des frames validées pour l'événement courant
        self._update_all_validated_frames_labels()

    def _on_frame_done_toggled(self, checked):
        """Met à jour le statut de la frame lorsque la case est cochée/décochée."""
        current_event_type = self.event_selector.currentText()
        current_annotation_frame = self._current_annotation_frame
        self.frame_status[current_event_type][current_annotation_frame] = checked # Met à jour le statut pour l'événement sélectionné
        
        # Met à jour tous les compteurs de frames validées
        self._update_all_validated_frames_labels()

        # Passe automatiquement à la frame suivante si la case est cochée
        if checked:
            self._next_annotation_frame()

    def _update_all_validated_frames_labels(self):
        """Met à jour les labels de frames validées pour tous les types d'événements."""
        for event_type in self.frame_status.keys():
            validated_count = sum(self.frame_status[event_type])
            if event_type == "Divisions":
                self.frames_validated_div_label.setText(f"Frames validées: {validated_count} / {self.total_frames}")
            elif event_type == "Apoptoses":
                self.frames_validated_apo_label.setText(f"Frames validées: {validated_count} / {self.total_frames}")

    def _set_annotation_frame(self, frame_index: int):
        """Définit la frame d'annotation actuelle et met à jour le viewer."""
        if 0 <= frame_index < self.total_frames:
            self._current_annotation_frame = frame_index
            # Met à jour le viewer pour afficher la frame d'annotation
            self.viewer.dims.current_step = (frame_index,) + self.viewer.dims.current_step[1:]
            self._update_frame_display() # Met à jour l'interface utilisateur
        else:
            print(f"Frame index {frame_index} est hors limites [0, {self.total_frames-1}]")

    def _next_annotation_frame(self):
        """Passe à la frame d'annotation suivante."""
        self._set_annotation_frame(self._current_annotation_frame + 1)

    def _prev_annotation_frame(self):
        """Passe à la frame d'annotation précédente."""
        self._set_annotation_frame(self._current_annotation_frame - 1)

    def _sync_annotation_frame_with_viewer(self):
        """Synchronise la frame d'annotation avec la frame actuellement affichée dans le viewer."""
        self._set_annotation_frame(self.viewer.dims.current_step[0])

    def _on_spinbox_changed(self, value: int):
        """Appelé lorsque la valeur du spinbox est modifiée par l'utilisateur."""
        self._set_annotation_frame(value)

    def _go_to_next_undone_frame(self):
        """Trouve et saute à la prochaine frame non validée."""
        current_event_type = self.event_selector.currentText()
        start_frame = self._current_annotation_frame
        # Cherche de la frame actuelle à la fin
        for i in range(start_frame + 1, self.total_frames):
            if not self.frame_status[current_event_type][i]:
                self._set_annotation_frame(i)
                return
        print("Toutes les frames suivantes ont déjà été validées.")

    def get_base_categories(self):
        """Retourne un dictionnaire des catégories d'annotation de base (directement cliquables)."""
        return {
            'tp_accord': 'TP (Accord)', # Modèle et Matlab d'accord sur un vrai positif
            'tp_model_only': 'TP (Modèle seul)', # Modèle a raison, Matlab a tort (FN Matlab)
            'tp_matlab_only': 'TP (Matlab seul)', # Matlab a raison, Modèle a tort (FN Modèle)
            'fp_model': 'FP (Modèle)', # Modèle a tort
            'fp_matlab': 'FP (Matlab)', # Matlab a tort
            'fn_both': 'FN (Accord)', # Modèle et Matlab ont tous les deux tort (manqué un vrai positif)
        }

    # --- MÉTHODE _update_counts ---
    def _update_counts(self, event=None):
        """Recompte les points et met à jour l'interface."""
        for event_name, layers in self.layers.items():
            # Obtenir les comptes des catégories de base
            base_counts = {cat_key: len(layers[cat_key].data) if layers.get(cat_key) else 0
                           for cat_key in self.get_base_categories().keys()}

            # Calculer les métriques dérivées
            tp_model = base_counts['tp_accord'] + base_counts['tp_model_only']
            fp_model = base_counts['fp_model']
            fn_model = base_counts['tp_matlab_only'] + base_counts['fn_both']
            tp_matlab = base_counts['tp_accord'] + base_counts['tp_matlab_only']
            fp_matlab = base_counts['fp_matlab']
            fn_matlab = base_counts['tp_model_only'] + base_counts['fn_both']
            
            if event_name == "Divisions":
                self.tp_model_div_label.setText(f"TP (Modèle): {tp_model}")
                self.fp_model_div_label.setText(f"FP (Modèle): {fp_model}")
                self.fn_model_div_label.setText(f"FN (Modèle): {fn_model}")
                self.tp_matlab_div_label.setText(f"TP (Matlab): {tp_matlab}")
                self.fp_matlab_div_label.setText(f"FP (Matlab): {fp_matlab}")
                self.fn_matlab_div_label.setText(f"FN (Matlab): {fn_matlab}")
            elif event_name == "Apoptoses":
                self.tp_model_apo_label.setText(f"TP (Modèle): {tp_model}")
                self.fp_model_apo_label.setText(f"FP (Modèle): {fp_model}")
                self.fn_model_apo_label.setText(f"FN (Modèle): {fn_model}")
                self.tp_matlab_apo_label.setText(f"TP (Matlab): {tp_matlab}")
                self.fp_matlab_apo_label.setText(f"FP (Matlab): {fp_matlab}")
                self.fn_matlab_apo_label.setText(f"FN (Matlab): {fn_matlab}")

    def _reset_annotations(self):
        for event_layers in self.layers.values():
            for cat in self.get_base_categories().keys():
                if event_layers.get(cat):
                    event_layers[cat].data = np.empty((0, 3))
        self.action_history.clear()
        # Réinitialiser aussi le statut des frames
        for event_type in self.frame_status.keys(): # Réinitialise pour chaque type d'événement
            self.frame_status[event_type] = [False] * self.total_frames
        self._current_annotation_frame = 0
        self.viewer.dims.current_step = (0,) + self.viewer.dims.current_step[1:]
        self._update_frame_display()
        print("Toutes les annotations ont été réinitialisées.")

    def _on_selection_change(self, selected_event):
        """Gère la visibilité des couches en fonction de l'événement sélectionné."""
        # D'abord, on parcourt toutes les couches gérées par le widget
        for event_name, event_layers in self.layers.items():
            # On détermine si les couches de cet événement doivent être visibles
            is_visible = (event_name == selected_event)
            # On applique la visibilité à toutes les couches de cet événement
            # (couches de points d'annotation, couche de détection du modèle, couche de référence)
            for layer in event_layers.values():
                if layer:
                    layer.visible = is_visible
        
        # Ensuite, on active/désactive les widgets de l'interface
        for event_name, widgets in self.ui_elements.items():
            is_enabled = (event_name == selected_event)
            for widget in widgets:
                widget.setEnabled(is_enabled)

        print(f"Affichage des couches pour : {selected_event}")
        # Mettre à jour l'affichage de la frame courante et le label des frames validées
        self._update_frame_display() # Met à jour la checkbox et spinbox
        self._update_all_validated_frames_labels() # Met à jour les deux labels de validation

    def _toggle_ref_visibility(self, viewer):
        # ... (inchangé)
        # Amélioration: Bascule la visibilité de TOUTES les couches de référence pour l'événement sélectionné
        selected_event = self.event_selector.currentText()
        event_layers = self.layers.get(selected_event, {})
        
        for key, layer in event_layers.items():
            if 'ref' in key and layer: # Cible 'ref_matlab' et 'ref_csv'
                layer.visible = not layer.visible

    def _undo(self, viewer):
        # ... (inchangé)
        if not self.action_history: return
        target_layer, point_to_remove = self.action_history.pop()
        point_index = np.where(np.all(target_layer.data == point_to_remove, axis=1))[0]
        if len(point_index) > 0:
            target_layer.data = np.delete(target_layer.data, point_index[-1], axis=0)
    
    def _generate_stats_text(self):
        """Génère une chaîne de caractères formatée avec toutes les statistiques."""
        lines = ["RAPPORT DE STATISTIQUES D'ANNOTATION\n", "="*40 + "\n"]
        
        grand_totals = {
            'tp_model': 0, 'fp_model': 0, 'fn_model': 0,
            'tp_matlab': 0, 'fp_matlab': 0, 'fn_matlab': 0
        }

        for event_name, layers in self.layers.items():
            lines.append(f"--- {event_name.upper()} ---\n")
            base_counts = {cat_key: len(layers[cat_key].data) if layers.get(cat_key) else 0
                           for cat_key in self.get_base_categories().keys()}

            tp_model = base_counts['tp_accord'] + base_counts['tp_model_only']
            fp_model = base_counts['fp_model']
            fn_model = base_counts['tp_matlab_only'] + base_counts['fn_both']
            tp_matlab = base_counts['tp_accord'] + base_counts['tp_matlab_only']
            fp_matlab = base_counts['fp_matlab']
            fn_matlab = base_counts['tp_model_only'] + base_counts['fn_both']
            
            lines.append(f"  Modèle:\n    - TP: {tp_model}\n    - FP: {fp_model}\n    - FN: {fn_model}\n")
            lines.append(f"  Matlab:\n    - TP: {tp_matlab}\n    - FP: {fp_matlab}\n    - FN: {fn_matlab}\n")
            validated_count = sum(self.frame_status[event_name])
            lines.append(f"  Frames validées: {validated_count} / {self.total_frames}\n\n")

            grand_totals['tp_model'] += tp_model; grand_totals['fp_model'] += fp_model; grand_totals['fn_model'] += fn_model
            grand_totals['tp_matlab'] += tp_matlab; grand_totals['fp_matlab'] += fp_matlab; grand_totals['fn_matlab'] += fn_matlab

        lines.append("--- TOTAL GLOBAL ---\n")
        lines.append(f"  Modèle:\n    - TP: {grand_totals['tp_model']}\n    - FP: {grand_totals['fp_model']}\n    - FN: {grand_totals['fn_model']}\n")
        lines.append(f"  Matlab:\n    - TP: {grand_totals['tp_matlab']}\n    - FP: {grand_totals['fp_matlab']}\n    - FN: {grand_totals['fn_matlab']}\n")

        return "".join(lines)

    def _save_annotations(self):
        """Sauvegarde l'état actuel des annotations dans un fichier JSON."""
        # S'assurer que le dossier de sauvegarde existe
        os.makedirs(os.path.dirname(ANNOTATION_SAVE_PATH), exist_ok=True)

        # 1. Sauvegarder l'état des points et des frames
        points_data = {}
        for event_name, event_layers in self.layers.items():
            points_data[event_name] = {}
            for cat_key in self.get_base_categories().keys():
                layer = event_layers.get(cat_key)
                if layer:
                    points_data[event_name][cat_key] = layer.data.tolist()

        data_to_save = {
            "frame_status": self.frame_status,
            "points": points_data
        }

        try:
            with open(ANNOTATION_SAVE_PATH, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            print(f"✅ Annotation sauvegardée dans {ANNOTATION_SAVE_PATH}")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde de l'annotation : {e}")

        # 2. Sauvegarder le fichier de statistiques
        stats_text = self._generate_stats_text()
        try:
            with open(STATS_SAVE_PATH, 'w') as f:
                f.write(stats_text)
            print(f"✅ Statistiques sauvegardées dans {STATS_SAVE_PATH}")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde des statistiques : {e}")

    def _load_annotations(self):
        """Charge un état d'annotation depuis un fichier JSON."""
        filePath = ANNOTATION_SAVE_PATH
        try:
            with open(filePath, 'r') as f:
                loaded_data = json.load(f)
            
            self._reset_annotations() # Réinitialise l'état avant de charger
            self.frame_status = loaded_data["frame_status"]
            for event_name, event_points in loaded_data["points"].items():
                for cat_key, points in event_points.items():
                    self.layers[event_name][cat_key].data = np.array(points)
            print(f"✅ Annotation chargée depuis {filePath}")
        except FileNotFoundError:
            print(f"⚠️ Fichier d'annotation non trouvé à l'emplacement : {filePath}")
        except Exception as e:
            print(f"❌ Erreur lors du chargement : {e}")

    def _on_click(self, viewer, event):
        # ... (inchangé)
        if event.type != 'mouse_press': 
            return
        event_type = self.event_selector.currentText()
        coords = np.array(viewer.cursor.position)
        action_type = None
        
        if event.button == 1: # Clic Gauche (pour les Faux Positifs)
            if 'Shift' in event.modifiers: # Shift + Clic Gauche
                action_type = 'fp_model' # FP (Modèle)
            elif 'Control' in event.modifiers: # Ctrl + Clic Gauche
                action_type = 'fp_matlab' # FP (Matlab)
        elif event.button == 2: # Clic Droit
            if 'Shift' in event.modifiers: # Shift + Clic Droit
                action_type = 'tp_model_only' # TP (Modèle seul)
            elif 'Control' in event.modifiers: # Ctrl + Clic Droit
                action_type = 'tp_matlab_only' # TP (Matlab seul)
            else: # Juste Clic Droit
                action_type = 'tp_accord' # TP (Accord)
        elif event.button == 3: # Clic Molette (pour les Faux Négatifs)
            # Pas de modificateur pour Clic Molette seul, car c'est le seul FN restant
            action_type = 'fn_both' # FN (Accord)

        if action_type:
            target_layer = self.layers[event_type].get(action_type)
            if target_layer:
                target_layer.add(coords)
                self.action_history.append((target_layer, coords))


# --- LANCEMENT DE NAPARI ---
print("🚀 Lancement de Napari...")
viewer = napari.Viewer()
# ... (création des couches identique)
viewer.add_image(raw_image_zarr, name='Raw Image'); viewer.add_labels(labels_zarr, name='Segmentation Labels', visible=False)
viewer.add_labels(animal.IMAGE.D2.labels_three6trackia, name = 'Labels from model', visible=False)
viewer.add_image(animal.IMAGE.D2.apoptosis_detection, name = 'Apoptoses model', visible=True, blending="additive",colormap='red', contrast_limits=(0.5, 1.0))
viewer.add_image(animal.IMAGE.D2.division_detection, name = 'Divisions model', visible=True, blending="additive", colormap='blue', contrast_limits=(0.7, 1.0))
# Ajout de la nouvelle couche de points depuis le fichier CSV
if len(division_points_csv) > 0:
    viewer.add_points(division_points_csv, name='Divisions (CSV Ref)', face_color='gold', symbol='diamond', size=15)
viewer.add_points(np.array(division_points_ref), name='Divisions (Matlab)', face_color='cyan', size=10)
viewer.add_points(np.array(apoptosis_points_ref), name='Apoptoses (Matlab)', face_color='lime', size=10)

# Création des nouvelles couches de points pour chaque catégorie et chaque événement
categories = {
    'TP (Accord)': ('green', 'o'), 'TP (Modèle seul)': ('blue', 'star'), 'TP (Matlab seul)': ('purple', 'diamond'),
    'FP (Modèle)': ('red', 'cross'), 'FP (Matlab)': ('magenta', '+'),
    'FN (Accord)': ('yellow', 'x')
}
for event in ["Divisions", "Apoptoses"]:
    for name, (color, symbol) in categories.items(): # Utilise les nouvelles catégories
        viewer.add_points(ndim=3, name=f'{name} {event}', face_color=color, symbol=symbol, size=10)

scoring_widget_instance = ScoringWidget(viewer)
scoring_widget_instance.link_layers()
viewer.window.add_dock_widget(scoring_widget_instance, area='right', name='Compteur d\'Événements')

napari.run()