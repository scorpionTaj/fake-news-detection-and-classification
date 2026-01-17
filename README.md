# 🕵️ Détection et Classification de Fake News avec LLM (RoBERTa) — V3 Optimisée

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.x-yellow.svg)](https://huggingface.co/transformers/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Projet Master SDIA** — NLP & Web Mining  
> Un pipeline **robuste** et **honnête** de Deep Learning pour détecter les fake news, utilisant **RoBERTa** avec stratégies avancées de régularisation, déduplication stricte et seuil dynamique.

---

## 📖 Table des Matières

- [Aperçu du Projet](#-aperçu-du-projet)
- [Démonstration](#-démonstration)
- [Architecture du Modèle](#-architecture-du-modèle)
- [Structure du Projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Pipeline d'Entraînement](#-pipeline-dentraînement)
- [Performances](#-performances)
- [Configuration](#-configuration)
- [Références](#-références)

---

## 🎯 Aperçu du Projet (V3 — Optimisée)

Cette version corrige les biais majeurs présents dans la plupart des projets de détection de fake news grâce à trois innovations techniques :

1. **🛡️ Déduplication Stricte** : Suppression rigoureuse des doublons AVANT le split Train/Val (évite la fuite de données/data leakage)
2. **🧠 Seuil de Décision Dynamique** : Au lieu d'utiliser un seuil fixe (0.50), le modèle trouve le seuil optimal (ex: 0.45 ou 0.60) qui maximise le F1-Score par dataset
3. **📉 Régularisation Avancée** : Dropout renforcé (0.2) + **Focal Loss** pour empêcher l'overfitting sur les datasets bruyants

Le modèle utilise **RoBERTa** de Facebook/Meta, fine-tuné sur les datasets **FakeNewsNet** (GossipCop & Politifact) avec validation stricte.

### Fonctionnalités Principales

| Composant           | V3 — Description                                                           |
| ------------------- | -------------------------------------------------------------------------- |
| 🗃️ **Données**      | **Déduplication stricte** des doublons (1397 supprimés dans GossipCop)     |
| 🧠 **Modèle**       | Fine-tuning de `roberta-base` (125M) avec **Dropout=0.2** anti-overfitting |
| 🔥 **Loss**         | **Focal Loss** ($\gamma=2$) + `WeightedRandomSampler` pour équilibrage     |
| 🎯 **Décision**     | **Seuil Dynamique** : ~0.45 (PolitiFact), ~0.60 (GossipCop)                |
| 🚀 **Optimisation** | AdamW + Linear Warmup + Mixed Precision (FP16) + Early Stopping agressif   |
| 🖥️ **Interface**    | Application web **Gradio** avec thème personnalisé                         |
| 📚 **Pédagogie**    | Démos interactives : tokenisation, analyse d'erreurs, visualisations       |

---

## 🎬 Démonstration

L'application analyse un texte et retourne :

- **✅ Vrai (Real)** : Contenu véridique et factuel
- **🚨 Faux (Fake)** : Contenu potentiellement trompeur ou fabriqué

```
📰 Titre : "Pope Francis endorses Donald Trump for president."
   Résultat : 🚨 FAUX (FAKE)
   Confiance: [████████████████░░░░] 82.3%
```

---

## 🏗️ Architecture du Modèle

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT TEXT                           │
│         "Scientists confirm the earth is flat."             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    RoBERTa TOKENIZER                        │
│  Tokens: ['Scientists', 'Ġconfirm', 'Ġthe', 'Ġearth', ...]  │
│  IDs:    [10868, 5765, 5, 4015, 16, 5765, 4, ...]           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 RoBERTa ENCODER (12 layers)                 │
│            Attention Heads: 12 | Hidden: 768                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              CLASSIFICATION HEAD (Linear Layer)             │
│                    768 → 2 (Real/Fake)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      SOFTMAX OUTPUT                         │
│              [P(Real)=0.12, P(Fake)=0.88]                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Structure du Projet

```
fake-news-detection-and-classification/
│
├── 📓 fake-news-detection-and-classification-using-llm.ipynb
│       └── Notebook principal d'entraînement (9 sections détaillées)
│
├── 🚀 app.py
│       └── Application Gradio pour l'inférence en temps réel
│
├── 📋 requirements.txt
│       └── Dépendances Python du projet
│
├── 📖 README.md
│       └── Documentation principale (ce fichier)
│
└── 🤖 mon_modele_fake_news/
        ├── config.json              # Configuration architecture RoBERTa
        ├── model.safetensors        # Poids du modèle (format sécurisé)
        ├── vocab.json               # Vocabulaire (50265 tokens)
        ├── merges.txt               # Règles de fusion BPE
        ├── tokenizer_config.json    # Configuration du tokenizer
        └── special_tokens_map.json  # Tokens spéciaux (<s>, </s>, <pad>)
```

---

## ⚙️ Installation

### Prérequis

- **Python:** 3.8 ou supérieur
- **GPU:** Recommandé pour l'entraînement (NVIDIA CUDA), optionnel pour l'inférence
- **RAM:** 8 Go minimum (16 Go recommandé)

### Étapes d'Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/scorpionTaj/fake-news-detection-and-classification.git

# 2. Accéder au répertoire
cd fake-news-detection-and-classification

# 3. (Optionnel) Créer un environnement virtuel
python -m venv venv
# source venv/bin/activate     # Pour Bash/Zsh

# 4. Installer les dépendances
pip install -r requirements.txt
```

### Dépendances Principales

| Package                  | Version | Utilité                         |
| ------------------------ | ------- | ------------------------------- |
| `torch`                  | ≥2.0    | Framework Deep Learning         |
| `transformers`           | ≥4.30   | Modèles pré-entraînés (RoBERTa) |
| `gradio`                 | ≥4.0    | Interface web interactive       |
| `scikit-learn`           | ≥1.0    | Métriques d'évaluation          |
| `matplotlib` / `seaborn` | -       | Visualisations                  |

---

## 🚀 Utilisation

### 1. Lancer l'Application Web (Inférence)

Si vous avez déjà le modèle entraîné dans `mon_modele_fake_news/` :

```bash
python app.py
```

Ouvrez `http://127.0.0.1:7860` dans votre navigateur.

### 2. Entraîner le Modèle (Notebook)

Ouvrez le notebook dans Jupyter ou Google Colab :

```bash
jupyter notebook fake-news-detection-and-classification-using-llm.ipynb
```

---

## 🔄 Pipeline d'Entraînement

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   DONNÉES    │ → │ PRÉPARATION  │ → │ ENTRAÎNEMENT │ → │  ÉVALUATION  │
│  FakeNewsNet │    │  Nettoyage   │    │   RoBERTa    │    │   F1-Score   │
│  (CSV URLs)  │    │  Tokenisation│    │   Fine-tune  │    │   Confusion  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                    ┌──────────────┐    ┌──────────────┐            │
                    │   DÉMO WEB   │ ← │  SAUVEGARDE  │ ←──────────┘
                    │    Gradio    │    │  .safetensors│
                    └──────────────┘    └──────────────┘
```

### Étapes Détaillées

1. **Chargement** : Téléchargement des 4 CSV (Politifact + GossipCop × Real/Fake)
2. **EDA** : Analyse exploratoire (distribution, doublons, valeurs manquantes)
3. **Prétraitement** : Nettoyage, tokenisation BPE, padding/truncation
4. **Équilibrage** : **Focal Loss (γ=2)** + WeightedRandomSampler (double stratégie)
5. **Fine-tuning** : Jusqu'à 100 époques, Mixed Precision, Early Stopping (patience=4)
6. **Évaluation** : F1-Score, Matrice de confusion, Rapport de classification
7. **Export** : Sauvegarde au format Hugging Face (.safetensors)

---

## 📊 Performances V3 (Sans Fuite de Données)

Contrairement aux approches classiques qui gonflent les scores via des doublons, ces résultats sont **honnêtes** et validés sur des données uniques après déduplication stricte.

### 🏛️ PolitiFact (Politique)

> Modèle très performant, capable de saisir les nuances politiques.

| Métrique          | Résultat |
| ----------------- | -------- |
| **F1-Score**      | ~0.89    |
| **Seuil Optimal** | **0.45** |
| **Erreurs**       | ~10/148  |

### 🌟 GossipCop (Célébrités)

> Dataset difficile et bruyant (tabloïds), stabilisé par Dropout=0.2.

| Métrique               | Résultat             |
| ---------------------- | -------------------- |
| **F1-Score**           | ~0.67                |
| **Seuil Optimal**      | **0.60**             |
| **Gain V3**            | Overfitting maîtrisé |
| **Doublons supprimés** | 1397                 |

### Labels

- **Label 0** : ✅ Vrai (Real) — Article vérifié comme factuel
- **Label 1** : 🚨 Faux (Fake) — Article identifié comme trompeur

> **Note** : Les scores V3 sont légèrement inférieurs à V2 (0.85 → 0.67 sur GossipCop) car la déduplication a supprimé les doublons qui gonflaient artificiellement les métriques. Ces scores V3 reflètent la **vraie** capacité du modèle.

---

## 🛠️ Configuration V3 (Robuste)

Les hyperparamètres ont été ajustés pour la stabilité et l'honnêteté des évaluations :

```python
class ProjectConfig:
    SEED = 42              # Reproductibilité
    MAX_LEN = 128          # Longueur max des séquences
    BATCH_SIZE = 32        # Petit batch pour meilleure généralisation
    EPOCHS = 8             # Early Stopping agressif pour capturer le pic
    LEARNING_RATE = 1e-5   # Taux très faible pour fine-tuning précis
    WEIGHT_DECAY = 0.1     # Régularisation L2
    DROPOUT_RATE = 0.2     # AUGMENTÉ (0.1 → 0.2) pour anti-overfitting
    PATIENCE = 4           # Early stopping après 4 époques
    MODEL_NAME = 'roberta-base'
```

### Focal Loss + Dropout Renforcé

Le projet combine **Focal Loss** + **Dropout augmenté** pour combattre le surapprentissage (overfitting), particulièrement sur GossipCop (dataset bruyant).

| Technique           | Paramètre        | Rôle                                                                          |
| ------------------- | ---------------- | ----------------------------------------------------------------------------- |
| **Focal Loss**      | gamma=2.0        | Réduit le poids des exemples faciles, focus sur les fakes subtils             |
| **Dropout**         | 0.2 (20%)        | Force le modèle à apprendre les patterns robustes, pas les titres spécifiques |
| **WeightedSampler** | Auto-équilibrage | Assure que chaque batch contient 50/50 Vrai/Faux                              |

### Seuil Dynamique (Nouveauté V3)

Au lieu d'utiliser le seuil classique de 0.50, le modèle calcule automatiquement le seuil optimal pour chaque dataset. Cela permet d'adapter la sensibilité du modèle selon la distribution des données.

**Résultats** :

- **PolitiFact** → Seuil **0.45** (être soupçonneux pour ne rien rater)
- **GossipCop** → Seuil **0.60** (être strict pour filtrer le bruit)

---

## 📚 Références

### Datasets

- [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) — Shu et al., 2020

### Modèle

- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — Liu et al., 2019

### Librairies

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Gradio Documentation](https://gradio.app/docs/)
- [PyTorch](https://pytorch.org/)

---

## 👤 Auteur

**scorpionTaj** — Master SDIA, Université Moulay Ismail
**ana3ss7z** — Master SDIA, Université Moulay Ismail
**Nawfal Khallou** — Master SDIA, Université Moulay Ismail

---

<p align="center">
  <i>Développé avec ❤️ pour le cours de NLP & Web Mining</i>
</p>
