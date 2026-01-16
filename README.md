# 🕵️ Détection et Classification de Fake News avec LLM (RoBERTa)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.x-yellow.svg)](https://huggingface.co/transformers/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Projet Master SDIA** — NLP & Web Mining  
> Un pipeline complet de Deep Learning pour détecter les fake news en utilisant le modèle **RoBERTa** (Robustly Optimized BERT Approach).

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

## 🎯 Aperçu du Projet

Ce projet implémente un système de détection de fake news basé sur l'apprentissage profond. Il utilise le modèle pré-entraîné **RoBERTa** de Facebook/Meta, fine-tuné sur les datasets **FakeNewsNet** (GossipCop & Politifact).

### Fonctionnalités Principales

| Composant           | Description                                                                     |
| ------------------- | ------------------------------------------------------------------------------- |
| 🗃️ **Données**      | Datasets **GossipCop** (célébrités) & **Politifact** (politique) de FakeNewsNet |
| 🧠 **Modèle**       | Fine-tuning de `roberta-base` (125M paramètres) pour classification binaire     |
| ⚖️ **Équilibrage**  | **Focal Loss** + `WeightedRandomSampler` pour combattre le déséquilibre         |
| 🚀 **Optimisation** | AdamW + Linear Warmup + Mixed Precision (FP16) + Early Stopping                 |
| 🖥️ **Interface**    | Application web **Gradio** avec thème personnalisé                              |
| 📚 **Pédagogie**    | Démos interactives : tokenisation, analyse d'erreurs, visualisations            |

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
├── 📖 GUIDE_NOTEBOOK_FR.md
│       └── Guide pédagogique détaillé du notebook (en français)
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

> 📖 Consultez [GUIDE_NOTEBOOK_FR.md](GUIDE_NOTEBOOK_FR.md) pour une explication détaillée de chaque cellule.

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

## 📊 Performances

### Résultats sur GossipCop (Dataset Principal)

| Métrique             | Score |
| -------------------- | ----- |
| **F1-Score**         | ~0.85 |
| **Précision (Vrai)** | ~0.87 |
| **Rappel (Faux)**    | ~0.82 |

### Labels

- **Label 0** : ✅ Vrai (Real) — Article vérifié comme factuel
- **Label 1** : 🚨 Faux (Fake) — Article identifié comme trompeur

---

## 🛠️ Configuration

Les hyperparamètres sont définis dans la classe `ProjectConfig` du notebook :

```python
class ProjectConfig:
    SEED = 42              # Reproductibilité
    MAX_LEN = 128          # Longueur max des séquences
    BATCH_SIZE = 32        # Taille des lots (réduit pour stabilité)
    EPOCHS = 100           # Nombre max d'époques (Early Stopping actif)
    LEARNING_RATE = 1e-5   # Taux d'apprentissage (plus conservateur)
    WEIGHT_DECAY = 0.1     # Régularisation L2
    PATIENCE = 4           # Early stopping après 4 époques sans amélioration
    MODEL_NAME = 'roberta-base'
```

### Focal Loss

Le projet utilise la **Focal Loss** au lieu de la Cross-Entropy standard :

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        # gamma=2 : Focus sur les exemples difficiles (classe minoritaire)
```

| Paramètre | Valeur | Effet                                                          |
| --------- | ------ | -------------------------------------------------------------- |
| `gamma`   | 2.0    | Réduit le poids des exemples faciles, focus sur les difficiles |
| `alpha`   | 1.0    | Poids égal pour les deux classes                               |

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
