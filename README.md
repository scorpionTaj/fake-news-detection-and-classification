# Détection et Classification de Fake News avec LLM (RoBERTa)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/Transformers-4.x-green.svg)](https://huggingface.co/transformers/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)](https://gradio.app/)

Ce projet propose un pipeline complet de Deep Learning pour détecter les fake news en utilisant le modèle **RoBERTa** (Robustly Optimized BERT Approach). Il inclut un notebook d'entraînement détaillé et une application web interactive basée sur **Gradio** pour tester le modèle en temps réel.

## 📁 Structure du Projet

```
├── fake-news-detection-and-classification-using-llm.ipynb  # Notebook d'entraînement principal
├── app.py                                                   # Application Gradio pour l'inférence
├── requirements.txt                                         # Dépendances Python
├── README.md                                               # Ce fichier
└── mon_modele_fake_news/                                  # Dossier du modèle entraîné
    ├── config.json                                         # Configuration du modèle
    ├── model.safetensors                                   # Poids du modèle
    ├── vocab.json                                          # Vocabulaire
    ├── merges.txt                                          # Fichiers de fusion BPE
    ├── tokenizer_config.json                               # Config du tokenizer
    └── special_tokens_map.json                             # Map des tokens spéciaux
```

## 🎯 Fonctionnalités Clés

| Composant | Détails |
|Chart | Description |
| **Données** | Utilisation des datasets **GossipCop** & **Politifact** (FakeNewsNet). |
| **Modèle** | Fine-tuning de `roberta-base` pour la classification binaire. |
| **Entraînement** | Optimiseur AdamW, Warmup, Sampling pondéré (Weighted Sampler) pour le déséquilibre des classes. |
| **Interface** | Application web **Gradio** pour tester des phrases personnalisées. |
| **Pédagogie** | Le notebook inclut des démos explicatives sur la tokenisation et l'analyse d'erreurs. |

## ⚙️ Prérequis

- **Python:** 3.8 ou supérieur
- **GPU:** Recommandé pour l'entraînement (Google Colab ou GPU local), Optionnel pour l'inférence.

### Installation

1. Cloner le projet :

   ```bash
   cd "Votre/Chemin/Vers/Le/Projet"
   ```

2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
   _(Assurez-vous d'avoir `gradio`, `torch`, `transformers`, `scikit-learn` installés)_

## 🚀 Utilisation

### 1. Entraînement du Modèle (Notebook)

Ouvrez et exécutez le notebook `fake-news-detection-and-classification-using-llm.ipynb` pour :

- Télécharger et préparer les données.
- Entraîner le modèle RoBERTa.
- Évaluer les performances (F1-score, Matrice de confusion).
- Sauvegarder le modèle dans le dossier `mon_modele_fake_news`.

### 2. Lancer l'Application Web (Demo)

Une fois le modèle entraîné (ou si vous avez déjà le dossier `mon_modele_fake_news`), lancez l'interface :

```bash
python app.py
```

Ouvrez ensuite le lien local affiché (généralement `http://127.0.0.1:7860`) dans votre navigateur.

## 📊 Performances Attendues

Le modèle est évalué principalement sur le dataset **GossipCop**.

- **Label 0 :** Vrai (Real)
- **Label 1 :** Faux (Fake)

L'application affiche la probabilité de confiance pour chaque classe.

## 🛠️ Configuration du Modèle

Le modèle utilisé est `roberta-base` fine-tuné avec les hyperparamètres suivants (configurables dans le notebook) :

- **Max Len:** 128 tokens
- **Batch Size:** 64
- **Learning Rate:** 2e-5
- **Epochs:** 5

## 📚 Références

- [FakeNewsNet Dataset](https://github.com/KaiDMML/FakeNewsNet)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
