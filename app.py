import gradio as gr
import torch
import numpy as np
import os
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Path to the directory containing the model files (config.json, model.safetensors, etc.)
MODEL_PATH = "mon_modele_fake_news"

def load_model_and_tokenizer():
    """
    Loads the fine-tuned RoBERTa model and tokenizer from the specified directory.
    """
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model directory not found at: {MODEL_PATH}")

    try:
        # Load tokenizer and model
        # Using Auto classes is safer for general loading, but notebook used Roberta specific classes.
        # Given we know it's RoBERTa, explicit is fine, or Auto. Auto is more robust.
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval() # Set to evaluation mode
        print("✅ Model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

# Initialize model and tokenizer
tokenizer, model = load_model_and_tokenizer()

# ==========================================
# 2. PREDICTION FUNCTION
# ==========================================
def predict_fake_news(text):
    """
    Predicts whether the input text is Real or Fake news.
    """
    if not text or not text.strip():
        return None

    if model is None or tokenizer is None:
        return {"Error": 0.0, "Model not loaded": 0.0}

    # Preprocess the text (Tokenization)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).numpy()[0]

    # Map probabilities to labels
    # Based on notebook: Class 0 = Real, Class 1 = Fake
    prob_real = float(probs[0])
    prob_fake = float(probs[1])

    # Return dictionary for Gradio Label component
    return {
        "✅ Vrai (Real)": prob_real,
        "🚨 Faux (Fake)": prob_fake
    }

# ==========================================
# 3. GRADIO INTERFACE
# ==========================================
# Create a custom theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    text_size="lg",
    font=[gr.themes.GoogleFont("Ubuntu Mono"), "ui-monospace", "monospace"]
)

with gr.Blocks(title="Détection de Fake News") as demo:
    with gr.Column(elem_id="main_container"):
        gr.Markdown(
            """
            <div style="text-align: center; max-width: 800px; margin: 0 auto;">
                <h1 style="color: #2b39cc; font-size: 2.5rem; margin-bottom: 20px;">
                    🕵️ Détecteur de Fake News
                </h1>
                <p style="font-size: 1.2rem; margin-bottom: 10px;">
                    Analysez la crédibilité de vos articles grâce à l'Intelligence Artificielle.
                </p>
                <div style="padding: 10px; background-color: #f0f4ff; color: #333; border-radius: 10px; display: inline-block;">
                    <span style="font-weight: bold; color: #333;">Modèle :</span> RoBERTa Fine-Tuned
                </div>
            </div>
            """
        )

        gr.HTML("<div style='height: 30px;'></div>")

        with gr.Row(variant="panel", equal_height=True):
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="📝 Entrée du texte",
                    placeholder="Copiez le titre ou le contenu d'un article ici pour vérifier sa véracité...",
                    lines=8,
                    max_lines=15,
                    info="Supporte l'anglais."
                )
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Effacer", variant="secondary")
                    analyze_btn = gr.Button("🔍 Analyser la véracité", variant="primary", scale=2)

            with gr.Column(scale=1):
                label_output = gr.Label(
                    label="📊 Résultat de l'analyse",
                    num_top_classes=2,
                    scale=1
                )
                gr.Markdown(
                    """
                    ### ℹ️ Légende
                    - **✅ Vrai (Real)** : Contenu véridique et factuel.
                    - **🚨 Faux (Fake)** : Contenu trompeur ou fabriqué.
                    """
                )

        gr.HTML("<div style='height: 30px;'></div>")

        gr.Markdown("### 🧪 Exemples Prêts à l'Emploi")
        gr.Examples(
            examples=[
                ["Pope Francis endorses Donald Trump for president.", "Exemple Faux (Connu)"],
                ["The economy grew by 2% last quarter as reported by the central bank.", "Exemple Vrai (Écon)"],
                ["Scientists confirm that vaccines cause magnetism.", "Exemple Faux (Santé)"],
                ["NASA successfully launched the Artemis I mission to the Moon.", "Exemple Vrai (Science)"]
            ],
            inputs=[input_text], # Pass as list to match examples
            outputs=label_output,
            fn=predict_fake_news,
            cache_examples=False,
            label="Cliquez sur un exemple pour tester"
        )

        # Footer
        gr.Markdown(
            """
            <div style="text-align: center; margin-top: 40px; color: #888; font-size: 0.9rem;">
                Developed for Master SDIA • NLP Project • Powered by Hugging Face Transformers
            </div>
            """
        )

    # Event listeners
    analyze_btn.click(
        fn=predict_fake_news,
        inputs=input_text,
        outputs=label_output
    )

    # Also trigger on submit (Enter key in textbox)
    input_text.submit(
        fn=predict_fake_news,
        inputs=input_text,
        outputs=label_output
    )

    clear_btn.click(lambda: (None, None), outputs=[input_text, label_output])

# Launch the app
if __name__ == "__main__":
    demo.launch(theme=theme)
