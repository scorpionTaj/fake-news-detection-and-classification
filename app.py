import gradio as gr
import torch
import numpy as np
import os
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. SETUP & CONFIGURATION (V3)
# ==========================================
MODEL_PATH = "mon_modele_fake_news"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Best Model Configuration
OPTIMAL_THRESHOLD = 0.45
BEST_DATASET = "politifact"

def load_model_and_tokenizer():
    """
    Loads the fine-tuned RoBERTa model and tokenizer from the specified directory.
    """
    print(f"🤖 Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model directory not found at: {MODEL_PATH}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        print("✅ Model loaded successfully on", DEVICE)
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

# Initialize model and tokenizer
tokenizer, model = load_model_and_tokenizer()

# ==========================================
# 2. PIPELINE PROCESSING FUNCTION
# ==========================================
def process_pipeline(text):
    """
    Complete pipeline: Text → Tokenization → Inference → Threshold → Result
    Returns detailed information about each step
    """
    if not text or not text.strip():
        return None, None, None, None

    if model is None or tokenizer is None:
        return None, None, None, "❌ Model not loaded"

    # Step 1: Tokenization
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )

    num_tokens = inputs['input_ids'].shape[1]
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Step 2: Model Inference
    with torch.no_grad():
        outputs = model(**{k: v.to(DEVICE) for k, v in inputs.items()})
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    prob_real = float(probs[0])
    prob_fake = float(probs[1])

    # Step 3: Apply Optimal Threshold (Best Dataset: PolitiFact)
    prediction = "🚨 FAUX (FAKE)" if prob_fake >= OPTIMAL_THRESHOLD else "✅ VRAI (REAL)"

    # Build pipeline info
    pipeline_info = {
        "tokens": tokens,
        "num_tokens": num_tokens,
        "threshold": OPTIMAL_THRESHOLD,
        "prediction": prediction
    }

    return {
        "✅ Vrai (Real)": prob_real,
        "🚨 Faux (Fake)": prob_fake
    }, pipeline_info, prob_real, prob_fake

# ==========================================
# 3. GRADIO INTERFACE (V3 Enhanced)
# ==========================================
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    text_size="lg",
    font=[gr.themes.GoogleFont("Ubuntu Mono"), "ui-monospace", "monospace"]
)

with gr.Blocks(title="Détection de Fake News - V3 Optimisée", theme=theme) as demo:

    # ===== HEADER =====
    gr.Markdown(
        """
        <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; border-radius: 15px; margin-bottom: 30px;">
            <h1 style="color: white; font-size: 2.8rem; margin: 0; font-weight: bold;">
                🕵️ Détecteur de Fake News — V3 Optimisée
            </h1>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-top: 10px;">
                Analyse intelligente de la crédibilité des articles avec <b>RoBERTa</b> + <b>Focal Loss</b>
            </p>
            <div style="display: flex; gap: 15px; justify-content: center; margin-top: 15px; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 20px; color: white; font-size: 0.9rem;">
                    🛡️ Déduplication Stricte
                </span>
                <span style="background: rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 20px; color: white; font-size: 0.9rem;">
                    🎯 Seuil Dynamique
                </span>
            </div>
        </div>
        """
    )

    # ===== MAIN INPUT/OUTPUT SECTION =====
    with gr.Row(equal_height=False):
        with gr.Column(scale=1.2):
            gr.Markdown("### 📝 Étape 1 : Entrée du Texte")
            input_text = gr.Textbox(
                label="Article ou Titre à Analyser",
                placeholder="Collez le titre ou le contenu d'un article ici...",
                lines=10,
                max_lines=20,
                info="✓ Supporte l'anglais et le français | Max 128 tokens"
            )

            with gr.Row():
                clear_btn = gr.Button("🗑️ Réinitialiser", variant="secondary", scale=1)
                analyze_btn = gr.Button("🔍 Analyser", variant="primary", scale=2)

        with gr.Column(scale=1.2):
            gr.Markdown("### 📊 Étape 5 : Résultat Final")
            label_output = gr.Label(
                label="Classification",
                num_top_classes=2,
                scale=1
            )

            with gr.Row():
                confidence_real = gr.Textbox(label="🟢 Confiance (Vrai)", interactive=False, value="0%")
                confidence_fake = gr.Textbox(label="🔴 Confiance (Faux)", interactive=False, value="0%")

    # ===== PIPELINE VISUALIZATION =====
    gr.Markdown("---")
    gr.Markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <h3>🔄 Pipeline de Traitement</h3>
        <p style="color: #666; font-size: 0.95rem;">Comment votre texte est transformé en prédiction</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            gr.Markdown("""
            <div style="text-align: center; padding: 20px; background: #f0f4ff; border-radius: 10px; border: 2px solid #667eea;">
                <h4>1️⃣ Tokenisation</h4>
                <p style="font-size: 0.9rem; color: #555;">Découpe du texte en tokens</p>
            </div>
            """)

        with gr.Column(scale=0.3, min_width=30):
            gr.Markdown("<div style='text-align: center; font-size: 1.5rem; color: #667eea;'>→</div>")

        with gr.Column(scale=1, min_width=100):
            gr.Markdown("""
            <div style="text-align: center; padding: 20px; background: #fff4f0; border-radius: 10px; border: 2px solid #764ba2;">
                <h4>2️⃣ Encodage RoBERTa</h4>
                <p style="font-size: 0.9rem; color: #555;">Analyse par réseau de neurones</p>
            </div>
            """)

        with gr.Column(scale=0.3, min_width=30):
            gr.Markdown("<div style='text-align: center; font-size: 1.5rem; color: #667eea;'>→</div>")

        with gr.Column(scale=1, min_width=100):
            gr.Markdown("""
            <div style="text-align: center; padding: 20px; background: #f0fff4; border-radius: 10px; border: 2px solid #48bb78;">
                <h4>3️⃣ Seuil Optimal</h4>
                <p style="font-size: 0.9rem; color: #555;">Décision PolitiFact</p>
            </div>
            """)

    # ===== PIPELINE DETAILS =====
    gr.Markdown("---")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔬 Détails du Traitement")
            tokens_display = gr.Textbox(
                label="Tokens Détectés",
                interactive=False,
                lines=4,
                max_lines=6
            )
            num_tokens_display = gr.Number(label="Nombre de Tokens", interactive=False)

        with gr.Column():
            gr.Markdown("### ⚙️ Configuration V3 Optimale")
            threshold_display = gr.Number(label="🎯 Seuil Optimal (PolitiFact)", interactive=False, value=OPTIMAL_THRESHOLD)

    # ===== EXAMPLES SECTION =====
    gr.Markdown("---")
    gr.Markdown("### 🧪 Exemples Prêts à Tester")
    gr.Examples(
        examples=[
            ["Pope Francis endorses Donald Trump for president."],
            ["The economy grew by 2% last quarter as reported by the central bank."],
            ["Scientists confirm that vaccines cause magnetism."],
            ["NASA successfully launched the Artemis I mission to the Moon."],
            ["Local politician passes environmental protection law."],
            ["Government announces new healthcare reform initiative."]
        ],
        inputs=[input_text],
        outputs=[label_output, tokens_display, confidence_real, confidence_fake],
        fn=lambda text: (*process_pipeline(text)[:2], *process_pipeline(text)[2:]),
        cache_examples=False,
        label="Cliquez pour tester"
    )

    # ===== FOOTER =====
    gr.Markdown("""
    ---
    <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
        <p style="color: #666; margin: 0;">
            🎓 <b>Projet Master SDIA</b> — NLP & Web Mining
        </p>
        <p style="color: #999; font-size: 0.9rem; margin-top: 8px;">
            Développé avec ❤️ • Modèle: RoBERTa <br>
            Dataset: FakeNewsNet (GossipCop + Politifact) • Version: V3 Optimisée
        </p>
    </div>
    """)

    # ===== EVENT LISTENERS =====
    def on_analyze(text):
        result, pipeline_info, conf_real, conf_fake = process_pipeline(text)
        if result is None:
            return None, "❌ Erreur: Entrez un texte valide", 0, OPTIMAL_THRESHOLD, "0%", "0%"

        tokens_str = ", ".join(pipeline_info["tokens"][:20])  # Show first 20 tokens
        if len(pipeline_info["tokens"]) > 20:
            tokens_str += "..."

        # Convert to percentage with 2 decimals
        conf_real_pct = f"{round(conf_real * 100, 2)}%"
        conf_fake_pct = f"{round(conf_fake * 100, 2)}%"

        return (
            result,
            tokens_str,
            pipeline_info["num_tokens"],
            pipeline_info["threshold"],
            conf_real_pct,
            conf_fake_pct
        )

    analyze_btn.click(
        fn=on_analyze,
        inputs=[input_text],
        outputs=[label_output, tokens_display, num_tokens_display, threshold_display, confidence_real, confidence_fake]
    )

    input_text.submit(
        fn=on_analyze,
        inputs=[input_text],
        outputs=[label_output, tokens_display, num_tokens_display, threshold_display, confidence_real, confidence_fake]
    )

    clear_btn.click(
        lambda: (None, "", 0, OPTIMAL_THRESHOLD, "0%", "0%"),
        outputs=[input_text, tokens_display, num_tokens_display, threshold_display, confidence_real, confidence_fake]
    )

# ===== LAUNCH =====
if __name__ == "__main__":
    demo.launch(theme=theme)
