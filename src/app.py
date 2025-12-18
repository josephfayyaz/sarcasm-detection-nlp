"""
Minimal FastAPI application for serving the BESSTIE sarcasm/sentiment detector.

This API exposes a single endpoint, ``/predict``, which accepts a JSON
payload containing a list of texts and returns the predicted labels (0 or
1).  The model and tokenizer are loaded at startup to avoid overhead on
each request.  To run the server locally::

    uvicorn app:app --host 0.0.0.0 --port 8000

Make sure to install ``fastapi`` and ``uvicorn``.  You can then query the
endpoint via HTTP POST:

    curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" \
         -d '{"texts": ["I love this", "This is awful"]}'
"""

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from inference import predict_binary


class PredictRequest(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    predictions: List[int]


def create_app(model_name: str, checkpoint_dir: str) -> FastAPI:
    """Factory to create a FastAPI app with pre‑loaded model.

    Parameters
    ----------
    model_name : str
        Base model architecture used for training.
    checkpoint_dir : str
        Directory containing the fine‑tuned model files.

    Returns
    -------
    FastAPI
        Configured application.
    """
    app = FastAPI(title="BESSTIE Figurative Language Detection API")
    # Pre‑load model and tokenizer once
    # Use closure to capture them in endpoint
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()
    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        # Perform tokenisation and prediction
        inputs = tokenizer(request.texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = logits.argmax(dim=-1).tolist()
        return PredictResponse(predictions=preds)
    return app


# Instantiate default app using environment variables or hardcoded paths
import os
MODEL_NAME = os.environ.get("BESSTIE_MODEL_NAME", "roberta-base")
CHECKPOINT_DIR = os.environ.get("BESSTIE_CHECKPOINT_DIR", "./model_output")
app = create_app(MODEL_NAME, CHECKPOINT_DIR)