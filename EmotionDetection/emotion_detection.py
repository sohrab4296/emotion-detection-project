"""Emotion Detection module for Watson NLP service."""
import requests


def emotion_detector(text_to_analyze):
    """
    Analyze text for emotions using IBM Watson NLP service.
    
    Args:
        text_to_analyze (str): Text to analyze for emotional content.
        
    Returns:
        dict: Dictionary containing emotion scores and dominant emotion.
              Keys: anger, disgust, fear, joy, sadness, dominant_emotion
              Returns None values if text is empty or API error occurs.
    """
    # Check for blank or null input
    if not text_to_analyze or text_to_analyze.strip() == "":
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }

    url = (
        "https://sn-watson-emotion.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )

    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }

    payload = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)

        # Handle bad request (status code 400)
        if response.status_code == 400:
            return {
                "anger": None,
                "disgust": None,
                "fear": None,
                "joy": None,
                "sadness": None,
                "dominant_emotion": None
            }

        response.raise_for_status()
        response_data = response.json()

        # Extract emotions dictionary
        emotions = response_data.get("emotionPredictions", [{}])[0].get("emotion", {})

        anger_score = emotions.get("anger", 0)
        disgust_score = emotions.get("disgust", 0)
        fear_score = emotions.get("fear", 0)
        joy_score = emotions.get("joy", 0)
        sadness_score = emotions.get("sadness", 0)

        # Determine dominant emotion
        dominant_emotion = max(emotions, key=emotions.get) if emotions else None

        return {
            "anger": anger_score,
            "disgust": disgust_score,
            "fear": fear_score,
            "joy": joy_score,
            "sadness": sadness_score,
            "dominant_emotion": dominant_emotion
        }

    except requests.RequestException:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }
