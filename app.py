"""
End-of-turn detection Python implementation
Original source: LiveKit Agents Project

"""

from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
from pathlib import Path
import time

# Constants
UNLIKELY_THRESHOLD = 0.15

chat_example1 = [
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "It's sunny and warm."},
    {"role": "user", "content": "I like the weather. but"},
    {"role": "user", "content": "I'm not sure what to"}
]

chat_example2 = [
    {"role": "user", "content": "¿Cómo está el clima hoy?"},
    {"role": "assistant", "content": "Hoy hace sol y calor."},
    {"role": "user", "content": "Me gusta el clima, pero"},
    {"role": "user", "content": "No sé qué"}
]

chat_example3 = [
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "It's sunny and warm."},
    {"role": "user", "content": "I like the weather. but"},
    {"role": "user", "content": "I'm not sure what to do? But maybe"}
]

def initialize_model():
    """
    Initialize the ONNX model and tokenizer.
    
    Returns:
        tuple: (tokenizer, session, eou_index)
    """
    try:
        start_time = time.time()
        
        # Get the current file's directory and construct model path
        model_path = 'models/model_quantized.onnx'
        session = ort.InferenceSession(str(model_path))
        
        tokenizer = AutoTokenizer.from_pretrained('models')
        eou_index = tokenizer.encode("<|im_end|>")[0]
        
        print(f"Model initialization took: {time.time() - start_time:.2f} seconds")
        return tokenizer, session, eou_index
    
    except Exception as e:
        print(f"Error: {e}")
        raise

def normalize(text):
    """
    Normalize the input text by removing punctuation and standardizing whitespace.
    """
    PUNCS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'  # Punctuation without single quote
    
    # Remove punctuation and normalize whitespace
    stripped = ''.join(char for char in text if char not in PUNCS)
    return ' '.join(stripped.lower().split())

def format_chat_context(chat_context, tokenizer):
    """
    Format the chat context for model input.
    """
    # Normalize and filter empty messages
    normalized_context = [
        msg for msg in [
            {**msg, 'content': normalize(msg['content'])}
            for msg in chat_context
        ]
        if msg['content']
    ]
    
    # Apply chat template
    convo_text = tokenizer.apply_chat_template(
        normalized_context,
        add_generation_prompt=True,
        add_special_tokens=False,
        tokenize=False
    )
    
    # Handle end of utterance token
    eou_token = "<|im_end|>"
    last_eou_index = convo_text.rfind(eou_token)
    return convo_text[:last_eou_index] if last_eou_index >= 0 else convo_text

def softmax(logits):
    """
    Compute softmax probabilities for logits.
    """
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

def predict_end_of_turn(chat_context, model_data):
    """
    Predict whether the current turn is complete.
    
    Args:
        chat_context (list): List of chat messages
        model_data (tuple): (tokenizer, session, eou_index)
        
    Returns:
        float: Probability of end of turn
    """
    tokenizer, session, eou_index = model_data
    
    formatted_text = format_chat_context(chat_context, tokenizer)
    
    inputs = tokenizer(
            formatted_text,
            add_special_tokens=False,
            return_tensors="np",
            max_length=MAX_HISTORY_TOKENS,
            truncation=True,
        )

    input_dict = {"input_ids": np.array(inputs["input_ids"], dtype=np.int64)}
    
    # Run inference
    output = session.run(["logits"], input_dict)
    
    # Process output
    logits = output[0]
    last_token_logits = logits[0, -1]
    probs = softmax(last_token_logits)
    
    return float(probs[eou_index])

def main():
    """
    Main function to demonstrate usage.
    """
    model_data = initialize_model()
    
    start_time = time.time()
    
    # Run predictions
    for i, example in enumerate([chat_example1, chat_example2, chat_example3], 1):
        probability = predict_end_of_turn(example, model_data)
        print(f'End of turn probability{i}: {probability}')
    
    print(f"If probability is less than {UNLIKELY_THRESHOLD}, "
          "the model predicts that the user hasn't finished speaking.")
    
    print(f"Prediction time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
