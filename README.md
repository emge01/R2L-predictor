# Reader to Leader Prediction Model (R2L)
A fine-tuned BERT transformer model for predicting behavioural roles in text, classifying messages into Contributor, Collaborator, or Leader categories.

# Overview
This model analyses linguistic and semantic cues in written text (messages, posts, chats) to identify behavioural roles in digital communication. It was developed with application to cybercriminal activity profiling, specifically for analysing patterns of interaction in Telegram-style group communication.

# Files
File Description:
- train.py: Fine-tunes BERT on labelled behavioural role data
- predict.py: Loads the trained model and runs predictions on new text

