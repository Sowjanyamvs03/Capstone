import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# VARK mapping for all 30 questions (same as in app.py)
vark_mapping = {
    1: {'a': 'r', 'b': 'a', 'c': 'k'},
    2: {'a': 'v', 'b': 'a', 'c': 'k'},
    3: {'a': 'r', 'b': 'a', 'c': 'k'},
    4: {'a': 'r', 'b': 'a', 'c': 'k'},
    5: {'a': 'v', 'b': 'a', 'c': 'k'},
    6: {'a': 'v', 'b': 'a', 'c': 'k'},
    7: {'a': 'v', 'b': 'a', 'c': 'k'},
    8: {'a': 'r', 'b': 'a', 'c': 'v'},
    9: {'a': 'r', 'b': 'a', 'c': 'k'},
    10: {'a': 'v', 'b': 'a', 'c': 'k'},
    11: {'a': 'v', 'b': 'a', 'c': 'k'},
    12: {'a': 'v', 'b': 'a', 'c': 'k'},
    13: {'a': 'v', 'b': 'a', 'c': 'k'},
    14: {'a': 'v', 'b': 'a', 'c': 'k'},
    15: {'a': 'v', 'b': 'a', 'c': 'k'},
    16: {'a': 'v', 'b': 'a', 'c': 'k'},
    17: {'a': 'v', 'b': 'a', 'c': 'k'},
    18: {'a': 'r', 'b': 'a', 'c': 'k'},
    19: {'a': 'v', 'b': 'a', 'c': 'k'},
    20: {'a': 'v', 'b': 'a', 'c': 'k'},
    21: {'a': 'v', 'b': 'a', 'c': 'k'},
    22: {'a': 'v', 'b': 'a', 'c': 'k'},
    23: {'a': 'v', 'b': 'a', 'c': 'k'},
    24: {'a': 'v', 'b': 'a', 'c': 'k'},
    25: {'a': 'v', 'b': 'a', 'c': 'k'},
    26: {'a': 'v', 'b': 'a', 'c': 'k'},
    27: {'a': 'v', 'b': 'a', 'c': 'k'},
    28: {'a': 'r', 'b': 'a', 'c': 'k'},
    29: {'a': 'r', 'b': 'a', 'c': 'k'},
    30: {'a': 'v', 'b': 'a', 'c': 'k'}
}


# Function to generate synthetic responses for a given dominant style
def generate_synthetic_responses(dominant_style, num_samples):
    responses = []
    labels = []

    for _ in range(num_samples):
        user_responses = []
        v, a, r, k = 0, 0, 0, 0

        # Generate responses biased towards the dominant style
        for q_num in range(1, 31):
            options = ['a', 'b', 'c']
            style_weights = {'v': 0, 'a': 0, 'r': 0, 'k': 0}

            # Increase the probability of choosing options that match the dominant style
            for option in options:
                style = vark_mapping[q_num][option]
                style_weights[style] += 1
                if style == dominant_style.lower()[0]:
                    style_weights[style] += 2  # Bias towards the dominant style

            # Normalize weights and choose an option
            total_weight = sum(style_weights.values())
            probabilities = [style_weights[vark_mapping[q_num][opt]] / total_weight for opt in options]
            chosen_option = np.random.choice(options, p=probabilities)
            user_responses.append(chosen_option)

            # Update scores for label verification
            style = vark_mapping[q_num][chosen_option]
            if style == 'v':
                v += 1
            elif style == 'a':
                a += 1
            elif style == 'r':
                r += 1
            elif style == 'k':
                k += 1

        # Determine the actual dominant style based on scores
        scores = {'v': v, 'a': a, 'r': r, 'k': k}
        max_score = max(scores.values())
        actual_dominant = [style for style, score in scores.items() if score == max_score][0]

        responses.append(user_responses)
        labels.append(actual_dominant)

    return responses, labels


# Generate synthetic dataset
num_samples_per_style = 250  # 250 samples per learning style
styles = ['v', 'a', 'r', 'k']
all_responses = []
all_labels = []

for style in styles:
    responses, labels = generate_synthetic_responses(style, num_samples_per_style)
    all_responses.extend(responses)
    all_labels.extend(labels)

# Convert responses to a DataFrame
# Encode responses as numerical values: a=0, b=1, c=2
response_df = pd.DataFrame(all_responses, columns=[f'q{i}' for i in range(1, 31)])
response_df = response_df.replace({'a': 0, 'b': 1, 'c': 2})

# Convert labels to numerical values: v=0, a=1, r=2, k=3
label_mapping = {'v': 0, 'a': 1, 'r': 2, 'k': 3}
labels = [label_mapping[label] for label in all_labels]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(response_df, labels, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
joblib.dump(model, 'vark_model.joblib')

# Save the label mapping for decoding predictions
joblib.dump(label_mapping, 'label_mapping.joblib')