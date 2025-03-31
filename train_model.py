import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# VARK mapping for all 20 questions (updated to match the new quiz structure)
# Each question has four options: a=Visual, b=Auditory, c=Read/Write, d=Kinesthetic
vark_mapping = {
    q: {'a': 'v', 'b': 'a', 'c': 'r', 'd': 'k'} for q in range(1, 21)
}

# Function to generate synthetic responses for a given dominant style
def generate_synthetic_responses(dominant_style, num_samples):
    responses = []
    labels = []

    for _ in range(num_samples):
        user_responses = []

        # Generate responses biased towards the dominant style
        for q_num in range(1, 21):  # Now 20 questions
            options = ['a', 'b', 'c', 'd']  # Four options
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

        # Directly assign the dominant_style as the label
        responses.append(user_responses)
        labels.append(dominant_style)

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

# Debug: Check the distribution of labels
print("Label distribution before encoding:", pd.Series(all_labels).value_counts())

# Convert responses to a DataFrame
# Encode responses as numerical values: a=0, b=1, c=2, d=3
response_df = pd.DataFrame(all_responses, columns=[f'q{i}' for i in range(1, 21)])  # Now 20 questions
response_df = response_df.replace({'a': 0, 'b': 1, 'c': 2, 'd': 3})

# Convert labels to numerical values: v=0, a=1, r=2, k=3
label_mapping = {'v': 0, 'a': 1, 'r': 2, 'k': 3}
labels = [label_mapping[label] for label in all_labels]

# Debug: Check the distribution of encoded labels
print("Encoded label distribution:", pd.Series(labels).value_counts())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(response_df, labels, test_size=0.2, random_state=42)

# Debug: Check the distribution of labels in the training set
print("Training label distribution:", pd.Series(y_train).value_counts())

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Debug: Check the model's classes
print("Model classes after training:", model.classes_)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
joblib.dump(model, 'vark_model.joblib')

# Save the label mapping for decoding predictions
joblib.dump(label_mapping, 'label_mapping.joblib')

print("Model and label mapping saved successfully.")
