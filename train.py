
import os
import re
import random
import joblib
import numpy as np
import pandas as pd
import torch

from kan import KAN

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, f1_score


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR, "cybercrime_india_synthetic_1000.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\n========== LOADING DATASET ==========")

df = pd.read_csv(DATASET_PATH)

required_columns = [
    "case_description",
    "cybercrime_type",
    "candidate_law_sections"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column missing: {column}"
        )

df = df.dropna(
    subset=required_columns
).copy()

df["case_description"] = (
    df["case_description"].astype(str)
)

df["cybercrime_type"] = (
    df["cybercrime_type"].astype(str).str.strip()
)

print("Dataset shape:", df.shape)
print("Device:", DEVICE)


# ==========================================
# 3. EXTRACT LEGAL SECTION LABELS
# ==========================================

def extract_legal_sections(text):

    text = str(text)

    statute_patterns = [
        (
            r"(IT\s*Act(?:,\s*2000)?|Information Technology Act(?:,\s*2000)?)"
            r"\s+sections?\s+([^.;]+)",
            "IT Act"
        ),
        (
            r"(IPC|Indian Penal Code)"
            r"\s+sections?\s+([^.;]+)",
            "IPC"
        ),
        (
            r"(BNS|Bharatiya Nyaya Sanhita)"
            r"\s+sections?\s+([^.;]+)",
            "BNS"
        ),
        (
            r"(BNSS|Bharatiya Nagarik Suraksha Sanhita)"
            r"\s+sections?\s+([^.;]+)",
            "BNSS"
        ),
        (
            r"(POCSO Act)"
            r"\s+sections?\s+([^.;]+)",
            "POCSO Act"
        )
    ]

    labels = set()

    for pattern, statute in statute_patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for match in matches:

            section_text = match[1]

            section_numbers = re.findall(
                r"\b\d+[A-Za-z]?(?:\([0-9A-Za-z]+\))?\b",
                section_text
            )

            for number in section_numbers:

                labels.add(
                    f"{statute} Section {number}"
                )

    return sorted(labels)


df["legal_labels"] = df[
    "candidate_law_sections"
].apply(extract_legal_sections)

df = df[
    df["legal_labels"].apply(len) > 0
].copy()

print("\nLegal labels extracted successfully.")

print(
    "Unique legal sections:",
    len(set(
        label
        for labels in df["legal_labels"]
        for label in labels
    ))
)

print(
    "Records with legal labels:",
    len(df)
)


# ==========================================
# 4. CHECK TARGET LABELS
# ==========================================

if df["cybercrime_type"].nunique() < 2:
    raise ValueError(
        "At least two cybercrime categories are required."
    )

legal_target_sets = {
    tuple(labels)
    for labels in df["legal_labels"]
}

if len(legal_target_sets) < 2:
    raise ValueError(
        "The dataset contains fewer than two distinct "
        "legal-section combinations. A useful legal "
        "classification model cannot be trained from "
        "identical targets."
    )

print("\nCrime categories:")
print(df["cybercrime_type"].value_counts())

print("\nExtracted legal-section frequencies:")

all_legal_labels = [
    label
    for labels in df["legal_labels"]
    for label in labels
]

print(
    pd.Series(all_legal_labels).value_counts()
)


# ==========================================
# 5. ENCODE TARGET LABELS
# ==========================================

crime_encoder = LabelEncoder()

y_crime = crime_encoder.fit_transform(
    df["cybercrime_type"]
)

legal_encoder = MultiLabelBinarizer()

y_legal = legal_encoder.fit_transform(
    df["legal_labels"]
)

print("\nCrime classes:", len(crime_encoder.classes_))
print("Legal classes:", len(legal_encoder.classes_))


# ==========================================
# 6. TRAIN-TEST SPLIT
# ==========================================

indices = np.arange(len(df))

class_counts = np.bincount(y_crime)

test_count = int(np.ceil(len(df) * 0.20))
train_count = len(df) - test_count

can_stratify = (
    class_counts.min() >= 2
    and test_count >= len(class_counts)
    and train_count >= len(class_counts)
)

stratify_labels = y_crime if can_stratify else None

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=stratify_labels
)

print("\nTraining records:", len(train_idx))
print("Testing records:", len(test_idx))


# ==========================================
# 7. TEXT PREPROCESSING
# ==========================================

print("\n========== TEXT PREPROCESSING ==========")

texts = df["case_description"].tolist()

X_train_text = [texts[i] for i in train_idx]
X_test_text = [texts[i] for i in test_idx]

tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(
    X_train_text
)

X_test_tfidf = tfidf.transform(
    X_test_text
)

n_components = min(
    64,
    X_train_tfidf.shape[0] - 1,
    X_train_tfidf.shape[1] - 1
)

if n_components < 2:
    raise ValueError(
        "Not enough text features for SVD."
    )

svd = TruncatedSVD(
    n_components=n_components,
    random_state=SEED
)

X_train = svd.fit_transform(
    X_train_tfidf
)

X_test = svd.transform(
    X_test_tfidf
)

input_features = X_train.shape[1]

print("TF-IDF features:", X_train_tfidf.shape[1])
print("SVD input features:", input_features)


# ==========================================
# 8. CONVERT TO PYTORCH TENSORS
# ==========================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32,
    device=DEVICE
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32,
    device=DEVICE
)

y_crime_train = torch.tensor(
    y_crime[train_idx],
    dtype=torch.long,
    device=DEVICE
)

y_crime_test = torch.tensor(
    y_crime[test_idx],
    dtype=torch.long,
    device=DEVICE
)

y_legal_train = torch.tensor(
    y_legal[train_idx],
    dtype=torch.float32,
    device=DEVICE
)

y_legal_test = torch.tensor(
    y_legal[test_idx],
    dtype=torch.float32,
    device=DEVICE
)


# ==========================================
# 9. TRAINING FUNCTION
# ==========================================

def train_kan(model, X, y, loss_function, name):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    number_of_samples = X.shape[0]

    print(f"\nTraining {name}...")

    for epoch in range(EPOCHS):

        model.train()

        permutation = torch.randperm(
            number_of_samples,
            device=DEVICE
        )

        total_loss = 0.0

        for start in range(
            0,
            number_of_samples,
            BATCH_SIZE
        ):

            batch_indices = permutation[
                start:start + BATCH_SIZE
            ]

            batch_X = X[batch_indices]
            batch_y = y[batch_indices]

            optimizer.zero_grad()

            predictions = model(batch_X)

            loss = loss_function(
                predictions,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:

            average_loss = (
                total_loss /
                max(1, int(np.ceil(
                    number_of_samples / BATCH_SIZE
                )))
            )

            print(
                f"{name} | Epoch "
                f"{epoch + 1}/{EPOCHS} | "
                f"Loss: {average_loss:.4f}"
            )

    model.eval()

    return model


# ==========================================
# 10. CYBERCRIME KAN CLASSIFICATION
# ==========================================

print("\n========== CYBERCRIME KAN ==========")

crime_classes = len(crime_encoder.classes_)

crime_model = KAN(
    width=[
        input_features,
        16,
        crime_classes
    ],
    grid=5,
    k=3,
    seed=SEED,
    device=str(DEVICE)
)

crime_loss = torch.nn.CrossEntropyLoss()

crime_model = train_kan(
    crime_model,
    X_train_tensor,
    y_crime_train,
    crime_loss,
    "Cybercrime KAN"
)

with torch.no_grad():

    crime_predictions = crime_model(
        X_test_tensor
    ).argmax(dim=1)

crime_accuracy = accuracy_score(
    y_crime_test.cpu().numpy(),
    crime_predictions.cpu().numpy()
)

print("\nCybercrime Accuracy:", crime_accuracy)

print(
    classification_report(
        y_crime_test.cpu().numpy(),
        crime_predictions.cpu().numpy(),
        labels=np.arange(crime_classes),
        target_names=crime_encoder.classes_,
        zero_division=0
    )
)


# ==========================================
# 11. LEGAL SECTION MULTI-LABEL KAN
# ==========================================

print("\n========== LEGAL SECTION KAN ==========")

legal_classes = len(legal_encoder.classes_)

legal_model = KAN(
    width=[
        input_features,
        16,
        legal_classes
    ],
    grid=5,
    k=3,
    seed=SEED,
    device=str(DEVICE)
)

positive_counts = y_legal_train.sum(dim=0)

negative_counts = (
    len(train_idx) - positive_counts
)

pos_weight = (
    negative_counts /
    positive_counts.clamp(min=1)
).clamp(max=20)

legal_loss = torch.nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

legal_model = train_kan(
    legal_model,
    X_train_tensor,
    y_legal_train,
    legal_loss,
    "Legal Section KAN"
)

with torch.no_grad():

    legal_logits = legal_model(
        X_test_tensor
    )

    legal_probabilities = torch.sigmoid(
        legal_logits
    )

    legal_predictions = (
        legal_probabilities >= 0.5
    ).cpu().numpy()

legal_f1 = f1_score(
    y_legal_test.cpu().numpy(),
    legal_predictions,
    average="micro",
    zero_division=0
)

print("\nLegal Section Micro F1:", legal_f1)

print(
    classification_report(
        y_legal_test.cpu().numpy(),
        legal_predictions,
        target_names=legal_encoder.classes_,
        zero_division=0
    )
)


# ==========================================
# 12. SAVE TRAINED MODELS
# ==========================================

print("\n========== SAVING MODELS ==========")

torch.save(
    crime_model.state_dict(),
    os.path.join(MODEL_DIR, "crime_kan.pth")
)

torch.save(
    legal_model.state_dict(),
    os.path.join(MODEL_DIR, "legal_kan.pth")
)

artifacts = {
    "tfidf": tfidf,
    "svd": svd,
    "crime_encoder": crime_encoder,
    "legal_encoder": legal_encoder,
    "input_features": input_features,
    "crime_classes": crime_classes,
    "legal_classes": legal_classes
}

joblib.dump(
    artifacts,
    os.path.join(MODEL_DIR, "preprocessing.pkl")
)


# ==========================================
# 13. SAVE RESULTS
# ==========================================

results = {
    "dataset_records": int(len(df)),
    "training_records": int(len(train_idx)),
    "testing_records": int(len(test_idx)),
    "crime_categories": int(crime_classes),
    "legal_sections": int(legal_classes),
    "crime_accuracy": float(crime_accuracy),
    "legal_micro_f1": float(legal_f1),
    "note": (
        "Synthetic provisional labels. "
        "Not verified legal outcomes or legal advice."
    )
}

joblib.dump(
    results,
    os.path.join(OUTPUT_DIR, "results.pkl")
)

print("\n========== TRAINING COMPLETED ==========")

print("Saved files:")

print("models/crime_kan.pth")
print("models/legal_kan.pth")
print("models/preprocessing.pkl")
print("outputs/results.pkl")

print("\nYou can now run: python index.py")
