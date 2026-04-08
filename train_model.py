from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json

# Load base model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load training data
with open("data/training/train.json") as f:
    data = json.load(f)

# Convert to training format
train_examples = [
    InputExample(texts=[d["resume"], d["job"]], label=float(d["label"]))
    for d in data
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Loss function
train_loss = losses.CosineSimilarityLoss(model)

# Train model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3
)

# Save model
model.save("models/resume-matcher")

print("✅ Model trained successfully!")