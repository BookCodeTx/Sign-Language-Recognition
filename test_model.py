import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os


def test_model():
    """
    Test the trained model on the test dataset
    """
    print("=" * 60)
    print("SIGN LANGUAGE MODEL TESTING")
    print("=" * 60)

    # Check if model exists
    model_path = "models/sign_language_model.pkl"
    if not os.path.exists(model_path):
        print(f"\nError: Model file not found at {model_path}")
        print("Please run train_model.py first to train a model.")
        return

    # Check if data exists
    data_path = "data/sign_language_data.pkl"
    if not os.path.exists(data_path):
        print(f"\nError: Data file not found at {data_path}")
        print("Please run collect_data.py first to collect training data.")
        return

    # Load model
    print(f"\nLoading model from {model_path}...")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    labels_map = model_data["labels_map"]

    print(f"✓ Model loaded successfully")
    print(f"Available labels: {sorted(labels_map.values())}")

    # Load data
    print(f"\nLoading data from {data_path}...")
    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)

    X = np.array(data_dict["data"])
    y_labels = np.array(data_dict["labels"])

    print(f"✓ Data loaded successfully")
    print(f"Total samples: {len(X)}")

    # Convert labels to numeric
    reverse_labels_map = {label: i for i, label in labels_map.items()}
    y = np.array([reverse_labels_map[label] for label in y_labels])

    # Make predictions
    print("\nMaking predictions on entire dataset...")
    y_pred = model.predict(X)

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

    # Show per-class accuracy
    print("\nPer-Letter Accuracy:")
    print("-" * 60)

    unique_labels = sorted(set(y))
    for label_idx in unique_labels:
        label_name = labels_map[label_idx]
        mask = y == label_idx
        if mask.sum() > 0:
            label_accuracy = accuracy_score(y[mask], y_pred[mask])
            correct = (y[mask] == y_pred[mask]).sum()
            total = mask.sum()
            print(
                f"  {label_name}: {label_accuracy * 100:5.1f}% ({correct}/{total} correct)"
            )

    # Classification report
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 60)
    label_names = [labels_map[i] for i in sorted(labels_map.keys())]
    print(classification_report(y, y_pred, target_names=label_names))

    # Find most confused pairs
    print("\n" + "=" * 60)
    print("MOST COMMON MISCLASSIFICATIONS")
    print("=" * 60)

    confusion_pairs = {}
    for true_label, pred_label in zip(y, y_pred):
        if true_label != pred_label:
            pair = (labels_map[true_label], labels_map[pred_label])
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    if confusion_pairs:
        sorted_confusions = sorted(
            confusion_pairs.items(), key=lambda x: x[1], reverse=True
        )
        print("\nTop 10 confused letter pairs:")
        for i, ((true_letter, pred_letter), count) in enumerate(
            sorted_confusions[:10], 1
        ):
            print(f"  {i}. {true_letter} → {pred_letter}: {count} times")
    else:
        print("\n✓ No misclassifications found!")

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)

    # Summary
    if accuracy >= 0.95:
        print("\n✓ Excellent! Model performance is outstanding.")
    elif accuracy >= 0.85:
        print("\n✓ Good! Model performance is solid.")
    elif accuracy >= 0.75:
        print("\n⚠ Fair. Consider collecting more training data.")
    else:
        print("\n⚠ Poor performance. Recommend:")
        print("  - Collect more training data (200+ samples per letter)")
        print("  - Ensure consistent hand positioning")
        print("  - Improve lighting conditions during data collection")


def test_single_prediction():
    """
    Test model on a random sample
    """
    print("\n" + "=" * 60)
    print("TESTING SINGLE PREDICTION")
    print("=" * 60)

    # Load model
    model_path = "models/sign_language_model.pkl"
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    labels_map = model_data["labels_map"]

    # Load data
    data_path = "data/sign_language_data.pkl"
    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)

    X = np.array(data_dict["data"])
    y_labels = np.array(data_dict["labels"])

    # Pick a random sample
    idx = np.random.randint(0, len(X))
    sample = X[idx]
    true_label = y_labels[idx]

    # Make prediction
    pred_idx = model.predict(sample.reshape(1, -1))[0]
    pred_label = labels_map[pred_idx]
    probabilities = model.predict_proba(sample.reshape(1, -1))[0]
    confidence = probabilities[pred_idx]

    print(f"\nRandom sample #{idx}")
    print(f"True label: {true_label}")
    print(f"Predicted label: {pred_label}")
    print(f"Confidence: {confidence * 100:.2f}%")

    if true_label == pred_label:
        print("✓ Correct prediction!")
    else:
        print("✗ Incorrect prediction")

    # Show top 3 predictions
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    print("\nTop 3 predictions:")
    for i, idx in enumerate(top_3_idx, 1):
        print(f"  {i}. {labels_map[idx]}: {probabilities[idx] * 100:.2f}%")


if __name__ == "__main__":
    try:
        test_model()

        # Ask if user wants to test single prediction
        print("\n")
        response = input("Test a single random prediction? (y/n): ").lower()
        if response == "y":
            test_single_prediction()

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have:")
        print("1. Collected data using collect_data.py")
        print("2. Trained model using train_model.py")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()

