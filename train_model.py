import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


class SignLanguageTrainer:
    def __init__(self, data_dir="data", model_dir="models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model = None
        self.labels_map = None

        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def load_data(self, filename="sign_language_data.pkl"):
        """
        Load the collected data from pickle file
        """
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Data file not found: {filepath}\n"
                "Please run collect_data.py first to collect training data."
            )

        with open(filepath, "rb") as f:
            data_dict = pickle.load(f)

        print(f"Loaded data from {filepath}")
        return data_dict["data"], data_dict["labels"]

    def prepare_data(self, data, labels):
        """
        Prepare data for training
        """
        # Convert to numpy arrays
        X = np.array(data)
        y = np.array(labels)

        print(f"\nData shape: {X.shape}")
        print(f"Number of samples: {len(X)}")
        print(f"Number of features per sample: {X.shape[1]}")
        print(f"Number of unique labels: {len(set(y))}")
        print(f"Labels: {sorted(set(y))}")

        # Create label mapping
        unique_labels = sorted(set(y))
        self.labels_map = {i: label for i, label in enumerate(unique_labels)}
        reverse_labels_map = {label: i for i, label in enumerate(unique_labels)}

        # Convert labels to integers
        y_numeric = np.array([reverse_labels_map[label] for label in y])

        return X, y_numeric

    def train_model(self, X_train, y_train, n_estimators=100, random_state=42):
        """
        Train Random Forest classifier
        """
        print("\nTraining Random Forest Classifier...")
        print(f"Number of trees: {n_estimators}")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
        )

        self.model.fit(X_train, y_train)
        print("Training complete!")

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy * 100:.2f}%")

        # Classification report
        label_names = [self.labels_map[i] for i in sorted(self.labels_map.keys())]
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_names))

        return y_pred, accuracy

    def plot_confusion_matrix(self, y_test, y_pred, save_path=None):
        """
        Plot confusion matrix
        """
        label_names = [self.labels_map[i] for i in sorted(self.labels_map.keys())]

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
        )
        plt.title("Confusion Matrix - Sign Language Recognition")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"\nConfusion matrix saved to {save_path}")

        plt.show()

    def plot_feature_importance(self, top_n=20, save_path=None):
        """
        Plot feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 6))
        plt.title(f"Top {top_n} Feature Importances")
        plt.bar(range(top_n), importances[indices])
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Feature importance plot saved to {save_path}")

        plt.show()

    def save_model(self, filename="sign_language_model.pkl"):
        """
        Save the trained model and label mapping
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        filepath = os.path.join(self.model_dir, filename)

        model_data = {"model": self.model, "labels_map": self.labels_map}

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"\nModel saved to {filepath}")

    def load_model(self, filename="sign_language_model.pkl"):
        """
        Load a trained model
        """
        filepath = os.path.join(self.model_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.labels_map = model_data["labels_map"]

        print(f"Model loaded from {filepath}")

    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        """
        from sklearn.model_selection import cross_val_score

        print("\nPerforming cross-validation...")
        scores = cross_val_score(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            X,
            y,
            cv=cv,
            scoring="accuracy",
        )

        print(f"Cross-validation scores: {scores}")
        print(
            f"Mean accuracy: {scores.mean() * 100:.2f}% (+/- {scores.std() * 2 * 100:.2f}%)"
        )


def main():
    print("=" * 50)
    print("SIGN LANGUAGE MODEL TRAINER")
    print("=" * 50)

    # Initialize trainer
    trainer = SignLanguageTrainer()

    # Load data
    try:
        data, labels = trainer.load_data()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    # Prepare data
    X, y = trainer.prepare_data(data, labels)

    # Split data
    test_size = 0.2
    print(
        f"\nSplitting data: {int((1 - test_size) * 100)}% train, {int(test_size * 100)}% test"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Train model
    trainer.train_model(X_train, y_train, n_estimators=100)

    # Evaluate model
    y_pred, accuracy = trainer.evaluate_model(X_test, y_test)

    # Optional: Cross-validation
    perform_cv = input("\nPerform cross-validation? (y/n): ").lower()
    if perform_cv == "y":
        trainer.cross_validate(X, y, cv=5)

    # Plot confusion matrix
    plot_cm = input("\nPlot confusion matrix? (y/n): ").lower()
    if plot_cm == "y":
        save_path = os.path.join(trainer.model_dir, "confusion_matrix.png")
        trainer.plot_confusion_matrix(y_test, y_pred, save_path)

    # Plot feature importance
    plot_fi = input("\nPlot feature importance? (y/n): ").lower()
    if plot_fi == "y":
        save_path = os.path.join(trainer.model_dir, "feature_importance.png")
        trainer.plot_feature_importance(top_n=20, save_path=save_path)

    # Save model
    save_model = input("\nSave trained model? (y/n): ").lower()
    if save_model == "y":
        trainer.save_model()
        print("\n✓ Model training complete and saved!")
    else:
        print("\nModel not saved.")


if __name__ == "__main__":
    main()

