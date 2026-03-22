#!/usr/bin/env python3
"""
Quick Start Script for Sign Language Detection
This script provides a simple menu-driven interface to guide users through the process.
"""

import os
import sys
import subprocess
from train_model import SignLanguageTrainer
from test_model import test_model
from collect_data import DataCollector


def print_header():
    """Print welcome header"""
    print("\n" + "=" * 70)
    print(" " * 15 + "SIGN LANGUAGE DETECTION SYSTEM")
    print(" " * 20 + "Quick Start Guide")
    print("=" * 70)


def print_menu():
    """Print main menu"""
    print("\n📋 Main Menu:")
    print("-" * 70)
    print("  1. Collect Training Data")
    print("  2. Train Model")
    print("  3. Start Real-time Detection")
    print("  4. Test Model Performance")
    print("  5. Check Project Status")
    print("  6. View Instructions")
    print("  7. Exit")
    print("-" * 70)


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        "cv2",
        "mediapipe",
        "numpy",
        "sklearn",
        "matplotlib",
        "pandas",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print("\n⚠️  Warning: Missing required packages!")
        print(f"Missing: {', '.join(missing)}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt\n")
        return False
    return True


def check_status():
    """Check project status"""
    print("\n" + "=" * 70)
    print("PROJECT STATUS")
    print("=" * 70)

    # Check single hand data
    data_exists = os.path.exists("data/sign_language_data.pkl")
    print(f"\n1️⃣  Training Data (Single Hand): {'✓ Found' if data_exists else '✗ Not found'}")
    if data_exists:
        import pickle

        try:
            with open("data/sign_language_data.pkl", "rb") as f:
                data_dict = pickle.load(f)
            num_samples = len(data_dict["data"])
            num_labels = len(set(data_dict["labels"]))
            print(f"   - Total samples: {num_samples}")
            print(f"   - Number of letters: {num_labels}")
            print(f"   - Letters: {sorted(set(data_dict['labels']))}")
        except Exception as e:
            print(f"   - Error reading data: {e}")
    else:
        print("   ⚠️  Run 'Collect Training Data (Single Hand)' first")

    # Check model
    model_exists = os.path.exists("models/sign_language_model.pkl")
    print(f"\n2️⃣  Trained Model: {'✓ Found' if model_exists else '✗ Not found'}")
    if model_exists:
        import pickle

        try:
            with open("models/sign_language_model.pkl", "rb") as f:
                model_data = pickle.load(f)
            labels = sorted(model_data["labels_map"].values())
            print(f"   - Can recognize: {len(labels)} letters")
            print(f"   - Letters: {labels}")
        except Exception as e:
            print(f"   - Error reading model: {e}")
    else:
        print("   ⚠️  Run 'Train Model' after collecting data")

    # Check dependencies
    print(f"\n3️⃣  Dependencies: ", end="")
    if check_dependencies():
        print("✓ All installed")
    else:
        print("✗ Some missing (see above)")

    # Ready status
    print("\n" + "-" * 70)
    if data_exists and model_exists:
        print("✅ System ready! You can start real-time detection.")
    elif data_exists and not model_exists:
        print("⚠️  Data collected. Next: Train the model.")
    else:
        print("⚠️  Getting started: Collect training data first.")
    print("-" * 70)


def show_instructions():
    """Show detailed instructions"""
    print("\n" + "=" * 70)
    print("INSTRUCTIONS")
    print("=" * 70)

    print("\n🎯 GOAL:")
    print("   Create a system that recognizes ASL (American Sign Language)")
    print("   alphabet gestures in real-time using your webcam.")

    print("\n📝 STEPS:")
    print("\n   Step 1: COLLECT TRAINING DATA")
    print("   " + "-" * 40)
    print("   - Make hand gestures for each letter (A-Z)")
    print("   - The system captures 100 samples per letter")
    print("   - Hold your hand steady in the sign position")
    print("   - Good lighting and plain background help")
    print("   - Takes about 15-30 minutes for all letters")

    print("\n   Step 2: TRAIN THE MODEL")
    print("   " + "-" * 40)
    print("   - The system learns from your collected data")
    print("   - Creates a classifier to recognize gestures")
    print("   - Shows accuracy and performance metrics")
    print("   - Saves the trained model for later use")
    print("   - Takes about 1-5 minutes")

    print("\n   Step 3: REAL-TIME DETECTION")
    print("   " + "-" * 40)
    print("   - Opens your webcam")
    print("   - Shows hand landmarks overlay")
    print("   - Displays predicted letter with confidence")
    print("   - Works in real-time as you make gestures")
    print("   - Press 'q' to quit, 'c' to clear predictions")

    print("\n💡 TIPS:")
    print("   - Collect data with consistent hand positioning")
    print("   - Use the same environment for collection and detection")
    print("   - Practice ASL alphabet before collecting data")
    print("   - More training data = better accuracy")
    print("   - Aim for 85-95% accuracy after training")

    print("\n📚 ASL ALPHABET REFERENCE:")
    print("   https://www.ai-media.tv/wp-content/uploads/ASL_Alphabet.jpg")

    print("\n" + "=" * 70)


def trainmodel():
    """Train model using SignLanguageTrainer class"""
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    if not os.path.exists("data/sign_language_data.pkl"):
        print("\n⚠️  No training data found!")
        print("Please collect data first (Option 1)")
        return False
    
    try:
        trainer = SignLanguageTrainer()
        data, labels = trainer.load_data()
        X, y = trainer.prepare_data(data, labels)
        
        from sklearn.model_selection import train_test_split  # pyright: ignore[reportMissingImports]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        trainer.train_model(X_train, y_train, n_estimators=100)
        y_pred, accuracy = trainer.evaluate_model(X_test, y_test)
        
        trainer.save_model()
        print("\n✅ Model training completed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_function():
    """Test model using test_model function"""
    print("\n" + "=" * 70)
    print("TESTING MODEL")
    print("=" * 70)
    
    if not os.path.exists("models/sign_language_model.pkl"):
        print("\n⚠️  No trained model found!")
        print("Please train the model first (Option 3)")
        return False
    
    try:
        test_model()
        print("\n✅ Model testing completed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_script(script_name, description):
    """Run a Python script"""
    print(f"\n{'=' * 70}")
    print(f"Running: {description}")
    print("=" * 70)

    if not os.path.exists(script_name):
        print(f"\n❌ Error: {script_name} not found!")
        return False

    try:
        result = subprocess.run([sys.executable, script_name])
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully!")
            return True
        else:
            print(f"\n⚠️  {description} exited with code {result.returncode}")
            return False
    except KeyboardInterrupt:
        print(f"\n\n⚠️  {description} interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error running {script_name}: {e}")
        return False


def main():
    """Main program loop"""
    print_header()

    # Check dependencies on startup
    print("\n🔍 Checking dependencies...")
    deps_ok = check_dependencies()
    if deps_ok:
        print("✅ All dependencies installed!")
    else:
        response = input("\nContinue anyway? (y/n): ").lower()
        if response != "y":
            print("\nExiting. Please install dependencies first.")
            return

    while True:
        print_menu()

        try:
            choice = input("\n👉 Enter your choice (1-7): ").strip()

            if choice == "1":
                # Collect data
                print("\n📸 Starting Data Collection...\n")
                print("ℹ️  Instructions:")
                print("   - The script will ask which letters to collect")
                print("   - For each letter, position your hand in ASL sign")
                print("   - Press SPACE to start collecting samples")
                print("   - Keep your hand steady during collection")
                print("   - Press 'q' to skip a letter\n")

                input("Press ENTER to continue...")
                run_script("collect_data.py", "Data Collection")

            elif choice == "2":
                # Train model
                if not os.path.exists("data/sign_language_data.pkl"):
                    print("\n⚠️  No training data found!")
                    print("Please collect data first (Option 1)")
                    continue

                print("\n🧠 Starting Model Training...\n")
                print("ℹ️  The training process will:")
                print("   - Load your collected data")
                print("   - Train a Random Forest classifier")
                print("   - Show accuracy and performance metrics")
                print("   - Save the trained model\n")

                input("Press ENTER to continue...")
                trainmodel()

            elif choice == "3":
                # Real-time detection
                if not os.path.exists("models/sign_language_model.pkl"):
                    print("\n⚠️  No trained model found!")
                    print("Please train the model first (Option 2)")
                    continue

                print("\n🎥 Starting Real-time Detection...\n")
                print("ℹ️  Controls:")
                print("   - Press 'q' to quit")
                print("   - Press 'c' to clear prediction history")
                print("\n   Make ASL hand signs in front of your camera!\n")

                input("Press ENTER to continue...")
                run_script("detect_sign.py", "Real-time Detection")

            elif choice == "4":
                # Test model
                if not os.path.exists("models/sign_language_model.pkl"):
                    print("\n⚠️  No trained model found!")
                    print("Please train the model first (Option 2)")
                    continue

                print("\n🧪 Testing Model Performance...\n")
                input("Press ENTER to continue...")
                test_model_function()

            elif choice == "5":
                # Check status
                check_status()

            elif choice == "6":
                # Show instructions
                show_instructions()

            elif choice == "7":
                # Exit
                print("\n👋 Thank you for using Sign Language Detection!")
                print("Happy signing! 🤟\n")
                break

            else:
                print("\n⚠️  Invalid choice! Please enter a number from 1 to 7.")

        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

        input("\nPress ENTER to continue...")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
