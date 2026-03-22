import cv2
import mediapipe as mp  # pyright: ignore[reportMissingImports]
import numpy as np
import os
import pickle
import time

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class DataCollector:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def normalize_landmarks(self, landmarks):
        """
        Normalize hand landmarks relative to the wrist position
        and scale to make them translation and scale invariant
        """
        # Convert landmarks to numpy array
        coords = []
        for landmark in landmarks:
            coords.extend([landmark.x, landmark.y, landmark.z])

        coords = np.array(coords).reshape(-1, 3)

        # Get wrist position (first landmark)
        wrist = coords[0]

        # Translate to origin (wrist at 0,0,0)
        coords = coords - wrist

        # Calculate max distance for normalization
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist > 0:
            coords = coords / max_dist

        # Flatten back to 1D array
        return coords.flatten()

    def collect_samples(self, label, num_samples=100):
        """
        Collect samples for a specific alphabet label (single hand)
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        samples = []
        count = 0
        collecting = False

        print(f"\n=== Collecting data for letter: {label} ===")
        print(f"Press SPACE to start collecting {num_samples} samples")
        print("Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame
            results = self.hands.process(rgb_frame)

            # Draw hand landmarks
            num_hands_detected = 0
            if results.multi_hand_landmarks:
                # Get hand classifications
                hand_classifications = []
                if results.multi_handedness:
                    for classification in results.multi_handedness:
                        hand_classifications.append(classification.classification[0].label)

                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    num_hands_detected += 1
                    
                    # Determine hand type for display
                    hand_type = "Unknown"
                    if idx < len(hand_classifications):
                        hand_type = hand_classifications[idx]

                    # Draw landmarks with color coding
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    # Collect data from first hand only if in collecting mode
                    if collecting and count < num_samples and idx == 0:
                        normalized_landmarks = self.normalize_landmarks(
                            hand_landmarks.landmark
                        )
                        samples.append(normalized_landmarks)
                        count += 1

                        # Small delay to avoid collecting too similar frames
                        time.sleep(0.05)

            # Display info on frame
            info_text = f"Label: {label} | Samples: {count}/{num_samples}"
            cv2.putText(
                frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # Show number of hands detected
            if num_hands_detected > 0:
                hands_text = f"Hands detected: {num_hands_detected} (using first hand)"
                cv2.putText(
                    frame, hands_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                )

            if not collecting:
                cv2.putText(
                    frame,
                    "Press SPACE to start",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "COLLECTING...",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            if not results.multi_hand_landmarks:
                cv2.putText(
                    frame,
                    "No hand detected!",
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Data Collection", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nQuitting data collection...")
                break
            elif key == ord(" ") and not collecting:
                collecting = True
                print("Started collecting...")

            # Stop when we have enough samples
            if count >= num_samples:
                print(f"Collected {count} samples for '{label}'")
                break

        cap.release()
        cv2.destroyAllWindows()

        return samples

    def save_data(self, data, labels, filename="sign_language_data.pkl"):
        """
        Save collected data and labels to a pickle file
        """
        filepath = os.path.join(self.data_dir, filename)

        with open(filepath, "wb") as f:
            pickle.dump({"data": data, "labels": labels}, f)

        print(f"\nData saved to {filepath}")
        print(f"Total samples: {len(data)}")

    def load_existing_data(self, filename="sign_language_data.pkl"):
        """
        Load existing data if available
        """
        filepath = os.path.join(self.data_dir, filename)

        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                data_dict = pickle.load(f)
            return data_dict["data"], data_dict["labels"]
        return [], []

    def collect_alphabet_data(self, alphabets, samples_per_letter=100):
        """
        Collect data for multiple alphabets (single hand)
        """
        # Load existing data if any
        all_data, all_labels = self.load_existing_data()

        print("=" * 50)
        print("SIGN LANGUAGE DATA COLLECTION")
        print("=" * 50)
        print(f"\nCollecting data for alphabets: {', '.join(alphabets)}")
        print(f"Samples per letter: {samples_per_letter}")

        for alphabet in alphabets:
            response = input(
                f"\nReady to collect data for '{alphabet}'? (y/n/skip): "
            ).lower()

            if response == "skip" or response == "s":
                print(f"Skipping '{alphabet}'")
                continue
            elif response != "y":
                print("Stopping data collection")
                break

            samples = self.collect_samples(alphabet, samples_per_letter)

            if samples:
                # Add to dataset
                all_data.extend(samples)
                all_labels.extend([alphabet] * len(samples))

                # Save after each letter
                self.save_data(all_data, all_labels)

                print(f"Successfully collected {len(samples)} samples for '{alphabet}'")
            else:
                print(f"No samples collected for '{alphabet}'")

        print("\n" + "=" * 50)
        print("DATA COLLECTION COMPLETE")
        print("=" * 50)
        print(f"Total samples collected: {len(all_data)}")
        print(f"Total labels: {len(set(all_labels))}")

        return all_data, all_labels


def main():
    # Initialize data collector
    collector = DataCollector()

    # Define alphabets to collect (A-Z)
    # You can modify this list to collect only specific letters
    alphabets = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

    # Or collect specific letters for testing:
    # alphabets = ['A', 'B', 'C', 'D', 'E']

    print("\nWELCOME TO SIGN LANGUAGE DATA COLLECTOR")
    print("-" * 50)
    print("This tool will help you collect hand gesture data")
    print("for sign language alphabet recognition.")
    print("\nInstructions:")
    print("1. Position your hand in the sign language gesture")
    print("2. Press SPACE to start collecting samples")
    print("3. Hold the gesture steady while samples are collected")
    print("4. Press 'q' to quit current collection")
    print("-" * 50)

    # Ask user which letters to collect
    choice = input("\nCollect all alphabets A-Z? (y/n): ").lower()

    if choice != "y":
        custom_input = input("Enter letters to collect (e.g., A B C D E): ").upper()
        alphabets = custom_input.split()

    # Ask for samples per letter
    try:
        samples_input = input("Samples per letter (default 100): ")
        samples_per_letter = int(samples_input) if samples_input else 100
    except ValueError:
        samples_per_letter = 100

    # Start collection
    collector.collect_alphabet_data(alphabets, samples_per_letter)


if __name__ == "__main__":
    main()
