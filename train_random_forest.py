import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from models.random_forest_baseline import create_random_forest_model


def main():
  # Temporary fake data just to make sure the code runs
    num_examples = 200
    num_pose_features = 34
    num_classes = 4

    # X shape: number of frames, pose features
    X = np.random.randn(num_examples, num_pose_features)

    # y shape: technique error label for each frame
    y = np.random.randint(0, num_classes, size=num_examples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = create_random_forest_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
