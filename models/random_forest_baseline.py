from sklearn.ensemble import RandomForestClassifier


def create_random_forest_model():
    """
    Create the frame-by-frame Random Forest baseline.
    Input:
        One pose frame represented as a feature vector.
    Output:
        A predicted freestyle technique-error label.
    Example:
        If there are 17 joints and each joint has x and y coordinates, then one frame has 34 features.

    This is a non-temporal model,frame-by-frame model, because it does not use a sequence of frames.
    """

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42
    )

    return model
