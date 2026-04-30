import os
import glob
import json

def load_exercise_data(data_dir: str) -> list[dict]:
    """
    Loads all exercise JSON files from the given directory.
    Returns a list of exercise data dicts.
    """
    exercise_files = glob.glob(os.path.join(data_dir, "*.json"))
    data = []
    for path in exercise_files:
        with open(path, "r") as f:
            try:
                exercise = json.load(f)
                data.append(exercise)
                print(f"Loaded: {path}")
            except json.JSONDecodeError as e:
                print(f"Error parsing {path}: {e}")
    return data
