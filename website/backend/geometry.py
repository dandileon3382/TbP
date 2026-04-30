import numpy as np

def calculate_angle(a, b, c):
    """
    Computes the 2D angle at point b given three points a, b, c.
    Returns the angle in degrees between 0.0 and 180.0.
    Each point should be a dict with 'x', 'y' keys.
    """
    def to_array(pt):
        if isinstance(pt, dict):
            return np.array([pt.get('x', 0), pt.get('y', 0)], dtype=np.float64)
        return np.array(pt[:2], dtype=np.float64)

    a = to_array(a)
    b = to_array(b)
    c = to_array(c)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # Guard against zero-length vectors
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 180.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle_deg = np.degrees(np.arccos(cosine_angle))
    return float(angle_deg)
