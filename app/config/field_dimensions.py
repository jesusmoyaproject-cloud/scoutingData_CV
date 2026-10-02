from dataclasses import dataclass

@dataclass(frozen=True)
class FieldDimensions:
    """Dimensions of a standard FIFA soccer pitch in meters."""
    length: float = 105.0
    width: float = 68.0
    
    # Area grande (Penalty area)
    penalty_box_length: float = 16.5
    penalty_box_width: float = 40.32
    
    # Area chica (Goal area)
    goal_box_length: float = 5.5
    goal_box_width: float = 18.32
    
    # Circulo central
    centre_circle_radius: float = 9.15
    
    # Punto penal
    penalty_spot_distance: float = 11.0
