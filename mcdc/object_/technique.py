from mcdc.object_.base import ObjectSingleton

# ======================================================================================
# Implicit capture
# ======================================================================================


class ImplicitCapture(ObjectSingleton):
    # Annotations for Numba mode
    label: str = 'implicit_capture'
    active: bool

    def __init__(self):
        self.active = False
    
    def __call__(self, active: bool = True):
        self.active = active


# ======================================================================================
# Weight roulette
# ======================================================================================


class WeightRoulette(ObjectSingleton):
    # Annotations for Numba mode
    label: str = 'weight_roulette'

    weight_threshold: float
    weight_target: float

    def __init__(self):
        self.weight_threshold = 0.0
        self.weight_target = 1.0

    def __call__(self, weight_threshold: float = 0.0, weight_target: float = 1.0):
        self.weight_threshold = weight_threshold
        self.weight_target = weight_target
