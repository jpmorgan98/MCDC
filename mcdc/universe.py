from mcdc.objects import ObjectNonSingleton


# ======================================================================================
# Universe
# ======================================================================================

class Universe(ObjectNonSingleton):
    def __init__(self, name='', region=None, fill=None, translation=np.zeros(3), rotation=np.zeros(3)):
        label = "cell"
        super().__init__(label)

        self.name = name


# ======================================================================================
# Lattice
# ======================================================================================


class Universe(ObjectNonSingleton):
    def __init__(self, name='', region=None, fill=None, translation=np.zeros(3), rotation=np.zeros(3)):
        label = "cell"
        super().__init__(label)

        self.name = name
