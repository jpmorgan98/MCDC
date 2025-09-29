import numpy as np

from mcdc.constant import (
    PARTICLE_NEUTRON,
)


class InputCard:
    def __init__(self, tag):
        self.tag = tag

    def __str__(self):
        text = "%s card\n" % self.tag

        for name in [
            a
            for a in dir(self)
            if not a.startswith("__")
            and not callable(getattr(self, a))
            and a != "tag"
            and not a.startswith("_")
        ]:
            text += "  %s : %s\n" % (name, str(getattr(self, name)))
        return text


class SourceCard(InputCard):
    def __init__(self):
        InputCard.__init__(self, "Source")

        # Set card data
        self.ID = None
        self.box = False
        self.isotropic = True
        self.white = False
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.box_x = np.array([0.0, 0.0])
        self.box_y = np.array([0.0, 0.0])
        self.box_z = np.array([0.0, 0.0])
        self.ux = 0.0
        self.uy = 0.0
        self.uz = 0.0
        self.white_x = 0.0
        self.white_y = 0.0
        self.white_z = 0.0
        self.group = np.array([1.0])
        self.energy = np.array([[1e6 - 1.0, 1e6 + 1.0], [1.0, 1.0]])
        self.time = np.array([0.0, 0.0])
        self.particle_type = PARTICLE_NEUTRON
        self.prob = 1.0
