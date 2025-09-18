import numpy as np

####

from mcdc.constant import (
    BC_NONE,
    BC_REFLECTIVE,
    BC_VACUUM,
    INF,
    SURFACE_CYLINDER_X,
    SURFACE_CYLINDER_Y,
    SURFACE_CYLINDER_Z,
    SURFACE_PLANE,
    SURFACE_PLANE_X,
    SURFACE_PLANE_Y,
    SURFACE_PLANE_Z,
    SURFACE_QUADRIC,
    SURFACE_SPHERE,
)
from mcdc.objects import ObjectNonSingleton


class Surface(ObjectNonSingleton):
    def __init__(self, type_, boundary_condition, name):
        label = "surface"
        super().__init__(label)

        # Type and name
        self.type = type_
        self.name = name

        # Boundary condition
        if boundary_condition == 'none':
            self.boundary_condition = BC_NONE
        elif boundary_condition == 'vacuum':
            self.boundary_condition = BC_VACUUM
        elif boundary_condition == 'reflective':
            self.boundary_condition = BC_REFLECTIVE

        # Quadric surface coefficients
        self.A = 0
        self.B = 0
        self.C = 0
        self.D = 0
        self.E = 0
        self.F = 0
        self.G = 0
        self.H = 0
        self.I = 0
        self.J = 0

        # Helpers
        self.linear = True
        # Surface normal direction (if linear)
        self.nx = 0.0
        self.ny = 0.0
        self.nz = 0.0

        # Moving surface parameters
        self.moving = False
        self.move_time_grid = np.array([0.0, INF])
        self.move_translations = np.array([[0.0, 0.0, 0.0]])
        self.move_velocities = np.array([0.0])

        # TODO: Surface tally
    

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        if self.name != '':
            text += f"  - Name: {self.name}\n"
        text += f"  - Boundary condition: {decode_BC_type(self.boundary_condition)}\n"

        # ==============================================================================
        # Type-based repr
        # ==============================================================================

        if self.type == SURFACE_PLANE_X:
            text += f"  - x0: {-self.J} cm\n"
        elif self.type == SURFACE_PLANE_Y:
            text += f"  - y0: {-self.J} cm\n"
        elif self.type == SURFACE_PLANE_Z:
            text += f"  - z0: {-self.J} cm\n"
        elif self.type == SURFACE_PLANE:
            text += f"  - Coeffs.: {self.G}, {self.H}, {self.I}, {self.J}\n"
            text += f"  - Normal: ({self.nx}, {self.ny}, {self.nz})\n"
        elif self.type == SURFACE_CYLINDER_X:
            y = -0.5 * self.H
            z = -0.5 * self.I
            r = (y**2 + z**2 - self.J)**0.5
            text += f"  - Center (y, z): ({y}, {z}) cm\n"
            text += f"  - Radius: {r} cm\n"
        elif self.type == SURFACE_CYLINDER_Y:
            x = -0.5 * self.G
            z = -0.5 * self.I
            r = (x**2 + z**2 - self.J)**0.5
            text += f"  - Center (x, z): ({x}, {z}) cm\n"
            text += f"  - Radius: {r} cm\n"
        elif self.type == SURFACE_CYLINDER_Z:
            x = -0.5 * self.G
            y = -0.5 * self.H
            r = (x**2 + y**2 - self.J)**0.5
            text += f"  - Center (x, y): ({x}, {y}) cm\n"
            text += f"  - Radius: {r} cm\n"
        elif self.type == SURFACE_SPHERE:
            x = -0.5 * self.G
            y = -0.5 * self.H
            z = -0.5 * self.I
            r = (x**2 + y**2 + z**2 - self.J)**0.5
            text += f"  - Center (x, y, z): ({x}, {y}, {z}) cm\n"
            text += f"  - Radius: {r} cm\n"
        elif self.type == SURFACE_QUADRIC:
            text += f"  - Coeffs.: {self.A}, {self.B}, {self.C},\n"
            text += f"             {self.D}, {self.E}, {self.F},\n"
            text += f"             {self.G}, {self.H}, {self.I}, {self.J}\n"

        return text


    
    # ==================================================================================
    # Type-based creation methods
    # ==================================================================================

    @classmethod
    def PlaneX(cls, x=0.0, boundary_condition='none', name=''):
        type_ = SURFACE_PLANE_X
        surface = cls(type_, boundary_condition, name)
        
        surface.linear = True
        surface.G = 1.0
        surface.J = -x
        surface.nx = 1.0

        return surface


    @classmethod
    def PlaneY(cls, y=0.0, boundary_condition='none', name=''):
        type_ = SURFACE_PLANE_Y
        surface = cls(type_, boundary_condition, name)
        
        surface.linear = True
        surface.H = 1.0
        surface.J = -y
        surface.ny = 1.0

        return surface


    @classmethod
    def PlaneZ(cls, z=0.0, boundary_condition='none', name=''):
        type_ = SURFACE_PLANE_Z
        surface = cls(type_, boundary_condition, name)
        
        surface.linear = True
        surface.I = 1.0
        surface.J = -z
        surface.nz = 1.0

        return surface
   

    @classmethod
    def Plane(cls, A=0.0, B=0.0, C=0.0, D=0.0, boundary_condition='none', name=''):
        type_ = SURFACE_PLANE
        surface = cls(type_, boundary_condition, name)

        surface.linear = True

        # Normalize
        norm = (A**2 + B**2 + C**2) ** 0.5
        A /= norm
        B /= norm
        C /= norm
        D /= norm

        # Coefficients
        surface.G = A
        surface.H = B
        surface.I = C
        surface.J = D

        # Surface normal direction
        surface.nx = A
        surface.nx = B
        surface.nx = C
        return surface

    @classmethod
    def CylinderX(cls, center=[0.0, 0.0], radius=1.0, boundary_condition='none', name=''):
        type_ = SURFACE_CYLINDER_X
        surface = cls(type_, boundary_condition, name)

        surface.linear = False

        # Center and radius
        y, z = center
        r = radius

        # Coefficients
        surface.B = 1.0
        surface.C = 1.0
        surface.H = -2.0 * y
        surface.I = -2.0 * z
        surface.J = y**2 + z**2 - r**2
        return surface

    @classmethod
    def CylinderY(cls, center=[0.0, 0.0], radius=1.0, boundary_condition='none', name=''):
        type_ = SURFACE_CYLINDER_Y
        surface = cls(type_, boundary_condition, name)

        surface.linear = False

        # Center and radius
        x, z = center
        r = radius

        # Coefficients
        surface.A = 1.0
        surface.C = 1.0
        surface.G = -2.0 * x
        surface.I = -2.0 * z
        surface.J = x**2 + z**2 - r**2
        return surface

    @classmethod
    def CylinderZ(cls, center=[0.0, 0.0], radius=1.0, boundary_condition='none', name=''):
        type_ = SURFACE_CYLINDER_Z
        surface = cls(type_, boundary_condition, name)
        surface.linear = False

        # Center and radius
        x, y = center
        r = radius

        # Coefficients
        surface.A = 1.0
        surface.B = 1.0
        surface.G = -2.0 * x
        surface.H = -2.0 * y
        surface.J = x**2 + y**2 - r**2
 
        return surface

    @classmethod
    def Sphere(cls, center=[0.0, 0.0, 0.0], radius=1.0, boundary_condition='none', name=''):
        type_ = SURFACE_SPHERE
        surface = cls(type_, boundary_condition, name)

        surface.linear = False

        # Center and radius
        x, y, z = center
        r = radius

        # Coefficients
        surface.A = 1.0
        surface.B = 1.0
        surface.C = 1.0
        surface.G = -2.0 * x
        surface.H = -2.0 * y
        surface.I = -2.0 * z
        surface.J = x**2 + y**2 + z**2 - r**2
        return surface

    @classmethod
    def Quadric(cls, A=0.0, B=0.0, C=0.0, D=0.0, E=0.0, F=0.0, G=0.0, H=0.0, I=0.0, J=0.0, boundary_condition='none', name=''):
        type_ = SURFACE_QUADRIC
        surface = cls(type_, boundary_condition, name)

        surface.linear = False

        # Coefficients
        surface.A = A
        surface.B = B
        surface.C = C
        surface.D = D
        surface.E = E
        surface.F = F
        surface.G = G
        surface.H = H
        surface.I = I
        surface.J = J
        return surface


    # ==================================================================================
    # Region building
    # ==================================================================================

    def _create_halfspace(self, positive):
        region = RegionCard("halfspace")
        region.A = self.ID
        if positive:
            region.B = 1
        else:
            region.B = -1

        # Check if an identical halfspace region already existed
        for idx, existing_region in enumerate(global_.input_deck.regions):
            if (
                existing_region.type == "halfspace"
                and region.A == existing_region.A
                and region.B == existing_region.B
            ):
                return global_.input_deck.regions[idx]

        # Set ID and push to deck
        region.ID = len(global_.input_deck.regions)
        global_.input_deck.regions.append(region)
        return region

    def __pos__(self):
        return self._create_halfspace(True)

    def __neg__(self):
        return self._create_halfspace(False)


    # ==================================================================================
    # Surface moving
    # ==================================================================================

    def move(self, velocities, durations):
        self.moving = True
        self.N_move = len(durations) + 1

        if isinstance(velocities, np.ndarray):
            velocities = velocities.tolist()
            durations = durations.tolist()

        self.move_velocities = velocities
        self.move_velocities.append([0.0, 0.0, 0.0])
        self.move_velocities = np.array(self.move_velocities)

        self.move_durations = durations
        self.move_durations.append(INF)
        self.move_durations = np.array(self.move_durations)


# ======================================================================================
# Type decoder
# ======================================================================================

def decode_type(type_):
    if type_ == SURFACE_PLANE_X:
        return "Plane-X surface"
    elif type_ == SURFACE_PLANE_Y:
        return "Plane-Y surface"
    elif type_ == SURFACE_PLANE_Z:
        return "Plane-Z surface"
    elif type_ == SURFACE_PLANE:
        return "Plane surface"
    elif type_ == SURFACE_CYLINDER_X:
        return "Infinite cylinder-X surface"
    elif type_ == SURFACE_CYLINDER_Y:
        return "Infinite cylinder-Y surface"
    elif type_ == SURFACE_CYLINDER_Z:
        return "Infinite cylinder-Z surface"
    elif type_ == SURFACE_SPHERE:
        return "Sphere surface"
    elif type_ == SURFACE_QUADRIC:
        return "Quadric surface"


def decode_BC_type(type_):
    if type_ == BC_NONE:
        return "None"
    elif type_ == BC_VACUUM:
        return "Vacuum"
    elif type_ == BC_REFLECTIVE:
        return "Reflective"
