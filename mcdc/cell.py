from operator import attrgetter
import numpy as np
import sympy

####

from mcdc import objects
from mcdc.constant import BOOL_AND, BOOL_NOT, BOOL_OR, FILL_LATTICE, FILL_MATERIAL, FILL_UNIVERSE, INF, PI
from mcdc.material import MaterialBase
from mcdc.objects import ObjectNonSingleton
from mcdc.prints import print_error
from mcdc.util import flatten

# ======================================================================================
# Region
# ======================================================================================

# Region-making helper that checks if an identical region is already created
def make_region(type_, A, B):
    for existing_region in objects.regions:
        if (
            type_ == existing_region.type
            and A == existing_region.A
            and B == existing_region.B
        ):
            return existing_region
    return Region(type_, A, B)


class Region(ObjectNonSingleton):
    def __init__(self, type_, A, B):
        label = "region"
        super().__init__(label)

        self.type = type_
        self.A = A
        self.B = B
   
    @classmethod
    def make_halfspace(cls, surface, sense):
        region = make_region('halfspace', surface, sense)
        return region

    def __and__(self, other):
        return make_region('intersection', self, other)

    def __or__(self, other):
        return make_region('union', self, other)

    def __invert__(self):
        return make_region('complement', self, None)

    def __repr__(self):
        text = "Region: "
        if self.type == "halfspace":
            if self.B > 0:
                text += "+s%i" % self.A.ID
            else:
                text += "-s%i" % self.A.ID
        elif self.type == "intersection":
            text += "r%i & r%i" % (self.A.ID, self.B.ID)
        elif self.type == "union":
            text += "r%i | r%i" % (self.A.ID, self.B.ID)
        elif self.type == "complement":
            text += "~r%i" % (self.A.ID)
        elif self.type == "all":
            text += "all"

        return text


# ======================================================================================
# Cell
# ======================================================================================


class Cell(ObjectNonSingleton):
    def __init__(self, name='', region=None, fill=None, translation=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0]):
        label = "cell"
        super().__init__(label)

        self.name = name

        # Set region
        if region is None:
            self.region = make_region('all', None, None)
        else:
            self.region = region
        
        # Set fill
        self.fill = fill

        # For Numba mode: primitive fill
        self.non_numba += ['fill']
        if isinstance(fill, MaterialBase):
            self.fill_type = FILL_MATERIAL
            self.fill_ID = fill.ID
        elif isinstance(fill, Universe):
            self.fill_type = FILL_UNIVERSE
            self.fill_ID = fill.ID
        elif isinstance(fill, Lattice):
            self.fill_type = FILL_LATTICE
            self.fill_ID = fill.ID
        else:
            print_error(f"Unsupported cell fill: {fill}")

        # Local coordinate modifier
        self.translation = np.array(translation, dtype=float)
        self.rotation = np.array(rotation, dtype=float)
        self.fill_translated = False
        self.fill_rotated = False
        if (self.translation != 0.0).any():
            self.fill_translated = True
        if (self.rotation != 0.0).any():
            self.fill_rotated = True
            # Convert ritation
            self.rotation *= PI / 180.0

        # Set region Reversed Polished Notation (RPN)
        if self.region.type != 'all':
            self.region_RPN_tokens = generate_RPN_tokens(self.region)
            self.region_RPN = generate_RPN(self.region_RPN_tokens)
        else:
            self.region_RPN_tokens = []
            self.region_RPN = ""

        # List surfaces
        self.surfaces = list_surfaces(self.region_RPN_tokens)

        # TODO: Tally
   
        # Non Numba
        self.non_numba += ['region', "region_RPN"]


    def __repr__(self):
        text = "\n"
        text += f"Cell\n"
        text += f"  - ID: {self.ID}\n"
        if self.name != '':
            text += f"  - Name: {self.name}\n"
        text += f"  - {self.region}\n"
        if isinstance(self.fill, MaterialBase):
            text += f"  - Fill: Material [ID: {self.fill.ID}]\n"
        if self.fill_translated:
            text += f"  - Translation: {self.translation}\n"
        if self.fill_rotated:
            text += f"  - Rotation: {self.rotation * 180 / PI}\n"
        text += f"Bounding surfaces: {[x.ID for x in self.surfaces]}"
        return text


def generate_RPN_tokens(region):
    # The RPN tokens
    rpn_tokens = []

    # Build RPN based on recursive evaluation of the region
    stack = [region]
    while len(stack) > 0:
        token = stack.pop()
        if isinstance(token, Region):
            if token.type == "halfspace":
                rpn_tokens.append(token.A.ID)
                if token.B < 0:
                    rpn_tokens.append(BOOL_NOT)
            elif token.type == "intersection":
                stack += ["&", token.A, token.B]
            elif token.type == "union":
                stack += ["|", token.A, token.B]
            elif token.type == "complement":
                stack += ["~", token.A]
        else:
            if token == "&":
                rpn_tokens.append(BOOL_AND)
            elif token == "|":
                rpn_tokens.append(BOOL_OR)
            elif token == "~":
                rpn_tokens.append(BOOL_NOT)
            else:
                print_error(f"Unrecognized token in the generating region RPN: {token}")

    return np.array(rpn_tokens)


def generate_RPN(rpn_tokens):
    stack = []

    for token in rpn_tokens:
        if token >= 0:
            stack.append(sympy.symbols(f"s{token}"))
        else:
            if token == BOOL_AND or token == BOOL_OR:
                item_1 = stack.pop()
                item_2 = stack.pop()
                if token == BOOL_AND:
                    stack.append(item_1 & item_2)
                else:
                    stack.append(item_1 | item_2)

            elif token == BOOL_NOT:
                item = stack.pop()
                if isinstance(item, Region):
                    item = sympy.symbols(str(item)[8:])

                stack.append(~item)
        
    return sympy.logic.boolalg.simplify_logic(stack[0])
   

def list_surfaces(rpn_tokens):
    surfaces = []

    for token in rpn_tokens:
        if token >= 0:
            surface = objects.surfaces[token]
            if surface not in surfaces:
                surfaces.append(surface)

    return sorted(surfaces, key=attrgetter("ID"))


# ======================================================================================
# Universe
# ======================================================================================

class Universe(ObjectNonSingleton):
    def __init__(self, name='', cells=[], root=False):
        # Custom treatment for root universe
        label = "universe"
        if not root:
            super().__init__(label)
        else:
            super().__init__(label, automatic_registration=False)
            objects.universes[0] = self
            self.ID = 0

        self.name = name
        self.cells = cells
    

    def __repr__(self):
        text = "\n"
        text += f"Universe\n"
        if self.ID == 0:
            text += f"  - ID: {self.ID} (root)\n"
        else:
            text += f"  - ID: {self.ID}\n"
        if self.name != '':
            text += f"  - Name: {self.name}\n"
        text += f"Cells: {[x.ID for x in self.cells]}"
        return text


# ======================================================================================
# Lattice
# ======================================================================================


class Lattice(ObjectNonSingleton):
    def __init__(self, name='', x=None, y=None, z=None, universes=None):
        label = "lattice"
        super().__init__(label)

        self.name = name

        # Default uniform grids
        self.x0 = -INF
        self.dx = 2 * INF
        self.Nx = 1
        self.y0 = -INF
        self.dy = 2 * INF
        self.Ny = 1
        self.z0 = -INF
        self.dz = 2 * INF
        self.Nz = 1
        self.t0 = 0.0 # Placeholder time grid is needed to use mesh indexing function
        self.dt = INF
        self.Nt = 1

        # Set the grid
        if x is not None:
            self.x0 = x[0]
            self.dx = x[1]
            self.Nx = x[2]
        if y is not None:
            self.y0 = y[0]
            self.dy = y[1]
            self.Ny = y[2]
        if z is not None:
            self.z0 = z[0]
            self.dz = z[1]
            self.Nz = z[2]
    
        # Set universe IDs
        get_ID = np.vectorize(lambda obj: obj.ID)
        universe_IDs = get_ID(universes)
        ax_expand = []
        if x is None:
            ax_expand.append(2)
        if y is None:
            ax_expand.append(1)
        if z is None:
            ax_expand.append(0)
        for ax in ax_expand:
            universe_IDs = np.expand_dims(universe_IDs, axis=ax)

        # Change indexing structure: [z(flip), y(flip), x] --> [x, y, z]
        universe_IDs = np.transpose(universe_IDs)
        universe_IDs = np.flip(universe_IDs, axis=1)
        universe_IDs = np.flip(universe_IDs, axis=2)

        # Set up the universes
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        self.universes = []
        for ix in range(Nx):
            self.universes.append([])
            for iy in range(Ny):
                self.universes[-1].append([])
                for iz in range(Nz):
                    ID = universe_IDs[ix, iy, iz]
                    self.universes[-1][-1].append(objects.universes[ID])


    def __repr__(self):
        text = "\n"
        text += f"Lattice\n"
        text += f"  - ID: {self.ID}\n"
        if self.name != '':
            text += f"  - Name: {self.name}\n"
        text += f"  - (x0, dx, Nx): ({self.x0}, {self.dx}, {self.Nx})\n"
        text += f"  - (y0, dy, Ny): ({self.y0}, {self.dy}, {self.Ny})\n"
        text += f"  - (z0, dz, Nz): ({self.z0}, {self.dz}, {self.Nz})\n"
        text += f"Universes: {set([x.ID for x in list(flatten(self.universes))])}"
        return text
