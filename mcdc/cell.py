from operator import attrgetter
import numpy as np
import sympy

####

from mcdc import objects
from mcdc.constant import BOOL_AND, BOOL_NOT, BOOL_OR, FILL_MATERIAL
from mcdc.material import MaterialBase
from mcdc.objects import ObjectNonSingleton
from mcdc.prints import print_error

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
        else:
            print_error(f"Unsupported cell fill: {fill}")

        # Local coordinate modifier
        self.translation = np.array(translation)
        self.rotation = np.array(rotation)
        self.fill_translated = False
        self.fill_rotated = False
        if (self.translation != 0.0).any():
            self.fill_translated = True
        if (self.rotation != 0.0).any():
            self.fill_rotated = True

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
        text += f"  - Region: {self.region}\n"
        if isinstance(self.fill, MaterialBase):
            text += f"  - Fill: Material [ID: {self.fill.ID}]\n"
        if self.fill_translated:
            text += f"  - Translation: {self.translation}\n"
        if self.fill_rotated:
            text += f"  - Rotation: {self.rotation}\n"
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
