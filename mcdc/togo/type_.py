import math
import numpy as np

from mpi4py import MPI
from mpi4py.util.dtlib import from_numpy_dtype
from numba import njit

####

from mcdc.constant import WW_PREVIOUS
from mcdc.object_.simulation import simulation
from mcdc.print_ import print_error


# ==============================================================================
# Basic types
# ==============================================================================

float64 = np.float64
int64 = np.int64
int32 = np.int32
uint64 = np.uint64
uint8 = np.uint8
bool_ = np.bool_
uintp = np.uintp
str_ = "U32"


# ==============================================================================
# MC/DC types
# ==============================================================================
"""
Some types are problem-dependent and defined in code_factory.py
"""

particle = None
particle_record = None

source = None
technique = None

gpu_meta = None

global_ = None
global_size = None

# ==============================================================================
# MC/DC Member Array Sizes
# ==============================================================================


def literalize(value):
    jit_str = f"@njit\ndef impl():\n    return {value}\n"
    exec(jit_str, globals(), locals())
    return eval("impl")


def material_g_size():
    pass


def material_j_size():
    pass


def rpn_buffer_size():
    pass


def make_size_rpn(cells):
    global rpn_buffer_size
    size = max([np.sum(np.array(x.region_RPN_tokens) >= 0.0) for x in cells])
    rpn_buffer_size = literalize(size)


# ==============================================================================
# Copy Logic
# ==============================================================================


type_roster = {}


def copy_fn_for(kind, name):
    code = f"@njit\ndef copy_{name}(dst,src):\n"
    for f_name, spec in kind.fields.items():
        f_dtype = spec[0]
        if f_dtype in type_roster:
            kind_name = type_roster[f_dtype]["name"]
            code += f"    copy_{kind_name}(dst['{f_name}'],src['{f_name}'])"
        else:
            code += f"    dst['{f_name}'] = src['{f_name}']\n"
    type_roster[kind] = {}
    type_roster[kind]["name"] = name
    exec(code)
    return eval(f"copy_{name}")


# ==============================================================================
# Particle bank
# ==============================================================================


def full_particle_bank(max_size):
    return into_dtype(
        [
            ("particles", particle, (max_size,)),
            ("size", int64, (1,)),
            ("tag", str_),
        ]
    )


def particle_bank(max_size):
    return [
        ("particles", particle_record, (max_size,)),
        ("size", int64, (1,)),
        ("tag", str_),
    ]


# ==============================================================================
# GPU Metadata
# ==============================================================================


def make_type_gpu_meta():
    global gpu_meta

    gpu_meta = into_dtype(
        [
            ("state_pointer", uintp),
            ("source_program_pointer", uintp),
            ("precursor_program_pointer", uintp),
            ("global_pointer", uintp),
            ("tally_pointer", uintp),
        ]
    )


# ==============================================================================
# Global
# ==============================================================================


def make_type_global(input_deck, structures, records):
    global global_

    # Get modes
    mode_MG = simulation.settings.multigroup_mode
    mode_CE = not mode_MG

    # Numbers of simulation
    N_source = len(input_deck.sources)

    # Simulation parameters
    settings = simulation.settings
    N_particle = settings.N_particle
    N_cycle = settings.N_inactive + settings.N_active

    # Particle bank buffers
    bank_active_buff = settings.active_bank_buffer
    bank_census_buff = settings.census_bank_buffer_ratio
    bank_source_buff = settings.source_bank_buffer_ratio
    bank_future_buff = settings.future_bank_buffer_ratio

    # Number of precursor groups
    if mode_MG:
        J = simulation.materials[0].J
    if mode_CE:
        J = 6

    # Number of work
    N_work = math.ceil(N_particle / MPI.COMM_WORLD.Get_size())

    # Particle bank types
    bank_active = particle_bank(1 + bank_active_buff)
    if settings.eigenvalue_mode or settings.N_census > 1:
        bank_census = particle_bank(int((1 + bank_census_buff) * N_work))
        bank_source = particle_bank(int((1 + bank_source_buff) * N_work))
        bank_future = particle_bank(int((1 + bank_future_buff) * N_work))
    else:
        bank_census = particle_bank(0)
        bank_source = particle_bank(0)
        bank_future = particle_bank(0)

    # iQMC bank adjustment
    if input_deck.technique["iQMC"]:
        bank_source = particle_bank(N_work)
        if input_deck.setting["mode_eigenvalue"]:
            bank_census = particle_bank(0)
            bank_future = particle_bank(0)

    # Source and IC files bank adjustments
    if not settings.eigenvalue_mode:
        if settings.use_source_file:
            bank_source = particle_bank(N_work)

    if (
        settings.use_source_file and not settings.eigenvalue_mode
    ) or input_deck.technique["iQMC"]:
        bank_source = particle_bank(N_work)

    # Set the global structure
    global_structure = structures["simulation"]
    global_structure += [
        ("bank_active", bank_active),
        ("bank_census", bank_census),
        ("bank_source", bank_source),
        ("bank_future", bank_future),
    ]

    # GLobal type
    global_ = global_structure
