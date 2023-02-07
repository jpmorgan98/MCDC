import math
import numpy as np

FLOAT_DTYPE = np.float64
INT_DTYPE = np.int32

# Events
EVENT_COLLISION            :INT_DTYPE = 1
EVENT_SURFACE              :INT_DTYPE = 2
EVENT_CENSUS               :INT_DTYPE = 3
EVENT_CENSUS_N_MESH        :INT_DTYPE = 30
EVENT_MESH                 :INT_DTYPE = 4
EVENT_SURFACE_N_MESH       :INT_DTYPE = 40
EVENT_SURFACE_MOVE         :INT_DTYPE = 41
EVENT_SURFACE_MOVE_N_MESH  :INT_DTYPE = 42
EVENT_SCATTERING           :INT_DTYPE = 5
EVENT_FISSION              :INT_DTYPE = 6
EVENT_CAPTURE              :INT_DTYPE = 7
EVENT_TIME_BOUNDARY        :INT_DTYPE = 8
EVENT_TIME_BOUNDARY_N_MESH :INT_DTYPE = 80
EVENT_LATTICE              :INT_DTYPE = 9
EVENT_LATTICE_N_MESH       :INT_DTYPE = 90

# Mesh crossing flags
MESH_X :INT_DTYPE = 0
MESH_Y :INT_DTYPE = 1
MESH_Z :INT_DTYPE = 2
MESH_T :INT_DTYPE = 3

# Gyration raius type
GR_ALL        :INT_DTYPE = 0
GR_INFINITE_X :INT_DTYPE = 1
GR_INFINITE_Y :INT_DTYPE = 2
GR_INFINITE_Z :INT_DTYPE = 3
GR_ONLY_X     :INT_DTYPE = 4
GR_ONLY_Y     :INT_DTYPE = 5
GR_ONLY_Z     :INT_DTYPE = 6

# Population control
PCT_NONE           :INT_DTYPE = 0
PCT_COMBING        :INT_DTYPE = 1
PCT_COMBING_WEIGHT :INT_DTYPE = 10

# Misc.
INF   :FLOAT_DTYPE = 1E10
PI    :FLOAT_DTYPE = math.acos(-1.0)
SHIFT :FLOAT_DTYPE = 1E-10 # To ensure lattice, surface, and mesh crossings
PREC  :FLOAT_DTYPE = 1.0+1E-5 # Precision factor to determine if a distance is smaller than
                 # another (for lattice, surface, and mesh)
BANKMAX :INT_DTYPE = 100 # Default maximum active bank
#PLANCK = 
#BOLTZMANN
#

