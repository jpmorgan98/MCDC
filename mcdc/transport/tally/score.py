from numba import njit

####

import mcdc.mcdc_get as mcdc_get
import mcdc.transport.mesh as mesh_
import mcdc.transport.physics as physics

from mcdc.code_factory import adapt
from mcdc.constant import AXIS_T, AXIS_X, AXIS_Y, AXIS_Z, COINCIDENCE_TOLERANCE, INF, MESH_STRUCTURED, MESH_UNIFORM, SCORE_FLUX
from mcdc.transport.geometry.surface import get_normal_component
from mcdc.transport.tally.filter import (
    get_direction_index,
    get_energy_index,
    get_time_index,
)


@njit
def cell_tally(particle_container, distance, tally, mcdc, data):
    particle = particle_container[0]

    # Simulation settings
    MG_mode = mcdc["settings"]["multigroup_mode"]

    # Particle/track properties
    ut = 1.0 / physics.particle_speed(particle_container, mcdc, data)
    t = particle["t"]
    t_final = t + ut * distance

    # Ini_timeial bin indices
    i_mu, i_azi, i_energy, i_time = 0, 0, 0, 0
    if tally["filter_direction"]:
        i_mu, i_azi = get_direction_index(particle_container, tally, data)
        if i_mu == -1 or i_azi == -1:
            return
    if tally["filter_energy"]:
        i_energy = get_energy_index(particle_container, tally, data, MG_mode)
        if i_energy == -1:
            return
    if tally["filter_time"]:
        i_time = get_time_index(particle_container, tally, data)
        if i_time == -1:
            return

    # Tally base index
    idx_base = (
        tally["bin_offset"]
        + i_mu * tally["stride_mu"]
        + i_azi * tally["stride_azi"]
        + i_energy * tally["stride_energy"]
        + i_time * tally["stride_time"]
    )

    # Sweep through the distance
    distance_swept = 0.0
    while distance_swept < distance - COINCIDENCE_TOLERANCE:
        t_next = mcdc_get.cell_tally.time(i_time + 1, tally, data)
        distance_scored = (min(t_next, t_final) - t) / ut

        # Score
        flux = distance_scored * particle["w"]
        for i_score in range(tally["scores_length"]):
            score_type = mcdc_get.cell_tally.scores(i_score, tally, data)
            score = 0.0
            if score_type == SCORE_FLUX:
                score = flux
            adapt.global_add(data, idx_base + i_score, score)

        # Accumulate distance swept
        distance_swept += distance_scored

        # Increment the time
        t += distance_scored * ut

        # Increment index and check if out of bounds
        i_time += 1
        if i_time == tally["time_length"]:
            break
        idx_base += tally["stride_time"]


@njit
def surface_tally(particle_container, surface, tally, mcdc, data):
    particle = particle_container[0]
    material = mcdc["materials"][particle["material_ID"]]

    # Simulation settings
    MG_mode = mcdc["settings"]["multigroup_mode"]

    # Bin indices
    i_mu, i_azi, i_energy, i_time = 0, 0, 0, 0
    if tally["filter_direction"]:
        i_mu, i_azi = get_direction_index(particle_container, tally, data)
        if i_mu == -1 or i_azi == -1:
            return
    if tally["filter_energy"]:
        i_energy = get_energy_index(particle_container, tally, data, MG_mode)
        if i_energy == -1:
            return
    if tally["filter_time"]:
        i_time = get_time_index(particle_container, tally, data)
        if i_time == -1:
            return

    # Tally index
    idx_base = (
        tally["bin_offset"]
        + i_mu * tally["stride_mu"]
        + i_azi * tally["stride_azi"]
        + i_energy * tally["stride_energy"]
        + i_time * tally["stride_time"]
    )

    # Flux
    speed = physics.particle_speed(particle_container, mcdc, data)
    mu = get_normal_component(particle_container, speed, surface, data)
    flux = particle["w"] / abs(mu)

    # Score
    for i_score in range(tally["scores_length"]):
        score_type = mcdc_get.cell_tally.scores(i_score, tally, data)
        score = 0.0
        if score_type == SCORE_FLUX:
            score = flux
        adapt.global_add(data, idx_base + i_score, score)


@njit
def mesh_tally(particle_container, distance, tally, mcdc, data):
    particle = particle_container[0]
    mesh_type = tally["mesh_type"]
    mesh_ID = tally["mesh_ID"]
    if mesh_type == MESH_UNIFORM:
        mesh = mcdc['uniform_meshes'][mesh_ID]
    elif mesh_type == MESH_STRUCTURED:
        mesh = mcdc['structured_meshes'][mesh_ID]

    # Simulation settings
    MG_mode = mcdc["settings"]["multigroup_mode"]

    # Particle/track properties
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    t = particle["t"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]
    ut = 1.0 / physics.particle_speed(particle_container, mcdc, data)
    x_final = x + ux * distance
    y_final = y + uy * distance
    z_final = z + uz * distance
    t_final = t + ut * distance

    # Initial bin indices
    i_mu, i_azi, i_energy, i_time = 0, 0, 0, 0
    if tally["filter_direction"]:
        i_mu, i_azi = get_direction_index(particle_container, tally, data)
        if i_mu == -1 or i_azi == -1:
            return
    if tally["filter_energy"]:
        i_energy = get_energy_index(particle_container, tally, data, MG_mode)
        if i_energy == -1:
            return
    if tally["filter_time"]:
        i_time = get_time_index(particle_container, tally, data)
        if i_time == -1:
            return
    i_x, i_y, i_z = mesh_.get_indices(particle_container, mesh_type, mesh_ID, mcdc, data)
    if i_time == -1 or i_x == -1 or i_y == -1 or i_z == -1:
        return
    
    # Tally base index
    idx_base = (
        tally["bin_offset"]
        + i_mu * tally["stride_mu"]
        + i_azi * tally["stride_azi"]
        + i_energy * tally["stride_energy"]
        + i_time * tally["stride_time"]
        + i_x * tally["stride_x"]
        + i_y * tally["stride_y"]
        + i_z * tally["stride_z"]
    )

    # Sweep through the distance
    distance_swept = 0.0
    while distance_swept < distance - COINCIDENCE_TOLERANCE:
        # ==============================================================================
        # Find distances to the mesh grids
        # ==============================================================================
        
        # x-direction
        if ux == 0.0:
            dx = INF
        else:
            if ux > 0.0:
                x_next = mesh_.get_x(i_x + 1, mesh_type, mesh_ID, mcdc, data)
                x_next = min(x_next, x_final)
            else:
                x_next = mesh_.get_x(i_x, mesh_type, mesh_ID, mcdc, data)
                x_next = max(x_next, x_final)
            dx = (x_next - x) / ux
        
        # y-direction
        if uy == 0.0:
            dy = INF
        else:
            if uy > 0.0:
                y_next = mesh_.get_y(i_y + 1, mesh_type, mesh_ID, mcdc, data)
                y_next = min(y_next, y_final)
            else:
                y_next = mesh_.get_y(i_y, mesh_type, mesh_ID, mcdc, data)
                y_next = max(y_next, y_final)
            dy = (y_next - y) / uy
        
        
        # z-direction
        if uz == 0.0:
            dz = INF
        else:
            if uz > 0.0:
                z_next = mesh_.get_z(i_z + 1, mesh_type, mesh_ID, mcdc, data)
                z_next = min(z_next, z_final)
            else:
                z_next = mesh_.get_z(i_z, mesh_type, mesh_ID, mcdc, data)
                z_next = max(z_next, z_final)
            dz = (z_next - z) / uz

        # t-direction
        t_next = mcdc_get.cell_tally.time(i_time + 1, tally, data)
        dt = (min(t_next, t_final) - t) / ut

        # ==============================================================================
        # Evaluate grid crossings
        # ==============================================================================

        distance_scored = INF
        axis_crossed = -1
        if dx <= distance_scored:
            axis_crossed = AXIS_X
            distance_scored = dx
        if dy <= distance_scored:
            axis_crossed = AXIS_Y
            distance_scored = dy
        if dz <= distance_scored:
            axis_crossed = AXIS_Z
            distance_scored = dz
        if dt <= distance_scored:
            axis_crossed = AXIS_T
            distance_scored = dt

        # Score
        flux = distance_scored * particle["w"]
        for i_score in range(tally["scores_length"]):
            score_type = mcdc_get.cell_tally.scores(i_score, tally, data)
            score = 0.0
            if score_type == SCORE_FLUX:
                score = flux
            adapt.global_add(data, idx_base + i_score, score)

        # Accumulate distance swept
        distance_swept += distance_scored

        # Move the 4D position
        x += distance_scored * ux
        y += distance_scored * uy
        z += distance_scored * uz
        t += distance_scored * ut

        # Increment index and check if out of bounds
        if axis_crossed == AXIS_X:
            if ux > 0.0:
                i_x += 1
                if i_x == mesh["Nx"]:
                    break
                idx_base += tally['stride_x']
            else:
                i_x -= 1
                if i_x == -1:
                    break
                idx_base -= tally['stride_x']
        elif axis_crossed == AXIS_Y:
            if uy > 0.0:
                i_y += 1
                if i_y == mesh["Ny"]:
                    break
                idx_base += tally["stride_y"]
            else:
                i_y -= 1
                if i_y == -1:
                    break
                idx_base -= tally["stride_y"]
        elif axis_crossed == AXIS_Z:
            if uz > 0.0:
                i_z += 1
                if i_z == mesh["Nz"]:
                    break
                idx_base += tally["stride_z"]
            else:
                i_z -= 1
                if i_z == -1:
                    break
                idx_base -= tally["stride_z"]
        elif axis_crossed == AXIS_T:
            i_time += 1
            if i_time == tally["time_length"] - 1:
                break
            idx_base += tally["stride_time"]
