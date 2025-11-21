import math
import numpy as np

from numba import njit

####

import mcdc.code_factory.adapt as adapt
import mcdc.mcdc_get as mcdc_get
import mcdc.object_.numba_types as type_
import mcdc.transport.particle as particle_module
import mcdc.transport.rng as rng

from mcdc.constant import (
    E_THERMAL_THRESHOLD,
    PI,
    PI_HALF,
    PI_SQRT,
    REACTION_TOTAL,
    REACTION_NEUTRON_CAPTURE,
    REACTION_NEUTRON_ELASTIC_SCATTERING,
    REACTION_NEUTRON_FISSION,
    SQRT_E_TO_SPEED,
    SQRD_SPEED_TO_E,
)
from mcdc.transport.data import evaluate_data
from mcdc.transport.distribution import (
    sample_distribution,
    sample_isotropic_direction,
    sample_multi_table,
)
from mcdc.transport.physics.util import evaluate_xs_energy_grid, scatter_direction
from mcdc.transport.util import linear_interpolation


# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container):
    particle = particle_container[0]
    return math.sqrt(particle["E"]) * SQRT_E_TO_SPEED


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, mcdc, data):
    particle = particle_container[0]
    material = mcdc["native_materials"][particle["material_ID"]]
    E = particle["E"]

    total = 0.0
    for i in range(material["N_nuclide"]):
        nuclide_ID = int(mcdc_get.native_material.nuclide_IDs(i, material, data))
        nuclide = mcdc["nuclides"][nuclide_ID]

        nuclide_density = mcdc_get.native_material.nuclide_densities(i, material, data)
        if reaction_type == REACTION_TOTAL:
            xs = total_micro_xs(E, nuclide, data)
        else:
            xs = micro_xs(reaction_type, E, nuclide, mcdc, data)

        total += nuclide_density * xs

    return total


@njit
def total_micro_xs(E, nuclide, data):
    idx, E0, E1 = evaluate_xs_energy_grid(E, nuclide, data)
    xs0 = mcdc_get.nuclide.total_xs(idx, nuclide, data)
    xs1 = mcdc_get.nuclide.total_xs(idx + 1, nuclide, data)
    return linear_interpolation(E, E0, E1, xs0, xs1)


@njit
def micro_xs(reaction_type, E, nuclide, mcdc, data):
    for i in range(nuclide["N_reaction"]):
        reaction_ID = int(mcdc_get.nuclide.reaction_IDs(i, nuclide, data))
        reaction = mcdc["reactions"][reaction_ID]
        if reaction_type == reaction["child_type"]:
            return reaction_micro_xs(E, reaction, nuclide, data)
    return 0.0


@njit
def reaction_micro_xs(E, reaction, nuclide, data):
    idx, E0, E1 = evaluate_xs_energy_grid(E, nuclide, data)
    xs0 = mcdc_get.reaction.xs(idx, reaction, data)
    xs1 = mcdc_get.reaction.xs(idx + 1, reaction, data)
    return linear_interpolation(E, E0, E1, xs0, xs1)


@njit
def neutron_production_xs(reaction_type, particle_container, mcdc, data):
    particle = particle_container[0]
    material_base = mcdc["materials"][particle["material_ID"]]
    material = mcdc["native_materials"][material_base["child_ID"]]

    if reaction_type == REACTION_TOTAL:
        elastic_type = REACTION_NEUTRON_ELASTIC_SCATTERING
        fission_type = REACTION_NEUTRON_FISSION
        elastic_xs = neutron_production_xs(elastic_type, particle_container, mcdc, data)
        fission_xs = neutron_production_xs(fission_type, particle_container, mcdc, data)
        return elastic_xs + fission_xs

    elif reaction_type == REACTION_NEUTRON_CAPTURE:
        return 0.0

    elif reaction_type == REACTION_NEUTRON_ELASTIC_SCATTERING:
        return macro_xs(reaction_type, particle_container, mcdc, data)

    elif reaction_type == REACTION_NEUTRON_FISSION:
        if not material_base["fissionable"]:
            return 0.0

        total = 0.0
        for i in range(material["N_nuclide"]):
            nuclide_ID = int(mcdc_get.native_material.nuclide_IDs(i, material, data))
            nuclide = mcdc["nuclides"][nuclide_ID]
            if not nuclide["fissionable"]:
                continue

            E = particle["E"]
            nuclide_density = mcdc_get.native_material.nuclide_densities(
                i, material, data
            )
            xs = micro_xs(reaction_type, E, nuclide, mcdc, data)

            for j in range(nuclide["N_reaction"]):
                reaction_ID = int(mcdc_get.nuclide.reaction_IDs(j, nuclide, data))
                reaction_base = mcdc["reactions"][reaction_ID]
                reaction = mcdc["neutron_fission_reactions"][reaction_base["child_ID"]]
                nu = fission_yield_prompt(E, reaction, mcdc, data)
                for group in range(reaction["N_delayed"]):
                    nu += fission_yield_delayed(E, group, reaction, mcdc, data)
                total += nuclide_density * nu * xs
        return total

    else:
        return -1.0


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, prog, data):
    particle = particle_container[0]
    mcdc = adapt.mcdc_global(prog)
    material = mcdc["native_materials"][particle["material_ID"]]

    # ==================================================================================
    # Sample colliding nuclide
    # ==================================================================================

    SigmaT = macro_xs(REACTION_TOTAL, particle_container, mcdc, data)

    # Implicit capture
    if mcdc["implicit_capture"]["active"]:
        SigmaC = macro_xs(REACTION_NEUTRON_CAPTURE, particle_container, mcdc, data)
        particle["w"] *= (SigmaT - SigmaC) / SigmaT
        SigmaT -= SigmaC

    xi = rng.lcg(particle_container) * SigmaT
    total = 0.0
    for i in range(material["N_nuclide"]):
        nuclide_ID = int(mcdc_get.native_material.nuclide_IDs(i, material, data))
        nuclide = mcdc["nuclides"][nuclide_ID]

        nuclide_density = mcdc_get.native_material.nuclide_densities(i, material, data)
        sigmaT = total_micro_xs(particle["E"], nuclide, data)

        if mcdc["implicit_capture"]["active"]:
            sigmaC = micro_xs(
                particle["E"], REACTION_NEUTRON_CAPTURE, nuclide, mcdc, data
            )
            particle["w"] *= (sigmaT - sigmaC) / sigmaT
            sigmaT -= sigmaC

        SigmaT_nuclide = nuclide_density * sigmaT
        total += SigmaT_nuclide

        if total > xi:
            break

    # ==================================================================================
    # Sample and perform reaction
    # ==================================================================================

    xi = rng.lcg(particle_container) * sigmaT
    total = 0.0
    for i in range(nuclide["N_reaction"]):
        reaction_ID = int(mcdc_get.nuclide.reaction_IDs(i, nuclide, data))
        reaction = mcdc["reactions"][reaction_ID]
        reaction_type = reaction["child_type"]

        if (
            mcdc["implicit_capture"]["active"]
            and reaction_type == REACTION_NEUTRON_CAPTURE
        ):
            continue

        reaction_xs = reaction_micro_xs(particle["E"], reaction, nuclide, data)
        total += reaction_xs
        if total < xi:
            continue

        # Execute the sampled reaction
        execute_reaction(reaction, particle_container, nuclide, prog, data)

        return


@njit
def execute_reaction(reaction_base, particle_container, nuclide, prog, data):
    particle = particle_container[0]
    reaction_type = reaction_base["child_type"]

    if reaction_type == REACTION_NEUTRON_CAPTURE:
        particle["alive"] = False
    elif reaction_type == REACTION_NEUTRON_ELASTIC_SCATTERING:
        elastic_scattering(reaction_base, particle_container, nuclide, prog, data)
    elif reaction_type == REACTION_NEUTRON_FISSION:
        fission(reaction_base, particle_container, nuclide, prog, data)


# ======================================================================================
# Elastic scattering
# ======================================================================================


@njit
def elastic_scattering(reaction_base, particle_container, nuclide, prog, data):
    mcdc = adapt.mcdc_global(prog)
    reaction_ID = reaction_base["child_ID"]
    reaction = mcdc["neutron_elastic_scattering_reactions"][reaction_ID]

    # Particle attributes
    particle = particle_container[0]
    E = particle["E"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Sample nucleus thermal velocity
    A = nuclide["atomic_weight_ratio"]
    if E > E_THERMAL_THRESHOLD:
        Vx = 0.0
        Vy = 0.0
        Vz = 0.0
    else:
        Vx, Vy, Vz = sample_nucleus_velocity(A, particle_container, mcdc, data)

    # =========================================================================
    # COM kinematics
    # =========================================================================

    # Particle speed
    speed = particle_speed(particle_container)

    # Neutron velocity - LAB
    vx = speed * ux
    vy = speed * uy
    vz = speed * uz

    # COM velocity
    COM_x = (vx + A * Vx) / (1.0 + A)
    COM_y = (vy + A * Vy) / (1.0 + A)
    COM_z = (vz + A * Vz) / (1.0 + A)

    # Neutron velocity - COM
    vx = vx - COM_x
    vy = vy - COM_y
    vz = vz - COM_z

    # Neutron speed - COM
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)

    # Neutron initial direction - COM
    ux = vx / speed
    uy = vy / speed
    uz = vz / speed

    # Sample the scattering cosine from the multi-PDF distribution
    multi_table = mcdc["multi_table_distributions"][reaction["mu_ID"]]
    mu0 = sample_multi_table(E, particle_container, multi_table, data)

    # Scatter the direction in COM
    azi = 2.0 * PI * rng.lcg(particle_container)
    ux_new, uy_new, uz_new = scatter_direction(ux, uy, uz, mu0, azi)

    # Neutron final velocity - COM
    vx = speed * ux_new
    vy = speed * uy_new
    vz = speed * uz_new

    # =========================================================================
    # COM to LAB
    # =========================================================================

    # Final velocity - LAB
    vx = vx + COM_x
    vy = vy + COM_y
    vz = vz + COM_z

    # Final energy - LAB
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    particle["E"] = SQRD_SPEED_TO_E * speed * speed

    # Final direction - LAB
    particle["ux"] = vx / speed
    particle["uy"] = vy / speed
    particle["uz"] = vz / speed


@njit
def sample_nucleus_velocity(A, particle_container, mcdc, data):
    particle = particle_container[0]

    # Particle speed
    speed = particle_speed(particle_container)

    # Maxwellian parameter
    beta = math.sqrt(2.0659834e-11 * A)
    # The constant above is
    #   (1.674927471e-27 kg) / (1.38064852e-19 cm^2 kg s^-2 K^-1) / (293.6 K)/2

    # Sample nuclide speed candidate V_tilda and
    #   nuclide-neutron polar cosine candidate mu_tilda via
    #   rejection sampling
    y = beta * speed
    while True:
        if rng.lcg(particle_container) < 2.0 / (2.0 + PI_SQRT * y):
            x = math.sqrt(
                -math.log(rng.lcg(particle_container) * rng.lcg(particle_container))
            )
        else:
            cos_val = math.cos(PI_HALF * rng.lcg(particle_container))
            x = math.sqrt(
                -math.log(rng.lcg(particle_container))
                - math.log(rng.lcg(particle_container)) * cos_val * cos_val
            )
        V_tilda = x / beta
        mu_tilda = 2.0 * rng.lcg(particle_container) - 1.0

        # Accept candidate V_tilda and mu_tilda?
        if rng.lcg(particle_container) > math.sqrt(
            speed * speed + V_tilda * V_tilda - 2.0 * speed * V_tilda * mu_tilda
        ) / (speed + V_tilda):
            break

    # Set nuclide velocity - LAB
    azi = 2.0 * PI * rng.lcg(particle_container)
    ux, uy, uz = scatter_direction(
        particle["ux"], particle["uy"], particle["uz"], mu_tilda, azi
    )
    Vx = ux * V_tilda
    Vy = uy * V_tilda
    Vz = uz * V_tilda

    return Vx, Vy, Vz


# ======================================================================================
# Fission
# ======================================================================================


@njit
def fission(reaction_base, particle_container, nuclide, prog, data):
    mcdc = adapt.mcdc_global(prog)
    settings = mcdc["settings"]

    reaction_ID = reaction_base["child_ID"]
    reaction = mcdc["neutron_fission_reactions"][reaction_ID]

    # Particle properties
    particle = particle_container[0]
    E = particle["E"]

    # Kill the current particle
    particle["alive"] = False

    # Adjust production and product weights if weighted emission
    weight_production = 1.0
    weight_product = particle["w"]
    if mcdc["weighted_emission"]["active"]:
        weight_target = mcdc["weighted_emission"]["weight_target"]
        weight_production = particle["w"] / weight_target
        weight_product = weight_target

    # Fission yields
    N_delayed = reaction["N_delayed"]
    nu_p = fission_yield_prompt(E, reaction, mcdc, data)
    nu_d = np.zeros(N_delayed)
    nu_d_total = 0.0
    for j in range(N_delayed):
        nu_d[j] = fission_yield_delayed(E, j, reaction, mcdc, data)
        nu_d_total += nu_d[j]
    nu = nu_p + nu_d_total

    # Get number of secondaries
    N = int(
        math.floor(weight_production * nu / mcdc["k_eff"] + rng.lcg(particle_container))
    )

    # Set up secondary partice container
    particle_container_new = np.zeros(1, type_.particle_data)
    particle_new = particle_container_new[0]

    # Create the secondaries
    for n in range(N):
        # Set default attributes
        particle_module.copy_as_child(particle_container_new, particle_container)

        # Set weight
        particle_new["w"] = weight_product

        # Sample isotropic direction
        ux_new, uy_new, uz_new = sample_isotropic_direction(particle_container_new)
        particle_new["ux"] = ux_new
        particle_new["uy"] = uy_new
        particle_new["uz"] = uz_new

        # Prompt or delayed?
        prompt = True
        delayed_group = -1
        xi = rng.lcg(particle_container_new) * nu
        total = nu_p
        if xi > total:
            prompt = False
            # Determine delayed group
            for j in range(N_delayed):
                total += nu_d[j]
                if xi < total:
                    delayed_group = j
                    break

        # Sample outgoing energy
        if prompt:
            particle_new["E"] = sample_fission_spectrum_prompt(
                E, reaction, particle_container_new, mcdc, data
            )
        else:
            particle_new["E"] = sample_fission_spectrum_delayed(
                E, delayed_group, reaction, particle_container_new, mcdc, data
            )

        # Sample emission time
        decay = mcdc_get.neutron_fission_reaction.delayed_decay_rates(
            delayed_group, reaction, data
        )
        if not prompt:
            xi = rng.lcg(particle_container_new)
            particle_new["t"] -= math.log(xi) / decay

        # Eigenvalue mode: bank right away
        if settings["eigenvalue_mode"]:
            adapt.add_census(particle_container_new, prog)
            continue
        # Below is only relevant for fixed-source problem

        # Skip if it's beyond time boundary
        if particle_new["t"] > settings["time_boundary"]:
            continue

        # Check if it hits current or next census times
        hit_current_census = False
        hit_future_census = False
        idx_census = mcdc["idx_census"]
        if settings["N_census"] > 1:
            if particle_new["t"] > mcdc_get.settings.census_time(
                idx_census, settings, data
            ):
                hit_current_census = True
                if particle_new["t"] > mcdc_get.settings.census_time(
                    idx_census + 1, settings, data
                ):
                    hit_future_census = True

        # Not hitting census --> add to active bank
        if not hit_current_census:
            # Keep it if it is the last particle
            if n == N - 1:
                particle["alive"] = True
                particle["ux"] = particle_new["ux"]
                particle["uy"] = particle_new["uy"]
                particle["uz"] = particle_new["uz"]
                particle["t"] = particle_new["t"]
                particle["g"] = particle_new["g"]
                particle["E"] = particle_new["E"]
                particle["w"] = particle_new["w"]
            else:
                adapt.add_active(particle_container_new, prog)

        # Hit future census --> add to future bank
        elif hit_future_census:
            # Particle will participate in the future
            adapt.add_future(particle_container_new, prog)

        # Hit current census --> add to census bank
        else:
            # Particle will participate after the current census is completed
            adapt.add_census(particle_container_new, prog)


@njit
def fission_yield_prompt(E, reaction, mcdc, data):
    data_base = mcdc["data"][reaction["prompt_yield_ID"]]
    return evaluate_data(E, data_base, mcdc, data)


@njit
def fission_yield_delayed(E, group, reaction, mcdc, data):
    ID = int(mcdc_get.neutron_fission_reaction.delayed_yield_IDs(group, reaction, data))
    data_base = mcdc["data"][ID]
    return evaluate_data(E, data_base, mcdc, data)


@njit
def sample_fission_spectrum_prompt(E, reaction, rng_state, mcdc, data):
    distribution = mcdc["distributions"][reaction["prompt_spectrum_ID"]]
    return sample_distribution(E, distribution, rng_state, mcdc, data, scale=True)


@njit
def sample_fission_spectrum_delayed(E, group, reaction, rng_state, mcdc, data):
    ID = int(
        mcdc_get.neutron_fission_reaction.delayed_spectrum_IDs(group, reaction, data)
    )
    distribution = mcdc["distributions"][ID]
    return sample_distribution(E, distribution, rng_state, mcdc, data, scale=True)
