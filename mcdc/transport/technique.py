import mcdc.transport.kernel as kernel

def weight_roulette(particle_container, mcdc):
    particle = particle_container[0]
    if particle['w'] < mcdc['weight_roulette']['weight_threshold']:
        w_target = mcdc['weight_roulette']['weight_target']
        survival_probability = particle['w'] / w_target
        if kernel.rng(particle_container) < survival_probability:
            particle['w'] = w_target
        else:
            particle['alive'] = False
