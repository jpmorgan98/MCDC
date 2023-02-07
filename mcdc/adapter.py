from numba import njit, cuda, jit

import type_, kernel

from constant import *

# =============================================================================
# Loop adapters
# =============================================================================

def loop(func, target):
    if target == 'cpu':
        return jit(func)
    else:
        def wrap(mcdc, hostco):
            # Create device copies
            d_mcdc = cuda.to_device(mcdc)
            func(d_mcdc, hostco)
        return wrap

# =============================================================================
# Kernel adapters
# =============================================================================
# gives the kernel it's data

def event(func, alg, target, event, branching=False, naive=False):
    func = compiler(func, target)

    if alg == 'history':
        return func
   
    # Event-based zone below

    wrap = None

    def wrap_streaming(mcdc, hostco):
        # Stack index of the current event
        stack = mcdc['stack_idx'][event]
        
        # Stack size
        N = mcdc['stack_'][stack]['size']
        start, stride = kernel.get_idx()
        for i in range(start, N, stride):
            # Get particle index from stack
            idx = mcdc['stack_'][stack]['content'][i]
            
            # "Pop" particle from bank
            P = kernel.read_particle(mcdc['bank']['content'][idx])

            # Set RNG seed
            P['seed'] = mcdc['seed']
            kernel.rng_skip_ahead(i*mcdc['event_stride'][event], P, mcdc)

            # Perform event
            func(P, mcdc)
           
            # Update particle in the bank
            mcdc['bank']['content'][idx] = kernel.record_particle(P)
               
            # Get stack index of the next event
            next_event = P['event']
            next_stack = mcdc['stack_idx'][next_event]

            # Update stack of the next event
            idx_offset = mcdc['stack_'][next_stack]['size']
            mcdc['stack_'][next_stack]['content'][idx_offset+i] = idx

            # If last particle 
            if i == N-1:
                # Update main seed
                mcdc['seed'] = P['seed']

                # Reset current event stack size
                mcdc['stack_'][stack]['size'] = 0
                hostco['stack_size'][stack]   = 0

                # Update next event stack size
                mcdc['stack_'][next_stack]['size'] += N
                hostco['stack_size'][next_stack]   += N
        
    def wrap_branching(mcdc, hostco):
        # Stack index of the current event
        stack = mcdc['stack_idx'][event]

        # Stack size
        N = mcdc['stack_'][stack]['size']
        start, stride = kernel.get_idx()
        for i in range(start, N, stride):
            # Get particle index from stack
            idx = mcdc['stack_'][stack]['content'][i]

            # "Pop" particle from bank
            P = kernel.read_particle(mcdc['bank']['content'][idx])

            # Set RNG seed
            P['seed'] = mcdc['seed']
            kernel.rng_skip_ahead(i*mcdc['event_stride'][event], P, mcdc)

            # Perform event
            func(P, mcdc)
           
            # Update particle in the bank
            mcdc['bank']['content'][idx] = kernel.record_particle(P)

            # Get stack index of the next event
            next_event = P['event']
            next_stack = mcdc['stack_idx'][next_event]

            # Update secondaries parameter (for sync. later)
            mcdc['secondaries_stack'][i]                = next_stack
            mcdc['secondaries_counter'][i, next_stack] += 1
            
            # If last particle 
            if i == N-1:
                # Update main seed
                mcdc['seed'] = P['seed']

                # Reset current event stack size
                mcdc['stack_'][stack]['size'] = 0
                hostco['stack_size'][stack]   = 0

        # Launch exclusive scan algorithm [M. Harris 2007]
        #  to get secondaries global indices
        kernel.exscan(mcdc['secondaries_counter'], mcdc['secondaries_idx'], N)

        # Update all events stack based on the secondaries parameters
        for i in range(start, N, stride):
            # Get the stack and index
            next_stack = mcdc['secondaries_stack'][i]
            idx        = mcdc['secondaries_idx'][i, next_stack] + \
                         mcdc['stack_'][next_stack]['size']
                         
            mcdc['stack_'][next_stack]['content'][idx] = \
                    mcdc['stack_'][stack]['content'][i]

            # If last particle, update stack sizes
            if i == N-1:
                for j in range(mcdc['N_stack']):
                    # Get secondaries size
                    secondary_size = mcdc['secondaries_idx'][N-1,j] + \
                                     mcdc['secondaries_counter'][N-1,j]

                    # Update stack sizes
                    mcdc['stack_'][j]['size'] += secondary_size
                    hostco['stack_size'][j]   += secondary_size
            
            # Reset secondaries parameters
            for j in range(mcdc['N_stack']):
                mcdc['secondaries_counter'][i, j] = 0
                mcdc['secondaries_idx'][i, j]     = 0
    
    def wrap_naive(mcdc, hostco):
        # Stack index of the current event
        stack = mcdc['stack_idx'][event]

        # Stack size
        N = mcdc['stack_'][stack]['size']
        start, stride = kernel.get_idx()
        for i in range(start, N, stride):
            # Get particle index from stack
            idx = mcdc['stack_'][stack]['content'][i]

            # "Pop" particle from bank
            P = kernel.read_particle(mcdc['bank']['content'][idx])

            # Set RNG seed
            P['seed'] = mcdc['seed']
            kernel.rng_skip_ahead(i*mcdc['event_stride'][event], P, mcdc)
            
            func(P, mcdc)
           
            # Update particle in the bank
            mcdc['bank']['content'][idx] = kernel.record_particle(P)
               
            # Get stack index of the next event
            next_event = P['event']
            next_stack = mcdc['stack_idx'][next_event]

            # Update stack of the next event
            idx_next_stack = mcdc['stack_'][next_stack]['size']
            mcdc['stack_'][next_stack]['content'][idx_next_stack] = idx
            mcdc['stack_'][next_stack]['size'] += 1

            # If last particle 
            if i == N-1:
                # Update main seed
                mcdc['seed'] = P['seed']

                # Reset current event stack size
                mcdc['stack_'][stack]['size'] = 0
                
                # Update hostc controller
                for j in range(mcdc['N_stack']):
                    hostco['stack_size'][j] = mcdc['stack_'][j]['size']

    if naive:
        wrap = compiler(wrap_naive, target)
    elif branching:
        wrap = compiler(wrap_branching, target)
    else:
        wrap = compiler(wrap_streaming, target)

    if target == 'cpu':
        return wrap

    # GPU-Event-based zone below

    def hardware_wrap(mcdc, hostco):
        #event: int = mcdc['stack_idx'][event]
        N_block, N_thread = gpu_config(hostco['stack_size'][event], hostco)
        wrap[N_block, N_thread](mcdc, hostco)
    return hardware_wrap

# =============================================================================
# Utilities
# =============================================================================

def compiler(func, target):
    if target == 'cpu':
        return jit(func, nopython=True, nogil=True)#, parallel=True)
    elif target == 'cpus':
        return jit(func, nopython=True, nogil=True, parallel=True)
    else:
        return cuda.jit(func)

def parallel_compile(func):
    return jit(func, nopython=True, nogil=True, parallel=True)

def gpu_config(N, hostco):
    N_thread = hostco['N_thread']
    N_block = (N + (N_thread - 1)) // N_thread
    return N_block, N_thread
