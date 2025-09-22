import ast

targets = {
    "material": [
        ("nuclide_densities", 1),
        ("mgxs_speed", 1),
        ("mgxs_decay_rate", 1),
        ("mgxs_capture", 1),
        ("mgxs_scatter", 1),
        ("mgxs_fission", 1),
        ("mgxs_total", 1),
        ("mgxs_nu_s", 1),
        ("mgxs_nu_p", 1),
        ("mgxs_nu_d", 2, "J"),
        ("mgxs_nu_d_total", 1),
        ("mgxs_nu_f", 1),
        ("mgxs_chi_s", 2, "G"),
        ("mgxs_chi_p", 2, "G"),
        ("mgxs_chi_d", 2, "G"),
    ],
    "nuclide": [
        ("xs_energy_grid", 1),
        ("total_xs", 1),
        ("reaction_type", 1),
        ("reaction_index", 1),
    ],
    "settings": [
        ("census_time", 1),
    ],
    "reaction": [
        ("xs", 1),
    ],
    "neutron_fission": [
        ("delayed_yield_type", 1),
        ("delayed_yield_index", 1),
        ("delayed_spectrum_type", 1),
        ("delayed_spectrum_index", 1),
        ("delayed_decay_rates", 1),
    ],
    "table": [
        ("x", 1),
        ("y", 1),
    ],
    "polynomial": [
        ("coefficients", 1),
    ],
    "multipdf": [
        ("grid", 1),
        ("offset", 1),
        ("value", 1),
        ("pdf", 1),
        ("cdf", 1),
    ],
    "cell": [
        ("region_RPN_tokens", 1),
    ],
    "surface": [
        ("move_time_grid", 1),
        ("move_translations", 2, 3),
        ("move_velocities", 2, 3),
    ],
}


def getter_1d_length(object_name, attribute_name):
    text = f"@njit\n"
    text += f"def {attribute_name}_length({object_name}):\n"
    text += f'    return int({object_name}["{attribute_name}_length"])\n\n\n'
    return text


def getter_1d_element(object_name, attribute_name):
    text = f"@njit\n"
    text += f"def {attribute_name}(index, {object_name}, data):\n"
    text += f'    offset = {object_name}["{attribute_name}_offset"]\n'
    text += f"    return data[offset + index]\n\n\n"
    return text


def getter_1d_all(object_name, attribute_name):
    text = f"@njit\n"
    text += f"def {attribute_name}_all({object_name}, data):\n"
    text += f'    start = {object_name}["{attribute_name}_offset"]\n'
    text += f'    end = start + {object_name}["{attribute_name}_length"]\n'
    text += f"    return data[start:end]\n\n\n"
    return text


def getter_chunk(object_name, attribute_name):
    text = f"@njit\n"
    text += f"def {attribute_name}_chunk(start, size, {object_name}, data):\n"
    text += f'    start += {object_name}["{attribute_name}_offset"]\n'
    text += f"    end = start + size\n"
    text += f"    return data[start:end]\n\n\n"
    return text


def getter_2d_element(object_name, attribute_name, stride):
    text = f"@njit\n"
    text += f"def {attribute_name}(index_1, index_2, {object_name}, data):\n"
    text += f'    offset = {object_name}["{attribute_name}_offset"]\n'
    if isinstance(stride, str):
        text += f'    stride = {object_name}["{stride}"]\n'
    else:
        text += f'    stride = {stride}\n'
    text += f"    return data[offset + index_1 * stride + index_2]\n\n\n"
    return text


def getter_2d_vector(object_name, attribute_name, stride):
    text = f"@njit\n"
    text += f"def {attribute_name}_vector(index_1, {object_name}, data):\n"
    text += f'    offset = {object_name}["{attribute_name}_offset"]\n'
    if isinstance(stride, str):
        text += f'    stride = {object_name}["{stride}"]\n'
    else:
        text += f'    stride = {stride}\n'
    text += f"    start = offset + index_1 * stride\n"
    text += f"    end = start + stride\n"
    text += f"    return data[start:end]\n\n\n"
    return text


for object_name in targets:
    with open(f"mcdc_get/{object_name}.py", "w") as f:
        text = "from numba import njit\n\n\n"
        for attribute in targets[object_name]:
            attribute_name = attribute[0]
            attribute_dim = attribute[1]
            if attribute_dim == 1:
                text += getter_1d_length(object_name, attribute_name)
                text += getter_1d_all(object_name, attribute_name)
                text += getter_1d_element(object_name, attribute_name)
            if attribute_dim == 2:
                stride = attribute[2]
                text += getter_2d_vector(object_name, attribute_name, stride)
                text += getter_2d_element(object_name, attribute_name, stride)
            text += getter_chunk(object_name, attribute_name)
        f.write(text[:-2])


with open(f"mcdc_get/__init__.py", "w") as f:
    text = ""
    for object_name in targets:
        text += f"import mcdc.mcdc_get.{object_name} as {object_name}\n"
    f.write(text[:-1])
