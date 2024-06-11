from typing import List, TextIO

import numpy as np
import numpy.typing as npt


def save_iq_data(data: npt.NDArray[np.complex64], windows: int, window_length: int, file: str):
    output_length = windows * window_length

    data_offset = 0

    indices = np.arange(0, len(data), step=4)
    data = data[indices]
    
    data = data[data_offset:data_offset + output_length]

    max_value = np.max(data).real

    # Scale the data to be between -128 and 128 so it fits in a signed 8-bit integer
    data_scale_factor: float = 128 / max_value
    data = data * data_scale_factor

    real_list: List[str] = []
    imag_list: List[str] = []

    def format_num(n) -> str:
        # return np.format_float_positional(np.float16(n))
        return str(np.int8(n))

    for n in data:
        real_list.append(format_num(n.real))
        imag_list.append(format_num(n.imag))

    with open(file, "w") as f:
        f.write("#include <avr/pgmspace.h>\n")

        write_variable(real_list, f, "real", "int8_t")
        write_variable(imag_list, f, "imag", "int8_t")


def write_variable(x: List, f: TextIO, name: str, type: str):
    f.write(f"const static {type} {name}[] PROGMEM = " + "{\n")

    for (i, n) in enumerate(x):
        f.write(f"    {n}")

        if i < len(x) - 1:
            f.write(",\n")

    f.write("\n};\n\n")
