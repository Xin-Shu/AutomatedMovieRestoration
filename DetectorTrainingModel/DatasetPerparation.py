"""
Main script that calls every script that should run for perparing dataset.
"""

import os
import sys

output_size = (320, 180)


if __name__ == '__main__':
    import DeFramePicking
    import degraded_module
    import VerticalAssembly
    import SamplePicking

    print(f'INFO: Running script DeFramePicking...')
    DeFramePicking.main(sys.argv)
    os.system('cls||clear')

    print(f'INFO: Running script degraded_module...')
    degraded_module.main(sys.argv)
    os.system('cls||clear')

    print(f'INFO: Running script VerticalAssembly...')
    VerticalAssembly.main(sys.argv)
    os.system('cls||clear')

    print(f'INFO: Running script SamplePicking...')
    SamplePicking.main(sys.argv)
    os.system('cls||clear')

    print(f'INFO: Finished dataset preparation!!!')

