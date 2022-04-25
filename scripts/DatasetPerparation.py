"""
Main script that calls every script that should run for perparing dataset.
"""

import os
import sys
import datetime

output_size = (320, 180)


if __name__ == '__main__':
    import DeFramePicking
    import DegradedModule
    import VerticalAssembly
    import SamplePicking
    import CropsExtraction

    timeStamp = datetime.datetime.now()

    print(f'INFO: Running script DeFramePicking...')
    DeFramePicking.main(sys.argv)
    os.system('cls||clear')

    # print(f'INFO: Running script CropsExtraction...')
    # CropsExtraction.main(sys.argv)
    # os.system('cls||clear')

    print(f'INFO: Running script degraded_module...')
    DegradedModule.main(sys.argv)
    os.system('cls||clear')

    # print(f'INFO: Running script VerticalAssembly...')
    # VerticalAssembly.main(sys.argv)
    # os.system('cls||clear')

    print(f'INFO: Running script SamplePicking...')
    SamplePicking.main(sys.argv)
    os.system('cls||clear')

    timeDiffer = datetime.datetime.now() - timeStamp
    print(f'INFO: Finished dataset preparation, total time taken: {timeDiffer}.')

