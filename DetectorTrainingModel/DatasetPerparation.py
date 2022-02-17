import sys

import degraded_module
import VerticalAssembly
import FramesPicking


if __name__ == '__main__':
    degraded_module.main(sys.argv)
    VerticalAssembly.main(sys.argv)
    FramesPicking.main(sys.argv)

