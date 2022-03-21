import sys

output_size = (320, 180)


if __name__ == '__main__':
    import DeFramePicking
    import degraded_module

    import VerticalAssembly
    import SamplePicking

    DeFramePicking.main(sys.argv)
    degraded_module.main(sys.argv)
    VerticalAssembly.main(sys.argv)
    SamplePicking.main(sys.argv)

