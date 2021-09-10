import os
import sys

# appending root of project folder to python path for convenient imports
dirname = os.path.dirname(__file__)
abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(1, abspath)


# testing imports
if __name__ == "__main__":
    from eos.relativistic_ISCT import Relativistic_ISCT

    print(type(Relativistic_ISCT()))
