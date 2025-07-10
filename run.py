#!/usr/bin/env python3
import os
import sys

# Add the project root directory to the PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from opencompass.cli.main import main

if __name__ == '__main__':
    main()
