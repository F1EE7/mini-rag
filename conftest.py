"""让 pytest 能找到仓库根目录下以数字开头的模块（01_chunking 等）。"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
