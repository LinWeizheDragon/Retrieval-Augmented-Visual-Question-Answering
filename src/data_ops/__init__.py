

import importlib
import os
import sys

def import_files(fl_dir):
    for file in os.listdir(fl_dir):
        path = os.path.join(fl_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            file_name = file[: file.find(".py")] if file.endswith(".py") else file
            this_pkg_name = os.path.dirname(__file__).split('/')[-1]
            importlib.import_module("."+file_name, package=this_pkg_name)

# automatically import any Python files in the data_processing/ directory
fl_dir = os.path.dirname(__file__)
import_files(fl_dir)

path = os.path.dirname(os.path.abspath(__file__))

for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
    mod = __import__('.'.join([__name__, py]), fromlist=[py])
    classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
    for cls in classes:
        setattr(sys.modules[__name__], cls.__name__, cls)