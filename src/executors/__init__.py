

import importlib
import os

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