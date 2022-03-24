# numpy, ply, ipdb, sympy
#
#
# clingo, sdd,

import os

current_path = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
packages_to_install = ["numpy","sympy","ply","Cython"]


def install(packages):
    import importlib
    try:
        from pip._internal import main as pip
    except ImportError:
        print("Cannot import pip module. Please install pip first.")
        exit(0)
    try:
        import clingo
    except ImportError:
        print("Cannot import Clingo module. Please install Clingo, and enable Clingo Python Interface.")
        exit(0)
    try:
        from lib.sdd_source import setup
        print("Installing SDD library")
        setup.build_sdd()
        print("Finished installing SDD library")
    except:
        print("error during installing SDD library. ")
        exit(0)


    global package_dir
    global current_path
    try:
        print("Installing PySDD library")
        pip(['install', os.path.join(package_dir, 'pysdd')])
        # For unknown reason, in order to import pysdd.sdd, the pysdd package has to be installed twice
        pip(['install', os.path.join(package_dir, 'pysdd')])
        print("Finished installing PySDD library")
    except:
        print("error during installing PySDD library")
        exit(0)

    try:
        print("Installing all other dependencies")
        for package in packages:
            import subprocess
            subprocess.check_call(['pip', 'install', package])
        print("Finished installing all other dependencies")
    except:
        print("error during installing dependencies.")
        exit(0)

    try:
        import sys
        with open(current_path + "/lpmln_infer.py", 'r+') as file:
            originalContent = file.read()
            file.seek(0, 0)  # Move the cursor to top line
            file.write("#! " + sys.executable + '\n')  # Add a new blank line
            file.write(originalContent)

        with open(current_path + "/lpmln_dec.py", 'r+') as file:
            originalContent = file.read()
            file.seek(0, 0)  # Move the cursor to top line
            file.write("#! " + sys.executable + '\n')  # Add a new blank line
            file.write(originalContent)
        with open(current_path + "/lpmln_learn.py", 'r+') as file:
            originalContent = file.read()
            file.seek(0, 0)  # Move the cursor to top line
            file.write("#! " + sys.executable + '\n')  # Add a new blank line
            file.write(originalContent)

    except:
        print("error during writing entry file path")
        exit(0)


    print("All Done!")

if __name__ == '__main__':
    install(packages_to_install)

