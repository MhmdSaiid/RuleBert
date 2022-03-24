from lib import sdd as sddm
import pickle
import json
import os

# =========
# INTERFACE
# =========


def save_programs(compiled_programs, dirname):
    """
    Saves all the programs in compiled_programs to files in the directory dirname.
    The main file containing information on the meaning of other files is _main_file(dirname)
    :param compiled_programs: a list of TwoValuedCompilationResult objects
    :param dirname: a directory name to store the files in
    :return: nothing
    """
    dirname = os.path.abspath(dirname)
    if os.path.isfile(dirname):
        raise Exception("Input of --save-state option should be a directory, not a file")
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    stored_programs = []
    for i in range(0, len(stored_programs)+1):
        program = compiled_programs[i]
        filename = os.path.join(dirname, "program_" + str(i))
        stored_programs.append(filename)
        _save_program_to_file(program, filename)

    with open(_main_file(dirname), 'w') as outfile:
        json.dump(stored_programs, outfile)


def load_programs(dirname):
    """
    Inverse operation of the save_programs method: Creates a list of  TwoValuedCompilationResults by reading in the
    contents of the directory dirname, more specifically reading all programs refered to in _main_file(dirname)
    :param dirname: a directory containing at least one file _main_file(dirname)
    :return: a list of TwoValuedCompilationResults
    """
    dirname = os.path.abspath(dirname)
    if not os.path.isdir(dirname):
        raise Exception("Input of --load-state option should be an existing directory")
    mainfile = _main_file(dirname)
    if not os.path.isfile(mainfile):
        raise Exception("Could not find the file " + mainfile)

    with open(mainfile) as data_file:
        stored_programs = json.load(data_file)
    compiled_programs = []
    for stored_program in stored_programs:
        compiled_programs.append(_load_program_from_file(stored_program))

    return compiled_programs

# =========
# INTERNALS
# =========


def _load_program_from_file(filename):
    vtree = sddm.sdd_vtree_read(_vtree_name(filename))
    manager = sddm.sdd_manager_new(vtree)

    constraint = sddm.sdd_read(_constraint_name(filename), manager)
    entire_prog = sddm.sdd_read(_entireprog_assdd_name(filename), manager)

    with open(_atom2num_name(filename), "rb") as data_file:
        atom2num = pickle.load(data_file)
    with open(_program_name(filename), "rb") as data_file:
        program = pickle.load(data_file)
    with open(_atom2sdd_name(filename), "rb") as data_file:
        atom2sddfile = pickle.load(data_file)
    atom2sdd = {}
    for atom, file in atom2sddfile.items():
        atom2sdd[atom] = sddm.sdd_read(file, manager)

    from src import compile
    result = compile.TwoValuedCompiledProgram(atom2sdd,atom2num, manager, program)
    result.set_entire_program_as_sdd(entire_prog)
    result.set_constraint(constraint)
    return result


def _save_program_to_file(program, filename):
    """
    Saves a TwoValuedCompiledProgram to a file
    :param program: the TwoValuedCompiledProgram to save
    :param filename: the name of the file to save it to
    :return: nothing
    """
    manager = program.manager
    vtree = sddm.sdd_manager_vtree(manager)
    sddm.sdd_vtree_save(_vtree_name(filename), vtree)

    constraint = program.constraint
    sddm.sdd_save(_constraint_name(filename), constraint)

    entire_prog = program.get_entire_program_as_sdd()
    sddm.sdd_save(_entireprog_assdd_name(filename), entire_prog)

    with open(_atom2num_name(filename), 'wb') as outfile:
        pickle.dump(program.atom2sddnum, outfile)
    with open(_program_name(filename), 'wb') as outfile:
        pickle.dump(program.program, outfile)
    atom2sddfile = {}
    for atom, sdd in program.atom2sdd.items():
        sddfile = _sdd_name(filename)
        sddm.sdd_save(sddfile, sdd)
        atom2sddfile[atom] = sddfile
    with open(_atom2sdd_name(filename), 'wb') as outfile:
        pickle.dump(atom2sddfile, outfile)


def _main_file(dirname):
    return os.path.join(dirname, 'all_programs.json')


def _vtree_name(filename):
    return filename + ".vtree"


def _constraint_name(filename):
    return filename + ".constraint"


def _entireprog_assdd_name(filename):
    return filename + ".entireprog"


def _atom2num_name(filename):
    return filename + ".atom2num"


def _atom2sdd_name(filename):
    return filename + ".atom2sdd"


def _program_name(filename):
    return filename + ".program"


_counter = 0


def _get_unique_id():
    global _counter
    _counter += 1
    return _counter


def _sdd_name(filename):
    return filename + ".sdd." + str(_get_unique_id())
