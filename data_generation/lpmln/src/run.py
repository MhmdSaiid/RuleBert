import os
import subprocess
from _io import StringIO
from src import save
import time
from lib import sdd as sddm
import json


'''def run(in_files, args):
    if args.install:
        from lib.sdd_source import setup
        print("Installing SDD library")
        setup.build_sdd()
        print("Finished installing SDD library")
        return
    # Put all imports after installation code to avoid getting errors that program is not yet installed
    from src import save
    from src import inference  

    if args.loaddir is not None:
        if args.verbosity > 0:
            print(">>>Reading previously compiled programs from file")
        compiled_programs = save.load_programs(args.loaddir)
        if args.verbosity > 0:
            print(">>>Finished reading previously compiled programs from file")
    else:
        inference.check_nb_inputs(args.inference, in_files)
        compiled_programs = _parse_and_compile(in_files, args)

    if args.savedir is not None:
        save.save_programs(compiled_programs, args.savedir)
        return

    if args.verbosity > 0:
        print(">>>Starting inference")

    inference.do_inference(args, compiled_programs)

    if args.verbosity > 0:
        print(">>>Inference done")'''


from src import inference

class run:
    def __init__(self,content, args):

        build_sdd_start = time.time()
        bytesContent = str.encode(content)

        fn_gringo = os.path.join(os.path.dirname(__file__), '../binSupport/gringo')
        p = subprocess.Popen([fn_gringo], shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        p.stdin.write(bytesContent)
        gringo_out = p.communicate()[0]
        p.stdin.close()
        p.stdout.close()

        fn_lp2normal = os.path.join(os.path.dirname(__file__), '../binSupport/lp2normal')
        p = subprocess.Popen([fn_lp2normal], shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        p.stdin.write(gringo_out)
        lp2normal_out = p.communicate()[0]
        p.stdin.close()
        p.stdout.close()
        self.args = args
        self.compiled_programs = self._parse_and_compile(StringIO(lp2normal_out.decode()))

        if self.args.verbosity > 4:
            print("Time for building SDD: ", time.time() - build_sdd_start)

    def get_info_4_learn(self,name=None):
        mapping={}
        for key, item in self.compiled_programs[0].program.symbolTable.items():
            for atom, sdd_num in self.compiled_programs[0].atom2sddnum.items():
                if key == atom.atomnumber:
                    mapping[item] = sdd_num

        manager = self.compiled_programs[0].manager
        sdd = self.compiled_programs[0].get_entire_program_as_sdd()
        vtree = sddm.sdd_manager_vtree(manager)

        if name is not None:
            sddm.sdd_vtree_save(name+".vtree", vtree)
            sddm.sdd_save(name+".sdd", sdd)

            open(name + ".map", "w").close()
            with open(name + ".map", "w") as f:
                json.dump(mapping, f)
            return name
        else:
            sddm.sdd_vtree_save("lpmln_bart_sdd.vtree", vtree)
            sddm.sdd_save("lpmln_bart_sdd.sdd", sdd)
            open("lpmln_bart_sdd.map", "w").close()
            with open("lpmln_bart_sdd.map", "w") as f:
                json.dump(mapping, f)
            return "lpmln_bart_sdd"







    def infer(self,query, evidence):
        infer_time_start = time.time()
        if self.args.verbosity>5:
            print(">>>Start doing inference")
            print(query)
            print(evidence)


        result = inference.do_inference(self.compiled_programs, query, evidence)

        if self.args.verbosity > 4:
            print("Time for doing inference on SDD: ", time.time() - infer_time_start)
        return result








    def _parse_and_compile(self,in_file):
        import src.parse as parse
        import src.compileProgramToSDD as Convert
        import src.compile as compiler
        from src import simplify
        compiled_programs = []
        if self.args.verbosity > 0:
            print(">>>Start parsing program")

        parsed_program = parse.parse(in_file)
        if self.args.verbosity > 0:
            print(">>>Done parsing program")
        if self.args.verbosity > 4:
            print("Parsed logic program:")
            print(parsed_program)

        if self.args.verbosity > 15:
            print("Warning: verbosity is greater than 15")
            print("    This means that all IDs in logic programs will be replaced by their string-value in "
                  "symboltables. This might affect the behaviour of lp2sdd. If you wish to avoid this, make "
                  "sure your verbosity is lower than 15.")
            parsed_program = simplify.simplify(parsed_program)
            print(">>>Done simplifying program")
            print("Simplified logic program:")
            print(parsed_program)

        if self.args.verbosity > 0:
            print(">>>Start compiling program")

        compilation_result = Convert.convert_program_to_sdd(self.args, parsed_program)
        compiled_programs.append(compilation_result)

        if self.args.verbosity > 0:
            print(">>>Bottom up part of compilation done. Compiling entire program as one SDD")


        sdd = compilation_result.get_entire_program_as_sdd()
        if self.args.verbosity > 0:
            print(">>>Size of the compiled SDD: ")
            print(compiler.smart_size(sdd))
        if self.args.verbosity > 0:
            print(">>>Done compiling program.")

        if self.args.verbosity > 0:
            print(">>>Done compiling all programs.")
        return compiled_programs