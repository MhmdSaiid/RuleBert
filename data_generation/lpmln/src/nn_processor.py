import random
import clingo
import xml.etree.ElementTree as ET



def process(input):
    return nn_processor().d2asp(input)


class nn_processor(object):
    def __init__(self):
        self.config_dic = {}

    '''
    Check whether xml file follows the formate
    '''
    def _xml_chcker(self):
        pass



    def _xml_processor(self,input_file):
        root = ET.parse(input_file).getroot()
        if root.tag.lower() != "deeplpmln":
            print("Please provide correct Deep LPMLN configure file")
            exit(0)
        else:
            for m in root:
                if (m.attrib['name'] in self.config_dic):
                    print("Name of model:\"",m.attrib['name'],"\" exists")
                    exit(0)
                serialization = m.find("serialization").text
                path_to_model = m.find("path").text
                model_class = (m.find("class").attrib['name'],m.find("class").text)
                predicate_tuple=(m.find("predicate").text,m.find("predicate").attrib['arity'])
                input_list = []
                for input in m.findall("input"):
                    input_list.append((input.attrib['name'],input.attrib['map'],input.attrib['pos'],input.text))
                output_list = []
                for output in m.findall("output"):
                    output_list.append(output.text)
                cad_min =  m.find("min").text
                cad_max =  m.find("max").text
                self.config_dic[m.attrib['name']] = {
                    "serialization": serialization,
                    "path_to_model": path_to_model,
                    "model_class": model_class,
                    "predicate": predicate_tuple,
                    "input_list": input_list,
                    "output_list": output_list,
                    "cad_min": cad_min,
                    "cad_max": cad_max
                }


    def d2asp(self,input):
        self._xml_processor(input)
        g_lpmln=""
        for key,items in self.config_dic.items():
            out_dom = "dom_out_"+key
            in_dom = "input_"+key

            card_rule = items["cad_min"]+"{" + \
                        (items["predicate"][0] +'(I_'+ key+ ",D_" +key+ ')') + \
                        ":"+ (out_dom+'(D_' + key + ')')+\
                        "}" + items["cad_max"] + \
                        " :- " + (in_dom+'(I_' + key + ')') + ".\n"

            out_dom_rules =""
            in_dom_rules = in_dom + '('
            for out in items["output_list"]:
                out_dom_rules += str(clingo.parse_term(out_dom + '(' +out + ')')) + ".\n"


            for input in items["input_list"]:
                in_dom_rules +=input[0]+ ";"

            in_dom_rules = in_dom_rules[:-1] + ').\n'

            soft_atoms = self._weight_bounder(items)

            g_lpmln+=card_rule + out_dom_rules + in_dom_rules+ soft_atoms

        return g_lpmln



    def _weight_bounder(self,items):
        bounds = ""
        for input in items["input_list"]:
            ground_atoms = self._grounder(input,items["output_list"],items["predicate"][0])
            prediction_list = self._predict(items,input)

            for i in range(len(prediction_list)):
                bounds+= str(prediction_list[i]) + " " + str(ground_atoms[i]) + ".\n"
        return bounds




    def _grounder(self,input_list,output_list,predicate):
        ground_atoms = []
        for out in output_list:
            symbol = clingo.parse_term(predicate+'('+input_list[0] + "," + out+')')
            ground_atoms.append(symbol)
        return ground_atoms



    def _predict(self,items,input_list):
        if items["serialization"] == "pytorch":
            return self._torch_predictor(items,input_list)

        elif items["serialization"] == "tensor_flow":
            return self._tf_predictor(items,input_list)



    def _torch_predictor(self,items,input_list):
        import torch
        class_name, class_path = items["model_class"]
        module = __import__(class_path.replace(".py",""), fromlist=[class_name])
        mode_path = items["path_to_model"]
        model = getattr(module,class_name)()
        model.load_state_dict(torch.load(mode_path))
        model.eval()

        input_name_map = input_list[:2]
        input_func = input_list[3]
        input = getattr(model,input_func)(input_name_map)

        out = model(input)

        if len(out.detach().numpy()[0]) != len(items["output_list"]):
            print("Number of output from model: ",len(out.detach().numpy()[0]) ," does not equal to the expected output: ",len(items["output_list"]))
            exit(0)

        return (out.detach().numpy()[0])




    def _tf_predictor(self,items,input_list):
        pass



