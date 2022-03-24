import math

def prob_to_weight(path):
    with open(path) as f:
        content = f.readlines()

        total = ""
        for c in content:
            l = c.split()
            line = ""
            if len(l) > 0:
                if l[0].replace('.', '', 1).isdigit():
                    if eval(l[0]) < 1 and eval(l[0]) > 0:
                        line = str(math.log((eval(l[0]) / (1 - eval(l[0]))))) + " " + ''.join(l[1:])
                        total += line
                        total += '\n'
                        continue
            total += c

    with open(path+".ptw", "w") as fw:
        fw.write(total)


def weight_to_prob(path):
    with open(path) as f:
        content = f.readlines()

        total = ""
        for c in content:
            l = c.split()
            line = ""
            if len(l) > 0:
                if l[0].replace('.', '', 1).lstrip('-+').isdigit():
                    if eval(l[0]) < 0 or eval(l[0]) > 1:
                        print(eval(l[0]))
                        line = str(math.exp(eval(l[0]))/(1+math.exp(eval(l[0])))) + " " + ''.join(l[1:])
                        total += line
                        total += '\n'
                        continue
            total += c

    with open(path+".wtp", "w") as fw:
        fw.write(total)


weight_to_prob("test_saeed/example_2.txt")

