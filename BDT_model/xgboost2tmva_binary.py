import re
import xml.etree.cElementTree as ET
regex_float_pattern = r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'

def build_tree(xgtree, base_xml_element, var_indices):
    parent_element_dict = {'0':base_xml_element}
    pos_dict = {'0':'s'}
    for line in xgtree.split('\n'):
        if not line: continue
        if ':leaf=' in line:
            #leaf node
            result = re.match(r'(\t*)(\d+):leaf=({0})$'.format(regex_float_pattern), line)
            if not result:
                print line
            depth = result.group(1).count('\t')
            inode = result.group(2)
            res = result.group(3)
            node_elementTree = ET.SubElement(parent_element_dict[inode], "Node", pos=str(pos_dict[inode]),
                                             depth=str(depth), NCoef="0", IVar="-1", Cut="0.0e+00", cType="1", res=str(res), rms="0.0e+00", purity="0.0e+00", nType="-99")
        else:
            #\t\t3:[var_topcand_mass<138.19] yes=7,no=8,missing=7
            result = re.match(r'(\t*)([0-9]+):\[(?P<var>.+)<(?P<cut>{0})\]\syes=(?P<yes>\d+),no=(?P<no>\d+)'.format(regex_float_pattern),line)
            if not result:
                print line
            depth = result.group(1).count('\t')
            inode = result.group(2)
            var = result.group('var')
            cut = result.group('cut')
            lnode = result.group('yes')
            rnode = result.group('no')
            pos_dict[lnode] = 'l'
            pos_dict[rnode] = 'r'
            node_elementTree = ET.SubElement(parent_element_dict[inode], "Node", pos=str(pos_dict[inode]),
                                             depth=str(depth), NCoef="0", IVar=str(var_indices[var]), Cut=str(cut),
                                             cType="1", res="0.0e+00", rms="0.0e+00", purity="0.0e+00", nType="0")
            parent_element_dict[lnode] = node_elementTree
            parent_element_dict[rnode] = node_elementTree

def convert_model(model, input_variables, output_xml):
    NTrees = len(model)
    var_list = input_variables
    var_indices = {}
    
    # <MethodSetup>
    MethodSetup = ET.Element("MethodSetup", Method="BDT::BDT")

    # <Variables>
    Variables = ET.SubElement(MethodSetup, "Variables", NVar=str(len(var_list)))
    for ind, val in enumerate(var_list):
        name = val[0]
        var_type = val[1]
        var_indices[name] = ind
        Variable = ET.SubElement(Variables, "Variable", VarIndex=str(ind), Type=val[1], 
            Expression=name, Label=name, Title=name, Unit="", Internal=name, 
            Min="0.0e+00", Max="0.0e+00")

    # <GeneralInfo>
    GeneralInfo = ET.SubElement(MethodSetup, "GeneralInfo")
    Info_Creator = ET.SubElement(GeneralInfo, "Info", name="Creator", value="xgboost2TMVA")
    Info_AnalysisType = ET.SubElement(GeneralInfo, "Info", name="AnalysisType", value="Classification")

    # <Options>
    Options = ET.SubElement(MethodSetup, "Options")
    Option_NodePurityLimit = ET.SubElement(Options, "Option", name="NodePurityLimit", modified="No").text = "5.00e-01"
    Option_BoostType = ET.SubElement(Options, "Option", name="BoostType", modified="Yes").text = "Grad"
    
    # <Weights>
    Weights = ET.SubElement(MethodSetup, "Weights", NTrees=str(NTrees), AnalysisType="1")
    
    for itree in range(NTrees):
        BinaryTree = ET.SubElement(Weights, "BinaryTree", type="DecisionTree", boostWeight="1.0e+00", itree=str(itree))
        build_tree(model[itree], BinaryTree, var_indices)

    tree = ET.ElementTree(MethodSetup)
    tree.write(output_xml)
    # format it with 'xmllint --format'



"""
cd model
mkdir xml
mkdir data
python xgboost2tmva.py BDT_model.model n_variables
xmllint --format xml/BDT_model_0.xml > data/xgb_BDT_model_0.xml
xmllint --format xml/BDT_model_1.xml > data/xgb_BDT_model_1.xml
xmllint --format xml/BDT_model_2.xml > data/xgb_BDT_model_2.xml
xmllint --format xml/BDT_model_3.xml > data/xgb_BDT_model_3.xml
"""
if __name__ == '__main__':
    import sys
    import xgboost as xgb

    if len(sys.argv) < 2:
        print "python xgboost2tmva.py model_file n_variables"
        sys.exit()

    model_path = sys.argv[1]
    n_variables = sys.argv[2] if len(sys.argv) > 2 else 14
    n_variables = int(n_variables)

    bst = xgb.Booster()
    bst.load_model(model_path)
    model_all = bst.get_dump()

    
    model = []
    for l in (model_all):
            model.append(l)

    # model = [ [], [], [], [] ]
    # for i, l in enumerate(model_all):
    #     if i%4 == 0:
    #         model[0].append(l)
    #     elif i%4 == 1:
    #         model[1].append(l)
    #     elif i%4 == 2:
    #         model[2].append(l)
    #     elif i%4 == 3:
    #         model[3].append(l)

    _input_variables = [ ('f%d' % i , 'F') for i in range(n_variables) ]

    # for c in range(4):
    for c in range(1):
        output_name = 'xml/' + model_path.split("/")[-1].replace(".model", "_%d.xml"%c)
        # convert_model(model[c], input_variables=_input_variables, output_xml=output_name)
        convert_model(model, input_variables=_input_variables, output_xml=output_name)



