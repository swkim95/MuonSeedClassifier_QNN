#!/bin/bash

python3 HLTXGB_r300d7.py -i "/home/wjun/phase2_won_fromRun3/data/*Spring23*.pkl" -v CMSSW_13_2_2_AllData_r300d7
python3 HLTXGB_r500d6.py -i "/home/wjun/phase2_won_fromRun3/data/*Spring23*.pkl" -v CMSSW_13_2_2_AllData_r500d6
python3 HLTXGB_r500d7.py -i "/home/wjun/phase2_won_fromRun3/data/*Spring23*.pkl" -v CMSSW_13_2_2_AllData_r500d7
python3 HLTXGB_r500d8.py -i "/home/wjun/phase2_won_fromRun3/data/*Spring23*.pkl" -v CMSSW_13_2_2_AllData_r500d8
python3 HLTXGB_r700d7.py -i "/home/wjun/phase2_won_fromRun3/data/*Spring23*.pkl" -v CMSSW_13_2_2_AllData_r700d7