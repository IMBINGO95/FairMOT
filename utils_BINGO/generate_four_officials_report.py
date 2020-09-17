import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import json
import codecs
import pandas as pd
import csv

if __name__ == '__main__':
    for index in range(32,33):
        print(index)
        json_file = '/datanew/hwb/data/Football/SoftWare/{}/{}.json'.format(index,index)
        with codecs.open(json_file, 'r', 'utf-8-sig') as f:
            action_datas = json.load(f)

        data = action_datas['data']
        Team = {}
        for item in data:
            team = item['team']
            name = item['name']
            num = item['num']
            # name = team
            if team not in Team:
                Team[team] = {}

            if num not in Team[team]:
                Team[team][num] = name
            else:
                continue

        with open('/datanew/hwb/data/Football/SoftWare/四官报告/{}_siguan.csv'.format(index),'w',encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            for index, team in enumerate(Team.keys()):
                writer.writerow([team])

            for index, team in enumerate(Team.keys()):
                for [num,name] in Team[team].items():
                    writer.writerow([index+1,num,name])
                    # f.writelines('{}\t{}\t{}\n'.format(index+1,num, name))
                    print(index+1 ,num,name)
    print()