import os
import numpy as np
import codecs
import json

class Number_Rectifier():
    '''
    这个纠正器，用来根据四官报告来对算饭检测出来的号码进行更正。
    '''

    def __init__(self,Json_File):
        '''
        载入基本信息
        Json_File : 算法运行完成后的写入的Json file
        '''
        #从文件中读取数据
        with codecs.open(Json_File, 'r', 'utf-8-sig') as f:
            predict_data = json.load(f)

        # 获取四官报告
        self.Four_officials_report =  predict_data['four_officials_report']
        self.action_data = predict_data['data'] # 总体数据
        self.action_index = 0 # 当前纠正到了哪个顺序
        self.len = len(self.action_data) # 数据的总长度
        self.stack_action_index = [] #同属于一个possession的所有action_index

        # 主客队的大名单，只有号码信息
        self.Team_Roster = {'Home':None, 'Away':None}
        # 主客队的
        self.Players = {'Home':None, 'Away':None}
        self.Players['Home'],self.Team_Roster['Home'] = self.generate_number_list(self.Four_officials_report['home']['players'])
        self.Players['Away'],self.Team_Roster['Away'] = self.generate_number_list(self.Four_officials_report['away']['players'])


    def generate_number_list(self, players):
        '''
        生成号码列表,和球员号码字典
        '''
        number_list = []
        player_dcit = {}
        for player in players:
            player_dcit[int(player['num'])] = player
            number_list.append(int(player['num']))
        return player_dcit, number_list

    def rectify(self):
        '''
        开始根据四官来纠正号码
        '''
        while self.action_index < self.len: # 确保索引号小于总长度

            self.stack_action_index = [self.action_index] # 记录每个possession开头的位置的index

            action_item = self.action_data[self.action_index]
            possession = action_item['possessChangeNums']
            TeamType = action_item['teamType']

            NumsArray = action_item['predicted_nums'] # 这个动作含有的所有预测的数字
            # 把同一个possession的所有的号码记录下来
            NumsArray.extend(self.collect_imgs_of_the_same_possesion(possession))

            preNum = self.number_select(NumsArray)
            for index in self.stack_action_index:
                # 更新结果
                self.write_predict_infomation(preNum, TeamType, index)

        return self.action_data

    def number_select(self,NumsArray):
        '''
        根据同一个Possession的所有的number，根据数量来预测
        数量最多的为最终的预测号码
        '''
        if len(NumsArray) > 1:
            # NumberArray range from 0 to 99.
            # We need to count how many times does each number appear!
            NumsArray = np.histogram(NumsArray, bins=100, range=(0, 100))[0]
            preNum = np.argmax(NumsArray)
            preNum_count = NumsArray[preNum]
            if np.where(NumsArray == preNum_count)[0].size > 1:
                # if there are more than one number have the maximun counts, then return -1
                # can sort by number classification scores.
                preNum = -1
        else:
            preNum = -1

        return preNum

    def write_predict_infomation(self,preNum,TeamType,current_index):
        '''
        把预测的信息写入 self.action_data
        '''
        self.action_data[current_index]['team'] = self.Four_officials_report[TeamType.lower()]['team_name']
        self.action_data[current_index]['teamID'] = self.Four_officials_report[TeamType.lower()]['team_id']
        self.action_data[current_index].pop('predicted_nums')  # 删除这个键值对

        if preNum not in self.Team_Roster[TeamType] and preNum != -1 :
            # print('{} not in Team {}'.format(preNum, TeamType))
            preNum = -1

        preNum_raw = self.action_data[current_index]['num'] #一次算法的输出结果，未综合所有结果
        # if str(preNum) != preNum_raw:
        #     print( 'action {}, preNum = {}, predict_num = {}'.format(current_index, preNum, preNum_raw))

        if preNum == -1 :
            self.action_data[current_index]['num'] = '-1'
            self.action_data[current_index]['name'] = None
            self.action_data[current_index]['PlayerID'] = None
        else:
            self.action_data[current_index]['num'] = self.Players[TeamType][preNum]['num']
            self.action_data[current_index]['name'] = self.Players[TeamType][preNum]['name']
            self.action_data[current_index]['PlayerID'] = self.Players[TeamType][preNum]['PlayerID']


    def collect_imgs_of_the_same_possesion(self, possession):
        '''
        通过递归的方式，把相同的possession的动作的号码收集起来
        有则继续递归，无责结束递归
        '''

        self.action_index += 1 # index + 1
        if self.action_index >= self.len:
            return []
        else:
            current_action_item = self.action_data[self.action_index]
            current_possession = current_action_item['possessChangeNums']
            if current_possession != possession:
                return []
            else:
                self.stack_action_index.append(self.action_index)
                current_NumsArray = current_action_item['predicted_nums']  # 这个动作含有的所有预测的数字
                current_NumsArray.extend(self.collect_imgs_of_the_same_possesion(current_possession))
                return current_NumsArray