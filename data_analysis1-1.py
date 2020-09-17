# coding=utf-8
# /usr/bin/env python
"""
Author: 12345ayu
Email: 19858118660@163.com
"""
import pandas as pd
import os.path as osp
import os
import re
import json
import codecs
import numpy as np
import networkx as nx
import logging
import math

# 这段代码修改了x和y的方向

# 超参数的设定
# 需要手动修改部分
# 涛哥定义的比赛场次
game = 25
# 判断四官报告中主队是否是上半场先触球的球队
# 如果是 team_judge = True
# 如果不是team_judge = False
team_judge = True
# 判断四官定义的主队上半场进攻方向
# 如果是从左往右攻 home_left = True
# 如果是从右往左攻 home_left = False
home_left = True
# 判断超参数是否已经修改
super_parameter = input('是否已经修改完超参数(yes/no)：')
assert super_parameter == 'yes', '请先修改超参数设定'

# 输入原始数据文件名 json文件路径
file_name_first = input('请输入上半场比赛数据文件名：')
file_name_second = input('请输入下半场比赛数据文件名：')
# 如果文件路径两端额外增加了引号，则去除
if file_name_first[0] in ['\'', '\"']:
    l = len(file_name_first)
    file_name_first = file_name_first[1:l-1]
if file_name_second[0] in ['\'', '\"']:
    l = len(file_name_second)
    file_name_second = file_name_second[1:l-1]

# 指定一系列文件路径
analysis_output_path = osp.dirname(file_name_first) + '/分析结果/'
log_path = osp.join(osp.dirname(file_name_first), 'run_data.log')
base = osp.basename(file_name_first)
[fname, fename] = os.path.splitext(base)
# 数据输出文件名
if not osp.exists(analysis_output_path):
    os.makedirs(analysis_output_path)
outputFilePath = osp.join(analysis_output_path, '分析数据.xlsx')
if osp.exists(outputFilePath):
    os.remove(outputFilePath)
# 写运行日志
if osp.exists(log_path):
    os.remove(log_path)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(log_path, encoding='utf-8')
handler.setLevel(logging.INFO)
# 分别代表  运行时间、模块名称、日志级别、日志内容
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# 写日志 记录输出文件路径
logger.info('输出文件 outputFilePath: {}'.format(outputFilePath))

# 超参数的设定
# 主客队名称
TEAM_LIST = ['Home', 'Away']  # 两队名称
# 软件拉出来的数据是实际物理坐标
# 涛哥画板上有个坐标
# 要将实际物理坐标转换成画板上的坐标

# 实际场地的物理长宽
# 设置物理场地长度与宽度，单位：米
LENGTH_F = 104
WIDTH_F = 70

# 涛哥画板的坐标
# 设置场地图上左下角和右上角坐标，单位：像素
MIN_X = 24
MIN_Y = 29
MAX_X = 1178
MAX_Y = 771

# 读入上下半场json文件
dataFirst_raw = json.load(codecs.open(file_name_first, 'r', 'utf-8-sig'))
dataFirst_pro = dataFirst_raw['data']
dataFirst = pd.DataFrame(dataFirst_pro)
dataSecond_raw = json.load(codecs.open(file_name_second, 'r', 'utf-8-sig'))
dataSecond_pro = dataSecond_raw['data']
dataSecond = pd.DataFrame(dataSecond_pro)

# 一些保存数据的参数
playerNumList = [[], []]  # 球员号码
playerDict = [{}, {}]  # 球员索引
playerNumber = [0, 0]  # 两队球员人数
playerNameList = [{}, {}]  # 球员名字索引

# 提取两队球员名称
teamName = []
# 上/下半场，h=0/1
for h in range(2):
    if h == 0:
        data = dataFirst
    else:
        data = dataSecond
    for i in range(len(data)):
        # 添加队伍名称，上半场先出现的队伍为主队，在teamName的0序号
        if not (data.team[i] in teamName):
            teamName.append(data.team[i])
        # 提取队伍与球员名称
        team = int(data.team[i] != teamName[0])  # 主、客队转换为序号：主球队为0序号，客队为1序号
        playerNum = data.num[i]  # 读取队员号码/名称
        playerName = data.name[i]

        # 添加球员进入列表
        if playerNum not in playerNumList[team]:
            playerNumList[team].append(playerNum)
            playerNameList[team][playerNum] = playerName
            playerNumber[team] += 1

# 球员总数totalPlayerNumber
totalPlayerNumber = playerNumber[0] + playerNumber[1]

# 做排序，这样输出的时候就是按号码从小到大输出的

# for i in playerNumList[0]:
#     if not i:
#         playerNumList[0].remove(i)
# print(playerNumList)
playerNumList[0] = sorted(playerNumList[0], key=lambda i: int(re.match(r'(\d+)', i).group()))
playerNumList[1] = sorted(playerNumList[1], key=lambda i: int(re.match(r'(\d+)', i).group()))

for i in range(len(playerNumList)):
    for j in range(len(playerNumList[i])):
        playerDict[i][playerNumList[i][j]] = j
    # print(playerDict)

# 调整主客队顺序
if not team_judge:
    playerNumList.reverse()
    playerDict.reverse()
    playerNumber.reverse()
    teamName.reverse()
    playerNameList.reverse()

logger.info('playerNumList: {}'.format(playerNumList))
logger.info('playerDict: {}'.format(playerDict))
logger.info('playerNumber: {}'.format(playerNumber))
logger.info('teamName: {}'.format(teamName))
logger.info('totalPlayerNumber: {}'.format(totalPlayerNumber))
logger.info('playerNameList: {}'.format(playerNameList))


# 传接表、防守数据提取
def creatPassDefenseList():
    "传接表数据提取，并存为DataFrame形式：pass_list，写入输出数据文件"
    global passListFirst, passListSecond, defenseListFirst, defenseListSecond

    # 上/下半场 h=0/1
    for h in range(2):
        # 读入原始数据
        if h == 0:
            data = dataFirst
        else:
            data = dataSecond
        length_data = len(data)
        # 传接数据表变量
        pass_list = pd.DataFrame(
            columns=['Team', 'Chuan', 'Result1', 'X1', 'Y1', 'Time1', 'Jie', 'Result2', 'X2', 'Y2', 'Time2'])
        # 防守数据表变量
        defense_list = pd.DataFrame(
            columns=['Team', 'Fang', 'Result1', 'X1', 'Y1', 'Time1', 'Gong', 'Result2', 'X2', 'Y2', 'Time2'])

        for i in range(length_data):
            # 提取动作一信息
            team = int(data.team[i] != teamName[0])  # 主、客队转换为序号：主球队为0号，客队为1号
            # print(team)
            # print(data.num[i])
            # print(data.action_time[i])
            player1 = playerDict[team][data.num[i]]  # 球员1序号，每队按照球员出现次序编号
            # XY方向互换
            x1 = data.world_x[i]
            y1 = data.world_y[i]
            t1 = data.action_time[i]

            # 判断是否为传球
            if 'Pass' in data.action[i]:
                # 初始预设传接球结果为失败
                result1 = 0
                player2 = player1
                x2 = x1
                y2 = y1
                t2 = t1
                result2 = 0
                # 判断如果是传球成功，进一步提取动作二信息
                if data.action[i] == 'PassSucceeded' and i < length_data-1:
                    # 判断下一动作是否为同队队友接球
                    if data.team[i] == data.team[i + 1] and 'Catch' in data.action[i + 1]:
                        # 此时才能确定传球为成功，确定第一行为结果为成功（1），并提取第二行为信息
                        result1 = 1
                        player2 = playerDict[team][data.num[i+1]]
                        # XY方向互换
                        x2 = data.world_x[i+1]
                        y2 = data.world_y[i+1]
                        t2 = data.action_time[i+1]
                        # 判断接球是否成功
                        if data.action[i+1] == 'CatchSucceeded':
                            result2 = 1
                        else:
                            result2 = 0
                # 传球结果存入pass_list变量
                chuanjie_row = pd.DataFrame([[team, player1, result1, x1, y1, t1, player2, result2, x2, y2, t2]],
                                            columns=['Team', 'Chuan', 'Result1', 'X1', 'Y1', 'Time1', 'Jie', 'Result2',
                                                     'X2', 'Y2', 'Time2'])
                pass_list = pass_list.append(chuanjie_row, ignore_index=True)

            # 判断是否为防守行为，防守行为必须有前序行为（i>0）
            elif i > 0 and (data.action[i] in ['Break', 'Steal', 'Intercept', 'Save', 'Block']):
                result1 = 'yes'
                # 提取上一行为信息，判断防守行为的上一行为人是对手还是本方球员。如果是本方球员，说明防守未解决问题，在Result2项中添加‘-’
                j = i - 1
                team2 = int(data.team[j] != teamName[0])
                player2 = playerDict[team2][data.num[j]]
                # XY方向互换
                x2 = data.world_x[j]
                y2 = data.world_y[j]
                t2 = data.action_time[j]
                # 如果上一行为人是本方球员，则Result2为上一行为的结果添加‘-’
                if team2 == team:
                    # 如果上一行为人是本方的防守行为，则在上一防守记录后面添加减号；
                    if data.action[j] in ['Break', 'Steal', 'Intercept', 'Save', 'Block']:
                        result2 = data.action[j] + '-'
                    # 如果上一行为是本方的进攻行为，则将上一行为记为失误
                    else:
                        result2 = 'fault-'
                else:
                    # 如果上一行为人是对方的进攻行为，则判定对方进攻球员的进攻是否失败
                    result2 = 'no'
                # 防守结果存入defense_list变量
                fangshou_row = pd.DataFrame([[team, player1, result1, x1, y1, t1, player2, result2, x2, y2, t2]],
                                            columns=['Team', 'Fang', 'Result1', 'X1', 'Y1', 'Time1', 'Gong', 'Result2',
                                                     'X2', 'Y2', 'Time2'])
                defense_list = defense_list.append(fangshou_row, ignore_index=True)

        if h == 0:
            passListFirst = pass_list
            defenseListFirst = defense_list
        else:
            passListSecond = pass_list
            defenseListSecond = defense_list
    # return passListFirst, passListSecond, defenseListFirst, defenseListSecond


def creatPassNet():
    "从传接表中生成传接图数据，并存入：ChuanJieNet"
    global passNetFirst, passNetSecond

    # 上/下半场 h=0/1
    for h in range(2):
        # 读入传接数据变量
        if h == 0:
            chuanjie_chart = passListFirst
        else:
            chuanjie_chart = passListSecond
        # 传接网络，函数内部变量
        # 统计球员之间传接关系
        pass_net = [np.zeros([playerNumber[0], playerNumber[0]]), np.zeros([playerNumber[1], playerNumber[1]])]
        for i in range(len(chuanjie_chart)):
            if chuanjie_chart.Result1[i] * chuanjie_chart.Result2[i] == 1:
                pass_net[chuanjie_chart.Team[i]][chuanjie_chart.Chuan[i], chuanjie_chart.Jie[i]] += 1

        if h == 0:
            passNetFirst = pass_net
        else:
            passNetSecond = pass_net
    logger.info('passNetFirst: {}'.format(passNetFirst))
    logger.info('passNetSecond: {}'.format(passNetSecond))
    # return passNetFirst, passNetSecond


# 基础统计数据生成
def creatStatData():
    "从Data、传接表中生成传接统计数据，并存入DataFrame形式:Stat"
    # 全局变量声明
    global statDataFirst, statDataSecond

    # 上/下半场 h=0/1
    for h in range(2):
        # 读入传接数据变量
        if h == 0:
            chuanjie_chart = passListFirst
            data = dataFirst
        else:
            chuanjie_chart = passListSecond
            data = dataSecond
        # 定义统计数据内部变量
        stat = pd.DataFrame(np.zeros([totalPlayerNumber, 7]),
                            columns=['Team', 'Player', 'Chuan', 'ChuanCheng', 'ChuanJie', 'JieQiu', 'JieCheng'])

        # 统计传球、传成、传接成功、接球、接球成功次数
        for i in range(len(chuanjie_chart)):
            team = chuanjie_chart.Team[i]
            player1 = chuanjie_chart.Chuan[i] + team * playerNumber[0]
            player2 = chuanjie_chart.Jie[i] + team * playerNumber[0]
            # 只要有数据，player1添加一次传球
            stat.Chuan[player1] += 1
            if chuanjie_chart.Result1[i] == 1:
                # 只要传球成功，player1添加一次传球，player2添加一次接球
                stat.ChuanCheng[player1] += 1
                stat.JieQiu[player2] += 1
                # 当接球成功，player1添加一次传接成功，player2添加一次接球成功
                if chuanjie_chart.Result2[i] == 1:
                    stat.ChuanJie[player1] += 1
                    stat.JieCheng[player2] += 1

        # 统计射门次数，射正次数，进球次数，抢球次数，解围次数，截取次数
        # 生成1*totalPlayerNumber的numpy矩阵
        stat['SheMen'] = np.zeros(totalPlayerNumber)
        stat['SheZheng'] = np.zeros(totalPlayerNumber)
        stat['JinQiu'] = np.zeros(totalPlayerNumber)
        stat['QiangQiu'] = np.zeros(totalPlayerNumber)
        stat['JieWei'] = np.zeros(totalPlayerNumber)
        stat['JieQu'] = np.zeros(totalPlayerNumber)
        for i in range(len(data) - 1):
            if data.action[i] in ['Goal', 'Miss', 'ShootFailed', 'DangerousShoot']:
                team = int(data.team[i] != teamName[0])
                player = playerDict[team][data.num[i]] + team * playerNumber[0]
                stat.SheMen[player] += 1
                if data.action[i] != 'Miss':
                    stat.SheZheng[player] += 1
                    if data.action[i] == 'Goal':
                        stat.JinQiu[player] += 1
            elif data.action[i] in ['Break', 'Steal', 'Intercept', 'Save', 'Block']:
                team = int(data.team[i] != teamName[0])
                player = playerDict[team][data.num[i]] + team * playerNumber[0]
                stat.QiangQiu[player] += 1
                if data.action[i] in ['Break', 'Block', 'Save']:
                    stat.JieWei[player] += 1
                elif data.action[i] in ['Steal', 'Intercept']:
                    stat.JieQu[player] += 1

        # 添加球队、球员信息
        for i in range(playerNumber[0]):
            stat.iloc[i, 0] = 0
            stat.iloc[i, 1] = playerNumList[0][i]
        for i in range(playerNumber[1]):
            j = i + playerNumber[0]
            stat.iloc[j, 0] = 1
            stat.iloc[j, 1] = playerNumList[1][i]
        # 统计结果存入全局变量
        if h == 0:
            statDataFirst = stat
        else:
            statDataSecond = stat
    # return statDataFirst, statDataSecond


# 自定评价数据生成
def creatScoreData():
    "从Data、Stat中生成自定评价数据，并存入DataFrame形式:Score"
    # 全局变量声明
    global scoreDataFirst, scoreDataSecond

    # 上/下半场 h=0/1
    for h in range(2):
        # 读入原始数据
        if h == 0:
            data = dataFirst
            stat = statDataFirst
        else:
            data = dataSecond
            stat = statDataSecond
        # 定义统计数据score
        score = pd.DataFrame(np.zeros([totalPlayerNumber, 11]),
                             columns=['Team', 'Player', 'ChuanCheng', 'JieCheng', 'LiLiangShiJi', 'ZhuanHua',
                                      'SheZheng', 'ZhuGong', 'WeiXie', 'FangCheng', 'ZhuanGong'])

        # 统计传球成功率（传成/传球），接球成功率（接成/接球），力量时机（传接成/传成），射门转化率（进球/射门），射正率（射正/射门），防守成功率（（解围+截取）/抢球）
        for i in range(len(stat)):
            if stat.Chuan[i] > 0:
                score.ChuanCheng[i] = stat.ChuanCheng[i] / stat.Chuan[i]
            else:
                score.ChuanCheng[i] = 0
            if stat.JieQiu[i] > 0:
                score.JieCheng[i] = stat.JieCheng[i] / stat.JieQiu[i]
            else:
                score.JieCheng[i] = 0
            if stat.ChuanCheng[i] > 0:
                score.LiLiangShiJi[i] = stat.ChuanJie[i] / stat.ChuanCheng[i]
            else:
                score.LiLiangShiJi[i] = 0
            if stat.SheMen[i] > 0:
                score.ZhuanHua[i] = stat.JinQiu[i] / stat.SheMen[i]
            else:
                score.ZhuanHua[i] = 0
            if stat.SheMen[i] > 0:
                score.SheZheng[i] = stat.SheZheng[i] / stat.SheMen[i]
            else:
                score.SheZheng[i] = 0
            if stat.QiangQiu[i] > 0:
                score.FangCheng[i] = (stat.JieWei[i] + stat.JieQu[i]) / stat.QiangQiu[i]
            else:
                score.FangCheng[i] = 0

        # 统计助攻次数，威胁值
        # 设定威胁计算参数cc，设定射门前触球次数nn
        cc = 8
        nn = 5
        for i in range(len(data)):
            if data.action[i] in ['Goal', 'Miss', 'ShootFailed', 'DangerousShoot']:
                # 换算射门行为球队编号，初始化射门球员编号为-9999
                team = int(data.team[i] != teamName[0])
                player = -9999
                # 计算威胁权重，进球=1，射正不进=0.5，射飞=0.25
                aa = 0.5
                if data.action[i] == 'Miss':
                    aa = 0.25
                elif data.action[i] == 'Goal':
                    aa = 1
                # 从第i号数据开始回溯，找到同队触球nn次
                j = i
                count = 0
                while (j >= 0) and (count < nn):
                    # 换算当前球队与队员编号
                    team1 = int(data.team[j] != teamName[0])
                    player1 = playerDict[team1][data.num[j]] + team1 * playerNumber[0]

                    # 判断当前球队与射门球队相同，且当前触球人与上一触球人不同
                    if (team1 == team) and (player1 != player):
                        count += 1
                        # 当count=2时，且进球，即为助攻球员
                        if (count == 2) and (aa == 1):
                            score.ZhuGong[player1] += 1
                        # 根据正态分布公式，离射门越远威胁值越低
                        vv = (aa / math.sqrt(math.pi * cc)) * math.exp(-(count ** 2) / cc)
                        score.WeiXie[player1] += vv
                    j = j - 1
                    team = team1
                    player = player1

        # 统计防守转攻次数，判断标准：抢球之后下一个行为人是队友(包含自己)，且队友行为不是抢球
        for i in range(len(data) - 1):
            if ((data.action[i] in ['Break', 'Steal', 'Intercept', 'Save', 'Block']) and (data.team[i] == data.team[i + 1]) and (
                    data.action[i + 1] not in ['Break', 'Steal', 'Intercept', 'Save', 'Block'])):
                team = int(data.team[i] != teamName[0])
                player = playerDict[team][data.num[i]] + team * playerNumber[0]
                score.ZhuanGong[player] += 1

        # 添加球队、球员信息
        for i in range(playerNumber[0]):
            score.iloc[i, 0] = stat.Team[i]
            score.iloc[i, 1] = stat.Player[i]
        for i in range(playerNumber[1]):
            j = i + playerNumber[0]
            score.iloc[j, 0] = stat.Team[j]
            score.iloc[j, 1] = stat.Player[j]
        # 保存score数据到全局变量
        if h == 0:
            scoreDataFirst = score
        else:
            scoreDataSecond = score
    # return scoreDataFirst, scoreDataSecond


# 传球角度数据生成
def creatAngleData():
    "从Data、Stat中生成自定评价数据，并存入DataFrame形式:Angle"
    # 全局变量声明
    global angleDataFirst, angleDataSecond

    # 上半场
    chuan_jie_chart = passListFirst
    # 定义统计数据DataFrame，传球总数计入T1~T6，传球成功计入S1~S6
    angle = pd.DataFrame(np.zeros([totalPlayerNumber, 16]),
                         columns=['Team', 'Player', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'S1', 'S2', 'S3', 'S4', 'S5',
                                  'S6', 'Turn_Left', 'Turn_Right'])
    zhu_left = home_left
    # 统计传球角度
    for i in range(len(chuan_jie_chart)):
        if chuan_jie_chart.Result1[i] == 1:
            # 提取传球队员信息
            team = chuan_jie_chart.Team[i]
            player = chuan_jie_chart.Chuan[i] + team * playerNumber[0]
            # 计算传球角度
            dx = chuan_jie_chart.X2[i]-chuan_jie_chart.X1[i]
            dy = chuan_jie_chart.Y2[i]-chuan_jie_chart.Y1[i]
            ang = math.atan(dy/dx) * 180 / math.pi
            team = chuan_jie_chart.Team[i]
            # 根据进攻方向换算传球角度
            # 0 == False (True)
            if team != zhu_left:
                if dx < 0:
                    ang = ang+180
            else:
                if dx > 0:
                    ang = ang+180
            ang = ang+90
            # 判断方向，进行累加
            if 0 <= ang < 60:
                angle.T3[player] += 1
                # 判断是否接球成功
                if chuan_jie_chart.Result2[i] == 1:
                    angle.S3[player] += 1
            elif 60 <= ang < 120:
                angle.T2[player] += 1
                # 判断是否接球成功
                if chuan_jie_chart.Result2[i] == 1:
                    angle.S2[player] += 1
            elif 120 <= ang < 180:
                angle.T1[player] += 1
                # 判断是否接球成功
                if chuan_jie_chart.Result2[i] == 1:
                    angle.S1[player] += 1
            elif 180 <= ang < 240:
                angle.T6[player] += 1
                # 判断是否接球成功
                if chuan_jie_chart.Result2[i] == 1:
                    angle.S6[player] += 1
            elif 240 <= ang < 300:
                angle.T5[player] += 1
                # 判断是否接球成功
                if chuan_jie_chart.Result2[i] == 1:
                    angle.S5[player] += 1
            elif 300 <= ang < 360:
                angle.T4[player] += 1
                # 判断是否接球成功
                if chuan_jie_chart.Result2[i] == 1:
                    angle.S4[player] += 1
    # 计算转角
    t_l = np.zeros([totalPlayerNumber, 2])
    t_r = np.zeros([totalPlayerNumber, 2])
    # 判断当前接球与下一传球行为人相同，且两个都是传接成功，且接球到传球之间时间差小于等于10秒
    for i in range(len(chuan_jie_chart)-1):
        if chuan_jie_chart.Team[i] == chuan_jie_chart.Team[i+1] and chuan_jie_chart.Jie[i] == chuan_jie_chart.Chuan[i+1] \
                and chuan_jie_chart.Result2[i]*chuan_jie_chart.Result2[i+1] == 1 and chuan_jie_chart.Time1[i+1] - chuan_jie_chart.Time2[i] <= 10:
            # 计算第一脚传球的接球角度ang1
            dx = chuan_jie_chart.X2[i] - chuan_jie_chart.X1[i]
            dy = chuan_jie_chart.Y2[i] - chuan_jie_chart.Y1[i]
            ang1 = math.atan(dy / dx) * 180 / math.pi+90
            team = chuan_jie_chart.Team[i]
            # 根据进攻方向换算传球角度
            if team != zhu_left:
                if dx < 0:
                    ang1 = ang1 + 180
            else:
                if dx > 0:
                    ang1 = ang1 + 180

            # 计算第二脚传球的传球角度ang2
            dx = chuan_jie_chart.X2[i+1] - chuan_jie_chart.X1[i+1]
            dy = chuan_jie_chart.Y2[i+1] - chuan_jie_chart.Y1[i+1]
            ang2 = math.atan(dy / dx) * 180 / math.pi + 90
            team = chuan_jie_chart.Team[i+1]
            player = chuan_jie_chart.Chuan[i+1] + team * playerNumber[0]
            # 根据进攻方向换算传球角度
            if team != zhu_left:
                if dx < 0:
                    ang2 = ang2 + 180
            else:
                if dx > 0:
                    ang2 = ang2 + 180

            # 计算传球与接球之间夹角
            d_ang = ang2-ang1
            if d_ang < 0:
                d_ang += 360
            # 判断是否向球员右侧转移
            if 0 <= d_ang < 180:
                t_r[player, 0] += d_ang
                t_r[player, 1] += 1
            else:
                t_l[player, 0] += d_ang
                t_l[player, 1] += 1

    # 计算平均转角
    for i in range(totalPlayerNumber):
        if t_l[i, 1] > 0:
            a1 = t_l[i, 0]/t_l[i, 1]
        else:
            a1 = 0
        if t_r[i, 1] > 0:
            a2 = t_r[i, 0]/t_r[i, 1]
        else:
            a2 = 0

        angle.Turn_Left[i] += a1
        angle.Turn_Right[i] += a2

    # 添加球队、球员信息
    for i in range(playerNumber[0]):
        angle.iloc[i, 0] = 0
        angle.iloc[i, 1] = playerNumList[0][i]
    for i in range(playerNumber[1]):
        j = i + playerNumber[0]
        angle.iloc[j, 0] = 1
        angle.iloc[j, 1] = playerNumList[1][i]

    angleDataFirst = angle
    t_l1 = t_l
    t_r1 = t_r

    # 下半场
    chuan_jie_chart = passListSecond
    # 定义统计数据DataFrame，传球总数计入T1~T6，传球成功计入S1~S6
    angle = pd.DataFrame(np.zeros([totalPlayerNumber, 16]),
                         columns=['Team', 'Player', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'S1', 'S2', 'S3', 'S4', 'S5',
                                  'S6', 'Turn_Left', 'Turn_Right'])
    zhu_left = not home_left
    # 统计传球角度
    for i in range(len(chuan_jie_chart)):
        if chuan_jie_chart.Result1[i] == 1:
            # 提取传球队员信息
            team = chuan_jie_chart.Team[i]
            player = chuan_jie_chart.Chuan[i] + team * playerNumber[0]
            # 计算传球角度
            dx = chuan_jie_chart.X2[i]-chuan_jie_chart.X1[i]
            dy = chuan_jie_chart.Y2[i]-chuan_jie_chart.Y1[i]
            ang = math.atan(dy/dx) * 180 / math.pi
            team = chuan_jie_chart.Team[i]
            # 根据进攻方向换算传球角度
            if team != zhu_left:
                if dx < 0:
                    ang = ang+180
            else:
                if dx > 0:
                    ang = ang+180
            ang = ang+90
            # 判断方向，进行累加
            if 0 <= ang < 60:
                angle.T3[player] += 1
                # 判断是否接球成功
                if chuan_jie_chart.Result2[i] == 1:
                    angle.S3[player] += 1
            elif 60 <= ang < 120:
                angle.T2[player] += 1
                # 判断是否接球成功
                if chuan_jie_chart.Result2[i] == 1:
                    angle.S2[player] += 1
            elif 120 <= ang < 180:
                angle.T1[player] += 1
                # 判断是否接球成功
                if chuan_jie_chart.Result2[i] == 1:
                    angle.S1[player] += 1
            elif 180 <= ang < 240:
                angle.T6[player] += 1
                # 判断是否接球成功
                if chuan_jie_chart.Result2[i] == 1:
                    angle.S6[player] += 1
            elif 240 <= ang < 300:
                angle.T5[player] += 1
                # 判断是否接球成功
                if chuan_jie_chart.Result2[i] == 1:
                    angle.S5[player] += 1
            elif 300 <= ang < 360:
                angle.T4[player] += 1
                # 判断是否接球成功
                if chuan_jie_chart.Result2[i] == 1:
                    angle.S4[player] += 1
    # 计算转角
    t_l = np.zeros([totalPlayerNumber, 2])
    t_r = np.zeros([totalPlayerNumber, 2])
    # 判断当前接球与下一传球行为人相同，且两个都是传接成功，且接球到传球之间时间差小于等于10秒
    for i in range(len(chuan_jie_chart)-1):
        if chuan_jie_chart.Team[i] == chuan_jie_chart.Team[i+1] \
                and chuan_jie_chart.Jie[i] == chuan_jie_chart.Chuan[i+1] \
                and chuan_jie_chart.Result2[i]*chuan_jie_chart.Result2[i+1] == 1 \
                and chuan_jie_chart.Time1[i+1]-chuan_jie_chart.Time2[i] <= 10:
            # 计算第一脚传球的接球角度ang1
            dx = chuan_jie_chart.X2[i] - chuan_jie_chart.X1[i]
            dy = chuan_jie_chart.Y2[i] - chuan_jie_chart.Y1[i]
            ang1 = math.atan(dy / dx) * 180 / math.pi+90
            team = chuan_jie_chart.Team[i]
            # 根据进攻方向换算传球角度
            if team != zhu_left:
                if dx < 0:
                    ang1 = ang1 + 180
            else:
                if dx > 0:
                    ang1 = ang1 + 180
            # 计算第二脚传球的传球角度ang2
            dx = chuan_jie_chart.X2[i+1] - chuan_jie_chart.X1[i+1]
            dy = chuan_jie_chart.Y2[i+1] - chuan_jie_chart.Y1[i+1]
            ang2 = math.atan(dy / dx) * 180 / math.pi + 90
            team = chuan_jie_chart.Team[i+1]
            player = chuan_jie_chart.Chuan[i+1] + team * playerNumber[0]
            # 根据进攻方向换算传球角度
            if team != zhu_left:
                if dx < 0:
                    ang2 = ang2 + 180
            else:
                if dx > 0:
                    ang2 = ang2 + 180
            # 计算传球与接球之间夹角
            d_ang = ang2-ang1
            if d_ang < 0:
                d_ang += 360
            # 判断是否向球员右侧转移
            if 0 <= d_ang < 180:
                t_r[player, 0] += d_ang
                t_r[player, 1] += 1
            else:
                t_l[player, 0] += d_ang
                t_l[player, 1] += 1

    # 计算平均转角
    for i in range(totalPlayerNumber):
        if t_l[i, 1] > 0:
            a1 = t_l[i, 0]/t_l[i, 1]
        else:
            a1 = 0
        if t_r[i, 1] > 0:
            a2 = t_r[i, 0]/t_r[i, 1]
        else:
            a2 = 0

        angle.Turn_Left[i] += a1
        angle.Turn_Right[i] += a2

    # 添加球队、球员信息
    for i in range(playerNumber[0]):
        angle.iloc[i, 0] = 0
        angle.iloc[i, 1] = playerNumList[0][i]
    for i in range(playerNumber[1]):
        j = i + playerNumber[0]
        angle.iloc[j, 0] = 1
        angle.iloc[j, 1] = playerNumList[1][i]

    angleDataSecond = angle

    # return angleDataFirst, angleDataSecond


# 画传接热图
def creatHeatmapData():
    "将触球位置画成热图"
    # 全局变量声明
    global passDefenseTouch, touchFirst, touchSecond, touchDefenseFirst, touchDefenseSecond

    # 计算场地图上场地区域长度与宽度，单位：像素
    length = MAX_X-MIN_X
    width = MAX_Y-MIN_Y

    ######
    # 无论上半场、下半场、全场，都定义主队由左向右攻，客队由右向左攻
    # 在触球数据生成过程中进行归一化转换
    ######

    # 上半场
    # 球员触球位置
    # [[]]*6 = [[], [], [], [], [], []]
    touch = [[]]*totalPlayerNumber
    # 读入传接列表数据
    serial_number = 0
    chuan_jie_chart = passListFirst
    # 从ChuanJie_Chart中将球员传接球位置坐标（tuple）格式存入touch变量
    for i in range(len(chuan_jie_chart)):
        # 统计每个球员的传球坐标
        team = chuan_jie_chart.Team[i]
        player = chuan_jie_chart.Chuan[i] + team * playerNumber[0]
        # 判断坐标是否出场地
        xx = chuan_jie_chart.X1[i]
        if np.isnan(xx):
            xx = 0

        if xx < 0:
            xx = 0
        elif xx > LENGTH_F:
            xx = LENGTH_F
        yy = chuan_jie_chart.Y1[i]
        if np.isnan(yy):
            yy = 0
        if yy < 0:
            yy = 0
        elif yy > WIDTH_F:
            yy = WIDTH_F

        if touch[player] == []:
            # 'F'表示失败，先默认失败
            touch[player] = [[int(xx/LENGTH_F*length+MIN_X), int(yy/WIDTH_F*width+MIN_Y), -1, 'F']]
        else:
            touch[player].append([int(xx/LENGTH_F*length+MIN_X), int(yy/WIDTH_F*width+MIN_Y), -1, 'F'])

        playerPass = player

        # 如果传球成功，统计接球球员的接球坐标
        if chuan_jie_chart.Result1[i] == 1:
            player = chuan_jie_chart.Jie[i] + team * playerNumber[0]
            touch[playerPass][-1][2] = angleDataFirst['Player'][player]
            # 'S'表示成功，如果球员接球成功，则传球队员传球成功
            touch[playerPass][-1][3] = 'S'
            # 判断坐标是否出场地
            xx = chuan_jie_chart.X2[i]
            if np.isnan(xx):
                xx = 0
            if xx < 0:
                xx = 0
            elif xx > LENGTH_F:
                xx = LENGTH_F
            yy = chuan_jie_chart.Y2[i]
            if np.isnan(yy):
                yy = 0
            if yy < 0:
                yy = 0
            elif yy > WIDTH_F:
                yy = WIDTH_F

            if touch[player] == []:
                touch[player] = [[int(xx/LENGTH_F*length+MIN_X), int(yy/WIDTH_F*width+MIN_Y), -1, 'S']]
            else:
                touch[player].append([int(xx/LENGTH_F*length+MIN_X), int(yy/WIDTH_F*width+MIN_Y), -1, 'S'])

    # 球员防守触球位置
    touch_d = [[]]*totalPlayerNumber
    # 读入防守列表数据
    fangshou_chart = defenseListFirst
    # 从fangshou_chart中将球员传接球位置坐标（tuple）格式存入touch_d变量
    for i in range(len(fangshou_chart)):
        # 统计每个球员的防守坐标
        team = fangshou_chart.Team[i]
        player = fangshou_chart.Fang[i] + team * playerNumber[0]
        # 判断坐标是否出场地
        xx = fangshou_chart.X1[i]
        if np.isnan(xx):
            xx = 0
        if xx < 0:
            xx = 0
        elif xx > LENGTH_F:
            xx = LENGTH_F
        yy = fangshou_chart.Y1[i]
        if np.isnan(yy):
            yy = 0
        if yy < 0:
            yy = 0
        elif yy > WIDTH_F:
            yy = WIDTH_F
        if fangshou_chart.Result1[i] == 'yes':
            rr = 'S'
        else:
            rr = 'F'

        if touch_d[player] == []:
            touch_d[player] = [[int(xx/LENGTH_F*length+MIN_X), int(yy/WIDTH_F*width+MIN_Y), -1, rr]]
        else:
            touch_d[player].append([int(xx/LENGTH_F*length+MIN_X), int(yy/WIDTH_F*width+MIN_Y), -1, rr])

    # 判断如果主队开局时不是从左向右攻，则主客双方上半场的所有触球位置需要进行方向转换
    if not home_left:
        for i in range(totalPlayerNumber):
            for j in range(len(touch[i])):
                (xx, yy, nn, rr) = touch[i][j]
                touch[i][j] = [(length-(xx-MIN_X))+MIN_X, (width-(yy-MIN_Y))+MIN_Y, nn, rr]
            for j in range(len(touch_d[i])):
                (xx, yy, nn, rr) = touch_d[i][j]
                touch_d[i][j] = [(length-(xx-MIN_X))+MIN_X, (width-(yy-MIN_Y))+MIN_Y, nn, rr]

    # 传接触球位置存入全局变量
    touchFirst = touch
    touchDefenseFirst = touch_d

    # 下半场
    # 球员触球位置
    touch = [[]] * totalPlayerNumber
    # 读入传接列表数据
    chuan_jie_chart = passListSecond
    # 从ChuanJie_Chart中将球员传接球位置坐标（tuple）格式存入touch变量
    for i in range(len(chuan_jie_chart)):
        # 统计每个球员的传球坐标
        team = chuan_jie_chart.Team[i]
        player = chuan_jie_chart.Chuan[i] + team * playerNumber[0]
        # 判断坐标是否出场地
        xx = chuan_jie_chart.X1[i]
        if np.isnan(xx):
            xx = 0
        if xx < 0:
            xx = 0
        elif xx > LENGTH_F:
            xx = LENGTH_F
        yy = chuan_jie_chart.Y1[i]
        if np.isnan(yy):
            yy = 0
        if yy < 0:
            yy = 0
        elif yy > WIDTH_F:
            yy = WIDTH_F
        if touch[player] == []:
            touch[player] = [
                [int(xx / LENGTH_F * length + MIN_X), int(yy / WIDTH_F * width + MIN_Y), -1, 'F']]
        else:
            touch[player].append(
                [int(xx / LENGTH_F * length + MIN_X), int(yy / WIDTH_F * width + MIN_Y), -1, 'F'])

        playerPass = player

        # 如果传球成功，统计接球球员的接球坐标，和传球来自哪个队员
        if chuan_jie_chart.Result1[i] == 1:
            player = chuan_jie_chart.Jie[i] + team * playerNumber[0]
            touch[playerPass][-1][2] = angleDataFirst['Player'][player]
            touch[playerPass][-1][3] = 'S'
            # 判断坐标是否出场地
            xx = chuan_jie_chart.X2[i]
            if np.isnan(xx):
                xx = 0
            if xx < 0:
                xx = 0
            elif xx > LENGTH_F:
                xx = LENGTH_F
            yy = chuan_jie_chart.Y2[i]
            if np.isnan(yy):
                yy = 0
            if yy < 0:
                yy = 0
            elif yy > WIDTH_F:
                yy = WIDTH_F

            if touch[player] == []:
                touch[player] = [
                    [int(xx / LENGTH_F * length + MIN_X), int(yy / WIDTH_F * width + MIN_Y), -1, 'S']]
            else:
                touch[player].append(
                    [int(xx / LENGTH_F * length + MIN_X), int(yy / WIDTH_F * width + MIN_Y), -1, 'S'])

    # 球员防守触球位置
    touch_d = [[]] * totalPlayerNumber
    # 读入防守列表数据
    fangshou_chart = defenseListSecond
    # 从fangshou_chart中将球员防守位置坐标（tuple）格式存入touch_d变量
    for i in range(len(fangshou_chart)):
        # 统计每个球员的防守坐标
        team = fangshou_chart.Team[i]
        player = fangshou_chart.Fang[i] + team * playerNumber[0]
        # 判断坐标是否出场地
        xx = fangshou_chart.X1[i]
        if np.isnan(xx):
            xx = 0
        if xx < 0:
            xx = 0
        elif xx > LENGTH_F:
            xx = LENGTH_F
        yy = fangshou_chart.Y1[i]
        if np.isnan(yy):
            yy = 0
        if yy < 0:
            yy = 0
        elif yy > WIDTH_F:
            yy = WIDTH_F
        if fangshou_chart.Result1[i] == 'yes':
            rr = 'S'
        else:
            rr = 'F'
        if touch_d[player] == []:
            touch_d[player] = [[int(xx / LENGTH_F * length + MIN_X),
                                int(yy / WIDTH_F * width + MIN_Y), -1, rr]]
        else:
            touch_d[player].append([int(xx / LENGTH_F * length + MIN_X),
                                    int(yy / WIDTH_F * width + MIN_Y), -1, rr])

    # 判断如果主队开局时从左向右攻，则主客双方下半场的所有触球位置需要进行方向转换
    if home_left:
        for i in range(totalPlayerNumber):
            for j in range(len(touch[i])):
                (xx, yy, nn, rr) = touch[i][j]
                touch[i][j] = [(length-(xx-MIN_X))+MIN_X, (width-(yy-MIN_Y))+MIN_Y, nn, rr]
            for j in range(len(touch_d[i])):
                (xx, yy, nn, rr) = touch_d[i][j]
                touch_d[i][j] = [(length-(xx-MIN_X))+MIN_X, (width-(yy-MIN_Y))+MIN_Y, nn, rr]

    # 传接触球位置存入全局变量
    touchSecond = touch
    touchDefenseSecond = touch_d

    # 热图触球数据:passDefenseTouch
    # 攻防触球数据
    passDefenseTouch = pd.DataFrame(
        columns=['game', '顺序', 'Gong/Fang', 'Result', '1/2 half', 'Home/Away', 'Number', 'Name', 'X in pixel', 'Y in pixel',
                 'Receive'])
    # 顺序变量
    cx = 0
    # 读处理上半场数据
    for i in range(len(touchFirst)):
        # 判断是主队还是客队
        if i < playerNumber[0]:
            side = 'H'
        else:
            side = 'A'
        # 状态为上半场进攻
        half = 1
        gongfang = 'G'
        if side == 'H':
            tm = 0
        else:
            tm = 1

        # 处理进攻数据
        for j in range(len(touchFirst[i])):
            # 存入第i号球员的第j个数据
            cx += 1
            touch_row = pd.DataFrame([[game, cx, gongfang, touchFirst[i][j][3], half, side,
                                       int(angleDataFirst['Player'][i]), playerNameList[tm][angleDataFirst['Player'][i]],
                                       touchFirst[i][j][0], touchFirst[i][j][1], int(touchFirst[i][j][2])]],
                                     columns=['game', '顺序', 'Gong/Fang', 'Result', '1/2 half', 'Home/Away', 'Number',
                                              'Name', 'X in pixel', 'Y in pixel', 'Receive'])
            passDefenseTouch = passDefenseTouch.append(touch_row, ignore_index=True, sort=False)

        # 状态为防守
        gongfang = 'F'
        # 处理防守数据
        for j in range(len(touchDefenseFirst[i])):
            cx += 1
            # 存入第i号球员的第j个数据
            touch_row = pd.DataFrame([[game, cx, gongfang, touchDefenseFirst[i][j][3], half, side,
                                       int(angleDataFirst['Player'][i]), playerNameList[tm][angleDataFirst['Player'][i]],
                                       touchDefenseFirst[i][j][0], touchDefenseFirst[i][j][1], int(touchDefenseFirst[i][j][2])]],
                                     columns=['game', '顺序', 'Gong/Fang', 'Result', '1/2 half', 'Home/Away', 'Number',
                                              'Name', 'X in pixel', 'Y in pixel', 'Receive'])
            passDefenseTouch = passDefenseTouch.append(touch_row, ignore_index=True, sort=False)

    # 读处理下半场数据
    for i in range(len(touchSecond)):
        # 判断是主队还是客队
        if i < playerNumber[0]:
            side = 'H'
        else:
            side = 'A'
        # 状态为下半场进攻
        half = 2
        gongfang = 'G'
        if side == 'H':
            tm = 0
        else:
            tm = 1

        # 处理进攻数据
        for j in range(len(touchSecond[i])):
            cx += 1
            # 存入第i号球员的第j个数据
            touch_row = pd.DataFrame([[game, cx, gongfang, touchSecond[i][j][3], half, side, int(angleDataFirst['Player'][i]),
                                       playerNameList[tm][angleDataFirst['Player'][i]], touchSecond[i][j][0],
                                       touchSecond[i][j][1], int(touchSecond[i][j][2])]],
                                     columns=['game', '顺序', 'Gong/Fang', 'Result', '1/2 half', 'Home/Away', 'Number',
                                              'Name', 'X in pixel', 'Y in pixel', 'Receive'])
            passDefenseTouch = passDefenseTouch.append(touch_row, ignore_index=True, sort=False)

        # 状态为防守
        gongfang = 'F'
        # 处理防守数据
        for j in range(len(touchDefenseSecond[i])):
            cx += 1
            # 存入第i号球员的第j个数据
            touch_row = pd.DataFrame([[game, cx, gongfang, touchDefenseSecond[i][j][3], half, side,
                                       int(angleDataFirst['Player'][i]),  playerNameList[tm][angleDataFirst['Player'][i]],
                                       touchDefenseSecond[i][j][0], touchDefenseSecond[i][j][1], int(touchDefenseSecond[i][j][2])]],
                                     columns=['game', '顺序', 'Gong/Fang', 'Result', '1/2 half', 'Home/Away', 'Number',
                                              'Name', 'X in pixel', 'Y in pixel', 'Receive'])
            passDefenseTouch = passDefenseTouch.append(touch_row, ignore_index=True, sort=False)
    # 把game前面
    # ggm = passDefenseTouch.game
    # passDefenseTouch.drop(labels=['game'], axis=1, inplace=True)
    # passDefenseTouch.insert(0, 'game', ggm)

    return ()


# 将数据写入Excel文件
def writeDataFile():
    "将传接图等数据写入Excel文件"
    # global  #全局变量声明

    writer = pd.ExcelWriter(outputFilePath)
    # 如果数据有问题可以用下列代码来查看出问题的地方在哪里
    # passListFirst.to_excel(writer, sheet_name='passListFirst')
    # passListSecond.to_excel(writer, sheet_name='passListSecond')
    # defenseListFirst.to_excel(writer, sheet_name='defenseListFirst')
    # defenseListSecond.to_excel(writer, sheet_name='defenseListSecond')
    # statDataFirst.to_excel(writer, sheet_name='statDataFirst')
    # statDataSecond.to_excel(writer, sheet_name='statDataSecond')
    # scoreDataFirst.to_excel(writer, sheet_name='scoreDataFirst')
    # scoreDataSecond.to_excel(writer, sheet_name='scoreDataSecond')
    # angleDataFirst.to_excel(writer, sheet_name='angleDataFirst')
    # angleDataSecond.to_excel(writer, sheet_name='angleDataSecond')

    # 雷达图数据
    radar_chart = pd.DataFrame(np.zeros([totalPlayerNumber*2, 41]),
                               columns=['比赛', '顺序', '主客', '上下半场', '球员号码', '球员姓名', '中心性', '传球数', '传球到位',
                                        '传接成功', '接球数', '接球成功', '射门数', '射正数', '进球', '抢球数', '解围', '截取', '传球成功率',
                                        '接球成功率', '力量时机', '射门转化率', '射正率', '助攻次数', '进攻威胁', '抢球成功率', '守转攻',
                                        '传球方向1', '传球方向2', '传球方向3', '传球方向4', '传球方向5', '传球方向6',
                                        '传成方向1', '传成方向2', '传成方向3', '传成方向4', '传成方向5', '传成方向6',
                                        '向左转移', '向右转移'])
    # 数据顺序编号，id从1开始
    id = 0
    # 上/下半场 h=0/1
    for h in range(2):
        if h == 0:
            half = 1
            stat = statDataFirst
            score = scoreDataFirst
            angle = angleDataFirst
            pass_net = passNetFirst

        else:
            half = 2
            stat = statDataSecond
            score = scoreDataSecond
            angle = angleDataSecond
            pass_net = passNetSecond

        # 主客队 i=0/1
        for i in range(2):
            if i == 0:
                team = 'H'
            else:
                team = 'A'
            net = pass_net[i]

            G = nx.DiGraph()   # 传接图变量
            for ii in range(playerNumber[i]):
                for jj in range(playerNumber[i]):
                    if (net[ii][jj] > 0) and (ii != jj):
                        G.add_edge(ii, jj, weight=1/net[ii][jj])
            # 计算介数中心性，字典变量
            bc = nx.algorithms.centrality.betweenness_centrality(G, normalized=True, weight='weight')

            # 球员编号 j=1~playerNumber[i]
            for j in range(playerNumber[i]):
                id_player = i*playerNumber[0]+j
                # 常规参数
                radar_chart.loc[id, '比赛'] = game
                radar_chart.loc[id, '顺序'] = id+1
                radar_chart.loc[id, '主客'] = team
                radar_chart.loc[id, '上下半场'] = half
                radar_chart.loc[id, '球员号码'] = int(playerNumList[i][j])
                radar_chart.loc[id, '球员姓名'] = playerNameList[i][playerNumList[i][j]]

                if j in bc:
                    radar_chart.loc[id, '中心性'] = bc[j]
                radar_chart.loc[id, '传球数'] = stat['Chuan'][id_player]
                radar_chart.loc[id, '传球到位'] = stat['ChuanCheng'][id_player]
                radar_chart.loc[id, '传接成功'] = stat['ChuanJie'][id_player]
                radar_chart.loc[id, '接球数'] = stat['JieQiu'][id_player]
                radar_chart.loc[id, '接球成功'] = stat['JieCheng'][id_player]
                radar_chart.loc[id, '射门数'] = stat['SheMen'][id_player]
                radar_chart.loc[id, '射正数'] = stat['SheZheng'][id_player]
                radar_chart.loc[id, '进球'] = stat['JinQiu'][id_player]
                radar_chart.loc[id, '抢球数'] = stat['QiangQiu'][id_player]
                radar_chart.loc[id, '解围'] = stat['JieWei'][id_player]
                radar_chart.loc[id, '截取'] = stat['JieQu'][id_player]

                if radar_chart['传球数'][id] > 0:
                    radar_chart.loc[id, '传球成功率'] = radar_chart['传接成功'][id]/radar_chart['传球数'][id]

                if radar_chart['接球数'][id] > 0:
                    radar_chart.loc[id, '接球成功率'] = radar_chart['接球成功'][id]/radar_chart['接球数'][id]

                if radar_chart['传球到位'][id] > 0:
                    radar_chart.loc[id, '力量时机'] = radar_chart['传接成功'][id]/radar_chart['传球到位'][id]

                if radar_chart['射门数'][id] > 0:
                    radar_chart.loc[id, '射门转化率'] = radar_chart['进球'][id]/radar_chart['射门数'][id]

                if radar_chart['射门数'][id] > 0:
                    radar_chart.loc[id, '射正率'] = radar_chart['射正数'][id]/radar_chart['射门数'][id]

                if radar_chart['抢球数'][id] > 0:
                    radar_chart.loc[id, '抢球成功率'] = (radar_chart['解围'][id]+radar_chart['截取'][id])/radar_chart['抢球数'][id]

                radar_chart.loc[id, '助攻次数'] = score['ZhuGong'][id_player]
                radar_chart.loc[id, '进攻威胁'] = score['WeiXie'][id_player]
                radar_chart.loc[id, '守转攻'] = score['ZhuanGong'][id_player]

                # 传球方向
                # 要改输出数据的话，这里也得改
                for k in range(14):
                    radar_chart.iloc[id, k+27] = angle.iloc[id_player, k+2]
                id += 1

    # 网站数据
    radar_chart.to_excel(writer, sheet_name='比赛数据', index=False)
    # 触球与热图
    passDefenseTouch.to_excel(writer, sheet_name='触球位置与传球位置', index=False)
    # 完成保存
    writer.save()
    return ()


if __name__ == '__main__':
    creatPassDefenseList()
    creatPassNet()
    creatStatData()
    creatScoreData()
    creatAngleData()
    creatHeatmapData()
    writeDataFile()

