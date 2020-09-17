



def divide_range(Ori_img_W, Ori_img_H, section_num, mode=0):
    '''

    :param Ori_img_W:
    :param Ori_img_H:
    :param section_num: divide the weight/height to (%s section_num) parts.
    :param mode: mode=0 to divide weight, mode=1 to divide height
    :return:
    '''
    if mode==0:
        section = []
        U_rate = 0.2
        # n-1 incomplete(1-U_rate) plus a complete one
        part_img_w = int(Ori_img_W / ((section_num - 1) * (1 - U_rate) + 1))
        w_range = []
        for i in range(section_num):
            part_ = [0, Ori_img_H, int(i * part_img_w * (1 - U_rate)), int(i * part_img_w * (1 - U_rate)) + part_img_w]
            if i == section_num - 1:
                part_[3] = Ori_img_W
            section.append(part_)
            w_range.append(part_[2])
            w_range.append(part_[3])
        box_range = []
        middle = []
        w_range.sort()
        middle.append(w_range[0])
        for i in range(1, len(w_range) - 1, 2):
            w_ = int((w_range[i] + w_range[i + 1]) / 2)
            middle.append(w_)
        middle.append(w_range[-1])
        for i in range(section_num):
            box_range.append([middle[i], middle[i + 1]])
        return section,box_range,part_img_w
    else:
        section = []
        U_rate = 0.4
        part_img_h = int(Ori_img_H / ((section_num - 1) * (1 - U_rate) + 1))
        h_range = []
        for i in range(section_num):
            part_ = [int(i * part_img_h * (1 - U_rate)), int(i * part_img_h * (1 - U_rate)) + part_img_h, 0, Ori_img_W]
            if i == section_num - 1:
                part_[1] = Ori_img_H
            section.append(part_)
            h_range.append(part_[0])
            h_range.append(part_[1])
        box_range = []
        middle = []
        h_range.sort()
        middle.append(h_range[0])
        for i in range(1, len(h_range) - 1, 2):
            h_ = int((h_range[i] + h_range[i + 1]) / 2)
            middle.append(h_)
        middle.append(h_range[-1])
        for i in range(section_num):
            box_range.append([middle[i], middle[i + 1]])
        return section, box_range,part_img_h