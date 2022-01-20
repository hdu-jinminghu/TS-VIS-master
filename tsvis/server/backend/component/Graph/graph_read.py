# -*- coding: UTF-8 -*-
"""
 Copyright 2021 Tianshu AI Platform. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 =============================================================
"""
from tsvis.server.backend.component.Graph.graph import Node
from tsvis.server.backend.component.Graph.graph import Proxy

# Base_RATE = 16
# nodes保留每个节点的单独信息，以节点全称为key进行索引
nodes = {}


# 去除控制边的^
def get_source_name(s):
    tmp = s
    if ":" in s:
        tmp = tmp.split(":")[0]
    if '^' in s:
        tmp = tmp[1:]
        return tmp
    return tmp


# 同根同支，以短边的尽头为分叉点
# 同根不同支，
# 不同根
def diff_index_find(small_len, node_edge_list, target_edge_list, same_root, same_branch):
    # 遍历寻找分叉点
    for i in range(small_len):

        # 判断是否同根
        if i == 0:
            if node_edge_list[i] == target_edge_list[i]:
                same_root = True
            else:
                same_root = False
                break

        # 判断是否同支
        if i == small_len - 1:
            if node_edge_list[i] == target_edge_list[i]:
                same_branch = True
            else:
                same_branch = False
                break

        if node_edge_list[i] != target_edge_list[i]:
            break

    if same_branch:
        return -1
    else:
        return i


def edge_build(edge_list, diff_index):
    nodes_list = []
    curr_edge = ""
    for edge in edge_list:
        if curr_edge == "":
            curr_edge = edge
        else:
            curr_edge = curr_edge + "/" + edge
        nodes_list.append(curr_edge)
    for i in range(diff_index):
        # print(str(i)+"-"+str(diff_index)+"-"+str(len(nodes_list)))
        del nodes_list[0]
    return nodes_list


def edge_deal(node, edges_info, edges_info_num, edges_info_list, edges_control_info):
    # 获取当前节点的全称
    node_path = node["uid"]
    # 获取当前节点的目标节点集
    targets = node["targets"]
    # targets = []
    # for target in targets_old:
    #     target = get_source_name(target)
    #     targets.append(target)
    # 获取当前节点的输出维度集
    if "_output_shapes" in node["attrs"]:
        output_shapes = node["attrs"]["_output_shapes"]
    else:
        output_shapes = [""]

    # 判断输出维度集中是否只有一个输出
    sign = 1
    if len(output_shapes) == 1:
        sign = 0
    else:
        sign = 1
    # 存储当sign为1时，两个节点之间的边的信息
    cur2targets_edge_info = []
    if sign == 1:
        # 存储当前输出维度与目标节点输出维度均不一致的情况
        # candidate_list = []
        for i, target in enumerate(targets):

            target_clean = get_source_name(target)
            cur2targets_edge_info.append("{?}")
            # 标记是否唯一匹配
            unique_sign = False
            duplication_sign = False
            if "_output_shapes" in nodes[target_clean]["attrs"]:
                target_output_shapes = nodes[target_clean]["attrs"]["_output_shapes"]
            else:
                target_output_shapes = [""]
            # 若有匹配
            # 若有单个匹配，则在cur2targets_edge_info的相应位置存储维度信息
            # 若有多个匹配，则在cur2targets_edge_info的相应位置存储？，并将其存入候选列
            for output_shape in output_shapes:
                if duplication_sign:
                    break
                for target_output_shape in target_output_shapes:
                    if (output_shape == target_output_shape) & (unique_sign is False):
                        unique_sign = True
                        cur2targets_edge_info[i] = output_shape
                        break
                    elif (output_shape == target_output_shape) & (unique_sign is True):
                        duplication_sign = True
                        cur2targets_edge_info[i] = "{?}"
                        # candidate_list.append(target)
                        break
                    else:
                        continue
            # 若无匹配，则在cur2targets_edge_info的相应位置存储？，并将其存入候选列
            if (unique_sign is False) & (duplication_sign is False):
                cur2targets_edge_info[i] = "{?}"
                # candidate_list.append(target)

    # 判断两节点是否同根
    same_root = False
    # 判断两节点是否同支
    same_branch = False
    # 拆分指出边的路径
    node_edge_list = node_path.strip("/").split("/")
    node_edge_len = len(node_edge_list)

    # 根据node_path和target的联合建立一个key,add中根据索引取出边的信息和j组成字典
    edges_info_temp = {}
    edges_info_num_temp = {}
    for k, target in enumerate(targets):
        # 判断是否是控制边
        if '^' in target:
            control_sign = True
        else:
            control_sign = False
        target_clean = get_source_name(target)

        # 拆分每个指入边的的路径
        target_edge_list = target_clean.strip("/").split("/")
        target_edge_len = len(target_edge_list)

        # 寻找分叉点
        if node_edge_len < target_edge_len:
            diff_index = diff_index_find(node_edge_len, node_edge_list, target_edge_list, same_root, same_branch)
        else:
            diff_index = diff_index_find(target_edge_len, node_edge_list, target_edge_list, same_root, same_branch)

        # 构边与插入
        # 同支情况下由于展开父节点消失，故不进行边的构建
        if diff_index == -1:
            continue
        else:
            from_nodes = edge_build(node_edge_list, diff_index)
            to_nodes = edge_build(target_edge_list, diff_index)
            for i in from_nodes:
                for j in to_nodes:
                    # 构造和存储每条边的信息
                    edge_id = i + "__" + j
                    if sign == 0:
                        edges_info_temp[edge_id] = output_shapes[0]
                    else:
                        edges_info_temp[edge_id] = cur2targets_edge_info[k]

                    # 构造和存储每条边的控制信息
                    # 若对于一条边既存在实边也存在控制边，则绘制为实边
                    if control_sign:
                        # 若该边存在多条
                        # 若有实边存在，则绘制为实边，否在绘制为虚边
                        if edge_id not in edges_control_info.keys():
                            edges_control_info[edge_id] = 'true'
                    else:
                        edges_control_info[edge_id] = 'false'
                    # 存储同一节点出来边的数量信息
                    edges_info_num_temp[edge_id] = 1
                    # 存储每个节点的目标节点
                    nodes[i]["targets"].add(j)

    # 实现基于不同节点边的累加
    node_name_len = len(node_edge_list)
    node_name_simplified = node_edge_list[-1]
    for key in edges_info_temp.keys():
        if key in edges_info.keys():
            # 判断是否有节点及其虚节点
            # 若有则返回ture
            # 在true情况下，由于节点及其虚节点实际为同一节点，为避免重复计算，故不加入
            for i, name_and_len in enumerate(edges_info_list[key]):
                if i == 0:
                    continue
                else:
                    name = name_and_len.split(',')[0]
                    name_len = int(name_and_len.split(',')[1])
                    if (((name.strip("(").strip(")") == node_name_simplified) & ((name_len - 1) == node_name_len)) | (
                            (node_name_simplified.strip("(").strip(")") == name) & ((name_len + 1) == node_name_len))):
                        edges_info_list[key][0] = 'true'
                        break
                    else:
                        continue

            if edges_info_list[key][0] == 'false':
                # 重合边信息处理
                if ((edges_info_temp[key] == '') & (edges_info[key] == '')):
                    edges_info[key] = ''
                else:
                    if edges_info[key] == '':
                        for i in range(edges_info_num[key]):
                            if i == 0:
                                edges_info[key] = '()'
                            else:
                                edges_info[key] = edges_info[key] + ';' + '()'
                    if edges_info_temp[key] == '':
                        edges_info_temp[key] = '()'
                    edges_info[key] = edges_info[key] + ';' + edges_info_temp[key]

                edges_info_num[key] += 1

                edges_info_list[key].append(node_name_simplified + ',' + str(node_name_len))
            else:
                continue


        else:
            edges_info[key] = edges_info_temp[key]
            edges_info_num[key] = edges_info_num_temp[key]

            # edges_info_list[key]中第一个元素用于判断是否已经做过节点及其虚节点重复信息的删除
            edges_info_list[key] = ['false']
            # 其它元素用于保存节点的缩略名及其长度，用于节点与其对应虚节点的判断
            edges_info_list[key].append(node_name_simplified + ',' + str(node_name_len))





# 提取所需信息构造数据结构
# label  节点简称
# index  节点的唯一数字索引，每层赋予16个编号
# parent 节点的父节点全称
# uid    节点的全称
# op     操作
# layer  节点当前所在的层级
# targets 目标节点
# attrs  属性
def data_build(tree, graph, data, level, curr_path=None):
    # parent用于存储父节点名称
    # curr_path用于存储当前路径，主要用于虚节点的构造上
    parent = curr_path

    if len(tree.child) > 0:
        for i, sub_tree in enumerate(tree.child):

            # tree.child 是一个由缩略名作为key，TreeNode结构体作为value的字典
            node_name = sub_tree
            # index += 1

            if level == 0:
                # 第一层节点不存在父节点
                node_parent = ""
                curr_path = node_name
            else:
                node_parent = parent
                curr_path = node_parent + '/' + node_name

            # 用于判断节点是否存在，即是否在log中存有信息
            exits = True
            if tree.child[node_name].node is None:
                exits = False

            node_targets = []
            if exits:
                node_info = tree.child[node_name].node  # 一个字典
                for target in node_info.output:
                    node_targets.append(target)
                    # edge_deal(target,node_targets)
            else:
                # 为虚节点构造一个空的Node结构体
                # {
                # 	 name:
                # 	 op:
                # 	 input:
                # 	 output:
                # 	 attrs{}
                # }
                # 构造函数中需要包含name和op信息
                node_info = Node(curr_path, "")

            # 从node_info中获取节点的全称
            node_name_full = node_info.name
            # 处理attr的shape和_output_shape属性，处理成{1,1,1}形式的字符串
            node_attr = {}
            for key in node_info.attr.keys():
                if (key != "_output_shapes") & (key != "shape"):
                    node_attr[key] = str(node_info.attr[key]).replace('\n', '')
                elif key == "shape":
                    node_attr[key] = str(node_info.attr[key]).replace('\n', '')
                elif key == "_output_shapes":
                    # 在_output_shapes中用一个list存储当前节点到其他节点边的维度信息，每一个是一个shape
                    shape_list = node_info.attr[key].list.shape
                    output_shapes = []
                    for shape in shape_list:
                        raw_dim = shape.dim
                        raw_dim_length = len(raw_dim)
                        new_dim = ""
                        for j, dim in enumerate(raw_dim):
                            str_dim = ""
                            if dim.size == -1:
                                str_dim = "?"
                            else:
                                str_dim = str(dim.size)
                            if j == 0:
                                new_dim = '{' + str_dim
                            else:
                                new_dim += ',' + str_dim
                            if j == raw_dim_length - 1:
                                new_dim += '}'
                        output_shapes.append(new_dim)
                    node_attr[key] = output_shapes

            # 节点构造与添加
            node = {}
            node["label"] = node_name
            # node["index"] = index
            node["parent"] = node_parent
            node["uid"] = node_name_full
            node["op"] = node_info.op
            node["layer"] = level + 1
            node["attrs"] = node_attr

            node2nodes = node.copy()
            # nodes中node的边不重复，且仅含当前节点的信息,构建时为空，在处理后添加
            node2nodes["targets"] = set()
            nodes[node_name_full] = node2nodes

            node["targets"] = node_targets
            node["sub_net"] = []
            if level == 0:
                data.append(node)
                data_build(tree.child[node_name], graph, data[i], level + 1, curr_path)
            else:
                data["sub_net"].append(node)
                data_build(tree.child[node_name], graph, data["sub_net"][i], level + 1, curr_path)


def data_search(data, level, edges_info, edges_info_num, edges_info_list, edges_control_info, build=True):
    # 由于第一层和其他层数据的存储形式不同
    # 第一层直接存放，其他层存在各自的subnet中
    if level == 1:
        for sub_data in data:
            if build:
                edge_deal(sub_data, edges_info, edges_info_num, edges_info_list, edges_control_info)
            else:
                targets_list = list(nodes[sub_data["uid"]]["targets"])
                sub_data["targets"] = []
                for target in targets_list:
                    edge_dic = {}
                    edge_id = sub_data["uid"] + "__" + target
                    edge_dic["id"] = target
                    edge_dic["info"] = edges_info[edge_id]
                    edge_dic["control"] = edges_control_info[edge_id]
                    edge_dic["num"] = edges_info_num[edge_id]
                    sub_data["targets"].append(edge_dic)
            data_search(sub_data, level + 1, edges_info, edges_info_num, edges_info_list, edges_control_info, build)
    else:
        data = data["sub_net"]
        if len(data) > 0:
            for sub_data in data:
                # print(sub_data["label"])
                if build:
                    # 将重组的目标节点信息存进nodes中
                    edge_deal(sub_data, edges_info, edges_info_num, edges_info_list, edges_control_info)
                else:
                    # 从nodes中提取回信息放入data中
                    targets_list = list(nodes[sub_data["uid"]]["targets"])
                    sub_data["targets"] = []
                    for target in targets_list:
                        edge_dic = {}
                        edge_id = sub_data["uid"] + "__" + target
                        edge_dic["id"] = target
                        edge_dic["info"] = edges_info[edge_id]
                        edge_dic["control"] = edges_control_info[edge_id]
                        edge_dic["num"] = edges_info_num[edge_id]
                        sub_data["targets"].append(edge_dic)
                data_search(sub_data, level + 1, edges_info, edges_info_num, edges_info_list, edges_control_info, build)


def get_data(graph):
    proxy = Proxy(graph=graph)
    tree = proxy.tree

    """
    # 显示树型结构
    tree_struc = tree.dump()

    f = open('tree_struc.txt', 'w')
    f.write(str(tree_struc))
    f.close()
    """

    # 建立data的层次型数据结构以及nodes的索引型结构
    data = []
    level = 0
    graph = proxy.graph

    # 信息存储，以起始节点+‘__’+目标节点为key进行索引
    # edges_info 保留每条线（或多条）的维度信息
    edges_info = {}
    # edges_info_num 保留重合边的数量
    edges_info_num = {}
    # 保留重合边中的各个起始节点的缩略名，用于去除节点与其虚节点重复计算的问题
    edges_info_list = {}
    edges_control_info = {}
    data_build(tree, graph, data, level)

    # 边的重新构造，存入nodes中
    level = 1
    data_search(data, level, edges_info, edges_info_num, edges_info_list, edges_control_info, build=True)
    # 从nodes中取出边，赋值回data中
    data_search(data, level, edges_info, edges_info_num, edges_info_list, edges_control_info, build=False)

    # 数据存入
    # 放回data
    data = filter(data)
    return data


def filter(g):
    import re
    moudle_base_name = set(["Identity", "Linear", "Bilinear", "LazyLinear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d",
                           "ConvTranspose3d", "LazyConv1d", "LazyConv2d", "LazyConv3d", "LazyConvTranspose1d", "LazyConvTranspose2d",
                           "LazyConvTranspose3d", "Threshold", "ReLU", "Hardtanh", "ReLU6", "Sigmoid", "Tanh", "Softmax", "Softmax2d",
                           "LogSoftmax", "ELU", "SELU", "CELU", "GELU", "Hardshrink", "LeakyReLU", "LogSigmoid", "Softplus", "Softshrink",
                           "MultiheadAttention", "PReLU", "Softsign", "Softmin", "Tanhshrink", "RReLU", "GLU", "Hardsigmoid", "Hardswish",
                           "SiLU", "Mish", "L1Loss", "NLLLoss", "KLDivLoss", "MSELoss", "BCELoss", "BCEWithLogitsLoss", "NLLLoss2d",
                           "CosineEmbeddingLoss", "CTCLoss", "HingeEmbeddingLoss", "MarginRankingLoss", "MultiLabelMarginLoss", "MultiLabelSoftMarginLoss",
                           "MultiMarginLoss", "SmoothL1Loss", "HuberLoss",  "SoftMarginLoss", "CrossEntropyLoss", "TripletMarginLoss", "TripletMarginWithDistanceLoss",
                           "PoissonNLLLoss", "GaussianNLLLoss",
                           "AvgPool1d", "AvgPool2d", "AvgPool3d", "MaxPool1d", "MaxPool2d", "MaxPool3d", "MaxUnpool1d",
                           "MaxUnpool2d", "MaxUnpool3d", "FractionalMaxPool2d", "FractionalMaxPool3d", "LPPool1d", "LPPool2d", "AdaptiveMaxPool1d",
                           "AdaptiveMaxPool2d", "AdaptiveMaxPool3d", "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d", "BatchNorm1d",
                           "BatchNorm2d", "BatchNorm3d", "SyncBatchNorm", "LazyBatchNorm1d", "LazyBatchNorm2d", "LazyBatchNorm3d", "InstanceNorm1d",
                           "InstanceNorm2d", "InstanceNorm3d", "LazyInstanceNorm1d", "LazyInstanceNorm2d", "LazyInstanceNorm3d", "LocalResponseNorm",
                           "CrossMapLRN2d", "LayerNorm", "GroupNorm", "Dropout", "Dropout2d", "Dropout3d", "AlphaDropout", "FeatureAlphaDropout",
                           "ReflectionPad1d", "ReflectionPad2d", "ReflectionPad3d", "ReplicationPad1d", "ReplicationPad2d", "ReplicationPad3d",
                           "ZeroPad2d", "ConstantPad1d", "ConstantPad2d", "ConstantPad3d"])
    base_tree_name = []
    def del_node_sub(node):
        if node["uid"].find("/") != -1:
            start = node["uid"].rindex("/")
            node_name = node["uid"][start+1:]
            node_name = re.sub(r"\[(.*?)\]", "", node_name)
            if node_name in moudle_base_name:
                node["sub_net"] = []
                base_tree_name.append(node["uid"])

        for kid_node in node["sub_net"]:
            del_node_sub(kid_node)


    for top_node in g:
        if top_node["label"].lower() == 'input' or top_node["label"].lower() == "output":
            top_node["sub_net"] = []
        else:
            del_node_sub(top_node)

    def del_node_kid(node):
        for target_node in reversed(node["targets"]):
            if target_node["id"].find("/") != -1:
                start = target_node["id"].rindex("/")
                node_name = target_node["id"][0:start]
                if node_name.find("/") != -1:
                    start = node_name.rindex("/")
                    need_name = node_name[start+1:]
                    need_name = re.sub(r"\[(.*?)\]", "", need_name)
                    if need_name in moudle_base_name:
                        node["targets"].remove(target_node)

        for sub_node in node["sub_net"]:
            del_node_kid(sub_node)


    for top_node in g:
        del_node_kid(top_node)

    new_g = []

    def find_need_node(node):
        for kid_sub_net in node["sub_net"]:
            find_need_node(kid_sub_net)

        for target_node in reversed(node["targets"]):
            for base_name in base_tree_name:
                if target_node["id"] in base_name and target_node["id"] != base_name:
                    node["targets"].remove(target_node)
                    break
        tip = 0
        for base_name in base_tree_name:
            if node["uid"] in base_name and node["uid"] != base_name:
                tip = 1
                break
        if tip == 0:
            new_g.append(node)


    for top_node in g:
        if top_node["label"].lower() == 'input' or top_node["label"].lower() == "output":
            for io_target in reversed(top_node["targets"]):
                for base_name in base_tree_name:
                    if io_target["id"] in base_name and io_target["id"] != base_name:
                        top_node["targets"].remove(io_target)
                        break
            new_g.append(top_node)
        else:
            find_need_node(top_node)

    for need_node in new_g:
        need_node["parent"] = ""
        need_node["uid"] = need_node["uid"].replace("/", "to")
        need_node["layer"] = 1

        for change_name in need_node["targets"]:
            change_name["id"] = change_name["id"].replace("/", "to")
    for s_node in reversed(new_g):
        for i in s_node["targets"]:
            if (("output" in i["id"]) and i["id"]!="output"):
                s_node["targets"].remove(i)
    for need_node in new_g:
        if len(need_node["targets"]) >1:
            for target_node in reversed(need_node["targets"]):
                if target_node["info"] == "":
                    need_node["targets"].remove(target_node)
        if need_node["op"] !="":
            need_node["label"] = need_node["op"].split("::")[1]


    return new_g





    # import copy

    # moudle_name_new = copy.deepcopy(moudle_name)
    # i = 0
    # j = 1
    # while j < len(moudle_name):
    #     if(moudle_name[i] in moudle_name[j]):
    #         moudle_name_new.remove(moudle_name[i])
    #     i+=1
    #     j+=1
    #
    #
    # def del_node(node):
    #     new_name = node["uid"].replace("/", "to")
    #     for node_name in moudle_name_new:
    #
    #         if new_name == node_name:
    #             node["sub_net"] = []
    #             return
    #     for kid_node in node["sub_net"]:
    #         del_node(kid_node)
    #
    # def del_node_kid(node):
    #     for target_node in reversed(node["targets"]):
    #         new_name = target_node["id"].replace("/", "to")
    #         for node_name in moudle_name_new:
    #             if(( node_name in new_name ) and new_name != node_name):
    #                 node["targets"].remove(target_node)
    #                 break
    #
    #     for sub_node in node["sub_net"]:
    #         del_node_kid(sub_node)
    #
    # for top_node in g:
    #     if top_node["label"] == 'input' or top_node["label"] == "output":
    #         top_node["sub_net"] = []
    #     else:
    #         del_node(top_node)
    #
    # for top_node1 in g:
    #     del_node_kid(top_node1)
    #
    # # return g
    #
    # need_del_list = []
    #
    # def find_del_node(node):
    #     if len(node["sub_net"]) != 0:
    #         need_del_list.append(node["uid"])
    #         for kid_sub_net in node["sub_net"]:
    #             find_del_node(kid_sub_net)
    # for top_node in g:
    #     if (top_node["label"] != 'input' and top_node["label"] != "output"):
    #         find_del_node(top_node)
    # new_g = []
    # def find_need_node(node):
    #     for kid_sub_net in node["sub_net"]:
    #         find_need_node(kid_sub_net)
    #         tip = 0
    #         for need_del_name in need_del_list:
    #             if kid_sub_net["uid"] == need_del_name:
    #                 tip = 1
    #         if tip == 0:
    #             new_g.append(kid_sub_net)
    #
    #     for target_node in reversed(node["targets"]):
    #         for need_del_name in need_del_list:
    #             if target_node["id"] == need_del_name:
    #                 node["targets"].remove(target_node)
    #
    #
    # for top_node in g:
    #     if (top_node["label"].lower() == 'input' or top_node["label"].lower() == "output"):
    #         top_node["sub_net"] = []
    #         for target_node in reversed(top_node["targets"]):
    #             for del_name in need_del_list:
    #                 if target_node["id"] == del_name:
    #                     top_node["targets"].remove(target_node)
    #         new_g.append(top_node)
    #     else:
    #         find_need_node(top_node)
    # for need_node in new_g:
    #     need_node["parent"] = ""
    #     need_node["uid"] = need_node["uid"].replace("/", "to")
    #     need_node["layer"] = 1
    #     # need_node["label"] = need_node["uid"]
    #     for change_name in need_node["targets"]:
    #         change_name["id"] = change_name["id"].replace("/", "to")
    # for s_node in reversed(new_g):
    #     for i in s_node["targets"]:
    #         if (("output" in i["id"]) and i["id"]!="output"):
    #             s_node["targets"].remove(i)
    # for need_node in new_g:
    #     if len(need_node["targets"]) >1:
    #         for target_node in reversed(need_node["targets"]):
    #             if target_node["info"] == "":
    #                 need_node["targets"].remove(target_node)
    #     if need_node["op"] !="":
    #         need_node["label"] = need_node["op"].split("::")[1]
    #
    # i = 0
    # link_data = ""
    # while(i < len(new_g)):
    #     if "relu" in new_g[i]["label"].lower() and len(new_g[i]["targets"]) > 1:     #relu多次使用
    #         pass
    #
    #         # for prenode in new_g[i-1]["targets"]:
    #         #     if prenode["id"] == new_g[i]["uid"]:
    #         #         link_data = prenode["info"]
    #         # relu_num = 0
    #         # while(relu_num < new_g[i]["targets"]):
    #         #     new_relu_node = copy.deepcopy(new_g[i])
    #         #     new_relu_node["targets"] = new_g[i]["targets"][relu_num]
    #         #     new_relu_node["uid"] = new_g[i]["uid"]+str(relu_num)
    #         #     if new_g[i]["targets"][relu_num]["info"] == link_data:
    #         #         for prenode in new_g[i-1]["targets"]:
    #         #             if prenode["id"] == new_g[i]["uid"]:
    #         #                 prenode["id"] = new_relu_node["uid"]
    #
    #
    #
    #     i+= 1
    #
    # return new_g
    #
    #




