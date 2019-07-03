# -*- coding:utf-8 -*-
"""
   File Name：     weights_feature_log.py
   Description :   打印训练过程中每一层的权重、偏置、特征参数信息
   Author :        royce.mao
   date：          2019/07/03
"""
import torch
import pandas as pd
import numpy as np
import torchvision.models as models


def weights_bias_parm(net):
    """
    网络的权重偏置信息（日志）
    :param net: 
    :return: 
    """
    parm = {}
    for name,parameters in net.named_parameters():
        # print(name,':',parameters.size())
        parm[name] = parameters

    return parm

def parm_to_excel(excel_name,key_name,parm):
    """
    指定某key层的权重偏置信息写入excel（日志）
    :param excel_name: 
    :param key_name: 
    :param parm: 
    :return: 
    """
    with pd.ExcelWriter(excel_name) as writer:
        # print(parm[key_name].shape)
        output_num,input_num,filter_size,_ = parm[key_name].size()
        for i in range(output_num):
            for j in range(input_num):
                data=pd.DataFrame(parm[key_name][i,j,:,:].detach().cpu().numpy())
                # print(data)
                data.to_excel(writer,index=False,header=True,startrow=i*(filter_size+1),startcol=j*filter_size)
