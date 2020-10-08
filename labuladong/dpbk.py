#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/5 16:23
# @Author  : uyplayer
# @Site    : uyplayer.xyz
# @File    : dpbk.py
# @Software: PyCharm
'''
动态规划和回溯问题
'''


# 动态规划问题
def findTargetSumWays(nums,S):
    memo = {}
    le = len(nums)
    def dfs(i,res):
        if i == le:
            if res == S:
                return 1
            else:
                return 0
        if (i, res) in memo:
            return memo[(i, res)]
        res = dfs(i + 1, res + nums[i]) + dfs(i + 1, res - nums[i])
        memo[(i, res)] = res
        return res
    return dfs(0, 0)