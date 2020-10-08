#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/5 15:16
# @Author  : uyplayer
# @Site    : uyplayer.xyz
# @File    : zuixiaocharuhuiwen.py
# @Software: PyCharm
'''
构造回文的最小插入次数
'''
# 用二位数组
def minInsertions(s):
    n = len(s)
    dp = [[0]*n]*n
    for i in range(n-2)[::-1]:
        for j in range(i+1):
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1]
            else:
                dp[i][j] =  min(dp[i + 1][j], dp[i][j - 1]) + 1
    return dp[0][n - 1]

# 用一位数组
def minInsertions1(self, s: str) -> int:
    size = len(s)
    if size <= 1: return 0

    dp = [0] * size

    for i in range(size - 2, -1, -1):
        pre = 0
        for j in range(i + 1, size):
            tmp = dp[j]
            if s[i] == s[j]:
                dp[j] = pre
            else:
                dp[j] = min(dp[j], dp[j - 1]) + 1
            pre = tmp
    return dp[-1]
