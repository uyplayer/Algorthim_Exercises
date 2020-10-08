#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/6 10:06
# @Author  : uyplayer
# @Site    : uyplayer.xyz
# @File    : wanquanbeibao.py
# @Software: PyCharm

'''
完全背包问题
'''

def change( amount, coins):
    length = len(coins)
    if length == 0:
        return 0
    dp = [0] * (amount + 1)
    dp[0] = 1
    for i in range(length):
        for j in range(1, amount + 1):
            if j - coins[i] >= 0:
                dp[j] = dp[j] + dp[j - coins[i]]
    return dp[amount]
