# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Time : 2020/9/25 20:36 
# @Author : UYPLAYER
# @File : Knapsack problem.py 
# @Software: PyCharm
Change Activity:2020/9/25
-------------------------------------------------
"""

'''
背包问题
https://labuladong.gitbook.io/algo/di-ling-zhang-bi-du-xi-lie/bei-bao-zi-ji
https://www.jianshu.com/p/50af9094a2ac
'''
def pro01(nums):
    Sum = sum(nums)
    if Sum % 2 != 0:
        return False
    amount = Sum // 2
    print(amount)
    dp = [True] + [False for _ in range(1, amount + 1)]
    print(dp)
    for num in nums:
        for i in range(amount, num - 1, -1):
            dp[i] |= dp[i - num]
    return dp[-1]


def p01(nums):
    if not nums:
        return  False
    if sum(nums)%2 !=0:
        return False
    all_sum, N = sum(nums), len(nums)
    half_sum = all_sum//2
    nums.sort()
    flag = [[True]*(half_sum+1) for _ in range(N)]
    print(flag)
    for i in range(N):
        for j in range(nums[i],half_sum+1):
            if j == nums[i]:
                flag[i][j] = True
            else:
                flag[i][j] = flag[i - 1][j] or flag[i - 1][j - nums[i]]
            return flag[-1][-1]

def all01(amount,coins):
    n = len(coins)
    dp = [0] * (amount+1)
    dp[0] = 1
    for coin in coins:
        for j in range(coin,amount+1):
            dp[j] = dp[j] + dp[j - coin]
    print(dp)
    return dp[amount]

# li = [1, 5, 11, 5]
# print(p01(li))

print(all01(5,[1,2,5]))