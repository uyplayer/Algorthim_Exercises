#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/5 10:48
# @Author  : uyplayer
# @Site    : uyplayer.xyz
# @File    : baseic_framework.py
# @Software: PyCharm

'''
int fib(int N) {
    if (N < 1) return 0;
    // 备忘录全初始化为 0
    vector<int> memo(N + 1, 0);
    // 进行带备忘录的递归
    return helper(memo, N);
}

int helper(vector<int>& memo, int n) {
    // base case
    if (n == 1 || n == 2) return 1;
    // 已经计算过
    if (memo[n] != 0) return memo[n];
    memo[n] = helper(memo, n - 1) + helper(memo, n - 2);
    return memo[n];
}
'''

def fib(N):
    if N<1: return 0
    memo = [0] * (N+1)
    return  helper(memo,N)
def helper(memo,N):
    # base case
    if N == 1 or N==2:return  1
    if memo[N] != 0: return memo[N]
    memo[N] = helper(memo,N-1) + helper(memo,N-2)
    return memo[N]

# 迭代时解决问题
def fib1(N):
    memo = [0] * (N+1)
    memo[1],memo[2] = 1,1
    for i in range(3,N+1):
        memo[i] = memo[i-1] + memo[i-2]
    return memo[N]
#降低空间复杂
def fib2(N):
    if N==1 or N==2:return  1
    pre = 1
    cur = 1
    for i in range(3,N+1):
        s = pre+cur
        pre = cur
        cur = s
    return cur


# 凑出硬币（递归方法）
def Changecoins(coins,amount):

    def dp(n):
        if n==0:return 0
        if n<0:return -1
        # 求最小值，所以初始化为正无穷
        res = float('INF')
        for coin in coins:
            sub = dp(n-coin)
            if sub == -1:continue
            res = min(res,1+sub)
        return res if res != float('INF') else -1
    return dp(amount)

# 凑出硬币（带备忘录的递归）
def Changecoins1(coins,amount):
    # 备忘录
    memo = dict()
    def dp(n):
        if n in memo:return memo[n]
        if n==0:return 0
        if n<0:return -1
        res = float("INF")
        for coin in coins:
            sub = dp(n-coin)
            if sub == -1:continue
            res = min(res,1+sub)
        return res if res != float('INF') else -1
    return dp(amount)

# 数组的迭代解法
def Changecoins2(coins,amount):
    dp = [amount+1]*(amount+1)
    dp[0] = 0
    for i in range(len(dp)):
        for coin in coins:
            if i-coin <0:continue
            dp[i] = min(dp[i], 1 + dp[i - coin])
    return -1 if (dp[amount] == amount + 1) else dp[amount]
