# https://labuladong.gitbook.io/algo/di-ling-zhang-bi-du-xi-lie/dong-tai-gui-hua-xiang-jie-jin-jie
'''动态规划问题'''


# 首先来一个递归看看它的效率
def fo(n):
    if n == 1 or n == 2:
        return 1
    return fo(n - 1) + fo(n - 2)


# print(fo(10))

'''动态规划'''
def fib(n):
    if n < 1:
        return 0
    memo = [0 for _ in range(n+1)] # memo 初始化为0
    return  helper(memo,n)
def helper(memo,n):
    if n ==1 or n ==2 :
        return 1
    if memo[n] != 0: # 如不是0，那说明已经求解了
        return  memo[n]
    memo[n] = helper(memo,n-1) + helper(memo,n-2)
    return  memo[n]
#上面简化办
def fibb(n):   #每一个元素进行状态变化 f(n) = f(n-1) + f(n-2)
    dp = [0 for _ in range(n+1)]
    dp[1]=dp[2] = 1
    for i in range(1,n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return  dp[n]

def fibbb(n):   #状态变化  pre = curr , cur = sum
    if n == 1 or n ==2 :
        return 1
    pre = 1
    curr = 1
    for i in range(3,n+1):
        sum = pre + curr
        pre = curr
        curr = sum
    return  curr

# 硬币问题

# def coinChange(coins,amount):
#     def dp(n):
#         for coin in coins:
#             res = min(res , 1+dp(n - coin))
#         return  res
#     return dp(amount)

# def coinChange(coins,amount):
#     def dp(n):
#         if n == 0:
#             return 0
#         if n<0:
#             return -1
#         res = float('INF') # 非常大的一个数字
#          for coin in coins:
#              sub = dp(n-coin)
#              if sub == -1: continue
#              res = min(res,sub+1)
#         if res != float('INF'):
#             return res
#         else:
#             return -1
#
#     return dp(res)

# 使用带备忘录的递归来解决硬币问题
# def coinChange(coins, amount):
#     memo = dict()
#     def dp(n):
#         if n in memo: return memo[n]
#         if n == 0: return  0
#         if n<0 : return -1
#         res = float('INF')
#         for coin in coins:
#             sub =dp(n-coin)
#             if sub == -1:continue
#             res = min(res,sub+1)
#         if res != float('INF'):
#             memo[n] = res
#         else:
#             memo[n] = -1
#     return dp(amount)

# dp 数组的迭代解法
# def coinChange(coins, amount):
#     dp = [amount+1 for i in range(amount+1)]
#     dp[0] = 0
#     for i in range(amount):
#         for coin in coins:
#             if (i - coin < 0):continue
#             dp[i] = min(dp[i], 1 + dp[i - coin]);
#     return (dp[amount] == amount + 1) ? -1: dp[amount];


