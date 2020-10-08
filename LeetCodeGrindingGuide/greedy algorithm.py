#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/8 9:41
# @Author  : uyplayer
# @Site    : uyplayer.xyz
# @File    : greedy algorithm.py
# @Software: PyCharm

'''
匹配饼干问题
https://leetcode.com/problems/assign-cookies/
吃饼干的过程：
首先对孩子和饼干进行排序，这题目要求是求最大的解，说明一个局部最优解到全局最优解的过程
一个孩子吃饱等于他得吃自己的吃饱度或者比吃饱度大的饼干，为了达到最优解，我们得看base case，
如果一个孩子吃了一个饼干这个等于他没有才一次吃的机会，并且他吃的饼干再也不能用了，
如果一个孩子吃了一个饼干，并且这个饼干大于他的吃饱度，那问题来了，后面的孩子吃饱度等于这个已经被吃的饼干的时候，他们吃什么？这个现显然不是最优解了
'''
def findContentChildren(children,cookies):
    children.sort()
    cookies.sort()
    child = 0
    cookie = 0
    while child < len(children) and cookie < len(cookies):
        if children[child] <= cookies[cookie]:child += 1
        cookie += 1
    return child

'''
分配糖的问题
https://leetcode-cn.com/problems/candy/
每一个孩子至少一个糖？意思是无论还是score多少就初始的时候有一个糖,这个问题思路是首先给孩子分配糖，每一个人一个，然后如果一个孩子的score大于左边的孩子，那就这个孩子再给糖，这样情况下，score大的孩子的糖比右边score
小的孩子的糖大得多
题目要求是：老师至少需要准备多少颗糖果呢？
是不是一个优解问题？

看下下面的代码：
def candy(self, ratings: List[int]) -> int:
        candies =[1]+[1]*len(ratings)+[1]
        ratings_inf = [float('inf')]+ratings+[float('inf')]
        for i in range(1,len(ratings)+1):
            if ratings_inf[i]>ratings_inf[i-1]:
                candies[i] = candies[i-1]+1
            if ratings_inf[i]>ratings_inf[i+1]:
                candies[i] = candies[i+1]+1
        candies.pop(0)
        candies.pop(len(candies)-1)
        return sum(candies)
这个代码哪里有错误呢？
[1,2,87,87,87,2,1]，这个情况下，它从1一开始比较；【1，2，3，1，1，2，1】 ,但是最后一个87明显比2大，但是为什么结果87-》1，2-》2 呢？
所有我们从左边开始比较一遍，右边开始比较一边才对
-----------------------------------------------
candies[i - 1] = max(candies[i - 1], candies[i] + 1)
candies[i - 1] = candies[i] + 1
这两个行又不同的结果
我们先考虑两个循环中结果保存到两个不同candies1和candies2中，然后针对每一个值比较candies=max（candies1【i】，candies2【i】）这个结果也是对的
'''
def candy(ratings):
    candies = [1] * len(ratings)
    for i in range(1, len(ratings)):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1
    for i in range(len(ratings) - 1, 0, -1):
        if ratings[i] < ratings[i - 1]:
            candies[i - 1] = max(candies[i - 1], candies[i] + 1)
            # candies[i - 1] = candies[i] + 1
    return sum(candies)

