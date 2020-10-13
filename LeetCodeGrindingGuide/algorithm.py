#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/8 9:41
# @Author  : uyplayer
# @Site    : uyplayer.xyz
# @File    : algorithm.py
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
---------------------------
从左端遍历的结果：[1, 2, 3, 1, 1, 1, 1]
从有段开始遍历的结果：[1, 1, 1, 1, 3, 2, 1]
max：[1,2,3,1,3,2,1]
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

'''
 无重叠区间
https://leetcode-cn.com/problems/non-overlapping-intervals
--------------------------------------------------------
 [ [1,2], [2,3], [3,4], [1,3] ]
 [ [1,2], [1,2], [1,2] ]
 [ [1,2], [2,3] ]
 [[1,100],[11,22],[1,11],[2,12]]
 首先我们用sort来进行排序，[[1,100],[11,22],[1,11],[2,12]] -》 [[1, 11], [1, 100], [2, 12], [11, 22]]
 问题来了 [1,11]和[1,100] 对比的时候要去掉哪一个？ 
 去掉[1,11]的话：[1, 100], [2, 12], [11, 22]-》去掉[2, 12], [11, 22]，共三次
 如果去掉[1, 100]的话：[[1, 11], [2, 12], [11, 22]]-》去掉[2, 12]，共二四
 从上面可以知道我们选择的时候选择小区间
 
 对下面的数据结果不对
 ---------------------------------------------------------------------
 [[-100,-87],[-99,-44],[-98,-19],[-97,-33],[-96,-60],[-95,-17],[-94,-44],[-93,-9],[-92,-63],[-91,-76],[-90,-44],[-89,-18],[-88,10],[-87,-39],[-86,7],[-85,-76],[-84,-51],[-83,-48],[-82,-36],[-81,-63],[-80,-71],[-79,-4],[-78,-63],[-77,-14],[-76,-10],[-75,-36],[-74,31],[-73,11],[-72,-50],[-71,-30],[-70,33],[-69,-37],[-68,-50],[-67,6],[-66,-50],[-65,-26],[-64,21],[-63,-8],[-62,23],[-61,-34],[-60,13],[-59,19],[-58,41],[-57,-15],[-56,35],[-55,-4],[-54,-20],[-53,44],[-52,48],[-51,12],[-50,-43],[-49,10],[-48,-34],[-47,3],[-46,28],[-45,51],[-44,-14],[-43,59],[-42,-6],[-41,-32],[-40,-12],[-39,33],[-38,17],[-37,-7],[-36,-29],[-35,24],[-34,49],[-33,-19],[-32,2],[-31,8],[-30,74],[-29,58],[-28,13],[-27,-8],[-26,45],[-25,-5],[-24,45],[-23,19],[-22,9],[-21,54],[-20,1],[-19,81],[-18,17],[-17,-10],[-16,7],[-15,86],[-14,-3],[-13,-3],[-12,45],[-11,93],[-10,84],[-9,20],[-8,3],[-7,81],[-6,52],[-5,67],[-4,18],[-3,40],[-2,42],[-1,49],[0,7],[1,104],[2,79],[3,37],[4,47],[5,69],[6,89],[7,110],[8,108],[9,19],[10,25],[11,48],[12,63],[13,94],[14,55],[15,119],[16,64],[17,122],[18,92],[19,37],[20,86],[21,84],[22,122],[23,37],[24,125],[25,99],[26,45],[27,63],[28,40],[29,97],[30,78],[31,102],[32,120],[33,91],[34,107],[35,62],[36,137],[37,55],[38,115],[39,46],[40,136],[41,78],[42,86],[43,106],[44,66],[45,141],[46,92],[47,132],[48,89],[49,61],[50,128],[51,155],[52,153],[53,78],[54,114],[55,84],[56,151],[57,123],[58,69],[59,91],[60,89],[61,73],[62,81],[63,139],[64,108],[65,165],[66,92],[67,117],[68,140],[69,109],[70,102],[71,171],[72,141],[73,117],[74,124],[75,171],[76,132],[77,142],[78,107],[79,132],[80,171],[81,104],[82,160],[83,128],[84,137],[85,176],[86,188],[87,178],[88,117],[89,115],[90,140],[91,165],[92,133],[93,114],[94,125],[95,135],[96,144],[97,114],[98,183],[99,157]]
 --------------------------------------------------------------------
 其实下面的方面代码有个漏洞，intervals.sort()，我们需要针对第二个key进行排序
 def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        n = len(intervals)
        r = 0
        if n == 1 or n==0:
            return 0
        # 先对取键进行排序
        intervals.sort()
        tag = intervals[0] 
        for i in range(1,n):
            if tag == intervals[i]:
                r += 1
                continue
            if intervals[i][0] < tag[1]:
                r += 1
            else:
                tag = intervals[i]
            if tag[1] - tag[0] > intervals[i][1] - intervals[i][0]:
                tag = intervals[i]
        return r
'''
def eraseOverlapIntervals(intervals):
    n = len(intervals)
    if n == 0:return 0
    # 根据区间右端点排序
    intervals = sorted(intervals,key = lambda x: x[1])
    # intervals.sort() 这个排序按照key【0】 来排序，比如 [[1,100],[11,22],[1,11],[2,12]] -》 [[1, 11], [1, 100], [2, 12], [11, 22]]
    # 用sorted函数对key【1】排序：[[1, 11], [2, 12], [11, 22], [1, 100]]
    num = 0
    new = intervals [0]
    for interval in intervals[1:]:
        if interval[0] < new[1]:
            num += 1
        else:
            new = interval
    print(num)


'''
https://leetcode-cn.com/problems/can-place-flowers/submissions/
种花问题，1：已经有花，0：空地，
下面的代码能优化吗
def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
    # 1:已经中了,0:空地方,我们现在做的是还没中的地方0上中这个花;
    if len(flowerbed) == 1 and n == 1 and flowerbed[0] == 0:
        return True
    if len(flowerbed) == 1 and n == 1:
        return False
    total = 0
    if flowerbed[:2] == [0, 0]:
        total += 1
        flowerbed[0] = 1
    i = 1
    while i < len(flowerbed) - 1:
        if flowerbed[i] == 1:
            i += 1
            continue
        if flowerbed[i - 1] == 0 and flowerbed[i + 1] == 0:
            total += 1
            flowerbed[i] = 1
        i += 1
    if flowerbed[-2:] == [0, 0]:
        total += 1
        flowerbed[-1] = 1
    if n <= total:
        return True
    else:
        return False
'''
# 优化版本
def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
    # 1:已经中了,0:空地方,我们现在做的是还没中的地方0上中这个花;
    # 两个段插入0
    flowerbed = [0] + flowerbed
    flowerbed = flowerbed + [0]
    total = 1
    i = 1
    while i < len(flowerbed) - 1:
        if flowerbed[i - 1] == 0 and flowerbed[i + 1] == 0 and flowerbed[i] == 0:
            total += 1
            flowerbed[i] = 1
        i += 1
    if n <= total:
        return True
    else:
        return False

'''
 用最少数量的箭引爆气球
 https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/
 至少需要多少个射箭呢？
 如果他们坐标又交集，那就需要一个射箭
'''
def findMinArrowShots(points):
    n = len(points)
    if n==0:
        return 0
    points = sorted(points,key = lambda x: x[1])
    arrow = 1
    change_point = points[0][1]
    for x_start,x_end in points:
        if change_point < x_start:
            arrow += 1
            change_point = x_end
    return arrow

'''
划分字母区间
https://leetcode-cn.com/problems/partition-labels/
解释：题目要求对字符串进行分段，前提是分段里相互不重复字母，就是一个分段里字母不再出现其他分段里
a	b	a	b	c	b	a	c	a	d	e	f	e	g	d	e	h	i	j	h	k	l	i	j
a									d							h							
a	b								d	e						h	i						
a	b	a							d	e	f					h	i	j					
a	b	a	b						d	e	f	e				h	i	j	h				
a	b	a	b	c					d	e	f	e	g			h	i	j	h	k			
a	b	a	b	c	a				d	e	f	e	g	d		h	i	j	h	k	l		
a	b	a	b	c	a	c			d	e	f	e	g	d	e	h	i	j	h	k	l	i	
a	b	a	b	c	a	c	a									h	i	j	h	k	l	i	j

'''
# 方法
def partitionLabels():
    S = 'ababcbacadefegdehijhklij'
    n = len(S)
    all = [0]
    i = 1
    while i < n:
        ch = list(S[sum(all):i])
        tem_ch = ch[:]
        flag = True
        while ch:
            test_ch = ch.pop()
            if test_ch in S[i:]:
                flag = False
                break
        if flag and len(ch)==0:
            all.append(len(tem_ch))
        i += 1
    all.append(n-sum(all))
    print(all[1:])
# 力扣上的一个解题方法
def partitionLabels_lecode(S):

    start = 0
    result = []
    while start < len(S):
        end = S.rfind(S[start]) # 找到S[start]字符最后的出现位置
        set_point = set(S[start:end + 1]) # 去掉重负的字符
        run = max([S.rfind(k) for k in set_point]) # 记录所有的没有重复字符的最大位置
        while run > end:
            end = run
            set_point = set(S[start:end + 1])
            run = max([S.rfind(k) for k in set_point])
        result.append(end - start + 1)
        start = end + 1
    return result

'''
买卖股票的最佳时机 II
https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/
'''
def maxProfit(prices):
    profit = 0
    for i in range(1,len(prices)):
        tmp = prices[i] - prices[i-1]
        if tmp > 0:
            profit += tmp
    return profit

'''
根据身高重建队列
https://leetcode-cn.com/problems/queue-reconstruction-by-height/
按照（h,k)进行排序:想象每一个人再队列中位置就是相同的高度h下，他们对应的位置是index=k；如果不相同的高度呢
'''
# def reconstructQueue(people):
#     n = len(people)
#     lo = [True] * n
#     people = sorted(people,key=lambda x:x[0])
#     print(people)
#     people = sorted(people, key=lambda x: x[1])
#     print(people)
#     print(lo)
#     i = 0
#     while i < n and lo[i]:
#         line = people[i][1]
#         count = 0
#         for j in range(i,n):
#             if people[i][0] > people[j][0]:
#                 count += 1
#             if line == count:
#                 people[i],people[j] = people[j],people[i]
#                 lo[i] = False
#                 break
#         i += 1
#     print(people)
def reconstructQueue(people,answ):
    tmp = people[:]
    people.sort(key=lambda x: (-x[0], x[1])) # 先按h降序，再按k升序
    output = []
    for proson in people:
        output.insert(proson[1],proson)
    return output

'''
非递减数列
https://leetcode-cn.com/problems/non-decreasing-array/
'''
def checkPossibility(nums):
    count = 0
    for i in range(1,len(nums)):
        if nums[i] < nums[i-1]:
            if i-1>0 and i+1 < len(nums):
                if nums[i+1] < nums[i-1] and nums[i] < nums[i-2]:
                    count += 1
    return count<=1

'''
两数之和 II - 输入有序数组
https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted
解释：对于一个有排序的序列数组求两个数，并它们两个的和等于给定的值。我我们当初想问的时候突然想起来两两组合问题，但是这样的话性能不好并我们不会得到我们想要那种快速解决这个问题的方法；
所有我们想出来的双向指针方案：我们同时从左端和右端开始搜索，它们两个分别l和r，因此我们数组是从小大排序好的，l指向左端最小值，r
指向右端最大值。开始遍历两个值加起来后的值小于我们目标值的话，我们把左边的指针往右移动，如果两个值的和大于我们目标值的话，那么我们把右边的指针的往左边移动，直到找到我们想要的合适的值
'''
def twoSum(numbers,target):
    n = len(numbers)
    l = 0
    r = n - 1
    sum = 0
    while l < r:
        sum = numbers[l] + numbers[r]
        if sum == target:
            break
        if sum < target:
            l = l + 1
        else:
            r = r - 1
    return [l + 1, r + 1]
'''
合并两个有序数组
https://leetcode-cn.com/problems/merge-sorted-array/
'''
# 下面的代码对 但是无法通过力扣
# def merge(nums1, m, nums2, n):
#     nums1 = [i for i in nums1 if i != 0]
#     nums2 = [i for i in nums2 if i != 0]
#     num = nums1+nums2
#     num.sort()
#     return num
#-------------------------------------、
# 下面的代码正确
def merge(nums1, m, nums2, n):
    nums1[:] = sorted(nums1[:m] + nums2)
    return nums1
print(merge([1,2,3,0,0,0],3,[2,5,6],3))

'''
环形链表 II
https://leetcode-cn.com/problems/linked-list-cycle-ii/
求这个链表的循环的开始位置：解决办法是设计两个指针，一个快，一个慢，就是快慢指针；快直接走的比慢指针快一些，当两个指针等于的时候就是说明，我们这个数据中存在环；
然后我们怎么求这个环的开始位置呢？当我们通过循环找到以后fast == slow 以后while 推出，我们设 fast = head 重新赋值；再一次循环当他们两个等于的时候就等于fast 找到这个循环入口
'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def detectCycle(head):
        fast = head
        slow = head
        while True:
            if not fast or not fast.next:
                return
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                break
        # 返回入环的第一个点
        fast = head
        while fast != slow:
            fast, slow = fast.next, slow.next
        return fast
