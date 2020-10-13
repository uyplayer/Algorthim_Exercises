#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/8 11:50
# @Author  : uyplayer
# @Site    : uyplayer.xyz
# @File    : test.py
# @Software: PyCharm
def eraseOverlapIntervals():
    intervals = [[-100,-87],[-99,-44],[-98,-19],[-97,-33],[-96,-60],[-95,-17],[-94,-44],[-93,-9],[-92,-63],[-91,-76],[-90,-44],[-89,-18],[-88,10],[-87,-39],[-86,7],[-85,-76],[-84,-51],[-83,-48],[-82,-36],[-81,-63],[-80,-71],[-79,-4],[-78,-63],[-77,-14],[-76,-10],[-75,-36],[-74,31],[-73,11],[-72,-50],[-71,-30],[-70,33],[-69,-37],[-68,-50],[-67,6],[-66,-50],[-65,-26],[-64,21],[-63,-8],[-62,23],[-61,-34],[-60,13],[-59,19],[-58,41],[-57,-15],[-56,35],[-55,-4],[-54,-20],[-53,44],[-52,48],[-51,12],[-50,-43],[-49,10],[-48,-34],[-47,3],[-46,28],[-45,51],[-44,-14],[-43,59],[-42,-6],[-41,-32],[-40,-12],[-39,33],[-38,17],[-37,-7],[-36,-29],[-35,24],[-34,49],[-33,-19],[-32,2],[-31,8],[-30,74],[-29,58],[-28,13],[-27,-8],[-26,45],[-25,-5],[-24,45],[-23,19],[-22,9],[-21,54],[-20,1],[-19,81],[-18,17],[-17,-10],[-16,7],[-15,86],[-14,-3],[-13,-3],[-12,45],[-11,93],[-10,84],[-9,20],[-8,3],[-7,81],[-6,52],[-5,67],[-4,18],[-3,40],[-2,42],[-1,49],[0,7],[1,104],[2,79],[3,37],[4,47],[5,69],[6,89],[7,110],[8,108],[9,19],[10,25],[11,48],[12,63],[13,94],[14,55],[15,119],[16,64],[17,122],[18,92],[19,37],[20,86],[21,84],[22,122],[23,37],[24,125],[25,99],[26,45],[27,63],[28,40],[29,97],[30,78],[31,102],[32,120],[33,91],[34,107],[35,62],[36,137],[37,55],[38,115],[39,46],[40,136],[41,78],[42,86],[43,106],[44,66],[45,141],[46,92],[47,132],[48,89],[49,61],[50,128],[51,155],[52,153],[53,78],[54,114],[55,84],[56,151],[57,123],[58,69],[59,91],[60,89],[61,73],[62,81],[63,139],[64,108],[65,165],[66,92],[67,117],[68,140],[69,109],[70,102],[71,171],[72,141],[73,117],[74,124],[75,171],[76,132],[77,142],[78,107],[79,132],[80,171],[81,104],[82,160],[83,128],[84,137],[85,176],[86,188],[87,178],[88,117],[89,115],[90,140],[91,165],[92,133],[93,114],[94,125],[95,135],[96,144],[97,114],[98,183],[99,157]]
    n = len(intervals)
    r = 0
    if n == 1 or n == 0:
        return 0
    # 先对取键进行排序
    intervals.sort()
    print(intervals)
    tag = intervals[0]
    for i in range(1, n):
        if tag == intervals[i]:
            r += 1
            continue
        if intervals[i][0] < tag[1]:
            r += 1
        else:
            tag = intervals[i]
        if tag[1] - tag[0] > intervals[i][1] - intervals[i][0]:
            tag = intervals[i]
    print(r)
    # n = len(intervals)
    # if n == 1 or n == 0:return 0
    # intervals.sort()
    # total = 0
    # pre = intervals[0][1]
    # for i in range(1,n):
    #     if intervals[i][0] < pre:
    #         total += 1
    #     else:
    #         pre = intervals[i][1]
    # print(total)

# eraseOverlapIntervals()


def doc_select():
    ids = input()
    if ids == 'd1':
        print("I like to watch the sun set with my freidn")
    elif ids == 'd1':
        print("The best place to watch the sunset")
    elif ids == 'd1':
        print("My friend watch the sunset come out")
    else:
        pass

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
# partitionLabels()

# def reconstructQueue(people,answ):
#     n = len(people)
#     tmp = people[:]
#     i = 0
#     while i < n:
#         line = people[i][1]
#         count = 0
#         for j in range(i,n):
#             if people[i][0] > people[j][0]:
#                 count += 1
#                 print(people[i])
#                 print(count)
#             if line == count:
#                 people[i],people[j] = people[j],people[i]
#                 break
#         i += 1
#     print(people==answ)
#     print("tmp:",tmp)
#     print("answ:",answ)
#     print("Output:",people)
def reconstructQueue(people,answ):
    tmp = people[:]
    people.sort(key=lambda x: (-x[0], x[1])) # 先按h降序，再按k升序
    output = []
    for proson in people:
        output.insert(proson[1],proson)
    print(answ == output)
    print("tmp:",tmp)
    print("answ:",output)
# reconstructQueue([[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]],[[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]])
def checkPossibility(nums):
    count = 0
    for i in range(1, len(nums)):
        if nums[i] < nums[i - 1]:
            count += 1
            if i + 1 < len(nums) and i - 2 >= 0:
                if nums[i + 1] < nums[i - 1] and nums[i - 2] > nums[i]:
                    return False
        if count > 1:
            return False
    return True
# print(checkPossibility([4,2,1]))
def merge(nums1, m, nums2, n):
    nums1 = [i for i in nums1 if i != 0]
    nums2 = [i for i in nums2 if i != 0]
    nums2 = nums1+nums2
    nums2.sort()
    return nums2

# print(merge([1,2,3,0,0,0],3,[2,5,6],3))

def minWindow(s,t):
    import copy
    position = 0
    sub_str = []
    temp_t = list(t)
    temp_t.sort()
    temp_t = "".join(temp_t)
    while position < len(s):
        for i in range(position,len(s)):
            strip_s = s[position:i]
            temp_strip_s = strip_s[:]
            temp_strip_s = list(temp_strip_s)
            temp_strip_s.sort()
            temp_strip_s = "".join(temp_strip_s)
            print(strip_s)
            if temp_t in temp_strip_s:
                sub_str.append(strip_s)
                position += 1
                break

    return sub_str
print(minWindow("ADOBECODEBANC","ABC"))













