# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Time : 2020/9/24 17:55 
# @Author : UYPLAYER
# @File : nsum.py 
# @Software: PyCharm
Change Activity:2020/9/24
-------------------------------------------------
"""
def tar(nums,target):
    # 两边开始找
    j = len(nums) - 1
    i = 0
    while i< j:
        sum = nums[i] + nums[j]
        if sum > target:
            i = i+1
        elif sum < target:
            j =j-1
        elif sum == target:
            return [nums[i],nums[j]]
# def tarr(nums,target):
#     # 两边开始找
#     j = len(nums) - 1
#     i = 0
#     all
#     while i < j:
#         sum = nums[i] + nums[j]
#         if sum > target:
#             i = i + 1
#         elif sum < target:
#             j = j - 1
#         elif sum == target:
#             all.append([nums[i], nums[j]])
#             i = i + 1
#             j = j - 1

def threeSum(nums,target):
    nums = sorted(nums)
    res = []
    for i in range(len(nums)-2):
        if nums[i] >0:
            break
        if i >0 and nums[i] == nums[i-1]:
            continue
        l,r = i+1,len(nums)-1
        while l < r:
            s = nums[i] + nums[l] +nums[r]
            if s<0:
                l += 1
            elif s>0:
                r += 1
            else:
                res.append([[nums[i], nums[l], nums[r]]])
                while l<r and nums[l] == nums[l + 1]:
                    l += 1
                while l < r and nums[r] == nums[r - 1]:
                    r += 1
                l += 1
                r -+ 1
    return  res













# nums = [1,3,1,2,2,3]
# target = 4
# print(tar(nums,target))

li = [-1, 0, 1, 2, -1, -4]
t = 0
print(threeSum(li,t))