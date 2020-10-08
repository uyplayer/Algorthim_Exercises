'''
二分查询
https://labuladong.gitbook.io/algo/di-ling-zhang-bi-du-xi-lie/er-fen-cha-zhao-xiang-jie
'''

def binarySearch(nums,target):
    all = []
    left = 0
    right  = len(nums) -1
    while left <= right :
        mid = left + int((right - left) / 2)
        if nums[mid] == target :
            return  mid
        elif nums[mid] < target:
            left = mid +1
        elif nums[mid] > target:
            right = mid + 1
    return  -1
s = [5,7,7,8,8,10]
t = 8
print(binarySearch(s,t))