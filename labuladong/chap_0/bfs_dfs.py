'''
https://labuladong.gitbook.io/algo/di-ling-zhang-bi-du-xi-lie/bfs-kuang-jia
'''

# 二叉树的最小高度
def tree(root):
    if not root:
        return 0
    queue = []
    queue.append(root)
    depth = 1
    while queue:
        le =len(queue)
        for i in range(le):
            cur = queue.pop(0)
            if cur.left == None and cur.right == None:
                return depth
            if cur.left != None:
                queue.append(cur.left)
            if cur.right != None:
                queue.append(cur.right)
        depth += 1
    return depth

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        queue = [root]
        depth = 1
        while queue:
            le = len(queue)
            for i in range(le):
                cur = queue.pop(0)
                if cur.left == None and cur.right == None: # 一旦到叶子节点就结束
                    return depth
                if  cur.left != None:
                    queue.append(cur.left)
                if cur.right != None:
                    queue.append(cur.right)
            depth += 1

        return  depth


# 密码问题









