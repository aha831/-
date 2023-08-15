'''
存放/记录/整理 自己前前后后学习到的各种轮子，主要用于面试，偶尔用于跑着用
'''




def hungarian(matrix):
    '''
    手撕匈牙利算法(m->n匹配)
    input:
        matrix: List[List[m], n], (m >= n, n个任务m个工人/n个真实目标m个先验框)<--开始是这样想的, 后来发现无所谓, m和n谁大都行
    output:
        res: List[min(m, n)]
    '''
    
    zero_mask = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    
    for i in range(len(matrix)):
        min_val = min(matrix[i])
        for j in range(len(matrix[0])):
            matrix[i][j] -= min_val
    
    while True:
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if not matrix[i][j] and (not any(row[j] for row in zero_mask)) and (not any(zero_mask[i])):
                    zero_mask[i][j] = 1
        
        if sum(sum(zero_line) for zero_line in zero_mask) == len(matrix): break
        
        min_val = min(matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix[0])) if not any(row[j] for row in zero_mask) and not any(zero_mask[i]))
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if not any(row[j] for row in zero_mask) and not any(zero_mask[i]):
                    matrix[i][j] -= min_val
                elif zero_mask[i][j] == 1:
                    matrix[i][j] += min_val
        
        zero_mask = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    
    res = [(i, j) for i in range(len(matrix)) for j in range(len(matrix[0])) if zero_mask[i][j] == 1]
    
    return res




if __name__ == '__main__':
    # test hungarian
    cost_matrix = [[10, 9, 1], [10, 1, 9], [1, 10, 9]]
    print(hungarian(cost_matrix))
    
    # verify hungarian
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    print("Assigned Rows:", row_ind)
    print("Assigned Columns:", col_ind)

