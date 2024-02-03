

#longest abcbefg'
# Stack abbcde


def longestLength(input): 
    inputArr = list(input)
    print(inputArr)
    max_count = 0
    for i in range(0, len(inputArr)): 
        first_char = inputArr[i]
        count = 1
        
        for j in range(i + 1, len(inputArr)): 
             print(count)
             if first_char != inputArr[j]: 
                count +=1
             else :
                count = 0 
               
        if max_count < count :
            max_count = count
    return max_count

print(longestLength('abccc'))