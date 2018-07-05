python_string="I am a student in RUC. I Like playing basketball"
word_list =python_string.split(' ')


dictionary = {}
for word in word_list:
    if not word in dictionary:
        dictionary[word] = 1
    else:
        dictionary[word]+= 1
print(dictionary)

from collections import defaultdict # again, collections!
dictionary = defaultdict(int)
for word in word_list:
    dictionary[word] += 1
print(dictionary,dictionary['I'])


from collections import Counter

print(word_list)
counter = Counter(word_list)
dictionary=dict(counter)
print(dictionary) # 统计词频
print(counter.most_common(2))