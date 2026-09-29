# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

file = open("sample.txt", "r")
text = file.read().lower()
file.close()

words = []
raw_words = text.split()
for w in raw_words:
    clean_word = w.strip(".,!?")
    words.append(clean_word)

word_counts = {}
for word in words:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

print("Words: Words Count")
for word in word_counts:
    print(f"{word} : {word_counts[word]}")
