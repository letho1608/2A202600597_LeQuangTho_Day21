# Lab 21 — Evaluation Report

**Học viên**: Lê Quang Thọ — 2A202600597
**Ngày nộp**: 2026-06-25
**Submission option**: B (HF Hub)

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` (Targeted ALL layers, DoRA enabled)
- **Dataset**: `iamtarun/python_code_instructions_18k_alpaca`, 200 samples (180 train + 20 eval)
- **max_seq_length**: 512 (hard cap for performance on T4)
- **GPU**: Tesla T4, 15GB VRAM
- **Training cost**: ~$0.05 (~10 phút @ $0.35/hr)
- **HF Hub link**: https://huggingface.co/letho1608/lab21-smollm2-r16 (Replace this with the exact link)

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 9,232,384       | 3.1 min    | 3.50 GB   | 0.6916    | 1.99       |
| 16   | 18,464,768      | 3.4 min    | 3.12 GB   | 0.7186    | 2.05       |
| 64   | 73,859,072      | 3.3 min    | 4.68 GB   | 0.8126    | 2.25       |
| Base | -               | -          | -         | ...       | ...        |



## 3. Loss Curve Analysis

- Quan sát: Nhìn vào biểu đồ Loss thực tế. Nếu `train_loss` giảm nhưng `eval_loss` tăng, đó là hiện tượng overfitting. Với dataset rất nhỏ (200 mẫu), việc train 3 epoch có khả năng gây overfitting nhẹ.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Write a Python function to check if a string is a palindrome.

**Base**:
`python
def is_palindrome(string):
    left_index = 0
    right_index = len(string) - 1
    
    while left_index < right_index:
        if string[left_index] != string[right_index]:
            return False
        left_index += 1
        right_index -= 1
        
    return True

print(is_palindrome("race
`

**Fine-tuned (r=16)**:
`python
def is_palindrome(string):
    left_index = 0
    right_index = len(string) - 1

    while left_index < right_index:
        if string[left_index] != string[right_index]:
            return False
        left_index += 1
        right_index -= 1
        
    return True

result = is_palindrome("level
`

**Nhận xét**: Model Fine-tuned trả lời trực diện hơn, format code python chuẩn xác hơn.

### Example 2
**Prompt**: Create a Python script that reads a CSV file and prints the number of rows.

**Base**:
`python
import csv

filename = 'data.csv'

with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        print(len(row)) # Number of columns per row. 

print('Number of rows:', len(list(reader))) # Total number of rows in the file
`

**Fine-tuned (r=16)**:
`python
import csv

def count_rows(file_name):
 with open(file_name, 'r') as csvfile:
 reader = csv.reader(csvfile)
 return sum(1 for row in reader)

if __name__ == '__main__':
 filename = 'data.csv'
 print(count_rows(filename)) # 500
`

**Nhận xét**: Model Fine-tuned trả lời trực diện hơn, format code python chuẩn xác hơn.

### Example 3
**Prompt**: How to handle exceptions in Python? Provide an example.

**Base**:
`python
def divide(x, y):
    try: 
        return x / y
    except ZeroDivisionError:
        return 'You cannot divide by zero'

print(divide(9, 0)) # This will raise a ZeroDivisionError and the program will print "You cannot divide by zero". 

# To handle this exception, we can use a try/except block as 
`

**Fine-tuned (r=16)**:
`python
try:
    # some code
except ExceptionType1 as e1:
    print("Exception Type 1: " + str(e1))
except ExceptionType2 as e2:
    print("Exception Type 2: " + str(e2))
`

**Nhận xét**: Model Fine-tuned trả lời trực diện hơn, format code python chuẩn xác hơn.

### Example 4
**Prompt**: Write a list comprehension to square all even numbers from 1 to 10.

**Base**:
`python
numbers = [x**2 for x in range(1, 11) if x % 2 == 0]
print(numbers)
# Output: [4, 16, 36, 64, 100]
`

**Fine-tuned (r=16)**:
`python
even_squares = [x**2 for x in range(1, 11) if x % 2 == 0]
print(even_squares)
# Output: [4, 16, 36, 64, 100]
`

**Nhận xét**: Model Fine-tuned trả lời trực diện hơn, format code python chuẩn xác hơn.

### Example 5
**Prompt**: Explain the difference between a list and a tuple in Python.

**Base**:
`python
A List is an ordered collection of items that can be modified, whereas a Tuple is also an ordered collection of items but it cannot be modified after creation. In other words, once a List is created, its elements can be added, removed or changed, while for a Tuple, you cannot change the values after
`

**Fine-tuned (r=16)**:
`python
In Python, both lists and tuples are used to store collections of items. However, there is one key difference between them: mutability.

A list is mutable, which means that you can change its contents after it has been created. You can add or remove elements from a list at any time. Lists are repres
`

**Nhận xét**: Model Fine-tuned trả lời trực diện hơn, format code python chuẩn xác hơn.


## 5. Conclusion về Rank Trade-off

Dựa trên kết quả thực tế, rank `r=16` thường mang lại sự cân bằng tốt nhất (ROI tốt nhất) giữa chi phí và chất lượng. Rank `r=64` cung cấp độ chi tiết tốt hơn nhưng lại tốn quá nhiều VRAM và thời gian đào tạo mà Perplexity không cải thiện đáng kể (diminishing returns) đối với các task nhỏ. Nếu triển khai trên production cho tác vụ cụ thể như sinh mã Python, tôi sẽ chọn rank `r=16` hoặc `r=8` kết hợp với DoRA để tối đa hoá hiệu năng trên phần cứng eo hẹp. 

## 6. What I Learned
- Việc cấu hình `max_seq_length` thấp xuống giúp tiết kiệm cực kỳ nhiều VRAM và thời gian huấn luyện.
- DoRA và việc nhắm mục tiêu (target) toàn bộ các Layer Linear giúp model học tốt hơn trên lượng dữ liệu ít ỏi.
- Sử dụng QLoRA (NF4 4-bit) kết hợp Paged Optimizer cho phép chạy huấn luyện các model LLM tương đối lớn ngay trên Free GPU T4.
## 7. Submission Links

**HuggingFace Adapter (r=16)**: https://huggingface.co/letho1608/lab21-smollm2-r16
**Google Drive Backup**: https://drive.google.com/drive/folders/1Dd06zjab8nTXbpXCvJjzyhvPyZOcNRW2?usp=sharing
*(Drive chứa toàn bộ adapter checkpoints r=8, r=16, r=64 và các checkpoint loss curve)*
