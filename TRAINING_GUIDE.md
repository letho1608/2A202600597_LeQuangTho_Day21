# Hướng dẫn Fine-Tuning Qwen2.5-1.5B chuyên Python (Chống Ảo Giác)

## 1. Bối Cảnh & Mục Đích

Mô hình gốc `Qwen2.5-1.5B-Instruct` là một mô hình cực kỳ tối ưu cho các thiết bị tài nguyên thấp (như Google Colab T4 15GB VRAM). Tuy nhiên, với kích thước 1.5 tỷ tham số, mô hình thường gặp phải hiện tượng **Ảo Giác (Hallucination)** khi xử lý các tác vụ code Python:
- Bịa ra các thư viện hoặc API không tồn tại.
- Giải thích dài dòng, rườm rà (Rambling) thay vì tập trung vào code.
- Dễ bị đánh lừa bởi các prompt không rõ ràng.

**Mục đích của việc chỉnh sửa code & chiến lược training này là:**
1. **Ép xung tài nguyên:** Tối ưu hóa tuyệt đối để chạy mượt mà trên Google Colab Free (T4).
2. **Kìm hãm ảo giác:** Thông qua việc tinh chỉnh dữ liệu (Data Curation) và thay đổi tham số huấn luyện.
3. **Thay đổi hành vi:** Dạy cho mô hình cách từ chối các yêu cầu sai và chỉ tập trung xuất ra format code Python chuẩn xác.

---

## 2. Chiến Lược Cải Thiện Chất Lượng (Chống Ảo Giác)

Để train một mô hình thực sự dùng được (Production-ready) thay vì chỉ demo, ta áp dụng 3 trụ cột sau:

### Trụ cột 1: Data Curation (Quyết định 80% thành bại)
Không nạp toàn bộ 18.000 mẫu một cách mù quáng. Dữ liệu cần được xử lý:
- **Tiêm "Vaccine Từ Chối":** Trộn vào 200-500 mẫu dữ liệu hướng dẫn mô hình nói *"Không"*. 
  *(VD: Nếu user yêu cầu dùng thư viện `fake_pandas`, mô hình phải trả lời "Thư viện này không tồn tại, tôi sẽ dùng `pandas` chuẩn...").*
- **Thêm Data Debugging:** Dạy mô hình bằng các đoạn code lỗi và yêu cầu sửa lỗi. Điều này ép mô hình phải *suy luận logic (reasoning)* thay vì chỉ học vẹt cách viết code mới.
- **Loại bỏ nhiễu:** Xóa bỏ các prompt quá ngắn, các đoạn giải thích tiếng Anh dài dòng không có code.

### Trụ cột 2: Cấu Hình LoRA & Optimizer
- **Rank:** Giữ ở mức `r=16` hoặc `r=32`. (Rank quá lớn như `r=64` trên tập dữ liệu nhỏ sẽ gây học vẹt - overfitting, làm trầm trọng thêm ảo giác).
- **Target Modules:** Nhắm vào **ALL Layers** (`q, k, v, o, gate, up, down`). Bắt buộc phải target vào các lớp MLP (`gate, up, down`) vì đây là nơi chứa kiến thức suy luận của mô hình.
- **DoRA:** TẮT DoRA khi target ALL Layers trên Colab T4 để tránh Out-of-Memory (OOM).

### Trụ cột 3: Hyperparameters
- **Weight Decay:** Tăng từ `0.01` lên `0.05` để giảm overfitting.
- **Learning Rate:** Cố định ở `1e-4` hoặc `5e-5` (tránh Catastrophic Forgetting - làm mất kiến thức cơ bản của mô hình gốc).
- **Sequence Length:** Cắt cứng `max_seq_length = 512`. Nếu data dài hơn 512 tokens, cắt bỏ hoặc chia nhỏ.

---

## 3. Cách Thức Chạy Code Trên Colab T4

Code trong file `notebooks/Lab21_LoRA_Finetuning_T4.ipynb` đã được cấu hình sẵn theo chiến lược trên (dùng batch 1, grad_accum 8, max_seq_length 512, target ALL Layers).

**Các bước thực hiện:**
1. Upload file `Lab21_LoRA_Finetuning_T4.ipynb` lên Google Colab.
2. Đổi Runtime: `Runtime -> Change runtime type -> T4 GPU`.
3. Thay đổi Dataset: 
   - Kéo xuống Cell 5 (Option B). Code hiện tại đang load 100 mẫu ngẫu nhiên từ tập `iamtarun/python_code_instructions_18k_alpaca`.
   - Nếu bạn có tập dữ liệu "đã tiêm vaccine từ chối" của riêng mình, hãy sửa đường dẫn dataset tại đây.
4. Chạy toàn bộ (Run All).
5. Khi có yêu cầu kết nối Google Drive, hãy cấp quyền. Toàn bộ Checkpoints sẽ được lưu an toàn trên Drive.
6. Ở Cell cuối cùng, nhập Token HuggingFace của bạn (Nhớ cấp quyền WRITE) để hệ thống tự động lưu model lên đám mây.

## 4. Ghi Chú Khi Kiểm Thử (Inference)
Sau khi train xong, để mô hình sinh code ít ảo giác nhất:
- Cài đặt `temperature = 0.1` (Hoặc 0.2). Không để cao vì code cần sự chính xác, không cần sự sáng tạo ngẫu nhiên.
- Dùng `top_p = 0.9` và `repetition_penalty = 1.1`.