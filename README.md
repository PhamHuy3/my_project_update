1. Chạy create_owner_emb_from_tflite trên máy tính tạo file owner_embedding.npy
2. Chuyển đổi để lưu file owner_embedding.npy vào STM32
3. face_detection_front.tflite là Model dùng để phát hiện khuôn mặt
4. facenet_mcu_int8.tflite là Model tạo owner_embedding, cần code để so sánh giữa owner_embedding được tạo từ Model với owner_embedding đã lưu trước đó để xác định chu_nha/nguoi_la
5. recognize_owner là code test trên PC
