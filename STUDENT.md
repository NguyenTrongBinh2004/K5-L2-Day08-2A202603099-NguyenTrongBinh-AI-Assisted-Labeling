# Thông tin học viên

- Họ và tên: Nguyễn Trọng Bình
- MSHV: 2A202603099
- Lớp (L2 / L3 / khác): L2
- Tài khoản GitHub: https://github.com/NguyenTrongBinh2004
- Repo bài nộp (URL): https://github.com/NguyenTrongBinh2004/K5-L2-Day08-2A202603099-NguyenTrongBinh-AI-Assisted-Labeling
- Commit chốt bài (hash): <<hash commit cuối, lấy bằng `git rev-parse HEAD` sau khi push>>
- Máy: Windows 11 (bản 10.0.26100), RAM 8 GB, chạy WSL2 Ubuntu 26.04; không dùng GPU cục bộ, model chạy trên Google Colab (GPU T4, CUDA)
- CVAT (Docker): v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 463 / 20 / 19 (mẫu số: 505 box máy); added: 56 (mẫu số: 539 đối tượng)
- So với gold: precision / recall của **máy** 60.9% / 57.4% (mẫu số: 68 đối tượng gold)
- Bắt bẫy: 4/12 = 33.3%
- Thời gian: gán tay 7.5 phút/ảnh; duyệt có pre-label khoảng 2.1 phút/ảnh (cả hai là ước lượng, không bấm giờ)
- Lỗi QC: <<...% theo ảnh (.../...)>>, <<...% theo đối tượng (.../...)>>
- **Quyết định:** <<Release / Hold / Rework>> — lý do một dòng: <<...>>