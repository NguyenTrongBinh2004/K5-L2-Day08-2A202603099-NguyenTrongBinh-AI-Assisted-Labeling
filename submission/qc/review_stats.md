# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 505; đối tượng sau duyệt: 539; ngưỡng accepted: IoU >= 0.9
- Đã **loại 24 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 463 | 463/505 = 91.7% | 505 box máy |
| edited (sửa) | 20 | 20/505 = 4.0% | 505 box máy |
| deleted (xoá) | 19 | 19/505 = 3.8% | 505 box máy |
| added (vẽ thêm) | 56 | 56/539 = 10.4% | 539 đối tượng sau duyệt |

Trong số deleted + added, **3 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 53.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 257 | 5 | 4 | 25 |
| bicycle | 7 | 0 | 2 | 1 |
| motorbike | 127 | 8 | 1 | 16 |
| car | 50 | 2 | 3 | 5 |
| bus | 3 | 0 | 0 | 1 |
| truck | 5 | 0 | 2 | 1 |
| three_wheeler | 14 | 5 | 7 | 7 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 68 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 39 | 25 | 29 | 60.9% (KTC 95%: 49–72%) | 57.4% (KTC 95%: 46–68%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 4/12 = 33.3% (KTC 95%: 14–61%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - loose: 1/3
  - extra: 2/3
  - wrong_class: 1/3
  - missing: 0/3
  - lọt: img_033.jpg (loose), img_028.jpg (extra), img_006.jpg (wrong_class), img_031.jpg (missing), img_008.jpg (wrong_class), img_024.jpg (missing), img_027.jpg (missing), img_018.jpg (loose)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
