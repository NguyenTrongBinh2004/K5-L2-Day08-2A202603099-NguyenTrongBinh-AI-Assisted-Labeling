# Báo cáo Day 08 — 2A202603099 - Nguyễn Trọng Bình

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m | Mặc định 0.3 cho mọi lớp (`threshold_sweep` không chọn được ngưỡng riêng) | `t_y26` (conf 0.25) ra 215 box, `t_y26_c05` (conf 0.5) chỉ còn 139 box (person 131→87, motorbike 55→32, bicycle 8→3), nên ngưỡng cao làm mất nhiều box. `threshold_sweep`: person, bicycle, motorbike, car không ngưỡng nào ≤ 0.5 đạt recall 90% (tối đa lần lượt 69%, 50%, 69%, 88%); bus (1 vật) và truck (0 vật) trong gold dưới 5 vật nên không chọn ngưỡng riêng. |
| three_wheeler | YOLOE-26s, prompt: `tuk-tuk`, `cyclo`, `cargo tricycle` (ánh xạ về `three_wheeler`) | Mặc định 0.3 (chỉ 4 vật trong gold, dưới 5) | YOLO26 (80 lớp COCO) không có lớp này. `t_yoloe` ra 8 box `three_wheeler` trên 8 ảnh pilot, 0.19s/ảnh. |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên):
  `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
  Kết quả gộp: 591 box (person 311, motorbike 154, car 68, three_wheeler 32, bicycle 14, truck 8, bus 4).
- **Phiên bản SAM bạn chọn:** SAM 2.1 base (`sam2.1_b.pt`). Lý do: nằm trong bộ `core`, chạy được trên GPU T4 của Colab, cho mask cho cả 16/16 box ở lần chạy `t_sam` (2.13s/ảnh, `img_004` và `img_013`). Chất lượng mask: tốt khi click đúng chỗ (img_013 lần 2 phủ gần cả xe, 214.871 px; img_018 bám sát vật nhỏ), nhưng phụ thuộc vào vị trí click; đặt click ở tâm box rơi vào khoảng hở làm img_013 lần 1 chỉ ra 12.587 px.
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? Ở bước duyệt có 3 ca đổi lớp (máy gán sai lớp, box đúng chỗ), và `three_wheeler` là lớp bị sửa nhiều nhất (xem mục 2). Đổi ngưỡng không tạo ra lớp `three_wheeler` cho YOLO26, nên phải dùng YOLOE gõ chữ và đổi lớp thủ công khi duyệt.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): điểm cao mà thừa: img_003, box `three_wheeler` #13 (0.80) nằm gọn trong box #3 (0.90) cho cùng một xe ba gác. Điểm thấp mà sai: img_031, box `person` #467 (0.34) nằm trên mũ treo ở quầy hàng, không phải người.
- Giới hạn: ngưỡng được chọn trên chính gold (chỉ 4 ảnh, 68 đối tượng) nên số đo máy trên gold là lạc quan; lớp có ít vật (bus 1, truck 0, three_wheeler 4) thì không đủ căn cứ chọn ngưỡng riêng, nên dùng mặc định.

## 2. AI error profile (từ `qc/review_stats.md`)

Mẫu: 24 ảnh batch, 505 box máy, 539 đối tượng sau duyệt, ngưỡng accepted IoU ≥ 0.9. Đã loại 24 đối tượng nằm quanh các bẫy khỏi bảng này (lỗi nhân tạo, không phản ánh chất lượng máy).

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 463 | 91.7% | 505 box máy |
| edited | 20 | 4.0% | 505 box máy |
| deleted | 19 | 3.8% | 505 box máy |
| added | 56 | 10.4% | 539 đối tượng sau duyệt |

Trong deleted + added có 3 trường hợp là đổi lớp (box đúng chỗ, máy gán sai lớp), không phải máy bỏ sót. Số vật máy sót thật ≈ 56 − 3 = 53.

Theo lớp (số box):

| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 257 | 5 | 4 | 25 |
| bicycle | 7 | 0 | 2 | 1 |
| motorbike | 127 | 8 | 1 | 16 |
| car | 50 | 2 | 3 | 5 |
| bus | 3 | 0 | 0 | 1 |
| truck | 5 | 0 | 2 | 1 |
| three_wheeler | 14 | 5 | 7 | 7 |

- Lớp nào máy tệ nhất? **three_wheeler**: trong 26 box máy chỉ 14 giữ nguyên, 5 phải sửa, 7 bị xoá, và phải vẽ thêm 7; đây là lớp tỉ lệ sửa cao nhất. Về số lượng tuyệt đối, máy sót nhiều nhất ở person (25 vật vẽ thêm) và motorbike (16). Ví dụ ảnh: img_003 (hai box `three_wheeler` chồng nhau cho một xe) và img_018 (vật nhỏ nghi không phải xe ba bánh).
- So với gold: precision / recall của **máy** là 60.9% (KTC 95%: 49–72%) và 57.4% (KTC 95%: 46–68%). Mẫu số: 68 đối tượng gold (TP 39, FP 25, FN 29). Đây là số lạc quan (ngưỡng chọn trên chính gold, chỉ 4 ảnh).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: **4/12 = 33.3%** (KTC 95%: 14–61%), thấp hơn ngưỡng đạt 80%. Theo loại: extra 2/3, loose 1/3, wrong_class 1/3, missing 0/3. Bẫy lọt: img_033 (loose), img_028 (extra), img_006 (wrong_class), img_008 (wrong_class), img_031 (missing), img_024 (missing), img_027 (missing), img_018 (loose). Với 12 bẫy, khoảng tin cậy rất rộng nên đây là dấu hiệu, không phải kết luận chắc. Vì sao lọt: 3 trong 8 bẫy lọt rơi vào ảnh dày xe máy (018, 024, 028) và bẫy `missing` lọt cả 3/3, nên nhiều khả năng tôi đã không zoom từng vùng và không quét lại để tìm vật chưa có box.
- Accepted cao (91.7%) mà bắt bẫy thấp (33.3%) nói gì về cách bạn duyệt? Nhãn máy trông đúng ở phần lớn box nên dễ tin cả những box sai, nhất là lỗi thiếu box (`missing` lọt 0/3). Thời gian duyệt khoảng 2.1 phút/ảnh thấp hơn mức 3 phút/ảnh mà checklist gợi ý, điều này có thể khớp với việc tôi đã bỏ qua bước quét ảnh trước khi nhìn nhãn máy.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`, cả hai là **ước lượng**, không bấm giờ): gán tay 30 phút / 4 ảnh = 7.5 phút/ảnh; duyệt 50 phút / 24 ảnh ≈ 2.1 phút/ảnh, tức duyệt nhanh hơn khoảng 3.6 lần theo ảnh. Tính theo đối tượng thì gán tay 30 phút / 72 đối tượng ≈ 25 giây/đối tượng, còn duyệt 50 phút / 554 đối tượng ≈ 5 giây/đối tượng (phần lớn box đã có sẵn từ máy). Thiên lệch: tôi gán gold trước nên đã quen cảnh và quy tắc khi duyệt, và 2.1 phút/ảnh thấp hơn mức khoảng 3 phút/ảnh mà checklist gợi ý, có thể liên quan đến việc bắt bẫy thấp (33.3%). Vì thời gian chỉ là ước lượng nên chênh lệch này chỉ mang tính tham khảo.
- Kết luận: chỉ dùng làm bản nháp, không dùng thẳng, và đặc biệt không tin lớp `three_wheeler`; vẫn phải duyệt có checklist, dựa trên: accepted 91.7%, nhưng recall của máy trên gold chỉ 57.4% (máy sót nhiều vật) và bắt bẫy chỉ 33.3%.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. Ẩn nhãn máy và quét ảnh, đếm vật theo lớp trước khi nhìn box máy. Kiểm tra bằng số bẫy `extra` và `missing` bắt được ở lô sau.
  2. Ở ảnh dày (005, 018, 021, 024, 028, 029): zoom duyệt theo từng vùng, vật dưới 20 px dùng `ignore_region`. Kiểm tra bằng số cảnh báo "cao < 20 px" của `validate_labels.py` giảm xuống.
  3. Sau khi duyệt box có sẵn, quét lại một lượt chỉ để tìm vật chưa có box (bẫy `missing` lọt 0/3). Kiểm tra bằng tỉ lệ `added` và bắt bẫy `missing`.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013 (xe ba bánh lớn, có người lái) | SAM 2.1 base (`sam2.1_b.pt`) | Lần 1: 3 (2 pos, 1 neg); lần 2: 7 (4 pos, 3 neg) | Không gồm người lái (áo xanh, quần jean); không có hàng hoá trên xe | Lần 1 sai: hai điểm pos nằm trong khoảng hở của xe (người ngồi quán phía sau và lan can) nên mask chỉ 12.587 px (1.8% ảnh, 2 polygon), không phủ thân xe. Lần 2 đặt pos trên phần đặc (mái, thành xe, bánh, thân đỏ), neg trên xe van và xe máy: mask 214.871 px (30.8% ảnh), 1 polygon, phủ gần cả xe, không tràn sang xe van, xe máy hay người phía sau. Còn lốm đốm: thủng ở ghế và giữa thanh lan can, vài đốm rải gần chân người lái | `tuktuk_img013.jpg` |
| img_004 (vật bị xe bên cạnh che) | SAM 2.1 base (`sam2.1_b.pt`) | 4 (2 pos, 2 neg) | Chỉ có phần nhìn thấy của xe; không có người lái hay hàng hoá rõ ràng trong vùng mask | Mask 11.904 px (1.7% ảnh), 1 polygon, bằng khoảng 70% diện tích box (11.904/16.940). Mask bám phần xe lộ ra, không tràn sang xe xanh lá và xe xanh dương nhờ hai điểm neg; chỉ phủ phần nhìn thấy, không đoán phần bị che | `tuktuk_img004_bi_che.jpg` |
| img_018 (vật nhỏ trong cảnh dày xe máy) | SAM 2.1 base (`sam2.1_b.pt`) | 4 (2 pos, 2 neg) | Không nhìn thấy người lái hay hàng hoá ở kích thước này | Mask 1.580 px (0.2% ảnh), 1 polygon, bằng khoảng 89% diện tích box (1.580/1.785); mask bám khá sát. Vật trông giống xe điện du lịch hoặc xe buýt nhỏ hơn xe ba bánh, nên có thể bị gán nhãn `three_wheeler` chưa đúng | `tuktuk_img018_nho.jpg` |

- Đối chiếu guideline mục 4.2: 4.2.4 (vùng chạm thuộc vật phía trước) cùng 4.1.4 cho biết mask xe không gồm người lái (img_013), và 4.1.2 cho biết vật bị che chỉ lấy phần nhìn thấy (img_004). Thiếu: 4.2 không nói khung, lan can, ghế nhìn xuyên thấy người phía sau thì xử lý sao (img_013); không có quy tắc mask cho vật bị che bao nhiêu thì bỏ (img_004); không có ngưỡng lệch biên cho vật nhỏ, trong khi sai số 3 px trên vật cao khoảng 35 px là gần 9% (img_018); không nói về xe điện du lịch. Các điểm này đã đưa vào calibration (CAL-01 đến CAL-05).
- *(Tuỳ chọn)* Interactor SAM trong CVAT: Không thực hiện (phần tuỳ chọn).

## 6. Calibration (Part 6)

- Số quyết định: 6 (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): CAL-01 và CAL-02 (mask tuk-tuk có gồm người lái không; khung lan can nhìn xuyên thấy người phía sau) và CAL-03 (cách đặt click SAM khi tâm box rơi vào chỗ hở).
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: mask xe không gồm người lái; ghế, lan can, mái, thành xe giữ trong polygon và chỉ bỏ khoảng hở lớn (≥ 5% diện tích vật); vật bị che chỉ lấy phần nhìn thấy; xe điện du lịch nhỏ không thuộc `three_wheeler`; vật rời cao 15–20 px vẫn vẽ nếu nhìn rõ loại; quy trình đặt click dương/âm cho SAM.
- Nguyên nhân gốc: chủ yếu guideline mơ hồ (v1.0 mục 4.2 không nói khung xuyên thấy, vật bị che, vật nhỏ), cộng với lỗi của máy (SAM phụ thuộc vị trí click, YOLO không có lớp `three_wheeler`).

## 7. QC và quyết định (Part 6)

- Mẫu: 5 ảnh (0 ảnh `can_review` + 5 ngẫu nhiên, seed 42), 115 đối tượng; người kiểm: tôi (**hạn chế**: tôi vừa là người duyệt nên có thể bỏ sót đúng những lỗi tôi bỏ sót lúc duyệt).
- Lỗi theo **đối tượng**: 15/115 = 13.0% · theo **ảnh**: 5/5 = 100.0% (ngưỡng Release ≤ 2%). Nguồn: `qc/qc_report_auto.md`.
- Top lỗi + ảnh ví dụ:
  - sai vị trí/kích thước (5 lỗi): img_031 (box `bicycle` chỉ ôm bánh trước, thiếu khung và giỏ), img_006, img_014;
  - bỏ sót (4 lỗi): img_031 (người ngồi ghế xanh không có box), img_014, img_035;
  - thừa (3 lỗi): img_003 (hai box `three_wheeler` cho một xe ba gác), img_031 (box `person` trên mũ treo ở quầy), img_035;
  - sai lớp (3 lỗi): xem các dòng `sai_lop` trong `qc/qc_review.csv`.
- **Giới hạn mẫu nhỏ:** chỉ 5 ảnh và cả 5 đều có lỗi, nên cận trên 95% (Wilson) của tỉ lệ ảnh lỗi thật là 100%. Mẫu này đủ để nói bản release chưa đạt ngưỡng 2%, nhưng không đủ để ước lượng chính xác tỉ lệ lỗi thật.
- Script kiểm được / không kiểm được: script chỉ đếm theo phiếu tôi điền bằng mắt, không tự phát hiện lỗi; nó không kiểm được lỗi mà tôi cùng bỏ sót với máy, và 19 ảnh còn lại của lô chưa được kiểm.

**Quyết định: Rework**, ngày 2026-11-18. Lý do: ảnh có lỗi 5/5 (100%) và lỗi theo đối tượng 13.0% đều vượt xa ngưỡng 2%; lỗi lặp lại ở nhiều ảnh (sai kích thước box, bỏ sót, box thừa/trùng, sai lớp); bắt bẫy 4/12 = 33.3% (< 80%) nên lô duyệt chưa đáng tin; guideline v1.0 đủ để áp dụng nên không phải Hold.
Việc cần làm tiếp: duyệt lại cả 24 ảnh theo checklist (quét ảnh trước khi nhìn nhãn máy), ưu tiên ảnh dày xe máy (005, 018, 021, 024, 028, 029) và ảnh có `three_wheeler`; rà box trùng và box thiếu phần (xe đạp chở hàng); chốt các câu hỏi mở của guideline v1.1; sau đó lấy mẫu QC lớn hơn (≥ 10 ảnh) để kiểm lại.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài:** dùng Claude (Anthropic) để hướng dẫn cài đặt môi trường (WSL2, Docker, CVAT, Google Colab), đọc thông báo lỗi, giải thích số liệu `review_stats`, soạn nháp `model_log.csv` / `REPORT.md`, và nhận diện lỗi trên 5 ảnh QC. Kiểm tra bằng cách: tự chạy lại các lệnh trên máy, đối chiếu số liệu với các file trong `qc/`, và tự nhìn ảnh trước khi dùng các nhận xét.
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? Không. Tôi gán tay 4 ảnh gold trong CVAT trước khi chạy model lên toàn bộ ảnh (lệnh `--split all` chạy với `--no-viz`, không xem trước nhãn máy); Part 2 chỉ thử model trên 8 ảnh `pilot`, không phải gold.
- Máy / GPU / CVAT dùng: Windows 10.0.26100, RAM 8 GB, không GPU cục bộ; WSL2 Ubuntu 26.04, Python 3.12 (tạo bằng `uv` vì Ubuntu mặc định là Python 3.14), Docker Desktop, CVAT 2.81.0. Model chạy trên Google Colab (GPU T4). Vướng mắc cài đặt: CVAT bản cũ 2.74.1 đang chạy chiếm cổng 8080 nên phải `docker compose down` rồi cài 2.81; Task tạo ngoài Project thiếu nhãn nên nạp COCO báo `Label 'motorbike' is not registered`, phải tạo Project từ `cvat/labels_generated.json` rồi tạo lại Task.
- Dùng cache `data/cache/` thay vì tự chạy? Không, đã tự chạy `prelabel.py` trên Colab T4.
- Lỗi / bug script đã gặp: YOLOE lần đầu báo cần restart runtime sau khi tự cài CLIP (kết quả vẫn ra). Các lỗi khác đều thuộc môi trường, không phải script: Task CVAT tạo ngoài Project thiếu nhãn nên không nạp được COCO; git trong WSL chưa có tên người dùng; gõ sai tên loại lỗi trong `qc_review.csv` nên script bỏ qua dòng đó và đếm thiếu lỗi cho đến khi sửa.
- Một điều bạn làm khác lần sau: ẩn nhãn máy và quét, đếm vật trong ảnh trước khi nhìn box máy; zoom các ảnh dày xe máy và ảnh có `three_wheeler`; dành gần 3 phút/ảnh khi duyệt thay vì khoảng 2 phút; bấm giờ thật để `time_log.csv` không phải là ước lượng.