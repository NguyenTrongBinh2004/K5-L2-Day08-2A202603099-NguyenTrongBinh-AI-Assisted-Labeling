# Label contract — Phố châu Á (v1.1)

> Bản guideline cho lô ảnh Day 08, **đã calibration** từ v1.0 (bản giảng viên viết sẵn) bằng các lỗi quan sát ở Part 4 (review) và Part 5 (mask SAM). Các quyết định calibration được tóm tắt ở mục 8 và ghi đầy đủ trong `qc/calibration_decisions.csv` (CAL-01 → CAL-06).

- Phiên bản: **v1.1**
- Loại nhãn: **detection** (bounding box) cho 4 + 24 ảnh (gold + batch); **mask** (SAM) cho 3 vật khó ở Part 5 (bài nâng cao: polygon cho 8 ảnh `seg`)
- Công cụ: CVAT. Xuất COCO 1.0
- Căn cứ calibration: lỗi quan sát ở Part 4 và Part 5; xem `qc/calibration_decisions.csv` (CAL-01 → CAL-06)

---

## 1. Mục đích và phạm vi

- **Mục đích:** huấn luyện detector / segmenter đếm và phân loại phương tiện, người trên đường phố Đông Nam Á (camera giao thông, dashcam). Model cần nhận ra **xe ba bánh địa phương** (tuk-tuk, xích lô, ba gác) vì model COCO có sẵn gọi chúng là car / truck / motorbike.
- **Phạm vi:** ảnh ngoài trời ban ngày và ban đêm, đường phố, chợ; không dùng ảnh trong nhà.
- **Nguyên tắc chung:** gán **mọi** vật thuộc lớp mà **nhìn thấy được** và **đủ lớn** (mục 4). Máy gợi ý, người quyết định.

## 2. Lớp (mỗi vật thuộc đúng một lớp)

| Lớp | Định nghĩa | Gồm | Không gồm (dễ nhầm) |
|---|---|---|---|
| `person` | Một người, từ đầu đến chân (phần nhìn thấy). | người đi bộ, người ngồi bán hàng, người lái / người ngồi sau xe | tượng, ma-nơ-canh, người trong tranh / biển quảng cáo / phản chiếu |
| `bicycle` | Xe đạp thường, đạp bằng chân. | xe đạp chở hàng, xe đạp có giỏ | xe đạp điện (xem mục 6), xe đạp gấp gọn dựng trong cửa hàng nhưng không ai dùng vẫn tính |
| `motorbike` | Xe máy, xe tay ga, mô tô 2 bánh có động cơ. | xe đang đỗ, xe chở hàng | **người ngồi trên xe** (người là box `person` riêng) |
| `car` | Ô tô con, taxi, SUV, bán tải, minivan. | taxi màu, xe công ty | xe ba bánh (→ `three_wheeler`) |
| `bus` | Xe buýt, xe khách lớn. | minibus chở khách | xe buýt trên biển quảng cáo |
| `truck` | Xe tải có buồng lái và thùng hàng, ≥ 4 bánh. | xe tải nhỏ | **xe ba gác / xe ba bánh chở hàng** (→ `three_wheeler`) |
| `three_wheeler` | Xe **ba bánh**: tuk-tuk (Thái), xích lô (cyclo), xe ba gác / xe ba bánh chở hàng. | cả xe lẫn khung mái / thùng hàng gắn liền xe | xe máy kéo rơ-moóc nhỏ có 4 bánh (hiếm, ghi `can_review`) |
| `ignore_region` | **Vùng** không gán từng vật: đám đông xe / người dày đặc, mờ, không tách được từng cá thể. | khối xe máy đỗ sát nhau ở xa | không dùng để "lờ đi" vật rõ ràng chỉ vì khó vẽ |

**Vật không thuộc lớp nào** (cây, biển báo, thùng hàng, tượng…): **không gán, không vẽ ignore**.

## 3. Thuộc tính

| Thuộc tính | Giá trị | Cách dùng |
|---|---|---|
| `conf` | chuỗi (ví dụ `0.87`), tự điền khi nạp pre-label | Điểm tin cậy của máy. Bạn **không** sửa. Nhãn bạn vẽ tay để trống. Chỉ dùng để xếp thứ tự duyệt, **không** thay việc nhìn ảnh. |
| `can_review` | tick / không | Tick khi bạn **không chắc quy tắc** (không phải khi vật khó vẽ). Mỗi ca tick phải được ghi vào bảng ca khó (mục 5) hoặc vào calibration. |

## 4. Quy tắc hình học

### 4.1 Box (detection)

1. Box là hình chữ nhật **ôm sát phần nhìn thấy** của vật: cạnh box chạm điểm ngoài cùng của vật theo từng phía, sai số cho phép ≤ ~2 px ở ảnh 1024 px.
2. Vật **bị che một phần**: vẽ box quanh **phần nhìn thấy**, không đoán phần bị che. Vật bị che > 80% (còn nhìn thấy < 20% diện tích) thì **không gán**.
3. Vật **bị cắt mép ảnh**: vẽ box đến mép ảnh (không vẽ ra ngoài ảnh).
4. **Người ngồi trên xe**: hai box riêng, một `person` (cả người), một `motorbike` / `bicycle` / ... (cả xe). Hai box chồng nhau là bình thường.
5. **Hàng hoá chở trên xe**: không tính vào box xe nếu nhô ra khỏi xe hơn ~20% chiều dài xe (xem mục 6, câu hỏi 3).
6. **Vật quá nhỏ** *(đã hiệu chỉnh ở v1.1 — CAL-06)*: box cao **< 15 px** → không vẽ từng vật. Vật **rời rạc** cao **15–20 px** vẫn vẽ nếu nhìn rõ loại. Nếu là **cả cụm** nhỏ nằm sát nhau (≥ 5 vật) → vẽ **một** `ignore_region` bao cụm.
7. **Hình in, tượng, phản chiếu, ảnh trên biển quảng cáo**: không gán.

### 4.2 Mask / polygon (Part 5: 3 vật; nâng cao: 8 ảnh `seg`)

1. Polygon bám **đường biên của phần nhìn thấy**; đỉnh đủ dày để lệch biên ≤ ~3 px, không cần chi tiết nhỏ hơn bánh xe / tay lái.
2. **Lỗ thủng** (khoảng trống nhìn thấy xuyên qua khung xích lô, giữa tay và thân người): **không** vẽ lỗ nếu lỗ nhỏ hơn ~5% diện tích vật; lỗ lớn thì bỏ ra khỏi polygon.
3. **Bóng đổ** không thuộc vật. **Kính xe** thuộc vật (xem mục 6, câu hỏi 5).
4. Hai vật chạm nhau: mỗi vật một polygon, **không chồng lên nhau**; vùng chạm thuộc vật ở phía trước.

### 4.3 Chọn lớp khi nhìn không rõ

- Không chắc lớp (ví dụ xe ba bánh ở xa, mờ): chọn lớp **bạn tin nhất** và tick `can_review`. Không chọn lớp ngẫu nhiên để "cho có".
- Hai lớp có thể đúng (xe ba gác có mui / xe tải nhỏ): theo đếm **số bánh nhìn thấy**; không đếm được → `can_review`.

### 4.4 Quy trình dùng SAM để tạo mask (mới v1.1 — CAL-03)

1. Đặt click **dương** trên **phần đặc** của vật (mái, thành xe, bánh, thân), **không** đặt ở tâm box nếu tâm rơi vào chỗ hở.
2. Đặt click **âm** trên vật kề bên (xe khác, người, nền) để mask không tràn.
3. Kiểm tra bằng số: mask **nhỏ hơn ~10% diện tích box** thì nghi sai và phải đặt lại click.
   - Ví dụ `img_013`: lần 1 mask chỉ **12.587 px** (2 polygon, tâm box rơi vào khoảng hở); lần 2 đặt lại click trên phần đặc → **214.871 px** (1 polygon).

## 5. Ca khó đã chốt (bảng sống: cập nhật ở v1.1)

| Ca | Quyết định | Lý do | Ảnh ví dụ |
|---|---|---|---|
| Xích lô: khách ngồi trước, người đạp ngồi sau | 1 `three_wheeler` (cả xe + mui), mỗi người một `person` | Người là vật riêng; xe là vật riêng | — |
| Người bán hàng dắt xe đạp chở đầy hàng | `person` + `bicycle` (box xe ôm cả giỏ hàng gắn cố định) | Giỏ gắn cố định thuộc xe | — |
| Xe máy đỗ sát nhau thành hàng, vật ở xa ít thấy | Vật rõ → gán từng xe; phần ở xa, cao < 15 px → `ignore_region` | Mục 4.1.6 (đã hiệu chỉnh v1.1) | — |
| Tuk-tuk có hành khách ngồi nhìn thấy qua cạnh hở | Gán `person` nếu còn nhìn thấy ≥ 20% người | Quy tắc che chung (4.1.2) | — |
| **Mask tuk-tuk, người lái ngồi phía trước thân xe** | Polygon xe **không gồm** người lái; người lái là `person` riêng | 4.1.4 (người và xe là hai vật) và 4.2.4 (vùng chạm thuộc vật phía trước) | img_013 |
| **Khung lan can, ghế, thanh mái nhìn xuyên thấy người phía sau** | Ghế, thanh lan can, mái, thành xe là phần xe, **giữ trong polygon**; chỉ bỏ khoảng hở lớn (≥ 5% diện tích vật) nhìn thấy hẳn người hoặc vật phía sau | 4.2.2: lỗ nhỏ không vẽ, lỗ lớn thì bỏ | img_013 |
| **Vật bị xe bên cạnh che một phần** | Mask chỉ **phần nhìn thấy**; áp ngưỡng 4.1.2 cho mask (còn thấy < 20% thì không gán); vùng chạm thuộc xe phía trước | 4.1.2 và 4.2.4 | img_004 |
| **Vật nhỏ (~50×35 px), không đếm được số bánh** | Chọn lớp tin nhất và tick `can_review` (4.3); xe điện du lịch nhỏ 4 bánh **không** thuộc `three_wheeler` | 4.3 và định nghĩa `three_wheeler` (xe ba bánh) | img_018 |
| **Vật rời rạc cao 15–20 px (người, xe máy ở xa), không thuộc cụm ≥ 5 vật** | Vẫn vẽ nếu nhìn rõ loại; dưới 15 px hoặc cả cụm thì dùng `ignore_region` | Mở rộng 4.1.6 theo thực tế gán nhãn ở ảnh dày | img_018 |

## 6. Câu hỏi CÒN MỞ (trạng thái cập nhật ở v1.1)

1. **Xe đạp điện / xe máy điện** nhỏ: `bicycle` hay `motorbike`? — *chưa chốt*
2. **Khối xe máy đỗ dày đặc gần camera** (không phải ở xa): từng xe hay `ignore_region`? Ngưỡng cụ thể? — *chưa chốt*
3. **Hàng nhô ra**: xe đạp / ba gác chở tre, rổ nhô xa: box xe có gồm hàng không? Mask thì sao? — **chưa chốt** (3 ảnh mask v1.1 không có ca đủ rõ)
4. **Người ngồi trong xe ba bánh có mui che** (chỉ thấy thấp thoáng): ngưỡng "còn thấy ≥ 20%" ở mục 4.1.2 có hợp lý không? — *chưa chốt*
5. **Kính chắn gió / cửa sổ xe**: thuộc mask xe (nhìn xuyên qua thấy nội thất / người) hay không? — **chưa chốt** (3 ảnh mask v1.1 không có ca đủ rõ)
6. **Box máy chồng nhau quá nhiều** trong ảnh đêm: gộp hay giữ? — *chưa chốt*
7. Mức **tin cậy tối thiểu** để chấp nhận box máy không cần sửa (chỉ là `conf`, chưa là quy tắc chất lượng): có đặt không? — *chưa chốt*

**Đã chốt trong v1.1:** mask vật bị che một phần → chỉ lấy phần nhìn thấy, áp ngưỡng 4.1.2 (xem CAL-04 và mục 5).

Mỗi câu chốt xong phải có **ảnh ví dụ** (`img_0xx`) và ghi vào bảng ca khó + lịch sử thay đổi.

## 7. QC và phiên bản

- Ngưỡng QC: xem `submission/taxonomy.json` mục `rules` (**không sửa**): IoU chấp nhận 0.9 (mask 0.85), ảnh lỗi ≤ 2%, bắt bẫy ≥ 80% (≥ 10/12).
- Người quyết ca mơ hồ: bạn (lab cá nhân); ghi từng quyết định vào `calibration_decisions.csv`.
- Mọi thay đổi làm đổi nhãn → tăng phiên bản.

| Phiên bản | Thay đổi | Lý do (ca nào) |
|---|---|---|
| v1.0 | Bản giảng viên viết sẵn | — |
| v1.1 | Thêm 5 ca khó vào mục 5; thêm quy trình SAM (4.4); hiệu chỉnh 4.1.6 (ngưỡng 15 px thay vì 20 px, thêm ca 15–20 px rời rạc vẫn vẽ); ghi 6 quyết định calibration vào mục 8; chưa chốt câu 3 và 5 ở mục 6 | CAL-01 → CAL-06 |

## 8. Calibration decisions log (mới v1.1)

Bản đầy đủ nằm trong `qc/calibration_decisions.csv`. Tóm tắt:

| ID | Ảnh | Nguồn | Mô tả | Quyết định | Mục v1.0 |
|---|---|---|---|---|---|
| CAL-01 | img_013 | cau_hoi_mo | Mask tuk-tuk có gồm người lái không? v1.0 mục 4.2 không nói rõ | Không gồm: người lái là `person` riêng, polygon xe chỉ gồm xe | 4.1.4; 4.2.4 |
| CAL-02 | img_013 | cau_hoi_mo | Khung lan can, ghế, thanh mái nhìn xuyên thấy người phía sau: có bỏ khỏi polygon không? | Ghế, lan can, mái, thành xe thuộc xe và giữ trong polygon; chỉ bỏ khoảng hở lớn (từ 5% diện tích vật) nhìn thấy hẳn người hoặc vật phía sau | 4.2.2 |
| CAL-03 | img_013 | loi_may | SAM: click dương ở tâm box rơi vào khoảng hở của xe nên mask chỉ 12.587 px, 2 polygon; đặt lại click trên phần đặc được 214.871 px, 1 polygon | Click dương đặt trên phần đặc, click âm trên vật kề bên; mask nhỏ hơn 10% diện tích box thì kiểm tra lại | 4.2.1 (nay là 4.4) |
| CAL-04 | img_004 | cau_hoi_mo | Vật bị xe bên cạnh che một phần: mask lấy phần nhìn thấy hay đoán phần che? | Mask chỉ phần nhìn thấy; áp ngưỡng 4.1.2 (còn thấy dưới 20% thì không gán); vùng chạm thuộc xe phía trước | 4.1.2; 4.2.4 |
| CAL-05 | img_018 | cau_hoi_mo | Vật nhỏ khoảng 51×35 px màu hồng, trông giống xe điện du lịch, không đếm được bánh: có phải `three_wheeler`? | Xe điện du lịch nhỏ 4 bánh không thuộc `three_wheeler`; không đếm được bánh thì chọn lớp tin nhất và tick `can_review` | 2; 4.3 |
| CAL-06 | img_018 | cau_hoi_mo | Vật rời rạc cao 15–20 px (người, xe máy ở xa): vẽ hay `ignore_region`? `validate_labels` cảnh báo nhiều box dưới 20 px ở img_001, img_018, img_036 | Vật rời rạc cao 15–20 px vẫn vẽ nếu nhìn rõ loại; dưới 15 px hoặc cả cụm thì dùng `ignore_region` | 4.1.6; 6 (câu 2) |