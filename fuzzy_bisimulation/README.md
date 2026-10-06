# Pipeline học có giám sát cho Fuzzy Bisimulation

Cài đặt toàn bộ quy trình học có giám sát để xấp xỉ độ tương đồng fuzzy bisimulation trên đồ thị tri thức mờ, theo Algorithm 1 của bài báo *"Learning to scale similarity reasoning in large knowledge graphs"*.

## Cấu trúc thư mục

```
├── fuzzy_bisimulation/
│   ├── fuzzy_kg.py             # Đọc bộ ba, fuzzy hoá, tạo đặc trưng nút
│   ├── exact_bisimulation.py   # Tính fuzzy bisimulation chính xác bằng điểm bất động
│   ├── local_sampling.py       # Lấy mẫu đồ thị con cục bộ h-hop
│   ├── generate_labels.py      # Sinh tập dữ liệu và gán nhãn chính xác
│   ├── model.py                # Bộ mã hoá GCNConv và đầu dự đoán theo cặp
│   ├── train.py                # Huấn luyện mô hình có giám sát
│   ├── evaluate.py             # Đánh giá, so sánh và phân tích sai số
│   └── README.md               # Tài liệu của gói
│
├── data/                       # Đồ thị gốc và tập cặp đã gán nhãn
├── checkpoints/                # Trọng số mô hình và checkpoint huấn luyện
└── results/                    # Chỉ số, log và biểu đồ
```

## Tổng quan quy trình

1. **Fuzzy hoá (`fuzzy_kg.py`)**:
   Đọc các bộ ba `head<TAB>relation<TAB>tail` và gán cho mỗi bộ ba trọng số mờ
   `w(s, r, s') = Sigmoid(0.6 · Freq(r) + 0.4 · Deg(s, s'))`.

2. **Lấy mẫu cục bộ (`local_sampling.py`)**:
   Trích đồ thị con cảm sinh bởi lân cận BFS 2-hop của một thực thể hạt giống, tối đa `max_nodes` thực thể.

3. **Nhãn chính xác (`exact_bisimulation.py`)**:
   Tính điểm bất động lớn nhất `B* = gfp(F)` bằng cách lặp `B_{k+1} = F(B_k)` từ quan hệ toàn 1, trong đó

   ```
   F(B)(s, t) = min_r min_s' [ w(s, r, s') → max_t' ( w(t, r, t') ⊗ B(s', t') ) ]
   ```

   kết hợp với điều kiện theo chiều ngược lại. Hỗ trợ ngữ nghĩa Gödel (mặc định), Łukasiewicz và product.

4. **Sinh tập dữ liệu (`generate_labels.py`)**:
   Chạy thuật toán chính xác trên từng đồ thị con, gán nhãn `y_ij = B*_{G_k}(s_i, s_j)` cho các cặp thực thể được lấy mẫu, rồi ghi `graph.pt`, `train.pt`, `val.pt`, `test.pt` và `meta.json` vào `data/<dataset>/processed/`. Tỉ lệ chia 80/10/10 được thực hiện theo đồ thị con, nên các cặp của cùng một đồ thị con không bao giờ nằm ở hai tập khác nhau.

5. **Huấn luyện (`train.py`)**:
   Huấn luyện `f(s_i, s_j) = Sigmoid(MLP([h_i ‖ h_j ‖ |h_i − h_j|]))` với `h = GCN(G)`, dùng Huber loss, Adam và dừng sớm theo MAE trên tập validation. Trọng số được lưu vào `checkpoints/<dataset>/`.

6. **Đánh giá (`evaluate.py`)**:
   Tính MAE, MSE, RMSE, hệ số tương quan Pearson / Spearman / Kendall-Tau và thời gian suy luận so với nhãn chính xác, đặt cạnh baseline Common Neighbors, rồi ghi `metrics.json`, `predictions.csv` và `predictions.png` vào `results/<dataset>/`.

## Cách chạy

Chạy các module từ thư mục gốc của repo:

```bash
python -m fuzzy_bisimulation.generate_labels --dataset WN18RR --num_pairs 80000
python -m fuzzy_bisimulation.train --dataset WN18RR
python -m fuzzy_bisimulation.evaluate --dataset WN18RR
```

Bài báo dùng 50.000 / 80.000 / 150.000 cặp có nhãn cho FB15k-237 / WN18RR / YAGO3-10. Mỗi script liệt kê các tuỳ chọn bằng `--help`; giá trị mặc định khớp với cấu hình trong bài báo (GCN 3 lớp, kích thước ẩn 128, đầu MLP 3 lớp, learning rate 1e-3, batch size 128, tối đa 100 epoch). Để chạy nhanh hơn, giảm số epoch bằng `--epochs`, ví dụ `--epochs 10`.

Các biến thể ablation (Mục 5.6) được huấn luyện với `--no_gnn` hoặc `--no_diff`:

```bash
python -m fuzzy_bisimulation.train --dataset WN18RR --no_gnn --checkpoint_dir checkpoints/WN18RR_no_gnn
python -m fuzzy_bisimulation.evaluate --dataset WN18RR --checkpoint_path checkpoints/WN18RR_no_gnn/best_model.pt --results_dir results/WN18RR_no_gnn
```

## Các lựa chọn khi cài đặt

Bài báo không nêu rõ những chi tiết dưới đây; trong code chúng được cố định như sau.

- **Ngữ nghĩa mờ.** Mặc định là ngữ nghĩa Gödel (t-norm minimum và phép kéo theo tương ứng). Chọn ngữ nghĩa khác bằng `--semantics`.
- **Chuẩn hoá khi fuzzy hoá.** `Freq(r)` là số bộ ba của `r` chia cho số bộ ba của quan hệ phổ biến nhất; `Deg(s, s')` là trung bình bậc theo thang log của hai thực thể. Cả hai nằm trong [0, 1].
- **Chọn nút biên.** Khi một lớp BFS vượt quá số nút cho phép, giữ lại các thực thể có tổng trọng số mờ nối về lớp trước lớn nhất.
- **Đặc trưng nút ban đầu.** Trọng số mờ lớn nhất đi ra và đi vào theo từng loại quan hệ, cùng với bậc ra và bậc vào theo thang log.
- **Lan truyền GCN.** Bộ mã hoá dùng `GCNConv` của `torch_geometric` với trọng số mờ làm trọng số cạnh, giống `main.py` của repo tham chiếu. Thông tin truyền theo chiều của cạnh, nên mỗi thực thể gom thông tin từ các thực thể trỏ vào nó (và chính nó).
- **Baseline Common Neighbors.** Số lân cận chung được chia cho kích thước của lân cận lớn hơn để có điểm trong [0, 1].
