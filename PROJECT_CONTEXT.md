# CODING GUIDE

Project: Hệ thống gợi ý phim bằng Luật kết hợp

Ngôn ngữ:

* Python 3.10

Framework:

* Streamlit

Dataset:

* MovieLens Latest Small

Thuật toán:

* Apriori
* FP-Growth

Nhóm:

* Thành viên 1: Thu thập dữ liệu, EDA, Tiền xử lý
* Thành viên 2: Apriori, FP-Growth, Đánh giá
* Thành viên 3: Streamlit App, Báo cáo, Slide, Demo

Yêu cầu giao diện:

* Sử dụng tiếng Việt
* Đơn giản
* Dễ demo
* Không cần đăng nhập
* Tập trung vào chức năng gợi ý phim

Cấu trúc dự án:

data/raw/
data/processed/
models/
src/
app/
reports/
slides/
demo/

Không thay đổi cấu trúc thư mục nếu không được yêu cầu.

Nếu tạo code mới:

* Có type hints khi phù hợp.
* Viết hàm ngắn gọn.
* Có docstring.
* Không hard-code đường dẫn.
* Ưu tiên đọc dữ liệu từ thư mục data.
