# DictCO

`DictCO` là dự án [mô tả ngắn về mục đích dự án nếu muốn thêm].

## 1. Cài đặt và tải dataset

1. Tạo môi trường ảo Python và kích hoạt:

```bash
python3 -m venv fo_env
source fo_env/bin/activate
```

2. Cài đặt `fiftyone`:

```bash
pip install fiftyone
```

3. Tải dataset:

```bash
python scripts/download_data.py
```


> **Lưu ý [!!! Quan trọng] :** Khi chạy script, nếu thấy dòng thông báo tải CSV, nhấn `CTRL+C` để hủy.
> Sau đó, chạy lại script để chỉ tải ảnh (không cần CSV).

---

## 2. Khởi chạy Docker

Chạy lệnh sau để build và chạy container:

```bash
docker-compose up --build -d
```
