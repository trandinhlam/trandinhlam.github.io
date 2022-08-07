---
layout: post
title:  "Kiến trúc cơ bản của python Scrapy!"
date:   2022-02-27
categories: python
---
## Giới thiệu
Đây là bài mô tả kiến trúc của bản của framework [Scrapy](https://docs.scrapy.org/en/latest/), dùng để cạo dữ liệu từ một website, bao gồm thông tin metadata và file (audio/video/image) nếu có. Hi vọng bạn đọc và cả mình sẽ hiểu sâu hơn framework này và nhiều kiến thức hay ho từ nó.


## Kiến trúc Scrapy framework
![Kiến trúc Scrapy](https://doc.scrapy.org/en/latest/_images/scrapy_architecture_02.png)


1. Khởi tạo crawler
2. Phân tích trang web
3. Cài đặt spider
4. Cài đặt File Pipeline để download file về
5. Tổng kết

### Nguồn tham khảo và trích dẫn:
+ https://doc.scrapy.org/en/latest/topics/architecture.html
+ https://docs.scrapy.org
