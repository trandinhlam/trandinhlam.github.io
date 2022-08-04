---
layout: post
title:  "Audio Processing căn bản - Các phép biến đổi tín hiệu thô cơ bản (Phần 1)"
date:   2022-02-20
categories: machine learning
---
## 1. Ý nghĩa của các phép biến đổi tín hiệu
Khi biểu diễn dữ liệu tín hiệu số, dữ liệu thô thường biểu diễn ở miền thời gian (time-domain) theo trục x và biên độ (amplitude/dB) theo trục y. Tín hiệu thô ban đầu này chứa khá ít thông tin, vì vậy các nhà toán học đã đưa ra các công cụ, công thức để biến đổi tín hiệu thô sang tín hiệu đã được xử lý. Tín hiệu nhận được qua phép biến đổi sẽ được biểu diễn trên một miền khác (chẳng hạn miền tần số- frequency domain). Kể từ khi phép biến đổi đầu tiên được đề xuất, đến nay đã có rất nhiều phép biến đổi khác nhau được tìm ra và áp dụng, với những công dụng cũng như mục đích sử dụng khác nhau. Trong đó Fourier và Wavelet là hai phép biến đổi phổ biến nhất.
## 2. Biến đổi Fourier

Phép biến đổi Fourier Transform (FT) là phép biến đổi 2 chiều từ tín hiệu từ miền thời gian (time-domain) sang miền tần số (frequency-domain) và ngược lại.
FT cho ta biết độ lớn của tín hiệu trong mỗi thành phần tần số là bao nhiêu, từ đó rút ra được một vài tần số chính nổi bật có trong tín hiệu thô ban đầu.  
![image.png](https://images.viblo.asia/77ffd884-68ea-4bca-b833-738ecf1b4233.png)

Phép biến đổi Fourier cho ra được thông tin trung bình toàn cục (global average) của tín hiệu. Nói cách khác, nó chỉ phù hợp khi thực hiện biến đổi trên các tín hiệu có tính tuần hoàn (tín hiệu dừng).  Đối với các trường hợp tìn hiệu chứa nhiều biến động không lường trước được, cần xem xét cục bộ ở những khoảng lấy mẫu ngắn, thì Fourier làm che mất đi thông tin ở cục bộ. Vì vậy cần một phép biến đổi khác có thể trích xuất thông tin cục bộ một cách hiệu quả hơn, đó là biến đổi Wavelet.
## 3. Biến đổi Wavelet
Phép biến đổi Wavelet ngoài việc trích xuất, cho ra được thông tin cục bộ, còn có thể trích xuất được thông tin về thời gian (temporal) của tín hiệu thô