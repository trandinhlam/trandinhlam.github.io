---
layout: post
title:  "Mạng neural nhân tạo cơ bản"
date:   2022-10-07
categories: nnetwork
---

## Mở đầu

Học sâu (Deep Learning) là một lĩnh vực hẹp của machine learning ra đời vào những năm 80-90 của thể kỷ trước,
nhưng mới phát triển mạnh mẽ gần đây nhờ vào lượng dữ liệu lớn khổng lồ sản sinh ra từ những hoạt động của con người trên không gian mạng và ngoài đời thực,
cộng thêm sự gia tăng không ngừng về năng lực tính toán và lưu trữ của các thiết bị máy tính hiện đại.

::: center
![Mối liên hệ giữa trí tuệ nhân tạo, học máy và học sâu](https://editor.analyticsvidhya.com/uploads/945011.png)

{Mối liên hệ giữa trí tuệ nhân tạo, học máy và học sâu}
:::


Học sâu có thể thực hiện cả việc học có giám sát, không giám sát và bán giám sát,
với đặc điểm là cần một lượng mẫu dữ liệu đầu vào đủ lớn và lượng đặc trưng trong từng mẫu dữ liệu là đủ nhiều.
Học sâu tỏ ra không hiệu quả nếu có quá ít mẫu dữ liệu hoặc dữ liệu có quá ít đặc trưng, điều mà các thuật toán học máy truyền thống khác có thể giải quyết rất tốt và tốn ít chi phí huấn luyện hơn các mô hình học sâu.

Ý tưởng của học sâu được mô phỏng dựa trên cấu trúc não bộ của con người và các sinh vật khác, với hàng tỉ tế bào neural thần kinh và hàng nghìn tỉ các kết nối liên kết với nhau giữa chúng tạo thành nhiều lớp.
Mỗi neural đóng vai trò vừa là đơn vị lưu trữ vừa là đơn vị xử lý thông tin. Chúng nhận vào các tín hiệu thần kinh, lan truyền và xuất kết quả dưới dạng tín hiệu thần kinh đã được biến đổi phần nào.
Với vô số các lớp như vậy tạo nên một mạng thần kinh phối hợp với nhau trong quá trình học và phân tích dữ liệu từ các giác quan.

Những ứng dụng cơ bản của học sâu có thể kể đến bài toán nhận dạng chữ viết tay; nhận diện giọng nói; phát hiện đối tượng trong ảnh; phân loại ảnh; nhận diện khuôn mặt;
xử lý ngôn ngữ tự nhiên; dịch máy; sinh-tin học;...

## Perceptron - tế bào của mạng neural nhân tạo
Thuật toán học perceptron là một phương pháp học máy đơn giản, nhưng là nền tảng cơ bản cho các thuật toán khác của deep learning và mạng neural nhân tạo.
Ở đó mỗi perceptron đóng vai trò là một "tế bào" đơn vị xử lý dữ liệu và ghi nhớ lại mức độ điều chỉnh thông qua kinh nghiệm sau mỗi lần tín hiệu đi qua perceptron.
Perceptron dùng để phân loại nhị phân khi dữ liệu chỉ có 2 phân lớp được gán nhãn từ trước, với vector đặc trưng có thể là nhiều chiều.
Giả sử mỗi lớp bao gồm các vector chiếm một vùng nào đó trong không gian vector, nhiệm vụ của perceptron là tìm ra ranh giới phân tách 2 lớp dữ liệu đó,
tức một đường thẳng trong không gian 2 chiều, một mặt phẳng trong không gian 3 chiều, hay một siêu phẳng trong không gian nhiều chiều.
Công thức tổng quát cho một siêu phẳng được thể hiện như phương trình \ref{eqn:perceptron}:
\begin{equation}
    \label{eqn:perceptron}
    f_{w}(X) = \omega_{1}x_{1} + \ldots + \omega_{d}x_{d} + \omega_0 = X^TW +  \omega_0 = 0
\end{equation}
Trong đó:
\begin{itemize}
    \item $x_1, x_2, \ldots,x_d$ là giá trị của mỗi điểm dữ liệu $d$ chiều trong tập dữ liệu, nên $X^T$ coi như là vector đặc trưng của điểm dữ liệu đó.
    \item $\omega_1, \omega_2, \ldots,\omega_d$ là các giá trị trọng số theo từng chiều, nên $W$ là vector trong số của siêu phẳng.
    \item $\omega_0$ là hệ số điều chỉnh để dịch chuyển gốc tọa độ.
\end{itemize}

## Gradient Descent - thuật toán tối ưu phổ biến

## Tài liệu Tham khảo:
+ https://en.wikipedia.org/wiki/Gradient_descent
+ https://machinelearningcoban.com/2017/01/12/gradientdescent/
+ 