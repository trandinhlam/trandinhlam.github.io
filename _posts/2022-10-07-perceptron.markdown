---
layout: post
title:  "Perceptron: mạng neural nhân tạo cơ bản"
date:   2022-10-07
categories: nnetwork
---

## Mở đầu

Học sâu (Deep Learning) là một lĩnh vực hẹp của machine learning ra đời vào những năm 80-90 của thể kỷ trước,
nhưng mới phát triển mạnh mẽ gần đây nhờ vào lượng dữ liệu lớn khổng lồ sản sinh ra từ những hoạt động của con người trên không gian mạng và ngoài đời thực,
cộng thêm sự gia tăng không ngừng về năng lực tính toán và lưu trữ của các thiết bị máy tính hiện đại.

![Mối liên hệ giữa trí tuệ nhân tạo, học máy và học sâu](https://editor.analyticsvidhya.com/uploads/945011.png)


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
Công thức tổng quát cho một siêu phẳng được thể hiện như phương trình:

![Phương trình siêu phẳng perceptron]({{ "/assets/images/perceptron/perceptron1.png" | relative_url }})

Với đường phẳng phân cách 2 lớp riêng biệt, nhãn của một điểm dữ liệu X bất kỳ được xác định bởi công thức:

![Phương trình siêu phẳng perceptron]({{ "/assets/images/perceptron/perceptron1_labelx.png" | relative_url }})

Chẳng hạn trong không gian 2 chiều, hình sau minh họa đường thẳng phân cách 2 phân lớp xanh và đỏ:

![Phương trình siêu phẳng perceptron]({{ "/assets/images/perceptron/pla.png" | relative_url }})

Cấu trúc cơ bản của siêu phẳng perceptron được đề xuất bởi Rosen-blatt(1958). 
Theo đó, vector đầu vào X sau khi nhân với vector trọng số W sẽ đi qua một hàm kích hoạt (Activation function) để cho ra output. 
Trong trường hợp phân lớp trong không gian 2 chiều thì hàm kích hoạt là hàm sgn().

![Phương trình siêu phẳng perceptron]({{ "/assets/images/perceptron/Rosenblatt-1958.png" | relative_url }})


### Hàm mất mát của perceptron:

Khi một vài điểm dữ liệu bị phân lớp lỗi, tức điểm xanh vuông ( _f<sub>w</sub>(X<sub>i</sub> > 0_) nằm bên
vùng đỏ ( _f<sub>w</sub>(X<sub>i</sub>) < 0_) hoặc ngược lại, tập hợp các điểm bị lỗi đó thành một tập _E_. Giả sử
ta tính độ lỗi phân lớp của thuật toán bằng cách đếm được số lượng phần tử sai, thì hàm
mất mát sẽ là:     
![Phương trình siêu phẳng perceptron]({{ "/assets/images/perceptron/loss.png" | relative_url }})

với yi là nhãn thật của điểm dữ liệu. Mỗi điểm sai có giá trị mất mát là hằng số 1 nên
_L(E)_ là tổng số điểm dữ liệu bị sai. Điều này dẫn tới sự thiếu công bằng khi đánh giá các
điểm sai là như nhau mà không quan tâm đến vị trí hay mức độ sai lệch của điểm dữ liệu
so với đường phân cách. Vì vậy ta cần thay thế giá trị hằng số 1 bằng giá trị khác, chẳng
hạn giá trị _−y<sub>i</sub>fw(x<sub>i</sub>)_ vì tỉ lệ thuận với khoảng cách từ điểm x đến đường phân cách. Khi
điểm dữ liệu lỗi nằm xa đường biên giới, mất mát sẽ nhiều hơn so với các điểm lỗi gần
đường biên giới. Khi đó hàm mất mát được viết lại như sau:

Hàm mất mát trên là một hàm liên tục, đạt cực tiểu = 0 khi không còn điểm lỗi trong
tập E. Vì vậy ta có thể tối ưu hàm này bằng phương pháp phổ biến là Gradient Descent.



## Tài liệu Tham khảo:
+ https://en.wikipedia.org/wiki/Gradient_descent
+ https://machinelearningcoban.com/2017/01/12/gradientdescent/