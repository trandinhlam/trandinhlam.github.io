---
layout: post
title:  "Phân loại các phương pháp trong Machine learning"
date:   2023-01-27
categories: nnetwork
---

## Trí tuệ nhân tạo
Trí tuệ nhân tạo (artificial intelligence) là lĩnh vực nghiên cứu khả năng của hệ thống máy tính có thể
xử lý những tác vụ đòi hỏi trí thông minh của con người.
Nói cách khác, hệ thống máy tính có thể bắt chước nhận thức của con người trong việc thu thập, hiểu thông tin, học tập và giải quyết bài toán đặt ra.
Trí tuệ nhân tạo là nhân tố chính cốt lõi của cuộc cách mạng công nghiệp lần thứ 4,
sau động cơ hơi nước lần đầu tiên, năng lượng điện lần thứ 2 và công nghệ thông tin lần thứ 3.
Trí tuệ nhân tạo được hình thành và phát triển nhờ vào các thành tựu trong toán học, xác suất thống kê, năng lực tính toán của máy tính và quan sát những ý tưởng trong thế giới tự nhiên.
Các mục tiêu cụ thể của trí tuệ nhân tạo có thể chia nhỏ thành các tác vụ như: lý luận và biểu diễn tri thức; xử lý ngôn ngữ tự nhiên;
khả năng học tập và lên kế hoạch; di chuyển và thao tác với đồ vật; robot thông minh;...

## Học máy
Học máy (machine learning) là một tập con của trí tuệ nhân tạo, là lĩnh vực nghiên cứu sự tự học hỏi từ dữ liệu
thu thập trước của hệ thống máy tính, từ đó có thể xử lý được tác vụ cụ thể trên dữ liệu mới mà không cần phải được
lập trình từ trước.
Nói cách khác, máy tính tự hình thành nên cơ chế thích nghi để có thể học qua kinh nghiệm, học qua các ví dụ và các quy luật chung.
Ngày nay, do sự bùng nổ dữ liệu cả về quy mô lẫn tốc độ, các phương pháp machine learning
đã có những bước tiến rất đáng kể để ngày càng đi sâu vào thực tế đời sống nhờ cải thiện được độ chính xác
và tốc độ xử lý tác vụ.

Trong học máy, dữ liệu thô ban đầu được xử lý, biến đổi để rút ra được đặc trưng của tập dữ liệu đó, thể hiện qua tập hợp các đơn vị số học.
Đặc trưng tự rút trích này do con người quản lý, tinh chỉnh, chọn lọc để sau đó đưa vào quá trình huấn luyện lặp đi lặp lại
cho mô hình học máy, sao cho mô hình có thể học được càng nhiều càng tốt.
Dữ liệu sử dụng trong quá trình tạo ra mô hình học máy thường được chia thành 3 tập là tập huấn luyện (training set), tập kiểm tra (test set) và tập xác thực (validation set).
Tập huấn luyện là tập dữ liệu chính và lớn nhất, dạy cho mô hình các bước cần thiết để hình thành nên tri thức ban đầu.
Tập xác thực được sử dụng cuối mỗi vòng lặp huấn luyện, mục đích là để hỗ trợ đánh giá độ hiệu quả của mô hình trong quá trình huấn luyện.
Tập kiểm tra được dùng để đánh giá hiệu quả tổng thể của mô hình sau khi quá trình huấn luyện hoàn tất.
Trên thực tế, các tập dữ liệu sử dụng có thể chuyển đổi và trùng lắp lên nhau mà không gây ảnh hưởng đến tính tổng quát của mô hình.
Ngoài ra dữ liệu cũng có thể được bổ sung nhờ vào công tác thu thập, làm sạch, chọn lọc, thêm bớt dữ liệu qua thời gian một cách thủ công hay tự động.

Trong bất cứ phương pháp học máy nào thì yêu cầu tiên quyết đầu tiên là chất lượng của dữ liệu.
Độ giàu có, phong phú, tổng quát, chuẩn hóa và độ tin cậy của dữ liệu là những yếu tố hàng đầu quyết định đến hiệu quả của phương pháp và chất lượng của lời giải sau cùng.
Vì vậy, việc có được dữ liệu tốt và tiền xử lý dữ liệu ban đầu đóng vai trò rất quan trọng và thường chiếm hầu hết thời gian và công sức trong toàn bộ quá trình giải quyết bài toán.

Về căn bản, dựa trên phương pháp huấn luyện ta có thể chia học máy thành các loại sau như hình:

![Các hướng tiếp cận và phân loại thuật toán học máy]({{ "/assets/images/nnetwork/MachineLearning_Algorithms.png" | relative_url }})

### Học có giám sát (Supervised Learning):

  Học có giám sát là kỹ thuật truyền thống, phổ biến nhất và có ứng dụng nhiều nhất trong học máy, dựa trên việc học từ các ví dụ cho sẵn với mục tiêu được xác định rõ ràng ngay từ đầu.
  Dữ liệu sử dụng là dữ liệu được gán nhãn hoặc giá trị từ trước, từ đó nhiệm vụ của mô hình là dự đoán nhãn với đầu vào chưa biết trước.
  Trong tập dữ liệu gán nhãn, mỗi mẫu dữ liệu cần được biến đổi thành các vector đặc trưng trước khi đưa vào mô hình.
  Tùy vào mức độ quan trọng được đánh giá của từng thành phần trong vector đặc trưng và thuật toán được lựa chọn,
  mô hình trải qua quá trình lặp đi lặp lại nhiều lần từ huấn luyện, đánh giá, kiểm thử độ chính xác và điều chỉnh tham số ở các bước,
  cho tới khi mô hình đạt được hiệu quả tối đa dựa trên một hoặc một vài độ đo nhất định.

  Một số bài toán tiêu biểu của học có giám sát là: phân lớp (classification); hồi quy (regression); nhận dạng mẫu (pattern recognition); mạng thần kinh nhân tạo (neural network);...

### Học không giám sát (Unsupervised Learning):

  Học không giám sát ngược lại với học có giám sát, tức dữ liệu chưa được gán nhãn từ trước, từ đó nhiệm vụ của mô hình là tìm ra quy luật hoặc dạng mẫu của dữ liệu để phân nhóm hoặc tiền xử lý dữ liệu.
  Đặc điểm nổi bật của mô hình học không giám sát là hệ thống thường được huấn luyện trên quần thể dữ liệu khá lớn, ở đó dữ liệu được xử lý theo nhiều cách khác nhau mà không có tiêu chuẩn nào quá cụ thể được đề ra từ trước.

  Các mô hình học không giám sát không xác định cụ thể tập nhãn nào, mà thường thể hiện sự tự tổ chức nội bộ, kết hợp các yếu tố trong không gian chỉ định,
  từ đó mô hình tự sinh ra thông tin tri thức của chính nó.
  Một số bài toán và hướng tiếp cận tiêu biểu là: bài toán phân cụm (clustering); bài toán giảm chiều dữ liệu (dimensionality reduction); mô hình mạng tạo sinh dữ liệu (generative adversarial network);...

### Học tăng cường (Reinforcement Learning):

  Học tăng cường là kỹ thuật học máy nghiên cứu cách thức một chủ thể, khi được đặt trong một môi trường cụ thể, nên đựa ra quyết định lựa chọn thực hiện các hành động nào
  nhằm mục đích cực đại hóa phần thưởng nào đó về lâu dài.
  Tại một trạng thái bất kỳ trong môi trường, chủ thể thực hiện một hành động nào đó để thay đổi trạng thái, và nhận về một phần thưởng hoặc hình phạt nhất định.
  
  Về mặt tổng quát, bài toán học tăng cường cần giải quyết bao gồm nhiều bước theo thứ tự tuần tự, với tập không gian môi trường và các biến số xác định.
  Điểm đặc biệt ở đây là trong hầu hết các trường hợp, các hành động đó không chỉ ảnh hưởng đến phần thưởng trước mắt, mà còn ảnh hưởng đến giá trị các phần thưởng ở những trạng thái tiếp theo, thậm chí rất lâu sau đó.
  Vì tính chất trì hoãn của phần thưởng này, việc tối đa hóa tổng lượng phần thưởng nhận về chung cuộc từ bất kỳ trạng thái hay vị trí xuất phát điểm nào phải được thực hiện theo một chiến lược nhất định tùy vào bài toán.
  Hệ quả là mô hình phải tự khám phá con đường đi và thử lại rất nhiều lần, tối ưu và ghi nhớ từng phản ứng của nó với môi trường.

  Khi học tăng cường, mô hình không cần phải biết trước tri thức về bài toán cần học mà phải tự khám phá ra hành động nào cần phải thực hiện, khác với dynamic programming cần phải chuẩn bị kiến thức từ trước thì mới triển khai được thuật toán.
  Nhờ đó, đối với các bài toán tương tác trong thực tế nơi mà ta không dễ có được các hướng dẫn, ví dụ mẫu đầy đủ cho tất cả các trường hợp,
  thì khả năng tự học từ kinh nghiệm của phương pháp học tăng cường là rất hữu ích.
  Vì vậy, xu hướng hiện nay của học tăng cường là giúp kết nối ngày càng sâu sắc hơn giữa trí tuệ nhân tạo, khoa học máy tính tương tác với máy móc cơ khí điện tử,
  giúp tạo ra những ứng dụng thực tế, có hình dạng gần gũi hơn trong đời sống con người.
  
  Một số bài toán tiêu biểu sử dụng phương pháp học tăng cường là: điều hướng robot; ra quyết định thời gian thực (realtime); chơi cờ; chơi game; xe tự lái;...

