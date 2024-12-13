import streamlit as st
import pandas as pd
import pickle
import re
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter

# Load necessary files (best model, stopwords, product data, and tfidf vectorizer)
model_filename = "best_model_SVM.pkl"
vectorizer_filename = "tfidf_vectorizer.pkl"
stopwords_df = pd.read_csv('stopwords_lst.csv')
df_resampled = pd.read_csv('train_resampled.csv')
df_sanpham = pd.read_csv('San_pham.csv')

# Load the best model and tfidf vectorizer
with open(model_filename, 'rb') as file:
    best_model = pickle.load(file)

with open(vectorizer_filename, 'rb') as file:
    tfidf_vectorizer = joblib.load(file)

stopwords_list = stopwords_df['stopwords'].tolist()


# Thêm CSS để tạo định dạng cho sidebar và menu
st.markdown("""
    <style>
        .title {
            text-align: center; /* Căn giữa tiêu đề */
            background-color: #f1f8e9;            
            padding: 20px;
            font-size: 40px;
            font-weight: bold;
            border-radius: 8px;
        }
    
        /* Định dạng cho tiêu đề các section */
        .section-title {
            font-size: 24px;
            font-weight: bold;
            color: #388E3C;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        /* Định dạng cho mục lục trong sidebar */
        .sidebar-section {
            border: 2px solid #4CAF50;  /* Khung viền xanh */
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 8px;
            background-color: #f1f8e9; /* Nền xanh nhạt */
        }
        .sidebar-section h4 {
            color: #388E3C;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .sidebar-section p {
            color: #388E3C;
            font-size: 14px;
            margin: 5px 0;
        }

        /* Định dạng cho các bảng kết quả và hình ảnh */
        .stDataFrame, .stImage {
            border: 2px solid #388E3C;
            border-radius: 8px;
            padding: 10px;
        }

        /* Định dạng cho kết quả dự đoán */
        .prediction-result {
            background-color: #f1f8e9;
            padding: 15px;
            border-radius: 10px;
            margin-top: 10px;
            border: 1px solid #388E3C;
        }
        
        .prediction-result h4 {
            font-size: 18px;
            color: #388E3C;
        }
            
        /* Đảm bảo hình ảnh không bị tràn */
        .stImage img {
            max-width: 100% !important;
            height: auto !important;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
            
        /* Điều chỉnh độ rộng của hình ảnh */
        .wordcloud-container {
            width: 100%; /* Điều chỉnh khung chứa hình ảnh */
            max-width: 800px; /* Giới hạn độ rộng tối đa */
            margin-left: auto;
            margin-right: auto;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Tiêu đề với định dạng CSS
st.markdown('<div class="title">Sentiment Analysis of Hasaki\'s Product Reviews</div>', unsafe_allow_html=True)
st.image('hasaki_banner.jpg', use_container_width=True)

# Menu trong sidebar
menu = ["Tổng quan về dự án", "Xây dựng dự án", "Phân tích sản phẩm", "Phân tích bình luận mới"]
choice = st.sidebar.radio('Mục lục', menu)

# Sidebar content with styled sections
st.sidebar.markdown("""
    <div class="sidebar-section">
        <h4>Thành viên:</h4>
        <p>- Phan Thanh Hải</p>
        <p>- Trần Thị Kim Hoa</p>
    </div>
    
    <div class="sidebar-section">
        <h4>Giáo viên hướng dẫn:</h4>
        <p>Khuất Thùy Phương</p>
    </div>
    
    <div class="sidebar-section">
        <h4>Thời gian báo cáo dự án:</h4>
        <p>14/12/2024</p>
    </div>
""", unsafe_allow_html=True)


if choice == "Tổng quan về dự án":  
    st.markdown('<div class="section-title">1. Mô tả chung về dự án</div>', unsafe_allow_html=True)
    st.markdown("""
    Dự án nhằm phát triển một hệ thống sử dụng **Machine Learning với Python** để phân tích và trích xuất các thông tin giá trị từ các **đánh giá của khách hàng** về sản phẩm mỹ phẩm tại **HASAKI.VN**, giúp nhãn hàng hiểu rõ hơn về cảm nhận và nhu cầu của khách hàng.
    Mục tiêu chính là tối ưu hóa quy trình phân tích đánh giá sản phẩm, giúp nhãn hàng **cải thiện chất lượng sản phẩm và dịch vụ dựa trên phản hồi của người tiêu dùng**.
    """)
    
    st.markdown('<div class="section-title">2. Mục tiêu và yêu cầu</div>', unsafe_allow_html=True)
    st.markdown("""
- **Phân tích đánh giá khách hàng**: Sử dụng **Machine Learning** để phân tích các đánh giá, giúp nhãn hàng hiểu rõ phản hồi và cảm nhận của khách hàng.
- **Phân loại cảm xúc**: Áp dụng các mô hình để phân loại đánh giá thành tích cực, tiêu cực, từ đó đánh giá mức độ hài lòng của khách hàng.
- **Khuyến nghị cải tiến sản phẩm**: Cung cấp các khuyến nghị để cải thiện chất lượng sản phẩm và dịch vụ dựa trên phân tích đánh giá.
- **Tối ưu trải nghiệm người dùng**: Cải thiện trải nghiệm mua sắm qua việc cung cấp thông tin phản hồi chi tiết và hữu ích.
- **Tăng cường dịch vụ khách hàng**: Phân tích các vấn đề khách hàng gặp phải để tối ưu dịch vụ hỗ trợ.
    """)
    
    st.markdown('<div class="section-title">3. Yêu cầu kỹ thuật</div>', unsafe_allow_html=True)
    st.markdown("""
    Sử dụng **Machine learning với Python** và các thư viện như **Scikit-learn...** để xử lý dữ liệu và huấn luyện mô hình phân tích cảm xúc, phân loại đánh giá.
    """)

elif choice == "Xây dựng dự án":
    st.markdown('<div class="section-title">1. Dữ liệu đầu vào</div>', unsafe_allow_html=True)
    # Hiển thị một vài dòng dữ liệu từ đầu vào Danh_gia.csv
    df = pd.read_csv('Danh_gia.csv')
    st.write("Dữ liệu đầu vào (5 dòng đầu tiên):")
    st.dataframe(df.head())
    st.image("Tong_quan_ve_du_lieu.png")
    st.image("Phan_loai_noi_dung_binh_luan.png")
    
    st.markdown('<div class="section-title">2. Tiền xử lý dữ liệu</div>', unsafe_allow_html=True)
    st.markdown("""
- Bước 1: Loại bỏ / xử lý các ký tự đặc biệt, số và các từ dư thừa không mang ý nghĩa thành có ý nghĩa phân tích
- Bước 2: Chuyển đổi tất cả ký tự Unicode về dạng chuẩn
- Bước 3: Phân loại các từ trong bình luận (danh từ, động từ, tính từ,...)
- Bước 4: Loại bỏ Stopwords
- Bước 5: Chuẩn hóa các ký tự lặp lại               
- Bước 6: Xử lý các từ/ icon theo phân loại thông nghiệp ý nghĩa tiêu cực (negative) và tích cực (positive)                
    """)

    st.markdown('<div class="section-title">3. Thêm 2 cột đếm từ tích cực và tiêu cực</div>', unsafe_allow_html=True)
    st.markdown("""
- 2 cột thêm: 'positive_count' và 'negative_count'
- Mục đích: Sử dụng danh sách các từ tích cực và tiêu cực để đếm số từ xuất hiện trong bình luận.
    """)

    st.markdown('<div class="section-title">4. Phân chia dữ liệu và dán nhãn</div>', unsafe_allow_html=True)
    st.markdown("""
- Dán nhãn dữ liệu:
    - Số sao >= 4: 1 tương đương Positive - Tích cực
    - Số sao < 4: 0 tương đương Negative - Tiêu cực
- Xử lý imbalance dữ liệu bằng function resample và chia tập tin thành train và test
    """)
    st.image('du_lieu_train_test.png')        

    st.markdown('<div class="section-title">5. Xây dựng mô hình</div>', unsafe_allow_html=True)
    st.markdown("""
Xây dựng mô hình sử dụng các thuật toán đa dạng như Naive Bayes, SVM và Random Forest.
    """)

    st.markdown('<div class="section-title">6. Các chỉ số đánh giá</div>', unsafe_allow_html=True)
    st.markdown("""
- Mô hình Naive Bayes:
    - Độ chính xác: 92.75%
    - Thời gian huấn luyện: 0.0101s
- Mô hình SVM:
    - Độ chính xác: 98.78%
    - Thời gian huấn luyện: 36.1645s
- Mô hình Random Forest:
    - Độ chính xác: 98.68%
    - Thời gian huấn luyện: 8.9240s

**Lựa chọn mô hình SVM vì cho ra kết quả chính xác nhất: 98.78%.**
""")
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image("model_comparison.png", caption="Model Comparison")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image("confusion_matrix_SVM.png", caption="Confusion Matrix: SVM")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image("confusion_matrix_Naive Bayes.png", caption="Confusion Matrix: Naive Bayes")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image("confusion_matrix_Random Forest.png", caption="Confusion Matrix: Random Forest")
    st.markdown('</div>', unsafe_allow_html=True)

    # Display word clouds
    st.markdown('<div class="section-title">7. WordCloud cho đánh giá tích cực và tiêu cực</div>', unsafe_allow_html=True)    
    # Positive & Negative WordClouds
    all_positive_reviews = ' '.join(df_resampled[df_resampled['label'] == 1]['noi_dung_binh_luan_clean'])
    all_negative_reviews = ' '.join(df_resampled[df_resampled['label'] == 0]['noi_dung_binh_luan_clean'])
    
    positive_wc = WordCloud(width=800, height=400,background_color="#55CBCD").generate(all_positive_reviews)
    negative_wc = WordCloud(width=800, height=400,background_color="#FFB6C1").generate(all_negative_reviews)
    
    # Hiển thị word cloud với khung chứa CSS
    st.markdown('<div class="wordcloud-container">', unsafe_allow_html=True)
    st.image(positive_wc.to_array(), caption="Đánh giá tích cực")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="wordcloud-container">', unsafe_allow_html=True)
    st.image(negative_wc.to_array(), caption="Đánh giá tiêu cực")
    st.markdown('</div>', unsafe_allow_html=True)

    # Hiển thị biểu đồ hình tròn cho phân bố cảm xúc theo sản phẩm
    product_sentiment = df_resampled.groupby('ma_san_pham')['label'].value_counts().unstack().fillna(0)

    # Chọn một sản phẩm để hiển thị biểu đồ
    selected_product_id = st.selectbox("Chọn Mã Sản Phẩm để xem biểu đồ", product_sentiment.index)
    selected_product_sentiment = product_sentiment.loc[selected_product_id]

    # Tạo biểu đồ hình tròn
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(selected_product_sentiment, labels=['Tiêu cực', 'Tích cực'], autopct='%1.1f%%', startangle=90, colors=['#e57373', '#00c853'])
    ax.axis('equal')
    
    # Hiển thị biểu đồ trong Streamlit
    st.markdown('<div class="section-title">8. Phân bố đánh giá của một sản phẩm</div>', unsafe_allow_html=True)
    st.pyplot(fig)

elif choice == "Phân tích sản phẩm":
    # Product Selection
    product_choice = st.radio("**Vui lòng chọn**:", ('Tên sản phẩm', 'Mã sản phẩm'))
    
    if product_choice == 'Tên sản phẩm':
        product_list = df_sanpham['ten_san_pham'].tolist()
        product_name = st.selectbox("Vui lòng chọn sản phẩm", product_list)
        product_id = df_sanpham[df_sanpham['ten_san_pham'] == product_name]['ma_san_pham'].iloc[0]
    elif product_choice == 'Mã sản phẩm':
        product_ids = df_sanpham['ma_san_pham'].tolist()
        product_id = st.selectbox("Vui lòng chọn mã sản phẩm", product_ids)
        product_name = df_sanpham[df_sanpham['ma_san_pham'] == product_id]['ten_san_pham'].iloc[0]
    if st.button('Phân tích sản phẩm'):
        if product_id:
            # Fetch product reviews for the selected product
            product_reviews = df_resampled[df_resampled['ma_san_pham'] == product_id]

            st.markdown('<div class="section-title">1. Thông tin phân tích sản phẩm</div>', unsafe_allow_html=True)
            product_stats = product_reviews.groupby('ma_san_pham').agg(
                total_reviews=('noi_dung_binh_luan_clean', 'count'),
                average_rating=('so_sao', 'mean'),
                positive_reviews=('label', lambda x: (x == 1).sum()),
                negative_reviews=('label', lambda x: (x == 0).sum())
            ).reset_index()

            product_stats['positive_review_percentage'] = (product_stats['positive_reviews'] / product_stats['total_reviews']) * 100
            product_stats['negative_review_percentage'] = (product_stats['negative_reviews'] / product_stats['total_reviews']) * 100

            total_reviews = product_stats['total_reviews'].iloc[0]
            avg_rating = product_stats['average_rating'].iloc[0]
            positive_reviews = product_stats['positive_reviews'].iloc[0]
            negative_reviews = product_stats['negative_reviews'].iloc[0]
            positive_review_percentage = product_stats['positive_review_percentage'].iloc[0]
            negative_review_percentage = product_stats['negative_review_percentage'].iloc[0]

            st.write(f"**Tổng số đánh giá**: {total_reviews}")
            st.write(f"**Điểm đánh giá trung bình**: {avg_rating:.2f}")
            st.write(f"**Tổng đánh giá tích cực**: {positive_reviews} ({positive_review_percentage:.2f}%)")
            st.write(f"**Tổng đánh giá tiêu cực**: {negative_reviews} ({negative_review_percentage:.2f}%)")

        # WordCloud visualization for the reviews
        positive_reviews_text = product_reviews[product_reviews['so_sao'] >= 4]['noi_dung_binh_luan_clean']
        negative_reviews_text = product_reviews[product_reviews['so_sao'] <= 2]['noi_dung_binh_luan_clean']
        
        st.markdown('<div class="section-title">2. WordCloud cho đánh giá tích cực và tiêu cực</div>', unsafe_allow_html=True)

        positive_wc = WordCloud(width=800, height=400, background_color="#55CBCD").generate(' '.join(positive_reviews_text))
        negative_wc = WordCloud(width=800, height=400, background_color="#FFB6C1").generate(' '.join(negative_reviews_text))
        
        # Hiển thị word cloud với khung chứa CSS
        st.markdown('<div class="wordcloud-container">', unsafe_allow_html=True)
        st.image(positive_wc.to_array(), caption="Đánh giá tích cực")
        st.markdown('</div>', unsafe_allow_html=True)
    
        st.markdown('<div class="wordcloud-container">', unsafe_allow_html=True)
        st.image(negative_wc.to_array(), caption="Đánh giá tiêu cực")
        st.markdown('</div>', unsafe_allow_html=True)

        # Top 10 keywords chart
        all_reviews = " ".join(product_reviews['noi_dung_binh_luan_clean'])
        all_reviews_cleaned = " ".join([word.lower() for word in re.findall(r'\b\w+\b', all_reviews) if word.lower() not in stopwords_list])

        word_counts = Counter(all_reviews_cleaned.split())
        common_keywords = word_counts.most_common(10)
        top_keywords_df = pd.DataFrame(common_keywords, columns=["Từ khóa", "Tần suất"])

        # Display bar chart for the top 10 keywords
        st.markdown('<div class="section-title">3. Top 10 từ khóa chính</div>', unsafe_allow_html=True)
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Tần suất', y='Từ khóa', data=top_keywords_df, palette='viridis')
        st.pyplot(plt)
        
elif choice == "Phân tích bình luận mới":
    # Sample CSV file creation function
    def create_sample_csv():
        sample_data = {
            'noi_dung_binh_luan': [
                'Ví dụ bình luận 1',
                'Ví dụ bình luận 2',
                'Ví dụ bình luận 3'
            ]
        }
        sample_df = pd.DataFrame(sample_data)
        return sample_df

    # User input for new comment
    input_type = st.radio("Nhập mới hoặc tải file", ('Nhập mới', 'Tải file'))

    if input_type == 'Nhập mới':
        # Option to manually input comments
        new_comments = st.text_area("Nhập mới bình luận (mỗi dòng là một bình luận)", height=200)

        if st.button('Phân tích bình luận'):
            # Split comments into lines
            comments_list = new_comments.splitlines()

            # Preprocess the comments
            comments_list_clean = [re.sub(r'[^\w\s]', '', comment.lower()) for comment in comments_list]

            # Vectorize and predict sentiment using the pre-trained model
            X_new = tfidf_vectorizer.transform(comments_list_clean)
            predictions = best_model.predict(X_new)

            # Prepare the result as a DataFrame
            sentiment_results = pd.DataFrame({
                'Nội dung bình luận': comments_list,
                'Kết quả phân tích': ['Tích cực' if pred == 0 else 'Tiêu cực' for pred in predictions]
            })

            # Display the result in a table
            st.write("### Kết quả phân tích:")
            st.dataframe(sentiment_results)

            # Add a download button for the result CSV
            result_csv = sentiment_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Tải kết quả phân tích (CSV)",
                data=result_csv,
                file_name='sentiment_analysis_results.csv',
                mime='text/csv'
            )

    elif input_type == 'Tải file':
        # Option to upload new comments file
        uploaded_file = st.file_uploader("Tải lên file bình luận (csv hoặc txt)", type=['csv', 'txt'])

        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                # If it's a CSV file
                new_comments_df = pd.read_csv(uploaded_file)
                comments_list = new_comments_df['noi_dung_binh_luan'].tolist()
            elif uploaded_file.name.endswith('.txt'):
                # If it's a text file
                new_comments = uploaded_file.read().decode('utf-8')
                comments_list = new_comments.splitlines()

            # Preprocess the comments
            comments_list_clean = [re.sub(r'[^\w\s]', '', comment.lower()) for comment in comments_list]

            # Vectorize and predict sentiment using the pre-trained model
            X_new = tfidf_vectorizer.transform(comments_list_clean)
            predictions = best_model.predict(X_new)

            # Prepare the result as a DataFrame
            sentiment_results = pd.DataFrame({
                'Nội dung bình luận': comments_list,
                'Kết quả phân tích': ['Tích cực' if pred == 0 else 'Tiêu cực' for pred in predictions]
            })

            # Display the result in a table
            st.write("### Kết quả phân tích:")
            st.dataframe(sentiment_results)

            # Add a download button for the result CSV
            result_csv = sentiment_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Tải kết quả phân tích (CSV)",
                data=result_csv,
                file_name='sentiment_analysis_results.csv',
                mime='text/csv'
            )

            # Provide a sample CSV for download
        st.write("### Tải file mẫu CSV:")
        sample_df = create_sample_csv()
        sample_file = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Tải file mẫu CSV",
            data=sample_file,
            file_name='sample_comments.csv',
            mime='text/csv'
        )
