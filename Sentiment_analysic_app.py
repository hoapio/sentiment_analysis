import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import re
from collections import Counter
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, classification_report

# Load necessary files (best model, stopwords, product data)
model_filename = "best_model_SVM.pkl"
stopwords_df = pd.read_csv('stopwords_lst.csv')
df_resampled = pd.read_csv('train_resampled.csv')
df_sanpham = pd.read_csv('San_pham.csv')

# Load the best model
with open(model_filename, 'rb') as file:
    best_model = pickle.load(file)

# Load stopwords
stopwords = set(stopwords_df['stopwords'].str.lower())

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    max_df=0.9,  # Ignore terms that appear in more than 90% of the documents
    min_df=2     # Include terms that appear in at least 2 documents
)

# Function to clean and preprocess text
def clean_text(text):
    text = " ".join([word.lower() for word in re.findall(r'\b\w+\b', text)])
    text = " ".join([word for word in text.split() if word not in stopwords])
    return text

# Streamlit app layout
st.title("Sentiment Analysis for Hasaki.vn Product Reviews")
st.image('hasaki_banner.jpg', use_container_width= True)

st.sidebar.write("""
    #### Thành viên:
    - Phan Thanh Hải
    - Trần Thị Kim Hoa
""")
st.sidebar.write("""
    #### Giáo viên hướng dẫn: Khuất Thùy Phương
    #### Thời gian báo cáo dự án: 14/12/2024
""")

tab1, tab2, tab3 = st.tabs(["Business Objective", "Insight", "Prediction"])

with tab1:  
    st.write("#### Mô tả chung về dự án")  
    st.markdown("""
    Dự án xây dựng một hệ thống phân tích và dự đoán cảm xúc khách hàng dựa trên các đánh giá đã có trước đó. Dữ liệu được thu thập từ phần bình luận và đánh giá của khách hàng trên Hasaki.vn, tạo ra một kho thông tin phong phú và đa dạng.
    """)
    st.write("#### Mục tiêu")
    st.markdown("""
    Hướng tới xây dựng một mô hình dự đoán mạnh mẽ, giúp Hasaki.vn nhanh chóng nắm bắt được phản hồi của khách hàng về sản phẩm hoặc dịch vụ (tích cực hoặc tiêu cực). Điều này không chỉ giúp họ cải thiện chất lượng sản phẩm và dịch vụ, mà còn mang lại sự hài lòng cao nhất cho khách hàng.
    """)
    st.write("#### Thuật toán sử dụng")
    st.markdown("""
    Machine Learning với Python
    """)

with tab2:
    st.write("#### Hiệu suất và Đánh giá mô hình")
    st.markdown("""
    Xây dựng một mô hình sử dụng các thuật toán đa dạng như Naive Bayes, SVM và Random Forest.
    Mô hình này được huấn luyện trên các đánh giá của khách hàng về sản phẩm trên Hasaki.vn để phân loại chúng thành các cảm xúc tích cực hoặc tiêu cực.
    """)

    st.write("#### Các chỉ số đánh giá")
    st.markdown("""
    #### a. **Accuracy, Confusion Matrix, and Classification Report**:
    The best model is SVM with an accuracy of **98.78%**.
    """)

    st.text("Best Model is SVM with accuracy: 0.9878419452887538 and training time: 39.7541 seconds")
    st.image("model_comparison.png", caption="Model Comparison")
    st.image("confusion_matrix_Naive Bayes.png", caption="Confusion Matrix: Naive Bayes")
    st.image("confusion_matrix_Random Forest.png", caption="Confusion Matrix: Random Forest")
    st.image("confusion_matrix_SVM.png", caption="Confusion Matrix: SVM")

    # Display word clouds
    st.subheader("b. Word Clouds: Positive vs Negative Reviews")
    col1, col2 = st.columns(2)
    
    # Positive & Negative WordClouds
    all_positive_reviews = ' '.join(df_resampled[df_resampled['label'] == 1]['noi_dung_binh_luan_clean'])
    all_negative_reviews = ' '.join(df_resampled[df_resampled['label'] == 0]['noi_dung_binh_luan_clean'])
    
    with col1:
        positive_wc = WordCloud(width=800, height=400).generate(all_positive_reviews)
        plt.imshow(positive_wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    
    with col2:
        negative_wc = WordCloud(width=800, height=400).generate(all_negative_reviews)
        plt.imshow(negative_wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

    # Hiển thị biểu đồ thanh cho phân bố cảm xúc theo sản phẩm
    product_sentiment = df_resampled.groupby('ma_san_pham')['label'].value_counts().unstack().fillna(0)
    product_sentiment.plot(kind='bar', stacked=True, figsize=(10, 5), color=['#e57373', '#00c853'])

    # Thêm tiêu đề và nhãn trục
    plt.title("Sentiment Distribution by Product")
    plt.xlabel("Product ID")
    plt.ylabel("Number of Reviews")

    # Thêm chú thích cho các nhãn
    plt.legend(['Label 0: Tiêu cực', 'Label 1: Tích cực'])

    # Hiển thị biểu đồ trong Streamlit
    st.subheader("c. Phân bố cảm xúc theo sản phẩm của tất cả sản phẩm")
    st.pyplot(plt)

    # Hiển thị biểu đồ hình tròn cho phân bố cảm xúc theo sản phẩm
    product_sentiment = df_resampled.groupby('ma_san_pham')['label'].value_counts().unstack().fillna(0)

    # Chọn một sản phẩm để hiển thị biểu đồ
    selected_product_id = st.selectbox("Chọn Mã Sản Phẩm để xem biểu đồ cảm xúc", product_sentiment.index)
    selected_product_sentiment = product_sentiment.loc[selected_product_id]

    # Tạo biểu đồ hình tròn
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(selected_product_sentiment, labels=['Tiêu cực', 'Tích cực'], autopct='%1.1f%%', startangle=90, colors=['#e57373', '#00c853'])
    ax.axis('equal')  # Đảm bảo biểu đồ hình tròn là hình tròn
    plt.title(f"Phân bố cảm xúc của sản phẩm: {selected_product_id}")

    # Hiển thị biểu đồ trong Streamlit
    st.subheader("d. Phân bố cảm xúc theo một sản phẩm")
    st.pyplot(fig)


with tab3:
    # 1. Product Selection or New Review Input
    st.subheader("Vui lòng lựa chọn sản phẩm hoặc thêm bình luận mới để phân tích:")

    # Option 1: Select product by name or ID
    product_choice = st.radio("Vui lòng chọn:", ('Tên sản phẩm', 'Mã sản phẩm'))

    if product_choice == 'Tên sản phẩm':
        product_list = df_sanpham['ten_san_pham'].tolist()
        product_name = st.selectbox("Vui lòng chọn sản phẩm", product_list)
        product_id = df_sanpham[df_sanpham['ten_san_pham'] == product_name]['ma_san_pham'].iloc[0]
    elif product_choice == 'Mã sản phẩm':
        product_ids = df_sanpham['ma_san_pham'].tolist()
        product_id = st.selectbox("Vui lòng chọn mã sản phẩm", product_ids)
        product_name = df_sanpham[df_sanpham['ma_san_pham'] == product_id]['ten_san_pham'].iloc[0]

    # Option 2: Enter a new review
    new_review = st.text_area("Hoặc nhập bình luận mới:", "")

    if product_id:
        # Fetch product reviews for the selected product
        product_data = df_sanpham[df_sanpham['ma_san_pham'] == product_id]
        product_name = product_data['ten_san_pham'].iloc[0]

        # Filter reviews for the selected product
        product_reviews = df_resampled[df_resampled['ma_san_pham'] == product_id]

        st.subheader("Thông tin phân tích cảm xúc của sản phẩm:")

        # Basic statistics of the selected product (converted to text)
        product_stats = product_reviews.groupby('ma_san_pham').agg(
            total_reviews=('noi_dung_binh_luan_clean', 'count'),
            average_rating=('so_sao', 'mean'),
            positive_reviews=('label', lambda x: (x == 1).sum()),
            negative_reviews=('label', lambda x: (x == 0).sum())
        ).reset_index()

        product_stats['positive_review_percentage'] = (product_stats['positive_reviews'] / product_stats['total_reviews']) * 100
        product_stats['negative_review_percentage'] = (product_stats['negative_reviews'] / product_stats['total_reviews']) * 100

        # Display statistics as text
        total_reviews = product_stats['total_reviews'].iloc[0]
        avg_rating = product_stats['average_rating'].iloc[0]
        positive_reviews = product_stats['positive_reviews'].iloc[0]
        negative_reviews = product_stats['negative_reviews'].iloc[0]
        positive_review_percentage = product_stats['positive_review_percentage'].iloc[0]
        negative_review_percentage = product_stats['negative_review_percentage'].iloc[0]

        st.write(f"**Sản phẩm:** {product_name}")
        st.write(f"Total Reviews: {total_reviews}")
        st.write(f"Average Rating: {avg_rating:.2f}")
        st.write(f"Positive Reviews: {positive_reviews} ({positive_review_percentage:.2f}%)")
        st.write(f"Negative Reviews: {negative_reviews} ({negative_review_percentage:.2f}%)")

        # WordCloud visualization for the reviews
        positive_reviews_text = product_reviews[product_reviews['so_sao'] >= 4]['noi_dung_binh_luan_clean']
        negative_reviews_text = product_reviews[product_reviews['so_sao'] < 4]['noi_dung_binh_luan_clean']
        
        positive_text = ' '.join(positive_reviews_text)
        negative_text = ' '.join(negative_reviews_text)

        st.subheader("WordCloud Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            if positive_reviews_text.any():
                positive_wc = WordCloud(width=800, height=400).generate(positive_text)
                plt.imshow(positive_wc, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            else:
                st.write("No data for positive reviews")

        with col2:
            if negative_reviews_text.any():
                negative_wc = WordCloud(width=800, height=400).generate(negative_text)
                plt.imshow(negative_wc, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
            else:
                st.write("No data for negative reviews")

        # Top 10 keywords chart
        all_reviews = " ".join(product_reviews['noi_dung_binh_luan_clean'])
        all_reviews_cleaned = " ".join([word.lower() for word in re.findall(r'\b\w+\b', all_reviews) if word.lower() not in stopwords])

        word_counts = Counter(all_reviews_cleaned.split())
        common_keywords = word_counts.most_common(10)
        top_keywords_df = pd.DataFrame(common_keywords, columns=["Keyword", "Frequency"])

        st.subheader("Top 10 Keywords")
        st.dataframe(top_keywords_df)

        # Display bar chart for the top 10 keywords
        st.subheader("Top 10 Keywords Bar Chart")
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Frequency', y='Keyword', data=top_keywords_df, palette='viridis')
        plt.title("Top 10 Keywords in Reviews")
        st.pyplot(plt)

    else:
        st.write("Please select a product or enter a new review.")  
