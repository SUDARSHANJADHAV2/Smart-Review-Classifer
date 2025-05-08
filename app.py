import streamlit as st
import joblib
import re
import pandas as pd

# Set up the page
st.set_page_config(
    page_title="Review Classifier",
    page_icon="✨",
    layout="centered"
)

# Color scheme
COLORS = {
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'input_text': '#000000',
    'primary': '#3498db',
    'positive': '#2ecc71',
    'negative': '#e74c3c',
    'neutral': '#95a5a6',
    'card_bg': '#ffffff',
    'analysis_bg': '#f1f8fe'
}

# Load models
@st.cache_resource()
@st.cache_resource()
def load_models():
    return {
        'sentiment': joblib.load('models/sentiment_model.pkl'),
        'review_type': joblib.load('models/reviewtype_model.pkl'),
        'product_category': joblib.load('models/department_model.pkl'),
        'topic_mapping': joblib.load('models/topic_category_mapping.pkl'),
        'review_keywords': joblib.load('models/reviewtype_keywords.pkl'),
        'vectorizer': joblib.load('models/vectorizer.pkl')
    }

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def get_top_keywords(text, keyword_dict, review_type):
    """Get matching keywords for review type classification"""
    text_lower = text.lower()
    keywords = keyword_dict.get(review_type.lower(), [])
    return [kw for kw in keywords if kw in text_lower]

def get_category_keywords(topic_mapping, category):
    """Get representative keywords for a product category"""
    # Reverse mapping to find topic number
    topic_num = [k for k, v in topic_mapping.items() if v == category][0]
    return f"Topic {topic_num} keywords"  # Placeholder - modify based on your data

def main():
    # Custom CSS
    st.markdown(f"""
    <style>
        .stTextArea textarea {{
            background-color: {COLORS['card_bg']} !important;
            border-radius: 8px !important;
            color: {COLORS['input_text']} !important;
        }}
        .stButton>button {{
            background-color: {COLORS['primary']} !important;
            color: white !important;
            border-radius: 8px !important;
        }}
        .analysis-box {{
            background-color: {COLORS['analysis_bg']};
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            border-left: 4px solid {COLORS['primary']};
            color: #000000;  /* Add this line to make text black */
        }}
        .analysis-box h4 {{
            color: #000000;  /* Make headings black */
        }}
        .analysis-box p {{
            color: #000000;  /* Make paragraphs black */
        }}
    </style>
    """, unsafe_allow_html=True)

    st.title("📝 Review Classifier")
    st.markdown("### Paste a product review below to analyze its:")
    st.markdown("• **Sentiment** • **Type** • **Category**")
    review = st.text_area(
        "Enter your review:", 
        height=150,
        placeholder="This product was amazing! The quality exceeded my expectations..."
    )

    if st.button("Analyze", type="primary"):
        if not review.strip():
            st.warning("Please enter a review")
        else:
            models = load_models()
            cleaned_text = clean_text(review)
            
            # Get predictions
            sentiment = models['sentiment'].predict([cleaned_text])[0]
            review_type = models['review_type'].predict([cleaned_text])[0]
            category = models['product_category'].predict([cleaned_text])[0]
            
            # Display basic results
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style='
                    background: {COLORS['card_bg']};
                    padding: 20px;
                    border-radius: 10px;
                    margin: 10px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                '>
                    <h3 style='color: {COLORS['primary']}; margin-top: 0;'>Sentiment</h3>
                    <p style='font-size: 24px; font-weight: bold; color: {COLORS["positive"] if sentiment == "Positive" else COLORS["negative"] if sentiment == "Negative" else COLORS["neutral"]};'>
                        {sentiment}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style='
                    background: {COLORS['card_bg']};
                    padding: 20px;
                    border-radius: 10px;
                    margin: 10px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                '>
                    <h3 style='color: {COLORS['primary']}; margin-top: 0;'>Product Category</h3>
                    <p style='font-size: 24px; font-weight: bold; color: {COLORS['text']};'>
                        {category}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='
                    background: {COLORS['card_bg']};
                    padding: 20px;
                    border-radius: 10px;
                    margin: 10px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                '>
                    <h3 style='color: {COLORS['primary']}; margin-top: 0;'>Review Type</h3>
                    <p style='font-size: 24px; font-weight: bold; color: {COLORS['text']};'>
                        {review_type}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Advanced analysis options
            st.markdown("---")
            st.subheader("Advanced Analysis")
            
            # Create expanders for different analysis types
            with st.expander("Review Type Reason", expanded=False):
                keywords = get_top_keywords(
                    review, 
                    models['review_keywords'], 
                    review_type
                )
                
                st.markdown(f"""
                <div class="analysis-box">
                    <h4>Why this is classified as {review_type}:</h4>
                    {f"<p><strong>Matching keywords:</strong> {', '.join(keywords)}</p>" if keywords else "<p>No specific keywords detected (classified as general feedback)</p>"}
                    <p><strong>Cleaned text used for analysis:</strong></p>
                    <p style='background: #f5f5f5; padding: 10px; border-radius: 4px;'>{cleaned_text}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with st.expander("Product Category Info", expanded=False):
                # Get the topic number for this category
                topic_num = [k for k, v in models['topic_mapping'].items() if v == category][0]
                
                st.markdown(f"""
                <div class="analysis-box">
                    <h4>About {category} category:</h4>
                    <p><strong>Topic Number:</strong> {topic_num}</p>
                    <p><strong>Common terms in this category:</strong></p>
                    <p>Note: Add your category keywords here based on your LDA analysis</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()