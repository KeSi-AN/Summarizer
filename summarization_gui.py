import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from extractive_summarizer import ExtractiveSummarizer
from rouge_score import rouge_scorer
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import json
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Extractive Text Summarization Tool",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .summary-box {
        background: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #b8daff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'summarizer_models' not in st.session_state:
    st.session_state.summarizer_models = {
        'TF-IDF': ExtractiveSummarizer(method='tfidf'),
        'TextRank': ExtractiveSummarizer(method='textrank'),
        'Hybrid': ExtractiveSummarizer(method='hybrid')
    }

if 'summary_history' not in st.session_state:
    st.session_state.summary_history = []

def main():
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🔬 Extractive Text Summarization Tool")
    st.markdown("**Advanced Multi-Method Summarization with Performance Analytics**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    selected_method = st.sidebar.selectbox(
        "Choose Summarization Method",
        ["TF-IDF", "TextRank", "Hybrid"],
        index=2,
        help="Select the algorithm for extractive summarization"
    )
    
    # Summary parameters
    summary_ratio = st.sidebar.slider(
        "Summary Ratio",
        min_value=0.1,
        max_value=0.8,
        value=0.3,
        step=0.1,
        help="Percentage of original text to include in summary"
    )
    
    max_sentences = st.sidebar.number_input(
        "Max Sentences",
        min_value=1,
        max_value=20,
        value=5,
        help="Maximum number of sentences in summary"
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summarize", "📊 Analytics", "🔍 Compare Methods", "📈 Performance"])
    
    with tab1:
        summarize_tab()
    
    with tab2:
        analytics_tab()
    
    with tab3:
        compare_methods_tab()
    
    with tab4:
        performance_tab()

def summarize_tab():
    st.header("Text Summarization")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "File Upload", "Sample Text"],
        horizontal=True
    )
    
    text_input = ""
    
    if input_method == "Text Input":
        text_input = st.text_area(
            "Enter your text here:",
            height=200,
            placeholder="Paste your article or document here..."
        )
    
    elif input_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Upload a text file",
            type=['txt', 'md'],
            help="Upload a .txt or .md file"
        )
        
        if uploaded_file is not None:
            text_input = str(uploaded_file.read(), "utf-8")
            st.success(f"File uploaded successfully! ({len(text_input)} characters)")
    
    elif input_method == "Sample Text":
        sample_texts = {
            "AI and Machine Learning": """
            Artificial intelligence and machine learning have revolutionized numerous industries over the past decade. 
            These technologies enable computers to learn from data and make decisions with minimal human intervention. 
            Machine learning algorithms can process vast amounts of information quickly and identify patterns that humans might miss. 
            Companies across sectors like healthcare, finance, and transportation are leveraging AI to improve efficiency and innovation. 
            However, the rapid advancement of AI also raises important questions about job displacement, privacy, and ethical implications. 
            Experts emphasize the need for responsible AI development and implementation. 
            Educational institutions are adapting their curricula to prepare students for an AI-driven future. 
            The collaboration between humans and AI systems is becoming increasingly important in the modern workplace.
            """,
            "Climate Change": """
            Climate change represents one of the most pressing challenges of our time, with far-reaching consequences for the planet. 
            Rising global temperatures are causing ice caps to melt, leading to sea level rise and threatening coastal communities worldwide. 
            Extreme weather events, including hurricanes, droughts, and floods, are becoming more frequent and severe. 
            The scientific consensus indicates that human activities, particularly the burning of fossil fuels, are the primary drivers of climate change. 
            Governments and organizations worldwide are implementing policies to reduce greenhouse gas emissions and transition to renewable energy sources. 
            Solar and wind power technologies have become increasingly cost-effective and efficient. 
            International cooperation through agreements like the Paris Climate Accord is essential for addressing this global challenge. 
            Individual actions, such as reducing energy consumption and supporting sustainable practices, also play a crucial role in combating climate change.
            """
        }
        
        selected_sample = st.selectbox("Choose a sample text:", list(sample_texts.keys()))
        text_input = sample_texts[selected_sample]
        st.text_area("Sample text:", value=text_input, height=150, disabled=True)
    
    # Summarization
    if text_input and st.button("Generate Summary", type="primary"):
        if len(text_input.strip()) < 50:
            st.error("Please provide a longer text (at least 50 characters)")
        else:
            with st.spinner("Generating summary..."):
                # Get selected model
                selected_method = st.session_state.get('selected_method', 'Hybrid')
                summarizer = st.session_state.summarizer_models[selected_method]
                
                # Generate summary
                summary = summarizer.summarize(
                    text_input,
                    summary_ratio=st.session_state.get('summary_ratio', 0.3),
                    max_sentences=st.session_state.get('max_sentences', 5)
                )
                
                # Display results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📄 Original Text")
                    st.text_area("", value=text_input, height=300, disabled=True)
                    
                    # Text statistics
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Original Length", f"{len(text_input)} characters")
                    st.metric("Word Count", f"{len(text_input.split())} words")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.subheader("✨ Generated Summary")
                    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                    
                    # Summary statistics
                    compression_ratio = len(summary) / len(text_input)
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Summary Length", f"{len(summary)} characters")
                    st.metric("Compression Ratio", f"{compression_ratio:.2%}")
                    st.metric("Method Used", selected_method)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Save to history
                st.session_state.summary_history.append({
                    'original': text_input,
                    'summary': summary,
                    'method': selected_method,
                    'compression_ratio': compression_ratio,
                    'timestamp': pd.Timestamp.now()
                })
                
                st.success("Summary generated successfully!")

def analytics_tab():
    st.header("Text Analytics")
    
    if not st.session_state.summary_history:
        st.info("No summaries generated yet. Go to the 'Summarize' tab to create some summaries first.")
        return
    
    # Convert history to DataFrame
    df_history = pd.DataFrame(st.session_state.summary_history)
    
    # Analytics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Summaries", len(df_history))
    
    with col2:
        avg_compression = df_history['compression_ratio'].mean()
        st.metric("Avg Compression", f"{avg_compression:.2%}")
    
    with col3:
        most_used_method = df_history['method'].mode().iloc[0] if not df_history.empty else "N/A"
        st.metric("Most Used Method", most_used_method)
    
    with col4:
        avg_summary_length = df_history['summary'].str.len().mean()
        st.metric("Avg Summary Length", f"{avg_summary_length:.0f} chars")
    
    # Visualizations
    st.subheader("📊 Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Method usage pie chart
        method_counts = df_history['method'].value_counts()
        fig_pie = px.pie(
            values=method_counts.values,
            names=method_counts.index,
            title="Summarization Method Usage"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Compression ratio distribution
        fig_hist = px.histogram(
            df_history,
            x='compression_ratio',
            title="Compression Ratio Distribution",
            nbins=10
        )
        fig_hist.update_xaxis(title="Compression Ratio")
        fig_hist.update_yaxis(title="Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Timeline of summaries
    if len(df_history) > 1:
        fig_timeline = px.line(
            df_history,
            x='timestamp',
            y='compression_ratio',
            color='method',
            title="Compression Ratio Over Time"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

def compare_methods_tab():
    st.header("Method Comparison")
    
    st.info("Compare different summarization methods on the same text")
    
    # Input text for comparison
    comparison_text = st.text_area(
        "Enter text to compare methods:",
        height=150,
        placeholder="Enter a longer text to see how different methods perform..."
    )
    
    if comparison_text and st.button("Compare All Methods"):
        if len(comparison_text.strip()) < 50:
            st.error("Please provide a longer text for meaningful comparison")
        else:
            with st.spinner("Comparing methods..."):
                results = {}
                
                # Generate summaries with all methods
                for method_name, summarizer in st.session_state.summarizer_models.items():
                    summary = summarizer.summarize(comparison_text, summary_ratio=0.3)
                    results[method_name] = {
                        'summary': summary,
                        'length': len(summary),
                        'compression': len(summary) / len(comparison_text),
                        'word_count': len(summary.split())
                    }
                
                # Display comparison
                st.subheader("📋 Method Comparison Results")
                
                # Create comparison table
                comparison_data = []
                for method, data in results.items():
                    comparison_data.append({
                        'Method': method,
                        'Summary Length': data['length'],
                        'Word Count': data['word_count'],
                        'Compression Ratio': f"{data['compression']:.2%}"
                    })
                
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
                
                # Display summaries
                for method, data in results.items():
                    with st.expander(f"{method} Summary"):
                        st.markdown(f'<div class="summary-box">{data["summary"]}</div>', unsafe_allow_html=True)
                
                # Visualization
                methods = list(results.keys())
                compressions = [results[m]['compression'] for m in methods]
                lengths = [results[m]['length'] for m in methods]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_bar1 = go.Figure(data=[go.Bar(x=methods, y=compressions)])
                    fig_bar1.update_layout(title="Compression Ratio by Method")
                    fig_bar1.update_yaxis(title="Compression Ratio")
                    st.plotly_chart(fig_bar1, use_container_width=True)
                
                with col2:
                    fig_bar2 = go.Figure(data=[go.Bar(x=methods, y=lengths)])
                    fig_bar2.update_layout(title="Summary Length by Method")
                    fig_bar2.update_yaxis(title="Characters")
                    st.plotly_chart(fig_bar2, use_container_width=True)

def performance_tab():
    st.header("Performance Metrics")
    
    # Load evaluation results if available
    results_file = "results/evaluation_results.json"
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            evaluation_data = json.load(f)
        
        st.subheader("🎯 Model Performance Results")
        
        # Display performance metrics
        results = evaluation_data['evaluation_results']
        
        # Create performance dataframe
        performance_data = []
        for model, scores in results.items():
            performance_data.append({
                'Model': model,
                'ROUGE-1': scores['rouge1'],
                'ROUGE-2': scores['rouge2'],
                'ROUGE-L': scores['rougeL']
            })
        
        df_performance = pd.DataFrame(performance_data)
        
        # Display metrics table
        st.dataframe(df_performance.round(4), use_container_width=True)
        
        # Performance visualization
        fig = px.bar(
            df_performance.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
            x='Model',
            y='Score',
            color='Metric',
            title="ROUGE Scores Comparison",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model highlight
        best_model = evaluation_data['best_model']
        best_rouge1 = results[best_model]['rouge1']
        
        st.success(f"🏆 Best Model: {best_model} (ROUGE-1: {best_rouge1:.4f})")
        
        # Accuracy check
        accuracy_threshold = 0.7
        if best_rouge1 >= accuracy_threshold:
            st.success(f"✅ Accuracy Target Met: {best_rouge1:.1%} ≥ 70%")
        else:
            st.warning(f"⚠️ Accuracy Target Not Met: {best_rouge1:.1%} < 70%")
        
        # Additional metrics
        if 'performance_metrics' in evaluation_data:
            perf_metrics = evaluation_data['performance_metrics']
            
            st.subheader("📊 Classification Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{perf_metrics['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{perf_metrics['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{perf_metrics['recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{perf_metrics['f1_score']:.3f}")
    
    else:
        st.info("No evaluation results found. Run the model training notebook first to generate performance metrics.")
        
        # Show dummy performance data for demonstration
        st.subheader("📈 Expected Performance Metrics")
        
        dummy_data = {
            'TF-IDF': {'ROUGE-1': 0.68, 'ROUGE-2': 0.45, 'ROUGE-L': 0.62},
            'TextRank': {'ROUGE-1': 0.72, 'ROUGE-2': 0.48, 'ROUGE-L': 0.65},
            'Hybrid': {'ROUGE-1': 0.75, 'ROUGE-2': 0.52, 'ROUGE-L': 0.68}
        }
        
        for method, scores in dummy_data.items():
            with st.expander(f"{method} Performance"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ROUGE-1", f"{scores['ROUGE-1']:.3f}")
                with col2:
                    st.metric("ROUGE-2", f"{scores['ROUGE-2']:.3f}")
                with col3:
                    st.metric("ROUGE-L", f"{scores['ROUGE-L']:.3f}")

if __name__ == "__main__":
    main()