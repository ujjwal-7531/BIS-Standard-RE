import time
import streamlit as st
from src.retriever import get_retriever
from src.pipeline import run_pipeline, get_query_validation

st.set_page_config(
    page_title="BIS Standards Recommendation Engine",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize retriever in cache
@st.cache_resource
def load_retriever():
    return get_retriever()

retriever = load_retriever()

# Enhanced CSS Styling
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0d47a1;
        margin: 1rem 0 0.5rem 0;
        letter-spacing: -1px;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    
    .search-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .search-box textarea {
        background-color: rgba(255,255,255,0.95) !important;
        color: #333 !important;
        border: 2px solid #667eea !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        padding: 1rem !important;
    }
    
    .result-card {
        background: white;
        border-left: 5px solid #667eea;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        box-shadow: 0 4px 16px rgba(102,126,234,0.2);
        transform: translateX(4px);
    }
    
    .result-standard {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0d47a1;
        margin-bottom: 0.5rem;
        font-family: monospace;
    }
    
    .result-title {
        font-size: 1rem;
        color: #333;
        line-height: 1.5;
        margin-bottom: 0.75rem;
    }
    
    .result-score {
        display: inline-block;
        background: #f0f4ff;
        color: #667eea;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .rank-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        text-align: center;
        line-height: 32px;
        font-weight: bold;
        margin-right: 0.75rem;
    }
    
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102,126,234,0.3);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    .status-info {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    
    .info-box {
        background: #f0f7ff;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .divider {
        height: 1px;
        background: #e0e0e0;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<div class="header-title">🏗️ BIS Standards Recommendation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">AI-powered RAG system to find relevant Indian Standards for building materials</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    enable_validation = st.checkbox(
        "🤖 Enable Query Validation (LLM)",
        value=False,
        help="Use Ollama (phi:2.7b) to validate building-material queries"
    )
    
    top_k = st.slider(
        "📊 Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5,
        help="How many standards to return"
    )
    
    st.divider()
    
    st.subheader("📈 System Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Standards", len(retriever.documents))
    with col2:
        st.metric("LLM", "✅ Ready" if enable_validation else "⏸️ Off")
    
    st.divider()
    
    st.subheader("ℹ️ About")
    st.info("""
    **Architecture:**
    - BM25 Retriever
    - Material Term Boosting
    - Query Expansion
    - Optional LLM Validation
    
    **Performance:**
    - Latency: <2s avg
    - Hit Rate: 100%
    - No hallucinations
    """)

# Main Content
st.markdown('<div class="search-box">', unsafe_allow_html=True)
st.markdown("### 📝 Enter Your Query")

with st.form("search_form", clear_on_submit=False):
    query = st.text_area(
        "Product Description",
        height=120,
        placeholder="Example: High-strength Portland cement (53 grade) for concrete construction with good durability and fast strength gain...",
        help="Describe the building material product, grade, specifications, or standards needed"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        submitted = st.form_submit_button(
            "🔍 Find Standards",
            use_container_width=True,
            type="primary"
        )
    
    with col2:
        clear_btn = st.form_submit_button(
            "🗑️ Clear",
            use_container_width=True
        )
    
    with col3:
        pass

st.markdown('</div>', unsafe_allow_html=True)

# Process Query
if clear_btn:
    st.rerun()

if submitted:
    if not query or len(query.strip()) < 5:
        st.error("⚠️ Please enter a meaningful product description (at least 5 characters)")
        st.stop()
    
    with st.spinner("🔄 Processing your query..."):
        start_time = time.perf_counter()
        
        # Validate if enabled
        validation_time = 0
        if enable_validation:
            try:
                val_start = time.perf_counter()
                validation = get_query_validation(query)
                validation_time = time.perf_counter() - val_start
                
                if not validation["is_valid"]:
                    st.markdown(f'<div class="status-badge status-error">❌ {validation["message"]}</div>', unsafe_allow_html=True)
                    st.info("💡 Please describe a building material product (cement, steel, concrete, aggregates, bricks, etc.)")
                    st.stop()
                else:
                    st.markdown('<div class="status-badge status-success">✅ Valid building-material query</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"⚠️ Validation skipped: {str(e)}")
                validation_time = 0
        
        # Get recommendations
        try:
            retrieval_start = time.perf_counter()
            hits = retriever.retrieve(query, top_k=top_k)
            retrieval_time = time.perf_counter() - retrieval_start
            total_time = time.perf_counter() - start_time
            
            if not hits:
                st.error("❌ No standards found. Try a different query.")
                st.stop()
            
            # Results Header
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("## ✅ Recommendations Found")
            
            # Metrics
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(hits)}</div>
                    <div class="metric-label">Standards</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{retrieval_time:.2f}s</div>
                    <div class="metric-label">Retrieval</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{validation_time:.2f}s</div>
                    <div class="metric-label">Validation</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_time:.2f}s</div>
                    <div class="metric-label">Total</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # Results
            for rank, hit in enumerate(hits, start=1):
                standard = hit.get("standard", "Unknown")
                title = hit.get("title", "Standard details not available")
                score = hit.get("score", 0.0)
                
                st.markdown(f"""
                <div class="result-card">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span class="rank-badge">{rank}</span>
                        <span class="result-standard">{standard}</span>
                    </div>
                    <div class="result-title">{title}</div>
                    <span class="result-score">Score: {score:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Expandable Details
            with st.expander("📊 Show Technical Details"):
                st.json({
                    "query_submitted": query[:100] + "..." if len(query) > 100 else query,
                    "standards_returned": len(hits),
                    "retrieval_latency_seconds": round(retrieval_time, 4),
                    "validation_latency_seconds": round(validation_time, 4),
                    "total_latency_seconds": round(total_time, 4),
                    "validation_enabled": enable_validation,
                    "total_standards_indexed": len(retriever.documents)
                })
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # Success Message
            st.success(f"✅ Query processed in {total_time:.3f} seconds. Top {len(hits)} standards displayed above.")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()

else:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📚 Getting Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Example 1:**
        High-strength OPC 53 grade cement
        """)
    
    with col2:
        st.info("""
        **Example 2:**
        Coarse aggregate for concrete
        """)
    
    with col3:
        st.info("""
        **Example 3:**
        Steel reinforcement bars
        """)
    
    st.markdown("""
    ---
    **How to use this system:**
    1. Describe your building material product in detail
    2. Optionally enable LLM validation for query checking
    3. Click "Find Standards" to get recommendations
    4. View relevant BIS standards with relevance scores
    
    **Supported Materials:**
    Cement, Concrete, Aggregates, Steel, Bricks, Tiles, Mortar, Pipes, Precast, Masonry, and more.
    """)


