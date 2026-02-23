import { useState, useRef, useEffect } from 'react';
import { Search, Send, Loader2, Link, FileText, CheckCircle, ShieldAlert, Star, Bot, Sparkles } from 'lucide-react';
import './App.css';

// Use the deployed backend URL if in production, otherwise use localhost
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

function App() {
  const [url, setUrl] = useState('');
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoadingProduct, setIsLoadingProduct] = useState(false);
  const [productData, setProductData] = useState(null);
  const [errorItem, setErrorItem] = useState(null);

  const [question, setQuestion] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [chatHistory, setChatHistory] = useState([
    { role: 'ai', text: 'Hi! Paste an Amazon product URL above, load the reviews, and then ask me anything to find the truth.' }
  ]);

  const chatEndRef = useRef(null);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const handleLoadProduct = async () => {
    if (!url.trim()) return;

    setIsLoadingProduct(true);
    setErrorItem(null);
    setProductData(null);
    setIsLoaded(false);

    try {
      const response = await fetch(`${API_BASE}/load-product`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ product_url: url })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to load product');
      }

      setProductData(data);
      setIsLoaded(true);
      setChatHistory(prev => [
        ...prev,
        { role: 'ai', text: `Successfully digested ${data.chunks_created} review constraints from ASIN: ${data.asin}. What do you want to know about this product?` }
      ]);

    } catch (err) {
      setErrorItem(err.message);
    } finally {
      setIsLoadingProduct(false);
    }
  };

  const handleAskQuestion = async () => {
    if (!question.trim() || isAsking || !isLoaded) return;

    const userQ = question.trim();
    setQuestion('');
    setIsAsking(true);

    // Optimistic UI
    setChatHistory(prev => [...prev, { role: 'user', text: userQ }]);

    try {
      const response = await fetch(`${API_BASE}/ask-question`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userQ })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to get answer');
      }

      setChatHistory(prev => [
        ...prev,
        {
          role: 'ai',
          text: data.answer,
          sources: data.sources,
          metrics: data.metrics
        }
      ]);

    } catch (err) {
      setChatHistory(prev => [...prev, { role: 'ai', text: `Error: ${err.message}` }]);
    } finally {
      setIsAsking(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAskQuestion();
    }
  };

  return (
    <div className="app-container">
      <header className="header fade-in">
        <h1>Review<span className="text-gradient">RAG</span></h1>
        <p>Anti-Fake Review Analyzer powered by Hybrid Search & Groq</p>
      </header>

      <section className="input-section fade-in" style={{ animationDelay: '0.1s' }}>
        <input
          type="text"
          className="url-input"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Paste Amazon Product URL or ASIN here..."
          disabled={isLoadingProduct}
        />
        <button
          className="btn-primary"
          onClick={handleLoadProduct}
          disabled={isLoadingProduct || !url.trim()}
        >
          {isLoadingProduct ? <Loader2 className="loading-spinner" size={20} /> : <Search size={20} />}
          {isLoadingProduct ? 'Digesting...' : 'Analyze Product'}
        </button>
      </section>

      {errorItem && (
        <div className="error-banner fade-in" style={{ alignSelf: 'center', minWidth: '300px' }}>
          <ShieldAlert size={20} />
          <span>{errorItem}</span>
        </div>
      )}

      {/* Main Split Interface */}
      <div className="split-pane fade-in" style={{ animationDelay: '0.2s' }}>

        {/* Left Panel: Analytics */}
        <div className="panel glass-panel">
          <div className="panel-header">
            <FileText size={20} color="var(--primary)" />
            Product Insight Engine
          </div>

          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            {!isLoaded ? (
              <div style={{ textAlign: 'center', opacity: 0.5, padding: '2rem' }}>
                <Search size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
                <p>Awaiting Product Data...</p>
                <p style={{ fontSize: '0.85rem', marginTop: '0.5rem' }}>Paste a URL above to initialize the vector space.</p>
              </div>
            ) : (
              <div className="fade-in">
                <div className="stat-card">
                  <div className="stat-value">{productData.status === "success" ? <CheckCircle color="var(--success)" size={32} /> : null}</div>
                  <div className="stat-label">System Readiness</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{productData.chunks_created}</div>
                  <div className="stat-label">Knowledge Chunks Generated</div>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>From verified 3 & 4 star reviews</p>
                </div>
                <div className="stat-card">
                  <div className="stat-value" style={{ fontSize: '1.2rem', color: 'white' }}>{productData.asin}</div>
                  <div className="stat-label">Active ASIN</div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Panel: Chat */}
        <div className="panel glass-panel">
          <div className="panel-header">
            <Search size={20} color="var(--primary)" />
            Truth Query
          </div>

          <div className="chat-history">
            {chatHistory.map((msg, idx) => (
              <div key={idx} className={`chat-bubble ${msg.role} fade-in`}>
                {msg.role === 'ai' && (
                  <div className="ai-header">
                    <Bot size={18} /> ReviewRAG System
                  </div>
                )}
                <div>{msg.text}</div>

                {msg.metrics && (
                  <div className="metrics-container">
                    <div className="metric">
                      <CheckCircle size={12} color="var(--success)" />
                      RAGAS Faithfulness: {(msg.metrics.faithfulness * 100).toFixed(0)}%
                    </div>
                    <div className="metric">
                      <CheckCircle size={12} color="var(--primary)" />
                      Relevancy: {(msg.metrics.answer_relevancy * 100).toFixed(0)}%
                    </div>
                  </div>
                )}

                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources-list">
                    <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '4px' }}>Citational Sources (Top 3):</div>
                    {msg.sources.slice(0, 3).map((src, i) => (
                      <div key={i} className="source-badge" title={src.full_chunk}>
                        <div className="source-header">
                          <span className="rating-stars">
                            {Array(src.rating).fill().map((_, i) => <Star key={i} size={10} fill="currentColor" stroke="none" />)}
                          </span>
                          <span className="score-badge">Relevance: {src.relevance_score.toFixed(2)}</span>
                        </div>
                        "{src.content_excerpt}"
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {isAsking && (
              <div className="chat-bubble ai animate-pulse-slow">
                <Loader2 className="loading-spinner" size={20} style={{ display: 'inline', marginRight: '8px' }} />
                Executing Reciprocal Rank Fusion & Generation...
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <div className="chat-input-area">
            <textarea
              className="chat-input"
              placeholder={isLoaded ? "Ask specific details (e.g., 'Is it noisy?')" : "Load a product first..."}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyPress}
              disabled={!isLoaded || isAsking}
              rows={1}
            />
            <button
              className="send-btn"
              onClick={handleAskQuestion}
              disabled={!isLoaded || isAsking || !question.trim()}
            >
              <Send size={18} />
            </button>
          </div>

        </div>

      </div>
    </div>
  );
}

export default App;
