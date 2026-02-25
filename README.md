# 🔗 AI on Blockchain: Enterprise-Grade Integration Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Solidity](https://img.shields.io/badge/Solidity-0.8.20-363636.svg)](https://soliditylang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)](https://fastapi.tiangolo.com/)

A production-ready platform demonstrating the powerful integration of **Artificial Intelligence** and **Blockchain Technology** for building trustworthy, transparent, and intelligent decentralized applications. This project implements concepts from [GeeksforGeeks: Integration of Blockchain and AI](https://www.geeksforgeeks.org/ethical-hacking/integration-of-blockchain-and-ai/) with state-of-the-art LLM, RAG, MLOps, and Smart Contract technologies.

---

## 🎯 Project Vision

This platform addresses critical challenges in AI trustworthiness and blockchain intelligence by combining:
- **LLM-powered decision making** with blockchain's immutable audit trails
- **RAG (Retrieval Augmented Generation)** for transparent AI reasoning
- **Federated Learning** with decentralized data marketplaces
- **Smart Contract analysis** using fine-tuned LLMs
- **MLOps pipelines** integrated with blockchain for model versioning

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Streamlit)                     │
│  Dashboard │ Model Registry │ Audit Trail │ Data Market    │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────────┐
│              Backend API (FastAPI + Web3.py)                │
│  ┌─────────────┬──────────────┬──────────────────────────┐ │
│  │ LLM Engine  │ RAG Pipeline │ ML Training & Inference   │ │
│  └─────────────┴──────────────┴──────────────────────────┘ │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────────┐
│              Blockchain Layer (Ethereum/Polygon)            │
│  ┌──────────────────┬───────────────────┬─────────────────┐│
│  │ AI Model Registry│ Audit Trail SC    │ Data Marketplace││
│  │ Smart Contract   │ Smart Contract    │ Smart Contract  ││
│  └──────────────────┴───────────────────┴─────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 1. **AI Model Registry on Blockchain**
- Immutable model versioning and metadata storage
- Cryptographic verification of model weights
- Transparent model lineage and training history
- Smart contract-based access control

### 2. **LLM-Powered Fraud Detection**
- Real-time transaction analysis using fine-tuned LLMs
- Anomaly detection with explainable AI
- Blockchain logging of all fraud detection decisions
- Adaptive learning from labeled fraud cases

### 3. **RAG-Based Audit Trail System**
- Vector database (ChromaDB/FAISS) for AI decision storage
- Natural language queries over blockchain audit logs
- Retrieval-augmented explainability for AI decisions
- Compliance reporting with provable data lineage

### 4. **Federated Learning Data Marketplace**
- Privacy-preserving model training across distributed nodes
- Blockchain-based data contribution tracking
- Homomorphic encryption for secure computation
- Smart contract-mediated payments for data providers

### 5. **Smart Contract Security Analyzer**
- Fine-tuned CodeLLaMA/StarCoder for vulnerability detection
- Automated security audit report generation
- Integration with Slither and MythX
- Historical vulnerability pattern learning

### 6. **MLOps with Blockchain**
- Model training orchestration with Ray
- Experiment tracking on-chain (hyperparameters, metrics)
- A/B testing framework with immutable results
- Model serving with versioned endpoints

---

## 📁 Project Structure

```
AI-on-blockchain/
├── contracts/              # Solidity smart contracts
│   ├── AIModelRegistry.sol
│   ├── AuditTrail.sol
│   ├── DataMarketplace.sol
│   └── FraudDetection.sol
├── backend/               # FastAPI backend
│   ├── api/              # API routes
│   ├── blockchain/       # Web3 integration
│   ├── ml/              # ML models & training
│   ├── llm/             # LLM engines (OpenAI, HuggingFace)
│   ├── rag/             # RAG pipeline
│   └── config/          # Configuration files
├── frontend/             # Streamlit dashboard
│   ├── app.py
│   ├── pages/
│   └── components/
├── ml_models/            # Trained models
│   ├── fraud_detection/
│   ├── contract_analyzer/
│   └── embeddings/
├── scripts/              # Deployment & utility scripts
│   ├── deploy_contracts.py
│   ├── train_models.py
│   └── setup_env.py
├── tests/                # Test suites
├── docs/                 # Documentation
├── docker-compose.yml    # Docker orchestration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Node.js 16+ (for Hardhat)
- Docker & Docker Compose
- Ethereum wallet (MetaMask)
- OpenAI API key or HuggingFace token

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/mayankbot01/AI-on-blockchain.git
cd AI-on-blockchain
```

2. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Install blockchain dependencies**
```bash
cd contracts
npm install
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. **Deploy smart contracts (local testnet)**
```bash
npx hardhat node  # In one terminal
python scripts/deploy_contracts.py  # In another terminal
```

6. **Start the backend**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

7. **Launch the frontend**
```bash
cd frontend
streamlit run app.py
```

### Docker Deployment (Recommended)
```bash
docker-compose up --build
```

Access the application at:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 🎓 Technical Implementation Details

### LLM Stack
- **Base Models**: GPT-4, Claude-3, Llama-3-70B
- **Fine-tuning**: LoRA/QLoRA for domain adaptation
- **Frameworks**: LangChain, LlamaIndex, Haystack
- **Prompt Engineering**: Chain-of-Thought, ReAct, Tree of Thoughts

### RAG Pipeline
- **Vector DB**: ChromaDB, FAISS, Pinecone
- **Embeddings**: text-embedding-ada-002, all-MiniLM-L6-v2
- **Chunking**: RecursiveCharacterTextSplitter with overlap
- **Retrieval**: Hybrid search (dense + sparse)

### Blockchain Integration
- **Networks**: Ethereum, Polygon, Arbitrum
- **Libraries**: Web3.py, Ethers.js
- **Standards**: ERC-721 (Model NFTs), ERC-20 (Data tokens)
- **Oracles**: Chainlink for off-chain data

### MLOps Tools
- **Training**: PyTorch, TensorFlow, scikit-learn
- **Experiment Tracking**: MLflow, Weights & Biases
- **Model Serving**: Ray Serve, TorchServe
- **Monitoring**: Prometheus, Grafana

---

## 📊 Use Cases Implemented

### 1. Transparent AI Healthcare Diagnosis
- Medical image analysis with blockchain-logged decisions
- Patient data encrypted and stored on IPFS
- Doctor-AI collaboration with audit trails

### 2. Financial Fraud Detection
- Real-time transaction monitoring with LLMs
- Explainable fraud scoring
- Regulatory compliance reporting

### 3. Supply Chain Intelligence
- IoT data + AI predictions on blockchain
- Automated quality control with provable results
- Smart contract-triggered actions

### 4. Decentralized Model Marketplace
- Buy/sell trained AI models as NFTs
- On-chain model performance metrics
- Revenue sharing for model contributors

---

## 🛠️ API Endpoints

### AI Model Registry
```
POST   /api/v1/models/register          # Register new AI model
GET    /api/v1/models/{model_id}        # Get model details
PUT    /api/v1/models/{model_id}/update # Update model version
GET    /api/v1/models/verify/{hash}     # Verify model integrity
```

### Fraud Detection
```
POST   /api/v1/fraud/analyze             # Analyze transaction
GET    /api/v1/fraud/history             # Get detection history
POST   /api/v1/fraud/feedback            # Submit labeled data
```

### RAG Audit Trail
```
POST   /api/v1/audit/log                 # Log AI decision
GET    /api/v1/audit/query               # Natural language query
GET    /api/v1/audit/compliance-report   # Generate report
```

### Data Marketplace
```
POST   /api/v1/marketplace/list-data     # List dataset
POST   /api/v1/marketplace/purchase      # Purchase data access
GET    /api/v1/marketplace/my-earnings   # View contributor earnings
```

---

## 🔐 Security & Privacy

- **Data Encryption**: AES-256 for data at rest, TLS 1.3 in transit
- **Homomorphic Encryption**: SEAL library for private computations
- **Zero-Knowledge Proofs**: zk-SNARKs for privacy-preserving verification
- **Smart Contract Auditing**: Automated + manual security reviews
- **GDPR Compliance**: Right to erasure via data pointer invalidation

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Smart contract tests
cd contracts && npx hardhat test

# Load testing
locust -f tests/load_test.py
```

---

## 📈 Performance Benchmarks

| Component | Metric | Value |
|-----------|--------|-------|
| LLM Inference | Latency (p95) | <2s |
| Blockchain Write | Gas Cost | ~150k gas |
| RAG Query | Retrieval Time | <500ms |
| Model Training | Federated Round | ~5min |

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Workflow
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📚 Resources & References

- [GeeksforGeeks: AI + Blockchain Integration](https://www.geeksforgeeks.org/ethical-hacking/integration-of-blockchain-and-ai/)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenZeppelin Smart Contracts](https://docs.openzeppelin.com/contracts/)
- [Ray Framework for Distributed ML](https://docs.ray.io/)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Mayank Kumar**
- GitHub: [@mayankbot01](https://github.com/mayankbot01)
- Research Focus: AI/ML, Blockchain, Computer Vision, LLM Applications

---

## 🙏 Acknowledgments

- GeeksforGeeks for the foundational concepts
- OpenAI & HuggingFace for LLM infrastructure
- Ethereum Foundation for blockchain tooling
- Open-source AI/ML community

---

## 🗺️ Roadmap

- [x] Core architecture design
- [x] Smart contract development
- [x] LLM integration
- [x] RAG pipeline implementation
- [ ] Multi-chain support (Solana, Avalanche)
- [ ] Mobile app (React Native)
- [ ] On-chain ML inference (ZKML)
- [ ] Enterprise SaaS offering

---

**Built with ❤️ for the future of trustworthy AI**
