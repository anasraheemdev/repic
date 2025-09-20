# 🚀 Repic - AI-Powered Instagram Content Generator

A sophisticated RAG (Retrieval-Augmented Generation) + NLP application that creates viral Instagram captions, engaging content scripts, and strategic hook placements to maximize social media engagement and reach.

## ✨ Features

- **🎯 Viral Caption Generation**: AI-powered captions optimized for Instagram engagement
- **📝 Content Script Creation**: Complete scripts for Instagram posts, reels, and stories
- **🎣 Strategic Hook Analysis**: Identifies optimal hooks and suggests placement strategies
- **🔍 RAG-Enhanced Content**: Leverages retrieval-augmented generation for contextually relevant content
- **🧠 NLP Processing**: Advanced natural language processing for content optimization
- **📊 Engagement Optimization**: Content tailored for maximum viral potential

## 🛠️ Tech Stack

- **Backend**: Python
- **NLP Framework**: Advanced natural language processing models
- **Vector Storage**: Efficient vector database for RAG implementation
- **Models**: Custom-trained models for Instagram content generation
- **Caching**: Optimized caching system for faster response times

## 📁 Project Structure

```
├── __pycache__/          # Python cache files
├── data/                 # Training data and datasets
├── models/              # Trained ML models and configurations
├── templates/           # Content templates and formats
├── vector_store/        # Vector database for RAG system
├── app.py              # Main application entry point
├── model_utils.py      # Model utilities and helper functions
├── rag_utils.py        # RAG system implementation
├── requirements.txt    # Project dependencies
└── .gitignore         # Git ignore configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/anasraheemdev/repic.git
   cd repic
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize the vector store**
   ```bash
   python -c "from rag_utils import initialize_vector_store; initialize_vector_store()"
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

## 🎯 Usage

### Generate Viral Captions

```python
from app import RepicGenerator

generator = RepicGenerator()

# Generate caption for a specific niche
caption = generator.generate_caption(
    topic="fitness motivation",
    style="inspirational", 
    target_audience="young adults"
)

print(caption)
```

### Create Content Scripts

```python
# Generate complete content script
script = generator.create_script(
    content_type="reel",
    duration=30,
    topic="productivity tips",
    hook_style="question"
)

print(script)
```

### Hook Analysis and Placement

```python
# Analyze and suggest hook placements
hook_analysis = generator.analyze_hooks(
    content="Your existing content here...",
    target_engagement="viral"
)

print(f"Suggested hooks: {hook_analysis['hooks']}")
print(f"Optimal placements: {hook_analysis['placements']}")
```

## 📊 Content Types Supported

- **📸 Photo Posts**: Static image captions with engagement hooks
- **🎥 Reels**: Short-form video scripts with viral elements
- **📖 Carousel Posts**: Multi-slide content with narrative flow
- **📱 Stories**: Ephemeral content with immediate engagement focus
- **🎬 IGTV**: Long-form video scripts with structured hooks

## 🎣 Hook Strategies

The application identifies and implements various hook types:

- **Question Hooks**: Engaging questions that prompt responses
- **Controversial Hooks**: Thought-provoking statements that spark debate
- **Curiosity Hooks**: Intriguing openings that create suspense
- **Emotional Hooks**: Content that triggers strong emotional responses
- **Trend Hooks**: Timely references to current trends and events

## ⚙️ Configuration

### Model Configuration

Edit `models/config.json` to customize:

```json
{
  "generation_model": "gpt-3.5-turbo",
  "max_caption_length": 2200,
  "hook_density": 0.3,
  "engagement_threshold": 0.85,
  "content_categories": ["lifestyle", "business", "fitness", "travel"]
}
```

### RAG Settings

Configure retrieval settings in `rag_utils.py`:

```python
RAG_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "similarity_threshold": 0.7,
    "max_retrieved_docs": 5
}
```

## 📈 Performance Optimization

- **Vector Caching**: Pre-computed embeddings for faster retrieval
- **Model Optimization**: Quantized models for reduced memory usage
- **Batch Processing**: Efficient handling of multiple requests
- **Smart Caching**: Context-aware caching of generated content

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_caption_generation.py
python -m pytest tests/test_hook_analysis.py
python -m pytest tests/test_rag_system.py
```

## 📊 Analytics and Metrics

The application tracks:

- **Generation Success Rate**: Percentage of successful content generation
- **Hook Effectiveness**: Performance metrics for different hook types
- **Engagement Predictions**: Estimated engagement rates for generated content
- **Content Variety**: Diversity metrics for generated content

## 🔧 API Endpoints

If running as a web service:

```bash
POST /api/generate-caption
POST /api/create-script
POST /api/analyze-hooks
POST /api/optimize-content
GET /api/content-templates
GET /api/analytics
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build the image
docker build -t repic-generator .

# Run the container
docker run -p 8080:8080 repic-generator
```

### Cloud Deployment

The application supports deployment on:
- AWS EC2/Lambda
- Google Cloud Platform
- Heroku
- DigitalOcean

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 💡 Key Features Deep Dive

### 🎯 Viral Caption Generation
Repic analyzes thousands of high-performing Instagram posts to understand what makes content go viral. The AI considers:
- Trending topics and hashtags
- Audience engagement patterns
- Optimal caption length and structure
- Emotional triggers and psychological hooks

### 🎣 Hook Strategy Intelligence
The application identifies the perfect hooks for your content:
- **Opening Hooks**: Grab attention in the first 3 seconds
- **Middle Hooks**: Maintain engagement throughout the content
- **Closing Hooks**: Drive action and comments
- **Strategic Placement**: Tells you exactly where to place each hook

### 📊 Content Script Generation
Creates complete scripts for various Instagram formats:
- **Reel Scripts**: 15-90 second engaging video content
- **Story Scripts**: Sequential story content with clear CTAs
- **Post Scripts**: Detailed content plans for static posts
- **IGTV Scripts**: Long-form content with chapter breakdowns

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- 📧 Email: contact@anasraheemdev.com
- 💼 LinkedIn: [Anas Raheem](https://linkedin.com/in/anasraheem)
- 🐛 Issues: [GitHub Issues](https://github.com/anasraheemdev/repic/issues)
- ⭐ Repository: [Repic on GitHub](https://github.com/anasraheemdev/repic)

## 🎉 Acknowledgments

- OpenAI for GPT models
- Hugging Face for NLP tools
- The open-source community for various libraries
- Instagram creators who inspired the hook strategies

## 📈 Roadmap

- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Integration with Instagram API
- [ ] Real-time trend analysis
- [ ] Custom model training interface
- [ ] Collaborative content creation
- [ ] A/B testing for content variations

---

⭐ **Star this repository if Repic helped you create viral content!**

**Built with ❤️ by [Anas Raheem](https://github.com/anasraheemdev)**

*Repic - Where AI meets viral content creation* 🚀
