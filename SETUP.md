# Clash Royale Win Predictor AI - Setup Guide

## 🎮 Project Overview

A comprehensive real-time AI system that predicts and visualizes the probability of winning Clash Royale matches using live player, clan, and battle data.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL database
- Redis (optional, for caching)
- Clash Royale API key

### 1. Get API Key
1. Visit [Clash Royale Developer Portal](https://developer.clashroyale.com/)
2. Create an account and generate an API key
3. Note down your API key

### 2. Database Setup
```bash
# Install PostgreSQL and create database
createdb clash_royale_db

# Or use a cloud provider like Neon, Supabase, or Railway
```

### 3. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Required variables:
# - CLASH_ROYALE_API_KEY=your_api_key_here
# - DATABASE_URL=postgresql://username:password@localhost:5432/clash_royale_db
# - SECRET_KEY=your_secret_key_here
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
# Automated startup (recommended)
python start.py

# Or manual startup
alembic upgrade head
uvicorn app.main:app --reload
```

### 6. Access the Application
- **Frontend**: http://localhost:8000/static/index.html
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🏗️ Architecture

### Backend Stack
- **FastAPI**: High-performance web framework
- **WebSockets**: Real-time communication
- **PostgreSQL**: Primary database
- **Redis**: Caching layer
- **TensorFlow**: ML model framework
- **SQLAlchemy**: Database ORM

### Frontend Stack
- **HTML5**: Structure
- **TailwindCSS**: Styling
- **JavaScript**: Logic
- **Chart.js**: Data visualization
- **Anime.js**: Animations
- **Socket.IO**: Real-time updates

### ML Pipeline
- **Transformer-LSTM Hybrid**: Win prediction model
- **Feature Engineering**: Battle data processing
- **Real-time Inference**: Live predictions
- **Self-learning**: Model improvement

## 📊 Features

### Core Features
- ✅ Real-time win probability prediction
- ✅ Live battle monitoring
- ✅ Player statistics analysis
- ✅ Deck synergy evaluation
- ✅ AI-powered recommendations
- ✅ Interactive visualizations
- ✅ WebSocket real-time updates

### Advanced Features
- ✅ Transformer-LSTM ML model
- ✅ Feature engineering pipeline
- ✅ Clash Royale API integration
- ✅ Animated UI components
- ✅ Performance analytics
- ✅ Battle history tracking

## 🔧 API Endpoints

### Players
- `GET /api/v1/players/{player_tag}` - Get player info
- `GET /api/v1/players/{player_tag}/stats` - Get player statistics
- `GET /api/v1/players/{player_tag}/battles` - Get battle history
- `POST /api/v1/players/{player_tag}/refresh` - Refresh player data

### Predictions
- `POST /api/v1/predictions/{player_tag}/predict` - Make prediction
- `GET /api/v1/predictions/{player_tag}/predictions` - Get prediction history
- `GET /api/v1/predictions/{player_tag}/live-prediction` - Get live prediction

### Battles
- `POST /api/v1/battles/analyze` - Analyze battle
- `GET /api/v1/battles/recent` - Get recent battles
- `GET /api/v1/battles/stats` - Get battle statistics

### Clans
- `GET /api/v1/clans/{clan_tag}` - Get clan info
- `GET /api/v1/clans/{clan_tag}/members` - Get clan members
- `GET /api/v1/clans/{clan_tag}/performance` - Get clan performance

### WebSocket Endpoints
- `ws://localhost:8000/ws/predictions/{player_tag}` - Live predictions
- `ws://localhost:8000/ws/battles/{player_tag}` - Live battle updates

## 🎯 Usage

### 1. Search for a Player
1. Enter a player tag (e.g., #2PP)
2. Click "Search Player"
3. View player information and statistics

### 2. Start Live Prediction
1. After searching a player, click "Start Live Prediction"
2. View real-time win probability updates
3. See AI recommendations and influencing factors

### 3. Monitor Battle Progress
1. Watch live battle status updates
2. View tower health and elixir status
3. Track battle timeline and events

## 🚀 Deployment

### Frontend (Vercel)
```bash
# Deploy frontend to Vercel
vercel --prod
```

### Backend (Railway)
```bash
# Deploy backend to Railway
railway login
railway init
railway up
```

### Environment Variables for Production
```env
# Production environment variables
CLASH_ROYALE_API_KEY=your_production_api_key
DATABASE_URL=your_production_database_url
REDIS_URL=your_production_redis_url
SECRET_KEY=your_production_secret_key
DEBUG=False
```

## 🔒 Security Features

### API Security
- Rate limiting (100 requests/minute)
- API key authentication
- CORS configuration
- Input validation

### Data Security
- Environment variable configuration
- Secure database connections
- Error handling and logging

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app
```

### Manual Testing
1. Test player search functionality
2. Verify live prediction updates
3. Check WebSocket connections
4. Validate API responses

## 🐛 Troubleshooting

### Common Issues

**Backend won't start**
- Check Python version (3.9+ required)
- Verify database connection
- Ensure all dependencies are installed

**API key errors**
- Verify Clash Royale API key is valid
- Check API key permissions
- Ensure proper environment variable setup

**Database connection issues**
- Verify PostgreSQL is running
- Check DATABASE_URL format
- Run database migrations

**WebSocket connection fails**
- Check firewall settings
- Verify backend is running
- Test with browser developer tools

## 📈 Performance

### Optimization Features
- Redis caching for API responses
- Database connection pooling
- Efficient ML model inference
- Optimized frontend assets

### Monitoring
- Health check endpoint
- Application logging
- Performance metrics
- Error tracking

## 🔮 Future Enhancements

### Planned Features
- User authentication system
- Tournament prediction mode
- Clan war analytics
- Mobile app companion
- Voice assistant integration
- Discord bot integration

### ML Improvements
- Enhanced feature engineering
- Model ensemble techniques
- Real-time model updates
- Cross-validation improvements

## 📚 Documentation

- **API Docs**: http://localhost:8000/docs
- **Code Documentation**: Generated with docstrings
- **Architecture Diagrams**: In `/docs` folder
- **Database Schema**: In `/migrations` folder

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
1. Check this setup guide
2. Review the troubleshooting section
3. Check the GitHub issues
4. Create a new issue with detailed information

---

**Happy Clashing! 🏆**
