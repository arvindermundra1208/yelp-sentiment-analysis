# Yelp Sentiment Analysis Project Website

A modern, responsive website showcasing the Yelp Review Classification project using BiLSTM and CNN-BiLSTM models.

## 🌐 Live Demo

Once deployed, your website will be available at: `https://[your-username].github.io/[repository-name]`

## 📋 Project Overview

This website presents a comprehensive data mining project that develops and evaluates deep learning models for predicting Yelp review star ratings (0–4) using only review text.

### Key Features
- **Two Model Architectures**: BiLSTM and CNN-BiLSTM comparison
- **700K Reviews Dataset**: Comprehensive Yelp Review Full dataset analysis
- **67% Test Accuracy**: Strong performance on sentiment classification
- **Business Applications**: Real-world integration strategies for platforms like DoorDash

## 🚀 Quick Start - GitHub Pages Deployment

### Option 1: Using GitHub Web Interface (Easiest)

1. **Create a new repository on GitHub**
   - Go to [github.com](https://github.com)
   - Click "New repository"
   - Name it something like `yelp-sentiment-analysis` or `data-mining-project`
   - Make it **Public**
   - Do NOT initialize with README
   - Click "Create repository"

2. **Upload your files**
   - Click "uploading an existing file"
   - Drag and drop all files from this directory:
     - `index.html`
     - `styles.css`
     - `images/` folder (with all images)
   - Commit the files

3. **Enable GitHub Pages**
   - Go to repository Settings
   - Scroll to "Pages" section (left sidebar)
   - Under "Source", select `main` branch
   - Click "Save"
   - Wait 1-2 minutes for deployment

4. **Access your website**
   - Your site will be live at: `https://[your-username].github.io/[repository-name]`
   - Link will appear in the Pages settings

### Option 2: Using Git Command Line

```bash
# Navigate to the website directory
cd /Users/arvindermundra/.gemini/antigravity/scratch/yelp-sentiment-analysis-website

# Initialize git repository
git init

# Add all files
git add .

# Commit files
git commit -m "Initial commit: Yelp sentiment analysis project website"

# Add your GitHub repository as remote
# Replace [your-username] and [repository-name] with your actual GitHub details
git remote add origin https://github.com/[your-username]/[repository-name].git

# Push to GitHub
git branch -M main
git push -u origin main

# Enable GitHub Pages
# Go to repository Settings > Pages
# Select 'main' branch as source
# Save and wait for deployment
```

## 📂 Project Structure

```
yelp-sentiment-analysis-website/
├── index.html          # Main website file
├── styles.css          # Premium design system
├── images/            # Visualization images
│   ├── training-validation-loss.jpg
│   ├── confusion-matrix-bilstm.jpg
│   ├── confusion-matrix-cnn-bilstm.jpg
│   ├── training-validation-accuracy.jpg
│   └── training-validation-loss-bilstm.jpg
└── README.md          # This file
```

## 🎨 Design Features

- **Modern Gradient Design**: Vibrant purple, pink, and cyan color palette
- **Glassmorphism Effects**: Frosted glass card backgrounds
- **Smooth Animations**: Fade-in effects and hover transitions
- **Responsive Layout**: Mobile-friendly design
- **Premium Typography**: Inter font family
- **Interactive Elements**: Animated stats and navigation

## 🔧 Customization

### Update GitHub Link
In `index.html`, find the footer section and update:
```html
<a href="https://github.com/YOUR-USERNAME/YOUR-REPO" target="_blank">View Code on GitHub</a>
```

### Modify Colors
Edit `styles.css` root variables:
```css
:root {
  --accent-purple: #667eea;
  --accent-pink: #f5576c;
  --accent-cyan: #00f2fe;
}
```

### Add More Content
Insert new sections between existing `<section>` tags in `index.html`

## 📊 Technologies Used

- **HTML5**: Semantic markup
- **CSS3**: Modern styling with animations
- **JavaScript**: Smooth scroll and intersection observers
- **Google Fonts**: Inter font family

## 🌟 Key Sections

1. **Hero**: Team information and project title
2. **Overview**: Project goals and dataset composition
3. **EDA**: Exploratory data analysis findings
4. **Models**: BiLSTM and CNN-BiLSTM architectures
5. **Results**: Performance metrics and confusion matrices
6. **Insights**: Business applications and future work

## 📱 Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge
- Mobile browsers

## 📝 Academic Citation

If you reference this project, please cite:
```
Group 16 (Akanksha Shah, Arvinder Singh Mundra, Kyren Liu, Tasfin Mahmud)
"Yelp Full Review Classification using BiLSTM and CNN-BiLSTM Models"
Data Mining Final Project, 2025
```

## 🤝 Team Members

- **Akanksha Shah** - UIN 136005001
- **Arvinder Singh Mundra** - UIN 335007465
- **Kyren Liu** - UIN 830004917
- **Tasfin Mahmud** - UIN 437004953

## 📧 Contact

For questions about the project, please reach out to any team member.

## 📄 License

This project is created for academic purposes as part of a Data Mining course final project.

---

**Built with ❤️ by Group 16**
