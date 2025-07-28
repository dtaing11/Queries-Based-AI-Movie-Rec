# 🎬 Queries-Based AI Movie Recommender

An AI system that recommends movies based on natural language descriptions of user preferences — especially climax and ending styles. Built from scratch with PyTorch, BERT, and cosine similarity matching.

---

## 🚀 Features

- 🔍 Understands user prompts like:  
  _“I like thrillers with twist endings where the main character turns out to be the villain.”_

- 🎯 Returns top 10 recommended films with match percentages
- 🧠 Uses fine-tuned BERT to encode both user preferences and movie metadata
- ⚙️ Triplet-loss–based training from custom dataset
- 📝 Optional: generates sample "ideal climax/ending" descriptions

---


## 📊 Data Requirements

You’ll need:

1. **Movie Metadata**  
   - Title, Genre, Plot Summary, Tags (e.g., "tragic ending", "twist")

2. **User Prompt Dataset**  
   Triplet format for training:
   ```json
   {
     "user_input": "I like sci-fi movies with philosophical endings",
     "positive_movie": "Inception",
     "negative_movie": "The Notebook"
   }
   ```

3. Movie-to-description lookup in JSON:
   ```json
   {
     "Inception": "A skilled thief is offered a chance to have his past crimes erased...",
     "The Notebook": "A romantic drama spanning decades..."
   }
   ```

---

## 🧠 Training

```bash
python train/train_matcher.py
```

Uses Triplet Margin Loss to teach the model that:
> `"I like twists"` → `Inception` should be closer than `The Notebook`

---

## 🧪 Inference

```bash
python inference/recommend.py --prompt "I love emotional sci-fi movies with tragic endings"
```

Example output:

```
Top 10 matches:
1. Interstellar — 100.00%
2. Arrival — 96.15%
3. Inception — 91.73%
...
```

---

## 🧰 Setup

```bash
git clone https://github.com/dtaing11/Queries-Based-AI-Movie-Rec.git
cd Queries-Based-AI-Movie-Rec
pip install -r requirements.txt
```

---

## 🔮 Future Features

- [ ] GPT-based generation of preferred endings
- [ ] Web interface for prompt input + movie cards
- [ ] Clustering similar users for cold-start fallback

---

## 📚 Credits

- Movie data via [TMDB](https://www.themoviedb.org/)
- Language models via [HuggingFace Transformers](https://huggingface.co/transformers/)
- Inspiration: personalized narrative-based recommendation systems

---

## 📜 License

This project is for educational and non-commercial use only.  
Attribution required for TMDB data usage.
