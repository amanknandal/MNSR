import numpy as np
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ReflectionMemory:
    """
    Episodic Reflection Memory
    Stores previous reasoning failures and uses vector-based similarity
    to retrieve similar experiences for future reasoning.
    """

    def __init__(self):
        self.memory: List[Dict] = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None

    def add(
        self,
        question: str,
        reasoning: str,
        answer: str,
        errors: List[Dict],
        corrected_reasoning: str = "",
        success: bool = False
    ):
        episode = {
            "question": question,
            "reasoning": reasoning,
            "answer": answer,
            "errors": errors,
            "corrected_reasoning": corrected_reasoning,
            "success": success
        }
        self.memory.append(episode)
        
        # Incrementally rebuild the vector index
        questions = [ep["question"].lower() for ep in self.memory]
        self.tfidf_matrix = self.vectorizer.fit_transform(questions)

    def retrieve(
        self,
        question: str,
        threshold: float = 0.60
    ) -> Optional[Dict]:
        
        if not self.memory or self.tfidf_matrix is None:
            return None

        # Convert the new question to the same vector space
        query_vec = self.vectorizer.transform([question.lower()])
        
        # Calculate fast dot-product similarity matrix
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score < threshold:
            return None

        return {
            "similarity": round(best_score, 3),
            "episode": self.memory[best_idx]
        }

    def size(self) -> int:
        return len(self.memory)

    def clear(self):
        self.memory = []
        self.tfidf_matrix = None

    def export(self) -> List[Dict]:
        return self.memory