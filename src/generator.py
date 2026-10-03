import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Generator:
    def __init__(self, model_name="google/flan-t5-base"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

    def generate(self, query, results):
        context = "\n\n".join(
            f"[Page {result['page']}]\n{result['text']}"
            for result in results[:3]
        )

        prompt = f"""
    Answer the question using only the document context below.

    If the answer is not supported by the context, say:
    "I do not have enough information in the provided context."

    Give a direct and concise answer.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )
        answer = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        pages = []
        for result in results[:3]:
            if result["page"] not in pages:
                pages.append(result["page"])

        return {
            "answer": answer,
            "sources": pages
        }