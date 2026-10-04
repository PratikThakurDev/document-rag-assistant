import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

class Generator:
    def __init__(
        self,
        model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct"
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )

        self.model.eval()

    def generate(self, query, results,intent="general"):
        context = "\n\n".join(
            f"[Page {result['page']}]\n{result['text']}"
            for result in results[:3]
        )

        if intent == "definition":
            system_instruction = (
                "Extract the definition directly from the document context. "
                "State only the definition and any conditions or properties "
                "explicitly given with it. "
                "Do not paraphrase mathematical conditions. "
                "Do not add explanations, examples, uses, or interpretations. "
                "If the definition is not present, say: "
                "'I do not have enough information in the provided context.'"
            )

        elif intent == "properties":
            system_instruction = (
                "Extract only the properties of the subject explicitly asked about. "
                "Do not include properties of related or contrasting concepts. "
                "Do not include general formulas unless the context explicitly "
                "states them as a property of the requested subject. "
                "Keep mathematical expressions exactly as stated in the context. "
                "Do not add interpretations."
            )

        else:
            system_instruction = (
                "Answer the question using only information explicitly stated "
                "in the provided document context. "
                "Do not add unsupported facts or interpretations. "
                "If the answer is not supported, say: "
                "'I do not have enough information in the provided context.'"
            )

        messages = [
            {
                "role": "system",
                "content": system_instruction
            },
            {
                "role": "user",
                "content": (
                    f"Document context:\n{context}\n\n"
                    f"Question:\n{query}"
                )
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False
            )

        # Remove the input prompt from causal-model output
        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        pages = []

        for result in results[:3]:
            if result["page"] not in pages:
                pages.append(result["page"])

        return {
            "answer": answer,
            "sources": pages,
 
        }