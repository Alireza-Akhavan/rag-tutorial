"""
Utility functions for the ChromaDB ungraded labs.

Now that both labs use Ollama (via `langchain_ollama.OllamaEmbeddings`) for
embeddings and OpenAI (via `langchain.chat_models.init_chat_model`) for
generation directly, none of the old Together AI / course-proxy helpers
(`generate_embedding`, `generate_with_single_input`, `get_proxy_url`, etc.)
are needed anymore, so they've been removed.

Only one thing is exported:
- print_object_properties(): pretty-prints a Chroma metadata dict (or list of dicts).
"""

from typing import Union


def print_object_properties(obj: Union[dict, list]) -> None:
    t = ""
    if isinstance(obj, dict):
        for x, y in obj.items():
            if x in ("article_content", "chunk"):
                t += f"{x}: {y[:100]}...(truncated)\n"
            elif x == "main_vector":
                t += f"{x}: {y[:30]}...(truncated)\n"
            else:
                t += f"{x}: {y}\n"
    else:
        for item in obj:
            for x, y in item.items():
                if x in ("article_content", "chunk"):
                    t += f"{x}: {y[:100]}...(truncated)\n"
                elif x == "main_vector":
                    t += f"{x}: {y[:30]}...(truncated)\n"
                else:
                    t += f"{x}: {y}\n"
            t += "\n\n"

    print(t)
