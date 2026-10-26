# -*- coding: utf-8 -*-
"""
ابزارهای کمکی برای فراخوانی مدل‌های زبانی (چت) و مدل‌های embedding.

این ماژول از دو backend پشتیبانی می‌کند:
  1) Ollama    -> کاملاً لوکال، رایگان، بدون نیاز به اینترنت یا API key
  2) OpenAI    -> سرویس ابری OpenAI، نیاز به OPENAI_API_KEY دارد

هر دو از طریق کتابخانه‌ی رسمی `openai` صدا زده می‌شوند، چون Ollama یک
endpoint سازگار با OpenAI (‎/v1‎) ارائه می‌دهد. یعنی فقط با عوض کردن
`LLM_BACKEND` می‌توانید بین این دو جابه‌جا شوید، بدون تغییر در بقیه‌ی کد.

نکته‌ی مهم: قبلاً این کارها با یک سرور Flask محلی (flask_app.py) و
کتابخانه‌های سنگین FlagEmbedding/torch انجام می‌شد. چون Weaviate از
نسخه‌های اخیر ماژول‌های بومی text2vec-ollama و text2vec-openai را دارد،
دیگر نیازی به آن سرور واسط نیست.
"""

import os
import json
from typing import Dict, List, Union

from openai import OpenAI

# ---------------------------------------------------------------------------
# تنظیمات backend
# ---------------------------------------------------------------------------
# مقدار را با متغیر محیطی LLM_BACKEND عوض کنید: "ollama" یا "openai"
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()

# مدل‌های پیش‌فرض برای هر backend
DEFAULT_CHAT_MODEL = {
    "ollama": "gemma3:4b",
    "openai": "gpt-4o-mini",
}
DEFAULT_EMBED_MODEL = {
    "ollama": "bge-m3",              # چندزبانه، برای متن فارسی هم خوب کار می‌کند
    "openai": "text-embedding-3-small",
}


def get_client() -> OpenAI:
    """
    یک کلاینت OpenAI-سازگار برمی‌گرداند که بسته به LLM_BACKEND
    یا به Ollama لوکال یا به سرویس ابری OpenAI وصل می‌شود.
    """
    if LLM_BACKEND == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "برای استفاده از OpenAI باید متغیر محیطی OPENAI_API_KEY را ست کنید."
            )
        return OpenAI(api_key=api_key)
    else:
        # Ollama باید از قبل روی سیستم در حال اجرا باشد: `ollama serve`
        return OpenAI(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",  # مقدار واقعی چک نمی‌شود، فقط باید غیرخالی باشد
        )


def generate_embedding(prompt: Union[str, List[str]], model: str = None, **kwargs):
    """
    برای یک یا چند متن، بردار embedding تولید می‌کند.

    Args:
        prompt: یک رشته یا لیستی از رشته‌ها
        model: نام مدل embedding (پیش‌فرض بر اساس backend انتخاب می‌شود)

    Returns:
        اگر ورودی یک رشته باشد: یک لیست عدد (بردار)
        اگر ورودی لیستی از رشته‌ها باشد: لیستی از بردارها
    """
    client = get_client()
    model = model or DEFAULT_EMBED_MODEL[LLM_BACKEND]

    try:
        response = client.embeddings.create(model=model, input=prompt, **kwargs)
    except Exception as e:
        raise Exception(f"خطا در فراخوانی مدل embedding ({LLM_BACKEND}): {e}")

    if isinstance(prompt, str):
        return response.data[0].embedding
    return [d.embedding for d in response.data]


def generate_with_single_input(
    prompt: str,
    role: str = "user",
    top_p: float = None,
    temperature: float = None,
    max_tokens: int = 500,
    model: str = None,
    **kwargs,
) -> Dict:
    """
    یک پیام تکی به مدل چت می‌فرستد و پاسخ را برمی‌گرداند.

    خروجی: دیکشنری {'role': ..., 'content': ...}
    """
    return generate_with_multiple_input(
        messages=[{"role": role, "content": prompt}],
        top_p=top_p,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        **kwargs,
    )


def generate_with_multiple_input(
    messages: List[Dict],
    top_p: float = None,
    temperature: float = None,
    max_tokens: int = 500,
    model: str = None,
    **kwargs,
) -> Dict:
    """
    چند پیام (تاریخچه‌ی مکالمه) را به مدل چت می‌فرستد.

    خروجی: دیکشنری {'role': ..., 'content': ...}
    """
    client = get_client()
    model = model or DEFAULT_CHAT_MODEL[LLM_BACKEND]

    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, **kwargs}
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    try:
        response = client.chat.completions.create(**payload)
    except Exception as e:
        raise Exception(f"خطا در فراخوانی مدل چت ({LLM_BACKEND}): {e}")

    choice = response.choices[-1]
    return {"role": choice.message.role, "content": choice.message.content}


def llm_rerank(
    query: str,
    candidates: List[Dict],
    text_key: str = "attractions",
    model: str = None,
) -> List[Dict]:
    """
    بازرتبه‌بندی کاندیدها با استفاده از یک مدل زبانی (به‌جای cross-encoder).

    این تابع جایگزین ماژول reranker-transformers قبلی Weaviate است که
    نیاز به دانلود مدل و سرور جداگانه داشت. در عوض، از همان مدل چت
    (Ollama یا OpenAI) برای امتیازدهی به هر کاندید استفاده می‌شود.

    Args:
        query: عبارت جستجو برای رتبه‌بندی مجدد
        candidates: لیستی از دیکشنری‌ها (خروجی metadatas از کوئری قبلی)
        text_key: کلیدی از هر کاندید که باید برای مقایسه استفاده شود

    Returns:
        همان لیست candidates، اما مرتب‌شده بر اساس نمره‌ی ربط (نزولی)،
        با یک کلید اضافه‌ی '_rerank_score' در هر آیتم.
    """
    scored = []
    for cand in candidates:
        doc_text = str(cand.get(text_key, ""))
        prompt = (
            "به عبارت جستجو و یک سند داده‌شده، فقط یک عدد بین 0 تا 10 برای "
            "میزان ارتباط سند با جستجو برگردان (0 یعنی کاملاً بی‌ربط، "
            "10 یعنی کاملاً مرتبط). فقط عدد را برگردان، بدون هیچ توضیح اضافه.\n\n"
            f"جستجو: {query}\nسند: {doc_text}\nنمره:"
        )
        try:
            result = generate_with_single_input(prompt, temperature=0, max_tokens=5, model=model)
            score_text = "".join(ch for ch in result["content"] if (ch.isdigit() or ch == "."))
            score = float(score_text) if score_text else 0.0
        except Exception:
            score = 0.0
        cand_copy = dict(cand)
        cand_copy["_rerank_score"] = score
        scored.append(cand_copy)

    return sorted(scored, key=lambda c: c["_rerank_score"], reverse=True)


def print_object_properties(obj: Union[dict, list]) -> None:
    """چاپ خوانا از خصوصیات یک آبجکت یا لیستی از آبجکت‌ها (با کوتاه‌سازی فیلدهای طولانی)."""
    t = ""
    truncated_keys = {"article_content", "main_vector", "chunk", "body"}

    def _format_item(d: dict) -> str:
        s = ""
        for x, y in d.items():
            if x in truncated_keys and isinstance(y, str):
                s += f"{x}: {y[:100]}...(بریده‌شده)\n"
            elif x == "main_vector":
                s += f"{x}: {y[:30]}...(بریده‌شده)\n"
            else:
                s += f"{x}: {y}\n"
        return s

    if isinstance(obj, dict):
        t = _format_item(obj)
    else:
        for item in obj:
            t += _format_item(item) + "\n\n"

    print(t)


def print_properties(item):
    print(json.dumps(item.properties, indent=2, sort_keys=True, default=str, ensure_ascii=False))


# ---------------------------------------------------------------------------
# توابع کمکی مدیریت پورت (بدون تغییر نسبت به نسخه‌ی قبلی)
# ---------------------------------------------------------------------------
def kill_processes_on_ports(
    ports,
    *,
    only_listening: bool = True,
    include_udp: bool = True,
    force: bool = True,
    timeout: float = 5.0,
) -> Dict:
    """
    تمام پردازش‌هایی که روی هر یک از پورت‌های داده‌شده در حال گوش‌دادن هستند را می‌کشد.

    مثال:
        kill_processes_on_ports([5000, 8080, 3000])
    """
    import socket
    import psutil

    target_ports = {int(p) for p in ports}
    results = {
        "pids_targeted": [],
        "terminated": [],
        "killed": [],
        "errors": [],
        "ports_with_no_match": [],
    }

    try:
        conns = psutil.net_connections(kind="inet")
    except Exception as e:
        raise RuntimeError(f"خطا در خواندن اتصالات شبکه: {e}")

    pids, matched_ports = set(), set()
    for c in conns:
        if not c.laddr:
            continue
        port = c.laddr.port
        if port not in target_ports:
            continue
        if c.type == socket.SOCK_STREAM and only_listening and c.status != psutil.CONN_LISTEN:
            continue
        if c.type == socket.SOCK_DGRAM and not include_udp:
            continue
        if c.pid is not None:
            pids.add(c.pid)
            matched_ports.add(port)

    results["pids_targeted"] = sorted(pids)
    results["ports_with_no_match"] = sorted(target_ports - matched_ports)

    procs = []
    for pid in list(pids):
        try:
            p = psutil.Process(pid)
            if not p.is_running():
                continue
            p.terminate()
            procs.append(p)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied as e:
            results["errors"].append({"pid": pid, "error": f"دسترسی غیرمجاز: {e}"})
        except Exception as e:
            results["errors"].append({"pid": pid, "error": str(e)})

    gone, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in gone:
        try:
            results["terminated"].append({"pid": p.pid, "name": p.name()})
        except Exception:
            results["terminated"].append({"pid": p.pid, "name": "?"})

    if alive and force:
        for p in alive:
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except psutil.AccessDenied as e:
                results["errors"].append({"pid": p.pid, "error": f"دسترسی غیرمجاز هنگام kill: {e}"})
            except Exception as e:
                results["errors"].append({"pid": p.pid, "error": str(e)})
        gone2, alive2 = psutil.wait_procs(alive, timeout=timeout)
        for p in gone2:
            try:
                results["killed"].append({"pid": p.pid, "name": p.name()})
            except Exception:
                results["killed"].append({"pid": p.pid, "name": "?"})
        for p in alive2:
            results["errors"].append({"pid": p.pid, "error": "هنوز بعد از kill() زنده است"})

    return results


import subprocess
from contextlib import contextmanager


@contextmanager
def suppress_subprocess_output():
    """
    Context manager برای خاموش کردن خروجی stdout/stderr هر subprocess.Popen
    که داخل این بلوک اجرا می‌شود (مثلاً برای بی‌صدا کردن لاگ‌های Weaviate Embedded).
    """
    original_popen = subprocess.Popen

    def patched_popen(*args, **kwargs):
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
        return original_popen(*args, **kwargs)

    try:
        subprocess.Popen = patched_popen
        yield
    finally:
        subprocess.Popen = original_popen
