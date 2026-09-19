from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from jev_rag_benchmark.metrics import normalize_answer, paired_bootstrap_ci, percentile


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def rate(items: list[dict], field: str) -> float:
    values = [float(item[field]) for item in items if item.get(field) is not None]
    return mean(values) if values else 0.0


def latency(items: list[dict], field: str) -> str:
    values = [float(item[field]) for item in items]
    return (
        f"{percentile(values, 0.50):.1f} / {percentile(values, 0.95):.1f} / "
        f"{percentile(values, 0.99):.1f}"
    )


def clipped(text: str, limit: int = 360) -> str:
    clean = " ".join(text.split()).replace("|", "\\|")
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def contains_reference(row: dict) -> bool:
    answer = normalize_answer(row["answer"])
    return any(normalize_answer(reference) in answer for reference in row["references"])


def gold_token_recall(row: dict) -> float:
    answer_tokens = Counter(normalize_answer(row["answer"]).split())
    scores = []
    for reference in row["references"]:
        gold_tokens = Counter(normalize_answer(reference).split())
        overlap = sum((answer_tokens & gold_tokens).values())
        scores.append(overlap / sum(gold_tokens.values()) if gold_tokens else 0.0)
    return max(scores, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("documents", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    branches = {branch: [row for row in rows if row["branch"] == branch] for branch in ("A", "D")}
    paired: dict[str, dict[str, dict]] = {}
    for row in rows:
        paired.setdefault(row["query_id"], {})[row["branch"]] = row
    pairs = [(value["A"], value["D"]) for value in paired.values() if set(value) >= {"A", "D"}]

    documents = {row["doc_id"]: row for row in load_jsonl(args.documents)}
    titles = Counter()
    for baseline, _ in pairs:
        for doc_id in baseline["candidate_ids"]:
            if doc_id in documents and documents[doc_id].get("title"):
                titles[documents[doc_id]["title"]] += 1

    a_f1 = [float(a["answer_f1"]) for a, _ in pairs]
    d_f1 = [float(d["answer_f1"]) for _, d in pairs]
    a_success = [float(value >= 0.5) for value in a_f1]
    d_success = [float(value >= 0.5) for value in d_f1]
    f1_delta, f1_low, f1_high = paired_bootstrap_ci(a_f1, d_f1, seed=args.seed)
    success_delta, success_low, success_high = paired_bootstrap_ci(
        a_success, d_success, seed=args.seed
    )

    d_wins = [(d["answer_f1"] - a["answer_f1"], a, d) for a, d in pairs if d["answer_f1"] > a["answer_f1"]]
    a_wins = [(a["answer_f1"] - d["answer_f1"], a, d) for a, d in pairs if a["answer_f1"] > d["answer_f1"]]
    ties = len(pairs) - len(d_wins) - len(a_wins)
    fail_to_success = sum(a["answer_f1"] < 0.5 <= d["answer_f1"] for a, d in pairs)
    success_to_fail = sum(d["answer_f1"] < 0.5 <= a["answer_f1"] for a, d in pairs)
    support_gain = sum(not a["source_contains_gold"] and d["source_contains_gold"] for a, d in pairs)
    support_loss = sum(a["source_contains_gold"] and not d["source_contains_gold"] for a, d in pairs)
    answer_gain = sum(a["abstained"] and not d["abstained"] for a, d in pairs)
    answer_loss = sum(not a["abstained"] and d["abstained"] for a, d in pairs)
    candidate_mismatches = sum(a["candidate_ids"] != d["candidate_ids"] for a, d in pairs)
    a_contains = [float(contains_reference(a)) for a, _ in pairs]
    d_contains = [float(contains_reference(d)) for _, d in pairs]
    a_gold_recall = [gold_token_recall(a) for a, _ in pairs]
    d_gold_recall = [gold_token_recall(d) for _, d in pairs]

    lines = [
        "# XQuAD-TR tam OpenRouter Jev analizi",
        "",
        f"- Eşleştirilmiş benzersiz soru: **{len(pairs)}**",
        f"- Sonuç satırı: **{len(rows)}** (A ve D)",
        f"- Aday liste uyuşmazlığı: **{candidate_mismatches}**",
        f"- Aday belgelerde temsil edilen farklı makale başlığı: **{len(titles)}**",
        "- Başarı tanımı: atıflar çıkarıldıktan sonra token F1 ≥ 0.5.",
        "",
        "## Ana metrikler",
        "",
        "| Metrik | A: normal RAG | D: Jev |",
        "|---|---:|---:|",
    ]
    for label, field in (
        ("nDCG@10", "ndcg_10_context"),
        ("Recall@5", "recall_context_k"),
        ("Kaynakta altın cevap", "source_contains_gold"),
        ("EM", "answer_em"),
        ("F1", "answer_f1"),
        ("Başarı oranı", None),
        ("Geçerli atıf", "citation_validity"),
        ("Yanıtsız bırakma", "abstained"),
        ("Yanlış cevap", "wrong_answer"),
        ("Fallback", "fallback"),
    ):
        if field is None:
            av, dv = mean(a_success), mean(d_success)
        else:
            av, dv = rate(branches["A"], field), rate(branches["D"], field)
        lines.append(f"| {label} | {av:.4f} | {dv:.4f} |")
    lines.append(f"| Altın cevap aynen içeriliyor | {mean(a_contains):.4f} | {mean(d_contains):.4f} |")
    lines.append(f"| Altın token recall | {mean(a_gold_recall):.4f} | {mean(d_gold_recall):.4f} |")

    lines.extend(
        [
            "",
            "## Eşleştirilmiş farklar",
            "",
            f"- Ortalama F1 farkı: **{f1_delta:+.4f}** (%95 GA {f1_low:+.4f}, {f1_high:+.4f}).",
            f"- Başarı oranı farkı: **{success_delta:+.4f}** (%95 GA {success_low:+.4f}, {success_high:+.4f}).",
            f"- F1'e göre Jev daha iyi / normal daha iyi / eşit: **{len(d_wins)} / {len(a_wins)} / {ties}**.",
            f"- Başarısız→başarılı / başarılı→başarısız: **{fail_to_success} / {success_to_fail}**.",
            f"- Altın kaynağı bağlama kazandırdı / kaybetti: **{support_gain} / {support_loss}**.",
            f"- Yanıtsızdan cevaba / cevaptan yanıtsıza: **{answer_gain} / {answer_loss}**.",
            "",
            "## Gecikme (p50 / p95 / p99 ms)",
            "",
            "| Aşama | A | D |",
            "|---|---:|---:|",
        ]
    )
    for label, field in (
        ("Retrieval", "retrieval_ms"),
        ("Reranking", "rerank_ms"),
        ("Generation", "generation_ms"),
        ("Uçtan uca", "end_to_end_ms"),
    ):
        lines.append(f"| {label} | {latency(branches['A'], field)} | {latency(branches['D'], field)} |")

    total_rerank = sum(float(row.get("reranker_cost_usd", 0)) for row in rows)
    total_generation = sum(float(row.get("generator_cost_usd", 0)) for row in rows)
    a_cost = sum(float(row["online_cost_usd"]) for row in branches["A"])
    d_cost = sum(float(row["online_cost_usd"]) for row in branches["D"])
    a_success_count = sum(a_success)
    d_success_count = sum(d_success)
    lines.extend(
        [
            "",
            "## Maliyet ve güvenilirlik",
            "",
            f"- Reranking: **${total_rerank:.6f}**; cevap üretimi: **${total_generation:.6f}**; toplam: **${total_rerank + total_generation:.6f}**.",
            f"- A maliyet/sorgu: **${mean(row['online_cost_usd'] for row in branches['A']):.6f}**.",
            f"- D maliyet/sorgu: **${mean(row['online_cost_usd'] for row in branches['D']):.6f}**.",
            f"- A başarı başına maliyet: **${a_cost / a_success_count:.6f}** ({int(a_success_count)} başarı).",
            f"- D başarı başına maliyet: **${d_cost / d_success_count:.6f}** ({int(d_success_count)} başarı).",
            f"- Reranker hatası / generator hatası: **{sum(bool(row['reranker_error']) for row in rows)} / {sum(bool(row['generation_error']) for row in rows)}**.",
            f"- Retry kullanılan satır / toplam retry: **{sum(row['retry_count'] > 0 for row in rows)} / {sum(row['retry_count'] for row in rows)}**.",
            "",
            "## En sık temsil edilen makaleler (aday havuzu)",
            "",
            "| Makale | Soru-aday görünümü |",
            "|---|---:|",
        ]
    )
    for title, count in titles.most_common(20):
        lines.append(f"| {title.replace('|', '/')} | {count} |")

    def add_examples(title: str, examples: list[tuple[float, dict, dict]]) -> None:
        lines.extend(["", f"## {title}", ""])
        for delta, baseline, jev in sorted(examples, key=lambda item: item[0], reverse=True)[:8]:
            lines.extend(
                [
                    f"### {clipped(baseline['query'], 180)}",
                    "",
                    f"- Altın: {clipped(' / '.join(baseline['references']), 240)}",
                    f"- A (F1 {baseline['answer_f1']:.3f}): {clipped(baseline['answer'])}",
                    f"- D (F1 {jev['answer_f1']:.3f}): {clipped(jev['answer'])}",
                    f"- Fark: **{jev['answer_f1'] - baseline['answer_f1']:+.3f}**",
                    "",
                ]
            )

    add_examples("Jev'in en fazla iyileştirdiği cevap örnekleri", d_wins)
    add_examples("Jev'in en fazla kötüleştirdiği cevap örnekleri", [(v, a, d) for v, a, d in a_wins])

    lines.extend(
        [
            "",
            "## Yorumlama",
            "",
            "- Retrieval kazanımı cevap başarısına bire bir taşınmaz; cevaplayıcı doğru kaynak mevcutken de uzun, eksik veya yanlış odaklı yanıt üretebilir.",
            "- EM tam metin eşleşmesidir ve kaynaklı tam cümleleri sert biçimde cezalandırır; F1 ve kaynak desteği birlikte okunmalıdır.",
            "- Bu çalışma tüm benzersiz XQuAD-TR test sorularını kapsar; sonuç başka alanlara otomatik genellenmemelidir.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
