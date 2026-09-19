from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from jev_rag_benchmark.metrics import normalize_answer, paired_bootstrap_ci, percentile


LABELS = {
    "A": "Normal RAG + Qwen3.7 Flash",
    "D": "Jev + Qwen3.7 Flash",
    "G": "Jev + Gemini 3.8 Flash (medium)",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def rate(rows: list[dict], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return mean(values) if values else 0.0


def success(row: dict) -> bool:
    return float(row["answer_f1"]) >= 0.5


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


def pct(values: list[float]) -> str:
    return (
        f"{percentile(values, 0.50):.1f} / {percentile(values, 0.95):.1f} / "
        f"{percentile(values, 0.99):.1f}"
    )


def clipped(text: str, limit: int = 320) -> str:
    value = " ".join(text.split()).replace("|", "\\|")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    branches = {key: [row for row in rows if row["branch"] == key] for key in LABELS}
    paired: dict[str, dict[str, dict]] = {}
    for row in rows:
        paired.setdefault(row["query_id"], {})[row["branch"]] = row
    triples = [
        (value["A"], value["D"], value["G"])
        for value in paired.values()
        if set(value) >= set(LABELS)
    ]
    if not triples:
        raise SystemExit("no A/D/G triples found")

    d_f1 = [float(d["answer_f1"]) for _, d, _ in triples]
    g_f1 = [float(g["answer_f1"]) for _, _, g in triples]
    d_success = [float(value >= 0.5) for value in d_f1]
    g_success = [float(value >= 0.5) for value in g_f1]
    f1_delta, f1_low, f1_high = paired_bootstrap_ci(d_f1, g_f1, seed=args.seed)
    success_delta, success_low, success_high = paired_bootstrap_ci(
        d_success, g_success, seed=args.seed
    )

    g_wins = [(g["answer_f1"] - d["answer_f1"], d, g) for _, d, g in triples if g["answer_f1"] > d["answer_f1"]]
    d_wins = [(d["answer_f1"] - g["answer_f1"], d, g) for _, d, g in triples if d["answer_f1"] > g["answer_f1"]]
    ties = len(triples) - len(g_wins) - len(d_wins)
    fail_to_success = sum(not success(d) and success(g) for _, d, g in triples)
    success_to_fail = sum(success(d) and not success(g) for _, d, g in triples)
    wrong_fixed = sum(d["wrong_answer"] and not g["wrong_answer"] for _, d, g in triples)
    wrong_added = sum(not d["wrong_answer"] and g["wrong_answer"] for _, d, g in triples)
    abstain_to_answer = sum(d["abstained"] and not g["abstained"] for _, d, g in triples)
    answer_to_abstain = sum(not d["abstained"] and g["abstained"] for _, d, g in triples)

    lines = [
        "# XQuAD-TR üçlü üretici karşılaştırması",
        "",
        f"- Eşleştirilmiş benzersiz soru: **{len(triples)}**",
        f"- Toplam sonuç satırı: **{len(rows)}**",
        "- G kolu, D kolunun dondurulmuş Jev bağlamlarını aynen kullanır; D→G farkı yalnızca üretici modeldir.",
        "- Başarı tanımı: atıflar çıkarıldıktan sonra token F1 ≥ 0.5.",
        "",
        "## Ana sonuçlar",
        "",
        "| Metrik | Normal + Qwen | Jev + Qwen | Jev + Gemini 3.8 medium |",
        "|---|---:|---:|---:|",
    ]
    metrics = (
        ("nDCG@10", "ndcg_10_context"),
        ("Recall@5", "recall_context_k"),
        ("Kaynakta altın cevap", "source_contains_gold"),
        ("EM", "answer_em"),
        ("F1", "answer_f1"),
        ("Geçerli atıf", "citation_validity"),
        ("Yanıtsız bırakma", "abstained"),
        ("Yanlış cevap", "wrong_answer"),
    )
    for label, field in metrics:
        values = [rate(branches[key], field) for key in LABELS]
        lines.append(f"| {label} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")
    lines.append(
        "| Başarı oranı | "
        + " | ".join(f"{mean(success(row) for row in branches[key]):.4f}" for key in LABELS)
        + " |"
    )
    lines.append(
        "| Altın cevap aynen içeriliyor | "
        + " | ".join(f"{mean(contains_reference(row) for row in branches[key]):.4f}" for key in LABELS)
        + " |"
    )
    lines.append(
        "| Altın token recall | "
        + " | ".join(f"{mean(gold_token_recall(row) for row in branches[key]):.4f}" for key in LABELS)
        + " |"
    )

    lines.extend(
        [
            "",
            "## D→G: yalnızca üretici model farkı",
            "",
            f"- Ortalama F1 farkı: **{f1_delta:+.4f}** (%95 GA {f1_low:+.4f}, {f1_high:+.4f}).",
            f"- Başarı oranı farkı: **{success_delta:+.4f}** (%95 GA {success_low:+.4f}, {success_high:+.4f}).",
            f"- Gemini daha iyi / Qwen daha iyi / eşit F1: **{len(g_wins)} / {len(d_wins)} / {ties}**.",
            f"- Başarısız→başarılı / başarılı→başarısız: **{fail_to_success} / {success_to_fail}**.",
            f"- Yanlış cevabı düzeltti / yeni yanlış cevap üretti: **{wrong_fixed} / {wrong_added}**.",
            f"- Yanıtsızdan cevaba / cevaptan yanıtsıza: **{abstain_to_answer} / {answer_to_abstain}**.",
            "",
            "## Gecikme (p50 / p95 / p99 ms)",
            "",
            "| Aşama | Normal + Qwen | Jev + Qwen | Jev + Gemini |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, field in (("Generation", "generation_ms"), ("Uçtan uca", "end_to_end_ms")):
        values = [pct([float(row[field]) for row in branches[key]]) for key in LABELS]
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")

    costs = {key: sum(float(row["online_cost_usd"]) for row in branches[key]) for key in LABELS}
    successes = {key: sum(success(row) for row in branches[key]) for key in LABELS}
    incremental_g = sum(float(row.get("replay_incremental_cost_usd", 0.0)) for row in branches["G"])
    lines.extend(["", "## Maliyet", ""])
    for key in LABELS:
        lines.append(
            f"- {LABELS[key]}: **${costs[key] / len(branches[key]):.6f}/sorgu**, "
            f"**${costs[key] / successes[key]:.6f}/başarı** "
            f"({successes[key]} başarı)."
        )
    lines.append(f"- Nihai G satırlarının Gemini üretim gideri: **${incremental_g:.6f}**.")
    lines.append("- Bu tutar yalnızca nihai tutulan cevapları kapsar; token-sınırı ayarı sırasında atılan ve yeniden çalıştırılan deneme çağrıları dahil değildir.")
    lines.append(
        f"- Jev + Gemini kolunun üretimde varsayımsal toplam gideri: **${costs['G']:.6f}** "
        "(Jev maliyeti önceki koşudan kopyalanmıştır; ikinci kez ödenmemiştir)."
    )

    def examples(title: str, items: list[tuple[float, dict, dict]]) -> None:
        lines.extend(["", f"## {title}", ""])
        for _, qwen, gemini in sorted(items, key=lambda item: item[0], reverse=True)[:8]:
            lines.extend(
                [
                    f"### {clipped(qwen['query'], 180)}",
                    "",
                    f"- Altın: {clipped(' / '.join(qwen['references']), 220)}",
                    f"- Jev + Qwen (F1 {qwen['answer_f1']:.3f}): {clipped(qwen['answer'])}",
                    f"- Jev + Gemini (F1 {gemini['answer_f1']:.3f}): {clipped(gemini['answer'])}",
                    f"- Fark: **{gemini['answer_f1'] - qwen['answer_f1']:+.3f}**",
                    "",
                ]
            )

    examples("Gemini'nin en fazla iyileştirdiği örnekler", g_wins)
    examples("Gemini'nin en fazla kötüleştirdiği örnekler", d_wins)
    lines.extend(
        [
            "",
            "## Yorumlama",
            "",
            "- D ve G aynı sorgu, aynı adaylar, aynı Jev sırası ve aynı beş bağlamı kullanır; üretici karşılaştırması kontrollüdür.",
            "- Gemini medium gizli reasoning tokenları kullanır; görünen cevap kısa olsa da completion maliyeti bunları içerir.",
            "- XQuAD kısa altın cevapları nedeniyle EM kaynaklı tam cümleleri sert cezalandırır; F1, başarı ve yanlış cevap oranı birlikte okunmalıdır.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
