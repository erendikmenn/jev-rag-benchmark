from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from jev_rag_benchmark.io import read_jsonl
from jev_rag_benchmark.metrics import paired_bootstrap_ci, percentile


def rate(rows: list[dict], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return mean(values) if values else 0.0


def success(row: dict) -> bool:
    return float(row.get("answer_f1") or 0.0) >= 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gemini-branch", default="G")
    parser.add_argument("--deepseek-branch", default="K")
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    rows = read_jsonl(args.results)
    gemini = {row["query_id"]: row for row in rows if row["branch"] == args.gemini_branch}
    deepseek = {row["query_id"]: row for row in rows if row["branch"] == args.deepseek_branch}
    shared = sorted(set(gemini) & set(deepseek))
    if not shared:
        raise SystemExit("no paired Gemini/DeepSeek rows found")
    g_rows = [gemini[key] for key in shared]
    d_rows = [deepseek[key] for key in shared]

    g_f1 = [float(row["answer_f1"]) for row in g_rows]
    d_f1 = [float(row["answer_f1"]) for row in d_rows]
    f1_delta, f1_low, f1_high = paired_bootstrap_ci(g_f1, d_f1, seed=args.seed)
    g_success = [float(success(row)) for row in g_rows]
    d_success = [float(success(row)) for row in d_rows]
    success_delta, success_low, success_high = paired_bootstrap_ci(
        g_success, d_success, seed=args.seed
    )
    deepseek_wins = sum(d > g for g, d in zip(g_f1, d_f1))
    gemini_wins = sum(g > d for g, d in zip(g_f1, d_f1))
    ties = len(shared) - deepseek_wins - gemini_wins

    labels = ("Gemini 3.8 Flash", "DeepSeek V4.1 Flash")
    branches = (g_rows, d_rows)
    lines = [
        "# Gemini 3.8 Flash vs DeepSeek V4.1 Flash",
        "",
        f"- Eşleştirilmiş soru: **{len(shared)}**",
        "- İki model de aynı dondurulmuş sorguları, aynı Jev sırasını ve aynı bağlamları kullandı.",
        "- Başarı tanımı: atıflar çıkarıldıktan sonra token F1 ≥ 0.5.",
        "",
        "| Metrik | Gemini 3.8 Flash | DeepSeek V4.1 Flash |",
        "|---|---:|---:|",
    ]
    for label, field in (
        ("EM", "answer_em"),
        ("F1", "answer_f1"),
        ("Başarı", None),
        ("Geçerli atıf", "citation_validity"),
        ("Yanıtsız bırakma", "abstained"),
        ("Yanlış cevap", "wrong_answer"),
    ):
        values = [
            mean(success(row) for row in branch) if field is None else rate(branch, field)
            for branch in branches
        ]
        lines.append(f"| {label} | {values[0]:.4f} | {values[1]:.4f} |")

    lines.extend(
        [
            "",
            "## Eşleştirilmiş farklar (DeepSeek − Gemini)",
            "",
            f"- Ortalama F1 farkı: **{f1_delta:+.4f}** (%95 GA {f1_low:+.4f}, {f1_high:+.4f}).",
            f"- Başarı oranı farkı: **{success_delta:+.4f}** (%95 GA {success_low:+.4f}, {success_high:+.4f}).",
            f"- DeepSeek daha iyi / Gemini daha iyi / eşit: **{deepseek_wins} / {gemini_wins} / {ties}**.",
            "",
            "## Hız ve maliyet",
            "",
            "| Model | p50 ms | p95 ms | USD/sorgu | Toplam USD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, branch in zip(labels, branches):
        latencies = [float(row["generation_ms"]) for row in branch]
        costs = [
            float(row.get("replay_incremental_cost_usd", row["generator_cost_usd"]))
            for row in branch
        ]
        lines.append(
            f"| {label} | {percentile(latencies, 0.50):.1f} | "
            f"{percentile(latencies, 0.95):.1f} | ${mean(costs):.6f} | ${sum(costs):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Çözümlenen model kimlikleri",
            "",
            f"- Gemini: `{', '.join(sorted({row['generator_model'] for row in g_rows}))}`",
            f"- DeepSeek: `{', '.join(sorted({row['generator_model'] for row in d_rows}))}`",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
