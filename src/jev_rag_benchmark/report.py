from __future__ import annotations

import html
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .io import read_jsonl
from .metrics import paired_bootstrap_ci, percentile


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def generate_report(results_path: Path, output_dir: Path, seed: int) -> Path:
    rows = read_jsonl(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["language"], row["branch"])].append(row)
    lines = [
        "# Jev RAG Benchmark Raporu",
        "",
        "> Bu rapor sonuç dosyasından otomatik üretilmiştir. `run_kind=fixture` satırları gerçek Jev veya gerçek üretici LLM sonucu değildir.",
        "",
        "## Özet",
        "",
        "| Dil | Kol | n | nDCG@10 bağlam | Recall@k | EM | F1 | p50 ms | p95 ms | USD/sorgu | Fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summary = []
    for (language, branch), items in sorted(grouped.items()):
        valid_em = [item["answer_em"] for item in items if item["answer_em"] is not None]
        valid_f1 = [item["answer_f1"] for item in items if item["answer_f1"] is not None]
        latencies = [item["end_to_end_ms"] for item in items]
        record = {
            "language": language,
            "branch": branch,
            "n": len(items),
            "ndcg": mean(item["ndcg_10_context"] for item in items),
            "recall": mean(item["recall_context_k"] for item in items),
            "em": mean(valid_em) if valid_em else None,
            "f1": mean(valid_f1) if valid_f1 else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "cost": mean(item["online_cost_usd"] for item in items),
            "fallback": mean(float(item["fallback"]) for item in items),
            "citation": mean(
                item["citation_validity"] for item in items if item["citation_validity"] is not None
            ) if any(item["citation_validity"] is not None for item in items) else None,
            "support": mean(
                float(item["source_contains_gold"])
                for item in items
                if item["source_contains_gold"] is not None
            ) if any(item["source_contains_gold"] is not None for item in items) else None,
            "abstain": mean(float(item["abstained"]) for item in items),
            "wrong": mean(float(item["wrong_answer"]) for item in items),
        }
        summary.append(record)
        lines.append(
            f"| {language} | {branch} | {record['n']} | {_fmt(record['ndcg'])} | {_fmt(record['recall'])} | {_fmt(record['em'])} | {_fmt(record['f1'])} | {_fmt(record['p50'], 1)} | {_fmt(record['p95'], 1)} | {_fmt(record['cost'], 6)} | {_fmt(record['fallback'])} |"
        )

    lines.extend(
        [
            "",
            "## Cevap desteği ve hata davranışı",
            "",
            "| Dil | Kol | Kaynakta altın cevap | Geçerli atıf | Yanıtsız | Yanlış cevap |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for record in summary:
        lines.append(
            f"| {record['language']} | {record['branch']} | {_fmt(record['support'])} | {_fmt(record['citation'])} | {_fmt(record['abstain'])} | {_fmt(record['wrong'])} |"
        )

    lines.extend(
        [
            "",
            "## Gecikme kırılımı",
            "",
            "| Dil | Kol | Retrieval p50/p95 ms | Reranking p50/p95 ms | Generation p50/p95 ms | Uçtan uca p50/p95 ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for (language, branch), items in sorted(grouped.items()):
        def pair(field: str) -> str:
            values = [float(item[field]) for item in items]
            return f"{percentile(values, 0.5):.1f}/{percentile(values, 0.95):.1f}"
        lines.append(
            f"| {language} | {branch} | {pair('retrieval_ms')} | {pair('rerank_ms')} | {pair('generation_ms')} | {pair('end_to_end_ms')} |"
        )

    total_cost = sum(float(row["online_cost_usd"]) for row in rows)
    successful = [row for row in rows if row["answer_f1"] is not None and row["answer_f1"] >= 0.5]
    one_time_index_ms = sum(float(row["index_ms_once"] or 0) for row in rows)
    lines.extend(
        [
            "",
            "## Maliyet",
            "",
            f"- Toplam ölçülen çevrimiçi deney gideri: **${total_cost:.6f}**.",
            f"- Bir defalık indeksleme süresi: **{one_time_index_ms:.1f} ms**; API gideri: **$0**.",
            f"- Başarılı cevap tanımı: `F1 >= 0.5`; başarılı cevap sayısı: **{len(successful)}**.",
            f"- Başarılı cevap başına ölçülen çevrimiçi gider: **{'—' if not successful else f'${total_cost / len(successful):.6f}'}**.",
        ]
    )
    for record in summary:
        lines.append(
            f"- `{record['language']}-{record['branch']}`: ${record['cost']:.6f}/sorgu; ${record['cost'] * 1000:.4f}/1000 sorgu."
        )

    lines.extend(["", "## Eşleştirilmiş bootstrap (%95 GA)", ""])
    by_language = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_language[row["language"]][row["branch"]][row["query_id"]] = row
    for language, branches in sorted(by_language.items()):
        if "A" not in branches:
            continue
        for branch in sorted(set(branches) - {"A"}):
            common = sorted(set(branches["A"]) & set(branches[branch]))
            if not common:
                continue
            baseline = [branches["A"][qid]["ndcg_10_context"] for qid in common]
            treatment = [branches[branch][qid]["ndcg_10_context"] for qid in common]
            delta, low, high = paired_bootstrap_ci(baseline, treatment, seed=seed)
            lines.append(
                f"- `{language}` A→{branch} nDCG@10 farkı: **{delta:+.3f}** (GA {low:+.3f}, {high:+.3f}; n={len(common)})."
            )

    lines.extend(
        [
            "",
            "## Yorumlama sınırları",
            "",
            "- SciFact qrels yalnızca retrieval değerlendirmesidir; RAG cevap doğruluğu etiketi değildir.",
            "- XQuAD cevapları EM/F1 için kullanılır; kaynakta altın cevabın bulunması ayrıca raporlanır.",
            "- Yerel cross-encoder için API bedeli sıfırdır; gecikme ve yerel donanım süresi maliyet telemetrisidir.",
            "- Tasarruf yüzdesi raporlanacaksa payda A kolunun USD/sorgu değeridir; negatif değerler korunur.",
            "- Küçük smoke örneği kesin sonuç değildir.",
            "",
        ]
    )
    report_path = output_dir / "benchmark-report.tr.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    width, height = 920, 360
    max_ndcg = max((item["ndcg"] for item in summary), default=1) or 1
    bars = []
    for idx, item in enumerate(summary):
        x = 50 + idx * max(80, 800 // max(1, len(summary)))
        bar_h = 240 * item["ndcg"] / max_ndcg
        y = 290 - bar_h
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="48" height="{bar_h:.1f}" fill="#6c63ff"/>'
            f'<text x="{x + 24}" y="315" text-anchor="middle" font-size="12">{html.escape(item["language"])}-{item["branch"]}</text>'
            f'<text x="{x + 24}" y="{y - 7:.1f}" text-anchor="middle" font-size="12">{item["ndcg"]:.3f}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><text x="32" y="32" font-size="20" font-family="sans-serif">Bağlam nDCG@10</text>
<line x1="35" y1="290" x2="890" y2="290" stroke="#333"/>{''.join(bars)}</svg>'''
    (output_dir / "ndcg-context.svg").write_text(svg, encoding="utf-8")
    latency_bars = []
    max_latency = max((item["p50"] for item in summary), default=1) or 1
    for idx, item in enumerate(summary):
        x = 50 + idx * max(80, 800 // max(1, len(summary)))
        bar_h = 240 * item["p50"] / max_latency
        y = 290 - bar_h
        latency_bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="48" height="{bar_h:.1f}" fill="#00a896"/>'
            f'<text x="{x + 24}" y="315" text-anchor="middle" font-size="12">{html.escape(item["language"])}-{item["branch"]}</text>'
            f'<text x="{x + 24}" y="{y - 7:.1f}" text-anchor="middle" font-size="12">{item["p50"]:.1f}</text>'
        )
    latency_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><text x="32" y="32" font-size="20" font-family="sans-serif">Uçtan uca p50 gecikme (ms)</text>
<line x1="35" y1="290" x2="890" y2="290" stroke="#333"/>{''.join(latency_bars)}</svg>'''
    (output_dir / "latency-p50.svg").write_text(latency_svg, encoding="utf-8")
    return report_path
