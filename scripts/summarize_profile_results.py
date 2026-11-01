"""Export report and profile/locus graph from verified saved artifacts."""

import json
from pathlib import Path

import numpy as np

from src.data import load_fasta, read_rows, write_rows


def main():
    validation = Path("runs/profile_validation_v3_frozen")
    final = Path("runs/profile_final_seed43_v3")
    audit = json.loads((final / "acceptance_audit.json").read_text())
    assert audit["passed"]
    for root in (validation, final):
        for directory in root.rglob("*"):
            if not directory.is_dir() or not (directory / "loci.jsonl").is_file():
                continue
            loci = read_rows(directory / "loci.jsonl")
            histories = json.loads((directory / "profile_history.json").read_text())
            graph = []
            profiles = directory / "profiles"
            profiles.mkdir(exist_ok=True)
            for experiment in histories:
                if "family_id" not in experiment:
                    continue
                family = experiment["family_id"]
                last = experiment["history"][-1]
                np.savez_compressed(
                    profiles / f"{family}.npz",
                    log_odds=np.asarray(last["log_odds"], dtype=np.float32),
                    threshold=last["threshold"],
                    alphabet=np.array(list("ACGT")),
                )
            for i, locus in enumerate(loci):
                graph.append(
                    {
                        "profile": locus["family_id"],
                        "locus": i,
                        "start": locus["start"],
                        "end": locus["end"],
                        "reverse": locus["reverse"],
                        "score": locus["score"],
                        "source": "profile_scan",
                    }
                )
            write_rows(directory / "profile_graph.jsonl", graph)
    rows = [
        r for r in json.loads((final / "metrics.json").read_text()) if r["iou"] == 0.5
    ]
    report = [
        "# Результат: классический профильный поиск до x=1.5",
        "",
        "Финальный тест seed=43: 500 000 bp, 150 копий по 300 bp. Параметры и исходники заморожены до теста; поиск выполнен один раз для каждого уровня. Истинные координаты и предок используются только генератором и оцениванием.",
        "",
        "| x | событий на копию | TP | FP | locus precision | locus recall | best-family F1 | pair F1 | средний IoU | время, с |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        report.append(
            f"| {r['x']:g} | {round(r['x'] / 2 * 300)} | {r['locus_tp']} | {r['locus_fp']} | {r['locus_precision']:.4f} | {r['locus_recall']:.4f} | {r['best_family_f1']:.4f} | {r['pair_f1']:.4f} | {r['mean_matched_iou']:.4f} | {r['seconds']:.1f} |"
        )
    report.extend(
        [
            "",
            "Все 150 настоящих копий группируются в одно правильное семейство при x≤1.5. При x=0.5 и x=1.5 также найдено отдельное ложное семейство из шести локусов. При x=2 находок нет. Время всего финального sweep: "
            + f"{sum(r['seconds'] for r in rows):.1f} с на CPU.",
            "",
            "На validation seed=44 найдены все 120 локусов четырёх семейств при x=0 и x=0.5; precision/recall/best-family F1=1, число слияний=0. На отдельном отрицательном контроле seed=49 ложных локусов нет. Фон seed=45 используется для калибровки.",
            "",
            "Независимый аудит повторно вычислил метрики при IoU≥0.2 и ≥0.5, проверил финальные FASTA и truth по исходному генератору, неизменность исходников и параметры. Для каждого локуса score пересчитан напрямую по сохранённой матрице log-odds; score и порог согласуются. Поиск аудит не повторяет.",
            "",
            "Основной результат использует тройки копий, профильный scan и итеративное уточнение частот. Он не требует NT, GPU или Parasail. Это результат для замен без инделов и равномерного IID-фона; переносимость на настоящие геномы и инделы пока не установлена.",
            "",
            "## Диагностика вариантов на одинаковом validation-геноме",
            "",
            "| Вариант | x | precision | recall | best-family F1 |",
            "|---|---|---|---|---|",
        ]
    )
    variants = [
        ("SW ±200, scalar reference", "runs/v2_anchor_validation_reference"),
        (
            "SW adaptive до ±800, scalar reference",
            "runs/v2_anchor_validation_adaptive_reference",
        ),
        ("Ungapped ±800", "runs/v2_anchor_validation_ungapped"),
        ("Ungapped ±800 + отбор границ", "runs/v2_anchor_validation_ungapped_nms"),
        ("Тройка + профиль", str(validation)),
    ]
    for label, path in variants:
        root = Path(path)
        for row in json.loads((root / "metrics.json").read_text()):
            if row["iou"] != 0.5 or row.get("method", "anchors") != "anchors":
                continue
            # Compare only identical inputs, never mix seed42 with seed44.
            level = "x_0" if row["x"] == 0 else "x_0p5"
            assert load_fasta(
                root / "validation_seed44" / level / "genome.fa"
            ) == load_fasta(validation / "validation_seed44" / level / "genome.fa")
            report.append(
                f"| {label} | {row['x']:g} | {row['locus_precision']:.4f} | {row['locus_recall']:.4f} | {row['best_family_f1']:.4f} |"
            )
    report.extend(
        [
            "",
            "SW-проверки выполнены точным скалярным C backend с +2/-1, gap-open 5, gap-extend 2; Parasail установить не удалось из-за DNS. Его SIMD реализация здесь не проверялась. Сравнение frozen NT / проекции NT с новым выравниванием не выполнено; улучшение модели этим результатом не заявляется. Старые результаты seed=42 сохранены.",
            "",
            "## Артефакты и воспроизведение",
            "",
            "- [Описание метода и команды](PROFILE_METHOD.md).",
            "- [Финальные метрики](runs/profile_final_seed43_v3/metrics.tsv).",
            "- [Независимый аудит](runs/profile_final_seed43_v3/acceptance_audit.json).",
            "- [Замороженные параметры](runs/profile_validation_v3_frozen/frozen_parameters.json).",
            "- [Validation](runs/profile_validation_v3_frozen/metrics.tsv).",
            "- В каталогах уровней сохранены FASTA, truth отдельно, опоры, локусы, ungapped-выравнивания с CIGAR, матрицы профилей, история scan, граф profile→locus и журнал.",
            "- 42 модульных/интеграционных теста проходят; один тест Parasail пропущен.",
            "",
        ]
    )
    Path("PROFILE_RESULTS.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
