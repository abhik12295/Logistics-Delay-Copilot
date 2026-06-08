from __future__ import annotations

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset


DATASETS = [
    "Cainiao-AI/LaDe-D",
    "Cainiao-AI/LaDe-P",
]


def inspect_dataset(dataset_name: str) -> None:
    print("\n" + "=" * 100)
    print(f"Dataset: {dataset_name}")

    try:
        configs = get_dataset_config_names(dataset_name)
    except Exception as exc:
        print(f"Could not get configs: {exc}")
        return

    print("\nAvailable configs:")
    for config in configs:
        print(f"- {config}")

    for config in configs:
        print("\n" + "-" * 100)
        print(f"Config: {config}")

        try:
            splits = get_dataset_split_names(dataset_name, config)
            print(f"Splits: {splits}")
        except Exception as exc:
            print(f"Could not get splits for {config}: {exc}")
            continue

        for split in splits:
            print("\n" + "." * 100)
            print(f"Split: {split}")

            try:
                dataset = load_dataset(
                    dataset_name,
                    config,
                    split=split,
                    streaming=True,
                )

                first_row = next(iter(dataset))

                print("First row keys:")
                for key in first_row.keys():
                    print(f"  - {key}")

                print("\nFirst row sample:")
                for key, value in first_row.items():
                    print(f"  {key}: {value}")

            except Exception as exc:
                print(f"Could not inspect {dataset_name}/{config}/{split}: {exc}")


def main() -> None:
    for dataset_name in DATASETS:
        inspect_dataset(dataset_name)


if __name__ == "__main__":
    main()