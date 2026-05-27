import pandas as pd
import yaml
import great_expectations as ge

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load data as GE DataFrame
df = ge.read_csv(config["data"]["raw_data_path"])

print("=== Running Data Validation ===")

# Expectation 1 — Columns must exist
df.expect_table_columns_to_match_ordered_list(
    column_list=["clean_text", "is_depression"]
)

# Expectation 2 — No null values in clean_text
df.expect_column_values_to_not_be_null("clean_text")

# Expectation 3 — No null values in is_depression
df.expect_column_values_to_not_be_null("is_depression")

# Expectation 4 — is_depression must be 0 or 1 only
df.expect_column_values_to_be_in_set("is_depression", [0, 1])

# Expectation 5 — Text length must be greater than 0
df.expect_column_value_lengths_to_be_between("clean_text", min_value=1)

# Expectation 6 — Row count must be between 7000 and 8000
df.expect_table_row_count_to_be_between(min_value=7000, max_value=8000)

# Save validation report
results = df.validate()
print("\n=== Validation Results ===")
print(f"Total Expectations : {results['statistics']['evaluated_expectations']}")
print(f"Passed             : {results['statistics']['successful_expectations']}")
print(f"Failed             : {results['statistics']['unsuccessful_expectations']}")

if results["success"]:
    print("\n✅ All validations passed!")
else:
    print("\n❌ Some validations failed — check above results!")