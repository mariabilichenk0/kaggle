from sklearn.pipeline      import Pipeline
from sklearn.impute        import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose       import ColumnTransformer
from category_encoders     import TargetEncoder

def make_preprocessor(
    num_cols,
    cat_cols,
    cat_encoder: str = "onehot",   # choose from "onehot", "ordinal", "target"
):
    # 1) Numeric pipeline stays the same
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",   StandardScaler())
    ])

    # 2) Categorical pipeline depends on your choice
    if cat_encoder == "onehot":
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encode", OneHotEncoder(
                drop="first", handle_unknown="ignore", sparse_output=False
            )),
        ])

    elif cat_encoder == "ordinal":
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encode", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )),
        ])

    elif cat_encoder == "target":
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encode", TargetEncoder()),            # mean‐encoding
        ])

    else:
        raise ValueError(f"Unknown cat_encoder={cat_encoder!r}")

    # 3) Put them together
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")
