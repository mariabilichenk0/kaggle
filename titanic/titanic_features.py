import pandas as pd

def add_features(
    df,
    fare_bins:      list = None,
    age_fill:       float = None,
    fare_fill:      float = None,
    embarked_fill:  str   = None
):
    # work on a copy so we don’t clobber the original
    df = df.copy()

    # 0) — Embarked: fill‐and‐group
    if embarked_fill is None:
        embarked_fill = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_fill)
    df['Embarked'] = df['Embarked'].where(
        df['Embarked'].isin(['C','Q','S']),
        embarked_fill
    )

    # 1) — Fare: fill missing, then either qcut or reuse bins
    if fare_fill is None:
        fare_fill = df['Fare'].median()
    df['Fare'] = df['Fare'].fillna(fare_fill)

    if fare_bins is None:
        df['FareBin'], fare_bins = pd.qcut(
            df['Fare'],
            4,
            labels=False,
            retbins=True,
            duplicates='drop'
        )
    else:
        df['FareBin'] = pd.cut(
            df['Fare'],
            bins=fare_bins,
            labels=False,
            include_lowest=True
        )
    # any stragglers get shoved into the lowest bin
    df['FareBin'] = df['FareBin'].fillna(0).astype(int)

    # 2) — Title
    pattern = r',\s*([^\.]+)\.'
    df['Title'] = df['Name'].str.extract(pattern, expand=False)
    common = ['Mr','Miss','Mrs','Master']
    df['Title'] = df['Title'].where(df['Title'].isin(common), 'Other')

    # 3) — Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']    = (df['FamilySize'] == 1).astype(int)

    # 4) — Deck: fill missing as 'M', map any truly new letter back to 'M'
    df['Deck'] = df['Cabin'].fillna('M').str[0]
    allowed_decks = list('ABCDEFGM')      # decks seen in train
    df['Deck'] = df['Deck'].where(df['Deck'].isin(allowed_decks), 'M')

    # 5) — Age: fill & bin
    if age_fill is None:
        age_fill = df['Age'].median()
    df['Age'] = df['Age'].fillna(age_fill)

    age_edges  = [0, 12, 20, 40, 60, 80]
    age_labels = ['Child','Teen','Adult','MidAge','Senior']
    df['AgeBin'] = pd.cut(
        df['Age'],
        bins=age_edges,
        labels=age_labels,
        include_lowest=True
    )

    return df, fare_bins, age_fill, fare_fill, embarked_fill
