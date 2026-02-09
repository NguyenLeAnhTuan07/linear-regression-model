def encode_categorical(df):
    # Encode Fuel_Type
    if "Fuel_Type" in df.columns:
        df["Fuel_Type"] = df["Fuel_Type"].map({
            "Petrol": 0,
            "Diesel": 1,
            "CNG": 2
        })

    # Encode Seller_Type
    if "Seller_Type" in df.columns:
        df["Seller_Type"] = df["Seller_Type"].map({
            "Dealer": 0,
            "Individual": 1
        })

    # Encode Transmission
    if "Transmission" in df.columns:
        df["Transmission"] = df["Transmission"].map({
            "Manual": 0,
            "Automatic": 1
        })

    return df
