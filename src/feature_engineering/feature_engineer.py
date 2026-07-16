from pathlib import Path
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)
class FeatureEngineer:
    def __init__(
        self,
        input_path = "data/interim/clean_transactions.csv",
        output_dir = "data/processed"
    ):
        self.input_path = Path(input_path)
        self.output_dir =  Path(output_dir)
        
    def load_data(self):
        logger.info("loading cleaned dataset....")
        df = pd.read_csv(
            self.input_path,
            parse_dates=["invoicedate"],
        )

        return df

    def create_customer_features(self, df):

        logger.info("Creating customer features...")

        reference_date = (
            df["invoicedate"].max() +
            pd.Timedelta(days=1)
        )

        customer = (
            df.groupby("customer_id")
            .agg(
                recency=(
                    "invoicedate",
                    lambda x: (
                        reference_date - x.max()
                    ).days
                ),

                frequency=("invoice", "nunique"),

                monetary=("totalprice", "sum"),

                average_order_value=("totalprice", "mean"),

                first_purchase=("invoicedate", "min"),

                last_purchase=("invoicedate", "max"),
            )
            .reset_index()
        )

        customer["customer_lifetime_days"] = (
            customer["last_purchase"] -
            customer["first_purchase"]
        ).dt.days

        customer["purchase_frequency"] = (
            customer["frequency"] /
            customer["customer_lifetime_days"]
            .replace(0, 1)
        )

        basket = (
            df.groupby(
                ["customer_id", "invoice"]
            )["quantity"]
            .sum()
            .groupby("customer_id")
            .mean()
        )

        customer["basket_size"] = (
            customer["customer_id"]
            .map(basket)
        )

        return customer

    def create_product_features(self, df):

        logger.info("Creating product features...")

        product = (
            df.groupby("stockcode")
            .agg(
                product_popularity=("invoice", "count"),

                average_quantity=("quantity", "mean"),

                average_price=("price", "mean"),

                total_sales=("totalprice", "sum"),

                unique_customers=(
                    "customer_id",
                    "nunique",
                ),
            )
            .reset_index()
        )

        return product

    def create_transaction_features(self, df):

        logger.info(
            "Creating transaction features..."
        )

        transaction = df.copy()

        transaction["year"] = (
            transaction["invoicedate"].dt.year
        )

        transaction["month"] = (
            transaction["invoicedate"].dt.month
        )

        transaction["quarter"] = (
            transaction["invoicedate"].dt.quarter
        )

        transaction["weekday"] = (
            transaction["invoicedate"]
            .dt.day_name()
        )

        transaction["hour"] = (
            transaction["invoicedate"].dt.hour
        )

        return transaction

    def save(self,
             customer,
             product,
             transaction):

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        customer.to_csv(
            self.output_dir /
            "customer_features.csv",
            index=False,
        )

        product.to_csv(
            self.output_dir /
            "product_features.csv",
            index=False,
        )

        transaction.to_csv(
            self.output_dir /
            "transaction_features.csv",
            index=False,
        )

        logger.info(
            "Feature datasets saved."
        )

    def run(self):

        logger.info("=" * 60)
        logger.info(
            "STARTING FEATURE ENGINEERING"
        )

        df = self.load_data()

        customer = self.create_customer_features(df)

        product = self.create_product_features(df)

        transaction = (
            self.create_transaction_features(df)
        )

        self.save(
            customer,
            product,
            transaction,
        )

        logger.info(
            "FEATURE ENGINEERING COMPLETED"
        )

        logger.info("=" * 60)

        return (
            customer,
            product,
            transaction,
        )