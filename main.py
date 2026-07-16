from src.pipeline.etl_pipeline import ETLPipeline


def main():

    pipeline = ETLPipeline()

    pipeline.run()


if __name__ == "__main__":
    main()