class Preprocessor:

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.batch = 0

    def process(self, processor_options):
        if self.batch == 0:
            return self.preprocessor.set_up_preprocessor_from_base_data()
        else:
            return self.preprocessor.process()
