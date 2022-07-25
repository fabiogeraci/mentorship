import os
from icevision.all import *

negative_sample_record = ObjectDetectionRecord()
Parser.generate_template(negative_sample_record)


class NegativeSamplesParser(Parser):
    def __init__(self, template_record, df, data_dir):
        super().__init__(template_record=template_record)

        self.data_dir = data_dir
        self.df = df

    def __iter__(self) -> Any:
        for o in self.df.itertuples():
            yield o

    def __len__(self) -> int:
        return len(self.df)

    def record_id(self, o) -> Hashable:
        return o.filename

    def parse_fields(self, o, record, is_new):
        if is_new:
            record.set_filepath(os.path.join(self.data_dir, o.filename))
            record.set_img_size(get_img_size(os.path.join(self.data_dir, o.filename)))
