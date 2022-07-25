import os
from icevision.all import *

bbox_record = ObjectDetectionRecord()
Parser.generate_template(bbox_record)


class ProstateParser(Parser):
    def __init__(self, template_record, df, data_dir):
        super().__init__(template_record=template_record)

        self.data_dir = data_dir
        self.df = df
        self.class_map = ClassMap(list(self.df['label'].unique()))

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
            record.set_img_size(ImgSize(width=o.width, height=o.height))
            record.detection.set_class_map(self.class_map)

        record.detection.add_bboxes([BBox.from_xyxy(o.xmin, o.ymin, o.xmax, o.ymax)])
        record.detection.add_labels([o.label])