
class configs:
    def __init__(self):
        self.dataset_root = '/path/to/fruit/dataset'
        self.SCOUNT_model_path = '/path/to/SCOUNT/model'
        self.PAC_model_path = '/path/to/PAC/model'
        self.WSCOUNT_model_path = '/path/to/WSCOUNT/model'

        # subsampled_dim1 and subsampled_dim2 are width_img/32 and height_img/32 approximate by excess
        self.subsampled_dim1 = 10
        self.subsampled_dim2 = 10

        # subsampled_dim1_t4 and subsampled_dim2_t4 are subsampled_dim1/2 and subsampled_dim2/2 approximate by excess
        self.subsampled_dim1_t4 = 5
        self.subsampled_dim2_t4 = 5

        # subsampled_dim1_t16 and subsampled_dim2_t16 are subsampled_dim1_t4/2 and subsampled_dim2_t4/2 approximate by excess
        self.subsampled_dim1_t16 = 3
        self.subsampled_dim2_t16 = 3
