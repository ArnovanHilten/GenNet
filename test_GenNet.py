import GenNet
import warnings
import warnings
warnings.filterwarnings('ignore')



GenNet.main(datapath='/home/avanhilten/PycharmProjects/GenNet/processed_data/example_study/',
            jobid=1,
            wpc=1,
            lr_opt=0.01,
            batch_size=32,
            epochs=2,
            l1_value=0.01,
            mode="classification")

