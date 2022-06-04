import numpy as np

from climatehack import BaseEvaluator
from model import Model
from model import Model2
from model import Model3
from model import Model4
from model import Model5
from model import Model6


class Evaluator(BaseEvaluator):

    def predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        assert data.shape == (12, 128, 128)
        
        model = Model(data)
        prediction = model.generate()
        data = np.vstack((data, prediction))
        del prediction

        
        model = Model2(data)
        prediction = model.generate()
        data = np.vstack((data, prediction))
        del prediction
        

        model = Model3(data)
        prediction = model.generate()
        avg1 = prediction.mean(1, keepdims=1)
        prediction1 = np.broadcast_to(avg1, prediction.shape)
        avg2 = prediction.mean(-1, keepdims=1)
        prediction2 = np.broadcast_to(avg2, prediction.shape)
        prediction = np.mean( np.array([prediction1,prediction2]), axis=0)
        data = np.vstack((data, prediction))
        del prediction
        del prediction1
        del prediction2


        model = Model4(data)
        prediction = model.generate()
        avg1 = prediction.mean(1, keepdims=1)
        prediction1 = np.broadcast_to(avg1, prediction.shape)
        avg2 = prediction.mean(-1, keepdims=1)
        prediction2 = np.broadcast_to(avg2, prediction.shape)
        prediction = np.mean( np.array([prediction1,prediction2]), axis=0)
        data = np.vstack((data, prediction))
        del prediction
        del prediction1
        del prediction2
        

        model = Model5(data)
        prediction = model.generate()
        avg1 = prediction.mean(1, keepdims=1)
        prediction1 = np.broadcast_to(avg1, prediction.shape)
        avg2 = prediction.mean(-1, keepdims=1)
        prediction2 = np.broadcast_to(avg2, prediction.shape)
        prediction = np.mean( np.array([prediction1,prediction2]), axis=0)
        data = np.vstack((data, prediction))
        del prediction
        del prediction1
        del prediction2

        model = Model6(data)
        prediction = model.generate()
        avg1 = prediction.mean(1, keepdims=1)
        prediction1 = np.broadcast_to(avg1, prediction.shape)
        avg2 = prediction.mean(-1, keepdims=1)
        prediction2 = np.broadcast_to(avg2, prediction.shape)
        prediction = np.mean(np.array([prediction1,prediction2]), axis=0)
        data = np.vstack((data, prediction))
        del prediction
        del prediction1
        del prediction2

        output = data[12:, 32:96, 32:96]

        assert output.shape == (24, 64, 64)

        return output
    
    


def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()