from flask import Flask, render_template, request, jsonify
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomClass
from src.exception import CustomException
import sys, os  


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

@app.route('/', methods=['GET', 'POST'])

def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('home.html')
        else:
            data = CustomClass(
                age=int(request.form.get('age')),
                workclass=int(request.form.get('workclass')),
                education_num=int(request.form.get('education_num')),
                marital_status=int(request.form.get('marital_status')),
                occupation=int(request.form.get('occupation')),
                relationship=int(request.form.get('relationship')),
                race=int(request.form.get('race')),
                sex=int(request.form.get('sex')),
                capital_gain=int(request.form.get('capital_gain')),
                capital_loss=int(request.form.get('capital_loss')),
                hours_per_week=int(request.form.get('hours_per_week')),
                native_country=int(request.form.get('native_country'))
            )
            
            
        final_new_data = data.custom_data_frame()
        predict_pipeline = PredictionPipeline()
        pred = predict_pipeline.predict(final_new_data)
        
        result = int(pred[0])
        
        if result == 0:
            return render_template('results.html', final_result="The person is likely to earn per year less than or equal to 50K: {}.".format(result))
        elif result == 1:
            return render_template('results.html', final_result="The person is likely to earn per year more than 50K: {}.".format(result))
        else:
            return render_template('results.html', final_result="Prediction is not clear.")
        
    except Exception as e:
        raise CustomException(e, sys)
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
        
    
    
    