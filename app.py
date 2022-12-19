from flask import Flask
from flask import render_template,request
import pandas as pd
import numpy as np
import pickle
print("File opened")

file1 = open('pipe.pkl', 'rb')
rf = pickle.load(file1)
print("File opened")
file1.close()
data = pd.read_csv("traineddata.csv")
app = Flask(__name__)
lis=['Apple','Microsoft']
@app.route('/')
def home():
   return render_template('index.html',company=data['Company'].unique(),
   type=data['TypeName'].unique(),
   ram=[2, 4, 6, 8, 12, 16, 24, 32, 64],os=['Mac','Windows'],
   screenres=['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'],
   cpu=data['CPU_name'].unique(),
   hdd=[0, 128, 256, 512, 1024, 2048],
   sdd=[0, 8, 128, 256, 512, 1024],
   gpu=data['Gpu brand'].unique())

@app.route('/', methods = ['POST'])
def predict():
    if request.method == 'POST':
        inputs=list(request.form.values())
        print(inputs)
        company=inputs[0]
        types=inputs[1]
        ram=int(inputs[2])
        os=inputs[3]
        weight=float(inputs[4])
        touchscreen=inputs[5]
        ips=inputs[6]
        screen_size=float(inputs[7])
        resolution=inputs[8]
        cpu=inputs[9]
        hdd=int(inputs[10])
        ssd=int(inputs[11])
        gpu=inputs[12]
        ppi = None
        if touchscreen == 'Yes':
            touchscreen = 1
        else:
            touchscreen = 0

        if ips == 'Yes':
            ips = 1
        else:
            ips = 0

        X_resolution = int(resolution.split('x')[0])
        Y_resolution = int(resolution.split('x')[1])

        ppi = ((X_resolution**2)+(Y_resolution**2))**0.5/(screen_size)

        query = np.array([company, types, ram, weight,
                        touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

        query = query.reshape(1, 12)

        prediction = int(np.exp(rf.predict(query)[0]))
        print(prediction)
        return render_template('prediction.html',p1=str(prediction-1000),p2=str(prediction+1000))

	
if __name__ == '__main__':
   app.run(debug = True)