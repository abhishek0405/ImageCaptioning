from flask import Flask,render_template,redirect,request
import  Caption_final

app=Flask(__name__)
@app.route('/')

def hello():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    if request.method=='POST':
        f=request.files['userfile']
        path=f'./static/{f.filename}'
        f.save(path)
        cap = Caption_final.predict_caption(path)
        cap = cap.capitalize()
        print(cap)
        details={
            "img":path,
            "caption":cap
        }
    return render_template('index.html',details=details)
    
 
if __name__=='__main__':
    app.run(debug=False,threaded=False)
