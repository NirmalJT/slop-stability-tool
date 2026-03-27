import joblib

model = joblib.load("model.pkl")

def convert_slope(slope):
    try:
        return float(slope.split(":")[0])
    except:
        return 1

def encode_soil(soil):
    return 0 if soil == "CL" else 1

def predict_fos(c, phi, H, water, slope, soil):

    slope = convert_slope(slope)
    soil = encode_soil(soil)

    return model.predict([[c, phi, H, water, slope, soil]])[0]