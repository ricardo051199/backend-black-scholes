from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from scipy.stats import norm

app = Flask(__name__)
CORS(app)

def calcular_d1_d2(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def precio_opcion_compra(S, K, r, sigma, T):
    d1, d2 = calcular_d1_d2(S, K, r, sigma, T)
    precio_compra = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return precio_compra

def precio_opcion_venta(S, K, r, sigma, T):
    d1, d2 = calcular_d1_d2(S, K, r, sigma, T)
    precio_venta = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return precio_venta

def ftbs_opciones_black_scholes(S0, K, r, sigma, T, nx, nt, dt, dx, tipo_opcion, x):
    if tipo_opcion == 'compra':
        condicion_inicial = np.maximum(x - K, 0)
    elif tipo_opcion == 'venta':
        condicion_inicial = np.maximum(K - x, 0)
    
    historial_opcion = [condicion_inicial.copy()]
    valores_opcion = condicion_inicial.copy()
    
    for n in range(nt):
        if tipo_opcion == 'compra':
            d1, d2 = calcular_d1_d2(S0, K, r, sigma, T - n * dt)
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T - n * dt))
            theta = -0.5 * sigma ** 2 * S0 ** 2 * gamma - r * K * np.exp(-r * (T - n * dt)) * norm.cdf(d2)
        elif tipo_opcion == 'venta':
            d1, d2 = calcular_d1_d2(S0, K, r, sigma, T - n * dt)
            delta = norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T - n * dt))
            theta = -0.5 * sigma ** 2 * S0 ** 2 * gamma + r * K * np.exp(-r * (T - n * dt)) * norm.cdf(-d2)
        
        valores_opcion[1:-1] = valores_opcion[1:-1] - dt / dx * (delta * valores_opcion[1:-1] - r * valores_opcion[1:-1] * dt + 0.5 * sigma ** 2 * valores_opcion[1:-1] * dx ** 2 * gamma - theta * dt)
        
        valores_opcion[0] = 0 if tipo_opcion == 'compra' else K * np.exp(-r * (T - n * dt)) - S0
        valores_opcion[-1] = S0 - K * np.exp(-r * (T - n * dt)) if tipo_opcion == 'compra' else 0
        
        historial_opcion.append(valores_opcion.copy())
    
    return historial_opcion

@app.route('/')
def hola_mundo():
    return jsonify(mensaje="¡Hola desde el backend en Python!")

@app.route('/modelo-black-scholes', methods=['POST'])
def calcular_black_scholes():
    try:
        data = request.get_json()
        tipo_opcion = data.get('tipo_opcion', 'ambas')
        S0 = float(data.get('S0', 0))
        K = float(data.get('K', 0))
        r = float(data.get('r', 0))
        sigma = float(data.get('sigma', 0))
        T = float(data.get('T', 0))

        nx = 100
        xmin = 0
        xmax = 200
        dx = (xmax - xmin) / (nx - 1)

        nt = 100
        dt = T / nt

        x = np.linspace(xmin, xmax, nx)

        if tipo_opcion == 'ambas':
            historial_venta = ftbs_opciones_black_scholes(S0, K, r, sigma, T, nx, nt, dt, dx, 'venta', x)
            historial_compra = ftbs_opciones_black_scholes(S0, K, r, sigma, T, nx, nt, dt, dx, 'compra', x)
            historial_venta_serializable = [arr.tolist() for arr in historial_venta]
            historial_compra_serializable = [arr.tolist() for arr in historial_compra]
            resultado = {'x': x.tolist(), 'historial-venta': historial_venta_serializable, 'historial-compra': historial_compra_serializable}
        else:
            historial = ftbs_opciones_black_scholes(S0, K, r, sigma, T, nx, nt, dt, dx, tipo_opcion, x)
            historial_serializable = [arr.tolist() for arr in historial]
            resultado = {'x': x.tolist(), 'historial': historial_serializable}
        
        return jsonify(resultado=resultado)
    except ValueError:
        return jsonify(error="Los datos proporcionados no son válidos"), 400

if __name__ == '__main__':
    app.run(debug=True)
