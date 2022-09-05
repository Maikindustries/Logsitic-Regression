"""
Regresión Logística

Miguel Ángel Pérez López A01750145
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoide(Z):#Z entrada lineal
  """Función sigmoide"""
  return 1/(1+np.exp(-Z))

def iniciar_pesos_cero(longitudEntrada):#numero de atributos
  """Función para iniciar pesos y bias"""
  w = np.zeros((longitudEntrada, 1))
  b = 0
  return w, b

def forward_propagation(w, b, X): 
  """Función de forward propagation y = ax + b"""
  A = sigmoide(np.dot(w.T, X) + b) #predicción
  return A

def back_propagation(A, X, Y):
  """Función de backpropagation"""
  m = X.shape[1]
  wGrad = (1/m)*(np.dot(X, (A-Y).T)) #dervivada de w
  bGrad = (1/m)*(np.sum(A-Y))
  return wGrad, bGrad

def calcular_costo(A, Y): #predicción y salida #1 x numero de muestras
  """Función para calcular el costo ó error"""
  m = Y.shape[1]
  costo = -(1/m)*np.sum(Y*np.log(A)+((1-Y)*np.log(1-A)))
  return costo 

def descenso_gradiente(X, Y, w, b, learning_rate, epocas, 
                      mostrar_costo=False):
  """Función del descenso del gradiente"""
  costos = [] #Para graficar los costos y ver si converge
  for i in range(epocas):
    A = forward_propagation(w, b, X)
    costo = calcular_costo(A, Y)
    wGrad, bGrad =  back_propagation(A, X, Y)

    #Actualizar los pesos
    w = w - learning_rate * wGrad
    b = b - learning_rate * bGrad
    
    if i%100 == 0:
    #Almacenar los costos
      costos.append(costo)
    if mostrar_costo and i%100 == 0:
      print("Costo de la epoca " + str(i)+" : "+str(costo))
    
  #Guardar parámetros
  parametros = {"w":w, "b":b}
  #gradientes = {"wGrad":wGrad, "bGrad":bGrad}
  return parametros, costos

def predecir(w,b,X):
  """Función para predecir con el modelo"""
  m = X.shape[1]
  resultadoY = np.zeros((1,m))
  w = w.reshape(X.shape[0],1)

  #Computamos propagazión hacia adelante
  A = forward_propagation(w, b, X) #predicciones
  for i in range(A.shape[1]): #m 
    #convertir valores flotantes a 0 o 1
    if (A[0][i] <= 0.5):
      resultadoY[0][i] = 0
    else:
      resultadoY[0][i] = 1
  return resultadoY

def regresion_logistica(entrenaX, entrenaY, pruebaX, pruebaY, epocas=2000,
                       learning_rate=0.005, depurar=False):
  """Función de regresión logística"""
  #Iniciar pesos
  w, b = iniciar_pesos_cero(entrenaX.shape[0])
  #Decenso del gradiente
  parametros, costos = descenso_gradiente(entrenaX, entrenaY, w, b,
                                            learning_rate, epocas, 
                                            mostrar_costo=depurar)
  #Devolver parámetros
  w = parametros["w"]
  b = parametros["b"]

  #Predecir los conjuntos de entrenamiento y prueba
  prediccionPrueba = predecir(w, b, pruebaX)
  prediccionEntrena = predecir(w, b, entrenaX)

  #Imprimir la precisión
  precisionEntrena = 100-np.mean((np.abs(prediccionEntrena-entrenaY))*100)
  precisionPruebas = 100-np.mean((np.abs(prediccionPrueba-pruebaY))*100)
  print(f"Precisión en el conjunto de entrenamiento: {precisionEntrena}")
  print(f"Precisión en el conjunto de pruebas: {precisionPruebas}")

  #Guardar los resultados
  resultados = {"costos":costos, "prediccionPrueba": prediccionPrueba, 
                "w":w, "b":b, "learningRate":learning_rate,
                "epocas": epocas, "precisionPruebas":precisionPruebas,
                "precisionEntrenamiento": precisionEntrena}
  return resultados

def prueba_dataset(df, train_size, epocass=30000, learning_ratee=0.005, graficar=False, depurar_costo=False):
  #Separar el dataset en train y test
  df['is_train'] = np.random.uniform(0, 1, len(df)) <= train_size
  train, test = df[df['is_train']==True], df[df['is_train']==False]
  #Borrar columna auxiliar
  train, test = train.drop('is_train', axis=1), test.drop('is_train', axis=1)

  X_train, y_train = train.iloc[:,:-1], train.iloc[:, -1]
  X_test, y_test = test.iloc[:,:-1], test.iloc[:, -1]

  #Normalizar el dataset
  normalizacion = lambda x, minn, maxx: (x-minn)/(maxx-minn)
  
  minn = X_train.min()
  maxx = X_train.max()
  X_train = normalizacion(X_train, minn, maxx)
  
  minn = X_test.min()
  maxx = X_test.max()
  X_test = normalizacion(X_test, minn, maxx)
  
  #Preparar las variables x y y para los calculos matriciales
  X_train = X_train.to_numpy().T
  y_train = y_train.to_numpy().reshape(1, X_train.shape[1])

  X_test = X_test.to_numpy().T
  y_test = y_test.to_numpy().reshape(1, X_test.shape[1])

  #Entrenar modelo
  resultados = regresion_logistica(X_train, y_train, X_test, y_test,
                                  epocas=epocass,
                                  learning_rate=learning_ratee, 
                                  depurar=depurar_costo)

  #Graficar los costos
  if graficar:
    costos = np.squeeze(resultados["costos"]) 
    plt.plot(costos)
    plt.ylabel("Costo")
    plt.xlabel("Epocas (cada 100)")
    plt.title(f"Learning Rate = {resultados['learningRate']}")
    plt.show()
  return resultados

def main():
  #Cargar el dataset
  df = pd.read_csv("winequality_red.csv")

  #Pruebas con un dataset
  print()
  resultado_1 = prueba_dataset(df, train_size=0.8, graficar=True,
                               epocass=30000, learning_ratee=0.005,
                               depurar_costo=True)
  resultado_2 = prueba_dataset(df, train_size=0.7, graficar=True,
                               epocass=30000, learning_ratee=0.005,
                               depurar_costo=True)

  print()
  if resultado_1["precisionPruebas"] > resultado_2["precisionPruebas"]:
    print("El modelo con 80% de datos de entrenamiento tuvo un mejor " \
          "aprendizaje")
    print(f"{resultado_1['precisionPruebas']} > " \
          f"{resultado_2['precisionPruebas']}")
  else:
    print("El modelo con 70% de datos de entrenamiento tuvo un mejor " \
          "aprendizaje")
    print(f"{resultado_2['precisionPruebas']} > "\
          f"{resultado_1['precisionPruebas']}")
  print()

  #Predicciones
  resultados = prueba_dataset(df, train_size=0.8, epocass=30000,
                              learning_ratee=0.005,graficar=False)
  #Valores que regresan una clase de 0.
  prueba_1 = np.array([7.8, 0.88, 0., 2.6, 0.098, 25., 67.,  
                       0.9968, 3.2, 0.68, 9.8]).reshape(11, 1)#1
  prueba_2 = np.array([8.6, 0.38, 0.36, 3., 0.081, 30., 119., 
                       0.997, 3.2, 0.56, 9.4]).reshape(11, 1)#53
  for prueba in [prueba_1, prueba_2]:
    prediccion = predecir(resultados["w"], resultados["b"], prueba)
    if prediccion == 0.:
      print("Predicción correcta")
    else:
      print("Predicción incorrecta")

  #Valores que regresan una clase de 1.
  prueba_3 = np.array([7.8, 0.5, 0.3, 1.9, 0.075, 8., 22., 
                       0.9959, 3.31, 0.56, 10.4]).reshape(11, 1)#101
  prediccion = predecir(resultados["w"], resultados["b"], prueba_3)
  if prediccion == 1.:
    print("Predicción correcta")
  else:
    print("Predicción incorrecta")


if __name__ == "__main__":
    main()