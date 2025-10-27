#  KMS Inference Service using ConcreteML

The **KMS Inference Service** is a Flask-based system that handles **Fully Homomorphic Encryption (FHE)** model inference and **Key Management** for secure machine learning deployments.  
It provides APIs for model distribution, encrypted inference, and client key handling across distributed components. The client fetches private homomorphic keys and an evaluation key, then the client sends the public evaluation key to the server for encrypted computations 

---

##  Architecture Overview

The system is composed of two main services:

| Service | Description |
|----------|-------------|
| **KMS Server (`kms`)** | Handles FHE model packaging, client key generation, and key distribution. |
| **Inference Server (`model`)** | Receives encrypted data and performs secure FHE inference using the distributed model. |

Each service runs as an independent Flask API with automatic OpenAPI documentation.

---

##  Features

-  **Fully Homomorphic Encryption (FHE)** model inference using [Concrete ML](https://github.com/zama-ai/concrete-ml)  
-  **Dynamic model and key generation** for each client  
-  **Automatic model packaging** and transfer over HTTP  
-  **SQLite-based key persistence layer**  
-  **CORS + Smorest API documentation**  
-  **Configurable via `.env`** environment variables  
-  **Thread-safe multiprocessing Flask runner**

## Instructions

To run the services
```
   python3 <service>.py
```
OR 
```
  make <service>
```
To stop the service
```
  make stop-<service>
```


