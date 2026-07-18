"""Boyut-bazlı z-skor standardizasyonu.

Gerekçe (ölçülen): Qwen2.5'in orta katmanlarında "massive activation" /
attention-sink boyutları var (|değer| ~2000, katman normu ~1500 sabit).
Bu boyutlar kosinüs, CKA ve MSE'yi tek başına domine eder; anlamsal sinyal
görünmez olur. Tüm MCT eğitimi ve metrikler standardize uzayda yapılır;
hedef modele enjeksiyondan önce `zinv` ile gerçek uzaya dönülür.

İstatistikler daima EĞİTİM alt-kümesinden çıkarılır (test sızıntısı yok).
"""
import numpy as np


def zfit(X_train):
    mu = X_train.mean(0)
    sd = X_train.std(0) + 1e-6
    return mu, sd


def zapply(X, mu, sd):
    return (X - mu) / sd


def zinv(X, mu, sd):
    return X * sd + mu
