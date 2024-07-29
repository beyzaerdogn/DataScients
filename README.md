# DataScients
RANDOMFOREST MODELİ
KLASİK YÖNTEM

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Modeli oluştur
model = RandomForestClassifier(n_estimators=50, random_state=42)

# Modeli eğit
model.fit(X_train_split, y_train_split)

# Eğitim seti üzerindeki tahminleri yap
y_train_pred = model.predict(X_train_split)
y_valid_pred = model.predict(X_valid_split)

# Test seti üzerindeki tahminleri yap
y_test_pred = model.predict(X_test)
# Eğitim ve doğrulama setleri üzerindeki doğruluk ve raporları yazdır
print("Training Accuracy:", accuracy_score(y_train_split, y_train_pred))
print("Validation Accuracy:", accuracy_score(y_valid_split, y_valid_pred))
print("\nClassification Report:\n", classification_report(y_valid_split, y_valid_pred))
Training Accuracy: 0.9997876302806548
Validation Accuracy: 0.8240329325666492

Classification Report:
               precision    recall  f1-score   support

     Dropout       0.89      0.83      0.86      5028
    Enrolled       0.65      0.58      0.61      3017
    Graduate       0.84      0.92      0.88      7259

    accuracy                           0.82     15304
   macro avg       0.79      0.78      0.78     15304
weighted avg       0.82      0.82      0.82     15304

# Test seti tahminlerini bir DataFrame'e dönüştür
test_results = pd.DataFrame({'id': test_df['id'], 'Prediction': y_test_pred})

# Test sonuçlarını yazdır
print(test_results.head())
      id Prediction
0  76518    Dropout
1  76519   Graduate
2  76520   Graduate
3  76521   Graduate
4  76522    Dropout
import pandas as pd

# Test seti üzerinde tahminleri yap
y_test_pred = model.predict(X_test)

# Tahminleri bir DataFrame'e dönüştürün
submission_df = pd.DataFrame({
    'id': test_df['id'],  # Test veri setinde 'id' sütununu kullanın
    'Target': y_test_pred  # 'Target' sütununu tahmin sonuçlarınızla doldurun
})

# sample_submission DataFrame'ini tahminlerle güncelleyin
sample_submission['Target'] = y_test_pred

# sample_submission DataFrame'ini dosyaya kaydedin
sample_submission.to_csv('/kaggle/working/sample_submission9q.csv', index=False)

# Alternatif olarak, kendi oluşturduğunuz DataFrame'i de kaydedebilirsiniz
submission_df.to_csv('/kaggle/working/sample_submission.csv', index=False)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Hedef değişkeni belirleyin ve çıkartın
y = train_df['Target']
X = train_df.drop(columns=['Target'])

# Eğitim ve test veri setlerini oluşturun
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Kategorik verileri dönüştürün
X_train_dummies = pd.get_dummies(X_train)
X_valid_dummies = pd.get_dummies(X_valid)

# Eğitim ve doğrulama veri setlerinde sütun isimlerini senkronize edin
X_train_dummies, X_valid_dummies = X_train_dummies.align(X_valid_dummies, join='left', axis=1, fill_value=0)

# Özellikleri ölçeklendirin
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_dummies)
X_valid_scaled = scaler.transform(X_valid_dummies)

# Modeli oluşturun
log_model = LogisticRegression(max_iter=2000, random_state=42)  # Max iterasyonu artırdık

# Modeli eğitin
log_model.fit(X_train_scaled, y_train)

# Eğitim ve doğrulama setleri üzerindeki tahminleri yapın
y_train_pred = log_model.predict(X_train_scaled)
y_valid_pred = log_model.predict(X_valid_scaled)

# Performans metriklerini hesaplayın
train_accuracy = accuracy_score(y_train, y_train_pred)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
classification_rep = classification_report(y_valid, y_valid_pred)

print(f"Eğitim Doğruluk Oranı: {train_accuracy}")
print(f"Doğrulama Doğruluk Oranı: {valid_accuracy}")
print("Sınıflandırma Raporu:")
print(classification_rep)
Eğitim Doğruluk Oranı: 0.8233247296370111
Doğrulama Doğruluk Oranı: 0.82723470987977
Sınıflandırma Raporu:
              precision    recall  f1-score   support

     Dropout       0.89      0.84      0.86      5028
    Enrolled       0.66      0.57      0.61      3017
    Graduate       0.85      0.92      0.88      7259

    accuracy                           0.83     15304
   macro avg       0.80      0.78      0.79     15304
weighted avg       0.82      0.83      0.82     15304
