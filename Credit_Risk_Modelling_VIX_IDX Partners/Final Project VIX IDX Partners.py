#!/usr/bin/env python
# coding: utf-8

# # Credit Risk Prediction Model
# 
# By DIMAS ANGGA WIJAYA MUKTI

# # Import Library

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import precision_score,recall_score, f1_score


# # Business Understanding

# Sebagai tugas akhir dari masa kontrak saya sebagai intern Data Scientist di ID/X Partners, kali ini saya akan dilibatkan dalam projek dari sebuah lending company. Saya akan berkolaborasi dengan berbagai departemen lain dalam projek ini untuk menyediakan solusi teknologi bagi company tersebut. Saya diminta untuk membangun model yang dapat memprediksi credit risk menggunakan dataset yang disediakan oleh company yang terdiri dari data pinjaman yang diterima dan yang ditolak. Selain itu saya juga perlu mempersiapkan media visual untuk mempresentasikan solusi ke klien.

# # Analytical Approach

# setelah mengetahui tentang permasalahan yang diperoleh dari resiko kredit, maka akan dilakukan analisis dan modeling sebagai berikut:
# 
# 1. Logistic Regression
# 2. Random Forest
# 3. K-Nearest Neightbor
# 4. Support Vector Machines
# 5. Neural Network

# # Data Requirements

# Data yang digunakan sudah disediakan oleh perusahaan terkait. oleh karena itu, data yang digunakan cukup itu saja

# # Data Collection

# Data diperoleh dengan mendownload melalui website perusahaan terkait

# # Data Understanding

# In[2]:


#Membaca dataframe yang akan digunakan
df_loan = pd.read_csv('Downloads\loan_data_2007_2014.csv')


# In[3]:


#Melihat 5 baris pertama pada dataframe
pd.options.display.max_columns = None
df_loan.head()


# In[4]:


#Menngetahui informasi tentang dataframe
df_loan.info()


# Data df_loan memiliki jumlah kolom sebanyak 75 dan jumlah data sebanyak 466285. Data memiliki kolom yang cukup banyak, sehingga perlu dilihat lagi terkait kelengkapan data setiap kolom

# In[5]:


pd.options.display.max_rows = None
df_loan.isnull().sum()


# Hasil pengecekan jumlah data null dalam setiap kolom, dapat dilihat bahwa masih terdapat nilai null yang cukup banyak. Ada juga yang seluruh data berisi nilai null. Perlu dilihat lagi lebih jauh tentang data terkait

# # Eksploratory Data Analysis

# ## Loan Distribution

# In[6]:


#Membuat histogram dari jumlah yang diajukan peminjam
s = sns.displot(data=df_loan, x="loan_amnt",kind="kde")
s.set(title='Jumlah Pinjaman yang Diajukan Peminjam')
s.set_axis_labels("Jumlah pinjaman","Density")


# In[7]:


#Membuat histogram dari jumlah yang diberikan lender
s = sns.displot(data=df_loan, x="funded_amnt",kind="kde",color='red')
s.set(title='Jumlah Pinjaman yang Diberikan Lender')
s.set_axis_labels("Jumlah pinjaman","Density")


# In[8]:


#Membuat histogram dari jumlah komitmen investor
s = sns.displot(data=df_loan, x="funded_amnt",kind="kde",color='orange')
s.set(title='Jumlah Komitmen Investor')
s.set_axis_labels("Jumlah","Density")


# Terlihat bahwa ketiga variabel memiliki grafik distribusi yang mirip, sehingga dapat kita simpulkan bahwa semua pengajuan sesuai dengan yang diberikan oleh lender

# ## Interest Rate

# In[10]:


def datetime_imputer(df,col):
    df[col] = pd.to_datetime(df[col], format = "%b-%y")

datetime_imputer(df_loan,'issue_d')


# In[11]:


#Membuat plot dari interest rate
s = sns.lineplot(data=df_loan, x="issue_d", y="int_rate",color='green')
s.set(xlabel="Tanggal",ylabel="Suku Bunga",title='Tingkat Suku Bunga')
plt.show()


# Terlihat bahwa suku bunga yang ditetapkan oleh lender cenderung memiliki tren naik dan fluktuatif. Dari 2013 sampai 2014 akhir menunjukkan penurunan

# ## Tujuan Pinjaman

# In[12]:


a = df_loan.groupby('purpose')['id'].count().sort_values(ascending = False).reset_index()
a.rename(columns = {'id' : 'count'}, inplace = True)
s = sns.barplot(data=a,x="count",y="purpose",color="r")


# Informasi dari plot di atas adalah tujuan peminjam terbanyak adalah untuk membayar hutang. Terbanyak kedua adalah untuk membayar kartu kredit

# ## Pinjaman Bermasalah

# In[13]:


problem = ['Charged Off','Default','Late (31-120 days)','In Grace Period','Late (16-30 days)','Does not meet the credit policy. Status:Charged Off']
problem_col = df_loan[df_loan['loan_status'].isin(problem)]
b = problem_col.groupby('loan_status')['id'].count().sort_values(ascending = False).reset_index()
b.rename(columns = {'id' : 'count'}, inplace = True)
s = sns.barplot(data=b,x="count",y="loan_status",color="b")


# Terlihat bahwa peminjam bermasalah terbanyak adalah charged off. Yaitu peminjam yang sudah dipastikan bermasalah oleh lender dalam proses pembayarannya

# ## Loan Grades

# In[14]:


count_grades = df_loan.groupby('grade')['id'].count().reset_index()
s = sns.barplot(data=count_grades,x="id",y="grade",color="g")


# Peminjam banyak yang berada di kelas B dan C

# # Data Preparation

# ## Missing Value handling

# In[15]:


#Menghapus kolom yang bernilai null semua
df_loan.dropna(how='all',axis=1,inplace=True)


# In[16]:


pd.options.display.max_rows = 100
df_loan.isnull().sum()


# ## Redundan Kolom

# Pada df_loan terdapat kolom yang redundan dan tidak perlu dimasukkan untuk pemodelan, yaitu:
# 
# 1. id        : sudah terwakilkan dengan index
# 2. member_id : sudah terwakilkan dengan index
# 3. sub_grade : sudah terwakilkan dengan grade
# 4. pymnt_plan
# 5. url
# 6. title
# 7. zip_code
# 8. addr_state
# 9. emp_title
# 10. desc
# 11. application_type
# 12. policy_code

# In[17]:


df_loan.drop(['Unnamed: 0','id','member_id','sub_grade','application_type','policy_code','emp_title','desc','pymnt_plan','url','title','zip_code','addr_state'],axis=1,inplace=True)
df_loan


# In[18]:


df_loan['total_acc'].unique()


# ## Imputasi Missing Value

# In[19]:


df_loan.isnull().sum()


# In[20]:


df_loan.info()


# ### emp_length

# In[21]:


df_loan['emp_length'].unique()


# In[22]:


#Mengubah tipe data emp_length menjadi integer dan mengisi null dengan rata-rata
def emp_length_num_fill(df,col):
    df[col] = df[col].str.replace('\+ years','')
    df[col] = df[col].str.replace('< 1 year','0')
    df[col] = df[col].str.replace(' years','')
    df[col] = df[col].str.replace(' year','')
    df[col] = pd.to_numeric(df[col])
    df[col].fillna(df[col].mean(),inplace=True)
    
emp_length_num_fill(df_loan,'emp_length')


# In[23]:


df_loan['emp_length'].isnull().sum()


# ### issue_d, earliest_cr_line,last_pymnt_d, next_pymnt_d, last_credit_pull_d

# In[24]:


df_loan['issue_d'].unique()


# In[25]:


#Mengubah tipe data issue_d
def datetime_convert(df,col):
    date = pd.to_datetime('2022-09-01')
    df_loan[col+'_day_diff'] = (date - df[col])/np.timedelta64(1,'D')
    df_loan[col+'_day_diff'] = df_loan[col+'_day_diff'].apply(lambda x: df_loan[col+'_day_diff'].max() if x < 0 else x)

datetime_convert(df_loan,'issue_d')


# In[26]:


df_loan['issue_d_day_diff'].unique()


# In[27]:


#Mengisi missing value dengan
def datetime_imputer(df,col):
    df[col] = pd.to_datetime(df[col],format='%b-%y')
    date = pd.to_datetime('2022-09-01')
    df_loan[col+'_day_diff'] = (date - df[col])/np.timedelta64(1,'D')
    df_loan[col+'_day_diff'] = df_loan[col+'_day_diff'].apply(lambda x: df_loan[col+'_day_diff'].max() if x < 0 else x)
    df_loan[col+'_day_diff'].fillna(df_loan[col+'_day_diff'].value_counts().index[0],inplace=True)

datetime_imputer(df_loan,'earliest_cr_line')
datetime_imputer(df_loan,'last_pymnt_d')
datetime_imputer(df_loan,'next_pymnt_d')
datetime_imputer(df_loan,'last_credit_pull_d')


# In[28]:


#Menghapus kolom earliest_cr_line dan diganti dengan earliest_cr_line_day_diff
df_loan.drop(['issue_d','earliest_cr_line','last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d'],axis=1,inplace=True)


# ### annual_inc, delinq_2yrs, inq_last_6mths, mths_since_last_delinq, mths_since_last_record, open_acc, pub_rec, revol_util, total_acc, collections_12_mths_ex_med, mths_since_last_major_derog, acc_now_delinq, tot_coll_amt, tot_cur_bal, total_rev_hi_lim

# In[29]:


#Mengisi missing value dengan rata-rata
def num_imputer(df,col):
    df[col].fillna(round(df[col].mean()),inplace=True)

num_imputer(df_loan,'annual_inc')
num_imputer(df_loan,'delinq_2yrs')
num_imputer(df_loan,'inq_last_6mths')
num_imputer(df_loan,'mths_since_last_delinq')
num_imputer(df_loan,'mths_since_last_record')
num_imputer(df_loan,'open_acc')
num_imputer(df_loan,'pub_rec')
num_imputer(df_loan,'revol_util')
num_imputer(df_loan,'total_acc')
num_imputer(df_loan,'collections_12_mths_ex_med')
num_imputer(df_loan,'mths_since_last_major_derog')
num_imputer(df_loan,'tot_coll_amt')
num_imputer(df_loan,'acc_now_delinq')
num_imputer(df_loan,'tot_cur_bal')
num_imputer(df_loan,'total_rev_hi_lim')


# In[30]:


df_loan.isnull().sum()


# ### Term

# In[31]:


#Mengubah tipe data term menjadi integer dan mengisi null dengan rata-rata
def term_imputer(df,col):
    df[col] = df[col].str.replace(' months','')
    df[col] = pd.to_numeric(df[col])
    
term_imputer(df_loan,'term')


# In[32]:


df_loan['term']


# Deskripsi
# 
# loan_amnt                 : jumlah pinjaman yang diajukan peminjam
# funded_amnt               : jumlah dana yang dipinjamkan
# term                      : jangka waktu peminjaman
# int_rate                  : suku bunga
# installment               : angsuran
# grade                     : kelas
# sub_grade                 : sub-kelas
# emp_title                 : jabatan peminjam
# emp_length                : lama kerja peminjam
# home_ownership            : kepemilikan rumah dari peminjam
# annual_inc                : penghasilan tahunan yang disediakan oleh peminjam selama pendaftaran
# verrification_status      : status pendapatan diverifikasi oleh LC
# issue_d                   : bulan peminjaman didanai
# loan_status               : status pinjaman saat ini
# pymnt_plan                : rencana pembayaran untuk pinjaman
# url                       : url untuk halaman LC
# desc                      : deskripsi pinjaman dari peminjam
# purpose                   : kategori yang disediakan oleh peminjam untuk permintaan
# title                     : judul pinjaman oleh peminjam
# zip_code                  : 3 digit pertama kode pos
# addr_state                : negara peminjam
# dti                       : Rasio yang dihitung menggunakan total pembayaran utang bulanan peminjam dan pendapatan bulanan peminjam 
# delinq_2yrs               : Jumlah 30+ hari kejadian tunggakan yang jatuh tempo sebelumnya dalam file kredit peminjam selama 2 tahun terakhir
# earliest_cr_line          : Tanggal batas kredit peminjam yang paling awal dilaporkan
# inq_last_6mths            : jumlah pertanyaan dalam 6 bulan terakhir
# mths_since_last_delinq    : jumlah bulan sejak kenakalan terakhir peminjam
# mths_since_last_record    : jumlah bulan sejak catatan publik terakhir
# open_acc                  : jumlah jalur kredit terbuka dalam file kredit peminjam
# pub_rec                   : jumlah catatan publik yang menghina
# revol_bal                 : Total saldo bergulir kredit
# revol_util                : Tingkat pemanfaatan jalur bergulir
# total_acc                 : jumlah total jalur kredit saat ini
# initial_list_status       : status daftar awal peminjam
# out_prncp                 : Sisa pokok terutang untuk jumlah total yang didanai
# out_prncp_inv             : Sisa pokok yang beredar untuk porsi jumlah total yang didanai oleh investor
# total_pymnt               : pembayaran diterima hingga saat ini untuk total yang didanai
# total_pymnt_inv           : pembayaran diterima hingga saat ini untuk sebagian yang didanai
# total_rec_prncp           : sisa pokok diterima saat ini
# total_rec_int             : bunga yang diterima saat ini
# total_rec_late_fee        : biaya keterlambatan yang diterima saat ini
# recoveries                : rencana pembayaran untuk pinjaman
# collection_recovery_fee   : biaya pasca biaya pengumpulan
# last_pymnt_d              : bulan terakhir pembayaran diterima
# last_pymnt_amnt           : bulan terakhir pembayaran diterima
# next_pymnt_d              : tanggal pembayaran terjadwal berikutnya
# collections_12_mths_ex_med: jumlah koleksi dalam 12 bulan 
# mths_since_last_major_derog: Bulan sejak peringkat 90 hari atau lebih buruk terbaru
# policy_code                : kode keamanan yang tersedia
# application_type           : Menunjukkan apakah pinjaman adalah aplikasi individu atau aplikasi bersama dengan dua peminjam bersama
# acc_now_delinq             : Jumlah akun di mana peminjam sekarang nakal
# tot_coll_amt               : Total jumlah pengumpulan yang pernah ada
# tot_cur_bal                : Total Saldo Saat Ini dari Semua Akun
# total_rev_hi_lim           : Total revolving batas kredit/kredit yang tinggi

# In[39]:


# Analisis Korelasi
f, ax = plt.subplots(figsize=(15, 15))
corr = df_loan.corr()
sns.heatmap(corr, linewidths=.5, ax=ax)
plt.show()


# Dari heatmap tersebut dapat dilihat ada beberapa variabel yang akan menyebabkan multikolinearitas, yaitu:
# 
# 1. funded_amnt_inv
# 2. out_prncp_inv
# 3. total_pymnt_inv
# 4. loan_amnt (sudah diwakili oleh funded_amnt)

# In[41]:


#Menghilangkan variabel pemicu multikolinearitas
df_loan.drop(['funded_amnt_inv','out_prncp_inv','total_pymnt_inv','loan_amnt'],axis=1,inplace=True)


# In[42]:


# Analisis Korelasi
f, ax = plt.subplots(figsize=(15, 15))
corr = df_loan.corr()
sns.heatmap(corr, linewidths=.5, ax=ax)
plt.show()


# ## Mengidentifikasi variabel target (loan_status)

# In[43]:


#Melihat nilai unique dan jumlahnya pada loan_status
df_loan['loan_status'].value_counts()


# ### Diketahui peminjaman yang bermasalah:
# 
# 1. Charged off       : pembayaran diperkirakan tidak lunas
# 2. Default           : pembayaran default
# 3. Does not meet the credit policy. Status:Charged Off : pembayaran diperkirakan tidak lunas
# 
# Diketahui peminjaman yang tidak bermasalah:
# 1. Current           : pembayaran tepat waktu
# 2. Fully Paid        : pembayaran lunas
# 3. Does not meet the credit policy. Status:Fully Paid : pembayaran lunas
# 
# Peminjaman yang masih belum tau masuk kategori mana:
# 1. Late (16-31 days) : pembayaran telat dilunasi dalam 16-31 hari
# 2. In Grace Period   : pembayaran dalam masa tenggang
# 3. Late (31-120 days) : pembayaran telat dilunasi dalam 31-120 hari
# 
# oleh karena itu, akan dilihat bagaimana kondisi peminjam yang belum masuk dalam kategori bermasalah atau tidak. akan dilakukan analisis deskriptif

# In[44]:


df_loan.isnull().sum()


# ### Late (16-31 days) 

# In[45]:


#Akan dicari rasio peminjam yang sudah membayar kurang dari 80% dengan jumlah total peminjam late (16-30 days)
late_16_30 = df_loan.loc[df_loan['loan_status']=='Late (16-30 days)']
percentage = round((len(late_16_30[late_16_30['total_pymnt']<late_16_30['funded_amnt']*0.8])/len(late_16_30))*100,2)
percentage


# Karena jumlah peminjam pada Late (16-31 days) 75.62% total pembayarannya kurang dari 80% pinjaman, maka untuk loan_status Late (16-31 days) masuk dalam Default

# ### In Grace Period

# In[46]:


#Akan dicari rasio peminjam yang sudah membayar kurang dari 80% dengan jumlah total peminjam late (16-30 days)
In_Grace_Period = df_loan.loc[df_loan['loan_status']=='In Grace Period']
percentage = round((len(In_Grace_Period[In_Grace_Period['total_pymnt']<In_Grace_Period['funded_amnt']*0.8])/len(In_Grace_Period))*100,2)
percentage


# Karena jumlah peminjam yang berada pada masa tenggang sebesar 71,33% total pembayaran kurang dari 80% pinjaman, dan ada kemungkinan akan terlambat dan akan default,maka In Grace periode masuk ke dalam default

# Disini diperoleh pengelompokan loan_status ke dalam 2 jenis yaitu:
#     
# 1. Default     : Charged off, Late (31-120 days), Default, Does not meet the credit policy. Status:Charged Off, In Grace Period, Late (16-30 days)
# 2. non-Default : Current, Fully Paid, Does not meet the credit policy. Status:Fully Paid

# In[47]:


#Membuat kolom baru dengan nama Default_Non-Default dan beri nilai 1 untuk Non-Default, 0 untuk Default
Deft = np.array(['Current', 'Fully Paid', 'Does not meet the credit policy. Status:Fully Paid'])
df_loan['Default_Non-Default'] = np.where(df_loan['loan_status'].isin(Deft),1,0)
df_loan['Default_Non-Default']


# In[48]:


df_loan


# In[49]:


#Mengapus kolom loan_status karena sudah diwakili oleh Default_Non-Default
df_loan.drop(['loan_status'],axis=1,inplace=True)


# In[50]:


df_loan.head()


# ## Membuat variabel dummy dari data kategori

# In[51]:


df_loan_dummies = [pd.get_dummies(df_loan['grade'], prefix = 'grade', prefix_sep = ':'),
                     pd.get_dummies(df_loan['home_ownership'], prefix = 'home_ownership', prefix_sep = ':'),
                     pd.get_dummies(df_loan['verification_status'], prefix = 'verification_status', prefix_sep = ':'),
                     pd.get_dummies(df_loan['purpose'], prefix = 'purpose', prefix_sep = ':'),
                     pd.get_dummies(df_loan['initial_list_status'], prefix = 'initial_list_status', prefix_sep = ':')]


# In[52]:


df_loan_dummies = pd.concat(df_loan_dummies,axis=1)
df_loan_dummies


# In[53]:


df_loan = pd.concat([df_loan,df_loan_dummies],axis=1)
df_loan


# In[54]:


df_loan.drop(['initial_list_status','grade','home_ownership','verification_status','purpose'],axis=1,inplace=True)


# In[55]:


df_loan.info()


# # Data Preprocessing

# ## Data Split

# In[56]:


X = pd.concat([df_loan.iloc[:,:40],df_loan.iloc[:,41:]],axis=1).values
y = df_loan.iloc[:,40]


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[58]:


X_train


# # Modeling

# ## Logistic Regression

# In[59]:


#Logistic Regression
LR = LogisticRegression(random_state=42)
Logistic = LR.fit(X_train, y_train)


# In[60]:


y_pred_LogReg = LR.predict(X_test)


# In[61]:


y_pred_LogReg


# In[62]:


conf_matrix_LogReg = confusion_matrix(y_test, y_pred_LogReg)
conf_matrix_LogReg


# In[63]:


print("Accuracy:","%.2f%%" % (accuracy_score(y_test, y_pred_LogReg)*100))


# In[64]:


LR.intercept_


# In[65]:


LR.coef_


# In[66]:


y_pred_prob = LR.predict_proba(X_test)
y_pred_prob


# In[67]:


print(classification_report(y_test, y_pred_LogReg))


# In[68]:


conf_matrix_percentage_LogReg = (conf_matrix_LogReg/X_test.shape[0])*100
sns.heatmap(conf_matrix_percentage_LogReg, annot = True)
plt.title("Confusion Matrix Logistic Regression")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()


# In[69]:


accuracy = accuracy_score(y_test, y_pred_LogReg)
precision = precision_score(y_test, y_pred_LogReg)
recall = recall_score(y_test, y_pred_LogReg)
fscore = f1_score(y_test, y_pred_LogReg)

print('Accuracy predict: %.2f%%' % (accuracy * 100))
print('Precision: %.2f%%' % (precision * 100))
print('Recall: %.2f%%' % (recall * 100))
print('F1_score: %.2f%%' % (fscore * 100))


# ## Random Forest

# In[70]:


RFC = RandomForestClassifier(max_depth = 3, min_samples_split = 2, min_samples_leaf = 1)
RFC.fit(X_train, y_train)


# In[87]:


y_pred_RFC = RFC.predict(X_test)


# In[88]:


y_pred_RFC


# In[89]:


conf_matrix_RFC = confusion_matrix(y_test, y_pred_RFC)
conf_matrix_RFC


# In[90]:


print("Accuracy:","%.2f%%" % (accuracy_score(y_test, y_pred_RFC)*100))


# In[124]:


print(classification_report(y_test, y_pred_RFC))


# In[93]:


conf_matrix_percentage_RFC = (conf_matrix_RFC/X_test.shape[0])*100
sns.heatmap(conf_matrix_percentage_RFC, annot = True)
plt.title("Confusion Matrix Random Forest Classifier")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()


# In[122]:


accuracy = accuracy_score(y_test, y_pred_RFC)
precision = precision_score(y_test, y_pred_RFC,average='weighted')
recall = recall_score(y_test, y_pred_RFC,average='weighted')
fscore = f1_score(y_test, y_pred_RFC,average='weighted')

print('Accuracy predict: %.2f%%' % (accuracy * 100))
print('Precision: %.2f%%' % (precision * 100))
print('Recall: %.2f%%' % (recall * 100))
print('F1_score: %.2f%%' % (fscore * 100))


# ## K-Nearest Neighbor

# In[95]:


KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train, y_train)


# In[96]:


y_pred_KNN = KNN.predict(X_test)


# In[97]:


y_pred_KNN


# In[98]:


conf_matrix_KNN = confusion_matrix(y_test, y_pred_KNN)
conf_matrix_KNN


# In[99]:


print("Accuracy:","%.2f%%" % (accuracy_score(y_test, y_pred_KNN)*100))


# In[100]:


print(classification_report(y_test, y_pred_KNN))


# In[101]:


conf_matrix_percentage_KNN = (conf_matrix_KNN/X_test.shape[0])*100
sns.heatmap(conf_matrix_percentage_KNN, annot = True)
plt.title("Confusion Matrix K-Nearest Neightbor")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()


# In[102]:


accuracy = accuracy_score(y_test, y_pred_KNN)
precision = precision_score(y_test, y_pred_KNN)
recall = recall_score(y_test, y_pred_KNN)
fscore = f1_score(y_test, y_pred_KNN)

print('Accuracy predict: %.2f%%' % (accuracy * 100))
print('Precision: %.2f%%' % (precision * 100))
print('Recall: %.2f%%' % (recall * 100))
print('F1_score: %.2f%%' % (fscore * 100))


# ## Support Vector Machines

# In[103]:


SVM_Clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
SVM_Clf.fit(X_train, y_train)


# In[104]:


y_pred_SVM_Clf = SVM_Clf.predict(X_test)


# In[105]:


y_pred_SVM_Clf


# In[106]:


conf_matrix_SVM_Clf = confusion_matrix(y_test, y_pred_SVM_Clf)
conf_matrix_SVM_Clf


# In[107]:


print("Accuracy:","%.2f%%" % (accuracy_score(y_test, y_pred_SVM_Clf)*100))


# In[108]:


print(classification_report(y_test, y_pred_SVM_Clf))


# In[109]:


conf_matrix_percentage_SVM_Clf = (conf_matrix_SVM_Clf/X_test.shape[0])*100
sns.heatmap(conf_matrix_percentage_SVM_Clf, annot = True)
plt.title("Confusion Matrix Support Vector Machines")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()


# In[110]:


accuracy = accuracy_score(y_test, y_pred_SVM_Clf)
precision = precision_score(y_test, y_pred_SVM_Clf)
recall = recall_score(y_test, y_pred_SVM_Clf)
fscore = f1_score(y_test, y_pred_SVM_Clf)

print('Accuracy predict: %.2f%%' % (accuracy * 100))
print('Precision: %.2f%%' % (precision * 100))
print('Recall: %.2f%%' % (recall * 100))
print('F1_score: %.2f%%' % (fscore * 100))


# In[ ]:





# ## Neural Network

# In[111]:


NN = MLPClassifier(random_state=1, max_iter=300)
NN.fit(X_train, y_train)


# In[112]:


y_pred_NN = NN.predict(X_test)


# In[113]:


y_pred_NN


# In[114]:


conf_matrix_NN = confusion_matrix(y_test, y_pred_NN)
conf_matrix_NN


# In[115]:


print("Accuracy:","%.2f%%" % (accuracy_score(y_test, y_pred_NN)*100))


# In[116]:


print(classification_report(y_test, y_pred_NN))


# In[117]:


conf_matrix_percentage_NN = (conf_matrix_NN/X_test.shape[0])*100
sns.heatmap(conf_matrix_percentage_NN, annot = True)
plt.title("Confusion Matrix Neural Network")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()


# In[118]:


accuracy = accuracy_score(y_test, y_pred_NN)
precision = precision_score(y_test, y_pred_NN)
recall = recall_score(y_test, y_pred_NN)
fscore = f1_score(y_test, y_pred_NN)

print('Accuracy predict: %.2f%%' % (accuracy * 100))
print('Precision: %.2f%%' % (precision * 100))
print('Recall: %.2f%%' % (recall * 100))
print('F1_score: %.2f%%' % (fscore * 100))


# # Kesimpulan

# 1. Analisis Deskriptif:
# 
#   a) Terjadi kesamaan anatara jumlah pinjaman yang diajukan oleh peminjam dengan jumlah yang diberikan oleh pihak Lender
#   
#   b) Tingkat suku bunga cenderung memiliki tren naik. Tetapi tetap berfluktuasi setiap tahunnya. Pada awal 2013 sampai akhir 2014 mengalami penurunan yang mungkin disebabkan oleh kebijakan lender 
#   
#   c) Peminjam terbanyak melakukan pinjaman dengan tujuan untuk membayar hutang
#  
#   d) Peminjam bermasalah terbanyak yaitu Charged Off
# 
#   g) Peminjam pada grade B memiliki jumlah terbanyak, dan grade G memiliki jumlah paling sedikit
# 
# 
# 2. Modeling:
#   
#   Didapatkan model yang terbaik adalah model Support Vector Machine dengan nilai akurasi 99,9% dan presisi 100%
