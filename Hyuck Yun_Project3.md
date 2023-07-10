```python
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
```

# Reading and Cleaning the Dataset


```python
df = pd.read_csv("application_data.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>...</th>
      <th>FLAG_DOCUMENT_18</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 122 columns</p>
</div>




```python
df1 = pd.read_csv("previous_application.csv")
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_APPLICATION</th>
      <th>AMT_CREDIT</th>
      <th>AMT_DOWN_PAYMENT</th>
      <th>AMT_GOODS_PRICE</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>...</th>
      <th>NAME_SELLER_INDUSTRY</th>
      <th>CNT_PAYMENT</th>
      <th>NAME_YIELD_GROUP</th>
      <th>PRODUCT_COMBINATION</th>
      <th>DAYS_FIRST_DRAWING</th>
      <th>DAYS_FIRST_DUE</th>
      <th>DAYS_LAST_DUE_1ST_VERSION</th>
      <th>DAYS_LAST_DUE</th>
      <th>DAYS_TERMINATION</th>
      <th>NFLAG_INSURED_ON_APPROVAL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2030495</td>
      <td>271877</td>
      <td>Consumer loans</td>
      <td>1730.430</td>
      <td>17145.0</td>
      <td>17145.0</td>
      <td>0.0</td>
      <td>17145.0</td>
      <td>SATURDAY</td>
      <td>15</td>
      <td>...</td>
      <td>Connectivity</td>
      <td>12.0</td>
      <td>middle</td>
      <td>POS mobile with interest</td>
      <td>365243.0</td>
      <td>-42.0</td>
      <td>300.0</td>
      <td>-42.0</td>
      <td>-37.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2802425</td>
      <td>108129</td>
      <td>Cash loans</td>
      <td>25188.615</td>
      <td>607500.0</td>
      <td>679671.0</td>
      <td>NaN</td>
      <td>607500.0</td>
      <td>THURSDAY</td>
      <td>11</td>
      <td>...</td>
      <td>XNA</td>
      <td>36.0</td>
      <td>low_action</td>
      <td>Cash X-Sell: low</td>
      <td>365243.0</td>
      <td>-134.0</td>
      <td>916.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2523466</td>
      <td>122040</td>
      <td>Cash loans</td>
      <td>15060.735</td>
      <td>112500.0</td>
      <td>136444.5</td>
      <td>NaN</td>
      <td>112500.0</td>
      <td>TUESDAY</td>
      <td>11</td>
      <td>...</td>
      <td>XNA</td>
      <td>12.0</td>
      <td>high</td>
      <td>Cash X-Sell: high</td>
      <td>365243.0</td>
      <td>-271.0</td>
      <td>59.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2819243</td>
      <td>176158</td>
      <td>Cash loans</td>
      <td>47041.335</td>
      <td>450000.0</td>
      <td>470790.0</td>
      <td>NaN</td>
      <td>450000.0</td>
      <td>MONDAY</td>
      <td>7</td>
      <td>...</td>
      <td>XNA</td>
      <td>12.0</td>
      <td>middle</td>
      <td>Cash X-Sell: middle</td>
      <td>365243.0</td>
      <td>-482.0</td>
      <td>-152.0</td>
      <td>-182.0</td>
      <td>-177.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1784265</td>
      <td>202054</td>
      <td>Cash loans</td>
      <td>31924.395</td>
      <td>337500.0</td>
      <td>404055.0</td>
      <td>NaN</td>
      <td>337500.0</td>
      <td>THURSDAY</td>
      <td>9</td>
      <td>...</td>
      <td>XNA</td>
      <td>24.0</td>
      <td>high</td>
      <td>Cash Street: high</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>



# Dropping Unecessary Column


```python
#dropping unwanted column in df
df = df.drop(["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL", "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY", "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13", "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16", "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21", "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"], axis=1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>...</th>
      <th>FONDKAPREMONT_MODE</th>
      <th>HOUSETYPE_MODE</th>
      <th>TOTALAREA_MODE</th>
      <th>WALLSMATERIAL_MODE</th>
      <th>EMERGENCYSTATE_MODE</th>
      <th>OBS_30_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_30_CNT_SOCIAL_CIRCLE</th>
      <th>OBS_60_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_60_CNT_SOCIAL_CIRCLE</th>
      <th>DAYS_LAST_PHONE_CHANGE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>...</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.0149</td>
      <td>Stone, brick</td>
      <td>No</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>-1134.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>...</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.0714</td>
      <td>Block</td>
      <td>No</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-828.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-815.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-617.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1106.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 84 columns</p>
</div>



# Converting Negative Values to Absolute Values


```python
#For df
df["DAYS_BIRTH"]=abs(df["DAYS_BIRTH"])
df["DAYS_REGISTRATION"]=abs(df["DAYS_REGISTRATION"])
df["DAYS_EMPLOYED"]=abs(df["DAYS_EMPLOYED"])
df["DAYS_ID_PUBLISH"]=abs(df["DAYS_ID_PUBLISH"])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>...</th>
      <th>FONDKAPREMONT_MODE</th>
      <th>HOUSETYPE_MODE</th>
      <th>TOTALAREA_MODE</th>
      <th>WALLSMATERIAL_MODE</th>
      <th>EMERGENCYSTATE_MODE</th>
      <th>OBS_30_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_30_CNT_SOCIAL_CIRCLE</th>
      <th>OBS_60_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_60_CNT_SOCIAL_CIRCLE</th>
      <th>DAYS_LAST_PHONE_CHANGE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>...</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.0149</td>
      <td>Stone, brick</td>
      <td>No</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>-1134.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>...</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.0714</td>
      <td>Block</td>
      <td>No</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-828.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-815.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-617.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1106.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 84 columns</p>
</div>




```python
#For df1
df1["DAYS_FIRST_DRAWING"]=abs(df1["DAYS_FIRST_DRAWING"])
df1["DAYS_FIRST_DUE"]=abs(df1["DAYS_FIRST_DUE"])
df1["DAYS_LAST_DUE_1ST_VERSION"]=abs(df1["DAYS_LAST_DUE_1ST_VERSION"])
df1["DAYS_LAST_DUE"]=abs(df1["DAYS_LAST_DUE"])
df1["DAYS_DECISION"]=abs(df1["DAYS_DECISION"])
df1["DAYS_TERMINATION"]=abs(df1["DAYS_TERMINATION"])
df1["SELLERPLACE_AREA"]=abs(df1["SELLERPLACE_AREA"])
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_APPLICATION</th>
      <th>AMT_CREDIT</th>
      <th>AMT_DOWN_PAYMENT</th>
      <th>AMT_GOODS_PRICE</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>...</th>
      <th>NAME_SELLER_INDUSTRY</th>
      <th>CNT_PAYMENT</th>
      <th>NAME_YIELD_GROUP</th>
      <th>PRODUCT_COMBINATION</th>
      <th>DAYS_FIRST_DRAWING</th>
      <th>DAYS_FIRST_DUE</th>
      <th>DAYS_LAST_DUE_1ST_VERSION</th>
      <th>DAYS_LAST_DUE</th>
      <th>DAYS_TERMINATION</th>
      <th>NFLAG_INSURED_ON_APPROVAL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2030495</td>
      <td>271877</td>
      <td>Consumer loans</td>
      <td>1730.430</td>
      <td>17145.0</td>
      <td>17145.0</td>
      <td>0.0</td>
      <td>17145.0</td>
      <td>SATURDAY</td>
      <td>15</td>
      <td>...</td>
      <td>Connectivity</td>
      <td>12.0</td>
      <td>middle</td>
      <td>POS mobile with interest</td>
      <td>365243.0</td>
      <td>42.0</td>
      <td>300.0</td>
      <td>42.0</td>
      <td>37.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2802425</td>
      <td>108129</td>
      <td>Cash loans</td>
      <td>25188.615</td>
      <td>607500.0</td>
      <td>679671.0</td>
      <td>NaN</td>
      <td>607500.0</td>
      <td>THURSDAY</td>
      <td>11</td>
      <td>...</td>
      <td>XNA</td>
      <td>36.0</td>
      <td>low_action</td>
      <td>Cash X-Sell: low</td>
      <td>365243.0</td>
      <td>134.0</td>
      <td>916.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2523466</td>
      <td>122040</td>
      <td>Cash loans</td>
      <td>15060.735</td>
      <td>112500.0</td>
      <td>136444.5</td>
      <td>NaN</td>
      <td>112500.0</td>
      <td>TUESDAY</td>
      <td>11</td>
      <td>...</td>
      <td>XNA</td>
      <td>12.0</td>
      <td>high</td>
      <td>Cash X-Sell: high</td>
      <td>365243.0</td>
      <td>271.0</td>
      <td>59.0</td>
      <td>365243.0</td>
      <td>365243.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2819243</td>
      <td>176158</td>
      <td>Cash loans</td>
      <td>47041.335</td>
      <td>450000.0</td>
      <td>470790.0</td>
      <td>NaN</td>
      <td>450000.0</td>
      <td>MONDAY</td>
      <td>7</td>
      <td>...</td>
      <td>XNA</td>
      <td>12.0</td>
      <td>middle</td>
      <td>Cash X-Sell: middle</td>
      <td>365243.0</td>
      <td>482.0</td>
      <td>152.0</td>
      <td>182.0</td>
      <td>177.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1784265</td>
      <td>202054</td>
      <td>Cash loans</td>
      <td>31924.395</td>
      <td>337500.0</td>
      <td>404055.0</td>
      <td>NaN</td>
      <td>337500.0</td>
      <td>THURSDAY</td>
      <td>9</td>
      <td>...</td>
      <td>XNA</td>
      <td>24.0</td>
      <td>high</td>
      <td>Cash Street: high</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>



# Find and Fix Missing Values


```python
#Find the columns with null values more than 30%
emptycol=df.isnull().sum()
emptycol=emptycol[emptycol.values>(0.3*len(emptycol))]
len(emptycol)
```




    58




```python
# Dropping the columns
emptycol = list(emptycol[emptycol.values > 0.3].index)
df.drop(labels=emptycol, axis=1, inplace=True)
print(len(emptycol))
```

    58



```python
# Checking which column to fill
df.isnull().sum() / len(df) * 100
```




    SK_ID_CURR                     0.000000
    TARGET                         0.000000
    NAME_CONTRACT_TYPE             0.000000
    CODE_GENDER                    0.000000
    FLAG_OWN_CAR                   0.000000
    FLAG_OWN_REALTY                0.000000
    CNT_CHILDREN                   0.000000
    AMT_INCOME_TOTAL               0.000000
    AMT_CREDIT                     0.000000
    AMT_ANNUITY                    0.003902
    NAME_INCOME_TYPE               0.000000
    NAME_EDUCATION_TYPE            0.000000
    NAME_FAMILY_STATUS             0.000000
    NAME_HOUSING_TYPE              0.000000
    REGION_POPULATION_RELATIVE     0.000000
    DAYS_BIRTH                     0.000000
    DAYS_EMPLOYED                  0.000000
    DAYS_REGISTRATION              0.000000
    DAYS_ID_PUBLISH                0.000000
    CNT_FAM_MEMBERS                0.000650
    REGION_RATING_CLIENT           0.000000
    REGION_RATING_CLIENT_W_CITY    0.000000
    WEEKDAY_APPR_PROCESS_START     0.000000
    HOUR_APPR_PROCESS_START        0.000000
    ORGANIZATION_TYPE              0.000000
    DAYS_LAST_PHONE_CHANGE         0.000325
    dtype: float64




```python
# Replace missing values in the column "AMT_ANNUITY" with the median value
values = df["AMT_ANNUITY"].median()
df.loc[df["AMT_ANNUITY"].isnull(), "AMT_ANNUITY"] = values
```


```python
df.isnull().sum()
```




    SK_ID_CURR                     0
    TARGET                         0
    NAME_CONTRACT_TYPE             0
    CODE_GENDER                    0
    FLAG_OWN_CAR                   0
    FLAG_OWN_REALTY                0
    CNT_CHILDREN                   0
    AMT_INCOME_TOTAL               0
    AMT_CREDIT                     0
    AMT_ANNUITY                    0
    NAME_INCOME_TYPE               0
    NAME_EDUCATION_TYPE            0
    NAME_FAMILY_STATUS             0
    NAME_HOUSING_TYPE              0
    REGION_POPULATION_RELATIVE     0
    DAYS_BIRTH                     0
    DAYS_EMPLOYED                  0
    DAYS_REGISTRATION              0
    DAYS_ID_PUBLISH                0
    CNT_FAM_MEMBERS                2
    REGION_RATING_CLIENT           0
    REGION_RATING_CLIENT_W_CITY    0
    WEEKDAY_APPR_PROCESS_START     0
    HOUR_APPR_PROCESS_START        0
    ORGANIZATION_TYPE              0
    DAYS_LAST_PHONE_CHANGE         1
    dtype: int64




```python
# Remove rows having null values more than 30%
emptyrow = df.isnull().sum(axis=1)
emptyrow = list(emptyrow[emptyrow.values > 0.3*len(df)].index)
df.drop(labels=emptyrow, inplace=True)
print(len(emptyrow))
```

    0



```python
#Do it for df1 as well
emptycol1 = df1.isnull().sum()
emptycol1 = emptycol1[emptycol1.values>(0.3*len(emptycol1))]
len(emptycol1)
```




    15




```python
emptycol1 = list(emptycol1[emptycol1.values > 0.3].index)
df1.drop(labels=emptycol1, axis=1, inplace=True)
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_PREV</th>
      <th>SK_ID_CURR</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>AMT_APPLICATION</th>
      <th>AMT_CREDIT</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>FLAG_LAST_APPL_PER_CONTRACT</th>
      <th>NFLAG_LAST_APPL_IN_DAY</th>
      <th>NAME_CASH_LOAN_PURPOSE</th>
      <th>...</th>
      <th>NAME_PAYMENT_TYPE</th>
      <th>CODE_REJECT_REASON</th>
      <th>NAME_CLIENT_TYPE</th>
      <th>NAME_GOODS_CATEGORY</th>
      <th>NAME_PORTFOLIO</th>
      <th>NAME_PRODUCT_TYPE</th>
      <th>CHANNEL_TYPE</th>
      <th>SELLERPLACE_AREA</th>
      <th>NAME_SELLER_INDUSTRY</th>
      <th>NAME_YIELD_GROUP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2030495</td>
      <td>271877</td>
      <td>Consumer loans</td>
      <td>17145.0</td>
      <td>17145.0</td>
      <td>SATURDAY</td>
      <td>15</td>
      <td>Y</td>
      <td>1</td>
      <td>XAP</td>
      <td>...</td>
      <td>Cash through the bank</td>
      <td>XAP</td>
      <td>Repeater</td>
      <td>Mobile</td>
      <td>POS</td>
      <td>XNA</td>
      <td>Country-wide</td>
      <td>35</td>
      <td>Connectivity</td>
      <td>middle</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2802425</td>
      <td>108129</td>
      <td>Cash loans</td>
      <td>607500.0</td>
      <td>679671.0</td>
      <td>THURSDAY</td>
      <td>11</td>
      <td>Y</td>
      <td>1</td>
      <td>XNA</td>
      <td>...</td>
      <td>XNA</td>
      <td>XAP</td>
      <td>Repeater</td>
      <td>XNA</td>
      <td>Cash</td>
      <td>x-sell</td>
      <td>Contact center</td>
      <td>1</td>
      <td>XNA</td>
      <td>low_action</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2523466</td>
      <td>122040</td>
      <td>Cash loans</td>
      <td>112500.0</td>
      <td>136444.5</td>
      <td>TUESDAY</td>
      <td>11</td>
      <td>Y</td>
      <td>1</td>
      <td>XNA</td>
      <td>...</td>
      <td>Cash through the bank</td>
      <td>XAP</td>
      <td>Repeater</td>
      <td>XNA</td>
      <td>Cash</td>
      <td>x-sell</td>
      <td>Credit and cash offices</td>
      <td>1</td>
      <td>XNA</td>
      <td>high</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2819243</td>
      <td>176158</td>
      <td>Cash loans</td>
      <td>450000.0</td>
      <td>470790.0</td>
      <td>MONDAY</td>
      <td>7</td>
      <td>Y</td>
      <td>1</td>
      <td>XNA</td>
      <td>...</td>
      <td>Cash through the bank</td>
      <td>XAP</td>
      <td>Repeater</td>
      <td>XNA</td>
      <td>Cash</td>
      <td>x-sell</td>
      <td>Credit and cash offices</td>
      <td>1</td>
      <td>XNA</td>
      <td>middle</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1784265</td>
      <td>202054</td>
      <td>Cash loans</td>
      <td>337500.0</td>
      <td>404055.0</td>
      <td>THURSDAY</td>
      <td>9</td>
      <td>Y</td>
      <td>1</td>
      <td>Repairs</td>
      <td>...</td>
      <td>Cash through the bank</td>
      <td>HC</td>
      <td>Repeater</td>
      <td>XNA</td>
      <td>Cash</td>
      <td>walk-in</td>
      <td>Credit and cash offices</td>
      <td>1</td>
      <td>XNA</td>
      <td>high</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



# Finding Duplicate Values


```python
df.duplicated(). sum()
```




    0




```python
df1.duplicated(). sum()
```




    0




```python
#There are no duplicated values
```

# Handling the XNA/XAP Values


```python
#There were XNA values in GENDER column
df["CODE_GENDER"].value_counts()
```




    F      202448
    M      105059
    XNA         4
    Name: CODE_GENDER, dtype: int64




```python
#Just replace it with F (since its 4 values it won't make big difference)
df.loc[df["CODE_GENDER"]=="XNA", "CODE_GENDER"]= "F"
df["CODE_GENDER"].value_counts()
```




    F    202452
    M    105059
    Name: CODE_GENDER, dtype: int64




```python
#Threr were XNA values in ORGANIZATION_TYPE
df["ORGANIZATION_TYPE"].value_counts()
```




    Business Entity Type 3    67992
    XNA                       55374
    Self-employed             38412
    Other                     16683
    Medicine                  11193
    Business Entity Type 2    10553
    Government                10404
    School                     8893
    Trade: type 7              7831
    Kindergarten               6880
    Construction               6721
    Business Entity Type 1     5984
    Transport: type 4          5398
    Trade: type 3              3492
    Industry: type 9           3368
    Industry: type 3           3278
    Security                   3247
    Housing                    2958
    Industry: type 11          2704
    Military                   2634
    Bank                       2507
    Agriculture                2454
    Police                     2341
    Transport: type 2          2204
    Postal                     2157
    Security Ministries        1974
    Trade: type 2              1900
    Restaurant                 1811
    Services                   1575
    University                 1327
    Industry: type 7           1307
    Transport: type 3          1187
    Industry: type 1           1039
    Hotel                       966
    Electricity                 950
    Industry: type 4            877
    Trade: type 6               631
    Industry: type 5            599
    Insurance                   597
    Telecom                     577
    Emergency                   560
    Industry: type 2            458
    Advertising                 429
    Realtor                     396
    Culture                     379
    Industry: type 12           369
    Trade: type 1               348
    Mobile                      317
    Legal Services              305
    Cleaning                    260
    Transport: type 1           201
    Industry: type 6            112
    Industry: type 10           109
    Religion                     85
    Industry: type 13            67
    Trade: type 4                64
    Trade: type 5                49
    Industry: type 8             24
    Name: ORGANIZATION_TYPE, dtype: int64




```python
#Just drop those rows with XNA in ORGANIZATION_TYPE
df=df.drop(df.loc[df["ORGANIZATION_TYPE"]=="XNA"].index)
df["ORGANIZATION_TYPE"].value_counts()
```




    Business Entity Type 3    67992
    Self-employed             38412
    Other                     16683
    Medicine                  11193
    Business Entity Type 2    10553
    Government                10404
    School                     8893
    Trade: type 7              7831
    Kindergarten               6880
    Construction               6721
    Business Entity Type 1     5984
    Transport: type 4          5398
    Trade: type 3              3492
    Industry: type 9           3368
    Industry: type 3           3278
    Security                   3247
    Housing                    2958
    Industry: type 11          2704
    Military                   2634
    Bank                       2507
    Agriculture                2454
    Police                     2341
    Transport: type 2          2204
    Postal                     2157
    Security Ministries        1974
    Trade: type 2              1900
    Restaurant                 1811
    Services                   1575
    University                 1327
    Industry: type 7           1307
    Transport: type 3          1187
    Industry: type 1           1039
    Hotel                       966
    Electricity                 950
    Industry: type 4            877
    Trade: type 6               631
    Industry: type 5            599
    Insurance                   597
    Telecom                     577
    Emergency                   560
    Industry: type 2            458
    Advertising                 429
    Realtor                     396
    Culture                     379
    Industry: type 12           369
    Trade: type 1               348
    Mobile                      317
    Legal Services              305
    Cleaning                    260
    Transport: type 1           201
    Industry: type 6            112
    Industry: type 10           109
    Religion                     85
    Industry: type 13            67
    Trade: type 4                64
    Trade: type 5                49
    Industry: type 8             24
    Name: ORGANIZATION_TYPE, dtype: int64




```python
#There are XNA values in df1 as well
df1["NAME_CONTRACT_TYPE"].value_counts()
```




    Cash loans         747553
    Consumer loans     729151
    Revolving loans    193164
    XNA                   346
    Name: NAME_CONTRACT_TYPE, dtype: int64




```python
#Just replace it to Cash Loans
df1.loc[df1["NAME_CONTRACT_TYPE"]=="XNA", 'NAME_CONTRACT_TYPE']="Cash loans"
df1["NAME_CONTRACT_TYPE"].value_counts()
```




    Cash loans         747899
    Consumer loans     729151
    Revolving loans    193164
    Name: NAME_CONTRACT_TYPE, dtype: int64




```python
#Now the NAME_CASH_LOAN_PURPOSE column
df1["NAME_CASH_LOAN_PURPOSE"].value_counts()
```




    XAP                                 922661
    XNA                                 677918
    Repairs                              23765
    Other                                15608
    Urgent needs                          8412
    Buying a used car                     2888
    Building a house or an annex          2693
    Everyday expenses                     2416
    Medicine                              2174
    Payments on other loans               1931
    Education                             1573
    Journey                               1239
    Purchase of electronic equipment      1061
    Buying a new car                      1012
    Wedding / gift / holiday               962
    Buying a home                          865
    Car repairs                            797
    Furniture                              749
    Buying a holiday home / land           533
    Business development                   426
    Gasification / water supply            300
    Buying a garage                        136
    Hobby                                   55
    Money for a third person                25
    Refusal to name the goal                15
    Name: NAME_CASH_LOAN_PURPOSE, dtype: int64




```python
#Dropping the XNA and XAP
df1=df1.drop(df1.loc[df1["NAME_CASH_LOAN_PURPOSE"]=="XNA"].index)
df1=df1.drop(df1.loc[df1["NAME_CASH_LOAN_PURPOSE"]=="XAP"].index)
df1["NAME_CASH_LOAN_PURPOSE"].value_counts()
```




    Repairs                             23765
    Other                               15608
    Urgent needs                         8412
    Buying a used car                    2888
    Building a house or an annex         2693
    Everyday expenses                    2416
    Medicine                             2174
    Payments on other loans              1931
    Education                            1573
    Journey                              1239
    Purchase of electronic equipment     1061
    Buying a new car                     1012
    Wedding / gift / holiday              962
    Buying a home                         865
    Car repairs                           797
    Furniture                             749
    Buying a holiday home / land          533
    Business development                  426
    Gasification / water supply           300
    Buying a garage                       136
    Hobby                                  55
    Money for a third person               25
    Refusal to name the goal               15
    Name: NAME_CASH_LOAN_PURPOSE, dtype: int64




```python
#Lastly for NAME_CLIENT_TYPE column
df1["NAME_CLIENT_TYPE"].value_counts()
```




    Repeater     56256
    New           9964
    Refreshed     3362
    XNA             53
    Name: NAME_CLIENT_TYPE, dtype: int64




```python
#Just replace it with Repeater
df1.loc[df1["NAME_CLIENT_TYPE"]=="XNA", "NAME_CLIENT_TYPE"]= "Repeater"
df1["NAME_CLIENT_TYPE"].value_counts()
```




    Repeater     56309
    New           9964
    Refreshed     3362
    Name: NAME_CLIENT_TYPE, dtype: int64



# Merging the Dataset


```python
df2 = pd.merge(df, df1, how="inner", on="SK_ID_CURR")
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE_x</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT_x</th>
      <th>AMT_ANNUITY</th>
      <th>...</th>
      <th>NAME_PAYMENT_TYPE</th>
      <th>CODE_REJECT_REASON</th>
      <th>NAME_CLIENT_TYPE</th>
      <th>NAME_GOODS_CATEGORY</th>
      <th>NAME_PORTFOLIO</th>
      <th>NAME_PRODUCT_TYPE</th>
      <th>CHANNEL_TYPE</th>
      <th>SELLERPLACE_AREA</th>
      <th>NAME_SELLER_INDUSTRY</th>
      <th>NAME_YIELD_GROUP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100034</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>90000.0</td>
      <td>180000.0</td>
      <td>9000.0</td>
      <td>...</td>
      <td>Cash through the bank</td>
      <td>XAP</td>
      <td>New</td>
      <td>XNA</td>
      <td>Cash</td>
      <td>walk-in</td>
      <td>Credit and cash offices</td>
      <td>1</td>
      <td>XNA</td>
      <td>high</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100035</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>292500.0</td>
      <td>665892.0</td>
      <td>24592.5</td>
      <td>...</td>
      <td>Cash through the bank</td>
      <td>HC</td>
      <td>Repeater</td>
      <td>XNA</td>
      <td>Cash</td>
      <td>walk-in</td>
      <td>Credit and cash offices</td>
      <td>1</td>
      <td>XNA</td>
      <td>low_action</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100039</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>N</td>
      <td>1</td>
      <td>360000.0</td>
      <td>733315.5</td>
      <td>39069.0</td>
      <td>...</td>
      <td>Cash through the bank</td>
      <td>XAP</td>
      <td>Refreshed</td>
      <td>XNA</td>
      <td>Cash</td>
      <td>walk-in</td>
      <td>Channel of corporate sales</td>
      <td>1</td>
      <td>XNA</td>
      <td>low_normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100046</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>180000.0</td>
      <td>540000.0</td>
      <td>27000.0</td>
      <td>...</td>
      <td>Cash through the bank</td>
      <td>XAP</td>
      <td>New</td>
      <td>XNA</td>
      <td>Cash</td>
      <td>walk-in</td>
      <td>Credit and cash offices</td>
      <td>1</td>
      <td>XNA</td>
      <td>low_normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100046</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>180000.0</td>
      <td>540000.0</td>
      <td>27000.0</td>
      <td>...</td>
      <td>Cash through the bank</td>
      <td>LIMIT</td>
      <td>Repeater</td>
      <td>XNA</td>
      <td>Cash</td>
      <td>walk-in</td>
      <td>Credit and cash offices</td>
      <td>1</td>
      <td>XNA</td>
      <td>low_normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 47 columns</p>
</div>



# Visualising Data


```python
#To continue divdide the dataset into two groups, target 1: clients with payment difficulties and target 0: all other
target0 = df2.loc[df2["TARGET"]==0]
target1 = df2.loc[df2["TARGET"]==1]
```


```python
plt.figure(figsize=[10,5])
sns.set_style("whitegrid")
sns.countplot(data=target0, x = "NAME_INCOME_TYPE", hue = "CODE_GENDER")
plt.title("Income Type by gender")
plt.xlabel("Income Type")
plt.yscale("log")
plt.show()
```


    
![png](output_37_0.png)
    



```python
sns.set_style("whitegrid")
sns.countplot(data=target0, x = "NAME_CONTRACT_TYPE_x", hue = "CODE_GENDER")
plt.title("Type of Contract by gender")
plt.xlabel("Contract Type")
plt.yscale("log")
plt.show()
```


    
![png](output_38_0.png)
    



```python
plt.figure(figsize=[10,5])
sns.set_style("whitegrid")
sns.countplot(data=target0, x = "ORGANIZATION_TYPE")
plt.title("Organization Type the Client Work In", fontsize = 20)
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_39_0.png)
    



```python
df2.NAME_CONTRACT_STATUS.value_counts(normalize=True).plot.pie()
plt.title("Distribution of Contract Status")
plt.show()
```


    
![png](output_40_0.png)
    



```python
plt.figure(figsize=[10,5])
sns.set_style("whitegrid")
sns.countplot(data=df2, x = "NAME_CASH_LOAN_PURPOSE", hue = "NAME_CONTRACT_STATUS")
plt.xticks(rotation=90)
plt.title("Contract Status by the Loan Purpose", fontsize = 20)
plt.yscale("log")
plt.show()
```


    
![png](output_41_0.png)
    


# Bivariate Analyss


```python
#Credit Amount to the Income Amount
plt.figure(figsize=[10,5])
sns.scatterplot(data=df2, x="AMT_CREDIT_x", y="AMT_INCOME_TOTAL", color="orange")
sns.regplot(data=df2, x="AMT_CREDIT_x", y="AMT_INCOME_TOTAL", color="green", line_kws={"color": "red"})
plt.title("Credit Amount by The Income Amount", fontsize=20)
plt.xlabel("Credit Amount")
plt.ylabel("Income Amount")
plt.show()
```


    
![png](output_43_0.png)
    



```python
#Credit Amount related to the Number of Children
plt.figure(figsize=[10,5])
sns.scatterplot(data=target0, x="AMT_CREDIT_x", y="CNT_CHILDREN", color="orange")
plt.title("Credit Amount by The Number of Children for Safe Clients", fontsize=20)
plt.xlabel("Credit Amount")
plt.ylabel("Number of Children")

plt.figure(figsize=[10,5])
sns.scatterplot(data=target1, x="AMT_CREDIT_x", y="CNT_CHILDREN", color="green")
plt.title("Credit Amount by The Number of Children for Risky Clients", fontsize=20)
plt.xlabel("Credit Amount")
plt.ylabel("Number of Children")
plt.show()
```


    
![png](output_44_0.png)
    



    
![png](output_44_1.png)
    



```python
#Credit Amount related to Income Type
plt.figure(figsize=[10,5])
sns.boxplot(data=target0, x="AMT_CREDIT_x", y="NAME_INCOME_TYPE", color="orange")
plt.title("Income Type of Safe Clients", fontsize=20)
plt.xlabel("Credit Amount")
plt.ylabel("Income Type")

plt.figure(figsize=[10,5])
sns.boxplot(data=target0, x="AMT_CREDIT_x", y="NAME_INCOME_TYPE", color="green")
plt.title("Income Type of Risky Clients", fontsize=20)
plt.xlabel("Credit Amount")
plt.ylabel("Income Type")
plt.show()
```


    
![png](output_45_0.png)
    



    
![png](output_45_1.png)
    



```python
#Annuity Amount related to Family Status
plt.figure(figsize=[10,5])
sns.boxplot(data=target0, x="AMT_ANNUITY", y="NAME_FAMILY_STATUS", color="purple")
plt.title("Family Status of Safe Clients", fontsize=20)
plt.xlabel("Annuity Amount")
plt.ylabel("Family Status")

plt.figure(figsize=[10,5])
sns.boxplot(data=target1, x="AMT_ANNUITY", y="NAME_FAMILY_STATUS", color="yellow")
plt.title("Family Status of Risky Clients", fontsize=20)
plt.xlabel("Annuity Amount")
plt.ylabel("Family Status")
plt.show()
```


    
![png](output_46_0.png)
    



    
![png](output_46_1.png)
    


# Finding Outliers for Both Target Groups


```python
#Target0 and 1 Total Income
sns.set_style("whitegrid")
sns.boxplot(data = target0, y = "AMT_INCOME_TOTAL", color = "pink")
plt.title("Distribution of Income of Safe Clients", fontsize = "20")
plt.yscale("log")
plt.ylabel("Income")
plt.show()

sns.set_style("whitegrid")
sns.boxplot(data = target1, y = "AMT_INCOME_TOTAL", color = "gold")
plt.title("Distribution of Income of Risky Clients", fontsize = "20")
plt.yscale("log")
plt.ylabel("Income")
plt.show()
```


    
![png](output_48_0.png)
    



    
![png](output_48_1.png)
    



```python
#Target0 and 1 AMT_CREDIT
sns.set_style("whitegrid")
sns.boxplot(data = target0, y = "AMT_CREDIT_x", color = "pink")
plt.title("Distribution of Credit Amount of Safe Clients", fontsize = "20")
plt.yscale("log")
plt.ylabel("Credit")
plt.show()

sns.set_style("whitegrid")
sns.boxplot(data = target1, y = "AMT_CREDIT_x", color = "gold")
plt.title("Distribution of Credit Amount of Risky Clients", fontsize = "20")
plt.yscale("log")
plt.ylabel("Credit")
plt.show()
```


    
![png](output_49_0.png)
    



    
![png](output_49_1.png)
    



```python
#Target0 and 1 AMT_ANNUITY
sns.set_style("whitegrid")
sns.boxplot(data = target0, y = "AMT_ANNUITY", color = "pink")
plt.title("Distribution of Annuity Amount of Safe Clients", fontsize = "20")
plt.yscale("log")
plt.ylabel("Credit")
plt.show()

sns.set_style("whitegrid")
sns.boxplot(data = target1, y = "AMT_ANNUITY", color = "gold")
plt.title("Distribution of Annuity Amount of Risky Clients", fontsize = "20")
plt.yscale("log")
plt.ylabel("Credit")
plt.show()
```


    
![png](output_50_0.png)
    



    
![png](output_50_1.png)
    


# Correlation


```python
#Target0 Correlation
target0_corr = target0.corr()
target0_corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT_x</th>
      <th>AMT_ANNUITY</th>
      <th>REGION_POPULATION_RELATIVE</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>DAYS_REGISTRATION</th>
      <th>...</th>
      <th>REGION_RATING_CLIENT_W_CITY</th>
      <th>HOUR_APPR_PROCESS_START_x</th>
      <th>DAYS_LAST_PHONE_CHANGE</th>
      <th>SK_ID_PREV</th>
      <th>AMT_APPLICATION</th>
      <th>AMT_CREDIT_y</th>
      <th>HOUR_APPR_PROCESS_START_y</th>
      <th>NFLAG_LAST_APPL_IN_DAY</th>
      <th>DAYS_DECISION</th>
      <th>SELLERPLACE_AREA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SK_ID_CURR</th>
      <td>1.000000</td>
      <td>NaN</td>
      <td>-0.006037</td>
      <td>0.017100</td>
      <td>0.012659</td>
      <td>0.009538</td>
      <td>0.006980</td>
      <td>0.008561</td>
      <td>-0.006665</td>
      <td>0.010790</td>
      <td>...</td>
      <td>-0.007112</td>
      <td>0.013980</td>
      <td>-0.018568</td>
      <td>-0.000422</td>
      <td>0.011175</td>
      <td>0.011376</td>
      <td>0.011370</td>
      <td>-0.005516</td>
      <td>0.007873</td>
      <td>-0.000588</td>
    </tr>
    <tr>
      <th>TARGET</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CNT_CHILDREN</th>
      <td>-0.006037</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>-0.010682</td>
      <td>-0.024786</td>
      <td>-0.002871</td>
      <td>-0.020406</td>
      <td>-0.302100</td>
      <td>-0.086167</td>
      <td>-0.177459</td>
      <td>...</td>
      <td>0.043048</td>
      <td>-0.028429</td>
      <td>0.018035</td>
      <td>0.000010</td>
      <td>0.012827</td>
      <td>0.015877</td>
      <td>-0.031366</td>
      <td>0.004116</td>
      <td>-0.055731</td>
      <td>0.007275</td>
    </tr>
    <tr>
      <th>AMT_INCOME_TOTAL</th>
      <td>0.017100</td>
      <td>NaN</td>
      <td>-0.010682</td>
      <td>1.000000</td>
      <td>0.360258</td>
      <td>0.431864</td>
      <td>0.198151</td>
      <td>0.062324</td>
      <td>0.038150</td>
      <td>-0.018723</td>
      <td>...</td>
      <td>-0.206652</td>
      <td>0.066602</td>
      <td>-0.086287</td>
      <td>-0.000695</td>
      <td>0.300513</td>
      <td>0.294700</td>
      <td>0.078394</td>
      <td>0.000021</td>
      <td>0.047859</td>
      <td>-0.018950</td>
    </tr>
    <tr>
      <th>AMT_CREDIT_x</th>
      <td>0.012659</td>
      <td>NaN</td>
      <td>-0.024786</td>
      <td>0.360258</td>
      <td>1.000000</td>
      <td>0.738256</td>
      <td>0.116404</td>
      <td>0.135468</td>
      <td>0.078451</td>
      <td>0.017591</td>
      <td>...</td>
      <td>-0.106219</td>
      <td>0.044392</td>
      <td>-0.086591</td>
      <td>0.006547</td>
      <td>0.227408</td>
      <td>0.222643</td>
      <td>0.045647</td>
      <td>0.009826</td>
      <td>0.096053</td>
      <td>-0.015793</td>
    </tr>
    <tr>
      <th>AMT_ANNUITY</th>
      <td>0.009538</td>
      <td>NaN</td>
      <td>-0.002871</td>
      <td>0.431864</td>
      <td>0.738256</td>
      <td>1.000000</td>
      <td>0.128786</td>
      <td>0.062634</td>
      <td>0.027694</td>
      <td>-0.027158</td>
      <td>...</td>
      <td>-0.133167</td>
      <td>0.039750</td>
      <td>-0.085515</td>
      <td>0.012823</td>
      <td>0.195222</td>
      <td>0.192036</td>
      <td>0.038970</td>
      <td>0.009241</td>
      <td>0.086239</td>
      <td>-0.014281</td>
    </tr>
    <tr>
      <th>REGION_POPULATION_RELATIVE</th>
      <td>0.006980</td>
      <td>NaN</td>
      <td>-0.020406</td>
      <td>0.198151</td>
      <td>0.116404</td>
      <td>0.128786</td>
      <td>1.000000</td>
      <td>0.064844</td>
      <td>0.021510</td>
      <td>0.064565</td>
      <td>...</td>
      <td>-0.537040</td>
      <td>0.174023</td>
      <td>-0.068674</td>
      <td>0.003431</td>
      <td>0.065479</td>
      <td>0.065071</td>
      <td>0.169277</td>
      <td>0.000758</td>
      <td>0.103395</td>
      <td>-0.022850</td>
    </tr>
    <tr>
      <th>DAYS_BIRTH</th>
      <td>0.008561</td>
      <td>NaN</td>
      <td>-0.302100</td>
      <td>0.062324</td>
      <td>0.135468</td>
      <td>0.062634</td>
      <td>0.064844</td>
      <td>1.000000</td>
      <td>0.318569</td>
      <td>0.288762</td>
      <td>...</td>
      <td>-0.069969</td>
      <td>-0.021160</td>
      <td>-0.105164</td>
      <td>-0.001963</td>
      <td>-0.010605</td>
      <td>-0.011982</td>
      <td>-0.029896</td>
      <td>-0.008040</td>
      <td>0.123949</td>
      <td>-0.015445</td>
    </tr>
    <tr>
      <th>DAYS_EMPLOYED</th>
      <td>-0.006665</td>
      <td>NaN</td>
      <td>-0.086167</td>
      <td>0.038150</td>
      <td>0.078451</td>
      <td>0.027694</td>
      <td>0.021510</td>
      <td>0.318569</td>
      <td>1.000000</td>
      <td>0.174780</td>
      <td>...</td>
      <td>-0.004916</td>
      <td>-0.009277</td>
      <td>-0.132179</td>
      <td>-0.000446</td>
      <td>-0.002668</td>
      <td>-0.004181</td>
      <td>-0.011223</td>
      <td>-0.008167</td>
      <td>0.097438</td>
      <td>-0.005402</td>
    </tr>
    <tr>
      <th>DAYS_REGISTRATION</th>
      <td>0.010790</td>
      <td>NaN</td>
      <td>-0.177459</td>
      <td>-0.018723</td>
      <td>0.017591</td>
      <td>-0.027158</td>
      <td>0.064565</td>
      <td>0.288762</td>
      <td>0.174780</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-0.111771</td>
      <td>0.048032</td>
      <td>-0.061664</td>
      <td>0.003921</td>
      <td>-0.060546</td>
      <td>-0.063287</td>
      <td>0.039791</td>
      <td>0.003057</td>
      <td>0.069947</td>
      <td>-0.007716</td>
    </tr>
    <tr>
      <th>DAYS_ID_PUBLISH</th>
      <td>0.004447</td>
      <td>NaN</td>
      <td>0.088893</td>
      <td>0.025948</td>
      <td>0.006830</td>
      <td>-0.004517</td>
      <td>0.009754</td>
      <td>0.065995</td>
      <td>0.054927</td>
      <td>0.030515</td>
      <td>...</td>
      <td>-0.003218</td>
      <td>0.017667</td>
      <td>-0.073171</td>
      <td>-0.007529</td>
      <td>-0.000199</td>
      <td>-0.001018</td>
      <td>0.007199</td>
      <td>0.000228</td>
      <td>0.048393</td>
      <td>-0.015637</td>
    </tr>
    <tr>
      <th>CNT_FAM_MEMBERS</th>
      <td>-0.001507</td>
      <td>NaN</td>
      <td>0.892873</td>
      <td>-0.011583</td>
      <td>0.015691</td>
      <td>0.038660</td>
      <td>-0.020130</td>
      <td>-0.265168</td>
      <td>-0.069015</td>
      <td>-0.171321</td>
      <td>...</td>
      <td>0.051381</td>
      <td>-0.028621</td>
      <td>0.007169</td>
      <td>0.001909</td>
      <td>0.036095</td>
      <td>0.038697</td>
      <td>-0.034260</td>
      <td>0.007430</td>
      <td>-0.045352</td>
      <td>0.005004</td>
    </tr>
    <tr>
      <th>REGION_RATING_CLIENT</th>
      <td>-0.009973</td>
      <td>NaN</td>
      <td>0.041297</td>
      <td>-0.191274</td>
      <td>-0.096344</td>
      <td>-0.119296</td>
      <td>-0.537818</td>
      <td>-0.072335</td>
      <td>-0.006448</td>
      <td>-0.117907</td>
      <td>...</td>
      <td>0.941626</td>
      <td>-0.281318</td>
      <td>0.041380</td>
      <td>-0.003539</td>
      <td>-0.025298</td>
      <td>-0.025817</td>
      <td>-0.275637</td>
      <td>0.008228</td>
      <td>-0.076832</td>
      <td>0.018604</td>
    </tr>
    <tr>
      <th>REGION_RATING_CLIENT_W_CITY</th>
      <td>-0.007112</td>
      <td>NaN</td>
      <td>0.043048</td>
      <td>-0.206652</td>
      <td>-0.106219</td>
      <td>-0.133167</td>
      <td>-0.537040</td>
      <td>-0.069969</td>
      <td>-0.004916</td>
      <td>-0.111771</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-0.257051</td>
      <td>0.039331</td>
      <td>-0.001864</td>
      <td>-0.039991</td>
      <td>-0.040421</td>
      <td>-0.253101</td>
      <td>0.010075</td>
      <td>-0.072618</td>
      <td>0.021053</td>
    </tr>
    <tr>
      <th>HOUR_APPR_PROCESS_START_x</th>
      <td>0.013980</td>
      <td>NaN</td>
      <td>-0.028429</td>
      <td>0.066602</td>
      <td>0.044392</td>
      <td>0.039750</td>
      <td>0.174023</td>
      <td>-0.021160</td>
      <td>-0.009277</td>
      <td>0.048032</td>
      <td>...</td>
      <td>-0.257051</td>
      <td>1.000000</td>
      <td>-0.047350</td>
      <td>-0.000166</td>
      <td>0.031927</td>
      <td>0.029598</td>
      <td>0.383850</td>
      <td>0.006703</td>
      <td>0.063301</td>
      <td>-0.011257</td>
    </tr>
    <tr>
      <th>DAYS_LAST_PHONE_CHANGE</th>
      <td>-0.018568</td>
      <td>NaN</td>
      <td>0.018035</td>
      <td>-0.086287</td>
      <td>-0.086591</td>
      <td>-0.085515</td>
      <td>-0.068674</td>
      <td>-0.105164</td>
      <td>-0.132179</td>
      <td>-0.061664</td>
      <td>...</td>
      <td>0.039331</td>
      <td>-0.047350</td>
      <td>1.000000</td>
      <td>-0.002848</td>
      <td>-0.076724</td>
      <td>-0.077038</td>
      <td>-0.028985</td>
      <td>-0.003917</td>
      <td>-0.141526</td>
      <td>0.011312</td>
    </tr>
    <tr>
      <th>SK_ID_PREV</th>
      <td>-0.000422</td>
      <td>NaN</td>
      <td>0.000010</td>
      <td>-0.000695</td>
      <td>0.006547</td>
      <td>0.012823</td>
      <td>0.003431</td>
      <td>-0.001963</td>
      <td>-0.000446</td>
      <td>0.003921</td>
      <td>...</td>
      <td>-0.001864</td>
      <td>-0.000166</td>
      <td>-0.002848</td>
      <td>1.000000</td>
      <td>0.011880</td>
      <td>0.012903</td>
      <td>0.002116</td>
      <td>0.004786</td>
      <td>-0.012313</td>
      <td>-0.008877</td>
    </tr>
    <tr>
      <th>AMT_APPLICATION</th>
      <td>0.011175</td>
      <td>NaN</td>
      <td>0.012827</td>
      <td>0.300513</td>
      <td>0.227408</td>
      <td>0.195222</td>
      <td>0.065479</td>
      <td>-0.010605</td>
      <td>-0.002668</td>
      <td>-0.060546</td>
      <td>...</td>
      <td>-0.039991</td>
      <td>0.031927</td>
      <td>-0.076724</td>
      <td>0.011880</td>
      <td>1.000000</td>
      <td>0.994932</td>
      <td>0.055685</td>
      <td>0.007184</td>
      <td>-0.212045</td>
      <td>-0.011573</td>
    </tr>
    <tr>
      <th>AMT_CREDIT_y</th>
      <td>0.011376</td>
      <td>NaN</td>
      <td>0.015877</td>
      <td>0.294700</td>
      <td>0.222643</td>
      <td>0.192036</td>
      <td>0.065071</td>
      <td>-0.011982</td>
      <td>-0.004181</td>
      <td>-0.063287</td>
      <td>...</td>
      <td>-0.040421</td>
      <td>0.029598</td>
      <td>-0.077038</td>
      <td>0.012903</td>
      <td>0.994932</td>
      <td>1.000000</td>
      <td>0.052930</td>
      <td>0.007219</td>
      <td>-0.219651</td>
      <td>-0.014721</td>
    </tr>
    <tr>
      <th>HOUR_APPR_PROCESS_START_y</th>
      <td>0.011370</td>
      <td>NaN</td>
      <td>-0.031366</td>
      <td>0.078394</td>
      <td>0.045647</td>
      <td>0.038970</td>
      <td>0.169277</td>
      <td>-0.029896</td>
      <td>-0.011223</td>
      <td>0.039791</td>
      <td>...</td>
      <td>-0.253101</td>
      <td>0.383850</td>
      <td>-0.028985</td>
      <td>0.002116</td>
      <td>0.055685</td>
      <td>0.052930</td>
      <td>1.000000</td>
      <td>-0.000886</td>
      <td>0.029068</td>
      <td>0.004616</td>
    </tr>
    <tr>
      <th>NFLAG_LAST_APPL_IN_DAY</th>
      <td>-0.005516</td>
      <td>NaN</td>
      <td>0.004116</td>
      <td>0.000021</td>
      <td>0.009826</td>
      <td>0.009241</td>
      <td>0.000758</td>
      <td>-0.008040</td>
      <td>-0.008167</td>
      <td>0.003057</td>
      <td>...</td>
      <td>0.010075</td>
      <td>0.006703</td>
      <td>-0.003917</td>
      <td>0.004786</td>
      <td>0.007184</td>
      <td>0.007219</td>
      <td>-0.000886</td>
      <td>1.000000</td>
      <td>0.001793</td>
      <td>0.000403</td>
    </tr>
    <tr>
      <th>DAYS_DECISION</th>
      <td>0.007873</td>
      <td>NaN</td>
      <td>-0.055731</td>
      <td>0.047859</td>
      <td>0.096053</td>
      <td>0.086239</td>
      <td>0.103395</td>
      <td>0.123949</td>
      <td>0.097438</td>
      <td>0.069947</td>
      <td>...</td>
      <td>-0.072618</td>
      <td>0.063301</td>
      <td>-0.141526</td>
      <td>-0.012313</td>
      <td>-0.212045</td>
      <td>-0.219651</td>
      <td>0.029068</td>
      <td>0.001793</td>
      <td>1.000000</td>
      <td>-0.005210</td>
    </tr>
    <tr>
      <th>SELLERPLACE_AREA</th>
      <td>-0.000588</td>
      <td>NaN</td>
      <td>0.007275</td>
      <td>-0.018950</td>
      <td>-0.015793</td>
      <td>-0.014281</td>
      <td>-0.022850</td>
      <td>-0.015445</td>
      <td>-0.005402</td>
      <td>-0.007716</td>
      <td>...</td>
      <td>0.021053</td>
      <td>-0.011257</td>
      <td>0.011312</td>
      <td>-0.008877</td>
      <td>-0.011573</td>
      <td>-0.014721</td>
      <td>0.004616</td>
      <td>0.000403</td>
      <td>-0.005210</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>23 rows × 23 columns</p>
</div>




```python
sns.heatmap(target0_corr, annot = False)
```




    <Axes: >




    
![png](output_53_1.png)
    



```python
#Target1 Correlation
target1_corr = target1.corr()
target1_corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT_x</th>
      <th>AMT_ANNUITY</th>
      <th>REGION_POPULATION_RELATIVE</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>DAYS_REGISTRATION</th>
      <th>...</th>
      <th>REGION_RATING_CLIENT_W_CITY</th>
      <th>HOUR_APPR_PROCESS_START_x</th>
      <th>DAYS_LAST_PHONE_CHANGE</th>
      <th>SK_ID_PREV</th>
      <th>AMT_APPLICATION</th>
      <th>AMT_CREDIT_y</th>
      <th>HOUR_APPR_PROCESS_START_y</th>
      <th>NFLAG_LAST_APPL_IN_DAY</th>
      <th>DAYS_DECISION</th>
      <th>SELLERPLACE_AREA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SK_ID_CURR</th>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.005126</td>
      <td>-0.037568</td>
      <td>-0.005566</td>
      <td>0.001634</td>
      <td>-0.008610</td>
      <td>0.020318</td>
      <td>0.016349</td>
      <td>-0.021894</td>
      <td>...</td>
      <td>-0.002909</td>
      <td>-0.023113</td>
      <td>-0.016405</td>
      <td>0.015419</td>
      <td>0.000108</td>
      <td>-0.000333</td>
      <td>-0.025633</td>
      <td>-0.012382</td>
      <td>-0.001449</td>
      <td>-0.003719</td>
    </tr>
    <tr>
      <th>TARGET</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CNT_CHILDREN</th>
      <td>0.005126</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>-0.035847</td>
      <td>-0.007188</td>
      <td>0.018410</td>
      <td>-0.010586</td>
      <td>-0.257941</td>
      <td>-0.063030</td>
      <td>-0.136036</td>
      <td>...</td>
      <td>0.037620</td>
      <td>-0.030181</td>
      <td>-0.005476</td>
      <td>0.011051</td>
      <td>0.002811</td>
      <td>0.006312</td>
      <td>0.012549</td>
      <td>0.003611</td>
      <td>-0.052248</td>
      <td>0.003352</td>
    </tr>
    <tr>
      <th>AMT_INCOME_TOTAL</th>
      <td>-0.037568</td>
      <td>NaN</td>
      <td>-0.035847</td>
      <td>1.000000</td>
      <td>0.324198</td>
      <td>0.423826</td>
      <td>0.151186</td>
      <td>0.123274</td>
      <td>0.023616</td>
      <td>0.017577</td>
      <td>...</td>
      <td>-0.230353</td>
      <td>0.083930</td>
      <td>-0.094849</td>
      <td>0.011133</td>
      <td>0.298302</td>
      <td>0.289874</td>
      <td>0.087790</td>
      <td>-0.002749</td>
      <td>0.083365</td>
      <td>-0.021814</td>
    </tr>
    <tr>
      <th>AMT_CREDIT_x</th>
      <td>-0.005566</td>
      <td>NaN</td>
      <td>-0.007188</td>
      <td>0.324198</td>
      <td>1.000000</td>
      <td>0.732868</td>
      <td>0.105101</td>
      <td>0.137818</td>
      <td>0.111457</td>
      <td>0.031843</td>
      <td>...</td>
      <td>-0.091791</td>
      <td>0.034573</td>
      <td>-0.127131</td>
      <td>-0.006904</td>
      <td>0.193733</td>
      <td>0.188154</td>
      <td>0.047090</td>
      <td>-0.028978</td>
      <td>0.138454</td>
      <td>-0.018208</td>
    </tr>
    <tr>
      <th>AMT_ANNUITY</th>
      <td>0.001634</td>
      <td>NaN</td>
      <td>0.018410</td>
      <td>0.423826</td>
      <td>0.732868</td>
      <td>1.000000</td>
      <td>0.104575</td>
      <td>0.052066</td>
      <td>0.025776</td>
      <td>-0.015355</td>
      <td>...</td>
      <td>-0.128208</td>
      <td>0.023707</td>
      <td>-0.102083</td>
      <td>0.009984</td>
      <td>0.186557</td>
      <td>0.181332</td>
      <td>0.050135</td>
      <td>-0.024029</td>
      <td>0.113740</td>
      <td>-0.023130</td>
    </tr>
    <tr>
      <th>REGION_POPULATION_RELATIVE</th>
      <td>-0.008610</td>
      <td>NaN</td>
      <td>-0.010586</td>
      <td>0.151186</td>
      <td>0.105101</td>
      <td>0.104575</td>
      <td>1.000000</td>
      <td>0.080304</td>
      <td>0.013207</td>
      <td>0.046034</td>
      <td>...</td>
      <td>-0.424015</td>
      <td>0.159362</td>
      <td>-0.062059</td>
      <td>0.006459</td>
      <td>0.076213</td>
      <td>0.075862</td>
      <td>0.160219</td>
      <td>0.004923</td>
      <td>0.104372</td>
      <td>-0.035194</td>
    </tr>
    <tr>
      <th>DAYS_BIRTH</th>
      <td>0.020318</td>
      <td>NaN</td>
      <td>-0.257941</td>
      <td>0.123274</td>
      <td>0.137818</td>
      <td>0.052066</td>
      <td>0.080304</td>
      <td>1.000000</td>
      <td>0.276295</td>
      <td>0.255613</td>
      <td>...</td>
      <td>-0.100126</td>
      <td>-0.013595</td>
      <td>-0.115923</td>
      <td>-0.014792</td>
      <td>0.029589</td>
      <td>0.027703</td>
      <td>-0.018615</td>
      <td>-0.018656</td>
      <td>0.104179</td>
      <td>-0.019812</td>
    </tr>
    <tr>
      <th>DAYS_EMPLOYED</th>
      <td>0.016349</td>
      <td>NaN</td>
      <td>-0.063030</td>
      <td>0.023616</td>
      <td>0.111457</td>
      <td>0.025776</td>
      <td>0.013207</td>
      <td>0.276295</td>
      <td>1.000000</td>
      <td>0.172965</td>
      <td>...</td>
      <td>-0.009525</td>
      <td>0.047234</td>
      <td>-0.155590</td>
      <td>-0.008203</td>
      <td>0.045446</td>
      <td>0.044613</td>
      <td>0.008730</td>
      <td>0.006787</td>
      <td>0.066107</td>
      <td>0.008442</td>
    </tr>
    <tr>
      <th>DAYS_REGISTRATION</th>
      <td>-0.021894</td>
      <td>NaN</td>
      <td>-0.136036</td>
      <td>0.017577</td>
      <td>0.031843</td>
      <td>-0.015355</td>
      <td>0.046034</td>
      <td>0.255613</td>
      <td>0.172965</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-0.091371</td>
      <td>0.073646</td>
      <td>-0.064022</td>
      <td>0.000520</td>
      <td>-0.033195</td>
      <td>-0.034732</td>
      <td>0.049536</td>
      <td>0.010652</td>
      <td>-0.004939</td>
      <td>-0.017652</td>
    </tr>
    <tr>
      <th>DAYS_ID_PUBLISH</th>
      <td>-0.009283</td>
      <td>NaN</td>
      <td>0.057739</td>
      <td>0.008427</td>
      <td>-0.007603</td>
      <td>-0.012948</td>
      <td>0.044455</td>
      <td>0.042564</td>
      <td>0.067601</td>
      <td>0.063571</td>
      <td>...</td>
      <td>-0.018145</td>
      <td>-0.005112</td>
      <td>-0.114507</td>
      <td>-0.018240</td>
      <td>0.051663</td>
      <td>0.048974</td>
      <td>0.007691</td>
      <td>-0.001412</td>
      <td>0.034881</td>
      <td>-0.033115</td>
    </tr>
    <tr>
      <th>CNT_FAM_MEMBERS</th>
      <td>0.019209</td>
      <td>NaN</td>
      <td>0.893205</td>
      <td>-0.042838</td>
      <td>0.017477</td>
      <td>0.040426</td>
      <td>-0.011657</td>
      <td>-0.208074</td>
      <td>-0.037991</td>
      <td>-0.132558</td>
      <td>...</td>
      <td>0.037450</td>
      <td>-0.033979</td>
      <td>-0.009525</td>
      <td>0.001505</td>
      <td>0.017420</td>
      <td>0.021665</td>
      <td>0.008789</td>
      <td>-0.003713</td>
      <td>-0.041978</td>
      <td>0.004327</td>
    </tr>
    <tr>
      <th>REGION_RATING_CLIENT</th>
      <td>-0.002429</td>
      <td>NaN</td>
      <td>0.030927</td>
      <td>-0.215437</td>
      <td>-0.075484</td>
      <td>-0.107471</td>
      <td>-0.423041</td>
      <td>-0.095563</td>
      <td>-0.007105</td>
      <td>-0.087238</td>
      <td>...</td>
      <td>0.960513</td>
      <td>-0.276499</td>
      <td>0.016294</td>
      <td>-0.003418</td>
      <td>-0.039510</td>
      <td>-0.037851</td>
      <td>-0.302257</td>
      <td>0.007859</td>
      <td>-0.075527</td>
      <td>0.031956</td>
    </tr>
    <tr>
      <th>REGION_RATING_CLIENT_W_CITY</th>
      <td>-0.002909</td>
      <td>NaN</td>
      <td>0.037620</td>
      <td>-0.230353</td>
      <td>-0.091791</td>
      <td>-0.128208</td>
      <td>-0.424015</td>
      <td>-0.100126</td>
      <td>-0.009525</td>
      <td>-0.091371</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-0.262221</td>
      <td>0.024399</td>
      <td>-0.004075</td>
      <td>-0.050389</td>
      <td>-0.048507</td>
      <td>-0.288778</td>
      <td>0.006904</td>
      <td>-0.081618</td>
      <td>0.033968</td>
    </tr>
    <tr>
      <th>HOUR_APPR_PROCESS_START_x</th>
      <td>-0.023113</td>
      <td>NaN</td>
      <td>-0.030181</td>
      <td>0.083930</td>
      <td>0.034573</td>
      <td>0.023707</td>
      <td>0.159362</td>
      <td>-0.013595</td>
      <td>0.047234</td>
      <td>0.073646</td>
      <td>...</td>
      <td>-0.262221</td>
      <td>1.000000</td>
      <td>-0.003108</td>
      <td>-0.005289</td>
      <td>0.052598</td>
      <td>0.053795</td>
      <td>0.387797</td>
      <td>0.012165</td>
      <td>0.033615</td>
      <td>-0.019984</td>
    </tr>
    <tr>
      <th>DAYS_LAST_PHONE_CHANGE</th>
      <td>-0.016405</td>
      <td>NaN</td>
      <td>-0.005476</td>
      <td>-0.094849</td>
      <td>-0.127131</td>
      <td>-0.102083</td>
      <td>-0.062059</td>
      <td>-0.115923</td>
      <td>-0.155590</td>
      <td>-0.064022</td>
      <td>...</td>
      <td>0.024399</td>
      <td>-0.003108</td>
      <td>1.000000</td>
      <td>-0.017658</td>
      <td>-0.134644</td>
      <td>-0.134132</td>
      <td>-0.017754</td>
      <td>-0.018681</td>
      <td>-0.132585</td>
      <td>-0.020934</td>
    </tr>
    <tr>
      <th>SK_ID_PREV</th>
      <td>0.015419</td>
      <td>NaN</td>
      <td>0.011051</td>
      <td>0.011133</td>
      <td>-0.006904</td>
      <td>0.009984</td>
      <td>0.006459</td>
      <td>-0.014792</td>
      <td>-0.008203</td>
      <td>0.000520</td>
      <td>...</td>
      <td>-0.004075</td>
      <td>-0.005289</td>
      <td>-0.017658</td>
      <td>1.000000</td>
      <td>0.027327</td>
      <td>0.027789</td>
      <td>0.007158</td>
      <td>0.000802</td>
      <td>-0.014186</td>
      <td>0.027034</td>
    </tr>
    <tr>
      <th>AMT_APPLICATION</th>
      <td>0.000108</td>
      <td>NaN</td>
      <td>0.002811</td>
      <td>0.298302</td>
      <td>0.193733</td>
      <td>0.186557</td>
      <td>0.076213</td>
      <td>0.029589</td>
      <td>0.045446</td>
      <td>-0.033195</td>
      <td>...</td>
      <td>-0.050389</td>
      <td>0.052598</td>
      <td>-0.134644</td>
      <td>0.027327</td>
      <td>1.000000</td>
      <td>0.994354</td>
      <td>0.044914</td>
      <td>0.007278</td>
      <td>-0.146108</td>
      <td>-0.005874</td>
    </tr>
    <tr>
      <th>AMT_CREDIT_y</th>
      <td>-0.000333</td>
      <td>NaN</td>
      <td>0.006312</td>
      <td>0.289874</td>
      <td>0.188154</td>
      <td>0.181332</td>
      <td>0.075862</td>
      <td>0.027703</td>
      <td>0.044613</td>
      <td>-0.034732</td>
      <td>...</td>
      <td>-0.048507</td>
      <td>0.053795</td>
      <td>-0.134132</td>
      <td>0.027789</td>
      <td>0.994354</td>
      <td>1.000000</td>
      <td>0.043203</td>
      <td>0.007307</td>
      <td>-0.154843</td>
      <td>-0.007787</td>
    </tr>
    <tr>
      <th>HOUR_APPR_PROCESS_START_y</th>
      <td>-0.025633</td>
      <td>NaN</td>
      <td>0.012549</td>
      <td>0.087790</td>
      <td>0.047090</td>
      <td>0.050135</td>
      <td>0.160219</td>
      <td>-0.018615</td>
      <td>0.008730</td>
      <td>0.049536</td>
      <td>...</td>
      <td>-0.288778</td>
      <td>0.387797</td>
      <td>-0.017754</td>
      <td>0.007158</td>
      <td>0.044914</td>
      <td>0.043203</td>
      <td>1.000000</td>
      <td>0.024207</td>
      <td>0.051651</td>
      <td>0.013954</td>
    </tr>
    <tr>
      <th>NFLAG_LAST_APPL_IN_DAY</th>
      <td>-0.012382</td>
      <td>NaN</td>
      <td>0.003611</td>
      <td>-0.002749</td>
      <td>-0.028978</td>
      <td>-0.024029</td>
      <td>0.004923</td>
      <td>-0.018656</td>
      <td>0.006787</td>
      <td>0.010652</td>
      <td>...</td>
      <td>0.006904</td>
      <td>0.012165</td>
      <td>-0.018681</td>
      <td>0.000802</td>
      <td>0.007278</td>
      <td>0.007307</td>
      <td>0.024207</td>
      <td>1.000000</td>
      <td>-0.001861</td>
      <td>0.001766</td>
    </tr>
    <tr>
      <th>DAYS_DECISION</th>
      <td>-0.001449</td>
      <td>NaN</td>
      <td>-0.052248</td>
      <td>0.083365</td>
      <td>0.138454</td>
      <td>0.113740</td>
      <td>0.104372</td>
      <td>0.104179</td>
      <td>0.066107</td>
      <td>-0.004939</td>
      <td>...</td>
      <td>-0.081618</td>
      <td>0.033615</td>
      <td>-0.132585</td>
      <td>-0.014186</td>
      <td>-0.146108</td>
      <td>-0.154843</td>
      <td>0.051651</td>
      <td>-0.001861</td>
      <td>1.000000</td>
      <td>0.003763</td>
    </tr>
    <tr>
      <th>SELLERPLACE_AREA</th>
      <td>-0.003719</td>
      <td>NaN</td>
      <td>0.003352</td>
      <td>-0.021814</td>
      <td>-0.018208</td>
      <td>-0.023130</td>
      <td>-0.035194</td>
      <td>-0.019812</td>
      <td>0.008442</td>
      <td>-0.017652</td>
      <td>...</td>
      <td>0.033968</td>
      <td>-0.019984</td>
      <td>-0.020934</td>
      <td>0.027034</td>
      <td>-0.005874</td>
      <td>-0.007787</td>
      <td>0.013954</td>
      <td>0.001766</td>
      <td>0.003763</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>23 rows × 23 columns</p>
</div>




```python
sns.heatmap(target1_corr, annot = False)
```




    <Axes: >




    
![png](output_55_1.png)
    



```python

```
