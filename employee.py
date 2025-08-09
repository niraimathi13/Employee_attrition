import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,roc_curve,classification_report
st.set_page_config(page_title="Employee Attrition Prediction",layout="wide")
st.title("💼 Employee Attrition Prediction")
st.markdown("Predict whether an employee is likely to leave or stay in the company.")
@st.cache_resource
def load_pipeline():
    pipeline=joblib.load(r"C:\Users\nirai\Downloads\Python codes\model_pipeline.pkl")
    return pipeline
pipeline=load_pipeline()
@st.cache_data
def load_test_data():
    test_df=pd.read_csv(r"C:\Users\nirai\Downloads\Python codes\employee_attrition_test.csv")
    if test_df['Attrition'].dtype != 'int64':
        test_df['Attrition']=test_df['Attrition'].map({'Yes': 1, 'No': 0})
    return test_df
test_df=load_test_data()
st.subheader("📊 Model Performance on Test Dataset")
feature_cols = ["Age","BusinessTravel","DailyRate","Department","DistanceFromHome","Education","EducationField","EnvironmentSatisfaction","Gender","HourlyRate","JobInvolvement","JobLevel","JobRole","JobSatisfaction","MaritalStatus","MonthlyIncome","NumCompaniesWorked","OverTime","PercentSalaryHike","PerformanceRating","RelationshipSatisfaction","StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"]
X_test=test_df[feature_cols]
y_test=test_df['Attrition']
y_pred=pipeline.predict(X_test)
y_proba=pipeline.predict_proba(X_test)[:, 1]
acc=accuracy_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
rec=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
roc_auc=roc_auc_score(y_test,y_proba)
cm=confusion_matrix(y_test,y_pred)
class_report=classification_report(y_test,y_pred,output_dict=True)
st.write(f"**Accuracy:** {acc:.4f}")
st.write(f"**Precision:** {prec:.4f}")
st.write(f"**Recall:** {rec:.4f}")
st.write(f"**F1 Score:** {f1:.4f}")
st.write(f"**ROC-AUC:** {roc_auc:.4f}")
# Classification report table
st.write("### Classification Report")
st.dataframe(pd.DataFrame(class_report).transpose())
# Confusion matrix heatmap
fig_cm,ax_cm=plt.subplots()
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)
# ROC curve plot
fpr,tpr,_=roc_curve(y_test,y_proba)
fig_roc,ax_roc=plt.subplots()
ax_roc.plot(fpr,tpr,label=f"AUC = {roc_auc:.2f}")
ax_roc.plot([0, 1], [0, 1],'k--')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend()
st.pyplot(fig_roc)
st.subheader("🔍 Predict Attrition")
def user_input_features():
    Age=st.slider("Age", 18, 60, 30)
    BusinessTravel=st.selectbox("Business Travel",["Non-Travel","Travel_Rarely","Travel_Frequently"])
    DailyRate=st.slider("Daily Rate",100,1500,800)
    Department=st.selectbox("Department",["Sales","Research & Development","Human Resources"])
    DistanceFromHome=st.slider("Distance From Home (km)",1,30,5)
    Education=st.selectbox("Education (1-Below College,2-College,3-Bachelor,4-Master,5-Doctor)",[1, 2, 3, 4, 5])
    EducationField=st.selectbox("Education Field",["Life Sciences","Other","Medical","Marketing","Technical Degree","Human Resources"])
    EnvironmentSatisfaction=st.selectbox("Environment Satisfaction (1-Low,2-Medium,3-High,4-Very High)",[1, 2, 3, 4])
    Gender=st.selectbox("Gender",["Male","Female"])
    HourlyRate=st.slider("Hourly Rate",30,100,50)
    JobInvolvement=st.selectbox("Job Involvement (1-Low,2-Medium,3-High,4-Very High)",[1, 2, 3, 4])
    JobLevel=st.slider("Job Level",1,5,2)
    JobRole=st.selectbox("Job Role", ["Sales Executive","Research Scientist","Laboratory Technician","Manufacturing Director","Healthcare Representative","Manager","Sales Representative","Research Director","Human Resources"])
    JobSatisfaction=st.selectbox("Job Satisfaction (1-Low,2-Medium,3-High,4-Very High)",[1, 2, 3, 4])
    MaritalStatus=st.selectbox("Marital Status",["Single","Married","Divorced"])
    MonthlyIncome=st.slider("Monthly Income",1000,20000,5000)
    NumCompaniesWorked=st.slider("Number of Companies Worked",0,10,2)
    OverTime=st.selectbox("OverTime", ["Yes","No"])
    PercentSalaryHike=st.slider("Percent Salary Hike",10,25,15)
    PerformanceRating=st.selectbox("Performance Rating",[1, 2, 3, 4])
    RelationshipSatisfaction=st.selectbox("Relationship Satisfaction (1-Low,2-Medium,3-High,4-Very High)",[1, 2, 3, 4])
    StockOptionLevel=st.selectbox("Stock Option Level",[0, 1, 2, 3])
    TotalWorkingYears=st.slider("Total Working Years",0,40,10)
    TrainingTimesLastYear=st.slider("Training Times Last Year",0,10,2)
    WorkLifeBalance=st.selectbox("Work Life Balance (1-Bad,2-Good,3-Better,4-Best)",[1, 2, 3, 4])
    YearsAtCompany=st.slider("Years At Company",0,40,5)
    YearsInCurrentRole=st.slider("Years In Current Role",0,18,3)
    YearsSinceLastPromotion=st.slider("Years Since Last Promotion",0,15,1)
    YearsWithCurrManager=st.slider("Years With Current Manager",0,17,3)
    data={
        "Age": Age,
        "BusinessTravel": BusinessTravel,
        "DailyRate": DailyRate,
        "Department": Department,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EducationField": EducationField,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "Gender": Gender,
        "HourlyRate": HourlyRate,
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobRole": JobRole,
        "JobSatisfaction": JobSatisfaction,
        "MaritalStatus": MaritalStatus,
        "MonthlyIncome": MonthlyIncome,
        "NumCompaniesWorked": NumCompaniesWorked,
        "OverTime": OverTime,
        "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager
    }
    features=pd.DataFrame(data,index=[0])
    return features
input_df=user_input_features()
if st.button("Predict Attrition"):
    pred_prob=pipeline.predict_proba(input_df)[0][1]
    pred_class=pipeline.predict(input_df)[0]
    st.write(f"**Attrition Probability:** {pred_prob:.2f}")
    if pred_class == 1:
        st.error("⚠️ This employee is likely to leave.")
    else:
        st.success("✅ This employee is likely to stay.")
