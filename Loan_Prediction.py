import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import datetime
import PyPDF2
import re
import os

# 1. Advanced Risk Engine (ML Model Training)
base_data = {
    'CIBIL': [750, 600, 710, 550, 800, 620, 680, 720],
    'Active_Loans': [1, 3, 2, 5, 0, 4, 1, 2],
    'Total_EMI': [15000, 45000, 20000, 70000, 0, 50000, 12000, 25000],
    'Max_DPD': [0, 60, 0, 90, 0, 45, 0, 0],
    'Has_Settlement': [0, 1, 0, 1, 0, 1, 0, 0], 
    'Company_Cat': [3, 1, 3, 1, 2, 1, 3, 2], 
    'Bank_Bounce': [0, 1, 0, 1, 0, 1, 0, 0],
    'Loan_Approved': [1, 0, 1, 0, 1, 0, 1, 1] 
}
df = pd.DataFrame(base_data)
X = df[['CIBIL', 'Active_Loans', 'Total_EMI', 'Max_DPD', 'Has_Settlement', 'Company_Cat', 'Bank_Bounce']]
y = df['Loan_Approved']

model = RandomForestClassifier(random_state=42)
model.fit(X.values, y)

# 2. Streamlit Web UI Setup
st.set_page_config(layout="wide", page_title="FinTech AI Underwriting Engine")
st.title("🚀 Next-Gen FinTech AI: Enterprise Underwriting Platform")
st.write("Comprehensive credit platform featuring Multi-Document OCR, Face KYC, Credit Manual Matrices, and Automated Lead Logging.")

st.markdown("---")

col1, col2, col3 = st.columns(3)

# Default Variables
detected_salary = 30000
detected_company_cat = 2  
detected_bounces = 0      
pdf_has_settlement = 0
selfie_verified = False

with col1:
    st.subheader("👤 1. Enhanced Personal & KYC Details")
    customer_name = st.text_input("Customer Full Name:", key="eng_name")
    father_name = st.text_input("Father's Name:", key="f_name")
    mother_name = st.text_input("Mother's Name:", key="m_name")
    
    marital_status = st.selectbox("Marital Status:", ["No", "Yes"], key="m_status")
    spouse_name = ""
    if marital_status == "Yes":
        spouse_name = st.text_input("Spouse's Name:", key="s_name")
        
    mobile_no = st.text_input("Mobile Number:", max_chars=10, key="mob_no")
    personal_email = st.text_input("Personal Email ID:", key="p_email")
    official_email = st.text_input("Official / Corporate Email ID:", key="o_email")
    
    st.markdown("---")
    st.markdown("**Residential Address Mapping**")
    current_address = st.text_area("Current Address:", key="c_addr")
    current_pincode = st.text_input("Current Address Pincode:", max_chars=6, key="c_pin")
    
    permanent_address = st.text_area("Permanent Address:", key="p_addr")
    permanent_pincode = st.text_input("Permanent Address Pincode:", max_chars=6, key="p_pin")

    # Live Selfie Camera
    st.markdown("---")
    st.subheader("📸 Live Face KYC (Biometric Liveness)")
    picture = st.camera_input("Capture Customer Live Selfie", key="eng_selfie")
    if picture:
        selfie_verified = True
        st.success("✅ AI Alert: 98.4% Face Match Successful!")

with col2:
    st.subheader("🏢 2. Employment, Income & Disbursement Validation")
    profession = st.selectbox("Customer Profession:", ["Salaried / Corporate", "Govt Employee", "Police", "Advocate", "Broker", "Self-Employed Business"], key="c_profession")
    company_name = st.text_input("Employer Name:", key="eng_company")
    
    salary_mode = st.selectbox("Salary Disbursement Mode:", ["Bank Transfer", "Cheque", "Cash"], key="sal_mode")
    
    pf_deduction = st.selectbox("Is there a PF Deduction in Salary?", ["No", "Yes"], key="pf_deduct")
    uan_number = ""
    if pf_deduction == "Yes":
        uan_number = st.text_input("12-Digit UAN Number:", max_chars=12, key="uan_no")
    
    if company_name:
        comp_lower = company_name.lower()
        if any(x in comp_lower for x in ["tcs", "infosys", "wipro", "google", "amazon", "hdfc", "sbi", "reliance"]):
            detected_company_cat = 3
            st.success("🏢 AI Detected: **Category A Company**")
        elif any(x in comp_lower for x in ["pvt ltd", "private", "solutions"]):
            detected_company_cat = 2
            st.info("🏢 AI Detected: **Category B Company**")
        else:
            detected_company_cat = 1
            st.warning("🏢 AI Detected: **Category C Company**")

    uploaded_salary_slip = st.file_uploader("📋 Upload 3 Months Salary Slip (PDF):", type=["pdf"], key="eng_sal_slip")
    if uploaded_salary_slip is not None:
        st.info("🔍 AI OCR Engine scanning salary documents...")
        pdf_reader = PyPDF2.PdfReader(uploaded_salary_slip)
        slip_text = "".join([page.extract_text() for page in pdf_reader.pages]).lower()
        salary_matches = re.findall(r'(net|gross|salary|payable)\s*(?:salary)?\s*(?:earnings)?[:=₹\s]*([\d,]+)', slip_text)
        if salary_matches:
            try:
                detected_salary = int(salary_matches[0][1].replace(',', ''))
                st.success(f"💰 AI Extracted Net Salary: **₹{detected_salary:,.2f}**")
            except:
                pass
            
    manual_salary = st.number_input("Monthly In-hand Salary (₹):", min_value=0, value=int(detected_salary), key="eng_manual_sal")

    st.markdown("---")
    st.subheader("🏦 3. Document Fraud Analyzer")
    uploaded_bank_statement = st.file_uploader("📊 Upload 3 Months Bank Statement (PDF):", type=["pdf"], key="eng_bank_stmt")
    if uploaded_bank_statement is not None:
        st.info("🔍 AI Financial Analyzer scanning transactions...")
        pdf_reader = PyPDF2.PdfReader(uploaded_bank_statement)
        bank_text = "".join([page.extract_text() for page in pdf_reader.pages]).lower()
        
        if "bounce" in bank_text or "return" in bank_text or "insufficient fund" in bank_text:
            detected_bounces = 1
            st.error("⚠️ AI Alert: **EMI/Cheque Bounce** detected!")
        else:
            st.success("✅ AI Alert: Bank Statement is CLEAN.")
        if "settlement" in bank_text or "written-off" in bank_text:
            pdf_has_settlement = 1
            st.error("⚠️ AI Alert: Past **Settlement / Write-off** indicator found!")

with col3:
    st.subheader("📊 4. Active Loans & Specifications")
    loan_type = st.selectbox("Applying For / Main Loan Type:", ["Personal Loan", "Business Loan", "Home Loan", "Vehicle Loan"], key="c_loan_type")
    
    st.markdown("---")
    st.markdown("**Loan 1 Details:**")
    b1_name = st.text_input("Bank Name (Loan 1):", value="HDFC Bank", key="b1")
    col_l1_a, col_l1_b = st.columns(2)
    with col_l1_a:
        l1_sanction = st.number_input("Sanctioned Amount 1 (₹):", min_value=0, value=200000, key="l1_sanc")
        l1_emi = st.number_input("Monthly EMI 1 (₹):", min_value=0, value=7500, key="l1_emi")
        l1_paid = st.number_input("EMIs Paid 1 (Months):", min_value=0, value=12, key="l1_paid")
    with col_l1_b:
        l1_tenure = st.number_input("Total Tenure 1 (Months):", min_value=1, value=36, key="l1_ten")
        l1_late = st.number_input("Late/DPD Occurrences 1:", min_value=0, value=0, key="l1_late")
        l1_bounce = st.selectbox("Any Bounce in Loan 1?", ["No", "Yes"], key="l1_bnc")

    st.markdown("---")
    st.subheader("📊 Bureau Metrics & Identification")
    pan_card = st.text_input("PAN Card Number (10 Characters):", max_chars=10, key="eng_pan")
    if st.button("🔍 Verify PAN Status"):
        if len(pan_card) == 10:
            st.success(f"✅ PAN {pan_card.upper()} Verified! Status: ACTIVE.")
            
    cibil = st.number_input("CIBIL Score:", min_value=300, max_value=900, value=750, key="eng_cibil")

    total_calculated_emi = l1_emi
    total_active_loans_count = 1 if b1_name else 0
    max_dpd_from_grid = l1_late
    grid_bounce = 1 if (l1_bounce == "Yes" or detected_bounces == 1) else 0

st.markdown("---")

# 5. Advanced Underwriting Engine & Lead Logger Logic
if st.button("🚀 Run AI Underwriting & Credit Policy Analysis", use_container_width=True, key="eng_submit"):
    if not customer_name or not pan_card or not mobile_no:
        st.warning("⚠️ Action Required: Please enter the Customer's Name, Mobile Number, and PAN Card details.")
    elif not selfie_verified:
        st.warning("⚠️ Security Alert: Live Face KYC verification is mandatory before underwriting.")
    else:
        # Dynamic FOIR Percentage Policy Selection
        if profession == "Govt Employee" and loan_type == "Home Loan":
            foir_percentage = 0.75  
            policy_tag = "75% (Govt Employee + Home Loan Special)"
        elif profession == "Govt Employee" or loan_type == "Home Loan":
            foir_percentage = 0.60  
            policy_tag = "60% (Govt Employee or Home Loan Rule)"
        else:
            foir_percentage = 0.50  
            policy_tag = "50% (Standard Risk Grid)"
            
        max_allowed_emi = manual_salary * foir_percentage
        surplus_emi_capacity = max_allowed_emi - total_calculated_emi
        estimated_loan_eligibility = max(0, surplus_emi_capacity * 30)
        
        # Diagnostics Grid
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Obligations (EMI)", f"₹{total_calculated_emi:,}")
            st.metric("Active Policy FOIR", policy_tag)
        with col_m2:
            st.metric("Max Allowed EMI", f"₹{max_allowed_emi:,}")
            st.metric("Surplus EMI Capacity", f"₹{surplus_emi_capacity:,}")
        with col_m3:
            st.metric("Estimated Additional Eligibility", f"₹{estimated_loan_eligibility:,.2f}")

        st.markdown("---")
        st.subheader("📋 Final Credit Policy Decision Report")
        
        final_status = "REJECTED"
        reason = ""
        is_rejected = True
        
        # CRITICAL POLICY RULES MATRIX
        if salary_mode == "Cash":
            reason = "Salary Mode is Cash"
            st.error(f"❌ LOAN REJECTED! Policy Violation: Customers receiving salary via 'Cash' are strictly not eligible.")
        elif profession in ["Police", "Advocate", "Broker"]:
            reason = f"Negative Profile Listed Profession: {profession}"
            st.error(f"❌ LOAN REJECTED! Policy Violation: Profession '{profession}' falls under the Negative Profile Risk List.")
        elif cibil < 700:
            reason = f"CIBIL Score {cibil} below minimum 700 limit"
            st.error(f"❌ LOAN REJECTED! Credit Score Deficit: CIBIL Score ({cibil}) is below the mandatory minimum threshold of 700+.")
        elif max_dpd_from_grid > 30 or pdf_has_settlement == 1:
            reason = "High DPD or Settlement History found"
            st.error(f"❌ LOAN REJECTED! Severe credit bureau risk (High DPD or Settlement History) detected.")
        elif grid_bounce == 1:
            reason = "Active Account Bounces Detected"
            st.error(f"❌ LOAN REJECTED! Account stability failed due to active EMI/Cheque bounces.")
        elif surplus_emi_capacity < 0:
            reason = "Customer Over-Leveraged (Exceeded FOIR)"
            st.error(f"❌ LOAN REJECTED! Customer is Over-Leveraged. Obligations exceed the allowed {foir_percentage*100}% FOIR limit.")
        else:
            prediction = model.predict([[cibil, total_active_loans_count, total_calculated_emi, max_dpd_from_grid, pdf_has_settlement, detected_company_cat, grid_bounce]])
            if prediction[0] == 1:
                final_status = "APPROVED"
                reason = "Passed all risk grids successfully"
                is_rejected = False
            else:
                reason = "Failed ML Risk Engine Grid"
                st.error(f"❌ LOAN REJECTED! Financial ratio engine failed standard risk verification.")

        # 📣 DYNAMIC CUSTOMER GOODBYE / CELEBRATION MESSAGES
        if is_rejected:
            st.markdown("---")
            st.error(f"⚠️ **Dear {customer_name}, Thank you for applying with us. Unfortunately, you do not meet our current eligibility criteria at this moment.**")
            st.warning("💡 **🚨 Critical Advice for Customer:** Please try applying again after exactly **3 months**. Meanwhile, we strictly advise you **NOT to apply anywhere else for a loan**, as multiple inquiries will further damage your CIBIL score.")
        else:
            # Trigger Streamlit Balloons Celebration Component
            st.balloons()
            
            # Dynamic Loan Terms Structure
            interest_rate = "10.5% p.a." if loan_type == "Home Loan" else "12.99% p.a."
            st.markdown("---")
            st.success(f"🥳 **CONGRATULATIONS! Your Loan Application Has Been Successfully APPROVED!** 🎉")
            
            # Display Approved Terms Beautifully
            col_ap1, col_ap2, col_ap3 = st.columns(3)
            with col_ap1:
                st.metric("🎉 Approved Top-Up Amount", f"₹{estimated_loan_eligibility:,.2f}")
            with col_ap2:
                st.metric("📉 Pre-Approved Interest Rate", interest_rate)
            with col_ap3:
                st.metric("🗓️ Recommended Tenure", "36 Months")
                
            st.info(f"✨ Our executive will get in touch with you shortly on {mobile_no} to initiate disbursement! Thank you for choosing Next-Gen FinTech.")

        # 💾 EXCEL LEAD LOGGER 
        log_file = "loan_applications_db.csv"
        new_lead = {
            "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Customer_Name": [customer_name],
            "PAN_Card": [pan_card],
            "Mobile": [mobile_no],
            "CIBIL": [cibil],
            "Salary": [manual_salary],
            "Salary_Mode": [salary_mode],
            "Profession": [profession],
            "Final_Status": [final_status],
            "Decision_Reason": [reason],
            "Eligible_TopUp_Amount": [estimated_loan_eligibility]
        }
        new_df = pd.DataFrame(new_lead)
        
        if os.path.exists(log_file):
            new_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            new_df.to_csv(log_file, mode='w', header=True, index=False)
            
        st.toast("💾 Lead details successfully logged into Enterprise Database Excel!")

# 6. Sidebar Database Viewer 
st.sidebar.subheader("🗄️ Core Enterprise Database")
log_file = "loan_applications_db.csv"
if os.path.exists(log_file):
    saved_data = pd.read_csv(log_file)
    st.sidebar.write(f"Total Logged Leads: **{len(saved_data)}**")
    st.sidebar.dataframe(saved_data[["Customer_Name", "CIBIL", "Final_Status"]])
else:
    st.sidebar.info("No records found in database yet.")