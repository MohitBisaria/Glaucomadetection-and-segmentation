import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from torch import nn
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import joblib
import streamlit_authenticator as stauth
from fpdf import FPDF  # For PDF generation
import tempfile      # For handling temporary files

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide")


# --- PDF GENERATION CLASS AND FUNCTION ---
class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'Glaucoma Screening Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_report_pdf(patient_info, original_img, overlay_img, metrics_df, metrics_fig):
    pdf = PDF('P', 'mm', 'A4')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Patient Details ---
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Patient Details', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 11)
    for key, value in patient_info.items():
        pdf.cell(40, 8, f"{key}:", 0, 0)
        pdf.cell(0, 8, str(value), 0, 1)
    pdf.ln(10)

    # --- Images ---
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Screening Images', 0, 1, 'L')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_orig, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_overlay:
        cv2.imwrite(tmp_orig.name, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(tmp_overlay.name, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        pdf.image(tmp_orig.name, x=15, w=80, type='PNG')
        pdf.image(tmp_overlay.name, x=110, w=80, type='PNG')
    
    pdf.set_font('Helvetica', 'I', 10)
    pdf.text(x=45, y=pdf.get_y() + 65, txt='Original Image')
    pdf.text(x=135, y=pdf.get_y() + 65, txt='Segmented Overlay')
    pdf.ln(75)

    # --- Analysis Results Table ---
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Analysis Results', 0, 1, 'L')
    
    pdf.set_font('Helvetica', 'B', 10)
    col_widths = [25, 25, 25, 25, 35, 30]
    headers = list(metrics_df.columns)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
    pdf.ln()
    
    pdf.set_font('Helvetica', '', 10)
    for _, row in metrics_df.iterrows():
        for i, header in enumerate(headers):
            value_to_display = f"{row[header]:.3f}" if header == 'Confidence' and isinstance(row[header], float) else str(row[header])
            pdf.cell(col_widths[i], 10, value_to_display, 1, 0, 'C')
    pdf.ln(15)

    # --- Bar Chart ---
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Glaucoma-Specific Metrics Chart', 0, 1, 'L')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart:
        if metrics_fig:
            metrics_fig.write_image(tmp_chart.name, scale=2)
            pdf.image(tmp_chart.name, w=180, type='PNG')

    return pdf.output(dest='S').encode('latin-1')


# --- USER AUTHENTICATION CONFIG ---
config = {
    'credentials': {
        'usernames': {
            'testuser': {
                'email': 'test@user.com',
                'name': 'Test User',
                'password': '$2b$12$pMQfhnxFyeKAUJ6IYOBsC.LU/RRQELL9jrpfa3o6j3U39GnaQj4oy' # Hashed password for 'password123'
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'a_random_secret_key_for_this_app',
        'name': 'glaucoma_app_cookie'
    }
}

# --- Initialize the Authenticator ---
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- LOGIN/ABOUT/CONTACTS PAGE LOGIC ---
if not st.session_state.get("authentication_status"):
    if 'page_view' not in st.session_state:
        st.session_state.page_view = 'login'

    st.markdown("""
        <style>
            .block-container { padding: 2rem 5rem 2rem 5rem !important; }
            hr { margin-top: 0 !important; margin-bottom: 2rem !important; }
            div[data-testid="stForm"] label { font-size: 0px !important; }
            div[data-testid="stTextInput"] label[for*="Username"]::before {
                content: "Email Address/LoginID"; font-size: 1.1rem !important; font-weight: bold; color: #00008B;
            }
            div[data-testid="stTextInput"] label[for*="Password"]::before {
                content: "Password"; font-size: 1.1rem !important; font-weight: bold; color: #00008B;
            }
            input[type="text"], input[type="password"] {
                border: 2px solid #00008B !important; border-radius: 10px !important; height: 50px !important;
            }
            div[data-testid="stImage"] > img { display: block; margin-left: auto; margin-right: auto; }
        </style>
    """, unsafe_allow_html=True)

    nav1, nav2, nav3, _ = st.columns([0.15, 0.25, 0.3, 0.3])
    with nav1:
        button_type = "primary" if st.session_state.page_view == 'login' else "secondary"
        if st.button("Home Page", type=button_type, use_container_width=True):
            st.session_state.page_view = 'login'; st.rerun()
    with nav2:
        button_type = "primary" if st.session_state.page_view == 'about' else "secondary"
        if st.button("About the Project", type=button_type, use_container_width=True):
            st.session_state.page_view = 'about'; st.rerun()
    with nav3:
        button_type = "primary" if st.session_state.page_view == 'contacts' else "secondary"
        if st.button("Acknowledgement & Contacts", type=button_type, use_container_width=True):
            st.session_state.page_view = 'contacts'; st.rerun()

    st.write("") 

    if st.session_state.page_view == 'login':
        header_cols = st.columns([1, 2])
        with header_cols[0]:
            try: st.image("image.png")
            except FileNotFoundError: st.error("Logo file 'logo_main.png' not found.")
        with header_cols[1]:
            st.markdown("<h1 style='text-align: center; color: #00008B;'>Glaucoma Screening from Retinal Fundus Images</h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        body_cols = st.columns([1, 1])
        with body_cols[0]:
            authenticator.login()
            if st.session_state.get("authentication_status") is False:
                st.error('Username/password is incorrect')
        with body_cols[1]:
            st.markdown("<h3 style='text-align: center;'>Why Glaucoma is Serious – Some Facts</h3>", unsafe_allow_html=True)
            info_cols = st.columns([1, 2])
            with info_cols[0]:
                try: st.image("glaucoma.png")
                except FileNotFoundError: st.error("Diagram file 'diagram_eye.png' not found.")
            with info_cols[1]:
                st.markdown("""
                **Blindness Can Be Prevented By Following Doctor/Ophthalmologist Instructions**
                - Glaucoma is like diabetes or hypertension, no pain or symptoms and it can’t be cured, but regular medication can keep it in control.
                - There about 12 Million people with Glaucoma in India. Only half of them aware of it.
                - For every person diagnosed to have Glaucoma, there is another person with undetected Glaucoma.
                - Many people don’t know they have Glaucoma, until they start to lose 50% of their eye sight, gradually however the doctor can detect and treat Glaucoma before most patients experience any symptoms.
                - Patients with glaucoma usually have less field of vision (total area of sight) when they have glaucoma and when they have lost all of the visual field, they are prone to blindness.
                - In Glaucoma all efforts are aimed to preserve the existing vision of a person.
                - Glaucoma is hereditary. All patients with Glaucoma should inform their family members to get screened for Glaucoma
                """)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        footer_cols = st.columns(3)
        with footer_cols[0]:
            st.markdown("<h4 style='text-align: center;'>Funding Support</h4>", unsafe_allow_html=True)
            logo_cols = st.columns(2)
            with logo_cols[0]:
                try: st.image("hub.png")
                except FileNotFoundError: st.error("File 'logo_funding_hub.png' not found.")
            with logo_cols[1]:
                try: st.image("money.png")
                except FileNotFoundError: st.error("File 'logo_funding_dst.png' not found.")
        with footer_cols[1]:
            st.markdown("<h4 style='text-align: center;'>Project Development & Execution</h4>", unsafe_allow_html=True)
            try: st.image("mahindra university.png")
            except FileNotFoundError: st.error("File 'logo_execution_mu.png' not found.")
        with footer_cols[2]:
            st.markdown("<h4 style='text-align: center;'>Support for Data Collection</h4>", unsafe_allow_html=True)
            try: st.image("government of telangna.png")
            except FileNotFoundError: st.error("File 'logo_data_telangana.png' not found.")

    elif st.session_state.page_view == 'about':
        st.markdown("<h3 style='color: red;'>About the Project Page</h3>", unsafe_allow_html=True)
        st.markdown("""
        - **GlauMitra AI** is an initiative driven by the need for advanced tools to enable early and accurate diagnosis of glaucoma in Indian patients, particularly in resource-limited settings, to ensure timely referral to ophthalmologists.
        - Early and accurate diagnosis of glaucoma is critical for effective treatment and the prevention of irreversible blindness. With this goal, we developed GlauMitra AI—an advanced artificial intelligence system designed to automatically detect early signs of glaucoma from retinal fundus images of Indian patients, specifically from Telangana State, using both conventional and handheld fundus cameras.
        - To build this system, we employed state-of-the-art AI and image processing techniques. Our model was trained on a self-curated dataset of approximately 15,000 fundus images—comprising both Glaucoma and Non-Glaucoma cases—collected from patients in the Nalgonda and Hyderabad districts. The dataset was split 80:20 for training and testing purposes, and the resulting system achieved an impressive 90% classification accuracy.
        - We envision **GlauMitra AI** as a valuable screening tool for early glaucoma detection, especially in resource-limited settings, enabling timely referrals to ophthalmologists and improving patient outcomes.
        """)
    
    elif st.session_state.page_view == 'contacts':
        st.markdown("<h3 style='color: red;'>Acknowledgement & Contacts Page</h3>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #00008B;'>Contact Information</h4>", unsafe_allow_html=True)
        st.markdown("Feel free to reach out to us with any inquiries or feedback.")
        
        # --- [CORRECTED] FULL HTML FOR THE CONTACTS TABLE ---
        contact_table = """
        <table style="width:100%; border-collapse: collapse;">
            <tr style="border: 1px solid #ddd;">
                <th style="padding: 8px; text-align: left; background-color: #f2f2f2; border: 1px solid #ddd;">Project Investigator and Co-Project Investigators</th>
                <th style="padding: 8px; text-align: left; background-color: #f2f2f2; border: 1px solid #ddd;">E-mail</th>
            </tr>
            <tr style="border: 1px solid #ddd;">
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;">Dr. Bipin Singh (PI)<br>Assistant Professor, Centre for Life Sciences, Mahindra University,<br>Hyderabad, Telangana</td>
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;">bipin.singh@mahindrauniversity.edu.in</td>
            </tr>
            <tr style="border: 1px solid #ddd;">
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;">Dr. Santosh Thakur (Co-PI)<br>Assistant Professor, Centre for Life Sciences, Mahindra University,<br>Hyderabad, Telangana</td>
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;">santosh.thakur@mahindrauniversity.edu.in</td>
            </tr>
            <tr style="border: 1px solid #ddd;">
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;">Dr. Superna Mahendra (Collaborator)<br>Civil Surgeon Ophthalmologist, Government General Hospital,<br>Nalgonda, Telangana</td>
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;">drsuperna95@gmail.com</td>
            </tr>
            <tr style="border: 1px solid #ddd;">
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;">Mr. Mohit Bisaria<br>Senior Research Fellow, Centre for Life Sciences, Mahindra University,<br>Hyderabad, Telangana</td>
                <td rowspan-2 style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;"></td>
            </tr>
            <tr style="border: 1px solid #ddd;">
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #e6f5e6;">Mr. Sujal Shinde<br>BTech Final Year Student<br>Research Intern, Centre for Life Sciences, Mahindra University,<br>Hyderabad</td>
            </tr>
        </table>
        """
        st.markdown(contact_table, unsafe_allow_html=True)
        
        st.markdown("<br><h4 style='color: #00008B;'>Acknowledgements</h4>", unsafe_allow_html=True)
        
        # --- [CORRECTED] FULL LIST OF ACKNOWLEDGEMENTS ---
        st.markdown("""
        - Mr. Komatireddy Venkat Reddy, Minister of Roads, Buildings and Cinematography,Government of Telangana.
        - Mr. Jayesh Ranjan, Special Chief Secretary, Government of Telangana.
        - Mr. Bhavesh Mishra, Deputy Secretary, Government of Telangana.
        - Ms. Ila Tripathi, Collector, Nalgonda, Telangana.
        - Dr. G Ranjit Kumar, Ophthalmologist, GRK Visual Fields, Hyderabad, Telangana.
        """)

# --- MAIN APPLICATION (Runs only after successful login) ---
elif st.session_state["authentication_status"]:

    header_cols = st.columns([1.5, 3, 1])
    with header_cols[0]:
        st.markdown("<p style='color: red; font-size: 1.2rem; font-weight: bold;'>Prediction Page</p>", unsafe_allow_html=True)
        try: st.image("image.png", width=300)
        except FileNotFoundError: st.error("Logo 'logo_main.png' not found.")
    with header_cols[1]:
        st.markdown("""
            <div style='background-color: #f0f2f6; border-radius: 10px; padding: 1rem; margin-top: 2rem; display: flex; align-items: center; justify-content: center; height: 100px;'>
                <h2 style='color: #0d6efd; text-align: center; font-weight: bold;'>Glaucoma Screening from Retinal Fundus Images</h2>
            </div>
        """, unsafe_allow_html=True)
    with header_cols[2]:
        st.markdown(f"<p style='text-align: right; margin-top: 2rem;'>Welcome <i>{st.session_state['name']}</i></p>", unsafe_allow_html=True)
        authenticator.logout('Logout', 'main')

    st.markdown("---")

    st.markdown("##### Patient Details")
    row1_cols = st.columns(2)
    with row1_cols[0]:
        patient_name = st.text_input("Patient Name")
    with row1_cols[1]:
        age = st.selectbox("Age", options=list(range(1, 101)))
    row2_cols = st.columns(2)
    with row2_cols[0]:
        gender = st.selectbox("Gender", options=["Male", "Female", "Others"])
    with row2_cols[1]:
        comorbidities = st.multiselect("Comorbidities", ["BP", "Diabetes"], placeholder="Select (optional)")

    st.markdown("---")
    st.markdown("##### Upload a retinal image")
    uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    @st.cache_resource
    def load_models():
        processor = AutoImageProcessor.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")
        model = SegformerForSemanticSegmentation.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")
        model.eval()
        try:
            clf = joblib.load("random_forest_pipeline_80_20_4.pkl")
        except FileNotFoundError:
            st.error('Classifier model file `random_forest_pipeline_80_20_4.pkl` not found.')
            clf = None
        return processor, model, clf

    processor, model, clf = load_models()

    def vertical_cdr(disc_mask, cup_mask):
        y_disc, y_cup = np.where(disc_mask)[0], np.where(cup_mask)[0]
        if not (len(y_disc) and len(y_cup)): return None
        return round((y_cup.max() - y_cup.min() + 1) / (y_disc.max() - y_disc.min() + 1), 3)

    def acdr_area(disc_mask, cup_mask):
        disc_sum = np.sum(disc_mask)
        if disc_sum == 0: return None
        return round(np.sum(cup_mask) / disc_sum, 3)

    def horizontal_cdr(disc_mask, cup_mask):
        x_disc, x_cup = np.where(disc_mask)[1], np.where(cup_mask)[1]
        if not (len(x_disc) and len(x_cup)): return None
        return round((x_cup.max() - x_cup.min() + 1) / (x_disc.max() - x_disc.min() + 1), 3)

    def cup_shape_index(disc_mask, cup_mask):
        disc_area = np.sum(disc_mask)
        cup_area = np.sum(cup_mask)
        rim_area = disc_area - cup_area
        if not (cup_area and rim_area > 0): return None
        return round(rim_area / cup_area, 3)

    def process_image(image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = processor(rgb, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits.cpu()
        upsampled_logits = nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
        pred = upsampled_logits.argmax(dim=1)[0].numpy().astype(np.uint8)
        disc_mask, cup_mask = (pred == 1), (pred == 2)
        overlay = rgb.copy()
        overlay[disc_mask] = [255, 255, 0]
        overlay[cup_mask] = [255, 0, 0]
        vcdr = vertical_cdr(disc_mask, cup_mask)
        acdr = acdr_area(disc_mask, cup_mask)
        hcdr = horizontal_cdr(disc_mask, cup_mask)
        csi = cup_shape_index(disc_mask, cup_mask)
        prediction_label, confidence = 'N/A', 'N/A'
        if clf is not None and all(x is not None for x in [vcdr, acdr, hcdr, csi]):
            feature_data = {'VCDR': [vcdr],'ACDR': [acdr],'HCDR': [hcdr],'CSI': [csi]}
            features = pd.DataFrame(feature_data)
            prob = clf.predict_proba(features)[0]
            prediction = np.argmax(prob)
            prediction_label = "Glaucoma" if prediction == 1 else "Normal"
            confidence = prob[prediction]
        return rgb, overlay, {"VCDR": vcdr, "ACDR": acdr, "HCDR": hcdr, "CSI": csi, "Prediction": prediction_label, "Confidence": confidence}

    if uploaded_file and clf is not None:
        if not patient_name:
            st.warning("Please enter a Patient Name to generate a report.")
        else:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            with st.spinner("Analyzing image..."):
                original, overlay, metrics = process_image(image)
            
            metrics_df = pd.DataFrame([{
                "VCDR": metrics.get("VCDR"), "ACDR": metrics.get("ACDR"),
                "HCDR": metrics.get("HCDR"), "CSI": metrics.get("CSI"),
                "Prediction": metrics.get("Prediction"), "Confidence": metrics.get("Confidence")
            }])
            
            glaucoma_metrics = {k: v for k, v in metrics.items() if k in ["VCDR", "ACDR", "HCDR", "CSI"] and v is not None}
            fig = None
            if glaucoma_metrics:
                fig = go.Figure(data=[go.Bar(x=list(glaucoma_metrics.keys()), y=list(glaucoma_metrics.values()), marker_color='teal', text=[str(v) for v in glaucoma_metrics.values()], textposition='outside')])
                fig.update_layout(title="Glaucoma-Specific Metrics", height=500, yaxis=dict(range=[0, max(glaucoma_metrics.values() or [0]) + 0.1]))

            patient_info = {
                "Patient Name": patient_name, "Age": age, "Gender": gender,
                "Comorbidities": ", ".join(comorbidities) if comorbidities else "None"
            }
            
            pdf_bytes = create_report_pdf(patient_info, original, overlay, metrics_df, fig)
            st.download_button(
                label="📥 Download Report as PDF",
                data=pdf_bytes,
                file_name=f"Glaucoma_Report_{patient_name.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )

            col1, col2 = st.columns(2)
            with col1:
                st.image(original, caption="Original Image", use_container_width=True)
            with col2:
                st.image(overlay, caption="Segmented Overlay", use_container_width=True)
            
            st.subheader("Analysis Results")
            st.dataframe(metrics_df.style.format({"Confidence": "{:.3f}"}), use_container_width=True)
            
            st.markdown(f"### Prediction: **{metrics['Prediction']}**")
            if isinstance(metrics['Confidence'], (float, np.floating)):
                 st.markdown(f"**Confidence:** {metrics['Confidence'] * 100:.1f}%")
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
