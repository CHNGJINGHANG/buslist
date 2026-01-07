import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import urllib.parse
import html

def initialize_session_state():
    if 'schedule_data' not in st.session_state:
        st.session_state.schedule_data = []
    if 'recipient_emails' not in st.session_state:
        st.session_state.recipient_emails = []


def create_schedule():
    """Create new schedule entries"""
    st.header("📅 Add Schedule Entry")
    
    today = datetime.now()
    upcoming_saturday = today + timedelta((5 - today.weekday()) % 7)
    
    with st.form("schedule_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date = st.date_input("Date", value=upcoming_saturday)
            activity = st.text_input("Activity", value="Dragon Boat (M)")
            pickup_point = st.text_input("Pick-Up Point", value="NTU Hall 8 & 9 Bus Stop")
        
        with col2:
            departure_time = st.text_input("Departure Time", value="0740 hrs")
            bus_capacity = st.selectbox("Bus Capacity", ["1 x 20 seater bus", "1 x 40 seater bus"])
            return_time = st.text_input("Return Time", value="NIL")
        
        with col3:
            contact_name = st.text_input("Contact Name", value="Jing Hang")
            contact_number = st.text_input("Contact Number", value="88479136")
        
        st.subheader("Destinations")
        num_destinations = st.number_input("Number of Destinations", min_value=1, max_value=5, value=2)
        
        destinations = []
        for i in range(num_destinations):
            dest = st.text_input(f"Destination {i+1}", 
                                value="Venture Ave (Jurong East)" if i == 0 else "SDBA" if i == 1 else "",
                                key=f"dest_{i}")
            if dest:
                destinations.append(dest)
        
        submitted = st.form_submit_button("➕ Add to Schedule", type="primary")
        
        if submitted:
            new_entry = {
                'date': date.strftime("%d/%m/%Y"),
                'day': date.strftime("%A").upper(),
                'activity': activity,
                'pickup_point': pickup_point,
                'departure_time': departure_time,
                'destinations': destinations,
                'return_time': return_time,
                'contact_name': contact_name,
                'contact_number': contact_number,
                'bus_capacity': bus_capacity
            }
            
            duplicate_found = False
            for existing in st.session_state.schedule_data:
                if (existing['date'] == new_entry['date'] and 
                    existing['departure_time'] == new_entry['departure_time']):
                    duplicate_found = True
                    st.warning(f"Schedule already exists for {new_entry['date']} at {new_entry['departure_time']}")
                    break
            
            if not duplicate_found:
                st.session_state.schedule_data.append(new_entry)
                st.success(f"✓ Added schedule for {new_entry['date']} ({new_entry['day']})")
                st.rerun()


def view_schedule():
    """View and manage schedule"""
    st.header("📋 Current Schedule")
    
    if not st.session_state.schedule_data:
        st.info("No schedule entries yet. Create one in the 'Create Schedule' tab.")
        return
    
    display_data = []
    for entry in st.session_state.schedule_data:
        destinations_str = ", ".join([f"{i+1}. {dest}" for i, dest in enumerate(entry['destinations'])])
        
        display_data.append({
            'Date': f"{entry['date']}\n{entry['day']}",
            'Activity': entry['activity'],
            'Pick-Up': entry['pickup_point'],
            'Time': entry['departure_time'],
            'Destination': destinations_str,
            'Return': entry['return_time'],
            'Contact': f"{entry['contact_name']}\n{entry['contact_number']}",
            'Seats': entry['bus_capacity']
        })
    
    df = pd.DataFrame(display_data)
    st.dataframe(df, use_container_width=True, height=400)
    
    st.markdown("---")
    st.subheader("Remove Entry")
    
    if len(st.session_state.schedule_data) > 0:
        selected_index = st.selectbox(
            "Select entry to remove:",
            range(len(st.session_state.schedule_data)),
            format_func=lambda x: f"{st.session_state.schedule_data[x]['date']} - {st.session_state.schedule_data[x]['activity']}"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Remove Selected Entry"):
                st.session_state.schedule_data.pop(selected_index)
                st.success("Entry removed")
                st.rerun()
        
        with col2:
            if st.button("⚠️ Clear All Schedules"):
                st.session_state.schedule_data = []
                st.success("All schedules cleared")
                st.rerun()


def generate_schedule_html():
    """Generate HTML table for email"""
    if not st.session_state.schedule_data:
        return ""
    
    html_content = """<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;">
    <thead>
    <tr style="background-color: #f2f2f2;">
        <th>Date (2025)</th>
        <th>Activity</th>
        <th>Pick-Up Point</th>
        <th>Departure Time</th>
        <th>Destination</th>
        <th>Return Time</th>
        <th>Name & Contact No.</th>
        <th>Seats</th>
        <th>Price</th>
    </tr>
    </thead>
    <tbody>"""
    
    for entry in st.session_state.schedule_data:
        destinations_html = "<br>".join([f"{i+1}. {dest}" for i, dest in enumerate(entry['destinations'])])
        
        html_content += f"""
    <tr>
        <td>{entry['date']}<br><strong>{entry['day']}</strong></td>
        <td>{entry['activity']}</td>
        <td>{entry['pickup_point']}</td>
        <td>{entry['departure_time']}</td>
        <td>{destinations_html}</td>
        <td>{entry['return_time']}</td>
        <td>{entry['contact_name']},<br>{entry['contact_number']}</td>
        <td>{entry['bus_capacity']}</td>
        <td></td>
    </tr>"""
    
    html_content += """
    </tbody>
    </table>"""
    return html_content


def send_schedule_email():
    """Generate mailto link for schedule email"""
    st.header("📧 Email Schedule")
    
    if not st.session_state.schedule_data:
        st.warning("⚠️ No schedule data available. Please create a schedule first.")
        return
    
    # Manage recipients
    with st.expander("👥 Manage Recipients"):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_email = st.text_input("Add recipient email:", placeholder="example@email.com")
        with col2:
            st.write("")
            st.write("")
            if st.button("Add"):
                if new_email and '@' in new_email:
                    if new_email not in st.session_state.recipient_emails:
                        st.session_state.recipient_emails.append(new_email)
                        st.success(f"✓ Added: {new_email}")
                        st.rerun()
                    else:
                        st.warning("Email already exists")
                else:
                    st.warning("Enter valid email")
        
        if st.session_state.recipient_emails:
            st.write("**Current recipients:**")
            for i, email in enumerate(st.session_state.recipient_emails):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.text(f"• {email}")
                with col2:
                    if st.button("❌", key=f"rm_email_{i}"):
                        st.session_state.recipient_emails.remove(email)
                        st.rerun()
    
    st.markdown("---")
    
    # Email composition
    st.subheader("✉️ Compose Email")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.recipient_emails:
            selected_recipients = st.multiselect(
                "Recipients:",
                st.session_state.recipient_emails,
                default=st.session_state.recipient_emails
            )
        else:
            selected_recipients = []
            manual_recipient = st.text_input("Recipient Email:")
            if manual_recipient:
                selected_recipients = [manual_recipient]
    
    with col2:
        cc_email = st.text_input("CC (Optional):")
    
    recipient_name = st.text_input("Recipient's Name:", placeholder="Ms. Ivyna")
    sender_name = st.text_input("Your Name:", placeholder="Your name")
    
    st.markdown("---")
    
    # Preview HTML table
    st.subheader("📋 Email Preview")
    html_table = generate_schedule_html()
    st.markdown(html_table, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("📧 Generate Email", type="primary"):
        if not selected_recipients:
            st.warning("Please enter at least one recipient email")
            return
        
        recipient = ','.join(selected_recipients)  # Use comma for multiple recipients
        subject = "NTUDB(M) Bus Schedule"
        
        # Create HTML email body with proper spacing
        html_body = f"""<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6;">
<p>Dear {html.escape(recipient_name if recipient_name else 'Recipient')},</p>

<p>The Bus Schedule for NTU Dragon Boat (M) is as follows:</p>

<br>

{html_table}

<br>
<br>

<p>Thank you for your support!</p>

<p>Warm regards,<br>
{html.escape(sender_name if sender_name else 'NTU Dragon Boat (M)')}</p>
</body>
</html>"""
        
        # Create mailto link with HTML body
        mailto_link = f"mailto:{recipient}?subject={urllib.parse.quote(subject)}"
        
        if cc_email:
            mailto_link += f"&cc={urllib.parse.quote(cc_email)}"
        
        # Use HTML body directly
        mailto_link += f"&body={urllib.parse.quote(html_body)}"
        
        # Display the link
        st.success("✅ Email content generated!")
        
        st.markdown(f'<a href="{mailto_link}" target="_blank" style="display: inline-block; padding: 12px 24px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; margin: 10px 0;">📧 Open Email Client with HTML Table</a>', unsafe_allow_html=True)
        
        st.info("💡 **Instructions:**\n1. Click the button above to open your email client\n2. The HTML table should appear formatted in the email\n3. If the table doesn't appear formatted, use Gmail web interface or copy the HTML below")
        
        # Provide HTML code as backup
        with st.expander("📋 Alternative: Copy & Paste HTML (if mailto doesn't work)"):
            st.markdown("**Instructions:** Copy the HTML below and paste it directly into Gmail or Outlook compose window")
            st.code(html_body, language="html")
            st.download_button(
                "📥 Download HTML Email",
                data=html_body,
                file_name=f"email_schedule_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html"
            )


def main():
    st.set_page_config(
        page_title="Bus Schedule Manager",
        page_icon="📅",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("📅 Bus Schedule Manager")
    st.markdown("### NTU Dragon Boat (M)")
    
    # Main tabs
    tabs = st.tabs(["➕ Create Schedule", "📋 View & Manage", "📧 Send Email"])
    
    with tabs[0]:
        create_schedule()
    
    with tabs[1]:
        view_schedule()
    
    with tabs[2]:
        send_schedule_email()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Status")
    st.sidebar.metric("Total Schedules", len(st.session_state.schedule_data))
    
    if st.session_state.schedule_data:
        st.sidebar.write("**Upcoming Schedules:**")
        for entry in st.session_state.schedule_data[:5]:  # Show first 5
            st.sidebar.text(f"• {entry['date']} - {entry['activity']}")
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Quick Guide:**\n\n1. Create schedules with details\n2. Review and manage entries\n3. Generate and send emails")


if __name__ == "__main__":
    main()
