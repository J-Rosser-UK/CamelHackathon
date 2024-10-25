import urllib.parse

def generate_ukri_url(query, page):
    # Base URL for UKRI opportunities with predefined parameters
    base_url = f"https://www.ukri.org/opportunity/page/{page}/"
    # Define the parameters for the URL
    params = {
        'keywords': query,
        'filter_status[]': ['open', 'upcoming'],
        'filter_order': 'publication_date',
        'filter_submitted': 'true'
    }
    # Encode the parameters for the URL
    encoded_params = urllib.parse.urlencode(params, doseq=True)
    # Concatenate base URL with encoded parameters
    full_url = f"{base_url}?{encoded_params}"
    return full_url
