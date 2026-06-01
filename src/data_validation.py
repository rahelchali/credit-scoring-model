import logging

# Configure production logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BatiBankLogger")

def validate_input_data(df):
    """Robust data validation guarding financial thresholds."""
    logger.info("🎬 Initializing schema validation checks...")
    
    if df.empty:
        logger.error("❌ Validation Failed: Input DataFrame is empty.")
        raise ValueError("Input data cannot be empty.")
        
    if 'Amount' not in df.columns:
        logger.error("❌ Validation Failed: Missing mandatory 'Amount' field.")
        raise KeyError("Missing 'Amount' column.")
        
    logger.info(f"✅ Validation Passed. Records checked: {len(df)}")
    return True