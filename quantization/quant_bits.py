"""
INT4 Quantization Stub for RxLM-Med Agent
"""

def quantize_model_to_int4(model_name: str, output_dir: str):

    try:
        # Option 1: AutoGPTQ (preferred for Qwen)
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        
        quantize_config = BaseQuantizeConfig(
            bits=4,
            group_size=128,
            damp_percent=0.01,
            desc_act=False,  # faster inference
            sym=True,
            true_sequential=True
        )
        
        model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model.quantize([])  # placeholder
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)
        
    except ImportError:
        # Option 2: bitsandbytes
        print("AutoGPTQ not installed. Use `pip install auto-gptq` for INT4.")
        print("Alternatively, load with `load_in_4bit=True` in transformers.")
        return None

if __name__ == "__main__":
    # Example usage
    quantize_model_to_int4(
        model_name="./final_agent",
        output_dir="./final_agent_int4"
    )