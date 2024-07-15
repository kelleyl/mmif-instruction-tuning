import os
import json
import argparse
from typing import Dict, List
import mmif
from jinja2 import Template
import yaml
from mmif import Mmif, DocumentTypes, AnnotationTypes
from mmif.utils import video_document_helper as vdh
from PIL import Image
import base64
from io import BytesIO

def load_config(config_file: str) -> Dict:
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def process_mmif_file(file_path: str, config: Dict, output_dir: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        mmif = Mmif(f.read())
    
    qa_pairs = []
    video_doc = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
    
    for view in 
        # timeframes = view.get_annotations(AnnotationTypes.TimeFrame)        
        for timeframe in timeframes:
            label = timeframe.get_property('label')
            if label in config:
                qa_template = config[label]
                context = {
                    'OCR_RESULT': '',
                }
                
                # Extract frame from the timeframe
                if "representatives" in timeframe.properties:
                    frame = vdh.extract_representative_frame(mmif_data, timeframe, as_PIL=True)
                else:
                    frame = vdh.extract_mid_frame(mmif_data, timeframe, as_PIL=True)
                
                # Get aligned text if available
                aligned_text = get_aligned_text(mmif_data, timeframe)
                if aligned_text:
                    context['OCR_RESULT'] = aligned_text
                
                conversations = []
                for qa in qa_template:
                    human = Template(qa['human']).render(context)
                    gpt = Template(qa['gpt']).render(context)
                    conversations.append({"human": human, "gpt": gpt})
                
                # Save the frame as an image
                image_filename = f"{os.path.basename(file_path)}_{timeframe.id}.jpg"
                image_path = os.path.join(output_dir, image_filename)
                frame.save(image_path)
                
                # Convert image to base64
                buffered = BytesIO()
                frame.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                qa_pairs.append({
                    "image": img_str,
                    "conversations": conversations
                })
    
    return qa_pairs

def get_aligned_text(mmif_data: Mmif, timeframe) -> str:
    for view in mmif_data.views:
        alignments = view.get_annotations(AnnotationTypes.Alignment)
        for alignment in alignments:
            if alignment.get_property('source') == timeframe.id:
                target_id = alignment.get_property('target')
                text_doc = mmif_data.get_document_by_id(target_id)
                if text_doc:
                    return text_doc.text_value
    return ''

def main(input_dir: str, output_file: str, config_file: str, image_output_dir: str):
    config = load_config(config_file)
    all_qa_pairs = []

    os.makedirs(image_output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.mmif'):
            file_path = os.path.join(input_dir, filename)
            all_qa_pairs.extend(process_mmif_file(file_path, config, image_output_dir))

    with open(output_file, 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate instruction tuning data from MMIF files')
    parser.add_argument('input_dir', help='Directory containing MMIF files')
    parser.add_argument('output_file', help='Output file for instruction tuning data')
    parser.add_argument('config_file', help='Configuration file for QA templates')
    parser.add_argument('image_output_dir', help='Directory to save extracted images')
    args = parser.parse_args()

    main(args.input_dir, args.output_file, args.config_file, args.image_output_dir)