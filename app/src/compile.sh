find . -type d \( -name "compiled_model" -o -name "__pycache__" -o -name "compiler_workdir" \) -exec rm -rf {} +

python "./text_encoder_1/compile.py"
python "./text_encoder_2/compile.py" -m 32
python "./transformer/compile.py" -hh $HEIGHT -w $WIDTH -m 32
python "./decoder/compile.py" -hh $HEIGHT -w $WIDTH
echo "done compiling; cleaning compiler_workdir"
find . -type d \( -name "__pycache__" -o -name "compiler_workdir" \) -exec rm -rf {} +

