# flux-enigma

clean previous compilation
```bash
rm -rf decoder/compiler_workdir/ decoder/__pycache__/ text_encoder_1/compiler_workdir text_encoder_2/compiler_workdir/ text_encoder_2/compiler_workdir/ text_encoder_2/__pycache__/ transformer/compiler_workdir/ transformer/__pycache__/ transformer/compiled_model decoder/compiled_model text_encoder_1/compiled_model text_encoder_2/compiled_model/ text_encoder_1/__pycache__/
```
modify the shapes and:....

```bash
./compile.sh > compile.log 2>&1 &
```
