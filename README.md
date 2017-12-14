## Redaction Network code

To install requirements:
```python
pip install -r requirements.txt
```

To run network:
```python
cd data
python generate_train_test_split.py <float>TEST_SPLIT_SIZE
cd ..
python redaction.py <int>NUM_EPOCHS <int>BATCH_SIZE <int>ENCODING_SIZE <float>RECON_ALPHA <int>MODULE_SIZE <int>NUM_DISCR_LAYERS
```
