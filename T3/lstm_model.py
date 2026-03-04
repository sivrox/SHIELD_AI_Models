import tensorflow as tf

def get_shield_model():
    """
    Returns the Industry-Grade S.H.I.E.L.D. LSTM architecture.
    """
    model = tf.keras.Sequential([
        # Input: 60 seconds of 8 vitals
        tf.keras.layers.InputLayer(input_shape=(60, 8)), 
        
        # Bi-Directional: Reads heart patterns forward and backward
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        
        # Stabilizer: Prevents 'Exploding Math' which causes 100% scores
        tf.keras.layers.BatchNormalization(),
        
        # Pooling: Averages the minute to ignore tiny sensor flickers
        tf.keras.layers.GlobalAveragePooling1D(),
        
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        # Final Risk Score: 0.0 (Safe) to 1.0 (Emergency)
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compiling with a slower learning rate for high stability
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# WHY BI-DIRECTIONAL?
# In cardiology, what happens AFTER a spike is just as important as 
# what happened BEFORE. This layer allows the AI to see the 'Full Story'.