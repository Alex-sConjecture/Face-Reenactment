import numpy as np

from nnlib import nnlib
from models import ModelBase
from facelib import FaceType
from samples import *

class Model(ModelBase):

    encoderH5 = 'encoder.h5'
    decoder_srcH5 = 'decoder_src.h5'
    decoder_dstH5 = 'decoder_dst.h5'
    decoder_srcmH5 = 'decoder_srcm.h5'
    decoder_dstmH5 = 'decoder_dstm.h5'
    #override
    def onInitialize(self, **in_options):
        exec(nnlib.import_all(), locals(), globals())
        self.set_vram_batch_requirements( {6:8} )
               
        resolution = 128
        bgr_shape = (resolution, resolution, 3)
        mask_shape = (resolution, resolution, 1)
        
        
        
        enc, dec, decm = self.GetTensors()
        
        self.encoder = modelify (enc) ( Input(bgr_shape))
        
        enc_output_shape = K.int_shape ( self.encoder.outputs[0] )[1:]
        
        self.decoder_src = modelify(dec) ( Input(enc_output_shape) )
        self.decoder_dst = modelify(dec) ( Input(enc_output_shape) )
        self.decoder_srcm = modelify(decm) ( Input(enc_output_shape) )
        self.decoder_dstm = modelify(decm) ( Input(enc_output_shape) )
        
        
        

        if not self.is_first_run():
            self.encoder.load_weights     (self.get_strpath_storage_for_file(self.encoderH5))
            self.decoder_src.load_weights (self.get_strpath_storage_for_file(self.decoder_srcH5))
            self.decoder_dst.load_weights (self.get_strpath_storage_for_file(self.decoder_dstH5))
            self.decoder_srcm.load_weights (self.get_strpath_storage_for_file(self.decoder_srcmH5))
            self.decoder_dstm.load_weights (self.get_strpath_storage_for_file(self.decoder_dstmH5))
 
        warped_src = Input(bgr_shape)
        target_src = Input(bgr_shape)
        target_srcm = Input(mask_shape)
        
        warped_src_code = self.encoder (warped_src)
        pred_src_src = self.decoder_src(warped_src_code)
        pred_dst_src = self.decoder_dst(warped_src_code)
        pred_dst_src_code = self.encoder (pred_dst_src)
        pred_src_pred_dst_src = self.decoder_src(pred_dst_src_code)        
        pred_src_srcm = self.decoder_srcm(warped_src_code)
        
        
        
        warped_dst = Input(bgr_shape)
        target_dst = Input(bgr_shape)
        target_dstm = Input(mask_shape)
        warped_dst_code = self.encoder (warped_dst)
        pred_dst_dst = self.decoder_dst(warped_dst_code)
        pred_src_dst = self.decoder_src(warped_dst_code)
        pred_src_dst_code = self.encoder (pred_src_dst)
        pred_dst_pred_src_dst = self.decoder_dst(pred_src_dst_code)
        pred_dst_dstm = self.decoder_dstm(warped_dst_code)
        
        AE_loss = K.mean(  K.mean(K.abs((target_src - pred_src_pred_dst_src))) \
                         + K.mean(K.abs((target_dst - pred_dst_pred_src_dst))) \
                         )
        
        mask_loss = K.mean(K.square(target_srcm-pred_src_srcm)) + K.mean(K.square(target_dstm-pred_dst_dstm))
        
        weights_AE = self.encoder.trainable_weights + self.decoder_src.trainable_weights + self.decoder_dst.trainable_weights
        weights_mask = self.decoder_srcm.trainable_weights + self.decoder_dstm.trainable_weights
        
        self.AE_train = K.function ([warped_src, target_src, target_srcm, warped_dst, target_dst, target_dstm],[AE_loss],
                                    Adam(lr=2e-4, beta_1=0.5, beta_2=0.999).get_updates(AE_loss, weights_AE) )
                                    
        self.mask_train = K.function ([warped_src, target_srcm, warped_dst, target_dstm],[mask_loss],
                                    Adam(lr=2e-4, beta_1=0.5, beta_2=0.999).get_updates(mask_loss, weights_mask) )

        self.AE_view = K.function ([warped_src, warped_dst],[pred_src_src, pred_dst_src, pred_src_pred_dst_src, pred_src_srcm, pred_dst_dst, pred_src_dst, pred_dst_pred_src_dst, pred_dst_dstm])
         
        if self.is_training_mode:
            f = SampleProcessor.TypeFlags
            self.set_training_data_generators ([            
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size, 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_M | f.FACE_MASK_FULL, resolution] ] ),
                                              
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_M | f.FACE_MASK_FULL, resolution] ] )
                ])
    #override
    def onSave(self):        
        self.save_weights_safe( [[self.encoder, self.get_strpath_storage_for_file(self.encoderH5)],
                                [self.decoder_src, self.get_strpath_storage_for_file(self.decoder_srcH5)],
                                [self.decoder_dst, self.get_strpath_storage_for_file(self.decoder_dstH5)],
                                [self.decoder_srcm, self.get_strpath_storage_for_file(self.decoder_srcmH5)],
                                [self.decoder_dstm, self.get_strpath_storage_for_file(self.decoder_dstmH5)]] )
        
    #override
    def onTrainOneEpoch(self, sample):
        warped_src, target_src, target_src_mask = sample[0]
        warped_dst, target_dst, target_dst_mask = sample[1]    

        AE_loss, = self.AE_train ([warped_src, target_src, target_src_mask, warped_dst, target_dst, target_dst_mask])
        mask_loss, = self.mask_train ([target_src, target_src_mask, target_dst, target_dst_mask])
        
        return ( ('AE_loss', AE_loss), ('mask_loss', mask_loss) )
        

    #override
    def onGetPreview(self, sample):
        test_A   = sample[0][1][0:4] #first 4 samples
        test_A_m = sample[0][2][0:4] #first 4 samples
        test_B   = sample[1][1][0:4]
        test_B_m = sample[1][2][0:4]
        
        S = test_A
        D = test_B
        SS, SD, SDS, Smask, DD, DS, DSD, Dmask = self.AE_view ([test_A, test_B])
        
        #AA, mAA = self.autoencoder_src.predict([test_A, test_A_m])                                       
        #AB, mAB = self.autoencoder_src.predict([test_B, test_B_m])
        #BB, mBB = self.autoencoder_dst.predict([test_B, test_B_m])
        Smask = np.repeat ( Smask, (3,), -1)
        Dmask = np.repeat ( Dmask, (3,), -1)
        #mBB = np.repeat ( mBB, (3,), -1)
        
        st = []
        for i in range(0, len(test_A)):
            st.append ( np.concatenate ( (
                S[i], SS[i], SD[i], SDS[i], Smask[i], 
                D[i], DD[i], DS[i], DSD[i], Dmask[i],
                ), axis=1) )
            
        return [ ('TEST', np.concatenate ( st, axis=0 ) ) ]
    
    def predictor_func (self, face):
        
        face_128_bgr = face[...,0:3]
        face_128_mask = np.expand_dims(face[...,3],-1)
        
        x, mx = self.autoencoder_src.predict ( [ np.expand_dims(face_128_bgr,0), np.expand_dims(face_128_mask,0) ] )
        x, mx = x[0], mx[0]
        
        return np.concatenate ( (x,mx), -1 )
        
    #override
    def get_converter(self, **in_options):
        from models import ConverterMasked
        
        if 'erode_mask_modifier' not in in_options.keys():
            in_options['erode_mask_modifier'] = 0
        in_options['erode_mask_modifier'] += 30
            
        if 'blur_mask_modifier' not in in_options.keys():
            in_options['blur_mask_modifier'] = 0
            
        return ConverterMasked(self.predictor_func, predictor_input_size=128, output_size=128, face_type=FaceType.FULL, clip_border_mask_per=0.046875, **in_options)
        
    def GetTensors(self):
        exec(nnlib.code_import_all, locals(), globals())
    
        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)(Conv2D(dim, 5, strides=2, padding='same')(x))
            return func 
            
        def upscale (dim):
            def func(x):
                return PixelShuffler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func   
            
        def Encoder(input):            
            x = input
            x = downscale(128)(x)
            x = downscale(256)(x)
            x = downscale(512)(x)
            x = downscale(1024)(x)

            x = Dense(512)(Flatten()(x))
            x = Dense(8 * 8 * 512)(x)
            x = Reshape((8, 8, 512))(x)
            x = upscale(512)(x)
                
            return x

        def Decoder(input):
            x = input
            x = upscale(512)(x)
            x = upscale(256)(x)
            x = upscale(128)(x)
                
            x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
            
            return x
            
        def DecoderMask(input):
            y = input  #mask decoder
            y = upscale(512)(y)
            y = upscale(256)(y)
            y = upscale(128)(y)
                
            y = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(y)
            
            return y
            
        return Encoder, Decoder, DecoderMask