import numpy as np
from mathlib import get_power_of_two
from nnlib import nnlib
from models import ModelBase
from facelib import FaceType
from samples import *

class Model(ModelBase):

    GAH5 = 'GA.h5'
    GBH5 = 'GB.h5'
    
    #override
    def onInitialize(self, **in_options):
        exec(nnlib.import_all(), locals(), globals())
        self.set_vram_batch_requirements( {6:32} )
               
        if self.epoch == 0: 
            #first run
            
            print ("\nModel first run. Enter options.")
            
            try:
                created_resolution = int ( input ("Resolution (default:64, valid: 64,128,256) : ") )
            except:
                created_resolution = 64
                
            if created_resolution not in [64,128,256]:
                created_resolution = 64

            try:
                created_batch_size = int ( input ("Batch_size (minimum/default - 1) : ") )
            except:
                created_batch_size = 1
            created_batch_size = max(created_batch_size,1)
            
            print ("Done. If training won't start, decrease resolution")
     
            self.options['created_resolution'] = created_resolution
            self.options['created_batch_size'] = created_batch_size
            self.created_vram_gb = self.device_config.gpu_total_vram_gb
        else: 
            #not first run
            if 'created_batch_size' in self.options.keys():
                created_batch_size = self.options['created_batch_size']
            else:
                raise Exception("Continue training, but created_batch_size not found.")
                
            if 'created_resolution' in self.options.keys():
                created_resolution = self.options['created_resolution']
            else:
                raise Exception("Continue training, but created_resolution not found.")
                
        self.set_batch_size(created_batch_size)
        
        resolution = created_resolution
        bgr_shape = (resolution, resolution, 3)
        mask_shape = (resolution, resolution, 1)
        

        use_batch_norm = created_batch_size > 1
        #self.GA = modelify(UNet (bgr_shape[2], use_batch_norm, num_downs=get_power_of_two(resolution)-1, ngf=64, use_dropout=False))(Input(bgr_shape))
        #self.GB = modelify(UNet (bgr_shape[2], use_batch_norm, num_downs=get_power_of_two(resolution)-1, ngf=64, use_dropout=False))(Input(bgr_shape))
        self.GA = modelify(ResNet (bgr_shape[2], use_batch_norm, n_blocks=6, ngf=64, use_dropout=False))(Input(bgr_shape))
        self.GB = modelify(ResNet (bgr_shape[2], use_batch_norm, n_blocks=6, ngf=64, use_dropout=False))(Input(bgr_shape))
        
        if not self.is_first_run():
            self.GA.load_weights (self.get_strpath_storage_for_file(self.GAH5))
            self.GB.load_weights (self.get_strpath_storage_for_file(self.GBH5))
 
        real_A0 = Input(bgr_shape, name="real_A0")
        real_B0 = Input(bgr_shape, name="real_B0")
        target_A0 = Input(bgr_shape, name="target_A0")
        target_A0m = Input(bgr_shape)
        target_B0 = Input(bgr_shape)
        target_B0m = Input(bgr_shape)
        
        fake_B0 = self.GA(real_A0)        
        fake_A0 = self.GB(real_B0)  

        pred_A0 = self.GB(fake_B0)
        pred_B0 = self.GA(fake_A0)
        
        AE_loss = K.mean(  tf_dssim(2.0)( (target_A0+1)*(target_A0m+1), (pred_A0+1)*(target_A0m+1)) \
                         + tf_dssim(2.0)( (target_B0+1)*(target_B0m+1), (pred_B0+1)*(target_B0m+1)) \
                         )
        #AE_loss = K.mean(  K.mean(K.abs((target_A0, pred_A0))) \
        #                 + K.mean(K.abs((target_B0, pred_B0))) \
        #                 )*10
        #mask_loss = K.mean(K.square(target_srcm-pred_src_srcm)) + K.mean(K.square(target_dstm-pred_dst_dstm))
        
        weights_AE = self.GA.trainable_weights + self.GB.trainable_weights
        #weights_mask = self.decoder_srcm.trainable_weights + self.decoder_dstm.trainable_weights
        
        self.AE_train = K.function ([real_A0, target_A0, target_A0m, real_B0, target_B0, target_B0m],[AE_loss],
                                    Adam(lr=2e-4, beta_1=0.5, beta_2=0.999).get_updates(AE_loss, weights_AE) )
                                    
        AE1_loss = K.mean(  tf_dssim(2.0)( (target_A0+1), (pred_A0+1) ) )        #*(target_A0m+1)     
                         
        self.AE1_train = K.function ([real_A0, target_A0],[AE1_loss],
                            Adam(lr=2e-4, beta_1=0.5, beta_2=0.999).get_updates(AE1_loss, weights_AE) )
                            
        AE2_loss = K.mean(  tf_dssim(2.0)( (target_B0+1), (pred_B0+1) ) )             
        
        self.AE2_train = K.function ([real_B0, target_B0],[AE2_loss],
                            Adam(lr=2e-4, beta_1=0.5, beta_2=0.999).get_updates(AE2_loss, weights_AE) )
        #                            
        #self.mask_train = K.function ([warped_src, target_srcm, warped_dst, target_dstm],[mask_loss],
        #                            Adam(lr=2e-4, beta_1=0.5, beta_2=0.999).get_updates(mask_loss, weights_mask) )

        self.AE_view = K.function ([real_A0, real_B0],[fake_B0, pred_A0, fake_A0, pred_B0])
         
        if self.is_training_mode:
            f = SampleProcessor.TypeFlags
            self.set_training_data_generators ([            
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size, 
                        sample_process_options=SampleProcessor.Options(normalize_tanh = True), 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_M | f.FACE_MASK_FULL, resolution] ] ),
                                              
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, 
                        sample_process_options=SampleProcessor.Options(normalize_tanh = True), 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | f.FACE_ALIGN_FULL | f.MODE_M | f.FACE_MASK_FULL, resolution] ] )
                ])
    #override
    def onSave(self):        
        self.save_weights_safe( [[self.GA,    self.get_strpath_storage_for_file(self.GAH5)],
                                 [self.GB,    self.get_strpath_storage_for_file(self.GBH5)]] )
        
    #override
    def onTrainOneEpoch(self, sample):
        warped_src, target_src, target_src_mask = sample[0]
        warped_dst, target_dst, target_dst_mask = sample[1]    

        #AE_loss, = self.AE_train ([warped_src, target_src, target_src_mask, warped_dst, target_dst, target_dst_mask])
        AE1_loss, = self.AE1_train ([warped_src, target_src])
        AE2_loss, = self.AE2_train ([warped_dst, target_dst])
        mask_loss = 0#self.mask_train ([target_src, target_src_mask, target_dst, target_dst_mask])
        
        return ( ('AE_loss', (AE1_loss+AE2_loss)/2), ('mask_loss', mask_loss) )
        

    #override
    def onGetPreview(self, sample):
        test_A   = sample[0][1][0:4] #first 4 samples
        test_A_m = sample[0][2][0:4] #first 4 samples
        test_B   = sample[1][1][0:4]
        test_B_m = sample[1][2][0:4]
        
        S = test_A
        D = test_B
        SD, SDS, DS, DSD = self.AE_view ([test_A, test_B])
        
        S, D, SD, SDS, DS, DSD = [ x / 2 + 0.5 for x in [S, D, SD, SDS, DS, DSD] ]
        
        
        #AA, mAA = self.autoencoder_src.predict([test_A, test_A_m])                                       
        #AB, mAB = self.autoencoder_src.predict([test_B, test_B_m])
        #BB, mBB = self.autoencoder_dst.predict([test_B, test_B_m])
        #Smask = np.repeat ( Smask, (3,), -1)
        #Dmask = np.repeat ( Dmask, (3,), -1)
        #mBB = np.repeat ( mBB, (3,), -1)
        
        st = []
        for i in range(0, len(test_A)):
            st.append ( np.concatenate ( (
                S[i], SD[i], SDS[i],  
                D[i], DS[i], DSD[i], 
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