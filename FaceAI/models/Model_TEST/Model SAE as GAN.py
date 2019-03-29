from models import ModelBase
import numpy as np
import cv2
from mathlib import get_power_of_two
from nnlib import nnlib
from facelib import FaceType
from samples import *
from utils.console_utils import *

class Model(ModelBase):
    GH5 = 'G.h5'
    GAH5 = 'GA.h5'
    DAH5 = 'DA.h5'
    GBH5 = 'GB.h5'
    DBH5 = 'DB.h5'

    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        default_resolution = 128
        default_archi = 'df'
        default_face_type = 'f'
        
        if is_first_run:
            self.options['resolution'] = input_int("Resolution (64,128 ?:help skip:128) : ", default_resolution, [64,128], help_message="More resolution requires more VRAM.")
            self.options['face_type'] = input_str ("Half or Full face? (h/f, ?:help skip:f) : ", default_face_type, ['h','f'], help_message="Half face has better resolution, but covers less area of cheeks.").lower()            
            #self.options['archi'] = input_str ("AE architecture (df, liae, ?:help skip:%s) : " % (default_archi) , default_archi, ['df','liae'], help_message="DF keeps faces more natural, while LIAE can fix overly different face shapes.").lower()            
            #self.options['lighter_encoder'] = input_bool ("Use lightweight encoder? (y/n, ?:help skip:n) : ", False, help_message="Lightweight encoder is 35% faster, requires less VRAM, sacrificing overall quality.")
            #self.options['learn_mask'] = input_bool ("Learn mask? (y/n, ?:help skip:y) : ", True, help_message="Choose NO to reduce model size. In this case converter forced to use 'not predicted mask' that is not smooth as predicted. Styled SAE can learn without mask and produce same quality fake if you choose high blur value in converter.")
        else:
            self.options['resolution'] = self.options.get('resolution', default_resolution)
            self.options['face_type'] = self.options.get('face_type', default_face_type)
            #self.options['archi'] = self.options.get('archi', default_archi)
            #self.options['lighter_encoder'] = self.options.get('lighter_encoder', False)
            #self.options['learn_mask'] = self.options.get('learn_mask', True)
            
        '''
        default_face_style_power = 10.0
        if is_first_run or ask_override:
            default_face_style_power = default_face_style_power if is_first_run else self.options.get('face_style_power', default_face_style_power)
            self.options['face_style_power'] = np.clip ( input_number("Face style power ( 0.0 .. 100.0 ?:help skip:%.2f) : " % (default_face_style_power), default_face_style_power, help_message="How fast NN will learn dst face style during generalization of src and dst faces. If style is learned good enough, set this value to 0.01 to prevent artifacts appearing."), 0.0, 100.0 )            
        else:
            self.options['face_style_power'] = self.options.get('face_style_power', default_face_style_power)
        
        default_bg_style_power = 10.0        
        if is_first_run or ask_override: 
            default_bg_style_power = default_bg_style_power if is_first_run else self.options.get('bg_style_power', default_bg_style_power)
            self.options['bg_style_power'] = np.clip ( input_number("Background style power ( 0.0 .. 100.0 ?:help skip:%.2f) : " % (default_bg_style_power), default_bg_style_power, help_message="How fast NN will learn dst background style during generalization of src and dst faces. If style is learned good enough, set this value to 0.1-0.3 to prevent artifacts appearing."), 0.0, 100.0 )            
        else:
            self.options['bg_style_power'] = self.options.get('bg_style_power', default_bg_style_power)
        
        if is_first_run or ask_override: 
            default_pixel_loss = False if is_first_run else self.options.get('pixel_loss', False)
            self.options['pixel_loss'] = input_bool ("Use pixel loss? (y/n, ?:help skip: n/default ) : ", default_pixel_loss, help_message="Default DSSIM loss good for initial understanding structure of faces. Use pixel loss after 30-40k epochs to enhance fine details.")
        else:
            self.options['pixel_loss'] = self.options.get('pixel_loss', False)
            
        default_ae_dims = 256 if self.options['archi'] == 'liae' else 512
        default_ed_ch_dims = 42
        if is_first_run:
            self.options['ae_dims'] = np.clip ( input_int("AutoEncoder dims (32-1024 ?:help skip:%d) : " % (default_ae_dims) , default_ae_dims, help_message="More dims are better, but requires more VRAM. You can fine-tune model size to fit your GPU." ), 32, 1024 )
            self.options['ed_ch_dims'] = np.clip ( input_int("Encoder/Decoder dims per channel (21-85 ?:help skip:%d) : " % (default_ed_ch_dims) , default_ed_ch_dims, help_message="More dims are better, but requires more VRAM. You can fine-tune model size to fit your GPU." ), 21, 85 )
        else:
            self.options['ae_dims'] = self.options.get('ae_dims', default_ae_dims)
            self.options['ed_ch_dims'] = self.options.get('ed_ch_dims', default_ed_ch_dims)
        '''    
        
    #override
    def onInitialize(self, batch_size=-1, **in_options):
        exec(nnlib.code_import_all, locals(), globals())

        self.set_vram_batch_requirements({6:8})
        
        resolution = self.options['resolution']
        bgr_shape = (resolution, resolution, 3)
        ngf = 64
        npf = 64
        ndf = 64
        lambda_A = 10
        lambda_B = 10
        
        #self.set_batch_size(created_batch_size)
        
        use_batch_norm = False #created_batch_size > 1
        #self.GA = modelify(ResNet (bgr_shape[2], use_batch_norm, n_blocks=6, ngf=ngf, use_dropout=True))(Input(bgr_shape))
        #self.GB = modelify(ResNet (bgr_shape[2], use_batch_norm, n_blocks=6, ngf=ngf, use_dropout=True))(Input(bgr_shape))
        #self.GA = modelify(UNet (bgr_shape[2], use_batch_norm, num_downs=get_power_of_two(resolution)-1, ngf=ngf, use_dropout=True))(Input(bgr_shape))
        #self.GB = modelify(UNet (bgr_shape[2], use_batch_norm, num_downs=get_power_of_two(resolution)-1, ngf=ngf, use_dropout=True))(Input(bgr_shape))
        
        self.G = modelify ( self.GeneratorFlow(resolution, 3, ae_dims=512, ed_ch_dims=42 ) ) (Input(bgr_shape))
        #import code
        #code.interact(local=dict(globals(), **locals()))
        
        self.DA = modelify(NLayerDiscriminator(use_batch_norm, ndf=ndf, n_layers=3) ) (Input(bgr_shape))
        self.DB = modelify(NLayerDiscriminator(use_batch_norm, ndf=ndf, n_layers=3) ) (Input(bgr_shape))

        if not self.is_first_run():
            self.G.load_weights (self.get_strpath_storage_for_file(self.GH5))
            #self.GA.load_weights (self.get_strpath_storage_for_file(self.GAH5))
            #self.GB.load_weights (self.get_strpath_storage_for_file(self.GBH5))
            self.DA.load_weights (self.get_strpath_storage_for_file(self.DAH5))            
            self.DB.load_weights (self.get_strpath_storage_for_file(self.DBH5))
            
        real_A0 = Input(bgr_shape, name="real_A0")        
        real_B0 = Input(bgr_shape, name="real_B0")
        
        DA_ones =  K.ones ( K.int_shape(self.DA.outputs[0])[1:] )
        DB_ones = K.ones ( K.int_shape(self.DB.outputs[0])[1:] )

        def CycleLoss (t1,t2):
            return K.mean(K.square(t1 - t2))        
            
        #fake_B0 = self.GA(real_A0)        
        #fake_A0 = self.GB(real_B0)        
        #rec_FB0 = self.GA(fake_A0)        
        #rec_FA0 = self.GB(fake_B0)
        fake_B0 = self.G(real_A0)        
        fake_A0 = self.G(real_B0)        
        rec_FB0 = self.G(fake_A0)        
        rec_FA0 = self.G(fake_B0)
        
        #loss_GA = K.mean(K.square(self.DB(fake_B0))) + lambda_A * ( CycleLoss(rec_FA0, real_A0) )        
        #loss_GB = K.mean(K.square(self.DA(fake_A0))) + lambda_B * ( CycleLoss(rec_FB0, real_B0) ) 
        loss_GA = K.mean(K.square(self.DB(fake_B0))) + lambda_A * ( CycleLoss(rec_FA0, real_A0) )        
        loss_GB = K.mean(K.square(self.DA(fake_A0))) + lambda_B * ( CycleLoss(rec_FB0, real_B0) ) 
                 
        if self.is_training_mode:
            def optimizer():
                return Adam(lr=2e-4, beta_1=0.5, beta_2=0.999)

            #self.GA_train = K.function ([real_A0, real_B0],[loss_GA], optimizer().get_updates(loss_GA, self.GA.trainable_weights + self.GB.trainable_weights) )
            #self.GB_train = K.function ([real_A0, real_B0],[loss_GB], optimizer().get_updates(loss_GB, self.GA.trainable_weights + self.GB.trainable_weights) )
            self.GA_train = K.function ([real_A0, real_B0],[loss_GA], optimizer().get_updates(loss_GA, self.G.trainable_weights) )
            self.GB_train = K.function ([real_A0, real_B0],[loss_GB], optimizer().get_updates(loss_GB, self.G.trainable_weights) )
            
            
            ###########
            
            loss_D_A = ( K.mean(K.square( self.DA(real_A0) )) + \
                         K.mean(K.square( self.DA(fake_A0) - DA_ones)) ) * 0.5

            self.DA_train = K.function ([real_A0, real_B0],[loss_D_A],
                                         optimizer().get_updates(loss_D_A, self.DA.trainable_weights) )
            
            ############
            
            loss_D_B = ( K.mean(K.square( self.DB(real_B0) )) + \
                         K.mean(K.square( self.DB(fake_B0) - DB_ones)) ) * 0.5
                          
            self.DB_train = K.function ([real_A0, real_B0],[loss_D_B],
                                         optimizer().get_updates(loss_D_B, self.DB.trainable_weights) )
            
            ############
            self.G_view = K.function([real_A0, real_B0],[fake_A0, fake_B0, rec_FA0, rec_FB0])
        else:
            self.G_convert = K.function([real_B0],[fake_A0])
 
        if self.is_training_mode:
            f = SampleProcessor.TypeFlags            
            face_type = f.FACE_ALIGN_FULL if self.options['face_type'] == 'f' else f.FACE_ALIGN_HALF
            self.set_training_data_generators ([            
                    SampleGeneratorFace(self.training_data_src_path, sort_by_yaw_target_samples_path=self.training_data_dst_path if self.sort_by_yaw else None, 
                                                                     debug=self.is_debug(), batch_size=self.batch_size, 
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, normalize_tanh = True, scale_range=np.array([-0.05, 0.05])+self.src_scale_mod / 100.0 ), 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | face_type | f.MODE_BGR, resolution],                        
                                              [f.SOURCE | face_type | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | face_type | f.MODE_M | f.FACE_MASK_FULL, resolution]                                              
                                            ] ),
                                              
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, normalize_tanh = True), 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | face_type | f.MODE_BGR, resolution],                                             
                                              [f.SOURCE | face_type | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | face_type | f.MODE_M | f.FACE_MASK_FULL, resolution]                                               
                                            ] )
                ])
        
    #override
    def onSave(self):
        self.save_weights_safe( [[self.G,    self.get_strpath_storage_for_file(self.GH5)],
                                 #[self.GA,    self.get_strpath_storage_for_file(self.GAH5)],
                                 #[self.GB,    self.get_strpath_storage_for_file(self.GBH5)],
                                 [self.DA,    self.get_strpath_storage_for_file(self.DAH5)],
                                 [self.DB,    self.get_strpath_storage_for_file(self.DBH5)] ])
        
    #override
    def onTrainOneEpoch(self, sample, generators_list):
        warped_src, target_src, target_src_mask = sample[0]
        warped_dst, target_dst, target_dst_mask = sample[1]  
     
        feed = [target_src, target_dst]
        loss_GA, = self.GA_train( feed )
        loss_GB, = self.GB_train( feed )
        loss_DA, = self.DA_train( feed )
        loss_DB, = self.DB_train( feed )
        return ( ('GA', loss_GA), ('GB', loss_GB), ('DA', loss_DA),  ('DB', loss_DB)  )

    #override
    def onGetPreview(self, sample):
        test_A0   = sample[0][1][0] #first sample only
        test_B0   = sample[1][1][0] #first sample only
        test_A0 = np.expand_dims(test_A0, 0)
        test_B0 = np.expand_dims(test_B0, 0)
        test_A0, test_B0, fake_A0, fake_B0, rec_A0, rec_B0 = [ x[0] / 2 + 0.5 for x in [test_A0,test_B0] + self.G_view([test_A0, test_B0])  ]        

        
        r = np.concatenate ((np.concatenate ( (test_A0, fake_B0, rec_A0), axis=1),
                             np.concatenate ( (test_B0, fake_A0, rec_B0), axis=1)
                             ), axis=0)                            
                
        return [ ('GAN', r ) ]
    
    def predictor_func (self, face):
        x = self.G_convert ( [ np.expand_dims(face *2 - 1,0)]  )[0]
        return x[0] / 2 + 0.5
        
    #override
    def get_converter(self, **in_options):
        from models import ConverterImage                   
        return ConverterImage(self.predictor_func, 
                              predictor_input_size=self.options['created_resolution'], 
                              output_size=self.options['created_resolution'], 
                              **in_options)

    @staticmethod
    def GeneratorFlow(resolution, output_nc, ae_dims=512, ed_ch_dims=42):
        exec (nnlib.import_all(), locals(), globals())
        k_size = 5
        strides = 2
        lowest_dense_res = resolution // 16
        
        d_dims = output_nc * ed_ch_dims
        
        def Conv2D (filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer=RandomNormal(0, 0.02), bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
            return keras.layers.Conv2D( filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint )

        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)(Conv2D(dim, k_size, strides=strides, padding='same')(x))
            return func 
            
        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func   
    
        def to_bgr ():
            def func(x):
                return Conv2D(output_nc, kernel_size=5, padding='same', activation='tanh')(x)
            return func
            
        def func(input):     
            x = input
            
            e_dims = K.int_shape(input)[-1]*ed_ch_dims            
            x = downscale(e_dims)(x)            
            x = downscale(e_dims*2)(x)
            x = downscale(e_dims*4)(x)
            x = downscale(e_dims*8)(x)    
            x = Dense(ae_dims)(Flatten()(x))
            x = Dense(lowest_dense_res * lowest_dense_res * ae_dims)(x)
            x = Reshape((lowest_dense_res, lowest_dense_res, ae_dims))(x)
            x = upscale(ae_dims)(x)
            x1     = upscale(d_dims*4)( x )       
            x2     = upscale(d_dims*2)( x1 )   
            x3     = upscale(d_dims*1)( x2 )

            return [ to_bgr() ( x3 ) ]            
            
        return func