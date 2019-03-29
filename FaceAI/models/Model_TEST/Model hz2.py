from models import ModelBase
import numpy as np
import cv2
from mathlib import get_power_of_two
from nnlib import nnlib
from facelib import FaceType
from samples import *
from utils.console_utils import *

class TestModel(ModelBase):
    GH5 = 'G.h5'
    GAH5 = 'GA.h5'
    DZH5 = 'DZ.h5'
    DAH5 = 'DA.h5'
    GBH5 = 'GB.h5'
    DBH5 = 'DB.h5'
    encH5 = 'enc.h5'
    decH5 = 'dec.h5'
    PYEH5 = 'pye.h5'

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

        self.set_vram_batch_requirements({6:64})

        resolution = self.options['resolution']
        bgr_shape = (resolution, resolution, 3)
        mask_shape = (resolution, resolution, 1)
        pitch_yaw_shape = (2, )
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



        #resnet = keras.applications.ResNet50 (include_top=False, weights=None, input_tensor=None, input_shape=(resolution, resolution, 3), pooling='avg')
        #prediction = Dense(units=2, kernel_initializer="he_normal", use_bias=False, activation="softmax")(resnet.output)

        #self.enc = Model(inputs=resnet.input, outputs=prediction)
        ae_dims = 256
        self.enc = modelify ( TestModel.EncFlow(resolution, ae_dims=ae_dims, ed_ch_dims=21 ) ) ( Input(bgr_shape) )
        self.dec = modelify ( TestModel.DecFlow(resolution, 3, ae_dims=ae_dims, ed_ch_dims=21 ) ) ( Input( (ae_dims,) ) )

        #self.dec = modelify ( TestModel.DecFlow(resolution, 3, ae_dims=ae_dims, ed_ch_dims=21 ) ) ( [Input( (ae_dims,) ), Input(pitch_yaw_shape) ] )

        self.pye = modelify ( TestModel.PYLatentFlow() ) ( Input( (ae_dims,) ) )

        #dec_Inputs = [ Input(K.int_shape(x)[1:]) for x in self.enc.outputs ]
        #self.DZ = modelify(TestModel.ZDiscriminatorFlow()) ( Input( (ae_dims,) ) )
        #self.DA = modelify(TestModel.DiscriminatorFlow()) ( [Input(bgr_shape), Input(pitch_yaw_shape)] )

        #import code
        #code.interact(local=dict(globals(), **locals()))

        #self.DA = modelify(TestModel.DiscriminatorFlow()) ( [Input(bgr_shape), Input(pitch_yaw_shape)] )
        #self.DB = modelify(NLayerDiscriminator(use_batch_norm, ndf=ndf, n_layers=3) ) (Input(bgr_shape))

        if not self.is_first_run():
            self.enc.load_weights (self.get_strpath_storage_for_file(self.encH5))
            self.dec.load_weights (self.get_strpath_storage_for_file(self.decH5))
            #self.pye.load_weights (self.get_strpath_storage_for_file(self.PYEH5))
            #self.DZ.load_weights (self.get_strpath_storage_for_file(self.DZH5))
            #self.DA.load_weights (self.get_strpath_storage_for_file(self.DAH5))
            #self.G.load_weights (self.get_strpath_storage_for_file(self.GH5))
            #self.GA.load_weights (self.get_strpath_storage_for_file(self.GAH5))
            #self.GB.load_weights (self.get_strpath_storage_for_file(self.GBH5))
            #self.DA.load_weights (self.get_strpath_storage_for_file(self.DAH5))
            #self.DB.load_weights (self.get_strpath_storage_for_file(self.DBH5))




        warped_A0 = Input(bgr_shape)
        real_A0 = Input(bgr_shape)
        mask_A0 = Input(mask_shape)
        real_pitch_A0 = Input( (1,), name="real_pitch_A0")
        real_yaw_A0 = Input( (1,), name="real_yaw_A0" )

        #real_pitch_yaw_A0 = Input( (2,))
        real_pitch_yaw_A0 = K.concatenate([real_pitch_A0, real_yaw_A0])
        #zero_pitch_yaw_A0 = K.zeros_like(real_pitch_yaw_A0)
        #pitch_yaw_random_t = Input( (2,))

        code_A0 = self.enc(warped_A0)
        rec_pitch_yaw_A0 = self.pye(code_A0)
        rec_A0 = self.dec(code_A0)

        mask = mask_A0/2.0 + 0.5

        loss_PY = K.mean(K.square(rec_pitch_yaw_A0 - real_pitch_yaw_A0))
        loss_G = K.mean( 100*K.square(tf_dssim(2.0)( (real_A0+1)*mask, (rec_A0+1)*mask )) )


        #import code
        #code.interact(local=dict(globals(), **locals()))
        #
        #rec_A0 = self.dec([code_A0, real_pitch_yaw_A0])
        #rec_random_A0 = self.dec([code_A0, pitch_yaw_random_t])
        #
        #mask = mask_A0/2.0 + 0.5
        #
        #loss_G = K.mean( 100*K.square(tf_dssim(2.0)( (real_A0+1)*mask, (rec_A0+1)*mask )) ) +\
        #         K.mean(K.square( self.DZ(code_A0) - K.ones ( K.int_shape(self.DZ.outputs[0])[1:] ) ) ) + \
        #         K.mean(K.square( self.DA([ (rec_random_A0+1)*mask, pitch_yaw_random_t]) - K.ones ( K.int_shape(self.DA.outputs[0])[1:] ) ) )
        #
        #         #K.mean(K.square( self.DA([ (rec_A0+1)*mask, real_pitch_yaw_A0]) - K.ones ( K.int_shape(self.DA.outputs[0])[1:] ) ) )
        #
        #loss_D = K.mean(K.square( self.DA([ (rec_random_A0+1)*mask, pitch_yaw_random_t]) )) + \
        #         K.mean(K.square( self.DA([ (real_A0+1)*mask, real_pitch_yaw_A0]) - K.ones ( K.int_shape(self.DA.outputs[0])[1:] ) )) +\
        #         K.mean(K.square( self.DZ(code_A0) )) + \
        #         K.mean(K.square( self.DZ(K.random_uniform( K.shape(code_A0), -1.0, 1.0)) - K.ones ( K.int_shape(self.DZ.outputs[0])[1:] ) ))
        #         #K.mean(K.square( self.DA([ (rec_A0+1)*mask, real_pitch_yaw_A0]) ))

        #rec_random_A0 = self.dec([code_A0, pitch_yaw_random_t])
        #zero_pitch_yaw_rec_A0 = self.dec([code_A0, zero_pitch_yaw_A0])

        #loss_PYE = K.mean(K.abs( self.pye(real_A0) - real_pitch_yaw_A0 ) )
        #pye_preds = self.pye(real_A0)
        #loss_PYE = K.mean ( K.categorical_crossentropy ( real_pitch_yaw_A0, pye_preds) )
        #metric_PYE = K.mean ( keras.metrics.categorical_accuracy (real_pitch_yaw_A0, pye_preds) )

        #loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

        #import code
        #code.interact(local=dict(globals(), **locals()))


        #batch_size = K.shape(z_mean)[0]
        ##rec_loss = K.mean(K.binary_crossentropy(
        ##        K.reshape( real_A0, (batch_size, np.prod(bgr_shape) )),
        ##        K.reshape( rec_A0 , (batch_size, np.prod(bgr_shape) ))
        ##        ), axis=-1)
        #mask = mask_A0/2.0 + 0.5
        #rec_loss = K.mean( 100*K.square(tf_dssim(2.0)( (real_A0+1)*mask , (rec_A0+1)*mask )) )
        #kl_loss = - 0.5 * K.sum(1. + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        #loss_G = K.mean(rec_loss + kl_loss)
        #
        #rec_random_A0 = self.dec([z, pitch_yaw_random_t])
        #zero_pitch_yaw_rec_A0 = self.dec([z, zero_pitch_yaw_A0])

        #import code
        #code.interact(local=dict(globals(), **locals()))

        #code_A0 = self.enc(warped_A0)
        #rec_A0 = self.dec( K.concatenate ([real_pitch_yaw_A0, code_A0[:,2:]]) )
        ##rec_A0 = self.dec( code_A
        #rec_random_A0 = self.dec( K.concatenate ([pitch_yaw_random_t, code_A0[:,2:]]) )
        #mask = mask_A0/2.0 + 0.5
        #loss_G = K.mean( 100*K.square(tf_dssim(2.0)( (real_A0+1)*mask , (rec_A0+1)*mask )) ) +\
        #         K.mean(K.square( self.DA([ (rec_random_A0+1)*mask, pitch_yaw_random_t]) ) )
        ##rec_d = self.DA([rec_A0, real_pitch_yaw_A0])

        #zero_pitch_yaw_code_A0 = K.concatenate( [ K.zeros_like(real_pitch_yaw_A0), code_A0[:,2:]] )
        #zero_pitch_yaw_rec_A0 = self.dec(zero_pitch_yaw_code_A0)

        #import code
        #code.interact(local=dict(globals(), **locals()))

        #loss_G = K.categorical_crossentropy (real_yaw_A0, code_A0[:,1:2] )
        #metric_mae = K.mean( K.abs(real_pitch_A0 - code_A0[:,0:1])) +
        #metric_mae = K.mean( K.abs(real_yaw_A0 - code_A0[:,1:2]))

        #K.mean( 1*K.square(real_pitch_A0 - code_A0[:,0:1])) + K.mean( 1*K.square(real_yaw_A0 - code_A0[:,1:2]))


        #code_rec_A0 = self.enc(rec_A0)


        #loss_G += K.mean( 10*K.square( K.zeros_like(code_zero_pitch_yaw_rec_A0)[:,0:2] - code_zero_pitch_yaw_rec_A0[:,0:2]))



        if self.is_training_mode:
            def optimizer():
                return Adam(lr=2e-4, beta_1=0.5, beta_2=0.999)

            #self.GA_train = K.function ([real_A0, real_B0],[loss_GA], optimizer().get_updates(loss_GA, self.GA.trainable_weights + self.GB.trainable_weights) )
            #self.GB_train = K.function ([real_A0, real_B0],[loss_GB], optimizer().get_updates(loss_GB, self.GA.trainable_weights + self.GB.trainable_weights) )
            #self.PYE_train = K.function ([real_A0, real_pitch_A0, real_yaw_A0],[metric_PYE], optimizer().get_updates(loss_PYE, self.pye.trainable_weights) )


            self.G_train = K.function ([warped_A0, real_A0, mask_A0, real_pitch_A0, real_yaw_A0],[loss_G], optimizer().get_updates(loss_G, self.enc.trainable_weights + self.dec.trainable_weights ) )
            self.P_train = K.function ([warped_A0, real_A0, mask_A0, real_pitch_A0, real_yaw_A0],[loss_PY], optimizer().get_updates(loss_PY, self.pye.trainable_weights) )
            self.P_test = K.function ([warped_A0, real_A0, mask_A0, real_pitch_A0, real_yaw_A0],[loss_PY])

            #self.D_train = K.function ([warped_A0, real_A0, mask_A0, real_pitch_A0, real_yaw_A0, pitch_yaw_random_t],[loss_D], optimizer().get_updates(loss_D, self.DZ.trainable_weights + self.DA.trainable_weights ) )
            #self.dec_train = K.function ([warped_A0, real_A0, mask_A0, real_pitch_A0, real_yaw_A0],[loss_G], optimizer().get_updates(loss_G, self.dec.trainable_weights ) )
            #self.GB_train = K.function ([real_A0, real_B0],[loss_GB], optimizer().get_updates(loss_GB, self.G.trainable_weights) )

            #import code
            #code.interact(local=dict(globals(), **locals()))
            ############
            #
            #loss_D_A = ( K.mean(K.square( self.DA(real_A0) )) + \
            #             K.mean(K.square( self.DA(fake_A0) - DA_ones)) ) * 0.5
            #
            #self.DA_train = K.function ([real_A0, real_B0],[loss_D_A],
            #                             optimizer().get_updates(loss_D_A, self.DA.trainable_weights) )
            #
            #############
            #
            #loss_D_B = ( K.mean(K.square( self.DB(real_B0) )) + \
            #             K.mean(K.square( self.DB(fake_B0) - DB_ones)) ) * 0.5
            #
            #self.DB_train = K.function ([real_A0, real_B0],[loss_D_B],
            #                             optimizer().get_updates(loss_D_B, self.DB.trainable_weights) )
            #
            ############
            self.G_view = K.function([warped_A0, real_pitch_A0, real_yaw_A0],[rec_A0])#, rec_random_A0, zero_pitch_yaw_rec_A0
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
                                              [f.TRANSFORMED | face_type | f.MODE_BGR, resolution],
                                              [f.TRANSFORMED | face_type | f.MODE_M | f.FACE_MASK_FULL, resolution]
                                            ], add_pitch=True, add_yaw=True ),

                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, normalize_tanh = True),
                        output_sample_types=[ [f.WARPED_TRANSFORMED | face_type | f.MODE_BGR, resolution],
                                              [f.TRANSFORMED | face_type | f.MODE_BGR, resolution],
                                              [f.TRANSFORMED | face_type | f.MODE_M | f.FACE_MASK_FULL, resolution]
                                            ], add_pitch=True, add_yaw=True )
                ])

    #override
    def onSave(self):
        self.save_weights_safe( [[self.pye,    self.get_strpath_storage_for_file(self.PYEH5)],
                                 [self.enc,    self.get_strpath_storage_for_file(self.encH5)],
                                 [self.dec,    self.get_strpath_storage_for_file(self.decH5)],
                                 #[self.DA,    self.get_strpath_storage_for_file(self.DAH5)],
                                 #[self.DZ,    self.get_strpath_storage_for_file(self.DZH5)],
                                 ])

    #override
    def onTrainOneEpoch(self, sample, generators_list):
        warped_src, target_src, target_src_mask, src_pitch, src_yaw = sample[0]
        warped_dst, target_dst, target_dst_mask, dst_pitch, dst_yaw = sample[1]

        #src_pitch = (src_pitch + 1) / 2.0
        #src_yaw = (src_yaw + 1) / 2.0

        src_random_pitch_yaw = np.random.uniform( size=( len(warped_src) ,2) )*2 -1

        feed = [warped_src, target_src, target_src_mask, src_pitch, src_yaw]#, src_random_pitch_yaw]
        #loss_G, = self.G_train( feed )
        loss_G = 0
        loss_PY, = self.P_train( feed )

        loss_test_PY, = self.P_test ( [target_dst, target_dst, target_dst_mask, dst_pitch, dst_yaw] )
        #loss_PY = 0
        #
        return ( ('G', loss_G), ('PY', loss_PY), ('test_PY', loss_test_PY) )

        #import code
        #code.interact(local=dict(globals(), **locals()))

        #loss_G, = self.G_train( feed )
        #loss_D, = self.D_train( feed )
        #
        #return ( ('G', loss_G), ('D', loss_D), )

    #override
    def onGetPreview(self, sample):
        test_A0       = sample[0][1][0:4]
        test_A0_pitch = sample[0][3][0:4]
        test_A0_yaw   = sample[0][4][0:4]

        #test_B0   = sample[1][1][0] #first sample only
        #test_B0 = np.expand_dims(test_B0, 0)

        #import code
        #code.interact(local=dict(globals(), **locals()))

        #rec_random_A0, zero_pitch_yaw_rec_A0
        test_A0, rec_A0,  = [ x / 2 + 0.5 for x in [batch for batch in [test_A0] + self.G_view([test_A0, test_A0_pitch, test_A0_yaw, np.random.uniform( size=( len(test_A0) ,2) )*2 -1 ])  ]  ]

        st = []
        for i in range(len(test_A0)):
            st += [ np.concatenate ( (test_A0[i], rec_A0[i], ), axis=1) ] #rec_random_A0[i], zero_pitch_yaw_rec_A0[i]

        st = np.concatenate (st, axis=0)
        return [ ('TEST', st ) ]

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
    def PYLatentFlow():
        exec (nnlib.import_all(), locals(), globals())
        k_size = 5
        strides = 2

        def func(input):
            x = input
            x = Dense (512, kernel_regularizer='l2') (x)
            x = Dense (256, kernel_regularizer='l2') (x)
            x = Dense (128, kernel_regularizer='l2') (x)
            x = Dense (64, kernel_regularizer='l2') (x)
            x = Dense (32, kernel_regularizer='l2') (x)
            x = Dense (16, kernel_regularizer='l2') (x)
            x = Dense (8, kernel_regularizer='l2') (x)
            x = Dense (4, kernel_regularizer='l2') (x)
            x = Dense (2, activation='tanh') (x)
            return x

        return func

    @staticmethod
    def EncFlow(resolution, ae_dims=512, ed_ch_dims=42):
        exec (nnlib.import_all(), locals(), globals())
        k_size = 5
        strides = 2
        lowest_dense_res = resolution // 16


        use_bias = True
        def XNormalization(x):
            return x
        #def XNormalization(x):
        #    return BatchNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)

        def Conv2D (filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias, kernel_initializer=RandomNormal(0, 0.02), bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
            return keras.layers.Conv2D( filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint )

        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)(XNormalization(Conv2D(dim, k_size, strides=strides, padding='same')(x)))
            return func

        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(XNormalization(Conv2D(dim * 4, 3, strides=1, padding='same')(x))))
            return func


        def func(input):
            x = input

            e_dims = K.int_shape(x)[-1]*ed_ch_dims

            x = downscale(e_dims)(x)
            x = downscale(e_dims*2)(x)
            x = downscale(e_dims*4)(x)
            x = downscale(e_dims*8)(x)

            x = Flatten()(x)
            x = Dense (ae_dims, activation='tanh')(x)


            return x

            #x = Conv2D(e_dims, kernel_size=3, strides=1, padding='same')(x)
            #x = MaxPooling2D(strides=2, padding='same')(x)
            #x = Conv2D(e_dims*2, kernel_size=3, strides=1, padding='same')(x)
            #x = MaxPooling2D(strides=2, padding='same')(x)
            #x = Conv2D(e_dims*4, kernel_size=3, strides=1, padding='same')(x)
            #x = MaxPooling2D(strides=2, padding='same')(x)
            #x = Conv2D(e_dims*8, kernel_size=3, strides=1, padding='same')(x)
            #x = MaxPooling2D(strides=2, padding='same')(x)


            #z_mean = Dense(ae_dims)(x)
            #z_log_var = Dense(ae_dims)(x)
            #
            #def sampler(args):
            #    z_mean, z_log_var = args
            #    batch_size = K.shape(z_mean)[0]
            #    dim = K.shape(z_mean)[1]
            #    epsilon = K.random_normal(shape=(batch_size, dim))
            #    return z_mean + K.exp(0.5 * z_log_var) * epsilon
            #
            #z = Lambda(sampler, output_shape=(ae_dims,))([z_mean, z_log_var])

            #import code
            #code.interact(local=dict(globals(), **locals()))



        return func

    @staticmethod
    def DecFlow(resolution, output_nc, ae_dims=512, ed_ch_dims=42):
        exec (nnlib.import_all(), locals(), globals())
        k_size = 5
        strides = 2
        lowest_dense_res = resolution // 16

        d_dims = output_nc * ed_ch_dims

        use_bias = True
        def XNormalization(x):
            return x

        #def XNormalization(x):
        #    return BatchNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)

        def Conv2D (filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias, kernel_initializer=RandomNormal(0, 0.02), bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
            return keras.layers.Conv2D( filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint )

        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)(XNormalization(Conv2D(dim, k_size, strides=strides, padding='same')(x)))
            return func

        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()( LeakyReLU(0.1)(XNormalization(Conv2D(dim * 4, 3, strides=1, padding='same')(x))))
            return func

        def to_bgr ():
            def func(x):
                return Conv2D(output_nc, kernel_size=5, padding='same', activation='tanh')(x)
            return func

        def func(input):
            #x, lbls = input
            x = input
            #x = Concatenate(axis=-1)([x, lbls])

            x = Dense(lowest_dense_res * lowest_dense_res * ae_dims)(x)
            x = Reshape((lowest_dense_res, lowest_dense_res, ae_dims))(x)
            x = upscale(ae_dims)(x)
            x = upscale(d_dims*4)(x)
            x = upscale(d_dims*2)(x)
            x = upscale(d_dims*1)(x)

            return [ to_bgr() ( x ) ]

        return func

    @staticmethod
    def ZDiscriminatorFlow():
        exec (nnlib.import_all(), locals(), globals())

        def func(input):
            x = input
            x = Dense(64, activation='relu')(x)
            x = Dense(32, activation='relu')(x)
            x = Dense(16, activation='relu')(x)
            x = Dense(1)(x)
            return x
        return func

    @staticmethod
    def DiscriminatorFlow():
        exec (nnlib.import_all(), locals(), globals())


        use_bias = True
        def XNormalization(x):
            return x
        #def XNormalization(x):
        #    return BatchNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)

        def Conv2D (filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias, kernel_initializer=RandomNormal(0, 0.02), bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
            return keras.layers.Conv2D( filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint )

        def func(inputs):
            input_layer, condition_layer = inputs

            x = input_layer
            x = Conv2D(16, kernel_size=5, strides=2, padding='same')(x)
            x = XNormalization(x)
            x = ReLU(0.2)(x)

            conv_dims = K.int_shape(x)[1]
            conv_chs = K.int_shape(x)[3]
            labels = K.int_shape(condition_layer)[1]

            def l(inputs):
                x, condition_layer = inputs
                x = K.concatenate( [x,
                K.reshape ( K.repeat(condition_layer, conv_dims*conv_dims),
                           (K.shape (condition_layer)[0],) + (conv_dims,)*2 + (labels,) )
                           ])
                return x

            x = Lambda( l, output_shape=(conv_dims,conv_dims,conv_chs+labels)  ) ( [x,condition_layer] )


            #import code
            #code.interact(local=dict(globals(), **locals()))

            x = Conv2D(32, kernel_size=5, strides=2, padding='same')(x)
            x = XNormalization(x)
            x = ReLU(0.2)(x)

            x = Conv2D(64, kernel_size=5, strides=2, padding='same')(x)
            x = XNormalization(x)
            x = ReLU(0.2)(x)

            x = Conv2D(128, kernel_size=5, strides=2, padding='same')(x)
            x = XNormalization(x)
            x = ReLU(0.2)(x)
            x = Flatten()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dense(1)(x)
            return x
        return func

    @staticmethod
    def PYEstimatorFlow():
        exec (nnlib.import_all(), locals(), globals())


        use_bias = True
        def XNormalization(x):
            return x
        #def XNormalization(x):
        #    return BatchNormalization (axis=3, gamma_initializer=RandomNormal(1., 0.02))(x)

        def Conv2D (filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias, kernel_initializer=RandomNormal(0, 0.02), bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
            return keras.layers.Conv2D( filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint )

        def func(inputs):

            x = inputs

            #x = LeakyReLU(0.1)(Conv2D(128, 5, strides=2, padding='same')(x))
            #x = LeakyReLU(0.1)(Conv2D(256, 5, strides=2, padding='same')(x))
            #x = LeakyReLU(0.1)(Conv2D(512, 5, strides=2, padding='same')(x))
            #x = LeakyReLU(0.1)(Conv2D(1024, 5, strides=2, padding='same')(x))
            #x = Flatten()(x)
            #x = Dense(1024)(x)
            #x = Dense(2, activation='softmax')(x)
            #return x

            x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
            x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
            x = MaxPooling2D(strides=2, padding='same')(x)

            x = Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
            x = Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
            x = MaxPooling2D(strides=2, padding='same')(x)

            x = Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
            x = Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
            x = Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
            x = MaxPooling2D(strides=2, padding='same')(x)

            x = Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
            x = Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
            x = Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
            x = MaxPooling2D(strides=2, padding='same')(x)

            x = Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
            x = Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
            x = Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
            x = MaxPooling2D(strides=2, padding='same')(x)

            x = Flatten()(x)
            x = Dense(4096, activation='relu')(x)
            x = Dense(4096, activation='relu')(x)
            x = Dense(2, activation='softmax')(x)
            return x
        return func

Model = TestModel
