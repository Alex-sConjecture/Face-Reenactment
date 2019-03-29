import numpy as np

from nnlib import nnlib
from models import ModelBase
from facelib import FaceType
from samples import *
from utils.console_utils import *

#SAE - Styled AutoEncoder
class SAEModel(ModelBase):

    encoderH5 = 'encoder.h5'    
    inter_BH5 = 'inter_B.h5'
    inter_ABH5 = 'inter_AB.h5'
    decoderH5 = 'decoder.h5'
    decodermH5 = 'decoderm.h5'
    
    decoder_srcH5 = 'decoder_src.h5'
    decoder_srcmH5 = 'decoder_srcm.h5'
    decoder_dstH5 = 'decoder_dst.h5'
    decoder_dstmH5 = 'decoder_dstm.h5'
    
    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        default_resolution = 128
        default_archi = 'df'
        default_face_type = 'f'
        
        if is_first_run:
            self.options['resolution'] = input_int("Resolution (64,128 ?:help skip:128) : ", default_resolution, [64,128], help_message="More resolution requires more VRAM.")
            self.options['archi'] = input_str ("AE architecture (df, liae, ?:help skip:%s) : " % (default_archi) , default_archi, ['df','liae'], help_message="DF keeps faces more natural, while LIAE can fix overly different face shapes.").lower()            
            self.options['lighter_encoder'] = input_bool ("Use lightweight encoder? (y/n, ?:help skip:n) : ", False, help_message="Lightweight encoder is 35% faster, requires less VRAM, sacrificing overall quality.")
        else:
            self.options['resolution'] = self.options.get('resolution', default_resolution)
            self.options['archi'] = self.options.get('archi', default_archi)
            self.options['lighter_encoder'] = self.options.get('lighter_encoder', False)

        default_face_style_power = 10.0
        if is_first_run or ask_override:
            default_face_style_power = default_face_style_power if is_first_run else self.options.get('face_style_power', default_face_style_power)
            self.options['face_style_power'] = np.clip ( input_number("Face style power ( 0.0 .. 100.0 ?:help skip:%.1f) : " % (default_face_style_power), default_face_style_power, help_message="How fast NN will learn dst face style during generalization of src and dst faces."), 0.0, 100.0 )            
        else:
            self.options['face_style_power'] = self.options.get('face_style_power', default_face_style_power)
        
        default_bg_style_power = 10.0        
        if is_first_run or ask_override: 
            default_bg_style_power = default_bg_style_power if is_first_run else self.options.get('bg_style_power', default_bg_style_power)
            self.options['bg_style_power'] = np.clip ( input_number("Background style power ( 0.0 .. 100.0 ?:help skip:%.1f) : " % (default_bg_style_power), default_bg_style_power, help_message="How fast NN will learn dst background style during generalization of src and dst faces."), 0.0, 100.0 )            
        else:
            self.options['bg_style_power'] = self.options.get('bg_style_power', default_bg_style_power)
            
        default_ae_dims = 256 if self.options['archi'] == 'liae' else 512
        default_ed_ch_dims = 42
        if is_first_run:
            self.options['ae_dims'] = np.clip ( input_int("AutoEncoder dims (128-1024 ?:help skip:%d) : " % (default_ae_dims) , default_ae_dims, help_message="More dims are better, but requires more VRAM. You can fine-tune model size to fit your GPU." ), 128, 1024 )
            self.options['ed_ch_dims'] = np.clip ( input_int("Encoder/Decoder dims per channel (21-85 ?:help skip:%d) : " % (default_ed_ch_dims) , default_ed_ch_dims, help_message="More dims are better, but requires more VRAM. You can fine-tune model size to fit your GPU." ), 21, 85 )
            
            if self.options['resolution'] != 64:
                self.options['adapt_k_size'] = input_bool("Use adaptive kernel size? (y/n, ?:help skip:n) : ", False, help_message="In some cases, adaptive kernel size can fix bad generalization, for example warping parts of face." )
            else:
                self.options['adapt_k_size'] = False
                
            self.options['face_type'] = input_str ("Half or Full face? (h/f, ?:help skip:f) : ", default_face_type, ['h','f'], help_message="Half face has better resolution, but covers less area of cheeks.").lower()            
        else:
            self.options['ae_dims'] = self.options.get('ae_dims', default_ae_dims)
            self.options['ed_ch_dims'] = self.options.get('ed_ch_dims', default_ed_ch_dims)
            self.options['adapt_k_size'] = self.options.get('adapt_k_size', False)
            self.options['face_type'] = self.options.get('face_type', default_face_type)
        
        

    #override
    def onInitialize(self, **in_options):
        exec(nnlib.import_all(), locals(), globals())

        self.set_vram_batch_requirements({2:1,3:2,4:3,5:6,6:8,7:12,8:16})
        
        resolution = self.options['resolution']
        ae_dims = self.options['ae_dims']
        ed_ch_dims = self.options['ed_ch_dims']
        adapt_k_size = self.options['adapt_k_size']
        bgr_shape = (resolution, resolution, 3)
        mask_shape = (resolution, resolution, 1)

        alpha = Input ( (1,) )
        
        warped_src = Input(bgr_shape)
        target_src = Input(bgr_shape)
        target_srcm = Input(mask_shape)
        
        warped_dst = Input(bgr_shape)
        target_dst = Input(bgr_shape)
        target_dstm = Input(mask_shape)
            
        if self.options['archi'] == 'liae':
            self.encoder = modelify(SAEModel.LIAEEncFlow(resolution, adapt_k_size, self.options['lighter_encoder'], ed_ch_dims=ed_ch_dims)  ) (Input(bgr_shape))
            
            enc_output_Inputs = [ Input(K.int_shape(x)[1:]) for x in self.encoder.outputs ] 
            
            self.inter_B = modelify(SAEModel.LIAEInterFlow(resolution, ae_dims=ae_dims)) (enc_output_Inputs)
            self.inter_AB = modelify(SAEModel.LIAEInterFlow(resolution, ae_dims=ae_dims)) (enc_output_Inputs)
            
            inter_output_Inputs = [ Input( np.array(K.int_shape(x)[1:])*(1,1,2) ) for x in self.inter_B.outputs ] 

            self.decoder = modelify(SAEModel.LIAEDecFlow (bgr_shape[2],ed_ch_dims=ed_ch_dims//2)) (inter_output_Inputs)
            self.decoderm = modelify(SAEModel.LIAEDecFlow (mask_shape[2],ed_ch_dims=int(ed_ch_dims/1.5) )) (inter_output_Inputs)
            
            
            if not self.is_first_run():
                self.encoder.load_weights  (self.get_strpath_storage_for_file(self.encoderH5))
                self.inter_B.load_weights  (self.get_strpath_storage_for_file(self.inter_BH5))
                self.inter_AB.load_weights (self.get_strpath_storage_for_file(self.inter_ABH5))
                self.decoder.load_weights (self.get_strpath_storage_for_file(self.decoderH5))
                self.decoderm.load_weights (self.get_strpath_storage_for_file(self.decodermH5))
     
            warped_src_code = self.encoder (warped_src)
            
            warped_src_inter_AB_code = self.inter_AB (warped_src_code)
            warped_src_inter_code = Concatenate()([warped_src_inter_AB_code,warped_src_inter_AB_code])
            
            pred_src_src = self.decoder(warped_src_inter_code)
            pred_src_srcm = self.decoderm(warped_src_inter_code)
            
  
            warped_dst_code = self.encoder (warped_dst)
            warped_dst_inter_B_code = self.inter_B (warped_dst_code)
            warped_dst_inter_AB_code = self.inter_AB (warped_dst_code)
            warped_dst_inter_code = Concatenate()([warped_dst_inter_B_code,warped_dst_inter_AB_code])
            pred_dst_dst = self.decoder(warped_dst_inter_code)
            pred_dst_dstm = self.decoderm(warped_dst_inter_code)
            
            warped_src_dst_inter_code = Concatenate()([warped_dst_inter_AB_code,warped_dst_inter_AB_code])
            pred_src_dst = self.decoder(warped_src_dst_inter_code)
            pred_src_dstm = self.decoderm(warped_src_dst_inter_code)
        else:
            self.encoder = modelify(SAEModel.DFEncFlow(resolution, adapt_k_size, self.options['lighter_encoder'], ae_dims=ae_dims, ed_ch_dims=ed_ch_dims)  ) (Input(bgr_shape))
            
            def outputsAsInputs(x):
                return [ Input(K.int_shape(x)[1:]) for x in x.outputs ] 

            self.decoders_src = SAEModel.DFBuildDecoders ( outputsAsInputs(self.encoder), bgr_shape[2], ed_ch_dims=ed_ch_dims//2 )
            self.decoders_dst = SAEModel.DFBuildDecoders ( outputsAsInputs(self.encoder), bgr_shape[2], ed_ch_dims=ed_ch_dims//2 )
            self.decoders_srcm = SAEModel.DFBuildDecoders ( outputsAsInputs(self.encoder), bgr_shape[2], ed_ch_dims=int(ed_ch_dims/1.5) )
            self.decoders_dstm = SAEModel.DFBuildDecoders ( outputsAsInputs(self.encoder), bgr_shape[2], ed_ch_dims=int(ed_ch_dims/1.5) )
            

            #self.decoder_src = modelify(SAEModel.DFDecFlow (bgr_shape[2],ed_ch_dims=ed_ch_dims//2)) (dec_Inputs)
            #self.decoder_dst = modelify(SAEModel.DFDecFlow (bgr_shape[2],ed_ch_dims=ed_ch_dims//2)) (dec_Inputs)
            
            #def decoder_stack(encoder, ed_ch_dims):
            #    dec_Inputs = [ Input(K.int_shape(x)[1:]) for x in encoder.outputs ] 
            #    result = []
            #    for i in range(4):                    
            #        result += [ modelify(SAEModel.DFDecFlow (bgr_shape[2], [0,8,4,2][i], ed_ch_dims=ed_ch_dims)) (dec_Inputs) ]
            #        dec_Inputs = [ Input(K.int_shape(x)[1:]) for x in result[-1].outputs ]
            #    return result
            #self.decoder_src = modelify(SAEModel.DFDecFlow (bgr_shape[2],ed_ch_dims=ed_ch_dims//2)) (dec_Inputs)
            #self.decoder_dst = modelify(SAEModel.DFDecFlow (bgr_shape[2],ed_ch_dims=ed_ch_dims//2)) (dec_Inputs)
            #self.decoder_srcm = modelify(SAEModel.DFDecFlow (mask_shape[2],ed_ch_dims=int(ed_ch_dims/1.5))) (dec_Inputs)
            #self.decoder_dstm = modelify(SAEModel.DFDecFlow (mask_shape[2],ed_ch_dims=int(ed_ch_dims/1.5))) (dec_Inputs)
            
            

            #if not self.is_first_run():
            #    self.encoder.load_weights      (self.get_strpath_storage_for_file(self.encoderH5))
            #    self.decoder_src.load_weights  (self.get_strpath_storage_for_file(self.decoder_srcH5))
            #    self.decoder_srcm.load_weights (self.get_strpath_storage_for_file(self.decoder_srcmH5))
            #    self.decoder_dst.load_weights  (self.get_strpath_storage_for_file(self.decoder_dstH5))
            #    self.decoder_dstm.load_weights (self.get_strpath_storage_for_file(self.decoder_dstmH5))
                
            warped_src_code = self.encoder (warped_src)
            
            pred_src_src_x1     = self.decoders_src[0](warped_src_code)
            pred_src_src_x1_bgr = self.decoders_src[1](pred_src_src_x1)            
            pred_src_src_x2     = self.decoders_src[2](pred_src_src_x1)
            pred_src_src_x2_bgr = self.decoders_src[3](pred_src_src_x2)            
            pred_src_src_x3     = self.decoders_src[4](pred_src_src_x2)
            pred_src_src_x3_bgr = self.decoders_src[5](pred_src_src_x3)            
            pred_src_src_x4     = self.decoders_src[6](pred_src_src_x3)
            pred_src_src_x4_bgr = self.decoders_src[7](pred_src_src_x4)            
            pred_src_src = [pred_src_src_x1_bgr, pred_src_src_x2_bgr, pred_src_src_x3_bgr, pred_src_src_x4_bgr]
            
            pred_src_srcm_x1     = self.decoders_srcm[0](warped_src_code)
            pred_src_srcm_x1_bgr = self.decoders_srcm[1](pred_src_srcm_x1)            
            pred_src_srcm_x2     = self.decoders_srcm[2](pred_src_srcm_x1)
            pred_src_srcm_x2_bgr = self.decoders_srcm[3](pred_src_srcm_x2)            
            pred_src_srcm_x3     = self.decoders_srcm[4](pred_src_srcm_x2)
            pred_src_srcm_x3_bgr = self.decoders_srcm[5](pred_src_srcm_x3)            
            pred_src_srcm_x4     = self.decoders_srcm[6](pred_src_srcm_x3)
            pred_src_srcm_x4_bgr = self.decoders_srcm[7](pred_src_srcm_x4)
            pred_src_srcm = [pred_src_srcm_x1_bgr, pred_src_srcm_x2_bgr, pred_src_srcm_x3_bgr, pred_src_srcm_x4_bgr]
            
            warped_dst_code = self.encoder (warped_dst)
            
            pred_dst_dst_x1     = self.decoders_dst[0](warped_dst_code)
            pred_dst_dst_x1_bgr = self.decoders_dst[1](pred_dst_dst_x1)            
            pred_dst_dst_x2     = self.decoders_dst[2](pred_dst_dst_x1)
            pred_dst_dst_x2_bgr = self.decoders_dst[3](pred_dst_dst_x2)            
            pred_dst_dst_x3     = self.decoders_dst[4](pred_dst_dst_x2)
            pred_dst_dst_x3_bgr = self.decoders_dst[5](pred_dst_dst_x3)            
            pred_dst_dst_x4     = self.decoders_dst[6](pred_dst_dst_x3)
            pred_dst_dst_x4_bgr = self.decoders_dst[7](pred_dst_dst_x4)
            pred_dst_dst = [pred_dst_dst_x1_bgr, pred_dst_dst_x2_bgr, pred_dst_dst_x3_bgr, pred_dst_dst_x4_bgr]
            
            pred_dst_dstm_x1     = self.decoders_dstm[0](warped_dst_code)
            pred_dst_dstm_x1_bgr = self.decoders_dstm[1](pred_dst_dstm_x1)            
            pred_dst_dstm_x2     = self.decoders_dstm[2](pred_dst_dstm_x1)
            pred_dst_dstm_x2_bgr = self.decoders_dstm[3](pred_dst_dstm_x2)            
            pred_dst_dstm_x3     = self.decoders_dstm[4](pred_dst_dstm_x2)
            pred_dst_dstm_x3_bgr = self.decoders_dstm[5](pred_dst_dstm_x3)            
            pred_dst_dstm_x4     = self.decoders_dstm[6](pred_dst_dstm_x3)
            pred_dst_dstm_x4_bgr = self.decoders_dstm[7](pred_dst_dstm_x4)
            pred_dst_dstm = [pred_dst_dstm_x1_bgr, pred_dst_dstm_x2_bgr, pred_dst_dstm_x3_bgr, pred_dst_dstm_x4_bgr]
            
            pred_src_dst_x1     = self.decoders_src[0](warped_dst_code)
            pred_src_dst_x1_bgr = self.decoders_src[1](pred_src_dst_x1)            
            pred_src_dst_x2     = self.decoders_src[2](pred_src_dst_x1)
            pred_src_dst_x2_bgr = self.decoders_src[3](pred_src_dst_x2)            
            pred_src_dst_x3     = self.decoders_src[4](pred_src_dst_x2)
            pred_src_dst_x3_bgr = self.decoders_src[5](pred_src_dst_x3)            
            pred_src_dst_x4     = self.decoders_src[6](pred_src_dst_x3)
            pred_src_dst_x4_bgr = self.decoders_src[7](pred_src_dst_x4)
            pred_src_dst = [pred_src_dst_x1_bgr, pred_src_dst_x2_bgr, pred_src_dst_x3_bgr, pred_src_dst_x4_bgr]
            
            pred_src_dstm_x1     = self.decoders_srcm[0](warped_dst_code)
            pred_src_dstm_x1_bgr = self.decoders_srcm[1](pred_src_dstm_x1)            
            pred_src_dstm_x2     = self.decoders_srcm[2](pred_src_dstm_x1)
            pred_src_dstm_x2_bgr = self.decoders_srcm[3](pred_src_dstm_x2)            
            pred_src_dstm_x3     = self.decoders_srcm[4](pred_src_dstm_x2)
            pred_src_dstm_x3_bgr = self.decoders_srcm[5](pred_src_dstm_x3)            
            pred_src_dstm_x4     = self.decoders_srcm[6](pred_src_dstm_x3)
            pred_src_dstm_x4_bgr = self.decoders_srcm[7](pred_src_dstm_x4)
            pred_src_dstm = [pred_src_dstm_x1_bgr, pred_src_dstm_x2_bgr, pred_src_dstm_x3_bgr, pred_src_dstm_x4_bgr]
            


        target_src_ar = [tf.image.resize_bicubic( target_src, (resolution // 8,)*2 ), 
                         tf.image.resize_bicubic( target_src, (resolution // 4,)*2 ),
                         tf.image.resize_bicubic( target_src, (resolution // 2,)*2 ),
                         target_src]

        target_srcm_ar = [tf.image.resize_bicubic( target_srcm, (resolution // 8,)*2 ),
                          tf.image.resize_bicubic( target_srcm, (resolution // 4,)*2 ),
                          tf.image.resize_bicubic( target_srcm, (resolution // 2,)*2 ),
                          target_srcm]

        target_dst_ar = [ tf.image.resize_bicubic( target_dst, (resolution // 8,)*2 ),
                          tf.image.resize_bicubic( target_dst, (resolution // 4,)*2 ),
                          tf.image.resize_bicubic( target_dst, (resolution // 2,)*2 ),
                          target_dst ]      

        
        target_dstm_ar = [ tf.image.resize_bicubic( target_dstm, (resolution // 8,)*2 ),
                           tf.image.resize_bicubic( target_dstm, (resolution // 4,)*2 ),
                           tf.image.resize_bicubic( target_dstm, (resolution // 2,)*2 ),       
                           target_dstm ]       

        target_srcm_blurred_ar = [ tf_gaussian_blur( max(1, x.get_shape().as_list()[1] // 32) )(x) for x in target_srcm_ar]
        #target_srcm_blurred = tf_gaussian_blur(resolution // 32)(target_srcm)   
        
        target_srcm_sigm_ar = [ x / 2.0 + 0.5 for x in target_srcm_blurred_ar] 
        #target_srcm_sigm = target_srcm_blurred / 2.0 + 0.5
        
        target_srcm_anti_sigm_ar = [ 1.0 - x for x in target_srcm_sigm_ar] 
        #target_srcm_anti_sigm = 1.0 - target_srcm_sigm
        
        target_dstm_blurred_ar = [ tf_gaussian_blur( max(1, x.get_shape().as_list()[1] // 32) )(x) for x in target_dstm_ar]
        #target_dstm_blurred = tf_gaussian_blur(resolution // 32)(target_dstm)
        
        target_dstm_sigm_ar = [ x / 2.0 + 0.5 for x in target_dstm_blurred_ar] 
        #target_dstm_sigm = target_dstm_blurred / 2.0 + 0.5
        
        target_dstm_anti_sigm_ar = [ 1.0 - x for x in target_dstm_sigm_ar] 
        #target_dstm_anti_sigm = 1.0 - target_dstm_sigm
        
        
 
        target_src_sigm_ar = [ x + 1 for x in target_src_ar]
        #target_src_sigm = target_src+1
        
        target_dst_sigm_ar = [ x + 1 for x in target_dst_ar]
        #target_dst_sigm = target_dst+1
        

        pred_src_src_sigm_ar = [ x + 1 for x in pred_src_src]
        #pred_src_src_sigm = pred_src_src+1
        
        pred_dst_dst_sigm_ar = [ x + 1 for x in pred_dst_dst]
        #pred_dst_dst_sigm = pred_dst_dst+1
        
        pred_src_dst_sigm_ar = [ x + 1 for x in pred_src_dst] 
        #pred_src_dst_sigm = pred_src_dst+1
        
        #[ x1_bgr, x1_bgr_up, x2_bgr, x2_bgr_up, x3_bgr, x3_bgr_up, x4_bgr ]
        
        
        
        target_src_masked_ar = [ target_src_sigm_ar[i]*target_srcm_sigm_ar[i]  for i in range(len(target_src_sigm_ar))]
        #target_src_masked = target_src_sigm*target_srcm_sigm
        
        target_dst_masked_ar = [ target_dst_sigm_ar[i]*target_dstm_sigm_ar[i]  for i in range(len(target_dst_sigm_ar))]
        #target_dst_masked = target_dst_sigm * target_dstm_sigm
        
        target_dst_anti_masked_ar = [ target_dst_sigm_ar[i]*target_dstm_anti_sigm_ar[i]  for i in range(len(target_dst_sigm_ar))]
        #target_dst_anti_masked = target_dst_sigm * target_dstm_anti_sigm
        
        #pred_src_src_masked_ar = [ pred_src_src_sigm_ar[i]*target_srcm_sigm_ar[i]  for i in range(len(pred_src_src_sigm_ar))]
        ##pred_src_src_masked = pred_src_src_sigm * target_srcm_sigm
        #
        #pred_dst_dst_masked_ar = [ pred_dst_dst_sigm_ar[i]*target_dstm_sigm_ar[i]  for i in range(len(pred_dst_dst_sigm_ar))]
        ##pred_dst_dst_masked = pred_dst_dst_sigm * target_dstm_sigm
        #
        #psd_target_dst_masked_ar = [ pred_src_dst_sigm_ar[i]*target_dstm_sigm_ar[i]  for i in range(len(pred_src_dst_sigm_ar))]
        ##psd_target_dst_masked = pred_src_dst_sigm * target_dstm_sigm
        #
        #psd_target_dst_anti_masked_ar = [ pred_src_dst_sigm_ar[i]*target_dstm_anti_sigm_ar[i]  for i in range(len(pred_src_dst_sigm_ar))]
        ##psd_target_dst_anti_masked = pred_src_dst_sigm * target_dstm_anti_sigm
        def optimizer():
            return Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        
        src_loss_x1 = K.mean( K.abs ( target_src_masked_ar[0] - pred_src_src_sigm_ar[0] * target_srcm_sigm_ar[0] ) )        
        
        src_loss_x2 = K.mean( 100*K.square(tf_dssim(2.0)( target_src_masked_ar[1], \
                                (1.0 - alpha) * ( tf.image.resize_nearest_neighbor( pred_src_src_sigm_ar[0], K.int_shape(pred_src_src_sigm_ar[1])[1:3] ) * target_srcm_sigm_ar[1] ) + \
                                (      alpha) * (                                                                                pred_src_src_sigm_ar[1] * target_srcm_sigm_ar[1] ) )))

        src_loss_x3 = K.mean( 100*K.square(tf_dssim(2.0)( target_src_masked_ar[2],\
                                (1.0 - alpha) * ( tf.image.resize_nearest_neighbor( pred_src_src_sigm_ar[1], K.int_shape(pred_src_src_sigm_ar[2])[1:3] ) * target_srcm_sigm_ar[2] ) + \
                                (      alpha) * (                                                                                pred_src_src_sigm_ar[2] * target_srcm_sigm_ar[2] ) )))
                      
        src_loss_x4 = K.mean( 100*K.square(tf_dssim(2.0)( target_src_masked_ar[3], \
                                (1.0 - alpha) * ( tf.image.resize_nearest_neighbor( pred_src_src_sigm_ar[2], K.int_shape(pred_src_src_sigm_ar[3])[1:3] ) * target_srcm_sigm_ar[3] ) + \
                                (      alpha) * (                                                                                pred_src_src_sigm_ar[3] * target_srcm_sigm_ar[3] ) )))    
                                
        src_loss_x5 = K.mean( 100*K.square(tf_dssim(2.0)( target_src_masked_ar[3], \
                                                ( pred_src_src_sigm_ar[3] * target_srcm_sigm_ar[3] ) )))

        if self.options['archi'] == 'liae':
            src_loss_x1_train_weights = self.encoder.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights
        else:
            src_loss_x1_train_weights = self.encoder.trainable_weights + self.decoders_src[0].trainable_weights + self.decoders_src[1].trainable_weights

        self.src_x1_train = K.function ([warped_src, target_src, target_srcm, warped_dst, target_dst, target_dstm ],[src_loss_x1],
                                    optimizer().get_updates(src_loss_x1, src_loss_x1_train_weights) )
        
        if self.options['archi'] == 'liae':
            src_loss_x2_train_weights = self.encoder.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights
        else:
            src_loss_x2_train_weights = self.encoder.trainable_weights + self.decoders_src[0].trainable_weights + self.decoders_src[1].trainable_weights \
                                                                       + self.decoders_src[2].trainable_weights + self.decoders_src[3].trainable_weights
                                                                       
        self.src_x2_train = K.function ([alpha, warped_src, target_src, target_srcm, warped_dst, target_dst, target_dstm ],[src_loss_x2],
                                    optimizer().get_updates(src_loss_x2, src_loss_x2_train_weights) )
                   
        if self.options['archi'] == 'liae':
            src_loss_x3_train_weights = self.encoder.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights
        else:
            src_loss_x3_train_weights = self.encoder.trainable_weights + self.decoders_src[0].trainable_weights \
                                                                       + self.decoders_src[2].trainable_weights + self.decoders_src[3].trainable_weights \
                                                                       + self.decoders_src[4].trainable_weights + self.decoders_src[5].trainable_weights \
                                                                       
        self.src_x3_train = K.function ([alpha, warped_src, target_src, target_srcm, warped_dst, target_dst, target_dstm ],[src_loss_x3],
                                    optimizer().get_updates(src_loss_x3, src_loss_x3_train_weights) )
                                    
                                    
                                    
        if self.options['archi'] == 'liae':
            src_loss_x4_train_weights = self.encoder.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights
        else:
            src_loss_x4_train_weights = self.encoder.trainable_weights + self.decoders_src[0].trainable_weights \
                                                                       + self.decoders_src[2].trainable_weights \
                                                                       + self.decoders_src[4].trainable_weights + self.decoders_src[5].trainable_weights \
                                                                       + self.decoders_src[6].trainable_weights + self.decoders_src[7].trainable_weights \
                                                                       
        self.src_x4_train = K.function ([alpha, warped_src, target_src, target_srcm, warped_dst, target_dst, target_dstm ],[src_loss_x4],
                                    optimizer().get_updates(src_loss_x4, src_loss_x4_train_weights) )                            
                                    
        if self.options['archi'] == 'liae':
            src_loss_x5_train_weights = self.encoder.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights
        else:
            src_loss_x5_train_weights = self.encoder.trainable_weights + self.decoders_src[0].trainable_weights \
                                                                       + self.decoders_src[2].trainable_weights \
                                                                       + self.decoders_src[4].trainable_weights \
                                                                       + self.decoders_src[6].trainable_weights + self.decoders_src[7].trainable_weights \
                                                                       
        self.src_x5_train = K.function ([alpha, warped_src, target_src, target_srcm, warped_dst, target_dst, target_dstm ],[src_loss_x5],
                                    optimizer().get_updates(src_loss_x5, src_loss_x5_train_weights) )                       
                                    
        
        #src_mask_loss_x1 = K.mean(K.square(target_srcm_ar[0]-pred_src_srcm[0]))   
        #
        #src_mask_loss_x2 = K.mean( 100*K.square(tf_dssim(2.0)( target_src_masked_ar[1], \
        #                        (1.0 - alpha) * ( tf.image.resize_nearest_neighbor( pred_src_src_sigm_ar[0], K.int_shape(pred_src_src_sigm_ar[1])[1:3] ) * target_srcm_sigm_ar[1] ) + \
        #                        (      alpha) * (                                                                                pred_src_src_sigm_ar[1] * target_srcm_sigm_ar[1] ) )))
        #
        #src_mask_loss_x3 = K.mean( 100*K.square(tf_dssim(2.0)( target_src_masked_ar[2],\
        #                        (1.0 - alpha) * ( tf.image.resize_nearest_neighbor( pred_src_src_sigm_ar[1], K.int_shape(pred_src_src_sigm_ar[2])[1:3] ) * target_srcm_sigm_ar[2] ) + \
        #                        (      alpha) * (                                                                                pred_src_src_sigm_ar[2] * target_srcm_sigm_ar[2] ) )))
        #              
        #src_mask_loss_x4 = K.mean( 100*K.square(tf_dssim(2.0)( target_src_masked_ar[3], \
        #                        (1.0 - alpha) * ( tf.image.resize_nearest_neighbor( pred_src_src_sigm_ar[2], K.int_shape(pred_src_src_sigm_ar[3])[1:3] ) * target_srcm_sigm_ar[3] ) + \
        #                        (      alpha) * (                                                                                pred_src_src_sigm_ar[3] * target_srcm_sigm_ar[3] ) )))    
        #                        
        #src_mask_loss_x5 = K.mean( 100*K.square(tf_dssim(2.0)( target_src_masked_ar[3], \
        #                                        ( pred_src_src_sigm_ar[3] * target_srcm_sigm_ar[3] ) )))
        #
        #
        #
        #if self.options['archi'] == 'liae':
        #    src_mask_loss_x1_train_weights = self.encoder.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights
        #else:
        #    src_mask_loss_x1_train_weights = self.encoder.trainable_weights + self.decoders_srcm[0].trainable_weights + self.decoders_srcm[1].trainable_weights
        #
        #self.src_mask_x1_train = K.function ([warped_src, target_src, target_srcm, warped_dst, target_dst, target_dstm ],[src_mask_loss_x1],
        #                            optimizer().get_updates(src_mask_loss_x1, src_mask_loss_x1_train_weights) )
        
        
        dst_loss_x1 = K.mean( K.abs ( target_dst_masked_ar[0] - pred_dst_dst_sigm_ar[0] * target_dstm_sigm_ar[0] ) )        
        
        dst_loss_x2 = K.mean( 100*K.square(tf_dssim(2.0)( target_dst_masked_ar[1], \
                                (1.0 - alpha) * ( tf.image.resize_nearest_neighbor( pred_dst_dst_sigm_ar[0], K.int_shape(pred_dst_dst_sigm_ar[1])[1:3] ) * target_dstm_sigm_ar[1] ) + \
                                (      alpha) * (                                                                                pred_dst_dst_sigm_ar[1] * target_dstm_sigm_ar[1] ) )))

        dst_loss_x3 = K.mean( 100*K.square(tf_dssim(2.0)( target_dst_masked_ar[2],\
                                (1.0 - alpha) * ( tf.image.resize_nearest_neighbor( pred_dst_dst_sigm_ar[1], K.int_shape(pred_dst_dst_sigm_ar[2])[1:3] ) * target_dstm_sigm_ar[2] ) + \
                                (      alpha) * (                                                                                pred_dst_dst_sigm_ar[2] * target_dstm_sigm_ar[2] ) )))
                      
        dst_loss_x4 = K.mean( 100*K.square(tf_dssim(2.0)( target_dst_masked_ar[3], \
                                (1.0 - alpha) * ( tf.image.resize_nearest_neighbor( pred_dst_dst_sigm_ar[2], K.int_shape(pred_dst_dst_sigm_ar[3])[1:3] ) * target_dstm_sigm_ar[3] ) + \
                                (      alpha) * (                                                                                pred_dst_dst_sigm_ar[3] * target_dstm_sigm_ar[3] ) )))    
                                
        dst_loss_x5 = K.mean( 100*K.square(tf_dssim(2.0)( target_dst_masked_ar[3], \
                                                ( pred_dst_dst_sigm_ar[3] * target_dstm_sigm_ar[3] ) )))

        if self.options['archi'] == 'liae':
            dst_loss_x1_train_weights = self.encoder.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights
        else:
            dst_loss_x1_train_weights = self.encoder.trainable_weights + self.decoders_dst[0].trainable_weights + self.decoders_dst[1].trainable_weights

        self.dst_x1_train = K.function ([warped_src, target_src, target_srcm, warped_dst, target_dst, target_dstm ],[dst_loss_x1],
                                    optimizer().get_updates(dst_loss_x1, dst_loss_x1_train_weights) )
        
        if self.options['archi'] == 'liae':
            dst_loss_x2_train_weights = self.encoder.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights
        else:
            dst_loss_x2_train_weights = self.encoder.trainable_weights + self.decoders_dst[0].trainable_weights + self.decoders_dst[1].trainable_weights \
                                                                       + self.decoders_dst[2].trainable_weights + self.decoders_dst[3].trainable_weights
                                                                       
        self.dst_x2_train = K.function ([alpha, warped_src, target_src, target_srcm,warped_dst, target_dst, target_dstm ],[dst_loss_x2],
                                    optimizer().get_updates(dst_loss_x2, dst_loss_x2_train_weights) )
                   
        if self.options['archi'] == 'liae':
            dst_loss_x3_train_weights = self.encoder.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights
        else:
            dst_loss_x3_train_weights = self.encoder.trainable_weights + self.decoders_dst[0].trainable_weights \
                                                                       + self.decoders_dst[2].trainable_weights + self.decoders_dst[3].trainable_weights \
                                                                       + self.decoders_dst[4].trainable_weights + self.decoders_dst[5].trainable_weights \
                                                                       
        self.dst_x3_train = K.function ([alpha, warped_src, target_src, target_srcm,warped_dst, target_dst, target_dstm ],[dst_loss_x3],
                                    optimizer().get_updates(dst_loss_x3, dst_loss_x3_train_weights) )
                                    
                                    
                                    
        if self.options['archi'] == 'liae':
            dst_loss_x4_train_weights = self.encoder.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights
        else:
            dst_loss_x4_train_weights = self.encoder.trainable_weights + self.decoders_dst[0].trainable_weights \
                                                                       + self.decoders_dst[2].trainable_weights \
                                                                       + self.decoders_dst[4].trainable_weights + self.decoders_dst[5].trainable_weights \
                                                                       + self.decoders_dst[6].trainable_weights + self.decoders_dst[7].trainable_weights \
                                                                       
        self.dst_x4_train = K.function ([alpha, warped_src, target_src, target_srcm,warped_dst, target_dst, target_dstm ],[dst_loss_x4],
                                    optimizer().get_updates(dst_loss_x4, dst_loss_x4_train_weights) )                            
                                    
        if self.options['archi'] == 'liae':
            dst_loss_x5_train_weights = self.encoder.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights
        else:
            dst_loss_x5_train_weights = self.encoder.trainable_weights + self.decoders_dst[0].trainable_weights \
                                                                       + self.decoders_dst[2].trainable_weights \
                                                                       + self.decoders_dst[4].trainable_weights \
                                                                       + self.decoders_dst[6].trainable_weights + self.decoders_dst[7].trainable_weights \
                                                                       
        self.dst_x5_train = K.function ([alpha, warped_src, target_src, target_srcm, warped_dst, target_dst, target_dstm ],[dst_loss_x5],
                                    optimizer().get_updates(dst_loss_x5, dst_loss_x5_train_weights) )   
                                    
        self.AE_view = K.function ([warped_src, warped_dst], pred_src_src + pred_src_srcm + pred_dst_dst + pred_dst_dstm + pred_src_dst + pred_src_dstm) 
        #import code
        #code.interact(local=dict(globals(), **locals()))  
        
        '''
        src_loss = K.mean( 100*K.square(tf_dssim(2.0)( target_src_masked, pred_src_src_masked )) ) 
        
        

        if self.options['face_style_power'] != 0:
            face_style_power = self.options['face_style_power'] / 100.0
            src_loss += tf_style_loss(gaussian_blur_radius=resolution // 8, loss_weight=0.2*face_style_power)(psd_target_dst_masked, target_dst_masked) 
            
        if self.options['bg_style_power'] != 0:
            bg_style_power = self.options['bg_style_power'] / 100.0
            src_loss += K.mean( (100*bg_style_power)*K.square(tf_dssim(2.0)( psd_target_dst_anti_masked, target_dst_anti_masked )))

        

        dst_loss = K.mean( 100*K.square(tf_dssim(2.0)( target_dst_masked, pred_dst_dst_masked )) )        
        
        if self.options['archi'] == 'liae':
            dst_train_weights = self.encoder.trainable_weights + self.inter_B.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights
        else:
            dst_train_weights = self.encoder.trainable_weights + self.decoder_dst.trainable_weights
        self.dst_train = K.function ([warped_dst, target_dst, target_dstm],[dst_loss],
                                    optimizer().get_updates(dst_loss, dst_train_weights) )
  
        src_mask_loss = K.mean(K.square(target_srcm-pred_src_srcm))    

        if self.options['archi'] == 'liae':
            src_mask_train_weights = self.encoder.trainable_weights + self.inter_AB.trainable_weights + self.decoderm.trainable_weights
        else:
            src_mask_train_weights = self.encoder.trainable_weights + self.decoder_srcm.trainable_weights
            
        self.src_mask_train = K.function ([warped_src, target_srcm],[src_mask_loss],
                                    optimizer().get_updates(src_mask_loss, src_mask_train_weights ) )
        
        dst_mask_loss = K.mean(K.square(target_dstm-pred_dst_dstm))   

        if self.options['archi'] == 'liae':
            dst_mask_train_weights = self.encoder.trainable_weights + self.inter_B.trainable_weights + self.inter_AB.trainable_weights + self.decoderm.trainable_weights
        else:
            dst_mask_train_weights = self.encoder.trainable_weights + self.decoder_dstm.trainable_weights
            
        self.dst_mask_train = K.function ([warped_dst, target_dstm],[dst_mask_loss],
                                    optimizer().get_updates(dst_mask_loss, dst_mask_train_weights) )
                                    
        self.AE_view = K.function ([warped_src, warped_dst],[pred_src_src, pred_src_srcm, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm])
        self.AE_convert = K.function ([warped_dst],[pred_src_dst, pred_src_dstm])
        '''
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
                                              
                                            ] ),
                                              
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, normalize_tanh = True), 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | face_type | f.MODE_BGR, resolution], 
                                            
                                              [f.TRANSFORMED | face_type | f.MODE_BGR, resolution], 
                                              [f.TRANSFORMED | face_type | f.MODE_M | f.FACE_MASK_FULL, resolution] 
                                              
                                            ] )
                ])
    #override
    def onSave(self):
        pass
        #if self.options['archi'] == 'liae':
        #    self.save_weights_safe( [[self.encoder, self.get_strpath_storage_for_file(self.encoderH5)],
        #                             [self.inter_B, self.get_strpath_storage_for_file(self.inter_BH5)],
        #                             [self.inter_AB, self.get_strpath_storage_for_file(self.inter_ABH5)],
        #                             [self.decoder, self.get_strpath_storage_for_file(self.decoderH5)],
        #                             [self.decoderm, self.get_strpath_storage_for_file(self.decodermH5)],
        #                            ] )
        #else:
        #    self.save_weights_safe( [[self.encoder, self.get_strpath_storage_for_file(self.encoderH5)],
        #                             [self.decoder_src, self.get_strpath_storage_for_file(self.decoder_srcH5)],
        #                             [self.decoder_srcm, self.get_strpath_storage_for_file(self.decoder_srcmH5)],
        #                             [self.decoder_dst, self.get_strpath_storage_for_file(self.decoder_dstH5)],
        #                             [self.decoder_dstm, self.get_strpath_storage_for_file(self.decoder_dstmH5)],
        #                            ] )
                                    
    #override
    def onTrainOneEpoch(self, sample):
        warped_src, target_src, target_src_mask = sample[0]
        warped_dst, target_dst, target_dst_mask = sample[1]
        
        if self.epoch < 10000:
            #a = float(self.epoch) / 10000.0
            
            src_loss, = self.src_x1_train ([warped_src, target_src, target_src_mask, warped_dst, target_dst, target_dst_mask])
            #dst_loss, = self.dst_x1_train ([warped_src, target_src, target_src_mask, warped_dst, target_dst, target_dst_mask])
        else:
            pass
        
        
        #src_loss, = self.src_train ([warped_src, target_src, target_src_mask, warped_dst, target_dst, target_dst_mask])
        #dst_loss, = self.dst_train ([warped_dst, target_dst, target_dst_mask])
        #
        #src_mask_loss, = self.src_mask_train ([warped_src, target_src_mask])
        #dst_mask_loss, = self.dst_mask_train ([warped_dst, target_dst_mask])
        
        return ( ('src_loss', src_loss), ('dst_loss', dst_loss) )
        

    #override
    def onGetPreview(self, sample):
        test_A   = sample[0][1][0:4] #first 4 samples
        test_A_m = sample[0][2][0:4] #first 4 samples
        test_B   = sample[1][1][0:4]
        test_B_m = sample[1][2][0:4]
        
        S = test_A
        D = test_B
        
        SS1, SM1, SS2, SM2, SS3, SM3, SS4, SM4, \
        DD1, DM1, DD2, DM2, DD3, DM3, DD4, DM4, \
        SD1, SDM1, SD2, SDM2, SD3, SDM3, SD4, SDM4 = self.AE_view ([test_A, test_B])       

        
        S, D, \
        SS1, SM1, SS2, SM2, SS3, SM3, SS4, SM4, \
        DD1, DM1, DD2, DM2, DD3, DM3, DD4, DM4, \
        SD1, SDM1, SD2, SDM2, SD3, SDM3, SD4, SDM4 = [ x / 2 + 0.5 for x in \
                [S, D, 
                 SS1, SM1, SS2, SM2, SS3, SM3, SS4, SM4, 
                 DD1, DM1, DD2, DM2, DD3, DM3, DD4, DM4, 
                 SD1, SDM1, SD2, SDM2, SD3, SDM3, SD4, SDM4 ] ]

        #SM, DM, SDM = [ np.repeat (x, (3,), -1) for x in [SM, DM, SDM] ]
        
        st = []
        for i in range(0, len(test_A)):
            st.append ( np.concatenate ( (
                SS1[i], #SM[i],
                DD1[i], #DM[i],
                SD1[i], #SDM[i]
                ), axis=1) )
            
        return [ ('SAE', np.concatenate ( st, axis=0 ) ) ]
    
    def predictor_func (self, face):        
        face = face * 2.0 - 1.0        
        face_128_bgr = face[...,0:3] 
        x, mx = [ (x[0] + 1.0) / 2.0 for x in self.AE_convert ( [ np.expand_dims(face_128_bgr,0) ] ) ]
        return np.concatenate ( (x,mx), -1 )
        
    #override
    def get_converter(self, **in_options):
        from models import ConverterMasked

        base_erode_mask_modifier = 40 if self.options['face_type'] == 'f' else 100
        base_blur_mask_modifier = 10 if self.options['face_type'] == 'f' else 100
        
        face_type = FaceType.FULL if self.options['face_type'] == 'f' else FaceType.HALF
        
        return ConverterMasked(self.predictor_func, 
                               predictor_input_size=self.options['resolution'], 
                               output_size=self.options['resolution'], 
                               face_type=face_type, 
                               base_erode_mask_modifier=base_erode_mask_modifier,
                               base_blur_mask_modifier=base_blur_mask_modifier,
                               **in_options)
    
    @staticmethod
    def LIAEEncFlow(resolution, adapt_k_size, light_enc, ed_ch_dims=42):
        exec (nnlib.import_all(), locals(), globals())
        
        k_size = resolution // 16 + 1 if adapt_k_size else 5
        strides = resolution // 32 if adapt_k_size else 2
        
        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)(Conv2D(dim, k_size, strides=strides, padding='same')(x))
            return func 

        def downscale_sep (dim):
            def func(x):
                return LeakyReLU(0.1)(SeparableConv2D(dim, k_size, strides=strides, padding='same')(x))
            return func 
            
        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func   
    
        def func(input):
            ed_dims = K.int_shape(input)[-1]*ed_ch_dims
            
            x = input            
            x = downscale(ed_dims)(x)
            if not light_enc:                
                x = downscale(ed_dims*2)(x)
                x = downscale(ed_dims*4)(x)
                x = downscale(ed_dims*8)(x)
            else:
                x = downscale_sep(ed_dims*2)(x)
                x = downscale_sep(ed_dims*4)(x)
                x = downscale_sep(ed_dims*8)(x)
            
            x = Flatten()(x)               
            return x
        return func
    
    @staticmethod
    def LIAEInterFlow(resolution, ae_dims=256):
        exec (nnlib.import_all(), locals(), globals())
        lowest_dense_res=resolution // 16
        
        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func 
        
        def func(input):   
            x = input[0]
            x = Dense(ae_dims)(x)
            x = Dense(lowest_dense_res * lowest_dense_res * ae_dims*2)(x)
            x = Reshape((lowest_dense_res, lowest_dense_res, ae_dims*2))(x)
            x = upscale(ae_dims*2)(x)
            return x
        return func
        
    @staticmethod
    def LIAEDecFlow(output_nc,ed_ch_dims=21,activation='tanh'):
        exec (nnlib.import_all(), locals(), globals())
        
        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func   
            
        def func(input):
            ed_dims = output_nc * ed_ch_dims
            
            x = input[0]
            x = upscale(ed_dims*8)(x)
            x = upscale(ed_dims*4)(x)
            x = upscale(ed_dims*2)(x)
                
            x = Conv2D(output_nc, kernel_size=5, padding='same', activation=activation)(x)
            return x
            
        return func

    '''
    @staticmethod
    def PGDFEncFlow(resolution, ed_ch_dims=42):
        exec (nnlib.import_all(), locals(), globals())
        
        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)(Conv2D(dim, 5, strides=2, padding='same')(x))
            return func
            
         def func(input):
            ed_dims = K.int_shape(input)[-1]*ed_ch_dims
            
            x = input
            
            if resolution == 16:
                x = downscale(ed_dims // 2)(x)
                x = Dense(ed_dims // 2)(Flatten()(x))
                x = Dense(1 * 1 * ed_dims // 2)(x)
                x = Reshape((1, 1, ed_dims // 2))(x)
                x = upscale(ed_dims // 2)(x)
                
            if resolution == 32:
                x = downscale(ed_dims // 2)(x)
                x = downscale(ed_dims)(x)
                x = Dense(ed_dims)(Flatten()(x))
                x = Dense(2 * 2 * ed_dims)(x)
                x = Reshape((2, 2, ed_dims))(x)
                x = upscale(ed_dims)(x)
                
            if resolution == 64:
                x = downscale(ed_dims // 2)(x)
                x = downscale(ed_dims)(x)
                x = downscale(ed_dims * 2)(x)
                x = Dense(ed_dims * 2)(Flatten()(x))
                x = Dense(4 * 4 * ed_dims * 2)(x)
                x = Reshape((4, 4, ed_dims * 2))(x)
                x = upscale(ed_dims * 2)(x)
                
            if resolution == 128:
                x = downscale(ed_dims // 2)(x)
                x = downscale(ed_dims)(x)
                x = downscale(ed_dims * 2)(x)
                x = downscale(ed_dims * 4)(x)
                x = Dense(ed_dims * 4)(Flatten()(x))
                x = Dense(8 * 8 * ed_dims * 4)(x)
                x = Reshape((8, 8, ed_dims * 4))(x)
                x = upscale(ed_dims * 4)(x)
                
                
        return func
        
    @staticmethod
    def PGDFDecFlow(output_nc, resolution, ed_ch_dims=42):
        exec (nnlib.import_all(), locals(), globals())
        ed_dims = output_nc * ed_ch_dims
        
        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func   
      
        def func(input, prevs):
            
                       16 8  4   2   dense 1 2     4 8 16 3
                    32 16 8      4   dense 2 4       8 16 32 3
                 64 32 16        8   dense 4 8         16 32 64 3
             128 64 32          16   dense 8 16           32 64 128 3
            
            

                 64 32 16        8   dense 4 8         16 32 64 3
             128 64 32          16   dense 8 16           32 64 128 3          

            
            r16, r32, r64
            
            x = input
            
            x = Add()([r16, input])
            upscale(ed_dims*8)(x)
            
            if resolution == 16:
                x = Conv2D(output_nc, kernel_size=5, padding='same', activation='tanh')(x)
                
                return x
                
                
                x = downscale(ed_dims // 2)(x)
                x = Dense(ed_dims // 2)(Flatten()(x))
                x = Dense(8 * 8 * ed_dims // 2)(x)
                x = Reshape((8, 8, ed_dims // 2))(x)
                x = upscale(ed_dims // 2)(x)
                
            #x = upscale(ed_dims*8)(x)    
            
        return func    
    '''
    
    @staticmethod
    def DFEncFlow(resolution, adapt_k_size, light_enc, ae_dims=512, ed_ch_dims=42):
        exec (nnlib.import_all(), locals(), globals())
        k_size = resolution // 16 + 1 if adapt_k_size else 5
        strides = resolution // 32 if adapt_k_size else 2
        lowest_dense_res = resolution // 16
        
        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)(Conv2D(dim, k_size, strides=strides, padding='same')(x))
            return func 
            
        def downscale_sep (dim):
            def func(x):
                return LeakyReLU(0.1)(SeparableConv2D(dim, k_size, strides=strides, padding='same')(x))
            return func 
            
        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func   
    
        def func(input):     
            x = input
            
            ed_dims = K.int_shape(input)[-1]*ed_ch_dims
            
            x = downscale(ed_dims)(x)
            if not light_enc:
                x = downscale(ed_dims*2)(x)
                x = downscale(ed_dims*4)(x)
                x = downscale(ed_dims*8)(x)
            else:
                x = downscale_sep(ed_dims*2)(x)
                x = downscale_sep(ed_dims*4)(x)
                x = downscale_sep(ed_dims*8)(x)
    
            x = Dense(ae_dims)(Flatten()(x))
            x = Dense(lowest_dense_res * lowest_dense_res * ae_dims)(x)
            x = Reshape((lowest_dense_res, lowest_dense_res, ae_dims))(x)
            x = upscale(ae_dims)(x)
               
            return x
        return func
    
    @staticmethod
    def DFBuildDecoders(inputs, output_nc, ed_ch_dims=21):
        exec (nnlib.import_all(), locals(), globals())
        ed_dims = output_nc * ed_ch_dims
        
        def outputsAsInputs(x):
            return [Input(K.int_shape(x)[1:]) for x in x.outputs]
                
        def upscale (dim):
            def func(x):
                x = x[0]
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func
            
        def to_bgr ():
            def func(x):
                x = x[0]
                return Conv2D(output_nc, kernel_size=5, padding='same', activation='tanh')(x)
            return func
            
        input = inputs[0]
        
        x1     = Model ( input, input) #LeakyReLU(0.1)(Conv2D(ed_dims*16, 3, strides=1, padding='same')    (   input)) )
        x1_bgr = modelify (to_bgr())(outputsAsInputs(x1))

        x2     = modelify (upscale(ed_dims*8))( outputsAsInputs(x1) )
        x2_bgr = modelify (to_bgr()          )( outputsAsInputs(x2) )
        
        x3     = modelify (upscale(ed_dims*4))( outputsAsInputs(x2) )
        x3_bgr = modelify (to_bgr()          )( outputsAsInputs(x3) )
        
        x4     = modelify (upscale(ed_dims*2))( outputsAsInputs(x3) )
        x4_bgr = modelify (to_bgr()          )( outputsAsInputs(x4) )
        
        return [ x1, x1_bgr, x2, x2_bgr, x3, x3_bgr, x4, x4_bgr ]

        
Model = SAEModel