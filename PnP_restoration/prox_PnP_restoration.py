import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import utils_sr
import torch
from argparse import ArgumentParser
from utils.utils_restoration import rgb2y, psnr, array2tensor, tensor2array
import sys
from matplotlib.ticker import MaxNLocator


class PnP_restoration():

    def __init__(self, hparams):

        self.hparams = hparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialize_cuda_denoiser()

    def initialize_cuda_denoiser(self):
        '''
        Initialize the denoiser model with the given pretrained ckpt
        '''
        sys.path.append('../GS_denoising/')
        from lightning_denoiser import GradMatch
        parser2 = ArgumentParser(prog='utils_restoration.py')
        parser2 = GradMatch.add_model_specific_args(parser2)
        parser2 = GradMatch.add_optim_specific_args(parser2)
        hparams = parser2.parse_known_args()[0]
        if 'nb_4' in self.hparams.pretrained_checkpoint :
            hparams.DRUNET_nb = 4
        hparams.grad_matching = self.hparams.grad_matching
        hparams.act_mode = 's'
        self.denoiser_model = GradMatch(hparams)
        checkpoint = torch.load(self.hparams.pretrained_checkpoint, map_location=self.device)
        self.denoiser_model.load_state_dict(checkpoint['state_dict'])
        self.denoiser_model.eval()
        for i, v in self.denoiser_model.named_parameters():
            v.requires_grad = False
        self.denoiser_model = self.denoiser_model.to(self.device)
        if self.hparams.precision == 'double' :
            if self.denoiser_model is not None:
                self.denoiser_model.double()


    def initialize_prox(self, img, degradation):
        '''
        calculus for future prox computatations
        :param img: degraded image
        :param degradation: 2D blur kernel for deblurring and SR, mask for inpainting
        '''
        if self.hparams.degradation_mode == 'deblurring' or self.hparams.degradation_mode == 'SR':
            self.k = degradation
            self.k_tensor = array2tensor(np.expand_dims(self.k, 2)).double().to(self.device)
            self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate_prox(img, self.k_tensor, 1)
        elif self.hparams.degradation_mode == 'inpainting':
            self.M = array2tensor(degradation).double().to(self.device)
            self.My = self.M*img
        else:
            print('degradation mode not treated')


    def calculate_prox(self, img):
        '''
        Calculation of the proximal mapping of the data term f
        :param img: input for the prox
        :return: prox_f(img)
        '''
        if self.hparams.degradation_mode == 'deblurring':
            rho = torch.tensor([1/self.hparams.lamb]).double().repeat(1, 1, 1, 1).to(self.device)
            px = utils_sr.prox_solution(img.double(), self.FB, self.FBC, self.F2B, self.FBFy, rho, 1)
        elif self.hparams.degradation_mode == 'SR':
            rho = torch.tensor([1 /self.hparams.lamb]).double().repeat(1, 1, 1, 1).to(self.device)
            px = utils_sr.prox_solution(img.double(), self.FB, self.FBC, self.F2B, self.FBFy, rho, self.hparams.sf)
        elif self.hparams.degradation_mode == 'inpainting':
            if self.hparams.noise_level_img > 1e-2:
                px = (self.hparams.lamb*self.My + img)/(self.hparams.lamb*self.M+1)
            else :
                px = self.My + (1-self.M)*img
        else:
            print('degradation mode not treated')
        return px

    def calculate_grad(self, img):
        '''
        Calculation of the proximal mapping of the data term f
        :param img: input for the prox
        :return: prox_f(img)
        '''
        if self.hparams.degradation_mode == 'deblurring' :
            grad = utils_sr.grad_solution(img.double(), self.FB, self.FBC, self.F2B, self.FBFy, 1)
        if self.hparams.degradation_mode == 'SR' :
            grad = utils_sr.grad_solution(img.double(), self.FB, self.FBC, self.FBFy, self.hparams.sf)
        return grad

    def calculate_regul(self,y,x,g):
        '''
        Calculation of the regularization (1/tau)*phi_sigma(y)
        :param y: Point where to evaluate
        :param x: D^{-1}(y)
        :param g: Precomputed regularization function value at x
        :return: regul(y)
        '''
        regul = (1 / self.hparams.lamb) * (g - (1 / 2) * torch.norm(x - y, p=2) ** 2)
        return regul

    def calulate_data_term(self,y,img):
        '''
        Calculation of the data term value f(y)
        :param y: Point where to evaluate F
        :param img: Degraded image
        :return: f(y)
        '''
        if self.hparams.degradation_mode == 'deblurring':
            deg_y = utils_sr.imfilter(y.double(), self.k_tensor[0].double().flip(1).flip(2).expand(3, -1, -1, -1))
            f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
        elif self.hparams.degradation_mode == 'SR':
            deg_y = utils_sr.imfilter(y.double(), self.k_tensor[0].double().flip(1).flip(2).expand(3, -1, -1, -1))
            deg_y = deg_y[..., 0::self.hparams.sf, 0::self.hparams.sf]
            f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
        elif self.hparams.degradation_mode == 'inpainting':
            deg_y = self.M * y.double()
            f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
        else:
            print('degradation not implemented')
        return f

    def calculate_F(self, y, x, g, img):
        '''
        Calculation of the objective function value f(y) + (1/tau)*phi_sigma(y)
        :param y: Point where to evaluate F
        :param x: D^{-1}(y)
        :param g: Precomputed regularization function value at x
        :param img: Degraded image
        :return: F(y)
        '''
        regul = self.calculate_regul(y,x,g)
        if self.hparams.no_data_term:
            F = regul
        else:
            f = self.calulate_data_term(y,img)
            F = f + regul
        return F.item()

    def calculate_lyapunov_DRS(self,y,z,x,g,img):
        '''
            Calculation of the Lyapunov function value Psi(x)
            :param x: Point where to evaluate F
            :param y,z: DRS iterations initialized at x
            :param g: Precomputed regularization function value at x
            :param img: Degraded image
            :return: Psi(x)
        '''
        regul = self.calculate_regul(y,x,g)
        f = self.calulate_data_term(z, img)
        Psi = regul + f + (1 / self.hparams.lamb) * (torch.sum(torch.mul(y-x,y-z)) + (1/2) * torch.norm(y - z, p=2) ** 2)
        return Psi


    def restore(self, img, init_im, clean_img, degradation,extract_results=False):
        '''
        Compute GS-PnP restoration algorithm
        :param img: Degraded image
        :param clean_img: ground-truth clean image
        :param degradation: 2D blur kernel for deblurring and SR, mask for inpainting
        :param extract_results: Extract information for subsequent image or curve saving
        '''

        if extract_results:
            y_list, z_list, x_list, Dg_list, psnr_tab, g_list, Dx_list, F_list, Psi_list = [], [], [], [], [], [], [], [], []

        i = 0 # iteration counter

        img_tensor = array2tensor(img).to(self.device) # for GPU computations (if GPU available)

        self.initialize_prox(img_tensor, degradation) # prox calculus that can be done outside of the loop

        # Initialization of the algorithm
        if self.hparams.degradation_mode == 'SR':
            x0 = cv2.resize(init_im, (img.shape[1] * self.hparams.sf, img.shape[0] * self.hparams.sf),interpolation=cv2.INTER_CUBIC)
            x0 = utils_sr.shift_pixel(x0, self.hparams.sf)
            x0 = array2tensor(x0).to(self.device)
        else:
            x0 = array2tensor(init_im).to(self.device)

        if extract_results:  # extract np images and PSNR values
            out_x = tensor2array(x0.cpu())
            current_x_psnr = psnr(clean_img, out_x)
            if self.hparams.print_each_step:
                print('current x PSNR : ', current_x_psnr)
            psnr_tab.append(current_x_psnr)
            x_list.append(out_x)

        x = x0

        if self.hparams.use_hard_constraint:
            x = torch.clamp(x, 0, 1)

        # Initialize Lyapunov
        diff_Psi = 1
        Psi_old = 1
        Psi = Psi_old


        while i < self.hparams.maxitr and abs(diff_Psi)/Psi_old > self.hparams.relative_diff_Psi_min:

            if self.hparams.inpainting_init :
                if i < self.hparams.n_init:
                    self.sigma_denoiser = 50
                else :
                    self.sigma_denoiser = self.hparams.sigma_denoiser
            else :
                self.sigma_denoiser = self.hparams.sigma_denoiser


            x_old = x
            Psi_old = Psi

            if self.hparams.PnP_algo == 'pGSPnP-PGD':
                # Gradient step
                gradx = self.calculate_grad(x_old)
                z = x_old - self.hparams.lamb*gradx
                # Denoising step
                torch.set_grad_enabled(True)
                Dg, N = self.denoiser_model.calculate_grad(z, self.hparams.sigma_denoiser / 255.)
                torch.set_grad_enabled(False)
                Dg = Dg.detach()
                N = N.detach()
                g = 0.5 * (torch.norm(z.double() - N.double(), p=2) ** 2)
                Dz = z - Dg
                Dx = Dz
                x = (1 - self.hparams.alpha) * z + self.hparams.alpha*Dz
                y = x
                # Hard constraint
                if self.hparams.use_hard_constraint:
                    x = torch.clamp(x,0,1)
                # Calculate Objective
                F = self.calculate_F(x, z, g, img_tensor)

            elif self.hparams.PnP_algo == 'pGSPnP-DRS':
                # Denoising step
                torch.set_grad_enabled(True)
                Dg, N = self.denoiser_model.calculate_grad(x_old, self.hparams.sigma_denoiser / 255.)
                torch.set_grad_enabled(False)
                Dg = Dg.detach()
                N = N.detach()
                g = 0.5 * (torch.norm(x_old.double() - N.double(), p=2) ** 2)
                Dx = x_old - Dg
                y = (1 - self.hparams.alpha)*x_old + self.hparams.alpha*Dx
                # Hard constraint
                if self.hparams.use_hard_constraint:
                    y = torch.clamp(y,0,1)
                # Proximal step
                z = self.calculate_prox(2*y-x_old)
                # Calculate Lyapunov
                Psi = self.calculate_lyapunov_DRS(y,z,x,g,img_tensor)
                diff_Psi = Psi-Psi_old
                # Calculate Objective
                F = self.calculate_F(y, x, g, img_tensor)
                # Final step
                x = x_old + (z-y)

            elif self.hparams.PnP_algo == 'pGSPnP-DRSdiff':

                # Proximal step
                y = self.calculate_prox(x_old)
                y2 = 2*y-x_old

                # Denoising step
                torch.set_grad_enabled(True)
                Dg, N = self.denoiser_model.calculate_grad(y2, self.hparams.sigma_denoiser / 255.)
                torch.set_grad_enabled(False)
                Dg = Dg.detach()
                N = N.detach()
                g = 0.5 * (torch.norm(y2.double() - N.double(), p=2) ** 2)
                Dx = y2 - Dg
                z = (1 - self.hparams.alpha) * y2 + self.hparams.alpha * Dx
                # Hard constraint
                if self.hparams.use_hard_constraint:
                    z = torch.clamp(z, 0, 1)

                # Calculate Lyapunov
                Psi = self.calculate_lyapunov_DRS(y,z,x,g,img_tensor)
                diff_Psi = Psi-Psi_old
                # Calculate Objective
                F = self.calculate_F(y, x, g, img_tensor)
                # Final step
                x = x_old + (z-y)

            else :
                print('algo not implemented')

            # Logging
            if extract_results:
                out_y = tensor2array(y.cpu())
                out_z = tensor2array(z.cpu())
                out_x = tensor2array(x.cpu())
                current_y_psnr = psnr(clean_img, out_y)
                current_z_psnr = psnr(clean_img, out_z)
                current_x_psnr = psnr(clean_img, out_x)
                if self.hparams.print_each_step:
                    print('iteration : ', i)
                    print('current y PSNR : ', current_y_psnr)
                    print('current z PSNR : ', current_z_psnr)
                    print('current x PSNR : ', current_x_psnr)
                y_list.append(out_y)
                x_list.append(out_x)
                z_list.append(out_z)
                Dx_list.append(tensor2array(Dx.cpu()))
                Dg_list.append(torch.norm(Dg).cpu().item())
                g_list.append(g.cpu().item())
                psnr_tab.append(current_x_psnr)
                F_list.append(F)
                Psi_list.append(Psi)

            # next iteration
            i += 1

        output_img = tensor2array(y.cpu())
        output_psnr = psnr(clean_img, output_img)
        output_psnrY = psnr(rgb2y(clean_img), rgb2y(output_img))

        if extract_results:
            return output_img, output_psnr, output_psnrY, x_list, np.array(z_list), np.array(y_list), np.array(Dg_list), np.array(psnr_tab), np.array(Dx_list), np.array(g_list), np.array(F_list), np.array(Psi_list)
        else:
            return output_img, output_psnr, output_psnrY

    def initialize_curves(self):

        self.conv = []
        self.PSNR = []
        self.g = []
        self.Dg = []
        self.F = []
        self.Psi = []
        self.lip_algo = []
        self.lip_D = []
        self.lip_Dg = []

    def update_curves(self, x_list, Dx_list, psnr_tab, Dg_list, g_list, F_list, Psi_list):

        self.F.append(F_list)
        self.Psi.append(Psi_list)
        self.g.append(g_list)
        self.Dg.append(Dg_list)
        self.PSNR.append(psnr_tab)
        self.conv.append(np.array([(np.linalg.norm(x_list[k + 1] - x_list[k]) ** 2) for k in range(len(x_list) - 1)]) / np.sum(np.abs(x_list[0]) ** 2))
        self.lip_algo.append(np.sqrt(np.array([np.sum(np.abs(x_list[k + 1] - x_list[k]) ** 2) for k in range(1, len(x_list) - 1)]) / np.array([np.sum(np.abs(x_list[k] - x_list[k - 1]) ** 2) for k in range(1, len(x_list[:-1]))])))
        self.lip_D.append(np.sqrt(np.array([np.sum(np.abs(Dx_list[i + 2] - Dx_list[i+1]) ** 2) for i in range(len(Dx_list) - 2)]) / np.array([np.sum(np.abs(x_list[i+1] - x_list[i]) ** 2) for i in range(len(Dx_list) - 2)])))
        self.lip_Dg.append(np.sqrt(np.array([np.sum(np.abs(Dg_list[i + 2] - Dg_list[i+1]) ** 2) for i in range(len(Dx_list) - 2)]) / np.array([np.sum(np.abs(x_list[i+1] - x_list[i]) ** 2) for i in range(len(Dg_list) - 2)])))


    def save_curves(self, save_path):

        import matplotlib
        matplotlib.rcParams.update({'font.size': 10})
        matplotlib.rcParams['lines.linewidth'] = 2
        matplotlib.style.use('seaborn-darkgrid')

        plt.figure(0)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if len(self.g) > 1:
            for i in range(len(self.g)):
                plt.plot(self.g[i], markevery=10)
        else:
            plt.plot(self.g[0], '-o', markersize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(os.path.join(save_path, 'g.png'),bbox_inches="tight")

        plt.figure(1)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if len(self.PSNR) > 1:
            for i in range(len(self.PSNR)):
                plt.plot(self.PSNR[i], markevery=10)
        else:
            plt.plot(self.PSNR[0], '-o', markersize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(os.path.join(save_path, 'PSNR.png'),bbox_inches="tight")

        plt.figure(2)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.F)):
            if 'DRS' in self.hparams.PnP_algo :
                plt.plot(self.Psi[i], markevery=10)
            else :
                plt.plot(self.F[i], markevery=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(os.path.join(save_path, 'Liapunov.png'), bbox_inches="tight")

        plt.figure(22)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.F)):
            if 'DRS' in algo_list[i]:
                plt.plot(self.F[i], markevery=10)
            else:
                plt.plot(self.F[i], markevery=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(os.path.join(save_path, 'F.png'), bbox_inches="tight")



        plt.figure(4)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if len(self.Dg) > 1:
            for i in range(len(self.Dg)):
                Ds_norm = [np.linalg.norm(np.array(self.Dg[i][j])) for j in range(len(self.Dg[i]))]
                plt.plot(Ds_norm, linewidth=1.5, markevery=10)
        else :
            Ds_norm = [np.linalg.norm(np.array(self.Dg[0][j])) for j in range(len(self.Dg[0]))]
            plt.plot(Ds_norm, '-o', markersize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(os.path.join(save_path, 'Dg.png'), bbox_inches="tight")

        plt.figure(5)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if len(self.conv) > 1:
            for i in range(len(self.conv)):
                plt.plot(self.conv[i], '-o', markevery=10)
        else:
            plt.plot(self.conv[0], '-o', markersize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(os.path.join(save_path, 'conv_log.png'), bbox_inches="tight")

        self.conv2 = [[np.min(self.conv[i][:k]) for k in range(1, len(self.conv[i]))] for i in range(len(self.conv))]
        conv_rate = [self.conv2[i][0]*np.array([(1/k) for k in range(1,len(self.conv2[i]))]) for i in range(len(self.conv2))]
        plt.figure(6)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if len(self.conv) > 1:
            for i in range(len(self.conv)):
                plt.plot(self.conv2[i], '-', markevery=10)
        else:
            plt.plot(self.conv2[0], '-', markersize=10)
        plt.plot(conv_rate[i], '--', color='red', label=r'$\mathcal{O}(\frac{1}{K})$')
        plt.semilogy()
        plt.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'conv_log2.png'), bbox_inches="tight")

        self.conv_sum = [[np.sum(self.conv[i][:k]) for k in range(1, len(self.conv[i]))] for i in range(len(self.conv))]

        plt.figure(7)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if len(self.conv_sum) > 1:
            for i in range(len(self.conv_sum)):
                plt.plot(self.conv_sum[i], markevery=10)
        else:
            plt.plot(self.conv_sum[0], '-o', markersize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(os.path.join(save_path, 'conv_log_sum.png'), bbox_inches="tight")


        plt.figure(8)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.lip_algo)):
            plt.plot(self.lip_algo[i], '-o', markersize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'lip_algo.png'))

        plt.figure(9)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.lip_D)):
            plt.plot(self.lip_D[i], '-o', markersize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'lip_D.png'))

        plt.figure(10)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.lip_Dg)):
            plt.plot(self.lip_Dg[i], '-o', markersize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'lip_Dg.png'))


    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataset_path', type=str, default='../datasets')
        parser.add_argument('--pretrained_checkpoint', type=str,default='../GS_denoising/ckpts/prox-DRUNET_S-good.ckpt')
        parser.add_argument('--PnP_algo', type=str, default='pGSPnP-PGD')
        parser.add_argument('--dataset_name', type=str, default='CBSD68')
        parser.add_argument('--sigma_denoiser', type=float)
        parser.add_argument('--sigma_k_denoiser', type=float)
        parser.add_argument('--noise_level_img', type=float, default=2.55)
        parser.add_argument('--maxitr', type=int, default=1000)
        parser.add_argument('--alpha', type=float, default=1)
        parser.add_argument('--lamb', type=float)
        parser.add_argument('--n_images', type=int, default=68)
        parser.add_argument('--relative_diff_Psi_min', type=float, default=1e-8)
        parser.add_argument('--inpainting_init', dest='inpainting_init', action='store_true')
        parser.set_defaults(inpainting_init=False)
        parser.add_argument('--precision', type=str, default='simple')
        parser.add_argument('--n_init', type=int, default=10)
        parser.add_argument('--patch_size', type=int, default=256)
        parser.add_argument('--extract_curves', dest='extract_curves', action='store_true')
        parser.set_defaults(extract_curves=False)
        parser.add_argument('--extract_images', dest='extract_images', action='store_true')
        parser.set_defaults(extract_images=False)
        parser.add_argument('--print_each_step', dest='print_each_step', action='store_true')
        parser.set_defaults(print_each_step=False)
        parser.add_argument('--no_grad_matching', dest='grad_matching', action='store_false')
        parser.set_defaults(grad_matching=True)
        parser.add_argument('--no_data_term', dest='no_data_term', action='store_true')
        parser.set_defaults(no_data_term=False)
        parser.add_argument('--use_hard_constraint', dest='use_hard_constraint', action='store_true')
        parser.set_defaults(no_data_term=False)
        return parser
