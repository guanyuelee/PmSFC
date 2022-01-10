import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
x_labels = ['Layer 3', 'Layer 5', 'Layer 6', 'Layer 7']
models = ['baseline', 'classifier', 'encoder', 'discriminator']
LPIPS = [
    [0.6235, 0.6440, 0.6657, 0.6305],
    [0.2964, 0.4226, 0.3874, 0.3881],
    [0.3002, 0.2475, 0.3266, 0.4016],
    [0.1147, 0.1277, 0.1508, 0.1223]
]

PSNR = [
    [9.72, 10.45, 6.52, 6.09],
    [20.41, 16.66, 15.5, 11.31],
    [18.74, 17.82, 15.38, 11.22],
    [21.10, 18.74, 17.59, 16.09]
]


def main():
    plt.rcParams['axes.grid'] = True
    fig, (psnr_ax, lpips_ax) = plt.subplots(1, 2)
    plt.setp((psnr_ax, lpips_ax), xticks=x, xticklabels=x_labels)
    psnr_ax.set_title('PSNR')
    psnr_ax.set_xlim([0.5, 4.5])
    psnr_ax.plot(x, PSNR[0], 's-', color='#3484BA', label="baseline")
    psnr_ax.plot(x, PSNR[1], 'o-', color='#FF7F0E', label="classifier")
    psnr_ax.plot(x, PSNR[2], 'x-', color='#2B9F2B', label="encoder")
    psnr_ax.plot(x, PSNR[3], '^-', color='#D62728', label="discriminator")
    lpips_ax.set_title('LPIPS')
    lpips_ax.set_xlim([0.5, 4.5])
    lpips_ax.set_ylim([0, 1])
    lpips_ax.plot(x, LPIPS[0], 's-', color='#3484BA', label="baseline")
    lpips_ax.plot(x, LPIPS[1], 'o-', color='#FF7F0E', label="classifier")
    lpips_ax.plot(x, LPIPS[2], 'x-', color='#2B9F2B', label="encoder")
    lpips_ax.plot(x, LPIPS[3], '^-', color='#D62728', label="discriminator")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    main()
