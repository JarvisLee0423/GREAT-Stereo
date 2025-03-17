import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Any, Union


def sigmoid(inputs: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-inputs))


def mouse_click_event(event: Any, x: Union[int, float], y: Union[int, float], flag: Any, param: Any) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        param["clicked_x"] = x
        param["clicked_y"] = y
        param["clicked"] = True


def cv_visualizer(model: nn.Module, left_img: torch.Tensor, right_img: torch.Tensor, disp_gt: torch.Tensor, inputs: tuple, dataset: str, mouse_event: bool=True) -> None:
    # Preprocess the input data.
    disp_init = inputs[0]
    disp_pr = inputs[1]
    if dataset != "sceneflow":
        disp_gt = disp_pr.detach().clone()
    _, dim, _, w_init = disp_init.shape
    _, _, _, w_pr = disp_pr.shape
    if w_pr != w_init * 4 or dim != 1:
        disp_init = None
    cost_volumes = {}
    batch, height, width = inputs[-1][1:]
    for key, value in inputs[-1][0].items():
        _, c, t, w2 = value[0].shape
        temp = value[0].reshape(batch, height, width, c, t, w2).squeeze(4).permute(0, 3, 4, 1, 2)
        if temp.shape[1] > 1:
            cost_volumes[key] = temp.mean(1, keepdim=True)
            for i in range(temp.shape[1]):
                cost_volumes[key + f"_group_{i + 1}"] = temp[:, i].unsqueeze(1)
        else:
            cost_volumes[key] = temp.mean(1, keepdim=True)
        if key == "ccv" and "great" in str(type(model)):
            b, c, d, h, w = temp.shape
            left_feat = getattr(getattr(getattr(model, "excitive_attention_volume"), "eav_agg"), "cv_left_feat")
            temp = torch.einsum(
                "bid, bjd -> bij",
                rearrange(temp, "b c d h w -> (b h w) d c"),
                rearrange(left_feat, "b c h w -> (b h w) c").unsqueeze(-2),
            ) * (c ** -0.5)
            cost_volumes[key + "_softmax"] = rearrange(temp, "(b h w) d c -> b c d h w", b=b, h=h, w=w)
        if key == "geo" and "igev" in str(type(model)):
            cost_volumes[key + "_softmax"] = getattr(model, "classifier")(temp)
    
    if width != left_img.shape[-1]:
        scale = width / left_img.shape[-1]
        left_img = F.interpolate(left_img, scale_factor=scale, mode="bilinear", align_corners=True)
        right_img = F.interpolate(right_img, scale_factor=scale, mode="bilinear", align_corners=True)
        disp_gt = F.interpolate(disp_gt, scale_factor=scale, mode="nearest") * scale
        disp_pr = F.interpolate(disp_pr, scale_factor=scale, mode="nearest") * scale
    
    left_img = left_img.squeeze().permute(1, 2, 0).cpu().numpy()
    right_img = right_img.squeeze().permute(1, 2, 0).cpu().numpy()
    disp_gt = disp_gt.squeeze().cpu().numpy()
    disp_pr = disp_pr.squeeze().detach().cpu().numpy()
    disp_pr_error = np.abs(disp_gt - disp_pr)
    if disp_init is not None:
        disp_init = disp_init.squeeze().detach().cpu().numpy()
        disp_init_error = np.abs(disp_gt - disp_init)

    # Get the clicked point.
    params = {"clicked_x": -1, "clicked_y": -1, "clicked": False}
    temp_img = np.zeros((left_img.shape[0], 10, 3))
    input_img = np.concatenate((left_img, temp_img, right_img), axis=1)
    cv2.imshow("Input Images", cv2.cvtColor(np.uint8(input_img), cv2.COLOR_RGB2BGR))
    if mouse_event:
        cv2.setMouseCallback("Input Images", mouse_click_event, param=params)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        if not mouse_event:
            params["clicked_y"] = eval(input("Please input the y coordinate for the pixel (-1 for quit): "))
            params["clicked_x"] = eval(input("Please input the x coordinate for the pixel (-1 for quit): "))
            params["clicked"] = True
            if params["clicked_y"] == -1 or params["clicked_x"] == -1:
                break

        if params["clicked"]:
            w = params['clicked_x']
            h = params['clicked_y']
            print(f"Clicked at (height, width): ({h}, {w}).")
            
            try:
                delta_w = disp_gt[h, w]
                left_img_marked = cv2.circle(cv2.cvtColor(np.uint8(left_img), cv2.COLOR_RGB2BGR), (w, h), 3, (0, 0, 255), -1)
                right_img_marked = cv2.circle(cv2.cvtColor(np.uint8(right_img), cv2.COLOR_RGB2BGR), (round(w - delta_w), h), 3, (0, 0, 255), -1)
            
                # Visualize the cost volumes.
                for key, value in cost_volumes.items():
                    gt = disp_gt[h, w]
                    pr = disp_pr[h, w]
                    if disp_init is not None:
                        init = disp_init[h, w]
                    if key == "apc" or key in ["local_cv", "global_cv", "global_cv_softmax"]:
                        print(f"Original disparity: gt->{gt} | pred->{pr}", end="")
                        if disp_init is not None:
                            print(f" | init->{init}")
                        else:
                            print("\n", end="")
                        gt = w - gt
                        pr = w - pr
                        if disp_init is not None:
                            init = w - init

                    cv_vector = value.squeeze().permute(1, 2, 0)[h, w, :].detach().cpu().numpy()
                    # cv_vector = (cv_vector - np.min(cv_vector)) / (np.max(cv_vector) - np.min(cv_vector))
                    # cv_vector = sigmoid(cv_vector)

                    plt.figure(figsize=(8, 6))
                    plt.plot(cv_vector, linestyle="-", color="b", label=f"{key.upper()} at ({h}, {w})")
                    plt.axvline(x=gt, color="g", linestyle="-", linewidth=2, label=f"GT Disp ({gt})")
                    plt.axvline(x=pr, color="r", linestyle="--", linewidth=2, label=f"Pred Disp ({pr})")
                    if disp_init is not None:
                        plt.axvline(x=init, color="y", linestyle="-.", linewidth=2, label=f"Pred init Disp ({init})")
                    plt.title(f"{key.upper()} Cost Volume Visualization at Pixel ({h}, {w})")
                    plt.xlabel("Disp Candidate")
                    plt.ylabel("Distribution")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"./{key}_vis.png")
                    plt.show()
            except IndexError:
                print("Point coordinates out of the boundary.")
                params["clicked"] = False
                continue

            cv2.imwrite("./left.png", left_img_marked)
            cv2.imwrite("./right.png", right_img_marked)
            cv2.imwrite("./disp_gt.png", cv2.applyColorMap(np.uint8(disp_gt / disp_gt.max() * 255.0), cv2.COLORMAP_JET))
            cv2.imwrite("./disp_pr.png", cv2.applyColorMap(np.uint8(disp_pr / disp_gt.max() * 255.0), cv2.COLORMAP_JET))
            cv2.imwrite("./disp_pr_error.png", cv2.applyColorMap(np.uint8(disp_pr_error / 10 * 255.0), cv2.COLORMAP_JET))
            if disp_init is not None:
                cv2.imwrite("./disp_init.png", cv2.applyColorMap(np.uint8(disp_init / disp_gt.max() * 255.0), cv2.COLORMAP_JET))
                cv2.imwrite("./disp_init_error.png", cv2.applyColorMap(np.uint8(disp_init_error / 10 * 255.0), cv2.COLORMAP_JET))
        params["clicked"] = False

    cv2.destroyAllWindows()
