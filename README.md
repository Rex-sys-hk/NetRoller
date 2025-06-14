# NetRoller

This is the official repository of

**NetRoller: Interfacing General and Specialized Models for End-to-End Autonomous Driving**

[Ren Xin](https://rex-sys-hk.github.io), [Hongji Liu](http://liuhongji.site), [Xiaodong Mei](), [Wenru Liu]() and [Jun Ma](https://personal.hkust-gz.edu.cn/junma/index.html)


<!-- <p align="left">
<a href="https://rex-sys-hk.github.io/pub_webs/PlanScope/">
<img src="https://img.shields.io/badge/Project-Page-blue?style=flat">
</a>
<a href='https://arxiv.org/abs/2411.00476' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=wihte' alt='arXiv PDF'>
</a>
</p> -->

![overview](doc/overview.png)

## Abstract

In this work, we explored the design of adapter to facilitate the seamless integration of Vision Language Models(VLMs, AKA. GMs) and Specialized driving Models(SMs). Specifically, key components of  are organized into three key stages:
- Harvests semantically rich and computationally efficient representations from the reasoning processes of LLMs using an early stopping mechanism, which preserves critical insights on driving context while maintaining low overhead. 
- Applies learnable query embeddings, nonsensical embeddings, and positional layer embeddings to facilitate robust and efficient cross-modality translation.
- Employs computationally efficient Query Shift and Feature Shift mechanisms to enhance the performance of the driving model through few-epoch fine-tuning.

Based on the mechanisms formalized in these three stages, NetRoller enables specialized driving models to operate at their native frequencies while maintaining situational awareness of the VLMs.

Experiments conducted on the nuScenes dataset demonstrate that integrating GM through NetRoller significantly improves human similarity and safety in planning tasks, and it also achieves noticeable precision improvements in detection and mapping tasks for end-to-end autonomous driving.

## Concept Comparison with Exsting Methods
![concept_comarison](doc/concept_comparison.png)
    This diagram illustrates the evolution of the asynchronous frameworks between GMs and SMs. In mode~(a), the traditional approach involves generating textual information using GM, which is then parsed by regular expressions and displayed as received by the fast system. This method relies heavily on predefined patterns and interpretable explicit interfaces.
    Some advanced studies have adopted mode~(b), where selected feature vectors from GMs are inter-modally generated as the latent variables required by SMs. This approach enhances the robustness of information across modalities but still faces challenges in accurately capturing the nuances of the feature vectors and establishing their relevance to SM.
    We propose mode~(c), which fully leverages the latent variables generated during the reasoning process. These latent variables are extracted and robustly translated by a carefully designed module and can be applied to arbitrary targeted feature streams in SM. This method improves the efficiency and accuracy of information transfer, consequently enabling systems to handle complex tasks.


## Performance Comparison with Exsting Methods

![comarison](doc/comparison.png)
![qualitative](doc/qualitative.png)


## Setup Environment

Comming soon.

## Checkpoint


<!-- | Model            | Download |
| ---------------- | -------- |
| Pluto-aux-nocil-m6-baseline  | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EYkVd-OcOTFLlP5KE7ZnG-0BrluObe4vd7jNAhHeKtmcjw?e=UBmqf1)|
| PlanScope-Ih10-DWT | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EXjVIgwKh3hCmMfJ-rQArcABRn3tH1RZhptPOLYRJjkS2A?e=scYt4e)    |
| PlanScope-Mh10-DWH | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EXVaD_lc3kJBtUxGSQBBgPwBl8isEQzRaDtfrJ-geDB-XQ?e=pnbSPy)    |
| PlanScope-Mh20-DWT | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EajN1DzBjKhMg4GiqkuuHuoBGilZzJbkK5QiPD9_GuoDLQ?e=BgidZM)    |
| --- |
| PlanScope-Th20 | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EcHd8CFgBH1JqKT9yMyPsr0BukUsXTjfJpNSik_vQQrsLw?e=48VbzA)    |
| PlanScope-timedecay | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EdMfIvFKuFlLh-SyHVvMB74Bs3TxH5hEp3HCSU34b6yAjg?e=KmVDGh)    |
| PlanScope-timenorm | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/EUMawRA-i-NIimhVp_I_Ft8BeuHWrCJzsVXb-E4BEMMQuA?e=0uRrDN)    |
| --- |
| Pluto-1M-aux-cil-m12-original | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jchengai_connect_ust_hk/EaFpLwwHFYVKsPVLH2nW5nEBNbPS7gqqu_Rv2V1dzODO-Q?e=LAZQcI)    |
| PlanScope-timenorm-cil-m12 | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/rxin_connect_ust_hk/Ed863-9h9ZtFm145JyWGjCIBbF-rInj8P2smuXeG0SAPsg?e=g860Ho)    | -->

Comming soon.

## Evaluation

Comming soon.


## To Do

The code is under cleaning and will be released gradually.

- [ ] Improve docs
- [ ] Tutorial
- [ ] Evaluation
- [ ] Checkpoint
- [ ] Source
- [x] Initial repo & paper

## Citation

If you find this repo useful, please consider giving us a star 🌟 and citing our related paper.

```bibtex
Coming soon.
```

## Thanks
- [DriveLM](https://github.com/OpenDriveLab/DriveLM)
- [VAD](https://github.com/hustvl/VAD)
- [Circuit_Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)


<!-- ## Special Announcement (Updated on 4 March 2025)

Our approach has achieved a CLS-NR score of 91.32% without rule-based post-processing, which currently is the highest score in pure-model-mode. 
However, the main objective is to find a general method for addressing horizon fusing problem, thus enhance the performance of planning models during execution. -->

<!-- This work investigates a technique to enhance the performance of planning models in a pure learning framework. We have deliberately omitted the rule-based pre- and post-processing modules from the baseline approach to mitigate the impact of artificially crafted rules, as claimed in our paper. A certain unauthorized publication led to **inaccuracies in the depiction of its state-of-the-art (SOTA) capabilities**. We hereby clarify this to prevent misunderstanding.

Nevertheless, the method introduced in our article is worth trying and could potentially serve as an add-on to augment the performance of the models you are developing, especially when the dataset is small. We are open to sharing and discussing evaluation results to foster a collaborative exchange. -->

## Others

