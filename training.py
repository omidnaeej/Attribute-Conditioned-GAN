import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

def train_gan(config, dataloader, netG, netD, netC, device):
    os.makedirs("model_artifacts", exist_ok=True)

    optG = optim.Adam(netG.parameters(), lr=config["gan"]["lr_g"], betas=(config["gan"]["beta1"], 0.999))
    optD = optim.Adam(netD.parameters(), lr=config["gan"]["lr_d"], betas=(config["gan"]["beta1"], 0.999))
    optC = optim.Adam(netC.parameters(), lr=0.001)

    criterion = nn.BCELoss()
    criterion_cls = nn.CrossEntropyLoss()

    print(f"--- Starting GAN Training for {config['gan']['epochs']} Epochs ---")

    for epoch in tqdm(range(config["gan"]["epochs"]), desc="GAN Epochs", leave=True):
        for i, (data, attr) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} Batches", leave=False)):
            netD.zero_grad()
            real_images = data.to(device)
            b_size = real_images.size(0)

            # Train D: real
            label = torch.full((b_size,), config["gan"]["label_smoothing"], device=device)
            output = netD(real_images)
            errD_real = criterion(output, label)
            errD_real.backward()

            # Train D: fake
            noise = torch.randn(b_size, config["gan"]["latent_dim"], 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(0.0)
            output = netD(fake_images.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optD.step()

            # Train G
            netG.zero_grad()
            label.fill_(1.0)
            output = netD(fake_images)
            errG = criterion(output, label)
            errG.backward()
            optG.step()

            # Train classifier on real
            netC.zero_grad()
            attr = attr.to(device)
            preds = netC(real_images)
            errC = 0
            for idx, pred_head in enumerate(preds):
                if idx < attr.shape[1]:
                    errC += criterion_cls(pred_head, attr[:, idx])
            errC.backward()
            optC.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch}][Batch {i}] Loss_D: {(errD_real + errD_fake).item():.4f} Loss_G: {errG.item():.4f}")

    torch.save(netG.state_dict(), "model_artifacts/generator_pretrained.pth")
    torch.save(netD.state_dict(), "model_artifacts/discriminator_frozen.pth")
    torch.save(netC.state_dict(), "model_artifacts/classifier_frozen.pth")
    print("GAN Training Complete. Weights Saved.")

def train_rl(config, netG, netD, netC, agent, device, num_classes_list):
    print("--- Starting RL Agent Training ---")

    netG.load_state_dict(torch.load("model_artifacts/generator_pretrained.pth", map_location=device))
    netD.load_state_dict(torch.load("model_artifacts/discriminator_frozen.pth", map_location=device))
    netC.load_state_dict(torch.load("model_artifacts/classifier_frozen.pth", map_location=device))

    for p in netG.parameters(): p.requires_grad = False
    for p in netD.parameters(): p.requires_grad = False
    for p in netC.parameters(): p.requires_grad = False

    netG.eval(); netD.eval(); netC.eval()
    agent.train()

    optAgent = optim.Adam(agent.parameters(), lr=config["rl"]["lr_agent"])
    num_episodes = config["rl"]["rl_epochs"] * 10
    criterion_cls = nn.CrossEntropyLoss()

    for episode in tqdm(range(num_episodes), desc="RL Training Episodes", leave=True):
        bs = config["data"]["batch_size"]

        target_indices_list = []
        for num_classes in num_classes_list:
            target_indices_list.append(torch.randint(0, num_classes, (bs, 1), device=device))
        target_indices = torch.cat(target_indices_list, dim=1)

        target_attrs_for_agent = target_indices.float()

        optAgent.zero_grad()

        z_action = agent(target_attrs_for_agent)
        z_reshaped = z_action.view(bs, config["gan"]["latent_dim"], 1, 1)

        gen_imgs = netG(z_reshaped)

        d_out = netD(gen_imgs)
        reward_quality = torch.mean(d_out)

        c_outs = netC(gen_imgs)
        loss_meta = 0
        for idx, pred in enumerate(c_outs):
            if idx < target_indices.shape[1]:
                loss_meta += criterion_cls(pred, target_indices[:, idx])

        total_loss = -(config["rl"]["alpha_quality"] * reward_quality) + (config["rl"]["alpha_meta"] * loss_meta)
        total_loss.backward()
        optAgent.step()

        if episode % 20 == 0:
            print(f"RL Episode {episode}: Loss {total_loss.item():.4f}")

    torch.save(agent.state_dict(), "model_artifacts/agent_final.pth")
    print("RL Training Complete.")
