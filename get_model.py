import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import timm



def get_model(model_name, image_size=224, pretrained=True, num_classes=1000):
    """
    获取指定的轻量级CNN或Transformer模型用于图像分类，并加载预训练权重
    
    Args:
        model_name (str): 模型名称，必须是model_names列表中的一个
        pretrained (bool): 是否加载预训练权重，默认为True
        num_classes (int): 分类类别数量，默认为1000（ImageNet类别数）
        
    Returns:
        torch.nn.Module: 预训练的模型
    """
    
    # 为每个模型名称指定具体的timm模型标识符
    model_mapping = {
        'MobileNetV2': 'mobilenetv2_100',
        'MobileNetV3': 'mobilenetv3_large_100',
        'EfficientNet': 'efficientnet_b0',
        'ResNet50': 'resnet50',
        'ResNet101': 'resnet101',
        'DeiT': 'deit_tiny_patch16_224',  #Input height (512) doesn't match model (224).
        # 'Tiny-ViT': 'tiny_vit_5m_224',
        'simple-vit': 'vit_small_patch16_224',
        'Inception': 'inception_v3',
        'densenet': 'densenet121'
        # 'efficientvit':'efficientvit_b0'
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"不支持的模型名称: {model_name}. 支持的模型: {list(model_mapping.keys())}")
    
    # 获取模型的具体标识符
    model_id = model_mapping[model_name]
    
    try:
        # 使用timm创建模型并加载预训练权重
        if model_id in ['deit_tiny_patch16_224','vit_small_patch16_224']:
            model = timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes,img_size=image_size)
        else:
            model = timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)
        print(f"成功加载 {model_name} 模型 (timm标识符: {model_id}), 预训练权重: {pretrained}")
        return model
    except Exception as e:
        print(f"加载模型时出错: {e}")
        # 尝试查找类似的可用模型
        similar_models = [m for m in timm.list_models() if model_name.lower().replace('-', '') in m.lower()]
        if similar_models:
            print(f"可用的类似模型: {similar_models[:5]}")
        raise

if __name__ == "__main__":
    # 示例：加载预训练的MobileNetV2模型
    model_names = ['MobileNetV2','MobileNetV3','ResNet50','ResNet101','EfficientNet','DeiT','Tiny-ViT','simple-vit','inception','densenet']
    for m in model_names:
        model = get_model(m)
    
        # 示例：使用模型进行前向传播
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"输入形状: {dummy_input.shape}, 输出形状: {output.shape}")
