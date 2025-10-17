struct VSInput
{
    float3 inPosition;
    float3 inColor;
    float2 inTexCoord;
    float3 inNormal;
};

struct UniformBuffer
{
    float4x4 view;
    float4x4 proj;
    float4 lightPosition;
    float4 lightColor;
    float4 viewPosition;
};
ConstantBuffer<UniformBuffer> ubo;

struct PushConstants
{
    float4x4 model;
}
[vk::push_constant]
PushConstants pc;

struct VSOutput
{
    float4 pos : SV_Position;
    float3 fragColor;
    float2 fragTexCoord;
    float3 fragNormal;
    float3 fragPosWorld;
}

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
    VSOutput output;
    output.pos = mul(ubo.proj, mul(ubo.view, mul(pc.model, float4(input.inPosition, 1.0))));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    output.fragPosWorld = mul(pc.model, float4(input.inPosition, 1.0)).xyz;
    // TODO: This is incorrect for non-uniform scaling
    output.fragNormal = normalize(mul(pc.model, float4(input.inNormal, 0.0))).xyz;
    return output;
}

Sampler2D texture;

[shader("fragment")]
float4 fragMain(VSOutput vertIn)
    : SV_Target
{
    float3 ambient = 0.1 * ubo.lightColor.rgb;

    float3 lightDir = normalize(ubo.lightPosition.xyz - vertIn.fragPosWorld);
    float3 diffuse = max(dot(vertIn.fragNormal, lightDir), 0.0) * ubo.lightColor.rgb;

    float3 finalColor = (ambient + diffuse) * texture.Sample(vertIn.fragTexCoord).rgb;

    return float4(vertIn.fragColor * finalColor, 1.0);
}
