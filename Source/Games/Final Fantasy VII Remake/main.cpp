#define GAME_FF7_REMAKE 1

#define LUMA_PATCH_BYTECODE_SYNC 0
#define CHECK_GRAPHICS_API_COMPATIBILITY 1
// Needed to re-draw the post processing passes into the output resolution textures with a fixed render scale (see "DrawFixedScaleTwinPass()")
#define ENABLE_POST_DRAW_DISPATCH_CALLBACK 1
#ifdef _DEBUG
#define ALLOW_SHADERS_DUMPING 1
#define ALLOW_SHADER_PATCHES_DUMPING 1
#endif

#include <chrono>
#include <random>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <d3d11TokenizedProgramFormat.hpp>
#include "includes\settings.hpp"
#include "..\..\Core\core.hpp"

namespace
{
#if ENABLE_FAST_NOISE_TEXTURES || LUMA_HAS_RECIPE_PROVIDERS
   static constexpr uint32_t kCoreFastNoiseTextureHash = "FAST Noise"_h;
#endif
#if LUMA_HAS_RECIPE_PROVIDERS
   static constexpr const char* kDitheringFixRecipe = "PS/dithering_fix.recipe.yml";
#endif
   static constexpr uint32_t SSAO_TEMPORAL_SHADER_HASH = 0x04BFE575;

   float2 projection_jitters = {0, 0};
   float vert_fov = 60.f * (M_PI / 180.f);
   float near_plane = 0.1f;
   std::unique_ptr<float4[]> downsample_buffer_data;
   std::unique_ptr<float4[]> upsample_buffer_data;

   using FF7RDrawShaderHashesList = ShaderHashesList<ShaderHashesCount::Multiple, ShaderHashesStages::Graphics>;

   FF7RDrawShaderHashesList shader_hashes_TAA;
   ShaderHashesList shader_hashes_Title;
   ShaderHashesList shader_hashes_MotionVectors;
   FF7RDrawShaderHashesList shader_hashes_DOF;
   FF7RDrawShaderHashesList shader_hashes_Motion_Blur;
   ShaderHashesList shader_hashes_Downsample_Bloom;
   ShaderHashesList shader_hashes_Bloom;
   ShaderHashesList shader_hashes_MenuSlowdown;
   ShaderHashesList shader_hashes_Tonemap;
   ShaderHashesList shader_hashes_Velocity_Flatten;
   ShaderHashesList shader_hashes_Velocity_Gather;
   FF7RDrawShaderHashesList shader_hashes_Output_HDR;
   FF7RDrawShaderHashesList shader_hashes_Output_SDR;
   FF7RDrawShaderHashesList shader_hashes_SSAO_Temporal;

#if ENABLE_FRAMEGEN
   // Map from game PS hash to native shader key for Output HDR UI composition pass
   std::unordered_map<uint32_t, size_t> output_hdr_ui_shader_keys;
   // Map from game PS hash to native shader key for Output HDR tonemap pass
   std::unordered_map<uint32_t, size_t> output_hdr_tonemap_shader_keys;
   // Map from game PS hash to native shader key for Output SDR UI composition pass
   std::unordered_map<uint32_t, size_t> output_sdr_ui_shader_keys;
   // Map from game PS hash to native shader key for Output SDR tonemap pass
   std::unordered_map<uint32_t, size_t> output_sdr_tonemap_shader_keys;
#endif // ENABLE_FRAMEGEN

   const uint32_t CBPerViewGlobal_buffer_size = 4096;
   std::atomic<bool> can_sharpen = true;
   float enabled_dithering_fix = 1.f; // Legacy bytecode dithering fix, only used without recipe providers (see the "Dithering Fix" recipe)
   float sr_custom_pre_exposure = 0.f; // Ignored at 0
   float ignore_warnings = 0.f;

   Luma::Settings::Settings settings = {
      new Luma::Settings::Section{
         .label = "Post Processing",
         .settings = {
            new Luma::Settings::Setting{
               .key = "FXBloom",
               .binding = &cb_luma_global_settings.GameSettings.custom_bloom,
               .type = Luma::Settings::SettingValueType::FLOAT,
               .default_value = 50.f,
               .can_reset = true,
               .label = "Bloom",
               .tooltip = "Bloom strength multiplier. Default is 50.",
               .min = 0.f,
               .max = 100.f,
               .parse = [](float value)
               { return value * 0.02f; } // Scale down to 0.0-2.0 for the shader
            },
            new Luma::Settings::Setting{
               .key = "FXVignette",
               .binding = &cb_luma_global_settings.GameSettings.custom_vignette,
               .type = Luma::Settings::SettingValueType::FLOAT,
               .default_value = 50.f,
               .can_reset = true,
               .label = "Vignette",
               .tooltip = "Vignette strength multiplier. Default is 50.",
               .min = 0.f,
               .max = 100.f,
               .parse = [](float value)
               { return value * 0.02f; }
            },
            new Luma::Settings::Setting{
               .key = "FXFilmGrain",
               .binding = &cb_luma_global_settings.GameSettings.custom_film_grain_strength,
               .type = Luma::Settings::SettingValueType::FLOAT,
               .default_value = 50.f,
               .can_reset = true,
               .label = "Film Grain",
               .tooltip = "Film grain strength multiplier. Default is 50, for Vanilla look set to 0.",
               .min = 0.f,
               .max = 100.f,
               //.is_visible = []() { return cb_luma_global_settings.DisplayMode == DisplayModeType::HDR; },
               .parse = [](float value)
               { return value * 0.02f; }
            },
            new Luma::Settings::Setting{
               .key = "FXRCAS",
               .binding = &cb_luma_global_settings.GameSettings.custom_sharpness_strength,
               .type = Luma::Settings::SettingValueType::FLOAT,
               .default_value = 50.f,
               .can_reset = true,
               .label = "Sharpness",
               .tooltip = "RCAS strength multiplier. Default is 50, for Vanilla look set to 0.",
               .min = 0.f,
               .max = 100.f,
               .is_enabled = []()
               { return (sr_user_type != SR::UserType::None) && can_sharpen; },
               .is_visible = []()
               { return sr_user_type != SR::UserType::None; },
               .parse = [](float value)
               { return value * 0.01f; }
            },

            new Luma::Settings::Setting{
               .key = "FXHDRVideos",
               .binding = &cb_luma_global_settings.GameSettings.custom_hdr_videos,
               .type = Luma::Settings::SettingValueType::BOOLEAN,
               .default_value = 1,
               .can_reset = true,
               .label = "HDR Videos",
               .tooltip = "Enable or disable HDR video playback. Default is On.",
               .min = 0,
               .max = 1,
               .is_visible = []()
               { return cb_luma_global_settings.DisplayMode == DisplayModeType::HDR; }
            }
         }
      },
      new Luma::Settings::Section{
         .label = "Color Grading",
         .is_visible = []()
         { return cb_luma_global_settings.DisplayMode == DisplayModeType::HDR; },
         .settings = {
            new Luma::Settings::Setting{
               .key = "TonemapType",
               .binding = &cb_luma_global_settings.GameSettings.tonemap_type,
               .type = Luma::Settings::SettingValueType::INTEGER,
               .default_value = 1.f,
               .can_reset = true,
               .label = "Tone Map Type",
               .tooltip = "Tone mapping algorithm to use",
               .labels = {"Vanilla","Neutwo"},
               .min = 0.f,
               .max = 1.f,
               .is_enabled = []()
               { return cb_luma_global_settings.DisplayMode == DisplayModeType::HDR; },
               .is_visible = []()
               { return cb_luma_global_settings.DisplayMode == DisplayModeType::HDR; }
            },
            new Luma::Settings::Setting{
               .key = "CustomLUTStrength",
               .binding = &cb_luma_global_settings.GameSettings.custom_lut_strength,
               .type = Luma::Settings::SettingValueType::FLOAT,
               .default_value = 100.f,
               .can_reset = true,
               .label = "LUT Strength",
               .tooltip = "LUT strength multiplier.",
               .min = 0.f,
               .max = 100.f,
               .is_visible = []()
               { return cb_luma_global_settings.DisplayMode == DisplayModeType::HDR; },
               .parse = [](float value)
               { return value * 0.01f; }
            },
            new Luma::Settings::Setting{
               .key = "ColorGradeHueCorrection",
               .binding = &cb_luma_global_settings.GameSettings.hue_correction_strength,
               .type = Luma::Settings::SettingValueType::FLOAT,
               .default_value = 0.3f,
               .can_reset = true,
               .label = "Hue Shift",
               .min = 0.f,
               .max = 1.f,
               .format = "%.2f",
            },
            new Luma::Settings::Setting{
               .key = "ColorGradeExposure",
               .binding = &cb_luma_global_settings.GameSettings.exposure,
               .type = Luma::Settings::SettingValueType::FLOAT,
               .default_value = 1.f,
               .can_reset = true,
               .label = "Exposure",
               .min = 0.f,
               .max = 2.f,
               .format = "%.2f",
            },
            new Luma::Settings::Setting{
               .key = "ColorGradeHighlights",
               .binding = &cb_luma_global_settings.GameSettings.highlights,
               .type = Luma::Settings::SettingValueType::FLOAT,
               .default_value = 50.f,
               .can_reset = true,
               .label = "Highlights",
               .min = 0.f,
               .max = 100.f,
               .parse = [](float value)
               { return value * 0.02f;}
            },
            new Luma::Settings::Setting{
               .key = "ColorGradeShadows",
               .binding = &cb_luma_global_settings.GameSettings.shadows,
               .type = Luma::Settings::SettingValueType::FLOAT,
               .default_value = 50.f,
               .can_reset = true,
               .label = "Shadows",
               .min = 0.f,
               .max = 100.f,
               .parse = [](float value)
               { return value * 0.02f; }
            },
            new Luma::Settings::Setting{
               .key = "ColorGradeContrast",
               .binding = &cb_luma_global_settings.GameSettings.contrast,
               .type = Luma::Settings::SettingValueType::FLOAT,
               .default_value = 50.f,
               .can_reset = true,
               .label = "Contrast",
               .min = 0.f,
               .max = 100.f,
               .parse = [](float value)
               { return value * 0.02f; }
            },
            new Luma::Settings::Setting{
               .key = "ColorGradeSaturation",
               .binding = &cb_luma_global_settings.GameSettings.saturation,
               .type = Luma::Settings::SettingValueType::FLOAT,
               .default_value = 50.f,
               .can_reset = true,
               .label = "Saturation",
               .min = 0.f,
               .max = 100.f,
               .parse = [](float value)
               { return value * 0.02f; }
            },
            new Luma::Settings::Setting{
               .key = "ColorGradeHighlightSaturation",
               .binding = &cb_luma_global_settings.GameSettings.highlight_saturation,
               .type = Luma::Settings::SettingValueType::FLOAT,
               .default_value = 50.f,
               .can_reset = true,
               .label = "Highlight Saturation",
               .min = 0.f,
               .max = 100.f,
               .parse = [](float value)
               { return value * 0.02f; }
            },
            new Luma::Settings::Setting{
               .key = "ColorGradeBlowout",
               .binding = &cb_luma_global_settings.GameSettings.blowout,
               .type = Luma::Settings::SettingValueType::FLOAT,
               .default_value = 20.f,
               .can_reset = true,
               .label = "Blowout",
               .min = 0.f,
               .max = 100.f,
               .parse = [](float value)
               { return value * 0.02f; }
            },
            new Luma::Settings::Setting{
               .key = "ColorGradeFlare",
               .binding = &cb_luma_global_settings.GameSettings.flare,
               .type = Luma::Settings::SettingValueType::FLOAT,
               .default_value = 0.f,
               .can_reset = true,
               .label = "Flare",
               .min = 0.f,
               .max = 100.f,
               .parse = [](float value)
               { return value * 0.02f; }
            }
         }
      },
      new Luma::Settings::Section{
         .label = "Advanced Settings", 
         .settings = {
            new Luma::Settings::Setting{
               .key = "IgnoreWarnings", 
               .binding = &ignore_warnings, 
               .type = Luma::Settings::SettingValueType::BOOLEAN, 
               .can_reset = false, .label = "Ignore Warnings", 
               .tooltip = "Ignore warning messages. Default is Off."
            }, 
         }
      }
   };

   static inline bool NearZero(float v, float eps)
   {
      return std::fabs(v) <= eps;
   }

   static inline float ComputeNearPlane(const Math::Matrix44F& view_to_clip)
   {
      return (view_to_clip.m33 - view_to_clip.m32) / (view_to_clip.m22 - view_to_clip.m23);
   }

   static inline float ComputeFovY(const Math::Matrix44F& view_to_clip)
   {
      float eps = 1e-3f;
      if (!NearZero(view_to_clip.m11, eps))
      {
         return atan(1.f / view_to_clip.m11) * 2.0;
      }
      return 0.0f;
   }
} // namespace

// An output resolution copy of a render resolution sized game texture, see "GameDeviceDataFF7Remake::fixed_scale_twins"
struct FixedScaleTwin
{
   com_ptr<ID3D11Texture2D> texture;
   com_ptr<ID3D11ShaderResourceView> srv;
   com_ptr<ID3D11RenderTargetView> rtv;
};

struct GameDeviceDataFF7Remake final : public GameDeviceData
{
#if ENABLE_SR
   // SR
   com_ptr<ID3D11Texture2D> sr_motion_vectors;
   com_ptr<ID3D11ShaderResourceView> sr_motion_vectors_srv; // NEW: SRV for Motion Vectors
   com_ptr<ID3D11Resource> sr_source_color;
   com_ptr<ID3D11Resource> depth_buffer;
   com_ptr<ID3D11RenderTargetView> sr_motion_vectors_rtv;
   std::unique_ptr<SR::SettingsData> sr_settings_data;
   std::unique_ptr<SR::SuperResolutionImpl::DrawData> sr_draw_data;
#endif // ENABLE_SR

   // NEW: Bloom History Resources
   com_ptr<ID3D11Texture2D> prev_bloom_texture;
   com_ptr<ID3D11ShaderResourceView> prev_bloom_srv;
   bool first_bloom_frame = true;
   std::atomic<bool> camera_cut = false;
   std::atomic<bool> has_drawn_title = false;
   std::atomic<bool> has_drawn_taa = false;
   std::atomic<bool> has_drawn_upscaling = false;
   std::atomic<bool> found_per_view_globals = false;
   std::atomic<bool> drs_active = false;
   std::atomic<uint32_t> jitterless_frames_count = 0;
   std::atomic<bool> is_in_menu = true; // Start in menu
   float2 upscaled_render_resolution = {1, 1};
   float resolution_scale = 1.0f;
   uint4 viewport_rect = {0, 0, 1, 1};

   // Fixed render scale upscaling.
   // Under DX11 the game's DRS never engages, so the render resolution can only be lowered with "r.ScreenPercentage",
   // which shrinks the scene textures themselves (instead of rendering in the top left of output sized ones, like DRS does).
   // SR then writes into an output resolution "twin" of the TAA output, and each following post process pass that reads a twin
   // is drawn twice: natively at render resolution (for everything that keeps reading the original textures), and into twins of its targets.
   // The output pass then reads the twins instead of upscaling the render resolution image.
   // Keyed by the game texture (not ref counted, the twin desc is validated on every use)
   std::unordered_map<ID3D11Resource*, FixedScaleTwin> fixed_scale_twins;
   // Game textures whose twin holds this frame's image
   std::unordered_set<ID3D11Resource*> valid_fixed_scale_twins;
   uint2 fixed_scale_render_target_size = {0, 0}; // The scene textures size (UE pads it a bit beyond the render resolution)
   uint2 fixed_scale_render_size = {0, 0};
   uint2 fixed_scale_output_size = {0, 0};
   // SR upscaled into a twin this frame
   bool fixed_scale_upscaling = false;
   // A pass is currently being drawn into twins (or the output pass reads them), the shaders need to treat their inputs as output resolution sized
   bool drawing_fixed_scale_twin = false;
   // Whether the passes are also drawn at render resolution (the game's TAA included). Not needed as long as every pass that reads the upscaled image
   // is one of ours, so it starts off, and turns on for good the first time another pass reads it (that frame will be wrong).
   bool fixed_scale_native_draws = false;
   std::unordered_set<uint32_t> logged_fixed_scale_chain_breakers;

   // Output HDR two-pass split: intermediate hudless render target (R10G10B10A2_UNORM, PQ-encoded)
#if ENABLE_FRAMEGEN
   com_ptr<ID3D11Texture2D> output_hdr_hudless_texture;
   com_ptr<ID3D11ShaderResourceView> output_hdr_hudless_srv;
   com_ptr<ID3D11RenderTargetView> output_hdr_hudless_rtv;
   UINT output_hdr_hudless_width = 0;
   UINT output_hdr_hudless_height = 0;

   void CleanOutputHDRHudlessResources()
   {
      output_hdr_hudless_texture = nullptr;
      output_hdr_hudless_srv = nullptr;
      output_hdr_hudless_rtv = nullptr;
      output_hdr_hudless_width = 0;
      output_hdr_hudless_height = 0;
   }

   // Output SDR two-pass split: intermediate hudless render target (R10G10B10A2_UNORM, linear)
   com_ptr<ID3D11Texture2D> output_sdr_hudless_texture;
   com_ptr<ID3D11ShaderResourceView> output_sdr_hudless_srv;
   com_ptr<ID3D11RenderTargetView> output_sdr_hudless_rtv;
   UINT output_sdr_hudless_width = 0;
   UINT output_sdr_hudless_height = 0;

   void CleanOutputSDRHudlessResources()
   {
      output_sdr_hudless_texture = nullptr;
      output_sdr_hudless_srv = nullptr;
      output_sdr_hudless_rtv = nullptr;
      output_sdr_hudless_width = 0;
      output_sdr_hudless_height = 0;
   }
#endif // ENABLE_FRAMEGEN

};

class FF7Remake final : public Game
{
   static GameDeviceDataFF7Remake& GetGameDeviceData(DeviceData& device_data)
   {
      return *static_cast<GameDeviceDataFF7Remake*>(device_data.game);
   }

public:
   void OnLoad(std::filesystem::path& file_path, bool failed) override
   {
      if (!failed)
      {
         reshade::register_event<reshade::addon_event::map_buffer_region>(FF7Remake::OnMapBufferRegion);
         reshade::register_event<reshade::addon_event::unmap_buffer_region>(FF7Remake::OnUnmapBufferRegion);
         reshade::register_event<reshade::addon_event::bind_viewports>(FF7Remake::OnBindViewports);
         reshade::register_event<reshade::addon_event::bind_scissor_rects>(FF7Remake::OnBindScissorRects);
         // reshade::register_event<reshade::addon_event::update_buffer_region>(FF7Remake::OnUpdateBufferRegion);
      }
   }

   void OnInit(bool async) override
   {
#if LUMA_HAS_RECIPE_PROVIDERS
      Patch::RegisterRecipe({
         .identity = kDitheringFixRecipe,
         .display_name = "Dithering Fix (Experimental)",
         .description = "Fixes dithering that can cause checkered patterns when using Super Resolution.",
         .default_enabled = true,
         .visibility = Patch::RecipeVisibility::Always,
      });
#endif
      GetShaderDefineData(POST_PROCESS_SPACE_TYPE_HASH).SetDefaultValue('1');
      GetShaderDefineData(EARLY_DISPLAY_ENCODING_HASH).SetDefaultValue('1');
      GetShaderDefineData(VANILLA_ENCODING_TYPE_HASH).SetDefaultValue('1');
      GetShaderDefineData(GAMMA_CORRECTION_TYPE_HASH).SetDefaultValue('1');
      GetShaderDefineData(UI_DRAW_TYPE_HASH).SetDefaultValue('1');

      native_shaders_definitions.emplace(CompileTimeStringHash("Decode MVs"), ShaderDefinition{"Luma_MotionVec_UE4_Decode", reshade::api::pipeline_subobject_type::pixel_shader});

#if ENABLE_FRAMEGEN
      // Register Output HDR UI composition pixel shaders (one per permutation)
      // Each is compiled from the wrapper file with UI_COMPOSITION_PASS=1 and the hash-specific define
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output HDR UI 922A71D1"),
         ShaderDefinition{"Luma_FF7R_Output_HDR_UI_Composition", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"UI_COMPOSITION_PASS", "1"}, {"_922A71D1", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output HDR UI A8EB118F"),
         ShaderDefinition{"Luma_FF7R_Output_HDR_UI_Composition", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"UI_COMPOSITION_PASS", "1"}, {"_A8EB118F", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output HDR UI 3A4D858E"),
         ShaderDefinition{"Luma_FF7R_Output_HDR_UI_Composition", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"UI_COMPOSITION_PASS", "1"}, {"_3A4D858E", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output HDR UI D950DA01"),
         ShaderDefinition{"Luma_FF7R_Output_HDR_UI_Composition", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"UI_COMPOSITION_PASS", "1"}, {"_D950DA01", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output HDR UI 5CD12E67"),
         ShaderDefinition{"Luma_FF7R_Output_HDR_UI_Composition", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"UI_COMPOSITION_PASS", "1"}, {"_5CD12E67", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output HDR UI 3B489929"),
         ShaderDefinition{"Luma_FF7R_Output_HDR_UI_Composition", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"UI_COMPOSITION_PASS", "1"}, {"_3B489929", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output HDR UI 8D04181D"),
         ShaderDefinition{"Luma_FF7R_Output_HDR_UI_Composition", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"UI_COMPOSITION_PASS", "1"}, {"_8D04181D", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output HDR UI 6846FF90"),
         ShaderDefinition{"Luma_FF7R_Output_HDR_UI_Composition", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"UI_COMPOSITION_PASS", "1"}, {"_6846FF90", "1"}}});

      // Register Output HDR tonemap pixel shaders (one per permutation)
      // Each is compiled from the tonemap wrapper file with TONEMAP_PASS=1 and the hash-specific define
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output HDR TM 922A71D1"),
         ShaderDefinition{"Luma_FF7R_Output_HDR_Tonemap", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"TONEMAP_PASS", "1"}, {"_922A71D1", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output HDR TM A8EB118F"),
         ShaderDefinition{"Luma_FF7R_Output_HDR_Tonemap", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"TONEMAP_PASS", "1"}, {"_A8EB118F", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output HDR TM 3A4D858E"),
         ShaderDefinition{"Luma_FF7R_Output_HDR_Tonemap", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"TONEMAP_PASS", "1"}, {"_3A4D858E", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output HDR TM D950DA01"),
         ShaderDefinition{"Luma_FF7R_Output_HDR_Tonemap", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"TONEMAP_PASS", "1"}, {"_D950DA01", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output HDR TM 5CD12E67"),
         ShaderDefinition{"Luma_FF7R_Output_HDR_Tonemap", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"TONEMAP_PASS", "1"}, {"_5CD12E67", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output HDR TM 3B489929"),
         ShaderDefinition{"Luma_FF7R_Output_HDR_Tonemap", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"TONEMAP_PASS", "1"}, {"_3B489929", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output HDR TM 8D04181D"),
         ShaderDefinition{"Luma_FF7R_Output_HDR_Tonemap", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"TONEMAP_PASS", "1"}, {"_8D04181D", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output HDR TM 6846FF90"),
         ShaderDefinition{"Luma_FF7R_Output_HDR_Tonemap", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"TONEMAP_PASS", "1"}, {"_6846FF90", "1"}}});

      // Register Output SDR UI composition pixel shaders (one per permutation)
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output SDR UI 506D5998"),
         ShaderDefinition{"Luma_FF7R_Output_SDR_UI_Composition", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"UI_COMPOSITION_PASS", "1"}, {"_506D5998", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output SDR UI F68D39B5"),
         ShaderDefinition{"Luma_FF7R_Output_SDR_UI_Composition", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"UI_COMPOSITION_PASS", "1"}, {"_F68D39B5", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output SDR UI BBB9CE42"),
         ShaderDefinition{"Luma_FF7R_Output_SDR_UI_Composition", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"UI_COMPOSITION_PASS", "1"}, {"_BBB9CE42", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output SDR UI 51E2B894"),
         ShaderDefinition{"Luma_FF7R_Output_SDR_UI_Composition", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"UI_COMPOSITION_PASS", "1"}, {"_51E2B894", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output SDR UI 803889E8"),
         ShaderDefinition{"Luma_FF7R_Output_SDR_UI_Composition", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"UI_COMPOSITION_PASS", "1"}, {"_803889E8", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output SDR UI D96EF76D"),
         ShaderDefinition{"Luma_FF7R_Output_SDR_UI_Composition", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"UI_COMPOSITION_PASS", "1"}, {"_D96EF76D", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output SDR UI 5C2D3A71"),
         ShaderDefinition{"Luma_FF7R_Output_SDR_UI_Composition", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"UI_COMPOSITION_PASS", "1"}, {"_5C2D3A71", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output SDR UI 66162229"),
         ShaderDefinition{"Luma_FF7R_Output_SDR_UI_Composition", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"UI_COMPOSITION_PASS", "1"}, {"_66162229", "1"}}});

      // Register Output SDR tonemap pixel shaders (one per permutation)
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output SDR TM 506D5998"),
         ShaderDefinition{"Luma_FF7R_Output_SDR_Tonemap", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"TONEMAP_PASS", "1"}, {"_506D5998", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output SDR TM F68D39B5"),
         ShaderDefinition{"Luma_FF7R_Output_SDR_Tonemap", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"TONEMAP_PASS", "1"}, {"_F68D39B5", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output SDR TM BBB9CE42"),
         ShaderDefinition{"Luma_FF7R_Output_SDR_Tonemap", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"TONEMAP_PASS", "1"}, {"_BBB9CE42", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output SDR TM 51E2B894"),
         ShaderDefinition{"Luma_FF7R_Output_SDR_Tonemap", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"TONEMAP_PASS", "1"}, {"_51E2B894", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output SDR TM 803889E8"),
         ShaderDefinition{"Luma_FF7R_Output_SDR_Tonemap", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"TONEMAP_PASS", "1"}, {"_803889E8", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output SDR TM D96EF76D"),
         ShaderDefinition{"Luma_FF7R_Output_SDR_Tonemap", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"TONEMAP_PASS", "1"}, {"_D96EF76D", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output SDR TM 5C2D3A71"),
         ShaderDefinition{"Luma_FF7R_Output_SDR_Tonemap", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"TONEMAP_PASS", "1"}, {"_5C2D3A71", "1"}}});
      native_shaders_definitions.emplace(CompileTimeStringHash("FF7R Output SDR TM 66162229"),
         ShaderDefinition{"Luma_FF7R_Output_SDR_Tonemap", reshade::api::pipeline_subobject_type::pixel_shader, nullptr, nullptr,
            {{"TONEMAP_PASS", "1"}, {"_66162229", "1"}}});
#endif // ENABLE_FRAMEGEN

      luma_settings_cbuffer_index = 13;
      luma_data_cbuffer_index = 12;

#if ENABLE_FAST_NOISE_TEXTURES
      shader_hashes_SSAO_Temporal.pixel_shaders.emplace(SSAO_TEMPORAL_SHADER_HASH);
#endif
      draw_callbacks_by_shader_hashes = {
         { shader_hashes_TAA, &OnDraw_TAA },
         { shader_hashes_Output_HDR, &OnDraw_OutputHDR },
         { shader_hashes_Output_SDR, &OnDraw_OutputSDR },
#if ENABLE_FAST_NOISE_TEXTURES
         { shader_hashes_SSAO_Temporal, &OnDraw_SSAO_Temporal },
#endif
         { shader_hashes_Motion_Blur, &OnDraw_DOFAndMotionBlur },
         { shader_hashes_DOF, &OnDraw_DOFAndMotionBlur },
      };
   }

   void OnInitSwapchain(reshade::api::swapchain* swapchain) override
   {
      auto& device_data = *swapchain->get_device()->get_private_data<DeviceData>();
      auto& game_device_data = GetGameDeviceData(device_data);

      // Start from here, we then update it later in case the game rendered with black bars due to forcing a different aspect ratio from the swapchain buffer
      game_device_data.upscaled_render_resolution = device_data.output_resolution;
   }

   void OnCreateDevice(ID3D11Device* native_device, DeviceData& device_data) override
   {
      device_data.game = new GameDeviceDataFF7Remake;
      auto& game_device_data = GetGameDeviceData(device_data);
   }

   std::unique_ptr<std::byte[]> PatchShaderBytecodeSync(const std::byte* code, size_t& size,
      reshade::api::pipeline_subobject_type type,
      uint64_t shader_hash = -1,
      const std::byte* shader_object = nullptr,
      size_t shader_object_size = 0) override
   {
#if LUMA_HAS_RECIPE_PROVIDERS
      // Dithering fix is handled by recipe async; skip the sync path entirely.
      // Other sync patches (tonemap, output, etc.) continue below.
      if (type == reshade::api::pipeline_subobject_type::pixel_shader || type == reshade::api::pipeline_subobject_type::compute_shader)
         return nullptr;
#endif
      if (enabled_dithering_fix == 0.f)
         return nullptr;
      if (type != reshade::api::pipeline_subobject_type::pixel_shader && type != reshade::api::pipeline_subobject_type::compute_shader)
         return nullptr;

      // Pattern: cb1[139].z + literal l(3) used around the ishl sequence
      const std::vector<std::byte> dithering_anchor = {
         std::byte{0x2A}, std::byte{0x80}, std::byte{0x20}, std::byte{0x00}, // CONST_BUFFER, 4-comp SELECT_1 (.z)
         std::byte{0x01}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, // cb index = 1
         std::byte{0x8B}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, // element = 139
         std::byte{0x01}, std::byte{0x40}, std::byte{0x00}, std::byte{0x00}, // IMMEDIATE32 (1 component)
         std::byte{0x03}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}  // literal value = 3
      };

      // Scan for our dithering anchor
      std::vector<std::byte*> anchors = System::ScanMemoryForPattern(code, size, dithering_anchor);
      if (anchors.empty())
         return nullptr;

      // Confirm we are inside an 'ishl' (cb-pattern is +12 bytes from instruction start)
      const std::byte* cb_match = anchors[0];
      if (cb_match < code + 12)
         return nullptr;
      const std::byte* ishl_start = cb_match - 12;

      // Verify ishl opcode token = 0x08000029 -> bytes LE: 29 00 00 08
      if (ishl_start[0] != std::byte{0x29} || ishl_start[1] != std::byte{0x00} ||
          ishl_start[2] != std::byte{0x00} || ishl_start[3] != std::byte{0x08})
         return nullptr;

      constexpr size_t ishl_instruction_size = 32; // keep existing behavior
      constexpr size_t iadd_instruction_size = 32;

      const size_t injection_point_offset = static_cast<size_t>(ishl_start - code) + ishl_instruction_size;

      // check if has SampleGrad/sample_d_indexable instruction - if so, skip dithering fix
      bool has_samplegrad = false;
      {
         const std::vector<std::byte> samplegrad_pattern = {std::byte{uint8_t(D3D10_SB_OPCODE_SAMPLE_D)}, std::byte{0x00}};
         auto samplegrad_hits = System::ScanMemoryForPattern(code, size, samplegrad_pattern, true);
         if (!samplegrad_hits.empty())
         {
            // confirm it's start of instruction
            const size_t offset = static_cast<size_t>(samplegrad_hits[0] - code);
            if (!((offset & 0x3) != 0 || offset + 4 > size))
            {
               has_samplegrad = true;
            }
         }
      }

      // Additional check: Only inject if a DISCARD instruction exists after the ISHL.
      bool discard_found = false;
      if (injection_point_offset + sizeof(uint32_t) <= size)
      {
         const size_t window_begin = injection_point_offset;
         const size_t window_len = size - window_begin;

         const std::vector<std::byte> discard_pat = {std::byte{uint8_t(D3D10_SB_OPCODE_DISCARD)}, std::byte{0x00}};
         auto disc_hits = System::ScanMemoryForPattern(code + window_begin, window_len, discard_pat);

         for (const std::byte* pbyte : disc_hits)
         {
            const size_t offset = static_cast<size_t>(pbyte - code);
            if ((offset & 0x3) != 0 || offset + 4 > size)
               continue;

            const uint32_t opcode_tok0 = *reinterpret_cast<const uint32_t*>(code + offset);
            if (DECODE_D3D10_SB_OPCODE_TYPE(opcode_tok0) != D3D10_SB_OPCODE_DISCARD)
               continue;

            const uint32_t len = DECODE_D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH(opcode_tok0);
            if (len >= 1 && len <= MAX_D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH)
            {
               discard_found = true;
               break;
            }
         }
      }

      if (discard_found || has_samplegrad)
      {
         const uint32_t dest_token = *reinterpret_cast<const uint32_t*>(ishl_start + 4);
         const uint32_t dest_index = *reinterpret_cast<const uint32_t*>(ishl_start + 8);

         auto dest_to_src_select1 = [](uint32_t token) -> uint32_t
         {
            const uint32_t mask = (token >> 4) & 0xF; // xyzw bits
            uint32_t comp = (mask & 0x1) ? 0u : (mask & 0x2) ? 1u
                                             : (mask & 0x4)  ? 2u
                                                             : 3u;
            token &= ~(0x3u << 2);
            token |= (2u << 2); // SELECT_1
            token &= ~(0x30u);
            token |= ((comp & 3u) << 4);
            return token;
         };

         const uint32_t src0_token = dest_to_src_select1(dest_token);
         const uint32_t src0_index = dest_index;

         const uint32_t cb_token_z = *reinterpret_cast<const uint32_t*>(cb_match + 0);
         const uint32_t cb_index = *reinterpret_cast<const uint32_t*>(cb_match + 4);
         const uint32_t cb_elem = *reinterpret_cast<const uint32_t*>(cb_match + 8);

         auto set_select1_comp = [](uint32_t token, uint32_t comp) -> uint32_t
         {
            token &= ~(0x3u << 2);
            token |= (2u << 2);          // SELECT_1
            token &= ~(0x30u);           // clear selected component (bits 5:4)
            token |= ((comp & 3u) << 4); // set to X/Y/Z/W = 0/1/2/3
            return token;
         };
         const uint32_t cb_token_w = set_select1_comp(cb_token_z, 3u);

         const size_t new_size = size + iadd_instruction_size;
         auto new_code = std::make_unique<std::byte[]>(new_size);

         std::memcpy(new_code.get(), code, injection_point_offset);

         uint32_t iadd[8] = {};
         iadd[0] = ENCODE_D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH(8u) | D3D10_SB_OPCODE_IADD;
         iadd[1] = dest_token;
         iadd[2] = dest_index;
         iadd[3] = src0_token;
         iadd[4] = src0_index;
         iadd[5] = cb_token_w;
         iadd[6] = cb_index;
         iadd[7] = cb_elem;

         std::memcpy(new_code.get() + injection_point_offset, iadd, sizeof(iadd));
         std::memcpy(new_code.get() + injection_point_offset + sizeof(iadd), code + injection_point_offset, size - injection_point_offset);

         size += sizeof(iadd);
         return new_code;
      }

      // No DISCARD => remove dithering: find the LD reading t0 near ISHL and replace with MOV dest, l(1)
      {
         // ld[_aoffimmi(u,v,w)] dest[.mask], srcAddress[.swizzle], srcResource[.swizzle]
         const size_t window_begin = injection_point_offset;
         const size_t window_len = size - window_begin;

         const std::vector<std::byte> ld_pat = {std::byte{uint8_t(D3D10_SB_OPCODE_LD)}, std::byte{0x00}};
         auto ld_hits = System::ScanMemoryForPattern(code + window_begin, window_len, ld_pat);

         for (const std::byte* pbyte : ld_hits)
         {
            size_t offset_to_ld = static_cast<size_t>(pbyte - code);
            if ((offset_to_ld & 0x3) != 0 || offset_to_ld + 4 > size)
               continue;

            const uint32_t* ld_start = reinterpret_cast<const uint32_t*>(code + offset_to_ld);
            const uint32_t opcode_tok0 = ld_start[0];
            if (DECODE_D3D10_SB_OPCODE_TYPE(opcode_tok0) != D3D10_SB_OPCODE_LD)
               continue;

            const uint32_t old_len = DECODE_D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH(opcode_tok0);
            if (old_len < 3 || offset_to_ld + old_len * 4 > size)
               continue;

            // check if extensions
            uint32_t is_ext = 0;
            while (DECODE_IS_D3D10_SB_OPCODE_EXTENDED(ld_start[is_ext]) != 0)
            {
               is_ext++;
            }

            // Operand 0: dest (TEMP). If this is not TEMP we likely misaligned; skip this hit.
            const uint32_t* dest_operand_start = ld_start + 1 + is_ext;
            uint32_t index_represntation = DECODE_D3D10_SB_OPERAND_INDEX_REPRESENTATION(DECODE_D3D10_SB_RESOURCE_DIMENSION(dest_operand_start[0]), dest_operand_start[0]);
            if (index_represntation != D3D10_SB_OPERAND_INDEX_IMMEDIATE32)
               continue;
            is_ext = DECODE_IS_D3D10_SB_OPERAND_EXTENDED(dest_operand_start[0]);
            ASSERT_ONCE(is_ext == 0); // should not be extended for LD
            if (DECODE_D3D10_SB_OPERAND_TYPE(dest_operand_start[0]) != D3D10_SB_OPERAND_TYPE_TEMP)
               continue;

            // Operand 1: uv source. Must be Temp
            const uint32_t* src_operand_start = dest_operand_start + 2 + is_ext;
            index_represntation = DECODE_D3D10_SB_OPERAND_INDEX_REPRESENTATION(DECODE_D3D10_SB_RESOURCE_DIMENSION(src_operand_start[0]), src_operand_start[0]);
            if (index_represntation != D3D10_SB_OPERAND_INDEX_IMMEDIATE32)
               continue;
            is_ext = DECODE_IS_D3D10_SB_OPERAND_EXTENDED(src_operand_start[0]);
            ASSERT_ONCE(is_ext == 0); // should not be extended for LD
            if (DECODE_D3D10_SB_OPERAND_TYPE(src_operand_start[0]) != D3D10_SB_OPERAND_TYPE_TEMP)
               continue;

            // Operand 2: resource. Must be Resource
            const uint32_t* res_operand_start = src_operand_start + 2 + is_ext;
            is_ext = DECODE_IS_D3D10_SB_OPERAND_EXTENDED(res_operand_start[0]);
            ASSERT_ONCE(is_ext == 0); // should not be extended for LD
            if (DECODE_D3D10_SB_OPERAND_TYPE(res_operand_start[0]) != D3D10_SB_OPERAND_TYPE_RESOURCE)
               continue;
            if (DECODE_D3D10_SB_OPERAND_INDEX_DIMENSION(res_operand_start[0]) != D3D10_SB_OPERAND_INDEX_1D)
               continue;
            if (DECODE_D3D10_SB_OPERAND_INDEX_REPRESENTATION(D3D10_SB_OPERAND_INDEX_1D, res_operand_start[0]) !=
                D3D10_SB_OPERAND_INDEX_IMMEDIATE32)
               continue;

            const uint32_t res_index = res_operand_start[1 + is_ext];
            if (res_index != 0)
               continue; // must be resource 0

            // Replace LD with MOV dest, l(1) and pad with NOPs
            auto new_code = std::make_unique<std::byte[]>(size);
            std::memcpy(new_code.get(), code, size);

            std::vector<uint32_t> repl;
            repl.reserve(old_len);

            // placeholder for MOV token0
            repl.push_back(0u);

            // copy dest operand
            repl.push_back(dest_operand_start[0]);
            repl.push_back(dest_operand_start[1]); // index
                                                   // repl.push_back(op_dest[i]);

            // Immediate32 operand: match dest component count (1 or 4)
            const uint32_t dest_tok0 = dest_operand_start[0];
            const auto dest_numc = DECODE_D3D10_SB_OPERAND_NUM_COMPONENTS(dest_tok0);

            if (dest_numc == D3D10_SB_OPERAND_1_COMPONENT)
            {
               const uint32_t imm_tok =
                  ENCODE_D3D10_SB_OPERAND_TYPE(D3D10_SB_OPERAND_TYPE_IMMEDIATE32) |
                  ENCODE_D3D10_SB_OPERAND_NUM_COMPONENTS(D3D10_SB_OPERAND_1_COMPONENT);
               repl.push_back(imm_tok);
               repl.push_back(0x3F000000u); // 1.0f
            }
            else
            {
               const uint32_t imm_tok =
                  ENCODE_D3D10_SB_OPERAND_TYPE(D3D10_SB_OPERAND_TYPE_IMMEDIATE32) |
                  ENCODE_D3D10_SB_OPERAND_NUM_COMPONENTS(D3D10_SB_OPERAND_4_COMPONENT) |
                  ENCODE_D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE(D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_MODE) |
                  D3D10_SB_OPERAND_4_COMPONENT_NOSWIZZLE;
               repl.push_back(imm_tok);
               repl.push_back(0x3F000000u);
               repl.push_back(0x3F000000u);
               repl.push_back(0x3F000000u);
               repl.push_back(0x3F000000u);
            }

            // finalize MOV token0
            const uint32_t mov_len = static_cast<uint32_t>(repl.size());
            repl[0] = ENCODE_D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH(mov_len) | D3D10_SB_OPCODE_MOV;

            // pad with NOPs to preserve original instruction size
            while (repl.size() < old_len)
               repl.push_back(ENCODE_D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH(1u) | D3D10_SB_OPCODE_NOP);

            std::memcpy(reinterpret_cast<uint8_t*>(new_code.get()) + offset_to_ld, repl.data(), old_len * sizeof(uint32_t));
            return new_code;
         }
      }

      // No LD t0 near ISHL found; do nothing for now.
      return nullptr;
   }

   static bool GetEarlyUpscaleTargetResolution(ID3D11DeviceContext* native_device_context, DeviceData& device_data, GameDeviceDataFF7Remake& game_device_data, UINT& width, UINT& height)
   {
      // With a fixed render scale, the post processing targets keep their (render) size, the twins are drawn separately
      if (device_data.has_drawn_main_post_processing || !device_data.has_drawn_sr || !game_device_data.drs_active || !game_device_data.found_per_view_globals || game_device_data.fixed_scale_upscaling)
      {
         return false;
      }

      width = static_cast<UINT>(std::lrintf(game_device_data.upscaled_render_resolution.x));
      height = static_cast<UINT>(std::lrintf(game_device_data.upscaled_render_resolution.y));

      if (width == 0 || height == 0)
      {
         return false;
      }

      // Only resize the main output-sized target. Bloom/exposure mip targets have independent dimensions.
      com_ptr<ID3D11RenderTargetView> render_target_views[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
      com_ptr<ID3D11DepthStencilView> depth_stencil_view;
      native_device_context->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, &render_target_views[0], &depth_stencil_view);
      if (render_target_views[0].get() == nullptr)
      {
         return false;
      }

      com_ptr<ID3D11Resource> render_target_resource;
      render_target_views[0]->GetResource(&render_target_resource);

      com_ptr<ID3D11Texture2D> render_target_texture;
      HRESULT hr = render_target_resource->QueryInterface(&render_target_texture);
      ASSERT_ONCE(SUCCEEDED(hr));
      if (render_target_texture.get() == nullptr)
      {
         return false;
      }

      D3D11_TEXTURE2D_DESC render_target_desc;
      render_target_texture->GetDesc(&render_target_desc);
      return width == render_target_desc.Width && height == render_target_desc.Height;
   }

   static void ResizeCurrentViewportForEarlyUpscaling(ID3D11DeviceContext* native_device_context, UINT width, UINT height)
   {
      UINT num_viewports = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
      D3D11_VIEWPORT current_viewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] = {};
      native_device_context->RSGetViewports(&num_viewports, current_viewports);

      const D3D11_VIEWPORT output_viewport = {0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f};
      const bool already_resized = num_viewports == 1
         && current_viewports[0].TopLeftX == output_viewport.TopLeftX
         && current_viewports[0].TopLeftY == output_viewport.TopLeftY
         && current_viewports[0].Width == output_viewport.Width
         && current_viewports[0].Height == output_viewport.Height
         && current_viewports[0].MinDepth == output_viewport.MinDepth
         && current_viewports[0].MaxDepth == output_viewport.MaxDepth;
      if (!already_resized)
      {
         native_device_context->RSSetViewports(1, &output_viewport);
      }
   }

   static void ResizeCurrentScissorForEarlyUpscaling(ID3D11DeviceContext* native_device_context, UINT width, UINT height)
   {
      UINT num_scissors = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
      D3D11_RECT current_scissors[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] = {};
      native_device_context->RSGetScissorRects(&num_scissors, current_scissors);

      const D3D11_RECT output_scissor = {0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
      const bool already_resized = num_scissors == 1
         && current_scissors[0].left == output_scissor.left
         && current_scissors[0].top == output_scissor.top
         && current_scissors[0].right == output_scissor.right
         && current_scissors[0].bottom == output_scissor.bottom;
      if (!already_resized)
      {
         native_device_context->RSSetScissorRects(1, &output_scissor);
      }
   }

   static void ResizeCurrentViewportAndScissorForEarlyUpscaling(ID3D11DeviceContext* native_device_context, DeviceData& device_data, GameDeviceDataFF7Remake& game_device_data)
   {
      UINT width = 0;
      UINT height = 0;
      if (!GetEarlyUpscaleTargetResolution(native_device_context, device_data, game_device_data, width, height))
      {
         return;
      }

      ResizeCurrentViewportForEarlyUpscaling(native_device_context, width, height);
      ResizeCurrentScissorForEarlyUpscaling(native_device_context, width, height);
   }

   static ID3D11DeviceContext* GetNativeDeviceContext(reshade::api::command_list* cmd_list, com_ptr<ID3D11DeviceContext>& native_device_context)
   {
      SKIP_UNSUPPORTED_DEVICE_API(cmd_list->get_device()->get_api(), nullptr);

      auto* device_child = reinterpret_cast<ID3D11DeviceChild*>(cmd_list->get_native());
      if (device_child == nullptr || FAILED(device_child->QueryInterface(&native_device_context)))
      {
         return nullptr;
      }
      return native_device_context.get();
   }

   static void OnBindViewports(reshade::api::command_list* cmd_list, uint32_t first, uint32_t count, const reshade::api::viewport* viewports)
   {
      if (first != 0 || count == 0)
      {
         return;
      }

      auto& device_data = *cmd_list->get_device()->get_private_data<DeviceData>();
      auto& game_device_data = GetGameDeviceData(device_data);
      com_ptr<ID3D11DeviceContext> native_device_context;
      if (ID3D11DeviceContext* context = GetNativeDeviceContext(cmd_list, native_device_context))
      {
         UINT width = 0;
         UINT height = 0;
         if (GetEarlyUpscaleTargetResolution(context, device_data, game_device_data, width, height))
         {
            ResizeCurrentViewportForEarlyUpscaling(context, width, height);
         }
      }
   }

   static void OnBindScissorRects(reshade::api::command_list* cmd_list, uint32_t first, uint32_t count, const reshade::api::rect* rects)
   {
      if (first != 0 || count == 0)
      {
         return;
      }

      auto& device_data = *cmd_list->get_device()->get_private_data<DeviceData>();
      auto& game_device_data = GetGameDeviceData(device_data);
      com_ptr<ID3D11DeviceContext> native_device_context;
      if (ID3D11DeviceContext* context = GetNativeDeviceContext(cmd_list, native_device_context))
      {
         UINT width = 0;
         UINT height = 0;
         if (GetEarlyUpscaleTargetResolution(context, device_data, game_device_data, width, height))
         {
            ResizeCurrentScissorForEarlyUpscaling(context, width, height);
         }
      }
   }

   static FixedScaleTwin* GetOrCreateFixedScaleTwin(ID3D11Device* native_device, GameDeviceDataFF7Remake& game_device_data, ID3D11Resource* resource, bool unordered_access)
   {
      com_ptr<ID3D11Texture2D> texture;
      if (resource == nullptr || FAILED(resource->QueryInterface(&texture)))
      {
         return nullptr;
      }
      D3D11_TEXTURE2D_DESC desc;
      texture->GetDesc(&desc);
      if (desc.MipLevels != 1 || desc.ArraySize != 1 || desc.SampleDesc.Count != 1)
      {
         return nullptr;
      }
      desc.Width = game_device_data.fixed_scale_output_size.x;
      desc.Height = game_device_data.fixed_scale_output_size.y;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.CPUAccessFlags = 0;
      desc.MiscFlags = 0;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | (unordered_access ? D3D11_BIND_UNORDERED_ACCESS : 0);

      // Pooled targets come and go, don't let stale twins pile up
      if (game_device_data.fixed_scale_twins.size() > 32 && !game_device_data.fixed_scale_twins.contains(resource))
      {
         game_device_data.fixed_scale_twins.clear();
         game_device_data.valid_fixed_scale_twins.clear();
      }

      FixedScaleTwin& twin = game_device_data.fixed_scale_twins[resource];
      if (twin.texture.get())
      {
         D3D11_TEXTURE2D_DESC twin_desc;
         twin.texture->GetDesc(&twin_desc);
         if (twin_desc.Width == desc.Width && twin_desc.Height == desc.Height && twin_desc.Format == desc.Format && (twin_desc.BindFlags & desc.BindFlags) == desc.BindFlags)
         {
            return &twin;
         }
         twin = {};
      }
      if (FAILED(native_device->CreateTexture2D(&desc, nullptr, &twin.texture))
         || FAILED(native_device->CreateShaderResourceView(twin.texture.get(), nullptr, &twin.srv))
         || FAILED(native_device->CreateRenderTargetView(twin.texture.get(), nullptr, &twin.rtv)))
      {
         game_device_data.fixed_scale_twins.erase(resource);
         return nullptr;
      }
      return &twin;
   }

   static void LogFixedScalePass(GameDeviceDataFF7Remake& game_device_data, const ShaderHashesList<OneShaderPerPipeline>& original_shader_hashes, const std::string& message)
   {
      const bool is_compute = Shader::ShaderHashesEmpty(original_shader_hashes.pixel_shaders);
      uint32_t hash = 0;
      if (!is_compute)
      {
         hash = static_cast<uint32_t>(original_shader_hashes.pixel_shaders[0]);
      }
      else if (!Shader::ShaderHashesEmpty(original_shader_hashes.compute_shaders))
      {
         hash = static_cast<uint32_t>(original_shader_hashes.compute_shaders[0]);
      }
      if (game_device_data.logged_fixed_scale_chain_breakers.emplace(hash).second)
      {
         const std::string text = std::string("Fixed scale upscaling: ") + (is_compute ? "compute" : "pixel") + " shader " + Shader::Hash_NumToStr(hash) + " " + message;
         reshade::log::message(reshade::log::level::warning, text.c_str());
      }
   }

   // A pass reads the render resolution version of an upscaled texture: from now on keep drawing those too
   static void RequireFixedScaleNativeDraws(GameDeviceDataFF7Remake& game_device_data, const ShaderHashesList<OneShaderPerPipeline>& original_shader_hashes)
   {
      if (!game_device_data.fixed_scale_native_draws)
      {
         game_device_data.fixed_scale_native_draws = true;
         LogFixedScalePass(game_device_data, original_shader_hashes, "reads the render resolution image, drawing the passes at both resolutions from now on");
      }
   }

   // Stops drawing twins for this frame, the output pass will fall back to the game's own upscaling
   static void BreakFixedScaleChain(GameDeviceDataFF7Remake& game_device_data, const ShaderHashesList<OneShaderPerPipeline>& original_shader_hashes, const char* reason)
   {
      game_device_data.fixed_scale_upscaling = false;
      game_device_data.valid_fixed_scale_twins.clear();
      RequireFixedScaleNativeDraws(game_device_data, original_shader_hashes);
      LogFixedScalePass(game_device_data, original_shader_hashes, std::string(reason) + ", using the game's upscaling this frame");
   }

   static bool ReadsFixedScaleTwin(GameDeviceDataFF7Remake& game_device_data, ID3D11ShaderResourceView* const* srvs)
   {
      for (UINT i = 0; i < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT; i++)
      {
         if (srvs[i] == nullptr)
         {
            continue;
         }
         com_ptr<ID3D11Resource> resource;
         srvs[i]->GetResource(&resource);
         if (game_device_data.valid_fixed_scale_twins.contains(resource.get()))
         {
            return true;
         }
      }
      return false;
   }

   // Binds the twins of this frame's upscaled textures in place of the game ones, returns whether any was bound
   static bool BindFixedScaleTwinShaderResources(ID3D11DeviceContext* native_device_context, GameDeviceDataFF7Remake& game_device_data, com_ptr<ID3D11ShaderResourceView> (&original_srvs)[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT])
   {
      native_device_context->PSGetShaderResources(0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, &original_srvs[0]);
      ID3D11ShaderResourceView* twin_srvs[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT];
      bool any_twin = false;
      for (UINT i = 0; i < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT; i++)
      {
         twin_srvs[i] = original_srvs[i].get();
         if (original_srvs[i].get() == nullptr)
         {
            continue;
         }
         com_ptr<ID3D11Resource> resource;
         original_srvs[i]->GetResource(&resource);
         if (game_device_data.valid_fixed_scale_twins.contains(resource.get()))
         {
            twin_srvs[i] = game_device_data.fixed_scale_twins[resource.get()].srv.get();
            any_twin = true;
         }
      }
      if (any_twin)
      {
         native_device_context->PSSetShaderResources(0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, twin_srvs);
      }
      return any_twin;
   }

   // Draws the current pass reading the twins, at output resolution. If "target_twin" is set, it's drawn into it (with scaled viewports), otherwise into the game's targets.
   static void DrawWithFixedScaleTwins(ID3D11DeviceContext* native_device_context, CommandListData& cmd_list_data, DeviceData& device_data, reshade::api::shader_stage stages, std::function<void()>* original_draw_dispatch_func, FixedScaleTwin* target_twin)
   {
      auto& game_device_data = GetGameDeviceData(device_data);

      com_ptr<ID3D11RenderTargetView> rtvs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
      com_ptr<ID3D11DepthStencilView> dsv;
      D3D11_VIEWPORT viewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
      UINT num_viewports = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
      D3D11_RECT scissors[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
      UINT num_scissors = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;

      com_ptr<ID3D11ShaderResourceView> original_srvs[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT];
      BindFixedScaleTwinShaderResources(native_device_context, game_device_data, original_srvs);

      if (target_twin)
      {
         native_device_context->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, &rtvs[0], &dsv);
         native_device_context->RSGetViewports(&num_viewports, viewports);
         native_device_context->RSGetScissorRects(&num_scissors, scissors);

         ID3D11RenderTargetView* const twin_rtv = target_twin->rtv.get();
         native_device_context->OMSetRenderTargets(1, &twin_rtv, nullptr);

         // Scale from the render resolution, not the (padded) texture size, so the render area maps to the whole twin
         const float scale_x = static_cast<float>(game_device_data.fixed_scale_output_size.x) / static_cast<float>(game_device_data.fixed_scale_render_size.x);
         const float scale_y = static_cast<float>(game_device_data.fixed_scale_output_size.y) / static_cast<float>(game_device_data.fixed_scale_render_size.y);
         D3D11_VIEWPORT twin_viewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
         for (UINT i = 0; i < num_viewports; i++)
         {
            twin_viewports[i] = viewports[i];
            twin_viewports[i].TopLeftX = std::round(viewports[i].TopLeftX * scale_x);
            twin_viewports[i].TopLeftY = std::round(viewports[i].TopLeftY * scale_y);
            twin_viewports[i].Width = std::round((viewports[i].TopLeftX + viewports[i].Width) * scale_x) - twin_viewports[i].TopLeftX;
            twin_viewports[i].Height = std::round((viewports[i].TopLeftY + viewports[i].Height) * scale_y) - twin_viewports[i].TopLeftY;
         }
         if (num_viewports > 0)
         {
            native_device_context->RSSetViewports(num_viewports, twin_viewports);
         }
         D3D11_RECT twin_scissors[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
         for (UINT i = 0; i < num_scissors; i++)
         {
            twin_scissors[i].left = std::lrintf(scissors[i].left * scale_x);
            twin_scissors[i].top = std::lrintf(scissors[i].top * scale_y);
            twin_scissors[i].right = std::lrintf(scissors[i].right * scale_x);
            twin_scissors[i].bottom = std::lrintf(scissors[i].bottom * scale_y);
         }
         if (num_scissors > 0)
         {
            native_device_context->RSSetScissorRects(num_scissors, twin_scissors);
         }
      }

      game_device_data.drawing_fixed_scale_twin = true;
      SetLumaConstantBuffers(native_device_context, cmd_list_data, device_data, stages, LumaConstantBufferType::LumaSettings);
      SetLumaConstantBuffers(native_device_context, cmd_list_data, device_data, stages, LumaConstantBufferType::LumaData);
      (*original_draw_dispatch_func)();
      game_device_data.drawing_fixed_scale_twin = false;

      // Restore the game's state
      ID3D11ShaderResourceView* const* original_srvs_const = (ID3D11ShaderResourceView**)std::addressof(original_srvs[0]);
      native_device_context->PSSetShaderResources(0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, original_srvs_const);
      if (target_twin)
      {
         ID3D11RenderTargetView* const* rtvs_const = (ID3D11RenderTargetView**)std::addressof(rtvs[0]);
         native_device_context->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtvs_const, dsv.get());
         if (num_viewports > 0)
         {
            native_device_context->RSSetViewports(num_viewports, viewports);
         }
         if (num_scissors > 0)
         {
            native_device_context->RSSetScissorRects(num_scissors, scissors);
         }
      }
      SetLumaConstantBuffers(native_device_context, cmd_list_data, device_data, stages, LumaConstantBufferType::LumaData);
   }

   // Called for the post processing draws and dispatches between SR and the output pass.
   // Returns true if the draw was done by us (into the twins, and also at render resolution if "fixed_scale_native_draws" is on).
   static bool DrawFixedScaleTwinPass(ID3D11Device* native_device, ID3D11DeviceContext* native_device_context, CommandListData& cmd_list_data, DeviceData& device_data, reshade::api::shader_stage stages, const ShaderHashesList<OneShaderPerPipeline>& original_shader_hashes, bool is_custom_pass, bool& updated_cbuffers, std::function<void()>* original_draw_dispatch_func)
   {
      auto& game_device_data = GetGameDeviceData(device_data);
      if (!game_device_data.fixed_scale_upscaling || device_data.has_drawn_main_post_processing)
      {
         return false;
      }

      // We can't draw compute passes at output resolution, they need the render resolution image
      if ((stages & reshade::api::shader_stage::compute) != 0)
      {
         if (!game_device_data.fixed_scale_native_draws)
         {
            com_ptr<ID3D11ShaderResourceView> srvs[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT];
            native_device_context->CSGetShaderResources(0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, &srvs[0]);
            if (ReadsFixedScaleTwin(game_device_data, (ID3D11ShaderResourceView**)std::addressof(srvs[0])))
            {
               RequireFixedScaleNativeDraws(game_device_data, original_shader_hashes);
            }
         }
         return false;
      }
      if ((stages & reshade::api::shader_stage::pixel) == 0)
      {
         return false;
      }

      com_ptr<ID3D11RenderTargetView> rtvs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
      com_ptr<ID3D11DepthStencilView> dsv;
      native_device_context->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, &rtvs[0], &dsv);
      com_ptr<ID3D11Resource> target;
      uint2 target_size = {0, 0};
      if (rtvs[0].get())
      {
         rtvs[0]->GetResource(&target);
         com_ptr<ID3D11Texture2D> target_texture;
         if (SUCCEEDED(target->QueryInterface(&target_texture)))
         {
            D3D11_TEXTURE2D_DESC target_desc;
            target_texture->GetDesc(&target_desc);
            target_size = {target_desc.Width, target_desc.Height};
         }
      }
      // The game (re)writes these, so their twins (if any) won't be current anymore, unless we redraw them below
      auto InvalidateTargetTwins = [&]()
      {
         for (const auto& rtv : rtvs)
         {
            if (rtv.get())
            {
               com_ptr<ID3D11Resource> resource;
               rtv->GetResource(&resource);
               game_device_data.valid_fixed_scale_twins.erase(resource.get());
            }
         }
      };

      bool reads_twin = false;
      {
         com_ptr<ID3D11ShaderResourceView> srvs[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT];
         native_device_context->PSGetShaderResources(0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, &srvs[0]);
         reads_twin = ReadsFixedScaleTwin(game_device_data, (ID3D11ShaderResourceView**)std::addressof(srvs[0]));
      }
      if (!reads_twin)
      {
         InvalidateTargetTwins();
         return false;
      }

      // Only our replaced shaders know how to draw at output resolution (through "FixedScaleUpscaling")
      const bool can_draw_scaled = is_custom_pass && original_draw_dispatch_func != nullptr && rtvs[1].get() == nullptr && dsv.get() == nullptr;
      const bool writes_scene_target = target.get() != nullptr && target_size.x == game_device_data.fixed_scale_render_target_size.x && target_size.y == game_device_data.fixed_scale_render_target_size.y;

      // Passes that read the upscaled image into their own sized targets (e.g. the bloom downscale)
      if (!writes_scene_target)
      {
         InvalidateTargetTwins();
         if (!can_draw_scaled)
         {
            RequireFixedScaleNativeDraws(game_device_data, original_shader_hashes);
            return false;
         }
         if (game_device_data.fixed_scale_native_draws)
         {
            return false; // They read the render resolution image, which is there
         }
         DrawWithFixedScaleTwins(native_device_context, cmd_list_data, device_data, stages, original_draw_dispatch_func, nullptr);
         updated_cbuffers = true;
         return true;
      }

      if (!can_draw_scaled)
      {
         BreakFixedScaleChain(game_device_data, original_shader_hashes, "reads the upscaled image but can't be drawn at output resolution");
         return false;
      }
      FixedScaleTwin* target_twin = GetOrCreateFixedScaleTwin(native_device, game_device_data, target.get(), false);
      if (target_twin == nullptr)
      {
         BreakFixedScaleChain(game_device_data, original_shader_hashes, "has a target we couldn't make a twin of");
         return false;
      }

      if (game_device_data.fixed_scale_native_draws)
      {
         SetLumaConstantBuffers(native_device_context, cmd_list_data, device_data, stages, LumaConstantBufferType::LumaSettings);
         SetLumaConstantBuffers(native_device_context, cmd_list_data, device_data, stages, LumaConstantBufferType::LumaData);
         (*original_draw_dispatch_func)();
      }
      DrawWithFixedScaleTwins(native_device_context, cmd_list_data, device_data, stages, original_draw_dispatch_func, target_twin);
      updated_cbuffers = true;

      game_device_data.valid_fixed_scale_twins.insert(target.get());
      return true;
   }

   DrawOrDispatchOverrideType OnDrawOrDispatch(ID3D11Device* native_device, ID3D11DeviceContext* native_device_context, CommandListData& cmd_list_data, DeviceData& device_data, reshade::api::shader_stage stages, const ShaderHashesList<OneShaderPerPipeline>& original_shader_hashes, bool is_custom_pass, bool& updated_cbuffers, std::function<void()>* original_draw_dispatch_func) override
   {
      if (DrawFixedScaleTwinPass(native_device, native_device_context, cmd_list_data, device_data, stages, original_shader_hashes, is_custom_pass, updated_cbuffers, original_draw_dispatch_func))
      {
         return DrawOrDispatchOverrideType::Replaced;
      }
      return DrawOrDispatchOverrideType::None;
   }

#if ENABLE_FAST_NOISE_TEXTURES
   static DrawOrDispatchOverrideType OnDraw_SSAO_Temporal(ID3D11Device* native_device, ID3D11DeviceContext* native_device_context, CommandListData& cmd_list_data, DeviceData& device_data, reshade::api::shader_stage stages, const ShaderHashesList<OneShaderPerPipeline>& original_shader_hashes, bool is_custom_pass, bool& updated_cbuffers, std::function<void()>* original_draw_dispatch_func)
   {
      if ((stages & reshade::api::shader_stage::pixel) != 0)
      {
         const auto fast_noise_it = device_data.managed_resources.shader_resource_views.find(kCoreFastNoiseTextureHash);
         if (fast_noise_it != device_data.managed_resources.shader_resource_views.end() && fast_noise_it->second)
         {
            // t6 is free in the original 04BFE575 signature; t0-t5 retain the game's bindings.
            native_device_context->PSSetShaderResources(6, 1, &fast_noise_it->second);
         }
      }

      return DrawOrDispatchOverrideType::None;
   }
#endif

   static DrawOrDispatchOverrideType OnDraw_OutputHDR(ID3D11Device* native_device, ID3D11DeviceContext* native_device_context, CommandListData& cmd_list_data, DeviceData& device_data, reshade::api::shader_stage stages, const ShaderHashesList<OneShaderPerPipeline>& original_shader_hashes, bool is_custom_pass, bool& updated_cbuffers, std::function<void()>* original_draw_dispatch_func)
   {
      return OnDraw_OutputPass(native_device, native_device_context, cmd_list_data, device_data, stages, original_shader_hashes, is_custom_pass, updated_cbuffers, original_draw_dispatch_func, true);
   }

   static DrawOrDispatchOverrideType OnDraw_OutputSDR(ID3D11Device* native_device, ID3D11DeviceContext* native_device_context, CommandListData& cmd_list_data, DeviceData& device_data, reshade::api::shader_stage stages, const ShaderHashesList<OneShaderPerPipeline>& original_shader_hashes, bool is_custom_pass, bool& updated_cbuffers, std::function<void()>* original_draw_dispatch_func)
   {
      return OnDraw_OutputPass(native_device, native_device_context, cmd_list_data, device_data, stages, original_shader_hashes, is_custom_pass, updated_cbuffers, original_draw_dispatch_func, false);
   }

   static DrawOrDispatchOverrideType OnDraw_OutputPass(ID3D11Device* native_device, ID3D11DeviceContext* native_device_context, CommandListData& cmd_list_data, DeviceData& device_data, reshade::api::shader_stage stages, const ShaderHashesList<OneShaderPerPipeline>& original_shader_hashes, bool is_custom_pass, bool& updated_cbuffers, std::function<void()>* original_draw_dispatch_func, bool is_hdr)
   {
      auto& game_device_data = GetGameDeviceData(device_data);

      if (device_data.has_drawn_main_post_processing || !game_device_data.has_drawn_taa)
      {
         return DrawOrDispatchOverrideType::None;
      }
      const bool is_tonemapping_sdr = !is_hdr;
      const bool is_tonemapping_hdr = is_hdr;
      DisplayModeType current_display_mode = cb_luma_global_settings.DisplayMode;
      bool display_mode_changed = (is_tonemapping_sdr && current_display_mode == DisplayModeType::HDR) || (is_tonemapping_hdr && current_display_mode == DisplayModeType::SDR);
      bool enable_hdr = is_tonemapping_hdr && current_display_mode != DisplayModeType::HDR;
      if (display_mode_changed)
      {
         if (enable_hdr)
         {
            Display::SetHDREnabled(game_window);
            bool dummy_bool;
            Display::IsHDRSupportedAndEnabled(game_window, dummy_bool, hdr_enabled_display, nullptr);
            if (!reshade::get_config_value(nullptr, NAME, "ScenePeakWhite", cb_luma_global_settings.ScenePeakWhite) || cb_luma_global_settings.ScenePeakWhite <= 0.f)
            {
               cb_luma_global_settings.ScenePeakWhite = device_data.default_user_peak_white;
            }
            if (!reshade::get_config_value(nullptr, NAME, "ScenePaperWhite", cb_luma_global_settings.ScenePaperWhite))
            {
               cb_luma_global_settings.ScenePaperWhite = default_paper_white;
            }
            if (!reshade::get_config_value(nullptr, NAME, "UIPaperWhite", cb_luma_global_settings.UIPaperWhite))
            {
               cb_luma_global_settings.UIPaperWhite = default_paper_white;
            }
         }
         else
         {
            cb_luma_global_settings.ScenePeakWhite = srgb_white_level;
            cb_luma_global_settings.ScenePaperWhite = srgb_white_level;
            cb_luma_global_settings.UIPaperWhite = srgb_white_level;
         }

         cb_luma_global_settings.DisplayMode = is_tonemapping_hdr ? DisplayModeType::HDR : DisplayModeType::SDR;
         device_data.cb_luma_global_settings_dirty = true;
      }
      // Read the upscaled twin of the scene instead of upscaling the render resolution one (this stays bound for the following output passes)
      if (game_device_data.fixed_scale_upscaling)
      {
         com_ptr<ID3D11ShaderResourceView> original_srvs[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT];
         if (BindFixedScaleTwinShaderResources(native_device_context, game_device_data, original_srvs))
         {
            game_device_data.drawing_fixed_scale_twin = true;
         }
         else
         {
            BreakFixedScaleChain(game_device_data, original_shader_hashes, "(output) doesn't read the upscaled image");
         }
      }
      game_device_data.has_drawn_upscaling = true;       // This pass will make upscaling happen if DLSS didn't already do it before
      device_data.has_drawn_main_post_processing = true; // Post processing is finished, nothing more to fix

      // ============================================================================
      // Output two-pass split: tonemap to intermediate, then composite UI
      // ============================================================================
#if ENABLE_FRAMEGEN
      if (is_tonemapping_hdr)
      {
         // Look up UI composition native PS for the current permutation
         uint32_t current_ps_hash = static_cast<uint32_t>(original_shader_hashes.pixel_shaders[0]);
         auto ui_key_it = output_hdr_ui_shader_keys.find(current_ps_hash);
         auto tm_key_it = output_hdr_tonemap_shader_keys.find(current_ps_hash);
         if (ui_key_it == output_hdr_ui_shader_keys.end() || tm_key_it == output_hdr_tonemap_shader_keys.end())
         {
            // Unknown permutation — fall through to single-pass
            return DrawOrDispatchOverrideType::None;
         }

         auto ui_ps_it = device_data.native_pixel_shaders.find(ui_key_it->second);
         auto tm_ps_it = device_data.native_pixel_shaders.find(tm_key_it->second);
         auto copy_vs_it = device_data.native_vertex_shaders.find(CompileTimeStringHash("Copy VS"));
         if (ui_ps_it == device_data.native_pixel_shaders.end() || !ui_ps_it->second.get()
             || tm_ps_it == device_data.native_pixel_shaders.end() || !tm_ps_it->second.get()
             || copy_vs_it == device_data.native_vertex_shaders.end() || !copy_vs_it->second.get())
         {
            // Native shaders not compiled yet — fall through to single-pass
            return DrawOrDispatchOverrideType::None;
         }

         // Get the game's current RTV to determine dimensions and to restore later
         com_ptr<ID3D11RenderTargetView> original_rtvs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
         com_ptr<ID3D11DepthStencilView> original_dsv;
         native_device_context->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, &original_rtvs[0], &original_dsv);

         if (!original_rtvs[0].get())
         {
            return DrawOrDispatchOverrideType::None;
         }

         // Get render target dimensions
         com_ptr<ID3D11Resource> rtv_resource;
         original_rtvs[0]->GetResource(&rtv_resource);
         com_ptr<ID3D11Texture2D> rtv_texture;
         rtv_resource->QueryInterface(&rtv_texture);
         D3D11_TEXTURE2D_DESC rtv_desc;
         rtv_texture->GetDesc(&rtv_desc);

         // Create/recreate intermediate hudless RT if dimensions changed
         if (!game_device_data.output_hdr_hudless_texture.get()
             || game_device_data.output_hdr_hudless_width != rtv_desc.Width
             || game_device_data.output_hdr_hudless_height != rtv_desc.Height)
         {
            game_device_data.CleanOutputHDRHudlessResources();

            D3D11_TEXTURE2D_DESC desc = {};
            desc.Width = rtv_desc.Width;
            desc.Height = rtv_desc.Height;
            desc.MipLevels = 1;
            desc.ArraySize = 1;
            desc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
            desc.SampleDesc.Count = 1;
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;

            HRESULT hr = native_device->CreateTexture2D(&desc, nullptr, &game_device_data.output_hdr_hudless_texture);
            ASSERT_ONCE(SUCCEEDED(hr));
            if (SUCCEEDED(hr))
            {
               hr = native_device->CreateRenderTargetView(game_device_data.output_hdr_hudless_texture.get(), nullptr, &game_device_data.output_hdr_hudless_rtv);
               ASSERT_ONCE(SUCCEEDED(hr));
               hr = native_device->CreateShaderResourceView(game_device_data.output_hdr_hudless_texture.get(), nullptr, &game_device_data.output_hdr_hudless_srv);
               ASSERT_ONCE(SUCCEEDED(hr));
            }

            if (FAILED(hr))
            {
               game_device_data.CleanOutputHDRHudlessResources();
               return DrawOrDispatchOverrideType::None;
            }

            game_device_data.output_hdr_hudless_width = rtv_desc.Width;
            game_device_data.output_hdr_hudless_height = rtv_desc.Height;
         }

         // Cache GPU state
         com_ptr<ID3D11VertexShader> prev_vs;
         com_ptr<ID3D11PixelShader> prev_ps;
         native_device_context->VSGetShader(&prev_vs, nullptr, nullptr);
         native_device_context->PSGetShader(&prev_ps, nullptr, nullptr);
         D3D11_PRIMITIVE_TOPOLOGY prev_topology;
         native_device_context->IAGetPrimitiveTopology(&prev_topology);

         // ===== Pass 1: Tonemap to intermediate (hudless) =====
         // Redirect output to intermediate RT and set the native tonemap PS (with TONEMAP_PASS=1).
         // All other state (VS, SRVs, CBs, samplers) stays as-is from the game.
         ID3D11RenderTargetView* const hudless_rtv = game_device_data.output_hdr_hudless_rtv.get();
         native_device_context->OMSetRenderTargets(1, &hudless_rtv, nullptr);
         native_device_context->PSSetShader(tm_ps_it->second.get(), nullptr, 0);

         // Ensure Luma CBs are bound for the PS stage
         SetLumaConstantBuffers(native_device_context, cmd_list_data, device_data, reshade::api::shader_stage::pixel, LumaConstantBufferType::LumaSettings);
         SetLumaConstantBuffers(native_device_context, cmd_list_data, device_data, reshade::api::shader_stage::pixel, LumaConstantBufferType::LumaData);

         // Issue the game's original draw call with our tonemap PS
         if (original_draw_dispatch_func)
         {
            (*original_draw_dispatch_func)();
         }
         else
         {
            native_device_context->DrawIndexed(3, 6, 0);
         }
         Trace::CustomDraw(native_device_context, "Output HDR Tonemap", 3);

         // ===== Pass 2: UI Composition to final RT =====
         // Restore original RTV
         ID3D11RenderTargetView* const* restored_rtvs = (ID3D11RenderTargetView**)std::addressof(original_rtvs[0]);
         native_device_context->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, restored_rtvs, original_dsv.get());

         // Rebind t1 (colorTex register) to the intermediate hudless SRV
         ID3D11ShaderResourceView* const hudless_srv = game_device_data.output_hdr_hudless_srv.get();
         native_device_context->PSSetShaderResources(1, 1, &hudless_srv);

         // Set UI composition PS and Copy VS
         native_device_context->PSSetShader(ui_ps_it->second.get(), nullptr, 0);
         native_device_context->VSSetShader(copy_vs_it->second.get(), nullptr, 0);
         native_device_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

         // Ensure Luma CBs are bound for the PS stage
         SetLumaConstantBuffers(native_device_context, cmd_list_data, device_data, reshade::api::shader_stage::pixel, LumaConstantBufferType::LumaSettings);
         SetLumaConstantBuffers(native_device_context, cmd_list_data, device_data, reshade::api::shader_stage::pixel, LumaConstantBufferType::LumaData);
         updated_cbuffers = true;

         // Draw fullscreen quad
         native_device_context->Draw(4, 0);
         Trace::CustomDraw(native_device_context, "Output HDR UI Composition", 4);

         // Restore VS, PS, topology
         native_device_context->VSSetShader(prev_vs.get(), nullptr, 0);
         native_device_context->PSSetShader(prev_ps.get(), nullptr, 0);
         native_device_context->IASetPrimitiveTopology(prev_topology);

         // Restore original SRV on t1 (so we don't leave the intermediate bound)
         com_ptr<ID3D11ShaderResourceView> original_color_srv;
         // We don't have the original t1 cached, but it's no longer needed since post-processing is done.
         // Just unbind to be safe.
         ID3D11ShaderResourceView* null_srv = nullptr;
         native_device_context->PSSetShaderResources(1, 1, &null_srv);

         return DrawOrDispatchOverrideType::Replaced;
      }

      // ============================================================================
      // Output SDR two-pass split: tonemap to intermediate, then composite UI
      // ============================================================================
      if (is_tonemapping_sdr)
      {
         // Look up UI composition and tonemap native PS for the current SDR permutation
         uint32_t current_ps_hash = static_cast<uint32_t>(original_shader_hashes.pixel_shaders[0]);
         auto ui_key_it = output_sdr_ui_shader_keys.find(current_ps_hash);
         auto tm_key_it = output_sdr_tonemap_shader_keys.find(current_ps_hash);
         if (ui_key_it == output_sdr_ui_shader_keys.end() || tm_key_it == output_sdr_tonemap_shader_keys.end())
         {
            // Unknown permutation — fall through to single-pass
            return DrawOrDispatchOverrideType::None;
         }

         auto ui_ps_it = device_data.native_pixel_shaders.find(ui_key_it->second);
         auto tm_ps_it = device_data.native_pixel_shaders.find(tm_key_it->second);
         auto copy_vs_it = device_data.native_vertex_shaders.find(CompileTimeStringHash("Copy VS"));
         if (ui_ps_it == device_data.native_pixel_shaders.end() || !ui_ps_it->second.get()
             || tm_ps_it == device_data.native_pixel_shaders.end() || !tm_ps_it->second.get()
             || copy_vs_it == device_data.native_vertex_shaders.end() || !copy_vs_it->second.get())
         {
            // Native shaders not compiled yet — fall through to single-pass
            return DrawOrDispatchOverrideType::None;
         }

         // Get the game's current RTV to determine dimensions and to restore later
         com_ptr<ID3D11RenderTargetView> original_rtvs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
         com_ptr<ID3D11DepthStencilView> original_dsv;
         native_device_context->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, &original_rtvs[0], &original_dsv);

         if (!original_rtvs[0].get())
         {
            return DrawOrDispatchOverrideType::None;
         }

         // Get render target dimensions
         com_ptr<ID3D11Resource> rtv_resource;
         original_rtvs[0]->GetResource(&rtv_resource);
         com_ptr<ID3D11Texture2D> rtv_texture;
         rtv_resource->QueryInterface(&rtv_texture);
         D3D11_TEXTURE2D_DESC rtv_desc;
         rtv_texture->GetDesc(&rtv_desc);

         // Create/recreate intermediate hudless RT if dimensions changed
         if (!game_device_data.output_sdr_hudless_texture.get()
             || game_device_data.output_sdr_hudless_width != rtv_desc.Width
             || game_device_data.output_sdr_hudless_height != rtv_desc.Height)
         {
            game_device_data.CleanOutputSDRHudlessResources();

            D3D11_TEXTURE2D_DESC desc = {};
            desc.Width = rtv_desc.Width;
            desc.Height = rtv_desc.Height;
            desc.MipLevels = 1;
            desc.ArraySize = 1;
            desc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
            desc.SampleDesc.Count = 1;
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;

            HRESULT hr = native_device->CreateTexture2D(&desc, nullptr, &game_device_data.output_sdr_hudless_texture);
            ASSERT_ONCE(SUCCEEDED(hr));
            if (SUCCEEDED(hr))
            {
               hr = native_device->CreateRenderTargetView(game_device_data.output_sdr_hudless_texture.get(), nullptr, &game_device_data.output_sdr_hudless_rtv);
               ASSERT_ONCE(SUCCEEDED(hr));
               hr = native_device->CreateShaderResourceView(game_device_data.output_sdr_hudless_texture.get(), nullptr, &game_device_data.output_sdr_hudless_srv);
               ASSERT_ONCE(SUCCEEDED(hr));
            }

            if (FAILED(hr))
            {
               game_device_data.CleanOutputSDRHudlessResources();
               return DrawOrDispatchOverrideType::None;
            }

            game_device_data.output_sdr_hudless_width = rtv_desc.Width;
            game_device_data.output_sdr_hudless_height = rtv_desc.Height;
         }

         // Cache GPU state
         com_ptr<ID3D11VertexShader> prev_vs;
         com_ptr<ID3D11PixelShader> prev_ps;
         native_device_context->VSGetShader(&prev_vs, nullptr, nullptr);
         native_device_context->PSGetShader(&prev_ps, nullptr, nullptr);
         D3D11_PRIMITIVE_TOPOLOGY prev_topology;
         native_device_context->IAGetPrimitiveTopology(&prev_topology);

         // ===== Pass 1: Tonemap to intermediate (hudless) =====
         ID3D11RenderTargetView* const hudless_rtv = game_device_data.output_sdr_hudless_rtv.get();
         native_device_context->OMSetRenderTargets(1, &hudless_rtv, nullptr);
         native_device_context->PSSetShader(tm_ps_it->second.get(), nullptr, 0);

         // Ensure Luma CBs are bound for the PS stage
         SetLumaConstantBuffers(native_device_context, cmd_list_data, device_data, reshade::api::shader_stage::pixel, LumaConstantBufferType::LumaSettings);
         SetLumaConstantBuffers(native_device_context, cmd_list_data, device_data, reshade::api::shader_stage::pixel, LumaConstantBufferType::LumaData);
         updated_cbuffers = true;

         // Issue the game's original draw call with our tonemap PS
         if (original_draw_dispatch_func)
         {
            (*original_draw_dispatch_func)();
         }
         else
         {
            native_device_context->DrawIndexed(3, 6, 0);
         }
         Trace::CustomDraw(native_device_context, "Output SDR Tonemap", 3);

         // ===== Pass 2: UI Composition to final RT =====
         ID3D11RenderTargetView* const* restored_rtvs = (ID3D11RenderTargetView**)std::addressof(original_rtvs[0]);
         native_device_context->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, restored_rtvs, original_dsv.get());

         // Rebind t1 (colorTex register) to the intermediate hudless SRV
         ID3D11ShaderResourceView* const hudless_srv = game_device_data.output_sdr_hudless_srv.get();
         native_device_context->PSSetShaderResources(1, 1, &hudless_srv);

         // Set UI composition PS and Copy VS
         native_device_context->PSSetShader(ui_ps_it->second.get(), nullptr, 0);
         native_device_context->VSSetShader(copy_vs_it->second.get(), nullptr, 0);
         native_device_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

         // Ensure Luma CBs are bound for the PS stage
         SetLumaConstantBuffers(native_device_context, cmd_list_data, device_data, reshade::api::shader_stage::pixel, LumaConstantBufferType::LumaSettings);
         SetLumaConstantBuffers(native_device_context, cmd_list_data, device_data, reshade::api::shader_stage::pixel, LumaConstantBufferType::LumaData);
         updated_cbuffers = true;

         // Draw fullscreen quad
         native_device_context->Draw(4, 0);
         Trace::CustomDraw(native_device_context, "Output SDR UI Composition", 4);

         // Restore VS, PS, topology
         native_device_context->VSSetShader(prev_vs.get(), nullptr, 0);
         native_device_context->PSSetShader(prev_ps.get(), nullptr, 0);
         native_device_context->IASetPrimitiveTopology(prev_topology);

         // Unbind intermediate from t1
         ID3D11ShaderResourceView* null_srv = nullptr;
         native_device_context->PSSetShaderResources(1, 1, &null_srv);

         return DrawOrDispatchOverrideType::Replaced;
      }
#endif // ENABLE_FRAMEGEN

      return DrawOrDispatchOverrideType::None;
   }

   static DrawOrDispatchOverrideType OnDraw_TAA(ID3D11Device* native_device, ID3D11DeviceContext* native_device_context, CommandListData& cmd_list_data, DeviceData& device_data, reshade::api::shader_stage stages, const ShaderHashesList<OneShaderPerPipeline>& original_shader_hashes, bool is_custom_pass, bool& updated_cbuffers, std::function<void()>* original_draw_dispatch_func)
   {
      auto& game_device_data = GetGameDeviceData(device_data);

      if (device_data.has_drawn_main_post_processing)
      {
         return DrawOrDispatchOverrideType::None;
      }

      game_device_data.has_drawn_taa = true;
      device_data.taa_detected = true;
#if ENABLE_SR
      if (device_data.sr_type != SR::Type::None && !device_data.sr_suppressed)
      {
         if (device_data.native_pixel_shaders[CompileTimeStringHash("Decode MVs")].get() == nullptr)
         {
            device_data.force_reset_sr = true;
            return DrawOrDispatchOverrideType::None;
         }
         // 1 depth
         // 2 current color source
         // 3 previous color source (previous frame)
         // 4 motion vectors
         com_ptr<ID3D11ShaderResourceView> ps_shader_resources[5];
         native_device_context->PSGetShaderResources(0, ARRAYSIZE(ps_shader_resources), &ps_shader_resources[0]);

         com_ptr<ID3D11RenderTargetView> render_target_views[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT]; // There should only be 1 or 2
         com_ptr<ID3D11DepthStencilView> depth_stencil_view;
         native_device_context->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, &render_target_views[0], &depth_stencil_view);
         const bool dlss_inputs_valid = ps_shader_resources[0].get() != nullptr && ps_shader_resources[1].get() != nullptr && ps_shader_resources[2].get() != nullptr && ps_shader_resources[4].get() != nullptr && render_target_views[0].get() != nullptr;
         ASSERT_ONCE(dlss_inputs_valid);

         if (dlss_inputs_valid)
         {
            auto* sr_instance_data = device_data.GetSRInstanceData();
            ASSERT_ONCE(sr_instance_data);

            com_ptr<ID3D11Resource> output_color_resource;
            render_target_views[0]->GetResource(&output_color_resource);
            com_ptr<ID3D11Texture2D> output_color;
            HRESULT hr = output_color_resource->QueryInterface(&output_color);
            ASSERT_ONCE(SUCCEEDED(hr));

            D3D11_TEXTURE2D_DESC taa_output_texture_desc;
            output_color->GetDesc(&taa_output_texture_desc);

            D3D11_VIEWPORT viewport;
            uint32_t num_viewports = 1;
            native_device_context->RSGetViewports(&num_viewports, &viewport);
            device_data.render_resolution = {viewport.Width, viewport.Height};

            // A fixed render scale ("r.ScreenPercentage"): the view covers the whole (render resolution sized, slightly padded) texture, which is smaller than the output.
            // DRS instead renders in the top left of output sized textures.
            constexpr uint32_t max_render_target_padding = 16;
            const float output_aspect_ratio = device_data.output_resolution.x / device_data.output_resolution.y;
            const bool fixed_scale = viewport.TopLeftX == 0.f && viewport.TopLeftY == 0.f && viewport.Width >= 1.f && viewport.Height >= 1.f
               && taa_output_texture_desc.Width >= viewport.Width && taa_output_texture_desc.Width - viewport.Width <= max_render_target_padding
               && taa_output_texture_desc.Height >= viewport.Height && taa_output_texture_desc.Height - viewport.Height <= max_render_target_padding
               && viewport.Width < device_data.output_resolution.x * 0.99f && viewport.Height < device_data.output_resolution.y * 0.99f
               && std::abs((viewport.Width / viewport.Height) / output_aspect_ratio - 1.f) < 0.01f;

            if (fixed_scale)
            {
               game_device_data.upscaled_render_resolution = device_data.output_resolution;
               game_device_data.resolution_scale = device_data.render_resolution.x / game_device_data.upscaled_render_resolution.x;
               game_device_data.viewport_rect = {0, 0, static_cast<uint32_t>(game_device_data.upscaled_render_resolution.x), static_cast<uint32_t>(game_device_data.upscaled_render_resolution.y)};
               game_device_data.fixed_scale_render_target_size = {taa_output_texture_desc.Width, taa_output_texture_desc.Height};
               game_device_data.fixed_scale_render_size = {static_cast<uint32_t>(std::lrintf(viewport.Width)), static_cast<uint32_t>(std::lrintf(viewport.Height))};
               game_device_data.fixed_scale_output_size = {static_cast<uint32_t>(std::lrintf(device_data.output_resolution.x)), static_cast<uint32_t>(std::lrintf(device_data.output_resolution.y))};
               device_data.sr_render_resolution_scale = game_device_data.resolution_scale;
            }
            else
            {
               game_device_data.upscaled_render_resolution = {(float)taa_output_texture_desc.Width, (float)taa_output_texture_desc.Height};
               game_device_data.resolution_scale = (float)device_data.render_resolution.x / (float)game_device_data.upscaled_render_resolution.x;
               game_device_data.viewport_rect = {static_cast<uint32_t>(viewport.TopLeftX), static_cast<uint32_t>(viewport.TopLeftY), static_cast<uint32_t>(game_device_data.upscaled_render_resolution.x), static_cast<uint32_t>(game_device_data.upscaled_render_resolution.y)};
            }
            // Once DRS has engaged once, we can't really detect if it's been turned off ever again, anyway it's always active by default in this game (unless one has mods to disable it, or fix a scaled render resolution)
            if (!fixed_scale)
            {
               if (!game_device_data.drs_active && game_device_data.resolution_scale == 1.0f)
               {
                  device_data.sr_render_resolution_scale = 1.0f;
               }
               else if (game_device_data.resolution_scale < 0.5f - FLT_EPSILON)
               {
                  device_data.sr_render_resolution_scale = game_device_data.resolution_scale;
                  game_device_data.drs_active = true;
               }
               else
               {
                  // This should pick quality or balanced mode, with a range from 100% to 50% resolution scale
                  device_data.sr_render_resolution_scale = 1.f / 1.5f;
                  game_device_data.drs_active = true;
               }
            }

            // The TAA input and output textures were guaranteed to be of the same size, so we pass in the output one as render res,
            // scaled by the DLSS render resolution scaling factor (which is a fixed multiplication to enabled dynamic res scaling in DLSS, it doesn't change every frame, as long as the res doesn't drop below 50%).
            double target_aspect_ratio = (double)game_device_data.upscaled_render_resolution.x / (double)game_device_data.upscaled_render_resolution.y;
            std::array<uint32_t, 2> dlss_input_resolution = FindClosestIntegerResolutionForAspectRatio((double)taa_output_texture_desc.Width * device_data.sr_render_resolution_scale, (double)taa_output_texture_desc.Height * device_data.sr_render_resolution_scale, target_aspect_ratio);
            if (fixed_scale)
            {
               dlss_input_resolution = {game_device_data.fixed_scale_render_size.x, game_device_data.fixed_scale_render_size.y};
            }

            if (dlss_input_resolution[0] > game_device_data.upscaled_render_resolution.x || dlss_input_resolution[1] > game_device_data.upscaled_render_resolution.y)
            {
               device_data.force_reset_sr = true;
               return DrawOrDispatchOverrideType::None;
            }

            SR::SettingsData settings_data;
            settings_data.output_width = game_device_data.upscaled_render_resolution.x;
            settings_data.output_height = game_device_data.upscaled_render_resolution.y;
            settings_data.render_width = dlss_input_resolution[0];
            settings_data.render_height = dlss_input_resolution[1];
            settings_data.dynamic_resolution = game_device_data.drs_active && !fixed_scale;
            settings_data.hdr = true; // Unreal Engine does DLSS before tonemapping, in HDR linear space
            settings_data.inverted_depth = true;
            settings_data.mvs_jittered = false;
            settings_data.auto_exposure = false;
            settings_data.mvs_x_scale = 1.0f;
            settings_data.mvs_y_scale = 1.0f;
            settings_data.render_preset = dlss_render_preset;
            sr_implementations[device_data.sr_type]->UpdateSettings(sr_instance_data, native_device_context, settings_data);

            bool skip_dlss = taa_output_texture_desc.Width < sr_instance_data->min_resolution || taa_output_texture_desc.Height < sr_instance_data->min_resolution;
            bool dlss_output_changed = false;

            constexpr bool dlss_use_native_uav = true;
            bool dlss_output_supports_uav = dlss_use_native_uav && (taa_output_texture_desc.BindFlags & D3D11_BIND_UNORDERED_ACCESS) != 0;
            if (fixed_scale)
            {
               // SR writes into the output resolution twin of the TAA output, the game's TAA still fills the original
               const auto prev_twin = game_device_data.fixed_scale_twins.find(output_color_resource.get());
               const ID3D11Texture2D* prev_twin_texture = prev_twin != game_device_data.fixed_scale_twins.end() ? prev_twin->second.texture.get() : nullptr;
               FixedScaleTwin* twin = GetOrCreateFixedScaleTwin(native_device, game_device_data, output_color_resource.get(), true);
               if (twin == nullptr)
               {
                  static bool logged_twin_failure = false;
                  if (!logged_twin_failure)
                  {
                     logged_twin_failure = true;
                     reshade::log::message(reshade::log::level::warning, "Fixed scale upscaling: couldn't create the output resolution TAA texture");
                  }
                  skip_dlss = true;
               }
               else
               {
                  dlss_output_changed = twin->texture.get() != prev_twin_texture;
                  device_data.sr_output_color = twin->texture;
               }
            }
            // Create a copy that supports Unordered Access if it wasn't already supported
            else if (!dlss_output_supports_uav)
            {
               D3D11_TEXTURE2D_DESC dlss_output_texture_desc = taa_output_texture_desc;
               dlss_output_texture_desc.Width = std::lrintf(game_device_data.upscaled_render_resolution.x);
               dlss_output_texture_desc.Height = std::lrintf(game_device_data.upscaled_render_resolution.y);
               dlss_output_texture_desc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;

               if (device_data.sr_output_color.get())
               {
                  D3D11_TEXTURE2D_DESC prev_dlss_output_texture_desc;
                  device_data.sr_output_color->GetDesc(&prev_dlss_output_texture_desc);
                  dlss_output_changed = prev_dlss_output_texture_desc.Width != dlss_output_texture_desc.Width || prev_dlss_output_texture_desc.Height != dlss_output_texture_desc.Height || prev_dlss_output_texture_desc.Format != dlss_output_texture_desc.Format;
               }
               if (!device_data.sr_output_color.get() || dlss_output_changed)
               {
                  device_data.sr_output_color = nullptr; // Make sure we discard the previous one
                  hr = native_device->CreateTexture2D(&dlss_output_texture_desc, nullptr, &device_data.sr_output_color);
                  ASSERT_ONCE(SUCCEEDED(hr));
               }
               // Texture creation failed, we can't proceed with DLSS
               if (!device_data.sr_output_color.get())
               {
                  skip_dlss = true;
               }
            }
            else
            {
               ASSERT_ONCE(device_data.sr_output_color == nullptr);
               device_data.sr_output_color = output_color;
            }

            if (!skip_dlss)
            {
               game_device_data.sr_source_color = nullptr;
               ps_shader_resources[2]->GetResource(&game_device_data.sr_source_color);
               game_device_data.depth_buffer = nullptr;
               ps_shader_resources[1]->GetResource(&game_device_data.depth_buffer);
               com_ptr<ID3D11Resource> object_velocity;
               ps_shader_resources[4]->GetResource(&object_velocity);

               // TODO: add exposure texture support (it's possibly calculated just earlier in the auto exposure steps, but they could be after DLSS too, depends on UE), either way auto exposure is ok
               DrawStateStack<DrawStateStackType::FullGraphics> draw_state_stack;
               DrawStateStack<DrawStateStackType::Compute> compute_state_stack;
               // We don't actually replace the shaders with the classic luma shader swapping feature, so we need to set the CBs manually
               draw_state_stack.Cache(native_device_context, device_data.uav_max_count);
               compute_state_stack.Cache(native_device_context, device_data.uav_max_count);
               // Decode the motion vector from pixel shader
               {
                  if (!AreResourcesEqual(object_velocity.get(), game_device_data.sr_motion_vectors.get(), false /*check_format*/))
                  {
                     com_ptr<ID3D11Texture2D> object_velocity_texture;
                     hr = object_velocity->QueryInterface(&object_velocity_texture);
                     ASSERT_ONCE(SUCCEEDED(hr));
                     D3D11_TEXTURE2D_DESC object_velocity_texture_desc;
                     object_velocity_texture->GetDesc(&object_velocity_texture_desc);
                     ASSERT_ONCE((object_velocity_texture_desc.BindFlags & D3D11_BIND_RENDER_TARGET) == D3D11_BIND_RENDER_TARGET);
#if 0 // Use the higher quality for MVs, the game's one were R16G16F. This has a ~1% cost on performance but helps with reducing shimmering on fine lines (stright lines looking segmented, like Bart's hair or Shark's teeth) when the camera is moving in a linear fashion.
                     object_velocity_texture_desc.Format = DXGI_FORMAT_R32G32_FLOAT;
#else // Note: for FF7, 16bit might be enough, to be tried and compared, but the extra precision won't hurt
                     object_velocity_texture_desc.Format = DXGI_FORMAT_R16G16_FLOAT;
#endif

                     game_device_data.sr_motion_vectors = nullptr; // Make sure we discard the previous one

                     // FIX: Add SHADER_RESOURCE bind flag so we can read MVs in Bloom pass
                     object_velocity_texture_desc.BindFlags |= D3D11_BIND_SHADER_RESOURCE;

                     hr = native_device->CreateTexture2D(&object_velocity_texture_desc, nullptr, &game_device_data.sr_motion_vectors);
                     ASSERT_ONCE(SUCCEEDED(hr));

                     game_device_data.sr_motion_vectors_rtv = nullptr; // Make sure we discard the previous one
                     game_device_data.sr_motion_vectors_srv = nullptr; // Reset SRV
                     if (SUCCEEDED(hr))
                     {
                        hr = native_device->CreateRenderTargetView(game_device_data.sr_motion_vectors.get(), nullptr, &game_device_data.sr_motion_vectors_rtv);
                        ASSERT_ONCE(SUCCEEDED(hr));

                        // Create SRV for MVs
                        hr = native_device->CreateShaderResourceView(game_device_data.sr_motion_vectors.get(), nullptr, &game_device_data.sr_motion_vectors_srv);
                        ASSERT_ONCE(SUCCEEDED(hr));
                     }
                  }

                  com_ptr<ID3D11VertexShader> prev_shader_vx;
                  com_ptr<ID3D11PixelShader> prev_shader_px;
                  native_device_context->VSGetShader(&prev_shader_vx, nullptr, nullptr);
                  native_device_context->PSGetShader(&prev_shader_px, nullptr, nullptr);
                  D3D11_PRIMITIVE_TOPOLOGY primitive_topology;
                  native_device_context->IAGetPrimitiveTopology(&primitive_topology);

                  // Set up for motion vector shader
                  ID3D11RenderTargetView* const dlss_motion_vectors_rtv_const = game_device_data.sr_motion_vectors_rtv.get();
                  native_device_context->OMSetRenderTargets(1, &dlss_motion_vectors_rtv_const, nullptr);

                  // We only need to swap the pixel/vertex shaders, depth and blend were already in the right state
                  native_device_context->VSSetShader(device_data.native_vertex_shaders[CompileTimeStringHash("Copy VS")].get(), nullptr, 0);
                  native_device_context->PSSetShader(device_data.native_pixel_shaders[CompileTimeStringHash("Decode MVs")].get(), nullptr, 0);

                  // We could probably keep the original vertex shader too, but whatever
                  native_device_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
                  // native_device_context->IASetInputLayout(nullptr); // Seemengly not needed
                  // native_device_context->RSSetState(nullptr); // Seemengly not needed

                  // Finally draw:
                  native_device_context->Draw(4, 0);
                  Trace::CustomDraw(native_device_context, "DLSS Decode Motion Vectors", 4);
                  // native_device_context->DrawIndexed(3, 6, 0); // Original call would have been this, but we swap the pixel and vertex shaders

                  // Restore the state
                  native_device_context->VSSetShader(prev_shader_vx.get(), nullptr, 0);
                  native_device_context->PSSetShader(prev_shader_px.get(), nullptr, 0);

                  native_device_context->IASetPrimitiveTopology(primitive_topology);
                  ID3D11RenderTargetView* const* rtvs_const = (ID3D11RenderTargetView**)std::addressof(render_target_views[0]);
                  native_device_context->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtvs_const, depth_stencil_view.get());
               }

               bool reset_dlss = device_data.force_reset_sr || dlss_output_changed || game_device_data.camera_cut;
               device_data.force_reset_sr = false;
               game_device_data.camera_cut = false;
               // Render resolution doesn't necessarily match with the source texture size, DRS draws on the top left of textures
               uint32_t render_width_dlss = 0;
               uint32_t render_height_dlss = 0;
               if (game_device_data.found_per_view_globals)
               {
                  render_width_dlss = std::lrintf(device_data.render_resolution.x);
                  render_height_dlss = std::lrintf(device_data.render_resolution.y);
               }
               else // Shouldn't happen!
               {
                  render_width_dlss = taa_output_texture_desc.Width;
                  render_height_dlss = taa_output_texture_desc.Height;
               }

               float dlss_pre_exposure = 0.f;

               SR::SuperResolutionImpl::DrawData draw_data;
               draw_data.source_color = game_device_data.sr_source_color.get();
               draw_data.output_color = device_data.sr_output_color.get();
               draw_data.motion_vectors = game_device_data.sr_motion_vectors.get();
               draw_data.depth_buffer = game_device_data.depth_buffer.get();
               draw_data.pre_exposure = sr_custom_pre_exposure;
               draw_data.jitter_x = projection_jitters.x * device_data.render_resolution.x * 0.5f;
               draw_data.jitter_y = projection_jitters.y * device_data.render_resolution.y * -0.5f;
               draw_data.user_sharpness = false;
               draw_data.far_plane = FLT_MAX;
               draw_data.near_plane = near_plane;
               draw_data.vert_fov = vert_fov;
               draw_data.reset = reset_dlss;
               draw_data.render_width = render_width_dlss;
               draw_data.render_height = render_height_dlss;

               bool dlss_succeeded = sr_implementations[device_data.sr_type]->Draw(sr_instance_data, native_device_context, draw_data);
               if (dlss_succeeded)
               {
                  device_data.has_drawn_sr = true;
               }
               game_device_data.camera_cut = false;
               game_device_data.sr_source_color = nullptr;
               game_device_data.depth_buffer = nullptr;
               draw_state_stack.Restore(native_device_context, device_data.uav_max_count);
               compute_state_stack.Restore(native_device_context, device_data.uav_max_count);

               if (device_data.has_drawn_sr && fixed_scale)
               {
                  device_data.sr_output_color = nullptr;
                  game_device_data.fixed_scale_upscaling = true;
                  game_device_data.valid_fixed_scale_twins.clear();
                  game_device_data.valid_fixed_scale_twins.insert(output_color_resource.get());
                  // The game's TAA only needs to draw if some pass reads its render resolution output
                  return game_device_data.fixed_scale_native_draws ? DrawOrDispatchOverrideType::None : DrawOrDispatchOverrideType::Replaced;
               }
               if (device_data.has_drawn_sr)
               {
                  // Upscaling happened later (during tonemapping) natively but we anticipate it with DLSS
                  game_device_data.has_drawn_upscaling = true;

                  // The TAA viewport bind happened before SR succeeded. Correct it once here; later rebinds are handled by the bind callbacks.
                  ResizeCurrentViewportAndScissorForEarlyUpscaling(native_device_context, device_data, game_device_data);

                  if (!dlss_output_supports_uav)
                  {
                     native_device_context->CopyResource(output_color.get(), device_data.sr_output_color.get()); // DX11 doesn't need barriers
                  }
                  else
                  {
                     device_data.sr_output_color = nullptr;
                  }

                  return DrawOrDispatchOverrideType::Replaced;
               }
               else
               {
                  // ASSERT_ONCE(false);
                  // cb_luma_global_settings.SRType = 0;
                  // device_data.cb_luma_global_settings_dirty = true;
                  // device_data.sr_suppressed = true;
                  device_data.force_reset_sr = true;
               }
            }
            if (dlss_output_supports_uav || fixed_scale)
            {
               device_data.sr_output_color = nullptr;
            }
         }
      }

#endif // ENABLE_SR
      return DrawOrDispatchOverrideType::None;
   }

   static DrawOrDispatchOverrideType OnDraw_DOFAndMotionBlur(ID3D11Device* native_device, ID3D11DeviceContext* native_device_context, CommandListData& cmd_list_data, DeviceData& device_data, reshade::api::shader_stage stages, const ShaderHashesList<OneShaderPerPipeline>& original_shader_hashes, bool is_custom_pass, bool& updated_cbuffers, std::function<void()>* original_draw_dispatch_func)
   {
      auto& game_device_data = GetGameDeviceData(device_data);

      if (device_data.has_drawn_main_post_processing || !game_device_data.has_drawn_taa)
      {
         return DrawOrDispatchOverrideType::None;
      }
#if ENABLE_SR
      ID3D11SamplerState* const linear_sampler = device_data.sampler_state_linear.get();
      native_device_context->PSSetSamplers(0, 1, &linear_sampler);
#endif
      if (DrawFixedScaleTwinPass(native_device, native_device_context, cmd_list_data, device_data, stages, original_shader_hashes, is_custom_pass, updated_cbuffers, original_draw_dispatch_func))
      {
         return DrawOrDispatchOverrideType::Replaced;
      }
      return DrawOrDispatchOverrideType::None;
   }

   void OnPresent(ID3D11Device* native_device, DeviceData& device_data) override
   {
      auto& game_device_data = GetGameDeviceData(device_data);

      // if (game_device_data.has_drawn_title)
      //{
      //    ASSERT_ONCE(game_device_data.found_per_view_globals);
      // }

      game_device_data.camera_cut = (!game_device_data.has_drawn_taa && !device_data.has_drawn_sr && !device_data.force_reset_sr) || game_device_data.camera_cut;
      device_data.has_drawn_main_post_processing = false;
      game_device_data.has_drawn_upscaling = false;
      if (!game_device_data.has_drawn_taa)
      {
         device_data.sr_suppressed = false;
      }
      game_device_data.has_drawn_taa = false;
      device_data.has_drawn_sr = false;
      game_device_data.fixed_scale_upscaling = false;
      game_device_data.drawing_fixed_scale_twin = false;
      game_device_data.valid_fixed_scale_twins.clear();
      game_device_data.found_per_view_globals = false;
      device_data.cb_luma_global_settings_dirty = true;
      static std::mt19937 random_generator(std::chrono::system_clock::now().time_since_epoch().count());
      static auto random_range = static_cast<float>((std::mt19937::max)() - (std::mt19937::min)());
      cb_luma_global_settings.GameSettings.custom_random = static_cast<float>(random_generator() + (std::mt19937::min)()) / random_range;
      cb_luma_global_settings.SRType = static_cast<uint32_t>(device_data.sr_type);
   }

   void CleanExtraSRResources(DeviceData& device_data) override
   {
      auto& game_device_data = GetGameDeviceData(device_data);
      game_device_data.sr_motion_vectors = nullptr;
      game_device_data.sr_motion_vectors_srv = nullptr;
      game_device_data.sr_motion_vectors_rtv = nullptr;
      game_device_data.sr_source_color = nullptr;
      game_device_data.depth_buffer = nullptr;
      game_device_data.sr_settings_data = nullptr;
      game_device_data.sr_draw_data = nullptr;
      game_device_data.fixed_scale_twins.clear();
      game_device_data.valid_fixed_scale_twins.clear();
   }

   void PrintImGuiAbout() override
   {
      ImGui::PushTextWrapPos(0.0f);
      ImGui::Text("Luma for \"Final Fantasy VII Remake\" is developed by Izueh and Pumbo and is open source and free.\n"
                  "It adds DLSS and improved HDR tonemapping.\n"
                  "Additional thanks to ShortFuse and Musa from the RenoDX team and their HDR mods for Remake and Rebirth which served as reference.\n",
         "If you enjoy it, consider donating to any of the contributors.", "");
      ImGui::PopTextWrapPos();

      ImGui::NewLine();

      const auto button_color = ImGui::GetStyleColorVec4(ImGuiCol_Button);
      const auto button_hovered_color = ImGui::GetStyleColorVec4(ImGuiCol_ButtonHovered);
      const auto button_active_color = ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive);

      ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(30, 136, 124, 255));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(17, 149, 134, 255));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, IM_COL32(57, 133, 111, 255));
      static const std::string donation_link_izueh = std::string("Buy Izueh a Coffee on ko-fi ") + std::string(ICON_FK_OK);
      if (ImGui::Button(donation_link_izueh.c_str()))
      {
         system("start https://ko-fi.com/izueh");
      }
      ImGui::PopStyleColor(3);
      ImGui::NewLine();

      ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(70, 134, 0, 255));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(70 + 9, 134 + 9, 0, 255));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, IM_COL32(70 + 18, 134 + 18, 0, 255));

      static const std::string donation_link_pumbo = std::string("Buy Pumbo a Coffee on buymeacoffee ") + std::string(ICON_FK_OK);
      if (ImGui::Button(donation_link_pumbo.c_str()))
      {
         system("start https://buymeacoffee.com/realfiloppi");
      }
      static const std::string donation_link_pumbo_2 = std::string("Buy Pumbo a Coffee on ko-fi ") + std::string(ICON_FK_OK);
      if (ImGui::Button(donation_link_pumbo_2.c_str()))
      {
         system("start https://ko-fi.com/realpumbo");
      }
      ImGui::PopStyleColor(3);

      ImGui::NewLine();
      // Restore the previous color, otherwise the state we set would persist even if we popped it
      ImGui::PushStyleColor(ImGuiCol_Button, button_color);
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, button_hovered_color);
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, button_active_color);
      static const std::string social_link = std::string("Join our \"HDR Den\" Discord ") + std::string(ICON_FK_SEARCH);
      if (ImGui::Button(social_link.c_str()))
      {
         // Unique link for Luma by Pumbo (to track the origin of people joining), do not share for other purposes
         static const std::string obfuscated_link = std::string("start https://discord.gg/J9fM") + std::string("3EVuEZ");
         system(obfuscated_link.c_str());
      }
      static const std::string contributing_link = std::string("Contribute on Github ") + std::string(ICON_FK_FILE_CODE);
      if (ImGui::Button(contributing_link.c_str()))
      {
         system("start https://github.com/Filoppi/Luma-Framework");
      }
      ImGui::PopStyleColor(3);

      ImGui::NewLine();
      ImGui::Text("Credits:"
                  "\n\nMain:"
                  "\nIzueh"
                  "\nPumbo"

                  "\n\nAcknowledgments:"
                  "\nShortFuse"
                  "\nMusa"

                  "\n\nThird Party:"
                  "\nReShade"
                  "\nImGui"
                  "\nRenoDX"
                  "\n3Dmigoto"
                  "\nOklab"
                  "\nDICE (HDR tonemapper)",
         "");
   }

   void UpdateLumaInstanceDataCB(CB::LumaInstanceDataPadded& data, CommandListData& cmd_list_data, DeviceData& device_data) override
   {
      auto& game_device_data = GetGameDeviceData(device_data);
      data.GameData.RenderResolution = {device_data.render_resolution.x, device_data.render_resolution.y, 1.0f / device_data.render_resolution.x, 1.0f / device_data.render_resolution.y};
      data.GameData.OutputResolution = {game_device_data.upscaled_render_resolution.x, game_device_data.upscaled_render_resolution.y, 1.0f / game_device_data.upscaled_render_resolution.x, 1.0f / game_device_data.upscaled_render_resolution.y};
      data.GameData.ResolutionScale = {game_device_data.resolution_scale, 1.0f / game_device_data.resolution_scale};
      data.GameData.DrewUpscaling = device_data.has_drawn_sr ? 1 : 0;
      data.GameData.FixedScaleUpscaling = 0;
      if (game_device_data.fixed_scale_upscaling)
      {
         // Only the draws into the twins (and the output pass reading them) are at output resolution, the native ones keep the game's sizes
         data.GameData.DrewUpscaling = game_device_data.drawing_fixed_scale_twin ? 1 : 0;
         data.GameData.FixedScaleUpscaling = game_device_data.drawing_fixed_scale_twin ? 1 : 0;
      }
      data.GameData.ViewportRect = game_device_data.viewport_rect;

      can_sharpen = device_data.output_resolution.x == game_device_data.upscaled_render_resolution.x && device_data.output_resolution.y == game_device_data.upscaled_render_resolution.y;
      cb_luma_global_settings.GameSettings.can_sharpen = can_sharpen ? 1.f : 0.f;
   }

   static void UpdateLODBias(reshade::api::device* device)
   {
      DeviceData& device_data = *device->get_private_data<DeviceData>();
      auto& game_device_data = GetGameDeviceData(device_data);

      if (!custom_texture_mip_lod_bias_offset)
      {
         std::shared_lock shared_lock_samplers(s_mutex_samplers);

         const auto prev_texture_mip_lod_bias_offset = device_data.texture_mip_lod_bias_offset;
         if (device_data.sr_type != SR::Type::None && !device_data.sr_suppressed && device_data.taa_detected)
         {
            device_data.texture_mip_lod_bias_offset = SR::GetMipLODBias(device_data.render_resolution.y, device_data.output_resolution.y); // This results in -1 at output res
         }
         else
         {
            // Reset to default (our mip offset is additive, so this is neutral)
            device_data.texture_mip_lod_bias_offset = 0.f;
         }
         const auto new_texture_mip_lod_bias_offset = device_data.texture_mip_lod_bias_offset;

         bool texture_mip_lod_bias_offset_changed = prev_texture_mip_lod_bias_offset != new_texture_mip_lod_bias_offset;
         // Re-create all samplers immediately here instead of doing it at the end of the frame.
         // This allows us to avoid possible (but very unlikely) hitches that could happen if we re-created a new sampler for a new resolution later on when samplers descriptors are set.
         // It also allows us to use the right samplers for this frame's resolution.
         if (texture_mip_lod_bias_offset_changed)
         {
            ID3D11Device* native_device = (ID3D11Device*)(device->get_native());
            for (auto& samplers_handle : device_data.custom_sampler_by_original_sampler)
            {
               if (samplers_handle.second.contains(new_texture_mip_lod_bias_offset))
                  continue; // Skip "resolutions" that already got their custom samplers created
               ID3D11SamplerState* native_sampler = reinterpret_cast<ID3D11SamplerState*>(samplers_handle.first);
               shared_lock_samplers.unlock(); // This is fine!
               {
                  D3D11_SAMPLER_DESC native_desc;
                  native_sampler->GetDesc(&native_desc);
                  com_ptr<ID3D11SamplerState> custom_sampler = CreateCustomSampler(device_data, native_device, native_desc);
                  const std::unique_lock unique_lock_samplers(s_mutex_samplers);
                  samplers_handle.second[new_texture_mip_lod_bias_offset] = custom_sampler;
               }
               shared_lock_samplers.lock();
            }
         }
      }
   }

   static void OnMapBufferRegion(reshade::api::device* device, reshade::api::resource resource, uint64_t offset, uint64_t size, reshade::api::map_access access, void** data)
   {
      SKIP_UNSUPPORTED_DEVICE_API(device->get_api());

      ID3D11Device* native_device = (ID3D11Device*)(device->get_native());
      ID3D11Buffer* buffer = reinterpret_cast<ID3D11Buffer*>(resource.handle);
      DeviceData& device_data = *device->get_private_data<DeviceData>();
      auto& game_device_data = GetGameDeviceData(device_data);

      // The frames until this draw have a defaulted 1920x1080 resolution (and slightly after too)
      if (game_device_data.found_per_view_globals)
      {
         return;
      }

      // No need to convert to native DX11 flags
      if (access == reshade::api::map_access::write_only || access == reshade::api::map_access::write_discard || access == reshade::api::map_access::read_write)
      {
         D3D11_BUFFER_DESC buffer_desc;
         buffer->GetDesc(&buffer_desc);

         if (buffer_desc.ByteWidth == CBPerViewGlobal_buffer_size)
         {
            device_data.cb_per_view_global_buffer = buffer;
            ASSERT_ONCE(!device_data.cb_per_view_global_buffer_map_data);
            device_data.cb_per_view_global_buffer_map_data = *data;
         }
      }
   }

   static void OnUnmapBufferRegion(reshade::api::device* device, reshade::api::resource resource)
   {
      SKIP_UNSUPPORTED_DEVICE_API(device->get_api());
      
      ID3D11Device* native_device = (ID3D11Device*)(device->get_native());
      ID3D11Buffer* buffer = reinterpret_cast<ID3D11Buffer*>(resource.handle);
      DeviceData& device_data = *device->get_private_data<DeviceData>();
      auto& game_device_data = GetGameDeviceData(device_data);
      if (game_device_data.found_per_view_globals)
      {
         return;
      }
      bool is_global_cbuffer = device_data.cb_per_view_global_buffer != nullptr && device_data.cb_per_view_global_buffer == buffer;
      ASSERT_ONCE(!device_data.cb_per_view_global_buffer_map_data || is_global_cbuffer);
      if (is_global_cbuffer && device_data.cb_per_view_global_buffer_map_data != nullptr)
      {
         float4(&float_data)[CBPerViewGlobal_buffer_size / sizeof(float4)] = *((float4(*)[CBPerViewGlobal_buffer_size / sizeof(float4)]) device_data.cb_per_view_global_buffer_map_data);

         // TAA cbuffer, called once (?) per frame, possibly at the beginning of the frame
         // 122 target texture res
         // 125 render res
         // 126 upscaled render res (doesn't necessarily match the display/swapchain resolution, there might be black bars)
         // 118 jitter
         bool is_valid_cbuffer = true && float_data[126].z == 1.0f / float_data[126].x && float_data[126].w == 1.0f / float_data[126].y && float_data[125].z == 1.0f / float_data[125].x && float_data[125].w == 1.0f / float_data[125].y;
         //&& float_data[122].x == float_data[125].x && float_data[122].y == float_data[125].y && float_data[122].z == float_data[125].z && float_data[122].w == float_data[125].w;

         // Make absolutely sure the jitters aren't both 0, which should never happen if they used proper jitter generation math, but we don't know,
         // though this happens in menus or when TAA is disabed (through mods)
         bool jitters_valid = std::abs(float_data[118].x) <= 1.f && std::abs(float_data[118].y) <= 1.f; // TODO: the jitters range is probably 0.5/render_res or so, hence we could restrict the check to that range to make it safer?
         jitters_valid &= (std::abs(float_data[118].x) > 0.f || std::abs(float_data[118].y) > 0.f);
         is_valid_cbuffer &= jitters_valid;

         if (is_valid_cbuffer)
         {
            size_t thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
            ASSERT_ONCE(!game_device_data.found_per_view_globals); // We found this twice? Shouldn't happen, we should probably reject one of the two
            bool has_jitters = float_data[118].x != 0.f || float_data[118].y != 0.f;
            if (has_jitters)
            {
               game_device_data.jitterless_frames_count = 0;
               game_device_data.is_in_menu = false;
            }
            // Give it a two frames tolerance just to make 100% sure that the jitters random generation didn't actually pick 0 for both in one frame (it might be possible depending on the random pattern generation they used, but probably impossible for two frames, see "Halton")
            else
            {
               // Note: for now we don't disable DLSS even if jitters are off
               game_device_data.jitterless_frames_count++;
               if (game_device_data.jitterless_frames_count >= 2)
               {
                  if (!game_device_data.is_in_menu)
                  {
                     device_data.force_reset_sr = true; // TODO: make sure this doesn't happen when pausing the game and the scene in the background remains the same, it'd mean we get a couple blurry frames when we go back to the game

                     game_device_data.is_in_menu = true;
                  }
               }
            }

            game_device_data.found_per_view_globals = true;
            // Extract jitter from constant buffer 1
            projection_jitters.x = float_data[118].x;
            projection_jitters.y = float_data[118].y;
            
            // Update render resolution from cb1[122] before TAA runs.
            // cb1[122].xy = actual render size (the rendered region within the texture)
            if (float_data[122].x > 0 && float_data[122].y > 0)
            {
               device_data.render_resolution = {float_data[122].x, float_data[122].y};
            }

            Math::Matrix44F projection_matrix;
            std::memcpy(&projection_matrix, &float_data[24], sizeof(Math::Matrix44F));
            near_plane = ComputeNearPlane(projection_matrix) / 100.0f; // Game uses cm
            vert_fov = ComputeFovY(projection_matrix);

            // check valid near plane and fov
            if (near_plane < 0.01f || near_plane > 100.f)
               near_plane = 0.1f;
            if (vert_fov < (10.f * (M_PI / 180.f)) || vert_fov > (170.f * (M_PI / 180.f)))
               vert_fov = 60.f * (M_PI / 180.f);

            game_device_data.camera_cut = float_data[140].y != 0.f;
         }
        
         device_data.cb_per_view_global_buffer_map_data = nullptr;
         device_data.cb_per_view_global_buffer = nullptr;
         UpdateLODBias(device);
      }
   }

   static bool OnUpdateBufferRegion(reshade::api::device* device, const void* data, reshade::api::resource resource, uint64_t offset, uint64_t size)
   {
      ID3D11Device* native_device = (ID3D11Device*)(device->get_native());
      DeviceData& device_data = *device->get_private_data<DeviceData>();
      auto& game_device_data = GetGameDeviceData(device_data);

      if (device_data.has_drawn_sr && size == 512)
      {
         // It's not very nice to const cast, but we know for a fact this is dynamic memory, so it's probably fine to edit it (ReShade doesn't offer an interface for replacing it easily, and doesn't pass in the command list)
         float4* mutable_float_data = reinterpret_cast<float4*>(const_cast<void*>(data));
         const float4* float_data = reinterpret_cast<const float4*>(data);
         if (float_data[20].z == device_data.render_resolution.x && float_data[20].w == device_data.render_resolution.y && float_data[21].x == game_device_data.upscaled_render_resolution.x / 2.0f && float_data[21].y == game_device_data.upscaled_render_resolution.y / 2.0f)
         {
            mutable_float_data[20].z = game_device_data.upscaled_render_resolution.x;
            mutable_float_data[20].w = game_device_data.upscaled_render_resolution.y;
         }
      }
      else if (device_data.has_drawn_sr && size == 1024)
      {
         float4* mutable_float_data = reinterpret_cast<float4*>(const_cast<void*>(data));
         const float4* float_data = reinterpret_cast<const float4*>(data);
         // float_data[30].zw and [31].xy have render_res
         if (float_data[30].z == device_data.render_resolution.x && float_data[30].w == device_data.render_resolution.y && float_data[31].x == device_data.render_resolution.x && float_data[31].y == device_data.render_resolution.y)
         {
            mutable_float_data[30].z = game_device_data.upscaled_render_resolution.x;
            mutable_float_data[30].w = game_device_data.upscaled_render_resolution.y;
            mutable_float_data[31].x = game_device_data.upscaled_render_resolution.x;
            mutable_float_data[31].y = game_device_data.upscaled_render_resolution.y;
         }
      }
      return false;
   }

#if DEVELOPMENT
   void DrawImGuiDevSettings(DeviceData& device_data) override
   {
#if ENABLE_SR
      ImGui::NewLine();
      // ImGui::SliderFloat("SR Custom Exposure", &sr_custom_exposure, 0.0, 10.0);
      ImGui::SliderFloat("SR Custom Pre-Exposure", &sr_custom_pre_exposure, 0.0, 10.0);
#endif // ENABLE_SR
   }
#endif // DEVELOPMENT

   void LoadConfigs() override
   {
      Luma::Settings::LoadSettings();
   }

   void DrawImGuiSettings(DeviceData& device_data) override
   {
      Luma::Settings::DrawSettings();
   }


};
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
   if (ul_reason_for_call == DLL_PROCESS_ATTACH)
   {
      default_paper_white = 250.f;
      Globals::SetGlobals(PROJECT_NAME, "Final Fantasy VII Remake Luma mod");
      Globals::VERSION = 2;

      shader_hashes_TAA.pixel_shaders.emplace(std::stoul("4729683B", nullptr, 16));
      shader_hashes_Title.pixel_shaders.emplace(std::stoul("5FEE74F9", nullptr, 16));
      shader_hashes_DOF.pixel_shaders.emplace(std::stoul("B400FAF6", nullptr, 16));
      shader_hashes_Motion_Blur.pixel_shaders.emplace(std::stoul("B0F56393", nullptr, 16));
      shader_hashes_Downsample_Bloom.pixel_shaders.emplace(std::stoul("2174B927", nullptr, 16));
      shader_hashes_Bloom.pixel_shaders.emplace(std::stoul("4D6F937E", nullptr, 16));
      shader_hashes_MenuSlowdown.pixel_shaders.emplace(std::stoul("968B821F", nullptr, 16));
      shader_hashes_Tonemap.pixel_shaders.emplace(std::stoul("F68D39B5", nullptr, 16));
      shader_hashes_Velocity_Flatten.compute_shaders.emplace(std::stoul("4EB2EA5B", nullptr, 16));
      shader_hashes_Velocity_Gather.compute_shaders.emplace(std::stoul("FEE03685", nullptr, 16));
      std::vector<ShaderDefineData> game_shader_defines_data = {
         {"SSAO_ALGORITHM", '1', true, false, "AO algorithm\n0 - Original game SSAO\n1 - VBAO", 1}
      };
      shader_defines_data.append_range(game_shader_defines_data);

#if DEVELOPMENT
      // These make things messy in this game, given it renders at lower resolutions and then upscales and adds black bars beyond 16:9
      Trace::debug_draw_options &= ~(uint32_t)Trace::DebugDrawTextureOptionsMask::Fullscreen;

      forced_shader_names.emplace(std::stoul("4729683B", nullptr, 16), "TAA");
      forced_shader_names.emplace(std::stoul("B400FAF6", nullptr, 16), "DoF");
      forced_shader_names.emplace(std::stoul("B0F56393", nullptr, 16), "Motion Blur");
      forced_shader_names.emplace(std::stoul("2174B927", nullptr, 16), "Downsample Bloom"); // This is the first bloom downsample pass to run, it does a maximum of 50% downscaling, but if the resolution scale was e.g. 75% of the output one, it will convert from 75% to 50%, so the bloom chain isn't really affected by the render res
      forced_shader_names.emplace(std::stoul("A77F0B56", nullptr, 16), "Downsample Bloom"); // The result of this (4x3 texture or something) is used for bloom, maybe also as a local exposure map or something
      forced_shader_names.emplace(std::stoul("D9E87012", nullptr, 16), "Upscale Bloom");
      forced_shader_names.emplace(std::stoul("46727E9A", nullptr, 16), "Upscale Bloom"); // There's multiple versions of this (maybe one for DRS?)
      forced_shader_names.emplace(std::stoul("CCD7FA05", nullptr, 16), "Blur Bloom");
      forced_shader_names.emplace(std::stoul("69467442", nullptr, 16), "Blur Bloom");
      forced_shader_names.emplace(std::stoul("4D6F937E", nullptr, 16), "Apply Bloom");
      forced_shader_names.emplace(std::stoul("968B821F", nullptr, 16), "Slowdown Menu");
      forced_shader_names.emplace(std::stoul("F68D39B5", nullptr, 16), "Upscale and Tonemap");
      forced_shader_names.emplace(std::stoul("4EB2EA5B", nullptr, 16), "Velocity Flatten");
      forced_shader_names.emplace(std::stoul("FEE03685", nullptr, 16), "Velocity Gather");
      forced_shader_names.emplace(std::stoul("1D610CBA", nullptr, 16), "Ambient Occlusion");
      forced_shader_names.emplace(std::stoul("04BFE575", nullptr, 16), "Ambient Occlusion Temporal (SSAO algorithm)");
      forced_shader_names.emplace(std::stoul("8BD60486", nullptr, 16), "Ambient Occlusion Denoise 1");
      forced_shader_names.emplace(std::stoul("E6A8D4FB", nullptr, 16), "Ambient Occlusion Denoise 2");
#endif

#if !DEVELOPMENT
      old_shader_file_names.emplace("Output_HDR_0xA8EB118F_0x922A71D1_0x3A4D858E_0xD950DA01.ps_5_0.hlsl");
      old_shader_file_names.emplace("Output_SDR_0x506D5998_0xF68D39B5_0xBBB9CE42_0x51E2B894.ps_5_0.hlsl");
      old_shader_file_names.emplace("Velocity_Flatten_0x4EB2EA5B.cs_5_0.hlsl");
      old_shader_file_names.emplace("Velocity_Gather_0xFEE03685.cs_5_0.hlsl");
#endif

#if !DEVELOPMENT
      swapchain_format_upgrade_type = TextureFormatUpgradesType::AllowedEnabled;
      swapchain_upgrade_type = SwapchainUpgradeType::HDR10;
#else
      swapchain_format_upgrade_type = TextureFormatUpgradesType::AllowedEnabled;
      swapchain_upgrade_type = SwapchainUpgradeType::scRGB;
#endif

      texture_format_upgrades_type = TextureFormatUpgradesType::AllowedEnabled;
      // Texture upgrades (8 bit unorm and 11 bit float etc to 16 bit float)
      texture_upgrade_formats = {
#if 0 // Probably not needed
				reshade::api::format::r8g8b8a8_unorm,
				reshade::api::format::r8g8b8a8_unorm_srgb,
				reshade::api::format::r8g8b8a8_typeless,
				reshade::api::format::r8g8b8x8_unorm,
				reshade::api::format::r8g8b8x8_unorm_srgb,
				reshade::api::format::b8g8r8a8_unorm,
				reshade::api::format::b8g8r8a8_unorm_srgb,
				//reshade::api::format::b8g8r8a8_typeless, // currently causes validation issues due to some odd 1x1x1 textures (that's just an assert Luma put for mods devs to double check though). TODO: Figure out what these textures do.
				reshade::api::format::b8g8r8x8_unorm,
				reshade::api::format::b8g8r8x8_unorm_srgb,
				reshade::api::format::b8g8r8x8_typeless,
#endif

         reshade::api::format::r10g10b10a2_unorm,
         reshade::api::format::r10g10b10a2_typeless,

         reshade::api::format::r11g11b10_float,
      };
      // Upgrade all 16:9 render targets too, because the game defaults to that aspect ratio internally unless mods are applied
      texture_format_upgrades_2d_size_filters |= (uint32_t)TextureFormatUpgrades2DSizeFilters::CustomAspectRatio;
      texture_format_upgrades_2d_custom_aspect_ratios = {16.f / 9.f};
      // LUT is 3D 32x
      texture_format_upgrades_lut_size = 32;
      texture_format_upgrades_lut_dimensions = LUTDimensions::_3D;

      enable_samplers_upgrade = true;

      redirected_shader_hashes["Output_HDR"] = {"A8EB118F", "922A71D1", "3A4D858E", "D950DA01", "5CD12E67", "3B489929", "8D04181D", "6846FF90"};
      redirected_shader_hashes["Output_SDR"] = {"506D5998", "F68D39B5", "BBB9CE42", "51E2B894", "803889E8", "D96EF76D", "5C2D3A71", "66162229"};
      for (const auto& output_hdr_shader_hash : redirected_shader_hashes["Output_HDR"])
      {
         shader_hashes_Output_HDR.pixel_shaders.emplace(std::stoul(output_hdr_shader_hash, nullptr, 16));
      }

      // Build mapping from game PS hash to native UI composition shader key
#if ENABLE_FRAMEGEN
      output_hdr_ui_shader_keys[0x922A71D1] = CompileTimeStringHash("FF7R Output HDR UI 922A71D1");
      output_hdr_ui_shader_keys[0xA8EB118F] = CompileTimeStringHash("FF7R Output HDR UI A8EB118F");
      output_hdr_ui_shader_keys[0x3A4D858E] = CompileTimeStringHash("FF7R Output HDR UI 3A4D858E");
      output_hdr_ui_shader_keys[0xD950DA01] = CompileTimeStringHash("FF7R Output HDR UI D950DA01");
      output_hdr_ui_shader_keys[0x5CD12E67] = CompileTimeStringHash("FF7R Output HDR UI 5CD12E67");
      output_hdr_ui_shader_keys[0x3B489929] = CompileTimeStringHash("FF7R Output HDR UI 3B489929");
      output_hdr_ui_shader_keys[0x8D04181D] = CompileTimeStringHash("FF7R Output HDR UI 8D04181D");
      output_hdr_ui_shader_keys[0x6846FF90] = CompileTimeStringHash("FF7R Output HDR UI 6846FF90");

      // Build mapping from game PS hash to native tonemap shader key
      output_hdr_tonemap_shader_keys[0x922A71D1] = CompileTimeStringHash("FF7R Output HDR TM 922A71D1");
      output_hdr_tonemap_shader_keys[0xA8EB118F] = CompileTimeStringHash("FF7R Output HDR TM A8EB118F");
      output_hdr_tonemap_shader_keys[0x3A4D858E] = CompileTimeStringHash("FF7R Output HDR TM 3A4D858E");
      output_hdr_tonemap_shader_keys[0xD950DA01] = CompileTimeStringHash("FF7R Output HDR TM D950DA01");
      output_hdr_tonemap_shader_keys[0x5CD12E67] = CompileTimeStringHash("FF7R Output HDR TM 5CD12E67");
      output_hdr_tonemap_shader_keys[0x3B489929] = CompileTimeStringHash("FF7R Output HDR TM 3B489929");
      output_hdr_tonemap_shader_keys[0x8D04181D] = CompileTimeStringHash("FF7R Output HDR TM 8D04181D");
      output_hdr_tonemap_shader_keys[0x6846FF90] = CompileTimeStringHash("FF7R Output HDR TM 6846FF90");

      // Build mapping from game PS hash to native SDR UI composition shader key
      output_sdr_ui_shader_keys[0x506D5998] = CompileTimeStringHash("FF7R Output SDR UI 506D5998");
      output_sdr_ui_shader_keys[0xF68D39B5] = CompileTimeStringHash("FF7R Output SDR UI F68D39B5");
      output_sdr_ui_shader_keys[0xBBB9CE42] = CompileTimeStringHash("FF7R Output SDR UI BBB9CE42");
      output_sdr_ui_shader_keys[0x51E2B894] = CompileTimeStringHash("FF7R Output SDR UI 51E2B894");
      output_sdr_ui_shader_keys[0x803889E8] = CompileTimeStringHash("FF7R Output SDR UI 803889E8");
      output_sdr_ui_shader_keys[0xD96EF76D] = CompileTimeStringHash("FF7R Output SDR UI D96EF76D");
      output_sdr_ui_shader_keys[0x5C2D3A71] = CompileTimeStringHash("FF7R Output SDR UI 5C2D3A71");
      output_sdr_ui_shader_keys[0x66162229] = CompileTimeStringHash("FF7R Output SDR UI 66162229");

      // Build mapping from game PS hash to native SDR tonemap shader key
      output_sdr_tonemap_shader_keys[0x506D5998] = CompileTimeStringHash("FF7R Output SDR TM 506D5998");
      output_sdr_tonemap_shader_keys[0xF68D39B5] = CompileTimeStringHash("FF7R Output SDR TM F68D39B5");
      output_sdr_tonemap_shader_keys[0xBBB9CE42] = CompileTimeStringHash("FF7R Output SDR TM BBB9CE42");
      output_sdr_tonemap_shader_keys[0x51E2B894] = CompileTimeStringHash("FF7R Output SDR TM 51E2B894");
      output_sdr_tonemap_shader_keys[0x803889E8] = CompileTimeStringHash("FF7R Output SDR TM 803889E8");
      output_sdr_tonemap_shader_keys[0xD96EF76D] = CompileTimeStringHash("FF7R Output SDR TM D96EF76D");
      output_sdr_tonemap_shader_keys[0x5C2D3A71] = CompileTimeStringHash("FF7R Output SDR TM 5C2D3A71");
      output_sdr_tonemap_shader_keys[0x66162229] = CompileTimeStringHash("FF7R Output SDR TM 66162229");
#endif // ENABLE_FRAMEGEN
      for (const auto& output_sdr_shader_hash : redirected_shader_hashes["Output_SDR"])
      {
         shader_hashes_Output_SDR.pixel_shaders.emplace(std::stoul(output_sdr_shader_hash, nullptr, 16));
      }
      // shader_hashes_Output_HDR.pixel_shaders.emplace(redirected_shader_hashes["Output_HDR"]);
      // shader_hashes_Output_SDR.pixel_shaders.emplace(redirected_shader_hashes["Output_SDR"]);

      Luma::Settings::Initialize(&settings);

      game = new FF7Remake();
   }
   else if (ul_reason_for_call == DLL_PROCESS_DETACH)
   {
      reshade::unregister_event<reshade::addon_event::map_buffer_region>(FF7Remake::OnMapBufferRegion);
      reshade::unregister_event<reshade::addon_event::unmap_buffer_region>(FF7Remake::OnUnmapBufferRegion);
      reshade::unregister_event<reshade::addon_event::bind_viewports>(FF7Remake::OnBindViewports);
      reshade::unregister_event<reshade::addon_event::bind_scissor_rects>(FF7Remake::OnBindScissorRects);
      // reshade::unregister_event<reshade::addon_event::update_buffer_region>(FF7Remake::OnUpdateBufferRegion);
   }

   CoreMain(hModule, ul_reason_for_call, lpReserved);

   return TRUE;
}
