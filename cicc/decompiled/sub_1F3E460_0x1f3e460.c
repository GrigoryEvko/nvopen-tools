// Function: sub_1F3E460
// Address: 0x1f3e460
//
__int64 __fastcall sub_1F3E460(_QWORD *a1, _DWORD *a2)
{
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // eax
  int v7; // edx
  __int64 result; // rax
  bool v9; // al
  bool v10; // al
  unsigned int v11; // [rsp+4h] [rbp-1Ch] BYREF
  unsigned int v12; // [rsp+8h] [rbp-18h] BYREF
  _DWORD v13[5]; // [rsp+Ch] [rbp-14h] BYREF

  v3 = (unsigned __int64)(a1 + 9957);
  *(_QWORD *)(v3 - 5560) = "__ashlhi3";
  *(_QWORD *)(v3 - 5552) = "__ashlsi3";
  *(_QWORD *)(v3 - 5544) = "__ashldi3";
  *(_QWORD *)(v3 - 5536) = "__ashlti3";
  *(_QWORD *)(v3 - 5528) = "__lshrhi3";
  *(_QWORD *)(v3 - 5520) = "__lshrsi3";
  *(_QWORD *)(v3 - 5512) = "__lshrdi3";
  *(_QWORD *)(v3 - 5504) = "__lshrti3";
  *(_QWORD *)(v3 - 5496) = "__ashrhi3";
  *(_QWORD *)(v3 - 5488) = "__ashrsi3";
  *(_QWORD *)(v3 - 5480) = "__ashrdi3";
  *(_QWORD *)(v3 - 5472) = "__ashrti3";
  *(_QWORD *)(v3 - 5464) = "__mulqi3";
  *(_QWORD *)(v3 - 5456) = "__mulhi3";
  *(_QWORD *)(v3 - 5448) = "__mulsi3";
  *(_QWORD *)(v3 - 5440) = "__muldi3";
  *(_QWORD *)(v3 - 5432) = "__multi3";
  *(_QWORD *)(v3 - 5424) = "__mulosi4";
  *(_QWORD *)(v3 - 5416) = "__mulodi4";
  *(_QWORD *)(v3 - 5408) = "__muloti4";
  *(_QWORD *)(v3 - 5400) = "__divqi3";
  *(_QWORD *)(v3 - 5392) = "__divhi3";
  *(_QWORD *)(v3 - 5384) = "__divsi3";
  *(_QWORD *)(v3 - 5376) = "__divdi3";
  *(_QWORD *)(v3 - 5368) = "__divti3";
  *(_QWORD *)(v3 - 5360) = "__udivqi3";
  *(_QWORD *)(v3 - 5352) = "__udivhi3";
  *(_QWORD *)(v3 - 5344) = "__udivsi3";
  *(_QWORD *)(v3 - 5336) = "__udivdi3";
  *(_QWORD *)(v3 - 5328) = "__udivti3";
  *(_QWORD *)(v3 - 5320) = "__modqi3";
  *(_QWORD *)(v3 - 5312) = "__modhi3";
  *(_QWORD *)(v3 - 5304) = "__modsi3";
  *(_QWORD *)(v3 - 5296) = "__moddi3";
  *(_QWORD *)(v3 - 5288) = "__modti3";
  *(_QWORD *)(v3 - 5280) = "__umodqi3";
  *(_QWORD *)(v3 - 5272) = "__umodhi3";
  *(_QWORD *)(v3 - 5264) = "__umodsi3";
  *(_QWORD *)(v3 - 5256) = "__umoddi3";
  *(_QWORD *)(v3 - 5248) = "__umodti3";
  *(_QWORD *)(v3 - 5160) = "__negsi2";
  *(_QWORD *)(v3 - 5152) = "__negdi2";
  *(_QWORD *)(v3 - 5144) = "__addsf3";
  *(_QWORD *)(v3 - 5136) = "__adddf3";
  *(_QWORD *)(v3 - 5128) = "__addxf3";
  *(_QWORD *)(v3 - 5120) = "__addtf3";
  *(_QWORD *)(v3 - 5112) = "__gcc_qadd";
  *(_QWORD *)(v3 - 5104) = "__subsf3";
  *(_QWORD *)(v3 - 5096) = "__subdf3";
  *(_QWORD *)(v3 - 5088) = "__subxf3";
  *(_QWORD *)(v3 - 5080) = "__subtf3";
  *(_QWORD *)(v3 - 5240) = 0;
  *(_QWORD *)(v3 - 5232) = 0;
  *(_QWORD *)(v3 - 5224) = 0;
  *(_QWORD *)(v3 - 5216) = 0;
  *(_QWORD *)(v3 - 5208) = 0;
  *(_QWORD *)(v3 - 5200) = 0;
  *(_QWORD *)(v3 - 5192) = 0;
  *(_QWORD *)(v3 - 5184) = 0;
  *(_QWORD *)(v3 - 5176) = 0;
  *(_QWORD *)(v3 - 5168) = 0;
  *(_QWORD *)(v3 - 5072) = "__gcc_qsub";
  *(_QWORD *)(v3 - 5064) = "__mulsf3";
  *(_QWORD *)(v3 - 5056) = "__muldf3";
  *(_QWORD *)(v3 - 5048) = "__mulxf3";
  *(_QWORD *)(v3 - 5040) = "__multf3";
  *(_QWORD *)(v3 - 5032) = "__gcc_qmul";
  *(_QWORD *)(v3 - 5024) = "__divsf3";
  *(_QWORD *)(v3 - 5016) = "__divdf3";
  *(_QWORD *)(v3 - 5008) = "__divxf3";
  *(_QWORD *)(v3 - 5000) = "__divtf3";
  *(_QWORD *)(v3 - 4992) = "__gcc_qdiv";
  *(_QWORD *)(v3 - 4984) = "fmodf";
  *(_QWORD *)(v3 - 4976) = "fmod";
  *(_QWORD *)(v3 - 4968) = "fmodl";
  *(_QWORD *)(v3 - 4960) = "fmodl";
  *(_QWORD *)(v3 - 4952) = "fmodl";
  *(_QWORD *)(v3 - 4944) = "fmaf";
  *(_QWORD *)(v3 - 4936) = "fma";
  *(_QWORD *)(v3 - 4928) = "fmal";
  *(_QWORD *)(v3 - 4920) = "fmal";
  *(_QWORD *)(v3 - 4912) = "fmal";
  *(_QWORD *)(v3 - 4904) = "__powisf2";
  *(_QWORD *)(v3 - 4896) = "__powidf2";
  *(_QWORD *)(v3 - 4888) = "__powixf2";
  *(_QWORD *)(v3 - 4880) = "__powitf2";
  *(_QWORD *)(v3 - 4872) = "__powitf2";
  *(_QWORD *)(v3 - 4864) = "sqrtf";
  *(_QWORD *)(v3 - 4856) = "sqrt";
  *(_QWORD *)(v3 - 4848) = "sqrtl";
  *(_QWORD *)(v3 - 4840) = "sqrtl";
  *(_QWORD *)(v3 - 4832) = "sqrtl";
  *(_QWORD *)(v3 - 4824) = "logf";
  *(_QWORD *)(v3 - 4816) = "log";
  *(_QWORD *)(v3 - 4808) = "logl";
  *(_QWORD *)(v3 - 4800) = "logl";
  *(_QWORD *)(v3 - 4792) = "logl";
  *(_QWORD *)(v3 - 4784) = "__logf_finite";
  *(_QWORD *)(v3 - 4776) = "__log_finite";
  *(_QWORD *)(v3 - 4768) = "__logl_finite";
  *(_QWORD *)(v3 - 4760) = "__logl_finite";
  *(_QWORD *)(v3 - 4752) = "__logl_finite";
  *(_QWORD *)(v3 - 4744) = "log2f";
  *(_QWORD *)(v3 - 4736) = "log2";
  *(_QWORD *)(v3 - 4728) = "log2l";
  *(_QWORD *)(v3 - 4720) = "log2l";
  *(_QWORD *)(v3 - 4712) = "log2l";
  *(_QWORD *)(v3 - 4704) = "__log2f_finite";
  *(_QWORD *)(v3 - 4696) = "__log2_finite";
  *(_QWORD *)(v3 - 4688) = "__log2l_finite";
  *(_QWORD *)(v3 - 4680) = "__log2l_finite";
  *(_QWORD *)(v3 - 4672) = "__log2l_finite";
  *(_QWORD *)(v3 - 4664) = "log10f";
  *(_QWORD *)(v3 - 4656) = "log10";
  *(_QWORD *)(v3 - 4648) = "log10l";
  *(_QWORD *)(v3 - 4640) = "log10l";
  *(_QWORD *)(v3 - 4632) = "log10l";
  *(_QWORD *)(v3 - 4624) = "__log10f_finite";
  *(_QWORD *)(v3 - 4616) = "__log10_finite";
  *(_QWORD *)(v3 - 4608) = "__log10l_finite";
  *(_QWORD *)(v3 - 4600) = "__log10l_finite";
  *(_QWORD *)(v3 - 4592) = "__log10l_finite";
  *(_QWORD *)(v3 - 4584) = "expf";
  *(_QWORD *)(v3 - 4576) = "exp";
  *(_QWORD *)(v3 - 4568) = "expl";
  *(_QWORD *)(v3 - 4560) = "expl";
  *(_QWORD *)(v3 - 4552) = "expl";
  *(_QWORD *)(v3 - 4544) = "__expf_finite";
  *(_QWORD *)(v3 - 4536) = "__exp_finite";
  *(_QWORD *)(v3 - 4528) = "__expl_finite";
  *(_QWORD *)(v3 - 4520) = "__expl_finite";
  *(_QWORD *)(v3 - 4512) = "__expl_finite";
  *(_QWORD *)(v3 - 4504) = "exp2f";
  *(_QWORD *)(v3 - 4496) = "exp2";
  *(_QWORD *)(v3 - 4488) = "exp2l";
  *(_QWORD *)(v3 - 4480) = "exp2l";
  *(_QWORD *)(v3 - 4472) = "exp2l";
  *(_QWORD *)(v3 - 4464) = "__exp2f_finite";
  *(_QWORD *)(v3 - 4456) = "__exp2_finite";
  *(_QWORD *)(v3 - 4448) = "__exp2l_finite";
  *(_QWORD *)(v3 - 4440) = "__exp2l_finite";
  *(_QWORD *)(v3 - 4432) = "__exp2l_finite";
  *(_QWORD *)(v3 - 4424) = "sinf";
  *(_QWORD *)(v3 - 4416) = "sin";
  *(_QWORD *)(v3 - 4408) = "sinl";
  *(_QWORD *)(v3 - 4400) = "sinl";
  *(_QWORD *)(v3 - 4392) = "sinl";
  *(_QWORD *)(v3 - 4384) = "cosf";
  *(_QWORD *)(v3 - 4376) = "cos";
  *(_QWORD *)(v3 - 4368) = "cosl";
  *(_QWORD *)(v3 - 4360) = "cosl";
  *(_QWORD *)(v3 - 4352) = "cosl";
  *(_QWORD *)(v3 - 4288) = "powf";
  *(_QWORD *)(v3 - 4344) = 0;
  *(_QWORD *)(v3 - 4336) = 0;
  *(_QWORD *)(v3 - 4328) = 0;
  *(_QWORD *)(v3 - 4320) = 0;
  *(_QWORD *)(v3 - 4312) = 0;
  *(_QWORD *)(v3 - 4304) = 0;
  *(_QWORD *)(v3 - 4296) = 0;
  *(_QWORD *)(v3 - 4280) = "pow";
  *(_QWORD *)(v3 - 4272) = "powl";
  *(_QWORD *)(v3 - 4264) = "powl";
  *(_QWORD *)(v3 - 4256) = "powl";
  *(_QWORD *)(v3 - 4248) = "__powf_finite";
  *(_QWORD *)(v3 - 4240) = "__pow_finite";
  *(_QWORD *)(v3 - 4232) = "__powl_finite";
  *(_QWORD *)(v3 - 4224) = "__powl_finite";
  *(_QWORD *)(v3 - 4216) = "__powl_finite";
  *(_QWORD *)(v3 - 4208) = "ceilf";
  *(_QWORD *)(v3 - 4200) = "ceil";
  *(_QWORD *)(v3 - 4192) = "ceill";
  *(_QWORD *)(v3 - 4184) = "ceill";
  *(_QWORD *)(v3 - 4176) = "ceill";
  *(_QWORD *)(v3 - 4168) = "truncf";
  *(_QWORD *)(v3 - 4160) = "trunc";
  *(_QWORD *)(v3 - 4152) = "truncl";
  *(_QWORD *)(v3 - 4144) = "truncl";
  *(_QWORD *)(v3 - 4136) = "truncl";
  *(_QWORD *)(v3 - 4128) = "rintf";
  *(_QWORD *)(v3 - 4120) = "rint";
  *(_QWORD *)(v3 - 4112) = "rintl";
  *(_QWORD *)(v3 - 4104) = "rintl";
  *(_QWORD *)(v3 - 4096) = "rintl";
  *(_QWORD *)(v3 - 4088) = "nearbyintf";
  *(_QWORD *)(v3 - 4080) = "nearbyint";
  *(_QWORD *)(v3 - 4072) = "nearbyintl";
  *(_QWORD *)(v3 - 4064) = "nearbyintl";
  *(_QWORD *)(v3 - 4056) = "nearbyintl";
  *(_QWORD *)(v3 - 4048) = "roundf";
  *(_QWORD *)(v3 - 4040) = "round";
  *(_QWORD *)(v3 - 4032) = "roundl";
  *(_QWORD *)(v3 - 4024) = "roundl";
  *(_QWORD *)(v3 - 4016) = "roundl";
  *(_QWORD *)(v3 - 4008) = "floorf";
  *(_QWORD *)(v3 - 4000) = "floor";
  *(_QWORD *)(v3 - 3992) = "floorl";
  *(_QWORD *)(v3 - 3984) = "floorl";
  *(_QWORD *)(v3 - 3976) = "floorl";
  *(_QWORD *)(v3 - 3968) = "copysignf";
  *(_QWORD *)(v3 - 3960) = "copysign";
  *(_QWORD *)(v3 - 3952) = "copysignl";
  *(_QWORD *)(v3 - 3944) = "copysignl";
  *(_QWORD *)(v3 - 3936) = "copysignl";
  *(_QWORD *)(v3 - 3928) = "fminf";
  *(_QWORD *)(v3 - 3920) = "fmin";
  *(_QWORD *)(v3 - 3912) = "fminl";
  *(_QWORD *)(v3 - 3904) = "fminl";
  *(_QWORD *)(v3 - 3896) = "fminl";
  *(_QWORD *)(v3 - 3888) = "fmaxf";
  *(_QWORD *)(v3 - 3880) = "fmax";
  *(_QWORD *)(v3 - 3872) = "fmaxl";
  *(_QWORD *)(v3 - 3864) = "fmaxl";
  *(_QWORD *)(v3 - 3856) = "fmaxl";
  *(_QWORD *)(v3 - 3848) = "__gcc_stoq";
  *(_QWORD *)(v3 - 3840) = "__gcc_dtoq";
  *(_QWORD *)(v3 - 3832) = "__extendxftf2";
  *(_QWORD *)(v3 - 3824) = "__extenddftf2";
  *(_QWORD *)(v3 - 3816) = "__extendsftf2";
  *(_QWORD *)(v3 - 3808) = "__extendsfdf2";
  *(_QWORD *)(v3 - 3800) = "__gnu_h2f_ieee";
  *(_QWORD *)(v3 - 3792) = "__gnu_f2h_ieee";
  *(_QWORD *)(v3 - 3784) = "__truncdfhf2";
  *(_QWORD *)(v3 - 3776) = "__truncxfhf2";
  *(_QWORD *)(v3 - 3768) = "__trunctfhf2";
  *(_QWORD *)(v3 - 3760) = "__trunctfhf2";
  *(_QWORD *)(v3 - 3752) = "__truncdfsf2";
  *(_QWORD *)(v3 - 3744) = "__truncxfsf2";
  *(_QWORD *)(v3 - 3736) = "__trunctfsf2";
  *(_QWORD *)(v3 - 3728) = "__gcc_qtos";
  *(_QWORD *)(v3 - 3720) = "__truncxfdf2";
  *(_QWORD *)(v3 - 3712) = "__trunctfdf2";
  *(_QWORD *)(v3 - 3704) = "__gcc_qtod";
  *(_QWORD *)(v3 - 3696) = "__trunctfxf2";
  *(_QWORD *)(v3 - 3688) = "__fixsfsi";
  *(_QWORD *)(v3 - 3680) = "__fixsfdi";
  *(_QWORD *)(v3 - 3672) = "__fixsfti";
  *(_QWORD *)(v3 - 3664) = "__fixdfsi";
  *(_QWORD *)(v3 - 3656) = "__fixdfdi";
  *(_QWORD *)(v3 - 3648) = "__fixdfti";
  *(_QWORD *)(v3 - 3640) = "__fixxfsi";
  *(_QWORD *)(v3 - 3632) = "__fixxfdi";
  *(_QWORD *)(v3 - 3624) = "__fixxfti";
  *(_QWORD *)(v3 - 3616) = "__fixtfsi";
  *(_QWORD *)(v3 - 3600) = "__fixtfti";
  *(_QWORD *)(v3 - 3576) = "__fixtfti";
  *(_QWORD *)(v3 - 3568) = "__fixunssfsi";
  *(_QWORD *)(v3 - 3560) = "__fixunssfdi";
  *(_QWORD *)(v3 - 3552) = "__fixunssfti";
  *(_QWORD *)(v3 - 3544) = "__fixunsdfsi";
  *(_QWORD *)(v3 - 3536) = "__fixunsdfdi";
  *(_QWORD *)(v3 - 3528) = "__fixunsdfti";
  *(_QWORD *)(v3 - 3520) = "__fixunsxfsi";
  *(_QWORD *)(v3 - 3512) = "__fixunsxfdi";
  *(_QWORD *)(v3 - 3608) = "__fixtfdi";
  *(_QWORD *)(v3 - 3584) = "__fixtfdi";
  *(_QWORD *)(v3 - 3504) = "__fixunsxfti";
  *(_QWORD *)(v3 - 3592) = "__gcc_qtou";
  *(_QWORD *)(v3 - 3496) = "__fixunstfsi";
  *(_QWORD *)(v3 - 3488) = "__fixunstfdi";
  *(_QWORD *)(v3 - 3480) = "__fixunstfti";
  *(_QWORD *)(v3 - 3456) = "__fixunstfti";
  *(_QWORD *)(v3 - 3448) = "__floatsisf";
  *(_QWORD *)(v3 - 3440) = "__floatsidf";
  *(_QWORD *)(v3 - 3432) = "__floatsixf";
  *(_QWORD *)(v3 - 3424) = "__floatsitf";
  *(_QWORD *)(v3 - 3416) = "__gcc_itoq";
  *(_QWORD *)(v3 - 3408) = "__floatdisf";
  *(_QWORD *)(v3 - 3400) = "__floatdidf";
  *(_QWORD *)(v3 - 3392) = "__floatdixf";
  *(_QWORD *)(v3 - 3384) = "__floatditf";
  *(_QWORD *)(v3 - 3376) = "__floatditf";
  *(_QWORD *)(v3 - 3368) = "__floattisf";
  *(_QWORD *)(v3 - 3360) = "__floattidf";
  *(_QWORD *)(v3 - 3352) = "__floattixf";
  *(_QWORD *)(v3 - 3344) = "__floattitf";
  *(_QWORD *)(v3 - 3336) = "__floattitf";
  *(_QWORD *)(v3 - 3328) = "__floatunsisf";
  *(_QWORD *)(v3 - 3320) = "__floatunsidf";
  *(_QWORD *)(v3 - 3312) = "__floatunsixf";
  *(_QWORD *)(v3 - 3304) = "__floatunsitf";
  *(_QWORD *)(v3 - 3296) = "__gcc_utoq";
  *(_QWORD *)(v3 - 3288) = "__floatundisf";
  *(_QWORD *)(v3 - 3280) = "__floatundidf";
  *(_QWORD *)(v3 - 3272) = "__floatundixf";
  *(_QWORD *)(v3 - 3264) = "__floatunditf";
  *(_QWORD *)(v3 - 3256) = "__floatunditf";
  *(_QWORD *)(v3 - 3248) = "__floatuntisf";
  *(_QWORD *)(v3 - 3240) = "__floatuntidf";
  *(_QWORD *)(v3 - 3232) = "__floatuntixf";
  *(_QWORD *)(v3 - 3472) = "__fixunstfsi";
  *(_QWORD *)(v3 - 3464) = "__fixunstfdi";
  *(_QWORD *)(v3 - 3224) = "__floatuntitf";
  *(_QWORD *)(v3 - 3216) = "__floatuntitf";
  *(_QWORD *)(v3 - 3208) = "__eqsf2";
  *(_QWORD *)(v3 - 3200) = "__eqdf2";
  *(_QWORD *)(v3 - 3192) = "__eqtf2";
  *(_QWORD *)(v3 - 3184) = "__gcc_qeq";
  *(_QWORD *)(v3 - 3176) = "__nesf2";
  *(_QWORD *)(v3 - 3168) = "__nedf2";
  *(_QWORD *)(v3 - 3160) = "__netf2";
  *(_QWORD *)(v3 - 3152) = "__gcc_qne";
  *(_QWORD *)(v3 - 3144) = "__gesf2";
  *(_QWORD *)(v3 - 3136) = "__gedf2";
  *(_QWORD *)(v3 - 3128) = "__getf2";
  *(_QWORD *)(v3 - 3120) = "__gcc_qge";
  *(_QWORD *)(v3 - 3112) = "__ltsf2";
  *(_QWORD *)(v3 - 3104) = "__ltdf2";
  *(_QWORD *)(v3 - 3096) = "__lttf2";
  *(_QWORD *)(v3 - 3088) = "__gcc_qlt";
  *(_QWORD *)(v3 - 3080) = "__lesf2";
  *(_QWORD *)(v3 - 3072) = "__ledf2";
  *(_QWORD *)(v3 - 3064) = "__letf2";
  *(_QWORD *)(v3 - 3056) = "__gcc_qle";
  *(_QWORD *)(v3 - 3048) = "__gtsf2";
  *(_QWORD *)(v3 - 3040) = "__gtdf2";
  *(_QWORD *)(v3 - 3032) = "__gttf2";
  *(_QWORD *)(v3 - 3024) = "__gcc_qgt";
  *(_QWORD *)(v3 - 3008) = "__unorddf2";
  *(_QWORD *)(v3 - 2992) = "__gcc_qunord";
  *(_QWORD *)(v3 - 2976) = "__unorddf2";
  *(_QWORD *)(v3 - 3016) = "__unordsf2";
  *(_QWORD *)(v3 - 3000) = "__unordtf2";
  *(_QWORD *)(v3 - 2984) = "__unordsf2";
  *(_QWORD *)(v3 - 2968) = "__unordtf2";
  *(_QWORD *)(v3 - 2960) = "__gcc_qunord";
  *(_QWORD *)(v3 - 2952) = "memcpy";
  *(_QWORD *)(v3 - 2944) = "memmove";
  *(_QWORD *)(v3 - 2936) = "memset";
  *(_QWORD *)(v3 - 2920) = "__llvm_memcpy_element_unordered_atomic_1";
  *(_QWORD *)(v3 - 2912) = "__llvm_memcpy_element_unordered_atomic_2";
  *(_QWORD *)(v3 - 2904) = "__llvm_memcpy_element_unordered_atomic_4";
  *(_QWORD *)(v3 - 2896) = "__llvm_memcpy_element_unordered_atomic_8";
  *(_QWORD *)(v3 - 2888) = "__llvm_memcpy_element_unordered_atomic_16";
  *(_QWORD *)(v3 - 2880) = "__llvm_memmove_element_unordered_atomic_1";
  *(_QWORD *)(v3 - 2872) = "__llvm_memmove_element_unordered_atomic_2";
  *(_QWORD *)(v3 - 2864) = "__llvm_memmove_element_unordered_atomic_4";
  *(_QWORD *)(v3 - 2856) = "__llvm_memmove_element_unordered_atomic_8";
  *(_QWORD *)(v3 - 2848) = "__llvm_memmove_element_unordered_atomic_16";
  *(_QWORD *)(v3 - 2840) = "__llvm_memset_element_unordered_atomic_1";
  *(_QWORD *)(v3 - 2832) = "__llvm_memset_element_unordered_atomic_2";
  *(_QWORD *)(v3 - 2824) = "__llvm_memset_element_unordered_atomic_4";
  *(_QWORD *)(v3 - 2816) = "__llvm_memset_element_unordered_atomic_8";
  *(_QWORD *)(v3 - 2808) = "__llvm_memset_element_unordered_atomic_16";
  *(_QWORD *)(v3 - 2800) = "_Unwind_Resume";
  *(_QWORD *)(v3 - 2792) = "__sync_val_compare_and_swap_1";
  *(_QWORD *)(v3 - 2784) = "__sync_val_compare_and_swap_2";
  *(_QWORD *)(v3 - 2776) = "__sync_val_compare_and_swap_4";
  *(_QWORD *)(v3 - 2768) = "__sync_val_compare_and_swap_8";
  *(_QWORD *)(v3 - 2760) = "__sync_val_compare_and_swap_16";
  *(_QWORD *)(v3 - 2752) = "__sync_lock_test_and_set_1";
  *(_QWORD *)(v3 - 2744) = "__sync_lock_test_and_set_2";
  *(_QWORD *)(v3 - 2736) = "__sync_lock_test_and_set_4";
  *(_QWORD *)(v3 - 2728) = "__sync_lock_test_and_set_8";
  *(_QWORD *)(v3 - 2720) = "__sync_lock_test_and_set_16";
  *(_QWORD *)(v3 - 2712) = "__sync_fetch_and_add_1";
  *(_QWORD *)(v3 - 2704) = "__sync_fetch_and_add_2";
  *(_QWORD *)(v3 - 2928) = 0;
  *(_QWORD *)(v3 - 2696) = "__sync_fetch_and_add_4";
  *(_QWORD *)(v3 - 2688) = "__sync_fetch_and_add_8";
  *(_QWORD *)(v3 - 2680) = "__sync_fetch_and_add_16";
  *(_QWORD *)(v3 - 2672) = "__sync_fetch_and_sub_1";
  *(_QWORD *)(v3 - 2664) = "__sync_fetch_and_sub_2";
  *(_QWORD *)(v3 - 2656) = "__sync_fetch_and_sub_4";
  *(_QWORD *)(v3 - 2648) = "__sync_fetch_and_sub_8";
  *(_QWORD *)(v3 - 2640) = "__sync_fetch_and_sub_16";
  *(_QWORD *)(v3 - 2632) = "__sync_fetch_and_and_1";
  *(_QWORD *)(v3 - 2624) = "__sync_fetch_and_and_2";
  *(_QWORD *)(v3 - 2616) = "__sync_fetch_and_and_4";
  *(_QWORD *)(v3 - 2608) = "__sync_fetch_and_and_8";
  *(_QWORD *)(v3 - 2600) = "__sync_fetch_and_and_16";
  *(_QWORD *)(v3 - 2592) = "__sync_fetch_and_or_1";
  *(_QWORD *)(v3 - 2584) = "__sync_fetch_and_or_2";
  *(_QWORD *)(v3 - 2576) = "__sync_fetch_and_or_4";
  *(_QWORD *)(v3 - 2568) = "__sync_fetch_and_or_8";
  *(_QWORD *)(v3 - 2560) = "__sync_fetch_and_or_16";
  *(_QWORD *)(v3 - 2552) = "__sync_fetch_and_xor_1";
  *(_QWORD *)(v3 - 2544) = "__sync_fetch_and_xor_2";
  *(_QWORD *)(v3 - 2536) = "__sync_fetch_and_xor_4";
  *(_QWORD *)(v3 - 2528) = "__sync_fetch_and_xor_8";
  *(_QWORD *)(v3 - 2520) = "__sync_fetch_and_xor_16";
  *(_QWORD *)(v3 - 2512) = "__sync_fetch_and_nand_1";
  *(_QWORD *)(v3 - 2504) = "__sync_fetch_and_nand_2";
  *(_QWORD *)(v3 - 2496) = "__sync_fetch_and_nand_4";
  *(_QWORD *)(v3 - 2488) = "__sync_fetch_and_nand_8";
  *(_QWORD *)(v3 - 2480) = "__sync_fetch_and_nand_16";
  *(_QWORD *)(v3 - 2472) = "__sync_fetch_and_max_1";
  *(_QWORD *)(v3 - 2464) = "__sync_fetch_and_max_2";
  *(_QWORD *)(v3 - 2456) = "__sync_fetch_and_max_4";
  *(_QWORD *)(v3 - 2448) = "__sync_fetch_and_max_8";
  *(_QWORD *)(v3 - 2440) = "__sync_fetch_and_max_16";
  *(_QWORD *)(v3 - 2432) = "__sync_fetch_and_umax_1";
  *(_QWORD *)(v3 - 2424) = "__sync_fetch_and_umax_2";
  *(_QWORD *)(v3 - 2416) = "__sync_fetch_and_umax_4";
  *(_QWORD *)(v3 - 2408) = "__sync_fetch_and_umax_8";
  *(_QWORD *)(v3 - 2400) = "__sync_fetch_and_umax_16";
  *(_QWORD *)(v3 - 2392) = "__sync_fetch_and_min_1";
  *(_QWORD *)(v3 - 2384) = "__sync_fetch_and_min_2";
  *(_QWORD *)(v3 - 2376) = "__sync_fetch_and_min_4";
  *(_QWORD *)(v3 - 2368) = "__sync_fetch_and_min_8";
  *(_QWORD *)(v3 - 2360) = "__sync_fetch_and_min_16";
  *(_QWORD *)(v3 - 2352) = "__sync_fetch_and_umin_1";
  *(_QWORD *)(v3 - 2344) = "__sync_fetch_and_umin_2";
  *(_QWORD *)(v3 - 2336) = "__sync_fetch_and_umin_4";
  *(_QWORD *)(v3 - 2328) = "__sync_fetch_and_umin_8";
  *(_QWORD *)(v3 - 2320) = "__sync_fetch_and_umin_16";
  *(_QWORD *)(v3 - 2312) = "__atomic_load";
  *(_QWORD *)(v3 - 2304) = "__atomic_load_1";
  *(_QWORD *)(v3 - 2296) = "__atomic_load_2";
  *(_QWORD *)(v3 - 2288) = "__atomic_load_4";
  *(_QWORD *)(v3 - 2280) = "__atomic_load_8";
  *(_QWORD *)(v3 - 2272) = "__atomic_load_16";
  *(_QWORD *)(v3 - 2264) = "__atomic_store";
  *(_QWORD *)(v3 - 2256) = "__atomic_store_1";
  *(_QWORD *)(v3 - 2248) = "__atomic_store_2";
  *(_QWORD *)(v3 - 2240) = "__atomic_store_4";
  *(_QWORD *)(v3 - 2232) = "__atomic_store_8";
  *(_QWORD *)(v3 - 2224) = "__atomic_store_16";
  *(_QWORD *)(v3 - 2216) = "__atomic_exchange";
  *(_QWORD *)(v3 - 2208) = "__atomic_exchange_1";
  *(_QWORD *)(v3 - 2200) = "__atomic_exchange_2";
  *(_QWORD *)(v3 - 2192) = "__atomic_exchange_4";
  *(_QWORD *)(v3 - 2184) = "__atomic_exchange_8";
  *(_QWORD *)(v3 - 2176) = "__atomic_exchange_16";
  *(_QWORD *)(v3 - 2168) = "__atomic_compare_exchange";
  *(_QWORD *)(v3 - 2160) = "__atomic_compare_exchange_1";
  *(_QWORD *)(v3 - 2152) = "__atomic_compare_exchange_2";
  *(_QWORD *)(v3 - 2144) = "__atomic_compare_exchange_4";
  *(_QWORD *)(v3 - 2136) = "__atomic_compare_exchange_8";
  *(_QWORD *)(v3 - 2128) = "__atomic_compare_exchange_16";
  *(_QWORD *)(v3 - 2120) = "__atomic_fetch_add_1";
  *(_QWORD *)(v3 - 2112) = "__atomic_fetch_add_2";
  *(_QWORD *)(v3 - 2104) = "__atomic_fetch_add_4";
  *(_QWORD *)(v3 - 2096) = "__atomic_fetch_add_8";
  *(_QWORD *)(v3 - 2088) = "__atomic_fetch_add_16";
  *(_QWORD *)(v3 - 2080) = "__atomic_fetch_sub_1";
  *(_QWORD *)(v3 - 2072) = "__atomic_fetch_sub_2";
  *(_QWORD *)(v3 - 2064) = "__atomic_fetch_sub_4";
  *(_QWORD *)(v3 - 2056) = "__atomic_fetch_sub_8";
  *(_QWORD *)(v3 - 2048) = "__atomic_fetch_sub_16";
  *(_QWORD *)(v3 - 2040) = "__atomic_fetch_and_1";
  *(_QWORD *)(v3 - 2032) = "__atomic_fetch_and_2";
  *(_QWORD *)(v3 - 2024) = "__atomic_fetch_and_4";
  *(_QWORD *)(v3 - 2016) = "__atomic_fetch_and_8";
  *(_QWORD *)(v3 - 2008) = "__atomic_fetch_and_16";
  *(_QWORD *)(v3 - 2000) = "__atomic_fetch_or_1";
  *(_QWORD *)(v3 - 1992) = "__atomic_fetch_or_2";
  *(_QWORD *)(v3 - 1984) = "__atomic_fetch_or_4";
  *(_QWORD *)(v3 - 1976) = "__atomic_fetch_or_8";
  *(_QWORD *)(v3 - 1968) = "__atomic_fetch_or_16";
  *(_QWORD *)(v3 - 1960) = "__atomic_fetch_xor_1";
  *(_QWORD *)(v3 - 1952) = "__atomic_fetch_xor_2";
  *(_QWORD *)(v3 - 1944) = "__atomic_fetch_xor_4";
  *(_QWORD *)(v3 - 1936) = "__atomic_fetch_xor_8";
  *(_QWORD *)(v3 - 1928) = "__atomic_fetch_xor_16";
  *(_QWORD *)(v3 - 1920) = "__atomic_fetch_nand_1";
  *(_QWORD *)(v3 - 1912) = "__atomic_fetch_nand_2";
  *(_QWORD *)(v3 - 1904) = "__atomic_fetch_nand_4";
  *(_QWORD *)(v3 - 1896) = "__atomic_fetch_nand_8";
  *(_QWORD *)(v3 - 1888) = "__atomic_fetch_nand_16";
  *(_QWORD *)(v3 - 1880) = "__stack_chk_fail";
  *(_QWORD *)(v3 - 1872) = "__llvm_deoptimize";
  *(_QWORD *)(v3 - 1864) = 0;
  *(_QWORD *)(v3 - 8) = 0;
  *(_QWORD *)(v3 + 1832) = 0;
  memset((void *)(v3 & 0xFFFFFFFFFFFFFFF8LL), 0, 8 * (((unsigned int)a1 - (v3 & 0xFFFFFFF8) + 81496) >> 3));
  v4 = (unsigned int)a2[11];
  if ( (unsigned int)v4 > 0x1E )
    goto LABEL_12;
  v5 = 1610614920;
  if ( !_bittest64(&v5, v4) )
    goto LABEL_12;
  a1[9482] = "__extendhfsf2";
  a1[9483] = "__truncsfhf2";
  v6 = a2[8];
  if ( v6 == 3 )
  {
    a1[9591] = "bzero";
    goto LABEL_19;
  }
  if ( (unsigned int)(v6 - 31) > 1 )
    goto LABEL_20;
  v7 = a2[11];
  if ( v7 == 3 )
  {
    sub_16E2390((__int64)a2, &v11, &v12, v13);
    if ( v11 < 0xA )
      goto LABEL_19;
  }
  else
  {
    if ( v7 != 11 )
    {
      if ( v6 == 31 )
        goto LABEL_12;
LABEL_8:
      if ( v7 == 29 || v7 == 7 )
      {
        sub_16E2390((__int64)a2, &v11, &v12, v13);
        if ( v11 <= 6 )
          goto LABEL_12;
      }
      goto LABEL_10;
    }
    sub_16E2390((__int64)a2, &v11, &v12, v13);
    if ( v11 != 10 )
    {
      v10 = v11 <= 9;
      goto LABEL_38;
    }
    if ( v12 != 6 )
    {
      v10 = v12 <= 5;
LABEL_38:
      if ( !v10 )
        goto LABEL_30;
LABEL_19:
      v6 = a2[8];
      goto LABEL_20;
    }
  }
LABEL_30:
  a1[9591] = "__bzero";
  v6 = a2[8];
LABEL_20:
  if ( v6 == 31 )
    goto LABEL_12;
  v7 = a2[11];
  if ( v7 == 3 )
  {
    sub_16E2390((__int64)a2, &v11, &v12, v13);
    if ( v11 <= 0xC )
      goto LABEL_12;
  }
  else
  {
    if ( v7 != 11 )
      goto LABEL_8;
    sub_16E2390((__int64)a2, &v11, &v12, v13);
    if ( v11 != 10 )
    {
      v9 = v11 <= 9;
LABEL_26:
      if ( v9 )
        goto LABEL_12;
      goto LABEL_27;
    }
    if ( v12 != 9 )
    {
      v9 = v12 <= 8;
      goto LABEL_26;
    }
  }
LABEL_27:
  if ( !sub_16E2900((__int64)a2) )
    goto LABEL_12;
LABEL_10:
  a1[9419] = "__sincosf_stret";
  a1[9420] = "__sincos_stret";
  if ( a2[9] == 13 )
    *(_QWORD *)((char *)a1 + 80276) = 0x4400000044LL;
LABEL_12:
  if ( (unsigned int)(a2[12] - 1) <= 5 || (result = (unsigned int)a2[11], (_DWORD)result == 6) )
  {
    a1[9414] = "sincosf";
    a1[9415] = "sincos";
    a1[9416] = "sincosl";
    a1[9417] = "sincosl";
    a1[9418] = "sincosl";
    result = (unsigned int)a2[11];
  }
  if ( (_DWORD)result == 13 )
    a1[9722] = 0;
  return result;
}
