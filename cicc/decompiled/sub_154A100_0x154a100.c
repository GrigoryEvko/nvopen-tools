// Function: sub_154A100
// Address: 0x154a100
//
__int64 __fastcall sub_154A100(unsigned int a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax

  switch ( a1 )
  {
    case 8u:
      result = sub_1263B40(a2, "fastcc");
      break;
    case 9u:
      result = sub_1263B40(a2, "coldcc");
      break;
    case 0xAu:
      result = sub_1263B40(a2, "ghccc");
      break;
    case 0xCu:
      result = sub_1263B40(a2, "webkit_jscc");
      break;
    case 0xDu:
      result = sub_1263B40(a2, "anyregcc");
      break;
    case 0xEu:
      result = sub_1263B40(a2, "preserve_mostcc");
      break;
    case 0xFu:
      result = sub_1263B40(a2, "preserve_allcc");
      break;
    case 0x10u:
      result = sub_1263B40(a2, "swiftcc");
      break;
    case 0x11u:
      result = sub_1263B40(a2, "cxx_fast_tlscc");
      break;
    case 0x40u:
      result = sub_1263B40(a2, "x86_stdcallcc");
      break;
    case 0x41u:
      result = sub_1263B40(a2, "x86_fastcallcc");
      break;
    case 0x42u:
      result = sub_1263B40(a2, "arm_apcscc");
      break;
    case 0x43u:
      result = sub_1263B40(a2, "arm_aapcscc");
      break;
    case 0x44u:
      result = sub_1263B40(a2, "arm_aapcs_vfpcc");
      break;
    case 0x45u:
      result = sub_1263B40(a2, "msp430_intrcc");
      break;
    case 0x46u:
      result = sub_1263B40(a2, "x86_thiscallcc");
      break;
    case 0x47u:
      result = sub_1263B40(a2, "ptx_kernel");
      break;
    case 0x48u:
      result = sub_1263B40(a2, "ptx_device");
      break;
    case 0x4Bu:
      result = sub_1263B40(a2, "spir_func");
      break;
    case 0x4Cu:
      result = sub_1263B40(a2, "spir_kernel");
      break;
    case 0x4Du:
      result = sub_1263B40(a2, "intel_ocl_bicc");
      break;
    case 0x4Eu:
      result = sub_1263B40(a2, "x86_64_sysvcc");
      break;
    case 0x4Fu:
      result = sub_1263B40(a2, "win64cc");
      break;
    case 0x50u:
      result = sub_1263B40(a2, "x86_vectorcallcc");
      break;
    case 0x51u:
      result = sub_1263B40(a2, "hhvmcc");
      break;
    case 0x52u:
      result = sub_1263B40(a2, "hhvm_ccc");
      break;
    case 0x53u:
      result = sub_1263B40(a2, "x86_intrcc");
      break;
    case 0x54u:
      result = sub_1263B40(a2, "avr_intrcc ");
      break;
    case 0x55u:
      result = sub_1263B40(a2, "avr_signalcc ");
      break;
    case 0x57u:
      result = sub_1263B40(a2, "amdgpu_vs");
      break;
    case 0x58u:
      result = sub_1263B40(a2, "amdgpu_gs");
      break;
    case 0x59u:
      result = sub_1263B40(a2, "amdgpu_ps");
      break;
    case 0x5Au:
      result = sub_1263B40(a2, "amdgpu_cs");
      break;
    case 0x5Bu:
      result = sub_1263B40(a2, "amdgpu_kernel");
      break;
    case 0x5Cu:
      result = sub_1263B40(a2, "x86_regcallcc");
      break;
    case 0x5Du:
      result = sub_1263B40(a2, "amdgpu_hs");
      break;
    case 0x5Fu:
      result = sub_1263B40(a2, "amdgpu_ls");
      break;
    case 0x60u:
      result = sub_1263B40(a2, "amdgpu_es");
      break;
    default:
      v2 = sub_1263B40(a2, "cc");
      result = sub_16E7A90(v2, a1);
      break;
  }
  return result;
}
