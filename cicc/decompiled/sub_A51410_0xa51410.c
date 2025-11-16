// Function: sub_A51410
// Address: 0xa51410
//
__int64 __fastcall sub_A51410(unsigned int a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax

  switch ( a1 )
  {
    case 8u:
      result = sub_904010(a2, "fastcc");
      break;
    case 9u:
      result = sub_904010(a2, "coldcc");
      break;
    case 0xAu:
      result = sub_904010(a2, "ghccc");
      break;
    case 0xDu:
      result = sub_904010(a2, "anyregcc");
      break;
    case 0xEu:
      result = sub_904010(a2, "preserve_mostcc");
      break;
    case 0xFu:
      result = sub_904010(a2, "preserve_allcc");
      break;
    case 0x10u:
      result = sub_904010(a2, "swiftcc");
      break;
    case 0x11u:
      result = sub_904010(a2, "cxx_fast_tlscc");
      break;
    case 0x12u:
      result = sub_904010(a2, "tailcc");
      break;
    case 0x13u:
      result = sub_904010(a2, "cfguard_checkcc");
      break;
    case 0x14u:
      result = sub_904010(a2, "swifttailcc");
      break;
    case 0x15u:
      result = sub_904010(a2, "preserve_nonecc");
      break;
    case 0x40u:
      result = sub_904010(a2, "x86_stdcallcc");
      break;
    case 0x41u:
      result = sub_904010(a2, "x86_fastcallcc");
      break;
    case 0x42u:
      result = sub_904010(a2, "arm_apcscc");
      break;
    case 0x43u:
      result = sub_904010(a2, "arm_aapcscc");
      break;
    case 0x44u:
      result = sub_904010(a2, "arm_aapcs_vfpcc");
      break;
    case 0x45u:
      result = sub_904010(a2, "msp430_intrcc");
      break;
    case 0x46u:
      result = sub_904010(a2, "x86_thiscallcc");
      break;
    case 0x47u:
      result = sub_904010(a2, "ptx_kernel");
      break;
    case 0x48u:
      result = sub_904010(a2, "ptx_device");
      break;
    case 0x4Bu:
      result = sub_904010(a2, "spir_func");
      break;
    case 0x4Cu:
      result = sub_904010(a2, "spir_kernel");
      break;
    case 0x4Du:
      result = sub_904010(a2, "intel_ocl_bicc");
      break;
    case 0x4Eu:
      result = sub_904010(a2, "x86_64_sysvcc");
      break;
    case 0x4Fu:
      result = sub_904010(a2, "win64cc");
      break;
    case 0x50u:
      result = sub_904010(a2, "x86_vectorcallcc");
      break;
    case 0x51u:
      result = sub_904010(a2, "hhvmcc");
      break;
    case 0x52u:
      result = sub_904010(a2, "hhvm_ccc");
      break;
    case 0x53u:
      result = sub_904010(a2, "x86_intrcc");
      break;
    case 0x54u:
      result = sub_904010(a2, "avr_intrcc ");
      break;
    case 0x55u:
      result = sub_904010(a2, "avr_signalcc ");
      break;
    case 0x57u:
      result = sub_904010(a2, "amdgpu_vs");
      break;
    case 0x58u:
      result = sub_904010(a2, "amdgpu_gs");
      break;
    case 0x59u:
      result = sub_904010(a2, "amdgpu_ps");
      break;
    case 0x5Au:
      result = sub_904010(a2, "amdgpu_cs");
      break;
    case 0x5Bu:
      result = sub_904010(a2, "amdgpu_kernel");
      break;
    case 0x5Cu:
      result = sub_904010(a2, "x86_regcallcc");
      break;
    case 0x5Du:
      result = sub_904010(a2, "amdgpu_hs");
      break;
    case 0x5Fu:
      result = sub_904010(a2, "amdgpu_ls");
      break;
    case 0x60u:
      result = sub_904010(a2, "amdgpu_es");
      break;
    case 0x61u:
      result = sub_904010(a2, "aarch64_vector_pcs");
      break;
    case 0x62u:
      result = sub_904010(a2, "aarch64_sve_vector_pcs");
      break;
    case 0x64u:
      result = sub_904010(a2, "amdgpu_gfx");
      break;
    case 0x66u:
      result = sub_904010(a2, "aarch64_sme_preservemost_from_x0");
      break;
    case 0x67u:
      result = sub_904010(a2, "aarch64_sme_preservemost_from_x2");
      break;
    case 0x68u:
      result = sub_904010(a2, "amdgpu_cs_chain");
      break;
    case 0x69u:
      result = sub_904010(a2, "amdgpu_cs_chain_preserve");
      break;
    case 0x6Au:
      result = sub_904010(a2, "m68k_rtdcc");
      break;
    case 0x6Bu:
      result = sub_904010(a2, "graalcc");
      break;
    case 0x6Eu:
      result = sub_904010(a2, "riscv_vector_cc");
      break;
    case 0x6Fu:
      result = sub_904010(a2, "aarch64_sme_preservemost_from_x1");
      break;
    case 0x70u:
      result = sub_904010(a2, "riscv_vls_cc(32)");
      break;
    case 0x71u:
      result = sub_904010(a2, "riscv_vls_cc(64)");
      break;
    case 0x72u:
      result = sub_904010(a2, "riscv_vls_cc(128)");
      break;
    case 0x73u:
      result = sub_904010(a2, "riscv_vls_cc(256)");
      break;
    case 0x74u:
      result = sub_904010(a2, "riscv_vls_cc(512)");
      break;
    case 0x75u:
      result = sub_904010(a2, "riscv_vls_cc(1024)");
      break;
    case 0x76u:
      result = sub_904010(a2, "riscv_vls_cc(2048)");
      break;
    case 0x77u:
      result = sub_904010(a2, "riscv_vls_cc(4096)");
      break;
    case 0x78u:
      result = sub_904010(a2, "riscv_vls_cc(8192)");
      break;
    case 0x79u:
      result = sub_904010(a2, "riscv_vls_cc(16384)");
      break;
    case 0x7Au:
      result = sub_904010(a2, "riscv_vls_cc(32768)");
      break;
    case 0x7Bu:
      result = sub_904010(a2, "riscv_vls_cc(65536)");
      break;
    default:
      v2 = sub_904010(a2, "cc");
      result = sub_CB59D0(v2, a1);
      break;
  }
  return result;
}
