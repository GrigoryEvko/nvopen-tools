// Function: sub_11FF4E0
// Address: 0x11ff4e0
//
__int64 __fastcall sub_11FF4E0(__int64 a1)
{
  char *v1; // r8
  char *v3; // rbx
  char *v4; // r14
  int v5; // r13d
  char v6; // r15
  int v7; // eax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rsi
  __int64 result; // rax
  __int64 v11; // rdx
  char *v12; // rax
  char *v13; // r15
  size_t v14; // rbx
  _BYTE *v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // r14d
  char *v19; // r13
  int v20; // r14d
  char *v21; // rbx
  unsigned int v22; // r15d
  signed __int64 v23; // rax
  unsigned int v24; // r13d
  unsigned __int64 v25; // rdx
  int v26; // edx
  int v27; // eax
  int v28; // eax
  unsigned int v29; // edx
  unsigned int v30; // eax
  bool v31; // r15
  bool v32; // cc
  unsigned int v33; // eax
  __int64 v34; // rdi
  unsigned int v35; // eax
  char *v36; // [rsp+0h] [rbp-90h]
  __int64 i; // [rsp+0h] [rbp-90h]
  char *v38; // [rsp+8h] [rbp-88h]
  char *v39; // [rsp+8h] [rbp-88h]
  char *v40; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v41; // [rsp+18h] [rbp-78h]
  char *v42; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v43; // [rsp+28h] [rbp-68h]
  char *v44; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v45; // [rsp+38h] [rbp-58h]
  bool v46; // [rsp+3Ch] [rbp-54h]
  char v47; // [rsp+50h] [rbp-40h]
  char v48; // [rsp+51h] [rbp-3Fh]

  v1 = 0;
  v38 = 0;
  v36 = *(char **)a1;
  v3 = *(char **)a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 - 1LL) != 105 )
    v1 = *(char **)a1;
  v4 = v1;
  while ( 1 )
  {
    v5 = (unsigned __int8)*v3;
    v6 = *v3;
    v7 = isalnum(v5);
    if ( !v7 )
    {
      if ( (unsigned __int8)(v5 - 36) > 0x3Bu )
        goto LABEL_6;
      v11 = 0x800000000000601LL;
      if ( !_bittest64(&v11, (unsigned int)(v5 - 36)) )
        break;
    }
    if ( v4 )
    {
      if ( v38 )
        goto LABEL_14;
LABEL_18:
      if ( !v7 )
      {
        v12 = v38;
        if ( v6 != 95 )
          v12 = v3;
        v38 = v12;
      }
      goto LABEL_14;
    }
    if ( (unsigned int)(v5 - 48) >= 0xA )
      v4 = v3;
    if ( !v38 )
      goto LABEL_18;
LABEL_14:
    *(_QWORD *)a1 = ++v3;
  }
  if ( *(_BYTE *)(a1 + 160) != 1 && (_BYTE)v5 == 58 )
  {
    v17 = *(_QWORD *)(a1 + 80);
    *(_QWORD *)a1 = v3 + 1;
    sub_2241130(a1 + 72, 0, v17, v36 - 1, v3 - (v36 - 1));
    return 507;
  }
LABEL_6:
  if ( !v4 )
    v4 = v3;
  if ( v4 != v36 )
  {
    *(_QWORD *)a1 = v4;
    v8 = sub_11FE300(a1, v36, v4);
    if ( v8 - 1 <= 0x7FFFFF )
    {
      *(_QWORD *)(a1 + 112) = sub_BCCE00(*(_QWORD **)(a1 + 48), v8);
      return 527;
    }
    else
    {
      v9 = *(_QWORD *)(a1 + 56);
      v48 = 1;
      v47 = 3;
      v44 = "bitwidth for integer type out of range";
      sub_11FD800(a1, v9, (__int64)&v44, 2);
      return 1;
    }
  }
  v13 = v4 - 1;
  if ( v38 )
    v3 = v38;
  *(_QWORD *)a1 = v3;
  v14 = v3 - v13;
  if ( v14 != 4 )
  {
    switch ( v14 )
    {
      case 5uLL:
        if ( *(_DWORD *)(v4 - 1) == 1936482662 )
        {
          result = 21;
          if ( v13[4] == 101 )
            return result;
        }
        break;
      case 7uLL:
        if ( *(_DWORD *)(v4 - 1) != 1818453348 || *((_WORD *)v13 + 2) != 29281 || (result = 22, v13[6] != 101) )
        {
          if ( *(_DWORD *)(v36 - 1) != 1986622064 || *((_WORD *)v13 + 2) != 29793 || (result = 28, v13[6] != 101) )
          {
            if ( *(_DWORD *)(v36 - 1) != 1634100580 )
              goto LABEL_27;
            if ( *((_WORD *)v13 + 2) != 27765 )
              goto LABEL_27;
            result = 39;
            if ( v13[6] != 116 )
              goto LABEL_27;
          }
        }
        return result;
      case 6uLL:
        if ( *(_DWORD *)(v4 - 1) != 1768318308 || (result = 23, *((_WORD *)v13 + 2) != 25966) )
        {
          if ( *(_DWORD *)(v36 - 1) == 1651469415 && *((_WORD *)v13 + 2) == 27745 )
            return 24;
          if ( *(_DWORD *)(v36 - 1) != 1835888483 || (result = 37, *((_WORD *)v13 + 2) != 28271) )
          {
            if ( *(_DWORD *)(v36 - 1) == 1684302184 && *((_WORD *)v13 + 2) == 28261 )
              return 40;
            goto LABEL_27;
          }
        }
        return result;
      case 8uLL:
        v16 = *(_QWORD *)(v4 - 1);
        if ( v16 == 0x746E6174736E6F63LL )
          return 25;
        if ( v16 == 0x6C616E7265746E69LL )
          return 29;
        if ( *(_QWORD *)(v36 - 1) == 0x65636E6F6B6E696CLL )
          return 30;
        if ( v16 == 0x72646F5F6B616577LL )
          return 33;
        goto LABEL_27;
      case 9uLL:
        if ( *(_QWORD *)(v4 - 1) != 0x61636F6C5F6F7364LL || (result = 26, v13[8] != 108) )
        {
          if ( *(_QWORD *)(v36 - 1) != 0x6E69646E65707061LL || (result = 34, v13[8] != 103) )
          {
            if ( *(_QWORD *)(v36 - 1) != 0x726F706D696C6C64LL || (result = 35, v13[8] != 116) )
            {
              if ( *(_QWORD *)(v36 - 1) != 0x726F7078656C6C64LL || (result = 36, v13[8] != 116) )
              {
                if ( !memcmp(v4 - 1, "protected", 9u) )
                  return 41;
                goto LABEL_27;
              }
            }
          }
        }
        return result;
      case 0xFuLL:
        if ( *(_QWORD *)(v4 - 1) != 0x656572705F6F7364LL )
          goto LABEL_27;
        if ( *((_DWORD *)v13 + 2) != 1635020909 )
          goto LABEL_27;
        if ( *((_WORD *)v13 + 6) != 27746 )
          goto LABEL_27;
        result = 27;
        if ( v13[14] != 101 )
          goto LABEL_27;
        return result;
      case 0x14uLL:
        if ( *(_QWORD *)(v36 - 1) ^ 0x6C62616C69617661LL | *((_QWORD *)v13 + 1) ^ 0x6E72657478655F65LL )
          goto LABEL_27;
        result = 38;
        if ( *((_DWORD *)v13 + 4) != 2037148769 )
          goto LABEL_27;
        return result;
      case 0xCuLL:
        if ( *(_QWORD *)(v36 - 1) != 0x65636E6F6B6E696CLL || (result = 31, *((_DWORD *)v13 + 2) != 1919184735) )
        {
          if ( !memcmp(v4 - 1, "unnamed_addr", 0xCu) )
            return 42;
          goto LABEL_27;
        }
        return result;
      case 0x12uLL:
        if ( !memcmp(v4 - 1, "local_unnamed_addr", 0x12u) )
          return 43;
        break;
      default:
        if ( v14 == 22 && !memcmp(v4 - 1, "externally_initialized", 0x16u) )
          return 44;
        break;
    }
    goto LABEL_27;
  }
  if ( *(_DWORD *)(v4 - 1) == 1702195828 )
    return 20;
  if ( *(_DWORD *)(v36 - 1) == 1801545079 )
    return 32;
LABEL_27:
  if ( sub_9691B0(v4 - 1, v14, "extern_weak", 11) )
    return 45;
  if ( sub_9691B0(v4 - 1, v14, "external", 8) )
    return 46;
  if ( sub_9691B0(v4 - 1, v14, "thread_local", 12) )
    return 47;
  if ( sub_9691B0(v4 - 1, v14, "localdynamic", 12) )
    return 48;
  if ( sub_9691B0(v4 - 1, v14, "initialexec", 11) )
    return 49;
  if ( sub_9691B0(v4 - 1, v14, "localexec", 9) )
    return 50;
  if ( sub_9691B0(v4 - 1, v14, "zeroinitializer", 15) )
    return 51;
  if ( sub_9691B0(v4 - 1, v14, "undef", 5) )
    return 52;
  if ( sub_9691B0(v4 - 1, v14, "null", 4) )
    return 54;
  if ( sub_9691B0(v4 - 1, v14, "none", 4) )
    return 55;
  if ( sub_9691B0(v4 - 1, v14, "poison", 6) )
    return 53;
  if ( sub_9691B0(v4 - 1, v14, "to", 2) )
    return 56;
  if ( sub_9691B0(v4 - 1, v14, "caller", 6) )
    return 57;
  if ( sub_9691B0(v4 - 1, v14, "within", 6) )
    return 58;
  if ( sub_9691B0(v4 - 1, v14, "from", 4) )
    return 59;
  if ( sub_9691B0(v4 - 1, v14, "tail", 4) )
    return 60;
  if ( sub_9691B0(v4 - 1, v14, "musttail", 8) )
    return 61;
  if ( sub_9691B0(v4 - 1, v14, "notail", 6) )
    return 62;
  if ( sub_9691B0(v4 - 1, v14, "target", 6) )
    return 63;
  if ( sub_9691B0(v4 - 1, v14, "triple", 6) )
    return 64;
  if ( sub_9691B0(v4 - 1, v14, "source_filename", 15) )
    return 65;
  if ( sub_9691B0(v4 - 1, v14, "unwind", 6) )
    return 66;
  if ( sub_9691B0(v4 - 1, v14, "datalayout", 10) )
    return 67;
  if ( sub_9691B0(v4 - 1, v14, "volatile", 8) )
    return 68;
  if ( sub_9691B0(v4 - 1, v14, "atomic", 6) )
    return 69;
  if ( sub_9691B0(v4 - 1, v14, "unordered", 9) )
    return 70;
  if ( sub_9691B0(v4 - 1, v14, "monotonic", 9) )
    return 71;
  if ( sub_9691B0(v4 - 1, v14, "acquire", 7) )
    return 72;
  if ( sub_9691B0(v4 - 1, v14, "release", 7) )
    return 73;
  if ( sub_9691B0(v4 - 1, v14, "acq_rel", 7) )
    return 74;
  if ( sub_9691B0(v4 - 1, v14, "seq_cst", 7) )
    return 75;
  if ( sub_9691B0(v4 - 1, v14, "syncscope", 9) )
    return 76;
  if ( sub_9691B0(v4 - 1, v14, "nnan", 4) )
    return 77;
  if ( sub_9691B0(v4 - 1, v14, "ninf", 4) )
    return 78;
  if ( sub_9691B0(v4 - 1, v14, "nsz", 3) )
    return 79;
  if ( sub_9691B0(v4 - 1, v14, "arcp", 4) )
    return 80;
  if ( sub_9691B0(v4 - 1, v14, "contract", 8) )
    return 81;
  if ( sub_9691B0(v4 - 1, v14, "reassoc", 7) )
    return 82;
  if ( sub_9691B0(v4 - 1, v14, "afn", 3) )
    return 83;
  if ( sub_9691B0(v4 - 1, v14, "fast", 4) )
    return 84;
  if ( sub_9691B0(v4 - 1, v14, "nuw", 3) )
    return 85;
  if ( sub_9691B0(v4 - 1, v14, "nsw", 3) )
    return 86;
  if ( sub_9691B0(v4 - 1, v14, "nusw", 4) )
    return 87;
  if ( sub_9691B0(v4 - 1, v14, "exact", 5) )
    return 88;
  if ( sub_9691B0(v4 - 1, v14, "disjoint", 8) )
    return 89;
  if ( sub_9691B0(v4 - 1, v14, "inbounds", 8) )
    return 90;
  if ( sub_9691B0(v4 - 1, v14, "nneg", 4) )
    return 91;
  if ( sub_9691B0(v4 - 1, v14, "samesign", 8) )
    return 92;
  if ( sub_9691B0(v4 - 1, v14, "inrange", 7) )
    return 93;
  if ( sub_9691B0(v4 - 1, v14, "addrspace", 9) )
    return 94;
  if ( sub_9691B0(v4 - 1, v14, "section", 7) )
    return 95;
  if ( sub_9691B0(v4 - 1, v14, "partition", 9) )
    return 96;
  if ( sub_9691B0(v4 - 1, v14, "code_model", 10) )
    return 97;
  if ( sub_9691B0(v4 - 1, v14, "alias", 5) )
    return 98;
  if ( sub_9691B0(v4 - 1, v14, "ifunc", 5) )
    return 99;
  if ( sub_9691B0(v4 - 1, v14, "module", 6) )
    return 100;
  if ( sub_9691B0(v4 - 1, v14, "asm", 3) )
    return 101;
  if ( sub_9691B0(v4 - 1, v14, "sideeffect", 10) )
    return 102;
  if ( sub_9691B0(v4 - 1, v14, "inteldialect", 12) )
    return 103;
  if ( sub_9691B0(v4 - 1, v14, "gc", 2) )
    return 104;
  if ( sub_9691B0(v4 - 1, v14, "prefix", 6) )
    return 105;
  if ( sub_9691B0(v4 - 1, v14, "prologue", 8) )
    return 106;
  if ( sub_9691B0(v4 - 1, v14, "no_sanitize_address", 19) )
    return 499;
  if ( sub_9691B0(v4 - 1, v14, "no_sanitize_hwaddress", 21) )
    return 500;
  if ( sub_9691B0(v4 - 1, v14, "sanitize_address_dyninit", 24) )
    return 501;
  if ( sub_9691B0(v4 - 1, v14, "ccc", 3) )
    return 109;
  if ( sub_9691B0(v4 - 1, v14, "fastcc", 6) )
    return 110;
  if ( sub_9691B0(v4 - 1, v14, "coldcc", 6) )
    return 111;
  if ( sub_9691B0(v4 - 1, v14, "cfguard_checkcc", 15) )
    return 113;
  if ( sub_9691B0(v4 - 1, v14, "x86_stdcallcc", 13) )
    return 114;
  if ( sub_9691B0(v4 - 1, v14, "x86_fastcallcc", 14) )
    return 115;
  if ( sub_9691B0(v4 - 1, v14, "x86_thiscallcc", 14) )
    return 116;
  if ( sub_9691B0(v4 - 1, v14, "x86_vectorcallcc", 16) )
    return 117;
  if ( sub_9691B0(v4 - 1, v14, "arm_apcscc", 10) )
    return 119;
  if ( sub_9691B0(v4 - 1, v14, "arm_aapcscc", 11) )
    return 120;
  if ( sub_9691B0(v4 - 1, v14, "arm_aapcs_vfpcc", 15) )
    return 121;
  if ( sub_9691B0(v4 - 1, v14, "aarch64_vector_pcs", 18) )
    return 122;
  if ( sub_9691B0(v4 - 1, v14, "aarch64_sve_vector_pcs", 22) )
    return 123;
  if ( sub_9691B0(v4 - 1, v14, "aarch64_sme_preservemost_from_x0", 32) )
    return 124;
  if ( sub_9691B0(v4 - 1, v14, "aarch64_sme_preservemost_from_x1", 32) )
    return 125;
  if ( sub_9691B0(v4 - 1, v14, "aarch64_sme_preservemost_from_x2", 32) )
    return 126;
  if ( sub_9691B0(v4 - 1, v14, "msp430_intrcc", 13) )
    return 127;
  if ( sub_9691B0(v4 - 1, v14, "avr_intrcc", 10) )
    return 128;
  if ( sub_9691B0(v4 - 1, v14, "avr_signalcc", 12) )
    return 129;
  if ( sub_9691B0(v4 - 1, v14, "ptx_kernel", 10) )
    return 130;
  if ( sub_9691B0(v4 - 1, v14, "ptx_device", 10) )
    return 131;
  if ( sub_9691B0(v4 - 1, v14, "spir_kernel", 11) )
    return 132;
  if ( sub_9691B0(v4 - 1, v14, "spir_func", 9) )
    return 133;
  if ( sub_9691B0(v4 - 1, v14, "intel_ocl_bicc", 14) )
    return 112;
  if ( sub_9691B0(v4 - 1, v14, "x86_64_sysvcc", 13) )
    return 134;
  if ( sub_9691B0(v4 - 1, v14, "win64cc", 7) )
    return 135;
  if ( sub_9691B0(v4 - 1, v14, "x86_regcallcc", 13) )
    return 118;
  if ( sub_9691B0(v4 - 1, v14, "swiftcc", 7) )
    return 137;
  if ( sub_9691B0(v4 - 1, v14, "swifttailcc", 11) )
    return 138;
  if ( sub_9691B0(v4 - 1, v14, "anyregcc", 8) )
    return 136;
  if ( sub_9691B0(v4 - 1, v14, "preserve_mostcc", 15) )
    return 139;
  if ( sub_9691B0(v4 - 1, v14, "preserve_allcc", 14) )
    return 140;
  if ( sub_9691B0(v4 - 1, v14, "preserve_nonecc", 15) )
    return 141;
  if ( sub_9691B0(v4 - 1, v14, "ghccc", 5) )
    return 142;
  if ( sub_9691B0(v4 - 1, v14, "x86_intrcc", 10) )
    return 143;
  if ( sub_9691B0(v4 - 1, v14, "hhvmcc", 6) )
    return 144;
  if ( sub_9691B0(v4 - 1, v14, "hhvm_ccc", 8) )
    return 145;
  if ( sub_9691B0(v4 - 1, v14, "cxx_fast_tlscc", 14) )
    return 146;
  if ( sub_9691B0(v4 - 1, v14, "amdgpu_vs", 9) )
    return 147;
  if ( sub_9691B0(v4 - 1, v14, "amdgpu_ls", 9) )
    return 148;
  if ( sub_9691B0(v4 - 1, v14, "amdgpu_hs", 9) )
    return 149;
  if ( sub_9691B0(v4 - 1, v14, "amdgpu_es", 9) )
    return 150;
  if ( sub_9691B0(v4 - 1, v14, "amdgpu_gs", 9) )
    return 151;
  if ( sub_9691B0(v4 - 1, v14, "amdgpu_ps", 9) )
    return 152;
  if ( sub_9691B0(v4 - 1, v14, "amdgpu_cs", 9) )
    return 153;
  if ( sub_9691B0(v4 - 1, v14, "amdgpu_cs_chain", 15) )
    return 154;
  if ( sub_9691B0(v4 - 1, v14, "amdgpu_cs_chain_preserve", 24) )
    return 155;
  if ( sub_9691B0(v4 - 1, v14, "amdgpu_kernel", 13) )
    return 156;
  if ( sub_9691B0(v4 - 1, v14, "amdgpu_gfx", 10) )
    return 157;
  if ( sub_9691B0(v4 - 1, v14, "tailcc", 6) )
    return 158;
  if ( sub_9691B0(v4 - 1, v14, "m68k_rtdcc", 10) )
    return 159;
  if ( sub_9691B0(v4 - 1, v14, "graalcc", 7) )
    return 160;
  if ( sub_9691B0(v4 - 1, v14, "riscv_vector_cc", 15) )
    return 161;
  if ( sub_9691B0(v4 - 1, v14, "riscv_vls_cc", 12) )
    return 162;
  if ( sub_9691B0(v4 - 1, v14, "cc", 2) )
    return 108;
  if ( sub_9691B0(v4 - 1, v14, "c", 1) )
    return 107;
  if ( sub_9691B0(v4 - 1, v14, "attributes", 10) )
    return 163;
  if ( sub_9691B0(v4 - 1, v14, "sync", 4) )
    return 164;
  if ( sub_9691B0(v4 - 1, v14, "async", 5) )
    return 165;
  if ( sub_9691B0(v4 - 1, v14, "allocalign", 10) )
    return 166;
  if ( sub_9691B0(v4 - 1, v14, "allocptr", 8) )
    return 167;
  if ( sub_9691B0(v4 - 1, v14, "alwaysinline", 12) )
    return 168;
  if ( sub_9691B0(v4 - 1, v14, "builtin", 7) )
    return 169;
  if ( sub_9691B0(v4 - 1, v14, "cold", 4) )
    return 170;
  if ( sub_9691B0(v4 - 1, v14, "convergent", 10) )
    return 171;
  if ( sub_9691B0(v4 - 1, v14, "coro_only_destroy_when_complete", 31) )
    return 172;
  if ( sub_9691B0(v4 - 1, v14, "coro_elide_safe", 15) )
    return 173;
  if ( sub_9691B0(v4 - 1, v14, "dead_on_unwind", 14) )
    return 174;
  if ( sub_9691B0(v4 - 1, v14, "disable_sanitizer_instrumentation", 33) )
    return 175;
  if ( sub_9691B0(v4 - 1, v14, "fn_ret_thunk_extern", 19) )
    return 176;
  if ( sub_9691B0(v4 - 1, v14, "hot", 3) )
    return 177;
  if ( sub_9691B0(v4 - 1, v14, "hybrid_patchable", 16) )
    return 178;
  if ( sub_9691B0(v4 - 1, v14, "immarg", 6) )
    return 179;
  if ( sub_9691B0(v4 - 1, v14, "inreg", 5) )
    return 180;
  if ( sub_9691B0(v4 - 1, v14, "inlinehint", 10) )
    return 181;
  if ( sub_9691B0(v4 - 1, v14, "jumptable", 9) )
    return 182;
  if ( sub_9691B0(v4 - 1, v14, "minsize", 7) )
    return 183;
  if ( sub_9691B0(v4 - 1, v14, "mustprogress", 12) )
    return 184;
  if ( sub_9691B0(v4 - 1, v14, "naked", 5) )
    return 185;
  if ( sub_9691B0(v4 - 1, v14, "nest", 4) )
    return 186;
  if ( sub_9691B0(v4 - 1, v14, "noalias", 7) )
    return 187;
  if ( sub_9691B0(v4 - 1, v14, "nobuiltin", 9) )
    return 188;
  if ( sub_9691B0(v4 - 1, v14, "nocallback", 10) )
    return 189;
  if ( sub_9691B0(v4 - 1, v14, "nocf_check", 10) )
    return 190;
  if ( sub_9691B0(v4 - 1, v14, "nodivergencesource", 18) )
    return 191;
  if ( sub_9691B0(v4 - 1, v14, "noduplicate", 11) )
    return 192;
  if ( sub_9691B0(v4 - 1, v14, "noext", 5) )
    return 193;
  if ( sub_9691B0(v4 - 1, v14, "nofree", 6) )
    return 194;
  if ( sub_9691B0(v4 - 1, v14, "noimplicitfloat", 15) )
    return 195;
  if ( sub_9691B0(v4 - 1, v14, "noinline", 8) )
    return 196;
  if ( sub_9691B0(v4 - 1, v14, "nomerge", 7) )
    return 197;
  if ( sub_9691B0(v4 - 1, v14, "noprofile", 9) )
    return 198;
  if ( sub_9691B0(v4 - 1, v14, "norecurse", 9) )
    return 199;
  if ( sub_9691B0(v4 - 1, v14, "noredzone", 9) )
    return 200;
  if ( sub_9691B0(v4 - 1, v14, "noreturn", 8) )
    return 201;
  if ( sub_9691B0(v4 - 1, v14, "nosanitize_bounds", 17) )
    return 202;
  if ( sub_9691B0(v4 - 1, v14, "nosanitize_coverage", 19) )
    return 203;
  if ( sub_9691B0(v4 - 1, v14, "nosync", 6) )
    return 204;
  if ( sub_9691B0(v4 - 1, v14, "noundef", 7) )
    return 205;
  if ( sub_9691B0(v4 - 1, v14, "nounwind", 8) )
    return 206;
  if ( sub_9691B0(v4 - 1, v14, "nonlazybind", 11) )
    return 207;
  if ( sub_9691B0(v4 - 1, v14, "nonnull", 7) )
    return 208;
  if ( sub_9691B0(v4 - 1, v14, "null_pointer_is_valid", 21) )
    return 209;
  if ( sub_9691B0(v4 - 1, v14, "optforfuzzing", 13) )
    return 210;
  if ( sub_9691B0(v4 - 1, v14, "optdebug", 8) )
    return 211;
  if ( sub_9691B0(v4 - 1, v14, "optsize", 7) )
    return 212;
  if ( sub_9691B0(v4 - 1, v14, "optnone", 7) )
    return 213;
  if ( sub_9691B0(v4 - 1, v14, "presplitcoroutine", 17) )
    return 214;
  if ( sub_9691B0(v4 - 1, v14, "readnone", 8) )
    return 215;
  if ( sub_9691B0(v4 - 1, v14, "readonly", 8) )
    return 216;
  if ( sub_9691B0(v4 - 1, v14, "returned", 8) )
    return 217;
  if ( sub_9691B0(v4 - 1, v14, "returns_twice", 13) )
    return 218;
  if ( sub_9691B0(v4 - 1, v14, "signext", 7) )
    return 219;
  if ( sub_9691B0(v4 - 1, v14, "safestack", 9) )
    return 220;
  if ( sub_9691B0(v4 - 1, v14, "sanitize_address", 16) )
    return 221;
  if ( sub_9691B0(v4 - 1, v14, "sanitize_hwaddress", 18) )
    return 222;
  if ( sub_9691B0(v4 - 1, v14, "sanitize_memtag", 15) )
    return 223;
  if ( sub_9691B0(v4 - 1, v14, "sanitize_memory", 15) )
    return 224;
  if ( sub_9691B0(v4 - 1, v14, "sanitize_numerical_stability", 28) )
    return 225;
  if ( sub_9691B0(v4 - 1, v14, "sanitize_realtime", 17) )
    return 226;
  if ( sub_9691B0(v4 - 1, v14, "sanitize_realtime_blocking", 26) )
    return 227;
  if ( sub_9691B0(v4 - 1, v14, "sanitize_thread", 15) )
    return 228;
  if ( sub_9691B0(v4 - 1, v14, "sanitize_type", 13) )
    return 229;
  if ( sub_9691B0(v4 - 1, v14, "shadowcallstack", 15) )
    return 230;
  if ( sub_9691B0(v4 - 1, v14, "skipprofile", 11) )
    return 231;
  if ( sub_9691B0(v4 - 1, v14, "speculatable", 12) )
    return 232;
  if ( sub_9691B0(v4 - 1, v14, "speculative_load_hardening", 26) )
    return 233;
  if ( sub_9691B0(v4 - 1, v14, "ssp", 3) )
    return 234;
  if ( sub_9691B0(v4 - 1, v14, "sspreq", 6) )
    return 235;
  if ( sub_9691B0(v4 - 1, v14, "sspstrong", 9) )
    return 236;
  if ( sub_9691B0(v4 - 1, v14, "strictfp", 8) )
    return 237;
  if ( sub_9691B0(v4 - 1, v14, "swiftasync", 10) )
    return 238;
  if ( sub_9691B0(v4 - 1, v14, "swifterror", 10) )
    return 239;
  if ( sub_9691B0(v4 - 1, v14, "swiftself", 9) )
    return 240;
  if ( sub_9691B0(v4 - 1, v14, "willreturn", 10) )
    return 241;
  if ( sub_9691B0(v4 - 1, v14, "writable", 8) )
    return 242;
  if ( sub_9691B0(v4 - 1, v14, "writeonly", 9) )
    return 243;
  if ( sub_9691B0(v4 - 1, v14, "zeroext", 7) )
    return 244;
  if ( sub_9691B0(v4 - 1, v14, "byref", 5) )
    return 245;
  if ( sub_9691B0(v4 - 1, v14, "byval", 5) )
    return 246;
  if ( sub_9691B0(v4 - 1, v14, "elementtype", 11) )
    return 247;
  if ( sub_9691B0(v4 - 1, v14, "inalloca", 8) )
    return 248;
  if ( sub_9691B0(v4 - 1, v14, "preallocated", 12) )
    return 249;
  if ( sub_9691B0(v4 - 1, v14, "sret", 4) )
    return 250;
  if ( sub_9691B0(v4 - 1, v14, "align", 5) )
    return 251;
  if ( sub_9691B0(v4 - 1, v14, "allockind", 9) )
    return 252;
  if ( sub_9691B0(v4 - 1, v14, "allocsize", 9) )
    return 253;
  if ( sub_9691B0(v4 - 1, v14, "captures", 8) )
    return 254;
  if ( sub_9691B0(v4 - 1, v14, "dereferenceable", 15) )
    return 255;
  if ( sub_9691B0(v4 - 1, v14, "dereferenceable_or_null", 23) )
    return 256;
  if ( sub_9691B0(v4 - 1, v14, "memory", 6) )
    return 257;
  if ( sub_9691B0(v4 - 1, v14, "nofpclass", 9) )
    return 258;
  if ( sub_9691B0(v4 - 1, v14, "alignstack", 10) )
    return 259;
  if ( sub_9691B0(v4 - 1, v14, "uwtable", 7) )
    return 260;
  if ( sub_9691B0(v4 - 1, v14, "vscale_range", 12) )
    return 261;
  if ( sub_9691B0(v4 - 1, v14, "range", 5) )
    return 262;
  if ( sub_9691B0(v4 - 1, v14, "initializes", 11) )
    return 263;
  if ( sub_9691B0(v4 - 1, v14, "read", 4) )
    return 264;
  if ( sub_9691B0(v4 - 1, v14, "write", 5) )
    return 265;
  if ( sub_9691B0(v4 - 1, v14, "readwrite", 9) )
    return 266;
  if ( sub_9691B0(v4 - 1, v14, "argmem", 6) )
    return 267;
  if ( sub_9691B0(v4 - 1, v14, "inaccessiblemem", 15) )
    return 268;
  if ( sub_9691B0(v4 - 1, v14, "errnomem", 8) )
    return 269;
  if ( sub_9691B0(v4 - 1, v14, "argmemonly", 10) )
    return 270;
  if ( sub_9691B0(v4 - 1, v14, "inaccessiblememonly", 19) )
    return 271;
  if ( sub_9691B0(v4 - 1, v14, "inaccessiblemem_or_argmemonly", 29) )
    return 272;
  if ( sub_9691B0(v4 - 1, v14, "nocapture", 9) )
    return 273;
  if ( sub_9691B0(v4 - 1, v14, "address_is_null", 15) )
    return 275;
  if ( sub_9691B0(v4 - 1, v14, "address", 7) )
    return 274;
  if ( sub_9691B0(v4 - 1, v14, "provenance", 10) )
    return 276;
  if ( sub_9691B0(v4 - 1, v14, "read_provenance", 15) )
    return 277;
  if ( sub_9691B0(v4 - 1, v14, "all", 3) )
    return 278;
  if ( sub_9691B0(v4 - 1, v14, "nan", 3) )
    return 279;
  if ( sub_9691B0(v4 - 1, v14, "snan", 4) )
    return 280;
  if ( sub_9691B0(v4 - 1, v14, "qnan", 4) )
    return 281;
  if ( sub_9691B0(v4 - 1, v14, "inf", 3) )
    return 282;
  if ( sub_9691B0(v4 - 1, v14, "pinf", 4) )
    return 283;
  if ( sub_9691B0(v4 - 1, v14, "norm", 4) )
    return 284;
  if ( sub_9691B0(v4 - 1, v14, "nnorm", 5) )
    return 285;
  if ( sub_9691B0(v4 - 1, v14, "pnorm", 5) )
    return 286;
  if ( sub_9691B0(v4 - 1, v14, "nsub", 4) )
    return 287;
  if ( sub_9691B0(v4 - 1, v14, "psub", 4) )
    return 288;
  if ( sub_9691B0(v4 - 1, v14, "zero", 4) )
    return 289;
  if ( sub_9691B0(v4 - 1, v14, "nzero", 5) )
    return 290;
  if ( sub_9691B0(v4 - 1, v14, "pzero", 5) )
    return 291;
  if ( sub_9691B0(v4 - 1, v14, "type", 4) )
    return 292;
  if ( sub_9691B0(v4 - 1, v14, "opaque", 6) )
    return 293;
  if ( sub_9691B0(v4 - 1, v14, "comdat", 6) )
    return 294;
  if ( sub_9691B0(v4 - 1, v14, "any", 3) )
    return 295;
  if ( sub_9691B0(v4 - 1, v14, "exactmatch", 10) )
    return 296;
  if ( sub_9691B0(v4 - 1, v14, "largest", 7) )
    return 297;
  if ( sub_9691B0(v4 - 1, v14, "nodeduplicate", 13) )
    return 298;
  if ( sub_9691B0(v4 - 1, v14, "samesize", 8) )
    return 299;
  if ( sub_9691B0(v4 - 1, v14, "eq", 2) )
    return 300;
  if ( sub_9691B0(v4 - 1, v14, &unk_432C6B1, 2) )
    return 301;
  if ( sub_9691B0(v4 - 1, v14, &unk_3F2AD88, 3) )
    return 302;
  if ( sub_9691B0(v4 - 1, v14, &unk_3F2AD80, 3) )
    return 303;
  if ( sub_9691B0(v4 - 1, v14, &unk_3F2AD8C, 3) )
    return 304;
  if ( sub_9691B0(v4 - 1, v14, &unk_3F2AD84, 3) )
    return 305;
  if ( sub_9691B0(v4 - 1, v14, "ult", 3) )
    return 306;
  if ( sub_9691B0(v4 - 1, v14, "ugt", 3) )
    return 307;
  if ( sub_9691B0(v4 - 1, v14, "ule", 3) )
    return 308;
  if ( sub_9691B0(v4 - 1, v14, "uge", 3) )
    return 309;
  if ( sub_9691B0(v4 - 1, v14, "oeq", 3) )
    return 310;
  if ( sub_9691B0(v4 - 1, v14, "one", 3) )
    return 311;
  if ( sub_9691B0(v4 - 1, v14, "olt", 3) )
    return 312;
  if ( sub_9691B0(v4 - 1, v14, "ogt", 3) )
    return 313;
  if ( sub_9691B0(v4 - 1, v14, "ole", 3) )
    return 314;
  if ( sub_9691B0(v4 - 1, v14, "oge", 3) )
    return 315;
  if ( sub_9691B0(v4 - 1, v14, "ord", 3) )
    return 316;
  if ( sub_9691B0(v4 - 1, v14, "uno", 3) )
    return 317;
  if ( sub_9691B0(v4 - 1, v14, "ueq", 3) )
    return 318;
  if ( sub_9691B0(v4 - 1, v14, "une", 3) )
    return 319;
  if ( sub_9691B0(v4 - 1, v14, "xchg", 4) )
    return 320;
  if ( sub_9691B0(v4 - 1, v14, "nand", 4) )
    return 321;
  if ( sub_9691B0(v4 - 1, v14, "max", 3) )
    return 322;
  if ( sub_9691B0(v4 - 1, v14, "min", 3) )
    return 323;
  if ( sub_9691B0(v4 - 1, v14, "umax", 4) )
    return 324;
  if ( sub_9691B0(v4 - 1, v14, "umin", 4) )
    return 325;
  if ( sub_9691B0(v4 - 1, v14, "fmax", 4) )
    return 326;
  if ( sub_9691B0(v4 - 1, v14, "fmin", 4) )
    return 327;
  if ( sub_9691B0(v4 - 1, v14, "uinc_wrap", 9) )
    return 328;
  if ( sub_9691B0(v4 - 1, v14, "udec_wrap", 9) )
    return 329;
  if ( sub_9691B0(v4 - 1, v14, "usub_cond", 9) )
    return 330;
  if ( sub_9691B0(v4 - 1, v14, "usub_sat", 8) )
    return 331;
  if ( sub_9691B0(v4 - 1, v14, "splat", 5) )
    return 398;
  if ( sub_9691B0(v4 - 1, v14, "vscale", 6) )
    return 18;
  if ( sub_9691B0(v4 - 1, v14, "x", 1) )
    return 19;
  if ( sub_9691B0(v4 - 1, v14, "blockaddress", 12) )
    return 401;
  if ( sub_9691B0(v4 - 1, v14, "dso_local_equivalent", 20) )
    return 402;
  if ( sub_9691B0(v4 - 1, v14, "no_cfi", 6) )
    return 403;
  if ( sub_9691B0(v4 - 1, v14, "ptrauth", 7) )
    return 404;
  if ( sub_9691B0(v4 - 1, v14, "distinct", 8) )
    return 406;
  if ( sub_9691B0(v4 - 1, v14, "uselistorder", 12) )
    return 407;
  if ( sub_9691B0(v4 - 1, v14, "uselistorder_bb", 15) )
    return 408;
  if ( sub_9691B0(v4 - 1, v14, "personality", 11) )
    return 371;
  if ( sub_9691B0(v4 - 1, v14, "cleanup", 7) )
    return 372;
  if ( sub_9691B0(v4 - 1, v14, "catch", 5) )
    return 373;
  if ( sub_9691B0(v4 - 1, v14, "filter", 6) )
    return 374;
  if ( sub_9691B0(v4 - 1, v14, "path", 4) )
    return 409;
  if ( sub_9691B0(v4 - 1, v14, "hash", 4) )
    return 410;
  if ( sub_9691B0(v4 - 1, v14, "gv", 2) )
    return 411;
  if ( sub_9691B0(v4 - 1, v14, "guid", 4) )
    return 412;
  if ( sub_9691B0(v4 - 1, v14, "name", 4) )
    return 413;
  if ( sub_9691B0(v4 - 1, v14, "summaries", 9) )
    return 414;
  if ( sub_9691B0(v4 - 1, v14, "flags", 5) )
    return 415;
  if ( sub_9691B0(v4 - 1, v14, "blockcount", 10) )
    return 416;
  if ( sub_9691B0(v4 - 1, v14, "linkage", 7) )
    return 417;
  if ( sub_9691B0(v4 - 1, v14, "visibility", 10) )
    return 418;
  if ( sub_9691B0(v4 - 1, v14, "notEligibleToImport", 19) )
    return 419;
  if ( sub_9691B0(v4 - 1, v14, "live", 4) )
    return 420;
  if ( sub_9691B0(v4 - 1, v14, "dsoLocal", 8) )
    return 421;
  if ( sub_9691B0(v4 - 1, v14, "canAutoHide", 11) )
    return 422;
  if ( sub_9691B0(v4 - 1, v14, "importType", 10) )
    return 423;
  if ( sub_9691B0(v4 - 1, v14, "definition", 10) )
    return 424;
  if ( sub_9691B0(v4 - 1, v14, "declaration", 11) )
    return 425;
  if ( sub_9691B0(v4 - 1, v14, "function", 8) )
    return 426;
  if ( sub_9691B0(v4 - 1, v14, "insts", 5) )
    return 427;
  if ( sub_9691B0(v4 - 1, v14, "funcFlags", 9) )
    return 428;
  if ( sub_9691B0(v4 - 1, v14, "readNone", 8) )
    return 429;
  if ( sub_9691B0(v4 - 1, v14, "readOnly", 8) )
    return 430;
  if ( sub_9691B0(v4 - 1, v14, "noRecurse", 9) )
    return 431;
  if ( sub_9691B0(v4 - 1, v14, "returnDoesNotAlias", 18) )
    return 432;
  if ( sub_9691B0(v4 - 1, v14, "noInline", 8) )
    return 433;
  if ( sub_9691B0(v4 - 1, v14, "alwaysInline", 12) )
    return 434;
  if ( sub_9691B0(v4 - 1, v14, "noUnwind", 8) )
    return 435;
  if ( sub_9691B0(v4 - 1, v14, "mayThrow", 8) )
    return 436;
  if ( sub_9691B0(v4 - 1, v14, "hasUnknownCall", 14) )
    return 437;
  if ( sub_9691B0(v4 - 1, v14, "mustBeUnreachable", 17) )
    return 438;
  if ( sub_9691B0(v4 - 1, v14, "calls", 5) )
    return 439;
  if ( sub_9691B0(v4 - 1, v14, "callee", 6) )
    return 440;
  if ( sub_9691B0(v4 - 1, v14, "params", 6) )
    return 441;
  if ( sub_9691B0(v4 - 1, v14, "param", 5) )
    return 442;
  if ( sub_9691B0(v4 - 1, v14, "hotness", 7) )
    return 443;
  if ( sub_9691B0(v4 - 1, v14, "unknown", 7) )
    return 444;
  if ( sub_9691B0(v4 - 1, v14, "critical", 8) )
    return 445;
  if ( sub_9691B0(v4 - 1, v14, "relbf", 5) )
    return 446;
  if ( sub_9691B0(v4 - 1, v14, "variable", 8) )
    return 447;
  if ( sub_9691B0(v4 - 1, v14, "vTableFuncs", 11) )
    return 448;
  if ( sub_9691B0(v4 - 1, v14, "virtFunc", 8) )
    return 449;
  if ( sub_9691B0(v4 - 1, v14, "aliasee", 7) )
    return 450;
  if ( sub_9691B0(v4 - 1, v14, "refs", 4) )
    return 451;
  if ( sub_9691B0(v4 - 1, v14, "typeIdInfo", 10) )
    return 452;
  if ( sub_9691B0(v4 - 1, v14, "typeTests", 9) )
    return 453;
  if ( sub_9691B0(v4 - 1, v14, "typeTestAssumeVCalls", 20) )
    return 454;
  if ( sub_9691B0(v4 - 1, v14, "typeCheckedLoadVCalls", 21) )
    return 455;
  if ( sub_9691B0(v4 - 1, v14, "typeTestAssumeConstVCalls", 25) )
    return 456;
  if ( sub_9691B0(v4 - 1, v14, "typeCheckedLoadConstVCalls", 26) )
    return 457;
  if ( sub_9691B0(v4 - 1, v14, "vFuncId", 7) )
    return 458;
  if ( sub_9691B0(v4 - 1, v14, "offset", 6) )
    return 459;
  if ( sub_9691B0(v4 - 1, v14, "args", 4) )
    return 460;
  if ( sub_9691B0(v4 - 1, v14, "typeid", 6) )
    return 461;
  if ( sub_9691B0(v4 - 1, v14, "typeidCompatibleVTable", 22) )
    return 462;
  if ( sub_9691B0(v4 - 1, v14, "summary", 7) )
    return 463;
  if ( sub_9691B0(v4 - 1, v14, "typeTestRes", 11) )
    return 464;
  if ( sub_9691B0(v4 - 1, v14, "kind", 4) )
    return 465;
  if ( sub_9691B0(v4 - 1, v14, "unsat", 5) )
    return 466;
  if ( sub_9691B0(v4 - 1, v14, "byteArray", 9) )
    return 467;
  if ( sub_9691B0(v4 - 1, v14, "inline", 6) )
    return 468;
  if ( sub_9691B0(v4 - 1, v14, "single", 6) )
    return 469;
  if ( sub_9691B0(v4 - 1, v14, "allOnes", 7) )
    return 470;
  if ( sub_9691B0(v4 - 1, v14, "sizeM1BitWidth", 14) )
    return 471;
  if ( sub_9691B0(v4 - 1, v14, "alignLog2", 9) )
    return 472;
  if ( sub_9691B0(v4 - 1, v14, "sizeM1", 6) )
    return 473;
  if ( sub_9691B0(v4 - 1, v14, "bitMask", 7) )
    return 474;
  if ( sub_9691B0(v4 - 1, v14, "inlineBits", 10) )
    return 475;
  if ( sub_9691B0(v4 - 1, v14, "vcall_visibility", 16) )
    return 476;
  if ( sub_9691B0(v4 - 1, v14, "wpdResolutions", 14) )
    return 477;
  if ( sub_9691B0(v4 - 1, v14, "wpdRes", 6) )
    return 478;
  if ( sub_9691B0(v4 - 1, v14, "indir", 5) )
    return 479;
  if ( sub_9691B0(v4 - 1, v14, "singleImpl", 10) )
    return 480;
  if ( sub_9691B0(v4 - 1, v14, "branchFunnel", 12) )
    return 481;
  if ( sub_9691B0(v4 - 1, v14, "singleImplName", 14) )
    return 482;
  if ( sub_9691B0(v4 - 1, v14, "resByArg", 8) )
    return 483;
  if ( sub_9691B0(v4 - 1, v14, "byArg", 5) )
    return 484;
  if ( sub_9691B0(v4 - 1, v14, "uniformRetVal", 13) )
    return 485;
  if ( sub_9691B0(v4 - 1, v14, "uniqueRetVal", 12) )
    return 486;
  if ( sub_9691B0(v4 - 1, v14, "virtualConstProp", 16) )
    return 487;
  if ( sub_9691B0(v4 - 1, v14, "info", 4) )
    return 488;
  if ( sub_9691B0(v4 - 1, v14, "byte", 4) )
    return 489;
  if ( sub_9691B0(v4 - 1, v14, "bit", 3) )
    return 490;
  if ( sub_9691B0(v4 - 1, v14, "varFlags", 8) )
    return 491;
  if ( sub_9691B0(v4 - 1, v14, "callsites", 9) )
    return 492;
  if ( sub_9691B0(v4 - 1, v14, "clones", 6) )
    return 493;
  if ( sub_9691B0(v4 - 1, v14, "stackIds", 8) )
    return 494;
  if ( sub_9691B0(v4 - 1, v14, "allocs", 6) )
    return 495;
  if ( sub_9691B0(v4 - 1, v14, "versions", 8) )
    return 496;
  if ( sub_9691B0(v4 - 1, v14, "memProf", 7) )
    return 497;
  if ( sub_9691B0(v4 - 1, v14, "notcold", 7) )
    return 498;
  if ( sub_9691B0(v4 - 1, v14, "void", 4) )
  {
    *(_QWORD *)(a1 + 112) = sub_BCB120(*(_QWORD **)(a1 + 48));
    return 527;
  }
  if ( sub_9691B0(v4 - 1, v14, "half", 4) )
  {
    *(_QWORD *)(a1 + 112) = sub_BCB140(*(_QWORD **)(a1 + 48));
    return 527;
  }
  if ( sub_9691B0(v4 - 1, v14, "bfloat", 6) )
  {
    *(_QWORD *)(a1 + 112) = sub_BCB150(*(_QWORD **)(a1 + 48));
    return 527;
  }
  if ( sub_9691B0(v4 - 1, v14, "float", 5) )
  {
    *(_QWORD *)(a1 + 112) = sub_BCB160(*(_QWORD **)(a1 + 48));
    return 527;
  }
  if ( sub_9691B0(v4 - 1, v14, "double", 6) )
  {
    *(_QWORD *)(a1 + 112) = sub_BCB170(*(_QWORD **)(a1 + 48));
    return 527;
  }
  if ( sub_9691B0(v4 - 1, v14, "x86_fp80", 8) )
  {
    *(_QWORD *)(a1 + 112) = sub_BCB1A0(*(_QWORD **)(a1 + 48));
    return 527;
  }
  if ( sub_9691B0(v4 - 1, v14, "fp128", 5) )
  {
    *(_QWORD *)(a1 + 112) = sub_BCB1B0(*(_QWORD **)(a1 + 48));
    return 527;
  }
  if ( sub_9691B0(v4 - 1, v14, "ppc_fp128", 9) )
  {
    *(_QWORD *)(a1 + 112) = sub_BCB1C0(*(_QWORD **)(a1 + 48));
    return 527;
  }
  if ( sub_9691B0(v4 - 1, v14, "label", 5) )
  {
    *(_QWORD *)(a1 + 112) = sub_BCB130(*(_QWORD **)(a1 + 48));
    return 527;
  }
  if ( sub_9691B0(v4 - 1, v14, "metadata", 8) )
  {
    *(_QWORD *)(a1 + 112) = sub_BCB180(*(_QWORD **)(a1 + 48));
    return 527;
  }
  if ( sub_9691B0(v4 - 1, v14, "x86_amx", 7) )
  {
    *(_QWORD *)(a1 + 112) = sub_BCB290(*(_QWORD **)(a1 + 48));
    return 527;
  }
  if ( sub_9691B0(v4 - 1, v14, "token", 5) )
  {
    *(_QWORD *)(a1 + 112) = sub_BCB190(*(_QWORD **)(a1 + 48));
    return 527;
  }
  if ( sub_9691B0(v4 - 1, v14, "ptr", 3) )
  {
    *(_QWORD *)(a1 + 112) = sub_BCE3C0(*(__int64 **)(a1 + 48), 0);
    return 527;
  }
  if ( sub_9691B0(v4 - 1, v14, "fneg", 4) )
  {
    *(_DWORD *)(a1 + 104) = 12;
    return 332;
  }
  if ( sub_9691B0(v4 - 1, v14, "add", 3) )
  {
    *(_DWORD *)(a1 + 104) = 13;
    return 333;
  }
  if ( sub_9691B0(v4 - 1, v14, "fadd", 4) )
  {
    *(_DWORD *)(a1 + 104) = 14;
    return 334;
  }
  if ( sub_9691B0(v4 - 1, v14, "sub", 3) )
  {
    *(_DWORD *)(a1 + 104) = 15;
    return 335;
  }
  if ( sub_9691B0(v4 - 1, v14, "fsub", 4) )
  {
    *(_DWORD *)(a1 + 104) = 16;
    return 336;
  }
  if ( sub_9691B0(v4 - 1, v14, "mul", 3) )
  {
    *(_DWORD *)(a1 + 104) = 17;
    return 337;
  }
  if ( sub_9691B0(v4 - 1, v14, "fmul", 4) )
  {
    *(_DWORD *)(a1 + 104) = 18;
    return 338;
  }
  if ( sub_9691B0(v4 - 1, v14, "udiv", 4) )
  {
    *(_DWORD *)(a1 + 104) = 19;
    return 339;
  }
  if ( sub_9691B0(v4 - 1, v14, "sdiv", 4) )
  {
    *(_DWORD *)(a1 + 104) = 20;
    return 340;
  }
  if ( sub_9691B0(v4 - 1, v14, "fdiv", 4) )
  {
    *(_DWORD *)(a1 + 104) = 21;
    return 341;
  }
  if ( sub_9691B0(v4 - 1, v14, "urem", 4) )
  {
    *(_DWORD *)(a1 + 104) = 22;
    return 342;
  }
  if ( sub_9691B0(v4 - 1, v14, "srem", 4) )
  {
    *(_DWORD *)(a1 + 104) = 23;
    return 343;
  }
  if ( sub_9691B0(v4 - 1, v14, "frem", 4) )
  {
    *(_DWORD *)(a1 + 104) = 24;
    return 344;
  }
  if ( sub_9691B0(v4 - 1, v14, "shl", 3) )
  {
    *(_DWORD *)(a1 + 104) = 25;
    return 345;
  }
  if ( sub_9691B0(v4 - 1, v14, "lshr", 4) )
  {
    *(_DWORD *)(a1 + 104) = 26;
    return 346;
  }
  if ( sub_9691B0(v4 - 1, v14, "ashr", 4) )
  {
    *(_DWORD *)(a1 + 104) = 27;
    return 347;
  }
  if ( sub_9691B0(v4 - 1, v14, "and", 3) )
  {
    *(_DWORD *)(a1 + 104) = 28;
    return 348;
  }
  if ( sub_9691B0(v4 - 1, v14, "or", 2) )
  {
    *(_DWORD *)(a1 + 104) = 29;
    return 349;
  }
  if ( sub_9691B0(v4 - 1, v14, "xor", 3) )
  {
    *(_DWORD *)(a1 + 104) = 30;
    return 350;
  }
  if ( sub_9691B0(v4 - 1, v14, "icmp", 4) )
  {
    *(_DWORD *)(a1 + 104) = 53;
    return 351;
  }
  if ( sub_9691B0(v4 - 1, v14, "fcmp", 4) )
  {
    *(_DWORD *)(a1 + 104) = 54;
    return 352;
  }
  if ( sub_9691B0(v4 - 1, v14, "phi", 3) )
  {
    *(_DWORD *)(a1 + 104) = 55;
    return 353;
  }
  if ( sub_9691B0(v4 - 1, v14, "call", 4) )
  {
    *(_DWORD *)(a1 + 104) = 56;
    return 354;
  }
  if ( sub_9691B0(v4 - 1, v14, "trunc", 5) )
  {
    *(_DWORD *)(a1 + 104) = 38;
    return 355;
  }
  if ( sub_9691B0(v4 - 1, v14, "zext", 4) )
  {
    *(_DWORD *)(a1 + 104) = 39;
    return 356;
  }
  if ( sub_9691B0(v4 - 1, v14, "sext", 4) )
  {
    *(_DWORD *)(a1 + 104) = 40;
    return 357;
  }
  if ( sub_9691B0(v4 - 1, v14, "fptrunc", 7) )
  {
    *(_DWORD *)(a1 + 104) = 45;
    return 358;
  }
  if ( sub_9691B0(v4 - 1, v14, "fpext", 5) )
  {
    *(_DWORD *)(a1 + 104) = 46;
    return 359;
  }
  if ( sub_9691B0(v4 - 1, v14, "uitofp", 6) )
  {
    *(_DWORD *)(a1 + 104) = 43;
    return 360;
  }
  if ( sub_9691B0(v4 - 1, v14, "sitofp", 6) )
  {
    *(_DWORD *)(a1 + 104) = 44;
    return 361;
  }
  if ( sub_9691B0(v4 - 1, v14, "fptoui", 6) )
  {
    *(_DWORD *)(a1 + 104) = 41;
    return 362;
  }
  if ( sub_9691B0(v4 - 1, v14, "fptosi", 6) )
  {
    *(_DWORD *)(a1 + 104) = 42;
    return 363;
  }
  if ( sub_9691B0(v4 - 1, v14, "inttoptr", 8) )
  {
    *(_DWORD *)(a1 + 104) = 48;
    return 364;
  }
  if ( sub_9691B0(v4 - 1, v14, "ptrtoint", 8) )
  {
    *(_DWORD *)(a1 + 104) = 47;
    return 365;
  }
  if ( sub_9691B0(v4 - 1, v14, "bitcast", 7) )
  {
    *(_DWORD *)(a1 + 104) = 49;
    return 366;
  }
  if ( sub_9691B0(v4 - 1, v14, "addrspacecast", 13) )
  {
    *(_DWORD *)(a1 + 104) = 50;
    return 367;
  }
  if ( sub_9691B0(v4 - 1, v14, "select", 6) )
  {
    *(_DWORD *)(a1 + 104) = 57;
    return 368;
  }
  if ( sub_9691B0(v4 - 1, v14, "va_arg", 6) )
  {
    *(_DWORD *)(a1 + 104) = 60;
    return 369;
  }
  if ( sub_9691B0(v4 - 1, v14, "ret", 3) )
  {
    *(_DWORD *)(a1 + 104) = 1;
    return 375;
  }
  if ( sub_9691B0(v4 - 1, v14, "br", 2) )
  {
    *(_DWORD *)(a1 + 104) = 2;
    return 376;
  }
  if ( sub_9691B0(v4 - 1, v14, "switch", 6) )
  {
    *(_DWORD *)(a1 + 104) = 3;
    return 377;
  }
  if ( sub_9691B0(v4 - 1, v14, "indirectbr", 10) )
  {
    *(_DWORD *)(a1 + 104) = 4;
    return 378;
  }
  if ( sub_9691B0(v4 - 1, v14, "invoke", 6) )
  {
    *(_DWORD *)(a1 + 104) = 5;
    return 379;
  }
  if ( sub_9691B0(v4 - 1, v14, "resume", 6) )
  {
    *(_DWORD *)(a1 + 104) = 6;
    return 380;
  }
  if ( sub_9691B0(v4 - 1, v14, "unreachable", 11) )
  {
    *(_DWORD *)(a1 + 104) = 7;
    return 381;
  }
  if ( sub_9691B0(v4 - 1, v14, "callbr", 6) )
  {
    *(_DWORD *)(a1 + 104) = 11;
    return 387;
  }
  if ( sub_9691B0(v4 - 1, v14, "alloca", 6) )
  {
    *(_DWORD *)(a1 + 104) = 31;
    return 388;
  }
  if ( sub_9691B0(v4 - 1, v14, "load", 4) )
  {
    *(_DWORD *)(a1 + 104) = 32;
    return 389;
  }
  if ( sub_9691B0(v4 - 1, v14, "store", 5) )
  {
    *(_DWORD *)(a1 + 104) = 33;
    return 390;
  }
  if ( sub_9691B0(v4 - 1, v14, "cmpxchg", 7) )
  {
    *(_DWORD *)(a1 + 104) = 36;
    return 392;
  }
  if ( sub_9691B0(v4 - 1, v14, "atomicrmw", 9) )
  {
    *(_DWORD *)(a1 + 104) = 37;
    return 393;
  }
  if ( sub_9691B0(v4 - 1, v14, "fence", 5) )
  {
    *(_DWORD *)(a1 + 104) = 35;
    return 391;
  }
  if ( sub_9691B0(v4 - 1, v14, "getelementptr", 13) )
  {
    *(_DWORD *)(a1 + 104) = 34;
    return 394;
  }
  if ( sub_9691B0(v4 - 1, v14, "extractelement", 14) )
  {
    *(_DWORD *)(a1 + 104) = 61;
    return 395;
  }
  if ( sub_9691B0(v4 - 1, v14, "insertelement", 13) )
  {
    *(_DWORD *)(a1 + 104) = 62;
    return 396;
  }
  if ( sub_9691B0(v4 - 1, v14, "shufflevector", 13) )
  {
    *(_DWORD *)(a1 + 104) = 63;
    return 397;
  }
  if ( sub_9691B0(v4 - 1, v14, "extractvalue", 12) )
  {
    *(_DWORD *)(a1 + 104) = 64;
    return 399;
  }
  if ( sub_9691B0(v4 - 1, v14, "insertvalue", 11) )
  {
    *(_DWORD *)(a1 + 104) = 65;
    return 400;
  }
  if ( sub_9691B0(v4 - 1, v14, "landingpad", 10) )
  {
    *(_DWORD *)(a1 + 104) = 66;
    return 370;
  }
  if ( sub_9691B0(v4 - 1, v14, "cleanupret", 10) )
  {
    *(_DWORD *)(a1 + 104) = 8;
    return 382;
  }
  if ( sub_9691B0(v4 - 1, v14, "catchret", 8) )
  {
    *(_DWORD *)(a1 + 104) = 9;
    return 384;
  }
  if ( sub_9691B0(v4 - 1, v14, "catchswitch", 11) )
  {
    *(_DWORD *)(a1 + 104) = 10;
    return 383;
  }
  if ( sub_9691B0(v4 - 1, v14, "catchpad", 8) )
  {
    *(_DWORD *)(a1 + 104) = 52;
    return 385;
  }
  if ( sub_9691B0(v4 - 1, v14, "cleanuppad", 10) )
  {
    *(_DWORD *)(a1 + 104) = 51;
    return 386;
  }
  if ( sub_9691B0(v4 - 1, v14, "freeze", 6) )
  {
    *(_DWORD *)(a1 + 104) = 67;
    return 405;
  }
  if ( sub_11FD080(v4 - 1, v14, "DW_TAG_", 7u) )
  {
    sub_11FD060(a1 + 72, (__int64)(v4 - 1), (__int64)&v13[v14]);
    return 513;
  }
  if ( sub_11FD080(v4 - 1, v14, "DW_ATE_", 7u) )
  {
    sub_11FD060(a1 + 72, (__int64)(v4 - 1), (__int64)&v13[v14]);
    return 514;
  }
  if ( sub_11FD080(v4 - 1, v14, "DW_VIRTUALITY_", 0xEu) )
  {
    sub_11FD060(a1 + 72, (__int64)(v4 - 1), (__int64)&v13[v14]);
    return 515;
  }
  if ( sub_11FD080(v4 - 1, v14, "DW_LANG_", 8u) )
  {
    sub_11FD060(a1 + 72, (__int64)(v4 - 1), (__int64)&v13[v14]);
    return 516;
  }
  if ( sub_11FD080(v4 - 1, v14, "DW_CC_", 6u) )
  {
    sub_11FD060(a1 + 72, (__int64)(v4 - 1), (__int64)&v13[v14]);
    return 517;
  }
  if ( sub_11FD080(v4 - 1, v14, "DW_OP_", 6u) )
  {
    sub_11FD060(a1 + 72, (__int64)(v4 - 1), (__int64)&v13[v14]);
    return 520;
  }
  if ( sub_11FD080(v4 - 1, v14, "DW_MACINFO_", 0xBu) )
  {
    sub_11FD060(a1 + 72, (__int64)(v4 - 1), (__int64)&v13[v14]);
    return 523;
  }
  if ( sub_11FD080(v4 - 1, v14, "DW_APPLE_ENUM_KIND_", 0x13u) )
  {
    sub_11FD060(a1 + 72, (__int64)(v4 - 1), (__int64)&v13[v14]);
    return 526;
  }
  if ( sub_9691B0(v4 - 1, v14, "dbg_value", 9) )
  {
    sub_11FD030(a1 + 72, "value");
    return 525;
  }
  if ( sub_9691B0(v4 - 1, v14, "dbg_declare", 11) )
  {
    sub_11FD030(a1 + 72, "declare");
    return 525;
  }
  if ( sub_9691B0(v4 - 1, v14, "dbg_assign", 10) )
  {
    sub_11FD030(a1 + 72, "assign");
    return 525;
  }
  if ( sub_9691B0(v4 - 1, v14, "dbg_label", 9) )
  {
    sub_11FD030(a1 + 72, "label");
    return 525;
  }
  if ( sub_11FD080(v4 - 1, v14, "DIFlag", 6u) )
  {
    sub_11FD060(a1 + 72, (__int64)(v4 - 1), (__int64)&v13[v14]);
    return 521;
  }
  if ( sub_11FD080(v4 - 1, v14, "DISPFlag", 8u) )
  {
    sub_11FD060(a1 + 72, (__int64)(v4 - 1), (__int64)&v13[v14]);
    return 522;
  }
  if ( sub_11FD080(v4 - 1, v14, "CSK_", 4u) )
  {
    sub_11FD060(a1 + 72, (__int64)(v4 - 1), (__int64)&v13[v14]);
    return 524;
  }
  if ( sub_9691B0(v4 - 1, v14, "NoDebug", 7)
    || sub_9691B0(v4 - 1, v14, "FullDebug", 9)
    || sub_9691B0(v4 - 1, v14, "LineTablesOnly", 14)
    || sub_9691B0(v4 - 1, v14, "DebugDirectivesOnly", 19) )
  {
    sub_11FD060(a1 + 72, (__int64)(v4 - 1), (__int64)&v13[v14]);
    return 518;
  }
  if ( sub_9691B0(v4 - 1, v14, "GNU", 3)
    || sub_9691B0(v4 - 1, v14, "Apple", 5)
    || sub_9691B0(v4 - 1, v14, "None", 4)
    || sub_9691B0(v4 - 1, v14, "Default", 7) )
  {
    sub_11FD060(a1 + 72, (__int64)(v4 - 1), (__int64)&v13[v14]);
    return 519;
  }
  v15 = *(_BYTE **)(a1 + 56);
  if ( ((*v15 - 115) & 0xFD) != 0 || v15[1] != 48 )
  {
    if ( *v15 == 99 && v15[1] == 99 )
    {
      *(_QWORD *)a1 = v15 + 2;
      return 108;
    }
LABEL_525:
    *(_QWORD *)a1 = v15 + 1;
    return 1;
  }
  if ( v15[2] != 120 || !isxdigit((unsigned __int8)v15[3]) )
    goto LABEL_525;
  v18 = *(_QWORD *)a1 - (_DWORD)v15;
  v19 = v15 + 3;
  v20 = v18 - 3;
  v21 = v19;
  v22 = 4 * v20;
  v39 = &v19[v20];
  for ( i = (__int64)v20 >> 2; i > 0; --i )
  {
    if ( !isxdigit(*v21) )
      goto LABEL_611;
    if ( !isxdigit(v21[1]) )
    {
      ++v21;
      goto LABEL_611;
    }
    if ( !isxdigit(v21[2]) )
    {
      v21 += 2;
      goto LABEL_611;
    }
    if ( !isxdigit(v21[3]) )
    {
      v21 += 3;
      goto LABEL_611;
    }
    v21 += 4;
  }
  v23 = v39 - v21;
  if ( v39 - v21 == 2 )
  {
LABEL_636:
    if ( !isxdigit(*v21) )
      goto LABEL_611;
    ++v21;
    goto LABEL_638;
  }
  if ( v23 == 3 )
  {
    if ( !isxdigit(*v21) )
      goto LABEL_611;
    ++v21;
    goto LABEL_636;
  }
  if ( v23 != 1 )
    goto LABEL_616;
LABEL_638:
  if ( isxdigit(*v21) )
    goto LABEL_616;
LABEL_611:
  if ( v39 != v21 )
  {
    *(_QWORD *)a1 = v19;
    return 1;
  }
LABEL_616:
  sub_C47AB0((__int64)&v40, v22, v19, v20, 0x10u);
  v24 = v41;
  if ( v41 > 0x40 )
  {
    v28 = sub_C444A0((__int64)&v40);
  }
  else
  {
    _BitScanReverse64(&v25, (unsigned __int64)v40);
    v26 = v25 ^ 0x3F;
    v27 = 64;
    if ( v40 )
      v27 = v26;
    v28 = v41 + v27 - 64;
  }
  v29 = v24 - v28;
  if ( v24 != v28 && v22 > v29 )
  {
    sub_C44740((__int64)&v44, &v40, v29);
    if ( v41 > 0x40 && v40 )
      j_j___libc_free_0_0(v40);
    v40 = v44;
    v30 = v45;
    v45 = 0;
    v41 = v30;
    sub_969240((__int64 *)&v44);
  }
  v31 = **(_BYTE **)(a1 + 56) == 117;
  v43 = v41;
  if ( v41 > 0x40 )
    sub_C43780((__int64)&v42, (const void **)&v40);
  else
    v42 = v40;
  v32 = *(_DWORD *)(a1 + 152) <= 0x40u;
  v33 = v43;
  v46 = v31;
  v43 = 0;
  v45 = v33;
  v44 = v42;
  if ( !v32 )
  {
    v34 = *(_QWORD *)(a1 + 144);
    if ( v34 )
      j_j___libc_free_0_0(v34);
  }
  *(_QWORD *)(a1 + 144) = v44;
  v35 = v45;
  v45 = 0;
  *(_DWORD *)(a1 + 152) = v35;
  *(_BYTE *)(a1 + 156) = v46;
  sub_969240((__int64 *)&v44);
  sub_969240((__int64 *)&v42);
  sub_969240((__int64 *)&v40);
  return 529;
}
