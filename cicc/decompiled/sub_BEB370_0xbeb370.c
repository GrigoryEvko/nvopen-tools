// Function: sub_BEB370
// Address: 0xbeb370
//
void __fastcall sub_BEB370(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, char a5, char a6)
{
  char v7; // dl
  __int64 *v8; // r14
  __int64 *v9; // rbx
  __int64 v10; // r13
  __int64 v11; // r14
  __int64 v12; // rdi
  _BYTE *v13; // rax
  _BYTE *v14; // rsi
  __int64 *v15; // r13
  __int64 v16; // rbx
  __int64 *v17; // rax
  __int64 v18; // r13
  __int64 v19; // rax
  unsigned __int64 v20; // rbx
  int v21; // r15d
  __int64 v22; // r14
  char v23; // al
  char v24; // al
  char v25; // al
  char v26; // al
  char v27; // al
  __int64 *v28; // rax
  __int64 v29; // rdx
  int v30; // eax
  const char *v31; // rdx
  const char *v32; // rax
  __int64 *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 *v36; // rbx
  __int64 v37; // r13
  int v38; // eax
  __int64 v39; // r15
  __int64 v40; // rax
  int v41; // ebx
  int v42; // ebx
  int v43; // r13d
  int v44; // ebx
  int v45; // ebx
  int v46; // ebx
  int v47; // ebx
  int v48; // r13d
  int v49; // ebx
  int v50; // ebx
  __m128i v51; // rax
  const void *v52; // rax
  size_t v53; // rdx
  size_t v54; // rbx
  const void *v55; // r14
  const void *v56; // rax
  size_t v57; // rdx
  size_t v58; // rbx
  const void *v59; // r14
  const void *v60; // rax
  size_t v61; // rdx
  size_t v62; // rbx
  const void *v63; // r14
  const void *v64; // rax
  size_t v65; // rdx
  size_t v66; // rbx
  const void *v67; // r14
  const void *v68; // rax
  size_t v69; // rdx
  size_t v70; // rbx
  const void *v71; // r14
  __m128i *v72; // rsi
  __m128i *v73; // r14
  __int64 v74; // rdx
  __int64 v75; // rbx
  __int64 v76; // r14
  unsigned __int64 v77; // rdx
  unsigned __int64 v78; // rbx
  char v79; // ah
  __int64 v80; // r14
  unsigned __int64 v81; // rdx
  unsigned __int64 v82; // rbx
  char v83; // ah
  __m128i v84; // rax
  const void *v85; // rax
  size_t v86; // rdx
  size_t v87; // rbx
  const void *v88; // r14
  unsigned int v89; // eax
  unsigned int v90; // ebx
  __int64 v91; // rax
  char v92; // al
  char v93; // r14
  char v94; // bl
  __m128i **v95; // rax
  int v96; // r14d
  int v97; // [rsp+8h] [rbp-1E8h]
  char v98; // [rsp+Dh] [rbp-1E3h]
  char v99; // [rsp+Eh] [rbp-1E2h]
  char v100; // [rsp+Fh] [rbp-1E1h]
  __int64 v101; // [rsp+18h] [rbp-1D8h]
  char v102; // [rsp+18h] [rbp-1D8h]
  char v103; // [rsp+24h] [rbp-1CCh]
  char v105; // [rsp+26h] [rbp-1CAh]
  _BYTE *v107; // [rsp+28h] [rbp-1C8h] BYREF
  __int64 v108; // [rsp+30h] [rbp-1C0h] BYREF
  __int64 v109; // [rsp+38h] [rbp-1B8h] BYREF
  unsigned int v110; // [rsp+44h] [rbp-1ACh]
  __int64 v111; // [rsp+48h] [rbp-1A8h] BYREF
  __int64 v112; // [rsp+50h] [rbp-1A0h] BYREF
  __int64 v113; // [rsp+58h] [rbp-198h] BYREF
  _BYTE v114[32]; // [rsp+60h] [rbp-190h] BYREF
  __m128i v115[2]; // [rsp+80h] [rbp-170h] BYREF
  __m128i v116; // [rsp+A0h] [rbp-150h] BYREF
  __m128i *v117; // [rsp+B0h] [rbp-140h]
  __int64 v118; // [rsp+B8h] [rbp-138h]
  __int16 v119; // [rsp+C0h] [rbp-130h]
  __m128i *v120; // [rsp+D0h] [rbp-120h] BYREF
  __int64 v121; // [rsp+D8h] [rbp-118h]
  __int64 v122; // [rsp+E0h] [rbp-110h]
  size_t v123; // [rsp+E8h] [rbp-108h] BYREF
  __int16 v124; // [rsp+F0h] [rbp-100h]
  __int64 *v125; // [rsp+168h] [rbp-88h]
  __int64 v126; // [rsp+178h] [rbp-78h] BYREF
  __int64 *v127; // [rsp+188h] [rbp-68h]
  __int64 v128; // [rsp+198h] [rbp-58h] BYREF
  char v129; // [rsp+1B0h] [rbp-40h]

  v109 = a2;
  v108 = a3;
  v107 = a4;
  if ( !a3 )
    return;
  sub_AE6EC0(a1 + 1280, a3);
  if ( v7 )
  {
    if ( !(unsigned __int8)sub_A743B0(&v108, *(_QWORD **)(a1 + 144)) )
    {
      v120 = (__m128i *)"Attribute list does not match Module context!";
      v124 = 259;
      sub_BDBF70((__int64 *)a1, (__int64)&v120);
      if ( *(_QWORD *)a1 )
      {
        sub_A76D90(&v108, *(_QWORD *)a1);
        v14 = v107;
        if ( v107 )
        {
LABEL_17:
          sub_BDBD80(a1, v14);
          return;
        }
      }
      return;
    }
    v8 = (__int64 *)sub_A74450(&v108);
    v101 = sub_A74460(&v108);
    if ( v8 != (__int64 *)v101 )
    {
      while ( 1 )
      {
        if ( *v8 && !(unsigned __int8)sub_A731B0((__int64)v8, *(_QWORD **)(a1 + 144)) )
        {
          v120 = (__m128i *)"Attribute set does not match Module context!";
          v124 = 259;
          sub_BDBF70((__int64 *)a1, (__int64)&v120);
          v39 = *(_QWORD *)a1;
          if ( !*(_QWORD *)a1 )
            return;
          sub_A76D00(v116.m128i_i64, v8, 0);
          v40 = sub_CB6200(v39, v116.m128i_i64[0], v116.m128i_i64[1]);
          sub_A51310(v40, 0xAu);
          goto LABEL_15;
        }
        v9 = (__int64 *)sub_A73280(v8);
        v10 = sub_A73290(v8);
        if ( v9 != (__int64 *)v10 )
          break;
LABEL_59:
        if ( (__int64 *)v101 == ++v8 )
          goto LABEL_19;
      }
      while ( (unsigned __int8)sub_A72E40(v9, *(_QWORD **)(a1 + 144)) )
      {
        if ( (__int64 *)v10 == ++v9 )
          goto LABEL_59;
      }
      v120 = (__m128i *)"Attribute does not match Module context!";
      v124 = 259;
      sub_BDBF70((__int64 *)a1, (__int64)&v120);
      v11 = *(_QWORD *)a1;
      if ( !*(_QWORD *)a1 )
        return;
      if ( v9 )
      {
        sub_A759D0((__int64)&v116, v9, 0);
        v12 = sub_CB6200(v11, v116.m128i_i64[0], v116.m128i_i64[1]);
        v13 = *(_BYTE **)(v12 + 32);
        if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 24) )
        {
          sub_CB5D20(v12, 10);
        }
        else
        {
          *(_QWORD *)(v12 + 32) = v13 + 1;
          *v13 = 10;
        }
LABEL_15:
        sub_2240A30(&v116);
      }
      v14 = v107;
      if ( !v107 )
        return;
      goto LABEL_17;
    }
  }
LABEL_19:
  v111 = sub_A74610(&v108);
  v15 = (__int64 *)sub_A73280(&v111);
  v16 = sub_A73290(&v111);
  if ( v15 != (__int64 *)v16 )
  {
    while ( 1 )
    {
      v113 = *v15;
      if ( !sub_A71840((__int64)&v113) )
      {
        v30 = sub_A71AE0(&v113);
        if ( !sub_A71A30(v30) )
          break;
      }
      if ( (__int64 *)v16 == ++v15 )
        goto LABEL_22;
    }
    sub_A759D0((__int64)v114, &v113, 0);
    sub_95D570(v115, "Attribute '", (__int64)v114);
    v31 = "' does not apply to function return values";
    goto LABEL_56;
  }
LABEL_22:
  v110 = 0;
  v17 = *(__int64 **)(v109 + 16);
  v18 = *v17;
  if ( *(_BYTE *)(*v17 + 8) == 17 )
  {
    v33 = (__int64 *)sub_BCAE30(*v17);
    v120 = (__m128i *)v33;
    v121 = v34;
    if ( (_DWORD)v33 )
      v110 = (unsigned int)v33;
  }
  sub_BEA6A0((__int64 *)a1, v111, v18, v107);
  v19 = v109;
  v97 = *(_DWORD *)(v109 + 12) - 1;
  if ( *(_DWORD *)(v109 + 12) != 1 )
  {
    v98 = 0;
    v20 = 0;
    v99 = 0;
    v100 = 0;
    v103 = 0;
    v105 = 0;
    v102 = 0;
    while ( 1 )
    {
      v21 = v20 + 1;
      v22 = *(_QWORD *)(*(_QWORD *)(v19 + 16) + 8 * (v20 + 1));
      v116.m128i_i64[0] = sub_A744E0(&v108, v20);
      if ( !a5 )
      {
        if ( (unsigned __int8)sub_A73170(&v116, 14) )
        {
          HIBYTE(v124) = 1;
          v32 = "immarg attribute only applies to intrinsics";
          goto LABEL_58;
        }
        if ( !a6 && (unsigned __int8)sub_A73170(&v116, 82) )
        {
          HIBYTE(v124) = 1;
          v32 = "Attribute 'elementtype' can only be applied to intrinsics and inline asm.";
          goto LABEL_58;
        }
      }
      sub_BEA6A0((__int64 *)a1, v116.m128i_i64[0], v22, v107);
      if ( *(_BYTE *)(v22 + 8) == 17 )
      {
        v28 = (__int64 *)sub_BCAE30(v22);
        v120 = (__m128i *)v28;
        v121 = v29;
        if ( (unsigned int)v28 > v110 )
          v110 = (unsigned int)v28;
      }
      v23 = sub_A73170(&v116, 21);
      if ( v23 )
      {
        if ( v102 )
        {
          HIBYTE(v124) = 1;
          v32 = "More than one parameter has attribute nest!";
          goto LABEL_58;
        }
        v102 = v23;
      }
      if ( (unsigned __int8)sub_A73170(&v116, 52) )
      {
        if ( v105 )
        {
          HIBYTE(v124) = 1;
          v32 = "More than one parameter has attribute returned!";
          goto LABEL_58;
        }
        v105 = sub_BCAF30(v22, **(_QWORD **)(v109 + 16));
        if ( !v105 )
        {
          HIBYTE(v124) = 1;
          v32 = "Incompatible argument and return types for 'returned' attribute";
          goto LABEL_58;
        }
      }
      v24 = sub_A73170(&v116, 85);
      if ( v24 )
      {
        if ( v103 )
        {
          HIBYTE(v124) = 1;
          v32 = "Cannot have multiple 'sret' parameters!";
          goto LABEL_58;
        }
        if ( v20 > 1 )
        {
          HIBYTE(v124) = 1;
          v32 = "Attribute 'sret' is not on first or second parameter!";
          goto LABEL_58;
        }
        v103 = v24;
      }
      v25 = sub_A73170(&v116, 75);
      if ( v25 )
      {
        if ( v100 )
        {
          HIBYTE(v124) = 1;
          v32 = "Cannot have multiple 'swiftself' parameters!";
          goto LABEL_58;
        }
        v100 = v25;
      }
      v26 = sub_A73170(&v116, 73);
      if ( v26 )
      {
        if ( v99 )
        {
          HIBYTE(v124) = 1;
          v32 = "Cannot have multiple 'swiftasync' parameters!";
          goto LABEL_58;
        }
        v99 = v26;
      }
      v27 = sub_A73170(&v116, 74);
      if ( v27 )
      {
        if ( v98 )
        {
          HIBYTE(v124) = 1;
          v32 = "Cannot have multiple 'swifterror' parameters!";
          goto LABEL_58;
        }
        v98 = v27;
      }
      if ( (unsigned __int8)sub_A73170(&v116, 83) && *(_DWORD *)(v109 + 12) - 2 != (_DWORD)v20 )
        break;
      ++v20;
      if ( v97 == v21 )
        goto LABEL_69;
      v19 = v109;
    }
    HIBYTE(v124) = 1;
    v32 = "inalloca isn't on the last parameter!";
    goto LABEL_58;
  }
LABEL_69:
  if ( sub_A74740(&v108, -1) )
  {
    v35 = sub_A74680(&v108);
    sub_BE7B90((__int64 *)a1, v35, v107);
    v113 = sub_A74680(&v108);
    v36 = (__int64 *)sub_A73280(&v113);
    v37 = sub_A73290(&v113);
    if ( v36 != (__int64 *)v37 )
    {
      while ( 1 )
      {
        v112 = *v36;
        if ( !sub_A71840((__int64)&v112) )
        {
          v38 = sub_A71AE0(&v112);
          if ( !(unsigned __int8)sub_A719F0(v38) )
            break;
        }
        if ( (__int64 *)v37 == ++v36 )
          goto LABEL_86;
      }
      sub_A759D0((__int64)v114, &v112, 0);
      sub_95D570(v115, "Attribute '", (__int64)v114);
      v31 = "' does not apply to functions!";
LABEL_56:
      sub_94F930(&v116, (__int64)v115, v31);
      v124 = 260;
      v120 = &v116;
      sub_BE7760((_BYTE *)a1, (__int64)&v120, &v107);
      sub_2240A30(&v116);
      sub_2240A30(v115);
      sub_2240A30(v114);
      return;
    }
LABEL_86:
    if ( (unsigned __int8)sub_A73ED0(&v108, 31) && (unsigned __int8)sub_A73ED0(&v108, 3) )
    {
      HIBYTE(v124) = 1;
      v32 = "Attributes 'noinline and alwaysinline' are incompatible!";
      goto LABEL_58;
    }
    if ( (unsigned __int8)sub_A73ED0(&v108, 48) )
    {
      if ( !(unsigned __int8)sub_A73ED0(&v108, 31) )
      {
        HIBYTE(v124) = 1;
        v32 = "Attribute 'optnone' requires 'noinline'!";
        goto LABEL_58;
      }
      if ( (unsigned __int8)sub_A73ED0(&v108, 47) )
      {
        HIBYTE(v124) = 1;
        v32 = "Attributes 'optsize and optnone' are incompatible!";
        goto LABEL_58;
      }
      if ( (unsigned __int8)sub_A73ED0(&v108, 18) )
      {
        HIBYTE(v124) = 1;
        v32 = "Attributes 'minsize and optnone' are incompatible!";
        goto LABEL_58;
      }
      if ( (unsigned __int8)sub_A73ED0(&v108, 46) )
      {
        HIBYTE(v124) = 1;
        v32 = "Attributes 'optdebug and optnone' are incompatible!";
        goto LABEL_58;
      }
    }
    if ( (unsigned __int8)sub_A73ED0(&v108, 61) && (unsigned __int8)sub_A73ED0(&v108, 62) )
    {
      HIBYTE(v124) = 1;
      v32 = "Attributes 'sanitize_realtime and sanitize_realtime_blocking' are incompatible!";
      goto LABEL_58;
    }
    if ( (unsigned __int8)sub_A73ED0(&v108, 46) )
    {
      if ( (unsigned __int8)sub_A73ED0(&v108, 47) )
      {
        HIBYTE(v124) = 1;
        v32 = "Attributes 'optsize and optdebug' are incompatible!";
        goto LABEL_58;
      }
      if ( (unsigned __int8)sub_A73ED0(&v108, 18) )
      {
        HIBYTE(v124) = 1;
        v32 = "Attributes 'minsize and optdebug' are incompatible!";
        goto LABEL_58;
      }
    }
    if ( (unsigned __int8)sub_A74390(&v108, 77, 0) && (sub_A746F0(&v108) & 2) == 0 )
    {
      HIBYTE(v124) = 1;
      v32 = "Attribute writable and memory without argmem: write are incompatible!";
    }
    else if ( (unsigned __int8)sub_A747A0(&v108, "aarch64_pstate_sm_enabled", 0x19u)
           && (unsigned __int8)sub_A747A0(&v108, "aarch64_pstate_sm_compatible", 0x1Cu) )
    {
      HIBYTE(v124) = 1;
      v32 = "Attributes 'aarch64_pstate_sm_enabled and aarch64_pstate_sm_compatible' are incompatible!";
    }
    else
    {
      v41 = (unsigned __int8)sub_A747A0(&v108, "aarch64_new_za", 0xEu);
      v42 = (unsigned __int8)sub_A747A0(&v108, "aarch64_in_za", 0xDu) + v41;
      v43 = v42 + (unsigned __int8)sub_A747A0(&v108, "aarch64_inout_za", 0x10u);
      v44 = (unsigned __int8)sub_A747A0(&v108, "aarch64_out_za", 0xEu);
      v45 = (unsigned __int8)sub_A747A0(&v108, "aarch64_preserves_za", 0x14u) + v43 + v44;
      if ( v45 + (unsigned __int8)sub_A747A0(&v108, "aarch64_za_state_agnostic", 0x19u) > 1 )
      {
        HIBYTE(v124) = 1;
        v32 = "Attributes 'aarch64_new_za', 'aarch64_in_za', 'aarch64_out_za', 'aarch64_inout_za', 'aarch64_preserves_za'"
              " and 'aarch64_za_state_agnostic' are mutually exclusive";
      }
      else
      {
        v46 = (unsigned __int8)sub_A747A0(&v108, "aarch64_new_zt0", 0xFu);
        v47 = (unsigned __int8)sub_A747A0(&v108, "aarch64_in_zt0", 0xEu) + v46;
        v48 = v47 + (unsigned __int8)sub_A747A0(&v108, "aarch64_inout_zt0", 0x11u);
        v49 = (unsigned __int8)sub_A747A0(&v108, "aarch64_out_zt0", 0xFu);
        v50 = (unsigned __int8)sub_A747A0(&v108, "aarch64_preserves_zt0", 0x15u) + v48 + v49;
        if ( v50 + (unsigned __int8)sub_A747A0(&v108, "aarch64_za_state_agnostic", 0x19u) > 1 )
        {
          HIBYTE(v124) = 1;
          v32 = "Attributes 'aarch64_new_zt0', 'aarch64_in_zt0', 'aarch64_out_zt0', 'aarch64_inout_zt0', 'aarch64_preserv"
                "es_zt0' and 'aarch64_za_state_agnostic' are mutually exclusive";
        }
        else
        {
          if ( !(unsigned __int8)sub_A73ED0(&v108, 17) || v107[32] >> 6 == 2 )
          {
            v120 = (__m128i *)sub_A74680(&v108);
            v51.m128i_i64[0] = sub_A738C0((__int64 *)&v120);
            v116 = v51;
            if ( v51.m128i_i8[12] )
            {
              v120 = (__m128i *)&v109;
              v121 = (__int64)&v107;
              v122 = a1;
              if ( !(unsigned __int8)sub_BE7860((__int64 **)&v120, (__int64)"element size", 12, v116.m128i_u32[0])
                || v116.m128i_i8[8]
                && !(unsigned __int8)sub_BE7860((__int64 **)&v120, (__int64)"number of elements", 18, v116.m128i_u32[1]) )
              {
                return;
              }
            }
            if ( (unsigned __int8)sub_A73ED0(&v108, 87) )
            {
              v92 = sub_A746D0(&v108);
              v120 = (__m128i *)1;
              v121 = 2;
              v93 = v92;
              v94 = v92;
              v95 = &v120;
              v122 = 4;
              v96 = v93 & 7;
              while ( *v95 != (__m128i *)(unsigned int)v96 )
              {
                if ( ++v95 == (__m128i **)&v123 )
                {
                  v120 = (__m128i *)"'allockind()' requires exactly one of alloc, realloc, and free";
                  v124 = 259;
                  sub_BDBF70((__int64 *)a1, (__int64)&v120);
                  break;
                }
              }
              if ( v96 == 4 && (v94 & 0x38) != 0 )
              {
                v120 = (__m128i *)"'allockind(\"free\")' doesn't allow uninitialized, zeroed, or aligned modifiers.";
                v124 = 259;
                sub_BDBF70((__int64 *)a1, (__int64)&v120);
              }
              if ( (v94 & 0x18) == 0x18 )
              {
                v120 = (__m128i *)"'allockind()' can't be both zeroed and uninitialized";
                v124 = 259;
                sub_BDBF70((__int64 *)a1, (__int64)&v120);
              }
            }
            if ( !(unsigned __int8)sub_A73ED0(&v108, 96) )
              goto LABEL_111;
            v120 = (__m128i *)sub_A74680(&v108);
            v89 = sub_A73930((__int64 *)&v120);
            v90 = v89;
            if ( v89 )
            {
              if ( (v89 & (v89 - 1)) != 0 )
              {
                v120 = (__m128i *)"'vscale_range' minimum must be power-of-two value";
                v124 = 259;
                sub_BE7760((_BYTE *)a1, (__int64)&v120, &v107);
              }
              v120 = (__m128i *)sub_A74680(&v108);
              v91 = sub_A739A0((__int64 *)&v120);
              v116.m128i_i64[0] = v91;
              if ( !BYTE4(v91) )
                goto LABEL_111;
              if ( v90 > (unsigned int)v91 )
              {
                v120 = (__m128i *)"'vscale_range' minimum cannot be greater than maximum";
                v124 = 259;
                sub_BE7760((_BYTE *)a1, (__int64)&v120, &v107);
                goto LABEL_111;
              }
            }
            else
            {
              v120 = (__m128i *)"'vscale_range' minimum must be greater than 0";
              v124 = 259;
              sub_BE7760((_BYTE *)a1, (__int64)&v120, &v107);
              v120 = (__m128i *)sub_A74680(&v108);
              v91 = sub_A739A0((__int64 *)&v120);
              v116.m128i_i64[0] = v91;
              if ( !BYTE4(v91) )
                goto LABEL_111;
              if ( !(_DWORD)v91 )
                goto LABEL_191;
            }
            if ( ((unsigned int)v91 & ((_DWORD)v91 - 1)) == 0 )
            {
LABEL_111:
              if ( (unsigned __int8)sub_A747A0(&v108, "frame-pointer", 0xDu) )
              {
                v120 = (__m128i *)sub_A747B0(&v108, -1, "frame-pointer", 0xDu);
                v85 = (const void *)sub_A72240((__int64 *)&v120);
                v87 = v86;
                v88 = v85;
                if ( !sub_9691B0(v85, v86, "all", 3)
                  && !sub_9691B0(v88, v87, "non-leaf", 8)
                  && !sub_9691B0(v88, v87, "none", 4)
                  && !sub_9691B0(v88, v87, "reserved", 8) )
                {
                  v124 = 1283;
                  v120 = (__m128i *)"invalid value for 'frame-pointer' attribute: ";
                  v122 = (__int64)v88;
                  v123 = v87;
                  sub_BE7760((_BYTE *)a1, (__int64)&v120, &v107);
                }
              }
              if ( v110 <= 0x1FF
                || !(unsigned __int8)sub_A747A0(&v108, "target-features", 0xFu)
                || (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 128) + 32LL) - 38) > 1
                || (v120 = (__m128i *)sub_A747B0(&v108, -1, "target-features", 0xFu),
                    v84.m128i_i64[0] = sub_A72240((__int64 *)&v120),
                    v116 = v84,
                    sub_C931B0(&v116, "+avx512f", 8, 0) == -1)
                || sub_C931B0(&v116, "-evex512", 8, 0) == -1 )
              {
                sub_BE7A40((_BYTE *)a1, v108, "patchable-function-prefix", 0x19u, v107);
                sub_BE7A40((_BYTE *)a1, v108, "patchable-function-entry", 0x18u, v107);
                sub_BE7A40((_BYTE *)a1, v108, "warn-stack-size", 0xFu, v107);
                v116.m128i_i64[0] = sub_A747B0(&v108, -1, "sign-return-address", 0x13u);
                if ( v116.m128i_i64[0] )
                {
                  v52 = (const void *)sub_A72240(v116.m128i_i64);
                  v54 = v53;
                  v55 = v52;
                  if ( !sub_9691B0(v52, v53, "none", 4)
                    && !sub_9691B0(v55, v54, "all", 3)
                    && !sub_9691B0(v55, v54, "non-leaf", 8) )
                  {
                    v124 = 1283;
                    v120 = (__m128i *)"invalid value for 'sign-return-address' attribute: ";
                    v122 = (__int64)v55;
                    v123 = v54;
                    sub_BE7760((_BYTE *)a1, (__int64)&v120, &v107);
                  }
                }
                v116.m128i_i64[0] = sub_A747B0(&v108, -1, "sign-return-address-key", 0x17u);
                if ( v116.m128i_i64[0] )
                {
                  v56 = (const void *)sub_A72240(v116.m128i_i64);
                  v58 = v57;
                  v59 = v56;
                  if ( !sub_9691B0(v56, v57, "a_key", 5) && !sub_9691B0(v59, v58, "b_key", 5) )
                  {
                    v124 = 1283;
                    v120 = (__m128i *)"invalid value for 'sign-return-address-key' attribute: ";
                    v122 = (__int64)v59;
                    v123 = v58;
                    sub_BE7760((_BYTE *)a1, (__int64)&v120, &v107);
                  }
                  if ( !sub_A747B0(&v108, -1, "sign-return-address", 0x13u) )
                  {
                    v120 = (__m128i *)"'sign-return-address-key' present without `sign-return-address`";
                    v124 = 259;
                    sub_BDBF70((__int64 *)a1, (__int64)&v120);
                  }
                }
                v116.m128i_i64[0] = sub_A747B0(&v108, -1, "branch-target-enforcement", 0x19u);
                if ( v116.m128i_i64[0] )
                {
                  v60 = (const void *)sub_A72240(v116.m128i_i64);
                  v62 = v61;
                  v63 = v60;
                  if ( !sub_9691B0(v60, v61, byte_3F871B3, 0)
                    && !sub_9691B0(v63, v62, "true", 4)
                    && !sub_9691B0(v63, v62, "false", 5) )
                  {
                    v124 = 1283;
                    v120 = (__m128i *)"invalid value for 'branch-target-enforcement' attribute: ";
                    v122 = (__int64)v63;
                    v123 = v62;
                    sub_BE7760((_BYTE *)a1, (__int64)&v120, &v107);
                  }
                }
                v116.m128i_i64[0] = sub_A747B0(&v108, -1, "branch-protection-pauth-lr", 0x1Au);
                if ( v116.m128i_i64[0] )
                {
                  v64 = (const void *)sub_A72240(v116.m128i_i64);
                  v66 = v65;
                  v67 = v64;
                  if ( !sub_9691B0(v64, v65, byte_3F871B3, 0)
                    && !sub_9691B0(v67, v66, "true", 4)
                    && !sub_9691B0(v67, v66, "false", 5) )
                  {
                    v124 = 1283;
                    v120 = (__m128i *)"invalid value for 'branch-protection-pauth-lr' attribute: ";
                    v122 = (__int64)v67;
                    v123 = v66;
                    sub_BE7760((_BYTE *)a1, (__int64)&v120, &v107);
                  }
                }
                v116.m128i_i64[0] = sub_A747B0(&v108, -1, "guarded-control-stack", 0x15u);
                if ( v116.m128i_i64[0] )
                {
                  v68 = (const void *)sub_A72240(v116.m128i_i64);
                  v70 = v69;
                  v71 = v68;
                  if ( !sub_9691B0(v68, v69, byte_3F871B3, 0)
                    && !sub_9691B0(v71, v70, "true", 4)
                    && !sub_9691B0(v71, v70, "false", 5) )
                  {
                    v124 = 1283;
                    v120 = (__m128i *)"invalid value for 'guarded-control-stack' attribute: ";
                    v122 = (__int64)v71;
                    v123 = v70;
                    sub_BE7760((_BYTE *)a1, (__int64)&v120, &v107);
                  }
                }
                v115[0].m128i_i64[0] = sub_A747B0(&v108, -1, "vector-function-abi-variant", 0x1Bu);
                if ( v115[0].m128i_i64[0] )
                {
                  v72 = (__m128i *)sub_A72240(v115[0].m128i_i64);
                  v73 = v72;
                  v75 = v74;
                  sub_C0A940(&v120, v72, v74, v109);
                  if ( v129
                    || (v119 = 1283,
                        v72 = &v116,
                        v116.m128i_i64[0] = (__int64)"invalid name for a VFABI variant: ",
                        v117 = v73,
                        v118 = v75,
                        sub_BE7760((_BYTE *)a1, (__int64)&v116, &v107),
                        v129) )
                  {
                    v129 = 0;
                    if ( v127 != &v128 )
                    {
                      v72 = (__m128i *)(v128 + 1);
                      j_j___libc_free_0(v127, v128 + 1);
                    }
                    if ( v125 != &v126 )
                    {
                      v72 = (__m128i *)(v126 + 1);
                      j_j___libc_free_0(v125, v126 + 1);
                    }
                    if ( (size_t *)v121 != &v123 )
                      _libc_free(v121, v72);
                  }
                }
                v116.m128i_i64[0] = sub_A747B0(&v108, -1, "denormal-fp-math", 0x10u);
                if ( v116.m128i_i64[0] )
                {
                  v76 = sub_A72240(v116.m128i_i64);
                  v78 = v77;
                  if ( sub_BDB940(v76, v77) == -1 || v79 == -1 )
                  {
                    v124 = 1283;
                    v120 = (__m128i *)"invalid value for 'denormal-fp-math' attribute: ";
                    v122 = v76;
                    v123 = v78;
                    sub_BE7760((_BYTE *)a1, (__int64)&v120, &v107);
                  }
                }
                v116.m128i_i64[0] = sub_A747B0(&v108, -1, "denormal-fp-math-f32", 0x14u);
                if ( v116.m128i_i64[0] )
                {
                  v80 = sub_A72240(v116.m128i_i64);
                  v82 = v81;
                  if ( sub_BDB940(v80, v81) == -1 || v83 == -1 )
                  {
                    v124 = 1283;
                    v120 = (__m128i *)"invalid value for 'denormal-fp-math-f32' attribute: ";
                    v122 = v80;
                    v123 = v82;
                    sub_BE7760((_BYTE *)a1, (__int64)&v120, &v107);
                  }
                }
              }
              else
              {
                v120 = (__m128i *)"512-bit vector arguments require 'evex512' for AVX512";
                v124 = 259;
                sub_BE7760((_BYTE *)a1, (__int64)&v120, &v107);
              }
              return;
            }
LABEL_191:
            v120 = (__m128i *)"'vscale_range' maximum must be power-of-two value";
            v124 = 259;
            sub_BE7760((_BYTE *)a1, (__int64)&v120, &v107);
            goto LABEL_111;
          }
          HIBYTE(v124) = 1;
          v32 = "Attribute 'jumptable' requires 'unnamed_addr'";
        }
      }
    }
LABEL_58:
    v120 = (__m128i *)v32;
    LOBYTE(v124) = 3;
    sub_BE7760((_BYTE *)a1, (__int64)&v120, &v107);
  }
}
