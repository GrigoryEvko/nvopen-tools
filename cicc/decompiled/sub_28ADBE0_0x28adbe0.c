// Function: sub_28ADBE0
// Address: 0x28adbe0
//
__int64 __fastcall sub_28ADBE0(__int64 a1, __int64 a2, __int64 a3, _QWORD **a4)
{
  unsigned __int8 *v7; // r14
  __int64 v8; // rdi
  unsigned int v9; // r14d
  unsigned int v10; // r12d
  __int64 v12; // r15
  unsigned __int8 *v13; // r14
  unsigned __int8 *v14; // r14
  unsigned __int8 *v15; // rax
  unsigned int *v16; // rax
  __int64 v17; // r14
  unsigned int *v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  _QWORD *v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rsi
  __m128i v24; // xmm0
  __m128i v25; // xmm1
  __m128i v26; // xmm2
  __m128i v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  _QWORD *v30; // rax
  unsigned __int8 *v31; // rax
  _QWORD *v32; // rdi
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rdi
  int v38; // r11d
  __int64 v39; // rcx
  int v40; // r11d
  unsigned int v41; // edx
  __int64 *v42; // rax
  __int64 v43; // r8
  __int64 v44; // r8
  unsigned int v45; // edx
  __int64 *v46; // rax
  __int64 v47; // rsi
  __int64 v48; // rdx
  __int64 v49; // r11
  unsigned __int64 v50; // rax
  unsigned __int8 *v51; // rax
  _QWORD *v52; // rdi
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __m128i v57; // xmm3
  _QWORD *v58; // rdi
  __m128i v59; // xmm4
  __m128i v60; // xmm5
  bool v61; // zf
  __int64 v62; // rax
  __int64 v63; // r12
  __int64 v64; // rax
  unsigned int v65; // ebx
  bool v66; // bl
  __int64 v67; // r14
  unsigned int v68; // eax
  unsigned int v69; // r12d
  unsigned __int8 *v70; // rax
  unsigned int v71; // ecx
  __int64 v72; // r12
  __int64 v73; // rax
  __int64 v74; // rcx
  int v75; // eax
  int v76; // esi
  unsigned int v77; // edx
  __int64 *v78; // rax
  __int64 v79; // rdi
  __int64 v80; // rcx
  __int64 v81; // rsi
  __int64 *v82; // rax
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 v86; // r9
  __int64 v87; // r12
  __int64 v88; // rdx
  unsigned int v89; // ebx
  __int64 v90; // rdi
  bool v91; // bl
  __int64 v92; // r14
  unsigned int v93; // eax
  unsigned int v94; // r12d
  unsigned __int8 *v95; // rax
  unsigned int v96; // ecx
  int v97; // eax
  int v98; // eax
  bool v99; // bl
  __int64 v100; // r14
  unsigned int v101; // eax
  unsigned int v102; // r12d
  unsigned __int8 *v103; // rax
  unsigned int v104; // ecx
  int v105; // eax
  int v106; // r8d
  int v107; // esi
  __int16 v108; // [rsp+Ah] [rbp-1A6h]
  __int16 v109; // [rsp+Dh] [rbp-1A3h]
  __int64 v110; // [rsp+10h] [rbp-1A0h]
  _QWORD *v111; // [rsp+20h] [rbp-190h]
  unsigned __int8 *v112; // [rsp+30h] [rbp-180h]
  _BYTE *v114; // [rsp+48h] [rbp-168h] BYREF
  __int64 v115; // [rsp+50h] [rbp-160h]
  __int64 v116; // [rsp+58h] [rbp-158h]
  __m128i v117; // [rsp+60h] [rbp-150h]
  __m128i v118; // [rsp+70h] [rbp-140h]
  __m128i v119; // [rsp+80h] [rbp-130h] BYREF
  __m128i v120; // [rsp+90h] [rbp-120h] BYREF
  __m128i v121; // [rsp+A0h] [rbp-110h] BYREF
  __m128i v122; // [rsp+B0h] [rbp-100h] BYREF
  __m128i v123; // [rsp+C0h] [rbp-F0h]
  __m128i v124; // [rsp+D0h] [rbp-E0h]
  char v125; // [rsp+E0h] [rbp-D0h]
  unsigned int *v126[2]; // [rsp+F0h] [rbp-C0h] BYREF
  char v127; // [rsp+100h] [rbp-B0h] BYREF
  _QWORD *v128; // [rsp+138h] [rbp-78h]
  void *v129; // [rsp+170h] [rbp-40h]

  v7 = sub_BD3990(*(unsigned __int8 **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), a2);
  if ( v7 == sub_BD3990(*(unsigned __int8 **)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))), a2) )
    return 0;
  v8 = *(_QWORD *)(a3 + 32 * (3LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)));
  v9 = *(_DWORD *)(v8 + 32);
  if ( v9 <= 0x40 )
  {
    if ( *(_QWORD *)(v8 + 24) )
      return 0;
  }
  else if ( v9 != (unsigned int)sub_C444A0(v8 + 24) )
  {
    return 0;
  }
  v12 = sub_B43CA0(a2) + 312;
  v13 = sub_BD3990(*(unsigned __int8 **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), a2);
  if ( v13 == sub_BD3990(*(unsigned __int8 **)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)), a2) )
    goto LABEL_22;
  v14 = sub_BD3990(*(unsigned __int8 **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), a2);
  v15 = sub_BD3990(*(unsigned __int8 **)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)), a2);
  v16 = (unsigned int *)sub_BD58A0((__int64)v14, (__int64)v15, v12);
  v126[0] = v16;
  v17 = (__int64)v16;
  v126[1] = v18;
  if ( (_BYTE)v18 != 1 || (__int64)v16 < 0 )
    return 0;
  if ( !v16 )
  {
LABEL_22:
    v17 = 0;
    v19 = *(_QWORD *)(a3 + 32 * (2LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)));
    v20 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    if ( v19 == v20 )
      goto LABEL_18;
    if ( *(_BYTE *)v19 != 17 )
      return 0;
  }
  else
  {
    v19 = *(_QWORD *)(a3 + 32 * (2LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)));
    v20 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    if ( *(_BYTE *)v19 != 17 )
      return 0;
  }
  if ( *(_BYTE *)v20 != 17 )
    return 0;
  v21 = *(_QWORD **)(v19 + 24);
  if ( *(_DWORD *)(v19 + 32) > 0x40u )
    v21 = (_QWORD *)*v21;
  v22 = *(_DWORD *)(v20 + 32) <= 0x40u ? *(_QWORD *)(v20 + 24) : **(_QWORD **)(v20 + 24);
  if ( v17 + v22 > (unsigned __int64)v21 )
    return 0;
LABEL_18:
  sub_23D0AB0((__int64)v126, a2, 0, 0, 0);
  v112 = sub_BD3990(*(unsigned __int8 **)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))), a2);
  v108 = sub_A74840((_QWORD *)(a3 + 72), 1);
  v109 = v108;
  sub_D671D0(&v119, a3);
  v23 = a2;
  v110 = v119.m128i_i64[0];
  sub_D671D0(&v122, a2);
  v24 = _mm_loadu_si128(&v119);
  v25 = _mm_loadu_si128(&v120);
  v111 = 0;
  v26 = _mm_loadu_si128(&v121);
  v115 = v24.m128i_i64[0];
  v117 = v25;
  v116 = v122.m128i_i64[1];
  v118 = v26;
  if ( v17 )
  {
    v23 = *(_QWORD *)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)));
    v27.m128i_i64[0] = sub_BD58A0(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), v23, v12);
    v119 = v27;
    if ( v27.m128i_i8[8] && v119.m128i_i64[0] == v17 )
    {
      v112 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), v23);
    }
    else
    {
      v124.m128i_i16[0] = 257;
      v28 = sub_BCB2E0(v128);
      v114 = (_BYTE *)sub_ACD640(v28, v17, 0);
      v29 = sub_BCB2B0(v128);
      v23 = sub_921130(v126, v29, (__int64)v112, &v114, 1, (__int64)&v122, 3u);
      v112 = (unsigned __int8 *)v23;
      v30 = 0;
      if ( *(_BYTE *)v23 >= 0x1Du )
        v30 = (_QWORD *)v23;
      v111 = v30;
    }
    if ( HIBYTE(v108) )
    {
      LOBYTE(v109) = -1;
      v49 = v17 | (1LL << v108);
      if ( (v49 & -v49) != 0 )
      {
        _BitScanReverse64(&v50, v49 & -(v17 | (1LL << v108)));
        LODWORD(v50) = v50 ^ 0x3F;
        v23 = (unsigned int)(63 - v50);
        LOBYTE(v109) = 63 - v50;
      }
      HIBYTE(v109) = 1;
    }
    v110 = (__int64)v112;
  }
  v31 = sub_BD3990(*(unsigned __int8 **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), v23);
  v32 = *a4;
  v122.m128i_i64[1] = 1;
  v123 = 0u;
  v122.m128i_i64[0] = (__int64)v112;
  v124 = 0u;
  v119.m128i_i64[0] = (__int64)v31;
  v119.m128i_i64[1] = 1;
  v120 = 0u;
  v121 = 0u;
  if ( (unsigned __int8)sub_CF4D50((__int64)v32, (__int64)&v119, (__int64)&v122, (__int64)(a4 + 1), 0) == 3 )
    goto LABEL_37;
  v37 = *(_QWORD *)(a1 + 40);
  v38 = *(_DWORD *)(v37 + 56);
  v39 = *(_QWORD *)(v37 + 40);
  if ( v38 )
  {
    v40 = v38 - 1;
    v41 = v40 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v42 = (__int64 *)(v39 + 16LL * v41);
    v43 = *v42;
    if ( *v42 == a2 )
    {
LABEL_33:
      v44 = v42[1];
    }
    else
    {
      v98 = 1;
      while ( v43 != -4096 )
      {
        v107 = v98 + 1;
        v41 = v40 & (v98 + v41);
        v42 = (__int64 *)(v39 + 16LL * v41);
        v43 = *v42;
        if ( a2 == *v42 )
          goto LABEL_33;
        v98 = v107;
      }
      v44 = 0;
    }
    v45 = v40 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v46 = (__int64 *)(v39 + 16LL * v45);
    v47 = *v46;
    if ( a3 == *v46 )
    {
LABEL_35:
      v48 = v46[1];
    }
    else
    {
      v97 = 1;
      while ( v47 != -4096 )
      {
        v36 = (unsigned int)(v97 + 1);
        v45 = v40 & (v97 + v45);
        v46 = (__int64 *)(v39 + 16LL * v45);
        v47 = *v46;
        if ( a3 == *v46 )
          goto LABEL_35;
        v97 = v36;
      }
      v48 = 0;
    }
  }
  else
  {
    v44 = 0;
    v48 = 0;
  }
  v115 = v110;
  if ( sub_28A97D0((_QWORD *)v37, a4, v48, v44, v44, v36, v24, v25, v26) )
    goto LABEL_37;
  v51 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), (__int64)a4);
  v52 = *a4;
  v122.m128i_i64[1] = 1;
  v123 = 0u;
  v122.m128i_i64[0] = (__int64)v112;
  v124 = 0u;
  v119.m128i_i64[0] = (__int64)v51;
  v119.m128i_i64[1] = 1;
  v120 = 0u;
  v121 = 0u;
  if ( (unsigned __int8)sub_CF4D50((__int64)v52, (__int64)&v119, (__int64)&v122, (__int64)(a4 + 1), 0) != 3 )
  {
    sub_D671D0(&v119, a3);
    v57 = _mm_loadu_si128(&v119);
    v58 = *a4;
    v59 = _mm_loadu_si128(&v120);
    v60 = _mm_loadu_si128(&v121);
    v125 = 1;
    v122 = v57;
    v123 = v59;
    v124 = v60;
    v61 = (sub_CF63E0(v58, (unsigned __int8 *)a2, &v122, (__int64)(a4 + 1)) & 2) == 0;
    v62 = *(_QWORD *)(a2 - 32);
    if ( v61 )
    {
      if ( !v62 || *(_BYTE *)v62 || *(_QWORD *)(v62 + 24) != *(_QWORD *)(a2 + 80) )
        BUG();
      v87 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      v88 = *(_QWORD *)(a2 + 32 * (3 - v87));
      v89 = *(_DWORD *)(v88 + 32);
      v90 = v88 + 24;
      if ( *(_DWORD *)(v62 + 36) == 240 )
      {
        if ( v89 <= 0x40 )
          v99 = *(_QWORD *)(v88 + 24) == 0;
        else
          v99 = (unsigned int)sub_C444A0(v90) == v89;
        v100 = *(_QWORD *)(a2 + 32 * (2 - v87));
        LOWORD(v101) = sub_A74840((_QWORD *)(a2 + 72), 0);
        v102 = v101;
        v103 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), 0);
        v104 = (unsigned __int8)v109;
        BYTE1(v104) = HIBYTE(v109);
        v72 = sub_B343C0((__int64)v126, 0xF0u, (__int64)v103, v102, (__int64)v112, v104, v100, !v99, 0, 0, 0, 0);
      }
      else
      {
        if ( v89 <= 0x40 )
          v91 = *(_QWORD *)(v88 + 24) == 0;
        else
          v91 = (unsigned int)sub_C444A0(v90) == v89;
        v92 = *(_QWORD *)(a2 + 32 * (2 - v87));
        LOWORD(v93) = sub_A74840((_QWORD *)(a2 + 72), 0);
        v94 = v93;
        v95 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), 0);
        v96 = (unsigned __int8)v109;
        BYTE1(v96) = HIBYTE(v109);
        v72 = sub_B343C0((__int64)v126, 0xEEu, (__int64)v95, v94, (__int64)v112, v96, v92, !v91, 0, 0, 0, 0);
      }
    }
    else
    {
      if ( !v62 || *(_BYTE *)v62 || *(_QWORD *)(v62 + 24) != *(_QWORD *)(a2 + 80) )
        BUG();
      if ( *(_DWORD *)(v62 + 36) == 240 )
      {
LABEL_37:
        v10 = 0;
        goto LABEL_38;
      }
      v63 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      v64 = *(_QWORD *)(a2 + 32 * (3 - v63));
      v65 = *(_DWORD *)(v64 + 32);
      if ( v65 <= 0x40 )
        v66 = *(_QWORD *)(v64 + 24) == 0;
      else
        v66 = v65 == (unsigned int)sub_C444A0(v64 + 24);
      v67 = *(_QWORD *)(a2 + 32 * (2 - v63));
      LOWORD(v68) = sub_A74840((_QWORD *)(a2 + 72), 0);
      v69 = v68;
      v70 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), 0);
      v71 = (unsigned __int8)v109;
      BYTE1(v71) = HIBYTE(v109);
      v72 = sub_B343C0((__int64)v126, 0xF1u, (__int64)v70, v69, (__int64)v112, v71, v67, !v66, 0, 0, 0, 0);
    }
    v122.m128i_i32[0] = 38;
    sub_B47C00(v72, a2, v122.m128i_i32, 1);
    v73 = *(_QWORD *)(a1 + 40);
    v74 = *(_QWORD *)(v73 + 40);
    v75 = *(_DWORD *)(v73 + 56);
    if ( v75 )
    {
      v76 = v75 - 1;
      v77 = (v75 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v78 = (__int64 *)(v74 + 16LL * v77);
      v79 = *v78;
      if ( a2 == *v78 )
      {
LABEL_57:
        v80 = v78[1];
LABEL_58:
        v81 = v72;
        v10 = 1;
        v82 = (__int64 *)sub_D69570(*(_QWORD **)(a1 + 48), v81, 0, v80);
        sub_D75120(*(__int64 **)(a1 + 48), v82, 1);
        sub_28AAD10(a1, (_QWORD *)a2, v83, v84, v85, v86);
        goto LABEL_38;
      }
      v105 = 1;
      while ( v79 != -4096 )
      {
        v106 = v105 + 1;
        v77 = v76 & (v105 + v77);
        v78 = (__int64 *)(v74 + 16LL * v77);
        v79 = *v78;
        if ( a2 == *v78 )
          goto LABEL_57;
        v105 = v106;
      }
    }
    v80 = 0;
    goto LABEL_58;
  }
  v10 = 1;
  sub_28AAD10(a1, (_QWORD *)a2, v53, v54, v55, v56);
LABEL_38:
  if ( v111 && !v111[2] )
    sub_28AAD10(a1, v111, v33, v34, v35, v36);
  nullsub_61();
  v129 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v126[0] != &v127 )
    _libc_free((unsigned __int64)v126[0]);
  return v10;
}
