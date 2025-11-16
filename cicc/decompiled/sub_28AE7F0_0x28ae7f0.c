// Function: sub_28AE7F0
// Address: 0x28ae7f0
//
__int64 __fastcall sub_28AE7F0(__int64 a1, __int64 a2, __int64 a3, _QWORD **a4)
{
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // r12
  unsigned int v8; // eax
  unsigned int v9; // r12d
  _QWORD *v11; // rdi
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  int v15; // eax
  __int64 v16; // rax
  int v17; // edx
  __int64 v18; // rsi
  int v19; // edx
  unsigned int v20; // ecx
  __int64 *v21; // rax
  __int64 v22; // r9
  __int64 v23; // r9
  unsigned int v24; // ecx
  __int64 *v25; // rax
  __int64 v26; // r10
  __int64 v27; // r10
  __int64 v28; // r8
  unsigned __int64 v29; // r14
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned __int16 v34; // ax
  unsigned __int8 v35; // r14
  _QWORD *v36; // rdi
  unsigned __int16 v37; // ax
  char v38; // cl
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rax
  __int64 v41; // rsi
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // r14
  __int64 v45; // rax
  __int32 v46; // ecx
  _QWORD *v47; // rdx
  __int64 v48; // rdi
  __int64 v49; // rdx
  __int64 v50; // rax
  unsigned int v51; // edx
  unsigned int v52; // eax
  __int64 **v53; // r14
  __int64 (__fastcall *v54)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v55; // rax
  __int64 (__fastcall *v56)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  __int64 v57; // r14
  __int64 (__fastcall *v58)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char); // rax
  __int64 v59; // rsi
  __int64 v60; // r10
  __int64 v61; // rax
  __int64 v62; // r14
  unsigned int v63; // eax
  unsigned int v64; // r10d
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rsi
  __int64 v69; // rax
  int v70; // ecx
  __int64 v71; // rdi
  int v72; // ecx
  unsigned int v73; // edx
  __int64 *v74; // rax
  __int64 v75; // r8
  __int64 v76; // rcx
  __int64 *v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // r8
  __int64 v81; // r9
  unsigned __int16 v82; // ax
  __int64 v83; // rdx
  __int64 v84; // r12
  unsigned int *v85; // r13
  unsigned int *v86; // rbx
  __int64 v87; // rdx
  _QWORD **v88; // rdx
  int v89; // ecx
  __int64 *v90; // rax
  __int64 v91; // rsi
  __int64 v92; // rdx
  __int64 v93; // r12
  __int64 v94; // rbx
  __int64 v95; // rdx
  unsigned int v96; // esi
  __int64 **v97; // r14
  __int64 (__fastcall *v98)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v99; // rax
  int v100; // eax
  int v101; // eax
  _QWORD *v102; // rax
  __int64 v103; // rsi
  __int64 v104; // r14
  __int64 v105; // r14
  __int64 v106; // r12
  __int64 v107; // rbx
  __int64 v108; // rdx
  unsigned int v109; // esi
  _QWORD *v110; // rax
  __int64 v111; // rsi
  __int64 v112; // r14
  __int64 v113; // r14
  __int64 v114; // r12
  __int64 v115; // rbx
  __int64 v116; // rdx
  unsigned int v117; // esi
  int v118; // eax
  int v119; // r9d
  unsigned __int64 v120; // rsi
  int v121; // edi
  int v122; // edi
  __int64 v123; // [rsp+18h] [rbp-1B8h]
  __int64 v124; // [rsp+20h] [rbp-1B0h]
  __int64 v125; // [rsp+20h] [rbp-1B0h]
  __int64 v126; // [rsp+28h] [rbp-1A8h]
  unsigned __int8 v127; // [rsp+28h] [rbp-1A8h]
  unsigned __int8 v128; // [rsp+28h] [rbp-1A8h]
  __int64 v129; // [rsp+28h] [rbp-1A8h]
  __int64 v130; // [rsp+28h] [rbp-1A8h]
  unsigned __int8 *v131; // [rsp+40h] [rbp-190h]
  __int64 v132; // [rsp+48h] [rbp-188h]
  unsigned __int8 v133; // [rsp+54h] [rbp-17Ch]
  unsigned int v134; // [rsp+54h] [rbp-17Ch]
  __int64 v135; // [rsp+58h] [rbp-178h]
  unsigned __int64 v136; // [rsp+58h] [rbp-178h]
  __int64 v137; // [rsp+58h] [rbp-178h]
  _QWORD *v138; // [rsp+58h] [rbp-178h]
  _QWORD *v139; // [rsp+58h] [rbp-178h]
  unsigned __int8 v140; // [rsp+58h] [rbp-178h]
  unsigned __int64 v142; // [rsp+68h] [rbp-168h]
  _QWORD *v143; // [rsp+68h] [rbp-168h]
  _QWORD *v144; // [rsp+68h] [rbp-168h]
  unsigned __int8 v145; // [rsp+68h] [rbp-168h]
  __int64 v146; // [rsp+78h] [rbp-158h]
  _BYTE *v147[4]; // [rsp+80h] [rbp-150h] BYREF
  __int16 v148; // [rsp+A0h] [rbp-130h]
  _QWORD v149[4]; // [rsp+B0h] [rbp-120h] BYREF
  __int16 v150; // [rsp+D0h] [rbp-100h]
  __m128i v151; // [rsp+E0h] [rbp-F0h] BYREF
  __m128i v152; // [rsp+F0h] [rbp-E0h] BYREF
  __m128i v153; // [rsp+100h] [rbp-D0h] BYREF
  __m128i v154; // [rsp+110h] [rbp-C0h] BYREF
  __m128i v155; // [rsp+120h] [rbp-B0h] BYREF
  __m128i v156; // [rsp+130h] [rbp-A0h]
  __int64 v157; // [rsp+140h] [rbp-90h]
  __int64 v158; // [rsp+148h] [rbp-88h]
  __int64 v159; // [rsp+150h] [rbp-80h]
  _QWORD *v160; // [rsp+158h] [rbp-78h]
  __int64 v161; // [rsp+160h] [rbp-70h]
  __int64 v162; // [rsp+168h] [rbp-68h]
  void *v163; // [rsp+190h] [rbp-40h]

  v5 = a1;
  v6 = a2;
  v7 = *(_QWORD *)(a1 + 16);
  v142 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v155.m128i_i64[1] = *(_QWORD *)(a1 + 24);
  v154 = (__m128i)(unsigned __int64)sub_B43CC0(a2);
  v156.m128i_i64[0] = v7;
  v155.m128i_i64[0] = 0;
  v156.m128i_i64[1] = a2;
  v157 = 0;
  v158 = 0;
  LOWORD(v159) = 257;
  v8 = sub_9B6260(v142, &v154, 0);
  if ( !(_BYTE)v8 )
    return 0;
  v9 = v8;
  sub_D671D0(&v151, a2);
  v11 = *a4;
  v12 = _mm_load_si128(&v151);
  v13 = _mm_load_si128(&v152);
  LOBYTE(v157) = 1;
  v14 = _mm_load_si128(&v153);
  v154 = v12;
  v155 = v13;
  v156 = v14;
  v15 = sub_CF63E0(v11, (unsigned __int8 *)a2, &v154, (__int64)(a4 + 1)) & 2;
  v133 = v15;
  if ( v15 )
    return 0;
  v16 = *(_QWORD *)(v5 + 40);
  v17 = *(_DWORD *)(v16 + 56);
  v18 = *(_QWORD *)(v16 + 40);
  if ( v17 )
  {
    v19 = v17 - 1;
    v20 = v19 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v21 = (__int64 *)(v18 + 16LL * v20);
    v22 = *v21;
    if ( v6 == *v21 )
    {
LABEL_7:
      v23 = v21[1];
    }
    else
    {
      v100 = 1;
      while ( v22 != -4096 )
      {
        v121 = v100 + 1;
        v20 = v19 & (v100 + v20);
        v21 = (__int64 *)(v18 + 16LL * v20);
        v22 = *v21;
        if ( v6 == *v21 )
          goto LABEL_7;
        v100 = v121;
      }
      v23 = 0;
    }
    v24 = v19 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v25 = (__int64 *)(v18 + 16LL * v24);
    v26 = *v25;
    if ( a3 == *v25 )
    {
LABEL_9:
      v27 = v25[1];
    }
    else
    {
      v101 = 1;
      while ( v26 != -4096 )
      {
        v122 = v101 + 1;
        v24 = v19 & (v101 + v24);
        v25 = (__int64 *)(v18 + 16LL * v24);
        v26 = *v25;
        if ( a3 == *v25 )
          goto LABEL_9;
        v101 = v122;
      }
      v27 = 0;
    }
  }
  else
  {
    v23 = 0;
    v27 = 0;
  }
  v132 = v27;
  v135 = v23;
  sub_D67210(&v151, a3);
  if ( sub_28A9CB0(a4, v132, v135, 0, v28, v135, *(_OWORD *)&v151, *(_OWORD *)&v152, *(_OWORD *)&v153) )
    return 0;
  v131 = *(unsigned __int8 **)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
  v29 = *(_QWORD *)(a3 + 32 * (2LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)));
  v136 = v29;
  if ( sub_28A9550(v131, a3, v6) )
    return 0;
  if ( v142 == v29 )
  {
    sub_28AAD10(v5, (_QWORD *)a3, v30, v31, v32, v33);
    return v9;
  }
  v34 = sub_A74840((_QWORD *)(v6 + 72), 0);
  v35 = v34;
  v36 = (_QWORD *)(a3 + 72);
  if ( HIBYTE(v34) )
  {
    v82 = sub_A74840(v36, 0);
    v38 = v35;
    if ( HIBYTE(v82) )
    {
      v38 = v82;
      if ( v35 >= (unsigned __int8)v82 )
        v38 = v35;
    }
    else if ( !v35 )
    {
      v133 = 0;
      goto LABEL_22;
    }
  }
  else
  {
    v37 = sub_A74840(v36, 0);
    if ( !HIBYTE(v37) )
      goto LABEL_22;
    v38 = v37;
  }
  if ( (unsigned __int64)(1LL << v38) > 1 && *(_BYTE *)v142 == 17 )
  {
    v39 = *(_QWORD *)(v142 + 24);
    if ( *(_DWORD *)(v142 + 32) > 0x40u )
      v39 = *(_QWORD *)v39;
    v133 = -1;
    v40 = -(__int64)(v39 | (1LL << v38)) & (v39 | (1LL << v38));
    if ( v40 )
    {
      _BitScanReverse64(&v40, v40);
      v133 = 63 - (v40 ^ 0x3F);
    }
  }
LABEL_22:
  sub_23D0AB0((__int64)&v154, v6, 0, 0, 0);
  v41 = *(_QWORD *)(a3 + 48);
  v149[0] = v41;
  if ( v41 && (sub_B96E90((__int64)v149, v41, 1), (v44 = v149[0]) != 0) )
  {
    v45 = v154.m128i_i64[0];
    v46 = v154.m128i_i32[2];
    v47 = (_QWORD *)(v154.m128i_i64[0] + 16LL * v154.m128i_u32[2]);
    if ( (_QWORD *)v154.m128i_i64[0] != v47 )
    {
      while ( *(_DWORD *)v45 )
      {
        v45 += 16;
        if ( v47 == (_QWORD *)v45 )
          goto LABEL_63;
      }
      *(_QWORD *)(v45 + 8) = v149[0];
      goto LABEL_29;
    }
LABEL_63:
    if ( v154.m128i_u32[2] >= (unsigned __int64)v154.m128i_u32[3] )
    {
      v120 = v154.m128i_u32[2] + 1LL;
      if ( v154.m128i_u32[3] < v120 )
      {
        sub_C8D5F0((__int64)&v154, &v155, v120, 0x10u, v42, v43);
        v47 = (_QWORD *)(v154.m128i_i64[0] + 16LL * v154.m128i_u32[2]);
      }
      *v47 = 0;
      v47[1] = v44;
      v44 = v149[0];
      ++v154.m128i_i32[2];
    }
    else
    {
      if ( v47 )
      {
        *(_DWORD *)v47 = 0;
        v47[1] = v44;
        v46 = v154.m128i_i32[2];
        v44 = v149[0];
      }
      v154.m128i_i32[2] = v46 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v154, 0);
    v44 = v149[0];
  }
  if ( v44 )
LABEL_29:
    sub_B91220((__int64)v149, v44);
  v48 = v161;
  v49 = *(_QWORD *)(v136 + 8);
  v50 = *(_QWORD *)(v142 + 8);
  if ( v49 != v50 )
  {
    v51 = *(_DWORD *)(v49 + 8);
    v52 = *(_DWORD *)(v50 + 8);
    v148 = 257;
    if ( v51 >> 8 > v52 >> 8 )
    {
      v53 = *(__int64 ***)(v136 + 8);
      if ( v53 == *(__int64 ***)(v142 + 8) )
      {
        v55 = v142;
        goto LABEL_38;
      }
      v54 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v161 + 120LL);
      if ( v54 == sub_920130 )
      {
        if ( *(_BYTE *)v142 > 0x15u )
          goto LABEL_101;
        if ( (unsigned __int8)sub_AC4810(0x27u) )
          v55 = sub_ADAB70(39, v142, v53, 0);
        else
          v55 = sub_AA93C0(0x27u, v142, (__int64)v53);
      }
      else
      {
        v55 = v54(v161, 39u, (_BYTE *)v142, *(_QWORD *)(v136 + 8));
      }
      v48 = v161;
      if ( v55 )
      {
LABEL_38:
        v142 = v55;
        goto LABEL_39;
      }
LABEL_101:
      v150 = 257;
      v102 = sub_BD2C40(72, unk_3F10A14);
      if ( v102 )
      {
        v103 = v142;
        v143 = v102;
        sub_B515B0((__int64)v102, v103, (__int64)v53, (__int64)v149, 0, 0);
        v102 = v143;
      }
      v144 = v102;
      (*(void (__fastcall **)(__int64, _QWORD *, _BYTE **, __int64, __int64))(*(_QWORD *)v162 + 16LL))(
        v162,
        v102,
        v147,
        v158,
        v159);
      v104 = 16LL * v154.m128i_u32[2];
      v55 = (__int64)v144;
      if ( v154.m128i_i64[0] != v154.m128i_i64[0] + v104 )
      {
        v145 = v9;
        v105 = v154.m128i_i64[0] + v104;
        v106 = v55;
        v129 = v6;
        v107 = v154.m128i_i64[0];
        do
        {
          v108 = *(_QWORD *)(v107 + 8);
          v109 = *(_DWORD *)v107;
          v107 += 16;
          sub_B99FD0(v106, v109, v108);
        }
        while ( v105 != v107 );
        v55 = v106;
        v6 = v129;
        v9 = v145;
      }
      v48 = v161;
      goto LABEL_38;
    }
    v97 = *(__int64 ***)(v142 + 8);
    if ( v97 == *(__int64 ***)(v136 + 8) )
    {
      v99 = v136;
      goto LABEL_85;
    }
    v98 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v161 + 120LL);
    if ( v98 == sub_920130 )
    {
      if ( *(_BYTE *)v136 > 0x15u )
        goto LABEL_108;
      if ( (unsigned __int8)sub_AC4810(0x27u) )
        v99 = sub_ADAB70(39, v136, v97, 0);
      else
        v99 = sub_AA93C0(0x27u, v136, (__int64)v97);
    }
    else
    {
      v99 = v98(v161, 39u, (_BYTE *)v136, *(_QWORD *)(v142 + 8));
    }
    v48 = v161;
    if ( v99 )
    {
LABEL_85:
      v136 = v99;
      goto LABEL_39;
    }
LABEL_108:
    v150 = 257;
    v110 = sub_BD2C40(72, unk_3F10A14);
    if ( v110 )
    {
      v111 = v136;
      v138 = v110;
      sub_B515B0((__int64)v110, v111, (__int64)v97, (__int64)v149, 0, 0);
      v110 = v138;
    }
    v139 = v110;
    (*(void (__fastcall **)(__int64, _QWORD *, _BYTE **, __int64, __int64))(*(_QWORD *)v162 + 16LL))(
      v162,
      v110,
      v147,
      v158,
      v159);
    v112 = 16LL * v154.m128i_u32[2];
    v99 = (__int64)v139;
    if ( v154.m128i_i64[0] != v154.m128i_i64[0] + v112 )
    {
      v140 = v9;
      v113 = v154.m128i_i64[0] + v112;
      v114 = v99;
      v130 = v6;
      v115 = v154.m128i_i64[0];
      do
      {
        v116 = *(_QWORD *)(v115 + 8);
        v117 = *(_DWORD *)v115;
        v115 += 16;
        sub_B99FD0(v114, v117, v116);
      }
      while ( v113 != v115 );
      v99 = v114;
      v6 = v130;
      v9 = v140;
    }
    v48 = v161;
    goto LABEL_85;
  }
LABEL_39:
  v148 = 257;
  v56 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v48 + 56LL);
  if ( v56 != sub_928890 )
  {
    v57 = v56(v48, 37u, (_BYTE *)v136, (_BYTE *)v142);
LABEL_43:
    if ( v57 )
      goto LABEL_44;
    goto LABEL_71;
  }
  if ( *(_BYTE *)v136 <= 0x15u && *(_BYTE *)v142 <= 0x15u )
  {
    v57 = sub_AAB310(0x25u, (unsigned __int8 *)v136, (unsigned __int8 *)v142);
    goto LABEL_43;
  }
LABEL_71:
  v150 = 257;
  v57 = (__int64)sub_BD2C40(72, unk_3F10FD0);
  if ( v57 )
  {
    v88 = *(_QWORD ***)(v136 + 8);
    v89 = *((unsigned __int8 *)v88 + 8);
    if ( (unsigned int)(v89 - 17) > 1 )
    {
      v91 = sub_BCB2A0(*v88);
    }
    else
    {
      BYTE4(v146) = (_BYTE)v89 == 18;
      LODWORD(v146) = *((_DWORD *)v88 + 8);
      v90 = (__int64 *)sub_BCB2A0(*v88);
      v91 = sub_BCE1B0(v90, v146);
    }
    sub_B523C0(v57, v91, 53, 37, v136, v142, (__int64)v149, 0, 0, 0);
  }
  (*(void (__fastcall **)(__int64, __int64, _BYTE **, __int64, __int64))(*(_QWORD *)v162 + 16LL))(
    v162,
    v57,
    v147,
    v158,
    v159);
  v92 = 16LL * v154.m128i_u32[2];
  if ( v154.m128i_i64[0] != v154.m128i_i64[0] + v92 )
  {
    v128 = v9;
    v93 = v154.m128i_i64[0] + v92;
    v125 = v6;
    v94 = v154.m128i_i64[0];
    do
    {
      v95 = *(_QWORD *)(v94 + 8);
      v96 = *(_DWORD *)v94;
      v94 += 16;
      sub_B99FD0(v57, v96, v95);
    }
    while ( v93 != v94 );
    v9 = v128;
    v6 = v125;
  }
LABEL_44:
  v148 = 257;
  v58 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char))(*(_QWORD *)v161 + 32LL);
  if ( v58 != sub_9201A0 )
  {
    v59 = 15;
    v60 = v58(v161, 15u, (_BYTE *)v136, (_BYTE *)v142, 0, 0);
    goto LABEL_49;
  }
  if ( *(_BYTE *)v136 <= 0x15u && *(_BYTE *)v142 <= 0x15u )
  {
    v59 = v136;
    if ( (unsigned __int8)sub_AC47B0(15) )
      v60 = sub_AD5570(15, v136, (unsigned __int8 *)v142, 0, 0);
    else
      v60 = sub_AABE40(0xFu, (unsigned __int8 *)v136, (unsigned __int8 *)v142);
LABEL_49:
    if ( v60 )
      goto LABEL_50;
  }
  v150 = 257;
  v59 = sub_B504D0(15, v136, v142, (__int64)v149, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE **, __int64, __int64))(*(_QWORD *)v162 + 16LL))(
    v162,
    v59,
    v147,
    v158,
    v159);
  v60 = v59;
  v83 = 16LL * v154.m128i_u32[2];
  if ( v154.m128i_i64[0] != v154.m128i_i64[0] + v83 )
  {
    v127 = v9;
    v84 = v59;
    v124 = v5;
    v85 = (unsigned int *)(v154.m128i_i64[0] + v83);
    v123 = v6;
    v86 = (unsigned int *)v154.m128i_i64[0];
    do
    {
      v87 = *((_QWORD *)v86 + 1);
      v59 = *v86;
      v86 += 4;
      sub_B99FD0(v84, v59, v87);
    }
    while ( v85 != v86 );
    v60 = v84;
    v5 = v124;
    v9 = v127;
    v6 = v123;
  }
LABEL_50:
  v126 = v60;
  v150 = 257;
  v61 = sub_AD6530(*(_QWORD *)(v136 + 8), v59);
  v62 = sub_B36550((unsigned int **)&v154, v57, v61, v126, (__int64)v149, 0);
  v63 = v133;
  BYTE1(v63) = 1;
  v64 = v63;
  v65 = 1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
  v134 = v64;
  v150 = 257;
  v137 = *(_QWORD *)(a3 + 32 * v65);
  v147[0] = (_BYTE *)v142;
  v66 = sub_BCB2B0(v160);
  v67 = sub_921130((unsigned int **)&v154, v66, (__int64)v131, v147, 1, (__int64)v149, 0);
  v68 = sub_B34240((__int64)&v154, v67, v137, v62, v134, 0, 0, 0, 0);
  v69 = *(_QWORD *)(v5 + 40);
  v70 = *(_DWORD *)(v69 + 56);
  v71 = *(_QWORD *)(v69 + 40);
  if ( v70 )
  {
    v72 = v70 - 1;
    v73 = v72 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v74 = (__int64 *)(v71 + 16LL * v73);
    v75 = *v74;
    if ( v6 == *v74 )
    {
LABEL_52:
      v76 = v74[1];
      goto LABEL_53;
    }
    v118 = 1;
    while ( v75 != -4096 )
    {
      v119 = v118 + 1;
      v73 = v72 & (v118 + v73);
      v74 = (__int64 *)(v71 + 16LL * v73);
      v75 = *v74;
      if ( v6 == *v74 )
        goto LABEL_52;
      v118 = v119;
    }
  }
  v76 = 0;
LABEL_53:
  v77 = (__int64 *)sub_D69520(*(_QWORD **)(v5 + 48), v68, 0, v76);
  sub_D75120(*(__int64 **)(v5 + 48), v77, 1);
  sub_28AAD10(v5, (_QWORD *)a3, v78, v79, v80, v81);
  nullsub_61();
  v163 = &unk_49DA100;
  nullsub_63();
  if ( (__m128i *)v154.m128i_i64[0] != &v155 )
    _libc_free(v154.m128i_u64[0]);
  return v9;
}
