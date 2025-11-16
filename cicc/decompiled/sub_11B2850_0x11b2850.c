// Function: sub_11B2850
// Address: 0x11b2850
//
unsigned __int8 *__fastcall sub_11B2850(const __m128i *a1, __int64 a2)
{
  unsigned int v2; // edx
  unsigned __int8 *v3; // r14
  int *v6; // rdi
  __int64 v8; // rsi
  unsigned __int8 *v9; // r8
  int v10; // edx
  int v11; // r14d
  _DWORD *v12; // r9
  unsigned __int64 v13; // r14
  unsigned __int8 *v14; // r13
  size_t v15; // r10
  unsigned __int32 v16; // r14d
  unsigned __int32 v17; // r9d
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  unsigned __int64 v20; // xmm2_8
  unsigned __int8 *v21; // r15
  __int64 *v22; // r14
  __m128i v23; // xmm3
  int v24; // eax
  unsigned __int8 v25; // al
  int v26; // ecx
  unsigned __int8 *v27; // rax
  unsigned __int8 *v28; // rax
  __int64 v29; // r10
  unsigned __int8 *v30; // r9
  __int64 *v31; // rax
  char v32; // r8
  unsigned __int8 v33; // al
  char v34; // al
  _DWORD *v35; // r13
  __int64 v36; // rcx
  _DWORD *v37; // rax
  _DWORD *v38; // r10
  __int64 v39; // r11
  __int64 v40; // rax
  unsigned __int8 *v41; // r14
  _BYTE *v42; // rdi
  __m128i *v43; // rsi
  int v44; // eax
  int v45; // r11d
  _DWORD *v46; // r13
  _DWORD *v47; // rsi
  _DWORD *v48; // rax
  __int64 v49; // r10
  int v50; // r11d
  __int64 v51; // rax
  _BYTE *v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rax
  unsigned int **v55; // rdi
  __int64 v56; // rax
  __int64 *v57; // rdi
  unsigned __int8 *v58; // rax
  unsigned __int8 *v59; // r13
  __int8 *v60; // rdi
  char v61; // r13
  signed __int32 *v62; // rdi
  unsigned int v63; // edx
  __int64 v64; // r9
  unsigned __int8 *v65; // r11
  const void *v66; // rax
  unsigned __int64 v67; // r8
  size_t v68; // r10
  unsigned int v69; // r8d
  __m128i *v70; // rcx
  unsigned __int64 v71; // r8
  __m128i *v72; // rax
  __m128i *v73; // rsi
  __m128i *v74; // rdx
  unsigned int v75; // eax
  __int64 v76; // rdx
  signed __int32 v77; // ecx
  unsigned __int64 v78; // rsi
  unsigned __int8 *v79; // rax
  unsigned int v80; // edx
  char v81; // al
  __m128i *v82; // rax
  signed __int32 v83; // edx
  unsigned __int32 v84; // edi
  unsigned __int8 *v85; // rax
  _DWORD *v86; // rax
  _DWORD *v87; // r10
  signed __int32 *v88; // rdi
  signed __int32 *v89; // rax
  signed __int32 *v90; // rsi
  signed __int32 v91; // edx
  signed __int32 v92; // edi
  unsigned __int8 *v93; // rax
  int v94; // eax
  size_t v95; // [rsp+8h] [rbp-1B8h]
  void *v96; // [rsp+10h] [rbp-1B0h]
  void *v97; // [rsp+10h] [rbp-1B0h]
  void *v98; // [rsp+10h] [rbp-1B0h]
  int v99; // [rsp+10h] [rbp-1B0h]
  char v100; // [rsp+18h] [rbp-1A8h]
  __int64 v101; // [rsp+18h] [rbp-1A8h]
  unsigned __int32 v102; // [rsp+18h] [rbp-1A8h]
  int v103; // [rsp+18h] [rbp-1A8h]
  unsigned __int8 *v104; // [rsp+18h] [rbp-1A8h]
  unsigned __int8 *v105; // [rsp+20h] [rbp-1A0h]
  unsigned int v106; // [rsp+20h] [rbp-1A0h]
  int v107; // [rsp+20h] [rbp-1A0h]
  __m128i *v108; // [rsp+20h] [rbp-1A0h]
  unsigned __int8 *v109; // [rsp+20h] [rbp-1A0h]
  unsigned int v110; // [rsp+20h] [rbp-1A0h]
  int v111; // [rsp+20h] [rbp-1A0h]
  char v112; // [rsp+2Fh] [rbp-191h]
  _BYTE *n; // [rsp+30h] [rbp-190h]
  size_t nf; // [rsp+30h] [rbp-190h]
  size_t na; // [rsp+30h] [rbp-190h]
  _DWORD *nb; // [rsp+30h] [rbp-190h]
  unsigned __int32 nc; // [rsp+30h] [rbp-190h]
  size_t nd; // [rsp+30h] [rbp-190h]
  int ng; // [rsp+30h] [rbp-190h]
  size_t nh; // [rsp+30h] [rbp-190h]
  int ne; // [rsp+30h] [rbp-190h]
  int src; // [rsp+38h] [rbp-188h]
  __int64 srca; // [rsp+38h] [rbp-188h]
  _DWORD *srcc; // [rsp+38h] [rbp-188h]
  unsigned __int8 *srcb; // [rsp+38h] [rbp-188h]
  void *srcd; // [rsp+38h] [rbp-188h]
  unsigned int v127; // [rsp+40h] [rbp-180h]
  bool v128; // [rsp+40h] [rbp-180h]
  unsigned __int8 *v129; // [rsp+40h] [rbp-180h]
  unsigned __int8 *v130; // [rsp+40h] [rbp-180h]
  unsigned __int8 *v131; // [rsp+40h] [rbp-180h]
  unsigned __int8 *v132; // [rsp+40h] [rbp-180h]
  char v133; // [rsp+48h] [rbp-178h]
  _BYTE *v134; // [rsp+50h] [rbp-170h] BYREF
  _BYTE *v135; // [rsp+58h] [rbp-168h] BYREF
  _QWORD v136[4]; // [rsp+60h] [rbp-160h] BYREF
  __int16 v137; // [rsp+80h] [rbp-140h]
  __m128i *v138; // [rsp+90h] [rbp-130h] BYREF
  __int64 v139; // [rsp+98h] [rbp-128h]
  _BYTE v140[64]; // [rsp+A0h] [rbp-120h] BYREF
  signed __int32 *v141; // [rsp+E0h] [rbp-E0h] BYREF
  __int64 v142; // [rsp+E8h] [rbp-D8h]
  _QWORD v143[2]; // [rsp+F0h] [rbp-D0h] BYREF
  __int16 v144; // [rsp+100h] [rbp-C0h]
  __m128i v145; // [rsp+130h] [rbp-90h] BYREF
  __m128i v146; // [rsp+140h] [rbp-80h] BYREF
  _QWORD v147[2]; // [rsp+150h] [rbp-70h] BYREF
  __m128i v148; // [rsp+160h] [rbp-60h]
  __int64 v149; // [rsp+170h] [rbp-50h]

  v2 = *(_DWORD *)(a2 + 80);
  if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL) + 32LL) != v2 )
    return 0;
  v6 = *(int **)(a2 + 72);
  v8 = v2;
  v112 = sub_B4EEA0(v6, v2, v2);
  if ( !v112 )
    return 0;
  v9 = *(unsigned __int8 **)(a2 - 32);
  v10 = *v9;
  if ( (_BYTE)v10 == 12 || v10 == 13 )
  {
    v12 = *(_DWORD **)(a2 + 72);
  }
  else
  {
    v11 = *(_DWORD *)(*(_QWORD *)(a2 + 8) + 32LL);
    if ( (unsigned __int8)(*v9 - 9) > 2u )
      goto LABEL_8;
    v8 = *(_QWORD *)(a2 - 32);
    v145.m128i_i64[0] = 0;
    v145.m128i_i64[1] = (__int64)v147;
    v139 = (__int64)&v141;
    v141 = (signed __int32 *)v143;
    v138 = &v145;
    v146.m128i_i64[0] = 8;
    v146.m128i_i32[2] = 0;
    v146.m128i_i8[12] = 1;
    v142 = 0x800000000LL;
    v61 = sub_AA8FD0(&v138, v8);
    if ( v61 )
    {
      while ( 1 )
      {
        v62 = v141;
        if ( !(_DWORD)v142 )
          break;
        v8 = *(_QWORD *)&v141[2 * (unsigned int)v142 - 2];
        LODWORD(v142) = v142 - 1;
        if ( !(unsigned __int8)sub_AA8FD0(&v138, v8) )
          goto LABEL_122;
      }
    }
    else
    {
LABEL_122:
      v62 = v141;
      v61 = 0;
    }
    if ( v62 != (signed __int32 *)v143 )
      _libc_free(v62, v8);
    if ( !v146.m128i_i8[12] )
      _libc_free(v145.m128i_i64[1], v8);
    if ( v61 )
    {
      v12 = *(_DWORD **)(a2 + 72);
      v9 = *(unsigned __int8 **)(a2 - 32);
    }
    else
    {
LABEL_8:
      v12 = *(_DWORD **)(a2 + 72);
      if ( v11 <= *v12 )
      {
        v3 = (unsigned __int8 *)a2;
        sub_B4E8A0(a2);
        return v3;
      }
      v9 = *(unsigned __int8 **)(a2 - 32);
    }
  }
  v13 = *(unsigned int *)(a2 + 80);
  v14 = *(unsigned __int8 **)(a2 - 64);
  v138 = (__m128i *)v140;
  v139 = 0x1000000000LL;
  v15 = 4 * v13;
  if ( v13 > 0x10 )
  {
    srcc = v12;
    v129 = v9;
    sub_C8D5F0((__int64)&v138, v140, v13, 4u, (__int64)v9, (__int64)v12);
    v9 = v129;
    v12 = srcc;
    v15 = 4 * v13;
    v60 = &v138->m128i_i8[4 * (unsigned int)v139];
LABEL_67:
    v8 = (__int64)v12;
    v130 = v9;
    memcpy(v60, v12, v15);
    LODWORD(v15) = v139;
    v9 = v130;
    goto LABEL_12;
  }
  if ( v15 )
  {
    v60 = v140;
    goto LABEL_67;
  }
LABEL_12:
  v16 = v15 + v13;
  LODWORD(v139) = v16;
  v17 = v16;
  if ( *v14 == 92 )
  {
    v80 = *((_DWORD *)v14 + 20);
    if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)v14 - 8) + 8LL) + 32LL) == v80 )
    {
      v8 = v80;
      v132 = v9;
      v81 = sub_B4EEA0(*((int **)v14 + 9), v80, v80);
      v9 = v132;
      v17 = v16;
      if ( v81 )
      {
        if ( v132 == *((unsigned __int8 **)v14 - 8) || v132 == *((unsigned __int8 **)v14 - 4) )
        {
          v82 = v138;
          v8 = (__int64)v138->m128i_i64 + 4 * (unsigned int)v139;
          if ( v138 != (__m128i *)v8 )
          {
            do
            {
              v83 = v82->m128i_i32[0];
              if ( v82->m128i_i32[0] != -1 )
              {
                v84 = v16 + v83;
                if ( v83 >= (int)v16 )
                  v84 = v83 - v16;
                v82->m128i_i32[0] = v84;
              }
              v82 = (__m128i *)((char *)v82 + 4);
            }
            while ( (__m128i *)v8 != v82 );
          }
          v85 = v14;
          v14 = v132;
          v9 = v85;
        }
      }
    }
  }
  if ( *v9 != 92
    || (v63 = *((_DWORD *)v9 + 20), *(_DWORD *)(*(_QWORD *)(*((_QWORD *)v9 - 8) + 8LL) + 32LL) != v63)
    || (v8 = v63, nc = v17, v131 = v9, !(unsigned __int8)sub_B4EEA0(*((int **)v9 + 9), v63, v63))
    || (v64 = nc,
        v65 = (unsigned __int8 *)*((_QWORD *)v131 - 4),
        srcb = (unsigned __int8 *)*((_QWORD *)v131 - 8),
        srcb != v14)
    && v14 != v65 )
  {
    if ( v138 != (__m128i *)v140 )
      _libc_free(v138, v8);
    goto LABEL_16;
  }
  v141 = (signed __int32 *)v143;
  v142 = 0x1000000000LL;
  v66 = (const void *)*((_QWORD *)v131 + 9);
  v67 = *((unsigned int *)v131 + 20);
  v68 = 4 * v67;
  if ( v67 > 0x10 )
  {
    v95 = 4 * v67;
    v98 = (void *)*((_QWORD *)v131 + 9);
    v102 = nc;
    v109 = v65;
    ng = *((_DWORD *)v131 + 20);
    sub_C8D5F0((__int64)&v141, v143, v67, 4u, v67, v64);
    LODWORD(v67) = ng;
    v65 = v109;
    LODWORD(v64) = v102;
    v66 = v98;
    v88 = &v141[(unsigned int)v142];
    v68 = v95;
  }
  else
  {
    if ( !v68 )
      goto LABEL_85;
    v88 = (signed __int32 *)v143;
  }
  v103 = v67;
  v110 = v64;
  nh = (size_t)v65;
  memcpy(v88, v66, v68);
  LODWORD(v68) = v142;
  LODWORD(v67) = v103;
  v64 = v110;
  v65 = (unsigned __int8 *)nh;
LABEL_85:
  v69 = v68 + v67;
  LODWORD(v142) = v69;
  if ( v14 == v65 )
  {
    v89 = v141;
    v90 = &v141[v69];
    if ( v141 != v90 )
    {
      do
      {
        v91 = *v89;
        if ( *v89 != -1 )
        {
          v92 = v64 + v91;
          if ( v91 >= (int)v16 )
            v92 = v91 - v64;
          *v89 = v92;
        }
        ++v89;
      }
      while ( v90 != v89 );
    }
    v93 = srcb;
    srcb = v65;
    v65 = v93;
  }
  v145.m128i_i64[0] = (__int64)&v146;
  v70 = &v146;
  v145.m128i_i64[1] = 0x1000000000LL;
  v71 = v16;
  if ( v16 )
  {
    v72 = &v146;
    v73 = &v146;
    if ( v16 > 0x10uLL )
    {
      v99 = v64;
      v104 = v65;
      sub_C8D5F0((__int64)&v145, &v146, v16, 4u, v16, v64);
      v73 = (__m128i *)v145.m128i_i64[0];
      LODWORD(v64) = v99;
      v65 = v104;
      v71 = v16;
      v72 = (__m128i *)(v145.m128i_i64[0] + 4LL * v145.m128i_u32[2]);
    }
    v74 = (__m128i *)((char *)v73 + 4 * v71);
    if ( v72 != v74 )
    {
      do
      {
        if ( v72 )
          v72->m128i_i32[0] = 0;
        v72 = (__m128i *)((char *)v72 + 4);
      }
      while ( v74 != v72 );
      v73 = (__m128i *)v145.m128i_i64[0];
    }
    v145.m128i_i32[2] = v16;
    v75 = 0;
    while ( 1 )
    {
      v76 = v75;
      v77 = v138->m128i_i32[v75];
      if ( v77 >= (int)v16 )
        v77 = v141[v75];
      ++v75;
      v73->m128i_i32[v76] = v77;
      if ( v75 == (_DWORD)v64 )
        break;
      v73 = (__m128i *)v145.m128i_i64[0];
    }
    v70 = (__m128i *)v145.m128i_i64[0];
    v71 = v145.m128i_u32[2];
  }
  v101 = (__int64)v65;
  v78 = unk_3F1FE60;
  v97 = (void *)v71;
  v108 = v70;
  v137 = 257;
  v79 = (unsigned __int8 *)sub_BD2C40(112, unk_3F1FE60);
  v3 = v79;
  if ( v79 )
  {
    v78 = (unsigned __int64)srcb;
    sub_B4E9E0((__int64)v79, (__int64)srcb, v101, v108, (__int64)v97, (__int64)v136, 0, 0);
  }
  if ( (__m128i *)v145.m128i_i64[0] != &v146 )
    _libc_free(v145.m128i_i64[0], v78);
  if ( v141 != (signed __int32 *)v143 )
    _libc_free(v141, v78);
  if ( v138 != (__m128i *)v140 )
    _libc_free(v138, v78);
  if ( v3 )
    return v3;
LABEL_16:
  v18 = _mm_loadu_si128(a1 + 6);
  v19 = _mm_loadu_si128(a1 + 7);
  v20 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v21 = *(unsigned __int8 **)(a2 - 64);
  v22 = *(__int64 **)(a2 - 32);
  v149 = a1[10].m128i_i64[0];
  v23 = _mm_loadu_si128(a1 + 9);
  v147[0] = v20;
  v145 = v18;
  v147[1] = a2;
  v146 = v19;
  v148 = v23;
  v24 = *v21;
  if ( (unsigned __int8)v24 > 0x1Cu )
  {
    src = *v21;
    if ( (unsigned int)(v24 - 42) <= 0x11 && v22 == *((__int64 **)v21 - 8) && **((_BYTE **)v21 - 4) <= 0x15u )
    {
      nd = *((_QWORD *)v21 - 4);
      v127 = v24 - 29;
      v30 = sub_AD93D0(v24 - 29, *(_QWORD *)(a2 + 8), 1, 0);
      if ( v30 )
      {
        v32 = v112;
        v29 = nd;
        goto LABEL_26;
      }
LABEL_36:
      v21 = *(unsigned __int8 **)(a2 - 64);
      goto LABEL_37;
    }
  }
  v25 = *(_BYTE *)v22;
  if ( *(_BYTE *)v22 <= 0x1Cu
    || (v26 = v25, src = v25, (unsigned int)v25 - 42 > 0x11)
    || (v27 = (unsigned __int8 *)*(v22 - 8)) == 0
    || v21 != v27
    || (n = (_BYTE *)*(v22 - 4), *n > 0x15u) )
  {
LABEL_37:
    if ( (unsigned __int8)(*v21 - 42) > 0x11u )
      return 0;
    v41 = *(unsigned __int8 **)(a2 - 32);
    if ( (unsigned __int8)(*v41 - 42) > 0x11u )
      return 0;
    v136[0] = 0;
    v138 = 0;
    v42 = (_BYTE *)*((_QWORD *)v21 - 8);
    if ( *v42 <= 0x15u
      && (v136[0] = *((_QWORD *)v21 - 8), *((_QWORD *)v21 - 4))
      && (v134 = (_BYTE *)*((_QWORD *)v21 - 4), v43 = (__m128i *)*((_QWORD *)v41 - 8), v43->m128i_i8[0] <= 0x15u)
      && (v138 = (__m128i *)*((_QWORD *)v41 - 8), *((_QWORD *)v41 - 4)) )
    {
      v135 = (_BYTE *)*((_QWORD *)v41 - 4);
      v44 = *v21;
      if ( (_BYTE)v44 != *v41 )
        return 0;
      v133 = 0;
      v45 = v44 - 29;
      v128 = 0;
    }
    else
    {
      v143[0] = 0;
      v141 = (signed __int32 *)&v134;
      v142 = (__int64)v136;
      v143[1] = &v134;
      if ( !(unsigned __int8)sub_11B1B70((__int64)&v141, v21) )
        return 0;
      v145.m128i_i64[0] = (__int64)&v135;
      v145.m128i_i64[1] = (__int64)&v138;
      v146.m128i_i64[0] = 0;
      v146.m128i_i64[1] = (__int64)&v135;
      v133 = sub_11B1B70((__int64)&v145, v41);
      if ( !v133 )
        return 0;
      ne = *v21 - 29;
      v94 = *v41 - 29;
      if ( *v41 == *v21 )
      {
        v128 = 0;
        v45 = *v21 - 29;
      }
      else
      {
        v111 = *v41 - 29;
        v128 = v94 == 25 || *v21 == 54;
        sub_11AF080((__int64)&v141, v21, a1[5].m128i_i64[1]);
        v45 = (int)v141;
        v94 = v111;
        if ( (_DWORD)v141 )
        {
          v136[0] = v143[0];
        }
        else
        {
          sub_11AF080((__int64)&v145, v41, a1[5].m128i_i64[1]);
          v94 = v111;
          if ( v145.m128i_i32[0] )
          {
            v138 = (__m128i *)v146.m128i_i64[0];
            v94 = v145.m128i_i32[0];
          }
          v45 = ne;
        }
      }
      if ( v45 != v94 )
        return 0;
      v42 = (_BYTE *)v136[0];
      if ( !v136[0] )
        return 0;
      v43 = v138;
      if ( !v138 )
        return 0;
    }
    v46 = *(_DWORD **)(a2 + 72);
    v106 = v45;
    nb = v46;
    srca = *(unsigned int *)(a2 + 80);
    sub_AD5CE0((__int64)v42, (__int64)v43, v46, srca, 0);
    v47 = &v46[srca];
    v48 = sub_11AEE30(v46, (__int64)v47, &dword_3F93574);
    v50 = v106;
    if ( v47 == v48 || v106 > 0x1B || ((0xED80000uLL >> v106) & 1) == 0 )
    {
      v112 = 0;
    }
    else
    {
      v51 = sub_F0BC20(v106, v49, v133);
      v50 = v106;
      v49 = v51;
    }
    v52 = v135;
    if ( v134 == v135 )
      goto LABEL_56;
    v53 = *((_QWORD *)v21 + 2);
    if ( (v53 && !*(_QWORD *)(v53 + 8) || (v54 = *((_QWORD *)v41 + 2)) != 0 && !*(_QWORD *)(v54 + 8))
      && (v133 == 1 || !v112) )
    {
      v55 = (unsigned int **)a1[2].m128i_i64[0];
      v96 = (void *)v49;
      v107 = v50;
      LOWORD(v147[0]) = 257;
      v56 = sub_A83CB0(v55, v134, v135, (__int64)v46, srca, (__int64)&v145);
      v49 = (__int64)v96;
      v50 = v107;
      v52 = (_BYTE *)v56;
LABEL_56:
      v57 = (__int64 *)a1[2].m128i_i64[0];
      LOWORD(v147[0]) = 257;
      if ( v133 )
        v58 = (unsigned __int8 *)sub_11AFCA0(v57, v50, (__int64)v52, v49, (int)v141, 0, (__int64)&v145, 0);
      else
        v58 = (unsigned __int8 *)sub_11AFCA0(v57, v50, v49, (__int64)v52, (int)v141, 0, (__int64)&v145, 0);
      v59 = v58;
      if ( *v58 > 0x1Cu )
      {
        sub_B45260(v58, (__int64)v21, 1);
        sub_B45560(v59, (unsigned __int64)v41);
        if ( v128 )
          sub_B44850(v59, 0);
        if ( v47 != sub_11AEE30(nb, (__int64)v47, &dword_3F93574) && !v112 )
          sub_B44F30(v59);
      }
      return sub_F162A0((__int64)a1, a2, (__int64)v59);
    }
    return 0;
  }
  v127 = v26 - 29;
  v28 = sub_AD93D0(v26 - 29, *(_QWORD *)(a2 + 8), 1, 0);
  v29 = (__int64)n;
  v30 = v28;
  if ( !v28 )
    goto LABEL_36;
  v31 = v22;
  v32 = 0;
  v22 = (__int64 *)v21;
  v21 = (unsigned __int8 *)v31;
LABEL_26:
  v33 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a2 + 8) + 24LL) + 8LL);
  if ( v33 <= 3u || v33 == 5 || (v33 & 0xFD) == 4 )
  {
    v100 = v32;
    v105 = v30;
    nf = v29;
    v34 = sub_9B4030(v22, 3, 0, &v145);
    v29 = nf;
    v30 = v105;
    v32 = v100;
    if ( (v34 & 3) != 0 )
      goto LABEL_36;
  }
  v35 = *(_DWORD **)(a2 + 72);
  v36 = *(unsigned int *)(a2 + 80);
  na = v36;
  if ( v32 )
    sub_AD5CE0(v29, (__int64)v30, v35, v36, 0);
  else
    sub_AD5CE0((__int64)v30, v29, v35, v36, 0);
  v37 = sub_11AEE30(v35, (__int64)&v35[na], &dword_3F93574);
  if ( v38 == v37 || v127 - 22 > 1 && (unsigned int)(src - 48) > 1 && (unsigned int)(src - 54) > 2 )
  {
    srcd = v38;
    v144 = 257;
    v3 = (unsigned __int8 *)sub_B504D0(v127, (__int64)v22, v39, (__int64)&v141, 0, 0);
    sub_B45260(v3, (__int64)v21, 1);
    v86 = sub_11AEE30(v35, (__int64)srcd, &dword_3F93574);
    if ( v87 != v86 )
      sub_B44F30(v3);
  }
  else
  {
    v40 = sub_F0BC20(v127, v39, 1);
    v144 = 257;
    v3 = (unsigned __int8 *)sub_B504D0(v127, (__int64)v22, v40, (__int64)&v141, 0, 0);
    sub_B45260(v3, (__int64)v21, 1);
  }
  if ( !v3 )
    goto LABEL_36;
  return v3;
}
