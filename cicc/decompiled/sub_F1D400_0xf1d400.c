// Function: sub_F1D400
// Address: 0xf1d400
//
__int64 __fastcall sub_F1D400(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 *v7; // r15
  __int64 v8; // rax
  __int64 *v9; // r13
  __int64 v12; // r9
  __int64 v13; // rax
  __int64 v14; // r8
  unsigned __int64 v15; // rdx
  __int64 *v16; // r14
  __int64 *v17; // r12
  __int64 v18; // r15
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 *v23; // r13
  __int64 v24; // rax
  __int64 *v25; // r15
  __int64 v26; // rbx
  __int64 v27; // r14
  char *v28; // rax
  char *v29; // r12
  __int64 v30; // rsi
  __int64 *v31; // rax
  __int64 v32; // rdx
  __int64 *v33; // rax
  __int64 *v34; // rbx
  __int64 result; // rax
  __int64 v36; // rax
  __m128i v37; // xmm5
  __m128i v38; // xmm6
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rcx
  __int64 *v42; // rdx
  __int64 v43; // rax
  __int64 v44; // r12
  __int64 v45; // rsi
  __int64 v46; // rax
  unsigned __int8 v47; // dl
  __int64 v48; // r14
  __int64 v49; // r15
  __int64 v50; // rax
  __int64 v51; // r9
  __int64 v52; // rdx
  __int64 v53; // r12
  int v54; // eax
  __int64 *v55; // rdx
  __int64 *v56; // r12
  __int64 *v57; // rbx
  __int64 n; // rbx
  __int64 v59; // rdi
  __int64 v60; // rax
  __int64 *v61; // rbx
  __int64 v62; // r13
  __int8 v63; // r14
  __int64 v64; // rax
  __m128i v65; // xmm1
  __m128i v66; // xmm0
  bool v67; // zf
  __int64 *v68; // rax
  unsigned int v69; // edx
  __int64 v70; // rdx
  __int64 v71; // r12
  __int64 v72; // rsi
  __int64 v73; // rax
  unsigned __int8 v74; // dl
  __int64 v75; // rax
  char v76; // dl
  __int64 *v77; // rdx
  __int64 v78; // rax
  const __m128i *v79; // rax
  __int64 *v80; // rdx
  __int64 v81; // rbx
  const __m128i *v82; // rbx
  const __m128i *v83; // rdx
  _QWORD *v84; // rdx
  __int64 v85; // rcx
  __int64 v86; // r8
  __int64 v87; // r9
  __int64 v88; // rax
  unsigned int v89; // edx
  unsigned int v90; // esi
  unsigned int v91; // edi
  _QWORD *v92; // rax
  __int64 ii; // rbx
  __int64 v94; // rdi
  __int64 v95; // rax
  __int64 v96; // rdi
  _QWORD *v97; // rdx
  _QWORD *v98; // r12
  _QWORD *i; // rbx
  unsigned __int64 m; // r15
  unsigned __int64 j; // r12
  __int64 *v102; // rsi
  __int64 v103; // rax
  unsigned __int8 v104; // dl
  __int8 v105; // r14
  __int64 v106; // rbx
  __int64 *v107; // r10
  int v108; // ebx
  unsigned int v109; // eax
  int v110; // esi
  unsigned int k; // eax
  __int64 *v112; // rdx
  unsigned int v113; // eax
  __int64 v114; // r9
  __int64 *v115; // rax
  __int64 v116; // rax
  __int64 v117; // rsi
  __int64 v118; // rax
  __int64 *v119; // rcx
  __int64 v120; // rax
  __int64 v124; // [rsp+38h] [rbp-3A8h]
  __int64 v125; // [rsp+40h] [rbp-3A0h]
  unsigned __int64 v126; // [rsp+50h] [rbp-390h]
  _QWORD *v127; // [rsp+68h] [rbp-378h]
  __int64 v128; // [rsp+70h] [rbp-370h]
  __int64 v129; // [rsp+78h] [rbp-368h]
  __int64 v130; // [rsp+80h] [rbp-360h]
  __int64 *v131; // [rsp+88h] [rbp-358h]
  __int64 *v132; // [rsp+88h] [rbp-358h]
  __int64 v133; // [rsp+90h] [rbp-350h]
  __int64 *v134; // [rsp+90h] [rbp-350h]
  __int64 *v135; // [rsp+98h] [rbp-348h]
  __int64 v136; // [rsp+98h] [rbp-348h]
  __int64 v137; // [rsp+98h] [rbp-348h]
  __int64 v138; // [rsp+98h] [rbp-348h]
  __int64 v139; // [rsp+A8h] [rbp-338h] BYREF
  __int64 *v140; // [rsp+B0h] [rbp-330h] BYREF
  __int64 v141; // [rsp+B8h] [rbp-328h]
  _BYTE v142[16]; // [rsp+C0h] [rbp-320h] BYREF
  __int64 *v143; // [rsp+D0h] [rbp-310h] BYREF
  __int64 v144; // [rsp+D8h] [rbp-308h]
  _BYTE v145[16]; // [rsp+E0h] [rbp-300h] BYREF
  __int64 *v146; // [rsp+F0h] [rbp-2F0h] BYREF
  __int64 *v147; // [rsp+F8h] [rbp-2E8h]
  const __m128i *v148; // [rsp+100h] [rbp-2E0h]
  const __m128i *v149; // [rsp+108h] [rbp-2D8h]
  __m128i v150; // [rsp+110h] [rbp-2D0h] BYREF
  __m128i v151; // [rsp+120h] [rbp-2C0h] BYREF
  __m128i v152; // [rsp+130h] [rbp-2B0h]
  __int64 v153; // [rsp+140h] [rbp-2A0h]
  void *src; // [rsp+150h] [rbp-290h] BYREF
  __int64 v155; // [rsp+158h] [rbp-288h]
  _BYTE v156[48]; // [rsp+160h] [rbp-280h] BYREF
  __int64 v157; // [rsp+190h] [rbp-250h] BYREF
  __m128i v158; // [rsp+198h] [rbp-248h] BYREF
  __m128i v159; // [rsp+1A8h] [rbp-238h] BYREF
  __int64 v160; // [rsp+1B8h] [rbp-228h]
  __int64 v161; // [rsp+1D0h] [rbp-210h] BYREF
  __int64 v162; // [rsp+1D8h] [rbp-208h]
  __int64 *v163; // [rsp+1E0h] [rbp-200h] BYREF
  unsigned int v164; // [rsp+1E8h] [rbp-1F8h]
  __int64 *v165; // [rsp+2C0h] [rbp-120h] BYREF
  __int64 v166; // [rsp+2C8h] [rbp-118h]
  __int64 *v167; // [rsp+2D0h] [rbp-110h] BYREF
  unsigned int v168; // [rsp+2D8h] [rbp-108h]
  int v169; // [rsp+378h] [rbp-68h] BYREF
  __int64 v170; // [rsp+380h] [rbp-60h]
  int *v171; // [rsp+388h] [rbp-58h]
  int *v172; // [rsp+390h] [rbp-50h]
  __int64 v173; // [rsp+398h] [rbp-48h]
  char v174; // [rsp+3B0h] [rbp-30h] BYREF

  v7 = *(__int64 **)a7;
  v8 = *(unsigned int *)(a7 + 8);
  v140 = (__int64 *)v142;
  v9 = &v7[v8];
  v141 = 0x200000000LL;
  if ( v7 == v9 )
  {
    src = v156;
    v155 = 0x600000000LL;
LABEL_127:
    v23 = (__int64 *)v156;
    v25 = (__int64 *)v156;
    goto LABEL_128;
  }
  do
  {
    if ( a6 != sub_B140C0(*v7) )
    {
      v13 = (unsigned int)v141;
      v14 = *v7;
      v15 = (unsigned int)v141 + 1LL;
      if ( v15 > HIDWORD(v141) )
      {
        v136 = *v7;
        sub_C8D5F0((__int64)&v140, v142, v15, 8u, v14, v12);
        v13 = (unsigned int)v141;
        v14 = v136;
      }
      v140[v13] = v14;
      LODWORD(v141) = v141 + 1;
    }
    ++v7;
  }
  while ( v9 != v7 );
  v16 = v140;
  v17 = &v140[(unsigned int)v141];
  src = v156;
  v155 = 0x600000000LL;
  if ( v17 == v140 )
    goto LABEL_127;
  do
  {
    while ( 1 )
    {
      v18 = *v16;
      if ( a5 == sub_B140C0(*v16) )
        break;
      if ( v17 == ++v16 )
        goto LABEL_14;
    }
    v21 = (unsigned int)v155;
    v22 = (unsigned int)v155 + 1LL;
    if ( v22 > HIDWORD(v155) )
    {
      sub_C8D5F0((__int64)&src, v156, v22, 8u, v19, v20);
      v21 = (unsigned int)v155;
    }
    ++v16;
    *((_QWORD *)src + v21) = v18;
    LODWORD(v155) = v155 + 1;
  }
  while ( v17 != v16 );
LABEL_14:
  v23 = (__int64 *)src;
  v24 = 8LL * (unsigned int)v155;
  v25 = (__int64 *)((char *)src + v24);
  v26 = v24 >> 3;
  if ( v24 )
  {
    while ( 1 )
    {
      v27 = 8 * v26;
      v28 = (char *)sub_2207800(8 * v26, &unk_435FF63);
      v29 = v28;
      if ( v28 )
        break;
      v26 >>= 1;
      if ( !v26 )
        goto LABEL_128;
    }
    sub_F08C40(v23, v25, v28, v26);
    goto LABEL_17;
  }
LABEL_128:
  v27 = 0;
  v29 = 0;
  sub_F084D0(v23, v25);
LABEL_17:
  v30 = v27;
  j_j___libc_free_0(v29, v27);
  v31 = (__int64 *)&v163;
  v161 = 0;
  v162 = 1;
  do
  {
    *v31 = -4096;
    v31 += 7;
    *(v31 - 6) = 0;
    *((_BYTE *)v31 - 24) = 0;
    *(v31 - 2) = 0;
  }
  while ( v31 != (__int64 *)&v165 );
  v32 = (unsigned int)v155;
  v33 = (__int64 *)&v167;
  if ( (unsigned int)v155 <= 1 )
    goto LABEL_20;
  v165 = 0;
  v166 = 1;
  do
  {
    *v33 = -4096;
    v33 += 7;
    *(v33 - 6) = 0;
    *((_BYTE *)v33 - 24) = 0;
    *(v33 - 2) = 0;
  }
  while ( v33 != (__int64 *)&v174 );
  v61 = (__int64 *)src;
  v131 = (__int64 *)((char *)src + 8 * (unsigned int)v32);
  do
  {
    v71 = *v61;
    v72 = *(_QWORD *)(*v61 + 24);
    v157 = v72;
    if ( v72 )
      sub_B96E90((__int64)&v157, v72, 1);
    v73 = sub_B10CD0((__int64)&v157);
    v74 = *(_BYTE *)(v73 - 16);
    if ( (v74 & 2) != 0 )
    {
      if ( *(_DWORD *)(v73 - 24) != 2 )
        goto LABEL_71;
      v75 = *(_QWORD *)(v73 - 32);
    }
    else
    {
      if ( ((*(_WORD *)(v73 - 16) >> 6) & 0xF) != 2 )
      {
LABEL_71:
        v62 = 0;
        goto LABEL_72;
      }
      v75 = v73 - 16 - 8LL * ((v74 >> 2) & 0xF);
    }
    v62 = *(_QWORD *)(v75 + 8);
LABEL_72:
    v63 = 0;
    v133 = sub_B11F60(v71 + 80);
    v137 = sub_B12000(v71 + 72);
    if ( v133 )
    {
      sub_AF47B0((__int64)&v150.m128i_i64[1], *(unsigned __int64 **)(v133 + 16), *(unsigned __int64 **)(v133 + 24));
      v63 = v151.m128i_i8[8];
    }
    if ( v157 )
      sub_B91220((__int64)&v157, v157);
    v64 = sub_B140A0(v71);
    v151.m128i_i8[8] = v63;
    v30 = (__int64)&v157;
    v157 = v64;
    v65 = _mm_loadu_si128(&v151);
    v152.m128i_i64[0] = v62;
    v150.m128i_i64[0] = v137;
    v66 = _mm_loadu_si128(&v150);
    v160 = v62;
    v158 = v66;
    v159 = v65;
    v67 = (unsigned __int8)sub_F15B30((__int64)&v165, &v157, &v143) == 0;
    v68 = v143;
    if ( v67 )
    {
      v165 = (__int64 *)((char *)v165 + 1);
      v146 = v143;
      v69 = ((unsigned int)v166 >> 1) + 1;
      if ( (v166 & 1) == 0 )
      {
        v30 = v168;
        if ( 4 * v69 < 3 * v168 )
          goto LABEL_79;
LABEL_92:
        LODWORD(v30) = 2 * v30;
        goto LABEL_93;
      }
      v30 = 4;
      if ( 4 * v69 >= 0xC )
        goto LABEL_92;
LABEL_79:
      if ( (unsigned int)v30 - (v69 + HIDWORD(v166)) <= (unsigned int)v30 >> 3 )
      {
LABEL_93:
        sub_F1CE00((__int64)&v165, v30);
        v30 = (__int64)&v157;
        sub_F15B30((__int64)&v165, &v157, &v146);
        v68 = v146;
        v69 = ((unsigned int)v166 >> 1) + 1;
      }
      LODWORD(v166) = v166 & 1 | (2 * v69);
      if ( *v68 != -4096 || v68[1] || *((_BYTE *)v68 + 32) || v68[5] )
        --HIDWORD(v166);
      *v68 = v157;
      *(__m128i *)(v68 + 1) = _mm_loadu_si128(&v158);
      *(__m128i *)(v68 + 3) = _mm_loadu_si128(&v159);
      v70 = v160;
      *((_DWORD *)v68 + 12) = 0;
      v68[5] = v70;
    }
    ++v61;
    ++*((_DWORD *)v68 + 12);
  }
  while ( v131 != v61 );
  v157 = 0;
  v158.m128i_i64[0] = (__int64)&v159.m128i_i64[1];
  v76 = v166 & 1;
  v159.m128i_i8[4] = 1;
  v158.m128i_i64[1] = 4;
  v159.m128i_i32[0] = 0;
  if ( (unsigned int)v166 >> 1 )
  {
    if ( v76 )
    {
      v77 = (__int64 *)&v167;
      v78 = 28;
    }
    else
    {
      v77 = v167;
      v78 = 7LL * v168;
    }
    v148 = (const __m128i *)v77;
    v149 = (const __m128i *)&v77[v78];
    v146 = (__int64 *)&v165;
    v147 = v165;
    sub_F15D40((__int64)&v146);
    v79 = v148;
    v76 = v166 & 1;
  }
  else
  {
    if ( v76 )
    {
      v119 = (__int64 *)&v167;
      v120 = 28;
    }
    else
    {
      v30 = v168;
      v119 = v167;
      v120 = 7LL * v168;
    }
    v79 = (const __m128i *)&v119[v120];
    v148 = v79;
    v146 = (__int64 *)&v165;
    v147 = v165;
    v149 = v79;
  }
  if ( v76 )
  {
    v80 = (__int64 *)&v167;
    v81 = 28;
  }
  else
  {
    v80 = v167;
    v81 = 7LL * v168;
  }
  v82 = (const __m128i *)&v80[v81];
  if ( v79 != v82 )
  {
    while ( 1 )
    {
      v150 = _mm_loadu_si128(v79);
      v151 = _mm_loadu_si128(v79 + 1);
      v152 = _mm_loadu_si128(v79 + 2);
      v153 = v79[3].m128i_i64[0];
      if ( (unsigned int)v153 <= 1 )
        goto LABEL_102;
      v67 = (unsigned __int8)sub_F15DB0((__int64)&v161, v150.m128i_i64, &v139) == 0;
      v88 = v139;
      if ( v67 )
        break;
LABEL_116:
      *(_QWORD *)(v88 + 48) = 0;
      v30 = v150.m128i_i64[0];
      if ( !v159.m128i_i8[4] )
        goto LABEL_138;
      v92 = (_QWORD *)v158.m128i_i64[0];
      v85 = v158.m128i_u32[3];
      v84 = (_QWORD *)(v158.m128i_i64[0] + 8LL * v158.m128i_u32[3]);
      if ( (_QWORD *)v158.m128i_i64[0] != v84 )
      {
        while ( v150.m128i_i64[0] != *v92 )
        {
          if ( v84 == ++v92 )
            goto LABEL_120;
        }
        goto LABEL_102;
      }
LABEL_120:
      if ( v158.m128i_i32[3] < (unsigned __int32)v158.m128i_i32[2] )
      {
        ++v158.m128i_i32[3];
        *v84 = v150.m128i_i64[0];
        ++v157;
      }
      else
      {
LABEL_138:
        sub_C8CC70((__int64)&v157, v150.m128i_i64[0], (__int64)v84, v85, v86, v87);
      }
LABEL_102:
      v83 = (const __m128i *)((char *)v148 + 56);
      v79 = v149;
      v148 = v83;
      if ( v83 == v149 )
        goto LABEL_106;
      while ( v83->m128i_i64[0] != -4096 )
      {
        if ( v83->m128i_i64[0] != -8192
          || v83->m128i_i64[1]
          || !v83[2].m128i_i8[0]
          || v83[1].m128i_i64[0]
          || v83[1].m128i_i64[1] )
        {
          goto LABEL_105;
        }
LABEL_135:
        if ( v83[2].m128i_i64[1] )
          goto LABEL_105;
        v83 = (const __m128i *)((char *)v83 + 56);
        v148 = v83;
        if ( v83 == v149 )
          goto LABEL_106;
      }
      if ( !v83->m128i_i64[1] && !v83[2].m128i_i8[0] )
        goto LABEL_135;
LABEL_105:
      v79 = v148;
LABEL_106:
      if ( v82 == v79 )
        goto LABEL_144;
    }
    ++v161;
    v143 = (__int64 *)v139;
    v89 = ((unsigned int)v162 >> 1) + 1;
    if ( (v162 & 1) != 0 )
    {
      v91 = 12;
      v90 = 4;
    }
    else
    {
      v90 = v164;
      v91 = 3 * v164;
    }
    if ( v91 <= 4 * v89 )
    {
      v90 *= 2;
    }
    else if ( v90 - (v89 + HIDWORD(v162)) > v90 >> 3 )
    {
      goto LABEL_113;
    }
    sub_F1D1B0((__int64)&v161, v90);
    sub_F15DB0((__int64)&v161, v150.m128i_i64, &v143);
    v88 = (__int64)v143;
    v89 = ((unsigned int)v162 >> 1) + 1;
LABEL_113:
    v85 = v162 & 1;
    LODWORD(v162) = v85 | (2 * v89);
    if ( *(_QWORD *)v88 != -4096 || *(_QWORD *)(v88 + 8) || *(_BYTE *)(v88 + 32) || *(_QWORD *)(v88 + 40) )
      --HIDWORD(v162);
    *(_QWORD *)v88 = v150.m128i_i64[0];
    *(__m128i *)(v88 + 8) = _mm_loadu_si128((const __m128i *)&v150.m128i_u64[1]);
    *(__m128i *)(v88 + 24) = _mm_loadu_si128((const __m128i *)&v151.m128i_u64[1]);
    v84 = (_QWORD *)v152.m128i_i64[1];
    *(_QWORD *)(v88 + 48) = 0;
    *(_QWORD *)(v88 + 40) = v84;
    goto LABEL_116;
  }
LABEL_144:
  v95 = v158.m128i_i64[0];
  if ( v159.m128i_i8[4] )
  {
    v125 = v158.m128i_i64[0] + 8LL * v158.m128i_u32[3];
    if ( v158.m128i_i64[0] == v125 )
      goto LABEL_150;
  }
  else
  {
    v125 = v158.m128i_i64[0] + 8LL * v158.m128i_u32[2];
    if ( v158.m128i_i64[0] == v125 )
    {
LABEL_202:
      _libc_free(v158.m128i_i64[0], v30);
      goto LABEL_150;
    }
  }
  v30 = v125;
  while ( *(_QWORD *)v95 >= 0xFFFFFFFFFFFFFFFELL )
  {
    v95 += 8;
    if ( v125 == v95 )
      goto LABEL_149;
  }
  v124 = v95;
  if ( v95 == v125 )
  {
LABEL_149:
    if ( !v159.m128i_i8[4] )
      goto LABEL_202;
    goto LABEL_150;
  }
  v128 = *(_QWORD *)v95;
  v96 = *(_QWORD *)(*(_QWORD *)v95 + 64LL);
  if ( !v96 )
    goto LABEL_213;
  while ( 2 )
  {
    v98 = (_QWORD *)sub_B14240(v96);
    for ( i = v97; v98 != v97; v98 = (_QWORD *)v98[1] )
    {
      if ( !*((_BYTE *)v98 + 32) )
        break;
    }
LABEL_159:
    if ( v98 != i )
    {
      v127 = v98;
      m = (unsigned __int64)i;
      do
      {
        for ( j = *(_QWORD *)m & 0xFFFFFFFFFFFFFFF8LL; *(_BYTE *)(j + 32); j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
          ;
        v102 = *(__int64 **)(j + 24);
        v146 = v102;
        if ( v102 )
          sub_B96E90((__int64)&v146, (__int64)v102, 1);
        v103 = sub_B10CD0((__int64)&v146);
        v104 = *(_BYTE *)(v103 - 16);
        if ( (v104 & 2) != 0 )
        {
          if ( *(_DWORD *)(v103 - 24) != 2 )
            goto LABEL_167;
          v116 = *(_QWORD *)(v103 - 32);
        }
        else
        {
          if ( ((*(_WORD *)(v103 - 16) >> 6) & 0xF) != 2 )
          {
LABEL_167:
            v134 = 0;
            goto LABEL_168;
          }
          v116 = v103 - 16 - 8LL * ((v104 >> 2) & 0xF);
        }
        v134 = *(__int64 **)(v116 + 8);
LABEL_168:
        v105 = 0;
        v106 = sub_B11F60(j + 80);
        v138 = sub_B12000(j + 72);
        if ( v106 )
        {
          sub_AF47B0((__int64)&v150.m128i_i64[1], *(unsigned __int64 **)(v106 + 16), *(unsigned __int64 **)(v106 + 24));
          v105 = v151.m128i_i8[8];
        }
        if ( v146 )
          sub_B91220((__int64)&v146, (__int64)v146);
        v129 = v150.m128i_i64[1];
        v130 = v151.m128i_i64[0];
        if ( (v162 & 1) != 0 )
        {
          v107 = (__int64 *)&v163;
          v108 = 3;
          if ( !v105 )
            goto LABEL_174;
LABEL_181:
          LODWORD(v139) = v151.m128i_u16[0] | (v150.m128i_i32[2] << 16);
          goto LABEL_175;
        }
        v107 = v163;
        v114 = v164;
        v115 = v163;
        if ( v164 )
        {
          v108 = v164 - 1;
          if ( v105 )
            goto LABEL_181;
LABEL_174:
          LODWORD(v139) = 0;
LABEL_175:
          v132 = v107;
          v146 = v134;
          v143 = (__int64 *)v138;
          v109 = sub_F11290((__int64 *)&v143, &v139, (__int64 *)&v146);
          v110 = 1;
          v126 = (unsigned __int64)(((unsigned int)v128 >> 9) ^ ((unsigned int)v128 >> 4)) << 32;
          for ( k = v108 & (((0xBF58476D1CE4E5B9LL * (v126 | v109)) >> 31) ^ (484763065 * (v126 | v109))); ; k = v108 & v113 )
          {
            v112 = &v132[7 * k];
            if ( v128 == *v112
              && v138 == v112[1]
              && v105 == *((_BYTE *)v112 + 32)
              && (!v105 || v112[2] == v129 && v112[3] == v130)
              && v134 == (__int64 *)v112[5] )
            {
              if ( (v162 & 1) != 0 )
              {
                v115 = (__int64 *)&v163;
                v117 = 28;
                goto LABEL_192;
              }
              v115 = v163;
              goto LABEL_204;
            }
            if ( *v112 == -4096 && !v112[1] && !*((_BYTE *)v112 + 32) && !v112[5] )
              break;
            v113 = v110 + k;
            ++v110;
          }
          if ( (v162 & 1) != 0 )
            goto LABEL_195;
          v115 = v163;
          v114 = v164;
        }
        v112 = &v115[7 * v114];
LABEL_204:
        v117 = 7LL * v164;
LABEL_192:
        if ( v112 != &v115[v117] && !v112[6] )
          v112[6] = j;
LABEL_195:
        for ( m = *(_QWORD *)m & 0xFFFFFFFFFFFFFFF8LL; *(_BYTE *)(m + 32); m = *(_QWORD *)m & 0xFFFFFFFFFFFFFFF8LL )
          ;
      }
      while ( (_QWORD *)m != v127 );
    }
    v30 = v125;
    v118 = v124 + 8;
    if ( v124 + 8 != v125 )
    {
      while ( *(_QWORD *)v118 >= 0xFFFFFFFFFFFFFFFELL )
      {
        v118 += 8;
        if ( v125 == v118 )
          goto LABEL_201;
      }
      v124 = v118;
      v128 = *(_QWORD *)v118;
      if ( v118 != v125 )
      {
        v96 = *(_QWORD *)(*(_QWORD *)v118 + 64LL);
        if ( v96 )
          continue;
LABEL_213:
        i = &qword_4F81430[1];
        v98 = &qword_4F81430[1];
        goto LABEL_159;
      }
    }
    break;
  }
LABEL_201:
  if ( !v159.m128i_i8[4] )
    goto LABEL_202;
LABEL_150:
  if ( (v166 & 1) == 0 )
  {
    v30 = 56LL * v168;
    sub_C7D6A0((__int64)v167, v30, 8);
  }
  v32 = (unsigned int)v155;
LABEL_20:
  v34 = (__int64 *)src;
  v169 = 0;
  v143 = (__int64 *)v145;
  v144 = 0x200000000LL;
  v170 = 0;
  v165 = (__int64 *)&v167;
  v166 = 0x400000000LL;
  v171 = &v169;
  v172 = &v169;
  result = (__int64)src + 8 * v32;
  v173 = 0;
  v135 = (__int64 *)result;
  if ( (void *)result == src )
    goto LABEL_52;
  while ( 2 )
  {
    while ( 2 )
    {
      v44 = *v34;
      if ( !*(_BYTE *)(*v34 + 64) )
        goto LABEL_26;
      v45 = *(_QWORD *)(v44 + 24);
      v157 = v45;
      if ( v45 )
        sub_B96E90((__int64)&v157, v45, 1);
      v46 = sub_B10CD0((__int64)&v157);
      v47 = *(_BYTE *)(v46 - 16);
      if ( (v47 & 2) != 0 )
      {
        if ( *(_DWORD *)(v46 - 24) != 2 )
          goto LABEL_32;
        v60 = *(_QWORD *)(v46 - 32);
LABEL_64:
        v48 = *(_QWORD *)(v60 + 8);
      }
      else
      {
        if ( ((*(_WORD *)(v46 - 16) >> 6) & 0xF) == 2 )
        {
          v60 = v46 - 16 - 8LL * ((v47 >> 2) & 0xF);
          goto LABEL_64;
        }
LABEL_32:
        v48 = 0;
      }
      v49 = sub_B11F60(v44 + 80);
      v150.m128i_i64[0] = sub_B12000(v44 + 72);
      if ( v49 )
        sub_AF47B0((__int64)&v150.m128i_i64[1], *(unsigned __int64 **)(v49 + 16), *(unsigned __int64 **)(v49 + 24));
      else
        v151.m128i_i8[8] = 0;
      v152.m128i_i64[0] = v48;
      if ( v157 )
        sub_B91220((__int64)&v157, v157);
      if ( (unsigned int)v162 >> 1 )
      {
        v36 = sub_B140A0(v44);
        v37 = _mm_loadu_si128(&v150);
        v38 = _mm_loadu_si128(&v151);
        v157 = v36;
        v30 = (__int64)&v161;
        v158 = v37;
        v160 = v152.m128i_i64[0];
        v159 = v38;
        sub_F11B40(&v146, &v161, (__int64)&v157);
        v41 = (__int64)v148;
        if ( (v162 & 1) != 0 )
        {
          v42 = (__int64 *)&v163;
          v43 = 28;
        }
        else
        {
          v30 = v164;
          v42 = v163;
          v43 = 7LL * v164;
        }
        if ( v148 != (const __m128i *)&v42[v43] && v148[3].m128i_i64[0] != v44 )
          goto LABEL_26;
      }
      v30 = (__int64)&v165;
      sub_F1C2C0((__int64)&v157, (__int64)&v165, &v150, v41, v39, v40);
      if ( !v158.m128i_i8[8] || *(_BYTE *)(v44 + 64) == 2 )
      {
LABEL_26:
        if ( v135 == ++v34 )
          goto LABEL_44;
        continue;
      }
      break;
    }
    v50 = sub_B13070(v44);
    v52 = (unsigned int)v144;
    v53 = v50;
    v54 = v144;
    if ( (unsigned int)v144 >= (unsigned __int64)HIDWORD(v144) )
    {
      if ( HIDWORD(v144) < (unsigned __int64)(unsigned int)v144 + 1 )
      {
        v30 = (__int64)v145;
        sub_C8D5F0((__int64)&v143, v145, (unsigned int)v144 + 1LL, 8u, (unsigned int)v144 + 1LL, v51);
        v52 = (unsigned int)v144;
      }
      v143[v52] = v53;
      LODWORD(v144) = v144 + 1;
      goto LABEL_26;
    }
    v55 = &v143[(unsigned int)v144];
    if ( v55 )
    {
      *v55 = v53;
      v54 = v144;
    }
    ++v34;
    LODWORD(v144) = v54 + 1;
    if ( v135 != v34 )
      continue;
    break;
  }
LABEL_44:
  result = (unsigned int)v144;
  if ( (_DWORD)v144 )
  {
    v30 = 0;
    sub_F54050(a2, 0, 0, v140, (unsigned int)v141);
    v56 = v143;
    result = (unsigned int)v144;
    v57 = &v143[(unsigned int)v144];
    if ( v57 != v143 )
    {
      do
      {
        v30 = *v56;
        if ( !a3 )
          BUG();
        ++v56;
        result = sub_AA8770(*(_QWORD *)(a3 + 16), v30, a3, a4);
      }
      while ( v57 != v56 );
    }
    for ( n = v170; n; result = j_j___libc_free_0(v59, 72) )
    {
      sub_F078B0(*(_QWORD *)(n + 24));
      v59 = n;
      n = *(_QWORD *)(n + 16);
      v30 = 72;
    }
  }
  else
  {
    for ( ii = v170; ii; result = j_j___libc_free_0(v94, 72) )
    {
      sub_F078B0(*(_QWORD *)(ii + 24));
      v94 = ii;
      ii = *(_QWORD *)(ii + 16);
      v30 = 72;
    }
  }
  if ( v165 != (__int64 *)&v167 )
    result = _libc_free(v165, v30);
LABEL_52:
  if ( v143 != (__int64 *)v145 )
    result = _libc_free(v143, v30);
  if ( (v162 & 1) == 0 )
  {
    v30 = 56LL * v164;
    result = sub_C7D6A0((__int64)v163, v30, 8);
  }
  if ( src != v156 )
    result = _libc_free(src, v30);
  if ( v140 != (__int64 *)v142 )
    return _libc_free(v140, v30);
  return result;
}
