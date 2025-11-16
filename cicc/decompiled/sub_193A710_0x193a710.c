// Function: sub_193A710
// Address: 0x193a710
//
__int64 __fastcall sub_193A710(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  char v5; // r13
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v9; // r14
  const __m128i *v10; // r12
  __int64 v11; // rbx
  __int64 v12; // r13
  const __m128i *v13; // r14
  __int64 v14; // rax
  __m128i v15; // xmm2
  __m128i *v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  const __m128i *v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // rax
  __m128i *v22; // rdi
  const char *v23; // rax
  const __m128i *v24; // rax
  __m128i v25; // xmm0
  __m128i *v26; // rax
  int v27; // ebx
  __m128i *v28; // r13
  size_t v29; // r14
  unsigned __int64 v30; // rbx
  __m128i *v31; // r12
  const __m128i *v32; // rbx
  const __m128i *v33; // rdi
  __int64 v34; // rbx
  unsigned int v35; // r12d
  __int64 *v36; // r15
  __int64 v37; // r14
  unsigned int v38; // eax
  int v39; // eax
  __int64 v40; // rax
  char *v41; // r15
  __int64 v42; // rcx
  __int64 v43; // rax
  __int64 v44; // rcx
  unsigned int v45; // edx
  int v46; // eax
  __int64 *v47; // r8
  unsigned int v48; // edx
  int v49; // eax
  __int64 *v50; // r8
  unsigned int v51; // edx
  int v52; // eax
  __int64 *v53; // r8
  unsigned int v54; // edx
  int v55; // eax
  __int64 *v56; // r8
  const __m128i *v57; // rbx
  int v58; // eax
  __m128i *v59; // rdx
  unsigned int v60; // eax
  char *v61; // r12
  __int64 v62; // rbx
  __m128i *v63; // rdx
  const __m128i *v64; // rbx
  const char *v65; // r12
  const char *v66; // rbx
  __int64 v67; // rdx
  __int64 v68; // rdi
  __int64 v69; // rax
  __int64 *v70; // r8
  unsigned int v71; // r12d
  int v72; // ebx
  __m128i *v73; // rbx
  __int64 v74; // rbx
  __int64 v75; // rax
  __int64 v76; // r12
  int v77; // r13d
  __int64 *v78; // r8
  unsigned int v79; // edx
  int v80; // eax
  __int64 *v81; // r8
  unsigned int v82; // edx
  int v83; // eax
  __int64 *v84; // rax
  __int64 v85; // r14
  _QWORD *v86; // r13
  _QWORD **v87; // rax
  __int16 v88; // r15
  __int64 *v89; // rax
  __int64 v90; // rsi
  int v91; // [rsp+Ch] [rbp-2B4h]
  unsigned __int64 v94; // [rsp+28h] [rbp-298h]
  char *v97; // [rsp+40h] [rbp-280h]
  unsigned __int8 v99; // [rsp+53h] [rbp-26Dh]
  unsigned int v100; // [rsp+54h] [rbp-26Ch]
  char *v101; // [rsp+70h] [rbp-250h]
  int v102; // [rsp+70h] [rbp-250h]
  const void **v103; // [rsp+78h] [rbp-248h]
  unsigned __int64 v104; // [rsp+80h] [rbp-240h]
  unsigned __int64 v105; // [rsp+80h] [rbp-240h]
  unsigned __int64 v106; // [rsp+80h] [rbp-240h]
  unsigned __int64 v107; // [rsp+80h] [rbp-240h]
  unsigned __int64 v108; // [rsp+80h] [rbp-240h]
  unsigned __int64 v109; // [rsp+80h] [rbp-240h]
  unsigned __int64 v110; // [rsp+80h] [rbp-240h]
  unsigned int v111; // [rsp+88h] [rbp-238h]
  int v112; // [rsp+88h] [rbp-238h]
  int v113; // [rsp+88h] [rbp-238h]
  unsigned int v114; // [rsp+88h] [rbp-238h]
  int v115; // [rsp+88h] [rbp-238h]
  unsigned int v116; // [rsp+88h] [rbp-238h]
  int v117; // [rsp+88h] [rbp-238h]
  unsigned int v118; // [rsp+88h] [rbp-238h]
  int v119; // [rsp+88h] [rbp-238h]
  unsigned int v120; // [rsp+88h] [rbp-238h]
  int v121; // [rsp+88h] [rbp-238h]
  __int64 *v122; // [rsp+88h] [rbp-238h]
  __int64 *v123; // [rsp+88h] [rbp-238h]
  __int64 *v124; // [rsp+88h] [rbp-238h]
  __int64 *v125; // [rsp+88h] [rbp-238h]
  unsigned __int64 v126; // [rsp+88h] [rbp-238h]
  unsigned int v127; // [rsp+88h] [rbp-238h]
  int v128; // [rsp+88h] [rbp-238h]
  unsigned int v129; // [rsp+88h] [rbp-238h]
  int v130; // [rsp+88h] [rbp-238h]
  __int64 *v131; // [rsp+88h] [rbp-238h]
  __int64 *v132; // [rsp+88h] [rbp-238h]
  __int64 *v133; // [rsp+88h] [rbp-238h]
  _QWORD *v134; // [rsp+88h] [rbp-238h]
  int v135; // [rsp+9Ch] [rbp-224h] BYREF
  unsigned __int64 v136; // [rsp+A0h] [rbp-220h] BYREF
  unsigned int v137; // [rsp+A8h] [rbp-218h]
  __int64 v138; // [rsp+B0h] [rbp-210h]
  unsigned int v139; // [rsp+B8h] [rbp-208h]
  unsigned __int64 v140; // [rsp+C0h] [rbp-200h] BYREF
  unsigned int v141; // [rsp+C8h] [rbp-1F8h]
  __int64 v142; // [rsp+D0h] [rbp-1F0h]
  unsigned int v143; // [rsp+D8h] [rbp-1E8h]
  unsigned __int64 v144; // [rsp+E0h] [rbp-1E0h] BYREF
  unsigned int v145; // [rsp+E8h] [rbp-1D8h]
  const void *v146; // [rsp+F0h] [rbp-1D0h] BYREF
  unsigned int v147; // [rsp+F8h] [rbp-1C8h]
  void *src; // [rsp+100h] [rbp-1C0h] BYREF
  __int64 v149; // [rsp+108h] [rbp-1B8h]
  _BYTE *v150; // [rsp+110h] [rbp-1B0h] BYREF
  __int64 v151; // [rsp+118h] [rbp-1A8h]
  int v152; // [rsp+120h] [rbp-1A0h]
  _BYTE v153[72]; // [rsp+128h] [rbp-198h] BYREF
  const char *v154; // [rsp+170h] [rbp-150h] BYREF
  __int64 v155; // [rsp+178h] [rbp-148h]
  __int64 v156; // [rsp+180h] [rbp-140h] BYREF
  unsigned int v157; // [rsp+188h] [rbp-138h]
  const char *v158; // [rsp+200h] [rbp-C0h] BYREF
  __int64 v159; // [rsp+208h] [rbp-B8h]
  const void *v160; // [rsp+210h] [rbp-B0h] BYREF
  unsigned int v161; // [rsp+218h] [rbp-A8h]

  if ( *(_BYTE *)(a2 + 16) == 75 )
  {
    v74 = *(_QWORD *)(a2 - 48);
    if ( v74 )
    {
      v75 = *(_QWORD *)(a2 - 24);
      if ( *(_BYTE *)(v75 + 16) == 13 && *(_BYTE *)(a3 + 16) == 75 && v74 == *(_QWORD *)(a3 - 48) )
      {
        v76 = *(_QWORD *)(a3 - 24);
        if ( *(_BYTE *)(v76 + 16) == 13 )
        {
          v77 = *(_WORD *)(a3 + 18) & 0x7FFF;
          sub_158B890((__int64)&v136, *(_WORD *)(a2 + 18) & 0x7FFF, v75 + 24);
          sub_158B890((__int64)&v140, v77, v76 + 24);
          sub_1590E70((__int64)&src, (__int64)&v136);
          sub_1590E70((__int64)&v154, (__int64)&v140);
          sub_158C3A0((__int64)&v158, (__int64)&src, (__int64)&v154);
          sub_1590E70((__int64)&v144, (__int64)&v158);
          if ( v161 > 0x40 && v160 )
            j_j___libc_free_0_0(v160);
          if ( (unsigned int)v159 > 0x40 && v158 )
            j_j___libc_free_0_0(v158);
          if ( v157 > 0x40 && v156 )
            j_j___libc_free_0_0(v156);
          if ( (unsigned int)v155 > 0x40 && v154 )
            j_j___libc_free_0_0(v154);
          if ( (unsigned int)v151 > 0x40 && v150 )
            j_j___libc_free_0_0(v150);
          if ( (unsigned int)v149 > 0x40 && src )
            j_j___libc_free_0_0(src);
          sub_158BE00((__int64)&v158, (__int64)&v136, (__int64)&v140);
          LODWORD(v149) = 1;
          src = 0;
          if ( v145 <= 0x40 )
          {
            if ( (const char *)v144 != v158 )
              goto LABEL_207;
          }
          else if ( !sub_16A5220((__int64)&v144, (const void **)&v158) )
          {
            goto LABEL_207;
          }
          if ( v147 <= 0x40 )
          {
            if ( v146 == v160 )
              goto LABEL_203;
          }
          else if ( sub_16A5220((__int64)&v146, &v160) )
          {
LABEL_203:
            v99 = sub_158A180((__int64)&v144, &v135, (__int64)&src);
            if ( v99 )
            {
              if ( a4 )
              {
                v84 = (__int64 *)sub_16498A0(a2);
                v85 = sub_159C0E0(v84, (__int64)&src);
                LOWORD(v156) = 259;
                v154 = "wide.chk";
                v86 = sub_1648A60(56, 2u);
                if ( v86 )
                {
                  v87 = *(_QWORD ***)v74;
                  v88 = v135;
                  if ( *(_BYTE *)(*(_QWORD *)v74 + 8LL) == 16 )
                  {
                    v134 = v87[4];
                    v89 = (__int64 *)sub_1643320(*v87);
                    v90 = (__int64)sub_16463B0(v89, (unsigned int)v134);
                  }
                  else
                  {
                    v90 = sub_1643320(*v87);
                  }
                  sub_15FEC10((__int64)v86, v90, 51, v88, v74, v85, (__int64)&v154, a4);
                }
                *a5 = (__int64)v86;
              }
              if ( (unsigned int)v149 > 0x40 && src )
                j_j___libc_free_0_0(src);
              if ( v161 > 0x40 && v160 )
                j_j___libc_free_0_0(v160);
              if ( (unsigned int)v159 > 0x40 && v158 )
                j_j___libc_free_0_0(v158);
              if ( v147 > 0x40 && v146 )
                j_j___libc_free_0_0(v146);
              if ( v145 > 0x40 && v144 )
                j_j___libc_free_0_0(v144);
              if ( v143 > 0x40 && v142 )
                j_j___libc_free_0_0(v142);
              if ( v141 > 0x40 && v140 )
                j_j___libc_free_0_0(v140);
              if ( v139 > 0x40 && v138 )
                j_j___libc_free_0_0(v138);
              if ( v137 > 0x40 && v136 )
                j_j___libc_free_0_0(v136);
              return v99;
            }
            if ( (unsigned int)v149 > 0x40 && src )
              j_j___libc_free_0_0(src);
          }
LABEL_207:
          if ( v161 > 0x40 && v160 )
            j_j___libc_free_0_0(v160);
          if ( (unsigned int)v159 > 0x40 && v158 )
            j_j___libc_free_0_0(v158);
          if ( v147 > 0x40 && v146 )
            j_j___libc_free_0_0(v146);
          if ( v145 > 0x40 && v144 )
            j_j___libc_free_0_0(v144);
          if ( v143 > 0x40 && v142 )
            j_j___libc_free_0_0(v142);
          if ( v141 > 0x40 && v140 )
            j_j___libc_free_0_0(v140);
          if ( v139 > 0x40 && v138 )
            j_j___libc_free_0_0(v138);
          if ( v137 > 0x40 && v136 )
            j_j___libc_free_0_0(v136);
        }
      }
    }
  }
  v154 = (const char *)&v156;
  v155 = 0x400000000LL;
  v158 = (const char *)&v160;
  v159 = 0x400000000LL;
  src = 0;
  v149 = (__int64)v153;
  v150 = v153;
  v151 = 8;
  v152 = 0;
  v5 = sub_193A010(a2, (__int64)&v154, (__int64)&src);
  if ( v150 != (_BYTE *)v149 )
    _libc_free((unsigned __int64)v150);
  if ( !v5 )
    goto LABEL_8;
  src = 0;
  v149 = (__int64)v153;
  v150 = v153;
  v151 = 8;
  v152 = 0;
  v99 = sub_193A010(a3, (__int64)&v154, (__int64)&src);
  if ( v150 != (_BYTE *)v149 )
    _libc_free((unsigned __int64)v150);
  if ( !v99 )
    goto LABEL_8;
  v91 = v155;
  if ( !(_DWORD)v155 )
  {
LABEL_121:
    if ( v91 == (_DWORD)v159 )
      goto LABEL_8;
    if ( a4 )
    {
      v65 = v158;
      v66 = &v158[32 * (unsigned int)v159];
      *a5 = 0;
      if ( v66 == v65 )
      {
        v68 = 0;
      }
      else
      {
        do
        {
          sub_1939AB0(a1, *((_QWORD *)v65 + 3), a4);
          v67 = *a5;
          if ( *a5 )
          {
            LOWORD(v150) = 257;
            *a5 = sub_15FB440(26, *((__int64 **)v65 + 3), v67, (__int64)&src, a4);
          }
          else
          {
            *a5 = *((_QWORD *)v65 + 3);
          }
          v65 += 32;
        }
        while ( v66 != v65 );
        v68 = *a5;
      }
      src = "wide.chk";
      LOWORD(v150) = 259;
      sub_164B780(v68, (__int64 *)&src);
    }
    if ( v158 != (const char *)&v160 )
      _libc_free((unsigned __int64)v158);
    if ( v154 != (const char *)&v156 )
      _libc_free((unsigned __int64)v154);
    return v99;
  }
  v9 = (unsigned int)v155;
  while ( 1 )
  {
    v10 = (const __m128i *)v154;
    v11 = *(_QWORD *)v154;
    v12 = *((_QWORD *)v154 + 2);
    src = &v150;
    v13 = (const __m128i *)&v154[32 * v9];
    v149 = 0x300000000LL;
    do
    {
      while ( v11 != v10->m128i_i64[0] || v12 != v10[1].m128i_i64[0] )
      {
        v10 += 2;
        if ( v13 == v10 )
          goto LABEL_25;
      }
      v14 = (unsigned int)v149;
      if ( (unsigned int)v149 >= HIDWORD(v149) )
      {
        sub_16CD150((__int64)&src, &v150, 0, 32, v6, v7);
        v14 = (unsigned int)v149;
      }
      v15 = _mm_loadu_si128(v10);
      v10 += 2;
      v16 = (__m128i *)((char *)src + 32 * v14);
      *v16 = v15;
      v16[1] = _mm_loadu_si128(v10 - 1);
      LODWORD(v149) = v149 + 1;
    }
    while ( v13 != v10 );
LABEL_25:
    v17 = (unsigned __int64)v154;
    v18 = 32LL * (unsigned int)v155;
    v19 = (const __m128i *)&v154[v18];
    v20 = v18 >> 5;
    v21 = v18 >> 7;
    if ( v21 )
    {
      v22 = (__m128i *)v154;
      v23 = &v154[128 * v21];
      while ( v11 != v22->m128i_i64[0] || v12 != v22[1].m128i_i64[0] )
      {
        if ( v11 == v22[2].m128i_i64[0] && v12 == v22[3].m128i_i64[0] )
        {
          v22 += 2;
          goto LABEL_33;
        }
        if ( v11 == v22[4].m128i_i64[0] && v12 == v22[5].m128i_i64[0] )
        {
          v22 += 4;
          goto LABEL_33;
        }
        if ( v11 == v22[6].m128i_i64[0] && v12 == v22[7].m128i_i64[0] )
        {
          v22 += 6;
          goto LABEL_33;
        }
        v22 += 8;
        if ( v22 == (__m128i *)v23 )
        {
          v20 = ((char *)v19 - (char *)v22) >> 5;
          goto LABEL_149;
        }
      }
      goto LABEL_33;
    }
    v22 = (__m128i *)v154;
LABEL_149:
    if ( v20 == 2 )
      goto LABEL_171;
    if ( v20 == 3 )
    {
      if ( v11 == v22->m128i_i64[0] && v12 == v22[1].m128i_i64[0] )
        goto LABEL_33;
      v22 += 2;
LABEL_171:
      if ( v11 == v22->m128i_i64[0] && v12 == v22[1].m128i_i64[0] )
        goto LABEL_33;
      v22 += 2;
      goto LABEL_173;
    }
    if ( v20 != 1 )
      goto LABEL_152;
LABEL_173:
    if ( v11 != v22->m128i_i64[0] || v12 != v22[1].m128i_i64[0] )
    {
LABEL_152:
      v73 = (__m128i *)v19;
      goto LABEL_42;
    }
LABEL_33:
    if ( v19 == v22 )
      goto LABEL_152;
    v24 = v22 + 2;
    if ( v19 == &v22[2] )
    {
      v73 = v22;
    }
    else
    {
      do
      {
        while ( v11 != v24->m128i_i64[0] || v12 != v24[1].m128i_i64[0] )
        {
          v25 = _mm_loadu_si128(v24);
          v24 += 2;
          v22 += 2;
          v22[-2] = v25;
          v22[-1] = _mm_loadu_si128(v24 - 1);
          if ( v19 == v24 )
            goto LABEL_40;
        }
        v24 += 2;
      }
      while ( v19 != v24 );
LABEL_40:
      v17 = (unsigned __int64)v154;
      v6 = &v154[32 * (unsigned int)v155] - (const char *)v19;
      v73 = (__m128i *)((char *)v22 + v6);
      if ( v19 != (const __m128i *)&v154[32 * (unsigned int)v155] )
      {
        memmove(v22, v19, &v154[32 * (unsigned int)v155] - (const char *)v19);
        v17 = (unsigned __int64)v154;
      }
    }
LABEL_42:
    v26 = v73;
    v27 = v149;
    v28 = (__m128i *)src;
    LODWORD(v155) = (__int64)((__int64)v26->m128i_i64 - v17) >> 5;
    v29 = 32LL * (unsigned int)v149;
    if ( (unsigned int)v149 <= 2uLL )
    {
      v69 = (unsigned int)v159;
      if ( (unsigned int)v149 > HIDWORD(v159) - (unsigned __int64)(unsigned int)v159 )
      {
        sub_16CD150((__int64)&v158, &v160, (unsigned int)v159 + (unsigned __int64)(unsigned int)v149, 32, v6, v7);
        v69 = (unsigned int)v159;
      }
      if ( v29 )
      {
        memcpy((void *)&v158[32 * v69], v28, v29);
        LODWORD(v69) = v159;
      }
      LODWORD(v159) = v27 + v69;
      goto LABEL_118;
    }
    _BitScanReverse64(&v30, (unsigned int)v149);
    v31 = (__m128i *)((char *)src + v29);
    sub_19395B0((__m128i *)src, (__m128i *)((char *)src + v29), 2LL * (int)(63 - (v30 ^ 0x3F)), v20, v6, v7);
    if ( v29 <= 0x200 )
    {
      sub_19399E0(v28, &v28[v29 / 0x10]);
    }
    else
    {
      v32 = v28 + 32;
      sub_19399E0(v28, v28 + 32);
      if ( v31 != &v28[32] )
      {
        do
        {
          v33 = v32;
          v32 += 2;
          sub_1939510(v33);
        }
        while ( v31 != v32 );
      }
    }
    v34 = *((_QWORD *)src + 4 * (unsigned int)v149 - 3);
    v36 = (__int64 *)(*((_QWORD *)src + 1) + 24LL);
    v103 = (const void **)(v34 + 24);
    v141 = *(_DWORD *)(v34 + 32);
    v35 = v141;
    v37 = 1LL << ((unsigned __int8)v141 - 1);
    if ( v141 <= 0x40 )
    {
      v140 = *(_QWORD *)(v34 + 24);
      sub_16A7590((__int64)&v140, v36);
      v38 = v141;
      v144 = 0;
      v141 = 0;
      v111 = v38;
      v137 = v38;
      v145 = v35;
      v104 = v140;
      v136 = v140;
LABEL_48:
      v144 |= v37;
      v39 = sub_16A9900((__int64)&v136, &v144);
      goto LABEL_49;
    }
    sub_16A4FD0((__int64)&v140, v103);
    sub_16A7590((__int64)&v140, v36);
    v145 = v35;
    v111 = v141;
    v137 = v141;
    v141 = 0;
    v104 = v140;
    v136 = v140;
    sub_16A4EF0((__int64)&v144, 0, 0);
    if ( v145 <= 0x40 )
      goto LABEL_48;
    *(_QWORD *)(v144 + 8LL * ((v35 - 1) >> 6)) |= v37;
    v39 = sub_16A9900((__int64)&v136, &v144);
    if ( v145 > 0x40 && v144 )
    {
      v102 = v39;
      j_j___libc_free_0_0(v144);
      v39 = v102;
    }
LABEL_49:
    if ( v111 > 0x40 && v104 )
    {
      v112 = v39;
      j_j___libc_free_0_0(v104);
      v39 = v112;
    }
    if ( v141 > 0x40 && v140 )
    {
      v113 = v39;
      j_j___libc_free_0_0(v140);
      v39 = v113;
    }
    if ( v39 > 0 )
      goto LABEL_145;
    v145 = *(_DWORD *)(v34 + 32);
    if ( v145 > 0x40 )
      sub_16A4FD0((__int64)&v144, v103);
    else
      v144 = *(_QWORD *)(v34 + 24);
    sub_16A7590((__int64)&v144, v36);
    v100 = v145;
    v137 = v145;
    v94 = v144;
    v136 = v144;
    if ( v145 <= 0x40 )
    {
      if ( !v144 )
        goto LABEL_145;
    }
    else if ( v100 == (unsigned int)sub_16A57B0((__int64)&v136) )
    {
      goto LABEL_258;
    }
    v40 = 32LL * (unsigned int)v149;
    v41 = (char *)src + 32;
    v97 = (char *)src + v40;
    v42 = v40 - 32;
    v43 = (v40 - 32) >> 7;
    v44 = v42 >> 5;
    if ( v43 > 0 )
    {
      v101 = (char *)src + 128 * v43 + 32;
      while ( 1 )
      {
        v56 = (__int64 *)(*((_QWORD *)v41 + 1) + 24LL);
        v145 = *(_DWORD *)(v34 + 32);
        if ( v145 <= 0x40 )
        {
          v144 = *(_QWORD *)(v34 + 24);
        }
        else
        {
          v122 = v56;
          sub_16A4FD0((__int64)&v144, v103);
          v56 = v122;
        }
        sub_16A7590((__int64)&v144, v56);
        v45 = v145;
        v145 = 0;
        v141 = v45;
        v114 = v45;
        v140 = v144;
        v105 = v144;
        v46 = sub_16A9900((__int64)&v140, &v136);
        if ( v114 > 0x40 )
        {
          v6 = v105;
          if ( v105 )
          {
            v115 = v46;
            j_j___libc_free_0_0(v105);
            v46 = v115;
            if ( v145 > 0x40 )
            {
              if ( v144 )
              {
                j_j___libc_free_0_0(v144);
                v46 = v115;
              }
            }
          }
        }
        if ( v46 >= 0 )
          goto LABEL_106;
        v47 = (__int64 *)(*((_QWORD *)v41 + 5) + 24LL);
        v145 = *(_DWORD *)(v34 + 32);
        if ( v145 > 0x40 )
        {
          v123 = v47;
          sub_16A4FD0((__int64)&v144, v103);
          v47 = v123;
        }
        else
        {
          v144 = *(_QWORD *)(v34 + 24);
        }
        sub_16A7590((__int64)&v144, v47);
        v48 = v145;
        v145 = 0;
        v141 = v48;
        v116 = v48;
        v140 = v144;
        v106 = v144;
        v49 = sub_16A9900((__int64)&v140, &v136);
        if ( v116 > 0x40 )
        {
          v6 = v106;
          if ( v106 )
          {
            v117 = v49;
            j_j___libc_free_0_0(v106);
            v49 = v117;
            if ( v145 > 0x40 )
            {
              if ( v144 )
              {
                j_j___libc_free_0_0(v144);
                v49 = v117;
              }
            }
          }
        }
        if ( v49 >= 0 )
        {
          v41 += 32;
          goto LABEL_106;
        }
        v50 = (__int64 *)(*((_QWORD *)v41 + 9) + 24LL);
        v145 = *(_DWORD *)(v34 + 32);
        if ( v145 > 0x40 )
        {
          v124 = v50;
          sub_16A4FD0((__int64)&v144, v103);
          v50 = v124;
        }
        else
        {
          v144 = *(_QWORD *)(v34 + 24);
        }
        sub_16A7590((__int64)&v144, v50);
        v51 = v145;
        v145 = 0;
        v141 = v51;
        v118 = v51;
        v140 = v144;
        v107 = v144;
        v52 = sub_16A9900((__int64)&v140, &v136);
        if ( v118 > 0x40 )
        {
          v6 = v107;
          if ( v107 )
          {
            v119 = v52;
            j_j___libc_free_0_0(v107);
            v52 = v119;
            if ( v145 > 0x40 )
            {
              if ( v144 )
              {
                j_j___libc_free_0_0(v144);
                v52 = v119;
              }
            }
          }
        }
        if ( v52 >= 0 )
        {
          v41 += 64;
          goto LABEL_106;
        }
        v53 = (__int64 *)(*((_QWORD *)v41 + 13) + 24LL);
        v145 = *(_DWORD *)(v34 + 32);
        if ( v145 > 0x40 )
        {
          v125 = v53;
          sub_16A4FD0((__int64)&v144, v103);
          v53 = v125;
        }
        else
        {
          v144 = *(_QWORD *)(v34 + 24);
        }
        sub_16A7590((__int64)&v144, v53);
        v54 = v145;
        v145 = 0;
        v141 = v54;
        v120 = v54;
        v140 = v144;
        v108 = v144;
        v55 = sub_16A9900((__int64)&v140, &v136);
        if ( v120 > 0x40 )
        {
          v6 = v108;
          if ( v108 )
          {
            v121 = v55;
            j_j___libc_free_0_0(v108);
            v55 = v121;
            if ( v145 > 0x40 )
            {
              if ( v144 )
              {
                j_j___libc_free_0_0(v144);
                v55 = v121;
              }
            }
          }
        }
        if ( v55 >= 0 )
        {
          v41 += 96;
          goto LABEL_106;
        }
        v41 += 128;
        if ( v101 == v41 )
        {
          v44 = (v97 - v41) >> 5;
          break;
        }
      }
    }
    if ( v44 == 2 )
      goto LABEL_240;
    if ( v44 == 3 )
    {
      v78 = (__int64 *)(*((_QWORD *)v41 + 1) + 24LL);
      v145 = *(_DWORD *)(v34 + 32);
      if ( v145 > 0x40 )
      {
        v131 = v78;
        sub_16A4FD0((__int64)&v144, v103);
        v78 = v131;
      }
      else
      {
        v144 = *(_QWORD *)(v34 + 24);
      }
      sub_16A7590((__int64)&v144, v78);
      v79 = v145;
      v145 = 0;
      v141 = v79;
      v127 = v79;
      v140 = v144;
      v109 = v144;
      v80 = sub_16A9900((__int64)&v140, &v136);
      if ( v127 > 0x40 )
      {
        v6 = v109;
        if ( v109 )
        {
          v128 = v80;
          j_j___libc_free_0_0(v109);
          v80 = v128;
          if ( v145 > 0x40 )
          {
            if ( v144 )
            {
              j_j___libc_free_0_0(v144);
              v80 = v128;
            }
          }
        }
      }
      if ( v80 >= 0 )
        goto LABEL_106;
      v41 += 32;
LABEL_240:
      v81 = (__int64 *)(*((_QWORD *)v41 + 1) + 24LL);
      v145 = *(_DWORD *)(v34 + 32);
      if ( v145 > 0x40 )
      {
        v133 = v81;
        sub_16A4FD0((__int64)&v144, v103);
        v81 = v133;
      }
      else
      {
        v144 = *(_QWORD *)(v34 + 24);
      }
      sub_16A7590((__int64)&v144, v81);
      v82 = v145;
      v145 = 0;
      v141 = v82;
      v129 = v82;
      v140 = v144;
      v110 = v144;
      v83 = sub_16A9900((__int64)&v140, &v136);
      if ( v129 > 0x40 )
      {
        v6 = v110;
        if ( v110 )
        {
          v130 = v83;
          j_j___libc_free_0_0(v110);
          v83 = v130;
          if ( v145 > 0x40 )
          {
            if ( v144 )
            {
              j_j___libc_free_0_0(v144);
              v83 = v130;
            }
          }
        }
      }
      if ( v83 >= 0 )
        goto LABEL_106;
      v41 += 32;
      goto LABEL_158;
    }
    if ( v44 != 1 )
      goto LABEL_107;
LABEL_158:
    v70 = (__int64 *)(*((_QWORD *)v41 + 1) + 24LL);
    v145 = *(_DWORD *)(v34 + 32);
    if ( v145 > 0x40 )
    {
      v132 = v70;
      sub_16A4FD0((__int64)&v144, v103);
      v70 = v132;
    }
    else
    {
      v144 = *(_QWORD *)(v34 + 24);
    }
    sub_16A7590((__int64)&v144, v70);
    v71 = v145;
    v145 = 0;
    v140 = v144;
    v141 = v71;
    v126 = v144;
    v72 = sub_16A9900((__int64)&v140, &v136);
    if ( v71 > 0x40 )
    {
      v6 = v126;
      if ( v126 )
      {
        j_j___libc_free_0_0(v126);
        if ( v145 > 0x40 )
        {
          if ( v144 )
            j_j___libc_free_0_0(v144);
        }
      }
    }
    if ( v72 < 0 )
      goto LABEL_107;
LABEL_106:
    if ( v97 != v41 )
      break;
LABEL_107:
    v57 = (const __m128i *)src;
    v58 = v159;
    if ( (unsigned int)v159 >= HIDWORD(v159) )
    {
      sub_16CD150((__int64)&v158, &v160, 0, 32, v6, v7);
      v58 = v159;
    }
    v59 = (__m128i *)&v158[32 * v58];
    if ( v59 )
    {
      *v59 = _mm_loadu_si128(v57);
      v59[1] = _mm_loadu_si128(v57 + 1);
      v58 = v159;
    }
    v60 = v58 + 1;
    v61 = (char *)src;
    v62 = (unsigned int)v149;
    LODWORD(v159) = v60;
    if ( HIDWORD(v159) <= v60 )
    {
      sub_16CD150((__int64)&v158, &v160, 0, 32, v6, v7);
      v60 = v159;
    }
    v63 = (__m128i *)&v158[32 * v60];
    if ( v63 )
    {
      v64 = (const __m128i *)&v61[32 * v62];
      *v63 = _mm_loadu_si128(v64 - 2);
      v63[1] = _mm_loadu_si128(v64 - 1);
      v60 = v159;
    }
    LODWORD(v159) = v60 + 1;
    if ( v100 > 0x40 && v94 )
      j_j___libc_free_0_0(v94);
LABEL_118:
    if ( src != &v150 )
      _libc_free((unsigned __int64)src);
    v9 = (unsigned int)v155;
    if ( !(_DWORD)v155 )
      goto LABEL_121;
  }
  if ( v100 <= 0x40 )
    goto LABEL_145;
LABEL_258:
  if ( v94 )
    j_j___libc_free_0_0(v94);
LABEL_145:
  if ( src != &v150 )
    _libc_free((unsigned __int64)src);
LABEL_8:
  if ( v158 != (const char *)&v160 )
    _libc_free((unsigned __int64)v158);
  if ( v154 != (const char *)&v156 )
    _libc_free((unsigned __int64)v154);
  if ( a4 )
  {
    sub_1939AB0(a1, a2, a4);
    sub_1939AB0(a1, a3, a4);
    v158 = "wide.chk";
    LOWORD(v160) = 259;
    *a5 = sub_15FB440(26, (__int64 *)a2, a3, (__int64)&v158, a4);
  }
  return 0;
}
