// Function: sub_2407980
// Address: 0x2407980
//
__int64 __fastcall sub_2407980(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 **v5; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 *v14; // rax
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int8 *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  const char *v21; // rax
  const char **v22; // rsi
  __int64 *v23; // rdi
  __int64 v24; // r8
  __int64 v25; // r9
  const __m128i *v26; // rcx
  const __m128i *v27; // rdx
  unsigned __int64 v28; // rbx
  __m128i *v29; // rax
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  const __m128i *v33; // rax
  const __m128i *v34; // rcx
  unsigned __int64 v35; // rbx
  __int64 v36; // rax
  const char *v37; // rdi
  __m128i *v38; // rdx
  const char *v39; // rax
  unsigned __int64 v40; // rcx
  __int64 v41; // rbx
  _QWORD *v42; // rax
  __int64 v43; // rdx
  __int64 *v44; // rsi
  __int64 v45; // rdi
  __int64 v46; // rdx
  _QWORD *v47; // rdx
  const char *v48; // rax
  char v49; // si
  _BYTE *v50; // rdi
  __int64 v51; // r14
  char *v52; // rbx
  __int64 v53; // rdx
  char *v54; // r12
  unsigned __int8 v55; // di
  __int64 v56; // rcx
  _QWORD *v57; // rax
  __int64 v58; // rsi
  __int64 v59; // r8
  __int64 v60; // rsi
  _QWORD *v61; // rsi
  _BYTE *v62; // rdi
  unsigned __int64 v64; // r8
  unsigned int v65; // esi
  __int64 v66; // r11
  int v67; // esi
  __int64 v68; // r9
  __int64 v69; // r8
  int v70; // r11d
  __int64 *v71; // r10
  unsigned int v72; // edx
  __int64 *v73; // rcx
  __int64 v74; // rdi
  int v75; // eax
  __int64 v76; // rax
  __int64 v77; // rbx
  unsigned __int64 v78; // rdx
  unsigned __int64 v79; // rdx
  __int64 v80; // rax
  _QWORD *v81; // rbx
  _QWORD *v82; // r12
  __int64 v83; // rax
  _QWORD *v84; // rdi
  __int64 v85; // r8
  __int64 v86; // rdx
  _QWORD *v87; // rcx
  __int64 v88; // r8
  int v89; // eax
  unsigned int v90; // r15d
  __int64 v91; // rdx
  char v92; // cl
  __int64 v93; // rax
  int v94; // ecx
  __int64 v95; // rax
  __int64 v96; // r12
  _QWORD *v97; // r15
  __int64 v98; // rax
  __int64 v99; // rsi
  __int64 v100; // r15
  __int64 v101; // rdx
  __int64 v102; // r8
  __int64 v103; // rcx
  __int64 v104; // rdx
  int v105; // eax
  int v106; // eax
  unsigned int v107; // esi
  __int64 v108; // rax
  __int64 v109; // rsi
  __int64 v110; // rsi
  unsigned int v111; // esi
  __int64 v112; // rcx
  __int64 v113; // r10
  unsigned int v114; // eax
  char *v115; // rdx
  __int64 v116; // rdi
  __int64 v117; // rbx
  __int64 v118; // r8
  __int64 v119; // r12
  __int64 v120; // rsi
  __int64 v121; // rcx
  unsigned int v122; // edi
  __int64 v123; // rax
  __int64 v124; // rdx
  __int64 v125; // rbx
  int v126; // eax
  int v127; // r11d
  _QWORD *v128; // rsi
  int v129; // edi
  __int64 v130; // rdx
  __int64 v131; // r8
  int v132; // r11d
  int v133; // eax
  int v134; // edi
  _BYTE *v135; // [rsp+38h] [rbp-308h]
  __int64 v136; // [rsp+48h] [rbp-2F8h]
  __int64 v137; // [rsp+50h] [rbp-2F0h]
  __int64 **v138; // [rsp+58h] [rbp-2E8h]
  __int64 v139; // [rsp+58h] [rbp-2E8h]
  __int64 v140; // [rsp+58h] [rbp-2E8h]
  int v141; // [rsp+60h] [rbp-2E0h]
  __int64 v142; // [rsp+60h] [rbp-2E0h]
  _BYTE *v143; // [rsp+68h] [rbp-2D8h]
  __int64 v144; // [rsp+80h] [rbp-2C0h]
  __int64 **v147; // [rsp+98h] [rbp-2A8h]
  __int64 v148; // [rsp+A0h] [rbp-2A0h] BYREF
  __int64 *v149; // [rsp+A8h] [rbp-298h] BYREF
  __int64 v150; // [rsp+B0h] [rbp-290h] BYREF
  __int64 v151; // [rsp+B8h] [rbp-288h]
  __int64 v152; // [rsp+C0h] [rbp-280h]
  __int64 v153; // [rsp+C8h] [rbp-278h]
  _BYTE *v154; // [rsp+D0h] [rbp-270h] BYREF
  __int64 v155; // [rsp+D8h] [rbp-268h]
  _BYTE v156[64]; // [rsp+E0h] [rbp-260h] BYREF
  __int64 v157; // [rsp+120h] [rbp-220h] BYREF
  __int64 *v158; // [rsp+128h] [rbp-218h]
  __int64 v159; // [rsp+130h] [rbp-210h]
  int v160; // [rsp+138h] [rbp-208h]
  char v161; // [rsp+13Ch] [rbp-204h]
  _QWORD v162[8]; // [rsp+140h] [rbp-200h] BYREF
  __m128i *v163; // [rsp+180h] [rbp-1C0h] BYREF
  __int64 v164; // [rsp+188h] [rbp-1B8h]
  __int8 *v165; // [rsp+190h] [rbp-1B0h]
  const char *v166[16]; // [rsp+1A0h] [rbp-1A0h] BYREF
  __m128i v167; // [rsp+220h] [rbp-120h] BYREF
  _BYTE v168[16]; // [rsp+230h] [rbp-110h] BYREF
  char v169[64]; // [rsp+240h] [rbp-100h] BYREF
  const __m128i *v170; // [rsp+280h] [rbp-C0h]
  const __m128i *v171; // [rsp+288h] [rbp-B8h]
  __int8 *v172; // [rsp+290h] [rbp-B0h]
  char v173[8]; // [rsp+298h] [rbp-A8h] BYREF
  unsigned __int64 v174; // [rsp+2A0h] [rbp-A0h]
  char v175; // [rsp+2B4h] [rbp-8Ch]
  char v176[64]; // [rsp+2B8h] [rbp-88h] BYREF
  const __m128i *v177; // [rsp+2F8h] [rbp-48h]
  const char *v178; // [rsp+300h] [rbp-40h]
  const char *v179; // [rsp+308h] [rbp-38h]

  v5 = *(__int64 ***)a1;
  v6 = *(unsigned int *)(a1 + 8);
  v154 = v156;
  v150 = 0;
  v151 = 0;
  v152 = 0;
  v153 = 0;
  v155 = 0x800000000LL;
  v138 = &v5[12 * v6];
  if ( v5 == v138 )
    return sub_C7D6A0(v151, 8LL * (unsigned int)v153, 8);
  v147 = v5;
  do
  {
    v7 = *v147;
    memset(v166, 0, 0x78u);
    LODWORD(v166[2]) = 8;
    BYTE4(v166[3]) = 1;
    v166[1] = (const char *)&v166[4];
    v8 = *v7;
    v9 = v7[4];
    v159 = 0x100000008LL;
    v158 = v162;
    v163 = 0;
    v164 = 0;
    v165 = 0;
    v160 = 0;
    v161 = 1;
    v162[0] = v8 & 0xFFFFFFFFFFFFFFF8LL;
    v157 = 1;
    v167.m128i_i64[0] = v8 & 0xFFFFFFFFFFFFFFF8LL;
    v168[8] = 0;
    sub_23FEDC0((__int64)&v163, &v167);
    if ( !v161 )
      goto LABEL_208;
    v14 = v158;
    v11 = HIDWORD(v159);
    v10 = &v158[HIDWORD(v159)];
    if ( v158 != v10 )
    {
      while ( v9 != *v14 )
      {
        if ( v10 == ++v14 )
          goto LABEL_211;
      }
      goto LABEL_8;
    }
LABEL_211:
    if ( HIDWORD(v159) < (unsigned int)v159 )
    {
      ++HIDWORD(v159);
      *v10 = v9;
      ++v157;
    }
    else
    {
LABEL_208:
      sub_C8CC70((__int64)&v157, v9, (__int64)v10, v11, v12, v13);
    }
LABEL_8:
    sub_C8CF70((__int64)&v167, v169, 8, (__int64)v162, (__int64)&v157);
    v15 = (unsigned __int64)v163;
    v163 = 0;
    v170 = (const __m128i *)v15;
    v16 = v164;
    v164 = 0;
    v171 = (const __m128i *)v16;
    v17 = v165;
    v165 = 0;
    v172 = v17;
    sub_C8CF70((__int64)v173, v176, 8, (__int64)&v166[4], (__int64)v166);
    v21 = v166[12];
    memset(&v166[12], 0, 24);
    v177 = (const __m128i *)v21;
    v178 = v166[13];
    v179 = v166[14];
    if ( v163 )
      j_j___libc_free_0((unsigned __int64)v163);
    if ( !v161 )
      _libc_free((unsigned __int64)v158);
    if ( v166[12] )
      j_j___libc_free_0((unsigned __int64)v166[12]);
    if ( !BYTE4(v166[3]) )
      _libc_free((unsigned __int64)v166[1]);
    v22 = (const char **)v162;
    v23 = &v157;
    sub_C8CD80((__int64)&v157, (__int64)v162, (__int64)&v167, v18, v19, v20);
    v26 = v171;
    v27 = v170;
    v163 = 0;
    v164 = 0;
    v165 = 0;
    v28 = (char *)v171 - (char *)v170;
    if ( v171 == v170 )
    {
      v28 = 0;
      v29 = 0;
    }
    else
    {
      if ( v28 > 0x7FFFFFFFFFFFFFE0LL )
        goto LABEL_261;
      v29 = (__m128i *)sub_22077B0((char *)v171 - (char *)v170);
      v26 = v171;
      v27 = v170;
    }
    v163 = v29;
    v164 = (__int64)v29;
    v165 = &v29->m128i_i8[v28];
    if ( v27 == v26 )
    {
      v30 = (__int64)v29;
    }
    else
    {
      v30 = (__int64)v29->m128i_i64 + (char *)v26 - (char *)v27;
      do
      {
        if ( v29 )
        {
          *v29 = _mm_loadu_si128(v27);
          v29[1] = _mm_loadu_si128(v27 + 1);
        }
        v29 += 2;
        v27 += 2;
      }
      while ( v29 != (__m128i *)v30 );
    }
    v22 = &v166[4];
    v164 = v30;
    v23 = (__int64 *)v166;
    sub_C8CD80((__int64)v166, (__int64)&v166[4], (__int64)v173, v30, v24, v25);
    v33 = (const __m128i *)v178;
    v34 = v177;
    memset(&v166[12], 0, 24);
    v35 = v178 - (const char *)v177;
    if ( v178 == (const char *)v177 )
    {
      v35 = 0;
      v37 = 0;
    }
    else
    {
      if ( v35 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_261:
        sub_4261EA(v23, v22, v27);
      v36 = sub_22077B0(v178 - (const char *)v177);
      v34 = v177;
      v37 = (const char *)v36;
      v33 = (const __m128i *)v178;
    }
    v166[12] = v37;
    v166[13] = v37;
    v166[14] = &v37[v35];
    if ( v33 == v34 )
    {
      v39 = v37;
    }
    else
    {
      v38 = (__m128i *)v37;
      v39 = &v37[(char *)v33 - (char *)v34];
      do
      {
        if ( v38 )
        {
          *v38 = _mm_loadu_si128(v34);
          v38[1] = _mm_loadu_si128(v34 + 1);
        }
        v38 += 2;
        v34 += 2;
      }
      while ( v38 != (__m128i *)v39 );
    }
    for ( v166[13] = v39; ; v39 = v166[13] )
    {
      v40 = (unsigned __int64)v163;
      if ( v164 - (_QWORD)v163 != v39 - v37 )
        goto LABEL_34;
      if ( v163 == (__m128i *)v164 )
        break;
      v48 = v37;
      while ( *(_QWORD *)v40 == *(_QWORD *)v48 )
      {
        v49 = *(_BYTE *)(v40 + 24);
        if ( v49 != v48[24] || v49 && *(_DWORD *)(v40 + 16) != *((_DWORD *)v48 + 4) )
          break;
        v40 += 32LL;
        v48 += 32;
        if ( v164 == v40 )
          goto LABEL_51;
      }
LABEL_34:
      v41 = *(_QWORD *)(v164 - 32);
      v148 = v41;
      if ( (_DWORD)v152 )
      {
        v67 = v153;
        if ( (_DWORD)v153 )
        {
          v68 = (unsigned int)(v153 - 1);
          v69 = v151;
          v70 = 1;
          v71 = 0;
          v72 = v68 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
          v73 = (__int64 *)(v151 + 8LL * v72);
          v74 = *v73;
          if ( v41 == *v73 )
            goto LABEL_43;
          while ( v74 != -4096 )
          {
            if ( v74 == -8192 && !v71 )
              v71 = v73;
            v72 = v68 & (v70 + v72);
            v73 = (__int64 *)(v151 + 8LL * v72);
            v74 = *v73;
            if ( v41 == *v73 )
              goto LABEL_43;
            ++v70;
          }
          if ( v71 )
            v73 = v71;
          v75 = v152 + 1;
          ++v150;
          v149 = v73;
          if ( 4 * ((int)v152 + 1) < (unsigned int)(3 * v153) )
          {
            if ( (int)v153 - HIDWORD(v152) - v75 > (unsigned int)v153 >> 3 )
            {
LABEL_126:
              LODWORD(v152) = v75;
              if ( *v73 != -4096 )
                --HIDWORD(v152);
              *v73 = v41;
              v76 = (unsigned int)v155;
              v77 = v148;
              v78 = (unsigned int)v155 + 1LL;
              if ( v78 > HIDWORD(v155) )
              {
                sub_C8D5F0((__int64)&v154, v156, v78, 8u, v69, v68);
                v76 = (unsigned int)v155;
              }
              *(_QWORD *)&v154[8 * v76] = v77;
              LODWORD(v155) = v155 + 1;
              goto LABEL_43;
            }
LABEL_229:
            sub_CF28B0((__int64)&v150, v67);
            sub_D6B660((__int64)&v150, &v148, &v149);
            v41 = v148;
            v73 = v149;
            v75 = v152 + 1;
            goto LABEL_126;
          }
        }
        else
        {
          ++v150;
          v149 = 0;
        }
        v67 = 2 * v153;
        goto LABEL_229;
      }
      v42 = v154;
      v43 = 8LL * (unsigned int)v155;
      v44 = (__int64 *)&v154[v43];
      v45 = v43 >> 3;
      v46 = v43 >> 5;
      if ( !v46 )
        goto LABEL_132;
      v47 = &v154[32 * v46];
      do
      {
        if ( v41 == *v42 )
          goto LABEL_42;
        if ( v41 == v42[1] )
        {
          ++v42;
          goto LABEL_42;
        }
        if ( v41 == v42[2] )
        {
          v42 += 2;
          goto LABEL_42;
        }
        if ( v41 == v42[3] )
        {
          v42 += 3;
          goto LABEL_42;
        }
        v42 += 4;
      }
      while ( v47 != v42 );
      v45 = v44 - v42;
LABEL_132:
      switch ( v45 )
      {
        case 2LL:
          goto LABEL_199;
        case 3LL:
          if ( v41 != *v42 )
          {
            ++v42;
LABEL_199:
            if ( v41 != *v42 )
            {
              ++v42;
LABEL_201:
              if ( v41 != *v42 )
              {
                v79 = (unsigned int)v155 + 1LL;
                if ( v79 > HIDWORD(v155) )
                  goto LABEL_203;
                goto LABEL_136;
              }
            }
          }
LABEL_42:
          if ( v44 == v42 )
            break;
          goto LABEL_43;
        case 1LL:
          goto LABEL_201;
      }
      v79 = (unsigned int)v155 + 1LL;
      if ( v79 <= HIDWORD(v155) )
        goto LABEL_136;
LABEL_203:
      sub_C8D5F0((__int64)&v154, v156, v79, 8u, v31, v32);
      v44 = (__int64 *)&v154[8 * (unsigned int)v155];
LABEL_136:
      *v44 = v41;
      v80 = (unsigned int)(v155 + 1);
      LODWORD(v155) = v80;
      if ( (unsigned int)v80 > 8 )
      {
        v81 = v154;
        v82 = &v154[8 * v80];
        while ( (_DWORD)v153 )
        {
          LODWORD(v83) = (v153 - 1) & (((unsigned int)*v81 >> 9) ^ ((unsigned int)*v81 >> 4));
          v84 = (_QWORD *)(v151 + 8LL * (unsigned int)v83);
          v85 = *v84;
          if ( *v81 != *v84 )
          {
            v127 = 1;
            v87 = 0;
            while ( v85 != -4096 )
            {
              if ( !v87 && v85 == -8192 )
                v87 = v84;
              v83 = ((_DWORD)v153 - 1) & (unsigned int)(v83 + v127);
              v84 = (_QWORD *)(v151 + 8 * v83);
              v85 = *v84;
              if ( *v81 == *v84 )
                goto LABEL_139;
              ++v127;
            }
            if ( !v87 )
              v87 = v84;
            ++v150;
            v89 = v152 + 1;
            if ( 4 * ((int)v152 + 1) < (unsigned int)(3 * v153) )
            {
              if ( (int)v153 - HIDWORD(v152) - v89 <= (unsigned int)v153 >> 3 )
              {
                sub_CF28B0((__int64)&v150, v153);
                if ( !(_DWORD)v153 )
                {
LABEL_269:
                  LODWORD(v152) = v152 + 1;
                  BUG();
                }
                v128 = 0;
                v129 = 1;
                LODWORD(v130) = (v153 - 1) & (((unsigned int)*v81 >> 9) ^ ((unsigned int)*v81 >> 4));
                v87 = (_QWORD *)(v151 + 8LL * (unsigned int)v130);
                v131 = *v87;
                v89 = v152 + 1;
                if ( *v81 != *v87 )
                {
                  while ( v131 != -4096 )
                  {
                    if ( !v128 && v131 == -8192 )
                      v128 = v87;
                    v130 = ((_DWORD)v153 - 1) & (unsigned int)(v130 + v129);
                    v87 = (_QWORD *)(v151 + 8 * v130);
                    v131 = *v87;
                    if ( *v81 == *v87 )
                      goto LABEL_144;
                    ++v129;
                  }
LABEL_253:
                  if ( v128 )
                    v87 = v128;
                }
              }
LABEL_144:
              LODWORD(v152) = v89;
              if ( *v87 != -4096 )
                --HIDWORD(v152);
              *v87 = *v81;
              goto LABEL_139;
            }
LABEL_142:
            sub_CF28B0((__int64)&v150, 2 * v153);
            if ( !(_DWORD)v153 )
              goto LABEL_269;
            LODWORD(v86) = (v153 - 1) & (((unsigned int)*v81 >> 9) ^ ((unsigned int)*v81 >> 4));
            v87 = (_QWORD *)(v151 + 8LL * (unsigned int)v86);
            v88 = *v87;
            v89 = v152 + 1;
            if ( *v81 != *v87 )
            {
              v134 = 1;
              v128 = 0;
              while ( v88 != -4096 )
              {
                if ( v88 == -8192 && !v128 )
                  v128 = v87;
                v86 = ((_DWORD)v153 - 1) & (unsigned int)(v86 + v134);
                v87 = (_QWORD *)(v151 + 8 * v86);
                v88 = *v87;
                if ( *v81 == *v87 )
                  goto LABEL_144;
                ++v134;
              }
              goto LABEL_253;
            }
            goto LABEL_144;
          }
LABEL_139:
          if ( v82 == ++v81 )
            goto LABEL_43;
        }
        ++v150;
        goto LABEL_142;
      }
LABEL_43:
      sub_23EC7E0((__int64)&v157);
      v37 = v166[12];
    }
LABEL_51:
    if ( v37 )
      j_j___libc_free_0((unsigned __int64)v37);
    if ( !BYTE4(v166[3]) )
      _libc_free((unsigned __int64)v166[1]);
    if ( v163 )
      j_j___libc_free_0((unsigned __int64)v163);
    if ( !v161 )
      _libc_free((unsigned __int64)v158);
    if ( v177 )
      j_j___libc_free_0((unsigned __int64)v177);
    if ( !v175 )
      _libc_free(v174);
    if ( v170 )
      j_j___libc_free_0((unsigned __int64)v170);
    if ( !v168[12] )
      _libc_free(v167.m128i_u64[1]);
    v147 += 12;
  }
  while ( v138 != v147 );
  v50 = v154;
  v135 = &v154[8 * (unsigned int)v155];
  if ( v135 != v154 )
  {
    v143 = v154;
    while ( 1 )
    {
      v51 = *(_QWORD *)(*(_QWORD *)v143 + 56LL);
      v144 = *(_QWORD *)v143 + 48LL;
      if ( v144 == v51 )
        goto LABEL_91;
      do
      {
        v167.m128i_i64[0] = (__int64)v168;
        v167.m128i_i64[1] = 0x800000000LL;
        if ( !v51 )
          BUG();
        v52 = *(char **)(v51 - 8);
        if ( !v52 )
          goto LABEL_90;
        v53 = 0;
        do
        {
          v54 = (char *)*((_QWORD *)v52 + 3);
          v55 = *v54;
          if ( (unsigned __int8)*v54 <= 0x1Cu )
            goto LABEL_85;
          v56 = *((_QWORD *)v54 + 5);
          if ( (_DWORD)v152 )
          {
            if ( !(_DWORD)v153 )
              goto LABEL_100;
            v65 = (v153 - 1) & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
            v32 = v151 + 8LL * v65;
            v66 = *(_QWORD *)v32;
            if ( v56 != *(_QWORD *)v32 )
            {
              v32 = 1;
              while ( v66 != -4096 )
              {
                v90 = v32 + 1;
                v65 = (v153 - 1) & (v65 + v32);
                v32 = v151 + 8LL * v65;
                v66 = *(_QWORD *)v32;
                if ( v56 == *(_QWORD *)v32 )
                  goto LABEL_106;
                v32 = v90;
              }
              goto LABEL_100;
            }
LABEL_106:
            v57 = (_QWORD *)(v151 + 8LL * (unsigned int)v153);
            goto LABEL_83;
          }
          v57 = v154;
          v58 = 8LL * (unsigned int)v155;
          v32 = (__int64)&v154[v58];
          v59 = v58 >> 3;
          v60 = v58 >> 5;
          if ( v60 )
          {
            v61 = &v154[32 * v60];
            while ( v56 != *v57 )
            {
              if ( v56 == v57[1] )
              {
                ++v57;
                goto LABEL_83;
              }
              if ( v56 == v57[2] )
              {
                v57 += 2;
                goto LABEL_83;
              }
              if ( v56 == v57[3] )
              {
                v57 += 3;
                goto LABEL_83;
              }
              v57 += 4;
              if ( v57 == v61 )
              {
                v59 = (v32 - (__int64)v57) >> 3;
                goto LABEL_97;
              }
            }
            goto LABEL_83;
          }
LABEL_97:
          if ( v59 == 2 )
            goto LABEL_112;
          if ( v59 != 3 )
          {
            if ( v59 != 1 )
              goto LABEL_100;
            goto LABEL_114;
          }
          if ( v56 != *v57 )
          {
            ++v57;
LABEL_112:
            if ( v56 != *v57 )
            {
              ++v57;
LABEL_114:
              if ( v56 != *v57 )
                goto LABEL_100;
            }
          }
LABEL_83:
          if ( (_QWORD *)v32 != v57 )
          {
            if ( v55 != 84 )
              goto LABEL_85;
            goto LABEL_107;
          }
LABEL_100:
          if ( a3 != v56 || v55 != 84 )
          {
            v64 = v53 + 1;
            if ( v53 + 1 <= (unsigned __int64)v167.m128i_u32[3] )
            {
LABEL_103:
              *(_QWORD *)(v167.m128i_i64[0] + 8 * v53) = v54;
              v53 = (unsigned int)++v167.m128i_i32[2];
              goto LABEL_85;
            }
LABEL_109:
            sub_C8D5F0((__int64)&v167, v168, v64, 8u, v64, v32);
            v53 = v167.m128i_u32[2];
            goto LABEL_103;
          }
LABEL_107:
          if ( a2 == v56 )
          {
            v64 = v53 + 1;
            if ( v53 + 1 <= (unsigned __int64)v167.m128i_u32[3] )
              goto LABEL_103;
            goto LABEL_109;
          }
LABEL_85:
          v52 = (char *)*((_QWORD *)v52 + 1);
        }
        while ( v52 );
        if ( (_DWORD)v53 )
        {
          v91 = *(_QWORD *)(a3 + 16);
          LOWORD(v166[4]) = 257;
          do
          {
            if ( !v91 )
            {
              v141 = 0;
              goto LABEL_159;
            }
            v92 = **(_BYTE **)(v91 + 24);
            v93 = v91;
            v91 = *(_QWORD *)(v91 + 8);
          }
          while ( (unsigned __int8)(v92 - 30) > 0xAu );
          v94 = 0;
          while ( 1 )
          {
            v93 = *(_QWORD *)(v93 + 8);
            if ( !v93 )
              break;
            while ( (unsigned __int8)(**(_BYTE **)(v93 + 24) - 30) <= 0xAu )
            {
              v93 = *(_QWORD *)(v93 + 8);
              ++v94;
              if ( !v93 )
                goto LABEL_158;
            }
          }
LABEL_158:
          v141 = v94 + 1;
LABEL_159:
          v139 = *(_QWORD *)(v51 - 16);
          v95 = sub_BD2DA0(80);
          v96 = v95;
          if ( v95 )
          {
            v97 = (_QWORD *)v95;
            sub_B44260(v95, v139, 55, 0x8000000u, 0, 0);
            *(_DWORD *)(v96 + 72) = v141;
            sub_BD6B50((unsigned __int8 *)v96, v166);
            sub_BD2A10(v96, *(_DWORD *)(v96 + 72), 1);
          }
          else
          {
            v97 = 0;
          }
          v98 = v136;
          v99 = *(_QWORD *)(a3 + 56);
          v157 = v96;
          LOWORD(v98) = 1;
          v136 = v98;
          sub_B44220(v97, v99, v98);
          v100 = *(_QWORD *)(a3 + 16);
          if ( v100 )
          {
            do
            {
              v101 = *(_QWORD *)(v100 + 24);
              if ( (unsigned __int8)(*(_BYTE *)v101 - 30) <= 0xAu )
              {
                v102 = v51 - 8;
LABEL_165:
                v103 = v157;
                v104 = *(_QWORD *)(v101 + 40);
                v105 = *(_DWORD *)(v157 + 4) & 0x7FFFFFF;
                if ( v105 == *(_DWORD *)(v157 + 72) )
                {
                  v137 = v102;
                  v140 = v104;
                  v142 = v157;
                  sub_B48D90(v157);
                  v103 = v142;
                  v102 = v137;
                  v104 = v140;
                  v105 = *(_DWORD *)(v142 + 4) & 0x7FFFFFF;
                }
                v106 = (v105 + 1) & 0x7FFFFFF;
                v107 = v106 | *(_DWORD *)(v103 + 4) & 0xF8000000;
                v108 = *(_QWORD *)(v103 - 8) + 32LL * (unsigned int)(v106 - 1);
                *(_DWORD *)(v103 + 4) = v107;
                if ( *(_QWORD *)v108 )
                {
                  v109 = *(_QWORD *)(v108 + 8);
                  **(_QWORD **)(v108 + 16) = v109;
                  if ( v109 )
                    *(_QWORD *)(v109 + 16) = *(_QWORD *)(v108 + 16);
                }
                *(_QWORD *)v108 = v51 - 24;
                v110 = *(_QWORD *)(v51 - 8);
                *(_QWORD *)(v108 + 8) = v110;
                if ( v110 )
                  *(_QWORD *)(v110 + 16) = v108 + 8;
                *(_QWORD *)(v108 + 16) = v102;
                *(_QWORD *)(v51 - 8) = v108;
                *(_QWORD *)(*(_QWORD *)(v103 - 8)
                          + 32LL * *(unsigned int *)(v103 + 72)
                          + 8LL * ((*(_DWORD *)(v103 + 4) & 0x7FFFFFFu) - 1)) = v104;
                while ( 1 )
                {
                  v100 = *(_QWORD *)(v100 + 8);
                  if ( !v100 )
                    goto LABEL_174;
                  v101 = *(_QWORD *)(v100 + 24);
                  if ( (unsigned __int8)(*(_BYTE *)v101 - 30) <= 0xAu )
                    goto LABEL_165;
                }
              }
              v100 = *(_QWORD *)(v100 + 8);
            }
            while ( v100 );
            v111 = *(_DWORD *)(a4 + 24);
            if ( v111 )
              goto LABEL_175;
LABEL_194:
            v166[0] = 0;
            ++*(_QWORD *)a4;
LABEL_195:
            v125 = a4;
            v111 *= 2;
            goto LABEL_196;
          }
LABEL_174:
          v111 = *(_DWORD *)(a4 + 24);
          if ( !v111 )
            goto LABEL_194;
LABEL_175:
          v112 = v157;
          v113 = *(_QWORD *)(a4 + 8);
          v114 = (v111 - 1) & (((unsigned int)v157 >> 9) ^ ((unsigned int)v157 >> 4));
          v115 = (char *)(v113 + 8LL * v114);
          v116 = *(_QWORD *)v115;
          if ( v157 != *(_QWORD *)v115 )
          {
            v132 = 1;
            while ( v116 != -4096 )
            {
              if ( v116 == -8192 && !v52 )
                v52 = v115;
              v32 = (unsigned int)(v132 + 1);
              v114 = (v111 - 1) & (v132 + v114);
              v115 = (char *)(v113 + 8LL * v114);
              v116 = *(_QWORD *)v115;
              if ( v157 == *(_QWORD *)v115 )
                goto LABEL_176;
              ++v132;
            }
            if ( v52 )
              v115 = v52;
            ++*(_QWORD *)a4;
            v133 = *(_DWORD *)(a4 + 16);
            v166[0] = v115;
            v126 = v133 + 1;
            if ( 4 * v126 >= 3 * v111 )
              goto LABEL_195;
            v125 = a4;
            if ( v111 - *(_DWORD *)(a4 + 20) - v126 <= v111 >> 3 )
            {
LABEL_196:
              sub_110B120(v125, v111);
              sub_23FE5C0(v125, &v157, v166);
              v112 = v157;
              v115 = (char *)v166[0];
              v126 = *(_DWORD *)(v125 + 16) + 1;
            }
            *(_DWORD *)(a4 + 16) = v126;
            if ( *(_QWORD *)v115 != -4096 )
              --*(_DWORD *)(a4 + 20);
            *(_QWORD *)v115 = v112;
          }
LABEL_176:
          v62 = (_BYTE *)v167.m128i_i64[0];
          v117 = v167.m128i_i64[0] + 8LL * v167.m128i_u32[2];
          if ( v117 == v167.m128i_i64[0] )
            goto LABEL_88;
          v118 = v167.m128i_i64[0];
          v119 = v51 - 24;
          while ( 1 )
          {
            v120 = *(_QWORD *)v118;
            v121 = 0;
            v122 = *(_DWORD *)(*(_QWORD *)v118 + 4LL) & 0x7FFFFFF;
            if ( v122 )
              break;
LABEL_190:
            v118 += 8;
            if ( v118 == v117 )
              goto LABEL_87;
          }
          while ( 1 )
          {
            if ( (*(_BYTE *)(v120 + 7) & 0x40) != 0 )
            {
              v123 = *(_QWORD *)(v120 - 8) + 32 * v121;
              if ( v119 != *(_QWORD *)v123 )
                goto LABEL_181;
            }
            else
            {
              v123 = v120 + 32 * (v121 - (*(_DWORD *)(v120 + 4) & 0x7FFFFFF));
              if ( v119 != *(_QWORD *)v123 )
                goto LABEL_181;
            }
            v32 = *(_QWORD *)(v123 + 8);
            v124 = v157;
            **(_QWORD **)(v123 + 16) = v32;
            if ( v32 )
              *(_QWORD *)(v32 + 16) = *(_QWORD *)(v123 + 16);
            *(_QWORD *)v123 = v124;
            if ( !v124 )
            {
LABEL_181:
              if ( v122 <= (unsigned int)++v121 )
                goto LABEL_190;
              continue;
            }
            v32 = *(_QWORD *)(v124 + 16);
            *(_QWORD *)(v123 + 8) = v32;
            if ( v32 )
              *(_QWORD *)(v32 + 16) = v123 + 8;
            ++v121;
            *(_QWORD *)(v123 + 16) = v124 + 16;
            *(_QWORD *)(v124 + 16) = v123;
            if ( v122 <= (unsigned int)v121 )
              goto LABEL_190;
          }
        }
LABEL_87:
        v62 = (_BYTE *)v167.m128i_i64[0];
LABEL_88:
        if ( v62 != v168 )
          _libc_free((unsigned __int64)v62);
LABEL_90:
        v51 = *(_QWORD *)(v51 + 8);
      }
      while ( v144 != v51 );
LABEL_91:
      v143 += 8;
      if ( v135 == v143 )
      {
        v50 = v154;
        break;
      }
    }
  }
  if ( v50 != v156 )
    _libc_free((unsigned __int64)v50);
  return sub_C7D6A0(v151, 8LL * (unsigned int)v153, 8);
}
