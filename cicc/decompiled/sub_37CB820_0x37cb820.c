// Function: sub_37CB820
// Address: 0x37cb820
//
__int64 __fastcall sub_37CB820(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4, __int64 a5, unsigned __int64 a6)
{
  unsigned int *v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rax
  unsigned int *v11; // rbx
  unsigned __int64 v12; // rdi
  unsigned int *v14; // r12
  __int64 i; // rax
  unsigned int *v16; // rdi
  unsigned int *v17; // rbx
  __int64 j; // rax
  unsigned int *v19; // rcx
  unsigned int *v20; // rax
  __int64 v21; // rax
  unsigned int v22; // eax
  unsigned int *v23; // r13
  unsigned int v24; // ecx
  char *v25; // rdx
  __int64 v26; // rdi
  __m128i v27; // rax
  int v28; // esi
  char *v29; // r8
  int v30; // ecx
  _QWORD *v31; // rax
  _QWORD *v32; // rcx
  unsigned int v33; // esi
  _QWORD *v34; // rax
  _QWORD *v35; // r8
  __int64 v36; // rax
  __int64 v37; // rdx
  _QWORD *v38; // rax
  _QWORD *v39; // rdx
  __int64 v40; // rdx
  _QWORD *v41; // rax
  _QWORD *v42; // rdx
  _QWORD *v43; // rdi
  __int64 v44; // rsi
  _QWORD *v45; // rax
  int v46; // r11d
  unsigned int v47; // ecx
  __int64 v48; // rsi
  unsigned int v49; // edx
  int v50; // edx
  unsigned __int64 v51; // r12
  unsigned int v52; // eax
  __int64 v53; // rax
  _QWORD *v54; // rdx
  unsigned int v55; // ecx
  unsigned int v56; // eax
  int v57; // r12d
  unsigned int v58; // eax
  unsigned int v59; // ecx
  _QWORD *v60; // rdi
  __int64 v61; // rsi
  unsigned int v62; // eax
  int v63; // eax
  unsigned __int64 v64; // r12
  unsigned int v65; // eax
  _QWORD *v66; // rax
  _QWORD *m; // rdx
  _QWORD *v68; // rax
  _QWORD *v69; // r14
  _QWORD *v70; // rbx
  unsigned __int64 v71; // r13
  unsigned __int64 *v72; // r15
  unsigned __int64 *v73; // r12
  __int64 v74; // rsi
  _QWORD *v75; // rax
  unsigned int *v76; // rbx
  char *v77; // rax
  __int64 v78; // rdx
  __int64 v79; // r8
  __int64 v80; // r9
  int v81; // esi
  char **v82; // rdx
  int v83; // eax
  __int64 *v84; // r14
  __int64 v85; // r12
  char *v86; // rdx
  __int64 *v87; // rbx
  __int64 v88; // rax
  char *v89; // r12
  __int64 v90; // rbx
  __int64 *v91; // rsi
  unsigned __int64 v92; // rax
  char *v93; // r13
  char *v94; // r15
  __int64 v95; // r14
  __int64 v96; // rdx
  char *v97; // rbx
  _QWORD *v98; // rbx
  __int64 v99; // r15
  __int64 **v100; // rsi
  int v101; // r14d
  __int64 *v102; // rcx
  __int64 v103; // rdi
  unsigned int v104; // eax
  __int64 v105; // rdx
  __int64 *v106; // r13
  __int64 *v107; // rbx
  unsigned __int64 k; // rax
  __int64 v109; // rdi
  unsigned int v110; // ecx
  __int64 v111; // rax
  __int64 *v112; // rbx
  __int64 *v113; // r12
  __int64 v114; // rsi
  __int64 v115; // rdi
  __int64 v116; // rdx
  unsigned int v117; // r8d
  __int64 v118; // rdi
  _QWORD *v119; // r9
  unsigned int v120; // edx
  __int64 **v121; // rax
  __int64 *v122; // r10
  __int64 *v123; // rdx
  unsigned int v124; // edx
  char **v125; // rax
  char *v126; // r9
  int v127; // eax
  int v128; // r8d
  int v129; // eax
  int v130; // r10d
  __int64 v131; // rax
  _QWORD *v132; // rsi
  _QWORD *v133; // rsi
  int v134; // esi
  char **v135; // rcx
  int v136; // edx
  int v137; // esi
  int v138; // [rsp+4h] [rbp-2BCh]
  __int64 v139; // [rsp+8h] [rbp-2B8h]
  char *v140; // [rsp+30h] [rbp-290h]
  unsigned int v141; // [rsp+40h] [rbp-280h]
  char *v142; // [rsp+40h] [rbp-280h]
  char *v143; // [rsp+48h] [rbp-278h]
  __int64 v145; // [rsp+58h] [rbp-268h]
  unsigned int *v147; // [rsp+60h] [rbp-260h]
  __int64 v148; // [rsp+60h] [rbp-260h]
  __int64 **v149; // [rsp+60h] [rbp-260h]
  char **v151; // [rsp+88h] [rbp-238h] BYREF
  __int128 v152; // [rsp+90h] [rbp-230h] BYREF
  __int64 v153; // [rsp+A0h] [rbp-220h] BYREF
  __int64 v154; // [rsp+A8h] [rbp-218h]
  __int64 v155; // [rsp+B0h] [rbp-210h]
  unsigned int v156; // [rsp+B8h] [rbp-208h]
  __int64 v157; // [rsp+C0h] [rbp-200h] BYREF
  __int64 v158; // [rsp+C8h] [rbp-1F8h]
  __int64 v159; // [rsp+D0h] [rbp-1F0h]
  unsigned int v160; // [rsp+D8h] [rbp-1E8h]
  __int64 *v161; // [rsp+E0h] [rbp-1E0h] BYREF
  __int64 v162; // [rsp+E8h] [rbp-1D8h]
  _BYTE v163[64]; // [rsp+F0h] [rbp-1D0h] BYREF
  char *v164; // [rsp+130h] [rbp-190h] BYREF
  __int64 v165; // [rsp+138h] [rbp-188h]
  _BYTE v166[64]; // [rsp+140h] [rbp-180h] BYREF
  __int64 v167; // [rsp+180h] [rbp-140h] BYREF
  _QWORD *v168; // [rsp+188h] [rbp-138h]
  __int64 v169; // [rsp+190h] [rbp-130h]
  __int64 v170; // [rsp+198h] [rbp-128h]
  __int64 v171; // [rsp+1A0h] [rbp-120h] BYREF
  _QWORD *v172; // [rsp+1A8h] [rbp-118h]
  __int64 v173; // [rsp+1B0h] [rbp-110h]
  __int64 v174; // [rsp+1B8h] [rbp-108h]
  __int64 v175; // [rsp+1C0h] [rbp-100h]
  _QWORD *v176; // [rsp+1C8h] [rbp-F8h]
  __int64 v177; // [rsp+1D0h] [rbp-F0h]
  __int64 v178; // [rsp+1D8h] [rbp-E8h]
  unsigned int v179; // [rsp+1E0h] [rbp-E0h]
  _QWORD *v180; // [rsp+1E8h] [rbp-D8h]
  __m128i v181; // [rsp+1F0h] [rbp-D0h] BYREF
  __int64 **v182; // [rsp+200h] [rbp-C0h]
  __int64 v183; // [rsp+208h] [rbp-B8h]
  __int64 v184; // [rsp+210h] [rbp-B0h]
  __int64 v185; // [rsp+218h] [rbp-A8h]
  unsigned int v186; // [rsp+220h] [rbp-A0h]
  __int64 v187; // [rsp+228h] [rbp-98h]
  __int64 v188; // [rsp+230h] [rbp-90h]
  __int64 *v189; // [rsp+238h] [rbp-88h]
  __int64 v190; // [rsp+240h] [rbp-80h]
  _BYTE v191[32]; // [rsp+248h] [rbp-78h] BYREF
  __int64 *v192; // [rsp+268h] [rbp-58h]
  __int64 v193; // [rsp+270h] [rbp-50h]
  _QWORD v194[9]; // [rsp+278h] [rbp-48h] BYREF

  v7 = *(unsigned int **)(a1 + 776);
  v8 = *(unsigned int *)(a1 + 784);
  if ( !(5 * v8) )
    goto LABEL_7;
  while ( 1 )
  {
    while ( 1 )
    {
      v9 = v8 >> 1;
      v10 = 10 * (v8 >> 1);
      v11 = &v7[v10];
      v12 = v7[v10];
      if ( v12 >= a6 )
        break;
      v7 = v11 + 10;
      v8 = v8 - v9 - 1;
      if ( v8 <= 0 )
        goto LABEL_7;
    }
    if ( v12 <= a6 )
      break;
    v8 >>= 1;
    if ( v9 <= 0 )
      goto LABEL_7;
  }
  v14 = v7;
  for ( i = 0xCCCCCCCCCCCCCCCDLL * ((v10 * 4) >> 3); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v16 = &v14[10 * (i >> 1)];
      if ( *v16 >= a6 )
        break;
      v14 = v16 + 10;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        goto LABEL_13;
    }
  }
LABEL_13:
  v17 = v11 + 10;
  for ( j = 0xCCCCCCCCCCCCCCCDLL * (((char *)&v7[10 * v8] - (char *)v17) >> 3); j > 0; j >>= 1 )
  {
    while ( 1 )
    {
      v19 = &v17[10 * (j >> 1)];
      if ( *v19 > a6 )
        break;
      v17 = v19 + 10;
      j = j - (j >> 1) - 1;
      if ( j <= 0 )
        goto LABEL_17;
    }
  }
LABEL_17:
  if ( v17 == v14 )
  {
LABEL_7:
    BYTE8(v152) = 0;
    return v152;
  }
  v20 = v14;
  do
  {
    if ( !*((_BYTE *)v20 + 24) )
      goto LABEL_7;
    v20 += 10;
  }
  while ( v20 != v17 );
  if ( (char *)v20 - (char *)v14 == 40 )
  {
    v21 = *((_QWORD *)v14 + 2);
    v181.m128i_i8[8] = 1;
    v181.m128i_i64[0] = v21;
    return _mm_loadu_si128(&v181).m128i_i64[0];
  }
  v22 = v14[8];
  v23 = v14;
  v167 = 0;
  v168 = 0;
  v141 = v22;
  v179 = v22;
  v169 = 0;
  v180 = a4;
  v161 = (__int64 *)v163;
  v162 = 0x800000000LL;
  v170 = 0;
  v171 = 0;
  v172 = 0;
  v173 = 0;
  v174 = 0;
  v175 = 0;
  v176 = 0;
  v177 = 0;
  v178 = 0;
  v153 = 0;
  v154 = 0;
  v155 = 0;
  v156 = 0;
  do
  {
    v27.m128i_i64[0] = (__int64)sub_37B98B0((__int64)&v167, *((_QWORD *)v23 + 1));
    v27.m128i_i64[1] = *((_QWORD *)v23 + 2);
    v28 = v156;
    v181 = v27;
    if ( v156 )
    {
      v24 = (v156 - 1) & (((unsigned __int32)v27.m128i_i32[0] >> 9) ^ ((unsigned __int32)v27.m128i_i32[0] >> 4));
      v25 = (char *)(v154 + 16LL * v24);
      v26 = *(_QWORD *)v25;
      if ( v27.m128i_i64[0] == *(_QWORD *)v25 )
        goto LABEL_27;
      v46 = 1;
      v29 = 0;
      while ( v26 != -4096 )
      {
        if ( v29 || v26 != -8192 )
          v25 = v29;
        v24 = (v156 - 1) & (v46 + v24);
        v26 = *(_QWORD *)(v154 + 16LL * v24);
        if ( v27.m128i_i64[0] == v26 )
          goto LABEL_27;
        ++v46;
        v29 = v25;
        v25 = (char *)(v154 + 16LL * v24);
      }
      if ( !v29 )
        v29 = v25;
      ++v153;
      v30 = v155 + 1;
      v164 = v29;
      if ( 4 * ((int)v155 + 1) < 3 * v156 )
      {
        if ( v156 - HIDWORD(v155) - v30 > v156 >> 3 )
          goto LABEL_65;
        goto LABEL_31;
      }
    }
    else
    {
      ++v153;
      v164 = 0;
    }
    v28 = 2 * v156;
LABEL_31:
    sub_37B90A0((__int64)&v153, v28);
    sub_37B5D40((__int64)&v153, v181.m128i_i64, &v164);
    v27.m128i_i64[0] = v181.m128i_i64[0];
    v29 = v164;
    v30 = v155 + 1;
LABEL_65:
    LODWORD(v155) = v30;
    if ( *(_QWORD *)v29 != -4096 )
      --HIDWORD(v155);
    *(_QWORD *)v29 = v27.m128i_i64[0];
    *((_QWORD *)v29 + 1) = v181.m128i_i64[1];
LABEL_27:
    v23 += 10;
  }
  while ( v23 != v17 );
  v31 = sub_37B98B0((__int64)&v167, *(_QWORD *)(a5 + 24));
  v32 = v31;
  if ( v156 )
  {
    v33 = (v156 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
    v34 = (_QWORD *)(v154 + 16LL * v33);
    v35 = (_QWORD *)*v34;
    if ( v32 == (_QWORD *)*v34 )
    {
LABEL_34:
      if ( v34 != (_QWORD *)(v154 + 16LL * v156) )
      {
        v36 = v34[1];
        BYTE8(v152) = 1;
        *(_QWORD *)&v152 = v36;
        goto LABEL_36;
      }
    }
    else
    {
      v129 = 1;
      while ( v35 != (_QWORD *)-4096LL )
      {
        v130 = v129 + 1;
        v131 = (v156 - 1) & (v33 + v129);
        v33 = v131;
        v34 = (_QWORD *)(v154 + 16 * v131);
        v35 = (_QWORD *)*v34;
        if ( v32 == (_QWORD *)*v34 )
          goto LABEL_34;
        v129 = v130;
      }
    }
  }
  v181.m128i_i64[0] = (__int64)&v167;
  v181.m128i_i64[1] = (__int64)&v153;
  v182 = &v161;
  v189 = (__int64 *)v191;
  v190 = 0x400000000LL;
  v192 = v194;
  v183 = 0;
  v74 = *(_QWORD *)(a5 + 24);
  v184 = 0;
  v185 = 0;
  v186 = 0;
  v187 = 0;
  v188 = 0;
  v193 = 0;
  v194[0] = 0;
  v194[1] = 1;
  v75 = sub_37B98B0((__int64)&v167, v74);
  v147 = v17;
  v139 = sub_37C8D80(&v181, (__int64)v75);
  v76 = v14;
  v157 = 0;
  v158 = 0;
  v159 = 0;
  v160 = 0;
  while ( 2 )
  {
    v77 = (char *)sub_37B98B0((__int64)&v167, *((_QWORD *)v76 + 1));
    v78 = *((_QWORD *)v76 + 2);
    v164 = v77;
    v165 = v78;
    if ( !(unsigned __int8)sub_37B5EC0((__int64)&v157, (__int64 *)&v164, &v151) )
    {
      v81 = v160;
      v82 = v151;
      ++v157;
      v83 = v159 + 1;
      v80 = (unsigned int)(4 * (v159 + 1));
      *(_QWORD *)&v152 = v151;
      if ( (unsigned int)v80 >= 3 * v160 )
      {
        v81 = 2 * v160;
      }
      else
      {
        v79 = v160 >> 3;
        if ( v160 - HIDWORD(v159) - v83 > (unsigned int)v79 )
          goto LABEL_133;
      }
      sub_37B9550((__int64)&v157, v81);
      sub_37B5EC0((__int64)&v157, (__int64 *)&v164, &v152);
      v82 = (char **)v152;
      v83 = v159 + 1;
LABEL_133:
      LODWORD(v159) = v83;
      if ( *v82 != (char *)-4096LL )
        --HIDWORD(v159);
      *v82 = v164;
      v82[1] = (char *)v165;
    }
    v76 += 10;
    if ( v76 != v147 )
      continue;
    break;
  }
  v164 = v166;
  v165 = 0x800000000LL;
  v84 = &v161[(unsigned int)v162];
  if ( v84 == v161 )
  {
    v143 = v166;
    goto LABEL_183;
  }
  v85 = *v161;
  v86 = v166;
  v87 = v161 + 1;
  v88 = 0;
  while ( 1 )
  {
    *(_QWORD *)&v86[8 * v88] = v85;
    v88 = (unsigned int)(v165 + 1);
    LODWORD(v165) = v165 + 1;
    if ( v84 == v87 )
      break;
    v85 = *v87;
    if ( v88 + 1 > (unsigned __int64)HIDWORD(v165) )
    {
      sub_C8D5F0((__int64)&v164, v166, v88 + 1, 8u, v79, v80);
      v88 = (unsigned int)v165;
    }
    v86 = v164;
    ++v87;
  }
  v89 = v164;
  v90 = 8 * v88;
  v143 = &v164[8 * v88];
  if ( v164 == v143 )
    goto LABEL_183;
  v91 = (__int64 *)&v164[8 * v88];
  _BitScanReverse64(&v92, v90 >> 3);
  sub_37C17B0((__int64)v164, v91, 2LL * (int)(63 - (v92 ^ 0x3F)), a1);
  if ( (unsigned __int64)v90 <= 0x80 )
  {
    sub_37C1010(v89, v143, a1);
  }
  else
  {
    sub_37C1010(v89, v89 + 128, a1);
    if ( v143 != v89 + 128 )
    {
      v148 = a1;
      v93 = v89 + 128;
      do
      {
        v94 = v93;
        *(_QWORD *)&v152 = v148;
        v95 = *(_QWORD *)v93;
        while ( 1 )
        {
          v96 = *((_QWORD *)v94 - 1);
          v97 = v94;
          v94 -= 8;
          if ( !sub_37C0D30((__int64 *)&v152, *(__int64 **)(v95 + 80), v96) )
            break;
          *((_QWORD *)v94 + 1) = *(_QWORD *)v94;
        }
        *(_QWORD *)v97 = v95;
        v93 += 8;
      }
      while ( v143 != v93 );
    }
  }
  v143 = &v164[8 * (unsigned int)v165];
  if ( v164 == v143 )
    goto LABEL_183;
  v140 = v164;
  v98 = a3;
  v99 = 8LL * v141;
  while ( 2 )
  {
    v100 = **(__int64 ****)v140;
    v142 = *(char **)(*(_QWORD *)v140 + 80LL);
    v149 = &v100[2 * *(unsigned int *)(*(_QWORD *)v140 + 8LL)];
    v145 = *(_QWORD *)(**(_QWORD **)(*a4 + 8LL * *(int *)(*(_QWORD *)v142 + 24LL)) + v99);
    if ( v149 != v100 )
    {
      v101 = v174 - 1;
      while ( 1 )
      {
        v102 = *v100;
        v103 = **v100;
        if ( (_DWORD)v174 )
        {
          v104 = v101 & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
          v105 = v172[2 * v104];
          if ( v103 == v105 )
            goto LABEL_156;
          v128 = 1;
          while ( v105 != -4096 )
          {
            v104 = v101 & (v128 + v104);
            v105 = v172[2 * v104];
            if ( v103 == v105 )
              goto LABEL_156;
            ++v128;
          }
        }
        v116 = *(int *)(v103 + 24);
        v117 = v160;
        v118 = v158;
        v119 = *(_QWORD **)(*v98 + 8 * v116);
        if ( v160 )
        {
          v120 = (v160 - 1) & (((unsigned int)v102 >> 9) ^ ((unsigned int)v102 >> 4));
          v121 = (__int64 **)(v158 + 16LL * v120);
          v122 = *v121;
          if ( v102 == *v121 )
          {
LABEL_175:
            if ( v121 != (__int64 **)(v158 + 16LL * v160) )
            {
              v123 = v121[1];
              goto LABEL_177;
            }
          }
          else
          {
            v127 = 1;
            while ( v122 != (__int64 *)-4096LL )
            {
              v120 = (v160 - 1) & (v127 + v120);
              v138 = v127 + 1;
              v121 = (__int64 **)(v158 + 16LL * v120);
              v122 = *v121;
              if ( v102 == *v121 )
                goto LABEL_175;
              v127 = v138;
            }
          }
        }
        v123 = *(__int64 **)(**(_QWORD **)(*a4 + 8LL * *(int *)(**(_QWORD **)(*(_QWORD *)v140 + 80LL) + 24LL)) + v99);
LABEL_177:
        if ( v123 != *(__int64 **)(*v119 + v99) )
        {
LABEL_156:
          BYTE8(v152) = 0;
          v143 = v164;
          goto LABEL_157;
        }
        v100 += 2;
        if ( v149 == v100 )
          goto LABEL_179;
      }
    }
    v118 = v158;
    v117 = v160;
LABEL_179:
    *(_QWORD *)&v152 = *(_QWORD *)(*(_QWORD *)v140 + 80LL);
    *((_QWORD *)&v152 + 1) = v145;
    if ( !v117 )
    {
      ++v157;
      v151 = 0;
LABEL_239:
      v137 = 2 * v117;
      goto LABEL_237;
    }
    v124 = (v117 - 1) & (((unsigned int)v142 >> 9) ^ ((unsigned int)v142 >> 4));
    v125 = (char **)(v118 + 16LL * v124);
    v126 = *v125;
    if ( v142 == *v125 )
      goto LABEL_181;
    v134 = 1;
    v135 = 0;
    while ( v126 != (char *)-4096LL )
    {
      if ( v126 == (char *)-8192LL && !v135 )
        v135 = v125;
      v124 = (v117 - 1) & (v134 + v124);
      v125 = (char **)(v118 + 16LL * v124);
      v126 = *v125;
      if ( v142 == *v125 )
        goto LABEL_181;
      ++v134;
    }
    if ( v135 )
      v125 = v135;
    ++v157;
    v136 = v159 + 1;
    v151 = v125;
    if ( 4 * ((int)v159 + 1) >= 3 * v117 )
      goto LABEL_239;
    if ( v117 - (v136 + HIDWORD(v159)) <= v117 >> 3 )
    {
      v137 = v117;
LABEL_237:
      sub_37B9550((__int64)&v157, v137);
      sub_37B5EC0((__int64)&v157, (__int64 *)&v152, &v151);
      v142 = (char *)v152;
      v136 = v159 + 1;
      v125 = v151;
    }
    LODWORD(v159) = v136;
    if ( *v125 != (char *)-4096LL )
      --HIDWORD(v159);
    *v125 = v142;
    v125[1] = (char *)*((_QWORD *)&v152 + 1);
LABEL_181:
    v140 += 8;
    if ( v143 != v140 )
      continue;
    break;
  }
  v143 = v164;
LABEL_183:
  BYTE8(v152) = 1;
  *(_QWORD *)&v152 = v139;
LABEL_157:
  if ( v143 != v166 )
    _libc_free((unsigned __int64)v143);
  sub_C7D6A0(v158, 16LL * v160, 8);
  v106 = v189;
  v107 = &v189[(unsigned int)v190];
  if ( v189 != v107 )
  {
    for ( k = (unsigned __int64)v189; ; k = (unsigned __int64)v189 )
    {
      v109 = *v106;
      v110 = (unsigned int)((__int64)((__int64)v106 - k) >> 3) >> 7;
      v111 = 4096LL << v110;
      if ( v110 >= 0x1E )
        v111 = 0x40000000000LL;
      ++v106;
      sub_C7D6A0(v109, v111, 16);
      if ( v107 == v106 )
        break;
    }
  }
  v112 = v192;
  v113 = &v192[2 * (unsigned int)v193];
  if ( v192 != v113 )
  {
    do
    {
      v114 = v112[1];
      v115 = *v112;
      v112 += 2;
      sub_C7D6A0(v115, v114, 16);
    }
    while ( v113 != v112 );
    v113 = v192;
  }
  if ( v113 != v194 )
    _libc_free((unsigned __int64)v113);
  if ( v189 != (__int64 *)v191 )
    _libc_free((unsigned __int64)v189);
  sub_C7D6A0(v184, 16LL * v186, 8);
LABEL_36:
  if ( v161 != (__int64 *)v163 )
    _libc_free((unsigned __int64)v161);
  sub_C7D6A0(v154, 16LL * v156, 8);
  if ( (_DWORD)v177 )
  {
    v68 = v176;
    v69 = &v176[2 * (unsigned int)v178];
    if ( v176 != v69 )
    {
      while ( 1 )
      {
        v70 = v68;
        if ( *v68 != -8192 && *v68 != -4096 )
          break;
        v68 += 2;
        if ( v69 == v68 )
          goto LABEL_39;
      }
      while ( v70 != v69 )
      {
        v71 = v70[1];
        if ( v71 )
        {
          v72 = *(unsigned __int64 **)(v71 + 16);
          v73 = &v72[12 * *(unsigned int *)(v71 + 24)];
          if ( v72 != v73 )
          {
            do
            {
              v73 -= 12;
              if ( (unsigned __int64 *)*v73 != v73 + 2 )
                _libc_free(*v73);
            }
            while ( v72 != v73 );
            v73 = *(unsigned __int64 **)(v71 + 16);
          }
          if ( v73 != (unsigned __int64 *)(v71 + 32) )
            _libc_free((unsigned __int64)v73);
          j_j___libc_free_0(v71);
        }
        v70 += 2;
        if ( v70 == v69 )
          break;
        while ( *v70 == -8192 || *v70 == -4096 )
        {
          v70 += 2;
          if ( v69 == v70 )
            goto LABEL_39;
        }
      }
    }
  }
LABEL_39:
  ++v167;
  if ( (_DWORD)v169 )
  {
    v59 = 4 * v169;
    v37 = (unsigned int)v170;
    if ( (unsigned int)(4 * v169) < 0x40 )
      v59 = 64;
    if ( (unsigned int)v170 <= v59 )
    {
LABEL_42:
      v38 = v168;
      v39 = &v168[2 * v37];
      if ( v168 != v39 )
      {
        do
        {
          *v38 = -1;
          v38 += 2;
        }
        while ( v39 != v38 );
      }
      goto LABEL_44;
    }
    v60 = v168;
    v61 = 2LL * (unsigned int)v170;
    if ( (_DWORD)v169 == 1 )
    {
      v64 = 86;
    }
    else
    {
      _BitScanReverse(&v62, v169 - 1);
      v63 = 1 << (33 - (v62 ^ 0x1F));
      if ( v63 < 64 )
        v63 = 64;
      if ( (_DWORD)v170 == v63 )
      {
        v169 = 0;
        v133 = &v168[v61];
        do
        {
          if ( v60 )
            *v60 = -1;
          v60 += 2;
        }
        while ( v133 != v60 );
        goto LABEL_45;
      }
      v64 = 4 * v63 / 3u + 1;
    }
    sub_C7D6A0((__int64)v168, v61 * 8, 8);
    v65 = sub_AF1560(v64);
    LODWORD(v170) = v65;
    if ( !v65 )
      goto LABEL_203;
    v66 = (_QWORD *)sub_C7D670(16LL * v65, 8);
    v169 = 0;
    v168 = v66;
    for ( m = &v66[2 * (unsigned int)v170]; m != v66; v66 += 2 )
    {
      if ( v66 )
        *v66 = -1;
    }
  }
  else if ( HIDWORD(v169) )
  {
    v37 = (unsigned int)v170;
    if ( (unsigned int)v170 <= 0x40 )
      goto LABEL_42;
    sub_C7D6A0((__int64)v168, 16LL * (unsigned int)v170, 8);
    LODWORD(v170) = 0;
LABEL_203:
    v168 = 0;
LABEL_44:
    v169 = 0;
  }
LABEL_45:
  ++v171;
  if ( (_DWORD)v173 )
  {
    v55 = 4 * v173;
    v40 = (unsigned int)v174;
    if ( (unsigned int)(4 * v173) < 0x40 )
      v55 = 64;
    if ( v55 >= (unsigned int)v174 )
    {
LABEL_48:
      v41 = v172;
      v42 = &v172[2 * v40];
      if ( v172 != v42 )
      {
        do
        {
          *v41 = -4096;
          v41 += 2;
        }
        while ( v42 != v41 );
      }
      goto LABEL_50;
    }
    if ( (_DWORD)v173 == 1 )
    {
      v57 = 64;
    }
    else
    {
      _BitScanReverse(&v56, v173 - 1);
      v57 = 1 << (33 - (v56 ^ 0x1F));
      if ( v57 < 64 )
        v57 = 64;
      if ( v57 == (_DWORD)v174 )
      {
LABEL_92:
        sub_37C4190((__int64)&v171);
        goto LABEL_51;
      }
    }
    sub_C7D6A0((__int64)v172, 16LL * (unsigned int)v174, 8);
    v58 = sub_37B8280(v57);
    LODWORD(v174) = v58;
    if ( !v58 )
      goto LABEL_201;
    v172 = (_QWORD *)sub_C7D670(16LL * v58, 8);
    goto LABEL_92;
  }
  if ( HIDWORD(v173) )
  {
    v40 = (unsigned int)v174;
    if ( (unsigned int)v174 <= 0x40 )
      goto LABEL_48;
    sub_C7D6A0((__int64)v172, 16LL * (unsigned int)v174, 8);
    LODWORD(v174) = 0;
LABEL_201:
    v172 = 0;
LABEL_50:
    v173 = 0;
  }
LABEL_51:
  ++v175;
  v43 = v176;
  if ( (_DWORD)v177 )
  {
    v47 = 4 * v177;
    if ( (unsigned int)(4 * v177) < 0x40 )
      v47 = 64;
    if ( v47 >= (unsigned int)v178 )
    {
LABEL_54:
      v44 = 2LL * (unsigned int)v178;
      v45 = &v176[v44];
      if ( v176 != &v176[v44] )
      {
        do
        {
          *v43 = -4096;
          v43 += 2;
        }
        while ( v45 != v43 );
        v43 = v176;
        v44 = 2LL * (unsigned int)v178;
      }
      v177 = 0;
      goto LABEL_58;
    }
    v48 = 2LL * (unsigned int)v178;
    if ( (_DWORD)v177 == 1 )
    {
      v51 = 86;
    }
    else
    {
      _BitScanReverse(&v49, v177 - 1);
      v50 = 1 << (33 - (v49 ^ 0x1F));
      if ( v50 < 64 )
        v50 = 64;
      if ( (_DWORD)v178 == v50 )
      {
        v177 = 0;
        v132 = &v176[v48];
        do
        {
          if ( v43 )
            *v43 = -4096;
          v43 += 2;
        }
        while ( v132 != v43 );
        v43 = v176;
        v44 = 2LL * (unsigned int)v178;
        goto LABEL_58;
      }
      v51 = 4 * v50 / 3u + 1;
    }
    sub_C7D6A0((__int64)v176, v48 * 8, 8);
    v52 = sub_AF1560(v51);
    LODWORD(v178) = v52;
    if ( !v52 )
      goto LABEL_205;
    v53 = sub_C7D670(16LL * v52, 8);
    v177 = 0;
    v176 = (_QWORD *)v53;
    v44 = 2LL * (unsigned int)v178;
    v43 = (_QWORD *)(v53 + v44 * 8);
    if ( v53 != v53 + v44 * 8 )
    {
      v54 = (_QWORD *)v53;
      do
      {
        if ( v54 )
          *v54 = -4096;
        v54 += 2;
      }
      while ( v43 != v54 );
      v43 = (_QWORD *)v53;
    }
  }
  else if ( HIDWORD(v177) )
  {
    if ( (unsigned int)v178 <= 0x40 )
      goto LABEL_54;
    sub_C7D6A0((__int64)v176, 16LL * (unsigned int)v178, 8);
    LODWORD(v178) = 0;
LABEL_205:
    v176 = 0;
    v43 = 0;
    v44 = 0;
    v177 = 0;
  }
  else
  {
    v44 = 2LL * (unsigned int)v178;
  }
LABEL_58:
  sub_C7D6A0((__int64)v43, v44 * 8, 8);
  sub_C7D6A0((__int64)v172, 16LL * (unsigned int)v174, 8);
  sub_C7D6A0((__int64)v168, 16LL * (unsigned int)v170, 8);
  return v152;
}
