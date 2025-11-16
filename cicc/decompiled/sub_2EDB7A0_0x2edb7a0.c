// Function: sub_2EDB7A0
// Address: 0x2edb7a0
//
__int64 __fastcall sub_2EDB7A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r14
  char v9; // al
  __int64 v10; // rdx
  __int64 v11; // rcx
  int v12; // r9d
  unsigned int i; // eax
  __int64 v14; // rsi
  unsigned int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rcx
  int v18; // r8d
  unsigned int j; // eax
  __int64 v20; // rsi
  unsigned int v21; // eax
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  unsigned __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  char *v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // r9
  __int64 v38; // r8
  unsigned __int64 v39; // rsi
  unsigned __int64 v40; // r13
  __int64 v41; // rax
  __int64 v42; // rcx
  __m128i *v43; // rdx
  const __m128i *v44; // rax
  __int64 v45; // r8
  __int64 v46; // r9
  const __m128i *v47; // rcx
  unsigned __int64 v48; // r13
  __int64 v49; // rax
  unsigned __int64 v50; // rdi
  __m128i *v51; // rdx
  const __m128i *v52; // rax
  unsigned __int64 v53; // rax
  char *v54; // r13
  int v55; // r15d
  __int64 v56; // rax
  __int64 v57; // rsi
  __int64 v58; // rcx
  __int64 v59; // rdx
  __int64 v60; // rdi
  unsigned __int64 v61; // rax
  __int64 v62; // r10
  unsigned int v63; // r12d
  int v64; // r11d
  int v65; // r11d
  _QWORD *v66; // r10
  int v67; // ecx
  unsigned __int8 v68; // al
  __int64 v69; // rcx
  __int64 v70; // rdx
  int v71; // eax
  __int64 v72; // rdi
  __int64 v73; // r12
  char v74; // r14
  int v75; // r13d
  __int64 v76; // r15
  char v77; // al
  int v78; // eax
  __int64 v79; // rax
  int v80; // eax
  char v81; // al
  __int64 v82; // rdx
  int v83; // r11d
  __int64 v84; // r10
  unsigned int n; // eax
  __int64 v86; // r13
  __int64 v87; // rdi
  unsigned int v88; // eax
  __int64 *v89; // r12
  __int64 v90; // r14
  __int64 *v91; // r13
  __int64 v92; // r14
  __int64 *v93; // r14
  __int64 *v94; // r13
  __int64 *v95; // rcx
  __int64 v96; // rax
  signed __int64 v97; // rax
  __int64 *v98; // r12
  __int64 v99; // r14
  int v100; // esi
  __int64 v101; // r10
  int v102; // edi
  __int64 v103; // rcx
  unsigned int v104; // eax
  unsigned int v105; // eax
  int v106; // r8d
  _QWORD *v107; // rsi
  unsigned int k; // edx
  __int64 v109; // r9
  int v110; // ecx
  unsigned int m; // edx
  _QWORD *v112; // r8
  __int64 v113; // r9
  int v114; // eax
  int v115; // edi
  int v116; // esi
  __int64 v117; // r10
  int v118; // edi
  unsigned int ii; // eax
  __int64 *v120; // rcx
  unsigned int v121; // eax
  __int64 *v122; // r13
  __int64 *v123; // rcx
  __int64 *v124; // r15
  __int64 *v125; // r12
  unsigned __int64 v126; // r13
  unsigned int v127; // edx
  unsigned int v128; // edx
  char *v129; // [rsp+0h] [rbp-3C0h]
  __int64 v130; // [rsp+8h] [rbp-3B8h]
  unsigned __int64 v131; // [rsp+10h] [rbp-3B0h]
  __int64 v132; // [rsp+28h] [rbp-398h]
  __int64 v133; // [rsp+30h] [rbp-390h]
  unsigned __int8 v134; // [rsp+38h] [rbp-388h]
  __int64 v135; // [rsp+40h] [rbp-380h]
  unsigned __int8 v136; // [rsp+48h] [rbp-378h]
  __int64 v137; // [rsp+50h] [rbp-370h] BYREF
  __int64 v138; // [rsp+58h] [rbp-368h]
  __int64 v139; // [rsp+60h] [rbp-360h] BYREF
  __int64 v140; // [rsp+68h] [rbp-358h]
  __int64 v141; // [rsp+70h] [rbp-350h]
  __int64 v142; // [rsp+78h] [rbp-348h]
  __int64 v143; // [rsp+80h] [rbp-340h] BYREF
  __int64 *v144; // [rsp+88h] [rbp-338h]
  __int64 v145; // [rsp+90h] [rbp-330h]
  __int64 v146; // [rsp+98h] [rbp-328h]
  __int64 v147[16]; // [rsp+A0h] [rbp-320h] BYREF
  __int64 v148; // [rsp+120h] [rbp-2A0h] BYREF
  _QWORD *v149; // [rsp+128h] [rbp-298h]
  __int64 v150; // [rsp+130h] [rbp-290h]
  int v151; // [rsp+138h] [rbp-288h]
  char v152; // [rsp+13Ch] [rbp-284h]
  _QWORD v153[8]; // [rsp+140h] [rbp-280h] BYREF
  unsigned __int64 v154; // [rsp+180h] [rbp-240h] BYREF
  __int64 v155; // [rsp+188h] [rbp-238h]
  unsigned __int64 v156; // [rsp+190h] [rbp-230h]
  char v157[8]; // [rsp+1A0h] [rbp-220h] BYREF
  unsigned __int64 v158; // [rsp+1A8h] [rbp-218h]
  char v159; // [rsp+1BCh] [rbp-204h]
  _BYTE v160[64]; // [rsp+1C0h] [rbp-200h] BYREF
  unsigned __int64 v161; // [rsp+200h] [rbp-1C0h]
  __int64 v162; // [rsp+208h] [rbp-1B8h]
  unsigned __int64 v163; // [rsp+210h] [rbp-1B0h]
  char v164[8]; // [rsp+220h] [rbp-1A0h] BYREF
  unsigned __int64 v165; // [rsp+228h] [rbp-198h]
  char v166; // [rsp+23Ch] [rbp-184h]
  _BYTE v167[64]; // [rsp+240h] [rbp-180h] BYREF
  unsigned __int64 v168; // [rsp+280h] [rbp-140h]
  __int64 v169; // [rsp+288h] [rbp-138h]
  __int64 v170; // [rsp+290h] [rbp-130h]
  __m128i v171; // [rsp+2A0h] [rbp-120h] BYREF
  char v172; // [rsp+2B0h] [rbp-110h]
  char v173; // [rsp+2BCh] [rbp-104h]
  char v174[64]; // [rsp+2C0h] [rbp-100h] BYREF
  unsigned __int64 v175; // [rsp+300h] [rbp-C0h]
  __int64 v176; // [rsp+308h] [rbp-B8h]
  unsigned __int64 v177; // [rsp+310h] [rbp-B0h]
  char v178[8]; // [rsp+318h] [rbp-A8h] BYREF
  unsigned __int64 v179; // [rsp+320h] [rbp-A0h]
  char v180; // [rsp+334h] [rbp-8Ch]
  char v181[64]; // [rsp+338h] [rbp-88h] BYREF
  const __m128i *v182; // [rsp+378h] [rbp-48h]
  const __m128i *v183; // [rsp+380h] [rbp-40h]
  __int64 v184; // [rsp+388h] [rbp-38h]

  v5 = a3;
  if ( !(unsigned __int8)sub_2E6D360(*(_QWORD *)(a1 + 32), a2, a3) )
    return 1;
  sub_2EB3EB0(*(_QWORD *)(a1 + 40), v5, a2);
  if ( !v9 )
    return 1;
  v10 = *(unsigned int *)(a1 + 1176);
  v137 = a2;
  v138 = v5;
  v11 = *(_QWORD *)(a1 + 1160);
  if ( (_DWORD)v10 )
  {
    v12 = 1;
    for ( i = (v10 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
                | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)))); ; i = (v10 - 1) & v15 )
    {
      v14 = v11 + 24LL * i;
      if ( *(_QWORD *)v14 == a2 && v5 == *(_QWORD *)(v14 + 8) )
        break;
      if ( *(_QWORD *)v14 == -4096 && *(_QWORD *)(v14 + 8) == -4096 )
        goto LABEL_14;
      v15 = v12 + i;
      ++v12;
    }
    if ( v14 != v11 + 24 * v10 )
      return *(unsigned __int8 *)(v14 + 16);
  }
LABEL_14:
  v16 = *(unsigned int *)(a1 + 1208);
  v17 = *(_QWORD *)(a1 + 1192);
  if ( (_DWORD)v16 )
  {
    v18 = 1;
    for ( j = (v16 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
                | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)))); ; j = (v16 - 1) & v21 )
    {
      v20 = v17 + 80LL * j;
      if ( *(_QWORD *)v20 == a2 && v5 == *(_QWORD *)(v20 + 8) )
        break;
      if ( *(_QWORD *)v20 == -4096 && *(_QWORD *)(v20 + 8) == -4096 )
        goto LABEL_20;
      v21 = v18 + j;
      ++v18;
    }
    if ( v20 != 80 * v16 + v17 )
    {
      v89 = *(__int64 **)(v20 + 16);
      v90 = *(unsigned int *)(v20 + 24);
      v91 = &v89[v90];
      v92 = (v90 * 8) >> 5;
      if ( v92 )
      {
        v93 = &v89[4 * v92];
        while ( !(unsigned __int8)sub_2E8AD10(*v89, *(_QWORD *)(a1 + 80), a4, 0) )
        {
          if ( (unsigned __int8)sub_2E8AD10(v89[1], *(_QWORD *)(a1 + 80), a4, 0) )
            return v91 != v89 + 1;
          if ( (unsigned __int8)sub_2E8AD10(v89[2], *(_QWORD *)(a1 + 80), a4, 0) )
            return v91 != v89 + 2;
          if ( (unsigned __int8)sub_2E8AD10(v89[3], *(_QWORD *)(a1 + 80), a4, 0) )
            return v91 != v89 + 3;
          v89 += 4;
          if ( v93 == v89 )
            goto LABEL_166;
        }
        return v91 != v89;
      }
LABEL_166:
      v97 = (char *)v91 - (char *)v89;
      if ( (char *)v91 - (char *)v89 != 16 )
      {
        if ( v97 != 24 )
        {
          if ( v97 != 8 )
            return 0;
          goto LABEL_181;
        }
        if ( (unsigned __int8)sub_2E8AD10(*v89, *(_QWORD *)(a1 + 80), a4, 0) )
          return v91 != v89;
        ++v89;
      }
      if ( !(unsigned __int8)sub_2E8AD10(*v89, *(_QWORD *)(a1 + 80), a4, 0) )
      {
        ++v89;
LABEL_181:
        v136 = sub_2E8AD10(*v89, *(_QWORD *)(a1 + 80), a4, 0);
        if ( !v136 )
          return v136;
        return v91 != v89;
      }
      return v91 != v89;
    }
  }
LABEL_20:
  memset(v147, 0, 0x78u);
  v150 = 0x100000008LL;
  v149 = v153;
  v147[1] = (__int64)&v147[4];
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v143 = 0;
  v144 = 0;
  v145 = 0;
  v146 = 0;
  LODWORD(v147[2]) = 8;
  BYTE4(v147[3]) = 1;
  v154 = 0;
  v155 = 0;
  v156 = 0;
  v151 = 0;
  v152 = 1;
  v153[0] = a2;
  v148 = 1;
  v171.m128i_i64[0] = a2;
  v172 = 0;
  sub_2ED7AB0(&v154, &v171);
  sub_C8CF70((__int64)v164, v167, 8, (__int64)&v147[4], (__int64)v147);
  v22 = v147[12];
  memset(&v147[12], 0, 24);
  v168 = v22;
  v169 = v147[13];
  v170 = v147[14];
  sub_C8CF70((__int64)v157, v160, 8, (__int64)v153, (__int64)&v148);
  v23 = v154;
  v154 = 0;
  v161 = v23;
  v24 = v155;
  v155 = 0;
  v162 = v24;
  v25 = v156;
  v156 = 0;
  v163 = v25;
  sub_C8CF70((__int64)&v171, v174, 8, (__int64)v160, (__int64)v157);
  v26 = v161;
  v161 = 0;
  v175 = v26;
  v27 = v162;
  v162 = 0;
  v176 = v27;
  v28 = v163;
  v163 = 0;
  v177 = v28;
  sub_C8CF70((__int64)v178, v181, 8, (__int64)v167, (__int64)v164);
  v32 = v168;
  v168 = 0;
  v182 = (const __m128i *)v32;
  v33 = v169;
  v169 = 0;
  v183 = (const __m128i *)v33;
  v34 = v170;
  v170 = 0;
  v184 = v34;
  if ( v161 )
    j_j___libc_free_0(v161);
  if ( !v159 )
    _libc_free(v158);
  if ( v168 )
    j_j___libc_free_0(v168);
  if ( !v166 )
    _libc_free(v165);
  if ( v154 )
    j_j___libc_free_0(v154);
  if ( !v152 )
    _libc_free((unsigned __int64)v149);
  if ( v147[12] )
    j_j___libc_free_0(v147[12]);
  if ( !BYTE4(v147[3]) )
    _libc_free(v147[1]);
  v35 = v157;
  sub_C8CD80((__int64)v157, (__int64)v160, (__int64)&v171, v29, v30, v31);
  v38 = v176;
  v39 = v175;
  v161 = 0;
  v162 = 0;
  v163 = 0;
  v40 = v176 - v175;
  if ( v176 == v175 )
  {
    v40 = 0;
    v42 = 0;
  }
  else
  {
    if ( v40 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_297;
    v41 = sub_22077B0(v176 - v175);
    v38 = v176;
    v39 = v175;
    v42 = v41;
  }
  v161 = v42;
  v162 = v42;
  v163 = v42 + v40;
  if ( v38 != v39 )
  {
    v43 = (__m128i *)v42;
    v44 = (const __m128i *)v39;
    do
    {
      if ( v43 )
      {
        *v43 = _mm_loadu_si128(v44);
        v43[1].m128i_i64[0] = v44[1].m128i_i64[0];
      }
      v44 = (const __m128i *)((char *)v44 + 24);
      v43 = (__m128i *)((char *)v43 + 24);
    }
    while ( v44 != (const __m128i *)v38 );
    v42 += 8 * (((unsigned __int64)&v44[-2].m128i_u64[1] - v39) >> 3) + 24;
  }
  v162 = v42;
  v35 = v164;
  sub_C8CD80((__int64)v164, (__int64)v167, (__int64)v178, v42, v38, v37);
  v47 = v183;
  v39 = (unsigned __int64)v182;
  v168 = 0;
  v169 = 0;
  v170 = 0;
  v48 = (char *)v183 - (char *)v182;
  if ( v183 == v182 )
  {
    v50 = 0;
    goto LABEL_48;
  }
  if ( v48 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_297:
    sub_4261EA(v35, v39, v36);
  v49 = sub_22077B0((char *)v183 - (char *)v182);
  v47 = v183;
  v39 = (unsigned __int64)v182;
  v50 = v49;
LABEL_48:
  v168 = v50;
  v169 = v50;
  v170 = v50 + v48;
  if ( (const __m128i *)v39 == v47 )
  {
    v53 = v50;
  }
  else
  {
    v51 = (__m128i *)v50;
    v52 = (const __m128i *)v39;
    do
    {
      if ( v51 )
      {
        *v51 = _mm_loadu_si128(v52);
        v45 = v52[1].m128i_i64[0];
        v51[1].m128i_i64[0] = v45;
      }
      v52 = (const __m128i *)((char *)v52 + 24);
      v51 = (__m128i *)((char *)v51 + 24);
    }
    while ( v52 != v47 );
    v53 = v50 + 8 * (((unsigned __int64)&v52[-2].m128i_u64[1] - v39) >> 3) + 24;
  }
  v133 = a4;
  v54 = v157;
  v132 = a1 + 1152;
  v130 = a1 + 1184;
  v169 = v53;
  v55 = 0;
  v136 = 0;
  v135 = a2;
  while ( 1 )
  {
    v59 = v162;
    v58 = v161;
    v57 = v162 - v161;
    if ( v162 - v161 == v53 - v50 )
    {
      if ( v161 == v162 )
      {
LABEL_68:
        if ( v50 )
          j_j___libc_free_0(v50);
        if ( !v166 )
          _libc_free(v165);
        if ( v161 )
          j_j___libc_free_0(v161);
        if ( !v159 )
          _libc_free(v158);
        if ( v182 )
          j_j___libc_free_0((unsigned __int64)v182);
        if ( !v180 )
          _libc_free(v179);
        if ( v175 )
          j_j___libc_free_0(v175);
        if ( !v173 )
          _libc_free(v171.m128i_u64[1]);
        if ( !(_BYTE)v55 )
          *(_BYTE *)sub_2ED81F0(v132, &v137) = 0;
        goto LABEL_86;
      }
      v61 = v50;
      while ( 1 )
      {
        v57 = *(_QWORD *)v61;
        if ( *(_QWORD *)v58 != *(_QWORD *)v61 )
          break;
        v57 = *(unsigned __int8 *)(v58 + 16);
        if ( (_BYTE)v57 != *(_BYTE *)(v61 + 16) )
          break;
        if ( (_BYTE)v57 )
        {
          v57 = *(_QWORD *)(v61 + 8);
          if ( *(_QWORD *)(v58 + 8) != v57 )
            break;
        }
        v58 += 24;
        v61 += 24LL;
        if ( v162 == v58 )
          goto LABEL_68;
      }
    }
    v56 = *(_QWORD *)(v162 - 24);
    v147[0] = v56;
    if ( v56 == v5 || v56 == v135 )
      goto LABEL_59;
    v57 = (unsigned int)v142;
    v58 = v140;
    if ( !(_DWORD)v142 )
    {
      ++v139;
LABEL_214:
      sub_2E52D10((__int64)&v139, 2 * v142);
      if ( !(_DWORD)v142 )
        goto LABEL_303;
      v56 = v147[0];
      v106 = 1;
      v107 = 0;
      for ( k = (v142 - 1) & ((LODWORD(v147[0]) >> 9) ^ (LODWORD(v147[0]) >> 4)); ; k = (v142 - 1) & v128 )
      {
        v66 = (_QWORD *)(v140 + 8LL * k);
        v109 = *v66;
        if ( v147[0] == *v66 )
        {
          v67 = v141 + 1;
          goto LABEL_95;
        }
        if ( v109 == -4096 )
          break;
        if ( v109 != -8192 || v107 )
          v66 = v107;
        v128 = v106 + k;
        v107 = v66;
        ++v106;
      }
      v67 = v141 + 1;
      if ( v107 )
        v66 = v107;
      goto LABEL_95;
    }
    v46 = (unsigned int)(v142 - 1);
    v59 = (unsigned int)v46 & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
    v45 = v140 + 8 * v59;
    v60 = *(_QWORD *)v45;
    if ( v56 == *(_QWORD *)v45 )
      goto LABEL_59;
    v62 = *(_QWORD *)v45;
    v63 = v59;
    v64 = 1;
    while ( v62 != -4096 )
    {
      v63 = v46 & (v64 + v63);
      v62 = *(_QWORD *)(v140 + 8LL * v63);
      if ( v56 == v62 )
        goto LABEL_59;
      ++v64;
    }
    v65 = 1;
    v66 = 0;
    while ( v60 != -4096 )
    {
      if ( v60 == -8192 && !v66 )
        v66 = (_QWORD *)v45;
      LODWORD(v59) = v46 & (v65 + v59);
      v45 = v140 + 8LL * (unsigned int)v59;
      v60 = *(_QWORD *)v45;
      if ( v56 == *(_QWORD *)v45 )
        goto LABEL_98;
      ++v65;
    }
    if ( !v66 )
      v66 = (_QWORD *)v45;
    ++v139;
    v67 = v141 + 1;
    if ( 4 * ((int)v141 + 1) >= (unsigned int)(3 * v142) )
      goto LABEL_214;
    if ( (int)v142 - HIDWORD(v141) - v67 <= (unsigned int)v142 >> 3 )
    {
      sub_2E52D10((__int64)&v139, v142);
      if ( !(_DWORD)v142 )
      {
LABEL_303:
        LODWORD(v141) = v141 + 1;
        BUG();
      }
      v56 = v147[0];
      v66 = 0;
      v110 = 1;
      for ( m = (v142 - 1) & ((LODWORD(v147[0]) >> 9) ^ (LODWORD(v147[0]) >> 4)); ; m = (v142 - 1) & v127 )
      {
        v112 = (_QWORD *)(v140 + 8LL * m);
        v113 = *v112;
        if ( v147[0] == *v112 )
        {
          v66 = (_QWORD *)(v140 + 8LL * m);
          v67 = v141 + 1;
          goto LABEL_95;
        }
        if ( v113 == -4096 )
          break;
        if ( v66 || v113 != -8192 )
          v112 = v66;
        v127 = v110 + m;
        v66 = v112;
        ++v110;
      }
      v67 = v141 + 1;
      if ( !v66 )
        v66 = (_QWORD *)(v140 + 8LL * m);
    }
LABEL_95:
    LODWORD(v141) = v67;
    if ( *v66 != -4096 )
      --HIDWORD(v141);
    *v66 = v56;
LABEL_98:
    v57 = v5;
    sub_2EB3EB0(*(_QWORD *)(a1 + 40), v5, v147[0]);
    v134 = v68;
    if ( !v68 )
      goto LABEL_59;
    v69 = v147[0];
    if ( !(_DWORD)v146 )
      goto LABEL_175;
    v70 = (unsigned int)(v146 - 1);
    v71 = v70 & ((LODWORD(v147[0]) >> 9) ^ (LODWORD(v147[0]) >> 4));
    v72 = v144[v71];
    if ( v147[0] != v72 )
    {
      LODWORD(v45) = 1;
      while ( v72 != -4096 )
      {
        v71 = v70 & (v45 + v71);
        v72 = v144[v71];
        if ( v147[0] == v72 )
          goto LABEL_101;
        LODWORD(v45) = v45 + 1;
      }
LABEL_175:
      sub_2EDB520((__int64)&v148, (__int64)&v143, v147);
      v72 = v147[0];
    }
LABEL_101:
    v57 = (unsigned int)dword_5022068;
    if ( (unsigned __int8)sub_2E33380(v72, dword_5022068, v70, v69, v45) || dword_5021F88 < (unsigned int)v145 )
      break;
    v73 = *(_QWORD *)(v147[0] + 56);
    if ( v147[0] + 48 == v73 )
      goto LABEL_59;
    v131 = v5;
    v74 = v136;
    v129 = v54;
    v75 = v55;
    v76 = v147[0] + 48;
    while ( 2 )
    {
      v80 = *(_DWORD *)(v73 + 44);
      if ( (v80 & 4) != 0 || (v80 & 8) == 0 )
      {
        v77 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v73 + 16) + 24LL) >> 7;
      }
      else
      {
        v57 = 128;
        v77 = sub_2E88A90(v73, 128, 1);
      }
      if ( v77 || (unsigned __int8)sub_2E8B100(v73, v57, v59, v58, (_QWORD *)v45) )
      {
        v94 = v144;
        v95 = &v144[(unsigned int)v146];
        if ( (_DWORD)v145 && v144 != v95 )
        {
          while ( *v94 == -8192 || *v94 == -4096 )
          {
            if ( v95 == ++v94 )
              goto LABEL_145;
          }
          if ( v94 != v95 )
          {
            v98 = &v144[(unsigned int)v146];
            do
            {
              v99 = *v94;
              if ( *v94 != v147[0] )
              {
                if ( (unsigned __int8)sub_2E6D360(*(_QWORD *)(a1 + 32), *v94, v147[0]) )
                {
                  v148 = v99;
                  v149 = (_QWORD *)v131;
                  *(_BYTE *)sub_2ED7EC0(v132, &v148) = 1;
                }
                else if ( v99 != v147[0] && (unsigned __int8)sub_2E6D360(*(_QWORD *)(a1 + 32), v147[0], v99) )
                {
                  v149 = (_QWORD *)v99;
                  v148 = v135;
                  *(_BYTE *)sub_2ED7EC0(v132, &v148) = 1;
                }
              }
              if ( ++v94 == v98 )
                break;
              while ( *v94 == -8192 || *v94 == -4096 )
              {
                if ( v98 == ++v94 )
                  goto LABEL_145;
              }
            }
            while ( v94 != v98 );
          }
        }
        goto LABEL_145;
      }
      if ( (unsigned int)*(unsigned __int16 *)(v73 + 68) - 1 > 1
        || (*(_BYTE *)(*(_QWORD *)(v73 + 32) + 64LL) & 0x10) == 0 )
      {
        v78 = *(_DWORD *)(v73 + 44);
        if ( (v78 & 4) != 0 || (v78 & 8) == 0 )
        {
          v79 = (*(_QWORD *)(*(_QWORD *)(v73 + 16) + 24LL) >> 20) & 1LL;
        }
        else
        {
          v57 = 0x100000;
          LOBYTE(v79) = sub_2E88A90(v73, 0x100000, 1);
        }
        if ( !(_BYTE)v79 )
          goto LABEL_115;
      }
      v81 = sub_2E8AD10(v73, *(_QWORD *)(a1 + 80), v133, 0);
      v57 = *(unsigned int *)(a1 + 1208);
      if ( v81 )
        v74 = v81;
      if ( !(_DWORD)v57 )
      {
        ++*(_QWORD *)(a1 + 1184);
        goto LABEL_204;
      }
      v82 = v137;
      v83 = 1;
      v84 = 0;
      v46 = *(_QWORD *)(a1 + 1192);
      v45 = (unsigned int)(v57 - 1);
      for ( n = v45
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v138 >> 9) ^ ((unsigned int)v138 >> 4)
                  | ((unsigned __int64)(((unsigned int)v137 >> 9) ^ ((unsigned int)v137 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v138 >> 9) ^ ((unsigned int)v138 >> 4)))); ; n = v45 & v88 )
      {
        v86 = v46 + 80LL * n;
        v87 = *(_QWORD *)v86;
        if ( *(_QWORD *)v86 == v137 && *(_QWORD *)(v86 + 8) == v138 )
        {
          v59 = *(unsigned int *)(v86 + 24);
          v96 = v86 + 16;
          v45 = v59 + 1;
          if ( *(unsigned int *)(v86 + 28) < (unsigned __int64)(v59 + 1) )
          {
            v57 = v86 + 32;
            sub_C8D5F0(v86 + 16, (const void *)(v86 + 32), v59 + 1, 8u, v45, v46);
            v59 = *(unsigned int *)(v86 + 24);
            v96 = v86 + 16;
          }
          goto LABEL_165;
        }
        if ( v87 == -4096 )
          break;
        if ( v87 == -8192 && *(_QWORD *)(v86 + 8) == -8192 && !v84 )
          v84 = v46 + 80LL * n;
LABEL_133:
        v88 = v83 + n;
        ++v83;
      }
      if ( *(_QWORD *)(v86 + 8) != -4096 )
        goto LABEL_133;
      v114 = *(_DWORD *)(a1 + 1200);
      if ( v84 )
        v86 = v84;
      ++*(_QWORD *)(a1 + 1184);
      v115 = v114 + 1;
      if ( 4 * (v114 + 1) < (unsigned int)(3 * v57) )
      {
        if ( (int)v57 - *(_DWORD *)(a1 + 1204) - v115 > (unsigned int)v57 >> 3 )
          goto LABEL_234;
        sub_2ED8520(v130, v57);
        v116 = *(_DWORD *)(a1 + 1208);
        if ( v116 )
        {
          v82 = v137;
          v57 = (unsigned int)(v116 - 1);
          v86 = 0;
          v45 = v138;
          v117 = *(_QWORD *)(a1 + 1192);
          v118 = 1;
          for ( ii = v57
                   & (((0xBF58476D1CE4E5B9LL
                      * (((unsigned int)v138 >> 9) ^ ((unsigned int)v138 >> 4)
                       | ((unsigned __int64)(((unsigned int)v137 >> 9) ^ ((unsigned int)v137 >> 4)) << 32))) >> 31)
                    ^ (484763065 * (((unsigned int)v138 >> 9) ^ ((unsigned int)v138 >> 4)))); ; ii = v57 & v121 )
          {
            v120 = (__int64 *)(v117 + 80LL * ii);
            v46 = *v120;
            if ( *v120 == v137 && v120[1] == v138 )
            {
              v86 = v117 + 80LL * ii;
              v115 = *(_DWORD *)(a1 + 1200) + 1;
              goto LABEL_234;
            }
            if ( v46 == -4096 )
            {
              if ( v120[1] == -4096 )
              {
                if ( !v86 )
                  v86 = v117 + 80LL * ii;
                v115 = *(_DWORD *)(a1 + 1200) + 1;
                goto LABEL_234;
              }
            }
            else if ( v46 == -8192 && v120[1] == -8192 && !v86 )
            {
              v86 = v117 + 80LL * ii;
            }
            v121 = v118 + ii;
            ++v118;
          }
        }
LABEL_304:
        ++*(_DWORD *)(a1 + 1200);
        BUG();
      }
LABEL_204:
      sub_2ED8520(v130, 2 * v57);
      v100 = *(_DWORD *)(a1 + 1208);
      if ( !v100 )
        goto LABEL_304;
      v82 = v137;
      v45 = v138;
      v57 = (unsigned int)(v100 - 1);
      v101 = *(_QWORD *)(a1 + 1192);
      v102 = 1;
      v103 = 0;
      v104 = v57
           & (((0xBF58476D1CE4E5B9LL
              * (((unsigned int)v138 >> 9) ^ ((unsigned int)v138 >> 4)
               | ((unsigned __int64)(((unsigned int)v137 >> 9) ^ ((unsigned int)v137 >> 4)) << 32))) >> 31)
            ^ (484763065 * (((unsigned int)v138 >> 9) ^ ((unsigned int)v138 >> 4))));
      while ( 2 )
      {
        v86 = v101 + 80LL * v104;
        v46 = *(_QWORD *)v86;
        if ( *(_QWORD *)v86 == v137 && *(_QWORD *)(v86 + 8) == v138 )
        {
          v115 = *(_DWORD *)(a1 + 1200) + 1;
          goto LABEL_234;
        }
        if ( v46 != -4096 )
        {
          if ( v46 == -8192 && *(_QWORD *)(v86 + 8) == -8192 && !v103 )
            v103 = v101 + 80LL * v104;
          goto LABEL_212;
        }
        if ( *(_QWORD *)(v86 + 8) != -4096 )
        {
LABEL_212:
          v105 = v102 + v104;
          ++v102;
          v104 = v57 & v105;
          continue;
        }
        break;
      }
      if ( v103 )
        v86 = v103;
      v115 = *(_DWORD *)(a1 + 1200) + 1;
LABEL_234:
      *(_DWORD *)(a1 + 1200) = v115;
      if ( *(_QWORD *)v86 != -4096 || *(_QWORD *)(v86 + 8) != -4096 )
        --*(_DWORD *)(a1 + 1204);
      *(_QWORD *)v86 = v82;
      v59 = 0;
      *(_QWORD *)(v86 + 8) = v138;
      *(_QWORD *)(v86 + 16) = v86 + 32;
      *(_QWORD *)(v86 + 24) = 0x600000000LL;
      v96 = v86 + 16;
LABEL_165:
      v58 = *(_QWORD *)v96;
      v75 = v134;
      *(_QWORD *)(*(_QWORD *)v96 + 8 * v59) = v73;
      ++*(_DWORD *)(v96 + 8);
LABEL_115:
      if ( (*(_BYTE *)v73 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v73 + 44) & 8) != 0 )
          v73 = *(_QWORD *)(v73 + 8);
      }
      v73 = *(_QWORD *)(v73 + 8);
      if ( v76 != v73 )
        continue;
      break;
    }
    v136 = v74;
    v55 = v75;
    v5 = v131;
    v54 = v129;
LABEL_59:
    sub_2ED7AF0((__int64)v54, v57, v59, v58, v45, v46);
    v50 = v168;
    v53 = v169;
  }
  v122 = v144;
  v123 = &v144[(unsigned int)v146];
  if ( (_DWORD)v145 && v123 != v144 )
  {
    while ( *v122 == -4096 || *v122 == -8192 )
    {
      if ( v123 == ++v122 )
        goto LABEL_145;
    }
    if ( v123 != v122 )
    {
      v124 = v122;
      v125 = &v144[(unsigned int)v146];
      do
      {
        v126 = *v124;
        if ( *v124 != v147[0] )
        {
          if ( (unsigned __int8)sub_2E6D360(*(_QWORD *)(a1 + 32), *v124, v147[0]) )
          {
            v148 = v126;
            v149 = (_QWORD *)v5;
            *(_BYTE *)sub_2ED7EC0(v132, &v148) = 1;
          }
          else if ( v126 != v147[0] && (unsigned __int8)sub_2E6D360(*(_QWORD *)(a1 + 32), v147[0], v126) )
          {
            v149 = (_QWORD *)v126;
            v148 = v135;
            *(_BYTE *)sub_2ED7EC0(v132, &v148) = 1;
          }
        }
        if ( ++v124 == v125 )
          break;
        while ( *v124 == -8192 || *v124 == -4096 )
        {
          if ( v125 == ++v124 )
            goto LABEL_145;
        }
      }
      while ( v124 != v125 );
    }
  }
LABEL_145:
  *(_BYTE *)sub_2ED81F0(v132, &v137) = 1;
  if ( v168 )
    j_j___libc_free_0(v168);
  if ( !v166 )
    _libc_free(v165);
  if ( v161 )
    j_j___libc_free_0(v161);
  if ( !v159 )
    _libc_free(v158);
  if ( v182 )
    j_j___libc_free_0((unsigned __int64)v182);
  if ( !v180 )
    _libc_free(v179);
  if ( v175 )
    j_j___libc_free_0(v175);
  if ( !v173 )
    _libc_free(v171.m128i_u64[1]);
  v136 = v134;
LABEL_86:
  sub_C7D6A0((__int64)v144, 8LL * (unsigned int)v146, 8);
  sub_C7D6A0(v140, 8LL * (unsigned int)v142, 8);
  return v136;
}
