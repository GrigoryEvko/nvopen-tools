// Function: sub_1A761E0
// Address: 0x1a761e0
//
__int64 *__fastcall sub_1A761E0(__int64 a1)
{
  __int64 v1; // r15
  _QWORD *v2; // rdi
  _QWORD *v3; // rax
  _QWORD *v4; // r12
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // rdi
  unsigned int i; // r15d
  int v9; // eax
  const __m128i *v10; // rsi
  __int32 v11; // ebx
  _BYTE *v12; // rsi
  _QWORD *v13; // rdi
  const __m128i *v14; // rcx
  const __m128i *v15; // rdx
  unsigned __int64 v16; // rbx
  __m128i *v17; // rax
  __m128i *v18; // rcx
  __m128i *v19; // rax
  __m128i *v20; // rax
  __int8 *v21; // rax
  __m128i *v22; // rcx
  unsigned __int64 v23; // rbx
  __m128i *v24; // rax
  __m128i *v25; // rcx
  __m128i *v26; // rax
  __m128i *v27; // rax
  __int8 *v28; // rax
  __m128i *v29; // rax
  _QWORD *v30; // r13
  _QWORD *v31; // r12
  __int32 v32; // esi
  __m128i *v33; // rdi
  unsigned int v34; // edx
  __m128i *v35; // rax
  __int64 v36; // r8
  int v37; // edx
  __int64 v38; // rax
  __int64 v39; // r14
  unsigned int v40; // esi
  unsigned __int32 v41; // edx
  unsigned __int32 v42; // edi
  unsigned int v43; // r8d
  __int64 *v44; // r14
  __int64 v45; // rdx
  __int64 *result; // rax
  unsigned int v47; // r9d
  __int64 v48; // r13
  __int64 *v49; // r11
  __int64 v50; // rsi
  __int64 v51; // rcx
  __int64 v52; // rdi
  unsigned int j; // ebx
  __int64 v54; // r8
  unsigned int v55; // ebx
  unsigned int v56; // esi
  __int64 *v57; // rcx
  __int64 v58; // r10
  _QWORD **v59; // rcx
  _QWORD *v60; // rcx
  __int64 *v61; // r10
  __int64 v62; // rsi
  __int64 v63; // rsi
  __int64 *v64; // rcx
  __int64 v65; // rax
  __m128i *v66; // r8
  __m128i *v67; // r9
  __m128i *v68; // rdi
  __int32 v69; // esi
  unsigned int v70; // edx
  __m128i *v71; // rax
  int v72; // edx
  __int64 v73; // rax
  __int64 v74; // rdx
  unsigned int v75; // ebx
  unsigned __int64 v76; // r10
  unsigned int v77; // esi
  __int64 *v78; // rcx
  __int64 v79; // r8
  _QWORD **v80; // rcx
  _QWORD *v81; // rcx
  unsigned int v82; // esi
  __int64 **v83; // r12
  __int8 v84; // r11
  __m128i *v85; // rcx
  __int32 v86; // edx
  unsigned int v87; // edi
  __m128i *v88; // r10
  __int64 v89; // rsi
  __int32 v90; // ebx
  int v91; // r8d
  int v92; // r9d
  __int64 v93; // r10
  unsigned int v94; // esi
  unsigned __int32 v95; // edx
  __m128i *v96; // rax
  unsigned __int32 v97; // ecx
  unsigned int v98; // edi
  __int64 v99; // rax
  int v100; // ebx
  int v101; // ecx
  __m128i *v102; // rdi
  __int32 v103; // ecx
  unsigned int v104; // r8d
  __int64 v105; // rsi
  int v106; // r9d
  __m128i *v107; // rdx
  unsigned __int32 v108; // edx
  unsigned __int32 v109; // edi
  int v110; // r11d
  __m128i *v111; // r9
  __m128i *v112; // rsi
  __int32 v113; // ecx
  unsigned int v114; // r8d
  __int64 v115; // rdi
  int v116; // r9d
  unsigned __int64 v117; // r14
  __m128i *v118; // rdi
  __int32 v119; // esi
  unsigned int v120; // ecx
  __int64 v121; // r8
  int v122; // r9d
  __m128i *v123; // rdx
  __m128i *v124; // rdi
  __int32 v125; // esi
  unsigned int v126; // ecx
  __int64 v127; // r8
  int v128; // r9d
  int v129; // ecx
  int v130; // r11d
  int v131; // r11d
  __m128i *v132; // r10
  __int32 v133; // esi
  unsigned int v134; // ecx
  int v135; // edi
  __m128i *v136; // rdx
  __int32 v137; // r10d
  unsigned int v138; // ecx
  __m128i *v139; // rdi
  int v140; // esi
  __m128i *v141; // r10
  int v142; // r11d
  const void *v143; // [rsp+8h] [rbp-398h]
  unsigned int v144; // [rsp+1Ch] [rbp-384h]
  __int64 *v145; // [rsp+20h] [rbp-380h]
  __int64 v146; // [rsp+28h] [rbp-378h]
  __int64 v147; // [rsp+30h] [rbp-370h]
  unsigned int v148; // [rsp+30h] [rbp-370h]
  __int64 *v149; // [rsp+38h] [rbp-368h]
  _QWORD *v150; // [rsp+40h] [rbp-360h] BYREF
  _QWORD *v151; // [rsp+48h] [rbp-358h]
  __int64 v152; // [rsp+50h] [rbp-350h]
  __int64 v153; // [rsp+60h] [rbp-340h] BYREF
  _QWORD *v154; // [rsp+68h] [rbp-338h]
  _QWORD *v155; // [rsp+70h] [rbp-330h]
  __int64 v156; // [rsp+78h] [rbp-328h]
  int v157; // [rsp+80h] [rbp-320h]
  _QWORD v158[8]; // [rsp+88h] [rbp-318h] BYREF
  const __m128i *v159; // [rsp+C8h] [rbp-2D8h] BYREF
  __m128i *v160; // [rsp+D0h] [rbp-2D0h]
  __m128i *v161; // [rsp+D8h] [rbp-2C8h]
  _QWORD v162[16]; // [rsp+E0h] [rbp-2C0h] BYREF
  _QWORD v163[2]; // [rsp+160h] [rbp-240h] BYREF
  unsigned __int64 v164; // [rsp+170h] [rbp-230h]
  _BYTE v165[64]; // [rsp+188h] [rbp-218h] BYREF
  __m128i *v166; // [rsp+1C8h] [rbp-1D8h]
  __m128i *v167; // [rsp+1D0h] [rbp-1D0h]
  __int8 *v168; // [rsp+1D8h] [rbp-1C8h]
  _QWORD v169[2]; // [rsp+1E0h] [rbp-1C0h] BYREF
  unsigned __int64 v170; // [rsp+1F0h] [rbp-1B0h]
  char v171[64]; // [rsp+208h] [rbp-198h] BYREF
  __m128i *v172; // [rsp+248h] [rbp-158h]
  __m128i *v173; // [rsp+250h] [rbp-150h]
  __int8 *v174; // [rsp+258h] [rbp-148h]
  _QWORD v175[2]; // [rsp+260h] [rbp-140h] BYREF
  unsigned __int64 v176; // [rsp+270h] [rbp-130h]
  _BYTE v177[64]; // [rsp+288h] [rbp-118h] BYREF
  __m128i *v178; // [rsp+2C8h] [rbp-D8h]
  __m128i *v179; // [rsp+2D0h] [rbp-D0h]
  __int8 *v180; // [rsp+2D8h] [rbp-C8h]
  __m128i v181; // [rsp+2E0h] [rbp-C0h] BYREF
  __m128i v182; // [rsp+2F0h] [rbp-B0h] BYREF
  char v183[64]; // [rsp+308h] [rbp-98h] BYREF
  __m128i *v184; // [rsp+348h] [rbp-58h]
  __m128i *v185; // [rsp+350h] [rbp-50h]
  __int8 *v186; // [rsp+358h] [rbp-48h]
  char v187; // [rsp+370h] [rbp-30h] BYREF

  v1 = a1;
  v2 = *(_QWORD **)(a1 + 200);
  v150 = 0;
  v151 = 0;
  v152 = 0;
  v3 = sub_1444E60(v2, *v2 & 0xFFFFFFFFFFFFFFF8LL);
  v159 = 0;
  v4 = v3;
  memset(v162, 0, sizeof(v162));
  LODWORD(v162[3]) = 8;
  v162[1] = &v162[5];
  v162[2] = &v162[5];
  v154 = v158;
  v155 = v158;
  v156 = 0x100000008LL;
  v160 = 0;
  v161 = 0;
  v157 = 0;
  v158[0] = v3;
  v153 = 1;
  v5 = v158[0] & 0xFFFFFFFFFFFFFFF9LL | (*(_QWORD *)v158[0] >> 1) & 2LL;
  v147 = v5;
  v6 = sub_157EBA0(*v3 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v5 & 2) != 0 )
  {
    v117 = v5;
    if ( *(_QWORD *)((v5 & 0xFFFFFFFFFFFFFFF8LL) + 32) == *(_QWORD *)(*(_QWORD *)((v5 & 0xFFFFFFFFFFFFFFF8LL) + 8) + 32LL) )
      v117 = v5 & 0xFFFFFFFFFFFFFFF9LL | 4;
    v181.m128i_i64[0] = (__int64)v4;
    v10 = 0;
    v181.m128i_i64[1] = v117;
    v182.m128i_i64[0] = v6;
    v182.m128i_i32[2] = 0;
    goto LABEL_192;
  }
  v146 = v1;
  v7 = v6;
  for ( i = 0; ; ++i )
  {
    v9 = 0;
    if ( v7 )
      v9 = sub_15F4D60(v7);
    if ( v9 == i || *(_QWORD *)(*(_QWORD *)((v5 & 0xFFFFFFFFFFFFFFF8LL) + 8) + 32LL) != sub_15F4DF0(v6, i) )
      break;
    v7 = sub_157EBA0(*v4 & 0xFFFFFFFFFFFFFFF8LL);
  }
  v10 = v160;
  v11 = i;
  v181.m128i_i64[0] = (__int64)v4;
  v182.m128i_i64[0] = v6;
  v181.m128i_i64[1] = v147;
  v1 = v146;
  v182.m128i_i32[2] = v11;
  if ( v160 == v161 )
  {
LABEL_192:
    sub_1A752D0(&v159, v10, &v181);
    goto LABEL_12;
  }
  if ( v160 )
  {
    *v160 = _mm_loadu_si128(&v181);
    v10[1] = _mm_loadu_si128(&v182);
    v10 = v160;
  }
  v160 = (__m128i *)&v10[2];
LABEL_12:
  sub_1A75460((__int64)&v153);
  v12 = v177;
  v13 = v175;
  sub_16CCCB0(v175, (__int64)v177, (__int64)v162);
  v14 = (const __m128i *)v162[14];
  v15 = (const __m128i *)v162[13];
  v178 = 0;
  v179 = 0;
  v180 = 0;
  v16 = v162[14] - v162[13];
  if ( v162[14] == v162[13] )
  {
    v17 = 0;
  }
  else
  {
    if ( v16 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_280;
    v17 = (__m128i *)sub_22077B0(v162[14] - v162[13]);
    v14 = (const __m128i *)v162[14];
    v15 = (const __m128i *)v162[13];
  }
  v178 = v17;
  v179 = v17;
  v180 = &v17->m128i_i8[v16];
  if ( v15 == v14 )
  {
    v18 = v17;
  }
  else
  {
    v18 = (__m128i *)((char *)v17 + (char *)v14 - (char *)v15);
    do
    {
      if ( v17 )
      {
        *v17 = _mm_loadu_si128(v15);
        v17[1] = _mm_loadu_si128(v15 + 1);
      }
      v17 += 2;
      v15 += 2;
    }
    while ( v17 != v18 );
  }
  v179 = v18;
  sub_16CCEE0(&v181, (__int64)v183, 8, (__int64)v175);
  v19 = v178;
  v13 = v163;
  v12 = v165;
  v178 = 0;
  v184 = v19;
  v20 = v179;
  v179 = 0;
  v185 = v20;
  v21 = v180;
  v180 = 0;
  v186 = v21;
  sub_16CCCB0(v163, (__int64)v165, (__int64)&v153);
  v22 = v160;
  v15 = v159;
  v166 = 0;
  v167 = 0;
  v168 = 0;
  v23 = (char *)v160 - (char *)v159;
  if ( v160 != v159 )
  {
    if ( v23 <= 0x7FFFFFFFFFFFFFE0LL )
    {
      v24 = (__m128i *)sub_22077B0((char *)v160 - (char *)v159);
      v22 = v160;
      v15 = v159;
      goto LABEL_23;
    }
LABEL_280:
    sub_4261EA(v13, v12, v15);
  }
  v24 = 0;
LABEL_23:
  v166 = v24;
  v167 = v24;
  v168 = &v24->m128i_i8[v23];
  if ( v22 == v15 )
  {
    v25 = v24;
  }
  else
  {
    v25 = (__m128i *)((char *)v24 + (char *)v22 - (char *)v15);
    do
    {
      if ( v24 )
      {
        *v24 = _mm_loadu_si128(v15);
        v24[1] = _mm_loadu_si128(v15 + 1);
      }
      v24 += 2;
      v15 += 2;
    }
    while ( v24 != v25 );
  }
  v167 = v25;
  sub_16CCEE0(v169, (__int64)v171, 8, (__int64)v163);
  v26 = v166;
  v166 = 0;
  v172 = v26;
  v27 = v167;
  v167 = 0;
  v173 = v27;
  v28 = v168;
  v168 = 0;
  v174 = v28;
  sub_1A758E0((__int64)v169, (__int64)&v181, (__int64)&v150);
  if ( v172 )
    j_j___libc_free_0(v172, v174 - (__int8 *)v172);
  if ( v170 != v169[1] )
    _libc_free(v170);
  if ( v166 )
    j_j___libc_free_0(v166, v168 - (__int8 *)v166);
  if ( v164 != v163[1] )
    _libc_free(v164);
  if ( v184 )
    j_j___libc_free_0(v184, v186 - (__int8 *)v184);
  if ( v182.m128i_i64[0] != v181.m128i_i64[1] )
    _libc_free(v182.m128i_u64[0]);
  if ( v178 )
    j_j___libc_free_0(v178, v180 - (__int8 *)v178);
  if ( v176 != v175[1] )
    _libc_free(v176);
  if ( v159 )
    j_j___libc_free_0(v159, (char *)v161 - (char *)v159);
  if ( v155 != v154 )
    _libc_free((unsigned __int64)v155);
  if ( v162[13] )
    j_j___libc_free_0(v162[13], v162[15] - v162[13]);
  if ( v162[2] != v162[1] )
    _libc_free(v162[2]);
  v29 = &v182;
  v181.m128i_i64[0] = 0;
  v181.m128i_i64[1] = 1;
  do
  {
    v29->m128i_i64[0] = -8;
    ++v29;
  }
  while ( v29 != (__m128i *)&v187 );
  v30 = v151;
  v31 = v150;
  if ( v151 == v150 )
  {
    result = *(__int64 **)(v1 + 232);
    v45 = *(unsigned int *)(v1 + 240);
    goto LABEL_249;
  }
LABEL_60:
  while ( 2 )
  {
    v38 = sub_1A6D2B0(v1, (__int64 *)*(v30 - 1));
    v39 = v38;
    if ( (v181.m128i_i8[8] & 1) != 0 )
    {
      v32 = 7;
      v33 = &v182;
    }
    else
    {
      v40 = v182.m128i_u32[2];
      v33 = (__m128i *)v182.m128i_i64[0];
      if ( !v182.m128i_i32[2] )
      {
        v41 = v181.m128i_u32[2];
        ++v181.m128i_i64[0];
        v35 = 0;
        v42 = ((unsigned __int32)v181.m128i_i32[2] >> 1) + 1;
LABEL_64:
        v43 = 3 * v40;
        goto LABEL_65;
      }
      v32 = v182.m128i_i32[2] - 1;
    }
    v34 = v32 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
    v35 = &v33[v34];
    v36 = v35->m128i_i64[0];
    if ( v39 == v35->m128i_i64[0] )
    {
      v37 = v35->m128i_i32[2] + 1;
      goto LABEL_59;
    }
    v110 = 1;
    v111 = 0;
    while ( v36 != -8 )
    {
      if ( v111 || v36 != -16 )
        v35 = v111;
      v34 = v32 & (v110 + v34);
      v141 = &v33[v34];
      v36 = v141->m128i_i64[0];
      if ( v39 == v141->m128i_i64[0] )
      {
        v37 = v141->m128i_i32[2] + 1;
        v35 = v141;
LABEL_59:
        --v30;
        v35->m128i_i32[2] = v37;
        if ( v31 == v30 )
          goto LABEL_70;
        goto LABEL_60;
      }
      ++v110;
      v111 = v35;
      v35 = &v33[v34];
    }
    v41 = v181.m128i_u32[2];
    v43 = 24;
    v40 = 8;
    if ( v111 )
      v35 = v111;
    ++v181.m128i_i64[0];
    v42 = ((unsigned __int32)v181.m128i_i32[2] >> 1) + 1;
    if ( (v181.m128i_i8[8] & 1) == 0 )
    {
      v40 = v182.m128i_u32[2];
      goto LABEL_64;
    }
LABEL_65:
    if ( 4 * v42 >= v43 )
    {
      sub_1A6EFA0((__int64)&v181, 2 * v40);
      if ( (v181.m128i_i8[8] & 1) != 0 )
      {
        v119 = 7;
        v118 = &v182;
      }
      else
      {
        v118 = (__m128i *)v182.m128i_i64[0];
        if ( !v182.m128i_i32[2] )
          goto LABEL_292;
        v119 = v182.m128i_i32[2] - 1;
      }
      v41 = v181.m128i_u32[2];
      v120 = v119 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
      v35 = &v118[v120];
      v121 = v35->m128i_i64[0];
      if ( v39 == v35->m128i_i64[0] )
        goto LABEL_67;
      v122 = 1;
      v123 = 0;
      while ( v121 != -8 )
      {
        if ( v121 == -16 && !v123 )
          v123 = v35;
        v120 = v119 & (v122 + v120);
        v35 = &v118[v120];
        v121 = v35->m128i_i64[0];
        if ( v39 == v35->m128i_i64[0] )
          goto LABEL_216;
        ++v122;
      }
      goto LABEL_214;
    }
    if ( v40 - v181.m128i_i32[3] - v42 > v40 >> 3 )
      goto LABEL_67;
    sub_1A6EFA0((__int64)&v181, v40);
    if ( (v181.m128i_i8[8] & 1) != 0 )
    {
      v125 = 7;
      v124 = &v182;
      goto LABEL_211;
    }
    v124 = (__m128i *)v182.m128i_i64[0];
    if ( !v182.m128i_i32[2] )
    {
LABEL_292:
      v181.m128i_i32[2] = (2 * ((unsigned __int32)v181.m128i_i32[2] >> 1) + 2) | v181.m128i_i8[8] & 1;
      BUG();
    }
    v125 = v182.m128i_i32[2] - 1;
LABEL_211:
    v41 = v181.m128i_u32[2];
    v126 = v125 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
    v35 = &v124[v126];
    v127 = v35->m128i_i64[0];
    if ( v39 == v35->m128i_i64[0] )
      goto LABEL_67;
    v128 = 1;
    v123 = 0;
    while ( v127 != -8 )
    {
      if ( !v123 && v127 == -16 )
        v123 = v35;
      v126 = v125 & (v128 + v126);
      v35 = &v124[v126];
      v127 = v35->m128i_i64[0];
      if ( v39 == v35->m128i_i64[0] )
        goto LABEL_216;
      ++v128;
    }
LABEL_214:
    if ( v123 )
      v35 = v123;
LABEL_216:
    v41 = v181.m128i_u32[2];
LABEL_67:
    v181.m128i_i32[2] = (2 * (v41 >> 1) + 2) | v41 & 1;
    if ( v35->m128i_i64[0] != -8 )
      --v181.m128i_i32[3];
    --v30;
    v35->m128i_i32[2] = 0;
    v35->m128i_i64[0] = v39;
    v35->m128i_i32[2] = 1;
    if ( v31 != v30 )
      continue;
    break;
  }
LABEL_70:
  v44 = v151;
  v45 = *(unsigned int *)(v1 + 240);
  v145 = v150;
  result = *(__int64 **)(v1 + 232);
  if ( v150 != v151 )
  {
    v47 = 0;
    v143 = (const void *)(v1 + 248);
    v48 = 0;
    while ( 1 )
    {
      v49 = (__int64 *)*(v44 - 1);
      v50 = *(_QWORD *)(v1 + 224);
      v51 = *v49;
      v52 = *(_QWORD *)(v50 + 8);
      v149 = v49;
      j = *(_DWORD *)(v50 + 24);
      if ( (*v49 & 4) != 0 )
      {
        if ( !j )
          goto LABEL_78;
        v54 = v49[4];
        v55 = j - 1;
        v56 = v55 & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
        v57 = (__int64 *)(v52 + 16LL * v56);
        v58 = *v57;
        if ( v54 != *v57 )
        {
          v129 = 1;
          while ( v58 != -8 )
          {
            v130 = v129 + 1;
            v56 = v55 & (v129 + v56);
            v57 = (__int64 *)(v52 + 16LL * v56);
            v58 = *v57;
            if ( v54 == *v57 )
              goto LABEL_75;
            v129 = v130;
          }
LABEL_150:
          j = 0;
          goto LABEL_78;
        }
LABEL_75:
        v59 = (_QWORD **)v57[1];
        if ( !v59 )
          goto LABEL_150;
        v60 = *v59;
        for ( j = 1; v60; ++j )
          v60 = (_QWORD *)*v60;
      }
      else
      {
        if ( !j )
          goto LABEL_78;
        v75 = j - 1;
        v76 = v51 & 0xFFFFFFFFFFFFFFF8LL;
        v77 = v75 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
        v78 = (__int64 *)(v52 + 16LL * v77);
        v79 = *v78;
        if ( v76 != *v78 )
        {
          v101 = 1;
          while ( v79 != -8 )
          {
            v142 = v101 + 1;
            v77 = v75 & (v101 + v77);
            v78 = (__int64 *)(v52 + 16LL * v77);
            v79 = *v78;
            if ( v76 == *v78 )
              goto LABEL_108;
            v101 = v142;
          }
          goto LABEL_150;
        }
LABEL_108:
        v80 = (_QWORD **)v78[1];
        if ( !v80 )
          goto LABEL_150;
        v81 = *v80;
        for ( j = 1; v81; ++j )
          v81 = (_QWORD *)*v81;
      }
LABEL_78:
      --v44;
      v61 = &result[v45];
      v62 = (8 * v45) >> 3;
      if ( (8 * v45) >> 5 )
      {
        v63 = *v44;
        v64 = result;
        while ( 1 )
        {
          if ( *v64 == v63 )
            goto LABEL_85;
          if ( v63 == v64[1] )
          {
            ++v64;
            goto LABEL_85;
          }
          if ( v63 == v64[2] )
          {
            v64 += 2;
            goto LABEL_85;
          }
          if ( v63 == v64[3] )
            break;
          v64 += 4;
          if ( &result[4 * ((8 * v45) >> 5)] == v64 )
          {
            v62 = v61 - v64;
            goto LABEL_118;
          }
        }
        v64 += 3;
        goto LABEL_85;
      }
      v64 = result;
LABEL_118:
      if ( v62 == 2 )
        goto LABEL_153;
      if ( v62 == 3 )
      {
        if ( *v64 == *v44 )
          goto LABEL_85;
        ++v64;
LABEL_153:
        if ( *v64 != *v44 && *++v64 != *v44 )
        {
LABEL_122:
          if ( j >= v47 )
            goto LABEL_87;
          goto LABEL_123;
        }
        goto LABEL_85;
      }
      if ( v62 != 1 )
      {
LABEL_86:
        if ( j >= v47 )
          goto LABEL_87;
LABEL_123:
        v144 = j;
        v83 = (__int64 **)(v44 - 1);
        v84 = v181.m128i_i8[8];
        v148 = ((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4);
        while ( 1 )
        {
          if ( (v84 & 1) != 0 )
          {
            v85 = &v182;
            v86 = 7;
          }
          else
          {
            v94 = v182.m128i_u32[2];
            v85 = (__m128i *)v182.m128i_i64[0];
            if ( !v182.m128i_i32[2] )
            {
              v95 = v181.m128i_u32[2];
              ++v181.m128i_i64[0];
              v96 = 0;
              j = v144;
              v97 = ((unsigned __int32)v181.m128i_i32[2] >> 1) + 1;
              goto LABEL_133;
            }
            v86 = v182.m128i_i32[2] - 1;
          }
          v87 = v86 & v148;
          v88 = &v85[v86 & v148];
          v89 = v88->m128i_i64[0];
          if ( v88->m128i_i64[0] != v48 )
            break;
LABEL_126:
          v90 = v88->m128i_i32[2];
          if ( !v90 )
          {
            j = v144;
            goto LABEL_87;
          }
          if ( v48 == sub_1A6D2B0(v1, *v83) )
          {
            *(_DWORD *)(v93 + 8) = v90 - 1;
            v99 = *(unsigned int *)(v1 + 240);
            if ( (unsigned int)v99 >= *(_DWORD *)(v1 + 244) )
            {
              sub_16CD150(v1 + 232, v143, 0, 8, v91, v92);
              v99 = *(unsigned int *)(v1 + 240);
            }
            *(_QWORD *)(*(_QWORD *)(v1 + 232) + 8 * v99) = *v83;
            v84 = v181.m128i_i8[8];
            ++*(_DWORD *)(v1 + 240);
          }
          --v83;
        }
        v100 = 1;
        v96 = 0;
        while ( v89 != -8 )
        {
          if ( !v96 && v89 == -16 )
            v96 = v88;
          v87 = v86 & (v100 + v87);
          v88 = &v85[v87];
          v89 = v88->m128i_i64[0];
          if ( v88->m128i_i64[0] == v48 )
            goto LABEL_126;
          ++v100;
        }
        v95 = v181.m128i_u32[2];
        j = v144;
        v98 = 24;
        if ( !v96 )
          v96 = v88;
        v94 = 8;
        ++v181.m128i_i64[0];
        v97 = ((unsigned __int32)v181.m128i_i32[2] >> 1) + 1;
        if ( (v84 & 1) == 0 )
        {
          v94 = v182.m128i_u32[2];
LABEL_133:
          v98 = 3 * v94;
        }
        if ( v98 <= 4 * v97 )
        {
          sub_1A6EFA0((__int64)&v181, 2 * v94);
          if ( (v181.m128i_i8[8] & 1) != 0 )
          {
            v102 = &v182;
            v103 = 7;
          }
          else
          {
            v102 = (__m128i *)v182.m128i_i64[0];
            if ( !v182.m128i_i32[2] )
              goto LABEL_292;
            v103 = v182.m128i_i32[2] - 1;
          }
          v95 = v181.m128i_u32[2];
          v104 = v103 & v148;
          v96 = &v102[v103 & v148];
          v105 = v96->m128i_i64[0];
          if ( v48 != v96->m128i_i64[0] )
          {
            v106 = 1;
            v107 = 0;
            while ( v105 != -8 )
            {
              if ( v105 == -16 && !v107 )
                v107 = v96;
              v104 = v103 & (v106 + v104);
              v96 = &v102[v104];
              v105 = v96->m128i_i64[0];
              if ( v48 == v96->m128i_i64[0] )
                goto LABEL_164;
              ++v106;
            }
LABEL_162:
            if ( v107 )
              v96 = v107;
LABEL_164:
            v95 = v181.m128i_u32[2];
          }
        }
        else if ( v94 - v181.m128i_i32[3] - v97 <= v94 >> 3 )
        {
          sub_1A6EFA0((__int64)&v181, v94);
          if ( (v181.m128i_i8[8] & 1) != 0 )
          {
            v112 = &v182;
            v113 = 7;
          }
          else
          {
            v112 = (__m128i *)v182.m128i_i64[0];
            if ( !v182.m128i_i32[2] )
              goto LABEL_292;
            v113 = v182.m128i_i32[2] - 1;
          }
          v95 = v181.m128i_u32[2];
          v114 = v113 & v148;
          v96 = &v112[v113 & v148];
          v115 = v96->m128i_i64[0];
          if ( v48 != v96->m128i_i64[0] )
          {
            v116 = 1;
            v107 = 0;
            while ( v115 != -8 )
            {
              if ( v115 == -16 && !v107 )
                v107 = v96;
              v114 = v113 & (v116 + v114);
              v96 = &v112[v114];
              v115 = v96->m128i_i64[0];
              if ( v48 == v96->m128i_i64[0] )
                goto LABEL_164;
              ++v116;
            }
            goto LABEL_162;
          }
        }
        v181.m128i_i32[2] = (2 * (v95 >> 1) + 2) | v95 & 1;
        if ( v96->m128i_i64[0] != -8 )
          --v181.m128i_i32[3];
        v96->m128i_i64[0] = v48;
        v96->m128i_i32[2] = 0;
LABEL_87:
        v65 = sub_1A6D2B0(v1, v149);
        v48 = v65;
        if ( v65 )
        {
          if ( (v181.m128i_i8[8] & 1) != 0 )
          {
            v68 = &v182;
            v69 = 7;
LABEL_90:
            v70 = v69 & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
            v71 = &v68[v70];
            v66 = (__m128i *)v71->m128i_i64[0];
            if ( v48 == v71->m128i_i64[0] )
            {
              v72 = v71->m128i_i32[2] - 1;
              goto LABEL_92;
            }
            v131 = 1;
            v132 = 0;
            while ( v66 != (__m128i *)-8LL )
            {
              if ( v66 != (__m128i *)-16LL || v132 )
                v71 = v132;
              v70 = v69 & (v131 + v70);
              v67 = &v68[v70];
              v66 = (__m128i *)v67->m128i_i64[0];
              if ( v48 == v67->m128i_i64[0] )
              {
                v72 = v67->m128i_i32[2] - 1;
                v71 = v67;
                goto LABEL_92;
              }
              ++v131;
              v132 = v71;
              v71 = &v68[v70];
            }
            v108 = v181.m128i_u32[2];
            LODWORD(v66) = 24;
            v82 = 8;
            if ( v132 )
              v71 = v132;
            ++v181.m128i_i64[0];
            v109 = ((unsigned __int32)v181.m128i_i32[2] >> 1) + 1;
            if ( (v181.m128i_i8[8] & 1) == 0 )
            {
              v82 = v182.m128i_u32[2];
              goto LABEL_166;
            }
          }
          else
          {
            v82 = v182.m128i_u32[2];
            v68 = (__m128i *)v182.m128i_i64[0];
            if ( v182.m128i_i32[2] )
            {
              v69 = v182.m128i_i32[2] - 1;
              goto LABEL_90;
            }
            v108 = v181.m128i_u32[2];
            ++v181.m128i_i64[0];
            v71 = 0;
            v109 = ((unsigned __int32)v181.m128i_i32[2] >> 1) + 1;
LABEL_166:
            LODWORD(v66) = 3 * v82;
          }
          if ( 4 * v109 >= (unsigned int)v66 )
          {
            sub_1A6EFA0((__int64)&v181, 2 * v82);
            if ( (v181.m128i_i8[8] & 1) != 0 )
            {
              v67 = &v182;
              v133 = 7;
            }
            else
            {
              v67 = (__m128i *)v182.m128i_i64[0];
              if ( !v182.m128i_i32[2] )
                goto LABEL_292;
              v133 = v182.m128i_i32[2] - 1;
            }
            v108 = v181.m128i_u32[2];
            v134 = v133 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
            v71 = &v67[v134];
            v66 = (__m128i *)v71->m128i_i64[0];
            if ( v48 != v71->m128i_i64[0] )
            {
              v135 = 1;
              v136 = 0;
              while ( v66 != (__m128i *)-8LL )
              {
                if ( v66 == (__m128i *)-16LL && !v136 )
                  v136 = v71;
                v134 = v133 & (v135 + v134);
                v71 = &v67[v134];
                v66 = (__m128i *)v71->m128i_i64[0];
                if ( v48 == v71->m128i_i64[0] )
                  goto LABEL_247;
                ++v135;
              }
LABEL_245:
              if ( v136 )
                v71 = v136;
LABEL_247:
              v108 = v181.m128i_u32[2];
            }
          }
          else if ( v82 - v181.m128i_i32[3] - v109 <= v82 >> 3 )
          {
            sub_1A6EFA0((__int64)&v181, v82);
            if ( (v181.m128i_i8[8] & 1) != 0 )
            {
              v66 = &v182;
              v137 = 7;
            }
            else
            {
              v66 = (__m128i *)v182.m128i_i64[0];
              if ( !v182.m128i_i32[2] )
                goto LABEL_292;
              v137 = v182.m128i_i32[2] - 1;
            }
            v108 = v181.m128i_u32[2];
            v138 = v137 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
            v71 = &v66[v138];
            v139 = (__m128i *)v71->m128i_i64[0];
            if ( v48 != v71->m128i_i64[0] )
            {
              v140 = 1;
              v136 = 0;
              while ( v139 != (__m128i *)-8LL )
              {
                if ( !v136 && v139 == (__m128i *)-16LL )
                  v136 = v71;
                LODWORD(v67) = v140 + 1;
                v138 = v137 & (v140 + v138);
                v71 = &v66[v138];
                v139 = (__m128i *)v71->m128i_i64[0];
                if ( v48 == v71->m128i_i64[0] )
                  goto LABEL_247;
                ++v140;
              }
              goto LABEL_245;
            }
          }
          v181.m128i_i32[2] = (2 * (v108 >> 1) + 2) | v108 & 1;
          if ( v71->m128i_i64[0] != -8 )
            --v181.m128i_i32[3];
          v71->m128i_i64[0] = v48;
          v72 = -1;
          v71->m128i_i32[2] = 0;
LABEL_92:
          v71->m128i_i32[2] = v72;
        }
        v73 = *(unsigned int *)(v1 + 240);
        if ( (unsigned int)v73 >= *(_DWORD *)(v1 + 244) )
        {
          sub_16CD150(v1 + 232, v143, 0, 8, (int)v66, (int)v67);
          v73 = *(unsigned int *)(v1 + 240);
        }
        v47 = j;
        *(_QWORD *)(*(_QWORD *)(v1 + 232) + 8 * v73) = *v44;
        v45 = (unsigned int)(*(_DWORD *)(v1 + 240) + 1);
        result = *(__int64 **)(v1 + 232);
        *(_DWORD *)(v1 + 240) = v45;
        v61 = &result[v45];
        goto LABEL_96;
      }
      if ( *v64 != *v44 )
        goto LABEL_122;
LABEL_85:
      if ( v61 == v64 )
        goto LABEL_86;
LABEL_96:
      if ( v145 == v44 )
        goto LABEL_97;
    }
  }
LABEL_249:
  v61 = &result[v45];
LABEL_97:
  if ( result != v61 )
  {
    while ( result < --v61 )
    {
      v74 = *result++;
      *(result - 1) = *v61;
      *v61 = v74;
    }
  }
  if ( (v181.m128i_i8[8] & 1) == 0 )
    result = (__int64 *)j___libc_free_0(v182.m128i_i64[0]);
  if ( v150 )
    return (__int64 *)j_j___libc_free_0(v150, v152 - (_QWORD)v150);
  return result;
}
