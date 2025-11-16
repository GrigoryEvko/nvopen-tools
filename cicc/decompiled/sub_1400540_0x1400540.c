// Function: sub_1400540
// Address: 0x1400540
//
void __fastcall sub_1400540(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __m128i *v4; // rax
  const __m128i *v5; // rax
  const __m128i *v6; // rax
  __m128i *v7; // rax
  const __m128i *v8; // rax
  const __m128i *v9; // rax
  __m128i *v10; // rax
  const __m128i *v11; // rax
  __m128i *v12; // rax
  __int8 *v13; // rax
  __int64 *v14; // rdi
  _BYTE *v15; // rsi
  const __m128i *v16; // rcx
  const __m128i *v17; // rdx
  unsigned __int64 v18; // rbx
  __m128i *v19; // rax
  __m128i *v20; // rcx
  const __m128i *v21; // rax
  const __m128i *v22; // rcx
  unsigned __int64 v23; // rbx
  __int64 v24; // rax
  __m128i *v25; // rdi
  __m128i *v26; // rdx
  __m128i *v27; // rax
  __m128i *v28; // rsi
  const __m128i *v29; // rdx
  __int64 v30; // r14
  __int64 v31; // rbx
  __int64 v32; // r12
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 v35; // r8
  _QWORD **v36; // rax
  _QWORD *v37; // rbx
  _QWORD *v38; // rax
  char v39; // dl
  void *v40; // rax
  __int64 v41; // rax
  __int64 v42; // rsi
  unsigned int v43; // ecx
  __int64 *v44; // rdx
  __int64 v45; // r8
  __int64 v46; // rax
  __int64 v47; // r13
  __int64 v48; // rdx
  __int64 v49; // rcx
  unsigned __int64 v50; // r12
  __int64 v51; // rax
  unsigned int v52; // edx
  __int64 v53; // r13
  __int64 v54; // rsi
  _QWORD *v55; // rax
  const void *v56; // rbx
  size_t v57; // r12
  char *v58; // rax
  char *v59; // r13
  char *v60; // rax
  unsigned int v61; // esi
  __int64 v62; // rbx
  __int64 v63; // rcx
  unsigned int v64; // edx
  __int64 *v65; // rax
  __int64 v66; // r9
  unsigned __int64 *v67; // rax
  unsigned __int64 *v68; // rbx
  unsigned __int64 v69; // r12
  const void *v70; // r8
  __int64 v71; // r13
  char *v72; // rbx
  signed __int64 v73; // rdx
  _QWORD *v74; // rcx
  _QWORD *v75; // rsi
  _QWORD *v76; // rdx
  _QWORD *v77; // rdx
  int v78; // eax
  __int64 v79; // rax
  __int64 v80; // rdi
  unsigned int v81; // r8d
  __int64 *v82; // rdx
  __int64 v83; // r9
  unsigned int v84; // r12d
  __int64 *v85; // rax
  __int64 v86; // rdx
  __int64 v87; // r14
  __int64 v88; // r12
  __int64 v89; // rbx
  unsigned __int64 v90; // rbx
  __int64 v91; // rax
  unsigned __int64 v92; // rdx
  __int64 v93; // r12
  __int64 v94; // rax
  int v95; // esi
  __int64 v96; // rdi
  unsigned int v97; // ecx
  __int64 *v98; // rax
  __int64 *v99; // r8
  __int64 *v100; // rdx
  int v101; // eax
  __int64 *v102; // rax
  int v103; // eax
  int v104; // r9d
  int v105; // edx
  int v106; // r9d
  _QWORD *v107; // rdi
  unsigned int v108; // r8d
  _QWORD *v109; // rcx
  size_t v110; // rdx
  unsigned __int64 v111; // rax
  bool v112; // cf
  unsigned __int64 v113; // rax
  size_t v114; // r12
  __int64 v115; // rbx
  char *v116; // rbx
  __int64 v117; // rax
  char *v118; // rbx
  __int64 v119; // rsi
  int v120; // edx
  int v121; // r11d
  int v122; // r10d
  __int64 *v123; // r8
  int v124; // edi
  int v125; // edx
  __int64 v126; // rdi
  int v127; // r8d
  size_t v128; // [rsp+8h] [rbp-3D8h]
  __int64 *v129; // [rsp+10h] [rbp-3D0h]
  char *v130; // [rsp+20h] [rbp-3C0h]
  char *v131; // [rsp+28h] [rbp-3B8h]
  char *v132; // [rsp+38h] [rbp-3A8h]
  __int64 v133; // [rsp+40h] [rbp-3A0h]
  unsigned int v135; // [rsp+6Ch] [rbp-374h]
  __int64 v136; // [rsp+70h] [rbp-370h]
  const void *v137; // [rsp+70h] [rbp-370h]
  unsigned int v138; // [rsp+78h] [rbp-368h]
  unsigned __int64 v139; // [rsp+88h] [rbp-358h]
  unsigned int dest; // [rsp+98h] [rbp-348h]
  char *desta; // [rsp+98h] [rbp-348h]
  __int64 v142; // [rsp+A0h] [rbp-340h] BYREF
  __int64 *v143; // [rsp+A8h] [rbp-338h] BYREF
  _QWORD v144[16]; // [rsp+B0h] [rbp-330h] BYREF
  void *src[2]; // [rsp+130h] [rbp-2B0h] BYREF
  unsigned __int64 v146[2]; // [rsp+140h] [rbp-2A0h] BYREF
  int v147; // [rsp+150h] [rbp-290h]
  _QWORD v148[8]; // [rsp+158h] [rbp-288h] BYREF
  const __m128i *v149; // [rsp+198h] [rbp-248h] BYREF
  const __m128i *v150; // [rsp+1A0h] [rbp-240h]
  __m128i *v151; // [rsp+1A8h] [rbp-238h]
  __int64 v152; // [rsp+1B0h] [rbp-230h] BYREF
  _QWORD *v153; // [rsp+1B8h] [rbp-228h]
  _QWORD *v154; // [rsp+1C0h] [rbp-220h]
  unsigned int v155; // [rsp+1C8h] [rbp-218h]
  unsigned int v156; // [rsp+1CCh] [rbp-214h]
  int v157; // [rsp+1D0h] [rbp-210h]
  _BYTE v158[64]; // [rsp+1D8h] [rbp-208h] BYREF
  const __m128i *v159; // [rsp+218h] [rbp-1C8h] BYREF
  const __m128i *v160; // [rsp+220h] [rbp-1C0h]
  __m128i *v161; // [rsp+228h] [rbp-1B8h]
  char v162[8]; // [rsp+230h] [rbp-1B0h] BYREF
  __int64 v163; // [rsp+238h] [rbp-1A8h]
  unsigned __int64 v164; // [rsp+240h] [rbp-1A0h]
  _BYTE v165[64]; // [rsp+258h] [rbp-188h] BYREF
  __m128i *v166; // [rsp+298h] [rbp-148h]
  __m128i *v167; // [rsp+2A0h] [rbp-140h]
  __int8 *v168; // [rsp+2A8h] [rbp-138h]
  __m128i v169; // [rsp+2B0h] [rbp-130h] BYREF
  unsigned __int64 v170; // [rsp+2C0h] [rbp-120h]
  char v171[64]; // [rsp+2D8h] [rbp-108h] BYREF
  const __m128i *v172; // [rsp+318h] [rbp-C8h]
  const __m128i *v173; // [rsp+320h] [rbp-C0h]
  __m128i *v174; // [rsp+328h] [rbp-B8h]
  char v175[8]; // [rsp+330h] [rbp-B0h] BYREF
  __int64 v176; // [rsp+338h] [rbp-A8h]
  unsigned __int64 v177; // [rsp+340h] [rbp-A0h]
  char v178[64]; // [rsp+358h] [rbp-88h] BYREF
  const __m128i *v179; // [rsp+398h] [rbp-48h]
  const __m128i *v180; // [rsp+3A0h] [rbp-40h]
  __int8 *v181; // [rsp+3A8h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 56);
  memset(v144, 0, sizeof(v144));
  LODWORD(v144[3]) = 8;
  v144[1] = &v144[5];
  v144[2] = &v144[5];
  v146[1] = 0x100000008LL;
  v148[0] = v3;
  src[1] = v148;
  v146[0] = (unsigned __int64)v148;
  v149 = 0;
  v150 = 0;
  v151 = 0;
  v147 = 0;
  src[0] = (void *)1;
  v129 = (__int64 *)v3;
  v169.m128i_i64[1] = *(_QWORD *)(v3 + 24);
  v169.m128i_i64[0] = v3;
  sub_13FF6D0(&v149, 0, &v169);
  sub_13FF850((__int64)src);
  sub_16CCEE0(v162, v165, 8, v144);
  v4 = (__m128i *)v144[13];
  memset(&v144[13], 0, 24);
  v166 = v4;
  v167 = (__m128i *)v144[14];
  v168 = (__int8 *)v144[15];
  sub_16CCEE0(&v152, v158, 8, src);
  v5 = v149;
  v149 = 0;
  v159 = v5;
  v6 = v150;
  v150 = 0;
  v160 = v6;
  v7 = v151;
  v151 = 0;
  v161 = v7;
  sub_16CCEE0(&v169, v171, 8, &v152);
  v8 = v159;
  v159 = 0;
  v172 = v8;
  v9 = v160;
  v160 = 0;
  v173 = v9;
  v10 = v161;
  v161 = 0;
  v174 = v10;
  sub_16CCEE0(v175, v178, 8, v162);
  v11 = v166;
  v166 = 0;
  v179 = v11;
  v12 = v167;
  v167 = 0;
  v180 = v12;
  v13 = v168;
  v168 = 0;
  v181 = v13;
  if ( v159 )
    j_j___libc_free_0(v159, (char *)v161 - (char *)v159);
  if ( v154 != v153 )
    _libc_free((unsigned __int64)v154);
  if ( v166 )
    j_j___libc_free_0(v166, v168 - (__int8 *)v166);
  if ( v164 != v163 )
    _libc_free(v164);
  if ( v149 )
    j_j___libc_free_0(v149, (char *)v151 - (char *)v149);
  if ( (void *)v146[0] != src[1] )
    _libc_free(v146[0]);
  if ( v144[13] )
    j_j___libc_free_0(v144[13], v144[15] - v144[13]);
  if ( v144[2] != v144[1] )
    _libc_free(v144[2]);
  v14 = &v152;
  v15 = v158;
  sub_16CCCB0(&v152, v158, &v169);
  v16 = v173;
  v17 = v172;
  v159 = 0;
  v160 = 0;
  v161 = 0;
  v18 = (char *)v173 - (char *)v172;
  if ( v173 == v172 )
  {
    v18 = 0;
    v19 = 0;
  }
  else
  {
    if ( v18 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_235;
    v19 = (__m128i *)sub_22077B0((char *)v173 - (char *)v172);
    v16 = v173;
    v17 = v172;
  }
  v159 = v19;
  v160 = v19;
  v161 = (__m128i *)((char *)v19 + v18);
  if ( v17 == v16 )
  {
    v20 = v19;
  }
  else
  {
    v20 = (__m128i *)((char *)v19 + (char *)v16 - (char *)v17);
    do
    {
      if ( v19 )
        *v19 = _mm_loadu_si128(v17);
      ++v19;
      ++v17;
    }
    while ( v19 != v20 );
  }
  v15 = v165;
  v14 = (__int64 *)v162;
  v160 = v20;
  sub_16CCCB0(v162, v165, v175);
  v21 = v180;
  v22 = v179;
  v166 = 0;
  v167 = 0;
  v168 = 0;
  v23 = (char *)v180 - (char *)v179;
  if ( v180 != v179 )
  {
    if ( v23 <= 0x7FFFFFFFFFFFFFF0LL )
    {
      v24 = sub_22077B0((char *)v180 - (char *)v179);
      v22 = v179;
      v25 = (__m128i *)v24;
      v21 = v180;
      goto LABEL_28;
    }
LABEL_235:
    sub_4261EA(v14, v15, v17);
  }
  v23 = 0;
  v25 = 0;
LABEL_28:
  v166 = v25;
  v167 = v25;
  v168 = &v25->m128i_i8[v23];
  if ( v22 == v21 )
  {
    v27 = v25;
  }
  else
  {
    v26 = v25;
    v27 = (__m128i *)((char *)v25 + (char *)v21 - (char *)v22);
    do
    {
      if ( v26 )
        *v26 = _mm_loadu_si128(v22);
      ++v26;
      ++v22;
    }
    while ( v26 != v27 );
  }
  v167 = v27;
  v28 = (__m128i *)v160;
LABEL_34:
  v29 = v159;
  v30 = a2;
  if ( (char *)v28 - (char *)v159 == (char *)v27 - (char *)v25 )
    goto LABEL_142;
  do
  {
LABEL_35:
    v31 = *(_QWORD *)v28[-1].m128i_i64[0];
    src[0] = v146;
    src[1] = (void *)0x400000000LL;
    v32 = *(_QWORD *)(v31 + 8);
    if ( !v32 )
      goto LABEL_40;
    while ( 1 )
    {
      v33 = sub_1648700(v32);
      if ( (unsigned __int8)(*(_BYTE *)(v33 + 16) - 25) <= 9u )
        break;
      v32 = *(_QWORD *)(v32 + 8);
      if ( !v32 )
        goto LABEL_38;
    }
    while ( 1 )
    {
      v47 = *(_QWORD *)(v33 + 40);
      if ( (unsigned __int8)sub_15CC8F0(v30, v31, v47, v34, v35) )
      {
        v41 = *(unsigned int *)(v30 + 48);
        if ( (_DWORD)v41 )
        {
          v42 = *(_QWORD *)(v30 + 32);
          v43 = (v41 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
          v44 = (__int64 *)(v42 + 16LL * v43);
          v45 = *v44;
          if ( v47 == *v44 )
          {
LABEL_51:
            if ( v44 != (__int64 *)(v42 + 16 * v41) && v44[1] )
            {
              v46 = LODWORD(src[1]);
              if ( LODWORD(src[1]) >= HIDWORD(src[1]) )
              {
                sub_16CD150(src, v146, 0, 8);
                v46 = LODWORD(src[1]);
              }
              *((_QWORD *)src[0] + v46) = v47;
              ++LODWORD(src[1]);
            }
          }
          else
          {
            v105 = 1;
            while ( v45 != -8 )
            {
              v106 = v105 + 1;
              v43 = (v41 - 1) & (v43 + v105);
              v44 = (__int64 *)(v42 + 16LL * v43);
              v45 = *v44;
              if ( v47 == *v44 )
                goto LABEL_51;
              v105 = v106;
            }
          }
        }
        goto LABEL_56;
      }
      v32 = *(_QWORD *)(v32 + 8);
      if ( !v32 )
        break;
      while ( 1 )
      {
        v33 = sub_1648700(v32);
        v34 = *(unsigned __int8 *)(v33 + 16);
        if ( (unsigned __int8)(v34 - 25) <= 9u )
          break;
LABEL_56:
        v32 = *(_QWORD *)(v32 + 8);
        if ( !v32 )
          goto LABEL_60;
      }
    }
LABEL_60:
    if ( !LODWORD(src[1]) )
      goto LABEL_38;
    v48 = *(_QWORD *)(a1 + 56);
    v49 = *(_QWORD *)(a1 + 64);
    *(_QWORD *)(a1 + 136) += 168LL;
    if ( ((v48 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v48 + 168 <= v49 - v48 )
    {
      v139 = (v48 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(a1 + 56) = v139 + 168;
    }
    else
    {
      v50 = 0x40000000000LL;
      dest = *(_DWORD *)(a1 + 80);
      if ( dest >> 7 < 0x1E )
        v50 = 4096LL << (dest >> 7);
      v51 = malloc(v50);
      v52 = dest;
      v53 = v51;
      if ( !v51 )
      {
        sub_16BD1C0("Allocation failed");
        v52 = *(_DWORD *)(a1 + 80);
      }
      if ( *(_DWORD *)(a1 + 84) <= v52 )
      {
        sub_16CD150(a1 + 72, a1 + 88, 0, 8);
        v52 = *(_DWORD *)(a1 + 80);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL * v52) = v53;
      ++*(_DWORD *)(a1 + 80);
      *(_QWORD *)(a1 + 64) = v53 + v50;
      v139 = (v53 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(a1 + 56) = v139 + 168;
    }
    v144[0] = v31;
    *(_QWORD *)v139 = 0;
    *(_QWORD *)(v139 + 64) = v139 + 96;
    *(_QWORD *)(v139 + 72) = v139 + 96;
    *(_QWORD *)(v139 + 8) = 0;
    *(_QWORD *)(v139 + 16) = 0;
    *(_QWORD *)(v139 + 24) = 0;
    *(_QWORD *)(v139 + 32) = 0;
    *(_QWORD *)(v139 + 40) = 0;
    *(_QWORD *)(v139 + 48) = 0;
    *(_QWORD *)(v139 + 56) = 0;
    *(_QWORD *)(v139 + 80) = 8;
    *(_DWORD *)(v139 + 88) = 0;
    *(_BYTE *)(v139 + 160) = 0;
    sub_1292090(v139 + 32, 0, v144);
    v54 = v144[0];
    v55 = *(_QWORD **)(v139 + 64);
    if ( *(_QWORD **)(v139 + 72) == v55 )
    {
      v107 = &v55[*(unsigned int *)(v139 + 84)];
      v108 = *(_DWORD *)(v139 + 84);
      if ( v55 == v107 )
      {
LABEL_229:
        if ( v108 >= *(_DWORD *)(v139 + 80) )
          goto LABEL_70;
        *(_DWORD *)(v139 + 84) = v108 + 1;
        *v107 = v54;
        ++*(_QWORD *)(v139 + 56);
      }
      else
      {
        v109 = 0;
        while ( v144[0] != *v55 )
        {
          if ( *v55 == -2 )
            v109 = v55;
          if ( v107 == ++v55 )
          {
            if ( !v109 )
              goto LABEL_229;
            *v109 = v144[0];
            --*(_DWORD *)(v139 + 88);
            ++*(_QWORD *)(v139 + 56);
            break;
          }
        }
      }
    }
    else
    {
LABEL_70:
      sub_16CCBA0(v139 + 56, v144[0]);
    }
    memset(v144, 0, 24);
    v56 = src[0];
    v57 = 8LL * LODWORD(src[1]);
    if ( !v57
      || (v58 = (char *)sub_22077B0(8LL * LODWORD(src[1])),
          v59 = &v58[v57],
          v144[0] = v58,
          v144[2] = &v58[v57],
          v60 = (char *)memcpy(v58, v56, v57),
          v144[1] = v59,
          v60 == v59) )
    {
      v69 = 0;
      goto LABEL_87;
    }
    v136 = v30;
    v135 = 0;
    v138 = 0;
    while ( 2 )
    {
      v61 = *(_DWORD *)(a1 + 24);
      v62 = *((_QWORD *)v59 - 1);
      v59 -= 8;
      v144[1] = v59;
      v63 = *(_QWORD *)(a1 + 8);
      if ( v61 )
      {
        v64 = (v61 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
        v65 = (__int64 *)(v63 + 16LL * v64);
        v66 = *v65;
        if ( v62 == *v65 )
        {
LABEL_76:
          v67 = (unsigned __int64 *)v65[1];
          if ( v67 )
          {
            do
            {
              v68 = v67;
              v67 = (unsigned __int64 *)*v67;
            }
            while ( v67 );
            if ( v68 != (unsigned __int64 *)v139 )
            {
              v92 = v68[4];
              ++v135;
              *v68 = v139;
              v138 += (__int64)(v68[6] - v92) >> 3;
              v93 = *(_QWORD *)(*(_QWORD *)v92 + 8LL);
              if ( v93 )
              {
                while ( 1 )
                {
                  v94 = sub_1648700(v93);
                  if ( (unsigned __int8)(*(_BYTE *)(v94 + 16) - 25) <= 9u )
                    break;
                  v93 = *(_QWORD *)(v93 + 8);
                  if ( !v93 )
                    goto LABEL_137;
                }
                v59 = (char *)v144[1];
LABEL_131:
                v100 = *(__int64 **)(v94 + 40);
                v101 = *(_DWORD *)(a1 + 24);
                v143 = v100;
                if ( v101 )
                {
                  v95 = v101 - 1;
                  v96 = *(_QWORD *)(a1 + 8);
                  v97 = (v101 - 1) & (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4));
                  v98 = (__int64 *)(v96 + 16LL * v97);
                  v99 = (__int64 *)*v98;
                  if ( v100 == (__int64 *)*v98 )
                  {
LABEL_128:
                    if ( v68 == (unsigned __int64 *)v98[1] )
                      goto LABEL_129;
                  }
                  else
                  {
                    v103 = 1;
                    while ( v99 != (__int64 *)-8LL )
                    {
                      v104 = v103 + 1;
                      v97 = v95 & (v103 + v97);
                      v98 = (__int64 *)(v96 + 16LL * v97);
                      v99 = (__int64 *)*v98;
                      if ( v100 == (__int64 *)*v98 )
                        goto LABEL_128;
                      v103 = v104;
                    }
                  }
                }
                if ( (char *)v144[2] == v59 )
                {
                  sub_1292090((__int64)v144, v59, &v143);
                  v59 = (char *)v144[1];
                }
                else
                {
                  if ( v59 )
                  {
                    *(_QWORD *)v59 = v100;
                    v59 = (char *)v144[1];
                  }
                  v59 += 8;
                  v144[1] = v59;
                }
LABEL_129:
                while ( 1 )
                {
                  v93 = *(_QWORD *)(v93 + 8);
                  if ( !v93 )
                    break;
                  v94 = sub_1648700(v93);
                  if ( (unsigned __int8)(*(_BYTE *)(v94 + 16) - 25) <= 9u )
                    goto LABEL_131;
                }
              }
              else
              {
LABEL_137:
                v59 = (char *)v144[1];
              }
            }
            goto LABEL_79;
          }
        }
        else
        {
          v78 = 1;
          while ( v66 != -8 )
          {
            v127 = v78 + 1;
            v64 = (v61 - 1) & (v78 + v64);
            v65 = (__int64 *)(v63 + 16LL * v64);
            v66 = *v65;
            if ( v62 == *v65 )
              goto LABEL_76;
            v78 = v127;
          }
        }
      }
      v79 = *(unsigned int *)(v136 + 48);
      if ( !(_DWORD)v79 )
        goto LABEL_79;
      v80 = *(_QWORD *)(v136 + 32);
      v81 = (v79 - 1) & (((unsigned int)v62 >> 4) ^ ((unsigned int)v62 >> 9));
      v82 = (__int64 *)(v80 + 16LL * v81);
      v83 = *v82;
      if ( v62 == *v82 )
      {
LABEL_103:
        if ( v82 == (__int64 *)(v80 + 16 * v79) || !v82[1] )
          goto LABEL_79;
        v142 = v62;
        if ( !v61 )
        {
          ++*(_QWORD *)a1;
LABEL_224:
          v61 *= 2;
LABEL_225:
          sub_1400170(a1, v61);
          sub_13FD8B0(a1, &v142, &v143);
          v85 = v143;
          v126 = v142;
          v125 = *(_DWORD *)(a1 + 16) + 1;
          goto LABEL_220;
        }
        v84 = (v61 - 1) & (((unsigned int)v62 >> 4) ^ ((unsigned int)v62 >> 9));
        v85 = (__int64 *)(v63 + 16LL * v84);
        v86 = *v85;
        if ( v62 == *v85 )
          goto LABEL_107;
        v122 = 1;
        v123 = 0;
        while ( v86 != -8 )
        {
          if ( v86 == -16 && !v123 )
            v123 = v85;
          v84 = (v61 - 1) & (v122 + v84);
          v85 = (__int64 *)(v63 + 16LL * v84);
          v86 = *v85;
          if ( v62 == *v85 )
            goto LABEL_107;
          ++v122;
        }
        v124 = *(_DWORD *)(a1 + 16);
        if ( v123 )
          v85 = v123;
        ++*(_QWORD *)a1;
        v125 = v124 + 1;
        if ( 4 * (v124 + 1) >= 3 * v61 )
          goto LABEL_224;
        v126 = v62;
        if ( v61 - *(_DWORD *)(a1 + 20) - v125 <= v61 >> 3 )
          goto LABEL_225;
LABEL_220:
        *(_DWORD *)(a1 + 16) = v125;
        if ( *v85 != -8 )
          --*(_DWORD *)(a1 + 20);
        *v85 = v126;
        v85[1] = 0;
LABEL_107:
        ++v138;
        v85[1] = v139;
        v59 = (char *)v144[1];
        desta = (char *)v144[0];
        if ( v62 != **(_QWORD **)(v139 + 32) )
        {
          v87 = *(_QWORD *)(v62 + 8);
          v59 = (char *)v144[1];
          if ( v87 )
          {
            while ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v87) + 16) - 25) > 9u )
            {
              v87 = *(_QWORD *)(v87 + 8);
              if ( !v87 )
                goto LABEL_80;
            }
            v88 = v87;
            v89 = 0;
            while ( 1 )
            {
              v88 = *(_QWORD *)(v88 + 8);
              if ( !v88 )
                break;
              while ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v88) + 16) - 25) <= 9u )
              {
                v88 = *(_QWORD *)(v88 + 8);
                ++v89;
                if ( !v88 )
                  goto LABEL_114;
              }
            }
LABEL_114:
            v90 = v89 + 1;
            v133 = v144[2];
            if ( v90 <= (__int64)(v144[2] - (_QWORD)v59) >> 3 )
            {
              v91 = sub_1648700(v87);
LABEL_118:
              if ( v59 )
                *(_QWORD *)v59 = *(_QWORD *)(v91 + 40);
              while ( 1 )
              {
                v87 = *(_QWORD *)(v87 + 8);
                if ( !v87 )
                  break;
                v91 = sub_1648700(v87);
                if ( (unsigned __int8)(*(_BYTE *)(v91 + 16) - 25) <= 9u )
                {
                  v59 += 8;
                  goto LABEL_118;
                }
              }
              v59 = (char *)(v144[1] + 8 * v90);
              v144[1] = v59;
              desta = (char *)v144[0];
              goto LABEL_80;
            }
            v110 = v59 - desta;
            v111 = (v59 - desta) >> 3;
            if ( v90 > 0xFFFFFFFFFFFFFFFLL - v111 )
              sub_4262D8((__int64)"vector::_M_range_insert");
            if ( v90 < v111 )
              v90 = (v59 - desta) >> 3;
            v112 = __CFADD__(v90, v111);
            v113 = v90 + v111;
            v114 = v113;
            if ( v112 )
            {
              v115 = 0x7FFFFFFFFFFFFFF8LL;
              goto LABEL_190;
            }
            if ( v113 )
            {
              if ( v113 > 0xFFFFFFFFFFFFFFFLL )
                v113 = 0xFFFFFFFFFFFFFFFLL;
              v115 = 8 * v113;
LABEL_190:
              desta = (char *)sub_22077B0(v115);
              v130 = (char *)v144[1];
              v110 = (size_t)&v59[-v144[0]];
              v114 = v144[1] - (_QWORD)v59;
              v132 = (char *)v144[0];
              v131 = &desta[v115];
              v133 = v144[2];
            }
            else
            {
              v130 = v59;
              v131 = 0;
              v132 = desta;
              desta = 0;
            }
            if ( v132 != v59 )
            {
              v128 = v110;
              memmove(desta, v132, v110);
              v110 = v128;
            }
            v116 = &desta[v110];
            v117 = sub_1648700(v87);
LABEL_196:
            if ( v116 )
              *(_QWORD *)v116 = *(_QWORD *)(v117 + 40);
            while ( 1 )
            {
              v87 = *(_QWORD *)(v87 + 8);
              if ( !v87 )
                break;
              v117 = sub_1648700(v87);
              if ( (unsigned __int8)(*(_BYTE *)(v117 + 16) - 25) <= 9u )
              {
                v116 += 8;
                goto LABEL_196;
              }
            }
            v118 = v116 + 8;
            if ( v130 != v59 )
              memmove(v118, v59, v114);
            v59 = &v118[v114];
            if ( v132 )
              j_j___libc_free_0(v132, v133 - (_QWORD)v132);
            v144[1] = &v118[v114];
            v144[0] = desta;
            v144[2] = v131;
          }
        }
      }
      else
      {
        v120 = 1;
        while ( v83 != -8 )
        {
          v121 = v120 + 1;
          v81 = (v79 - 1) & (v120 + v81);
          v82 = (__int64 *)(v80 + 16LL * v81);
          v83 = *v82;
          if ( v62 == *v82 )
            goto LABEL_103;
          v120 = v121;
        }
LABEL_79:
        desta = (char *)v144[0];
      }
LABEL_80:
      if ( desta != v59 )
        continue;
      break;
    }
    v30 = v136;
    v69 = v138;
    v70 = *(const void **)(v139 + 8);
    if ( v135 > (unsigned __int64)((__int64)(*(_QWORD *)(v139 + 24) - (_QWORD)v70) >> 3) )
    {
      v71 = *(_QWORD *)(v139 + 16) - (_QWORD)v70;
      if ( v135 )
      {
        v72 = (char *)sub_22077B0(8LL * v135);
        v70 = *(const void **)(v139 + 8);
        v73 = *(_QWORD *)(v139 + 16) - (_QWORD)v70;
      }
      else
      {
        v73 = *(_QWORD *)(v139 + 16) - (_QWORD)v70;
        v72 = 0;
      }
      if ( v73 > 0 )
      {
        v137 = v70;
        memmove(v72, v70, v73);
        v70 = v137;
        v119 = *(_QWORD *)(v139 + 24) - (_QWORD)v137;
      }
      else
      {
        if ( !v70 )
          goto LABEL_86;
        v119 = *(_QWORD *)(v139 + 24) - (_QWORD)v70;
      }
      j_j___libc_free_0(v70, v119);
LABEL_86:
      *(_QWORD *)(v139 + 8) = v72;
      *(_QWORD *)(v139 + 16) = &v72[v71];
      *(_QWORD *)(v139 + 24) = &v72[8 * v135];
    }
LABEL_87:
    sub_13FC0C0(v139 + 32, v69);
    if ( v144[0] )
      j_j___libc_free_0(v144[0], v144[2] - v144[0]);
LABEL_38:
    if ( src[0] != v146 )
      _libc_free((unsigned __int64)src[0]);
LABEL_40:
    --v160;
    v29 = v159;
    v28 = (__m128i *)v160;
    if ( v160 != v159 )
    {
LABEL_41:
      while ( 1 )
      {
        v36 = (_QWORD **)v28[-1].m128i_i64[1];
        if ( *(_QWORD ***)(v28[-1].m128i_i64[0] + 32) == v36 )
          break;
        while ( 1 )
        {
          v28[-1].m128i_i64[1] = (__int64)(v36 + 1);
          v37 = *v36;
          v38 = v153;
          if ( v154 != v153 )
            goto LABEL_43;
          v74 = &v153[v156];
          if ( v153 != v74 )
          {
            v75 = 0;
            while ( 2 )
            {
              v76 = (_QWORD *)*v38;
              if ( v37 == (_QWORD *)*v38 )
              {
LABEL_98:
                v28 = (__m128i *)v160;
                goto LABEL_41;
              }
              while ( v76 == (_QWORD *)-2LL )
              {
                v77 = v38 + 1;
                v75 = v38;
                if ( v74 == v38 + 1 )
                  goto LABEL_95;
                ++v38;
                v76 = (_QWORD *)*v77;
                if ( v37 == v76 )
                  goto LABEL_98;
              }
              if ( v74 != ++v38 )
                continue;
              break;
            }
            if ( v75 )
            {
LABEL_95:
              *v75 = v37;
              v28 = (__m128i *)v160;
              --v157;
              ++v152;
              goto LABEL_44;
            }
          }
          if ( v156 < v155 )
          {
            ++v156;
            *v74 = v37;
            v28 = (__m128i *)v160;
            ++v152;
          }
          else
          {
LABEL_43:
            sub_16CCBA0(&v152, v37);
            v28 = (__m128i *)v160;
            if ( !v39 )
              goto LABEL_41;
          }
LABEL_44:
          v40 = (void *)v37[3];
          src[0] = v37;
          src[1] = v40;
          if ( v161 == v28 )
            break;
          if ( v28 )
          {
            *v28 = _mm_loadu_si128((const __m128i *)src);
            v28 = (__m128i *)v160;
          }
          v160 = ++v28;
          v36 = (_QWORD **)v28[-1].m128i_i64[1];
          if ( *(_QWORD ***)(v28[-1].m128i_i64[0] + 32) == v36 )
            goto LABEL_48;
        }
        sub_13FF6D0(&v159, v28, (const __m128i *)src);
        v28 = (__m128i *)v160;
      }
LABEL_48:
      v25 = v166;
      v27 = v167;
      goto LABEL_34;
    }
    v25 = v166;
  }
  while ( (char *)v160 - (char *)v159 != (char *)v167 - (char *)v166 );
LABEL_142:
  if ( v29 != v28 )
  {
    v102 = (__int64 *)v25;
    while ( v29->m128i_i64[0] == *v102 && v29->m128i_i64[1] == v102[1] )
    {
      ++v29;
      v102 += 2;
      if ( v29 == v28 )
        goto LABEL_147;
    }
    goto LABEL_35;
  }
LABEL_147:
  if ( v25 )
    j_j___libc_free_0(v25, v168 - (__int8 *)v25);
  if ( v164 != v163 )
    _libc_free(v164);
  if ( v159 )
    j_j___libc_free_0(v159, (char *)v161 - (char *)v159);
  if ( v154 != v153 )
    _libc_free((unsigned __int64)v154);
  if ( v179 )
    j_j___libc_free_0(v179, v181 - (__int8 *)v179);
  if ( v177 != v176 )
    _libc_free(v177);
  if ( v172 )
    j_j___libc_free_0(v172, (char *)v174 - (char *)v172);
  if ( v170 != v169.m128i_i64[1] )
    _libc_free(v170);
  v169.m128i_i64[0] = a1;
  sub_13FE280(v169.m128i_i64, *v129);
}
