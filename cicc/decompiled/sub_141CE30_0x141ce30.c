// Function: sub_141CE30
// Address: 0x141ce30
//
__int64 __fastcall sub_141CE30(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __m128i *a4,
        unsigned __int8 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        unsigned __int8 a9,
        unsigned __int8 a10)
{
  __int64 *v11; // r15
  __m128i v13; // rdi
  __int64 v14; // r13
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // r8
  __int64 v20; // r12
  int v21; // r10d
  unsigned int v22; // edx
  __int64 v23; // r14
  __int64 v24; // rdi
  unsigned __int64 v25; // rax
  __int64 v26; // rbx
  __int64 v27; // r12
  __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rbx
  __int64 v33; // r12
  __int64 v34; // r13
  __int64 v35; // rax
  unsigned int v36; // r13d
  __int64 v38; // rdx
  int v39; // edx
  __int64 *v40; // r11
  unsigned __int64 v41; // rax
  __int64 *v42; // rbx
  int *v43; // rax
  int v44; // edx
  bool v45; // cc
  _QWORD *v46; // rdx
  int v47; // eax
  __m128i **v48; // r14
  unsigned int v49; // eax
  unsigned __int64 v50; // rax
  void *v51; // rcx
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rdi
  void **v55; // rdx
  void *v56; // r9
  __int64 v57; // rax
  __int64 v58; // rax
  __m128i *v59; // rax
  void *v60; // rbx
  int v61; // edx
  _QWORD *v62; // rax
  __int64 v63; // rcx
  __int64 v64; // r14
  __int64 v65; // r12
  unsigned __int64 v66; // rdi
  char *v67; // rax
  __int64 v68; // rdx
  void **v69; // r14
  void *v70; // rbx
  unsigned int v71; // r12d
  __int64 v72; // rax
  unsigned __int32 v73; // eax
  __int64 v74; // rdi
  __int64 v75; // rax
  __int64 v76; // r12
  __int64 v77; // r12
  __int64 v78; // rbx
  void *v79; // r14
  void *v80; // r9
  __int64 v81; // rax
  __m128i v82; // xmm2
  char v83; // al
  __int64 *v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rdx
  __int64 *v87; // rbx
  __int64 v88; // rbx
  unsigned __int64 v89; // r12
  unsigned __int64 v90; // rdi
  __int64 v91; // rax
  unsigned __int64 v92; // rdi
  __int64 v93; // rsi
  __int64 v94; // r9
  int v95; // eax
  int v96; // r8d
  __int64 v97; // rdi
  __int64 v98; // rcx
  unsigned int v99; // edx
  __int64 *v100; // rax
  __int64 v101; // r10
  __int64 *v102; // rax
  __m128i *v103; // rax
  __m128i *v104; // rdx
  __int64 v105; // rax
  char *v106; // rdi
  size_t v107; // rdx
  char *v108; // rax
  __int64 v109; // rdx
  __int64 *v110; // rbx
  __int64 *v111; // r15
  __int64 v112; // r12
  __int64 v113; // rdx
  __m128i *v114; // r12
  __int64 v115; // rbx
  __int64 v116; // rax
  __int64 v117; // r14
  __int64 v118; // rax
  __m128i *v119; // rax
  int v120; // eax
  int v121; // r11d
  __int64 v122; // r8
  __int64 *v123; // rdi
  void *v124; // r15
  __int64 v125; // rcx
  int v126; // r10d
  unsigned int v127; // edx
  __int64 *v128; // rax
  __int64 v129; // r14
  int v130; // edx
  __int64 v131; // rax
  int v132; // ecx
  int v133; // r8d
  __int64 v134; // rdi
  unsigned int v135; // ecx
  __int64 *v136; // rax
  __int64 v137; // r9
  int v138; // edx
  __int64 v139; // rax
  int v140; // ecx
  int v141; // r8d
  __int64 v142; // rdi
  unsigned int v143; // ecx
  __int64 *v144; // rax
  __int64 v145; // r9
  __int64 *v146; // r12
  __int64 v147; // rdx
  __int64 v148; // rax
  __m128i v149; // rdi
  __int64 v150; // rcx
  unsigned int v151; // r9d
  __int64 *v152; // rdx
  __int64 v153; // r10
  __int64 v154; // rax
  __m128i *v155; // rax
  int v156; // eax
  int v157; // r10d
  int v158; // edx
  int v159; // r10d
  int v160; // eax
  int v161; // r10d
  int v162; // edx
  int v163; // r11d
  int v164; // eax
  int v165; // r9d
  __int64 v166; // [rsp-10h] [rbp-850h]
  void *v167; // [rsp+8h] [rbp-838h]
  unsigned int v168; // [rsp+14h] [rbp-82Ch]
  __int64 v169; // [rsp+28h] [rbp-818h]
  char *v170; // [rsp+30h] [rbp-810h]
  int v172; // [rsp+44h] [rbp-7FCh]
  _QWORD *v173; // [rsp+48h] [rbp-7F8h]
  unsigned int v174; // [rsp+50h] [rbp-7F0h]
  char *v175; // [rsp+58h] [rbp-7E8h]
  __m128i **v176; // [rsp+60h] [rbp-7E0h]
  void *v177; // [rsp+60h] [rbp-7E0h]
  __int64 *v179; // [rsp+68h] [rbp-7D8h]
  unsigned __int8 v180; // [rsp+70h] [rbp-7D0h]
  __int64 v181; // [rsp+70h] [rbp-7D0h]
  void *desta; // [rsp+80h] [rbp-7C0h]
  unsigned __int64 v185; // [rsp+98h] [rbp-7A8h] BYREF
  __int64 v186; // [rsp+A0h] [rbp-7A0h] BYREF
  __int64 v187; // [rsp+A8h] [rbp-798h] BYREF
  void *v188; // [rsp+B0h] [rbp-790h] BYREF
  __int64 v189[3]; // [rsp+B8h] [rbp-788h] BYREF
  char v190; // [rsp+D0h] [rbp-770h]
  void *src[2]; // [rsp+E0h] [rbp-760h] BYREF
  __m128i v192; // [rsp+F0h] [rbp-750h] BYREF
  __int64 v193; // [rsp+100h] [rbp-740h]
  char *v194; // [rsp+108h] [rbp-738h] BYREF
  __int64 v195; // [rsp+110h] [rbp-730h]
  _BYTE v196[88]; // [rsp+118h] [rbp-728h] BYREF
  _QWORD *v197; // [rsp+170h] [rbp-6D0h] BYREF
  __int64 v198; // [rsp+178h] [rbp-6C8h] BYREF
  _QWORD v199[32]; // [rsp+180h] [rbp-6C0h] BYREF
  __m128i v200; // [rsp+280h] [rbp-5C0h] BYREF
  __m128i v201; // [rsp+290h] [rbp-5B0h] BYREF
  __int64 v202; // [rsp+2A0h] [rbp-5A0h]
  unsigned __int64 v203; // [rsp+2A8h] [rbp-598h]
  __m128i v204; // [rsp+2B0h] [rbp-590h] BYREF
  __int64 v205; // [rsp+2C0h] [rbp-580h]
  __int64 v206; // [rsp+858h] [rbp+18h]

  v11 = a3;
  v13 = a4[1];
  v180 = a9;
  v14 = a8;
  v172 = a5;
  v15 = a4->m128i_u64[1];
  v16 = *a3 & 0xFFFFFFFFFFFFFFFBLL | (4LL * a5);
  v17 = a4[2].m128i_i64[0];
  v185 = v16;
  v169 = a1 + 96;
  v200 = (__m128i)v16;
  v201 = 0u;
  v202 = 0;
  v203 = v15;
  v204 = v13;
  v18 = *(unsigned int *)(a1 + 120);
  v205 = v17;
  if ( !(_DWORD)v18 )
  {
    ++*(_QWORD *)(a1 + 96);
    goto LABEL_209;
  }
  v19 = *(_QWORD *)(a1 + 104);
  v20 = 0;
  v21 = 1;
  v22 = (v18 - 1) & (v16 ^ (v16 >> 9));
  v23 = v19 + 72LL * v22;
  v24 = *(_QWORD *)v23;
  if ( v16 != *(_QWORD *)v23 )
  {
    while ( v24 != -4 )
    {
      if ( !v20 && v24 == -16 )
        v20 = v23;
      v22 = (v18 - 1) & (v21 + v22);
      v23 = v19 + 72LL * v22;
      v24 = *(_QWORD *)v23;
      if ( v16 == *(_QWORD *)v23 )
        goto LABEL_3;
      ++v21;
    }
    if ( v20 )
      v23 = v20;
    ++*(_QWORD *)(a1 + 96);
    v39 = *(_DWORD *)(a1 + 112) + 1;
    if ( 4 * v39 < (unsigned int)(3 * v18) )
    {
      if ( (int)v18 - *(_DWORD *)(a1 + 116) - v39 > (unsigned int)v18 >> 3 )
      {
LABEL_39:
        *(_DWORD *)(a1 + 112) = v39;
        if ( *(_QWORD *)v23 != -4 )
          --*(_DWORD *)(a1 + 116);
        *(__m128i *)v23 = v200;
        *(__m128i *)(v23 + 16) = v201;
        *(_QWORD *)(v23 + 32) = v202;
        *(_QWORD *)(v23 + 40) = v203;
        *(__m128i *)(v23 + 48) = _mm_loadu_si128(&v204);
        *(_QWORD *)(v23 + 64) = v205;
        goto LABEL_42;
      }
LABEL_210:
      sub_1418D60(v169, v18);
      v18 = (__int64)&v200;
      sub_1414C80(v169, (unsigned __int64 *)&v200, &v197);
      v23 = (__int64)v197;
      v39 = *(_DWORD *)(a1 + 112) + 1;
      goto LABEL_39;
    }
LABEL_209:
    LODWORD(v18) = 2 * v18;
    goto LABEL_210;
  }
LABEL_3:
  v25 = *(_QWORD *)(v23 + 40);
  if ( v15 <= v25 )
  {
    if ( v15 < v25 )
    {
      v38 = a4[2].m128i_i64[0];
      v200.m128i_i64[0] = _mm_loadu_si128(a4).m128i_u64[0];
      v202 = v38;
      v200.m128i_i64[1] = v25;
      v201 = _mm_loadu_si128(a4 + 1);
      return (unsigned int)sub_141CE30(a1, a2, (_DWORD)v11, (unsigned int)&v200, v172, a6, a7, v14, a9, a10);
    }
  }
  else
  {
    v26 = *(_QWORD *)(v23 + 16);
    v27 = *(_QWORD *)(v23 + 24);
    *(_QWORD *)(v23 + 8) = 0;
    *(_QWORD *)(v23 + 40) = a4->m128i_i64[1];
    if ( v27 != v26 )
    {
      v28 = v26;
      do
      {
        v18 = *(_QWORD *)(v28 + 8);
        if ( (*(_DWORD *)(v28 + 8) & 7u) <= 2 )
        {
          v18 &= 0xFFFFFFFFFFFFFFF8LL;
          if ( v18 )
            sub_1412000(a1 + 128, v18, v185);
        }
        v28 += 16;
      }
      while ( v27 != v28 );
      v14 = a8;
      v29 = *(_QWORD *)(v23 + 16);
      if ( v29 != *(_QWORD *)(v23 + 24) )
        *(_QWORD *)(v23 + 24) = v29;
    }
  }
  v30 = *(_QWORD *)(v23 + 48);
  v31 = a4[1].m128i_i64[0];
  if ( v31 != v30
    || a4[1].m128i_i64[1] != *(_QWORD *)(v23 + 56)
    || (v18 = *(_QWORD *)(v23 + 64), a4[2].m128i_i64[0] != v18) )
  {
    if ( v30 || *(_QWORD *)(v23 + 56) || *(_QWORD *)(v23 + 64) )
    {
      v32 = *(_QWORD *)(v23 + 16);
      v33 = *(_QWORD *)(v23 + 24);
      *(_QWORD *)(v23 + 8) = 0;
      *(_QWORD *)(v23 + 48) = 0;
      *(_QWORD *)(v23 + 56) = 0;
      *(_QWORD *)(v23 + 64) = 0;
      if ( v32 != v33 )
      {
        v206 = v14;
        v34 = v32;
        do
        {
          v18 = *(_QWORD *)(v34 + 8);
          if ( (*(_DWORD *)(v34 + 8) & 7u) <= 2 )
          {
            v18 &= 0xFFFFFFFFFFFFFFF8LL;
            if ( v18 )
              sub_1412000(a1 + 128, v18, v185);
          }
          v34 += 16;
        }
        while ( v34 != v33 );
        v14 = v206;
        v35 = *(_QWORD *)(v23 + 16);
        if ( *(_QWORD *)(v23 + 24) != v35 )
          *(_QWORD *)(v23 + 24) = v35;
      }
      v31 = a4[1].m128i_i64[0];
    }
    if ( v31 || a4[1].m128i_i64[1] || a4[2].m128i_i64[0] )
    {
      v201 = 0u;
      v202 = 0;
      v200 = _mm_loadu_si128(a4);
      return (unsigned int)sub_141CE30(a1, a2, (_DWORD)v11, (unsigned int)&v200, v172, a6, a7, v14, a9, a10);
    }
  }
LABEL_42:
  v40 = *(__int64 **)(v23 + 24);
  v41 = (4LL * a9) | a6 & 0xFFFFFFFFFFFFFFFBLL;
  v42 = *(__int64 **)(v23 + 16);
  if ( *(_QWORD *)(v23 + 8) == v41 )
  {
    if ( *(_DWORD *)(v14 + 16) )
    {
      if ( v42 != v40 )
      {
        v122 = *(_QWORD *)(v14 + 8);
        v123 = *(__int64 **)(v23 + 16);
        v124 = (void *)*v11;
        v125 = *(unsigned int *)(v14 + 24);
        v126 = v125 - 1;
        while ( 1 )
        {
          if ( (_DWORD)v125 )
          {
            v127 = v126 & (((unsigned int)*v123 >> 9) ^ ((unsigned int)*v123 >> 4));
            v128 = (__int64 *)(v122 + 16LL * v127);
            v129 = *v128;
            if ( *v123 == *v128 )
            {
LABEL_156:
              if ( (__int64 *)(v122 + 16 * v125) != v128 && (void *)v128[1] != v124 )
                return 0;
            }
            else
            {
              v164 = 1;
              while ( v129 != -8 )
              {
                v165 = v164 + 1;
                v127 = v126 & (v164 + v127);
                v128 = (__int64 *)(v122 + 16LL * v127);
                v129 = *v128;
                if ( *v123 == *v128 )
                  goto LABEL_156;
                v164 = v165;
              }
            }
          }
          v123 += 2;
          if ( v40 == v123 )
            goto LABEL_189;
        }
      }
    }
    else
    {
      v124 = (void *)*v11;
      if ( v42 != v40 )
      {
LABEL_189:
        v146 = v42;
        v179 = v40;
        do
        {
          v197 = (_QWORD *)*v146;
          v198 = (__int64)v124;
          sub_141AAC0((__int64)&v200, v14, (__int64 *)&v197, &v198);
          v149.m128i_i64[1] = v146[1];
          if ( (v146[1] & 7) != 3 || (unsigned __int64)v149.m128i_i64[1] >> 61 != 1 )
          {
            v147 = *(_QWORD *)(a1 + 280);
            v148 = *(unsigned int *)(v147 + 48);
            if ( (_DWORD)v148 )
            {
              v149.m128i_i64[0] = *v146;
              v150 = *(_QWORD *)(v147 + 32);
              v151 = (v148 - 1) & (((unsigned int)*v146 >> 9) ^ ((unsigned int)*v146 >> 4));
              v152 = (__int64 *)(v150 + 16LL * v151);
              v153 = *v152;
              if ( *v146 == *v152 )
              {
LABEL_192:
                if ( v152 != (__int64 *)(v150 + 16 * v148) && v152[1] )
                {
                  v200 = v149;
                  v201.m128i_i64[0] = (__int64)v124;
                  v154 = *(unsigned int *)(a7 + 8);
                  if ( (unsigned int)v154 >= *(_DWORD *)(a7 + 12) )
                  {
                    sub_16CD150(a7, a7 + 16, 0, 24);
                    v154 = *(unsigned int *)(a7 + 8);
                  }
                  v155 = (__m128i *)(*(_QWORD *)a7 + 24 * v154);
                  *v155 = _mm_loadu_si128(&v200);
                  v155[1].m128i_i64[0] = v201.m128i_i64[0];
                  ++*(_DWORD *)(a7 + 8);
                }
              }
              else
              {
                v162 = 1;
                while ( v153 != -8 )
                {
                  v163 = v162 + 1;
                  v151 = (v148 - 1) & (v162 + v151);
                  v152 = (__int64 *)(v150 + 16LL * v151);
                  v153 = *v152;
                  if ( v149.m128i_i64[0] == *v152 )
                    goto LABEL_192;
                  v162 = v163;
                }
              }
            }
          }
          v146 += 2;
        }
        while ( v179 != v146 );
      }
    }
    return 1;
  }
  if ( v42 != v40 )
    v41 = 0;
  *(_QWORD *)(v23 + 8) = v41;
  v197 = v199;
  v199[0] = a6;
  v198 = 0x2000000001LL;
  v200.m128i_i64[0] = (__int64)&v201;
  v200.m128i_i64[1] = 0x1000000000LL;
  v174 = (__int64)(*(_QWORD *)(v23 + 24) - *(_QWORD *)(v23 + 16)) >> 4;
  v43 = (int *)sub_16D40F0(qword_4FBB410);
  if ( v43 )
    v44 = *v43;
  else
    v44 = qword_4FBB410[2];
  v45 = v44 <= 2;
  v46 = (_QWORD *)(v23 + 8);
  v47 = 2 * dword_4F993C0;
  if ( v45 )
    v47 = dword_4F993C0;
  v48 = (__m128i **)(v23 + 16);
  v168 = v47;
  v49 = v198;
  if ( !(_DWORD)v198 )
  {
LABEL_92:
    v36 = 1;
    sub_1414480(v48, v174);
    goto LABEL_93;
  }
  v173 = v46;
  v176 = v48;
LABEL_60:
  while ( 2 )
  {
    v60 = (void *)v197[v49 - 1];
    LODWORD(v198) = v49 - 1;
    if ( *(_DWORD *)(a7 + 8) <= 0x64u )
    {
      if ( !v180 )
      {
        v50 = sub_141C910(a1, a2, a4, v172, (__int64)v60, v176, v174, a10);
        v18 = v166;
        v51 = (void *)v50;
        if ( (v50 & 7) != 3 || v50 >> 61 != 1 )
        {
          v52 = *(_QWORD *)(a1 + 280);
          v53 = *(unsigned int *)(v52 + 48);
          if ( (_DWORD)v53 )
          {
            v54 = *(_QWORD *)(v52 + 32);
            v18 = ((_DWORD)v53 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
            v55 = (void **)(v54 + 16 * v18);
            v56 = *v55;
            if ( v60 == *v55 )
            {
LABEL_54:
              if ( v55 != (void **)(v54 + 16 * v53) && v55[1] )
              {
                v57 = *v11;
                v18 = a7;
                src[0] = v60;
                src[1] = v51;
                v192.m128i_i64[0] = v57;
                v58 = *(unsigned int *)(a7 + 8);
                if ( (unsigned int)v58 >= *(_DWORD *)(a7 + 12) )
                {
                  v18 = a7 + 16;
                  sub_16CD150(a7, a7 + 16, 0, 24);
                  v58 = *(unsigned int *)(a7 + 8);
                }
                goto LABEL_58;
              }
            }
            else
            {
              v158 = 1;
              while ( v56 != (void *)-8LL )
              {
                v159 = v158 + 1;
                v18 = ((_DWORD)v53 - 1) & (unsigned int)(v158 + v18);
                v55 = (void **)(v54 + 16LL * (unsigned int)v18);
                v56 = *v55;
                if ( v60 == *v55 )
                  goto LABEL_54;
                v158 = v159;
              }
            }
          }
        }
      }
      v61 = *((_DWORD *)v11 + 10);
      if ( !v61 )
      {
LABEL_126:
        v18 = (__int64)v60;
        src[0] = &v192;
        src[1] = (void *)0x1000000000LL;
        v108 = sub_1416080(a1 + 296, (__int64)v60);
        v175 = &v108[8 * v109];
        if ( v175 == v108 )
        {
LABEL_135:
          v114 = (__m128i *)src[0];
          if ( LODWORD(src[1]) <= v168 )
          {
            v115 = LODWORD(src[1]);
            v168 -= LODWORD(src[1]);
            v116 = (unsigned int)v198;
            v117 = 8LL * LODWORD(src[1]);
            if ( LODWORD(src[1]) > HIDWORD(v198) - (unsigned __int64)(unsigned int)v198 )
            {
              v18 = (__int64)v199;
              sub_16CD150(&v197, v199, LODWORD(src[1]) + (unsigned __int64)(unsigned int)v198, 8);
              v116 = (unsigned int)v198;
            }
            if ( v117 )
            {
              v18 = (__int64)v114;
              memcpy(&v197[v116], v114, 8 * v115);
              LODWORD(v116) = v198;
            }
            LODWORD(v198) = v116 + v115;
            v49 = v116 + v115;
            if ( src[0] != &v192 )
            {
              _libc_free((unsigned __int64)src[0]);
              v49 = v198;
            }
            goto LABEL_59;
          }
          if ( LODWORD(src[1]) )
          {
            v138 = 0;
            v139 = 0;
            do
            {
              v140 = *(_DWORD *)(v14 + 24);
              if ( v140 )
              {
                v18 = v114->m128i_i64[v139];
                v141 = v140 - 1;
                v142 = *(_QWORD *)(v14 + 8);
                v143 = (v140 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
                v144 = (__int64 *)(v142 + 16LL * v143);
                v145 = *v144;
                if ( *v144 == v18 )
                {
LABEL_185:
                  *v144 = -16;
                  v114 = (__m128i *)src[0];
                  --*(_DWORD *)(v14 + 16);
                  ++*(_DWORD *)(v14 + 20);
                }
                else
                {
                  v160 = 1;
                  while ( v145 != -8 )
                  {
                    v161 = v160 + 1;
                    v143 = v141 & (v160 + v143);
                    v144 = (__int64 *)(v142 + 16LL * v143);
                    v145 = *v144;
                    if ( v18 == *v144 )
                      goto LABEL_185;
                    v160 = v161;
                  }
                }
              }
              v139 = (unsigned int)(v138 + 1);
              v138 = v139;
            }
            while ( (unsigned int)v139 < LODWORD(src[1]) );
          }
        }
        else
        {
          v167 = v60;
          v110 = v11;
          v111 = (__int64 *)v108;
          while ( 1 )
          {
            while ( 1 )
            {
              v112 = *v111;
              v18 = v14;
              v187 = *v110;
              v186 = v112;
              sub_141AAC0((__int64)&v188, v14, &v186, &v187);
              if ( !v190 )
                break;
              v113 = LODWORD(src[1]);
              if ( LODWORD(src[1]) >= HIDWORD(src[1]) )
              {
                v18 = (__int64)&v192;
                sub_16CD150(src, &v192, 0, 8);
                v113 = LODWORD(src[1]);
              }
              ++v111;
              *((_QWORD *)src[0] + v113) = v112;
              ++LODWORD(src[1]);
              if ( v175 == (char *)v111 )
              {
LABEL_134:
                v11 = v110;
                v60 = v167;
                goto LABEL_135;
              }
            }
            if ( *(_QWORD *)(v189[1] + 8) != *v110 )
              break;
            if ( v175 == (char *)++v111 )
              goto LABEL_134;
          }
          v11 = v110;
          v114 = (__m128i *)src[0];
          v60 = v167;
          if ( LODWORD(src[1]) )
          {
            v130 = 0;
            v131 = 0;
            do
            {
              v132 = *(_DWORD *)(v14 + 24);
              if ( v132 )
              {
                v18 = v114->m128i_i64[v131];
                v133 = v132 - 1;
                v134 = *(_QWORD *)(v14 + 8);
                v135 = (v132 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
                v136 = (__int64 *)(v134 + 16LL * v135);
                v137 = *v136;
                if ( v18 == *v136 )
                {
LABEL_163:
                  *v136 = -16;
                  v114 = (__m128i *)src[0];
                  --*(_DWORD *)(v14 + 16);
                  ++*(_DWORD *)(v14 + 20);
                }
                else
                {
                  v156 = 1;
                  while ( v137 != -8 )
                  {
                    v157 = v156 + 1;
                    v135 = v133 & (v156 + v135);
                    v136 = (__int64 *)(v134 + 16LL * v135);
                    v137 = *v136;
                    if ( v18 == *v136 )
                      goto LABEL_163;
                    v156 = v157;
                  }
                }
              }
              v131 = (unsigned int)(v130 + 1);
              v130 = v131;
            }
            while ( (unsigned int)v131 < LODWORD(src[1]) );
          }
        }
        if ( v114 != &v192 )
          _libc_free((unsigned __int64)v114);
        *v173 = 0;
        goto LABEL_113;
      }
      v62 = (_QWORD *)v11[4];
      v63 = (__int64)&v62[(unsigned int)(v61 - 1) + 1];
      while ( v60 != *(void **)(*v62 + 40LL) )
      {
        if ( (_QWORD *)v63 == ++v62 )
          goto LABEL_126;
      }
      if ( !(unsigned __int8)sub_143B8D0(v11, v18, *v62, v63) )
        goto LABEL_112;
      if ( v174 != v176[1] - *v176 )
        sub_1414480(v176, v174);
      v64 = v200.m128i_i64[0];
      v65 = v200.m128i_i64[0] + 88LL * v200.m128i_u32[2];
      while ( v64 != v65 )
      {
        while ( 1 )
        {
          v65 -= 88;
          v66 = *(_QWORD *)(v65 + 40);
          if ( v66 == v65 + 56 )
            break;
          _libc_free(v66);
          if ( v64 == v65 )
            goto LABEL_73;
        }
      }
LABEL_73:
      v200.m128i_i32[2] = 0;
      v67 = sub_1416080(a1 + 296, (__int64)v60);
      v170 = &v67[8 * v68];
      if ( v67 == v170 )
      {
LABEL_85:
        if ( v200.m128i_i32[2] )
        {
          v78 = 0;
          v181 = 88LL * v200.m128i_u32[2];
          do
          {
            v79 = *(void **)(v78 + v200.m128i_i64[0] + 8);
            v80 = *(void **)(v78 + v200.m128i_i64[0]);
            if ( !v79 )
              goto LABEL_142;
            v81 = a4[2].m128i_i64[0];
            desta = *(void **)(v78 + v200.m128i_i64[0]);
            v82 = _mm_loadu_si128(a4 + 1);
            src[1] = (void *)_mm_loadu_si128(a4).m128i_i64[1];
            v193 = v81;
            src[0] = v79;
            v192 = v82;
            v83 = sub_141CE30(
                    a1,
                    a2,
                    (int)v78 + v200.m128i_i32[0] + 8,
                    (unsigned int)src,
                    v172,
                    (_DWORD)v80,
                    a7,
                    v14,
                    0,
                    1);
            v80 = desta;
            if ( !v83 )
            {
LABEL_142:
              src[0] = v80;
              src[1] = (void *)0x6000000000000003LL;
              v118 = *(unsigned int *)(a7 + 8);
              v192.m128i_i64[0] = (__int64)v79;
              if ( (unsigned int)v118 >= *(_DWORD *)(a7 + 12) )
              {
                sub_16CD150(a7, a7 + 16, 0, 24);
                v118 = *(unsigned int *)(a7 + 8);
              }
              v119 = (__m128i *)(*(_QWORD *)a7 + 24 * v118);
              *v119 = _mm_loadu_si128((const __m128i *)src);
              v119[1].m128i_i64[0] = v192.m128i_i64[0];
              ++*(_DWORD *)(a7 + 8);
              sub_1418FA0(v169, (__int64 *)&v185)[1] = 0;
            }
            v78 += 88;
          }
          while ( v181 != v78 );
        }
        v18 = (__int64)&v185;
        v84 = sub_1418FA0(v169, (__int64 *)&v185);
        v180 = 0;
        v85 = v84[3];
        v84[1] = 0;
        v86 = v85 - v84[2];
        v173 = v84 + 1;
        v87 = v84 + 2;
        v49 = v198;
        v176 = (__m128i **)v87;
        v174 = v86 >> 4;
        if ( !(_DWORD)v198 )
          goto LABEL_91;
        continue;
      }
      v69 = (void **)v67;
      v177 = v60;
      while ( 1 )
      {
        v70 = *v69;
        v71 = *((_DWORD *)v11 + 10);
        src[1] = (void *)*v11;
        v72 = v11[1];
        src[0] = v70;
        v192.m128i_i64[0] = v72;
        v192.m128i_i64[1] = v11[2];
        v193 = v11[3];
        v194 = v196;
        v195 = 0x400000000LL;
        if ( !v71 )
          break;
        v106 = v196;
        v107 = 8LL * v71;
        if ( v71 <= 4
          || (sub_16CD150(&v194, v196, v71, 8), v106 = v194, (v107 = 8LL * *((unsigned int *)v11 + 10)) != 0) )
        {
          memcpy(v106, (const void *)v11[4], v107);
        }
        LODWORD(v195) = v71;
        v73 = v200.m128i_u32[2];
        if ( v200.m128i_i32[2] >= (unsigned __int32)v200.m128i_i32[3] )
          goto LABEL_123;
LABEL_77:
        v74 = v200.m128i_i64[0] + 88LL * v73;
        if ( v74 )
        {
          *(void **)v74 = src[0];
          *(void **)(v74 + 8) = src[1];
          *(__m128i *)(v74 + 16) = v192;
          *(_QWORD *)(v74 + 32) = v193;
          *(_QWORD *)(v74 + 40) = v74 + 56;
          *(_QWORD *)(v74 + 48) = 0x400000000LL;
          if ( (_DWORD)v195 )
            sub_14117E0(v74 + 40, &v194);
          v73 = v200.m128i_u32[2];
        }
        v75 = v73 + 1;
        v200.m128i_i32[2] = v75;
        if ( v194 != v196 )
        {
          _libc_free((unsigned __int64)v194);
          v75 = v200.m128i_u32[2];
        }
        v76 = v200.m128i_i64[0] + 88 * v75 - 88;
        sub_143C480(v76 + 8, v177, v70, *(_QWORD *)(a1 + 280), 0);
        v77 = *(_QWORD *)(v76 + 8);
        v188 = v70;
        v189[0] = v77;
        sub_141AAC0((__int64)src, v14, (__int64 *)&v188, v189);
        if ( !(_BYTE)v193 )
        {
          --v200.m128i_i32[2];
          v91 = v200.m128i_i64[0] + 88LL * v200.m128i_u32[2];
          v92 = *(_QWORD *)(v91 + 40);
          if ( v92 != v91 + 56 )
            _libc_free(v92);
          if ( *(_QWORD *)(v192.m128i_i64[0] + 8) != v77 )
          {
            v60 = v177;
            if ( v200.m128i_i32[2] )
            {
              v93 = 0;
              v94 = 88LL * v200.m128i_u32[2];
              do
              {
                v95 = *(_DWORD *)(v14 + 24);
                if ( v95 )
                {
                  v96 = v95 - 1;
                  v97 = *(_QWORD *)(v14 + 8);
                  v98 = *(_QWORD *)(v200.m128i_i64[0] + v93);
                  v99 = (v95 - 1) & (((unsigned int)v98 >> 9) ^ ((unsigned int)v98 >> 4));
                  v100 = (__int64 *)(v97 + 16LL * v99);
                  v101 = *v100;
                  if ( v98 == *v100 )
                  {
LABEL_109:
                    *v100 = -16;
                    --*(_DWORD *)(v14 + 16);
                    ++*(_DWORD *)(v14 + 20);
                  }
                  else
                  {
                    v120 = 1;
                    while ( v101 != -8 )
                    {
                      v121 = v120 + 1;
                      v99 = v96 & (v120 + v99);
                      v100 = (__int64 *)(v97 + 16LL * v99);
                      v101 = *v100;
                      if ( v98 == *v100 )
                        goto LABEL_109;
                      v120 = v121;
                    }
                  }
                }
                v93 += 88;
              }
              while ( v94 != v93 );
            }
            v102 = sub_1418FA0(v169, (__int64 *)&v185);
            v18 = (__int64)(v102 + 1);
            v173 = v102 + 1;
            v176 = (__m128i **)(v102 + 2);
            v174 = (v102[3] - v102[2]) >> 4;
LABEL_112:
            *v173 = 0;
            if ( v180 )
            {
              v36 = 0;
              goto LABEL_93;
            }
LABEL_113:
            v103 = v176[1];
            if ( v103 == *v176 )
            {
LABEL_116:
              v49 = v198;
            }
            else
            {
              while ( 1 )
              {
                v104 = v103 - 1;
                if ( v60 == (void *)v103[-1].m128i_i64[0] )
                  break;
                --v103;
                if ( v104 == *v176 )
                  goto LABEL_116;
              }
              v103[-1].m128i_i64[1] = 0x6000000000000003LL;
              v105 = *v11;
              src[1] = (void *)0x6000000000000003LL;
              v18 = a7;
              v192.m128i_i64[0] = v105;
              src[0] = v60;
              v58 = *(unsigned int *)(a7 + 8);
              if ( (unsigned int)v58 >= *(_DWORD *)(a7 + 12) )
              {
                v18 = a7 + 16;
                sub_16CD150(a7, a7 + 16, 0, 24);
                v58 = *(unsigned int *)(a7 + 8);
              }
LABEL_58:
              v59 = (__m128i *)(*(_QWORD *)a7 + 24 * v58);
              *v59 = _mm_loadu_si128((const __m128i *)src);
              v59[1].m128i_i64[0] = v192.m128i_i64[0];
              ++*(_DWORD *)(a7 + 8);
              v49 = v198;
            }
LABEL_59:
            v180 = 0;
            if ( !v49 )
            {
LABEL_91:
              v48 = v176;
              goto LABEL_92;
            }
            goto LABEL_60;
          }
        }
        if ( v170 == (char *)++v69 )
          goto LABEL_85;
      }
      v73 = v200.m128i_u32[2];
      if ( v200.m128i_i32[2] < (unsigned __int32)v200.m128i_i32[3] )
        goto LABEL_77;
LABEL_123:
      sub_1414610((__int64)&v200, 0);
      v73 = v200.m128i_u32[2];
      goto LABEL_77;
    }
    break;
  }
  LODWORD(v198) = 0;
  if ( v174 != v176[1] - *v176 )
    sub_1414480(v176, v174);
  v36 = 0;
  *v173 = 0;
LABEL_93:
  v88 = v200.m128i_i64[0];
  v89 = v200.m128i_i64[0] + 88LL * v200.m128i_u32[2];
  if ( v200.m128i_i64[0] != v89 )
  {
    do
    {
      v89 -= 88LL;
      v90 = *(_QWORD *)(v89 + 40);
      if ( v90 != v89 + 56 )
        _libc_free(v90);
    }
    while ( v88 != v89 );
    v89 = v200.m128i_i64[0];
  }
  if ( (__m128i *)v89 != &v201 )
    _libc_free(v89);
  if ( v197 != v199 )
    _libc_free((unsigned __int64)v197);
  return v36;
}
