// Function: sub_1044BF0
// Address: 0x1044bf0
//
_BYTE *__fastcall sub_1044BF0(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, _DWORD *a5)
{
  __int64 v5; // r14
  _BYTE *v6; // r13
  __int64 v8; // r9
  _BYTE *v9; // r15
  char v10; // al
  _DWORD *v11; // rdx
  bool v12; // zf
  unsigned __int8 *v13; // r12
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rdi
  int v17; // eax
  unsigned __int64 v18; // rcx
  __m128i v19; // xmm7
  __m128i v20; // xmm0
  __m128i v21; // xmm7
  unsigned __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  unsigned int v29; // eax
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 *v32; // rcx
  __int64 v33; // r10
  int v34; // r11d
  _QWORD *v35; // rax
  unsigned int v36; // edx
  __int64 *v37; // rax
  int v38; // eax
  int v39; // r12d
  _BYTE **v40; // r15
  unsigned __int64 v41; // rax
  int v42; // eax
  unsigned int v43; // esi
  __int64 v44; // rax
  const __m128i *v45; // r15
  unsigned __int64 v46; // r13
  __int64 v47; // r12
  unsigned int *v48; // r14
  __int64 v49; // r11
  const __m128i *v50; // r15
  _DWORD *v51; // rax
  char *v52; // rbx
  char v53; // al
  _DWORD *v54; // rdx
  char **v55; // rbx
  __int64 v56; // rax
  unsigned __int64 v57; // r11
  unsigned __int64 v58; // rdx
  __int64 *m128i_i64; // rax
  unsigned __int8 *v60; // rbx
  int v61; // eax
  unsigned __int64 v62; // rax
  unsigned int *v63; // r13
  unsigned int *v64; // rbx
  unsigned int v65; // ecx
  int v66; // edi
  __int64 v67; // r12
  __int64 *v68; // rax
  int v69; // edx
  int v70; // ecx
  char *v71; // r12
  __int64 v72; // rax
  _DWORD *v73; // rax
  char *v74; // r13
  char v75; // al
  __int64 v76; // rdi
  _DWORD *v77; // rdx
  char **v78; // r13
  __int64 v79; // rax
  unsigned __int64 v80; // rdx
  __int64 v81; // rcx
  unsigned __int64 v82; // rax
  __int64 v83; // rax
  __int64 v84; // rdx
  char *v85; // rdi
  int v86; // eax
  __int64 v87; // rdx
  _QWORD *v88; // rax
  _QWORD *k; // rdx
  __int64 v91; // r13
  char *v92; // rbx
  const __m128i *v93; // r15
  __int64 v94; // rax
  __m128i v95; // xmm0
  unsigned __int64 v96; // rdx
  unsigned __int64 v97; // r15
  __int64 v98; // rax
  unsigned __int64 v99; // rdx
  char *m128i_i8; // rax
  __int64 v101; // r13
  unsigned __int64 v102; // rax
  _QWORD *v103; // rax
  __int64 v104; // r12
  unsigned __int64 v105; // rax
  _QWORD *v106; // rax
  __int64 v107; // rcx
  _QWORD *i; // rdx
  unsigned int v109; // ecx
  unsigned int v110; // eax
  int v111; // eax
  unsigned __int64 v112; // rax
  __int64 v113; // rax
  int v114; // r13d
  __int64 v115; // r12
  __m128i *v116; // r13
  int v117; // edx
  __int64 v118; // rax
  char *v119; // r12
  char *v120; // rbx
  __int64 v121; // rdx
  __int32 v122; // eax
  __int64 v123; // rax
  __int64 v124; // rdx
  __int64 v125; // rcx
  int v126; // edx
  unsigned __int8 *v127; // rdx
  unsigned __int8 *v128; // rdx
  __m128i *v129; // r13
  __int64 v130; // rdx
  __int64 *v131; // r12
  __int64 *v132; // rbx
  __int64 *v133; // rdi
  __int64 v134; // rdx
  __int32 v135; // eax
  __int64 v136; // rdx
  __int64 v137; // rcx
  __int64 v138; // rdx
  __int64 v139; // rcx
  _QWORD *j; // rdx
  __m128i v141; // xmm7
  __int64 v142; // r8
  __m128i v143; // xmm7
  __m128i v144; // xmm7
  const __m128i *v145; // rbx
  __m128i *v146; // rax
  __int64 v147; // rdi
  _BYTE *v148; // rbx
  unsigned __int64 v149; // [rsp+8h] [rbp-278h]
  unsigned __int64 v150; // [rsp+20h] [rbp-260h]
  unsigned __int64 v151; // [rsp+28h] [rbp-258h]
  __int64 v152; // [rsp+38h] [rbp-248h]
  unsigned __int8 *v153; // [rsp+38h] [rbp-248h]
  __int64 v154; // [rsp+48h] [rbp-238h]
  __m128i v155; // [rsp+50h] [rbp-230h] BYREF
  unsigned int *v156; // [rsp+60h] [rbp-220h]
  unsigned __int64 v157; // [rsp+68h] [rbp-218h]
  __int64 *v158; // [rsp+78h] [rbp-208h] BYREF
  unsigned int *v159; // [rsp+80h] [rbp-200h] BYREF
  __int64 v160; // [rsp+88h] [rbp-1F8h]
  _BYTE v161[32]; // [rsp+90h] [rbp-1F0h] BYREF
  __m128i v162; // [rsp+B0h] [rbp-1D0h] BYREF
  __m128i v163; // [rsp+C0h] [rbp-1C0h]
  __m128i v164; // [rsp+D0h] [rbp-1B0h]
  _BYTE *v165; // [rsp+E0h] [rbp-1A0h]
  _BYTE *v166; // [rsp+E8h] [rbp-198h]
  char v167; // [rsp+F4h] [rbp-18Ch]
  _BYTE *v168; // [rsp+100h] [rbp-180h] BYREF
  __int64 v169; // [rsp+108h] [rbp-178h]
  _BYTE v170[64]; // [rsp+110h] [rbp-170h] BYREF
  __m128i *v171; // [rsp+150h] [rbp-130h] BYREF
  __int64 v172; // [rsp+158h] [rbp-128h]
  _BYTE v173[64]; // [rsp+160h] [rbp-120h] BYREF
  __m128i *v174; // [rsp+1A0h] [rbp-E0h] BYREF
  __int64 v175; // [rsp+1A8h] [rbp-D8h]
  _BYTE v176[64]; // [rsp+1B0h] [rbp-D0h] BYREF
  _BYTE v177[56]; // [rsp+1F0h] [rbp-90h] BYREF
  unsigned __int8 *v178; // [rsp+228h] [rbp-58h]
  char v179; // [rsp+234h] [rbp-4Ch]

  v5 = a1;
  v6 = a3;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 24) = a4;
  *(_QWORD *)(a1 + 32) = a5;
  v8 = (unsigned int)*a5;
  if ( !(_DWORD)v8 )
  {
    *a5 = 1;
    a5 = *(_DWORD **)(a1 + 32);
  }
  if ( *a3 != 26 )
  {
    v165 = a3;
    v166 = a3;
    v162 = _mm_loadu_si128((const __m128i *)(a4 + 8));
    v167 = 0;
    v163 = _mm_loadu_si128((const __m128i *)(a4 + 24));
    v164 = _mm_loadu_si128((const __m128i *)(a4 + 40));
    if ( *a5 )
    {
      LOBYTE(v157) = 0;
      v9 = a3;
      do
      {
LABEL_7:
        v166 = v9;
        v10 = *v9;
        if ( *v9 == 27 )
        {
          if ( *(_BYTE **)(*(_QWORD *)a1 + 128LL) == v9 )
            return v9;
          v11 = *(_DWORD **)(a1 + 32);
          v12 = (*v11)-- == 1;
          if ( v12
            || (unsigned __int8)sub_103AFA0(
                                  (__int64)v9,
                                  &v162,
                                  *(unsigned __int8 **)(*(_QWORD *)(a1 + 24) + 56LL),
                                  *(_QWORD ***)(a1 + 16)) )
          {
            return v9;
          }
          v10 = *v9;
        }
        if ( v10 == 26 )
        {
          v40 = (_BYTE **)(v9 - 32);
        }
        else
        {
          if ( v10 != 27 )
            break;
          v40 = (_BYTE **)(v9 - 64);
        }
        v9 = *v40;
      }
      while ( v9 );
      if ( !(_BYTE)v157 )
        goto LABEL_15;
      goto LABEL_173;
    }
LABEL_5:
    *a5 = 1;
    v9 = v166;
    if ( v166 )
    {
      LOBYTE(v157) = 1;
      goto LABEL_7;
    }
LABEL_173:
    **(_DWORD **)(a1 + 32) = 0;
LABEL_15:
    v13 = v166;
    goto LABEL_16;
  }
  v6 = (_BYTE *)*((_QWORD *)a3 - 4);
  v167 = 0;
  v162 = _mm_loadu_si128((const __m128i *)(a4 + 8));
  v165 = v6;
  v163 = _mm_loadu_si128((const __m128i *)(a4 + 24));
  v166 = v6;
  v164 = _mm_loadu_si128((const __m128i *)(a4 + 40));
  if ( !*a5 )
    goto LABEL_5;
  LOBYTE(v157) = 0;
  v9 = v6;
  v13 = 0;
  if ( v6 )
    goto LABEL_7;
LABEL_16:
  v14 = *(unsigned int *)(a1 + 48);
  v15 = *(_QWORD *)(a1 + 40);
  v16 = *(unsigned int *)(a1 + 52);
  v17 = *(_DWORD *)(v5 + 48);
  v18 = v15 + 72 * v14;
  if ( v14 >= v16 )
  {
    v141 = _mm_loadu_si128((const __m128i *)(a4 + 8));
    v142 = v14 + 1;
    *(_QWORD *)&v177[48] = v6;
    v178 = v13;
    *(__m128i *)v177 = v141;
    v143 = _mm_loadu_si128((const __m128i *)(a4 + 24));
    v179 = 0;
    *(__m128i *)&v177[16] = v143;
    v144 = _mm_loadu_si128((const __m128i *)(a4 + 40));
    v145 = (const __m128i *)v177;
    *(__m128i *)&v177[32] = v144;
    if ( v16 < v14 + 1 )
    {
      v147 = v5 + 40;
      if ( v15 > (unsigned __int64)v177 || v18 <= (unsigned __int64)v177 )
      {
        sub_C8D5F0(v147, (const void *)(v5 + 56), v14 + 1, 0x48u, v142, v8);
        v15 = *(_QWORD *)(v5 + 40);
        v14 = *(unsigned int *)(v5 + 48);
      }
      else
      {
        v148 = &v177[-v15];
        sub_C8D5F0(v147, (const void *)(v5 + 56), v14 + 1, 0x48u, v142, v8);
        v15 = *(_QWORD *)(v5 + 40);
        v14 = *(unsigned int *)(v5 + 48);
        v145 = (const __m128i *)&v148[v15];
      }
    }
    v146 = (__m128i *)(v15 + 72 * v14);
    *v146 = _mm_loadu_si128(v145);
    v146[1] = _mm_loadu_si128(v145 + 1);
    v146[2] = _mm_loadu_si128(v145 + 2);
    v146[3] = _mm_loadu_si128(v145 + 3);
    v146[4].m128i_i64[0] = v145[4].m128i_i64[0];
    v22 = (unsigned int)(*(_DWORD *)(v5 + 48) + 1);
    *(_DWORD *)(v5 + 48) = v22;
  }
  else
  {
    if ( v18 )
    {
      v19 = _mm_loadu_si128((const __m128i *)(a4 + 8));
      v20 = _mm_loadu_si128((const __m128i *)(a4 + 24));
      *(_QWORD *)(v18 + 48) = v6;
      *(_QWORD *)(v18 + 56) = v13;
      *(__m128i *)v18 = v19;
      v21 = _mm_loadu_si128((const __m128i *)(a4 + 40));
      *(__m128i *)(v18 + 16) = v20;
      *(_BYTE *)(v18 + 68) = 0;
      *(__m128i *)(v18 + 32) = v21;
      v17 = *(_DWORD *)(v5 + 48);
    }
    v22 = (unsigned int)(v17 + 1);
    *(_DWORD *)(v5 + 48) = v22;
  }
  v150 = v22;
  v23 = (__int64)v13;
  v168 = v170;
  v169 = 0x1000000000LL;
  v159 = (unsigned int *)v161;
  v160 = 0x800000000LL;
  v171 = (__m128i *)v173;
  v172 = 0x400000000LL;
  sub_103E7F0(v5, v13, (__int64)&v168, 0);
LABEL_21:
  v26 = *((_QWORD *)v13 + 8);
  v27 = *(_QWORD *)(v5 + 8);
  if ( !v26 )
    goto LABEL_75;
LABEL_22:
  v28 = (unsigned int)(*(_DWORD *)(v26 + 44) + 1);
  v29 = *(_DWORD *)(v26 + 44) + 1;
LABEL_23:
  if ( v29 >= *(_DWORD *)(v27 + 32) )
    BUG();
  v30 = *(_QWORD *)v5;
  v31 = *(_QWORD *)(*(_QWORD *)v5 + 128LL);
  v32 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v27 + 24) + 8 * v28) + 8LL);
  if ( v32 )
  {
    v33 = *(_QWORD *)(v30 + 104);
    v24 = *(unsigned int *)(v30 + 120);
    v34 = v24 - 1;
    do
    {
      v23 = *v32;
      if ( (_DWORD)v24 )
      {
        v36 = v34 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v37 = (__int64 *)(v33 + 16LL * v36);
        v25 = *v37;
        if ( v23 == *v37 )
        {
LABEL_26:
          if ( (__int64 *)(v33 + 16 * v24) != v37 )
          {
            v35 = (_QWORD *)v37[1];
            if ( v35 )
            {
              v41 = *v35 & 0xFFFFFFFFFFFFFFF8LL;
              v31 = v41 - 48;
              if ( !v41 )
                v31 = 0;
              break;
            }
          }
        }
        else
        {
          v38 = 1;
          while ( v25 != -4096 )
          {
            v39 = v38 + 1;
            v36 = v34 & (v38 + v36);
            v37 = (__int64 *)(v33 + 16LL * v36);
            v25 = *v37;
            if ( v23 == *v37 )
              goto LABEL_26;
            v38 = v39;
          }
        }
      }
      v32 = (__int64 *)v32[1];
    }
    while ( v32 );
  }
  v42 = v169;
  if ( (_DWORD)v169 )
  {
    v155.m128i_i64[0] = v5 + 2360;
    while ( 1 )
    {
      v43 = *(_DWORD *)&v168[4 * v42 - 4];
      LODWORD(v169) = v42 - 1;
      v44 = v43;
      LODWORD(v156) = v43;
      v23 = (__int64)v177;
      v152 = v44;
      v45 = (const __m128i *)(*(_QWORD *)(v5 + 40) + 72 * v44);
      *(_QWORD *)v177 = v45[3].m128i_i64[1];
      *(__m128i *)&v177[8] = _mm_loadu_si128(v45);
      *(__m128i *)&v177[24] = _mm_loadu_si128(v45 + 1);
      *(__m128i *)&v177[40] = _mm_loadu_si128(v45 + 2);
      LOBYTE(v157) = sub_103F4F0(v155.m128i_i64[0], (__int64 *)v177, &v158);
      if ( (_BYTE)v157 )
        goto LABEL_45;
      v66 = *(_DWORD *)(v5 + 2376);
      v67 = *(unsigned int *)(v5 + 2384);
      v68 = v158;
      ++*(_QWORD *)(v5 + 2360);
      v69 = v66 + 1;
      v174 = (__m128i *)v68;
      if ( 4 * (v66 + 1) >= (unsigned int)(3 * v67) )
        break;
      if ( (int)v67 - *(_DWORD *)(v5 + 2380) - v69 <= (unsigned int)v67 >> 3 )
      {
        v101 = *(_QWORD *)(v5 + 2368);
        v105 = ((((((((((unsigned int)(v67 - 1) | ((unsigned __int64)(unsigned int)(v67 - 1) >> 1)) >> 2)
                    | (unsigned int)(v67 - 1)
                    | ((unsigned __int64)(unsigned int)(v67 - 1) >> 1)) >> 4)
                  | (((unsigned int)(v67 - 1) | ((unsigned __int64)(unsigned int)(v67 - 1) >> 1)) >> 2)
                  | (unsigned int)(v67 - 1)
                  | ((unsigned __int64)(unsigned int)(v67 - 1) >> 1)) >> 8)
                | (((((unsigned int)(v67 - 1) | ((unsigned __int64)(unsigned int)(v67 - 1) >> 1)) >> 2)
                  | (unsigned int)(v67 - 1)
                  | ((unsigned __int64)(unsigned int)(v67 - 1) >> 1)) >> 4)
                | (((unsigned int)(v67 - 1) | ((unsigned __int64)(unsigned int)(v67 - 1) >> 1)) >> 2)
                | (unsigned int)(v67 - 1)
                | ((unsigned __int64)(unsigned int)(v67 - 1) >> 1)) >> 16)
              | (((((((unsigned int)(v67 - 1) | ((unsigned __int64)(unsigned int)(v67 - 1) >> 1)) >> 2)
                  | (unsigned int)(v67 - 1)
                  | ((unsigned __int64)(unsigned int)(v67 - 1) >> 1)) >> 4)
                | (((unsigned int)(v67 - 1) | ((unsigned __int64)(unsigned int)(v67 - 1) >> 1)) >> 2)
                | (unsigned int)(v67 - 1)
                | ((unsigned __int64)(unsigned int)(v67 - 1) >> 1)) >> 8)
              | (((((unsigned int)(v67 - 1) | ((unsigned __int64)(unsigned int)(v67 - 1) >> 1)) >> 2)
                | (unsigned int)(v67 - 1)
                | ((unsigned __int64)(unsigned int)(v67 - 1) >> 1)) >> 4)
              | (((unsigned int)(v67 - 1) | ((unsigned __int64)(unsigned int)(v67 - 1) >> 1)) >> 2)
              | (unsigned int)(v67 - 1)
              | ((unsigned __int64)(unsigned int)(v67 - 1) >> 1))
             + 1;
        if ( (unsigned int)v105 < 0x40 )
          LODWORD(v105) = 64;
        *(_DWORD *)(v5 + 2384) = v105;
        v106 = (_QWORD *)sub_C7D670(56LL * (unsigned int)v105, 8);
        *(_QWORD *)(v5 + 2368) = v106;
        if ( !v101 )
        {
          v107 = *(unsigned int *)(v5 + 2384);
          *(_QWORD *)(v5 + 2376) = 0;
          for ( i = &v106[7 * v107]; i != v106; v106 += 7 )
          {
            if ( v106 )
            {
              *v106 = -4096;
              v106[1] = -4096;
              v106[2] = -3;
              v106[3] = 0;
              v106[4] = 0;
              v106[5] = 0;
              v106[6] = 0;
            }
          }
          goto LABEL_161;
        }
LABEL_160:
        v104 = 56 * v67;
        sub_1044A90(v155.m128i_i64[0], v101, v101 + v104);
        sub_C7D6A0(v101, v104, 8);
LABEL_161:
        sub_103F4F0(v155.m128i_i64[0], (__int64 *)v177, (__int64 **)&v174);
        v69 = *(_DWORD *)(v5 + 2376) + 1;
        v68 = (__int64 *)v174;
      }
      *(_DWORD *)(v5 + 2376) = v69;
      if ( *v68 != -4096 || v68[1] != -4096 || v68[2] != -3 || v68[3] || v68[4] || v68[5] || v68[6] )
        --*(_DWORD *)(v5 + 2380);
      v71 = 0;
      *v68 = *(_QWORD *)v177;
      *(__m128i *)(v68 + 1) = _mm_loadu_si128((const __m128i *)&v177[8]);
      *(__m128i *)(v68 + 3) = _mm_loadu_si128((const __m128i *)&v177[24]);
      *(__m128i *)(v68 + 5) = _mm_loadu_si128((const __m128i *)&v177[40]);
      v72 = *(_QWORD *)(v5 + 24);
      if ( *(_BYTE *)(v72 + 72)
        && v45->m128i_i64[0] == *(_QWORD *)(v72 + 8)
        && v45->m128i_i64[1] == *(_QWORD *)(v72 + 16)
        && v45[1].m128i_i64[0] == *(_QWORD *)(v72 + 24)
        && v45[1].m128i_i64[1] == *(_QWORD *)(v72 + 32)
        && v45[2].m128i_i64[0] == *(_QWORD *)(v72 + 40)
        && v45[2].m128i_i64[1] == *(_QWORD *)(v72 + 48) )
      {
        v71 = *(char **)(v72 + 64);
      }
      v73 = *(_DWORD **)(v5 + 32);
      v23 = (unsigned int)*v73;
      if ( (_DWORD)v23 )
      {
        v74 = (char *)v45[3].m128i_i64[1];
        if ( !v74 )
          goto LABEL_98;
      }
      else
      {
        *v73 = 1;
        v74 = (char *)v45[3].m128i_i64[1];
        LOBYTE(v157) = 1;
        if ( !v74 )
        {
LABEL_96:
          **(_DWORD **)(v5 + 32) = 0;
LABEL_97:
          v74 = (char *)v45[3].m128i_i64[1];
          goto LABEL_98;
        }
      }
      v45[3].m128i_i64[1] = (__int64)v74;
      if ( (char *)v31 == v74 )
        goto LABEL_104;
      while ( v71 != v74 )
      {
        v75 = *v74;
        if ( *v74 == 27 )
        {
          v76 = *(_QWORD *)v5;
          if ( v74 == *(char **)(*(_QWORD *)v5 + 128LL) )
            goto LABEL_112;
          v77 = *(_DWORD **)(v5 + 32);
          v12 = (*v77)-- == 1;
          if ( v12
            || (v23 = (__int64)v45,
                (unsigned __int8)sub_103AFA0(
                                   (__int64)v74,
                                   v45,
                                   *(unsigned __int8 **)(*(_QWORD *)(v5 + 24) + 56LL),
                                   *(_QWORD ***)(v5 + 16))) )
          {
            v76 = *(_QWORD *)v5;
LABEL_112:
            v23 = (__int64)v74;
            if ( sub_1041420(v76, (__int64)v74, v31) )
            {
              v97 = v149 & 0xFFFFFFFF00000000LL | v152;
              v98 = (unsigned int)v172;
              v149 = v97;
              v99 = (unsigned int)v172 + 1LL;
              if ( v99 > HIDWORD(v172) )
              {
                v23 = (__int64)v173;
                sub_C8D5F0((__int64)&v171, v173, v99, 0x10u, v24, v25);
                v98 = (unsigned int)v172;
              }
              m128i_i8 = v171[v98].m128i_i8;
              *(_QWORD *)m128i_i8 = v74;
              *((_QWORD *)m128i_i8 + 1) = v97;
              LODWORD(v172) = v172 + 1;
              goto LABEL_45;
            }
            v81 = *(_QWORD *)(v5 + 40);
            while ( 1 )
            {
              v23 = v81 + 72 * v152;
              v82 = 0x8E38E38E38E38E39LL * ((72 * v152) >> 3);
              if ( v82 < v150 )
                break;
              if ( !*(_BYTE *)(v23 + 68) )
              {
                v83 = 72LL * *(unsigned int *)(v23 + 64);
                v23 = v81 + v83;
                v82 = 0x8E38E38E38E38E39LL * (v83 >> 3);
                break;
              }
              v152 = *(unsigned int *)(v23 + 64);
            }
            v84 = *(_QWORD *)(v23 + 56);
            *(_DWORD *)&v177[8] = v82;
            *(_QWORD *)&v177[16] = &v177[32];
            v85 = (char *)v171;
            *(_QWORD *)v177 = v84;
            *(_QWORD *)&v177[24] = 0x400000000LL;
            goto LABEL_118;
          }
          v75 = *v74;
        }
        if ( v75 == 26 )
        {
          v78 = (char **)(v74 - 32);
        }
        else
        {
          if ( v75 != 27 )
            goto LABEL_95;
          v78 = (char **)(v74 - 64);
        }
        v74 = *v78;
        if ( !v74 )
        {
LABEL_95:
          if ( !(_BYTE)v157 )
            goto LABEL_97;
          goto LABEL_96;
        }
        v45[3].m128i_i64[1] = (__int64)v74;
        if ( (char *)v31 == v74 )
          goto LABEL_104;
      }
LABEL_98:
      if ( (char *)v31 == v74 )
      {
LABEL_104:
        if ( v71 != (char *)v31 )
        {
          v79 = (unsigned int)v160;
          v80 = (unsigned int)v160 + 1LL;
          if ( v80 > HIDWORD(v160) )
          {
            sub_C8D5F0((__int64)&v159, v161, v80, 4u, v24, v25);
            v79 = (unsigned int)v160;
          }
          v23 = (unsigned int)v156;
          v159[v79] = (unsigned int)v156;
          LODWORD(v160) = v160 + 1;
        }
        goto LABEL_45;
      }
      if ( v71 != v74 )
      {
        v23 = (__int64)v74;
        sub_103E7F0(v5, v74, (__int64)&v168, (unsigned int)v156);
      }
LABEL_45:
      v42 = v169;
      if ( !(_DWORD)v169 )
        goto LABEL_46;
    }
    v101 = *(_QWORD *)(v5 + 2368);
    v70 = 2 * v67;
    v102 = ((((((((((unsigned int)(v70 - 1) | ((unsigned __int64)(unsigned int)(v70 - 1) >> 1)) >> 2)
                | (unsigned int)(v70 - 1)
                | ((unsigned __int64)(unsigned int)(v70 - 1) >> 1)) >> 4)
              | (((unsigned int)(v70 - 1) | ((unsigned __int64)(unsigned int)(v70 - 1) >> 1)) >> 2)
              | (unsigned int)(v70 - 1)
              | ((unsigned __int64)(unsigned int)(v70 - 1) >> 1)) >> 8)
            | (((((unsigned int)(v70 - 1) | ((unsigned __int64)(unsigned int)(v70 - 1) >> 1)) >> 2)
              | (unsigned int)(v70 - 1)
              | ((unsigned __int64)(unsigned int)(v70 - 1) >> 1)) >> 4)
            | (((unsigned int)(v70 - 1) | ((unsigned __int64)(unsigned int)(v70 - 1) >> 1)) >> 2)
            | (unsigned int)(v70 - 1)
            | ((unsigned __int64)(unsigned int)(v70 - 1) >> 1)) >> 16)
          | (((((((unsigned int)(v70 - 1) | ((unsigned __int64)(unsigned int)(v70 - 1) >> 1)) >> 2)
              | (unsigned int)(v70 - 1)
              | ((unsigned __int64)(unsigned int)(v70 - 1) >> 1)) >> 4)
            | (((unsigned int)(v70 - 1) | ((unsigned __int64)(unsigned int)(v70 - 1) >> 1)) >> 2)
            | (unsigned int)(v70 - 1)
            | ((unsigned __int64)(unsigned int)(v70 - 1) >> 1)) >> 8)
          | (((((unsigned int)(v70 - 1) | ((unsigned __int64)(unsigned int)(v70 - 1) >> 1)) >> 2)
            | (unsigned int)(v70 - 1)
            | ((unsigned __int64)(unsigned int)(v70 - 1) >> 1)) >> 4)
          | (((unsigned int)(v70 - 1) | ((unsigned __int64)(unsigned int)(v70 - 1) >> 1)) >> 2)
          | (unsigned int)(v70 - 1)
          | ((unsigned __int64)(unsigned int)(v70 - 1) >> 1))
         + 1;
    if ( (unsigned int)v102 < 0x40 )
      LODWORD(v102) = 64;
    *(_DWORD *)(v5 + 2384) = v102;
    v103 = (_QWORD *)sub_C7D670(56LL * (unsigned int)v102, 8);
    *(_QWORD *)(v5 + 2368) = v103;
    if ( !v101 )
    {
      v139 = *(unsigned int *)(v5 + 2384);
      *(_QWORD *)(v5 + 2376) = 0;
      for ( j = &v103[7 * v139]; j != v103; v103 += 7 )
      {
        if ( v103 )
        {
          *v103 = -4096;
          v103[1] = -4096;
          v103[2] = -3;
          v103[3] = 0;
          v103[4] = 0;
          v103[5] = 0;
          v103[6] = 0;
        }
      }
      goto LABEL_161;
    }
    goto LABEL_160;
  }
LABEL_46:
  if ( (_DWORD)v160 )
  {
    v153 = (unsigned __int8 *)v31;
    v46 = v151;
    v174 = (__m128i *)v176;
    v154 = 0;
    v47 = v5;
    v48 = v159;
    v175 = 0x400000000LL;
    v156 = &v159[(unsigned int)v160];
    while ( 1 )
    {
LABEL_53:
      v49 = *v48;
      v50 = (const __m128i *)(*(_QWORD *)(v47 + 40) + 72 * v49);
      v51 = *(_DWORD **)(v47 + 32);
      if ( *v51 )
      {
        v52 = (char *)v50[3].m128i_i64[1];
        if ( v52 )
        {
          v155.m128i_i8[0] = 0;
          v157 = v49;
          while ( 1 )
          {
LABEL_56:
            v50[3].m128i_i64[1] = (__int64)v52;
            v53 = *v52;
            if ( *v52 == 27 )
            {
              if ( v52 == *(char **)(*(_QWORD *)v47 + 128LL)
                || (v54 = *(_DWORD **)(v47 + 32), v12 = *v54 == 1, --*v54, v12)
                || (v23 = (__int64)v50,
                    (unsigned __int8)sub_103AFA0(
                                       (__int64)v52,
                                       v50,
                                       *(unsigned __int8 **)(*(_QWORD *)(v47 + 24) + 56LL),
                                       *(_QWORD ***)(v47 + 16))) )
              {
                v56 = (unsigned int)v175;
                v57 = v46 & 0xFFFFFFFF00000000LL | v157;
                v58 = (unsigned int)v175 + 1LL;
                v46 = v57;
                if ( v58 > HIDWORD(v175) )
                {
                  v23 = (__int64)v176;
                  v157 = v57;
                  sub_C8D5F0((__int64)&v174, v176, v58, 0x10u, v24, v25);
                  v56 = (unsigned int)v175;
                  v57 = v157;
                }
                m128i_i64 = v174[v56].m128i_i64;
                ++v48;
                *m128i_i64 = (__int64)v52;
                m128i_i64[1] = v57;
                LODWORD(v175) = v175 + 1;
                if ( v156 != v48 )
                  goto LABEL_53;
LABEL_69:
                v5 = v47;
                v151 = v46;
                v60 = v153;
                v13 = (unsigned __int8 *)v154;
                if ( (_DWORD)v172 )
                {
                  if ( !v154 )
                  {
                    if ( v153 )
                    {
                      while ( 1 )
                      {
                        v126 = *v60;
                        if ( v126 == 26 )
                        {
                          v127 = v60 - 32;
                        }
                        else
                        {
                          if ( v126 != 27 )
                            break;
                          v127 = v60 - 64;
                        }
                        v128 = *(unsigned __int8 **)v127;
                        if ( !v128 )
                          break;
                        v60 = v128;
                      }
                    }
                    v13 = v60;
                  }
                  v91 = *((_QWORD *)v13 + 8);
                  v157 = (unsigned __int64)&v174;
                  v92 = v171[(unsigned int)v172].m128i_i8;
                  v93 = v171;
                  do
                  {
                    v23 = v91;
                    if ( (unsigned __int8)sub_B19720(*(_QWORD *)(v5 + 8), v91, *(_QWORD *)(v93->m128i_i64[0] + 64)) )
                    {
                      v94 = (unsigned int)v175;
                      v95 = _mm_loadu_si128(v93);
                      v96 = (unsigned int)v175 + 1LL;
                      if ( v96 > HIDWORD(v175) )
                      {
                        v23 = (__int64)v176;
                        v155 = v95;
                        sub_C8D5F0(v157, v176, v96, 0x10u, v24, v25);
                        v94 = (unsigned int)v175;
                        v95 = _mm_load_si128(&v155);
                      }
                      v174[v94] = v95;
                      LODWORD(v175) = v175 + 1;
                    }
                    ++v93;
                  }
                  while ( v92 != (char *)v93 );
                }
                v61 = v175;
                if ( !(_DWORD)v175 )
                {
                  v62 = *(unsigned int *)(v5 + 48);
                  v63 = v159;
                  LODWORD(v169) = 0;
                  v150 = v62;
                  v64 = &v159[(unsigned int)v160];
                  if ( v159 != v64 )
                  {
                    do
                    {
                      v65 = *v63;
                      v23 = (__int64)v13;
                      ++v63;
                      sub_103E7F0(v5, v13, (__int64)&v168, v65);
                    }
                    while ( v64 != v63 );
                  }
                  LODWORD(v160) = 0;
                  if ( v174 != (__m128i *)v176 )
                  {
                    _libc_free(v174, v23);
                    v26 = *((_QWORD *)v13 + 8);
                    v27 = *(_QWORD *)(v5 + 8);
                    if ( v26 )
                      goto LABEL_22;
LABEL_75:
                    v28 = 0;
                    v29 = 0;
                    goto LABEL_23;
                  }
                  goto LABEL_21;
                }
                v129 = v174;
                v130 = 2LL * (unsigned int)v175;
                v131 = v174[1].m128i_i64;
                v132 = v174[(unsigned __int64)v130 / 2].m128i_i64;
                if ( &v174[(unsigned __int64)v130 / 2] == &v174[1] )
                {
                  v133 = (__int64 *)v174;
                }
                else
                {
                  do
                  {
                    if ( !sub_1041420(*(_QWORD *)v5, *v131, v129->m128i_i64[0]) )
                      v129 = (__m128i *)v131;
                    v131 += 2;
                  }
                  while ( v132 != v131 );
                  v133 = (__int64 *)v174;
                  v61 = v175;
                  v130 = 2LL * (unsigned int)v175;
                  v131 = v174[(unsigned __int64)v130 / 2].m128i_i64;
                }
                if ( v129 != (__m128i *)(v131 - 2) )
                {
                  v134 = *(v131 - 2);
                  v135 = *((_DWORD *)v131 - 2);
                  *((__m128i *)v131 - 1) = _mm_loadu_si128(v129);
                  v129->m128i_i32[2] = v135;
                  v129->m128i_i64[0] = v134;
                  v133 = (__int64 *)v174;
                  v61 = v175;
                  v130 = 2LL * (unsigned int)v175;
                }
                v136 = (__int64)&v133[v130 - 2];
                v23 = 0x400000000LL;
                v137 = *(_QWORD *)v136;
                v138 = *(unsigned int *)(v136 + 8);
                LODWORD(v175) = v61 - 1;
                *(_QWORD *)&v177[16] = &v177[32];
                *(_QWORD *)v177 = v137;
                *(_DWORD *)&v177[8] = v138;
                *(_QWORD *)&v177[24] = 0x400000000LL;
                if ( v61 != 1 )
                {
                  v23 = (__int64)&v174;
                  sub_103AC50((__int64)&v177[16], (char **)&v174, v138, v137, v24, v25);
                  v133 = (__int64 *)v174;
                }
                if ( v133 != (__int64 *)v176 )
                  _libc_free(v133, v23);
LABEL_212:
                v85 = (char *)v171;
                goto LABEL_192;
              }
              v53 = *v52;
            }
            if ( v53 == 26 )
            {
              v55 = (char **)(v52 - 32);
            }
            else
            {
              if ( v53 != 27 )
                goto LABEL_49;
              v55 = (char **)(v52 - 64);
            }
            v52 = *v55;
            if ( !v52 )
            {
LABEL_49:
              if ( !v155.m128i_i8[0] )
                goto LABEL_51;
              goto LABEL_50;
            }
          }
        }
        v154 = 0;
      }
      else
      {
        *v51 = 1;
        v52 = (char *)v50[3].m128i_i64[1];
        v155.m128i_i8[0] = 1;
        if ( v52 )
        {
          v157 = v49;
          goto LABEL_56;
        }
LABEL_50:
        **(_DWORD **)(v47 + 32) = 0;
LABEL_51:
        v154 = v50[3].m128i_i64[1];
      }
      if ( v156 == ++v48 )
        goto LABEL_69;
    }
  }
  v116 = v171;
  v117 = v172;
  v118 = (unsigned int)v172;
  v119 = v171[1].m128i_i8;
  v120 = v171[v118].m128i_i8;
  if ( &v171[v118] == &v171[1] )
  {
    v85 = (char *)v171;
  }
  else
  {
    do
    {
      v23 = *(_QWORD *)v119;
      if ( !sub_1041420(*(_QWORD *)v5, *(_QWORD *)v119, v116->m128i_i64[0]) )
        v116 = (__m128i *)v119;
      v119 += 16;
    }
    while ( v120 != v119 );
    v85 = (char *)v171;
    v117 = v172;
    v118 = (unsigned int)v172;
    v119 = v171[v118].m128i_i8;
  }
  if ( v116 != (__m128i *)(v119 - 16) )
  {
    v121 = *((_QWORD *)v119 - 2);
    v122 = *((_DWORD *)v119 - 2);
    *((__m128i *)v119 - 1) = _mm_loadu_si128(v116);
    v116->m128i_i64[0] = v121;
    v85 = (char *)v171;
    v116->m128i_i32[2] = v122;
    v117 = v172;
    v118 = (unsigned int)v172;
  }
  v123 = (__int64)&v85[v118 * 16 - 16];
  v124 = (unsigned int)(v117 - 1);
  v125 = *(_QWORD *)v123;
  LODWORD(v123) = *(_DWORD *)(v123 + 8);
  LODWORD(v172) = v124;
  *(_QWORD *)&v177[16] = &v177[32];
  *(_DWORD *)&v177[8] = v123;
  *(_QWORD *)v177 = v125;
  *(_QWORD *)&v177[24] = 0x400000000LL;
  if ( (_DWORD)v124 )
  {
    v23 = (__int64)&v171;
    sub_103AC50((__int64)&v177[16], (char **)&v171, v124, v125, v24, v25);
    goto LABEL_212;
  }
LABEL_192:
  v155.m128i_i64[0] = v5 + 2360;
LABEL_118:
  if ( v85 != v173 )
    _libc_free(v85, v23);
  if ( v159 != (unsigned int *)v161 )
    _libc_free(v159, v23);
  if ( v168 != v170 )
    _libc_free(v168, v23);
  v86 = *(_DWORD *)(v5 + 2376);
  ++*(_QWORD *)(v5 + 2360);
  *(_DWORD *)(v5 + 48) = 0;
  if ( v86 )
  {
    v109 = 4 * v86;
    v23 = 64;
    v87 = *(unsigned int *)(v5 + 2384);
    if ( (unsigned int)(4 * v86) < 0x40 )
      v109 = 64;
    if ( (unsigned int)v87 <= v109 )
      goto LABEL_127;
    v110 = v86 - 1;
    if ( v110 )
    {
      _BitScanReverse(&v110, v110);
      v111 = 1 << (33 - (v110 ^ 0x1F));
      if ( v111 < 64 )
        v111 = 64;
      if ( v111 == (_DWORD)v87 )
        goto LABEL_183;
      v112 = ((unsigned __int64)(4 * v111 / 3u + 1) >> 1)
           | (4 * v111 / 3u + 1)
           | ((((unsigned __int64)(4 * v111 / 3u + 1) >> 1) | (4 * v111 / 3u + 1)) >> 2);
      v113 = (((v112 | (v112 >> 4)) >> 8)
            | v112
            | (v112 >> 4)
            | ((((v112 | (v112 >> 4)) >> 8) | v112 | (v112 >> 4)) >> 16))
           + 1;
      v114 = v113;
      v115 = 56 * v113;
    }
    else
    {
      v115 = 7168;
      v114 = 128;
    }
    sub_C7D6A0(*(_QWORD *)(v5 + 2368), 56 * v87, 8);
    *(_DWORD *)(v5 + 2384) = v114;
    v23 = 8;
    *(_QWORD *)(v5 + 2368) = sub_C7D670(v115, 8);
LABEL_183:
    sub_103F880(v155.m128i_i64[0]);
    goto LABEL_130;
  }
  if ( *(_DWORD *)(v5 + 2380) )
  {
    v87 = *(unsigned int *)(v5 + 2384);
    if ( (unsigned int)v87 <= 0x40 )
    {
LABEL_127:
      v88 = *(_QWORD **)(v5 + 2368);
      for ( k = &v88[7 * v87]; k != v88; *(v88 - 1) = 0 )
      {
        *v88 = -4096;
        v88 += 7;
        *(v88 - 6) = -4096;
        *(v88 - 5) = -3;
        *(v88 - 4) = 0;
        *(v88 - 3) = 0;
        *(v88 - 2) = 0;
      }
      *(_QWORD *)(v5 + 2376) = 0;
      goto LABEL_130;
    }
    v23 = 56 * v87;
    sub_C7D6A0(*(_QWORD *)(v5 + 2368), 56 * v87, 8);
    *(_QWORD *)(v5 + 2368) = 0;
    *(_QWORD *)(v5 + 2376) = 0;
    *(_DWORD *)(v5 + 2384) = 0;
  }
LABEL_130:
  v9 = *(_BYTE **)v177;
  if ( *(_BYTE **)&v177[16] != &v177[32] )
    _libc_free(*(_QWORD *)&v177[16], v23);
  return v9;
}
