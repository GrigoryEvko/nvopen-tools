// Function: sub_2E0C920
// Address: 0x2e0c920
//
void __fastcall sub_2E0C920(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 m,
        unsigned __int64 a4,
        unsigned int *a5,
        __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // r13
  unsigned __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // r15
  __int64 v13; // r11
  __int64 i; // rax
  __int64 v15; // rdx
  __int64 v16; // rdi
  unsigned int v17; // esi
  __int64 *v18; // rcx
  __int64 v19; // r9
  __int64 v20; // r15
  unsigned __int64 v21; // r15
  __int64 v22; // rsi
  unsigned int v23; // edi
  int v24; // eax
  __int64 v25; // r14
  __int64 v26; // rax
  unsigned int v27; // eax
  __int64 v28; // r12
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 v32; // r15
  __int64 v33; // r12
  int v34; // r13d
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  int v37; // eax
  __int64 v38; // rax
  __int64 v39; // r13
  __int64 v40; // rcx
  __int64 v41; // rdx
  _QWORD *v42; // rax
  _BYTE *v43; // r10
  _BYTE *v44; // rdi
  unsigned int **v45; // r12
  __int64 v46; // r11
  int v47; // eax
  unsigned __int64 v48; // r13
  unsigned __int64 v49; // r14
  const void *v50; // rbx
  __m128i v51; // xmm0
  const __m128i *v52; // rdx
  __int64 v53; // r15
  __int64 v54; // rax
  unsigned __int64 v55; // r11
  __m128i *v56; // rax
  const void *v57; // r11
  unsigned int **v58; // r13
  size_t v59; // rdx
  unsigned int v60; // r11d
  unsigned __int64 k; // r12
  unsigned int v62; // r13d
  int v63; // eax
  __int64 v64; // rdx
  unsigned int v65; // ebx
  __int64 v66; // rdx
  unsigned int *v67; // r15
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rdx
  _QWORD *v71; // rax
  unsigned __int64 v72; // rdx
  __int64 j; // rsi
  __int16 v74; // ax
  unsigned int v75; // esi
  __int64 v76; // rdi
  unsigned int v77; // ecx
  __int64 *v78; // rax
  __int64 v79; // r10
  unsigned __int64 v80; // r15
  __int64 *v81; // rax
  __int64 v82; // rsi
  unsigned int v83; // edi
  char v84; // al
  int v85; // ecx
  const void *v86; // rsi
  char v87; // al
  unsigned int *v88; // rax
  int v89; // eax
  int *v90; // rdi
  __int64 v91; // r8
  __int64 v92; // rdx
  __int64 v93; // rdx
  unsigned int **v94; // r12
  unsigned int **v95; // r15
  int v96; // eax
  unsigned __int64 v97; // r14
  const void *v98; // rbx
  __m128i v99; // xmm2
  const __m128i *v100; // r15
  __int64 v101; // r13
  __int64 v102; // rax
  unsigned __int64 v103; // rdx
  unsigned __int64 v104; // r10
  __m128i *v105; // rax
  const void *v106; // r15
  unsigned int **v107; // r13
  unsigned int v108; // r15d
  int *v109; // rdx
  unsigned int v110; // r12d
  int v111; // eax
  unsigned int v112; // r13d
  __int64 v113; // rdx
  __int64 v114; // r15
  __int64 v115; // rdx
  unsigned int *v116; // r14
  __int64 v117; // rax
  __int64 v118; // rbx
  __int64 v119; // rax
  unsigned __int64 v120; // rax
  __int64 v121; // rdx
  _QWORD *v122; // rax
  _QWORD *n; // rdx
  __int64 v124; // rax
  const void *v125; // rsi
  unsigned __int64 v126; // r15
  int v127; // r9d
  __int64 v128; // rax
  __int64 *v129; // [rsp+8h] [rbp-148h]
  int v130; // [rsp+1Ch] [rbp-134h]
  _BYTE *v131; // [rsp+20h] [rbp-130h]
  _BYTE *v132; // [rsp+20h] [rbp-130h]
  __int64 v133; // [rsp+28h] [rbp-128h]
  unsigned __int64 v134; // [rsp+28h] [rbp-128h]
  _BYTE *v135; // [rsp+28h] [rbp-128h]
  unsigned int *v136; // [rsp+28h] [rbp-128h]
  unsigned __int64 v137; // [rsp+38h] [rbp-118h]
  __int64 v138; // [rsp+40h] [rbp-110h]
  __int64 v139; // [rsp+40h] [rbp-110h]
  _BYTE *v140; // [rsp+40h] [rbp-110h]
  __int64 v141; // [rsp+40h] [rbp-110h]
  unsigned __int64 v142; // [rsp+48h] [rbp-108h]
  __int64 v144; // [rsp+60h] [rbp-F0h]
  __int64 v145; // [rsp+60h] [rbp-F0h]
  int v146; // [rsp+68h] [rbp-E8h]
  unsigned __int64 v147; // [rsp+68h] [rbp-E8h]
  __int64 v148; // [rsp+68h] [rbp-E8h]
  __int64 v149; // [rsp+68h] [rbp-E8h]
  __int64 v150; // [rsp+68h] [rbp-E8h]
  int v151; // [rsp+68h] [rbp-E8h]
  __int64 v152; // [rsp+68h] [rbp-E8h]
  _BYTE *v153; // [rsp+68h] [rbp-E8h]
  __int64 v154; // [rsp+68h] [rbp-E8h]
  unsigned int v155; // [rsp+68h] [rbp-E8h]
  _BYTE *v156; // [rsp+70h] [rbp-E0h] BYREF
  unsigned int v157; // [rsp+78h] [rbp-D8h]
  unsigned int v158; // [rsp+7Ch] [rbp-D4h]
  _BYTE v159[32]; // [rsp+80h] [rbp-D0h] BYREF
  _QWORD v160[2]; // [rsp+A0h] [rbp-B0h] BYREF
  _BYTE v161[32]; // [rsp+B0h] [rbp-A0h] BYREF
  int *v162; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v163; // [rsp+D8h] [rbp-78h]
  _BYTE v164[32]; // [rsp+E0h] [rbp-70h] BYREF
  int v165; // [rsp+100h] [rbp-50h]

  v6 = a2;
  v7 = *(unsigned int *)(a2 + 112);
  v137 = m;
  if ( (int)v7 < 0 )
  {
    v8 = *(_QWORD *)(*(_QWORD *)(a4 + 56) + 16 * (v7 & 0x7FFFFFFF) + 8);
  }
  else
  {
    m = *(_QWORD *)(a4 + 304);
    v8 = *(_QWORD *)(m + 8 * v7);
  }
  while ( v8 )
  {
    v9 = v8;
    v8 = *(_QWORD *)(v8 + 32);
    v10 = *(_QWORD *)(v9 + 16);
    v11 = *(_QWORD *)(*a1 + 32);
    if ( (unsigned __int16)(*(_WORD *)(v10 + 68) - 14) <= 1u )
    {
      v12 = *(_QWORD *)(v10 + 24);
      v13 = *(_QWORD *)(v12 + 56);
      if ( v10 == v13 )
      {
LABEL_16:
        v20 = *(_QWORD *)(*(_QWORD *)(v11 + 152) + 16LL * *(unsigned int *)(v12 + 24));
        goto LABEL_17;
      }
      while ( 1 )
      {
        v10 = *(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v10 )
          BUG();
        if ( (*(_QWORD *)v10 & 4) == 0 && (*(_BYTE *)(v10 + 44) & 4) != 0 )
        {
          for ( i = *(_QWORD *)v10; ; i = *(_QWORD *)v10 )
          {
            v10 = i & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v10 + 44) & 4) == 0 )
              break;
          }
        }
        v15 = *(unsigned int *)(v11 + 144);
        v16 = *(_QWORD *)(v11 + 128);
        if ( (_DWORD)v15 )
        {
          v17 = (v15 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v18 = (__int64 *)(v16 + 16LL * v17);
          v19 = *v18;
          if ( v10 == *v18 )
          {
LABEL_14:
            if ( v18 != (__int64 *)(v16 + 16 * v15) )
            {
              v20 = v18[1];
LABEL_17:
              v21 = v20 & 0xFFFFFFFFFFFFFFF8LL;
              m = sub_2E09D00((__int64 *)v6, v21);
              a4 = 3LL * *(unsigned int *)(v6 + 8);
              v22 = *(_QWORD *)v6 + 24LL * *(unsigned int *)(v6 + 8);
              if ( m != v22 )
              {
                v23 = *(_DWORD *)(v21 + 24);
                a4 = *(unsigned int *)((*(_QWORD *)m & 0xFFFFFFFFFFFFFFF8LL) + 24);
                if ( (unsigned __int64)((unsigned int)a4 | (*(__int64 *)m >> 1) & 3) <= v23
                  && v21 == (*(_QWORD *)(m + 8) & 0xFFFFFFFFFFFFFFF8LL) )
                {
                  a5 = (unsigned int *)(m + 24);
                  if ( v22 != m + 24 )
                  {
                    v124 = *(_QWORD *)(m + 24);
                    m += 24LL;
                    a4 = *(unsigned int *)((v124 & 0xFFFFFFFFFFFFFFF8LL) + 24);
                    goto LABEL_20;
                  }
                }
                else
                {
LABEL_20:
                  if ( (unsigned int)a4 <= v23 )
                  {
                    a5 = *(unsigned int **)(m + 16);
                    if ( ((*(_BYTE *)(m + 8) ^ 6) & 6) != 0 )
                      goto LABEL_22;
                  }
                }
              }
              goto LABEL_25;
            }
          }
          else
          {
            v85 = 1;
            while ( v19 != -4096 )
            {
              v17 = (v15 - 1) & (v85 + v17);
              v151 = v85 + 1;
              v18 = (__int64 *)(v16 + 16LL * v17);
              v19 = *v18;
              if ( v10 == *v18 )
                goto LABEL_14;
              v85 = v151;
            }
          }
        }
        if ( v13 == v10 )
          goto LABEL_16;
      }
    }
    v72 = *(_QWORD *)(v9 + 16);
    if ( (*(_DWORD *)(v10 + 44) & 4) != 0 )
    {
      do
        v72 = *(_QWORD *)v72 & 0xFFFFFFFFFFFFFFF8LL;
      while ( (*(_BYTE *)(v72 + 44) & 4) != 0 );
    }
    for ( ; (*(_BYTE *)(v10 + 44) & 8) != 0; v10 = *(_QWORD *)(v10 + 8) )
      ;
    for ( j = *(_QWORD *)(v10 + 8); j != v72; v72 = *(_QWORD *)(v72 + 8) )
    {
      v74 = *(_WORD *)(v72 + 68);
      if ( (unsigned __int16)(v74 - 14) > 4u && v74 != 24 )
        break;
    }
    v75 = *(_DWORD *)(v11 + 144);
    v76 = *(_QWORD *)(v11 + 128);
    if ( v75 )
    {
      v77 = (v75 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
      v78 = (__int64 *)(v76 + 16LL * v77);
      v79 = *v78;
      if ( v72 == *v78 )
        goto LABEL_101;
      v89 = 1;
      while ( v79 != -4096 )
      {
        v127 = v89 + 1;
        v128 = (v75 - 1) & (v77 + v89);
        v77 = v128;
        v78 = (__int64 *)(v76 + 16 * v128);
        v79 = *v78;
        if ( *v78 == v72 )
          goto LABEL_101;
        v89 = v127;
      }
    }
    v78 = (__int64 *)(v76 + 16LL * v75);
LABEL_101:
    v80 = v78[1] & 0xFFFFFFFFFFFFFFF8LL;
    v81 = (__int64 *)sub_2E09D00((__int64 *)v6, v80);
    a4 = 3LL * *(unsigned int *)(v6 + 8);
    m = *(_QWORD *)v6;
    v82 = *(_QWORD *)v6 + 24LL * *(unsigned int *)(v6 + 8);
    if ( v81 == (__int64 *)v82 )
    {
      v87 = *(_BYTE *)(v9 + 4);
      if ( (v87 & 1) != 0 || (v87 & 2) != 0 )
        continue;
      a5 = 0;
      v88 = 0;
      goto LABEL_126;
    }
    v83 = *(_DWORD *)(v80 + 24);
    a4 = *(unsigned int *)((*v81 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    if ( (unsigned __int64)((unsigned int)a4 | (*v81 >> 1) & 3) > v83 )
    {
      m = 0;
    }
    else
    {
      m = v81[2];
      if ( v80 == (v81[1] & 0xFFFFFFFFFFFFFFF8LL) )
      {
        a6 = (__int64)(v81 + 3);
        a5 = 0;
        if ( (__int64 *)v82 == v81 + 3 )
          goto LABEL_108;
        a4 = *(unsigned int *)((v81[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
        v81 += 3;
      }
      if ( v80 == *(_QWORD *)(m + 8) )
        m = 0;
    }
    a5 = 0;
    if ( v83 >= (unsigned int)a4 )
      a5 = (unsigned int *)v81[2];
LABEL_108:
    v84 = *(_BYTE *)(v9 + 4);
    if ( (v84 & 1) == 0 && (v84 & 2) == 0 )
    {
      v88 = a5;
      a5 = (unsigned int *)m;
LABEL_126:
      if ( (*(_BYTE *)(v9 + 3) & 0x10) == 0 || (*(_DWORD *)v9 & 0xFFF00) != 0 )
        goto LABEL_22;
      m = (unsigned __int64)a5;
      a5 = v88;
    }
    if ( a5 != (unsigned int *)m )
    {
LABEL_22:
      if ( a5 )
      {
        m = *a5;
        v24 = *(_DWORD *)(a1[1] + 4 * m);
        if ( v24 )
          sub_2EAB0C0(v9, *(unsigned int *)(*(_QWORD *)(v137 + 8LL * (unsigned int)(v24 - 1)) + 112LL));
      }
    }
LABEL_25:
    ;
  }
  if ( !*(_QWORD *)(v6 + 104) )
    goto LABEL_138;
  v25 = *(_QWORD *)(v6 + 104);
  v158 = 8;
  v26 = *a1;
  v146 = *((_DWORD *)a1 + 14);
  v156 = v159;
  v162 = (int *)v164;
  v129 = (__int64 *)(v26 + 56);
  v163 = 0x800000000LL;
  v130 = v146 - 1;
  v142 = (unsigned int)(v146 - 1);
  v27 = 8;
  while ( 2 )
  {
    v157 = 0;
    v28 = *(unsigned int *)(v25 + 72);
    if ( (unsigned int)v28 > v27 )
      sub_C8D5F0((__int64)&v156, v159, (unsigned int)v28, 4u, (__int64)a5, a6);
    LODWORD(v163) = 0;
    if ( v142 )
    {
      v29 = 0;
      if ( v142 > HIDWORD(v163) )
      {
        sub_C8D5F0((__int64)&v162, v164, v142, 8u, (__int64)a5, a6);
        v29 = 2LL * (unsigned int)v163;
      }
      memset(&v162[v29], 0, 8 * v142);
      LODWORD(v163) = v130 + v163;
    }
    if ( (_DWORD)v28 )
    {
      v30 = v6;
      v31 = 8 * v28;
      v32 = 0;
      v33 = v30;
      do
      {
        v38 = *(_QWORD *)(*(_QWORD *)(v25 + 64) + v32);
        v39 = *(_QWORD *)(v38 + 8);
        if ( (v39 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          goto LABEL_36;
        v147 = *(_QWORD *)(v38 + 8) & 0xFFFFFFFFFFFFFFF8LL;
        v40 = sub_2E09D00((__int64 *)v33, *(_QWORD *)(v38 + 8));
        if ( v40 == *(_QWORD *)v33 + 24LL * *(unsigned int *)(v33 + 8)
          || (*(_DWORD *)((*(_QWORD *)v40 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)v40 >> 1) & 3) > (*(_DWORD *)(v147 + 24) | (unsigned int)(v39 >> 1) & 3) )
        {
          BUG();
        }
        v34 = *(_DWORD *)(a1[1] + 4LL * **(unsigned int **)(v40 + 16));
        if ( v34 )
        {
          v41 = (unsigned int)(v34 - 1);
          if ( !*(_QWORD *)&v162[2 * v41] )
          {
            v138 = *(_QWORD *)(v25 + 112);
            v148 = *(_QWORD *)(v25 + 120);
            v133 = *(_QWORD *)(v137 + 8 * v41);
            v42 = (_QWORD *)sub_A777F0(0x80u, v129);
            if ( v42 )
            {
              a6 = v148;
              *v42 = v42 + 2;
              v42[1] = 0x200000000LL;
              v42[8] = v42 + 10;
              v42[9] = 0x200000000LL;
              v42[12] = 0;
              v42[13] = 0;
              v42[14] = v138;
              v42[15] = v148;
            }
            v42[13] = *(_QWORD *)(v133 + 104);
            *(_QWORD *)(v133 + 104) = v42;
            *(_QWORD *)&v162[2 * (v34 - 1)] = v42;
          }
        }
        else
        {
LABEL_36:
          v34 = 0;
        }
        v35 = v157;
        a4 = v158;
        v36 = v157 + 1LL;
        if ( v36 > v158 )
        {
          sub_C8D5F0((__int64)&v156, v159, v36, 4u, (__int64)a5, a6);
          v35 = v157;
        }
        m = (unsigned __int64)v156;
        v32 += 8;
        *(_DWORD *)&v156[4 * v35] = v34;
        v37 = ++v157;
      }
      while ( v31 != v32 );
      v6 = v33;
    }
    else
    {
      v37 = v157;
    }
    v43 = v161;
    v160[1] = 0x800000000LL;
    v44 = v161;
    v160[0] = v161;
    if ( v37 )
    {
      sub_2E093C0((__int64)v160, (__int64)&v156, m, a4, (__int64)a5, (__int64)v160);
      v44 = (_BYTE *)v160[0];
      v43 = v161;
    }
    m = *(_QWORD *)v25;
    a6 = (__int64)v162;
    v45 = *(unsigned int ***)v25;
    v46 = *(_QWORD *)v25 + 24LL * *(unsigned int *)(v25 + 8);
    if ( *(_QWORD *)v25 == v46 )
      goto LABEL_115;
    while ( 1 )
    {
      v47 = *(_DWORD *)&v44[4 * *v45[2]];
      if ( v47 )
        break;
      v45 += 3;
      if ( (unsigned int **)v46 == v45 )
        goto LABEL_115;
    }
    if ( v45 == (unsigned int **)v46 )
    {
LABEL_115:
      v58 = v45;
      goto LABEL_64;
    }
    v149 = v25;
    v48 = (unsigned __int64)v45;
    v49 = (unsigned __int64)v162;
    v139 = v6;
    v50 = (const void *)v46;
    while ( !v47 )
    {
      v51 = _mm_loadu_si128((const __m128i *)v48);
      v48 += 24LL;
      v45 += 3;
      *(__m128i *)(v45 - 3) = v51;
      *(v45 - 1) = *(unsigned int **)(v48 - 8);
      if ( v50 == (const void *)v48 )
        goto LABEL_62;
LABEL_58:
      v47 = *(_DWORD *)(v160[0] + 4LL * **(unsigned int **)(v48 + 16));
    }
    v52 = (const __m128i *)v48;
    v53 = *(_QWORD *)(v49 + 8LL * (unsigned int)(v47 - 1));
    v54 = *(unsigned int *)(v53 + 8);
    a4 = *(_QWORD *)v53;
    v55 = v54 + 1;
    if ( v54 + 1 > (unsigned __int64)*(unsigned int *)(v53 + 12) )
    {
      v86 = (const void *)(v53 + 16);
      if ( a4 > v48 || a4 + 24 * v54 <= v48 )
      {
        v135 = v43;
        sub_C8D5F0(v53, v86, v55, 0x18u, (__int64)a5, a6);
        a4 = *(_QWORD *)v53;
        v54 = *(unsigned int *)(v53 + 8);
        v52 = (const __m128i *)v48;
        v43 = v135;
      }
      else
      {
        v131 = v43;
        v134 = v48 - a4;
        sub_C8D5F0(v53, v86, v55, 0x18u, (__int64)a5, a6);
        a4 = *(_QWORD *)v53;
        v43 = v131;
        v52 = (const __m128i *)(*(_QWORD *)v53 + v134);
        v54 = *(unsigned int *)(v53 + 8);
      }
    }
    v48 += 24LL;
    v56 = (__m128i *)(a4 + 24 * v54);
    *v56 = _mm_loadu_si128(v52);
    v56[1].m128i_i64[0] = v52[1].m128i_i64[0];
    ++*(_DWORD *)(v53 + 8);
    if ( v50 != (const void *)v48 )
      goto LABEL_58;
LABEL_62:
    a6 = v49;
    v25 = v149;
    v57 = v50;
    v6 = v139;
    m = *(_QWORD *)v149;
    v58 = (unsigned int **)((char *)v45 + *(_QWORD *)v149 + 24LL * *(unsigned int *)(v149 + 8) - (_QWORD)v57);
    if ( v57 == (const void *)(*(_QWORD *)v149 + 24LL * *(unsigned int *)(v149 + 8)) )
    {
      v44 = (_BYTE *)v160[0];
    }
    else
    {
      v59 = *(_QWORD *)v149 + 24LL * *(unsigned int *)(v149 + 8) - (_QWORD)v57;
      v140 = v43;
      v150 = a6;
      memmove(v45, v57, v59);
      v44 = (_BYTE *)v160[0];
      m = *(_QWORD *)v25;
      v43 = v140;
      a6 = v150;
    }
LABEL_64:
    v60 = *(_DWORD *)(v25 + 72);
    *(_DWORD *)(v25 + 8) = -1431655765 * ((__int64)((__int64)v58 - m) >> 3);
    if ( v60 )
    {
      m = v60;
      for ( k = 0; k != v60; v62 = ++k )
      {
        v63 = *(_DWORD *)&v44[4 * k];
        v62 = k;
        if ( v63 )
        {
          if ( (_DWORD)k == v60 )
            goto LABEL_77;
          a5 = (unsigned int *)v6;
          v64 = (unsigned int)k;
          v65 = v60;
          while ( 1 )
          {
            a4 = *(_QWORD *)(v25 + 64);
            v67 = *(unsigned int **)(a4 + 8 * v64);
            if ( v63 )
            {
              v68 = a6 + 8LL * (unsigned int)(v63 - 1);
              *v67 = *(_DWORD *)(*(_QWORD *)v68 + 72LL);
              v69 = *(_QWORD *)v68;
              v70 = *(unsigned int *)(v69 + 72);
              if ( v70 + 1 > (unsigned __int64)*(unsigned int *)(v69 + 76) )
              {
                v132 = v43;
                v136 = a5;
                v141 = a6;
                v152 = v69;
                sub_C8D5F0(v69 + 64, (const void *)(v69 + 80), v70 + 1, 8u, (__int64)a5, a6);
                v69 = v152;
                v43 = v132;
                a5 = v136;
                a6 = v141;
                v70 = *(unsigned int *)(v152 + 72);
              }
              a4 = *(_QWORD *)(v69 + 64);
              LODWORD(k) = k + 1;
              *(_QWORD *)(a4 + 8 * v70) = v67;
              ++*(_DWORD *)(v69 + 72);
              if ( v65 == (_DWORD)k )
              {
LABEL_76:
                m = *(unsigned int *)(v25 + 72);
                v6 = (__int64)a5;
                k = v62;
                goto LABEL_77;
              }
            }
            else
            {
              *v67 = v62;
              v66 = v62;
              LODWORD(k) = k + 1;
              ++v62;
              *(_QWORD *)(*(_QWORD *)(v25 + 64) + 8 * v66) = v67;
              if ( v65 == (_DWORD)k )
                goto LABEL_76;
            }
            v64 = (unsigned int)k;
            v63 = *(_DWORD *)(v160[0] + 4LL * (unsigned int)k);
          }
        }
      }
      k = v62;
LABEL_77:
      if ( m != k )
      {
        if ( m <= k )
        {
          if ( *(unsigned int *)(v25 + 76) < k )
          {
            v153 = v43;
            sub_C8D5F0(v25 + 64, (const void *)(v25 + 80), k, 8u, (__int64)a5, a6);
            m = *(unsigned int *)(v25 + 72);
            v43 = v153;
          }
          a4 = *(_QWORD *)(v25 + 64);
          v71 = (_QWORD *)(a4 + 8 * m);
          for ( m = a4 + 8 * k; (_QWORD *)m != v71; ++v71 )
          {
            if ( v71 )
              *v71 = 0;
          }
        }
        *(_DWORD *)(v25 + 72) = v62;
      }
      v44 = (_BYTE *)v160[0];
    }
    if ( v44 != v43 )
      _libc_free((unsigned __int64)v44);
    v25 = *(_QWORD *)(v25 + 104);
    if ( v25 )
    {
      v27 = v158;
      continue;
    }
    break;
  }
  sub_2E0AF60(v6);
  if ( v162 != (int *)v164 )
    _libc_free((unsigned __int64)v162);
  if ( v156 != v159 )
    _libc_free((unsigned __int64)v156);
LABEL_138:
  v90 = (int *)v164;
  v91 = (__int64)&v162;
  v163 = 0x800000000LL;
  v162 = (int *)v164;
  v92 = *((unsigned int *)a1 + 4);
  if ( (_DWORD)v92 )
  {
    sub_2E093C0((__int64)&v162, (__int64)(a1 + 1), v92, a4, (__int64)&v162, a6);
    v90 = v162;
  }
  v93 = *(_QWORD *)v6;
  v94 = (unsigned int **)v93;
  v165 = *((_DWORD *)a1 + 14);
  v95 = (unsigned int **)(v93 + 24LL * *(unsigned int *)(v6 + 8));
  if ( (unsigned int **)v93 == v95 )
    goto LABEL_182;
  while ( 1 )
  {
    v96 = v90[*v94[2]];
    if ( v96 )
      break;
    v94 += 3;
    if ( v95 == v94 )
      goto LABEL_182;
  }
  if ( v95 == v94 )
  {
LABEL_182:
    v107 = v94;
    goto LABEL_153;
  }
  v154 = v6;
  a6 = v137;
  v97 = (unsigned __int64)v94;
  v98 = (const void *)(v93 + 24LL * *(unsigned int *)(v6 + 8));
  while ( 2 )
  {
    if ( !v96 )
    {
      v99 = _mm_loadu_si128((const __m128i *)v97);
      v97 += 24LL;
      v94 += 3;
      *(__m128i *)(v94 - 3) = v99;
      *(v94 - 1) = *(unsigned int **)(v97 - 8);
      if ( v98 == (const void *)v97 )
        break;
      goto LABEL_147;
    }
    v100 = (const __m128i *)v97;
    v101 = *(_QWORD *)(a6 + 8LL * (unsigned int)(v96 - 1));
    v102 = *(unsigned int *)(v101 + 8);
    v103 = *(_QWORD *)v101;
    v104 = v102 + 1;
    if ( v102 + 1 > (unsigned __int64)*(unsigned int *)(v101 + 12) )
    {
      v145 = a6;
      v125 = (const void *)(v101 + 16);
      if ( v103 > v97 || v97 >= v103 + 24 * v102 )
      {
        v100 = (const __m128i *)v97;
        sub_C8D5F0(v101, v125, v104, 0x18u, v91, a6);
        v103 = *(_QWORD *)v101;
        v102 = *(unsigned int *)(v101 + 8);
        a6 = v145;
      }
      else
      {
        v126 = v97 - v103;
        sub_C8D5F0(v101, v125, v104, 0x18u, v91, a6);
        v103 = *(_QWORD *)v101;
        v102 = *(unsigned int *)(v101 + 8);
        a6 = v145;
        v100 = (const __m128i *)(*(_QWORD *)v101 + v126);
      }
    }
    v97 += 24LL;
    v105 = (__m128i *)(v103 + 24 * v102);
    *v105 = _mm_loadu_si128(v100);
    v105[1].m128i_i64[0] = v100[1].m128i_i64[0];
    ++*(_DWORD *)(v101 + 8);
    if ( v98 != (const void *)v97 )
    {
LABEL_147:
      v96 = v162[**(unsigned int **)(v97 + 16)];
      continue;
    }
    break;
  }
  v106 = v98;
  v6 = v154;
  v93 = *(_QWORD *)v154;
  v91 = *(_QWORD *)v154 + 24LL * *(unsigned int *)(v154 + 8) - (_QWORD)v106;
  v107 = (unsigned int **)((char *)v94 + v91);
  if ( v106 == (const void *)(*(_QWORD *)v154 + 24LL * *(unsigned int *)(v154 + 8)) )
  {
    v90 = v162;
  }
  else
  {
    memmove(v94, v106, *(_QWORD *)v154 + 24LL * *(unsigned int *)(v154 + 8) - (_QWORD)v106);
    v90 = v162;
    v93 = *(_QWORD *)v154;
  }
LABEL_153:
  v108 = *(_DWORD *)(v6 + 72);
  *(_DWORD *)(v6 + 8) = -1431655765 * (((__int64)v107 - v93) >> 3);
  if ( v108 )
  {
    v109 = v90;
    v110 = 0;
    while ( 1 )
    {
      v111 = *v109;
      if ( *v109 )
        break;
      ++v110;
      ++v109;
      if ( v108 == v110 )
      {
        v120 = v108;
LABEL_167:
        if ( v108 != v120 )
        {
          if ( v108 >= v120 )
          {
            if ( v108 > (unsigned __int64)*(unsigned int *)(v6 + 76) )
            {
              sub_C8D5F0(v6 + 64, (const void *)(v6 + 80), v108, 8u, v91, a6);
              v120 = *(unsigned int *)(v6 + 72);
            }
            v121 = *(_QWORD *)(v6 + 64);
            v122 = (_QWORD *)(v121 + 8 * v120);
            for ( n = (_QWORD *)(v121 + 8LL * v108); n != v122; ++v122 )
            {
              if ( v122 )
                *v122 = 0;
            }
          }
          *(_DWORD *)(v6 + 72) = v108;
        }
        v90 = v162;
        goto LABEL_177;
      }
    }
    if ( v108 != v110 )
    {
      v91 = v108;
      a6 = v137;
      v112 = v110;
      v113 = v110;
      v114 = v6;
      while ( 1 )
      {
        v116 = *(unsigned int **)(*(_QWORD *)(v114 + 64) + 8 * v113);
        if ( v111 )
        {
          v117 = a6 + 8LL * (unsigned int)(v111 - 1);
          *v116 = *(_DWORD *)(*(_QWORD *)v117 + 72LL);
          v118 = *(_QWORD *)v117;
          v119 = *(unsigned int *)(*(_QWORD *)v117 + 72LL);
          if ( v119 + 1 > (unsigned __int64)*(unsigned int *)(v118 + 76) )
          {
            v144 = a6;
            v155 = v91;
            sub_C8D5F0(v118 + 64, (const void *)(v118 + 80), v119 + 1, 8u, v91, a6);
            v119 = *(unsigned int *)(v118 + 72);
            a6 = v144;
            v91 = v155;
          }
          ++v112;
          *(_QWORD *)(*(_QWORD *)(v118 + 64) + 8 * v119) = v116;
          ++*(_DWORD *)(v118 + 72);
          if ( (_DWORD)v91 == v112 )
          {
LABEL_165:
            v120 = *(unsigned int *)(v114 + 72);
            v6 = v114;
            v108 = v110;
            goto LABEL_167;
          }
        }
        else
        {
          *v116 = v110;
          v115 = v110;
          ++v112;
          ++v110;
          *(_QWORD *)(*(_QWORD *)(v114 + 64) + 8 * v115) = v116;
          if ( (_DWORD)v91 == v112 )
            goto LABEL_165;
        }
        v113 = v112;
        v111 = v162[v112];
      }
    }
  }
LABEL_177:
  if ( v90 != (int *)v164 )
    _libc_free((unsigned __int64)v90);
}
