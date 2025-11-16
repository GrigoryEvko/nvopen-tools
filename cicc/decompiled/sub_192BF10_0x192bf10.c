// Function: sub_192BF10
// Address: 0x192bf10
//
__int64 __fastcall sub_192BF10(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rsi
  _BYTE *v12; // rsi
  _QWORD *v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rcx
  char v19; // si
  double v20; // xmm4_8
  double v21; // xmm5_8
  __int64 v22; // rax
  __int64 v23; // rcx
  unsigned __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rax
  char v29; // si
  __int64 v30; // rcx
  unsigned int v31; // esi
  __int64 v32; // rbx
  __int64 v33; // rdi
  unsigned int v34; // ecx
  __int64 *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // r12
  int v38; // r13d
  __int64 v39; // r10
  unsigned __int64 v40; // rcx
  unsigned int v41; // edi
  __int64 *v42; // rax
  __int64 v43; // rsi
  unsigned int v44; // edx
  __int64 v45; // rbx
  __int64 *v46; // r15
  unsigned __int64 v47; // rcx
  unsigned __int64 v48; // rax
  _QWORD *v49; // rax
  _QWORD *v50; // rcx
  __int64 v51; // rax
  __int64 *v52; // r9
  _QWORD *k; // rax
  __int64 *m; // rax
  __int64 v55; // r11
  int v56; // esi
  int v57; // esi
  __int64 v58; // r8
  unsigned int v59; // ecx
  __int64 *v60; // rdx
  __int64 v61; // rdi
  int v62; // edx
  int v63; // edi
  int v64; // esi
  unsigned int v65; // edx
  __int64 v66; // r8
  int v67; // r11d
  __int64 *v68; // r8
  int v69; // edi
  __int64 v70; // rax
  _QWORD *v71; // rax
  unsigned int v72; // r9d
  _QWORD *v73; // rcx
  __int64 v74; // rax
  __int64 *v75; // rdi
  _QWORD *i; // rax
  __int64 *v77; // rax
  int v78; // esi
  int v79; // esi
  __int64 v80; // r10
  unsigned int v81; // ecx
  __int64 *v82; // rdx
  __int64 v83; // r8
  int v84; // edx
  int v85; // edx
  int v86; // r11d
  __int64 v87; // rax
  __int64 *v88; // r9
  unsigned int v89; // r8d
  __int64 v90; // rsi
  __int64 v91; // rsi
  char v92; // al
  char v93; // r8
  bool v94; // al
  int v95; // ebx
  unsigned int ii; // r12d
  unsigned __int64 v97; // rax
  __int64 v98; // rcx
  int v99; // r8d
  int v100; // r9d
  __int64 v101; // rsi
  _QWORD *n; // rsi
  int v103; // r10d
  __int64 *v104; // r9
  int v105; // edi
  int v106; // ecx
  __int64 v107; // rsi
  _QWORD *j; // rsi
  int v109; // eax
  int v110; // edi
  __int64 v111; // rsi
  unsigned int v112; // edx
  __int64 v113; // r8
  int v114; // r10d
  __int64 *v115; // r9
  int v116; // eax
  int v117; // edx
  __int64 v118; // rdi
  __int64 *v119; // r8
  unsigned int v120; // r12d
  int v121; // r9d
  __int64 v122; // rsi
  int v123; // r15d
  __int64 v125; // [rsp+8h] [rbp-268h]
  __int64 v127; // [rsp+18h] [rbp-258h]
  int v128; // [rsp+18h] [rbp-258h]
  __int64 v129; // [rsp+20h] [rbp-250h]
  int v130; // [rsp+20h] [rbp-250h]
  __int64 *v131; // [rsp+20h] [rbp-250h]
  unsigned int v132; // [rsp+28h] [rbp-248h]
  __int64 v133; // [rsp+28h] [rbp-248h]
  unsigned int v134; // [rsp+28h] [rbp-248h]
  __int64 v135; // [rsp+28h] [rbp-248h]
  __int64 *v136; // [rsp+28h] [rbp-248h]
  int v137; // [rsp+3Ch] [rbp-234h]
  _QWORD v138[2]; // [rsp+40h] [rbp-230h] BYREF
  unsigned __int64 v139; // [rsp+50h] [rbp-220h]
  _BYTE v140[64]; // [rsp+68h] [rbp-208h] BYREF
  __int64 v141; // [rsp+A8h] [rbp-1C8h]
  __int64 v142; // [rsp+B0h] [rbp-1C0h]
  unsigned __int64 v143; // [rsp+B8h] [rbp-1B8h]
  _QWORD v144[2]; // [rsp+C0h] [rbp-1B0h] BYREF
  unsigned __int64 v145; // [rsp+D0h] [rbp-1A0h]
  _BYTE v146[64]; // [rsp+E8h] [rbp-188h] BYREF
  __int64 v147; // [rsp+128h] [rbp-148h]
  __int64 v148; // [rsp+130h] [rbp-140h]
  unsigned __int64 v149; // [rsp+138h] [rbp-138h]
  _QWORD v150[2]; // [rsp+140h] [rbp-130h] BYREF
  unsigned __int64 v151; // [rsp+150h] [rbp-120h]
  __int64 v152; // [rsp+1A8h] [rbp-C8h]
  __int64 v153; // [rsp+1B0h] [rbp-C0h]
  __int64 v154; // [rsp+1B8h] [rbp-B8h]
  char v155[8]; // [rsp+1C0h] [rbp-B0h] BYREF
  __int64 v156; // [rsp+1C8h] [rbp-A8h]
  unsigned __int64 v157; // [rsp+1D0h] [rbp-A0h]
  __int64 v158; // [rsp+228h] [rbp-48h]
  __int64 v159; // [rsp+230h] [rbp-40h]
  __int64 v160; // [rsp+238h] [rbp-38h]

  *(_DWORD *)(a1 + 632) = *(_QWORD *)(a2 + 96);
  *(_QWORD *)(a1 + 200) = *(_QWORD *)(a1 + 216);
  *(_QWORD *)(a1 + 184) = *(_QWORD *)(a1 + 232);
  *(_QWORD *)(a1 + 192) = *(_QWORD *)(a1 + 240);
  v11 = *(_QWORD *)(a2 + 80);
  if ( v11 )
    v11 -= 24;
  sub_19235E0(v150, v11);
  v12 = v140;
  v13 = v138;
  sub_16CCCB0(v138, (__int64)v140, (__int64)v150);
  v14 = v153;
  v15 = v152;
  v141 = 0;
  v142 = 0;
  v143 = 0;
  v16 = v153 - v152;
  if ( v153 == v152 )
  {
    v17 = 0;
  }
  else
  {
    if ( v16 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_205;
    v17 = sub_22077B0(v153 - v152);
    v14 = v153;
    v15 = v152;
  }
  v141 = v17;
  v142 = v17;
  v143 = v17 + v16;
  if ( v15 == v14 )
  {
    v18 = v17;
  }
  else
  {
    v18 = v17 + v14 - v15;
    do
    {
      if ( v17 )
      {
        *(_QWORD *)v17 = *(_QWORD *)v15;
        v19 = *(_BYTE *)(v15 + 24);
        *(_BYTE *)(v17 + 24) = v19;
        if ( v19 )
        {
          a3 = (__m128)_mm_loadu_si128((const __m128i *)(v15 + 8));
          *(__m128 *)(v17 + 8) = a3;
        }
      }
      v17 += 32;
      v15 += 32;
    }
    while ( v18 != v17 );
  }
  v13 = v144;
  v142 = v18;
  v12 = v146;
  sub_16CCCB0(v144, (__int64)v146, (__int64)v155);
  v22 = v159;
  v23 = v158;
  v147 = 0;
  v148 = 0;
  v149 = 0;
  v24 = v159 - v158;
  if ( v159 == v158 )
  {
    v26 = 0;
    goto LABEL_15;
  }
  if ( v24 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_205:
    sub_4261EA(v13, v12, v15);
  v25 = sub_22077B0(v159 - v158);
  v23 = v158;
  v26 = v25;
  v22 = v159;
LABEL_15:
  v147 = v26;
  v148 = v26;
  v149 = v26 + v24;
  if ( v23 == v22 )
  {
    v28 = v26;
  }
  else
  {
    v27 = v26;
    v28 = v26 + v22 - v23;
    do
    {
      if ( v27 )
      {
        *(_QWORD *)v27 = *(_QWORD *)v23;
        v29 = *(_BYTE *)(v23 + 24);
        *(_BYTE *)(v27 + 24) = v29;
        if ( v29 )
        {
          a4 = (__m128)_mm_loadu_si128((const __m128i *)(v23 + 8));
          *(__m128 *)(v27 + 8) = a4;
        }
      }
      v27 += 32;
      v23 += 32;
    }
    while ( v27 != v28 );
  }
  v148 = v28;
  v137 = 0;
  v125 = a1 + 264;
  while ( 1 )
  {
    v30 = v141;
    if ( v142 - v141 != v28 - v26 )
      goto LABEL_23;
    if ( v141 == v142 )
      break;
    v91 = v26;
    while ( *(_QWORD *)v30 == *(_QWORD *)v91 )
    {
      v92 = *(_BYTE *)(v30 + 24);
      v93 = *(_BYTE *)(v91 + 24);
      if ( v92 && v93 )
        v94 = *(_DWORD *)(v30 + 16) == *(_DWORD *)(v91 + 16);
      else
        v94 = v92 == v93;
      if ( !v94 )
        break;
      v30 += 32;
      v91 += 32;
      if ( v142 == v30 )
        goto LABEL_89;
    }
LABEL_23:
    v31 = *(_DWORD *)(a1 + 288);
    ++v137;
    v32 = *(_QWORD *)(v142 - 32);
    if ( v31 )
    {
      v33 = *(_QWORD *)(a1 + 272);
      v34 = (v31 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
      v35 = (__int64 *)(v33 + 16LL * v34);
      v36 = *v35;
      if ( v32 == *v35 )
        goto LABEL_25;
      v103 = 1;
      v104 = 0;
      while ( v36 != -8 )
      {
        if ( v36 == -16 && !v104 )
          v104 = v35;
        v34 = (v31 - 1) & (v103 + v34);
        v35 = (__int64 *)(v33 + 16LL * v34);
        v36 = *v35;
        if ( v32 == *v35 )
          goto LABEL_25;
        ++v103;
      }
      v105 = *(_DWORD *)(a1 + 280);
      if ( v104 )
        v35 = v104;
      ++*(_QWORD *)(a1 + 264);
      v106 = v105 + 1;
      if ( 4 * (v105 + 1) < 3 * v31 )
      {
        if ( v31 - *(_DWORD *)(a1 + 284) - v106 <= v31 >> 3 )
        {
          sub_1542080(v125, v31);
          v116 = *(_DWORD *)(a1 + 288);
          if ( !v116 )
          {
LABEL_207:
            ++*(_DWORD *)(a1 + 280);
            BUG();
          }
          v117 = v116 - 1;
          v118 = *(_QWORD *)(a1 + 272);
          v119 = 0;
          v120 = (v116 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
          v121 = 1;
          v106 = *(_DWORD *)(a1 + 280) + 1;
          v35 = (__int64 *)(v118 + 16LL * v120);
          v122 = *v35;
          if ( v32 != *v35 )
          {
            while ( v122 != -8 )
            {
              if ( !v119 && v122 == -16 )
                v119 = v35;
              v120 = v117 & (v121 + v120);
              v35 = (__int64 *)(v118 + 16LL * v120);
              v122 = *v35;
              if ( v32 == *v35 )
                goto LABEL_123;
              ++v121;
            }
            if ( v119 )
              v35 = v119;
          }
        }
        goto LABEL_123;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 264);
    }
    sub_1542080(v125, 2 * v31);
    v109 = *(_DWORD *)(a1 + 288);
    if ( !v109 )
      goto LABEL_207;
    v110 = v109 - 1;
    v111 = *(_QWORD *)(a1 + 272);
    v112 = (v109 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
    v106 = *(_DWORD *)(a1 + 280) + 1;
    v35 = (__int64 *)(v111 + 16LL * v112);
    v113 = *v35;
    if ( v32 != *v35 )
    {
      v114 = 1;
      v115 = 0;
      while ( v113 != -8 )
      {
        if ( v113 == -16 && !v115 )
          v115 = v35;
        v112 = v110 & (v114 + v112);
        v35 = (__int64 *)(v111 + 16LL * v112);
        v113 = *v35;
        if ( v32 == *v35 )
          goto LABEL_123;
        ++v114;
      }
      if ( v115 )
        v35 = v115;
    }
LABEL_123:
    *(_DWORD *)(a1 + 280) = v106;
    if ( *v35 != -8 )
      --*(_DWORD *)(a1 + 284);
    *v35 = v32;
    *((_DWORD *)v35 + 2) = 0;
LABEL_25:
    *((_DWORD *)v35 + 2) = v137;
    if ( v32 + 40 != *(_QWORD *)(v32 + 48) )
    {
      v37 = *(_QWORD *)(v32 + 48);
      v38 = 0;
      v39 = v32 + 40;
      while ( 1 )
      {
        v44 = *(_DWORD *)(a1 + 288);
        v45 = v37 - 24;
        if ( !v37 )
          v45 = 0;
        v46 = *(__int64 **)(a1 + 272);
        ++v38;
        if ( !v44 )
          break;
        v40 = v44 - 1;
        v41 = v40 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
        v42 = &v46[2 * v41];
        v43 = *v42;
        if ( v45 == *v42 )
        {
LABEL_28:
          *((_DWORD *)v42 + 2) = v38;
          v37 = *(_QWORD *)(v37 + 8);
          if ( v39 == v37 )
            goto LABEL_53;
        }
        else
        {
          v67 = 1;
          v68 = 0;
          while ( v43 != -8 )
          {
            if ( v43 == -16 && !v68 )
              v68 = v42;
            v41 = v40 & (v67 + v41);
            v42 = &v46[2 * v41];
            v43 = *v42;
            if ( v45 == *v42 )
              goto LABEL_28;
            ++v67;
          }
          v69 = *(_DWORD *)(a1 + 280);
          if ( v68 )
            v42 = v68;
          ++*(_QWORD *)(a1 + 264);
          v63 = v69 + 1;
          if ( 4 * v63 >= 3 * v44 )
            goto LABEL_33;
          if ( v44 - *(_DWORD *)(a1 + 284) - v63 <= v44 >> 3 )
          {
            v127 = v39;
            v134 = v44;
            v70 = ((((((((v40 >> 1) | v40 | (((v40 >> 1) | v40) >> 2)) >> 4)
                     | (v40 >> 1)
                     | v40
                     | (((v40 >> 1) | v40) >> 2)) >> 8)
                   | (((v40 >> 1) | v40 | (((v40 >> 1) | v40) >> 2)) >> 4)
                   | (v40 >> 1)
                   | v40
                   | (((v40 >> 1) | v40) >> 2)) >> 16)
                 | (((((v40 >> 1) | v40 | (((v40 >> 1) | v40) >> 2)) >> 4) | (v40 >> 1)
                                                                           | v40
                                                                           | (((v40 >> 1) | v40) >> 2)) >> 8)
                 | (((v40 >> 1) | v40 | (((v40 >> 1) | v40) >> 2)) >> 4)
                 | (v40 >> 1)
                 | v40
                 | (((v40 >> 1) | v40) >> 2))
                + 1;
            if ( (unsigned int)v70 < 0x40 )
              LODWORD(v70) = 64;
            *(_DWORD *)(a1 + 288) = v70;
            v71 = (_QWORD *)sub_22077B0(16LL * (unsigned int)v70);
            v72 = ((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4);
            *(_QWORD *)(a1 + 272) = v71;
            v39 = v127;
            v73 = v71;
            if ( v46 )
            {
              v74 = *(unsigned int *)(a1 + 288);
              *(_QWORD *)(a1 + 280) = 0;
              v75 = &v46[2 * v134];
              for ( i = &v73[2 * v74]; i != v73; v73 += 2 )
              {
                if ( v73 )
                  *v73 = -8;
              }
              v135 = v127;
              v77 = v46;
              do
              {
                v55 = *v77;
                if ( *v77 != -16 && v55 != -8 )
                {
                  v78 = *(_DWORD *)(a1 + 288);
                  if ( !v78 )
                  {
LABEL_208:
                    MEMORY[0] = v55;
                    BUG();
                  }
                  v79 = v78 - 1;
                  v80 = *(_QWORD *)(a1 + 272);
                  v81 = v79 & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
                  v82 = (__int64 *)(v80 + 16LL * v81);
                  v83 = *v82;
                  if ( *v82 != v55 )
                  {
                    v128 = 1;
                    v131 = 0;
                    while ( v83 != -8 )
                    {
                      if ( !v131 )
                      {
                        if ( v83 != -16 )
                          v82 = 0;
                        v131 = v82;
                      }
                      v81 = v79 & (v128 + v81);
                      v82 = (__int64 *)(v80 + 16LL * v81);
                      v83 = *v82;
                      if ( v55 == *v82 )
                        goto LABEL_72;
                      ++v128;
                    }
                    if ( v131 )
                      v82 = v131;
                  }
LABEL_72:
                  *v82 = v55;
                  *((_DWORD *)v82 + 2) = *((_DWORD *)v77 + 2);
                  ++*(_DWORD *)(a1 + 280);
                }
                v77 += 2;
              }
              while ( v75 != v77 );
              j___libc_free_0(v46);
              v73 = *(_QWORD **)(a1 + 272);
              v84 = *(_DWORD *)(a1 + 288);
              v72 = ((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4);
              v39 = v135;
              v63 = *(_DWORD *)(a1 + 280) + 1;
            }
            else
            {
              v107 = *(unsigned int *)(a1 + 288);
              *(_QWORD *)(a1 + 280) = 0;
              v84 = v107;
              for ( j = &v71[2 * v107]; j != v71; v71 += 2 )
              {
                if ( v71 )
                  *v71 = -8;
              }
              v63 = 1;
            }
            if ( !v84 )
              goto LABEL_207;
            v85 = v84 - 1;
            v86 = 1;
            v87 = v85 & v72;
            v88 = 0;
            v89 = v87;
            v42 = &v73[2 * v87];
            v90 = *v42;
            if ( v45 != *v42 )
            {
              while ( v90 != -8 )
              {
                if ( v90 != -16 || v88 )
                  v42 = v88;
                v89 = v85 & (v86 + v89);
                v90 = v73[2 * v89];
                if ( v45 == v90 )
                {
                  v42 = &v73[2 * v89];
                  goto LABEL_50;
                }
                ++v86;
                v88 = v42;
                v42 = &v73[2 * v89];
              }
              goto LABEL_78;
            }
          }
LABEL_50:
          *(_DWORD *)(a1 + 280) = v63;
          if ( *v42 != -8 )
            --*(_DWORD *)(a1 + 284);
          *((_DWORD *)v42 + 2) = 0;
          *v42 = v45;
          *((_DWORD *)v42 + 2) = v38;
          v37 = *(_QWORD *)(v37 + 8);
          if ( v39 == v37 )
            goto LABEL_53;
        }
      }
      ++*(_QWORD *)(a1 + 264);
LABEL_33:
      v129 = v39;
      v132 = v44;
      v47 = ((((((((2 * v44 - 1) | ((unsigned __int64)(2 * v44 - 1) >> 1)) >> 2)
               | (2 * v44 - 1)
               | ((unsigned __int64)(2 * v44 - 1) >> 1)) >> 4)
             | (((2 * v44 - 1) | ((unsigned __int64)(2 * v44 - 1) >> 1)) >> 2)
             | (2 * v44 - 1)
             | ((unsigned __int64)(2 * v44 - 1) >> 1)) >> 8)
           | (((((2 * v44 - 1) | ((unsigned __int64)(2 * v44 - 1) >> 1)) >> 2)
             | (2 * v44 - 1)
             | ((unsigned __int64)(2 * v44 - 1) >> 1)) >> 4)
           | (((2 * v44 - 1) | ((unsigned __int64)(2 * v44 - 1) >> 1)) >> 2)
           | (2 * v44 - 1)
           | ((unsigned __int64)(2 * v44 - 1) >> 1)) >> 16;
      v48 = (v47
           | (((((((2 * v44 - 1) | ((unsigned __int64)(2 * v44 - 1) >> 1)) >> 2)
               | (2 * v44 - 1)
               | ((unsigned __int64)(2 * v44 - 1) >> 1)) >> 4)
             | (((2 * v44 - 1) | ((unsigned __int64)(2 * v44 - 1) >> 1)) >> 2)
             | (2 * v44 - 1)
             | ((unsigned __int64)(2 * v44 - 1) >> 1)) >> 8)
           | (((((2 * v44 - 1) | ((unsigned __int64)(2 * v44 - 1) >> 1)) >> 2)
             | (2 * v44 - 1)
             | ((unsigned __int64)(2 * v44 - 1) >> 1)) >> 4)
           | (((2 * v44 - 1) | ((unsigned __int64)(2 * v44 - 1) >> 1)) >> 2)
           | (2 * v44 - 1)
           | ((unsigned __int64)(2 * v44 - 1) >> 1))
          + 1;
      if ( (unsigned int)v48 < 0x40 )
        LODWORD(v48) = 64;
      *(_DWORD *)(a1 + 288) = v48;
      v49 = (_QWORD *)sub_22077B0(16LL * (unsigned int)v48);
      v39 = v129;
      *(_QWORD *)(a1 + 272) = v49;
      v50 = v49;
      if ( v46 )
      {
        v51 = *(unsigned int *)(a1 + 288);
        *(_QWORD *)(a1 + 280) = 0;
        v52 = &v46[2 * v132];
        for ( k = &v50[2 * v51]; k != v50; v50 += 2 )
        {
          if ( v50 )
            *v50 = -8;
        }
        for ( m = v46; v52 != m; m += 2 )
        {
          v55 = *m;
          if ( *m != -8 && v55 != -16 )
          {
            v56 = *(_DWORD *)(a1 + 288);
            if ( !v56 )
              goto LABEL_208;
            v57 = v56 - 1;
            v58 = *(_QWORD *)(a1 + 272);
            v59 = v57 & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
            v60 = (__int64 *)(v58 + 16LL * v59);
            v61 = *v60;
            if ( *v60 != v55 )
            {
              v130 = 1;
              v136 = 0;
              while ( v61 != -8 )
              {
                if ( !v136 )
                {
                  if ( v61 != -16 )
                    v60 = 0;
                  v136 = v60;
                }
                v59 = v57 & (v130 + v59);
                v60 = (__int64 *)(v58 + 16LL * v59);
                v61 = *v60;
                if ( v55 == *v60 )
                  goto LABEL_45;
                ++v130;
              }
              if ( v136 )
                v60 = v136;
            }
LABEL_45:
            *v60 = v55;
            *((_DWORD *)v60 + 2) = *((_DWORD *)m + 2);
            ++*(_DWORD *)(a1 + 280);
          }
        }
        v133 = v39;
        j___libc_free_0(v46);
        v50 = *(_QWORD **)(a1 + 272);
        v62 = *(_DWORD *)(a1 + 288);
        v39 = v133;
        v63 = *(_DWORD *)(a1 + 280) + 1;
      }
      else
      {
        v101 = *(unsigned int *)(a1 + 288);
        *(_QWORD *)(a1 + 280) = 0;
        v62 = v101;
        for ( n = &v49[2 * v101]; n != v49; v49 += 2 )
        {
          if ( v49 )
            *v49 = -8;
        }
        v63 = 1;
      }
      if ( !v62 )
        goto LABEL_207;
      v64 = v62 - 1;
      v65 = (v62 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
      v42 = &v50[2 * v65];
      v66 = *v42;
      if ( v45 != *v42 )
      {
        v123 = 1;
        v88 = 0;
        while ( v66 != -8 )
        {
          if ( !v88 && v66 == -16 )
            v88 = v42;
          v65 = v64 & (v123 + v65);
          v42 = &v50[2 * v65];
          v66 = *v42;
          if ( v45 == *v42 )
            goto LABEL_50;
          ++v123;
        }
LABEL_78:
        if ( v88 )
          v42 = v88;
        goto LABEL_50;
      }
      goto LABEL_50;
    }
LABEL_53:
    sub_17D3A30((__int64)v138);
    v26 = v147;
    v28 = v148;
  }
LABEL_89:
  if ( v26 )
    j_j___libc_free_0(v26, v149 - v26);
  if ( v145 != v144[1] )
    _libc_free(v145);
  if ( v141 )
    j_j___libc_free_0(v141, v143 - v141);
  if ( v139 != v138[1] )
    _libc_free(v139);
  if ( v158 )
    j_j___libc_free_0(v158, v160 - v158);
  if ( v157 != v156 )
    _libc_free(v157);
  if ( v152 )
    j_j___libc_free_0(v152, v154 - v152);
  if ( v151 != v150[1] )
    _libc_free(v151);
  v95 = 0;
  for ( ii = 0; dword_4FAF060 == -1 || ++v95 < dword_4FAF060; ii = 1 )
  {
    v97 = sub_192A150(a1, a2, a3, a4, a5, a6, v20, v21, a9, a10);
    if ( !(HIDWORD(v97) + (_DWORD)v97) )
      break;
    if ( HIDWORD(v97) )
      sub_190D990(a1, a2, HIDWORD(v97), v98, v99, v100);
  }
  return ii;
}
