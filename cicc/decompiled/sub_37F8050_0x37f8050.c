// Function: sub_37F8050
// Address: 0x37f8050
//
__int64 __fastcall sub_37F8050(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __m128i *v4; // rdx
  __int64 v5; // r12
  const char *v6; // rax
  size_t v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rax
  size_t v11; // r13
  __int64 *v12; // rax
  __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // r15
  unsigned int v16; // r12d
  unsigned int v17; // eax
  __int64 v18; // r12
  __int64 v19; // rdi
  _QWORD *v20; // r9
  char *v21; // rax
  char *v22; // r11
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdi
  char *v27; // r13
  char *v28; // r12
  int v29; // ebx
  __int64 v30; // rax
  _BYTE *v31; // rax
  __int64 v32; // rax
  _WORD *v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rax
  _WORD *v36; // rdx
  __int64 v37; // r12
  _BYTE *v38; // rax
  int v39; // r8d
  unsigned int v40; // ecx
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rdi
  __int64 v44; // rsi
  __int64 v46; // r14
  char *v47; // r12
  char *v48; // r13
  __int64 v49; // r11
  __int64 v50; // r15
  unsigned __int64 v51; // rcx
  __int64 v52; // r8
  _QWORD *v53; // rdx
  unsigned int v54; // edi
  __int64 v55; // rax
  __int64 v56; // rsi
  int v57; // ebx
  __int64 v58; // rax
  unsigned __int64 v59; // rdx
  unsigned int v60; // edx
  char *v61; // rax
  int v62; // esi
  unsigned __int64 v63; // rdx
  unsigned __int64 v64; // rax
  _QWORD *v65; // rax
  __int64 v66; // r10
  __int64 v67; // r9
  _QWORD *j; // rdx
  __int64 k; // rax
  __int64 v70; // r14
  unsigned int v71; // ecx
  _QWORD *v72; // rdx
  __int64 v73; // rdi
  _QWORD *v74; // rcx
  unsigned int v75; // edx
  __int64 v76; // rax
  __int64 v77; // rdi
  int v78; // r10d
  unsigned __int64 v79; // rax
  unsigned __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rax
  unsigned int v83; // r10d
  _QWORD *v84; // rdx
  __int64 v85; // r9
  _QWORD *i; // rax
  __int64 v87; // rax
  __int64 v88; // r14
  unsigned int v89; // ecx
  _QWORD *v90; // rdx
  __int64 v91; // rdi
  _QWORD *v92; // rdi
  unsigned int v93; // eax
  unsigned int v94; // r8d
  unsigned int v95; // eax
  int v96; // r10d
  __int64 v97; // rcx
  unsigned int v98; // ecx
  int v99; // edx
  __int64 v100; // r11
  unsigned int v101; // esi
  unsigned int v102; // esi
  unsigned int v103; // ebx
  __int64 v104; // rcx
  __int64 v105; // r8
  __int64 *v106; // [rsp+0h] [rbp-F0h]
  __int64 *v107; // [rsp+8h] [rbp-E8h]
  __int64 v108; // [rsp+18h] [rbp-D8h]
  int v109; // [rsp+18h] [rbp-D8h]
  __int64 v110; // [rsp+20h] [rbp-D0h]
  int v111; // [rsp+20h] [rbp-D0h]
  _QWORD *v112; // [rsp+20h] [rbp-D0h]
  signed __int64 v113; // [rsp+28h] [rbp-C8h]
  int v114; // [rsp+34h] [rbp-BCh]
  __int64 v116; // [rsp+40h] [rbp-B0h]
  __int64 v117; // [rsp+48h] [rbp-A8h]
  __int64 v118; // [rsp+48h] [rbp-A8h]
  __int64 v119; // [rsp+48h] [rbp-A8h]
  __int64 v120; // [rsp+48h] [rbp-A8h]
  _QWORD *v121; // [rsp+48h] [rbp-A8h]
  __int64 v122; // [rsp+58h] [rbp-98h]
  void *base; // [rsp+60h] [rbp-90h] BYREF
  __int64 v124; // [rsp+68h] [rbp-88h]
  __int64 v125; // [rsp+70h] [rbp-80h] BYREF
  _QWORD *v126; // [rsp+78h] [rbp-78h]
  __int64 v127; // [rsp+80h] [rbp-70h]
  unsigned int v128; // [rsp+88h] [rbp-68h]
  __int64 v129; // [rsp+90h] [rbp-60h] BYREF
  void *s; // [rsp+98h] [rbp-58h]
  _BYTE v131[12]; // [rsp+A0h] [rbp-50h]
  char v132; // [rsp+ACh] [rbp-44h]
  char v133; // [rsp+B0h] [rbp-40h] BYREF

  v3 = sub_C5F790(a1, (__int64)a2);
  v4 = *(__m128i **)(v3 + 32);
  v5 = v3;
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v4 <= 0xFu )
  {
    v5 = sub_CB6200(v3, "RDA results for ", 0x10u);
  }
  else
  {
    *v4 = _mm_load_si128((const __m128i *)&xmmword_45277F0);
    *(_QWORD *)(v3 + 32) += 16LL;
  }
  v6 = sub_2E791E0(a2);
  v8 = *(_QWORD *)(v5 + 32);
  v9 = (__int64)v6;
  v10 = *(_QWORD *)(v5 + 24);
  v11 = v7;
  if ( v10 - v8 < v7 )
  {
    v5 = sub_CB6200(v5, (unsigned __int8 *)v9, v7);
    v10 = *(_QWORD *)(v5 + 24);
    v8 = *(_QWORD *)(v5 + 32);
  }
  else if ( v7 )
  {
    memcpy((void *)v8, (const void *)v9, v7);
    v10 = *(_QWORD *)(v5 + 24);
    v8 = v11 + *(_QWORD *)(v5 + 32);
    *(_QWORD *)(v5 + 32) = v8;
  }
  if ( v8 == v10 )
  {
    v9 = (__int64)"\n";
    v8 = v5;
    sub_CB6200(v5, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *(_BYTE *)v8 = 10;
    ++*(_QWORD *)(v5 + 32);
  }
  v125 = 0;
  s = &v133;
  v12 = (__int64 *)a2[41];
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  *(_QWORD *)v131 = 2;
  *(_DWORD *)&v131[8] = 0;
  v132 = 1;
  v107 = v12;
  v106 = a2 + 40;
  if ( v12 == a2 + 40 )
  {
    v43 = 0;
    v44 = 0;
    return sub_C7D6A0(v43, v44, 8);
  }
  v114 = 0;
  do
  {
    if ( v107 + 6 == (__int64 *)v107[7] )
      goto LABEL_49;
    v13 = v107[7];
    v113 = v114;
    do
    {
      v14 = *(_QWORD *)(v13 + 32);
      v122 = v14 + 40LL * (*(_DWORD *)(v13 + 40) & 0xFFFFFF);
      if ( v14 == v122 )
        goto LABEL_42;
      v116 = v13;
      v15 = *(_QWORD *)(v13 + 32);
      do
      {
        while ( 1 )
        {
          if ( *(_BYTE *)v15 == 5 )
          {
            v16 = *(_DWORD *)(v15 + 24) + 0x40000000;
            break;
          }
          if ( !*(_BYTE *)v15 && (*(_BYTE *)(v15 + 3) & 0x10) == 0 )
          {
            v16 = *(_DWORD *)(v15 + 8);
            if ( v16 )
              break;
          }
LABEL_14:
          v15 += 40;
          if ( v122 == v15 )
            goto LABEL_41;
        }
        ++v129;
        if ( !v132 )
        {
          v17 = 4 * (*(_DWORD *)&v131[4] - *(_DWORD *)&v131[8]);
          if ( v17 < 0x20 )
            v17 = 32;
          if ( *(_DWORD *)v131 > v17 )
          {
            sub_C8C990((__int64)&v129, v9);
            goto LABEL_25;
          }
          memset(s, -1, 8LL * *(unsigned int *)v131);
        }
        *(_QWORD *)&v131[4] = 0;
LABEL_25:
        sub_37F62A0(a1, v116, v16, (__int64)&v129);
        v18 = *(_QWORD *)(a1 + 208);
        v19 = v15;
        v9 = sub_C5F790(a1, v116);
        sub_2EAF9A0(v15, v9, v18);
        v124 = 0;
        base = &v125;
        v21 = (char *)s;
        if ( v132 )
          v22 = (char *)s + 8 * *(unsigned int *)&v131[4];
        else
          v22 = (char *)s + 8 * *(unsigned int *)v131;
        if ( s != v22 )
        {
          while ( 1 )
          {
            v23 = *(_QWORD *)v21;
            if ( *(_QWORD *)v21 < 0xFFFFFFFFFFFFFFFELL )
              break;
            v21 += 8;
            if ( v22 == v21 )
              goto LABEL_30;
          }
          if ( v22 != v21 )
          {
            v46 = v128;
            v47 = v22;
            v48 = v21;
            v49 = v15;
            v50 = (__int64)v126;
            if ( v128 )
            {
LABEL_56:
              v51 = (unsigned int)(v46 - 1);
              v52 = 1;
              v53 = 0;
              v54 = v51 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
              v55 = v50 + 16LL * v54;
              v56 = *(_QWORD *)v55;
              if ( v23 == *(_QWORD *)v55 )
                goto LABEL_57;
              while ( v56 != -4096 )
              {
                if ( v56 != -8192 || v53 )
                  v55 = (__int64)v53;
                v54 = v51 & (v52 + v54);
                v20 = (_QWORD *)(v50 + 16LL * v54);
                v56 = *v20;
                if ( v23 == *v20 )
                {
                  v55 = v50 + 16LL * v54;
LABEL_57:
                  v57 = *(_DWORD *)(v55 + 8);
                  goto LABEL_58;
                }
                v52 = (unsigned int)(v52 + 1);
                v53 = (_QWORD *)v55;
                v55 = v50 + 16LL * v54;
              }
              if ( !v53 )
                v53 = (_QWORD *)v55;
              ++v125;
              v62 = v127 + 1;
              if ( 4 * ((int)v127 + 1) >= (unsigned int)(3 * v46) )
                goto LABEL_77;
              if ( (int)v46 - HIDWORD(v127) - v62 > (unsigned int)v46 >> 3 )
                goto LABEL_71;
              v110 = v49;
              v79 = (v51 >> 1) | v51 | (((v51 >> 1) | v51) >> 2);
              v80 = (((v79 >> 4) | v79) >> 8) | (v79 >> 4) | v79;
              v81 = ((v80 >> 16) | v80) + 1;
              if ( (unsigned int)v81 < 0x40 )
                LODWORD(v81) = 64;
              v128 = v81;
              v82 = sub_C7D670(16LL * (unsigned int)v81, 8);
              v83 = ((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4);
              v49 = v110;
              v126 = (_QWORD *)v82;
              v84 = (_QWORD *)v82;
              if ( v50 )
              {
                v127 = 0;
                v120 = 16 * v46;
                v85 = v50 + 16 * v46;
                for ( i = (_QWORD *)(v82 + 16LL * v128); i != v84; v84 += 2 )
                {
                  if ( v84 )
                    *v84 = -4096;
                }
                v87 = v50;
                do
                {
                  v88 = *(_QWORD *)v87;
                  if ( *(_QWORD *)v87 != -8192 && v88 != -4096 )
                  {
                    if ( !v128 )
                    {
                      MEMORY[0] = *(_QWORD *)v87;
                      BUG();
                    }
                    v89 = (v128 - 1) & (((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4));
                    v90 = &v126[2 * v89];
                    v91 = *v90;
                    if ( v88 != *v90 )
                    {
                      v109 = 1;
                      v112 = 0;
                      while ( v91 != -4096 )
                      {
                        if ( v91 == -8192 )
                        {
                          if ( v112 )
                            v90 = v112;
                          v112 = v90;
                        }
                        v89 = (v128 - 1) & (v109 + v89);
                        v90 = &v126[2 * v89];
                        v91 = *v90;
                        if ( v88 == *v90 )
                          goto LABEL_116;
                        ++v109;
                      }
                      if ( v112 )
                        v90 = v112;
                    }
LABEL_116:
                    *v90 = v88;
                    *((_DWORD *)v90 + 2) = *(_DWORD *)(v87 + 8);
                    LODWORD(v127) = v127 + 1;
                  }
                  v87 += 16;
                }
                while ( v85 != v87 );
                v108 = v49;
                sub_C7D6A0(v50, v120, 8);
                v92 = v126;
                v93 = v128;
                v83 = ((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4);
                v49 = v108;
                v62 = v127 + 1;
              }
              else
              {
                v127 = 0;
                v93 = v128;
                v102 = v128;
                v92 = &v84[2 * v128];
                if ( v84 == v92 )
                {
                  v62 = 1;
                }
                else
                {
                  do
                  {
                    if ( v84 )
                      *v84 = -4096;
                    v84 += 2;
                  }
                  while ( v92 != v84 );
                  v93 = v102;
                  v92 = v126;
                  v62 = v127 + 1;
                }
              }
              if ( !v93 )
                goto LABEL_219;
              v94 = v83;
              v95 = v93 - 1;
              v96 = 1;
              v20 = 0;
              v52 = v95 & v94;
              v53 = &v92[2 * (unsigned int)v52];
              v97 = *v53;
              if ( v23 == *v53 )
                goto LABEL_71;
              while ( v97 != -4096 )
              {
                if ( v97 == -8192 && !v20 )
                  v20 = v53;
                v52 = v95 & ((_DWORD)v52 + v96);
                v53 = &v92[2 * v52];
                v97 = *v53;
                if ( v23 == *v53 )
                  goto LABEL_71;
                ++v96;
              }
            }
            else
            {
              while ( 1 )
              {
                ++v125;
LABEL_77:
                v117 = v49;
                v63 = ((((((((unsigned int)(2 * v46 - 1) | ((unsigned __int64)(unsigned int)(2 * v46 - 1) >> 1)) >> 2)
                         | (unsigned int)(2 * v46 - 1)
                         | ((unsigned __int64)(unsigned int)(2 * v46 - 1) >> 1)) >> 4)
                       | (((unsigned int)(2 * v46 - 1) | ((unsigned __int64)(unsigned int)(2 * v46 - 1) >> 1)) >> 2)
                       | (unsigned int)(2 * v46 - 1)
                       | ((unsigned __int64)(unsigned int)(2 * v46 - 1) >> 1)) >> 8)
                     | (((((unsigned int)(2 * v46 - 1) | ((unsigned __int64)(unsigned int)(2 * v46 - 1) >> 1)) >> 2)
                       | (unsigned int)(2 * v46 - 1)
                       | ((unsigned __int64)(unsigned int)(2 * v46 - 1) >> 1)) >> 4)
                     | (((unsigned int)(2 * v46 - 1) | ((unsigned __int64)(unsigned int)(2 * v46 - 1) >> 1)) >> 2)
                     | (unsigned int)(2 * v46 - 1)
                     | ((unsigned __int64)(unsigned int)(2 * v46 - 1) >> 1)) >> 16;
                v64 = (v63
                     | (((((((unsigned int)(2 * v46 - 1) | ((unsigned __int64)(unsigned int)(2 * v46 - 1) >> 1)) >> 2)
                         | (unsigned int)(2 * v46 - 1)
                         | ((unsigned __int64)(unsigned int)(2 * v46 - 1) >> 1)) >> 4)
                       | (((unsigned int)(2 * v46 - 1) | ((unsigned __int64)(unsigned int)(2 * v46 - 1) >> 1)) >> 2)
                       | (unsigned int)(2 * v46 - 1)
                       | ((unsigned __int64)(unsigned int)(2 * v46 - 1) >> 1)) >> 8)
                     | (((((unsigned int)(2 * v46 - 1) | ((unsigned __int64)(unsigned int)(2 * v46 - 1) >> 1)) >> 2)
                       | (unsigned int)(2 * v46 - 1)
                       | ((unsigned __int64)(unsigned int)(2 * v46 - 1) >> 1)) >> 4)
                     | (((unsigned int)(2 * v46 - 1) | ((unsigned __int64)(unsigned int)(2 * v46 - 1) >> 1)) >> 2)
                     | (unsigned int)(2 * v46 - 1)
                     | ((unsigned __int64)(unsigned int)(2 * v46 - 1) >> 1))
                    + 1;
                if ( (unsigned int)v64 < 0x40 )
                  LODWORD(v64) = 64;
                v128 = v64;
                v65 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v64, 8);
                v49 = v117;
                v126 = v65;
                if ( v50 )
                {
                  v127 = 0;
                  v66 = 16 * v46;
                  v67 = v50 + 16 * v46;
                  for ( j = &v65[2 * v128]; j != v65; v65 += 2 )
                  {
                    if ( v65 )
                      *v65 = -4096;
                  }
                  for ( k = v50; v67 != k; k += 16 )
                  {
                    v70 = *(_QWORD *)k;
                    if ( *(_QWORD *)k != -4096 && v70 != -8192 )
                    {
                      if ( !v128 )
                      {
                        MEMORY[0] = *(_QWORD *)k;
                        BUG();
                      }
                      v71 = (v128 - 1) & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
                      v72 = &v126[2 * v71];
                      v73 = *v72;
                      if ( v70 != *v72 )
                      {
                        v111 = 1;
                        v121 = 0;
                        while ( v73 != -4096 )
                        {
                          if ( v73 == -8192 )
                          {
                            if ( v121 )
                              v72 = v121;
                            v121 = v72;
                          }
                          v71 = (v128 - 1) & (v111 + v71);
                          v72 = &v126[2 * v71];
                          v73 = *v72;
                          if ( v70 == *v72 )
                            goto LABEL_89;
                          ++v111;
                        }
                        if ( v121 )
                          v72 = v121;
                      }
LABEL_89:
                      *v72 = v70;
                      *((_DWORD *)v72 + 2) = *(_DWORD *)(k + 8);
                      LODWORD(v127) = v127 + 1;
                    }
                  }
                  v118 = v49;
                  sub_C7D6A0(v50, v66, 8);
                  v74 = v126;
                  v75 = v128;
                  v49 = v118;
                  v62 = v127 + 1;
                }
                else
                {
                  v127 = 0;
                  v75 = v128;
                  v101 = v128;
                  v74 = &v65[2 * v128];
                  if ( v65 == v74 )
                  {
                    v62 = 1;
                  }
                  else
                  {
                    do
                    {
                      if ( v65 )
                        *v65 = -4096;
                      v65 += 2;
                    }
                    while ( v74 != v65 );
                    v75 = v101;
                    v74 = v126;
                    v62 = v127 + 1;
                  }
                }
                if ( !v75 )
                {
LABEL_219:
                  LODWORD(v127) = v127 + 1;
                  BUG();
                }
                v52 = v75 - 1;
                LODWORD(v76) = v52 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
                v53 = &v74[2 * (unsigned int)v76];
                v77 = *v53;
                if ( v23 != *v53 )
                  break;
LABEL_71:
                LODWORD(v127) = v62;
                if ( *v53 != -4096 )
                  --HIDWORD(v127);
                *v53 = v23;
                v57 = 0;
                *((_DWORD *)v53 + 2) = 0;
LABEL_58:
                v58 = (unsigned int)v124;
                v59 = (unsigned int)v124 + 1LL;
                if ( v59 > HIDWORD(v124) )
                {
                  v119 = v49;
                  sub_C8D5F0((__int64)&base, &v125, v59, 4u, v52, (__int64)v20);
                  v58 = (unsigned int)v124;
                  v49 = v119;
                }
                *((_DWORD *)base + v58) = v57;
                v60 = v124 + 1;
                v61 = v48 + 8;
                LODWORD(v124) = v124 + 1;
                if ( v48 + 8 == v47 )
                  goto LABEL_63;
                while ( 1 )
                {
                  v23 = *(_QWORD *)v61;
                  v48 = v61;
                  if ( *(_QWORD *)v61 < 0xFFFFFFFFFFFFFFFELL )
                    break;
                  v61 += 8;
                  if ( v47 == v61 )
                    goto LABEL_63;
                }
                if ( v47 == v61 )
                {
LABEL_63:
                  v9 = v60;
                  v19 = (__int64)base;
                  v15 = v49;
                  if ( 4 * (unsigned __int64)v60 > 4 )
                    qsort(base, v60, 4u, (__compar_fn_t)sub_29F3DB0);
                  goto LABEL_30;
                }
                v46 = v128;
                v50 = (__int64)v126;
                if ( v128 )
                  goto LABEL_56;
              }
              v78 = 1;
              v20 = 0;
              while ( v77 != -4096 )
              {
                if ( v77 == -8192 && !v20 )
                  v20 = v53;
                v76 = (unsigned int)v52 & ((_DWORD)v76 + v78);
                v53 = &v74[2 * v76];
                v77 = *v53;
                if ( v23 == *v53 )
                  goto LABEL_71;
                ++v78;
              }
            }
            if ( v20 )
              v53 = v20;
            goto LABEL_71;
          }
        }
LABEL_30:
        v24 = sub_C5F790(v19, v9);
        v25 = *(_QWORD *)(v24 + 32);
        v26 = v24;
        if ( (unsigned __int64)(*(_QWORD *)(v24 + 24) - v25) <= 2 )
        {
          v9 = (__int64)":{ ";
          sub_CB6200(v24, ":{ ", 3u);
        }
        else
        {
          *(_BYTE *)(v25 + 2) = 32;
          *(_WORD *)v25 = 31546;
          *(_QWORD *)(v24 + 32) += 3LL;
        }
        v27 = (char *)base;
        v28 = (char *)base + 4 * (unsigned int)v124;
        if ( v28 != base )
        {
          do
          {
            while ( 1 )
            {
              v29 = *(_DWORD *)v27;
              v30 = sub_C5F790(v26, v9);
              v9 = v29;
              v26 = sub_CB59F0(v30, v29);
              v31 = *(_BYTE **)(v26 + 32);
              if ( *(_BYTE **)(v26 + 24) == v31 )
                break;
              v27 += 4;
              *v31 = 32;
              ++*(_QWORD *)(v26 + 32);
              if ( v28 == v27 )
                goto LABEL_37;
            }
            v9 = (__int64)" ";
            v27 += 4;
            sub_CB6200(v26, (unsigned __int8 *)" ", 1u);
          }
          while ( v28 != v27 );
        }
LABEL_37:
        v32 = sub_C5F790(v26, v9);
        v33 = *(_WORD **)(v32 + 32);
        if ( *(_QWORD *)(v32 + 24) - (_QWORD)v33 <= 1u )
        {
          v9 = (__int64)"}\n";
          sub_CB6200(v32, "}\n", 2u);
        }
        else
        {
          *v33 = 2685;
          *(_QWORD *)(v32 + 32) += 2LL;
        }
        v8 = (__int64)base;
        if ( base == &v125 )
          goto LABEL_14;
        _libc_free((unsigned __int64)base);
        v15 += 40;
      }
      while ( v122 != v15 );
LABEL_41:
      v13 = v116;
LABEL_42:
      v34 = sub_C5F790(v8, v9);
      v35 = sub_CB59F0(v34, v113);
      v36 = *(_WORD **)(v35 + 32);
      v37 = v35;
      if ( *(_QWORD *)(v35 + 24) - (_QWORD)v36 <= 1u )
      {
        v37 = sub_CB6200(v35, (unsigned __int8 *)": ", 2u);
      }
      else
      {
        *v36 = 8250;
        *(_QWORD *)(v35 + 32) += 2LL;
      }
      sub_2E91850(v13, v37, 1u, 0, 0, 1, 0);
      v38 = *(_BYTE **)(v37 + 32);
      if ( *(_BYTE **)(v37 + 24) != v38 )
      {
        *v38 = 10;
        v9 = v128;
        ++*(_QWORD *)(v37 + 32);
        if ( (_DWORD)v9 )
          goto LABEL_46;
LABEL_130:
        ++v125;
LABEL_131:
        v8 = (__int64)&v125;
        sub_354C5D0((__int64)&v125, 2 * v9);
        if ( !v128 )
          goto LABEL_217;
        v9 = (unsigned int)v127;
        v98 = (v128 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v99 = v127 + 1;
        v41 = (__int64)&v126[2 * v98];
        v100 = *(_QWORD *)v41;
        if ( v13 != *(_QWORD *)v41 )
        {
          v8 = 1;
          v9 = 0;
          while ( v100 != -4096 )
          {
            if ( !v9 && v100 == -8192 )
              v9 = v41;
            v98 = (v128 - 1) & (v8 + v98);
            v41 = (__int64)&v126[2 * v98];
            v100 = *(_QWORD *)v41;
            if ( v13 == *(_QWORD *)v41 )
              goto LABEL_174;
            v8 = (unsigned int)(v8 + 1);
          }
          if ( v9 )
            v41 = v9;
        }
        goto LABEL_174;
      }
      sub_CB6200(v37, (unsigned __int8 *)"\n", 1u);
      v9 = v128;
      if ( !v128 )
        goto LABEL_130;
LABEL_46:
      v39 = 1;
      v8 = 0;
      v40 = (v9 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v41 = (__int64)&v126[2 * v40];
      v42 = *(_QWORD *)v41;
      if ( v13 != *(_QWORD *)v41 )
      {
        while ( v42 != -4096 )
        {
          if ( v42 == -8192 && !v8 )
            v8 = v41;
          v40 = (v9 - 1) & (v39 + v40);
          v41 = (__int64)&v126[2 * v40];
          v42 = *(_QWORD *)v41;
          if ( v13 == *(_QWORD *)v41 )
            goto LABEL_47;
          ++v39;
        }
        if ( v8 )
          v41 = v8;
        ++v125;
        v99 = v127 + 1;
        if ( 4 * ((int)v127 + 1) >= (unsigned int)(3 * v9) )
          goto LABEL_131;
        v8 = (unsigned int)v9 >> 3;
        if ( (int)v9 - HIDWORD(v127) - v99 <= (unsigned int)v8 )
        {
          sub_354C5D0((__int64)&v125, v9);
          if ( !v128 )
          {
LABEL_217:
            LODWORD(v127) = v127 + 1;
            BUG();
          }
          v8 = v128 - 1;
          v9 = 1;
          v103 = v8 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v99 = v127 + 1;
          v104 = 0;
          v41 = (__int64)&v126[2 * v103];
          v105 = *(_QWORD *)v41;
          if ( v13 != *(_QWORD *)v41 )
          {
            while ( v105 != -4096 )
            {
              if ( !v104 && v105 == -8192 )
                v104 = v41;
              v103 = v8 & (v9 + v103);
              v41 = (__int64)&v126[2 * v103];
              v105 = *(_QWORD *)v41;
              if ( v13 == *(_QWORD *)v41 )
                goto LABEL_174;
              v9 = (unsigned int)(v9 + 1);
            }
            if ( v104 )
              v41 = v104;
          }
        }
LABEL_174:
        LODWORD(v127) = v99;
        if ( *(_QWORD *)v41 != -4096 )
          --HIDWORD(v127);
        *(_QWORD *)v41 = v13;
        *(_DWORD *)(v41 + 8) = 0;
      }
LABEL_47:
      *(_DWORD *)(v41 + 8) = v114++;
      if ( (*(_BYTE *)v13 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v13 + 44) & 8) != 0 )
          v13 = *(_QWORD *)(v13 + 8);
      }
      ++v113;
      v13 = *(_QWORD *)(v13 + 8);
    }
    while ( v107 + 6 != (__int64 *)v13 );
LABEL_49:
    v107 = (__int64 *)v107[1];
  }
  while ( v106 != v107 );
  if ( !v132 )
    _libc_free((unsigned __int64)s);
  v43 = (__int64)v126;
  v44 = 16LL * v128;
  return sub_C7D6A0(v43, v44, 8);
}
