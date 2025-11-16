// Function: sub_19AD670
// Address: 0x19ad670
//
void __fastcall sub_19AD670(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  __int64 *v10; // r12
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rbx
  __int64 v14; // r13
  float v15; // xmm0_4
  int v16; // esi
  unsigned int v17; // ecx
  __int64 v18; // rax
  int v19; // r8d
  __int64 v20; // rdi
  int v21; // ecx
  __int64 v22; // r15
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 *v27; // r15
  _BYTE *v28; // r13
  _QWORD *v29; // rax
  __int64 v30; // r12
  __int64 *v31; // r14
  unsigned __int64 v32; // rdx
  float *v33; // r13
  float v34; // xmm2_4
  __int64 v35; // rbx
  _QWORD *v36; // rax
  _BYTE *v37; // r12
  unsigned __int64 v38; // rsi
  __int64 *v39; // r10
  __int64 *v40; // rax
  int v41; // edi
  int v42; // esi
  unsigned int v43; // r8d
  unsigned int v44; // edx
  int v45; // eax
  unsigned __int64 v46; // r10
  unsigned int v47; // r12d
  __int64 v48; // r13
  unsigned int v49; // edx
  __int64 v50; // r9
  unsigned int v51; // r14d
  __int64 v52; // rbx
  float v53; // xmm2_4
  __int64 v54; // rdx
  unsigned __int64 v55; // rax
  __int64 v56; // rax
  __int64 *v57; // rax
  __int64 *v58; // rdx
  __int64 *v59; // r11
  __int64 *i; // rax
  __int64 *j; // rax
  __int64 v62; // r13
  int v63; // r10d
  __int64 *v64; // r9
  unsigned int v65; // ecx
  __int64 *v66; // rdx
  __int64 v67; // rdi
  unsigned int v68; // edi
  unsigned int v69; // edi
  unsigned int v70; // ecx
  __int64 v71; // r11
  int v72; // r9d
  __int64 *v73; // r8
  unsigned int v74; // eax
  __int64 *v75; // rax
  unsigned int v76; // r11d
  __int64 *v77; // rdx
  __int64 *m; // rax
  __int64 *v79; // rax
  __int64 v80; // r13
  int v81; // r11d
  __int64 *v82; // r9
  unsigned int v83; // ecx
  __int64 *v84; // rdx
  __int64 v85; // rdi
  unsigned int v86; // edi
  unsigned int v87; // r9d
  __int64 *v88; // rcx
  int v89; // edi
  unsigned int v90; // r11d
  __int64 v91; // r8
  int v92; // eax
  __int64 v93; // rax
  __int64 v94; // rax
  char *v95; // rdi
  __int64 v96; // r13
  __int64 *v97; // r14
  __int64 *v98; // rbx
  __int64 *v99; // r8
  __int64 v100; // rsi
  __int64 *v101; // rdi
  __int64 *v102; // rax
  __int64 *v103; // rcx
  unsigned int v104; // r11d
  int v105; // edi
  float *v106; // rax
  int v107; // edx
  float *v108; // rsi
  unsigned int v109; // r10d
  unsigned __int64 v110; // rcx
  __int64 v111; // r13
  __int64 v112; // r12
  __int64 v113; // r8
  __int64 v114; // rdx
  __int64 v115; // rcx
  int v116; // r8d
  int v117; // r9d
  __int64 *n; // rcx
  __int64 *k; // rcx
  float v120; // xmm0_4
  __int64 v121; // r8
  __int64 v122; // r11
  unsigned int v123; // edi
  int v124; // r10d
  __int64 v125; // rsi
  int v126; // eax
  int v127; // esi
  int v128; // eax
  int v129; // esi
  int v130; // esi
  float *v131; // rcx
  __int64 v132; // rax
  int v133; // r8d
  __int64 v134; // rsi
  float v135; // xmm0_4
  unsigned int v136; // [rsp+8h] [rbp-1A8h]
  unsigned int v137; // [rsp+Ch] [rbp-1A4h]
  unsigned __int64 v138; // [rsp+10h] [rbp-1A0h]
  float *v139; // [rsp+18h] [rbp-198h]
  __int64 v140; // [rsp+20h] [rbp-190h]
  __int64 v141; // [rsp+28h] [rbp-188h]
  unsigned int v142; // [rsp+30h] [rbp-180h]
  unsigned int v143; // [rsp+30h] [rbp-180h]
  unsigned int v144; // [rsp+30h] [rbp-180h]
  unsigned int v145; // [rsp+30h] [rbp-180h]
  __int64 v146; // [rsp+30h] [rbp-180h]
  __int64 v148; // [rsp+40h] [rbp-170h]
  __int64 v149; // [rsp+48h] [rbp-168h]
  unsigned int v150; // [rsp+50h] [rbp-160h]
  float v151; // [rsp+54h] [rbp-15Ch]
  unsigned __int64 v152; // [rsp+58h] [rbp-158h]
  __int64 v153; // [rsp+68h] [rbp-148h]
  float *v154; // [rsp+68h] [rbp-148h]
  float v155; // [rsp+70h] [rbp-140h]
  float v156; // [rsp+74h] [rbp-13Ch]
  __int64 v157; // [rsp+78h] [rbp-138h]
  __int64 v158; // [rsp+80h] [rbp-130h]
  __int64 v159; // [rsp+88h] [rbp-128h]
  __int64 *v160; // [rsp+90h] [rbp-120h]
  __int64 *v161; // [rsp+90h] [rbp-120h]
  __int64 v162; // [rsp+90h] [rbp-120h]
  float v163; // [rsp+98h] [rbp-118h]
  float v164; // [rsp+98h] [rbp-118h]
  __int64 v165; // [rsp+A8h] [rbp-108h] BYREF
  __int64 v166; // [rsp+B0h] [rbp-100h] BYREF
  __int64 *v167; // [rsp+B8h] [rbp-F8h]
  __int64 v168; // [rsp+C0h] [rbp-F0h]
  unsigned int v169; // [rsp+C8h] [rbp-E8h]
  __int64 v170; // [rsp+D0h] [rbp-E0h] BYREF
  _BYTE *v171; // [rsp+D8h] [rbp-D8h]
  _BYTE *v172; // [rsp+E0h] [rbp-D0h]
  __int64 v173; // [rsp+E8h] [rbp-C8h]
  int v174; // [rsp+F0h] [rbp-C0h]
  _BYTE v175[40]; // [rsp+F8h] [rbp-B8h] BYREF
  __int64 v176; // [rsp+120h] [rbp-90h] BYREF
  __int64 v177; // [rsp+128h] [rbp-88h]
  char v178; // [rsp+130h] [rbp-80h]
  __int64 v179; // [rsp+138h] [rbp-78h]
  char *v180[2]; // [rsp+140h] [rbp-70h] BYREF
  _BYTE v181[32]; // [rsp+150h] [rbp-60h] BYREF
  __int64 v182; // [rsp+170h] [rbp-40h]
  __int64 v183; // [rsp+178h] [rbp-38h]

  v6 = *(_QWORD *)(a1 + 368);
  v141 = *(unsigned int *)(a1 + 376);
  v7 = v6 + 1984 * v141;
  if ( v6 != v7 )
  {
    v8 = 1;
    do
    {
      v9 = *(unsigned int *)(v6 + 752);
      if ( v9 > 0xFFFE )
        goto LABEL_6;
      v8 *= v9;
      if ( v8 > 0x3FFFB )
        goto LABEL_6;
      v6 += 1984;
    }
    while ( v7 != v6 );
    if ( v8 > 0xFFFE )
    {
LABEL_6:
      v170 = 0;
      v171 = v175;
      v172 = v175;
      v173 = 4;
      v10 = *(__int64 **)(a1 + 32160);
      v11 = *(unsigned int *)(a1 + 32168);
      v174 = 0;
      v166 = 0;
      v167 = 0;
      v168 = 0;
      v169 = 0;
      v160 = &v10[v11];
      if ( v160 == v10 )
        goto LABEL_29;
      while ( 1 )
      {
        v12 = *v10;
        if ( sub_199CBE0((__int64)&v170, *v10) )
          goto LABEL_8;
        v13 = *(_QWORD *)(a1 + 368);
        v14 = v13 + 1984LL * *(unsigned int *)(a1 + 376);
        if ( v13 != v14 )
        {
          v163 = 1.0;
          while ( 1 )
          {
            if ( !sub_199CBE0(v13 + 1912, v12) )
              goto LABEL_13;
            v15 = sub_1993D70(v13, v12);
            if ( v15 == 0.0 )
            {
              v13 += 1984;
              sub_199AF80((__int64)&v176, (__int64)&v170, v12);
              if ( v13 == v14 )
                goto LABEL_17;
            }
            else
            {
              v163 = v15 * v163;
LABEL_13:
              v13 += 1984;
              if ( v13 == v14 )
                goto LABEL_17;
            }
          }
        }
        v163 = 1.0;
LABEL_17:
        v16 = v169;
        v176 = v12;
        *(float *)&v177 = v163;
        if ( !v169 )
        {
          ++v166;
LABEL_225:
          v16 = 2 * v169;
LABEL_226:
          sub_19AD4A0((__int64)&v166, v16);
          sub_19A7EE0((__int64)&v166, &v176, &v165);
          v18 = v165;
          v12 = v176;
          v21 = v168 + 1;
          goto LABEL_25;
        }
        v17 = (v169 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v18 = (__int64)&v167[2 * v17];
        a6 = *(_QWORD *)v18;
        if ( v12 == *(_QWORD *)v18 )
        {
LABEL_8:
          if ( v160 == ++v10 )
            goto LABEL_28;
        }
        else
        {
          v19 = 1;
          v20 = 0;
          while ( a6 != -8 )
          {
            if ( !v20 && a6 == -16 )
              v20 = v18;
            v17 = (v169 - 1) & (v19 + v17);
            v18 = (__int64)&v167[2 * v17];
            a6 = *(_QWORD *)v18;
            if ( v12 == *(_QWORD *)v18 )
              goto LABEL_8;
            ++v19;
          }
          if ( v20 )
            v18 = v20;
          ++v166;
          v21 = v168 + 1;
          if ( 4 * ((int)v168 + 1) >= 3 * v169 )
            goto LABEL_225;
          if ( v169 - HIDWORD(v168) - v21 <= v169 >> 3 )
            goto LABEL_226;
LABEL_25:
          LODWORD(v168) = v21;
          if ( *(_QWORD *)v18 != -8 )
            --HIDWORD(v168);
          *(_QWORD *)v18 = v12;
          ++v10;
          *(_DWORD *)(v18 + 8) = v177;
          if ( v160 == v10 )
          {
LABEL_28:
            v141 = *(unsigned int *)(a1 + 376);
LABEL_29:
            v140 = 0;
            v148 = 0;
            if ( !v141 )
            {
LABEL_105:
              j___libc_free_0(v167);
              if ( v172 != v171 )
                _libc_free((unsigned __int64)v172);
              return;
            }
LABEL_30:
            v22 = *(_QWORD *)(a1 + 368) + v140;
            v152 = *(unsigned int *)(v22 + 752);
            if ( v152 <= 1 )
              goto LABEL_104;
            v23 = *(_QWORD *)(v22 + 744);
            v158 = 0;
            v157 = 0;
            v24 = *(unsigned int *)(v23 + 40);
            v149 = 0;
            v159 = *(_QWORD *)(a1 + 368) + v140;
            v155 = (float)(int)(v24 - ((*(_QWORD *)(v23 + 80) == 0) - 1));
            v151 = v155;
            while ( 1 )
            {
              v25 = v158 + v23;
              v26 = *(_QWORD *)(v25 + 32);
              v153 = v25;
              v161 = (__int64 *)(v26 + 8 * v24);
              if ( (__int64 *)v26 != v161 )
              {
                v156 = 0.0;
                v27 = *(__int64 **)(v25 + 32);
                v164 = 0.0;
                while ( 1 )
                {
                  while ( 1 )
                  {
                    v29 = v171;
                    v30 = *v27;
                    if ( v172 == v171 )
                    {
                      v28 = &v171[8 * HIDWORD(v173)];
                      if ( v171 == v28 )
                      {
                        v26 = (__int64)v171;
                      }
                      else
                      {
                        do
                        {
                          if ( v30 == *v29 )
                            break;
                          ++v29;
                        }
                        while ( v28 != (_BYTE *)v29 );
                        v26 = (__int64)&v171[8 * HIDWORD(v173)];
                      }
LABEL_47:
                      while ( (_QWORD *)v26 != v29 )
                      {
                        if ( *v29 < 0xFFFFFFFFFFFFFFFELL )
                          goto LABEL_37;
                        ++v29;
                      }
                      if ( v29 != (_QWORD *)v28 )
                        goto LABEL_38;
                    }
                    else
                    {
                      v28 = &v172[8 * (unsigned int)v173];
                      v29 = sub_16CC9F0((__int64)&v170, *v27);
                      if ( v30 == *v29 )
                      {
                        if ( v172 == v171 )
                        {
                          v25 = HIDWORD(v173);
                          v26 = (__int64)&v172[8 * HIDWORD(v173)];
                        }
                        else
                        {
                          v25 = (unsigned int)v173;
                          v26 = (__int64)&v172[8 * (unsigned int)v173];
                        }
                        goto LABEL_47;
                      }
                      if ( v172 == v171 )
                      {
                        v29 = &v172[8 * HIDWORD(v173)];
                        v26 = (__int64)v29;
                        goto LABEL_47;
                      }
                      v26 = (unsigned int)v173;
                      v29 = &v172[8 * (unsigned int)v173];
LABEL_37:
                      if ( v29 != (_QWORD *)v28 )
                        goto LABEL_38;
                    }
                    v31 = v167;
                    if ( !v169 )
                    {
                      ++v166;
LABEL_110:
                      v143 = v169;
                      v55 = (((((((2 * v169 - 1) | ((unsigned __int64)(2 * v169 - 1) >> 1)) >> 2)
                              | (2 * v169 - 1)
                              | ((unsigned __int64)(2 * v169 - 1) >> 1)) >> 4)
                            | (((2 * v169 - 1) | ((unsigned __int64)(2 * v169 - 1) >> 1)) >> 2)
                            | (2 * v169 - 1)
                            | ((unsigned __int64)(2 * v169 - 1) >> 1)) >> 8)
                          | (((((2 * v169 - 1) | ((unsigned __int64)(2 * v169 - 1) >> 1)) >> 2)
                            | (2 * v169 - 1)
                            | ((unsigned __int64)(2 * v169 - 1) >> 1)) >> 4)
                          | (((2 * v169 - 1) | ((unsigned __int64)(2 * v169 - 1) >> 1)) >> 2)
                          | (2 * v169 - 1)
                          | ((unsigned __int64)(2 * v169 - 1) >> 1);
                      v56 = ((v55 >> 16) | v55) + 1;
                      if ( (unsigned int)v56 < 0x40 )
                        LODWORD(v56) = 64;
                      v169 = v56;
                      v57 = (__int64 *)sub_22077B0(16LL * (unsigned int)v56);
                      v167 = v57;
                      v58 = v57;
                      if ( v31 )
                      {
                        v168 = 0;
                        v59 = &v31[2 * v143];
                        for ( i = &v57[2 * v169]; i != v58; v58 += 2 )
                        {
                          if ( v58 )
                            *v58 = -8;
                        }
                        for ( j = v31; v59 != j; j += 2 )
                        {
                          v62 = *j;
                          if ( *j != -16 && v62 != -8 )
                          {
                            if ( !v169 )
                            {
                              MEMORY[0] = *j;
                              BUG();
                            }
                            v63 = 1;
                            v64 = 0;
                            v65 = (v169 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
                            v66 = &v167[2 * v65];
                            v67 = *v66;
                            if ( *v66 != v62 )
                            {
                              while ( v67 != -8 )
                              {
                                if ( v67 == -16 && !v64 )
                                  v64 = v66;
                                v65 = (v169 - 1) & (v63 + v65);
                                v66 = &v167[2 * v65];
                                v67 = *v66;
                                if ( v62 == *v66 )
                                  goto LABEL_122;
                                ++v63;
                              }
                              if ( v64 )
                                v66 = v64;
                            }
LABEL_122:
                            *v66 = v62;
                            *((_DWORD *)v66 + 2) = *((_DWORD *)j + 2);
                            LODWORD(v168) = v168 + 1;
                          }
                        }
                        j___libc_free_0(v31);
                        v58 = v167;
                        v68 = v169;
                        v42 = v168 + 1;
                      }
                      else
                      {
                        v168 = 0;
                        v68 = v169;
                        for ( k = &v57[2 * v169]; k != v57; v57 += 2 )
                        {
                          if ( v57 )
                            *v57 = -8;
                        }
                        v42 = 1;
                      }
                      if ( !v68 )
                        goto LABEL_308;
                      v69 = v68 - 1;
                      v70 = v69 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
                      v40 = &v58[2 * v70];
                      v71 = *v40;
                      if ( v30 != *v40 )
                      {
                        v72 = 1;
                        v73 = 0;
                        while ( v71 != -8 )
                        {
                          if ( v71 == -16 && !v73 )
                            v73 = v40;
                          v70 = v69 & (v72 + v70);
                          v40 = &v58[2 * v70];
                          v71 = *v40;
                          if ( v30 == *v40 )
                            goto LABEL_78;
                          ++v72;
                        }
                        if ( v73 )
                          v40 = v73;
                      }
                      goto LABEL_78;
                    }
                    v32 = v169 - 1;
                    v33 = (float *)&v167[2 * ((unsigned int)v32 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4)))];
                    if ( v30 == *(_QWORD *)v33 )
                    {
                      v164 = (float)(v33[2] / sub_1993D70(v159, v30)) + v164;
                      if ( *(_WORD *)(v30 + 24) != 7 )
                        goto LABEL_38;
                      goto LABEL_52;
                    }
                    v142 = v32 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
                    v38 = *(_QWORD *)v33;
                    v39 = &v167[2 * v142];
                    v40 = 0;
                    v41 = 1;
LABEL_73:
                    if ( v38 != -8 )
                      break;
                    if ( !v40 )
                      v40 = v39;
                    ++v166;
                    v42 = v168 + 1;
                    if ( 4 * ((int)v168 + 1) >= 3 * v169 )
                      goto LABEL_110;
                    if ( v169 - HIDWORD(v168) - v42 <= v169 >> 3 )
                    {
                      v144 = v169;
                      v74 = (((((((((v32 | (v32 >> 1)) >> 2) | v32 | (v32 >> 1)) >> 4)
                               | ((v32 | (v32 >> 1)) >> 2)
                               | v32
                               | (v32 >> 1)) >> 8)
                             | ((((v32 | (v32 >> 1)) >> 2) | v32 | (v32 >> 1)) >> 4)
                             | ((v32 | (v32 >> 1)) >> 2)
                             | v32
                             | (v32 >> 1)) >> 16)
                           | ((((((v32 | (v32 >> 1)) >> 2) | v32 | (v32 >> 1)) >> 4)
                             | ((v32 | (v32 >> 1)) >> 2)
                             | v32
                             | (v32 >> 1)) >> 8)
                           | ((((v32 | (v32 >> 1)) >> 2) | v32 | (v32 >> 1)) >> 4)
                           | ((v32 | (v32 >> 1)) >> 2)
                           | v32
                           | (v32 >> 1))
                          + 1;
                      if ( v74 < 0x40 )
                        v74 = 64;
                      v169 = v74;
                      v75 = (__int64 *)sub_22077B0(16LL * v74);
                      v76 = ((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4);
                      v167 = v75;
                      v77 = v75;
                      if ( v31 )
                      {
                        v168 = 0;
                        for ( m = &v75[2 * v169]; m != v77; v77 += 2 )
                        {
                          if ( v77 )
                            *v77 = -8;
                        }
                        v79 = v31;
                        do
                        {
                          v80 = *v79;
                          if ( *v79 != -8 && v80 != -16 )
                          {
                            if ( !v169 )
                            {
                              MEMORY[0] = *v79;
                              BUG();
                            }
                            v81 = 1;
                            v82 = 0;
                            v83 = (v169 - 1) & (((unsigned int)v80 >> 9) ^ ((unsigned int)v80 >> 4));
                            v84 = &v167[2 * v83];
                            v85 = *v84;
                            if ( v80 != *v84 )
                            {
                              while ( v85 != -8 )
                              {
                                if ( !v82 && v85 == -16 )
                                  v82 = v84;
                                v83 = (v169 - 1) & (v81 + v83);
                                v84 = &v167[2 * v83];
                                v85 = *v84;
                                if ( v80 == *v84 )
                                  goto LABEL_144;
                                ++v81;
                              }
                              if ( v82 )
                                v84 = v82;
                            }
LABEL_144:
                            *v84 = v80;
                            *((_DWORD *)v84 + 2) = *((_DWORD *)v79 + 2);
                            LODWORD(v168) = v168 + 1;
                          }
                          v79 += 2;
                        }
                        while ( &v31[2 * v144] != v79 );
                        j___libc_free_0(v31);
                        v77 = v167;
                        v86 = v169;
                        v76 = ((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4);
                        v42 = v168 + 1;
                      }
                      else
                      {
                        v168 = 0;
                        v86 = v169;
                        for ( n = &v75[2 * v169]; n != v75; v75 += 2 )
                        {
                          if ( v75 )
                            *v75 = -8;
                        }
                        v42 = 1;
                      }
                      if ( !v86 )
                      {
LABEL_308:
                        LODWORD(v168) = v168 + 1;
                        BUG();
                      }
                      v87 = v86 - 1;
                      v88 = 0;
                      v89 = 1;
                      v90 = v87 & v76;
                      v40 = &v77[2 * v90];
                      v91 = *v40;
                      if ( v30 != *v40 )
                      {
                        while ( v91 != -8 )
                        {
                          if ( v91 == -16 && !v88 )
                            v88 = v40;
                          v90 = v87 & (v89 + v90);
                          v40 = &v77[2 * v90];
                          v91 = *v40;
                          if ( v30 == *v40 )
                            goto LABEL_78;
                          ++v89;
                        }
                        if ( v88 )
                          v40 = v88;
                      }
                    }
LABEL_78:
                    LODWORD(v168) = v42;
                    if ( *v40 != -8 )
                      --HIDWORD(v168);
                    *v40 = v30;
                    *((_DWORD *)v40 + 2) = 0;
                    v164 = (float)(0.0 / sub_1993D70(v159, v30)) + v164;
                    if ( *(_WORD *)(v30 + 24) != 7 )
                      goto LABEL_38;
                    v43 = v169;
                    if ( !v169 )
                    {
                      ++v166;
                      goto LABEL_83;
                    }
                    v31 = v167;
                    v104 = ((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4);
                    LODWORD(v26) = v169 - 1;
                    LODWORD(v25) = v104 & (v169 - 1);
                    v33 = (float *)&v167[2 * (unsigned int)v25];
                    a6 = *(_QWORD *)v33;
                    if ( v30 != *(_QWORD *)v33 )
                    {
LABEL_175:
                      v105 = 1;
                      v106 = 0;
                      while ( a6 != -8 )
                      {
                        if ( !v106 && a6 == -16 )
                          v106 = v33;
                        v25 = (unsigned int)v26 & ((_DWORD)v25 + v105);
                        v33 = (float *)&v31[2 * v25];
                        a6 = *(_QWORD *)v33;
                        if ( v30 == *(_QWORD *)v33 )
                          goto LABEL_52;
                        ++v105;
                      }
                      if ( v106 )
                        v33 = v106;
                      ++v166;
                      v45 = v168 + 1;
                      if ( 4 * ((int)v168 + 1) >= 3 * v43 )
                      {
LABEL_83:
                        sub_19AD4A0((__int64)&v166, 2 * v43);
                        if ( !v169 )
                          goto LABEL_310;
                        v44 = (v169 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
                        v45 = v168 + 1;
                        v33 = (float *)&v167[2 * v44];
                        v46 = *(_QWORD *)v33;
                        if ( v30 != *(_QWORD *)v33 )
                        {
                          v130 = 1;
                          v131 = 0;
                          while ( v46 != -8 )
                          {
                            if ( !v131 && v46 == -16 )
                              v131 = v33;
                            v44 = (v169 - 1) & (v130 + v44);
                            v33 = (float *)&v167[2 * v44];
                            v46 = *(_QWORD *)v33;
                            if ( v30 == *(_QWORD *)v33 )
                              goto LABEL_85;
                            ++v130;
                          }
                          if ( v131 )
                            v33 = v131;
                        }
                      }
                      else if ( v43 - (v45 + HIDWORD(v168)) <= v43 >> 3 )
                      {
                        v145 = v104;
                        sub_19AD4A0((__int64)&v166, v43);
                        if ( !v169 )
                        {
LABEL_310:
                          LODWORD(v168) = v168 + 1;
                          BUG();
                        }
                        v107 = 1;
                        v108 = 0;
                        v45 = v168 + 1;
                        v109 = (v169 - 1) & v145;
                        v33 = (float *)&v167[2 * v109];
                        v110 = *(_QWORD *)v33;
                        if ( v30 != *(_QWORD *)v33 )
                        {
                          while ( v110 != -8 )
                          {
                            if ( v110 == -16 && !v108 )
                              v108 = v33;
                            v109 = (v169 - 1) & (v107 + v109);
                            v33 = (float *)&v167[2 * v109];
                            v110 = *(_QWORD *)v33;
                            if ( v30 == *(_QWORD *)v33 )
                              goto LABEL_85;
                            ++v107;
                          }
                          if ( v108 )
                            v33 = v108;
                        }
                      }
LABEL_85:
                      LODWORD(v168) = v45;
                      if ( *(_QWORD *)v33 != -8 )
                        --HIDWORD(v168);
                      *(_QWORD *)v33 = v30;
                      v34 = 0.0;
                      v33[2] = 0.0;
                      goto LABEL_53;
                    }
LABEL_52:
                    v34 = v33[2];
LABEL_53:
                    ++v27;
                    v156 = (float)(v34 / sub_1993D70(v159, v30)) + v156;
                    if ( v161 == v27 )
                      goto LABEL_54;
                  }
                  if ( v40 || v38 != -16 )
                    v39 = v40;
                  v142 = v32 & (v142 + v41);
                  v139 = (float *)&v167[2 * v142];
                  v38 = *(_QWORD *)v139;
                  if ( v30 != *(_QWORD *)v139 )
                  {
                    ++v41;
                    v40 = v39;
                    v39 = &v167[2 * v142];
                    goto LABEL_73;
                  }
                  v136 = v169 - 1;
                  v137 = v32 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
                  v150 = v169;
                  v138 = *(_QWORD *)v33;
                  v120 = sub_1993D70(v159, v30);
                  v104 = ((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4);
                  a6 = v138;
                  v43 = v150;
                  v25 = v137;
                  v26 = v136;
                  v164 = (float)(v139[2] / v120) + v164;
                  if ( *(_WORD *)(v30 + 24) == 7 )
                    goto LABEL_175;
LABEL_38:
                  if ( v161 == ++v27 )
                    goto LABEL_54;
                }
              }
              v156 = 0.0;
              v164 = 0.0;
LABEL_54:
              v35 = *(_QWORD *)(v153 + 80);
              v165 = v35;
              if ( !v35 )
                goto LABEL_60;
              v36 = v171;
              if ( v172 == v171 )
              {
                v37 = &v171[8 * HIDWORD(v173)];
                if ( v171 == v37 )
                {
                  v26 = (__int64)v171;
                }
                else
                {
                  do
                  {
                    if ( v35 == *v36 )
                      break;
                    ++v36;
                  }
                  while ( v37 != (_BYTE *)v36 );
                  v26 = (__int64)&v171[8 * HIDWORD(v173)];
                }
              }
              else
              {
                v37 = &v172[8 * (unsigned int)v173];
                v36 = sub_16CC9F0((__int64)&v170, v35);
                if ( v35 == *v36 )
                {
                  if ( v172 == v171 )
                  {
                    v25 = HIDWORD(v173);
                    v26 = (__int64)&v172[8 * HIDWORD(v173)];
                  }
                  else
                  {
                    v25 = (unsigned int)v173;
                    v26 = (__int64)&v172[8 * (unsigned int)v173];
                  }
                }
                else
                {
                  if ( v172 != v171 )
                  {
                    v26 = (unsigned int)v173;
                    v36 = &v172[8 * (unsigned int)v173];
LABEL_59:
                    if ( v37 != (_BYTE *)v36 )
                      goto LABEL_60;
                    goto LABEL_97;
                  }
                  v36 = &v172[8 * HIDWORD(v173)];
                  v26 = (__int64)v36;
                }
              }
              if ( v36 == (_QWORD *)v26 )
                goto LABEL_59;
              do
              {
                if ( *v36 < 0xFFFFFFFFFFFFFFFELL )
                  goto LABEL_59;
                ++v36;
              }
              while ( (_QWORD *)v26 != v36 );
              if ( v37 != (_BYTE *)v36 )
                goto LABEL_60;
LABEL_97:
              v47 = v169;
              if ( !v169 )
              {
                ++v166;
LABEL_246:
                v129 = 2 * v169;
                goto LABEL_244;
              }
              v48 = v165;
              v49 = v169 - 1;
              v50 = v165;
              v51 = (v169 - 1) & (((unsigned int)v165 >> 9) ^ ((unsigned int)v165 >> 4));
              v52 = (__int64)&v167[2 * v51];
              if ( v165 == *(_QWORD *)v52 )
              {
                v164 = (float)(*(float *)(v52 + 8) / sub_1993D70(v159, v165)) + v164;
                if ( *(_WORD *)(v48 + 24) != 7 )
                  goto LABEL_60;
                goto LABEL_100;
              }
              v121 = *(_QWORD *)v52;
              v122 = (__int64)&v167[2 * (v49 & (((unsigned int)v165 >> 9) ^ ((unsigned int)v165 >> 4)))];
              v123 = (v169 - 1) & (((unsigned int)v165 >> 9) ^ ((unsigned int)v165 >> 4));
              v124 = 1;
              v125 = 0;
              while ( v121 != -8 )
              {
                if ( !v125 && v121 == -16 )
                  v125 = v122;
                v123 = v49 & (v124 + v123);
                v122 = (__int64)&v167[2 * v123];
                v121 = *(_QWORD *)v122;
                if ( v165 == *(_QWORD *)v122 )
                {
                  v146 = *(_QWORD *)v52;
                  v154 = (float *)&v167[2 * v123];
                  v162 = (__int64)v167;
                  v135 = sub_1993D70(v159, v165);
                  v25 = v162;
                  v132 = v146;
                  v26 = v47 - 1;
                  v164 = (float)(v154[2] / v135) + v164;
                  if ( *(_WORD *)(v48 + 24) == 7 )
                    goto LABEL_283;
                  goto LABEL_60;
                }
                ++v124;
              }
              if ( v125 )
                v122 = v125;
              ++v166;
              v126 = v168 + 1;
              if ( 4 * ((int)v168 + 1) >= 3 * v169 )
                goto LABEL_246;
              if ( v169 - HIDWORD(v168) - v126 > v169 >> 3 )
                goto LABEL_233;
              v129 = v169;
LABEL_244:
              sub_19AD4A0((__int64)&v166, v129);
              sub_19A7EE0((__int64)&v166, &v165, &v176);
              v122 = v176;
              v50 = v165;
              v126 = v168 + 1;
LABEL_233:
              LODWORD(v168) = v126;
              if ( *(_QWORD *)v122 != -8 )
                --HIDWORD(v168);
              *(_QWORD *)v122 = v50;
              *(_DWORD *)(v122 + 8) = 0;
              v48 = v165;
              v164 = (float)(0.0 / sub_1993D70(v159, v165)) + v164;
              if ( *(_WORD *)(v48 + 24) == 7 )
              {
                v47 = v169;
                if ( !v169 )
                {
                  ++v166;
                  goto LABEL_238;
                }
                LODWORD(v26) = v169 - 1;
                v25 = (__int64)v167;
                v51 = (v169 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
                v52 = (__int64)&v167[2 * v51];
                v132 = *(_QWORD *)v52;
                if ( v48 != *(_QWORD *)v52 )
                {
LABEL_283:
                  v133 = 1;
                  v134 = 0;
                  while ( v132 != -8 )
                  {
                    if ( v132 == -16 && !v134 )
                      v134 = v52;
                    v51 = v26 & (v133 + v51);
                    v52 = v25 + 16LL * v51;
                    v132 = *(_QWORD *)v52;
                    if ( *(_QWORD *)v52 == v48 )
                      goto LABEL_100;
                    ++v133;
                  }
                  if ( v134 )
                    v52 = v134;
                  ++v166;
                  v128 = v168 + 1;
                  if ( 4 * ((int)v168 + 1) >= 3 * v47 )
                  {
LABEL_238:
                    v127 = 2 * v47;
                    goto LABEL_239;
                  }
                  if ( v47 - (v128 + HIDWORD(v168)) <= v47 >> 3 )
                  {
                    v127 = v47;
LABEL_239:
                    sub_19AD4A0((__int64)&v166, v127);
                    sub_19A7EE0((__int64)&v166, &v165, &v176);
                    v52 = v176;
                    v48 = v165;
                    v128 = v168 + 1;
                  }
                  LODWORD(v168) = v128;
                  if ( *(_QWORD *)v52 != -8 )
                    --HIDWORD(v168);
                  *(_QWORD *)v52 = v48;
                  v53 = 0.0;
                  *(_DWORD *)(v52 + 8) = 0;
                  v48 = v165;
LABEL_101:
                  v156 = (float)(v53 / sub_1993D70(v159, v48)) + v156;
                  goto LABEL_60;
                }
LABEL_100:
                v53 = *(float *)(v52 + 8);
                goto LABEL_101;
              }
LABEL_60:
              if ( v155 > v164 )
              {
                v149 = v157;
                v151 = v156;
                v155 = v164;
              }
              else if ( v164 == v155 && v151 > v156 )
              {
                v151 = v156;
                v155 = v164;
                v149 = v157;
              }
              ++v157;
              v158 += 96;
              if ( v157 == v152 )
              {
                if ( !v149 )
                  goto LABEL_154;
                v111 = *(_QWORD *)(v159 + 744);
                v112 = v111 + 96 * v149;
                v113 = v112 + 32;
                v176 = *(_QWORD *)v112;
                v177 = *(_QWORD *)(v112 + 8);
                v178 = *(_BYTE *)(v112 + 16);
                v179 = *(_QWORD *)(v112 + 24);
                v180[0] = v181;
                v180[1] = (char *)0x400000000LL;
                if ( *(_DWORD *)(v112 + 40) )
                {
                  sub_19931B0((__int64)v180, (char **)(v112 + 32), v26, v25, v113, a6);
                  v113 = v112 + 32;
                }
                v182 = *(_QWORD *)(v112 + 80);
                v183 = *(_QWORD *)(v112 + 88);
                *(_QWORD *)v112 = *(_QWORD *)v111;
                *(_QWORD *)(v112 + 8) = *(_QWORD *)(v111 + 8);
                *(_BYTE *)(v112 + 16) = *(_BYTE *)(v111 + 16);
                *(_QWORD *)(v112 + 24) = *(_QWORD *)(v111 + 24);
                sub_19931B0(v113, (char **)(v111 + 32), v26, v25, v113, a6);
                *(_QWORD *)(v112 + 80) = *(_QWORD *)(v111 + 80);
                *(_QWORD *)(v112 + 88) = *(_QWORD *)(v111 + 88);
                *(_QWORD *)v111 = v176;
                *(_QWORD *)(v111 + 8) = v177;
                *(_BYTE *)(v111 + 16) = v178;
                *(_QWORD *)(v111 + 24) = v179;
                sub_19931B0(v111 + 32, v180, v114, v115, v116, v117);
                *(_QWORD *)(v111 + 80) = v182;
                *(_QWORD *)(v111 + 88) = v183;
                v95 = v180[0];
                if ( v180[0] != v181 )
                {
LABEL_156:
                  _libc_free((unsigned __int64)v95);
                  v92 = *(_DWORD *)(v159 + 752);
                  if ( v92 != 1 )
                    goto LABEL_155;
                }
                else
                {
LABEL_154:
                  while ( 2 )
                  {
                    v92 = *(_DWORD *)(v159 + 752);
                    if ( v92 != 1 )
                    {
LABEL_155:
                      v93 = (unsigned int)(v92 - 1);
                      *(_DWORD *)(v159 + 752) = v93;
                      v94 = *(_QWORD *)(v159 + 744) + 96 * v93;
                      v95 = *(char **)(v94 + 32);
                      if ( v95 == (char *)(v94 + 48) )
                        continue;
                      goto LABEL_156;
                    }
                    break;
                  }
                }
                sub_1996C50(v159, v148, a1 + 32128);
                v96 = *(_QWORD *)(v159 + 744);
                v97 = *(__int64 **)(v96 + 32);
                v98 = &v97[*(unsigned int *)(v96 + 40)];
                if ( v97 != v98 )
                {
                  a6 = (unsigned __int64)v172;
                  v99 = (__int64 *)v171;
                  do
                  {
                    v100 = *v97;
                    if ( v99 != (__int64 *)a6 )
                      goto LABEL_159;
                    v101 = &v99[HIDWORD(v173)];
                    if ( v101 != v99 )
                    {
                      v102 = v99;
                      v103 = 0;
                      while ( v100 != *v102 )
                      {
                        if ( *v102 == -2 )
                          v103 = v102;
                        if ( v101 == ++v102 )
                        {
                          if ( !v103 )
                            goto LABEL_172;
                          *v103 = v100;
                          a6 = (unsigned __int64)v172;
                          --v174;
                          v99 = (__int64 *)v171;
                          ++v170;
                          goto LABEL_160;
                        }
                      }
                      goto LABEL_160;
                    }
LABEL_172:
                    if ( HIDWORD(v173) < (unsigned int)v173 )
                    {
                      ++HIDWORD(v173);
                      *v101 = v100;
                      v99 = (__int64 *)v171;
                      ++v170;
                      a6 = (unsigned __int64)v172;
                    }
                    else
                    {
LABEL_159:
                      sub_16CCBA0((__int64)&v170, v100);
                      a6 = (unsigned __int64)v172;
                      v99 = (__int64 *)v171;
                    }
LABEL_160:
                    ++v97;
                  }
                  while ( v98 != v97 );
                }
                v54 = *(_QWORD *)(v96 + 80);
                if ( v54 )
                  sub_199AF80((__int64)&v176, (__int64)&v170, v54);
LABEL_104:
                ++v148;
                v140 += 1984;
                if ( v148 == v141 )
                  goto LABEL_105;
                goto LABEL_30;
              }
              v23 = *(_QWORD *)(v159 + 744);
              v24 = *(unsigned int *)(v23 + v158 + 40);
            }
          }
        }
      }
    }
  }
}
