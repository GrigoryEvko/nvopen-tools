// Function: sub_29A0E30
// Address: 0x29a0e30
//
__int64 __fastcall sub_29A0E30(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r14
  __int64 v4; // rax
  __int64 v5; // r12
  int v6; // ebx
  unsigned __int8 v7; // al
  _QWORD *v8; // rdx
  _BYTE *v9; // rax
  unsigned __int8 v10; // dl
  unsigned __int8 v11; // dl
  char *v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rsi
  unsigned int v15; // r8d
  __int64 v16; // r15
  unsigned __int64 v17; // rax
  size_t v18; // r10
  char *v19; // rdi
  __m128i *v20; // rcx
  int v21; // r11d
  unsigned __int64 v22; // rbx
  unsigned int v23; // r8d
  unsigned int i; // ebx
  __int64 v25; // r9
  bool v26; // al
  const void *v27; // rsi
  unsigned int v28; // ebx
  int v29; // eax
  int v30; // edx
  int v31; // eax
  int v32; // esi
  int v33; // esi
  _DWORD *v34; // rcx
  int v35; // eax
  unsigned int *v36; // rcx
  unsigned int v37; // esi
  unsigned int v38; // eax
  _QWORD *v39; // rax
  void *v40; // rdx
  _QWORD *v41; // rbx
  _QWORD *v42; // r13
  char v43; // al
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // r14
  unsigned int v47; // r15d
  unsigned __int8 v48; // al
  _QWORD *v49; // rdx
  _BYTE *v50; // rax
  unsigned __int8 v51; // dl
  unsigned __int8 v52; // dl
  char *v53; // rdi
  void *v54; // rdx
  void *v55; // rsi
  int v56; // r8d
  unsigned __int64 v57; // rax
  __int64 v58; // r9
  int v59; // r11d
  int v60; // r8d
  void *v61; // rdx
  unsigned __int64 v62; // rax
  char *v63; // rdi
  unsigned int j; // ecx
  __int64 v65; // r10
  const void *v66; // r15
  bool v67; // al
  unsigned int v68; // ecx
  int v69; // esi
  int v70; // eax
  __int64 v71; // rdi
  __int64 v72; // rsi
  __int64 v73; // rdi
  __int64 v74; // rsi
  __int64 v75; // rax
  __int64 v76; // rbx
  __int64 v77; // r12
  unsigned int v78; // esi
  __int64 v79; // r10
  __int64 v80; // r15
  int v81; // ebx
  _QWORD *v82; // r11
  unsigned int v83; // edi
  _QWORD *v84; // rcx
  _QWORD *v85; // rax
  unsigned int v86; // r15d
  __int64 v87; // rbx
  unsigned __int64 v88; // rax
  int v89; // r10d
  _DWORD *v90; // r9
  unsigned int v91; // r11d
  unsigned __int64 v92; // rax
  char *v93; // rdi
  size_t v94; // rdx
  unsigned int v95; // r15d
  __int64 v96; // r8
  const void *v97; // rcx
  bool v98; // al
  unsigned int v99; // r15d
  int v100; // eax
  int v101; // eax
  unsigned int v102; // r8d
  __int64 v103; // r15
  __int64 v104; // rbx
  unsigned __int64 v105; // rax
  const void *v106; // rdi
  size_t v107; // rdx
  int v108; // r10d
  unsigned int v109; // r8d
  unsigned int v110; // ebx
  __int64 v111; // r9
  const void *v112; // rsi
  unsigned int v113; // ebx
  int v114; // eax
  int v115; // eax
  int v116; // eax
  __int32 v117; // eax
  int v118; // edx
  int v119; // edx
  __int64 v120; // rdi
  unsigned int v121; // ecx
  __int64 v122; // rsi
  int v123; // r10d
  _QWORD *v124; // r9
  __int64 v125; // rax
  __int64 v126; // rsi
  unsigned __int8 *v127; // rsi
  int v128; // eax
  unsigned int *v129; // rax
  unsigned int v130; // esi
  _QWORD *v131; // rax
  __int64 v132; // rdx
  unsigned __int8 v133; // r14
  __int64 v134; // rsi
  unsigned __int8 *v135; // rsi
  int v136; // esi
  __int64 v137; // rax
  int v138; // edx
  unsigned int v139; // edx
  int v140; // eax
  int v141; // edx
  int v142; // edx
  __int64 v143; // rdi
  int v144; // r10d
  unsigned int v145; // ecx
  __int64 v146; // rsi
  __m128i *v147; // [rsp+8h] [rbp-138h]
  int v148; // [rsp+10h] [rbp-130h]
  int v149; // [rsp+10h] [rbp-130h]
  size_t v150; // [rsp+10h] [rbp-130h]
  int v151; // [rsp+10h] [rbp-130h]
  unsigned int v152; // [rsp+1Ch] [rbp-124h]
  unsigned int v153; // [rsp+1Ch] [rbp-124h]
  int v154; // [rsp+1Ch] [rbp-124h]
  unsigned int v155; // [rsp+1Ch] [rbp-124h]
  size_t v156; // [rsp+20h] [rbp-120h]
  _DWORD *v157; // [rsp+20h] [rbp-120h]
  unsigned int v158; // [rsp+20h] [rbp-120h]
  _DWORD *v159; // [rsp+20h] [rbp-120h]
  size_t v160; // [rsp+28h] [rbp-118h]
  int v161; // [rsp+28h] [rbp-118h]
  size_t v162; // [rsp+28h] [rbp-118h]
  __int64 v164; // [rsp+30h] [rbp-110h]
  __int64 v165; // [rsp+38h] [rbp-108h]
  void *v166; // [rsp+38h] [rbp-108h]
  unsigned int v167; // [rsp+40h] [rbp-100h]
  unsigned int v168; // [rsp+48h] [rbp-F8h]
  int v169; // [rsp+48h] [rbp-F8h]
  const void *v170; // [rsp+48h] [rbp-F8h]
  __int64 v171; // [rsp+48h] [rbp-F8h]
  unsigned __int8 v172; // [rsp+50h] [rbp-F0h]
  int v173; // [rsp+50h] [rbp-F0h]
  unsigned int v174; // [rsp+50h] [rbp-F0h]
  _QWORD *v175; // [rsp+58h] [rbp-E8h]
  size_t v176; // [rsp+58h] [rbp-E8h]
  _QWORD *v177; // [rsp+60h] [rbp-E0h]
  _QWORD *v178; // [rsp+68h] [rbp-D8h]
  _QWORD *v179; // [rsp+68h] [rbp-D8h]
  __int64 v180; // [rsp+78h] [rbp-C8h] BYREF
  __int64 v181[2]; // [rsp+80h] [rbp-C0h] BYREF
  void *s1[2]; // [rsp+90h] [rbp-B0h] BYREF
  unsigned int v183; // [rsp+A0h] [rbp-A0h]
  __int64 v184; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v185; // [rsp+B8h] [rbp-88h]
  __int64 v186; // [rsp+C0h] [rbp-80h]
  unsigned int v187; // [rsp+C8h] [rbp-78h]
  __int64 v188; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v189; // [rsp+D8h] [rbp-68h]
  __int64 v190; // [rsp+E0h] [rbp-60h]
  unsigned int v191; // [rsp+E8h] [rbp-58h]
  size_t n[2]; // [rsp+F0h] [rbp-50h] BYREF
  __int64 v193; // [rsp+100h] [rbp-40h]
  __int64 v194; // [rsp+108h] [rbp-38h]

  v172 = byte_50079E8;
  if ( byte_50079E8 || !sub_B92180(a1) )
    return 0;
  if ( LOBYTE(qword_4F813A8[8]) )
    sub_2A61200(*(_QWORD *)(a1 + 40));
  v186 = 0;
  v184 = 0;
  v2 = *(_QWORD **)(a1 + 80);
  v185 = 0;
  v187 = 0;
  v188 = 0;
  v189 = 0;
  v190 = 0;
  v191 = 0;
  v177 = v2;
  v165 = a1 + 72;
  if ( v2 != (_QWORD *)(a1 + 72) )
  {
    while ( 1 )
    {
      if ( !v177 )
        BUG();
      v3 = (_QWORD *)v177[4];
      v175 = v177 - 3;
      v178 = v177 + 3;
      if ( v3 == v177 + 3 )
        goto LABEL_50;
      v167 = ((unsigned int)((_DWORD)v177 - 24) >> 9) ^ ((unsigned int)((_DWORD)v177 - 24) >> 4);
      do
      {
        while ( 1 )
        {
          if ( !v3 )
            BUG();
          if ( *((_BYTE *)v3 - 24) != 85 )
            break;
          v125 = *(v3 - 7);
          if ( !v125 || *(_BYTE *)v125 || *(_QWORD *)(v125 + 24) != v3[7] || (*(_BYTE *)(v125 + 33) & 0x20) == 0 )
            break;
          if ( (unsigned int)(*(_DWORD *)(v125 + 36) - 238) > 7 )
            goto LABEL_49;
          if ( ((1LL << (*(_BYTE *)(v125 + 36) + 18)) & 0xAD) != 0 )
            break;
          v3 = (_QWORD *)v3[1];
          if ( v178 == v3 )
            goto LABEL_50;
        }
        v4 = sub_B10CD0((__int64)(v3 + 3));
        v5 = v4;
        if ( !v4 )
          goto LABEL_49;
        v6 = *(_DWORD *)(v4 + 4);
        v7 = *(_BYTE *)(v4 - 16);
        if ( (v7 & 2) != 0 )
          v8 = *(_QWORD **)(v5 - 32);
        else
          v8 = (_QWORD *)(v5 - 16 - 8LL * ((v7 >> 2) & 0xF));
        v9 = (_BYTE *)*v8;
        if ( *(_BYTE *)*v8 != 16 )
        {
          v10 = *(v9 - 16);
          if ( (v10 & 2) != 0 )
          {
            v9 = (_BYTE *)**((_QWORD **)v9 - 4);
            if ( !v9 )
              goto LABEL_165;
          }
          else
          {
            v9 = *(_BYTE **)&v9[-8 * ((v10 >> 2) & 0xF) - 16];
            if ( !v9 )
            {
LABEL_165:
              v14 = 0;
              v12 = (char *)byte_3F871B3;
              goto LABEL_22;
            }
          }
        }
        v11 = *(v9 - 16);
        if ( (v11 & 2) != 0 )
        {
          v12 = (char *)**((_QWORD **)v9 - 4);
          if ( !v12 )
            goto LABEL_169;
        }
        else
        {
          v12 = *(char **)&v9[-8 * ((v11 >> 2) & 0xF) - 16];
          if ( !v12 )
          {
LABEL_169:
            v14 = 0;
            goto LABEL_22;
          }
        }
        v12 = (char *)sub_B91420((__int64)v12);
        v14 = v13;
LABEL_22:
        v15 = v187;
        n[0] = (size_t)v12;
        n[1] = v14;
        LODWORD(v193) = v6;
        if ( !v187 )
        {
          ++v184;
          s1[0] = 0;
          goto LABEL_140;
        }
        v168 = v187;
        v16 = v185;
        v17 = sub_C94890(v12, v14);
        v18 = n[1];
        v19 = (char *)n[0];
        v20 = 0;
        v21 = 1;
        v22 = 0xBF58476D1CE4E5B9LL * ((v17 << 32) | (unsigned int)(37 * v6));
        v23 = v168 - 1;
        for ( i = (v168 - 1) & ((v22 >> 31) ^ v22); ; i = v23 & v28 )
        {
          v25 = v16 + 56LL * i;
          v26 = v19 + 1 == 0;
          v27 = *(const void **)v25;
          if ( *(_QWORD *)v25 != -1 )
          {
            v26 = v19 + 2 == 0;
            if ( v27 != (const void *)-2LL )
            {
              if ( *(_QWORD *)(v25 + 8) != v18 )
                goto LABEL_27;
              if ( !v18 )
              {
                v30 = v193;
                if ( (_DWORD)v193 == *(_DWORD *)(v25 + 16) )
                  break;
LABEL_239:
                if ( v27 == (const void *)-2LL && *(_DWORD *)(v25 + 16) == -2 && !v20 )
                  v20 = (__m128i *)v25;
                goto LABEL_27;
              }
              v147 = v20;
              v148 = v21;
              v152 = v23;
              v156 = v18;
              v29 = memcmp(v19, v27, v18);
              v18 = v156;
              v23 = v152;
              v21 = v148;
              v20 = v147;
              v26 = v29 == 0;
              v25 = v16 + 56LL * i;
            }
          }
          if ( v26 )
          {
            v30 = v193;
            if ( (_DWORD)v193 == *(_DWORD *)(v25 + 16) )
              break;
          }
          if ( v27 != (const void *)-1LL )
            goto LABEL_239;
          if ( *(_DWORD *)(v25 + 16) == -1 )
          {
            v15 = v187;
            if ( !v20 )
              v20 = (__m128i *)v25;
            ++v184;
            v31 = v186 + 1;
            s1[0] = v20;
            if ( 4 * ((int)v186 + 1) < 3 * v187 )
            {
              if ( v187 - (v31 + HIDWORD(v186)) > v187 >> 3 )
              {
LABEL_142:
                LODWORD(v186) = v31;
                if ( v20->m128i_i64[0] != -1 || v20[1].m128i_i32[0] != -1 )
                  --HIDWORD(v186);
                v80 = (__int64)&v20[1].m128i_i64[1];
                *v20 = _mm_loadu_si128((const __m128i *)n);
                v117 = v193;
                v20[1].m128i_i64[1] = 0;
                v20[1].m128i_i32[0] = v117;
                v20[2].m128i_i64[0] = 0;
                v20[2].m128i_i64[1] = 0;
                v20[3].m128i_i32[0] = 0;
LABEL_146:
                ++*(_QWORD *)v80;
                v78 = 0;
LABEL_147:
                sub_E3B4A0(v80, 2 * v78);
                v118 = *(_DWORD *)(v80 + 24);
                if ( v118 )
                {
                  v119 = v118 - 1;
                  v120 = *(_QWORD *)(v80 + 8);
                  v121 = v119 & v167;
                  v82 = (_QWORD *)(v120 + 8LL * (v119 & v167));
                  v122 = *v82;
                  v101 = *(_DWORD *)(v80 + 16) + 1;
                  if ( v175 != (_QWORD *)*v82 )
                  {
                    v123 = 1;
                    v124 = 0;
                    while ( v122 != -4096 )
                    {
                      if ( !v124 && v122 == -8192 )
                        v124 = v82;
                      v121 = v119 & (v123 + v121);
                      v82 = (_QWORD *)(v120 + 8LL * v121);
                      v122 = *v82;
                      if ( v175 == (_QWORD *)*v82 )
                        goto LABEL_115;
                      ++v123;
                    }
                    goto LABEL_151;
                  }
                  goto LABEL_115;
                }
LABEL_273:
                ++*(_DWORD *)(v80 + 16);
                BUG();
              }
              v32 = v187;
LABEL_141:
              sub_29A03D0((__int64)&v184, v32);
              sub_299FF50((__int64)&v184, (char **)n, s1);
              v20 = (__m128i *)s1[0];
              v31 = v186 + 1;
              goto LABEL_142;
            }
LABEL_140:
            v32 = 2 * v15;
            goto LABEL_141;
          }
LABEL_27:
          v28 = v21 + i;
          ++v21;
        }
        v78 = *(_DWORD *)(v25 + 48);
        v79 = *(_QWORD *)(v25 + 32);
        v80 = v25 + 24;
        if ( !v78 )
          goto LABEL_146;
        v81 = 1;
        v82 = 0;
        v83 = (v78 - 1) & v167;
        v84 = (_QWORD *)(v79 + 8LL * v83);
        v85 = (_QWORD *)*v84;
        if ( v175 == (_QWORD *)*v84 )
        {
LABEL_99:
          if ( *(_DWORD *)(v25 + 40) == 1 )
            goto LABEL_49;
          v86 = v191;
          v173 = v30;
          if ( !v191 )
          {
            ++v188;
            s1[0] = 0;
            goto LABEL_178;
          }
          v87 = v189;
          v88 = sub_C94890((_QWORD *)n[0], n[1]);
          v89 = 1;
          v90 = 0;
          v91 = v86 - 1;
          v92 = 0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v173) | (v88 << 32));
          v93 = (char *)n[0];
          v94 = n[1];
          v95 = (v86 - 1) & ((v92 >> 31) ^ v92);
          while ( 2 )
          {
            v96 = v87 + 32LL * v95;
            v97 = *(const void **)v96;
            v98 = v93 + 1 == 0;
            if ( *(_QWORD *)v96 == -1 )
              goto LABEL_126;
            v98 = v93 + 2 == 0;
            if ( v97 == (const void *)-2LL )
              goto LABEL_126;
            if ( *(_QWORD *)(v96 + 8) != v94 )
            {
LABEL_105:
              v99 = v89 + v95;
              ++v89;
              v95 = v91 & v99;
              continue;
            }
            break;
          }
          if ( v94 )
          {
            v149 = v89;
            v153 = v91;
            v157 = v90;
            v160 = v94;
            v170 = *(const void **)v96;
            v114 = memcmp(v93, *(const void **)v96, v94);
            v97 = v170;
            v94 = v160;
            v90 = v157;
            v91 = v153;
            v89 = v149;
            v98 = v114 == 0;
            v96 = v87 + 32LL * v95;
LABEL_126:
            if ( v98 && (_DWORD)v193 == *(_DWORD *)(v96 + 16) )
            {
LABEL_244:
              v37 = *(_DWORD *)(v96 + 24);
              goto LABEL_47;
            }
            if ( v97 == (const void *)-1LL )
            {
              if ( *(_DWORD *)(v96 + 16) == -1 )
              {
                v86 = v191;
                if ( !v90 )
                  v90 = (_DWORD *)v96;
                ++v188;
                v115 = v190 + 1;
                s1[0] = v90;
                if ( 4 * ((int)v190 + 1) < 3 * v191 )
                {
                  if ( v191 - (v115 + HIDWORD(v190)) <= v191 >> 3 )
                  {
                    sub_29A07B0((__int64)&v188, v191);
                    sub_29A00D0((__int64)&v188, (char **)n, s1);
                    v90 = s1[0];
                    v115 = v190 + 1;
                  }
                  goto LABEL_135;
                }
LABEL_178:
                sub_29A07B0((__int64)&v188, 2 * v86);
                sub_29A00D0((__int64)&v188, (char **)n, s1);
                v90 = s1[0];
                v115 = v190 + 1;
LABEL_135:
                LODWORD(v190) = v115;
                if ( *(_QWORD *)v90 != -1 || v90[4] != -1 )
                  --HIDWORD(v190);
                v37 = 0;
                *(__m128i *)v90 = _mm_loadu_si128((const __m128i *)n);
                v116 = v193;
                v90[6] = 0;
                v90[4] = v116;
                goto LABEL_47;
              }
              goto LABEL_105;
            }
          }
          else if ( *(_DWORD *)(v96 + 16) == (_DWORD)v193 )
          {
            goto LABEL_244;
          }
          if ( v97 == (const void *)-2LL && *(_DWORD *)(v96 + 16) == -2 && !v90 )
            v90 = (_DWORD *)v96;
          goto LABEL_105;
        }
        while ( v85 != (_QWORD *)-4096LL )
        {
          if ( v82 || v85 != (_QWORD *)-8192LL )
            v84 = v82;
          v83 = (v78 - 1) & (v81 + v83);
          v85 = *(_QWORD **)(v79 + 8LL * v83);
          if ( v175 == v85 )
            goto LABEL_99;
          ++v81;
          v82 = v84;
          v84 = (_QWORD *)(v79 + 8LL * v83);
        }
        v100 = *(_DWORD *)(v25 + 40);
        if ( !v82 )
          v82 = v84;
        ++*(_QWORD *)(v25 + 24);
        v101 = v100 + 1;
        if ( 4 * v101 >= 3 * v78 )
          goto LABEL_147;
        if ( v78 - *(_DWORD *)(v25 + 44) - v101 > v78 >> 3 )
          goto LABEL_115;
        v171 = v25;
        sub_E3B4A0(v25 + 24, v78);
        v141 = *(_DWORD *)(v171 + 48);
        if ( !v141 )
          goto LABEL_273;
        v142 = v141 - 1;
        v143 = *(_QWORD *)(v171 + 32);
        v144 = 1;
        v145 = v142 & v167;
        v82 = (_QWORD *)(v143 + 8LL * (v142 & v167));
        v124 = 0;
        v146 = *v82;
        v101 = *(_DWORD *)(v171 + 40) + 1;
        if ( v175 != (_QWORD *)*v82 )
        {
          while ( v146 != -4096 )
          {
            if ( !v124 && v146 == -8192 )
              v124 = v82;
            v145 = v142 & (v144 + v145);
            v82 = (_QWORD *)(v143 + 8LL * v145);
            v146 = *v82;
            if ( v175 == (_QWORD *)*v82 )
              goto LABEL_115;
            ++v144;
          }
LABEL_151:
          if ( v124 )
            v82 = v124;
        }
LABEL_115:
        *(_DWORD *)(v80 + 16) = v101;
        if ( *v82 != -4096 )
          --*(_DWORD *)(v80 + 20);
        *v82 = v175;
        if ( *(_DWORD *)(v80 + 16) == 1 )
          goto LABEL_49;
        v102 = v191;
        if ( !v191 )
        {
          ++v188;
          s1[0] = 0;
LABEL_40:
          v33 = 2 * v102;
          goto LABEL_41;
        }
        v174 = v191;
        v103 = v189;
        v104 = (unsigned int)(37 * v193);
        v105 = sub_C94890((_QWORD *)n[0], n[1]);
        v106 = (const void *)n[0];
        v107 = n[1];
        v34 = 0;
        v108 = 1;
        v109 = v174 - 1;
        v110 = (v174 - 1) & (((0xBF58476D1CE4E5B9LL * ((v105 << 32) | v104)) >> 31) ^ (484763065 * v104));
        while ( 2 )
        {
          v111 = v103 + 32LL * v110;
          v112 = *(const void **)v111;
          if ( *(_QWORD *)v111 == -1 )
          {
            if ( v106 == (const void *)-1LL )
            {
LABEL_224:
              if ( (_DWORD)v193 == *(_DWORD *)(v111 + 16) )
                goto LABEL_246;
              goto LABEL_225;
            }
          }
          else
          {
            if ( v112 == (const void *)-2LL )
            {
              if ( v106 == (const void *)-2LL )
                goto LABEL_224;
LABEL_234:
              if ( *(_DWORD *)(v111 + 16) == -2 && !v34 )
                v34 = (_DWORD *)v111;
              goto LABEL_123;
            }
            if ( v107 != *(_QWORD *)(v111 + 8) )
            {
LABEL_123:
              v113 = v108 + v110;
              ++v108;
              v110 = v109 & v113;
              continue;
            }
            if ( !v107 )
            {
              if ( (_DWORD)v193 == *(_DWORD *)(v111 + 16) )
              {
LABEL_246:
                v36 = (unsigned int *)(v111 + 24);
                v37 = *(_DWORD *)(v111 + 24) + 1;
                goto LABEL_46;
              }
              goto LABEL_226;
            }
            v151 = v108;
            v155 = v109;
            v159 = v34;
            v162 = v107;
            v140 = memcmp(v106, v112, v107);
            v107 = v162;
            v34 = v159;
            v109 = v155;
            v108 = v151;
            v111 = v103 + 32LL * v110;
            if ( !v140 )
              goto LABEL_224;
LABEL_225:
            if ( v112 != (const void *)-1LL )
            {
LABEL_226:
              if ( v112 != (const void *)-2LL )
                goto LABEL_123;
              goto LABEL_234;
            }
          }
          break;
        }
        if ( *(_DWORD *)(v111 + 16) != -1 )
          goto LABEL_123;
        v102 = v191;
        if ( !v34 )
          v34 = (_DWORD *)v111;
        ++v188;
        v35 = v190 + 1;
        s1[0] = v34;
        if ( 4 * ((int)v190 + 1) >= 3 * v191 )
          goto LABEL_40;
        if ( v191 - (v35 + HIDWORD(v190)) <= v191 >> 3 )
        {
          v33 = v191;
LABEL_41:
          sub_29A07B0((__int64)&v188, v33);
          sub_29A00D0((__int64)&v188, (char **)n, s1);
          v34 = s1[0];
          v35 = v190 + 1;
        }
        LODWORD(v190) = v35;
        if ( *(_QWORD *)v34 != -1 || v34[4] != -1 )
          --HIDWORD(v190);
        v36 = v34 + 6;
        v37 = 1;
        *(__m128i *)(v36 - 6) = _mm_loadu_si128((const __m128i *)n);
        v38 = v193;
        *v36 = 0;
        *(v36 - 2) = v38;
LABEL_46:
        *v36 = v37;
LABEL_47:
        v39 = sub_299FBF0(v5, v37);
        s1[1] = v40;
        s1[0] = v39;
        if ( (_BYTE)v40 )
        {
          sub_B10CB0(v181, (__int64)s1[0]);
          if ( v3 + 3 == v181 )
          {
            if ( v181[0] )
              sub_B91220((__int64)(v3 + 3), v181[0]);
          }
          else
          {
            v126 = v3[3];
            if ( v126 )
              sub_B91220((__int64)(v3 + 3), v126);
            v127 = (unsigned __int8 *)v181[0];
            v3[3] = v181[0];
            if ( v127 )
              sub_B976B0((__int64)v181, v127, (__int64)(v3 + 3));
          }
        }
        v172 = 1;
LABEL_49:
        v3 = (_QWORD *)v3[1];
      }
      while ( v178 != v3 );
LABEL_50:
      v177 = (_QWORD *)v177[1];
      if ( (_QWORD *)v165 == v177 )
      {
        v179 = *(_QWORD **)(a1 + 80);
        if ( (_QWORD *)v165 == v179 )
          goto LABEL_85;
        while ( 1 )
        {
          n[0] = 0;
          n[1] = 0;
          v193 = 0;
          v194 = 0;
          if ( !v179 )
            BUG();
          v41 = (_QWORD *)v179[4];
          v42 = v179 + 3;
          if ( v179 + 3 == v41 )
          {
            v71 = 0;
            v72 = 0;
            goto LABEL_84;
          }
          while ( 1 )
          {
LABEL_54:
            if ( !v41 )
              BUG();
            v43 = *((_BYTE *)v41 - 24);
            if ( v43 == 34 )
              break;
            if ( v43 == 85 )
            {
              v44 = *(v41 - 7);
              if ( !v44 || *(_BYTE *)v44 || *(_QWORD *)(v44 + 24) != v41[7] || (*(_BYTE *)(v44 + 33) & 0x20) == 0 )
                break;
              v41 = (_QWORD *)v41[1];
              if ( v42 == v41 )
                goto LABEL_83;
            }
            else
            {
LABEL_82:
              v41 = (_QWORD *)v41[1];
              if ( v42 == v41 )
                goto LABEL_83;
            }
          }
          v45 = sub_B10CD0((__int64)(v41 + 3));
          v46 = v45;
          if ( !v45 )
            goto LABEL_82;
          v47 = *(_DWORD *)(v45 + 4);
          v48 = *(_BYTE *)(v45 - 16);
          if ( (v48 & 2) != 0 )
            v49 = *(_QWORD **)(v46 - 32);
          else
            v49 = (_QWORD *)(v46 - 16 - 8LL * ((v48 >> 2) & 0xF));
          v50 = (_BYTE *)*v49;
          if ( *(_BYTE *)*v49 == 16 )
            goto LABEL_66;
          v51 = *(v50 - 16);
          if ( (v51 & 2) == 0 )
          {
            v50 = *(_BYTE **)&v50[-8 * ((v51 >> 2) & 0xF) - 16];
            if ( !v50 )
              goto LABEL_167;
LABEL_66:
            v52 = *(v50 - 16);
            if ( (v52 & 2) != 0 )
            {
              v53 = (char *)**((_QWORD **)v50 - 4);
              if ( v53 )
              {
LABEL_68:
                v53 = (char *)sub_B91420((__int64)v53);
                v55 = v54;
                goto LABEL_69;
              }
            }
            else
            {
              v53 = *(char **)&v50[-8 * ((v52 >> 2) & 0xF) - 16];
              if ( v53 )
                goto LABEL_68;
            }
            v55 = 0;
            goto LABEL_69;
          }
          v50 = (_BYTE *)**((_QWORD **)v50 - 4);
          if ( v50 )
            goto LABEL_66;
LABEL_167:
          v55 = 0;
          v53 = (char *)byte_3F871B3;
LABEL_69:
          v56 = v194;
          s1[0] = v53;
          s1[1] = v55;
          v183 = v47;
          if ( !(_DWORD)v194 )
          {
            ++n[0];
            v181[0] = 0;
LABEL_76:
            v69 = 2 * v56;
LABEL_77:
            sub_29A0AE0((__int64)n, v69);
            sub_29A0250((__int64)n, (char **)s1, v181);
            v58 = v181[0];
            v70 = v193 + 1;
LABEL_78:
            LODWORD(v193) = v70;
            if ( *(_QWORD *)v58 != -1 || *(_DWORD *)(v58 + 16) != -1 )
              --HIDWORD(v193);
            *(__m128i *)v58 = _mm_loadu_si128((const __m128i *)s1);
            *(_DWORD *)(v58 + 16) = v183;
            goto LABEL_82;
          }
          v169 = v194;
          v176 = n[1];
          v57 = sub_C94890(v53, (__int64)v55);
          v58 = 0;
          v59 = 1;
          v60 = v169 - 1;
          v61 = s1[1];
          v62 = 0xBF58476D1CE4E5B9LL * ((37 * v47) | (v57 << 32));
          v63 = (char *)s1[0];
          for ( j = (v169 - 1) & ((v62 >> 31) ^ v62); ; j = v60 & v68 )
          {
            v65 = v176 + 24LL * j;
            v66 = *(const void **)v65;
            if ( *(_QWORD *)v65 == -1 )
            {
              if ( v63 != (char *)-1LL )
                goto LABEL_190;
              goto LABEL_188;
            }
            v67 = v63 + 2 == 0;
            if ( v66 != (const void *)-2LL )
              break;
LABEL_181:
            if ( !v67 )
              goto LABEL_182;
LABEL_188:
            if ( v183 == *(_DWORD *)(v65 + 16) )
              goto LABEL_197;
            if ( v66 != (const void *)-1LL )
              goto LABEL_182;
LABEL_190:
            if ( *(_DWORD *)(v65 + 16) == -1 )
            {
              v56 = v194;
              if ( !v58 )
                v58 = v65;
              ++n[0];
              v70 = v193 + 1;
              v181[0] = v58;
              if ( 4 * ((int)v193 + 1) >= (unsigned int)(3 * v194) )
                goto LABEL_76;
              if ( (int)v194 - (v70 + HIDWORD(v193)) > (unsigned int)v194 >> 3 )
                goto LABEL_78;
              v69 = v194;
              goto LABEL_77;
            }
LABEL_74:
            v68 = v59 + j;
            ++v59;
          }
          if ( *(void **)(v65 + 8) != v61 )
            goto LABEL_74;
          if ( v61 )
          {
            v150 = v176 + 24LL * j;
            v154 = v59;
            v158 = j;
            v161 = v60;
            v164 = v58;
            v166 = v61;
            v128 = memcmp(v63, *(const void **)v65, (size_t)v61);
            v61 = v166;
            v58 = v164;
            v60 = v161;
            j = v158;
            v59 = v154;
            v67 = v128 == 0;
            v65 = v150;
            goto LABEL_181;
          }
          if ( *(_DWORD *)(v65 + 16) != v183 )
          {
LABEL_182:
            if ( v66 == (const void *)-2LL && *(_DWORD *)(v65 + 16) == -2 && !v58 )
              v58 = v65;
            goto LABEL_74;
          }
LABEL_197:
          if ( !(unsigned __int8)sub_29A00D0((__int64)&v188, (char **)s1, &v180) )
          {
            v136 = v191;
            v137 = v180;
            ++v188;
            v138 = v190 + 1;
            v181[0] = v180;
            if ( 4 * ((int)v190 + 1) >= 3 * v191 )
            {
              v136 = 2 * v191;
            }
            else if ( v191 - HIDWORD(v190) - v138 > v191 >> 3 )
            {
              goto LABEL_209;
            }
            sub_29A07B0((__int64)&v188, v136);
            sub_29A00D0((__int64)&v188, (char **)s1, v181);
            v138 = v190 + 1;
            v137 = v181[0];
LABEL_209:
            LODWORD(v190) = v138;
            if ( *(_QWORD *)v137 != -1 || *(_DWORD *)(v137 + 16) != -1 )
              --HIDWORD(v190);
            v129 = (unsigned int *)(v137 + 24);
            *(__m128i *)(v129 - 6) = _mm_loadu_si128((const __m128i *)s1);
            v139 = v183;
            *v129 = 0;
            *(v129 - 2) = v139;
            goto LABEL_199;
          }
          v129 = (unsigned int *)(v180 + 24);
LABEL_199:
          v130 = *v129 + 1;
          *v129 = v130;
          v131 = sub_299FBF0(v46, v130);
          v181[1] = v132;
          v133 = v132;
          v181[0] = (__int64)v131;
          if ( !(_BYTE)v132 )
            goto LABEL_82;
          sub_B10CB0(&v180, v181[0]);
          if ( v41 + 3 == &v180 )
          {
            if ( v180 )
              sub_B91220((__int64)(v41 + 3), v180);
          }
          else
          {
            v134 = v41[3];
            if ( v134 )
              sub_B91220((__int64)(v41 + 3), v134);
            v135 = (unsigned __int8 *)v180;
            v41[3] = v180;
            if ( v135 )
              sub_B976B0((__int64)&v180, v135, (__int64)(v41 + 3));
          }
          v41 = (_QWORD *)v41[1];
          v172 = v133;
          if ( v42 != v41 )
            goto LABEL_54;
LABEL_83:
          v71 = n[1];
          v72 = 24LL * (unsigned int)v194;
LABEL_84:
          sub_C7D6A0(v71, v72, 8);
          v179 = (_QWORD *)v179[1];
          if ( v177 == v179 )
          {
LABEL_85:
            v73 = v189;
            v74 = 32LL * v191;
            goto LABEL_86;
          }
        }
      }
    }
  }
  v73 = 0;
  v74 = 0;
LABEL_86:
  sub_C7D6A0(v73, v74, 8);
  v75 = v187;
  if ( !v187 )
    goto LABEL_96;
  v76 = v185;
  v77 = v185 + 56LL * v187;
  do
  {
    while ( *(_QWORD *)v76 == -1 )
    {
      if ( *(_DWORD *)(v76 + 16) != -1 )
        goto LABEL_89;
LABEL_90:
      v76 += 56;
      if ( v77 == v76 )
        goto LABEL_95;
    }
    if ( *(_QWORD *)v76 != -2 || *(_DWORD *)(v76 + 16) != -2 )
    {
LABEL_89:
      sub_C7D6A0(*(_QWORD *)(v76 + 32), 8LL * *(unsigned int *)(v76 + 48), 8);
      goto LABEL_90;
    }
    v76 += 56;
  }
  while ( v77 != v76 );
LABEL_95:
  v75 = v187;
LABEL_96:
  sub_C7D6A0(v185, 56 * v75, 8);
  return v172;
}
