// Function: sub_2A86DC0
// Address: 0x2a86dc0
//
__int64 __fastcall sub_2A86DC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned __int64 v6; // r14
  __int64 v7; // r13
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rcx
  __int64 v10; // r15
  _QWORD *v11; // rdi
  __int64 v12; // r11
  _QWORD *v13; // rdx
  _QWORD *v14; // rax
  _QWORD *v15; // rsi
  __int64 v16; // rbx
  __int64 v17; // rax
  unsigned __int64 v18; // rcx
  int v19; // r11d
  unsigned __int64 *v20; // rdx
  _QWORD **v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  _QWORD *v30; // rbx
  _QWORD *v31; // r14
  void (__fastcall *v32)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v33; // rax
  const __m128i *v35; // r15
  __m128i *v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // r13
  __int64 v41; // r14
  __int64 v42; // r15
  __int64 v43; // rbx
  const char *v44; // r14
  __int64 v45; // r12
  __int64 v46; // rsi
  _QWORD *v47; // rax
  _QWORD *v48; // rdx
  __int64 v49; // r14
  __int64 v50; // rdi
  const char *v51; // rax
  int v52; // ebx
  unsigned __int64 v53; // rdx
  __int64 v54; // r12
  __int64 v55; // rax
  __int64 v56; // r15
  __int64 v57; // r12
  __int64 *v58; // r14
  int v59; // eax
  int v60; // eax
  unsigned int v61; // esi
  __int64 v62; // rax
  __int64 v63; // rsi
  __int64 v64; // rsi
  __int64 v65; // rbx
  __int64 v66; // rsi
  int v67; // eax
  int v68; // eax
  unsigned int v69; // edi
  __int64 v70; // rax
  __int64 v71; // rdi
  __int64 v72; // rdi
  unsigned __int64 v73; // r8
  __int64 v74; // r9
  int v75; // r10d
  unsigned int v76; // ecx
  __int64 v77; // rax
  const char *v78; // rdx
  __int64 v79; // rax
  unsigned __int64 *v80; // rax
  __int64 v81; // rdx
  int v82; // eax
  __int64 v83; // rdx
  __int64 v84; // rcx
  const char **v85; // rsi
  const char **v86; // rdx
  __int64 *v87; // r12
  __int64 *v88; // rbx
  __int64 v89; // rdi
  unsigned __int64 *v90; // rbx
  unsigned __int64 *v91; // r12
  unsigned __int64 v92; // rdi
  __int64 *v93; // r13
  _QWORD *v94; // r12
  _QWORD *v95; // rbx
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // r12
  unsigned __int64 *v99; // rbx
  unsigned __int64 *v100; // r13
  unsigned __int64 v101; // rcx
  __int64 v102; // rcx
  __int64 v103; // r8
  char **v104; // rsi
  __int64 v105; // rdi
  unsigned __int64 *v106; // rbx
  unsigned __int64 *v107; // r14
  unsigned __int64 v108; // rdi
  int v109; // eax
  __int64 v110; // rdi
  int v111; // esi
  __int64 v112; // rcx
  int v113; // esi
  __int64 v114; // rdi
  char *v115; // r15
  unsigned __int8 v116; // [rsp+20h] [rbp-570h]
  char v118; // [rsp+37h] [rbp-559h]
  __int64 v120; // [rsp+48h] [rbp-548h]
  __int64 v121; // [rsp+50h] [rbp-540h]
  __int64 v122; // [rsp+58h] [rbp-538h]
  __int64 v123; // [rsp+58h] [rbp-538h]
  __int64 v124; // [rsp+60h] [rbp-530h]
  __int64 v125; // [rsp+60h] [rbp-530h]
  __int64 v126; // [rsp+68h] [rbp-528h]
  __int64 v127; // [rsp+68h] [rbp-528h]
  const char *v128; // [rsp+68h] [rbp-528h]
  int v129; // [rsp+68h] [rbp-528h]
  __int64 v130; // [rsp+70h] [rbp-520h]
  __int64 v131; // [rsp+80h] [rbp-510h]
  __int64 v132; // [rsp+98h] [rbp-4F8h]
  __int64 v133; // [rsp+A0h] [rbp-4F0h]
  __int64 v134; // [rsp+A0h] [rbp-4F0h]
  __int64 v135; // [rsp+A8h] [rbp-4E8h]
  unsigned __int64 *v136; // [rsp+A8h] [rbp-4E8h]
  __int64 v137; // [rsp+A8h] [rbp-4E8h]
  char v138; // [rsp+A8h] [rbp-4E8h]
  unsigned __int64 v139; // [rsp+B0h] [rbp-4E0h]
  unsigned __int64 *v140; // [rsp+B0h] [rbp-4E0h]
  __int64 v141; // [rsp+B8h] [rbp-4D8h]
  unsigned __int64 *v142; // [rsp+B8h] [rbp-4D8h]
  __int64 v143; // [rsp+C0h] [rbp-4D0h]
  __int64 *v144; // [rsp+C8h] [rbp-4C8h]
  unsigned __int64 v145; // [rsp+D8h] [rbp-4B8h] BYREF
  __int64 v146; // [rsp+E0h] [rbp-4B0h] BYREF
  __int64 v147; // [rsp+E8h] [rbp-4A8h]
  __int64 v148; // [rsp+F0h] [rbp-4A0h]
  unsigned int v149; // [rsp+F8h] [rbp-498h]
  unsigned __int64 *v150; // [rsp+100h] [rbp-490h] BYREF
  __int64 v151; // [rsp+108h] [rbp-488h]
  unsigned __int64 v152; // [rsp+110h] [rbp-480h] BYREF
  __int64 v153; // [rsp+118h] [rbp-478h]
  _BYTE v154[48]; // [rsp+120h] [rbp-470h] BYREF
  __int64 *v155; // [rsp+150h] [rbp-440h] BYREF
  __int64 v156; // [rsp+158h] [rbp-438h]
  _BYTE v157[64]; // [rsp+160h] [rbp-430h] BYREF
  _BYTE *v158; // [rsp+1A0h] [rbp-3F0h] BYREF
  __int64 v159; // [rsp+1A8h] [rbp-3E8h]
  _BYTE v160[64]; // [rsp+1B0h] [rbp-3E0h] BYREF
  char *v161; // [rsp+1F0h] [rbp-3A0h]
  __int64 v162; // [rsp+1F8h] [rbp-398h]
  char v163; // [rsp+200h] [rbp-390h] BYREF
  const char *v164; // [rsp+240h] [rbp-350h] BYREF
  char *v165; // [rsp+248h] [rbp-348h]
  __int64 v166; // [rsp+250h] [rbp-340h]
  char v167; // [rsp+258h] [rbp-338h] BYREF
  __int16 v168; // [rsp+260h] [rbp-330h]
  _QWORD *v169; // [rsp+2A0h] [rbp-2F0h] BYREF
  __int64 v170; // [rsp+2A8h] [rbp-2E8h]
  _QWORD v171[68]; // [rsp+2B0h] [rbp-2E0h] BYREF
  char v172; // [rsp+4D0h] [rbp-C0h]
  __int64 v173; // [rsp+4D8h] [rbp-B8h]
  char *v174; // [rsp+4E0h] [rbp-B0h]
  __int64 v175; // [rsp+4E8h] [rbp-A8h]
  int v176; // [rsp+4F0h] [rbp-A0h]
  char v177; // [rsp+4F4h] [rbp-9Ch]
  char v178; // [rsp+4F8h] [rbp-98h] BYREF
  __int16 v179; // [rsp+538h] [rbp-58h]
  _QWORD *v180; // [rsp+540h] [rbp-50h]
  _QWORD *v181; // [rsp+548h] [rbp-48h]
  __int64 v182; // [rsp+550h] [rbp-40h]

  v3 = a3;
  v155 = (__int64 *)v157;
  v156 = 0x800000000LL;
  sub_D46D90(a3, (__int64)&v155);
  v152 = (unsigned __int64)v154;
  v153 = 0x200000000LL;
  v143 = v3 + 56;
  v144 = &v155[(unsigned int)v156];
  if ( v155 != v144 )
  {
    v6 = (unsigned __int64)v155;
    v7 = v3;
    while ( 1 )
    {
      v8 = *(_QWORD *)v6;
      v9 = *(_QWORD *)(*(_QWORD *)v6 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v9 == *(_QWORD *)v6 + 48LL )
        goto LABEL_193;
      if ( !v9 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 > 0xA )
LABEL_193:
        BUG();
      v10 = *(_QWORD *)(v9 - 56);
      if ( *(_BYTE *)(v7 + 84) )
      {
        v11 = *(_QWORD **)(v7 + 64);
        v12 = *(unsigned int *)(v7 + 76);
        v13 = &v11[v12];
        v14 = v11;
        if ( v11 != v13 )
        {
          v15 = *(_QWORD **)(v7 + 64);
          while ( v10 != *v15 )
          {
            if ( v13 == ++v15 )
              goto LABEL_12;
          }
          v10 = 0;
LABEL_12:
          if ( (*(_DWORD *)(v9 - 20) & 0x7FFFFFF) == 1 )
          {
            v16 = 0;
LABEL_17:
            while ( *v14 != v16 )
            {
              if ( ++v14 == v13 )
                goto LABEL_19;
            }
            v16 = 0;
            goto LABEL_19;
          }
LABEL_13:
          v16 = *(_QWORD *)(v9 - 88);
LABEL_14:
          v13 = &v11[v12];
          v14 = v11;
          if ( v11 != v13 )
            goto LABEL_17;
          goto LABEL_19;
        }
        v16 = 0;
        if ( (*(_DWORD *)(v9 - 20) & 0x7FFFFFF) != 1 )
          goto LABEL_13;
      }
      else
      {
        v139 = *(_QWORD *)(*(_QWORD *)v6 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
        if ( sub_C8CA60(v143, *(_QWORD *)(v9 - 56)) )
          v10 = 0;
        v16 = 0;
        if ( (*(_DWORD *)(v139 - 20) & 0x7FFFFFF) != 1 )
          v16 = *(_QWORD *)(v139 - 88);
        if ( *(_BYTE *)(v7 + 84) )
        {
          v11 = *(_QWORD **)(v7 + 64);
          v12 = *(unsigned int *)(v7 + 76);
          goto LABEL_14;
        }
        if ( sub_C8CA60(v143, v16) )
          v16 = 0;
      }
LABEL_19:
      v17 = (unsigned int)v153;
      v18 = v152;
      v19 = v153;
      v20 = (unsigned __int64 *)(v152 + 24LL * (unsigned int)v153);
      if ( (unsigned int)v153 >= (unsigned __int64)HIDWORD(v153) )
      {
        v170 = v10;
        v35 = (const __m128i *)&v169;
        v169 = (_QWORD *)v8;
        v171[0] = v16;
        if ( HIDWORD(v153) < (unsigned __int64)(unsigned int)v153 + 1 )
        {
          if ( v152 > (unsigned __int64)&v169 || v20 <= (unsigned __int64 *)&v169 )
          {
            sub_C8D5F0((__int64)&v152, v154, (unsigned int)v153 + 1LL, 0x18u, v4, v5);
            v18 = v152;
            v17 = (unsigned int)v153;
          }
          else
          {
            v115 = (char *)&v169 - v152;
            sub_C8D5F0((__int64)&v152, v154, (unsigned int)v153 + 1LL, 0x18u, v4, v5);
            v18 = v152;
            v17 = (unsigned int)v153;
            v35 = (const __m128i *)&v115[v152];
          }
        }
        v36 = (__m128i *)(v18 + 24 * v17);
        *v36 = _mm_loadu_si128(v35);
        v37 = v35[1].m128i_i64[0];
        LODWORD(v153) = v153 + 1;
        v36[1].m128i_i64[0] = v37;
      }
      else
      {
        if ( v20 )
        {
          v20[2] = v16;
          *v20 = v8;
          v20[1] = v10;
          v19 = v153;
        }
        LODWORD(v153) = v19 + 1;
      }
      v6 += 8LL;
      if ( v144 == (__int64 *)v6 )
      {
        v3 = v7;
        break;
      }
    }
  }
  BYTE4(v164) = 1;
  v21 = &v169;
  v158 = v160;
  v159 = 0x800000000LL;
  v169 = v171;
  v170 = 0x1000000000LL;
  v179 = 0;
  v171[66] = a1;
  v174 = &v178;
  LODWORD(v164) = qword_500C1A8;
  v171[64] = 0;
  v171[65] = 0;
  v171[67] = 0;
  v172 = 0;
  v173 = 0;
  v175 = 8;
  v176 = 0;
  v177 = 1;
  v180 = 0;
  v181 = 0;
  v182 = 0;
  v131 = sub_3194260(&v152, &v169, &v158, "loop.exit", 9, v164);
  v116 = v22;
  v118 = v22;
  if ( (_BYTE)v22 )
  {
    v38 = *(_QWORD *)(v3 + 40);
    v146 = 0;
    v147 = 0;
    v150 = &v152;
    v39 = *(_QWORD *)(v3 + 32);
    v148 = 0;
    v149 = 0;
    v151 = 0;
    v130 = v38;
    if ( v39 != v38 )
    {
      v133 = v39;
      v40 = v131;
      v41 = v3;
      do
      {
        v42 = v41;
        v141 = *(_QWORD *)v133 + 48LL;
        v144 = *(__int64 **)(*(_QWORD *)v133 + 56LL);
        if ( v144 != (__int64 *)v141 )
        {
          do
          {
            if ( !v144 )
              BUG();
            v43 = *(v144 - 1);
            v44 = (const char *)(v144 - 3);
            LODWORD(v143) = ((unsigned int)((_DWORD)v144 - 24) >> 9) ^ ((unsigned int)((_DWORD)v144 - 24) >> 4);
            if ( v43 )
            {
              while ( 1 )
              {
                v45 = *(_QWORD *)(v43 + 24);
                v46 = *(_QWORD *)(v45 + 40);
                if ( v40 == v46 )
                  goto LABEL_70;
                if ( *(_BYTE *)(v42 + 84) )
                {
                  v47 = *(_QWORD **)(v42 + 64);
                  v48 = &v47[*(unsigned int *)(v42 + 76)];
                  if ( v47 != v48 )
                  {
                    while ( v46 != *v47 )
                    {
                      if ( v48 == ++v47 )
                        goto LABEL_101;
                    }
                    goto LABEL_70;
                  }
LABEL_101:
                  if ( !v149 )
                  {
                    ++v146;
                    goto LABEL_165;
                  }
                  v73 = v149 - 1;
                  v74 = 0;
                  v75 = 1;
                  v76 = v73 & v143;
                  v77 = v147 + 16LL * ((unsigned int)v73 & (unsigned int)v143);
                  v78 = *(const char **)v77;
                  if ( v44 != *(const char **)v77 )
                  {
                    while ( v78 != (const char *)-4096LL )
                    {
                      if ( !v74 && v78 == (const char *)-8192LL )
                        v74 = v77;
                      v76 = v73 & (v75 + v76);
                      v77 = v147 + 16LL * v76;
                      v78 = *(const char **)v77;
                      if ( v44 == *(const char **)v77 )
                        goto LABEL_103;
                      ++v75;
                    }
                    if ( !v74 )
                      v74 = v77;
                    ++v146;
                    v82 = v148 + 1;
                    if ( 4 * ((int)v148 + 1) < 3 * v149 )
                    {
                      if ( v149 - HIDWORD(v148) - v82 <= v149 >> 3 )
                      {
                        sub_9BAAD0((__int64)&v146, v149);
                        if ( !v149 )
                        {
LABEL_195:
                          LODWORD(v148) = v148 + 1;
                          BUG();
                        }
                        v113 = 1;
                        v112 = 0;
                        v82 = v148 + 1;
                        v73 = (v149 - 1) & (unsigned int)v143;
                        v74 = v147 + 16 * v73;
                        v114 = *(_QWORD *)v74;
                        if ( v44 != *(const char **)v74 )
                        {
                          while ( v114 != -4096 )
                          {
                            if ( v114 == -8192 && !v112 )
                              v112 = v74;
                            v73 = (v149 - 1) & (v113 + (_DWORD)v73);
                            v74 = v147 + 16LL * (unsigned int)v73;
                            v114 = *(_QWORD *)v74;
                            if ( v44 == *(const char **)v74 )
                              goto LABEL_118;
                            ++v113;
                          }
                          goto LABEL_177;
                        }
                      }
                      goto LABEL_118;
                    }
LABEL_165:
                    sub_9BAAD0((__int64)&v146, 2 * v149);
                    if ( !v149 )
                      goto LABEL_195;
                    v73 = (v149 - 1) & (unsigned int)v143;
                    v82 = v148 + 1;
                    v74 = v147 + 16 * v73;
                    v110 = *(_QWORD *)v74;
                    if ( v44 != *(const char **)v74 )
                    {
                      v111 = 1;
                      v112 = 0;
                      while ( v110 != -4096 )
                      {
                        if ( !v112 && v110 == -8192 )
                          v112 = v74;
                        v73 = (v149 - 1) & (v111 + (_DWORD)v73);
                        v74 = v147 + 16LL * (unsigned int)v73;
                        v110 = *(_QWORD *)v74;
                        if ( v44 == *(const char **)v74 )
                          goto LABEL_118;
                        ++v111;
                      }
LABEL_177:
                      if ( v112 )
                        v74 = v112;
                    }
LABEL_118:
                    LODWORD(v148) = v82;
                    if ( *(_QWORD *)v74 != -4096 )
                      --HIDWORD(v148);
                    *(_QWORD *)v74 = v44;
                    *(_DWORD *)(v74 + 8) = 0;
                    v83 = (unsigned int)v151;
                    v161 = &v163;
                    v162 = 0x800000000LL;
                    v166 = 0x800000000LL;
                    v79 = (unsigned int)v151;
                    v164 = v44;
                    v165 = &v167;
                    if ( (unsigned __int64)(unsigned int)v151 + 1 > HIDWORD(v151) )
                    {
                      if ( v150 > (unsigned __int64 *)&v164 || &v164 >= (const char **)&v150[11 * (unsigned int)v151] )
                      {
                        v132 = -1;
                        v138 = 0;
                      }
                      else
                      {
                        v132 = 0x2E8BA2E8BA2E8BA3LL * (((char *)&v164 - (char *)v150) >> 3);
                        v138 = v118;
                      }
                      v126 = v74;
                      v96 = sub_C8D7D0((__int64)&v150, (__int64)&v152, (unsigned int)v151 + 1LL, 0x58u, &v145, v74);
                      v97 = (__int64)v150;
                      v74 = v126;
                      v84 = v96;
                      v73 = (unsigned __int64)&v150[11 * (unsigned int)v151];
                      if ( v150 != (unsigned __int64 *)v73 )
                      {
                        v127 = v45;
                        v98 = v96;
                        v124 = v43;
                        v99 = &v150[11 * (unsigned int)v151];
                        v120 = v40;
                        v100 = v150;
                        v122 = v74;
                        v121 = v96;
                        do
                        {
                          while ( 1 )
                          {
                            if ( v98 )
                            {
                              v101 = *v100;
                              *(_DWORD *)(v98 + 16) = 0;
                              *(_DWORD *)(v98 + 20) = 8;
                              *(_QWORD *)v98 = v101;
                              v102 = v98 + 24;
                              *(_QWORD *)(v98 + 8) = v98 + 24;
                              v103 = *((unsigned int *)v100 + 4);
                              if ( (_DWORD)v103 )
                                break;
                            }
                            v100 += 11;
                            v98 += 88;
                            if ( v99 == v100 )
                              goto LABEL_154;
                          }
                          v104 = (char **)(v100 + 1);
                          v105 = v98 + 8;
                          v100 += 11;
                          v98 += 88;
                          sub_2A86650(v105, v104, v97, v102, v103, v74);
                        }
                        while ( v99 != v100 );
LABEL_154:
                        v45 = v127;
                        v43 = v124;
                        v74 = v122;
                        v84 = v121;
                        v40 = v120;
                        v73 = (unsigned __int64)&v150[11 * (unsigned int)v151];
                        if ( v150 != (unsigned __int64 *)v73 )
                        {
                          v106 = &v150[11 * (unsigned int)v151];
                          v128 = v44;
                          v107 = v150;
                          do
                          {
                            v106 -= 11;
                            v108 = v106[1];
                            if ( (unsigned __int64 *)v108 != v106 + 3 )
                              _libc_free(v108);
                          }
                          while ( v106 != v107 );
                          v44 = v128;
                          v43 = v124;
                          v74 = v122;
                          v84 = v121;
                          v73 = (unsigned __int64)v150;
                        }
                      }
                      v109 = v145;
                      if ( (unsigned __int64 *)v73 != &v152 )
                      {
                        v123 = v84;
                        v125 = v74;
                        v129 = v145;
                        _libc_free(v73);
                        v84 = v123;
                        v74 = v125;
                        v109 = v129;
                      }
                      v83 = (unsigned int)v151;
                      HIDWORD(v151) = v109;
                      v150 = (unsigned __int64 *)v84;
                      v85 = &v164;
                      v79 = (unsigned int)v151;
                      if ( v138 )
                        v85 = (const char **)(v84 + 88 * v132);
                    }
                    else
                    {
                      v84 = (__int64)v150;
                      v85 = &v164;
                    }
                    v86 = (const char **)(v84 + 88 * v83);
                    if ( v86 )
                    {
                      *v86 = *v85;
                      v86[1] = (const char *)(v86 + 3);
                      v86[2] = (const char *)0x800000000LL;
                      if ( *((_DWORD *)v85 + 4) )
                      {
                        v137 = v74;
                        sub_2A86650((__int64)(v86 + 1), (char **)v85 + 1, (__int64)v86, v84, v73, v74);
                        v79 = (unsigned int)v151;
                        v74 = v137;
                      }
                      else
                      {
                        v79 = (unsigned int)v151;
                      }
                    }
                    LODWORD(v151) = v79 + 1;
                    if ( v165 != &v167 )
                    {
                      v135 = v74;
                      _libc_free((unsigned __int64)v165);
                      v74 = v135;
                      v79 = (unsigned int)(v151 - 1);
                    }
                    *(_DWORD *)(v74 + 8) = v79;
                    goto LABEL_104;
                  }
LABEL_103:
                  v79 = *(unsigned int *)(v77 + 8);
LABEL_104:
                  v80 = &v150[11 * v79];
                  v81 = *((unsigned int *)v80 + 4);
                  if ( v81 + 1 > (unsigned __int64)*((unsigned int *)v80 + 5) )
                  {
                    v136 = v80;
                    sub_C8D5F0((__int64)(v80 + 1), v80 + 3, v81 + 1, 8u, v81 + 1, v74);
                    v80 = v136;
                    v81 = *((unsigned int *)v136 + 4);
                  }
                  *(_QWORD *)(v80[1] + 8 * v81) = v45;
                  ++*((_DWORD *)v80 + 4);
                  v43 = *(_QWORD *)(v43 + 8);
                  if ( !v43 )
                    break;
                }
                else
                {
                  if ( !sub_C8CA60(v42 + 56, v46) )
                    goto LABEL_101;
LABEL_70:
                  v43 = *(_QWORD *)(v43 + 8);
                  if ( !v43 )
                    break;
                }
              }
            }
            v144 = (__int64 *)v144[1];
          }
          while ( (__int64 *)v141 != v144 );
          v41 = v42;
        }
        v133 += 8;
      }
      while ( v130 != v133 );
      v3 = v41;
      v140 = &v150[11 * (unsigned int)v151];
      if ( v150 != v140 )
      {
        v142 = v150;
        v134 = v41;
        do
        {
          v49 = *v142;
          v50 = *v142;
          v144 = *(__int64 **)(v131 + 56);
          v51 = sub_BD5D20(v50);
          v52 = v156;
          v164 = v51;
          v168 = 773;
          v165 = (char *)v53;
          v166 = (__int64)".moved";
          v54 = *(_QWORD *)(v49 + 8);
          v55 = sub_BD2DA0(80);
          v56 = v55;
          if ( v55 )
          {
            sub_B44260(v55, v54, 55, 0x8000000u, (__int64)v144, 1u);
            *(_DWORD *)(v56 + 72) = v52;
            sub_BD6B50((unsigned __int8 *)v56, &v164);
            sub_BD2A10(v56, *(_DWORD *)(v56 + 72), 1);
          }
          v144 = &v155[(unsigned int)v156];
          v143 = v49 + 16;
          if ( v155 != v144 )
          {
            v57 = v49;
            v58 = v155;
            do
            {
              v65 = *v58;
              if ( *v58 == *(_QWORD *)(v57 + 40) || (unsigned __int8)sub_B19D00(a1, v57, *v58) )
              {
                v59 = *(_DWORD *)(v56 + 4) & 0x7FFFFFF;
                if ( v59 == *(_DWORD *)(v56 + 72) )
                {
                  sub_B48D90(v56);
                  v59 = *(_DWORD *)(v56 + 4) & 0x7FFFFFF;
                }
                v60 = (v59 + 1) & 0x7FFFFFF;
                v61 = v60 | *(_DWORD *)(v56 + 4) & 0xF8000000;
                v62 = *(_QWORD *)(v56 - 8) + 32LL * (unsigned int)(v60 - 1);
                *(_DWORD *)(v56 + 4) = v61;
                if ( *(_QWORD *)v62 )
                {
                  v63 = *(_QWORD *)(v62 + 8);
                  **(_QWORD **)(v62 + 16) = v63;
                  if ( v63 )
                    *(_QWORD *)(v63 + 16) = *(_QWORD *)(v62 + 16);
                }
                *(_QWORD *)v62 = v57;
                v64 = *(_QWORD *)(v57 + 16);
                *(_QWORD *)(v62 + 8) = v64;
                if ( v64 )
                  *(_QWORD *)(v64 + 16) = v62 + 8;
                *(_QWORD *)(v62 + 16) = v143;
                *(_QWORD *)(v57 + 16) = v62;
              }
              else
              {
                v66 = sub_ACADE0(*(__int64 ***)(v57 + 8));
                v67 = *(_DWORD *)(v56 + 4) & 0x7FFFFFF;
                if ( v67 == *(_DWORD *)(v56 + 72) )
                {
                  sub_B48D90(v56);
                  v67 = *(_DWORD *)(v56 + 4) & 0x7FFFFFF;
                }
                v68 = (v67 + 1) & 0x7FFFFFF;
                v69 = v68 | *(_DWORD *)(v56 + 4) & 0xF8000000;
                v70 = *(_QWORD *)(v56 - 8) + 32LL * (unsigned int)(v68 - 1);
                *(_DWORD *)(v56 + 4) = v69;
                if ( *(_QWORD *)v70 )
                {
                  v71 = *(_QWORD *)(v70 + 8);
                  **(_QWORD **)(v70 + 16) = v71;
                  if ( v71 )
                    *(_QWORD *)(v71 + 16) = *(_QWORD *)(v70 + 16);
                }
                *(_QWORD *)v70 = v66;
                if ( v66 )
                {
                  v72 = *(_QWORD *)(v66 + 16);
                  *(_QWORD *)(v70 + 8) = v72;
                  if ( v72 )
                    *(_QWORD *)(v72 + 16) = v70 + 8;
                  *(_QWORD *)(v70 + 16) = v66 + 16;
                  *(_QWORD *)(v66 + 16) = v70;
                }
              }
              ++v58;
              *(_QWORD *)(*(_QWORD *)(v56 - 8)
                        + 32LL * *(unsigned int *)(v56 + 72)
                        + 8LL * ((*(_DWORD *)(v56 + 4) & 0x7FFFFFFu) - 1)) = v65;
            }
            while ( v144 != v58 );
            v49 = v57;
          }
          v87 = (__int64 *)v142[1];
          v88 = &v87[*((unsigned int *)v142 + 4)];
          while ( v88 != v87 )
          {
            v89 = *v87++;
            sub_BD2ED0(v89, v49, v56);
          }
          v142 += 11;
        }
        while ( v140 != v142 );
        v90 = v150;
        v3 = v134;
        v140 = &v150[11 * (unsigned int)v151];
        if ( v150 != v140 )
        {
          v91 = &v150[11 * (unsigned int)v151];
          do
          {
            v91 -= 11;
            v92 = v91[1];
            if ( (unsigned __int64 *)v92 != v91 + 3 )
              _libc_free(v92);
          }
          while ( v90 != v91 );
          v140 = v150;
        }
      }
      if ( v140 != &v152 )
        _libc_free((unsigned __int64)v140);
    }
    v21 = (_QWORD **)(16LL * v149);
    sub_C7D6A0(v147, (__int64)v21, 8);
    nullsub_188();
    v93 = *(__int64 **)v3;
    if ( *(_QWORD *)v3 )
    {
      v94 = v158;
      v95 = &v158[8 * (unsigned int)v159];
      if ( v95 != (_QWORD *)v158 )
      {
        do
        {
          v21 = (_QWORD **)*v94++;
          sub_D4F330(v93, (__int64)v21, a2);
        }
        while ( v95 != v94 );
      }
      nullsub_188();
    }
  }
  sub_FFCE90((__int64)&v169, (__int64)v21, v22, v23, v24, v25);
  sub_FFD870((__int64)&v169, (__int64)v21, v26, v27, v28, v29);
  sub_FFBC40((__int64)&v169, (__int64)v21);
  v30 = v181;
  v31 = v180;
  if ( v181 != v180 )
  {
    do
    {
      v32 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v31[7];
      *v31 = &unk_49E5048;
      if ( v32 )
        v32(v31 + 5, v31 + 5, 3);
      *v31 = &unk_49DB368;
      v33 = v31[3];
      if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
        sub_BD60C0(v31 + 1);
      v31 += 9;
    }
    while ( v30 != v31 );
    v31 = v180;
  }
  if ( v31 )
    j_j___libc_free_0((unsigned __int64)v31);
  if ( !v177 )
    _libc_free((unsigned __int64)v174);
  if ( v169 != v171 )
    _libc_free((unsigned __int64)v169);
  if ( v158 != v160 )
    _libc_free((unsigned __int64)v158);
  if ( (_BYTE *)v152 != v154 )
    _libc_free(v152);
  if ( v155 != (__int64 *)v157 )
    _libc_free((unsigned __int64)v155);
  return v116;
}
