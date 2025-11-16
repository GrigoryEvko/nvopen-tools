// Function: sub_1E99E80
// Address: 0x1e99e80
//
__int64 __fastcall sub_1E99E80(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v5; // rax
  unsigned __int8 v6; // bl
  unsigned int v7; // eax
  unsigned __int8 v8; // al
  void *v9; // r8
  __int64 v10; // rdx
  unsigned __int8 v11; // bl
  __int64 v12; // rdi
  int v13; // r13d
  int k; // esi
  __int64 v15; // rbx
  __int64 v16; // r8
  int v17; // r9d
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned int v20; // edx
  __int64 v21; // r12
  __int64 v22; // r14
  int m; // esi
  __int64 v24; // rax
  bool v25; // zf
  __int64 v26; // rbx
  unsigned __int8 v27; // r14
  unsigned int v28; // r13d
  int i; // esi
  _QWORD *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  char v33; // dl
  unsigned int v34; // r11d
  _QWORD *v35; // r9
  int v36; // r10d
  _QWORD *v37; // rdi
  __int64 v38; // rcx
  _QWORD *v39; // rcx
  int v40; // eax
  unsigned int v41; // r14d
  __int64 v42; // r13
  __int64 v43; // rbx
  int j; // esi
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rdx
  char v48; // al
  unsigned int v49; // r9d
  __int64 *v50; // rdx
  unsigned int v51; // r11d
  __int64 *v52; // rax
  __int64 v53; // rdi
  int v54; // esi
  _QWORD *v55; // rbx
  _QWORD *v56; // r12
  __int64 v57; // rdi
  unsigned int v58; // ecx
  int v59; // edx
  __int64 v60; // r9
  int v61; // r12d
  _QWORD *v62; // rdx
  _QWORD *v63; // rcx
  _QWORD *v64; // r13
  __int64 v65; // r12
  __int64 v66; // rax
  __int64 *v67; // rdx
  __int64 v68; // r15
  __int64 v69; // rbx
  __int64 v70; // r13
  __int64 v71; // r12
  _QWORD *v72; // r14
  __int64 v73; // r8
  int v74; // r9d
  __int32 v75; // eax
  __int64 v76; // rbx
  __int32 v77; // r10d
  unsigned int v78; // edx
  __int64 *v79; // rax
  __int64 v80; // rcx
  __int64 v81; // r12
  __int64 v82; // rax
  unsigned __int64 v83; // rdx
  char *v84; // r13
  char *v85; // rax
  _BYTE *v86; // rdi
  char *v87; // r12
  _BYTE *v88; // r14
  char *v89; // rax
  __int64 v90; // rax
  unsigned __int64 v91; // rdx
  __int64 v92; // r8
  unsigned int v93; // r8d
  int v94; // r9d
  int v95; // esi
  unsigned int v96; // r9d
  _QWORD *v97; // rcx
  __int64 v98; // r8
  __int64 v99; // rax
  unsigned __int64 v100; // rdx
  _QWORD *v101; // r12
  _QWORD *v102; // rbx
  int v103; // r10d
  int v104; // eax
  __int64 v105; // r10
  __int64 *v106; // r10
  int v107; // r10d
  __int64 *v108; // rdx
  int v109; // eax
  __int64 v110; // r14
  __int64 v111; // r13
  __int64 v112; // r12
  __int64 v113; // rbx
  __int64 v114; // rdx
  __int64 v115; // rax
  int v116; // r9d
  __int64 *v117; // r11
  int v118; // edx
  unsigned int v119; // ecx
  __int64 v120; // rdi
  int v121; // r12d
  __int64 *v122; // rsi
  unsigned int v123; // r12d
  int v124; // r11d
  __int64 *v125; // rcx
  __int64 v126; // rsi
  unsigned int v127; // r8d
  __int64 v128; // rdi
  int v129; // esi
  __int64 *v130; // rcx
  int v131; // esi
  unsigned int v132; // r8d
  __int64 v133; // rdi
  _QWORD *v134; // rdx
  _QWORD *v135; // rdi
  int v136; // eax
  int v137; // r8d
  _QWORD *v138; // rsi
  __int64 v139; // [rsp+0h] [rbp-1A0h]
  __int64 *v140; // [rsp+8h] [rbp-198h]
  __int64 v141; // [rsp+10h] [rbp-190h]
  _BYTE *v142; // [rsp+20h] [rbp-180h]
  _QWORD *v143; // [rsp+28h] [rbp-178h]
  unsigned __int8 v144; // [rsp+32h] [rbp-16Eh]
  unsigned __int8 v145; // [rsp+33h] [rbp-16Dh]
  int v146; // [rsp+34h] [rbp-16Ch]
  _QWORD *v147; // [rsp+38h] [rbp-168h]
  _BYTE *v148; // [rsp+48h] [rbp-158h]
  __int64 v149; // [rsp+50h] [rbp-150h]
  int v150; // [rsp+50h] [rbp-150h]
  size_t *v151; // [rsp+58h] [rbp-148h]
  _QWORD *v152; // [rsp+60h] [rbp-140h]
  _QWORD *v153; // [rsp+60h] [rbp-140h]
  int v154; // [rsp+60h] [rbp-140h]
  __int64 v155; // [rsp+60h] [rbp-140h]
  __int64 v156; // [rsp+68h] [rbp-138h]
  __int32 v157; // [rsp+70h] [rbp-130h]
  __int32 v158; // [rsp+70h] [rbp-130h]
  int v159; // [rsp+70h] [rbp-130h]
  _QWORD *v160; // [rsp+70h] [rbp-130h]
  __int32 v161; // [rsp+70h] [rbp-130h]
  int v162; // [rsp+70h] [rbp-130h]
  int v163; // [rsp+70h] [rbp-130h]
  _QWORD *v164; // [rsp+70h] [rbp-130h]
  int v165; // [rsp+78h] [rbp-128h]
  _QWORD *v166; // [rsp+78h] [rbp-128h]
  _QWORD *v167; // [rsp+78h] [rbp-128h]
  __int64 v168; // [rsp+78h] [rbp-128h]
  unsigned int v169; // [rsp+78h] [rbp-128h]
  _QWORD *v170; // [rsp+78h] [rbp-128h]
  __int64 v171; // [rsp+78h] [rbp-128h]
  __int32 v172; // [rsp+78h] [rbp-128h]
  __int32 v173; // [rsp+78h] [rbp-128h]
  __int64 v174; // [rsp+80h] [rbp-120h] BYREF
  _QWORD *v175; // [rsp+88h] [rbp-118h]
  __int64 v176; // [rsp+90h] [rbp-110h]
  unsigned int v177; // [rsp+98h] [rbp-108h]
  __int64 v178; // [rsp+A0h] [rbp-100h] BYREF
  __int64 v179; // [rsp+A8h] [rbp-F8h]
  __int64 v180; // [rsp+B0h] [rbp-F0h]
  unsigned int v181; // [rsp+B8h] [rbp-E8h]
  __m128i v182; // [rsp+C0h] [rbp-E0h] BYREF
  void *s; // [rsp+D0h] [rbp-D0h]
  __int128 v184; // [rsp+D8h] [rbp-C8h]
  _BYTE v185[184]; // [rsp+E8h] [rbp-B8h] BYREF

  v142 = (_BYTE *)(a2 + 24);
  v156 = *(_QWORD *)(a2 + 32);
  v144 = 0;
  if ( a2 + 24 != v156 )
  {
    do
    {
      if ( !v156 )
        BUG();
      v3 = v156;
      if ( (*(_BYTE *)v156 & 4) == 0 && (*(_BYTE *)(v156 + 46) & 8) != 0 )
      {
        do
          v3 = *(_QWORD *)(v3 + 8);
        while ( (*(_BYTE *)(v3 + 46) & 8) != 0 );
      }
      v148 = *(_BYTE **)(v3 + 8);
      if ( **(_WORD **)(v156 + 16) != 45 && **(_WORD **)(v156 + 16) )
        return v144;
      v145 = *(_BYTE *)(a1 + 256);
      if ( !v145 )
        goto LABEL_12;
      if ( !dword_4FC84A0 )
      {
        v5 = v156;
        if ( *(_DWORD *)(v156 + 40) > 2u )
          goto LABEL_26;
        goto LABEL_12;
      }
      v174 = 0;
      v175 = 0;
      v177 = 0;
      v25 = *(_DWORD *)(v156 + 40) == 1;
      v176 = 0;
      if ( v25 )
      {
        v101 = 0;
LABEL_175:
        j___libc_free_0(v101);
        v5 = v156;
        if ( *(_DWORD *)(v156 + 40) > 2u )
          goto LABEL_26;
LABEL_12:
        LODWORD(v178) = 0;
        v182.m128i_i64[0] = 0;
        v182.m128i_i64[1] = (__int64)v185;
        s = v185;
        *(_QWORD *)&v184 = 16;
        DWORD2(v184) = 0;
        v6 = sub_1E993A0(a1, v156, (int *)&v178, (__int64)&v182);
        if ( v6 && (_DWORD)v178 )
        {
          v61 = *(_DWORD *)(*(_QWORD *)(v156 + 32) + 8LL);
          if ( sub_1E69410(
                 *(__int64 **)(a1 + 232),
                 v178,
                 *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 232) + 24LL) + 16LL * (v61 & 0x7FFFFFFF))
               & 0xFFFFFFFFFFFFFFF8LL,
                 0) )
          {
            sub_1E69BA0(*(_QWORD **)(a1 + 232), v61, v178);
            sub_1E16240(v156);
            v144 = v6;
          }
          if ( s != (void *)v182.m128i_i64[1] )
            _libc_free((unsigned __int64)s);
          goto LABEL_99;
        }
        ++v182.m128i_i64[0];
        if ( s == (void *)v182.m128i_i64[1] )
        {
LABEL_19:
          *(_QWORD *)((char *)&v184 + 4) = 0;
        }
        else
        {
          v7 = 4 * (DWORD1(v184) - DWORD2(v184));
          if ( v7 < 0x20 )
            v7 = 32;
          if ( v7 >= (unsigned int)v184 )
          {
            memset(s, -1, 8LL * (unsigned int)v184);
            goto LABEL_19;
          }
          sub_16CC920((__int64)&v182);
        }
        v8 = sub_1E99220(a1, v156, (__int64)&v182);
        v9 = s;
        v10 = v182.m128i_i64[1];
        v11 = v8;
        if ( v8 )
        {
          if ( s == (void *)v182.m128i_i64[1] )
            v84 = (char *)s + 8 * DWORD1(v184);
          else
            v84 = (char *)s + 8 * (unsigned int)v184;
          if ( s != v84 )
          {
            v85 = (char *)s;
            while ( 1 )
            {
              v86 = *(_BYTE **)v85;
              v87 = v85;
              if ( *(_QWORD *)v85 < 0xFFFFFFFFFFFFFFFELL )
                break;
              v85 += 8;
              if ( v84 == v85 )
                goto LABEL_137;
            }
            if ( v85 != v84 )
            {
              v88 = v148;
              if ( v86 == v148 )
                goto LABEL_146;
              while ( 1 )
              {
                sub_1E16240((__int64)v86);
                v89 = v87 + 8;
                if ( v87 + 8 == v84 )
                  break;
                while ( 1 )
                {
                  v86 = *(_BYTE **)v89;
                  v87 = v89;
                  if ( *(_QWORD *)v89 < 0xFFFFFFFFFFFFFFFELL )
                    break;
                  v89 += 8;
                  if ( v84 == v89 )
                    goto LABEL_143;
                }
                if ( v89 == v84 )
                  break;
                if ( v86 == v88 )
                {
LABEL_146:
                  if ( !v86 )
                    BUG();
                  if ( (*v86 & 4) == 0 && (v86[46] & 8) != 0 )
                  {
                    do
                      v88 = (_BYTE *)*((_QWORD *)v88 + 1);
                    while ( (v88[46] & 8) != 0 );
                  }
                  v88 = (_BYTE *)*((_QWORD *)v88 + 1);
                }
              }
LABEL_143:
              v148 = v88;
              v9 = s;
              v10 = v182.m128i_i64[1];
            }
          }
LABEL_137:
          v144 = v11;
          v156 = (__int64)v148;
        }
        else
        {
          v156 = (__int64)v148;
        }
        if ( (void *)v10 != v9 )
          _libc_free((unsigned __int64)v9);
        continue;
      }
      v26 = v141;
      v27 = 0;
      v28 = 1;
      v143 = 0;
      v146 = 0;
      do
      {
        for ( i = *(_DWORD *)(*(_QWORD *)(v156 + 32) + 40LL * v28 + 8); ; i = *(_DWORD *)(v30[4] + 48LL) )
        {
          v30 = (_QWORD *)sub_1E69D00(*(_QWORD *)(a1 + 232), i);
          v31 = v30[2];
          if ( *(_WORD *)v31 != 15 )
            break;
        }
        if ( (*(_BYTE *)(v31 + 9) & 4) != 0 )
        {
          v32 = v30[4];
          v33 = *(_BYTE *)(v32 + 40);
          if ( dword_4FC84A0 != 1 )
          {
            switch ( v33 )
            {
              case 3:
LABEL_87:
                v146 = 1;
                v26 = *(_QWORD *)(v32 + 64);
                break;
              case 2:
                v146 = 2;
                v26 = *(_QWORD *)(v32 + 64);
                break;
              case 1:
                v146 = 3;
                v26 = *(_QWORD *)(v32 + 64);
                break;
            }
            if ( v177 )
            {
              v34 = v177 - 1;
              v35 = &v175[7 * v177];
              v36 = 37 * v26;
              v37 = &v175[7 * ((v177 - 1) & (37 * (_DWORD)v26))];
              v38 = *v37;
              if ( *v37 == v26 )
              {
                v39 = v143;
                if ( v35 != v37 )
                  v39 = v30;
                v143 = v39;
                if ( v35 != v37 )
                  v27 = v145;
                goto LABEL_56;
              }
              v159 = 1;
              v169 = (v177 - 1) & (37 * v26);
              v153 = &v175[7 * v177];
              v92 = *v37;
              while ( v92 != -1 )
              {
                v150 = v159 + 1;
                v169 = v34 & (v159 + v169);
                v92 = v175[7 * v169];
                v164 = &v175[7 * v169];
                if ( v92 == v26 )
                {
                  v170 = &v175[7 * (v34 & (37 * (_DWORD)v26))];
                  v93 = (v177 - 1) & (37 * v26);
                  if ( v153 != v164 )
                    v27 = v145;
                  v135 = v143;
                  if ( v153 != v164 )
                    v135 = v30;
                  v143 = v135;
                  goto LABEL_155;
                }
                v159 = v150;
              }
              v36 = 37 * v26;
              v93 = v34 & (37 * v26);
              v170 = &v175[7 * v93];
              v38 = *v170;
              if ( *v170 == v26 )
              {
                v37 = &v175[7 * (v34 & (37 * (_DWORD)v26))];
LABEL_56:
                v182.m128i_i64[0] = v30[3];
                sub_1E99A80(v37 + 1, v182.m128i_i64);
                goto LABEL_57;
              }
LABEL_155:
              v94 = 1;
              v37 = 0;
              while ( v38 != -1 )
              {
                if ( v37 || v38 != -2 )
                  v170 = v37;
                v93 = v34 & (v94 + v93);
                v37 = &v175[7 * v93];
                v38 = *v37;
                if ( *v37 == v26 )
                  goto LABEL_56;
                ++v94;
                v134 = v170;
                v170 = &v175[7 * v93];
                v37 = v134;
              }
              if ( !v37 )
                v37 = v170;
              ++v174;
              v59 = v176 + 1;
              if ( 4 * ((int)v176 + 1) < 3 * v177 )
              {
                if ( v177 - HIDWORD(v176) - v59 <= v177 >> 3 )
                {
                  v154 = v36;
                  v160 = v30;
                  sub_1E99BB0((__int64)&v174, v177);
                  if ( !v177 )
                  {
LABEL_300:
                    LODWORD(v176) = v176 + 1;
                    BUG();
                  }
                  v95 = 1;
                  v96 = (v177 - 1) & v154;
                  v97 = 0;
                  v37 = &v175[7 * v96];
                  v59 = v176 + 1;
                  v30 = v160;
                  v98 = *v37;
                  if ( *v37 != v26 )
                  {
                    while ( v98 != -1 )
                    {
                      if ( v98 == -2 && !v97 )
                        v97 = v37;
                      v96 = (v177 - 1) & (v95 + v96);
                      v37 = &v175[7 * v96];
                      v98 = *v37;
                      if ( *v37 == v26 )
                        goto LABEL_91;
                      ++v95;
                    }
                    if ( v97 )
                      v37 = v97;
                  }
                }
LABEL_91:
                LODWORD(v176) = v59;
                if ( *v37 != -1 )
                  --HIDWORD(v176);
                *v37 = v26;
                *((_DWORD *)v37 + 4) = 0;
                v37[3] = 0;
                v37[4] = v37 + 2;
                v37[5] = v37 + 2;
                v37[6] = 0;
                goto LABEL_56;
              }
            }
            else
            {
              ++v174;
            }
            v166 = v30;
            sub_1E99BB0((__int64)&v174, 2 * v177);
            if ( !v177 )
              goto LABEL_300;
            v58 = (v177 - 1) & (37 * v26);
            v37 = &v175[7 * v58];
            v59 = v176 + 1;
            v30 = v166;
            v60 = *v37;
            if ( *v37 != v26 )
            {
              v137 = 1;
              v138 = 0;
              while ( v60 != -1 )
              {
                if ( !v138 && v60 == -2 )
                  v138 = v37;
                v58 = (v177 - 1) & (v137 + v58);
                v37 = &v175[7 * v58];
                v60 = *v37;
                if ( *v37 == v26 )
                  goto LABEL_91;
                ++v137;
              }
              if ( v138 )
                v37 = v138;
            }
            goto LABEL_91;
          }
          if ( v33 == 3 )
            goto LABEL_87;
        }
LABEL_57:
        v40 = *(_DWORD *)(v156 + 40);
        v28 += 2;
      }
      while ( v40 != v28 );
      v141 = v26;
      if ( !v27 )
      {
        v101 = v175;
        if ( v177 )
        {
          v102 = &v175[7 * v177];
          do
          {
            if ( *v101 <= 0xFFFFFFFFFFFFFFFDLL )
              sub_1E99580(v101[3]);
            v101 += 7;
          }
          while ( v102 != v101 );
          v101 = v175;
        }
        goto LABEL_175;
      }
      v178 = 0;
      v179 = 0;
      v180 = 0;
      v181 = 0;
      if ( (_DWORD)v176 )
      {
        v62 = v175;
        v63 = &v175[7 * v177];
        v147 = v63;
        if ( v175 != v63 )
        {
          while ( 1 )
          {
            v64 = v62;
            if ( *v62 <= 0xFFFFFFFFFFFFFFFDLL )
              break;
            v62 += 7;
            if ( v63 == v62 )
              goto LABEL_60;
          }
          if ( v63 == v62 )
            goto LABEL_60;
          v151 = (size_t *)a1;
          v140 = (__int64 *)(v156 + 64);
LABEL_106:
          if ( v64[6] == 1 )
            goto LABEL_122;
          v65 = v64[4];
          v167 = v64 + 2;
          if ( (_QWORD *)v65 == v64 + 2 )
            goto LABEL_122;
          v152 = v64;
          while ( 1 )
          {
            v66 = *(_QWORD *)(v65 + 32);
            v67 = *(__int64 **)(v66 + 64);
            if ( (unsigned int)((__int64)(*(_QWORD *)(v66 + 72) - (_QWORD)v67) >> 3) == 1 )
            {
              v68 = *v67;
              v69 = v151[31];
              v70 = v152[4];
              if ( v167 == (_QWORD *)v70 )
              {
LABEL_116:
                v64 = v152;
                if ( v68 )
                {
                  v72 = (_QWORD *)sub_1DD5EE0(v68);
                  v75 = sub_1E6B9A0(
                          v151[29],
                          *(_QWORD *)(*(_QWORD *)(v151[29] + 24)
                                    + 16LL * (*(_DWORD *)(*(_QWORD *)(v156 + 32) + 8LL) & 0x7FFFFFFF))
                        & 0xFFFFFFFFFFFFFFF8LL,
                          (unsigned __int8 *)byte_3F871B3,
                          0,
                          v73,
                          v74);
                  v76 = *v152;
                  v77 = v75;
                  if ( v181 )
                  {
                    v78 = (v181 - 1) & (37 * v76);
                    v79 = (__int64 *)(v179 + 16LL * v78);
                    v80 = *v79;
                    if ( v76 == *v79 )
                      goto LABEL_119;
                    v116 = 1;
                    v117 = 0;
                    while ( v80 != -1 )
                    {
                      if ( !v117 && v80 == -2 )
                        v117 = v79;
                      v78 = (v181 - 1) & (v116 + v78);
                      v79 = (__int64 *)(v179 + 16LL * v78);
                      v80 = *v79;
                      if ( v76 == *v79 )
                        goto LABEL_119;
                      ++v116;
                    }
                    if ( v117 )
                      v79 = v117;
                    ++v178;
                    v118 = v180 + 1;
                    if ( 4 * ((int)v180 + 1) < 3 * v181 )
                    {
                      if ( v181 - HIDWORD(v180) - v118 <= v181 >> 3 )
                      {
                        v173 = v77;
                        sub_1556300((__int64)&v178, v181);
                        if ( !v181 )
                        {
LABEL_298:
                          LODWORD(v180) = v180 + 1;
                          BUG();
                        }
                        v123 = (v181 - 1) & (37 * v76);
                        v124 = 1;
                        v77 = v173;
                        v118 = v180 + 1;
                        v125 = 0;
                        v79 = (__int64 *)(v179 + 16LL * v123);
                        v126 = *v79;
                        if ( v76 != *v79 )
                        {
                          while ( v126 != -1 )
                          {
                            if ( v126 == -2 && !v125 )
                              v125 = v79;
                            v123 = (v181 - 1) & (v124 + v123);
                            v79 = (__int64 *)(v179 + 16LL * v123);
                            v126 = *v79;
                            if ( v76 == *v79 )
                              goto LABEL_207;
                            ++v124;
                          }
                          if ( v125 )
                            v79 = v125;
                        }
                      }
LABEL_207:
                      LODWORD(v180) = v118;
                      if ( *v79 != -1 )
                        --HIDWORD(v180);
                      *v79 = v76;
                      *((_DWORD *)v79 + 2) = 0;
LABEL_119:
                      *((_DWORD *)v79 + 2) = v77;
                      switch ( v146 )
                      {
                        case 1:
                          v157 = v77;
                          v81 = *(_QWORD *)(v68 + 56);
                          v168 = (__int64)sub_1E0B640(
                                            v81,
                                            *(_QWORD *)(v151[30] + 8)
                                          + ((unsigned __int64)*(unsigned __int16 *)v143[2] << 6),
                                            v140,
                                            0);
                          sub_1DD5BA0((__int64 *)(v68 + 16), v168);
                          v82 = *(_QWORD *)v168;
                          v83 = *v72 & 0xFFFFFFFFFFFFFFF8LL;
                          *(_QWORD *)(v168 + 8) = v72;
                          *(_QWORD *)v168 = v83 | v82 & 7;
                          *(_QWORD *)(v83 + 8) = v168;
                          *v72 = v168 | *v72 & 7LL;
                          v182.m128i_i64[0] = 0x10000000;
                          s = 0;
                          v182.m128i_i32[2] = v157;
                          v184 = 0u;
                          sub_1E1A9C0(v168, v81, &v182);
                          v182.m128i_i64[0] = 3;
                          break;
                        case 2:
                          v158 = v77;
                          v81 = *(_QWORD *)(v68 + 56);
                          v168 = (__int64)sub_1E0B640(
                                            v81,
                                            *(_QWORD *)(v151[30] + 8)
                                          + ((unsigned __int64)*(unsigned __int16 *)v143[2] << 6),
                                            v140,
                                            0);
                          sub_1DD5BA0((__int64 *)(v68 + 16), v168);
                          v90 = *(_QWORD *)v168;
                          v91 = *v72 & 0xFFFFFFFFFFFFFFF8LL;
                          *(_QWORD *)(v168 + 8) = v72;
                          *(_QWORD *)v168 = v91 | v90 & 7;
                          *(_QWORD *)(v91 + 8) = v168;
                          *v72 = v168 | *v72 & 7LL;
                          v182.m128i_i64[0] = 0x10000000;
                          s = 0;
                          v182.m128i_i32[2] = v158;
                          v184 = 0u;
                          sub_1E1A9C0(v168, v81, &v182);
                          v182.m128i_i64[0] = 2;
                          break;
                        case 3:
                          v161 = v77;
                          v81 = *(_QWORD *)(v68 + 56);
                          v168 = (__int64)sub_1E0B640(
                                            v81,
                                            *(_QWORD *)(v151[30] + 8)
                                          + ((unsigned __int64)*(unsigned __int16 *)v143[2] << 6),
                                            v140,
                                            0);
                          sub_1DD5BA0((__int64 *)(v68 + 16), v168);
                          v99 = *(_QWORD *)v168;
                          v100 = *v72 & 0xFFFFFFFFFFFFFFF8LL;
                          *(_QWORD *)(v168 + 8) = v72;
                          *(_QWORD *)v168 = v100 | v99 & 7;
                          *(_QWORD *)(v100 + 8) = v168;
                          *v72 = v168 | *v72 & 7LL;
                          v182.m128i_i64[0] = 0x10000000;
                          s = 0;
                          v182.m128i_i32[2] = v161;
                          v184 = 0u;
                          sub_1E1A9C0(v168, v81, &v182);
                          v182.m128i_i64[0] = 1;
                          break;
                        default:
                          goto LABEL_122;
                      }
                      s = 0;
                      *(_QWORD *)&v184 = v76;
                      sub_1E1A9C0(v168, v81, &v182);
                      goto LABEL_122;
                    }
                  }
                  else
                  {
                    ++v178;
                  }
                  v172 = v77;
                  sub_1556300((__int64)&v178, 2 * v181);
                  if ( !v181 )
                    goto LABEL_298;
                  v77 = v172;
                  v119 = (v181 - 1) & (37 * v76);
                  v118 = v180 + 1;
                  v79 = (__int64 *)(v179 + 16LL * v119);
                  v120 = *v79;
                  if ( v76 != *v79 )
                  {
                    v121 = 1;
                    v122 = 0;
                    while ( v120 != -1 )
                    {
                      if ( !v122 && v120 == -2 )
                        v122 = v79;
                      v119 = (v181 - 1) & (v121 + v119);
                      v79 = (__int64 *)(v179 + 16LL * v119);
                      v120 = *v79;
                      if ( v76 == *v79 )
                        goto LABEL_207;
                      ++v121;
                    }
                    if ( v122 )
                      v79 = v122;
                  }
                  goto LABEL_207;
                }
LABEL_122:
                v64 += 7;
                if ( v64 == v147 )
                  goto LABEL_125;
                while ( *v64 > 0xFFFFFFFFFFFFFFFDLL )
                {
                  v64 += 7;
                  if ( v147 == v64 )
                    goto LABEL_125;
                }
                if ( v147 == v64 )
                {
LABEL_125:
                  a1 = (__int64)v151;
                  v40 = *(_DWORD *)(v156 + 40);
                  break;
                }
                goto LABEL_106;
              }
              v149 = v65;
              while ( 1 )
              {
                v71 = *(_QWORD *)(v70 + 32);
                sub_1E06620(v69);
                if ( !sub_1E05550(*(_QWORD *)(v69 + 1312), v68, v71) )
                  break;
                v70 = sub_220EF30(v70);
                if ( v167 == (_QWORD *)v70 )
                  goto LABEL_116;
              }
              v65 = v149;
            }
            v65 = sub_220EF30(v65);
            if ( v167 == (_QWORD *)v65 )
            {
              v64 = v152;
              goto LABEL_122;
            }
          }
        }
      }
LABEL_60:
      v41 = 1;
      if ( v40 == 1 )
        goto LABEL_77;
      v42 = v139;
      while ( 2 )
      {
        v43 = 40LL * v41;
        for ( j = *(_DWORD *)(*(_QWORD *)(v156 + 32) + v43 + 8); ; j = *(_DWORD *)(*(_QWORD *)(v45 + 32) + 48LL) )
        {
          v45 = sub_1E69D00(*(_QWORD *)(a1 + 232), j);
          v46 = *(_QWORD *)(v45 + 16);
          if ( *(_WORD *)v46 != 15 )
            break;
        }
        if ( (*(_BYTE *)(v46 + 9) & 4) == 0 )
          goto LABEL_75;
        v47 = *(_QWORD *)(v45 + 32);
        v48 = *(_BYTE *)(v47 + 40);
        if ( dword_4FC84A0 == 1 )
        {
          if ( v48 != 3 )
            goto LABEL_75;
LABEL_85:
          v42 = *(_QWORD *)(v47 + 64);
          goto LABEL_70;
        }
        if ( v48 == 3 || v48 == 2 || v48 == 1 )
          goto LABEL_85;
LABEL_70:
        if ( !v181 )
          goto LABEL_75;
        v49 = v181 - 1;
        v50 = (__int64 *)(v179 + 16LL * v181);
        v51 = (v181 - 1) & (37 * v42);
        v52 = (__int64 *)(v179 + 16LL * v51);
        v53 = *v52;
        if ( *v52 == v42 )
        {
          if ( v50 != v52 )
          {
LABEL_73:
            v54 = *((_DWORD *)v52 + 2);
            goto LABEL_74;
          }
          goto LABEL_75;
        }
        v171 = *v52;
        v103 = 1;
        v162 = (v181 - 1) & (37 * v42);
        while ( 1 )
        {
          if ( v171 == -1 )
            goto LABEL_75;
          v104 = v103 + 1;
          v105 = v49 & (v162 + v103);
          v162 = v105;
          v106 = (__int64 *)(v179 + 16 * v105);
          v171 = *v106;
          if ( *v106 == v42 )
            break;
          v103 = v104;
        }
        v52 = (__int64 *)(v179 + 16LL * (v49 & (37 * (_DWORD)v42)));
        if ( v50 == v106 )
          goto LABEL_75;
        v107 = 1;
        v108 = 0;
        while ( v53 != -1 )
        {
          if ( v53 == -2 && !v108 )
            v108 = v52;
          v136 = v107++;
          v51 = v49 & (v136 + v51);
          v52 = (__int64 *)(v179 + 16LL * v51);
          v53 = *v52;
          if ( *v52 == v42 )
            goto LABEL_73;
        }
        if ( !v108 )
          v108 = v52;
        ++v178;
        v109 = v180 + 1;
        if ( 4 * ((int)v180 + 1) >= 3 * v181 )
        {
          sub_1556300((__int64)&v178, 2 * v181);
          if ( !v181 )
            goto LABEL_299;
          v127 = (v181 - 1) & (37 * v42);
          v109 = v180 + 1;
          v108 = (__int64 *)(v179 + 16LL * v127);
          v128 = *v108;
          if ( *v108 != v42 )
          {
            v129 = 1;
            v130 = 0;
            while ( v128 != -1 )
            {
              if ( !v130 && v128 == -2 )
                v130 = v108;
              v127 = (v181 - 1) & (v129 + v127);
              v108 = (__int64 *)(v179 + 16LL * v127);
              v128 = *v108;
              if ( *v108 == v42 )
                goto LABEL_187;
              ++v129;
            }
LABEL_230:
            if ( v130 )
              v108 = v130;
          }
        }
        else if ( v181 - HIDWORD(v180) - v109 <= v181 >> 3 )
        {
          sub_1556300((__int64)&v178, v181);
          if ( !v181 )
          {
LABEL_299:
            LODWORD(v180) = v180 + 1;
            BUG();
          }
          v131 = 1;
          v130 = 0;
          v132 = (v181 - 1) & (37 * v42);
          v109 = v180 + 1;
          v108 = (__int64 *)(v179 + 16LL * v132);
          v133 = *v108;
          if ( *v108 != v42 )
          {
            while ( v133 != -1 )
            {
              if ( v133 == -2 && !v130 )
                v130 = v108;
              v132 = (v181 - 1) & (v131 + v132);
              v108 = (__int64 *)(v179 + 16LL * v132);
              v133 = *v108;
              if ( *v108 == v42 )
                goto LABEL_187;
              ++v131;
            }
            goto LABEL_230;
          }
        }
LABEL_187:
        LODWORD(v180) = v109;
        if ( *v108 != -1 )
          --HIDWORD(v180);
        *v108 = v42;
        v54 = 0;
        *((_DWORD *)v108 + 2) = 0;
LABEL_74:
        sub_1E310D0(*(_QWORD *)(v156 + 32) + v43, v54);
LABEL_75:
        v41 += 2;
        if ( *(_DWORD *)(v156 + 40) != v41 )
          continue;
        break;
      }
      v139 = v42;
LABEL_77:
      j___libc_free_0(v179);
      if ( v177 )
      {
        v55 = v175;
        v56 = &v175[7 * v177];
        do
        {
          while ( *v55 > 0xFFFFFFFFFFFFFFFDLL )
          {
            v55 += 7;
            if ( v56 == v55 )
              goto LABEL_82;
          }
          v57 = v55[3];
          v55 += 7;
          sub_1E99580(v57);
        }
        while ( v56 != v55 );
      }
LABEL_82:
      j___libc_free_0(v175);
      v5 = v156;
      if ( *(_DWORD *)(v156 + 40) <= 2u )
        goto LABEL_12;
LABEL_26:
      v12 = *(_QWORD *)(a1 + 232);
      v13 = *(_DWORD *)(*(_QWORD *)(v5 + 32) + 48LL);
      for ( k = v13; ; k = *(_DWORD *)(*(_QWORD *)(v15 + 32) + 48LL) )
      {
        v15 = sub_1E69D00(v12, k);
        v18 = *(_QWORD *)(v15 + 16);
        if ( *(_WORD *)v18 != 15 )
          break;
        v12 = *(_QWORD *)(a1 + 232);
      }
      if ( (*(_BYTE *)(v18 + 9) & 4) == 0 || *(_BYTE *)(*(_QWORD *)(v15 + 32) + 40LL) != 1 )
        goto LABEL_12;
      v19 = *(_QWORD *)(v156 + 32);
      v20 = *(_DWORD *)(v156 + 40);
      v165 = *(_DWORD *)(v19 + 8);
      if ( v20 > 3 )
      {
        v21 = 120;
        v22 = 80LL * ((v20 - 4) >> 1) + 200;
        while ( 1 )
        {
          for ( m = *(_DWORD *)(v19 + v21 + 8); ; m = *(_DWORD *)(*(_QWORD *)(v24 + 32) + 48LL) )
          {
            v24 = sub_1E69D00(*(_QWORD *)(a1 + 232), m);
            if ( **(_WORD **)(v24 + 16) != 15 )
              break;
          }
          if ( !(unsigned __int8)sub_1E15D60(v24, v15, 2u) )
            goto LABEL_12;
          v21 += 80;
          if ( v22 == v21 )
            break;
          v19 = *(_QWORD *)(v156 + 32);
        }
      }
      v163 = sub_1E6B9A0(
               *(_QWORD *)(a1 + 232),
               *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 232) + 24LL) + 16LL * (v13 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
               (unsigned __int8 *)byte_3F871B3,
               0,
               v16,
               v17);
      v110 = *(_QWORD *)(v156 + 24);
      if ( v156 == v110 + 24 )
      {
        v111 = v156;
      }
      else
      {
        v111 = v156;
        do
        {
          if ( **(_WORD **)(v111 + 16) && **(_WORD **)(v111 + 16) != 45 )
            break;
          if ( (*(_BYTE *)v111 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v111 + 46) & 8) != 0 )
              v111 = *(_QWORD *)(v111 + 8);
          }
          v111 = *(_QWORD *)(v111 + 8);
        }
        while ( v110 + 24 != v111 );
      }
      v112 = *(_QWORD *)(v110 + 56);
      v155 = *(_QWORD *)(*(_QWORD *)(v15 + 32) + 64LL);
      v113 = (__int64)sub_1E0B640(
                        v112,
                        *(_QWORD *)(*(_QWORD *)(a1 + 240) + 8LL)
                      + ((unsigned __int64)**(unsigned __int16 **)(v15 + 16) << 6),
                        (__int64 *)(v156 + 64),
                        0);
      sub_1DD5BA0((__int64 *)(v110 + 16), v113);
      v114 = *(_QWORD *)v111;
      v115 = *(_QWORD *)v113;
      *(_QWORD *)(v113 + 8) = v111;
      v114 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v113 = v114 | v115 & 7;
      *(_QWORD *)(v114 + 8) = v113;
      *(_QWORD *)v111 = v113 | *(_QWORD *)v111 & 7LL;
      v182.m128i_i64[0] = 0x10000000;
      s = 0;
      v182.m128i_i32[2] = v163;
      v184 = 0u;
      sub_1E1A9C0(v113, v112, &v182);
      v182.m128i_i64[0] = 1;
      s = 0;
      *(_QWORD *)&v184 = v155;
      sub_1E1A9C0(v113, v112, &v182);
      sub_1E69C40(*(_QWORD *)(a1 + 232), v165, v163);
      sub_1E16240(v156);
      v144 = v145;
LABEL_99:
      v156 = (__int64)v148;
    }
    while ( v142 != v148 );
  }
  return v144;
}
