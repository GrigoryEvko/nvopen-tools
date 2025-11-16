// Function: sub_2F26B60
// Address: 0x2f26b60
//
__int64 __fastcall sub_2F26B60(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v9; // rcx
  unsigned __int8 v10; // bl
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned int v13; // eax
  unsigned __int8 v14; // bl
  char v15; // dl
  int v16; // r13d
  unsigned __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // rax
  unsigned int v20; // edx
  __int64 v21; // r12
  __int64 v22; // r15
  unsigned __int64 j; // rax
  int v24; // eax
  unsigned int v25; // r12d
  __int64 v26; // r13
  unsigned __int64 v27; // rax
  _QWORD *v28; // rbx
  __int64 v29; // rdx
  char v30; // al
  unsigned int v31; // r8d
  int v32; // r11d
  unsigned int v33; // eax
  __int64 *v34; // rdi
  __int64 v35; // rcx
  unsigned __int8 v36; // al
  _QWORD *v37; // rax
  _QWORD *v38; // rdi
  int v39; // eax
  unsigned int v40; // r12d
  __int64 v41; // r15
  __int64 v42; // rbx
  unsigned __int64 i; // rax
  __int64 v44; // rdx
  char v45; // al
  unsigned int v46; // ecx
  __int64 *v47; // rdx
  __int64 v48; // r8
  unsigned int v49; // eax
  _QWORD *v50; // rbx
  _QWORD *v51; // r12
  unsigned __int64 v52; // rdi
  unsigned int v53; // edx
  _QWORD *v54; // r9
  __int64 v55; // rsi
  int v56; // eax
  int v57; // r12d
  _QWORD *v58; // rdx
  _QWORD *v59; // rcx
  _QWORD *v60; // r13
  __int64 *v61; // r15
  __int64 v62; // r12
  _QWORD *v63; // r13
  __int64 v64; // rbx
  __int64 v65; // r14
  __int64 v66; // r14
  __int64 v67; // r8
  __int64 v68; // r9
  __int32 v69; // ebx
  int v70; // r12d
  _QWORD *v71; // rax
  unsigned int v72; // edx
  _QWORD *v73; // rcx
  __int64 v74; // r8
  __int32 *v75; // rax
  __int64 v76; // rcx
  unsigned __int8 *v77; // rsi
  _QWORD *v78; // rax
  __int64 v79; // rsi
  __int64 v80; // rdx
  __int64 v81; // rdi
  __int64 v82; // rcx
  unsigned __int8 *v83; // rsi
  _QWORD *v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rcx
  unsigned __int8 *v87; // rsi
  _QWORD *v88; // rax
  __int64 v89; // rdx
  _BYTE **v90; // rax
  __int64 v91; // r13
  _BYTE *v92; // rdi
  _BYTE **v93; // r12
  __int64 v94; // r9
  int v95; // r10d
  __int64 v96; // rdi
  _QWORD *v97; // r10
  int v98; // ecx
  unsigned int v99; // edi
  _QWORD *v100; // rdx
  __int64 v101; // rsi
  _BYTE *v102; // r15
  _BYTE **v103; // rax
  int v104; // edx
  int v105; // r9d
  _QWORD *v106; // r12
  __int64 v107; // rsi
  _QWORD *v108; // rbx
  __int32 v109; // r12d
  __int64 v110; // r13
  __int64 *v111; // r9
  __int64 v112; // r15
  __int64 v113; // rcx
  unsigned __int8 *v114; // rsi
  _QWORD *v115; // rax
  __int64 v116; // rdx
  int v117; // edx
  __int64 v118; // rdx
  unsigned int v119; // ecx
  __int64 v120; // rdi
  int v121; // r12d
  _QWORD *v122; // rsi
  int v123; // r12d
  unsigned int v124; // ecx
  __int64 v125; // rdi
  int v126; // r9d
  __int64 v127; // r10
  __int64 *v128; // r10
  bool v129; // zf
  unsigned __int8 v130; // r10
  _QWORD *v131; // r10
  int v132; // r11d
  _QWORD *v133; // rdi
  __int64 v134; // [rsp+8h] [rbp-1A8h]
  __int64 v135; // [rsp+8h] [rbp-1A8h]
  __int64 v136; // [rsp+8h] [rbp-1A8h]
  __int64 v137; // [rsp+18h] [rbp-198h]
  __int64 v138; // [rsp+20h] [rbp-190h]
  _BYTE *v139; // [rsp+28h] [rbp-188h]
  unsigned __int8 v140; // [rsp+32h] [rbp-17Eh]
  unsigned __int8 v141; // [rsp+33h] [rbp-17Dh]
  int v142; // [rsp+34h] [rbp-17Ch]
  _QWORD *v143; // [rsp+38h] [rbp-178h]
  _QWORD *v144; // [rsp+48h] [rbp-168h]
  __int64 *v145; // [rsp+48h] [rbp-168h]
  int v146; // [rsp+48h] [rbp-168h]
  int v147; // [rsp+50h] [rbp-160h]
  _QWORD *v148; // [rsp+50h] [rbp-160h]
  __int64 *v149; // [rsp+50h] [rbp-160h]
  int v150; // [rsp+50h] [rbp-160h]
  int v151; // [rsp+50h] [rbp-160h]
  int v152; // [rsp+50h] [rbp-160h]
  _BYTE *v153; // [rsp+58h] [rbp-158h]
  unsigned __int8 v154; // [rsp+60h] [rbp-150h]
  __int64 v155; // [rsp+60h] [rbp-150h]
  __int64 v156; // [rsp+60h] [rbp-150h]
  __int64 v157; // [rsp+68h] [rbp-148h]
  unsigned __int8 *v158; // [rsp+78h] [rbp-138h] BYREF
  unsigned __int8 *v159; // [rsp+80h] [rbp-130h] BYREF
  __int64 v160; // [rsp+88h] [rbp-128h]
  __int64 v161; // [rsp+90h] [rbp-120h]
  unsigned __int8 *v162; // [rsp+A0h] [rbp-110h] BYREF
  _QWORD *v163; // [rsp+A8h] [rbp-108h]
  __int64 v164; // [rsp+B0h] [rbp-100h]
  unsigned int v165; // [rsp+B8h] [rbp-F8h]
  unsigned __int8 *v166; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v167; // [rsp+C8h] [rbp-E8h]
  __int64 v168; // [rsp+D0h] [rbp-E0h]
  unsigned int v169; // [rsp+D8h] [rbp-D8h]
  __m128i v170; // [rsp+E0h] [rbp-D0h] BYREF
  __int128 v171; // [rsp+F0h] [rbp-C0h]
  char v172; // [rsp+100h] [rbp-B0h] BYREF

  v139 = (_BYTE *)(a2 + 48);
  v140 = 0;
  v157 = *(_QWORD *)(a2 + 56);
  if ( a2 + 48 != v157 )
  {
    do
    {
      if ( !v157 )
        BUG();
      v7 = v157;
      if ( (*(_BYTE *)v157 & 4) == 0 && (*(_BYTE *)(v157 + 44) & 8) != 0 )
      {
        do
          v7 = *(_QWORD *)(v7 + 8);
        while ( (*(_BYTE *)(v7 + 44) & 8) != 0 );
      }
      v153 = *(_BYTE **)(v7 + 8);
      if ( *(_WORD *)(v157 + 68) && *(_WORD *)(v157 + 68) != 68 )
        return v140;
      v141 = *((_BYTE *)a1 + 16);
      if ( !v141 )
        goto LABEL_12;
      if ( !dword_50226A8 )
        goto LABEL_11;
      v24 = *(_DWORD *)(v157 + 40);
      v162 = 0;
      v163 = 0;
      v164 = 0;
      v165 = 0;
      if ( (v24 & 0xFFFFFF) == 1 )
      {
        v106 = 0;
        v107 = 0;
LABEL_194:
        sub_C7D6A0((__int64)v106, v107, 8);
        goto LABEL_11;
      }
      v25 = 1;
      v143 = 0;
      v142 = 0;
      v154 = 0;
      v26 = v138;
      do
      {
        v27 = sub_2EBEE10(*a1, *(_DWORD *)(*(_QWORD *)(v157 + 32) + 40LL * v25 + 8));
        v28 = (_QWORD *)v27;
        if ( *(_WORD *)(v27 + 68) == 20 )
        {
          do
            v27 = sub_2EBEE10(*a1, *(_DWORD *)(*(_QWORD *)(v27 + 32) + 48LL));
          while ( *(_WORD *)(v27 + 68) == 20 );
          v28 = (_QWORD *)v27;
        }
        if ( (*(_BYTE *)(v28[2] + 25LL) & 0x20) != 0 )
        {
          v29 = v28[4];
          v30 = *(_BYTE *)(v29 + 40);
          if ( dword_50226A8 != 1 )
          {
            switch ( v30 )
            {
              case 3:
LABEL_86:
                v142 = 1;
                v26 = *(_QWORD *)(v29 + 64);
                break;
              case 2:
                v142 = 2;
                v26 = *(_QWORD *)(v29 + 64);
                break;
              case 1:
                v142 = 3;
                v26 = *(_QWORD *)(v29 + 64);
                break;
            }
            if ( v165 )
            {
              v31 = v165 - 1;
              v32 = ((0xBF58476D1CE4E5B9LL * v26) >> 31) ^ (484763065 * v26);
              v33 = (v165 - 1) & v32;
              v34 = &v163[7 * v33];
              v35 = *v34;
              if ( *v34 == v26 )
              {
                v36 = v154;
                if ( v34 != &v163[7 * v165] )
                  v36 = v141;
                v154 = v36;
                v37 = v143;
                if ( v34 != &v163[7 * v165] )
                  v37 = v28;
                v143 = v37;
LABEL_56:
                v38 = v34 + 1;
LABEL_57:
                v170.m128i_i64[0] = v28[3];
                sub_2F26740(v38, v170.m128i_i64);
                goto LABEL_58;
              }
              v150 = (v165 - 1) & v32;
              v94 = *v34;
              v95 = 1;
              while ( v94 != -1 )
              {
                v126 = v95 + 1;
                v127 = v31 & (v150 + v95);
                v146 = v126;
                v150 = v127;
                v128 = &v163[7 * v127];
                v94 = *v128;
                if ( *v128 == v26 )
                {
                  v129 = v128 == &v163[7 * v165];
                  v130 = v154;
                  if ( !v129 )
                    v130 = v141;
                  v154 = v130;
                  v131 = v143;
                  if ( !v129 )
                    v131 = v28;
                  v143 = v131;
                  v97 = &v163[7 * (v31 & v32)];
                  goto LABEL_157;
                }
                v95 = v146;
              }
              v32 = (484763065 * v26) ^ ((0xBF58476D1CE4E5B9LL * v26) >> 31);
              v33 = v31 & v32;
              v96 = v31 & v32;
              v97 = &v163[7 * v96];
              v35 = *v97;
              if ( *v97 == v26 )
              {
                v34 = &v163[7 * v96];
                goto LABEL_56;
              }
LABEL_157:
              v151 = 1;
              v54 = 0;
              while ( v35 != -1 )
              {
                if ( v35 != -2 || v54 )
                  v97 = v54;
                v33 = v31 & (v151 + v33);
                v34 = &v163[7 * v33];
                v35 = *v34;
                if ( *v34 == v26 )
                  goto LABEL_56;
                ++v151;
                v54 = v97;
                v97 = &v163[7 * v33];
              }
              if ( !v54 )
                v54 = v97;
              ++v162;
              v56 = v164 + 1;
              if ( 4 * ((int)v164 + 1) < 3 * v165 )
              {
                if ( v165 - HIDWORD(v164) - v56 <= v165 >> 3 )
                {
                  v152 = v32;
                  sub_2F26870((__int64)&v162, v165);
                  if ( !v165 )
                  {
LABEL_274:
                    LODWORD(v164) = v164 + 1;
                    BUG();
                  }
                  v98 = 1;
                  v99 = (v165 - 1) & v152;
                  v100 = 0;
                  v54 = &v163[7 * v99];
                  v101 = *v54;
                  v56 = v164 + 1;
                  if ( *v54 != v26 )
                  {
                    while ( v101 != -1 )
                    {
                      if ( !v100 && v101 == -2 )
                        v100 = v54;
                      v99 = (v165 - 1) & (v98 + v99);
                      v54 = &v163[7 * v99];
                      v101 = *v54;
                      if ( *v54 == v26 )
                        goto LABEL_91;
                      ++v98;
                    }
                    if ( v100 )
                      v54 = v100;
                  }
                }
LABEL_91:
                LODWORD(v164) = v56;
                if ( *v54 != -1 )
                  --HIDWORD(v164);
                *v54 = v26;
                v38 = v54 + 1;
                *((_DWORD *)v54 + 4) = 0;
                v54[3] = 0;
                v54[4] = v54 + 2;
                v54[5] = v54 + 2;
                v54[6] = 0;
                goto LABEL_57;
              }
            }
            else
            {
              ++v162;
            }
            sub_2F26870((__int64)&v162, 2 * v165);
            if ( !v165 )
              goto LABEL_274;
            v53 = (v165 - 1) & (((0xBF58476D1CE4E5B9LL * v26) >> 31) ^ (484763065 * v26));
            v54 = &v163[7 * v53];
            v55 = *v54;
            v56 = v164 + 1;
            if ( *v54 != v26 )
            {
              v132 = 1;
              v133 = 0;
              while ( v55 != -1 )
              {
                if ( !v133 && v55 == -2 )
                  v133 = v54;
                v53 = (v165 - 1) & (v132 + v53);
                v54 = &v163[7 * v53];
                v55 = *v54;
                if ( *v54 == v26 )
                  goto LABEL_91;
                ++v132;
              }
              if ( v133 )
                v54 = v133;
            }
            goto LABEL_91;
          }
          if ( v30 == 3 )
            goto LABEL_86;
        }
LABEL_58:
        v25 += 2;
        v39 = *(_DWORD *)(v157 + 40) & 0xFFFFFF;
      }
      while ( v39 != v25 );
      v138 = v26;
      if ( !v154 )
      {
        v106 = v163;
        v107 = 56LL * v165;
        if ( v165 )
        {
          v108 = &v163[7 * v165];
          do
          {
            if ( *v106 <= 0xFFFFFFFFFFFFFFFDLL )
              sub_2F25D80(v106[3]);
            v106 += 7;
          }
          while ( v108 != v106 );
          v106 = v163;
          v107 = 56LL * v165;
        }
        goto LABEL_194;
      }
      v166 = 0;
      v167 = 0;
      v168 = 0;
      v169 = 0;
      if ( (_DWORD)v164 )
      {
        v58 = v163;
        v59 = &v163[7 * v165];
        v144 = v59;
        if ( v163 != v59 )
        {
          while ( 1 )
          {
            v60 = v58;
            if ( *v58 <= 0xFFFFFFFFFFFFFFFDLL )
              break;
            v58 += 7;
            if ( v59 == v58 )
              goto LABEL_61;
          }
          if ( v59 != v58 )
          {
            v61 = a1;
LABEL_106:
            if ( v60[6] == 1 )
              goto LABEL_131;
            v62 = v60[4];
            if ( v60 + 2 == (_QWORD *)v62 )
              goto LABEL_131;
            v148 = v60;
            v63 = v60 + 2;
            while ( 1 )
            {
              v64 = *(_QWORD *)(v62 + 32);
              if ( *(_DWORD *)(v64 + 72) == 1 )
              {
                v65 = v148[4];
                if ( v63 == (_QWORD *)v65 )
                {
                  v60 = v148;
                  goto LABEL_117;
                }
                while ( (unsigned __int8)sub_2E6D360(v61[3], **(_QWORD **)(v64 + 64), *(_QWORD *)(v65 + 32)) )
                {
                  v65 = sub_220EF30(v65);
                  if ( v63 == (_QWORD *)v65 )
                    goto LABEL_116;
                }
                if ( v63 == (_QWORD *)v65 )
                  break;
              }
              v62 = sub_220EF30(v62);
              if ( v63 == (_QWORD *)v62 )
              {
                v60 = v148;
                goto LABEL_131;
              }
            }
LABEL_116:
            v60 = v148;
            v64 = *(_QWORD *)(v62 + 32);
LABEL_117:
            v66 = **(_QWORD **)(v64 + 64);
            v149 = (__int64 *)sub_2E313E0(v66);
            v69 = sub_2EC06C0(
                    *v61,
                    *(_QWORD *)(*(_QWORD *)(*v61 + 56) + 16LL * (*(_DWORD *)(*(_QWORD *)(v157 + 32) + 8LL) & 0x7FFFFFFF))
                  & 0xFFFFFFFFFFFFFFF8LL,
                    byte_3F871B3,
                    0,
                    v67,
                    v68);
            if ( v169 )
            {
              v70 = 1;
              v71 = 0;
              v72 = (v169 - 1) & (((0xBF58476D1CE4E5B9LL * *v60) >> 31) ^ (484763065 * *(_DWORD *)v60));
              v73 = (_QWORD *)(v167 + 16LL * v72);
              v74 = *v73;
              if ( *v73 == *v60 )
                goto LABEL_119;
              while ( v74 != -1 )
              {
                if ( v74 != -2 || v71 )
                  v73 = v71;
                v72 = (v169 - 1) & (v70 + v72);
                v74 = *(_QWORD *)(v167 + 16LL * v72);
                if ( *v60 == v74 )
                {
                  v73 = (_QWORD *)(v167 + 16LL * v72);
LABEL_119:
                  v75 = (__int32 *)(v73 + 1);
LABEL_120:
                  *v75 = v69;
                  switch ( v142 )
                  {
                    case 1:
                      v155 = *v60;
                      v82 = *(_QWORD *)(v61[1] + 8) - 40LL * *((unsigned __int16 *)v143 + 34);
                      v83 = *(unsigned __int8 **)(v157 + 56);
                      v158 = v83;
                      if ( v83 )
                      {
                        v135 = v82;
                        sub_B96E90((__int64)&v158, (__int64)v83, 1);
                        v82 = v135;
                        v159 = v158;
                        if ( v158 )
                        {
                          sub_B976B0((__int64)&v158, v158, (__int64)&v159);
                          v82 = v135;
                          v158 = 0;
                        }
                      }
                      else
                      {
                        v159 = 0;
                      }
                      v160 = 0;
                      v161 = 0;
                      v84 = sub_2F26260(v66, v149, (__int64 *)&v159, v82, v69);
                      v170.m128i_i64[0] = 3;
                      v79 = (__int64)v84;
                      v81 = v85;
                      break;
                    case 2:
                      v155 = *v60;
                      v86 = *(_QWORD *)(v61[1] + 8) - 40LL * *((unsigned __int16 *)v143 + 34);
                      v87 = *(unsigned __int8 **)(v157 + 56);
                      v158 = v87;
                      if ( v87 )
                      {
                        v136 = v86;
                        sub_B96E90((__int64)&v158, (__int64)v87, 1);
                        v86 = v136;
                        v159 = v158;
                        if ( v158 )
                        {
                          sub_B976B0((__int64)&v158, v158, (__int64)&v159);
                          v86 = v136;
                          v158 = 0;
                        }
                      }
                      else
                      {
                        v159 = 0;
                      }
                      v160 = 0;
                      v161 = 0;
                      v88 = sub_2F26260(v66, v149, (__int64 *)&v159, v86, v69);
                      v170.m128i_i64[0] = 2;
                      v79 = (__int64)v88;
                      v81 = v89;
                      break;
                    case 3:
                      v155 = *v60;
                      v76 = *(_QWORD *)(v61[1] + 8) - 40LL * *((unsigned __int16 *)v143 + 34);
                      v77 = *(unsigned __int8 **)(v157 + 56);
                      v158 = v77;
                      if ( v77 )
                      {
                        v134 = v76;
                        sub_B96E90((__int64)&v158, (__int64)v77, 1);
                        v76 = v134;
                        v159 = v158;
                        if ( v158 )
                        {
                          sub_B976B0((__int64)&v158, v158, (__int64)&v159);
                          v76 = v134;
                          v158 = 0;
                        }
                      }
                      else
                      {
                        v159 = 0;
                      }
                      v160 = 0;
                      v161 = 0;
                      v78 = sub_2F26260(v66, v149, (__int64 *)&v159, v76, v69);
                      v170.m128i_i64[0] = 1;
                      v79 = (__int64)v78;
                      v81 = v80;
                      break;
                    default:
LABEL_131:
                      v60 += 7;
                      if ( v60 == v144 )
                        goto LABEL_134;
                      while ( *v60 > 0xFFFFFFFFFFFFFFFDLL )
                      {
                        v60 += 7;
                        if ( v144 == v60 )
                          goto LABEL_134;
                      }
                      if ( v144 == v60 )
                      {
LABEL_134:
                        a1 = v61;
                        v39 = *(_DWORD *)(v157 + 40) & 0xFFFFFF;
                        goto LABEL_61;
                      }
                      goto LABEL_106;
                  }
                  *(_QWORD *)&v171 = 0;
                  *((_QWORD *)&v171 + 1) = v155;
                  sub_2E8EAD0(v81, v79, &v170);
                  if ( v159 )
                    sub_B91220((__int64)&v159, (__int64)v159);
                  if ( v158 )
                    sub_B91220((__int64)&v158, (__int64)v158);
                  goto LABEL_131;
                }
                ++v70;
                v71 = v73;
                v73 = (_QWORD *)(v167 + 16LL * v72);
              }
              if ( !v71 )
                v71 = v73;
              ++v166;
              v117 = v168 + 1;
              if ( 4 * ((int)v168 + 1) < 3 * v169 )
              {
                if ( v169 - HIDWORD(v168) - v117 <= v169 >> 3 )
                {
                  sub_9E25D0((__int64)&v166, v169);
                  if ( !v169 )
                  {
LABEL_273:
                    LODWORD(v168) = v168 + 1;
                    BUG();
                  }
                  v122 = 0;
                  v123 = 1;
                  v117 = v168 + 1;
                  v124 = (v169 - 1) & (((0xBF58476D1CE4E5B9LL * *v60) >> 31) ^ (484763065 * *(_DWORD *)v60));
                  v71 = (_QWORD *)(v167 + 16LL * v124);
                  v125 = *v71;
                  if ( *v60 != *v71 )
                  {
                    while ( v125 != -1 )
                    {
                      if ( !v122 && v125 == -2 )
                        v122 = v71;
                      v124 = (v169 - 1) & (v123 + v124);
                      v71 = (_QWORD *)(v167 + 16LL * v124);
                      v125 = *v71;
                      if ( *v60 == *v71 )
                        goto LABEL_212;
                      ++v123;
                    }
                    goto LABEL_220;
                  }
                }
                goto LABEL_212;
              }
            }
            else
            {
              ++v166;
            }
            sub_9E25D0((__int64)&v166, 2 * v169);
            if ( !v169 )
              goto LABEL_273;
            v117 = v168 + 1;
            v119 = (v169 - 1) & (((0xBF58476D1CE4E5B9LL * *v60) >> 31) ^ (484763065 * *(_DWORD *)v60));
            v71 = (_QWORD *)(v167 + 16LL * v119);
            v120 = *v71;
            if ( *v60 != *v71 )
            {
              v121 = 1;
              v122 = 0;
              while ( v120 != -1 )
              {
                if ( !v122 && v120 == -2 )
                  v122 = v71;
                v119 = (v169 - 1) & (v121 + v119);
                v71 = (_QWORD *)(v167 + 16LL * v119);
                v120 = *v71;
                if ( *v60 == *v71 )
                  goto LABEL_212;
                ++v121;
              }
LABEL_220:
              if ( v122 )
                v71 = v122;
            }
LABEL_212:
            LODWORD(v168) = v117;
            if ( *v71 != -1 )
              --HIDWORD(v168);
            v118 = *v60;
            v75 = (__int32 *)(v71 + 1);
            *v75 = 0;
            *((_QWORD *)v75 - 1) = v118;
            goto LABEL_120;
          }
        }
      }
LABEL_61:
      v40 = 1;
      if ( v39 == 1 )
        goto LABEL_76;
      v41 = v137;
      while ( 2 )
      {
        v42 = 40LL * v40;
        for ( i = sub_2EBEE10(*a1, *(_DWORD *)(*(_QWORD *)(v157 + 32) + v42 + 8));
              *(_WORD *)(i + 68) == 20;
              i = sub_2EBEE10(*a1, *(_DWORD *)(*(_QWORD *)(i + 32) + 48LL)) )
        {
          ;
        }
        if ( (*(_BYTE *)(*(_QWORD *)(i + 16) + 25LL) & 0x20) != 0 )
        {
          v44 = *(_QWORD *)(i + 32);
          v45 = *(_BYTE *)(v44 + 40);
          if ( dword_50226A8 != 1 )
          {
            if ( v45 != 3 && v45 != 2 && v45 != 1 )
              goto LABEL_70;
            goto LABEL_84;
          }
          if ( v45 == 3 )
          {
LABEL_84:
            v41 = *(_QWORD *)(v44 + 64);
LABEL_70:
            if ( v169 )
            {
              v46 = (v169 - 1) & (((0xBF58476D1CE4E5B9LL * v41) >> 31) ^ (484763065 * v41));
              v47 = (__int64 *)(v167 + 16LL * v46);
              v48 = *v47;
              if ( *v47 == v41 )
              {
LABEL_72:
                if ( v47 != (__int64 *)(v167 + 16LL * v169) )
                  sub_2EAB0C0(*(_QWORD *)(v157 + 32) + v42, *((_DWORD *)v47 + 2));
              }
              else
              {
                v104 = 1;
                while ( v48 != -1 )
                {
                  v105 = v104 + 1;
                  v46 = (v169 - 1) & (v104 + v46);
                  v47 = (__int64 *)(v167 + 16LL * v46);
                  v48 = *v47;
                  if ( *v47 == v41 )
                    goto LABEL_72;
                  v104 = v105;
                }
              }
            }
          }
        }
        v40 += 2;
        if ( (*(_DWORD *)(v157 + 40) & 0xFFFFFF) != v40 )
          continue;
        break;
      }
      v137 = v41;
LABEL_76:
      sub_C7D6A0(v167, 16LL * v169, 8);
      v49 = v165;
      if ( v165 )
      {
        v50 = v163;
        v51 = &v163[7 * v165];
        do
        {
          while ( *v50 > 0xFFFFFFFFFFFFFFFDLL )
          {
            v50 += 7;
            if ( v51 == v50 )
              goto LABEL_81;
          }
          v52 = v50[3];
          v50 += 7;
          sub_2F25D80(v52);
        }
        while ( v51 != v50 );
LABEL_81:
        v49 = v165;
      }
      sub_C7D6A0((__int64)v163, 56LL * v49, 8);
LABEL_11:
      if ( (*(_DWORD *)(v157 + 40) & 0xFFFFFFu) <= 2 )
        goto LABEL_12;
      v16 = *(_DWORD *)(*(_QWORD *)(v157 + 32) + 48LL);
      v17 = sub_2EBEE10(*a1, v16);
      v18 = v17;
      if ( *(_WORD *)(v17 + 68) == 20 )
      {
        do
          v17 = sub_2EBEE10(*a1, *(_DWORD *)(*(_QWORD *)(v17 + 32) + 48LL));
        while ( *(_WORD *)(v17 + 68) == 20 );
        v18 = v17;
      }
      if ( (*(_BYTE *)(*(_QWORD *)(v18 + 16) + 25LL) & 0x20) == 0 || *(_BYTE *)(*(_QWORD *)(v18 + 32) + 40LL) != 1 )
      {
LABEL_12:
        LODWORD(v166) = 0;
        v170.m128i_i64[0] = 0;
        v170.m128i_i64[1] = (__int64)&v172;
        *(_QWORD *)&v171 = 16;
        DWORD2(v171) = 0;
        BYTE12(v171) = 1;
        v10 = sub_2F25F50(a1, v157, (__int64 *)&v166, (__int64)&v170, a5, a6);
        if ( v10 && (_DWORD)v166 )
        {
          v57 = *(_DWORD *)(*(_QWORD *)(v157 + 32) + 8LL);
          if ( sub_2EBE590(
                 *a1,
                 (int)v166,
                 *(_QWORD *)(*(_QWORD *)(*a1 + 56) + 16LL * (v57 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                 0) )
          {
            sub_2EBECB0((_QWORD *)*a1, v57, (unsigned int)v166);
            sub_2E88E20(v157);
            sub_2EBF120(*a1, (int)v166);
            v140 = v10;
          }
          if ( !BYTE12(v171) )
            _libc_free(v170.m128i_u64[1]);
          goto LABEL_98;
        }
        ++v170.m128i_i64[0];
        if ( BYTE12(v171) )
        {
LABEL_19:
          *(_QWORD *)((char *)&v171 + 4) = 0;
        }
        else
        {
          v13 = 4 * (DWORD1(v171) - DWORD2(v171));
          if ( v13 < 0x20 )
            v13 = 32;
          if ( (unsigned int)v171 <= v13 )
          {
            memset((void *)v170.m128i_i64[1], -1, 8LL * (unsigned int)v171);
            goto LABEL_19;
          }
          sub_C8C990((__int64)&v170, v157);
        }
        v14 = sub_2F26100(a1, v157, (__int64)&v170, v9, v11, v12);
        if ( v14 )
        {
          v15 = BYTE12(v171);
          v90 = (_BYTE **)v170.m128i_i64[1];
          if ( BYTE12(v171) )
            v91 = v170.m128i_i64[1] + 8LL * DWORD1(v171);
          else
            v91 = v170.m128i_i64[1] + 8LL * (unsigned int)v171;
          if ( v170.m128i_i64[1] != v91 )
          {
            while ( 1 )
            {
              v92 = *v90;
              v93 = v90;
              if ( (unsigned __int64)*v90 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( (_BYTE **)v91 == ++v90 )
                goto LABEL_151;
            }
            if ( v90 != (_BYTE **)v91 )
            {
              v102 = v153;
              if ( v92 == v153 )
                goto LABEL_177;
              while ( 1 )
              {
                sub_2E88E20((__int64)v92);
                v103 = v93 + 1;
                if ( v93 + 1 == (_BYTE **)v91 )
                  break;
                while ( 1 )
                {
                  v92 = *v103;
                  v93 = v103;
                  if ( (unsigned __int64)*v103 < 0xFFFFFFFFFFFFFFFELL )
                    break;
                  if ( (_BYTE **)v91 == ++v103 )
                    goto LABEL_174;
                }
                if ( v103 == (_BYTE **)v91 )
                  break;
                if ( v92 == v102 )
                {
LABEL_177:
                  if ( !v92 )
                    BUG();
                  if ( (*v92 & 4) == 0 && (v92[44] & 8) != 0 )
                  {
                    do
                      v102 = (_BYTE *)*((_QWORD *)v102 + 1);
                    while ( (v102[44] & 8) != 0 );
                  }
                  v102 = (_BYTE *)*((_QWORD *)v102 + 1);
                }
              }
LABEL_174:
              v153 = v102;
              v15 = BYTE12(v171);
            }
          }
LABEL_151:
          v140 = v14;
          v157 = (__int64)v153;
        }
        else
        {
          v15 = BYTE12(v171);
          v157 = (__int64)v153;
        }
        if ( !v15 )
          _libc_free(v170.m128i_u64[1]);
        continue;
      }
      v19 = *(_QWORD *)(v157 + 32);
      v147 = *(_DWORD *)(v19 + 8);
      v20 = *(_DWORD *)(v157 + 40) & 0xFFFFFF;
      if ( v20 > 3 )
      {
        v21 = 120;
        v22 = 80LL * ((v20 - 4) >> 1) + 200;
        while ( 1 )
        {
          for ( j = sub_2EBEE10(*a1, *(_DWORD *)(v19 + v21 + 8));
                *(_WORD *)(j + 68) == 20;
                j = sub_2EBEE10(*a1, *(_DWORD *)(*(_QWORD *)(j + 32) + 48LL)) )
          {
            ;
          }
          if ( !sub_2E88AF0(j, v18, 2u) )
            goto LABEL_12;
          v21 += 80;
          if ( v22 == v21 )
            break;
          v19 = *(_QWORD *)(v157 + 32);
        }
      }
      v109 = sub_2EC06C0(
               *a1,
               *(_QWORD *)(*(_QWORD *)(*a1 + 56) + 16LL * (v16 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
               byte_3F871B3,
               0,
               a5,
               a6);
      v110 = *(_QWORD *)(v157 + 24);
      v111 = (__int64 *)sub_2E311E0(v110);
      v112 = *(_QWORD *)(*(_QWORD *)(v18 + 32) + 64LL);
      v113 = *(_QWORD *)(a1[1] + 8) - 40LL * *(unsigned __int16 *)(v18 + 68);
      v114 = *(unsigned __int8 **)(v157 + 56);
      v162 = v114;
      if ( v114 )
      {
        v145 = v111;
        v156 = v113;
        sub_B96E90((__int64)&v162, (__int64)v114, 1);
        v113 = v156;
        v111 = v145;
        v166 = v162;
        if ( v162 )
        {
          sub_B976B0((__int64)&v162, v162, (__int64)&v166);
          v111 = v145;
          v162 = 0;
          v113 = v156;
        }
      }
      else
      {
        v166 = 0;
      }
      v167 = 0;
      v168 = 0;
      v115 = sub_2F26260(v110, v111, (__int64 *)&v166, v113, v109);
      *((_QWORD *)&v171 + 1) = v112;
      v170.m128i_i64[0] = 1;
      *(_QWORD *)&v171 = 0;
      sub_2E8EAD0(v116, (__int64)v115, &v170);
      if ( v166 )
        sub_B91220((__int64)&v166, (__int64)v166);
      if ( v162 )
        sub_B91220((__int64)&v162, (__int64)v162);
      sub_2EBED50(*a1, v147, v109);
      sub_2E88E20(v157);
      v140 = v141;
LABEL_98:
      v157 = (__int64)v153;
    }
    while ( v139 != v153 );
  }
  return v140;
}
