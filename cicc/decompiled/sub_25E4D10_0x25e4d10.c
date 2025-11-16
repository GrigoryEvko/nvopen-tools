// Function: sub_25E4D10
// Address: 0x25e4d10
//
__int64 __fastcall sub_25E4D10(__int64 *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r14
  __int64 *v6; // rbx
  __int64 v7; // rsi
  bool v8; // zf
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // r10
  char *v14; // r11
  unsigned int v15; // r15d
  unsigned int v16; // esi
  __int64 v17; // rcx
  int v18; // r12d
  __int64 *v19; // r8
  unsigned int v20; // edi
  __int64 *v21; // rdx
  __int64 v22; // r9
  unsigned __int64 v23; // r13
  int v24; // r14d
  __int64 *v25; // rdx
  unsigned int v26; // r8d
  __int64 *v27; // rax
  __int64 v28; // rdi
  unsigned __int64 v29; // rax
  __int64 v30; // rbx
  unsigned int v31; // edx
  unsigned __int64 v32; // rcx
  int v33; // r9d
  __int64 *v34; // rsi
  unsigned int v35; // r8d
  __int64 *v36; // rax
  __int64 v37; // rdi
  unsigned __int64 v38; // r15
  __int64 v39; // r9
  unsigned __int64 *v40; // rsi
  unsigned int v41; // r13d
  unsigned __int64 v42; // r8
  unsigned __int64 *v43; // rax
  _QWORD *v44; // rdi
  _QWORD *v45; // r12
  __int64 v46; // r14
  __int64 v47; // r13
  unsigned __int64 v48; // rcx
  unsigned __int64 v49; // rax
  _QWORD *v50; // rax
  __int64 v51; // rdx
  __int64 v52; // r15
  __int64 *v53; // rdi
  _QWORD *j; // rdx
  __int64 *v55; // rax
  __int64 v56; // r9
  int v57; // esi
  int v58; // esi
  __int64 v59; // r10
  unsigned int v60; // ecx
  _QWORD *v61; // rdx
  __int64 v62; // r8
  _QWORD *v63; // rcx
  int v64; // edx
  int v65; // edi
  int v66; // edx
  unsigned int v67; // eax
  __int64 v68; // r8
  unsigned __int64 v69; // rcx
  unsigned __int64 v70; // rax
  _QWORD *v71; // rax
  __int64 v72; // rdx
  __int64 v73; // r13
  __int64 *v74; // rdi
  _QWORD *k; // rdx
  __int64 *v76; // rax
  __int64 v77; // r9
  int v78; // esi
  int v79; // esi
  __int64 v80; // r10
  unsigned int v81; // ecx
  _QWORD *v82; // rdx
  __int64 v83; // r8
  _QWORD *v84; // rdi
  int v85; // edx
  int v86; // eax
  int v87; // ecx
  unsigned int v88; // edx
  __int64 v89; // rax
  int v90; // eax
  __int64 v91; // rax
  _QWORD *v92; // rax
  __int64 v93; // rdx
  _QWORD *m; // rdx
  __int64 *v95; // rax
  __int64 v96; // r9
  int v97; // esi
  int v98; // esi
  __int64 v99; // r8
  unsigned int v100; // ecx
  _QWORD *v101; // rdx
  __int64 v102; // rdi
  _QWORD *v103; // rcx
  int v104; // edx
  int v105; // edx
  unsigned int v106; // r13d
  unsigned __int64 v107; // rdi
  int v108; // eax
  __int64 v109; // rax
  _QWORD *v110; // rax
  __int64 v111; // rdx
  _QWORD *i; // rdx
  __int64 *v113; // rax
  __int64 v114; // r9
  int v115; // esi
  int v116; // esi
  __int64 v117; // r8
  unsigned int v118; // ecx
  _QWORD *v119; // rdx
  __int64 v120; // rdi
  _QWORD *v121; // rcx
  int v122; // edx
  int v123; // edx
  int v124; // r9d
  __int64 *v125; // r8
  unsigned int v126; // r15d
  __int64 v127; // rax
  int v128; // esi
  int v129; // ecx
  int v130; // esi
  int v131; // esi
  __int64 v132; // r8
  __int64 v133; // rcx
  int v134; // eax
  __int64 v135; // rdi
  int v136; // eax
  int v137; // ecx
  int v138; // ecx
  __int64 v139; // rdi
  __int64 *v140; // r9
  __int64 v141; // r12
  int v142; // r14d
  __int64 v143; // rsi
  int v144; // ecx
  int v145; // ecx
  __int64 v146; // rdi
  unsigned int v147; // edx
  __int64 v148; // rsi
  int v149; // esi
  __int64 v150; // rcx
  __int64 v151; // rdi
  __int64 v152; // rcx
  __int64 v153; // rcx
  int v154; // esi
  __int64 v155; // rbx
  __int64 v156; // r12
  __int64 *v157; // rbx
  __int64 v158; // rcx
  int v159; // r14d
  __int64 *v160; // r9
  int v161; // r13d
  char *v162; // r14
  __int64 *v163; // rbx
  __int64 v164; // rax
  __int64 *v165; // rax
  __int64 v166; // rax
  int v167; // r14d
  __int64 *v168; // r12
  int v169; // r12d
  __int64 v170; // r9
  unsigned int v171; // r10d
  int v172; // [rsp+4h] [rbp-9Ch]
  int v173; // [rsp+4h] [rbp-9Ch]
  char *v174; // [rsp+8h] [rbp-98h]
  __int64 v175; // [rsp+20h] [rbp-80h]
  char *v177; // [rsp+30h] [rbp-70h]
  char *v178; // [rsp+38h] [rbp-68h]
  char *v179; // [rsp+38h] [rbp-68h]
  unsigned int v180; // [rsp+38h] [rbp-68h]
  char *v181; // [rsp+38h] [rbp-68h]
  unsigned int v182; // [rsp+38h] [rbp-68h]
  char *v183; // [rsp+38h] [rbp-68h]
  char *v184; // [rsp+38h] [rbp-68h]
  char *v185; // [rsp+38h] [rbp-68h]
  char *v186; // [rsp+38h] [rbp-68h]
  char *v187; // [rsp+38h] [rbp-68h]
  int v188; // [rsp+38h] [rbp-68h]
  int v189; // [rsp+38h] [rbp-68h]
  _QWORD *v190; // [rsp+38h] [rbp-68h]
  _QWORD *v191; // [rsp+38h] [rbp-68h]
  unsigned int v192; // [rsp+40h] [rbp-60h]
  char *v193; // [rsp+40h] [rbp-60h]
  unsigned int v194; // [rsp+40h] [rbp-60h]
  char *v195; // [rsp+40h] [rbp-60h]
  char *v196; // [rsp+40h] [rbp-60h]
  __int64 v197; // [rsp+40h] [rbp-60h]
  char *v198; // [rsp+40h] [rbp-60h]
  __int64 v199; // [rsp+40h] [rbp-60h]
  __int64 v200; // [rsp+40h] [rbp-60h]
  __int64 v201; // [rsp+40h] [rbp-60h]
  __int64 v202; // [rsp+40h] [rbp-60h]
  __int64 v203; // [rsp+40h] [rbp-60h]
  _QWORD *v204; // [rsp+40h] [rbp-60h]
  _QWORD *v205; // [rsp+40h] [rbp-60h]
  __int64 *v206; // [rsp+48h] [rbp-58h]
  char *v207; // [rsp+50h] [rbp-50h]
  __int64 v209; // [rsp+60h] [rbp-40h] BYREF
  __int64 v210[7]; // [rsp+68h] [rbp-38h] BYREF

  result = a2 - (char *)a1;
  v177 = a2;
  v175 = a3;
  if ( a2 - (char *)a1 <= 128 )
    return result;
  v5 = a4;
  if ( !a3 )
  {
    v206 = (__int64 *)a2;
    goto LABEL_217;
  }
  v174 = (char *)(a1 + 1);
  while ( 2 )
  {
    --v175;
    v6 = &a1[result >> 4];
    v7 = a1[1];
    v210[0] = a4;
    v8 = !sub_25E4BC0(v210, v7, *v6);
    v9 = *((_QWORD *)v177 - 1);
    if ( v8 )
    {
      if ( sub_25E4BC0(v210, a1[1], v9) )
      {
        v12 = *a1;
        v11 = a1[1];
        a1[1] = *a1;
        *a1 = v11;
        goto LABEL_7;
      }
      v162 = v177;
      if ( !sub_25E4BC0(v210, *v6, *((_QWORD *)v177 - 1)) )
      {
        v166 = *a1;
        *a1 = *v6;
        *v6 = v166;
        v11 = *a1;
        v12 = a1[1];
        goto LABEL_7;
      }
      v163 = a1;
LABEL_233:
      v164 = *v163;
      *v163 = *((_QWORD *)v162 - 1);
      *((_QWORD *)v162 - 1) = v164;
      v165 = v163;
      v11 = *v163;
      v12 = v165[1];
      goto LABEL_7;
    }
    if ( !sub_25E4BC0(v210, *v6, v9) )
    {
      v162 = v177;
      v163 = a1;
      if ( !sub_25E4BC0(v210, a1[1], *((_QWORD *)v177 - 1)) )
      {
        v12 = *a1;
        v11 = a1[1];
        a1[1] = *a1;
        *a1 = v11;
        goto LABEL_7;
      }
      goto LABEL_233;
    }
    v10 = *a1;
    *a1 = *v6;
    *v6 = v10;
    v11 = *a1;
    v12 = a1[1];
LABEL_7:
    v13 = a4;
    v207 = v174;
    v14 = v177;
    v15 = *(_DWORD *)(a4 + 24);
    while ( 2 )
    {
      v209 = v12;
      v206 = (__int64 *)v207;
      if ( !v15 )
      {
        ++*(_QWORD *)v13;
        v210[0] = 0;
LABEL_170:
        v186 = v14;
        v202 = v13;
        sub_9DDA50(v13, 2 * v15);
        v13 = v202;
        v14 = v186;
        v144 = *(_DWORD *)(v202 + 24);
        if ( v144 )
        {
          v12 = v209;
          v145 = v144 - 1;
          v146 = *(_QWORD *)(v202 + 8);
          v147 = v145 & (((unsigned int)v209 >> 9) ^ ((unsigned int)v209 >> 4));
          v19 = (__int64 *)(v146 + 16LL * v147);
          v148 = *v19;
          if ( v209 == *v19 )
          {
LABEL_172:
            v149 = *(_DWORD *)(v202 + 16);
            v210[0] = (__int64)v19;
            v129 = v149 + 1;
          }
          else
          {
            v169 = 1;
            v170 = 0;
            while ( v148 != -4096 )
            {
              if ( !v170 && v148 == -8192 )
                v170 = (__int64)v19;
              v147 = v145 & (v169 + v147);
              v19 = (__int64 *)(v146 + 16LL * v147);
              v148 = *v19;
              if ( v209 == *v19 )
                goto LABEL_172;
              ++v169;
            }
            if ( !v170 )
              v170 = (__int64)v19;
            v129 = *(_DWORD *)(v202 + 16) + 1;
            v210[0] = v170;
            v19 = (__int64 *)v170;
          }
        }
        else
        {
          v154 = *(_DWORD *)(v202 + 16);
          v210[0] = 0;
          v19 = 0;
          v12 = v209;
          v129 = v154 + 1;
        }
        goto LABEL_139;
      }
      v16 = v15 - 1;
      v17 = *(_QWORD *)(v13 + 8);
      v18 = 1;
      v19 = 0;
      v20 = (v15 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v21 = (__int64 *)(v17 + 16LL * v20);
      v22 = *v21;
      if ( v12 == *v21 )
      {
LABEL_10:
        v23 = v21[1];
        goto LABEL_11;
      }
      while ( v22 != -4096 )
      {
        if ( v22 == -8192 && !v19 )
          v19 = v21;
        v20 = v16 & (v18 + v20);
        v21 = (__int64 *)(v17 + 16LL * v20);
        v22 = *v21;
        if ( *v21 == v12 )
          goto LABEL_10;
        ++v18;
      }
      v128 = *(_DWORD *)(v13 + 16);
      if ( !v19 )
        v19 = v21;
      ++*(_QWORD *)v13;
      v129 = v128 + 1;
      v210[0] = (__int64)v19;
      if ( 4 * (v128 + 1) >= 3 * v15 )
        goto LABEL_170;
      if ( v15 - *(_DWORD *)(v13 + 20) - v129 <= v15 >> 3 )
      {
        v187 = v14;
        v203 = v13;
        sub_9DDA50(v13, v15);
        sub_25E0C90(v203, &v209, v210);
        v13 = v203;
        v12 = v209;
        v19 = (__int64 *)v210[0];
        v14 = v187;
        v129 = *(_DWORD *)(v203 + 16) + 1;
      }
LABEL_139:
      *(_DWORD *)(v13 + 16) = v129;
      if ( *v19 != -4096 )
        --*(_DWORD *)(v13 + 20);
      *v19 = v12;
      v19[1] = 0;
      v15 = *(_DWORD *)(v13 + 24);
      if ( !v15 )
      {
        ++*(_QWORD *)v13;
        v23 = 0;
        goto LABEL_143;
      }
      v17 = *(_QWORD *)(v13 + 8);
      v16 = v15 - 1;
      v23 = 0;
LABEL_11:
      v24 = 1;
      v25 = 0;
      v26 = v16 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v27 = (__int64 *)(v17 + 16LL * v26);
      v28 = *v27;
      if ( v11 != *v27 )
      {
        while ( v28 != -4096 )
        {
          if ( !v25 && v28 == -8192 )
            v25 = v27;
          v26 = v16 & (v24 + v26);
          v27 = (__int64 *)(v17 + 16LL * v26);
          v28 = *v27;
          if ( *v27 == v11 )
            goto LABEL_12;
          ++v24;
        }
        if ( !v25 )
          v25 = v27;
        v136 = *(_DWORD *)(v13 + 16);
        ++*(_QWORD *)v13;
        v134 = v136 + 1;
        if ( 4 * v134 < 3 * v15 )
        {
          if ( v15 - (v134 + *(_DWORD *)(v13 + 20)) <= v15 >> 3 )
          {
            v201 = v13;
            v185 = v14;
            sub_9DDA50(v13, v15);
            v13 = v201;
            v137 = *(_DWORD *)(v201 + 24);
            if ( !v137 )
            {
LABEL_317:
              ++*(_DWORD *)(v13 + 16);
              BUG();
            }
            v138 = v137 - 1;
            v139 = *(_QWORD *)(v201 + 8);
            v140 = 0;
            LODWORD(v141) = v138 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
            v14 = v185;
            v142 = 1;
            v134 = *(_DWORD *)(v201 + 16) + 1;
            v25 = (__int64 *)(v139 + 16LL * (unsigned int)v141);
            v143 = *v25;
            if ( *v25 != v11 )
            {
              while ( v143 != -4096 )
              {
                if ( v143 == -8192 && !v140 )
                  v140 = v25;
                v141 = v138 & (unsigned int)(v141 + v142);
                v25 = (__int64 *)(v139 + 16 * v141);
                v143 = *v25;
                if ( *v25 == v11 )
                  goto LABEL_145;
                ++v142;
              }
              if ( v140 )
                v25 = v140;
            }
          }
          goto LABEL_145;
        }
LABEL_143:
        v200 = v13;
        v184 = v14;
        sub_9DDA50(v13, 2 * v15);
        v13 = v200;
        v130 = *(_DWORD *)(v200 + 24);
        if ( !v130 )
          goto LABEL_317;
        v131 = v130 - 1;
        v132 = *(_QWORD *)(v200 + 8);
        v14 = v184;
        LODWORD(v133) = v131 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v134 = *(_DWORD *)(v200 + 16) + 1;
        v25 = (__int64 *)(v132 + 16LL * (unsigned int)v133);
        v135 = *v25;
        if ( v11 != *v25 )
        {
          v167 = 1;
          v168 = 0;
          while ( v135 != -4096 )
          {
            if ( v135 == -8192 && !v168 )
              v168 = v25;
            v133 = v131 & (unsigned int)(v133 + v167);
            v25 = (__int64 *)(v132 + 16 * v133);
            v135 = *v25;
            if ( *v25 == v11 )
              goto LABEL_145;
            ++v167;
          }
          if ( v168 )
            v25 = v168;
        }
LABEL_145:
        *(_DWORD *)(v13 + 16) = v134;
        if ( *v25 != -4096 )
          --*(_DWORD *)(v13 + 20);
        *v25 = v11;
        v29 = 0;
        v25[1] = 0;
        v15 = *(_DWORD *)(v13 + 24);
        goto LABEL_13;
      }
LABEL_12:
      v29 = v27[1];
LABEL_13:
      if ( v23 > v29 )
        goto LABEL_65;
      v14 -= 8;
      v30 = v13;
      v31 = v15;
      while ( 1 )
      {
        v45 = *(_QWORD **)v14;
        v46 = *(_QWORD *)(v30 + 8);
        v47 = *a1;
        if ( v31 )
        {
          v32 = v31 - 1;
          v33 = 1;
          v34 = 0;
          v35 = v32 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
          v36 = (__int64 *)(v46 + 16LL * v35);
          v37 = *v36;
          if ( v47 == *v36 )
          {
LABEL_16:
            v38 = v36[1];
            goto LABEL_17;
          }
          while ( v37 != -4096 )
          {
            if ( v37 == -8192 && !v34 )
              v34 = v36;
            v35 = v32 & (v33 + v35);
            v36 = (__int64 *)(v46 + 16LL * v35);
            v37 = *v36;
            if ( v47 == *v36 )
              goto LABEL_16;
            ++v33;
          }
          if ( !v34 )
            v34 = v36;
          v108 = *(_DWORD *)(v30 + 16);
          ++*(_QWORD *)v30;
          v65 = v108 + 1;
          if ( 4 * (v108 + 1) < 3 * v31 )
          {
            if ( v31 - *(_DWORD *)(v30 + 20) - v65 <= v31 >> 3 )
            {
              v182 = v31;
              v198 = v14;
              v109 = (((((((((v32 | (v32 >> 1)) >> 2) | v32 | (v32 >> 1)) >> 4)
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
              if ( (unsigned int)v109 < 0x40 )
                LODWORD(v109) = 64;
              *(_DWORD *)(v30 + 24) = v109;
              v110 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v109, 8);
              v14 = v198;
              *(_QWORD *)(v30 + 8) = v110;
              if ( v46 )
              {
                v111 = *(unsigned int *)(v30 + 24);
                *(_QWORD *)(v30 + 16) = 0;
                v199 = 16LL * v182;
                for ( i = &v110[2 * v111]; i != v110; v110 += 2 )
                {
                  if ( v110 )
                    *v110 = -4096;
                }
                v113 = (__int64 *)v46;
                do
                {
                  v114 = *v113;
                  if ( *v113 != -4096 && v114 != -8192 )
                  {
                    v115 = *(_DWORD *)(v30 + 24);
                    if ( !v115 )
                    {
                      MEMORY[0] = *v113;
                      BUG();
                    }
                    v116 = v115 - 1;
                    v117 = *(_QWORD *)(v30 + 8);
                    v118 = v116 & (((unsigned int)v114 >> 9) ^ ((unsigned int)v114 >> 4));
                    v119 = (_QWORD *)(v117 + 16LL * v118);
                    v120 = *v119;
                    if ( v114 != *v119 )
                    {
                      v172 = 1;
                      v190 = 0;
                      while ( v120 != -4096 )
                      {
                        if ( !v190 )
                        {
                          if ( v120 != -8192 )
                            v119 = 0;
                          v190 = v119;
                        }
                        v118 = v116 & (v172 + v118);
                        v119 = (_QWORD *)(v117 + 16LL * v118);
                        v120 = *v119;
                        if ( v114 == *v119 )
                          goto LABEL_119;
                        ++v172;
                      }
                      if ( v190 )
                        v119 = v190;
                    }
LABEL_119:
                    *v119 = v114;
                    v119[1] = v113[1];
                    ++*(_DWORD *)(v30 + 16);
                  }
                  v113 += 2;
                }
                while ( (__int64 *)(v46 + v199) != v113 );
                v183 = v14;
                sub_C7D6A0(v46, v199, 8);
                v121 = *(_QWORD **)(v30 + 8);
                v122 = *(_DWORD *)(v30 + 24);
                v14 = v183;
                v65 = *(_DWORD *)(v30 + 16) + 1;
              }
              else
              {
                v153 = *(unsigned int *)(v30 + 24);
                *(_QWORD *)(v30 + 16) = 0;
                v122 = v153;
                v121 = &v110[2 * v153];
                if ( v110 == v121 )
                {
                  v65 = 1;
                }
                else
                {
                  do
                  {
                    if ( v110 )
                      *v110 = -4096;
                    v110 += 2;
                  }
                  while ( v121 != v110 );
                  v121 = *(_QWORD **)(v30 + 8);
                  v122 = *(_DWORD *)(v30 + 24);
                  v65 = *(_DWORD *)(v30 + 16) + 1;
                }
              }
              if ( !v122 )
              {
LABEL_314:
                ++*(_DWORD *)(v30 + 16);
                BUG();
              }
              v123 = v122 - 1;
              v124 = 1;
              v125 = 0;
              v126 = v123 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
              v34 = &v121[2 * v126];
              v127 = *v34;
              if ( v47 != *v34 )
              {
                while ( v127 != -4096 )
                {
                  if ( !v125 && v127 == -8192 )
                    v125 = v34;
                  v126 = v123 & (v124 + v126);
                  v34 = &v121[2 * v126];
                  v127 = *v34;
                  if ( v47 == *v34 )
                    goto LABEL_39;
                  ++v124;
                }
                if ( v125 )
                  v34 = v125;
              }
            }
            goto LABEL_39;
          }
        }
        else
        {
          ++*(_QWORD *)v30;
        }
        v178 = v14;
        v192 = v31;
        v48 = ((((((((2 * v31 - 1) | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 2)
                 | (2 * v31 - 1)
                 | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 4)
               | (((2 * v31 - 1) | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 2)
               | (2 * v31 - 1)
               | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 8)
             | (((((2 * v31 - 1) | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 2)
               | (2 * v31 - 1)
               | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 4)
             | (((2 * v31 - 1) | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 2)
             | (2 * v31 - 1)
             | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 16;
        v49 = (v48
             | (((((((2 * v31 - 1) | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 2)
                 | (2 * v31 - 1)
                 | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 4)
               | (((2 * v31 - 1) | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 2)
               | (2 * v31 - 1)
               | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 8)
             | (((((2 * v31 - 1) | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 2)
               | (2 * v31 - 1)
               | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 4)
             | (((2 * v31 - 1) | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 2)
             | (2 * v31 - 1)
             | ((unsigned __int64)(2 * v31 - 1) >> 1))
            + 1;
        if ( (unsigned int)v49 < 0x40 )
          LODWORD(v49) = 64;
        *(_DWORD *)(v30 + 24) = v49;
        v50 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v49, 8);
        v14 = v178;
        *(_QWORD *)(v30 + 8) = v50;
        if ( v46 )
        {
          v51 = *(unsigned int *)(v30 + 24);
          *(_QWORD *)(v30 + 16) = 0;
          v52 = 16LL * v192;
          v53 = (__int64 *)(v46 + v52);
          for ( j = &v50[2 * v51]; j != v50; v50 += 2 )
          {
            if ( v50 )
              *v50 = -4096;
          }
          v55 = (__int64 *)v46;
          if ( (__int64 *)v46 != v53 )
          {
            do
            {
              v56 = *v55;
              if ( *v55 != -8192 && v56 != -4096 )
              {
                v57 = *(_DWORD *)(v30 + 24);
                if ( !v57 )
                {
                  MEMORY[0] = *v55;
                  BUG();
                }
                v58 = v57 - 1;
                v59 = *(_QWORD *)(v30 + 8);
                v60 = v58 & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
                v61 = (_QWORD *)(v59 + 16LL * v60);
                v62 = *v61;
                if ( v56 != *v61 )
                {
                  v188 = 1;
                  v204 = 0;
                  while ( v62 != -4096 )
                  {
                    if ( v62 == -8192 )
                    {
                      if ( v204 )
                        v61 = v204;
                      v204 = v61;
                    }
                    v60 = v58 & (v188 + v60);
                    v61 = (_QWORD *)(v59 + 16LL * v60);
                    v62 = *v61;
                    if ( v56 == *v61 )
                      goto LABEL_34;
                    ++v188;
                  }
                  if ( v204 )
                    v61 = v204;
                }
LABEL_34:
                *v61 = v56;
                v61[1] = v55[1];
                ++*(_DWORD *)(v30 + 16);
              }
              v55 += 2;
            }
            while ( v53 != v55 );
          }
          v193 = v14;
          sub_C7D6A0(v46, v52, 8);
          v63 = *(_QWORD **)(v30 + 8);
          v64 = *(_DWORD *)(v30 + 24);
          v14 = v193;
          v65 = *(_DWORD *)(v30 + 16) + 1;
        }
        else
        {
          v150 = *(unsigned int *)(v30 + 24);
          *(_QWORD *)(v30 + 16) = 0;
          v64 = v150;
          v63 = &v50[2 * v150];
          if ( v50 == v63 )
          {
            v65 = 1;
          }
          else
          {
            do
            {
              if ( v50 )
                *v50 = -4096;
              v50 += 2;
            }
            while ( v63 != v50 );
            v63 = *(_QWORD **)(v30 + 8);
            v64 = *(_DWORD *)(v30 + 24);
            v65 = *(_DWORD *)(v30 + 16) + 1;
          }
        }
        if ( !v64 )
          goto LABEL_314;
        v66 = v64 - 1;
        v67 = v66 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
        v34 = &v63[2 * v67];
        v68 = *v34;
        if ( v47 != *v34 )
        {
          v159 = 1;
          v160 = 0;
          while ( v68 != -4096 )
          {
            if ( v68 != -8192 || v160 )
              v34 = v160;
            v67 = v66 & (v159 + v67);
            v68 = v63[2 * v67];
            if ( v47 == v68 )
            {
              v34 = &v63[2 * v67];
              goto LABEL_39;
            }
            ++v159;
            v160 = v34;
            v34 = &v63[2 * v67];
          }
          if ( v160 )
            v34 = v160;
        }
LABEL_39:
        *(_DWORD *)(v30 + 16) = v65;
        if ( *v34 != -4096 )
          --*(_DWORD *)(v30 + 20);
        *v34 = v47;
        v34[1] = 0;
        v31 = *(_DWORD *)(v30 + 24);
        v46 = *(_QWORD *)(v30 + 8);
        if ( !v31 )
        {
          ++*(_QWORD *)v30;
          v38 = 0;
LABEL_43:
          v179 = v14;
          v194 = v31;
          v69 = ((((((((2 * v31 - 1) | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 2)
                   | (2 * v31 - 1)
                   | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 4)
                 | (((2 * v31 - 1) | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 2)
                 | (2 * v31 - 1)
                 | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 8)
               | (((((2 * v31 - 1) | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 2)
                 | (2 * v31 - 1)
                 | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 4)
               | (((2 * v31 - 1) | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 2)
               | (2 * v31 - 1)
               | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 16;
          v70 = (v69
               | (((((((2 * v31 - 1) | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 2)
                   | (2 * v31 - 1)
                   | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 4)
                 | (((2 * v31 - 1) | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 2)
                 | (2 * v31 - 1)
                 | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 8)
               | (((((2 * v31 - 1) | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 2)
                 | (2 * v31 - 1)
                 | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 4)
               | (((2 * v31 - 1) | ((unsigned __int64)(2 * v31 - 1) >> 1)) >> 2)
               | (2 * v31 - 1)
               | ((unsigned __int64)(2 * v31 - 1) >> 1))
              + 1;
          if ( (unsigned int)v70 < 0x40 )
            LODWORD(v70) = 64;
          *(_DWORD *)(v30 + 24) = v70;
          v71 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v70, 8);
          v14 = v179;
          *(_QWORD *)(v30 + 8) = v71;
          if ( v46 )
          {
            v72 = *(unsigned int *)(v30 + 24);
            *(_QWORD *)(v30 + 16) = 0;
            v73 = 16LL * v194;
            v74 = (__int64 *)(v46 + v73);
            for ( k = &v71[2 * v72]; k != v71; v71 += 2 )
            {
              if ( v71 )
                *v71 = -4096;
            }
            v76 = (__int64 *)v46;
            if ( (__int64 *)v46 != v74 )
            {
              do
              {
                v77 = *v76;
                if ( *v76 != -8192 && v77 != -4096 )
                {
                  v78 = *(_DWORD *)(v30 + 24);
                  if ( !v78 )
                  {
                    MEMORY[0] = *v76;
                    BUG();
                  }
                  v79 = v78 - 1;
                  v80 = *(_QWORD *)(v30 + 8);
                  v81 = v79 & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
                  v82 = (_QWORD *)(v80 + 16LL * v81);
                  v83 = *v82;
                  if ( *v82 != v77 )
                  {
                    v189 = 1;
                    v205 = 0;
                    while ( v83 != -4096 )
                    {
                      if ( v83 == -8192 )
                      {
                        if ( v205 )
                          v82 = v205;
                        v205 = v82;
                      }
                      v81 = v79 & (v189 + v81);
                      v82 = (_QWORD *)(v80 + 16LL * v81);
                      v83 = *v82;
                      if ( v77 == *v82 )
                        goto LABEL_55;
                      ++v189;
                    }
                    if ( v205 )
                      v82 = v205;
                  }
LABEL_55:
                  *v82 = v77;
                  v82[1] = v76[1];
                  ++*(_DWORD *)(v30 + 16);
                }
                v76 += 2;
              }
              while ( v74 != v76 );
            }
            v195 = v14;
            sub_C7D6A0(v46, v73, 8);
            v84 = *(_QWORD **)(v30 + 8);
            v85 = *(_DWORD *)(v30 + 24);
            v14 = v195;
            v86 = *(_DWORD *)(v30 + 16) + 1;
          }
          else
          {
            v151 = *(unsigned int *)(v30 + 24);
            *(_QWORD *)(v30 + 16) = 0;
            v85 = v151;
            v84 = &v71[2 * v151];
            if ( v71 == v84 )
            {
              v86 = 1;
            }
            else
            {
              do
              {
                if ( v71 )
                  *v71 = -4096;
                v71 += 2;
              }
              while ( v84 != v71 );
              v84 = *(_QWORD **)(v30 + 8);
              v85 = *(_DWORD *)(v30 + 24);
              v86 = *(_DWORD *)(v30 + 16) + 1;
            }
          }
          if ( !v85 )
            goto LABEL_316;
          v87 = v85 - 1;
          v88 = (v85 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
          v40 = &v84[2 * v88];
          v42 = *v40;
          if ( v45 != (_QWORD *)*v40 )
          {
            v161 = 1;
            v39 = 0;
            while ( v42 != -4096 )
            {
              if ( v42 != -8192 || v39 )
                v40 = (unsigned __int64 *)v39;
              v39 = (unsigned int)(v161 + 1);
              v88 = v87 & (v161 + v88);
              v42 = v84[2 * v88];
              if ( v45 == (_QWORD *)v42 )
              {
                v40 = &v84[2 * v88];
                goto LABEL_60;
              }
              ++v161;
              v39 = (__int64)v40;
              v40 = &v84[2 * v88];
            }
            if ( v39 )
              v40 = (unsigned __int64 *)v39;
          }
          goto LABEL_60;
        }
        v32 = v31 - 1;
        v38 = 0;
LABEL_17:
        v39 = 1;
        v40 = 0;
        v41 = ((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4);
        v42 = v41 & (unsigned int)v32;
        v43 = (unsigned __int64 *)(v46 + 16 * v42);
        v44 = (_QWORD *)*v43;
        if ( v45 != (_QWORD *)*v43 )
          break;
LABEL_18:
        if ( v43[1] >= v38 )
          goto LABEL_63;
LABEL_19:
        v31 = *(_DWORD *)(v30 + 24);
        v14 -= 8;
      }
      while ( v44 != (_QWORD *)-4096LL )
      {
        if ( v44 == (_QWORD *)-8192LL && !v40 )
          v40 = v43;
        v42 = (unsigned int)v32 & ((_DWORD)v39 + (_DWORD)v42);
        v43 = (unsigned __int64 *)(v46 + 16LL * (unsigned int)v42);
        v44 = (_QWORD *)*v43;
        if ( v45 == (_QWORD *)*v43 )
          goto LABEL_18;
        v39 = (unsigned int)(v39 + 1);
      }
      if ( !v40 )
        v40 = v43;
      v90 = *(_DWORD *)(v30 + 16);
      ++*(_QWORD *)v30;
      v86 = v90 + 1;
      if ( 4 * v86 >= 3 * v31 )
        goto LABEL_43;
      v42 = v31 - (v86 + *(_DWORD *)(v30 + 20));
      if ( (unsigned int)v42 <= v31 >> 3 )
      {
        v180 = v31;
        v196 = v14;
        v91 = ((((((((v32 >> 1) | v32 | (((v32 >> 1) | v32) >> 2)) >> 4) | (v32 >> 1) | v32 | (((v32 >> 1) | v32) >> 2)) >> 8)
               | (((v32 >> 1) | v32 | (((v32 >> 1) | v32) >> 2)) >> 4)
               | (v32 >> 1)
               | v32
               | (((v32 >> 1) | v32) >> 2)) >> 16)
             | (((((v32 >> 1) | v32 | (((v32 >> 1) | v32) >> 2)) >> 4) | (v32 >> 1) | v32 | (((v32 >> 1) | v32) >> 2)) >> 8)
             | (((v32 >> 1) | v32 | (((v32 >> 1) | v32) >> 2)) >> 4)
             | (v32 >> 1)
             | v32
             | (((v32 >> 1) | v32) >> 2))
            + 1;
        if ( (unsigned int)v91 < 0x40 )
          LODWORD(v91) = 64;
        *(_DWORD *)(v30 + 24) = v91;
        v92 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v91, 8);
        v14 = v196;
        *(_QWORD *)(v30 + 8) = v92;
        if ( v46 )
        {
          v93 = *(unsigned int *)(v30 + 24);
          *(_QWORD *)(v30 + 16) = 0;
          v197 = 16LL * v180;
          for ( m = &v92[2 * v93]; m != v92; v92 += 2 )
          {
            if ( v92 )
              *v92 = -4096;
          }
          v95 = (__int64 *)v46;
          do
          {
            v96 = *v95;
            if ( *v95 != -8192 && v96 != -4096 )
            {
              v97 = *(_DWORD *)(v30 + 24);
              if ( !v97 )
              {
                MEMORY[0] = *v95;
                BUG();
              }
              v98 = v97 - 1;
              v99 = *(_QWORD *)(v30 + 8);
              v100 = v98 & (((unsigned int)v96 >> 9) ^ ((unsigned int)v96 >> 4));
              v101 = (_QWORD *)(v99 + 16LL * v100);
              v102 = *v101;
              if ( v96 != *v101 )
              {
                v173 = 1;
                v191 = 0;
                while ( v102 != -4096 )
                {
                  if ( !v191 )
                  {
                    if ( v102 != -8192 )
                      v101 = 0;
                    v191 = v101;
                  }
                  v100 = v98 & (v173 + v100);
                  v101 = (_QWORD *)(v99 + 16LL * v100);
                  v102 = *v101;
                  if ( v96 == *v101 )
                    goto LABEL_88;
                  ++v173;
                }
                if ( v191 )
                  v101 = v191;
              }
LABEL_88:
              *v101 = v96;
              v101[1] = v95[1];
              ++*(_DWORD *)(v30 + 16);
            }
            v95 += 2;
          }
          while ( (__int64 *)(v46 + v197) != v95 );
          v181 = v14;
          sub_C7D6A0(v46, v197, 8);
          v103 = *(_QWORD **)(v30 + 8);
          v104 = *(_DWORD *)(v30 + 24);
          v14 = v181;
          v86 = *(_DWORD *)(v30 + 16) + 1;
        }
        else
        {
          v152 = *(unsigned int *)(v30 + 24);
          *(_QWORD *)(v30 + 16) = 0;
          v104 = v152;
          v103 = &v92[2 * v152];
          if ( v92 == v103 )
          {
            v86 = 1;
          }
          else
          {
            do
            {
              if ( v92 )
                *v92 = -4096;
              v92 += 2;
            }
            while ( v103 != v92 );
            v103 = *(_QWORD **)(v30 + 8);
            v104 = *(_DWORD *)(v30 + 24);
            v86 = *(_DWORD *)(v30 + 16) + 1;
          }
        }
        if ( !v104 )
        {
LABEL_316:
          ++*(_DWORD *)(v30 + 16);
          BUG();
        }
        v105 = v104 - 1;
        v39 = 1;
        v42 = 0;
        v106 = v105 & v41;
        v40 = &v103[2 * v106];
        v107 = *v40;
        if ( v45 != (_QWORD *)*v40 )
        {
          while ( v107 != -4096 )
          {
            if ( v107 == -8192 && !v42 )
              v42 = (unsigned __int64)v40;
            v171 = v39 + 1;
            v39 = v105 & (v106 + (unsigned int)v39);
            v106 = v39;
            v40 = &v103[2 * (unsigned int)v39];
            v107 = *v40;
            if ( v45 == (_QWORD *)*v40 )
              goto LABEL_60;
            v39 = v171;
          }
          if ( v42 )
            v40 = (unsigned __int64 *)v42;
        }
      }
LABEL_60:
      *(_DWORD *)(v30 + 16) = v86;
      if ( *v40 != -4096 )
        --*(_DWORD *)(v30 + 20);
      *v40 = (unsigned __int64)v45;
      v40[1] = 0;
      if ( v38 )
        goto LABEL_19;
LABEL_63:
      v13 = v30;
      if ( v207 < v14 )
      {
        v89 = *(_QWORD *)v207;
        *(_QWORD *)v207 = *(_QWORD *)v14;
        *(_QWORD *)v14 = v89;
        v15 = *(_DWORD *)(v30 + 24);
LABEL_65:
        v11 = *a1;
        v12 = *((_QWORD *)v207 + 1);
        v207 += 8;
        continue;
      }
      break;
    }
    sub_25E4D10(v207, v177, v175, a4, v42, v39);
    result = v207 - (char *)a1;
    if ( v207 - (char *)a1 > 128 )
    {
      if ( v175 )
      {
        v177 = v207;
        continue;
      }
      v5 = a4;
LABEL_217:
      v155 = result >> 3;
      v156 = ((result >> 3) - 2) >> 1;
      sub_25E2930((__int64)a1, v156, result >> 3, a1[v156], v5);
      do
      {
        --v156;
        sub_25E2930((__int64)a1, v156, v155, a1[v156], v5);
      }
      while ( v156 );
      v157 = v206;
      do
      {
        v158 = *--v157;
        *v157 = *a1;
        result = sub_25E2930((__int64)a1, 0, v157 - a1, v158, v5);
      }
      while ( (char *)v157 - (char *)a1 > 8 );
    }
    return result;
  }
}
