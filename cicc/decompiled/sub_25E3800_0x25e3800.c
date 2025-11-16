// Function: sub_25E3800
// Address: 0x25e3800
//
__int64 __fastcall sub_25E3800(__int64 *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v6; // rbx
  bool v7; // zf
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  char *v12; // r10
  unsigned int v13; // r15d
  __int64 v14; // r8
  unsigned int v15; // r9d
  __int64 v16; // r12
  int v17; // ebx
  __int64 *v18; // rdi
  unsigned int v19; // esi
  __int64 *v20; // rcx
  __int64 v21; // r13
  unsigned __int64 v22; // rbx
  int v23; // r11d
  __int64 *v24; // rsi
  unsigned int v25; // ecx
  __int64 *v26; // rdx
  __int64 v27; // rdi
  unsigned __int64 v28; // rax
  __int64 v29; // rbx
  unsigned int v30; // edx
  unsigned __int64 v31; // rcx
  int v32; // r9d
  __int64 *v33; // rsi
  unsigned int v34; // r8d
  __int64 *v35; // rax
  __int64 v36; // rdi
  unsigned __int64 v37; // r15
  __int64 v38; // r9
  _QWORD *v39; // rsi
  unsigned int v40; // r13d
  unsigned int v41; // r8d
  _QWORD *v42; // rax
  __int64 v43; // rdi
  __int64 v44; // r12
  __int64 v45; // r14
  __int64 v46; // r13
  unsigned __int64 v47; // rcx
  unsigned __int64 v48; // rax
  _QWORD *v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r15
  __int64 *v52; // rdi
  _QWORD *j; // rdx
  __int64 *v54; // rax
  char *v55; // r11
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
  char *v77; // r11
  __int64 v78; // r9
  int v79; // esi
  int v80; // esi
  __int64 v81; // r10
  unsigned int v82; // ecx
  _QWORD *v83; // rdx
  __int64 v84; // r8
  _QWORD *v85; // rdi
  int v86; // edx
  int v87; // eax
  int v88; // ecx
  unsigned int v89; // edx
  __int64 v90; // r8
  __int64 v91; // rax
  int v92; // eax
  __int64 v93; // rax
  _QWORD *v94; // rax
  __int64 v95; // rdx
  _QWORD *m; // rdx
  __int64 *v97; // rax
  __int64 v98; // r11
  int v99; // esi
  int v100; // esi
  __int64 v101; // r8
  unsigned int v102; // ecx
  _QWORD *v103; // rdx
  __int64 v104; // rdi
  _QWORD *v105; // rcx
  int v106; // edx
  int v107; // edx
  _QWORD *v108; // r8
  unsigned int v109; // r13d
  __int64 v110; // rdi
  int v111; // eax
  __int64 v112; // rax
  _QWORD *v113; // rax
  __int64 v114; // rdx
  _QWORD *i; // rdx
  __int64 *v116; // rax
  __int64 v117; // r11
  int v118; // esi
  int v119; // esi
  __int64 v120; // r8
  unsigned int v121; // ecx
  _QWORD *v122; // rdx
  __int64 v123; // rdi
  _QWORD *v124; // rcx
  int v125; // edx
  int v126; // edx
  int v127; // r9d
  __int64 *v128; // r8
  unsigned int v129; // r15d
  __int64 v130; // rax
  int v131; // eax
  int v132; // ecx
  int v133; // ecx
  int v134; // ecx
  __int64 v135; // r12
  unsigned int v136; // edx
  __int64 v137; // rdi
  int v138; // edi
  int v139; // edx
  int v140; // edi
  int v141; // ecx
  int v142; // ecx
  __int64 v143; // r12
  unsigned int v144; // eax
  __int64 v145; // rsi
  int v146; // eax
  __int64 v147; // rcx
  __int64 v148; // rdi
  __int64 v149; // rcx
  __int64 v150; // rcx
  int v151; // eax
  int v152; // esi
  __int64 v153; // rbx
  __int64 v154; // r12
  __int64 *v155; // rbx
  __int64 v156; // rcx
  int v157; // r11d
  __int64 *v158; // r9
  int v159; // r13d
  char *v160; // r15
  __int64 *v161; // rbx
  __int64 v162; // rax
  __int64 v163; // rax
  int v164; // ebx
  __int64 v165; // r9
  int v166; // r11d
  __int64 v167; // r9
  int v168; // esi
  int v169; // [rsp+Ch] [rbp-A4h]
  int v170; // [rsp+Ch] [rbp-A4h]
  char *v171; // [rsp+10h] [rbp-A0h]
  __int64 v172; // [rsp+20h] [rbp-90h]
  char *v173; // [rsp+30h] [rbp-80h]
  char *v174; // [rsp+38h] [rbp-78h]
  char *v175; // [rsp+38h] [rbp-78h]
  unsigned int v176; // [rsp+38h] [rbp-78h]
  char *v177; // [rsp+38h] [rbp-78h]
  unsigned int v178; // [rsp+38h] [rbp-78h]
  char *v179; // [rsp+38h] [rbp-78h]
  char *v180; // [rsp+38h] [rbp-78h]
  char *v181; // [rsp+38h] [rbp-78h]
  char *v182; // [rsp+38h] [rbp-78h]
  char *v183; // [rsp+38h] [rbp-78h]
  int v184; // [rsp+38h] [rbp-78h]
  int v185; // [rsp+38h] [rbp-78h]
  _QWORD *v186; // [rsp+38h] [rbp-78h]
  _QWORD *v187; // [rsp+38h] [rbp-78h]
  unsigned int v188; // [rsp+40h] [rbp-70h]
  char *v189; // [rsp+40h] [rbp-70h]
  unsigned int v190; // [rsp+40h] [rbp-70h]
  char *v191; // [rsp+40h] [rbp-70h]
  char *v192; // [rsp+40h] [rbp-70h]
  __int64 v193; // [rsp+40h] [rbp-70h]
  char *v194; // [rsp+40h] [rbp-70h]
  __int64 v195; // [rsp+40h] [rbp-70h]
  __int64 v196; // [rsp+40h] [rbp-70h]
  __int64 v197; // [rsp+40h] [rbp-70h]
  __int64 v198; // [rsp+40h] [rbp-70h]
  __int64 v199; // [rsp+40h] [rbp-70h]
  _QWORD *v200; // [rsp+40h] [rbp-70h]
  _QWORD *v201; // [rsp+40h] [rbp-70h]
  __int64 *v202; // [rsp+48h] [rbp-68h]
  char *v203; // [rsp+50h] [rbp-60h]
  __int64 v205; // [rsp+68h] [rbp-48h] BYREF
  __int64 v206; // [rsp+70h] [rbp-40h] BYREF
  __int64 v207[7]; // [rsp+78h] [rbp-38h] BYREF

  result = a2 - (char *)a1;
  v173 = a2;
  v172 = a3;
  if ( a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v202 = (__int64 *)a2;
    goto LABEL_216;
  }
  v171 = (char *)(a1 + 1);
  while ( 2 )
  {
    v207[0] = a4;
    --v172;
    v6 = &a1[result >> 4];
    v7 = !sub_25E36B0(v207, a1[1], *v6);
    v8 = *((_QWORD *)v173 - 1);
    if ( v7 )
    {
      if ( sub_25E36B0(v207, a1[1], v8) )
      {
        v11 = *a1;
        v10 = a1[1];
        a1[1] = *a1;
        *a1 = v10;
        goto LABEL_7;
      }
      v160 = v173;
      if ( !sub_25E36B0(v207, *v6, *((_QWORD *)v173 - 1)) )
      {
        v163 = *a1;
        *a1 = *v6;
        *v6 = v163;
        v10 = *a1;
        v11 = a1[1];
        goto LABEL_7;
      }
      v161 = a1;
LABEL_232:
      v162 = *v161;
      *v161 = *((_QWORD *)v160 - 1);
      *((_QWORD *)v160 - 1) = v162;
      v10 = *v161;
      v11 = v161[1];
      goto LABEL_7;
    }
    if ( !sub_25E36B0(v207, *v6, v8) )
    {
      v160 = v173;
      v161 = a1;
      if ( !sub_25E36B0(v207, a1[1], *((_QWORD *)v173 - 1)) )
      {
        v11 = *a1;
        v10 = a1[1];
        a1[1] = *a1;
        *a1 = v10;
        goto LABEL_7;
      }
      goto LABEL_232;
    }
    v9 = *a1;
    *a1 = *v6;
    *v6 = v9;
    v10 = *a1;
    v11 = a1[1];
LABEL_7:
    v12 = v173;
    v203 = v171;
    v13 = *(_DWORD *)(a4 + 24);
    v14 = a4;
    while ( 1 )
    {
      v205 = v11;
      v206 = v10;
      v202 = (__int64 *)v203;
      if ( !v13 )
      {
        ++*(_QWORD *)v14;
        v207[0] = 0;
LABEL_168:
        v182 = v12;
        v198 = v14;
        sub_9DDA50(v14, 2 * v13);
        v14 = v198;
        v12 = v182;
        v141 = *(_DWORD *)(v198 + 24);
        if ( v141 )
        {
          v11 = v205;
          v142 = v141 - 1;
          v143 = *(_QWORD *)(v198 + 8);
          v144 = v142 & (((unsigned int)v205 >> 9) ^ ((unsigned int)v205 >> 4));
          v18 = (__int64 *)(v143 + 16LL * v144);
          v145 = *v18;
          if ( v205 == *v18 )
          {
LABEL_170:
            v146 = *(_DWORD *)(v198 + 16);
            v207[0] = (__int64)v18;
            v132 = v146 + 1;
          }
          else
          {
            v164 = 1;
            v165 = 0;
            while ( v145 != -4096 )
            {
              if ( !v165 && v145 == -8192 )
                v165 = (__int64)v18;
              v144 = v142 & (v164 + v144);
              v18 = (__int64 *)(v143 + 16LL * v144);
              v145 = *v18;
              if ( v205 == *v18 )
                goto LABEL_170;
              ++v164;
            }
            if ( !v165 )
              v165 = (__int64)v18;
            v132 = *(_DWORD *)(v198 + 16) + 1;
            v207[0] = v165;
            v18 = (__int64 *)v165;
          }
        }
        else
        {
          v151 = *(_DWORD *)(v198 + 16);
          v11 = v205;
          v207[0] = 0;
          v18 = 0;
          v132 = v151 + 1;
        }
        goto LABEL_141;
      }
      v15 = v13 - 1;
      v16 = *(_QWORD *)(v14 + 8);
      v17 = 1;
      v18 = 0;
      v19 = (v13 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v20 = (__int64 *)(v16 + 16LL * v19);
      v21 = *v20;
      if ( v11 == *v20 )
      {
LABEL_10:
        v22 = v20[1];
        goto LABEL_11;
      }
      while ( v21 != -4096 )
      {
        if ( !v18 && v21 == -8192 )
          v18 = v20;
        v19 = v15 & (v17 + v19);
        v20 = (__int64 *)(v16 + 16LL * v19);
        v21 = *v20;
        if ( v11 == *v20 )
          goto LABEL_10;
        ++v17;
      }
      v131 = *(_DWORD *)(v14 + 16);
      if ( !v18 )
        v18 = v20;
      ++*(_QWORD *)v14;
      v132 = v131 + 1;
      v207[0] = (__int64)v18;
      if ( 4 * (v131 + 1) >= 3 * v13 )
        goto LABEL_168;
      if ( v13 - *(_DWORD *)(v14 + 20) - v132 <= v13 >> 3 )
      {
        v183 = v12;
        v199 = v14;
        sub_9DDA50(v14, v13);
        sub_25E0C90(v199, &v205, v207);
        v14 = v199;
        v11 = v205;
        v18 = (__int64 *)v207[0];
        v12 = v183;
        v132 = *(_DWORD *)(v199 + 16) + 1;
      }
LABEL_141:
      *(_DWORD *)(v14 + 16) = v132;
      if ( *v18 != -4096 )
        --*(_DWORD *)(v14 + 20);
      *v18 = v11;
      v18[1] = 0;
      v13 = *(_DWORD *)(v14 + 24);
      if ( !v13 )
      {
        ++*(_QWORD *)v14;
        v22 = 0;
        v207[0] = 0;
LABEL_145:
        v180 = v12;
        v196 = v14;
        sub_9DDA50(v14, 2 * v13);
        v14 = v196;
        v12 = v180;
        v133 = *(_DWORD *)(v196 + 24);
        if ( v133 )
        {
          v10 = v206;
          v134 = v133 - 1;
          v135 = *(_QWORD *)(v196 + 8);
          v136 = v134 & (((unsigned int)v206 >> 9) ^ ((unsigned int)v206 >> 4));
          v24 = (__int64 *)(v135 + 16LL * v136);
          v137 = *v24;
          if ( v206 == *v24 )
          {
LABEL_147:
            v138 = *(_DWORD *)(v196 + 16);
            v207[0] = (__int64)v24;
            v139 = v138 + 1;
          }
          else
          {
            v166 = 1;
            v167 = 0;
            while ( v137 != -4096 )
            {
              if ( !v167 && v137 == -8192 )
                v167 = (__int64)v24;
              v136 = v134 & (v166 + v136);
              v24 = (__int64 *)(v135 + 16LL * v136);
              v137 = *v24;
              if ( v206 == *v24 )
                goto LABEL_147;
              ++v166;
            }
            if ( !v167 )
              v167 = (__int64)v24;
            v168 = *(_DWORD *)(v196 + 16);
            v207[0] = v167;
            v139 = v168 + 1;
            v24 = (__int64 *)v167;
          }
        }
        else
        {
          v152 = *(_DWORD *)(v196 + 16);
          v10 = v206;
          v207[0] = 0;
          v139 = v152 + 1;
          v24 = 0;
        }
        goto LABEL_148;
      }
      v16 = *(_QWORD *)(v14 + 8);
      v10 = v206;
      v15 = v13 - 1;
      v22 = 0;
LABEL_11:
      v23 = 1;
      v24 = 0;
      v25 = v15 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v26 = (__int64 *)(v16 + 16LL * v25);
      v27 = *v26;
      if ( v10 == *v26 )
      {
LABEL_12:
        v28 = v26[1];
        goto LABEL_13;
      }
      while ( v27 != -4096 )
      {
        if ( !v24 && v27 == -8192 )
          v24 = v26;
        v25 = v15 & (v23 + v25);
        v26 = (__int64 *)(v16 + 16LL * v25);
        v27 = *v26;
        if ( v10 == *v26 )
          goto LABEL_12;
        ++v23;
      }
      v140 = *(_DWORD *)(v14 + 16);
      if ( !v24 )
        v24 = v26;
      ++*(_QWORD *)v14;
      v139 = v140 + 1;
      v207[0] = (__int64)v24;
      if ( 4 * (v140 + 1) >= 3 * v13 )
        goto LABEL_145;
      if ( v13 - (v139 + *(_DWORD *)(v14 + 20)) <= v13 >> 3 )
      {
        v181 = v12;
        v197 = v14;
        sub_9DDA50(v14, v13);
        sub_25E0C90(v197, &v206, v207);
        v14 = v197;
        v10 = v206;
        v12 = v181;
        v139 = *(_DWORD *)(v197 + 16) + 1;
        v24 = (__int64 *)v207[0];
      }
LABEL_148:
      *(_DWORD *)(v14 + 16) = v139;
      if ( *v24 != -4096 )
        --*(_DWORD *)(v14 + 20);
      *v24 = v10;
      v28 = 0;
      v24[1] = 0;
      v13 = *(_DWORD *)(v14 + 24);
LABEL_13:
      if ( v28 >= v22 )
        break;
LABEL_67:
      v11 = *((_QWORD *)v203 + 1);
      v10 = *a1;
      v203 += 8;
    }
    v12 -= 8;
    v29 = v14;
    v30 = v13;
    while ( 1 )
    {
      v44 = *(_QWORD *)v12;
      v45 = *(_QWORD *)(v29 + 8);
      v46 = *a1;
      if ( v30 )
      {
        v31 = v30 - 1;
        v32 = 1;
        v33 = 0;
        v34 = v31 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
        v35 = (__int64 *)(v45 + 16LL * v34);
        v36 = *v35;
        if ( v46 == *v35 )
        {
LABEL_16:
          v37 = v35[1];
          goto LABEL_17;
        }
        while ( v36 != -4096 )
        {
          if ( !v33 && v36 == -8192 )
            v33 = v35;
          v34 = v31 & (v32 + v34);
          v35 = (__int64 *)(v45 + 16LL * v34);
          v36 = *v35;
          if ( v46 == *v35 )
            goto LABEL_16;
          ++v32;
        }
        if ( !v33 )
          v33 = v35;
        v111 = *(_DWORD *)(v29 + 16);
        ++*(_QWORD *)v29;
        v65 = v111 + 1;
        if ( 4 * (v111 + 1) < 3 * v30 )
        {
          if ( v30 - *(_DWORD *)(v29 + 20) - v65 <= v30 >> 3 )
          {
            v178 = v30;
            v194 = v12;
            v112 = (((((((((v31 | (v31 >> 1)) >> 2) | v31 | (v31 >> 1)) >> 4)
                      | ((v31 | (v31 >> 1)) >> 2)
                      | v31
                      | (v31 >> 1)) >> 8)
                    | ((((v31 | (v31 >> 1)) >> 2) | v31 | (v31 >> 1)) >> 4)
                    | ((v31 | (v31 >> 1)) >> 2)
                    | v31
                    | (v31 >> 1)) >> 16)
                  | ((((((v31 | (v31 >> 1)) >> 2) | v31 | (v31 >> 1)) >> 4)
                    | ((v31 | (v31 >> 1)) >> 2)
                    | v31
                    | (v31 >> 1)) >> 8)
                  | ((((v31 | (v31 >> 1)) >> 2) | v31 | (v31 >> 1)) >> 4)
                  | ((v31 | (v31 >> 1)) >> 2)
                  | v31
                  | (v31 >> 1))
                 + 1;
            if ( (unsigned int)v112 < 0x40 )
              LODWORD(v112) = 64;
            *(_DWORD *)(v29 + 24) = v112;
            v113 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v112, 8);
            v12 = v194;
            *(_QWORD *)(v29 + 8) = v113;
            if ( v45 )
            {
              v114 = *(unsigned int *)(v29 + 24);
              *(_QWORD *)(v29 + 16) = 0;
              v195 = 16LL * v178;
              for ( i = &v113[2 * v114]; i != v113; v113 += 2 )
              {
                if ( v113 )
                  *v113 = -4096;
              }
              v116 = (__int64 *)v45;
              do
              {
                v117 = *v116;
                if ( *v116 != -8192 && v117 != -4096 )
                {
                  v118 = *(_DWORD *)(v29 + 24);
                  if ( !v118 )
                  {
                    MEMORY[0] = *v116;
                    BUG();
                  }
                  v119 = v118 - 1;
                  v120 = *(_QWORD *)(v29 + 8);
                  v121 = v119 & (((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4));
                  v122 = (_QWORD *)(v120 + 16LL * v121);
                  v123 = *v122;
                  if ( v117 != *v122 )
                  {
                    v169 = 1;
                    v186 = 0;
                    while ( v123 != -4096 )
                    {
                      if ( !v186 )
                      {
                        if ( v123 != -8192 )
                          v122 = 0;
                        v186 = v122;
                      }
                      v121 = v119 & (v169 + v121);
                      v122 = (_QWORD *)(v120 + 16LL * v121);
                      v123 = *v122;
                      if ( v117 == *v122 )
                        goto LABEL_121;
                      ++v169;
                    }
                    if ( v186 )
                      v122 = v186;
                  }
LABEL_121:
                  *v122 = v117;
                  v122[1] = v116[1];
                  ++*(_DWORD *)(v29 + 16);
                }
                v116 += 2;
              }
              while ( (__int64 *)(v45 + v195) != v116 );
              v179 = v12;
              sub_C7D6A0(v45, v195, 8);
              v124 = *(_QWORD **)(v29 + 8);
              v125 = *(_DWORD *)(v29 + 24);
              v12 = v179;
              v65 = *(_DWORD *)(v29 + 16) + 1;
            }
            else
            {
              v150 = *(unsigned int *)(v29 + 24);
              *(_QWORD *)(v29 + 16) = 0;
              v125 = v150;
              v124 = &v113[2 * v150];
              if ( v113 == v124 )
              {
                v65 = 1;
              }
              else
              {
                do
                {
                  if ( v113 )
                    *v113 = -4096;
                  v113 += 2;
                }
                while ( v124 != v113 );
                v124 = *(_QWORD **)(v29 + 8);
                v125 = *(_DWORD *)(v29 + 24);
                v65 = *(_DWORD *)(v29 + 16) + 1;
              }
            }
            if ( !v125 )
            {
LABEL_309:
              ++*(_DWORD *)(v29 + 16);
              BUG();
            }
            v126 = v125 - 1;
            v127 = 1;
            v128 = 0;
            v129 = v126 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
            v33 = &v124[2 * v129];
            v130 = *v33;
            if ( v46 != *v33 )
            {
              while ( v130 != -4096 )
              {
                if ( !v128 && v130 == -8192 )
                  v128 = v33;
                v129 = v126 & (v127 + v129);
                v33 = &v124[2 * v129];
                v130 = *v33;
                if ( v46 == *v33 )
                  goto LABEL_40;
                ++v127;
              }
              if ( v128 )
                v33 = v128;
            }
          }
          goto LABEL_40;
        }
      }
      else
      {
        ++*(_QWORD *)v29;
      }
      v174 = v12;
      v188 = v30;
      v47 = ((((((((2 * v30 - 1) | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 2)
               | (2 * v30 - 1)
               | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 4)
             | (((2 * v30 - 1) | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 2)
             | (2 * v30 - 1)
             | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 8)
           | (((((2 * v30 - 1) | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 2)
             | (2 * v30 - 1)
             | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 4)
           | (((2 * v30 - 1) | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 2)
           | (2 * v30 - 1)
           | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 16;
      v48 = (v47
           | (((((((2 * v30 - 1) | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 2)
               | (2 * v30 - 1)
               | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 4)
             | (((2 * v30 - 1) | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 2)
             | (2 * v30 - 1)
             | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 8)
           | (((((2 * v30 - 1) | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 2)
             | (2 * v30 - 1)
             | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 4)
           | (((2 * v30 - 1) | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 2)
           | (2 * v30 - 1)
           | ((unsigned __int64)(2 * v30 - 1) >> 1))
          + 1;
      if ( (unsigned int)v48 < 0x40 )
        LODWORD(v48) = 64;
      *(_DWORD *)(v29 + 24) = v48;
      v49 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v48, 8);
      v12 = v174;
      *(_QWORD *)(v29 + 8) = v49;
      if ( v45 )
      {
        v50 = *(unsigned int *)(v29 + 24);
        *(_QWORD *)(v29 + 16) = 0;
        v51 = 16LL * v188;
        v52 = (__int64 *)(v45 + v51);
        for ( j = &v49[2 * v50]; j != v49; v49 += 2 )
        {
          if ( v49 )
            *v49 = -4096;
        }
        v54 = (__int64 *)v45;
        v55 = v174;
        if ( (__int64 *)v45 != v52 )
        {
          do
          {
            v56 = *v54;
            if ( *v54 != -8192 && v56 != -4096 )
            {
              v57 = *(_DWORD *)(v29 + 24);
              if ( !v57 )
              {
                MEMORY[0] = *v54;
                BUG();
              }
              v58 = v57 - 1;
              v59 = *(_QWORD *)(v29 + 8);
              v60 = v58 & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
              v61 = (_QWORD *)(v59 + 16LL * v60);
              v62 = *v61;
              if ( v56 != *v61 )
              {
                v185 = 1;
                v201 = 0;
                while ( v62 != -4096 )
                {
                  if ( !v201 )
                  {
                    if ( v62 != -8192 )
                      v61 = 0;
                    v201 = v61;
                  }
                  v60 = v58 & (v185 + v60);
                  v61 = (_QWORD *)(v59 + 16LL * v60);
                  v62 = *v61;
                  if ( v56 == *v61 )
                    goto LABEL_34;
                  ++v185;
                }
                if ( v201 )
                  v61 = v201;
              }
LABEL_34:
              *v61 = v56;
              v61[1] = v54[1];
              ++*(_DWORD *)(v29 + 16);
            }
            v54 += 2;
          }
          while ( v52 != v54 );
          v12 = v55;
        }
        v189 = v12;
        sub_C7D6A0(v45, v51, 8);
        v63 = *(_QWORD **)(v29 + 8);
        v64 = *(_DWORD *)(v29 + 24);
        v12 = v189;
        v65 = *(_DWORD *)(v29 + 16) + 1;
      }
      else
      {
        v147 = *(unsigned int *)(v29 + 24);
        *(_QWORD *)(v29 + 16) = 0;
        v64 = v147;
        v63 = &v49[2 * v147];
        if ( v49 == v63 )
        {
          v65 = 1;
        }
        else
        {
          do
          {
            if ( v49 )
              *v49 = -4096;
            v49 += 2;
          }
          while ( v63 != v49 );
          v63 = *(_QWORD **)(v29 + 8);
          v64 = *(_DWORD *)(v29 + 24);
          v65 = *(_DWORD *)(v29 + 16) + 1;
        }
      }
      if ( !v64 )
        goto LABEL_309;
      v66 = v64 - 1;
      v67 = v66 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
      v33 = &v63[2 * v67];
      v68 = *v33;
      if ( v46 != *v33 )
      {
        v157 = 1;
        v158 = 0;
        while ( v68 != -4096 )
        {
          if ( v68 != -8192 || v158 )
            v33 = v158;
          v67 = v66 & (v157 + v67);
          v68 = v63[2 * v67];
          if ( v46 == v68 )
          {
            v33 = &v63[2 * v67];
            goto LABEL_40;
          }
          ++v157;
          v158 = v33;
          v33 = &v63[2 * v67];
        }
        if ( v158 )
          v33 = v158;
      }
LABEL_40:
      *(_DWORD *)(v29 + 16) = v65;
      if ( *v33 != -4096 )
        --*(_DWORD *)(v29 + 20);
      *v33 = v46;
      v33[1] = 0;
      v30 = *(_DWORD *)(v29 + 24);
      v45 = *(_QWORD *)(v29 + 8);
      if ( !v30 )
      {
        ++*(_QWORD *)v29;
        v37 = 0;
LABEL_44:
        v175 = v12;
        v190 = v30;
        v69 = ((((((((2 * v30 - 1) | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 2)
                 | (2 * v30 - 1)
                 | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 4)
               | (((2 * v30 - 1) | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 2)
               | (2 * v30 - 1)
               | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 8)
             | (((((2 * v30 - 1) | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 2)
               | (2 * v30 - 1)
               | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 4)
             | (((2 * v30 - 1) | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 2)
             | (2 * v30 - 1)
             | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 16;
        v70 = (v69
             | (((((((2 * v30 - 1) | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 2)
                 | (2 * v30 - 1)
                 | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 4)
               | (((2 * v30 - 1) | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 2)
               | (2 * v30 - 1)
               | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 8)
             | (((((2 * v30 - 1) | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 2)
               | (2 * v30 - 1)
               | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 4)
             | (((2 * v30 - 1) | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 2)
             | (2 * v30 - 1)
             | ((unsigned __int64)(2 * v30 - 1) >> 1))
            + 1;
        if ( (unsigned int)v70 < 0x40 )
          LODWORD(v70) = 64;
        *(_DWORD *)(v29 + 24) = v70;
        v71 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v70, 8);
        v12 = v175;
        *(_QWORD *)(v29 + 8) = v71;
        if ( v45 )
        {
          v72 = *(unsigned int *)(v29 + 24);
          *(_QWORD *)(v29 + 16) = 0;
          v73 = 16LL * v190;
          v74 = (__int64 *)(v45 + v73);
          for ( k = &v71[2 * v72]; k != v71; v71 += 2 )
          {
            if ( v71 )
              *v71 = -4096;
          }
          v76 = (__int64 *)v45;
          v77 = v175;
          if ( v74 != (__int64 *)v45 )
          {
            do
            {
              v78 = *v76;
              if ( *v76 != -4096 && v78 != -8192 )
              {
                v79 = *(_DWORD *)(v29 + 24);
                if ( !v79 )
                {
                  MEMORY[0] = *v76;
                  BUG();
                }
                v80 = v79 - 1;
                v81 = *(_QWORD *)(v29 + 8);
                v82 = v80 & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
                v83 = (_QWORD *)(v81 + 16LL * v82);
                v84 = *v83;
                if ( *v83 != v78 )
                {
                  v184 = 1;
                  v200 = 0;
                  while ( v84 != -4096 )
                  {
                    if ( v84 == -8192 )
                    {
                      if ( v200 )
                        v83 = v200;
                      v200 = v83;
                    }
                    v82 = v80 & (v184 + v82);
                    v83 = (_QWORD *)(v81 + 16LL * v82);
                    v84 = *v83;
                    if ( v78 == *v83 )
                      goto LABEL_56;
                    ++v184;
                  }
                  if ( v200 )
                    v83 = v200;
                }
LABEL_56:
                *v83 = v78;
                v83[1] = v76[1];
                ++*(_DWORD *)(v29 + 16);
              }
              v76 += 2;
            }
            while ( v74 != v76 );
            v12 = v77;
          }
          v191 = v12;
          sub_C7D6A0(v45, v73, 8);
          v85 = *(_QWORD **)(v29 + 8);
          v86 = *(_DWORD *)(v29 + 24);
          v12 = v191;
          v87 = *(_DWORD *)(v29 + 16) + 1;
        }
        else
        {
          v148 = *(unsigned int *)(v29 + 24);
          *(_QWORD *)(v29 + 16) = 0;
          v86 = v148;
          v85 = &v71[2 * v148];
          if ( v71 == v85 )
          {
            v87 = 1;
          }
          else
          {
            do
            {
              if ( v71 )
                *v71 = -4096;
              v71 += 2;
            }
            while ( v85 != v71 );
            v85 = *(_QWORD **)(v29 + 8);
            v86 = *(_DWORD *)(v29 + 24);
            v87 = *(_DWORD *)(v29 + 16) + 1;
          }
        }
        if ( !v86 )
          goto LABEL_306;
        v88 = v86 - 1;
        v89 = (v86 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
        v39 = &v85[2 * v89];
        v90 = *v39;
        if ( v44 != *v39 )
        {
          v159 = 1;
          v38 = 0;
          while ( v90 != -4096 )
          {
            if ( !v38 && v90 == -8192 )
              v38 = (__int64)v39;
            v89 = v88 & (v159 + v89);
            v39 = &v85[2 * v89];
            v90 = *v39;
            if ( v44 == *v39 )
              goto LABEL_62;
            ++v159;
          }
          if ( v38 )
            v39 = (_QWORD *)v38;
        }
        goto LABEL_62;
      }
      v31 = v30 - 1;
      v37 = 0;
LABEL_17:
      v38 = 1;
      v39 = 0;
      v40 = ((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4);
      v41 = v40 & v31;
      v42 = (_QWORD *)(v45 + 16LL * (v40 & (unsigned int)v31));
      v43 = *v42;
      if ( v44 != *v42 )
        break;
LABEL_18:
      if ( v37 <= v42[1] )
        goto LABEL_65;
LABEL_19:
      v30 = *(_DWORD *)(v29 + 24);
      v12 -= 8;
    }
    while ( v43 != -4096 )
    {
      if ( v43 == -8192 && !v39 )
        v39 = v42;
      v41 = v31 & (v38 + v41);
      v42 = (_QWORD *)(v45 + 16LL * v41);
      v43 = *v42;
      if ( v44 == *v42 )
        goto LABEL_18;
      v38 = (unsigned int)(v38 + 1);
    }
    if ( !v39 )
      v39 = v42;
    v92 = *(_DWORD *)(v29 + 16);
    ++*(_QWORD *)v29;
    v87 = v92 + 1;
    if ( 4 * v87 >= 3 * v30 )
      goto LABEL_44;
    if ( v30 - (v87 + *(_DWORD *)(v29 + 20)) <= v30 >> 3 )
    {
      v176 = v30;
      v192 = v12;
      v93 = ((((((((v31 >> 1) | v31 | (((v31 >> 1) | v31) >> 2)) >> 4) | (v31 >> 1) | v31 | (((v31 >> 1) | v31) >> 2)) >> 8)
             | (((v31 >> 1) | v31 | (((v31 >> 1) | v31) >> 2)) >> 4)
             | (v31 >> 1)
             | v31
             | (((v31 >> 1) | v31) >> 2)) >> 16)
           | (((((v31 >> 1) | v31 | (((v31 >> 1) | v31) >> 2)) >> 4) | (v31 >> 1) | v31 | (((v31 >> 1) | v31) >> 2)) >> 8)
           | (((v31 >> 1) | v31 | (((v31 >> 1) | v31) >> 2)) >> 4)
           | (v31 >> 1)
           | v31
           | (((v31 >> 1) | v31) >> 2))
          + 1;
      if ( (unsigned int)v93 < 0x40 )
        LODWORD(v93) = 64;
      *(_DWORD *)(v29 + 24) = v93;
      v94 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v93, 8);
      v12 = v192;
      *(_QWORD *)(v29 + 8) = v94;
      if ( v45 )
      {
        v95 = *(unsigned int *)(v29 + 24);
        *(_QWORD *)(v29 + 16) = 0;
        v193 = 16LL * v176;
        for ( m = &v94[2 * v95]; m != v94; v94 += 2 )
        {
          if ( v94 )
            *v94 = -4096;
        }
        v97 = (__int64 *)v45;
        do
        {
          v98 = *v97;
          if ( *v97 != -4096 && v98 != -8192 )
          {
            v99 = *(_DWORD *)(v29 + 24);
            if ( !v99 )
            {
              MEMORY[0] = *v97;
              BUG();
            }
            v100 = v99 - 1;
            v101 = *(_QWORD *)(v29 + 8);
            v102 = v100 & (((unsigned int)v98 >> 9) ^ ((unsigned int)v98 >> 4));
            v103 = (_QWORD *)(v101 + 16LL * v102);
            v104 = *v103;
            if ( v98 != *v103 )
            {
              v170 = 1;
              v187 = 0;
              while ( v104 != -4096 )
              {
                if ( !v187 )
                {
                  if ( v104 != -8192 )
                    v103 = 0;
                  v187 = v103;
                }
                v102 = v100 & (v170 + v102);
                v103 = (_QWORD *)(v101 + 16LL * v102);
                v104 = *v103;
                if ( v98 == *v103 )
                  goto LABEL_90;
                ++v170;
              }
              if ( v187 )
                v103 = v187;
            }
LABEL_90:
            *v103 = v98;
            v103[1] = v97[1];
            ++*(_DWORD *)(v29 + 16);
          }
          v97 += 2;
        }
        while ( (__int64 *)(v45 + v193) != v97 );
        v177 = v12;
        sub_C7D6A0(v45, v193, 8);
        v105 = *(_QWORD **)(v29 + 8);
        v106 = *(_DWORD *)(v29 + 24);
        v12 = v177;
        v87 = *(_DWORD *)(v29 + 16) + 1;
      }
      else
      {
        v149 = *(unsigned int *)(v29 + 24);
        *(_QWORD *)(v29 + 16) = 0;
        v106 = v149;
        v105 = &v94[2 * v149];
        if ( v94 == v105 )
        {
          v87 = 1;
        }
        else
        {
          do
          {
            if ( v94 )
              *v94 = -4096;
            v94 += 2;
          }
          while ( v105 != v94 );
          v105 = *(_QWORD **)(v29 + 8);
          v106 = *(_DWORD *)(v29 + 24);
          v87 = *(_DWORD *)(v29 + 16) + 1;
        }
      }
      if ( !v106 )
      {
LABEL_306:
        ++*(_DWORD *)(v29 + 16);
        BUG();
      }
      v107 = v106 - 1;
      v38 = 1;
      v108 = 0;
      v109 = v107 & v40;
      v39 = &v105[2 * v109];
      v110 = *v39;
      if ( v44 != *v39 )
      {
        while ( v110 != -4096 )
        {
          if ( !v108 && v110 == -8192 )
            v108 = v39;
          v109 = v107 & (v38 + v109);
          v39 = &v105[2 * v109];
          v110 = *v39;
          if ( v44 == *v39 )
            goto LABEL_62;
          v38 = (unsigned int)(v38 + 1);
        }
        if ( v108 )
          v39 = v108;
      }
    }
LABEL_62:
    *(_DWORD *)(v29 + 16) = v87;
    if ( *v39 != -4096 )
      --*(_DWORD *)(v29 + 20);
    *v39 = v44;
    v39[1] = 0;
    if ( v37 )
      goto LABEL_19;
LABEL_65:
    v14 = v29;
    if ( v203 < v12 )
    {
      v91 = *(_QWORD *)v203;
      *(_QWORD *)v203 = *(_QWORD *)v12;
      *(_QWORD *)v12 = v91;
      v13 = *(_DWORD *)(v29 + 24);
      goto LABEL_67;
    }
    a4 = v29;
    sub_25E3800(v203, v173, v172, v29, v29, v38);
    result = v203 - (char *)a1;
    if ( v203 - (char *)a1 > 128 )
    {
      if ( v172 )
      {
        v173 = v203;
        continue;
      }
LABEL_216:
      v153 = result >> 3;
      v154 = ((result >> 3) - 2) >> 1;
      sub_25E2FF0((__int64)a1, v154, result >> 3, a1[v154], a4);
      do
      {
        --v154;
        sub_25E2FF0((__int64)a1, v154, v153, a1[v154], a4);
      }
      while ( v154 );
      v155 = v202;
      do
      {
        v156 = *--v155;
        *v155 = *a1;
        result = sub_25E2FF0((__int64)a1, 0, v155 - a1, v156, a4);
      }
      while ( (char *)v155 - (char *)a1 > 8 );
    }
    return result;
  }
}
