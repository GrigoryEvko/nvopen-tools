// Function: sub_30C0440
// Address: 0x30c0440
//
__int64 __fastcall sub_30C0440(__int64 *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v6; // rbx
  bool v7; // zf
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rbx
  char *v12; // r15
  __int64 v13; // r13
  int v14; // ecx
  __int64 v15; // r9
  int v16; // r10d
  __int64 *v17; // rdx
  __int64 v18; // rdi
  __int64 *v19; // rax
  __int64 v20; // rsi
  unsigned __int64 v21; // r8
  int v22; // r10d
  __int64 *v23; // rdx
  unsigned int v24; // edi
  __int64 *v25; // rax
  __int64 v26; // rsi
  int v27; // esi
  int v28; // esi
  __int64 v29; // r8
  __int64 v30; // rcx
  int v31; // eax
  __int64 v32; // rdi
  int v33; // esi
  int v34; // esi
  __int64 v35; // r8
  unsigned int v36; // ecx
  int v37; // eax
  __int64 v38; // rdi
  int v39; // r8d
  __int64 *v40; // rcx
  unsigned __int64 v41; // rdx
  unsigned int v42; // edi
  __int64 *v43; // rax
  __int64 v44; // rsi
  unsigned __int64 v45; // rsi
  int v46; // r10d
  _QWORD *v47; // rcx
  unsigned int v48; // ebx
  unsigned int v49; // r8d
  _QWORD *v50; // rax
  __int64 v51; // rdi
  __int64 v52; // r12
  __int64 v53; // rbx
  unsigned __int64 v54; // rdx
  unsigned __int64 v55; // rax
  _QWORD *v56; // rax
  __int64 v57; // r9
  __int64 v58; // rdx
  __int64 v59; // r11
  __int64 *v60; // r13
  _QWORD *j; // rdx
  __int64 *k; // rax
  __int64 v63; // r10
  int v64; // esi
  int v65; // esi
  __int64 v66; // r8
  unsigned int v67; // ecx
  _QWORD *v68; // rdx
  __int64 v69; // rdi
  _QWORD *v70; // rdi
  int v71; // edx
  int v72; // esi
  int v73; // edx
  unsigned int v74; // eax
  __int64 v75; // r8
  __int64 v76; // r11
  unsigned __int64 v77; // rax
  __int64 v78; // rax
  _QWORD *v79; // rax
  __int64 v80; // rdx
  __int64 v81; // r13
  __int64 *v82; // r8
  _QWORD *m; // rdx
  __int64 *v84; // rax
  __int64 v85; // rdx
  int v86; // edi
  int v87; // edi
  __int64 v88; // rbx
  unsigned int v89; // esi
  _QWORD *v90; // rcx
  __int64 v91; // r10
  _QWORD *v92; // rsi
  int v93; // edx
  int v94; // eax
  int v95; // edx
  unsigned int v96; // edi
  __int64 v97; // r8
  __int64 v98; // rax
  int v99; // eax
  unsigned int v100; // eax
  _QWORD *v101; // rax
  __int64 v102; // r9
  __int64 v103; // r13
  __int64 *v104; // rdi
  _QWORD *n; // rdx
  __int64 *v106; // rax
  int v107; // esi
  int v108; // esi
  __int64 v109; // r12
  unsigned int v110; // ecx
  _QWORD *v111; // rdx
  __int64 v112; // r8
  _QWORD *v113; // rsi
  int v114; // edx
  int v115; // edx
  int v116; // r9d
  _QWORD *v117; // r8
  unsigned int v118; // ebx
  __int64 v119; // rdi
  int v120; // eax
  __int64 v121; // rax
  _QWORD *v122; // rax
  __int64 v123; // r9
  unsigned int v124; // r11d
  __int64 v125; // rdx
  __int64 *v126; // r13
  _QWORD *i; // rdx
  __int64 *v128; // rax
  __int64 v129; // rdx
  int v130; // edi
  int v131; // edi
  __int64 v132; // r10
  unsigned int v133; // esi
  _QWORD *v134; // rcx
  __int64 v135; // r8
  _QWORD *v136; // rdi
  int v137; // edx
  int v138; // edx
  int v139; // r10d
  __int64 *v140; // r9
  unsigned int v141; // r8d
  __int64 v142; // rax
  int v143; // eax
  int v144; // ecx
  int v145; // ecx
  __int64 v146; // rdi
  __int64 *v147; // r9
  unsigned int v148; // ebx
  int v149; // r10d
  __int64 v150; // rsi
  int v151; // eax
  int v152; // ecx
  int v153; // ecx
  __int64 v154; // rdi
  __int64 *v155; // r9
  int v156; // r10d
  __int64 v157; // r11
  __int64 v158; // rsi
  __int64 v159; // rsi
  __int64 v160; // rdi
  __int64 v161; // rdi
  _QWORD *v162; // r12
  __int64 v163; // rsi
  __int64 v164; // rbx
  __int64 v165; // r12
  __int64 *v166; // rbx
  __int64 v167; // rcx
  int v168; // r10d
  _QWORD *v169; // r9
  int v170; // r10d
  char *v171; // r15
  __int64 *v172; // rbx
  __int64 v173; // rax
  __int64 v174; // rax
  int v175; // r11d
  __int64 *v176; // r10
  int v177; // r11d
  __int64 *v178; // r10
  int v179; // [rsp+Ch] [rbp-94h]
  char *v180; // [rsp+10h] [rbp-90h]
  __int64 v181; // [rsp+20h] [rbp-80h]
  __int64 v182; // [rsp+28h] [rbp-78h]
  char *v183; // [rsp+30h] [rbp-70h]
  __int64 v184; // [rsp+38h] [rbp-68h]
  __int64 *v185; // [rsp+38h] [rbp-68h]
  __int64 *v186; // [rsp+38h] [rbp-68h]
  int v187; // [rsp+38h] [rbp-68h]
  _QWORD *v188; // [rsp+38h] [rbp-68h]
  int v189; // [rsp+38h] [rbp-68h]
  __int64 *v190; // [rsp+40h] [rbp-60h]
  __int64 v191; // [rsp+40h] [rbp-60h]
  __int64 v192; // [rsp+40h] [rbp-60h]
  __int64 v193; // [rsp+40h] [rbp-60h]
  __int64 v194; // [rsp+40h] [rbp-60h]
  unsigned int v195; // [rsp+40h] [rbp-60h]
  int v196; // [rsp+40h] [rbp-60h]
  _QWORD *v197; // [rsp+40h] [rbp-60h]
  _QWORD *v198; // [rsp+40h] [rbp-60h]
  __int64 *v199; // [rsp+48h] [rbp-58h]
  char *v200; // [rsp+50h] [rbp-50h]
  __int64 v202[7]; // [rsp+68h] [rbp-38h] BYREF

  result = a2 - (char *)a1;
  v183 = a2;
  v181 = a3;
  if ( a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v199 = (__int64 *)a2;
    goto LABEL_218;
  }
  v180 = (char *)(a1 + 1);
  v182 = a4 + 96;
  while ( 2 )
  {
    v202[0] = a4;
    --v181;
    v6 = &a1[result >> 4];
    v7 = !sub_30BBEA0(v202, a1[1], *v6);
    v8 = *((_QWORD *)v183 - 1);
    if ( v7 )
    {
      if ( sub_30BBEA0(v202, a1[1], v8) )
      {
        v11 = *a1;
        v10 = a1[1];
        a1[1] = *a1;
        *a1 = v10;
        goto LABEL_7;
      }
      v171 = v183;
      if ( !sub_30BBEA0(v202, *v6, *((_QWORD *)v183 - 1)) )
      {
        v174 = *a1;
        *a1 = *v6;
        *v6 = v174;
        v10 = *a1;
        v11 = a1[1];
        goto LABEL_7;
      }
      v172 = a1;
LABEL_236:
      v173 = *v172;
      *v172 = *((_QWORD *)v171 - 1);
      *((_QWORD *)v171 - 1) = v173;
      v10 = *v172;
      v11 = v172[1];
      goto LABEL_7;
    }
    if ( !sub_30BBEA0(v202, *v6, v8) )
    {
      v171 = v183;
      v172 = a1;
      if ( !sub_30BBEA0(v202, a1[1], *((_QWORD *)v183 - 1)) )
      {
        v11 = *a1;
        v10 = a1[1];
        a1[1] = *a1;
        *a1 = v10;
        goto LABEL_7;
      }
      goto LABEL_236;
    }
    v9 = *a1;
    *a1 = *v6;
    *v6 = v9;
    v10 = *a1;
    v11 = a1[1];
LABEL_7:
    v200 = v180;
    v12 = v183;
    v13 = *(unsigned int *)(a4 + 120);
    while ( 1 )
    {
      v199 = (__int64 *)v200;
      if ( (_DWORD)v13 )
      {
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a4 + 104);
        v16 = 1;
        v17 = 0;
        v18 = ((_DWORD)v13 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v19 = (__int64 *)(v15 + 16 * v18);
        v20 = *v19;
        if ( *v19 == v11 )
        {
LABEL_9:
          v21 = v19[1];
          goto LABEL_10;
        }
        while ( v20 != -4096 )
        {
          if ( !v17 && v20 == -8192 )
            v17 = v19;
          LODWORD(v18) = v14 & (v16 + v18);
          v19 = (__int64 *)(v15 + 16LL * (unsigned int)v18);
          v20 = *v19;
          if ( *v19 == v11 )
            goto LABEL_9;
          ++v16;
        }
        if ( !v17 )
          v17 = v19;
        v151 = *(_DWORD *)(a4 + 112);
        ++*(_QWORD *)(a4 + 96);
        v31 = v151 + 1;
        if ( 4 * v31 < (unsigned int)(3 * v13) )
        {
          if ( (int)v13 - *(_DWORD *)(a4 + 116) - v31 <= (unsigned int)v13 >> 3 )
          {
            v195 = ((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4);
            sub_30BBCC0(v182, v13);
            v152 = *(_DWORD *)(a4 + 120);
            if ( !v152 )
              goto LABEL_313;
            v153 = v152 - 1;
            v154 = *(_QWORD *)(a4 + 104);
            v155 = 0;
            v156 = 1;
            LODWORD(v157) = v153 & v195;
            v31 = *(_DWORD *)(a4 + 112) + 1;
            v17 = (__int64 *)(v154 + 16LL * (v153 & v195));
            v158 = *v17;
            if ( v11 != *v17 )
            {
              while ( v158 != -4096 )
              {
                if ( v158 == -8192 && !v155 )
                  v155 = v17;
                v157 = v153 & (unsigned int)(v157 + v156);
                v17 = (__int64 *)(v154 + 16 * v157);
                v158 = *v17;
                if ( *v17 == v11 )
                  goto LABEL_17;
                ++v156;
              }
              if ( v155 )
                v17 = v155;
            }
          }
          goto LABEL_17;
        }
      }
      else
      {
        ++*(_QWORD *)(a4 + 96);
      }
      sub_30BBCC0(v182, 2 * v13);
      v27 = *(_DWORD *)(a4 + 120);
      if ( !v27 )
        goto LABEL_313;
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a4 + 104);
      LODWORD(v30) = v28 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v31 = *(_DWORD *)(a4 + 112) + 1;
      v17 = (__int64 *)(v29 + 16LL * (unsigned int)v30);
      v32 = *v17;
      if ( v11 != *v17 )
      {
        v177 = 1;
        v178 = 0;
        while ( v32 != -4096 )
        {
          if ( !v178 && v32 == -8192 )
            v178 = v17;
          v30 = v28 & (unsigned int)(v30 + v177);
          v17 = (__int64 *)(v29 + 16 * v30);
          v32 = *v17;
          if ( *v17 == v11 )
            goto LABEL_17;
          ++v177;
        }
        if ( v178 )
          v17 = v178;
      }
LABEL_17:
      *(_DWORD *)(a4 + 112) = v31;
      if ( *v17 != -4096 )
        --*(_DWORD *)(a4 + 116);
      *v17 = v11;
      v17[1] = 0;
      v13 = *(unsigned int *)(a4 + 120);
      if ( !(_DWORD)v13 )
      {
        ++*(_QWORD *)(a4 + 96);
LABEL_21:
        sub_30BBCC0(v182, 2 * v13);
        v33 = *(_DWORD *)(a4 + 120);
        if ( !v33 )
          goto LABEL_313;
        v34 = v33 - 1;
        v35 = *(_QWORD *)(a4 + 104);
        v36 = v34 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v37 = *(_DWORD *)(a4 + 112) + 1;
        v23 = (__int64 *)(v35 + 16LL * v36);
        v38 = *v23;
        if ( *v23 != v10 )
        {
          v175 = 1;
          v176 = 0;
          while ( v38 != -4096 )
          {
            if ( !v176 && v38 == -8192 )
              v176 = v23;
            v36 = v34 & (v175 + v36);
            v23 = (__int64 *)(v35 + 16LL * v36);
            v38 = *v23;
            if ( *v23 == v10 )
              goto LABEL_23;
            ++v175;
          }
          if ( v176 )
            v23 = v176;
        }
        goto LABEL_23;
      }
      v15 = *(_QWORD *)(a4 + 104);
      v14 = v13 - 1;
      v21 = 0;
LABEL_10:
      v22 = 1;
      v23 = 0;
      v24 = v14 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v25 = (__int64 *)(v15 + 16LL * v24);
      v26 = *v25;
      if ( v10 != *v25 )
        break;
LABEL_11:
      if ( v21 >= v25[1] )
        goto LABEL_30;
LABEL_12:
      v10 = *a1;
      v11 = *((_QWORD *)v200 + 1);
      v200 += 8;
    }
    while ( v26 != -4096 )
    {
      if ( v26 == -8192 && !v23 )
        v23 = v25;
      v24 = v14 & (v22 + v24);
      v25 = (__int64 *)(v15 + 16LL * v24);
      v26 = *v25;
      if ( *v25 == v10 )
        goto LABEL_11;
      ++v22;
    }
    if ( !v23 )
      v23 = v25;
    v143 = *(_DWORD *)(a4 + 112);
    ++*(_QWORD *)(a4 + 96);
    v37 = v143 + 1;
    if ( 4 * v37 >= (unsigned int)(3 * v13) )
      goto LABEL_21;
    if ( (int)v13 - (v37 + *(_DWORD *)(a4 + 116)) <= (unsigned int)v13 >> 3 )
    {
      sub_30BBCC0(v182, v13);
      v144 = *(_DWORD *)(a4 + 120);
      if ( !v144 )
        goto LABEL_313;
      v145 = v144 - 1;
      v146 = *(_QWORD *)(a4 + 104);
      v147 = 0;
      v148 = v145 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v149 = 1;
      v37 = *(_DWORD *)(a4 + 112) + 1;
      v23 = (__int64 *)(v146 + 16LL * v148);
      v150 = *v23;
      if ( v10 != *v23 )
      {
        while ( v150 != -4096 )
        {
          if ( v150 == -8192 && !v147 )
            v147 = v23;
          v148 = v145 & (v149 + v148);
          v23 = (__int64 *)(v146 + 16LL * v148);
          v150 = *v23;
          if ( *v23 == v10 )
            goto LABEL_23;
          ++v149;
        }
        if ( v147 )
          v23 = v147;
      }
    }
LABEL_23:
    *(_DWORD *)(a4 + 112) = v37;
    if ( *v23 != -4096 )
      --*(_DWORD *)(a4 + 116);
    *v23 = v10;
    v23[1] = 0;
    v13 = *(unsigned int *)(a4 + 120);
    v15 = *(_QWORD *)(a4 + 104);
LABEL_30:
    while ( 2 )
    {
      v12 -= 8;
      v52 = *(_QWORD *)v12;
      v53 = *a1;
      if ( !(_DWORD)v13 )
      {
        ++*(_QWORD *)(a4 + 96);
        goto LABEL_32;
      }
      v39 = 1;
      v40 = 0;
      v41 = (unsigned int)(v13 - 1);
      v42 = v41 & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
      v43 = (__int64 *)(v15 + 16LL * v42);
      v44 = *v43;
      if ( v53 == *v43 )
      {
LABEL_27:
        v45 = v43[1];
        goto LABEL_28;
      }
      while ( v44 != -4096 )
      {
        if ( v44 == -8192 && !v40 )
          v40 = v43;
        v42 = v41 & (v39 + v42);
        v43 = (__int64 *)(v15 + 16LL * v42);
        v44 = *v43;
        if ( v53 == *v43 )
          goto LABEL_27;
        ++v39;
      }
      if ( !v40 )
        v40 = v43;
      v120 = *(_DWORD *)(a4 + 112);
      ++*(_QWORD *)(a4 + 96);
      v72 = v120 + 1;
      if ( 4 * (v120 + 1) < (unsigned int)(3 * v13) )
      {
        if ( (int)v13 - *(_DWORD *)(a4 + 116) - v72 <= (unsigned int)v13 >> 3 )
        {
          v186 = (__int64 *)v15;
          v121 = (((((((((v41 | (v41 >> 1)) >> 2) | v41 | (v41 >> 1)) >> 4)
                    | ((v41 | (v41 >> 1)) >> 2)
                    | v41
                    | (v41 >> 1)) >> 8)
                  | ((((v41 | (v41 >> 1)) >> 2) | v41 | (v41 >> 1)) >> 4)
                  | ((v41 | (v41 >> 1)) >> 2)
                  | v41
                  | (v41 >> 1)) >> 16)
                | ((((((v41 | (v41 >> 1)) >> 2) | v41 | (v41 >> 1)) >> 4) | ((v41 | (v41 >> 1)) >> 2) | v41 | (v41 >> 1)) >> 8)
                | ((((v41 | (v41 >> 1)) >> 2) | v41 | (v41 >> 1)) >> 4)
                | ((v41 | (v41 >> 1)) >> 2)
                | v41
                | (v41 >> 1))
               + 1;
          if ( (unsigned int)v121 < 0x40 )
            LODWORD(v121) = 64;
          *(_DWORD *)(a4 + 120) = v121;
          v122 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v121, 8);
          v123 = (__int64)v186;
          v124 = ((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4);
          *(_QWORD *)(a4 + 104) = v122;
          if ( v186 )
          {
            v125 = *(unsigned int *)(a4 + 120);
            *(_QWORD *)(a4 + 112) = 0;
            v194 = 2LL * (unsigned int)v13;
            v126 = &v186[v194];
            for ( i = &v122[2 * v125]; i != v122; v122 += 2 )
            {
              if ( v122 )
                *v122 = -4096;
            }
            v128 = v186;
            do
            {
              v129 = *v128;
              if ( *v128 != -8192 && v129 != -4096 )
              {
                v130 = *(_DWORD *)(a4 + 120);
                if ( !v130 )
                {
                  MEMORY[0] = *v128;
                  BUG();
                }
                v131 = v130 - 1;
                v132 = *(_QWORD *)(a4 + 104);
                v133 = v131 & (((unsigned int)v129 >> 9) ^ ((unsigned int)v129 >> 4));
                v134 = (_QWORD *)(v132 + 16LL * v133);
                v135 = *v134;
                if ( v129 != *v134 )
                {
                  v179 = 1;
                  v188 = 0;
                  while ( v135 != -4096 )
                  {
                    if ( !v188 )
                    {
                      if ( v135 != -8192 )
                        v134 = 0;
                      v188 = v134;
                    }
                    v133 = v131 & (v179 + v133);
                    v134 = (_QWORD *)(v132 + 16LL * v133);
                    v135 = *v134;
                    if ( v129 == *v134 )
                      goto LABEL_129;
                    ++v179;
                  }
                  if ( v188 )
                    v134 = v188;
                }
LABEL_129:
                *v134 = v129;
                v134[1] = v128[1];
                ++*(_DWORD *)(a4 + 112);
              }
              v128 += 2;
            }
            while ( v126 != v128 );
            sub_C7D6A0(v123, v194 * 8, 8);
            v136 = *(_QWORD **)(a4 + 104);
            v137 = *(_DWORD *)(a4 + 120);
            v124 = ((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4);
            v72 = *(_DWORD *)(a4 + 112) + 1;
          }
          else
          {
            v161 = *(unsigned int *)(a4 + 120);
            *(_QWORD *)(a4 + 112) = 0;
            v137 = v161;
            v136 = &v122[2 * v161];
            if ( v122 == v136 )
            {
              v72 = 1;
            }
            else
            {
              do
              {
                if ( v122 )
                  *v122 = -4096;
                v122 += 2;
              }
              while ( v136 != v122 );
              v136 = *(_QWORD **)(a4 + 104);
              v137 = *(_DWORD *)(a4 + 120);
              v72 = *(_DWORD *)(a4 + 112) + 1;
            }
          }
          if ( !v137 )
            goto LABEL_313;
          v138 = v137 - 1;
          v139 = 1;
          v140 = 0;
          v141 = v138 & v124;
          v40 = &v136[2 * (v138 & v124)];
          v142 = *v40;
          if ( v53 != *v40 )
          {
            while ( v142 != -4096 )
            {
              if ( !v140 && v142 == -8192 )
                v140 = v40;
              v141 = v138 & (v139 + v141);
              v40 = &v136[2 * v141];
              v142 = *v40;
              if ( v53 == *v40 )
                goto LABEL_50;
              ++v139;
            }
LABEL_135:
            if ( v140 )
              v40 = v140;
          }
        }
        goto LABEL_50;
      }
LABEL_32:
      v190 = (__int64 *)v15;
      v54 = ((((((((unsigned int)(2 * v13 - 1) | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v13 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 4)
             | (((unsigned int)(2 * v13 - 1) | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 2)
             | (unsigned int)(2 * v13 - 1)
             | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 8)
           | (((((unsigned int)(2 * v13 - 1) | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 2)
             | (unsigned int)(2 * v13 - 1)
             | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 4)
           | (((unsigned int)(2 * v13 - 1) | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 2)
           | (unsigned int)(2 * v13 - 1)
           | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 16;
      v55 = (v54
           | (((((((unsigned int)(2 * v13 - 1) | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v13 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 4)
             | (((unsigned int)(2 * v13 - 1) | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 2)
             | (unsigned int)(2 * v13 - 1)
             | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 8)
           | (((((unsigned int)(2 * v13 - 1) | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 2)
             | (unsigned int)(2 * v13 - 1)
             | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 4)
           | (((unsigned int)(2 * v13 - 1) | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 2)
           | (unsigned int)(2 * v13 - 1)
           | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1))
          + 1;
      if ( (unsigned int)v55 < 0x40 )
        LODWORD(v55) = 64;
      *(_DWORD *)(a4 + 120) = v55;
      v56 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v55, 8);
      v57 = (__int64)v190;
      *(_QWORD *)(a4 + 104) = v56;
      if ( v190 )
      {
        v58 = *(unsigned int *)(a4 + 120);
        *(_QWORD *)(a4 + 112) = 0;
        v59 = 2LL * (unsigned int)v13;
        v60 = &v190[v59];
        for ( j = &v56[2 * v58]; j != v56; v56 += 2 )
        {
          if ( v56 )
            *v56 = -4096;
        }
        for ( k = v190; v60 != k; k += 2 )
        {
          v63 = *k;
          if ( *k != -4096 && v63 != -8192 )
          {
            v64 = *(_DWORD *)(a4 + 120);
            if ( !v64 )
            {
LABEL_316:
              MEMORY[0] = v63;
              BUG();
            }
            v65 = v64 - 1;
            v66 = *(_QWORD *)(a4 + 104);
            v67 = v65 & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
            v68 = (_QWORD *)(v66 + 16LL * v67);
            v69 = *v68;
            if ( v63 != *v68 )
            {
              v187 = 1;
              v197 = 0;
              while ( v69 != -4096 )
              {
                if ( !v197 )
                {
                  if ( v69 != -8192 )
                    v68 = 0;
                  v197 = v68;
                }
                v67 = v65 & (v187 + v67);
                v68 = (_QWORD *)(v66 + 16LL * v67);
                v69 = *v68;
                if ( v63 == *v68 )
                  goto LABEL_44;
                ++v187;
              }
              if ( v197 )
                v68 = v197;
            }
LABEL_44:
            *v68 = v63;
            v68[1] = k[1];
            ++*(_DWORD *)(a4 + 112);
          }
        }
        sub_C7D6A0(v57, v59 * 8, 8);
LABEL_47:
        v70 = *(_QWORD **)(a4 + 104);
        v71 = *(_DWORD *)(a4 + 120);
        v72 = *(_DWORD *)(a4 + 112) + 1;
      }
      else
      {
        v160 = *(unsigned int *)(a4 + 120);
        *(_QWORD *)(a4 + 112) = 0;
        v71 = v160;
        v70 = &v56[2 * v160];
        if ( v56 != v70 )
        {
          do
          {
            if ( v56 )
              *v56 = -4096;
            v56 += 2;
          }
          while ( v70 != v56 );
          goto LABEL_47;
        }
        v72 = 1;
      }
      if ( !v71 )
        goto LABEL_313;
      v73 = v71 - 1;
      v74 = v73 & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
      v40 = &v70[2 * v74];
      v75 = *v40;
      if ( v53 != *v40 )
      {
        v170 = 1;
        v140 = 0;
        while ( v75 != -4096 )
        {
          if ( v75 == -8192 && !v140 )
            v140 = v40;
          v74 = v73 & (v170 + v74);
          v40 = &v70[2 * v74];
          v75 = *v40;
          if ( v53 == *v40 )
            goto LABEL_50;
          ++v170;
        }
        goto LABEL_135;
      }
LABEL_50:
      *(_DWORD *)(a4 + 112) = v72;
      if ( *v40 != -4096 )
        --*(_DWORD *)(a4 + 116);
      *v40 = v53;
      v40[1] = 0;
      v13 = *(unsigned int *)(a4 + 120);
      v15 = *(_QWORD *)(a4 + 104);
      if ( !(_DWORD)v13 )
      {
        ++*(_QWORD *)(a4 + 96);
        v76 = v52;
        goto LABEL_54;
      }
      v41 = (unsigned int)(v13 - 1);
      v45 = 0;
LABEL_28:
      v46 = 1;
      v47 = 0;
      v48 = ((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4);
      v49 = v48 & v41;
      v50 = (_QWORD *)(v15 + 16LL * (v48 & (unsigned int)v41));
      v51 = *v50;
      if ( v52 == *v50 )
      {
LABEL_29:
        if ( v45 >= v50[1] )
          goto LABEL_74;
        continue;
      }
      break;
    }
    while ( v51 != -4096 )
    {
      if ( !v47 && v51 == -8192 )
        v47 = v50;
      v49 = v41 & (v46 + v49);
      v50 = (_QWORD *)(v15 + 16LL * v49);
      v51 = *v50;
      if ( v52 == *v50 )
        goto LABEL_29;
      ++v46;
    }
    v76 = v52;
    if ( !v47 )
      v47 = v50;
    v99 = *(_DWORD *)(a4 + 112);
    ++*(_QWORD *)(a4 + 96);
    v94 = v99 + 1;
    if ( 4 * v94 >= (unsigned int)(3 * v13) )
    {
LABEL_54:
      v184 = v15;
      v191 = v76;
      v77 = (((((((unsigned int)(2 * v13 - 1) | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 2)
              | (unsigned int)(2 * v13 - 1)
              | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 4)
            | (((unsigned int)(2 * v13 - 1) | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 2)
            | (unsigned int)(2 * v13 - 1)
            | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 8)
          | (((((unsigned int)(2 * v13 - 1) | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 2)
            | (unsigned int)(2 * v13 - 1)
            | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 4)
          | (((unsigned int)(2 * v13 - 1) | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1)) >> 2)
          | (unsigned int)(2 * v13 - 1)
          | ((unsigned __int64)(unsigned int)(2 * v13 - 1) >> 1);
      v78 = ((v77 >> 16) | v77) + 1;
      if ( (unsigned int)v78 < 0x40 )
        LODWORD(v78) = 64;
      *(_DWORD *)(a4 + 120) = v78;
      v79 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v78, 8);
      v76 = v191;
      *(_QWORD *)(a4 + 104) = v79;
      if ( v184 )
      {
        v80 = *(unsigned int *)(a4 + 120);
        v81 = 16 * v13;
        *(_QWORD *)(a4 + 112) = 0;
        v82 = (__int64 *)(v184 + v81);
        for ( m = &v79[2 * v80]; m != v79; v79 += 2 )
        {
          if ( v79 )
            *v79 = -4096;
        }
        v84 = (__int64 *)v184;
        if ( (__int64 *)v184 != v82 )
        {
          do
          {
            v85 = *v84;
            if ( *v84 != -4096 && v85 != -8192 )
            {
              v86 = *(_DWORD *)(a4 + 120);
              if ( !v86 )
              {
                MEMORY[0] = *v84;
                BUG();
              }
              v87 = v86 - 1;
              v88 = *(_QWORD *)(a4 + 104);
              v89 = v87 & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
              v90 = (_QWORD *)(v88 + 16LL * v89);
              v91 = *v90;
              if ( *v90 != v85 )
              {
                v196 = 1;
                v162 = 0;
                while ( v91 != -4096 )
                {
                  if ( !v162 && v91 == -8192 )
                    v162 = v90;
                  v89 = v87 & (v196 + v89);
                  v90 = (_QWORD *)(v88 + 16LL * v89);
                  v91 = *v90;
                  if ( v85 == *v90 )
                    goto LABEL_66;
                  ++v196;
                }
                if ( v162 )
                  v90 = v162;
              }
LABEL_66:
              *v90 = v85;
              v90[1] = v84[1];
              ++*(_DWORD *)(a4 + 112);
            }
            v84 += 2;
          }
          while ( v82 != v84 );
        }
        v192 = v76;
        sub_C7D6A0(v184, v81, 8);
        v92 = *(_QWORD **)(a4 + 104);
        v93 = *(_DWORD *)(a4 + 120);
        v76 = v192;
        v94 = *(_DWORD *)(a4 + 112) + 1;
      }
      else
      {
        v159 = *(unsigned int *)(a4 + 120);
        *(_QWORD *)(a4 + 112) = 0;
        v93 = v159;
        v92 = &v79[2 * v159];
        if ( v79 == v92 )
        {
          v94 = 1;
        }
        else
        {
          do
          {
            if ( v79 )
              *v79 = -4096;
            v79 += 2;
          }
          while ( v92 != v79 );
          v92 = *(_QWORD **)(a4 + 104);
          v93 = *(_DWORD *)(a4 + 120);
          v94 = *(_DWORD *)(a4 + 112) + 1;
        }
      }
      if ( !v93 )
        goto LABEL_313;
      v95 = v93 - 1;
      v96 = v95 & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
      v47 = &v92[2 * v96];
      v97 = *v47;
      if ( v76 != *v47 )
      {
        v168 = 1;
        v169 = 0;
        while ( v97 != -4096 )
        {
          if ( v97 == -8192 && !v169 )
            v169 = v47;
          v96 = v95 & (v168 + v96);
          v47 = &v92[2 * v96];
          v97 = *v47;
          if ( v76 == *v47 )
            goto LABEL_71;
          ++v168;
        }
        if ( v169 )
          v47 = v169;
      }
    }
    else if ( (int)v13 - (v94 + *(_DWORD *)(a4 + 116)) <= (unsigned int)v13 >> 3 )
    {
      v185 = (__int64 *)v15;
      v100 = ((((((((v41 >> 1) | v41 | (((v41 >> 1) | v41) >> 2)) >> 4) | (v41 >> 1) | v41 | (((v41 >> 1) | v41) >> 2)) >> 8)
              | (((v41 >> 1) | v41 | (((v41 >> 1) | v41) >> 2)) >> 4)
              | (v41 >> 1)
              | v41
              | (((v41 >> 1) | v41) >> 2)) >> 16)
            | (((((v41 >> 1) | v41 | (((v41 >> 1) | v41) >> 2)) >> 4) | (v41 >> 1) | v41 | (((v41 >> 1) | v41) >> 2)) >> 8)
            | (((v41 >> 1) | v41 | (((v41 >> 1) | v41) >> 2)) >> 4)
            | (v41 >> 1)
            | v41
            | (((v41 >> 1) | v41) >> 2))
           + 1;
      if ( v100 < 0x40 )
        v100 = 64;
      *(_DWORD *)(a4 + 120) = v100;
      v101 = (_QWORD *)sub_C7D670(16LL * v100, 8);
      v102 = (__int64)v185;
      v76 = v52;
      *(_QWORD *)(a4 + 104) = v101;
      if ( v185 )
      {
        *(_QWORD *)(a4 + 112) = 0;
        v103 = 2LL * (unsigned int)v13;
        v104 = &v185[v103];
        for ( n = &v101[2 * *(unsigned int *)(a4 + 120)]; n != v101; v101 += 2 )
        {
          if ( v101 )
            *v101 = -4096;
        }
        v106 = v185;
        do
        {
          v63 = *v106;
          if ( *v106 != -8192 && v63 != -4096 )
          {
            v107 = *(_DWORD *)(a4 + 120);
            if ( !v107 )
              goto LABEL_316;
            v108 = v107 - 1;
            v109 = *(_QWORD *)(a4 + 104);
            v110 = v108 & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
            v111 = (_QWORD *)(v109 + 16LL * v110);
            v112 = *v111;
            if ( *v111 != v63 )
            {
              v189 = 1;
              v198 = 0;
              while ( v112 != -4096 )
              {
                if ( !v198 )
                {
                  if ( v112 != -8192 )
                    v111 = 0;
                  v198 = v111;
                }
                v110 = v108 & (v189 + v110);
                v111 = (_QWORD *)(v109 + 16LL * v110);
                v112 = *v111;
                if ( v63 == *v111 )
                  goto LABEL_98;
                ++v189;
              }
              if ( v198 )
                v111 = v198;
            }
LABEL_98:
            *v111 = v63;
            v111[1] = v106[1];
            ++*(_DWORD *)(a4 + 112);
          }
          v106 += 2;
        }
        while ( v104 != v106 );
        v193 = v76;
        sub_C7D6A0(v102, v103 * 8, 8);
        v113 = *(_QWORD **)(a4 + 104);
        v114 = *(_DWORD *)(a4 + 120);
        v76 = v193;
        v94 = *(_DWORD *)(a4 + 112) + 1;
      }
      else
      {
        v163 = *(unsigned int *)(a4 + 120);
        *(_QWORD *)(a4 + 112) = 0;
        v114 = v163;
        v113 = &v101[2 * v163];
        if ( v101 == v113 )
        {
          v94 = 1;
        }
        else
        {
          do
          {
            if ( v101 )
              *v101 = -4096;
            v101 += 2;
          }
          while ( v113 != v101 );
          v113 = *(_QWORD **)(a4 + 104);
          v114 = *(_DWORD *)(a4 + 120);
          v94 = *(_DWORD *)(a4 + 112) + 1;
        }
      }
      if ( v114 )
      {
        v115 = v114 - 1;
        v116 = 1;
        v117 = 0;
        v118 = v115 & v48;
        v47 = &v113[2 * v118];
        v119 = *v47;
        if ( v76 != *v47 )
        {
          while ( v119 != -4096 )
          {
            if ( v119 == -8192 && !v117 )
              v117 = v47;
            v118 = v115 & (v116 + v118);
            v47 = &v113[2 * v118];
            v119 = *v47;
            if ( v76 == *v47 )
              goto LABEL_71;
            ++v116;
          }
          if ( v117 )
            v47 = v117;
        }
        goto LABEL_71;
      }
LABEL_313:
      ++*(_DWORD *)(a4 + 112);
      BUG();
    }
LABEL_71:
    *(_DWORD *)(a4 + 112) = v94;
    if ( *v47 != -4096 )
      --*(_DWORD *)(a4 + 116);
    *v47 = v76;
    v47[1] = 0;
LABEL_74:
    if ( v200 < v12 )
    {
      v98 = *(_QWORD *)v200;
      *(_QWORD *)v200 = *(_QWORD *)v12;
      *(_QWORD *)v12 = v98;
      v13 = *(unsigned int *)(a4 + 120);
      goto LABEL_12;
    }
    sub_30C0440(v200, v183, v181, a4);
    result = v200 - (char *)a1;
    if ( v200 - (char *)a1 > 128 )
    {
      if ( v181 )
      {
        v183 = v200;
        continue;
      }
LABEL_218:
      v164 = result >> 3;
      v165 = ((result >> 3) - 2) >> 1;
      sub_30BFDF0((__int64)a1, v165, result >> 3, a1[v165], a4);
      do
      {
        --v165;
        sub_30BFDF0((__int64)a1, v165, v164, a1[v165], a4);
      }
      while ( v165 );
      v166 = v199;
      do
      {
        v167 = *--v166;
        *v166 = *a1;
        result = sub_30BFDF0((__int64)a1, 0, v166 - a1, v167, a4);
      }
      while ( (char *)v166 - (char *)a1 > 8 );
    }
    return result;
  }
}
