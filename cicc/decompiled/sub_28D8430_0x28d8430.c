// Function: sub_28D8430
// Address: 0x28d8430
//
__int64 __fastcall sub_28D8430(__int64 *a1, char *a2, __int64 a3, __int64 a4)
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
  __int64 v19; // rax
  __int64 v20; // rsi
  unsigned int v21; // r8d
  int v22; // r10d
  __int64 *v23; // rdx
  unsigned int v24; // edi
  __int64 v25; // rax
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
  unsigned __int64 v39; // rdi
  int v40; // r8d
  _QWORD *v41; // rdx
  unsigned int v42; // esi
  __int64 v43; // rax
  __int64 v44; // rcx
  unsigned int v45; // ecx
  int v46; // r10d
  _QWORD *v47; // rdx
  unsigned int v48; // ebx
  unsigned int v49; // r8d
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // r12
  __int64 v53; // rbx
  unsigned int v54; // eax
  _QWORD *v55; // rax
  __int64 v56; // r9
  __int64 v57; // rdx
  __int64 v58; // r11
  __int64 v59; // r13
  _QWORD *j; // rdx
  __int64 k; // rax
  __int64 v62; // r10
  int v63; // esi
  int v64; // esi
  __int64 v65; // r8
  unsigned int v66; // ecx
  __int64 *v67; // rdx
  __int64 v68; // rdi
  _QWORD *v69; // rdi
  int v70; // edx
  int v71; // ecx
  int v72; // esi
  unsigned int v73; // eax
  __int64 v74; // r8
  __int64 v75; // r11
  unsigned int v76; // eax
  _QWORD *v77; // rax
  __int64 v78; // rdx
  __int64 v79; // r13
  __int64 v80; // r8
  _QWORD *m; // rdx
  __int64 n; // rax
  __int64 v83; // rdx
  int v84; // edi
  int v85; // edi
  __int64 v86; // rbx
  unsigned int v87; // esi
  __int64 *v88; // rcx
  __int64 v89; // r10
  _QWORD *v90; // rsi
  int v91; // ecx
  int v92; // eax
  int v93; // ecx
  unsigned int v94; // edi
  __int64 v95; // r8
  __int64 v96; // rax
  int v97; // eax
  unsigned int v98; // eax
  _QWORD *v99; // rax
  __int64 v100; // r9
  __int64 v101; // r13
  __int64 v102; // rdi
  _QWORD *ii; // rdx
  __int64 v104; // rax
  __int64 v105; // r10
  int v106; // esi
  int v107; // esi
  __int64 v108; // r12
  unsigned int v109; // ecx
  __int64 *v110; // rdx
  __int64 v111; // r8
  _QWORD *v112; // rsi
  int v113; // ecx
  int v114; // ecx
  int v115; // r9d
  _QWORD *v116; // r8
  unsigned int v117; // ebx
  __int64 v118; // rdi
  int v119; // eax
  unsigned int v120; // eax
  _QWORD *v121; // rax
  __int64 v122; // r9
  unsigned int v123; // r11d
  __int64 v124; // rdx
  __int64 v125; // r13
  _QWORD *i; // rdx
  __int64 v127; // rax
  __int64 v128; // rdx
  int v129; // edi
  int v130; // edi
  __int64 v131; // r10
  unsigned int v132; // esi
  __int64 *v133; // rcx
  __int64 v134; // r8
  _QWORD *v135; // rsi
  int v136; // edx
  int v137; // eax
  int v138; // r10d
  _QWORD *v139; // r9
  unsigned int v140; // r8d
  __int64 v141; // rdi
  int v142; // eax
  int v143; // ecx
  int v144; // ecx
  __int64 v145; // rdi
  __int64 *v146; // r9
  unsigned int v147; // ebx
  int v148; // r10d
  __int64 v149; // rsi
  int v150; // eax
  int v151; // ecx
  int v152; // ecx
  int v153; // r10d
  __int64 *v154; // r9
  __int64 v155; // rdi
  __int64 v156; // r11
  __int64 v157; // rsi
  __int64 v158; // rsi
  __int64 v159; // rdi
  __int64 v160; // rsi
  __int64 *v161; // r12
  __int64 v162; // rsi
  __int64 v163; // rbx
  __int64 v164; // r12
  __int64 *v165; // rbx
  __int64 v166; // rcx
  int v167; // ebx
  _QWORD *v168; // r9
  int v169; // r11d
  char *v170; // r15
  __int64 *v171; // rdi
  __int64 v172; // rax
  __int64 v173; // rax
  int v174; // r11d
  __int64 *v175; // r10
  int v176; // r11d
  __int64 *v177; // r10
  int v178; // r9d
  unsigned int v179; // ebx
  _QWORD *v180; // r10
  int v181; // [rsp+Ch] [rbp-94h]
  char *v182; // [rsp+10h] [rbp-90h]
  __int64 v183; // [rsp+20h] [rbp-80h]
  __int64 v184; // [rsp+28h] [rbp-78h]
  char *v185; // [rsp+30h] [rbp-70h]
  __int64 v186; // [rsp+38h] [rbp-68h]
  __int64 v187; // [rsp+38h] [rbp-68h]
  __int64 v188; // [rsp+38h] [rbp-68h]
  int v189; // [rsp+38h] [rbp-68h]
  __int64 *v190; // [rsp+38h] [rbp-68h]
  int v191; // [rsp+38h] [rbp-68h]
  __int64 v192; // [rsp+40h] [rbp-60h]
  __int64 v193; // [rsp+40h] [rbp-60h]
  __int64 v194; // [rsp+40h] [rbp-60h]
  __int64 v195; // [rsp+40h] [rbp-60h]
  __int64 v196; // [rsp+40h] [rbp-60h]
  unsigned int v197; // [rsp+40h] [rbp-60h]
  int v198; // [rsp+40h] [rbp-60h]
  __int64 *v199; // [rsp+40h] [rbp-60h]
  __int64 *v200; // [rsp+40h] [rbp-60h]
  __int64 *v201; // [rsp+48h] [rbp-58h]
  char *v202; // [rsp+50h] [rbp-50h]
  __int64 v204[7]; // [rsp+68h] [rbp-38h] BYREF

  result = a2 - (char *)a1;
  v185 = a2;
  v183 = a3;
  if ( a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v201 = (__int64 *)a2;
    goto LABEL_218;
  }
  v182 = (char *)(a1 + 1);
  v184 = a4 + 1360;
  while ( 2 )
  {
    v204[0] = a4;
    --v183;
    v6 = &a1[result >> 4];
    v7 = !sub_28D8160(v204, a1[1], *v6);
    v8 = *((_QWORD *)v185 - 1);
    if ( v7 )
    {
      if ( sub_28D8160(v204, a1[1], v8) )
      {
        v11 = *a1;
        v10 = a1[1];
        a1[1] = *a1;
        *a1 = v10;
        goto LABEL_7;
      }
      v170 = v185;
      v171 = a1;
      if ( !sub_28D8160(v204, *v6, *((_QWORD *)v185 - 1)) )
      {
        v173 = *a1;
        *a1 = *v6;
        *v6 = v173;
        v10 = *a1;
        v11 = a1[1];
        goto LABEL_7;
      }
LABEL_237:
      v172 = *v171;
      *v171 = *((_QWORD *)v170 - 1);
      *((_QWORD *)v170 - 1) = v172;
      v10 = *v171;
      v11 = v171[1];
      goto LABEL_7;
    }
    if ( !sub_28D8160(v204, *v6, v8) )
    {
      v170 = v185;
      v171 = a1;
      if ( !sub_28D8160(v204, a1[1], *((_QWORD *)v185 - 1)) )
      {
        v11 = *a1;
        v10 = a1[1];
        a1[1] = *a1;
        *a1 = v10;
        goto LABEL_7;
      }
      goto LABEL_237;
    }
    v9 = *a1;
    *a1 = *v6;
    *v6 = v9;
    v10 = *a1;
    v11 = a1[1];
LABEL_7:
    v202 = v182;
    v12 = v185;
    v13 = *(unsigned int *)(a4 + 1384);
    while ( 1 )
    {
      v201 = (__int64 *)v202;
      if ( (_DWORD)v13 )
      {
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a4 + 1368);
        v16 = 1;
        v17 = 0;
        v18 = ((_DWORD)v13 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v19 = v15 + 16 * v18;
        v20 = *(_QWORD *)v19;
        if ( v11 == *(_QWORD *)v19 )
        {
LABEL_9:
          v21 = *(_DWORD *)(v19 + 8);
          goto LABEL_10;
        }
        while ( v20 != -4096 )
        {
          if ( !v17 && v20 == -8192 )
            v17 = (__int64 *)v19;
          LODWORD(v18) = v14 & (v16 + v18);
          v19 = v15 + 16LL * (unsigned int)v18;
          v20 = *(_QWORD *)v19;
          if ( v11 == *(_QWORD *)v19 )
            goto LABEL_9;
          ++v16;
        }
        if ( !v17 )
          v17 = (__int64 *)v19;
        v150 = *(_DWORD *)(a4 + 1376);
        ++*(_QWORD *)(a4 + 1360);
        v31 = v150 + 1;
        if ( 4 * v31 < (unsigned int)(3 * v13) )
        {
          if ( (int)v13 - *(_DWORD *)(a4 + 1380) - v31 <= (unsigned int)v13 >> 3 )
          {
            v197 = ((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4);
            sub_CE3370(v184, v13);
            v151 = *(_DWORD *)(a4 + 1384);
            if ( !v151 )
            {
LABEL_319:
              ++*(_DWORD *)(a4 + 1376);
              BUG();
            }
            v152 = v151 - 1;
            v153 = 1;
            v154 = 0;
            v155 = *(_QWORD *)(a4 + 1368);
            LODWORD(v156) = v152 & v197;
            v31 = *(_DWORD *)(a4 + 1376) + 1;
            v17 = (__int64 *)(v155 + 16LL * (v152 & v197));
            v157 = *v17;
            if ( v11 != *v17 )
            {
              while ( v157 != -4096 )
              {
                if ( v157 == -8192 && !v154 )
                  v154 = v17;
                v156 = v152 & (unsigned int)(v156 + v153);
                v17 = (__int64 *)(v155 + 16 * v156);
                v157 = *v17;
                if ( v11 == *v17 )
                  goto LABEL_17;
                ++v153;
              }
              if ( v154 )
                v17 = v154;
            }
          }
          goto LABEL_17;
        }
      }
      else
      {
        ++*(_QWORD *)(a4 + 1360);
      }
      sub_CE3370(v184, 2 * v13);
      v27 = *(_DWORD *)(a4 + 1384);
      if ( !v27 )
        goto LABEL_319;
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a4 + 1368);
      LODWORD(v30) = v28 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v31 = *(_DWORD *)(a4 + 1376) + 1;
      v17 = (__int64 *)(v29 + 16LL * (unsigned int)v30);
      v32 = *v17;
      if ( v11 != *v17 )
      {
        v176 = 1;
        v177 = 0;
        while ( v32 != -4096 )
        {
          if ( !v177 && v32 == -8192 )
            v177 = v17;
          v30 = v28 & (unsigned int)(v30 + v176);
          v17 = (__int64 *)(v29 + 16 * v30);
          v32 = *v17;
          if ( v11 == *v17 )
            goto LABEL_17;
          ++v176;
        }
        if ( v177 )
          v17 = v177;
      }
LABEL_17:
      *(_DWORD *)(a4 + 1376) = v31;
      if ( *v17 != -4096 )
        --*(_DWORD *)(a4 + 1380);
      *v17 = v11;
      *((_DWORD *)v17 + 2) = 0;
      v13 = *(unsigned int *)(a4 + 1384);
      if ( !(_DWORD)v13 )
      {
        ++*(_QWORD *)(a4 + 1360);
        goto LABEL_21;
      }
      v15 = *(_QWORD *)(a4 + 1368);
      v14 = v13 - 1;
      v21 = 0;
LABEL_10:
      v22 = 1;
      v23 = 0;
      v24 = v14 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v25 = v15 + 16LL * v24;
      v26 = *(_QWORD *)v25;
      if ( v10 != *(_QWORD *)v25 )
        break;
LABEL_11:
      if ( v21 >= *(_DWORD *)(v25 + 8) )
        goto LABEL_30;
LABEL_12:
      v10 = *a1;
      v11 = *((_QWORD *)v202 + 1);
      v202 += 8;
    }
    while ( v26 != -4096 )
    {
      if ( v26 == -8192 && !v23 )
        v23 = (__int64 *)v25;
      v24 = v14 & (v22 + v24);
      v25 = v15 + 16LL * v24;
      v26 = *(_QWORD *)v25;
      if ( *(_QWORD *)v25 == v10 )
        goto LABEL_11;
      ++v22;
    }
    if ( !v23 )
      v23 = (__int64 *)v25;
    v142 = *(_DWORD *)(a4 + 1376);
    ++*(_QWORD *)(a4 + 1360);
    v37 = v142 + 1;
    if ( 4 * v37 >= (unsigned int)(3 * v13) )
    {
LABEL_21:
      sub_CE3370(v184, 2 * v13);
      v33 = *(_DWORD *)(a4 + 1384);
      if ( v33 )
      {
        v34 = v33 - 1;
        v35 = *(_QWORD *)(a4 + 1368);
        v36 = v34 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v37 = *(_DWORD *)(a4 + 1376) + 1;
        v23 = (__int64 *)(v35 + 16LL * v36);
        v38 = *v23;
        if ( v10 != *v23 )
        {
          v174 = 1;
          v175 = 0;
          while ( v38 != -4096 )
          {
            if ( !v175 && v38 == -8192 )
              v175 = v23;
            v36 = v34 & (v174 + v36);
            v23 = (__int64 *)(v35 + 16LL * v36);
            v38 = *v23;
            if ( *v23 == v10 )
              goto LABEL_23;
            ++v174;
          }
          if ( v175 )
            v23 = v175;
        }
        goto LABEL_23;
      }
LABEL_316:
      ++*(_DWORD *)(a4 + 1376);
      BUG();
    }
    if ( (int)v13 - (v37 + *(_DWORD *)(a4 + 1380)) > (unsigned int)v13 >> 3 )
      goto LABEL_23;
    sub_CE3370(v184, v13);
    v143 = *(_DWORD *)(a4 + 1384);
    if ( !v143 )
      goto LABEL_316;
    v144 = v143 - 1;
    v145 = *(_QWORD *)(a4 + 1368);
    v146 = 0;
    v147 = v144 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
    v148 = 1;
    v37 = *(_DWORD *)(a4 + 1376) + 1;
    v23 = (__int64 *)(v145 + 16LL * v147);
    v149 = *v23;
    if ( v10 != *v23 )
    {
      while ( v149 != -4096 )
      {
        if ( v149 == -8192 && !v146 )
          v146 = v23;
        v147 = v144 & (v148 + v147);
        v23 = (__int64 *)(v145 + 16LL * v147);
        v149 = *v23;
        if ( *v23 == v10 )
          goto LABEL_23;
        ++v148;
      }
      if ( v146 )
        v23 = v146;
    }
LABEL_23:
    *(_DWORD *)(a4 + 1376) = v37;
    if ( *v23 != -4096 )
      --*(_DWORD *)(a4 + 1380);
    *v23 = v10;
    *((_DWORD *)v23 + 2) = 0;
    v13 = *(unsigned int *)(a4 + 1384);
    v15 = *(_QWORD *)(a4 + 1368);
LABEL_30:
    while ( 2 )
    {
      v12 -= 8;
      v52 = *(_QWORD *)v12;
      v53 = *a1;
      if ( !(_DWORD)v13 )
      {
        ++*(_QWORD *)(a4 + 1360);
        goto LABEL_32;
      }
      v39 = (unsigned int)(v13 - 1);
      v40 = 1;
      v41 = 0;
      v42 = v39 & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
      v43 = v15 + 16LL * v42;
      v44 = *(_QWORD *)v43;
      if ( v53 == *(_QWORD *)v43 )
      {
LABEL_27:
        v45 = *(_DWORD *)(v43 + 8);
        goto LABEL_28;
      }
      while ( v44 != -4096 )
      {
        if ( v44 == -8192 && !v41 )
          v41 = (_QWORD *)v43;
        v42 = v39 & (v40 + v42);
        v43 = v15 + 16LL * v42;
        v44 = *(_QWORD *)v43;
        if ( v53 == *(_QWORD *)v43 )
          goto LABEL_27;
        ++v40;
      }
      if ( !v41 )
        v41 = (_QWORD *)v43;
      v119 = *(_DWORD *)(a4 + 1376);
      ++*(_QWORD *)(a4 + 1360);
      v71 = v119 + 1;
      if ( 4 * (v119 + 1) < (unsigned int)(3 * v13) )
      {
        if ( (int)v13 - *(_DWORD *)(a4 + 1380) - v71 <= (unsigned int)v13 >> 3 )
        {
          v188 = v15;
          v120 = sub_AF1560(v39);
          if ( v120 < 0x40 )
            v120 = 64;
          *(_DWORD *)(a4 + 1384) = v120;
          v121 = (_QWORD *)sub_C7D670(16LL * v120, 8);
          v122 = v188;
          v123 = ((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4);
          *(_QWORD *)(a4 + 1368) = v121;
          if ( v188 )
          {
            v124 = *(unsigned int *)(a4 + 1384);
            *(_QWORD *)(a4 + 1376) = 0;
            v196 = 16LL * (unsigned int)v13;
            v125 = v188 + v196;
            for ( i = &v121[2 * v124]; i != v121; v121 += 2 )
            {
              if ( v121 )
                *v121 = -4096;
            }
            v127 = v188;
            do
            {
              v128 = *(_QWORD *)v127;
              if ( *(_QWORD *)v127 != -8192 && v128 != -4096 )
              {
                v129 = *(_DWORD *)(a4 + 1384);
                if ( !v129 )
                {
                  MEMORY[0] = *(_QWORD *)v127;
                  BUG();
                }
                v130 = v129 - 1;
                v131 = *(_QWORD *)(a4 + 1368);
                v132 = v130 & (((unsigned int)v128 >> 9) ^ ((unsigned int)v128 >> 4));
                v133 = (__int64 *)(v131 + 16LL * v132);
                v134 = *v133;
                if ( v128 != *v133 )
                {
                  v181 = 1;
                  v190 = 0;
                  while ( v134 != -4096 )
                  {
                    if ( !v190 )
                    {
                      if ( v134 != -8192 )
                        v133 = 0;
                      v190 = v133;
                    }
                    v132 = v130 & (v181 + v132);
                    v133 = (__int64 *)(v131 + 16LL * v132);
                    v134 = *v133;
                    if ( v128 == *v133 )
                      goto LABEL_129;
                    ++v181;
                  }
                  if ( v190 )
                    v133 = v190;
                }
LABEL_129:
                *v133 = v128;
                *((_DWORD *)v133 + 2) = *(_DWORD *)(v127 + 8);
                ++*(_DWORD *)(a4 + 1376);
              }
              v127 += 16;
            }
            while ( v125 != v127 );
            sub_C7D6A0(v122, v196, 8);
            v135 = *(_QWORD **)(a4 + 1368);
            v136 = *(_DWORD *)(a4 + 1384);
            v123 = ((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4);
            v71 = *(_DWORD *)(a4 + 1376) + 1;
          }
          else
          {
            v160 = *(unsigned int *)(a4 + 1384);
            *(_QWORD *)(a4 + 1376) = 0;
            v136 = v160;
            v135 = &v121[2 * v160];
            if ( v121 == v135 )
            {
              v71 = 1;
            }
            else
            {
              do
              {
                if ( v121 )
                  *v121 = -4096;
                v121 += 2;
              }
              while ( v135 != v121 );
              v135 = *(_QWORD **)(a4 + 1368);
              v136 = *(_DWORD *)(a4 + 1384);
              v71 = *(_DWORD *)(a4 + 1376) + 1;
            }
          }
          if ( !v136 )
          {
LABEL_320:
            ++*(_DWORD *)(a4 + 1376);
            BUG();
          }
          v137 = v136 - 1;
          v138 = 1;
          v139 = 0;
          v140 = (v136 - 1) & v123;
          v41 = &v135[2 * v140];
          v141 = *v41;
          if ( v53 != *v41 )
          {
            while ( v141 != -4096 )
            {
              if ( !v139 && v141 == -8192 )
                v139 = v41;
              v140 = v137 & (v138 + v140);
              v41 = &v135[2 * v140];
              v141 = *v41;
              if ( v53 == *v41 )
                goto LABEL_50;
              ++v138;
            }
LABEL_135:
            if ( v139 )
              v41 = v139;
          }
        }
        goto LABEL_50;
      }
LABEL_32:
      v192 = v15;
      v54 = sub_AF1560((unsigned int)(2 * v13 - 1));
      if ( v54 < 0x40 )
        v54 = 64;
      *(_DWORD *)(a4 + 1384) = v54;
      v55 = (_QWORD *)sub_C7D670(16LL * v54, 8);
      v56 = v192;
      *(_QWORD *)(a4 + 1368) = v55;
      if ( v192 )
      {
        v57 = *(unsigned int *)(a4 + 1384);
        *(_QWORD *)(a4 + 1376) = 0;
        v58 = 16LL * (unsigned int)v13;
        v59 = v192 + v58;
        for ( j = &v55[2 * v57]; j != v55; v55 += 2 )
        {
          if ( v55 )
            *v55 = -4096;
        }
        for ( k = v192; v59 != k; k += 16 )
        {
          v62 = *(_QWORD *)k;
          if ( *(_QWORD *)k != -4096 && v62 != -8192 )
          {
            v63 = *(_DWORD *)(a4 + 1384);
            if ( !v63 )
            {
              MEMORY[0] = *(_QWORD *)k;
              BUG();
            }
            v64 = v63 - 1;
            v65 = *(_QWORD *)(a4 + 1368);
            v66 = v64 & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
            v67 = (__int64 *)(v65 + 16LL * v66);
            v68 = *v67;
            if ( v62 != *v67 )
            {
              v189 = 1;
              v199 = 0;
              while ( v68 != -4096 )
              {
                if ( !v199 )
                {
                  if ( v68 != -8192 )
                    v67 = 0;
                  v199 = v67;
                }
                v66 = v64 & (v189 + v66);
                v67 = (__int64 *)(v65 + 16LL * v66);
                v68 = *v67;
                if ( v62 == *v67 )
                  goto LABEL_44;
                ++v189;
              }
              if ( v199 )
                v67 = v199;
            }
LABEL_44:
            *v67 = v62;
            *((_DWORD *)v67 + 2) = *(_DWORD *)(k + 8);
            ++*(_DWORD *)(a4 + 1376);
          }
        }
        sub_C7D6A0(v56, v58, 8);
LABEL_47:
        v69 = *(_QWORD **)(a4 + 1368);
        v70 = *(_DWORD *)(a4 + 1384);
        v71 = *(_DWORD *)(a4 + 1376) + 1;
      }
      else
      {
        v159 = *(unsigned int *)(a4 + 1384);
        *(_QWORD *)(a4 + 1376) = 0;
        v70 = v159;
        v69 = &v55[2 * v159];
        if ( v55 != v69 )
        {
          do
          {
            if ( v55 )
              *v55 = -4096;
            v55 += 2;
          }
          while ( v69 != v55 );
          goto LABEL_47;
        }
        v71 = 1;
      }
      if ( !v70 )
        goto LABEL_320;
      v72 = v70 - 1;
      v73 = (v70 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
      v41 = &v69[2 * v73];
      v74 = *v41;
      if ( v53 != *v41 )
      {
        v169 = 1;
        v139 = 0;
        while ( v74 != -4096 )
        {
          if ( v74 != -8192 || v139 )
            v41 = v139;
          v73 = v72 & (v169 + v73);
          v74 = v69[2 * v73];
          if ( v53 == v74 )
          {
            v41 = &v69[2 * v73];
            goto LABEL_50;
          }
          ++v169;
          v139 = v41;
          v41 = &v69[2 * v73];
        }
        goto LABEL_135;
      }
LABEL_50:
      *(_DWORD *)(a4 + 1376) = v71;
      if ( *v41 != -4096 )
        --*(_DWORD *)(a4 + 1380);
      *v41 = v53;
      *((_DWORD *)v41 + 2) = 0;
      v13 = *(unsigned int *)(a4 + 1384);
      v15 = *(_QWORD *)(a4 + 1368);
      if ( !(_DWORD)v13 )
      {
        ++*(_QWORD *)(a4 + 1360);
        v75 = v52;
        goto LABEL_54;
      }
      v39 = (unsigned int)(v13 - 1);
      v45 = 0;
LABEL_28:
      v46 = 1;
      v47 = 0;
      v48 = ((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4);
      v49 = v39 & v48;
      v50 = v15 + 16LL * ((unsigned int)v39 & v48);
      v51 = *(_QWORD *)v50;
      if ( v52 == *(_QWORD *)v50 )
      {
LABEL_29:
        if ( v45 >= *(_DWORD *)(v50 + 8) )
          goto LABEL_74;
        continue;
      }
      break;
    }
    while ( v51 != -4096 )
    {
      if ( !v47 && v51 == -8192 )
        v47 = (_QWORD *)v50;
      v49 = v39 & (v46 + v49);
      v50 = v15 + 16LL * v49;
      v51 = *(_QWORD *)v50;
      if ( v52 == *(_QWORD *)v50 )
        goto LABEL_29;
      ++v46;
    }
    v75 = v52;
    if ( !v47 )
      v47 = (_QWORD *)v50;
    v97 = *(_DWORD *)(a4 + 1376);
    ++*(_QWORD *)(a4 + 1360);
    v92 = v97 + 1;
    if ( 4 * v92 >= (unsigned int)(3 * v13) )
    {
LABEL_54:
      v186 = v15;
      v193 = v75;
      v76 = sub_AF1560((unsigned int)(2 * v13 - 1));
      if ( v76 < 0x40 )
        v76 = 64;
      *(_DWORD *)(a4 + 1384) = v76;
      v77 = (_QWORD *)sub_C7D670(16LL * v76, 8);
      v75 = v193;
      *(_QWORD *)(a4 + 1368) = v77;
      if ( v186 )
      {
        v78 = *(unsigned int *)(a4 + 1384);
        v79 = 16 * v13;
        *(_QWORD *)(a4 + 1376) = 0;
        v80 = v186 + v79;
        for ( m = &v77[2 * v78]; m != v77; v77 += 2 )
        {
          if ( v77 )
            *v77 = -4096;
        }
        for ( n = v186; v80 != n; n += 16 )
        {
          v83 = *(_QWORD *)n;
          if ( *(_QWORD *)n != -4096 && v83 != -8192 )
          {
            v84 = *(_DWORD *)(a4 + 1384);
            if ( !v84 )
            {
              MEMORY[0] = *(_QWORD *)n;
              BUG();
            }
            v85 = v84 - 1;
            v86 = *(_QWORD *)(a4 + 1368);
            v87 = v85 & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
            v88 = (__int64 *)(v86 + 16LL * v87);
            v89 = *v88;
            if ( *v88 != v83 )
            {
              v198 = 1;
              v161 = 0;
              while ( v89 != -4096 )
              {
                if ( !v161 && v89 == -8192 )
                  v161 = v88;
                v87 = v85 & (v198 + v87);
                v88 = (__int64 *)(v86 + 16LL * v87);
                v89 = *v88;
                if ( v83 == *v88 )
                  goto LABEL_66;
                ++v198;
              }
              if ( v161 )
                v88 = v161;
            }
LABEL_66:
            *v88 = v83;
            *((_DWORD *)v88 + 2) = *(_DWORD *)(n + 8);
            ++*(_DWORD *)(a4 + 1376);
          }
        }
        v194 = v75;
        sub_C7D6A0(v186, v79, 8);
        v90 = *(_QWORD **)(a4 + 1368);
        v91 = *(_DWORD *)(a4 + 1384);
        v75 = v194;
        v92 = *(_DWORD *)(a4 + 1376) + 1;
      }
      else
      {
        v158 = *(unsigned int *)(a4 + 1384);
        *(_QWORD *)(a4 + 1376) = 0;
        v91 = v158;
        v90 = &v77[2 * v158];
        if ( v77 == v90 )
        {
          v92 = 1;
        }
        else
        {
          do
          {
            if ( v77 )
              *v77 = -4096;
            v77 += 2;
          }
          while ( v90 != v77 );
          v90 = *(_QWORD **)(a4 + 1368);
          v91 = *(_DWORD *)(a4 + 1384);
          v92 = *(_DWORD *)(a4 + 1376) + 1;
        }
      }
      if ( !v91 )
        goto LABEL_314;
      v93 = v91 - 1;
      v94 = v93 & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
      v47 = &v90[2 * v94];
      v95 = *v47;
      if ( v75 != *v47 )
      {
        v167 = 1;
        v168 = 0;
        while ( v95 != -4096 )
        {
          if ( v95 != -8192 || v168 )
            v47 = v168;
          v178 = v167 + 1;
          v179 = v94 + v167;
          v94 = v93 & v179;
          v180 = &v90[2 * (v93 & v179)];
          v95 = *v180;
          if ( v75 == *v180 )
          {
            v47 = &v90[2 * (v93 & v179)];
            goto LABEL_71;
          }
          v167 = v178;
          v168 = v47;
          v47 = v180;
        }
        if ( v168 )
          v47 = v168;
      }
    }
    else if ( (int)v13 - (v92 + *(_DWORD *)(a4 + 1380)) <= (unsigned int)v13 >> 3 )
    {
      v187 = v15;
      v98 = sub_AF1560(v39);
      if ( v98 < 0x40 )
        v98 = 64;
      *(_DWORD *)(a4 + 1384) = v98;
      v99 = (_QWORD *)sub_C7D670(16LL * v98, 8);
      v100 = v187;
      v75 = v52;
      *(_QWORD *)(a4 + 1368) = v99;
      if ( v187 )
      {
        *(_QWORD *)(a4 + 1376) = 0;
        v101 = 16LL * (unsigned int)v13;
        v102 = v187 + v101;
        for ( ii = &v99[2 * *(unsigned int *)(a4 + 1384)]; ii != v99; v99 += 2 )
        {
          if ( v99 )
            *v99 = -4096;
        }
        v104 = v187;
        do
        {
          v105 = *(_QWORD *)v104;
          if ( *(_QWORD *)v104 != -8192 && v105 != -4096 )
          {
            v106 = *(_DWORD *)(a4 + 1384);
            if ( !v106 )
            {
              MEMORY[0] = *(_QWORD *)v104;
              BUG();
            }
            v107 = v106 - 1;
            v108 = *(_QWORD *)(a4 + 1368);
            v109 = v107 & (((unsigned int)v105 >> 9) ^ ((unsigned int)v105 >> 4));
            v110 = (__int64 *)(v108 + 16LL * v109);
            v111 = *v110;
            if ( *v110 != v105 )
            {
              v191 = 1;
              v200 = 0;
              while ( v111 != -4096 )
              {
                if ( !v200 )
                {
                  if ( v111 != -8192 )
                    v110 = 0;
                  v200 = v110;
                }
                v109 = v107 & (v191 + v109);
                v110 = (__int64 *)(v108 + 16LL * v109);
                v111 = *v110;
                if ( v105 == *v110 )
                  goto LABEL_98;
                ++v191;
              }
              if ( v200 )
                v110 = v200;
            }
LABEL_98:
            *v110 = v105;
            *((_DWORD *)v110 + 2) = *(_DWORD *)(v104 + 8);
            ++*(_DWORD *)(a4 + 1376);
          }
          v104 += 16;
        }
        while ( v102 != v104 );
        v195 = v75;
        sub_C7D6A0(v100, v101, 8);
        v112 = *(_QWORD **)(a4 + 1368);
        v113 = *(_DWORD *)(a4 + 1384);
        v75 = v195;
        v92 = *(_DWORD *)(a4 + 1376) + 1;
      }
      else
      {
        v162 = *(unsigned int *)(a4 + 1384);
        *(_QWORD *)(a4 + 1376) = 0;
        v113 = v162;
        v112 = &v99[2 * v162];
        if ( v99 == v112 )
        {
          v92 = 1;
        }
        else
        {
          do
          {
            if ( v99 )
              *v99 = -4096;
            v99 += 2;
          }
          while ( v112 != v99 );
          v112 = *(_QWORD **)(a4 + 1368);
          v113 = *(_DWORD *)(a4 + 1384);
          v92 = *(_DWORD *)(a4 + 1376) + 1;
        }
      }
      if ( v113 )
      {
        v114 = v113 - 1;
        v115 = 1;
        v116 = 0;
        v117 = v114 & v48;
        v47 = &v112[2 * v117];
        v118 = *v47;
        if ( v75 != *v47 )
        {
          while ( v118 != -4096 )
          {
            if ( v118 == -8192 && !v116 )
              v116 = v47;
            v117 = v114 & (v115 + v117);
            v47 = &v112[2 * v117];
            v118 = *v47;
            if ( v75 == *v47 )
              goto LABEL_71;
            ++v115;
          }
          if ( v116 )
            v47 = v116;
        }
        goto LABEL_71;
      }
LABEL_314:
      ++*(_DWORD *)(a4 + 1376);
      BUG();
    }
LABEL_71:
    *(_DWORD *)(a4 + 1376) = v92;
    if ( *v47 != -4096 )
      --*(_DWORD *)(a4 + 1380);
    *v47 = v75;
    *((_DWORD *)v47 + 2) = 0;
LABEL_74:
    if ( v202 < v12 )
    {
      v96 = *(_QWORD *)v202;
      *(_QWORD *)v202 = *(_QWORD *)v12;
      *(_QWORD *)v12 = v96;
      v13 = *(unsigned int *)(a4 + 1384);
      goto LABEL_12;
    }
    sub_28D8430(v202, v185, v183, a4);
    result = v202 - (char *)a1;
    if ( v202 - (char *)a1 > 128 )
    {
      if ( v183 )
      {
        v185 = v202;
        continue;
      }
LABEL_218:
      v163 = result >> 3;
      v164 = ((result >> 3) - 2) >> 1;
      sub_28D7150((__int64)a1, v164, result >> 3, a1[v164], a4);
      do
      {
        --v164;
        sub_28D7150((__int64)a1, v164, v163, a1[v164], a4);
      }
      while ( v164 );
      v165 = v201;
      do
      {
        v166 = *--v165;
        *v165 = *a1;
        result = sub_28D7150((__int64)a1, 0, v165 - a1, v166, a4);
      }
      while ( (char *)v165 - (char *)a1 > 8 );
    }
    return result;
  }
}
