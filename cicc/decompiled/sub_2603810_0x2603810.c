// Function: sub_2603810
// Address: 0x2603810
//
_BYTE *__fastcall sub_2603810(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  _BYTE *result; // rax
  __int64 v9; // rdx
  __int64 v10; // r15
  unsigned int v11; // r10d
  __int64 v12; // rbx
  __int64 v13; // r12
  unsigned int **v14; // r11
  __int64 v15; // rsi
  unsigned int v16; // r14d
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // r9
  unsigned int v20; // edx
  __int64 *v21; // rdi
  __int64 v22; // r8
  unsigned int v23; // esi
  __int64 v24; // rax
  __int64 v25; // r8
  unsigned int v26; // edx
  __int64 *v27; // rcx
  __int64 v28; // rdi
  int v29; // edx
  __int64 v30; // rdi
  int v31; // edx
  unsigned int v32; // eax
  __int64 *v33; // rsi
  __int64 v34; // r8
  unsigned int *v35; // rdi
  int v36; // edi
  __int64 v37; // rcx
  unsigned int v38; // esi
  __int64 v39; // r8
  __int64 v40; // r9
  int v41; // ebx
  __int64 *v42; // rdi
  unsigned int v43; // r11d
  __int64 *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r14
  __int64 v47; // r15
  __int64 v48; // rax
  __int64 v49; // r12
  int v50; // edx
  __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 v53; // rbx
  __int64 v54; // rcx
  __int64 v55; // rdx
  __int64 v56; // rdi
  __int64 v57; // rsi
  __int64 v58; // rsi
  __int64 v59; // r8
  __int64 v60; // r8
  __int64 v61; // r8
  __int64 v62; // r13
  __int64 v63; // r8
  __int64 v64; // r10
  __int64 v65; // r9
  __int64 v66; // r8
  __int64 *v67; // r11
  __int64 v68; // r13
  unsigned int v69; // esi
  __int64 v70; // rdx
  __int64 v71; // r8
  unsigned int v72; // eax
  __int64 *v73; // rcx
  __int64 v74; // rdi
  __int64 v75; // rsi
  int v76; // eax
  int v77; // edx
  unsigned int v78; // eax
  __int64 *v79; // rdi
  __int64 v80; // r8
  __int64 v81; // r10
  __int64 v82; // r9
  __int64 v83; // r8
  __int64 *v84; // r11
  __int64 v85; // r13
  __int64 v86; // rbx
  char *v87; // rax
  unsigned __int8 v88; // cl
  __int64 v89; // rsi
  __int64 v90; // rax
  __int64 v91; // rdi
  unsigned int v92; // edx
  __int64 *v93; // r8
  __int64 v94; // r9
  _QWORD *v95; // rax
  _QWORD *v96; // rdx
  __int64 v97; // rdx
  __int64 v98; // rsi
  __int64 v99; // rax
  unsigned int v100; // ecx
  __int64 *v101; // rdi
  __int64 v102; // r8
  unsigned int v103; // esi
  __int64 v104; // r8
  unsigned int v105; // edx
  __int64 *v106; // rcx
  __int64 v107; // rdi
  int v108; // ebx
  __int64 *v109; // r10
  int v110; // ecx
  int v111; // ecx
  __int64 v112; // r10
  __int64 v113; // r9
  __int64 v114; // r8
  __int64 *v115; // r11
  __int64 v116; // r13
  __int64 v117; // r10
  __int64 v118; // r9
  __int64 v119; // r8
  __int64 *v120; // r11
  __int64 v121; // r13
  int v122; // r8d
  int v123; // r11d
  __int64 *v124; // r10
  int v125; // ecx
  int v126; // r11d
  int v127; // r13d
  __int64 v128; // r11
  int v129; // r11d
  int v130; // r13d
  __int64 v131; // r11
  int v132; // r11d
  int v133; // r13d
  __int64 v134; // r11
  int v135; // r11d
  int v136; // r13d
  __int64 v137; // r11
  int v138; // edi
  int v139; // r10d
  int v140; // edi
  int v141; // r9d
  int v142; // ebx
  int v143; // eax
  int v144; // ebx
  __int64 *v145; // r10
  int v146; // ecx
  int v147; // edx
  int v148; // eax
  int v149; // eax
  __int64 v150; // rbx
  __int64 v151; // r8
  __int64 v152; // rbx
  __int64 v153; // rax
  __int64 *v154; // r11
  int v155; // [rsp+4h] [rbp-DCh]
  int v156; // [rsp+4h] [rbp-DCh]
  int v157; // [rsp+4h] [rbp-DCh]
  int v158; // [rsp+4h] [rbp-DCh]
  __int64 v160; // [rsp+10h] [rbp-D0h]
  unsigned int *v162; // [rsp+28h] [rbp-B8h]
  unsigned int v164; // [rsp+38h] [rbp-A8h]
  unsigned int v165; // [rsp+38h] [rbp-A8h]
  unsigned int v166; // [rsp+38h] [rbp-A8h]
  unsigned int v167; // [rsp+38h] [rbp-A8h]
  __int64 v169; // [rsp+48h] [rbp-98h]
  unsigned int *v170; // [rsp+50h] [rbp-90h]
  unsigned int v171; // [rsp+50h] [rbp-90h]
  int v172; // [rsp+58h] [rbp-88h]
  __int64 v173; // [rsp+58h] [rbp-88h]
  unsigned int **v174; // [rsp+58h] [rbp-88h]
  __int64 v175; // [rsp+68h] [rbp-78h] BYREF
  unsigned int *v176; // [rsp+70h] [rbp-70h] BYREF
  __int64 v177; // [rsp+78h] [rbp-68h]
  _BYTE v178[16]; // [rsp+80h] [rbp-60h] BYREF
  _QWORD v179[10]; // [rsp+90h] [rbp-50h] BYREF

  v7 = sub_AA5930(a1);
  result = v178;
  v160 = v9;
  if ( v9 == v7 )
    return result;
  v10 = a3;
  do
  {
    v176 = (unsigned int *)v178;
    v177 = 0x200000000LL;
    v11 = *(_DWORD *)(v7 + 4) & 0x7FFFFFF;
    if ( !v11 )
      goto LABEL_22;
    v12 = 0;
    v13 = 0;
    v14 = &v176;
    do
    {
      while ( 1 )
      {
        v15 = *(_QWORD *)(v10 + 8);
        v16 = v12;
        v17 = *(_QWORD *)(*(_QWORD *)(v7 - 8) + 32LL * *(unsigned int *)(v7 + 72) + 8 * v12);
        v18 = *(unsigned int *)(v10 + 24);
        if ( (_DWORD)v18 )
        {
          v19 = (unsigned int)(v18 - 1);
          v20 = v19 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
          v21 = (__int64 *)(v15 + 8LL * v20);
          v22 = *v21;
          if ( v17 != *v21 )
          {
            v36 = 1;
            while ( v22 != -4096 )
            {
              v20 = v19 & (v36 + v20);
              v172 = v36 + 1;
              v21 = (__int64 *)(v15 + 8LL * v20);
              v22 = *v21;
              if ( v17 == *v21 )
                goto LABEL_8;
              v36 = v172;
            }
            goto LABEL_5;
          }
LABEL_8:
          if ( v21 != (__int64 *)(v15 + 8 * v18) )
            break;
        }
LABEL_5:
        if ( v11 <= (unsigned int)++v12 )
          goto LABEL_12;
      }
      if ( v13 + 1 > (unsigned __int64)HIDWORD(v177) )
      {
        v171 = v11;
        v174 = v14;
        sub_C8D5F0((__int64)v14, v178, v13 + 1, 4u, v22, v19);
        v13 = (unsigned int)v177;
        v11 = v171;
        v14 = v174;
      }
      ++v12;
      v176[v13] = v16;
      v13 = (unsigned int)(v177 + 1);
      LODWORD(v177) = v177 + 1;
    }
    while ( v11 > (unsigned int)v12 );
LABEL_12:
    if ( !(_DWORD)v13 )
      goto LABEL_19;
    if ( (_DWORD)v13 != 1 )
    {
      v175 = v7;
      v37 = v7;
      v38 = *(_DWORD *)(a4 + 24);
      if ( v38 )
      {
        v39 = *(_QWORD *)(a4 + 8);
        v40 = v38 - 1;
        v41 = 1;
        v42 = 0;
        v43 = v40 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v44 = (__int64 *)(v39 + 8LL * v43);
        v45 = *v44;
        if ( v7 == *v44 )
          goto LABEL_33;
        while ( v45 != -4096 )
        {
          if ( v45 == -8192 && !v42 )
            v42 = v44;
          v43 = v40 & (v41 + v43);
          v44 = (__int64 *)(v39 + 8LL * v43);
          v45 = *v44;
          if ( v7 == *v44 )
            goto LABEL_33;
          ++v41;
        }
        if ( !v42 )
          v42 = v44;
        ++*(_QWORD *)a4;
        v148 = *(_DWORD *)(a4 + 16);
        v179[0] = v42;
        v149 = v148 + 1;
        if ( 4 * v149 < 3 * v38 )
        {
          v150 = a4;
          v151 = v38 >> 3;
          if ( v38 - *(_DWORD *)(a4 + 20) - v149 > (unsigned int)v151 )
          {
LABEL_179:
            *(_DWORD *)(a4 + 16) = v149;
            if ( *v42 != -4096 )
              --*(_DWORD *)(a4 + 20);
            *v42 = v37;
            v152 = v175;
            v153 = *(unsigned int *)(a4 + 40);
            if ( v153 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 44) )
            {
              sub_C8D5F0(a4 + 32, (const void *)(a4 + 48), v153 + 1, 8u, v151, v40);
              v153 = *(unsigned int *)(a4 + 40);
            }
            *(_QWORD *)(*(_QWORD *)(a4 + 32) + 8 * v153) = v152;
            v13 = (unsigned int)v177;
            ++*(_DWORD *)(a4 + 40);
LABEL_33:
            v35 = v176;
            v162 = &v176[v13];
            if ( v162 == v176 )
              goto LABEL_20;
            v170 = v176;
            v46 = v10;
            v47 = v7;
LABEL_35:
            v48 = *v170;
            v49 = *(_QWORD *)(*(_QWORD *)(v47 - 8) + 32 * v48);
            v175 = v49;
            v50 = *(_DWORD *)(v47 + 4);
            LODWORD(v179[0]) = v48;
            v179[1] = v47;
            v51 = v50 & 0x7FFFFFF;
            v179[2] = v49;
            v169 = v51;
            v52 = v51;
            v53 = v51 >> 2;
            v179[3] = v46;
            if ( !(v51 >> 2) )
            {
              v62 = 0;
              goto LABEL_106;
            }
            v54 = 1;
            v55 = 96;
            v56 = 3;
            v57 = 4 * v53;
            v53 = 0;
            v173 = v57;
            v58 = 2;
            while ( 1 )
            {
              if ( (_DWORD)v48 != (_DWORD)v53 )
              {
                v63 = *(_QWORD *)(v47 - 8);
                if ( v49 == *(_QWORD *)(v63 + 32 * v53) )
                {
                  v64 = *(_QWORD *)(v46 + 8);
                  v65 = *(_QWORD *)(v63 + 32LL * *(unsigned int *)(v47 + 72) + 8 * v53);
                  v66 = *(unsigned int *)(v46 + 24);
                  if ( !(_DWORD)v66 )
                    goto LABEL_49;
                  v164 = (v66 - 1) & (((unsigned int)v65 >> 9) ^ ((unsigned int)v65 >> 4));
                  v67 = (__int64 *)(v64 + 8LL * v164);
                  v68 = *v67;
                  if ( *v67 != v65 )
                  {
                    v126 = 1;
                    while ( v68 != -4096 )
                    {
                      v127 = v126 + 1;
                      v128 = ((_DWORD)v66 - 1) & (v164 + v126);
                      v164 = v128;
                      v67 = (__int64 *)(v64 + 8 * v128);
                      v155 = v127;
                      v68 = *v67;
                      if ( v65 == *v67 )
                        goto LABEL_48;
                      v126 = v155;
                    }
                    goto LABEL_49;
                  }
LABEL_48:
                  if ( v67 == (__int64 *)(v64 + 8 * v66) )
                    goto LABEL_49;
                }
              }
              if ( (_DWORD)v48 != (_DWORD)v54 )
              {
                v59 = *(_QWORD *)(v47 - 8);
                if ( v49 == *(_QWORD *)(v59 + v55 - 64) )
                {
                  v81 = *(_QWORD *)(v46 + 8);
                  v82 = *(_QWORD *)(v59 + 32LL * *(unsigned int *)(v47 + 72) + 8 * v54);
                  v83 = *(unsigned int *)(v46 + 24);
                  if ( !(_DWORD)v83 )
                    goto LABEL_60;
                  v165 = (v83 - 1) & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
                  v84 = (__int64 *)(v81 + 8LL * v165);
                  v85 = *v84;
                  if ( v82 != *v84 )
                  {
                    v129 = 1;
                    while ( v85 != -4096 )
                    {
                      v130 = v129 + 1;
                      v131 = ((_DWORD)v83 - 1) & (v165 + v129);
                      v165 = v131;
                      v84 = (__int64 *)(v81 + 8 * v131);
                      v156 = v130;
                      v85 = *v84;
                      if ( v82 == *v84 )
                        goto LABEL_59;
                      v129 = v156;
                    }
LABEL_60:
                    if ( v54 != v169 )
                      goto LABEL_50;
                    goto LABEL_61;
                  }
LABEL_59:
                  if ( v84 == (__int64 *)(v81 + 8 * v83) )
                    goto LABEL_60;
                }
              }
              if ( (_DWORD)v48 != (_DWORD)v58 )
              {
                v60 = *(_QWORD *)(v47 - 8);
                if ( v49 == *(_QWORD *)(v60 + v55 - 32) )
                {
                  v112 = *(_QWORD *)(v46 + 8);
                  v113 = *(_QWORD *)(v60 + 32LL * *(unsigned int *)(v47 + 72) + 8 * v58);
                  v114 = *(unsigned int *)(v46 + 24);
                  if ( !(_DWORD)v114 )
                    goto LABEL_90;
                  v166 = (v114 - 1) & (((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4));
                  v115 = (__int64 *)(v112 + 8LL * v166);
                  v116 = *v115;
                  if ( *v115 != v113 )
                  {
                    v132 = 1;
                    while ( v116 != -4096 )
                    {
                      v133 = v132 + 1;
                      v134 = ((_DWORD)v114 - 1) & (v166 + v132);
                      v166 = v134;
                      v115 = (__int64 *)(v112 + 8 * v134);
                      v157 = v133;
                      v116 = *v115;
                      if ( v113 == *v115 )
                        goto LABEL_89;
                      v132 = v157;
                    }
LABEL_90:
                    v53 = v58;
                    goto LABEL_49;
                  }
LABEL_89:
                  if ( v115 == (__int64 *)(v112 + 8 * v114) )
                    goto LABEL_90;
                }
              }
              if ( (_DWORD)v48 != (_DWORD)v56 )
              {
                v61 = *(_QWORD *)(v47 - 8);
                if ( v49 == *(_QWORD *)(v61 + v55) )
                {
                  v117 = *(_QWORD *)(v46 + 8);
                  v118 = *(_QWORD *)(v61 + 32LL * *(unsigned int *)(v47 + 72) + 8 * v56);
                  v119 = *(unsigned int *)(v46 + 24);
                  if ( !(_DWORD)v119 )
                    goto LABEL_94;
                  v167 = (v119 - 1) & (((unsigned int)v118 >> 9) ^ ((unsigned int)v118 >> 4));
                  v120 = (__int64 *)(v117 + 8LL * v167);
                  v121 = *v120;
                  if ( *v120 != v118 )
                  {
                    v135 = 1;
                    while ( v121 != -4096 )
                    {
                      v136 = v135 + 1;
                      v137 = ((_DWORD)v119 - 1) & (v167 + v135);
                      v167 = v137;
                      v120 = (__int64 *)(v117 + 8 * v137);
                      v158 = v136;
                      v121 = *v120;
                      if ( v118 == *v120 )
                        goto LABEL_93;
                      v135 = v158;
                    }
LABEL_94:
                    v53 = v56;
LABEL_49:
                    if ( v53 == v169 )
                      goto LABEL_61;
LABEL_50:
                    v69 = *(_DWORD *)(a6 + 24);
                    if ( !v69 )
                    {
                      v179[0] = 0;
                      ++*(_QWORD *)a6;
                      goto LABEL_103;
                    }
                    v70 = v175;
                    v71 = *(_QWORD *)(a6 + 8);
                    v72 = (v69 - 1) & (((unsigned int)v175 >> 9) ^ ((unsigned int)v175 >> 4));
                    v73 = (__int64 *)(v71 + 8LL * v72);
                    v74 = *v73;
                    if ( v175 != *v73 )
                    {
                      v142 = 1;
                      v124 = 0;
                      while ( v74 != -4096 )
                      {
                        if ( v124 || v74 != -8192 )
                          v73 = v124;
                        v72 = (v69 - 1) & (v142 + v72);
                        v154 = (__int64 *)(v71 + 8LL * v72);
                        v74 = *v154;
                        if ( v175 == *v154 )
                          goto LABEL_52;
                        ++v142;
                        v124 = v73;
                        v73 = (__int64 *)(v71 + 8LL * v72);
                      }
                      if ( !v124 )
                        v124 = v73;
                      ++*(_QWORD *)a6;
                      v143 = *(_DWORD *)(a6 + 16);
                      v179[0] = v124;
                      v125 = v143 + 1;
                      if ( 4 * (v143 + 1) >= 3 * v69 )
                      {
LABEL_103:
                        v69 *= 2;
                      }
                      else if ( v69 - *(_DWORD *)(a6 + 20) - v125 > v69 >> 3 )
                      {
                        goto LABEL_146;
                      }
                      sub_CE2A30(a6, v69);
                      sub_DA5B20(a6, &v175, v179);
                      v70 = v175;
                      v124 = (__int64 *)v179[0];
                      v125 = *(_DWORD *)(a6 + 16) + 1;
LABEL_146:
                      *(_DWORD *)(a6 + 16) = v125;
                      if ( *v124 != -4096 )
                        --*(_DWORD *)(a6 + 20);
                      *v124 = v70;
                    }
LABEL_52:
                    v75 = *(_QWORD *)(a5 + 8);
                    v76 = *(_DWORD *)(a5 + 24);
                    if ( v76 )
                    {
                      v77 = v76 - 1;
                      v78 = (v76 - 1) & (((unsigned int)v175 >> 9) ^ ((unsigned int)v175 >> 4));
                      v79 = (__int64 *)(v75 + 8LL * v78);
                      v80 = *v79;
                      if ( v175 == *v79 )
                      {
LABEL_54:
                        *v79 = -8192;
                        --*(_DWORD *)(a5 + 16);
                        ++*(_DWORD *)(a5 + 20);
                      }
                      else
                      {
                        v140 = 1;
                        while ( v80 != -4096 )
                        {
                          v141 = v140 + 1;
                          v78 = v77 & (v140 + v78);
                          v79 = (__int64 *)(v75 + 8LL * v78);
                          v80 = *v79;
                          if ( v175 == *v79 )
                            goto LABEL_54;
                          v140 = v141;
                        }
                      }
                    }
LABEL_55:
                    if ( v162 == ++v170 )
                    {
                      v7 = v47;
                      v10 = v46;
                      goto LABEL_19;
                    }
                    goto LABEL_35;
                  }
LABEL_93:
                  if ( v120 == (__int64 *)(v117 + 8 * v119) )
                    goto LABEL_94;
                }
              }
              v62 = v53 + 4;
              v56 += 4;
              v55 += 128;
              v58 += 4;
              v53 = v62;
              v54 += 4;
              if ( v62 == v173 )
              {
                v52 = v169 - v62;
LABEL_106:
                if ( v52 != 2 )
                {
                  if ( v52 != 3 )
                  {
                    if ( v52 == 1 && sub_25F5BA0((__int64)v179, v53) )
                      goto LABEL_49;
                    goto LABEL_61;
                  }
                  if ( sub_25F5BA0((__int64)v179, v53) )
                    goto LABEL_49;
                  v53 = v62 + 1;
                }
                if ( sub_25F5BA0((__int64)v179, v53) )
                  goto LABEL_49;
                if ( sub_25F5BA0((__int64)v179, ++v53) )
                  goto LABEL_49;
LABEL_61:
                v86 = *(_QWORD *)(v49 + 16);
                if ( v86 )
                {
                  while ( 1 )
                  {
LABEL_62:
                    v87 = *(char **)(v86 + 24);
                    v88 = *v87;
                    if ( (unsigned __int8)*v87 <= 0x1Cu )
                      goto LABEL_72;
                    v89 = *((_QWORD *)v87 + 5);
                    v90 = *(unsigned int *)(v46 + 24);
                    v91 = *(_QWORD *)(v46 + 8);
                    if ( (_DWORD)v90 )
                    {
                      v92 = (v90 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
                      v93 = (__int64 *)(v91 + 8LL * v92);
                      v94 = *v93;
                      if ( *v93 == v89 )
                      {
LABEL_65:
                        if ( v93 != (__int64 *)(v91 + 8 * v90) )
                          goto LABEL_72;
                      }
                      else
                      {
                        v122 = 1;
                        while ( v94 != -4096 )
                        {
                          v123 = v122 + 1;
                          v92 = (v90 - 1) & (v122 + v92);
                          v93 = (__int64 *)(v91 + 8LL * v92);
                          v94 = *v93;
                          if ( v89 == *v93 )
                            goto LABEL_65;
                          v122 = v123;
                        }
                      }
                    }
                    if ( v88 != 84 )
                      goto LABEL_50;
                    if ( *(_BYTE *)(a2 + 28) )
                      break;
                    if ( !sub_C8CA60(a2, v89) )
                      goto LABEL_50;
                    v86 = *(_QWORD *)(v86 + 8);
                    if ( !v86 )
                      goto LABEL_73;
                  }
                  v95 = *(_QWORD **)(a2 + 8);
                  v96 = &v95[*(unsigned int *)(a2 + 20)];
                  if ( v95 == v96 )
                    goto LABEL_50;
                  while ( v89 != *v95 )
                  {
                    if ( v96 == ++v95 )
                      goto LABEL_50;
                  }
LABEL_72:
                  v86 = *(_QWORD *)(v86 + 8);
                  if ( !v86 )
                    goto LABEL_73;
                  goto LABEL_62;
                }
LABEL_73:
                v97 = *(unsigned int *)(a6 + 24);
                v98 = *(_QWORD *)(a6 + 8);
                v99 = v175;
                if ( (_DWORD)v97 )
                {
                  v100 = (v97 - 1) & (((unsigned int)v175 >> 9) ^ ((unsigned int)v175 >> 4));
                  v101 = (__int64 *)(v98 + 8LL * v100);
                  v102 = *v101;
                  if ( v175 == *v101 )
                  {
LABEL_75:
                    if ( v101 != (__int64 *)(v98 + 8 * v97) )
                      goto LABEL_55;
                  }
                  else
                  {
                    v138 = 1;
                    while ( v102 != -4096 )
                    {
                      v139 = v138 + 1;
                      v100 = (v97 - 1) & (v138 + v100);
                      v101 = (__int64 *)(v98 + 8LL * v100);
                      v102 = *v101;
                      if ( v175 == *v101 )
                        goto LABEL_75;
                      v138 = v139;
                    }
                  }
                }
                v103 = *(_DWORD *)(a5 + 24);
                if ( v103 )
                {
                  v104 = *(_QWORD *)(a5 + 8);
                  v105 = (v103 - 1) & (((unsigned int)v175 >> 9) ^ ((unsigned int)v175 >> 4));
                  v106 = (__int64 *)(v104 + 8LL * v105);
                  v107 = *v106;
                  if ( v175 == *v106 )
                    goto LABEL_55;
                  v108 = 1;
                  v109 = 0;
                  while ( v107 != -4096 )
                  {
                    if ( !v109 && v107 == -8192 )
                      v109 = v106;
                    v105 = (v103 - 1) & (v108 + v105);
                    v106 = (__int64 *)(v104 + 8LL * v105);
                    v107 = *v106;
                    if ( v175 == *v106 )
                      goto LABEL_55;
                    ++v108;
                  }
                  if ( !v109 )
                    v109 = v106;
                  ++*(_QWORD *)a5;
                  v110 = *(_DWORD *)(a5 + 16);
                  v179[0] = v109;
                  v111 = v110 + 1;
                  if ( 4 * v111 < 3 * v103 )
                  {
                    if ( v103 - *(_DWORD *)(a5 + 20) - v111 > v103 >> 3 )
                    {
LABEL_84:
                      *(_DWORD *)(a5 + 16) = v111;
                      if ( *v109 != -4096 )
                        --*(_DWORD *)(a5 + 20);
                      *v109 = v99;
                      goto LABEL_55;
                    }
LABEL_152:
                    sub_CE2A30(a5, v103);
                    sub_DA5B20(a5, &v175, v179);
                    v99 = v175;
                    v109 = (__int64 *)v179[0];
                    v111 = *(_DWORD *)(a5 + 16) + 1;
                    goto LABEL_84;
                  }
                }
                else
                {
                  v179[0] = 0;
                  ++*(_QWORD *)a5;
                }
                v103 *= 2;
                goto LABEL_152;
              }
            }
          }
LABEL_186:
          sub_CE2A30(v150, v38);
          sub_DA5B20(v150, &v175, v179);
          v37 = v175;
          v42 = (__int64 *)v179[0];
          v149 = *(_DWORD *)(v150 + 16) + 1;
          goto LABEL_179;
        }
      }
      else
      {
        v179[0] = 0;
        ++*(_QWORD *)a4;
      }
      v150 = a4;
      v38 *= 2;
      goto LABEL_186;
    }
    v23 = *(_DWORD *)(a6 + 24);
    v24 = *(_QWORD *)(*(_QWORD *)(v7 - 8) + 32LL * *v176);
    v175 = v24;
    if ( !v23 )
    {
      v179[0] = 0;
      ++*(_QWORD *)a6;
LABEL_167:
      v23 *= 2;
LABEL_168:
      sub_CE2A30(a6, v23);
      sub_DA5B20(a6, &v175, v179);
      v24 = v175;
      v145 = (__int64 *)v179[0];
      v147 = *(_DWORD *)(a6 + 16) + 1;
      goto LABEL_163;
    }
    v25 = *(_QWORD *)(a6 + 8);
    v26 = (v23 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
    v27 = (__int64 *)(v25 + 8LL * v26);
    v28 = *v27;
    if ( v24 == *v27 )
      goto LABEL_16;
    v144 = 1;
    v145 = 0;
    while ( v28 != -4096 )
    {
      if ( !v145 && v28 == -8192 )
        v145 = v27;
      v26 = (v23 - 1) & (v144 + v26);
      v27 = (__int64 *)(v25 + 8LL * v26);
      v28 = *v27;
      if ( v24 == *v27 )
        goto LABEL_16;
      ++v144;
    }
    if ( !v145 )
      v145 = v27;
    ++*(_QWORD *)a6;
    v146 = *(_DWORD *)(a6 + 16);
    v179[0] = v145;
    v147 = v146 + 1;
    if ( 4 * (v146 + 1) >= 3 * v23 )
      goto LABEL_167;
    if ( v23 - *(_DWORD *)(a6 + 20) - v147 <= v23 >> 3 )
      goto LABEL_168;
LABEL_163:
    *(_DWORD *)(a6 + 16) = v147;
    if ( *v145 != -4096 )
      --*(_DWORD *)(a6 + 20);
    *v145 = v24;
LABEL_16:
    v29 = *(_DWORD *)(a5 + 24);
    v30 = *(_QWORD *)(a5 + 8);
    if ( v29 )
    {
      v31 = v29 - 1;
      v32 = v31 & (((unsigned int)v175 >> 9) ^ ((unsigned int)v175 >> 4));
      v33 = (__int64 *)(v30 + 8LL * v32);
      v34 = *v33;
      if ( *v33 == v175 )
      {
LABEL_18:
        *v33 = -8192;
        --*(_DWORD *)(a5 + 16);
        ++*(_DWORD *)(a5 + 20);
      }
      else
      {
        while ( v34 != -4096 )
        {
          v32 = v31 & (v13 + v32);
          v33 = (__int64 *)(v30 + 8LL * v32);
          v34 = *v33;
          if ( v175 == *v33 )
            goto LABEL_18;
          LODWORD(v13) = v13 + 1;
        }
      }
    }
LABEL_19:
    v35 = v176;
LABEL_20:
    if ( v35 != (unsigned int *)v178 )
      _libc_free((unsigned __int64)v35);
LABEL_22:
    result = *(_BYTE **)(v7 + 32);
    if ( !result )
      BUG();
    v7 = 0;
    if ( *(result - 24) == 84 )
      v7 = (__int64)(result - 24);
  }
  while ( v160 != v7 );
  return result;
}
