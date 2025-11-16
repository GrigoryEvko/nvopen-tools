// Function: sub_1401B00
// Address: 0x1401b00
//
__int64 __fastcall sub_1401B00(__int64 a1, const __m128i *a2)
{
  __int64 *v3; // rdx
  __int64 *v4; // rcx
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // rax
  unsigned int v7; // eax
  unsigned __int64 v8; // rdi
  __int64 v9; // rdi
  _QWORD *v10; // rax
  _QWORD *j; // rdx
  const __m128i *v12; // rax
  const __m128i **v13; // r12
  int v14; // edx
  __int64 v15; // r8
  int v16; // esi
  unsigned int v17; // ecx
  const __m128i **v18; // rdx
  const __m128i *v19; // r9
  const __m128i *v20; // rbx
  const __m128i *v21; // rdx
  __int64 *v22; // r13
  unsigned int v23; // ebx
  __int64 v24; // rdi
  __int64 **v25; // rcx
  __int64 *v26; // r11
  const __m128i *v27; // r15
  const __m128i **v28; // rax
  __int64 v29; // rax
  char *v30; // rax
  char *v31; // rdx
  char *v32; // rsi
  const __m128i *v33; // r14
  const __m128i **v34; // rdx
  __int64 v35; // rdx
  unsigned int v36; // edi
  __m128i **v37; // rax
  __m128i *v38; // rcx
  __int64 v39; // rax
  _BYTE *v40; // rsi
  __m128i *v41; // rbx
  unsigned int v42; // eax
  int v43; // ecx
  __m128i **v44; // rdx
  __m128i *v45; // r8
  __int64 v46; // rax
  _BYTE *v47; // rsi
  __int64 v48; // rdx
  __int64 v49; // rcx
  _BYTE *v50; // rsi
  _BYTE *v51; // rax
  __int64 v52; // rdi
  __int64 v54; // rbx
  __int64 *v55; // r14
  int v56; // eax
  __int64 v57; // rsi
  int v58; // ecx
  unsigned int v59; // edx
  __int64 *v60; // rax
  __int64 v61; // rdi
  __int64 *v62; // rax
  __int64 *v63; // r12
  __int64 v64; // r14
  __int64 v65; // rcx
  unsigned int v66; // r8d
  unsigned int v67; // edx
  __int64 *v68; // rax
  __int64 v69; // rsi
  const __m128i *v70; // rbx
  __int64 v71; // rdx
  __int64 v72; // rax
  _BYTE *v73; // rsi
  __int64 v74; // rbx
  unsigned int v75; // esi
  __int64 v76; // r12
  __int64 v77; // rcx
  __int64 v78; // rdi
  unsigned int v79; // edx
  __int64 *v80; // rax
  __int64 v81; // r9
  __int64 v82; // rdi
  __int64 v83; // rax
  __int64 *v84; // rbx
  char v85; // r15
  __int64 *v86; // r12
  unsigned int v87; // r15d
  unsigned int v88; // r9d
  __int64 *v89; // rdx
  __int64 v90; // r8
  __int64 v91; // rsi
  __int64 *v92; // r13
  int v93; // eax
  __int64 v94; // rcx
  int v95; // edi
  unsigned int v96; // edx
  __int64 *v97; // rax
  __int64 v98; // r8
  __int64 *v99; // rax
  __int64 v100; // r13
  __int64 v101; // rcx
  __int64 v102; // rdi
  unsigned int v103; // esi
  int v104; // r8d
  int v105; // r8d
  __int64 v106; // r10
  int v107; // edi
  unsigned int v108; // esi
  __int64 v109; // r9
  unsigned int v110; // esi
  unsigned int v111; // edx
  __int64 *v112; // rax
  __int64 v113; // r8
  int v114; // eax
  __int64 *v115; // r11
  int v116; // edi
  int v117; // edx
  int v118; // r10d
  __int64 v119; // r11
  __int64 *v120; // r8
  unsigned int v121; // r15d
  int v122; // r9d
  __int64 v123; // rsi
  int v124; // edx
  int v125; // r10d
  int v126; // esi
  __int64 v127; // r13
  __m128i **v128; // rax
  __m128i *v129; // rdi
  int v130; // edi
  __int64 v131; // r8
  int v132; // edi
  __int64 v133; // r9
  unsigned int v134; // esi
  __int64 *v135; // rax
  __int64 v136; // r11
  _BYTE *v137; // rsi
  _BYTE *v138; // rax
  __int64 v139; // rax
  __int64 v140; // r12
  _BYTE *v141; // rsi
  const __m128i *v142; // r14
  int v143; // r9d
  int v144; // eax
  unsigned int v145; // r8d
  unsigned int v146; // edx
  __int64 *v147; // rax
  __int64 v148; // rsi
  __int64 v149; // r13
  unsigned int v150; // ebx
  int i; // eax
  int v152; // r12d
  __int64 *v153; // rax
  int v154; // eax
  int v155; // ebx
  int v156; // eax
  int v157; // r11d
  __int64 *v158; // r10
  int v159; // edi
  int v160; // edx
  int v161; // r10d
  int v162; // r10d
  __int64 v163; // r11
  int v164; // edx
  unsigned int v165; // esi
  __int64 v166; // r9
  int v167; // r15d
  __int64 **v168; // rdx
  int v169; // eax
  unsigned int v170; // ecx
  __int64 *v171; // r9
  int v172; // edi
  __int64 **v173; // rsi
  int v174; // ecx
  unsigned int v175; // ebx
  __int64 **v176; // rdi
  __int64 *v177; // rsi
  int v178; // r11d
  __int64 *v179; // r15
  int v180; // r10d
  __int64 *v181; // r9
  int v182; // ecx
  int v183; // edx
  __int64 v184; // rdx
  int v185; // edi
  __m128i **v186; // rsi
  int v187; // esi
  int v188; // r10d
  int v189; // eax
  int v190; // edi
  int v191; // r8d
  int v192; // r9d
  int v193; // r8d
  __int64 *v194; // rdi
  __int64 v195; // [rsp+0h] [rbp-110h]
  __int64 v196; // [rsp+0h] [rbp-110h]
  __int64 *v197; // [rsp+8h] [rbp-108h]
  int v198; // [rsp+8h] [rbp-108h]
  __int64 *v199; // [rsp+8h] [rbp-108h]
  const __m128i **v201; // [rsp+18h] [rbp-F8h]
  char v202; // [rsp+18h] [rbp-F8h]
  __int64 v203; // [rsp+18h] [rbp-F8h]
  __int64 v204; // [rsp+20h] [rbp-F0h] BYREF
  __int64 *v205; // [rsp+28h] [rbp-E8h] BYREF
  __m128i v206; // [rsp+30h] [rbp-E0h] BYREF
  const __m128i *v207; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v208; // [rsp+48h] [rbp-C8h]
  __int64 v209; // [rsp+50h] [rbp-C0h]
  __int64 v210; // [rsp+58h] [rbp-B8h]
  const __m128i *v211; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v212; // [rsp+68h] [rbp-A8h]
  _QWORD v213[2]; // [rsp+70h] [rbp-A0h] BYREF
  _QWORD *v214; // [rsp+80h] [rbp-90h]
  __int64 v215; // [rsp+88h] [rbp-88h]
  int v216; // [rsp+90h] [rbp-80h]
  __int64 *v217; // [rsp+98h] [rbp-78h] BYREF
  __int64 *v218; // [rsp+A0h] [rbp-70h]
  __int64 v219; // [rsp+A8h] [rbp-68h]
  __int64 v220; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v221; // [rsp+B8h] [rbp-58h]
  __int64 v222; // [rsp+C0h] [rbp-50h]
  unsigned int v223; // [rsp+C8h] [rbp-48h]
  char v224; // [rsp+D0h] [rbp-40h]

  v3 = (__int64 *)a2[2].m128i_i64[1];
  v4 = (__int64 *)a2[2].m128i_i64[0];
  if ( !a2->m128i_i64[0] )
  {
    if ( v3 == v4 )
    {
LABEL_155:
      v137 = *(_BYTE **)(a1 + 32);
      if ( *(const __m128i **)v137 != a2 )
      {
        v138 = v137 + 8;
        do
        {
          v137 = v138;
          v138 += 8;
        }
        while ( *((const __m128i **)v138 - 1) != a2 );
      }
      sub_13FDAF0(a1 + 32, v137);
      v139 = a2[1].m128i_i64[0];
      if ( a2->m128i_i64[1] != v139 )
      {
        v203 = a1 + 32;
        v140 = a1;
        do
        {
          while ( 1 )
          {
            v142 = *(const __m128i **)(v139 - 8);
            sub_13FDAF0((__int64)&a2->m128i_i64[1], (_BYTE *)(v139 - 8));
            v142->m128i_i64[0] = 0;
            v141 = *(_BYTE **)(v140 + 40);
            v211 = v142;
            if ( v141 != *(_BYTE **)(v140 + 48) )
              break;
            sub_13FD960(v203, v141, &v211);
            v139 = a2[1].m128i_i64[0];
            if ( v139 == a2->m128i_i64[1] )
              return sub_13FACC0((__int64)a2);
          }
          if ( v141 )
          {
            *(_QWORD *)v141 = v142;
            v141 = *(_BYTE **)(v140 + 40);
          }
          *(_QWORD *)(v140 + 40) = v141 + 8;
          v139 = a2[1].m128i_i64[0];
        }
        while ( v139 != a2->m128i_i64[1] );
      }
      return sub_13FACC0((__int64)a2);
    }
    while ( 1 )
    {
      v130 = *(_DWORD *)(a1 + 24);
      if ( !v130 )
        goto LABEL_150;
      v131 = *v4;
      v132 = v130 - 1;
      v133 = *(_QWORD *)(a1 + 8);
      v134 = v132 & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
      v135 = (__int64 *)(v133 + 16LL * v134);
      v136 = *v135;
      if ( *v4 != *v135 )
      {
        v149 = *v135;
        v150 = v132 & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
        for ( i = 1; ; i = v152 )
        {
          if ( v149 == -8 )
            goto LABEL_150;
          v152 = i + 1;
          v150 = v132 & (i + v150);
          v153 = (__int64 *)(v133 + 16LL * v150);
          v149 = *v153;
          if ( v131 == *v153 )
            break;
        }
        if ( (const __m128i *)v153[1] == a2 )
        {
          v154 = 1;
          while ( v136 != -8 )
          {
            v155 = v154 + 1;
            v134 = v132 & (v154 + v134);
            v135 = (__int64 *)(v133 + 16LL * v134);
            v136 = *v135;
            if ( v131 == *v135 )
              goto LABEL_154;
            v154 = v155;
          }
        }
        goto LABEL_150;
      }
      if ( a2 == (const __m128i *)v135[1] )
      {
LABEL_154:
        ++v4;
        *v135 = -16;
        --*(_DWORD *)(a1 + 16);
        ++*(_DWORD *)(a1 + 20);
        if ( v4 == v3 )
          goto LABEL_155;
      }
      else
      {
LABEL_150:
        if ( ++v4 == v3 )
          goto LABEL_155;
      }
    }
  }
  v212 = a1;
  v213[1] = 0;
  v211 = a2;
  v5 = (unsigned int)(v3 - v4);
  v213[0] = a2;
  v6 = ((((((((v5 | (v5 >> 1)) >> 2) | v5 | (v5 >> 1)) >> 4) | ((v5 | (v5 >> 1)) >> 2) | v5 | (v5 >> 1)) >> 8)
       | ((((v5 | (v5 >> 1)) >> 2) | v5 | (v5 >> 1)) >> 4)
       | ((v5 | (v5 >> 1)) >> 2)
       | v5
       | (v5 >> 1)) >> 16)
     | ((((((v5 | (v5 >> 1)) >> 2) | v5 | (v5 >> 1)) >> 4) | ((v5 | (v5 >> 1)) >> 2) | v5 | (v5 >> 1)) >> 8)
     | ((((v5 | (v5 >> 1)) >> 2) | v5 | (v5 >> 1)) >> 4)
     | ((v5 | (v5 >> 1)) >> 2)
     | v5
     | (v5 >> 1);
  if ( (_DWORD)v6 == -1 )
  {
    v214 = 0;
    v215 = 0;
    v216 = 0;
  }
  else
  {
    v7 = 4 * (v6 + 1);
    v8 = (((((((v7 / 3 + 1) | ((unsigned __int64)(v7 / 3 + 1) >> 1)) >> 2)
           | (v7 / 3 + 1)
           | ((unsigned __int64)(v7 / 3 + 1) >> 1)) >> 4)
         | (((v7 / 3 + 1) | ((unsigned __int64)(v7 / 3 + 1) >> 1)) >> 2)
         | (v7 / 3 + 1)
         | ((unsigned __int64)(v7 / 3 + 1) >> 1)) >> 8)
       | (((((v7 / 3 + 1) | ((unsigned __int64)(v7 / 3 + 1) >> 1)) >> 2)
         | (v7 / 3 + 1)
         | ((unsigned __int64)(v7 / 3 + 1) >> 1)) >> 4)
       | (((v7 / 3 + 1) | ((unsigned __int64)(v7 / 3 + 1) >> 1)) >> 2)
       | (v7 / 3 + 1)
       | ((unsigned __int64)(v7 / 3 + 1) >> 1);
    v9 = ((v8 >> 16) | v8) + 1;
    v216 = v9;
    v10 = (_QWORD *)sub_22077B0(16 * v9);
    v215 = 0;
    v214 = v10;
    for ( j = &v10[2 * (unsigned int)v9]; j != v10; v10 += 2 )
    {
      if ( v10 )
        *v10 = -8;
    }
    v5 = (unsigned int)((a2[2].m128i_i64[1] - a2[2].m128i_i64[0]) >> 3);
  }
  v217 = 0;
  v218 = 0;
  v219 = 0;
  sub_13FC0C0((__int64)&v217, v5);
  v12 = v211;
  v220 = 0;
  v221 = 0;
  v222 = 0;
  v223 = 0;
  v224 = 0;
  v13 = (const __m128i **)v211[2].m128i_i64[0];
  v201 = (const __m128i **)v211[2].m128i_i64[1];
  if ( (unsigned int)(v201 - v13) )
  {
    v206.m128i_i64[0] = (__int64)v213;
    v206.m128i_i64[1] = v212;
    sub_13FF050((__int64 *)&v207, &v206);
LABEL_89:
    v82 = v208;
    v83 = v209;
    while ( 1 )
    {
      if ( v82 == v83 )
      {
        if ( v82 )
          j_j___libc_free_0(v82, v210 - v82);
        v202 = v224;
        if ( !v224 )
        {
LABEL_112:
          v12 = v211;
          v13 = (const __m128i **)v211[2].m128i_i64[0];
          v201 = (const __m128i **)v211[2].m128i_i64[1];
          break;
        }
        while ( 2 )
        {
          v84 = v217;
          if ( v217 == v218 )
            goto LABEL_112;
          v85 = 0;
          v86 = v218;
LABEL_99:
          v91 = *v84;
          v92 = 0;
          v93 = *(_DWORD *)(v212 + 24);
          if ( v93 )
          {
            v94 = *(_QWORD *)(v212 + 8);
            v95 = v93 - 1;
            v96 = (v93 - 1) & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
            v97 = (__int64 *)(v94 + 16LL * v96);
            v98 = *v97;
            if ( v91 == *v97 )
            {
LABEL_101:
              v92 = (__int64 *)v97[1];
            }
            else
            {
              v114 = 1;
              while ( v98 != -8 )
              {
                v143 = v114 + 1;
                v96 = v95 & (v114 + v96);
                v97 = (__int64 *)(v94 + 16LL * v96);
                v98 = *v97;
                if ( v91 == *v97 )
                  goto LABEL_101;
                v114 = v143;
              }
              v92 = 0;
            }
          }
          v99 = sub_13FFB50((__int64)&v211, v91, v92);
          if ( v99 != v92 )
          {
            v100 = v212;
            v101 = *v84;
            v102 = *(_QWORD *)(v212 + 8);
            v103 = *(_DWORD *)(v212 + 24);
            if ( v99 )
            {
              if ( !v103 )
              {
                ++*(_QWORD *)v212;
                goto LABEL_106;
              }
              v87 = ((unsigned int)v101 >> 9) ^ ((unsigned int)v101 >> 4);
              v88 = (v103 - 1) & v87;
              v89 = (__int64 *)(v102 + 16LL * v88);
              v90 = *v89;
              if ( v101 != *v89 )
              {
                v198 = 1;
                v115 = 0;
                while ( v90 != -8 )
                {
                  if ( !v115 && v90 == -16 )
                    v115 = v89;
                  v88 = (v103 - 1) & (v198 + v88);
                  v89 = (__int64 *)(v102 + 16LL * v88);
                  v90 = *v89;
                  if ( v101 == *v89 )
                    goto LABEL_97;
                  ++v198;
                }
                v116 = *(_DWORD *)(v212 + 16);
                if ( v115 )
                  v89 = v115;
                ++*(_QWORD *)v212;
                v107 = v116 + 1;
                if ( 4 * v107 >= 3 * v103 )
                {
LABEL_106:
                  v195 = v101;
                  v197 = v99;
                  sub_1400170(v100, 2 * v103);
                  v104 = *(_DWORD *)(v100 + 24);
                  if ( !v104 )
                    goto LABEL_325;
                  v101 = v195;
                  v105 = v104 - 1;
                  v106 = *(_QWORD *)(v100 + 8);
                  v107 = *(_DWORD *)(v100 + 16) + 1;
                  v99 = v197;
                  v108 = v105 & (((unsigned int)v195 >> 9) ^ ((unsigned int)v195 >> 4));
                  v89 = (__int64 *)(v106 + 16LL * v108);
                  v109 = *v89;
                  if ( v195 != *v89 )
                  {
                    v178 = 1;
                    v179 = 0;
                    while ( v109 != -8 )
                    {
                      if ( !v179 && v109 == -16 )
                        v179 = v89;
                      v108 = v105 & (v178 + v108);
                      v89 = (__int64 *)(v106 + 16LL * v108);
                      v109 = *v89;
                      if ( v195 == *v89 )
                        goto LABEL_108;
                      ++v178;
                    }
                    if ( v179 )
                      v89 = v179;
                  }
                }
                else if ( v103 - *(_DWORD *)(v100 + 20) - v107 <= v103 >> 3 )
                {
                  v196 = v101;
                  v199 = v99;
                  sub_1400170(v100, v103);
                  v117 = *(_DWORD *)(v100 + 24);
                  if ( !v117 )
                  {
LABEL_325:
                    ++*(_DWORD *)(v100 + 16);
                    BUG();
                  }
                  v118 = v117 - 1;
                  v119 = *(_QWORD *)(v100 + 8);
                  v120 = 0;
                  v121 = (v117 - 1) & v87;
                  v101 = v196;
                  v122 = 1;
                  v107 = *(_DWORD *)(v100 + 16) + 1;
                  v99 = v199;
                  v89 = (__int64 *)(v119 + 16LL * v121);
                  v123 = *v89;
                  if ( v196 != *v89 )
                  {
                    while ( v123 != -8 )
                    {
                      if ( !v120 && v123 == -16 )
                        v120 = v89;
                      v121 = v118 & (v122 + v121);
                      v89 = (__int64 *)(v119 + 16LL * v121);
                      v123 = *v89;
                      if ( v196 == *v89 )
                        goto LABEL_108;
                      ++v122;
                    }
                    if ( v120 )
                      v89 = v120;
                  }
                }
LABEL_108:
                *(_DWORD *)(v100 + 16) = v107;
                if ( *v89 != -8 )
                  --*(_DWORD *)(v100 + 20);
                *v89 = v101;
                v89[1] = 0;
              }
LABEL_97:
              v89[1] = (__int64)v99;
              v85 = v202;
            }
            else
            {
              v85 = v202;
              if ( v103 )
              {
                v110 = v103 - 1;
                v111 = v110 & (((unsigned int)v101 >> 9) ^ ((unsigned int)v101 >> 4));
                v112 = (__int64 *)(v102 + 16LL * v111);
                v113 = *v112;
                if ( v101 == *v112 )
                {
LABEL_115:
                  *v112 = -16;
                  v85 = v202;
                  --*(_DWORD *)(v100 + 16);
                  ++*(_DWORD *)(v100 + 20);
                }
                else
                {
                  v144 = 1;
                  while ( v113 != -8 )
                  {
                    v192 = v144 + 1;
                    v111 = v110 & (v144 + v111);
                    v112 = (__int64 *)(v102 + 16LL * v111);
                    v113 = *v112;
                    if ( v101 == *v112 )
                      goto LABEL_115;
                    v144 = v192;
                  }
                  v85 = v202;
                }
              }
            }
          }
          if ( v86 == ++v84 )
          {
            if ( !v85 )
              goto LABEL_112;
            continue;
          }
          goto LABEL_99;
        }
      }
      v54 = *(_QWORD *)(v83 - 24);
      v55 = 0;
      v56 = *(_DWORD *)(v212 + 24);
      if ( v56 )
      {
        v57 = *(_QWORD *)(v212 + 8);
        v58 = v56 - 1;
        v59 = (v56 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
        v60 = (__int64 *)(v57 + 16LL * v59);
        v61 = *v60;
        if ( v54 == *v60 )
        {
LABEL_75:
          v55 = (__int64 *)v60[1];
        }
        else
        {
          v156 = 1;
          while ( v61 != -8 )
          {
            v191 = v156 + 1;
            v59 = v58 & (v156 + v59);
            v60 = (__int64 *)(v57 + 16LL * v59);
            v61 = *v60;
            if ( v54 == *v60 )
              goto LABEL_75;
            v156 = v191;
          }
          v55 = 0;
        }
      }
      v62 = sub_13FFB50((__int64)&v211, v54, v55);
      v63 = v62;
      if ( v62 != v55 )
      {
        v64 = v212;
        v204 = v54;
        v65 = *(_QWORD *)(v212 + 8);
        v66 = *(_DWORD *)(v212 + 24);
        if ( v62 )
        {
          if ( v66 )
          {
            v67 = (v66 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
            v68 = (__int64 *)(v65 + 16LL * v67);
            v69 = *v68;
            if ( v54 == *v68 )
            {
LABEL_80:
              v68[1] = (__int64)v63;
              goto LABEL_81;
            }
            v180 = 1;
            v181 = 0;
            while ( v69 != -8 )
            {
              if ( !v181 && v69 == -16 )
                v181 = v68;
              v67 = (v66 - 1) & (v180 + v67);
              v68 = (__int64 *)(v65 + 16LL * v67);
              v69 = *v68;
              if ( v54 == *v68 )
                goto LABEL_80;
              ++v180;
            }
            v182 = *(_DWORD *)(v212 + 16);
            if ( v181 )
              v68 = v181;
            ++*(_QWORD *)v212;
            v183 = v182 + 1;
            if ( 4 * (v182 + 1) < 3 * v66 )
            {
              if ( v66 - *(_DWORD *)(v64 + 20) - v183 > v66 >> 3 )
              {
LABEL_240:
                *(_DWORD *)(v64 + 16) = v183;
                if ( *v68 != -8 )
                  --*(_DWORD *)(v64 + 20);
                v184 = v204;
                v68[1] = 0;
                *v68 = v184;
                goto LABEL_80;
              }
              v187 = v66;
LABEL_249:
              sub_1400170(v64, v187);
              sub_13FD8B0(v64, &v204, &v205);
              v68 = v205;
              v183 = *(_DWORD *)(v64 + 16) + 1;
              goto LABEL_240;
            }
          }
          else
          {
            ++*(_QWORD *)v212;
          }
          v187 = 2 * v66;
          goto LABEL_249;
        }
        if ( v66 )
        {
          v145 = v66 - 1;
          v146 = v145 & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
          v147 = (__int64 *)(v65 + 16LL * v146);
          v148 = *v147;
          if ( v54 == *v147 )
          {
LABEL_175:
            *v147 = -16;
            --*(_DWORD *)(v64 + 16);
            ++*(_DWORD *)(v64 + 20);
          }
          else
          {
            v189 = 1;
            while ( v148 != -8 )
            {
              v190 = v189 + 1;
              v146 = v145 & (v189 + v146);
              v147 = (__int64 *)(v65 + 16LL * v146);
              v148 = *v147;
              if ( v54 == *v147 )
                goto LABEL_175;
              v189 = v190;
            }
          }
        }
      }
LABEL_81:
      v70 = v207;
      v71 = *(_QWORD *)(v209 - 24);
      v204 = v71;
      v72 = v207->m128i_i64[0];
      v73 = *(_BYTE **)(v207->m128i_i64[0] + 48);
      if ( v73 == *(_BYTE **)(v207->m128i_i64[0] + 56) )
      {
        sub_1292090(v72 + 40, v73, &v204);
      }
      else
      {
        if ( v73 )
        {
          *(_QWORD *)v73 = v71;
          v73 = *(_BYTE **)(v72 + 48);
        }
        *(_QWORD *)(v72 + 48) = v73 + 8;
      }
      v74 = v70->m128i_i64[0];
      v75 = *(_DWORD *)(v74 + 32);
      v76 = (__int64)(*(_QWORD *)(v74 + 48) - *(_QWORD *)(v74 + 40)) >> 3;
      if ( !v75 )
      {
        ++*(_QWORD *)(v74 + 8);
        goto LABEL_203;
      }
      v77 = v204;
      v78 = *(_QWORD *)(v74 + 16);
      v79 = (v75 - 1) & (((unsigned int)v204 >> 9) ^ ((unsigned int)v204 >> 4));
      v80 = (__int64 *)(v78 + 16LL * v79);
      v81 = *v80;
      if ( v204 != *v80 )
      {
        v157 = 1;
        v158 = 0;
        while ( v81 != -8 )
        {
          if ( !v158 && v81 == -16 )
            v158 = v80;
          v79 = (v75 - 1) & (v157 + v79);
          v80 = (__int64 *)(v78 + 16LL * v79);
          v81 = *v80;
          if ( v204 == *v80 )
            goto LABEL_87;
          ++v157;
        }
        v159 = *(_DWORD *)(v74 + 24);
        if ( v158 )
          v80 = v158;
        ++*(_QWORD *)(v74 + 8);
        v160 = v159 + 1;
        if ( 4 * (v159 + 1) >= 3 * v75 )
        {
LABEL_203:
          sub_13FEAC0(v74 + 8, 2 * v75);
          v161 = *(_DWORD *)(v74 + 32);
          if ( !v161 )
          {
            ++*(_DWORD *)(v74 + 24);
            BUG();
          }
          v77 = v204;
          v162 = v161 - 1;
          v163 = *(_QWORD *)(v74 + 16);
          v164 = *(_DWORD *)(v74 + 24);
          v165 = v162 & (((unsigned int)v204 >> 9) ^ ((unsigned int)v204 >> 4));
          v80 = (__int64 *)(v163 + 16LL * v165);
          v166 = *v80;
          if ( v204 == *v80 )
          {
LABEL_205:
            v160 = v164 + 1;
          }
          else
          {
            v193 = 1;
            v194 = 0;
            while ( v166 != -8 )
            {
              if ( v166 == -16 && !v194 )
                v194 = v80;
              v165 = v162 & (v193 + v165);
              v80 = (__int64 *)(v163 + 16LL * v165);
              v166 = *v80;
              if ( v204 == *v80 )
                goto LABEL_205;
              ++v193;
            }
            v160 = v164 + 1;
            if ( v194 )
              v80 = v194;
          }
        }
        else if ( v75 - *(_DWORD *)(v74 + 28) - v160 <= v75 >> 3 )
        {
          sub_13FEAC0(v74 + 8, v75);
          sub_13FDDE0(v74 + 8, &v204, &v205);
          v80 = v205;
          v77 = v204;
          v160 = *(_DWORD *)(v74 + 24) + 1;
        }
        *(_DWORD *)(v74 + 24) = v160;
        if ( *v80 != -8 )
          --*(_DWORD *)(v74 + 28);
        *v80 = v77;
        *((_DWORD *)v80 + 2) = 0;
      }
LABEL_87:
      *((_DWORD *)v80 + 2) = v76;
      v82 = v208;
      v209 -= 24;
      v83 = v209;
      if ( v209 != v208 )
      {
        sub_13FEC80(&v207);
        goto LABEL_89;
      }
    }
  }
  if ( v13 == v201 )
    goto LABEL_38;
  do
  {
    v14 = *(_DWORD *)(v212 + 24);
    if ( v14 )
    {
      v15 = *(_QWORD *)(v212 + 8);
      v16 = v14 - 1;
      v17 = (v14 - 1) & (((unsigned int)*v13 >> 9) ^ ((unsigned int)*v13 >> 4));
      v18 = (const __m128i **)(v15 + 16LL * v17);
      v19 = *v18;
      if ( *v13 == *v18 )
      {
LABEL_13:
        v20 = v18[1];
        if ( v20 != v12 )
        {
          if ( !v20 )
            goto LABEL_22;
          v21 = v18[1];
          while ( 1 )
          {
            v21 = (const __m128i *)v21->m128i_i64[0];
            if ( v21 == v12 )
              break;
            if ( !v21 )
              goto LABEL_22;
          }
        }
        do
        {
LABEL_18:
          v22 = (__int64 *)v20;
          v20 = (const __m128i *)v20->m128i_i64[0];
        }
        while ( v20 != v12 );
        goto LABEL_19;
      }
      v124 = 1;
      while ( v19 != (const __m128i *)-8LL )
      {
        v188 = v124 + 1;
        v17 = v16 & (v124 + v17);
        v18 = (const __m128i **)(v15 + 16LL * v17);
        v19 = *v18;
        if ( *v13 == *v18 )
          goto LABEL_13;
        v124 = v188;
      }
      v20 = 0;
      if ( v12 )
        goto LABEL_22;
      v22 = 0;
      v20 = (const __m128i *)MEMORY[0];
      if ( MEMORY[0] )
        goto LABEL_18;
    }
    else
    {
      v20 = 0;
      if ( v12 )
        goto LABEL_22;
      v22 = 0;
      v20 = (const __m128i *)MEMORY[0];
      if ( MEMORY[0] )
        goto LABEL_18;
    }
LABEL_19:
    if ( v223 )
    {
      v23 = ((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4);
      LODWORD(v24) = (v223 - 1) & v23;
      v25 = (__int64 **)(v221 + 16LL * (unsigned int)v24);
      v26 = *v25;
      if ( *v25 == v22 )
      {
LABEL_21:
        v20 = (const __m128i *)v25[1];
        goto LABEL_22;
      }
      v167 = 1;
      v168 = 0;
      while ( v26 != (__int64 *)-8LL )
      {
        if ( v26 == (__int64 *)-16LL && !v168 )
          v168 = v25;
        v24 = (v223 - 1) & ((_DWORD)v24 + v167);
        v25 = (__int64 **)(v221 + 16 * v24);
        v26 = *v25;
        if ( *v25 == v22 )
          goto LABEL_21;
        ++v167;
      }
      if ( !v168 )
        v168 = v25;
      ++v220;
      v169 = v222 + 1;
      if ( 4 * ((int)v222 + 1) < 3 * v223 )
      {
        if ( v223 - HIDWORD(v222) - v169 <= v223 >> 3 )
        {
          sub_13FF990((__int64)&v220, v223);
          if ( !v223 )
          {
LABEL_324:
            LODWORD(v222) = v222 + 1;
            BUG();
          }
          v174 = 1;
          v175 = (v223 - 1) & v23;
          v176 = 0;
          v169 = v222 + 1;
          v168 = (__int64 **)(v221 + 16LL * v175);
          v177 = *v168;
          if ( *v168 != v22 )
          {
            while ( v177 != (__int64 *)-8LL )
            {
              if ( !v176 && v177 == (__int64 *)-16LL )
                v176 = v168;
              v175 = (v223 - 1) & (v174 + v175);
              v168 = (__int64 **)(v221 + 16LL * v175);
              v177 = *v168;
              if ( *v168 == v22 )
                goto LABEL_212;
              ++v174;
            }
            if ( v176 )
              v168 = v176;
          }
        }
        goto LABEL_212;
      }
    }
    else
    {
      ++v220;
    }
    sub_13FF990((__int64)&v220, 2 * v223);
    if ( !v223 )
      goto LABEL_324;
    v170 = (v223 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v169 = v222 + 1;
    v168 = (__int64 **)(v221 + 16LL * v170);
    v171 = *v168;
    if ( *v168 != v22 )
    {
      v172 = 1;
      v173 = 0;
      while ( v171 != (__int64 *)-8LL )
      {
        if ( !v173 && v171 == (__int64 *)-16LL )
          v173 = v168;
        v170 = (v223 - 1) & (v172 + v170);
        v168 = (__int64 **)(v221 + 16LL * v170);
        v171 = *v168;
        if ( *v168 == v22 )
          goto LABEL_212;
        ++v172;
      }
      if ( v173 )
        v168 = v173;
    }
LABEL_212:
    LODWORD(v222) = v169;
    if ( *v168 != (__int64 *)-8LL )
      --HIDWORD(v222);
    *v168 = v22;
    v20 = 0;
    v168[1] = 0;
    v12 = v211;
LABEL_22:
    v27 = (const __m128i *)v12->m128i_i64[0];
    if ( (const __m128i *)v12->m128i_i64[0] == v20 )
      goto LABEL_37;
    do
    {
      while ( 1 )
      {
        v207 = *v13;
        v30 = (char *)sub_13F9BE0((_QWORD *)v27[2].m128i_i64[0], v27[2].m128i_i64[1], (__int64 *)&v207);
        v31 = (char *)v27[2].m128i_i64[1];
        v32 = v30 + 8;
        if ( v31 != v30 + 8 )
        {
          memmove(v30, v32, v31 - v32);
          v32 = (char *)v27[2].m128i_i64[1];
        }
        v28 = (const __m128i **)v27[4].m128i_i64[0];
        v27[2].m128i_i64[1] = (__int64)(v32 - 8);
        v33 = v207;
        if ( (const __m128i **)v27[4].m128i_i64[1] == v28 )
          break;
        v28 = (const __m128i **)sub_16CC9F0(&v27[3].m128i_u64[1], v207);
        if ( v33 == *v28 )
        {
          v48 = v27[4].m128i_i64[1];
          if ( v48 == v27[4].m128i_i64[0] )
            v49 = v27[5].m128i_u32[1];
          else
            v49 = v27[5].m128i_u32[0];
          v34 = (const __m128i **)(v48 + 8 * v49);
          goto LABEL_34;
        }
        v29 = v27[4].m128i_i64[1];
        if ( v29 == v27[4].m128i_i64[0] )
        {
          v28 = (const __m128i **)(v29 + 8LL * v27[5].m128i_u32[1]);
          v34 = v28;
          goto LABEL_34;
        }
LABEL_26:
        v27 = (const __m128i *)v27->m128i_i64[0];
        if ( v27 == v20 )
          goto LABEL_36;
      }
      v34 = &v28[v27[5].m128i_u32[1]];
      if ( v28 == v34 )
      {
LABEL_61:
        v28 = v34;
      }
      else
      {
        while ( v207 != *v28 )
        {
          if ( v34 == ++v28 )
            goto LABEL_61;
        }
      }
LABEL_34:
      if ( v28 == v34 )
        goto LABEL_26;
      *v28 = (const __m128i *)-2LL;
      ++v27[5].m128i_i32[2];
      v27 = (const __m128i *)v27->m128i_i64[0];
    }
    while ( v27 != v20 );
LABEL_36:
    v12 = v211;
LABEL_37:
    ++v13;
  }
  while ( v13 != v201 );
LABEL_38:
  v35 = v12[1].m128i_i64[0];
  if ( v12->m128i_i64[1] != v35 )
  {
    while ( 1 )
    {
      v41 = *(__m128i **)(v35 - 8);
      sub_13FDAF0((__int64)&v12->m128i_i64[1], (_BYTE *)(v35 - 8));
      v41->m128i_i64[0] = 0;
      if ( !v223 )
        break;
      v36 = (v223 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
      v37 = (__m128i **)(v221 + 16LL * v36);
      v38 = *v37;
      if ( v41 != *v37 )
      {
        v125 = 1;
        v44 = 0;
        while ( v38 != (__m128i *)-8LL )
        {
          if ( !v44 && v38 == (__m128i *)-16LL )
            v44 = v37;
          v36 = (v223 - 1) & (v125 + v36);
          v37 = (__m128i **)(v221 + 16LL * v36);
          v38 = *v37;
          if ( v41 == *v37 )
            goto LABEL_41;
          ++v125;
        }
        if ( !v44 )
          v44 = v37;
        ++v220;
        v43 = v222 + 1;
        if ( 4 * ((int)v222 + 1) < 3 * v223 )
        {
          if ( v223 - HIDWORD(v222) - v43 <= v223 >> 3 )
          {
            sub_13FF990((__int64)&v220, v223);
            if ( !v223 )
              goto LABEL_324;
            v126 = 1;
            LODWORD(v127) = (v223 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
            v43 = v222 + 1;
            v128 = 0;
            v44 = (__m128i **)(v221 + 16LL * (unsigned int)v127);
            v129 = *v44;
            if ( v41 != *v44 )
            {
              while ( v129 != (__m128i *)-8LL )
              {
                if ( v129 == (__m128i *)-16LL && !v128 )
                  v128 = v44;
                v127 = (v223 - 1) & ((_DWORD)v127 + v126);
                v44 = (__m128i **)(v221 + 16 * v127);
                v129 = *v44;
                if ( v41 == *v44 )
                  goto LABEL_51;
                ++v126;
              }
              if ( v128 )
                v44 = v128;
            }
          }
          goto LABEL_51;
        }
LABEL_49:
        sub_13FF990((__int64)&v220, 2 * v223);
        if ( !v223 )
          goto LABEL_324;
        v42 = (v223 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
        v43 = v222 + 1;
        v44 = (__m128i **)(v221 + 16LL * v42);
        v45 = *v44;
        if ( v41 != *v44 )
        {
          v185 = 1;
          v186 = 0;
          while ( v45 != (__m128i *)-8LL )
          {
            if ( !v186 && v45 == (__m128i *)-16LL )
              v186 = v44;
            v42 = (v223 - 1) & (v185 + v42);
            v44 = (__m128i **)(v221 + 16LL * v42);
            v45 = *v44;
            if ( v41 == *v44 )
              goto LABEL_51;
            ++v185;
          }
          if ( v186 )
            v44 = v186;
        }
LABEL_51:
        LODWORD(v222) = v43;
        if ( *v44 != (__m128i *)-8LL )
          --HIDWORD(v222);
        *v44 = v41;
        v44[1] = 0;
        goto LABEL_54;
      }
LABEL_41:
      v39 = (__int64)v37[1];
      if ( v39 )
      {
        v207 = v41;
        v41->m128i_i64[0] = v39;
        v40 = *(_BYTE **)(v39 + 16);
        if ( v40 == *(_BYTE **)(v39 + 24) )
        {
          sub_13FD960(v39 + 8, v40, &v207);
        }
        else
        {
          if ( v40 )
          {
            *(_QWORD *)v40 = v207;
            v40 = *(_BYTE **)(v39 + 16);
          }
          *(_QWORD *)(v39 + 16) = v40 + 8;
        }
        goto LABEL_46;
      }
LABEL_54:
      v46 = v212;
      v207 = v41;
      v47 = *(_BYTE **)(v212 + 40);
      if ( v47 == *(_BYTE **)(v212 + 48) )
      {
        sub_13FD960(v212 + 32, v47, &v207);
      }
      else
      {
        if ( v47 )
        {
          *(_QWORD *)v47 = v41;
          v47 = *(_BYTE **)(v46 + 40);
        }
        *(_QWORD *)(v46 + 40) = v47 + 8;
      }
LABEL_46:
      v12 = v211;
      v35 = v211[1].m128i_i64[0];
      if ( v35 == v211->m128i_i64[1] )
        goto LABEL_67;
    }
    ++v220;
    goto LABEL_49;
  }
LABEL_67:
  v50 = *(_BYTE **)(a2->m128i_i64[0] + 8);
  v51 = v50 + 8;
  if ( *(const __m128i **)v50 != a2 )
  {
    do
    {
      v50 = v51;
      v51 += 8;
    }
    while ( *((const __m128i **)v51 - 1) != a2 );
  }
  sub_13FDAF0(a2->m128i_i64[0] + 8, v50);
  v52 = v221;
  a2->m128i_i64[0] = 0;
  j___libc_free_0(v52);
  if ( v217 )
    j_j___libc_free_0(v217, v219 - (_QWORD)v217);
  j___libc_free_0(v214);
  return sub_13FACC0((__int64)a2);
}
