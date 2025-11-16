// Function: sub_2D85AC0
// Address: 0x2d85ac0
//
__int64 __fastcall sub_2D85AC0(char *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // r14
  char *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rsi
  bool v13; // r13
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rax
  bool v17; // r13
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rcx
  char *v21; // r13
  __int64 v22; // r15
  __int64 v23; // rdx
  bool v24; // bl
  __int64 v25; // r12
  __int64 v26; // rdx
  char *v27; // rbx
  char *j; // r13
  bool v29; // r12
  unsigned int v30; // r8d
  __int64 v31; // r11
  unsigned __int64 v32; // rax
  __int64 v33; // rax
  _QWORD *v34; // rax
  __int64 v35; // r11
  __int64 v36; // rdx
  __int64 v37; // r8
  _QWORD *v38; // rdx
  __int64 v39; // r15
  __int64 v40; // r12
  char *v41; // r14
  __int64 v42; // rdx
  __int64 v43; // rbx
  __int64 v44; // rcx
  int v45; // esi
  int v46; // esi
  __int64 v47; // r10
  __int64 v48; // rdi
  __int64 v49; // r13
  __int64 v50; // r9
  bool v51; // al
  __int64 v52; // rdi
  __int64 v53; // rcx
  __int64 v54; // r15
  int v55; // edx
  __int64 v56; // rax
  int v57; // r15d
  unsigned int v58; // r8d
  __int64 v59; // r10
  unsigned __int64 v60; // rax
  __int64 v61; // rax
  _QWORD *v62; // rax
  __int64 v63; // r10
  __int64 v64; // rdx
  __int64 v65; // r8
  _QWORD *v66; // rdx
  __int64 v67; // r12
  __int64 v68; // rbx
  __int64 v69; // rdx
  __int64 v70; // r13
  __int64 v71; // rcx
  int v72; // esi
  int v73; // esi
  __int64 v74; // r10
  __int64 v75; // rdi
  __int64 v76; // r14
  __int64 v77; // r9
  bool v78; // al
  __int64 v79; // rdi
  __int64 v80; // rsi
  __int64 v81; // r12
  int v82; // edx
  __int64 v83; // rdx
  __int64 v84; // rax
  int v85; // edx
  __int64 v86; // r13
  __int64 v87; // rax
  __int64 v88; // r15
  bool v89; // r12
  bool v90; // zf
  bool v91; // r12
  __int64 v92; // rax
  bool v93; // r12
  bool v94; // zf
  __int64 v95; // rax
  unsigned int v96; // ecx
  __int64 v97; // rdx
  __int64 v98; // rdi
  unsigned int v99; // edx
  __int64 v100; // rax
  __int64 v101; // rsi
  unsigned int v102; // esi
  __int64 v103; // r12
  __int64 v104; // rbx
  int v105; // edx
  __int64 v106; // rax
  int v107; // r9d
  unsigned int v108; // esi
  __int64 v109; // rbx
  __int64 v110; // r12
  int v111; // edx
  __int64 v112; // rdx
  __int64 v113; // rax
  int v114; // edx
  __int64 v115; // rdi
  unsigned int v116; // r10d
  __int64 v117; // rdx
  __int64 v118; // rcx
  __int64 v119; // rdi
  unsigned int v120; // edx
  __int64 v121; // rax
  __int64 v122; // rcx
  int v123; // r11d
  int v124; // eax
  int v125; // r9d
  int v126; // eax
  char v127; // al
  __int64 v128; // rcx
  int *v129; // rax
  int v130; // eax
  __int64 v131; // r13
  __int64 v132; // r8
  __int64 v133; // rax
  int v134; // esi
  int v135; // edx
  unsigned int v136; // esi
  __int64 v137; // rbx
  char *v138; // r13
  __int64 v139; // rax
  __int64 v140; // rsi
  __int64 v141; // rax
  __int64 v142; // rax
  int v143; // r11d
  int v144; // eax
  int v145; // r10d
  int v146; // eax
  __int64 v147; // rdx
  _QWORD *v148; // rdx
  __int64 v149; // rdx
  _QWORD *v150; // rdx
  unsigned int v151; // esi
  int v152; // eax
  __int64 v153; // r13
  int v154; // eax
  __int64 v155; // r13
  char v156; // al
  __int64 v157; // rcx
  int *v158; // r13
  int v159; // edx
  __int64 v160; // rax
  __int64 v161; // rsi
  int v162; // edx
  __int64 v163; // rax
  __int64 v164; // rsi
  __int64 v165; // r13
  int v166; // eax
  __int64 v167; // r13
  _DWORD *v168; // rax
  __int64 v169; // rax
  _QWORD *v170; // r15
  __int64 v171; // r13
  __int64 m; // rbx
  __int64 v173; // rdx
  __int64 v174; // rax
  __int64 v175; // r13
  __int64 v176; // r13
  __int64 v177; // [rsp+8h] [rbp-128h]
  __int64 v178; // [rsp+8h] [rbp-128h]
  __int64 v179; // [rsp+10h] [rbp-120h]
  char *v180; // [rsp+10h] [rbp-120h]
  char *v181; // [rsp+18h] [rbp-118h]
  __int64 v182; // [rsp+18h] [rbp-118h]
  char *v183; // [rsp+20h] [rbp-110h]
  __int64 v184; // [rsp+28h] [rbp-108h]
  __int64 v185; // [rsp+28h] [rbp-108h]
  __int64 v186; // [rsp+28h] [rbp-108h]
  __int64 v187; // [rsp+28h] [rbp-108h]
  char *v188; // [rsp+28h] [rbp-108h]
  __int64 v189; // [rsp+28h] [rbp-108h]
  __int64 v190; // [rsp+30h] [rbp-100h]
  char *v191; // [rsp+38h] [rbp-F8h]
  __int64 v192; // [rsp+40h] [rbp-F0h]
  unsigned int v193; // [rsp+48h] [rbp-E8h]
  __int64 v194; // [rsp+48h] [rbp-E8h]
  __int64 v195; // [rsp+48h] [rbp-E8h]
  __int64 v196; // [rsp+48h] [rbp-E8h]
  unsigned int v197; // [rsp+48h] [rbp-E8h]
  __int64 v198; // [rsp+48h] [rbp-E8h]
  char *v199; // [rsp+50h] [rbp-E0h]
  int v201; // [rsp+60h] [rbp-D0h]
  int v202; // [rsp+60h] [rbp-D0h]
  int v203; // [rsp+60h] [rbp-D0h]
  int v204; // [rsp+60h] [rbp-D0h]
  __int64 v205; // [rsp+60h] [rbp-D0h]
  __int64 v206; // [rsp+60h] [rbp-D0h]
  __int64 v207; // [rsp+60h] [rbp-D0h]
  __int64 v208; // [rsp+60h] [rbp-D0h]
  char *i; // [rsp+68h] [rbp-C8h]
  __int64 v210; // [rsp+68h] [rbp-C8h]
  __int64 v211; // [rsp+68h] [rbp-C8h]
  int v212; // [rsp+68h] [rbp-C8h]
  __int64 v213; // [rsp+68h] [rbp-C8h]
  __int64 v214; // [rsp+68h] [rbp-C8h]
  __int64 v215; // [rsp+68h] [rbp-C8h]
  int v216; // [rsp+68h] [rbp-C8h]
  __int64 v217; // [rsp+78h] [rbp-B8h] BYREF
  _QWORD v218[2]; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v219; // [rsp+90h] [rbp-A0h]
  __int64 v220[2]; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v221; // [rsp+B0h] [rbp-80h]
  __int64 v222; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v223; // [rsp+C8h] [rbp-68h]
  __int64 v224; // [rsp+D0h] [rbp-60h]
  __int64 v225; // [rsp+D8h] [rbp-58h]
  __int64 v226; // [rsp+E0h] [rbp-50h] BYREF
  __int64 v227; // [rsp+E8h] [rbp-48h]
  __int64 k; // [rsp+F0h] [rbp-40h]
  __int64 v229; // [rsp+F8h] [rbp-38h]

  result = a2 - a1;
  v191 = a2;
  v190 = a3;
  if ( a2 - a1 <= 512 )
    return result;
  v5 = (__int64)a1;
  v6 = a4;
  if ( !a3 )
  {
    v199 = a2;
    goto LABEL_328;
  }
  v7 = a4;
  v183 = a1 + 32;
  v192 = a4 + 728;
  while ( 2 )
  {
    --v190;
    v217 = v7;
    v8 = &a1[32 * (result >> 6)];
    v9 = *((_QWORD *)a1 + 6);
    v10 = *((_QWORD *)a1 + 7);
    v11 = *((_QWORD *)v8 + 2);
    v12 = *((_QWORD *)v8 + 3);
    if ( v11 == v9 )
      goto LABEL_214;
    if ( v12 == v10 )
    {
      v222 = 0;
      v223 = 0;
      v224 = v9;
      if ( v9 != -4096 && v9 != 0 && v9 != -8192 )
      {
        v210 = v11;
        sub_BD73F0((__int64)&v222);
        v11 = v210;
      }
      v211 = v11;
      v127 = sub_2D67BB0(v192, (__int64)&v222, v220);
      v128 = v211;
      if ( v127 )
      {
        v129 = (int *)(v220[0] + 24);
        goto LABEL_223;
      }
      v151 = *(_DWORD *)(v7 + 752);
      v152 = *(_DWORD *)(v7 + 744);
      v153 = v220[0];
      ++*(_QWORD *)(v7 + 728);
      v154 = v152 + 1;
      v226 = v153;
      if ( 4 * v154 >= 3 * v151 )
      {
        v176 = v192;
        sub_2D6E640(v192, 2 * v151);
      }
      else
      {
        if ( v151 - *(_DWORD *)(v7 + 748) - v154 > v151 >> 3 )
        {
LABEL_279:
          *(_DWORD *)(v7 + 744) = v154;
          if ( *(_QWORD *)(v153 + 16) != -4096 )
            --*(_DWORD *)(v7 + 748);
          v213 = v128;
          sub_2D57220((_QWORD *)v153, v224);
          v128 = v213;
          v129 = (int *)(v153 + 24);
          *(_DWORD *)(v153 + 24) = 0;
LABEL_223:
          v130 = *v129;
          k = v128;
          v131 = v217;
          v226 = 0;
          v212 = v130;
          v227 = 0;
          v132 = v217 + 728;
          if ( v128 != 0 && v128 != -4096 && v128 != -8192 )
          {
            v205 = v217 + 728;
            sub_BD73F0((__int64)&v226);
            v132 = v205;
          }
          v206 = v132;
          v90 = (unsigned __int8)sub_2D67BB0(v132, (__int64)&v226, v218) == 0;
          v133 = v218[0];
          if ( !v90 )
            goto LABEL_232;
          v220[0] = v218[0];
          v134 = *(_DWORD *)(v131 + 744);
          ++*(_QWORD *)(v131 + 728);
          v135 = v134 + 1;
          v136 = *(_DWORD *)(v131 + 752);
          if ( 4 * v135 >= 3 * v136 )
          {
            v136 *= 2;
          }
          else if ( v136 - *(_DWORD *)(v131 + 748) - v135 > v136 >> 3 )
          {
LABEL_229:
            *(_DWORD *)(v131 + 744) = v135;
            if ( *(_QWORD *)(v133 + 16) != -4096 )
              --*(_DWORD *)(v131 + 748);
            v207 = v133;
            sub_2D57220((_QWORD *)v133, k);
            v133 = v207;
            *(_DWORD *)(v207 + 24) = 0;
LABEL_232:
            v13 = v212 < *(_DWORD *)(v133 + 24);
            sub_D68D70(&v226);
            sub_D68D70(&v222);
            goto LABEL_7;
          }
          sub_2D6E640(v206, v136);
          sub_2D67BB0(v206, (__int64)&v226, v220);
          v135 = *(_DWORD *)(v131 + 744) + 1;
          v133 = v220[0];
          goto LABEL_229;
        }
        v176 = v192;
        sub_2D6E640(v192, v151);
      }
      sub_2D67BB0(v176, (__int64)&v222, &v226);
      v153 = v226;
      v128 = v211;
      v154 = *(_DWORD *)(v7 + 744) + 1;
      goto LABEL_279;
    }
    v13 = v12 > v10;
LABEL_7:
    if ( v13 )
    {
      v14 = *((_QWORD *)v191 - 2);
      v15 = *((_QWORD *)v191 - 1);
      v16 = *((_QWORD *)v8 + 2);
      if ( v14 != v16 )
      {
        if ( *((_QWORD *)v8 + 3) == v15 )
        {
          v224 = *((_QWORD *)v8 + 2);
          v222 = 0;
          v155 = v217 + 728;
          v223 = 0;
          if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
          {
            v214 = v14;
            sub_BD73F0((__int64)&v222);
            v14 = v214;
          }
          v215 = v14;
          v156 = sub_2D67BB0(v155, (__int64)&v222, &v226);
          v157 = v215;
          if ( v156 )
          {
            v158 = (int *)(v226 + 24);
          }
          else
          {
            v165 = sub_2D6E9C0(v155, (__int64)&v222, v226);
            sub_2D57220((_QWORD *)v165, v224);
            v157 = v215;
            *(_DWORD *)(v165 + 24) = 0;
            v158 = (int *)(v165 + 24);
          }
          v166 = *v158;
          v226 = 0;
          v227 = 0;
          v216 = v166;
          k = v157;
          v167 = v217 + 728;
          if ( v157 != 0 && v157 != -4096 && v157 != -8192 )
            sub_BD73F0((__int64)&v226);
          if ( (unsigned __int8)sub_2D67BB0(v167, (__int64)&v226, v220) )
          {
            v168 = (_DWORD *)(v220[0] + 24);
          }
          else
          {
            v175 = sub_2D6E9C0(v167, (__int64)&v226, v220[0]);
            sub_2D57220((_QWORD *)v175, k);
            *(_DWORD *)(v175 + 24) = 0;
            v168 = (_DWORD *)(v175 + 24);
          }
          v17 = v216 < *v168;
          sub_D68D70(&v226);
          sub_D68D70(&v222);
        }
        else
        {
          v17 = *((_QWORD *)v8 + 3) < v15;
        }
        if ( v17 )
          goto LABEL_12;
        v14 = *((_QWORD *)v191 - 2);
        v15 = *((_QWORD *)v191 - 1);
      }
      v227 = v15;
      v226 = v14;
      v222 = *((_QWORD *)a1 + 6);
      v223 = *((_QWORD *)a1 + 7);
      if ( (unsigned __int8)sub_2D701F0(&v217, &v222, &v226) )
      {
LABEL_249:
        sub_2D85950(a1, (_QWORD *)v191 - 4);
        v142 = *((_QWORD *)a1 + 3);
        *((_QWORD *)a1 + 3) = *((_QWORD *)v191 - 1);
        *((_QWORD *)v191 - 1) = v142;
        v19 = *((_QWORD *)a1 + 3);
        v20 = *((_QWORD *)a1 + 7);
        goto LABEL_13;
      }
LABEL_215:
      sub_2D85950(a1, v183);
      v20 = *((_QWORD *)a1 + 3);
      v19 = *((_QWORD *)a1 + 7);
      *((_QWORD *)a1 + 7) = v20;
      *((_QWORD *)a1 + 3) = v19;
      goto LABEL_13;
    }
    v9 = *((_QWORD *)a1 + 6);
    v10 = *((_QWORD *)a1 + 7);
LABEL_214:
    v223 = v10;
    v222 = v9;
    v226 = *((_QWORD *)v191 - 2);
    v227 = *((_QWORD *)v191 - 1);
    if ( (unsigned __int8)sub_2D701F0(&v217, &v222, &v226) )
      goto LABEL_215;
    v226 = *((_QWORD *)v191 - 2);
    v227 = *((_QWORD *)v191 - 1);
    v222 = *((_QWORD *)v8 + 2);
    v223 = *((_QWORD *)v8 + 3);
    if ( (unsigned __int8)sub_2D701F0(&v217, &v222, &v226) )
      goto LABEL_249;
LABEL_12:
    sub_2D85950(a1, v8);
    v18 = *((_QWORD *)a1 + 3);
    *((_QWORD *)a1 + 3) = *((_QWORD *)v8 + 3);
    *((_QWORD *)v8 + 3) = v18;
    v19 = *((_QWORD *)a1 + 3);
    v20 = *((_QWORD *)a1 + 7);
LABEL_13:
    v21 = v191;
    v22 = *((_QWORD *)a1 + 2);
    for ( i = v183; ; i += 32 )
    {
      v23 = *((_QWORD *)i + 2);
      v199 = i;
      if ( v23 == v22 )
        goto LABEL_19;
      if ( v20 != v19 )
      {
        v24 = v20 < v19;
        goto LABEL_18;
      }
      v222 = 0;
      v223 = 0;
      v224 = v23;
      if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
        sub_BD73F0((__int64)&v222);
      v102 = *(_DWORD *)(v7 + 752);
      if ( !v102 )
      {
        ++*(_QWORD *)(v7 + 728);
        v226 = 0;
LABEL_147:
        v102 *= 2;
LABEL_148:
        sub_2D6E640(v192, v102);
        sub_2D67BB0(v192, (__int64)&v222, &v226);
        v103 = v226;
        v104 = v224;
        v105 = *(_DWORD *)(v7 + 744) + 1;
        goto LABEL_149;
      }
      v104 = v224;
      v119 = *(_QWORD *)(v7 + 736);
      v120 = (v102 - 1) & (((unsigned int)v224 >> 9) ^ ((unsigned int)v224 >> 4));
      v121 = v119 + 32LL * v120;
      v122 = *(_QWORD *)(v121 + 16);
      if ( v224 == v122 )
      {
LABEL_188:
        v107 = *(_DWORD *)(v121 + 24);
        goto LABEL_158;
      }
      v145 = 1;
      v103 = 0;
      while ( v122 != -4096 )
      {
        if ( !v103 && v122 == -8192 )
          v103 = v121;
        v120 = (v102 - 1) & (v145 + v120);
        v121 = v119 + 32LL * v120;
        v122 = *(_QWORD *)(v121 + 16);
        if ( v224 == v122 )
          goto LABEL_188;
        ++v145;
      }
      if ( !v103 )
        v103 = v121;
      v146 = *(_DWORD *)(v7 + 744);
      ++*(_QWORD *)(v7 + 728);
      v105 = v146 + 1;
      v226 = v103;
      if ( 4 * (v146 + 1) >= 3 * v102 )
        goto LABEL_147;
      if ( v102 - *(_DWORD *)(v7 + 748) - v105 <= v102 >> 3 )
        goto LABEL_148;
LABEL_149:
      *(_DWORD *)(v7 + 744) = v105;
      if ( *(_QWORD *)(v103 + 16) == -4096 )
      {
        if ( v104 != -4096 )
          goto LABEL_154;
      }
      else
      {
        --*(_DWORD *)(v7 + 748);
        v106 = *(_QWORD *)(v103 + 16);
        if ( v106 != v104 )
        {
          if ( v106 != -4096 && v106 != 0 && v106 != -8192 )
            sub_BD60C0((_QWORD *)v103);
LABEL_154:
          *(_QWORD *)(v103 + 16) = v104;
          if ( v104 != -4096 && v104 != 0 && v104 != -8192 )
            sub_BD73F0(v103);
        }
      }
      *(_DWORD *)(v103 + 24) = 0;
      v107 = 0;
LABEL_158:
      v226 = 0;
      v227 = 0;
      k = v22;
      if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
      {
        v201 = v107;
        sub_BD73F0((__int64)&v226);
        v107 = v201;
      }
      v108 = *(_DWORD *)(v7 + 752);
      if ( !v108 )
      {
        ++*(_QWORD *)(v7 + 728);
        v220[0] = 0;
        goto LABEL_163;
      }
      v109 = k;
      v115 = *(_QWORD *)(v7 + 736);
      v113 = k;
      v116 = (v108 - 1) & (((unsigned int)k >> 9) ^ ((unsigned int)k >> 4));
      v117 = v115 + 32LL * v116;
      v118 = *(_QWORD *)(v117 + 16);
      if ( k != v118 )
      {
        v143 = 1;
        v110 = 0;
        while ( v118 != -4096 )
        {
          if ( !v110 && v118 == -8192 )
            v110 = v117;
          v116 = (v108 - 1) & (v143 + v116);
          v117 = v115 + 32LL * v116;
          v118 = *(_QWORD *)(v117 + 16);
          if ( k == v118 )
            goto LABEL_186;
          ++v143;
        }
        v144 = *(_DWORD *)(v7 + 744);
        if ( !v110 )
          v110 = v117;
        ++*(_QWORD *)(v7 + 728);
        v111 = v144 + 1;
        v220[0] = v110;
        if ( 4 * (v144 + 1) < 3 * v108 )
        {
          if ( v108 - *(_DWORD *)(v7 + 748) - v111 > v108 >> 3 )
          {
LABEL_165:
            *(_DWORD *)(v7 + 744) = v111;
            if ( *(_QWORD *)(v110 + 16) == -4096 )
            {
              v113 = v109;
              if ( v109 != -4096 )
              {
LABEL_170:
                *(_QWORD *)(v110 + 16) = v109;
                if ( v109 == 0 || v109 == -4096 || v109 == -8192 )
                {
                  v113 = k;
                }
                else
                {
                  v204 = v107;
                  sub_BD73F0(v110);
                  v113 = k;
                  v107 = v204;
                }
              }
            }
            else
            {
              --*(_DWORD *)(v7 + 748);
              v112 = *(_QWORD *)(v110 + 16);
              v113 = v109;
              if ( v112 != v109 )
              {
                if ( v112 != -4096 && v112 != 0 && v112 != -8192 )
                {
                  v203 = v107;
                  sub_BD60C0((_QWORD *)v110);
                  v107 = v203;
                }
                goto LABEL_170;
              }
            }
            *(_DWORD *)(v110 + 24) = 0;
            v114 = 0;
            goto LABEL_174;
          }
          v202 = v107;
LABEL_164:
          sub_2D6E640(v192, v108);
          sub_2D67BB0(v192, (__int64)&v226, v220);
          v109 = k;
          v110 = v220[0];
          v107 = v202;
          v111 = *(_DWORD *)(v7 + 744) + 1;
          goto LABEL_165;
        }
LABEL_163:
        v202 = v107;
        v108 *= 2;
        goto LABEL_164;
      }
LABEL_186:
      v114 = *(_DWORD *)(v117 + 24);
LABEL_174:
      v24 = v107 < v114;
      if ( v113 != -4096 && v113 != 0 && v113 != -8192 )
        sub_BD60C0(&v226);
      if ( v224 != 0 && v224 != -4096 && v224 != -8192 )
        sub_BD60C0(&v222);
      v22 = *((_QWORD *)a1 + 2);
      v19 = *((_QWORD *)a1 + 3);
LABEL_18:
      if ( v24 )
        goto LABEL_14;
LABEL_19:
      v25 = *((_QWORD *)v21 - 2);
      v26 = *((_QWORD *)v21 - 1);
      v27 = v21 - 32;
      if ( v22 != v25 )
      {
        for ( j = a1; ; v19 = *((_QWORD *)j + 3) )
        {
          if ( v26 != v19 )
          {
            v29 = v26 > v19;
LABEL_22:
            if ( !v29 )
              goto LABEL_114;
            goto LABEL_23;
          }
          v219 = v22;
          v218[0] = 0;
          v218[1] = 0;
          if ( v22 != -4096 && v22 != 0 && v22 != -8192 )
            sub_BD73F0((__int64)v218);
          v30 = *(_DWORD *)(v7 + 752);
          v31 = *(_QWORD *)(v7 + 736);
          if ( !v30 )
            break;
          v54 = v219;
          v99 = (v30 - 1) & (((unsigned int)v219 >> 9) ^ ((unsigned int)v219 >> 4));
          v100 = v31 + 32LL * v99;
          v101 = *(_QWORD *)(v100 + 16);
          if ( v219 == v101 )
          {
LABEL_138:
            v57 = *(_DWORD *)(v100 + 24);
            goto LABEL_67;
          }
          v125 = 1;
          v53 = 0;
          while ( v101 != -4096 )
          {
            if ( !v53 && v101 == -8192 )
              v53 = v100;
            v99 = (v30 - 1) & (v125 + v99);
            v100 = v31 + 32LL * v99;
            v101 = *(_QWORD *)(v100 + 16);
            if ( v219 == v101 )
              goto LABEL_138;
            ++v125;
          }
          if ( !v53 )
            v53 = v100;
          v126 = *(_DWORD *)(v7 + 744);
          ++*(_QWORD *)(v7 + 728);
          v55 = v126 + 1;
          v220[0] = v53;
          if ( 4 * (v126 + 1) >= 3 * v30 )
            goto LABEL_31;
          if ( v30 - *(_DWORD *)(v7 + 748) - v55 <= v30 >> 3 )
          {
            sub_2D6E640(v192, v30);
            v52 = v192;
            goto LABEL_57;
          }
LABEL_58:
          *(_DWORD *)(v7 + 744) = v55;
          if ( *(_QWORD *)(v53 + 16) == -4096 )
          {
            if ( v54 != -4096 )
              goto LABEL_63;
          }
          else
          {
            --*(_DWORD *)(v7 + 748);
            v56 = *(_QWORD *)(v53 + 16);
            if ( v56 != v54 )
            {
              if ( v56 != 0 && v56 != -4096 && v56 != -8192 )
              {
                v195 = v53;
                sub_BD60C0((_QWORD *)v53);
                v53 = v195;
              }
LABEL_63:
              *(_QWORD *)(v53 + 16) = v54;
              if ( v54 != 0 && v54 != -4096 && v54 != -8192 )
              {
                v196 = v53;
                sub_BD73F0(v53);
                v53 = v196;
              }
            }
          }
          *(_DWORD *)(v53 + 24) = 0;
          v57 = 0;
LABEL_67:
          v221 = v25;
          v220[0] = 0;
          v220[1] = 0;
          if ( v25 != -4096 && v25 != 0 && v25 != -8192 )
            sub_BD73F0((__int64)v220);
          v58 = *(_DWORD *)(v7 + 752);
          v59 = *(_QWORD *)(v7 + 736);
          if ( !v58 )
          {
            ++*(_QWORD *)(v7 + 728);
            v217 = 0;
LABEL_72:
            v187 = v59;
            v197 = v58;
            v60 = (((((((2 * v58 - 1) | ((unsigned __int64)(2 * v58 - 1) >> 1)) >> 2)
                    | (2 * v58 - 1)
                    | ((unsigned __int64)(2 * v58 - 1) >> 1)) >> 4)
                  | (((2 * v58 - 1) | ((unsigned __int64)(2 * v58 - 1) >> 1)) >> 2)
                  | (2 * v58 - 1)
                  | ((unsigned __int64)(2 * v58 - 1) >> 1)) >> 8)
                | (((((2 * v58 - 1) | ((unsigned __int64)(2 * v58 - 1) >> 1)) >> 2)
                  | (2 * v58 - 1)
                  | ((unsigned __int64)(2 * v58 - 1) >> 1)) >> 4)
                | (((2 * v58 - 1) | ((unsigned __int64)(2 * v58 - 1) >> 1)) >> 2)
                | (2 * v58 - 1)
                | ((unsigned __int64)(2 * v58 - 1) >> 1);
            v61 = ((v60 >> 16) | v60) + 1;
            if ( (unsigned int)v61 < 0x40 )
              LODWORD(v61) = 64;
            *(_DWORD *)(v7 + 752) = v61;
            v62 = (_QWORD *)sub_C7D670(32LL * (unsigned int)v61, 8);
            v63 = v187;
            *(_QWORD *)(v7 + 736) = v62;
            if ( v187 )
            {
              v64 = *(unsigned int *)(v7 + 752);
              v65 = 32LL * v197;
              *(_QWORD *)(v7 + 744) = 0;
              v198 = v65;
              v226 = 0;
              v66 = &v62[4 * v64];
              v227 = 0;
              for ( k = -4096; v66 != v62; v62 += 4 )
              {
                if ( v62 )
                {
                  *v62 = 0;
                  v62[1] = 0;
                  v62[2] = -4096;
                }
              }
              v222 = 0;
              v223 = 0;
              v224 = -4096;
              v226 = 0;
              v227 = 0;
              k = -8192;
              if ( v187 != v187 + v65 )
              {
                v188 = v27;
                v67 = v63 + v65;
                v68 = v63;
                v69 = -4096;
                v180 = j;
                v70 = v7;
                v182 = v63;
                while ( 1 )
                {
                  v71 = *(_QWORD *)(v68 + 16);
                  if ( v69 != v71 )
                  {
                    v69 = k;
                    if ( v71 != k )
                    {
                      v72 = *(_DWORD *)(v70 + 752);
                      if ( !v72 )
                        BUG();
                      v73 = v72 - 1;
                      v74 = *(_QWORD *)(v70 + 736);
                      LODWORD(v75) = v73 & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
                      v76 = v74 + 32LL * (unsigned int)v75;
                      v77 = *(_QWORD *)(v76 + 16);
                      if ( v71 != v77 )
                      {
                        v159 = 1;
                        v160 = 0;
                        while ( v77 != -4096 )
                        {
                          if ( v77 == -8192 && !v160 )
                            v160 = v76;
                          v75 = v73 & (unsigned int)(v75 + v159);
                          v76 = v74 + 32 * v75;
                          v77 = *(_QWORD *)(v76 + 16);
                          if ( v71 == v77 )
                            goto LABEL_86;
                          ++v159;
                        }
                        if ( v160 )
                        {
                          v161 = *(_QWORD *)(v160 + 16);
                          v76 = v160;
                        }
                        else
                        {
                          v161 = *(_QWORD *)(v76 + 16);
                        }
                        if ( v161 != v71 )
                        {
                          if ( v161 != -4096 && v161 != 0 && v161 != -8192 )
                          {
                            v177 = *(_QWORD *)(v68 + 16);
                            sub_BD60C0((_QWORD *)v76);
                            v71 = v177;
                          }
                          *(_QWORD *)(v76 + 16) = v71;
                          if ( v71 != 0 && v71 != -4096 && v71 != -8192 )
                            sub_BD73F0(v76);
                        }
                      }
LABEL_86:
                      *(_DWORD *)(v76 + 24) = *(_DWORD *)(v68 + 24);
                      ++*(_DWORD *)(v70 + 744);
                      v69 = *(_QWORD *)(v68 + 16);
                    }
                  }
                  if ( v69 != 0 && v69 != -4096 && v69 != -8192 )
                    sub_BD60C0((_QWORD *)v68);
                  v68 += 32;
                  if ( v67 == v68 )
                    break;
                  v69 = v224;
                }
                v7 = v70;
                v27 = v188;
                v63 = v182;
                j = v180;
                if ( k == 0 || k == -4096 || k == -8192 )
                {
                  v78 = v224 != -8192 && v224 != 0 && v224 != -4096;
                }
                else
                {
                  sub_BD60C0(&v226);
                  v63 = v182;
                  v78 = v224 != -8192 && v224 != 0 && v224 != -4096;
                }
                if ( v78 )
                {
                  v189 = v63;
                  sub_BD60C0(&v222);
                  v63 = v189;
                }
              }
              sub_C7D6A0(v63, v198, 8);
            }
            else
            {
              v149 = *(unsigned int *)(v7 + 752);
              v226 = 0;
              *(_QWORD *)(v7 + 744) = 0;
              v227 = 0;
              v150 = &v62[4 * v149];
              for ( k = -4096; v150 != v62; v62 += 4 )
              {
                if ( v62 )
                {
                  *v62 = 0;
                  v62[1] = 0;
                  v62[2] = -4096;
                }
              }
            }
            v79 = v192;
LABEL_98:
            sub_2D67BB0(v79, (__int64)v220, &v217);
            v80 = v217;
            v81 = v221;
            v82 = *(_DWORD *)(v7 + 744) + 1;
            goto LABEL_99;
          }
          v81 = v221;
          v84 = v221;
          v96 = (v58 - 1) & (((unsigned int)v221 >> 9) ^ ((unsigned int)v221 >> 4));
          v97 = v59 + 32LL * v96;
          v98 = *(_QWORD *)(v97 + 16);
          if ( v98 == v221 )
          {
LABEL_136:
            v85 = *(_DWORD *)(v97 + 24);
            goto LABEL_108;
          }
          v123 = 1;
          v80 = 0;
          while ( v98 != -4096 )
          {
            if ( !v80 && v98 == -8192 )
              v80 = v97;
            v96 = (v58 - 1) & (v123 + v96);
            v97 = v59 + 32LL * v96;
            v98 = *(_QWORD *)(v97 + 16);
            if ( v221 == v98 )
              goto LABEL_136;
            ++v123;
          }
          v124 = *(_DWORD *)(v7 + 744);
          if ( !v80 )
            v80 = v97;
          ++*(_QWORD *)(v7 + 728);
          v82 = v124 + 1;
          v217 = v80;
          if ( 4 * (v124 + 1) >= 3 * v58 )
            goto LABEL_72;
          if ( v58 - *(_DWORD *)(v7 + 748) - v82 <= v58 >> 3 )
          {
            sub_2D6E640(v192, v58);
            v79 = v192;
            goto LABEL_98;
          }
LABEL_99:
          *(_DWORD *)(v7 + 744) = v82;
          if ( *(_QWORD *)(v80 + 16) == -4096 )
          {
            v84 = v81;
            if ( v81 != -4096 )
              goto LABEL_104;
          }
          else
          {
            --*(_DWORD *)(v7 + 748);
            v83 = *(_QWORD *)(v80 + 16);
            v84 = v81;
            if ( v83 != v81 )
            {
              if ( v83 != -4096 && v83 != 0 && v83 != -8192 )
                sub_BD60C0((_QWORD *)v80);
LABEL_104:
              *(_QWORD *)(v80 + 16) = v81;
              if ( v81 == 0 || v81 == -4096 || v81 == -8192 )
              {
                v84 = v221;
              }
              else
              {
                sub_BD73F0(v80);
                v84 = v221;
              }
            }
          }
          *(_DWORD *)(v80 + 24) = 0;
          v85 = 0;
LABEL_108:
          v29 = v85 > v57;
          if ( v84 != -4096 && v84 != 0 && v84 != -8192 )
            sub_BD60C0(v220);
          if ( v219 == 0 || v219 == -4096 || v219 == -8192 )
            goto LABEL_22;
          sub_BD60C0(v218);
          if ( !v29 )
            goto LABEL_114;
LABEL_23:
          v22 = *((_QWORD *)j + 2);
          v25 = *((_QWORD *)v27 - 2);
          v27 -= 32;
          v26 = *((_QWORD *)v27 + 3);
          if ( v25 == v22 )
            goto LABEL_114;
        }
        ++*(_QWORD *)(v7 + 728);
        v220[0] = 0;
LABEL_31:
        v184 = v31;
        v193 = v30;
        v32 = (((((((2 * v30 - 1) | ((unsigned __int64)(2 * v30 - 1) >> 1)) >> 2)
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
            | ((unsigned __int64)(2 * v30 - 1) >> 1);
        v33 = ((v32 >> 16) | v32) + 1;
        if ( (unsigned int)v33 < 0x40 )
          LODWORD(v33) = 64;
        *(_DWORD *)(v7 + 752) = v33;
        v34 = (_QWORD *)sub_C7D670(32LL * (unsigned int)v33, 8);
        v35 = v184;
        *(_QWORD *)(v7 + 736) = v34;
        if ( v184 )
        {
          v36 = *(unsigned int *)(v7 + 752);
          v37 = 32LL * v193;
          *(_QWORD *)(v7 + 744) = 0;
          v194 = v37;
          v226 = 0;
          v38 = &v34[4 * v36];
          v227 = 0;
          for ( k = -4096; v38 != v34; v34 += 4 )
          {
            if ( v34 )
            {
              *v34 = 0;
              v34[1] = 0;
              v34[2] = -4096;
            }
          }
          v222 = 0;
          v223 = 0;
          v224 = -4096;
          v226 = 0;
          v227 = 0;
          k = -8192;
          if ( v184 != v184 + v37 )
          {
            v185 = v25;
            v39 = v35 + v37;
            v40 = v7;
            v41 = j;
            v181 = v27;
            v42 = -4096;
            v43 = v35;
            v179 = v35;
            while ( 1 )
            {
              v44 = *(_QWORD *)(v43 + 16);
              if ( v44 != v42 )
              {
                v42 = k;
                if ( v44 != k )
                {
                  v45 = *(_DWORD *)(v40 + 752);
                  if ( !v45 )
                    BUG();
                  v46 = v45 - 1;
                  v47 = *(_QWORD *)(v40 + 736);
                  LODWORD(v48) = v46 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
                  v49 = v47 + 32LL * (unsigned int)v48;
                  v50 = *(_QWORD *)(v49 + 16);
                  if ( v44 != v50 )
                  {
                    v162 = 1;
                    v163 = 0;
                    while ( v50 != -4096 )
                    {
                      if ( !v163 && v50 == -8192 )
                        v163 = v49;
                      v48 = v46 & (unsigned int)(v48 + v162);
                      v49 = v47 + 32 * v48;
                      v50 = *(_QWORD *)(v49 + 16);
                      if ( v44 == v50 )
                        goto LABEL_45;
                      ++v162;
                    }
                    if ( v163 )
                    {
                      v164 = *(_QWORD *)(v163 + 16);
                      v49 = v163;
                    }
                    else
                    {
                      v164 = *(_QWORD *)(v49 + 16);
                    }
                    if ( v164 != v44 )
                    {
                      if ( v164 != 0 && v164 != -4096 && v164 != -8192 )
                      {
                        v178 = *(_QWORD *)(v43 + 16);
                        sub_BD60C0((_QWORD *)v49);
                        v44 = v178;
                      }
                      *(_QWORD *)(v49 + 16) = v44;
                      if ( v44 != 0 && v44 != -4096 && v44 != -8192 )
                        sub_BD73F0(v49);
                    }
                  }
LABEL_45:
                  *(_DWORD *)(v49 + 24) = *(_DWORD *)(v43 + 24);
                  ++*(_DWORD *)(v40 + 744);
                  v42 = *(_QWORD *)(v43 + 16);
                }
              }
              if ( v42 != 0 && v42 != -4096 && v42 != -8192 )
                sub_BD60C0((_QWORD *)v43);
              v43 += 32;
              if ( v39 == v43 )
                break;
              v42 = v224;
            }
            j = v41;
            v27 = v181;
            v7 = v40;
            v35 = v179;
            v25 = v185;
            if ( k == -4096 || k == 0 || k == -8192 )
            {
              v51 = v224 != -8192 && v224 != -4096 && v224 != 0;
            }
            else
            {
              sub_BD60C0(&v226);
              v35 = v179;
              v51 = v224 != -8192 && v224 != -4096 && v224 != 0;
            }
            if ( v51 )
            {
              v186 = v35;
              sub_BD60C0(&v222);
              v35 = v186;
            }
          }
          sub_C7D6A0(v35, v194, 8);
        }
        else
        {
          v147 = *(unsigned int *)(v7 + 752);
          v226 = 0;
          *(_QWORD *)(v7 + 744) = 0;
          v227 = 0;
          v148 = &v34[4 * v147];
          for ( k = -4096; v148 != v34; v34 += 4 )
          {
            if ( v34 )
            {
              *v34 = 0;
              v34[1] = 0;
              v34[2] = -4096;
            }
          }
        }
        v52 = v192;
LABEL_57:
        sub_2D67BB0(v52, (__int64)v218, v220);
        v53 = v220[0];
        v54 = v219;
        v55 = *(_DWORD *)(v7 + 744) + 1;
        goto LABEL_58;
      }
LABEL_114:
      if ( i >= v27 )
        break;
      v226 = 0;
      v227 = 0;
      k = *((_QWORD *)i + 2);
      if ( k != -4096 && k != 0 && k != -8192 )
      {
        sub_BD6050((unsigned __int64 *)&v226, *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL);
        v86 = *((_QWORD *)v27 + 2);
        v87 = *((_QWORD *)i + 2);
        if ( v86 == v87 )
        {
          v88 = k;
          v89 = k != -4096;
          v90 = k == 0;
        }
        else
        {
          if ( v87 != -4096 && v87 != 0 && v87 != -8192 )
            sub_BD60C0(i);
LABEL_121:
          *((_QWORD *)i + 2) = v86;
          if ( v86 != 0 && v86 != -4096 && v86 != -8192 )
            sub_BD73F0((__int64)i);
          v88 = k;
          v86 = *((_QWORD *)v27 + 2);
          v89 = k != 0;
          v90 = k == -4096;
        }
        v91 = v88 != -8192 && !v90 && v89;
        if ( v86 != v88 )
        {
          if ( v86 != 0 && v86 != -4096 && v86 != -8192 )
            sub_BD60C0(v27);
          *((_QWORD *)v27 + 2) = v88;
          if ( v91 )
          {
            sub_BD73F0((__int64)v27);
            v92 = k;
            v93 = k != -4096;
            v94 = k == 0;
          }
          else
          {
            v92 = k;
            v93 = k != 0;
            v94 = k == -4096;
          }
          v91 = v92 != -8192 && !v94 && v93;
        }
        if ( v91 )
          sub_BD60C0(&v226);
        goto LABEL_134;
      }
      v86 = *((_QWORD *)v27 + 2);
      if ( k != v86 )
        goto LABEL_121;
LABEL_134:
      v21 = v27;
      v95 = *((_QWORD *)i + 3);
      *((_QWORD *)i + 3) = *((_QWORD *)v27 + 3);
      *((_QWORD *)v27 + 3) = v95;
      v22 = *((_QWORD *)a1 + 2);
      v19 = *((_QWORD *)a1 + 3);
LABEL_14:
      v20 = *((_QWORD *)i + 7);
    }
    sub_2D85AC0(i, v191, v190, v7);
    result = i - a1;
    if ( i - a1 <= 512 )
      return result;
    if ( v190 )
    {
      v191 = i;
      continue;
    }
    break;
  }
  v6 = v7;
  v5 = (__int64)a1;
LABEL_328:
  v208 = result >> 5;
  v169 = ((result >> 5) - 2) >> 1;
  v170 = (_QWORD *)(v5 + 32 * v169);
  v171 = v6;
  for ( m = v169; ; --m )
  {
    v174 = v170[2];
    v222 = 0;
    v223 = 0;
    v224 = v174;
    if ( v174 == 0 || v174 == -4096 || v174 == -8192 )
    {
      v173 = v170[3];
      v226 = 0;
      v227 = 0;
      v225 = v173;
      k = v174;
    }
    else
    {
      sub_BD6050((unsigned __int64 *)&v222, *v170 & 0xFFFFFFFFFFFFFFF8LL);
      v173 = v170[3];
      v226 = 0;
      v227 = 0;
      v225 = v173;
      k = v224;
      if ( v224 != 0 && v224 != -8192 && v224 != -4096 )
      {
        sub_BD6050((unsigned __int64 *)&v226, v222 & 0xFFFFFFFFFFFFFFF8LL);
        v173 = v225;
      }
    }
    v229 = v173;
    sub_2D6F650(v5, m, v208, &v226, v171);
    if ( k != 0 && k != -4096 && k != -8192 )
      sub_BD60C0(&v226);
    if ( !m )
      break;
    if ( v224 != -4096 && v224 != 0 && v224 != -8192 )
      sub_BD60C0(&v222);
    v170 -= 4;
  }
  v137 = v171;
  sub_D68D70(&v222);
  v138 = v199;
  do
  {
    v139 = *((_QWORD *)v138 - 2);
    v138 -= 32;
    v222 = 0;
    v223 = 0;
    v224 = v139;
    if ( v139 != -4096 && v139 != 0 && v139 != -8192 )
      sub_BD6050((unsigned __int64 *)&v222, *(_QWORD *)v138 & 0xFFFFFFFFFFFFFFF8LL);
    v140 = *(_QWORD *)(v5 + 16);
    v225 = *((_QWORD *)v138 + 3);
    sub_2D57220(v138, v140);
    v141 = *(_QWORD *)(v5 + 24);
    v226 = 0;
    v227 = 0;
    *((_QWORD *)v138 + 3) = v141;
    k = v224;
    if ( v224 != -4096 && v224 != 0 && v224 != -8192 )
      sub_BD6050((unsigned __int64 *)&v226, v222 & 0xFFFFFFFFFFFFFFF8LL);
    v229 = v225;
    sub_2D6F650(v5, 0, (__int64)&v138[-v5] >> 5, &v226, v137);
    if ( k != 0 && k != -4096 && k != -8192 )
      sub_BD60C0(&v226);
    result = v224;
    if ( v224 != -4096 && v224 != 0 && v224 != -8192 )
      result = sub_BD60C0(&v222);
  }
  while ( (__int64)&v138[-v5] > 32 );
  return result;
}
