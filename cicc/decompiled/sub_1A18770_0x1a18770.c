// Function: sub_1A18770
// Address: 0x1a18770
//
_BOOL8 __fastcall sub_1A18770(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // r14
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r13
  __int64 v17; // rsi
  __int64 v18; // r12
  __int64 v19; // rbx
  _QWORD *v20; // rax
  char v21; // dl
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 k; // r12
  __int64 v25; // r12
  __int64 v26; // rbx
  unsigned __int64 v27; // rax
  __int64 *v28; // rax
  __int64 v29; // rdi
  __int64 v30; // r14
  char v31; // r13
  __int64 v32; // rsi
  char v33; // al
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  __int64 v39; // r14
  __int64 v40; // rsi
  __int64 v41; // rcx
  int v42; // r8d
  int v43; // r9d
  double v44; // xmm4_8
  double v45; // xmm5_8
  __int64 v46; // r12
  _QWORD *v47; // rax
  __int64 v48; // rbx
  _QWORD *v49; // rcx
  __int64 v50; // rdx
  __int64 v51; // r14
  __int64 v52; // rbx
  __int64 v53; // r15
  __int64 v54; // rax
  __int64 v55; // rcx
  _QWORD *v56; // r12
  char v57; // al
  __int64 *v58; // r12
  __int64 *v59; // rbx
  __int64 v60; // rdi
  __int64 v61; // rax
  __int64 v62; // rsi
  __int64 v63; // rdi
  __int64 v64; // rax
  unsigned __int64 v65; // r14
  _QWORD *v66; // r12
  __int64 v67; // r15
  _QWORD *v68; // rbx
  _QWORD *v69; // rax
  char v70; // al
  char v71; // al
  _QWORD *v72; // rdx
  __int64 v73; // rdx
  __int64 *v74; // rax
  __int64 v75; // rsi
  unsigned __int64 v76; // rcx
  __int64 v77; // rcx
  unsigned __int64 *v78; // rcx
  unsigned __int64 v79; // rdx
  __int64 *v80; // rax
  __int64 *v81; // rax
  __int64 v82; // rbx
  __int64 *v83; // rax
  int v84; // r14d
  __int64 v85; // r13
  unsigned __int64 v86; // rax
  unsigned int v87; // r12d
  int v88; // edi
  unsigned __int64 v89; // rdx
  unsigned __int64 v90; // rdx
  unsigned __int64 v91; // rbx
  __int64 *v92; // rdx
  unsigned __int64 v93; // rbx
  unsigned int i; // ecx
  __int64 *v95; // rax
  __int64 v96; // r8
  __int64 v97; // rax
  __int64 *v98; // rax
  __int64 v99; // rax
  __int64 v100; // rax
  _QWORD *v101; // rax
  unsigned __int64 v102; // rdi
  __int64 *v103; // rdx
  __int64 *v104; // r14
  __int64 *v105; // rax
  __int64 v106; // r12
  __int64 *v107; // rbx
  __int64 v108; // r12
  __int64 v109; // rbx
  __int64 v110; // r13
  __int64 v111; // rdx
  __int64 *v112; // rax
  __int64 v113; // rsi
  unsigned __int64 v114; // rcx
  __int64 v115; // rsi
  int v117; // r10d
  unsigned int v118; // esi
  int v119; // ecx
  unsigned __int64 v120; // rbx
  unsigned int v121; // r13d
  __int64 v122; // rdx
  int v123; // r15d
  unsigned __int64 v124; // rdx
  unsigned __int64 v125; // rdx
  unsigned int ii; // eax
  unsigned int v127; // eax
  __int64 *v128; // rax
  __int64 *v129; // rdi
  _QWORD *v130; // rbx
  _QWORD *v131; // rax
  __int64 v132; // rdx
  _QWORD *n; // r12
  __int64 v134; // r8
  __int64 *v135; // rax
  __int64 v136; // rdi
  int v137; // ecx
  __int64 *v138; // rdx
  int v139; // edx
  __int64 *v140; // rcx
  unsigned int v141; // ebx
  int v142; // esi
  __int64 v143; // r8
  __int64 *v144; // rsi
  __int64 *v145; // rcx
  __int64 *v146; // rax
  __int64 *v147; // rbx
  __int64 v148; // r12
  __int64 *v149; // r13
  _QWORD *v150; // rax
  __int64 v151; // rdi
  unsigned __int64 *v152; // rcx
  unsigned __int64 v153; // rdx
  double v154; // xmm4_8
  double v155; // xmm5_8
  __int64 *v156; // rax
  __int64 *v157; // r14
  __int64 v158; // r13
  __int64 *v159; // r12
  unsigned int v160; // ecx
  int v161; // edi
  unsigned __int64 v162; // rsi
  unsigned __int64 v163; // rsi
  unsigned __int64 v164; // rax
  __int64 *v165; // rsi
  unsigned int v166; // edx
  __int64 v167; // r8
  unsigned int v168; // edx
  int v169; // ecx
  __int64 v170; // rdx
  unsigned int v171; // ecx
  __int64 v172; // r9
  int v173; // edi
  __int64 *v174; // rsi
  int v175; // edx
  int v176; // esi
  unsigned int j; // ebx
  __int64 *v178; // rcx
  __int64 v179; // rdi
  unsigned int v180; // ebx
  __int64 v181; // rcx
  __int64 v182; // r11
  int v183; // edi
  __int64 *v184; // rcx
  unsigned int v185; // r13d
  __int64 v186; // rdi
  unsigned int v187; // r10d
  __int64 v188; // [rsp+10h] [rbp-1B10h]
  __int64 v189; // [rsp+18h] [rbp-1B08h]
  __int64 m; // [rsp+20h] [rbp-1B00h]
  __int64 v191; // [rsp+30h] [rbp-1AF0h]
  __int64 v193; // [rsp+38h] [rbp-1AE8h]
  __int64 v194; // [rsp+40h] [rbp-1AE0h]
  __int64 v195; // [rsp+48h] [rbp-1AD8h]
  _QWORD *v196; // [rsp+48h] [rbp-1AD8h]
  bool v197; // [rsp+48h] [rbp-1AD8h]
  __int64 v198; // [rsp+48h] [rbp-1AD8h]
  __int64 *v199; // [rsp+48h] [rbp-1AD8h]
  bool v200; // [rsp+50h] [rbp-1AD0h]
  unsigned __int64 v201; // [rsp+50h] [rbp-1AD0h]
  char v202; // [rsp+58h] [rbp-1AC8h]
  __int64 v203; // [rsp+58h] [rbp-1AC8h]
  int v204; // [rsp+58h] [rbp-1AC8h]
  _BYTE *v205; // [rsp+60h] [rbp-1AC0h] BYREF
  __int64 v206; // [rsp+68h] [rbp-1AB8h]
  _BYTE v207[64]; // [rsp+70h] [rbp-1AB0h] BYREF
  __int64 v208; // [rsp+B0h] [rbp-1A70h] BYREF
  __int64 *v209; // [rsp+B8h] [rbp-1A68h]
  __int64 *v210; // [rsp+C0h] [rbp-1A60h]
  int v211; // [rsp+C8h] [rbp-1A58h]
  int v212; // [rsp+CCh] [rbp-1A54h]
  char v213[136]; // [rsp+D8h] [rbp-1A48h] BYREF
  _BYTE v214[16]; // [rsp+160h] [rbp-19C0h] BYREF
  __int64 v215; // [rsp+170h] [rbp-19B0h] BYREF
  _QWORD *v216; // [rsp+178h] [rbp-19A8h]
  _QWORD *v217; // [rsp+180h] [rbp-19A0h]
  unsigned int v218; // [rsp+188h] [rbp-1998h]
  unsigned int v219; // [rsp+18Ch] [rbp-1994h]
  int v220; // [rsp+190h] [rbp-1990h]
  __int64 v221; // [rsp+238h] [rbp-18E8h] BYREF
  __int64 *v222; // [rsp+240h] [rbp-18E0h]
  unsigned int v223; // [rsp+248h] [rbp-18D8h]
  int v224; // [rsp+24Ch] [rbp-18D4h]
  unsigned int v225; // [rsp+250h] [rbp-18D0h]
  __int64 v226; // [rsp+258h] [rbp-18C8h] BYREF
  __int64 *v227; // [rsp+260h] [rbp-18C0h]
  int v228; // [rsp+268h] [rbp-18B8h]
  int v229; // [rsp+26Ch] [rbp-18B4h]
  unsigned int v230; // [rsp+270h] [rbp-18B0h]
  __int64 v231; // [rsp+278h] [rbp-18A8h] BYREF
  __int64 v232; // [rsp+280h] [rbp-18A0h]
  int v233; // [rsp+288h] [rbp-1898h]
  int v234; // [rsp+28Ch] [rbp-1894h]
  unsigned int v235; // [rsp+290h] [rbp-1890h]
  __int64 v236; // [rsp+298h] [rbp-1888h] BYREF
  __int64 *v237; // [rsp+2A0h] [rbp-1880h]
  __int64 *v238; // [rsp+2A8h] [rbp-1878h]
  unsigned int v239; // [rsp+2B0h] [rbp-1870h]
  unsigned int v240; // [rsp+2B4h] [rbp-186Ch]
  int v241; // [rsp+2B8h] [rbp-1868h]
  _BYTE v242[168]; // [rsp+340h] [rbp-17E0h] BYREF
  __int64 v243; // [rsp+3E8h] [rbp-1738h] BYREF
  __int64 *v244; // [rsp+3F0h] [rbp-1730h]
  __int64 *v245; // [rsp+3F8h] [rbp-1728h]
  unsigned int v246; // [rsp+400h] [rbp-1720h]
  unsigned int v247; // [rsp+404h] [rbp-171Ch]
  int v248; // [rsp+408h] [rbp-1718h]
  __int64 v249; // [rsp+8B0h] [rbp-1270h] BYREF
  unsigned int v250; // [rsp+8B8h] [rbp-1268h]
  unsigned int v251; // [rsp+8BCh] [rbp-1264h]
  _BYTE v252[544]; // [rsp+8C0h] [rbp-1260h] BYREF
  __int64 *v253; // [rsp+AE0h] [rbp-1040h] BYREF
  __int64 v254; // [rsp+AE8h] [rbp-1038h]
  _BYTE v255[4144]; // [rsp+AF0h] [rbp-1030h] BYREF

  v11 = a1;
  sub_1A0F500((__int64)v214, a2, a3);
  v16 = *(_QWORD *)(a1 + 32);
  v17 = (__int64)&v231;
  v189 = a1 + 24;
  if ( v16 != a1 + 24 )
  {
    while ( 1 )
    {
      v18 = v16 - 56;
      if ( !v16 )
        v18 = 0;
      if ( sub_15E4F60(v18) )
        goto LABEL_15;
      if ( !(unsigned __int8)sub_387E010(v18, v17) )
        goto LABEL_6;
      v81 = *(__int64 **)(*(_QWORD *)(v18 + 24) + 16LL);
      v82 = *v81;
      if ( *(_BYTE *)(*v81 + 8) == 13 )
      {
        v83 = v237;
        if ( v238 != v237 )
          goto LABEL_106;
        v144 = &v237[v240];
        if ( v237 == v144 )
          goto LABEL_308;
        v145 = 0;
        do
        {
          if ( v18 == *v83 )
            goto LABEL_107;
          if ( *v83 == -2 )
            v145 = v83;
          ++v83;
        }
        while ( v144 != v83 );
        if ( !v145 )
        {
LABEL_308:
          if ( v240 >= v239 )
          {
LABEL_106:
            sub_16CCBA0((__int64)&v236, v18);
          }
          else
          {
            ++v240;
            *v144 = v18;
            ++v236;
          }
        }
        else
        {
          *v145 = v18;
          --v241;
          ++v236;
        }
LABEL_107:
        v204 = *(_DWORD *)(v82 + 12);
        if ( !v204 )
          goto LABEL_6;
        v191 = v16;
        v84 = 0;
        v85 = v18;
        v86 = (unsigned __int64)(((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)) << 32;
        v87 = 0;
        v201 = v86;
LABEL_109:
        if ( !v235 )
        {
          ++v231;
          goto LABEL_299;
        }
        v88 = 1;
        v89 = (((v87 | v201) - 1 - ((unsigned __int64)v87 << 32)) >> 22)
            ^ ((v87 | v201) - 1 - ((unsigned __int64)v87 << 32));
        v90 = ((9 * (((v89 - 1 - (v89 << 13)) >> 8) ^ (v89 - 1 - (v89 << 13)))) >> 15)
            ^ (9 * (((v89 - 1 - (v89 << 13)) >> 8) ^ (v89 - 1 - (v89 << 13))));
        v91 = v90 - 1 - (v90 << 27);
        v92 = 0;
        v93 = (v91 >> 31) ^ v91;
        for ( i = v93 & (v235 - 1); ; i = (v235 - 1) & v160 )
        {
          v95 = (__int64 *)(v232 + 24LL * i);
          v96 = *v95;
          if ( v85 == *v95 && *((_DWORD *)v95 + 2) == v84 )
            goto LABEL_248;
          if ( v96 == -8 )
          {
            if ( *((_DWORD *)v95 + 2) == -1 )
            {
              if ( v92 )
                v95 = v92;
              ++v231;
              v175 = v233 + 1;
              if ( 4 * (v233 + 1) < 3 * v235 )
              {
                if ( v235 - v234 - v175 > v235 >> 3 )
                  goto LABEL_339;
                sub_1A102D0((__int64)&v231, v235);
                if ( v235 )
                {
                  v176 = 1;
                  v95 = 0;
                  for ( j = (v235 - 1) & v93; ; j = (v235 - 1) & v180 )
                  {
                    v178 = (__int64 *)(v232 + 24LL * j);
                    v179 = *v178;
                    if ( v85 == *v178 && *((_DWORD *)v178 + 2) == v84 )
                    {
                      v175 = v233 + 1;
                      v95 = (__int64 *)(v232 + 24LL * j);
                      goto LABEL_339;
                    }
                    if ( v179 == -8 )
                    {
                      if ( *((_DWORD *)v178 + 2) == -1 )
                      {
                        if ( !v95 )
                          v95 = (__int64 *)(v232 + 24LL * j);
                        v175 = v233 + 1;
                        goto LABEL_339;
                      }
                    }
                    else if ( v179 == -16 && *((_DWORD *)v178 + 2) == -2 && !v95 )
                    {
                      v95 = (__int64 *)(v232 + 24LL * j);
                    }
                    v180 = v176 + j;
                    ++v176;
                  }
                }
LABEL_410:
                ++v233;
                BUG();
              }
LABEL_299:
              sub_1A102D0((__int64)&v231, 2 * v235);
              if ( !v235 )
                goto LABEL_410;
              v161 = 1;
              v162 = (((v87 | v201) - 1 - ((unsigned __int64)v87 << 32)) >> 22)
                   ^ ((v87 | v201) - 1 - ((unsigned __int64)v87 << 32));
              v163 = ((9 * (((v162 - 1 - (v162 << 13)) >> 8) ^ (v162 - 1 - (v162 << 13)))) >> 15)
                   ^ (9 * (((v162 - 1 - (v162 << 13)) >> 8) ^ (v162 - 1 - (v162 << 13))));
              v164 = ((v163 - 1 - (v163 << 27)) >> 31) ^ (v163 - 1 - (v163 << 27));
              v165 = 0;
              v166 = (v235 - 1) & v164;
              while ( 2 )
              {
                v95 = (__int64 *)(v232 + 24LL * v166);
                v167 = *v95;
                if ( v85 == *v95 && *((_DWORD *)v95 + 2) == v84 )
                {
                  v175 = v233 + 1;
                  goto LABEL_339;
                }
                if ( v167 != -8 )
                {
                  if ( v167 == -16 && *((_DWORD *)v95 + 2) == -2 && !v165 )
                    v165 = (__int64 *)(v232 + 24LL * v166);
                  goto LABEL_307;
                }
                if ( *((_DWORD *)v95 + 2) != -1 )
                {
LABEL_307:
                  v168 = v161 + v166;
                  ++v161;
                  v166 = (v235 - 1) & v168;
                  continue;
                }
                break;
              }
              if ( v165 )
                v95 = v165;
              v175 = v233 + 1;
LABEL_339:
              v233 = v175;
              if ( *v95 != -8 || *((_DWORD *)v95 + 2) != -1 )
                --v234;
              *v95 = v85;
              *((_DWORD *)v95 + 2) = v84;
              v95[2] = 0;
LABEL_248:
              ++v84;
              v87 += 37;
              if ( v204 == v84 )
              {
                v18 = v85;
                v16 = v191;
                goto LABEL_6;
              }
              goto LABEL_109;
            }
          }
          else if ( v96 == -16 && *((_DWORD *)v95 + 2) == -2 && !v92 )
          {
            v92 = (__int64 *)(v232 + 24LL * i);
          }
          v160 = v88 + i;
          ++v88;
        }
      }
      if ( !v230 )
        break;
      LODWORD(v134) = (v230 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v135 = &v227[2 * (unsigned int)v134];
      v136 = *v135;
      if ( v18 == *v135 )
        goto LABEL_6;
      v137 = 1;
      v138 = 0;
      while ( v136 != -8 )
      {
        if ( !v138 && v136 == -16 )
          v138 = v135;
        v134 = (v230 - 1) & ((_DWORD)v134 + v137);
        v135 = &v227[2 * v134];
        v136 = *v135;
        if ( v18 == *v135 )
          goto LABEL_6;
        ++v137;
      }
      if ( v138 )
        v135 = v138;
      ++v226;
      v139 = v228 + 1;
      if ( 4 * (v228 + 1) >= 3 * v230 )
        goto LABEL_322;
      if ( v230 - v229 - v139 <= v230 >> 3 )
      {
        sub_1A0FCC0((__int64)&v226, v230);
        if ( !v230 )
          goto LABEL_411;
        v140 = 0;
        v141 = (v230 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v139 = v228 + 1;
        v142 = 1;
        v135 = &v227[2 * v141];
        v143 = *v135;
        if ( v18 != *v135 )
        {
          while ( v143 != -8 )
          {
            if ( v143 == -16 && !v140 )
              v140 = v135;
            v141 = (v230 - 1) & (v142 + v141);
            v135 = &v227[2 * v141];
            v143 = *v135;
            if ( v18 == *v135 )
              goto LABEL_324;
            ++v142;
          }
          if ( v140 )
            v135 = v140;
        }
      }
LABEL_324:
      v228 = v139;
      if ( *v135 != -8 )
        --v229;
      *v135 = v18;
      v135[1] = 0;
LABEL_6:
      if ( !(unsigned __int8)sub_387DFE0(v18) )
      {
        v19 = *(_QWORD *)(v18 + 80);
        if ( v19 )
          v19 -= 24;
        v20 = v216;
        if ( v217 == v216 )
        {
          v17 = (__int64)&v216[v219];
          if ( v216 == (_QWORD *)v17 )
          {
LABEL_291:
            if ( v219 >= v218 )
              goto LABEL_10;
            ++v219;
            *(_QWORD *)v17 = v19;
            ++v215;
            goto LABEL_126;
          }
          v13 = 0;
          while ( v19 != *v20 )
          {
            if ( *v20 == -2 )
              v13 = (__int64)v20;
            if ( (_QWORD *)v17 == ++v20 )
            {
              if ( !v13 )
                goto LABEL_291;
              *(_QWORD *)v13 = v19;
              --v220;
              ++v215;
              goto LABEL_126;
            }
          }
LABEL_11:
          if ( (*(_BYTE *)(v18 + 18) & 1) == 0 )
            goto LABEL_12;
LABEL_129:
          sub_15E08E0(v18, v17);
          v22 = *(_QWORD *)(v18 + 88);
          v12 = 5LL * *(_QWORD *)(v18 + 96);
          v23 = v22 + 40LL * *(_QWORD *)(v18 + 96);
          if ( (*(_BYTE *)(v18 + 18) & 1) != 0 )
          {
            sub_15E08E0(v18, v17);
            v22 = *(_QWORD *)(v18 + 88);
          }
        }
        else
        {
LABEL_10:
          v17 = v19;
          sub_16CCBA0((__int64)&v215, v19);
          if ( !v21 )
            goto LABEL_11;
LABEL_126:
          v97 = v250;
          if ( v250 >= v251 )
          {
            v17 = (__int64)v252;
            sub_16CD150((__int64)&v249, v252, 0, 8, v14, v15);
            v97 = v250;
          }
          *(_QWORD *)(v249 + 8 * v97) = v19;
          ++v250;
          if ( (*(_BYTE *)(v18 + 18) & 1) != 0 )
            goto LABEL_129;
LABEL_12:
          v22 = *(_QWORD *)(v18 + 88);
          v12 = 5LL * *(_QWORD *)(v18 + 96);
          v23 = v22 + 40LL * *(_QWORD *)(v18 + 96);
        }
        for ( k = v22; v23 != k; k += 40 )
        {
          v17 = k;
          sub_1A11830((__int64)v214, v17);
        }
        goto LABEL_15;
      }
      v80 = v244;
      if ( v245 == v244 )
      {
        v12 = v247;
        v129 = &v244[v247];
        v17 = v247;
        if ( v244 == v129 )
          goto LABEL_295;
        v13 = 0;
        do
        {
          v12 = *v80;
          if ( v18 == *v80 )
            goto LABEL_15;
          if ( v12 == -2 )
            v13 = (__int64)v80;
          ++v80;
        }
        while ( v129 != v80 );
        if ( !v13 )
        {
LABEL_295:
          if ( v247 >= v246 )
            goto LABEL_102;
          v17 = ++v247;
          *v129 = v18;
          ++v243;
LABEL_15:
          v16 = *(_QWORD *)(v16 + 8);
          if ( v189 == v16 )
            goto LABEL_16;
        }
        else
        {
          *(_QWORD *)v13 = v18;
          --v248;
          ++v243;
          v16 = *(_QWORD *)(v16 + 8);
          if ( v189 == v16 )
          {
LABEL_16:
            v11 = a1;
            goto LABEL_17;
          }
        }
      }
      else
      {
LABEL_102:
        v17 = v18;
        sub_16CCBA0((__int64)&v243, v18);
        v16 = *(_QWORD *)(v16 + 8);
        if ( v189 == v16 )
          goto LABEL_16;
      }
    }
    ++v226;
LABEL_322:
    sub_1A0FCC0((__int64)&v226, 2 * v230);
    if ( !v230 )
    {
LABEL_411:
      ++v228;
      BUG();
    }
    v139 = v228 + 1;
    v171 = (v230 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
    v135 = &v227[2 * v171];
    v172 = *v135;
    if ( v18 != *v135 )
    {
      v173 = 1;
      v174 = 0;
      while ( v172 != -8 )
      {
        if ( v172 == -16 && !v174 )
          v174 = v135;
        v171 = (v230 - 1) & (v173 + v171);
        v135 = &v227[2 * v171];
        v172 = *v135;
        if ( v18 == *v135 )
          goto LABEL_324;
        ++v173;
      }
      if ( v174 )
        v135 = v174;
    }
    goto LABEL_324;
  }
LABEL_17:
  v25 = *(_QWORD *)(v11 + 16);
  v188 = v11 + 8;
  if ( v25 != v11 + 8 )
  {
    while ( 1 )
    {
      v26 = v25 - 56;
      if ( !v25 )
        v26 = 0;
      sub_159D9E0(v26);
      if ( !(unsigned __int8)sub_387E060(v26) )
        goto LABEL_19;
      v27 = *(unsigned __int8 *)(*(_QWORD *)(v26 + 24) + 8LL);
      if ( (unsigned __int8)v27 > 0x10u )
        goto LABEL_19;
      v12 = 100990;
      if ( !_bittest64(&v12, v27) )
        goto LABEL_19;
      v17 = v225;
      if ( !v225 )
        break;
      v15 = (__int64)v222;
      v14 = (v225 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v28 = &v222[2 * v14];
      v29 = *v28;
      if ( v26 != *v28 )
      {
        v169 = 1;
        v12 = 0;
        while ( v29 != -8 )
        {
          if ( v29 == -16 && !v12 )
            v12 = (__int64)v28;
          v14 = (v225 - 1) & (v169 + (_DWORD)v14);
          v28 = &v222[2 * (unsigned int)v14];
          v29 = *v28;
          if ( v26 == *v28 )
            goto LABEL_27;
          ++v169;
        }
        if ( v12 )
          v28 = (__int64 *)v12;
        ++v221;
        v12 = v223 + 1;
        if ( 4 * (int)v12 < 3 * v225 )
        {
          if ( v225 - v224 - (unsigned int)v12 <= v225 >> 3 )
          {
            sub_1A0FB10((__int64)&v221, v225);
            if ( !v225 )
            {
LABEL_409:
              ++v223;
              BUG();
            }
            v14 = v225 - 1;
            v15 = (__int64)v222;
            v184 = 0;
            v185 = v14 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
            v12 = v223 + 1;
            v17 = 1;
            v28 = &v222[2 * v185];
            v186 = *v28;
            if ( v26 != *v28 )
            {
              while ( v186 != -8 )
              {
                if ( v186 == -16 && !v184 )
                  v184 = v28;
                v187 = v17 + 1;
                v17 = (unsigned int)v14 & (v185 + (_DWORD)v17);
                v185 = v17;
                v28 = &v222[2 * (unsigned int)v17];
                v186 = *v28;
                if ( v26 == *v28 )
                  goto LABEL_316;
                v17 = v187;
              }
              if ( v184 )
                v28 = v184;
            }
          }
          goto LABEL_316;
        }
LABEL_371:
        sub_1A0FB10((__int64)&v221, 2 * v225);
        if ( !v225 )
          goto LABEL_409;
        v14 = v225 - 1;
        v15 = (__int64)v222;
        v17 = v223;
        LODWORD(v181) = v14 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v12 = v223 + 1;
        v28 = &v222[2 * (unsigned int)v181];
        v182 = *v28;
        if ( v26 != *v28 )
        {
          v183 = 1;
          v17 = 0;
          while ( v182 != -8 )
          {
            if ( !v17 && v182 == -16 )
              v17 = (__int64)v28;
            v181 = (unsigned int)v14 & ((_DWORD)v181 + v183);
            v28 = &v222[2 * v181];
            v182 = *v28;
            if ( v26 == *v28 )
              goto LABEL_316;
            ++v183;
          }
          if ( v17 )
            v28 = (__int64 *)v17;
        }
LABEL_316:
        v223 = v12;
        if ( *v28 != -8 )
          --v224;
        *v28 = v26;
        v28[1] = 0;
        v13 = *(_QWORD *)(v26 - 24);
        if ( *(_BYTE *)(v13 + 16) == 9 )
          goto LABEL_19;
        v170 = 0;
        goto LABEL_320;
      }
LABEL_27:
      v13 = *(_QWORD *)(v26 - 24);
      if ( *(_BYTE *)(v13 + 16) == 9 )
        goto LABEL_19;
      v12 = v28[1];
      v17 = (v12 >> 1) & 3;
      if ( v17 == 1 )
        goto LABEL_19;
      if ( !v17 )
      {
        v170 = v12 & 1;
LABEL_320:
        v12 = v13 | v170 | 2;
        v28[1] = v12;
        goto LABEL_19;
      }
      v17 = v12 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v13 == (v12 & 0xFFFFFFFFFFFFFFF8LL) )
      {
LABEL_19:
        v25 = *(_QWORD *)(v25 + 8);
        if ( v25 == v188 )
          goto LABEL_32;
      }
      else
      {
        v12 |= 6uLL;
        v28[1] = v12;
        v25 = *(_QWORD *)(v25 + 8);
        if ( v25 == v188 )
          goto LABEL_32;
      }
    }
    ++v221;
    goto LABEL_371;
  }
LABEL_32:
  sub_1A174E0((__int64)v214, *(double *)a4.m128_u64, a5, a6, v17, v12, v13, v14, v15);
  v195 = v11;
  do
  {
    v30 = *(_QWORD *)(v195 + 32);
    if ( v189 == v30 )
    {
      v200 = 0;
      v253 = (__int64 *)v255;
      v254 = 0x20000000000LL;
      goto LABEL_163;
    }
    v31 = 0;
    do
    {
      while ( 1 )
      {
        v32 = v30 - 56;
        if ( !v30 )
          v32 = 0;
        v33 = sub_1A17850((__int64)v214, v32);
        if ( v33 )
          break;
        v30 = *(_QWORD *)(v30 + 8);
        if ( v189 == v30 )
          goto LABEL_40;
      }
      v202 = v33;
      sub_1A174E0((__int64)v214, *(double *)a4.m128_u64, a5, a6, v32, v34, v35, v36, v37);
      v30 = *(_QWORD *)(v30 + 8);
      v31 = v202;
    }
    while ( v189 != v30 );
LABEL_40:
    ;
  }
  while ( v31 );
  v200 = 0;
  v38 = *(_QWORD *)(v195 + 32);
  v253 = (__int64 *)v255;
  v254 = 0x20000000000LL;
  for ( m = v38; v189 != m; m = *(_QWORD *)(m + 8) )
  {
    v39 = m - 56;
    if ( !m )
      v39 = 0;
    v194 = v39;
    if ( !sub_15E4F60(v39) )
    {
      v40 = *(_QWORD *)(v39 + 80);
      if ( v40 )
        v40 -= 24;
      if ( sub_183E920((__int64)&v215, v40) )
      {
        if ( (*(_BYTE *)(v39 + 18) & 1) != 0 )
        {
          sub_15E08E0(v39, v40);
          v130 = *(_QWORD **)(v39 + 88);
          if ( (*(_BYTE *)(v39 + 18) & 1) != 0 )
            sub_15E08E0(v39, v40);
          v131 = *(_QWORD **)(v39 + 88);
        }
        else
        {
          v130 = *(_QWORD **)(v39 + 88);
          v131 = v130;
        }
        v132 = 5LL * *(_QWORD *)(v39 + 96);
        for ( n = &v131[5 * *(_QWORD *)(v39 + 96)]; v130 != n; v130 += 5 )
        {
          if ( v130[1] )
            sub_1A13400((__int64)v214, v130, a4, a5, a6, a7, v44, v45, a10, a11, v132, v41);
        }
      }
      v46 = *(_QWORD *)(v39 + 80);
      v203 = v39 + 72;
      if ( v39 + 72 != v46 )
      {
        while ( 1 )
        {
          v47 = v216;
          v48 = v46 - 24;
          if ( !v46 )
            v48 = 0;
          if ( v217 == v216 )
          {
            v49 = &v216[v219];
            if ( v216 == v49 )
            {
              v50 = (__int64)v216;
            }
            else
            {
              do
              {
                if ( v48 == *v47 )
                  break;
                ++v47;
              }
              while ( v49 != v47 );
              v50 = (__int64)&v216[v219];
            }
          }
          else
          {
            v196 = &v217[v218];
            v47 = sub_16CC9F0((__int64)&v215, v48);
            v49 = v196;
            if ( v48 == *v47 )
            {
              if ( v217 == v216 )
                v50 = (__int64)&v217[v219];
              else
                v50 = (__int64)&v217[v218];
            }
            else
            {
              if ( v217 != v216 )
              {
                v50 = v218;
                v47 = &v217[v218];
                goto LABEL_57;
              }
              v47 = &v217[v219];
              v50 = (__int64)v47;
            }
          }
          for ( ; (_QWORD *)v50 != v47; ++v47 )
          {
            if ( *v47 < 0xFFFFFFFFFFFFFFFELL )
              break;
          }
LABEL_57:
          if ( v47 != v49 )
          {
            v51 = *(_QWORD *)(v48 + 48);
            v52 = v48 + 40;
            if ( v52 == v51 )
              goto LABEL_66;
            v53 = v51;
            v193 = v46;
            while ( 1 )
            {
LABEL_61:
              v54 = v53;
              v53 = *(_QWORD *)(v53 + 8);
              v55 = *(_QWORD *)(v54 - 24);
              if ( !*(_BYTE *)(v55 + 8) )
                goto LABEL_60;
              v56 = (_QWORD *)(v54 - 24);
              v57 = sub_1A13400((__int64)v214, (_QWORD *)(v54 - 24), a4, a5, a6, a7, v44, v45, a10, a11, v50, v55);
              if ( !v57 )
                goto LABEL_60;
              v197 = v57;
              v200 = sub_15F33D0((__int64)v56);
              if ( v200 )
                break;
              v200 = v197;
              if ( v52 == v53 )
              {
LABEL_65:
                v46 = v193;
                goto LABEL_66;
              }
            }
            sub_15F20C0(v56);
LABEL_60:
            if ( v52 == v53 )
              goto LABEL_65;
            goto LABEL_61;
          }
          v200 = 1;
          v99 = *(_QWORD *)(v194 + 80);
          if ( v99 )
            v99 -= 24;
          if ( v48 == v99 )
          {
LABEL_66:
            v46 = *(_QWORD *)(v46 + 8);
            if ( v203 == v46 )
              break;
          }
          else
          {
            v100 = (unsigned int)v254;
            if ( (unsigned int)v254 >= HIDWORD(v254) )
            {
              sub_16CD150((__int64)&v253, v255, 0, 8, v42, v43);
              v100 = (unsigned int)v254;
            }
            v200 = 1;
            v253[v100] = v48;
            LODWORD(v254) = v254 + 1;
            v46 = *(_QWORD *)(v46 + 8);
            if ( v203 == v46 )
              break;
          }
        }
      }
      v58 = v253;
      v59 = &v253[(unsigned int)v254];
      if ( v253 != v59 )
      {
        do
        {
          v60 = *v58++;
          v61 = sub_157ED20(v60);
          sub_1AEE6A0(v61, 0, 0, 0);
        }
        while ( v59 != v58 );
      }
      v62 = *(_QWORD *)(v194 + 80);
      if ( v62 )
        v62 -= 24;
      if ( !sub_183E920((__int64)&v215, v62) )
      {
        v63 = *(_QWORD *)(v194 + 80);
        if ( v63 )
          v63 -= 24;
        v64 = sub_157ED20(v63);
        sub_1AEE6A0(v64, 0, 0, 0);
      }
      v65 = 0;
      v198 = 8LL * (unsigned int)v254;
      if ( (_DWORD)v254 )
      {
        do
        {
          v66 = (_QWORD *)v253[v65 / 8];
          v67 = v66[1];
          while ( v67 )
          {
LABEL_77:
            v68 = sub_1648700(v67);
LABEL_78:
            if ( *((_BYTE *)v68 + 16) <= 0x17u )
              v68 = 0;
            while ( 1 )
            {
              v67 = *(_QWORD *)(v67 + 8);
              if ( !v67 )
                break;
              v69 = sub_1648700(v67);
              if ( v68 != v69 )
              {
                if ( !v68 )
                {
                  v68 = v69;
                  goto LABEL_78;
                }
                if ( (unsigned __int8)sub_1AEE9C0(v68[5], 0, 0, 0) )
                  goto LABEL_77;
                goto LABEL_85;
              }
            }
            if ( !v68 || (unsigned __int8)sub_1AEE9C0(v68[5], 0, 0, 0) )
              break;
LABEL_85:
            v70 = *((_BYTE *)v68 + 16);
            if ( v70 == 27 )
            {
              v71 = *((_BYTE *)v68 + 23) & 0x40;
              if ( v71 )
                v72 = (_QWORD *)*(v68 - 1);
              else
                v72 = &v68[-3 * (*((_DWORD *)v68 + 5) & 0xFFFFFFF)];
              v73 = v72[6];
            }
            else if ( v70 == 26 )
            {
              v98 = (__int64 *)sub_16498A0((__int64)v68);
              v73 = sub_159C540(v98);
              v71 = *((_BYTE *)v68 + 23) & 0x40;
            }
            else
            {
              if ( (*((_BYTE *)v68 + 23) & 0x40) != 0 )
                v101 = (_QWORD *)*(v68 - 1);
              else
                v101 = &v68[-3 * (*((_DWORD *)v68 + 5) & 0xFFFFFFF)];
              v73 = sub_159BF40(v101[3]);
              v71 = *((_BYTE *)v68 + 23) & 0x40;
            }
            if ( v71 )
              v74 = (__int64 *)*(v68 - 1);
            else
              v74 = &v68[-3 * (*((_DWORD *)v68 + 5) & 0xFFFFFFF)];
            if ( *v74 )
            {
              v75 = v74[1];
              v76 = v74[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v76 = v75;
              if ( v75 )
                *(_QWORD *)(v75 + 16) = *(_QWORD *)(v75 + 16) & 3LL | v76;
            }
            *v74 = v73;
            if ( v73 )
            {
              v77 = *(_QWORD *)(v73 + 8);
              v74[1] = v77;
              if ( v77 )
                *(_QWORD *)(v77 + 16) = (unsigned __int64)(v74 + 1) | *(_QWORD *)(v77 + 16) & 3LL;
              v74[2] = (v73 + 8) | v74[2] & 3;
              *(_QWORD *)(v73 + 8) = v74;
            }
            sub_1AEE9C0(v68[5], 0, 0, 0);
          }
          v65 += 8LL;
          sub_15E0220(v203, (__int64)v66);
          v78 = (unsigned __int64 *)v66[4];
          v79 = v66[3] & 0xFFFFFFFFFFFFFFF8LL;
          *v78 = v79 | *v78 & 7;
          *(_QWORD *)(v79 + 8) = v78;
          v66[3] &= 7uLL;
          v66[4] = 0;
          sub_157EF40((__int64)v66);
          j_j___libc_free_0(v66, 64);
        }
        while ( v198 != v65 );
      }
      LODWORD(v254) = 0;
    }
  }
LABEL_163:
  v205 = v207;
  v206 = 0x800000000LL;
  if ( v228 )
  {
    v156 = v227;
    v157 = &v227[2 * v230];
    if ( v227 != v157 )
    {
      while ( 1 )
      {
        v158 = *v156;
        v159 = v156;
        if ( *v156 != -8 && v158 != -16 )
          break;
        v156 += 2;
        if ( v157 == v156 )
          goto LABEL_164;
      }
      if ( v157 != v156 )
      {
        while ( 1 )
        {
          if ( ((*((_BYTE *)v159 + 8) ^ 6) & 6) != 0
            && *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v158 + 24) + 16LL) + 8LL)
            && sub_186B010((__int64)&v243, v158)
            && !sub_186B010((__int64)v242, v158) )
          {
            sub_1A0F420(v158, (__int64)&v205);
          }
          v159 += 2;
          if ( v159 == v157 )
            break;
          while ( *v159 == -8 || *v159 == -16 )
          {
            v159 += 2;
            if ( v157 == v159 )
              goto LABEL_164;
          }
          if ( v157 == v159 )
            break;
          v158 = *v159;
        }
      }
    }
  }
LABEL_164:
  sub_16CCCB0(&v208, (__int64)v213, (__int64)&v236);
  v102 = (unsigned __int64)v210;
  v103 = v209;
  if ( v210 == v209 )
    v104 = &v210[v212];
  else
    v104 = &v210[v211];
  if ( v210 != v104 )
  {
    v105 = v210;
    while ( 1 )
    {
      v106 = *v105;
      v107 = v105;
      if ( (unsigned __int64)*v105 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v104 == ++v105 )
        goto LABEL_170;
    }
    if ( v104 != v105 )
    {
      do
      {
        v117 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(v106 + 24) + 16LL) + 12LL);
        if ( v117 )
        {
          v118 = 0;
          v199 = v107;
          v119 = 0;
          v121 = v235 - 1;
          while ( 1 )
          {
            v122 = v232 + 24LL * v235;
            if ( v235 )
            {
              v123 = 1;
              v120 = (unsigned __int64)(((unsigned int)v106 >> 9) ^ ((unsigned int)v106 >> 4)) << 32;
              v124 = (((v120 | v118) - 1 - ((unsigned __int64)v118 << 32)) >> 22)
                   ^ ((v120 | v118) - 1 - ((unsigned __int64)v118 << 32));
              v125 = ((9 * (((v124 - 1 - (v124 << 13)) >> 8) ^ (v124 - 1 - (v124 << 13)))) >> 15)
                   ^ (9 * (((v124 - 1 - (v124 << 13)) >> 8) ^ (v124 - 1 - (v124 << 13))));
              for ( ii = v121 & (((v125 - 1 - (v125 << 27)) >> 31) ^ (v125 - 1 - ((_DWORD)v125 << 27))); ; ii = v121 & v127 )
              {
                v122 = v232 + 24LL * ii;
                if ( *(_QWORD *)v122 == v106 && *(_DWORD *)(v122 + 8) == v119 )
                  break;
                if ( *(_QWORD *)v122 == -8 && *(_DWORD *)(v122 + 8) == -1 )
                {
                  v122 = v232 + 24LL * v235;
                  break;
                }
                v127 = v123 + ii;
                ++v123;
              }
            }
            if ( ((*(_BYTE *)(v122 + 16) ^ 6) & 6) == 0 )
              break;
            ++v119;
            v118 += 37;
            if ( v117 == v119 )
            {
              v107 = v199;
              goto LABEL_199;
            }
          }
          v107 = v199;
        }
        else
        {
LABEL_199:
          if ( sub_186B010((__int64)&v243, v106) && !sub_186B010((__int64)v242, v106) )
            sub_1A0F420(v106, (__int64)&v205);
        }
        v128 = v107 + 1;
        if ( v107 + 1 == v104 )
          break;
        while ( 1 )
        {
          v106 = *v128;
          v107 = v128;
          if ( (unsigned __int64)*v128 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v104 == ++v128 )
            goto LABEL_205;
        }
      }
      while ( v128 != v104 );
LABEL_205:
      v102 = (unsigned __int64)v210;
      v103 = v209;
    }
  }
LABEL_170:
  if ( v103 != (__int64 *)v102 )
    _libc_free(v102);
  v108 = 0;
  v109 = 8LL * (unsigned int)v206;
  if ( (_DWORD)v206 )
  {
    do
    {
      v110 = *(_QWORD *)&v205[v108];
      v111 = sub_1599EF0(**(__int64 ****)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v110 + 40) + 56LL) + 24LL) + 16LL));
      v112 = (__int64 *)(v110 - 24LL * (*(_DWORD *)(v110 + 20) & 0xFFFFFFF));
      if ( *v112 )
      {
        v113 = v112[1];
        v114 = v112[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v114 = v113;
        if ( v113 )
          *(_QWORD *)(v113 + 16) = *(_QWORD *)(v113 + 16) & 3LL | v114;
      }
      *v112 = v111;
      if ( v111 )
      {
        v115 = *(_QWORD *)(v111 + 8);
        v112[1] = v115;
        if ( v115 )
          *(_QWORD *)(v115 + 16) = (unsigned __int64)(v112 + 1) | *(_QWORD *)(v115 + 16) & 3LL;
        v112[2] = v112[2] & 3 | (v111 + 8);
        *(_QWORD *)(v111 + 8) = v112;
      }
      v108 += 8;
    }
    while ( v108 != v109 );
  }
  if ( v223 )
  {
    v146 = v222;
    v147 = &v222[2 * v225];
    if ( v222 != v147 )
    {
      while ( 1 )
      {
        v148 = *v146;
        v149 = v146;
        if ( *v146 != -16 && v148 != -8 )
          break;
        v146 += 2;
        if ( v147 == v146 )
          goto LABEL_182;
      }
      if ( v147 != v146 )
      {
        while ( 1 )
        {
          while ( 1 )
          {
            v151 = *(_QWORD *)(v148 + 8);
            if ( !v151 )
              break;
            v150 = sub_1648700(v151);
            sub_15F20C0(v150);
          }
          v149 += 2;
          sub_1631C10(v188, v148);
          v152 = *(unsigned __int64 **)(v148 + 64);
          v153 = *(_QWORD *)(v148 + 56) & 0xFFFFFFFFFFFFFFF8LL;
          *v152 = v153 | *v152 & 7;
          *(_QWORD *)(v153 + 8) = v152;
          *(_QWORD *)(v148 + 56) &= 7uLL;
          *(_QWORD *)(v148 + 64) = 0;
          sub_15E5530(v148);
          sub_159D9E0(v148);
          sub_164BE60(v148, a4, a5, a6, a7, v154, v155, a10, a11);
          *(_DWORD *)(v148 + 20) = *(_DWORD *)(v148 + 20) & 0xF0000000 | 1;
          sub_1648B90(v148);
          if ( v149 == v147 )
            break;
          while ( *v149 == -8 || *v149 == -16 )
          {
            v149 += 2;
            if ( v147 == v149 )
              goto LABEL_182;
          }
          if ( v147 == v149 )
            break;
          v148 = *v149;
        }
      }
    }
  }
LABEL_182:
  if ( v205 != v207 )
    _libc_free((unsigned __int64)v205);
  if ( v253 != (__int64 *)v255 )
    _libc_free((unsigned __int64)v253);
  sub_1A0F740((__int64)v214);
  return v200;
}
