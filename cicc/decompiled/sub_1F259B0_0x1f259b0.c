// Function: sub_1F259B0
// Address: 0x1f259b0
//
__int64 __fastcall sub_1F259B0(
        _QWORD *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rax
  __int64 v12; // rbx
  __int64 i; // r12
  unsigned int v14; // esi
  int v15; // edx
  unsigned int v16; // r9d
  __int64 v17; // r8
  unsigned int v18; // ecx
  _DWORD *v19; // rax
  int v20; // edi
  int v21; // eax
  __int64 *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r15
  __int64 v27; // rax
  __int64 v28; // r13
  unsigned __int64 v29; // rax
  __int64 v30; // rbx
  __int64 v31; // r12
  int v32; // eax
  unsigned int v33; // esi
  unsigned int v34; // r9d
  __int64 v35; // r8
  __int64 v36; // rdx
  _DWORD *v37; // rcx
  int v38; // edi
  int v39; // eax
  __int64 **v41; // rbx
  __int64 **v42; // r14
  __int64 *v43; // r12
  __int64 v44; // rax
  unsigned __int64 v45; // rax
  __int64 v46; // rbx
  __int64 m; // r13
  int v48; // r12d
  unsigned int v49; // esi
  unsigned int v50; // r9d
  __int64 v51; // r8
  unsigned int v52; // edi
  int *v53; // rax
  int v54; // ecx
  int v55; // eax
  unsigned __int64 v56; // rax
  __int64 v57; // r12
  int v58; // r15d
  __int64 v59; // r14
  unsigned __int64 v60; // rbx
  unsigned __int64 v61; // rax
  __int64 *v62; // r10
  __int64 v63; // rdi
  _QWORD *v64; // r13
  unsigned int v65; // r12d
  unsigned int v66; // r9d
  unsigned int v67; // r13d
  unsigned int v68; // edi
  _QWORD *v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // rax
  __int64 *v72; // rbx
  int v73; // eax
  __int64 v74; // r15
  _QWORD *v75; // r14
  unsigned int v76; // r13d
  __int64 *v77; // r12
  int v78; // edx
  __int64 v79; // rsi
  __int64 *v80; // rax
  __int64 *v81; // r8
  __int64 v82; // r11
  int k; // r10d
  int v84; // r11d
  __int64 v85; // r10
  _QWORD *v86; // rax
  int v87; // ecx
  int v88; // r10d
  int n; // r11d
  int *v90; // r10
  int v91; // eax
  int v92; // ecx
  unsigned int v93; // r13d
  __int64 v94; // rsi
  int v95; // r9d
  _QWORD *v96; // r8
  unsigned int v97; // r13d
  int v98; // r9d
  __int64 v99; // rsi
  int v100; // eax
  int v101; // esi
  __int64 v102; // r8
  __int64 v103; // rdx
  int v104; // eax
  int v105; // r9d
  int *v106; // rdi
  int v107; // eax
  int v108; // eax
  __int64 v109; // r9
  int *v110; // rsi
  int v111; // edi
  int v112; // edx
  int v113; // r8d
  int v114; // r11d
  unsigned int v115; // r15d
  int ii; // r10d
  int v117; // r11d
  _DWORD *v118; // r10
  int v119; // eax
  int v120; // edx
  int v121; // eax
  __int64 *v122; // rcx
  __int64 *v123; // r10
  _DWORD *v124; // rdx
  _DWORD *v125; // r13
  int v126; // eax
  _DWORD *v127; // rbx
  __int64 v128; // rcx
  int v129; // edx
  __int64 v130; // rcx
  __int64 v131; // r14
  _QWORD *v132; // r15
  unsigned int v133; // edx
  __int64 *v134; // rax
  __int64 v135; // rcx
  double v136; // xmm4_8
  double v137; // xmm5_8
  __int64 v138; // r12
  _QWORD *v139; // rax
  __int64 *v140; // rax
  unsigned __int64 v141; // rdx
  __int64 v142; // rdx
  int v143; // eax
  __int64 v144; // rcx
  char v145; // dl
  __int64 v146; // rax
  __int64 *v147; // r15
  _DWORD *v148; // rdx
  __int64 v149; // rbx
  __int64 v150; // r12
  __int64 v151; // rax
  double v152; // xmm4_8
  double v153; // xmm5_8
  __int64 *v154; // r8
  unsigned int v155; // esi
  __int64 *v156; // rdx
  __int64 *v157; // rdi
  __int64 *v158; // rdi
  __int64 *v159; // rcx
  __int64 v160; // rax
  double v161; // xmm4_8
  double v162; // xmm5_8
  int v163; // r10d
  __int64 *v164; // r9
  int v165; // edx
  __int64 v166; // rcx
  __int64 v167; // r8
  int v168; // edi
  __int64 *v169; // rsi
  int v170; // esi
  __int64 v171; // r12
  __int64 *v172; // rcx
  __int64 v173; // rdi
  int v174; // r15d
  unsigned int v175; // r11d
  int j; // r10d
  int v177; // r11d
  _DWORD *v178; // r10
  int v179; // ecx
  int v180; // edx
  int *v181; // r11
  int v182; // r8d
  int v183; // edi
  int v184; // r8d
  __int64 v185; // r9
  unsigned int v186; // eax
  int v187; // r15d
  int v188; // esi
  _DWORD *v189; // rcx
  int v190; // r8d
  int v191; // edi
  int v192; // r8d
  __int64 v193; // r9
  int v194; // esi
  unsigned int v195; // eax
  int v196; // r15d
  int v197; // eax
  int v198; // r8d
  int v199; // r10d
  __int64 v200; // r9
  _DWORD *v201; // r11
  unsigned int v202; // ecx
  int v203; // esi
  int v204; // edi
  int v205; // eax
  int v206; // r9d
  int v207; // r10d
  __int64 v208; // r11
  unsigned int v209; // ecx
  int v210; // r8d
  int v211; // edi
  _DWORD *v212; // rsi
  int v213; // r11d
  __int64 v214; // rdi
  __int64 v215; // [rsp+0h] [rbp-220h]
  __int64 v216; // [rsp+8h] [rbp-218h]
  __int64 v218; // [rsp+20h] [rbp-200h]
  __int64 *v219; // [rsp+28h] [rbp-1F8h]
  __int64 *v220; // [rsp+30h] [rbp-1F0h]
  __int64 v221; // [rsp+38h] [rbp-1E8h]
  int v222; // [rsp+40h] [rbp-1E0h]
  char v223; // [rsp+47h] [rbp-1D9h]
  __int64 v225; // [rsp+60h] [rbp-1C0h]
  __int64 v226; // [rsp+60h] [rbp-1C0h]
  int v227; // [rsp+60h] [rbp-1C0h]
  __int64 v228; // [rsp+68h] [rbp-1B8h]
  unsigned __int64 v229; // [rsp+68h] [rbp-1B8h]
  unsigned int v230; // [rsp+68h] [rbp-1B8h]
  int v231; // [rsp+68h] [rbp-1B8h]
  unsigned int v232; // [rsp+68h] [rbp-1B8h]
  int v233; // [rsp+68h] [rbp-1B8h]
  __int64 v234; // [rsp+68h] [rbp-1B8h]
  __int64 v235; // [rsp+68h] [rbp-1B8h]
  int v236; // [rsp+68h] [rbp-1B8h]
  __int64 **v237; // [rsp+68h] [rbp-1B8h]
  _DWORD *v238; // [rsp+68h] [rbp-1B8h]
  __int64 v239; // [rsp+70h] [rbp-1B0h] BYREF
  __int64 v240; // [rsp+78h] [rbp-1A8h]
  __int64 v241; // [rsp+80h] [rbp-1A0h]
  unsigned int v242; // [rsp+88h] [rbp-198h]
  __int64 *v243; // [rsp+90h] [rbp-190h] BYREF
  __int64 v244; // [rsp+98h] [rbp-188h]
  _QWORD v245[4]; // [rsp+A0h] [rbp-180h] BYREF
  __int64 v246; // [rsp+C0h] [rbp-160h] BYREF
  __int64 *v247; // [rsp+C8h] [rbp-158h]
  __int64 *v248; // [rsp+D0h] [rbp-150h]
  __int64 v249; // [rsp+D8h] [rbp-148h]
  int v250; // [rsp+E0h] [rbp-140h]
  _BYTE v251[312]; // [rsp+E8h] [rbp-138h] BYREF

  v10 = a1[30];
  v12 = *(_QWORD *)(v10 + 608);
  for ( i = v12 + 32LL * *(unsigned int *)(v10 + 616); i != v12; v12 += 32 )
  {
    if ( *(_QWORD *)v12 )
    {
      v14 = *(_DWORD *)(a2 + 24);
      if ( v14 )
      {
        v15 = *(_DWORD *)(v12 + 16);
        v16 = v14 - 1;
        v17 = *(_QWORD *)(a2 + 8);
        v18 = (v14 - 1) & (37 * v15);
        v19 = (_DWORD *)(v17 + 8LL * v18);
        v20 = *v19;
        if ( v15 == *v19 )
        {
LABEL_5:
          v21 = v19[1];
        }
        else
        {
          v174 = *v19;
          v175 = (v14 - 1) & (37 * v15);
          for ( j = 1; ; ++j )
          {
            if ( v174 == 0x7FFFFFFF )
              goto LABEL_7;
            v175 = v16 & (v175 + j);
            v174 = *(_DWORD *)(v17 + 8LL * v175);
            if ( v15 == v174 )
              break;
          }
          v177 = 1;
          v178 = 0;
          while ( v20 != 0x7FFFFFFF )
          {
            if ( v20 == 0x80000000 && !v178 )
              v178 = v19;
            v18 = v16 & (v177 + v18);
            v19 = (_DWORD *)(v17 + 8LL * v18);
            v20 = *v19;
            if ( v15 == *v19 )
              goto LABEL_5;
            ++v177;
          }
          v179 = *(_DWORD *)(a2 + 16);
          if ( v178 )
            v19 = v178;
          ++*(_QWORD *)a2;
          v180 = v179 + 1;
          if ( 4 * (v179 + 1) >= 3 * v14 )
          {
            sub_1E4B4F0(a2, 2 * v14);
            v205 = *(_DWORD *)(a2 + 24);
            if ( !v205 )
            {
LABEL_355:
              ++*(_DWORD *)(a2 + 16);
              BUG();
            }
            v206 = *(_DWORD *)(v12 + 16);
            v207 = v205 - 1;
            v208 = *(_QWORD *)(a2 + 8);
            v180 = *(_DWORD *)(a2 + 16) + 1;
            v209 = (v205 - 1) & (37 * v206);
            v19 = (_DWORD *)(v208 + 8LL * v209);
            v210 = *v19;
            if ( v206 != *v19 )
            {
              v211 = 1;
              v212 = 0;
              while ( v210 != 0x7FFFFFFF )
              {
                if ( v210 == 0x80000000 && !v212 )
                  v212 = v19;
                v209 = v207 & (v211 + v209);
                v19 = (_DWORD *)(v208 + 8LL * v209);
                v210 = *v19;
                if ( v206 == *v19 )
                  goto LABEL_253;
                ++v211;
              }
              if ( v212 )
                v19 = v212;
            }
          }
          else if ( v14 - *(_DWORD *)(a2 + 20) - v180 <= v14 >> 3 )
          {
            sub_1E4B4F0(a2, v14);
            v197 = *(_DWORD *)(a2 + 24);
            if ( !v197 )
              goto LABEL_355;
            v198 = *(_DWORD *)(v12 + 16);
            v199 = v197 - 1;
            v200 = *(_QWORD *)(a2 + 8);
            v201 = 0;
            v202 = (v197 - 1) & (37 * v198);
            v19 = (_DWORD *)(v200 + 8LL * v202);
            v180 = *(_DWORD *)(a2 + 16) + 1;
            v203 = 1;
            v204 = *v19;
            if ( v198 != *v19 )
            {
              while ( v204 != 0x7FFFFFFF )
              {
                if ( v204 == 0x80000000 && !v201 )
                  v201 = v19;
                v202 = v199 & (v203 + v202);
                v19 = (_DWORD *)(v200 + 8LL * v202);
                v204 = *v19;
                if ( v198 == *v19 )
                  goto LABEL_253;
                ++v203;
              }
              if ( v201 )
                v19 = v201;
            }
          }
LABEL_253:
          *(_DWORD *)(a2 + 16) = v180;
          if ( *v19 != 0x7FFFFFFF )
            --*(_DWORD *)(a2 + 20);
          *(_QWORD *)v19 = *(unsigned int *)(v12 + 16);
          v21 = 0;
        }
        *(_DWORD *)(v12 + 16) = v21;
      }
    }
LABEL_7:
    ;
  }
  v240 = 0;
  v247 = (__int64 *)v251;
  v248 = (__int64 *)v251;
  v241 = 0;
  v242 = 0;
  v22 = (__int64 *)a1[1];
  v249 = 32;
  v250 = 0;
  v239 = 0;
  v246 = 0;
  v23 = *v22;
  v24 = v22[1];
  if ( v23 == v24 )
LABEL_359:
    BUG();
  while ( *(_UNKNOWN **)v23 != &unk_4F9E06C )
  {
    v23 += 16;
    if ( v24 == v23 )
      goto LABEL_359;
  }
  v225 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(
           *(_QWORD *)(v23 + 8),
           &unk_4F9E06C)
       + 160;
  if ( *(_DWORD *)(a2 + 16) )
  {
    v124 = *(_DWORD **)(a2 + 8);
    v125 = &v124[2 * *(unsigned int *)(a2 + 24)];
    if ( v124 != v125 )
    {
      while ( 1 )
      {
        v126 = *v124;
        v127 = v124;
        if ( (unsigned int)(*v124 + 0x7FFFFFFF) <= 0xFFFFFFFD )
          break;
        v124 += 2;
        if ( v125 == v124 )
          goto LABEL_13;
      }
      if ( v125 != v124 )
      {
        while ( 1 )
        {
          v128 = a1[29];
          v129 = *(_DWORD *)(v128 + 32);
          v130 = *(_QWORD *)(v128 + 8);
          v131 = *(_QWORD *)(v130 + 40LL * (unsigned int)(v129 + v126) + 24);
          v132 = *(_QWORD **)(v130 + 40LL * (unsigned int)(v127[1] + v129) + 24);
          if ( !v242 )
            break;
          v133 = (v242 - 1) & (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4));
          v134 = (__int64 *)(v240 + 16LL * v133);
          v135 = *v134;
          if ( v131 != *v134 )
          {
            v163 = 1;
            v164 = 0;
            while ( v135 != -8 )
            {
              if ( v135 == -16 && !v164 )
                v164 = v134;
              v133 = (v242 - 1) & (v163 + v133);
              v134 = (__int64 *)(v240 + 16LL * v133);
              v135 = *v134;
              if ( v131 == *v134 )
                goto LABEL_170;
              ++v163;
            }
            if ( v164 )
              v134 = v164;
            ++v239;
            v165 = v241 + 1;
            if ( 4 * ((int)v241 + 1) < 3 * v242 )
            {
              if ( v242 - HIDWORD(v241) - v165 <= v242 >> 3 )
              {
                sub_1F257F0((__int64)&v239, v242);
                if ( !v242 )
                {
LABEL_358:
                  LODWORD(v241) = v241 + 1;
                  BUG();
                }
                v170 = 1;
                LODWORD(v171) = (v242 - 1) & (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4));
                v165 = v241 + 1;
                v172 = 0;
                v134 = (__int64 *)(v240 + 16LL * (unsigned int)v171);
                v173 = *v134;
                if ( v131 != *v134 )
                {
                  while ( v173 != -8 )
                  {
                    if ( v173 == -16 && !v172 )
                      v172 = v134;
                    v171 = (v242 - 1) & ((_DWORD)v171 + v170);
                    v134 = (__int64 *)(v240 + 16 * v171);
                    v173 = *v134;
                    if ( v131 == *v134 )
                      goto LABEL_227;
                    ++v170;
                  }
                  if ( v172 )
                    v134 = v172;
                }
              }
              goto LABEL_227;
            }
LABEL_231:
            sub_1F257F0((__int64)&v239, 2 * v242);
            if ( !v242 )
              goto LABEL_358;
            LODWORD(v166) = (v242 - 1) & (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4));
            v165 = v241 + 1;
            v134 = (__int64 *)(v240 + 16LL * (unsigned int)v166);
            v167 = *v134;
            if ( v131 != *v134 )
            {
              v168 = 1;
              v169 = 0;
              while ( v167 != -8 )
              {
                if ( !v169 && v167 == -16 )
                  v169 = v134;
                v166 = (v242 - 1) & ((_DWORD)v166 + v168);
                v134 = (__int64 *)(v240 + 16 * v166);
                v167 = *v134;
                if ( v131 == *v134 )
                  goto LABEL_227;
                ++v168;
              }
              if ( v169 )
                v134 = v169;
            }
LABEL_227:
            LODWORD(v241) = v165;
            if ( *v134 != -8 )
              --HIDWORD(v241);
            *v134 = v131;
            v134[1] = 0;
          }
LABEL_170:
          v134[1] = (__int64)v132;
          if ( sub_15CCEE0(v225, v131, (__int64)v132) )
            sub_15F22F0(v132, v131);
          v138 = (__int64)v132;
          v237 = *(__int64 ***)v131;
          if ( *v132 != *(_QWORD *)v131 )
          {
            LOWORD(v245[0]) = 257;
            v139 = sub_1648A60(56, 1u);
            v138 = (__int64)v139;
            if ( v139 )
              sub_15FD590((__int64)v139, (__int64)v132, (__int64)v237, (__int64)&v243, 0);
            sub_15F2180(v138, (__int64)v132);
          }
          v140 = v247;
          if ( v248 != v247 )
            goto LABEL_177;
          v154 = &v247[HIDWORD(v249)];
          v155 = HIDWORD(v249);
          if ( v247 != v154 )
          {
            v156 = v247;
            v157 = 0;
            do
            {
              if ( v131 == *v156 )
              {
                v158 = &v247[HIDWORD(v249)];
                if ( v247 == v158 )
                  goto LABEL_215;
                goto LABEL_207;
              }
              if ( *v156 == -2 )
                v157 = v156;
              ++v156;
            }
            while ( v154 != v156 );
            if ( !v157 )
              goto LABEL_219;
            *v157 = v131;
            v141 = (unsigned __int64)v248;
            --v250;
            v140 = v247;
            ++v246;
            goto LABEL_178;
          }
LABEL_219:
          if ( HIDWORD(v249) < (unsigned int)v249 )
          {
            ++HIDWORD(v249);
            *v154 = v131;
            v140 = v247;
            ++v246;
            v141 = (unsigned __int64)v248;
          }
          else
          {
LABEL_177:
            sub_16CCBA0((__int64)&v246, v131);
            v141 = (unsigned __int64)v248;
            v140 = v247;
          }
LABEL_178:
          if ( v140 != (__int64 *)v141 )
            goto LABEL_179;
          v155 = HIDWORD(v249);
          v158 = &v140[HIDWORD(v249)];
          if ( v140 != v158 )
          {
LABEL_207:
            v159 = 0;
            while ( v132 != (_QWORD *)*v140 )
            {
              if ( *v140 == -2 )
                v159 = v140;
              if ( v158 == ++v140 )
              {
                if ( !v159 )
                  goto LABEL_215;
                *v159 = (__int64)v132;
                --v250;
                ++v246;
                goto LABEL_180;
              }
            }
            goto LABEL_180;
          }
LABEL_215:
          if ( v155 < (unsigned int)v249 )
          {
            HIDWORD(v249) = v155 + 1;
            *v158 = (__int64)v132;
            ++v246;
          }
          else
          {
LABEL_179:
            sub_16CCBA0((__int64)&v246, (__int64)v132);
          }
LABEL_180:
          v142 = a1[29];
          v143 = *(_DWORD *)(v142 + 32);
          v144 = *(_QWORD *)(v142 + 8);
          v145 = *(_BYTE *)(v144 + 40LL * (unsigned int)(*v127 + v143) + 36);
          if ( v145 )
          {
            v146 = v144 + 40LL * (unsigned int)(v127[1] + v143);
            if ( !*(_BYTE *)(v146 + 36) || *(_BYTE *)(v146 + 36) != 1 && v145 != 3 )
              *(_BYTE *)(v146 + 36) = v145;
          }
          if ( (*(_BYTE *)(v131 + 23) & 0x10) != 0 )
          {
            v160 = sub_1599EF0(*(__int64 ***)v131);
            sub_16303F0(v131, v160, a3, a4, a5, a6, v161, v162, a9, a10);
          }
          v147 = *(__int64 **)(v131 + 8);
          if ( v147 )
          {
            v148 = v127;
            v149 = v138;
            do
            {
              while ( 1 )
              {
                v150 = *v147;
                if ( *(_BYTE *)(*v147 + 16) == 71 && (*(_BYTE *)(v150 + 23) & 0x10) != 0 )
                  break;
                v147 = (__int64 *)v147[1];
                if ( !v147 )
                  goto LABEL_193;
              }
              v238 = v148;
              v151 = sub_1599EF0(*(__int64 ***)v150);
              sub_16303F0(v150, v151, a3, a4, a5, a6, v152, v153, a9, a10);
              v147 = (__int64 *)v147[1];
              v148 = v238;
            }
            while ( v147 );
LABEL_193:
            v138 = v149;
            v127 = v148;
          }
          v127 += 2;
          sub_164D160(v131, v138, a3, a4, a5, a6, v136, v137, a9, a10);
          if ( v127 != v125 )
          {
            while ( 1 )
            {
              v126 = *v127;
              if ( (unsigned int)(*v127 + 0x7FFFFFFF) <= 0xFFFFFFFD )
                break;
              v127 += 2;
              if ( v125 == v127 )
                goto LABEL_13;
            }
            if ( v125 != v127 )
              continue;
          }
          goto LABEL_13;
        }
        ++v239;
        goto LABEL_231;
      }
    }
  }
LABEL_13:
  v25 = a1[30];
  v215 = v25 + 320;
  v216 = *(_QWORD *)(v25 + 328);
  if ( v25 + 320 != v216 )
  {
    while ( 1 )
    {
      v26 = *(_QWORD *)(v216 + 32);
      v221 = v216 + 24;
      if ( v216 + 24 != v26 )
        break;
LABEL_20:
      v216 = *(_QWORD *)(v216 + 8);
      if ( v215 == v216 )
      {
        v25 = a1[30];
        goto LABEL_22;
      }
    }
LABEL_17:
    if ( (unsigned int)**(unsigned __int16 **)(v26 + 16) - 17 <= 1 )
      goto LABEL_18;
    v41 = *(__int64 ***)(v26 + 56);
    v42 = &v41[*(unsigned __int8 *)(v26 + 49)];
    if ( v41 != v42 )
    {
      while ( 1 )
      {
        v43 = *v41;
        v44 = **v41;
        if ( (v44 & 4) == 0 )
        {
          v45 = v44 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v45 )
          {
            if ( *(_BYTE *)(v45 + 16) == 53 && v242 )
              break;
          }
        }
LABEL_43:
        if ( v42 == ++v41 )
          goto LABEL_44;
      }
      v66 = v242 - 1;
      v67 = ((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4);
      v68 = (v242 - 1) & v67;
      v69 = (_QWORD *)(v240 + 16LL * v68);
      v70 = *v69;
      if ( v45 == *v69 )
      {
        v71 = v69[1];
        goto LABEL_68;
      }
      v230 = (v242 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
      v82 = *v69;
      for ( k = 1; ; k = v227 )
      {
        if ( v82 == -8 )
          goto LABEL_43;
        v84 = k + 1;
        v85 = v66 & (v230 + k);
        v227 = v84;
        v230 = v85;
        v82 = *(_QWORD *)(v240 + 16 * v85);
        if ( v45 == v82 )
          break;
      }
      v231 = 1;
      v86 = 0;
      while ( v70 != -8 )
      {
        if ( v86 || v70 != -16 )
          v69 = v86;
        v68 = v66 & (v231 + v68);
        v123 = (__int64 *)(v240 + 16LL * v68);
        v70 = *v123;
        if ( v82 == *v123 )
        {
          v71 = v123[1];
          goto LABEL_68;
        }
        ++v231;
        v86 = v69;
        v69 = (_QWORD *)(v240 + 16LL * v68);
      }
      if ( !v86 )
        v86 = v69;
      ++v239;
      v87 = v241 + 1;
      if ( 4 * ((int)v241 + 1) >= 3 * v242 )
      {
        v234 = v82;
        sub_1F257F0((__int64)&v239, 2 * v242);
        if ( !v242 )
          goto LABEL_357;
        v93 = (v242 - 1) & v67;
        v82 = v234;
        v87 = v241 + 1;
        v86 = (_QWORD *)(v240 + 16LL * v93);
        v94 = *v86;
        if ( v234 == *v86 )
          goto LABEL_103;
        v95 = 1;
        v96 = 0;
        while ( v94 != -8 )
        {
          if ( v94 == -16 && !v96 )
            v96 = v86;
          v93 = (v242 - 1) & (v95 + v93);
          v86 = (_QWORD *)(v240 + 16LL * v93);
          v94 = *v86;
          if ( v234 == *v86 )
            goto LABEL_103;
          ++v95;
        }
      }
      else
      {
        if ( v242 - HIDWORD(v241) - v87 > v242 >> 3 )
          goto LABEL_103;
        v235 = v82;
        sub_1F257F0((__int64)&v239, v242);
        if ( !v242 )
        {
LABEL_357:
          LODWORD(v241) = v241 + 1;
          BUG();
        }
        v96 = 0;
        v97 = (v242 - 1) & v67;
        v82 = v235;
        v98 = 1;
        v87 = v241 + 1;
        v86 = (_QWORD *)(v240 + 16LL * v97);
        v99 = *v86;
        if ( v235 == *v86 )
          goto LABEL_103;
        while ( v99 != -8 )
        {
          if ( v99 == -16 && !v96 )
            v96 = v86;
          v97 = (v242 - 1) & (v98 + v97);
          v86 = (_QWORD *)(v240 + 16LL * v97);
          v99 = *v86;
          if ( v235 == *v86 )
            goto LABEL_103;
          ++v98;
        }
      }
      if ( v96 )
        v86 = v96;
LABEL_103:
      LODWORD(v241) = v87;
      if ( *v86 != -8 )
        --HIDWORD(v241);
      *v86 = v82;
      v86[1] = 0;
      v71 = 0;
LABEL_68:
      *v43 = v71;
      goto LABEL_43;
    }
LABEL_44:
    v46 = *(_QWORD *)(v26 + 32);
    for ( m = v46 + 40LL * *(unsigned int *)(v26 + 40); m != v46; v46 += 40 )
    {
      if ( *(_BYTE *)v46 == 5 )
      {
        v48 = *(_DWORD *)(v46 + 24);
        if ( v48 >= 0 )
        {
          v49 = *(_DWORD *)(a2 + 24);
          if ( v49 )
          {
            v50 = v49 - 1;
            v51 = *(_QWORD *)(a2 + 8);
            v52 = (v49 - 1) & (37 * v48);
            v53 = (int *)(v51 + 8LL * v52);
            v54 = *v53;
            if ( v48 == *v53 )
            {
              v55 = v53[1];
              goto LABEL_50;
            }
            v232 = (v49 - 1) & (37 * v48);
            v88 = *v53;
            for ( n = 1; ; ++n )
            {
              if ( v88 == 0x7FFFFFFF )
                goto LABEL_51;
              v232 = v50 & (v232 + n);
              v88 = *(_DWORD *)(v51 + 8LL * v232);
              if ( v48 == v88 )
                break;
            }
            v233 = 1;
            v90 = 0;
            while ( v54 != 0x7FFFFFFF )
            {
              if ( v90 || v54 != 0x80000000 )
                v53 = v90;
              v52 = v50 & (v233 + v52);
              v181 = (int *)(v51 + 8LL * v52);
              v54 = *v181;
              if ( v48 == *v181 )
              {
                v55 = v181[1];
                goto LABEL_50;
              }
              ++v233;
              v90 = v53;
              v53 = (int *)(v51 + 8LL * v52);
            }
            if ( !v90 )
              v90 = v53;
            v91 = *(_DWORD *)(a2 + 16);
            ++*(_QWORD *)a2;
            v92 = v91 + 1;
            if ( 4 * (v91 + 1) >= 3 * v49 )
            {
              sub_1E4B4F0(a2, 2 * v49);
              v100 = *(_DWORD *)(a2 + 24);
              if ( !v100 )
                goto LABEL_354;
              v101 = v100 - 1;
              v102 = *(_QWORD *)(a2 + 8);
              LODWORD(v103) = (v100 - 1) & (37 * v48);
              v92 = *(_DWORD *)(a2 + 16) + 1;
              v90 = (int *)(v102 + 8LL * (unsigned int)v103);
              v104 = *v90;
              if ( v48 != *v90 )
              {
                v105 = 1;
                v106 = 0;
                while ( v104 != 0x7FFFFFFF )
                {
                  if ( !v106 && v104 == 0x80000000 )
                    v106 = v90;
                  v103 = v101 & (unsigned int)(v103 + v105);
                  v90 = (int *)(v102 + 8 * v103);
                  v104 = *v90;
                  if ( v48 == *v90 )
                    goto LABEL_116;
                  ++v105;
                }
                if ( v106 )
                  v90 = v106;
              }
            }
            else if ( v49 - *(_DWORD *)(a2 + 20) - v92 <= v49 >> 3 )
            {
              v236 = 37 * v48;
              sub_1E4B4F0(a2, v49);
              v107 = *(_DWORD *)(a2 + 24);
              if ( !v107 )
              {
LABEL_354:
                ++*(_DWORD *)(a2 + 16);
                BUG();
              }
              v108 = v107 - 1;
              v109 = *(_QWORD *)(a2 + 8);
              v110 = 0;
              v111 = 1;
              v112 = v108 & v236;
              v90 = (int *)(v109 + 8LL * (v108 & (unsigned int)v236));
              v113 = *v90;
              v92 = *(_DWORD *)(a2 + 16) + 1;
              if ( v48 != *v90 )
              {
                while ( v113 != 0x7FFFFFFF )
                {
                  if ( !v110 && v113 == 0x80000000 )
                    v110 = v90;
                  v213 = v111 + 1;
                  v214 = v108 & (unsigned int)(v112 + v111);
                  v90 = (int *)(v109 + 8 * v214);
                  v112 = v214;
                  v113 = *v90;
                  if ( v48 == *v90 )
                    goto LABEL_116;
                  v111 = v213;
                }
                if ( v110 )
                  v90 = v110;
              }
            }
LABEL_116:
            *(_DWORD *)(a2 + 16) = v92;
            if ( *v90 != 0x7FFFFFFF )
              --*(_DWORD *)(a2 + 20);
            *v90 = v48;
            v55 = 0;
            v90[1] = 0;
LABEL_50:
            *(_DWORD *)(v46 + 24) = v55;
          }
        }
      }
LABEL_51:
      ;
    }
    v56 = sub_1E0A240(a1[30], *(unsigned __int8 *)(v26 + 49));
    v57 = *(_QWORD *)(v26 + 56);
    v229 = v56;
    v226 = v57 + 8LL * *(unsigned __int8 *)(v26 + 49);
    if ( v57 == v226 )
      goto LABEL_18;
    v218 = v26;
    v223 = 0;
    v58 = 1;
    v59 = v57;
LABEL_61:
    v64 = *(_QWORD **)v59;
    v65 = v58 - 1;
    if ( !*(_QWORD *)(*(_QWORD *)v59 + 40LL) && !v64[6] && !v64[7] )
      goto LABEL_64;
    if ( (*v64 & 4) != 0 )
      goto LABEL_64;
    v60 = *v64 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v60 )
      goto LABEL_64;
    v244 = 0x400000000LL;
    v243 = v245;
    v61 = sub_1E0A0C0(a1[30]);
    sub_14ADA90(v60, (__int64)&v243, v61);
    v62 = v245;
    if ( !(_DWORD)v244 )
    {
LABEL_57:
      if ( v243 != v62 )
        _libc_free((unsigned __int64)v243);
      v243 = 0;
      v63 = a1[30];
      v244 = 0;
      v245[0] = 0;
      v223 = 1;
      *(_QWORD *)(v229 + 8LL * v65) = sub_1E0B970(v63, (__int64)v64, (int)&v243);
      goto LABEL_60;
    }
    v72 = v243;
    v73 = v58;
    v74 = v59;
    v75 = v64;
    v76 = v65;
    v77 = &v243[(unsigned int)v244];
    v78 = v73;
    while ( 1 )
    {
      v79 = *v72;
      if ( *v72 )
      {
        if ( *(_BYTE *)(v79 + 16) == 53 )
          break;
      }
LABEL_72:
      if ( v77 == ++v72 )
      {
        v65 = v76;
        v64 = v75;
        v59 = v74;
        v58 = v78;
        if ( v243 != v62 )
          _libc_free((unsigned __int64)v243);
LABEL_64:
        *(_QWORD *)(v229 + 8LL * v65) = v64;
LABEL_60:
        v59 += 8;
        ++v58;
        if ( v226 != v59 )
          goto LABEL_61;
        v26 = v218;
        if ( v223 )
          *(_QWORD *)(v218 + 56) = v229;
LABEL_18:
        if ( (*(_BYTE *)v26 & 4) != 0 )
        {
          v26 = *(_QWORD *)(v26 + 8);
          if ( v26 == v221 )
            goto LABEL_20;
        }
        else
        {
          while ( (*(_BYTE *)(v26 + 46) & 8) != 0 )
            v26 = *(_QWORD *)(v26 + 8);
          v26 = *(_QWORD *)(v26 + 8);
          if ( v26 == v221 )
            goto LABEL_20;
        }
        goto LABEL_17;
      }
    }
    v80 = v247;
    if ( v248 == v247 )
    {
      v81 = &v247[HIDWORD(v249)];
      if ( v247 == v81 )
      {
        v122 = v247;
      }
      else
      {
        do
        {
          if ( v79 == *v80 )
            break;
          ++v80;
        }
        while ( v81 != v80 );
        v122 = &v247[HIDWORD(v249)];
      }
    }
    else
    {
      v219 = v62;
      v222 = v78;
      v220 = &v248[(unsigned int)v249];
      v80 = sub_16CC9F0((__int64)&v246, v79);
      v81 = v220;
      v78 = v222;
      v62 = v219;
      if ( v79 == *v80 )
      {
        if ( v248 == v247 )
          v122 = &v248[HIDWORD(v249)];
        else
          v122 = &v248[(unsigned int)v249];
      }
      else
      {
        if ( v248 != v247 )
        {
          v80 = &v248[(unsigned int)v249];
          goto LABEL_79;
        }
        v80 = &v248[HIDWORD(v249)];
        v122 = v80;
      }
    }
    for ( ; v122 != v80; ++v80 )
    {
      if ( (unsigned __int64)*v80 < 0xFFFFFFFFFFFFFFFELL )
        break;
    }
LABEL_79:
    if ( v81 != v80 )
    {
      v65 = v76;
      v64 = v75;
      v59 = v74;
      v58 = v78;
      goto LABEL_57;
    }
    goto LABEL_72;
  }
LABEL_22:
  v27 = *(_QWORD *)(v25 + 88);
  if ( v27 )
  {
    v28 = *(_QWORD *)(v27 + 208);
    v29 = (unsigned __int64)*(unsigned int *)(v27 + 216) << 6;
    if ( v28 != v28 + v29 )
    {
      v228 = v28 + v29;
      while ( 1 )
      {
        v30 = *(_QWORD *)(v28 + 16);
        v31 = v30 + 32LL * *(unsigned int *)(v28 + 24);
        if ( v30 != v31 )
          break;
LABEL_32:
        v28 += 64;
        if ( v28 == v228 )
          goto LABEL_33;
      }
      while ( 1 )
      {
        v32 = *(_DWORD *)(v30 + 8);
        if ( v32 == 0x7FFFFFFF )
          goto LABEL_31;
        v33 = *(_DWORD *)(a2 + 24);
        if ( !v33 )
          goto LABEL_31;
        v34 = v33 - 1;
        v35 = *(_QWORD *)(a2 + 8);
        LODWORD(v36) = (v33 - 1) & (37 * v32);
        v37 = (_DWORD *)(v35 + 8LL * (unsigned int)v36);
        v38 = *v37;
        if ( v32 != *v37 )
          break;
LABEL_29:
        v39 = v37[1];
LABEL_30:
        *(_DWORD *)(v30 + 8) = v39;
LABEL_31:
        v30 += 32;
        if ( v30 == v31 )
          goto LABEL_32;
      }
      v114 = *v37;
      v115 = (v33 - 1) & (37 * v32);
      for ( ii = 1; ; ++ii )
      {
        if ( v114 == 0x7FFFFFFF )
          goto LABEL_31;
        v115 = v34 & (ii + v115);
        v114 = *(_DWORD *)(v35 + 8LL * v115);
        if ( v32 == v114 )
          break;
      }
      v117 = 1;
      v118 = 0;
      while ( v38 != 0x7FFFFFFF )
      {
        if ( v38 == 0x80000000 && !v118 )
          v118 = v37;
        v36 = v34 & ((_DWORD)v36 + v117);
        v37 = (_DWORD *)(v35 + 8 * v36);
        v38 = *v37;
        if ( v32 == *v37 )
          goto LABEL_29;
        ++v117;
      }
      v119 = *(_DWORD *)(a2 + 16);
      if ( !v118 )
        v118 = v37;
      ++*(_QWORD *)a2;
      v120 = v119 + 1;
      if ( 4 * (v119 + 1) >= 3 * v33 )
      {
        sub_1E4B4F0(a2, 2 * v33);
        v182 = *(_DWORD *)(a2 + 24);
        if ( !v182 )
          goto LABEL_356;
        v183 = *(_DWORD *)(v30 + 8);
        v184 = v182 - 1;
        v185 = *(_QWORD *)(a2 + 8);
        v186 = v184 & (37 * v183);
        v118 = (_DWORD *)(v185 + 8LL * v186);
        v120 = *(_DWORD *)(a2 + 16) + 1;
        v187 = *v118;
        if ( *v118 == v183 )
          goto LABEL_156;
        v188 = 1;
        v189 = 0;
        while ( v187 != 0x7FFFFFFF )
        {
          if ( v187 == 0x80000000 && !v189 )
            v189 = v118;
          v186 = v184 & (v186 + v188);
          v118 = (_DWORD *)(v185 + 8LL * v186);
          v187 = *v118;
          if ( v183 == *v118 )
            goto LABEL_156;
          ++v188;
        }
      }
      else
      {
        if ( v33 - *(_DWORD *)(a2 + 20) - v120 > v33 >> 3 )
          goto LABEL_156;
        sub_1E4B4F0(a2, v33);
        v190 = *(_DWORD *)(a2 + 24);
        if ( !v190 )
        {
LABEL_356:
          ++*(_DWORD *)(a2 + 16);
          BUG();
        }
        v191 = *(_DWORD *)(v30 + 8);
        v192 = v190 - 1;
        v193 = *(_QWORD *)(a2 + 8);
        v194 = 1;
        v195 = v192 & (37 * v191);
        v118 = (_DWORD *)(v193 + 8LL * v195);
        v120 = *(_DWORD *)(a2 + 16) + 1;
        v189 = 0;
        v196 = *v118;
        if ( *v118 == v191 )
          goto LABEL_156;
        while ( v196 != 0x7FFFFFFF )
        {
          if ( !v189 && v196 == 0x80000000 )
            v189 = v118;
          v195 = v192 & (v195 + v194);
          v118 = (_DWORD *)(v193 + 8LL * v195);
          v196 = *v118;
          if ( v191 == *v118 )
            goto LABEL_156;
          ++v194;
        }
      }
      if ( v189 )
        v118 = v189;
LABEL_156:
      *(_DWORD *)(a2 + 16) = v120;
      if ( *v118 != 0x7FFFFFFF )
        --*(_DWORD *)(a2 + 20);
      v121 = *(_DWORD *)(v30 + 8);
      v118[1] = 0;
      *v118 = v121;
      v39 = 0;
      goto LABEL_30;
    }
  }
LABEL_33:
  if ( v248 != v247 )
    _libc_free((unsigned __int64)v248);
  return j___libc_free_0(v240);
}
