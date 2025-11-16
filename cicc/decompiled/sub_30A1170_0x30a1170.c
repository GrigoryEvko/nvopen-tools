// Function: sub_30A1170
// Address: 0x30a1170
//
__int64 __fastcall sub_30A1170(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r11
  __int64 v8; // rbx
  __int64 i; // r14
  __int64 v10; // r12
  void *v11; // rax
  const char *v12; // rax
  size_t v13; // rdx
  _WORD *v14; // rdi
  unsigned __int8 *v15; // rsi
  size_t v16; // r13
  unsigned __int64 v17; // rax
  __int64 v18; // rdi
  void *v19; // rax
  __int64 v20; // rdi
  void *v21; // rax
  __int64 *v22; // r12
  __int64 v23; // rdx
  __int64 *v24; // r13
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  int v29; // eax
  int v30; // edi
  __int64 v31; // r9
  __int64 *v32; // rbx
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int64 v35; // r13
  __int64 *v36; // rbx
  int v37; // eax
  __int64 *v38; // r13
  __int64 *v39; // r14
  int v40; // eax
  __int64 *v41; // r12
  __int64 v42; // rdx
  unsigned __int64 v43; // rax
  __int64 v44; // rdx
  __int64 *v45; // rax
  __int64 *v46; // rax
  unsigned __int8 v47; // r13
  __int64 v48; // r13
  _QWORD *v49; // rax
  _WORD *v50; // rdx
  __int64 v51; // rdi
  _QWORD *v52; // rdx
  _QWORD *v53; // rdx
  void *v54; // rax
  _QWORD *v55; // rax
  _WORD *v56; // rdx
  void *v57; // rax
  _QWORD *v58; // rax
  _DWORD *v59; // rdx
  __int64 v60; // r13
  _BYTE *v61; // rax
  unsigned __int8 **v62; // rbx
  __int64 v63; // rax
  _QWORD *v64; // r14
  unsigned __int8 *v65; // r13
  _QWORD *v66; // rax
  _WORD *v67; // rdx
  __int64 v68; // r8
  _QWORD *v69; // rdx
  _WORD *v70; // rdx
  _BYTE *v71; // r8
  __int64 v72; // rdx
  _BYTE *v73; // rax
  unsigned __int8 *v74; // r12
  unsigned __int8 v75; // al
  _QWORD *v76; // rax
  _WORD *v77; // rdx
  __int64 v78; // r8
  _QWORD *v79; // rdx
  __int64 v80; // rax
  _WORD *v81; // rdx
  __int64 v82; // rax
  _BYTE *v83; // r8
  __int64 v84; // rdx
  _BYTE *v85; // rax
  __int64 v86; // r13
  _QWORD *v87; // rax
  _WORD *v88; // rdx
  __int64 v89; // rdi
  void *v90; // rdx
  _QWORD *v91; // rdx
  void *v92; // rax
  _QWORD *v93; // rax
  _WORD *v94; // rdx
  void *v95; // rax
  _QWORD *v96; // rax
  _DWORD *v97; // rdx
  __int64 v98; // r13
  _BYTE *v99; // rax
  _QWORD *v100; // rax
  _WORD *v101; // rdx
  __int64 v102; // r8
  void *v103; // rdx
  _WORD *v104; // rdx
  _BYTE *v105; // r8
  __int64 v106; // rdx
  _BYTE *v107; // rax
  __int64 v108; // r13
  _QWORD *v109; // rax
  _WORD *v110; // rdx
  __int64 v111; // rdi
  _QWORD *v112; // rdx
  _QWORD *v113; // rdx
  void *v114; // rax
  _QWORD *v115; // rax
  _WORD *v116; // rdx
  void *v117; // rax
  _QWORD *v118; // rax
  _DWORD *v119; // rdx
  __int64 v120; // r13
  _BYTE *v121; // rax
  __int64 v122; // r13
  _QWORD *v123; // rax
  _WORD *v124; // rdx
  __int64 v125; // rdi
  _QWORD *v126; // rdx
  _QWORD *v127; // rdx
  void *v128; // rax
  _QWORD *v129; // rax
  _WORD *v130; // rdx
  void *v131; // rax
  _QWORD *v132; // rax
  _DWORD *v133; // rdx
  __int64 v134; // r13
  _BYTE *v135; // rax
  _QWORD *v136; // rax
  _WORD *v137; // rdx
  __int64 v138; // r8
  _QWORD *v139; // rdx
  _WORD *v140; // rdx
  _BYTE *v141; // r8
  __int64 v142; // rdx
  _BYTE *v143; // rax
  __int64 v144; // rax
  __int64 v145; // rax
  __int64 v146; // rax
  __int64 v147; // rax
  __int64 v148; // rax
  __int64 v149; // rax
  __int64 v150; // rax
  __int64 v151; // rax
  __int64 v152; // rax
  __int64 v153; // rax
  __int64 v154; // rax
  __int64 v155; // rax
  __int64 v156; // rax
  __int64 v157; // rax
  __int64 v159; // r12
  int v160; // eax
  unsigned __int8 *v161; // r13
  __int64 v162; // rax
  __int64 v163; // rax
  __int64 v164; // rax
  unsigned __int64 v165; // rax
  __int64 v166; // rcx
  unsigned __int8 **v167; // rax
  __int64 v168; // rdx
  unsigned __int8 **v169; // rsi
  __int64 v170; // rcx
  __int64 v171; // rdx
  unsigned __int8 **v172; // rdx
  __int64 v173; // rax
  __int64 *v174; // r13
  __int64 v175; // rdx
  __int64 *v176; // rax
  int v177; // edx
  __int64 v178; // rcx
  unsigned __int64 v179; // rax
  __int64 v180; // rax
  int v181; // ecx
  unsigned __int8 **v182; // rdx
  unsigned __int8 **v183; // rax
  unsigned __int8 *v184; // rdi
  int v185; // edx
  unsigned __int8 **v186; // rdi
  unsigned int v187; // r11d
  int v188; // ecx
  unsigned __int8 *v189; // rsi
  unsigned int v190; // ecx
  unsigned __int8 *v191; // r11
  __int64 v192; // rax
  unsigned __int64 v193; // rdx
  int v194; // edi
  unsigned __int8 **v195; // rsi
  int v196; // edi
  __int64 *v197; // rsi
  int v198; // edi
  __int64 *v199; // rcx
  int v200; // edi
  __int64 v201; // rcx
  unsigned __int8 **v202; // [rsp+18h] [rbp-4C8h]
  unsigned __int8 *v203; // [rsp+20h] [rbp-4C0h]
  unsigned __int8 *v204; // [rsp+20h] [rbp-4C0h]
  unsigned __int8 *v205; // [rsp+20h] [rbp-4C0h]
  unsigned __int8 *v206; // [rsp+20h] [rbp-4C0h]
  __int64 v207; // [rsp+28h] [rbp-4B8h]
  __int64 v208; // [rsp+28h] [rbp-4B8h]
  __int64 v209; // [rsp+28h] [rbp-4B8h]
  __int64 v210; // [rsp+28h] [rbp-4B8h]
  unsigned __int8 **v211; // [rsp+30h] [rbp-4B0h]
  __int64 v212; // [rsp+38h] [rbp-4A8h]
  __int64 *v213; // [rsp+58h] [rbp-488h]
  __int64 *v214; // [rsp+58h] [rbp-488h]
  __int64 *v216; // [rsp+68h] [rbp-478h]
  unsigned __int8 *v217; // [rsp+68h] [rbp-478h]
  unsigned __int8 **v218; // [rsp+68h] [rbp-478h]
  __int64 *v219; // [rsp+68h] [rbp-478h]
  __int64 *v220; // [rsp+78h] [rbp-468h]
  __int64 *v221; // [rsp+78h] [rbp-468h]
  __int64 v222; // [rsp+78h] [rbp-468h]
  __int64 v223; // [rsp+78h] [rbp-468h]
  __int64 v224; // [rsp+78h] [rbp-468h]
  __int64 v225; // [rsp+78h] [rbp-468h]
  __int64 v226; // [rsp+78h] [rbp-468h]
  __int64 v227; // [rsp+78h] [rbp-468h]
  __int64 v228; // [rsp+78h] [rbp-468h]
  __int64 v229; // [rsp+78h] [rbp-468h]
  unsigned int v230; // [rsp+78h] [rbp-468h]
  unsigned __int64 v231; // [rsp+80h] [rbp-460h]
  __int64 v232; // [rsp+80h] [rbp-460h]
  unsigned __int8 **v233; // [rsp+80h] [rbp-460h]
  unsigned __int8 **v235; // [rsp+88h] [rbp-458h]
  __int64 v236; // [rsp+90h] [rbp-450h] BYREF
  __int64 v237; // [rsp+98h] [rbp-448h]
  __int64 v238; // [rsp+A0h] [rbp-440h]
  __int64 v239; // [rsp+A8h] [rbp-438h]
  __int64 *v240; // [rsp+B0h] [rbp-430h]
  __int64 v241; // [rsp+B8h] [rbp-428h]
  __int64 v242; // [rsp+C0h] [rbp-420h] BYREF
  __int64 v243; // [rsp+C8h] [rbp-418h]
  __int64 v244; // [rsp+D0h] [rbp-410h]
  __int64 v245; // [rsp+D8h] [rbp-408h]
  __int64 *v246; // [rsp+E0h] [rbp-400h]
  __int64 v247; // [rsp+E8h] [rbp-3F8h]
  __int64 v248; // [rsp+F0h] [rbp-3F0h] BYREF
  __int64 v249; // [rsp+F8h] [rbp-3E8h]
  __int64 v250; // [rsp+100h] [rbp-3E0h]
  __int64 v251; // [rsp+108h] [rbp-3D8h]
  __m128i *v252; // [rsp+110h] [rbp-3D0h]
  __int64 v253; // [rsp+118h] [rbp-3C8h]
  __m128i v254; // [rsp+120h] [rbp-3C0h] BYREF
  __int64 v255; // [rsp+130h] [rbp-3B0h]
  __int64 v256; // [rsp+138h] [rbp-3A8h]
  __int64 v257; // [rsp+140h] [rbp-3A0h]
  __int64 v258; // [rsp+148h] [rbp-398h]
  char v259; // [rsp+150h] [rbp-390h]
  __int64 v260; // [rsp+160h] [rbp-380h] BYREF
  __int64 v261; // [rsp+168h] [rbp-378h]
  __int64 v262; // [rsp+170h] [rbp-370h]
  __int64 v263; // [rsp+178h] [rbp-368h]
  unsigned __int8 **v264; // [rsp+180h] [rbp-360h] BYREF
  __int64 v265; // [rsp+188h] [rbp-358h]
  _BYTE v266[128]; // [rsp+190h] [rbp-350h] BYREF
  __m128i v267; // [rsp+210h] [rbp-2D0h] BYREF
  __int64 v268; // [rsp+220h] [rbp-2C0h]
  __int64 v269; // [rsp+228h] [rbp-2B8h] BYREF
  __int64 v270; // [rsp+230h] [rbp-2B0h]
  __int64 v271; // [rsp+238h] [rbp-2A8h]
  _QWORD v272[2]; // [rsp+368h] [rbp-178h] BYREF
  char v273; // [rsp+378h] [rbp-168h]
  _BYTE *v274; // [rsp+380h] [rbp-160h]
  __int64 v275; // [rsp+388h] [rbp-158h]
  _BYTE v276[128]; // [rsp+390h] [rbp-150h] BYREF
  __int16 v277; // [rsp+410h] [rbp-D0h]
  _QWORD v278[2]; // [rsp+418h] [rbp-C8h] BYREF
  __int64 v279; // [rsp+428h] [rbp-B8h]
  __int64 v280; // [rsp+430h] [rbp-B0h] BYREF
  unsigned int v281; // [rsp+438h] [rbp-A8h]
  char v282; // [rsp+4B0h] [rbp-30h] BYREF

  v4 = sub_B2BEC0(a2);
  ++*a1;
  v7 = a2 + 72;
  v8 = *(_QWORD *)(a2 + 80);
  v212 = v4;
  v240 = &v242;
  v264 = (unsigned __int8 **)v266;
  v265 = 0x1000000000LL;
  v246 = &v248;
  v236 = 0;
  v237 = 0;
  v238 = 0;
  v239 = 0;
  v241 = 0;
  v260 = 0;
  v261 = 0;
  v262 = 0;
  v263 = 0;
  v242 = 0;
  v243 = 0;
  v244 = 0;
  v245 = 0;
  v247 = 0;
  v248 = 0;
  v249 = 0;
  v250 = 0;
  v251 = 0;
  v252 = &v254;
  v253 = 0;
  if ( a2 + 72 == v8 )
  {
    i = 0;
  }
  else
  {
    if ( !v8 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v8 + 32);
      if ( i != v8 + 24 )
        break;
      v8 = *(_QWORD *)(v8 + 8);
      if ( v7 == v8 )
        goto LABEL_7;
      if ( !v8 )
        BUG();
    }
  }
  if ( v7 != v8 )
  {
    v159 = a2 + 72;
    do
    {
      if ( !i )
        BUG();
      v160 = *(unsigned __int8 *)(i - 24);
      v161 = (unsigned __int8 *)(i - 24);
      if ( (_BYTE)v160 == 61 )
      {
        v162 = *(_QWORD *)(i - 16);
        v267.m128i_i64[0] = *(_QWORD *)(i - 56);
        v267.m128i_i64[1] = v162;
        sub_30A0B80((__int64)&v236, &v267);
        v267.m128i_i64[0] = i - 24;
        sub_30A0EF0((__int64)&v242, v267.m128i_i64);
        goto LABEL_262;
      }
      if ( (_BYTE)v160 == 62 )
      {
        v164 = *(_QWORD *)(*(_QWORD *)(i - 88) + 8LL);
        v267.m128i_i64[0] = *(_QWORD *)(i - 56);
        v267.m128i_i64[1] = v164;
        sub_30A0B80((__int64)&v236, &v267);
        v267.m128i_i64[0] = i - 24;
        sub_30A0EF0((__int64)&v248, v267.m128i_i64);
        goto LABEL_262;
      }
      v165 = (unsigned int)(v160 - 34);
      if ( (unsigned __int8)v165 <= 0x33u )
      {
        v166 = 0x8000000000041LL;
        if ( _bittest64(&v166, v165) )
        {
          if ( (_DWORD)v262 )
          {
            if ( (_DWORD)v263 )
            {
              v181 = 1;
              v6 = v261;
              v230 = ((unsigned int)v161 >> 9) ^ ((unsigned int)v161 >> 4);
              v182 = 0;
              v5 = ((_DWORD)v263 - 1) & v230;
              v183 = (unsigned __int8 **)(v261 + 8 * v5);
              v184 = *v183;
              if ( v161 == *v183 )
                goto LABEL_262;
              while ( v184 != (unsigned __int8 *)-4096LL )
              {
                if ( !v182 && v184 == (unsigned __int8 *)-8192LL )
                  v182 = v183;
                v5 = ((_DWORD)v263 - 1) & (unsigned int)(v5 + v181);
                v183 = (unsigned __int8 **)(v261 + 8 * v5);
                v184 = *v183;
                if ( v161 == *v183 )
                  goto LABEL_262;
                ++v181;
              }
              if ( v182 )
                v183 = v182;
              v185 = v262 + 1;
              ++v260;
              if ( 4 * ((int)v262 + 1) < (unsigned int)(3 * v263) )
              {
                if ( (int)v263 - HIDWORD(v262) - v185 <= (unsigned int)v263 >> 3 )
                {
                  sub_24FB720((__int64)&v260, v263);
                  if ( !(_DWORD)v263 )
                    goto LABEL_402;
                  v5 = (unsigned int)(v263 - 1);
                  v186 = 0;
                  v6 = v261;
                  v187 = v5 & v230;
                  v185 = v262 + 1;
                  v188 = 1;
                  v183 = (unsigned __int8 **)(v261 + 8LL * ((unsigned int)v5 & v230));
                  v189 = *v183;
                  if ( v161 != *v183 )
                  {
                    while ( v189 != (unsigned __int8 *)-4096LL )
                    {
                      if ( !v186 && v189 == (unsigned __int8 *)-8192LL )
                        v186 = v183;
                      v187 = v5 & (v188 + v187);
                      v183 = (unsigned __int8 **)(v261 + 8LL * v187);
                      v189 = *v183;
                      if ( v161 == *v183 )
                        goto LABEL_348;
                      ++v188;
                    }
                    if ( v186 )
                      v183 = v186;
                  }
                }
LABEL_348:
                LODWORD(v262) = v185;
                if ( *v183 != (unsigned __int8 *)-4096LL )
                  --HIDWORD(v262);
                *v183 = v161;
                v192 = (unsigned int)v265;
                v193 = (unsigned int)v265 + 1LL;
                if ( v193 > HIDWORD(v265) )
                {
                  sub_C8D5F0((__int64)&v264, v266, v193, 8u, v5, v6);
                  v192 = (unsigned int)v265;
                }
                v264[v192] = v161;
                LODWORD(v265) = v265 + 1;
                goto LABEL_262;
              }
            }
            else
            {
              ++v260;
            }
            sub_24FB720((__int64)&v260, 2 * v263);
            if ( !(_DWORD)v263 )
            {
LABEL_402:
              LODWORD(v262) = v262 + 1;
              BUG();
            }
            v5 = (unsigned int)(v263 - 1);
            v6 = v261;
            v185 = v262 + 1;
            v190 = v5 & (((unsigned int)v161 >> 9) ^ ((unsigned int)v161 >> 4));
            v183 = (unsigned __int8 **)(v261 + 8LL * v190);
            v191 = *v183;
            if ( v161 != *v183 )
            {
              v194 = 1;
              v195 = 0;
              while ( v191 != (unsigned __int8 *)-4096LL )
              {
                if ( !v195 && v191 == (unsigned __int8 *)-8192LL )
                  v195 = v183;
                v190 = v5 & (v194 + v190);
                v183 = (unsigned __int8 **)(v261 + 8LL * v190);
                v191 = *v183;
                if ( v161 == *v183 )
                  goto LABEL_348;
                ++v194;
              }
              if ( v195 )
                v183 = v195;
            }
            goto LABEL_348;
          }
          v167 = v264;
          v168 = 8LL * (unsigned int)v265;
          v169 = &v264[(unsigned __int64)v168 / 8];
          v170 = v168 >> 3;
          v171 = v168 >> 5;
          if ( !v171 )
            goto LABEL_335;
          v172 = &v264[4 * v171];
          do
          {
            if ( v161 == *v167 )
              goto LABEL_283;
            if ( v161 == v167[1] )
            {
              ++v167;
              goto LABEL_283;
            }
            if ( v161 == v167[2] )
            {
              v167 += 2;
              goto LABEL_283;
            }
            if ( v161 == v167[3] )
            {
              v167 += 3;
              goto LABEL_283;
            }
            v167 += 4;
          }
          while ( v172 != v167 );
          v170 = v169 - v167;
LABEL_335:
          if ( v170 == 2 )
            goto LABEL_343;
          if ( v170 != 3 )
          {
            if ( v170 == 1 )
              goto LABEL_338;
            goto LABEL_284;
          }
          if ( v161 == *v167 )
            goto LABEL_283;
          ++v167;
LABEL_343:
          if ( v161 == *v167 )
            goto LABEL_283;
          ++v167;
LABEL_338:
          if ( v161 == *v167 )
          {
LABEL_283:
            if ( v169 == v167 )
              goto LABEL_284;
            goto LABEL_262;
          }
LABEL_284:
          if ( (unsigned __int64)(unsigned int)v265 + 1 > HIDWORD(v265) )
          {
            sub_C8D5F0((__int64)&v264, v266, (unsigned int)v265 + 1LL, 8u, v5, v6);
            v169 = &v264[(unsigned int)v265];
          }
          *v169 = v161;
          v173 = (unsigned int)(v265 + 1);
          LODWORD(v265) = v173;
          if ( (unsigned int)v173 > 0x10 )
          {
            v174 = (__int64 *)v264;
            v219 = (__int64 *)&v264[v173];
            while ( (_DWORD)v263 )
            {
              v6 = *v174;
              LODWORD(v175) = (v263 - 1) & (((unsigned int)*v174 >> 9) ^ ((unsigned int)*v174 >> 4));
              v176 = (__int64 *)(v261 + 8LL * (unsigned int)v175);
              v5 = *v176;
              if ( *v174 != *v176 )
              {
                v198 = 1;
                v199 = 0;
                while ( v5 != -4096 )
                {
                  if ( !v199 && v5 == -8192 )
                    v199 = v176;
                  v175 = ((_DWORD)v263 - 1) & (unsigned int)(v175 + v198);
                  v176 = (__int64 *)(v261 + 8 * v175);
                  v5 = *v176;
                  if ( v6 == *v176 )
                    goto LABEL_289;
                  ++v198;
                }
                if ( v199 )
                  v176 = v199;
                ++v260;
                v177 = v262 + 1;
                if ( 4 * ((int)v262 + 1) < (unsigned int)(3 * v263) )
                {
                  if ( (int)v263 - HIDWORD(v262) - v177 <= (unsigned int)v263 >> 3 )
                  {
                    sub_24FB720((__int64)&v260, v263);
                    if ( !(_DWORD)v263 )
                    {
LABEL_404:
                      LODWORD(v262) = v262 + 1;
                      BUG();
                    }
                    v5 = (unsigned int)(v263 - 1);
                    v200 = 1;
                    v177 = v262 + 1;
                    v197 = 0;
                    LODWORD(v201) = v5 & (((unsigned int)*v174 >> 9) ^ ((unsigned int)*v174 >> 4));
                    v176 = (__int64 *)(v261 + 8LL * (unsigned int)v201);
                    v6 = *v176;
                    if ( *v174 != *v176 )
                    {
                      while ( v6 != -4096 )
                      {
                        if ( !v197 && v6 == -8192 )
                          v197 = v176;
                        v201 = (unsigned int)v5 & ((_DWORD)v201 + v200);
                        v176 = (__int64 *)(v261 + 8 * v201);
                        v6 = *v176;
                        if ( *v174 == *v176 )
                          goto LABEL_294;
                        ++v200;
                      }
LABEL_365:
                      if ( v197 )
                        v176 = v197;
                    }
                  }
LABEL_294:
                  LODWORD(v262) = v177;
                  if ( *v176 != -4096 )
                    --HIDWORD(v262);
                  *v176 = *v174;
                  goto LABEL_289;
                }
LABEL_292:
                sub_24FB720((__int64)&v260, 2 * v263);
                if ( !(_DWORD)v263 )
                  goto LABEL_404;
                v6 = (unsigned int)(v263 - 1);
                v177 = v262 + 1;
                LODWORD(v178) = v6 & (((unsigned int)*v174 >> 9) ^ ((unsigned int)*v174 >> 4));
                v176 = (__int64 *)(v261 + 8LL * (unsigned int)v178);
                v5 = *v176;
                if ( *v176 != *v174 )
                {
                  v196 = 1;
                  v197 = 0;
                  while ( v5 != -4096 )
                  {
                    if ( v5 == -8192 && !v197 )
                      v197 = v176;
                    v178 = (unsigned int)v6 & ((_DWORD)v178 + v196);
                    v176 = (__int64 *)(v261 + 8 * v178);
                    v5 = *v176;
                    if ( *v174 == *v176 )
                      goto LABEL_294;
                    ++v196;
                  }
                  goto LABEL_365;
                }
                goto LABEL_294;
              }
LABEL_289:
              if ( v219 == ++v174 )
                goto LABEL_262;
            }
            ++v260;
            goto LABEL_292;
          }
        }
      }
LABEL_262:
      for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v8 + 32) )
      {
        v163 = v8 - 24;
        if ( !v8 )
          v163 = 0;
        if ( i != v163 + 48 )
          break;
        v8 = *(_QWORD *)(v8 + 8);
        if ( v159 == v8 )
          goto LABEL_7;
        if ( !v8 )
          BUG();
      }
    }
    while ( v159 != v8 );
  }
LABEL_7:
  if ( !(_BYTE)qword_502E088
    && !(_BYTE)qword_502DFA8
    && !(_BYTE)qword_502DEC8
    && !(_BYTE)qword_502DDE8
    && !(_BYTE)qword_502DD08
    && !(_BYTE)qword_502DC28
    && !byte_502DA68
    && !byte_502DB48
    && !(_BYTE)qword_502D988 )
  {
    goto LABEL_19;
  }
  v10 = (__int64)sub_CB72A0();
  v11 = *(void **)(v10 + 32);
  if ( *(_QWORD *)(v10 + 24) - (_QWORD)v11 <= 9u )
  {
    v10 = sub_CB6200(v10, (unsigned __int8 *)"Function: ", 0xAu);
  }
  else
  {
    qmemcpy(v11, "Function: ", 10);
    *(_QWORD *)(v10 + 32) += 10LL;
  }
  v12 = sub_BD5D20(a2);
  v14 = *(_WORD **)(v10 + 32);
  v15 = (unsigned __int8 *)v12;
  v16 = v13;
  v17 = *(_QWORD *)(v10 + 24) - (_QWORD)v14;
  if ( v13 > v17 )
  {
    v180 = sub_CB6200(v10, v15, v13);
    v14 = *(_WORD **)(v180 + 32);
    v10 = v180;
    v17 = *(_QWORD *)(v180 + 24) - (_QWORD)v14;
  }
  else if ( v13 )
  {
    memcpy(v14, v15, v13);
    v14 = (_WORD *)(v16 + *(_QWORD *)(v10 + 32));
    v179 = *(_QWORD *)(v10 + 24) - (_QWORD)v14;
    *(_QWORD *)(v10 + 32) = v14;
    if ( v179 > 1 )
      goto LABEL_14;
LABEL_306:
    v10 = sub_CB6200(v10, (unsigned __int8 *)": ", 2u);
    goto LABEL_15;
  }
  if ( v17 <= 1 )
    goto LABEL_306;
LABEL_14:
  *v14 = 8250;
  *(_QWORD *)(v10 + 32) += 2LL;
LABEL_15:
  v18 = sub_CB59D0(v10, (unsigned int)v241);
  v19 = *(void **)(v18 + 32);
  if ( *(_QWORD *)(v18 + 24) - (_QWORD)v19 <= 0xAu )
  {
    v18 = sub_CB6200(v18, " pointers, ", 0xBu);
  }
  else
  {
    qmemcpy(v19, " pointers, ", 11);
    *(_QWORD *)(v18 + 32) += 11LL;
  }
  v20 = sub_CB59D0(v18, (unsigned int)v265);
  v21 = *(void **)(v20 + 32);
  if ( *(_QWORD *)(v20 + 24) - (_QWORD)v21 <= 0xBu )
  {
    sub_CB6200(v20, " call sites\n", 0xCu);
  }
  else
  {
    qmemcpy(v21, " call sites\n", 12);
    *(_QWORD *)(v20 + 32) += 12LL;
  }
LABEL_19:
  v22 = v240;
  v220 = &v240[2 * (unsigned int)v241];
  if ( v240 != v220 )
  {
    while ( 1 )
    {
      v267.m128i_i64[0] = sub_9208B0(v212, v22[1]);
      v267.m128i_i64[1] = v23;
      v231 = (unsigned __int64)(v267.m128i_i64[0] + 7) >> 3;
      if ( (_BYTE)v23 )
        v231 = ((unsigned __int64)(v267.m128i_i64[0] + 7) >> 3) | 0x4000000000000000LL;
      if ( v240 != v22 )
        break;
LABEL_32:
      v22 += 2;
      if ( v220 == v22 )
        goto LABEL_33;
    }
    v24 = v240;
    while ( 1 )
    {
      v267.m128i_i64[0] = sub_9208B0(v212, v24[1]);
      v267.m128i_i64[1] = v25;
      v26 = (unsigned __int64)(v267.m128i_i64[0] + 7) >> 3;
      if ( (_BYTE)v25 )
        v26 |= 0x4000000000000000uLL;
      v27 = *v22;
      v28 = *v24;
      v267.m128i_i64[1] = v26;
      v268 = 0;
      v254.m128i_i64[0] = v27;
      v267.m128i_i64[0] = v28;
      v269 = 0;
      v270 = 0;
      v271 = 0;
      v254.m128i_i64[1] = v231;
      v255 = 0;
      v256 = 0;
      v257 = 0;
      v258 = 0;
      v29 = sub_CF4E00((__int64)a3, (__int64)&v254, (__int64)&v267);
      v30 = v29;
      if ( (_BYTE)v29 == 2 )
        break;
      if ( (unsigned __int8)v29 > 2u )
      {
        if ( (_BYTE)v29 == 3 )
        {
          LOBYTE(v30) = 3;
          sub_309F310(v30, qword_502DD08, *v22, v22[1], *v24, v24[1], *(_QWORD *)(a2 + 40));
          ++a1[4];
        }
        goto LABEL_25;
      }
      if ( (_BYTE)v29 )
      {
        LOBYTE(v30) = 1;
        sub_309F310(v30, qword_502DEC8, *v22, v22[1], *v24, v24[1], *(_QWORD *)(a2 + 40));
        ++a1[2];
LABEL_25:
        v24 += 2;
        if ( v24 == v22 )
          goto LABEL_32;
      }
      else
      {
        v31 = v24[1];
        LOBYTE(v30) = 0;
        v24 += 2;
        sub_309F310(v30, qword_502DFA8, *v22, v22[1], *(v24 - 2), v31, *(_QWORD *)(a2 + 40));
        ++a1[1];
        if ( v24 == v22 )
          goto LABEL_32;
      }
    }
    LOBYTE(v30) = 2;
    sub_309F310(v30, qword_502DDE8, *v22, v22[1], *v24, v24[1], *(_QWORD *)(a2 + 40));
    ++a1[3];
    goto LABEL_25;
  }
LABEL_33:
  if ( !(_BYTE)qword_502D8A8 )
    goto LABEL_60;
  v32 = (__int64 *)v252;
  v213 = &v246[(unsigned int)v247];
  v33 = (unsigned int)v253;
  if ( v246 == v213 )
  {
    v38 = &v252->m128i_i64[(unsigned int)v253];
    goto LABEL_47;
  }
  v216 = v246;
  v34 = (unsigned __int64)v252;
  do
  {
    v35 = *v216;
    v221 = (__int64 *)(v34 + 8 * v33);
    if ( v221 != (__int64 *)v34 )
    {
      v36 = (__int64 *)v34;
      while ( 1 )
      {
        v232 = *v36;
        sub_D66630(&v267, *v36);
        sub_D665A0(&v254, v35);
        v37 = sub_CF4E00((__int64)a3, (__int64)&v254, (__int64)&v267);
        if ( (_BYTE)v37 == 2 )
          break;
        if ( (unsigned __int8)v37 > 2u )
        {
          if ( (_BYTE)v37 == 3 )
          {
            sub_309F1B0(v37, qword_502DD08, v35, v232);
            ++a1[4];
          }
          goto LABEL_39;
        }
        if ( (_BYTE)v37 )
        {
          sub_309F1B0(v37, qword_502DEC8, v35, v232);
          ++a1[2];
LABEL_39:
          if ( v221 == ++v36 )
            goto LABEL_44;
        }
        else
        {
          ++v36;
          sub_309F1B0(v37, qword_502DFA8, v35, v232);
          ++a1[1];
          if ( v221 == v36 )
          {
LABEL_44:
            v34 = (unsigned __int64)v252;
            v33 = (unsigned int)v253;
            v38 = &v252->m128i_i64[(unsigned int)v253];
            goto LABEL_45;
          }
        }
      }
      sub_309F1B0(v37, qword_502DDE8, v35, v232);
      ++a1[3];
      goto LABEL_39;
    }
    v38 = (__int64 *)v34;
LABEL_45:
    ++v216;
  }
  while ( v213 != v216 );
  v32 = (__int64 *)v34;
LABEL_47:
  if ( v32 != v38 )
  {
    while ( ++v32 != v38 )
    {
      v39 = (__int64 *)v252;
      if ( v32 != (__int64 *)v252 )
      {
        do
        {
          sub_D66630(&v267, *v39);
          sub_D66630(&v254, *v32);
          v40 = sub_CF4E00((__int64)a3, (__int64)&v254, (__int64)&v267);
          if ( (_BYTE)v40 == 2 )
          {
            sub_309F1B0(v40, qword_502DDE8, *v32, *v39);
            ++a1[3];
          }
          else if ( (unsigned __int8)v40 > 2u )
          {
            if ( (_BYTE)v40 == 3 )
            {
              sub_309F1B0(v40, qword_502DD08, *v32, *v39);
              ++a1[4];
            }
          }
          else if ( (_BYTE)v40 )
          {
            sub_309F1B0(v40, qword_502DEC8, *v32, *v39);
            ++a1[2];
          }
          else
          {
            sub_309F1B0(v40, qword_502DFA8, *v32, *v39);
            ++a1[1];
          }
          ++v39;
        }
        while ( v39 != v32 );
      }
    }
  }
LABEL_60:
  v202 = &v264[(unsigned int)v265];
  if ( v202 != v264 )
  {
    v211 = v264;
    do
    {
      v41 = v240;
      v217 = *v211;
      v214 = &v240[2 * (unsigned int)v241];
      if ( v214 != v240 )
      {
        do
        {
          v267.m128i_i64[0] = sub_9208B0(v212, v41[1]);
          v267.m128i_i64[1] = v42;
          v43 = (unsigned __int64)(v267.m128i_i64[0] + 7) >> 3;
          if ( (_BYTE)v42 )
            v43 |= 0x4000000000000000uLL;
          v44 = *v41;
          v259 = 1;
          v255 = 0;
          v254.m128i_i64[0] = v44;
          v256 = 0;
          v257 = 0;
          v258 = 0;
          v267 = (__m128i)(unsigned __int64)a3;
          v268 = 1;
          v254.m128i_i64[1] = v43;
          v45 = &v269;
          do
          {
            *v45 = -4;
            v45 += 5;
            *(v45 - 4) = -3;
            *(v45 - 3) = -4;
            *(v45 - 2) = -3;
          }
          while ( v45 != v272 );
          v272[0] = v278;
          v272[1] = 0;
          v274 = v276;
          v275 = 0x400000000LL;
          v273 = 0;
          v277 = 256;
          v278[1] = 0;
          v279 = 1;
          v278[0] = &unk_49DDBE8;
          v46 = &v280;
          do
          {
            *v46 = -4096;
            v46 += 2;
          }
          while ( v46 != (__int64 *)&v282 );
          v47 = sub_CF63E0(a3, v217, &v254, (__int64)&v267);
          v278[0] = &unk_49DDBE8;
          if ( (v279 & 1) == 0 )
            sub_C7D6A0(v280, 16LL * v281, 8);
          nullsub_184();
          if ( v274 != v276 )
            _libc_free((unsigned __int64)v274);
          if ( (v268 & 1) == 0 )
            sub_C7D6A0(v269, 40LL * (unsigned int)v270, 8);
          if ( v47 == 2 )
          {
            if ( byte_502DA68 || (_BYTE)qword_502E088 )
            {
              v122 = v41[1];
              v210 = *(_QWORD *)(a2 + 40);
              v206 = (unsigned __int8 *)*v41;
              v123 = sub_CB72A0();
              v124 = (_WORD *)v123[4];
              v125 = (__int64)v123;
              if ( v123[3] - (_QWORD)v124 <= 1u )
              {
                v145 = sub_CB6200((__int64)v123, (unsigned __int8 *)"  ", 2u);
                v126 = *(_QWORD **)(v145 + 32);
                v125 = v145;
              }
              else
              {
                *v124 = 8224;
                v126 = (_QWORD *)(v123[4] + 2LL);
                v123[4] = v126;
              }
              if ( *(_QWORD *)(v125 + 24) - (_QWORD)v126 <= 7u )
              {
                v144 = sub_CB6200(v125, "Just Mod", 8u);
                v127 = *(_QWORD **)(v144 + 32);
                v125 = v144;
              }
              else
              {
                *v126 = 0x646F4D207473754ALL;
                v127 = (_QWORD *)(*(_QWORD *)(v125 + 32) + 8LL);
                *(_QWORD *)(v125 + 32) = v127;
              }
              if ( *(_QWORD *)(v125 + 24) - (_QWORD)v127 <= 7u )
              {
                sub_CB6200(v125, ":  Ptr: ", 8u);
              }
              else
              {
                *v127 = 0x203A72745020203ALL;
                *(_QWORD *)(v125 + 32) += 8LL;
              }
              v128 = sub_CB72A0();
              sub_A587F0(v122, (__int64)v128, 0, 1);
              v129 = sub_CB72A0();
              v130 = (_WORD *)v129[4];
              if ( v129[3] - (_QWORD)v130 <= 1u )
              {
                sub_CB6200((__int64)v129, (unsigned __int8 *)"* ", 2u);
              }
              else
              {
                *v130 = 8234;
                v129[4] += 2LL;
              }
              v131 = sub_CB72A0();
              sub_A5BF40(v206, (__int64)v131, 0, v210);
              v132 = sub_CB72A0();
              v133 = (_DWORD *)v132[4];
              v134 = (__int64)v132;
              if ( v132[3] - (_QWORD)v133 <= 3u )
              {
                v134 = sub_CB6200((__int64)v132, "\t<->", 4u);
              }
              else
              {
                *v133 = 1043151881;
                v132[4] += 4LL;
              }
              sub_A69870((__int64)v217, (_BYTE *)v134, 0);
              v135 = *(_BYTE **)(v134 + 32);
              if ( (unsigned __int64)v135 >= *(_QWORD *)(v134 + 24) )
              {
                sub_CB5D20(v134, 10);
              }
              else
              {
                *(_QWORD *)(v134 + 32) = v135 + 1;
                *v135 = 10;
              }
            }
            ++a1[6];
          }
          else if ( v47 > 2u )
          {
            if ( v47 == 3 )
            {
              if ( (_BYTE)qword_502D988 || (_BYTE)qword_502E088 )
              {
                v86 = v41[1];
                v208 = *(_QWORD *)(a2 + 40);
                v204 = (unsigned __int8 *)*v41;
                v87 = sub_CB72A0();
                v88 = (_WORD *)v87[4];
                v89 = (__int64)v87;
                if ( v87[3] - (_QWORD)v88 <= 1u )
                {
                  v149 = sub_CB6200((__int64)v87, (unsigned __int8 *)"  ", 2u);
                  v90 = *(void **)(v149 + 32);
                  v89 = v149;
                }
                else
                {
                  *v88 = 8224;
                  v90 = (void *)(v87[4] + 2LL);
                  v87[4] = v90;
                }
                if ( *(_QWORD *)(v89 + 24) - (_QWORD)v90 <= 0xAu )
                {
                  v148 = sub_CB6200(v89, "Both ModRef", 0xBu);
                  v91 = *(_QWORD **)(v148 + 32);
                  v89 = v148;
                }
                else
                {
                  qmemcpy(v90, "Both ModRef", 11);
                  v91 = (_QWORD *)(*(_QWORD *)(v89 + 32) + 11LL);
                  *(_QWORD *)(v89 + 32) = v91;
                }
                if ( *(_QWORD *)(v89 + 24) - (_QWORD)v91 <= 7u )
                {
                  sub_CB6200(v89, ":  Ptr: ", 8u);
                }
                else
                {
                  *v91 = 0x203A72745020203ALL;
                  *(_QWORD *)(v89 + 32) += 8LL;
                }
                v92 = sub_CB72A0();
                sub_A587F0(v86, (__int64)v92, 0, 1);
                v93 = sub_CB72A0();
                v94 = (_WORD *)v93[4];
                if ( v93[3] - (_QWORD)v94 <= 1u )
                {
                  sub_CB6200((__int64)v93, (unsigned __int8 *)"* ", 2u);
                }
                else
                {
                  *v94 = 8234;
                  v93[4] += 2LL;
                }
                v95 = sub_CB72A0();
                sub_A5BF40(v204, (__int64)v95, 0, v208);
                v96 = sub_CB72A0();
                v97 = (_DWORD *)v96[4];
                v98 = (__int64)v96;
                if ( v96[3] - (_QWORD)v97 <= 3u )
                {
                  v98 = sub_CB6200((__int64)v96, "\t<->", 4u);
                }
                else
                {
                  *v97 = 1043151881;
                  v96[4] += 4LL;
                }
                sub_A69870((__int64)v217, (_BYTE *)v98, 0);
                v99 = *(_BYTE **)(v98 + 32);
                if ( (unsigned __int64)v99 >= *(_QWORD *)(v98 + 24) )
                {
                  sub_CB5D20(v98, 10);
                }
                else
                {
                  *(_QWORD *)(v98 + 32) = v99 + 1;
                  *v99 = 10;
                }
              }
              ++a1[8];
            }
          }
          else if ( v47 )
          {
            if ( byte_502DB48 || (_BYTE)qword_502E088 )
            {
              v48 = v41[1];
              v207 = *(_QWORD *)(a2 + 40);
              v203 = (unsigned __int8 *)*v41;
              v49 = sub_CB72A0();
              v50 = (_WORD *)v49[4];
              v51 = (__int64)v49;
              if ( v49[3] - (_QWORD)v50 <= 1u )
              {
                v151 = sub_CB6200((__int64)v49, (unsigned __int8 *)"  ", 2u);
                v52 = *(_QWORD **)(v151 + 32);
                v51 = v151;
              }
              else
              {
                *v50 = 8224;
                v52 = (_QWORD *)(v49[4] + 2LL);
                v49[4] = v52;
              }
              if ( *(_QWORD *)(v51 + 24) - (_QWORD)v52 <= 7u )
              {
                v150 = sub_CB6200(v51, "Just Ref", 8u);
                v53 = *(_QWORD **)(v150 + 32);
                v51 = v150;
              }
              else
              {
                *v52 = 0x666552207473754ALL;
                v53 = (_QWORD *)(*(_QWORD *)(v51 + 32) + 8LL);
                *(_QWORD *)(v51 + 32) = v53;
              }
              if ( *(_QWORD *)(v51 + 24) - (_QWORD)v53 <= 7u )
              {
                sub_CB6200(v51, ":  Ptr: ", 8u);
              }
              else
              {
                *v53 = 0x203A72745020203ALL;
                *(_QWORD *)(v51 + 32) += 8LL;
              }
              v54 = sub_CB72A0();
              sub_A587F0(v48, (__int64)v54, 0, 1);
              v55 = sub_CB72A0();
              v56 = (_WORD *)v55[4];
              if ( v55[3] - (_QWORD)v56 <= 1u )
              {
                sub_CB6200((__int64)v55, (unsigned __int8 *)"* ", 2u);
              }
              else
              {
                *v56 = 8234;
                v55[4] += 2LL;
              }
              v57 = sub_CB72A0();
              sub_A5BF40(v203, (__int64)v57, 0, v207);
              v58 = sub_CB72A0();
              v59 = (_DWORD *)v58[4];
              v60 = (__int64)v58;
              if ( v58[3] - (_QWORD)v59 <= 3u )
              {
                v60 = sub_CB6200((__int64)v58, "\t<->", 4u);
              }
              else
              {
                *v59 = 1043151881;
                v58[4] += 4LL;
              }
              sub_A69870((__int64)v217, (_BYTE *)v60, 0);
              v61 = *(_BYTE **)(v60 + 32);
              if ( (unsigned __int64)v61 >= *(_QWORD *)(v60 + 24) )
              {
                sub_CB5D20(v60, 10);
              }
              else
              {
                *(_QWORD *)(v60 + 32) = v61 + 1;
                *v61 = 10;
              }
            }
            ++a1[7];
          }
          else
          {
            if ( (_BYTE)qword_502DC28 || (_BYTE)qword_502E088 )
            {
              v108 = v41[1];
              v209 = *(_QWORD *)(a2 + 40);
              v205 = (unsigned __int8 *)*v41;
              v109 = sub_CB72A0();
              v110 = (_WORD *)v109[4];
              v111 = (__int64)v109;
              if ( v109[3] - (_QWORD)v110 <= 1u )
              {
                v147 = sub_CB6200((__int64)v109, (unsigned __int8 *)"  ", 2u);
                v112 = *(_QWORD **)(v147 + 32);
                v111 = v147;
              }
              else
              {
                *v110 = 8224;
                v112 = (_QWORD *)(v109[4] + 2LL);
                v109[4] = v112;
              }
              if ( *(_QWORD *)(v111 + 24) - (_QWORD)v112 <= 7u )
              {
                v146 = sub_CB6200(v111, "NoModRef", 8u);
                v113 = *(_QWORD **)(v146 + 32);
                v111 = v146;
              }
              else
              {
                *v112 = 0x666552646F4D6F4ELL;
                v113 = (_QWORD *)(*(_QWORD *)(v111 + 32) + 8LL);
                *(_QWORD *)(v111 + 32) = v113;
              }
              if ( *(_QWORD *)(v111 + 24) - (_QWORD)v113 <= 7u )
              {
                sub_CB6200(v111, ":  Ptr: ", 8u);
              }
              else
              {
                *v113 = 0x203A72745020203ALL;
                *(_QWORD *)(v111 + 32) += 8LL;
              }
              v114 = sub_CB72A0();
              sub_A587F0(v108, (__int64)v114, 0, 1);
              v115 = sub_CB72A0();
              v116 = (_WORD *)v115[4];
              if ( v115[3] - (_QWORD)v116 <= 1u )
              {
                sub_CB6200((__int64)v115, (unsigned __int8 *)"* ", 2u);
              }
              else
              {
                *v116 = 8234;
                v115[4] += 2LL;
              }
              v117 = sub_CB72A0();
              sub_A5BF40(v205, (__int64)v117, 0, v209);
              v118 = sub_CB72A0();
              v119 = (_DWORD *)v118[4];
              v120 = (__int64)v118;
              if ( v118[3] - (_QWORD)v119 <= 3u )
              {
                v120 = sub_CB6200((__int64)v118, "\t<->", 4u);
              }
              else
              {
                *v119 = 1043151881;
                v118[4] += 4LL;
              }
              sub_A69870((__int64)v217, (_BYTE *)v120, 0);
              v121 = *(_BYTE **)(v120 + 32);
              if ( (unsigned __int64)v121 >= *(_QWORD *)(v120 + 24) )
              {
                sub_CB5D20(v120, 10);
              }
              else
              {
                *(_QWORD *)(v120 + 32) = v121 + 1;
                *v121 = 10;
              }
            }
            ++a1[5];
          }
          v41 += 2;
        }
        while ( v214 != v41 );
      }
      ++v211;
    }
    while ( v202 != v211 );
    v62 = v264;
    v63 = (unsigned int)v265;
    v218 = &v264[(unsigned int)v265];
    if ( v218 != v264 )
    {
      v233 = v264;
      v64 = a1;
      while ( 1 )
      {
        v235 = &v62[v63];
        v65 = *v233;
        if ( v235 != v62 )
          break;
LABEL_128:
        if ( v218 == ++v233 )
          goto LABEL_247;
        v62 = v264;
        v63 = (unsigned int)v265;
      }
      while ( 2 )
      {
        while ( 2 )
        {
          v74 = *v62;
          if ( v65 == *v62 )
            goto LABEL_112;
          v75 = sub_CF5B00(a3, v65, *v62);
          if ( v75 == 2 )
          {
            if ( (_BYTE)qword_502E088 || byte_502DA68 )
            {
              v136 = sub_CB72A0();
              v137 = (_WORD *)v136[4];
              v138 = (__int64)v136;
              if ( v136[3] - (_QWORD)v137 <= 1u )
              {
                v155 = sub_CB6200((__int64)v136, (unsigned __int8 *)"  ", 2u);
                v139 = *(_QWORD **)(v155 + 32);
                v138 = v155;
              }
              else
              {
                *v137 = 8224;
                v139 = (_QWORD *)(v136[4] + 2LL);
                v136[4] = v139;
              }
              if ( *(_QWORD *)(v138 + 24) - (_QWORD)v139 <= 7u )
              {
                v154 = sub_CB6200(v138, "Just Mod", 8u);
                v140 = *(_WORD **)(v154 + 32);
                v138 = v154;
              }
              else
              {
                *v139 = 0x646F4D207473754ALL;
                v140 = (_WORD *)(*(_QWORD *)(v138 + 32) + 8LL);
                *(_QWORD *)(v138 + 32) = v140;
              }
              if ( *(_QWORD *)(v138 + 24) - (_QWORD)v140 <= 1u )
              {
                v138 = sub_CB6200(v138, (unsigned __int8 *)": ", 2u);
              }
              else
              {
                *v140 = 8250;
                *(_QWORD *)(v138 + 32) += 2LL;
              }
              v228 = v138;
              sub_A69870((__int64)v65, (_BYTE *)v138, 0);
              v141 = (_BYTE *)v228;
              v142 = *(_QWORD *)(v228 + 32);
              if ( (unsigned __int64)(*(_QWORD *)(v228 + 24) - v142) <= 4 )
              {
                v141 = (_BYTE *)sub_CB6200(v228, " <-> ", 5u);
              }
              else
              {
                *(_DWORD *)v142 = 1043151904;
                *(_BYTE *)(v142 + 4) = 32;
                *(_QWORD *)(v228 + 32) += 5LL;
              }
              v229 = (__int64)v141;
              sub_A69870((__int64)v74, v141, 0);
              v143 = *(_BYTE **)(v229 + 32);
              if ( (unsigned __int64)v143 >= *(_QWORD *)(v229 + 24) )
              {
                sub_CB5D20(v229, 10);
              }
              else
              {
                *(_QWORD *)(v229 + 32) = v143 + 1;
                *v143 = 10;
              }
            }
            ++v64[6];
LABEL_112:
            if ( v235 == ++v62 )
              goto LABEL_128;
            continue;
          }
          break;
        }
        if ( v75 > 2u )
        {
          if ( v75 == 3 )
          {
            if ( (_BYTE)qword_502E088 || (_BYTE)qword_502D988 )
            {
              v100 = sub_CB72A0();
              v101 = (_WORD *)v100[4];
              v102 = (__int64)v100;
              if ( v100[3] - (_QWORD)v101 <= 1u )
              {
                v153 = sub_CB6200((__int64)v100, (unsigned __int8 *)"  ", 2u);
                v103 = *(void **)(v153 + 32);
                v102 = v153;
              }
              else
              {
                *v101 = 8224;
                v103 = (void *)(v100[4] + 2LL);
                v100[4] = v103;
              }
              if ( *(_QWORD *)(v102 + 24) - (_QWORD)v103 <= 0xAu )
              {
                v152 = sub_CB6200(v102, "Both ModRef", 0xBu);
                v104 = *(_WORD **)(v152 + 32);
                v102 = v152;
              }
              else
              {
                qmemcpy(v103, "Both ModRef", 11);
                v104 = (_WORD *)(*(_QWORD *)(v102 + 32) + 11LL);
                *(_QWORD *)(v102 + 32) = v104;
              }
              if ( *(_QWORD *)(v102 + 24) - (_QWORD)v104 <= 1u )
              {
                v102 = sub_CB6200(v102, (unsigned __int8 *)": ", 2u);
              }
              else
              {
                *v104 = 8250;
                *(_QWORD *)(v102 + 32) += 2LL;
              }
              v226 = v102;
              sub_A69870((__int64)v65, (_BYTE *)v102, 0);
              v105 = (_BYTE *)v226;
              v106 = *(_QWORD *)(v226 + 32);
              if ( (unsigned __int64)(*(_QWORD *)(v226 + 24) - v106) <= 4 )
              {
                v105 = (_BYTE *)sub_CB6200(v226, " <-> ", 5u);
              }
              else
              {
                *(_DWORD *)v106 = 1043151904;
                *(_BYTE *)(v106 + 4) = 32;
                *(_QWORD *)(v226 + 32) += 5LL;
              }
              v227 = (__int64)v105;
              sub_A69870((__int64)v74, v105, 0);
              v107 = *(_BYTE **)(v227 + 32);
              if ( (unsigned __int64)v107 >= *(_QWORD *)(v227 + 24) )
              {
                sub_CB5D20(v227, 10);
              }
              else
              {
                *(_QWORD *)(v227 + 32) = v107 + 1;
                *v107 = 10;
              }
            }
            ++v64[8];
          }
          goto LABEL_112;
        }
        if ( v75 )
        {
          if ( (_BYTE)qword_502E088 || byte_502DB48 )
          {
            v66 = sub_CB72A0();
            v67 = (_WORD *)v66[4];
            v68 = (__int64)v66;
            if ( v66[3] - (_QWORD)v67 <= 1u )
            {
              v157 = sub_CB6200((__int64)v66, (unsigned __int8 *)"  ", 2u);
              v69 = *(_QWORD **)(v157 + 32);
              v68 = v157;
            }
            else
            {
              *v67 = 8224;
              v69 = (_QWORD *)(v66[4] + 2LL);
              v66[4] = v69;
            }
            if ( *(_QWORD *)(v68 + 24) - (_QWORD)v69 <= 7u )
            {
              v156 = sub_CB6200(v68, "Just Ref", 8u);
              v70 = *(_WORD **)(v156 + 32);
              v68 = v156;
            }
            else
            {
              *v69 = 0x666552207473754ALL;
              v70 = (_WORD *)(*(_QWORD *)(v68 + 32) + 8LL);
              *(_QWORD *)(v68 + 32) = v70;
            }
            if ( *(_QWORD *)(v68 + 24) - (_QWORD)v70 <= 1u )
            {
              v68 = sub_CB6200(v68, (unsigned __int8 *)": ", 2u);
            }
            else
            {
              *v70 = 8250;
              *(_QWORD *)(v68 + 32) += 2LL;
            }
            v222 = v68;
            sub_A69870((__int64)v65, (_BYTE *)v68, 0);
            v71 = (_BYTE *)v222;
            v72 = *(_QWORD *)(v222 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(v222 + 24) - v72) <= 4 )
            {
              v71 = (_BYTE *)sub_CB6200(v222, " <-> ", 5u);
            }
            else
            {
              *(_DWORD *)v72 = 1043151904;
              *(_BYTE *)(v72 + 4) = 32;
              *(_QWORD *)(v222 + 32) += 5LL;
            }
            v223 = (__int64)v71;
            sub_A69870((__int64)v74, v71, 0);
            v73 = *(_BYTE **)(v223 + 32);
            if ( (unsigned __int64)v73 >= *(_QWORD *)(v223 + 24) )
            {
              sub_CB5D20(v223, 10);
            }
            else
            {
              *(_QWORD *)(v223 + 32) = v73 + 1;
              *v73 = 10;
            }
          }
          ++v64[7];
          goto LABEL_112;
        }
        if ( (_BYTE)qword_502E088 || (_BYTE)qword_502DC28 )
        {
          v76 = sub_CB72A0();
          v77 = (_WORD *)v76[4];
          v78 = (__int64)v76;
          if ( v76[3] - (_QWORD)v77 <= 1u )
          {
            v78 = sub_CB6200((__int64)v76, (unsigned __int8 *)"  ", 2u);
            v79 = *(_QWORD **)(v78 + 32);
            if ( *(_QWORD *)(v78 + 24) - (_QWORD)v79 <= 7u )
              goto LABEL_245;
LABEL_121:
            *v79 = 0x666552646F4D6F4ELL;
            v81 = (_WORD *)(*(_QWORD *)(v78 + 32) + 8LL);
            v82 = *(_QWORD *)(v78 + 24);
            *(_QWORD *)(v78 + 32) = v81;
            if ( (unsigned __int64)(v82 - (_QWORD)v81) > 1 )
              goto LABEL_122;
LABEL_246:
            v78 = sub_CB6200(v78, (unsigned __int8 *)": ", 2u);
          }
          else
          {
            *v77 = 8224;
            v79 = (_QWORD *)(v76[4] + 2LL);
            v80 = v76[3];
            *(_QWORD *)(v78 + 32) = v79;
            if ( (unsigned __int64)(v80 - (_QWORD)v79) > 7 )
              goto LABEL_121;
LABEL_245:
            v78 = sub_CB6200(v78, "NoModRef", 8u);
            v81 = *(_WORD **)(v78 + 32);
            if ( *(_QWORD *)(v78 + 24) - (_QWORD)v81 <= 1u )
              goto LABEL_246;
LABEL_122:
            *v81 = 8250;
            *(_QWORD *)(v78 + 32) += 2LL;
          }
          v224 = v78;
          sub_A69870((__int64)v65, (_BYTE *)v78, 0);
          v83 = (_BYTE *)v224;
          v84 = *(_QWORD *)(v224 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(v224 + 24) - v84) <= 4 )
          {
            v83 = (_BYTE *)sub_CB6200(v224, " <-> ", 5u);
          }
          else
          {
            *(_DWORD *)v84 = 1043151904;
            *(_BYTE *)(v84 + 4) = 32;
            *(_QWORD *)(v224 + 32) += 5LL;
          }
          v225 = (__int64)v83;
          sub_A69870((__int64)v74, v83, 0);
          v85 = *(_BYTE **)(v225 + 32);
          if ( (unsigned __int64)v85 >= *(_QWORD *)(v225 + 24) )
          {
            sub_CB5D20(v225, 10);
          }
          else
          {
            *(_QWORD *)(v225 + 32) = v85 + 1;
            *v85 = 10;
          }
        }
        ++v64[5];
        if ( v235 == ++v62 )
          goto LABEL_128;
        continue;
      }
    }
  }
LABEL_247:
  if ( v252 != &v254 )
    _libc_free((unsigned __int64)v252);
  sub_C7D6A0(v249, 8LL * (unsigned int)v251, 8);
  if ( v246 != &v248 )
    _libc_free((unsigned __int64)v246);
  sub_C7D6A0(v243, 8LL * (unsigned int)v245, 8);
  if ( v264 != (unsigned __int8 **)v266 )
    _libc_free((unsigned __int64)v264);
  sub_C7D6A0(v261, 8LL * (unsigned int)v263, 8);
  if ( v240 != &v242 )
    _libc_free((unsigned __int64)v240);
  return sub_C7D6A0(v237, 16LL * (unsigned int)v239, 8);
}
