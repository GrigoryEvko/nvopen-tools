// Function: sub_297F050
// Address: 0x297f050
//
__int64 __fastcall sub_297F050(__int64 a1, __int32 a2, __int64 a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 *v10; // rax
  unsigned __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 *v14; // rdx
  __int64 v15; // rax
  __m128i v16; // xmm2
  __m128i v17; // xmm3
  __m128i v18; // xmm4
  unsigned int v19; // esi
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // r9
  int v23; // r11d
  __int64 *v24; // rbx
  __int64 v25; // r8
  unsigned int v26; // eax
  __int64 v27; // r12
  __int64 v28; // rcx
  __int64 v29; // rax
  _QWORD *v30; // rbx
  __int64 v31; // r14
  _QWORD *v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rdx
  __int64 v35; // rbx
  __int64 v36; // rcx
  __m128i *v37; // rax
  __int64 v38; // r15
  unsigned int v39; // esi
  __int64 v40; // r11
  __int64 v41; // rcx
  __int64 v42; // r9
  int v43; // r10d
  __int64 v44; // r8
  __int64 *v45; // r14
  _QWORD *v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rax
  _QWORD *v49; // r15
  __int64 v50; // r15
  unsigned int v51; // esi
  __int64 v52; // r11
  __int64 v53; // rcx
  __int64 v54; // r9
  int v55; // r10d
  __int64 v56; // r8
  __int64 *v57; // r14
  _QWORD *v58; // rax
  __int64 v59; // rdi
  __int64 v60; // rax
  _QWORD *v61; // r15
  __int64 v62; // r14
  unsigned int v63; // esi
  __int64 v64; // r9
  __int64 v65; // rdx
  __int64 v66; // r8
  int v67; // r10d
  unsigned int v68; // edi
  __int64 v69; // r13
  _QWORD *v70; // rax
  __int64 v71; // rcx
  __int64 result; // rax
  _QWORD *v73; // rbx
  _QWORD *v74; // r12
  __int64 v75; // rbx
  __int64 v76; // r14
  __int64 v77; // rax
  __int64 v78; // rsi
  __int64 v79; // rdi
  unsigned int v80; // edx
  __int64 *v81; // r13
  __int64 v82; // r9
  __int64 v83; // rdx
  __int64 v84; // rax
  __int64 v85; // rsi
  bool v86; // al
  __int16 v87; // ax
  unsigned int v88; // esi
  __int64 v89; // rdi
  __int64 v90; // rcx
  unsigned int v91; // r15d
  __int64 v92; // r11
  __int64 v93; // rax
  __int64 v94; // r10
  unsigned int *v95; // r11
  __int64 v96; // rax
  __int64 v97; // r15
  __int64 v98; // rdx
  __int64 v99; // rcx
  __int64 v100; // rsi
  __int64 v101; // r9
  int v102; // r12d
  unsigned __int64 v103; // rbx
  __int64 k; // r8
  __int64 v105; // rax
  unsigned int v106; // r8d
  int v107; // esi
  __int64 v108; // rax
  int v109; // edx
  unsigned int v110; // esi
  __int64 v111; // rdx
  int v112; // eax
  int v113; // r8d
  __int64 v114; // r9
  unsigned int v115; // ecx
  int v116; // edi
  __int64 v117; // rsi
  int v118; // r10d
  _QWORD *v119; // r11
  int v120; // esi
  __int64 v121; // rax
  int v122; // edx
  unsigned int v123; // esi
  __int64 v124; // rdx
  __m128i *v125; // rdx
  __int64 v126; // rdx
  int v127; // eax
  int v128; // edi
  __int64 v129; // r9
  __int64 v130; // rdx
  int v131; // r8d
  __int64 v132; // rsi
  int v133; // r14d
  _QWORD *v134; // r11
  int v135; // esi
  __int64 v136; // rax
  int v137; // edx
  unsigned int v138; // esi
  __int64 v139; // rdx
  __int64 v140; // rdx
  __int64 v141; // rdx
  int v142; // eax
  int v143; // edi
  __int64 v144; // r9
  __int64 v145; // rdx
  int v146; // r8d
  __int64 v147; // rsi
  int v148; // r14d
  _QWORD *v149; // r11
  char v150; // al
  int v151; // ecx
  int v152; // edi
  int v153; // eax
  int v154; // ecx
  _QWORD *v155; // rax
  int v156; // edi
  __int64 v157; // rdx
  __int64 v158; // rdx
  __int64 v159; // r9
  int v160; // r12d
  unsigned __int64 v161; // rbx
  __int64 m; // r8
  __int64 v163; // rax
  unsigned int v164; // r8d
  int v165; // ecx
  int v166; // eax
  int v167; // esi
  __int64 v168; // r8
  unsigned __int32 v169; // eax
  __int64 v170; // rdi
  int v171; // r11d
  __int64 *v172; // r9
  int v173; // eax
  int v174; // edi
  __int64 v175; // r9
  int v176; // r14d
  __int64 v177; // rdx
  __int64 v178; // rsi
  int v179; // eax
  int v180; // edi
  __int64 v181; // r9
  int v182; // r14d
  __int64 v183; // rdx
  __int64 v184; // rsi
  int v185; // eax
  int v186; // esi
  __int64 v187; // r8
  _QWORD *v188; // r10
  unsigned int v189; // r15d
  int v190; // r9d
  __int64 v191; // rcx
  int v192; // eax
  int v193; // esi
  int v194; // r11d
  __int64 v195; // r8
  unsigned __int32 v196; // eax
  __int64 v197; // rdi
  __int64 v198; // rcx
  __int64 v199; // rdx
  __int64 v200; // rdx
  int v201; // eax
  int v202; // eax
  __int64 v203; // rsi
  char *v204; // rdx
  char *v205; // rdi
  __int64 *v206; // r13
  __int64 v207; // rbx
  __int64 *v208; // r12
  __int64 *v209; // rdx
  int v210; // esi
  int v211; // esi
  int v212; // edi
  unsigned int i; // eax
  __int64 *v214; // rcx
  __int64 v215; // r11
  int v216; // esi
  int v217; // esi
  __int64 v218; // r10
  int v219; // ecx
  unsigned int j; // eax
  __int64 v221; // r11
  const void *v222; // rsi
  __int64 v223; // rdi
  unsigned int v224; // eax
  unsigned int v225; // eax
  __int64 v226; // [rsp+0h] [rbp-160h]
  __int64 v227; // [rsp+8h] [rbp-158h]
  __int64 v228; // [rsp+10h] [rbp-150h]
  __int32 v229; // [rsp+10h] [rbp-150h]
  __int64 v230; // [rsp+10h] [rbp-150h]
  __int64 v231; // [rsp+10h] [rbp-150h]
  _BYTE *v232; // [rsp+18h] [rbp-148h]
  unsigned int *v233; // [rsp+18h] [rbp-148h]
  int v234; // [rsp+18h] [rbp-148h]
  __int64 v235; // [rsp+18h] [rbp-148h]
  __int64 v236; // [rsp+18h] [rbp-148h]
  __int64 v237; // [rsp+18h] [rbp-148h]
  __int64 v238; // [rsp+18h] [rbp-148h]
  __int64 v239; // [rsp+18h] [rbp-148h]
  __int64 *v241; // [rsp+20h] [rbp-140h]
  unsigned int v242; // [rsp+20h] [rbp-140h]
  __int64 v243; // [rsp+20h] [rbp-140h]
  __int64 v244; // [rsp+20h] [rbp-140h]
  __int64 v245; // [rsp+20h] [rbp-140h]
  __int64 v246; // [rsp+20h] [rbp-140h]
  __int64 v247; // [rsp+20h] [rbp-140h]
  __int64 v248; // [rsp+20h] [rbp-140h]
  __int64 v249; // [rsp+20h] [rbp-140h]
  __int64 v250; // [rsp+20h] [rbp-140h]
  __int64 v251; // [rsp+20h] [rbp-140h]
  __int64 v252; // [rsp+20h] [rbp-140h]
  unsigned int *v253; // [rsp+20h] [rbp-140h]
  _QWORD *v255; // [rsp+28h] [rbp-138h]
  __int64 v256; // [rsp+28h] [rbp-138h]
  __int64 v257; // [rsp+30h] [rbp-130h] BYREF
  __int64 v258; // [rsp+38h] [rbp-128h] BYREF
  __int64 v259; // [rsp+40h] [rbp-120h]
  __int64 v260; // [rsp+48h] [rbp-118h]
  __m128i *v261; // [rsp+50h] [rbp-110h]
  __int64 v262; // [rsp+60h] [rbp-100h]
  __int64 v263; // [rsp+68h] [rbp-F8h]
  __m128i *v264; // [rsp+70h] [rbp-F0h]
  __int64 v265; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v266; // [rsp+88h] [rbp-D8h]
  __int64 v267; // [rsp+90h] [rbp-D0h]
  __int64 v268; // [rsp+A0h] [rbp-C0h] BYREF
  __m128i *v269; // [rsp+A8h] [rbp-B8h]
  __int64 v270; // [rsp+B0h] [rbp-B0h]
  _BYTE *v271; // [rsp+C0h] [rbp-A0h] BYREF
  __m128i *v272; // [rsp+C8h] [rbp-98h]
  __int64 v273; // [rsp+D0h] [rbp-90h]
  __m128i v274; // [rsp+E0h] [rbp-80h] BYREF
  __m128i v275; // [rsp+F0h] [rbp-70h] BYREF
  __m128i v276; // [rsp+100h] [rbp-60h] BYREF
  __m128i v277; // [rsp+110h] [rbp-50h] BYREF
  __int64 v278; // [rsp+120h] [rbp-40h]

  v7 = a1;
  v10 = sub_DD8400(*(_QWORD *)(a1 + 16), (__int64)a5);
  v12 = (__int64)a5;
  v13 = (__int64)v10;
  if ( *a5 <= 0x1Cu )
    goto LABEL_3;
  v271 = a5;
  if ( !(_BYTE)qword_50073C8 )
    goto LABEL_3;
  v232 = a5;
  v241 = v10;
  v86 = sub_D96A50((__int64)v10);
  v13 = (__int64)v241;
  v12 = (__int64)v232;
  if ( v86 )
    goto LABEL_3;
  v87 = *((_WORD *)v241 + 12);
  if ( !v87 || v87 == 15 )
    goto LABEL_3;
  v88 = *(_DWORD *)(a1 + 200);
  v89 = a1 + 176;
  if ( !v88 )
  {
    ++*(_QWORD *)(v7 + 176);
    goto LABEL_275;
  }
  v90 = *(_QWORD *)(v7 + 184);
  v242 = ((unsigned int)v241 >> 9) ^ ((unsigned int)v241 >> 4);
  v91 = (v88 - 1) & v242;
  v92 = v90 + 72LL * v91;
  v93 = *(_QWORD *)v92;
  if ( v13 != *(_QWORD *)v92 )
  {
    v234 = 1;
    v200 = 0;
    while ( v93 != -4096 )
    {
      if ( !v200 && v93 == -8192 )
        v200 = v92;
      v91 = (v88 - 1) & (v234 + v91);
      v92 = v90 + 72LL * v91;
      v93 = *(_QWORD *)v92;
      if ( v13 == *(_QWORD *)v92 )
        goto LABEL_53;
      ++v234;
    }
    v201 = *(_DWORD *)(v7 + 192);
    if ( !v200 )
      v200 = v92;
    ++*(_QWORD *)(v7 + 176);
    v202 = v201 + 1;
    if ( 4 * v202 < 3 * v88 )
    {
      if ( v88 - *(_DWORD *)(v7 + 196) - v202 > v88 >> 3 )
      {
LABEL_241:
        *(_DWORD *)(v7 + 192) = v202;
        if ( *(_QWORD *)v200 != -4096 )
          --*(_DWORD *)(v7 + 196);
        *(_QWORD *)v200 = v13;
        v94 = v200 + 8;
        *(_QWORD *)(v200 + 40) = v200 + 56;
        *(_QWORD *)(v200 + 48) = 0x200000000LL;
        *(_OWORD *)(v200 + 8) = 0;
        *(_OWORD *)(v200 + 24) = 0;
        *(_OWORD *)(v200 + 56) = 0;
        goto LABEL_244;
      }
      v230 = v12;
      v236 = v13;
      sub_297EA40(v89, v88);
      v210 = *(_DWORD *)(v7 + 200);
      if ( v210 )
      {
        v211 = v210 - 1;
        v212 = 1;
        v200 = 0;
        v13 = v236;
        v12 = v230;
        for ( i = v211 & v242; ; i = v211 & v225 )
        {
          v214 = (__int64 *)(*(_QWORD *)(v7 + 184) + 72LL * i);
          v215 = *v214;
          if ( v236 == *v214 )
          {
            v200 = *(_QWORD *)(v7 + 184) + 72LL * i;
            v202 = *(_DWORD *)(v7 + 192) + 1;
            goto LABEL_241;
          }
          if ( v215 == -4096 )
            break;
          if ( v200 || v215 != -8192 )
            v214 = (__int64 *)v200;
          v225 = v212 + i;
          v200 = (__int64)v214;
          ++v212;
        }
        v202 = *(_DWORD *)(v7 + 192) + 1;
        if ( !v200 )
          v200 = (__int64)v214;
        goto LABEL_241;
      }
LABEL_325:
      ++*(_DWORD *)(v7 + 192);
      BUG();
    }
LABEL_275:
    v237 = v12;
    v251 = v13;
    sub_297EA40(v89, 2 * v88);
    v216 = *(_DWORD *)(v7 + 200);
    if ( v216 )
    {
      v13 = v251;
      v217 = v216 - 1;
      v218 = 0;
      v12 = v237;
      v219 = 1;
      for ( j = v217 & (((unsigned int)v251 >> 9) ^ ((unsigned int)v251 >> 4)); ; j = v217 & v224 )
      {
        v200 = *(_QWORD *)(v7 + 184) + 72LL * j;
        v221 = *(_QWORD *)v200;
        if ( v251 == *(_QWORD *)v200 )
        {
          v202 = *(_DWORD *)(v7 + 192) + 1;
          goto LABEL_241;
        }
        if ( v221 == -4096 )
          break;
        if ( v221 != -8192 || v218 )
          v200 = v218;
        v224 = v219 + j;
        v218 = v200;
        ++v219;
      }
      v202 = *(_DWORD *)(v7 + 192) + 1;
      if ( v218 )
        v200 = v218;
      goto LABEL_241;
    }
    goto LABEL_325;
  }
LABEL_53:
  v94 = v92 + 8;
  if ( *(_DWORD *)(v92 + 24) )
  {
    v228 = v12;
    v233 = (unsigned int *)v92;
    v243 = v13;
    sub_2400480((__int64)&v274, v92 + 8, (__int64 *)&v271);
    v13 = v243;
    v95 = v233;
    v12 = v228;
    if ( v276.m128i_i8[0] )
    {
      v96 = v233[12];
      v11 = v233[13];
      v97 = (__int64)v271;
      if ( v96 + 1 > v11 )
      {
        v222 = v233 + 14;
        v223 = (__int64)(v233 + 10);
        v239 = v243;
        v253 = v95;
        sub_C8D5F0(v223, v222, v96 + 1, 8u, v228, v13);
        v95 = v253;
        v12 = v228;
        v13 = v239;
        v96 = v253[12];
      }
      *(_QWORD *)(*((_QWORD *)v95 + 5) + 8 * v96) = v97;
      ++v95[12];
    }
    goto LABEL_3;
  }
LABEL_244:
  v203 = *(unsigned int *)(v94 + 40);
  v204 = *(char **)(v94 + 32);
  v205 = &v204[8 * v203];
  if ( (8 * v203) >> 5 )
  {
    v11 = (unsigned __int64)&v204[32 * ((8 * v203) >> 5)];
    while ( *(_BYTE **)v204 != v271 )
    {
      if ( *((_BYTE **)v204 + 1) == v271 )
      {
        v204 += 8;
        break;
      }
      if ( *((_BYTE **)v204 + 2) == v271 )
      {
        v204 += 16;
        break;
      }
      if ( *((_BYTE **)v204 + 3) == v271 )
      {
        v204 += 24;
        break;
      }
      v204 += 32;
      if ( (char *)v11 == v204 )
        goto LABEL_258;
    }
LABEL_251:
    if ( v205 != v204 )
      goto LABEL_3;
    goto LABEL_252;
  }
LABEL_258:
  v11 = v205 - v204;
  if ( v205 - v204 != 16 )
  {
    if ( v11 != 24 )
    {
      if ( v11 != 8 )
        goto LABEL_252;
      goto LABEL_261;
    }
    if ( *(_BYTE **)v204 == v271 )
      goto LABEL_251;
    v204 += 8;
  }
  if ( *(_BYTE **)v204 == v271 )
    goto LABEL_251;
  v204 += 8;
LABEL_261:
  if ( *(_BYTE **)v204 == v271 )
    goto LABEL_251;
LABEL_252:
  if ( v203 + 1 > (unsigned __int64)*(unsigned int *)(v94 + 44) )
  {
    v231 = v12;
    v238 = v13;
    v252 = v94;
    sub_C8D5F0(v94 + 32, (const void *)(v94 + 48), v203 + 1, 8u, v12, v13);
    v94 = v252;
    v12 = v231;
    v13 = v238;
    v205 = (char *)(*(_QWORD *)(v252 + 32) + 8LL * *(unsigned int *)(v252 + 40));
  }
  *(_QWORD *)v205 = v271;
  v11 = (unsigned int)(*(_DWORD *)(v94 + 40) + 1);
  *(_DWORD *)(v94 + 40) = v11;
  if ( (unsigned int)v11 > 2 )
  {
    v250 = v13;
    v226 = v12;
    v235 = v7;
    v206 = *(__int64 **)(v94 + 32);
    v229 = a2;
    v207 = v94;
    v227 = a3;
    v208 = &v206[v11];
    do
    {
      v209 = v206++;
      sub_2400480((__int64)&v274, v207, v209);
    }
    while ( v208 != v206 );
    v13 = v250;
    v7 = v235;
    a2 = v229;
    a3 = v227;
    v12 = v226;
  }
LABEL_3:
  v274.m128i_i32[0] = a2;
  v274.m128i_i64[1] = a3;
  v14 = *(__int64 **)(v7 + 24);
  v275.m128i_i64[0] = a4;
  v275.m128i_i64[1] = v12;
  v276 = (__m128i)(unsigned __int64)a6;
  v277.m128i_i32[0] = 0;
  v277.m128i_i64[1] = v13;
  v278 = 0;
  if ( a2 == 1 )
  {
    v150 = sub_297BEB0(a3, a4, v14);
    goto LABEL_105;
  }
  if ( a2 == 3 )
  {
    v150 = sub_297BB80(a6, (__int64 **)v14, (__int64)v14, v11, v12, v13);
LABEL_105:
    if ( v150 )
      goto LABEL_7;
  }
  if ( (_BYTE)qword_5007588 || !sub_297C860(v274.m128i_i32) )
  {
    v271 = (_BYTE *)v7;
    v272 = &v274;
    v74 = sub_297C3E0(v7 + 56, &v274);
    if ( v74 )
    {
      v75 = *(_QWORD *)(v7 + 160);
      if ( *(_QWORD *)(v7 + 152) != v75 )
      {
        v256 = v7;
        v76 = *(_QWORD *)(v7 + 152);
        while ( 1 )
        {
          v77 = *((unsigned int *)v74 + 6);
          v78 = *(_QWORD *)(v75 - 8);
          v79 = v74[1];
          if ( (_DWORD)v77 )
          {
            v80 = (v77 - 1) & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
            v81 = (__int64 *)(v79 + 88LL * v80);
            v82 = *v81;
            if ( v78 == *v81 )
            {
LABEL_33:
              if ( v81 != (__int64 *)(v79 + 88 * v77) )
              {
                if ( *((_DWORD *)v81 + 4) )
                {
                  if ( (unsigned __int8)sub_B19720(*((_QWORD *)v271 + 1), v78, *(_QWORD *)(v272[2].m128i_i64[0] + 40)) )
                  {
                    v83 = v81[1];
                    v84 = v83 + 8LL * *((unsigned int *)v81 + 4);
                    if ( v83 != v84 )
                    {
                      while ( 1 )
                      {
                        v85 = *(_QWORD *)(v84 - 8);
                        if ( *(_QWORD *)(v85 + 32) != v276.m128i_i64[0] && *(_DWORD *)v85 == v274.m128i_i32[0] )
                          break;
                        v84 -= 8;
                        if ( v83 == v84 )
                          goto LABEL_30;
                      }
                      v276.m128i_i64[1] = *(_QWORD *)(v84 - 8);
                      v7 = v256;
                      v277.m128i_i32[0] = 1;
                      v278 = sub_297C710((__int64)&v274, v85);
                      goto LABEL_42;
                    }
                  }
                }
              }
            }
            else
            {
              v165 = 1;
              while ( v82 != -4096 )
              {
                v80 = (v77 - 1) & (v165 + v80);
                v81 = (__int64 *)(v79 + 88LL * v80);
                v82 = *v81;
                if ( v78 == *v81 )
                  goto LABEL_33;
                ++v165;
              }
            }
          }
LABEL_30:
          v75 -= 8;
          if ( v76 == v75 )
          {
            v7 = v256;
            break;
          }
        }
      }
    }
    if ( v274.m128i_i32[0] == 2 )
      goto LABEL_42;
    v98 = *(unsigned int *)(v7 + 112);
    v99 = v275.m128i_i64[0];
    v100 = *(_QWORD *)(v276.m128i_i64[0] + 8);
    v101 = *(_QWORD *)(v7 + 96);
    if ( (_DWORD)v98 )
    {
      v102 = 1;
      v103 = (0xBF58476D1CE4E5B9LL
            * ((969526130 * (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4)))
             | ((unsigned __int64)(((unsigned __int32)v275.m128i_i32[0] >> 9)
                                 ^ ((unsigned __int32)v275.m128i_i32[0] >> 4)) << 32))) >> 31;
      for ( k = ((_DWORD)v98 - 1)
              & ((unsigned int)((0xBF58476D1CE4E5B9LL
                               * (((unsigned __int64)(((unsigned __int32)v277.m128i_i32[2] >> 9)
                                                    ^ ((unsigned __int32)v277.m128i_i32[2] >> 4)) << 32)
                                | (unsigned int)v103
                                ^ (-279380126 * (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4))))) >> 31)
               ^ (484763065
                * ((unsigned int)v103 ^ (-279380126 * (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4))))));
            ;
            k = ((_DWORD)v98 - 1) & v106 )
      {
        v105 = v101 + 56LL * (unsigned int)k;
        if ( __PAIR128__(v277.m128i_u64[1], v275.m128i_u64[0]) == *(_OWORD *)(v105 + 8) && v100 == *(_QWORD *)v105 )
          break;
        if ( *(_QWORD *)(v105 + 16) == -4096 && *(_QWORD *)(v105 + 8) == -4096 && *(_QWORD *)v105 == -4096 )
          goto LABEL_173;
        v106 = v102 + k;
        ++v102;
      }
      v157 = v101 + 56 * v98;
      if ( v105 != v157 )
      {
        v263 = v7;
        LODWORD(v262) = 2;
        v264 = &v274;
        if ( sub_297C090(&v271, v105 + 24, v157, v275.m128i_i64[0], k, v101, 2, v7, (__int64)&v274) )
          goto LABEL_42;
        v99 = v275.m128i_i64[0];
        v100 = *(_QWORD *)(v276.m128i_i64[0] + 8);
      }
    }
LABEL_173:
    v158 = *(unsigned int *)(v7 + 144);
    v159 = *(_QWORD *)(v7 + 128);
    if ( (_DWORD)v158 )
    {
      v160 = 1;
      v161 = (0xBF58476D1CE4E5B9LL
            * ((969526130 * (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4)))
             | ((unsigned __int64)(((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4)) << 32))) >> 31;
      for ( m = ((_DWORD)v158 - 1)
              & ((unsigned int)((0xBF58476D1CE4E5B9LL
                               * (((unsigned __int64)(((unsigned __int32)v274.m128i_i32[2] >> 9)
                                                    ^ ((unsigned __int32)v274.m128i_i32[2] >> 4)) << 32)
                                | (unsigned int)v161
                                ^ (-279380126 * (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4))))) >> 31)
               ^ (484763065
                * ((unsigned int)v161 ^ (-279380126 * (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4))))));
            ;
            m = ((_DWORD)v158 - 1) & v164 )
      {
        v163 = v159 + 56LL * (unsigned int)m;
        if ( __PAIR128__(v274.m128i_u64[1], v99) == *(_OWORD *)(v163 + 8) && v100 == *(_QWORD *)v163 )
          break;
        if ( *(_QWORD *)(v163 + 16) == -4096 && *(_QWORD *)(v163 + 8) == -4096 && *(_QWORD *)v163 == -4096 )
          goto LABEL_42;
        v164 = v160 + m;
        ++v160;
      }
      v198 = 7 * v158;
      v199 = v159 + 56 * v158;
      if ( v163 != v199 )
      {
        v260 = v7;
        LODWORD(v259) = 3;
        v261 = &v274;
        sub_297C090(&v271, v163 + 24, v199, v198, m, v159, 3, v7, (__int64)&v274);
      }
    }
LABEL_42:
    if ( (_BYTE)qword_5007748 )
    {
      if ( v276.m128i_i64[1] )
      {
        if ( *(_QWORD *)(v276.m128i_i64[1] + 40) )
        {
          if ( v274.m128i_i32[0] != 2 )
          {
            sub_297CE50((__int64 *)&v271, v7, (__int64)&v274, v276.m128i_i64[1]);
            if ( v271 )
            {
              v276.m128i_i64[1] = (__int64)v271;
              v277.m128i_i32[0] = (int)v272;
              v278 = v273;
            }
          }
        }
      }
    }
  }
LABEL_7:
  v15 = sub_22077B0(0x58u);
  v16 = _mm_loadu_si128(&v275);
  v17 = _mm_loadu_si128(&v276);
  v18 = _mm_loadu_si128(&v277);
  *(__m128i *)(v15 + 16) = _mm_loadu_si128(&v274);
  *(__m128i *)(v15 + 32) = v16;
  *(__m128i *)(v15 + 48) = v17;
  *(__m128i *)(v15 + 64) = v18;
  *(_QWORD *)(v15 + 80) = v278;
  sub_2208C80((_QWORD *)v15, v7 + 32);
  v19 = *(_DWORD *)(v7 + 280);
  ++*(_QWORD *)(v7 + 48);
  v20 = v7 + 256;
  if ( !v19 )
  {
    ++*(_QWORD *)(v7 + 256);
    goto LABEL_195;
  }
  v21 = v276.m128i_i64[0];
  v22 = v19 - 1;
  v23 = 1;
  v24 = 0;
  v25 = *(_QWORD *)(v7 + 264);
  v26 = v22 & (((unsigned __int32)v276.m128i_i32[0] >> 9) ^ ((unsigned __int32)v276.m128i_i32[0] >> 4));
  v27 = v25 + 48LL * v26;
  v28 = *(_QWORD *)v27;
  if ( v276.m128i_i64[0] == *(_QWORD *)v27 )
  {
LABEL_9:
    v29 = *(unsigned int *)(v27 + 16);
    v30 = (_QWORD *)(v27 + 8);
    v31 = *(_QWORD *)(v7 + 40) + 16LL;
    if ( *(unsigned int *)(v27 + 20) < (unsigned __int64)(v29 + 1) )
    {
      sub_C8D5F0(v27 + 8, (const void *)(v27 + 24), v29 + 1, 8u, v25, v22);
      v29 = *(unsigned int *)(v27 + 16);
    }
    goto LABEL_11;
  }
  while ( v28 != -4096 )
  {
    if ( !v24 && v28 == -8192 )
      v24 = (__int64 *)v27;
    v26 = v22 & (v23 + v26);
    v27 = v25 + 48LL * v26;
    v28 = *(_QWORD *)v27;
    if ( v276.m128i_i64[0] == *(_QWORD *)v27 )
      goto LABEL_9;
    ++v23;
  }
  v153 = *(_DWORD *)(v7 + 272);
  if ( !v24 )
    v24 = (__int64 *)v27;
  ++*(_QWORD *)(v7 + 256);
  v154 = v153 + 1;
  if ( 4 * (v153 + 1) >= 3 * v19 )
  {
LABEL_195:
    sub_297DB60(v20, 2 * v19);
    v166 = *(_DWORD *)(v7 + 280);
    if ( v166 )
    {
      v21 = v276.m128i_i64[0];
      v167 = v166 - 1;
      v168 = *(_QWORD *)(v7 + 264);
      v169 = (v166 - 1) & (((unsigned __int32)v276.m128i_i32[0] >> 9) ^ ((unsigned __int32)v276.m128i_i32[0] >> 4));
      v24 = (__int64 *)(v168 + 48LL * v169);
      v154 = *(_DWORD *)(v7 + 272) + 1;
      v170 = *v24;
      if ( *v24 == v276.m128i_i64[0] )
        goto LABEL_143;
      v171 = 1;
      v172 = 0;
      while ( v170 != -4096 )
      {
        if ( !v172 && v170 == -8192 )
          v172 = v24;
        v169 = v167 & (v171 + v169);
        v24 = (__int64 *)(v168 + 48LL * v169);
        v170 = *v24;
        if ( v276.m128i_i64[0] == *v24 )
          goto LABEL_143;
        ++v171;
      }
LABEL_199:
      if ( v172 )
        v24 = v172;
      goto LABEL_143;
    }
LABEL_329:
    ++*(_DWORD *)(v7 + 272);
    BUG();
  }
  if ( v19 - *(_DWORD *)(v7 + 276) - v154 <= v19 >> 3 )
  {
    sub_297DB60(v20, v19);
    v192 = *(_DWORD *)(v7 + 280);
    if ( v192 )
    {
      v21 = v276.m128i_i64[0];
      v193 = v192 - 1;
      v194 = 1;
      v172 = 0;
      v195 = *(_QWORD *)(v7 + 264);
      v196 = (v192 - 1) & (((unsigned __int32)v276.m128i_i32[0] >> 9) ^ ((unsigned __int32)v276.m128i_i32[0] >> 4));
      v24 = (__int64 *)(v195 + 48LL * v196);
      v154 = *(_DWORD *)(v7 + 272) + 1;
      v197 = *v24;
      if ( *v24 == v276.m128i_i64[0] )
        goto LABEL_143;
      while ( v197 != -4096 )
      {
        if ( v197 == -8192 && !v172 )
          v172 = v24;
        v196 = v193 & (v194 + v196);
        v24 = (__int64 *)(v195 + 48LL * v196);
        v197 = *v24;
        if ( v276.m128i_i64[0] == *v24 )
          goto LABEL_143;
        ++v194;
      }
      goto LABEL_199;
    }
    goto LABEL_329;
  }
LABEL_143:
  *(_DWORD *)(v7 + 272) = v154;
  if ( *v24 != -4096 )
    --*(_DWORD *)(v7 + 276);
  v155 = v24 + 3;
  *v24 = v21;
  v30 = v24 + 1;
  *v30 = v155;
  v30[1] = 0x300000000LL;
  v31 = *(_QWORD *)(v7 + 40) + 16LL;
  v29 = 0;
LABEL_11:
  *(_QWORD *)(*v30 + 8 * v29) = v31;
  ++*((_DWORD *)v30 + 2);
  v32 = *(_QWORD **)(v7 + 40);
  v33 = v32[6];
  v255 = v32 + 2;
  v34 = *(_QWORD *)(v33 + 8);
  v35 = *(_QWORD *)(v33 + 40);
  v266 = v32[9];
  v36 = v32[3];
  v37 = (__m128i *)v32[4];
  v270 = v266;
  v265 = v34;
  v268 = v34;
  v271 = (_BYTE *)v34;
  v267 = v36;
  v269 = v37;
  v272 = v37;
  v273 = v36;
  if ( !(unsigned __int8)sub_297C510(v7 + 56, &v265, &v257) )
  {
    v135 = *(_DWORD *)(v7 + 72);
    v136 = v257;
    ++*(_QWORD *)(v7 + 56);
    v137 = v135 + 1;
    v138 = *(_DWORD *)(v7 + 80);
    v258 = v136;
    if ( 4 * v137 >= 3 * v138 )
    {
      v138 *= 2;
    }
    else if ( v138 - *(_DWORD *)(v7 + 76) - v137 > v138 >> 3 )
    {
      goto LABEL_93;
    }
    sub_297E2D0(v7 + 56, v138);
    sub_297C510(v7 + 56, &v265, &v258);
    v137 = *(_DWORD *)(v7 + 72) + 1;
    v136 = v258;
LABEL_93:
    *(_DWORD *)(v7 + 72) = v137;
    if ( *(_QWORD *)(v136 + 16) != -4096 || *(_QWORD *)(v136 + 8) != -4096 || *(_QWORD *)v136 != -4096 )
      --*(_DWORD *)(v7 + 76);
    v139 = v267;
    *(_QWORD *)(v136 + 24) = 0;
    v41 = v136 + 24;
    *(_QWORD *)(v136 + 32) = 0;
    *(_QWORD *)(v136 + 16) = v139;
    v140 = v266;
    *(_QWORD *)(v136 + 40) = 0;
    *(_QWORD *)(v136 + 8) = v140;
    v141 = v265;
    *(_DWORD *)(v136 + 48) = 0;
    *(_QWORD *)v136 = v141;
    goto LABEL_96;
  }
  v38 = v257;
  v39 = *(_DWORD *)(v257 + 48);
  v40 = *(_QWORD *)(v257 + 32);
  v41 = v257 + 24;
  if ( !v39 )
  {
LABEL_96:
    ++*(_QWORD *)v41;
    v39 = 0;
    goto LABEL_97;
  }
  v42 = v39 - 1;
  v43 = 1;
  v44 = (unsigned int)v42 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
  v45 = (__int64 *)(v40 + 88 * v44);
  v46 = 0;
  v47 = *v45;
  if ( v35 == *v45 )
  {
LABEL_14:
    v48 = *((unsigned int *)v45 + 4);
    v49 = v45 + 1;
    if ( *((unsigned int *)v45 + 5) < (unsigned __int64)(v48 + 1) )
    {
      sub_C8D5F0((__int64)(v45 + 1), v45 + 3, v48 + 1, 8u, v44, v42);
      v48 = *((unsigned int *)v45 + 4);
    }
    goto LABEL_16;
  }
  while ( v47 != -4096 )
  {
    if ( !v46 && v47 == -8192 )
      v46 = v45;
    v44 = (unsigned int)v42 & (v43 + (_DWORD)v44);
    v45 = (__int64 *)(v40 + 88LL * (unsigned int)v44);
    v47 = *v45;
    if ( v35 == *v45 )
      goto LABEL_14;
    ++v43;
  }
  v156 = *(_DWORD *)(v257 + 40);
  if ( !v46 )
    v46 = v45;
  ++*(_QWORD *)(v257 + 24);
  v146 = v156 + 1;
  if ( 4 * (v156 + 1) < 3 * v39 )
  {
    if ( v39 - *(_DWORD *)(v38 + 44) - v146 > v39 >> 3 )
      goto LABEL_156;
    v247 = v41;
    sub_297D910(v41, v39);
    v173 = *(_DWORD *)(v38 + 48);
    v41 = v247;
    if ( v173 )
    {
      v174 = v173 - 1;
      v175 = *(_QWORD *)(v38 + 32);
      v149 = 0;
      v176 = 1;
      LODWORD(v177) = (v173 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v146 = *(_DWORD *)(v38 + 40) + 1;
      v46 = (_QWORD *)(v175 + 88LL * (unsigned int)v177);
      v178 = *v46;
      if ( v35 == *v46 )
        goto LABEL_156;
      while ( v178 != -4096 )
      {
        if ( !v149 && v178 == -8192 )
          v149 = v46;
        v177 = v174 & (unsigned int)(v177 + v176);
        v46 = (_QWORD *)(v175 + 88 * v177);
        v178 = *v46;
        if ( v35 == *v46 )
          goto LABEL_156;
        ++v176;
      }
LABEL_101:
      if ( v149 )
        v46 = v149;
      goto LABEL_156;
    }
LABEL_326:
    ++*(_DWORD *)(v41 + 16);
    BUG();
  }
LABEL_97:
  v246 = v41;
  sub_297D910(v41, 2 * v39);
  v41 = v246;
  v142 = *(_DWORD *)(v246 + 24);
  if ( !v142 )
    goto LABEL_326;
  v143 = v142 - 1;
  v144 = *(_QWORD *)(v246 + 8);
  LODWORD(v145) = (v142 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
  v146 = *(_DWORD *)(v246 + 16) + 1;
  v46 = (_QWORD *)(v144 + 88LL * (unsigned int)v145);
  v147 = *v46;
  if ( v35 != *v46 )
  {
    v148 = 1;
    v149 = 0;
    while ( v147 != -4096 )
    {
      if ( v147 == -8192 && !v149 )
        v149 = v46;
      v145 = v143 & (unsigned int)(v145 + v148);
      v46 = (_QWORD *)(v144 + 88 * v145);
      v147 = *v46;
      if ( v35 == *v46 )
        goto LABEL_156;
      ++v148;
    }
    goto LABEL_101;
  }
LABEL_156:
  *(_DWORD *)(v41 + 16) = v146;
  if ( *v46 != -4096 )
    --*(_DWORD *)(v41 + 20);
  *v46 = v35;
  v49 = v46 + 1;
  v46[1] = v46 + 3;
  v46[2] = 0x800000000LL;
  v48 = 0;
LABEL_16:
  *(_QWORD *)(*v49 + 8 * v48) = v255;
  ++*((_DWORD *)v49 + 2);
  if ( !(unsigned __int8)sub_297D7C0(v7 + 88, &v268, &v257) )
  {
    v120 = *(_DWORD *)(v7 + 104);
    v121 = v257;
    ++*(_QWORD *)(v7 + 88);
    v122 = v120 + 1;
    v123 = *(_DWORD *)(v7 + 112);
    v258 = v121;
    if ( 4 * v122 >= 3 * v123 )
    {
      v123 *= 2;
    }
    else if ( v123 - *(_DWORD *)(v7 + 108) - v122 > v123 >> 3 )
    {
      goto LABEL_80;
    }
    sub_297DFD0(v7 + 88, v123);
    sub_297D7C0(v7 + 88, &v268, &v258);
    v122 = *(_DWORD *)(v7 + 104) + 1;
    v121 = v258;
LABEL_80:
    *(_DWORD *)(v7 + 104) = v122;
    if ( *(_QWORD *)(v121 + 16) != -4096 || *(_QWORD *)(v121 + 8) != -4096 || *(_QWORD *)v121 != -4096 )
      --*(_DWORD *)(v7 + 108);
    v124 = v270;
    *(_QWORD *)(v121 + 24) = 0;
    v53 = v121 + 24;
    *(_QWORD *)(v121 + 32) = 0;
    *(_QWORD *)(v121 + 16) = v124;
    v125 = v269;
    *(_QWORD *)(v121 + 40) = 0;
    *(_QWORD *)(v121 + 8) = v125;
    v126 = v268;
    *(_DWORD *)(v121 + 48) = 0;
    *(_QWORD *)v121 = v126;
    goto LABEL_83;
  }
  v50 = v257;
  v51 = *(_DWORD *)(v257 + 48);
  v52 = *(_QWORD *)(v257 + 32);
  v53 = v257 + 24;
  if ( !v51 )
  {
LABEL_83:
    ++*(_QWORD *)v53;
    v51 = 0;
    goto LABEL_84;
  }
  v54 = v51 - 1;
  v55 = 1;
  v56 = (unsigned int)v54 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
  v57 = (__int64 *)(v52 + 88 * v56);
  v58 = 0;
  v59 = *v57;
  if ( *v57 == v35 )
  {
LABEL_19:
    v60 = *((unsigned int *)v57 + 4);
    v61 = v57 + 1;
    if ( v60 + 1 > (unsigned __int64)*((unsigned int *)v57 + 5) )
    {
      sub_C8D5F0((__int64)(v57 + 1), v57 + 3, v60 + 1, 8u, v56, v54);
      v60 = *((unsigned int *)v57 + 4);
    }
    goto LABEL_21;
  }
  while ( v59 != -4096 )
  {
    if ( !v58 && v59 == -8192 )
      v58 = v57;
    v56 = (unsigned int)v54 & (v55 + (_DWORD)v56);
    v57 = (__int64 *)(v52 + 88LL * (unsigned int)v56);
    v59 = *v57;
    if ( v35 == *v57 )
      goto LABEL_19;
    ++v55;
  }
  v152 = *(_DWORD *)(v257 + 40);
  if ( !v58 )
    v58 = v57;
  ++*(_QWORD *)(v257 + 24);
  v131 = v152 + 1;
  if ( 4 * (v152 + 1) < 3 * v51 )
  {
    if ( v51 - *(_DWORD *)(v50 + 44) - v131 > v51 >> 3 )
      goto LABEL_130;
    v248 = v53;
    sub_297D910(v53, v51);
    v179 = *(_DWORD *)(v50 + 48);
    v53 = v248;
    if ( v179 )
    {
      v180 = v179 - 1;
      v181 = *(_QWORD *)(v50 + 32);
      v134 = 0;
      v182 = 1;
      LODWORD(v183) = (v179 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v131 = *(_DWORD *)(v50 + 40) + 1;
      v58 = (_QWORD *)(v181 + 88LL * (unsigned int)v183);
      v184 = *v58;
      if ( v35 == *v58 )
        goto LABEL_130;
      while ( v184 != -4096 )
      {
        if ( !v134 && v184 == -8192 )
          v134 = v58;
        v183 = v180 & (unsigned int)(v183 + v182);
        v58 = (_QWORD *)(v181 + 88 * v183);
        v184 = *v58;
        if ( v35 == *v58 )
          goto LABEL_130;
        ++v182;
      }
LABEL_88:
      if ( v134 )
        v58 = v134;
      goto LABEL_130;
    }
LABEL_328:
    ++*(_DWORD *)(v53 + 16);
    BUG();
  }
LABEL_84:
  v245 = v53;
  sub_297D910(v53, 2 * v51);
  v53 = v245;
  v127 = *(_DWORD *)(v245 + 24);
  if ( !v127 )
    goto LABEL_328;
  v128 = v127 - 1;
  v129 = *(_QWORD *)(v245 + 8);
  LODWORD(v130) = (v127 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
  v131 = *(_DWORD *)(v245 + 16) + 1;
  v58 = (_QWORD *)(v129 + 88LL * (unsigned int)v130);
  v132 = *v58;
  if ( v35 != *v58 )
  {
    v133 = 1;
    v134 = 0;
    while ( v132 != -4096 )
    {
      if ( v132 == -8192 && !v134 )
        v134 = v58;
      v130 = v128 & (unsigned int)(v130 + v133);
      v58 = (_QWORD *)(v129 + 88 * v130);
      v132 = *v58;
      if ( v35 == *v58 )
        goto LABEL_130;
      ++v133;
    }
    goto LABEL_88;
  }
LABEL_130:
  *(_DWORD *)(v53 + 16) = v131;
  if ( *v58 != -4096 )
    --*(_DWORD *)(v53 + 20);
  *v58 = v35;
  v61 = v58 + 1;
  v58[1] = v58 + 3;
  v58[2] = 0x800000000LL;
  v60 = 0;
LABEL_21:
  *(_QWORD *)(*v61 + 8 * v60) = v255;
  ++*((_DWORD *)v61 + 2);
  if ( !(unsigned __int8)sub_297D7C0(v7 + 120, (__int64 *)&v271, &v257) )
  {
    v107 = *(_DWORD *)(v7 + 136);
    v108 = v257;
    ++*(_QWORD *)(v7 + 120);
    v109 = v107 + 1;
    v110 = *(_DWORD *)(v7 + 144);
    v258 = v108;
    if ( 4 * v109 >= 3 * v110 )
    {
      v110 *= 2;
    }
    else if ( v110 - *(_DWORD *)(v7 + 140) - v109 > v110 >> 3 )
    {
      goto LABEL_67;
    }
    sub_297DFD0(v7 + 120, v110);
    sub_297D7C0(v7 + 120, (__int64 *)&v271, &v258);
    v109 = *(_DWORD *)(v7 + 136) + 1;
    v108 = v258;
LABEL_67:
    *(_DWORD *)(v7 + 136) = v109;
    if ( *(_QWORD *)(v108 + 16) != -4096 || *(_QWORD *)(v108 + 8) != -4096 || *(_QWORD *)v108 != -4096 )
      --*(_DWORD *)(v7 + 140);
    *(_QWORD *)(v108 + 16) = v273;
    *(_QWORD *)(v108 + 8) = v272;
    v111 = (__int64)v271;
    *(_QWORD *)(v108 + 24) = 0;
    *(_QWORD *)(v108 + 32) = 0;
    *(_QWORD *)(v108 + 40) = 0;
    *(_DWORD *)(v108 + 48) = 0;
    *(_QWORD *)v108 = v111;
    v65 = v108 + 24;
    goto LABEL_70;
  }
  v62 = v257;
  v63 = *(_DWORD *)(v257 + 48);
  v64 = *(_QWORD *)(v257 + 32);
  v65 = v257 + 24;
  if ( !v63 )
  {
LABEL_70:
    ++*(_QWORD *)v65;
    v63 = 0;
    goto LABEL_71;
  }
  v66 = v63 - 1;
  v67 = 1;
  v68 = v66 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
  v69 = v64 + 88LL * v68;
  v70 = 0;
  v71 = *(_QWORD *)v69;
  if ( v35 == *(_QWORD *)v69 )
  {
LABEL_24:
    result = *(unsigned int *)(v69 + 16);
    v73 = (_QWORD *)(v69 + 8);
    if ( *(unsigned int *)(v69 + 20) < (unsigned __int64)(result + 1) )
    {
      sub_C8D5F0(v69 + 8, (const void *)(v69 + 24), result + 1, 8u, v66, v64);
      result = *(unsigned int *)(v69 + 16);
    }
    goto LABEL_26;
  }
  while ( v71 != -4096 )
  {
    if ( !v70 && v71 == -8192 )
      v70 = (_QWORD *)v69;
    v68 = v66 & (v67 + v68);
    v69 = v64 + 88LL * v68;
    v71 = *(_QWORD *)v69;
    if ( v35 == *(_QWORD *)v69 )
      goto LABEL_24;
    ++v67;
  }
  v151 = *(_DWORD *)(v257 + 40);
  if ( !v70 )
    v70 = (_QWORD *)v69;
  ++*(_QWORD *)(v257 + 24);
  v116 = v151 + 1;
  if ( 4 * (v151 + 1) >= 3 * v63 )
  {
LABEL_71:
    v244 = v65;
    sub_297D910(v65, 2 * v63);
    v65 = v244;
    v112 = *(_DWORD *)(v244 + 24);
    if ( v112 )
    {
      v113 = v112 - 1;
      v114 = *(_QWORD *)(v244 + 8);
      v115 = (v112 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v116 = *(_DWORD *)(v244 + 16) + 1;
      v70 = (_QWORD *)(v114 + 88LL * v115);
      v117 = *v70;
      if ( v35 != *v70 )
      {
        v118 = 1;
        v119 = 0;
        while ( v117 != -4096 )
        {
          if ( v117 == -8192 && !v119 )
            v119 = v70;
          v115 = v113 & (v118 + v115);
          v70 = (_QWORD *)(v114 + 88LL * v115);
          v117 = *v70;
          if ( v35 == *v70 )
            goto LABEL_117;
          ++v118;
        }
        if ( v119 )
          v70 = v119;
      }
      goto LABEL_117;
    }
    goto LABEL_327;
  }
  if ( v63 - *(_DWORD *)(v62 + 44) - v116 <= v63 >> 3 )
  {
    v249 = v65;
    sub_297D910(v65, v63);
    v185 = *(_DWORD *)(v62 + 48);
    v65 = v249;
    if ( v185 )
    {
      v186 = v185 - 1;
      v187 = *(_QWORD *)(v62 + 32);
      v188 = 0;
      v189 = (v185 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v190 = 1;
      v116 = *(_DWORD *)(v62 + 40) + 1;
      v70 = (_QWORD *)(v187 + 88LL * v189);
      v191 = *v70;
      if ( v35 != *v70 )
      {
        while ( v191 != -4096 )
        {
          if ( !v188 && v191 == -8192 )
            v188 = v70;
          v189 = v186 & (v190 + v189);
          v70 = (_QWORD *)(v187 + 88LL * v189);
          v191 = *v70;
          if ( v35 == *v70 )
            goto LABEL_117;
          ++v190;
        }
        if ( v188 )
          v70 = v188;
      }
      goto LABEL_117;
    }
LABEL_327:
    ++*(_DWORD *)(v65 + 16);
    BUG();
  }
LABEL_117:
  *(_DWORD *)(v65 + 16) = v116;
  if ( *v70 != -4096 )
    --*(_DWORD *)(v65 + 20);
  *v70 = v35;
  v70[2] = 0x800000000LL;
  v73 = v70 + 1;
  v70[1] = v70 + 3;
  result = 0;
LABEL_26:
  *(_QWORD *)(*v73 + 8 * result) = v255;
  ++*((_DWORD *)v73 + 2);
  return result;
}
