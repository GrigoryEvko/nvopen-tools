// Function: sub_1CDC1F0
// Address: 0x1cdc1f0
//
__int64 __fastcall sub_1CDC1F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15)
{
  __int64 v16; // rax
  __int64 v17; // r14
  unsigned int v19; // edi
  unsigned int v20; // edx
  __int64 **v21; // r9
  __int64 *v22; // r11
  __int64 v23; // r10
  unsigned int v24; // edx
  unsigned __int64 v25; // rbx
  __int64 v26; // r9
  __int64 v27; // rax
  _QWORD *v28; // rax
  int v29; // esi
  __int64 v30; // rcx
  unsigned int v31; // edx
  __int64 *v32; // rax
  __int64 v33; // rdi
  unsigned __int64 v34; // r8
  __int64 v35; // rax
  _QWORD *v36; // rcx
  unsigned int v37; // esi
  __int64 v38; // rax
  int v39; // eax
  __int64 *v40; // r14
  int v41; // esi
  __int64 v42; // rax
  int v43; // r8d
  unsigned int v44; // edx
  __int64 *v45; // r12
  __int64 v46; // r9
  __int64 *v47; // rax
  __int64 v48; // rdx
  unsigned int v49; // esi
  __int64 v50; // r8
  unsigned int v51; // edi
  __int64 *v52; // rcx
  __int64 v53; // r10
  __int64 v54; // r13
  unsigned int v55; // r14d
  unsigned int v56; // ecx
  __int64 **v57; // rax
  __int64 *v58; // r9
  __int64 v59; // rsi
  __int64 v60; // rax
  int v61; // esi
  __int64 v62; // rbx
  __int64 *v63; // rcx
  unsigned int v64; // edx
  __int64 **v65; // rax
  __int64 *v66; // rdi
  int v67; // esi
  __int64 *v68; // rcx
  unsigned int v69; // edx
  __int64 **v70; // rax
  __int64 *v71; // rdi
  __int64 **v72; // rsi
  __int64 *v73; // rdx
  __int64 **v74; // rdi
  int v75; // eax
  int v76; // r11d
  __int64 *v77; // rcx
  int v78; // edi
  unsigned __int64 v79; // rax
  __int64 **v80; // rcx
  __int64 **v81; // rax
  __int64 v82; // rdi
  __int64 v83; // rdx
  __int64 v84; // r12
  __int64 v85; // rcx
  int v86; // edi
  __int64 v87; // rsi
  unsigned int v88; // edx
  __int64 *v89; // rax
  __int64 v90; // r9
  __int64 v91; // rax
  int v92; // esi
  __int64 v93; // rdx
  unsigned int v94; // ecx
  __int64 *v95; // rax
  __int64 v96; // r9
  __int64 v97; // rax
  _QWORD *v98; // r12
  __int64 *v99; // rax
  int v100; // esi
  __int64 v101; // r14
  __int64 v102; // rcx
  unsigned int v103; // edx
  __int64 *v104; // rax
  __int64 v105; // r8
  bool v106; // al
  double v107; // xmm4_8
  double v108; // xmm5_8
  __int64 v109; // rbx
  int v110; // edx
  __int64 v111; // rdi
  int v112; // edx
  __int64 v113; // r8
  unsigned int v114; // esi
  __int64 *v115; // rax
  __int64 v116; // r10
  __int64 v117; // r13
  int v118; // r8d
  _QWORD *v119; // r15
  int v120; // eax
  __int64 v121; // rax
  char v122; // al
  _QWORD *v123; // rdx
  __int64 v124; // rbx
  int v125; // eax
  unsigned int v126; // esi
  int v127; // eax
  int v128; // r11d
  __int64 **v129; // r10
  int v130; // edi
  int v131; // r11d
  __int64 **v132; // r10
  int v133; // edi
  int v134; // r11d
  _QWORD *v135; // rax
  __int64 i; // rdx
  _QWORD *v137; // rbx
  _QWORD *v138; // r13
  unsigned __int64 v139; // rdi
  int v141; // eax
  __int64 *v142; // r12
  __int64 v143; // rcx
  unsigned __int64 v144; // r15
  __int64 v145; // r12
  __int64 v146; // rdx
  int v147; // edx
  unsigned int v148; // ebx
  unsigned int v149; // eax
  __int64 *v150; // r15
  __int64 v151; // rcx
  __int64 v152; // r14
  __int64 *v153; // rax
  __int64 v154; // rcx
  unsigned int v155; // esi
  __int64 *v156; // rdx
  __int64 v157; // r8
  __int64 v158; // r10
  __int64 v159; // rax
  __int64 *v160; // r14
  __int64 v161; // rdx
  unsigned __int64 v162; // rax
  __int64 v163; // rax
  __int64 v164; // rbx
  _QWORD *v165; // rbx
  int v166; // edx
  bool v167; // al
  _QWORD *v168; // r10
  __int64 *v169; // rax
  int v170; // r11d
  __int64 *v171; // rdi
  int v172; // edx
  int v173; // r11d
  __int64 *v174; // r8
  int v175; // ecx
  int v176; // r8d
  __int64 *v177; // r11
  int v178; // edi
  int v179; // r11d
  int v180; // ebx
  int v181; // eax
  __int64 v182; // rax
  _QWORD *v183; // rax
  _QWORD *v184; // r13
  unsigned __int64 v185; // rax
  _QWORD *v186; // r10
  _QWORD *v187; // rax
  int v188; // r13d
  __int64 *v189; // r9
  int v190; // ecx
  __int64 **v191; // rbx
  int v192; // r9d
  __int64 *v193; // rbx
  __int64 *v194; // rbx
  int v195; // ecx
  __int64 v196; // [rsp+8h] [rbp-228h]
  __int64 v197; // [rsp+10h] [rbp-220h]
  __int64 v198; // [rsp+18h] [rbp-218h]
  __int64 v199; // [rsp+20h] [rbp-210h]
  unsigned __int64 v200; // [rsp+20h] [rbp-210h]
  __int64 v201; // [rsp+20h] [rbp-210h]
  _QWORD *v203; // [rsp+38h] [rbp-1F8h]
  bool v204; // [rsp+40h] [rbp-1F0h]
  unsigned __int64 v205; // [rsp+48h] [rbp-1E8h]
  __int64 v206; // [rsp+48h] [rbp-1E8h]
  __int64 *v208; // [rsp+50h] [rbp-1E0h]
  __int64 *v211; // [rsp+68h] [rbp-1C8h]
  __int64 v212; // [rsp+68h] [rbp-1C8h]
  unsigned int v214; // [rsp+80h] [rbp-1B0h]
  __int64 v215; // [rsp+88h] [rbp-1A8h]
  _QWORD *v216; // [rsp+88h] [rbp-1A8h]
  unsigned int v217; // [rsp+90h] [rbp-1A0h]
  __int64 v218; // [rsp+90h] [rbp-1A0h]
  __int64 v219; // [rsp+98h] [rbp-198h]
  __int64 *v220; // [rsp+98h] [rbp-198h]
  unsigned __int64 v221; // [rsp+98h] [rbp-198h]
  __int64 v222; // [rsp+A8h] [rbp-188h] BYREF
  __int64 v223; // [rsp+B0h] [rbp-180h] BYREF
  __int64 v224; // [rsp+B8h] [rbp-178h] BYREF
  __int64 **v225; // [rsp+C0h] [rbp-170h] BYREF
  __int64 **v226; // [rsp+C8h] [rbp-168h]
  __int64 **v227; // [rsp+D0h] [rbp-160h]
  __int64 v228; // [rsp+E0h] [rbp-150h] BYREF
  _QWORD *v229; // [rsp+E8h] [rbp-148h]
  __int64 v230; // [rsp+F0h] [rbp-140h]
  unsigned int v231; // [rsp+F8h] [rbp-138h]
  __int64 v232; // [rsp+100h] [rbp-130h] BYREF
  __int64 v233; // [rsp+108h] [rbp-128h]
  __int64 v234; // [rsp+110h] [rbp-120h]
  unsigned int v235; // [rsp+118h] [rbp-118h]
  __int64 v236; // [rsp+120h] [rbp-110h] BYREF
  __int64 v237; // [rsp+128h] [rbp-108h]
  __int64 v238; // [rsp+130h] [rbp-100h]
  unsigned int v239; // [rsp+138h] [rbp-F8h]
  __int64 v240; // [rsp+140h] [rbp-F0h] BYREF
  __int64 v241; // [rsp+148h] [rbp-E8h]
  __int64 v242; // [rsp+150h] [rbp-E0h]
  unsigned int v243; // [rsp+158h] [rbp-D8h]
  __int64 *v244; // [rsp+160h] [rbp-D0h] BYREF
  __int64 v245; // [rsp+168h] [rbp-C8h]
  __int64 v246; // [rsp+170h] [rbp-C0h]
  unsigned int v247; // [rsp+178h] [rbp-B8h]
  __int64 *v248[2]; // [rsp+180h] [rbp-B0h] BYREF
  _QWORD v249[2]; // [rsp+190h] [rbp-A0h] BYREF
  __int64 *v250; // [rsp+1A0h] [rbp-90h] BYREF
  __int64 v251; // [rsp+1A8h] [rbp-88h]
  _BYTE v252[32]; // [rsp+1B0h] [rbp-80h] BYREF
  __int64 *v253; // [rsp+1D0h] [rbp-60h] BYREF
  __int64 v254; // [rsp+1D8h] [rbp-58h]
  _BYTE v255[80]; // [rsp+1E0h] [rbp-50h] BYREF

  v250 = (__int64 *)v252;
  v251 = 0x400000000LL;
  v16 = *(unsigned int *)(a2 + 8);
  v228 = 0;
  v229 = 0;
  v230 = 0;
  v231 = 0;
  v232 = 0;
  v233 = 0;
  v234 = 0;
  v235 = 0;
  if ( (_DWORD)v16 )
  {
    v17 = 0;
    v219 = 8 * v16;
    while ( 1 )
    {
      v240 = *(_QWORD *)(*(_QWORD *)a2 + v17);
      sub_1C55090(v240, a5, (__int64 *)&v244, (__int64 *)v248, a7, a8);
      v35 = (__int64)v244;
      if ( !v244 )
        goto LABEL_11;
      v36 = v229;
      v37 = v231;
      if ( v231 )
      {
        v19 = v231 - 1;
        v20 = (v231 - 1) & (((unsigned int)v244 >> 9) ^ ((unsigned int)v244 >> 4));
        v21 = (__int64 **)&v229[7 * v20];
        v22 = *v21;
        if ( v244 == *v21 )
        {
LABEL_4:
          if ( &v229[7 * v231] != v21 )
            goto LABEL_5;
        }
        else
        {
          LODWORD(v21) = 1;
          while ( v22 != (__int64 *)-8LL )
          {
            v180 = (_DWORD)v21 + 1;
            v20 = v19 & ((_DWORD)v21 + v20);
            LODWORD(v34) = 7 * v20;
            v21 = (__int64 **)&v229[7 * v20];
            v22 = *v21;
            if ( v244 == *v21 )
              goto LABEL_4;
            LODWORD(v21) = v180;
          }
        }
      }
      v38 = (unsigned int)v251;
      if ( (unsigned int)v251 >= HIDWORD(v251) )
      {
        sub_16CD150((__int64)&v250, v252, 0, 8, v34, (int)v21);
        v38 = (unsigned int)v251;
      }
      v250[v38] = (__int64)v244;
      v37 = v231;
      LODWORD(v251) = v251 + 1;
      v36 = v229;
      if ( !v231 )
      {
        ++v228;
        goto LABEL_18;
      }
      v35 = (__int64)v244;
      v19 = v231 - 1;
LABEL_5:
      v23 = v35;
      v24 = v19 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v25 = (unsigned __int64)&v36[7 * v24];
      v26 = *(_QWORD *)v25;
      if ( v35 == *(_QWORD *)v25 )
      {
LABEL_6:
        v27 = *(unsigned int *)(v25 + 16);
        if ( (unsigned int)v27 >= *(_DWORD *)(v25 + 20) )
        {
          sub_16CD150(v25 + 8, (const void *)(v25 + 24), 0, 8, v34, v26);
          v28 = (_QWORD *)(*(_QWORD *)(v25 + 8) + 8LL * *(unsigned int *)(v25 + 16));
        }
        else
        {
          v28 = (_QWORD *)(*(_QWORD *)(v25 + 8) + 8 * v27);
        }
        goto LABEL_8;
      }
      v179 = 1;
      v34 = 0;
      while ( v26 != -8 )
      {
        if ( v26 == -16 && !v34 )
          v34 = v25;
        v24 = v19 & (v179 + v24);
        v25 = (unsigned __int64)&v36[7 * v24];
        v26 = *(_QWORD *)v25;
        if ( v35 == *(_QWORD *)v25 )
          goto LABEL_6;
        ++v179;
      }
      if ( v34 )
        v25 = v34;
      ++v228;
      v39 = v230 + 1;
      if ( 4 * ((int)v230 + 1) < 3 * v37 )
      {
        if ( v37 - (v39 + HIDWORD(v230)) > v37 >> 3 )
          goto LABEL_20;
        goto LABEL_19;
      }
LABEL_18:
      v37 *= 2;
LABEL_19:
      sub_1CDBD70((__int64)&v228, v37);
      sub_1CD2E10((__int64)&v228, (__int64 *)&v244, &v253);
      v25 = (unsigned __int64)v253;
      v23 = (__int64)v244;
      v39 = v230 + 1;
LABEL_20:
      LODWORD(v230) = v39;
      if ( *(_QWORD *)v25 != -8 )
        --HIDWORD(v230);
      v28 = (_QWORD *)(v25 + 24);
      *(_QWORD *)v25 = v23;
      *(_QWORD *)(v25 + 8) = v25 + 24;
      *(_QWORD *)(v25 + 16) = 0x400000000LL;
LABEL_8:
      *v28 = v240;
      ++*(_DWORD *)(v25 + 16);
      v29 = v235;
      if ( !v235 )
      {
        ++v232;
        goto LABEL_250;
      }
      v30 = v240;
      v31 = (v235 - 1) & (((unsigned int)v240 >> 9) ^ ((unsigned int)v240 >> 4));
      v32 = (__int64 *)(v233 + 16LL * v31);
      v33 = *v32;
      if ( v240 != *v32 )
      {
        v176 = 1;
        v177 = 0;
        while ( v33 != -8 )
        {
          if ( v33 == -16 && !v177 )
            v177 = v32;
          v31 = (v235 - 1) & (v176 + v31);
          v32 = (__int64 *)(v233 + 16LL * v31);
          v33 = *v32;
          if ( v240 == *v32 )
            goto LABEL_10;
          ++v176;
        }
        if ( v177 )
          v32 = v177;
        ++v232;
        v178 = v234 + 1;
        if ( 4 * ((int)v234 + 1) < 3 * v235 )
        {
          if ( v235 - HIDWORD(v234) - v178 > v235 >> 3 )
          {
LABEL_246:
            LODWORD(v234) = v178;
            if ( *v32 != -8 )
              --HIDWORD(v234);
            *v32 = v30;
            v32[1] = 0;
            goto LABEL_10;
          }
LABEL_251:
          sub_1466670((__int64)&v232, v29);
          sub_14614C0((__int64)&v232, &v240, &v253);
          v32 = v253;
          v30 = v240;
          v178 = v234 + 1;
          goto LABEL_246;
        }
LABEL_250:
        v29 = 2 * v235;
        goto LABEL_251;
      }
LABEL_10:
      v32[1] = (__int64)v248[0];
LABEL_11:
      v17 += 8;
      if ( v17 == v219 )
      {
        v40 = v250;
        v211 = &v250[(unsigned int)v251];
        goto LABEL_24;
      }
    }
  }
  v40 = (__int64 *)v252;
  v211 = (__int64 *)v252;
LABEL_24:
  v236 = 0;
  v237 = 0;
  v238 = 0;
  v239 = 0;
  v225 = 0;
  v226 = 0;
  v227 = 0;
  v240 = 0;
  v241 = 0;
  v242 = 0;
  v243 = 0;
  if ( v211 == v40 )
  {
    v82 = 0;
    LODWORD(v84) = *(_DWORD *)(a2 + 8);
    if ( !(_DWORD)v84 )
      goto LABEL_142;
    v214 = 0;
    LODWORD(v83) = 0;
    v81 = 0;
    v80 = 0;
LABEL_67:
    *(_DWORD *)(a2 + 8) = v214;
    if ( v80 == v81 )
      goto LABEL_141;
    goto LABEL_68;
  }
  v220 = v40;
  v214 = 0;
  do
  {
    v41 = v231;
    v42 = *v220;
    v224 = *v220;
    if ( !v231 )
    {
      ++v228;
      goto LABEL_156;
    }
    v43 = v231 - 1;
    v44 = (v231 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
    v45 = &v229[7 * v44];
    v46 = *v45;
    if ( v42 == *v45 )
    {
      v47 = (__int64 *)v45[1];
      v217 = *((_DWORD *)v45 + 4);
      goto LABEL_29;
    }
    v76 = 1;
    v77 = 0;
    while ( 1 )
    {
      if ( v46 == -8 )
      {
        if ( !v77 )
          v77 = v45;
        ++v228;
        v78 = v230 + 1;
        if ( 4 * ((int)v230 + 1) < 3 * v231 )
        {
          v43 = v231 >> 3;
          if ( v231 - HIDWORD(v230) - v78 > v231 >> 3 )
          {
LABEL_60:
            LODWORD(v230) = v78;
            if ( *v77 != -8 )
              --HIDWORD(v230);
            *v77 = v42;
            v48 = v77[3];
            v77[1] = (__int64)(v77 + 3);
            v77[2] = 0x400000000LL;
LABEL_63:
            *(_QWORD *)(*(_QWORD *)a2 + 8LL * v214++) = v48;
            goto LABEL_64;
          }
LABEL_157:
          sub_1CDBD70((__int64)&v228, v41);
          sub_1CD2E10((__int64)&v228, &v224, &v253);
          v77 = v253;
          v42 = v224;
          v78 = v230 + 1;
          goto LABEL_60;
        }
LABEL_156:
        v41 = 2 * v231;
        goto LABEL_157;
      }
      if ( v46 != -16 || v77 )
        v45 = v77;
      v44 = v43 & (v76 + v44);
      v186 = &v229[7 * v44];
      v46 = *v186;
      if ( v42 == *v186 )
        break;
      v77 = v45;
      ++v76;
      v45 = &v229[7 * v44];
    }
    v47 = (__int64 *)v186[1];
    v45 = &v229[7 * v44];
    v217 = *((_DWORD *)v186 + 4);
LABEL_29:
    v48 = *v47;
    v244 = (__int64 *)*v47;
    if ( v217 <= 1 )
      goto LABEL_63;
    v49 = v235;
    if ( !v235 )
    {
      ++v232;
LABEL_303:
      sub_1466670((__int64)&v232, 2 * v235);
      goto LABEL_304;
    }
    v50 = v233;
    v51 = (v235 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
    v52 = (__int64 *)(v233 + 16LL * v51);
    v53 = *v52;
    if ( v48 == *v52 )
    {
LABEL_32:
      v54 = v52[1];
      goto LABEL_33;
    }
    v188 = 1;
    v189 = 0;
    while ( v53 != -8 )
    {
      if ( !v189 && v53 == -16 )
        v189 = v52;
      v51 = (v235 - 1) & (v188 + v51);
      v52 = (__int64 *)(v233 + 16LL * v51);
      v53 = *v52;
      if ( v48 == *v52 )
        goto LABEL_32;
      ++v188;
    }
    if ( !v189 )
      v189 = v52;
    ++v232;
    v190 = v234 + 1;
    if ( 4 * ((int)v234 + 1) >= 3 * v235 )
      goto LABEL_303;
    if ( v235 - HIDWORD(v234) - v190 > v235 >> 3 )
      goto LABEL_282;
    sub_1466670((__int64)&v232, v235);
LABEL_304:
    sub_14614C0((__int64)&v232, (__int64 *)&v244, &v253);
    v189 = v253;
    v48 = (__int64)v244;
    v190 = v234 + 1;
LABEL_282:
    LODWORD(v234) = v190;
    if ( *v189 != -8 )
      --HIDWORD(v234);
    *v189 = v48;
    v189[1] = 0;
    v217 = *((_DWORD *)v45 + 4);
    if ( v217 != 1 )
    {
      v47 = (__int64 *)v45[1];
      v50 = v233;
      v54 = 0;
      v49 = v235;
LABEL_33:
      v55 = 1;
      while ( 2 )
      {
        v73 = (__int64 *)v47[v55];
        v248[0] = v73;
        if ( !v49 )
        {
          ++v232;
          goto LABEL_49;
        }
        v56 = (v49 - 1) & (((unsigned int)v73 >> 9) ^ ((unsigned int)v73 >> 4));
        v57 = (__int64 **)(v50 + 16LL * v56);
        v58 = *v57;
        if ( v73 == *v57 )
        {
          v59 = (__int64)v57[1];
        }
        else
        {
          v134 = 1;
          v74 = 0;
          while ( v58 != (__int64 *)-8LL )
          {
            if ( v74 || v58 != (__int64 *)-16LL )
              v57 = v74;
            v56 = (v49 - 1) & (v134 + v56);
            v191 = (__int64 **)(v50 + 16LL * v56);
            v58 = *v191;
            if ( v73 == *v191 )
            {
              v59 = (__int64)v191[1];
              goto LABEL_36;
            }
            ++v134;
            v74 = v57;
            v57 = (__int64 **)(v50 + 16LL * v56);
          }
          if ( !v74 )
            v74 = v57;
          ++v232;
          v75 = v234 + 1;
          if ( 4 * ((int)v234 + 1) >= 3 * v49 )
          {
LABEL_49:
            v49 *= 2;
          }
          else if ( v49 - (v75 + HIDWORD(v234)) > v49 >> 3 )
          {
            goto LABEL_51;
          }
          sub_1466670((__int64)&v232, v49);
          sub_14614C0((__int64)&v232, (__int64 *)v248, &v253);
          v74 = (__int64 **)v253;
          v73 = v248[0];
          v75 = v234 + 1;
LABEL_51:
          LODWORD(v234) = v75;
          if ( *v74 != (__int64 *)-8LL )
            --HIDWORD(v234);
          *v74 = v73;
          v59 = 0;
          v74[1] = 0;
        }
LABEL_36:
        v60 = sub_14806B0(a5, v59, v54, 0, 0);
        v61 = v243;
        v62 = v60;
        if ( *(_WORD *)(v60 + 24) )
          v62 = 0;
        if ( v243 )
        {
          v63 = v248[0];
          v64 = (v243 - 1) & ((LODWORD(v248[0]) >> 9) ^ (LODWORD(v248[0]) >> 4));
          v65 = (__int64 **)(v241 + 16LL * v64);
          v66 = *v65;
          if ( v248[0] == *v65 )
            goto LABEL_40;
          v131 = 1;
          v132 = 0;
          while ( v66 != (__int64 *)-8LL )
          {
            if ( !v132 && v66 == (__int64 *)-16LL )
              v132 = v65;
            v64 = (v243 - 1) & (v131 + v64);
            v65 = (__int64 **)(v241 + 16LL * v64);
            v66 = *v65;
            if ( v248[0] == *v65 )
              goto LABEL_40;
            ++v131;
          }
          if ( v132 )
            v65 = v132;
          ++v240;
          v133 = v242 + 1;
          if ( 4 * ((int)v242 + 1) < 3 * v243 )
          {
            if ( v243 - HIDWORD(v242) - v133 > v243 >> 3 )
            {
LABEL_118:
              LODWORD(v242) = v133;
              if ( *v65 != (__int64 *)-8LL )
                --HIDWORD(v242);
              *v65 = v63;
              v65[1] = 0;
LABEL_40:
              v65[1] = v244;
              v67 = v239;
              if ( v239 )
              {
                v68 = v248[0];
                v69 = (v239 - 1) & ((LODWORD(v248[0]) >> 9) ^ (LODWORD(v248[0]) >> 4));
                v70 = (__int64 **)(v237 + 16LL * v69);
                v71 = *v70;
                if ( *v70 == v248[0] )
                {
LABEL_42:
                  v70[1] = (__int64 *)v62;
                  v72 = v226;
                  if ( v226 == v227 )
                  {
                    ++v55;
                    sub_1C55B50((__int64)&v225, v226, v248);
                    if ( v55 == v217 )
                      goto LABEL_94;
                  }
                  else
                  {
                    if ( v226 )
                    {
                      *v226 = v248[0];
                      v72 = v226;
                    }
                    ++v55;
                    v226 = v72 + 1;
                    if ( v55 == v217 )
                      goto LABEL_94;
                  }
                  v50 = v233;
                  v49 = v235;
                  v47 = (__int64 *)v45[1];
                  continue;
                }
                v128 = 1;
                v129 = 0;
                while ( v71 != (__int64 *)-8LL )
                {
                  if ( v71 == (__int64 *)-16LL && !v129 )
                    v129 = v70;
                  v69 = (v239 - 1) & (v128 + v69);
                  v70 = (__int64 **)(v237 + 16LL * v69);
                  v71 = *v70;
                  if ( v248[0] == *v70 )
                    goto LABEL_42;
                  ++v128;
                }
                if ( v129 )
                  v70 = v129;
                ++v236;
                v130 = v238 + 1;
                if ( 4 * ((int)v238 + 1) < 3 * v239 )
                {
                  if ( v239 - HIDWORD(v238) - v130 > v239 >> 3 )
                  {
LABEL_106:
                    LODWORD(v238) = v130;
                    if ( *v70 != (__int64 *)-8LL )
                      --HIDWORD(v238);
                    *v70 = v68;
                    v70[1] = 0;
                    goto LABEL_42;
                  }
LABEL_111:
                  sub_1CDC060((__int64)&v236, v67);
                  sub_1CD37D0((__int64)&v236, (__int64 *)v248, &v253);
                  v70 = (__int64 **)v253;
                  v68 = v248[0];
                  v130 = v238 + 1;
                  goto LABEL_106;
                }
              }
              else
              {
                ++v236;
              }
              v67 = 2 * v239;
              goto LABEL_111;
            }
LABEL_123:
            sub_1466670((__int64)&v240, v61);
            sub_14614C0((__int64)&v240, (__int64 *)v248, &v253);
            v65 = (__int64 **)v253;
            v63 = v248[0];
            v133 = v242 + 1;
            goto LABEL_118;
          }
        }
        else
        {
          ++v240;
        }
        break;
      }
      v61 = 2 * v243;
      goto LABEL_123;
    }
LABEL_94:
    v122 = sub_19A5930(a4, (__int64 *)&v244, &v253);
    v123 = v253;
    if ( !v122 )
    {
      v124 = a4;
      v125 = *(_DWORD *)(a4 + 16);
      v126 = *(_DWORD *)(a4 + 24);
      ++*(_QWORD *)a4;
      v43 = 2 * v126;
      v127 = v125 + 1;
      if ( 4 * v127 >= 3 * v126 )
      {
        v126 *= 2;
      }
      else
      {
        v124 = a4;
        if ( v126 - *(_DWORD *)(a4 + 20) - v127 > v126 >> 3 )
          goto LABEL_97;
      }
      sub_19AEEA0(v124, v126);
      sub_19A5930(v124, (__int64 *)&v244, &v253);
      v123 = v253;
      v127 = *(_DWORD *)(v124 + 16) + 1;
LABEL_97:
      *(_DWORD *)(a4 + 16) = v127;
      if ( *v123 != -8 )
        --*(_DWORD *)(a4 + 20);
      *v123 = v244;
    }
LABEL_64:
    ++v220;
  }
  while ( v220 != v211 );
  v79 = *(unsigned int *)(a2 + 8);
  if ( v79 > v214 )
  {
    v80 = v226;
    v81 = v225;
    v82 = v241;
    v83 = v226 - v225;
    goto LABEL_67;
  }
  if ( v79 < v214 )
  {
    if ( v214 > *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, (const void *)(a2 + 16), v214, 8, v43, v46);
      v79 = *(unsigned int *)(a2 + 8);
    }
    v135 = (_QWORD *)(*(_QWORD *)a2 + 8 * v79);
    for ( i = *(_QWORD *)a2 + 8LL * v214; (_QWORD *)i != v135; ++v135 )
    {
      if ( v135 )
        *v135 = 0;
    }
    *(_DWORD *)(a2 + 8) = v214;
  }
  v81 = v225;
  v82 = v241;
  v83 = v226 - v225;
  if ( v226 == v225 )
  {
LABEL_141:
    LODWORD(v84) = 0;
    goto LABEL_142;
  }
LABEL_68:
  LODWORD(v84) = v83;
  if ( !(_DWORD)v83 )
    goto LABEL_142;
  v198 = a5;
  v221 = 0;
  v212 = 8LL * (unsigned int)v83;
  while ( 2 )
  {
    v92 = v243;
    v93 = (__int64)v81[v221 / 8];
    v222 = v93;
    if ( !v243 )
    {
      ++v240;
LABEL_238:
      v92 = 2 * v243;
LABEL_239:
      sub_1466670((__int64)&v240, v92);
      sub_14614C0((__int64)&v240, &v222, &v253);
      v174 = v253;
      v93 = v222;
      v175 = v242 + 1;
      goto LABEL_234;
    }
    v94 = (v243 - 1) & (((unsigned int)v93 >> 9) ^ ((unsigned int)v93 >> 4));
    v95 = (__int64 *)(v82 + 16LL * v94);
    v96 = *v95;
    if ( v93 == *v95 )
    {
      v97 = v95[1];
      goto LABEL_79;
    }
    v173 = 1;
    v174 = 0;
    while ( v96 != -8 )
    {
      if ( v96 != -16 || v174 )
        v95 = v174;
      v94 = (v243 - 1) & (v173 + v94);
      v194 = (__int64 *)(v82 + 16LL * v94);
      v96 = *v194;
      if ( v93 == *v194 )
      {
        v97 = v194[1];
        goto LABEL_79;
      }
      ++v173;
      v174 = v95;
      v95 = (__int64 *)(v82 + 16LL * v94);
    }
    if ( !v174 )
      v174 = v95;
    ++v240;
    v175 = v242 + 1;
    if ( 4 * ((int)v242 + 1) >= 3 * v243 )
      goto LABEL_238;
    if ( v243 - HIDWORD(v242) - v175 <= v243 >> 3 )
      goto LABEL_239;
LABEL_234:
    LODWORD(v242) = v175;
    if ( *v174 != -8 )
      --HIDWORD(v242);
    *v174 = v93;
    v97 = 0;
    v174[1] = 0;
LABEL_79:
    v223 = v97;
    v98 = (_QWORD *)sub_1CDBC40(a3, &v223)[1];
    v99 = sub_1CDBC40(a3, &v222);
    v100 = v239;
    v101 = v99[1];
    if ( !v239 )
    {
      ++v236;
LABEL_226:
      v100 = 2 * v239;
LABEL_227:
      sub_1CDC060((__int64)&v236, v100);
      sub_1CD37D0((__int64)&v236, &v222, &v253);
      v171 = v253;
      v102 = v222;
      v172 = v238 + 1;
      goto LABEL_222;
    }
    v102 = v222;
    v103 = (v239 - 1) & (((unsigned int)v222 >> 9) ^ ((unsigned int)v222 >> 4));
    v104 = (__int64 *)(v237 + 16LL * v103);
    v105 = *v104;
    if ( *v104 == v222 )
    {
      v218 = v104[1];
      v106 = sub_14560B0(v218);
      goto LABEL_82;
    }
    v170 = 1;
    v171 = 0;
    while ( v105 != -8 )
    {
      if ( v171 || v105 != -16 )
        v104 = v171;
      v103 = (v239 - 1) & (v170 + v103);
      v193 = (__int64 *)(v237 + 16LL * v103);
      v105 = *v193;
      if ( v222 == *v193 )
      {
        v218 = v193[1];
        v106 = sub_14560B0(v218);
        goto LABEL_82;
      }
      ++v170;
      v171 = v104;
      v104 = (__int64 *)(v237 + 16LL * v103);
    }
    if ( !v171 )
      v171 = v104;
    ++v236;
    v172 = v238 + 1;
    if ( 4 * ((int)v238 + 1) >= 3 * v239 )
      goto LABEL_226;
    if ( v239 - HIDWORD(v238) - v172 <= v239 >> 3 )
      goto LABEL_227;
LABEL_222:
    LODWORD(v238) = v172;
    if ( *v171 != -8 )
      --HIDWORD(v238);
    *v171 = v102;
    v171[1] = 0;
    v218 = 0;
    v106 = sub_14560B0(0);
LABEL_82:
    if ( v106 && *(_QWORD *)v101 == *v98 )
    {
      sub_164D160(v101, (__int64)v98, (__m128)a7, *(double *)a8.m128i_i64, a9, a10, v107, v108, a13, a14);
      sub_1CDBC40(a3, &v222)[1] = (__int64)v98;
      goto LABEL_75;
    }
    v215 = 0;
    v253 = (__int64 *)v255;
    v254 = 0x400000000LL;
    v109 = *(_QWORD *)(a1 + 160);
    v110 = *(_DWORD *)(v109 + 24);
    if ( v110 )
    {
      v111 = *(_QWORD *)(v101 + 40);
      v112 = v110 - 1;
      v113 = *(_QWORD *)(v109 + 8);
      v114 = v112 & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
      v115 = (__int64 *)(v113 + 16LL * v114);
      v116 = *v115;
      if ( v111 == *v115 )
      {
LABEL_86:
        v215 = v115[1];
      }
      else
      {
        v181 = 1;
        while ( v116 != -8 )
        {
          v195 = v181 + 1;
          v114 = v112 & (v181 + v114);
          v115 = (__int64 *)(v113 + 16LL * v114);
          v116 = *v115;
          if ( v111 == *v115 )
            goto LABEL_86;
          v181 = v195;
        }
        v215 = 0;
      }
    }
    v117 = *(_QWORD *)(v101 + 8);
    if ( v117 )
    {
      while ( 1 )
      {
        v119 = sub_1648700(v117);
        v120 = *(_DWORD *)(v109 + 24);
        if ( v120 )
        {
          v85 = v119[5];
          v86 = v120 - 1;
          v87 = *(_QWORD *)(v109 + 8);
          v88 = (v120 - 1) & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
          v89 = (__int64 *)(v87 + 16LL * v88);
          v90 = *v89;
          if ( v85 == *v89 )
          {
LABEL_71:
            v91 = v89[1];
            if ( v91 && v215 != v91 )
              goto LABEL_73;
          }
          else
          {
            v141 = 1;
            while ( v90 != -8 )
            {
              v118 = v141 + 1;
              v88 = v86 & (v141 + v88);
              v89 = (__int64 *)(v87 + 16LL * v88);
              v90 = *v89;
              if ( v85 == *v89 )
                goto LABEL_71;
              v141 = v118;
            }
          }
        }
        v121 = (unsigned int)v254;
        if ( (unsigned int)v254 >= HIDWORD(v254) )
        {
          sub_16CD150((__int64)&v253, v255, 0, 8, v118, v90);
          v121 = (unsigned int)v254;
        }
        v253[v121] = (__int64)v119;
        LODWORD(v254) = v254 + 1;
        v117 = *(_QWORD *)(v117 + 8);
        if ( !v117 )
          break;
        v109 = *(_QWORD *)(a1 + 160);
      }
    }
    if ( *(_BYTE *)(v101 + 16) == 35 && v98 == *(_QWORD **)sub_13CF970(v101) )
    {
      v203 = (_QWORD *)v101;
      v204 = 1;
    }
    else
    {
      v249[0] = sub_145DC80(v198, (__int64)v98);
      v248[0] = v249;
      v249[1] = v218;
      v248[1] = (__int64 *)0x200000002LL;
      v142 = sub_147DD40(v198, (__int64 *)v248, 0, 0, a7, a8);
      if ( v248[0] != v249 )
        _libc_free((unsigned __int64)v248[0]);
      v143 = v101;
      if ( *(_BYTE *)(v101 + 16) == 77 )
        v143 = sub_157ED20(*(_QWORD *)(v101 + 40));
      v203 = (_QWORD *)sub_38767A0(a15, v142, 0, v143);
      v204 = *(_QWORD *)v101 == *v203;
    }
    v144 = (unsigned __int64)v253;
    v244 = 0;
    v245 = 0;
    v246 = 0;
    v247 = 0;
    v208 = &v253[(unsigned int)v254];
    if ( v253 == v208 )
      goto LABEL_182;
    while ( 2 )
    {
      while ( 1 )
      {
        v145 = *(_QWORD *)v144;
        v146 = *(_QWORD *)(*(_QWORD *)v144 + 40LL);
        v224 = v146;
        if ( *(_BYTE *)(v145 + 16) == 77 )
          break;
        v164 = v245 + 16LL * v247;
        sub_1CD1D30(v248, (__int64 *)&v244, v146);
        if ( v249[0] == v164 )
        {
          v165 = v203;
          if ( !sub_14560B0(v218) )
          {
            v165 = (_QWORD *)sub_15F4880((__int64)v203);
            sub_15F2120((__int64)v165, v145);
            v248[0] = (__int64 *)"nloNewAdd";
            LOWORD(v249[0]) = 259;
            sub_164B780((__int64)v165, (__int64 *)v248);
          }
          if ( !v204 )
          {
            v206 = *(_QWORD *)v101;
            v248[0] = (__int64 *)"nloNewBit";
            LOWORD(v249[0]) = 259;
            v187 = sub_1648A60(56, 1u);
            if ( v187 )
            {
              v216 = v187;
              sub_15FD590((__int64)v187, (__int64)v165, v206, (__int64)v248, v145);
              v187 = v216;
            }
            v165 = v187;
          }
          sub_1CD3CC0((__int64)&v244, &v224)[1] = (__int64)v165;
        }
        else
        {
          v165 = (_QWORD *)sub_1CD3CC0((__int64)&v244, &v224)[1];
          if ( sub_15CCEE0(a6, v145, (__int64)v165) )
          {
            if ( !v204 && !sub_14560B0(v218) )
              sub_15F22F0((_QWORD *)*(v165 - 3), v145);
            sub_15F22F0(v165, v145);
          }
        }
        v144 += 8LL;
        sub_1648780(v145, v101, (__int64)v165);
        if ( v208 == (__int64 *)v144 )
          goto LABEL_182;
      }
      v147 = *(_DWORD *)(v145 + 20);
      v148 = 0;
      v149 = v147 & 0xFFFFFFF;
      if ( (v147 & 0xFFFFFFF) == 0 )
        goto LABEL_181;
      v205 = v144;
      v150 = (__int64 *)v101;
      while ( 2 )
      {
        if ( (*(_BYTE *)(v145 + 23) & 0x40) != 0 )
          v151 = *(_QWORD *)(v145 - 8);
        else
          v151 = v145 - 24LL * v149;
        v152 = 24LL * v148;
        v153 = *(__int64 **)(v151 + v152);
        if ( v153 && v150 == v153 )
        {
          v154 = *(_QWORD *)(v151 + 24LL * *(unsigned int *)(v145 + 56) + 8LL * v148 + 8);
          v224 = v154;
          if ( v247 )
          {
            v155 = (v247 - 1) & (((unsigned int)v154 >> 9) ^ ((unsigned int)v154 >> 4));
            v156 = (__int64 *)(v245 + 16LL * v155);
            v157 = *v156;
            if ( v154 == *v156 )
            {
LABEL_185:
              if ( v156 != (__int64 *)(v245 + 16LL * v247) )
              {
                v158 = sub_1CD3CC0((__int64)&v244, &v224)[1];
LABEL_187:
                if ( (*(_BYTE *)(v145 + 23) & 0x40) != 0 )
                  v159 = *(_QWORD *)(v145 - 8);
                else
                  v159 = v145 - 24LL * (*(_DWORD *)(v145 + 20) & 0xFFFFFFF);
                v160 = (__int64 *)(v159 + v152);
                if ( *v160 )
                {
                  v161 = v160[1];
                  v162 = v160[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v162 = v161;
                  if ( v161 )
                    *(_QWORD *)(v161 + 16) = *(_QWORD *)(v161 + 16) & 3LL | v162;
                }
                *v160 = v158;
                if ( v158 )
                {
                  v163 = *(_QWORD *)(v158 + 8);
                  v160[1] = v163;
                  if ( v163 )
                    *(_QWORD *)(v163 + 16) = (unsigned __int64)(v160 + 1) | *(_QWORD *)(v163 + 16) & 3LL;
                  v160[2] = (v158 + 8) | v160[2] & 3;
                  *(_QWORD *)(v158 + 8) = v160;
                }
                v147 = *(_DWORD *)(v145 + 20);
                goto LABEL_177;
              }
            }
            else
            {
              v166 = 1;
              while ( v157 != -8 )
              {
                v192 = v166 + 1;
                v155 = (v247 - 1) & (v166 + v155);
                v156 = (__int64 *)(v245 + 16LL * v155);
                v157 = *v156;
                if ( v154 == *v156 )
                  goto LABEL_185;
                v166 = v192;
              }
            }
          }
          v167 = sub_14560B0(v218);
          v168 = v203;
          if ( !v167 )
          {
            v201 = sub_15F4880((__int64)v203);
            v185 = sub_157EBA0(v224);
            sub_15F2120(v201, v185);
            v248[0] = (__int64 *)"nloNewAdd";
            LOWORD(v249[0]) = 259;
            sub_164B780(v201, (__int64 *)v248);
            v168 = (_QWORD *)v201;
          }
          if ( !v204 )
          {
            v196 = (__int64)v168;
            v197 = *v150;
            v248[0] = (__int64 *)"nloNewBit";
            LOWORD(v249[0]) = 259;
            v182 = sub_193FF80(v145);
            v200 = sub_157EBA0(*(_QWORD *)(v182 + 8LL * v148));
            v183 = sub_1648A60(56, 1u);
            v184 = v183;
            if ( v183 )
              sub_15FD590((__int64)v183, v196, v197, (__int64)v248, v200);
            v168 = v184;
          }
          v199 = (__int64)v168;
          v169 = sub_1CD3CC0((__int64)&v244, &v224);
          v158 = v199;
          v169[1] = v199;
          goto LABEL_187;
        }
LABEL_177:
        ++v148;
        v149 = v147 & 0xFFFFFFF;
        if ( v148 != (v147 & 0xFFFFFFF) )
          continue;
        break;
      }
      v101 = (__int64)v150;
      v144 = v205;
LABEL_181:
      v144 += 8LL;
      if ( v208 != (__int64 *)v144 )
        continue;
      break;
    }
LABEL_182:
    sub_15F20C0((_QWORD *)v101);
    j___libc_free_0(v245);
LABEL_73:
    if ( v253 != (__int64 *)v255 )
      _libc_free((unsigned __int64)v253);
LABEL_75:
    v221 += 8LL;
    v81 = v225;
    v82 = v241;
    if ( v221 != v212 )
      continue;
    break;
  }
  v84 = v226 - v225;
LABEL_142:
  j___libc_free_0(v82);
  if ( v225 )
    j_j___libc_free_0(v225, (char *)v227 - (char *)v225);
  j___libc_free_0(v237);
  if ( v250 != (__int64 *)v252 )
    _libc_free((unsigned __int64)v250);
  j___libc_free_0(v233);
  if ( v231 )
  {
    v137 = v229;
    v138 = &v229[7 * v231];
    do
    {
      if ( *v137 != -8 && *v137 != -16 )
      {
        v139 = v137[1];
        if ( (_QWORD *)v139 != v137 + 3 )
          _libc_free(v139);
      }
      v137 += 7;
    }
    while ( v138 != v137 );
  }
  j___libc_free_0(v229);
  return (unsigned int)v84;
}
