// Function: sub_1B83220
// Address: 0x1b83220
//
__int64 __fastcall sub_1B83220(
        __int64 *a1,
        __int64 *a2,
        unsigned __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v12; // r12
  __int64 v13; // r9
  __int64 *v14; // r8
  __int64 *v16; // r14
  __int64 **v18; // rbx
  __int64 **v19; // rdx
  char v20; // al
  _BYTE *v21; // rsi
  __int64 *v22; // rdx
  char v23; // di
  __int64 **i; // rax
  char v25; // dl
  char v26; // dl
  unsigned int v27; // r10d
  __int64 v29; // rcx
  unsigned int v30; // eax
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned int v33; // eax
  __int64 *v34; // r8
  __int64 v35; // r9
  __int64 *v36; // r9
  __int64 *v37; // r10
  __int64 v38; // rsi
  __int64 *v39; // rdi
  unsigned int v40; // r8d
  __int64 *v41; // rax
  __int64 *v42; // rcx
  unsigned int v43; // eax
  unsigned __int64 v44; // rax
  __int64 *v45; // r8
  unsigned __int8 v46; // r10
  unsigned __int64 v47; // r11
  __int64 *v48; // r14
  unsigned __int8 v49; // al
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 *v52; // rsi
  unsigned int v53; // ebx
  __int64 *v54; // rax
  unsigned int v55; // r11d
  __int64 v56; // r9
  unsigned int v57; // eax
  __int64 v58; // r9
  unsigned __int64 v59; // rdx
  __int64 *v60; // rax
  __int64 *v61; // rdi
  unsigned int v62; // r12d
  unsigned int v63; // r11d
  __int64 v64; // r12
  __int64 *v65; // rsi
  int v66; // ebx
  unsigned __int64 v67; // rdx
  __int64 *v68; // r10
  __int64 *v69; // r9
  __int64 v70; // rsi
  __int64 *v71; // rdi
  unsigned int v72; // r8d
  __int64 *v73; // rax
  __int64 *v74; // rcx
  unsigned int v75; // ebx
  unsigned int v76; // r12d
  __int64 v77; // rax
  __int64 *v78; // r15
  unsigned int v79; // eax
  unsigned int v80; // edi
  unsigned __int64 v81; // rcx
  unsigned __int64 v82; // rax
  unsigned __int64 v83; // rdx
  unsigned int v84; // r14d
  __int64 v85; // r15
  char v86; // al
  __int64 *v87; // r8
  unsigned int v88; // eax
  unsigned __int64 v89; // rdi
  __int64 v90; // rax
  __int64 *v91; // rdi
  char v92; // r10
  __int64 v93; // rax
  __int64 v94; // rdx
  unsigned __int8 v95; // r10
  __int64 v96; // rsi
  unsigned __int8 *v97; // rsi
  __int64 v98; // rax
  __int64 v99; // rcx
  __int64 v100; // rax
  __int64 v101; // rbx
  char v102; // al
  __int64 v103; // rax
  __int64 v104; // rax
  int v105; // r9d
  int v106; // r8d
  unsigned __int8 v107; // r10
  __int64 v108; // rcx
  __int64 v109; // r8
  int v110; // r9d
  unsigned __int8 v111; // r10
  __int64 j; // r14
  __int64 *v113; // rdx
  __int64 v114; // rax
  _QWORD *v115; // r12
  const char *v116; // rax
  _QWORD *v117; // rdi
  __int64 v118; // rdx
  __int64 v119; // rax
  __int64 v120; // rax
  __int64 v121; // r15
  _QWORD *v122; // r12
  double v123; // xmm4_8
  double v124; // xmm5_8
  __int64 v125; // rax
  __int64 *v126; // rax
  __int64 *v127; // rbx
  _QWORD *v128; // rax
  __int64 v129; // rax
  unsigned __int64 *v130; // r15
  __int64 v131; // rax
  unsigned __int64 v132; // rsi
  __int64 v133; // rsi
  __int64 v134; // rsi
  __int64 v135; // rdx
  unsigned __int8 *v136; // rsi
  unsigned __int8 v137; // r10
  __int64 *v138; // rdi
  unsigned __int8 v139; // r15
  __int64 *v140; // r12
  __int64 *v141; // rbx
  _QWORD *v142; // rdi
  unsigned int v143; // ebx
  unsigned int v144; // eax
  __int64 v145; // rdx
  __int64 v146; // rax
  int v147; // ebx
  __int64 v148; // rdi
  __int64 *v149; // rax
  __int64 v150; // r10
  unsigned int v151; // eax
  __int64 v152; // rax
  __int64 v153; // rax
  unsigned int v154; // esi
  __int64 v155; // rax
  __int64 *v156; // r14
  const char *v157; // rax
  _QWORD *v158; // rdi
  __int64 v159; // rdx
  __int64 v160; // rax
  __int64 v161; // rax
  __int64 *v162; // r12
  unsigned int v163; // ebx
  _QWORD *v164; // rax
  __int64 v165; // rax
  __int64 k; // rbx
  _QWORD *v167; // r14
  double v168; // xmm4_8
  double v169; // xmm5_8
  __int64 *v170; // r12
  _QWORD *v171; // rdi
  __int64 v172; // rdx
  __int64 v173; // rax
  __int64 v174; // rax
  _QWORD *v175; // rax
  __int64 v176; // rax
  unsigned __int64 v177; // rsi
  __int64 v178; // rax
  __int64 v179; // rsi
  __int64 v180; // rsi
  __int64 v181; // rdx
  unsigned __int8 *v182; // rsi
  __int64 m; // rbx
  __int64 *v184; // r12
  const char *v185; // rax
  _QWORD *v186; // rdi
  __int64 v187; // rdx
  __int64 v188; // rax
  __int64 v189; // rax
  _QWORD *v190; // rsi
  double v191; // xmm4_8
  double v192; // xmm5_8
  __int64 **v193; // rdi
  __int64 v194; // rax
  __int64 v195; // r8
  __int64 v196; // rax
  _QWORD *v197; // rax
  __int64 v198; // [rsp+0h] [rbp-1B0h]
  unsigned __int64 *v199; // [rsp+0h] [rbp-1B0h]
  unsigned int v200; // [rsp+18h] [rbp-198h]
  __int64 v201; // [rsp+20h] [rbp-190h]
  unsigned __int8 v202; // [rsp+20h] [rbp-190h]
  __int64 v203; // [rsp+20h] [rbp-190h]
  unsigned int v204; // [rsp+28h] [rbp-188h]
  __int64 v205; // [rsp+28h] [rbp-188h]
  unsigned __int8 v206; // [rsp+28h] [rbp-188h]
  __int64 *v207; // [rsp+28h] [rbp-188h]
  __int64 v208; // [rsp+30h] [rbp-180h]
  __int64 v209; // [rsp+30h] [rbp-180h]
  __int64 *v210; // [rsp+30h] [rbp-180h]
  __int64 *v211; // [rsp+38h] [rbp-178h]
  __int64 v212; // [rsp+38h] [rbp-178h]
  unsigned int v213; // [rsp+40h] [rbp-170h]
  __int64 *v214; // [rsp+40h] [rbp-170h]
  unsigned int v215; // [rsp+4Ch] [rbp-164h]
  int v216; // [rsp+4Ch] [rbp-164h]
  int v217; // [rsp+4Ch] [rbp-164h]
  unsigned int v218; // [rsp+4Ch] [rbp-164h]
  unsigned int v219; // [rsp+50h] [rbp-160h]
  unsigned int v220; // [rsp+50h] [rbp-160h]
  char v221; // [rsp+50h] [rbp-160h]
  __int64 v222; // [rsp+50h] [rbp-160h]
  unsigned int v223; // [rsp+50h] [rbp-160h]
  unsigned int v224; // [rsp+58h] [rbp-158h]
  unsigned int v225; // [rsp+58h] [rbp-158h]
  __int64 *v226; // [rsp+58h] [rbp-158h]
  unsigned __int8 v227; // [rsp+58h] [rbp-158h]
  __int64 *v228; // [rsp+58h] [rbp-158h]
  __int64 v229; // [rsp+58h] [rbp-158h]
  unsigned int v230; // [rsp+60h] [rbp-150h]
  unsigned __int8 v231; // [rsp+60h] [rbp-150h]
  unsigned __int8 v232; // [rsp+60h] [rbp-150h]
  unsigned __int8 v233; // [rsp+60h] [rbp-150h]
  __int64 v234; // [rsp+60h] [rbp-150h]
  int v235; // [rsp+68h] [rbp-148h]
  unsigned __int8 v236; // [rsp+68h] [rbp-148h]
  __int64 *v237; // [rsp+68h] [rbp-148h]
  unsigned __int8 v238; // [rsp+68h] [rbp-148h]
  __int64 v239; // [rsp+70h] [rbp-140h]
  __int64 v240; // [rsp+70h] [rbp-140h]
  __int64 v241; // [rsp+70h] [rbp-140h]
  int v242; // [rsp+70h] [rbp-140h]
  unsigned __int64 v243; // [rsp+70h] [rbp-140h]
  __int64 *v244; // [rsp+70h] [rbp-140h]
  __int64 *v245; // [rsp+70h] [rbp-140h]
  __int64 v246; // [rsp+70h] [rbp-140h]
  __int64 v247; // [rsp+70h] [rbp-140h]
  __int64 v248; // [rsp+78h] [rbp-138h]
  unsigned int v249; // [rsp+78h] [rbp-138h]
  unsigned __int64 v250; // [rsp+78h] [rbp-138h]
  __int64 *v251; // [rsp+78h] [rbp-138h]
  unsigned int v252; // [rsp+78h] [rbp-138h]
  unsigned int v253; // [rsp+78h] [rbp-138h]
  __int64 *v254; // [rsp+78h] [rbp-138h]
  __int64 *v255; // [rsp+80h] [rbp-130h]
  __int64 *v256; // [rsp+80h] [rbp-130h]
  unsigned int v257; // [rsp+80h] [rbp-130h]
  __int64 *v258; // [rsp+80h] [rbp-130h]
  unsigned __int8 v259; // [rsp+80h] [rbp-130h]
  unsigned __int8 v260; // [rsp+80h] [rbp-130h]
  unsigned __int8 v262; // [rsp+88h] [rbp-128h]
  unsigned __int8 *v263; // [rsp+98h] [rbp-118h] BYREF
  _QWORD v264[2]; // [rsp+A0h] [rbp-110h] BYREF
  const char *v265; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v266; // [rsp+B8h] [rbp-F8h]
  __int16 v267; // [rsp+C0h] [rbp-F0h]
  __int64 *v268; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v269; // [rsp+D8h] [rbp-D8h]
  __int64 v270; // [rsp+E0h] [rbp-D0h]
  __int64 *v271; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v272; // [rsp+F8h] [rbp-B8h]
  __int64 *v273; // [rsp+100h] [rbp-B0h] BYREF
  unsigned __int64 v274; // [rsp+108h] [rbp-A8h]

  v13 = 8 * a3;
  v14 = a2;
  v16 = a2;
  v18 = (__int64 **)&a2[a3];
  if ( a2 == (__int64 *)v18 )
  {
    v22 = (__int64 *)*a2;
    if ( *(_BYTE *)(*a2 + 16) == 78 )
    {
      v32 = *(v22 - 3);
      if ( !*(_BYTE *)(v32 + 16) && (*(_BYTE *)(v32 + 33) & 0x20) != 0 )
      {
LABEL_57:
        v23 = *(_DWORD *)(v32 + 36) == 4085 || *(_DWORD *)(v32 + 36) == 4057;
        if ( v14 != (__int64 *)v18 )
          goto LABEL_58;
      }
    }
    v240 = v13;
    v256 = v14;
    v224 = sub_127FA20(a1[5], v12);
    v235 = sub_1B7C940(*v256);
    v219 = sub_14A38B0(a1[4]);
    v43 = sub_1B7C680((__int64)a1, *v256);
    v34 = v256;
    v230 = v43;
    v35 = v240;
    if ( !v224 )
      return 0;
  }
  else
  {
    v19 = (__int64 **)a2;
    while ( 1 )
    {
      v12 = **v19;
      v20 = *(_BYTE *)(v12 + 8);
      if ( v20 == 16 )
        v20 = *(_BYTE *)(**(_QWORD **)(v12 + 16) + 8LL);
      if ( v20 == 11 )
      {
LABEL_8:
        v21 = (_BYTE *)*a2;
        v22 = (__int64 *)*v14;
        if ( *(_BYTE *)(*v14 + 16) != 78 )
          goto LABEL_9;
LABEL_24:
        v32 = *((_QWORD *)v21 - 3);
        if ( *(_BYTE *)(v32 + 16) || (*(_BYTE *)(v32 + 33) & 0x20) == 0 )
        {
          v23 = 0;
LABEL_58:
          v21 = v22;
          goto LABEL_10;
        }
        goto LABEL_57;
      }
      if ( v20 == 15 )
        break;
      if ( v18 == ++v19 )
        goto LABEL_8;
    }
    v248 = v13;
    v30 = sub_127FA20(a1[5], **v19);
    v31 = sub_1644C60(**(_QWORD ***)(*a1 + 40), v30);
    v14 = a2;
    v13 = v248;
    v12 = v31;
    v21 = (_BYTE *)*a2;
    v22 = (__int64 *)v21;
    if ( *(_BYTE *)(*v14 + 16) == 78 )
      goto LABEL_24;
LABEL_9:
    v23 = 0;
LABEL_10:
    for ( i = (__int64 **)(v14 + 1); ; ++i )
    {
      v26 = *(_BYTE *)(*(_QWORD *)v21 + 8LL);
      if ( *(_BYTE *)(v12 + 8) == 16 )
      {
        if ( v26 != 16 )
          return 0;
      }
      else if ( v26 == 16 )
      {
        return 0;
      }
      v25 = 0;
      if ( v21[16] == 78 )
      {
        v29 = *((_QWORD *)v21 - 3);
        if ( !*(_BYTE *)(v29 + 16) && (*(_BYTE *)(v29 + 33) & 0x20) != 0 )
          v25 = *(_DWORD *)(v29 + 36) == 4057 || *(_DWORD *)(v29 + 36) == 4085;
      }
      if ( v25 != v23 )
        return 0;
      if ( v18 == i )
        break;
      v21 = *i;
    }
    v239 = v13;
    v255 = v14;
    v224 = sub_127FA20(a1[5], v12);
    v235 = sub_1B7C940(*v255);
    v219 = sub_14A38B0(a1[4]);
    v33 = sub_1B7C680((__int64)a1, *v255);
    v34 = v255;
    v230 = v33;
    v35 = v239;
    if ( !v224 )
    {
LABEL_28:
      if ( v34 == (__int64 *)v18 )
        return 0;
      v36 = *(__int64 **)(a4 + 16);
      v37 = *(__int64 **)(a4 + 8);
      while ( 2 )
      {
        v38 = *v16;
        if ( v36 == v37 )
        {
          v39 = &v36[*(unsigned int *)(a4 + 28)];
          v40 = *(_DWORD *)(a4 + 28);
          if ( v39 != v36 )
          {
            v41 = v36;
            v42 = 0;
            while ( v38 != *v41 )
            {
              if ( *v41 == -2 )
                v42 = v41;
              if ( v39 == ++v41 )
              {
                if ( !v42 )
                  goto LABEL_53;
                *v42 = v38;
                v36 = *(__int64 **)(a4 + 16);
                --*(_DWORD *)(a4 + 32);
                v37 = *(__int64 **)(a4 + 8);
                ++*(_QWORD *)a4;
                break;
              }
            }
LABEL_31:
            if ( v18 == (__int64 **)++v16 )
              return 0;
            continue;
          }
LABEL_53:
          if ( v40 < *(_DWORD *)(a4 + 24) )
          {
            *(_DWORD *)(a4 + 28) = v40 + 1;
            *v39 = v38;
            v37 = *(__int64 **)(a4 + 8);
            ++*(_QWORD *)a4;
            v36 = *(__int64 **)(a4 + 16);
            goto LABEL_31;
          }
        }
        break;
      }
      sub_16CCBA0(a4, v38);
      v36 = *(__int64 **)(a4 + 16);
      v37 = *(__int64 **)(a4 + 8);
      goto LABEL_31;
    }
  }
  v241 = v35;
  v257 = v224 & (v224 - 1);
  if ( v257 )
    goto LABEL_28;
  v249 = v219 / v224;
  if ( v219 / v224 <= 1 || (unsigned int)a3 <= 1 )
    goto LABEL_28;
  v211 = v34;
  v44 = sub_1B80360((__int64)a1, v34, a3);
  v45 = v211;
  v46 = 0;
  v220 = v44;
  v47 = HIDWORD(v44);
  v213 = HIDWORD(v44);
  if ( !HIDWORD(v44) )
  {
    if ( v211 == (__int64 *)v18 )
      return 0;
    v68 = *(__int64 **)(a4 + 16);
    v69 = *(__int64 **)(a4 + 8);
    while ( 1 )
    {
      v70 = *v16;
      if ( v69 != v68 )
        goto LABEL_82;
      v71 = &v69[*(unsigned int *)(a4 + 28)];
      v72 = *(_DWORD *)(a4 + 28);
      if ( v69 != v71 )
      {
        v73 = v69;
        v74 = 0;
        while ( v70 != *v73 )
        {
          if ( *v73 == -2 )
            v74 = v73;
          if ( v71 == ++v73 )
          {
            if ( !v74 )
              goto LABEL_93;
            *v74 = v70;
            v68 = *(__int64 **)(a4 + 16);
            --*(_DWORD *)(a4 + 32);
            v69 = *(__int64 **)(a4 + 8);
            ++*(_QWORD *)a4;
            goto LABEL_83;
          }
        }
        goto LABEL_83;
      }
LABEL_93:
      if ( v72 < *(_DWORD *)(a4 + 24) )
      {
        *(_DWORD *)(a4 + 28) = v72 + 1;
        *v71 = v70;
        v69 = *(__int64 **)(a4 + 8);
        ++*(_QWORD *)a4;
        v68 = *(__int64 **)(a4 + 16);
      }
      else
      {
LABEL_82:
        sub_16CCBA0(a4, v70);
        v68 = *(__int64 **)(a4 + 16);
        v69 = *(__int64 **)(a4 + 8);
      }
LABEL_83:
      if ( v18 == (__int64 **)++v16 )
        return 0;
    }
  }
  v48 = &v211[(unsigned int)v44];
  if ( HIDWORD(v44) != 1 )
  {
    v212 = HIDWORD(v44);
    if ( v47 != a3 )
    {
      if ( (unsigned int)v44 > 1 )
      {
        v250 = HIDWORD(v44);
        v258 = v45;
        v49 = sub_1B83220(a1, v45, (unsigned int)v44, a4);
        LODWORD(v47) = v250;
        v45 = v258;
        v46 = v49;
      }
      v242 = v47;
      v251 = v45;
      v27 = sub_1B83220(a1, v48, v212, a4) | v46;
      v50 = v242 + v220;
      v51 = a3 - v50;
      if ( a3 - v50 > 1 )
      {
        v262 = v27;
        v52 = &v251[v50];
        return (unsigned int)sub_1B83220(a1, v52, v51, a4) | v262;
      }
      return v27;
    }
    v208 = v241;
    v243 = HIDWORD(v44);
    v215 = v224 >> 3;
    v53 = HIDWORD(v44) * (v224 >> 3);
    v221 = sub_14A3940(a1[4]);
    if ( !v221 )
    {
      v143 = v53 & 0xFFFFFFFC;
      v144 = v143 / v215;
      v145 = v143 / v215;
      if ( v145 == a3 )
      {
        if ( (v144 & 1) != 0 )
          v145 = v144 - 1;
        else
          v145 = v144 >> 1;
        v146 = v145;
      }
      else if ( v215 > v143 )
      {
        v146 = 1;
        v145 = 1;
      }
      else
      {
        v146 = v145;
      }
      v273 = &v48[v146];
      v272 = v145;
      v271 = v48;
      v274 = a3 - v145;
      v147 = sub_1B83220(a1, v48, v145, a4);
      return (unsigned int)sub_1B83220(a1, v273, v274, a4) | v147;
    }
    v201 = v208;
    v204 = v243;
    if ( *(_BYTE *)(v12 + 8) == 16 )
    {
      v54 = sub_16463B0(**(__int64 ***)(v12 + 16), *(_DWORD *)(v12 + 32) * (int)v243);
      v209 = v12;
      v55 = v243;
      v244 = v54;
      v56 = v201;
    }
    else
    {
      v209 = 0;
      v244 = sub_16463B0((__int64 *)v12, v243);
      v56 = v201;
      v55 = v204;
    }
    v200 = v55;
    v205 = v56;
    v57 = sub_14A39A0((__int64 *)a1[4], v249, v224, v53, (__int64)v244);
    v225 = v57;
    v58 = v205;
    if ( v249 > v200 && (v200 & 1) != 0 && *(_BYTE *)(v12 + 8) != 16 )
    {
      v59 = ((((((a3 - 1) | ((a3 - 1) >> 1)) >> 2) | (a3 - 1) | ((a3 - 1) >> 1)) >> 4)
           | (((a3 - 1) | ((a3 - 1) >> 1)) >> 2)
           | (a3 - 1)
           | ((a3 - 1) >> 1)) >> 8;
      v252 = (((v59
              | (((((a3 - 1) | ((a3 - 1) >> 1)) >> 2) | (a3 - 1) | ((a3 - 1) >> 1)) >> 4)
              | (((a3 - 1) | ((a3 - 1) >> 1)) >> 2)
              | (a3 - 1)
              | ((a3 - 1) >> 1)) >> 16)
            | v59
            | (((((a3 - 1) | ((a3 - 1) >> 1)) >> 2) | (a3 - 1) | ((a3 - 1) >> 1)) >> 4)
            | (((a3 - 1) | ((a3 - 1) >> 1)) >> 2)
            | (a3 - 1)
            | ((a3 - 1) >> 1))
           + 1;
      v53 = v252 * v215;
      v60 = sub_16463B0((__int64 *)v12, v252);
      v58 = v205;
      v244 = v60;
LABEL_66:
      v226 = (__int64 *)((char *)v48 + v58);
      if ( !(v230 % v53) )
      {
        v91 = (__int64 *)sub_1B7F9F0(a1, v12, v252);
        if ( v91 )
        {
          v244 = sub_16463B0(v91, 4u);
          sub_1B811B0(a4, v48, v226);
          v92 = v221;
        }
        else
        {
          sub_1B811B0(a4, v48, v226);
          v92 = 0;
        }
        v221 = v92;
        v62 = v230;
LABEL_127:
        v231 = sub_1B7CAC0(v48, a3);
        if ( v231 )
        {
          v93 = sub_1B7D6B0(v48, a3);
          v254 = a1 + 6;
          if ( !v93 )
            BUG();
          v94 = *(_QWORD *)(v93 + 16);
          v95 = v231;
          a1[8] = v93;
          a1[7] = v94;
          v271 = *(__int64 **)(v93 + 24);
          if ( v271 )
          {
            sub_1623A60((__int64)&v271, (__int64)v271, 2);
            v96 = a1[6];
            v95 = v231;
            if ( !v96 )
              goto LABEL_132;
          }
          else
          {
            v96 = a1[6];
            if ( !v96 )
              goto LABEL_134;
          }
          v232 = v95;
          sub_161E7C0((__int64)v254, v96);
          v95 = v232;
LABEL_132:
          v97 = (unsigned __int8 *)v271;
          a1[6] = (__int64)v271;
          if ( v97 )
          {
            v233 = v95;
            sub_1623210((__int64)&v271, v97, (__int64)v254);
            v95 = v233;
          }
LABEL_134:
          v227 = v95;
          LOWORD(v273) = 257;
          sub_1647190(v244, v235);
          v98 = sub_1B7CA20(*v48);
          v100 = sub_12AA3B0(v254, 0x2Fu, v98, v99, (__int64)&v271);
          v101 = *v48;
          v234 = v100;
          v102 = *(_BYTE *)(*v48 + 16);
          if ( v102 == 54 )
          {
            LOWORD(v273) = 257;
            v197 = sub_156E5B0(v254, v234, (__int64)&v271);
            v107 = v227;
            v246 = (__int64)v197;
            goto LABEL_139;
          }
          v206 = v227;
          if ( v102 != 78 )
            BUG();
          v103 = *(_QWORD *)(v101 - 24);
          if ( *(_BYTE *)(v103 + 16) )
            BUG();
          v216 = *(_DWORD *)(v103 + 36);
          v228 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v101 + 40) + 56LL) + 40LL);
          v265 = (const char *)v244;
          v266 = sub_1647190(v244, v235);
          v104 = sub_15E26F0(v228, v216, (__int64 *)&v265, 2);
          v106 = v216;
          v107 = v206;
          if ( v216 == 4085 )
          {
            LOWORD(v273) = 257;
            v238 = v206;
            v268 = *(__int64 **)(v101 - 24LL * (*(_DWORD *)(v101 + 20) & 0xFFFFFFF));
            v269 = v234;
            v155 = sub_1285290(v254, *(_QWORD *)(*(_QWORD *)v104 + 24LL), v104, (int)&v268, 2, (__int64)&v271, 0);
          }
          else
          {
            v246 = 0;
            if ( v216 != 4057 )
            {
LABEL_139:
              v236 = v107;
              sub_1B7F470(v246, (__int64 **)v48, a3, a1[5], v106, v105);
              sub_1B7DE50(v246, v62);
              v111 = v236;
              if ( v209 )
              {
                v237 = v48;
                v271 = (__int64 *)&v273;
                v207 = v48;
                v272 = 0x1000000000LL;
                v202 = v111;
                v217 = *(_DWORD *)(v209 + 32);
                v214 = &v48[v212];
                do
                {
                  for ( j = *(_QWORD *)(*v237 + 8); j; j = *(_QWORD *)(j + 8) )
                  {
                    v126 = sub_1648700(j);
                    v127 = v126;
                    if ( (*((_BYTE *)v126 + 23) & 0x40) != 0 )
                      v113 = (__int64 *)*(v126 - 1);
                    else
                      v113 = &v126[-3 * (*((_DWORD *)v126 + 5) & 0xFFFFFFF)];
                    v114 = v113[3];
                    v115 = *(_QWORD **)(v114 + 24);
                    if ( *(_DWORD *)(v114 + 32) > 0x40u )
                      v115 = (_QWORD *)*v115;
                    v116 = sub_1649960((__int64)v127);
                    v117 = (_QWORD *)a1[9];
                    v267 = 261;
                    v264[0] = v116;
                    v264[1] = v118;
                    v265 = (const char *)v264;
                    v119 = sub_1643350(v117);
                    v120 = sub_159C470(v119, v257 + (unsigned int)v115, 0);
                    v121 = v120;
                    if ( *(_BYTE *)(v246 + 16) > 0x10u || *(_BYTE *)(v120 + 16) > 0x10u )
                    {
                      LOWORD(v270) = 257;
                      v128 = sub_1648A60(56, 2u);
                      v122 = v128;
                      if ( v128 )
                        sub_15FA320((__int64)v128, (_QWORD *)v246, v121, (__int64)&v268, 0);
                      v129 = a1[7];
                      if ( v129 )
                      {
                        v130 = (unsigned __int64 *)a1[8];
                        sub_157E9D0(v129 + 40, (__int64)v122);
                        v131 = v122[3];
                        v132 = *v130;
                        v122[4] = v130;
                        v132 &= 0xFFFFFFFFFFFFFFF8LL;
                        v122[3] = v132 | v131 & 7;
                        *(_QWORD *)(v132 + 8) = v122 + 3;
                        *v130 = *v130 & 7 | (unsigned __int64)(v122 + 3);
                      }
                      sub_164B780((__int64)v122, (__int64 *)&v265);
                      v133 = a1[6];
                      if ( v133 )
                      {
                        v263 = (unsigned __int8 *)a1[6];
                        sub_1623A60((__int64)&v263, v133, 2);
                        v134 = v122[6];
                        v135 = (__int64)(v122 + 6);
                        if ( v134 )
                        {
                          sub_161E7C0((__int64)(v122 + 6), v134);
                          v135 = (__int64)(v122 + 6);
                        }
                        v136 = v263;
                        v122[6] = v263;
                        if ( v136 )
                          sub_1623210((__int64)&v263, v136, v135);
                      }
                    }
                    else
                    {
                      v122 = (_QWORD *)sub_15A37D0((_BYTE *)v246, v120, 0);
                    }
                    if ( *v127 != *v122 )
                    {
                      LOWORD(v270) = 257;
                      v122 = (_QWORD *)sub_12AA3B0(v254, 0x2Fu, (__int64)v122, *v127, (__int64)&v268);
                    }
                    sub_164D160((__int64)v127, (__int64)v122, a5, a6, a7, a8, v123, v124, a11, a12);
                    v125 = (unsigned int)v272;
                    if ( (unsigned int)v272 >= HIDWORD(v272) )
                    {
                      sub_16CD150((__int64)&v271, &v273, 0, 8, v109, v110);
                      v125 = (unsigned int)v272;
                    }
                    v271[v125] = (__int64)v127;
                    LODWORD(v272) = v272 + 1;
                  }
                  ++v237;
                  v257 += v217;
                }
                while ( v237 != v214 );
                v48 = v207;
                v137 = v202;
                if ( *(_BYTE *)(v234 + 16) > 0x17u )
                {
                  sub_1B80E80(v234);
                  v137 = v202;
                }
                v138 = v271;
                v139 = v137;
                v140 = &v271[(unsigned int)v272];
                v141 = v271;
                if ( v271 != v140 )
                {
                  do
                  {
                    v142 = (_QWORD *)*v141++;
                    sub_15F20C0(v142);
                  }
                  while ( v140 != v141 );
                  v138 = v271;
                  v137 = v139;
                }
                if ( v138 != (__int64 *)&v273 )
                {
                  v259 = v137;
                  _libc_free((unsigned __int64)v138);
                  v137 = v259;
                }
              }
              else if ( v221 )
              {
                v210 = v48;
                v223 = 0;
                do
                {
                  v156 = (__int64 *)v210[v257];
                  v157 = sub_1649960((__int64)v156);
                  v158 = (_QWORD *)a1[9];
                  v268 = (__int64 *)v157;
                  LOWORD(v273) = 261;
                  v269 = v159;
                  v271 = (__int64 *)&v268;
                  v160 = sub_1643350(v158);
                  v161 = sub_159C470(v160, v223, 0);
                  v162 = (__int64 *)sub_156D5F0(v254, v246, v161, (__int64)&v271);
                  v163 = sub_127FA20(a1[5], *v156);
                  v203 = sub_127FA20(a1[5], *v162);
                  v218 = (unsigned int)v203 / v163;
                  v164 = sub_16463B0((__int64 *)*v156, (unsigned int)v203 / v163);
                  LOWORD(v273) = 257;
                  v165 = sub_17FE280(v254, (__int64)v162, (__int64)v164, (__int64 *)&v271);
                  v109 = v203;
                  v229 = v165;
                  if ( v163 <= (unsigned int)v203 && v213 > v257 )
                  {
                    for ( k = 0; ; ++k )
                    {
                      v170 = (__int64 *)v210[v257 + k];
                      v265 = sub_1649960((__int64)v170);
                      LOWORD(v270) = 261;
                      v171 = (_QWORD *)a1[9];
                      v266 = v172;
                      v268 = (__int64 *)&v265;
                      v173 = sub_1643350(v171);
                      v174 = sub_159C470(v173, k, 0);
                      if ( *(_BYTE *)(v229 + 16) > 0x10u || *(_BYTE *)(v174 + 16) > 0x10u )
                      {
                        v198 = v174;
                        LOWORD(v273) = 257;
                        v175 = sub_1648A60(56, 2u);
                        v167 = v175;
                        if ( v175 )
                          sub_15FA320((__int64)v175, (_QWORD *)v229, v198, (__int64)&v271, 0);
                        v176 = a1[7];
                        if ( v176 )
                        {
                          v199 = (unsigned __int64 *)a1[8];
                          sub_157E9D0(v176 + 40, (__int64)v167);
                          v177 = *v199;
                          v178 = v167[3] & 7LL;
                          v167[4] = v199;
                          v177 &= 0xFFFFFFFFFFFFFFF8LL;
                          v167[3] = v177 | v178;
                          *(_QWORD *)(v177 + 8) = v167 + 3;
                          *v199 = *v199 & 7 | (unsigned __int64)(v167 + 3);
                        }
                        sub_164B780((__int64)v167, (__int64 *)&v268);
                        v179 = a1[6];
                        if ( v179 )
                        {
                          v264[0] = a1[6];
                          sub_1623A60((__int64)v264, v179, 2);
                          v180 = v167[6];
                          v181 = (__int64)(v167 + 6);
                          if ( v180 )
                          {
                            sub_161E7C0((__int64)(v167 + 6), v180);
                            v181 = (__int64)(v167 + 6);
                          }
                          v182 = (unsigned __int8 *)v264[0];
                          v167[6] = v264[0];
                          if ( v182 )
                            sub_1623210((__int64)v264, v182, v181);
                        }
                      }
                      else
                      {
                        v167 = (_QWORD *)sub_15A37D0((_BYTE *)v229, v174, 0);
                      }
                      if ( *v170 != *v167 )
                      {
                        LOWORD(v273) = 257;
                        v167 = (_QWORD *)sub_12AA3B0(v254, 0x2Fu, (__int64)v167, *v170, (__int64)&v271);
                      }
                      sub_164D160((__int64)v170, (__int64)v167, a5, a6, a7, a8, v168, v169, a11, a12);
                      if ( v218 <= (int)k + 1 || k == v213 - 1 - v257 )
                        break;
                    }
                  }
                  ++v223;
                  v257 += v218;
                }
                while ( v257 < v213 );
                v48 = v210;
                v137 = v236;
                if ( *(_BYTE *)(v234 + 16) > 0x17u )
                {
                  sub_1B80E80(v234);
                  v137 = v236;
                }
              }
              else
              {
                for ( m = 0; m != v212; ++m )
                {
                  v184 = (__int64 *)v48[m];
                  v185 = sub_1649960((__int64)v184);
                  v186 = (_QWORD *)a1[9];
                  v268 = (__int64 *)v185;
                  v269 = v187;
                  LOWORD(v273) = 261;
                  v271 = (__int64 *)&v268;
                  v188 = sub_1643350(v186);
                  v189 = sub_159C470(v188, m, 0);
                  v190 = (_QWORD *)sub_156D5F0(v254, v246, v189, (__int64)&v271);
                  if ( *v184 != *v190 )
                  {
                    LOWORD(v273) = 257;
                    v190 = (_QWORD *)sub_17FE280(v254, (__int64)v190, *v184, (__int64 *)&v271);
                  }
                  sub_164D160((__int64)v184, (__int64)v190, a5, a6, a7, a8, v191, v192, a11, a12);
                }
                v137 = v236;
                if ( *(_BYTE *)(v234 + 16) > 0x17u )
                {
                  sub_1B80E80(v234);
                  v137 = v236;
                }
              }
              v260 = v137;
              sub_1B7E640((__int64)a1, v48, a3, v108, v109, v110);
              return v260;
            }
            v238 = v206;
            v247 = v104;
            v193 = (__int64 **)sub_1643360((_QWORD *)a1[9]);
            v194 = sub_1599EF0(v193);
            LOWORD(v273) = 257;
            v195 = v194;
            v196 = *(_QWORD *)(v101 - 24LL * (*(_DWORD *)(v101 + 20) & 0xFFFFFFF));
            v270 = v195;
            v268 = (__int64 *)v196;
            v269 = v234;
            v155 = sub_1285290(v254, *(_QWORD *)(*(_QWORD *)v247 + 24LL), v247, (int)&v268, 3, (__int64)&v271, 0);
          }
          v107 = v238;
          v246 = v155;
          goto LABEL_139;
        }
        return 0;
      }
      if ( (unsigned __int8)sub_1B7C8F0((__int64)a1) || (v61 = (__int64 *)sub_1B7F9F0(a1, v12, v252)) == 0 )
      {
        sub_1B811B0(a4, v48, v226);
        v221 = 0;
      }
      else
      {
        v244 = sub_16463B0(v61, 4u);
        sub_1B811B0(a4, v48, v226);
        v252 = 4;
      }
      if ( (unsigned __int8)sub_1B7C8F0((__int64)a1)
        && (v152 = sub_1B7CA20(*v48),
            v153 = sub_146F1B0(a1[3], v152),
            v154 = sub_1BFA7B0(a1[3], v153, a1 + 26, 0),
            v62 = v154,
            v230 < v154) )
      {
        sub_1B7DE50(*v48, v154);
        if ( !(v154 % v53) )
          goto LABEL_127;
      }
      else
      {
        v62 = v230;
      }
      if ( !(unsigned __int8)sub_1B7C8F0((__int64)a1) )
        goto LABEL_127;
      if ( (v252 & 3) != 0 || v62 % (v53 >> 1) && (unsigned __int8)sub_1B7C8F0((__int64)a1) )
      {
        if ( (v252 & 7) != 0 || v62 % (v53 >> 2) && (unsigned __int8)sub_1B7C8F0((__int64)a1) )
        {
          if ( (unsigned int)sub_1B7C680((__int64)a1, v48[1]) > v62 )
          {
            v65 = v48 + 1;
            v66 = sub_1B83220(a1, v48, 1, a4);
            v67 = a3 - 1;
          }
          else
          {
            if ( v252 != 3 )
            {
              v148 = *v48;
              if ( (unsigned int)sub_1B7C940(*v48) )
                return 0;
              v149 = (__int64 *)sub_1B7CA20(v148);
              v151 = sub_1AE99B0(v149, 4u, a1[5], v150, 0, a1[2]);
              v62 = v151;
              if ( v151 <= 3 || v151 % v53 && (unsigned __int8)sub_1B7C8F0((__int64)a1) )
                return 0;
              goto LABEL_127;
            }
            v65 = v48 + 2;
            v66 = sub_1B83220(a1, v48, 2, a4);
            v67 = a3 - 2;
          }
          return (unsigned int)sub_1B83220(a1, v65, v67, a4) | v66;
        }
        v63 = v252 >> 2;
      }
      else
      {
        v63 = v252 >> 1;
      }
      v64 = v63;
LABEL_78:
      v65 = &v48[v64];
      v66 = sub_1B83220(a1, v48, v64, a4);
      v67 = a3 - v64;
      return (unsigned int)sub_1B83220(a1, v65, v67, a4) | v66;
    }
    if ( v249 >= v200 )
    {
      if ( v249 == v57 || v57 >= v200 )
      {
        v252 = v200;
        goto LABEL_66;
      }
    }
    else
    {
      v245 = v48;
      v75 = 0;
      v76 = 0;
      v253 = 0;
      v222 = a4;
      v77 = 0;
      do
      {
        v78 = &v245[v77];
        v79 = sub_1B7C680((__int64)a1, *v78);
        v80 = v79;
        if ( v79 > v253 )
        {
          v253 = v79;
          v257 = v75;
        }
        v81 = v76;
        if ( v76 )
        {
          v82 = v230;
          while ( 1 )
          {
            v83 = v82 % v81;
            v82 = v81;
            if ( !v83 )
              break;
            v81 = v83;
          }
          v84 = v81;
        }
        else
        {
          v84 = v230;
        }
        if ( v80 < v84 && (v84 & (v84 - 1)) == 0 )
        {
          v85 = *v78;
          v86 = *(_BYTE *)(v85 + 16);
          if ( v86 == 54 )
          {
            sub_15F8F50(v85, v84);
            v86 = *(_BYTE *)(v85 + 16);
          }
          if ( v86 == 55 )
          {
            sub_15F9450(v85, v84);
            v86 = *(_BYTE *)(v85 + 16);
          }
          if ( v86 == 78 )
            sub_1B7DC70(v85, v84);
        }
        v76 += v215;
        v77 = ++v75;
      }
      while ( v75 < a3 );
      v48 = v245;
      a4 = v222;
      if ( v257 )
      {
        v64 = v257;
        goto LABEL_78;
      }
    }
    v64 = v225;
    goto LABEL_78;
  }
  sub_165A590((__int64)&v271, a4, *v48);
  v27 = 0;
  v87 = v211;
  if ( v220 > 1 )
  {
    v88 = sub_1B83220(a1, v211, v220, a4);
    v87 = v211;
    v27 = v88;
  }
  v89 = a3;
  if ( a3 - v220 > 1 )
  {
    v262 = v27;
    v90 = v220 + 1;
    v52 = &v87[v90];
    v51 = v89 - v90;
    return (unsigned int)sub_1B83220(a1, v52, v51, a4) | v262;
  }
  return v27;
}
