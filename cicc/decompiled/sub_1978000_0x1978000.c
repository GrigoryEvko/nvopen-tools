// Function: sub_1978000
// Address: 0x1978000
//
__int64 __fastcall sub_1978000(
        __int64 *a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 *v9; // rbx
  __int64 v10; // rdi
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // r9
  __int64 v15; // r10
  unsigned __int64 v16; // r14
  __int64 v17; // rdx
  int v18; // r8d
  int v19; // r9d
  __int64 v20; // r14
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 v25; // rdx
  _BYTE *v26; // rax
  _BYTE *v27; // r15
  unsigned __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // r12
  char v31; // di
  unsigned int v32; // esi
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rdi
  __int64 v38; // r14
  __int64 v39; // r15
  __int64 v40; // r11
  __int64 v41; // r12
  _BYTE *v42; // rax
  __int64 v43; // r8
  __int64 v44; // r13
  __int64 v45; // r11
  _QWORD *v46; // rax
  __int64 v47; // rax
  __int64 v48; // r13
  __int64 v49; // rdx
  unsigned int v50; // esi
  unsigned int v51; // r8d
  __int64 *v52; // rax
  __int64 v53; // rcx
  int v54; // r8d
  int v55; // r9d
  _QWORD *v56; // r11
  _BYTE *v57; // rsi
  _BYTE *v58; // rsi
  __int64 v59; // r12
  __int64 v60; // r13
  _QWORD *v61; // r14
  __int64 v62; // rbx
  _BYTE *v63; // rsi
  __int64 *v64; // r15
  unsigned int v65; // r12d
  __int64 v67; // rdx
  __int64 v68; // rcx
  _QWORD *v69; // rdx
  _BYTE *v70; // rsi
  _BYTE *v71; // rax
  _BYTE *v72; // r13
  __int64 *v73; // rdi
  int v74; // edx
  size_t v75; // r14
  __int64 v76; // r12
  __int64 *v77; // r12
  __int64 v78; // rcx
  __int64 *v79; // r14
  _QWORD *v80; // r13
  __int64 v81; // rax
  int v82; // edx
  __int64 v83; // r8
  __int64 v84; // rdi
  int v85; // edx
  unsigned int v86; // esi
  __int64 *v87; // rax
  __int64 v88; // r10
  _BYTE *v89; // rsi
  __int64 *v90; // rax
  __int64 v91; // rax
  _QWORD *v92; // r11
  __int64 *v93; // r11
  __int64 *v94; // r12
  __int64 v95; // r14
  unsigned int v96; // esi
  __int64 v97; // rbx
  unsigned int v98; // r9d
  __int64 v99; // r8
  unsigned int v100; // r13d
  unsigned int v101; // edi
  __int64 *v102; // rax
  __int64 v103; // rdx
  __int64 *v104; // rcx
  _BYTE *v105; // rax
  _QWORD *v106; // rax
  __int64 v107; // rbx
  __int64 v108; // rax
  _QWORD *v109; // rdx
  _BYTE *v110; // rsi
  __int64 v111; // r8
  __int64 *v112; // rax
  __int64 v113; // r12
  unsigned int v114; // esi
  __int64 v115; // rdi
  __int64 v116; // rdx
  __int64 *v117; // rax
  __int64 v118; // rcx
  __int64 v119; // rax
  int v120; // r8d
  int v121; // r9d
  __int64 v122; // rdx
  __int64 v123; // r13
  __int64 v124; // rax
  __int64 v125; // r12
  __int64 v126; // rax
  __int64 v127; // rdx
  __int64 v128; // rax
  __int64 v129; // rdx
  int v130; // r8d
  int v131; // r9d
  __int64 v132; // r13
  __int64 v133; // rax
  __int64 v134; // r12
  __int64 v135; // rax
  __int64 v136; // rdx
  __int64 *v137; // r13
  __int64 *v138; // r12
  _QWORD *v139; // r15
  __int64 v140; // rax
  __int64 *v141; // r12
  __int64 v142; // r14
  __int64 v143; // rax
  double v144; // xmm4_8
  double v145; // xmm5_8
  __int64 v146; // rcx
  __int64 v147; // rax
  __int64 v148; // r9
  __int64 *v149; // rbx
  __int64 *v150; // r15
  __int64 v151; // r14
  __int64 v152; // r13
  _QWORD *v153; // rdx
  _QWORD *v154; // rax
  _QWORD *v155; // r12
  __int64 v156; // rax
  __int64 v157; // rdx
  char v158; // r9
  unsigned int v159; // edi
  __int64 v160; // rdx
  __int64 v161; // rax
  __int64 v162; // rcx
  __int64 v163; // rax
  __int64 v164; // rdx
  __int64 v165; // rdi
  __int64 v166; // rdx
  __int64 v167; // rcx
  __int64 *v168; // rdi
  unsigned int v169; // r9d
  __int64 *v170; // rsi
  int v171; // eax
  int v172; // r9d
  __int64 v173; // r10
  int v174; // ecx
  int v175; // r10d
  __int64 v176; // rcx
  __int64 *v177; // rsi
  unsigned int v178; // edi
  __int64 *v179; // rcx
  _QWORD *v180; // rdx
  unsigned int v181; // esi
  unsigned int v182; // ecx
  __int64 *v183; // rax
  __int64 v184; // r8
  __int64 v185; // rax
  char *v186; // rcx
  char *v187; // rax
  __int64 v188; // rdx
  char *v189; // rdx
  int v190; // ecx
  __int64 *v191; // r10
  int v192; // ecx
  int v193; // edx
  int v194; // r10d
  int v195; // r10d
  __int64 v196; // r9
  unsigned int v197; // r13d
  __int64 v198; // rdi
  int v199; // r8d
  __int64 *v200; // rcx
  signed __int64 v201; // rdx
  int v202; // r10d
  int v203; // r10d
  __int64 v204; // r9
  __int64 *v205; // rdi
  unsigned int v206; // r13d
  int v207; // r8d
  __int64 v208; // rcx
  int v209; // eax
  int v210; // r13d
  __int64 v211; // r10
  int v212; // edx
  unsigned int v213; // ecx
  __int64 v214; // r8
  int v215; // r14d
  __int64 *v216; // r10
  int v217; // ecx
  int v218; // r10d
  int v219; // r10d
  __int64 v220; // r9
  int v221; // esi
  unsigned int v222; // r13d
  __int64 *v223; // rcx
  __int64 v224; // rdi
  int v225; // edi
  __int64 *v226; // r10
  int v227; // ecx
  int v228; // edx
  int v229; // eax
  int v230; // r14d
  __int64 v231; // r10
  unsigned int v232; // r8d
  __int64 v233; // rsi
  int v234; // r9d
  __int64 *v235; // rcx
  int v236; // r10d
  int v237; // r10d
  __int64 v238; // r9
  int v239; // r8d
  unsigned int v240; // r14d
  __int64 v241; // rsi
  int v242; // eax
  int v243; // edi
  int v244; // edi
  __int64 *v245; // rsi
  int v246; // [rsp+4h] [rbp-1ACh]
  __int64 v247; // [rsp+8h] [rbp-1A8h]
  __int64 v248; // [rsp+20h] [rbp-190h]
  __int64 *v249; // [rsp+28h] [rbp-188h]
  unsigned int v250; // [rsp+28h] [rbp-188h]
  __int64 *v251; // [rsp+28h] [rbp-188h]
  __int64 *v252; // [rsp+28h] [rbp-188h]
  __int64 v253; // [rsp+40h] [rbp-170h]
  __int64 *v254; // [rsp+40h] [rbp-170h]
  __int64 v255; // [rsp+40h] [rbp-170h]
  __int64 *v256; // [rsp+40h] [rbp-170h]
  unsigned __int64 v257; // [rsp+48h] [rbp-168h]
  unsigned __int64 v258; // [rsp+48h] [rbp-168h]
  __int64 v259; // [rsp+48h] [rbp-168h]
  __int64 v260; // [rsp+48h] [rbp-168h]
  __int64 v261; // [rsp+50h] [rbp-160h]
  __int64 v262; // [rsp+58h] [rbp-158h]
  __int64 v263; // [rsp+60h] [rbp-150h]
  __int64 v264; // [rsp+68h] [rbp-148h]
  __int64 v265; // [rsp+70h] [rbp-140h]
  __int64 v266; // [rsp+70h] [rbp-140h]
  __int64 v267; // [rsp+70h] [rbp-140h]
  _QWORD *v268; // [rsp+70h] [rbp-140h]
  __int64 *v269; // [rsp+70h] [rbp-140h]
  _BYTE *v270; // [rsp+70h] [rbp-140h]
  __int64 v271; // [rsp+78h] [rbp-138h]
  __int64 v272; // [rsp+78h] [rbp-138h]
  _QWORD *v273; // [rsp+78h] [rbp-138h]
  __int64 *v274; // [rsp+78h] [rbp-138h]
  _QWORD *v275; // [rsp+78h] [rbp-138h]
  __int64 v276; // [rsp+78h] [rbp-138h]
  __int64 *v277; // [rsp+78h] [rbp-138h]
  _QWORD *v278; // [rsp+78h] [rbp-138h]
  __int64 *v279; // [rsp+78h] [rbp-138h]
  _QWORD *v280; // [rsp+78h] [rbp-138h]
  __int64 v281; // [rsp+78h] [rbp-138h]
  _QWORD *v282; // [rsp+78h] [rbp-138h]
  _QWORD *v283; // [rsp+78h] [rbp-138h]
  _QWORD *v284; // [rsp+78h] [rbp-138h]
  _QWORD *v285; // [rsp+78h] [rbp-138h]
  _QWORD *v286; // [rsp+78h] [rbp-138h]
  __int64 v287; // [rsp+78h] [rbp-138h]
  __int64 v288; // [rsp+78h] [rbp-138h]
  __int64 v289; // [rsp+80h] [rbp-130h]
  __int64 v290; // [rsp+88h] [rbp-128h]
  __int64 *v291; // [rsp+90h] [rbp-120h] BYREF
  __int64 v292; // [rsp+98h] [rbp-118h]
  __int64 v293; // [rsp+A0h] [rbp-110h]
  __int64 *v294; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v295; // [rsp+B8h] [rbp-F8h]
  _BYTE v296[32]; // [rsp+C0h] [rbp-F0h] BYREF
  _BYTE *v297; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v298; // [rsp+E8h] [rbp-C8h]
  _BYTE v299[64]; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 *v300; // [rsp+130h] [rbp-80h] BYREF
  __int64 v301; // [rsp+138h] [rbp-78h]
  _BYTE dest[112]; // [rsp+140h] [rbp-70h] BYREF

  v9 = a1;
  v10 = a1[1];
  v291 = 0;
  v292 = 0;
  v293 = 0;
  v289 = **(_QWORD **)(v10 + 32);
  v290 = **(_QWORD **)(*v9 + 32);
  v262 = sub_13FCB50(v10);
  v261 = sub_13FCB50(*v9);
  v264 = sub_13FC520(*v9);
  v263 = sub_13FC520(v9[1]);
  v271 = sub_157F120(v264);
  v265 = sub_157F120(v262);
  v11 = sub_157EBA0(v261);
  if ( *(_BYTE *)(v11 + 16) != 26 )
    v11 = 0;
  v12 = sub_157EBA0(v262);
  if ( *(_BYTE *)(v12 + 16) != 26 )
    v12 = 0;
  v13 = sub_157EBA0(v290);
  if ( *(_BYTE *)(v13 + 16) != 26 )
    v13 = 0;
  v14 = sub_157EBA0(v289);
  if ( *(_BYTE *)(v14 + 16) != 26 )
    v14 = 0;
  if ( !v271
    || (v15 = v265) == 0
    || !v11
    || !v12
    || !v13
    || (v266 = v14) == 0
    || (v16 = sub_157EBA0(v15), *(_BYTE *)(v16 + 16) != 26)
    || (v257 = sub_157EBA0(v271), *(_BYTE *)(v257 + 16) != 26)
    || (v253 = sub_157F210(v289)) == 0 )
  {
    v65 = 0;
    goto LABEL_63;
  }
  sub_1977E60(v257, v264, v263, (__int64)&v291);
  sub_1977E60(v13, v261, v9[5], (__int64)&v291);
  sub_1977E60(v13, v263, v253, (__int64)&v291);
  sub_1974500(v253, v289, v290);
  sub_1977E60(v266, v253, v264, (__int64)&v291);
  v267 = *(_QWORD *)(v12 - 24);
  if ( v289 == v267 )
    v267 = *(_QWORD *)(v12 - 48);
  sub_1977E60(v16, v262, v267, (__int64)&v291);
  v297 = v299;
  v298 = 0x800000000LL;
  v20 = sub_157F280(v267);
  v23 = (unsigned int)v298;
  if ( v17 != v20 )
  {
    v24 = v17;
    do
    {
      if ( HIDWORD(v298) <= (unsigned int)v23 )
      {
        sub_16CD150((__int64)&v297, v299, 0, 8, v18, v19);
        v23 = (unsigned int)v298;
      }
      *(_QWORD *)&v297[8 * v23] = v20;
      v23 = (unsigned int)(v298 + 1);
      LODWORD(v298) = v298 + 1;
      if ( !v20 )
        BUG();
      v25 = *(_QWORD *)(v20 + 32);
      if ( !v25 )
        BUG();
      v20 = 0;
      if ( *(_BYTE *)(v25 - 8) == 77 )
        v20 = v25 - 24;
    }
    while ( v24 != v20 );
  }
  v26 = &v297[8 * v23];
  if ( v297 != v26 )
  {
    v258 = v11;
    v27 = v26;
    v254 = v9;
    v28 = (unsigned __int64)v297;
    do
    {
      v29 = 0x17FFFFFFE8LL;
      v30 = *(_QWORD *)v28;
      v31 = *(_BYTE *)(*(_QWORD *)v28 + 23LL) & 0x40;
      v32 = *(_DWORD *)(*(_QWORD *)v28 + 20LL) & 0xFFFFFFF;
      if ( v32 )
      {
        v33 = 24LL * *(unsigned int *)(v30 + 56) + 8;
        v34 = 0;
        do
        {
          v35 = v30 - 24LL * v32;
          if ( v31 )
            v35 = *(_QWORD *)(v30 - 8);
          if ( v262 == *(_QWORD *)(v35 + v33) )
          {
            v29 = 24 * v34;
            goto LABEL_38;
          }
          ++v34;
          v33 += 8;
        }
        while ( v32 != (_DWORD)v34 );
        v29 = 0x17FFFFFFE8LL;
      }
LABEL_38:
      if ( v31 )
        v36 = *(_QWORD *)(v30 - 8);
      else
        v36 = v30 - 24LL * v32;
      v37 = *(_QWORD *)v28;
      v28 += 8LL;
      sub_164D160(v37, *(_QWORD *)(v36 + v29), a2, a3, a4, a5, v21, v22, a8, a9);
      sub_15F20C0((_QWORD *)v30);
    }
    while ( v27 != (_BYTE *)v28 );
    v11 = v258;
    v9 = v254;
  }
  v38 = *(_QWORD *)(v11 - 24);
  if ( v290 == v38 )
    v38 = *(_QWORD *)(v11 - 48);
  sub_1977E60(v12, v267, v38, (__int64)&v291);
  sub_1977E60(v11, v38, v262, (__int64)&v291);
  sub_1974500(v38, v261, v262);
  sub_15DC140(v9[4], v291, (v292 - (__int64)v291) >> 4);
  v39 = *v9;
  v40 = v9[1];
  v41 = *(_QWORD *)*v9;
  v300 = (__int64 *)v263;
  v272 = v40;
  v248 = v39 + 32;
  v42 = sub_1974060(*(_QWORD **)(v39 + 32), *(_QWORD *)(v39 + 40), (__int64 *)&v300);
  sub_1977D00(v43, v42);
  v44 = (__int64)v300;
  v45 = v272;
  v247 = v39 + 56;
  v46 = *(_QWORD **)(v39 + 64);
  if ( *(_QWORD **)(v39 + 72) == v46 )
  {
    v69 = &v46[*(unsigned int *)(v39 + 84)];
    if ( v46 == v69 )
    {
LABEL_229:
      v46 = v69;
    }
    else
    {
      while ( v300 != (__int64 *)*v46 )
      {
        if ( v69 == ++v46 )
          goto LABEL_229;
      }
    }
  }
  else
  {
    v46 = sub_16CC9F0(v247, (__int64)v300);
    v45 = v272;
    if ( v44 == *v46 )
    {
      v67 = *(_QWORD *)(v39 + 72);
      if ( v67 == *(_QWORD *)(v39 + 64) )
        v68 = *(unsigned int *)(v39 + 84);
      else
        v68 = *(unsigned int *)(v39 + 80);
      v69 = (_QWORD *)(v67 + 8 * v68);
    }
    else
    {
      v47 = *(_QWORD *)(v39 + 72);
      if ( v47 != *(_QWORD *)(v39 + 64) )
        goto LABEL_47;
      v46 = (_QWORD *)(v47 + 8LL * *(unsigned int *)(v39 + 84));
      v69 = v46;
    }
  }
  if ( v69 != v46 )
  {
    *v46 = -2;
    ++*(_DWORD *)(v39 + 88);
  }
LABEL_47:
  v48 = v9[3];
  v49 = *(_QWORD *)(v48 + 8);
  v50 = *(_DWORD *)(v48 + 24);
  if ( v41 )
  {
    if ( v50 )
    {
      v51 = (v50 - 1) & (((unsigned int)v263 >> 4) ^ ((unsigned int)v263 >> 9));
      v52 = (__int64 *)(v49 + 16LL * v51);
      v53 = *v52;
      if ( v263 == *v52 )
      {
LABEL_50:
        v52[1] = v41;
        v273 = (_QWORD *)v45;
        sub_1977D40(v41, v39);
        sub_1977D40(v39, (__int64)v273);
        v56 = v273;
        v300 = v273;
        *v273 = v41;
        v57 = *(_BYTE **)(v41 + 16);
        if ( v57 == *(_BYTE **)(v41 + 24) )
        {
          sub_13FD960(v41 + 8, v57, &v300);
          v56 = v273;
        }
        else
        {
          if ( v57 )
          {
            *(_QWORD *)v57 = v300;
            v57 = *(_BYTE **)(v41 + 16);
          }
          *(_QWORD *)(v41 + 16) = v57 + 8;
        }
        goto LABEL_54;
      }
      v225 = 1;
      v226 = 0;
      while ( v53 != -8 )
      {
        if ( v53 == -16 && !v226 )
          v226 = v52;
        v51 = (v50 - 1) & (v225 + v51);
        v52 = (__int64 *)(v49 + 16LL * v51);
        v53 = *v52;
        if ( v263 == *v52 )
          goto LABEL_50;
        ++v225;
      }
      v227 = *(_DWORD *)(v48 + 16);
      if ( v226 )
        v52 = v226;
      ++*(_QWORD *)v48;
      v228 = v227 + 1;
      if ( 4 * (v227 + 1) < 3 * v50 )
      {
        if ( v50 - *(_DWORD *)(v48 + 20) - v228 > v50 >> 3 )
        {
LABEL_303:
          *(_DWORD *)(v48 + 16) = v228;
          if ( *v52 != -8 )
            --*(_DWORD *)(v48 + 20);
          v52[1] = 0;
          *v52 = v263;
          goto LABEL_50;
        }
        v288 = v45;
        sub_1400170(v48, v50);
        v236 = *(_DWORD *)(v48 + 24);
        if ( v236 )
        {
          v237 = v236 - 1;
          v238 = *(_QWORD *)(v48 + 8);
          v239 = 1;
          v240 = v237 & (((unsigned int)v263 >> 4) ^ ((unsigned int)v263 >> 9));
          v45 = v288;
          v228 = *(_DWORD *)(v48 + 16) + 1;
          v235 = 0;
          v52 = (__int64 *)(v238 + 16LL * v240);
          v241 = *v52;
          if ( v263 == *v52 )
            goto LABEL_303;
          while ( v241 != -8 )
          {
            if ( v241 == -16 && !v235 )
              v235 = v52;
            v240 = v237 & (v239 + v240);
            v52 = (__int64 *)(v238 + 16LL * v240);
            v241 = *v52;
            if ( v263 == *v52 )
              goto LABEL_303;
            ++v239;
          }
          goto LABEL_311;
        }
        goto LABEL_393;
      }
    }
    else
    {
      ++*(_QWORD *)v48;
    }
    v287 = v45;
    sub_1400170(v48, 2 * v50);
    v229 = *(_DWORD *)(v48 + 24);
    if ( v229 )
    {
      v230 = v229 - 1;
      v231 = *(_QWORD *)(v48 + 8);
      v45 = v287;
      v228 = *(_DWORD *)(v48 + 16) + 1;
      v232 = (v229 - 1) & (((unsigned int)v263 >> 9) ^ ((unsigned int)v263 >> 4));
      v52 = (__int64 *)(v231 + 16LL * v232);
      v233 = *v52;
      if ( *v52 == v263 )
        goto LABEL_303;
      v234 = 1;
      v235 = 0;
      while ( v233 != -8 )
      {
        if ( !v235 && v233 == -16 )
          v235 = v52;
        v232 = v230 & (v234 + v232);
        v52 = (__int64 *)(v231 + 16LL * v232);
        v233 = *v52;
        if ( v263 == *v52 )
          goto LABEL_303;
        ++v234;
      }
LABEL_311:
      if ( v235 )
        v52 = v235;
      goto LABEL_303;
    }
LABEL_393:
    ++*(_DWORD *)(v48 + 16);
    BUG();
  }
  if ( v50 )
  {
    v181 = v50 - 1;
    v182 = v181 & (((unsigned int)v263 >> 9) ^ ((unsigned int)v263 >> 4));
    v183 = (__int64 *)(v49 + 16LL * v182);
    v184 = *v183;
    if ( v263 == *v183 )
    {
LABEL_233:
      *v183 = -16;
      --*(_DWORD *)(v48 + 16);
      ++*(_DWORD *)(v48 + 20);
    }
    else
    {
      v242 = 1;
      while ( v184 != -8 )
      {
        v243 = v242 + 1;
        v182 = v181 & (v242 + v182);
        v183 = (__int64 *)(v49 + 16LL * v182);
        v184 = *v183;
        if ( v263 == *v183 )
          goto LABEL_233;
        v242 = v243;
      }
    }
  }
  v282 = (_QWORD *)v45;
  sub_1977D40(v39, v45);
  v185 = v9[3];
  v56 = v282;
  v186 = *(char **)(v185 + 40);
  v187 = *(char **)(v185 + 32);
  v188 = (v186 - v187) >> 5;
  if ( v188 > 0 )
  {
    v189 = &v187[32 * v188];
    while ( v39 != *(_QWORD *)v187 )
    {
      if ( v39 == *((_QWORD *)v187 + 1) )
      {
        v187 += 8;
        goto LABEL_241;
      }
      if ( v39 == *((_QWORD *)v187 + 2) )
      {
        v187 += 16;
        goto LABEL_241;
      }
      if ( v39 == *((_QWORD *)v187 + 3) )
      {
        v187 += 24;
        goto LABEL_241;
      }
      v187 += 32;
      if ( v189 == v187 )
        goto LABEL_266;
    }
    goto LABEL_241;
  }
LABEL_266:
  v201 = v186 - v187;
  if ( v186 - v187 == 16 )
  {
LABEL_324:
    if ( v39 == *(_QWORD *)v187 )
      goto LABEL_241;
    v187 += 8;
LABEL_326:
    if ( v39 != *(_QWORD *)v187 )
      v187 = v186;
    goto LABEL_241;
  }
  if ( v201 == 24 )
  {
    if ( v39 == *(_QWORD *)v187 )
      goto LABEL_241;
    v187 += 8;
    goto LABEL_324;
  }
  if ( v201 == 8 )
    goto LABEL_326;
  v187 = v186;
LABEL_241:
  *(_QWORD *)v187 = v282;
LABEL_54:
  v58 = (_BYTE *)v56[1];
  v59 = (__int64)(v56 + 1);
  v60 = v39 + 8;
  if ( v58 != (_BYTE *)v56[2] )
  {
    v274 = v9;
    v61 = v56;
    v62 = v39;
    do
    {
      v64 = *(__int64 **)v58;
      sub_13FDAF0(v59, v58);
      v300 = v64;
      *v64 = v62;
      v63 = *(_BYTE **)(v62 + 16);
      if ( v63 == *(_BYTE **)(v62 + 24) )
      {
        sub_13FD960(v60, v63, &v300);
      }
      else
      {
        if ( v63 )
        {
          *(_QWORD *)v63 = v300;
          v63 = *(_BYTE **)(v62 + 16);
        }
        *(_QWORD *)(v62 + 16) = v63 + 8;
      }
      v58 = (_BYTE *)v61[1];
    }
    while ( (_BYTE *)v61[2] != v58 );
    v39 = v62;
    v9 = v274;
    v56 = v61;
  }
  v300 = (__int64 *)v39;
  *(_QWORD *)v39 = v56;
  v70 = (_BYTE *)v56[2];
  if ( v70 == (_BYTE *)v56[3] )
  {
    v283 = v56;
    sub_13FD960(v59, v70, &v300);
    v56 = v283;
  }
  else
  {
    if ( v70 )
    {
      *(_QWORD *)v70 = v300;
      v70 = (_BYTE *)v56[2];
    }
    v56[2] = v70 + 8;
  }
  v71 = (_BYTE *)v56[5];
  v72 = (_BYTE *)v56[4];
  v73 = (__int64 *)dest;
  v74 = 0;
  v300 = (__int64 *)dest;
  v75 = v71 - v72;
  v301 = 0x800000000LL;
  v76 = (v71 - v72) >> 3;
  if ( (unsigned __int64)(v71 - v72) > 0x40 )
  {
    v270 = v71;
    v280 = v56;
    sub_16CD150((__int64)&v300, dest, (v71 - v72) >> 3, 8, v54, v55);
    v74 = v301;
    v71 = v270;
    v56 = v280;
    v73 = &v300[(unsigned int)v301];
  }
  if ( v72 != v71 )
  {
    v275 = v56;
    memcpy(v73, v72, v75);
    v74 = v301;
    v56 = v275;
  }
  LODWORD(v301) = v74 + v76;
  v77 = *(__int64 **)(v39 + 32);
  if ( v77 != *(__int64 **)(v39 + 40) )
  {
    v78 = (__int64)(v56 + 7);
    v79 = *(__int64 **)(v39 + 40);
    v80 = v56;
    while ( 1 )
    {
      v81 = v9[3];
      v82 = *(_DWORD *)(v81 + 24);
      if ( v82 )
      {
        v83 = *v77;
        v84 = *(_QWORD *)(v81 + 8);
        v85 = v82 - 1;
        v86 = v85 & (((unsigned int)*v77 >> 9) ^ ((unsigned int)*v77 >> 4));
        v87 = (__int64 *)(v84 + 16LL * v86);
        v88 = *v87;
        if ( *v77 != *v87 )
        {
          v171 = 1;
          while ( v88 != -8 )
          {
            v172 = v171 + 1;
            v86 = v85 & (v171 + v86);
            v87 = (__int64 *)(v84 + 16LL * v86);
            v88 = *v87;
            if ( v83 == *v87 )
              goto LABEL_87;
            v171 = v172;
          }
          goto LABEL_84;
        }
LABEL_87:
        if ( v39 == v87[1] )
        {
          v294 = (__int64 *)*v77;
          v89 = (_BYTE *)v80[5];
          if ( v89 == (_BYTE *)v80[6] )
          {
            v281 = v78;
            sub_1292090((__int64)(v80 + 4), v89, &v294);
            v83 = (__int64)v294;
            v78 = v281;
          }
          else
          {
            if ( v89 )
            {
              *(_QWORD *)v89 = v83;
              v89 = (_BYTE *)v80[5];
            }
            v80[5] = v89 + 8;
          }
          v90 = (__int64 *)v80[8];
          if ( (__int64 *)v80[9] != v90 )
            goto LABEL_93;
          v168 = &v90[*((unsigned int *)v80 + 21)];
          v169 = *((_DWORD *)v80 + 21);
          if ( v90 != v168 )
          {
            v170 = 0;
            while ( *v90 != v83 )
            {
              if ( *v90 == -2 )
                v170 = v90;
              if ( v168 == ++v90 )
              {
                if ( !v170 )
                  goto LABEL_255;
                *v170 = v83;
                --*((_DWORD *)v80 + 22);
                ++v80[7];
                goto LABEL_84;
              }
            }
            goto LABEL_84;
          }
LABEL_255:
          if ( v169 < *((_DWORD *)v80 + 20) )
          {
            *((_DWORD *)v80 + 21) = v169 + 1;
            *v168 = v83;
            ++v80[7];
          }
          else
          {
LABEL_93:
            v276 = v78;
            sub_16CCBA0(v78, v83);
            v78 = v276;
          }
        }
      }
LABEL_84:
      if ( v79 == ++v77 )
      {
        v56 = v80;
        break;
      }
    }
  }
  v268 = v56;
  v259 = *(_QWORD *)v56[4];
  v91 = sub_13FCB50((__int64)v56);
  v92 = v268;
  v255 = v91;
  v277 = &v300[(unsigned int)v301];
  if ( v300 == v277 )
    goto LABEL_112;
  v93 = v9;
  v94 = v300;
  do
  {
    v95 = v93[3];
    v96 = *(_DWORD *)(v95 + 24);
    if ( !v96 )
      goto LABEL_99;
    v97 = *v94;
    v98 = v96 - 1;
    v99 = *(_QWORD *)(v95 + 8);
    v100 = ((unsigned int)*v94 >> 9) ^ ((unsigned int)*v94 >> 4);
    v101 = (v96 - 1) & v100;
    v102 = (__int64 *)(v99 + 16LL * v101);
    v103 = *v102;
    v104 = v102;
    if ( *v94 == *v102 )
    {
LABEL_102:
      if ( v268 != (_QWORD *)v104[1] )
        goto LABEL_99;
      if ( v259 == v97 || v255 == v97 )
      {
        v294 = (__int64 *)*v94;
        v249 = v93;
        v105 = sub_1974060(*(_QWORD **)(v39 + 32), *(_QWORD *)(v39 + 40), (__int64 *)&v294);
        sub_1977D00(v248, v105);
        v106 = *(_QWORD **)(v39 + 64);
        v107 = (__int64)v294;
        v93 = v249;
        if ( *(_QWORD **)(v39 + 72) == v106 )
        {
          v109 = &v106[*(unsigned int *)(v39 + 84)];
          if ( v106 == v109 )
          {
LABEL_198:
            v106 = v109;
          }
          else
          {
            while ( v294 != (__int64 *)*v106 )
            {
              if ( v109 == ++v106 )
                goto LABEL_198;
            }
          }
        }
        else
        {
          v106 = sub_16CC9F0(v247, (__int64)v294);
          v93 = v249;
          if ( v107 == *v106 )
          {
            v166 = *(_QWORD *)(v39 + 72);
            if ( v166 == *(_QWORD *)(v39 + 64) )
              v167 = *(unsigned int *)(v39 + 84);
            else
              v167 = *(unsigned int *)(v39 + 80);
            v109 = (_QWORD *)(v166 + 8 * v167);
          }
          else
          {
            v108 = *(_QWORD *)(v39 + 72);
            if ( v108 != *(_QWORD *)(v39 + 64) )
              goto LABEL_99;
            v106 = (_QWORD *)(v108 + 8LL * *(unsigned int *)(v39 + 84));
            v109 = v106;
          }
        }
        if ( v109 != v106 )
        {
          *v106 = -2;
          ++*(_DWORD *)(v39 + 88);
        }
      }
      else
      {
        if ( v97 != v103 )
        {
          v190 = 1;
          v191 = 0;
          while ( v103 != -8 )
          {
            if ( v103 == -16 && !v191 )
              v191 = v102;
            v101 = v98 & (v190 + v101);
            v102 = (__int64 *)(v99 + 16LL * v101);
            v103 = *v102;
            if ( v97 == *v102 )
              goto LABEL_98;
            ++v190;
          }
          v192 = *(_DWORD *)(v95 + 16);
          if ( v191 )
            v102 = v191;
          ++*(_QWORD *)v95;
          v193 = v192 + 1;
          if ( 4 * (v192 + 1) >= 3 * v96 )
          {
            v251 = v93;
            sub_1400170(v95, 2 * v96);
            v194 = *(_DWORD *)(v95 + 24);
            if ( !v194 )
              goto LABEL_391;
            v195 = v194 - 1;
            v196 = *(_QWORD *)(v95 + 8);
            v197 = v195 & v100;
            v93 = v251;
            v193 = *(_DWORD *)(v95 + 16) + 1;
            v102 = (__int64 *)(v196 + 16LL * v197);
            v198 = *v102;
            if ( v97 != *v102 )
            {
              v199 = 1;
              v200 = 0;
              while ( v198 != -8 )
              {
                if ( !v200 && v198 == -16 )
                  v200 = v102;
                v197 = v195 & (v199 + v197);
                v102 = (__int64 *)(v196 + 16LL * v197);
                v198 = *v102;
                if ( v97 == *v102 )
                  goto LABEL_250;
                ++v199;
              }
              if ( v200 )
                v102 = v200;
            }
          }
          else if ( v96 - *(_DWORD *)(v95 + 20) - v193 <= v96 >> 3 )
          {
            v252 = v93;
            sub_1400170(v95, v96);
            v202 = *(_DWORD *)(v95 + 24);
            if ( !v202 )
            {
LABEL_391:
              ++*(_DWORD *)(v95 + 16);
              BUG();
            }
            v203 = v202 - 1;
            v204 = *(_QWORD *)(v95 + 8);
            v205 = 0;
            v206 = v203 & v100;
            v93 = v252;
            v207 = 1;
            v193 = *(_DWORD *)(v95 + 16) + 1;
            v102 = (__int64 *)(v204 + 16LL * v206);
            v208 = *v102;
            if ( v97 != *v102 )
            {
              while ( v208 != -8 )
              {
                if ( v208 == -16 && !v205 )
                  v205 = v102;
                v206 = v203 & (v207 + v206);
                v102 = (__int64 *)(v204 + 16LL * v206);
                v208 = *v102;
                if ( v97 == *v102 )
                  goto LABEL_250;
                ++v207;
              }
              if ( v205 )
                v102 = v205;
            }
          }
LABEL_250:
          *(_DWORD *)(v95 + 16) = v193;
          if ( *v102 != -8 )
            --*(_DWORD *)(v95 + 20);
          *v102 = v97;
          v102[1] = 0;
        }
LABEL_98:
        v102[1] = v39;
      }
    }
    else
    {
      v250 = (v96 - 1) & (((unsigned int)*v94 >> 9) ^ ((unsigned int)*v94 >> 4));
      v173 = *v102;
      v174 = 1;
      while ( v173 != -8 )
      {
        v175 = v174 + 1;
        v176 = v98 & (v250 + v174);
        v246 = v175;
        v250 = v176;
        v104 = (__int64 *)(v99 + 16 * v176);
        v173 = *v104;
        if ( v97 == *v104 )
          goto LABEL_102;
        v174 = v246;
      }
    }
LABEL_99:
    ++v94;
  }
  while ( v277 != v94 );
  v9 = v93;
  v92 = v268;
LABEL_112:
  v294 = (__int64 *)v264;
  v110 = (_BYTE *)v92[5];
  if ( v110 == (_BYTE *)v92[6] )
  {
    v284 = v92;
    sub_1292090((__int64)(v92 + 4), v110, &v294);
    v111 = (__int64)v294;
    v92 = v284;
  }
  else
  {
    if ( v110 )
    {
      *(_QWORD *)v110 = v264;
      v110 = (_BYTE *)v92[5];
    }
    v111 = v264;
    v92[5] = v110 + 8;
  }
  v112 = (__int64 *)v92[8];
  if ( (__int64 *)v92[9] == v112 )
  {
    v177 = &v112[*((unsigned int *)v92 + 21)];
    v178 = *((_DWORD *)v92 + 21);
    if ( v112 == v177 )
    {
LABEL_257:
      if ( v178 >= *((_DWORD *)v92 + 20) )
        goto LABEL_117;
      *((_DWORD *)v92 + 21) = v178 + 1;
      *v177 = v111;
      ++v92[7];
    }
    else
    {
      v179 = 0;
      while ( *v112 != v111 )
      {
        if ( *v112 == -2 )
          v179 = v112;
        if ( v177 == ++v112 )
        {
          if ( !v179 )
            goto LABEL_257;
          *v179 = v111;
          --*((_DWORD *)v92 + 22);
          ++v92[7];
          break;
        }
      }
    }
  }
  else
  {
LABEL_117:
    v278 = v92;
    sub_16CCBA0((__int64)(v92 + 7), v111);
    v92 = v278;
  }
  v113 = v9[3];
  v114 = *(_DWORD *)(v113 + 24);
  if ( !v114 )
  {
    ++*(_QWORD *)v113;
    goto LABEL_280;
  }
  v115 = *(_QWORD *)(v113 + 8);
  LODWORD(v116) = (v114 - 1) & (((unsigned int)v264 >> 4) ^ ((unsigned int)v264 >> 9));
  v117 = (__int64 *)(v115 + 16LL * (unsigned int)v116);
  v118 = *v117;
  if ( v264 != *v117 )
  {
    v215 = 1;
    v216 = 0;
    while ( v118 != -8 )
    {
      if ( !v216 && v118 == -16 )
        v216 = v117;
      v116 = (v114 - 1) & ((_DWORD)v116 + v215);
      v117 = (__int64 *)(v115 + 16 * v116);
      v118 = *v117;
      if ( v264 == *v117 )
        goto LABEL_120;
      ++v215;
    }
    v217 = *(_DWORD *)(v113 + 16);
    if ( v216 )
      v117 = v216;
    ++*(_QWORD *)v113;
    v212 = v217 + 1;
    if ( 4 * (v217 + 1) < 3 * v114 )
    {
      if ( v114 - *(_DWORD *)(v113 + 20) - v212 > v114 >> 3 )
      {
LABEL_282:
        *(_DWORD *)(v113 + 16) = v212;
        if ( *v117 != -8 )
          --*(_DWORD *)(v113 + 20);
        v117[1] = 0;
        *v117 = v264;
        goto LABEL_120;
      }
      v286 = v92;
      sub_1400170(v113, v114);
      v218 = *(_DWORD *)(v113 + 24);
      if ( v218 )
      {
        v219 = v218 - 1;
        v220 = *(_QWORD *)(v113 + 8);
        v221 = 1;
        v222 = v219 & (((unsigned int)v264 >> 4) ^ ((unsigned int)v264 >> 9));
        v92 = v286;
        v212 = *(_DWORD *)(v113 + 16) + 1;
        v223 = 0;
        v117 = (__int64 *)(v220 + 16LL * v222);
        v224 = *v117;
        if ( v264 != *v117 )
        {
          while ( v224 != -8 )
          {
            if ( !v223 && v224 == -16 )
              v223 = v117;
            v222 = v219 & (v221 + v222);
            v117 = (__int64 *)(v220 + 16LL * v222);
            v224 = *v117;
            if ( v264 == *v117 )
              goto LABEL_282;
            ++v221;
          }
          if ( v223 )
            v117 = v223;
        }
        goto LABEL_282;
      }
LABEL_392:
      ++*(_DWORD *)(v113 + 16);
      BUG();
    }
LABEL_280:
    v285 = v92;
    sub_1400170(v113, 2 * v114);
    v209 = *(_DWORD *)(v113 + 24);
    if ( v209 )
    {
      v210 = v209 - 1;
      v211 = *(_QWORD *)(v113 + 8);
      v92 = v285;
      v212 = *(_DWORD *)(v113 + 16) + 1;
      v213 = (v209 - 1) & (((unsigned int)v264 >> 9) ^ ((unsigned int)v264 >> 4));
      v117 = (__int64 *)(v211 + 16LL * v213);
      v214 = *v117;
      if ( v264 != *v117 )
      {
        v244 = 1;
        v245 = 0;
        while ( v214 != -8 )
        {
          if ( !v245 && v214 == -16 )
            v245 = v117;
          v213 = v210 & (v244 + v213);
          v117 = (__int64 *)(v211 + 16LL * v213);
          v214 = *v117;
          if ( v264 == *v117 )
            goto LABEL_282;
          ++v244;
        }
        if ( v245 )
          v117 = v245;
      }
      goto LABEL_282;
    }
    goto LABEL_392;
  }
LABEL_120:
  v117[1] = (__int64)v92;
  if ( v300 != (__int64 *)dest )
    _libc_free((unsigned __int64)v300);
  v294 = (__int64 *)v296;
  v295 = 0x400000000LL;
  v300 = (__int64 *)dest;
  v301 = 0x400000000LL;
  v119 = sub_157F280(v289);
  v123 = v122;
  if ( !v119 )
    BUG();
  v124 = *(_QWORD *)(v119 + 32);
  if ( !v124 )
    BUG();
  v125 = 0;
  if ( *(_BYTE *)(v124 - 8) == 77 )
    v125 = v124 - 24;
  v126 = (unsigned int)v295;
  while ( v123 != v125 )
  {
    while ( 1 )
    {
      if ( (unsigned int)v126 >= HIDWORD(v295) )
      {
        sub_16CD150((__int64)&v294, v296, 0, 8, v120, v121);
        v126 = (unsigned int)v295;
      }
      v294[v126] = v125;
      v126 = (unsigned int)(v295 + 1);
      LODWORD(v295) = v295 + 1;
      if ( !v125 )
        BUG();
      v127 = *(_QWORD *)(v125 + 32);
      if ( !v127 )
        BUG();
      v125 = 0;
      if ( *(_BYTE *)(v127 - 8) != 77 )
        break;
      v125 = v127 - 24;
      if ( v123 == v127 - 24 )
        goto LABEL_134;
    }
  }
LABEL_134:
  v128 = sub_157F280(v290);
  v132 = v129;
  if ( !v128 )
    BUG();
  v133 = *(_QWORD *)(v128 + 32);
  if ( !v133 )
    BUG();
  v134 = 0;
  if ( *(_BYTE *)(v133 - 8) == 77 )
    v134 = v133 - 24;
  v135 = (unsigned int)v301;
  if ( v129 != v134 )
  {
    do
    {
      if ( (unsigned int)v135 >= HIDWORD(v301) )
      {
        sub_16CD150((__int64)&v300, dest, 0, 8, v130, v131);
        v135 = (unsigned int)v301;
      }
      v300[v135] = v134;
      v135 = (unsigned int)(v301 + 1);
      LODWORD(v301) = v301 + 1;
      if ( !v134 )
        BUG();
      v136 = *(_QWORD *)(v134 + 32);
      if ( !v136 )
        BUG();
      v134 = 0;
      if ( *(_BYTE *)(v136 - 8) == 77 )
        v134 = v136 - 24;
    }
    while ( v132 != v134 );
  }
  v137 = v300;
  v138 = &v300[v135];
  if ( v300 != v138 )
  {
    do
    {
      v139 = (_QWORD *)*v137++;
      v140 = sub_157ED20(v289);
      sub_15F22F0(v139, v140);
    }
    while ( v138 != v137 );
  }
  v141 = v294;
  v256 = &v294[(unsigned int)v295];
  if ( v294 == v256 )
    goto LABEL_162;
  v279 = v9;
  while ( 2 )
  {
    v142 = *v141;
    v143 = sub_157ED20(v290);
    sub_15F22F0((_QWORD *)v142, v143);
    if ( (*(_BYTE *)(v142 + 23) & 0x40) != 0 )
    {
      v146 = *(_QWORD *)(v142 - 8);
      v147 = *(_DWORD *)(v142 + 20) & 0xFFFFFFF;
    }
    else
    {
      v147 = *(_DWORD *)(v142 + 20) & 0xFFFFFFF;
      v146 = v142 - 24 * v147;
    }
    v148 = v146 + 24LL * *(unsigned int *)(v142 + 56) + 8;
    v149 = (__int64 *)(v148 + 8 * v147);
    if ( v149 == (__int64 *)v148 )
      goto LABEL_161;
    v269 = v141;
    v150 = (__int64 *)(v146 + 24LL * *(unsigned int *)(v142 + 56) + 8);
    v260 = v142;
    while ( 2 )
    {
      v151 = *v150;
      v152 = v279[1];
      v153 = *(_QWORD **)(v152 + 72);
      v154 = *(_QWORD **)(v152 + 64);
      if ( v153 != v154 )
      {
        v155 = &v153[*(unsigned int *)(v152 + 80)];
        v154 = sub_16CC9F0(v152 + 56, *v150);
        if ( v151 == *v154 )
        {
          v164 = *(_QWORD *)(v152 + 72);
          if ( v164 == *(_QWORD *)(v152 + 64) )
            v165 = *(unsigned int *)(v152 + 84);
          else
            v165 = *(unsigned int *)(v152 + 80);
          v180 = (_QWORD *)(v164 + 8 * v165);
          goto LABEL_176;
        }
        v156 = *(_QWORD *)(v152 + 72);
        if ( v156 == *(_QWORD *)(v152 + 64) )
        {
          v154 = (_QWORD *)(v156 + 8LL * *(unsigned int *)(v152 + 84));
          v180 = v154;
          goto LABEL_176;
        }
        v154 = (_QWORD *)(v156 + 8LL * *(unsigned int *)(v152 + 80));
LABEL_158:
        if ( v155 == v154 )
          break;
        goto LABEL_159;
      }
      v155 = &v154[*(unsigned int *)(v152 + 84)];
      if ( v154 == v155 )
      {
        v180 = *(_QWORD **)(v152 + 64);
      }
      else
      {
        do
        {
          if ( v151 == *v154 )
            break;
          ++v154;
        }
        while ( v155 != v154 );
        v180 = v155;
      }
LABEL_176:
      while ( v180 != v154 )
      {
        if ( *v154 < 0xFFFFFFFFFFFFFFFELL )
          goto LABEL_158;
        ++v154;
      }
      if ( v155 != v154 )
      {
LABEL_159:
        if ( v149 == ++v150 )
        {
          v141 = v269;
          goto LABEL_161;
        }
        continue;
      }
      break;
    }
    v141 = v269;
    v157 = 0x17FFFFFFE8LL;
    v158 = *(_BYTE *)(v260 + 23) & 0x40;
    v159 = *(_DWORD *)(v260 + 20) & 0xFFFFFFF;
    if ( v159 )
    {
      v160 = 24LL * *(unsigned int *)(v260 + 56) + 8;
      v161 = 0;
      do
      {
        v162 = v260 - 24LL * v159;
        if ( v158 )
          v162 = *(_QWORD *)(v260 - 8);
        if ( v151 == *(_QWORD *)(v162 + v160) )
        {
          v157 = 24 * v161;
          goto LABEL_185;
        }
        ++v161;
        v160 += 8;
      }
      while ( v159 != (_DWORD)v161 );
      v157 = 0x17FFFFFFE8LL;
    }
LABEL_185:
    if ( v158 )
      v163 = *(_QWORD *)(v260 - 8);
    else
      v163 = v260 - 24LL * v159;
    sub_164D160(v260, *(_QWORD *)(v163 + v157), a2, a3, a4, a5, v144, v145, a8, a9);
    sub_15F20C0((_QWORD *)v260);
LABEL_161:
    if ( v256 != ++v141 )
      continue;
    break;
  }
LABEL_162:
  sub_1974500(v290, v263, v264);
  sub_1974500(v290, v262, v261);
  sub_1974500(v289, v264, v263);
  sub_1974500(v289, v261, v262);
  if ( v300 != (__int64 *)dest )
    _libc_free((unsigned __int64)v300);
  if ( v294 != (__int64 *)v296 )
    _libc_free((unsigned __int64)v294);
  if ( v297 != v299 )
    _libc_free((unsigned __int64)v297);
  v65 = 1;
LABEL_63:
  if ( v291 )
    j_j___libc_free_0(v291, v293 - (_QWORD)v291);
  return v65;
}
