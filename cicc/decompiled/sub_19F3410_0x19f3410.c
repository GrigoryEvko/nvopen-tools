// Function: sub_19F3410
// Address: 0x19f3410
//
__int64 __fastcall sub_19F3410(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rbx
  int v17; // r8d
  int v18; // r9d
  int v19; // eax
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // r13
  __int64 v22; // r15
  unsigned __int64 v23; // r14
  __int64 *v24; // rdi
  int v25; // eax
  __int64 *v26; // rdx
  __int64 *v27; // rax
  __int64 v28; // rcx
  __int64 *v29; // r13
  __int64 *v30; // r15
  __int64 v31; // r14
  __int64 v32; // rbx
  __int64 v33; // rax
  __int64 v34; // r10
  __int64 *v36; // rbx
  __int64 *v37; // r15
  __int64 *v38; // r13
  unsigned int v39; // eax
  __int64 v40; // rdi
  unsigned int v41; // ecx
  unsigned int v42; // edx
  unsigned int v43; // eax
  unsigned int v44; // eax
  __int64 *v45; // r15
  __int64 **v46; // r13
  __int64 v47; // rdx
  __int64 *v48; // rax
  char v49; // bl
  __int64 v50; // r14
  __int64 *v51; // rbx
  unsigned int v52; // eax
  int v53; // r8d
  int v54; // r9d
  __int64 v55; // rax
  __int64 *v56; // rax
  __int64 v57; // rax
  __int64 v58; // r12
  char v59; // al
  __int64 **v60; // rdx
  __int64 v61; // r9
  __int64 k; // r12
  __int64 v63; // r14
  __int64 v64; // r13
  __int64 v65; // rsi
  __int64 v66; // rdx
  __int64 v67; // rcx
  double v68; // xmm4_8
  double v69; // xmm5_8
  __int64 v70; // rax
  __int64 v71; // rsi
  unsigned __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // r13
  __int64 *v75; // rdx
  int v76; // eax
  __int64 *v77; // rcx
  __int64 v78; // r15
  __int64 v79; // r12
  __int64 *v80; // rdx
  __int64 *v81; // rax
  unsigned __int64 v82; // rdi
  int v83; // ebx
  char v84; // al
  __int64 *v85; // rdx
  __int64 v86; // r12
  char v87; // al
  __int64 **v88; // rdx
  __int64 *v89; // rsi
  char v90; // al
  __int64 v91; // rcx
  double v92; // xmm4_8
  double v93; // xmm5_8
  __int64 v94; // rdx
  char v95; // al
  _QWORD *v96; // rbx
  int v97; // eax
  __int64 v98; // rbx
  __int64 *v99; // rax
  __int64 v100; // rdx
  unsigned int v101; // edi
  __int64 v102; // rdx
  __int64 v103; // rax
  __int64 v104; // rcx
  __int64 *v105; // rax
  __int64 v106; // r12
  int v107; // eax
  int v108; // edi
  __int64 *v109; // rax
  __int64 v110; // r8
  __int64 v111; // rax
  __int64 *v112; // r12
  __int64 *v113; // rbx
  __int64 *v114; // rax
  __int64 v115; // r12
  __int64 **v116; // rbx
  __int64 v117; // r14
  char v118; // r8
  __int64 *v119; // rax
  int v120; // ecx
  unsigned int v121; // esi
  int v122; // edx
  _QWORD *v123; // rdx
  __int64 v124; // rsi
  __int64 v125; // rdx
  __int64 v126; // rcx
  double v127; // xmm4_8
  double v128; // xmm5_8
  _QWORD *v129; // rax
  int v130; // eax
  unsigned int v131; // esi
  int v132; // eax
  int v133; // eax
  unsigned int v134; // esi
  int v135; // eax
  int v136; // eax
  unsigned int v137; // esi
  int v138; // eax
  __int64 *v139; // r13
  __int64 *j; // rax
  const char *v141; // rbx
  __int64 *i; // rax
  const char *v143; // rax
  __int64 v144; // rbx
  __int64 *v145; // rax
  __int64 v146; // r10
  __int64 *v147; // r15
  unsigned int v148; // r14d
  __int64 v149; // r12
  __int64 v150; // r13
  __int64 v151; // rax
  __int64 *v152; // rbx
  unsigned int v153; // eax
  __int64 v154; // rax
  __int64 v155; // r12
  int v156; // ebx
  __int64 v157; // rax
  __int64 v158; // r10
  __int64 v159; // r14
  __int64 v160; // rbx
  char v161; // al
  char *v162; // rdx
  __int64 v163; // r10
  char v164; // al
  char *v165; // rdx
  __int64 v166; // r10
  unsigned int v167; // esi
  int v168; // eax
  int v169; // eax
  __int64 *v170; // rax
  __int64 v171; // rsi
  char v172; // al
  __int64 v173; // rcx
  __int64 v174; // r8
  __int64 v175; // r9
  char *v176; // rdx
  int v177; // eax
  int v178; // eax
  __int64 *v179; // rax
  __int64 v180; // r12
  __int64 v181; // rbx
  __int64 v182; // r13
  __int64 *v183; // rdx
  __int64 v184; // r13
  __int64 *v185; // r12
  int v186; // ebx
  __int64 v187; // rax
  __int64 v188; // r15
  __int64 v189; // r13
  int v190; // eax
  unsigned int v191; // esi
  int v192; // eax
  int v193; // eax
  int v194; // eax
  int v195; // r9d
  __int64 v196; // [rsp+30h] [rbp-3C0h]
  __int64 v197; // [rsp+40h] [rbp-3B0h]
  __int64 v198; // [rsp+48h] [rbp-3A8h]
  __int64 v199; // [rsp+58h] [rbp-398h]
  __int64 v200; // [rsp+68h] [rbp-388h]
  __int64 v201; // [rsp+70h] [rbp-380h]
  __int64 v202; // [rsp+78h] [rbp-378h]
  __int64 v204; // [rsp+90h] [rbp-360h]
  __int64 **v205; // [rsp+90h] [rbp-360h]
  __int64 v206; // [rsp+90h] [rbp-360h]
  __int64 v207; // [rsp+90h] [rbp-360h]
  __int64 v208; // [rsp+98h] [rbp-358h]
  __int64 v209; // [rsp+A0h] [rbp-350h]
  __int64 v210; // [rsp+A0h] [rbp-350h]
  __int64 v211; // [rsp+A0h] [rbp-350h]
  __int64 *v212; // [rsp+A8h] [rbp-348h]
  __int64 v213; // [rsp+A8h] [rbp-348h]
  __int64 v214; // [rsp+A8h] [rbp-348h]
  __int64 v215; // [rsp+A8h] [rbp-348h]
  __int64 v216; // [rsp+A8h] [rbp-348h]
  __int64 v217; // [rsp+A8h] [rbp-348h]
  __int64 *v218; // [rsp+A8h] [rbp-348h]
  __int64 v219; // [rsp+A8h] [rbp-348h]
  __int64 v220; // [rsp+A8h] [rbp-348h]
  __int64 *v221; // [rsp+A8h] [rbp-348h]
  __int64 v222; // [rsp+B0h] [rbp-340h]
  _BYTE *v223; // [rsp+B0h] [rbp-340h]
  unsigned __int8 v224; // [rsp+B8h] [rbp-338h]
  _BYTE *v225; // [rsp+B8h] [rbp-338h]
  __int64 v226; // [rsp+B8h] [rbp-338h]
  __int64 v227; // [rsp+B8h] [rbp-338h]
  __int64 v228; // [rsp+C0h] [rbp-330h] BYREF
  __int64 *v229; // [rsp+C8h] [rbp-328h] BYREF
  _QWORD *v230; // [rsp+D0h] [rbp-320h] BYREF
  char v231[8]; // [rsp+D8h] [rbp-318h] BYREF
  __int64 v232[4]; // [rsp+E0h] [rbp-310h] BYREF
  __m128i v233; // [rsp+100h] [rbp-2F0h] BYREF
  __m128i v234; // [rsp+110h] [rbp-2E0h] BYREF
  _OWORD v235[3]; // [rsp+120h] [rbp-2D0h] BYREF
  __int64 *v236; // [rsp+150h] [rbp-2A0h] BYREF
  __int64 v237; // [rsp+158h] [rbp-298h]
  _BYTE v238[32]; // [rsp+160h] [rbp-290h] BYREF
  __int64 *v239; // [rsp+180h] [rbp-270h] BYREF
  __int64 v240; // [rsp+188h] [rbp-268h]
  _BYTE v241[32]; // [rsp+190h] [rbp-260h] BYREF
  __int64 v242; // [rsp+1B0h] [rbp-240h] BYREF
  _QWORD *v243; // [rsp+1B8h] [rbp-238h]
  _QWORD *v244; // [rsp+1C0h] [rbp-230h]
  __int64 v245; // [rsp+1C8h] [rbp-228h]
  int v246; // [rsp+1D0h] [rbp-220h]
  _BYTE v247[40]; // [rsp+1D8h] [rbp-218h] BYREF
  const char *v248; // [rsp+200h] [rbp-1F0h] BYREF
  _QWORD *v249; // [rsp+208h] [rbp-1E8h]
  _QWORD *v250; // [rsp+210h] [rbp-1E0h]
  __int64 v251; // [rsp+218h] [rbp-1D8h]
  int v252; // [rsp+220h] [rbp-1D0h]
  _BYTE v253[40]; // [rsp+228h] [rbp-1C8h] BYREF
  __int64 *v254; // [rsp+250h] [rbp-1A0h] BYREF
  __int64 v255; // [rsp+258h] [rbp-198h]
  _BYTE v256[64]; // [rsp+260h] [rbp-190h] BYREF
  __int64 v257; // [rsp+2A0h] [rbp-150h]
  _BYTE *v258; // [rsp+2A8h] [rbp-148h]
  _BYTE *v259; // [rsp+2B0h] [rbp-140h]
  __int64 v260; // [rsp+2B8h] [rbp-138h]
  int v261; // [rsp+2C0h] [rbp-130h]
  _BYTE v262[72]; // [rsp+2C8h] [rbp-128h] BYREF
  __int64 v263; // [rsp+310h] [rbp-E0h] BYREF
  _BYTE *v264; // [rsp+318h] [rbp-D8h]
  _BYTE *v265; // [rsp+320h] [rbp-D0h]
  __int64 v266; // [rsp+328h] [rbp-C8h]
  int v267; // [rsp+330h] [rbp-C0h]
  _BYTE v268[184]; // [rsp+338h] [rbp-B8h] BYREF

  if ( !byte_4FB3BA0 )
    return 0;
  v224 = *(_BYTE *)(a2 + 16);
  if ( (unsigned __int8)(v224 - 35) > 0x2Cu )
    return 0;
  v12 = 0x1300000BFFFFLL;
  if ( !_bittest64(&v12, (unsigned int)v224 - 35) )
    return 0;
  v13 = a1;
  sub_19E5640((__int64)&v263, a3, a2);
  if ( !(_BYTE)v267 )
    return 0;
  if ( !(unsigned __int8)sub_19ECD70(a1, a2) )
    return 0;
  v257 = 0;
  v225 = v262;
  v258 = v262;
  v259 = v262;
  v260 = 8;
  v261 = 0;
  v14 = sub_19E6CE0(a1, a2);
  v198 = v14;
  if ( v14 )
  {
    v15 = *(_QWORD *)(v14 - 24);
    if ( *(_BYTE *)(v15 + 16) != 23 && *(_QWORD *)(v15 + 64) == *(_QWORD *)(a2 + 40) )
      return 0;
  }
  v263 = 0;
  v264 = v268;
  v265 = v268;
  v266 = 16;
  v267 = 0;
  v16 = sub_13CF970(a2);
  v19 = *(_DWORD *)(a2 + 20);
  v236 = (__int64 *)v238;
  v237 = 0x400000000LL;
  v20 = 3LL * (v19 & 0xFFFFFFF);
  v21 = 0xAAAAAAAAAAAAAAABLL * v20;
  v22 = 8 * v20;
  v23 = 0xAAAAAAAAAAAAAAABLL * v20;
  if ( v20 > 0xC )
  {
    sub_16CD150((__int64)&v236, v238, v21, 8, v17, v18);
    v24 = v236;
    v25 = v237;
    v26 = &v236[(unsigned int)v237];
  }
  else
  {
    v24 = (__int64 *)v238;
    v25 = 0;
    v26 = (__int64 *)v238;
  }
  if ( v22 )
  {
    v27 = (__int64 *)v16;
    do
    {
      v28 = *v27;
      ++v26;
      v27 += 3;
      *(v26 - 1) = v28;
      --v23;
    }
    while ( v23 );
    v24 = v236;
    v25 = v237;
  }
  LODWORD(v237) = v25 + v21;
  v29 = &v24[(unsigned int)(v25 + v21)];
  if ( v29 == v24 )
  {
    v34 = 0;
    v223 = v262;
    goto LABEL_164;
  }
  v30 = v24;
  v31 = 0;
  v32 = 0;
  do
  {
    v33 = *v30;
    if ( *(_BYTE *)(*v30 + 16) == 77
      || (v248 = (const char *)*v30, (unsigned __int8)sub_19E74C0(v13 + 1704, (__int64 *)&v248, &v254))
      && (v33 = v254[1]) != 0 )
    {
      v31 = v33;
      if ( v32 )
      {
        if ( v32 != sub_19E73A0(v13, v33) )
        {
          v24 = v236;
          v34 = 0;
          v223 = v259;
          v225 = v258;
          goto LABEL_164;
        }
      }
      else
      {
        v32 = sub_19E73A0(v13, v33);
      }
      if ( (*(_DWORD *)(v31 + 20) & 0xFFFFFFF) == 1 )
        v31 = 0;
    }
    ++v30;
  }
  while ( v29 != v30 );
  if ( !v31 )
  {
    v24 = v236;
    v34 = 0;
    v223 = v262;
    goto LABEL_164;
  }
  v242 = 0;
  v254 = (__int64 *)v256;
  v255 = 0x400000000LL;
  v243 = v247;
  v244 = v247;
  v245 = 4;
  v246 = 0;
  v228 = sub_19E73A0(v13, v31);
  v196 = v13 + 1864;
  v36 = sub_19F1A30(v13 + 1864, &v228);
  v37 = (__int64 *)v36[2];
  v38 = v36 + 2;
  v200 = v13 + 2392;
  if ( v37 != v36 + 2 )
  {
    v39 = sub_19E5210(v13 + 2392, a2);
    v40 = v36[1];
    v41 = v39;
    if ( v38 == (__int64 *)v40 )
    {
      v40 = *(_QWORD *)(v40 + 8);
      v43 = v39 >> 7;
      v36[1] = v40;
      v42 = *(_DWORD *)(v40 + 16);
      if ( v43 == v42 )
      {
        if ( v38 == (__int64 *)v40 )
          goto LABEL_39;
        goto LABEL_187;
      }
    }
    else
    {
      v42 = *(_DWORD *)(v40 + 16);
      v43 = v39 >> 7;
      if ( v43 == v42 )
      {
LABEL_187:
        *(_QWORD *)(v40 + 8LL * ((v41 >> 6) & 1) + 24) &= ~(1LL << v41);
        if ( !*(_QWORD *)(v40 + 24) && !*(_QWORD *)(v40 + 32) )
        {
          v154 = *(_QWORD *)v36[1];
          --v36[4];
          v36[1] = v154;
          sub_2208CA0(v40);
          j_j___libc_free_0(v40, 40);
        }
        goto LABEL_39;
      }
    }
    if ( v43 >= v42 )
    {
      if ( v38 == (__int64 *)v40 )
      {
LABEL_238:
        v36[1] = v40;
        goto LABEL_39;
      }
      while ( v43 > v42 )
      {
        v40 = *(_QWORD *)v40;
        if ( v38 == (__int64 *)v40 )
          goto LABEL_238;
        v42 = *(_DWORD *)(v40 + 16);
      }
    }
    else
    {
      if ( v37 == (__int64 *)v40 )
      {
        v36[1] = v40;
        goto LABEL_38;
      }
      do
        v40 = *(_QWORD *)(v40 + 8);
      while ( v37 != (__int64 *)v40 && v43 < *(_DWORD *)(v40 + 16) );
    }
    v36[1] = v40;
    if ( v38 == (__int64 *)v40 )
      goto LABEL_39;
LABEL_38:
    if ( *(_DWORD *)(v40 + 16) != v43 )
      goto LABEL_39;
    goto LABEL_187;
  }
LABEL_39:
  v44 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
  if ( !v44 )
  {
    v45 = (__int64 *)&v239;
    v223 = v262;
LABEL_176:
    sub_19E68D0(v45, &v242, v244);
    sub_19E54A0(&v248, &v242);
    v141 = v248;
    for ( i = v239; v239 != (__int64 *)v141; i = v239 )
    {
      sub_19EDA30(v13, *i, a2);
      ++v239;
      sub_19E4730((__int64)v45);
    }
    sub_19E3350(v13, v254, (unsigned int)v255);
    v34 = sub_19ED210(v13, (__int64 ****)v254, (unsigned int)v255, (__int64 ***)a2, v228);
    if ( (unsigned int)(*(_DWORD *)(v34 + 8) - 1) <= 1 )
      goto LABEL_122;
    v210 = v34;
    v208 = v13 + 1704;
    v143 = (const char *)sub_19E5730(v13 + 1704, a2);
    v144 = (__int64)v143;
    v217 = v13 + 1672;
    if ( v143 )
    {
      v248 = v143;
      v145 = sub_19F2860(v217, (__int64 *)&v248);
      v146 = v210;
      v145[1] = v228;
      v147 = v254;
      if ( v254 == &v254[2 * (unsigned int)v255] )
      {
LABEL_184:
        v219 = v146;
        v152 = sub_19F1A30(v196, &v228);
        v153 = sub_19E5210(v200, a2);
        sub_1369D60(v152 + 1, v153);
        v34 = v219;
        goto LABEL_122;
      }
      v218 = &v254[2 * (unsigned int)v255];
      v148 = 0;
      do
      {
        v149 = v148;
        v150 = v147[1];
        ++v148;
        v151 = sub_13CF970(v144);
        v147 += 2;
        sub_1593B40((_QWORD *)(v151 + 24 * v149), *(v147 - 2));
        *(_QWORD *)(sub_13CF970(v144) + 8 * (v149 + 3LL * *(unsigned int *)(v144 + 56)) + 8) = v150;
      }
      while ( v218 != v147 );
LABEL_183:
      v146 = v210;
      goto LABEL_184;
    }
    v248 = "phiofops";
    LOWORD(v250) = 259;
    v156 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
    v205 = *(__int64 ***)a2;
    v157 = sub_1648B60(64);
    v158 = v210;
    v159 = v157;
    if ( v157 )
    {
      sub_15F1EA0(v157, (__int64)v205, 53, 0, 0, 0);
      *(_DWORD *)(v159 + 56) = v156;
      sub_164B780(v159, (__int64 *)&v248);
      sub_1648880(v159, *(_DWORD *)(v159 + 56), 1);
      v158 = v210;
    }
    v206 = v158;
    v248 = (const char *)v159;
    v211 = v228;
    v160 = sub_19EC730(v200, (__int64 *)&v248);
    *(_DWORD *)(v160 + 8) = sub_19E5210(v200, a2);
    v239 = (__int64 *)v159;
    v161 = sub_1463A20(v13 + 1832, v45, &v248);
    v162 = (char *)v248;
    v163 = v206;
    if ( v161 )
    {
LABEL_216:
      v207 = v163;
      v239 = (__int64 *)v159;
      v164 = sub_19E72F0(v217, v45, &v248);
      v165 = (char *)v248;
      v166 = v207;
      if ( v164 )
        goto LABEL_222;
      v167 = *(_DWORD *)(v13 + 1696);
      v168 = *(_DWORD *)(v13 + 1688);
      ++*(_QWORD *)(v13 + 1672);
      v169 = v168 + 1;
      if ( 4 * v169 >= 3 * v167 )
      {
        v167 *= 2;
      }
      else if ( v167 - *(_DWORD *)(v13 + 1692) - v169 > v167 >> 3 )
      {
LABEL_219:
        *(_DWORD *)(v13 + 1688) = v169;
        if ( *(_QWORD *)v165 != -8 )
          --*(_DWORD *)(v13 + 1692);
        v170 = v239;
        *((_QWORD *)v165 + 1) = 0;
        *(_QWORD *)v165 = v170;
LABEL_222:
        v171 = (__int64)v45;
        v220 = v166;
        *((_QWORD *)v165 + 1) = v211;
        v239 = (__int64 *)a2;
        v172 = sub_19E74C0(v208, v45, &v248);
        v176 = (char *)v248;
        v146 = v220;
        if ( v172 )
        {
LABEL_228:
          *((_QWORD *)v176 + 1) = v159;
          v180 = v13 + 1536;
          v181 = v146;
          v182 = *(_QWORD *)(a2 + 8);
          if ( v182 )
          {
            do
            {
              v171 = (__int64)sub_1648700(v182);
              if ( *(_BYTE *)(v171 + 16) > 0x17u )
                sub_1412190(v180, v171);
              v182 = *(_QWORD *)(v182 + 8);
            }
            while ( v182 );
            v146 = v181;
          }
          v183 = v254;
          v184 = 2LL * (unsigned int)v255;
          v221 = &v254[v184];
          if ( v254 == &v254[v184] )
            goto LABEL_184;
          v210 = v146;
          v185 = v254;
          do
          {
            v188 = *v185;
            v189 = v185[1];
            v190 = *(_DWORD *)(v159 + 20) & 0xFFFFFFF;
            if ( v190 == *(_DWORD *)(v159 + 56) )
            {
              sub_15F55D0(v159, v171, (__int64)v183, v173, v174, v175);
              v190 = *(_DWORD *)(v159 + 20) & 0xFFFFFFF;
            }
            v185 += 2;
            v186 = (v190 + 1) & 0xFFFFFFF;
            *(_DWORD *)(v159 + 20) = v186 | *(_DWORD *)(v159 + 20) & 0xF0000000;
            v187 = sub_13CF970(v159);
            v171 = v188;
            sub_1593B40((_QWORD *)(v187 + 24LL * (unsigned int)(v186 - 1)), v188);
            v174 = sub_13CF970(v159);
            v173 = (*(_DWORD *)(v159 + 20) & 0xFFFFFFFu) - 1;
            *(_QWORD *)(v174 + 8 * (v173 + 3LL * *(unsigned int *)(v159 + 56)) + 8) = v189;
          }
          while ( v221 != v185 );
          goto LABEL_183;
        }
        v171 = *(unsigned int *)(v13 + 1728);
        v177 = *(_DWORD *)(v13 + 1720);
        ++*(_QWORD *)(v13 + 1704);
        v178 = v177 + 1;
        v174 = (unsigned int)(2 * v171);
        if ( 4 * v178 >= (unsigned int)(3 * v171) )
        {
          LODWORD(v171) = 2 * v171;
        }
        else
        {
          v173 = (unsigned int)(v171 - *(_DWORD *)(v13 + 1724) - v178);
          if ( (unsigned int)v173 > (unsigned int)v171 >> 3 )
          {
LABEL_225:
            *(_DWORD *)(v13 + 1720) = v178;
            if ( *(_QWORD *)v176 != -8 )
              --*(_DWORD *)(v13 + 1724);
            v179 = v239;
            *((_QWORD *)v176 + 1) = 0;
            *(_QWORD *)v176 = v179;
            goto LABEL_228;
          }
        }
        sub_19F2990(v208, v171);
        v171 = (__int64)v45;
        sub_19E74C0(v208, v45, &v248);
        v176 = (char *)v248;
        v146 = v220;
        v178 = *(_DWORD *)(v13 + 1720) + 1;
        goto LABEL_225;
      }
      sub_19F26D0(v217, v167);
      sub_19E72F0(v217, v45, &v248);
      v165 = (char *)v248;
      v166 = v207;
      v169 = *(_DWORD *)(v13 + 1688) + 1;
      goto LABEL_219;
    }
    v191 = *(_DWORD *)(v13 + 1856);
    v192 = *(_DWORD *)(v13 + 1848);
    ++*(_QWORD *)(v13 + 1832);
    v193 = v192 + 1;
    if ( 4 * v193 >= 3 * v191 )
    {
      v191 *= 2;
    }
    else if ( v191 - *(_DWORD *)(v13 + 1852) - v193 > v191 >> 3 )
    {
LABEL_242:
      *(_DWORD *)(v13 + 1848) = v193;
      if ( *(_QWORD *)v162 != -8 )
        --*(_DWORD *)(v13 + 1852);
      *(_QWORD *)v162 = v239;
      goto LABEL_216;
    }
    sub_1467110(v13 + 1832, v191);
    sub_1463A20(v13 + 1832, v45, &v248);
    v162 = (char *)v248;
    v163 = v206;
    v193 = *(_DWORD *)(v13 + 1848) + 1;
    goto LABEL_242;
  }
  v209 = 0;
  v199 = v13 + 2200;
  v45 = (__int64 *)&v239;
  v197 = v13 + 1832;
  v46 = (__int64 **)v235;
  v204 = v31;
  v226 = v13;
  while ( 1 )
  {
    if ( (*(_BYTE *)(v204 + 23) & 0x40) != 0 )
      v47 = *(_QWORD *)(v204 - 8);
    else
      v47 = v204 - 24LL * v44;
    v48 = *(__int64 **)(v47 + 8 * v209 + 24LL * *(unsigned int *)(v204 + 56) + 8);
    v248 = 0;
    v249 = v253;
    v202 = (__int64)v48;
    v239 = v48;
    v250 = v253;
    v251 = 4;
    v252 = 0;
    v240 = v228;
    v49 = sub_19E8F30(v199, v45, v46);
    if ( !v49 )
    {
      v50 = sub_1599EF0(*(__int64 ***)a2);
      v51 = sub_19F1A30(v196, &v228);
      v52 = sub_19E5210(v200, a2);
      sub_1369D60(v51 + 1, v52);
      goto LABEL_44;
    }
    v57 = sub_15F4880(a2);
    v201 = v57;
    if ( v198 )
    {
      v239 = (__int64 *)v57;
      v240 = v198;
      v58 = v226 + 1800;
      v59 = sub_19E6C30(v226 + 1800, v45, v46);
      v60 = *(__int64 ***)&v235[0];
      if ( !v59 )
      {
        v136 = *(_DWORD *)(v226 + 1816);
        v137 = *(_DWORD *)(v226 + 1824);
        ++*(_QWORD *)(v226 + 1800);
        v138 = v136 + 1;
        if ( 4 * v138 >= 3 * v137 )
        {
          v137 *= 2;
        }
        else if ( v137 - *(_DWORD *)(v226 + 1820) - v138 > v137 >> 3 )
        {
LABEL_160:
          *(_DWORD *)(v226 + 1816) = v138;
          if ( *v60 != (__int64 *)-8LL )
            --*(_DWORD *)(v226 + 1820);
          *v60 = v239;
          v60[1] = (__int64 *)v240;
          goto LABEL_52;
        }
        sub_19F3280(v58, v137);
        sub_19E6C30(v58, v45, v46);
        v60 = *(__int64 ***)&v235[0];
        v138 = *(_DWORD *)(v226 + 1816) + 1;
        goto LABEL_160;
      }
    }
LABEL_52:
    sub_18CE100((__int64)&v263);
    if ( (*(_BYTE *)(v201 + 23) & 0x40) != 0 )
    {
      v61 = *(_QWORD *)(v201 - 8);
      v222 = v61 + 24LL * (*(_DWORD *)(v201 + 20) & 0xFFFFFFF);
    }
    else
    {
      v222 = v201;
      v61 = v201 - 24LL * (*(_DWORD *)(v201 + 20) & 0xFFFFFFF);
    }
    if ( v61 != v222 )
      break;
LABEL_87:
    v229 = (__int64 *)v201;
    v83 = sub_19E5210(v200, a2);
    v84 = sub_1463A20(v197, (__int64 *)&v229, v45);
    v85 = v239;
    if ( v84 )
      goto LABEL_88;
    v133 = *(_DWORD *)(v226 + 1848);
    v134 = *(_DWORD *)(v226 + 1856);
    ++*(_QWORD *)(v226 + 1832);
    v135 = v133 + 1;
    if ( 4 * v135 >= 3 * v134 )
    {
      v134 *= 2;
LABEL_200:
      sub_1467110(v197, v134);
      sub_1463A20(v197, (__int64 *)&v229, v45);
      v85 = v239;
      v135 = *(_DWORD *)(v226 + 1848) + 1;
      goto LABEL_155;
    }
    if ( v134 - *(_DWORD *)(v226 + 1852) - v135 <= v134 >> 3 )
      goto LABEL_200;
LABEL_155:
    *(_DWORD *)(v226 + 1848) = v135;
    if ( *v85 != -8 )
      --*(_DWORD *)(v226 + 1852);
    *v85 = (__int64)v229;
LABEL_88:
    v86 = v226 + 1672;
    v239 = v229;
    v240 = v202;
    v87 = sub_19E72F0(v226 + 1672, v45, v46);
    v88 = *(__int64 ***)&v235[0];
    if ( !v87 )
    {
      v130 = *(_DWORD *)(v226 + 1688);
      v131 = *(_DWORD *)(v226 + 1696);
      ++*(_QWORD *)(v226 + 1672);
      v132 = v130 + 1;
      if ( 4 * v132 >= 3 * v131 )
      {
        v131 *= 2;
      }
      else if ( v131 - *(_DWORD *)(v226 + 1692) - v132 > v131 >> 3 )
      {
LABEL_150:
        *(_DWORD *)(v226 + 1688) = v132;
        if ( *v88 != (__int64 *)-8LL )
          --*(_DWORD *)(v226 + 1692);
        *v88 = v239;
        v88[1] = (__int64 *)v240;
        goto LABEL_89;
      }
      sub_19F26D0(v86, v131);
      sub_19E72F0(v86, v45, v46);
      v88 = *(__int64 ***)&v235[0];
      v132 = *(_DWORD *)(v226 + 1688) + 1;
      goto LABEL_150;
    }
LABEL_89:
    LODWORD(v240) = v83;
    v239 = v229;
    if ( !(unsigned __int8)sub_154CC80(v200, v45, v46) )
    {
      v129 = sub_19EC6A0(v200, v45, *(_QWORD **)&v235[0]);
      *v129 = v239;
      *((_DWORD *)v129 + 2) = v240;
    }
    v230 = sub_19EEAC0(v226, (__int64)v229);
    *(_QWORD *)&v235[0] = v229;
    if ( (unsigned __int8)sub_154CC80(v200, (__int64 *)v46, v45) )
    {
      *v239 = -16;
      --*(_DWORD *)(v226 + 2408);
      ++*(_DWORD *)(v226 + 2412);
    }
    if ( (unsigned __int8)sub_1463A20(v197, (__int64 *)&v229, v45) )
    {
      *v239 = -16;
      --*(_DWORD *)(v226 + 1848);
      ++*(_DWORD *)(v226 + 1852);
    }
    v89 = (__int64 *)v46;
    *(_QWORD *)&v235[0] = v229;
    v90 = sub_19E72F0(v86, (__int64 *)v46, v45);
    v94 = (__int64)v239;
    if ( v90 )
    {
      *v239 = -16;
      --*(_DWORD *)(v226 + 1688);
      ++*(_DWORD *)(v226 + 1692);
    }
    if ( v198 )
    {
      v89 = (__int64 *)&v229;
      v95 = sub_19E6C30(v226 + 1800, (__int64 *)&v229, v45);
      v94 = (__int64)v239;
      if ( v95 )
      {
        *v239 = -16;
        --*(_DWORD *)(v226 + 1816);
        ++*(_DWORD *)(v226 + 1820);
      }
    }
    v96 = v230;
    if ( !v230 )
    {
LABEL_171:
      v115 = v226;
      v116 = v46;
      sub_164BEC0(v201, (__int64)v89, v94, v91, a4, *(double *)a5.m128i_i64, a6, a7, v92, v93, a10, a11);
      goto LABEL_172;
    }
    v97 = *((_DWORD *)v230 + 2);
    if ( v97 == 1 )
      goto LABEL_102;
    if ( v97 != 2 )
      goto LABEL_195;
    v50 = v230[3];
    if ( *(_BYTE *)(v50 + 16) <= 0x11u )
      goto LABEL_103;
    v106 = *(_QWORD *)(v226 + 8);
    v89 = (__int64 *)sub_19E73A0(v226, v230[3]);
    if ( sub_15CC8F0(v106, (__int64)v89, v202) )
    {
LABEL_102:
      v50 = v96[3];
      if ( !v50 )
        goto LABEL_141;
      goto LABEL_103;
    }
    v97 = *((_DWORD *)v96 + 2);
    if ( v97 == 2 )
    {
      v107 = *(_DWORD *)(v226 + 1496);
      if ( !v107 )
        goto LABEL_141;
      v91 = v96[3];
      v108 = v107 - 1;
      v89 = *(__int64 **)(v226 + 1480);
      v94 = (v107 - 1) & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
      v109 = &v89[2 * v94];
      v110 = *v109;
      if ( v91 != *v109 )
      {
        v194 = 1;
        while ( v110 != -8 )
        {
          v195 = v194 + 1;
          v94 = v108 & (unsigned int)(v194 + v94);
          v109 = &v89[2 * (unsigned int)v94];
          v110 = *v109;
          if ( v91 == *v109 )
            goto LABEL_137;
          v194 = v195;
        }
LABEL_141:
        v115 = v226;
        v116 = v46;
        v117 = v226 + 1768;
        v118 = sub_19EB1D0(v226 + 1768, (__int64 *)&v230, v45);
        v119 = v239;
        if ( !v118 )
        {
          v120 = *(_DWORD *)(v226 + 1784);
          v121 = *(_DWORD *)(v226 + 1792);
          ++*(_QWORD *)(v226 + 1768);
          v122 = v120 + 1;
          if ( 4 * (v120 + 1) >= 3 * v121 )
          {
            v121 *= 2;
          }
          else if ( v121 - *(_DWORD *)(v226 + 1788) - v122 > v121 >> 3 )
          {
            goto LABEL_144;
          }
          sub_19F3110(v117, v121);
          sub_19EB1D0(v117, (__int64 *)&v230, v45);
          v119 = v239;
          v122 = *(_DWORD *)(v226 + 1784) + 1;
LABEL_144:
          *(_DWORD *)(v226 + 1784) = v122;
          if ( *v119 != -8 )
            --*(_DWORD *)(v226 + 1788);
          v123 = v230;
          v119[1] = 0;
          v119[4] = 2;
          *v119 = (__int64)v123;
          v119[2] = (__int64)(v119 + 6);
          v119[3] = (__int64)(v119 + 6);
          *((_DWORD *)v119 + 10) = 0;
        }
        v124 = (__int64)(v119 + 1);
        sub_165A590((__int64)v45, (__int64)(v119 + 1), a2);
        sub_164BEC0(v201, v124, v125, v126, a4, *(double *)a5.m128i_i64, a6, a7, v127, v128, a10, a11);
LABEL_172:
        sub_19E68D0(v116, (__int64 *)&v248, v250);
        sub_19E54A0(v45, (__int64 *)&v248);
        v139 = v239;
        for ( j = *(__int64 **)&v235[0]; v139 != *(__int64 **)&v235[0]; j = *(__int64 **)&v235[0] )
        {
          sub_19EDA30(v115, *j, a2);
          *(_QWORD *)&v235[0] += 8LL;
          sub_19E4730((__int64)v116);
        }
        goto LABEL_119;
      }
    }
    else
    {
LABEL_195:
      if ( v97 == 3 )
      {
        v111 = *(_QWORD *)(v226 + 1432);
        goto LABEL_138;
      }
      v89 = (__int64 *)v46;
      *(_QWORD *)&v235[0] = v96;
      if ( !(unsigned __int8)sub_19E3400(v226 + 2056, (__int64 *)v46, v45) )
        goto LABEL_141;
      v109 = v239;
    }
LABEL_137:
    v111 = v109[1];
LABEL_138:
    if ( !v111 )
      goto LABEL_141;
    v50 = *(_QWORD *)(v111 + 8);
    if ( *(_BYTE *)(v50 + 16) <= 0x11u )
      goto LABEL_104;
    v112 = (__int64 *)(v111 + 56);
    sub_19E68D0(v46, (__int64 *)(v111 + 56), *(_QWORD **)(v111 + 72));
    v89 = v112;
    sub_19E54A0(v45, v112);
    v113 = v239;
    v114 = *(__int64 **)&v235[0];
    if ( v239 == *(__int64 **)&v235[0] )
      goto LABEL_141;
    while ( 1 )
    {
      v50 = *v114;
      if ( *(_BYTE *)(*v114 + 16) <= 0x17u )
        break;
      if ( a2 != v50 )
      {
        v155 = *(_QWORD *)(v226 + 8);
        v89 = (__int64 *)sub_19E73A0(v226, *v114);
        if ( sub_15CC8F0(v155, (__int64)v89, v202) )
          break;
      }
      *(_QWORD *)&v235[0] += 8LL;
      sub_19E4730((__int64)v46);
      v114 = *(__int64 **)&v235[0];
      if ( v113 == *(__int64 **)&v235[0] )
        goto LABEL_141;
    }
LABEL_103:
    if ( *(_BYTE *)(v50 + 16) == 55 )
    {
      v50 = *(_QWORD *)(v50 - 48);
      if ( !v50 )
        goto LABEL_171;
    }
LABEL_104:
    sub_164BEC0(v201, (__int64)v89, v94, v91, a4, *(double *)a5.m128i_i64, a6, a7, v92, v93, a10, a11);
    sub_19E54A0(v232, (__int64 *)&v248);
    sub_19E68D0(&v233, (__int64 *)&v248, v250);
    a4 = (__m128)_mm_load_si128(&v233);
    a5 = _mm_load_si128(&v234);
    v98 = v232[0];
    v99 = (__int64 *)v233.m128i_i64[0];
    v235[0] = a4;
    v235[1] = a5;
    if ( v232[0] != v233.m128i_i64[0] )
    {
      do
      {
        sub_19E5640((__int64)v45, (__int64)&v242, *v99);
        *(_QWORD *)&v235[0] += 8LL;
        sub_19E4730((__int64)v46);
        v99 = *(__int64 **)&v235[0];
      }
      while ( v98 != *(_QWORD *)&v235[0] );
      v55 = (unsigned int)v255;
      if ( (unsigned int)v255 < HIDWORD(v255) )
        goto LABEL_45;
LABEL_107:
      sub_16CD150((__int64)&v254, v256, 0, 16, v53, v54);
      v55 = (unsigned int)v255;
      goto LABEL_45;
    }
LABEL_44:
    v55 = (unsigned int)v255;
    if ( (unsigned int)v255 >= HIDWORD(v255) )
      goto LABEL_107;
LABEL_45:
    v56 = &v254[2 * v55];
    *v56 = v50;
    v56[1] = v202;
    LODWORD(v255) = v255 + 1;
    if ( v250 != v249 )
      _libc_free((unsigned __int64)v250);
    ++v209;
    v44 = *(_DWORD *)(v204 + 20) & 0xFFFFFFF;
    if ( v44 <= (unsigned int)v209 )
    {
      v13 = v226;
      v31 = v204;
      v223 = v259;
      v225 = v258;
      goto LABEL_176;
    }
  }
  v212 = (__int64 *)v46;
  for ( k = v61 + 24; ; k += 24 )
  {
    v64 = *(_QWORD *)(k - 24);
    if ( *(_BYTE *)(v64 + 16) == 77 )
    {
      v66 = sub_16497E0(*(_QWORD *)(k - 24), v228, v202);
      if ( *(_QWORD *)(k - 24) )
      {
        v71 = *(_QWORD *)(k - 16);
        v72 = *(_QWORD *)(k - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v72 = v71;
        if ( v71 )
          *(_QWORD *)(v71 + 16) = *(_QWORD *)(v71 + 16) & 3LL | v72;
      }
      *(_QWORD *)(k - 24) = v66;
      if ( !v66 )
        goto LABEL_74;
      v73 = *(_QWORD *)(v66 + 8);
      v65 = v66 + 8;
      *(_QWORD *)(k - 16) = v73;
      if ( v73 )
        *(_QWORD *)(v73 + 16) = (k - 16) | *(_QWORD *)(v73 + 16) & 3LL;
      *(_QWORD *)(k - 8) = v65 | *(_QWORD *)(k - 8) & 3LL;
      *(_QWORD *)(v66 + 8) = k - 24;
      v66 = *(_QWORD *)(k - 24);
      if ( v64 != v66 && a2 != v66 )
      {
LABEL_74:
        v65 = (__int64)&v248;
        sub_19E5640((__int64)v45, (__int64)&v248, v66);
      }
    }
    else
    {
      v65 = (__int64)v212;
      *(_QWORD *)&v235[0] = *(_QWORD *)(k - 24);
      if ( (unsigned __int8)sub_19E74C0(v226 + 1704, v212, v45) )
      {
        if ( v239[1] )
        {
          v65 = v239[1];
          v70 = sub_19E73A0(v226, v65);
          if ( v228 == v70 )
          {
            v100 = 0x17FFFFFFE8LL;
            v101 = *(_DWORD *)(v65 + 20) & 0xFFFFFFF;
            if ( v101 )
            {
              v102 = 24LL * *(unsigned int *)(v65 + 56) + 8;
              v103 = 0;
              do
              {
                v104 = v65 - 24LL * v101;
                if ( (*(_BYTE *)(v65 + 23) & 0x40) != 0 )
                  v104 = *(_QWORD *)(v65 - 8);
                if ( v202 == *(_QWORD *)(v104 + v102) )
                {
                  v100 = 24 * v103;
                  goto LABEL_115;
                }
                ++v103;
                v102 += 8;
              }
              while ( v101 != (_DWORD)v103 );
              v100 = 0x17FFFFFFE8LL;
            }
LABEL_115:
            v65 = *(_QWORD *)(sub_13CF970(v65) + v100);
            sub_1593B40((_QWORD *)(k - 24), v65);
          }
        }
      }
    }
    if ( !v49 )
      goto LABEL_64;
    v63 = *(_QWORD *)(k - 24);
    if ( v64 == v63 )
      break;
LABEL_57:
    if ( k == v222 )
      goto LABEL_86;
LABEL_58:
    ;
  }
  v74 = v228;
  v239 = (__int64 *)v241;
  v240 = 0x400000000LL;
  if ( *(_BYTE *)(v63 + 16) > 0x17u )
  {
    v65 = v63;
    if ( (unsigned __int8)sub_19F2E20(v226, v63, v228, (__int64)&v263, (__int64)v45) )
    {
      v75 = v239;
      v76 = v240;
      goto LABEL_78;
    }
LABEL_128:
    v82 = (unsigned __int64)v239;
    v49 = 0;
    if ( v239 != (__int64 *)v241 )
      goto LABEL_85;
LABEL_64:
    if ( k == v222 )
      goto LABEL_118;
    v49 = 0;
    goto LABEL_58;
  }
  v75 = (__int64 *)v241;
  v76 = 0;
LABEL_78:
  v77 = v45;
  v78 = k;
  v79 = (__int64)v77;
LABEL_82:
  v80 = &v75[v76];
  while ( v76 )
  {
    v65 = *(v80 - 1);
    --v76;
    --v80;
    LODWORD(v240) = v76;
    if ( *(_BYTE *)(v65 + 16) > 0x17u )
    {
      if ( (unsigned __int8)sub_19F2E20(v226, v65, v74, (__int64)&v263, v79) )
      {
        v75 = v239;
        v76 = v240;
        goto LABEL_82;
      }
      v105 = (__int64 *)v79;
      k = v78;
      v45 = v105;
      goto LABEL_128;
    }
  }
  v81 = (__int64 *)v79;
  k = v78;
  v45 = v81;
  v230 = (_QWORD *)v63;
  v65 = v226 + 1640;
  v231[0] = 1;
  sub_19F2CB0((__int64)v212, v226 + 1640, (__int64 *)&v230, v231);
  v82 = (unsigned __int64)v239;
  if ( v239 == (__int64 *)v241 )
    goto LABEL_57;
LABEL_85:
  _libc_free(v82);
  if ( k != v222 )
    goto LABEL_58;
LABEL_86:
  v46 = (__int64 **)v212;
  if ( v49 )
    goto LABEL_87;
LABEL_118:
  sub_164BEC0(v201, v65, v66, v67, a4, *(double *)a5.m128i_i64, a6, a7, v68, v69, a10, a11);
LABEL_119:
  if ( v250 != v249 )
    _libc_free((unsigned __int64)v250);
  v34 = 0;
  v223 = v259;
  v225 = v258;
LABEL_122:
  if ( v244 != v243 )
  {
    v213 = v34;
    _libc_free((unsigned __int64)v244);
    v34 = v213;
  }
  if ( v254 != (__int64 *)v256 )
  {
    v214 = v34;
    _libc_free((unsigned __int64)v254);
    v34 = v214;
  }
  v24 = v236;
LABEL_164:
  if ( v24 != (__int64 *)v238 )
  {
    v215 = v34;
    _libc_free((unsigned __int64)v24);
    v34 = v215;
  }
  if ( v265 != v264 )
  {
    v216 = v34;
    _libc_free((unsigned __int64)v265);
    v34 = v216;
  }
  if ( v225 != v223 )
  {
    v227 = v34;
    _libc_free((unsigned __int64)v223);
    return v227;
  }
  return v34;
}
