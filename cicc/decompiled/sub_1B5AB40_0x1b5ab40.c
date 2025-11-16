// Function: sub_1B5AB40
// Address: 0x1b5ab40
//
_BOOL8 __fastcall sub_1B5AB40(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __m128 a4,
        __m128i a5,
        __m128i a6,
        __m128i a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // rbx
  int v12; // r8d
  int v13; // r9d
  __int64 v14; // r13
  __int64 v15; // r14
  __int64 v16; // rbx
  signed __int64 v17; // rbx
  _QWORD *v18; // r12
  _QWORD *v19; // rax
  _BYTE *v20; // rdi
  int v21; // ebx
  _BOOL4 v22; // r13d
  __int64 v23; // rdi
  unsigned __int64 v24; // r12
  unsigned __int64 *v26; // rax
  double v27; // xmm4_8
  double v28; // xmm5_8
  __int64 v29; // rax
  bool v30; // bl
  __int64 v31; // r8
  int v32; // r9d
  const __m128i *v33; // rax
  __int64 v34; // rcx
  __int64 v35; // rdx
  unsigned int v36; // ebx
  int v37; // r12d
  const __m128i *j; // r13
  __int64 v39; // r15
  __m128i *v40; // r13
  __int64 v41; // rdx
  _QWORD *v42; // rsi
  __m128i *v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rdx
  __m128i *v46; // roff
  __int64 v47; // r13
  __int64 v48; // r12
  __int64 v49; // rbx
  __m128i *v50; // r12
  __m128i *v51; // rsi
  __int64 v52; // rax
  int *v53; // r15
  _QWORD *v54; // rbx
  int *v55; // r12
  unsigned __int64 *v56; // rbx
  int v57; // eax
  int *v58; // r13
  unsigned __int64 *v59; // r13
  bool v60; // al
  int v61; // r15d
  int v62; // r12d
  __int64 v63; // rdx
  _QWORD *v64; // rsi
  __m128i *v65; // r13
  __m128i *v66; // rax
  __int64 v67; // rdx
  __m128i *v68; // roff
  __int64 v69; // rcx
  unsigned __int64 *v70; // rax
  unsigned __int64 *v71; // rdx
  unsigned __int64 v72; // rcx
  int v73; // r8d
  int v74; // r9d
  __int64 v75; // rax
  __int64 v76; // r8
  __int64 v77; // rax
  __int64 v78; // r15
  __int64 v79; // rbx
  __int64 v80; // r14
  __m128i *v81; // r12
  int v82; // r8d
  int v83; // r9d
  const __m128i *v84; // rsi
  __int64 v85; // rax
  int *v86; // r13
  int v87; // eax
  __int64 v88; // rcx
  __int64 v89; // rdx
  int v90; // eax
  __int64 v91; // rdx
  int *v92; // rdi
  __int64 v93; // rax
  __int64 *v94; // rax
  unsigned __int64 v95; // rsi
  __int64 *v96; // r13
  __int64 v97; // rcx
  __int64 v98; // rdx
  __int64 v99; // rax
  unsigned __int64 v100; // r12
  __int64 *v101; // rsi
  __int64 *v102; // rax
  __int64 *v103; // rdx
  _BOOL8 v104; // rdi
  __int64 v105; // rax
  __int64 v106; // r12
  const __m128i *v107; // rsi
  __int64 v108; // rax
  unsigned __int64 v109; // r13
  __int64 *v110; // rax
  __int64 *v111; // r10
  __int64 v112; // rcx
  __int64 v113; // rdx
  __int64 v114; // rax
  __int64 *v115; // rax
  __int64 *v116; // rdx
  _BOOL8 v117; // rdi
  __int64 v118; // rax
  __int64 v119; // rdi
  __int64 *v120; // r13
  __int64 *v121; // rbx
  __int64 v122; // rdi
  __int64 v123; // rbx
  __int64 v124; // rax
  __int64 v125; // r13
  __int64 v126; // rax
  __int64 *v127; // rbx
  __int64 v128; // rax
  __int64 v129; // rcx
  __int64 *v130; // r14
  __int64 v131; // rcx
  __int64 v132; // r8
  __int64 v133; // r9
  __int64 v134; // rsi
  __m128i *v135; // rbx
  const __m128i *k; // r14
  __int64 v137; // rdx
  __int64 v138; // rsi
  __int64 v139; // r12
  __int64 v140; // r14
  __int64 v141; // rbx
  __int64 v142; // rdx
  __int64 v143; // rdx
  char v144; // al
  __int64 v145; // rdi
  _QWORD *v146; // rdi
  __int64 v147; // rsi
  unsigned __int8 *v148; // rsi
  __int64 v149; // rcx
  _QWORD *v150; // rax
  _QWORD *v151; // rdi
  __int64 *v152; // r12
  __int64 *v153; // r14
  __int64 v154; // rdi
  __int64 *v155; // rdi
  int v156; // r8d
  int v157; // r9d
  __int64 v158; // r15
  unsigned __int64 *v159; // rbx
  int v160; // eax
  __int64 *v161; // rsi
  __int64 v162; // rax
  __int64 *v163; // rsi
  __int64 v164; // rdx
  unsigned __int64 *v165; // rcx
  unsigned __int64 v166; // rdx
  _BYTE *v167; // rax
  unsigned __int64 v168; // r13
  _BYTE *i; // rdx
  unsigned __int64 v170; // rbx
  unsigned __int64 *v171; // rax
  unsigned __int64 *v172; // rdx
  __int64 *v173; // rdi
  __int64 v174; // rax
  __int64 v175; // rax
  __int64 v176; // r13
  __int64 v177; // rax
  unsigned __int64 *v178; // rdx
  unsigned __int64 v179; // [rsp+0h] [rbp-380h]
  int *v180; // [rsp+20h] [rbp-360h]
  __int64 v181; // [rsp+20h] [rbp-360h]
  __int64 v182; // [rsp+38h] [rbp-348h]
  __int64 v183; // [rsp+48h] [rbp-338h]
  __int64 v184; // [rsp+50h] [rbp-330h]
  unsigned __int64 *v185; // [rsp+50h] [rbp-330h]
  __int64 v187; // [rsp+68h] [rbp-318h]
  unsigned int v188; // [rsp+70h] [rbp-310h]
  unsigned __int64 v189; // [rsp+70h] [rbp-310h]
  unsigned __int64 v190; // [rsp+78h] [rbp-308h]
  int v191; // [rsp+78h] [rbp-308h]
  __int64 v192; // [rsp+78h] [rbp-308h]
  __int64 *v193; // [rsp+78h] [rbp-308h]
  unsigned int v194; // [rsp+90h] [rbp-2F0h]
  unsigned __int64 v195; // [rsp+90h] [rbp-2F0h]
  unsigned __int64 v196; // [rsp+90h] [rbp-2F0h]
  __int64 *v197; // [rsp+90h] [rbp-2F0h]
  __int64 v198; // [rsp+90h] [rbp-2F0h]
  unsigned __int64 v199; // [rsp+90h] [rbp-2F0h]
  __int64 *v200; // [rsp+90h] [rbp-2F0h]
  __int64 *v201; // [rsp+90h] [rbp-2F0h]
  __int64 v202; // [rsp+98h] [rbp-2E8h]
  __int64 v203; // [rsp+98h] [rbp-2E8h]
  bool v204; // [rsp+A6h] [rbp-2DAh]
  bool v205; // [rsp+A7h] [rbp-2D9h]
  __int64 *v207; // [rsp+B0h] [rbp-2D0h]
  __m128i v209; // [rsp+C0h] [rbp-2C0h] BYREF
  __int64 v210; // [rsp+D0h] [rbp-2B0h] BYREF
  __int64 v211; // [rsp+D8h] [rbp-2A8h]
  __int64 v212; // [rsp+E0h] [rbp-2A0h]
  const __m128i *v213; // [rsp+F0h] [rbp-290h] BYREF
  __m128i *v214; // [rsp+F8h] [rbp-288h]
  const __m128i *v215; // [rsp+100h] [rbp-280h]
  unsigned __int64 v216; // [rsp+110h] [rbp-270h] BYREF
  int v217; // [rsp+118h] [rbp-268h] BYREF
  int *v218; // [rsp+120h] [rbp-260h]
  int *v219; // [rsp+128h] [rbp-258h]
  int *v220; // [rsp+130h] [rbp-250h]
  __int64 v221; // [rsp+138h] [rbp-248h]
  __int64 *v222; // [rsp+140h] [rbp-240h] BYREF
  __int64 v223; // [rsp+148h] [rbp-238h] BYREF
  __int64 *v224; // [rsp+150h] [rbp-230h] BYREF
  __int64 *v225; // [rsp+158h] [rbp-228h]
  __int64 *v226; // [rsp+160h] [rbp-220h]
  __int64 v227; // [rsp+168h] [rbp-218h]
  __int64 *v228; // [rsp+170h] [rbp-210h] BYREF
  __int64 v229; // [rsp+178h] [rbp-208h]
  _BYTE v230[64]; // [rsp+180h] [rbp-200h] BYREF
  unsigned __int64 *v231; // [rsp+1C0h] [rbp-1C0h] BYREF
  __int64 v232; // [rsp+1C8h] [rbp-1B8h]
  _BYTE v233[64]; // [rsp+1D0h] [rbp-1B0h] BYREF
  _BYTE *v234; // [rsp+210h] [rbp-170h] BYREF
  __int64 v235; // [rsp+218h] [rbp-168h]
  _BYTE v236[64]; // [rsp+220h] [rbp-160h] BYREF
  __int64 v237; // [rsp+260h] [rbp-120h] BYREF
  __int64 v238; // [rsp+268h] [rbp-118h]
  __int64 v239; // [rsp+270h] [rbp-110h] BYREF
  __int64 *v240; // [rsp+290h] [rbp-F0h] BYREF
  __int64 v241; // [rsp+298h] [rbp-E8h]
  _BYTE v242[32]; // [rsp+2A0h] [rbp-E0h] BYREF
  _BYTE *v243; // [rsp+2C0h] [rbp-C0h] BYREF
  __int64 v244; // [rsp+2C8h] [rbp-B8h]
  _BYTE v245[176]; // [rsp+2D0h] [rbp-B0h] BYREF

  v11 = *(_QWORD *)(a2 + 40);
  v187 = v11;
  v207 = sub_1B44C50(a1, a2);
  v243 = *(_BYTE **)(v11 + 8);
  sub_15CDD40((__int64 *)&v243);
  v14 = (__int64)v243;
  v243 = v245;
  v244 = 0x1000000000LL;
  if ( v14 )
  {
    v15 = v14;
    v16 = 0;
    while ( 1 )
    {
      v15 = *(_QWORD *)(v15 + 8);
      if ( !v15 )
        break;
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v15) + 16) - 25) <= 9u )
      {
        v15 = *(_QWORD *)(v15 + 8);
        ++v16;
        if ( !v15 )
          goto LABEL_6;
      }
    }
LABEL_6:
    v17 = v16 + 1;
    if ( v17 > 16 )
    {
      sub_16CD150((__int64)&v243, v245, v17, 8, v12, v13);
      v18 = &v243[8 * (unsigned int)v244];
    }
    else
    {
      v18 = v245;
    }
    v19 = sub_1648700(v14);
LABEL_11:
    if ( v18 )
      *v18 = v19[5];
    while ( 1 )
    {
      v14 = *(_QWORD *)(v14 + 8);
      if ( !v14 )
        break;
      v19 = sub_1648700(v14);
      if ( (unsigned __int8)(*((_BYTE *)v19 + 16) - 25) <= 9u )
      {
        ++v18;
        goto LABEL_11;
      }
    }
    v20 = v243;
    v21 = v244 + v17;
  }
  else
  {
    v20 = v245;
    v21 = 0;
  }
  LODWORD(v244) = v21;
  v22 = 0;
  while ( 1 )
  {
    if ( !v21 )
      goto LABEL_19;
    v23 = *(_QWORD *)&v20[8 * v21 - 8];
    LODWORD(v244) = v21 - 1;
    v202 = v23;
    v24 = sub_157EBA0(v23);
    v205 = (a2 != v24) & (v207 == sub_1B44C50(a1, v24));
    if ( v205 )
      break;
LABEL_17:
    v20 = v243;
    v21 = v244;
  }
  v26 = (unsigned __int64 *)&v239;
  v237 = 0;
  v238 = 1;
  do
    *v26++ = -8;
  while ( v26 != (unsigned __int64 *)&v240 );
  v240 = (__int64 *)v242;
  v241 = 0x400000000LL;
  if ( (unsigned __int8)sub_1B5A690(a2, v24, (__int64)&v237) )
  {
    if ( *(_BYTE *)(a2 + 16) == 26
      && (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) == 3
      && (unsigned int)sub_1C105D0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL), *(_QWORD *)(a2 - 72), 0)
      || *(_BYTE *)(v24 + 16) == 26
      && (*(_DWORD *)(v24 + 20) & 0xFFFFFFF) == 3
      && (unsigned int)sub_1C105D0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL), *(_QWORD *)(v24 - 72), 0) )
    {
      if ( v240 != (__int64 *)v242 )
        _libc_free((unsigned __int64)v240);
      if ( (v238 & 1) == 0 )
        j___libc_free_0(v239);
      goto LABEL_17;
    }
LABEL_29:
    v210 = 0;
    v211 = 0;
    v212 = 0;
    v29 = sub_1B449D0(a1, a2, (__int64)&v210);
    v213 = 0;
    v182 = v29;
    v214 = 0;
    v215 = 0;
    v183 = sub_1B449D0(a1, v24, (__int64)&v213);
    v228 = (__int64 *)v230;
    v231 = (unsigned __int64 *)v233;
    v229 = 0x800000000LL;
    v232 = 0x800000000LL;
    v30 = sub_1B43680(v24);
    v204 = sub_1B43680(a2);
    if ( v30 )
    {
      sub_1B43970(v24, (__int64)&v231);
      v33 = v213;
      v35 = v214 - v213;
      if ( v35 + 1 != (unsigned int)v232 )
      {
        v204 = 0;
        v234 = v236;
        v235 = 0x800000000LL;
        if ( v183 != v187 )
          goto LABEL_33;
        goto LABEL_65;
      }
      v234 = v236;
      v235 = 0x800000000LL;
      if ( !v204 )
      {
        v167 = v236;
        v168 = ((v211 - v210) >> 4) + 1;
        if ( v168 > 8 )
        {
          sub_16CD150((__int64)&v234, v236, ((v211 - v210) >> 4) + 1, 8, v31, v32);
          v167 = v234;
        }
        LODWORD(v235) = v168;
        for ( i = &v167[8 * (unsigned int)v168]; i != v167; v167 += 8 )
          *(_QWORD *)v167 = 1;
        v33 = v213;
        v204 = v30;
        v35 = v214 - v213;
LABEL_32:
        if ( v183 != v187 )
        {
LABEL_33:
          v217 = 0;
          v218 = 0;
          v219 = &v217;
          v220 = &v217;
          v221 = 0;
          LODWORD(v223) = 0;
          v224 = 0;
          v225 = &v223;
          v226 = &v223;
          v227 = 0;
          if ( !(_DWORD)v35 )
          {
            v47 = v210;
            v191 = (v211 - v210) >> 4;
            if ( v191 )
              goto LABEL_44;
            v119 = 0;
            goto LABEL_153;
          }
          v190 = v24;
          v36 = 0;
          v37 = v35;
          for ( j = v33; ; j = v213 )
          {
            v39 = v36;
            v40 = (__m128i *)&j[v39];
            if ( v40->m128i_i64[1] == v187 )
            {
              v42 = sub_1B42500((__int64)&v216, v40);
              if ( v41 )
              {
                sub_1B426D0((__int64)&v216, (__int64)v42, v41, v40);
                v40 = (__m128i *)&v213[v39];
              }
              if ( v204 )
              {
                v181 = v36 + 1;
                v185 = &v231[v181];
                *sub_1B4F260(&v222, (unsigned __int64 *)v40) = *v185;
                v165 = &v231[(unsigned int)v232 - 1];
                v166 = v231[v181];
                v231[v181] = *v165;
                *v165 = v166;
                LODWORD(v232) = v232 - 1;
                v40 = (__m128i *)&v213[v39];
              }
              v43 = v214;
              v44 = v40->m128i_i64[0];
              --v37;
              v45 = v40->m128i_i64[1];
              v46 = v214 - 1;
              a7 = _mm_loadu_si128(v214 - 1);
              *v40 = a7;
              v46->m128i_i64[0] = v44;
              v43[-1].m128i_i64[1] = v45;
              --v214;
              if ( v37 == v36 )
              {
LABEL_43:
                v47 = v210;
                v24 = v190;
                v191 = (v211 - v210) >> 4;
                if ( !v191 )
                  goto LABEL_128;
LABEL_44:
                v179 = v24;
                v48 = v47;
                v194 = 0;
                while ( 2 )
                {
                  v49 = 16LL * v194;
                  v50 = (__m128i *)(v49 + v48);
                  if ( &v217 != (int *)sub_1B42770((__int64)&v216, v50) )
                  {
                    if ( !v204 )
                    {
                      v51 = v214;
                      if ( v214 != v215 )
                        goto LABEL_48;
LABEL_125:
                      sub_1B43170(&v213, v51, v50);
                      v52 = (unsigned int)v229;
                      v50 = (__m128i *)(v49 + v210);
                      if ( (unsigned int)v229 >= HIDWORD(v229) )
                        goto LABEL_126;
                      goto LABEL_51;
                    }
                    v94 = v224;
                    if ( !v224 )
                    {
                      v96 = &v223;
                      goto LABEL_117;
                    }
                    v95 = v50->m128i_i64[0];
                    v96 = &v223;
                    do
                    {
                      while ( 1 )
                      {
                        v97 = v94[2];
                        v98 = v94[3];
                        if ( v94[4] >= v95 )
                          break;
                        v94 = (__int64 *)v94[3];
                        if ( !v98 )
                          goto LABEL_115;
                      }
                      v96 = v94;
                      v94 = (__int64 *)v94[2];
                    }
                    while ( v97 );
LABEL_115:
                    if ( v96 == &v223 || v96[4] > v95 )
                    {
LABEL_117:
                      v99 = sub_22077B0(48);
                      v100 = v50->m128i_i64[0];
                      v101 = v96;
                      *(_QWORD *)(v99 + 40) = 0;
                      v96 = (__int64 *)v99;
                      *(_QWORD *)(v99 + 32) = v100;
                      v102 = sub_1B4F160(&v222, v101, (unsigned __int64 *)(v99 + 32));
                      if ( v103 )
                      {
                        v104 = v102 || &v223 == v103 || v100 < v103[4];
                        sub_220F040(v104, v96, v103, &v223);
                        ++v227;
                      }
                      else
                      {
                        v173 = v96;
                        v96 = v102;
                        j_j___libc_free_0(v173, 48);
                      }
                    }
                    v105 = (unsigned int)v232;
                    if ( (unsigned int)v232 >= HIDWORD(v232) )
                    {
                      sub_16CD150((__int64)&v231, v233, 0, 8, v31, v32);
                      v105 = (unsigned int)v232;
                    }
                    v231[v105] = v96[5];
                    LODWORD(v232) = v232 + 1;
                    v51 = v214;
                    v50 = (__m128i *)(v49 + v210);
                    if ( v214 == v215 )
                      goto LABEL_125;
LABEL_48:
                    if ( v51 )
                    {
                      a5 = _mm_loadu_si128(v50);
                      *v51 = a5;
                      v51 = v214;
                      v50 = (__m128i *)(v49 + v210);
                    }
                    v52 = (unsigned int)v229;
                    v214 = v51 + 1;
                    if ( (unsigned int)v229 >= HIDWORD(v229) )
                    {
LABEL_126:
                      sub_16CD150((__int64)&v228, v230, 0, 8, v31, v32);
                      v52 = (unsigned int)v229;
                    }
LABEL_51:
                    v228[v52] = v50->m128i_i64[1];
                    v53 = v218;
                    LODWORD(v229) = v229 + 1;
                    v54 = (_QWORD *)(v210 + v49);
                    v184 = (__int64)v218;
                    if ( v218 )
                    {
                      v55 = &v217;
                      v56 = (unsigned __int64 *)(*v54 + 24LL);
                      while ( 1 )
                      {
                        while ( 1 )
                        {
                          v59 = (unsigned __int64 *)(*((_QWORD *)v53 + 4) + 24LL);
                          if ( (int)sub_16A9900((__int64)v59, v56) >= 0 )
                            break;
                          v53 = (int *)*((_QWORD *)v53 + 3);
                          if ( !v53 )
                          {
LABEL_57:
                            v60 = v55 == &v217;
                            goto LABEL_58;
                          }
                        }
                        v57 = sub_16A9900((__int64)v56, v59);
                        v58 = (int *)*((_QWORD *)v53 + 2);
                        if ( v57 >= 0 )
                          break;
                        v55 = v53;
                        v53 = (int *)*((_QWORD *)v53 + 2);
                        if ( !v58 )
                          goto LABEL_57;
                      }
                      v31 = *((_QWORD *)v53 + 3);
                      if ( v31 )
                      {
                        v180 = (int *)*((_QWORD *)v53 + 2);
                        v86 = (int *)*((_QWORD *)v53 + 3);
                        do
                        {
                          while ( 1 )
                          {
                            v87 = sub_16A9900((__int64)v56, (unsigned __int64 *)(*((_QWORD *)v86 + 4) + 24LL));
                            v88 = *((_QWORD *)v86 + 2);
                            v89 = *((_QWORD *)v86 + 3);
                            if ( v87 < 0 )
                              break;
                            v86 = (int *)*((_QWORD *)v86 + 3);
                            if ( !v89 )
                              goto LABEL_99;
                          }
                          v55 = v86;
                          v86 = (int *)*((_QWORD *)v86 + 2);
                        }
                        while ( v88 );
LABEL_99:
                        v58 = v180;
                      }
                      while ( v58 )
                      {
                        while ( 1 )
                        {
                          v90 = sub_16A9900(*((_QWORD *)v58 + 4) + 24LL, v56);
                          v91 = *((_QWORD *)v58 + 2);
                          if ( v90 < 0 )
                            break;
                          v53 = v58;
                          v58 = (int *)*((_QWORD *)v58 + 2);
                          if ( !v91 )
                            goto LABEL_103;
                        }
                        v58 = (int *)*((_QWORD *)v58 + 3);
                      }
LABEL_103:
                      if ( v219 == v53 && v55 == &v217 )
                      {
LABEL_60:
                        sub_1B44180(v184);
                        v219 = &v217;
                        v218 = 0;
                        v220 = &v217;
                        v221 = 0;
                      }
                      else
                      {
                        while ( v55 != v53 )
                        {
                          v92 = v53;
                          v53 = (int *)sub_220EF30(v53);
                          v93 = sub_220F330(v92, &v217);
                          j_j___libc_free_0(v93, 40);
                          --v221;
                        }
                      }
                    }
                    else
                    {
                      v60 = v205;
                      v55 = &v217;
LABEL_58:
                      if ( v219 == v55 && v60 )
                        goto LABEL_60;
                    }
                  }
                  if ( ++v194 != v191 )
                  {
                    v48 = v210;
                    continue;
                  }
                  break;
                }
                v24 = v179;
LABEL_128:
                if ( v219 != &v217 )
                {
                  v189 = v24;
                  v106 = (__int64)v219;
                  while ( 2 )
                  {
                    v109 = *(_QWORD *)(v106 + 32);
                    if ( !v204 )
                      goto LABEL_130;
                    v110 = v224;
                    v111 = &v223;
                    if ( !v224 )
                      goto LABEL_143;
                    do
                    {
                      while ( 1 )
                      {
                        v112 = v110[2];
                        v113 = v110[3];
                        if ( v110[4] >= v109 )
                          break;
                        v110 = (__int64 *)v110[3];
                        if ( !v113 )
                          goto LABEL_141;
                      }
                      v111 = v110;
                      v110 = (__int64 *)v110[2];
                    }
                    while ( v112 );
LABEL_141:
                    if ( v111 != &v223 && v111[4] <= v109 )
                    {
LABEL_148:
                      v118 = (unsigned int)v232;
                      if ( (unsigned int)v232 >= HIDWORD(v232) )
                        goto LABEL_218;
                    }
                    else
                    {
LABEL_143:
                      v193 = v111;
                      v114 = sub_22077B0(48);
                      *(_QWORD *)(v114 + 32) = v109;
                      *(_QWORD *)(v114 + 40) = 0;
                      v197 = (__int64 *)v114;
                      v115 = sub_1B4F160(&v222, v193, (unsigned __int64 *)(v114 + 32));
                      if ( v116 )
                      {
                        v117 = &v223 == v116 || v115 || v109 < v116[4];
                        sub_220F040(v117, v197, v116, &v223);
                        ++v227;
                        v111 = v197;
                        goto LABEL_148;
                      }
                      v155 = v197;
                      v200 = v115;
                      j_j___libc_free_0(v155, 48);
                      v111 = v200;
                      v118 = (unsigned int)v232;
                      if ( (unsigned int)v232 >= HIDWORD(v232) )
                      {
LABEL_218:
                        v201 = v111;
                        sub_16CD150((__int64)&v231, v233, 0, 8, v31, v32);
                        v118 = (unsigned int)v232;
                        v111 = v201;
                      }
                    }
                    v231[v118] = v111[5];
                    LODWORD(v232) = v232 + 1;
LABEL_130:
                    v209.m128i_i64[0] = v109;
                    v107 = v214;
                    v209.m128i_i64[1] = v182;
                    if ( v214 != v215 )
                    {
                      if ( v214 )
                      {
                        a4 = (__m128)_mm_loadu_si128(&v209);
                        *v214 = (__m128i)a4;
                        v107 = v214;
                      }
                      v108 = (unsigned int)v229;
                      v214 = (__m128i *)&v107[1];
                      if ( (unsigned int)v229 < HIDWORD(v229) )
                        goto LABEL_134;
LABEL_151:
                      sub_16CD150((__int64)&v228, v230, 0, 8, v31, v32);
                      v108 = (unsigned int)v229;
                      goto LABEL_134;
                    }
                    sub_1B43170(&v213, v214, &v209);
                    v108 = (unsigned int)v229;
                    if ( (unsigned int)v229 >= HIDWORD(v229) )
                      goto LABEL_151;
LABEL_134:
                    v228[v108] = v182;
                    LODWORD(v229) = v229 + 1;
                    v106 = sub_220EF30(v106);
                    if ( (int *)v106 == &v217 )
                    {
                      v24 = v189;
                      v119 = (__int64)v224;
                      goto LABEL_153;
                    }
                    continue;
                  }
                }
                v119 = (__int64)v224;
LABEL_153:
                sub_1B43C00(v119);
                sub_1B44180((__int64)v218);
LABEL_154:
                v120 = v228;
                v121 = &v228[(unsigned int)v229];
                if ( v228 != v121 )
                {
                  do
                  {
                    v122 = *v120++;
                    sub_1B44430(v122, v202, v187);
                  }
                  while ( v121 != v120 );
                }
                sub_17050D0(a3, v24);
                if ( *(_BYTE *)(*v207 + 8) == 15 )
                {
                  v222 = (__int64 *)"magicptr";
                  LOWORD(v224) = 259;
                  v174 = sub_15A9650(*(_QWORD *)(a1 + 8), *v207);
                  v207 = (__int64 *)sub_12AA3B0(a3, 0x2Du, (__int64)v207, v174, (__int64)&v222);
                }
                LOWORD(v224) = 257;
                v123 = v214 - v213;
                v124 = sub_1648B60(64);
                v125 = v124;
                if ( v124 )
                  sub_15FFAB0(v124, (__int64)v207, v183, v123, 0);
                v126 = a3[1];
                if ( v126 )
                {
                  v127 = (__int64 *)a3[2];
                  sub_157E9D0(v126 + 40, v125);
                  v128 = *(_QWORD *)(v125 + 24);
                  v129 = *v127;
                  *(_QWORD *)(v125 + 32) = v127;
                  v129 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v125 + 24) = v129 | v128 & 7;
                  *(_QWORD *)(v129 + 8) = v125 + 24;
                  *v127 = *v127 & 7 | (v125 + 24);
                }
                v130 = (__int64 *)(v125 + 48);
                sub_164B780(v125, (__int64 *)&v222);
                sub_12A86E0(a3, v125);
                v134 = *(_QWORD *)(v24 + 48);
                v222 = (__int64 *)v134;
                if ( v134 )
                {
                  sub_1623A60((__int64)&v222, v134, 2);
                  if ( v130 == (__int64 *)&v222 )
                  {
                    if ( v222 )
                      sub_161E7C0((__int64)&v222, (__int64)v222);
                    goto LABEL_166;
                  }
                  v147 = *(_QWORD *)(v125 + 48);
                  if ( !v147 )
                  {
LABEL_201:
                    v148 = (unsigned __int8 *)v222;
                    *(_QWORD *)(v125 + 48) = v222;
                    if ( v148 )
                      sub_1623210((__int64)&v222, v148, v125 + 48);
LABEL_166:
                    v135 = v214;
                    for ( k = v213; v135 != k; ++k )
                    {
                      v137 = k->m128i_i64[1];
                      v138 = k->m128i_i64[0];
                      sub_15FFFB0(v125, v138, v137, v131, v132, v133);
                    }
                    if ( !v204 )
                    {
LABEL_169:
                      sub_1B44FE0(v24);
                      if ( (*(_DWORD *)(v125 + 20) & 0xFFFFFFFu) >> 1 )
                      {
                        v139 = 24;
                        v140 = 0;
                        v141 = 48LL * (((*(_DWORD *)(v125 + 20) & 0xFFFFFFFu) >> 1) - 1) + 72;
                        do
                        {
                          while ( 1 )
                          {
                            v144 = *(_BYTE *)(v125 + 23) & 0x40;
                            v142 = v144 ? *(_QWORD *)(v125 - 8) : v125 - 24LL * (*(_DWORD *)(v125 + 20) & 0xFFFFFFF);
                            v143 = *(_QWORD *)(v142 + v139);
                            if ( v143 )
                            {
                              if ( v187 == v143 )
                                break;
                            }
                            v139 += 48;
                            if ( v139 == v141 )
                              goto LABEL_181;
                          }
                          if ( !v140 )
                          {
                            v149 = *(_QWORD *)(v187 + 56);
                            v222 = (__int64 *)"infloop";
                            v198 = v149;
                            LOWORD(v224) = 259;
                            v203 = sub_157E9C0(v187);
                            v150 = (_QWORD *)sub_22077B0(64);
                            v140 = (__int64)v150;
                            if ( v150 )
                              sub_157FB60(v150, v203, (__int64)&v222, v198, 0);
                            v151 = sub_1648A60(56, 1u);
                            if ( v151 )
                              sub_15F8590((__int64)v151, v140, v140);
                            v144 = *(_BYTE *)(v125 + 23) & 0x40;
                          }
                          if ( v144 )
                            v145 = *(_QWORD *)(v125 - 8);
                          else
                            v145 = v125 - 24LL * (*(_DWORD *)(v125 + 20) & 0xFFFFFFF);
                          v146 = (_QWORD *)(v139 + v145);
                          v139 += 48;
                          sub_1593B40(v146, v140);
                        }
                        while ( v139 != v141 );
                      }
LABEL_181:
                      if ( v234 != v236 )
                        _libc_free((unsigned __int64)v234);
                      if ( v231 != (unsigned __int64 *)v233 )
                        _libc_free((unsigned __int64)v231);
                      if ( v228 != (__int64 *)v230 )
                        _libc_free((unsigned __int64)v228);
                      if ( v213 )
                        j_j___libc_free_0(v213, (char *)v215 - (char *)v213);
                      if ( v210 )
                        j_j___libc_free_0(v210, v212 - v210);
                      if ( v240 != (__int64 *)v242 )
                        _libc_free((unsigned __int64)v240);
                      if ( (v238 & 1) == 0 )
                        j___libc_free_0(v239);
                      v22 = v205;
                      goto LABEL_17;
                    }
                    sub_1B425D0(v231, (unsigned int)v232);
                    v158 = (unsigned int)v232;
                    v222 = (__int64 *)&v224;
                    v223 = 0x800000000LL;
                    v159 = v231;
                    v160 = v232;
                    if ( (unsigned int)v232 <= 8uLL )
                    {
                      if ( 8LL * (unsigned int)v232 )
                      {
                        v161 = (__int64 *)&v224;
LABEL_222:
                        v162 = 0;
                        do
                        {
                          *(_DWORD *)((char *)v161 + v162 * 4) = v159[v162];
                          ++v162;
                        }
                        while ( v158 != v162 );
                        v163 = v222;
                        v160 = v158 + v223;
                        v164 = (unsigned int)(v158 + v223);
                      }
                      else
                      {
                        v164 = (unsigned int)v232;
                        v163 = (__int64 *)&v224;
                      }
                      LODWORD(v223) = v160;
                      sub_1B42940(v125, (unsigned int *)v163, v164);
                      if ( v222 != (__int64 *)&v224 )
                        _libc_free((unsigned __int64)v222);
                      goto LABEL_169;
                    }
                    sub_16CD150((__int64)&v222, &v224, (unsigned int)v232, 4, v156, v157);
                    v161 = (__int64 *)((char *)v222 + 4 * (unsigned int)v223);
                    goto LABEL_222;
                  }
                }
                else
                {
                  if ( v130 == (__int64 *)&v222 )
                    goto LABEL_166;
                  v147 = *(_QWORD *)(v125 + 48);
                  if ( !v147 )
                    goto LABEL_166;
                }
                sub_161E7C0(v125 + 48, v147);
                goto LABEL_201;
              }
            }
            else if ( v37 == ++v36 )
            {
              goto LABEL_43;
            }
          }
        }
LABEL_65:
        LODWORD(v223) = 0;
        v224 = 0;
        v225 = &v223;
        v226 = &v223;
        v227 = 0;
        if ( (_DWORD)v35 )
        {
          v195 = v24;
          v61 = 0;
          v62 = v35;
          while ( 1 )
          {
            v65 = (__m128i *)&v33[v61];
            if ( v65->m128i_i64[1] == v187 )
            {
              v66 = v214;
              v67 = v65->m128i_i64[0];
              v68 = v214 - 1;
              a6 = _mm_loadu_si128(v214 - 1);
              *v65 = a6;
              v68->m128i_i64[0] = v67;
              v66[-1].m128i_i64[1] = v187;
              if ( v204 )
              {
                v69 = (unsigned int)(v61 + 1);
                *v231 += v231[v69];
                v70 = &v231[(unsigned int)v232 - 1];
                v71 = &v231[v69];
                v72 = *v71;
                *v71 = *v70;
                *v70 = v72;
                LODWORD(v232) = v232 - 1;
              }
              --v62;
              --v214;
              if ( v62 == v61 )
                goto LABEL_75;
            }
            else
            {
              v64 = sub_1B42500((__int64)&v222, (const __m128i *)v33[v61].m128i_i64);
              if ( v63 )
                sub_1B426D0((__int64)&v222, (__int64)v64, v63, v65);
              if ( v62 == ++v61 )
              {
LABEL_75:
                v24 = v195;
                break;
              }
            }
            v33 = v213;
          }
        }
        v183 = v182;
        if ( v182 != v187 )
        {
          sub_157F2D0(v187, v202, 0);
          v75 = (unsigned int)v229;
          if ( (unsigned int)v229 >= HIDWORD(v229) )
          {
            sub_16CD150((__int64)&v228, v230, 0, 8, v73, v74);
            v75 = (unsigned int)v229;
          }
          v228[v75] = v182;
          LODWORD(v229) = v229 + 1;
          v183 = v182;
        }
        v76 = v210;
        v188 = v232;
        v77 = (v211 - v210) >> 4;
        if ( (_DWORD)v77 )
        {
          v78 = 0;
          v192 = 0;
          v79 = 8LL * (unsigned int)(v77 - 1);
          v196 = v24;
          while ( 1 )
          {
            v80 = 2 * v78;
            v81 = (__m128i *)(v76 + 2 * v78);
            if ( &v223 == sub_1B42770((__int64)&v222, v81) && v182 != v81->m128i_i64[1] )
            {
              v84 = v214;
              if ( v214 == v215 )
              {
                sub_1B43170(&v213, v214, v81);
                v81 = (__m128i *)(v210 + v80);
              }
              else
              {
                if ( v214 )
                {
                  *v214 = _mm_loadu_si128(v81);
                  v84 = v214;
                  v81 = (__m128i *)(v210 + v80);
                }
                v214 = (__m128i *)&v84[1];
              }
              v85 = (unsigned int)v229;
              if ( (unsigned int)v229 >= HIDWORD(v229) )
              {
                sub_16CD150((__int64)&v228, v230, 0, 8, v82, v83);
                v85 = (unsigned int)v229;
              }
              v228[v85] = v81->m128i_i64[1];
              LODWORD(v229) = v229 + 1;
              if ( v204 )
              {
                v216 = *v231 * *(_QWORD *)&v234[v78 + 8];
                sub_1525CA0((__int64)&v231, &v216);
                v192 += *(_QWORD *)&v234[v78 + 8];
              }
            }
            if ( v79 == v78 )
              break;
            v76 = v210;
            v78 += 8;
          }
          v24 = v196;
        }
        else
        {
          v192 = 0;
        }
        if ( v204 )
        {
          v175 = *(_QWORD *)v234;
          v176 = *(_QWORD *)v234 + v192;
          if ( v188 > 1 )
          {
            v177 = 1;
            do
            {
              v178 = &v231[v177++];
              *v178 *= v176;
            }
            while ( v188 != v177 );
            v175 = *(_QWORD *)v234;
          }
          *v231 *= v175;
        }
        sub_1B44180((__int64)v224);
        goto LABEL_154;
      }
    }
    else
    {
      v33 = v213;
      v34 = v214 - v213;
      LODWORD(v35) = v34;
      if ( !v204 )
      {
        v234 = v236;
        v235 = 0x800000000LL;
        goto LABEL_32;
      }
      v170 = v34 + 1;
      LODWORD(v232) = 0;
      if ( v34 + 1 > (unsigned __int64)HIDWORD(v232) )
        sub_16CD150((__int64)&v231, v233, v170, 8, v31, v32);
      v171 = v231;
      LODWORD(v232) = v170;
      v172 = &v231[(unsigned int)v170];
      if ( v231 != v172 )
      {
        do
          *v171++ = 1;
        while ( v172 != v171 );
      }
      v234 = v236;
      v235 = 0x800000000LL;
    }
    sub_1B43970(a2, (__int64)&v234);
    v33 = v213;
    if ( ((v211 - v210) >> 4) + 1 == (unsigned int)v235 )
    {
      v35 = v214 - v213;
      v204 = v205;
    }
    else
    {
      v204 = 0;
      v35 = v214 - v213;
    }
    goto LABEL_32;
  }
  if ( v240 == &v240[(unsigned int)v241] )
    goto LABEL_29;
  v199 = v24;
  v152 = v240;
  v153 = &v240[(unsigned int)v241];
  while ( 1 )
  {
    v154 = *v152;
    v234 = *(_BYTE **)(a2 + 40);
    if ( !sub_1AAB350(
            v154,
            (__int64 *)&v234,
            1,
            ".fold.split",
            0,
            0,
            a4,
            *(double *)a5.m128i_i64,
            *(double *)a6.m128i_i64,
            *(double *)a7.m128i_i64,
            v27,
            v28,
            a10,
            a11,
            0) )
      break;
    if ( v153 == ++v152 )
    {
      v24 = v199;
      goto LABEL_29;
    }
  }
  if ( v240 != (__int64 *)v242 )
    _libc_free((unsigned __int64)v240);
  if ( (v238 & 1) == 0 )
    j___libc_free_0(v239);
  v20 = v243;
  v22 = 0;
LABEL_19:
  if ( v20 != v245 )
    _libc_free((unsigned __int64)v20);
  return v22;
}
