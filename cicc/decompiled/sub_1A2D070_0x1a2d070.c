// Function: sub_1A2D070
// Address: 0x1a2d070
//
__int64 __fastcall sub_1A2D070(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // r12
  __int64 v10; // rax
  char v11; // dl
  __int64 *v12; // r15
  __int64 v13; // r13
  __int64 v14; // rbx
  _QWORD *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rbx
  __int64 v20; // rdx
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 *v23; // rbx
  __int64 v24; // r15
  __int64 v25; // r13
  __int64 v26; // r12
  _QWORD *v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // r12
  int v32; // r13d
  __int64 v33; // rdx
  __int64 *v34; // rax
  __int64 *v36; // r13
  char v37; // dl
  __int64 v38; // rbx
  _QWORD *v39; // rax
  unsigned __int8 *v40; // r9
  double v41; // xmm4_8
  double v42; // xmm5_8
  unsigned __int8 *v43; // rsi
  __int64 v44; // rax
  _QWORD *v45; // r8
  int v46; // esi
  unsigned int v47; // edx
  unsigned __int8 *v48; // rbx
  __int64 v49; // rdi
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rdi
  unsigned int v54; // esi
  unsigned __int32 v55; // edx
  unsigned __int32 v56; // edi
  unsigned int v57; // r8d
  __int64 v58; // r8
  int v59; // edx
  __int64 *v60; // rbx
  __int64 v61; // r12
  unsigned __int8 *v62; // rdx
  __int64 v63; // rax
  __int64 *v64; // r14
  __int64 v65; // rsi
  unsigned __int8 *v66; // rsi
  __int64 v67; // r14
  __int64 v68; // r9
  __int64 *v69; // rax
  __int64 v70; // rbx
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // r14
  __int64 v74; // rax
  __int64 v75; // rbx
  __int64 v76; // rbx
  __int64 v77; // rax
  __int64 v78; // rsi
  unsigned __int8 *v79; // rsi
  unsigned __int8 *v80; // rdi
  _QWORD *v81; // rax
  unsigned __int64 v82; // rdi
  _QWORD *v83; // rax
  _QWORD *v84; // r10
  __int64 v85; // rcx
  unsigned __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // rdx
  unsigned __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // rdx
  unsigned __int64 v92; // rax
  __int64 v93; // rdx
  __int64 *v94; // rbx
  __int64 v95; // r12
  unsigned __int8 *v96; // rdx
  __int64 v97; // rax
  __int64 *v98; // r14
  __int64 v99; // rsi
  unsigned __int8 *v100; // rsi
  int v101; // ebx
  unsigned int v102; // r12d
  __int64 *v103; // rax
  __int64 v104; // r12
  __int64 v105; // rax
  __int64 v106; // r14
  _QWORD *v107; // rax
  __int64 *v108; // r9
  unsigned int v109; // eax
  __int64 v110; // rsi
  __int64 v111; // rcx
  __int64 v112; // r13
  __int64 v113; // rdi
  __int64 v114; // r12
  __int64 v115; // r8
  __int64 *v116; // rax
  __int64 v117; // rdi
  __int64 v118; // r14
  int v119; // eax
  __int64 v120; // rax
  int v121; // edx
  __int64 v122; // rdx
  __int64 *v123; // rax
  __int64 v124; // rcx
  unsigned __int64 v125; // rdx
  __int64 v126; // rdx
  __int64 v127; // rdx
  __int64 v128; // rcx
  __int64 v129; // rdx
  __int64 v130; // rsi
  unsigned __int8 *v131; // rsi
  int v132; // r11d
  int v133; // r11d
  int v134; // eax
  __int64 v135; // rdx
  __int64 v136; // r14
  unsigned __int64 v137; // rax
  __int64 *v138; // r9
  unsigned __int8 *v139; // rsi
  __int64 v140; // rax
  __int64 v141; // rax
  __int64 v142; // rdx
  __int64 v143; // rdi
  int v144; // esi
  __int64 *v145; // rcx
  __int64 *v146; // rdx
  __int64 v147; // r14
  int v148; // ecx
  __int64 v149; // rsi
  _QWORD *v150; // rax
  _QWORD *v151; // rax
  _QWORD *v152; // rax
  __int64 *v153; // r10
  __int64 v154; // [rsp+8h] [rbp-328h]
  __int64 *v155; // [rsp+10h] [rbp-320h]
  __int64 v156; // [rsp+18h] [rbp-318h]
  __int64 v157; // [rsp+18h] [rbp-318h]
  _QWORD *v158; // [rsp+20h] [rbp-310h]
  __int64 *v159; // [rsp+20h] [rbp-310h]
  __int64 *v160; // [rsp+20h] [rbp-310h]
  __int64 *v161; // [rsp+20h] [rbp-310h]
  __int64 *v162; // [rsp+20h] [rbp-310h]
  __int64 *v163; // [rsp+20h] [rbp-310h]
  __int64 v164; // [rsp+38h] [rbp-2F8h]
  __int64 v165; // [rsp+40h] [rbp-2F0h]
  __int64 v166; // [rsp+40h] [rbp-2F0h]
  __int64 *v167; // [rsp+50h] [rbp-2E0h]
  char v168; // [rsp+50h] [rbp-2E0h]
  unsigned __int8 v169; // [rsp+58h] [rbp-2D8h]
  __int64 v170; // [rsp+58h] [rbp-2D8h]
  __int64 v171; // [rsp+58h] [rbp-2D8h]
  __int64 v172; // [rsp+58h] [rbp-2D8h]
  __int64 v173; // [rsp+58h] [rbp-2D8h]
  _BYTE *v174; // [rsp+58h] [rbp-2D8h]
  __int64 v175; // [rsp+58h] [rbp-2D8h]
  unsigned __int8 *v176; // [rsp+68h] [rbp-2C8h]
  __int64 v177; // [rsp+68h] [rbp-2C8h]
  __int64 v178; // [rsp+68h] [rbp-2C8h]
  __int64 v179; // [rsp+68h] [rbp-2C8h]
  __int64 v180; // [rsp+68h] [rbp-2C8h]
  __int64 v181; // [rsp+68h] [rbp-2C8h]
  __int64 v182; // [rsp+70h] [rbp-2C0h]
  __int64 v183; // [rsp+78h] [rbp-2B8h]
  __int64 v184; // [rsp+80h] [rbp-2B0h]
  unsigned __int8 *v185; // [rsp+80h] [rbp-2B0h]
  const char *v186; // [rsp+88h] [rbp-2A8h]
  __int64 *v187; // [rsp+90h] [rbp-2A0h]
  unsigned __int8 *v188; // [rsp+98h] [rbp-298h]
  __int64 *v189; // [rsp+A0h] [rbp-290h] BYREF
  unsigned int v190; // [rsp+A8h] [rbp-288h]
  __m128i v191; // [rsp+B0h] [rbp-280h] BYREF
  __int16 v192; // [rsp+C0h] [rbp-270h]
  __m128i v193; // [rsp+D0h] [rbp-260h] BYREF
  __int16 v194; // [rsp+E0h] [rbp-250h]
  __m128i v195; // [rsp+F0h] [rbp-240h] BYREF
  __int64 v196; // [rsp+100h] [rbp-230h]
  unsigned int v197; // [rsp+108h] [rbp-228h]
  __int64 *v198; // [rsp+110h] [rbp-220h] BYREF
  __int64 v199; // [rsp+118h] [rbp-218h]
  _BYTE v200[32]; // [rsp+120h] [rbp-210h] BYREF
  __int64 *v201; // [rsp+140h] [rbp-1F0h] BYREF
  __int64 v202; // [rsp+148h] [rbp-1E8h]
  _BYTE v203[32]; // [rsp+150h] [rbp-1E0h] BYREF
  const char *v204; // [rsp+170h] [rbp-1C0h] BYREF
  __int64 v205; // [rsp+178h] [rbp-1B8h]
  _BYTE v206[32]; // [rsp+180h] [rbp-1B0h] BYREF
  unsigned __int8 *v207; // [rsp+1A0h] [rbp-190h] BYREF
  __int64 v208; // [rsp+1A8h] [rbp-188h]
  _QWORD v209[4]; // [rsp+1B0h] [rbp-180h] BYREF
  __m128i v210; // [rsp+1D0h] [rbp-160h] BYREF
  _QWORD *v211; // [rsp+1E0h] [rbp-150h] BYREF
  unsigned int v212; // [rsp+1E8h] [rbp-148h]
  unsigned __int8 *v213; // [rsp+220h] [rbp-110h] BYREF
  __int64 v214; // [rsp+228h] [rbp-108h]
  __int64 v215; // [rsp+230h] [rbp-100h]
  _QWORD *v216; // [rsp+238h] [rbp-F8h]
  __int64 v217; // [rsp+240h] [rbp-F0h]
  int v218; // [rsp+248h] [rbp-E8h]
  __int64 v219; // [rsp+250h] [rbp-E0h]
  __int64 v220; // [rsp+258h] [rbp-D8h]
  _QWORD *v221; // [rsp+260h] [rbp-D0h]
  __int64 v222; // [rsp+268h] [rbp-C8h]
  _QWORD v223[3]; // [rsp+270h] [rbp-C0h] BYREF
  _BYTE *v224; // [rsp+288h] [rbp-A8h]
  __int64 v225; // [rsp+290h] [rbp-A0h]
  _BYTE v226[16]; // [rsp+298h] [rbp-98h] BYREF
  _QWORD *v227; // [rsp+2A8h] [rbp-88h]
  __int64 v228; // [rsp+2B0h] [rbp-80h]
  _QWORD v229[4]; // [rsp+2B8h] [rbp-78h] BYREF
  __int64 v230; // [rsp+2D8h] [rbp-58h]
  unsigned __int8 *v231; // [rsp+2E0h] [rbp-50h]
  __int64 v232; // [rsp+2E8h] [rbp-48h]
  __int64 v233; // [rsp+2F0h] [rbp-40h]

  v164 = *(_QWORD *)a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 15 )
    return 0;
  v9 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
  if ( *(_BYTE *)(v9 + 8) != 13 )
    return 0;
  v10 = sub_15F2050(a1);
  v182 = sub_1632FA0(v10);
  v198 = (__int64 *)v200;
  v199 = 0x400000000LL;
  sub_1A24B00(a1, (__int64)&v198);
  v187 = &v198[(unsigned int)v199];
  if ( v198 == v187 )
  {
    v169 = 0;
  }
  else
  {
    v11 = 0;
    v12 = v198;
    do
    {
      while ( v9 != *(_QWORD *)*v12 )
      {
        if ( v187 == ++v12 )
          goto LABEL_16;
      }
      v207 = 0;
      v208 = 0;
      v209[0] = 0;
      sub_14A8180(*v12, (__int64 *)&v207, 0);
      v13 = *v12;
      v14 = v209[0];
      v176 = v207;
      v184 = v208;
      v15 = (_QWORD *)sub_16498A0(*v12);
      v213 = 0;
      v216 = v15;
      v221 = v223;
      v215 = 0;
      v217 = 0;
      v218 = 0;
      v219 = 0;
      v220 = 0;
      v214 = 0;
      v222 = 0;
      LOBYTE(v223[0]) = 0;
      sub_17050D0((__int64 *)&v213, v13);
      v224 = v226;
      v225 = 0x400000000LL;
      v16 = sub_1643350(v216);
      v17 = sub_159C470(v16, 0, 0);
      v233 = v14;
      v229[0] = v17;
      v228 = 0x400000001LL;
      v227 = v229;
      v231 = v176;
      v232 = v184;
      v230 = a1;
      v18 = sub_1599EF0((__int64 **)v9);
      v201 = 0;
      v195.m128i_i64[0] = v18;
      v19 = (sub_127FA20(v182, v9) + 7) & 0xFFFFFFFFFFFFFFF8LL;
      LODWORD(v184) = 1 << (*(unsigned __int16 *)(*v12 + 18) >> 1) >> 1;
      v204 = sub_1649960(a1);
      v205 = v20;
      LOWORD(v211) = 773;
      v210.m128i_i64[0] = (__int64)&v204;
      v210.m128i_i64[1] = (__int64)".fca";
      sub_1A1F890((__int64)&v213, v9, (__int64 **)&v195, &v210, v184, &v201, v19);
      sub_164D160(*v12, v195.m128i_i64[0], a2, a3, a4, a5, v21, v22, a8, a9);
      sub_15F20C0((_QWORD *)*v12);
      if ( v227 != v229 )
        _libc_free((unsigned __int64)v227);
      if ( v224 != v226 )
        _libc_free((unsigned __int64)v224);
      if ( v221 != v223 )
        j_j___libc_free_0(v221, v223[0] + 1LL);
      if ( v213 )
        sub_161E7C0((__int64)&v213, (__int64)v213);
      v11 = 1;
      ++v12;
    }
    while ( v187 != v12 );
LABEL_16:
    v169 = v11;
  }
  v201 = (__int64 *)v203;
  v202 = 0x400000000LL;
  sub_1A247B0(a1, (__int64)&v201);
  v23 = v201;
  v167 = &v201[(unsigned int)v202];
  if ( v201 != v167 )
  {
    v24 = v9;
    do
    {
      while ( 1 )
      {
        v207 = 0;
        v208 = 0;
        v209[0] = 0;
        sub_14A8180(*v23, (__int64 *)&v207, 0);
        v25 = v208;
        v26 = v209[0];
        v183 = *v23;
        v185 = v207;
        v27 = (_QWORD *)sub_16498A0(*v23);
        v213 = 0;
        v216 = v27;
        v217 = 0;
        v218 = 0;
        v219 = 0;
        v220 = 0;
        v221 = v223;
        v222 = 0;
        LOBYTE(v223[0]) = 0;
        v214 = *(_QWORD *)(v183 + 40);
        v215 = v183 + 24;
        v28 = *(_QWORD *)(v183 + 48);
        v210.m128i_i64[0] = v28;
        if ( v28 )
        {
          sub_1623A60((__int64)&v210, v28, 2);
          if ( v213 )
            sub_161E7C0((__int64)&v213, (__int64)v213);
          v213 = (unsigned __int8 *)v210.m128i_i64[0];
          if ( v210.m128i_i64[0] )
            sub_1623210((__int64)&v210, (unsigned __int8 *)v210.m128i_i64[0], (__int64)&v213);
        }
        v224 = v226;
        v225 = 0x400000000LL;
        v29 = sub_1643350(v216);
        v30 = sub_159C470(v29, 0, 0);
        v232 = v25;
        v229[0] = v30;
        v227 = v229;
        v230 = a1;
        v228 = 0x400000001LL;
        v231 = v185;
        v233 = v26;
        v193.m128i_i64[0] = *(_QWORD *)(*v23 - 48);
        if ( v24 != *(_QWORD *)v193.m128i_i64[0] )
          break;
        v195.m128i_i64[0] = 0;
        v31 = (sub_127FA20(v182, v24) + 7) & 0xFFFFFFFFFFFFFFF8LL;
        v32 = 1 << (*(unsigned __int16 *)(*v23 + 18) >> 1);
        v204 = sub_1649960(v193.m128i_i64[0]);
        LOWORD(v211) = 773;
        v205 = v33;
        v210.m128i_i64[0] = (__int64)&v204;
        v210.m128i_i64[1] = (__int64)".fca";
        sub_1A202B0((__int64)&v213, v24, v193.m128i_i64, &v210, v32 >> 1, &v195, v31);
        sub_15F20C0((_QWORD *)*v23);
        if ( v227 != v229 )
          _libc_free((unsigned __int64)v227);
        if ( v224 != v226 )
          _libc_free((unsigned __int64)v224);
        if ( v221 != v223 )
          j_j___libc_free_0(v221, v223[0] + 1LL);
        if ( v213 )
          sub_161E7C0((__int64)&v213, (__int64)v213);
        v169 = 1;
        if ( v167 == ++v23 )
          goto LABEL_41;
      }
      if ( v224 != v226 )
        _libc_free((unsigned __int64)v224);
      if ( v221 != v223 )
        j_j___libc_free_0(v221, v223[0] + 1LL);
      if ( v213 )
        sub_161E7C0((__int64)&v213, (__int64)v213);
      ++v23;
    }
    while ( v167 != v23 );
  }
LABEL_41:
  v204 = v206;
  v205 = 0x400000000LL;
  sub_1A254F0(a1, (__int64)&v204);
  v34 = (__int64 *)&v211;
  v210.m128i_i64[0] = 0;
  v210.m128i_i64[1] = 1;
  do
  {
    *v34 = -1;
    v34 += 2;
  }
  while ( v34 != (__int64 *)&v213 );
  v36 = (__int64 *)v204;
  v168 = *(_BYTE *)(a1 + 16);
  v186 = &v204[8 * (unsigned int)v205];
  if ( v204 == v186 )
    goto LABEL_121;
  v37 = v169;
  while ( 2 )
  {
    if ( !*(_QWORD *)(*v36 + 8) )
      goto LABEL_69;
    v190 = 8 * sub_15A9520(v182, *(_DWORD *)(v164 + 8) >> 8);
    if ( v190 <= 0x40 )
      v189 = 0;
    else
      sub_16A4EF0((__int64)&v189, 0, 0);
    sub_15FA310(*v36, v182, (__int64)&v189);
    v38 = *v36;
    v39 = (_QWORD *)sub_16498A0(*v36);
    LOBYTE(v223[0]) = 0;
    v216 = v39;
    v213 = 0;
    v217 = 0;
    v218 = 0;
    v219 = 0;
    v220 = 0;
    v221 = v223;
    v222 = 0;
    v214 = *(_QWORD *)(v38 + 40);
    v215 = v38 + 24;
    v43 = *(unsigned __int8 **)(v38 + 48);
    v207 = v43;
    if ( v43 )
    {
      sub_1623A60((__int64)&v207, (__int64)v43, 2);
      if ( v213 )
        sub_161E7C0((__int64)&v213, (__int64)v213);
      v213 = v207;
      if ( v207 )
        sub_1623210((__int64)&v207, v207, (__int64)&v213);
    }
    v44 = (__int64)v189;
    if ( v190 > 0x40 )
      v44 = *v189;
    v195.m128i_i64[0] = v44;
    if ( (v210.m128i_i8[8] & 1) != 0 )
    {
      v45 = &v211;
      v46 = 3;
    }
    else
    {
      v54 = v212;
      v45 = v211;
      if ( !v212 )
      {
        v55 = v210.m128i_u32[2];
        ++v210.m128i_i64[0];
        v188 = 0;
        v56 = ((unsigned __int32)v210.m128i_i32[2] >> 1) + 1;
        goto LABEL_78;
      }
      v46 = v212 - 1;
    }
    v47 = v46 & (37 * v44);
    v48 = (unsigned __int8 *)&v45[2 * v47];
    v49 = *(_QWORD *)v48;
    v188 = v48;
    if ( v44 == *(_QWORD *)v48 )
    {
      v50 = *((_QWORD *)v48 + 1);
      goto LABEL_57;
    }
    v132 = 1;
    v40 = 0;
    while ( 1 )
    {
      if ( v49 == -1 )
      {
        v55 = v210.m128i_u32[2];
        v57 = 12;
        if ( !v40 )
          v40 = v188;
        v54 = 4;
        ++v210.m128i_i64[0];
        v188 = v40;
        v56 = ((unsigned __int32)v210.m128i_i32[2] >> 1) + 1;
        if ( (v210.m128i_i8[8] & 1) != 0 )
        {
LABEL_79:
          if ( v57 <= 4 * v56 )
          {
            v54 *= 2;
          }
          else if ( v54 - v210.m128i_i32[3] - v56 > v54 >> 3 )
          {
LABEL_81:
            v210.m128i_i32[2] = (2 * (v55 >> 1) + 2) | v55 & 1;
            if ( *(_QWORD *)v188 != -1 )
              --v210.m128i_i32[3];
            *(_QWORD *)v188 = v44;
            *((_QWORD *)v188 + 1) = 0;
            goto LABEL_84;
          }
          sub_1A2CCD0((__int64)&v210, v54);
          sub_1A27390((__int64)&v210, v195.m128i_i64, &v207);
          v55 = v210.m128i_u32[2];
          v188 = v207;
          v44 = v195.m128i_i64[0];
          goto LABEL_81;
        }
        v54 = v212;
LABEL_78:
        v57 = 3 * v54;
        goto LABEL_79;
      }
      if ( v40 || v49 != -2 )
        v188 = v40;
      LODWORD(v40) = v132 + 1;
      v47 = v46 & (v132 + v47);
      v153 = &v45[2 * v47];
      v49 = *v153;
      if ( v44 == *v153 )
        break;
      ++v132;
      v40 = v188;
      v188 = (unsigned __int8 *)&v45[2 * v47];
    }
    v188 = (unsigned __int8 *)&v45[2 * v47];
    v50 = v153[1];
LABEL_57:
    if ( !v50 )
    {
LABEL_84:
      v58 = *v36;
      v207 = (unsigned __int8 *)v209;
      v208 = 0x400000000LL;
      v59 = *(_DWORD *)(v58 + 20);
      if ( v168 != 77 )
      {
        v60 = (__int64 *)(v58 + 24 * (1LL - (v59 & 0xFFFFFFF)));
        if ( (__int64 *)v58 != v60 )
        {
          v61 = *v60;
          v62 = (unsigned __int8 *)v209;
          v63 = 0;
          v64 = (__int64 *)v58;
          while ( 1 )
          {
            *(_QWORD *)&v62[8 * v63] = v61;
            v60 += 3;
            v63 = (unsigned int)(v208 + 1);
            LODWORD(v208) = v208 + 1;
            if ( v64 == v60 )
              break;
            v61 = *v60;
            if ( HIDWORD(v208) <= (unsigned int)v63 )
            {
              sub_16CD150((__int64)&v207, v209, 0, 8, v58, (int)v40);
              v63 = (unsigned int)v208;
            }
            v62 = v207;
          }
          v58 = (__int64)v64;
        }
        v65 = *(_QWORD *)(a1 + 48);
        v165 = v215;
        v195.m128i_i64[0] = v65;
        v177 = v214;
        v214 = *(_QWORD *)(a1 + 40);
        v215 = a1 + 24;
        if ( v65 )
        {
          v170 = v58;
          sub_1623A60((__int64)&v195, v65, 2);
          v66 = v213;
          v58 = v170;
          if ( !v213 )
            goto LABEL_95;
        }
        else
        {
          v66 = v213;
          if ( !v213 )
            goto LABEL_97;
        }
        v171 = v58;
        sub_161E7C0((__int64)&v213, (__int64)v66);
        v58 = v171;
LABEL_95:
        v213 = (unsigned __int8 *)v195.m128i_i64[0];
        if ( v195.m128i_i64[0] )
        {
          v172 = v58;
          sub_1623210((__int64)&v195, (unsigned __int8 *)v195.m128i_i64[0], (__int64)&v213);
          v58 = v172;
        }
LABEL_97:
        v67 = *(_QWORD *)(a1 - 48);
        v68 = *(_QWORD *)(a1 - 24);
        v69 = *(__int64 **)(v58 - 24LL * (*(_DWORD *)(v58 + 20) & 0xFFFFFFF));
        v70 = *v69;
        if ( *v69 != *(_QWORD *)a1 )
        {
          v194 = 257;
          if ( v70 != *(_QWORD *)v67 )
          {
            v173 = v68;
            if ( *(_BYTE *)(v67 + 16) > 0x10u )
            {
              LOWORD(v196) = 257;
              v151 = (_QWORD *)sub_15FDF90(v67, v70, (__int64)&v195, 0);
              v71 = (__int64)sub_1A1C7B0((__int64 *)&v213, v151, &v193);
            }
            else
            {
              v71 = sub_15A4AD0((__int64 ***)v67, v70);
            }
            v68 = v173;
            v67 = v71;
          }
          v194 = 257;
          if ( v70 != *(_QWORD *)v68 )
          {
            if ( *(_BYTE *)(v68 + 16) > 0x10u )
            {
              LOWORD(v196) = 257;
              v152 = (_QWORD *)sub_15FDF90(v68, v70, (__int64)&v195, 0);
              v68 = (__int64)sub_1A1C7B0((__int64 *)&v213, v152, &v193);
            }
            else
            {
              v68 = sub_15A4AD0((__int64 ***)v68, v70);
            }
          }
        }
        v174 = (_BYTE *)v68;
        v195.m128i_i64[0] = (__int64)"select.gep.sroa";
        LOWORD(v196) = 259;
        v72 = sub_1A1D720((__int64 *)&v213, (_BYTE *)v67, (__int64 **)v207, (unsigned int)v208, &v195);
        v195.m128i_i64[0] = (__int64)"select.gep.sroa";
        v73 = v72;
        LOWORD(v196) = 259;
        v74 = sub_1A1D720((__int64 *)&v213, v174, (__int64 **)v207, (unsigned int)v208, &v195);
        v75 = *(_QWORD *)(a1 - 72);
        v194 = 259;
        v193.m128i_i64[0] = (__int64)"select.sroa";
        if ( *(_BYTE *)(v75 + 16) > 0x10u || *(_BYTE *)(v73 + 16) > 0x10u || *(_BYTE *)(v74 + 16) > 0x10u )
        {
          v156 = v74;
          LOWORD(v196) = 257;
          v83 = sub_1648A60(56, 3u);
          v84 = v83;
          if ( v83 )
          {
            v158 = v83 - 9;
            v175 = (__int64)v83;
            sub_15F1EA0((__int64)v83, *(_QWORD *)v73, 55, (__int64)(v83 - 9), 3, 0);
            if ( *(_QWORD *)(v175 - 72) )
            {
              v85 = *(_QWORD *)(v175 - 64);
              v86 = *(_QWORD *)(v175 - 56) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v86 = v85;
              if ( v85 )
                *(_QWORD *)(v85 + 16) = *(_QWORD *)(v85 + 16) & 3LL | v86;
            }
            *(_QWORD *)(v175 - 72) = v75;
            v87 = *(_QWORD *)(v75 + 8);
            *(_QWORD *)(v175 - 64) = v87;
            if ( v87 )
              *(_QWORD *)(v87 + 16) = (v175 - 64) | *(_QWORD *)(v87 + 16) & 3LL;
            *(_QWORD *)(v175 - 56) = (v75 + 8) | *(_QWORD *)(v175 - 56) & 3LL;
            *(_QWORD *)(v75 + 8) = v158;
            if ( *(_QWORD *)(v175 - 48) )
            {
              v88 = *(_QWORD *)(v175 - 40);
              v89 = *(_QWORD *)(v175 - 32) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v89 = v88;
              if ( v88 )
                *(_QWORD *)(v88 + 16) = *(_QWORD *)(v88 + 16) & 3LL | v89;
            }
            *(_QWORD *)(v175 - 48) = v73;
            v90 = *(_QWORD *)(v73 + 8);
            *(_QWORD *)(v175 - 40) = v90;
            if ( v90 )
              *(_QWORD *)(v90 + 16) = (v175 - 40) | *(_QWORD *)(v90 + 16) & 3LL;
            *(_QWORD *)(v175 - 32) = (v73 + 8) | *(_QWORD *)(v175 - 32) & 3LL;
            *(_QWORD *)(v73 + 8) = v175 - 48;
            if ( *(_QWORD *)(v175 - 24) )
            {
              v91 = *(_QWORD *)(v175 - 16);
              v92 = *(_QWORD *)(v175 - 8) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v92 = v91;
              if ( v91 )
                *(_QWORD *)(v91 + 16) = *(_QWORD *)(v91 + 16) & 3LL | v92;
            }
            *(_QWORD *)(v175 - 24) = v156;
            if ( v156 )
            {
              v93 = *(_QWORD *)(v156 + 8);
              *(_QWORD *)(v175 - 16) = v93;
              if ( v93 )
                *(_QWORD *)(v93 + 16) = (v175 - 16) | *(_QWORD *)(v93 + 16) & 3LL;
              *(_QWORD *)(v175 - 8) = *(_QWORD *)(v175 - 8) & 3LL | (v156 + 8);
              *(_QWORD *)(v156 + 8) = v175 - 24;
            }
            sub_164B780(v175, v195.m128i_i64);
            v84 = (_QWORD *)v175;
          }
          v76 = (__int64)sub_1A1C7B0((__int64 *)&v213, v84, &v193);
          v77 = v177;
          if ( v177 )
          {
LABEL_109:
            v214 = v77;
            v215 = v165;
            if ( v165 == v77 + 40 )
              goto LABEL_116;
            if ( !v165 )
              BUG();
            v78 = *(_QWORD *)(v165 + 24);
            v195.m128i_i64[0] = v78;
            if ( v78 )
            {
              sub_1623A60((__int64)&v195, v78, 2);
              v79 = v213;
              if ( !v213 )
                goto LABEL_114;
            }
            else
            {
              v79 = v213;
              if ( !v213 )
                goto LABEL_116;
            }
            sub_161E7C0((__int64)&v213, (__int64)v79);
LABEL_114:
            v213 = (unsigned __int8 *)v195.m128i_i64[0];
            if ( v195.m128i_i64[0] )
              sub_1623210((__int64)&v195, (unsigned __int8 *)v195.m128i_i64[0], (__int64)&v213);
LABEL_116:
            v80 = v207;
            if ( v207 == (unsigned __int8 *)v209 )
            {
LABEL_118:
              *((_QWORD *)v188 + 1) = v76;
              goto LABEL_58;
            }
LABEL_117:
            _libc_free((unsigned __int64)v80);
            goto LABEL_118;
          }
        }
        else
        {
          v76 = sub_15A2DC0(v75, (__int64 *)v73, v74, 0);
          v77 = v177;
          if ( v177 )
            goto LABEL_109;
        }
        v214 = 0;
        v215 = 0;
        goto LABEL_116;
      }
      v94 = (__int64 *)(v58 + 24 * (1LL - (v59 & 0xFFFFFFF)));
      if ( (__int64 *)v58 != v94 )
      {
        v95 = *v94;
        v96 = (unsigned __int8 *)v209;
        v97 = 0;
        v98 = (__int64 *)v58;
        while ( 1 )
        {
          *(_QWORD *)&v96[8 * v97] = v95;
          v94 += 3;
          v97 = (unsigned int)(v208 + 1);
          LODWORD(v208) = v208 + 1;
          if ( v98 == v94 )
            break;
          v95 = *v94;
          if ( HIDWORD(v208) <= (unsigned int)v97 )
          {
            sub_16CD150((__int64)&v207, v209, 0, 8, v58, (int)v40);
            v97 = (unsigned int)v208;
          }
          v96 = v207;
        }
        v58 = (__int64)v98;
      }
      v99 = *(_QWORD *)(a1 + 48);
      v154 = v215;
      v195.m128i_i64[0] = v99;
      v157 = v214;
      v214 = *(_QWORD *)(a1 + 40);
      v215 = a1 + 24;
      if ( v99 )
      {
        v178 = v58;
        sub_1623A60((__int64)&v195, v99, 2);
        v100 = v213;
        v58 = v178;
        if ( !v213 )
          goto LABEL_163;
      }
      else
      {
        v100 = v213;
        if ( !v213 )
          goto LABEL_165;
      }
      v179 = v58;
      sub_161E7C0((__int64)&v213, (__int64)v100);
      v58 = v179;
LABEL_163:
      v213 = (unsigned __int8 *)v195.m128i_i64[0];
      if ( v195.m128i_i64[0] )
      {
        v180 = v58;
        sub_1623210((__int64)&v195, (unsigned __int8 *)v195.m128i_i64[0], (__int64)&v213);
        v58 = v180;
      }
LABEL_165:
      v101 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
      v166 = **(_QWORD **)(v58 - 24LL * (*(_DWORD *)(v58 + 20) & 0xFFFFFFF));
      v102 = *(_DWORD *)(v166 + 8);
      v103 = (__int64 *)sub_15F9F50(*(_QWORD *)(v58 + 56), (__int64)v207, (unsigned int)v208);
      v104 = sub_1646BA0(v103, v102 >> 8);
      v194 = 259;
      v193.m128i_i64[0] = (__int64)"phi.sroa";
      LOWORD(v196) = 257;
      v105 = sub_1648B60(64);
      v106 = v105;
      if ( v105 )
      {
        sub_15F1EA0(v105, v104, 53, 0, 0, 0);
        *(_DWORD *)(v106 + 56) = v101;
        sub_164B780(v106, v195.m128i_i64);
        sub_1648880(v106, *(_DWORD *)(v106 + 56), 1);
      }
      v107 = sub_1A1C7B0((__int64 *)&v213, (_QWORD *)v106, &v193);
      v195 = 0u;
      v76 = (__int64)v107;
      LODWORD(v107) = *(_DWORD *)(a1 + 20);
      v196 = 0;
      v197 = 0;
      v109 = (unsigned int)v107 & 0xFFFFFFF;
      if ( v109 )
      {
        v155 = v36;
        v110 = 0;
        v111 = 0;
        v112 = 0;
        v181 = 8LL * v109;
        while ( 1 )
        {
          v129 = *(unsigned __int8 *)(a1 + 23);
          v113 = (v129 & 0x40) != 0 ? *(_QWORD *)(a1 - 8) : a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
          v114 = *(_QWORD *)(v112 + v113 + 24LL * *(unsigned int *)(a1 + 56) + 8);
          if ( !(_DWORD)v110 )
            break;
          v115 = ((_DWORD)v110 - 1) & (((unsigned int)v114 >> 9) ^ ((unsigned int)v114 >> 4));
          v116 = (__int64 *)(v111 + 16 * v115);
          v117 = *v116;
          if ( *v116 != v114 )
          {
            v133 = 1;
            v108 = 0;
            while ( v117 != -8 )
            {
              if ( !v108 && v117 == -16 )
                v108 = v116;
              v115 = ((_DWORD)v110 - 1) & (unsigned int)(v133 + v115);
              v116 = (__int64 *)(v111 + 16LL * (unsigned int)v115);
              v117 = *v116;
              if ( v114 == *v116 )
                goto LABEL_172;
              ++v133;
            }
            if ( !v108 )
              v108 = v116;
            ++v195.m128i_i64[0];
            v134 = v196 + 1;
            if ( 4 * ((int)v196 + 1) < (unsigned int)(3 * v110) )
            {
              if ( (int)v110 - (v134 + HIDWORD(v196)) <= (unsigned int)v110 >> 3 )
              {
                sub_141A900((__int64)&v195, v110);
                if ( !v197 )
                {
LABEL_283:
                  LODWORD(v196) = v196 + 1;
                  BUG();
                }
                v146 = 0;
                LODWORD(v147) = (v197 - 1) & (((unsigned int)v114 >> 9) ^ ((unsigned int)v114 >> 4));
                v148 = 1;
                v134 = v196 + 1;
                v108 = (__int64 *)(v195.m128i_i64[1] + 16LL * (unsigned int)v147);
                v149 = *v108;
                if ( *v108 != v114 )
                {
                  while ( v149 != -8 )
                  {
                    if ( !v146 && v149 == -16 )
                      v146 = v108;
                    v147 = (v197 - 1) & ((_DWORD)v147 + v148);
                    v108 = (__int64 *)(v195.m128i_i64[1] + 16 * v147);
                    v149 = *v108;
                    if ( v114 == *v108 )
                      goto LABEL_217;
                    ++v148;
                  }
                  if ( v146 )
                    v108 = v146;
                }
              }
              goto LABEL_217;
            }
LABEL_239:
            sub_141A900((__int64)&v195, 2 * v110);
            if ( !v197 )
              goto LABEL_283;
            LODWORD(v142) = (v197 - 1) & (((unsigned int)v114 >> 9) ^ ((unsigned int)v114 >> 4));
            v134 = v196 + 1;
            v108 = (__int64 *)(v195.m128i_i64[1] + 16LL * (unsigned int)v142);
            v143 = *v108;
            if ( v114 != *v108 )
            {
              v144 = 1;
              v145 = 0;
              while ( v143 != -8 )
              {
                if ( !v145 && v143 == -16 )
                  v145 = v108;
                v142 = (v197 - 1) & ((_DWORD)v142 + v144);
                v108 = (__int64 *)(v195.m128i_i64[1] + 16 * v142);
                v143 = *v108;
                if ( v114 == *v108 )
                  goto LABEL_217;
                ++v144;
              }
              if ( v145 )
                v108 = v145;
            }
LABEL_217:
            LODWORD(v196) = v134;
            if ( *v108 != -8 )
              --HIDWORD(v196);
            *v108 = v114;
            v108[1] = 0;
            LOBYTE(v129) = *(_BYTE *)(a1 + 23);
            goto LABEL_220;
          }
LABEL_172:
          v118 = v116[1];
          if ( v118 )
            goto LABEL_173;
          v108 = v116;
LABEL_220:
          if ( (v129 & 0x40) != 0 )
            v135 = *(_QWORD *)(a1 - 8);
          else
            v135 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
          v159 = v108;
          v136 = *(_QWORD *)(v135 + 3 * v112);
          v137 = sub_157EBA0(v114);
          v138 = v159;
          v214 = *(_QWORD *)(v137 + 40);
          v215 = v137 + 24;
          v193.m128i_i64[0] = *(_QWORD *)(v137 + 48);
          if ( v193.m128i_i64[0] )
          {
            sub_1623A60((__int64)&v193, v193.m128i_i64[0], 2);
            v139 = v213;
            v138 = v159;
            if ( v213 )
            {
LABEL_224:
              v160 = v138;
              sub_161E7C0((__int64)&v213, (__int64)v139);
              v138 = v160;
            }
            v213 = (unsigned __int8 *)v193.m128i_i64[0];
            if ( v193.m128i_i64[0] )
            {
              v161 = v138;
              sub_1623210((__int64)&v193, (unsigned __int8 *)v193.m128i_i64[0], (__int64)&v213);
              v138 = v161;
            }
            goto LABEL_227;
          }
          v139 = v213;
          if ( v213 )
            goto LABEL_224;
LABEL_227:
          v192 = 257;
          if ( v166 != *(_QWORD *)v136 )
          {
            v162 = v138;
            if ( *(_BYTE *)(v136 + 16) > 0x10u )
            {
              v194 = 257;
              v150 = (_QWORD *)sub_15FDF90(v136, v166, (__int64)&v193, 0);
              v140 = (__int64)sub_1A1C7B0((__int64 *)&v213, v150, &v191);
            }
            else
            {
              v140 = sub_15A4AD0((__int64 ***)v136, v166);
            }
            v138 = v162;
            v136 = v140;
          }
          v110 = v136;
          v163 = v138;
          v193.m128i_i64[0] = (__int64)"phi.gep.sroa";
          v194 = 259;
          v141 = sub_1A1D720((__int64 *)&v213, (_BYTE *)v136, (__int64 **)v207, (unsigned int)v208, &v193);
          v108 = v163;
          v118 = v141;
          v163[1] = v141;
LABEL_173:
          v119 = *(_DWORD *)(v76 + 20) & 0xFFFFFFF;
          if ( v119 == *(_DWORD *)(v76 + 56) )
          {
            sub_15F55D0(v76, v110, v129, v111, v115, (__int64)v108);
            v119 = *(_DWORD *)(v76 + 20) & 0xFFFFFFF;
          }
          v120 = (v119 + 1) & 0xFFFFFFF;
          v121 = v120 | *(_DWORD *)(v76 + 20) & 0xF0000000;
          *(_DWORD *)(v76 + 20) = v121;
          if ( (v121 & 0x40000000) != 0 )
            v122 = *(_QWORD *)(v76 - 8);
          else
            v122 = v76 - 24 * v120;
          v123 = (__int64 *)(v122 + 24LL * (unsigned int)(v120 - 1));
          if ( *v123 )
          {
            v124 = v123[1];
            v125 = v123[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v125 = v124;
            if ( v124 )
              *(_QWORD *)(v124 + 16) = *(_QWORD *)(v124 + 16) & 3LL | v125;
          }
          *v123 = v118;
          if ( v118 )
          {
            v126 = *(_QWORD *)(v118 + 8);
            v123[1] = v126;
            if ( v126 )
              *(_QWORD *)(v126 + 16) = (unsigned __int64)(v123 + 1) | *(_QWORD *)(v126 + 16) & 3LL;
            v123[2] = (v118 + 8) | v123[2] & 3;
            *(_QWORD *)(v118 + 8) = v123;
          }
          v127 = *(_DWORD *)(v76 + 20) & 0xFFFFFFF;
          if ( (*(_BYTE *)(v76 + 23) & 0x40) != 0 )
            v128 = *(_QWORD *)(v76 - 8);
          else
            v128 = v76 - 24 * v127;
          v112 += 8;
          *(_QWORD *)(v128 + 8LL * (unsigned int)(v127 - 1) + 24LL * *(unsigned int *)(v76 + 56) + 8) = v114;
          if ( v181 == v112 )
          {
            v36 = v155;
            goto LABEL_193;
          }
          v111 = v195.m128i_i64[1];
          v110 = v197;
        }
        ++v195.m128i_i64[0];
        goto LABEL_239;
      }
LABEL_193:
      if ( !v157 )
      {
        v214 = 0;
        v215 = 0;
        goto LABEL_201;
      }
      v214 = v157;
      v215 = v154;
      if ( v154 == v157 + 40 )
        goto LABEL_201;
      if ( !v154 )
        BUG();
      v130 = *(_QWORD *)(v154 + 24);
      v193.m128i_i64[0] = v130;
      if ( v130 )
      {
        sub_1623A60((__int64)&v193, v130, 2);
        v131 = v213;
        if ( !v213 )
          goto LABEL_199;
      }
      else
      {
        v131 = v213;
        if ( !v213 )
          goto LABEL_201;
      }
      sub_161E7C0((__int64)&v213, (__int64)v131);
LABEL_199:
      v213 = (unsigned __int8 *)v193.m128i_i64[0];
      if ( v193.m128i_i64[0] )
        sub_1623210((__int64)&v193, (unsigned __int8 *)v193.m128i_i64[0], (__int64)&v213);
LABEL_201:
      j___libc_free_0(v195.m128i_i64[1]);
      v80 = v207;
      if ( v207 == (unsigned __int8 *)v209 )
        goto LABEL_118;
      goto LABEL_117;
    }
LABEL_58:
    LOWORD(v196) = 257;
    v51 = *(_QWORD *)*v36;
    v52 = *((_QWORD *)v188 + 1);
    if ( v51 != *(_QWORD *)v52 )
    {
      if ( *(_BYTE *)(v52 + 16) > 0x10u )
      {
        LOWORD(v209[0]) = 257;
        v81 = (_QWORD *)sub_15FDF90(v52, v51, (__int64)&v207, 0);
        v52 = (__int64)sub_1A1C7B0((__int64 *)&v213, v81, &v195);
      }
      else
      {
        v52 = sub_15A4AD0((__int64 ***)v52, v51);
      }
    }
    sub_164D160(*v36, v52, a2, a3, a4, a5, v41, v42, a8, a9);
    sub_15F20C0((_QWORD *)*v36);
    if ( v221 != v223 )
      j_j___libc_free_0(v221, v223[0] + 1LL);
    if ( v213 )
      sub_161E7C0((__int64)&v213, (__int64)v213);
    if ( v190 > 0x40 && v189 )
      j_j___libc_free_0_0(v189);
    v37 = 1;
LABEL_69:
    if ( v186 != (const char *)++v36 )
      continue;
    break;
  }
  v169 = v37;
LABEL_121:
  if ( (v210.m128i_i8[8] & 1) != 0 )
  {
    v82 = (unsigned __int64)v204;
    if ( v204 != v206 )
      goto LABEL_123;
  }
  else
  {
    j___libc_free_0(v211);
    v82 = (unsigned __int64)v204;
    if ( v204 != v206 )
LABEL_123:
      _libc_free(v82);
  }
  if ( v201 != (__int64 *)v203 )
    _libc_free((unsigned __int64)v201);
  if ( v198 != (__int64 *)v200 )
    _libc_free((unsigned __int64)v198);
  return v169;
}
