// Function: sub_1FF5310
// Address: 0x1ff5310
//
__int64 *__fastcall sub_1FF5310(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  unsigned int v6; // r13d
  __int64 *v8; // rax
  __int64 v9; // rsi
  unsigned __int64 v10; // rcx
  __m128i v11; // xmm0
  __m128i v12; // xmm1
  __int64 v13; // r12
  unsigned int v14; // edx
  unsigned int v15; // r15d
  _QWORD *v16; // rdi
  unsigned __int8 v17; // r10
  char v18; // al
  __int64 *result; // rax
  unsigned int v20; // edx
  __int64 v21; // r12
  __int64 v22; // rdx
  __int64 v23; // rcx
  _QWORD *v24; // r8
  _QWORD *v25; // r9
  __int64 v26; // rdi
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rcx
  _QWORD *v30; // r8
  _QWORD *v31; // r9
  __int64 v32; // r12
  __int64 v33; // rdx
  __int64 v34; // rdi
  char v35; // di
  unsigned __int64 v36; // rax
  unsigned int v37; // r15d
  unsigned int v38; // eax
  __int64 v39; // r9
  __m128i v40; // xmm2
  __int64 v41; // rax
  int v42; // eax
  __int64 v43; // r9
  unsigned __int8 v44; // dl
  unsigned int v45; // esi
  unsigned __int8 *v46; // rax
  _BYTE *v47; // r8
  unsigned __int8 v48; // di
  __int64 v49; // r11
  char v50; // cl
  __int64 v51; // r10
  unsigned int v52; // r12d
  __int64 v53; // rax
  unsigned int v54; // r9d
  char v55; // r8
  __int64 v56; // rax
  _QWORD *v57; // rsi
  unsigned __int8 v58; // al
  __int128 v59; // rax
  unsigned int v60; // edx
  unsigned int v61; // ecx
  __int64 v62; // rdx
  char v63; // cl
  unsigned int v64; // esi
  _QWORD *v65; // rdi
  unsigned int v66; // r15d
  unsigned int v67; // r11d
  unsigned int v68; // r10d
  __int64 v69; // rax
  __int64 v70; // rcx
  __int64 v71; // rsi
  __int64 v72; // rax
  __int64 v73; // rcx
  __int64 v74; // rsi
  __int64 v75; // r12
  _BYTE *v76; // rax
  unsigned __int16 v77; // r9
  unsigned int v78; // edx
  __int128 v79; // rax
  __int64 *v80; // rax
  unsigned int v81; // edx
  _QWORD *v82; // rdi
  __int64 v83; // rax
  unsigned __int64 v84; // rcx
  int v85; // eax
  unsigned int v86; // edx
  __int64 v87; // r12
  __int64 *v88; // r15
  unsigned int v89; // edx
  __int64 v90; // rax
  const void **v91; // rdx
  __int128 v92; // rax
  unsigned int v93; // edx
  unsigned int v94; // edx
  __int64 v95; // r12
  __int64 v96; // rdx
  int v97; // eax
  __int64 v98; // rdi
  char v99; // r12
  __int64 v100; // r10
  __int64 v101; // rax
  char *v102; // rcx
  unsigned __int8 v103; // r9
  __int64 v104; // r8
  __int64 v105; // rax
  __int64 v106; // rdx
  __int64 v107; // r8
  __int64 v108; // r9
  __int64 v109; // rcx
  __int64 v110; // r11
  __int64 v111; // rdx
  __int64 v112; // rcx
  _QWORD *v113; // r8
  _QWORD *v114; // r9
  __int64 v115; // rdi
  __int64 v116; // rdi
  __int64 v117; // rdx
  __int64 v118; // rcx
  _QWORD *v119; // r8
  _QWORD *v120; // r9
  __int64 v121; // r12
  __int64 v122; // rdx
  __int64 v123; // rdi
  __int64 v124; // rdx
  __int64 v125; // rsi
  int v126; // r11d
  int v127; // ecx
  __int64 v128; // r12
  unsigned int v129; // r15d
  __int64 v130; // rax
  __int64 v131; // r8
  unsigned int v132; // r9d
  __int64 v133; // rdx
  __int64 *v134; // r12
  __int128 v135; // rax
  char v136; // al
  __int64 v137; // r12
  __int64 v138; // rdx
  __int64 v139; // r13
  unsigned __int8 v140; // al
  __int128 v141; // rax
  unsigned int v142; // edx
  __int64 v143; // rdx
  unsigned __int8 v144; // r10
  char v145; // si
  __int128 v146; // rax
  char v147; // di
  unsigned int v148; // eax
  unsigned int v149; // edx
  __int64 v150; // rax
  __int64 v151; // rdx
  __int128 v152; // rax
  __int64 v153; // rcx
  __int64 v154; // r8
  __int64 v155; // r9
  __int128 v156; // kr10_16
  __int64 v157; // r8
  __int64 *v158; // r12
  __int64 v159; // r9
  unsigned int v160; // edx
  unsigned int v161; // edx
  unsigned int v162; // edx
  __int128 v163; // rax
  __int64 *v164; // rax
  unsigned int v165; // edx
  _QWORD *v166; // rdi
  __int64 v167; // rax
  unsigned __int64 v168; // rcx
  int v169; // eax
  unsigned int v170; // edx
  unsigned int v171; // edx
  __int64 v172; // rax
  const void **v173; // rdx
  __int128 v174; // rax
  unsigned int v175; // edx
  unsigned int v176; // edx
  __int64 *v177; // r12
  __int128 v178; // rax
  __int64 v179; // rcx
  __int64 v180; // rdx
  __int64 v181; // r8
  char v182; // si
  __int64 v183; // rax
  bool v184; // zf
  __int64 v185; // r8
  char v186; // si
  __int64 v187; // rax
  __int64 *v188; // r12
  __int128 v189; // rax
  bool v190; // al
  __int64 v191; // rdx
  __int128 v192; // [rsp-10h] [rbp-250h]
  __int128 v193; // [rsp-10h] [rbp-250h]
  __int64 *v194; // [rsp-10h] [rbp-250h]
  __int128 v195; // [rsp-10h] [rbp-250h]
  __int128 v196; // [rsp-10h] [rbp-250h]
  unsigned int v197; // [rsp+8h] [rbp-238h]
  __int64 *v198; // [rsp+8h] [rbp-238h]
  __int64 v199; // [rsp+10h] [rbp-230h]
  __int64 v200; // [rsp+10h] [rbp-230h]
  __int64 v201; // [rsp+18h] [rbp-228h]
  unsigned int v202; // [rsp+18h] [rbp-228h]
  __int64 v203; // [rsp+18h] [rbp-228h]
  __int64 v204; // [rsp+20h] [rbp-220h]
  unsigned int v205; // [rsp+20h] [rbp-220h]
  unsigned __int16 v206; // [rsp+28h] [rbp-218h]
  __int64 v207; // [rsp+28h] [rbp-218h]
  unsigned __int8 v208; // [rsp+30h] [rbp-210h]
  __int64 v209; // [rsp+30h] [rbp-210h]
  unsigned int v210; // [rsp+30h] [rbp-210h]
  __int64 v211; // [rsp+38h] [rbp-208h]
  __int64 v212; // [rsp+40h] [rbp-200h]
  __int64 v213; // [rsp+40h] [rbp-200h]
  __int64 v214; // [rsp+40h] [rbp-200h]
  __int64 v215; // [rsp+40h] [rbp-200h]
  __int64 v216; // [rsp+48h] [rbp-1F8h]
  unsigned __int16 v217; // [rsp+50h] [rbp-1F0h]
  __int64 v218; // [rsp+50h] [rbp-1F0h]
  unsigned int v219; // [rsp+58h] [rbp-1E8h]
  __int64 *v220; // [rsp+58h] [rbp-1E8h]
  __int64 v221; // [rsp+58h] [rbp-1E8h]
  __int64 v222; // [rsp+58h] [rbp-1E8h]
  unsigned __int64 v223; // [rsp+60h] [rbp-1E0h]
  unsigned int v224; // [rsp+60h] [rbp-1E0h]
  unsigned int v225; // [rsp+60h] [rbp-1E0h]
  __int64 v226; // [rsp+60h] [rbp-1E0h]
  unsigned __int64 v227; // [rsp+68h] [rbp-1D8h]
  unsigned __int64 v228; // [rsp+68h] [rbp-1D8h]
  __int64 v229; // [rsp+70h] [rbp-1D0h]
  __int64 v230; // [rsp+70h] [rbp-1D0h]
  __int64 *v231; // [rsp+70h] [rbp-1D0h]
  __int64 v232; // [rsp+70h] [rbp-1D0h]
  __int64 *v233; // [rsp+70h] [rbp-1D0h]
  __int128 v234; // [rsp+70h] [rbp-1D0h]
  int v235; // [rsp+80h] [rbp-1C0h]
  unsigned int v236; // [rsp+80h] [rbp-1C0h]
  unsigned __int8 v237; // [rsp+80h] [rbp-1C0h]
  __int64 *v238; // [rsp+80h] [rbp-1C0h]
  __int64 v239; // [rsp+80h] [rbp-1C0h]
  __int64 v240; // [rsp+80h] [rbp-1C0h]
  __int64 *v241; // [rsp+80h] [rbp-1C0h]
  __int64 v242; // [rsp+80h] [rbp-1C0h]
  unsigned __int64 v243; // [rsp+88h] [rbp-1B8h]
  unsigned __int64 v244; // [rsp+88h] [rbp-1B8h]
  unsigned __int64 v245; // [rsp+88h] [rbp-1B8h]
  unsigned __int64 v246; // [rsp+88h] [rbp-1B8h]
  __int64 v247; // [rsp+90h] [rbp-1B0h]
  __int64 v248; // [rsp+90h] [rbp-1B0h]
  unsigned int v249; // [rsp+90h] [rbp-1B0h]
  unsigned int v250; // [rsp+90h] [rbp-1B0h]
  __int64 v251; // [rsp+90h] [rbp-1B0h]
  __int64 *v252; // [rsp+90h] [rbp-1B0h]
  __int128 v253; // [rsp+90h] [rbp-1B0h]
  __int64 v254; // [rsp+90h] [rbp-1B0h]
  unsigned int v255; // [rsp+90h] [rbp-1B0h]
  __int128 v256; // [rsp+90h] [rbp-1B0h]
  __int64 *v257; // [rsp+A0h] [rbp-1A0h]
  unsigned int v258; // [rsp+A0h] [rbp-1A0h]
  unsigned int v259; // [rsp+A8h] [rbp-198h]
  __int64 v260; // [rsp+B0h] [rbp-190h]
  __int64 v261; // [rsp+188h] [rbp-B8h] BYREF
  __int64 v262; // [rsp+190h] [rbp-B0h] BYREF
  int v263; // [rsp+198h] [rbp-A8h]
  __int64 v264; // [rsp+1A0h] [rbp-A0h] BYREF
  unsigned __int64 v265; // [rsp+1A8h] [rbp-98h]
  __m128i v266; // [rsp+1B0h] [rbp-90h] BYREF
  __int64 v267; // [rsp+1C0h] [rbp-80h]
  __int128 v268; // [rsp+1D0h] [rbp-70h]
  __int64 v269; // [rsp+1E0h] [rbp-60h]
  __int128 v270; // [rsp+1F0h] [rbp-50h] BYREF
  __int64 *v271; // [rsp+200h] [rbp-40h]
  unsigned int v272; // [rsp+208h] [rbp-38h]

  v6 = 0;
  v8 = *(__int64 **)(a2 + 32);
  v9 = *(_QWORD *)(a2 + 72);
  v10 = *v8;
  v11 = _mm_loadu_si128((const __m128i *)v8);
  v262 = v9;
  v12 = _mm_loadu_si128((const __m128i *)(v8 + 5));
  v13 = *((unsigned int *)v8 + 12);
  v223 = v10;
  v229 = v8[5];
  if ( v9 )
    sub_1623A60((__int64)&v262, v9, 2);
  v263 = *(_DWORD *)(a2 + 64);
  v235 = (*(_BYTE *)(a2 + 27) >> 2) & 3;
  if ( ((*(_BYTE *)(a2 + 27) >> 2) & 3) == 0 )
  {
    v14 = *(unsigned __int16 *)(a2 + 24);
    v15 = 1;
    v236 = 0;
    v16 = *(_QWORD **)(a1 + 8);
    v17 = **(_BYTE **)(a2 + 40);
    if ( v14 <= 0x102 )
    {
      v18 = *((_BYTE *)v16 + 259 * v17 + *(unsigned __int16 *)(a2 + 24) + 2422);
      if ( v18 == 1 )
      {
        v56 = v16[9258];
        if ( !v56 )
          goto LABEL_41;
        v57 = v16 + 9257;
        do
        {
          while ( v14 <= *(_DWORD *)(v56 + 32) && (v14 != *(_DWORD *)(v56 + 32) || v17 <= *(_BYTE *)(v56 + 36)) )
          {
            v57 = (_QWORD *)v56;
            v56 = *(_QWORD *)(v56 + 16);
            if ( !v56 )
              goto LABEL_39;
          }
          v56 = *(_QWORD *)(v56 + 24);
        }
        while ( v56 );
LABEL_39:
        if ( v16 + 9257 == v57
          || v14 < *((_DWORD *)v57 + 8)
          || v14 == *((_DWORD *)v57 + 8) && v17 < *((_BYTE *)v57 + 36) )
        {
LABEL_41:
          v58 = **(_BYTE **)(a2 + 40);
          do
          {
            do
              ++v58;
            while ( !v58 );
          }
          while ( !v16[v58 + 15] || *((_BYTE *)v16 + 259 * v58 + *(unsigned __int16 *)(a2 + 24) + 2422) == 1 );
        }
        else
        {
          v58 = *((_BYTE *)v57 + 40);
        }
        v237 = **(_BYTE **)(a2 + 40);
        *(_QWORD *)&v59 = sub_1D2B660(
                            *(_QWORD **)(a1 + 16),
                            v58,
                            0,
                            (__int64)&v262,
                            v11.m128i_i64[0],
                            v11.m128i_i64[1],
                            v12.m128i_i64[0],
                            v12.m128i_i64[1],
                            *(_QWORD *)(a2 + 104));
        v21 = v59;
        result = (__int64 *)sub_1D309E0(
                              *(__int64 **)(a1 + 16),
                              158,
                              (__int64)&v262,
                              v237,
                              0,
                              0,
                              *(double *)v11.m128i_i64,
                              *(double *)v12.m128i_i64,
                              a5,
                              v59);
        v236 = v60;
      }
      else
      {
        if ( v18 == 4 )
          goto LABEL_7;
        v230 = *(_QWORD *)(a2 + 96);
        v247 = *(unsigned __int8 *)(a2 + 88);
        v52 = sub_1E340A0(*(_QWORD *)(a2 + 104));
        v258 = sub_1E34390(*(_QWORD *)(a2 + 104));
        v53 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL));
        v54 = v52;
        v21 = a2;
        v55 = sub_1F43CC0(*(_QWORD *)(a1 + 8), *(_QWORD *)(*(_QWORD *)(a1 + 16) + 48LL), v53, v247, v230, v54, v258, 0);
        result = (__int64 *)a2;
        if ( !v55 )
        {
          sub_20B9E10(&v270, *(_QWORD *)(a1 + 8), a2, *(_QWORD *)(a1 + 16));
          v15 = v272;
          v21 = (__int64)v271;
          v236 = DWORD2(v270);
          result = (__int64 *)v270;
        }
      }
LABEL_10:
      if ( a2 == v21 )
        goto LABEL_18;
      v257 = result;
      sub_1D44C70(*(_QWORD *)(a1 + 16), a2, 0, (__int64)result, v236);
      sub_1D44C70(*(_QWORD *)(a1 + 16), a2, 1, v21, v15);
      v26 = *(_QWORD *)(a1 + 32);
      if ( v26 )
      {
        *(_QWORD *)&v270 = v257;
        sub_1FF5010(v26, &v270, v22, v23, v24, v25);
        v27 = *(_QWORD *)(a1 + 32);
        *(_QWORD *)&v270 = v21;
        sub_1FF5010(v27, &v270, v28, v29, v30, v31);
      }
      v32 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)&v270 = a2;
      result = *(__int64 **)(v32 + 8);
      if ( *(__int64 **)(v32 + 16) == result )
      {
        v33 = (__int64)&result[*(unsigned int *)(v32 + 28)];
        if ( result == (__int64 *)v33 )
        {
LABEL_108:
          result = (__int64 *)v33;
        }
        else
        {
          while ( a2 != *result )
          {
            if ( (__int64 *)v33 == ++result )
              goto LABEL_108;
          }
        }
      }
      else
      {
        result = sub_16CC9F0(v32, a2);
        if ( a2 == *result )
        {
          v124 = *(_QWORD *)(v32 + 16);
          if ( v124 == *(_QWORD *)(v32 + 8) )
            v23 = *(unsigned int *)(v32 + 28);
          else
            v23 = *(unsigned int *)(v32 + 24);
          v33 = v124 + 8 * v23;
        }
        else
        {
          result = *(__int64 **)(v32 + 16);
          if ( result != *(__int64 **)(v32 + 8) )
          {
LABEL_16:
            v34 = *(_QWORD *)(a1 + 32);
            if ( v34 )
              result = (__int64 *)sub_1FF5010(v34, &v270, v33, v23, v24, v25);
            goto LABEL_18;
          }
          result += *(unsigned int *)(v32 + 28);
          v33 = (__int64)result;
        }
      }
      if ( (__int64 *)v33 != result )
      {
        *result = -2;
        ++*(_DWORD *)(v32 + 32);
      }
      goto LABEL_16;
    }
LABEL_7:
    result = (__int64 *)(*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD, _QWORD))(*v16 + 1312LL))(
                          v16,
                          a2,
                          0,
                          *(_QWORD *)(a1 + 16));
    if ( result )
      v236 = v20;
    else
      result = (__int64 *)a2;
    v21 = (__int64)result;
    goto LABEL_10;
  }
  v35 = *(_BYTE *)(a2 + 88);
  v36 = *(_QWORD *)(a2 + 96);
  LOBYTE(v264) = v35;
  v265 = v36;
  if ( v35 )
    v37 = sub_1FEB8F0(v35);
  else
    v37 = sub_1F58D40((__int64)&v264);
  v38 = sub_1E34390(*(_QWORD *)(a2 + 104));
  v39 = *(_QWORD *)(a2 + 104);
  v219 = v38;
  v40 = _mm_loadu_si128((const __m128i *)(v39 + 40));
  v212 = v39;
  v217 = *(_WORD *)(v39 + 32);
  v41 = *(_QWORD *)(v39 + 56);
  v266 = v40;
  v267 = v41;
  if ( !(_BYTE)v264 )
  {
    v207 = v39;
    v97 = sub_1F58D40((__int64)&v264);
    v44 = 0;
    v43 = v207;
    v45 = (v97 + 7) & 0xFFFFFFF8;
    if ( v45 != v37 )
    {
LABEL_74:
      v98 = *(_QWORD *)(a1 + 16);
      if ( v45 == 32 )
      {
        v99 = 5;
        goto LABEL_78;
      }
      if ( v45 > 0x20 )
      {
        if ( v45 == 64 )
        {
          v99 = 6;
          goto LABEL_78;
        }
        if ( v45 == 128 )
        {
          v99 = 7;
          goto LABEL_78;
        }
      }
      else
      {
        if ( v45 == 8 )
        {
          v99 = 3;
          goto LABEL_78;
        }
        v99 = 4;
        if ( v45 == 16 )
        {
LABEL_78:
          v100 = 0;
          goto LABEL_79;
        }
      }
      v150 = sub_1F58CC0(*(_QWORD **)(v98 + 48), v45);
      v98 = *(_QWORD *)(a1 + 16);
      v204 = v150;
      v99 = v150;
      v100 = v151;
      v212 = *(_QWORD *)(a2 + 104);
LABEL_79:
      v101 = v204;
      LOBYTE(v101) = v99;
      v102 = *(char **)(a2 + 40);
      v103 = *v102;
      v104 = *((_QWORD *)v102 + 1);
      if ( v235 == 3 )
      {
        v107 = sub_1D2B810(
                 (_QWORD *)v98,
                 3u,
                 (__int64)&v262,
                 v103,
                 v104,
                 v219,
                 *(_OWORD *)&v11,
                 v12.m128i_i64[0],
                 v12.m128i_i64[1],
                 *(_OWORD *)v212,
                 *(_QWORD *)(v212 + 16),
                 v101,
                 v100,
                 v217,
                 (__int64)&v266);
        v108 = v180;
        v88 = (__int64 *)v107;
        v110 = 16LL * (unsigned int)v180;
      }
      else
      {
        v232 = v100;
        v105 = sub_1D2B810(
                 (_QWORD *)v98,
                 1u,
                 (__int64)&v262,
                 v103,
                 v104,
                 v219,
                 *(_OWORD *)&v11,
                 v12.m128i_i64[0],
                 v12.m128i_i64[1],
                 *(_OWORD *)v212,
                 *(_QWORD *)(v212 + 16),
                 v101,
                 v100,
                 v217,
                 (__int64)&v266);
        v107 = v105;
        v108 = v106;
        v88 = (__int64 *)v105;
        v109 = (unsigned int)v106;
        if ( v235 == 2 )
        {
          v177 = *(__int64 **)(a1 + 16);
          v242 = v105;
          v246 = v106;
          v255 = v106;
          *(_QWORD *)&v178 = sub_1D2EF30(v177, (unsigned int)v264, v265, (unsigned int)v106, v105, v106);
          result = sub_1D332F0(
                     v177,
                     148,
                     (__int64)&v262,
                     *(unsigned __int8 *)(v88[5] + 16LL * v255),
                     *(const void ***)(v88[5] + 16LL * v255 + 8),
                     0,
                     *(double *)v11.m128i_i64,
                     *(double *)v12.m128i_i64,
                     v40,
                     v242,
                     v246,
                     v178);
          goto LABEL_124;
        }
        v110 = 16LL * (unsigned int)v106;
        result = (__int64 *)(v110 + *(_QWORD *)(v105 + 40));
        if ( *(_BYTE *)result != v99 || !v99 && result[1] != v232 )
        {
          v95 = v107;
LABEL_83:
          v6 = v106;
LABEL_84:
          v259 = 1;
          goto LABEL_85;
        }
      }
      v134 = *(__int64 **)(a1 + 16);
      v240 = v107;
      v244 = v108;
      v251 = v110;
      *(_QWORD *)&v135 = sub_1D2EF30(v134, (unsigned int)v264, v265, v109, v107, v108);
      result = sub_1D332F0(
                 v134,
                 4,
                 (__int64)&v262,
                 *(unsigned __int8 *)(v88[5] + v251),
                 *(const void ***)(v88[5] + v251 + 8),
                 0,
                 *(double *)v11.m128i_i64,
                 *(double *)v12.m128i_i64,
                 v40,
                 v240,
                 v244,
                 v135);
LABEL_124:
      v95 = (__int64)result;
      goto LABEL_83;
    }
LABEL_99:
    if ( (v37 & (v37 - 1)) != 0 )
      goto LABEL_53;
    v46 = *(unsigned __int8 **)(a2 + 40);
    v47 = *(_BYTE **)(a1 + 8);
    v48 = *v46;
    v51 = *((_QWORD *)v46 + 1);
    if ( *v46 )
    {
      if ( v44 )
      {
        v49 = v48;
        v50 = 4 * v235;
        goto LABEL_103;
      }
      LOBYTE(v270) = *v46;
      *((_QWORD *)&v270 + 1) = v51;
    }
    else
    {
      v125 = v44;
      LOBYTE(v270) = 0;
      *((_QWORD *)&v270 + 1) = v51;
      v144 = v47[v44 + 1155];
      if ( v44 )
        goto LABEL_141;
    }
    if ( sub_1F58D20((__int64)&v264) )
    {
      v136 = sub_1F596B0((__int64)&v264);
      goto LABEL_129;
    }
LABEL_128:
    v136 = v264;
LABEL_129:
    if ( v136 == 8 )
    {
LABEL_130:
      v137 = sub_1FF1380((char *)&v264);
      v139 = v138;
      v140 = sub_1FF1380((char *)&v270);
      *(_QWORD *)&v141 = sub_1D2B590(
                           *(_QWORD **)(a1 + 16),
                           3,
                           (__int64)&v262,
                           *(unsigned __int8 *)(*(_QWORD *)(a1 + 8) + v140 + 1155LL),
                           0,
                           *(_QWORD *)(a2 + 104),
                           __PAIR128__(v11.m128i_u64[1], v223),
                           v12.m128i_i64[0],
                           v12.m128i_i64[1],
                           v137,
                           v139);
      v252 = (__int64 *)v141;
      v260 = sub_1D309E0(
               *(__int64 **)(a1 + 16),
               160,
               (__int64)&v262,
               (unsigned int)v270,
               *((const void ***)&v270 + 1),
               0,
               *(double *)v11.m128i_i64,
               *(double *)v12.m128i_i64,
               *(double *)v40.m128i_i64,
               v141);
      v6 = v142;
      v95 = v260;
      v88 = v252;
LABEL_131:
      result = v194;
      v259 = 1;
      goto LABEL_85;
    }
    v46 = *(unsigned __int8 **)(a2 + 40);
    v43 = *(_QWORD *)(a2 + 104);
LABEL_153:
    *(_QWORD *)&v152 = sub_1D2B590(
                         *(_QWORD **)(a1 + 16),
                         1,
                         (__int64)&v262,
                         *v46,
                         *((_QWORD *)v46 + 1),
                         v43,
                         __PAIR128__(v11.m128i_u64[1], v223),
                         v12.m128i_i64[0],
                         v12.m128i_i64[1],
                         v264,
                         v265);
    v156 = v152;
    v88 = (__int64 *)v152;
    if ( v235 == 2 )
    {
      v188 = *(__int64 **)(a1 + 16);
      v256 = v152;
      *(_QWORD *)&v189 = sub_1D2EF30(v188, (unsigned int)v264, v265, v153, v154, v155);
      result = sub_1D332F0(
                 v188,
                 148,
                 (__int64)&v262,
                 *(unsigned __int8 *)(v88[5] + 16LL * DWORD2(v256)),
                 *(const void ***)(v88[5] + 16LL * DWORD2(v256) + 8),
                 0,
                 *(double *)v11.m128i_i64,
                 *(double *)v12.m128i_i64,
                 v40,
                 v256,
                 *((unsigned __int64 *)&v256 + 1),
                 v189);
      v95 = (__int64)result;
      goto LABEL_158;
    }
    v157 = (unsigned __int8)v264;
    v158 = *(__int64 **)(a1 + 16);
    if ( (_BYTE)v264 )
    {
      if ( (unsigned __int8)(v264 - 14) <= 0x5Fu )
      {
        switch ( (char)v264 )
        {
          case 24:
          case 25:
          case 26:
          case 27:
          case 28:
          case 29:
          case 30:
          case 31:
          case 32:
          case 62:
          case 63:
          case 64:
          case 65:
          case 66:
          case 67:
            v157 = 3;
            break;
          case 33:
          case 34:
          case 35:
          case 36:
          case 37:
          case 38:
          case 39:
          case 40:
          case 68:
          case 69:
          case 70:
          case 71:
          case 72:
          case 73:
            v157 = 4;
            break;
          case 41:
          case 42:
          case 43:
          case 44:
          case 45:
          case 46:
          case 47:
          case 48:
          case 74:
          case 75:
          case 76:
          case 77:
          case 78:
          case 79:
            v157 = 5;
            break;
          case 49:
          case 50:
          case 51:
          case 52:
          case 53:
          case 54:
          case 80:
          case 81:
          case 82:
          case 83:
          case 84:
          case 85:
            v157 = 6;
            break;
          case 55:
            v157 = 7;
            break;
          case 86:
          case 87:
          case 88:
          case 98:
          case 99:
          case 100:
            v157 = 8;
            break;
          case 89:
          case 90:
          case 91:
          case 92:
          case 93:
          case 101:
          case 102:
          case 103:
          case 104:
          case 105:
            v157 = 9;
            break;
          case 94:
          case 95:
          case 96:
          case 97:
          case 106:
          case 107:
          case 108:
          case 109:
            v157 = 10;
            break;
          default:
            v157 = 2;
            break;
        }
        v159 = 0;
        goto LABEL_157;
      }
    }
    else
    {
      v234 = v152;
      v190 = sub_1F58D20((__int64)&v264);
      v157 = 0;
      v156 = v234;
      if ( v190 )
      {
        v157 = (unsigned __int8)sub_1F596B0((__int64)&v264);
        v159 = v191;
        v156 = v234;
        goto LABEL_157;
      }
    }
    v159 = v265;
LABEL_157:
    result = sub_1D3BC50(
               v158,
               v156,
               *((unsigned __int64 *)&v156 + 1),
               (__int64)&v262,
               v157,
               v159,
               v11,
               *(double *)v12.m128i_i64,
               v40);
    v95 = (__int64)result;
LABEL_158:
    v6 = v160;
    goto LABEL_84;
  }
  v208 = v264;
  v42 = sub_1FEB8F0(v264);
  v44 = v208;
  v45 = (v42 + 7) & 0xFFFFFFF8;
  if ( v45 == v37 )
    goto LABEL_99;
  if ( v208 != 2 )
    goto LABEL_74;
  v46 = *(unsigned __int8 **)(a2 + 40);
  v47 = *(_BYTE **)(a1 + 8);
  v48 = *v46;
  if ( !*v46 )
  {
    if ( (v37 & (v37 - 1)) != 0 )
      goto LABEL_53;
    v179 = *((_QWORD *)v46 + 1);
    v125 = 2;
    LOBYTE(v270) = 0;
    *((_QWORD *)&v270 + 1) = v179;
    v144 = v47[1157];
LABEL_141:
    if ( *(_QWORD *)&v47[8 * v125 + 120] )
    {
      v145 = v235;
      if ( v44 == v144 )
      {
        v145 = 0;
        if ( !v144 )
        {
          if ( v265 )
            v145 = v235;
        }
      }
LABEL_146:
      *(_QWORD *)&v146 = sub_1D2B590(
                           *(_QWORD **)(a1 + 16),
                           v145,
                           (__int64)&v262,
                           v144,
                           0,
                           v43,
                           *(_OWORD *)&v11,
                           v12.m128i_i64[0],
                           v12.m128i_i64[1],
                           v264,
                           v265);
      v253 = v146;
      if ( (_BYTE)v264 )
        v147 = (unsigned __int8)(v264 - 86) <= 0x17u || (unsigned __int8)(v264 - 8) <= 5u;
      else
        v147 = sub_1F58CD0((__int64)&v264);
      v148 = sub_1D16EA0(v147, v235);
      v95 = sub_1D309E0(
              *(__int64 **)(a1 + 16),
              v148,
              (__int64)&v262,
              **(unsigned __int8 **)(a2 + 40),
              *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
              0,
              *(double *)v11.m128i_i64,
              *(double *)v12.m128i_i64,
              *(double *)v40.m128i_i64,
              v253);
      v6 = v149;
      v88 = (__int64 *)v253;
      goto LABEL_131;
    }
    if ( v144 && (((int)*(unsigned __int16 *)&v47[230 * v144 + 32208 + 2 * v44] >> (4 * v235)) & 0xF) == 0 )
    {
      v145 = 0;
      if ( v44 != v144 )
        v145 = v235;
      goto LABEL_146;
    }
    if ( (unsigned __int8)(v44 - 14) <= 0x5Fu )
    {
      switch ( v44 )
      {
        case 'V':
        case 'W':
        case 'X':
        case 'b':
        case 'c':
        case 'd':
          goto LABEL_130;
        default:
          goto LABEL_153;
      }
    }
    goto LABEL_128;
  }
  v49 = v48;
  v50 = 4 * v235;
  if ( (((int)*(unsigned __int16 *)&v47[230 * v48 + 32212] >> (4 * v235)) & 0xF) == 1 )
    goto LABEL_74;
  if ( (v37 & (v37 - 1)) == 0 )
  {
    v51 = *((_QWORD *)v46 + 1);
LABEL_103:
    v125 = v44;
    v126 = *(unsigned __int16 *)&v47[230 * v49 + 32208 + 2 * v44];
    v127 = (v126 >> v50) & 0xF;
    if ( (_BYTE)v127 != 2 )
    {
      if ( (_BYTE)v127 == 4 )
      {
        v259 = 1;
        result = (__int64 *)(*(__int64 (__fastcall **)(_BYTE *, __int64, _QWORD, _QWORD))(*(_QWORD *)v47 + 1312LL))(
                              v47,
                              a2,
                              0,
                              *(_QWORD *)(a1 + 16));
        v88 = result;
        if ( result )
          v6 = v161;
        else
          v88 = (__int64 *)a2;
        v95 = (__int64)v88;
      }
      else
      {
        v128 = *(_QWORD *)(a2 + 96);
        v239 = *(unsigned __int8 *)(a2 + 88);
        v259 = 1;
        v129 = sub_1E340A0(v43);
        v250 = sub_1E34390(*(_QWORD *)(a2 + 104));
        v130 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL));
        v131 = v128;
        v95 = a2;
        v132 = v129;
        v88 = (__int64 *)a2;
        result = (__int64 *)sub_1F43CC0(
                              *(_QWORD *)(a1 + 8),
                              *(_QWORD *)(*(_QWORD *)(a1 + 16) + 48LL),
                              v130,
                              v239,
                              v131,
                              v132,
                              v250,
                              0);
        if ( !(_BYTE)result )
        {
          sub_20B9E10(&v270, *(_QWORD *)(a1 + 8), a2, *(_QWORD *)(a1 + 16));
          v6 = DWORD2(v270);
          v95 = v270;
          v88 = v271;
          result = (__int64 *)(v272 | v11.m128i_i64[1] & 0xFFFFFFFF00000000LL);
          v259 = v272;
        }
      }
      goto LABEL_85;
    }
    LOBYTE(v270) = v48;
    *((_QWORD *)&v270 + 1) = v51;
    if ( !((unsigned __int8)v126 >> 4) )
      goto LABEL_153;
    v144 = v47[v44 + 1155];
    goto LABEL_141;
  }
LABEL_53:
  _BitScanReverse(&v61, v37);
  v62 = *(_QWORD *)(a1 + 16);
  v63 = v61 ^ 0x1F;
  v64 = 0x80000000 >> v63;
  v65 = *(_QWORD **)(v62 + 48);
  v66 = v37 - (0x80000000 >> v63);
  v67 = 0x80000000 >> v63;
  v68 = v66;
  if ( 0x80000000 >> v63 == 32 )
  {
    LOBYTE(v69) = 5;
    goto LABEL_57;
  }
  if ( v64 <= 0x20 )
  {
    if ( v64 == 8 )
    {
      LOBYTE(v69) = 3;
    }
    else
    {
      LOBYTE(v69) = 4;
      if ( v64 != 16 )
      {
        LOBYTE(v69) = 2;
        if ( v64 != 1 )
          goto LABEL_72;
      }
    }
LABEL_57:
    v70 = 0;
    goto LABEL_58;
  }
  if ( v64 == 64 )
  {
    LOBYTE(v69) = 6;
    goto LABEL_57;
  }
  if ( v64 == 128 )
  {
    LOBYTE(v69) = 7;
    goto LABEL_57;
  }
LABEL_72:
  v224 = 0x80000000 >> v63;
  v69 = sub_1F58CC0(v65, v67);
  v68 = v66;
  v67 = v224;
  v70 = v96;
  v62 = *(_QWORD *)(a1 + 16);
  v201 = v69;
  v65 = *(_QWORD **)(v62 + 48);
LABEL_58:
  v71 = v201;
  v216 = v70;
  LOBYTE(v71) = v69;
  v213 = v71;
  if ( v68 == 32 )
  {
    LOBYTE(v72) = 5;
  }
  else if ( v68 > 0x20 )
  {
    if ( v68 == 64 )
    {
      LOBYTE(v72) = 6;
    }
    else
    {
      if ( v68 != 128 )
      {
LABEL_133:
        v210 = v67;
        v225 = v68;
        v72 = sub_1F58CC0(v65, v68);
        v67 = v210;
        v68 = v225;
        v73 = v143;
        v199 = v72;
        v62 = *(_QWORD *)(a1 + 16);
        goto LABEL_63;
      }
      LOBYTE(v72) = 7;
    }
  }
  else if ( v68 == 8 )
  {
    LOBYTE(v72) = 3;
  }
  else
  {
    LOBYTE(v72) = 4;
    if ( v68 != 16 )
    {
      LOBYTE(v72) = 2;
      if ( v68 != 1 )
        goto LABEL_133;
    }
  }
  v73 = 0;
LABEL_63:
  v74 = v199;
  v202 = v67;
  v205 = v68;
  v75 = 16 * v13;
  LOBYTE(v74) = v72;
  v211 = v73;
  v209 = v74;
  v76 = (_BYTE *)sub_1E0A0C0(*(_QWORD *)(v62 + 32));
  v77 = v217;
  v200 = (__int64)v76;
  v197 = v202;
  v206 = v217;
  v218 = v202 >> 3;
  v203 = (v218 | v219) & -(v218 | v219);
  if ( *v76 )
  {
    v222 = sub_1D2B810(
             *(_QWORD **)(a1 + 16),
             v235,
             (__int64)&v262,
             **(unsigned __int8 **)(a2 + 40),
             *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
             v219,
             *(_OWORD *)&v11,
             v12.m128i_i64[0],
             v12.m128i_i64[1],
             *(_OWORD *)*(_QWORD *)(a2 + 104),
             *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
             v213,
             v216,
             v206,
             (__int64)&v266);
    v215 = v162;
    v198 = *(__int64 **)(a1 + 16);
    v245 = v162;
    *(_QWORD *)&v163 = sub_1D38BB0(
                         (__int64)v198,
                         v218,
                         (__int64)&v262,
                         *(unsigned __int8 *)(v75 + *(_QWORD *)(v229 + 40)),
                         *(const void ***)(v75 + *(_QWORD *)(v229 + 40) + 8),
                         0,
                         v11,
                         *(double *)v12.m128i_i64,
                         v40,
                         0);
    v164 = sub_1D332F0(
             v198,
             52,
             (__int64)&v262,
             *(unsigned __int8 *)(*(_QWORD *)(v229 + 40) + v75),
             *(const void ***)(*(_QWORD *)(v229 + 40) + v75 + 8),
             0,
             *(double *)v11.m128i_i64,
             *(double *)v12.m128i_i64,
             v40,
             v12.m128i_i64[0],
             v12.m128i_u64[1],
             v163);
    v166 = *(_QWORD **)(a1 + 16);
    v254 = (__int64)v164;
    v167 = *(_QWORD *)(a2 + 104);
    v168 = *(_QWORD *)v167 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v168 )
    {
      v185 = *(_QWORD *)(v167 + 8) + v218;
      v186 = *(_BYTE *)(v167 + 16);
      if ( (*(_QWORD *)v167 & 4) != 0 )
      {
        *((_QWORD *)&v270 + 1) = *(_QWORD *)(v167 + 8) + v218;
        LOBYTE(v271) = v186;
        *(_QWORD *)&v270 = v168 | 4;
        HIDWORD(v271) = *(_DWORD *)(v168 + 12);
      }
      else
      {
        *(_QWORD *)&v270 = *(_QWORD *)v167 & 0xFFFFFFFFFFFFFFF8LL;
        *((_QWORD *)&v270 + 1) = v185;
        LOBYTE(v271) = v186;
        v187 = *(_QWORD *)v168;
        if ( *(_BYTE *)(*(_QWORD *)v168 + 8LL) == 16 )
          v187 = **(_QWORD **)(v187 + 16);
        HIDWORD(v271) = *(_DWORD *)(v187 + 8) >> 8;
      }
    }
    else
    {
      v169 = *(_DWORD *)(v167 + 20);
      LODWORD(v271) = 0;
      v270 = 0u;
      HIDWORD(v271) = v169;
    }
    v226 = sub_1D2B810(
             v166,
             3u,
             (__int64)&v262,
             **(unsigned __int8 **)(a2 + 40),
             *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
             v203,
             *(_OWORD *)&v11,
             v254,
             v165 | v12.m128i_i64[1] & 0xFFFFFFFF00000000LL,
             v270,
             (__int64)v271,
             v209,
             v211,
             v206,
             (__int64)&v266);
    v228 = v170;
    *((_QWORD *)&v195 + 1) = 1;
    *(_QWORD *)&v195 = v222;
    v88 = sub_1D332F0(
            *(__int64 **)(a1 + 16),
            2,
            (__int64)&v262,
            1,
            0,
            0,
            *(double *)v11.m128i_i64,
            *(double *)v12.m128i_i64,
            v40,
            v226,
            1u,
            v195);
    v249 = v171;
    v233 = *(__int64 **)(a1 + 16);
    v172 = sub_1F40B60(
             *(_QWORD *)(a1 + 8),
             *(unsigned __int8 *)(16 * v215 + *(_QWORD *)(v222 + 40)),
             *(_QWORD *)(16 * v215 + *(_QWORD *)(v222 + 40) + 8),
             v200,
             1);
    *(_QWORD *)&v174 = sub_1D38BB0(
                         (__int64)v233,
                         v205,
                         (__int64)&v262,
                         v172,
                         v173,
                         0,
                         v11,
                         *(double *)v12.m128i_i64,
                         v40,
                         0);
    v241 = sub_1D332F0(
             v233,
             122,
             (__int64)&v262,
             *(unsigned __int8 *)(*(_QWORD *)(v222 + 40) + 16 * v215),
             *(const void ***)(*(_QWORD *)(v222 + 40) + 16 * v215 + 8),
             0,
             *(double *)v11.m128i_i64,
             *(double *)v12.m128i_i64,
             v40,
             v222,
             v245,
             v174);
    *((_QWORD *)&v196 + 1) = v175 | v245 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v196 = v241;
    result = sub_1D332F0(
               *(__int64 **)(a1 + 16),
               119,
               (__int64)&v262,
               **(unsigned __int8 **)(a2 + 40),
               *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
               0,
               *(double *)v11.m128i_i64,
               *(double *)v12.m128i_i64,
               v40,
               v226,
               v228,
               v196);
    v6 = v176;
    v95 = (__int64)result;
  }
  else
  {
    v214 = sub_1D2B810(
             *(_QWORD **)(a1 + 16),
             3u,
             (__int64)&v262,
             **(unsigned __int8 **)(a2 + 40),
             *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
             v219,
             *(_OWORD *)&v11,
             v12.m128i_i64[0],
             v12.m128i_i64[1],
             *(_OWORD *)*(_QWORD *)(a2 + 104),
             *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
             v213,
             v216,
             v77,
             (__int64)&v266);
    v220 = *(__int64 **)(a1 + 16);
    v227 = v78;
    *(_QWORD *)&v79 = sub_1D38BB0(
                        (__int64)v220,
                        v218,
                        (__int64)&v262,
                        *(unsigned __int8 *)(v75 + *(_QWORD *)(v229 + 40)),
                        *(const void ***)(v75 + *(_QWORD *)(v229 + 40) + 8),
                        0,
                        v11,
                        *(double *)v12.m128i_i64,
                        v40,
                        0);
    v80 = sub_1D332F0(
            v220,
            52,
            (__int64)&v262,
            *(unsigned __int8 *)(*(_QWORD *)(v229 + 40) + v75),
            *(const void ***)(*(_QWORD *)(v229 + 40) + v75 + 8),
            0,
            *(double *)v11.m128i_i64,
            *(double *)v12.m128i_i64,
            v40,
            v12.m128i_i64[0],
            v12.m128i_u64[1],
            v79);
    v82 = *(_QWORD **)(a1 + 16);
    v248 = (__int64)v80;
    v83 = *(_QWORD *)(a2 + 104);
    v84 = *(_QWORD *)v83 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v84 )
    {
      v181 = *(_QWORD *)(v83 + 8) + v218;
      v182 = *(_BYTE *)(v83 + 16);
      if ( (*(_QWORD *)v83 & 4) != 0 )
      {
        *((_QWORD *)&v268 + 1) = *(_QWORD *)(v83 + 8) + v218;
        LOBYTE(v269) = v182;
        *(_QWORD *)&v268 = v84 | 4;
        HIDWORD(v269) = *(_DWORD *)(v84 + 12);
      }
      else
      {
        v183 = *(_QWORD *)v84;
        *(_QWORD *)&v268 = v84;
        *((_QWORD *)&v268 + 1) = v181;
        v184 = *(_BYTE *)(v183 + 8) == 16;
        LOBYTE(v269) = v182;
        if ( v184 )
          v183 = **(_QWORD **)(v183 + 16);
        HIDWORD(v269) = *(_DWORD *)(v183 + 8) >> 8;
      }
    }
    else
    {
      v85 = *(_DWORD *)(v83 + 20);
      LODWORD(v269) = 0;
      v268 = 0u;
      HIDWORD(v269) = v85;
    }
    v221 = sub_1D2B810(
             v82,
             v235,
             (__int64)&v262,
             **(unsigned __int8 **)(a2 + 40),
             *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
             v203,
             *(_OWORD *)&v11,
             v248,
             v81 | v12.m128i_i64[1] & 0xFFFFFFFF00000000LL,
             v268,
             v269,
             v209,
             v211,
             v206,
             (__int64)&v266);
    v87 = 16LL * v86;
    v243 = v86;
    *((_QWORD *)&v192 + 1) = 1;
    *(_QWORD *)&v192 = v221;
    v88 = sub_1D332F0(
            *(__int64 **)(a1 + 16),
            2,
            (__int64)&v262,
            1,
            0,
            0,
            *(double *)v11.m128i_i64,
            *(double *)v12.m128i_i64,
            v40,
            v214,
            1u,
            v192);
    v249 = v89;
    v231 = *(__int64 **)(a1 + 16);
    v90 = sub_1F40B60(
            *(_QWORD *)(a1 + 8),
            *(unsigned __int8 *)(v87 + *(_QWORD *)(v221 + 40)),
            *(_QWORD *)(v87 + *(_QWORD *)(v221 + 40) + 8),
            v200,
            1);
    *(_QWORD *)&v92 = sub_1D38BB0(
                        (__int64)v231,
                        v197,
                        (__int64)&v262,
                        v90,
                        v91,
                        0,
                        v11,
                        *(double *)v12.m128i_i64,
                        v40,
                        0);
    v238 = sub_1D332F0(
             v231,
             122,
             (__int64)&v262,
             *(unsigned __int8 *)(*(_QWORD *)(v221 + 40) + v87),
             *(const void ***)(*(_QWORD *)(v221 + 40) + v87 + 8),
             0,
             *(double *)v11.m128i_i64,
             *(double *)v12.m128i_i64,
             v40,
             v221,
             v243,
             v92);
    *((_QWORD *)&v193 + 1) = v93 | v243 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v193 = v238;
    result = sub_1D332F0(
               *(__int64 **)(a1 + 16),
               119,
               (__int64)&v262,
               **(unsigned __int8 **)(a2 + 40),
               *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
               0,
               *(double *)v11.m128i_i64,
               *(double *)v12.m128i_i64,
               v40,
               v214,
               v227,
               v193);
    v6 = v94;
    v95 = (__int64)result;
  }
  v259 = v249;
LABEL_85:
  if ( (__int64 *)a2 == v88 )
    goto LABEL_18;
  sub_1D44C70(*(_QWORD *)(a1 + 16), a2, 0, v95, v6);
  sub_1D44C70(*(_QWORD *)(a1 + 16), a2, 1, (__int64)v88, v259);
  v115 = *(_QWORD *)(a1 + 32);
  if ( v115 )
  {
    v261 = v95;
    sub_1FF5010(v115, &v261, v111, v112, v113, v114);
    v116 = *(_QWORD *)(a1 + 32);
    v261 = (__int64)v88;
    sub_1FF5010(v116, &v261, v117, v118, v119, v120);
  }
  v121 = *(_QWORD *)(a1 + 24);
  v261 = a2;
  result = *(__int64 **)(v121 + 8);
  if ( *(__int64 **)(v121 + 16) == result )
  {
    v122 = (__int64)&result[*(unsigned int *)(v121 + 28)];
    if ( result == (__int64 *)v122 )
    {
LABEL_150:
      result = (__int64 *)v122;
    }
    else
    {
      while ( a2 != *result )
      {
        if ( (__int64 *)v122 == ++result )
          goto LABEL_150;
      }
    }
    goto LABEL_113;
  }
  result = sub_16CC9F0(v121, a2);
  if ( a2 == *result )
  {
    v133 = *(_QWORD *)(v121 + 16);
    if ( v133 == *(_QWORD *)(v121 + 8) )
      v112 = *(unsigned int *)(v121 + 28);
    else
      v112 = *(unsigned int *)(v121 + 24);
    v122 = v133 + 8 * v112;
LABEL_113:
    if ( (__int64 *)v122 != result )
    {
      *result = -2;
      ++*(_DWORD *)(v121 + 32);
    }
    goto LABEL_91;
  }
  result = *(__int64 **)(v121 + 16);
  if ( result == *(__int64 **)(v121 + 8) )
  {
    result += *(unsigned int *)(v121 + 28);
    v122 = (__int64)result;
    goto LABEL_113;
  }
LABEL_91:
  v123 = *(_QWORD *)(a1 + 32);
  if ( v123 )
    result = (__int64 *)sub_1FF5010(v123, &v261, v122, v112, v113, v114);
LABEL_18:
  if ( v262 )
    return (__int64 *)sub_161E7C0((__int64)&v262, v262);
  return result;
}
