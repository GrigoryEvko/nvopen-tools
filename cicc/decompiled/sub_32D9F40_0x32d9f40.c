// Function: sub_32D9F40
// Address: 0x32d9f40
//
__int64 __fastcall sub_32D9F40(__int64 *a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 result; // rax
  __int64 v8; // rsi
  __int64 v9; // r12
  __int64 v10; // r13
  unsigned __int16 *v11; // rdx
  int v12; // eax
  __int64 v13; // rcx
  unsigned __int16 *v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // rdi
  __m128i v21; // xmm3
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __m128i v25; // rdi
  int v26; // eax
  unsigned int *v27; // rdx
  __int64 v28; // r10
  const __m128i *v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // r11
  int v33; // eax
  unsigned int v34; // edx
  __int64 *v35; // rax
  __int64 *v36; // rsi
  unsigned __int64 v37; // r8
  __m128i v38; // xmm7
  __int64 v39; // rdx
  __int64 v40; // rcx
  unsigned int v41; // edi
  int v42; // esi
  int v43; // r9d
  unsigned int v44; // edx
  __int64 v45; // r11
  unsigned int v46; // edx
  int v47; // eax
  int v48; // eax
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rax
  __int16 v52; // dx
  __int64 v53; // rax
  __int64 v54; // rax
  int v55; // r9d
  __int64 v56; // rax
  __int64 v57; // rdi
  __int64 (__fastcall *v58)(__int64, __int64); // rcx
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rdx
  __int128 v66; // rax
  int v67; // r9d
  int v68; // eax
  __int64 v69; // rax
  const __m128i *v70; // rax
  __int64 v71; // rcx
  unsigned __int16 *v72; // rax
  __int64 v73; // rdi
  __m128i v74; // xmm5
  __int128 v75; // rax
  int v76; // r9d
  int v77; // r9d
  __int64 v78; // r11
  unsigned int v79; // r9d
  __m128i v80; // xmm4
  __int64 v81; // rdi
  __int128 v82; // rax
  __int64 v83; // r14
  int v84; // r9d
  __int64 v85; // rax
  __int64 v86; // rdx
  __int128 v87; // rax
  int v88; // r9d
  __int128 v89; // rax
  int v90; // r9d
  __int128 v91; // rax
  int v92; // r9d
  __int64 v93; // rax
  __int128 v94; // rax
  __int64 v95; // r9
  __int128 v96; // rax
  unsigned int v97; // esi
  unsigned __int64 v98; // rsi
  __int64 v99; // rax
  __int64 v100; // rdx
  __int64 v101; // rdx
  __int64 v102; // r14
  int v103; // r9d
  __int128 v104; // rax
  __int64 v105; // r8
  __int64 v106; // r9
  int v107; // esi
  int v108; // r9d
  __int64 v109; // r14
  __int64 v110; // r13
  __int128 v111; // xmm6
  __int64 v112; // rax
  __int64 v113; // rdi
  __int64 v114; // rsi
  __int128 v115; // rax
  int v116; // r9d
  __int128 v117; // rax
  int v118; // r9d
  __int128 v119; // rax
  int v120; // r9d
  __int128 v121; // rax
  __int128 v122; // rax
  __int64 v123; // r15
  int v124; // r9d
  unsigned int v125; // edx
  unsigned __int64 v126; // r15
  __int64 v127; // r14
  unsigned int v128; // edx
  unsigned __int64 v129; // r15
  __int128 v130; // rax
  int v131; // r9d
  __int128 v132; // rax
  __int128 v133; // rax
  __int64 v134; // r15
  int v135; // r9d
  unsigned int v136; // edx
  int v137; // r9d
  __int128 v138; // rax
  int v139; // r9d
  __int128 v140; // rax
  __int128 v141; // rax
  __int64 v142; // r15
  int v143; // r9d
  unsigned int v144; // edx
  int v145; // r9d
  __int64 v146; // rdx
  __int64 v147; // rdi
  __m128i v148; // rax
  __int64 v149; // rdi
  __m128i v150; // xmm5
  __int128 v151; // rax
  __int128 v152; // rax
  int v153; // r9d
  __int128 v154; // rax
  unsigned __int16 *v155; // rax
  __int64 v156; // r13
  __int64 v157; // rdx
  __int64 v158; // r14
  unsigned __int16 *v159; // rax
  int v160; // r9d
  __int64 v161; // r13
  unsigned int v162; // edx
  unsigned __int64 v163; // r14
  __int64 v164; // r8
  __int64 v165; // r9
  __int64 v166; // rbx
  int v167; // r9d
  __int128 v168; // rax
  int v169; // r9d
  __int128 v170; // rax
  int v171; // r9d
  __int128 v172; // [rsp-20h] [rbp-200h]
  __int128 v173; // [rsp-10h] [rbp-1F0h]
  __int128 v174; // [rsp-10h] [rbp-1F0h]
  __int128 v175; // [rsp-10h] [rbp-1F0h]
  __int128 v176; // [rsp-10h] [rbp-1F0h]
  __int128 v177; // [rsp-10h] [rbp-1F0h]
  __int64 v178; // [rsp-8h] [rbp-1E8h]
  __int64 v179; // [rsp+8h] [rbp-1D8h]
  unsigned int v180; // [rsp+14h] [rbp-1CCh]
  unsigned __int64 v181; // [rsp+18h] [rbp-1C8h]
  __int64 v182; // [rsp+20h] [rbp-1C0h]
  __int64 v183; // [rsp+30h] [rbp-1B0h]
  __int64 v184; // [rsp+30h] [rbp-1B0h]
  __int64 v185; // [rsp+30h] [rbp-1B0h]
  unsigned int v186; // [rsp+38h] [rbp-1A8h]
  char v187; // [rsp+38h] [rbp-1A8h]
  char v188; // [rsp+38h] [rbp-1A8h]
  __int64 v189; // [rsp+38h] [rbp-1A8h]
  int v190; // [rsp+38h] [rbp-1A8h]
  char v191; // [rsp+38h] [rbp-1A8h]
  unsigned __int64 v192; // [rsp+40h] [rbp-1A0h]
  __int64 v193; // [rsp+40h] [rbp-1A0h]
  __int64 v194; // [rsp+40h] [rbp-1A0h]
  __int128 v195; // [rsp+40h] [rbp-1A0h]
  __int64 v196; // [rsp+50h] [rbp-190h]
  __int64 v197; // [rsp+50h] [rbp-190h]
  __int64 v198; // [rsp+50h] [rbp-190h]
  __int128 v199; // [rsp+50h] [rbp-190h]
  char v200; // [rsp+50h] [rbp-190h]
  char v201; // [rsp+50h] [rbp-190h]
  char v202; // [rsp+50h] [rbp-190h]
  __int64 v203; // [rsp+50h] [rbp-190h]
  char v204; // [rsp+50h] [rbp-190h]
  unsigned __int32 v205; // [rsp+50h] [rbp-190h]
  char v206; // [rsp+50h] [rbp-190h]
  __int32 v207; // [rsp+60h] [rbp-180h]
  __int128 v208; // [rsp+60h] [rbp-180h]
  __int32 v209; // [rsp+60h] [rbp-180h]
  unsigned int v210; // [rsp+60h] [rbp-180h]
  __int128 v211; // [rsp+60h] [rbp-180h]
  char v212; // [rsp+60h] [rbp-180h]
  __int128 v213; // [rsp+60h] [rbp-180h]
  __int32 v214; // [rsp+70h] [rbp-170h]
  __int128 v215; // [rsp+70h] [rbp-170h]
  __int128 v216; // [rsp+70h] [rbp-170h]
  __int64 v217; // [rsp+70h] [rbp-170h]
  __int128 v218; // [rsp+70h] [rbp-170h]
  __int64 v219; // [rsp+70h] [rbp-170h]
  __int64 v220; // [rsp+80h] [rbp-160h]
  __int64 v221; // [rsp+80h] [rbp-160h]
  char v222; // [rsp+80h] [rbp-160h]
  int v223; // [rsp+80h] [rbp-160h]
  int v224; // [rsp+80h] [rbp-160h]
  __int64 v225; // [rsp+80h] [rbp-160h]
  __int64 v226; // [rsp+80h] [rbp-160h]
  __int64 v227; // [rsp+80h] [rbp-160h]
  __int64 v228; // [rsp+80h] [rbp-160h]
  __int64 v229; // [rsp+80h] [rbp-160h]
  unsigned __int64 v230; // [rsp+80h] [rbp-160h]
  __int64 v231; // [rsp+80h] [rbp-160h]
  __int64 v232; // [rsp+80h] [rbp-160h]
  __int128 v233; // [rsp+80h] [rbp-160h]
  __int64 v234; // [rsp+80h] [rbp-160h]
  __int64 v235; // [rsp+80h] [rbp-160h]
  __int64 v236; // [rsp+90h] [rbp-150h]
  __int64 v237; // [rsp+90h] [rbp-150h]
  __int64 v238; // [rsp+90h] [rbp-150h]
  __int128 v239; // [rsp+90h] [rbp-150h]
  __int128 *v240; // [rsp+90h] [rbp-150h]
  __int128 v241; // [rsp+90h] [rbp-150h]
  __int64 v242; // [rsp+90h] [rbp-150h]
  __int128 v243; // [rsp+90h] [rbp-150h]
  __int128 v244; // [rsp+90h] [rbp-150h]
  __int128 v245; // [rsp+90h] [rbp-150h]
  __int64 v246; // [rsp+E0h] [rbp-100h]
  __int64 v247; // [rsp+100h] [rbp-E0h]
  __m128i v248; // [rsp+110h] [rbp-D0h] BYREF
  __m128i v249; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v250; // [rsp+130h] [rbp-B0h] BYREF
  int v251; // [rsp+138h] [rbp-A8h]
  unsigned int v252; // [rsp+140h] [rbp-A0h] BYREF
  __int64 v253; // [rsp+148h] [rbp-98h]
  unsigned int v254; // [rsp+150h] [rbp-90h] BYREF
  __int64 v255; // [rsp+158h] [rbp-88h]
  __int64 v256; // [rsp+160h] [rbp-80h] BYREF
  int v257; // [rsp+168h] [rbp-78h]
  __int64 v258; // [rsp+170h] [rbp-70h] BYREF
  __int64 v259; // [rsp+178h] [rbp-68h]
  __int64 v260; // [rsp+180h] [rbp-60h]
  __int64 v261; // [rsp+188h] [rbp-58h]
  __m128i v262; // [rsp+190h] [rbp-50h] BYREF
  __m128i v263; // [rsp+1A0h] [rbp-40h]

  v4 = *(__int64 **)(a2 + 40);
  v5 = *a1;
  v6 = *v4;
  v249 = _mm_loadu_si128((const __m128i *)(v4 + 5));
  v248 = _mm_loadu_si128((const __m128i *)v4);
  result = sub_3401190(v5, v6, v248.m128i_i64[1], v249.m128i_i64[0], v249.m128i_i64[1]);
  if ( !result )
  {
    v8 = *(_QWORD *)(a2 + 80);
    v9 = 0;
    v250 = v8;
    if ( v8 )
      sub_B96E90((__int64)&v250, v8, 1);
    v10 = v248.m128i_i64[0];
    v11 = (unsigned __int16 *)(*(_QWORD *)(v248.m128i_i64[0] + 48) + 16LL * v248.m128i_u32[2]);
    v251 = *(_DWORD *)(a2 + 72);
    v12 = *v11;
    v13 = *((_QWORD *)v11 + 1);
    v236 = v249.m128i_i64[0];
    LOWORD(v252) = v12;
    v214 = v249.m128i_i32[2];
    v14 = (unsigned __int16 *)(*(_QWORD *)(v249.m128i_i64[0] + 48) + 16LL * v249.m128i_u32[2]);
    v253 = v13;
    v15 = *v14;
    v16 = *((_QWORD *)v14 + 1);
    LOWORD(v254) = v15;
    v255 = v16;
    if ( (_WORD)v12 )
    {
      if ( (unsigned __int16)(v12 - 17) > 0xD3u )
      {
        v262.m128i_i16[0] = v12;
        v262.m128i_i64[1] = v13;
        goto LABEL_14;
      }
      LOWORD(v12) = word_4456580[v12 - 1];
    }
    else
    {
      v220 = v13;
      if ( !sub_30070B0((__int64)&v252) )
      {
        v262.m128i_i64[1] = v220;
        v262.m128i_i16[0] = 0;
LABEL_8:
        v260 = sub_3007260((__int64)&v262);
        v261 = v19;
        LODWORD(v221) = v260;
        goto LABEL_9;
      }
      LOWORD(v12) = sub_3009970((__int64)&v252, v15, v17, v220, v18);
      v9 = v49;
    }
    v262.m128i_i16[0] = v12;
    v262.m128i_i64[1] = v9;
    if ( !(_WORD)v12 )
      goto LABEL_8;
LABEL_14:
    if ( (_WORD)v12 == 1 || (unsigned __int16)(v12 - 504) <= 7u )
      BUG();
    v221 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v12 - 16];
LABEL_9:
    v20 = *a1;
    v21 = _mm_loadu_si128(&v249);
    v262 = _mm_loadu_si128(&v248);
    v263 = v21;
    result = sub_3402EA0(v20, 190, (unsigned int)&v250, v252, v253, 0, (__int64)&v262, 2);
    v23 = v178;
    if ( result )
      goto LABEL_10;
    if ( (_WORD)v252 )
    {
      if ( (unsigned __int16)(v252 - 17) > 0xD3u )
        goto LABEL_23;
    }
    else if ( !sub_30070B0((__int64)&v252) )
    {
      goto LABEL_23;
    }
    result = sub_3295970(a1, a2, (__int64)&v250, v22, v23);
    if ( result )
      goto LABEL_10;
    if ( *(_DWORD *)(v236 + 24) == 156 )
    {
      if ( (unsigned __int8)sub_33E22F0(v236) )
      {
        if ( *(_DWORD *)(v10 + 24) == 186 )
        {
          v70 = *(const __m128i **)(v10 + 40);
          v71 = v70[2].m128i_i64[1];
          v199 = (__int128)_mm_loadu_si128(v70);
          if ( *(_DWORD *)(v71 + 24) == 156 )
          {
            v194 = v70[2].m128i_i64[1];
            v189 = v70->m128i_i64[0];
            v209 = v70[3].m128i_i32[0];
            if ( (unsigned __int8)sub_33E22F0(v71) )
            {
              if ( *(_DWORD *)(v189 + 24) == 208 )
              {
                v72 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(v189 + 40) + 48LL)
                                         + 16LL * *(unsigned int *)(*(_QWORD *)(v189 + 40) + 8LL));
                if ( (unsigned int)sub_3289F80((unsigned int *)a1[1], *v72, *((_QWORD *)v72 + 1)) == 2 )
                {
                  v73 = *a1;
                  v262.m128i_i64[0] = v194;
                  v74 = _mm_loadu_si128(&v249);
                  v262.m128i_i32[2] = v209;
                  v263 = v74;
                  *(_QWORD *)&v75 = sub_3402EA0(v73, 190, (unsigned int)&v250, v252, v253, 0, (__int64)&v262, 2);
                  if ( (_QWORD)v75 )
                  {
                    result = sub_3406EB0(*a1, 186, (unsigned int)&v250, v252, v253, v76, v199, v75);
                    goto LABEL_10;
                  }
                }
              }
            }
          }
        }
      }
    }
LABEL_23:
    result = sub_329BF20(a1, a2);
    if ( result )
      goto LABEL_10;
    v24 = *a1;
    v207 = v221;
    v262.m128i_i32[2] = v221;
    if ( (unsigned int)v221 > 0x40 )
    {
      v225 = v24;
      sub_C43690((__int64)&v262, -1, 1);
      v24 = v225;
    }
    else
    {
      if ( (_DWORD)v221 )
        result = 0xFFFFFFFFFFFFFFFFLL >> -(char)v221;
      v262.m128i_i64[0] = result;
    }
    v25.m128i_i64[1] = a2;
    v222 = sub_33DD210(v24, a2, 0, &v262, 0);
    if ( v262.m128i_i32[2] > 0x40u && v262.m128i_i64[0] )
      j_j___libc_free_0_0(v262.m128i_u64[0]);
    if ( v222 )
      goto LABEL_66;
    if ( *(_DWORD *)(v236 + 24) == 216 && *(_DWORD *)(**(_QWORD **)(v236 + 40) + 24LL) == 186 )
    {
      v25.m128i_i64[1] = v236;
      *(_QWORD *)&v66 = sub_32CB9C0((__int64)a1, (_QWORD *)v236);
      if ( (_QWORD)v66 )
      {
        result = sub_3406EB0(*a1, 190, (unsigned int)&v250, v252, v253, v67, *(_OWORD *)&v248, v66);
        goto LABEL_10;
      }
    }
    v26 = *(_DWORD *)(v10 + 24);
    if ( v26 == 190 )
    {
      v262.m128i_i32[0] = v207;
      v263.m128i_i64[1] = (__int64)sub_32629F0;
      v263.m128i_i64[0] = (__int64)sub_325DB90;
      v200 = sub_33CACD0(
               v249.m128i_i32[0],
               v249.m128i_i32[2],
               *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL),
               *(_QWORD *)(*(_QWORD *)(v10 + 40) + 48LL),
               (unsigned int)&v262,
               0,
               0);
      sub_A17130((__int64)&v262);
      if ( v200 )
        goto LABEL_66;
      v25 = v249;
      v262.m128i_i32[0] = v207;
      v263.m128i_i64[1] = (__int64)sub_32626E0;
      v263.m128i_i64[0] = (__int64)sub_325DBC0;
      v201 = sub_33CACD0(
               v25.m128i_i32[0],
               v25.m128i_i32[2],
               *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL),
               *(_QWORD *)(*(_QWORD *)(v10 + 40) + 48LL),
               (unsigned int)&v262,
               0,
               0);
      sub_A17130((__int64)&v262);
      if ( v201 )
      {
        *(_QWORD *)&v91 = sub_3406EB0(
                            *a1,
                            56,
                            (unsigned int)&v250,
                            v254,
                            v255,
                            v77,
                            *(_OWORD *)&v249,
                            *(_OWORD *)(*(_QWORD *)(v10 + 40) + 40LL));
        goto LABEL_119;
      }
      v26 = *(_DWORD *)(v10 + 24);
    }
    if ( (unsigned int)(v26 - 213) > 2 )
      goto LABEL_71;
    v27 = *(unsigned int **)(v10 + 40);
    v28 = *(_QWORD *)v27;
    if ( *(_DWORD *)(*(_QWORD *)v27 + 24LL) != 190 )
    {
      if ( v26 != 214 )
        goto LABEL_79;
      goto LABEL_37;
    }
    v50 = *(_QWORD *)(v28 + 40);
    v182 = *(_QWORD *)v27;
    v193 = *(_QWORD *)(v50 + 48);
    v197 = *(_QWORD *)(v50 + 40);
    v51 = *(_QWORD *)(v28 + 48) + 16LL * v27[2];
    v52 = *(_WORD *)v51;
    v53 = *(_QWORD *)(v51 + 8);
    LOWORD(v258) = v52;
    v259 = v53;
    v54 = sub_32844A0((unsigned __int16 *)&v258, v25.m128i_i64[1]);
    v262.m128i_i32[0] = v207;
    v262.m128i_i64[1] = v54;
    v184 = v54;
    v263.m128i_i64[1] = (__int64)sub_3265CC0;
    v263.m128i_i64[0] = (__int64)sub_325DBF0;
    v187 = sub_33CACD0(v197, v193, v249.m128i_i32[0], v249.m128i_i32[2], (unsigned int)&v262, 0, 1);
    sub_A17130((__int64)&v262);
    if ( !v187 )
    {
      v262.m128i_i64[1] = v184;
      v262.m128i_i32[0] = v207;
      v263.m128i_i64[1] = (__int64)sub_3265EC0;
      v263.m128i_i64[0] = (__int64)sub_325DC20;
      v188 = sub_33CACD0(v197, v193, v249.m128i_i32[0], v249.m128i_i32[2], (unsigned int)&v262, 0, 1);
      sub_A17130((__int64)&v262);
      if ( v188 )
      {
        *(_QWORD *)&v132 = sub_33FAF80(
                             *a1,
                             *(_DWORD *)(v10 + 24),
                             (unsigned int)&v250,
                             v252,
                             v253,
                             v55,
                             *(_OWORD *)*(_QWORD *)(v182 + 40));
        v244 = v132;
        *(_QWORD *)&v133 = sub_33FB310(*a1, v197, v193, &v250, v254, v255);
        v134 = *((_QWORD *)&v133 + 1);
        v247 = sub_3406EB0(*a1, 56, (unsigned int)&v250, v254, v255, v135, v133, *(_OWORD *)&v249);
        *((_QWORD *)&v175 + 1) = v136 | v134 & 0xFFFFFFFF00000000LL;
        *(_QWORD *)&v175 = v247;
        result = sub_3406EB0(*a1, 190, (unsigned int)&v250, v252, v253, v137, v244, v175);
        goto LABEL_10;
      }
      v26 = *(_DWORD *)(v10 + 24);
      if ( v26 == 214 )
      {
LABEL_37:
        if ( !(unsigned __int8)sub_3286E00(&v248) )
          goto LABEL_39;
        v29 = *(const __m128i **)(v10 + 40);
        if ( *(_DWORD *)(v29->m128i_i64[0] + 24) != 192 )
          goto LABEL_39;
        v111 = (__int128)_mm_loadu_si128(v29);
        v185 = v29->m128i_i64[0];
        v205 = v29->m128i_u32[2];
        v112 = *(_QWORD *)(v29->m128i_i64[0] + 40);
        v113 = *(_QWORD *)(v112 + 40);
        v114 = *(_QWORD *)(v112 + 48);
        v262.m128i_i16[0] = v252;
        v262.m128i_i64[1] = v253;
        v263.m128i_i64[1] = (__int64)sub_3268530;
        v263.m128i_i64[0] = (__int64)sub_325DC50;
        v191 = sub_33CACD0(v113, v114, v249.m128i_i32[0], v249.m128i_i32[2], (unsigned int)&v262, 0, 1);
        sub_A17130((__int64)&v262);
        if ( v191 )
        {
          v155 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v185 + 40) + 40LL) + 48LL)
                                    + 16LL * *(unsigned int *)(*(_QWORD *)(v185 + 40) + 48LL));
          v156 = sub_33FB310(*a1, v249.m128i_i64[0], v249.m128i_i64[1], &v250, *v155, *((_QWORD *)v155 + 1));
          v158 = v157;
          v159 = (unsigned __int16 *)(*(_QWORD *)(v185 + 48) + 16LL * v205);
          *((_QWORD *)&v176 + 1) = v157;
          *(_QWORD *)&v176 = v156;
          v161 = sub_3406EB0(*a1, 190, (unsigned int)&v250, *v159, *((_QWORD *)v159 + 1), v160, v111, v176);
          v163 = v162 | v158 & 0xFFFFFFFF00000000LL;
          sub_32B3E80((__int64)a1, v161, 1, 0, v164, v165);
          v166 = *a1;
          sub_3285E70((__int64)&v262, v248.m128i_i64[0]);
          *((_QWORD *)&v177 + 1) = v163;
          *(_QWORD *)&v177 = v161;
          v235 = sub_33FAF80(v166, 214, (unsigned int)&v262, v252, v253, v167, v177);
          sub_9C6650(&v262);
          result = v235;
          goto LABEL_10;
        }
        v26 = *(_DWORD *)(v10 + 24);
      }
LABEL_71:
      if ( (unsigned int)(v26 - 191) > 1 )
        goto LABEL_79;
      if ( (*(_BYTE *)(v10 + 28) & 4) == 0 )
      {
LABEL_73:
        if ( v26 != 192 )
          goto LABEL_74;
        v93 = *(_QWORD *)(v10 + 40);
        if ( (v236 != *(_QWORD *)(v93 + 40) || v214 != *(_DWORD *)(v93 + 48)) && !(unsigned __int8)sub_3286E00(&v248) )
          goto LABEL_39;
        if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1[1] + 416LL))(
                a1[1],
                a2,
                *((unsigned int *)a1 + 6)) )
        {
LABEL_123:
          v26 = *(_DWORD *)(v10 + 24);
LABEL_74:
          if ( v26 == 191 )
          {
            v56 = *(_QWORD *)(v10 + 40);
            if ( v236 != *(_QWORD *)(v56 + 40) || v214 != *(_DWORD *)(v56 + 48) )
              goto LABEL_39;
            if ( (unsigned __int8)sub_326A930(v249.m128i_i64[0], v249.m128i_u32[2], 1u) )
            {
              *(_QWORD *)&v168 = sub_34015B0(*a1, &v250, v252, v253, 0, 0);
              *(_QWORD *)&v170 = sub_3406EB0(*a1, 190, (unsigned int)&v250, v252, v253, v169, v168, *(_OWORD *)&v249);
              result = sub_3406EB0(
                         *a1,
                         186,
                         (unsigned int)&v250,
                         v252,
                         v253,
                         v171,
                         *(_OWORD *)*(_QWORD *)(v10 + 40),
                         v170);
              goto LABEL_10;
            }
            v26 = *(_DWORD *)(v10 + 24);
          }
LABEL_79:
          if ( v26 != 187 && v26 != 56 )
            goto LABEL_83;
          v57 = a1[1];
          v58 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v57 + 2152LL);
          if ( v58 == sub_302E0E0 )
          {
            v59 = **(_QWORD **)(a2 + 40);
            v60 = *(_QWORD *)(v59 + 56);
            if ( !v60 )
              goto LABEL_83;
            if ( *(_QWORD *)(v60 + 32) )
              goto LABEL_83;
            if ( *(_DWORD *)(v59 + 24) == 213 )
            {
              v146 = *(_QWORD *)(**(_QWORD **)(v59 + 40) + 56LL);
              if ( !v146 || *(_QWORD *)(v146 + 32) )
                goto LABEL_83;
            }
          }
          else if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))v58)(v57, a2, *((unsigned int *)a1 + 6)) )
          {
LABEL_147:
            v26 = *(_DWORD *)(v10 + 24);
LABEL_83:
            if ( v26 == 213 )
            {
              v61 = **(_QWORD **)(v10 + 40);
              if ( *(_DWORD *)(v61 + 24) != 56 || (*(_BYTE *)(v61 + 28) & 2) == 0 )
                goto LABEL_39;
              if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1[1] + 2152LL))(
                     a1[1],
                     a2,
                     *((unsigned int *)a1 + 6)) )
              {
                v219 = **(_QWORD **)(v10 + 40);
                sub_3285E70((__int64)&v258, v248.m128i_i64[0]);
                v147 = *a1;
                v262 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v219 + 40) + 40LL));
                v148.m128i_i64[0] = sub_3402EA0(
                                      v147,
                                      *(_DWORD *)(v10 + 24),
                                      (unsigned int)&v258,
                                      v252,
                                      v253,
                                      0,
                                      (__int64)&v262,
                                      1);
                if ( v148.m128i_i64[0] )
                {
                  v149 = *a1;
                  v150 = _mm_loadu_si128(&v249);
                  v262 = v148;
                  v263 = v150;
                  *(_QWORD *)&v151 = sub_3402EA0(v149, 190, (unsigned int)&v258, v252, v253, 0, (__int64)&v262, 2);
                  v213 = v151;
                  if ( (_QWORD)v151 )
                  {
                    *(_QWORD *)&v152 = sub_33FAF80(
                                         *a1,
                                         *(_DWORD *)(v10 + 24),
                                         (unsigned int)&v258,
                                         v252,
                                         v253,
                                         v151,
                                         *(_OWORD *)*(_QWORD *)(v219 + 40));
                    *(_QWORD *)&v154 = sub_3406EB0(
                                         *a1,
                                         190,
                                         (unsigned int)&v258,
                                         v252,
                                         v253,
                                         v153,
                                         v152,
                                         *(_OWORD *)&v249);
                    v234 = sub_3406EB0(*a1, 56, (unsigned int)&v258, v252, v253, v213, v154, v213);
                    sub_9C6650(&v258);
                    result = v234;
                    goto LABEL_10;
                  }
                }
                sub_9C6650(&v258);
              }
              v26 = *(_DWORD *)(v10 + 24);
            }
            if ( v26 == 58 )
            {
              v62 = *(_QWORD *)(v10 + 56);
              if ( v62 )
              {
                if ( !*(_QWORD *)(v62 + 32) )
                {
                  v63 = *(_QWORD *)(v10 + 40);
                  v64 = *(_QWORD *)(v63 + 40);
                  LODWORD(v63) = *(_DWORD *)(v63 + 48);
                  v198 = *a1;
                  v263 = _mm_loadu_si128(&v249);
                  v262.m128i_i64[0] = v64;
                  v262.m128i_i32[2] = v63;
                  sub_3285E70((__int64)&v258, v249.m128i_i64[0]);
                  *(_QWORD *)&v216 = sub_3402EA0(v198, 190, (unsigned int)&v258, v252, v253, 0, (__int64)&v262, 2);
                  *((_QWORD *)&v216 + 1) = v65;
                  sub_9C6650(&v258);
                  if ( (_QWORD)v216 )
                  {
                    result = sub_3406EB0(
                               *a1,
                               58,
                               (unsigned int)&v250,
                               v252,
                               v253,
                               DWORD2(v216),
                               *(_OWORD *)*(_QWORD *)(v10 + 40),
                               v216);
                    goto LABEL_10;
                  }
                }
              }
            }
LABEL_39:
            v30 = v249.m128i_i64[1];
            v31 = sub_33DFBC0(v249.m128i_i64[0], v249.m128i_i64[1], 0, 0);
            v32 = *a1;
            v196 = v31;
            if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)*a1 + 544LL) - 42) > 1 )
            {
              if ( !v31 )
                goto LABEL_56;
              goto LABEL_55;
            }
            if ( !v31 )
              goto LABEL_56;
            v33 = *(_DWORD *)(v10 + 24);
            if ( v33 != 186 )
            {
              v34 = v33 - 213;
              if ( (unsigned int)(v33 - 187) > 1 )
              {
                if ( v34 > 2 || (v35 = *(__int64 **)(v10 + 40), (unsigned int)(*(_DWORD *)(*v35 + 24) - 186) > 2) )
                {
LABEL_55:
                  if ( (*(_BYTE *)(v196 + 32) & 8) == 0 )
                  {
                    v30 = a2;
                    result = sub_327E0B0(a1, a2);
                    if ( result )
                      goto LABEL_10;
                  }
LABEL_56:
                  v48 = *(_DWORD *)(v236 + 24);
                  if ( v48 == 198 )
                  {
                    v230 = sub_32844A0((unsigned __int16 *)&v252, v30);
                    if ( v230 <= sub_32844A0((unsigned __int16 *)&v254, v30) )
                      goto LABEL_111;
                    v48 = *(_DWORD *)(v236 + 24);
                  }
                  if ( v48 != 203 )
                  {
LABEL_58:
                    if ( (unsigned __int8)sub_32D0FE0((__int64)a1, a2, 0) )
                    {
                      result = a2;
                      goto LABEL_10;
                    }
                    v68 = *(_DWORD *)(v10 + 24);
                    if ( v68 != 373 )
                    {
                      LODWORD(v259) = 1;
                      v258 = 0;
                      if ( v68 == 170 )
                      {
                        if ( (unsigned __int8)sub_33D1410(v236, &v258) )
                        {
                          v110 = *(_QWORD *)(**(_QWORD **)(v10 + 40) + 96LL);
                          if ( sub_986EE0((__int64)&v258, *(unsigned int *)(v110 + 32)) )
                          {
                            sub_9865C0((__int64)&v262, v110 + 24);
                            sub_C47AC0((__int64)&v262, (__int64)&v258);
                            v232 = sub_3402600(*a1, &v250, v252, v253, &v262);
                            sub_969240(v262.m128i_i64);
                            v69 = v232;
                            goto LABEL_101;
                          }
                        }
                      }
LABEL_100:
                      v69 = 0;
LABEL_101:
                      v238 = v69;
                      sub_969240(&v258);
                      result = v238;
                      goto LABEL_10;
                    }
                    if ( !v196 )
                    {
                      LODWORD(v259) = 1;
                      v258 = 0;
                      goto LABEL_100;
                    }
                    v109 = *a1;
                    v242 = *(_QWORD *)(v196 + 96) + 24LL;
                    sub_9865C0((__int64)&v262, *(_QWORD *)(**(_QWORD **)(v10 + 40) + 96LL) + 24LL);
                    sub_C47AC0((__int64)&v262, v242);
                    v231 = sub_3401900(v109, &v250, v252, v253, &v262, 1);
                    sub_969240(v262.m128i_i64);
                    result = v231;
LABEL_10:
                    if ( v250 )
                    {
                      v237 = result;
                      sub_B91220((__int64)&v250, v250);
                      return v237;
                    }
                    return result;
                  }
LABEL_111:
                  if ( (unsigned __int8)sub_3286E00(&v249) )
                  {
                    v210 = v254;
                    v217 = v255;
                    v226 = a1[1];
                    if ( !(unsigned __int8)sub_328A020(v226, 0xC6u, v254, v255, 0) )
                    {
                      if ( (unsigned __int8)sub_328A020(v226, 0x3Au, v252, v253, 0) )
                      {
                        v78 = v217;
                        v79 = v210;
                        v80 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v236 + 40));
                        v262.m128i_i64[0] = *(_QWORD *)(a2 + 80);
                        v239 = (__int128)v80;
                        if ( v262.m128i_i64[0] )
                        {
                          sub_325F5D0(v262.m128i_i64);
                          v79 = v254;
                          v78 = v255;
                        }
                        v81 = *a1;
                        v262.m128i_i32[2] = *(_DWORD *)(a2 + 72);
                        *(_QWORD *)&v82 = sub_3407430(v81, v239, *((_QWORD *)&v239 + 1), &v262, v79, v78);
                        v83 = *a1;
                        v85 = sub_3406EB0(*a1, 186, (unsigned int)&v262, v254, v255, v84, v239, v82);
                        *(_QWORD *)&v87 = sub_33FB310(v83, v85, v86, &v262, v252, v253);
                        v227 = sub_3406EB0(*a1, 58, (unsigned int)&v262, v252, v253, v88, v87, *(_OWORD *)&v248);
                        sub_9C6650(&v262);
                        result = v227;
                        goto LABEL_10;
                      }
                    }
                  }
                  goto LABEL_58;
                }
                goto LABEL_44;
              }
              v35 = *(__int64 **)(v10 + 40);
              if ( v34 <= 2 )
              {
LABEL_44:
                v222 = 1;
                v183 = *v35;
LABEL_45:
                v36 = *(__int64 **)(v183 + 40);
                v37 = *v36;
                v38 = _mm_loadu_si128((const __m128i *)(v36 + 5));
                v39 = v36[1];
                v40 = *((unsigned int *)v36 + 2);
                v192 = v36[5];
                v41 = *((_DWORD *)v36 + 12);
                v42 = *(_DWORD *)(*v36 + 24);
                v186 = v41;
                if ( v42 == 11 || v42 == 35 )
                {
                  v97 = v40;
                  v40 = v41;
                  v186 = v97;
                  v98 = v37;
                  v37 = v192;
                  v192 = v98;
                }
                v43 = 0;
                if ( v222 )
                {
                  v262.m128i_i64[0] = *(_QWORD *)(a2 + 80);
                  if ( v262.m128i_i64[0] )
                  {
                    v179 = v39;
                    v180 = v40;
                    v181 = v37;
                    v223 = v32;
                    sub_325F5D0(v262.m128i_i64);
                    v39 = v179;
                    v40 = v180;
                    v37 = v181;
                    LODWORD(v32) = v223;
                  }
                  v262.m128i_i32[2] = *(_DWORD *)(a2 + 72);
                  *((_QWORD *)&v173 + 1) = v40 | v39 & 0xFFFFFFFF00000000LL;
                  *(_QWORD *)&v173 = v37;
                  *(_QWORD *)&v215 = sub_33FAF80(v32, *(_DWORD *)(v10 + 24), (unsigned int)&v262, v252, v253, 0, v173);
                  *((_QWORD *)&v215 + 1) = v44;
                  sub_9C6650(&v262);
                  v45 = *a1;
                  v262.m128i_i64[0] = *(_QWORD *)(a2 + 80);
                  if ( v262.m128i_i64[0] )
                  {
                    v224 = v45;
                    sub_325F5D0(v262.m128i_i64);
                    LODWORD(v45) = v224;
                  }
                  v262.m128i_i32[2] = *(_DWORD *)(a2 + 72);
                  v30 = *(unsigned int *)(v10 + 24);
                  *(_QWORD *)&v208 = sub_33FAF80(
                                       v45,
                                       v30,
                                       (unsigned int)&v262,
                                       v252,
                                       v253,
                                       0,
                                       __PAIR128__(v186 | v38.m128i_i64[1] & 0xFFFFFFFF00000000LL, v192));
                  *((_QWORD *)&v208 + 1) = v46;
                  sub_9C6650(&v262);
                }
                else
                {
                  *(_QWORD *)&v215 = v37;
                  *((_QWORD *)&v215 + 1) = v40;
                  *(_QWORD *)&v208 = v192;
                  v30 = v186;
                  *((_QWORD *)&v208 + 1) = v186;
                }
                v47 = *(_DWORD *)(v192 + 24);
                if ( v47 == 11 || v47 == 35 )
                {
                  v228 = *a1;
                  v262.m128i_i64[0] = *(_QWORD *)(a2 + 80);
                  if ( v262.m128i_i64[0] )
                    sub_325F5D0(v262.m128i_i64);
                  v262.m128i_i32[2] = *(_DWORD *)(a2 + 72);
                  *(_QWORD *)&v94 = sub_3406EB0(v228, 190, (unsigned int)&v262, v252, v253, v43, v208, *(_OWORD *)&v249);
                  v95 = *a1;
                  v211 = v94;
                  v258 = *(_QWORD *)(a2 + 80);
                  if ( v258 )
                  {
                    v190 = v95;
                    sub_325F5D0(&v258);
                    LODWORD(v95) = v190;
                  }
                  LODWORD(v259) = *(_DWORD *)(a2 + 72);
                  *(_QWORD *)&v96 = sub_3406EB0(v95, 190, (unsigned int)&v258, v252, v253, v95, v215, *(_OWORD *)&v249);
                  v256 = *(_QWORD *)(a2 + 80);
                  if ( v256 )
                  {
                    v195 = v96;
                    sub_325F5D0(&v256);
                    v96 = v195;
                  }
                  v257 = *(_DWORD *)(a2 + 72);
                  v229 = sub_3406EB0(
                           v228,
                           *(_DWORD *)(v183 + 24),
                           (unsigned int)&v256,
                           v252,
                           v253,
                           (unsigned int)&v256,
                           v96,
                           v211);
                  sub_9C6650(&v256);
                  sub_9C6650(&v258);
                  sub_9C6650(&v262);
                  v30 = v229;
                  if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1[1] + 2184LL))(a1[1], v229) )
                  {
                    result = v229;
                    goto LABEL_10;
                  }
                }
                goto LABEL_55;
              }
            }
            v183 = v10;
            goto LABEL_45;
          }
          v99 = *(_QWORD *)(v10 + 40);
          v100 = *(_QWORD *)(v99 + 40);
          LODWORD(v99) = *(_DWORD *)(v99 + 48);
          v203 = *a1;
          v263 = _mm_loadu_si128(&v249);
          v262.m128i_i64[0] = v100;
          v262.m128i_i32[2] = v99;
          sub_3285E70((__int64)&v258, v249.m128i_i64[0]);
          *(_QWORD *)&v218 = sub_3402EA0(v203, 190, (unsigned int)&v258, v252, v253, 0, (__int64)&v262, 2);
          *((_QWORD *)&v218 + 1) = v101;
          sub_9C6650(&v258);
          if ( (_QWORD)v218 )
          {
            v102 = *a1;
            v240 = *(__int128 **)(v10 + 40);
            sub_3285E70((__int64)&v262, v248.m128i_i64[0]);
            *(_QWORD *)&v104 = sub_3406EB0(v102, 190, (unsigned int)&v262, v252, v253, v103, *v240, *(_OWORD *)&v249);
            v241 = v104;
            sub_9C6650(&v262);
            sub_32B3E80((__int64)a1, v241, 1, 0, v105, v106);
            v107 = *(_DWORD *)(v10 + 24);
            v108 = 0;
            if ( v107 == 187 )
            {
              v108 = *(_DWORD *)(v10 + 28) & 8;
              if ( v108 )
                v108 = 8;
            }
            result = sub_3405C90(*a1, v107, (unsigned int)&v250, v252, v253, v108, v241, v218);
            goto LABEL_10;
          }
          goto LABEL_147;
        }
        v263.m128i_i64[1] = (__int64)sub_32645C0;
        v263.m128i_i64[0] = (__int64)sub_325DC80;
        v262.m128i_i32[0] = v207;
        v206 = sub_33CACD0(
                 v249.m128i_i32[0],
                 v249.m128i_i32[2],
                 *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL),
                 *(_QWORD *)(*(_QWORD *)(v10 + 40) + 48LL),
                 (unsigned int)&v262,
                 0,
                 1);
        sub_A17130((__int64)&v262);
        if ( v206 )
        {
          *(_QWORD *)&v119 = sub_33FB310(
                               *a1,
                               *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL),
                               *(_QWORD *)(*(_QWORD *)(v10 + 40) + 48LL),
                               &v250,
                               v254,
                               v255);
          v233 = v119;
          *(_QWORD *)&v121 = sub_3406EB0(*a1, 57, (unsigned int)&v250, v254, v255, v120, v119, *(_OWORD *)&v249);
          v243 = v121;
          *(_QWORD *)&v122 = sub_34015B0(*a1, &v250, v252, v253, 0, 0);
          v123 = *((_QWORD *)&v122 + 1);
          v246 = sub_3406EB0(*a1, 190, (unsigned int)&v250, v252, v253, v124, v122, v233);
          v126 = v125 | v123 & 0xFFFFFFFF00000000LL;
          *((_QWORD *)&v172 + 1) = v126;
          *(_QWORD *)&v172 = v246;
          v127 = sub_3406EB0(*a1, 192, (unsigned int)&v250, v252, v253, 0, v172, v243);
          v129 = v128 | v126 & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v130 = sub_3406EB0(
                               *a1,
                               192,
                               (unsigned int)&v250,
                               v252,
                               v253,
                               0,
                               *(_OWORD *)*(_QWORD *)(v10 + 40),
                               v243);
        }
        else
        {
          v263.m128i_i64[1] = (__int64)sub_32645C0;
          v263.m128i_i64[0] = (__int64)sub_325DC80;
          v262.m128i_i32[0] = v207;
          v212 = sub_33CACD0(
                   *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL),
                   *(_QWORD *)(*(_QWORD *)(v10 + 40) + 48LL),
                   v249.m128i_i32[0],
                   v249.m128i_i32[2],
                   (unsigned int)&v262,
                   0,
                   1);
          sub_A17130((__int64)&v262);
          if ( !v212 )
            goto LABEL_123;
          *(_QWORD *)&v138 = sub_33FB310(
                               *a1,
                               *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL),
                               *(_QWORD *)(*(_QWORD *)(v10 + 40) + 48LL),
                               &v250,
                               v254,
                               v255);
          *(_QWORD *)&v140 = sub_3406EB0(*a1, 57, (unsigned int)&v250, v254, v255, v139, *(_OWORD *)&v249, v138);
          v245 = v140;
          *(_QWORD *)&v141 = sub_34015B0(*a1, &v250, v252, v253, 0, 0);
          v142 = *((_QWORD *)&v141 + 1);
          v127 = sub_3406EB0(*a1, 190, (unsigned int)&v250, v252, v253, v143, v141, *(_OWORD *)&v249);
          v129 = v144 | v142 & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v130 = sub_3406EB0(
                               *a1,
                               190,
                               (unsigned int)&v250,
                               v252,
                               v253,
                               v145,
                               *(_OWORD *)*(_QWORD *)(v10 + 40),
                               v245);
        }
        *((_QWORD *)&v174 + 1) = v129;
        *(_QWORD *)&v174 = v127;
        result = sub_3406EB0(*a1, 186, (unsigned int)&v250, v252, v253, v131, v130, v174);
        goto LABEL_10;
      }
      v263.m128i_i64[0] = (__int64)sub_325DC80;
      v263.m128i_i64[1] = (__int64)sub_32645C0;
      v262.m128i_i32[0] = v207;
      v202 = sub_33CACD0(
               *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL),
               *(_QWORD *)(*(_QWORD *)(v10 + 40) + 48LL),
               v249.m128i_i32[0],
               v249.m128i_i32[2],
               (unsigned int)&v262,
               0,
               1);
      sub_A17130((__int64)&v262);
      if ( !v202 )
      {
        v263.m128i_i64[1] = (__int64)sub_32645C0;
        v263.m128i_i64[0] = (__int64)sub_325DC80;
        v262.m128i_i32[0] = v207;
        v204 = sub_33CACD0(
                 v249.m128i_i32[0],
                 v249.m128i_i32[2],
                 *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL),
                 *(_QWORD *)(*(_QWORD *)(v10 + 40) + 48LL),
                 (unsigned int)&v262,
                 0,
                 1);
        sub_A17130((__int64)&v262);
        if ( v204 )
        {
          *(_QWORD *)&v115 = sub_33FB310(
                               *a1,
                               *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL),
                               *(_QWORD *)(*(_QWORD *)(v10 + 40) + 48LL),
                               &v250,
                               v254,
                               v255);
          *(_QWORD *)&v117 = sub_3406EB0(*a1, 57, (unsigned int)&v250, v254, v255, v116, v115, *(_OWORD *)&v249);
          result = sub_3406EB0(
                     *a1,
                     *(_DWORD *)(v10 + 24),
                     (unsigned int)&v250,
                     v252,
                     v253,
                     v118,
                     *(_OWORD *)*(_QWORD *)(v10 + 40),
                     v117);
          goto LABEL_10;
        }
        v26 = *(_DWORD *)(v10 + 24);
        goto LABEL_73;
      }
      *(_QWORD *)&v89 = sub_33FB310(
                          *a1,
                          *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL),
                          *(_QWORD *)(*(_QWORD *)(v10 + 40) + 48LL),
                          &v250,
                          v254,
                          v255);
      *(_QWORD *)&v91 = sub_3406EB0(*a1, 57, (unsigned int)&v250, v254, v255, v90, *(_OWORD *)&v249, v89);
LABEL_119:
      result = sub_3406EB0(*a1, 190, (unsigned int)&v250, v252, v253, v92, *(_OWORD *)*(_QWORD *)(v10 + 40), v91);
      goto LABEL_10;
    }
LABEL_66:
    result = sub_3400BD0(*a1, 0, (unsigned int)&v250, v252, v253, 0, 0);
    goto LABEL_10;
  }
  return result;
}
