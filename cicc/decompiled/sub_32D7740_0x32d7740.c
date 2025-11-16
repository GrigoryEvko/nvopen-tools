// Function: sub_32D7740
// Address: 0x32d7740
//
__int64 __fastcall sub_32D7740(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdi
  const __m128i *v5; // roff
  __int64 (__fastcall *v6)(_QWORD *, _DWORD *, int); // rcx
  __int64 (__fastcall *v7)(_QWORD *, _DWORD *, int); // r13
  void *v8; // r14
  __int64 v9; // rsi
  __int64 result; // rax
  __int64 v11; // rsi
  __int64 v12; // r15
  unsigned __int16 *v13; // rdx
  __int64 v14; // rcx
  int v15; // eax
  unsigned __int16 *v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rdx
  __int64 v21; // rdi
  __m128i v22; // xmm1
  __int64 v23; // r8
  __int64 v24; // rcx
  __int64 v25; // rsi
  __int64 v26; // r8
  __m128i *v27; // r9
  __int64 v28; // r9
  unsigned __int64 v29; // rax
  __int64 v30; // rdx
  int v31; // edx
  bool v32; // al
  int v33; // r9d
  __int64 v34; // rdi
  int v35; // ecx
  __int64 *v36; // rax
  __int64 v37; // rdx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rcx
  __int64 v43; // rax
  __int64 v44; // rax
  const __m128i *v45; // rax
  __int64 v46; // rcx
  unsigned int v47; // eax
  __int128 v48; // rax
  int v49; // r9d
  __int64 v50; // r13
  __int64 v51; // r14
  unsigned int *v52; // rax
  __int64 v53; // rax
  __int16 v54; // dx
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdx
  unsigned int v60; // eax
  __int64 v61; // rdx
  bool v62; // cc
  _QWORD *v63; // rdx
  __int64 v64; // rbx
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // r14
  __int64 v68; // r13
  int v69; // r9d
  __int64 v70; // rax
  __int16 v71; // dx
  __int64 v72; // rax
  unsigned int v73; // eax
  bool v74; // al
  __int64 v75; // rdx
  _QWORD *v76; // rbx
  __int64 v77; // r14
  __int128 v78; // rax
  int v79; // r9d
  __int64 v80; // rdx
  __int64 v81; // r8
  __int64 v82; // r9
  __int64 v83; // r14
  __int128 v84; // rax
  int v85; // r9d
  __int128 v86; // rax
  int v87; // r9d
  __int128 v88; // rax
  int v89; // r9d
  __int128 v90; // rax
  __int128 v91; // rax
  __int64 v92; // r14
  int v93; // r9d
  unsigned int v94; // edx
  unsigned __int64 v95; // r14
  int v96; // r9d
  __int64 v97; // r13
  unsigned int v98; // edx
  unsigned __int64 v99; // rbx
  int v100; // r9d
  __int128 v101; // rax
  int v102; // r9d
  __int64 v103; // rax
  __int64 v104; // rax
  __int64 v105; // rdx
  _QWORD *v106; // rax
  __int64 v107; // rdx
  _QWORD *v108; // rax
  __int64 v109; // rax
  __int16 v110; // dx
  __int64 v111; // rax
  unsigned __int16 *v112; // rax
  _QWORD *v113; // rax
  __int64 v114; // r11
  __int64 v115; // rsi
  int v116; // eax
  __int128 v117; // rax
  int v118; // r9d
  __int64 v119; // r13
  __int64 v120; // r8
  __int64 v121; // rax
  __int64 v122; // rax
  __int128 v123; // rax
  int v124; // r9d
  __int128 v125; // rax
  __int128 v126; // rax
  int v127; // r9d
  __int64 v128; // r10
  unsigned int v129; // edx
  int v130; // r9d
  __int64 v131; // rdi
  __int64 v132; // rax
  __int64 v133; // rdx
  __int64 v134; // rcx
  __int64 v135; // r8
  unsigned int v136; // eax
  const __m128i *v137; // rdx
  __int64 v138; // rbx
  __int64 v139; // r12
  __int128 v140; // rax
  int v141; // r9d
  __int64 v142; // rax
  __int128 v143; // rax
  int v144; // r9d
  __int64 v145; // rax
  __int64 v146; // rdx
  __int64 v147; // r14
  __int64 v148; // r13
  __int64 v149; // rbx
  __int128 v150; // rax
  int v151; // r9d
  __int128 v152; // rax
  int v153; // r9d
  __int128 v154; // rax
  int v155; // r9d
  __int64 v156; // rbx
  __int128 v157; // rax
  int v158; // r9d
  unsigned int v159; // edx
  __int64 v160; // r8
  __int64 v161; // r9
  __int64 v162; // r12
  __int64 v163; // rcx
  __int128 v164; // [rsp-30h] [rbp-1D0h]
  __int128 v165; // [rsp-20h] [rbp-1C0h]
  __int128 v166; // [rsp-20h] [rbp-1C0h]
  __int128 v167; // [rsp-20h] [rbp-1C0h]
  __int128 v168; // [rsp-20h] [rbp-1C0h]
  __int128 v169; // [rsp-20h] [rbp-1C0h]
  __int64 v170; // [rsp-10h] [rbp-1B0h]
  __int64 v171; // [rsp-10h] [rbp-1B0h]
  __int128 v172; // [rsp-10h] [rbp-1B0h]
  __int128 v173; // [rsp-10h] [rbp-1B0h]
  __int128 v174; // [rsp-10h] [rbp-1B0h]
  __int128 v175; // [rsp-10h] [rbp-1B0h]
  int v176; // [rsp+8h] [rbp-198h]
  __int64 v177; // [rsp+10h] [rbp-190h]
  unsigned __int64 v178; // [rsp+18h] [rbp-188h]
  _QWORD *v179; // [rsp+20h] [rbp-180h]
  _QWORD *v180; // [rsp+28h] [rbp-178h]
  char v181; // [rsp+30h] [rbp-170h]
  char v182; // [rsp+30h] [rbp-170h]
  __int64 v183; // [rsp+30h] [rbp-170h]
  __int64 v184; // [rsp+30h] [rbp-170h]
  __int64 v185; // [rsp+38h] [rbp-168h]
  __int64 v186; // [rsp+48h] [rbp-158h]
  unsigned int v187; // [rsp+50h] [rbp-150h]
  int v188; // [rsp+50h] [rbp-150h]
  __int64 v189; // [rsp+58h] [rbp-148h]
  __int64 v190; // [rsp+58h] [rbp-148h]
  __int64 v191; // [rsp+60h] [rbp-140h]
  __int64 v192; // [rsp+60h] [rbp-140h]
  unsigned int v193; // [rsp+60h] [rbp-140h]
  __int64 v194; // [rsp+60h] [rbp-140h]
  __int64 v195; // [rsp+68h] [rbp-138h]
  unsigned __int32 v196; // [rsp+70h] [rbp-130h]
  __int64 v197; // [rsp+70h] [rbp-130h]
  unsigned int v198; // [rsp+70h] [rbp-130h]
  __int64 v199; // [rsp+70h] [rbp-130h]
  __int64 v200; // [rsp+70h] [rbp-130h]
  __int128 v201; // [rsp+70h] [rbp-130h]
  char v202; // [rsp+70h] [rbp-130h]
  __int128 v203; // [rsp+70h] [rbp-130h]
  char v204; // [rsp+70h] [rbp-130h]
  __int128 v205; // [rsp+70h] [rbp-130h]
  __int64 v206; // [rsp+78h] [rbp-128h]
  _QWORD *v207; // [rsp+80h] [rbp-120h]
  __int64 v208; // [rsp+80h] [rbp-120h]
  __int128 v209; // [rsp+80h] [rbp-120h]
  __int128 v210; // [rsp+80h] [rbp-120h]
  __int128 v211; // [rsp+80h] [rbp-120h]
  __int64 v212; // [rsp+80h] [rbp-120h]
  __int128 v213; // [rsp+80h] [rbp-120h]
  __int128 v214; // [rsp+80h] [rbp-120h]
  __int64 v215; // [rsp+80h] [rbp-120h]
  __int128 v216; // [rsp+80h] [rbp-120h]
  __int64 v217; // [rsp+C0h] [rbp-E0h]
  __m128i v218; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v219; // [rsp+E0h] [rbp-C0h] BYREF
  int v220; // [rsp+E8h] [rbp-B8h]
  unsigned int v221; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v222; // [rsp+F8h] [rbp-A8h]
  unsigned __int16 v223; // [rsp+100h] [rbp-A0h] BYREF
  __int64 v224; // [rsp+108h] [rbp-98h]
  __int64 v225; // [rsp+110h] [rbp-90h]
  __int64 v226; // [rsp+118h] [rbp-88h]
  __m128i v227; // [rsp+120h] [rbp-80h] BYREF
  __int64 v228; // [rsp+130h] [rbp-70h] BYREF
  __int64 v229; // [rsp+138h] [rbp-68h]
  __int64 v230; // [rsp+140h] [rbp-60h]
  __int64 v231; // [rsp+148h] [rbp-58h]
  __m128i v232; // [rsp+150h] [rbp-50h] BYREF
  __int64 (__fastcall *v233)(_QWORD *, _DWORD *, int); // [rsp+160h] [rbp-40h] BYREF
  void *v234; // [rsp+168h] [rbp-38h]

  v4 = *a1;
  v5 = *(const __m128i **)(a2 + 40);
  v6 = (__int64 (__fastcall *)(_QWORD *, _DWORD *, int))v5[2].m128i_i64[1];
  v7 = v6;
  v8 = (void *)v5[3].m128i_i64[0];
  v9 = v5->m128i_i64[0];
  v218 = _mm_loadu_si128(v5);
  v207 = v6;
  v196 = v5[3].m128i_u32[0];
  result = sub_3401190(v4, v9, v218.m128i_i64[1], v6, v8);
  if ( !result )
  {
    v11 = *(_QWORD *)(a2 + 80);
    v12 = 0;
    v219 = v11;
    if ( v11 )
      sub_B96E90((__int64)&v219, v11, 1);
    v220 = *(_DWORD *)(a2 + 72);
    v13 = (unsigned __int16 *)(*(_QWORD *)(v218.m128i_i64[0] + 48) + 16LL * v218.m128i_u32[2]);
    v191 = v218.m128i_i64[0];
    v14 = *((_QWORD *)v13 + 1);
    v15 = *v13;
    v222 = v14;
    v16 = (unsigned __int16 *)(v207[6] + 16LL * v196);
    LOWORD(v221) = v15;
    v17 = *((_QWORD *)v16 + 1);
    v186 = v17;
    v187 = *v16;
    if ( (_WORD)v15 )
    {
      if ( (unsigned __int16)(v15 - 17) > 0xD3u )
      {
        v232.m128i_i16[0] = v15;
        v232.m128i_i64[1] = v14;
        goto LABEL_14;
      }
      LOWORD(v15) = word_4456580[v15 - 1];
    }
    else
    {
      v195 = v14;
      if ( !sub_30070B0((__int64)&v221) )
      {
        v232.m128i_i64[1] = v195;
        v232.m128i_i16[0] = 0;
LABEL_8:
        v230 = sub_3007260((__int64)&v232);
        v231 = v20;
        LODWORD(v189) = v230;
        goto LABEL_9;
      }
      LOWORD(v15) = sub_3009970((__int64)&v221, v17, v18, v195, v19);
      v12 = v30;
    }
    v232.m128i_i16[0] = v15;
    v232.m128i_i64[1] = v12;
    if ( !(_WORD)v15 )
      goto LABEL_8;
LABEL_14:
    if ( (_WORD)v15 == 1 || (unsigned __int16)(v15 - 504) <= 7u )
      BUG();
    v189 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v15 - 16];
LABEL_9:
    v21 = *a1;
    v22 = _mm_loadu_si128(&v218);
    v233 = v7;
    v234 = v8;
    v232 = v22;
    result = sub_3402EA0(v21, 192, (unsigned int)&v219, v221, v222, 0, (__int64)&v232, 2);
    v24 = v170;
    if ( result )
      goto LABEL_10;
    if ( (_WORD)v221 )
    {
      if ( (unsigned __int16)(v221 - 17) > 0xD3u )
        goto LABEL_20;
    }
    else if ( !sub_30070B0((__int64)&v221) )
    {
      goto LABEL_20;
    }
    result = sub_3295970(a1, a2, (__int64)&v219, v24, v23);
    if ( result )
      goto LABEL_10;
LABEL_20:
    result = sub_329BF20(a1, a2);
    if ( result )
      goto LABEL_10;
    v25 = (__int64)v8;
    v185 = sub_33DFBC0(v7, v8, 0, 0);
    if ( v185 )
    {
      v28 = *a1;
      v232.m128i_i32[2] = v189;
      if ( (unsigned int)v189 > 0x40 )
      {
        v183 = v28;
        sub_C43690((__int64)&v232, -1, 1);
        v28 = v183;
      }
      else
      {
        v29 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v189;
        if ( !(_DWORD)v189 )
          v29 = 0;
        v232.m128i_i64[0] = v29;
      }
      v25 = a2;
      if ( (unsigned __int8)sub_33DD210(v28, a2, 0, &v232, 0) )
      {
        if ( v232.m128i_i32[2] > 0x40u && v232.m128i_i64[0] )
          j_j___libc_free_0_0(v232.m128i_u64[0]);
        goto LABEL_30;
      }
      if ( v232.m128i_i32[2] > 0x40u && v232.m128i_i64[0] )
        j_j___libc_free_0_0(v232.m128i_u64[0]);
      v31 = *(_DWORD *)(v191 + 24);
      if ( v31 != 192 )
        goto LABEL_41;
    }
    else
    {
      v31 = *(_DWORD *)(v191 + 24);
      if ( v31 != 192 )
        goto LABEL_65;
    }
    v232.m128i_i32[0] = v189;
    v234 = sub_3262860;
    v233 = sub_325DCB0;
    v181 = sub_33CACD0(
             (_DWORD)v7,
             (_DWORD)v8,
             *(_QWORD *)(*(_QWORD *)(v191 + 40) + 40LL),
             *(_QWORD *)(*(_QWORD *)(v191 + 40) + 48LL),
             (unsigned int)&v232,
             0,
             0);
    sub_A17130((__int64)&v232);
    v25 = (__int64)v8;
    if ( v181 )
    {
LABEL_30:
      result = sub_3400BD0(*a1, 0, (unsigned int)&v219, v221, v222, 0, 0);
      goto LABEL_10;
    }
    v232.m128i_i32[0] = v189;
    v234 = sub_3262560;
    v233 = sub_325DCE0;
    v182 = sub_33CACD0(
             (_DWORD)v7,
             (_DWORD)v8,
             *(_QWORD *)(*(_QWORD *)(v191 + 40) + 40LL),
             *(_QWORD *)(*(_QWORD *)(v191 + 40) + 48LL),
             (unsigned int)&v232,
             0,
             0);
    sub_A17130((__int64)&v232);
    v26 = v171;
    v27 = &v232;
    if ( v182 )
    {
      *((_QWORD *)&v168 + 1) = v8;
      *(_QWORD *)&v168 = v7;
      *(_QWORD *)&v117 = sub_3406EB0(
                           *a1,
                           56,
                           (unsigned int)&v219,
                           v187,
                           v186,
                           (unsigned int)&v232,
                           v168,
                           *(_OWORD *)(*(_QWORD *)(v191 + 40) + 40LL));
      result = sub_3406EB0(*a1, 192, (unsigned int)&v219, v221, v222, v118, *(_OWORD *)*(_QWORD *)(v191 + 40), v117);
      goto LABEL_10;
    }
    v31 = *(_DWORD *)(v191 + 24);
    if ( v185 )
    {
LABEL_41:
      if ( v31 != 216 )
        goto LABEL_42;
      v45 = *(const __m128i **)(v191 + 40);
      v46 = v45->m128i_i64[0];
      v184 = v45->m128i_i64[0];
      if ( *(_DWORD *)(v45->m128i_i64[0] + 24) != 192 )
      {
        v35 = v189 - 1;
        v34 = *(_QWORD *)(v185 + 96) + 24LL;
        goto LABEL_46;
      }
      v227 = _mm_loadu_si128(v45);
      v103 = *(_QWORD *)(v46 + 40);
      v25 = *(_QWORD *)(v103 + 48);
      v104 = sub_33DFBC0(*(_QWORD *)(v103 + 40), v25, 0, 0);
      if ( v104 )
      {
        v105 = *(_QWORD *)(v104 + 96);
        v106 = *(_QWORD **)(v105 + 24);
        if ( *(_DWORD *)(v105 + 32) > 0x40u )
          v106 = (_QWORD *)*v106;
        v179 = v106;
        v107 = *(_QWORD *)(v185 + 96);
        v108 = *(_QWORD **)(v107 + 24);
        if ( *(_DWORD *)(v107 + 32) > 0x40u )
          v108 = (_QWORD *)*v108;
        v180 = v108;
        v109 = *(_QWORD *)(v184 + 48) + 16LL * v227.m128i_u32[2];
        v110 = *(_WORD *)v109;
        v111 = *(_QWORD *)(v109 + 8);
        LOWORD(v228) = v110;
        v229 = v111;
        v112 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v184 + 40) + 40LL) + 48LL)
                                  + 16LL * *(unsigned int *)(*(_QWORD *)(v184 + 40) + 48LL));
        v177 = *((_QWORD *)v112 + 1);
        v176 = *v112;
        v178 = sub_32844A0((unsigned __int16 *)&v228, v25);
        v113 = (_QWORD *)((char *)v179 + (unsigned int)v189);
        if ( v113 == (_QWORD *)v178 )
        {
          v131 = *a1;
          if ( (_QWORD *)((char *)v180 + (_QWORD)v179) >= v113 )
          {
            result = sub_3400BD0(v131, 0, (unsigned int)&v219, v221, v222, 0, 0);
            goto LABEL_10;
          }
          *(_QWORD *)&v154 = sub_3400BD0(
                               v131,
                               (int)v179 + (int)v180,
                               (unsigned int)&v219,
                               v176,
                               v177,
                               0,
                               0,
                               (char *)v179 + (unsigned int)v189);
          *(_QWORD *)&v152 = sub_3406EB0(
                               *a1,
                               192,
                               (unsigned int)&v219,
                               v228,
                               v229,
                               v155,
                               *(_OWORD *)*(_QWORD *)(v184 + 40),
                               v154);
        }
        else
        {
          if ( !(unsigned __int8)sub_3286E00(&v218) )
            goto LABEL_122;
          if ( !(unsigned __int8)sub_3286E00(&v227) )
            goto LABEL_122;
          v25 = (__int64)v179 + (_QWORD)v180;
          if ( (_QWORD *)((char *)v179 + (_QWORD)v180) >= (_QWORD *)v178 )
            goto LABEL_122;
          *(_QWORD *)&v143 = sub_3400BD0(*a1, v25, (unsigned int)&v219, v176, v177, 0, 0, v114);
          v145 = sub_3406EB0(*a1, 192, (unsigned int)&v219, v228, v229, v144, *(_OWORD *)*(_QWORD *)(v184 + 40), v143);
          v147 = v146;
          v148 = v145;
          v149 = *a1;
          sub_F0A5D0((__int64)&v232, v178, v189 - (_DWORD)v180);
          *(_QWORD *)&v150 = sub_34007B0(v149, (unsigned int)&v232, (unsigned int)&v219, v228, v229, 0, 0);
          v216 = v150;
          sub_969240(v232.m128i_i64);
          *((_QWORD *)&v164 + 1) = v147;
          *(_QWORD *)&v164 = v148;
          *(_QWORD *)&v152 = sub_3406EB0(*a1, 186, (unsigned int)&v219, v228, v229, v151, v164, v216);
        }
        result = sub_33FAF80(*a1, 216, (unsigned int)&v219, v221, v222, v153, v152);
        goto LABEL_10;
      }
LABEL_122:
      v31 = *(_DWORD *)(v191 + 24);
LABEL_42:
      if ( v31 != 190 )
        goto LABEL_43;
LABEL_69:
      v42 = (__int64)v207;
      v43 = *(_QWORD *)(v191 + 40);
      if ( *(_QWORD **)(v43 + 40) == v207 && (v42 = v196, *(_DWORD *)(v43 + 48) == v196)
        || (v44 = *(_QWORD *)(v191 + 56)) != 0 && !*(_QWORD *)(v44 + 32) )
      {
        v25 = a2;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, __int64, __int64, __m128i *))(*(_QWORD *)a1[1] + 416LL))(
               a1[1],
               a2,
               *((unsigned int *)a1 + 6),
               v42,
               v26,
               v27) )
        {
          v234 = sub_3264690;
          v232.m128i_i32[0] = v189;
          v233 = sub_325DD10;
          v202 = sub_33CACD0(
                   (_DWORD)v7,
                   (_DWORD)v8,
                   *(_QWORD *)(*(_QWORD *)(v191 + 40) + 40LL),
                   *(_QWORD *)(*(_QWORD *)(v191 + 40) + 48LL),
                   (unsigned int)&v232,
                   0,
                   1);
          sub_A17130((__int64)&v232);
          if ( v202 )
          {
            *(_QWORD *)&v88 = sub_33FB310(
                                *a1,
                                *(_QWORD *)(*(_QWORD *)(v191 + 40) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(v191 + 40) + 48LL),
                                &v219,
                                v187,
                                v186);
            *((_QWORD *)&v172 + 1) = v8;
            *(_QWORD *)&v172 = v7;
            v203 = v88;
            *(_QWORD *)&v90 = sub_3406EB0(*a1, 57, (unsigned int)&v219, v187, v186, v89, v88, v172);
            v211 = v90;
            *(_QWORD *)&v91 = sub_34015B0(*a1, &v219, v221, v222, 0, 0);
            v92 = *((_QWORD *)&v91 + 1);
            v217 = sub_3406EB0(*a1, 192, (unsigned int)&v219, v221, v222, v93, v91, v203);
            v95 = v94 | v92 & 0xFFFFFFFF00000000LL;
            *((_QWORD *)&v167 + 1) = v95;
            *(_QWORD *)&v167 = v217;
            v97 = sub_3406EB0(*a1, 190, (unsigned int)&v219, v221, v222, v96, v167, v211);
            v99 = v98 | v95 & 0xFFFFFFFF00000000LL;
            *(_QWORD *)&v101 = sub_3406EB0(
                                 *a1,
                                 190,
                                 (unsigned int)&v219,
                                 v221,
                                 v222,
                                 v100,
                                 *(_OWORD *)*(_QWORD *)(v191 + 40),
                                 v211);
            *((_QWORD *)&v173 + 1) = v99;
            *(_QWORD *)&v173 = v97;
LABEL_109:
            result = sub_3406EB0(*a1, 186, (unsigned int)&v219, v221, v222, v102, v101, v173);
            goto LABEL_10;
          }
          v234 = sub_3264690;
          v233 = sub_325DD10;
          v232.m128i_i32[0] = v189;
          v122 = *(_QWORD *)(v191 + 40);
          v25 = *(_QWORD *)(v122 + 48);
          v204 = sub_33CACD0(*(_QWORD *)(v122 + 40), v25, (_DWORD)v7, (_DWORD)v8, (unsigned int)&v232, 0, 1);
          sub_A17130((__int64)&v232);
          if ( v204 )
          {
            *(_QWORD *)&v123 = sub_33FB310(
                                 *a1,
                                 *(_QWORD *)(*(_QWORD *)(v191 + 40) + 40LL),
                                 *(_QWORD *)(*(_QWORD *)(v191 + 40) + 48LL),
                                 &v219,
                                 v187,
                                 v186);
            *((_QWORD *)&v169 + 1) = v8;
            *(_QWORD *)&v169 = v7;
            *(_QWORD *)&v125 = sub_3406EB0(*a1, 57, (unsigned int)&v219, v187, v186, v124, v169, v123);
            v213 = v125;
            *(_QWORD *)&v126 = sub_34015B0(*a1, &v219, v221, v222, 0, 0);
            *((_QWORD *)&v175 + 1) = v8;
            *(_QWORD *)&v175 = v7;
            v206 = *((_QWORD *)&v126 + 1);
            v128 = sub_3406EB0(*a1, 192, (unsigned int)&v219, v221, v222, v127, v126, v175);
            *((_QWORD *)&v205 + 1) = v129 | v206 & 0xFFFFFFFF00000000LL;
            *(_QWORD *)&v205 = v128;
            *(_QWORD *)&v101 = sub_3406EB0(
                                 *a1,
                                 192,
                                 (unsigned int)&v219,
                                 v221,
                                 v222,
                                 v130,
                                 *(_OWORD *)*(_QWORD *)(v191 + 40),
                                 v213);
            v173 = v205;
            goto LABEL_109;
          }
        }
      }
      if ( v185 )
      {
        v31 = *(_DWORD *)(v191 + 24);
LABEL_43:
        if ( v31 != 215 )
        {
LABEL_44:
          v188 = v31;
          v197 = *(_QWORD *)(v185 + 96) + 24LL;
          v32 = sub_D94970(v197, (_QWORD *)(unsigned int)(v189 - 1));
          v34 = v197;
          v35 = v189 - 1;
          v31 = v188;
          if ( v32 && v188 == 191 )
          {
            *((_QWORD *)&v174 + 1) = v8;
            *(_QWORD *)&v174 = v7;
            result = sub_3406EB0(
                       *a1,
                       192,
                       (unsigned int)&v219,
                       v221,
                       v222,
                       v33,
                       *(_OWORD *)*(_QWORD *)(v191 + 40),
                       v174);
            goto LABEL_10;
          }
LABEL_46:
          if ( !(_DWORD)v189 || v31 != 199 )
          {
LABEL_48:
            if ( *((_DWORD *)v207 + 6) != 216 )
            {
LABEL_49:
              if ( (unsigned int)(*(_DWORD *)(v191 + 24) - 186) > 2 )
                goto LABEL_52;
              v36 = *(__int64 **)(v191 + 40);
              v37 = v36[5];
              if ( *(_DWORD *)(*v36 + 24) == 190 )
              {
                v50 = v36[5];
                v37 = *v36;
                v51 = *((unsigned int *)v36 + 12);
              }
              else
              {
                if ( *(_DWORD *)(v37 + 24) != 190 )
                  goto LABEL_52;
                v50 = *v36;
                v51 = *((unsigned int *)v36 + 2);
              }
              v52 = *(unsigned int **)(v37 + 40);
              if ( v185 == *((_QWORD *)v52 + 5) )
              {
                *(_QWORD *)&v209 = *(_QWORD *)v52;
                *((_QWORD *)&v209 + 1) = v52[2];
                if ( *(_DWORD *)(*(_QWORD *)v52 + 24LL) == 214 )
                {
                  v190 = *(_QWORD *)v52;
                  v198 = v52[2];
                  v53 = *(_QWORD *)(**(_QWORD **)(*(_QWORD *)v52 + 40LL) + 48LL)
                      + 16LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)v52 + 40LL) + 8LL);
                  v54 = *(_WORD *)v53;
                  v55 = *(_QWORD *)(v53 + 8);
                  v227.m128i_i16[0] = v54;
                  v227.m128i_i64[1] = v55;
                  v228 = sub_2D5B750((unsigned __int16 *)&v227);
                  v229 = v56;
                  v57 = *(_QWORD *)(v190 + 48) + 16LL * v198;
                  LOWORD(v56) = *(_WORD *)v57;
                  v58 = *(_QWORD *)(v57 + 8);
                  v223 = v56;
                  v224 = v58;
                  v225 = sub_2D5B750(&v223);
                  v226 = v59;
                  if ( v228 )
                    LOBYTE(v59) = v229;
                  v232.m128i_i8[8] = v59;
                  v232.m128i_i64[0] = v225 - v228;
                  v60 = sub_CA1930(&v232);
                  v61 = *(_QWORD *)(v185 + 96);
                  v62 = *(_DWORD *)(v61 + 32) <= 0x40u;
                  v63 = *(_QWORD **)(v61 + 24);
                  if ( !v62 )
                    v63 = (_QWORD *)*v63;
                  if ( v60 >= (unsigned __int64)v63 )
                  {
                    v64 = *a1;
                    sub_3285E70((__int64)&v232, v218.m128i_i64[0]);
                    *((_QWORD *)&v165 + 1) = v51;
                    *(_QWORD *)&v165 = v50;
                    v65 = sub_3406EB0(v64, 192, (unsigned int)&v232, v221, v222, 0, v165, (unsigned __int64)v185);
                    v67 = v66;
                    v68 = v65;
                    sub_3285E70((__int64)&v228, v218.m128i_i64[0]);
                    *((_QWORD *)&v166 + 1) = v67;
                    *(_QWORD *)&v166 = v68;
                    v199 = sub_3406EB0(v64, *(_DWORD *)(v191 + 24), (unsigned int)&v228, v221, v222, v69, v166, v209);
                    sub_9C6650(&v228);
                    sub_9C6650(&v232);
                    result = v199;
                    goto LABEL_10;
                  }
                }
              }
LABEL_52:
              if ( !(unsigned __int8)sub_32D0FE0((__int64)a1, a2, 0) )
              {
                if ( (*(_BYTE *)(v185 + 32) & 8) == 0 )
                {
                  result = sub_327E0B0(a1, a2);
                  if ( result )
                    goto LABEL_10;
                }
LABEL_54:
                result = sub_32B3F40(a1, a2);
                if ( !result )
                {
                  v40 = *(_QWORD *)(a2 + 56);
                  if ( v40 && !*(_QWORD *)(v40 + 32) )
                  {
                    v115 = *(_QWORD *)(v40 + 16);
                    if ( *(_DWORD *)(v115 + 24) != 216 )
                    {
LABEL_126:
                      v116 = *(_DWORD *)(v115 + 24);
                      if ( (unsigned int)(v116 - 186) <= 2 || v116 == 305 )
                        sub_32B3E80((__int64)a1, v115, 1, 0, v38, v39);
                      goto LABEL_56;
                    }
                    v132 = *(_QWORD *)(v115 + 56);
                    if ( v132 && !*(_QWORD *)(v132 + 32) )
                    {
                      v115 = *(_QWORD *)(v132 + 16);
                      goto LABEL_126;
                    }
                  }
LABEL_56:
                  result = sub_328A0F0(a2, (__int64)&v219, *a1, a1[1]);
                  if ( !result )
                  {
                    v41 = sub_326C8E0(a1, a2);
                    result = 0;
                    if ( v41 )
                      result = v41;
                  }
                }
LABEL_10:
                if ( v219 )
                {
                  v208 = result;
                  sub_B91220((__int64)&v219, v219);
                  return v208;
                }
                return result;
              }
LABEL_68:
              result = a2;
              goto LABEL_10;
            }
            goto LABEL_79;
          }
          if ( (v35 & (unsigned int)v189) != 0
            || (_BitScanReverse(&v47, v189), !sub_D94970(v34, (_QWORD *)(int)(31 - (v47 ^ 0x1F)))) )
          {
            if ( *((_DWORD *)v207 + 6) != 216 )
              goto LABEL_52;
LABEL_79:
            if ( *(_DWORD *)(*(_QWORD *)v207[5] + 24LL) != 186 )
              goto LABEL_49;
LABEL_80:
            *(_QWORD *)&v48 = sub_32CB9C0((__int64)a1, v207);
            if ( (_QWORD)v48 )
            {
              result = sub_3406EB0(*a1, 192, (unsigned int)&v219, v221, v222, v49, *(_OWORD *)&v218, v48);
              goto LABEL_10;
            }
            if ( v185 )
              goto LABEL_49;
LABEL_67:
            if ( !(unsigned __int8)sub_32D0FE0((__int64)a1, a2, 0) )
              goto LABEL_54;
            goto LABEL_68;
          }
          sub_33DD090(&v232, *a1, **(_QWORD **)(v191 + 40), *(_QWORD *)(*(_QWORD *)(v191 + 40) + 8LL), 0);
          if ( sub_9867B0((__int64)&v233) )
          {
            sub_9865C0((__int64)&v228, (__int64)&v232);
            sub_987160((__int64)&v228, (__int64)&v232, v133, v134, v135);
            v227.m128i_i32[2] = v229;
            v227.m128i_i64[0] = v228;
            if ( sub_D94970((__int64)&v227, 0) )
            {
              v162 = *a1;
              sub_3285E70((__int64)&v228, v218.m128i_i64[0]);
              v194 = sub_3400BD0(v162, 1, (unsigned int)&v228, v221, v222, 0, 0, v163);
              sub_9C6650(&v228);
              v142 = v194;
            }
            else
            {
              if ( !sub_986BA0((__int64)&v227) )
              {
                sub_969240(v227.m128i_i64);
                sub_969240((__int64 *)&v233);
                sub_969240(v232.m128i_i64);
                goto LABEL_48;
              }
              v136 = sub_D949C0((__int64)&v227);
              v137 = *(const __m128i **)(v191 + 40);
              v214 = (__int128)_mm_loadu_si128(v137);
              if ( v136 )
              {
                v193 = v136;
                sub_3285E70((__int64)&v228, v218.m128i_i64[0]);
                v156 = *a1;
                *(_QWORD *)&v157 = sub_3400E40(*a1, v193, v221, v222, &v228);
                v138 = sub_3406EB0(v156, 192, (unsigned int)&v228, v221, v222, v158, v214, v157);
                *((_QWORD *)&v214 + 1) = v159 | *((_QWORD *)&v214 + 1) & 0xFFFFFFFF00000000LL;
                sub_32B3E80((__int64)a1, v138, 1, 0, v160, v161);
                sub_9C6650(&v228);
              }
              else
              {
                v138 = v137->m128i_i64[0];
              }
              v139 = *a1;
              *(_QWORD *)&v140 = sub_3400BD0(v139, 1, (unsigned int)&v219, v221, v222, 0, 0, v137);
              v142 = sub_3406EB0(
                       v139,
                       188,
                       (unsigned int)&v219,
                       v221,
                       v222,
                       v141,
                       __PAIR128__(*((unsigned __int64 *)&v214 + 1), v138),
                       v140);
            }
            v215 = v142;
            sub_969240(v227.m128i_i64);
            v121 = v215;
          }
          else
          {
            v119 = *a1;
            sub_3285E70((__int64)&v228, v218.m128i_i64[0]);
            v192 = sub_3400BD0(v119, 0, (unsigned int)&v228, v221, v222, 0, 0, v120);
            sub_9C6650(&v228);
            v121 = v192;
          }
          v212 = v121;
          sub_969240((__int64 *)&v233);
          sub_969240(v232.m128i_i64);
          result = v212;
          goto LABEL_10;
        }
        v70 = *(_QWORD *)(**(_QWORD **)(v191 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v191 + 40) + 8LL);
        v71 = *(_WORD *)v70;
        v72 = *(_QWORD *)(v70 + 8);
        v227.m128i_i16[0] = v71;
        v227.m128i_i64[1] = v72;
        v73 = sub_32844A0((unsigned __int16 *)&v227, v25);
        v200 = *(_QWORD *)(v185 + 96);
        v74 = sub_986EE0(v200 + 24, v73);
        v75 = v200;
        if ( !v74 )
        {
          result = sub_3288990(*a1, v221, v222);
          goto LABEL_10;
        }
        if ( *((_BYTE *)a1 + 34) )
        {
          if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)a1[1] + 2192LL))(
                  a1[1],
                  192,
                  v227.m128i_u32[0],
                  v227.m128i_i64[1]) )
          {
            v31 = *(_DWORD *)(v191 + 24);
            goto LABEL_44;
          }
          v75 = *(_QWORD *)(v185 + 96);
        }
        v76 = *(_QWORD **)(v75 + 24);
        if ( *(_DWORD *)(v75 + 32) > 0x40u )
          v76 = (_QWORD *)*v76;
        sub_3285E70((__int64)&v228, v218.m128i_i64[0]);
        v77 = *a1;
        *(_QWORD *)&v78 = sub_3400E40(*a1, v76, v227.m128i_u32[0], v227.m128i_i64[1], &v228);
        *(_QWORD *)&v201 = sub_3406EB0(
                             v77,
                             192,
                             (unsigned int)&v228,
                             v227.m128i_i32[0],
                             v227.m128i_i32[2],
                             v79,
                             *(_OWORD *)*(_QWORD *)(v191 + 40),
                             v78);
        *((_QWORD *)&v201 + 1) = v80;
        sub_32B3E80((__int64)a1, v201, 1, 0, v81, v82);
        sub_F0A5D0((__int64)&v232, v189, v189 - (_DWORD)v76);
        v83 = *a1;
        *(_QWORD *)&v84 = sub_34007B0(*a1, (unsigned int)&v232, (unsigned int)&v219, v221, v222, 0, 0);
        v210 = v84;
        *(_QWORD *)&v86 = sub_33FAF80(*a1, 215, (unsigned int)&v219, v221, v222, v85, v201);
        *(_QWORD *)&v201 = sub_3406EB0(v83, 186, (unsigned int)&v219, v221, v222, v87, v86, v210);
        sub_969240(v232.m128i_i64);
        sub_9C6650(&v228);
        result = v201;
        goto LABEL_10;
      }
LABEL_66:
      if ( *((_DWORD *)v207 + 6) != 216 || *(_DWORD *)(*(_QWORD *)v207[5] + 24LL) != 186 )
        goto LABEL_67;
      goto LABEL_80;
    }
LABEL_65:
    if ( v31 != 190 )
      goto LABEL_66;
    goto LABEL_69;
  }
  return result;
}
