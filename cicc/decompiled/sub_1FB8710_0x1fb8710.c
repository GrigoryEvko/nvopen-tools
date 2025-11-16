// Function: sub_1FB8710
// Address: 0x1fb8710
//
__int64 __fastcall sub_1FB8710(
        __int64 a1,
        __int64 a2,
        double a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v11; // rax
  __m128i v12; // xmm0
  __int64 v13; // r15
  __int64 v14; // rcx
  __m128i v15; // xmm1
  unsigned __int8 *v16; // rax
  __int64 v17; // rdx
  const void **v18; // rax
  int v19; // ecx
  int v20; // r8d
  __int64 v21; // r9
  __int64 result; // rax
  int v23; // ecx
  int v24; // r8d
  int v25; // r9d
  __int64 v26; // r14
  int v27; // eax
  __int64 v28; // rax
  unsigned int v29; // ecx
  __int64 v30; // rdi
  unsigned int v31; // edx
  __int64 v33; // rsi
  __int64 *v34; // r12
  __int64 v35; // rsi
  const __m128i *v36; // rax
  __int64 v37; // r14
  _DWORD *v38; // rdx
  unsigned __int8 *v39; // rcx
  int v40; // eax
  __int64 v41; // rcx
  bool v42; // r14
  int v43; // eax
  __int64 *v44; // r14
  __int64 v45; // rdx
  __int64 *v46; // r12
  __int64 *v47; // rax
  char v48; // al
  __int64 v49; // r9
  char v50; // cl
  __int64 v51; // rdi
  __int64 v52; // rsi
  __int64 *v53; // r12
  __int64 *v54; // r9
  __int64 v55; // rdx
  __int64 v56; // rcx
  int v57; // r8d
  __int64 v58; // r9
  __int64 *v59; // rdi
  _QWORD *v60; // rbx
  __int64 *v61; // r12
  int v62; // edx
  __int64 v63; // rdx
  int v64; // eax
  unsigned int v65; // edx
  __int64 *v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rsi
  __int64 v69; // r9
  __int64 v70; // rcx
  int v71; // eax
  __int64 v72; // r12
  __int64 *v73; // rax
  unsigned int v74; // edx
  __int64 *v75; // r10
  unsigned int v76; // edx
  int v77; // eax
  __int128 v78; // rax
  __int64 *v79; // r12
  _DWORD *v80; // rax
  __int64 v81; // rdx
  int v82; // ecx
  int v83; // r8d
  int v84; // r9d
  __int64 v85; // rax
  __int64 v86; // rax
  _QWORD *v87; // r12
  __int64 v88; // rax
  int v89; // eax
  __int64 *v90; // rbx
  const void ***v91; // rax
  __int128 v92; // rax
  unsigned __int64 v93; // rdx
  __int64 *v94; // r12
  __int128 v95; // rax
  bool v96; // al
  __int64 v97; // rax
  __int64 *v98; // rdi
  __int64 v99; // rax
  unsigned __int64 v100; // rdx
  __int128 v101; // rax
  __int64 *v102; // r15
  __int128 v103; // rax
  __int64 *v104; // r12
  __int128 v105; // rax
  __int64 v106; // rax
  __int64 v107; // rax
  __int64 v108; // rsi
  __int64 v109; // rax
  unsigned int v110; // eax
  __int64 v111; // rdx
  _QWORD *v112; // rax
  _QWORD *v113; // rsi
  const __m128i *v114; // roff
  __int64 v115; // r12
  unsigned __int8 *v116; // rax
  const void **v117; // r8
  __int64 v118; // rcx
  __int64 *v119; // r15
  __int64 v120; // rax
  __int64 v121; // rdx
  __int64 v122; // rdi
  const void ***v123; // rdx
  __int128 v124; // rax
  __int64 *v125; // r15
  __int64 v126; // rax
  __int64 v127; // rax
  char *v128; // r12
  __int64 v129; // rax
  char *v130; // r14
  __int64 *v131; // r13
  const void **v132; // r8
  __int64 v133; // rcx
  __int128 v134; // rax
  __int64 *v135; // rax
  __int64 v136; // rax
  __int128 v137; // rax
  __int64 v138; // r15
  __int64 *v139; // r14
  __int64 v140; // rax
  unsigned __int8 *v141; // rax
  __int64 v142; // rdx
  __int64 v143; // rax
  __int64 v144; // r8
  __int64 v145; // r9
  unsigned int v146; // eax
  unsigned int v147; // ebx
  bool v148; // al
  __int64 v149; // r11
  __int128 v150; // rax
  __int64 v151; // rax
  unsigned __int64 v152; // rdx
  __int64 v153; // rax
  __int64 *v154; // rdi
  const void ***v155; // rax
  __int128 v156; // rax
  __int128 v157; // rax
  __int64 v158; // r14
  __int64 *v159; // r12
  __int128 v160; // rax
  __int64 v161; // r12
  __int64 *v162; // r14
  __int128 v163; // rax
  __int64 *v164; // r12
  int v165; // eax
  __int64 *v166; // rbx
  const void ***v167; // rax
  __int128 v168; // rax
  __int64 *v169; // rax
  __int128 v170; // [rsp-20h] [rbp-1C0h]
  __int128 v171; // [rsp-10h] [rbp-1B0h]
  __int64 v172; // [rsp+8h] [rbp-198h]
  __int64 *v173; // [rsp+18h] [rbp-188h]
  unsigned int v174; // [rsp+20h] [rbp-180h]
  unsigned int v175; // [rsp+20h] [rbp-180h]
  __int64 *v176; // [rsp+28h] [rbp-178h]
  __int64 v177; // [rsp+28h] [rbp-178h]
  __int64 v178; // [rsp+30h] [rbp-170h]
  char v179; // [rsp+30h] [rbp-170h]
  __m128i v180; // [rsp+30h] [rbp-170h]
  _DWORD *v181; // [rsp+30h] [rbp-170h]
  __int64 v182; // [rsp+30h] [rbp-170h]
  char v183; // [rsp+30h] [rbp-170h]
  __int64 v184; // [rsp+40h] [rbp-160h]
  __int128 v185; // [rsp+40h] [rbp-160h]
  char v186; // [rsp+40h] [rbp-160h]
  __int64 *v187; // [rsp+40h] [rbp-160h]
  __int64 v188; // [rsp+40h] [rbp-160h]
  __int64 v189; // [rsp+40h] [rbp-160h]
  int v190; // [rsp+50h] [rbp-150h]
  char v191; // [rsp+50h] [rbp-150h]
  int v192; // [rsp+60h] [rbp-140h]
  unsigned int v193; // [rsp+60h] [rbp-140h]
  unsigned __int64 v194; // [rsp+60h] [rbp-140h]
  __int64 v195; // [rsp+60h] [rbp-140h]
  __int64 *v196; // [rsp+60h] [rbp-140h]
  unsigned int v197; // [rsp+68h] [rbp-138h]
  const void **v198; // [rsp+68h] [rbp-138h]
  unsigned int v199; // [rsp+70h] [rbp-130h]
  __int64 v200; // [rsp+70h] [rbp-130h]
  __int64 v201; // [rsp+70h] [rbp-130h]
  unsigned __int64 v202; // [rsp+78h] [rbp-128h]
  unsigned __int64 v203; // [rsp+80h] [rbp-120h]
  __int128 v204; // [rsp+80h] [rbp-120h]
  __m128i v205; // [rsp+80h] [rbp-120h]
  __int128 v206; // [rsp+80h] [rbp-120h]
  __int64 v207; // [rsp+90h] [rbp-110h]
  __int64 *v208; // [rsp+90h] [rbp-110h]
  __int64 v209; // [rsp+90h] [rbp-110h]
  __int64 v210; // [rsp+90h] [rbp-110h]
  __int64 *v211; // [rsp+90h] [rbp-110h]
  __int128 v212; // [rsp+90h] [rbp-110h]
  __int64 v213; // [rsp+90h] [rbp-110h]
  __int128 v214; // [rsp+90h] [rbp-110h]
  __int64 *v215; // [rsp+90h] [rbp-110h]
  __int128 v216; // [rsp+90h] [rbp-110h]
  unsigned __int64 v217; // [rsp+98h] [rbp-108h]
  __int64 *v218; // [rsp+A0h] [rbp-100h]
  __int128 v219; // [rsp+A0h] [rbp-100h]
  __int128 v220; // [rsp+A0h] [rbp-100h]
  unsigned __int32 v221; // [rsp+A0h] [rbp-100h]
  __int64 *v222; // [rsp+A0h] [rbp-100h]
  __int64 *v223; // [rsp+A0h] [rbp-100h]
  __int128 v224; // [rsp+A0h] [rbp-100h]
  __int64 v225; // [rsp+A0h] [rbp-100h]
  __int128 v226; // [rsp+A0h] [rbp-100h]
  unsigned __int64 v227; // [rsp+A8h] [rbp-F8h]
  __int64 v228; // [rsp+F0h] [rbp-B0h] BYREF
  const void **v229; // [rsp+F8h] [rbp-A8h]
  __int64 v230[2]; // [rsp+100h] [rbp-A0h] BYREF
  __int64 v231[2]; // [rsp+110h] [rbp-90h] BYREF
  char v232[8]; // [rsp+120h] [rbp-80h] BYREF
  __int64 v233; // [rsp+128h] [rbp-78h]
  __int64 v234; // [rsp+130h] [rbp-70h] BYREF
  int v235; // [rsp+138h] [rbp-68h]
  unsigned __int64 v236; // [rsp+140h] [rbp-60h] BYREF
  int v237; // [rsp+148h] [rbp-58h]
  unsigned __int64 v238; // [rsp+150h] [rbp-50h] BYREF
  __int64 v239; // [rsp+158h] [rbp-48h]
  __int64 (__fastcall *v240)(unsigned __int64 *, unsigned __int64 *, int); // [rsp+160h] [rbp-40h]
  void *v241; // [rsp+168h] [rbp-38h]

  v11 = *(_QWORD *)(a2 + 32);
  v12 = _mm_loadu_si128((const __m128i *)v11);
  v13 = *(_QWORD *)v11;
  v14 = *(unsigned int *)(v11 + 8);
  v15 = _mm_loadu_si128((const __m128i *)(v11 + 40));
  v203 = *(_QWORD *)(v11 + 40);
  v197 = *(_DWORD *)(v11 + 48);
  v16 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v11 + 40LL) + 16LL * (unsigned int)v14);
  v192 = v14;
  v17 = *v16;
  v18 = (const void **)*((_QWORD *)v16 + 1);
  LOBYTE(v228) = v17;
  v229 = v18;
  v199 = sub_1D159C0((__int64)&v228, a2, v17, v14, a8, a9);
  if ( (_BYTE)v228 )
  {
    if ( (unsigned __int8)(v228 - 14) > 0x5Fu )
      goto LABEL_6;
  }
  else if ( !sub_1F58D20((__int64)&v228) )
  {
    goto LABEL_6;
  }
  result = (__int64)sub_1FA8C50(a1, a2, *(double *)v12.m128i_i64, *(double *)v15.m128i_i64, a5);
  if ( result )
    return result;
  if ( *(_WORD *)(v203 + 24) == 104 )
  {
    if ( (unsigned __int8)sub_1D23510(v203) )
    {
      if ( *(_WORD *)(v13 + 24) == 118 )
      {
        v36 = *(const __m128i **)(v13 + 32);
        v21 = v36[2].m128i_i64[1];
        a5 = _mm_loadu_si128(v36);
        if ( *(_WORD *)(v21 + 24) == 104 )
        {
          v184 = v36[2].m128i_i64[1];
          v37 = v36->m128i_i64[0];
          if ( (unsigned __int8)sub_1D23510(v21) )
          {
            v21 = v184;
            if ( *(_WORD *)(v37 + 24) == 137 )
            {
              v38 = *(_DWORD **)(a1 + 8);
              v39 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(v37 + 32) + 40LL)
                                      + 16LL * *(unsigned int *)(*(_QWORD *)(v37 + 32) + 8LL));
              v40 = *v39;
              v41 = *((_QWORD *)v39 + 1);
              LOBYTE(v238) = v40;
              v239 = v41;
              if ( (_BYTE)v40 )
              {
                v19 = v40 - 14;
                if ( (unsigned __int8)(v40 - 14) > 0x5Fu )
                {
                  v19 = v40 - 8;
                  v42 = (unsigned __int8)(v40 - 86) <= 0x17u || (unsigned __int8)(v40 - 8) <= 5u;
                  goto LABEL_30;
                }
              }
              else
              {
                v181 = v38;
                v42 = sub_1F58CD0((__int64)&v238);
                v96 = sub_1F58D20((__int64)&v238);
                v38 = v181;
                v21 = v184;
                if ( !v96 )
                {
LABEL_30:
                  if ( v42 )
                    v43 = v38[16];
                  else
                    v43 = v38[15];
                  goto LABEL_32;
                }
              }
              v43 = v38[17];
LABEL_32:
              if ( v43 == 2 )
              {
                v44 = *(__int64 **)a1;
                v238 = *(_QWORD *)(a2 + 72);
                if ( v238 )
                {
                  v178 = v21;
                  sub_1F6CA20((__int64 *)&v238);
                  v21 = v178;
                }
                LODWORD(v239) = *(_DWORD *)(a2 + 64);
                *(_QWORD *)&v185 = sub_1D32920(
                                     v44,
                                     0x7Au,
                                     (__int64)&v238,
                                     (unsigned int)v228,
                                     (__int64)v229,
                                     v21,
                                     *(double *)v12.m128i_i64,
                                     *(double *)v15.m128i_i64,
                                     a5,
                                     v203);
                *((_QWORD *)&v185 + 1) = v45;
                sub_17CD270((__int64 *)&v238);
                if ( (_QWORD)v185 )
                {
                  v46 = *(__int64 **)a1;
                  v238 = *(_QWORD *)(a2 + 72);
                  if ( v238 )
                    sub_1F6CA20((__int64 *)&v238);
                  LODWORD(v239) = *(_DWORD *)(a2 + 64);
                  v47 = sub_1D332F0(
                          v46,
                          118,
                          (__int64)&v238,
                          (unsigned int)v228,
                          v229,
                          0,
                          *(double *)v12.m128i_i64,
                          *(double *)v15.m128i_i64,
                          a5,
                          a5.m128i_i64[0],
                          a5.m128i_u64[1],
                          v185);
LABEL_39:
                  v208 = v47;
LABEL_40:
                  sub_17CD270((__int64 *)&v238);
                  return (__int64)v208;
                }
              }
            }
          }
        }
      }
    }
  }
LABEL_6:
  v26 = sub_1D1ADA0(v15.m128i_i64[0], v15.m128i_u32[2], v15.m128i_i64[1], v19, v20, v21);
  v27 = *(unsigned __int16 *)(v13 + 24);
  if ( v27 != 32 && v27 != 10 || (*(_BYTE *)(v13 + 26) & 8) != 0 || !v26 || (*(_BYTE *)(v26 + 26) & 8) != 0 )
  {
    v28 = sub_1D1ADA0(v12.m128i_i64[0], v12.m128i_u32[2], v12.m128i_i64[1], v23, v24, v25);
    if ( v28 )
    {
      v30 = *(_QWORD *)(v28 + 88);
      v31 = *(_DWORD *)(v30 + 32);
      if ( v31 <= 0x40 ? *(_QWORD *)(v30 + 24) == 0 : v31 == (unsigned int)sub_16A57B0(v30 + 24) )
        return v12.m128i_i64[0];
    }
    LODWORD(v238) = v199;
    v241 = sub_1F6DB40;
    v240 = (__int64 (__fastcall *)(unsigned __int64 *, unsigned __int64 *, int))sub_1F6C0C0;
    v48 = sub_1D169E0(v15.m128i_i64[0], (_QWORD *)v15.m128i_i64[1], (__int64)&v238, v29);
    v50 = v48;
    if ( v240 )
    {
      v186 = v48;
      v240(&v238, &v238, 3);
      v50 = v186;
    }
    if ( v50 )
    {
      v59 = *(__int64 **)a1;
      v238 = 0;
      LODWORD(v239) = 0;
      v60 = sub_1D2B300(v59, 0x30u, (__int64)&v238, v228, (__int64)v229, v49);
      if ( v238 )
        sub_161E7C0((__int64)&v238, v238);
      return (__int64)v60;
    }
    if ( v26 )
    {
      v51 = *(_QWORD *)(v26 + 88);
      if ( *(_DWORD *)(v51 + 32) <= 0x40u )
      {
        if ( *(_QWORD *)(v51 + 24) )
          goto LABEL_47;
      }
      else
      {
        v190 = *(_DWORD *)(v51 + 32);
        if ( v190 != (unsigned int)sub_16A57B0(v51 + 24) )
          goto LABEL_47;
      }
      return v12.m128i_i64[0];
    }
LABEL_47:
    if ( *(_WORD *)(v13 + 24) == 48 )
      goto LABEL_48;
    result = (__int64)sub_1F77C50((__int64 **)a1, a2, *(double *)v12.m128i_i64, *(double *)v15.m128i_i64, a5);
    if ( result )
      return result;
    v54 = *(__int64 **)a1;
    LODWORD(v239) = v199;
    if ( v199 > 0x40 )
    {
      v187 = v54;
      sub_16A4EF0((__int64)&v238, -1, 1);
      v54 = v187;
    }
    else
    {
      v238 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v199;
    }
    v191 = sub_1D1F940((__int64)v54, a2, 0, (__int64)&v238, 0);
    if ( (unsigned int)v239 > 0x40 && v238 )
      j_j___libc_free_0_0(v238);
    if ( v191 )
    {
LABEL_48:
      v52 = *(_QWORD *)(a2 + 72);
      v53 = *(__int64 **)a1;
      v238 = v52;
      if ( v52 )
        sub_1623A60((__int64)&v238, v52, 2);
      LODWORD(v239) = *(_DWORD *)(a2 + 64);
      result = sub_1D38BB0(
                 (__int64)v53,
                 0,
                 (__int64)&v238,
                 (unsigned int)v228,
                 v229,
                 0,
                 v12,
                 *(double *)v15.m128i_i64,
                 a5,
                 0);
      v35 = v238;
      if ( v238 )
        goto LABEL_19;
      return result;
    }
    if ( *(_WORD *)(v203 + 24) == 145 && *(_WORD *)(**(_QWORD **)(v203 + 32) + 24LL) == 118 )
    {
      *(_QWORD *)&v78 = sub_1F87630((__int64 **)a1, v203, *(double *)v12.m128i_i64, *(double *)v15.m128i_i64, a5);
      if ( (_QWORD)v78 )
      {
        v79 = *(__int64 **)a1;
        v238 = *(_QWORD *)(a2 + 72);
        if ( v238 )
        {
          v204 = v78;
          sub_1F6CA20((__int64 *)&v238);
          v78 = v204;
        }
        LODWORD(v239) = *(_DWORD *)(a2 + 64);
        v47 = sub_1D332F0(
                v79,
                122,
                (__int64)&v238,
                (unsigned int)v228,
                v229,
                0,
                *(double *)v12.m128i_i64,
                *(double *)v15.m128i_i64,
                a5,
                v12.m128i_i64[0],
                v12.m128i_u64[1],
                v78);
        goto LABEL_39;
      }
    }
    if ( v26 )
    {
      if ( (unsigned __int8)sub_1FB1D70(a1, a2, 0) )
        return a2;
      if ( *(_WORD *)(v13 + 24) != 122 )
        goto LABEL_108;
    }
    else if ( *(_WORD *)(v13 + 24) != 122 )
    {
      goto LABEL_75;
    }
    LODWORD(v238) = v199;
    v241 = sub_1F6EA80;
    v240 = (__int64 (__fastcall *)(unsigned __int64 *, unsigned __int64 *, int))sub_1F6C0F0;
    v179 = sub_1D16BF0(
             v15.m128i_i64[0],
             v15.m128i_u32[2],
             *(_QWORD *)(*(_QWORD *)(v13 + 32) + 40LL),
             *(_QWORD *)(*(_QWORD *)(v13 + 32) + 48LL),
             (__int64)&v238);
    sub_A17130((__int64)&v238);
    if ( v179 )
    {
      v61 = *(__int64 **)a1;
      v238 = *(_QWORD *)(a2 + 72);
      if ( v238 )
        sub_1F6CA20((__int64 *)&v238);
      LODWORD(v239) = *(_DWORD *)(a2 + 64);
      v209 = sub_1D38BB0(
               (__int64)v61,
               0,
               (__int64)&v238,
               (unsigned int)v228,
               v229,
               0,
               v12,
               *(double *)v15.m128i_i64,
               a5,
               0);
      sub_17CD270((__int64 *)&v238);
      return v209;
    }
    LODWORD(v238) = v199;
    v241 = sub_1F6E0A0;
    v240 = (__int64 (__fastcall *)(unsigned __int64 *, unsigned __int64 *, int))sub_1F6C120;
    v183 = sub_1D16BF0(
             v15.m128i_i64[0],
             v15.m128i_u32[2],
             *(_QWORD *)(*(_QWORD *)(v13 + 32) + 40LL),
             *(_QWORD *)(*(_QWORD *)(v13 + 32) + 48LL),
             (__int64)&v238);
    sub_A17130((__int64)&v238);
    if ( v183 )
    {
      v238 = *(_QWORD *)(a2 + 72);
      if ( v238 )
        sub_1F6CA20((__int64 *)&v238);
      v154 = *(__int64 **)a1;
      LODWORD(v239) = *(_DWORD *)(a2 + 64);
      v155 = (const void ***)(*(_QWORD *)(v203 + 40) + 16LL * v197);
      *(_QWORD *)&v156 = sub_1D332F0(
                           v154,
                           52,
                           (__int64)&v238,
                           *(unsigned __int8 *)v155,
                           v155[1],
                           0,
                           *(double *)v12.m128i_i64,
                           *(double *)v15.m128i_i64,
                           a5,
                           v15.m128i_i64[0],
                           v15.m128i_u64[1],
                           *(_OWORD *)(*(_QWORD *)(v13 + 32) + 40LL));
      v208 = sub_1D332F0(
               *(__int64 **)a1,
               122,
               (__int64)&v238,
               (unsigned int)v228,
               v229,
               0,
               *(double *)v12.m128i_i64,
               *(double *)v15.m128i_i64,
               a5,
               **(_QWORD **)(v13 + 32),
               *(_QWORD *)(*(_QWORD *)(v13 + 32) + 8LL),
               v156);
      goto LABEL_40;
    }
    if ( !v26 )
    {
LABEL_75:
      if ( *(_WORD *)(v13 + 24) == 123 )
      {
        v97 = *(_QWORD *)(v13 + 32);
        if ( v203 != *(_QWORD *)(v97 + 40) || *(_DWORD *)(v97 + 48) != v197 )
          goto LABEL_99;
        if ( (unsigned __int8)sub_1F70310(v15.m128i_i64[0], v15.m128i_u32[2], 1u) )
        {
          v238 = *(_QWORD *)(a2 + 72);
          if ( v238 )
            sub_1F6CA20((__int64 *)&v238);
          v98 = *(__int64 **)a1;
          LODWORD(v239) = *(_DWORD *)(a2 + 64);
          v99 = sub_1D389D0(
                  (__int64)v98,
                  (__int64)&v238,
                  (unsigned int)v228,
                  v229,
                  0,
                  0,
                  v12,
                  *(double *)v15.m128i_i64,
                  a5);
          *(_QWORD *)&v101 = sub_1D332F0(
                               *(__int64 **)a1,
                               122,
                               (__int64)&v238,
                               (unsigned int)v228,
                               v229,
                               0,
                               *(double *)v12.m128i_i64,
                               *(double *)v15.m128i_i64,
                               a5,
                               v99,
                               v100,
                               *(_OWORD *)&v15);
          v208 = sub_1D332F0(
                   *(__int64 **)a1,
                   118,
                   (__int64)&v238,
                   (unsigned int)v228,
                   v229,
                   0,
                   *(double *)v12.m128i_i64,
                   *(double *)v15.m128i_i64,
                   a5,
                   **(_QWORD **)(v13 + 32),
                   *(_QWORD *)(*(_QWORD *)(v13 + 32) + 8LL),
                   v101);
          goto LABEL_40;
        }
      }
      v62 = *(unsigned __int16 *)(v13 + 24);
      if ( (_WORD)v62 == 52 || v62 == 119 )
      {
        v63 = *(_QWORD *)(v13 + 48);
        if ( !v63 )
        {
          if ( *(_WORD *)(v13 + 24) == 54 )
          {
LABEL_80:
            if ( !v26 )
              return 0;
            v64 = *(unsigned __int16 *)(v13 + 24);
LABEL_82:
            v65 = v64 - 142;
            if ( (unsigned int)(v64 - 119) > 1 )
            {
              if ( v65 > 2 )
                goto LABEL_95;
              v66 = *(__int64 **)(v13 + 32);
              if ( (unsigned __int16)(*(_WORD *)(*v66 + 24) - 118) > 2u )
                goto LABEL_95;
              goto LABEL_84;
            }
            v66 = *(__int64 **)(v13 + 32);
            if ( v65 <= 2 )
            {
LABEL_84:
              v191 = 1;
              v188 = *v66;
LABEL_85:
              v67 = *(_QWORD *)(v188 + 32);
              v68 = *(_QWORD *)v67;
              v69 = *(_QWORD *)(v67 + 8);
              v70 = *(unsigned int *)(v67 + 8);
              v193 = *(_DWORD *)(v67 + 48);
              v71 = *(unsigned __int16 *)(*(_QWORD *)v67 + 24LL);
              v72 = *(_QWORD *)(v67 + 40);
              v180 = _mm_loadu_si128((const __m128i *)(v67 + 40));
              if ( v71 == 32 || v71 == 10 )
              {
                v72 = *(_QWORD *)v67;
                v68 = *(_QWORD *)(v67 + 40);
                v70 = v193;
                v193 = *(_DWORD *)(v67 + 8);
              }
              if ( v191 )
              {
                v73 = *(__int64 **)a1;
                v238 = *(_QWORD *)(a2 + 72);
                if ( v238 )
                {
                  v172 = v69;
                  v173 = v73;
                  v174 = v70;
                  sub_1F6CA20((__int64 *)&v238);
                  v69 = v172;
                  v73 = v173;
                  v70 = v174;
                }
                LODWORD(v239) = *(_DWORD *)(a2 + 64);
                *((_QWORD *)&v171 + 1) = v70 | v69 & 0xFFFFFFFF00000000LL;
                *(_QWORD *)&v171 = v68;
                v200 = sub_1D309E0(
                         v73,
                         *(unsigned __int16 *)(v13 + 24),
                         (__int64)&v238,
                         (unsigned int)v228,
                         v229,
                         0,
                         *(double *)v12.m128i_i64,
                         *(double *)v15.m128i_i64,
                         *(double *)a5.m128i_i64,
                         v171);
                v202 = v74;
                sub_17CD270((__int64 *)&v238);
                v75 = *(__int64 **)a1;
                v238 = *(_QWORD *)(a2 + 72);
                if ( v238 )
                {
                  v176 = v75;
                  sub_1F6CA20((__int64 *)&v238);
                  v75 = v176;
                }
                LODWORD(v239) = *(_DWORD *)(a2 + 64);
                v210 = sub_1D309E0(
                         v75,
                         *(unsigned __int16 *)(v13 + 24),
                         (__int64)&v238,
                         (unsigned int)v228,
                         v229,
                         0,
                         *(double *)v12.m128i_i64,
                         *(double *)v15.m128i_i64,
                         *(double *)a5.m128i_i64,
                         __PAIR128__(v193 | v180.m128i_i64[1] & 0xFFFFFFFF00000000LL, v72));
                v217 = v76;
                sub_17CD270((__int64 *)&v238);
              }
              else
              {
                v200 = v68;
                v210 = v72;
                v202 = (unsigned int)v70;
                v217 = v193;
              }
              v77 = *(unsigned __int16 *)(v72 + 24);
              if ( v77 == 10 || v77 == 32 )
              {
                v102 = *(__int64 **)a1;
                v238 = *(_QWORD *)(a2 + 72);
                if ( v238 )
                  sub_1F6CA20((__int64 *)&v238);
                LODWORD(v239) = *(_DWORD *)(a2 + 64);
                *(_QWORD *)&v219 = v203;
                *((_QWORD *)&v219 + 1) = v197 | v15.m128i_i64[1] & 0xFFFFFFFF00000000LL;
                *(_QWORD *)&v103 = sub_1D332F0(
                                     v102,
                                     122,
                                     (__int64)&v238,
                                     (unsigned int)v228,
                                     v229,
                                     0,
                                     *(double *)v12.m128i_i64,
                                     *(double *)v15.m128i_i64,
                                     a5,
                                     v210,
                                     v217,
                                     v219);
                v104 = *(__int64 **)a1;
                v212 = v103;
                v236 = *(_QWORD *)(a2 + 72);
                if ( v236 )
                  sub_1F6CA20((__int64 *)&v236);
                v237 = *(_DWORD *)(a2 + 64);
                *(_QWORD *)&v105 = sub_1D332F0(
                                     v104,
                                     122,
                                     (__int64)&v236,
                                     (unsigned int)v228,
                                     v229,
                                     0,
                                     *(double *)v12.m128i_i64,
                                     *(double *)v15.m128i_i64,
                                     a5,
                                     v200,
                                     v202,
                                     v219);
                v234 = *(_QWORD *)(a2 + 72);
                if ( v234 )
                {
                  v220 = v105;
                  sub_1F6CA20(&v234);
                  v105 = v220;
                }
                v235 = *(_DWORD *)(a2 + 64);
                v208 = sub_1D332F0(
                         v102,
                         *(unsigned __int16 *)(v188 + 24),
                         (__int64)&v234,
                         (unsigned int)v228,
                         v229,
                         0,
                         *(double *)v12.m128i_i64,
                         *(double *)v15.m128i_i64,
                         a5,
                         v105,
                         *((unsigned __int64 *)&v105 + 1),
                         v212);
                sub_17CD270(&v234);
                sub_17CD270((__int64 *)&v236);
                goto LABEL_40;
              }
LABEL_95:
              if ( (*(_BYTE *)(v26 + 26) & 8) == 0 )
              {
                result = (__int64)sub_1F77880(
                                    (__int64 **)a1,
                                    a2,
                                    *(double *)v12.m128i_i64,
                                    *(double *)v15.m128i_i64,
                                    a5);
                if ( result )
                  return result;
              }
              return 0;
            }
LABEL_101:
            v188 = v13;
            goto LABEL_85;
          }
LABEL_99:
          if ( !v26 )
            return 0;
          v64 = *(unsigned __int16 *)(v13 + 24);
          if ( v64 == 118 )
            goto LABEL_101;
          goto LABEL_82;
        }
        if ( *(_QWORD *)(v63 + 32) )
        {
          if ( *(_WORD *)(v13 + 24) == 54 )
            goto LABEL_80;
          goto LABEL_99;
        }
        if ( (unsigned __int8)sub_1F70310(v15.m128i_i64[0], v15.m128i_u32[2], 1u)
          && (unsigned __int8)sub_1F70310(
                                *(_QWORD *)(*(_QWORD *)(v13 + 32) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(v13 + 32) + 48LL),
                                1u) )
        {
          v158 = *(_QWORD *)(v13 + 32);
          v159 = *(__int64 **)a1;
          sub_1F80610((__int64)&v238, v12.m128i_i64[0]);
          *(_QWORD *)&v160 = sub_1D332F0(
                               v159,
                               122,
                               (__int64)&v238,
                               (unsigned int)v228,
                               v229,
                               0,
                               *(double *)v12.m128i_i64,
                               *(double *)v15.m128i_i64,
                               a5,
                               *(_QWORD *)v158,
                               *(_QWORD *)(v158 + 8),
                               *(_OWORD *)&v15);
          v216 = v160;
          sub_17CD270((__int64 *)&v238);
          v161 = *(_QWORD *)(v13 + 32);
          v162 = *(__int64 **)a1;
          sub_1F80610((__int64)&v238, v15.m128i_i64[0]);
          *(_QWORD *)&v163 = sub_1D332F0(
                               v162,
                               122,
                               (__int64)&v238,
                               (unsigned int)v228,
                               v229,
                               0,
                               *(double *)v12.m128i_i64,
                               *(double *)v15.m128i_i64,
                               a5,
                               *(_QWORD *)(v161 + 40),
                               *(_QWORD *)(v161 + 48),
                               *(_OWORD *)&v15);
          v226 = v163;
          sub_17CD270((__int64 *)&v238);
          sub_1F81BC0(a1, v216);
          sub_1F81BC0(a1, v226);
          v164 = *(__int64 **)a1;
          v238 = *(_QWORD *)(a2 + 72);
          if ( v238 )
            sub_1F6CA20((__int64 *)&v238);
          LODWORD(v239) = *(_DWORD *)(a2 + 64);
          v47 = sub_1D332F0(
                  v164,
                  *(unsigned __int16 *)(v13 + 24),
                  (__int64)&v238,
                  (unsigned int)v228,
                  v229,
                  0,
                  *(double *)v12.m128i_i64,
                  *(double *)v15.m128i_i64,
                  a5,
                  v216,
                  *((unsigned __int64 *)&v216 + 1),
                  v226);
          goto LABEL_39;
        }
      }
      if ( *(_WORD *)(v13 + 24) == 54 )
      {
        v136 = *(_QWORD *)(v13 + 48);
        if ( !v136 || *(_QWORD *)(v136 + 32) )
          goto LABEL_80;
        if ( (unsigned __int8)sub_1F70310(v15.m128i_i64[0], v15.m128i_u32[2], 1u) )
        {
          if ( (unsigned __int8)sub_1F70310(
                                  *(_QWORD *)(*(_QWORD *)(v13 + 32) + 40LL),
                                  *(_QWORD *)(*(_QWORD *)(v13 + 32) + 48LL),
                                  1u) )
          {
            v196 = *(__int64 **)a1;
            v201 = *(_QWORD *)(v13 + 32);
            sub_1F80610((__int64)&v238, v15.m128i_i64[0]);
            *(_QWORD *)&v137 = sub_1D332F0(
                                 v196,
                                 122,
                                 (__int64)&v238,
                                 (unsigned int)v228,
                                 v229,
                                 0,
                                 *(double *)v12.m128i_i64,
                                 *(double *)v15.m128i_i64,
                                 a5,
                                 *(_QWORD *)(v201 + 40),
                                 *(_QWORD *)(v201 + 48),
                                 *(_OWORD *)&v15);
            v214 = v137;
            sub_17CD270((__int64 *)&v238);
            if ( (unsigned __int8)sub_1F70310(v214, DWORD2(v214), 0) )
            {
              v138 = *(_QWORD *)(v13 + 32);
              v139 = *(__int64 **)a1;
              v238 = *(_QWORD *)(a2 + 72);
              if ( v238 )
                sub_1F6CA20((__int64 *)&v238);
              LODWORD(v239) = *(_DWORD *)(a2 + 64);
              v208 = sub_1D332F0(
                       v139,
                       54,
                       (__int64)&v238,
                       (unsigned int)v228,
                       v229,
                       0,
                       *(double *)v12.m128i_i64,
                       *(double *)v15.m128i_i64,
                       a5,
                       *(_QWORD *)v138,
                       *(_QWORD *)(v138 + 8),
                       v214);
              goto LABEL_40;
            }
          }
        }
      }
      goto LABEL_99;
    }
LABEL_108:
    if ( (unsigned __int16)(*(_WORD *)(v13 + 24) - 142) <= 2u )
    {
      v80 = *(_DWORD **)(v13 + 32);
      v56 = *(_QWORD *)v80;
      v177 = *(_QWORD *)v80;
      if ( *(_WORD *)(*(_QWORD *)v80 + 24LL) == 122 )
      {
        v175 = v80[2];
        v140 = sub_1D1ADA0(
                 *(_QWORD *)(*(_QWORD *)(v56 + 32) + 40LL),
                 *(_QWORD *)(*(_QWORD *)(v56 + 32) + 48LL),
                 v55,
                 v56,
                 v57,
                 v58);
        if ( v140 )
        {
          sub_13A38D0((__int64)v230, *(_QWORD *)(v140 + 88) + 24LL);
          sub_13A38D0((__int64)v231, *(_QWORD *)(v26 + 88) + 24LL);
          sub_1F6DAA0((__int64)v230, (__int64)v231, 1);
          v141 = (unsigned __int8 *)(*(_QWORD *)(v177 + 40) + 16LL * v175);
          v142 = *v141;
          v143 = *((_QWORD *)v141 + 1);
          v232[0] = v142;
          v233 = v143;
          v146 = sub_1D159C0((__int64)v232, (__int64)v231, v142, v177, v144, v145);
          if ( !sub_13D0480((__int64)v231, v199 - (unsigned __int64)v146) )
          {
            sub_1F80610((__int64)&v234, v12.m128i_i64[0]);
            sub_13A38D0((__int64)&v238, (__int64)v230);
            sub_16A7200((__int64)&v238, v231);
            v147 = v239;
            v236 = v238;
            v223 = (__int64 *)v238;
            v237 = v239;
            v148 = sub_13D0480((__int64)&v236, v199);
            v149 = (__int64)v223;
            if ( v148 )
            {
              if ( v147 > 0x40 )
                v149 = *v223;
              v215 = *(__int64 **)a1;
              *(_QWORD *)&v150 = sub_1D38BB0(
                                   *(_QWORD *)a1,
                                   v149,
                                   (__int64)&v234,
                                   *(unsigned __int8 *)(*(_QWORD *)(v203 + 40) + 16LL * v197),
                                   *(const void ***)(*(_QWORD *)(v203 + 40) + 16LL * v197 + 8),
                                   0,
                                   v12,
                                   *(double *)v15.m128i_i64,
                                   a5,
                                   0);
              v224 = v150;
              v151 = sub_1D309E0(
                       *(__int64 **)a1,
                       *(unsigned __int16 *)(v13 + 24),
                       (__int64)&v234,
                       (unsigned int)v228,
                       v229,
                       0,
                       *(double *)v12.m128i_i64,
                       *(double *)v15.m128i_i64,
                       *(double *)a5.m128i_i64,
                       *(_OWORD *)*(_QWORD *)(v177 + 32));
              v153 = (__int64)sub_1D332F0(
                                v215,
                                122,
                                (__int64)&v234,
                                (unsigned int)v228,
                                v229,
                                0,
                                *(double *)v12.m128i_i64,
                                *(double *)v15.m128i_i64,
                                a5,
                                v151,
                                v152,
                                v224);
            }
            else
            {
              v153 = sub_1D38BB0(
                       *(_QWORD *)a1,
                       0,
                       (__int64)&v234,
                       (unsigned int)v228,
                       v229,
                       0,
                       v12,
                       *(double *)v15.m128i_i64,
                       a5,
                       0);
            }
            v225 = v153;
            sub_135E100((__int64 *)&v236);
            sub_17CD270(&v234);
            sub_135E100(v231);
            sub_135E100(v230);
            return v225;
          }
          sub_135E100(v231);
          sub_135E100(v230);
        }
      }
    }
    if ( *(_WORD *)(v13 + 24) == 143 && sub_1D18C00(v13, 1, v192) )
    {
      v106 = **(_QWORD **)(v13 + 32);
      if ( *(_WORD *)(v106 + 24) == 124 )
      {
        v107 = *(_QWORD *)(v106 + 32);
        v108 = *(_QWORD *)(v107 + 48);
        v109 = sub_1D1ADA0(*(_QWORD *)(v107 + 40), v108, v55, v56, v57, v58);
        v55 = v109;
        if ( v109 )
        {
          v182 = v109;
          v189 = *(_QWORD *)(v109 + 88) + 24LL;
          v110 = sub_1D159C0((__int64)&v228, v108, v109, v56, v189, v58);
          if ( sub_13D0480(v189, v110) )
          {
            v111 = *(_QWORD *)(v182 + 88);
            v112 = *(_QWORD **)(v111 + 24);
            if ( *(_DWORD *)(v111 + 32) > 0x40u )
              v112 = (_QWORD *)*v112;
            v55 = *(_QWORD *)(v26 + 88);
            v113 = *(_QWORD **)(v55 + 24);
            if ( *(_DWORD *)(v55 + 32) > 0x40u )
              v113 = (_QWORD *)*v113;
            if ( v113 == v112 )
            {
              v114 = *(const __m128i **)(v13 + 32);
              v115 = v114->m128i_i64[0];
              v205 = _mm_loadu_si128(v114);
              v221 = v114->m128i_u32[2];
              v116 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v115 + 32) + 40LL) + 40LL)
                                       + 16LL * *(unsigned int *)(*(_QWORD *)(v115 + 32) + 48LL));
              v117 = (const void **)*((_QWORD *)v116 + 1);
              v118 = *v116;
              v236 = *(_QWORD *)(a2 + 72);
              if ( v236 )
              {
                v195 = v118;
                v198 = v117;
                sub_1F6CA20((__int64 *)&v236);
                v118 = v195;
                v117 = v198;
              }
              v119 = *(__int64 **)a1;
              v237 = *(_DWORD *)(a2 + 64);
              v120 = sub_1D38BB0(
                       (__int64)v119,
                       (__int64)v113,
                       (__int64)&v236,
                       v118,
                       v117,
                       0,
                       v12,
                       *(double *)v15.m128i_i64,
                       a5,
                       0);
              v122 = v121;
              v123 = (const void ***)(*(_QWORD *)(v115 + 40) + 16LL * v221);
              *((_QWORD *)&v170 + 1) = v122;
              *(_QWORD *)&v170 = v120;
              *(_QWORD *)&v124 = sub_1D332F0(
                                   v119,
                                   122,
                                   (__int64)&v236,
                                   *(unsigned __int8 *)v123,
                                   v123[1],
                                   0,
                                   *(double *)v12.m128i_i64,
                                   *(double *)v15.m128i_i64,
                                   a5,
                                   v205.m128i_i64[0],
                                   v205.m128i_u64[1],
                                   v170);
              v206 = v124;
              sub_1F81BC0(a1, v124);
              v125 = *(__int64 **)a1;
              sub_1F80610((__int64)&v238, v12.m128i_i64[0]);
              v213 = sub_1D309E0(
                       v125,
                       143,
                       (__int64)&v238,
                       (unsigned int)v228,
                       v229,
                       0,
                       *(double *)v12.m128i_i64,
                       *(double *)v15.m128i_i64,
                       *(double *)a5.m128i_i64,
                       v206);
              sub_17CD270((__int64 *)&v238);
              sub_17CD270((__int64 *)&v236);
              return v213;
            }
          }
        }
      }
    }
    if ( (unsigned int)*(unsigned __int16 *)(v13 + 24) - 123 <= 1 && (*(_BYTE *)(v13 + 80) & 8) != 0 )
    {
      v126 = sub_1D1ADA0(
               *(_QWORD *)(*(_QWORD *)(v13 + 32) + 40LL),
               *(_QWORD *)(*(_QWORD *)(v13 + 32) + 48LL),
               v55,
               v56,
               v57,
               v58);
      if ( v126 )
      {
        v127 = *(_QWORD *)(v126 + 88);
        v128 = *(char **)(v127 + 24);
        if ( *(_DWORD *)(v127 + 32) > 0x40u )
          v128 = *(char **)v128;
        v129 = *(_QWORD *)(v26 + 88);
        v130 = *(char **)(v129 + 24);
        if ( *(_DWORD *)(v129 + 32) > 0x40u )
          v130 = *(char **)v130;
        v238 = *(_QWORD *)(a2 + 72);
        if ( v238 )
          sub_1F6CA20((__int64 *)&v238);
        v131 = *(__int64 **)a1;
        LODWORD(v239) = *(_DWORD *)(a2 + 64);
        v132 = *(const void ***)(*(_QWORD *)(v203 + 40) + 16LL * v197 + 8);
        v133 = *(unsigned __int8 *)(*(_QWORD *)(v203 + 40) + 16LL * v197);
        if ( v130 < v128 )
        {
          *(_QWORD *)&v157 = sub_1D38BB0(
                               (__int64)v131,
                               v128 - v130,
                               (__int64)&v238,
                               v133,
                               v132,
                               0,
                               v12,
                               *(double *)v15.m128i_i64,
                               a5,
                               0);
          v135 = sub_1D332F0(
                   v131,
                   *(unsigned __int16 *)(v13 + 24),
                   (__int64)&v238,
                   (unsigned int)v228,
                   v229,
                   0,
                   *(double *)v12.m128i_i64,
                   *(double *)v15.m128i_i64,
                   a5,
                   **(_QWORD **)(v13 + 32),
                   *(_QWORD *)(*(_QWORD *)(v13 + 32) + 8LL),
                   v157);
        }
        else
        {
          *(_QWORD *)&v134 = sub_1D38BB0(
                               (__int64)v131,
                               v130 - v128,
                               (__int64)&v238,
                               v133,
                               v132,
                               0,
                               v12,
                               *(double *)v15.m128i_i64,
                               a5,
                               0);
          v135 = sub_1D332F0(
                   v131,
                   122,
                   (__int64)&v238,
                   (unsigned int)v228,
                   v229,
                   0,
                   *(double *)v12.m128i_i64,
                   *(double *)v15.m128i_i64,
                   a5,
                   **(_QWORD **)(v13 + 32),
                   *(_QWORD *)(*(_QWORD *)(v13 + 32) + 8LL),
                   v134);
        }
        v222 = v135;
        sub_17CD270((__int64 *)&v238);
        return (__int64)v222;
      }
    }
    if ( *(_WORD *)(v13 + 24) == 124 && sub_1D18C00(v13, 1, v192) )
    {
      v85 = sub_1D1ADA0(
              *(_QWORD *)(*(_QWORD *)(v13 + 32) + 40LL),
              *(_QWORD *)(*(_QWORD *)(v13 + 32) + 48LL),
              v81,
              v82,
              v83,
              v84);
      if ( v85 )
      {
        v86 = *(_QWORD *)(v85 + 88);
        v87 = *(_QWORD **)(v86 + 24);
        if ( *(_DWORD *)(v86 + 32) > 0x40u )
          v87 = (_QWORD *)*v87;
        if ( v199 > (unsigned __int64)v87 )
        {
          v88 = *(_QWORD *)(v26 + 88);
          if ( *(_DWORD *)(v88 + 32) <= 0x40u )
            v194 = *(_QWORD *)(v88 + 24);
          else
            v194 = **(_QWORD **)(v88 + 24);
          sub_171A350((__int64)&v236, v199, v199 - (_DWORD)v87);
          if ( (unsigned __int64)v87 >= v194 )
          {
            sub_17A2760((__int64)&v236, (_DWORD)v87 - v194);
            v238 = *(_QWORD *)(a2 + 72);
            if ( v238 )
              sub_1F6CA20((__int64 *)&v238);
            v165 = *(_DWORD *)(a2 + 64);
            v166 = *(__int64 **)a1;
            LODWORD(v239) = v165;
            v167 = (const void ***)(*(_QWORD *)(v203 + 40) + 16LL * v197);
            *(_QWORD *)&v168 = sub_1D38BB0(
                                 (__int64)v166,
                                 (__int64)v87 - v194,
                                 (__int64)&v238,
                                 *(unsigned __int8 *)v167,
                                 v167[1],
                                 0,
                                 v12,
                                 *(double *)v15.m128i_i64,
                                 a5,
                                 0);
            v169 = sub_1D332F0(
                     v166,
                     124,
                     (__int64)&v238,
                     (unsigned int)v228,
                     v229,
                     0,
                     *(double *)v12.m128i_i64,
                     *(double *)v15.m128i_i64,
                     a5,
                     **(_QWORD **)(v13 + 32),
                     *(_QWORD *)(*(_QWORD *)(v13 + 32) + 8LL),
                     v168);
            v93 = (unsigned int)v93;
            v218 = v169;
          }
          else
          {
            sub_1F6D260((__int64 *)&v236, v194 - (_DWORD)v87);
            v238 = *(_QWORD *)(a2 + 72);
            if ( v238 )
              sub_1F6CA20((__int64 *)&v238);
            v89 = *(_DWORD *)(a2 + 64);
            v90 = *(__int64 **)a1;
            LODWORD(v239) = v89;
            v91 = (const void ***)(*(_QWORD *)(v203 + 40) + 16LL * v197);
            *(_QWORD *)&v92 = sub_1D38BB0(
                                (__int64)v90,
                                v194 - (_QWORD)v87,
                                (__int64)&v238,
                                *(unsigned __int8 *)v91,
                                v91[1],
                                0,
                                v12,
                                *(double *)v15.m128i_i64,
                                a5,
                                0);
            v218 = sub_1D332F0(
                     v90,
                     122,
                     (__int64)&v238,
                     (unsigned int)v228,
                     v229,
                     0,
                     *(double *)v12.m128i_i64,
                     *(double *)v15.m128i_i64,
                     a5,
                     **(_QWORD **)(v13 + 32),
                     *(_QWORD *)(*(_QWORD *)(v13 + 32) + 8LL),
                     v92);
            v93 = (unsigned int)v93;
          }
          v227 = v93;
          sub_17CD270((__int64 *)&v238);
          sub_1F80610((__int64)&v238, v12.m128i_i64[0]);
          v94 = *(__int64 **)a1;
          *(_QWORD *)&v95 = sub_1D38970(
                              *(_QWORD *)a1,
                              (__int64)&v236,
                              (__int64)&v238,
                              v228,
                              v229,
                              0,
                              v12,
                              *(double *)v15.m128i_i64,
                              a5,
                              0);
          v211 = sub_1D332F0(
                   v94,
                   118,
                   (__int64)&v238,
                   (unsigned int)v228,
                   v229,
                   0,
                   *(double *)v12.m128i_i64,
                   *(double *)v15.m128i_i64,
                   a5,
                   (__int64)v218,
                   v227,
                   v95);
          sub_17CD270((__int64 *)&v238);
          sub_135E100((__int64 *)&v236);
          return (__int64)v211;
        }
      }
    }
    goto LABEL_75;
  }
  v33 = *(_QWORD *)(a2 + 72);
  v34 = *(__int64 **)a1;
  v238 = v33;
  if ( v33 )
    sub_1623A60((__int64)&v238, v33, 2);
  LODWORD(v239) = *(_DWORD *)(a2 + 64);
  result = sub_1D392A0((__int64)v34, 122, (__int64)&v238, v228, v229, v13, v12, *(double *)v15.m128i_i64, a5, v26);
  v35 = v238;
  if ( v238 )
  {
LABEL_19:
    v207 = result;
    sub_161E7C0((__int64)&v238, v35);
    return v207;
  }
  return result;
}
