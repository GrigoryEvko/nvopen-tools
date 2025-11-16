// Function: sub_1735560
// Address: 0x1735560
//
__int64 __fastcall sub_1735560(
        __m128i *a1,
        __int64 a2,
        double a3,
        double a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // r12
  __m128 v12; // xmm0
  __m128i v13; // xmm1
  __int64 v14; // rsi
  __int64 v15; // rdi
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rbx
  __int64 v19; // r13
  __int64 v20; // r14
  _QWORD *v21; // rax
  double v22; // xmm4_8
  double v23; // xmm5_8
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rbx
  unsigned __int64 v29; // r14
  __int64 v30; // rax
  __int64 v31; // rax
  char v32; // al
  __int64 v33; // r8
  char v34; // al
  __int64 v35; // rdx
  __int64 v36; // rcx
  double v37; // xmm4_8
  double v38; // xmm5_8
  __int64 v39; // rsi
  __int64 *v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rdx
  unsigned __int8 *v43; // rax
  __int64 *v44; // rax
  int v45; // eax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r14
  __int64 v51; // rbx
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // rdi
  unsigned __int8 *v57; // rax
  __int64 v58; // rax
  __int64 v59; // rbx
  __int64 v60; // rbx
  __int32 v61; // eax
  char v62; // r10
  char v63; // r10
  char v64; // al
  __int32 v65; // eax
  __int64 v66; // r12
  int v67; // eax
  __int64 v68; // rax
  unsigned __int8 *v69; // rsi
  __int64 v70; // rdx
  int v71; // eax
  __int64 v72; // rdi
  __int64 v73; // rdx
  __int64 *v74; // r12
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 *v79; // r10
  char v80; // al
  char v81; // al
  __int64 v82; // rax
  unsigned __int8 v83; // dl
  unsigned __int8 *v84; // rsi
  unsigned __int64 v85; // rax
  char v86; // al
  __int64 v87; // rsi
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 v90; // rax
  unsigned __int8 v91; // dl
  unsigned __int8 v92; // al
  __int64 *v93; // rbx
  __int64 v94; // rbx
  __int64 v95; // r15
  unsigned __int8 v96; // al
  unsigned int v97; // edx
  __int64 *v98; // rax
  __int64 v99; // rax
  unsigned int v100; // edi
  __int64 *v101; // rdx
  __int64 v102; // rdx
  unsigned int v103; // ebx
  __int64 *v104; // r15
  _QWORD *v105; // rax
  __int64 v106; // r13
  _QWORD *v107; // rax
  __int64 v108; // rax
  __int64 v109; // r12
  __int64 v110; // rax
  int v111; // eax
  int v112; // eax
  __int32 v113; // eax
  char v114; // bl
  __int64 v115; // rax
  __int64 v116; // rax
  unsigned __int8 *v117; // rdx
  double v118; // xmm4_8
  double v119; // xmm5_8
  unsigned __int8 *v120; // rsi
  __int64 v121; // rdi
  unsigned __int8 *v122; // rdx
  unsigned __int64 v123; // rax
  __int64 v124; // rdx
  __int64 v125; // rdi
  __int64 *v126; // r12
  __int64 **v127; // rdi
  _QWORD *v128; // rax
  __int64 v129; // rdi
  unsigned __int8 *v130; // rax
  __int64 v131; // rdi
  unsigned __int8 *v132; // r12
  unsigned __int8 *v133; // rax
  __int64 v134; // rcx
  __int64 v135; // r12
  const char *v136; // rax
  unsigned __int64 *v137; // rdx
  __int64 *v138; // rax
  __int64 **v139; // rdi
  _QWORD *v140; // rax
  char v141; // r10
  char v142; // r10
  char v143; // al
  __int32 v144; // eax
  __int64 v145; // r12
  int v146; // eax
  __int64 v147; // rax
  int v148; // eax
  const char *v149; // rax
  unsigned __int64 *v150; // rdx
  const char *v151; // rax
  unsigned __int64 *v152; // rdx
  __int64 v153; // rbx
  unsigned __int8 v154; // al
  __int64 v155; // rdi
  unsigned __int8 *v156; // r12
  __int64 *v157; // rax
  __int64 v158; // rdi
  __int64 *v159; // rax
  __int64 v160; // rbx
  unsigned __int8 v161; // al
  __int64 v162; // rdi
  unsigned __int8 *v163; // r12
  __int64 *v164; // rax
  __int64 v165; // rdi
  __int64 *v166; // rax
  __int64 v167; // rdi
  unsigned __int8 *v168; // rcx
  unsigned __int8 *v169; // rax
  unsigned __int8 *v170; // rax
  char v171; // cl
  __int64 v172; // rcx
  __int64 v173; // r8
  int v174; // eax
  __int32 v175; // eax
  __int64 v176; // r12
  __int64 v177; // rax
  unsigned __int8 *v178; // rax
  __int64 v179; // r12
  int v180; // eax
  __int64 v181; // rax
  __int64 v182; // rdi
  unsigned __int8 *v183; // rcx
  unsigned __int8 *v184; // rax
  unsigned __int8 *v185; // rax
  char v186; // [rsp+0h] [rbp-180h]
  char v187; // [rsp+0h] [rbp-180h]
  char v188; // [rsp+8h] [rbp-178h]
  char v189; // [rsp+8h] [rbp-178h]
  char v190; // [rsp+8h] [rbp-178h]
  __int64 v191; // [rsp+8h] [rbp-178h]
  char v192; // [rsp+8h] [rbp-178h]
  char v193; // [rsp+10h] [rbp-170h]
  char v194; // [rsp+10h] [rbp-170h]
  __int64 v195; // [rsp+10h] [rbp-170h]
  __int64 v196; // [rsp+18h] [rbp-168h]
  __int64 v197; // [rsp+20h] [rbp-160h]
  unsigned __int8 *v198; // [rsp+28h] [rbp-158h]
  unsigned __int64 v199; // [rsp+28h] [rbp-158h]
  bool v200; // [rsp+30h] [rbp-150h]
  unsigned __int64 v201; // [rsp+30h] [rbp-150h]
  __int64 v202; // [rsp+30h] [rbp-150h]
  char v203; // [rsp+48h] [rbp-138h]
  __int64 *v204; // [rsp+48h] [rbp-138h]
  __m128 v205; // [rsp+58h] [rbp-128h]
  char v206; // [rsp+58h] [rbp-128h]
  __int64 v207; // [rsp+68h] [rbp-118h]
  unsigned __int64 v208; // [rsp+68h] [rbp-118h]
  unsigned __int8 *v209; // [rsp+78h] [rbp-108h] BYREF
  unsigned __int64 v210; // [rsp+80h] [rbp-100h] BYREF
  unsigned __int64 v211; // [rsp+88h] [rbp-F8h] BYREF
  unsigned __int64 v212; // [rsp+90h] [rbp-F0h] BYREF
  unsigned __int8 *v213; // [rsp+98h] [rbp-E8h] BYREF
  unsigned __int8 *v214; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 *v215; // [rsp+A8h] [rbp-D8h] BYREF
  _QWORD *v216[2]; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 *v217; // [rsp+C0h] [rbp-C0h] BYREF
  int v218; // [rsp+C8h] [rbp-B8h]
  __int64 *v219; // [rsp+D0h] [rbp-B0h] BYREF
  unsigned __int8 **v220; // [rsp+D8h] [rbp-A8h]
  unsigned __int8 *v221; // [rsp+E0h] [rbp-A0h] BYREF
  int v222; // [rsp+E8h] [rbp-98h]
  unsigned __int8 *v223; // [rsp+F0h] [rbp-90h] BYREF
  int v224; // [rsp+F8h] [rbp-88h]
  unsigned __int8 **v225; // [rsp+100h] [rbp-80h] BYREF
  unsigned __int64 *v226; // [rsp+108h] [rbp-78h]
  unsigned __int64 *v227; // [rsp+110h] [rbp-70h]
  __m128 v228; // [rsp+120h] [rbp-60h] BYREF
  __m128i v229; // [rsp+130h] [rbp-50h] BYREF
  __int64 v230; // [rsp+140h] [rbp-40h]

  v11 = a2;
  v230 = a2;
  v12 = (__m128)_mm_loadu_si128(a1 + 167);
  v13 = _mm_loadu_si128(a1 + 168);
  v14 = *(_QWORD *)(a2 - 24);
  v15 = *(_QWORD *)(v11 - 48);
  v228 = v12;
  v229 = v13;
  v16 = sub_13E1270(v15, v14, &v228);
  if ( v16 )
  {
    v18 = *(_QWORD *)(v11 + 8);
    if ( v18 )
    {
      v19 = a1->m128i_i64[0];
      v20 = (__int64)v16;
      do
      {
        v21 = sub_1648700(v18);
        sub_170B990(v19, (__int64)v21);
        v18 = *(_QWORD *)(v18 + 8);
      }
      while ( v18 );
      if ( v11 == v20 )
        v20 = sub_1599EF0(*(__int64 ***)v11);
      sub_164D160(v11, v20, v12, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64, a6, v22, v23, a9, a10);
      return v11;
    }
    return 0;
  }
  if ( (unsigned __int8)sub_170D400(a1, v11, v17, (__m128i)v12, *(double *)v13.m128i_i64, a5) )
    return v11;
  v25 = (__int64)sub_1707490(
                   (__int64)a1,
                   (unsigned __int8 *)v11,
                   *(double *)v12.m128_u64,
                   *(double *)v13.m128i_i64,
                   *(double *)a5.m128i_i64);
  if ( v25 )
    return v25;
  if ( (unsigned __int8)sub_17AD890(a1, v11) )
    return v11;
  v28 = *(_QWORD *)(v11 - 48);
  v29 = *(_QWORD *)(v11 - 24);
  v30 = *(_QWORD *)(v28 + 8);
  if ( !v30 || *(_QWORD *)(v30 + 8) )
  {
    v31 = *(_QWORD *)(v29 + 8);
    if ( !v31 || *(_QWORD *)(v31 + 8) )
      goto LABEL_22;
  }
  v32 = *(_BYTE *)(v28 + 16);
  v33 = a1->m128i_i64[1];
  if ( v32 == 50 )
  {
    v26 = *(_QWORD *)(v28 - 48);
    if ( v26 )
    {
      v221 = *(unsigned __int8 **)(v28 - 48);
      v43 = *(unsigned __int8 **)(v28 - 24);
      if ( v43 )
        goto LABEL_38;
    }
LABEL_22:
    v228.m128_u64[0] = (unsigned __int64)&v221;
    v228.m128_u64[1] = (unsigned __int64)&v223;
    v34 = *(_BYTE *)(v28 + 16);
    if ( v34 == 50 )
    {
      if ( !*(_QWORD *)(v28 - 48)
        || (v221 = *(unsigned __int8 **)(v28 - 48),
            !sub_171DA10((_QWORD **)&v228.m128_u64[1], *(_QWORD *)(v28 - 24), v26, v27)) )
      {
        v41 = *(_QWORD *)(v28 - 24);
        if ( !v41 )
          goto LABEL_25;
        v42 = v228.m128_u64[0];
        *(_QWORD *)v228.m128_u64[0] = v41;
        if ( !sub_171DA10((_QWORD **)&v228.m128_u64[1], *(_QWORD *)(v28 - 48), v42, v27) )
          goto LABEL_25;
      }
      goto LABEL_32;
    }
    if ( v34 != 5 || *(_WORD *)(v28 + 18) != 26 )
      goto LABEL_25;
    v45 = *(_DWORD *)(v28 + 20);
    if ( *(_QWORD *)(v28 - 24LL * (v45 & 0xFFFFFFF)) )
    {
      v221 = *(unsigned __int8 **)(v28 - 24LL * (*(_DWORD *)(v28 + 20) & 0xFFFFFFF));
      if ( sub_14B2B20((_QWORD **)&v228.m128_u64[1], *(_QWORD *)(v28 + 24 * (1LL - (v45 & 0xFFFFFFF)))) )
      {
LABEL_32:
        v225 = (unsigned __int8 **)v221;
        v227 = (unsigned __int64 *)v223;
        if ( !sub_13D7CA0((__int64 *)&v225, v29) )
          goto LABEL_25;
        v229.m128i_i16[0] = 257;
        v25 = sub_15FB440(28, (__int64 *)v221, (__int64)v223, (__int64)&v228, 0);
        goto LABEL_34;
      }
      v45 = *(_DWORD *)(v28 + 20);
    }
    v46 = *(_QWORD *)(v28 + 24 * (1LL - (v45 & 0xFFFFFFF)));
    if ( !v46 )
      goto LABEL_25;
    *(_QWORD *)v228.m128_u64[0] = v46;
    if ( !sub_14B2B20((_QWORD **)&v228.m128_u64[1], *(_QWORD *)(v28 - 24LL * (*(_DWORD *)(v28 + 20) & 0xFFFFFFF))) )
      goto LABEL_25;
    goto LABEL_32;
  }
  if ( v32 != 5 )
    goto LABEL_22;
  if ( *(_WORD *)(v28 + 18) != 26 )
    goto LABEL_22;
  v47 = *(_DWORD *)(v28 + 20) & 0xFFFFFFF;
  v27 = 4 * v47;
  v26 = *(_QWORD *)(v28 - 24 * v47);
  if ( !v26 )
    goto LABEL_22;
  v221 = *(unsigned __int8 **)(v28 - 24LL * (*(_DWORD *)(v28 + 20) & 0xFFFFFFF));
  v27 = 1 - v47;
  v43 = *(unsigned __int8 **)(v28 + 24 * (1 - v47));
  if ( !v43 )
    goto LABEL_22;
LABEL_38:
  v228.m128_u64[0] = v26;
  v207 = v33;
  v223 = v43;
  v228.m128_u64[1] = (unsigned __int64)v43;
  if ( !sub_1730BC0((__int64 *)&v228, v29) )
    goto LABEL_22;
  v229.m128i_i16[0] = 257;
  LOWORD(v227) = 257;
  v44 = (__int64 *)sub_172B670(
                     v207,
                     (__int64)v221,
                     (__int64)v223,
                     (__int64 *)&v225,
                     *(double *)v12.m128_u64,
                     *(double *)v13.m128i_i64,
                     *(double *)a5.m128i_i64);
  v25 = sub_15FB630(v44, (__int64)&v228, 0);
LABEL_34:
  if ( v25 )
    return v25;
LABEL_25:
  v35 = (__int64)sub_1708300(a1, (unsigned __int8 *)v11, (__m128i)v12, v13, a5);
  if ( !v35 )
  {
    v39 = v11;
    v40 = (__int64 *)a1;
    v35 = sub_172C850(
            v11,
            a1->m128i_i64[1],
            *(double *)v12.m128_u64,
            *(double *)v13.m128i_i64,
            *(double *)a5.m128i_i64,
            0,
            v36);
    if ( v35 )
      return sub_170E100(v40, v39, v35, v12, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64, a6, v37, v38, a9, a10);
    v25 = sub_1713A90(
            a1->m128i_i64,
            (_BYTE *)v11,
            v12,
            *(double *)v13.m128i_i64,
            *(double *)a5.m128i_i64,
            a6,
            v37,
            v38,
            a9,
            a10);
    if ( v25 )
      return v25;
    v25 = (__int64)sub_1734970(a1->m128i_i64, v11);
    if ( v25 )
      return v25;
    v50 = *(_QWORD *)(v11 - 48);
    v51 = *(_QWORD *)(v11 - 24);
    v205.m128_u64[0] = (unsigned __int64)&v223;
    v228.m128_u64[0] = (unsigned __int64)&v223;
    v208 = v51;
    v205.m128_u64[1] = (unsigned __int64)&v225;
    v228.m128_u64[1] = (unsigned __int64)&v225;
    if ( (unsigned __int8)sub_17310F0(&v228, v50, v48, v49)
      && (unsigned __int8)sub_17288A0(a1->m128i_i64, v51, (__int64)v225, 0, v11) )
    {
      v229.m128i_i16[0] = 257;
      v72 = a1->m128i_i64[1];
      v73 = v51;
    }
    else
    {
      v228 = v205;
      if ( !(unsigned __int8)sub_17310F0(&v228, v51, v52, v53)
        || !(unsigned __int8)sub_17288A0(a1->m128i_i64, v50, (__int64)v225, 0, v11) )
      {
        v211 = 0;
        v225 = &v209;
        v212 = 0;
        v226 = &v211;
        if ( !(unsigned __int8)sub_13D5EF0(&v225, v50) )
          goto LABEL_57;
        v228.m128_u64[0] = (unsigned __int64)&v210;
        v228.m128_u64[1] = (unsigned __int64)&v212;
        if ( !(unsigned __int8)sub_13D5EF0(&v228, v51) )
          goto LABEL_57;
        v197 = v211;
        if ( *(_BYTE *)(v211 + 16) != 13 )
          goto LABEL_151;
        v196 = v212;
        if ( *(_BYTE *)(v212 + 16) != 13 )
          goto LABEL_151;
        v60 = v211 + 24;
        v213 = 0;
        v204 = (__int64 *)(v212 + 24);
        v214 = 0;
        sub_13A38D0((__int64)&v225, v211 + 24);
        sub_1727280((__int64 *)&v225, v204);
        v61 = (int)v226;
        LODWORD(v226) = 0;
        v228.m128_i32[2] = v61;
        v228.m128_u64[0] = (unsigned __int64)v225;
        v200 = sub_13D01C0((__int64)&v228);
        sub_135E100((__int64 *)&v228);
        sub_135E100((__int64 *)&v225);
        if ( !v200 )
          goto LABEL_150;
        v219 = (__int64 *)&v213;
        v220 = &v214;
        v62 = sub_1731D30(&v219, (__int64)v209);
        if ( !v62 )
        {
LABEL_202:
          v219 = (__int64 *)&v213;
          v220 = &v214;
          v141 = sub_1731D30(&v219, v210);
          if ( !v141 )
            goto LABEL_147;
          if ( v213 == v209 )
          {
            v190 = v141;
            sub_13A38D0((__int64)&v221, (__int64)v204);
            sub_13D0570((__int64)&v221);
            v148 = v222;
            v222 = 0;
            v224 = v148;
            v223 = v221;
            v143 = sub_17288A0(a1->m128i_i64, (__int64)v214, (__int64)&v223, 0, v11);
            v142 = v143;
            if ( v143 )
              goto LABEL_207;
            v142 = v190;
          }
          else
          {
            v142 = 0;
          }
          v143 = 0;
          if ( v214 == v209 )
          {
            v189 = v142;
            sub_13A38D0((__int64)&v225, (__int64)v204);
            sub_13D0570((__int64)&v225);
            v144 = (int)v226;
            LODWORD(v226) = 0;
            v228.m128_i32[2] = v144;
            v228.m128_u64[0] = (unsigned __int64)v225;
            v193 = sub_17288A0(a1->m128i_i64, (__int64)v213, (__int64)&v228, 0, v11);
            sub_135E100((__int64 *)&v228);
            sub_135E100((__int64 *)&v225);
            v142 = v189;
            v143 = v193;
          }
LABEL_207:
          if ( v142 )
          {
            v194 = v143;
            sub_135E100((__int64 *)&v223);
            sub_135E100((__int64 *)&v221);
            v143 = v194;
          }
          if ( v143 )
          {
            v145 = a1->m128i_i64[1];
            v229.m128i_i16[0] = 257;
            sub_13A38D0((__int64)&v223, v60);
            sub_1727260((__int64 *)&v223, v204);
            v146 = v224;
            v224 = 0;
            LODWORD(v226) = v146;
            v225 = (unsigned __int8 **)v223;
            v147 = sub_159C0E0(*(__int64 **)(v145 + 24), (__int64)&v225);
            v69 = (unsigned __int8 *)v210;
            v70 = v147;
            goto LABEL_79;
          }
LABEL_147:
          v215 = 0;
          v216[0] = &v213;
          v216[1] = &v215;
          if ( (unsigned __int8)sub_1731DD0(v216, (__int64)v209) )
          {
            sub_13A38D0((__int64)&v217, v60);
            sub_13D0570((__int64)&v217);
            v111 = v218;
            v218 = 0;
            LODWORD(v220) = v111;
            v219 = v217;
            sub_1727280((__int64 *)&v219, v215 + 3);
            v112 = (int)v220;
            LODWORD(v220) = 0;
            v222 = v112;
            v221 = (unsigned __int8 *)v219;
            if ( !sub_13D01C0((__int64)&v221) )
            {
LABEL_149:
              sub_135E100((__int64 *)&v221);
              sub_135E100((__int64 *)&v219);
              sub_135E100((__int64 *)&v217);
              goto LABEL_150;
            }
            v171 = *(_BYTE *)(v210 + 16);
            if ( v171 == 51 )
            {
              if ( v213 != *(unsigned __int8 **)(v210 - 48) )
                goto LABEL_149;
              v173 = *(_QWORD *)(v210 - 24);
              if ( *(_BYTE *)(v173 + 16) != 13 )
                goto LABEL_149;
            }
            else
            {
              if ( v171 != 5 )
                goto LABEL_149;
              if ( *(_WORD *)(v210 + 18) != 27 )
                goto LABEL_149;
              v172 = *(_DWORD *)(v210 + 20) & 0xFFFFFFF;
              if ( v213 != *(unsigned __int8 **)(v210 - 24 * v172) )
                goto LABEL_149;
              v173 = *(_QWORD *)(v210 + 24 * (1 - v172));
              if ( *(_BYTE *)(v173 + 16) != 13 )
                goto LABEL_149;
            }
            v195 = v173;
            sub_13A38D0((__int64)&v223, (__int64)v204);
            sub_13D0570((__int64)&v223);
            v174 = v224;
            v224 = 0;
            LODWORD(v226) = v174;
            v191 = v195;
            v225 = (unsigned __int8 **)v223;
            sub_1727280((__int64 *)&v225, (__int64 *)(v195 + 24));
            v175 = (int)v226;
            LODWORD(v226) = 0;
            v228.m128_i32[2] = v175;
            v228.m128_u64[0] = (unsigned __int64)v225;
            LOBYTE(v195) = sub_13D01C0((__int64)&v228);
            sub_135E100((__int64 *)&v228);
            sub_135E100((__int64 *)&v225);
            sub_135E100((__int64 *)&v223);
            sub_135E100((__int64 *)&v221);
            sub_135E100((__int64 *)&v219);
            sub_135E100((__int64 *)&v217);
            if ( (_BYTE)v195 )
            {
              v176 = a1->m128i_i64[1];
              v228.m128_u64[0] = (unsigned __int64)"bitfield";
              v229.m128i_i16[0] = 259;
              v177 = sub_15A2D10(v215, v191, *(double *)v12.m128_u64, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64);
              v178 = sub_172AC10(
                       v176,
                       (__int64)v213,
                       v177,
                       (__int64 *)&v228,
                       *(double *)v12.m128_u64,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64);
              v179 = a1->m128i_i64[1];
              v214 = v178;
              v229.m128i_i16[0] = 257;
              sub_13A38D0((__int64)&v223, v60);
              sub_1727260((__int64 *)&v223, v204);
              v180 = v224;
              v224 = 0;
              LODWORD(v226) = v180;
              v225 = (unsigned __int8 **)v223;
              v181 = sub_159C0E0(*(__int64 **)(v179 + 24), (__int64)&v225);
              v69 = v214;
              v70 = v181;
              goto LABEL_79;
            }
          }
LABEL_150:
          sub_13A38D0((__int64)&v225, (__int64)v204);
          sub_13D0570((__int64)&v225);
          v113 = (int)v226;
          LODWORD(v226) = 0;
          v228.m128_i32[2] = v113;
          v228.m128_u64[0] = (unsigned __int64)v225;
          v114 = sub_1455820(v60, &v228);
          sub_135E100((__int64 *)&v228);
          sub_135E100((__int64 *)&v225);
          if ( !v114 )
            goto LABEL_151;
          v153 = v210;
          v154 = v209[16];
          if ( v154 == 51 )
          {
            if ( *((_QWORD *)v209 - 6) )
            {
              v223 = (unsigned __int8 *)*((_QWORD *)v209 - 6);
              if ( v210 == *((_QWORD *)v209 - 3) )
                goto LABEL_230;
            }
            if ( *((_QWORD *)v209 - 3) )
            {
              v223 = (unsigned __int8 *)*((_QWORD *)v209 - 3);
              if ( v210 == *((_QWORD *)v209 - 6) )
                goto LABEL_230;
            }
          }
          else if ( v154 == 5 && *((_WORD *)v209 + 9) == 27 )
          {
            if ( (v182 = *((_DWORD *)v209 + 5) & 0xFFFFFFF,
                  v183 = &v209[-24 * v182],
                  v184 = &v209[24 * (1 - v182)],
                  *(_QWORD *)v183)
              && (v223 = *(unsigned __int8 **)v183, v210 == *(_QWORD *)v184)
              || (v185 = *(unsigned __int8 **)v184) != 0 && (v223 = v185, v210 == *(_QWORD *)v183) )
            {
LABEL_230:
              v158 = a1->m128i_i64[1];
              v229.m128i_i16[0] = 257;
              LOWORD(v227) = 257;
              v159 = (__int64 *)sub_1729500(
                                  v158,
                                  v223,
                                  v197,
                                  (__int64 *)&v225,
                                  *(double *)v12.m128_u64,
                                  *(double *)v13.m128i_i64,
                                  *(double *)a5.m128i_i64);
              return sub_15FB440(27, v159, v153, (__int64)&v228, 0);
            }
          }
          v228.m128_u64[0] = (unsigned __int64)v209;
          v228.m128_u64[1] = (unsigned __int64)&v223;
          if ( (unsigned __int8)sub_1731E70((__int64)&v228, v210) )
          {
            v155 = a1->m128i_i64[1];
            v229.m128i_i16[0] = 257;
            LOWORD(v227) = 257;
            v156 = v209;
            v157 = (__int64 *)sub_1729500(
                                v155,
                                v223,
                                v196,
                                (__int64 *)&v225,
                                *(double *)v12.m128_u64,
                                *(double *)v13.m128i_i64,
                                *(double *)a5.m128i_i64);
            return sub_15FB440(27, v157, (__int64)v156, (__int64)&v228, 0);
          }
          v160 = v210;
          v161 = v209[16];
          if ( v161 == 52 )
          {
            if ( *((_QWORD *)v209 - 6) )
            {
              v223 = (unsigned __int8 *)*((_QWORD *)v209 - 6);
              if ( v210 == *((_QWORD *)v209 - 3) )
                goto LABEL_240;
            }
            if ( *((_QWORD *)v209 - 3) )
            {
              v223 = (unsigned __int8 *)*((_QWORD *)v209 - 3);
              if ( v210 == *((_QWORD *)v209 - 6) )
                goto LABEL_240;
            }
          }
          else if ( v161 == 5 && *((_WORD *)v209 + 9) == 28 )
          {
            if ( (v167 = *((_DWORD *)v209 + 5) & 0xFFFFFFF,
                  v168 = &v209[-24 * v167],
                  v169 = &v209[24 * (1 - v167)],
                  *(_QWORD *)v168)
              && (v223 = *(unsigned __int8 **)v168, v210 == *(_QWORD *)v169)
              || (v170 = *(unsigned __int8 **)v169) != 0 && (v223 = v170, v210 == *(_QWORD *)v168) )
            {
LABEL_240:
              v165 = a1->m128i_i64[1];
              v229.m128i_i16[0] = 257;
              LOWORD(v227) = 257;
              v166 = (__int64 *)sub_1729500(
                                  v165,
                                  v223,
                                  v197,
                                  (__int64 *)&v225,
                                  *(double *)v12.m128_u64,
                                  *(double *)v13.m128i_i64,
                                  *(double *)a5.m128i_i64);
              return sub_15FB440(28, v166, v160, (__int64)&v228, 0);
            }
          }
          v228.m128_u64[0] = (unsigned __int64)v209;
          v228.m128_u64[1] = (unsigned __int64)&v223;
          if ( (unsigned __int8)sub_1731F30((__int64)&v228, v210) )
          {
            v162 = a1->m128i_i64[1];
            v229.m128i_i16[0] = 257;
            LOWORD(v227) = 257;
            v163 = v209;
            v164 = (__int64 *)sub_1729500(
                                v162,
                                v223,
                                v196,
                                (__int64 *)&v225,
                                *(double *)v12.m128_u64,
                                *(double *)v13.m128i_i64,
                                *(double *)a5.m128i_i64);
            return sub_15FB440(28, v164, (__int64)v163, (__int64)&v228, 0);
          }
LABEL_151:
          v115 = *(_QWORD *)(v50 + 8);
          if ( v115 && !*(_QWORD *)(v115 + 8) || (v116 = *(_QWORD *)(v208 + 8)) != 0 && !*(_QWORD *)(v116 + 8) )
          {
            v117 = sub_17321F0(
                     (__int64)v209,
                     v211,
                     v210,
                     v212,
                     a1->m128i_i64[1],
                     *(double *)v12.m128_u64,
                     *(double *)v13.m128i_i64,
                     *(double *)a5.m128i_i64);
            if ( v117 )
              return sub_170E100(
                       a1->m128i_i64,
                       v11,
                       (__int64)v117,
                       v12,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64,
                       a6,
                       v118,
                       v119,
                       a9,
                       a10);
            v117 = sub_17321F0(
                     (__int64)v209,
                     v211,
                     v212,
                     v210,
                     a1->m128i_i64[1],
                     *(double *)v12.m128_u64,
                     *(double *)v13.m128i_i64,
                     *(double *)a5.m128i_i64);
            if ( v117 )
              return sub_170E100(
                       a1->m128i_i64,
                       v11,
                       (__int64)v117,
                       v12,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64,
                       a6,
                       v118,
                       v119,
                       a9,
                       a10);
            v117 = sub_17321F0(
                     v211,
                     (__int64)v209,
                     v210,
                     v212,
                     a1->m128i_i64[1],
                     *(double *)v12.m128_u64,
                     *(double *)v13.m128i_i64,
                     *(double *)a5.m128i_i64);
            if ( v117 )
              return sub_170E100(
                       a1->m128i_i64,
                       v11,
                       (__int64)v117,
                       v12,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64,
                       a6,
                       v118,
                       v119,
                       a9,
                       a10);
            v117 = sub_17321F0(
                     v211,
                     (__int64)v209,
                     v212,
                     v210,
                     a1->m128i_i64[1],
                     *(double *)v12.m128_u64,
                     *(double *)v13.m128i_i64,
                     *(double *)a5.m128i_i64);
            if ( v117 )
              return sub_170E100(
                       a1->m128i_i64,
                       v11,
                       (__int64)v117,
                       v12,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64,
                       a6,
                       v118,
                       v119,
                       a9,
                       a10);
            v117 = sub_17321F0(
                     v210,
                     v212,
                     (unsigned __int64)v209,
                     v211,
                     a1->m128i_i64[1],
                     *(double *)v12.m128_u64,
                     *(double *)v13.m128i_i64,
                     *(double *)a5.m128i_i64);
            if ( v117 )
              return sub_170E100(
                       a1->m128i_i64,
                       v11,
                       (__int64)v117,
                       v12,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64,
                       a6,
                       v118,
                       v119,
                       a9,
                       a10);
            v117 = sub_17321F0(
                     v210,
                     v212,
                     v211,
                     (__int64)v209,
                     a1->m128i_i64[1],
                     *(double *)v12.m128_u64,
                     *(double *)v13.m128i_i64,
                     *(double *)a5.m128i_i64);
            if ( v117 )
              return sub_170E100(
                       a1->m128i_i64,
                       v11,
                       (__int64)v117,
                       v12,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64,
                       a6,
                       v118,
                       v119,
                       a9,
                       a10);
            v117 = sub_17321F0(
                     v212,
                     v210,
                     (unsigned __int64)v209,
                     v211,
                     a1->m128i_i64[1],
                     *(double *)v12.m128_u64,
                     *(double *)v13.m128i_i64,
                     *(double *)a5.m128i_i64);
            if ( v117 )
              return sub_170E100(
                       a1->m128i_i64,
                       v11,
                       (__int64)v117,
                       v12,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64,
                       a6,
                       v118,
                       v119,
                       a9,
                       a10);
            v117 = sub_17321F0(
                     v212,
                     v210,
                     v211,
                     (__int64)v209,
                     a1->m128i_i64[1],
                     *(double *)v12.m128_u64,
                     *(double *)v13.m128i_i64,
                     *(double *)a5.m128i_i64);
            if ( v117 )
              return sub_170E100(
                       a1->m128i_i64,
                       v11,
                       (__int64)v117,
                       v12,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64,
                       a6,
                       v118,
                       v119,
                       a9,
                       a10);
          }
LABEL_57:
          v228.m128_u64[0] = (unsigned __int64)&v209;
          v228.m128_u64[1] = (unsigned __int64)&v210;
          if ( (unsigned __int8)sub_17317C0(&v228, v50) )
          {
            v228.m128_u64[0] = v210;
            v228.m128_u64[1] = (unsigned __int64)&v211;
            v229.m128i_i64[0] = (__int64)v209;
            if ( (unsigned __int8)sub_1731860((__int64)&v228, v208) )
            {
              v229.m128i_i16[0] = 257;
              return sub_15FB440(27, (__int64 *)v50, v211, (__int64)&v228, 0);
            }
          }
          v228.m128_u64[0] = (unsigned __int64)&v209;
          v228.m128_u64[1] = (unsigned __int64)&v211;
          v229.m128i_i64[0] = (__int64)&v210;
          if ( (unsigned __int8)sub_17319E0(&v228, v50) )
          {
            v228.m128_u64[0] = v210;
            v228.m128_u64[1] = (unsigned __int64)v209;
            if ( (unsigned __int8)sub_1731B90(&v228, v208) )
            {
              v229.m128i_i16[0] = 257;
              return sub_15FB440(27, (__int64 *)v208, v211, (__int64)&v228, 0);
            }
          }
          v228.m128_u64[1] = (unsigned __int64)&v211;
          v228.m128_u64[0] = v208;
          v229.m128i_i64[0] = (__int64)&v209;
          v203 = sub_1731FF0((__int64)&v228, v50);
          if ( v203 )
          {
            v56 = a1->m128i_i64[1];
            v229.m128i_i16[0] = 257;
            LOWORD(v227) = 257;
            v57 = sub_1729500(
                    v56,
                    v209,
                    v211,
                    (__int64 *)&v225,
                    *(double *)v12.m128_u64,
                    *(double *)v13.m128i_i64,
                    *(double *)a5.m128i_i64);
            return sub_15FB440(27, (__int64 *)v208, (__int64)v57, (__int64)&v228, 0);
          }
          v58 = sub_1732DB0(
                  v11,
                  a1->m128i_i64[1],
                  v54,
                  v55,
                  *(double *)v12.m128_u64,
                  *(double *)v13.m128i_i64,
                  *(double *)a5.m128i_i64);
          v59 = v58;
          if ( v58 )
            return v58;
          v76 = *(unsigned __int8 *)(v50 + 16);
          if ( (_BYTE)v76 == 52 || (_BYTE)v76 == 5 && *(_WORD *)(v50 + 18) == 28 )
          {
            v123 = v50;
            v203 = 1;
            v50 = v208;
            v208 = v123;
          }
          v228.m128_u64[0] = (unsigned __int64)&v209;
          v228.m128_u64[1] = (unsigned __int64)&v210;
          if ( (unsigned __int8)sub_17317C0(&v228, v208) )
          {
            v79 = (__int64 *)v209;
            v77 = v210;
            if ( v209 == (unsigned __int8 *)v50 )
              goto LABEL_215;
            if ( v50 == v210 )
              goto LABEL_215;
            v225 = (unsigned __int8 **)v209;
            v198 = v209;
            v226 = (unsigned __int64 *)v210;
            v201 = v210;
            v80 = sub_1732170(&v225, v50);
            v77 = v201;
            v79 = (__int64 *)v198;
            if ( v80
              || (v228.m128_u64[0] = v201,
                  v199 = v201,
                  v228.m128_u64[1] = (unsigned __int64)v79,
                  v202 = (__int64)v79,
                  v81 = sub_1732170(&v228, v50),
                  v79 = (__int64 *)v202,
                  v77 = v199,
                  v81) )
            {
LABEL_215:
              v229.m128i_i16[0] = 257;
              return sub_15FB440(27, v79, v77, (__int64)&v228, 0);
            }
            v82 = *(_QWORD *)(v208 + 8);
            if ( v82 )
            {
              if ( !*(_QWORD *)(v82 + 8) )
              {
                v228.m128_u64[0] = v50;
                if ( sub_13D1F50((__int64 *)&v228, v202) )
                {
                  v135 = a1->m128i_i64[1];
                  v149 = sub_1649960(v210);
                  v229.m128i_i16[0] = 773;
                  v84 = (unsigned __int8 *)v210;
                  v225 = (unsigned __int8 **)v149;
                  v226 = v150;
                  v228.m128_u64[0] = (unsigned __int64)&v225;
                  v228.m128_u64[1] = (unsigned __int64)".not";
LABEL_182:
                  v138 = (__int64 *)sub_171CA90(
                                      v135,
                                      (__int64)v84,
                                      (__int64 *)&v228,
                                      *(double *)v12.m128_u64,
                                      *(double *)v13.m128i_i64,
                                      *(double *)a5.m128i_i64);
                  v229.m128i_i16[0] = 257;
                  return sub_15FB440(27, v138, v50, (__int64)&v228, 0);
                }
                v82 = *(_QWORD *)(v208 + 8);
              }
              if ( v82 )
              {
                if ( !*(_QWORD *)(v82 + 8) )
                {
                  v228.m128_u64[0] = v50;
                  if ( sub_13D1F50((__int64 *)&v228, v210) )
                  {
                    v135 = a1->m128i_i64[1];
                    v151 = sub_1649960((__int64)v209);
                    v229.m128i_i16[0] = 773;
                    v84 = v209;
                    v225 = (unsigned __int8 **)v151;
                    v226 = v152;
                    v228.m128_u64[0] = (unsigned __int64)&v225;
                    v228.m128_u64[1] = (unsigned __int64)".not";
                    goto LABEL_182;
                  }
                }
              }
            }
          }
          v228.m128_u64[0] = (unsigned __int64)&v209;
          if ( sub_171DA10(&v228, v208, v77, v78) )
          {
            v83 = v209[16];
            if ( (unsigned __int8)(v83 - 35) <= 0x11u )
            {
              v84 = (unsigned __int8 *)*((_QWORD *)v209 - 6);
              if ( v84 == (unsigned __int8 *)v50 || *((_QWORD *)v209 - 3) == v50 )
              {
                v134 = *(_QWORD *)(v208 + 8);
                if ( v134 )
                {
                  if ( !*(_QWORD *)(v134 + 8) && (unsigned __int8)(v83 - 51) <= 1u )
                  {
                    if ( v84 == (unsigned __int8 *)v50 )
                      v84 = (unsigned __int8 *)*((_QWORD *)v209 - 3);
                    v135 = a1->m128i_i64[1];
                    v136 = sub_1649960((__int64)v84);
                    v229.m128i_i16[0] = 773;
                    v225 = (unsigned __int8 **)v136;
                    v226 = v137;
                    v228.m128_u64[0] = (unsigned __int64)&v225;
                    v228.m128_u64[1] = (unsigned __int64)".not";
                    goto LABEL_182;
                  }
                }
              }
            }
          }
          if ( v203 )
          {
            v85 = v50;
            v50 = v208;
            v208 = v85;
          }
          v86 = *(_BYTE *)(v208 + 16);
          if ( *(_BYTE *)(v50 + 16) == 75 )
          {
            if ( v86 == 75 )
            {
              v117 = sub_172F880(
                       a1,
                       v50,
                       v208,
                       v11,
                       *(double *)v12.m128_u64,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64);
              if ( v117 )
                return sub_170E100(
                         a1->m128i_i64,
                         v11,
                         (__int64)v117,
                         v12,
                         *(double *)v13.m128i_i64,
                         *(double *)a5.m128i_i64,
                         a6,
                         v118,
                         v119,
                         a9,
                         a10);
              v59 = v208;
            }
            v228 = v205;
            if ( (unsigned __int8)sub_1731410(&v228, v208) )
            {
              if ( v223[16] == 75 )
              {
                v120 = sub_172F880(
                         a1,
                         v50,
                         (__int64)v223,
                         v11,
                         *(double *)v12.m128_u64,
                         *(double *)v13.m128i_i64,
                         *(double *)a5.m128i_i64);
                if ( v120 )
                  goto LABEL_199;
              }
              if ( *((_BYTE *)v225 + 16) == 75 )
              {
                v120 = sub_172F880(
                         a1,
                         v50,
                         (__int64)v225,
                         v11,
                         *(double *)v12.m128_u64,
                         *(double *)v13.m128i_i64,
                         *(double *)a5.m128i_i64);
                if ( v120 )
                  goto LABEL_163;
              }
            }
          }
          else
          {
            if ( v86 != 75 )
              goto LABEL_106;
            v59 = v208;
          }
          if ( !v59 )
            goto LABEL_106;
          v228 = v205;
          if ( !(unsigned __int8)sub_1731410(&v228, v50) )
            goto LABEL_106;
          if ( v223[16] != 75
            || (v120 = sub_172F880(
                         a1,
                         (__int64)v223,
                         v59,
                         v11,
                         *(double *)v12.m128_u64,
                         *(double *)v13.m128i_i64,
                         *(double *)a5.m128i_i64)) == 0 )
          {
            if ( *((_BYTE *)v225 + 16) == 75 )
            {
              v120 = sub_172F880(
                       a1,
                       (__int64)v225,
                       v59,
                       v11,
                       *(double *)v12.m128_u64,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64);
              if ( v120 )
              {
LABEL_163:
                v229.m128i_i16[0] = 257;
                v121 = a1->m128i_i64[1];
                v122 = v223;
LABEL_164:
                v117 = sub_172AC10(
                         v121,
                         (__int64)v120,
                         (__int64)v122,
                         (__int64 *)&v228,
                         *(double *)v12.m128_u64,
                         *(double *)v13.m128i_i64,
                         *(double *)a5.m128i_i64);
                return sub_170E100(
                         a1->m128i_i64,
                         v11,
                         (__int64)v117,
                         v12,
                         *(double *)v13.m128i_i64,
                         *(double *)a5.m128i_i64,
                         a6,
                         v118,
                         v119,
                         a9,
                         a10);
              }
            }
LABEL_106:
            v87 = *(_QWORD *)(v11 - 48);
            if ( *(_BYTE *)(v87 + 16) != 76
              || (v124 = *(_QWORD *)(v11 - 24), *(_BYTE *)(v124 + 16) != 76)
              || (v117 = sub_1729330((__int64)a1, v87, v124, 0)) == 0 )
            {
              v25 = sub_1730740(a1, v11, *(double *)v12.m128_u64, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64);
              if ( v25 )
                return v25;
              v228.m128_u64[0] = (unsigned __int64)&v209;
              if ( (unsigned __int8)sub_1731CB0(&v228, v50) && sub_17287D0(*(_QWORD *)v209, 1) )
              {
                v139 = *(__int64 ***)v11;
                v229.m128i_i16[0] = 257;
                v140 = (_QWORD *)sub_15A0930((__int64)v139, -1);
                return sub_14EDD70((__int64)v209, v140, v208, (__int64)&v228, 0, 0);
              }
              v228.m128_u64[0] = (unsigned __int64)&v209;
              if ( (unsigned __int8)sub_1731CB0(&v228, v208) && sub_17287D0(*(_QWORD *)v209, 1) )
              {
                v127 = *(__int64 ***)v11;
                v229.m128i_i16[0] = 257;
                v128 = (_QWORD *)sub_15A0930((__int64)v127, -1);
                return sub_14EDD70((__int64)v209, v128, v50, (__int64)&v228, 0, 0);
              }
              v88 = *(_QWORD *)(v50 + 8);
              if ( v88 )
              {
                if ( !*(_QWORD *)(v88 + 8) && *(_BYTE *)(v208 + 16) != 13 )
                {
                  v228.m128_u64[0] = (unsigned __int64)&v209;
                  v228.m128_u64[1] = (unsigned __int64)&v219;
                  if ( (unsigned __int8)sub_1731DD0(&v228, v50) )
                  {
                    v125 = a1->m128i_i64[1];
                    v229.m128i_i16[0] = 257;
                    v126 = (__int64 *)sub_172AC10(
                                        v125,
                                        (__int64)v209,
                                        v208,
                                        (__int64 *)&v228,
                                        *(double *)v12.m128_u64,
                                        *(double *)v13.m128i_i64,
                                        *(double *)a5.m128i_i64);
                    sub_164B7C0((__int64)v126, v50);
                    v229.m128i_i16[0] = 257;
                    return sub_15FB440(27, v126, (__int64)v219, (__int64)&v228, 0);
                  }
                }
              }
              v221 = 0;
              v223 = 0;
              v89 = *(_QWORD *)(v50 + 8);
              if ( v89 )
              {
                if ( !*(_QWORD *)(v89 + 8) )
                {
                  v90 = *(_QWORD *)(v208 + 8);
                  if ( v90 )
                  {
                    if ( !*(_QWORD *)(v90 + 8) )
                    {
                      v225 = &v221;
                      v226 = (unsigned __int64 *)&v209;
                      v227 = &v210;
                      if ( (unsigned __int8)sub_17330B0(&v225, v50) )
                      {
                        v228.m128_u64[0] = (unsigned __int64)&v223;
                        v228.m128_u64[1] = (unsigned __int64)&v211;
                        v229.m128i_i64[0] = (__int64)&v212;
                        if ( (unsigned __int8)sub_17330B0(&v228, v208) )
                        {
                          if ( v221 == v223 )
                          {
                            v129 = a1->m128i_i64[1];
                            v229.m128i_i16[0] = 257;
                            v130 = sub_172AC10(
                                     v129,
                                     (__int64)v209,
                                     v211,
                                     (__int64 *)&v228,
                                     *(double *)v12.m128_u64,
                                     *(double *)v13.m128i_i64,
                                     *(double *)a5.m128i_i64);
                            v131 = a1->m128i_i64[1];
                            v132 = v130;
                            v229.m128i_i16[0] = 257;
                            v133 = sub_172AC10(
                                     v131,
                                     v210,
                                     v212,
                                     (__int64 *)&v228,
                                     *(double *)v12.m128_u64,
                                     *(double *)v13.m128i_i64,
                                     *(double *)a5.m128i_i64);
                            v229.m128i_i16[0] = 257;
                            return sub_14EDD70((__int64)v221, v132, (__int64)v133, (__int64)&v228, 0, 0);
                          }
                        }
                      }
                    }
                  }
                }
              }
              v25 = sub_172D090((__int64)a1, v11);
              if ( v25 )
                return v25;
              v25 = (__int64)sub_172CA50((__int64)a1, v11);
              if ( v25 )
                return v25;
              v91 = *(_BYTE *)(v50 + 16);
              if ( v91 > 0x17u )
              {
                v92 = *(_BYTE *)(v208 + 16);
                if ( v92 > 0x17u )
                {
                  if ( v91 == 47 )
                  {
                    if ( v92 != 48 )
                      return 0;
                  }
                  else if ( v91 != 48 || v92 != 47 )
                  {
                    return 0;
                  }
                  v93 = *(__int64 **)sub_13CF970(v50);
                  if ( *(__int64 **)sub_13CF970(v208) == v93 && sub_1642F90(*v93, 64) )
                  {
                    v206 = *(_BYTE *)(v50 + 16);
                    v94 = *(_QWORD *)(sub_13CF970(v50) + 24);
                    v95 = *(_QWORD *)(sub_13CF970(v208) + 24);
                    v96 = *(_BYTE *)(v94 + 16);
                    if ( v96 <= 0x17u )
                      goto LABEL_132;
                    if ( *(_BYTE *)(v95 + 16) == 61 && v96 == 61 )
                    {
                      v94 = *(_QWORD *)sub_13CF970(v94);
                      v95 = *(_QWORD *)sub_13CF970(v95);
LABEL_132:
                      if ( *(_BYTE *)(v94 + 16) == 13 && *(_BYTE *)(v95 + 16) == 13 )
                      {
                        v97 = *(_DWORD *)(v94 + 32);
                        v98 = *(__int64 **)(v94 + 24);
                        if ( v97 > 0x40 )
                          v99 = *v98;
                        else
                          v99 = (__int64)((_QWORD)v98 << (64 - (unsigned __int8)v97)) >> (64 - (unsigned __int8)v97);
                        v100 = *(_DWORD *)(v95 + 32);
                        v101 = *(__int64 **)(v95 + 24);
                        if ( v100 > 0x40 )
                          v102 = *v101;
                        else
                          v102 = (__int64)((_QWORD)v101 << (64 - (unsigned __int8)v100)) >> (64 - (unsigned __int8)v100);
                        v103 = v99;
                        if ( v206 != 48 )
                          v103 = v102;
                        if ( (_DWORD)v102 + (_DWORD)v99 == 64 && v103 != 32 && v103 <= 0x3F )
                        {
                          v104 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v50 + 40) + 56LL) + 40LL);
                          v105 = (_QWORD *)sub_16498A0(v11);
                          v228.m128_u64[0] = sub_1643360(v105);
                          v106 = sub_15E26F0(v104, 103, (__int64 *)&v228, 1);
                          v107 = (_QWORD *)sub_16498A0(v11);
                          v108 = sub_1643360(v107);
                          v109 = sub_159C470(v108, (int)v103, 1u);
                          v110 = *(_QWORD *)sub_13CF970(v50);
                          v230 = v109;
                          v228.m128_u64[0] = (unsigned __int64)&v229;
                          v229.m128i_i64[0] = v110;
                          v229.m128i_i64[1] = v110;
                          v228.m128_u64[1] = 0x300000003LL;
                          v225 = (unsigned __int8 **)"fshr64";
                          LOWORD(v227) = 259;
                          v11 = sub_17287F0(v106, v229.m128i_i64, 3, (__int64)&v225, 0);
                          if ( (__m128i *)v228.m128_u64[0] != &v229 )
                            _libc_free(v228.m128_u64[0]);
                          return v11;
                        }
                      }
                    }
                  }
                }
              }
              return 0;
            }
            return sub_170E100(
                     a1->m128i_i64,
                     v11,
                     (__int64)v117,
                     v12,
                     *(double *)v13.m128i_i64,
                     *(double *)a5.m128i_i64,
                     a6,
                     v118,
                     v119,
                     a9,
                     a10);
          }
LABEL_199:
          v229.m128i_i16[0] = 257;
          v121 = a1->m128i_i64[1];
          v122 = (unsigned __int8 *)v225;
          goto LABEL_164;
        }
        if ( v213 == (unsigned __int8 *)v210 )
        {
          v187 = v62;
          sub_13A38D0((__int64)&v221, v60);
          sub_13D0570((__int64)&v221);
          v71 = v222;
          v222 = 0;
          v224 = v71;
          v223 = v221;
          v64 = sub_17288A0(a1->m128i_i64, (__int64)v214, (__int64)&v223, 0, v11);
          v63 = v64;
          if ( v64 )
            goto LABEL_75;
          v63 = v187;
        }
        else
        {
          v63 = 0;
        }
        v64 = 0;
        if ( v214 == (unsigned __int8 *)v210 )
        {
          v186 = v63;
          sub_13A38D0((__int64)&v225, v60);
          sub_13D0570((__int64)&v225);
          v65 = (int)v226;
          LODWORD(v226) = 0;
          v228.m128_i32[2] = v65;
          v228.m128_u64[0] = (unsigned __int64)v225;
          v188 = sub_17288A0(a1->m128i_i64, (__int64)v213, (__int64)&v228, 0, v11);
          sub_135E100((__int64 *)&v228);
          sub_135E100((__int64 *)&v225);
          v63 = v186;
          v64 = v188;
        }
LABEL_75:
        if ( v63 )
        {
          v192 = v64;
          sub_135E100((__int64 *)&v223);
          sub_135E100((__int64 *)&v221);
          v64 = v192;
        }
        if ( v64 )
        {
          v66 = a1->m128i_i64[1];
          v229.m128i_i16[0] = 257;
          sub_13A38D0((__int64)&v223, v60);
          sub_1727260((__int64 *)&v223, v204);
          v67 = v224;
          v224 = 0;
          LODWORD(v226) = v67;
          v225 = (unsigned __int8 **)v223;
          v68 = sub_159C0E0(*(__int64 **)(v66 + 24), (__int64)&v225);
          v69 = v209;
          v70 = v68;
LABEL_79:
          v11 = sub_15FB440(26, (__int64 *)v69, v70, (__int64)&v228, 0);
          sub_135E100((__int64 *)&v225);
          sub_135E100((__int64 *)&v223);
          return v11;
        }
        goto LABEL_202;
      }
      v229.m128i_i16[0] = 257;
      v72 = a1->m128i_i64[1];
      v73 = v50;
    }
    v74 = (__int64 *)sub_172AC10(
                       v72,
                       (__int64)v223,
                       v73,
                       (__int64 *)&v228,
                       *(double *)v12.m128_u64,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64);
    sub_164B7C0((__int64)v74, v50);
    v229.m128i_i16[0] = 257;
    v75 = sub_15A1070(*v74, (__int64)v225);
    return sub_15FB440(28, v74, v75, (__int64)&v228, 0);
  }
  v39 = v11;
  v40 = (__int64 *)a1;
  return sub_170E100(v40, v39, v35, v12, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64, a6, v37, v38, a9, a10);
}
