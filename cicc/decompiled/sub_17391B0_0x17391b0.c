// Function: sub_17391B0
// Address: 0x17391b0
//
__int64 __fastcall sub_17391B0(
        __m128i *a1,
        __int64 ***a2,
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
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rbx
  __int64 v19; // r13
  __int64 v20; // r15
  _QWORD *v21; // rax
  double v22; // xmm4_8
  double v23; // xmm5_8
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 v27; // r15
  unsigned __int64 v28; // rdx
  char v29; // al
  __int64 v30; // rax
  __int64 v31; // rax
  char **v32; // rcx
  __int64 v33; // rbx
  char v34; // al
  __int64 v35; // rdx
  __int64 v36; // rcx
  double v37; // xmm4_8
  double v38; // xmm5_8
  char *v39; // rax
  char v40; // al
  unsigned __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 *v44; // rax
  __int64 v45; // rax
  __int64 v46; // r15
  __int64 v47; // rdx
  __int64 v48; // rcx
  int v49; // eax
  __int64 v50; // rdx
  double v51; // xmm4_8
  double v52; // xmm5_8
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // rax
  __int64 v56; // rbx
  int v57; // edx
  __int64 v58; // rsi
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 *v63; // rbx
  __int32 v64; // eax
  __int64 **v65; // rdi
  __int64 v66; // rbx
  __int64 v67; // rdi
  __int64 *v68; // r12
  __int64 **v69; // rbx
  unsigned int v70; // edx
  unsigned __int64 v71; // rax
  int v72; // eax
  __int64 v73; // rax
  __int64 v74; // rdi
  bool v75; // zf
  int v76; // esi
  unsigned __int8 *v77; // r15
  __int64 v78; // rax
  __int64 v79; // rdi
  unsigned __int8 *v80; // rax
  __int64 **v81; // r15
  __int64 v82; // r14
  _QWORD *v83; // rax
  __int64 **v84; // rcx
  __int64 v85; // rdi
  __int64 *v86; // r14
  __int64 *v87; // rax
  __int64 v88; // rax
  __int64 v89; // rdi
  __int64 *v90; // rsi
  unsigned __int8 *v91; // rax
  __int64 *v92; // rbx
  __int64 v93; // r14
  __int64 *v94; // rbx
  __int32 v95; // eax
  __int64 **v96; // rdi
  __int64 v97; // rax
  __int64 *v98; // r14
  __int64 **v99; // rdi
  __int64 v100; // rax
  __int64 v101; // rdi
  __int64 *v102; // rsi
  unsigned __int8 *v103; // rax
  int v104; // ebx
  __int64 v105; // r12
  __int64 **v106; // rdx
  unsigned __int8 *v107; // rax
  __int64 v108; // rdx
  __int64 v109; // rcx
  __int64 v110; // rax
  __int64 v111; // rax
  bool v112; // al
  __int64 v113; // r12
  __int64 **v114; // rdx
  __int64 *v115; // rax
  __int64 v116; // rdx
  __int64 v117; // rcx
  __int64 v118; // rax
  __int64 v119; // rax
  __int64 v120; // rdx
  __int64 v121; // rcx
  char v122; // al
  __int64 *v123; // r9
  char v124; // al
  __int64 v125; // rsi
  __int64 v126; // rdx
  __int64 v127; // rcx
  __int64 v128; // rdx
  __int64 v129; // rcx
  __int64 v130; // rax
  __int64 v131; // rdx
  unsigned __int8 *v132; // rdx
  double v133; // xmm4_8
  double v134; // xmm5_8
  __int64 v135; // rax
  unsigned __int8 *v136; // rsi
  __int64 v137; // rdi
  const char *v138; // rdx
  int v139; // eax
  __int64 **v140; // rdi
  __int64 v141; // r14
  __int64 v142; // rax
  __int64 *v143; // [rsp+8h] [rbp-F8h]
  unsigned __int8 v144; // [rsp+10h] [rbp-F0h]
  __int64 *v145; // [rsp+10h] [rbp-F0h]
  unsigned int v146; // [rsp+10h] [rbp-F0h]
  unsigned int v147; // [rsp+18h] [rbp-E8h]
  __int64 *v148; // [rsp+18h] [rbp-E8h]
  int v149; // [rsp+18h] [rbp-E8h]
  __m128 v150; // [rsp+20h] [rbp-E0h]
  __int64 *v151; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v152; // [rsp+30h] [rbp-D0h]
  unsigned __int64 v153; // [rsp+30h] [rbp-D0h]
  __int64 v155; // [rsp+40h] [rbp-C0h] BYREF
  unsigned __int8 *v156; // [rsp+48h] [rbp-B8h] BYREF
  __int64 *v157; // [rsp+50h] [rbp-B0h] BYREF
  __int64 *v158; // [rsp+58h] [rbp-A8h] BYREF
  __int64 *v159; // [rsp+60h] [rbp-A0h] BYREF
  __int32 v160; // [rsp+68h] [rbp-98h]
  char *v161; // [rsp+70h] [rbp-90h] BYREF
  __int32 v162; // [rsp+78h] [rbp-88h]
  const char *v163; // [rsp+80h] [rbp-80h] BYREF
  __int64 **v164; // [rsp+88h] [rbp-78h]
  __int16 v165; // [rsp+90h] [rbp-70h]
  __m128 v166; // [rsp+A0h] [rbp-60h] BYREF
  __m128i v167; // [rsp+B0h] [rbp-50h] BYREF
  __int64 ***v168; // [rsp+C0h] [rbp-40h]
  char **v169; // [rsp+C8h] [rbp-38h]

  v11 = (__int64)a2;
  v12 = (__m128)_mm_loadu_si128(a1 + 167);
  v13 = _mm_loadu_si128(a1 + 168);
  v168 = a2;
  v14 = (__int64)*(a2 - 6);
  v15 = (__int64)*(a2 - 3);
  v166 = v12;
  v167 = v13;
  v16 = sub_13E01B0(v14, v15, &v166);
  if ( v16 )
  {
    v18 = *(_QWORD *)(v11 + 8);
    if ( !v18 )
      return 0;
    v19 = a1->m128i_i64[0];
    v20 = v16;
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
  v26 = *(_QWORD *)(v11 - 48);
  v27 = a1->m128i_i64[1];
  v150.m128_u64[0] = (unsigned __int64)&v159;
  v28 = *(_QWORD *)(v11 - 24);
  v166.m128_u64[0] = (unsigned __int64)&v159;
  v166.m128_u64[1] = (unsigned __int64)&v161;
  v167.m128i_i64[0] = (__int64)&v159;
  v167.m128i_i64[1] = (__int64)&v161;
  v29 = *(_BYTE *)(v26 + 16);
  v150.m128_u64[1] = (unsigned __int64)&v161;
  if ( v29 == 51 )
  {
    if ( !*(_QWORD *)(v26 - 48) )
      goto LABEL_17;
    v159 = *(__int64 **)(v26 - 48);
    v39 = *(char **)(v26 - 24);
    if ( !v39 )
      goto LABEL_17;
  }
  else
  {
    if ( v29 != 5 )
      goto LABEL_17;
    if ( *(_WORD *)(v26 + 18) != 27 )
      goto LABEL_17;
    v45 = *(_DWORD *)(v26 + 20) & 0xFFFFFFF;
    if ( !*(_QWORD *)(v26 - 24 * v45) )
      goto LABEL_17;
    v159 = *(__int64 **)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
    v39 = *(char **)(v26 + 24 * (1 - v45));
    if ( !v39 )
      goto LABEL_17;
  }
  v152 = v28;
  v161 = v39;
  v40 = sub_1734DA0((__int64 **)&v167, v28);
  v28 = v152;
  if ( !v40 )
  {
LABEL_17:
    v30 = *(_QWORD *)(v26 + 8);
    if ( !v30 || *(_QWORD *)(v30 + 8) )
    {
      v31 = *(_QWORD *)(v28 + 8);
      if ( !v31 || *(_QWORD *)(v31 + 8) )
        goto LABEL_24;
    }
    v32 = &v161;
    v33 = *(_QWORD *)(v11 - 48);
    v166 = v150;
    v169 = &v161;
    v167.m128i_i64[1] = v150.m128_u64[0];
    v34 = *(_BYTE *)(v33 + 16);
    if ( v34 == 51 )
    {
      if ( !*(_QWORD *)(v33 - 48)
        || (v159 = *(__int64 **)(v33 - 48),
            !sub_171DA10((_QWORD **)&v166.m128_u64[1], *(_QWORD *)(v33 - 24), v28, (__int64)&v161)) )
      {
        v42 = *(_QWORD *)(v33 - 24);
        if ( !v42 )
          goto LABEL_24;
        v43 = v166.m128_u64[0];
        *(_QWORD *)v166.m128_u64[0] = v42;
        if ( !sub_171DA10((_QWORD **)&v166.m128_u64[1], *(_QWORD *)(v33 - 48), v43, (__int64)v32) )
          goto LABEL_24;
      }
      goto LABEL_36;
    }
    if ( v34 != 5 || *(_WORD *)(v33 + 18) != 27 )
      goto LABEL_24;
    v57 = *(_DWORD *)(v33 + 20);
    v58 = v57 & 0xFFFFFFF;
    if ( *(_QWORD *)(v33 - 24 * v58) )
    {
      v159 = *(__int64 **)(v33 - 24 * v58);
      if ( sub_14B2B20((_QWORD **)&v166.m128_u64[1], *(_QWORD *)(v33 + 24 * (1 - v58))) )
      {
LABEL_36:
        if ( !sub_17390A0((__int64 **)&v167.m128i_i64[1], *(_QWORD *)(v11 - 24), v41, (__int64)v32) )
          goto LABEL_24;
        v165 = 257;
        v167.m128i_i16[0] = 257;
        v44 = (__int64 *)sub_172B670(
                           v27,
                           (__int64)v159,
                           (__int64)v161,
                           (__int64 *)&v163,
                           *(double *)v12.m128_u64,
                           *(double *)v13.m128i_i64,
                           *(double *)a5.m128i_i64);
        v25 = sub_15FB630(v44, (__int64)&v166, 0);
        goto LABEL_30;
      }
      v57 = *(_DWORD *)(v33 + 20);
    }
    v59 = *(_QWORD *)(v33 + 24 * (1LL - (v57 & 0xFFFFFFF)));
    if ( !v59 )
      goto LABEL_24;
    *(_QWORD *)v166.m128_u64[0] = v59;
    if ( !sub_14B2B20((_QWORD **)&v166.m128_u64[1], *(_QWORD *)(v33 - 24LL * (*(_DWORD *)(v33 + 20) & 0xFFFFFFF))) )
      goto LABEL_24;
    goto LABEL_36;
  }
  v167.m128i_i16[0] = 257;
  v25 = sub_15FB440(28, v159, (__int64)v161, (__int64)&v166, 0);
LABEL_30:
  if ( v25 )
    return v25;
LABEL_24:
  v35 = (__int64)sub_1708300(a1, (unsigned __int8 *)v11, (__m128i)v12, v13, a5);
  if ( !v35 )
  {
    v35 = sub_172C850(
            v11,
            a1->m128i_i64[1],
            *(double *)v12.m128_u64,
            *(double *)v13.m128i_i64,
            *(double *)a5.m128i_i64,
            0,
            v36);
    if ( !v35 )
    {
      v46 = *(_QWORD *)(v11 - 48);
      v153 = *(_QWORD *)(v11 - 24);
      v166.m128_u64[0] = (unsigned __int64)&v155;
      if ( !(unsigned __int8)sub_13D2630(&v166, (_BYTE *)v153) )
        goto LABEL_43;
      v166.m128_u64[1] = (unsigned __int64)&v156;
      v60 = *(_QWORD *)(v46 + 8);
      if ( v60 && !*(_QWORD *)(v60 + 8) && (unsigned __int8)sub_17373D0((__int64)&v166, v46, v47, v48) )
      {
        v47 = *(unsigned int *)(v155 + 8);
        if ( (unsigned int)v47 <= 0x40 )
        {
          if ( *(_QWORD *)v155 == 1 )
            goto LABEL_150;
        }
        else
        {
          v149 = *(_DWORD *)(v155 + 8);
          v139 = sub_16A57B0(v155);
          v47 = (unsigned int)(v149 - 1);
          if ( v139 == (_DWORD)v47 )
          {
LABEL_150:
            v140 = *(__int64 ***)v11;
            v141 = a1->m128i_i64[1];
            v167.m128i_i16[0] = 257;
            v142 = sub_15A0680((__int64)v140, 0, 0);
            v80 = sub_17203D0(v141, 32, (__int64)v156, v142, (__int64 *)&v166);
LABEL_73:
            v81 = *(__int64 ***)v11;
            v82 = (__int64)v80;
            v167.m128i_i16[0] = 257;
            v83 = sub_1648A60(56, 1u);
            v11 = (__int64)v83;
            if ( v83 )
              sub_15FC690((__int64)v83, v82, (__int64)v81, (__int64)&v166, 0);
            return v11;
          }
        }
      }
      v166.m128_u64[0] = (unsigned __int64)&v156;
      v166.m128_u64[1] = (unsigned __int64)&v157;
      if ( (unsigned __int8)sub_17310F0(&v166, v46, v47, v48) )
      {
        v63 = v157;
        sub_13A38D0((__int64)&v163, v155);
        sub_1727280((__int64 *)&v163, v63);
        v64 = (int)v164;
        v65 = *(__int64 ***)v11;
        LODWORD(v164) = 0;
        v166.m128_i32[2] = v64;
        v166.m128_u64[0] = (unsigned __int64)v163;
        v66 = sub_15A1070((__int64)v65, (__int64)&v166);
        sub_135E100((__int64 *)&v166);
        sub_135E100((__int64 *)&v163);
        v67 = a1->m128i_i64[1];
        v167.m128i_i16[0] = 257;
        v68 = (__int64 *)sub_1729500(
                           v67,
                           v156,
                           v153,
                           (__int64 *)&v166,
                           *(double *)v12.m128_u64,
                           *(double *)v13.m128i_i64,
                           *(double *)a5.m128i_i64);
        sub_164B7C0((__int64)v68, v46);
        v167.m128i_i16[0] = 257;
        return sub_15FB440(28, v68, v66, (__int64)&v166, 0);
      }
      v166.m128_u64[0] = (unsigned __int64)&v156;
      v166.m128_u64[1] = (unsigned __int64)&v158;
      if ( (unsigned __int8)sub_1731230(&v166, v46, v61, v62) )
      {
        v92 = v158;
        sub_13A38D0((__int64)&v166, v155);
        sub_1727280((__int64 *)&v166, v92);
        v93 = a1->m128i_i64[1];
        v167.m128i_i16[0] = 257;
        v160 = v166.m128_i32[2];
        v94 = (__int64 *)v155;
        v159 = (__int64 *)v166.m128_u64[0];
        sub_13A38D0((__int64)&v161, (__int64)&v159);
        sub_1727240((__int64 *)&v161, v94);
        v95 = v162;
        v96 = *(__int64 ***)v11;
        v162 = 0;
        LODWORD(v164) = v95;
        v163 = v161;
        v97 = sub_15A1070((__int64)v96, (__int64)&v163);
        v98 = (__int64 *)sub_1729500(
                           v93,
                           v156,
                           v97,
                           (__int64 *)&v166,
                           *(double *)v12.m128_u64,
                           *(double *)v13.m128i_i64,
                           *(double *)a5.m128i_i64);
        sub_135E100((__int64 *)&v163);
        sub_135E100((__int64 *)&v161);
        sub_164B7C0((__int64)v98, v46);
        v99 = *(__int64 ***)v11;
        v167.m128i_i16[0] = 257;
        v100 = sub_15A1070((__int64)v99, (__int64)&v159);
        v11 = sub_15FB440(27, v98, v100, (__int64)&v166, 0);
        sub_135E100((__int64 *)&v159);
        return v11;
      }
      v163 = (const char *)&v156;
      v164 = &v159;
      if ( (unsigned __int8)sub_1731370((_QWORD **)&v163, v46)
        || (v166.m128_u64[0] = (unsigned __int64)&v156,
            v166.m128_u64[1] = (unsigned __int64)&v159,
            (unsigned __int8)sub_1731410(&v166, v46)) )
      {
        sub_13A38D0((__int64)&v166, v155);
        sub_13D0570((__int64)&v166);
        v162 = v166.m128_i32[2];
        v161 = (char *)v166.m128_u64[0];
        v104 = *(unsigned __int8 *)(v46 + 16) - 24;
        if ( (unsigned __int8)sub_17288A0(a1->m128i_i64, (__int64)v156, (__int64)&v161, 0, v11) )
        {
          v105 = a1->m128i_i64[1];
          v163 = sub_1649960((__int64)v159);
          v164 = v106;
          v166.m128_u64[0] = (unsigned __int64)&v163;
          v167.m128i_i16[0] = 773;
          v166.m128_u64[1] = (unsigned __int64)".masked";
          v107 = sub_1729500(
                   v105,
                   (unsigned __int8 *)v159,
                   v153,
                   (__int64 *)&v166,
                   *(double *)v12.m128_u64,
                   *(double *)v13.m128i_i64,
                   *(double *)a5.m128i_i64);
          v167.m128i_i16[0] = 257;
          v11 = sub_15FB440(v104, (__int64 *)v156, (__int64)v107, (__int64)&v166, 0);
        }
        else
        {
          if ( *((_BYTE *)v159 + 16) <= 0x10u
            || !(unsigned __int8)sub_17288A0(a1->m128i_i64, (__int64)v159, (__int64)&v161, 0, v11) )
          {
            sub_135E100((__int64 *)&v161);
            goto LABEL_43;
          }
          v113 = a1->m128i_i64[1];
          v163 = sub_1649960((__int64)v156);
          v164 = v114;
          v166.m128_u64[0] = (unsigned __int64)&v163;
          v167.m128i_i16[0] = 773;
          v166.m128_u64[1] = (unsigned __int64)".masked";
          v115 = (__int64 *)sub_1729500(
                              v113,
                              v156,
                              v153,
                              (__int64 *)&v166,
                              *(double *)v12.m128_u64,
                              *(double *)v13.m128i_i64,
                              *(double *)a5.m128i_i64);
          v167.m128i_i16[0] = 257;
          v11 = sub_15FB440(v104, v115, (__int64)v159, (__int64)&v166, 0);
        }
        sub_135E100((__int64 *)&v161);
        return v11;
      }
LABEL_43:
      if ( *(_BYTE *)(v153 + 16) == 13 )
      {
        v49 = *(unsigned __int8 *)(v46 + 16);
        if ( (unsigned __int8)v49 > 0x17u && (unsigned int)(v49 - 35) <= 0x11 )
        {
          if ( ((1LL << ((unsigned __int8)v49 - 24)) & 0x1800A800) != 0 )
          {
            v166.m128_u64[0] = (unsigned __int64)&v161;
            v166.m128_u64[1] = (unsigned __int64)&v163;
            if ( (unsigned __int8)sub_17314B0(&v166, v46) )
            {
              v69 = *(__int64 ***)v161;
              v147 = sub_16431D0(*(_QWORD *)v161);
              v70 = *(_DWORD *)(v153 + 32);
              if ( v70 > 0x40 )
              {
                v146 = *(_DWORD *)(v153 + 32);
                v72 = sub_16A57B0(v153 + 24);
                v70 = v146;
              }
              else
              {
                v71 = *(_QWORD *)(v153 + 24);
                if ( v71 )
                {
                  _BitScanReverse64(&v71, v71);
                  LODWORD(v71) = v71 ^ 0x3F;
                }
                else
                {
                  LODWORD(v71) = 64;
                }
                v72 = v70 + v71 - 64;
              }
              if ( v147 >= v70 - v72 )
              {
                v73 = sub_15A43B0((unsigned __int64)v163, v69, 0);
                v74 = a1->m128i_i64[1];
                v75 = *(_BYTE *)(*(_QWORD *)(v46 - 48) + 16LL) == 61;
                v167.m128i_i16[0] = 257;
                v76 = *(unsigned __int8 *)(v46 + 16);
                if ( v75 )
                  v77 = (unsigned __int8 *)sub_17066B0(
                                             v74,
                                             v76 - 24,
                                             (__int64)v161,
                                             v73,
                                             (__int64 *)&v166,
                                             0,
                                             *(double *)v12.m128_u64,
                                             *(double *)v13.m128i_i64,
                                             *(double *)a5.m128i_i64);
                else
                  v77 = (unsigned __int8 *)sub_17066B0(
                                             v74,
                                             v76 - 24,
                                             v73,
                                             (__int64)v161,
                                             (__int64 *)&v166,
                                             0,
                                             *(double *)v12.m128_u64,
                                             *(double *)v13.m128i_i64,
                                             *(double *)a5.m128i_i64);
                v78 = sub_15A43B0(v153, *(__int64 ***)v161, 0);
                v79 = a1->m128i_i64[1];
                v167.m128i_i16[0] = 257;
                v80 = sub_1729500(
                        v79,
                        v77,
                        v78,
                        (__int64 *)&v166,
                        *(double *)v12.m128_u64,
                        *(double *)v13.m128i_i64,
                        *(double *)a5.m128i_i64);
                goto LABEL_73;
              }
            }
          }
          v50 = *(_QWORD *)(v46 - 24);
          if ( *(_BYTE *)(v50 + 16) == 13 )
          {
            v25 = sub_17296C0(
                    (__int64)a1,
                    v46,
                    v50,
                    v153,
                    v11,
                    *(double *)v12.m128_u64,
                    *(double *)v13.m128i_i64,
                    *(double *)a5.m128i_i64);
            if ( v25 )
              return v25;
          }
        }
        v161 = 0;
        v163 = 0;
        v166.m128_u64[0] = (unsigned __int64)&v161;
        v166.m128_u64[1] = (unsigned __int64)&v163;
        if ( (unsigned __int8)sub_17315C0(&v166, v46) )
        {
          v84 = *(__int64 ***)v11;
          v85 = a1->m128i_i64[1];
          v166.m128_u64[0] = (unsigned __int64)"and.shrunk";
          v167.m128i_i16[0] = 259;
          v86 = (__int64 *)sub_1708970(v85, 36, (__int64)v161, v84, (__int64 *)&v166);
          v87 = (__int64 *)sub_15A43B0((unsigned __int64)v163, *(__int64 ***)v11, 0);
          v88 = sub_15A2CF0(v87, v153, *(double *)v12.m128_u64, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64);
          v167.m128i_i16[0] = 257;
          return sub_15FB440(26, v86, v88, (__int64)&v166, 0);
        }
      }
      v25 = (__int64)sub_172A7A0(
                       (__int64)a1,
                       (__int64 *)v11,
                       *(double *)v12.m128_u64,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64);
      if ( v25 )
        return v25;
      v25 = sub_1713A90(
              a1->m128i_i64,
              (_BYTE *)v11,
              v12,
              *(double *)v13.m128i_i64,
              *(double *)a5.m128i_i64,
              a6,
              v51,
              v52,
              a9,
              a10);
      if ( v25 )
        return v25;
      v55 = sub_1732DB0(
              v11,
              a1->m128i_i64[1],
              v53,
              v54,
              *(double *)v12.m128_u64,
              *(double *)v13.m128i_i64,
              *(double *)a5.m128i_i64);
      v56 = v55;
      if ( v55 )
        return v55;
      v166.m128_u64[0] = v46;
      v166.m128_u64[1] = (unsigned __int64)&v159;
      if ( (unsigned __int8)sub_17316C0((__int64)&v166, v153) )
      {
        v167.m128i_i16[0] = 257;
        v89 = a1->m128i_i64[1];
        v165 = 257;
        v90 = v159;
LABEL_78:
        v91 = sub_171CA90(
                v89,
                (__int64)v90,
                (__int64 *)&v163,
                *(double *)v12.m128_u64,
                *(double *)v13.m128i_i64,
                *(double *)a5.m128i_i64);
        return sub_15FB440(26, (__int64 *)v46, (__int64)v91, (__int64)&v166, 0);
      }
      v166.m128_u64[0] = v153;
      v166.m128_u64[1] = (unsigned __int64)&v159;
      v144 = sub_17316C0((__int64)&v166, v46);
      if ( v144 )
      {
        v167.m128i_i16[0] = 257;
        v101 = a1->m128i_i64[1];
        v165 = 257;
        v102 = v159;
LABEL_83:
        v103 = sub_171CA90(
                 v101,
                 (__int64)v102,
                 (__int64 *)&v163,
                 *(double *)v12.m128_u64,
                 *(double *)v13.m128i_i64,
                 *(double *)a5.m128i_i64);
        return sub_15FB440(26, (__int64 *)v153, (__int64)v103, (__int64)&v166, 0);
      }
      v166.m128_u64[0] = (unsigned __int64)&v158;
      v166.m128_u64[1] = (unsigned __int64)&v159;
      if ( (unsigned __int8)sub_17317C0(&v166, v46) )
      {
        v166.m128_u64[0] = (unsigned __int64)v159;
        v166.m128_u64[1] = (unsigned __int64)&v161;
        v167.m128i_i64[0] = (__int64)v158;
        if ( (unsigned __int8)sub_1731860((__int64)&v166, v153) )
        {
          if ( (v110 = *(_QWORD *)(v153 + 8)) != 0 && !*(_QWORD *)(v110 + 8)
            || ((v111 = *((_QWORD *)v161 + 1)) == 0 ? (v112 = 0) : (v112 = *(_QWORD *)(v111 + 8) == 0),
                (unsigned __int8)sub_1727170(v161, v112, v108, v109)) )
          {
            v167.m128i_i16[0] = 257;
            v89 = a1->m128i_i64[1];
            v165 = 257;
            v90 = (__int64 *)v161;
            goto LABEL_78;
          }
        }
      }
      v166.m128_u64[0] = (unsigned __int64)&v158;
      v166.m128_u64[1] = (unsigned __int64)&v161;
      v167.m128i_i64[0] = (__int64)&v159;
      if ( (unsigned __int8)sub_17319E0(&v166, v46) )
      {
        v166.m128_u64[0] = (unsigned __int64)v159;
        v166.m128_u64[1] = (unsigned __int64)v158;
        if ( (unsigned __int8)sub_1731B90(&v166, v153) )
        {
          v118 = *(_QWORD *)(v46 + 8);
          if ( v118 )
          {
            if ( !*(_QWORD *)(v118 + 8) )
              goto LABEL_109;
          }
          v119 = *((_QWORD *)v161 + 1);
          if ( v119 )
            v144 = *(_QWORD *)(v119 + 8) == 0;
          if ( (unsigned __int8)sub_1727170(v161, v144, v116, v117) )
          {
LABEL_109:
            v167.m128i_i16[0] = 257;
            v101 = a1->m128i_i64[1];
            v165 = 257;
            v102 = (__int64 *)v161;
            goto LABEL_83;
          }
        }
      }
      v166.m128_u64[0] = (unsigned __int64)&v158;
      v167.m128i_i64[0] = (__int64)&v159;
      if ( sub_1735320(&v166, v153, v116, v117) )
      {
        v163 = (const char *)v158;
        v143 = v158;
        v164 = (__int64 **)v159;
        v145 = v159;
        v122 = sub_1720000(&v163, v46);
        v120 = (__int64)v145;
        v123 = v143;
        if ( v122 )
          goto LABEL_140;
      }
      v166.m128_u64[0] = (unsigned __int64)&v158;
      v167.m128i_i64[0] = (__int64)&v159;
      if ( sub_1735320(&v166, v46, v120, v121) )
      {
        v163 = (const char *)v158;
        v148 = v158;
        v164 = (__int64 **)v159;
        v151 = v159;
        v124 = sub_1720000(&v163, v153);
        v120 = (__int64)v151;
        v123 = v148;
        if ( v124 )
        {
LABEL_140:
          v167.m128i_i16[0] = 257;
          return sub_15FB440(26, v123, v120, (__int64)&v166, 0);
        }
      }
      if ( *(_BYTE *)(v46 + 16) == 75 )
      {
        if ( *(_BYTE *)(v153 + 16) == 75 )
        {
          v132 = sub_17306D0(
                   a1,
                   v46,
                   v153,
                   v11,
                   *(double *)v12.m128_u64,
                   *(double *)v13.m128i_i64,
                   *(double *)a5.m128i_i64);
          if ( v132 )
            return sub_170E100(
                     a1->m128i_i64,
                     v11,
                     (__int64)v132,
                     v12,
                     *(double *)v13.m128i_i64,
                     *(double *)a5.m128i_i64,
                     a6,
                     v133,
                     v134,
                     a9,
                     a10);
          v56 = v153;
        }
        v166.m128_u64[0] = (unsigned __int64)&v161;
        v166.m128_u64[1] = (unsigned __int64)&v163;
        if ( (unsigned __int8)sub_1731C10(&v166, v153) )
        {
          if ( v161[16] == 75 )
          {
            v136 = sub_17306D0(
                     a1,
                     v46,
                     (__int64)v161,
                     v11,
                     *(double *)v12.m128_u64,
                     *(double *)v13.m128i_i64,
                     *(double *)a5.m128i_i64);
            if ( v136 )
              goto LABEL_142;
          }
          if ( v163[16] == 75 )
          {
            v136 = sub_17306D0(
                     a1,
                     v46,
                     (__int64)v163,
                     v11,
                     *(double *)v12.m128_u64,
                     *(double *)v13.m128i_i64,
                     *(double *)a5.m128i_i64);
            if ( v136 )
            {
LABEL_138:
              v167.m128i_i16[0] = 257;
              v137 = a1->m128i_i64[1];
              v138 = v161;
LABEL_139:
              v132 = sub_1729500(
                       v137,
                       v136,
                       (__int64)v138,
                       (__int64 *)&v166,
                       *(double *)v12.m128_u64,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64);
              return sub_170E100(
                       a1->m128i_i64,
                       v11,
                       (__int64)v132,
                       v12,
                       *(double *)v13.m128i_i64,
                       *(double *)a5.m128i_i64,
                       a6,
                       v133,
                       v134,
                       a9,
                       a10);
            }
          }
        }
      }
      else
      {
        v56 = v153;
        if ( *(_BYTE *)(v153 + 16) != 75 )
          goto LABEL_120;
      }
      if ( !v56
        || (v166.m128_u64[0] = (unsigned __int64)&v161,
            v166.m128_u64[1] = (unsigned __int64)&v163,
            !(unsigned __int8)sub_1731C10(&v166, v46)) )
      {
LABEL_120:
        v125 = *(_QWORD *)(v11 - 48);
        if ( *(_BYTE *)(v125 + 16) != 76
          || (v131 = *(_QWORD *)(v11 - 24), *(_BYTE *)(v131 + 16) != 76)
          || (v132 = sub_1729330((__int64)a1, v125, v131, 1)) == 0 )
        {
          v11 = sub_1730740(a1, v11, *(double *)v12.m128_u64, *(double *)v13.m128i_i64, *(double *)a5.m128i_i64);
          if ( !v11 )
          {
            v166.m128_u64[0] = (unsigned __int64)&v163;
            if ( (unsigned __int8)sub_1731CB0(&v166, v46) && sub_17287D0(*(_QWORD *)v163, 1) )
            {
              v167.m128i_i16[0] = 257;
              v135 = sub_15A06D0(*a2, 1, v126, v127);
              return sub_14EDD70((__int64)v163, (_QWORD *)v153, v135, (__int64)&v166, 0, 0);
            }
            else
            {
              v166.m128_u64[0] = (unsigned __int64)&v163;
              if ( (unsigned __int8)sub_1731CB0(&v166, v153) && sub_17287D0(*(_QWORD *)v163, 1) )
              {
                v167.m128i_i16[0] = 257;
                v130 = sub_15A06D0(*a2, 1, v128, v129);
                return sub_14EDD70((__int64)v163, (_QWORD *)v46, v130, (__int64)&v166, 0, 0);
              }
            }
          }
          return v11;
        }
        return sub_170E100(
                 a1->m128i_i64,
                 v11,
                 (__int64)v132,
                 v12,
                 *(double *)v13.m128i_i64,
                 *(double *)a5.m128i_i64,
                 a6,
                 v133,
                 v134,
                 a9,
                 a10);
      }
      if ( v161[16] != 75
        || (v136 = sub_17306D0(
                     a1,
                     (__int64)v161,
                     v56,
                     v11,
                     *(double *)v12.m128_u64,
                     *(double *)v13.m128i_i64,
                     *(double *)a5.m128i_i64)) == 0 )
      {
        if ( v163[16] != 75 )
          goto LABEL_120;
        v136 = sub_17306D0(
                 a1,
                 (__int64)v163,
                 v56,
                 v11,
                 *(double *)v12.m128_u64,
                 *(double *)v13.m128i_i64,
                 *(double *)a5.m128i_i64);
        if ( !v136 )
          goto LABEL_120;
        goto LABEL_138;
      }
LABEL_142:
      v167.m128i_i16[0] = 257;
      v137 = a1->m128i_i64[1];
      v138 = v163;
      goto LABEL_139;
    }
  }
  return sub_170E100(
           a1->m128i_i64,
           v11,
           v35,
           v12,
           *(double *)v13.m128i_i64,
           *(double *)a5.m128i_i64,
           a6,
           v37,
           v38,
           a9,
           a10);
}
