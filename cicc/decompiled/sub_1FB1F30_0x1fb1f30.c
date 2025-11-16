// Function: sub_1FB1F30
// Address: 0x1fb1f30
//
__int64 __fastcall sub_1FB1F30(__int64 **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rbx
  __int64 *v7; // rax
  __int64 v8; // r15
  __int64 v9; // r13
  __m128i v10; // xmm0
  __int64 v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // r9
  __m128i v14; // xmm1
  char *v15; // rcx
  char v16; // al
  const void **v17; // rcx
  bool v18; // al
  int v19; // eax
  int v20; // eax
  __int64 v21; // r14
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 result; // rax
  int v25; // eax
  void *v26; // rsi
  __int64 v27; // r12
  __int64 v28; // rdx
  void *v29; // rsi
  void *v30; // rsi
  __int64 *v31; // r12
  __int64 v32; // rdx
  __int64 *v33; // r13
  unsigned int v34; // eax
  __int64 v35; // rsi
  void *v36; // rdx
  char v37; // r13
  __int64 v38; // r9
  void *v39; // rsi
  __int64 *v40; // r12
  const void **v41; // r13
  unsigned int v42; // r15d
  char v43; // al
  unsigned __int8 *v44; // r14
  const void **v45; // r13
  unsigned int v46; // r15d
  __int64 v47; // r12
  __int64 v48; // rdx
  char v49; // r14
  char v50; // al
  int v51; // eax
  int v52; // r8d
  int v53; // r9d
  unsigned int v54; // r10d
  __int64 v55; // rsi
  __int64 v56; // r8
  char v57; // di
  char v58; // r11
  char v59; // di
  int v60; // edx
  int v61; // eax
  int v62; // edx
  unsigned __int8 *v63; // rdx
  char v64; // al
  const void **v65; // r13
  __int64 *v66; // rax
  __int64 *v67; // rax
  __int64 v68; // rcx
  void **p_s; // rdi
  __int64 v70; // r11
  __int64 v71; // r8
  __int64 v72; // r11
  __int64 v73; // r11
  void *v74; // rax
  __int64 *v75; // rdi
  __int64 v76; // rax
  unsigned __int64 v77; // rdx
  __int64 v78; // rax
  const void **v79; // rdx
  __int128 v80; // rax
  __int64 v81; // r8
  __int64 *v82; // rax
  __int64 *v83; // rax
  __int64 v84; // rcx
  unsigned __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // r15
  void *v88; // r12
  __int64 v89; // r13
  __int64 v90; // rdx
  unsigned __int64 v91; // rax
  __int64 v92; // rdx
  char v93; // al
  __int64 *v94; // r15
  unsigned __int8 *v95; // rax
  unsigned __int8 v96; // si
  bool v97; // zf
  __int64 v98; // rdi
  unsigned int v99; // eax
  __int64 v100; // rcx
  __int64 v101; // rdi
  unsigned __int64 v102; // r13
  __int8 v103; // al
  int v104; // r8d
  int v105; // r9d
  __int64 *v106; // rdi
  __int64 *v107; // rsi
  __int64 v108; // rdx
  __int64 v109; // rax
  __int64 v110; // r14
  __int64 v111; // r13
  __int64 v112; // rdx
  unsigned int v113; // ebx
  __int64 v114; // r15
  char v115; // r12
  __int64 v116; // r13
  __int64 v117; // rbx
  int v118; // eax
  int v119; // eax
  __int64 v120; // rax
  unsigned int v121; // eax
  unsigned int v122; // eax
  int v123; // eax
  __int64 v124; // r14
  unsigned int v125; // r13d
  __int64 v126; // rax
  __int64 v127; // r14
  __int64 *v128; // rax
  __int64 v129; // rdi
  __int64 v130; // rax
  __int64 v131; // rdx
  __int64 v132; // rsi
  __int128 v133; // rax
  unsigned __int64 v134; // rax
  __int64 *v135; // r13
  __int64 *v136; // r15
  __int64 *v137; // rax
  __int64 *v138; // r13
  __int64 v139; // rax
  __int64 *v140; // r13
  __m128i v141; // rax
  __int64 *v142; // r13
  __int64 v143; // r15
  __int128 v144; // rax
  __int64 *v145; // r12
  __int64 v146; // rdx
  __int128 v147; // rax
  __int64 *v148; // rax
  unsigned __int64 v149; // rdx
  __int64 *v150; // rdi
  __int64 v151; // rcx
  __int64 v152; // rax
  unsigned int v153; // eax
  unsigned int v154; // eax
  unsigned int v155; // eax
  __int64 v156; // rcx
  __int64 *v157; // rax
  __int64 v158; // [rsp+18h] [rbp-2C8h]
  __int64 v159; // [rsp+28h] [rbp-2B8h]
  unsigned __int8 v160; // [rsp+33h] [rbp-2ADh]
  int v161; // [rsp+34h] [rbp-2ACh]
  unsigned int v162; // [rsp+38h] [rbp-2A8h]
  unsigned int v163; // [rsp+40h] [rbp-2A0h]
  unsigned __int32 v164; // [rsp+40h] [rbp-2A0h]
  __int64 v165; // [rsp+40h] [rbp-2A0h]
  __int64 **v166; // [rsp+40h] [rbp-2A0h]
  __int64 v167; // [rsp+48h] [rbp-298h]
  __int64 *v168; // [rsp+48h] [rbp-298h]
  unsigned __int32 v169; // [rsp+50h] [rbp-290h]
  int v170; // [rsp+50h] [rbp-290h]
  unsigned int v171; // [rsp+50h] [rbp-290h]
  unsigned __int64 v172; // [rsp+50h] [rbp-290h]
  const void **v173; // [rsp+50h] [rbp-290h]
  __int128 v174; // [rsp+50h] [rbp-290h]
  char v175; // [rsp+60h] [rbp-280h]
  __int64 *v176; // [rsp+60h] [rbp-280h]
  __int128 v177; // [rsp+60h] [rbp-280h]
  __int64 v178; // [rsp+60h] [rbp-280h]
  __int64 v179; // [rsp+60h] [rbp-280h]
  __int64 v180; // [rsp+60h] [rbp-280h]
  char v181; // [rsp+70h] [rbp-270h]
  __int64 v182; // [rsp+70h] [rbp-270h]
  __int64 v183; // [rsp+70h] [rbp-270h]
  __int64 v184; // [rsp+70h] [rbp-270h]
  unsigned int v185; // [rsp+70h] [rbp-270h]
  int v186; // [rsp+70h] [rbp-270h]
  __int64 v187; // [rsp+70h] [rbp-270h]
  unsigned __int64 v188; // [rsp+78h] [rbp-268h]
  __m128i v189; // [rsp+80h] [rbp-260h] BYREF
  __int64 v190; // [rsp+90h] [rbp-250h]
  __int64 v191; // [rsp+98h] [rbp-248h]
  __m128i v192; // [rsp+A0h] [rbp-240h]
  __int128 v193; // [rsp+B0h] [rbp-230h]
  __int64 v194; // [rsp+C0h] [rbp-220h]
  unsigned __int64 v195; // [rsp+C8h] [rbp-218h]
  __m128i v196; // [rsp+D0h] [rbp-210h]
  unsigned int v197; // [rsp+ECh] [rbp-1F4h] BYREF
  int v198; // [rsp+F0h] [rbp-1F0h] BYREF
  unsigned int v199; // [rsp+F4h] [rbp-1ECh] BYREF
  int v200; // [rsp+F8h] [rbp-1E8h] BYREF
  int v201; // [rsp+FCh] [rbp-1E4h] BYREF
  int v202; // [rsp+100h] [rbp-1E0h] BYREF
  int v203; // [rsp+104h] [rbp-1DCh] BYREF
  int v204; // [rsp+108h] [rbp-1D8h] BYREF
  unsigned int v205; // [rsp+10Ch] [rbp-1D4h] BYREF
  int v206; // [rsp+110h] [rbp-1D0h] BYREF
  unsigned int v207; // [rsp+114h] [rbp-1CCh] BYREF
  __int64 v208; // [rsp+118h] [rbp-1C8h] BYREF
  unsigned int v209; // [rsp+120h] [rbp-1C0h] BYREF
  const void **v210; // [rsp+128h] [rbp-1B8h]
  int v211; // [rsp+130h] [rbp-1B0h] BYREF
  __int64 v212; // [rsp+138h] [rbp-1A8h]
  __int64 v213; // [rsp+140h] [rbp-1A0h] BYREF
  int v214; // [rsp+148h] [rbp-198h]
  char v215; // [rsp+150h] [rbp-190h]
  _BYTE v216[16]; // [rsp+160h] [rbp-180h] BYREF
  __int64 (__fastcall *v217)(_BYTE *, __int64, int); // [rsp+170h] [rbp-170h]
  __int64 (__fastcall *v218)(__int64, __int64, unsigned int *); // [rsp+178h] [rbp-168h]
  _BYTE v219[16]; // [rsp+180h] [rbp-160h] BYREF
  __int64 (__fastcall *v220)(_BYTE *, __int64, int); // [rsp+190h] [rbp-150h]
  __int64 (__fastcall *v221)(__int64, unsigned int *, int *); // [rsp+198h] [rbp-148h]
  void *v222; // [rsp+1A0h] [rbp-140h] BYREF
  __int64 v223; // [rsp+1A8h] [rbp-138h]
  _BYTE v224[32]; // [rsp+1B0h] [rbp-130h] BYREF
  __int64 v225; // [rsp+1D0h] [rbp-110h] BYREF
  __int64 v226; // [rsp+1D8h] [rbp-108h]
  __int64 v227; // [rsp+1E0h] [rbp-100h]
  void *v228; // [rsp+200h] [rbp-E0h] BYREF
  unsigned int v229; // [rsp+208h] [rbp-D8h]
  char v230; // [rsp+230h] [rbp-B0h]
  void *s; // [rsp+240h] [rbp-A0h] BYREF
  __int64 v232; // [rsp+248h] [rbp-98h]
  __int64 (__fastcall *v233)(_QWORD *, __int64, int); // [rsp+250h] [rbp-90h] BYREF
  __int64 v234; // [rsp+258h] [rbp-88h]
  int v235; // [rsp+260h] [rbp-80h]
  _BYTE v236[120]; // [rsp+268h] [rbp-78h] BYREF

  v6 = a2;
  v7 = *(__int64 **)(a2 + 32);
  v8 = v7[5];
  v9 = *((unsigned int *)v7 + 12);
  v10 = _mm_loadu_si128((const __m128i *)v7);
  v11 = *v7;
  v12 = *(_QWORD *)(v8 + 40);
  v13 = *((unsigned int *)v7 + 2);
  v14 = _mm_loadu_si128((const __m128i *)(v7 + 5));
  v191 = *v7;
  v15 = (char *)(16 * v9 + v12);
  v192 = v10;
  v16 = *v15;
  v17 = (const void **)*((_QWORD *)v15 + 1);
  v193 = (__int128)v14;
  LOBYTE(v209) = v16;
  v210 = v17;
  if ( (_DWORD)v9 == (_DWORD)v13 && v11 == v8 )
    return v192.m128i_i64[0];
  if ( v16 )
  {
    if ( (unsigned __int8)(v16 - 14) > 0x5Fu )
      goto LABEL_5;
  }
  else
  {
    v189.m128i_i32[0] = v13;
    v18 = sub_1F58D20((__int64)&v209);
    v13 = v189.m128i_u32[0];
    if ( !v18 )
      goto LABEL_5;
  }
  v189.m128i_i32[0] = v13;
  result = (__int64)sub_1FA8C50((__int64)a1, a2, *(double *)v10.m128i_i64, *(double *)v14.m128i_i64, a5);
  if ( result )
    return result;
  if ( (unsigned __int8)sub_1D16620(v191, (__int64 *)a2) )
    return v193;
  if ( (unsigned __int8)sub_1D16620(v8, (__int64 *)a2) )
    return v192.m128i_i64[0];
  if ( (unsigned __int8)sub_1D16340(v191, a2) )
  {
    v39 = *(void **)(a2 + 72);
    v40 = *a1;
    v41 = *(const void ***)(*(_QWORD *)(v191 + 40) + 16LL * v189.m128i_u32[0] + 8);
    v42 = *(unsigned __int8 *)(*(_QWORD *)(v191 + 40) + 16LL * v189.m128i_u32[0]);
    s = v39;
    if ( v39 )
      sub_1623A60((__int64)&s, (__int64)v39, 2);
    LODWORD(v232) = *(_DWORD *)(v6 + 64);
    result = sub_1D389D0((__int64)v40, (__int64)&s, v42, v41, 0, 0, v10, *(double *)v14.m128i_i64, a5);
    v29 = s;
    if ( !s )
      return result;
    goto LABEL_27;
  }
  v43 = sub_1D16340(v8, a2);
  v13 = v189.m128i_u32[0];
  if ( v43 )
  {
    v44 = (unsigned __int8 *)(*(_QWORD *)(v8 + 40) + 16 * v9);
    v45 = (const void **)*((_QWORD *)v44 + 1);
    v46 = *v44;
    s = *(void **)(a2 + 72);
    v47 = (__int64)*a1;
    if ( s )
      sub_1F6CA20((__int64 *)&s);
    LODWORD(v232) = *(_DWORD *)(a2 + 64);
    v192.m128i_i64[0] = sub_1D389D0(v47, (__int64)&s, v46, v45, 0, 0, v10, *(double *)v14.m128i_i64, a5);
    *(_QWORD *)&v193 = v48;
    sub_17CD270((__int64 *)&s);
    return v192.m128i_i64[0];
  }
  if ( *(_WORD *)(v191 + 24) != 110 )
    goto LABEL_5;
  if ( *(_WORD *)(v8 + 24) != 110 )
    goto LABEL_7;
  if ( !(_BYTE)v209 || !a1[1][(unsigned __int8)v209 + 15] )
  {
    v20 = *(unsigned __int16 *)(v8 + 24);
LABEL_8:
    v21 = 0;
    if ( v20 == 32 )
      v21 = v8;
    v189.m128i_i64[0] = v21;
    goto LABEL_11;
  }
  v169 = v189.m128i_i32[0];
  v189.m128i_i8[0] = sub_1D16620(**(_QWORD **)(v191 + 32), (__int64 *)a2);
  v49 = sub_1D16620(*(_QWORD *)(*(_QWORD *)(v191 + 32) + 40LL), (__int64 *)a2);
  v181 = sub_1D16620(**(_QWORD **)(v8 + 32), (__int64 *)a2);
  v50 = sub_1D16620(*(_QWORD *)(*(_QWORD *)(v8 + 32) + 40LL), (__int64 *)a2);
  v13 = v169;
  v175 = (v181 ^ v50) & (v189.m128i_i8[0] ^ v49);
  if ( !v175 )
    goto LABEL_5;
  v51 = sub_1D15970(&v209);
  v53 = v169;
  v54 = v51;
  s = &v233;
  v232 = 0x400000000LL;
  if ( (unsigned __int64)v51 > 4 )
  {
    v164 = v169;
    v171 = v51;
    sub_16CD150((__int64)&s, &v233, v51, 4, v52, v53);
    v53 = v164;
    v54 = v171;
  }
  v163 = v53;
  LODWORD(v232) = v54;
  v170 = v54;
  memset(s, 0, 4LL * v54);
  v55 = v191;
  v56 = 0;
  v13 = v163;
  while ( v170 != (_DWORD)v56 )
  {
    v61 = *(_DWORD *)(*(_QWORD *)(v55 + 88) + 4 * v56);
    v62 = *(_DWORD *)(*(_QWORD *)(v8 + 88) + 4 * v56);
    if ( v61 >= 0 )
    {
      v57 = v170 > v61;
      if ( v62 >= 0 )
      {
        v58 = v57 == v189.m128i_i8[0];
        v59 = v170 > v62 == v181;
        goto LABEL_76;
      }
      if ( v57 != v189.m128i_i8[0] )
        goto LABEL_112;
    }
    else if ( v62 >= 0 )
    {
      v59 = v170 > v62 == v181;
      if ( v170 > v62 != v181 )
      {
        v58 = v175;
LABEL_76:
        if ( v58 == v59 )
          goto LABEL_129;
        if ( v59 )
LABEL_112:
          v60 = v61 % v170;
        else
          v60 = v170 + v62 % v170;
        *((_DWORD *)s + v56) = v60;
        goto LABEL_80;
      }
    }
    *((_DWORD *)s + v56) = -1;
LABEL_80:
    ++v56;
  }
  v83 = *(__int64 **)(v191 + 32);
  if ( v189.m128i_i8[0] )
  {
    v84 = v83[5];
    v85 = v83[6];
  }
  else
  {
    v84 = *v83;
    v85 = v83[1];
  }
  v178 = v84;
  v172 = v85;
  v86 = *(_QWORD *)(v8 + 32);
  if ( v181 )
  {
    a5 = _mm_loadu_si128((const __m128i *)(v86 + 40));
    v189 = a5;
  }
  else
  {
    v189 = _mm_loadu_si128((const __m128i *)v86);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64 *, void *, _QWORD, _QWORD, const void **))(*a1[1] + 336))(
         a1[1],
         s,
         (unsigned int)v232,
         v209,
         v210) )
  {
    goto LABEL_124;
  }
  v91 = v172;
  v92 = v178;
  v178 = v189.m128i_i64[0];
  v196 = _mm_load_si128(&v189);
  v194 = v92;
  v189.m128i_i64[0] = v92;
  v172 = v196.m128i_u32[2] | v172 & 0xFFFFFFFF00000000LL;
  v195 = v91;
  v189.m128i_i64[1] = (unsigned int)v91 | v189.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  sub_1F806A0((int *)s, v232);
  v93 = (*(__int64 (__fastcall **)(__int64 *, void *, _QWORD, _QWORD, const void **))(*a1[1] + 336))(
          a1[1],
          s,
          (unsigned int)v232,
          v209,
          v210);
  v13 = v163;
  if ( v93 )
  {
LABEL_124:
    v87 = (__int64)*a1;
    v88 = s;
    v89 = (unsigned int)v232;
    v228 = *(void **)(v6 + 72);
    if ( v228 )
      sub_1F6CA20((__int64 *)&v228);
    v229 = *(_DWORD *)(v6 + 64);
    v192.m128i_i64[0] = (__int64)sub_1D41320(
                                   v87,
                                   v209,
                                   v210,
                                   (__int64)&v228,
                                   v178,
                                   v172,
                                   *(double *)v10.m128i_i64,
                                   *(double *)v14.m128i_i64,
                                   a5,
                                   v189.m128i_i64[0],
                                   v189.m128i_i64[1],
                                   v88,
                                   v89);
    *(_QWORD *)&v193 = v90;
    sub_17CD270((__int64 *)&v228);
    result = v192.m128i_i64[0];
    if ( s != &v233 )
    {
      v192.m128i_i64[0] = v193;
      *(_QWORD *)&v193 = result;
      _libc_free((unsigned __int64)s);
      return v193;
    }
    return result;
  }
LABEL_129:
  if ( s != &v233 )
  {
    v189.m128i_i32[0] = v13;
    _libc_free((unsigned __int64)s);
    v13 = v189.m128i_u32[0];
  }
LABEL_5:
  if ( (v19 = *(unsigned __int16 *)(v191 + 24), v19 != 10) && v19 != 32 || (*(_BYTE *)(v191 + 26) & 8) != 0 )
  {
LABEL_7:
    v20 = *(unsigned __int16 *)(v8 + 24);
    if ( v20 == 10 )
      goto LABEL_23;
    goto LABEL_8;
  }
  v25 = *(unsigned __int16 *)(v8 + 24);
  if ( v25 == 10 || (v189.m128i_i64[0] = 0, v25 == 32) )
  {
    if ( (*(_BYTE *)(v8 + 26) & 8) != 0 )
    {
LABEL_23:
      v189.m128i_i64[0] = v8;
      goto LABEL_11;
    }
    v26 = *(void **)(v6 + 72);
    v27 = (__int64)*a1;
    s = v26;
    if ( v26 )
      sub_1623A60((__int64)&s, (__int64)v26, 2);
    LODWORD(v232) = *(_DWORD *)(v6 + 64);
    result = sub_1D392A0(v27, 119, (__int64)&s, v209, v210, v191, v10, *(double *)v14.m128i_i64, a5, v8);
    v29 = s;
    if ( !s )
      return result;
LABEL_27:
    v192.m128i_i64[0] = v28;
    *(_QWORD *)&v193 = result;
    sub_161E7C0((__int64)&s, (__int64)v29);
    return v193;
  }
LABEL_11:
  v22 = (__int64)*a1;
  v192.m128i_i64[0] = v191;
  v192.m128i_i64[1] = v13 | v192.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  if ( sub_1D23600(v22, v191) )
  {
    v23 = (__int64)*a1;
    *(_QWORD *)&v193 = v8;
    *((_QWORD *)&v193 + 1) = v9 | *((_QWORD *)&v193 + 1) & 0xFFFFFFFF00000000LL;
    if ( !sub_1D23600(v23, v8) )
    {
      v30 = *(void **)(v6 + 72);
      v31 = *a1;
      s = v30;
      if ( v30 )
        sub_1623A60((__int64)&s, (__int64)v30, 2);
      LODWORD(v232) = *(_DWORD *)(v6 + 64);
      result = (__int64)sub_1D332F0(
                          v31,
                          119,
                          (__int64)&s,
                          v209,
                          v210,
                          0,
                          *(double *)v10.m128i_i64,
                          *(double *)v14.m128i_i64,
                          a5,
                          v193,
                          *((unsigned __int64 *)&v193 + 1),
                          *(_OWORD *)&v192);
      v29 = s;
      if ( !s )
        return result;
      goto LABEL_27;
    }
  }
  *(_QWORD *)&v193 = v8;
  *((_QWORD *)&v193 + 1) = *((_QWORD *)&v193 + 1) & 0xFFFFFFFF00000000LL | v9;
  if ( sub_1D185B0(v8) )
    return v192.m128i_i64[0];
  if ( sub_1D188A0(v193) )
    return v193;
  result = (__int64)sub_1F77C50(a1, v6, *(double *)v10.m128i_i64, *(double *)v14.m128i_i64, a5);
  if ( result )
    return result;
  if ( v189.m128i_i64[0] )
  {
    v32 = *(_QWORD *)(v189.m128i_i64[0] + 88);
    v33 = *a1;
    v34 = *(_DWORD *)(v32 + 32);
    v229 = v34;
    if ( v34 <= 0x40 )
    {
      v35 = *(_QWORD *)(v32 + 24);
      goto LABEL_37;
    }
    sub_16A4FD0((__int64)&v228, (const void **)(v32 + 24));
    v34 = v229;
    if ( v229 <= 0x40 )
    {
      v35 = (__int64)v228;
LABEL_37:
      v36 = (void *)(~v35 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v34));
      v228 = v36;
    }
    else
    {
      sub_16A8F40((__int64 *)&v228);
      v34 = v229;
      v36 = v228;
    }
    LODWORD(v232) = v34;
    s = v36;
    v229 = 0;
    v37 = sub_1D1F940((__int64)v33, v192.m128i_i64[0], v192.m128i_i64[1], (__int64)&s, 0);
    if ( (unsigned int)v232 > 0x40 && s )
      j_j___libc_free_0_0(s);
    if ( v229 > 0x40 && v228 )
      j_j___libc_free_0_0(v228);
    if ( v37 )
      return v193;
  }
  result = sub_1F86240(
             (__int64)a1,
             v192.m128i_i64[0],
             v192.m128i_u32[2],
             v193,
             DWORD2(v193),
             v6,
             v10,
             *(double *)v14.m128i_i64,
             a5);
  if ( result )
    return result;
  if ( *((_BYTE *)a1 + 24) )
  {
    v63 = *(unsigned __int8 **)(v6 + 40);
    v64 = *v63;
    v65 = (const void **)*((_QWORD *)v63 + 1);
    v189.m128i_i64[0] = *v63;
    if ( v64 == 5 )
    {
      v66 = a1[1];
      if ( v66[20] )
      {
        if ( (*((_BYTE *)v66 + 3844) & 0xFB) == 0 && *(_WORD *)(v191 + 24) == 119 )
        {
          v67 = *(__int64 **)(v191 + 32);
          v68 = 8;
          p_s = &s;
          v70 = *v67;
          v71 = v67[5];
          while ( v68 )
          {
            *(_DWORD *)p_s = 0;
            p_s = (void **)((char *)p_s + 4);
            --v68;
          }
          if ( *(_WORD *)(v8 + 24) == 119 && *(_DWORD *)(v70 + 56) == 2 && *(_DWORD *)(v71 + 56) == 2 )
          {
            v184 = v71;
            if ( (unsigned __int8)sub_1F6C3D0(v70, (__int64)&s)
              && (unsigned __int8)sub_1F6C3D0(v184, (__int64)&s)
              && (unsigned __int8)sub_1F6C3D0(**(_QWORD **)(v8 + 32), (__int64)&s)
              && (unsigned __int8)sub_1F6C3D0(*(_QWORD *)(*(_QWORD *)(v8 + 32) + 40LL), (__int64)&s) )
            {
LABEL_100:
              v74 = s;
              if ( s == (void *)v232 && s == v233 && s == (void *)v234 )
              {
                v228 = *(void **)(v6 + 72);
                if ( v228 )
                {
                  sub_1F6CA20((__int64 *)&v228);
                  v74 = s;
                }
                v75 = *a1;
                v229 = *(_DWORD *)(v6 + 64);
                v76 = sub_1D309E0(
                        v75,
                        127,
                        (__int64)&v228,
                        v189.m128i_u32[0],
                        v65,
                        0,
                        *(double *)v10.m128i_i64,
                        *(double *)v14.m128i_i64,
                        *(double *)a5.m128i_i64,
                        (unsigned __int64)v74);
                v188 = v77;
                v183 = v76;
                v176 = *a1;
                v78 = sub_1F6BF40((__int64)a1, v189.m128i_u32[0], (__int64)v65);
                *(_QWORD *)&v80 = sub_1D38BB0(
                                    (__int64)v176,
                                    16,
                                    (__int64)&v228,
                                    v78,
                                    v79,
                                    0,
                                    v10,
                                    *(double *)v14.m128i_i64,
                                    a5,
                                    0);
                v177 = v80;
                if ( sub_1F6C880((__int64)a1[1], 0x7Du, 5u) )
                {
                  v82 = sub_1D332F0(
                          *a1,
                          125,
                          (__int64)&v228,
                          v189.m128i_u32[0],
                          v65,
                          0,
                          *(double *)v10.m128i_i64,
                          *(double *)v14.m128i_i64,
                          a5,
                          v183,
                          v188,
                          v177);
                }
                else if ( sub_1F6C880(v81, 0x7Eu, 5u) )
                {
                  v82 = sub_1D332F0(
                          *a1,
                          126,
                          (__int64)&v228,
                          v189.m128i_u32[0],
                          v65,
                          0,
                          *(double *)v10.m128i_i64,
                          *(double *)v14.m128i_i64,
                          a5,
                          v183,
                          v188,
                          v177);
                }
                else
                {
                  v168 = *a1;
                  *(_QWORD *)&v147 = sub_1D332F0(
                                       *a1,
                                       124,
                                       (__int64)&v228,
                                       v189.m128i_u32[0],
                                       v65,
                                       0,
                                       *(double *)v10.m128i_i64,
                                       *(double *)v14.m128i_i64,
                                       a5,
                                       v183,
                                       v188,
                                       v177);
                  v174 = v147;
                  v148 = sub_1D332F0(
                           *a1,
                           122,
                           (__int64)&v228,
                           v189.m128i_u32[0],
                           v65,
                           0,
                           *(double *)v10.m128i_i64,
                           *(double *)v14.m128i_i64,
                           a5,
                           v183,
                           v188,
                           v177);
                  v82 = sub_1D332F0(
                          v168,
                          119,
                          (__int64)&v228,
                          v189.m128i_u32[0],
                          v65,
                          0,
                          *(double *)v10.m128i_i64,
                          *(double *)v14.m128i_i64,
                          a5,
                          (__int64)v148,
                          v149,
                          v174);
                }
                v189.m128i_i64[0] = (__int64)v82;
                sub_17CD270((__int64 *)&v228);
                result = v189.m128i_i64[0];
                if ( v189.m128i_i64[0] )
                  return result;
              }
            }
          }
          else
          {
            v182 = v71;
            if ( (unsigned __int8)sub_1F6C3D0(v8, (__int64)&s)
              && (unsigned __int8)sub_1F6C3D0(v182, (__int64)&s)
              && *(_WORD *)(v72 + 24) == 119
              && (unsigned __int8)sub_1F6C3D0(**(_QWORD **)(v72 + 32), (__int64)&s)
              && (unsigned __int8)sub_1F6C3D0(*(_QWORD *)(*(_QWORD *)(v73 + 32) + 40LL), (__int64)&s) )
            {
              goto LABEL_100;
            }
          }
        }
      }
    }
  }
  result = (__int64)sub_1F7E190((__int64)a1, v6, v191, v8, 1, *(double *)v10.m128i_i64, *(double *)v14.m128i_i64, a5);
  if ( result )
    return result;
  s = *(void **)(v6 + 72);
  if ( s )
    sub_1F6CA20((__int64 *)&s);
  LODWORD(v232) = *(_DWORD *)(v6 + 64);
  v189.m128i_i64[0] = (__int64)sub_1F82ED0(
                                 (__int64 *)a1,
                                 0x77u,
                                 (__int64)&s,
                                 v192.m128i_i64[0],
                                 v192.m128i_u64[1],
                                 *(double *)v10.m128i_i64,
                                 *(double *)v14.m128i_i64,
                                 a5,
                                 v38,
                                 v193);
  sub_17CD270((__int64 *)&s);
  result = v189.m128i_i64[0];
  if ( v189.m128i_i64[0] )
    return result;
  if ( *(_WORD *)(v191 + 24) == 118 )
  {
    v139 = *(_QWORD *)(v191 + 48);
    if ( v139 )
    {
      if ( !*(_QWORD *)(v139 + 32) )
      {
        v234 = (__int64)sub_1F6BF90;
        v233 = sub_1F6BC00;
        if ( sub_1D16BF0(
               *(_QWORD *)(*(_QWORD *)(v191 + 32) + 40LL),
               *(_QWORD *)(*(_QWORD *)(v191 + 32) + 48LL),
               v193,
               DWORD2(v193),
               (__int64)&s) )
        {
          sub_A17130((__int64)&s);
          v140 = *a1;
          v189.m128i_i64[0] = *(_QWORD *)(*(_QWORD *)(v191 + 32) + 40LL);
          sub_1F80610((__int64)&s, v193);
          v141.m128i_i64[0] = (__int64)sub_1D32920(
                                         v140,
                                         0x77u,
                                         (__int64)&s,
                                         v209,
                                         (__int64)v210,
                                         v8,
                                         *(double *)v10.m128i_i64,
                                         *(double *)v14.m128i_i64,
                                         a5,
                                         v189.m128i_u64[0]);
          v189 = v141;
          sub_17CD270((__int64 *)&s);
          if ( v189.m128i_i64[0] )
          {
            v142 = *a1;
            v143 = *(_QWORD *)(v191 + 32);
            sub_1F80610((__int64)&s, v192.m128i_i64[0]);
            *(_QWORD *)&v144 = sub_1D332F0(
                                 v142,
                                 119,
                                 (__int64)&s,
                                 v209,
                                 v210,
                                 0,
                                 *(double *)v10.m128i_i64,
                                 *(double *)v14.m128i_i64,
                                 a5,
                                 *(_QWORD *)v143,
                                 *(_QWORD *)(v143 + 8),
                                 v193);
            v193 = v144;
            sub_17CD270((__int64 *)&s);
            sub_1F81BC0((__int64)a1, v193);
            v145 = *a1;
            s = *(void **)(v6 + 72);
            if ( s )
              sub_1F6CA20((__int64 *)&s);
            LODWORD(v232) = *(_DWORD *)(v6 + 64);
            v192.m128i_i64[0] = (__int64)sub_1D332F0(
                                           v145,
                                           118,
                                           (__int64)&s,
                                           v209,
                                           v210,
                                           0,
                                           *(double *)v10.m128i_i64,
                                           *(double *)v14.m128i_i64,
                                           a5,
                                           v189.m128i_i64[0],
                                           v189.m128i_u64[1],
                                           v193);
            *(_QWORD *)&v193 = v146;
            sub_17CD270((__int64 *)&s);
            return v192.m128i_i64[0];
          }
        }
        else
        {
          sub_A17130((__int64)&s);
        }
      }
    }
  }
  if ( *(_WORD *)(v8 + 24) == *(_WORD *)(v191 + 24) )
  {
    result = sub_1F868B0((__int64)a1, v6, v10, v14, a5);
    if ( result )
      return result;
  }
  s = *(void **)(v6 + 72);
  if ( s )
    sub_1F6CA20((__int64 *)&s);
  LODWORD(v232) = *(_DWORD *)(v6 + 64);
  v94 = sub_1F806E0(
          (__int64)a1,
          v192.m128i_i64[0],
          v192.m128i_u32[2],
          v193,
          DWORD2(v193),
          (__int64)&s,
          v10,
          *(double *)v14.m128i_i64,
          a5,
          0);
  v191 = (__int64)v94;
  sub_17CD270((__int64 *)&s);
  result = (__int64)v94;
  if ( v94 )
    return result;
  v95 = *(unsigned __int8 **)(v6 + 40);
  v96 = *v95;
  v173 = (const void **)*((_QWORD *)v95 + 1);
  v160 = *v95;
  v162 = *v95;
  if ( (unsigned __int8)(*v95 - 4) > 2u )
    goto LABEL_143;
  v98 = (*a1)[2];
  v167 = v98;
  if ( *((_BYTE *)a1 + 24) )
  {
    if ( !*(_QWORD *)(v98 + 8LL * v96 + 120) || *(_BYTE *)(v98 + 259LL * v96 + 2607) )
      goto LABEL_143;
  }
  v99 = sub_1F6C8D0(v160);
  v101 = *(_QWORD *)(v100 + 32);
  v102 = v99 >> 3;
  v218 = sub_1F6BC10;
  v217 = (__int64 (__fastcall *)(_BYTE *, __int64, int))sub_1F6BC20;
  v221 = sub_1F6BC30;
  LODWORD(v193) = v99 >> 3;
  v220 = (__int64 (__fastcall *)(_BYTE *, __int64, int))sub_1F6BC40;
  v103 = *(_BYTE *)sub_1E0A0C0(v101);
  v230 = 0;
  s = 0;
  v189.m128i_i8[0] = v103;
  v232 = (__int64)v236;
  v233 = (__int64 (__fastcall *)(_QWORD *, __int64, int))v236;
  v222 = v224;
  v235 = 0;
  v234 = 8;
  v223 = 0x400000000LL;
  if ( (unsigned int)v102 > 4 )
    sub_16CD150((__int64)&v222, v224, v102, 8, v104, v105);
  v106 = (__int64 *)v222;
  v107 = 0;
  LODWORD(v223) = v193;
  memset(v222, 0, 8 * v102);
  v192.m128i_i8[0] = 0;
  v165 = 0;
  v185 = 0;
  v179 = 0x7FFFFFFFFFFFFFFFLL;
  v109 = 0;
  v110 = v6;
  while ( 2 )
  {
    if ( (unsigned int)v193 > (unsigned int)v165 )
    {
      v158 = v109;
      sub_1F738B0((__int64)&v213, v110, 0, v165, 0, 1);
      if ( !v215 || (v111 = v213) == 0 )
      {
LABEL_158:
        v6 = v110;
        goto LABEL_159;
      }
      v112 = *(_QWORD *)(v213 + 32);
      v113 = *(_DWORD *)(v112 + 8);
      v159 = *(_QWORD *)v112;
      if ( v158 )
      {
        if ( *(_QWORD *)v112 != v158 || v113 != v185 )
          goto LABEL_158;
        v159 = v158;
        v113 = v185;
      }
      sub_2043720(&v225, v213, *a1);
      v208 = 0;
      if ( v230 )
      {
        v107 = &v225;
        if ( !(unsigned __int8)sub_2043540(&v228, &v225, *a1, &v208) )
          goto LABEL_158;
      }
      else
      {
        v150 = (__int64 *)&v228;
        v151 = 12;
        v107 = &v225;
        while ( v151 )
        {
          *(_DWORD *)v150 = *(_DWORD *)v107;
          v107 = (__int64 *)((char *)v107 + 4);
          v150 = (__int64 *)((char *)v150 + 4);
          --v151;
        }
        v230 = 1;
      }
      v106 = (__int64 *)*(unsigned __int8 *)(v213 + 88);
      v152 = *(_QWORD *)(v213 + 96);
      v186 = v214;
      LOBYTE(v211) = (_BYTE)v106;
      v212 = v152;
      if ( (_BYTE)v106 )
      {
        v153 = sub_1F6C8D0((char)v106);
      }
      else
      {
        v106 = (__int64 *)&v211;
        v153 = sub_1F58D40((__int64)&v211);
      }
      v154 = v153 >> 3;
      if ( v189.m128i_i8[0] )
      {
        v197 = v154;
        v198 = v186;
        if ( !v220 )
          goto LABEL_176;
        v155 = v221((__int64)v219, &v197, &v198);
      }
      else
      {
        v199 = v154;
        v200 = v186;
        if ( !v217 )
          goto LABEL_176;
        v155 = v218((__int64)v216, (__int64)&v199, (unsigned int *)&v200);
      }
      v208 += v155;
      v156 = v208;
      *((_QWORD *)v222 + v165) = v208;
      if ( v156 < v179 )
      {
        if ( v215 )
        {
          v161 = v214;
          v190 = v213;
          if ( !v192.m128i_i8[0] )
            v192.m128i_i8[0] = 1;
        }
        else
        {
          v192.m128i_i8[0] = 0;
        }
      }
      else
      {
        v156 = v179;
      }
      v157 = (__int64 *)v232;
      if ( v233 == (__int64 (__fastcall *)(_QWORD *, __int64, int))v232 )
      {
        v108 = HIDWORD(v234);
        v106 = (__int64 *)(v232 + 8LL * HIDWORD(v234));
        v107 = 0;
        while ( v106 != v157 )
        {
          if ( v111 == *v157 )
            goto LABEL_232;
          if ( *v157 == -2 )
            v107 = v157;
          ++v157;
        }
        if ( v107 )
        {
          *v107 = v111;
          --v235;
          s = (char *)s + 1;
        }
        else
        {
          if ( HIDWORD(v234) >= (unsigned int)v234 )
            goto LABEL_231;
          v108 = (unsigned int)++HIDWORD(v234);
          *v106 = v111;
          s = (char *)s + 1;
        }
      }
      else
      {
LABEL_231:
        v107 = (__int64 *)v111;
        v106 = (__int64 *)&s;
        v187 = v156;
        sub_16CCBA0((__int64)&s, v111);
        v156 = v187;
      }
LABEL_232:
      ++v165;
      v109 = v159;
      v185 = v113;
      v179 = v156;
      continue;
    }
    break;
  }
  v114 = v109;
  v192.m128i_i8[0] = 1;
  v166 = a1;
  v115 = 1;
  v116 = 0;
  while ( (unsigned int)v193 > (unsigned int)v116 )
  {
    v117 = *((_QWORD *)v222 + v116) - v179;
    v203 = v193;
    v204 = v116;
    if ( !v217 )
      goto LABEL_176;
    v107 = (__int64 *)&v203;
    v106 = (__int64 *)v216;
    v118 = v218((__int64)v216, (__int64)&v203, (unsigned int *)&v204);
    v202 = v116;
    v201 = v193;
    v115 &= v118 == v117;
    if ( !v220 )
      goto LABEL_176;
    v107 = (__int64 *)&v201;
    v106 = (__int64 *)v219;
    v119 = v221((__int64)v219, (unsigned int *)&v201, &v202);
    ++v116;
    v192.m128i_i8[0] &= v119 == v117;
    if ( !v115 && !v192.m128i_i8[0] )
    {
      a1 = v166;
      goto LABEL_158;
    }
  }
  a1 = v166;
  v6 = v110;
  v106 = (__int64 *)*(unsigned __int8 *)(v190 + 88);
  v120 = *(_QWORD *)(v190 + 96);
  LOBYTE(v225) = (_BYTE)v106;
  v226 = v120;
  if ( (_BYTE)v106 )
  {
    v121 = sub_1F6C8D0((char)v106);
  }
  else
  {
    v106 = &v225;
    v121 = sub_1F58D40((__int64)&v225);
  }
  v122 = v121 >> 3;
  if ( v189.m128i_i8[0] )
  {
    v205 = v122;
    v206 = v161;
    if ( v220 )
    {
      v123 = v221((__int64)v219, &v205, &v206);
      goto LABEL_182;
    }
LABEL_176:
    sub_4263D6(v106, v107, v108);
  }
  v207 = v122;
  v211 = v161;
  if ( !v217 )
    goto LABEL_176;
  v123 = v218((__int64)v216, (__int64)&v207, (unsigned int *)&v211);
LABEL_182:
  if ( !v123
    && (v189.m128i_i8[0] == v192.m128i_i8[0]
     || !*((_BYTE *)v166 + 24)
     || *(_QWORD *)(v167 + 8LL * v160 + 120) && !*(_BYTE *)(v167 + 259LL * v160 + 2549)) )
  {
    v124 = v190;
    LOBYTE(v211) = 0;
    v125 = sub_1E34390(*(_QWORD *)(v190 + 104));
    LODWORD(v124) = sub_1E340A0(*(_QWORD *)(v124 + 104));
    v126 = sub_1E0A0C0((*v166)[4]);
    if ( (unsigned __int8)sub_1F43CC0(v167, (*v166)[6], v126, v162, (__int64)v173, v124, v125, &v211) )
    {
      if ( (_BYTE)v211 )
      {
        v127 = v190;
        v128 = *v166;
        v225 = 0;
        v226 = 0;
        v227 = 0;
        v129 = *(_QWORD *)(v190 + 104);
        *(_QWORD *)&v193 = v128;
        v130 = sub_1E34390(v129);
        v131 = *(_QWORD *)(v127 + 104);
        v132 = *(_QWORD *)(v127 + 32);
        v213 = *(_QWORD *)(v6 + 72);
        if ( v213 )
        {
          v190 = v132;
          v180 = v131;
          v191 = v130;
          sub_1F6CA20(&v213);
          v132 = v190;
          v131 = v180;
          LODWORD(v130) = v191;
        }
        v214 = *(_DWORD *)(v6 + 64);
        *(_QWORD *)&v133 = sub_1D2B730(
                             (_QWORD *)v193,
                             v162,
                             (__int64)v173,
                             (__int64)&v213,
                             v114,
                             v185,
                             *(_QWORD *)(v132 + 40),
                             *(_QWORD *)(v132 + 48),
                             *(_OWORD *)v131,
                             *(_QWORD *)(v131 + 16),
                             v130,
                             0,
                             (__int64)&v225,
                             0);
        v193 = v133;
        v191 = v133;
        sub_17CD270(&v213);
        v134 = (unsigned __int64)v233;
        if ( v233 == (__int64 (__fastcall *)(_QWORD *, __int64, int))v232 )
          v135 = (__int64 *)((char *)v233 + 8 * HIDWORD(v234));
        else
          v135 = (__int64 *)((char *)v233 + 8 * (unsigned int)v234);
        do
        {
          v136 = (__int64 *)v134;
          if ( v135 == (__int64 *)v134 )
            break;
          v134 += 8LL;
        }
        while ( (unsigned __int64)(*v136 + 2) <= 1 );
LABEL_195:
        if ( v135 != v136 )
        {
          sub_1D44C70((__int64)*v166, *v136, 1, v191, 1u);
          v137 = v136 + 1;
          while ( 1 )
          {
            v136 = v137;
            if ( v135 == v137 )
              break;
            ++v137;
            if ( (unsigned __int64)(*v136 + 2) > 1 )
              goto LABEL_195;
          }
        }
        if ( v189.m128i_i8[0] != v192.m128i_i8[0] )
        {
          v138 = *v166;
          v225 = *(_QWORD *)(v6 + 72);
          if ( v225 )
            sub_1F6CA20(&v225);
          LODWORD(v226) = *(_DWORD *)(v6 + 64);
          v191 = sub_1D309E0(
                   v138,
                   127,
                   (__int64)&v225,
                   v162,
                   v173,
                   0,
                   *(double *)v10.m128i_i64,
                   *(double *)v14.m128i_i64,
                   *(double *)a5.m128i_i64,
                   v193);
          sub_17CD270(&v225);
        }
      }
    }
  }
LABEL_159:
  if ( v222 != v224 )
    _libc_free((unsigned __int64)v222);
  if ( v233 != (__int64 (__fastcall *)(_QWORD *, __int64, int))v232 )
    _libc_free((unsigned __int64)v233);
  if ( v220 )
    v220(v219, (__int64)v219, 3);
  if ( v217 )
    v217(v216, (__int64)v216, 3);
  result = v191;
  if ( !v191 )
  {
LABEL_143:
    v97 = (unsigned __int8)sub_1FB1D70((__int64)a1, v6, 0) == 0;
    result = 0;
    if ( !v97 )
      return v6;
  }
  return result;
}
