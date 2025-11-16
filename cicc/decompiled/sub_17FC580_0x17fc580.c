// Function: sub_17FC580
// Address: 0x17fc580
//
unsigned __int64 __fastcall sub_17FC580(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v3; // rsi
  __int64 v4; // rbx
  __int64 *v5; // rax
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rax
  _QWORD *v10; // rdi
  __int64 *v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rax
  _QWORD *v16; // rdi
  __int64 *v17; // rax
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rax
  _QWORD *v22; // rdi
  __int64 *v23; // rax
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // rax
  _QWORD *v28; // rdi
  _BYTE *v29; // r12
  unsigned __int64 v30; // rcx
  int v31; // eax
  unsigned __int64 v32; // rsi
  char v33; // al
  char *v34; // r13
  _QWORD *v35; // rax
  _BYTE *v36; // r13
  unsigned __int64 v37; // rcx
  int v38; // eax
  unsigned __int64 v39; // rsi
  char v40; // al
  char *v41; // r14
  _QWORD *v42; // rax
  int v43; // r8d
  int v44; // r9d
  size_t v45; // rbx
  __int64 *v46; // r13
  __int64 v47; // r15
  __int64 *v48; // rax
  __int64 v49; // r13
  __int64 v50; // r14
  __int64 v51; // rbx
  __int64 v52; // rax
  __int64 v53; // r13
  int v54; // r8d
  int v55; // r9d
  size_t v56; // rbx
  __int64 *v57; // r13
  __int64 v58; // r15
  __int64 *v59; // rax
  __int64 v60; // r13
  __int64 v61; // r14
  __int64 v62; // rbx
  __int64 v63; // rax
  __int64 v64; // r13
  int v65; // r8d
  int v66; // r9d
  size_t v67; // rbx
  __int64 *v68; // r13
  __int64 v69; // r15
  __int64 *v70; // rax
  __int64 v71; // r13
  __int64 v72; // r14
  __int64 v73; // rbx
  __int64 v74; // rax
  __int64 v75; // r13
  int v76; // r8d
  int v77; // r9d
  size_t v78; // rbx
  __m128i *v79; // r13
  __int64 v80; // r15
  __int64 *v81; // rax
  __int64 v82; // r13
  void *v83; // r14
  __int64 v84; // rbx
  __int64 v85; // rax
  __int64 v86; // r13
  __int64 v87; // rcx
  __int64 v88; // rax
  int v89; // r8d
  int v90; // r9d
  size_t v91; // rbx
  __m128i *v92; // r12
  __int64 v93; // rax
  __int64 v94; // r12
  __int64 v95; // r13
  __int64 v96; // rbx
  __int64 v97; // rax
  __int64 v98; // r12
  __int64 v99; // rcx
  __int64 v100; // rax
  int v101; // r8d
  int v102; // r9d
  size_t v103; // rbx
  __m128i *v104; // r12
  __int64 v105; // r15
  __int64 *v106; // rax
  __int64 v107; // r13
  __int64 v108; // r14
  __int64 v109; // rbx
  __int64 v110; // rax
  __int64 v111; // r13
  __int64 *v112; // r13
  int j; // ebx
  const char *v114; // r14
  __int64 v115; // rcx
  __int64 v116; // rax
  int v117; // r8d
  int v118; // r9d
  size_t v119; // rbx
  __m128i *v120; // r13
  __int64 v121; // rax
  __int64 v122; // r13
  __int64 v123; // r14
  __int64 v124; // rbx
  __int64 v125; // rax
  __int64 v126; // r13
  __int64 v127; // rax
  __m128i *v128; // rdi
  __int64 v129; // rbx
  __int64 v130; // r12
  __int64 *v131; // rax
  __int64 v132; // r13
  __int64 v133; // rax
  __int64 v134; // r12
  __int64 v135; // rax
  _QWORD *v136; // rdi
  __int64 v137; // rbx
  __int64 *v138; // rax
  __int64 v139; // r12
  __int64 v140; // rax
  __int64 v141; // r12
  __int64 v142; // rax
  _QWORD *v143; // rdi
  __int64 v144; // r12
  __int64 *v145; // rax
  __int64 v146; // rbx
  __int64 v147; // rax
  __int64 v148; // r12
  __int64 v149; // rax
  _QWORD *v150; // rdi
  __int64 v151; // r12
  __int64 *v152; // rax
  __int64 v153; // rbx
  __int64 v154; // rax
  __int64 v155; // r12
  __int64 v156; // rax
  _QWORD *v157; // rdi
  __int64 v158; // rbx
  __int64 v159; // r12
  __int64 *v160; // rax
  __int64 v161; // r13
  __int64 v162; // rax
  __int64 v163; // r12
  __int64 v164; // rax
  _QWORD *v165; // rdi
  __int64 v166; // rbx
  __int64 v167; // r12
  __int64 *v168; // rax
  __int64 v169; // r13
  __int64 v170; // rax
  __int64 v171; // r12
  __int64 v172; // rax
  _QWORD *v173; // rdi
  __int64 v174; // rbx
  __int64 v175; // r12
  __int64 *v176; // rax
  __int64 v177; // r13
  __int64 v178; // rax
  __int64 v179; // r12
  unsigned __int64 result; // rax
  __int64 v181; // rsi
  _QWORD *v182; // rdi
  unsigned __int64 v183; // rcx
  __int8 *v184; // r15
  int v185; // eax
  unsigned __int64 v186; // rdi
  __int8 v187; // al
  char *v188; // rsi
  __m128i *v189; // r8
  _QWORD *v190; // rax
  __m128i *v191; // rax
  __int64 v192; // rcx
  size_t v193; // rdx
  __int64 v194; // rcx
  __m128i *v195; // rax
  int v196; // r8d
  int v197; // r9d
  size_t v198; // rdx
  __m128i *v199; // rax
  __int64 v200; // rax
  __int64 v201; // r14
  void *v202; // r15
  __int64 v203; // rax
  __int64 v204; // r14
  __int64 v205; // rax
  __m128i *v206; // rdi
  __m128i *v207; // rdi
  __int64 v208; // rax
  _QWORD *v209; // rdi
  _BYTE *v210; // rdi
  _QWORD *v211; // rdi
  char *v212; // rdi
  _BYTE *v213; // rdi
  _BYTE *v214; // rdi
  _BYTE *v215; // rdi
  __int64 *v216; // rdi
  __m128i *v217; // rdi
  __m128i *src; // [rsp+8h] [rbp-408h]
  size_t n; // [rsp+28h] [rbp-3E8h]
  int na; // [rsp+28h] [rbp-3E8h]
  __int64 v221; // [rsp+50h] [rbp-3C0h]
  __m128i v222; // [rsp+90h] [rbp-380h]
  __int64 i; // [rsp+B0h] [rbp-360h]
  __int64 *v224; // [rsp+C0h] [rbp-350h]
  _QWORD **v225; // [rsp+D8h] [rbp-338h]
  __int64 v226; // [rsp+E0h] [rbp-330h]
  __int64 v227; // [rsp+E0h] [rbp-330h]
  __int64 v228; // [rsp+E0h] [rbp-330h]
  __int64 v230; // [rsp+F8h] [rbp-318h] BYREF
  _QWORD v231[2]; // [rsp+100h] [rbp-310h] BYREF
  _QWORD v232[2]; // [rsp+110h] [rbp-300h] BYREF
  _QWORD v233[2]; // [rsp+120h] [rbp-2F0h] BYREF
  _QWORD v234[2]; // [rsp+130h] [rbp-2E0h] BYREF
  _QWORD v235[2]; // [rsp+140h] [rbp-2D0h] BYREF
  __int64 v236; // [rsp+150h] [rbp-2C0h] BYREF
  _QWORD v237[2]; // [rsp+160h] [rbp-2B0h] BYREF
  _QWORD v238[2]; // [rsp+170h] [rbp-2A0h] BYREF
  __m128i *v239; // [rsp+180h] [rbp-290h] BYREF
  __int64 v240; // [rsp+188h] [rbp-288h]
  __m128i v241; // [rsp+190h] [rbp-280h] BYREF
  _QWORD v242[2]; // [rsp+1A0h] [rbp-270h] BYREF
  __int64 v243; // [rsp+1B0h] [rbp-260h] BYREF
  _BYTE *v244; // [rsp+1C0h] [rbp-250h] BYREF
  __int64 v245; // [rsp+1C8h] [rbp-248h]
  _BYTE v246[32]; // [rsp+1D0h] [rbp-240h] BYREF
  _BYTE *v247; // [rsp+1F0h] [rbp-220h] BYREF
  __int64 v248; // [rsp+1F8h] [rbp-218h]
  _BYTE v249[32]; // [rsp+200h] [rbp-210h] BYREF
  _BYTE *v250; // [rsp+220h] [rbp-1F0h] BYREF
  __int64 v251; // [rsp+228h] [rbp-1E8h]
  _BYTE v252[32]; // [rsp+230h] [rbp-1E0h] BYREF
  _BYTE *v253; // [rsp+250h] [rbp-1C0h] BYREF
  __int64 v254; // [rsp+258h] [rbp-1B8h]
  _BYTE v255[32]; // [rsp+260h] [rbp-1B0h] BYREF
  __m128i *v256; // [rsp+280h] [rbp-190h] BYREF
  size_t v257; // [rsp+288h] [rbp-188h]
  __m128i v258; // [rsp+290h] [rbp-180h] BYREF
  __int64 v259; // [rsp+2A0h] [rbp-170h]
  void *v260; // [rsp+2B0h] [rbp-160h] BYREF
  size_t v261; // [rsp+2B8h] [rbp-158h]
  __m128i v262; // [rsp+2C0h] [rbp-150h] BYREF
  __int64 v263; // [rsp+2D0h] [rbp-140h]
  __int64 v264; // [rsp+2D8h] [rbp-138h]
  __int64 v265; // [rsp+2E0h] [rbp-130h]
  _QWORD v266[3]; // [rsp+2F0h] [rbp-120h] BYREF
  _QWORD *v267; // [rsp+308h] [rbp-108h]
  __int64 v268; // [rsp+310h] [rbp-100h]
  int v269; // [rsp+318h] [rbp-F8h]
  __int64 v270; // [rsp+320h] [rbp-F0h]
  __int64 v271; // [rsp+328h] [rbp-E8h]
  char *v272; // [rsp+340h] [rbp-D0h] BYREF
  __int64 v273; // [rsp+348h] [rbp-C8h]
  _BYTE v274[64]; // [rsp+350h] [rbp-C0h] BYREF
  void *v275; // [rsp+390h] [rbp-80h] BYREF
  size_t v276; // [rsp+398h] [rbp-78h]
  __int64 v277; // [rsp+3A0h] [rbp-70h] BYREF
  __int64 v278; // [rsp+3A8h] [rbp-68h]
  __int64 v279; // [rsp+3B0h] [rbp-60h]

  v3 = (_QWORD *)*a2;
  memset(v266, 0, sizeof(v266));
  v267 = v3;
  v268 = 0;
  v269 = 0;
  v270 = 0;
  v271 = 0;
  v230 = 0;
  v230 = sub_1563AB0(&v230, v3, -1, 30);
  v4 = sub_16471D0(v267, 0);
  v5 = (__int64 *)sub_1643270(v267);
  v6 = v230;
  v275 = &v277;
  v277 = v4;
  v276 = 0x100000001LL;
  v7 = sub_1644EA0(v5, &v277, 1, 0);
  v8 = sub_1632080((__int64)a2, (__int64)"__tsan_func_entry", 17, v7, v6);
  if ( v275 != &v277 )
    _libc_free((unsigned __int64)v275);
  v9 = sub_1B28080(v8);
  v10 = v267;
  a1[22] = v9;
  v11 = (__int64 *)sub_1643270(v10);
  v12 = v230;
  v275 = &v277;
  v276 = 0;
  v13 = sub_1644EA0(v11, &v277, 0, 0);
  v14 = sub_1632080((__int64)a2, (__int64)"__tsan_func_exit", 16, v13, v12);
  if ( v275 != &v277 )
    _libc_free((unsigned __int64)v275);
  v15 = sub_1B28080(v14);
  v16 = v267;
  a1[23] = v15;
  v17 = (__int64 *)sub_1643270(v16);
  v18 = v230;
  v275 = &v277;
  v276 = 0;
  v19 = sub_1644EA0(v17, &v277, 0, 0);
  v20 = sub_1632080((__int64)a2, (__int64)"__tsan_ignore_thread_begin", 26, v19, v18);
  if ( v275 != &v277 )
    _libc_free((unsigned __int64)v275);
  v21 = sub_1B28080(v20);
  v22 = v267;
  a1[24] = v21;
  v23 = (__int64 *)sub_1643270(v22);
  v24 = v230;
  v275 = &v277;
  v276 = 0;
  v25 = sub_1644EA0(v23, &v277, 0, 0);
  v26 = sub_1632080((__int64)a2, (__int64)"__tsan_ignore_thread_end", 24, v25, v24);
  if ( v275 != &v277 )
    _libc_free((unsigned __int64)v275);
  v27 = sub_1B28080(v26);
  v28 = v267;
  a1[25] = v27;
  v225 = (_QWORD **)a2;
  a1[21] = sub_1643350(v28);
  v224 = a1 + 56;
  for ( i = 0; i != 5; ++i )
  {
    v29 = (char *)&v277 + 5;
    v30 = (unsigned int)(1 << i);
    do
    {
      --v29;
      v31 = v30 % 0xA;
      v32 = v30;
      v30 /= 0xAu;
      v33 = v31 + 48;
      *v29 = v33;
    }
    while ( v32 > 9 );
    v34 = (char *)((char *)&v277 + 5 - v29);
    v231[0] = v232;
    v272 = (char *)((char *)&v277 + 5 - v29);
    if ( (unsigned __int64)((char *)&v277 + 5 - v29) > 0xF )
    {
      v231[0] = sub_22409D0(v231, &v272, 0);
      v182 = (_QWORD *)v231[0];
      v232[0] = v272;
      goto LABEL_125;
    }
    if ( v34 != (char *)1 )
    {
      if ( !v34 )
      {
        v35 = v232;
        goto LABEL_15;
      }
      v182 = v232;
LABEL_125:
      memcpy(v182, v29, (size_t)v34);
      v34 = v272;
      v35 = (_QWORD *)v231[0];
      goto LABEL_15;
    }
    LOBYTE(v232[0]) = v33;
    v35 = v232;
LABEL_15:
    v231[1] = v34;
    v34[(_QWORD)v35] = 0;
    v36 = (char *)&v277 + 5;
    v37 = (unsigned int)(8 << i);
    do
    {
      --v36;
      v38 = v37 % 0xA;
      v39 = v37;
      v37 /= 0xAu;
      v40 = v38 + 48;
      *v36 = v40;
    }
    while ( v39 > 9 );
    v41 = (char *)((char *)&v277 + 5 - v36);
    v233[0] = v234;
    v272 = (char *)((char *)&v277 + 5 - v36);
    if ( (unsigned __int64)((char *)&v277 + 5 - v36) > 0xF )
    {
      v233[0] = sub_22409D0(v233, &v272, 0);
      v211 = (_QWORD *)v233[0];
      v234[0] = v272;
    }
    else
    {
      if ( v41 == (char *)1 )
      {
        LOBYTE(v234[0]) = v40;
        v42 = v234;
        goto LABEL_20;
      }
      if ( !v41 )
      {
        v42 = v234;
        goto LABEL_20;
      }
      v211 = v234;
    }
    memcpy(v211, v36, (size_t)v41);
    v41 = v272;
    v42 = (_QWORD *)v233[0];
LABEL_20:
    v233[1] = v41;
    v41[(_QWORD)v42] = 0;
    sub_8FD6D0((__int64)&v275, "__tsan_read", v231);
    v45 = v276;
    v46 = (__int64 *)v275;
    v244 = v246;
    v245 = 0x2000000000LL;
    if ( v276 > 0x20 )
    {
      sub_16CD150((__int64)&v244, v246, v276, 1, v43, v44);
      v210 = &v244[(unsigned int)v245];
    }
    else
    {
      if ( !v276 )
        goto LABEL_22;
      v210 = v246;
    }
    memcpy(v210, v46, v45);
    v46 = (__int64 *)v275;
    LODWORD(v45) = v245 + v45;
LABEL_22:
    LODWORD(v245) = v45;
    if ( v46 != &v277 )
      j_j___libc_free_0(v46, v277 + 1);
    v47 = sub_16471D0(v267, 0);
    v48 = (__int64 *)sub_1643270(v267);
    v277 = v47;
    v49 = (unsigned int)v245;
    v276 = 0x100000001LL;
    v50 = (__int64)v244;
    v51 = v230;
    v275 = &v277;
    v52 = sub_1644EA0(v48, &v277, 1, 0);
    v53 = sub_1632080((__int64)v225, v50, v49, v52, v51);
    if ( v275 != &v277 )
      _libc_free((unsigned __int64)v275);
    *(v224 - 30) = sub_1B28080(v53);
    sub_8FD6D0((__int64)&v275, "__tsan_write", v231);
    v56 = v276;
    v57 = (__int64 *)v275;
    v247 = v249;
    v248 = 0x2000000000LL;
    if ( v276 > 0x20 )
    {
      sub_16CD150((__int64)&v247, v249, v276, 1, v54, v55);
      v213 = &v247[(unsigned int)v248];
    }
    else
    {
      if ( !v276 )
        goto LABEL_28;
      v213 = v249;
    }
    memcpy(v213, v57, v56);
    v57 = (__int64 *)v275;
    LODWORD(v56) = v248 + v56;
LABEL_28:
    LODWORD(v248) = v56;
    if ( v57 != &v277 )
      j_j___libc_free_0(v57, v277 + 1);
    v58 = sub_16471D0(v267, 0);
    v59 = (__int64 *)sub_1643270(v267);
    v60 = (unsigned int)v248;
    v61 = (__int64)v247;
    v62 = v230;
    v277 = v58;
    v276 = 0x100000001LL;
    v275 = &v277;
    v63 = sub_1644EA0(v59, &v277, 1, 0);
    v64 = sub_1632080((__int64)v225, v61, v60, v63, v62);
    if ( v275 != &v277 )
      _libc_free((unsigned __int64)v275);
    *(v224 - 25) = sub_1B28080(v64);
    sub_8FD6D0((__int64)&v275, "__tsan_unaligned_read", v231);
    v67 = v276;
    v68 = (__int64 *)v275;
    v272 = v274;
    v273 = 0x4000000000LL;
    if ( v276 > 0x40 )
    {
      sub_16CD150((__int64)&v272, v274, v276, 1, v65, v66);
      v212 = &v272[(unsigned int)v273];
    }
    else
    {
      if ( !v276 )
        goto LABEL_34;
      v212 = v274;
    }
    memcpy(v212, v68, v67);
    v68 = (__int64 *)v275;
    LODWORD(v67) = v273 + v67;
LABEL_34:
    LODWORD(v273) = v67;
    if ( v68 != &v277 )
      j_j___libc_free_0(v68, v277 + 1);
    v69 = sub_16471D0(v267, 0);
    v70 = (__int64 *)sub_1643270(v267);
    v277 = v69;
    v71 = (unsigned int)v273;
    v276 = 0x100000001LL;
    v72 = (__int64)v272;
    v73 = v230;
    v275 = &v277;
    v74 = sub_1644EA0(v70, &v277, 1, 0);
    v75 = sub_1632080((__int64)v225, v72, v71, v74, v73);
    if ( v275 != &v277 )
      _libc_free((unsigned __int64)v275);
    *(v224 - 20) = sub_1B28080(v75);
    sub_8FD6D0((__int64)&v260, "__tsan_unaligned_write", v231);
    v78 = v261;
    v79 = (__m128i *)v260;
    v275 = &v277;
    v276 = 0x4000000000LL;
    if ( v261 > 0x40 )
    {
      sub_16CD150((__int64)&v275, &v277, v261, 1, v76, v77);
      v216 = (__int64 *)((char *)v275 + (unsigned int)v276);
    }
    else
    {
      if ( !v261 )
        goto LABEL_40;
      v216 = &v277;
    }
    memcpy(v216, v79, v78);
    v79 = (__m128i *)v260;
    LODWORD(v78) = v276 + v78;
LABEL_40:
    LODWORD(v276) = v78;
    if ( v79 != &v262 )
      j_j___libc_free_0(v79, v262.m128i_i64[0] + 1);
    v80 = sub_16471D0(v267, 0);
    v81 = (__int64 *)sub_1643270(v267);
    v82 = (unsigned int)v276;
    v83 = v275;
    v261 = 0x100000001LL;
    v84 = v230;
    v262.m128i_i64[0] = v80;
    v260 = &v262;
    v85 = sub_1644EA0(v81, &v262, 1, 0);
    v86 = sub_1632080((__int64)v225, (__int64)v83, v82, v85, v84);
    if ( v260 != &v262 )
      _libc_free((unsigned __int64)v260);
    *(v224 - 15) = sub_1B28080(v86);
    v222.m128i_i64[1] = sub_1644C60(*v225, 8 << i);
    v222.m128i_i64[0] = sub_1647190((__int64 *)v222.m128i_i64[1], 0);
    sub_8FD6D0((__int64)v235, "__tsan_atomic", v233);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v235[1]) <= 4 )
LABEL_191:
      sub_4262D8((__int64)"basic_string::append");
    v88 = sub_2241490(v235, "_load", 5, v87);
    v260 = &v262;
    if ( *(_QWORD *)v88 == v88 + 16 )
    {
      v262 = _mm_loadu_si128((const __m128i *)(v88 + 16));
    }
    else
    {
      v260 = *(void **)v88;
      v262.m128i_i64[0] = *(_QWORD *)(v88 + 16);
    }
    v261 = *(_QWORD *)(v88 + 8);
    *(_QWORD *)v88 = v88 + 16;
    *(_QWORD *)(v88 + 8) = 0;
    *(_BYTE *)(v88 + 16) = 0;
    v91 = v261;
    v92 = (__m128i *)v260;
    v250 = v252;
    v251 = 0x2000000000LL;
    if ( v261 > 0x20 )
    {
      sub_16CD150((__int64)&v250, v252, v261, 1, v89, v90);
      v215 = &v250[(unsigned int)v251];
    }
    else
    {
      if ( !v261 )
        goto LABEL_49;
      v215 = v252;
    }
    memcpy(v215, v92, v91);
    v92 = (__m128i *)v260;
    LODWORD(v91) = v251 + v91;
LABEL_49:
    LODWORD(v251) = v91;
    if ( v92 != &v262 )
      j_j___libc_free_0(v92, v262.m128i_i64[0] + 1);
    if ( (__int64 *)v235[0] != &v236 )
      j_j___libc_free_0(v235[0], v236 + 1);
    v93 = a1[21];
    v94 = (unsigned int)v251;
    v260 = &v262;
    v262.m128i_i64[0] = v222.m128i_i64[0];
    v95 = (__int64)v250;
    v262.m128i_i64[1] = v93;
    v96 = v230;
    v261 = 0x200000002LL;
    v97 = sub_1644EA0((__int64 *)v222.m128i_i64[1], &v262, 2, 0);
    v98 = sub_1632080((__int64)v225, v95, v94, v97, v96);
    if ( v260 != &v262 )
      _libc_free((unsigned __int64)v260);
    *(v224 - 10) = sub_1B28080(v98);
    sub_8FD6D0((__int64)&v256, "__tsan_atomic", v233);
    if ( 0x3FFFFFFFFFFFFFFFLL - v257 <= 5 )
      goto LABEL_191;
    v100 = sub_2241490(&v256, "_store", 6, v99);
    v260 = &v262;
    if ( *(_QWORD *)v100 == v100 + 16 )
    {
      v262 = _mm_loadu_si128((const __m128i *)(v100 + 16));
    }
    else
    {
      v260 = *(void **)v100;
      v262.m128i_i64[0] = *(_QWORD *)(v100 + 16);
    }
    v261 = *(_QWORD *)(v100 + 8);
    *(_QWORD *)v100 = v100 + 16;
    *(_QWORD *)(v100 + 8) = 0;
    *(_BYTE *)(v100 + 16) = 0;
    v103 = v261;
    v104 = (__m128i *)v260;
    v253 = v255;
    v254 = 0x2000000000LL;
    if ( v261 > 0x20 )
    {
      sub_16CD150((__int64)&v253, v255, v261, 1, v101, v102);
      v214 = &v253[(unsigned int)v254];
LABEL_171:
      memcpy(v214, v104, v103);
      v104 = (__m128i *)v260;
      LODWORD(v103) = v254 + v103;
      goto LABEL_60;
    }
    if ( v261 )
    {
      v214 = v255;
      goto LABEL_171;
    }
LABEL_60:
    LODWORD(v254) = v103;
    if ( v104 != &v262 )
      j_j___libc_free_0(v104, v262.m128i_i64[0] + 1);
    if ( v256 != &v258 )
      j_j___libc_free_0(v256, v258.m128i_i64[0] + 1);
    v105 = a1[21];
    v106 = (__int64 *)sub_1643270(v267);
    v107 = (unsigned int)v254;
    v261 = 0x300000003LL;
    v262 = v222;
    v108 = (__int64)v253;
    v109 = v230;
    v263 = v105;
    v260 = &v262;
    v110 = sub_1644EA0(v106, &v262, 3, 0);
    v111 = sub_1632080((__int64)v225, v108, v107, v110, v109);
    if ( v260 != &v262 )
      _libc_free((unsigned __int64)v260);
    *(v224 - 5) = sub_1B28080(v111);
    v112 = v224;
    for ( j = 0; j != 11; ++j )
    {
      *v112 = 0;
      if ( j )
      {
        switch ( j )
        {
          case 1:
            v114 = "_fetch_add";
            break;
          case 2:
            v114 = "_fetch_sub";
            break;
          case 3:
            v114 = "_fetch_and";
            break;
          case 5:
            v114 = "_fetch_or";
            break;
          case 6:
            v114 = "_fetch_xor";
            break;
          default:
            v114 = "_fetch_nand";
            if ( j != 4 )
              goto LABEL_74;
            break;
        }
      }
      else
      {
        v114 = "_exchange";
      }
      v183 = (unsigned int)(8 << i);
      v184 = &v262.m128i_i8[5];
      do
      {
        --v184;
        v185 = v183 % 0xA;
        v186 = v183;
        v183 /= 0xAu;
        v187 = v185 + 48;
        *v184 = v187;
      }
      while ( v186 > 9 );
      v188 = (char *)(&v262.m128i_u8[5] - (unsigned __int8 *)v184);
      v189 = (__m128i *)(&v262.m128i_u8[5] - (unsigned __int8 *)v184);
      v237[0] = v238;
      v256 = (__m128i *)(&v262.m128i_u8[5] - (unsigned __int8 *)v184);
      if ( (unsigned __int64)(&v262.m128i_u8[5] - (unsigned __int8 *)v184) > 0xF )
      {
        v208 = sub_22409D0(v237, &v256, 0);
        v189 = (__m128i *)(&v262.m128i_u8[5] - (unsigned __int8 *)v184);
        v237[0] = v208;
        v209 = (_QWORD *)v208;
        v238[0] = v256;
      }
      else
      {
        if ( v188 == (char *)1 )
        {
          LOBYTE(v238[0]) = v187;
          v190 = v238;
          goto LABEL_134;
        }
        if ( !v188 )
        {
          v190 = v238;
          goto LABEL_134;
        }
        v209 = v238;
      }
      memcpy(v209, v184, (size_t)v189);
      v189 = v256;
      v190 = (_QWORD *)v237[0];
LABEL_134:
      v237[1] = v189;
      v189->m128i_i8[(_QWORD)v190] = 0;
      v191 = (__m128i *)sub_2241130(v237, 0, 0, "__tsan_atomic", 13);
      v239 = &v241;
      if ( (__m128i *)v191->m128i_i64[0] == &v191[1] )
      {
        v241 = _mm_loadu_si128(v191 + 1);
      }
      else
      {
        v239 = (__m128i *)v191->m128i_i64[0];
        v241.m128i_i64[0] = v191[1].m128i_i64[0];
      }
      v192 = v191->m128i_i64[1];
      v191[1].m128i_i8[0] = 0;
      v240 = v192;
      v191->m128i_i64[0] = (__int64)v191[1].m128i_i64;
      v191->m128i_i64[1] = 0;
      v193 = strlen(v114);
      if ( v193 > 0x3FFFFFFFFFFFFFFFLL - v240 )
        goto LABEL_191;
      v195 = (__m128i *)sub_2241490(&v239, v114, v193, v194);
      v256 = &v258;
      if ( (__m128i *)v195->m128i_i64[0] == &v195[1] )
      {
        v258 = _mm_loadu_si128(v195 + 1);
      }
      else
      {
        v256 = (__m128i *)v195->m128i_i64[0];
        v258.m128i_i64[0] = v195[1].m128i_i64[0];
      }
      v257 = v195->m128i_u64[1];
      v195->m128i_i64[0] = (__int64)v195[1].m128i_i64;
      v195->m128i_i64[1] = 0;
      v195[1].m128i_i8[0] = 0;
      v198 = v257;
      v260 = &v262;
      v199 = v256;
      v261 = 0x2000000000LL;
      if ( v257 > 0x20 )
      {
        src = v256;
        n = v257;
        sub_16CD150((__int64)&v260, &v262, v257, 1, v196, v197);
        v198 = n;
        v199 = src;
        v207 = (__m128i *)((char *)v260 + (unsigned int)v261);
LABEL_152:
        na = v198;
        memcpy(v207, v199, v198);
        v199 = v256;
        LODWORD(v198) = v261 + na;
        goto LABEL_141;
      }
      if ( v257 )
      {
        v207 = &v262;
        goto LABEL_152;
      }
LABEL_141:
      LODWORD(v261) = v198;
      if ( v199 != &v258 )
        j_j___libc_free_0(v199, v258.m128i_i64[0] + 1);
      if ( v239 != &v241 )
        j_j___libc_free_0(v239, v241.m128i_i64[0] + 1);
      if ( (_QWORD *)v237[0] != v238 )
        j_j___libc_free_0(v237[0], v238[0] + 1LL);
      v256 = &v258;
      v200 = a1[21];
      v201 = (unsigned int)v261;
      v258 = v222;
      v202 = v260;
      v221 = v230;
      v259 = v200;
      v257 = 0x300000003LL;
      v203 = sub_1644EA0((__int64 *)v222.m128i_i64[1], &v258, 3, 0);
      v204 = sub_1632080((__int64)v225, (__int64)v202, v201, v203, v221);
      if ( v256 != &v258 )
        _libc_free((unsigned __int64)v256);
      v205 = sub_1B28080(v204);
      v206 = (__m128i *)v260;
      *v112 = v205;
      if ( v206 != &v262 )
        _libc_free((unsigned __int64)v206);
LABEL_74:
      v112 += 5;
    }
    sub_8FD6D0((__int64)v242, "__tsan_atomic", v233);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v242[1]) <= 0x14 )
      goto LABEL_191;
    v116 = sub_2241490(v242, "_compare_exchange_val", 21, v115);
    v260 = &v262;
    if ( *(_QWORD *)v116 == v116 + 16 )
    {
      v262 = _mm_loadu_si128((const __m128i *)(v116 + 16));
    }
    else
    {
      v260 = *(void **)v116;
      v262.m128i_i64[0] = *(_QWORD *)(v116 + 16);
    }
    v261 = *(_QWORD *)(v116 + 8);
    *(_QWORD *)v116 = v116 + 16;
    *(_QWORD *)(v116 + 8) = 0;
    *(_BYTE *)(v116 + 16) = 0;
    v119 = v261;
    v256 = &v258;
    v120 = (__m128i *)v260;
    v257 = 0x2000000000LL;
    if ( v261 > 0x20 )
    {
      sub_16CD150((__int64)&v256, &v258, v261, 1, v117, v118);
      v217 = (__m128i *)((char *)v256 + (unsigned int)v257);
    }
    else
    {
      if ( !v261 )
        goto LABEL_80;
      v217 = &v258;
    }
    memcpy(v217, v120, v119);
    v120 = (__m128i *)v260;
    LODWORD(v119) = v257 + v119;
LABEL_80:
    LODWORD(v257) = v119;
    if ( v120 != &v262 )
      j_j___libc_free_0(v120, v262.m128i_i64[0] + 1);
    if ( (__int64 *)v242[0] != &v243 )
      j_j___libc_free_0(v242[0], v243 + 1);
    v121 = a1[21];
    v122 = (unsigned int)v257;
    v260 = &v262;
    v262 = v222;
    v123 = (__int64)v256;
    v264 = v121;
    v124 = v230;
    v265 = v121;
    v263 = v222.m128i_i64[1];
    v261 = 0x500000005LL;
    v125 = sub_1644EA0((__int64 *)v222.m128i_i64[1], &v262, 5, 0);
    v126 = sub_1632080((__int64)v225, v123, v122, v125, v124);
    if ( v260 != &v262 )
      _libc_free((unsigned __int64)v260);
    v127 = sub_1B28080(v126);
    v128 = v256;
    v224[55] = v127;
    if ( v128 != &v258 )
      _libc_free((unsigned __int64)v128);
    if ( v253 != v255 )
      _libc_free((unsigned __int64)v253);
    if ( v250 != v252 )
      _libc_free((unsigned __int64)v250);
    if ( v275 != &v277 )
      _libc_free((unsigned __int64)v275);
    if ( v272 != v274 )
      _libc_free((unsigned __int64)v272);
    if ( v247 != v249 )
      _libc_free((unsigned __int64)v247);
    if ( v244 != v246 )
      _libc_free((unsigned __int64)v244);
    if ( (_QWORD *)v233[0] != v234 )
      j_j___libc_free_0(v233[0], v234[0] + 1LL);
    if ( (_QWORD *)v231[0] != v232 )
      j_j___libc_free_0(v231[0], v232[0] + 1LL);
    ++v224;
  }
  v129 = sub_16471D0(v267, 0);
  v130 = sub_16471D0(v267, 0);
  v131 = (__int64 *)sub_1643270(v267);
  v132 = v230;
  v277 = v130;
  v275 = &v277;
  v278 = v129;
  v276 = 0x200000002LL;
  v133 = sub_1644EA0(v131, &v277, 2, 0);
  v134 = sub_1632080((__int64)v225, (__int64)"__tsan_vptr_update", 18, v133, v132);
  if ( v275 != &v277 )
    _libc_free((unsigned __int64)v275);
  v135 = sub_1B28080(v134);
  v136 = v267;
  a1[118] = v135;
  v137 = sub_16471D0(v136, 0);
  v138 = (__int64 *)sub_1643270(v267);
  v139 = v230;
  v275 = &v277;
  v277 = v137;
  v276 = 0x100000001LL;
  v140 = sub_1644EA0(v138, &v277, 1, 0);
  v141 = sub_1632080((__int64)v225, (__int64)"__tsan_vptr_read", 16, v140, v139);
  if ( v275 != &v277 )
    _libc_free((unsigned __int64)v275);
  v142 = sub_1B28080(v141);
  v143 = v267;
  a1[119] = v142;
  v144 = a1[21];
  v145 = (__int64 *)sub_1643270(v143);
  v146 = v230;
  v277 = v144;
  v275 = &v277;
  v276 = 0x100000001LL;
  v147 = sub_1644EA0(v145, &v277, 1, 0);
  v148 = sub_1632080((__int64)v225, (__int64)"__tsan_atomic_thread_fence", 26, v147, v146);
  if ( v275 != &v277 )
    _libc_free((unsigned __int64)v275);
  v149 = sub_1B28080(v148);
  v150 = v267;
  a1[116] = v149;
  v151 = a1[21];
  v152 = (__int64 *)sub_1643270(v150);
  v153 = v230;
  v277 = v151;
  v275 = &v277;
  v276 = 0x100000001LL;
  v154 = sub_1644EA0(v152, &v277, 1, 0);
  v155 = sub_1632080((__int64)v225, (__int64)"__tsan_atomic_signal_fence", 26, v154, v153);
  if ( v275 != &v277 )
    _libc_free((unsigned __int64)v275);
  v156 = sub_1B28080(v155);
  v157 = v267;
  a1[117] = v156;
  v226 = a1[20];
  v158 = sub_16471D0(v157, 0);
  v159 = sub_16471D0(v267, 0);
  v160 = (__int64 *)sub_16471D0(v267, 0);
  v277 = v159;
  v161 = v230;
  v275 = &v277;
  v278 = v158;
  v279 = v226;
  v276 = 0x300000003LL;
  v162 = sub_1644EA0(v160, &v277, 3, 0);
  v163 = sub_1632080((__int64)v225, (__int64)"memmove", 7, v162, v161);
  if ( v275 != &v277 )
    _libc_free((unsigned __int64)v275);
  v164 = sub_1B28080(v163);
  v165 = v267;
  a1[120] = v164;
  v227 = a1[20];
  v166 = sub_16471D0(v165, 0);
  v167 = sub_16471D0(v267, 0);
  v168 = (__int64 *)sub_16471D0(v267, 0);
  v277 = v167;
  v169 = v230;
  v275 = &v277;
  v278 = v166;
  v279 = v227;
  v276 = 0x300000003LL;
  v170 = sub_1644EA0(v168, &v277, 3, 0);
  v171 = sub_1632080((__int64)v225, (__int64)"memcpy", 6, v170, v169);
  if ( v275 != &v277 )
    _libc_free((unsigned __int64)v275);
  v172 = sub_1B28080(v171);
  v173 = v267;
  a1[121] = v172;
  v228 = a1[20];
  v174 = sub_1643350(v173);
  v175 = sub_16471D0(v267, 0);
  v176 = (__int64 *)sub_16471D0(v267, 0);
  v277 = v175;
  v177 = v230;
  v275 = &v277;
  v278 = v174;
  v279 = v228;
  v276 = 0x300000003LL;
  v178 = sub_1644EA0(v176, &v277, 3, 0);
  v179 = sub_1632080((__int64)v225, (__int64)"memset", 6, v178, v177);
  if ( v275 != &v277 )
    _libc_free((unsigned __int64)v275);
  result = sub_1B28080(v179);
  v181 = v266[0];
  a1[122] = result;
  if ( v181 )
    return sub_161E7C0((__int64)v266, v181);
  return result;
}
