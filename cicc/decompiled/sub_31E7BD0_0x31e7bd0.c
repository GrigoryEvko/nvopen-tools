// Function: sub_31E7BD0
// Address: 0x31e7bd0
//
__int64 __fastcall sub_31E7BD0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rbx
  __int64 *v3; // r15
  __int64 *v4; // r14
  __int64 *v5; // r12
  __int64 v6; // rdx
  __int64 *v7; // r12
  __int64 v8; // r15
  __int64 v9; // rbx
  __int64 v10; // r14
  char v11; // r13
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // r13
  char *v19; // r8
  __int64 *v20; // rax
  __int64 **v21; // rax
  __int64 v22; // rax
  __int64 **v23; // rax
  __int64 v24; // rax
  char v25; // cl
  char v26; // r8
  _QWORD *v27; // rdx
  unsigned __int8 v28; // r13
  __int16 v29; // r13
  unsigned __int8 *v30; // rax
  unsigned __int8 *v31; // rbx
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int16 v34; // cx
  __int64 i; // r12
  __int64 v36; // rsi
  _QWORD *v37; // r13
  __int64 *v38; // r12
  __int64 *v39; // r14
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // r15
  __int64 v44; // r8
  __int64 (*v45)(); // rax
  unsigned __int64 v46; // rsi
  void (*v47)(); // rax
  int v48; // eax
  __int64 v49; // rdi
  __int64 v50; // rdx
  __int64 v51; // r8
  unsigned __int64 v52; // r9
  __int64 v53; // rcx
  __int64 *v54; // r13
  __int64 v55; // r14
  __int64 *v56; // rbx
  __int64 **v57; // rax
  __int64 v58; // rbx
  _QWORD *v59; // r15
  __int64 *v60; // r12
  __int64 *v61; // r13
  __int64 v62; // rdi
  __int64 *v63; // r12
  __int64 *v64; // r13
  __int64 v65; // rdi
  __int64 v66; // r13
  __int64 v67; // r12
  __int64 v68; // rdi
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // r12
  __int64 v72; // r13
  __int64 v73; // rdi
  __int64 v74; // r13
  __int64 v75; // rax
  __m128i v76; // xmm1
  _BYTE *v77; // r15
  _BYTE *v78; // rbx
  __m128i *v79; // rdi
  __int64 (__fastcall *v80)(__int64); // rax
  __int64 v81; // rax
  __int64 v82; // rsi
  __int64 v83; // rdi
  void (__fastcall *v84)(__int64, __int64, __int64); // r15
  __int64 v85; // rax
  __m128i *v86; // r15
  __m128i *v87; // rbx
  __m128i *v88; // rdi
  __int64 (__fastcall *v89)(__int64); // rax
  __int64 v90; // rdi
  __int64 v91; // rax
  unsigned __int64 v92; // rdx
  __int64 v93; // rdx
  __int64 (*v94)(); // rax
  __int64 v95; // rax
  _QWORD *v96; // r13
  _QWORD *v97; // r12
  __int64 v98; // r13
  void (__fastcall *v99)(__int64, __int64, __int64); // r14
  __int64 v100; // rax
  __int64 v101; // rax
  __int64 v102; // rax
  __int64 v103; // r12
  __int64 v104; // r13
  __int64 v105; // r14
  __int64 v106; // rsi
  __int64 v107; // rax
  __int64 v108; // rdi
  void (*v109)(); // rax
  __int64 (__fastcall *v110)(__int64, __int64); // rax
  _BYTE *v111; // rax
  __int64 v112; // rax
  void (*v113)(void); // rax
  __m128i v114; // xmm3
  __m128i v115; // xmm4
  __m128i v116; // xmm5
  _BYTE *v117; // rbx
  _BYTE *v118; // r14
  __m128i *v119; // rdi
  __int64 (__fastcall *v120)(__int64); // rax
  __int64 v121; // rax
  __int64 v122; // rdi
  __int64 v123; // r14
  char v124; // al
  const char *v125; // rax
  unsigned __int64 v126; // rdx
  _BYTE *v127; // rbx
  _BYTE *v128; // r15
  __m128i *v129; // rdi
  __int64 (__fastcall *v130)(__int64); // rax
  __int64 v131; // rdi
  __int64 v132; // rdi
  __int64 (*v133)(); // rax
  __int64 v134; // rsi
  void (*v135)(); // rax
  unsigned __int64 v136; // rdi
  unsigned __int64 v137; // rdi
  unsigned __int64 v138; // r14
  __int64 v139; // rbx
  unsigned __int64 v140; // r12
  unsigned __int64 v141; // r13
  unsigned __int64 v142; // rdi
  __m128i v144; // xmm7
  __m128i v145; // xmm6
  __m128i v146; // xmm7
  size_t *v147; // r14
  __m128i *v148; // r12
  size_t *v149; // rbx
  __m128i *v150; // rdi
  __int64 (__fastcall *v151)(__int64); // rax
  __int64 v152; // rax
  size_t v153; // rdi
  __int64 v154; // r15
  __int64 v155; // rdi
  void (__fastcall *v156)(__int64, unsigned __int64, _QWORD); // rax
  unsigned __int64 v157; // rsi
  __int64 v158; // rdx
  const char *v159; // rax
  __int64 v160; // rax
  unsigned __int8 *v161; // rsi
  size_t *v162; // rbx
  __m128i *v163; // rdi
  __int64 (__fastcall *v164)(__int64); // rax
  size_t v165; // rdi
  __int64 *j; // r14
  __int64 v167; // rdx
  __int64 v168; // r12
  __int64 v169; // rax
  __int64 v170; // r8
  unsigned __int64 v171; // rdi
  _QWORD *v172; // r12
  __int64 v173; // r14
  size_t *v174; // rax
  size_t v175; // r15
  const void *v176; // r9
  unsigned __int64 v177; // rdx
  char *v178; // rdi
  _BYTE *v179; // rsi
  unsigned __int64 v180; // rdx
  void *v181; // r10
  size_t v182; // rdx
  _QWORD *v183; // rdi
  void (__fastcall *v184)(void *, unsigned __int64, _QWORD); // r15
  __int64 *v185; // r8
  __int64 v186; // r9
  __int64 v187; // r8
  unsigned __int64 v188; // rax
  __int64 v189; // rax
  unsigned int v190; // esi
  __int64 v191; // r15
  unsigned int v192; // eax
  __int64 v193; // r14
  __int64 v194; // r12
  __int64 *v195; // r14
  __int64 v196; // rax
  __int64 *v197; // r12
  __int64 v198; // rsi
  __int64 v199; // r15
  unsigned int v200; // eax
  void (*v201)(); // rbx
  __int64 v202; // rax
  __int64 v203; // r13
  __int64 v204; // rdi
  void (__fastcall *v205)(__int64, unsigned __int64, _QWORD); // r14
  unsigned __int64 v206; // rax
  __int64 v207; // r13
  __int64 v208; // rdi
  void (__fastcall *v209)(__int64, unsigned __int64, _QWORD); // r14
  unsigned __int64 v210; // rax
  __int64 v211; // rax
  __int128 v212; // [rsp+8h] [rbp-598h]
  __m128i v213; // [rsp+18h] [rbp-588h]
  __int64 v214; // [rsp+20h] [rbp-580h]
  unsigned __int64 v215; // [rsp+28h] [rbp-578h]
  unsigned __int64 v216; // [rsp+28h] [rbp-578h]
  _BYTE *v217; // [rsp+30h] [rbp-570h]
  unsigned __int64 v218; // [rsp+30h] [rbp-570h]
  __int64 v219; // [rsp+30h] [rbp-570h]
  __m128i v220; // [rsp+38h] [rbp-568h]
  __int64 v221; // [rsp+38h] [rbp-568h]
  __int64 v222; // [rsp+38h] [rbp-568h]
  __int64 v223; // [rsp+40h] [rbp-560h]
  __int128 v224; // [rsp+40h] [rbp-560h]
  __int64 v225; // [rsp+40h] [rbp-560h]
  _QWORD *v226; // [rsp+48h] [rbp-558h]
  unsigned __int64 v227; // [rsp+48h] [rbp-558h]
  void (__fastcall *v228)(__int64, unsigned __int64, _QWORD); // [rsp+48h] [rbp-558h]
  void (__fastcall *v229)(__int64, const char *); // [rsp+48h] [rbp-558h]
  _QWORD *v230; // [rsp+48h] [rbp-558h]
  char v231; // [rsp+50h] [rbp-550h]
  unsigned __int64 v232; // [rsp+50h] [rbp-550h]
  __int64 v233; // [rsp+50h] [rbp-550h]
  unsigned __int64 v234; // [rsp+50h] [rbp-550h]
  char *v235; // [rsp+58h] [rbp-548h]
  char v236; // [rsp+58h] [rbp-548h]
  _QWORD *v237; // [rsp+58h] [rbp-548h]
  void *v238; // [rsp+58h] [rbp-548h]
  int v239; // [rsp+58h] [rbp-548h]
  __int64 *v240; // [rsp+60h] [rbp-540h]
  _QWORD *v241; // [rsp+60h] [rbp-540h]
  __int64 v242; // [rsp+68h] [rbp-538h]
  unsigned __int64 v243; // [rsp+68h] [rbp-538h]
  unsigned __int64 v244; // [rsp+68h] [rbp-538h]
  void *src; // [rsp+70h] [rbp-530h]
  void *srca; // [rsp+70h] [rbp-530h]
  void *srcb; // [rsp+70h] [rbp-530h]
  void *srcc; // [rsp+70h] [rbp-530h]
  void *srcd; // [rsp+70h] [rbp-530h]
  void *srce; // [rsp+70h] [rbp-530h]
  __int64 *v251; // [rsp+78h] [rbp-528h]
  __int64 *v252; // [rsp+80h] [rbp-520h] BYREF
  __int64 *v253; // [rsp+88h] [rbp-518h]
  __int64 *v254; // [rsp+90h] [rbp-510h]
  __m128i v255; // [rsp+A0h] [rbp-500h] BYREF
  __m128i v256; // [rsp+B0h] [rbp-4F0h]
  _BYTE v257[16]; // [rsp+C0h] [rbp-4E0h] BYREF
  __int64 (__fastcall *v258)(__int64 *); // [rsp+D0h] [rbp-4D0h]
  __int64 v259; // [rsp+D8h] [rbp-4C8h]
  _BYTE v260[16]; // [rsp+E0h] [rbp-4C0h] BYREF
  __int64 (__fastcall *v261)(_QWORD *); // [rsp+F0h] [rbp-4B0h]
  __int64 v262; // [rsp+F8h] [rbp-4A8h]
  __m128i v263; // [rsp+100h] [rbp-4A0h] BYREF
  __m128i v264; // [rsp+110h] [rbp-490h] BYREF
  unsigned __int64 v265; // [rsp+120h] [rbp-480h]
  void *v266; // [rsp+128h] [rbp-478h]
  __int128 v267; // [rsp+130h] [rbp-470h]
  __m128i v268; // [rsp+140h] [rbp-460h] BYREF
  __m128i v269; // [rsp+150h] [rbp-450h]
  __m128i v270; // [rsp+160h] [rbp-440h]
  __m128i v271; // [rsp+170h] [rbp-430h]
  __m128i v272; // [rsp+180h] [rbp-420h] BYREF
  __m128i v273; // [rsp+190h] [rbp-410h]
  __m128i v274; // [rsp+1A0h] [rbp-400h]
  __m128i v275; // [rsp+1B0h] [rbp-3F0h]
  _BYTE v276[16]; // [rsp+1C0h] [rbp-3E0h] BYREF
  __int64 (__fastcall *v277)(__int64); // [rsp+1D0h] [rbp-3D0h]
  __int64 v278; // [rsp+1D8h] [rbp-3C8h]
  __int64 (__fastcall *v279)(__int64); // [rsp+1E0h] [rbp-3C0h]
  __int64 v280; // [rsp+1E8h] [rbp-3B8h]
  __int64 (__fastcall *v281)(__int64 *); // [rsp+1F0h] [rbp-3B0h]
  __int64 v282; // [rsp+1F8h] [rbp-3A8h]
  _BYTE v283[16]; // [rsp+200h] [rbp-3A0h] BYREF
  __int64 (__fastcall *v284)(__int64); // [rsp+210h] [rbp-390h]
  __int64 v285; // [rsp+218h] [rbp-388h]
  __int64 (__fastcall *v286)(__int64); // [rsp+220h] [rbp-380h]
  __int64 v287; // [rsp+228h] [rbp-378h]
  __int64 (__fastcall *v288)(_QWORD *); // [rsp+230h] [rbp-370h]
  __int64 v289; // [rsp+238h] [rbp-368h]
  _BYTE v290[16]; // [rsp+240h] [rbp-360h] BYREF
  __int64 (__fastcall *v291)(__int64); // [rsp+250h] [rbp-350h]
  __int64 v292; // [rsp+258h] [rbp-348h]
  __int64 (__fastcall *v293)(__int64); // [rsp+260h] [rbp-340h]
  __int64 v294; // [rsp+268h] [rbp-338h]
  __int64 (__fastcall *v295)(__int64 *); // [rsp+270h] [rbp-330h]
  __int64 v296; // [rsp+278h] [rbp-328h]
  size_t v297[4]; // [rsp+280h] [rbp-320h] BYREF
  __int64 (__fastcall *v298)(__int64); // [rsp+2A0h] [rbp-300h]
  __int64 v299; // [rsp+2A8h] [rbp-2F8h]
  __int64 (__fastcall *v300)(_QWORD *); // [rsp+2B0h] [rbp-2F0h]
  __int64 v301; // [rsp+2B8h] [rbp-2E8h]
  __m128i v302; // [rsp+2C0h] [rbp-2E0h] BYREF
  __m128i v303; // [rsp+2D0h] [rbp-2D0h] BYREF
  __m128i v304; // [rsp+2E0h] [rbp-2C0h] BYREF
  __m128i v305[2]; // [rsp+2F0h] [rbp-2B0h] BYREF
  void *v306; // [rsp+310h] [rbp-290h]
  void *v307; // [rsp+318h] [rbp-288h]
  unsigned __int64 v308; // [rsp+320h] [rbp-280h]
  unsigned __int64 v309; // [rsp+328h] [rbp-278h]
  unsigned __int64 v310; // [rsp+330h] [rbp-270h]
  unsigned __int64 v311; // [rsp+338h] [rbp-268h]
  __m128i v312; // [rsp+340h] [rbp-260h] BYREF
  __m128i v313; // [rsp+350h] [rbp-250h] BYREF
  __m128i v314; // [rsp+360h] [rbp-240h] BYREF
  __m128i v315[2]; // [rsp+370h] [rbp-230h] BYREF
  unsigned __int64 v316; // [rsp+390h] [rbp-210h]
  void *v317; // [rsp+398h] [rbp-208h]
  __int128 v318; // [rsp+3A0h] [rbp-200h]
  unsigned __int64 v319; // [rsp+3B0h] [rbp-1F0h]
  unsigned __int64 v320; // [rsp+3B8h] [rbp-1E8h]
  _QWORD *v321; // [rsp+3C0h] [rbp-1E0h] BYREF
  __int64 v322; // [rsp+3C8h] [rbp-1D8h]
  _QWORD v323[2]; // [rsp+3D0h] [rbp-1D0h] BYREF
  __int16 v324; // [rsp+3E0h] [rbp-1C0h]
  void *v325; // [rsp+450h] [rbp-150h] BYREF
  unsigned __int64 v326; // [rsp+458h] [rbp-148h]
  __int64 v327; // [rsp+460h] [rbp-140h]
  _BYTE s[7]; // [rsp+468h] [rbp-138h] BYREF
  char v329; // [rsp+46Fh] [rbp-131h] BYREF
  char v330; // [rsp+470h] [rbp-130h] BYREF

  v2 = a1;
  v217 = a2 + 1;
  v3 = a2 + 1;
  *(_QWORD *)(a1 + 232) = 0;
  v4 = (__int64 *)a2[2];
  v251 = a2;
  v252 = 0;
  v253 = 0;
  v254 = 0;
  if ( v4 == a2 + 1 )
    goto LABEL_29;
  do
  {
    while ( 1 )
    {
      v5 = v4 - 7;
      if ( !v4 )
        v5 = 0;
      if ( sub_B2FC80((__int64)v5)
        || (*((_BYTE *)v5 + 34) & 1) == 0
        || (*(_BYTE *)sub_B31490((__int64)v5, (__int64)a2, v6) & 4) == 0 )
      {
        goto LABEL_3;
      }
      v325 = v5;
      a2 = v253;
      if ( v253 != v254 )
        break;
      sub_24400C0((__int64)&v252, v253, &v325);
LABEL_3:
      v4 = (__int64 *)v4[1];
      if ( v4 == v3 )
        goto LABEL_13;
    }
    if ( v253 )
    {
      *v253 = (__int64)v5;
      a2 = v253;
    }
    v253 = ++a2;
    v4 = (__int64 *)v4[1];
  }
  while ( v4 != v3 );
LABEL_13:
  v240 = v253;
  if ( v252 != v253 )
  {
    v7 = v252;
    src = v251 + 39;
    do
    {
      v8 = *v7;
      v9 = *(_QWORD *)(*v7 - 32);
      v10 = *(_QWORD *)(v9 + 8);
      v11 = sub_AE5020((__int64)src, v10);
      v12 = sub_9208B0((__int64)src, v10);
      v326 = v13;
      v325 = (void *)(((1LL << v11) + ((unsigned __int64)(v12 + 7) >> 3) - 1) >> v11 << v11);
      v14 = sub_CA1930(&v325);
      v17 = 16 * ((v14 != 0) + ((v14 - (unsigned __int64)(v14 != 0)) >> 4));
      if ( v14 != v17 )
      {
        v18 = v17 - v14;
        v326 = 0;
        v325 = s;
        v327 = 40;
        if ( v18 > 0x28 )
        {
          sub_C8D290((__int64)&v325, s, v18, 1u, v15, v16);
          memset(v325, 0, v18);
          v326 = v18;
          v19 = (char *)v325;
        }
        else
        {
          memset(s, 0, v18);
          v326 = v18;
          v19 = s;
        }
        v235 = v19;
        v20 = (__int64 *)sub_BCD140((_QWORD *)*v251, 8u);
        v21 = (__int64 **)sub_BCD420(v20, v18);
        v22 = sub_AC9630(v235, v18, v21);
        v321 = (_QWORD *)v9;
        v322 = v22;
        v23 = (__int64 **)sub_AC34C0(&v321, 2, 0);
        v24 = sub_AD24A0(v23, (__int64 *)&v321, 2);
        v25 = *(_BYTE *)(v8 + 80);
        v26 = *(_BYTE *)(v8 + 32);
        v27 = *(_QWORD **)(v24 + 8);
        v223 = v24;
        v324 = 257;
        v28 = *(_BYTE *)(v8 + 33);
        v226 = v27;
        LODWORD(v24) = *(_DWORD *)(*(_QWORD *)(v8 + 8) + 8LL);
        v231 = v25 & 1;
        v236 = v26 & 0xF;
        v312.m128i_i8[4] = 1;
        v29 = (v28 >> 2) & 7;
        v312.m128i_i32[0] = (unsigned int)v24 >> 8;
        v30 = (unsigned __int8 *)sub_BD2C40(88, unk_3F0FAE8);
        v31 = v30;
        if ( v30 )
          sub_B30000((__int64)v30, (__int64)v251, v226, v231, v236, v223, (__int64)&v321, v8, v29, v312.m128i_i64[0], 0);
        sub_B32030((__int64)v31, v8);
        sub_B2F990((__int64)v31, *(_QWORD *)(v8 + 48), v32, v33);
        sub_B9E560((__int64)v31, v8, 0);
        sub_BD6B90(v31, (unsigned __int8 *)v8);
        sub_BD84D0(v8, (__int64)v31);
        sub_B30290(v8);
        if ( v325 != s )
          _libc_free((unsigned __int64)v325);
        v8 = (__int64)v31;
      }
      v34 = (*(_WORD *)(v8 + 34) >> 1) & 0x3F;
      if ( !v34 || (unsigned __int64)(1LL << ((unsigned __int8)v34 - 1)) <= 0xF )
        sub_B2F770(v8, 4u);
      *(_BYTE *)(v8 + 32) &= 0x3Fu;
      ++v7;
    }
    while ( v240 != v7 );
    v2 = a1;
  }
LABEL_29:
  sub_31E7A70(v2, (__int64)v251);
  for ( i = v251[2]; (_BYTE *)i != v217; i = *(_QWORD *)(i + 8) )
  {
    v36 = i - 56;
    if ( !i )
      v36 = 0;
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v2 + 224LL))(v2, v36);
  }
  sub_31DC120(v2);
  v37 = (_QWORD *)sub_31DA6B0(v2);
  v38 = (__int64 *)v251[4];
  v39 = v251 + 3;
  if ( v251 + 3 != v38 )
  {
    while ( 1 )
    {
LABEL_38:
      if ( !v38 )
        BUG();
      if ( (*(_BYTE *)(v38 - 3) & 0xF) != 1 && !sub_B2FC80((__int64)(v38 - 7)) )
        goto LABEL_37;
      v40 = sub_31DB510(v2, (__int64)(v38 - 7));
      v42 = *(_QWORD *)(v2 + 200);
      v43 = v40;
      if ( *(_DWORD *)(v42 + 564) != 8 )
        break;
      if ( (*((_BYTE *)v38 - 23) & 0x20) != 0 )
        goto LABEL_37;
      v44 = 0;
      v45 = *(__int64 (**)())(*v37 + 256LL);
      if ( v45 != sub_302E4D0 )
        v44 = ((__int64 (__fastcall *)(_QWORD *, __int64 *, __int64, __int64, _QWORD))v45)(v37, v38 - 7, v42, v41, 0);
      (*(void (__fastcall **)(__int64, __int64 *, __int64))(*(_QWORD *)v2 + 560LL))(v2, v38 - 7, v44);
      if ( !(unsigned __int8)sub_B2DDD0((__int64)(v38 - 7), 0, 0, 1, 0, 0, 0) )
        goto LABEL_37;
      (*(void (__fastcall **)(__int64, __int64 *, __int64))(*(_QWORD *)v2 + 560LL))(v2, v38 - 7, v43);
      v38 = (__int64 *)v38[1];
      if ( v39 == v38 )
        goto LABEL_47;
    }
    if ( ((*((_BYTE *)v38 - 24) >> 4) & 3) != 0 )
      sub_31DE970(v2, v40, (*((_BYTE *)v38 - 24) >> 4) & 3, 0);
LABEL_37:
    v38 = (__int64 *)v38[1];
    if ( v39 == v38 )
      goto LABEL_47;
    goto LABEL_38;
  }
LABEL_47:
  v46 = sub_B6E990(*v251);
  if ( v46 )
    sub_31DC450(v2, v46);
  v47 = *(void (**)())(*v37 + 40LL);
  if ( v47 != nullsub_1711 )
  {
    v46 = *(_QWORD *)(v2 + 224);
    ((void (__fastcall *)(_QWORD *, unsigned __int64, __int64 *))v47)(v37, v46, v251);
  }
  v48 = *(_DWORD *)(*(_QWORD *)(v2 + 200) + 564LL);
  if ( v48 == 3 )
  {
    v193 = *(_QWORD *)(v2 + 240);
    v194 = *(_QWORD *)(v193 + 2496);
    if ( !v194 )
    {
      v211 = sub_22077B0(0x50u);
      v194 = v211;
      if ( v211 )
        sub_3531AE0(v211, v193);
      *(_QWORD *)(v193 + 2496) = v194;
    }
    v46 = v194 + 8;
    sub_3531730(&v325, v194 + 8);
    v195 = (__int64 *)v326;
    if ( v325 != (void *)v326 )
    {
      (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(v2 + 224) + 176LL))(
        *(_QWORD *)(v2 + 224),
        v37[4],
        0);
      LODWORD(v196) = sub_AE4380((__int64)(v251 + 39), 0);
      v46 = 0xFFFFFFFFLL;
      if ( (_DWORD)v196 )
      {
        _BitScanReverse64((unsigned __int64 *)&v196, (unsigned int)v196);
        v46 = 63 - ((unsigned int)v196 ^ 0x3F);
      }
      sub_31DCA70(v2, v46, 0, 0);
      v197 = (__int64 *)v325;
      v195 = (__int64 *)v326;
      if ( v325 != (void *)v326 )
      {
        do
        {
          v198 = *v197;
          v197 += 2;
          (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v2 + 224) + 208LL))(
            *(_QWORD *)(v2 + 224),
            v198,
            0);
          v199 = *(_QWORD *)(v2 + 224);
          v200 = sub_AE4380((__int64)(v251 + 39), 0);
          v46 = *(v197 - 1) & 0xFFFFFFFFFFFFFFF8LL;
          sub_E9A500(v199, v46, v200, 0);
        }
        while ( v195 != v197 );
        v195 = (__int64 *)v325;
      }
    }
    if ( v195 )
    {
      v46 = v327 - (_QWORD)v195;
      j_j___libc_free_0((unsigned __int64)v195);
    }
    v48 = *(_DWORD *)(*(_QWORD *)(v2 + 200) + 564LL);
  }
  if ( v48 == 1 )
  {
    v168 = *(_QWORD *)(v2 + 240);
    v169 = *(_QWORD *)(v168 + 2496);
    if ( !v169 )
    {
      v169 = sub_22077B0(0x28u);
      if ( v169 )
      {
        *(_QWORD *)(v169 + 8) = 0;
        *(_QWORD *)(v169 + 16) = 0;
        *(_QWORD *)(v169 + 24) = 0;
        *(_QWORD *)v169 = &unk_4A38F50;
        *(_DWORD *)(v169 + 32) = 0;
      }
      *(_QWORD *)(v168 + 2496) = v169;
    }
    v46 = v169 + 8;
    sub_3531730(&v321, v169 + 8);
    v171 = (unsigned __int64)v321;
    v241 = (_QWORD *)v322;
    if ( (_QWORD *)v322 == v321 )
    {
LABEL_229:
      if ( v171 )
      {
        v46 = v323[0] - v171;
        j_j___libc_free_0(v171);
      }
      goto LABEL_53;
    }
    v172 = v321;
    v173 = (__int64)(v251 + 39);
    while ( 1 )
    {
      v325 = s;
      v327 = 256;
      qmemcpy(s, ".rdata$", sizeof(s));
      v326 = 7;
      if ( (*(_BYTE *)(*v172 + 8LL) & 1) != 0 )
      {
        v174 = *(size_t **)(*v172 - 8LL);
        v175 = *v174;
        v176 = v174 + 3;
        v177 = *v174 + 7;
        if ( v177 <= 0x100 )
        {
          if ( !v175 )
          {
            v180 = 7;
            v179 = s;
            goto LABEL_219;
          }
          v178 = &v329;
        }
        else
        {
          srcd = v174 + 3;
          sub_C8D290((__int64)&v325, s, v177, 1u, v170, (__int64)v176);
          v176 = srcd;
          v178 = (char *)v325 + v326;
        }
        memcpy(v178, v176, v175);
        v179 = v325;
        v180 = v326;
      }
      else
      {
        v180 = 7;
        v179 = s;
        v175 = 0;
      }
LABEL_219:
      v181 = *(void **)(v2 + 224);
      v182 = v175 + v180;
      v183 = *(_QWORD **)(v2 + 216);
      v326 = v182;
      v184 = *(void (__fastcall **)(void *, unsigned __int64, _QWORD))(*(_QWORD *)v181 + 176LL);
      if ( (*(_BYTE *)(*v172 + 8LL) & 1) != 0 )
      {
        v185 = *(__int64 **)(*v172 - 8LL);
        v186 = *v185;
        v187 = (__int64)(v185 + 3);
      }
      else
      {
        v186 = 0;
        v187 = 0;
      }
      srce = v181;
      v188 = sub_E6DEB0(v183, v179, v182, 0x40001040u, v187, v186, 2u, 0xFFFFFFFF);
      v184(srce, v188, 0);
      LODWORD(v189) = sub_AE4380(v173, 0);
      v190 = -1;
      if ( (_DWORD)v189 )
      {
        _BitScanReverse64((unsigned __int64 *)&v189, (unsigned int)v189);
        v190 = 63 - (v189 ^ 0x3F);
      }
      sub_31DCA70(v2, v190, 0, 0);
      (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(v2 + 224) + 296LL))(
        *(_QWORD *)(v2 + 224),
        *v172,
        9);
      (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(v2 + 224) + 208LL))(
        *(_QWORD *)(v2 + 224),
        *v172,
        0);
      v191 = *(_QWORD *)(v2 + 224);
      v192 = sub_AE4380(v173, 0);
      v46 = v172[1] & 0xFFFFFFFFFFFFFFF8LL;
      sub_E9A500(v191, v46, v192, 0);
      if ( v325 != s )
        _libc_free((unsigned __int64)v325);
      v172 += 2;
      if ( v241 == v172 )
      {
        v171 = (unsigned __int64)v321;
        goto LABEL_229;
      }
    }
  }
LABEL_53:
  v49 = *(_QWORD *)(*(_QWORD *)(v2 + 224) + 16LL);
  if ( v49 )
    (*(void (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v49 + 72LL))(v49, v46);
  sub_31E5100(v2);
  v53 = (__int64)v251;
  v325 = 0;
  v321 = v323;
  v54 = (__int64 *)v251[6];
  v322 = 0x1000000000LL;
  v326 = (unsigned __int64)&v330;
  v327 = 16;
  *(_DWORD *)s = 0;
  s[4] = 1;
  if ( v251 + 5 != v54 )
  {
    v55 = v2;
    while ( 1 )
    {
      if ( !v54 )
        BUG();
      if ( (*(_BYTE *)(v54 - 2) & 0xF) == 1 )
        goto LABEL_67;
      v56 = v54 - 6;
      while ( 1 )
      {
        if ( !s[4] )
        {
LABEL_100:
          v46 = (unsigned __int64)v56;
          sub_C8CC70((__int64)&v325, (__int64)v56, v50, v53, v51, v52);
          v91 = (unsigned int)v322;
          v52 = v92;
          v50 = (unsigned int)v322;
          if ( !(_BYTE)v52 )
            goto LABEL_103;
          v52 = (unsigned int)v322 + 1LL;
          if ( v52 <= HIDWORD(v322) )
            goto LABEL_102;
LABEL_108:
          v46 = (unsigned __int64)v323;
          sub_C8D5F0((__int64)&v321, v323, v52, 8u, v51, v52);
          v50 = (unsigned int)v322;
          goto LABEL_102;
        }
        v57 = (__int64 **)v326;
        v53 = HIDWORD(v327);
        v50 = v326 + 8LL * HIDWORD(v327);
        if ( v326 != v50 )
          break;
LABEL_106:
        if ( HIDWORD(v327) >= (unsigned int)v327 )
          goto LABEL_100;
        v53 = (unsigned int)++HIDWORD(v327);
        *(_QWORD *)v50 = v56;
        v50 = (unsigned int)v322;
        v325 = (char *)v325 + 1;
        v52 = (unsigned int)v322 + 1LL;
        if ( v52 > HIDWORD(v322) )
          goto LABEL_108;
LABEL_102:
        v321[v50] = v56;
        v91 = (unsigned int)(v322 + 1);
        LODWORD(v322) = v322 + 1;
        v56 = (__int64 *)*(v56 - 4);
        if ( *(_BYTE *)v56 != 1 )
        {
LABEL_103:
          v58 = (__int64)v321;
          v59 = &v321[v91];
          if ( v321 != v59 )
            goto LABEL_104;
LABEL_66:
          LODWORD(v322) = 0;
          goto LABEL_67;
        }
      }
      while ( v56 != *v57 )
      {
        if ( (__int64 **)v50 == ++v57 )
          goto LABEL_106;
      }
      v58 = (__int64)v321;
      v59 = &v321[(unsigned int)v322];
      if ( v321 == v59 )
        goto LABEL_66;
      do
      {
LABEL_104:
        v93 = *--v59;
        v46 = (unsigned __int64)v251;
        (*(void (__fastcall **)(__int64, __int64 *, __int64))(*(_QWORD *)v55 + 608LL))(v55, v251, v93);
      }
      while ( (_QWORD *)v58 != v59 );
      LODWORD(v322) = 0;
LABEL_67:
      v54 = (__int64 *)v54[1];
      if ( v251 + 5 == v54 )
      {
        v2 = v55;
        break;
      }
    }
  }
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(v2 + 200) + 544LL) - 42) > 1 )
  {
    for ( j = (__int64 *)v251[8]; v251 + 7 != j; j = (__int64 *)j[1] )
    {
      v167 = (__int64)(j - 7);
      v46 = (unsigned __int64)v251;
      if ( !j )
        v167 = 0;
      sub_31DE9E0(v2, (__int64)v251, v167);
    }
  }
  v60 = *(__int64 **)(v2 + 576);
  v61 = &v60[*(unsigned int *)(v2 + 584)];
  while ( v61 != v60 )
  {
    v62 = *v60++;
    (*(void (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v62 + 24LL))(v62, v46);
  }
  v63 = *(__int64 **)(v2 + 552);
  v64 = &v63[*(unsigned int *)(v2 + 560)];
  if ( v64 != v63 )
  {
    do
    {
      v65 = *v63++;
      (*(void (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v65 + 24LL))(v65, v46);
    }
    while ( v64 != v63 );
    v66 = *(_QWORD *)(v2 + 552);
    v67 = v66 + 8LL * *(unsigned int *)(v2 + 560);
    while ( v66 != v67 )
    {
      while ( 1 )
      {
        v68 = *(_QWORD *)(v67 - 8);
        v67 -= 8;
        if ( !v68 )
          break;
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v68 + 8LL))(v68);
        if ( v66 == v67 )
          goto LABEL_78;
      }
    }
  }
LABEL_78:
  v69 = *(_QWORD *)(v2 + 576);
  v70 = *(unsigned int *)(v2 + 584);
  *(_DWORD *)(v2 + 560) = 0;
  v71 = v69 + 8 * v70;
  v72 = v69 + 8LL * *(_QWORD *)(v2 + 608);
  if ( v72 != v71 )
  {
    do
    {
      v73 = *(_QWORD *)(v71 - 8);
      v71 -= 8;
      if ( v73 )
        (*(void (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v73 + 8LL))(v73, v46);
    }
    while ( v72 != v71 );
    v69 = *(_QWORD *)(v2 + 576);
  }
  *(_QWORD *)(v2 + 760) = 0;
  v74 = v72 - v69;
  v75 = *(_QWORD *)(v2 + 208);
  *(_DWORD *)(v2 + 584) = v74 >> 3;
  if ( *(_QWORD *)(v75 + 304) )
  {
    sub_BA9600(&v263, (__int64)v251);
    v237 = (_QWORD *)v2;
    v76 = _mm_loadu_si128(&v264);
    v232 = v265;
    v255 = _mm_loadu_si128(&v263);
    srca = v266;
    v256 = v76;
    v224 = v267;
    while ( 1 )
    {
      if ( __PAIR128__((unsigned __int64)srca, v232) == *(_OWORD *)&v255 && *(_OWORD *)&v256 == v224 )
      {
        v2 = (__int64)v237;
        v94 = *(__int64 (**)())(*v237 + 616LL);
        if ( v94 != sub_3020050
          && ((unsigned __int8 (__fastcall *)(_QWORD *))v94)(v237)
          && !sub_BA8CD0((__int64)v251, (__int64)"swift_async_extendedFramePointerFlags", 0x25u, 0) )
        {
          v95 = sub_BCE3C0((__int64 *)*v251, 0);
          v312.m128i_i64[0] = (__int64)"swift_async_extendedFramePointerFlags";
          v96 = (_QWORD *)v95;
          v314.m128i_i16[0] = 259;
          v302.m128i_i8[4] = 0;
          v97 = sub_BD2C40(88, unk_3F0FAE8);
          if ( v97 )
            sub_B30000((__int64)v97, (__int64)v251, v96, 0, 9, 0, (__int64)&v312, 0, 0, v302.m128i_i64[0], 0);
          v98 = v237[28];
          v99 = *(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v98 + 296LL);
          v100 = sub_31DB510((__int64)v237, (__int64)v97);
          v99(v98, v100, 26);
        }
        goto LABEL_116;
      }
      v77 = v257;
      v259 = 0;
      v78 = v257;
      v79 = &v255;
      v258 = sub_25AC5E0;
      v80 = sub_25AC5C0;
      if ( ((unsigned __int8)sub_25AC5C0 & 1) != 0 )
        goto LABEL_87;
LABEL_88:
      v81 = v80((__int64)v79);
      v82 = v81;
      if ( !v81 )
        break;
LABEL_92:
      if ( (*(_BYTE *)(v81 + 32) & 0xF) == 9 )
      {
        v242 = v237[28];
        v84 = *(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v242 + 296LL);
        v85 = sub_31DB510((__int64)v237, v82);
        v84(v242, v85, 26);
      }
      v86 = (__m128i *)v260;
      v262 = 0;
      v87 = (__m128i *)v260;
      v88 = &v255;
      v261 = sub_25AC590;
      v89 = sub_25AC560;
      if ( ((unsigned __int8)sub_25AC560 & 1) != 0 )
LABEL_95:
        v89 = *(__int64 (__fastcall **)(__int64))((char *)v89 + v88->m128i_i64[0] - 1);
      while ( !(unsigned __int8)v89((__int64)v88) )
      {
        if ( ++v86 == &v263 )
          goto LABEL_265;
        v90 = v87[1].m128i_i64[1];
        v89 = (__int64 (__fastcall *)(__int64))v87[1].m128i_i64[0];
        v87 = v86;
        v88 = (__m128i *)((char *)&v255 + v90);
        if ( ((unsigned __int8)v89 & 1) != 0 )
          goto LABEL_95;
      }
    }
    while ( 1 )
    {
      v77 += 16;
      if ( v260 == v77 )
        goto LABEL_265;
      v83 = *((_QWORD *)v78 + 3);
      v80 = (__int64 (__fastcall *)(__int64))*((_QWORD *)v78 + 2);
      v78 = v77;
      v79 = (__m128i *)((char *)&v255 + v83);
      if ( ((unsigned __int8)v80 & 1) != 0 )
        break;
      v81 = ((__int64 (__fastcall *)(__m128i *, __int64))v80)(v79, v82);
      v82 = v81;
      if ( v81 )
        goto LABEL_92;
    }
LABEL_87:
    v80 = *(__int64 (__fastcall **)(__int64))((char *)v80 + v79->m128i_i64[0] - 1);
    goto LABEL_88;
  }
LABEL_116:
  v101 = sub_B82360(*(_QWORD *)(v2 + 8), (__int64)&unk_501DA08);
  if ( !v101 )
    BUG();
  v102 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v101 + 104LL))(v101, &unk_501DA08);
  v103 = *(_QWORD *)(v102 + 176);
  v104 = v102;
  v105 = v103 + 8LL * *(unsigned int *)(v102 + 184);
  while ( v105 != v103 )
  {
    while ( 1 )
    {
      v106 = *(_QWORD *)(v105 - 8);
      v105 -= 8;
      v107 = sub_31E4CF0(v2, v106);
      v108 = v107;
      if ( v107 )
      {
        v109 = *(void (**)())(*(_QWORD *)v107 + 24LL);
        if ( v109 != nullsub_1845 )
          break;
      }
      if ( v105 == v103 )
        goto LABEL_123;
    }
    ((void (__fastcall *)(__int64, __int64 *, __int64, __int64))v109)(v108, v251, v104, v2);
  }
LABEL_123:
  v110 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v2 + 584LL);
  if ( v110 == sub_31D61A0 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(v2 + 208) + 291LL) )
      sub_31D60D0(v2, (__int64)v251);
  }
  else
  {
    v110(v2, (__int64)v251);
  }
  if ( *(_DWORD *)(*(_QWORD *)(v2 + 200) + 564LL) != 8 )
  {
    (*(void (__fastcall **)(__int64, __int64 *))(*(_QWORD *)v2 + 592LL))(v2, v251);
    if ( *(_DWORD *)(*(_QWORD *)(v2 + 200) + 564LL) == 3 )
    {
      if ( *(_BYTE *)(v2 + 780) )
      {
        v203 = *(_QWORD *)(v2 + 224);
        v204 = *(_QWORD *)(v2 + 216);
        v205 = *(void (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)v203 + 176LL);
        v314.m128i_i16[0] = 257;
        v302.m128i_i64[0] = (__int64)".note.GNU-split-stack";
        v304.m128i_i16[0] = 259;
        v206 = sub_E71CB0(v204, (size_t *)&v302, 1, 0, 0, (__int64)&v312, 0, -1, 0);
        v205(v203, v206, 0);
        if ( *(_BYTE *)(v2 + 781) )
        {
          v207 = *(_QWORD *)(v2 + 224);
          v208 = *(_QWORD *)(v2 + 216);
          v209 = *(void (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)v207 + 176LL);
          v314.m128i_i16[0] = 257;
          v302.m128i_i64[0] = (__int64)".note.GNU-no-split-stack";
          v304.m128i_i16[0] = 259;
          v210 = sub_E71CB0(v208, (size_t *)&v302, 1, 0, 0, (__int64)&v312, 0, -1, 0);
          v209(v207, v210, 0);
        }
      }
    }
  }
  v111 = sub_BA8CB0((__int64)v251, (__int64)"llvm.init.trampoline", 0x14u);
  if ( v111 && *((_QWORD *)v111 + 2)
    || (v132 = *(_QWORD *)(v2 + 208), v133 = *(__int64 (**)())(*(_QWORD *)v132 + 16LL), v133 == sub_E7D6D0)
    || (v134 = ((__int64 (__fastcall *)(__int64, _QWORD))v133)(v132, *(_QWORD *)(v2 + 216))) == 0 )
  {
    v112 = *(_QWORD *)(v2 + 200);
    if ( (*(_BYTE *)(v112 + 879) & 8) == 0 )
      goto LABEL_157;
    goto LABEL_132;
  }
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v2 + 224) + 176LL))(*(_QWORD *)(v2 + 224), v134, 0);
  v112 = *(_QWORD *)(v2 + 200);
  if ( (*(_BYTE *)(v112 + 879) & 8) != 0 )
  {
LABEL_132:
    v113 = *(void (**)(void))(**(_QWORD **)(v2 + 224) + 1208LL);
    if ( v113 != nullsub_113 )
      v113();
    sub_BA9680(&v302, v251);
    v233 = v2;
    v114 = _mm_loadu_si128(&v303);
    v115 = _mm_loadu_si128(&v304);
    v220 = v305[1];
    v116 = _mm_loadu_si128(v305);
    v268 = _mm_loadu_si128(&v302);
    v269 = v114;
    v238 = v306;
    v270 = v115;
    srcb = v307;
    v271 = v116;
    v218 = v308;
    v215 = v309;
    v227 = v310;
    v243 = v311;
    while ( 1 )
    {
      if ( __PAIR128__((unsigned __int64)srcb, (unsigned __int64)v238) == *(_OWORD *)&v269
        && *(_OWORD *)&v220 == *(_OWORD *)&v268
        && __PAIR128__(v243, v227) == *(_OWORD *)&v271
        && *(_OWORD *)&v270 == __PAIR128__(v215, v218) )
      {
        v2 = v233;
        if ( *(_DWORD *)(*(_QWORD *)(v233 + 200) + 564LL) == 3 )
          goto LABEL_188;
        goto LABEL_158;
      }
      v117 = v276;
      v278 = 0;
      v118 = v276;
      v119 = &v268;
      v277 = sub_C11C50;
      v280 = 0;
      v279 = sub_C11C70;
      v282 = 0;
      v281 = sub_C11C90;
      v120 = sub_C11C30;
      if ( ((unsigned __int8)sub_C11C30 & 1) != 0 )
        goto LABEL_137;
LABEL_138:
      v121 = v120((__int64)v119);
      if ( !v121 )
        break;
LABEL_142:
      v123 = v121;
      if ( *(_QWORD *)(v121 + 16) )
      {
        v124 = *(_BYTE *)(v121 + 33);
        if ( (v124 & 0x1C) == 0 && (v124 & 3) != 1 )
        {
          v125 = sub_BD5D20(v123);
          if ( (v126 <= 4 || *(_DWORD *)v125 != 1836477548 || v125[4] != 46) && !(*(_BYTE *)(v123 + 32) >> 6) )
          {
            v214 = *(_QWORD *)(v233 + 224);
            v201 = *(void (**)())(*(_QWORD *)v214 + 1216LL);
            v202 = sub_31DB510(v233, v123);
            if ( v201 != nullsub_114 )
              ((void (__fastcall *)(__int64, __int64))v201)(v214, v202);
          }
        }
      }
      v127 = v283;
      v285 = 0;
      v128 = v283;
      v129 = &v268;
      v287 = 0;
      v284 = sub_C11BA0;
      v289 = 0;
      v286 = sub_C11BD0;
      v288 = sub_C11C00;
      v130 = sub_C11B70;
      if ( ((unsigned __int8)sub_C11B70 & 1) != 0 )
LABEL_149:
        v130 = *(__int64 (__fastcall **)(__int64))((char *)v130 + v129->m128i_i64[0] - 1);
      while ( !(unsigned __int8)v130((__int64)v129) )
      {
        v127 += 16;
        if ( v290 == v127 )
          goto LABEL_265;
        v131 = *((_QWORD *)v128 + 3);
        v130 = (__int64 (__fastcall *)(__int64))*((_QWORD *)v128 + 2);
        v128 = v127;
        v129 = (__m128i *)((char *)&v268 + v131);
        if ( ((unsigned __int8)v130 & 1) != 0 )
          goto LABEL_149;
      }
    }
    while ( 1 )
    {
      v117 += 16;
      if ( v117 == v283 )
        goto LABEL_265;
      v122 = *((_QWORD *)v118 + 3);
      v120 = (__int64 (__fastcall *)(__int64))*((_QWORD *)v118 + 2);
      v118 = v117;
      v119 = (__m128i *)((char *)&v268 + v122);
      if ( ((unsigned __int8)v120 & 1) != 0 )
        break;
      v121 = v120((__int64)v119);
      if ( v121 )
        goto LABEL_142;
    }
LABEL_137:
    v120 = *(__int64 (__fastcall **)(__int64))((char *)v120 + v119->m128i_i64[0] - 1);
    goto LABEL_138;
  }
LABEL_157:
  if ( *(_DWORD *)(v112 + 564) == 3 )
  {
LABEL_188:
    sub_BA9680(&v312, v251);
    v239 = 0;
    v144 = _mm_loadu_si128(&v313);
    v225 = v2;
    v213 = v315[1];
    v272 = _mm_loadu_si128(&v312);
    v145 = _mm_loadu_si128(&v314);
    v273 = v144;
    v146 = _mm_loadu_si128(v315);
    v234 = v316;
    v274 = v145;
    srcc = v317;
    v275 = v146;
    v212 = v318;
    v216 = v319;
    v244 = v320;
    while ( __PAIR128__((unsigned __int64)srcc, v234) != *(_OWORD *)&v273
         || *(_OWORD *)&v213 != *(_OWORD *)&v272
         || __PAIR128__(v244, v216) != *(_OWORD *)&v275
         || *(_OWORD *)&v274 != v212 )
    {
      v147 = (size_t *)v290;
      v148 = (__m128i *)v297;
      v149 = (size_t *)v290;
      v150 = &v272;
      v292 = 0;
      v291 = sub_C11C50;
      v294 = 0;
      v293 = sub_C11C70;
      v296 = 0;
      v295 = sub_C11C90;
      v151 = sub_C11C30;
      if ( ((unsigned __int8)sub_C11C30 & 1) != 0 )
LABEL_191:
        v151 = *(__int64 (__fastcall **)(__int64))((char *)v151 + v150->m128i_i64[0] - 1);
      v152 = v151((__int64)v150);
      if ( !v152 )
      {
        while ( 1 )
        {
          v147 += 2;
          if ( v147 == v297 )
            break;
          v153 = v149[3];
          v151 = (__int64 (__fastcall *)(__int64))v149[2];
          v149 = v147;
          v150 = (__m128i *)((char *)&v272 + v153);
          if ( ((unsigned __int8)v151 & 1) != 0 )
            goto LABEL_191;
          v152 = v151((__int64)v150);
          if ( v152 )
            goto LABEL_196;
        }
LABEL_265:
        BUG();
      }
LABEL_196:
      v154 = v152;
      if ( *(char *)(v152 + 33) < 0
        && (*(_BYTE *)(v152 + 32) & 0xF) != 1
        && !sub_B2FC80(v152)
        && (*(_BYTE *)(v154 + 32) & 0x30) == 0 )
      {
        ++v239;
        v155 = *(_QWORD *)(v225 + 216);
        v221 = *(_QWORD *)(v225 + 224);
        v156 = *(void (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)v221 + 176LL);
        v304.m128i_i16[0] = 257;
        v228 = v156;
        v297[0] = (size_t)".llvm_sympart";
        LOWORD(v298) = 259;
        v157 = sub_E71CB0(v155, v297, 1879002117, 0, 0, (__int64)&v302, 0, v239, 0);
        v228(v221, v157, 0);
        v222 = *(_QWORD *)(v225 + 224);
        v229 = *(void (__fastcall **)(__int64, const char *))(*(_QWORD *)v222 + 512LL);
        v159 = sub_B30A70(v154, v157, v158);
        v229(v222, v159);
        sub_E99300(*(_QWORD ***)(v225 + 224), 1);
        v219 = *(_QWORD *)(v225 + 224);
        v230 = *(_QWORD **)(v225 + 216);
        v160 = sub_31DB510(v225, v154);
        v161 = (unsigned __int8 *)sub_E808D0(v160, 0, v230, 0);
        sub_E9A5B0(v219, v161);
      }
      v162 = v297;
      v297[3] = 0;
      v299 = 0;
      v163 = &v272;
      v297[2] = (size_t)sub_C11BA0;
      v301 = 0;
      v298 = sub_C11BD0;
      v300 = sub_C11C00;
      v164 = sub_C11B70;
      if ( ((unsigned __int8)sub_C11B70 & 1) != 0 )
LABEL_202:
        v164 = *(__int64 (__fastcall **)(__int64))((char *)v164 + v163->m128i_i64[0] - 1);
      while ( !(unsigned __int8)v164((__int64)v163) )
      {
        if ( &v302 == ++v148 )
          goto LABEL_265;
        v165 = v162[3];
        v164 = (__int64 (__fastcall *)(__int64))v162[2];
        v162 = (size_t *)v148;
        v163 = (__m128i *)((char *)&v272 + v165);
        if ( ((unsigned __int8)v164 & 1) != 0 )
          goto LABEL_202;
      }
    }
    v2 = v225;
  }
LABEL_158:
  v135 = *(void (**)())(*(_QWORD *)v2 + 256LL);
  if ( v135 != nullsub_1706 )
    ((void (__fastcall *)(__int64, __int64 *))v135)(v2, v251);
  v136 = *(_QWORD *)(v2 + 448);
  *(_QWORD *)(v2 + 240) = 0;
  *(_QWORD *)(v2 + 448) = 0;
  if ( v136 )
    sub_31D8060(v136);
  sub_E99F70(*(_QWORD **)(v2 + 224), 0);
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v2 + 224) + 72LL))(*(_QWORD *)(v2 + 224));
  v137 = *(_QWORD *)(v2 + 752);
  *(_QWORD *)(v2 + 752) = 0;
  if ( v137 )
    sub_31D7970(v137, 0);
  v138 = *(_QWORD *)(v2 + 744);
  *(_QWORD *)(v2 + 744) = 0;
  if ( v138 )
  {
    v139 = *(_QWORD *)(v138 + 24);
    v140 = v139 + 8LL * *(unsigned int *)(v138 + 32);
    if ( v139 != v140 )
    {
      do
      {
        v141 = *(_QWORD *)(v140 - 8);
        v140 -= 8LL;
        if ( v141 )
        {
          v142 = *(_QWORD *)(v141 + 24);
          if ( v142 != v141 + 40 )
            _libc_free(v142);
          j_j___libc_free_0(v141);
        }
      }
      while ( v139 != v140 );
      v140 = *(_QWORD *)(v138 + 24);
    }
    if ( v140 != v138 + 40 )
      _libc_free(v140);
    if ( *(_QWORD *)v138 != v138 + 16 )
      _libc_free(*(_QWORD *)v138);
    j_j___libc_free_0(v138);
  }
  if ( !s[4] )
    _libc_free(v326);
  if ( v321 != v323 )
    _libc_free((unsigned __int64)v321);
  if ( v252 )
    j_j___libc_free_0((unsigned __int64)v252);
  return 0;
}
