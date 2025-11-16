// Function: sub_1C77080
// Address: 0x1c77080
//
__int64 __fastcall sub_1C77080(
        __int64 a1,
        _QWORD *a2,
        _QWORD *a3,
        char a4,
        __int64 *a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15,
        __int64 a16,
        _QWORD *a17,
        __int64 a18,
        __int64 a19,
        __int64 a20,
        __int64 a21,
        __int64 a22,
        __int64 a23,
        __int64 a24,
        _QWORD *a25,
        _QWORD *a26,
        __int64 a27,
        __int64 *a28,
        __int64 *a29,
        __int64 a30,
        _QWORD *a31,
        _QWORD *a32,
        __int64 a33,
        _QWORD *a34,
        _QWORD *a35,
        __int64 *a36)
{
  __int64 *v36; // r12
  __int64 *v37; // r13
  char *v38; // r12
  _BYTE *v39; // r15
  _BYTE *v40; // r14
  signed __int64 v41; // r14
  __int64 *v42; // r12
  __int64 v43; // rax
  __int64 *v44; // r12
  _QWORD *v45; // rax
  _QWORD *v46; // rdx
  char v47; // cl
  char *v48; // rax
  __int64 v49; // rdx
  __int64 v50; // r12
  _BYTE *v51; // rsi
  __int64 v52; // r15
  _QWORD *v53; // rdi
  __int64 v54; // rax
  __int64 v55; // rax
  unsigned __int64 v56; // rax
  char *v57; // rbx
  unsigned __int64 v58; // rdx
  bool v59; // cf
  unsigned __int64 v60; // rax
  _BYTE *v61; // r8
  __int64 v62; // rbx
  _BYTE *v63; // rax
  __int64 v64; // rdx
  __int64 v65; // r10
  __int64 v66; // r9
  unsigned __int64 v67; // rdi
  __int64 v68; // rcx
  unsigned __int64 v69; // r8
  __int64 v70; // rcx
  __int64 v71; // rdx
  __int64 v72; // r12
  __int64 v73; // r15
  __int64 j; // r12
  __int64 v75; // r14
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r8
  int v79; // r9d
  double v80; // xmm4_8
  double v81; // xmm5_8
  _QWORD *v82; // rax
  __int64 v83; // rsi
  _QWORD *v84; // rax
  __int64 v85; // rsi
  _QWORD *v86; // rax
  __int64 v87; // rsi
  _QWORD *v88; // rax
  __int64 v89; // rsi
  _QWORD *v90; // rax
  __int64 v91; // rsi
  _QWORD *v92; // rax
  _QWORD *v93; // rax
  _QWORD *v94; // rdx
  char v95; // cl
  char *v96; // rax
  __int64 v97; // rdx
  __int64 v98; // r12
  _BYTE *v99; // rsi
  void **v100; // r13
  _QWORD *v101; // rdi
  void **v102; // rax
  void **v103; // rax
  __int64 v104; // rax
  __int64 v105; // rcx
  unsigned __int64 v106; // rsi
  __int64 v107; // rdx
  __int64 v108; // rax
  _BYTE *v109; // rax
  __int64 v110; // rdx
  __int64 v111; // r12
  __int64 v112; // r14
  __int64 k; // r12
  __int64 v114; // r13
  __int64 v115; // rdx
  __int64 v116; // rcx
  __int64 v117; // r8
  int v118; // r9d
  double v119; // xmm4_8
  double v120; // xmm5_8
  _QWORD *v121; // rax
  __int64 v122; // rsi
  _QWORD *v123; // rax
  __int64 v124; // rsi
  _QWORD *v125; // rax
  __int64 v126; // rsi
  _QWORD *v127; // rax
  __int64 v128; // rsi
  _QWORD *v129; // rax
  __int64 v130; // rsi
  __int64 v131; // rax
  _QWORD *v132; // rax
  _QWORD *v133; // rdx
  char v134; // cl
  __int64 v135; // r13
  _QWORD *v136; // r12
  __int64 v137; // rax
  __int64 v138; // r15
  __int64 ii; // r13
  __int64 v140; // r14
  __int64 v141; // rdx
  __int64 v142; // rcx
  __int64 v143; // r8
  int v144; // r9d
  double v145; // xmm4_8
  double v146; // xmm5_8
  __int64 v147; // r13
  _QWORD *v148; // r12
  __int64 v149; // rax
  __int64 v150; // r15
  __int64 jj; // r13
  __int64 v152; // r14
  __int64 v153; // rdx
  __int64 v154; // rcx
  __int64 v155; // r8
  int v156; // r9d
  double v157; // xmm4_8
  double v158; // xmm5_8
  __int64 v159; // r13
  _QWORD *v160; // r12
  __int64 v161; // rax
  __int64 v162; // r15
  __int64 kk; // r13
  __int64 v164; // r14
  __int64 v165; // rdx
  __int64 v166; // rcx
  __int64 v167; // r8
  int v168; // r9d
  double v169; // xmm4_8
  double v170; // xmm5_8
  __int64 v171; // r12
  __int64 v172; // r13
  __int64 v173; // r14
  __int64 v174; // rsi
  _QWORD *v175; // r15
  __int64 v176; // rax
  __int64 v177; // r12
  __int64 v178; // r13
  _QWORD *v179; // r15
  __int64 v180; // r14
  __int64 v181; // rdx
  __int64 v182; // r13
  __int64 v183; // r12
  __int64 v184; // r14
  __int64 v185; // rsi
  _QWORD *v186; // r15
  __int64 v187; // rax
  __int64 v188; // r12
  __int64 v189; // r13
  _QWORD *v190; // r15
  __int64 v191; // r14
  __int64 v192; // rdx
  __int64 v193; // r13
  __int64 v194; // r12
  __int64 v195; // r14
  __int64 v196; // rsi
  _QWORD *v197; // r15
  __int64 v198; // rax
  __int64 v199; // r12
  __int64 v200; // r13
  _QWORD *v201; // r15
  __int64 v202; // r14
  __int64 v203; // rdx
  __int64 v204; // r12
  __int64 v205; // r13
  __int64 v206; // r14
  __int64 v207; // rsi
  _QWORD *v208; // r15
  __int64 v209; // rax
  __int64 v210; // r12
  __int64 v211; // r13
  _QWORD *v212; // r15
  __int64 v213; // r14
  __int64 v214; // rdx
  __int64 v215; // r13
  __int64 v216; // r12
  __int64 v217; // r14
  __int64 v218; // rsi
  _QWORD *v219; // r15
  __int64 v220; // rax
  __int64 v221; // r13
  __int64 v222; // r14
  __int64 v223; // rdi
  double v224; // xmm4_8
  double v225; // xmm5_8
  __int64 result; // rax
  _QWORD *v227; // r12
  _QWORD *v228; // r13
  __int64 v229; // rax
  __int64 v230; // rdx
  __int64 v231; // rax
  _QWORD *v232; // r12
  _QWORD *v233; // r13
  __int64 v234; // rax
  __int64 v235; // rdx
  __int64 v236; // rax
  _QWORD *v237; // r12
  _QWORD *v238; // r13
  __int64 v239; // rax
  __int64 v240; // rdx
  __int64 v241; // rax
  _QWORD *v242; // r12
  _QWORD *v243; // r13
  __int64 v244; // rsi
  _QWORD *v245; // r12
  _QWORD *v246; // r13
  __int64 v247; // rsi
  _QWORD *v248; // r12
  _QWORD *v249; // r13
  __int64 v250; // rsi
  _QWORD *v251; // r12
  _QWORD *v252; // r13
  __int64 v253; // rsi
  _QWORD *v254; // rbx
  _QWORD *v255; // r12
  __int64 v256; // rsi
  _QWORD *v257; // rbx
  _QWORD *v258; // r12
  __int64 v259; // rsi
  _QWORD *v260; // rbx
  _QWORD *v261; // r12
  __int64 v262; // rsi
  _QWORD *v263; // r12
  _QWORD *v264; // r13
  __int64 v265; // rsi
  _QWORD *v266; // r12
  _QWORD *v267; // r13
  __int64 v268; // rsi
  _QWORD *v269; // r12
  _QWORD *v270; // r13
  __int64 v271; // rsi
  _QWORD *v272; // r12
  _QWORD *v273; // r13
  __int64 v274; // rsi
  __int64 v275; // rax
  _QWORD *v276; // r14
  __int64 v277; // rcx
  unsigned __int64 v278; // rsi
  __int64 v279; // rcx
  _QWORD *v280; // rax
  _QWORD *v281; // rax
  _QWORD *v282; // r14
  __int64 v283; // rcx
  unsigned __int64 v284; // rsi
  __int64 v285; // rcx
  _QWORD *v286; // rax
  _QWORD *v287; // rax
  _QWORD *v288; // r14
  __int64 v289; // rcx
  unsigned __int64 v290; // rsi
  __int64 v291; // rcx
  __int64 v292; // rax
  _QWORD *v293; // r14
  __int64 v294; // rcx
  unsigned __int64 v295; // rsi
  __int64 v296; // rcx
  _QWORD *v297; // rax
  _QWORD *v298; // rax
  _QWORD *v299; // r14
  __int64 v300; // rcx
  unsigned __int64 v301; // rsi
  __int64 v302; // rcx
  __int64 v303; // rdx
  _QWORD *v304; // r14
  __int64 v305; // rax
  _QWORD *v306; // r14
  __int64 v307; // rcx
  unsigned __int64 v308; // rsi
  __int64 v309; // rcx
  __int64 v310; // rdx
  _QWORD *v311; // r14
  _QWORD *v312; // rax
  _QWORD *v313; // rax
  _QWORD *v314; // r14
  __int64 v315; // rcx
  unsigned __int64 v316; // rsi
  __int64 v317; // rcx
  __int64 v318; // rdx
  _QWORD *v319; // r14
  __int64 v320; // rax
  _QWORD *v321; // r14
  __int64 v322; // rcx
  unsigned __int64 v323; // rsi
  __int64 v324; // rcx
  __int64 v325; // rdx
  _QWORD *v326; // r14
  __int64 *v327; // rax
  __int64 v328; // r13
  char *dest; // [rsp+0h] [rbp-250h]
  __int64 v331; // [rsp+10h] [rbp-240h]
  _BYTE *v332; // [rsp+10h] [rbp-240h]
  _BYTE *v333; // [rsp+10h] [rbp-240h]
  __int64 v335; // [rsp+20h] [rbp-230h]
  signed __int64 v336; // [rsp+20h] [rbp-230h]
  size_t n; // [rsp+38h] [rbp-218h]
  char *v339; // [rsp+40h] [rbp-210h]
  _BYTE *v340; // [rsp+40h] [rbp-210h]
  _BYTE *v341; // [rsp+48h] [rbp-208h]
  __int64 v342; // [rsp+48h] [rbp-208h]
  _BYTE *v343; // [rsp+48h] [rbp-208h]
  char *v344; // [rsp+48h] [rbp-208h]
  __int64 v348; // [rsp+60h] [rbp-1F0h]
  __int64 v349; // [rsp+68h] [rbp-1E8h]
  char *v350; // [rsp+68h] [rbp-1E8h]
  __int64 *v351; // [rsp+68h] [rbp-1E8h]
  __int64 v352; // [rsp+68h] [rbp-1E8h]
  __int64 v353; // [rsp+68h] [rbp-1E8h]
  __int64 v354; // [rsp+68h] [rbp-1E8h]
  __int64 v355; // [rsp+68h] [rbp-1E8h]
  __int64 v356; // [rsp+68h] [rbp-1E8h]
  __int64 v357; // [rsp+68h] [rbp-1E8h]
  __int64 v358; // [rsp+68h] [rbp-1E8h]
  __int64 v359; // [rsp+68h] [rbp-1E8h]
  void *src; // [rsp+70h] [rbp-1E0h] BYREF
  __int64 *v361; // [rsp+78h] [rbp-1D8h]
  __int64 *v362; // [rsp+80h] [rbp-1D0h]
  _BYTE *v363; // [rsp+90h] [rbp-1C0h] BYREF
  _BYTE *v364; // [rsp+98h] [rbp-1B8h]
  _BYTE *v365; // [rsp+A0h] [rbp-1B0h]
  _BYTE *v366; // [rsp+B0h] [rbp-1A0h] BYREF
  _BYTE *v367; // [rsp+B8h] [rbp-198h]
  _BYTE *v368; // [rsp+C0h] [rbp-190h]
  void *v369; // [rsp+D0h] [rbp-180h]
  __int64 v370; // [rsp+D8h] [rbp-178h] BYREF
  __int64 v371; // [rsp+E0h] [rbp-170h]
  __int64 v372; // [rsp+E8h] [rbp-168h]
  __int64 v373; // [rsp+F0h] [rbp-160h]
  void **p_src; // [rsp+100h] [rbp-150h] BYREF
  __int64 v375; // [rsp+108h] [rbp-148h] BYREF
  _BYTE **v376; // [rsp+110h] [rbp-140h]
  __int64 v377; // [rsp+118h] [rbp-138h]
  __int64 m; // [rsp+120h] [rbp-130h]
  __int64 v379; // [rsp+130h] [rbp-120h] BYREF
  _QWORD *v380; // [rsp+138h] [rbp-118h]
  __int64 v381; // [rsp+140h] [rbp-110h]
  unsigned int v382; // [rsp+148h] [rbp-108h]
  _QWORD *v383; // [rsp+158h] [rbp-F8h]
  unsigned int v384; // [rsp+168h] [rbp-E8h]
  char v385; // [rsp+170h] [rbp-E0h]
  char v386; // [rsp+179h] [rbp-D7h]
  __int64 v387; // [rsp+180h] [rbp-D0h] BYREF
  _QWORD *v388; // [rsp+188h] [rbp-C8h]
  __int64 v389; // [rsp+190h] [rbp-C0h]
  unsigned int v390; // [rsp+198h] [rbp-B8h]
  _QWORD *v391; // [rsp+1A8h] [rbp-A8h]
  unsigned int v392; // [rsp+1B8h] [rbp-98h]
  char v393; // [rsp+1C0h] [rbp-90h]
  char v394; // [rsp+1C9h] [rbp-87h]
  const char *v395; // [rsp+1D0h] [rbp-80h] BYREF
  __int64 v396; // [rsp+1D8h] [rbp-78h]
  __int64 v397; // [rsp+1E0h] [rbp-70h]
  __int64 v398; // [rsp+1E8h] [rbp-68h]
  __int64 i; // [rsp+1F0h] [rbp-60h]
  _QWORD *v400; // [rsp+1F8h] [rbp-58h]
  unsigned int v401; // [rsp+208h] [rbp-48h]
  char v402; // [rsp+210h] [rbp-40h]
  char v403; // [rsp+219h] [rbp-37h]

  src = 0;
  v361 = 0;
  v362 = 0;
  v363 = 0;
  v364 = 0;
  v365 = 0;
  v366 = 0;
  v367 = 0;
  v368 = 0;
  sub_1292090((__int64)&src, 0, &a18);
  v36 = v361;
  v37 = v362;
  if ( v361 == v362 )
  {
    sub_1292090((__int64)&src, v362, &a19);
    v38 = (char *)v361;
    v37 = v362;
  }
  else
  {
    if ( v361 )
    {
      *v361 = a19;
      v36 = v361;
      v37 = v362;
    }
    v38 = (char *)(v36 + 1);
    v361 = (__int64 *)v38;
  }
  v39 = *(_BYTE **)(a1 + 32);
  v40 = *(_BYTE **)(a1 + 40);
  if ( v39 != v40 )
  {
    v41 = v40 - v39;
    if ( (char *)v37 - v38 >= (unsigned __int64)v41 )
    {
      memmove(v38, *(const void **)(a1 + 32), v41);
      v361 = (__int64 *)((char *)v361 + v41);
      v38 = (char *)v361;
      if ( v361 != v362 )
        goto LABEL_8;
      goto LABEL_406;
    }
    v56 = v41 >> 3;
    v57 = (char *)(v38 - (_BYTE *)src);
    v58 = (v38 - (_BYTE *)src) >> 3;
    if ( v41 >> 3 > 0xFFFFFFFFFFFFFFFLL - v58 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v56 < v58 )
      v56 = (v38 - (_BYTE *)src) >> 3;
    v59 = __CFADD__(v58, v56);
    v60 = v58 + v56;
    if ( v59 )
    {
      v328 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v60 )
      {
        v350 = &v57[v41];
        if ( src != v38 )
        {
          v341 = src;
          memmove(0, src, v38 - (_BYTE *)src);
          memcpy(v57, v39, v41);
          v61 = v341;
          v62 = 0;
          v38 = v350;
          v351 = v37;
          v37 = 0;
LABEL_43:
          j_j___libc_free_0(v61, (char *)v351 - v61);
LABEL_44:
          src = (void *)v62;
          v361 = (__int64 *)v38;
          v362 = v37;
          goto LABEL_45;
        }
        v343 = src;
        memcpy((void *)(v38 - (_BYTE *)src), v39, v41);
        v38 = &v57[v41];
        v62 = 0;
        v327 = 0;
        v61 = v343;
        goto LABEL_492;
      }
      if ( v60 > 0xFFFFFFFFFFFFFFFLL )
        v60 = 0xFFFFFFFFFFFFFFFLL;
      v328 = 8 * v60;
    }
    v62 = sub_22077B0(v328);
    v351 = v362;
    v37 = (__int64 *)(v62 + v328);
    v339 = (char *)v361;
    n = (char *)v361 - v38;
    v336 = v41 + v38 - (_BYTE *)src;
    v344 = (char *)v361 + v41 - (_QWORD)src + v62;
    if ( v38 == src )
    {
      v333 = src;
      memcpy((void *)(v62 + v38 - (_BYTE *)src), v39, v41);
      v61 = v333;
      if ( v38 == v339 )
      {
        v327 = v37;
        v38 = v344;
        v37 = v351;
LABEL_492:
        v351 = v37;
        v37 = v327;
        goto LABEL_493;
      }
    }
    else
    {
      v332 = src;
      dest = (char *)(v62 + v38 - (_BYTE *)src);
      memmove((void *)v62, src, v38 - (_BYTE *)src);
      memcpy(dest, v39, v41);
      v61 = v332;
      if ( v38 == v339 )
      {
        v38 = v344;
        goto LABEL_43;
      }
    }
    v340 = v61;
    memcpy((void *)(v62 + v336), v38, n);
    v38 = v344;
    v61 = v340;
LABEL_493:
    if ( !v61 )
      goto LABEL_44;
    goto LABEL_43;
  }
LABEL_45:
  if ( v38 != (char *)v37 )
  {
    if ( !v38 )
    {
LABEL_9:
      v42 = (__int64 *)(v38 + 8);
      v43 = a23;
      v361 = v42;
      if ( v42 != v37 )
        goto LABEL_10;
      goto LABEL_445;
    }
LABEL_8:
    *(_QWORD *)v38 = a22;
    v37 = v362;
    v38 = (char *)v361;
    goto LABEL_9;
  }
LABEL_406:
  sub_1292090((__int64)&src, v38, &a22);
  v42 = v361;
  v37 = v362;
  if ( v361 != v362 )
  {
    if ( !v361 )
      goto LABEL_11;
    v43 = a23;
LABEL_10:
    *v42 = v43;
    v42 = v361;
LABEL_11:
    v44 = v42 + 1;
    v361 = v44;
    goto LABEL_12;
  }
LABEL_445:
  sub_1292090((__int64)&src, v37, &a23);
  v44 = v361;
LABEL_12:
  if ( v363 != v364 )
    v364 = v363;
  sub_13FC0C0((__int64)&v363, ((char *)v44 - (_BYTE *)src) >> 3);
  v379 = 0;
  v382 = 128;
  v45 = (_QWORD *)sub_22077B0(0x2000);
  v381 = 0;
  v380 = v45;
  v396 = 2;
  v397 = 0;
  v398 = -8;
  v46 = &v45[8 * (unsigned __int64)v382];
  v395 = (const char *)&unk_49E6B50;
  for ( i = 0; v46 != v45; v45 += 8 )
  {
    if ( v45 )
    {
      v47 = v396;
      v45[2] = 0;
      v45[3] = -8;
      *v45 = &unk_49E6B50;
      v45[1] = v47 & 6;
      v45[4] = i;
    }
  }
  v48 = (char *)src;
  v385 = 0;
  v386 = 1;
  v49 = ((char *)v361 - (_BYTE *)src) >> 3;
  if ( (_DWORD)v49 )
  {
    v50 = 0;
    v349 = 8LL * (unsigned int)(v49 - 1);
    while ( 1 )
    {
      v395 = ".s1";
      LOWORD(v397) = 259;
      v55 = sub_1AB5760(*(_QWORD *)&v48[v50], (__int64)&v379, (__int64 *)&v395, a6, 0, 0);
      v51 = v364;
      v387 = v55;
      v52 = v55;
      if ( v364 == v365 )
      {
        sub_1292090((__int64)&v363, v364, &v387);
        v52 = v387;
      }
      else
      {
        if ( v364 )
        {
          *(_QWORD *)v364 = v55;
          v51 = v364;
          v52 = v387;
        }
        v364 = v51 + 8;
      }
      v53 = sub_1C76B50((__int64)&v379, *(_QWORD *)((char *)src + v50));
      v54 = v53[2];
      if ( v54 != v52 )
      {
        if ( v54 != 0 && v54 != -8 && v54 != -16 )
          sub_1649B30(v53);
        v53[2] = v52;
        if ( v52 != -8 && v52 != 0 && v52 != -16 )
          sub_164C220((__int64)v53);
      }
      if ( a5 )
        sub_1404520((__int64)a5, *(_QWORD *)((char *)src + v50), v387, a1);
      if ( v349 == v50 )
        break;
      v48 = (char *)src;
      v50 += 8;
    }
  }
  v63 = v363;
  v64 = *(_QWORD *)v363;
  v65 = a6 + 72;
  v66 = a24 + 24;
  v335 = a6 + 72;
  v331 = a24 + 24;
  if ( a6 + 72 != *(_QWORD *)v363 + 24LL && v65 != v66 )
  {
    v67 = *(_QWORD *)(a6 + 72) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)((*(_QWORD *)(v64 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 8) = v65;
    *(_QWORD *)(a6 + 72) = *(_QWORD *)(a6 + 72) & 7LL | *(_QWORD *)(v64 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    v68 = *(_QWORD *)(a24 + 24);
    *(_QWORD *)(v67 + 8) = v66;
    v69 = v68 & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v64 + 24) = v68 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)(v64 + 24) & 7LL;
    v70 = a24;
    *(_QWORD *)(v69 + 8) = v64 + 24;
    *(_QWORD *)(a24 + 24) = v67 | *(_QWORD *)(v70 + 24) & 7LL;
  }
  if ( !a4 )
  {
    *a2 = sub_1C76EB0((_QWORD *)a1, 0, (__int64)&v379, a15, a5);
    v63 = v363;
  }
  v71 = (v364 - v63) >> 3;
  if ( (_DWORD)v71 )
  {
    v352 = 0;
    v342 = 8LL * (unsigned int)(v71 - 1);
    while ( 1 )
    {
      v72 = *(_QWORD *)&v63[v352];
      v73 = *(_QWORD *)(v72 + 48);
      for ( j = v72 + 40; v73 != j; v73 = *(_QWORD *)(v73 + 8) )
      {
        v75 = v73 - 24;
        if ( !v73 )
          v75 = 0;
        sub_1B75040((__int64 *)&v395, (__int64)&v379, 3, 0, 0);
        sub_1B79630((__int64 *)&v395, v75, v76, v77, v78, v79, a7, a8, a9, a10, v80, v81, a13, a14);
        sub_1B75110((__int64 *)&v395);
      }
      if ( v342 == v352 )
        break;
      v352 += 8;
      v63 = v363;
    }
  }
  v82 = sub_1C76B50((__int64)&v379, a18);
  v83 = a19;
  *a25 = v82[2];
  v84 = sub_1C76B50((__int64)&v379, v83);
  v85 = a20;
  *a26 = v84[2];
  v86 = sub_1C76B50((__int64)&v379, v85);
  v87 = a21;
  *(_QWORD *)a27 = v86[2];
  v88 = sub_1C76B50((__int64)&v379, v87);
  v89 = a22;
  *a28 = v88[2];
  v90 = sub_1C76B50((__int64)&v379, v89);
  v91 = a23;
  *a29 = v90[2];
  v92 = sub_1C76B50((__int64)&v379, v91);
  *(_QWORD *)a30 = v92[2];
  *a17 = sub_1C76B50((__int64)&v379, a16)[2];
  if ( v367 != v366 )
    v367 = v366;
  sub_13FC0C0((__int64)&v366, ((char *)v361 - (_BYTE *)src) >> 3);
  v387 = 0;
  v390 = 128;
  v93 = (_QWORD *)sub_22077B0(0x2000);
  v389 = 0;
  v388 = v93;
  v396 = 2;
  v94 = &v93[8 * (unsigned __int64)v390];
  v397 = 0;
  v398 = -8;
  v395 = (const char *)&unk_49E6B50;
  for ( i = 0; v94 != v93; v93 += 8 )
  {
    if ( v93 )
    {
      v95 = v396;
      v93[2] = 0;
      v93[3] = -8;
      *v93 = &unk_49E6B50;
      v93[1] = v95 & 6;
      v93[4] = i;
    }
  }
  v96 = (char *)src;
  v393 = 0;
  v394 = 1;
  v97 = ((char *)v361 - (_BYTE *)src) >> 3;
  if ( (_DWORD)v97 )
  {
    v98 = 0;
    v353 = 8LL * (unsigned int)(v97 - 1);
    while ( 1 )
    {
      v395 = ".s2";
      LOWORD(v397) = 259;
      v103 = (void **)sub_1AB5760(*(_QWORD *)&v96[v98], (__int64)&v387, (__int64 *)&v395, a6, 0, 0);
      v99 = v367;
      p_src = v103;
      v100 = v103;
      if ( v367 == v368 )
      {
        sub_1292090((__int64)&v366, v367, &p_src);
        v100 = p_src;
      }
      else
      {
        if ( v367 )
        {
          *(_QWORD *)v367 = v103;
          v99 = v367;
          v100 = p_src;
        }
        v367 = v99 + 8;
      }
      v101 = sub_1C76B50((__int64)&v387, *(_QWORD *)((char *)src + v98));
      v102 = (void **)v101[2];
      if ( v102 != v100 )
      {
        if ( v102 + 1 != 0 && v102 != 0 && v102 != (void **)-16LL )
          sub_1649B30(v101);
        v101[2] = v100;
        if ( v100 + 1 != 0 && v100 != 0 && v100 != (void **)-16LL )
          sub_164C220((__int64)v101);
      }
      if ( a5 )
        sub_1404520((__int64)a5, *(_QWORD *)((char *)src + v98), (__int64)p_src, a1);
      if ( v353 == v98 )
        break;
      v96 = (char *)src;
      v98 += 8;
    }
  }
  v104 = *(_QWORD *)v366;
  v105 = *(_QWORD *)v366 + 24LL;
  if ( v335 != v105 && v335 != v331 )
  {
    v106 = *(_QWORD *)(a6 + 72) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)((*(_QWORD *)(v104 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 8) = v335;
    *(_QWORD *)(a6 + 72) = *(_QWORD *)(a6 + 72) & 7LL | *(_QWORD *)(v104 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    v107 = *(_QWORD *)(a24 + 24);
    *(_QWORD *)(v106 + 8) = v331;
    *(_QWORD *)(v104 + 24) = v107 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)(v104 + 24) & 7LL;
    v108 = a24;
    *(_QWORD *)((v107 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v105;
    *(_QWORD *)(a24 + 24) = v106 | *(_QWORD *)(v108 + 24) & 7LL;
  }
  *a3 = sub_1C76EB0((_QWORD *)a1, 0, (__int64)&v387, a15, a5);
  v109 = v366;
  v110 = (v367 - v366) >> 3;
  if ( (_DWORD)v110 )
  {
    v354 = 0;
    v348 = 8LL * (unsigned int)(v110 - 1);
    while ( 1 )
    {
      v111 = *(_QWORD *)&v109[v354];
      v112 = *(_QWORD *)(v111 + 48);
      for ( k = v111 + 40; k != v112; v112 = *(_QWORD *)(v112 + 8) )
      {
        v114 = v112 - 24;
        if ( !v112 )
          v114 = 0;
        sub_1B75040((__int64 *)&v395, (__int64)&v387, 3, 0, 0);
        sub_1B79630((__int64 *)&v395, v114, v115, v116, v117, v118, a7, a8, a9, a10, v119, v120, a13, a14);
        sub_1B75110((__int64 *)&v395);
      }
      if ( v348 == v354 )
        break;
      v354 += 8;
      v109 = v366;
    }
  }
  v121 = sub_1C76B50((__int64)&v387, a18);
  v122 = a19;
  *a31 = v121[2];
  v123 = sub_1C76B50((__int64)&v387, v122);
  v124 = a20;
  *a32 = v123[2];
  v125 = sub_1C76B50((__int64)&v387, v124);
  v126 = a21;
  *(_QWORD *)a33 = v125[2];
  v127 = sub_1C76B50((__int64)&v387, v126);
  v128 = a22;
  *a34 = v127[2];
  v129 = sub_1C76B50((__int64)&v387, v128);
  v130 = a23;
  *a35 = v129[2];
  v131 = sub_1C76B50((__int64)&v387, v130)[2];
  v395 = 0;
  LODWORD(v398) = 128;
  *a36 = v131;
  v132 = (_QWORD *)sub_22077B0(0x2000);
  v397 = 0;
  v396 = (__int64)v132;
  v375 = 2;
  v133 = &v132[8 * (unsigned __int64)(unsigned int)v398];
  p_src = (void **)&unk_49E6B50;
  v376 = 0;
  v377 = -8;
  for ( m = 0; v133 != v132; v132 += 8 )
  {
    if ( v132 )
    {
      v134 = v375;
      v132[2] = 0;
      v132[3] = -8;
      *v132 = &unk_49E6B50;
      v132[1] = v134 & 6;
      v132[4] = m;
    }
  }
  v402 = 0;
  v403 = 1;
  sub_1C76340((__int64)&v395);
  if ( v402 )
  {
    if ( v401 )
    {
      v248 = v400;
      v249 = &v400[2 * v401];
      do
      {
        if ( *v248 != -4 && *v248 != -8 )
        {
          v250 = v248[1];
          if ( v250 )
            sub_161E7C0((__int64)(v248 + 1), v250);
        }
        v248 += 2;
      }
      while ( v249 != v248 );
    }
    j___libc_free_0(v400);
    v402 = 0;
  }
  v135 = *a25;
  v136 = sub_1C76B50((__int64)&v395, a24);
  v137 = v136[2];
  if ( v135 != v137 )
  {
    if ( v137 != -8 && v137 != 0 && v137 != -16 )
      sub_1649B30(v136);
    v136[2] = v135;
    if ( v135 != -8 && v135 != 0 && v135 != -16 )
      sub_164C220((__int64)v136);
  }
  v138 = *(_QWORD *)(a23 + 48);
  for ( ii = a23 + 40; ii != v138; v138 = *(_QWORD *)(v138 + 8) )
  {
    v140 = v138 - 24;
    if ( !v138 )
      v140 = 0;
    sub_1B75040((__int64 *)&p_src, (__int64)&v395, 3, 0, 0);
    sub_1B79630((__int64 *)&p_src, v140, v141, v142, v143, v144, a7, a8, a9, a10, v145, v146, a13, a14);
    sub_1B75110((__int64 *)&p_src);
  }
  sub_1C76340((__int64)&v395);
  if ( v402 )
  {
    if ( v401 )
    {
      v245 = v400;
      v246 = &v400[2 * v401];
      do
      {
        if ( *v245 != -4 && *v245 != -8 )
        {
          v247 = v245[1];
          if ( v247 )
            sub_161E7C0((__int64)(v245 + 1), v247);
        }
        v245 += 2;
      }
      while ( v246 != v245 );
    }
    j___libc_free_0(v400);
    v402 = 0;
  }
  v147 = *a31;
  v148 = sub_1C76B50((__int64)&v395, a24);
  v149 = v148[2];
  if ( v147 != v149 )
  {
    if ( v149 != -8 && v149 != 0 && v149 != -16 )
      sub_1649B30(v148);
    v148[2] = v147;
    if ( v147 != -8 && v147 != 0 && v147 != -16 )
      sub_164C220((__int64)v148);
  }
  v150 = *(_QWORD *)(*(_QWORD *)a30 + 48LL);
  for ( jj = *(_QWORD *)a30 + 40LL; jj != v150; v150 = *(_QWORD *)(v150 + 8) )
  {
    v152 = v150 - 24;
    if ( !v150 )
      v152 = 0;
    sub_1B75040((__int64 *)&p_src, (__int64)&v395, 3, 0, 0);
    sub_1B79630((__int64 *)&p_src, v152, v153, v154, v155, v156, a7, a8, a9, a10, v157, v158, a13, a14);
    sub_1B75110((__int64 *)&p_src);
  }
  sub_1C76340((__int64)&v395);
  if ( v402 )
  {
    if ( v401 )
    {
      v242 = v400;
      v243 = &v400[2 * v401];
      do
      {
        if ( *v242 != -8 && *v242 != -4 )
        {
          v244 = v242[1];
          if ( v244 )
            sub_161E7C0((__int64)(v242 + 1), v244);
        }
        v242 += 2;
      }
      while ( v243 != v242 );
    }
    j___libc_free_0(v400);
    v402 = 0;
  }
  v159 = *a36;
  v160 = sub_1C76B50((__int64)&v395, a23);
  v161 = v160[2];
  if ( v159 != v161 )
  {
    if ( v161 != -8 && v161 != 0 && v161 != -16 )
      sub_1649B30(v160);
    v160[2] = v159;
    if ( v159 != -8 && v159 != 0 && v159 != -16 )
      sub_164C220((__int64)v160);
  }
  v162 = *(_QWORD *)(a24 + 48);
  for ( kk = a24 + 40; kk != v162; v162 = *(_QWORD *)(v162 + 8) )
  {
    v164 = v162 - 24;
    if ( !v162 )
      v164 = 0;
    sub_1B75040((__int64 *)&p_src, (__int64)&v395, 3, 0, 0);
    sub_1B79630((__int64 *)&p_src, v164, v165, v166, v167, v168, a7, a8, a9, a10, v169, v170, a13, a14);
    sub_1B75110((__int64 *)&p_src);
  }
  sub_1C76340((__int64)&v395);
  if ( v402 )
  {
    if ( v401 )
    {
      v251 = v400;
      v252 = &v400[2 * v401];
      do
      {
        if ( *v251 != -8 && *v251 != -4 )
        {
          v253 = v251[1];
          if ( v253 )
            sub_161E7C0((__int64)(v251 + 1), v253);
        }
        v251 += 2;
      }
      while ( v252 != v251 );
    }
    j___libc_free_0(v400);
    v402 = 0;
  }
  v171 = *(_QWORD *)(a20 + 48);
  v172 = *(_QWORD *)(*(_QWORD *)a27 + 48LL);
  v355 = *(_QWORD *)a27 + 40LL;
  if ( v355 != v172 )
  {
    do
    {
      v173 = v171 - 24;
      if ( !v171 )
        v173 = 0;
      v174 = v172 - 24;
      if ( !v172 )
        v174 = 0;
      v175 = sub_1C76B50((__int64)&v395, v174);
      v176 = v175[2];
      if ( v173 != v176 )
      {
        if ( v176 != 0 && v176 != -8 && v176 != -16 )
          sub_1649B30(v175);
        v175[2] = v173;
        if ( v173 != 0 && v173 != -8 && v173 != -16 )
          sub_164C220((__int64)v175);
      }
      v172 = *(_QWORD *)(v172 + 8);
      v171 = *(_QWORD *)(v171 + 8);
    }
    while ( v355 != v172 );
    v177 = *(_QWORD *)(*(_QWORD *)a27 + 48LL);
    v178 = *(_QWORD *)a27 + 40LL;
    if ( v178 != v177 )
    {
      v179 = a26;
      while ( v177 )
      {
        if ( *(_BYTE *)(v177 - 8) == 77 )
        {
          v180 = v177 - 24;
          v181 = (*(_BYTE *)(v177 - 1) & 0x40) != 0
               ? *(_QWORD *)(v177 - 32)
               : v180 - 24LL * (*(_DWORD *)(v177 - 4) & 0xFFFFFFF);
          if ( *v179 == *(_QWORD *)(v181 + 24LL * *(unsigned int *)(v177 + 32) + 8) )
          {
            v286 = sub_1C76B50((__int64)&v395, v177 - 24);
            v287 = sub_1C74210(v286[2], a21, a22);
            if ( (*(_BYTE *)(v177 - 1) & 0x40) != 0 )
              v288 = *(_QWORD **)(v177 - 32);
            else
              v288 = (_QWORD *)(v180 - 24LL * (*(_DWORD *)(v177 - 4) & 0xFFFFFFF));
            if ( *v288 )
            {
              v289 = v288[1];
              v290 = v288[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v290 = v289;
              if ( v289 )
                *(_QWORD *)(v289 + 16) = v290 | *(_QWORD *)(v289 + 16) & 3LL;
            }
            *v288 = v287;
            if ( v287 )
            {
              v291 = v287[1];
              v288[1] = v291;
              if ( v291 )
                *(_QWORD *)(v291 + 16) = (unsigned __int64)(v288 + 1) | *(_QWORD *)(v291 + 16) & 3LL;
              v288[2] = (unsigned __int64)(v287 + 1) | v288[2] & 3LL;
              v287[1] = v288;
            }
          }
          else if ( *v179 == *(_QWORD *)(v181 + 24LL * *(unsigned int *)(v177 + 32) + 16) )
          {
            v312 = sub_1C76B50((__int64)&v395, v177 - 24);
            v313 = sub_1C74210(v312[2], a21, a22);
            if ( (*(_BYTE *)(v177 - 1) & 0x40) != 0 )
              v314 = *(_QWORD **)(v177 - 32);
            else
              v314 = (_QWORD *)(v180 - 24LL * (*(_DWORD *)(v177 - 4) & 0xFFFFFFF));
            if ( v314[3] )
            {
              v315 = v314[4];
              v316 = v314[5] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v316 = v315;
              if ( v315 )
                *(_QWORD *)(v315 + 16) = v316 | *(_QWORD *)(v315 + 16) & 3LL;
            }
            v314[3] = v313;
            if ( v313 )
            {
              v317 = v313[1];
              v314[4] = v317;
              if ( v317 )
                *(_QWORD *)(v317 + 16) = (unsigned __int64)(v314 + 4) | *(_QWORD *)(v317 + 16) & 3LL;
              v318 = v314[5];
              v319 = v314 + 3;
              v319[2] = (unsigned __int64)(v313 + 1) | v318 & 3;
              v313[1] = v319;
            }
          }
          v177 = *(_QWORD *)(v177 + 8);
          if ( v178 != v177 )
            continue;
        }
        goto LABEL_159;
      }
LABEL_506:
      BUG();
    }
  }
LABEL_159:
  sub_1C76340((__int64)&v395);
  if ( v402 )
  {
    if ( v401 )
    {
      v272 = v400;
      v273 = &v400[2 * v401];
      do
      {
        if ( *v272 != -8 && *v272 != -4 )
        {
          v274 = v272[1];
          if ( v274 )
            sub_161E7C0((__int64)(v272 + 1), v274);
        }
        v272 += 2;
      }
      while ( v273 != v272 );
    }
    j___libc_free_0(v400);
    v402 = 0;
  }
  v182 = *(_QWORD *)(*(_QWORD *)a33 + 48LL);
  v356 = *(_QWORD *)a33 + 40LL;
  v183 = *(_QWORD *)(*(_QWORD *)a27 + 48LL);
  if ( v356 != v182 )
  {
    do
    {
      v184 = v183 - 24;
      if ( !v183 )
        v184 = 0;
      v185 = v182 - 24;
      if ( !v182 )
        v185 = 0;
      v186 = sub_1C76B50((__int64)&v395, v185);
      v187 = v186[2];
      if ( v184 != v187 )
      {
        if ( v187 != -8 && v187 != 0 && v187 != -16 )
          sub_1649B30(v186);
        v186[2] = v184;
        if ( v184 != -8 && v184 != 0 && v184 != -16 )
          sub_164C220((__int64)v186);
      }
      v182 = *(_QWORD *)(v182 + 8);
      v183 = *(_QWORD *)(v183 + 8);
    }
    while ( v356 != v182 );
    v188 = *(_QWORD *)(*(_QWORD *)a33 + 48LL);
    v189 = *(_QWORD *)a33 + 40LL;
    if ( v189 != v188 )
    {
      v190 = a32;
      while ( v188 )
      {
        if ( *(_BYTE *)(v188 - 8) == 77 )
        {
          v191 = v188 - 24;
          v192 = (*(_BYTE *)(v188 - 1) & 0x40) != 0
               ? *(_QWORD *)(v188 - 32)
               : v191 - 24LL * (*(_DWORD *)(v188 - 4) & 0xFFFFFFF);
          if ( *v190 == *(_QWORD *)(v192 + 24LL * *(unsigned int *)(v188 + 32) + 8) )
          {
            v280 = sub_1C76B50((__int64)&v395, v188 - 24);
            v281 = sub_1C74210(v280[2], *a28, *a29);
            if ( (*(_BYTE *)(v188 - 1) & 0x40) != 0 )
              v282 = *(_QWORD **)(v188 - 32);
            else
              v282 = (_QWORD *)(v191 - 24LL * (*(_DWORD *)(v188 - 4) & 0xFFFFFFF));
            if ( *v282 )
            {
              v283 = v282[1];
              v284 = v282[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v284 = v283;
              if ( v283 )
                *(_QWORD *)(v283 + 16) = v284 | *(_QWORD *)(v283 + 16) & 3LL;
            }
            *v282 = v281;
            if ( v281 )
            {
              v285 = v281[1];
              v282[1] = v285;
              if ( v285 )
                *(_QWORD *)(v285 + 16) = (unsigned __int64)(v282 + 1) | *(_QWORD *)(v285 + 16) & 3LL;
              v282[2] = (unsigned __int64)(v281 + 1) | v282[2] & 3LL;
              v281[1] = v282;
            }
          }
          else if ( *v190 == *(_QWORD *)(v192 + 24LL * *(unsigned int *)(v188 + 32) + 16) )
          {
            v297 = sub_1C76B50((__int64)&v395, v188 - 24);
            v298 = sub_1C74210(v297[2], *a28, *a29);
            if ( (*(_BYTE *)(v188 - 1) & 0x40) != 0 )
              v299 = *(_QWORD **)(v188 - 32);
            else
              v299 = (_QWORD *)(v191 - 24LL * (*(_DWORD *)(v188 - 4) & 0xFFFFFFF));
            if ( v299[3] )
            {
              v300 = v299[4];
              v301 = v299[5] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v301 = v300;
              if ( v300 )
                *(_QWORD *)(v300 + 16) = v301 | *(_QWORD *)(v300 + 16) & 3LL;
            }
            v299[3] = v298;
            if ( v298 )
            {
              v302 = v298[1];
              v299[4] = v302;
              if ( v302 )
                *(_QWORD *)(v302 + 16) = (unsigned __int64)(v299 + 4) | *(_QWORD *)(v302 + 16) & 3LL;
              v303 = v299[5];
              v304 = v299 + 3;
              v304[2] = (unsigned __int64)(v298 + 1) | v303 & 3;
              v298[1] = v304;
            }
          }
          v188 = *(_QWORD *)(v188 + 8);
          if ( v189 != v188 )
            continue;
        }
        goto LABEL_182;
      }
      goto LABEL_506;
    }
  }
LABEL_182:
  sub_1C76340((__int64)&v395);
  if ( v402 )
  {
    if ( v401 )
    {
      v269 = v400;
      v270 = &v400[2 * v401];
      do
      {
        if ( *v269 != -4 && *v269 != -8 )
        {
          v271 = v269[1];
          if ( v271 )
            sub_161E7C0((__int64)(v269 + 1), v271);
        }
        v269 += 2;
      }
      while ( v270 != v269 );
    }
    j___libc_free_0(v400);
    v402 = 0;
  }
  v193 = *(_QWORD *)(*(_QWORD *)a30 + 48LL);
  v357 = *(_QWORD *)a30 + 40LL;
  v194 = *(_QWORD *)(a23 + 48);
  if ( v357 != v193 )
  {
    do
    {
      v195 = v194 - 24;
      if ( !v194 )
        v195 = 0;
      v196 = v193 - 24;
      if ( !v193 )
        v196 = 0;
      v197 = sub_1C76B50((__int64)&v395, v196);
      v198 = v197[2];
      if ( v195 != v198 )
      {
        if ( v198 != -8 && v198 != 0 && v198 != -16 )
          sub_1649B30(v197);
        v197[2] = v195;
        if ( v195 != 0 && v195 != -8 && v195 != -16 )
          sub_164C220((__int64)v197);
      }
      v193 = *(_QWORD *)(v193 + 8);
      v194 = *(_QWORD *)(v194 + 8);
    }
    while ( v357 != v193 );
    v199 = *(_QWORD *)(*(_QWORD *)a30 + 48LL);
    v200 = *(_QWORD *)a30 + 40LL;
    if ( v200 != v199 )
    {
      v201 = a25;
      while ( v199 )
      {
        if ( *(_BYTE *)(v199 - 8) == 77 )
        {
          v202 = v199 - 24;
          v203 = (*(_BYTE *)(v199 - 1) & 0x40) != 0
               ? *(_QWORD *)(v199 - 32)
               : v202 - 24LL * (*(_DWORD *)(v199 - 4) & 0xFFFFFFF);
          if ( *v201 == *(_QWORD *)(v203 + 24LL * *(unsigned int *)(v199 + 32) + 8) )
          {
            v292 = sub_1C76B50((__int64)&v395, v199 - 24)[2];
            if ( (*(_BYTE *)(v199 - 1) & 0x40) != 0 )
              v293 = *(_QWORD **)(v199 - 32);
            else
              v293 = (_QWORD *)(v202 - 24LL * (*(_DWORD *)(v199 - 4) & 0xFFFFFFF));
            if ( *v293 )
            {
              v294 = v293[1];
              v295 = v293[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v295 = v294;
              if ( v294 )
                *(_QWORD *)(v294 + 16) = v295 | *(_QWORD *)(v294 + 16) & 3LL;
            }
            *v293 = v292;
            if ( v292 )
            {
              v296 = *(_QWORD *)(v292 + 8);
              v293[1] = v296;
              if ( v296 )
                *(_QWORD *)(v296 + 16) = (unsigned __int64)(v293 + 1) | *(_QWORD *)(v296 + 16) & 3LL;
              v293[2] = (v292 + 8) | v293[2] & 3LL;
              *(_QWORD *)(v292 + 8) = v293;
            }
          }
          else if ( *v201 == *(_QWORD *)(v203 + 24LL * *(unsigned int *)(v199 + 32) + 16) )
          {
            v305 = sub_1C76B50((__int64)&v395, v199 - 24)[2];
            if ( (*(_BYTE *)(v199 - 1) & 0x40) != 0 )
              v306 = *(_QWORD **)(v199 - 32);
            else
              v306 = (_QWORD *)(v202 - 24LL * (*(_DWORD *)(v199 - 4) & 0xFFFFFFF));
            if ( v306[3] )
            {
              v307 = v306[4];
              v308 = v306[5] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v308 = v307;
              if ( v307 )
                *(_QWORD *)(v307 + 16) = v308 | *(_QWORD *)(v307 + 16) & 3LL;
            }
            v306[3] = v305;
            if ( v305 )
            {
              v309 = *(_QWORD *)(v305 + 8);
              v306[4] = v309;
              if ( v309 )
                *(_QWORD *)(v309 + 16) = (unsigned __int64)(v306 + 4) | *(_QWORD *)(v309 + 16) & 3LL;
              v310 = v306[5];
              v311 = v306 + 3;
              v311[2] = (v305 + 8) | v310 & 3;
              *(_QWORD *)(v305 + 8) = v311;
            }
          }
          v199 = *(_QWORD *)(v199 + 8);
          if ( v200 != v199 )
            continue;
        }
        goto LABEL_205;
      }
      goto LABEL_506;
    }
  }
LABEL_205:
  sub_1C76340((__int64)&v395);
  if ( v402 )
  {
    if ( v401 )
    {
      v266 = v400;
      v267 = &v400[2 * v401];
      do
      {
        if ( *v266 != -4 && *v266 != -8 )
        {
          v268 = v266[1];
          if ( v268 )
            sub_161E7C0((__int64)(v266 + 1), v268);
        }
        v266 += 2;
      }
      while ( v267 != v266 );
    }
    j___libc_free_0(v400);
    v402 = 0;
  }
  v358 = *a36 + 40;
  if ( v358 != *(_QWORD *)(*a36 + 48) )
  {
    v204 = *(_QWORD *)(*(_QWORD *)a30 + 48LL);
    v205 = *(_QWORD *)(*a36 + 48);
    do
    {
      v206 = v204 - 24;
      if ( !v204 )
        v206 = 0;
      v207 = v205 - 24;
      if ( !v205 )
        v207 = 0;
      v208 = sub_1C76B50((__int64)&v395, v207);
      v209 = v208[2];
      if ( v206 != v209 )
      {
        if ( v209 != -8 && v209 != 0 && v209 != -16 )
          sub_1649B30(v208);
        v208[2] = v206;
        if ( v206 != 0 && v206 != -8 && v206 != -16 )
          sub_164C220((__int64)v208);
      }
      v205 = *(_QWORD *)(v205 + 8);
      v204 = *(_QWORD *)(v204 + 8);
    }
    while ( v358 != v205 );
    v210 = *(_QWORD *)(*a36 + 48);
    v211 = *a36 + 40;
    if ( v211 != v210 )
    {
      v212 = a31;
      while ( v210 )
      {
        if ( *(_BYTE *)(v210 - 8) == 77 )
        {
          v213 = v210 - 24;
          v214 = (*(_BYTE *)(v210 - 1) & 0x40) != 0
               ? *(_QWORD *)(v210 - 32)
               : v213 - 24LL * (*(_DWORD *)(v210 - 4) & 0xFFFFFFF);
          if ( *v212 == *(_QWORD *)(v214 + 24LL * *(unsigned int *)(v210 + 32) + 8) )
          {
            v275 = sub_1C76B50((__int64)&v395, v210 - 24)[2];
            if ( (*(_BYTE *)(v210 - 1) & 0x40) != 0 )
              v276 = *(_QWORD **)(v210 - 32);
            else
              v276 = (_QWORD *)(v213 - 24LL * (*(_DWORD *)(v210 - 4) & 0xFFFFFFF));
            if ( *v276 )
            {
              v277 = v276[1];
              v278 = v276[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v278 = v277;
              if ( v277 )
                *(_QWORD *)(v277 + 16) = v278 | *(_QWORD *)(v277 + 16) & 3LL;
            }
            *v276 = v275;
            if ( v275 )
            {
              v279 = *(_QWORD *)(v275 + 8);
              v276[1] = v279;
              if ( v279 )
                *(_QWORD *)(v279 + 16) = (unsigned __int64)(v276 + 1) | *(_QWORD *)(v279 + 16) & 3LL;
              v276[2] = (v275 + 8) | v276[2] & 3LL;
              *(_QWORD *)(v275 + 8) = v276;
            }
          }
          else if ( *v212 == *(_QWORD *)(v214 + 24LL * *(unsigned int *)(v210 + 32) + 16) )
          {
            v320 = sub_1C76B50((__int64)&v395, v210 - 24)[2];
            if ( (*(_BYTE *)(v210 - 1) & 0x40) != 0 )
              v321 = *(_QWORD **)(v210 - 32);
            else
              v321 = (_QWORD *)(v213 - 24LL * (*(_DWORD *)(v210 - 4) & 0xFFFFFFF));
            if ( v321[3] )
            {
              v322 = v321[4];
              v323 = v321[5] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v323 = v322;
              if ( v322 )
                *(_QWORD *)(v322 + 16) = v323 | *(_QWORD *)(v322 + 16) & 3LL;
            }
            v321[3] = v320;
            if ( v320 )
            {
              v324 = *(_QWORD *)(v320 + 8);
              v321[4] = v324;
              if ( v324 )
                *(_QWORD *)(v324 + 16) = (unsigned __int64)(v321 + 4) | *(_QWORD *)(v324 + 16) & 3LL;
              v325 = v321[5];
              v326 = v321 + 3;
              v326[2] = (v320 + 8) | v325 & 3;
              *(_QWORD *)(v320 + 8) = v326;
            }
          }
          v210 = *(_QWORD *)(v210 + 8);
          if ( v211 != v210 )
            continue;
        }
        goto LABEL_229;
      }
      goto LABEL_506;
    }
  }
LABEL_229:
  sub_1C76340((__int64)&v395);
  if ( v402 )
  {
    if ( v401 )
    {
      v263 = v400;
      v264 = &v400[2 * v401];
      do
      {
        if ( *v263 != -8 && *v263 != -4 )
        {
          v265 = v263[1];
          if ( v265 )
            sub_161E7C0((__int64)(v263 + 1), v265);
        }
        v263 += 2;
      }
      while ( v264 != v263 );
    }
    j___libc_free_0(v400);
    v402 = 0;
  }
  v215 = *(_QWORD *)(*a36 + 48);
  v359 = *a36 + 40;
  v216 = *(_QWORD *)(a23 + 48);
  if ( v215 != v359 )
  {
    do
    {
      v217 = v216 - 24;
      if ( !v216 )
        v217 = 0;
      v218 = v215 - 24;
      if ( !v215 )
        v218 = 0;
      v219 = sub_1C76B50((__int64)&v395, v218);
      v220 = v219[2];
      if ( v217 != v220 )
      {
        if ( v220 != -8 && v220 != 0 && v220 != -16 )
          sub_1649B30(v219);
        v219[2] = v217;
        if ( v217 != -8 && v217 != 0 && v217 != -16 )
          sub_164C220((__int64)v219);
      }
      v215 = *(_QWORD *)(v215 + 8);
      v216 = *(_QWORD *)(v216 + 8);
    }
    while ( v215 != v359 );
    v221 = *(_QWORD *)(*a36 + 48);
    v222 = *a36 + 40;
    if ( v222 != v221 )
    {
      while ( v221 )
      {
        if ( *(_BYTE *)(v221 - 8) == 77 )
        {
          v223 = sub_1C76B50((__int64)&v395, v221 - 24)[2];
          p_src = &src;
          v375 = (__int64)&v363;
          v376 = &v366;
          sub_164C7D0(
            v223,
            v221 - 24,
            (unsigned __int8 (__fastcall *)(__int64, __int64 *))sub_1C745E0,
            (__int64)&p_src,
            a7,
            a8,
            a9,
            a10,
            v224,
            v225,
            a13,
            a14);
          v221 = *(_QWORD *)(v221 + 8);
          if ( v222 != v221 )
            continue;
        }
        goto LABEL_247;
      }
      goto LABEL_506;
    }
LABEL_247:
    if ( v402 )
    {
      if ( v401 )
      {
        v260 = v400;
        v261 = &v400[2 * v401];
        do
        {
          if ( *v260 != -4 && *v260 != -8 )
          {
            v262 = v260[1];
            if ( v262 )
              sub_161E7C0((__int64)(v260 + 1), v262);
          }
          v260 += 2;
        }
        while ( v261 != v260 );
      }
      j___libc_free_0(v400);
    }
  }
  if ( (_DWORD)v398 )
  {
    v237 = (_QWORD *)v396;
    v370 = 2;
    v371 = 0;
    v238 = (_QWORD *)(v396 + ((unsigned __int64)(unsigned int)v398 << 6));
    v372 = -8;
    v239 = -8;
    v369 = &unk_49E6B50;
    v373 = 0;
    v375 = 2;
    v376 = 0;
    v377 = -16;
    p_src = (void **)&unk_49E6B50;
    m = 0;
    while ( 1 )
    {
      v240 = v237[3];
      if ( v240 != v239 )
      {
        v239 = v377;
        if ( v240 != v377 )
        {
          v241 = v237[7];
          if ( v241 != 0 && v241 != -8 && v241 != -16 )
          {
            sub_1649B30(v237 + 5);
            v240 = v237[3];
          }
          v239 = v240;
        }
      }
      *v237 = &unk_49EE2B0;
      if ( v239 != 0 && v239 != -8 && v239 != -16 )
        sub_1649B30(v237 + 1);
      v237 += 8;
      if ( v238 == v237 )
        break;
      v239 = v372;
    }
    p_src = (void **)&unk_49EE2B0;
    if ( v377 != 0 && v377 != -8 && v377 != -16 )
      sub_1649B30(&v375);
    v369 = &unk_49EE2B0;
    if ( v372 != -8 && v372 != 0 && v372 != -16 )
      sub_1649B30(&v370);
  }
  j___libc_free_0(v396);
  if ( v393 )
  {
    if ( v392 )
    {
      v257 = v391;
      v258 = &v391[2 * v392];
      do
      {
        if ( *v257 != -8 && *v257 != -4 )
        {
          v259 = v257[1];
          if ( v259 )
            sub_161E7C0((__int64)(v257 + 1), v259);
        }
        v257 += 2;
      }
      while ( v258 != v257 );
    }
    j___libc_free_0(v391);
  }
  if ( v390 )
  {
    v232 = v388;
    v370 = 2;
    v371 = 0;
    v233 = &v388[8 * (unsigned __int64)v390];
    v372 = -8;
    v234 = -8;
    v369 = &unk_49E6B50;
    v373 = 0;
    v375 = 2;
    v376 = 0;
    v377 = -16;
    p_src = (void **)&unk_49E6B50;
    m = 0;
    while ( 1 )
    {
      v235 = v232[3];
      if ( v235 != v234 )
      {
        v234 = v377;
        if ( v235 != v377 )
        {
          v236 = v232[7];
          if ( v236 != 0 && v236 != -8 && v236 != -16 )
          {
            sub_1649B30(v232 + 5);
            v235 = v232[3];
          }
          v234 = v235;
        }
      }
      *v232 = &unk_49EE2B0;
      if ( v234 != -8 && v234 != 0 && v234 != -16 )
        sub_1649B30(v232 + 1);
      v232 += 8;
      if ( v233 == v232 )
        break;
      v234 = v372;
    }
    p_src = (void **)&unk_49EE2B0;
    if ( v377 != 0 && v377 != -8 && v377 != -16 )
      sub_1649B30(&v375);
    v369 = &unk_49EE2B0;
    if ( v372 != 0 && v372 != -8 && v372 != -16 )
      sub_1649B30(&v370);
  }
  j___libc_free_0(v388);
  if ( v385 )
  {
    if ( v384 )
    {
      v254 = v383;
      v255 = &v383[2 * v384];
      do
      {
        if ( *v254 != -4 && *v254 != -8 )
        {
          v256 = v254[1];
          if ( v256 )
            sub_161E7C0((__int64)(v254 + 1), v256);
        }
        v254 += 2;
      }
      while ( v255 != v254 );
    }
    j___libc_free_0(v383);
  }
  if ( v382 )
  {
    v227 = v380;
    v370 = 2;
    v371 = 0;
    v228 = &v380[8 * (unsigned __int64)v382];
    v372 = -8;
    v229 = -8;
    v369 = &unk_49E6B50;
    v373 = 0;
    v375 = 2;
    v376 = 0;
    v377 = -16;
    p_src = (void **)&unk_49E6B50;
    m = 0;
    while ( 1 )
    {
      v230 = v227[3];
      if ( v230 != v229 )
      {
        v229 = v377;
        if ( v230 != v377 )
        {
          v231 = v227[7];
          if ( v231 != -8 && v231 != 0 && v231 != -16 )
          {
            sub_1649B30(v227 + 5);
            v230 = v227[3];
          }
          v229 = v230;
        }
      }
      *v227 = &unk_49EE2B0;
      if ( v229 != 0 && v229 != -8 && v229 != -16 )
        sub_1649B30(v227 + 1);
      v227 += 8;
      if ( v228 == v227 )
        break;
      v229 = v372;
    }
    p_src = (void **)&unk_49EE2B0;
    if ( v377 != 0 && v377 != -8 && v377 != -16 )
      sub_1649B30(&v375);
    v369 = &unk_49EE2B0;
    if ( v372 != -8 && v372 != 0 && v372 != -16 )
      sub_1649B30(&v370);
  }
  result = j___libc_free_0(v380);
  if ( v366 )
    result = j_j___libc_free_0(v366, v368 - v366);
  if ( v363 )
    result = j_j___libc_free_0(v363, v365 - v363);
  if ( src )
    return j_j___libc_free_0(src, (char *)v362 - (_BYTE *)src);
  return result;
}
