// Function: sub_9624D0
// Address: 0x9624d0
//
__int64 __fastcall sub_9624D0(
        __int64 a1,
        int a2,
        const char **a3,
        int a4,
        int *a5,
        __int64 *a6,
        _DWORD *a7,
        __int64 *a8,
        int *a9,
        __int64 *a10,
        int *a11,
        __int64 *a12,
        int *a13,
        __int64 *a14)
{
  __int64 v14; // r8
  const char **v15; // rax
  unsigned int v16; // r13d
  const char *v18; // r12
  const char *v19; // rdi
  size_t v20; // rax
  size_t v21; // r13
  _QWORD *v22; // rdx
  const char **v23; // r12
  const char *v24; // r14
  size_t v25; // r13
  __int64 v26; // r15
  const char *v27; // rbx
  bool v28; // al
  bool v29; // cf
  bool v30; // zf
  __int64 v31; // rcx
  __int64 v32; // rsi
  const char *v33; // rdi
  const char *v34; // r15
  size_t v35; // rax
  size_t v36; // r13
  _QWORD *v37; // rdx
  _QWORD *v38; // r8
  const char *v39; // rbx
  size_t v40; // rax
  int v41; // edi
  __int64 v42; // rax
  int v43; // edx
  const char *v44; // r13
  size_t v45; // rax
  size_t v46; // r8
  _QWORD *v47; // rdx
  unsigned __int64 v48; // rax
  __int64 v49; // rbx
  bool v50; // al
  bool v51; // cf
  bool v52; // zf
  __int64 v53; // rcx
  const char *v54; // r14
  size_t v55; // rax
  size_t v56; // r9
  _QWORD *v57; // rdx
  __m128i *p_s1; // rsi
  __int64 v59; // rdx
  _QWORD *v60; // rdi
  __int64 v61; // rcx
  char *v62; // rdi
  __int64 v63; // rax
  _QWORD *v64; // rdi
  __int64 v65; // rax
  _QWORD *v66; // rdi
  __int64 v67; // r13
  _QWORD *v68; // rbx
  _QWORD *v69; // r12
  __m128i *v70; // rbx
  __m128i *v71; // r12
  const __m128i *v72; // rbx
  __m128i *v73; // r12
  const __m128i *v74; // rbx
  __m128i *v75; // r12
  const __m128i *v76; // rbx
  __m128i *v77; // r12
  __m128i *v78; // r14
  const char **v79; // rbx
  const char *v80; // rdi
  size_t v81; // rax
  __int64 v82; // r12
  unsigned __int64 v83; // rax
  __int64 v84; // rcx
  unsigned __int64 v85; // rax
  unsigned __int64 v86; // rax
  char v87; // si
  __m128i *v88; // r15
  int v89; // eax
  size_t v90; // rbx
  size_t v91; // rax
  size_t v92; // r13
  const char *v93; // r12
  size_t i; // r14
  size_t v95; // r13
  __int64 v96; // rdi
  size_t v97; // rax
  size_t v98; // r13
  __int64 v99; // rdx
  _QWORD *v100; // rbx
  bool v101; // cf
  bool v102; // zf
  __int64 v103; // rax
  size_t v104; // r8
  unsigned __int64 v105; // r12
  __int64 v106; // r15
  unsigned __int64 v107; // rdx
  unsigned __int64 v108; // rdx
  __int64 v109; // rax
  size_t v110; // r8
  unsigned __int64 v111; // r12
  __int64 v112; // r15
  unsigned __int64 v113; // rdx
  unsigned __int64 v114; // rdx
  __int64 v115; // rax
  size_t v116; // r8
  unsigned __int64 v117; // r12
  __int64 v118; // r15
  unsigned __int64 v119; // rdx
  unsigned __int64 v120; // rdx
  unsigned __int64 v121; // r8
  __int64 v122; // rdi
  int v123; // eax
  __int64 v124; // rax
  __m128i v125; // xmm0
  __int64 v126; // rax
  __m128i v127; // xmm0
  __int64 v128; // rax
  _QWORD *v129; // r12
  int v130; // eax
  __int64 v131; // rax
  __m128i v132; // xmm0
  __int64 v133; // rax
  __m128i v134; // xmm0
  const char *v135; // rsi
  _QWORD *v136; // rdx
  signed __int64 v137; // rax
  __int64 v138; // rdi
  __int64 v139; // rax
  __int64 v140; // rbx
  __int64 v141; // r13
  __int64 v142; // rax
  signed __int64 v143; // rax
  __int64 v144; // rdi
  __int64 v145; // rax
  __int64 v146; // rbx
  __int64 v147; // r13
  __int64 v148; // rax
  signed __int64 v149; // rax
  __int64 v150; // rdi
  __int64 v151; // rax
  __int64 v152; // rbx
  __int64 v153; // r13
  __int64 v154; // rax
  __int64 v155; // rax
  __int64 v156; // rdi
  __int64 v157; // rax
  __int64 v158; // rbx
  __int64 v159; // r13
  __int64 v160; // rax
  __int64 v161; // rcx
  const char *v162; // rdi
  size_t v163; // rax
  __m128i *v164; // rdi
  const char *v165; // r13
  const char *v166; // r8
  const char *v167; // r13
  size_t v168; // rax
  size_t v169; // r8
  char *v170; // rdx
  const __m128i *v171; // rax
  __m128i *v172; // rdi
  const char *v173; // r9
  size_t v174; // rax
  const char *v175; // r9
  size_t v176; // r8
  __m128i *v177; // rdx
  const char *v178; // r9
  size_t v179; // rax
  const char *v180; // r9
  size_t v181; // r8
  char *v182; // rdx
  __int64 v183; // rax
  __m128i si128; // xmm0
  __m128i v185; // xmm0
  __int64 v186; // rbx
  char *v187; // rax
  char *v188; // rdi
  size_t v189; // rax
  const void *v190; // r8
  size_t v191; // r9
  __int64 v192; // rdx
  const char *v193; // r9
  size_t v194; // rax
  const char *v195; // r9
  size_t v196; // r8
  _QWORD *v197; // rdx
  const __m128i *v198; // rax
  __m128i *v199; // rdi
  __int64 v200; // rax
  __m128i *v201; // rdi
  __m128i *v202; // rax
  __int64 v203; // rcx
  __m128i *v204; // rax
  __int64 v205; // rbx
  __int64 v206; // rax
  const __m128i *v207; // rax
  __m128i *v208; // rdi
  __int64 *v209; // rbx
  __int64 v210; // r12
  unsigned __int64 v211; // rcx
  char v212; // bl
  __int64 *v213; // rbx
  __int64 v214; // r12
  unsigned __int64 v215; // rcx
  __int64 *v216; // rbx
  __int64 v217; // r12
  unsigned __int64 v218; // rcx
  __int64 *v219; // rbx
  __int64 v220; // r12
  unsigned __int64 v221; // rcx
  __int64 v222; // rax
  _QWORD *v223; // rdi
  _QWORD *v224; // rdi
  char *v225; // rax
  char *v226; // rdi
  size_t v227; // rax
  const void *v228; // r8
  size_t v229; // r9
  __int64 v230; // rdx
  size_t v231; // rdx
  __m128i *v232; // rax
  __int64 v233; // rcx
  __m128i *v234; // rax
  __int64 v235; // rbx
  __int64 v236; // rax
  void *v237; // rdi
  __int64 v238; // rax
  __m128i v239; // xmm0
  __int64 v240; // rax
  __m128i v241; // xmm0
  __int64 v242; // rax
  __m128i v243; // xmm0
  __int64 v244; // rax
  __m128i v245; // xmm0
  size_t v246; // rax
  const void *v247; // r8
  size_t v248; // r9
  __int64 v249; // rdx
  __int64 *v250; // rbx
  __int64 v251; // r12
  unsigned __int64 v252; // rcx
  __int64 *v253; // rbx
  __int64 v254; // r12
  unsigned __int64 v255; // rcx
  __int64 *v256; // rbx
  __int64 v257; // r12
  unsigned __int64 v258; // rcx
  const char *v259; // rsi
  __int64 v260; // rax
  __m128i v261; // xmm0
  __int64 v262; // rax
  __m128i v263; // xmm0
  __int64 v264; // rax
  void *v265; // rdi
  __int64 v266; // rax
  void *v267; // rdi
  __m128i **v268; // rdi
  __m128i *v269; // rsi
  const char *v270; // rsi
  __int64 v271; // rbx
  const char *v272; // rsi
  bool v273; // bl
  __int64 v274; // rbx
  bool v275; // [rsp+7h] [rbp-579h]
  bool v276; // [rsp+8h] [rbp-578h]
  char v277; // [rsp+9h] [rbp-577h]
  char v278; // [rsp+Ah] [rbp-576h]
  char v279; // [rsp+Bh] [rbp-575h]
  char v280; // [rsp+Ch] [rbp-574h]
  char v281; // [rsp+Dh] [rbp-573h]
  bool v282; // [rsp+Eh] [rbp-572h]
  char v283; // [rsp+Fh] [rbp-571h]
  _QWORD *v286; // [rsp+28h] [rbp-558h]
  __int64 v287; // [rsp+38h] [rbp-548h]
  size_t v288; // [rsp+40h] [rbp-540h]
  size_t v289; // [rsp+40h] [rbp-540h]
  char *v290; // [rsp+40h] [rbp-540h]
  size_t v291; // [rsp+40h] [rbp-540h]
  size_t v292; // [rsp+40h] [rbp-540h]
  size_t v293; // [rsp+40h] [rbp-540h]
  char *v294; // [rsp+40h] [rbp-540h]
  size_t v295; // [rsp+40h] [rbp-540h]
  size_t v296; // [rsp+40h] [rbp-540h]
  size_t v297; // [rsp+40h] [rbp-540h]
  size_t v298; // [rsp+40h] [rbp-540h]
  const char **v301; // [rsp+50h] [rbp-530h]
  int v302; // [rsp+58h] [rbp-528h]
  size_t v303; // [rsp+60h] [rbp-520h]
  const char *src; // [rsp+68h] [rbp-518h]
  __m128i *srca; // [rsp+68h] [rbp-518h]
  char *srcb; // [rsp+68h] [rbp-518h]
  size_t n; // [rsp+70h] [rbp-510h]
  size_t v309; // [rsp+88h] [rbp-4F8h]
  size_t v310; // [rsp+88h] [rbp-4F8h]
  size_t v311; // [rsp+88h] [rbp-4F8h]
  size_t v312; // [rsp+88h] [rbp-4F8h]
  size_t v313; // [rsp+88h] [rbp-4F8h]
  size_t v314; // [rsp+88h] [rbp-4F8h]
  const char *v315; // [rsp+88h] [rbp-4F8h]
  const char *v316; // [rsp+88h] [rbp-4F8h]
  size_t v317; // [rsp+88h] [rbp-4F8h]
  _BYTE *v318; // [rsp+88h] [rbp-4F8h]
  const char *v319; // [rsp+88h] [rbp-4F8h]
  _BYTE *v320; // [rsp+88h] [rbp-4F8h]
  size_t v321; // [rsp+88h] [rbp-4F8h]
  const void *v322; // [rsp+90h] [rbp-4F0h] BYREF
  size_t v323; // [rsp+98h] [rbp-4E8h]
  __m128i *v324; // [rsp+A0h] [rbp-4E0h] BYREF
  __m128i *v325; // [rsp+A8h] [rbp-4D8h]
  const __m128i *v326; // [rsp+B0h] [rbp-4D0h]
  __m128i *v327; // [rsp+C0h] [rbp-4C0h] BYREF
  __m128i *v328; // [rsp+C8h] [rbp-4B8h]
  const __m128i *v329; // [rsp+D0h] [rbp-4B0h]
  __m128i *v330; // [rsp+E0h] [rbp-4A0h] BYREF
  __m128i *v331; // [rsp+E8h] [rbp-498h]
  const __m128i *v332; // [rsp+F0h] [rbp-490h]
  __m128i *v333; // [rsp+100h] [rbp-480h] BYREF
  __m128i *v334; // [rsp+108h] [rbp-478h]
  __int64 v335; // [rsp+110h] [rbp-470h]
  char *v336; // [rsp+120h] [rbp-460h] BYREF
  size_t v337; // [rsp+128h] [rbp-458h]
  _QWORD v338[2]; // [rsp+130h] [rbp-450h] BYREF
  char *v339; // [rsp+140h] [rbp-440h] BYREF
  size_t v340; // [rsp+148h] [rbp-438h]
  _QWORD v341[2]; // [rsp+150h] [rbp-430h] BYREF
  __m128i v342; // [rsp+160h] [rbp-420h] BYREF
  _QWORD v343[2]; // [rsp+170h] [rbp-410h] BYREF
  __m128i v344; // [rsp+180h] [rbp-400h] BYREF
  __m128i v345; // [rsp+190h] [rbp-3F0h] BYREF
  __m128i s1; // [rsp+1A0h] [rbp-3E0h] BYREF
  _QWORD v347[2]; // [rsp+1B0h] [rbp-3D0h] BYREF
  __m128i v348; // [rsp+1C0h] [rbp-3C0h] BYREF
  __m128i v349; // [rsp+1D0h] [rbp-3B0h] BYREF
  __int64 v350; // [rsp+1E0h] [rbp-3A0h]
  __int64 v351; // [rsp+1E8h] [rbp-398h]
  __int64 v352; // [rsp+1F0h] [rbp-390h]
  __int64 v353; // [rsp+1F8h] [rbp-388h]
  __int64 v354; // [rsp+200h] [rbp-380h]
  char v355[8]; // [rsp+208h] [rbp-378h] BYREF
  int v356; // [rsp+210h] [rbp-370h]
  _QWORD *v357; // [rsp+218h] [rbp-368h] BYREF
  _QWORD v358[2]; // [rsp+228h] [rbp-358h] BYREF
  _QWORD v359[28]; // [rsp+238h] [rbp-348h] BYREF
  __int16 v360; // [rsp+318h] [rbp-268h]
  __int64 v361; // [rsp+320h] [rbp-260h]
  __int64 v362; // [rsp+328h] [rbp-258h]
  __int64 v363; // [rsp+330h] [rbp-250h]
  __int64 v364; // [rsp+338h] [rbp-248h]
  _OWORD *v365; // [rsp+340h] [rbp-240h] BYREF
  __int64 v366; // [rsp+348h] [rbp-238h]
  _OWORD v367[35]; // [rsp+350h] [rbp-230h] BYREF

  v301 = a3;
  if ( a2 > 0 )
  {
    v14 = (__int64)&a3[(unsigned int)(a2 - 1) + 1];
    v15 = a3;
    do
    {
      if ( !*v15 )
        return 1;
      ++v15;
    }
    while ( (const char **)v14 != v15 );
    while ( 1 )
    {
      v18 = *a3;
      if ( !strcmp(*a3, "-time-passes") )
        break;
      if ( (const char **)v14 == ++a3 )
        goto LABEL_14;
    }
    if ( !a14 )
      return 1;
    v19 = *a3;
    s1.m128i_i64[0] = (__int64)v347;
    v20 = strlen(v19);
    v365 = (_OWORD *)v20;
    v21 = v20;
    if ( v20 > 0xF )
    {
      s1.m128i_i64[0] = sub_22409D0(&s1, &v365, 0);
      v224 = (_QWORD *)s1.m128i_i64[0];
      v347[0] = v365;
    }
    else
    {
      if ( v20 == 1 )
      {
        LOBYTE(v347[0]) = *v18;
        v22 = v347;
LABEL_357:
        s1.m128i_i64[1] = v20;
        *((_BYTE *)v22 + v20) = 0;
        v202 = (__m128i *)sub_2241130(&s1, 0, 0, "libnvvm : error: ", 17);
        v348.m128i_i64[0] = (__int64)&v349;
        if ( (__m128i *)v202->m128i_i64[0] == &v202[1] )
        {
          v349 = _mm_loadu_si128(v202 + 1);
        }
        else
        {
          v348.m128i_i64[0] = v202->m128i_i64[0];
          v349.m128i_i64[0] = v202[1].m128i_i64[0];
        }
        v203 = v202->m128i_i64[1];
        v348.m128i_i64[1] = v203;
        v202->m128i_i64[0] = (__int64)v202[1].m128i_i64;
        v202->m128i_i64[1] = 0;
        v202[1].m128i_i8[0] = 0;
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v348.m128i_i64[1]) <= 0x18 )
          goto LABEL_573;
        v204 = (__m128i *)sub_2241490(&v348, " is an unsupported option", 25, v203);
        v365 = v367;
        if ( (__m128i *)v204->m128i_i64[0] == &v204[1] )
        {
          v367[0] = _mm_loadu_si128(v204 + 1);
        }
        else
        {
          v365 = (_OWORD *)v204->m128i_i64[0];
          *(_QWORD *)&v367[0] = v204[1].m128i_i64[0];
        }
        v366 = v204->m128i_i64[1];
        v204->m128i_i64[0] = (__int64)v204[1].m128i_i64;
        v204->m128i_i64[1] = 0;
        v204[1].m128i_i8[0] = 0;
        if ( (__m128i *)v348.m128i_i64[0] != &v349 )
          j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
        if ( (_QWORD *)s1.m128i_i64[0] != v347 )
          j_j___libc_free_0(s1.m128i_i64[0], v347[0] + 1LL);
        v205 = v366;
        v206 = sub_2207820(v366 + 1);
        *a14 = v206;
        sub_2241570(&v365, v206, v205, 0);
        *(_BYTE *)(*a14 + v205) = 0;
        if ( v365 != v367 )
          j_j___libc_free_0(v365, *(_QWORD *)&v367[0] + 1LL);
        return 1;
      }
      if ( !v20 )
      {
        v22 = v347;
        goto LABEL_357;
      }
      v224 = v347;
    }
    memcpy(v224, v18, v21);
    v20 = (size_t)v365;
    v22 = (_QWORD *)s1.m128i_i64[0];
    goto LABEL_357;
  }
LABEL_14:
  v324 = 0;
  v365 = v367;
  v366 = 0x1000000000LL;
  v325 = 0;
  *(_DWORD *)(a1 + 8) = 75;
  *(_BYTE *)(a1 + 232) = 0;
  *(_DWORD *)(a1 + 1640) = 0;
  v326 = 0;
  v327 = 0;
  v328 = 0;
  v329 = 0;
  v330 = 0;
  v331 = 0;
  v332 = 0;
  v333 = 0;
  v334 = 0;
  v335 = 0;
  if ( a2 <= 0 )
  {
    *a11 = 0;
    *a9 = 0;
    *a7 = 0;
    *a5 = 0;
    *a12 = 0;
    *a10 = 0;
    *a8 = 0;
    *a6 = 0;
    if ( a2 )
      goto LABEL_35;
    v278 = 0;
    v87 = 0;
    v78 = &v348;
    *(_DWORD *)(a1 + 240) = 0;
    *a13 = 7;
    goto LABEL_144;
  }
  v23 = v301;
  do
  {
    v24 = *v23;
    v25 = strlen(*v23);
    if ( v25 )
    {
      v26 = 0;
      while ( 1 )
      {
        v27 = &v24[v26];
        if ( !isspace(v24[v26]) )
          break;
        if ( v25 == ++v26 )
        {
          v27 = &v24[v26];
          break;
        }
      }
    }
    else
    {
      v27 = v24;
      v26 = 0;
    }
    v28 = memcmp(v27, "-arch=compute_", 0xEu) != 0;
    v29 = 0;
    v30 = !v28;
    if ( !v28 )
    {
      v44 = &v24[v26 + 14];
      s1.m128i_i64[0] = (__int64)v347;
      v45 = strlen(v44);
      v348.m128i_i64[0] = v45;
      v46 = v45;
      if ( v45 > 0xF )
      {
        n = v45;
        v63 = sub_22409D0(&s1, &v348, 0);
        v46 = n;
        s1.m128i_i64[0] = v63;
        v64 = (_QWORD *)v63;
        v347[0] = v348.m128i_i64[0];
      }
      else
      {
        if ( v45 == 1 )
        {
          LOBYTE(v347[0]) = *v44;
          v47 = v347;
          goto LABEL_48;
        }
        if ( !v45 )
        {
          v47 = v347;
          goto LABEL_48;
        }
        v64 = v347;
      }
      memcpy(v64, v44, v46);
      v45 = v348.m128i_i64[0];
      v47 = (_QWORD *)s1.m128i_i64[0];
LABEL_48:
      s1.m128i_i64[1] = v45;
      *((_BYTE *)v47 + v45) = 0;
      sub_222DF20(v359);
      v360 = 0;
      v359[27] = 0;
      v359[0] = off_4A06798;
      v361 = 0;
      v362 = 0;
      v363 = 0;
      v348.m128i_i64[0] = (__int64)qword_4A07108;
      v364 = 0;
      *(__int64 *)((char *)v348.m128i_i64 + qword_4A07108[-3]) = (__int64)&unk_4A07130;
      v348.m128i_i64[1] = 0;
      sub_222DD70(&v348.m128i_i8[*(_QWORD *)(v348.m128i_i64[0] - 24)], 0);
      v349.m128i_i64[1] = 0;
      v350 = 0;
      v351 = 0;
      v348.m128i_i64[0] = (__int64)off_4A07178;
      v359[0] = off_4A071A0;
      v352 = 0;
      v349.m128i_i64[0] = (__int64)off_4A07480;
      v353 = 0;
      v354 = 0;
      sub_220A990(v355);
      v356 = 0;
      v349.m128i_i64[0] = (__int64)off_4A07080;
      v357 = v358;
      sub_95BA30((__int64 *)&v357, s1.m128i_i64[0], s1.m128i_i64[0] + s1.m128i_i64[1]);
      v356 = 8;
      sub_223FD50(&v349, v357, 0, 0);
      sub_222DD70(v359, &v349);
      if ( (_QWORD *)s1.m128i_i64[0] != v347 )
        j_j___libc_free_0(s1.m128i_i64[0], v347[0] + 1LL);
      v32 = a1 + 8;
      sub_222E4D0(&v348, a1 + 8);
      v48 = (unsigned int)(*(_DWORD *)(a1 + 8) - 75);
      if ( (unsigned int)v48 > 0x2E || (v53 = 0x60081200F821LL, !_bittest64(&v53, v48)) )
      {
        if ( a14 )
        {
          sub_95BEE0(v344.m128i_i64, &(*v23)[v26]);
          sub_95D570(&s1, "libnvvm : error: ", (__int64)&v344);
          sub_94F930(&v342, (__int64)&s1, " is an unsupported option");
          sub_2240A30(&s1);
          sub_2240A30(&v344);
          v49 = v342.m128i_i64[1];
          v32 = sub_2207820(v342.m128i_i64[1] + 1);
          *a14 = v32;
          sub_2241570(&v342, v32, v49, 0);
          *(_BYTE *)(*a14 + v49) = 0;
          sub_2240A30(&v342);
        }
        v16 = 1;
        sub_223F4B0(&v348);
        goto LABEL_92;
      }
      s1.m128i_i64[0] = (__int64)v347;
      v54 = &(*v23)[v26];
      if ( !v54 )
LABEL_58:
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v55 = strlen(&(*v23)[v26]);
      v344.m128i_i64[0] = v55;
      v56 = v55;
      if ( v55 > 0xF )
      {
        v303 = v55;
        v65 = sub_22409D0(&s1, &v344, 0);
        v56 = v303;
        s1.m128i_i64[0] = v65;
        v66 = (_QWORD *)v65;
        v347[0] = v344.m128i_i64[0];
      }
      else
      {
        if ( v55 == 1 )
        {
          LOBYTE(v347[0]) = *v54;
          v57 = v347;
LABEL_62:
          s1.m128i_i64[1] = v55;
          p_s1 = &s1;
          *((_BYTE *)v57 + v55) = 0;
          sub_95D700(&v330, &s1);
          if ( (_QWORD *)s1.m128i_i64[0] != v347 )
          {
            p_s1 = (__m128i *)(v347[0] + 1LL);
            j_j___libc_free_0(s1.m128i_i64[0], v347[0] + 1LL);
          }
          v348.m128i_i64[0] = (__int64)off_4A07178;
          v359[0] = off_4A071A0;
          v349.m128i_i64[0] = (__int64)off_4A07080;
          if ( v357 != v358 )
          {
            p_s1 = (__m128i *)(v358[0] + 1LL);
            j_j___libc_free_0(v357, v358[0] + 1LL);
          }
          v349.m128i_i64[0] = (__int64)off_4A07480;
          sub_2209150(v355, p_s1, v59);
          v348.m128i_i64[0] = (__int64)qword_4A07108;
          *(__int64 *)((char *)v348.m128i_i64 + qword_4A07108[-3]) = (__int64)&unk_4A07130;
          v348.m128i_i64[1] = 0;
          v359[0] = off_4A06798;
          sub_222E050(v359);
          v24 = *v23;
          v27 = &(*v23)[v26];
          goto LABEL_21;
        }
        if ( !v55 )
        {
          v57 = v347;
          goto LABEL_62;
        }
        v66 = v347;
      }
      memcpy(v66, v54, v56);
      v55 = v344.m128i_i64[0];
      v57 = (_QWORD *)s1.m128i_i64[0];
      goto LABEL_62;
    }
LABEL_21:
    v31 = 15;
    v32 = (__int64)v27;
    v33 = "-Ofast-compile=";
    do
    {
      if ( !v31 )
        break;
      v29 = *(_BYTE *)v32 < *v33;
      v30 = *(_BYTE *)v32++ == *v33++;
      --v31;
    }
    while ( v30 );
    if ( (!v29 && !v30) != v29 )
      goto LABEL_33;
    if ( *(_DWORD *)(a1 + 1640) )
    {
      if ( a14 )
      {
        s1.m128i_i64[0] = 54;
        v348.m128i_i64[0] = (__int64)&v349;
        v183 = sub_22409D0(&v348, &s1, 0);
        v348.m128i_i64[0] = v183;
        v349.m128i_i64[0] = s1.m128i_i64[0];
        *(__m128i *)v183 = _mm_load_si128((const __m128i *)&xmmword_3F15800);
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F15810);
        *(_WORD *)(v183 + 52) = 25955;
        *(__m128i *)(v183 + 16) = si128;
        v185 = _mm_load_si128((const __m128i *)&xmmword_3F15820);
        *(_DWORD *)(v183 + 48) = 1852776558;
        *(__m128i *)(v183 + 32) = v185;
        v348.m128i_i64[1] = s1.m128i_i64[0];
        *(_BYTE *)(v348.m128i_i64[0] + s1.m128i_i64[0]) = 0;
        v186 = v348.m128i_i64[1];
        v32 = sub_2207820(v348.m128i_i64[1] + 1);
        *a14 = v32;
        sub_2241570(&v348, v32, v186, 0);
        *(_BYTE *)(*a14 + v186) = 0;
        if ( (__m128i *)v348.m128i_i64[0] != &v349 )
        {
          v32 = v349.m128i_i64[0] + 1;
          j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
        }
      }
      goto LABEL_91;
    }
    v34 = &v24[v26 + 15];
    s1.m128i_i64[0] = (__int64)v347;
    v35 = strlen(v34);
    v348.m128i_i64[0] = v35;
    v36 = v35;
    if ( v35 > 0xF )
    {
      s1.m128i_i64[0] = sub_22409D0(&s1, &v348, 0);
      v60 = (_QWORD *)s1.m128i_i64[0];
      v347[0] = v348.m128i_i64[0];
LABEL_70:
      memcpy(v60, v34, v36);
      v35 = v348.m128i_i64[0];
      v37 = (_QWORD *)s1.m128i_i64[0];
      goto LABEL_29;
    }
    if ( v35 == 1 )
    {
      LOBYTE(v347[0]) = *v34;
      v37 = v347;
    }
    else
    {
      if ( v35 )
      {
        v60 = v347;
        goto LABEL_70;
      }
      v37 = v347;
    }
LABEL_29:
    s1.m128i_i64[1] = v35;
    *((_BYTE *)v37 + v35) = 0;
    v38 = (_QWORD *)s1.m128i_i64[0];
    if ( !strcmp((const char *)s1.m128i_i64[0], "max") )
    {
      *(_DWORD *)(a1 + 1640) = 2;
    }
    else
    {
      v50 = strcmp((const char *)s1.m128i_i64[0], "mid") != 0;
      v51 = 0;
      v52 = !v50;
      if ( v50 )
      {
        v61 = 4;
        v62 = "min";
        v32 = s1.m128i_i64[0];
        do
        {
          if ( !v61 )
            break;
          v51 = *(_BYTE *)v32 < (unsigned __int8)*v62;
          v52 = *(_BYTE *)v32++ == (unsigned __int8)*v62++;
          --v61;
        }
        while ( v52 );
        if ( (!v51 && !v52) == v51 )
        {
          *(_DWORD *)(a1 + 1640) = 4;
        }
        else
        {
          if ( *(_BYTE *)s1.m128i_i64[0] != 48 || *(_BYTE *)(s1.m128i_i64[0] + 1) )
          {
            if ( a14 )
            {
              sub_95BEE0(
                v348.m128i_i64,
                "libnvvm : error: -Ofast-compile called with unsupported level, only supports 0, min, mid, or max");
              v67 = v348.m128i_i64[1];
              v32 = sub_2207820(v348.m128i_i64[1] + 1);
              *a14 = v32;
              sub_2241570(&v348, v32, v67, 0);
              *(_BYTE *)(*a14 + v67) = 0;
              if ( (__m128i *)v348.m128i_i64[0] != &v349 )
              {
                v32 = v349.m128i_i64[0] + 1;
                j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
              }
              v38 = (_QWORD *)s1.m128i_i64[0];
            }
            if ( v38 != v347 )
            {
              v32 = v347[0] + 1LL;
              j_j___libc_free_0(v38, v347[0] + 1LL);
            }
            goto LABEL_91;
          }
          *(_DWORD *)(a1 + 1640) = 1;
        }
      }
      else
      {
        *(_DWORD *)(a1 + 1640) = 3;
      }
    }
    if ( v38 != v347 )
      j_j___libc_free_0(v38, v347[0] + 1LL);
LABEL_33:
    ++v23;
  }
  while ( &v301[(unsigned int)(a2 - 1) + 1] != v23 );
  *a11 = 0;
  *a9 = 0;
  *a7 = 0;
  *a5 = 0;
  *a12 = 0;
  *a10 = 0;
  *a8 = 0;
  *a6 = 0;
LABEL_35:
  v39 = *v301;
  if ( *v301 )
  {
    v40 = strlen(*v301);
    if ( v40 )
    {
      if ( v40 == 4 )
      {
        switch ( *(_DWORD *)v39 )
        {
          case 0x6B6E6C2D:
            v41 = 1;
            *(_DWORD *)(a1 + 240) = 1;
            goto LABEL_40;
          case 0x74706F2D:
            v41 = 2;
            *(_DWORD *)(a1 + 240) = 2;
            goto LABEL_40;
          case 0x63766E2D:
          case 0x636C6C2D:
            v41 = 3;
            *(_DWORD *)(a1 + 240) = 3;
LABEL_40:
            v348.m128i_i32[0] = 0;
            v32 = (unsigned int)a2;
            v42 = sub_95C880(v41, a2, v301, v348.m128i_i32, a13);
            v43 = *(_DWORD *)(a1 + 240);
            if ( v43 == 2 )
            {
              v16 = 0;
              *a7 = v348.m128i_i32[0];
              *a8 = v42;
            }
            else if ( v43 == 3 )
            {
              v16 = 0;
              *a11 = v348.m128i_i32[0];
              *a12 = v42;
            }
            else
            {
              v16 = 1;
              if ( v43 == 1 )
              {
                v16 = 0;
                *a5 = v348.m128i_i32[0];
                *a6 = v42;
              }
            }
            goto LABEL_92;
        }
      }
      else if ( v40 == 8 && *(_QWORD *)v39 == 0x6D76766E62696C2DLL )
      {
        v41 = 4;
        *(_DWORD *)(a1 + 240) = 4;
        goto LABEL_40;
      }
    }
  }
  *(_DWORD *)(a1 + 240) = 0;
  *a13 = 7;
  if ( a2 <= 0 )
  {
    v78 = &v348;
LABEL_143:
    v278 = 0;
    v87 = 0;
  }
  else
  {
    v78 = &v348;
    v79 = v301;
    while ( 1 )
    {
      v80 = *v79;
      v81 = 0;
      s1.m128i_i64[0] = (__int64)v80;
      if ( v80 )
        v81 = strlen(v80);
      v82 = 0;
      s1.m128i_i64[1] = v81;
      v83 = sub_C935B0(&s1, &unk_3F15413, 6, 0);
      v84 = s1.m128i_i64[1];
      if ( v83 < s1.m128i_i64[1] )
      {
        v82 = s1.m128i_i64[1] - v83;
        v84 = v83;
      }
      v348.m128i_i64[0] = s1.m128i_i64[0] + v84;
      v348.m128i_i64[1] = v82;
      v85 = sub_C93740(&v348, &unk_3F15413, 6, -1) + 1;
      if ( v85 > v348.m128i_i64[1] )
        v85 = v348.m128i_u64[1];
      v86 = v348.m128i_i64[1] - v82 + v85;
      if ( v86 > v348.m128i_i64[1] )
        v86 = v348.m128i_u64[1];
      if ( v86 == 8 && *(_QWORD *)v348.m128i_i64[0] == 0x65646F6D2D6C632DLL )
        break;
      if ( &v301[(unsigned int)(a2 - 1) + 1] == ++v79 )
        goto LABEL_143;
    }
    v278 = 1;
    v87 = 1;
  }
LABEL_144:
  v88 = &v349;
  sub_95EB40(a1, v87);
  v349.m128i_i32[0] = 7040620;
  v348.m128i_i64[0] = (__int64)&v349;
  v348.m128i_i64[1] = 3;
  sub_95D700(&v324, &v348);
  if ( (__m128i *)v348.m128i_i64[0] != &v349 )
    j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
  v348.m128i_i64[0] = (__int64)&v349;
  v349.m128i_i32[0] = 7630959;
  v348.m128i_i64[1] = 3;
  sub_95D700(&v327, &v348);
  if ( (__m128i *)v348.m128i_i64[0] != &v349 )
    j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
  v348.m128i_i64[0] = (__int64)&v349;
  v349.m128i_i32[0] = 6515820;
  v348.m128i_i64[1] = 3;
  sub_95D700(&v333, &v348);
  if ( (__m128i *)v348.m128i_i64[0] != &v349 )
    j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
  v89 = *(_DWORD *)(a1 + 1640);
  if ( v89 == 2 )
  {
    v348.m128i_i64[0] = (__int64)&v349;
    s1.m128i_i64[0] = 18;
    v240 = sub_22409D0(&v348, &s1, 0);
    v241 = _mm_load_si128((const __m128i *)&xmmword_3F15830);
    v348.m128i_i64[0] = v240;
    v349.m128i_i64[0] = s1.m128i_i64[0];
    *(_WORD *)(v240 + 16) = 30817;
    *(__m128i *)v240 = v241;
  }
  else if ( v89 == 3 )
  {
    v348.m128i_i64[0] = (__int64)&v349;
    s1.m128i_i64[0] = 18;
    v242 = sub_22409D0(&v348, &s1, 0);
    v243 = _mm_load_si128((const __m128i *)&xmmword_3F15830);
    v348.m128i_i64[0] = v242;
    v349.m128i_i64[0] = s1.m128i_i64[0];
    *(_WORD *)(v242 + 16) = 25705;
    *(__m128i *)v242 = v243;
    v348.m128i_i64[1] = s1.m128i_i64[0];
    *(_BYTE *)(v348.m128i_i64[0] + s1.m128i_i64[0]) = 0;
    sub_95D700(&v327, &v348);
    if ( (__m128i *)v348.m128i_i64[0] != &v349 )
      j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
    v348.m128i_i64[0] = (__int64)&v349;
    s1.m128i_i64[0] = 17;
    v244 = sub_22409D0(&v348, &s1, 0);
    v245 = _mm_load_si128((const __m128i *)&xmmword_3F15840);
    v348.m128i_i64[0] = v244;
    v349.m128i_i64[0] = s1.m128i_i64[0];
    *(_BYTE *)(v244 + 16) = 101;
    *(__m128i *)v244 = v245;
  }
  else
  {
    if ( v89 != 4 )
    {
      if ( v89 == 1 )
      {
        v348.m128i_i64[0] = (__int64)&v349;
        s1.m128i_i64[0] = 16;
        v348.m128i_i64[0] = sub_22409D0(&v348, &s1, 0);
        v349.m128i_i64[0] = s1.m128i_i64[0];
        *(__m128i *)v348.m128i_i64[0] = _mm_load_si128((const __m128i *)&xmmword_3F15850);
        v348.m128i_i64[1] = s1.m128i_i64[0];
        *(_BYTE *)(v348.m128i_i64[0] + s1.m128i_i64[0]) = 0;
        sub_95D700(&v327, &v348);
        if ( (__m128i *)v348.m128i_i64[0] != &v349 )
          j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
        *(_DWORD *)(a1 + 1640) = 0;
      }
      goto LABEL_154;
    }
    v348.m128i_i64[0] = (__int64)&v349;
    s1.m128i_i64[0] = 18;
    v260 = sub_22409D0(&v348, &s1, 0);
    v261 = _mm_load_si128((const __m128i *)&xmmword_3F15830);
    v348.m128i_i64[0] = v260;
    v349.m128i_i64[0] = s1.m128i_i64[0];
    *(_WORD *)(v260 + 16) = 28265;
    *(__m128i *)v260 = v261;
  }
  v348.m128i_i64[1] = s1.m128i_i64[0];
  *(_BYTE *)(v348.m128i_i64[0] + s1.m128i_i64[0]) = 0;
  sub_95D700(&v327, &v348);
  if ( (__m128i *)v348.m128i_i64[0] != &v349 )
    j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
LABEL_154:
  v348.m128i_i64[0] = (__int64)&v349;
  strcpy(v349.m128i_i8, "-march=nvptx");
  v348.m128i_i64[1] = 12;
  sub_95D700(&v333, &v348);
  if ( (__m128i *)v348.m128i_i64[0] != &v349 )
    j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
  if ( !*(_DWORD *)(a1 + 1640) )
  {
    if ( a2 > 0 )
    {
      v283 = 0;
      goto LABEL_159;
    }
    if ( a4 == 43962 )
    {
      v280 = 0;
      v212 = 0;
    }
    else
    {
      if ( a4 != 57069 )
        goto LABEL_222;
      v280 = 0;
      v281 = 0;
      v279 = 0;
      v282 = 0;
      v283 = 0;
      v276 = 0;
      v275 = 0;
LABEL_396:
      if ( (*(_BYTE *)a13 & 0x20) == 0 )
        goto LABEL_222;
LABEL_397:
      v212 = v283;
      if ( v283 )
      {
        v212 = v276;
        if ( !v276 )
          goto LABEL_216;
      }
      if ( v279 )
      {
        if ( !v275 )
          goto LABEL_216;
        if ( !v281 )
        {
          *(_BYTE *)(a1 + 232) = 1;
          if ( v212 )
            goto LABEL_403;
LABEL_543:
          sub_95BEE0(v78->m128i_i64, "-opt-discard-value-names=1");
          sub_95D700(&v327, v78);
          sub_2240A30(v78);
          if ( !v212 )
          {
LABEL_403:
            sub_95BEE0(v78->m128i_i64, "-lto-discard-value-names=1");
            sub_95D700(&v330, v78);
            sub_2240A30(v78);
          }
LABEL_216:
          if ( v280 )
          {
            v123 = *a13;
            if ( (*a13 & 0x20) != 0 )
            {
              LOBYTE(v123) = v123 | 0x80;
              v348.m128i_i64[0] = (__int64)v88;
              *a13 = v123;
              s1.m128i_i64[0] = 24;
              v124 = sub_22409D0(v78, &s1, 0);
              v125 = _mm_load_si128((const __m128i *)&xmmword_3F15870);
              v348.m128i_i64[0] = v124;
              v349.m128i_i64[0] = s1.m128i_i64[0];
              *(_QWORD *)(v124 + 16) = 0x676E697265776F6CLL;
              *(__m128i *)v124 = v125;
              v348.m128i_i64[1] = s1.m128i_i64[0];
              *(_BYTE *)(v348.m128i_i64[0] + s1.m128i_i64[0]) = 0;
              sub_95D700(&v327, v78);
              if ( (__m128i *)v348.m128i_i64[0] != v88 )
                j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
              v348.m128i_i64[0] = (__int64)v88;
              s1.m128i_i64[0] = 44;
              v126 = sub_22409D0(v78, &s1, 0);
              v348.m128i_i64[0] = v126;
              v349.m128i_i64[0] = s1.m128i_i64[0];
              *(__m128i *)v126 = _mm_load_si128((const __m128i *)&xmmword_3F15880);
              v127 = _mm_load_si128((const __m128i *)&xmmword_3F15890);
              qmemcpy((void *)(v126 + 32), "ll-as-inline", 12);
              *(__m128i *)(v126 + 16) = v127;
              v348.m128i_i64[1] = s1.m128i_i64[0];
              *(_BYTE *)(v348.m128i_i64[0] + s1.m128i_i64[0]) = 0;
              sub_95D700(&v327, v78);
              if ( (__m128i *)v348.m128i_i64[0] != v88 )
                j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
            }
          }
LABEL_222:
          if ( *(_QWORD *)(a1 + 400) )
            goto LABEL_223;
          v348.m128i_i64[0] = (__int64)v88;
          strcpy(v349.m128i_i8, "-opt=3");
          v348.m128i_i64[1] = 6;
          v213 = (__int64 *)sub_95D600(a1 + 248, (__int64)v78);
          if ( v213 == (__int64 *)(a1 + 256) )
            goto LABEL_223;
          v214 = sub_22417D0(v213 + 4, 61, 0);
          sub_95D770(&v324, v213 + 8);
          sub_95D770(&v327, v213 + 12);
          sub_95D770(&v333, v213 + 16);
          if ( v214 == -1 )
          {
            sub_95BEE0(v78->m128i_i64, "1");
            goto LABEL_408;
          }
          v215 = v213[5];
          if ( v214 + 1 > v215 )
            goto LABEL_565;
          v348.m128i_i64[0] = (__int64)v88;
          sub_95BA30(v78->m128i_i64, (_BYTE *)(v214 + 1 + v213[4]), v213[4] + v215);
LABEL_408:
          sub_2240AE0(v213[20], v78);
          if ( (__m128i *)v348.m128i_i64[0] != v88 )
            j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
LABEL_223:
          if ( *(_QWORD *)(a1 + 560) )
            goto LABEL_224;
          v348.m128i_i64[0] = (__int64)v88;
          s1.m128i_i64[0] = 16;
          v348.m128i_i64[0] = sub_22409D0(v78, &s1, 0);
          v349.m128i_i64[0] = s1.m128i_i64[0];
          *(__m128i *)v348.m128i_i64[0] = _mm_load_si128((const __m128i *)&xmmword_3F158A0);
          v348.m128i_i64[1] = s1.m128i_i64[0];
          *(_BYTE *)(v348.m128i_i64[0] + s1.m128i_i64[0]) = 0;
          v219 = (__int64 *)sub_95D600(a1 + 248, (__int64)v78);
          if ( (__m128i *)v348.m128i_i64[0] != v88 )
            j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
          if ( v219 == (__int64 *)(a1 + 256) )
            goto LABEL_224;
          v220 = sub_22417D0(v219 + 4, 61, 0);
          sub_95D770(&v324, v219 + 8);
          sub_95D770(&v327, v219 + 12);
          sub_95D770(&v333, v219 + 16);
          if ( v220 == -1 )
          {
            sub_95BEE0(v78->m128i_i64, "1");
            goto LABEL_422;
          }
          v221 = v219[5];
          if ( v220 + 1 > v221 )
            goto LABEL_565;
          v348.m128i_i64[0] = (__int64)v88;
          sub_95BA30(v78->m128i_i64, (_BYTE *)(v220 + 1 + v219[4]), v219[4] + v221);
LABEL_422:
          sub_2240AE0(v219[20], v78);
          if ( (__m128i *)v348.m128i_i64[0] != v88 )
            j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
LABEL_224:
          if ( *(_QWORD *)(a1 + 592) )
            goto LABEL_225;
          v348.m128i_i64[0] = (__int64)v88;
          strcpy(v349.m128i_i8, "-ftz=0");
          v348.m128i_i64[1] = 6;
          v216 = (__int64 *)sub_95D600(a1 + 248, (__int64)v78);
          if ( v216 == (__int64 *)(a1 + 256) )
            goto LABEL_225;
          v217 = sub_22417D0(v216 + 4, 61, 0);
          sub_95D770(&v324, v216 + 8);
          sub_95D770(&v327, v216 + 12);
          sub_95D770(&v333, v216 + 16);
          if ( v217 == -1 )
          {
            sub_95BEE0(v78->m128i_i64, "1");
            goto LABEL_414;
          }
          v218 = v216[5];
          if ( v217 + 1 > v218 )
            goto LABEL_565;
          v348.m128i_i64[0] = (__int64)v88;
          sub_95BA30(v78->m128i_i64, (_BYTE *)(v217 + 1 + v216[4]), v216[4] + v218);
LABEL_414:
          sub_2240AE0(v216[20], v78);
          if ( (__m128i *)v348.m128i_i64[0] != v88 )
            j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
LABEL_225:
          v128 = *(_QWORD *)(a1 + 624);
          if ( v278 )
          {
            if ( !v128 )
            {
              sub_95BEE0(v78->m128i_i64, "-prec-sqrt=0");
              v209 = (__int64 *)sub_95D600(a1 + 248, (__int64)v78);
              if ( (__m128i *)v348.m128i_i64[0] != v88 )
                j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
              if ( v209 != (__int64 *)(a1 + 256) )
                goto LABEL_381;
            }
            goto LABEL_227;
          }
          if ( v128 )
            goto LABEL_227;
          v348.m128i_i64[0] = (__int64)v88;
          strcpy(v349.m128i_i8, "-prec-sqrt=1");
          v348.m128i_i64[1] = 12;
          v209 = (__int64 *)sub_95D600(a1 + 248, (__int64)v78);
          if ( v209 == (__int64 *)(a1 + 256) )
            goto LABEL_227;
LABEL_381:
          v210 = sub_22417D0(v209 + 4, 61, 0);
          sub_95D770(&v324, v209 + 8);
          sub_95D770(&v327, v209 + 12);
          sub_95D770(&v333, v209 + 16);
          if ( v210 == -1 )
          {
            sub_95BEE0(v78->m128i_i64, "1");
            goto LABEL_384;
          }
          v211 = v209[5];
          if ( v210 + 1 > v211 )
            goto LABEL_565;
          v348.m128i_i64[0] = (__int64)v88;
          sub_95BA30(v78->m128i_i64, (_BYTE *)(v210 + 1 + v209[4]), v209[4] + v211);
LABEL_384:
          sub_2240AE0(v209[20], v78);
          if ( (__m128i *)v348.m128i_i64[0] != v88 )
            j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
LABEL_227:
          if ( *(_QWORD *)(a1 + 656) )
            goto LABEL_228;
          v348.m128i_i64[0] = (__int64)v88;
          strcpy(v349.m128i_i8, "-prec-div=1");
          v348.m128i_i64[1] = 11;
          v256 = (__int64 *)sub_95D600(a1 + 248, (__int64)v78);
          if ( v256 == (__int64 *)(a1 + 256) )
            goto LABEL_228;
          v257 = sub_22417D0(v256 + 4, 61, 0);
          sub_95D770(&v324, v256 + 8);
          sub_95D770(&v327, v256 + 12);
          sub_95D770(&v333, v256 + 16);
          if ( v257 == -1 )
          {
            sub_95BEE0(v78->m128i_i64, "1");
            goto LABEL_508;
          }
          v258 = v256[5];
          if ( v257 + 1 > v258 )
            goto LABEL_565;
          v348.m128i_i64[0] = (__int64)v88;
          sub_95BA30(v78->m128i_i64, (_BYTE *)(v257 + 1 + v256[4]), v256[4] + v258);
LABEL_508:
          sub_2240AE0(v256[20], v78);
          if ( (__m128i *)v348.m128i_i64[0] != v88 )
            j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
LABEL_228:
          if ( *(_QWORD *)(a1 + 688) )
            goto LABEL_229;
          v348.m128i_i64[0] = (__int64)v88;
          strcpy(v349.m128i_i8, "-fma=1");
          v348.m128i_i64[1] = 6;
          v253 = (__int64 *)sub_95D600(a1 + 248, (__int64)v78);
          if ( v253 == (__int64 *)(a1 + 256) )
            goto LABEL_229;
          v254 = sub_22417D0(v253 + 4, 61, 0);
          sub_95D770(&v324, v253 + 8);
          sub_95D770(&v327, v253 + 12);
          sub_95D770(&v333, v253 + 16);
          if ( v254 == -1 )
          {
            sub_95BEE0(v78->m128i_i64, "1");
            goto LABEL_502;
          }
          v255 = v253[5];
          if ( v254 + 1 > v255 )
            goto LABEL_565;
          v348.m128i_i64[0] = (__int64)v88;
          sub_95BA30(v78->m128i_i64, (_BYTE *)(v254 + 1 + v253[4]), v253[4] + v255);
LABEL_502:
          sub_2240AE0(v253[20], v78);
          if ( (__m128i *)v348.m128i_i64[0] != v88 )
            j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
LABEL_229:
          if ( *(_QWORD *)(a1 + 464) )
            goto LABEL_230;
          v348.m128i_i64[0] = (__int64)v88;
          strcpy(v349.m128i_i8, "-opt-fdiv=0");
          v348.m128i_i64[1] = 11;
          v250 = (__int64 *)sub_95D600(a1 + 248, (__int64)v78);
          if ( v250 == (__int64 *)(a1 + 256) )
            goto LABEL_230;
          v251 = sub_22417D0(v250 + 4, 61, 0);
          sub_95D770(&v324, v250 + 8);
          sub_95D770(&v327, v250 + 12);
          sub_95D770(&v333, v250 + 16);
          if ( v251 == -1 )
          {
            sub_95BEE0(v78->m128i_i64, "1");
            goto LABEL_496;
          }
          v252 = v250[5];
          if ( v251 + 1 > v252 )
LABEL_565:
            sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
          v348.m128i_i64[0] = (__int64)v88;
          sub_95BA30(v78->m128i_i64, (_BYTE *)(v251 + 1 + v250[4]), v250[4] + v252);
LABEL_496:
          sub_2240AE0(v250[20], v78);
          if ( (__m128i *)v348.m128i_i64[0] != v88 )
            j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
LABEL_230:
          if ( !(unsigned int)sub_2241AC0(a1 + 296, "1") || !(unsigned int)sub_2241AC0(a1 + 328, "1") )
            *a13 |= 0x10u;
          v129 = (_QWORD *)(a1 + 392);
          if ( v278 )
          {
            if ( !(unsigned int)sub_2241AC0(a1 + 392, "0") || *(_DWORD *)(a1 + 1640) == 2 )
            {
              v348.m128i_i64[0] = (__int64)v88;
              strcpy(v349.m128i_i8, "-lsa-opt=0");
              v348.m128i_i64[1] = 10;
              sub_95D700(&v327, v78);
              if ( (__m128i *)v348.m128i_i64[0] != v88 )
                j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
              v348.m128i_i64[0] = (__int64)v88;
              s1.m128i_i64[0] = 19;
              v262 = sub_22409D0(v78, &s1, 0);
              v263 = _mm_load_si128((const __m128i *)&xmmword_3F158B0);
              v348.m128i_i64[0] = v262;
              v349.m128i_i64[0] = s1.m128i_i64[0];
              *(_WORD *)(v262 + 16) = 15732;
              *(_BYTE *)(v262 + 18) = 48;
              *(__m128i *)v262 = v263;
LABEL_241:
              v348.m128i_i64[1] = s1.m128i_i64[0];
              *(_BYTE *)(v348.m128i_i64[0] + s1.m128i_i64[0]) = 0;
            }
            else
            {
              sub_95BEE0(v78->m128i_i64, "-memory-space-opt=1");
            }
            sub_95D700(&v327, v78);
            if ( (__m128i *)v348.m128i_i64[0] != v88 )
              j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
          }
          else
          {
            if ( *(_DWORD *)(a1 + 1640) == 2 )
            {
              v348.m128i_i64[0] = (__int64)v88;
            }
            else
            {
              v130 = sub_2241AC0(a1 + 392, "0");
              v348.m128i_i64[0] = (__int64)v88;
              if ( v130 )
              {
                s1.m128i_i64[0] = 19;
                v131 = sub_22409D0(v78, &s1, 0);
                v132 = _mm_load_si128((const __m128i *)&xmmword_3F158B0);
                v348.m128i_i64[0] = v131;
                v349.m128i_i64[0] = s1.m128i_i64[0];
                *(_WORD *)(v131 + 16) = 15732;
                *(_BYTE *)(v131 + 18) = 49;
                *(__m128i *)v131 = v132;
                goto LABEL_236;
              }
            }
            strcpy(v349.m128i_i8, "-lsa-opt=0");
            v348.m128i_i64[1] = 10;
            sub_95D700(&v327, v78);
            if ( (__m128i *)v348.m128i_i64[0] != v88 )
              j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
            v348.m128i_i64[0] = (__int64)v88;
            s1.m128i_i64[0] = 19;
            v238 = sub_22409D0(v78, &s1, 0);
            v239 = _mm_load_si128((const __m128i *)&xmmword_3F158B0);
            v348.m128i_i64[0] = v238;
            v349.m128i_i64[0] = s1.m128i_i64[0];
            *(_WORD *)(v238 + 16) = 15732;
            *(_BYTE *)(v238 + 18) = 48;
            *(__m128i *)v238 = v239;
LABEL_236:
            v348.m128i_i64[1] = s1.m128i_i64[0];
            *(_BYTE *)(v348.m128i_i64[0] + s1.m128i_i64[0]) = 0;
            sub_95D700(&v327, v78);
            if ( (__m128i *)v348.m128i_i64[0] != v88 )
              j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
            if ( (unsigned int)sub_2241AC0(a1 + 648, "0") || (unsigned int)sub_2241AC0(a1 + 616, "0") )
            {
              v348.m128i_i64[0] = (__int64)v88;
              s1.m128i_i64[0] = 25;
              v133 = sub_22409D0(v78, &s1, 0);
              v134 = _mm_load_si128((const __m128i *)&xmmword_3F158C0);
              v348.m128i_i64[0] = v133;
              v349.m128i_i64[0] = s1.m128i_i64[0];
              *(_QWORD *)(v133 + 16) = 0x3D74706F2D786F72LL;
              *(_BYTE *)(v133 + 24) = 48;
              *(__m128i *)v133 = v134;
              goto LABEL_241;
            }
          }
          v135 = "-passes=";
          v136 = (_QWORD *)(a1 + 1512);
          if ( !*(_QWORD *)(a1 + 1520) )
          {
            if ( *(_DWORD *)(a1 + 1640) )
            {
LABEL_246:
              if ( (unsigned int)sub_2241AC0(v129, "0") && (unsigned int)sub_2241AC0(a1 + 840, "1") )
              {
                sub_8FD6D0((__int64)v78, "-optO", v129);
                sub_95D700(&v333, v78);
                if ( (__m128i *)v348.m128i_i64[0] != v88 )
                  j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
                sub_95BEE0(v78->m128i_i64, "-llcO2");
                sub_95D700(&v333, v78);
                if ( (__m128i *)v348.m128i_i64[0] != v88 )
                  j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
              }
              v137 = ((char *)v325 - (char *)v324) >> 5;
              *a5 = v137;
              v138 = 8LL * (int)v137;
              if ( (unsigned __int64)(int)v137 > 0xFFFFFFFFFFFFFFFLL )
                v138 = -1;
              v139 = sub_2207820(v138);
              *a6 = v139;
              if ( *a5 > 0 )
              {
                v140 = 0;
                while ( 1 )
                {
                  v141 = v324[2 * v140].m128i_i64[1];
                  *(_QWORD *)(v139 + 8 * v140) = sub_2207820(v141 + 1);
                  sub_2241570(&v324[2 * v140], *(_QWORD *)(*a6 + 8 * v140), v141, 0);
                  v142 = *(_QWORD *)(*a6 + 8 * v140++);
                  *(_BYTE *)(v142 + v141) = 0;
                  if ( *a5 <= (int)v140 )
                    break;
                  v139 = *a6;
                }
              }
              v143 = ((char *)v328 - (char *)v327) >> 5;
              *a7 = v143;
              v144 = 8LL * (int)v143;
              if ( (unsigned __int64)(int)v143 > 0xFFFFFFFFFFFFFFFLL )
                v144 = -1;
              v145 = sub_2207820(v144);
              *a8 = v145;
              v32 = (unsigned int)*a7;
              if ( (int)v32 > 0 )
              {
                v146 = 0;
                while ( 1 )
                {
                  v147 = v327[2 * v146].m128i_i64[1];
                  *(_QWORD *)(v145 + 8 * v146) = sub_2207820(v147 + 1);
                  v32 = *(_QWORD *)(*a8 + 8 * v146);
                  sub_2241570(&v327[2 * v146], v32, v147, 0);
                  v148 = *(_QWORD *)(*a8 + 8 * v146++);
                  *(_BYTE *)(v148 + v147) = 0;
                  if ( *a7 <= (int)v146 )
                    break;
                  v145 = *a8;
                }
              }
              v149 = ((char *)v331 - (char *)v330) >> 5;
              *a9 = v149;
              v150 = 8LL * (int)v149;
              if ( (unsigned __int64)(int)v149 > 0xFFFFFFFFFFFFFFFLL )
                v150 = -1;
              v151 = sub_2207820(v150);
              *a10 = v151;
              if ( *a9 > 0 )
              {
                v152 = 0;
                while ( 1 )
                {
                  v153 = v330[2 * v152].m128i_i64[1];
                  *(_QWORD *)(v151 + 8 * v152) = sub_2207820(v153 + 1);
                  v32 = *(_QWORD *)(*a10 + 8 * v152);
                  sub_2241570(&v330[2 * v152], v32, v153, 0);
                  v154 = *(_QWORD *)(*a10 + 8 * v152++);
                  *(_BYTE *)(v154 + v153) = 0;
                  if ( *a9 <= (int)v152 )
                    break;
                  v151 = *a10;
                }
              }
              v155 = ((char *)v334 - (char *)v333) >> 5;
              *a11 = v155;
              v156 = 8LL * (int)v155;
              if ( (unsigned __int64)(int)v155 > 0xFFFFFFFFFFFFFFFLL )
                v156 = -1;
              v157 = sub_2207820(v156);
              *a12 = v157;
              if ( *a11 > 0 )
              {
                v158 = 0;
                while ( 1 )
                {
                  v159 = v333[2 * v158].m128i_i64[1];
                  *(_QWORD *)(v157 + 8 * v158) = sub_2207820(v159 + 1);
                  v32 = *(_QWORD *)(*a12 + 8 * v158);
                  sub_2241570(&v333[2 * v158], v32, v159, 0);
                  v160 = *(_QWORD *)(*a12 + 8 * v158++);
                  *(_BYTE *)(v160 + v159) = 0;
                  if ( *a11 <= (int)v158 )
                    break;
                  v157 = *a12;
                }
              }
              v16 = 0;
              goto LABEL_92;
            }
            v136 = (_QWORD *)(a1 + 392);
            v135 = "-O";
          }
          sub_8FD6D0((__int64)v78, v135, v136);
          sub_95D700(&v327, v78);
          if ( (__m128i *)v348.m128i_i64[0] != v88 )
            j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
          goto LABEL_246;
        }
        if ( !v282 )
          goto LABEL_216;
        *(_BYTE *)(a1 + 232) = 1;
        if ( v212 )
          goto LABEL_216;
LABEL_580:
        v212 = v282;
        goto LABEL_543;
      }
      if ( v281 )
      {
        if ( !v282 )
          goto LABEL_216;
        *(_BYTE *)(a1 + 232) = 1;
        sub_95BEE0(v78->m128i_i64, "-lnk-discard-value-names=1");
        sub_95D700(&v324, v78);
        sub_2240A30(v78);
        if ( v212 )
          goto LABEL_216;
        goto LABEL_580;
      }
    }
    *(_BYTE *)(a1 + 232) = 1;
    sub_95BEE0(v78->m128i_i64, "-lnk-discard-value-names=1");
    sub_95D700(&v324, v78);
    sub_2240A30(v78);
    if ( !v212 )
      goto LABEL_543;
    goto LABEL_403;
  }
  sub_2241130(a1 + 424, 0, *(_QWORD *)(a1 + 432), "1", 1);
  sub_2241130(a1 + 840, 0, *(_QWORD *)(a1 + 848), "1", 1);
  if ( a2 <= 0 )
    goto LABEL_222;
  v283 = 1;
LABEL_159:
  v280 = 0;
  v302 = 0;
  v286 = (_QWORD *)(a1 + 256);
  v282 = v283;
  v275 = v283;
  v276 = v283;
  v281 = v283;
  v279 = v283;
  v277 = v283;
  while ( 2 )
  {
    v90 = 0;
    v287 = v302;
    src = v301[v287];
    v91 = strlen(src);
    v92 = v91;
    if ( v91 )
    {
      v288 = v91;
      v93 = src;
      while ( isspace(src[v90]) )
      {
        if ( v92 == ++v90 )
        {
          v96 = 1;
          v92 = 0;
          goto LABEL_169;
        }
      }
      srca = v78;
      for ( i = v288; i > v90 && isspace(v93[i - 1]); --i )
        ;
      v95 = i;
      v78 = srca;
      v92 = v95 - v90;
      v96 = v92 + 1;
    }
    else
    {
      v96 = 1;
    }
LABEL_169:
    srcb = (char *)sub_2207820(v96);
    strncpy(srcb, &v301[v287][v90], v92);
    srcb[v92] = 0;
    v322 = srcb;
    v97 = strlen(srcb);
    v348.m128i_i64[0] = (__int64)v88;
    v323 = v97;
    v98 = v97;
    s1.m128i_i64[0] = v97;
    if ( v97 > 0xF )
    {
      v348.m128i_i64[0] = sub_22409D0(v78, &s1, 0);
      v164 = (__m128i *)v348.m128i_i64[0];
      v349.m128i_i64[0] = s1.m128i_i64[0];
    }
    else
    {
      if ( v97 == 1 )
      {
        v349.m128i_i8[0] = *srcb;
        v99 = (__int64)v88;
        goto LABEL_172;
      }
      if ( !v97 )
      {
        v99 = (__int64)v88;
        goto LABEL_172;
      }
      v164 = v88;
    }
    memcpy(v164, srcb, v98);
    v97 = s1.m128i_i64[0];
    v99 = v348.m128i_i64[0];
LABEL_172:
    v348.m128i_i64[1] = v97;
    *(_BYTE *)(v99 + v97) = 0;
    v100 = (_QWORD *)sub_95D600(a1 + 248, (__int64)v78);
    if ( (__m128i *)v348.m128i_i64[0] != v88 )
      j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
    v101 = v100 < v286;
    v102 = v100 == v286;
    if ( v100 != v286 )
    {
      v32 = 61;
      v289 = sub_22417D0(v100 + 4, 61, 0);
      if ( !*(_QWORD *)(v100[20] + 8LL) )
      {
        v309 = sub_2241A40(v100 + 8, 32, 0);
        v103 = sub_22417D0(v100 + 8, 32, v309);
        v104 = v309;
        if ( v309 != -1 )
        {
          v310 = (size_t)v88;
          v105 = v104;
          v106 = v103;
          do
          {
            v107 = v100[9];
            if ( v106 == -1 )
              v106 = v100[9];
            if ( v105 > v107 )
LABEL_516:
              sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
            v108 = v107 - v105;
            v348.m128i_i64[0] = v310;
            if ( v108 > v106 - v105 )
              v108 = v106 - v105;
            sub_95BA30(v78->m128i_i64, (_BYTE *)(v105 + v100[8]), v105 + v100[8] + v108);
            sub_95D700(&v324, v78);
            if ( v348.m128i_i64[0] != v310 )
              j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
            v105 = sub_2241A40(v100 + 8, 32, v106);
            v106 = sub_22417D0(v100 + 8, 32, v105);
          }
          while ( v105 != -1 );
          v88 = (__m128i *)v310;
        }
        v311 = sub_2241A40(v100 + 12, 32, 0);
        v109 = sub_22417D0(v100 + 12, 32, v311);
        v110 = v311;
        if ( v311 != -1 )
        {
          v312 = (size_t)v88;
          v111 = v110;
          v112 = v109;
          do
          {
            v113 = v100[13];
            if ( v112 == -1 )
              v112 = v100[13];
            if ( v111 > v113 )
              goto LABEL_516;
            v114 = v113 - v111;
            v348.m128i_i64[0] = v312;
            if ( v114 > v112 - v111 )
              v114 = v112 - v111;
            sub_95BA30(v78->m128i_i64, (_BYTE *)(v111 + v100[12]), v111 + v100[12] + v114);
            sub_95D700(&v327, v78);
            if ( v348.m128i_i64[0] != v312 )
              j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
            v111 = sub_2241A40(v100 + 12, 32, v112);
            v112 = sub_22417D0(v100 + 12, 32, v111);
          }
          while ( v111 != -1 );
          v88 = (__m128i *)v312;
        }
        v313 = sub_2241A40(v100 + 16, 32, 0);
        v115 = sub_22417D0(v100 + 16, 32, v313);
        v116 = v313;
        if ( v313 != -1 )
        {
          v314 = (size_t)v88;
          v117 = v116;
          v118 = v115;
          do
          {
            v119 = v100[17];
            if ( v118 == -1 )
              v118 = v100[17];
            if ( v117 > v119 )
              goto LABEL_516;
            v120 = v119 - v117;
            v348.m128i_i64[0] = v314;
            if ( v120 > v118 - v117 )
              v120 = v118 - v117;
            sub_95BA30(v78->m128i_i64, (_BYTE *)(v117 + v100[16]), v117 + v100[16] + v120);
            sub_95D700(&v333, v78);
            if ( v348.m128i_i64[0] != v314 )
              j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
            v117 = sub_2241A40(v100 + 16, 32, v118);
            v118 = sub_22417D0(v100 + 16, 32, v117);
          }
          while ( v117 != -1 );
          v88 = (__m128i *)v314;
        }
        if ( v289 == -1 )
        {
          v348.m128i_i64[0] = (__int64)v88;
          v348.m128i_i64[1] = 1;
          v349.m128i_i16[0] = 49;
        }
        else
        {
          v121 = v100[5];
          if ( v289 + 1 > v121 )
            sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
          v348.m128i_i64[0] = (__int64)v88;
          sub_95BA30(v78->m128i_i64, (_BYTE *)(v289 + 1 + v100[4]), v100[4] + v121);
        }
        sub_2240AE0(v100[20], v78);
        v122 = v348.m128i_i64[0];
        if ( (__m128i *)v348.m128i_i64[0] != v88 )
LABEL_213:
          j_j___libc_free_0(v122, v349.m128i_i64[0] + 1);
LABEL_214:
        j_j___libc_free_0_0(srcb);
        if ( a2 <= ++v302 )
        {
          if ( !v277 )
          {
            if ( a4 == 43962 )
              goto LABEL_397;
            if ( a4 == 57069 )
              goto LABEL_396;
          }
          goto LABEL_216;
        }
        continue;
      }
      if ( !a14 )
        goto LABEL_461;
      v231 = v100[5];
      v342.m128i_i64[0] = (__int64)v343;
      if ( v289 <= v231 )
        v231 = v289;
      sub_95BA30(v342.m128i_i64, (_BYTE *)v100[4], v100[4] + v231);
      v232 = (__m128i *)sub_2241130(&v342, 0, 0, "libnvvm : error: ", 17);
      v344.m128i_i64[0] = (__int64)&v345;
      if ( (__m128i *)v232->m128i_i64[0] == &v232[1] )
      {
        v345 = _mm_loadu_si128(v232 + 1);
      }
      else
      {
        v344.m128i_i64[0] = v232->m128i_i64[0];
        v345.m128i_i64[0] = v232[1].m128i_i64[0];
      }
      v233 = v232->m128i_i64[1];
      v344.m128i_i64[1] = v233;
      v232->m128i_i64[0] = (__int64)v232[1].m128i_i64;
      v232->m128i_i64[1] = 0;
      v232[1].m128i_i8[0] = 0;
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v344.m128i_i64[1]) > 0x16 )
      {
        v234 = (__m128i *)sub_2241490(&v344, " defined more than once", 23, v233);
        v348.m128i_i64[0] = (__int64)v88;
        if ( (__m128i *)v234->m128i_i64[0] == &v234[1] )
        {
          v349 = _mm_loadu_si128(v234 + 1);
        }
        else
        {
          v348.m128i_i64[0] = v234->m128i_i64[0];
          v349.m128i_i64[0] = v234[1].m128i_i64[0];
        }
        v348.m128i_i64[1] = v234->m128i_i64[1];
        v234->m128i_i64[0] = (__int64)v234[1].m128i_i64;
        v234->m128i_i64[1] = 0;
        v234[1].m128i_i8[0] = 0;
        if ( (__m128i *)v344.m128i_i64[0] != &v345 )
          j_j___libc_free_0(v344.m128i_i64[0], v345.m128i_i64[0] + 1);
        if ( (_QWORD *)v342.m128i_i64[0] != v343 )
          j_j___libc_free_0(v342.m128i_i64[0], v343[0] + 1LL);
        goto LABEL_459;
      }
LABEL_573:
      sub_4262D8((__int64)"basic_string::append");
    }
    break;
  }
  v32 = (__int64)srcb;
  v161 = 8;
  v162 = "-maxreg=";
  do
  {
    if ( !v161 )
      break;
    v101 = *(_BYTE *)v32 < *v162;
    v102 = *(_BYTE *)v32++ == *v162++;
    --v161;
  }
  while ( v102 );
  if ( (!v101 && !v102) == v101 )
  {
    if ( !*(_QWORD *)(a1 + 1200) )
    {
      v163 = strlen(srcb + 8);
      sub_2241130(a1 + 1192, 0, 0, srcb + 8, v163);
      sub_8FD6D0((__int64)v78, "-maxreg=", (_QWORD *)(a1 + 1192));
      sub_95D700(&v327, v78);
      if ( (__m128i *)v348.m128i_i64[0] != v88 )
        j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
      sub_8FD6D0((__int64)v78, "-maxreg=", (_QWORD *)(a1 + 1192));
      sub_95D700(&v333, v78);
      v122 = v348.m128i_i64[0];
      if ( (__m128i *)v348.m128i_i64[0] != v88 )
        goto LABEL_213;
      goto LABEL_214;
    }
    if ( !a14 )
      goto LABEL_461;
    sub_95BEE0(v78->m128i_i64, "libnvvm : error: -maxreg defined more than once");
LABEL_459:
    v235 = v348.m128i_i64[1];
    v32 = sub_2207820(v348.m128i_i64[1] + 1);
    *a14 = v32;
    sub_2241570(v78, v32, v235, 0);
    *(_BYTE *)(*a14 + v235) = 0;
    if ( (__m128i *)v348.m128i_i64[0] != v88 )
    {
      v32 = v349.m128i_i64[0] + 1;
      j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
    }
    goto LABEL_461;
  }
  if ( !strcmp(srcb, "-Xopt") )
  {
    ++v302;
    v167 = v301[v287 + 1];
    v336 = (char *)v338;
    if ( !v167 )
      goto LABEL_58;
    v168 = strlen(v167);
    v348.m128i_i64[0] = v168;
    v169 = v168;
    if ( v168 > 0xF )
    {
      v317 = v168;
      v187 = (char *)sub_22409D0(&v336, v78, 0);
      v169 = v317;
      v336 = v187;
      v188 = v187;
      v338[0] = v348.m128i_i64[0];
    }
    else
    {
      if ( v168 == 1 )
      {
        LOBYTE(v338[0]) = *v167;
        v170 = (char *)v338;
        goto LABEL_309;
      }
      if ( !v168 )
      {
        v170 = (char *)v338;
        goto LABEL_309;
      }
      v188 = (char *)v338;
    }
    memcpy(v188, v167, v169);
    v170 = v336;
    v168 = v348.m128i_i64[0];
LABEL_309:
    v337 = v168;
    v170[v168] = 0;
    if ( memcmp(v336, "-opt-discard-value-names=", 0x19u) )
    {
LABEL_310:
      v171 = v328;
      if ( v328 == v329 )
      {
        sub_8FD760(&v327, v328, (__int64)&v336);
      }
      else
      {
        if ( v328 )
        {
          v172 = v328;
          v328->m128i_i64[0] = (__int64)v328[1].m128i_i64;
          sub_95BD60(v172->m128i_i64, v336, (__int64)&v336[v337]);
          v171 = v328;
        }
        v328 = (__m128i *)&v171[2];
      }
      if ( v336 != (char *)v338 )
        j_j___libc_free_0(v336, v338[0] + 1LL);
      goto LABEL_214;
    }
    v290 = v336;
    v318 = v336 + 25;
    v348.m128i_i64[0] = (__int64)v88;
    v189 = strlen(v336 + 25);
    v190 = v318;
    s1.m128i_i64[0] = v189;
    v191 = v189;
    if ( v189 > 0xF )
    {
      v295 = v189;
      v236 = sub_22409D0(v78, &s1, 0);
      v190 = v318;
      v191 = v295;
      v348.m128i_i64[0] = v236;
      v237 = (void *)v236;
      v349.m128i_i64[0] = s1.m128i_i64[0];
    }
    else
    {
      if ( v189 == 1 )
      {
        v349.m128i_i8[0] = v290[25];
        v192 = (__int64)v88;
LABEL_388:
        v348.m128i_i64[1] = v189;
        *(_BYTE *)(v192 + v189) = 0;
        v276 = (unsigned int)sub_2241AC0(v78, "1") == 0;
        if ( (__m128i *)v348.m128i_i64[0] != v88 )
          j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
        v283 = 1;
        goto LABEL_310;
      }
      if ( !v189 )
      {
        v192 = (__int64)v88;
        goto LABEL_388;
      }
      v237 = v88;
    }
    memcpy(v237, v190, v191);
    v189 = s1.m128i_i64[0];
    v192 = v348.m128i_i64[0];
    goto LABEL_388;
  }
  if ( !strcmp(srcb, "-Xllc") )
  {
    ++v302;
    v173 = v301[v287 + 1];
    v344.m128i_i64[0] = (__int64)&v345;
    if ( !v173 )
      goto LABEL_58;
    v315 = v173;
    v174 = strlen(v173);
    v175 = v315;
    v348.m128i_i64[0] = v174;
    v176 = v174;
    if ( v174 > 0xF )
    {
      v291 = v174;
      v200 = sub_22409D0(&v344, v78, 0);
      v175 = v315;
      v176 = v291;
      v344.m128i_i64[0] = v200;
      v201 = (__m128i *)v200;
      v345.m128i_i64[0] = v348.m128i_i64[0];
    }
    else
    {
      if ( v174 == 1 )
      {
        v345.m128i_i8[0] = *v315;
        v177 = &v345;
        goto LABEL_321;
      }
      if ( !v174 )
      {
        v177 = &v345;
        goto LABEL_321;
      }
      v201 = &v345;
    }
    memcpy(v201, v175, v176);
    v174 = v348.m128i_i64[0];
    v177 = (__m128i *)v344.m128i_i64[0];
LABEL_321:
    v344.m128i_i64[1] = v174;
    v177->m128i_i8[v174] = 0;
    sub_95D700(&v333, &v344);
    if ( (__m128i *)v344.m128i_i64[0] != &v345 )
      j_j___libc_free_0(v344.m128i_i64[0], v345.m128i_i64[0] + 1);
    goto LABEL_214;
  }
  if ( !strcmp(srcb, "-Xlnk") )
  {
    ++v302;
    v178 = v301[v287 + 1];
    v339 = (char *)v341;
    if ( !v178 )
      goto LABEL_58;
    v316 = v178;
    v179 = strlen(v178);
    v180 = v316;
    v348.m128i_i64[0] = v179;
    v181 = v179;
    if ( v179 > 0xF )
    {
      v293 = v179;
      v225 = (char *)sub_22409D0(&v339, v78, 0);
      v180 = v316;
      v181 = v293;
      v339 = v225;
      v226 = v225;
      v341[0] = v348.m128i_i64[0];
    }
    else
    {
      if ( v179 == 1 )
      {
        LOBYTE(v341[0]) = *v316;
        v182 = (char *)v341;
        goto LABEL_370;
      }
      if ( !v179 )
      {
        v182 = (char *)v341;
        goto LABEL_370;
      }
      v226 = (char *)v341;
    }
    memcpy(v226, v180, v181);
    v179 = v348.m128i_i64[0];
    v182 = v339;
LABEL_370:
    v340 = v179;
    v182[v179] = 0;
    if ( memcmp(v339, "-lnk-discard-value-names=", 0x19u) )
    {
LABEL_371:
      v207 = v325;
      if ( v325 == v326 )
      {
        sub_8FD760(&v324, v325, (__int64)&v339);
      }
      else
      {
        if ( v325 )
        {
          v208 = v325;
          v325->m128i_i64[0] = (__int64)v325[1].m128i_i64;
          sub_95BD60(v208->m128i_i64, v339, (__int64)&v339[v340]);
          v207 = v325;
        }
        v325 = (__m128i *)&v207[2];
      }
      if ( v339 != (char *)v341 )
        j_j___libc_free_0(v339, v341[0] + 1LL);
      goto LABEL_214;
    }
    v294 = v339;
    v320 = v339 + 25;
    v348.m128i_i64[0] = (__int64)v88;
    v227 = strlen(v339 + 25);
    v228 = v320;
    s1.m128i_i64[0] = v227;
    v229 = v227;
    if ( v227 > 0xF )
    {
      v297 = v227;
      v264 = sub_22409D0(v78, &s1, 0);
      v228 = v320;
      v229 = v297;
      v348.m128i_i64[0] = v264;
      v265 = (void *)v264;
      v349.m128i_i64[0] = s1.m128i_i64[0];
    }
    else
    {
      if ( v227 == 1 )
      {
        v349.m128i_i8[0] = v294[25];
        v230 = (__int64)v88;
LABEL_441:
        v348.m128i_i64[1] = v227;
        *(_BYTE *)(v230 + v227) = 0;
        v275 = (unsigned int)sub_2241AC0(v78, "1") == 0;
        if ( (__m128i *)v348.m128i_i64[0] != v88 )
          j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
        v279 = 1;
        goto LABEL_371;
      }
      if ( !v227 )
      {
        v230 = (__int64)v88;
        goto LABEL_441;
      }
      v265 = v88;
    }
    memcpy(v265, v228, v229);
    v227 = s1.m128i_i64[0];
    v230 = v348.m128i_i64[0];
    goto LABEL_441;
  }
  if ( !strcmp(srcb, "-Xlto") )
  {
    ++v302;
    v193 = v301[v287 + 1];
    v342.m128i_i64[0] = (__int64)v343;
    if ( !v193 )
      goto LABEL_58;
    v319 = v193;
    v194 = strlen(v193);
    v195 = v319;
    v348.m128i_i64[0] = v194;
    v196 = v194;
    if ( v194 > 0xF )
    {
      v292 = v194;
      v222 = sub_22409D0(&v342, v78, 0);
      v195 = v319;
      v196 = v292;
      v342.m128i_i64[0] = v222;
      v223 = (_QWORD *)v222;
      v343[0] = v348.m128i_i64[0];
    }
    else
    {
      if ( v194 == 1 )
      {
        LOBYTE(v343[0]) = *v319;
        v197 = v343;
        goto LABEL_343;
      }
      if ( !v194 )
      {
        v197 = v343;
LABEL_343:
        v342.m128i_i64[1] = v194;
        *((_BYTE *)v197 + v194) = 0;
        if ( memcmp((const void *)v342.m128i_i64[0], "-lto-discard-value-names=", 0x19u) )
        {
LABEL_344:
          v198 = v331;
          if ( v331 == v332 )
          {
            sub_8FD760(&v330, v331, (__int64)&v342);
          }
          else
          {
            if ( v331 )
            {
              v199 = v331;
              v331->m128i_i64[0] = (__int64)v331[1].m128i_i64;
              sub_95BD60(v199->m128i_i64, v342.m128i_i64[0], v342.m128i_i64[0] + v342.m128i_i64[1]);
              v198 = v331;
            }
            v331 = (__m128i *)&v198[2];
          }
          if ( (_QWORD *)v342.m128i_i64[0] != v343 )
            j_j___libc_free_0(v342.m128i_i64[0], v343[0] + 1LL);
          goto LABEL_214;
        }
        v296 = v342.m128i_i64[0];
        v321 = v342.m128i_i64[0] + 25;
        v348.m128i_i64[0] = (__int64)v88;
        v246 = strlen((const char *)(v342.m128i_i64[0] + 25));
        v247 = (const void *)v321;
        s1.m128i_i64[0] = v246;
        v248 = v246;
        if ( v246 > 0xF )
        {
          v298 = v246;
          v266 = sub_22409D0(v78, &s1, 0);
          v247 = (const void *)v321;
          v248 = v298;
          v348.m128i_i64[0] = v266;
          v267 = (void *)v266;
          v349.m128i_i64[0] = s1.m128i_i64[0];
        }
        else
        {
          if ( v246 == 1 )
          {
            v349.m128i_i8[0] = *(_BYTE *)(v296 + 25);
            v249 = (__int64)v88;
LABEL_487:
            v348.m128i_i64[1] = v246;
            *(_BYTE *)(v249 + v246) = 0;
            v282 = (unsigned int)sub_2241AC0(v78, "1") == 0;
            if ( (__m128i *)v348.m128i_i64[0] != v88 )
              j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
            v281 = 1;
            goto LABEL_344;
          }
          if ( !v246 )
          {
            v249 = (__int64)v88;
            goto LABEL_487;
          }
          v267 = v88;
        }
        memcpy(v267, v247, v248);
        v246 = s1.m128i_i64[0];
        v249 = v348.m128i_i64[0];
        goto LABEL_487;
      }
      v223 = v343;
    }
    memcpy(v223, v195, v196);
    v197 = (_QWORD *)v342.m128i_i64[0];
    v194 = v348.m128i_i64[0];
    goto LABEL_343;
  }
  if ( !strcmp(srcb, "-cl-mode") )
    goto LABEL_214;
  if ( !strcmp(srcb, "--device-c") )
  {
    v348.m128i_i64[0] = (__int64)v88;
    qmemcpy(v88, "--device-c", 10);
    v348.m128i_i64[1] = 10;
    v349.m128i_i8[10] = 0;
    goto LABEL_425;
  }
  if ( !strcmp(srcb, "--force-device-c") )
  {
    v348.m128i_i64[0] = (__int64)v88;
    s1.m128i_i64[0] = 16;
    v348.m128i_i64[0] = sub_22409D0(v78, &s1, 0);
    v349.m128i_i64[0] = s1.m128i_i64[0];
    *(__m128i *)v348.m128i_i64[0] = _mm_load_si128((const __m128i *)&xmmword_3F15860);
    v348.m128i_i64[1] = s1.m128i_i64[0];
    *(_BYTE *)(v348.m128i_i64[0] + s1.m128i_i64[0]) = 0;
LABEL_425:
    sub_95D700(&v330, v78);
    v122 = v348.m128i_i64[0];
    if ( (__m128i *)v348.m128i_i64[0] != v88 )
      goto LABEL_213;
    goto LABEL_214;
  }
  if ( !memcmp(srcb, "-host-ref-ek=", 0xDu) )
  {
    sub_95BEE0(s1.m128i_i64, srcb + 13);
    v259 = "-host-ref-ek=";
    goto LABEL_511;
  }
  v165 = "-host-ref-ik=";
  if ( !memcmp(srcb, "-host-ref-ik=", 0xDu)
    || (v165 = "-host-ref-ec=", !memcmp(srcb, "-host-ref-ec=", 0xDu))
    || (v165 = "-host-ref-ic=", !memcmp(srcb, "-host-ref-ic=", 0xDu))
    || (v165 = "-host-ref-eg=", !memcmp(srcb, "-host-ref-eg=", 0xDu)) )
  {
    sub_95BEE0(s1.m128i_i64, srcb + 13);
    v259 = v165;
LABEL_511:
    sub_8FD6D0((__int64)v78, v259, &s1);
    sub_95D700(&v330, v78);
    if ( (__m128i *)v348.m128i_i64[0] != v88 )
      j_j___libc_free_0(v348.m128i_i64[0], v349.m128i_i64[0] + 1);
    if ( (_QWORD *)s1.m128i_i64[0] != v347 )
      j_j___libc_free_0(s1.m128i_i64[0], v347[0] + 1LL);
    goto LABEL_214;
  }
  if ( !memcmp(srcb, "-host-ref-ig=", 0xDu) )
  {
    sub_95BEE0(s1.m128i_i64, srcb + 13);
    sub_8FD6D0((__int64)v78, "-host-ref-ig=", &s1);
    v268 = &v330;
    v269 = v78;
LABEL_564:
    sub_95D700(v268, v269);
    sub_2240A30(v78);
    sub_2240A30(&s1);
    goto LABEL_214;
  }
  v166 = "-has-global-host-info";
  if ( !strcmp(srcb, "-has-global-host-info")
    || (v166 = "-optimize-unused-variables", !strcmp(srcb, "-optimize-unused-variables")) )
  {
    v270 = v166;
    goto LABEL_601;
  }
  if ( !strcmp(srcb, "--partial-link") )
    goto LABEL_214;
  if ( !strcmp(srcb, "-lto") )
  {
    *a13 = *a13 & 0x300 | 0x23;
    goto LABEL_214;
  }
  if ( !strcmp(srcb, "-olto") )
  {
    sub_95BEE0(v78->m128i_i64, "-olto");
    sub_95D700(&v330, v78);
    sub_2240A30(v78);
    ++v302;
    sub_95BEE0(v78->m128i_i64, v301[v287 + 1]);
    sub_95D700(&v330, v78);
    sub_2240A30(v78);
    goto LABEL_214;
  }
  if ( !strcmp(srcb, "-gen-lto") )
  {
    *a13 = *a13 & 0x300 | 0x21;
    goto LABEL_623;
  }
  if ( !strcmp(srcb, "-gen-lto-and-llc") )
  {
    *a13 |= 0x20u;
LABEL_623:
    v270 = "-gen-lto";
    goto LABEL_601;
  }
  if ( !strcmp(srcb, "-link-lto") )
  {
    v270 = "-link-lto";
    *a13 = *a13 & 0x300 | 0x26;
    goto LABEL_601;
  }
  if ( !strcmp(srcb, "-gen-opt-lto") )
  {
    v280 = 1;
    goto LABEL_214;
  }
  v270 = "--trace";
  if ( !strcmp(srcb, "--trace-lto") )
  {
LABEL_601:
    sub_95BEE0(v78->m128i_i64, v270);
    sub_95D700(&v330, v78);
    sub_2240A30(v78);
    goto LABEL_214;
  }
  if ( !strcmp(srcb, "-inline-info") )
  {
    sub_95BEE0(v78->m128i_i64, "-pass-remarks=inline");
    sub_95D700(&v327, v78);
    sub_2240A30(v78);
    sub_95BEE0(v78->m128i_i64, "-pass-remarks-missed=inline");
    sub_95D700(&v327, v78);
    sub_2240A30(v78);
    sub_95BEE0(v78->m128i_i64, "-pass-remarks-analysis=inline");
    sub_95D700(&v327, v78);
    sub_2240A30(v78);
    goto LABEL_214;
  }
  if ( (a4 == 57069 || a4 == 43962) && !strcmp(srcb, "--emit-optix-ir") )
  {
    sub_95BEE0(v78->m128i_i64, "-do-ip-msp=0");
    sub_95D700(&v327, v78);
    sub_2240A30(v78);
    sub_95BEE0(v78->m128i_i64, "-do-licm=0");
    sub_95D700(&v327, v78);
    sub_2240A30(v78);
    *a13 = *a13 & 0x300 | 0x43;
    goto LABEL_214;
  }
  if ( !strncmp(srcb, "-split-compile=", 0xFu) )
  {
    if ( *(_QWORD *)(a1 + 1488) )
      goto LABEL_603;
    sub_95BAE0(a1 + 1480, srcb + 15);
    sub_8FD6D0((__int64)v78, "-split-compile=", (_QWORD *)(a1 + 1480));
    goto LABEL_599;
  }
  if ( !strncmp(srcb, "-split-compile-extended=", 0x18u) )
  {
    if ( *(_QWORD *)(a1 + 1488) )
    {
LABEL_603:
      v32 = (__int64)"libnvvm : error: split compilation defined more than once";
      if ( !a14 )
        goto LABEL_461;
      goto LABEL_604;
    }
    sub_95BAE0(a1 + 1480, srcb + 24);
    sub_8FD6D0((__int64)v78, "-split-compile-extended=", (_QWORD *)(a1 + 1480));
    sub_95D700(&v327, v78);
    sub_2240A30(v78);
    *(_BYTE *)(a1 + 1644) = 1;
    goto LABEL_214;
  }
  if ( !strncmp(srcb, "-Ofast-compile=", 0xFu) )
    goto LABEL_214;
  if ( !strncmp(srcb, "-jump-table-density=", 0x14u) )
  {
    sub_95BEE0(s1.m128i_i64, srcb + 20);
    sub_8FD6D0((__int64)v78, "-jump-table-density=", &s1);
    v268 = &v333;
    v269 = v78;
    goto LABEL_564;
  }
  if ( !strncmp(srcb, "-discard-value-names=", 0x15u) )
  {
    if ( v277 || v283 || v279 || v281 )
    {
      v32 = (__int64)"libnvvm : error: -discard-value-names defined more than once, or defined for both libnvvm and sub-phase";
      if ( !a14 )
        goto LABEL_461;
LABEL_604:
      sub_95BEE0(v78->m128i_i64, (const char *)v32);
      v271 = v348.m128i_i64[1];
      v32 = sub_2207820(v348.m128i_i64[1] + 1);
      *a14 = v32;
      sub_2241570(v78, v32, v271, 0);
      *(_BYTE *)(*a14 + v271) = 0;
      sub_2240A30(v78);
      goto LABEL_461;
    }
    sub_95BEE0(s1.m128i_i64, srcb + 21);
    if ( !strcmp((const char *)s1.m128i_i64[0], "1") )
    {
      *(_BYTE *)(a1 + 232) = 1;
      sub_95BEE0(v78->m128i_i64, "-lnk-discard-value-names=1");
      sub_95D700(&v324, v78);
      sub_2240A30(v78);
      sub_95BEE0(v78->m128i_i64, "-opt-discard-value-names=1");
      sub_95D700(&v327, v78);
      sub_2240A30(v78);
      v272 = "-lto-discard-value-names=1";
    }
    else
    {
      v272 = "-lto-discard-value-names=0";
      *(_BYTE *)(a1 + 232) = 0;
    }
    sub_95BEE0(v78->m128i_i64, v272);
    sub_95D700(&v330, v78);
    sub_2240A30(v78);
    sub_2240A30(&s1);
    v277 = 1;
    goto LABEL_214;
  }
  if ( (unsigned __int8)sub_95CB50(&v322, "-opt-passes=", 0xCu) )
  {
    sub_95CA80(v78->m128i_i64, (__int64)&v322);
    sub_95BC80(a1 + 1512, (__int64)v78);
    sub_2240A30(v78);
    goto LABEL_214;
  }
  sub_95BC50(v78->m128i_i64, byte_3F157B0, 14);
  v273 = sub_95BB10(v322, v323, (const void *)v348.m128i_i64[0], v348.m128i_u64[1]);
  sub_2240A30(v78);
  if ( v273 )
  {
    v32 = (__int64)a14;
    if ( (unsigned int)sub_95C230(srcb, a14, a13) != 1 )
      goto LABEL_461;
    goto LABEL_214;
  }
  v32 = (__int64)"-jobserver";
  if ( !strncmp(srcb, "-jobserver", 0xAu) )
  {
    sub_95BEE0(v78->m128i_i64, "-jobserver");
LABEL_599:
    sub_95D700(&v327, v78);
    sub_2240A30(v78);
    goto LABEL_214;
  }
  if ( a14 )
  {
    sub_95BEE0(s1.m128i_i64, srcb);
    sub_95D570(v78, "libnvvm : error: ", (__int64)&s1);
    sub_94F930(&v344, (__int64)v78, " is an unsupported option");
    sub_2240A30(v78);
    sub_2240A30(&s1);
    v274 = v344.m128i_i64[1];
    v32 = sub_2207820(v344.m128i_i64[1] + 1);
    *a14 = v32;
    sub_2241570(&v344, v32, v274, 0);
    *(_BYTE *)(*a14 + v274) = 0;
    sub_2240A30(&v344);
  }
LABEL_461:
  j_j___libc_free_0_0(srcb);
LABEL_91:
  v16 = 1;
LABEL_92:
  v68 = v365;
  v69 = &v365[2 * (unsigned int)v366];
  if ( v365 != (_OWORD *)v69 )
  {
    do
    {
      v69 -= 4;
      if ( (_QWORD *)*v69 != v69 + 2 )
      {
        v32 = v69[2] + 1LL;
        j_j___libc_free_0(*v69, v32);
      }
    }
    while ( v68 != v69 );
    v69 = v365;
  }
  if ( v69 != (_QWORD *)v367 )
    _libc_free(v69, v32);
  v70 = v334;
  v71 = v333;
  if ( v334 != v333 )
  {
    do
    {
      if ( (__m128i *)v71->m128i_i64[0] != &v71[1] )
        j_j___libc_free_0(v71->m128i_i64[0], v71[1].m128i_i64[0] + 1);
      v71 += 2;
    }
    while ( v70 != v71 );
    v71 = v333;
  }
  if ( v71 )
    j_j___libc_free_0(v71, v335 - (_QWORD)v71);
  v72 = v331;
  v73 = v330;
  if ( v331 != v330 )
  {
    do
    {
      if ( (__m128i *)v73->m128i_i64[0] != &v73[1] )
        j_j___libc_free_0(v73->m128i_i64[0], v73[1].m128i_i64[0] + 1);
      v73 += 2;
    }
    while ( v72 != v73 );
    v73 = v330;
  }
  if ( v73 )
    j_j___libc_free_0(v73, (char *)v332 - (char *)v73);
  v74 = v328;
  v75 = v327;
  if ( v328 != v327 )
  {
    do
    {
      if ( (__m128i *)v75->m128i_i64[0] != &v75[1] )
        j_j___libc_free_0(v75->m128i_i64[0], v75[1].m128i_i64[0] + 1);
      v75 += 2;
    }
    while ( v74 != v75 );
    v75 = v327;
  }
  if ( v75 )
    j_j___libc_free_0(v75, (char *)v329 - (char *)v75);
  v76 = v325;
  v77 = v324;
  if ( v325 != v324 )
  {
    do
    {
      if ( (__m128i *)v77->m128i_i64[0] != &v77[1] )
        j_j___libc_free_0(v77->m128i_i64[0], v77[1].m128i_i64[0] + 1);
      v77 += 2;
    }
    while ( v76 != v77 );
    v77 = v324;
  }
  if ( v77 )
    j_j___libc_free_0(v77, (char *)v326 - (char *)v77);
  return v16;
}
