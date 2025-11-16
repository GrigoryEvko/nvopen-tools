// Function: sub_12CC750
// Address: 0x12cc750
//
__int64 __fastcall sub_12CC750(
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
  char v78; // si
  __m128i *p_s2; // r14
  __m128i *v80; // r15
  int v81; // eax
  __int64 v82; // rcx
  __int64 v83; // r8
  __int64 v84; // r9
  __int64 v85; // rcx
  size_t v86; // rbx
  size_t v87; // rax
  size_t v88; // r13
  const char *v89; // r12
  size_t i; // r14
  size_t v91; // r13
  __int64 v92; // rdi
  size_t v93; // rax
  size_t v94; // r13
  __int64 v95; // rdx
  _QWORD *v96; // rbx
  bool v97; // cf
  bool v98; // zf
  __int64 v99; // rax
  size_t v100; // r8
  unsigned __int64 v101; // r12
  __int64 v102; // r15
  unsigned __int64 v103; // rdx
  unsigned __int64 v104; // rdx
  __int64 v105; // rax
  size_t v106; // r8
  unsigned __int64 v107; // r12
  __int64 v108; // r15
  unsigned __int64 v109; // rdx
  unsigned __int64 v110; // rdx
  __int64 v111; // rax
  size_t v112; // r8
  unsigned __int64 v113; // r12
  __int64 v114; // r15
  unsigned __int64 v115; // rdx
  unsigned __int64 v116; // rdx
  unsigned __int64 v117; // r8
  __int64 v118; // rdi
  int v119; // eax
  __int64 v120; // rax
  __m128i v121; // xmm0
  __int64 v122; // rax
  _QWORD *v123; // r12
  __int64 v124; // rcx
  __int64 v125; // r8
  __int64 v126; // r9
  unsigned int v127; // edx
  __m128i *v128; // rax
  int v129; // eax
  __int64 v130; // rax
  __m128i v131; // xmm0
  __int64 v132; // r13
  _QWORD *v133; // r13
  signed __int64 v134; // rax
  __int64 v135; // rdi
  __int64 v136; // rax
  __int64 v137; // rbx
  __int64 v138; // r13
  __int64 v139; // rax
  signed __int64 v140; // rax
  __int64 v141; // rdi
  __int64 v142; // rax
  __int64 v143; // rbx
  __int64 v144; // r13
  __int64 v145; // rax
  signed __int64 v146; // rax
  __int64 v147; // rdi
  __int64 v148; // rax
  __int64 v149; // rbx
  __int64 v150; // r13
  __int64 v151; // rax
  __int64 v152; // rax
  __int64 v153; // rdi
  __int64 v154; // rax
  __int64 v155; // rbx
  __int64 v156; // r13
  __int64 v157; // rax
  __int64 v158; // rax
  __m128i si128; // xmm0
  __m128i v160; // xmm0
  __int64 v161; // rbx
  __m128i *v162; // rax
  __int64 v163; // rcx
  __m128i *v164; // rax
  __int64 v165; // rbx
  __int64 v166; // rax
  __int64 *v167; // rbx
  __int64 v168; // r12
  unsigned __int64 v169; // rcx
  const char **v170; // rbx
  const char *v171; // rdi
  size_t v172; // rax
  __int64 v173; // r12
  unsigned __int64 v174; // rdx
  __int64 v175; // rax
  unsigned __int64 v176; // rax
  unsigned __int64 v177; // rax
  __int64 *v178; // rbx
  __int64 v179; // r12
  unsigned __int64 v180; // rcx
  __int64 *v181; // rbx
  __int64 v182; // r12
  unsigned __int64 v183; // rcx
  __int64 *v184; // rbx
  __int64 v185; // r12
  unsigned __int64 v186; // rcx
  unsigned int v187; // edx
  __m128i *v188; // rax
  __int64 v189; // rcx
  const char *v190; // rdi
  size_t v191; // rax
  __m128i *v192; // rdi
  const char *v193; // r13
  const char *v194; // r8
  const char *v195; // r13
  size_t v196; // rax
  size_t v197; // r8
  char *v198; // rdx
  const __m128i *v199; // rax
  __m128i *v200; // rdi
  const char *v201; // r9
  size_t v202; // rax
  const char *v203; // r9
  size_t v204; // r8
  __m128i *v205; // rdx
  const char *v206; // r9
  size_t v207; // rax
  const char *v208; // r9
  size_t v209; // r8
  char *v210; // rdx
  const __m128i *v211; // rax
  __m128i *v212; // rdi
  size_t v213; // rax
  const void *v214; // r8
  size_t v215; // r9
  __int64 v216; // rdx
  char *v217; // rax
  char *v218; // rdi
  const char *v219; // r9
  size_t v220; // rax
  const char *v221; // r9
  size_t v222; // r8
  _QWORD *v223; // rdx
  const __m128i *v224; // rax
  __m128i *v225; // rdi
  __int64 v226; // rax
  __m128i *v227; // rdi
  _QWORD *v228; // rdi
  unsigned int v229; // edx
  __m128i *v230; // rax
  __int64 v231; // rax
  const char *v232; // rcx
  __m128i *v233; // rdx
  __int64 v234; // rax
  unsigned __int64 v235; // rsi
  __int8 *v236; // rdx
  unsigned int v237; // eax
  unsigned int v238; // eax
  unsigned int v239; // edx
  __int64 v240; // rdi
  unsigned int v241; // edx
  __m128i *v242; // rax
  __int64 v243; // rax
  __m128i v244; // xmm0
  __int64 v245; // rax
  __m128i v246; // xmm0
  char *v247; // rax
  char *v248; // rdi
  size_t v249; // rax
  const void *v250; // r8
  size_t v251; // r9
  __int64 v252; // rdx
  size_t v253; // rdx
  __m128i *v254; // rax
  __int64 v255; // rcx
  __m128i *v256; // rax
  __int64 v257; // rbx
  __m128i *v258; // rax
  __m128i v259; // xmm0
  __int64 v260; // rax
  void *v261; // rdi
  __int64 v262; // rax
  __m128i v263; // xmm0
  __int64 v264; // rax
  __m128i v265; // xmm0
  __int64 v266; // rax
  __m128i v267; // xmm0
  __int64 *v268; // rbx
  __int64 v269; // r12
  unsigned __int64 v270; // rcx
  __int64 *v271; // rbx
  __int64 v272; // r12
  unsigned __int64 v273; // rcx
  size_t v274; // rax
  const void *v275; // r8
  size_t v276; // r9
  __int64 v277; // rdx
  __int64 *v278; // rbx
  __int64 v279; // r12
  unsigned __int64 v280; // rcx
  __int64 v281; // rax
  _QWORD *v282; // rdi
  __int64 v283; // rax
  __m128i v284; // xmm0
  __int64 v285; // rax
  __m128i v286; // xmm0
  char v287; // bl
  char v288; // r12
  const char *v289; // rsi
  __int64 v290; // rax
  __m128i v291; // xmm0
  __int64 v292; // rcx
  __int64 v293; // r8
  __int64 v294; // r9
  unsigned int v295; // edx
  __m128i *v296; // rax
  unsigned int v297; // edx
  __m128i *v298; // rax
  __int64 v299; // rdx
  __int64 v300; // rcx
  __int64 v301; // r8
  __int64 v302; // r9
  __int64 v303; // rax
  void *v304; // rdi
  __int64 v305; // rdx
  __int64 v306; // rcx
  __int64 v307; // r8
  __int64 v308; // r9
  __int64 v309; // rdx
  __int64 v310; // rcx
  __int64 v311; // r8
  __int64 v312; // r9
  __int64 v313; // rdx
  __int64 v314; // rcx
  __int64 v315; // r8
  __int64 v316; // r9
  __int64 v317; // rdx
  __int64 v318; // rcx
  __int64 v319; // r8
  __int64 v320; // r9
  __int64 v321; // rdx
  __int64 v322; // rcx
  __int64 v323; // r8
  __int64 v324; // r9
  __int64 v325; // rdx
  __int64 v326; // rcx
  __int64 v327; // r8
  __int64 v328; // r9
  __int64 v329; // rax
  void *v330; // rdi
  __int64 v331; // rdx
  __int64 v332; // rcx
  __int64 v333; // r8
  __int64 v334; // r9
  const char *v335; // rsi
  __int64 v336; // rbx
  __m128i **v337; // rdi
  __m128i *v338; // rsi
  const char *v339; // rsi
  unsigned __int64 v340; // rbx
  char *v341; // r13
  __int64 v342; // rbx
  char v343; // [rsp+7h] [rbp-579h]
  bool v344; // [rsp+8h] [rbp-578h]
  char v345; // [rsp+9h] [rbp-577h]
  bool v346; // [rsp+Ah] [rbp-576h]
  char v347; // [rsp+Bh] [rbp-575h]
  char v348; // [rsp+Ch] [rbp-574h]
  bool v349; // [rsp+Dh] [rbp-573h]
  char v350; // [rsp+Eh] [rbp-572h]
  char v351; // [rsp+Fh] [rbp-571h]
  _QWORD *v354; // [rsp+28h] [rbp-558h]
  const char **v355; // [rsp+30h] [rbp-550h]
  unsigned int v356; // [rsp+38h] [rbp-548h]
  __int64 v357; // [rsp+38h] [rbp-548h]
  size_t v358; // [rsp+40h] [rbp-540h]
  size_t v359; // [rsp+40h] [rbp-540h]
  char *v360; // [rsp+40h] [rbp-540h]
  size_t v361; // [rsp+40h] [rbp-540h]
  size_t v362; // [rsp+40h] [rbp-540h]
  char *v363; // [rsp+40h] [rbp-540h]
  size_t v364; // [rsp+40h] [rbp-540h]
  size_t v365; // [rsp+40h] [rbp-540h]
  size_t v366; // [rsp+40h] [rbp-540h]
  size_t v367; // [rsp+40h] [rbp-540h]
  size_t v368; // [rsp+40h] [rbp-540h]
  const char **v371; // [rsp+50h] [rbp-530h]
  int v372; // [rsp+58h] [rbp-528h]
  size_t v373; // [rsp+60h] [rbp-520h]
  const char *src; // [rsp+68h] [rbp-518h]
  __m128i *srca; // [rsp+68h] [rbp-518h]
  char *srcb; // [rsp+68h] [rbp-518h]
  size_t n; // [rsp+70h] [rbp-510h]
  size_t v379; // [rsp+88h] [rbp-4F8h]
  size_t v380; // [rsp+88h] [rbp-4F8h]
  size_t v381; // [rsp+88h] [rbp-4F8h]
  size_t v382; // [rsp+88h] [rbp-4F8h]
  size_t v383; // [rsp+88h] [rbp-4F8h]
  size_t v384; // [rsp+88h] [rbp-4F8h]
  size_t v385; // [rsp+88h] [rbp-4F8h]
  const char *v386; // [rsp+88h] [rbp-4F8h]
  const char *v387; // [rsp+88h] [rbp-4F8h]
  _BYTE *v388; // [rsp+88h] [rbp-4F8h]
  size_t v389; // [rsp+88h] [rbp-4F8h]
  const char *v390; // [rsp+88h] [rbp-4F8h]
  _BYTE *v391; // [rsp+88h] [rbp-4F8h]
  size_t v392; // [rsp+88h] [rbp-4F8h]
  void *v393; // [rsp+90h] [rbp-4F0h] BYREF
  unsigned __int64 v394; // [rsp+98h] [rbp-4E8h]
  __m128i *v395; // [rsp+A0h] [rbp-4E0h] BYREF
  __m128i *v396; // [rsp+A8h] [rbp-4D8h]
  const __m128i *v397; // [rsp+B0h] [rbp-4D0h]
  __m128i *v398; // [rsp+C0h] [rbp-4C0h] BYREF
  __m128i *v399; // [rsp+C8h] [rbp-4B8h]
  const __m128i *v400; // [rsp+D0h] [rbp-4B0h]
  __m128i *v401; // [rsp+E0h] [rbp-4A0h] BYREF
  __m128i *v402; // [rsp+E8h] [rbp-498h]
  const __m128i *v403; // [rsp+F0h] [rbp-490h]
  __m128i *v404; // [rsp+100h] [rbp-480h] BYREF
  __m128i *v405; // [rsp+108h] [rbp-478h]
  __int64 v406; // [rsp+110h] [rbp-470h]
  char *v407; // [rsp+120h] [rbp-460h] BYREF
  size_t v408; // [rsp+128h] [rbp-458h]
  _QWORD v409[2]; // [rsp+130h] [rbp-450h] BYREF
  char *v410; // [rsp+140h] [rbp-440h] BYREF
  size_t v411; // [rsp+148h] [rbp-438h]
  _QWORD v412[2]; // [rsp+150h] [rbp-430h] BYREF
  __m128i v413; // [rsp+160h] [rbp-420h] BYREF
  _QWORD v414[2]; // [rsp+170h] [rbp-410h] BYREF
  __m128i v415; // [rsp+180h] [rbp-400h] BYREF
  __m128i v416; // [rsp+190h] [rbp-3F0h] BYREF
  __m128i s1; // [rsp+1A0h] [rbp-3E0h] BYREF
  _QWORD v418[2]; // [rsp+1B0h] [rbp-3D0h] BYREF
  __m128i s2; // [rsp+1C0h] [rbp-3C0h] BYREF
  __m128i v420; // [rsp+1D0h] [rbp-3B0h] BYREF
  __int64 v421; // [rsp+1E0h] [rbp-3A0h]
  __int64 v422; // [rsp+1E8h] [rbp-398h]
  __int64 v423; // [rsp+1F0h] [rbp-390h]
  __int64 v424; // [rsp+1F8h] [rbp-388h]
  __int64 v425; // [rsp+200h] [rbp-380h]
  char v426[8]; // [rsp+208h] [rbp-378h] BYREF
  int v427; // [rsp+210h] [rbp-370h]
  _QWORD *v428; // [rsp+218h] [rbp-368h] BYREF
  _QWORD v429[2]; // [rsp+228h] [rbp-358h] BYREF
  _QWORD v430[28]; // [rsp+238h] [rbp-348h] BYREF
  __int16 v431; // [rsp+318h] [rbp-268h]
  __int64 v432; // [rsp+320h] [rbp-260h]
  __int64 v433; // [rsp+328h] [rbp-258h]
  __int64 v434; // [rsp+330h] [rbp-250h]
  __int64 v435; // [rsp+338h] [rbp-248h]
  _OWORD *v436; // [rsp+340h] [rbp-240h] BYREF
  __int64 v437; // [rsp+348h] [rbp-238h]
  _OWORD v438[35]; // [rsp+350h] [rbp-230h] BYREF

  v371 = a3;
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
    s1.m128i_i64[0] = (__int64)v418;
    v20 = strlen(v19);
    v436 = (_OWORD *)v20;
    v21 = v20;
    if ( v20 > 0xF )
    {
      s1.m128i_i64[0] = sub_22409D0(&s1, &v436, 0);
      v228 = (_QWORD *)s1.m128i_i64[0];
      v418[0] = v436;
    }
    else
    {
      if ( v20 == 1 )
      {
        LOBYTE(v418[0]) = *v18;
        v22 = v418;
LABEL_277:
        s1.m128i_i64[1] = v20;
        *((_BYTE *)v22 + v20) = 0;
        v162 = (__m128i *)sub_2241130(&s1, 0, 0, "libnvvm : error: ", 17);
        s2.m128i_i64[0] = (__int64)&v420;
        if ( (__m128i *)v162->m128i_i64[0] == &v162[1] )
        {
          v420 = _mm_loadu_si128(v162 + 1);
        }
        else
        {
          s2.m128i_i64[0] = v162->m128i_i64[0];
          v420.m128i_i64[0] = v162[1].m128i_i64[0];
        }
        v163 = v162->m128i_i64[1];
        s2.m128i_i64[1] = v163;
        v162->m128i_i64[0] = (__int64)v162[1].m128i_i64;
        v162->m128i_i64[1] = 0;
        v162[1].m128i_i8[0] = 0;
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - s2.m128i_i64[1]) <= 0x18 )
          goto LABEL_734;
        v164 = (__m128i *)sub_2241490(&s2, " is an unsupported option", 25, v163);
        v436 = v438;
        if ( (__m128i *)v164->m128i_i64[0] == &v164[1] )
        {
          v438[0] = _mm_loadu_si128(v164 + 1);
        }
        else
        {
          v436 = (_OWORD *)v164->m128i_i64[0];
          *(_QWORD *)&v438[0] = v164[1].m128i_i64[0];
        }
        v437 = v164->m128i_i64[1];
        v164->m128i_i64[0] = (__int64)v164[1].m128i_i64;
        v164->m128i_i64[1] = 0;
        v164[1].m128i_i8[0] = 0;
        if ( (__m128i *)s2.m128i_i64[0] != &v420 )
          j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
        if ( (_QWORD *)s1.m128i_i64[0] != v418 )
          j_j___libc_free_0(s1.m128i_i64[0], v418[0] + 1LL);
        v165 = v437;
        v166 = sub_2207820(v437 + 1);
        *a14 = v166;
        sub_2241570(&v436, v166, v165, 0);
        *(_BYTE *)(*a14 + v165) = 0;
        if ( v436 != v438 )
          j_j___libc_free_0(v436, *(_QWORD *)&v438[0] + 1LL);
        return 1;
      }
      if ( !v20 )
      {
        v22 = v418;
        goto LABEL_277;
      }
      v228 = v418;
    }
    memcpy(v228, v18, v21);
    v20 = (size_t)v436;
    v22 = (_QWORD *)s1.m128i_i64[0];
    goto LABEL_277;
  }
LABEL_14:
  v395 = 0;
  v436 = v438;
  v437 = 0x1000000000LL;
  v396 = 0;
  *(_DWORD *)(a1 + 8) = 75;
  *(_BYTE *)(a1 + 240) = 0;
  *(_DWORD *)(a1 + 1648) = 0;
  v397 = 0;
  v398 = 0;
  v399 = 0;
  v400 = 0;
  v401 = 0;
  v402 = 0;
  v403 = 0;
  v404 = 0;
  v405 = 0;
  v406 = 0;
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
    if ( !a2 )
    {
      v347 = 0;
      v78 = 0;
      p_s2 = &s2;
      *(_DWORD *)(a1 + 248) = 0;
      *a13 = 7;
      goto LABEL_131;
    }
    v39 = *v371;
    if ( !*v371 || (v40 = strlen(*v371)) == 0 )
    {
      *(_DWORD *)(a1 + 248) = 0;
      *a13 = 7;
LABEL_743:
      p_s2 = &s2;
LABEL_308:
      v347 = 0;
      v78 = 0;
      goto LABEL_131;
    }
LABEL_36:
    if ( v40 == 4 )
    {
      switch ( *(_DWORD *)v39 )
      {
        case 0x6B6E6C2D:
          v41 = 1;
          *(_DWORD *)(a1 + 248) = 1;
          goto LABEL_39;
        case 0x74706F2D:
          v41 = 2;
          *(_DWORD *)(a1 + 248) = 2;
          goto LABEL_39;
        case 0x63766E2D:
        case 0x636C6C2D:
          v41 = 3;
          *(_DWORD *)(a1 + 248) = 3;
LABEL_39:
          s2.m128i_i32[0] = 0;
          v32 = (unsigned int)a2;
          v42 = sub_12C6E90(v41, a2, v371, s2.m128i_i32, a13);
          v43 = *(_DWORD *)(a1 + 248);
          if ( v43 == 2 )
          {
            v16 = 0;
            *a7 = s2.m128i_i32[0];
            *a8 = v42;
          }
          else if ( v43 == 3 )
          {
            v16 = 0;
            *a11 = s2.m128i_i32[0];
            *a12 = v42;
          }
          else
          {
            v16 = 1;
            if ( v43 == 1 )
            {
              v16 = 0;
              *a5 = s2.m128i_i32[0];
              *a6 = v42;
            }
          }
          goto LABEL_91;
      }
    }
    else if ( v40 == 8 && *(_QWORD *)v39 == 0x6D76766E62696C2DLL )
    {
      v41 = 4;
      *(_DWORD *)(a1 + 248) = 4;
      goto LABEL_39;
    }
    *(_DWORD *)(a1 + 248) = 0;
    *a13 = 7;
    if ( a2 > 0 )
    {
      v356 = a2 - 1;
      v355 = v371 + 1;
      goto LABEL_297;
    }
    goto LABEL_743;
  }
  v356 = a2 - 1;
  v23 = v371;
  v355 = v371 + 1;
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
      s1.m128i_i64[0] = (__int64)v418;
      v45 = strlen(v44);
      s2.m128i_i64[0] = v45;
      v46 = v45;
      if ( v45 > 0xF )
      {
        n = v45;
        v63 = sub_22409D0(&s1, &s2, 0);
        v46 = n;
        s1.m128i_i64[0] = v63;
        v64 = (_QWORD *)v63;
        v418[0] = s2.m128i_i64[0];
      }
      else
      {
        if ( v45 == 1 )
        {
          LOBYTE(v418[0]) = *v44;
          v47 = v418;
          goto LABEL_47;
        }
        if ( !v45 )
        {
          v47 = v418;
          goto LABEL_47;
        }
        v64 = v418;
      }
      memcpy(v64, v44, v46);
      v45 = s2.m128i_i64[0];
      v47 = (_QWORD *)s1.m128i_i64[0];
LABEL_47:
      s1.m128i_i64[1] = v45;
      *((_BYTE *)v47 + v45) = 0;
      sub_222DF20(v430);
      v430[27] = 0;
      v432 = 0;
      v433 = 0;
      v430[0] = off_4A06798;
      v431 = 0;
      v434 = 0;
      v435 = 0;
      s2.m128i_i64[0] = (__int64)qword_4A07108;
      *(__int64 *)((char *)s2.m128i_i64 + qword_4A07108[-3]) = (__int64)&unk_4A07130;
      s2.m128i_i64[1] = 0;
      sub_222DD70(&s2.m128i_i8[*(_QWORD *)(s2.m128i_i64[0] - 24)], 0);
      v420.m128i_i64[1] = 0;
      v421 = 0;
      v422 = 0;
      s2.m128i_i64[0] = (__int64)off_4A07178;
      v430[0] = off_4A071A0;
      v423 = 0;
      v420.m128i_i64[0] = (__int64)off_4A07480;
      v424 = 0;
      v425 = 0;
      sub_220A990(v426);
      v427 = 0;
      v420.m128i_i64[0] = (__int64)off_4A07080;
      v428 = v429;
      sub_12C6150((__int64 *)&v428, s1.m128i_i64[0], s1.m128i_i64[0] + s1.m128i_i64[1]);
      v427 = 8;
      sub_223FD50(&v420, v428, 0, 0);
      sub_222DD70(v430, &v420);
      if ( (_QWORD *)s1.m128i_i64[0] != v418 )
        j_j___libc_free_0(s1.m128i_i64[0], v418[0] + 1LL);
      v32 = a1 + 8;
      sub_222E4D0(&s2, a1 + 8);
      v48 = (unsigned int)(*(_DWORD *)(a1 + 8) - 75);
      if ( (unsigned int)v48 > 0x2E || (v53 = 0x60081200F821LL, !_bittest64(&v53, v48)) )
      {
        if ( a14 )
        {
          sub_12C65C0(v415.m128i_i64, &(*v23)[v26]);
          sub_95D570(&s1, "libnvvm : error: ", (__int64)&v415);
          sub_94F930(&v413, (__int64)&s1, " is an unsupported option");
          sub_2240A30(&s1);
          sub_2240A30(&v415);
          v49 = v413.m128i_i64[1];
          v32 = sub_2207820(v413.m128i_i64[1] + 1);
          *a14 = v32;
          sub_2241570(&v413, v32, v49, 0);
          *(_BYTE *)(*a14 + v49) = 0;
          sub_2240A30(&v413);
        }
        v16 = 1;
        sub_223F4B0(&s2);
        goto LABEL_91;
      }
      s1.m128i_i64[0] = (__int64)v418;
      v54 = &(*v23)[v26];
      if ( !v54 )
LABEL_57:
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v55 = strlen(&(*v23)[v26]);
      v415.m128i_i64[0] = v55;
      v56 = v55;
      if ( v55 > 0xF )
      {
        v373 = v55;
        v65 = sub_22409D0(&s1, &v415, 0);
        v56 = v373;
        s1.m128i_i64[0] = v65;
        v66 = (_QWORD *)v65;
        v418[0] = v415.m128i_i64[0];
      }
      else
      {
        if ( v55 == 1 )
        {
          LOBYTE(v418[0]) = *v54;
          v57 = v418;
LABEL_61:
          s1.m128i_i64[1] = v55;
          p_s1 = &s1;
          *((_BYTE *)v57 + v55) = 0;
          sub_8F9C20(&v401, &s1);
          if ( (_QWORD *)s1.m128i_i64[0] != v418 )
          {
            p_s1 = (__m128i *)(v418[0] + 1LL);
            j_j___libc_free_0(s1.m128i_i64[0], v418[0] + 1LL);
          }
          s2.m128i_i64[0] = (__int64)off_4A07178;
          v430[0] = off_4A071A0;
          v420.m128i_i64[0] = (__int64)off_4A07080;
          if ( v428 != v429 )
          {
            p_s1 = (__m128i *)(v429[0] + 1LL);
            j_j___libc_free_0(v428, v429[0] + 1LL);
          }
          v420.m128i_i64[0] = (__int64)off_4A07480;
          sub_2209150(v426, p_s1, v59);
          s2.m128i_i64[0] = (__int64)qword_4A07108;
          *(__int64 *)((char *)s2.m128i_i64 + qword_4A07108[-3]) = (__int64)&unk_4A07130;
          s2.m128i_i64[1] = 0;
          v430[0] = off_4A06798;
          sub_222E050(v430);
          v24 = *v23;
          v27 = &(*v23)[v26];
          goto LABEL_21;
        }
        if ( !v55 )
        {
          v57 = v418;
          goto LABEL_61;
        }
        v66 = v418;
      }
      memcpy(v66, v54, v56);
      v55 = v415.m128i_i64[0];
      v57 = (_QWORD *)s1.m128i_i64[0];
      goto LABEL_61;
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
    if ( *(_DWORD *)(a1 + 1648) )
    {
      if ( a14 )
      {
        s1.m128i_i64[0] = 54;
        s2.m128i_i64[0] = (__int64)&v420;
        v158 = sub_22409D0(&s2, &s1, 0);
        s2.m128i_i64[0] = v158;
        v420.m128i_i64[0] = s1.m128i_i64[0];
        *(__m128i *)v158 = _mm_load_si128((const __m128i *)&xmmword_3F15800);
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F15810);
        *(_DWORD *)(v158 + 48) = 1852776558;
        *(__m128i *)(v158 + 16) = si128;
        v160 = _mm_load_si128((const __m128i *)&xmmword_3F15820);
        *(_WORD *)(v158 + 52) = 25955;
        *(__m128i *)(v158 + 32) = v160;
        s2.m128i_i64[1] = s1.m128i_i64[0];
        *(_BYTE *)(s2.m128i_i64[0] + s1.m128i_i64[0]) = 0;
        v161 = s2.m128i_i64[1];
        v32 = sub_2207820(s2.m128i_i64[1] + 1);
        *a14 = v32;
        sub_2241570(&s2, v32, v161, 0);
        *(_BYTE *)(*a14 + v161) = 0;
        if ( (__m128i *)s2.m128i_i64[0] != &v420 )
        {
          v32 = v420.m128i_i64[0] + 1;
          j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
        }
      }
      goto LABEL_90;
    }
    v34 = &v24[v26 + 15];
    s1.m128i_i64[0] = (__int64)v418;
    v35 = strlen(v34);
    s2.m128i_i64[0] = v35;
    v36 = v35;
    if ( v35 > 0xF )
    {
      s1.m128i_i64[0] = sub_22409D0(&s1, &s2, 0);
      v60 = (_QWORD *)s1.m128i_i64[0];
      v418[0] = s2.m128i_i64[0];
LABEL_69:
      memcpy(v60, v34, v36);
      v35 = s2.m128i_i64[0];
      v37 = (_QWORD *)s1.m128i_i64[0];
      goto LABEL_29;
    }
    if ( v35 == 1 )
    {
      LOBYTE(v418[0]) = *v34;
      v37 = v418;
    }
    else
    {
      if ( v35 )
      {
        v60 = v418;
        goto LABEL_69;
      }
      v37 = v418;
    }
LABEL_29:
    s1.m128i_i64[1] = v35;
    *((_BYTE *)v37 + v35) = 0;
    v38 = (_QWORD *)s1.m128i_i64[0];
    if ( !strcmp((const char *)s1.m128i_i64[0], "max") )
    {
      *(_DWORD *)(a1 + 1648) = 2;
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
          *(_DWORD *)(a1 + 1648) = 4;
        }
        else
        {
          if ( *(_BYTE *)s1.m128i_i64[0] != 48 || *(_BYTE *)(s1.m128i_i64[0] + 1) )
          {
            if ( a14 )
            {
              sub_12C65C0(
                s2.m128i_i64,
                "libnvvm : error: -Ofast-compile called with unsupported level, only supports 0, min, mid, or max");
              v67 = s2.m128i_i64[1];
              v32 = sub_2207820(s2.m128i_i64[1] + 1);
              *a14 = v32;
              sub_2241570(&s2, v32, v67, 0);
              *(_BYTE *)(*a14 + v67) = 0;
              if ( (__m128i *)s2.m128i_i64[0] != &v420 )
              {
                v32 = v420.m128i_i64[0] + 1;
                j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
              }
              v38 = (_QWORD *)s1.m128i_i64[0];
            }
            if ( v38 != v418 )
            {
              v32 = v418[0] + 1LL;
              j_j___libc_free_0(v38, v418[0] + 1LL);
            }
            goto LABEL_90;
          }
          *(_DWORD *)(a1 + 1648) = 1;
        }
      }
      else
      {
        *(_DWORD *)(a1 + 1648) = 3;
      }
    }
    if ( v38 != v418 )
      j_j___libc_free_0(v38, v418[0] + 1LL);
LABEL_33:
    ++v23;
  }
  while ( &v371[(unsigned int)(a2 - 1) + 1] != v23 );
  *a11 = 0;
  *a9 = 0;
  *a7 = 0;
  *a5 = 0;
  *a12 = 0;
  *a10 = 0;
  *a8 = 0;
  *a6 = 0;
  v39 = *v371;
  if ( *v371 )
  {
    v40 = strlen(*v371);
    if ( v40 )
      goto LABEL_36;
  }
  *(_DWORD *)(a1 + 248) = 0;
  *a13 = 7;
LABEL_297:
  p_s2 = &s2;
  v170 = v371;
  while ( 1 )
  {
    v171 = *v170;
    v172 = 0;
    s1.m128i_i64[0] = (__int64)v171;
    if ( v171 )
      v172 = strlen(v171);
    v173 = 0;
    s1.m128i_i64[1] = v172;
    v174 = sub_16D24E0(&s1, byte_3F15413, 6, 0);
    v175 = s1.m128i_i64[1];
    if ( v174 < s1.m128i_i64[1] )
    {
      v173 = s1.m128i_i64[1] - v174;
      v175 = v174;
    }
    s2.m128i_i64[0] = s1.m128i_i64[0] + v175;
    s2.m128i_i64[1] = v173;
    v176 = sub_16D2680(&s2, byte_3F15413, 6, -1) + 1;
    if ( v176 > s2.m128i_i64[1] )
      v176 = s2.m128i_u64[1];
    v177 = s2.m128i_i64[1] - v173 + v176;
    if ( v177 > s2.m128i_i64[1] )
      v177 = s2.m128i_u64[1];
    if ( v177 == 8 && *(_QWORD *)s2.m128i_i64[0] == 0x65646F6D2D6C632DLL )
      break;
    if ( &v355[v356] == ++v170 )
      goto LABEL_308;
  }
  v347 = 1;
  v78 = 1;
LABEL_131:
  v80 = &v420;
  sub_12C8DD0(a1, v78);
  s2.m128i_i64[0] = (__int64)&v420;
  v420.m128i_i32[0] = 7040620;
  s2.m128i_i64[1] = 3;
  sub_8F9C20(&v395, &s2);
  if ( (__m128i *)s2.m128i_i64[0] != &v420 )
    j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
  s2.m128i_i64[0] = (__int64)&v420;
  v420.m128i_i32[0] = 7630959;
  s2.m128i_i64[1] = 3;
  sub_8F9C20(&v398, &s2);
  if ( (__m128i *)s2.m128i_i64[0] != &v420 )
    j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
  s2.m128i_i64[0] = (__int64)&v420;
  v420.m128i_i32[0] = 6515820;
  s2.m128i_i64[1] = 3;
  sub_8F9C20(&v404, &s2);
  if ( (__m128i *)s2.m128i_i64[0] != &v420 )
    j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
  v81 = *(_DWORD *)(a1 + 1648);
  if ( v81 == 2 )
  {
    s2.m128i_i64[0] = (__int64)&v420;
    s1.m128i_i64[0] = 18;
    v262 = sub_22409D0(&s2, &s1, 0);
    v263 = _mm_load_si128((const __m128i *)&xmmword_3F15830);
    s2.m128i_i64[0] = v262;
    v420.m128i_i64[0] = s1.m128i_i64[0];
    *(_WORD *)(v262 + 16) = 30817;
    *(__m128i *)v262 = v263;
  }
  else if ( v81 == 3 )
  {
    s2.m128i_i64[0] = (__int64)&v420;
    s1.m128i_i64[0] = 18;
    v264 = sub_22409D0(&s2, &s1, 0);
    v265 = _mm_load_si128((const __m128i *)&xmmword_3F15830);
    s2.m128i_i64[0] = v264;
    v420.m128i_i64[0] = s1.m128i_i64[0];
    *(_WORD *)(v264 + 16) = 25705;
    *(__m128i *)v264 = v265;
    s2.m128i_i64[1] = s1.m128i_i64[0];
    *(_BYTE *)(s2.m128i_i64[0] + s1.m128i_i64[0]) = 0;
    sub_8F9C20(&v398, &s2);
    if ( (__m128i *)s2.m128i_i64[0] != &v420 )
      j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
    s2.m128i_i64[0] = (__int64)&v420;
    s1.m128i_i64[0] = 17;
    v266 = sub_22409D0(&s2, &s1, 0);
    v267 = _mm_load_si128((const __m128i *)&xmmword_3F15840);
    s2.m128i_i64[0] = v266;
    v420.m128i_i64[0] = s1.m128i_i64[0];
    *(_BYTE *)(v266 + 16) = 101;
    *(__m128i *)v266 = v267;
  }
  else
  {
    if ( v81 != 4 )
    {
      if ( v81 == 1 )
      {
        s2.m128i_i64[0] = (__int64)&v420;
        s1.m128i_i64[0] = 16;
        s2.m128i_i64[0] = sub_22409D0(&s2, &s1, 0);
        v420.m128i_i64[0] = s1.m128i_i64[0];
        *(__m128i *)s2.m128i_i64[0] = _mm_load_si128((const __m128i *)&xmmword_3F15850);
        s2.m128i_i64[1] = s1.m128i_i64[0];
        *(_BYTE *)(s2.m128i_i64[0] + s1.m128i_i64[0]) = 0;
        sub_8F9C20(&v398, &s2);
        if ( (__m128i *)s2.m128i_i64[0] != &v420 )
          j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
        *(_DWORD *)(a1 + 1648) = 0;
      }
      goto LABEL_141;
    }
    s2.m128i_i64[0] = (__int64)&v420;
    s1.m128i_i64[0] = 18;
    v283 = sub_22409D0(&s2, &s1, 0);
    v284 = _mm_load_si128((const __m128i *)&xmmword_3F15830);
    s2.m128i_i64[0] = v283;
    v420.m128i_i64[0] = s1.m128i_i64[0];
    *(_WORD *)(v283 + 16) = 28265;
    *(__m128i *)v283 = v284;
  }
  s2.m128i_i64[1] = s1.m128i_i64[0];
  *(_BYTE *)(s2.m128i_i64[0] + s1.m128i_i64[0]) = 0;
  sub_8F9C20(&v398, &s2);
  if ( (__m128i *)s2.m128i_i64[0] != &v420 )
    j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
LABEL_141:
  s2.m128i_i64[0] = (__int64)&v420;
  strcpy(v420.m128i_i8, "-march=nvptx");
  s2.m128i_i64[1] = 12;
  sub_8F9C20(&v404, &s2);
  if ( (__m128i *)s2.m128i_i64[0] != &v420 )
    j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
  if ( a4 == 57069 )
  {
    s2.m128i_i64[0] = (__int64)&v420;
    s1.m128i_i64[0] = 27;
    v258 = (__m128i *)sub_22409D0(&s2, &s1, 0);
    v259 = _mm_load_si128((const __m128i *)&xmmword_4281A10);
    s2.m128i_i64[0] = (__int64)v258;
    v420.m128i_i64[0] = s1.m128i_i64[0];
    qmemcpy(&v258[1], "dding-check", 11);
    *v258 = v259;
    s2.m128i_i64[1] = s1.m128i_i64[0];
    *(_BYTE *)(s2.m128i_i64[0] + s1.m128i_i64[0]) = 0;
    sub_8F9C20(&v398, &s2);
    if ( (__m128i *)s2.m128i_i64[0] != &v420 )
      j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
  }
  if ( *(_DWORD *)(a1 + 1648) )
    goto LABEL_145;
  s2.m128i_i64[0] = (__int64)&v420;
  v229 = v437;
  strcpy(v420.m128i_i8, "nvvm-pretreat");
  s2.m128i_i64[1] = 13;
  if ( (unsigned int)v437 >= HIDWORD(v437) )
  {
    sub_12BE710((__int64)&v436, 0, (unsigned int)v437, v82, v83, v84);
    v229 = v437;
  }
  v230 = (__m128i *)&v436[2 * v229];
  if ( v230 )
  {
    v230->m128i_i64[0] = (__int64)v230[1].m128i_i64;
    if ( (__m128i *)s2.m128i_i64[0] == &v420 )
    {
      v230[1] = _mm_load_si128(&v420);
    }
    else
    {
      v230->m128i_i64[0] = s2.m128i_i64[0];
      v230[1].m128i_i64[0] = v420.m128i_i64[0];
    }
    v230->m128i_i64[1] = s2.m128i_i64[1];
    LODWORD(v437) = v437 + 1;
  }
  else
  {
    LODWORD(v437) = v229 + 1;
    if ( (__m128i *)s2.m128i_i64[0] != &v420 )
      j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
  }
  s2.m128i_i64[0] = (__int64)&v420;
  if ( v347 )
  {
    s1.m128i_i64[0] = 22;
    v231 = sub_22409D0(&s2, &s1, 0);
    v232 = "check-kernel-functions";
    s2.m128i_i64[0] = v231;
    v233 = (__m128i *)v231;
    v420.m128i_i64[0] = s1.m128i_i64[0];
    v234 = 22;
  }
  else
  {
    v80 = &v420;
    v234 = 15;
    v232 = "generic-to-nvvm";
    s1.m128i_i64[0] = 15;
    v233 = &v420;
  }
  v233->m128i_i64[0] = *(_QWORD *)v232;
  *(__int64 *)((char *)&v233->m128i_i64[-1] + v234) = *(_QWORD *)&v232[v234 - 8];
  v235 = (unsigned __int64)&v233->m128i_u64[1] & 0xFFFFFFFFFFFFFFF8LL;
  v236 = &v233->m128i_i8[-v235];
  v85 = v232 - v236;
  v237 = ((_DWORD)v236 + v234) & 0xFFFFFFF8;
  if ( v237 >= 8 )
  {
    v238 = v237 & 0xFFFFFFF8;
    v239 = 0;
    do
    {
      v240 = v239;
      v239 += 8;
      v83 = *(_QWORD *)(v85 + v240);
      *(_QWORD *)(v235 + v240) = v83;
    }
    while ( v239 < v238 );
  }
  s2.m128i_i64[1] = s1.m128i_i64[0];
  *(_BYTE *)(s2.m128i_i64[0] + s1.m128i_i64[0]) = 0;
  v241 = v437;
  if ( (unsigned int)v437 >= HIDWORD(v437) )
  {
    sub_12BE710((__int64)&v436, 0, (unsigned int)v437, v85, v83, v84);
    v241 = v437;
  }
  v242 = (__m128i *)&v436[2 * v241];
  if ( v242 )
  {
    v242->m128i_i64[0] = (__int64)v242[1].m128i_i64;
    if ( (__m128i *)s2.m128i_i64[0] == &v420 )
    {
      v242[1] = _mm_load_si128(&v420);
    }
    else
    {
      v242->m128i_i64[0] = s2.m128i_i64[0];
      v242[1].m128i_i64[0] = v420.m128i_i64[0];
    }
    v242->m128i_i64[1] = s2.m128i_i64[1];
    LODWORD(v437) = v437 + 1;
  }
  else
  {
    LODWORD(v437) = v241 + 1;
    if ( (__m128i *)s2.m128i_i64[0] != &v420 )
      j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
  }
  if ( *(_DWORD *)(a1 + 1648) )
  {
LABEL_145:
    sub_2241130(a1 + 432, 0, *(_QWORD *)(a1 + 440), "1", 1);
    sub_2241130(a1 + 848, 0, *(_QWORD *)(a1 + 856), "1", 1);
    if ( a2 > 0 )
    {
      v351 = 1;
      goto LABEL_147;
    }
    goto LABEL_212;
  }
  if ( a2 <= 0 )
  {
    if ( a4 != 43962 )
    {
      if ( a4 == 57069 )
      {
        v351 = 0;
        v346 = 1;
        v348 = 0;
        v344 = 1;
        v345 = 0;
        v349 = 1;
        v350 = 0;
        goto LABEL_567;
      }
      goto LABEL_212;
    }
    v345 = 0;
    v287 = 0;
    v349 = 1;
    v350 = 0;
    goto LABEL_651;
  }
  v351 = 0;
LABEL_147:
  v349 = 1;
  v346 = 1;
  v344 = 1;
  v354 = (_QWORD *)(a1 + 264);
  v350 = v351;
  v348 = v351;
  v343 = v351;
  v345 = 0;
  v372 = 0;
  while ( 2 )
  {
    v86 = 0;
    v357 = v372;
    src = v371[v357];
    v87 = strlen(src);
    v88 = v87;
    if ( v87 )
    {
      v358 = v87;
      v89 = src;
      while ( isspace(src[v86]) )
      {
        if ( v88 == ++v86 )
        {
          v92 = 1;
          v88 = 0;
          goto LABEL_157;
        }
      }
      srca = p_s2;
      for ( i = v358; i > v86 && isspace(v89[i - 1]); --i )
        ;
      v91 = i;
      p_s2 = srca;
      v88 = v91 - v86;
      v92 = v88 + 1;
    }
    else
    {
      v92 = 1;
    }
LABEL_157:
    srcb = (char *)sub_2207820(v92);
    strncpy(srcb, &v371[v357][v86], v88);
    srcb[v88] = 0;
    v393 = srcb;
    v93 = strlen(srcb);
    s2.m128i_i64[0] = (__int64)v80;
    v394 = v93;
    v94 = v93;
    s1.m128i_i64[0] = v93;
    if ( v93 > 0xF )
    {
      s2.m128i_i64[0] = sub_22409D0(p_s2, &s1, 0);
      v192 = (__m128i *)s2.m128i_i64[0];
      v420.m128i_i64[0] = s1.m128i_i64[0];
    }
    else
    {
      if ( v93 == 1 )
      {
        v420.m128i_i8[0] = *srcb;
        v95 = (__int64)v80;
        goto LABEL_160;
      }
      if ( !v93 )
      {
        v95 = (__int64)v80;
        goto LABEL_160;
      }
      v192 = v80;
    }
    memcpy(v192, srcb, v94);
    v93 = s1.m128i_i64[0];
    v95 = s2.m128i_i64[0];
LABEL_160:
    s2.m128i_i64[1] = v93;
    *(_BYTE *)(v95 + v93) = 0;
    v96 = (_QWORD *)sub_12C8530(a1 + 256, (__int64)p_s2);
    if ( (__m128i *)s2.m128i_i64[0] != v80 )
      j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
    v97 = v96 < v354;
    v98 = v96 == v354;
    if ( v96 != v354 )
    {
      v32 = 61;
      v359 = sub_22417D0(v96 + 4, 61, 0);
      if ( !*(_QWORD *)(v96[20] + 8LL) )
      {
        v379 = sub_2241A40(v96 + 8, 32, 0);
        v99 = sub_22417D0(v96 + 8, 32, v379);
        v100 = v379;
        if ( v379 != -1 )
        {
          v380 = (size_t)v80;
          v101 = v100;
          v102 = v99;
          do
          {
            v103 = v96[9];
            if ( v102 == -1 )
              v102 = v96[9];
            if ( v101 > v103 )
LABEL_553:
              sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
            v104 = v103 - v101;
            s2.m128i_i64[0] = v380;
            if ( v104 > v102 - v101 )
              v104 = v102 - v101;
            sub_12C6150(p_s2->m128i_i64, (_BYTE *)(v101 + v96[8]), v101 + v96[8] + v104);
            sub_8F9C20(&v395, p_s2);
            if ( s2.m128i_i64[0] != v380 )
              j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
            v101 = sub_2241A40(v96 + 8, 32, v102);
            v102 = sub_22417D0(v96 + 8, 32, v101);
          }
          while ( v101 != -1 );
          v80 = (__m128i *)v380;
        }
        v381 = sub_2241A40(v96 + 12, 32, 0);
        v105 = sub_22417D0(v96 + 12, 32, v381);
        v106 = v381;
        if ( v381 != -1 )
        {
          v382 = (size_t)v80;
          v107 = v106;
          v108 = v105;
          do
          {
            v109 = v96[13];
            if ( v108 == -1 )
              v108 = v96[13];
            if ( v107 > v109 )
              goto LABEL_553;
            v110 = v109 - v107;
            s2.m128i_i64[0] = v382;
            if ( v110 > v108 - v107 )
              v110 = v108 - v107;
            sub_12C6150(p_s2->m128i_i64, (_BYTE *)(v107 + v96[12]), v107 + v96[12] + v110);
            sub_8F9C20(&v398, p_s2);
            if ( s2.m128i_i64[0] != v382 )
              j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
            v107 = sub_2241A40(v96 + 12, 32, v108);
            v108 = sub_22417D0(v96 + 12, 32, v107);
          }
          while ( v107 != -1 );
          v80 = (__m128i *)v382;
        }
        v383 = sub_2241A40(v96 + 16, 32, 0);
        v111 = sub_22417D0(v96 + 16, 32, v383);
        v112 = v383;
        if ( v383 != -1 )
        {
          v384 = (size_t)v80;
          v113 = v112;
          v114 = v111;
          do
          {
            v115 = v96[17];
            if ( v114 == -1 )
              v114 = v96[17];
            if ( v113 > v115 )
              goto LABEL_553;
            v116 = v115 - v113;
            s2.m128i_i64[0] = v384;
            if ( v116 > v114 - v113 )
              v116 = v114 - v113;
            sub_12C6150(p_s2->m128i_i64, (_BYTE *)(v113 + v96[16]), v113 + v96[16] + v116);
            sub_8F9C20(&v404, p_s2);
            if ( s2.m128i_i64[0] != v384 )
              j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
            v113 = sub_2241A40(v96 + 16, 32, v114);
            v114 = sub_22417D0(v96 + 16, 32, v113);
          }
          while ( v113 != -1 );
          v80 = (__m128i *)v384;
        }
        if ( v359 == -1 )
        {
          s2.m128i_i64[0] = (__int64)v80;
          s2.m128i_i64[1] = 1;
          v420.m128i_i16[0] = 49;
        }
        else
        {
          v117 = v96[5];
          if ( v359 + 1 > v117 )
            sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
          s2.m128i_i64[0] = (__int64)v80;
          sub_12C6150(p_s2->m128i_i64, (_BYTE *)(v359 + 1 + v96[4]), v96[4] + v117);
        }
        sub_2240AE0(v96[20], p_s2);
        v118 = s2.m128i_i64[0];
        if ( (__m128i *)s2.m128i_i64[0] != v80 )
LABEL_201:
          j_j___libc_free_0(v118, v420.m128i_i64[0] + 1);
LABEL_202:
        j_j___libc_free_0_0(srcb);
        if ( a2 > ++v372 )
          continue;
        if ( v343 )
          goto LABEL_206;
        if ( a4 == 43962 )
          goto LABEL_568;
        if ( a4 != 57069 )
          goto LABEL_206;
LABEL_567:
        if ( (*(_BYTE *)a13 & 0x20) != 0 )
        {
LABEL_568:
          v287 = v351;
          if ( v351 )
          {
            v287 = v344;
            if ( !v344 )
              goto LABEL_206;
          }
          if ( v348 )
          {
            if ( v346 )
            {
              if ( v350 )
              {
                v288 = v349;
                if ( !v349 )
                  goto LABEL_206;
                *(_BYTE *)(a1 + 240) = 1;
                if ( v287 )
                  goto LABEL_206;
              }
              else
              {
                v288 = 0;
                *(_BYTE *)(a1 + 240) = 1;
                if ( v287 )
                  goto LABEL_574;
              }
LABEL_666:
              sub_12C65C0(p_s2->m128i_i64, "-opt-discard-value-names=1");
              sub_8F9C20(&v398, p_s2);
              sub_2240A30(p_s2);
LABEL_656:
              if ( !v288 )
              {
LABEL_574:
                sub_12C65C0(p_s2->m128i_i64, "-lto-discard-value-names=1");
                sub_8F9C20(&v401, p_s2);
                sub_2240A30(p_s2);
              }
            }
LABEL_206:
            if ( v345 )
            {
              v119 = *a13;
              if ( (*a13 & 0x20) != 0 )
              {
                LOBYTE(v119) = v119 | 0x80;
                *a13 = v119;
                sub_12C65C0(p_s2->m128i_i64, "-disable-struct-lowering");
                sub_8F9C20(&v398, p_s2);
                if ( (__m128i *)s2.m128i_i64[0] != v80 )
                  j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
                s2.m128i_i64[0] = (__int64)v80;
                s1.m128i_i64[0] = 44;
                v120 = sub_22409D0(p_s2, &s1, 0);
                s2.m128i_i64[0] = v120;
                v420.m128i_i64[0] = s1.m128i_i64[0];
                *(__m128i *)v120 = _mm_load_si128((const __m128i *)&xmmword_3F15880);
                v121 = _mm_load_si128((const __m128i *)&xmmword_3F15890);
                qmemcpy((void *)(v120 + 32), "ll-as-inline", 12);
                *(__m128i *)(v120 + 16) = v121;
                s2.m128i_i64[1] = s1.m128i_i64[0];
                *(_BYTE *)(s2.m128i_i64[0] + s1.m128i_i64[0]) = 0;
                sub_8F9C20(&v398, p_s2);
                if ( (__m128i *)s2.m128i_i64[0] != v80 )
                  j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
              }
            }
            goto LABEL_212;
          }
LABEL_651:
          v288 = v350;
          if ( v350 )
          {
            v288 = v349;
            if ( !v349 )
              goto LABEL_206;
          }
          *(_BYTE *)(a1 + 240) = 1;
          sub_12C65C0(p_s2->m128i_i64, "-lnk-discard-value-names=1");
          sub_8F9C20(&v395, p_s2);
          if ( (__m128i *)s2.m128i_i64[0] != v80 )
            j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
          if ( !v287 )
            goto LABEL_666;
          goto LABEL_656;
        }
LABEL_212:
        if ( !*(_QWORD *)(a1 + 440) )
        {
          s2.m128i_i64[0] = (__int64)v80;
          v187 = v437;
          strcpy(v420.m128i_i8, "nv-inline-must");
          s2.m128i_i64[1] = 14;
          if ( (unsigned int)v437 >= HIDWORD(v437) )
          {
            sub_12BE710((__int64)&v436, 0, (unsigned int)v437, v85, v83, v84);
            v187 = v437;
          }
          v188 = (__m128i *)&v436[2 * v187];
          if ( v188 )
          {
            v188->m128i_i64[0] = (__int64)v188[1].m128i_i64;
            if ( (__m128i *)s2.m128i_i64[0] == v80 )
            {
              v188[1] = _mm_load_si128(&v420);
            }
            else
            {
              v188->m128i_i64[0] = s2.m128i_i64[0];
              v188[1].m128i_i64[0] = v420.m128i_i64[0];
            }
            v188->m128i_i64[1] = s2.m128i_i64[1];
            LODWORD(v437) = v437 + 1;
          }
          else
          {
            LODWORD(v437) = v187 + 1;
            if ( (__m128i *)s2.m128i_i64[0] != v80 )
              j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
          }
        }
        if ( *(_QWORD *)(a1 + 408) )
          goto LABEL_214;
        s2.m128i_i64[0] = (__int64)v80;
        strcpy(v420.m128i_i8, "-opt=3");
        s2.m128i_i64[1] = 6;
        v184 = (__int64 *)sub_12C8530(a1 + 256, (__int64)p_s2);
        if ( v184 == (__int64 *)(a1 + 264) )
          goto LABEL_214;
        v185 = sub_22417D0(v184 + 4, 61, 0);
        sub_12C8630(&v395, v184 + 8);
        sub_12C8630(&v398, v184 + 12);
        sub_12C8630(&v404, v184 + 16);
        if ( v185 == -1 )
        {
          sub_12C65C0(p_s2->m128i_i64, "1");
          goto LABEL_327;
        }
        v186 = v184[5];
        if ( v185 + 1 > v186 )
          goto LABEL_737;
        s2.m128i_i64[0] = (__int64)v80;
        sub_12C6150(p_s2->m128i_i64, (_BYTE *)(v185 + 1 + v184[4]), v184[4] + v186);
LABEL_327:
        sub_2240AE0(v184[20], p_s2);
        if ( (__m128i *)s2.m128i_i64[0] != v80 )
          j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
LABEL_214:
        if ( *(_QWORD *)(a1 + 568) )
          goto LABEL_215;
        s2.m128i_i64[0] = (__int64)v80;
        s1.m128i_i64[0] = 16;
        s2.m128i_i64[0] = sub_22409D0(p_s2, &s1, 0);
        v420.m128i_i64[0] = s1.m128i_i64[0];
        *(__m128i *)s2.m128i_i64[0] = _mm_load_si128((const __m128i *)&xmmword_3F158A0);
        s2.m128i_i64[1] = s1.m128i_i64[0];
        *(_BYTE *)(s2.m128i_i64[0] + s1.m128i_i64[0]) = 0;
        v181 = (__int64 *)sub_12C8530(a1 + 256, (__int64)p_s2);
        if ( (__m128i *)s2.m128i_i64[0] != v80 )
          j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
        if ( v181 == (__int64 *)(a1 + 264) )
          goto LABEL_215;
        v182 = sub_22417D0(v181 + 4, 61, 0);
        sub_12C8630(&v395, v181 + 8);
        sub_12C8630(&v398, v181 + 12);
        sub_12C8630(&v404, v181 + 16);
        if ( v182 == -1 )
        {
          sub_12C65C0(p_s2->m128i_i64, "1");
          goto LABEL_321;
        }
        v183 = v181[5];
        if ( v182 + 1 > v183 )
          goto LABEL_737;
        s2.m128i_i64[0] = (__int64)v80;
        sub_12C6150(p_s2->m128i_i64, (_BYTE *)(v182 + 1 + v181[4]), v181[4] + v183);
LABEL_321:
        sub_2240AE0(v181[20], p_s2);
        if ( (__m128i *)s2.m128i_i64[0] != v80 )
          j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
LABEL_215:
        if ( *(_QWORD *)(a1 + 600) )
          goto LABEL_216;
        s2.m128i_i64[0] = (__int64)v80;
        strcpy(v420.m128i_i8, "-ftz=0");
        s2.m128i_i64[1] = 6;
        v178 = (__int64 *)sub_12C8530(a1 + 256, (__int64)p_s2);
        if ( v178 == (__int64 *)(a1 + 264) )
          goto LABEL_216;
        v179 = sub_22417D0(v178 + 4, 61, 0);
        sub_12C8630(&v395, v178 + 8);
        sub_12C8630(&v398, v178 + 12);
        sub_12C8630(&v404, v178 + 16);
        if ( v179 == -1 )
        {
          sub_12C65C0(p_s2->m128i_i64, "1");
          goto LABEL_313;
        }
        v180 = v178[5];
        if ( v179 + 1 > v180 )
          goto LABEL_737;
        s2.m128i_i64[0] = (__int64)v80;
        sub_12C6150(p_s2->m128i_i64, (_BYTE *)(v179 + 1 + v178[4]), v178[4] + v180);
LABEL_313:
        sub_2240AE0(v178[20], p_s2);
        if ( (__m128i *)s2.m128i_i64[0] != v80 )
          j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
LABEL_216:
        v122 = *(_QWORD *)(a1 + 632);
        if ( v347 )
        {
          if ( v122 )
            goto LABEL_218;
          s2.m128i_i64[0] = (__int64)v80;
          qmemcpy(&v420, "-prec-sqrt=0", 12);
        }
        else
        {
          if ( v122 )
            goto LABEL_218;
          s2.m128i_i64[0] = (__int64)v80;
          qmemcpy(&v420, "-prec-sqrt=1", 12);
        }
        s2.m128i_i64[1] = 12;
        v420.m128i_i8[12] = 0;
        v167 = (__int64 *)sub_12C8530(a1 + 256, (__int64)p_s2);
        if ( v167 == (__int64 *)(a1 + 264) )
          goto LABEL_218;
        v168 = sub_22417D0(v167 + 4, 61, 0);
        sub_12C8630(&v395, v167 + 8);
        sub_12C8630(&v398, v167 + 12);
        sub_12C8630(&v404, v167 + 16);
        if ( v168 != -1 )
        {
          v169 = v167[5];
          if ( v168 + 1 <= v169 )
          {
            s2.m128i_i64[0] = (__int64)v80;
            sub_12C6150(p_s2->m128i_i64, (_BYTE *)(v168 + 1 + v167[4]), v167[4] + v169);
            goto LABEL_294;
          }
LABEL_737:
          sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
        }
        sub_12C65C0(p_s2->m128i_i64, "1");
LABEL_294:
        sub_2240AE0(v167[20], p_s2);
        if ( (__m128i *)s2.m128i_i64[0] != v80 )
          j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
LABEL_218:
        if ( !*(_QWORD *)(a1 + 664) )
        {
          s2.m128i_i64[0] = (__int64)v80;
          strcpy(v420.m128i_i8, "-prec-div=1");
          s2.m128i_i64[1] = 11;
          v278 = (__int64 *)sub_12C8530(a1 + 256, (__int64)p_s2);
          if ( v278 != (__int64 *)(a1 + 264) )
          {
            v279 = sub_22417D0(v278 + 4, 61, 0);
            sub_12C8630(&v395, v278 + 8);
            sub_12C8630(&v398, v278 + 12);
            sub_12C8630(&v404, v278 + 16);
            if ( v279 == -1 )
            {
              sub_12C65C0(p_s2->m128i_i64, "1");
            }
            else
            {
              v280 = v278[5];
              if ( v279 + 1 > v280 )
                goto LABEL_737;
              s2.m128i_i64[0] = (__int64)v80;
              sub_12C6150(p_s2->m128i_i64, (_BYTE *)(v279 + 1 + v278[4]), v278[4] + v280);
            }
            sub_2240AE0(v278[20], p_s2);
            if ( (__m128i *)s2.m128i_i64[0] != v80 )
              j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
          }
        }
        if ( !*(_QWORD *)(a1 + 696) )
        {
          s2.m128i_i64[0] = (__int64)v80;
          strcpy(v420.m128i_i8, "-fma=1");
          s2.m128i_i64[1] = 6;
          v271 = (__int64 *)sub_12C8530(a1 + 256, (__int64)p_s2);
          if ( v271 != (__int64 *)(a1 + 264) )
          {
            v272 = sub_22417D0(v271 + 4, 61, 0);
            sub_12C8630(&v395, v271 + 8);
            sub_12C8630(&v398, v271 + 12);
            sub_12C8630(&v404, v271 + 16);
            if ( v272 == -1 )
            {
              sub_12C65C0(p_s2->m128i_i64, "1");
            }
            else
            {
              v273 = v271[5];
              if ( v272 + 1 > v273 )
                goto LABEL_737;
              s2.m128i_i64[0] = (__int64)v80;
              sub_12C6150(p_s2->m128i_i64, (_BYTE *)(v272 + 1 + v271[4]), v271[4] + v273);
            }
            sub_2240AE0(v271[20], p_s2);
            if ( (__m128i *)s2.m128i_i64[0] != v80 )
              j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
          }
        }
        if ( !*(_QWORD *)(a1 + 472) )
        {
          s2.m128i_i64[0] = (__int64)v80;
          strcpy(v420.m128i_i8, "-opt-fdiv=0");
          s2.m128i_i64[1] = 11;
          v268 = (__int64 *)sub_12C8530(a1 + 256, (__int64)p_s2);
          if ( v268 != (__int64 *)(a1 + 264) )
          {
            v269 = sub_22417D0(v268 + 4, 61, 0);
            sub_12C8630(&v395, v268 + 8);
            sub_12C8630(&v398, v268 + 12);
            sub_12C8630(&v404, v268 + 16);
            if ( v269 == -1 )
            {
              sub_12C65C0(p_s2->m128i_i64, "1");
            }
            else
            {
              v270 = v268[5];
              if ( v269 + 1 > v270 )
                goto LABEL_737;
              s2.m128i_i64[0] = (__int64)v80;
              sub_12C6150(p_s2->m128i_i64, (_BYTE *)(v269 + 1 + v268[4]), v268[4] + v270);
            }
            sub_2240AE0(v268[20], p_s2);
            if ( (__m128i *)s2.m128i_i64[0] != v80 )
              j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
          }
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 304, "1") || !(unsigned int)sub_2241AC0(a1 + 336, "1") )
          *a13 |= 0x10u;
        v123 = (_QWORD *)(a1 + 400);
        if ( v347 )
        {
          if ( (unsigned int)sub_2241AC0(a1 + 400, "0")
            && (unsigned int)sub_2241AC0(v123, "1")
            && (unsigned int)sub_2241AC0(a1 + 1072, "1") )
          {
            if ( !*(_DWORD *)(a1 + 1648) )
            {
              sub_12C65C0(p_s2->m128i_i64, "inline");
              sub_12C7B30((__int64)&v436, p_s2, v321, v322, v323, v324);
              sub_2240A30(p_s2);
              sub_12C65C0(p_s2->m128i_i64, "globaldce");
              sub_12C7B30((__int64)&v436, p_s2, v325, v326, v327, v328);
              sub_2240A30(p_s2);
            }
          }
          else
          {
            s2.m128i_i64[0] = (__int64)v80;
            v127 = v437;
            strcpy(v420.m128i_i8, "always-inline");
            s2.m128i_i64[1] = 13;
            if ( (unsigned int)v437 >= HIDWORD(v437) )
            {
              sub_12BE710((__int64)&v436, 0, (unsigned int)v437, v124, v125, v126);
              v127 = v437;
            }
            v128 = (__m128i *)&v436[2 * v127];
            if ( v128 )
            {
              v128->m128i_i64[0] = (__int64)v128[1].m128i_i64;
              if ( (__m128i *)s2.m128i_i64[0] == v80 )
              {
                v128[1] = _mm_load_si128(&v420);
              }
              else
              {
                v128->m128i_i64[0] = s2.m128i_i64[0];
                v128[1].m128i_i64[0] = v420.m128i_i64[0];
              }
              v128->m128i_i64[1] = s2.m128i_i64[1];
              LODWORD(v437) = v437 + 1;
            }
            else
            {
              LODWORD(v437) = v127 + 1;
              if ( (__m128i *)s2.m128i_i64[0] != v80 )
                j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
            }
          }
          if ( !(unsigned int)sub_2241AC0(v123, "0") || (v129 = *(_DWORD *)(a1 + 1648), v129 == 2) )
          {
            s2.m128i_i64[0] = (__int64)v80;
            strcpy(v420.m128i_i8, "-lsa-opt=0");
            s2.m128i_i64[1] = 10;
            sub_8F9C20(&v398, p_s2);
            if ( (__m128i *)s2.m128i_i64[0] != v80 )
              j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
            if ( !*(_DWORD *)(a1 + 1648) )
            {
              sub_12C65C0(p_s2->m128i_i64, "sroa");
              sub_12C7B30((__int64)&v436, p_s2, v309, v310, v311, v312);
              if ( (__m128i *)s2.m128i_i64[0] != v80 )
                j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
              sub_12C65C0(p_s2->m128i_i64, "mem2reg");
              sub_12C7B30((__int64)&v436, p_s2, v313, v314, v315, v316);
              if ( (__m128i *)s2.m128i_i64[0] != v80 )
                j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
            }
            s2.m128i_i64[0] = (__int64)v80;
            s1.m128i_i64[0] = 19;
            v285 = sub_22409D0(p_s2, &s1, 0);
            v286 = _mm_load_si128((const __m128i *)&xmmword_3F158B0);
            s2.m128i_i64[0] = v285;
            v420.m128i_i64[0] = s1.m128i_i64[0];
            *(_WORD *)(v285 + 16) = 15732;
            *(_BYTE *)(v285 + 18) = 48;
            *(__m128i *)v285 = v286;
          }
          else
          {
            if ( !v129 )
            {
              sub_12C65C0(p_s2->m128i_i64, "byval-mem2reg");
              sub_12C7B30((__int64)&v436, p_s2, v305, v306, v307, v308);
              if ( (__m128i *)s2.m128i_i64[0] != v80 )
                j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
            }
            s2.m128i_i64[0] = (__int64)v80;
            s1.m128i_i64[0] = 19;
            v130 = sub_22409D0(p_s2, &s1, 0);
            v131 = _mm_load_si128((const __m128i *)&xmmword_3F158B0);
            s2.m128i_i64[0] = v130;
            v420.m128i_i64[0] = s1.m128i_i64[0];
            *(_WORD *)(v130 + 16) = 15732;
            *(_BYTE *)(v130 + 18) = 49;
            *(__m128i *)v130 = v131;
          }
LABEL_236:
          s2.m128i_i64[1] = s1.m128i_i64[0];
          *(_BYTE *)(s2.m128i_i64[0] + s1.m128i_i64[0]) = 0;
          sub_8F9C20(&v398, p_s2);
          if ( (__m128i *)s2.m128i_i64[0] != v80 )
            j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
          goto LABEL_238;
        }
        if ( (unsigned int)sub_2241AC0(a1 + 1072, "1")
          && (unsigned int)sub_2241AC0(v123, "0")
          && (unsigned int)sub_2241AC0(v123, "1") )
        {
          if ( !*(_DWORD *)(a1 + 1648) )
          {
            sub_12C65C0(p_s2->m128i_i64, "inline");
            sub_12C7B30((__int64)&v436, p_s2, v299, v300, v301, v302);
            sub_2240A30(p_s2);
          }
          goto LABEL_598;
        }
        if ( !*(_QWORD *)(a1 + 440) )
        {
          sub_12C65C0(p_s2->m128i_i64, "always-inline");
          v295 = v437;
          if ( (unsigned int)v437 >= HIDWORD(v437) )
          {
            sub_12BE710((__int64)&v436, 0, (unsigned int)v437, v292, v293, v294);
            v295 = v437;
          }
          v296 = (__m128i *)&v436[2 * v295];
          if ( v296 )
          {
            v296->m128i_i64[0] = (__int64)v296[1].m128i_i64;
            if ( (__m128i *)s2.m128i_i64[0] == v80 )
            {
              v296[1] = _mm_load_si128(&v420);
            }
            else
            {
              v296->m128i_i64[0] = s2.m128i_i64[0];
              v296[1].m128i_i64[0] = v420.m128i_i64[0];
            }
            v296->m128i_i64[1] = s2.m128i_i64[1];
            LODWORD(v437) = v437 + 1;
          }
          else
          {
            LODWORD(v437) = v295 + 1;
            if ( (__m128i *)s2.m128i_i64[0] != v80 )
              j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
          }
LABEL_598:
          if ( !*(_QWORD *)(a1 + 440) )
          {
            s2.m128i_i64[0] = (__int64)v80;
            v297 = v437;
            strcpy(v420.m128i_i8, "globaldce");
            s2.m128i_i64[1] = 9;
            if ( (unsigned int)v437 >= HIDWORD(v437) )
            {
              sub_12BE710((__int64)&v436, 0, (unsigned int)v437, v292, v293, v294);
              v297 = v437;
            }
            v298 = (__m128i *)&v436[2 * v297];
            if ( v298 )
            {
              v298->m128i_i64[0] = (__int64)v298[1].m128i_i64;
              if ( (__m128i *)s2.m128i_i64[0] == v80 )
              {
                v298[1] = _mm_load_si128(&v420);
              }
              else
              {
                v298->m128i_i64[0] = s2.m128i_i64[0];
                v298[1].m128i_i64[0] = v420.m128i_i64[0];
              }
              v298->m128i_i64[1] = s2.m128i_i64[1];
              LODWORD(v437) = v437 + 1;
            }
            else
            {
              LODWORD(v437) = v297 + 1;
              if ( (__m128i *)s2.m128i_i64[0] != v80 )
                j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
            }
          }
        }
        if ( *(_DWORD *)(a1 + 1648) == 2 )
        {
          s2.m128i_i64[0] = (__int64)v80;
          strcpy(v420.m128i_i8, "-lsa-opt=0");
          s2.m128i_i64[1] = 10;
          sub_8F9C20(&v398, p_s2);
          if ( (__m128i *)s2.m128i_i64[0] != v80 )
            j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
          s2.m128i_i64[0] = (__int64)v80;
          s1.m128i_i64[0] = 19;
          v290 = sub_22409D0(p_s2, &s1, 0);
          v291 = _mm_load_si128((const __m128i *)&xmmword_3F158B0);
          s2.m128i_i64[0] = v290;
          v420.m128i_i64[0] = s1.m128i_i64[0];
          *(_WORD *)(v290 + 16) = 15732;
          *(_BYTE *)(v290 + 18) = 48;
          *(__m128i *)v290 = v291;
LABEL_457:
          s2.m128i_i64[1] = s1.m128i_i64[0];
          *(_BYTE *)(s2.m128i_i64[0] + s1.m128i_i64[0]) = 0;
        }
        else
        {
          if ( (unsigned int)sub_2241AC0(v123, "0") )
          {
            if ( !*(_DWORD *)(a1 + 1648) )
            {
              sub_12C65C0(p_s2->m128i_i64, "byval-mem2reg");
              sub_12C7B30((__int64)&v436, p_s2, v317, v318, v319, v320);
              if ( (__m128i *)s2.m128i_i64[0] != v80 )
                j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
            }
            s2.m128i_i64[0] = (__int64)v80;
            s1.m128i_i64[0] = 19;
            v243 = sub_22409D0(p_s2, &s1, 0);
            v244 = _mm_load_si128((const __m128i *)&xmmword_3F158B0);
            s2.m128i_i64[0] = v243;
            v420.m128i_i64[0] = s1.m128i_i64[0];
            *(_WORD *)(v243 + 16) = 15732;
            *(_BYTE *)(v243 + 18) = 49;
            *(__m128i *)v243 = v244;
            goto LABEL_457;
          }
          sub_12C65C0(p_s2->m128i_i64, "-lsa-opt=0");
          sub_8F9C20(&v398, p_s2);
          if ( (__m128i *)s2.m128i_i64[0] != v80 )
            j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
          if ( !*(_QWORD *)(a1 + 440) )
          {
            sub_12C65C0(p_s2->m128i_i64, "mem2reg");
            sub_12C7B30((__int64)&v436, p_s2, v331, v332, v333, v334);
            if ( (__m128i *)s2.m128i_i64[0] != v80 )
              j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
          }
          sub_12C65C0(p_s2->m128i_i64, "-memory-space-opt=0");
        }
        sub_8F9C20(&v398, p_s2);
        if ( (__m128i *)s2.m128i_i64[0] != v80 )
          j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
        if ( (unsigned int)sub_2241AC0(a1 + 656, "0") || (unsigned int)sub_2241AC0(a1 + 624, "0") )
        {
          s2.m128i_i64[0] = (__int64)v80;
          s1.m128i_i64[0] = 25;
          v245 = sub_22409D0(p_s2, &s1, 0);
          v246 = _mm_load_si128((const __m128i *)&xmmword_3F158C0);
          s2.m128i_i64[0] = v245;
          v420.m128i_i64[0] = s1.m128i_i64[0];
          *(_QWORD *)(v245 + 16) = 0x3D74706F2D786F72LL;
          *(_BYTE *)(v245 + 24) = 48;
          *(__m128i *)v245 = v246;
          goto LABEL_236;
        }
LABEL_238:
        if ( !*(_DWORD *)(a1 + 1648) )
        {
          v132 = 2LL * (unsigned int)v437;
          v385 = (size_t)&v436[v132];
          if ( v436 != &v436[v132] )
          {
            v133 = v436;
            do
            {
              sub_8FD6D0((__int64)p_s2, "-", v133);
              sub_8F9C20(&v398, p_s2);
              if ( (__m128i *)s2.m128i_i64[0] != v80 )
                j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
              v133 += 4;
            }
            while ( (_QWORD *)v385 != v133 );
          }
        }
        if ( (unsigned int)sub_2241AC0(v123, "0") )
        {
          if ( !*(_DWORD *)(a1 + 1648) )
          {
            sub_8FD6D0((__int64)p_s2, "-O", v123);
            sub_8F9C20(&v398, p_s2);
            if ( (__m128i *)s2.m128i_i64[0] != v80 )
              j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
          }
        }
        if ( (unsigned int)sub_2241AC0(v123, "0") && (unsigned int)sub_2241AC0(a1 + 848, "1") )
        {
          sub_8FD6D0((__int64)p_s2, "-optO", v123);
          sub_8F9C20(&v404, p_s2);
          if ( (__m128i *)s2.m128i_i64[0] != v80 )
            j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
          sub_12C65C0(p_s2->m128i_i64, "-llcO2");
          sub_8F9C20(&v404, p_s2);
          if ( (__m128i *)s2.m128i_i64[0] != v80 )
            j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
        }
        v134 = ((char *)v396 - (char *)v395) >> 5;
        *a5 = v134;
        v135 = 8LL * (int)v134;
        if ( (unsigned __int64)(int)v134 > 0xFFFFFFFFFFFFFFFLL )
          v135 = -1;
        v136 = sub_2207820(v135);
        *a6 = v136;
        if ( *a5 > 0 )
        {
          v137 = 0;
          while ( 1 )
          {
            v138 = v395[2 * v137].m128i_i64[1];
            *(_QWORD *)(v136 + 8 * v137) = sub_2207820(v138 + 1);
            sub_2241570(&v395[2 * v137], *(_QWORD *)(*a6 + 8 * v137), v138, 0);
            v139 = *(_QWORD *)(*a6 + 8 * v137++);
            *(_BYTE *)(v139 + v138) = 0;
            if ( *a5 <= (int)v137 )
              break;
            v136 = *a6;
          }
        }
        v140 = ((char *)v399 - (char *)v398) >> 5;
        *a7 = v140;
        v141 = 8LL * (int)v140;
        if ( (unsigned __int64)(int)v140 > 0xFFFFFFFFFFFFFFFLL )
          v141 = -1;
        v142 = sub_2207820(v141);
        *a8 = v142;
        v32 = (unsigned int)*a7;
        if ( (int)v32 > 0 )
        {
          v143 = 0;
          while ( 1 )
          {
            v144 = v398[2 * v143].m128i_i64[1];
            *(_QWORD *)(v142 + 8 * v143) = sub_2207820(v144 + 1);
            v32 = *(_QWORD *)(*a8 + 8 * v143);
            sub_2241570(&v398[2 * v143], v32, v144, 0);
            v145 = *(_QWORD *)(*a8 + 8 * v143++);
            *(_BYTE *)(v145 + v144) = 0;
            if ( *a7 <= (int)v143 )
              break;
            v142 = *a8;
          }
        }
        v146 = ((char *)v402 - (char *)v401) >> 5;
        *a9 = v146;
        v147 = 8LL * (int)v146;
        if ( (unsigned __int64)(int)v146 > 0xFFFFFFFFFFFFFFFLL )
          v147 = -1;
        v148 = sub_2207820(v147);
        *a10 = v148;
        if ( *a9 > 0 )
        {
          v149 = 0;
          while ( 1 )
          {
            v150 = v401[2 * v149].m128i_i64[1];
            *(_QWORD *)(v148 + 8 * v149) = sub_2207820(v150 + 1);
            v32 = *(_QWORD *)(*a10 + 8 * v149);
            sub_2241570(&v401[2 * v149], v32, v150, 0);
            v151 = *(_QWORD *)(*a10 + 8 * v149++);
            *(_BYTE *)(v151 + v150) = 0;
            if ( *a9 <= (int)v149 )
              break;
            v148 = *a10;
          }
        }
        v152 = ((char *)v405 - (char *)v404) >> 5;
        *a11 = v152;
        v153 = 8LL * (int)v152;
        if ( (unsigned __int64)(int)v152 > 0xFFFFFFFFFFFFFFFLL )
          v153 = -1;
        v154 = sub_2207820(v153);
        *a12 = v154;
        if ( *a11 > 0 )
        {
          v155 = 0;
          while ( 1 )
          {
            v156 = v404[2 * v155].m128i_i64[1];
            *(_QWORD *)(v154 + 8 * v155) = sub_2207820(v156 + 1);
            v32 = *(_QWORD *)(*a12 + 8 * v155);
            sub_2241570(&v404[2 * v155], v32, v156, 0);
            v157 = *(_QWORD *)(*a12 + 8 * v155++);
            *(_BYTE *)(v157 + v156) = 0;
            if ( *a11 <= (int)v155 )
              break;
            v154 = *a12;
          }
        }
        v16 = 0;
        goto LABEL_91;
      }
      if ( !a14 )
        goto LABEL_488;
      v253 = v96[5];
      v413.m128i_i64[0] = (__int64)v414;
      if ( v359 <= v253 )
        v253 = v359;
      sub_12C6150(v413.m128i_i64, (_BYTE *)v96[4], v96[4] + v253);
      v254 = (__m128i *)sub_2241130(&v413, 0, 0, "libnvvm : error: ", 17);
      v415.m128i_i64[0] = (__int64)&v416;
      if ( (__m128i *)v254->m128i_i64[0] == &v254[1] )
      {
        v416 = _mm_loadu_si128(v254 + 1);
      }
      else
      {
        v415.m128i_i64[0] = v254->m128i_i64[0];
        v416.m128i_i64[0] = v254[1].m128i_i64[0];
      }
      v255 = v254->m128i_i64[1];
      v415.m128i_i64[1] = v255;
      v254->m128i_i64[0] = (__int64)v254[1].m128i_i64;
      v254->m128i_i64[1] = 0;
      v254[1].m128i_i8[0] = 0;
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v415.m128i_i64[1]) > 0x16 )
      {
        v256 = (__m128i *)sub_2241490(&v415, " defined more than once", 23, v255);
        s2.m128i_i64[0] = (__int64)v80;
        if ( (__m128i *)v256->m128i_i64[0] == &v256[1] )
        {
          v420 = _mm_loadu_si128(v256 + 1);
        }
        else
        {
          s2.m128i_i64[0] = v256->m128i_i64[0];
          v420.m128i_i64[0] = v256[1].m128i_i64[0];
        }
        s2.m128i_i64[1] = v256->m128i_i64[1];
        v256->m128i_i64[0] = (__int64)v256[1].m128i_i64;
        v256->m128i_i64[1] = 0;
        v256[1].m128i_i8[0] = 0;
        if ( (__m128i *)v415.m128i_i64[0] != &v416 )
          j_j___libc_free_0(v415.m128i_i64[0], v416.m128i_i64[0] + 1);
        if ( (_QWORD *)v413.m128i_i64[0] != v414 )
          j_j___libc_free_0(v413.m128i_i64[0], v414[0] + 1LL);
        goto LABEL_486;
      }
LABEL_734:
      sub_4262D8((__int64)"basic_string::append");
    }
    break;
  }
  v32 = (__int64)srcb;
  v189 = 8;
  v190 = "-maxreg=";
  do
  {
    if ( !v189 )
      break;
    v97 = *(_BYTE *)v32 < *v190;
    v98 = *(_BYTE *)v32++ == *v190++;
    --v189;
  }
  while ( v98 );
  if ( (!v97 && !v98) == v97 )
  {
    if ( !*(_QWORD *)(a1 + 1208) )
    {
      v191 = strlen(srcb + 8);
      sub_2241130(a1 + 1200, 0, 0, srcb + 8, v191);
      sub_8FD6D0((__int64)p_s2, "-maxreg=", (_QWORD *)(a1 + 1200));
      sub_8F9C20(&v398, p_s2);
      if ( (__m128i *)s2.m128i_i64[0] != v80 )
        j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
      sub_8FD6D0((__int64)p_s2, "-maxreg=", (_QWORD *)(a1 + 1200));
      sub_8F9C20(&v404, p_s2);
      v118 = s2.m128i_i64[0];
      if ( (__m128i *)s2.m128i_i64[0] != v80 )
        goto LABEL_201;
      goto LABEL_202;
    }
    if ( !a14 )
      goto LABEL_488;
    sub_12C65C0(p_s2->m128i_i64, "libnvvm : error: -maxreg defined more than once");
LABEL_486:
    v257 = s2.m128i_i64[1];
    v32 = sub_2207820(s2.m128i_i64[1] + 1);
    *a14 = v32;
    sub_2241570(p_s2, v32, v257, 0);
    *(_BYTE *)(*a14 + v257) = 0;
    if ( (__m128i *)s2.m128i_i64[0] != v80 )
    {
      v32 = v420.m128i_i64[0] + 1;
      j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
    }
    goto LABEL_488;
  }
  if ( !strcmp(srcb, "-Xopt") )
  {
    ++v372;
    v195 = v371[v357 + 1];
    v407 = (char *)v409;
    if ( !v195 )
      goto LABEL_57;
    v196 = strlen(v195);
    s2.m128i_i64[0] = v196;
    v197 = v196;
    if ( v196 > 0xF )
    {
      v389 = v196;
      v217 = (char *)sub_22409D0(&v407, p_s2, 0);
      v197 = v389;
      v407 = v217;
      v218 = v217;
      v409[0] = s2.m128i_i64[0];
    }
    else
    {
      if ( v196 == 1 )
      {
        LOBYTE(v409[0]) = *v195;
        v198 = (char *)v409;
        goto LABEL_372;
      }
      if ( !v196 )
      {
        v198 = (char *)v409;
        goto LABEL_372;
      }
      v218 = (char *)v409;
    }
    memcpy(v218, v195, v197);
    v198 = v407;
    v196 = s2.m128i_i64[0];
LABEL_372:
    v408 = v196;
    v198[v196] = 0;
    if ( memcmp(v407, "-opt-discard-value-names=", 0x19u) )
    {
LABEL_373:
      v199 = v399;
      if ( v399 == v400 )
      {
        sub_8FD760(&v398, v399, (__int64)&v407);
      }
      else
      {
        if ( v399 )
        {
          v200 = v399;
          v399->m128i_i64[0] = (__int64)v399[1].m128i_i64;
          sub_12C6440(v200->m128i_i64, v407, (__int64)&v407[v408]);
          v199 = v399;
        }
        v399 = (__m128i *)&v199[2];
      }
      if ( v407 != (char *)v409 )
        j_j___libc_free_0(v407, v409[0] + 1LL);
      goto LABEL_202;
    }
    v360 = v407;
    v388 = v407 + 25;
    s2.m128i_i64[0] = (__int64)v80;
    v213 = strlen(v407 + 25);
    v214 = v388;
    s1.m128i_i64[0] = v213;
    v215 = v213;
    if ( v213 > 0xF )
    {
      v364 = v213;
      v260 = sub_22409D0(p_s2, &s1, 0);
      v214 = v388;
      v215 = v364;
      s2.m128i_i64[0] = v260;
      v261 = (void *)v260;
      v420.m128i_i64[0] = s1.m128i_i64[0];
    }
    else
    {
      if ( v213 == 1 )
      {
        v420.m128i_i8[0] = v360[25];
        v216 = (__int64)v80;
LABEL_402:
        s2.m128i_i64[1] = v213;
        *(_BYTE *)(v216 + v213) = 0;
        v344 = (unsigned int)sub_2241AC0(p_s2, "1") == 0;
        if ( !v351 )
          v351 = 1;
        if ( (__m128i *)s2.m128i_i64[0] != v80 )
          j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
        goto LABEL_373;
      }
      if ( !v213 )
      {
        v216 = (__int64)v80;
        goto LABEL_402;
      }
      v261 = v80;
    }
    memcpy(v261, v214, v215);
    v213 = s1.m128i_i64[0];
    v216 = s2.m128i_i64[0];
    goto LABEL_402;
  }
  if ( !strcmp(srcb, "-Xllc") )
  {
    ++v372;
    v201 = v371[v357 + 1];
    v415.m128i_i64[0] = (__int64)&v416;
    if ( !v201 )
      goto LABEL_57;
    v386 = v201;
    v202 = strlen(v201);
    v203 = v386;
    s2.m128i_i64[0] = v202;
    v204 = v202;
    if ( v202 > 0xF )
    {
      v361 = v202;
      v226 = sub_22409D0(&v415, p_s2, 0);
      v203 = v386;
      v204 = v361;
      v415.m128i_i64[0] = v226;
      v227 = (__m128i *)v226;
      v416.m128i_i64[0] = s2.m128i_i64[0];
    }
    else
    {
      if ( v202 == 1 )
      {
        v416.m128i_i8[0] = *v386;
        v205 = &v416;
        goto LABEL_386;
      }
      if ( !v202 )
      {
        v205 = &v416;
        goto LABEL_386;
      }
      v227 = &v416;
    }
    memcpy(v227, v203, v204);
    v202 = s2.m128i_i64[0];
    v205 = (__m128i *)v415.m128i_i64[0];
LABEL_386:
    v415.m128i_i64[1] = v202;
    v205->m128i_i8[v202] = 0;
    sub_8F9C20(&v404, &v415);
    if ( (__m128i *)v415.m128i_i64[0] != &v416 )
      j_j___libc_free_0(v415.m128i_i64[0], v416.m128i_i64[0] + 1);
    goto LABEL_202;
  }
  if ( !strcmp(srcb, "-Xlnk") )
  {
    ++v372;
    v206 = v371[v357 + 1];
    v410 = (char *)v412;
    if ( !v206 )
      goto LABEL_57;
    v387 = v206;
    v207 = strlen(v206);
    v208 = v387;
    s2.m128i_i64[0] = v207;
    v209 = v207;
    if ( v207 > 0xF )
    {
      v362 = v207;
      v247 = (char *)sub_22409D0(&v410, p_s2, 0);
      v208 = v387;
      v209 = v362;
      v410 = v247;
      v248 = v247;
      v412[0] = s2.m128i_i64[0];
    }
    else
    {
      if ( v207 == 1 )
      {
        LOBYTE(v412[0]) = *v387;
        v210 = (char *)v412;
        goto LABEL_392;
      }
      if ( !v207 )
      {
        v210 = (char *)v412;
LABEL_392:
        v411 = v207;
        v210[v207] = 0;
        if ( memcmp(v410, "-lnk-discard-value-names=", 0x19u) )
        {
LABEL_393:
          v211 = v396;
          if ( v396 == v397 )
          {
            sub_8FD760(&v395, v396, (__int64)&v410);
          }
          else
          {
            if ( v396 )
            {
              v212 = v396;
              v396->m128i_i64[0] = (__int64)v396[1].m128i_i64;
              sub_12C6440(v212->m128i_i64, v410, (__int64)&v410[v411]);
              v211 = v396;
            }
            v396 = (__m128i *)&v211[2];
          }
          if ( v410 != (char *)v412 )
            j_j___libc_free_0(v410, v412[0] + 1LL);
          goto LABEL_202;
        }
        v363 = v410;
        v391 = v410 + 25;
        s2.m128i_i64[0] = (__int64)v80;
        v249 = strlen(v410 + 25);
        v250 = v391;
        s1.m128i_i64[0] = v249;
        v251 = v249;
        if ( v249 > 0xF )
        {
          v367 = v249;
          v303 = sub_22409D0(p_s2, &s1, 0);
          v250 = v391;
          v251 = v367;
          s2.m128i_i64[0] = v303;
          v304 = (void *)v303;
          v420.m128i_i64[0] = s1.m128i_i64[0];
        }
        else
        {
          if ( v249 == 1 )
          {
            v420.m128i_i8[0] = v363[25];
            v252 = (__int64)v80;
LABEL_468:
            s2.m128i_i64[1] = v249;
            *(_BYTE *)(v252 + v249) = 0;
            v346 = (unsigned int)sub_2241AC0(p_s2, "1") == 0;
            if ( !v348 )
              v348 = 1;
            if ( (__m128i *)s2.m128i_i64[0] != v80 )
              j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
            goto LABEL_393;
          }
          if ( !v249 )
          {
            v252 = (__int64)v80;
            goto LABEL_468;
          }
          v304 = v80;
        }
        memcpy(v304, v250, v251);
        v249 = s1.m128i_i64[0];
        v252 = s2.m128i_i64[0];
        goto LABEL_468;
      }
      v248 = (char *)v412;
    }
    memcpy(v248, v208, v209);
    v207 = s2.m128i_i64[0];
    v210 = v410;
    goto LABEL_392;
  }
  if ( !strcmp(srcb, "-Xlto") )
  {
    ++v372;
    v219 = v371[v357 + 1];
    v413.m128i_i64[0] = (__int64)v414;
    if ( !v219 )
      goto LABEL_57;
    v390 = v219;
    v220 = strlen(v219);
    v221 = v390;
    s2.m128i_i64[0] = v220;
    v222 = v220;
    if ( v220 > 0xF )
    {
      v366 = v220;
      v281 = sub_22409D0(&v413, p_s2, 0);
      v221 = v390;
      v222 = v366;
      v413.m128i_i64[0] = v281;
      v282 = (_QWORD *)v281;
      v414[0] = s2.m128i_i64[0];
    }
    else
    {
      if ( v220 == 1 )
      {
        LOBYTE(v414[0]) = *v390;
        v223 = v414;
        goto LABEL_414;
      }
      if ( !v220 )
      {
        v223 = v414;
        goto LABEL_414;
      }
      v282 = v414;
    }
    memcpy(v282, v221, v222);
    v220 = s2.m128i_i64[0];
    v223 = (_QWORD *)v413.m128i_i64[0];
LABEL_414:
    v413.m128i_i64[1] = v220;
    *((_BYTE *)v223 + v220) = 0;
    if ( memcmp((const void *)v413.m128i_i64[0], "-lto-discard-value-names=", 0x19u) )
    {
LABEL_415:
      v224 = v402;
      if ( v402 == v403 )
      {
        sub_8FD760(&v401, v402, (__int64)&v413);
      }
      else
      {
        if ( v402 )
        {
          v225 = v402;
          v402->m128i_i64[0] = (__int64)v402[1].m128i_i64;
          sub_12C6440(v225->m128i_i64, v413.m128i_i64[0], v413.m128i_i64[0] + v413.m128i_i64[1]);
          v224 = v402;
        }
        v402 = (__m128i *)&v224[2];
      }
      if ( (_QWORD *)v413.m128i_i64[0] != v414 )
        j_j___libc_free_0(v413.m128i_i64[0], v414[0] + 1LL);
      goto LABEL_202;
    }
    v365 = v413.m128i_i64[0];
    v392 = v413.m128i_i64[0] + 25;
    s2.m128i_i64[0] = (__int64)v80;
    v274 = strlen((const char *)(v413.m128i_i64[0] + 25));
    v275 = (const void *)v392;
    s1.m128i_i64[0] = v274;
    v276 = v274;
    if ( v274 > 0xF )
    {
      v368 = v274;
      v329 = sub_22409D0(p_s2, &s1, 0);
      v275 = (const void *)v392;
      v276 = v368;
      s2.m128i_i64[0] = v329;
      v330 = (void *)v329;
      v420.m128i_i64[0] = s1.m128i_i64[0];
    }
    else
    {
      if ( v274 == 1 )
      {
        v420.m128i_i8[0] = *(_BYTE *)(v365 + 25);
        v277 = (__int64)v80;
LABEL_533:
        s2.m128i_i64[1] = v274;
        *(_BYTE *)(v277 + v274) = 0;
        v349 = (unsigned int)sub_2241AC0(p_s2, "1") == 0;
        if ( !v350 )
          v350 = 1;
        if ( (__m128i *)s2.m128i_i64[0] != v80 )
          j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
        goto LABEL_415;
      }
      if ( !v274 )
      {
        v277 = (__int64)v80;
        goto LABEL_533;
      }
      v330 = v80;
    }
    memcpy(v330, v275, v276);
    v274 = s1.m128i_i64[0];
    v277 = s2.m128i_i64[0];
    goto LABEL_533;
  }
  if ( !strcmp(srcb, "-cl-mode") )
    goto LABEL_202;
  if ( !strcmp(srcb, "--device-c") )
  {
    s2.m128i_i64[0] = (__int64)v80;
    qmemcpy(v80, "--device-c", 10);
    s2.m128i_i64[1] = 10;
    v420.m128i_i8[10] = 0;
    goto LABEL_504;
  }
  if ( !strcmp(srcb, "--force-device-c") )
  {
    s2.m128i_i64[0] = (__int64)v80;
    s1.m128i_i64[0] = 16;
    s2.m128i_i64[0] = sub_22409D0(p_s2, &s1, 0);
    v420.m128i_i64[0] = s1.m128i_i64[0];
    *(__m128i *)s2.m128i_i64[0] = _mm_load_si128((const __m128i *)&xmmword_3F15860);
    s2.m128i_i64[1] = s1.m128i_i64[0];
    *(_BYTE *)(s2.m128i_i64[0] + s1.m128i_i64[0]) = 0;
LABEL_504:
    sub_8F9C20(&v401, p_s2);
    v118 = s2.m128i_i64[0];
    if ( (__m128i *)s2.m128i_i64[0] != v80 )
      goto LABEL_201;
    goto LABEL_202;
  }
  if ( !memcmp(srcb, "-host-ref-ek=", 0xDu) )
  {
    sub_12C65C0(s1.m128i_i64, srcb + 13);
    v289 = "-host-ref-ek=";
    goto LABEL_576;
  }
  v193 = "-host-ref-ik=";
  if ( !memcmp(srcb, "-host-ref-ik=", 0xDu)
    || (v193 = "-host-ref-ec=", !memcmp(srcb, "-host-ref-ec=", 0xDu))
    || (v193 = "-host-ref-ic=", !memcmp(srcb, "-host-ref-ic=", 0xDu))
    || (v193 = "-host-ref-eg=", !memcmp(srcb, "-host-ref-eg=", 0xDu)) )
  {
    sub_12C65C0(s1.m128i_i64, srcb + 13);
    v289 = v193;
LABEL_576:
    sub_8FD6D0((__int64)p_s2, v289, &s1);
    sub_8F9C20(&v401, p_s2);
    if ( (__m128i *)s2.m128i_i64[0] != v80 )
      j_j___libc_free_0(s2.m128i_i64[0], v420.m128i_i64[0] + 1);
    if ( (_QWORD *)s1.m128i_i64[0] != v418 )
      j_j___libc_free_0(s1.m128i_i64[0], v418[0] + 1LL);
    goto LABEL_202;
  }
  if ( !memcmp(srcb, "-host-ref-ig=", 0xDu) )
  {
    sub_12C65C0(s1.m128i_i64, srcb + 13);
    sub_8FD6D0((__int64)p_s2, "-host-ref-ig=", &s1);
    v337 = &v401;
    v338 = p_s2;
LABEL_698:
    sub_8F9C20(v337, v338);
    sub_2240A30(p_s2);
    sub_2240A30(&s1);
    goto LABEL_202;
  }
  v194 = "-has-global-host-info";
  if ( !strcmp(srcb, "-has-global-host-info")
    || (v194 = "-optimize-unused-variables", !strcmp(srcb, "-optimize-unused-variables")) )
  {
    v335 = v194;
    goto LABEL_688;
  }
  if ( !strcmp(srcb, "--partial-link") )
    goto LABEL_202;
  if ( !strcmp(srcb, "-lto") )
  {
    *a13 = *a13 & 0x300 | 0x23;
    goto LABEL_202;
  }
  if ( !strcmp(srcb, "-olto") )
  {
    sub_12C65C0(p_s2->m128i_i64, "-olto");
    sub_8F9C20(&v401, p_s2);
    sub_2240A30(p_s2);
    ++v372;
    sub_12C65C0(p_s2->m128i_i64, v371[v357 + 1]);
    sub_8F9C20(&v401, p_s2);
    sub_2240A30(p_s2);
    goto LABEL_202;
  }
  if ( !strcmp(srcb, "-gen-lto") )
  {
    *a13 = *a13 & 0x300 | 0x21;
    goto LABEL_712;
  }
  if ( !strcmp(srcb, "-gen-lto-and-llc") )
  {
    *a13 |= 0x20u;
LABEL_712:
    v335 = "-gen-lto";
    goto LABEL_688;
  }
  if ( !strcmp(srcb, "-link-lto") )
  {
    v335 = "-link-lto";
    *a13 = *a13 & 0x300 | 0x26;
    goto LABEL_688;
  }
  if ( !strcmp(srcb, "-gen-opt-lto") )
  {
    v345 = 1;
    goto LABEL_202;
  }
  v335 = "--trace";
  if ( !strcmp(srcb, "--trace-lto") )
  {
LABEL_688:
    sub_12C65C0(p_s2->m128i_i64, v335);
    sub_8F9C20(&v401, p_s2);
    sub_2240A30(p_s2);
    goto LABEL_202;
  }
  if ( !strcmp(srcb, "-inline-info") )
  {
    sub_12C65C0(p_s2->m128i_i64, "-pass-remarks=inline");
    sub_8F9C20(&v398, p_s2);
    sub_2240A30(p_s2);
    sub_12C65C0(p_s2->m128i_i64, "-pass-remarks-missed=inline");
    sub_8F9C20(&v398, p_s2);
    sub_2240A30(p_s2);
    sub_12C65C0(p_s2->m128i_i64, "-pass-remarks-analysis=inline");
    sub_8F9C20(&v398, p_s2);
    sub_2240A30(p_s2);
    goto LABEL_202;
  }
  if ( (a4 == 57069 || a4 == 43962) && !strcmp(srcb, "--emit-optix-ir") )
  {
    sub_12C65C0(p_s2->m128i_i64, "-do-ip-msp=0");
    sub_8F9C20(&v398, p_s2);
    sub_2240A30(p_s2);
    sub_12C65C0(p_s2->m128i_i64, "-do-licm=0");
    sub_8F9C20(&v398, p_s2);
    sub_2240A30(p_s2);
    *a13 = *a13 & 0x300 | 0x43;
    goto LABEL_202;
  }
  if ( !strncmp(srcb, "-split-compile=", 0xFu) )
  {
    if ( *(_QWORD *)(a1 + 1496) )
      goto LABEL_692;
    sub_12C6200(a1 + 1488, srcb + 15);
    sub_8FD6D0((__int64)p_s2, "-split-compile=", (_QWORD *)(a1 + 1488));
    goto LABEL_686;
  }
  if ( !strncmp(srcb, "-split-compile-extended=", 0x18u) )
  {
    if ( *(_QWORD *)(a1 + 1496) )
    {
LABEL_692:
      v32 = (__int64)"libnvvm : error: split compilation defined more than once";
      if ( a14 )
        goto LABEL_693;
      goto LABEL_488;
    }
    sub_12C6200(a1 + 1488, srcb + 24);
    sub_8FD6D0((__int64)p_s2, "-split-compile-extended=", (_QWORD *)(a1 + 1488));
    sub_8F9C20(&v398, p_s2);
    sub_2240A30(p_s2);
    *(_BYTE *)(a1 + 1652) = 1;
    goto LABEL_202;
  }
  if ( !strncmp(srcb, "-Ofast-compile=", 0xFu) )
    goto LABEL_202;
  if ( !strncmp(srcb, "-jump-table-density=", 0x14u) )
  {
    sub_12C65C0(s1.m128i_i64, srcb + 20);
    sub_8FD6D0((__int64)p_s2, "-jump-table-density=", &s1);
    v337 = &v404;
    v338 = p_s2;
    goto LABEL_698;
  }
  v32 = (__int64)"-discard-value-names=";
  if ( strncmp(srcb, "-discard-value-names=", 0x15u) )
  {
    v340 = v394;
    if ( v394 > 0xB )
    {
      v341 = (char *)v393;
      if ( !memcmp(v393, "-opt-passes=", 0xCu) )
      {
        v393 = v341 + 12;
        v394 = v340 - 12;
        sub_12C70A0(p_s2->m128i_i64, (__int64)&v393);
        sub_12C6360(a1 + 1520, (__int64)p_s2);
        sub_2240A30(p_s2);
        goto LABEL_202;
      }
    }
    sub_12C6330(p_s2->m128i_i64, byte_42819B8, 14);
    if ( s2.m128i_i64[1] <= v394 && (!s2.m128i_i64[1] || !memcmp(v393, (const void *)s2.m128i_i64[0], s2.m128i_u64[1])) )
    {
      sub_2240A30(p_s2);
      v32 = (__int64)a14;
      if ( (unsigned int)sub_12C6910(srcb, a14, a13) != 1 )
        goto LABEL_488;
      goto LABEL_202;
    }
    sub_2240A30(p_s2);
    v32 = (__int64)"-jobserver";
    if ( strncmp(srcb, "-jobserver", 0xAu) )
    {
      if ( a14 )
      {
        sub_12C65C0(s1.m128i_i64, srcb);
        sub_95D570(p_s2, "libnvvm : error: ", (__int64)&s1);
        sub_94F930(&v415, (__int64)p_s2, " is an unsupported option");
        sub_2240A30(p_s2);
        sub_2240A30(&s1);
        v342 = v415.m128i_i64[1];
        v32 = sub_2207820(v415.m128i_i64[1] + 1);
        *a14 = v32;
        sub_2241570(&v415, v32, v342, 0);
        *(_BYTE *)(*a14 + v342) = 0;
        sub_2240A30(&v415);
      }
      goto LABEL_488;
    }
    sub_12C65C0(p_s2->m128i_i64, "-jobserver");
LABEL_686:
    sub_8F9C20(&v398, p_s2);
    sub_2240A30(p_s2);
    goto LABEL_202;
  }
  if ( !v343 && !v351 && !v348 && !v350 )
  {
    sub_12C65C0(s1.m128i_i64, srcb + 21);
    if ( !strcmp((const char *)s1.m128i_i64[0], "1") )
    {
      *(_BYTE *)(a1 + 240) = 1;
      sub_12C65C0(p_s2->m128i_i64, "-lnk-discard-value-names=1");
      sub_8F9C20(&v395, p_s2);
      sub_2240A30(p_s2);
      sub_12C65C0(p_s2->m128i_i64, "-opt-discard-value-names=1");
      sub_8F9C20(&v398, p_s2);
      sub_2240A30(p_s2);
      v339 = "-lto-discard-value-names=1";
    }
    else
    {
      v339 = "-lto-discard-value-names=0";
      *(_BYTE *)(a1 + 240) = 0;
    }
    sub_12C65C0(p_s2->m128i_i64, v339);
    sub_8F9C20(&v401, p_s2);
    sub_2240A30(p_s2);
    sub_2240A30(&s1);
    v343 = 1;
    goto LABEL_202;
  }
  if ( a14 )
  {
    v32 = (__int64)"libnvvm : error: -discard-value-names defined more than once, or defined for both libnvvm and sub-phase";
LABEL_693:
    sub_12C65C0(p_s2->m128i_i64, (const char *)v32);
    v336 = s2.m128i_i64[1];
    v32 = sub_2207820(s2.m128i_i64[1] + 1);
    *a14 = v32;
    sub_2241570(p_s2, v32, v336, 0);
    *(_BYTE *)(*a14 + v336) = 0;
    sub_2240A30(p_s2);
  }
LABEL_488:
  j_j___libc_free_0_0(srcb);
LABEL_90:
  v16 = 1;
LABEL_91:
  v68 = v436;
  v69 = &v436[2 * (unsigned int)v437];
  if ( v436 != (_OWORD *)v69 )
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
    v69 = v436;
  }
  if ( v69 != (_QWORD *)v438 )
    _libc_free(v69, v32);
  v70 = v405;
  v71 = v404;
  if ( v405 != v404 )
  {
    do
    {
      if ( (__m128i *)v71->m128i_i64[0] != &v71[1] )
        j_j___libc_free_0(v71->m128i_i64[0], v71[1].m128i_i64[0] + 1);
      v71 += 2;
    }
    while ( v70 != v71 );
    v71 = v404;
  }
  if ( v71 )
    j_j___libc_free_0(v71, v406 - (_QWORD)v71);
  v72 = v402;
  v73 = v401;
  if ( v402 != v401 )
  {
    do
    {
      if ( (__m128i *)v73->m128i_i64[0] != &v73[1] )
        j_j___libc_free_0(v73->m128i_i64[0], v73[1].m128i_i64[0] + 1);
      v73 += 2;
    }
    while ( v72 != v73 );
    v73 = v401;
  }
  if ( v73 )
    j_j___libc_free_0(v73, (char *)v403 - (char *)v73);
  v74 = v399;
  v75 = v398;
  if ( v399 != v398 )
  {
    do
    {
      if ( (__m128i *)v75->m128i_i64[0] != &v75[1] )
        j_j___libc_free_0(v75->m128i_i64[0], v75[1].m128i_i64[0] + 1);
      v75 += 2;
    }
    while ( v74 != v75 );
    v75 = v398;
  }
  if ( v75 )
    j_j___libc_free_0(v75, (char *)v400 - (char *)v75);
  v76 = v396;
  v77 = v395;
  if ( v396 != v395 )
  {
    do
    {
      if ( (__m128i *)v77->m128i_i64[0] != &v77[1] )
        j_j___libc_free_0(v77->m128i_i64[0], v77[1].m128i_i64[0] + 1);
      v77 += 2;
    }
    while ( v76 != v77 );
    v77 = v395;
  }
  if ( v77 )
    j_j___libc_free_0(v77, (char *)v397 - (char *)v77);
  return v16;
}
