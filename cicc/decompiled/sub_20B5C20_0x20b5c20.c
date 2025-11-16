// Function: sub_20B5C20
// Address: 0x20b5c20
//
_BOOL8 __fastcall sub_20B5C20(
        __int64 a1,
        int a2,
        __int64 a3,
        const void **a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9,
        __int128 a10,
        __int128 a11,
        unsigned int a12,
        const void **a13,
        __int64 *a14,
        int a15,
        __int128 a16,
        __int128 a17,
        __int128 a18,
        __int128 a19)
{
  __int64 *v19; // r12
  unsigned int v20; // ecx
  __int64 v21; // rax
  bool v22; // r8
  char v24; // bl
  const void **v25; // rdx
  unsigned int v26; // eax
  char v27; // r13
  unsigned int v28; // ebx
  unsigned int v29; // r13d
  __int64 v30; // rdx
  unsigned int v31; // edx
  unsigned int v32; // esi
  __int64 v33; // r9
  unsigned int v34; // r14d
  bool v35; // r14
  unsigned int v36; // ebx
  __int64 v37; // rax
  const void **v38; // rdx
  unsigned int v39; // eax
  unsigned __int64 v40; // r13
  __int64 v41; // r9
  __int64 v42; // rdx
  int v43; // ebx
  unsigned int v44; // eax
  char v45; // al
  const void **v46; // rdx
  unsigned int v47; // edx
  __int64 v48; // rdx
  __int64 v49; // rdx
  __int8 v50; // r10
  __int64 *v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rbx
  __int64 *v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rbx
  __int64 v57; // r13
  __int64 *v58; // rax
  int v59; // r8d
  int v60; // r9d
  __int8 v61; // r10
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 *v64; // rax
  __int64 v65; // rax
  __m128i v66; // xmm1
  __int64 v67; // rax
  int v68; // r9d
  __int8 v69; // r10
  __int64 v70; // r12
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // r13
  int v74; // r8d
  __int64 *v75; // rax
  __int64 v76; // rdx
  __int64 *v77; // rdx
  __int64 *v78; // rax
  __int64 v79; // rdx
  __int64 v80; // rbx
  __int64 *v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rbx
  __int64 v84; // r13
  int v85; // r8d
  int v86; // r9d
  __int64 v87; // rdx
  __int64 v88; // rax
  __int64 *v89; // rax
  __int64 v90; // r9
  unsigned __int64 v91; // rdx
  __int64 *v92; // rax
  __int64 v93; // rdx
  __int64 v94; // rbx
  __int64 *v95; // rdx
  __int64 v96; // rax
  __int64 v97; // rbx
  unsigned __int64 v98; // r13
  __int64 v99; // rdx
  __int64 *v100; // rax
  __int64 v101; // rdx
  __int64 v102; // rbx
  __int64 v103; // rdx
  __int64 v104; // rax
  __int64 *v105; // rax
  __int64 v106; // rdx
  __int64 v107; // rbx
  __int64 *v108; // rdx
  __int64 v109; // rax
  __int64 v110; // rbx
  __int64 v111; // r13
  int v112; // r8d
  int v113; // r9d
  __int64 v114; // rdx
  __int64 v115; // rax
  __int64 *v116; // rax
  __int64 v117; // rax
  __int64 *v118; // rax
  __int64 v119; // rdx
  __int64 v120; // rbx
  __int64 v121; // rdx
  __int64 v122; // rax
  __int128 v123; // rax
  __int64 v124; // r9
  __int64 v125; // rdx
  __int64 *v126; // rax
  __int64 v127; // rdx
  __int64 v128; // rbx
  __int64 *v129; // rdx
  __int64 v130; // rax
  __int64 v131; // rbx
  unsigned __int64 v132; // r13
  __int64 v133; // rdx
  __int128 v134; // rax
  const void ***v135; // rax
  int v136; // edx
  __int64 v137; // r9
  __int64 *v138; // rax
  __int64 v139; // rdx
  __int64 v140; // rax
  __int64 v141; // r8
  __int64 v142; // rdx
  __int64 v143; // r9
  __int64 v144; // rdx
  __int64 *v145; // rdx
  __int64 v146; // r9
  __int64 v147; // rdx
  __int64 *v148; // rax
  __int64 v149; // rdx
  __int64 v150; // rbx
  __int64 *v151; // rdx
  __int64 v152; // rax
  __int64 v153; // rbx
  unsigned __int64 v154; // r13
  __int64 v155; // rdx
  __int128 v156; // rax
  const void ***v157; // rax
  int v158; // edx
  __int64 v159; // r9
  __int64 *v160; // rax
  __int64 v161; // rdx
  __int128 v162; // rax
  __int64 *v163; // rax
  __int64 v164; // rdx
  int v165; // r8d
  int v166; // r9d
  __int64 v167; // r14
  __int64 v168; // rdx
  __int64 v169; // r15
  __int64 v170; // rdx
  __int64 *v171; // rdx
  __int64 v172; // rcx
  __int64 v173; // rdx
  int v174; // r8d
  int v175; // r9d
  __int64 v176; // r12
  __int64 v177; // rdx
  __int64 v178; // r13
  __int64 v179; // rdx
  __int64 *v180; // rdx
  unsigned int v181; // edx
  unsigned int v182; // edx
  __int64 v183; // rdx
  __int64 v184; // rdx
  __int64 v185; // rdx
  __int64 v186; // rdx
  __int64 *v187; // rax
  __int64 v188; // rdx
  __int64 v189; // rbx
  __int64 v190; // rdx
  __int64 v191; // rax
  __int64 *v192; // rax
  __int64 v193; // rdx
  __int64 v194; // rbx
  __int64 v195; // rdx
  __int64 v196; // rax
  __int64 v197; // rdx
  __int64 v198; // rdx
  __int64 v199; // rdx
  int v200; // r8d
  int v201; // r9d
  __int64 v202; // rdx
  __int64 v203; // rax
  __int64 *v204; // rax
  __int64 v205; // rdx
  __int64 v206; // rbx
  __int64 v207; // rdx
  __int64 v208; // rax
  __int64 v209; // rsi
  __int64 *v210; // rax
  __int64 v211; // rdx
  __int64 v212; // rbx
  __int64 v213; // rdx
  __int64 v214; // rax
  __int128 v215; // rax
  __m128i v216; // rax
  __int64 v217; // rbx
  __int64 v218; // r13
  __int64 v219; // rcx
  __int64 v220; // r8
  __int64 v221; // r9
  __int128 v222; // rax
  __int64 v223; // r9
  __int64 v224; // rdx
  __int128 v225; // rax
  __int64 v226; // rdx
  __int64 v227; // rcx
  __int64 v228; // r8
  __int64 v229; // r9
  __int128 v230; // rax
  const void ***v231; // rsi
  __int64 v232; // r9
  __int64 v233; // rdx
  __int128 v234; // [rsp-20h] [rbp-3A0h]
  __int128 v235; // [rsp-20h] [rbp-3A0h]
  __int128 v236; // [rsp-10h] [rbp-390h]
  __int128 v237; // [rsp-10h] [rbp-390h]
  __int128 v238; // [rsp-10h] [rbp-390h]
  int v239; // [rsp-8h] [rbp-388h]
  unsigned int v240; // [rsp+0h] [rbp-380h]
  unsigned int v241; // [rsp+4h] [rbp-37Ch]
  __int64 v242; // [rsp+8h] [rbp-378h]
  bool v243; // [rsp+11h] [rbp-36Fh]
  bool v244; // [rsp+12h] [rbp-36Eh]
  char v245; // [rsp+13h] [rbp-36Dh]
  __int64 v247; // [rsp+18h] [rbp-368h]
  __int64 v248; // [rsp+20h] [rbp-360h]
  __int64 v249; // [rsp+20h] [rbp-360h]
  __int64 v250; // [rsp+28h] [rbp-358h]
  __int64 v251; // [rsp+30h] [rbp-350h]
  __int128 v252; // [rsp+30h] [rbp-350h]
  __int64 *v253; // [rsp+30h] [rbp-350h]
  __int64 v256; // [rsp+50h] [rbp-330h]
  __int64 v257; // [rsp+50h] [rbp-330h]
  __int128 v258; // [rsp+50h] [rbp-330h]
  unsigned __int64 v259; // [rsp+58h] [rbp-328h]
  unsigned __int64 v260; // [rsp+58h] [rbp-328h]
  unsigned __int64 v261; // [rsp+58h] [rbp-328h]
  unsigned __int64 v262; // [rsp+58h] [rbp-328h]
  bool v263; // [rsp+60h] [rbp-320h]
  __int8 v264; // [rsp+60h] [rbp-320h]
  __int128 v265; // [rsp+60h] [rbp-320h]
  __int8 v266; // [rsp+60h] [rbp-320h]
  __int8 v267; // [rsp+60h] [rbp-320h]
  __int8 v268; // [rsp+60h] [rbp-320h]
  __m128i v269; // [rsp+70h] [rbp-310h] BYREF
  __int64 *v270; // [rsp+80h] [rbp-300h]
  __int64 v271; // [rsp+88h] [rbp-2F8h]
  __int64 *v272; // [rsp+90h] [rbp-2F0h]
  __int64 v273; // [rsp+98h] [rbp-2E8h]
  __int64 *v274; // [rsp+A0h] [rbp-2E0h]
  __int64 v275; // [rsp+A8h] [rbp-2D8h]
  __int64 *v276; // [rsp+B0h] [rbp-2D0h]
  __int64 v277; // [rsp+B8h] [rbp-2C8h]
  __int64 *v278; // [rsp+C0h] [rbp-2C0h]
  __int64 v279; // [rsp+C8h] [rbp-2B8h]
  __int64 *v280; // [rsp+D0h] [rbp-2B0h]
  __int64 v281; // [rsp+D8h] [rbp-2A8h]
  __int64 *v282; // [rsp+E0h] [rbp-2A0h]
  __int64 v283; // [rsp+E8h] [rbp-298h]
  __int64 *v284; // [rsp+F0h] [rbp-290h]
  __int64 v285; // [rsp+F8h] [rbp-288h]
  __int64 v286; // [rsp+100h] [rbp-280h]
  __int64 v287; // [rsp+108h] [rbp-278h]
  __int64 *v288; // [rsp+110h] [rbp-270h]
  __int64 v289; // [rsp+118h] [rbp-268h]
  __int64 *v290; // [rsp+120h] [rbp-260h]
  __int64 v291; // [rsp+128h] [rbp-258h]
  __int64 *v292; // [rsp+130h] [rbp-250h]
  __int64 v293; // [rsp+138h] [rbp-248h]
  __int64 *v294; // [rsp+140h] [rbp-240h]
  __int64 v295; // [rsp+148h] [rbp-238h]
  __int64 v296; // [rsp+150h] [rbp-230h]
  __int64 v297; // [rsp+158h] [rbp-228h]
  __int64 *v298; // [rsp+160h] [rbp-220h]
  __int64 v299; // [rsp+168h] [rbp-218h]
  __int64 *v300; // [rsp+170h] [rbp-210h]
  __int64 v301; // [rsp+178h] [rbp-208h]
  __int64 *v302; // [rsp+180h] [rbp-200h]
  __int64 v303; // [rsp+188h] [rbp-1F8h]
  __int64 v304; // [rsp+190h] [rbp-1F0h]
  __int64 v305; // [rsp+198h] [rbp-1E8h]
  __int64 *v306; // [rsp+1A0h] [rbp-1E0h]
  __int64 v307; // [rsp+1A8h] [rbp-1D8h]
  __int64 *v308; // [rsp+1B0h] [rbp-1D0h]
  __int64 v309; // [rsp+1B8h] [rbp-1C8h]
  __int64 v310; // [rsp+1C0h] [rbp-1C0h]
  __int64 v311; // [rsp+1C8h] [rbp-1B8h]
  __int64 *v312; // [rsp+1D0h] [rbp-1B0h]
  __int64 v313; // [rsp+1D8h] [rbp-1A8h]
  __int64 *v314; // [rsp+1E0h] [rbp-1A0h]
  __int64 v315; // [rsp+1E8h] [rbp-198h]
  __int64 *v316; // [rsp+1F0h] [rbp-190h]
  __int64 v317; // [rsp+1F8h] [rbp-188h]
  __int64 v318; // [rsp+200h] [rbp-180h]
  __int64 v319; // [rsp+208h] [rbp-178h]
  __int64 v320; // [rsp+210h] [rbp-170h]
  __int64 v321; // [rsp+218h] [rbp-168h]
  __int64 v322; // [rsp+220h] [rbp-160h]
  __int64 v323; // [rsp+228h] [rbp-158h]
  __int64 v324; // [rsp+230h] [rbp-150h]
  __int64 v325; // [rsp+238h] [rbp-148h]
  __int64 v326; // [rsp+240h] [rbp-140h]
  __int64 v327; // [rsp+248h] [rbp-138h]
  __int64 *v328; // [rsp+250h] [rbp-130h]
  __int64 v329; // [rsp+258h] [rbp-128h]
  __int64 *v330; // [rsp+260h] [rbp-120h]
  __int64 v331; // [rsp+268h] [rbp-118h]
  __int64 v332; // [rsp+270h] [rbp-110h]
  __int64 v333; // [rsp+278h] [rbp-108h]
  __int64 *v334; // [rsp+280h] [rbp-100h]
  __int64 v335; // [rsp+288h] [rbp-F8h]
  __int64 *v336; // [rsp+290h] [rbp-F0h]
  __int64 v337; // [rsp+298h] [rbp-E8h]
  __int64 v338; // [rsp+2A0h] [rbp-E0h]
  __int64 v339; // [rsp+2A8h] [rbp-D8h]
  __int64 v340; // [rsp+2B0h] [rbp-D0h]
  __int64 v341; // [rsp+2B8h] [rbp-C8h]
  __int64 v342; // [rsp+2C0h] [rbp-C0h]
  __int64 v343; // [rsp+2C8h] [rbp-B8h]
  __int64 v344; // [rsp+2D0h] [rbp-B0h] BYREF
  const void **v345; // [rsp+2D8h] [rbp-A8h]
  char v346; // [rsp+2ECh] [rbp-94h]
  char v347; // [rsp+2EDh] [rbp-93h]
  char v348; // [rsp+2EEh] [rbp-92h]
  char v349; // [rsp+2EFh] [rbp-91h]
  const void ***v350; // [rsp+2F0h] [rbp-90h]
  __int64 v351; // [rsp+2F8h] [rbp-88h]
  __int64 v352; // [rsp+300h] [rbp-80h] BYREF
  unsigned int v353; // [rsp+308h] [rbp-78h]
  unsigned int v354; // [rsp+310h] [rbp-70h] BYREF
  const void **v355; // [rsp+318h] [rbp-68h]
  __int128 v356; // [rsp+320h] [rbp-60h] BYREF
  unsigned __int64 *v357; // [rsp+330h] [rbp-50h] BYREF
  const void **v358; // [rsp+338h] [rbp-48h]
  __int64 *v359; // [rsp+340h] [rbp-40h]
  __int128 *v360; // [rsp+348h] [rbp-38h]

  v251 = a16;
  v19 = a14;
  v247 = a17;
  v344 = a3;
  v248 = a18;
  v345 = a4;
  v242 = a19;
  if ( !a15 )
  {
    v346 = 1;
    v347 = 1;
    v348 = 1;
    v349 = 1;
    v244 = 1;
    v263 = 1;
    v243 = 1;
    v245 = 1;
    goto LABEL_12;
  }
  v20 = 1;
  if ( (_BYTE)a12 != 1 )
  {
    if ( !(_BYTE)a12 )
      return 0;
    v20 = (unsigned __int8)a12;
    if ( !*(_QWORD *)(a1 + 8LL * (unsigned __int8)a12 + 120) )
    {
      v346 = 0;
      v243 = 0;
LABEL_49:
      v347 = 0;
      v263 = 0;
LABEL_50:
      v348 = 0;
      v244 = 0;
LABEL_51:
      v22 = v244 || v263 || v243;
      goto LABEL_8;
    }
  }
  v243 = (*(_BYTE *)(a1 + 259LL * v20 + 2535) & 0xFB) == 0;
  v346 = v243;
  v21 = 1;
  if ( (_BYTE)a12 != 1 )
  {
    if ( !*(_QWORD *)(a1 + 8LL * (int)v20 + 120) )
      goto LABEL_49;
    v21 = v20;
  }
  v263 = (*(_BYTE *)(a1 + 259 * v21 + 2534) & 0xFB) == 0;
  v347 = v263;
  if ( (_BYTE)a12 != 1 )
  {
    if ( !*(_QWORD *)(a1 + 8LL * (int)v20 + 120) )
      goto LABEL_50;
    v21 = v20;
  }
  v244 = (*(_BYTE *)(a1 + 259 * v21 + 2481) & 0xFB) == 0;
  v348 = v244;
  if ( (_BYTE)a12 != 1 )
  {
    if ( !*(_QWORD *)(a1 + 8LL * (int)v20 + 120) )
      goto LABEL_51;
    v21 = v20;
  }
  if ( (*(_BYTE *)(a1 + 259 * v21 + 2482) & 0xFB) == 0 )
  {
    v349 = 1;
    v245 = 1;
    goto LABEL_12;
  }
  v22 = v244 || v263 || v243;
LABEL_8:
  v349 = 0;
  v245 = 0;
  if ( !v22 )
    return 0;
LABEL_12:
  v24 = v344;
  if ( (_BYTE)v344 )
  {
    if ( (unsigned __int8)(v344 - 14) <= 0x5Fu )
    {
      switch ( (char)v344 )
      {
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
        case 32:
        case 62:
        case 63:
        case 64:
        case 65:
        case 66:
        case 67:
          LOBYTE(v357) = 3;
          v358 = 0;
          break;
        case 33:
        case 34:
        case 35:
        case 36:
        case 37:
        case 38:
        case 39:
        case 40:
        case 68:
        case 69:
        case 70:
        case 71:
        case 72:
        case 73:
          LOBYTE(v357) = 4;
          v358 = 0;
          break;
        case 41:
        case 42:
        case 43:
        case 44:
        case 45:
        case 46:
        case 47:
        case 48:
        case 74:
        case 75:
        case 76:
        case 77:
        case 78:
        case 79:
          LOBYTE(v357) = 5;
          v358 = 0;
          break;
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 80:
        case 81:
        case 82:
        case 83:
        case 84:
        case 85:
          LOBYTE(v357) = 6;
          v358 = 0;
          break;
        case 55:
          LOBYTE(v357) = 7;
          v358 = 0;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          LOBYTE(v357) = 8;
          v358 = 0;
          break;
        case 89:
        case 90:
        case 91:
        case 92:
        case 93:
        case 101:
        case 102:
        case 103:
        case 104:
        case 105:
          LOBYTE(v357) = 9;
          v358 = 0;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          LOBYTE(v357) = 10;
          v358 = 0;
          break;
        default:
          LOBYTE(v357) = 2;
          v358 = 0;
          break;
      }
      goto LABEL_59;
    }
    goto LABEL_14;
  }
  if ( !sub_1F58D20((__int64)&v344) )
  {
LABEL_14:
    v25 = v345;
    goto LABEL_15;
  }
  v24 = sub_1F596B0((__int64)&v344);
LABEL_15:
  LOBYTE(v357) = v24;
  v358 = v25;
  if ( !v24 )
  {
    v26 = sub_1F58D40((__int64)&v357);
    v27 = a12;
    v28 = v26;
    if ( (_BYTE)a12 )
      goto LABEL_17;
    goto LABEL_60;
  }
LABEL_59:
  v44 = sub_1F3E310(&v357);
  v27 = a12;
  v28 = v44;
  if ( (_BYTE)a12 )
  {
LABEL_17:
    if ( (unsigned __int8)(v27 - 14) <= 0x5Fu )
    {
      switch ( v27 )
      {
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
        case 32:
        case 62:
        case 63:
        case 64:
        case 65:
        case 66:
        case 67:
          LOBYTE(v357) = 3;
          v358 = 0;
          break;
        case 33:
        case 34:
        case 35:
        case 36:
        case 37:
        case 38:
        case 39:
        case 40:
        case 68:
        case 69:
        case 70:
        case 71:
        case 72:
        case 73:
          LOBYTE(v357) = 4;
          v358 = 0;
          break;
        case 41:
        case 42:
        case 43:
        case 44:
        case 45:
        case 46:
        case 47:
        case 48:
        case 74:
        case 75:
        case 76:
        case 77:
        case 78:
        case 79:
          LOBYTE(v357) = 5;
          v358 = 0;
          break;
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 80:
        case 81:
        case 82:
        case 83:
        case 84:
        case 85:
          LOBYTE(v357) = 6;
          v358 = 0;
          break;
        case 55:
          LOBYTE(v357) = 7;
          v358 = 0;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          LOBYTE(v357) = 8;
          v358 = 0;
          break;
        case 89:
        case 90:
        case 91:
        case 92:
        case 93:
        case 101:
        case 102:
        case 103:
        case 104:
        case 105:
          LOBYTE(v357) = 9;
          v358 = 0;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          LOBYTE(v357) = 10;
          v358 = 0;
          break;
        default:
          LOBYTE(v357) = 2;
          v358 = 0;
          break;
      }
      goto LABEL_62;
    }
    goto LABEL_18;
  }
LABEL_60:
  if ( sub_1F58D20((__int64)&a12) )
  {
    v45 = sub_1F596B0((__int64)&a12);
    v358 = v46;
    LOBYTE(v357) = v45;
    if ( !v45 )
      goto LABEL_19;
LABEL_62:
    v29 = sub_1F3E310(&v357);
    goto LABEL_20;
  }
LABEL_18:
  LOBYTE(v357) = v27;
  v358 = a13;
  if ( v27 )
    goto LABEL_62;
LABEL_19:
  v29 = sub_1F58D40((__int64)&v357);
LABEL_20:
  v241 = sub_1D23330((__int64)v19, a10, *((__int64 *)&a10 + 1), 0);
  v240 = sub_1D23330((__int64)v19, a11, *((__int64 *)&a11 + 1), 0);
  v350 = (const void ***)sub_1D252B0((__int64)v19, a12, (__int64)a13, a12, (__int64)a13);
  v351 = v30;
  v269 = 0u;
  if ( !v251 )
  {
    if ( v248 )
      return 0;
    v47 = 1;
    if ( (_BYTE)a12 != 1 )
    {
      if ( !(_BYTE)a12 )
        return 0;
      v47 = (unsigned __int8)a12;
      if ( !*(_QWORD *)(a1 + 8LL * (unsigned __int8)a12 + 120) )
        return 0;
    }
    if ( (*(_BYTE *)(a1 + 259LL * v47 + 2567) & 0xFB) != 0 )
      return 0;
    v342 = sub_1D309E0(v19, 145, a5, a12, a13, 0, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64, a10);
    v251 = v342;
    *(_QWORD *)&a16 = v342;
    v343 = v48;
    *((_QWORD *)&a16 + 1) = (unsigned int)v48 | *((_QWORD *)&a16 + 1) & 0xFFFFFFFF00000000LL;
    v248 = sub_1D309E0(v19, 145, a5, a12, a13, 0, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64, a11);
    v340 = v248;
    *(_QWORD *)&a18 = v248;
    v341 = v49;
    *((_QWORD *)&a18 + 1) = (unsigned int)v49 | *((_QWORD *)&a18 + 1) & 0xFFFFFFFF00000000LL;
    if ( !v342 )
      return 0;
  }
  v353 = v28;
  if ( v28 > 0x40 )
  {
    sub_16A4EF0((__int64)&v352, 0, 0);
    v31 = v353;
  }
  else
  {
    v352 = 0;
    v31 = v28;
  }
  v32 = v31 - v29;
  if ( v31 - v29 != v31 )
  {
    if ( v32 > 0x3F || v31 > 0x40 )
      sub_16A5260(&v352, v32, v31);
    else
      v352 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v29) << v32;
  }
  if ( (unsigned __int8)sub_1D1F940((__int64)v19, a10, *((__int64 *)&a10 + 1), (__int64)&v352, 0) )
  {
    v50 = sub_1D1F940((__int64)v19, a11, *((__int64 *)&a11 + 1), (__int64)&v352, 0);
    if ( v50 )
    {
      *(_QWORD *)&a16 = v251;
      *(_QWORD *)&a18 = v248;
      if ( v245 )
      {
        v266 = v50;
        v187 = sub_1D37440(v19, 60, a5, v350, v351, v33, *(double *)a7.m128i_i64, a8, a9, a16, a18);
        v61 = v266;
        v189 = v188;
        v190 = (__int64)v187;
        v191 = v189;
        v338 = v190;
        v56 = v190;
        v339 = v191;
        v269.m128i_i64[0] = v190;
        v57 = (unsigned int)v191;
        v269.m128i_i64[1] = v269.m128i_i64[1] & 0xFFFFFFFF00000000LL | 1;
        goto LABEL_78;
      }
      if ( v263 )
      {
        v264 = v50;
        v51 = sub_1D332F0(
                v19,
                54,
                a5,
                a12,
                a13,
                0,
                *(double *)a7.m128i_i64,
                a8,
                a9,
                a16,
                *((unsigned __int64 *)&a16 + 1),
                a18);
        v53 = v52;
        v54 = v51;
        v55 = v53;
        v56 = (__int64)v54;
        v336 = v54;
        v337 = v55;
        v57 = (unsigned int)v55;
        v58 = sub_1D332F0(
                v19,
                112,
                a5,
                a12,
                a13,
                0,
                *(double *)a7.m128i_i64,
                a8,
                a9,
                a16,
                *((unsigned __int64 *)&a16 + 1),
                a18);
        v61 = v264;
        v334 = v58;
        v269.m128i_i64[0] = (__int64)v58;
        v335 = v62;
        v269.m128i_i64[1] = (unsigned int)v62 | v269.m128i_i64[1] & 0xFFFFFFFF00000000LL;
LABEL_78:
        v63 = *(unsigned int *)(a6 + 8);
        if ( (unsigned int)v63 >= *(_DWORD *)(a6 + 12) )
        {
          v268 = v61;
          sub_16CD150(a6, (const void *)(a6 + 16), 0, 16, v59, v60);
          v63 = *(unsigned int *)(a6 + 8);
          v61 = v268;
        }
        v64 = (__int64 *)(*(_QWORD *)a6 + 16 * v63);
        *v64 = v56;
        v64[1] = v57;
        v65 = (unsigned int)(*(_DWORD *)(a6 + 8) + 1);
        *(_DWORD *)(a6 + 8) = v65;
        if ( *(_DWORD *)(a6 + 12) <= (unsigned int)v65 )
        {
          v267 = v61;
          sub_16CD150(a6, (const void *)(a6 + 16), 0, 16, v59, v60);
          v65 = *(unsigned int *)(a6 + 8);
          v61 = v267;
        }
        v66 = _mm_load_si128(&v269);
        v263 = v61;
        v269.m128i_i8[0] = v61;
        *(__m128i *)(*(_QWORD *)a6 + 16 * v65) = v66;
        ++*(_DWORD *)(a6 + 8);
        if ( a2 != 54 )
        {
          v67 = sub_1D38BB0((__int64)v19, 0, a5, a12, a13, 0, a7, *(double *)v66.m128i_i64, a9, 0);
          v69 = v269.m128i_i8[0];
          v70 = v67;
          v71 = *(unsigned int *)(a6 + 8);
          v73 = v72;
          v74 = v239;
          if ( (unsigned int)v71 >= *(_DWORD *)(a6 + 12) )
          {
            sub_16CD150(a6, (const void *)(a6 + 16), 0, 16, v239, v68);
            v71 = *(unsigned int *)(a6 + 8);
            v69 = v269.m128i_i8[0];
          }
          v75 = (__int64 *)(*(_QWORD *)a6 + 16 * v71);
          v75[1] = v73;
          *v75 = v70;
          v76 = (unsigned int)(*(_DWORD *)(a6 + 8) + 1);
          v269.m128i_i32[0] = *(_DWORD *)(a6 + 8);
          *(_DWORD *)(a6 + 8) = v76;
          if ( *(_DWORD *)(a6 + 12) <= (unsigned int)v76 )
          {
            v269.m128i_i8[0] = v69;
            sub_16CD150(a6, (const void *)(a6 + 16), 0, 16, v74, v68);
            v76 = *(unsigned int *)(a6 + 8);
            v69 = v269.m128i_i8[0];
          }
          v263 = v69;
          v77 = (__int64 *)(*(_QWORD *)a6 + 16 * v76);
          *v77 = v70;
          v77[1] = v73;
          ++*(_DWORD *)(a6 + 8);
        }
        goto LABEL_41;
      }
    }
  }
  if ( (_BYTE)v344 )
  {
    if ( (unsigned __int8)(v344 - 14) <= 0x5Fu )
      goto LABEL_33;
  }
  else if ( sub_1F58D20((__int64)&v344) )
  {
    goto LABEL_33;
  }
  v34 = v240;
  if ( v241 <= v240 )
    v34 = v241;
  v35 = a2 == 54 && v29 < v34;
  if ( v35 )
  {
    *(_QWORD *)&a16 = v251;
    *(_QWORD *)&a18 = v248;
    if ( v244 )
    {
      v192 = sub_1D37440(v19, 59, a5, v350, v351, v33, *(double *)a7.m128i_i64, a8, a9, a16, a18);
      v194 = v193;
      v195 = (__int64)v192;
      v196 = v194;
      v332 = v195;
      v110 = v195;
      v333 = v196;
      v269.m128i_i64[0] = v195;
      v111 = (unsigned int)v196;
      v269.m128i_i64[1] = v269.m128i_i64[1] & 0xFFFFFFFF00000000LL | 1;
    }
    else
    {
      if ( !v243 )
        goto LABEL_33;
      v105 = sub_1D332F0(
               v19,
               54,
               a5,
               a12,
               a13,
               0,
               *(double *)a7.m128i_i64,
               a8,
               a9,
               a16,
               *((unsigned __int64 *)&a16 + 1),
               a18);
      v107 = v106;
      v108 = v105;
      v109 = v107;
      v110 = (__int64)v108;
      v330 = v108;
      v331 = v109;
      v111 = (unsigned int)v109;
      v328 = sub_1D332F0(
               v19,
               113,
               a5,
               a12,
               a13,
               0,
               *(double *)a7.m128i_i64,
               a8,
               a9,
               a16,
               *((unsigned __int64 *)&a16 + 1),
               a18);
      v269.m128i_i64[0] = (__int64)v328;
      v329 = v114;
      v269.m128i_i64[1] = (unsigned int)v114 | v269.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    }
    v115 = *(unsigned int *)(a6 + 8);
    if ( (unsigned int)v115 >= *(_DWORD *)(a6 + 12) )
    {
      sub_16CD150(a6, (const void *)(a6 + 16), 0, 16, v112, v113);
      v115 = *(unsigned int *)(a6 + 8);
    }
    v116 = (__int64 *)(*(_QWORD *)a6 + 16 * v115);
    *v116 = v110;
    v116[1] = v111;
    v117 = (unsigned int)(*(_DWORD *)(a6 + 8) + 1);
    *(_DWORD *)(a6 + 8) = v117;
    if ( *(_DWORD *)(a6 + 12) <= (unsigned int)v117 )
    {
      sub_16CD150(a6, (const void *)(a6 + 16), 0, 16, v112, v113);
      v117 = *(unsigned int *)(a6 + 8);
    }
    v263 = v35;
    *(__m128i *)(*(_QWORD *)a6 + 16 * v117) = _mm_load_si128(&v269);
    ++*(_DWORD *)(a6 + 8);
    goto LABEL_41;
  }
LABEL_33:
  v36 = v28 - v29;
  v37 = sub_1E0A0C0(v19[4]);
  v354 = sub_1F40B60(a1, (unsigned int)v344, (__int64)v345, v37, 1);
  v355 = v38;
  if ( (_BYTE)v354 )
    v39 = sub_1F3E310(&v354);
  else
    v39 = sub_1F58D40((__int64)&v354);
  LODWORD(v358) = v39;
  v40 = v36;
  if ( v39 <= 0x40 )
  {
    v357 = (unsigned __int64 *)(0xFFFFFFFFFFFFFFFFLL >> -(char)v39);
    goto LABEL_37;
  }
  sub_16A4EF0((__int64)&v357, -1, 1);
  v43 = (int)v358;
  if ( (unsigned int)v358 <= 0x40 )
  {
LABEL_37:
    if ( v40 <= (unsigned __int64)v357 )
      goto LABEL_38;
LABEL_57:
    LOBYTE(v354) = 5;
    v355 = 0;
    goto LABEL_38;
  }
  if ( v43 - (unsigned int)sub_16A57B0((__int64)&v357) <= 0x40 && v40 > *v357 )
  {
    j_j___libc_free_0_0(v357);
    goto LABEL_57;
  }
  if ( v357 )
    j_j___libc_free_0_0(v357);
LABEL_38:
  *(_QWORD *)&v356 = sub_1D38BB0((__int64)v19, v40, a5, v354, v355, 0, a7, a8, a9, 0);
  *((_QWORD *)&v356 + 1) = v42;
  if ( !v247 )
  {
    if ( v242 )
      goto LABEL_102;
    v181 = 1;
    if ( (_BYTE)v344 != 1 )
    {
      if ( !(_BYTE)v344 )
        goto LABEL_102;
      v181 = (unsigned __int8)v344;
      if ( !*(_QWORD *)(a1 + 8LL * (unsigned __int8)v344 + 120) )
        goto LABEL_102;
    }
    if ( (*(_BYTE *)(a1 + 259LL * v181 + 2546) & 0xFB) != 0 )
      goto LABEL_102;
    v182 = 1;
    if ( (_BYTE)a12 != 1 )
    {
      if ( !(_BYTE)a12 )
        goto LABEL_102;
      v182 = (unsigned __int8)a12;
      if ( !*(_QWORD *)(a1 + 8LL * (unsigned __int8)a12 + 120) )
        goto LABEL_102;
    }
    if ( (*(_BYTE *)(a1 + 259LL * v182 + 2567) & 0xFB) != 0 )
      goto LABEL_102;
    *(_QWORD *)&a17 = sub_1D332F0(
                        v19,
                        124,
                        a5,
                        (unsigned int)v344,
                        v345,
                        0,
                        *(double *)a7.m128i_i64,
                        a8,
                        a9,
                        a10,
                        *((unsigned __int64 *)&a10 + 1),
                        v356);
    v326 = a17;
    v327 = v183;
    *((_QWORD *)&a17 + 1) = (unsigned int)v183 | *((_QWORD *)&a17 + 1) & 0xFFFFFFFF00000000LL;
    v247 = sub_1D309E0(v19, 145, a5, a12, a13, 0, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64, a17);
    v324 = v247;
    *(_QWORD *)&a17 = v247;
    v325 = v184;
    *((_QWORD *)&a17 + 1) = (unsigned int)v184 | *((_QWORD *)&a17 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&a19 = sub_1D332F0(
                        v19,
                        124,
                        a5,
                        (unsigned int)v344,
                        v345,
                        0,
                        *(double *)a7.m128i_i64,
                        a8,
                        a9,
                        a11,
                        *((unsigned __int64 *)&a11 + 1),
                        v356);
    v322 = a19;
    v323 = v185;
    *((_QWORD *)&a19 + 1) = (unsigned int)v185 | *((_QWORD *)&a19 + 1) & 0xFFFFFFFF00000000LL;
    v242 = sub_1D309E0(v19, 145, a5, a12, a13, 0, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64, a19);
    v320 = v242;
    *(_QWORD *)&a19 = v242;
    v321 = v186;
    *((_QWORD *)&a19 + 1) = (unsigned int)v186 | *((_QWORD *)&a19 + 1) & 0xFFFFFFFF00000000LL;
    if ( !v247 )
      goto LABEL_102;
  }
  *(_QWORD *)&a16 = v251;
  *(_QWORD *)&a18 = v248;
  if ( v245 )
  {
    v100 = sub_1D37440(v19, 60, a5, v350, v351, v41, *(double *)a7.m128i_i64, a8, a9, a16, a18);
    v102 = v101;
    v103 = (__int64)v100;
    v104 = v102;
    v318 = v103;
    v83 = v103;
    v319 = v104;
    v269.m128i_i64[0] = v103;
    v84 = (unsigned int)v104;
    v269.m128i_i64[1] = v269.m128i_i64[1] & 0xFFFFFFFF00000000LL | 1;
LABEL_89:
    v88 = *(unsigned int *)(a6 + 8);
    if ( (unsigned int)v88 >= *(_DWORD *)(a6 + 12) )
    {
      sub_16CD150(a6, (const void *)(a6 + 16), 0, 16, v85, v86);
      v88 = *(unsigned int *)(a6 + 8);
    }
    v89 = (__int64 *)(*(_QWORD *)a6 + 16 * v88);
    *v89 = v83;
    v89[1] = v84;
    ++*(_DWORD *)(a6 + 8);
    if ( a2 == 54 )
    {
      *((_QWORD *)&v238 + 1) = *((_QWORD *)&a19 + 1);
      *(_QWORD *)&a19 = v242;
      *(_QWORD *)&v238 = v242;
      v312 = sub_1D332F0(
               v19,
               54,
               a5,
               a12,
               a13,
               0,
               *(double *)a7.m128i_i64,
               a8,
               a9,
               a16,
               *((unsigned __int64 *)&a16 + 1),
               v238);
      *(_QWORD *)&a19 = v312;
      v313 = v197;
      *((_QWORD *)&a19 + 1) = (unsigned int)v197 | *((_QWORD *)&a19 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&a17 = v247;
      *(_QWORD *)&a17 = sub_1D332F0(
                          v19,
                          54,
                          a5,
                          a12,
                          a13,
                          0,
                          *(double *)a7.m128i_i64,
                          a8,
                          a9,
                          v247,
                          *((unsigned __int64 *)&a17 + 1),
                          a18);
      v310 = a17;
      v311 = v198;
      *((_QWORD *)&a17 + 1) = (unsigned int)v198 | *((_QWORD *)&a17 + 1) & 0xFFFFFFFF00000000LL;
      v308 = sub_1D332F0(
               v19,
               52,
               a5,
               a12,
               a13,
               0,
               *(double *)a7.m128i_i64,
               a8,
               a9,
               v269.m128i_i64[0],
               v269.m128i_u64[1],
               a19);
      v309 = v199;
      v269.m128i_i64[1] = (unsigned int)v199 | v269.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v306 = sub_1D332F0(
               v19,
               52,
               a5,
               a12,
               a13,
               0,
               *(double *)a7.m128i_i64,
               a8,
               a9,
               (__int64)v308,
               v269.m128i_u64[1],
               a17);
      v269.m128i_i64[0] = (__int64)v306;
      v307 = v202;
      v203 = *(unsigned int *)(a6 + 8);
      v269.m128i_i64[1] = (unsigned int)v202 | v269.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      if ( (unsigned int)v203 >= *(_DWORD *)(a6 + 12) )
      {
        sub_16CD150(a6, (const void *)(a6 + 16), 0, 16, v200, v201);
        v203 = *(unsigned int *)(a6 + 8);
      }
      v263 = 1;
      *(__m128i *)(*(_QWORD *)a6 + 16 * v203) = _mm_load_si128(&v269);
      ++*(_DWORD *)(a6 + 8);
      goto LABEL_41;
    }
    v359 = &v344;
    v357 = (unsigned __int64 *)v19;
    v358 = (const void **)a5;
    v360 = &v356;
    v256 = sub_1D309E0(
             v19,
             143,
             a5,
             (unsigned int)v344,
             v345,
             0,
             *(double *)a7.m128i_i64,
             a8,
             *(double *)a9.m128i_i64,
             *(_OWORD *)&v269);
    v259 = v91;
    *(_QWORD *)&a19 = v242;
    if ( v245 )
    {
      v118 = sub_1D37440(v19, 60, a5, v350, v351, v90, *(double *)a7.m128i_i64, a8, a9, a16, a19);
      v120 = v119;
      v121 = (__int64)v118;
      v122 = v120;
      v304 = v121;
      v97 = v121;
      v305 = v122;
      v269.m128i_i64[0] = v121;
      v98 = (unsigned int)v122 | v84 & 0xFFFFFFFF00000000LL;
      v269.m128i_i64[1] = v269.m128i_i64[1] & 0xFFFFFFFF00000000LL | 1;
    }
    else
    {
      if ( !v263 )
        goto LABEL_41;
      v92 = sub_1D332F0(
              v19,
              54,
              a5,
              a12,
              a13,
              0,
              *(double *)a7.m128i_i64,
              a8,
              a9,
              a16,
              *((unsigned __int64 *)&a16 + 1),
              a19);
      v94 = v93;
      v95 = v92;
      v96 = v94;
      v97 = (__int64)v95;
      v302 = v95;
      v303 = v96;
      v98 = (unsigned int)v96 | v84 & 0xFFFFFFFF00000000LL;
      v300 = sub_1D332F0(
               v19,
               112,
               a5,
               a12,
               a13,
               0,
               *(double *)a7.m128i_i64,
               a8,
               a9,
               a16,
               *((unsigned __int64 *)&a16 + 1),
               a19);
      v269.m128i_i64[0] = (__int64)v300;
      v301 = v99;
      v269.m128i_i64[1] = (unsigned int)v99 | v269.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    }
    *(_QWORD *)&v123 = sub_20A0A00(
                         (__int64)&v357,
                         v97,
                         v98,
                         v269.m128i_i64[0],
                         v269.m128i_i64[1],
                         *(double *)a7.m128i_i64,
                         a8,
                         a9);
    v298 = sub_1D332F0(v19, 52, a5, (unsigned int)v344, v345, 0, *(double *)a7.m128i_i64, a8, a9, v256, v259, v123);
    v299 = v125;
    v260 = (unsigned int)v125 | v259 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&a17 = v247;
    if ( v245 )
    {
      v204 = sub_1D37440(v19, 60, a5, v350, v351, v124, *(double *)a7.m128i_i64, a8, a9, a17, a18);
      v206 = v205;
      v207 = (__int64)v204;
      v208 = v206;
      v296 = v207;
      v131 = v207;
      v297 = v208;
      v269.m128i_i64[0] = v207;
      v132 = (unsigned int)v208 | v98 & 0xFFFFFFFF00000000LL;
      v269.m128i_i64[1] = v269.m128i_i64[1] & 0xFFFFFFFF00000000LL | 1;
    }
    else
    {
      if ( !v263 )
        goto LABEL_41;
      v126 = sub_1D332F0(
               v19,
               54,
               a5,
               a12,
               a13,
               0,
               *(double *)a7.m128i_i64,
               a8,
               a9,
               a17,
               *((unsigned __int64 *)&a17 + 1),
               a18);
      v128 = v127;
      v129 = v126;
      v130 = v128;
      v131 = (__int64)v129;
      v294 = v129;
      v295 = v130;
      v132 = (unsigned int)v130 | v98 & 0xFFFFFFFF00000000LL;
      v292 = sub_1D332F0(
               v19,
               112,
               a5,
               a12,
               a13,
               0,
               *(double *)a7.m128i_i64,
               a8,
               a9,
               a17,
               *((unsigned __int64 *)&a17 + 1),
               a18);
      v269.m128i_i64[0] = (__int64)v292;
      v293 = v133;
      v269.m128i_i64[1] = (unsigned int)v133 | v269.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    }
    *(_QWORD *)&v134 = sub_20A0A00(
                         (__int64)&v357,
                         v131,
                         v132,
                         v269.m128i_i64[0],
                         v269.m128i_i64[1],
                         *(double *)a7.m128i_i64,
                         a8,
                         a9);
    v252 = v134;
    v135 = (const void ***)sub_1D252B0((__int64)v19, (unsigned int)v344, (__int64)v345, 111, 0);
    *((_QWORD *)&v234 + 1) = v260;
    *(_QWORD *)&v234 = v298;
    v138 = sub_1D37440(v19, 64, a5, v135, v136, v137, *(double *)a7.m128i_i64, a8, a9, v234, v252);
    v291 = v139;
    v257 = (__int64)v138;
    v290 = v138;
    v253 = v138;
    v261 = (unsigned int)v139 | v260 & 0xFFFFFFFF00000000LL;
    v140 = sub_1D309E0(
             v19,
             145,
             a5,
             a12,
             a13,
             0,
             *(double *)a7.m128i_i64,
             a8,
             *(double *)a9.m128i_i64,
             __PAIR128__(v261, (unsigned __int64)v138));
    v141 = v140;
    v143 = v142;
    v144 = *(unsigned int *)(a6 + 8);
    if ( (unsigned int)v144 >= *(_DWORD *)(a6 + 12) )
    {
      v249 = v140;
      v250 = v143;
      sub_16CD150(a6, (const void *)(a6 + 16), 0, 16, v140, v143);
      v144 = *(unsigned int *)(a6 + 8);
      v141 = v249;
      v143 = v250;
    }
    v145 = (__int64 *)(*(_QWORD *)a6 + 16 * v144);
    v145[1] = v143;
    *v145 = v141;
    ++*(_DWORD *)(a6 + 8);
    v288 = sub_1D332F0(v19, 124, a5, (unsigned int)v344, v345, 0, *(double *)a7.m128i_i64, a8, a9, v257, v261, v356);
    v289 = v147;
    v262 = (unsigned int)v147 | v261 & 0xFFFFFFFF00000000LL;
    if ( a2 == 59 )
    {
      if ( !v244 )
      {
LABEL_123:
        if ( a2 == 59 && v243 || a2 != 59 && v263 )
        {
          v148 = sub_1D332F0(
                   v19,
                   54,
                   a5,
                   a12,
                   a13,
                   0,
                   *(double *)a7.m128i_i64,
                   a8,
                   a9,
                   a17,
                   *((unsigned __int64 *)&a17 + 1),
                   a19);
          v150 = v149;
          v151 = v148;
          v152 = v150;
          v153 = (__int64)v151;
          v285 = v152;
          v154 = (unsigned int)v152 | v132 & 0xFFFFFFFF00000000LL;
          v284 = v151;
          v282 = sub_1D332F0(
                   v19,
                   (unsigned int)(a2 == 59) + 112,
                   a5,
                   a12,
                   a13,
                   0,
                   *(double *)a7.m128i_i64,
                   a8,
                   a9,
                   a17,
                   *((unsigned __int64 *)&a17 + 1),
                   a19);
          v269.m128i_i64[0] = (__int64)v282;
          v283 = v155;
          v269.m128i_i64[1] = (unsigned int)v155 | v269.m128i_i64[1] & 0xFFFFFFFF00000000LL;
LABEL_127:
          *(_QWORD *)&v156 = sub_1D38BB0((__int64)v19, 0, a5, a12, a13, 0, a7, a8, a9, 0);
          v265 = v156;
          v157 = (const void ***)sub_1D252B0((__int64)v19, a12, (__int64)a13, 111, 0);
          *((_QWORD *)&v235 + 1) = 1;
          *(_QWORD *)&v235 = v253;
          v160 = sub_1D37470(v19, 66, a5, v157, v158, v159, *(_OWORD *)&v269, v265, v235);
          v281 = v161;
          v269.m128i_i64[0] = (__int64)v160;
          v280 = v160;
          *(_QWORD *)&v162 = sub_20A0A00(
                               (__int64)&v357,
                               v153,
                               v154,
                               (__int64)v160,
                               (unsigned int)v161 | v269.m128i_i64[1] & 0xFFFFFFFF00000000LL,
                               *(double *)a7.m128i_i64,
                               a8,
                               a9);
          v163 = sub_1D332F0(
                   v19,
                   52,
                   a5,
                   (unsigned int)v344,
                   v345,
                   0,
                   *(double *)a7.m128i_i64,
                   a8,
                   a9,
                   (__int64)v288,
                   v262,
                   v162);
          v279 = v164;
          v278 = v163;
          *(_QWORD *)&v258 = v163;
          *((_QWORD *)&v258 + 1) = (unsigned int)v164 | v262 & 0xFFFFFFFF00000000LL;
          if ( a2 == 59 )
          {
            *(_QWORD *)&v215 = sub_1D309E0(
                                 v19,
                                 143,
                                 a5,
                                 (unsigned int)v344,
                                 v345,
                                 0,
                                 *(double *)a7.m128i_i64,
                                 a8,
                                 *(double *)a9.m128i_i64,
                                 a18);
            v216.m128i_i64[0] = (__int64)sub_1D332F0(
                                           v19,
                                           53,
                                           a5,
                                           (unsigned int)v344,
                                           v345,
                                           0,
                                           *(double *)a7.m128i_i64,
                                           a8,
                                           a9,
                                           v258,
                                           *((unsigned __int64 *)&v258 + 1),
                                           v215);
            v217 = v216.m128i_i64[1];
            v218 = v216.m128i_i64[0];
            v269 = v216;
            *(_QWORD *)&v222 = sub_1D28D50(v19, 0x14u, v216.m128i_i64[1], v219, v220, v221);
            v276 = sub_1D36A20(
                     v19,
                     136,
                     a5,
                     *(unsigned __int8 *)(*(_QWORD *)(v218 + 40) + 16LL * (unsigned int)v217),
                     *(const void ***)(*(_QWORD *)(v218 + 40) + 16LL * (unsigned int)v217 + 8),
                     v223,
                     a17,
                     v265,
                     *(_OWORD *)&v269,
                     v258,
                     v222);
            *(_QWORD *)&v258 = v276;
            v277 = v224;
            *((_QWORD *)&v258 + 1) = (unsigned int)v224 | *((_QWORD *)&v258 + 1) & 0xFFFFFFFF00000000LL;
            *(_QWORD *)&v225 = sub_1D309E0(
                                 v19,
                                 143,
                                 a5,
                                 (unsigned int)v344,
                                 v345,
                                 0,
                                 *(double *)a7.m128i_i64,
                                 a8,
                                 *(double *)a9.m128i_i64,
                                 a16);
            v274 = sub_1D332F0(
                     v19,
                     53,
                     a5,
                     (unsigned int)v344,
                     v345,
                     0,
                     *(double *)a7.m128i_i64,
                     a8,
                     a9,
                     (__int64)v276,
                     *((unsigned __int64 *)&v258 + 1),
                     v225);
            v275 = v226;
            v269.m128i_i64[0] = (__int64)v274;
            v269.m128i_i64[1] = (unsigned int)v226 | v217 & 0xFFFFFFFF00000000LL;
            *(_QWORD *)&v230 = sub_1D28D50(v19, 0x14u, (__int64)v274, v227, v228, v229);
            v231 = (const void ***)(v274[5] + 16LL * v269.m128i_u32[2]);
            v163 = sub_1D36A20(
                     v19,
                     136,
                     a5,
                     *(unsigned __int8 *)v231,
                     v231[1],
                     v232,
                     a19,
                     v265,
                     *(_OWORD *)&v269,
                     v258,
                     v230);
            v273 = v233;
            v272 = v163;
            *((_QWORD *)&v258 + 1) = (unsigned int)v233 | *((_QWORD *)&v258 + 1) & 0xFFFFFFFF00000000LL;
          }
          *((_QWORD *)&v236 + 1) = *((_QWORD *)&v258 + 1);
          *(_QWORD *)&v258 = v163;
          *(_QWORD *)&v236 = v163;
          v167 = sub_1D309E0(v19, 145, a5, a12, a13, 0, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64, v236);
          v169 = v168;
          v170 = *(unsigned int *)(a6 + 8);
          if ( (unsigned int)v170 >= *(_DWORD *)(a6 + 12) )
          {
            sub_16CD150(a6, (const void *)(a6 + 16), 0, 16, v165, v166);
            v170 = *(unsigned int *)(a6 + 8);
          }
          v171 = (__int64 *)(*(_QWORD *)a6 + 16 * v170);
          v171[1] = v169;
          *v171 = v167;
          v172 = (unsigned int)v344;
          ++*(_DWORD *)(a6 + 8);
          v270 = sub_1D332F0(
                   v19,
                   124,
                   a5,
                   v172,
                   v345,
                   0,
                   *(double *)a7.m128i_i64,
                   a8,
                   a9,
                   v258,
                   *((unsigned __int64 *)&v258 + 1),
                   v356);
          v271 = v173;
          *((_QWORD *)&v237 + 1) = (unsigned int)v173 | *((_QWORD *)&v258 + 1) & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v237 = v270;
          v176 = sub_1D309E0(v19, 145, a5, a12, a13, 0, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64, v237);
          v178 = v177;
          v179 = *(unsigned int *)(a6 + 8);
          if ( (unsigned int)v179 >= *(_DWORD *)(a6 + 12) )
          {
            sub_16CD150(a6, (const void *)(a6 + 16), 0, 16, v174, v175);
            v179 = *(unsigned int *)(a6 + 8);
          }
          v263 = 1;
          v180 = (__int64 *)(*(_QWORD *)a6 + 16 * v179);
          *v180 = v176;
          v180[1] = v178;
          ++*(_DWORD *)(a6 + 8);
          goto LABEL_41;
        }
LABEL_102:
        v263 = 0;
        goto LABEL_41;
      }
      v209 = 59;
    }
    else
    {
      if ( !v245 )
        goto LABEL_123;
      v209 = (unsigned int)(a2 != 59) + 59;
    }
    v210 = sub_1D37440(v19, v209, a5, v350, v351, v146, *(double *)a7.m128i_i64, a8, a9, a17, a19);
    v212 = v211;
    v213 = (__int64)v210;
    v214 = v212;
    v286 = v213;
    v153 = v213;
    v287 = v214;
    v269.m128i_i64[0] = v213;
    v154 = (unsigned int)v214 | v132 & 0xFFFFFFFF00000000LL;
    v269.m128i_i64[1] = v269.m128i_i64[1] & 0xFFFFFFFF00000000LL | 1;
    goto LABEL_127;
  }
  if ( v263 )
  {
    v78 = sub_1D332F0(
            v19,
            54,
            a5,
            a12,
            a13,
            0,
            *(double *)a7.m128i_i64,
            a8,
            a9,
            a16,
            *((unsigned __int64 *)&a16 + 1),
            a18);
    v80 = v79;
    v81 = v78;
    v82 = v80;
    v83 = (__int64)v81;
    v316 = v81;
    v317 = v82;
    v84 = (unsigned int)v82;
    v314 = sub_1D332F0(
             v19,
             112,
             a5,
             a12,
             a13,
             0,
             *(double *)a7.m128i_i64,
             a8,
             a9,
             a16,
             *((unsigned __int64 *)&a16 + 1),
             a18);
    v269.m128i_i64[0] = (__int64)v314;
    v315 = v87;
    v269.m128i_i64[1] = (unsigned int)v87 | v269.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    goto LABEL_89;
  }
LABEL_41:
  if ( v353 > 0x40 && v352 )
    j_j___libc_free_0_0(v352);
  return v263;
}
