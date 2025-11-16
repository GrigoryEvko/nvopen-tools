// Function: sub_197EA40
// Address: 0x197ea40
//
__int64 __fastcall sub_197EA40(
        __int64 a1,
        __int64 a2,
        __int64 (__fastcall *a3)(__int64, __int64),
        __int64 a4,
        __m128i a5,
        __m128i a6,
        __m128 a7,
        __m128 a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  unsigned __int64 **v12; // r13
  unsigned __int64 *v13; // rdx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rax
  __int64 *v21; // rax
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // rdx
  __int64 *v25; // rsi
  __int64 *v26; // rdi
  unsigned __int64 v27; // r15
  __int64 v28; // rax
  unsigned __int64 v29; // rcx
  unsigned __int64 v30; // rdx
  __int64 *v31; // rax
  char v32; // r8
  __int64 v33; // r8
  int v34; // r9d
  __int64 *v35; // rcx
  unsigned __int64 v36; // r15
  __int64 v37; // rax
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rdx
  __int64 *v40; // rax
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // r15
  unsigned __int64 v43; // rdx
  __int64 v44; // r12
  __int64 *v45; // rax
  char v46; // dl
  __int64 v47; // r13
  __int64 *v48; // rax
  __int64 *v49; // rcx
  __int64 *v50; // rsi
  unsigned __int64 v51; // rcx
  char v52; // si
  char v53; // al
  bool v54; // al
  __int64 v55; // rbx
  const __m128i **v56; // rax
  __int64 *v57; // rbx
  __int64 v58; // rax
  unsigned int *v59; // r13
  unsigned int *v60; // r12
  __int64 v61; // rax
  __int64 v62; // r14
  __int64 v63; // r15
  __int64 v64; // rax
  __int64 *v65; // rax
  _QWORD *v66; // rax
  unsigned __int64 v67; // rcx
  _QWORD **v68; // r12
  const char *v69; // rbx
  const char *v70; // rax
  _QWORD *v71; // rdi
  __int64 v72; // r13
  unsigned __int64 v73; // rax
  unsigned __int64 v74; // rdi
  __int64 v75; // r14
  unsigned int v76; // esi
  unsigned __int64 v77; // r13
  __int64 v78; // rax
  int v79; // ebx
  unsigned int v80; // edx
  unsigned __int64 v81; // rax
  __int64 v82; // rcx
  __int64 *v83; // r15
  int v84; // ecx
  unsigned int v85; // edx
  __int64 v86; // r13
  __int64 v87; // rdx
  __int64 v88; // rbx
  __int64 v89; // r12
  __int64 v90; // r13
  __int64 v91; // r14
  _QWORD *v92; // rax
  __int64 v93; // rbx
  __int64 v94; // r15
  int v95; // r8d
  int v96; // r9d
  __int64 v97; // rax
  __int64 *v98; // r14
  __int64 v99; // rbx
  const char **v100; // r15
  __int64 v101; // rax
  __int64 v102; // rbx
  unsigned __int64 v103; // rdi
  __int64 v104; // rax
  _QWORD *v105; // rax
  _QWORD *v106; // rdx
  const char **v107; // rax
  _QWORD *v108; // r12
  _QWORD *v109; // rdi
  _QWORD *v110; // rbx
  _QWORD *v111; // r12
  unsigned __int64 v112; // rdi
  __int64 *v113; // r13
  __int64 *v114; // r12
  __int64 v115; // rax
  int v117; // r8d
  int v118; // r9d
  __int64 v119; // rax
  __int64 *v120; // rax
  _QWORD *v121; // r12
  _QWORD *v122; // rbx
  __int64 v123; // rsi
  int v124; // r10d
  unsigned __int64 v125; // r9
  unsigned __int64 v126; // rsi
  int v127; // edi
  unsigned int v128; // edx
  __int64 v129; // r13
  unsigned __int64 v130; // rsi
  unsigned __int64 v131; // r8
  __int64 v132; // rbx
  __int64 *v133; // r10
  unsigned int v134; // r14d
  __int64 v135; // r15
  unsigned int v136; // edi
  __int64 *v137; // rdx
  __int64 v138; // r11
  unsigned int v139; // r11d
  __int64 v140; // r9
  unsigned int v141; // edi
  __int64 *v142; // rdx
  __int64 v143; // r13
  unsigned int v144; // edx
  unsigned int v145; // r10d
  __int64 v146; // r9
  unsigned int v147; // edi
  __int64 *v148; // rdx
  __int64 v149; // r12
  unsigned int v150; // r11d
  __int64 v151; // r9
  unsigned int v152; // edi
  __int64 *v153; // rdx
  __int64 v154; // r13
  unsigned int v155; // edx
  __int64 v156; // rdi
  __int64 v157; // r12
  unsigned int v158; // esi
  __int64 *v159; // rdx
  __int64 v160; // r8
  unsigned int v161; // edx
  __int64 *v162; // rax
  __int64 v163; // rsi
  __m128i *v164; // r8
  __int64 *v165; // rsi
  _QWORD *v166; // r15
  unsigned __int64 v167; // rbx
  _BOOL4 v168; // r12d
  __int64 v169; // r13
  unsigned __int64 v170; // r12
  _QWORD *v171; // rax
  __m128i *v172; // rdx
  __m128i *v173; // r14
  __int64 v174; // rax
  const __m128i *v175; // rbx
  __int64 v176; // rax
  __int64 v177; // r12
  __int64 v178; // rax
  unsigned int *v179; // r9
  unsigned __int64 v180; // r10
  unsigned int *v181; // r15
  __int64 v182; // rdx
  unsigned __int64 v183; // rbx
  unsigned __int64 v184; // r13
  const char *v185; // rdx
  const char *v186; // r14
  const char *v187; // rax
  const char *v188; // r11
  __m128i *v189; // rax
  __int64 v190; // rsi
  __int64 v191; // rcx
  __int64 v192; // rax
  __int64 v193; // r12
  const char *v194; // rax
  __m128i *v195; // rax
  __m128i *v196; // rdi
  __int64 v197; // rcx
  __int64 v198; // rdx
  __int64 v199; // rax
  const char *v200; // rsi
  const char *v201; // rax
  __int64 *v202; // rsi
  __int64 *v203; // rcx
  int v204; // edx
  int v205; // eax
  int v206; // edx
  int v207; // edx
  __int64 *v208; // rsi
  __int64 *v209; // rcx
  int v210; // edx
  int v211; // edx
  const char *v212; // rdx
  __int64 v213; // rax
  __int64 v214; // rax
  __int64 v215; // rax
  __int64 v216; // r12
  __int64 v217; // rax
  _QWORD *v218; // r14
  __int64 v219; // rdx
  __int64 **v220; // rax
  __int64 *v221; // rcx
  unsigned __int64 v222; // rsi
  __int64 v223; // rcx
  __int64 v224; // rax
  __int64 v225; // rcx
  __int64 v226; // rsi
  __int64 v227; // rdx
  __int64 v228; // rcx
  __int64 v229; // r8
  __int64 v230; // r9
  __int64 v231; // r13
  double v232; // xmm4_8
  double v233; // xmm5_8
  __int64 v234; // r15
  int v235; // eax
  __int64 v236; // rax
  int v237; // edx
  __int64 v238; // rdx
  _QWORD *v239; // rax
  __int64 v240; // rsi
  unsigned __int64 v241; // rdi
  __int64 v242; // rsi
  __int64 v243; // rax
  __int64 v244; // r12
  _QWORD *v245; // rbx
  __int64 *v246; // r12
  __int64 v247; // r13
  unsigned __int64 v248; // rax
  __int64 v249; // r12
  unsigned int v250; // ebx
  __int64 *v251; // r15
  __int64 v252; // r9
  __int64 v253; // rax
  __int64 v254; // rdx
  __int64 v255; // rcx
  __int64 v256; // r8
  __int64 v257; // rsi
  __int64 v258; // r9
  __int64 v259; // rbx
  __int64 v260; // r12
  int v261; // eax
  __int64 v262; // rax
  int v263; // edx
  int v264; // edi
  _QWORD *v265; // r13
  _QWORD *v266; // r12
  __int64 v267; // rax
  int v268; // r12d
  int v269; // r11d
  int v270; // r12d
  int v271; // r9d
  int v272; // r8d
  int v273; // r9d
  __int32 v274; // ebx
  const __m128i *v275; // rax
  __int64 v276; // rdx
  __int64 v277; // rcx
  int v278; // r8d
  int v279; // r9d
  _QWORD *v280; // r14
  _QWORD *v281; // r13
  unsigned __int64 v282; // rdi
  unsigned __int64 *v283; // rsi
  size_t v284; // rdx
  unsigned __int64 *p_dest; // rdi
  int v286; // r9d
  int v287; // r11d
  unsigned int *v292; // [rsp+20h] [rbp-6F0h]
  const __m128i *v293; // [rsp+28h] [rbp-6E8h]
  unsigned __int8 v294; // [rsp+37h] [rbp-6D9h]
  __int64 *v295; // [rsp+68h] [rbp-6A8h]
  unsigned __int8 v296; // [rsp+78h] [rbp-698h]
  __int64 *v297; // [rsp+80h] [rbp-690h]
  unsigned int *v298; // [rsp+88h] [rbp-688h]
  const __m128i *v299; // [rsp+90h] [rbp-680h]
  __int64 v300; // [rsp+90h] [rbp-680h]
  unsigned __int64 **v301; // [rsp+98h] [rbp-678h]
  const char *v302; // [rsp+A8h] [rbp-668h]
  unsigned __int64 v303; // [rsp+A8h] [rbp-668h]
  __int64 v304; // [rsp+A8h] [rbp-668h]
  unsigned int *v305; // [rsp+B0h] [rbp-660h]
  _QWORD *v306; // [rsp+C0h] [rbp-650h]
  unsigned __int64 **v307; // [rsp+D8h] [rbp-638h]
  unsigned __int64 v308; // [rsp+D8h] [rbp-638h]
  unsigned __int64 v309; // [rsp+D8h] [rbp-638h]
  _QWORD *v310; // [rsp+E8h] [rbp-628h] BYREF
  __int64 *v311; // [rsp+F0h] [rbp-620h] BYREF
  __int64 v312; // [rsp+F8h] [rbp-618h]
  _BYTE v313[64]; // [rsp+100h] [rbp-610h] BYREF
  unsigned __int64 v314[16]; // [rsp+140h] [rbp-5D0h] BYREF
  __m128i v315; // [rsp+1C0h] [rbp-550h] BYREF
  unsigned __int64 v316[2]; // [rsp+1D0h] [rbp-540h] BYREF
  int v317; // [rsp+1E0h] [rbp-530h]
  _QWORD v318[8]; // [rsp+1E8h] [rbp-528h] BYREF
  unsigned __int64 v319; // [rsp+228h] [rbp-4E8h] BYREF
  unsigned __int64 v320; // [rsp+230h] [rbp-4E0h]
  unsigned __int64 v321; // [rsp+238h] [rbp-4D8h]
  __m128i v322; // [rsp+240h] [rbp-4D0h] BYREF
  unsigned __int64 dest; // [rsp+250h] [rbp-4C0h] BYREF
  __int64 v324; // [rsp+258h] [rbp-4B8h]
  __int64 *v325; // [rsp+260h] [rbp-4B0h]
  unsigned __int64 v326[2]; // [rsp+268h] [rbp-4A8h] BYREF
  char v327; // [rsp+278h] [rbp-498h] BYREF
  unsigned __int64 v328; // [rsp+2A8h] [rbp-468h] BYREF
  unsigned __int64 v329; // [rsp+2B0h] [rbp-460h]
  unsigned __int64 v330; // [rsp+2B8h] [rbp-458h]
  _QWORD *v331; // [rsp+300h] [rbp-410h]
  unsigned int v332; // [rsp+310h] [rbp-400h]
  __int64 v333; // [rsp+320h] [rbp-3F0h] BYREF
  __int64 v334; // [rsp+328h] [rbp-3E8h]
  unsigned __int64 v335; // [rsp+330h] [rbp-3E0h]
  unsigned __int64 v336; // [rsp+338h] [rbp-3D8h]
  unsigned int v337; // [rsp+340h] [rbp-3D0h]
  __int64 v338; // [rsp+348h] [rbp-3C8h] BYREF
  __int64 *v339; // [rsp+350h] [rbp-3C0h]
  __int64 v340; // [rsp+358h] [rbp-3B8h]
  __m128i v341[2]; // [rsp+360h] [rbp-3B0h] BYREF
  __int64 *v342; // [rsp+388h] [rbp-388h]
  unsigned __int64 v343; // [rsp+390h] [rbp-380h]
  unsigned __int64 v344; // [rsp+398h] [rbp-378h]
  _QWORD *v345; // [rsp+3A8h] [rbp-368h]
  unsigned int v346; // [rsp+3B8h] [rbp-358h]
  char v347; // [rsp+3C0h] [rbp-350h]
  __int64 v348; // [rsp+3D0h] [rbp-340h]
  void *v349; // [rsp+3E0h] [rbp-330h]
  char *v350; // [rsp+408h] [rbp-308h]
  char v351; // [rsp+418h] [rbp-2F8h] BYREF
  _QWORD *v352; // [rsp+4A0h] [rbp-270h]
  unsigned int v353; // [rsp+4B0h] [rbp-260h]
  unsigned __int64 *v354; // [rsp+4D0h] [rbp-240h] BYREF
  unsigned __int64 v355; // [rsp+4D8h] [rbp-238h] BYREF
  const char *v356; // [rsp+4E0h] [rbp-230h] BYREF
  __int64 v357; // [rsp+4E8h] [rbp-228h]
  _QWORD *v358; // [rsp+4F0h] [rbp-220h]
  __int64 v359; // [rsp+4F8h] [rbp-218h] BYREF
  unsigned int v360; // [rsp+500h] [rbp-210h]
  __int64 v361; // [rsp+508h] [rbp-208h]
  __int64 v362; // [rsp+510h] [rbp-200h]
  __int64 v363; // [rsp+518h] [rbp-1F8h]
  __int64 v364; // [rsp+520h] [rbp-1F0h]
  __int64 v365; // [rsp+528h] [rbp-1E8h]
  __int64 v366; // [rsp+530h] [rbp-1E0h]
  __int64 *v367; // [rsp+538h] [rbp-1D8h]
  __int64 *v368; // [rsp+540h] [rbp-1D0h]
  unsigned __int64 v369; // [rsp+548h] [rbp-1C8h]
  __int64 v370; // [rsp+550h] [rbp-1C0h] BYREF
  __int64 v371; // [rsp+558h] [rbp-1B8h]
  unsigned __int64 v372; // [rsp+560h] [rbp-1B0h]
  __int64 v373; // [rsp+568h] [rbp-1A8h]
  _BYTE *v374; // [rsp+570h] [rbp-1A0h]
  unsigned __int64 v375[2]; // [rsp+578h] [rbp-198h] BYREF
  int v376; // [rsp+588h] [rbp-188h]
  _BYTE v377[16]; // [rsp+590h] [rbp-180h] BYREF
  __int64 v378; // [rsp+5A0h] [rbp-170h]
  __int64 v379; // [rsp+5A8h] [rbp-168h]
  __int64 v380; // [rsp+5B0h] [rbp-160h]
  __int64 *v381; // [rsp+5B8h] [rbp-158h]
  unsigned __int64 v382; // [rsp+5C0h] [rbp-150h]
  unsigned __int64 v383; // [rsp+5C8h] [rbp-148h]
  __int16 v384; // [rsp+5D0h] [rbp-140h]
  __int64 v385[5]; // [rsp+5D8h] [rbp-138h] BYREF
  int v386; // [rsp+600h] [rbp-110h]
  __int64 v387; // [rsp+608h] [rbp-108h]
  __int64 v388; // [rsp+610h] [rbp-100h]
  __int64 v389; // [rsp+618h] [rbp-F8h]
  _BYTE *v390; // [rsp+620h] [rbp-F0h]
  __int64 v391; // [rsp+628h] [rbp-E8h]
  _BYTE v392[224]; // [rsp+630h] [rbp-E0h] BYREF

  v12 = *(unsigned __int64 ***)(a1 + 32);
  v311 = (__int64 *)v313;
  v312 = 0x800000000LL;
  v301 = *(unsigned __int64 ***)(a1 + 40);
  if ( v12 == v301 )
    return 0;
  v307 = v12;
  do
  {
    v13 = *v307;
    v319 = 0;
    memset(v314, 0, sizeof(v314));
    v318[0] = v13;
    v314[1] = (unsigned __int64)&v314[5];
    v314[2] = (unsigned __int64)&v314[5];
    v354 = v13;
    v315.m128i_i64[1] = (__int64)v318;
    v316[0] = (unsigned __int64)v318;
    v316[1] = 0x100000008LL;
    LODWORD(v314[3]) = 8;
    v320 = 0;
    v321 = 0;
    v317 = 0;
    v315.m128i_i64[0] = 1;
    LOBYTE(v356) = 0;
    sub_197E9F0(&v319, (__int64)&v354);
    sub_16CCEE0(&v333, (__int64)&v338, 8, (__int64)v314);
    v14 = v314[13];
    memset(&v314[13], 0, 24);
    v342 = (__int64 *)v14;
    v343 = v314[14];
    v344 = v314[15];
    sub_16CCEE0(&v322, (__int64)v326, 8, (__int64)&v315);
    v15 = v319;
    v319 = 0;
    v328 = v15;
    v16 = v320;
    v320 = 0;
    v329 = v16;
    v17 = v321;
    v321 = 0;
    v330 = v17;
    sub_16CCEE0(&v354, (__int64)&v359, 8, (__int64)&v322);
    v18 = v328;
    v328 = 0;
    v367 = (__int64 *)v18;
    v19 = v329;
    v329 = 0;
    v368 = (__int64 *)v19;
    v20 = v330;
    v330 = 0;
    v369 = v20;
    sub_16CCEE0(&v370, (__int64)v375, 8, (__int64)&v333);
    v21 = v342;
    v342 = 0;
    v381 = v21;
    v22 = v343;
    v343 = 0;
    v382 = v22;
    v23 = v344;
    v344 = 0;
    v383 = v23;
    if ( v328 )
      j_j___libc_free_0(v328, v330 - v328);
    if ( dest != v322.m128i_i64[1] )
      _libc_free(dest);
    if ( v342 )
      j_j___libc_free_0(v342, v344 - (_QWORD)v342);
    if ( v335 != v334 )
      _libc_free(v335);
    if ( v319 )
      j_j___libc_free_0(v319, v321 - v319);
    if ( v316[0] != v315.m128i_i64[1] )
      _libc_free(v316[0]);
    if ( v314[13] )
      j_j___libc_free_0(v314[13], v314[15] - v314[13]);
    if ( v314[2] != v314[1] )
      _libc_free(v314[2]);
    sub_16CCCB0(&v322, (__int64)v326, (__int64)&v354);
    v25 = v368;
    v26 = v367;
    v328 = 0;
    v329 = 0;
    v330 = 0;
    v27 = (char *)v368 - (char *)v367;
    if ( v368 == v367 )
    {
      v27 = 0;
      v29 = 0;
    }
    else
    {
      if ( v27 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_490;
      v28 = sub_22077B0((char *)v368 - (char *)v367);
      v25 = v368;
      v26 = v367;
      v29 = v28;
    }
    v328 = v29;
    v329 = v29;
    v330 = v29 + v27;
    if ( v26 != v25 )
    {
      v30 = v29;
      v31 = v26;
      do
      {
        if ( v30 )
        {
          *(_QWORD *)v30 = *v31;
          v32 = *((_BYTE *)v31 + 16);
          *(_BYTE *)(v30 + 16) = v32;
          if ( v32 )
            *(_QWORD *)(v30 + 8) = v31[1];
        }
        v31 += 3;
        v30 += 24LL;
      }
      while ( v31 != v25 );
      v29 += 8 * ((unsigned __int64)((char *)(v31 - 3) - (char *)v26) >> 3) + 24;
    }
    v329 = v29;
    v26 = &v333;
    sub_16CCCB0(&v333, (__int64)&v338, (__int64)&v370);
    v35 = (__int64 *)v382;
    v25 = v381;
    v342 = 0;
    v343 = 0;
    v344 = 0;
    v36 = v382 - (_QWORD)v381;
    if ( (__int64 *)v382 == v381 )
    {
      v36 = 0;
      v38 = 0;
    }
    else
    {
      if ( v36 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_490:
        sub_4261EA(v26, v25, v24);
      v37 = sub_22077B0(v382 - (_QWORD)v381);
      v35 = (__int64 *)v382;
      v25 = v381;
      v38 = v37;
    }
    v342 = (__int64 *)v38;
    v343 = v38;
    v344 = v38 + v36;
    if ( v25 == v35 )
    {
      v41 = v38;
    }
    else
    {
      v39 = v38;
      v40 = v25;
      do
      {
        if ( v39 )
        {
          *(_QWORD *)v39 = *v40;
          LODWORD(v33) = *((unsigned __int8 *)v40 + 16);
          *(_BYTE *)(v39 + 16) = v33;
          if ( (_BYTE)v33 )
          {
            v33 = v40[1];
            *(_QWORD *)(v39 + 8) = v33;
          }
        }
        v40 += 3;
        v39 += 24LL;
      }
      while ( v40 != v35 );
      v41 = v38 + 8 * ((unsigned __int64)((char *)(v40 - 3) - (char *)v25) >> 3) + 24;
    }
    v42 = v329;
    v43 = v328;
    v343 = v41;
    if ( v329 - v328 == v41 - v38 )
      goto LABEL_56;
    do
    {
LABEL_40:
      v44 = *(_QWORD *)(v42 - 24);
      if ( *(_QWORD *)(v44 + 16) == *(_QWORD *)(v44 + 8) )
      {
        v104 = (unsigned int)v312;
        if ( (unsigned int)v312 >= HIDWORD(v312) )
        {
          sub_16CD150((__int64)&v311, v313, 0, 8, v33, v34);
          v104 = (unsigned int)v312;
        }
        v311[v104] = v44;
        v42 = v329;
        LODWORD(v312) = v312 + 1;
LABEL_158:
        v44 = *(_QWORD *)(v42 - 24);
      }
      if ( !*(_BYTE *)(v42 - 8) )
      {
        v45 = *(__int64 **)(v44 + 8);
        *(_BYTE *)(v42 - 8) = 1;
        *(_QWORD *)(v42 - 16) = v45;
        goto LABEL_45;
      }
      while ( 1 )
      {
        v45 = *(__int64 **)(v42 - 16);
LABEL_45:
        if ( *(__int64 **)(v44 + 16) == v45 )
          break;
        *(_QWORD *)(v42 - 16) = v45 + 1;
        v47 = *v45;
        v48 = (__int64 *)v322.m128i_i64[1];
        if ( dest != v322.m128i_i64[1] )
          goto LABEL_43;
        v49 = (__int64 *)(v322.m128i_i64[1] + 8LL * HIDWORD(v324));
        if ( (__int64 *)v322.m128i_i64[1] == v49 )
        {
LABEL_155:
          if ( HIDWORD(v324) < (unsigned int)v324 )
          {
            ++HIDWORD(v324);
            *v49 = v47;
            ++v322.m128i_i64[0];
LABEL_54:
            v315.m128i_i64[0] = v47;
            LOBYTE(v316[0]) = 0;
            sub_197E9F0(&v328, (__int64)&v315);
            v43 = v328;
            v42 = v329;
            goto LABEL_55;
          }
LABEL_43:
          sub_16CCBA0((__int64)&v322, v47);
          if ( v46 )
            goto LABEL_54;
        }
        else
        {
          v50 = 0;
          while ( v47 != *v48 )
          {
            if ( *v48 == -2 )
            {
              v50 = v48;
              if ( v49 == v48 + 1 )
                goto LABEL_53;
              ++v48;
            }
            else if ( v49 == ++v48 )
            {
              if ( !v50 )
                goto LABEL_155;
LABEL_53:
              *v50 = v47;
              LODWORD(v325) = (_DWORD)v325 - 1;
              ++v322.m128i_i64[0];
              goto LABEL_54;
            }
          }
        }
      }
      v329 -= 24LL;
      v43 = v328;
      v42 = v329;
      if ( v329 != v328 )
        goto LABEL_158;
LABEL_55:
      v38 = (unsigned __int64)v342;
    }
    while ( v42 - v43 != v343 - (_QWORD)v342 );
LABEL_56:
    if ( v43 != v42 )
    {
      v51 = v38;
      while ( *(_QWORD *)v43 == *(_QWORD *)v51 )
      {
        v52 = *(_BYTE *)(v43 + 16);
        v53 = *(_BYTE *)(v51 + 16);
        if ( v52 && v53 )
          v54 = *(_QWORD *)(v43 + 8) == *(_QWORD *)(v51 + 8);
        else
          v54 = v52 == v53;
        if ( !v54 )
          break;
        v43 += 24LL;
        v51 += 24LL;
        if ( v43 == v42 )
          goto LABEL_64;
      }
      goto LABEL_40;
    }
LABEL_64:
    if ( v38 )
      j_j___libc_free_0(v38, v344 - v38);
    if ( v335 != v334 )
      _libc_free(v335);
    if ( v328 )
      j_j___libc_free_0(v328, v330 - v328);
    if ( dest != v322.m128i_i64[1] )
      _libc_free(dest);
    if ( v381 )
      j_j___libc_free_0(v381, v383 - (_QWORD)v381);
    if ( v372 != v371 )
      _libc_free(v372);
    if ( v367 )
      j_j___libc_free_0(v367, v369 - (_QWORD)v367);
    if ( v356 != (const char *)v355 )
      _libc_free((unsigned __int64)v356);
    ++v307;
  }
  while ( v301 != v307 );
  v295 = &v311[(unsigned int)v312];
  if ( v311 == v295 )
  {
    v296 = 0;
    goto LABEL_203;
  }
  v297 = v311;
  v296 = 0;
  do
  {
    v55 = *v297;
    v56 = (const __m128i **)a3(a4, *v297);
    v333 = v55;
    v339 = (__int64 *)v56;
    v338 = a1;
    v334 = 0;
    v340 = a2;
    v335 = 0;
    v336 = 0;
    v337 = 0;
    sub_1497820(v341, *v56);
    v57 = v339;
    v310 = 0;
    v58 = v339[2];
    v294 = *(_BYTE *)(v58 + 218);
    if ( !v294 )
      goto LABEL_179;
    v354 = 0;
    v357 = 4;
    v355 = (unsigned __int64)&v359;
    v356 = (const char *)&v359;
    LODWORD(v358) = 0;
    v59 = *(unsigned int **)(v58 + 224);
    v60 = &v59[3 * *(unsigned int *)(v58 + 232)];
    if ( v59 == v60 )
      goto LABEL_120;
    do
    {
      while ( 1 )
      {
        v64 = *(_QWORD *)(v57[2] + 48);
        v62 = *(_QWORD *)(v64 + 8LL * *v59);
        v63 = *(_QWORD *)(v64 + 8LL * v59[1]);
        if ( v59[2] != 1 )
        {
          if ( (unsigned __int8)sub_385F830(v59) )
          {
            v61 = v62;
            v62 = v63;
            v63 = v61;
          }
          if ( *(_BYTE *)(v62 + 16) == 55
            && *(_BYTE *)(v63 + 16) == 54
            && **(_QWORD **)(v62 - 24) == **(_QWORD **)(v63 - 24) )
          {
            v105 = (_QWORD *)sub_22077B0(24);
            v106 = v310;
            v105[1] = v63;
            v105[2] = v62;
            *v105 = v106;
            v310 = v105;
          }
          goto LABEL_92;
        }
        if ( *(_BYTE *)(v62 + 16) == 54 )
        {
          v120 = (__int64 *)v355;
          if ( v356 != (const char *)v355 )
            goto LABEL_230;
          v202 = (__int64 *)(v355 + 8LL * HIDWORD(v357));
          if ( (__int64 *)v355 == v202 )
          {
LABEL_395:
            if ( HIDWORD(v357) >= (unsigned int)v357 )
            {
LABEL_230:
              sub_16CCBA0((__int64)&v354, v62);
              goto LABEL_95;
            }
            ++HIDWORD(v357);
            *v202 = v62;
            v354 = (unsigned __int64 *)((char *)v354 + 1);
          }
          else
          {
            v203 = 0;
            while ( v62 != *v120 )
            {
              if ( *v120 == -2 )
                v203 = v120;
              if ( v202 == ++v120 )
              {
                if ( !v203 )
                  goto LABEL_395;
                *v203 = v62;
                LODWORD(v358) = (_DWORD)v358 - 1;
                v354 = (unsigned __int64 *)((char *)v354 + 1);
                break;
              }
            }
          }
        }
LABEL_95:
        if ( *(_BYTE *)(v63 + 16) == 54 )
          break;
LABEL_92:
        v59 += 3;
        if ( v60 == v59 )
          goto LABEL_98;
      }
      v65 = (__int64 *)v355;
      if ( v356 != (const char *)v355 )
        goto LABEL_97;
      v208 = (__int64 *)(v355 + 8LL * HIDWORD(v357));
      if ( (__int64 *)v355 != v208 )
      {
        v209 = 0;
        while ( v63 != *v65 )
        {
          if ( *v65 == -2 )
            v209 = v65;
          if ( v208 == ++v65 )
          {
            if ( !v209 )
              goto LABEL_393;
            *v209 = v63;
            LODWORD(v358) = (_DWORD)v358 - 1;
            v354 = (unsigned __int64 *)((char *)v354 + 1);
            goto LABEL_92;
          }
        }
        goto LABEL_92;
      }
LABEL_393:
      if ( HIDWORD(v357) < (unsigned int)v357 )
      {
        ++HIDWORD(v357);
        *v208 = v63;
        v354 = (unsigned __int64 *)((char *)v354 + 1);
        goto LABEL_92;
      }
LABEL_97:
      v59 += 3;
      sub_16CCBA0((__int64)&v354, v63);
    }
    while ( v60 != v59 );
LABEL_98:
    if ( HIDWORD(v357) == (_DWORD)v358 )
    {
      v74 = (unsigned __int64)v356;
      v73 = v355;
    }
    else
    {
      v66 = v310;
      v67 = (unsigned __int64)v356;
      if ( v310 )
      {
        v67 = (unsigned __int64)v356;
        v68 = &v310;
        do
        {
          while ( 1 )
          {
            v72 = v66[1];
            v70 = (const char *)v355;
            if ( v67 == v355 )
              break;
            v69 = (const char *)(v67 + 8LL * (unsigned int)v357);
            v70 = (const char *)sub_16CC9F0((__int64)&v354, v72);
            if ( v72 == *(_QWORD *)v70 )
            {
              v67 = (unsigned __int64)v356;
              if ( v356 == (const char *)v355 )
                v212 = &v356[8 * HIDWORD(v357)];
              else
                v212 = &v356[8 * (unsigned int)v357];
              goto LABEL_114;
            }
            v67 = (unsigned __int64)v356;
            if ( v356 == (const char *)v355 )
            {
              v70 = &v356[8 * HIDWORD(v357)];
              v212 = v70;
              goto LABEL_114;
            }
            v70 = &v356[8 * (unsigned int)v357];
LABEL_104:
            v71 = *v68;
            if ( v69 == v70 )
              goto LABEL_116;
LABEL_105:
            *v68 = (_QWORD *)*v71;
            j_j___libc_free_0(v71, 24);
            v66 = *v68;
            v67 = (unsigned __int64)v356;
            if ( !*v68 )
              goto LABEL_117;
          }
          v69 = (const char *)(v67 + 8LL * HIDWORD(v357));
          if ( (const char *)v67 == v69 )
          {
            v212 = (const char *)v67;
          }
          else
          {
            do
            {
              if ( v72 == *(_QWORD *)v70 )
                break;
              v70 += 8;
            }
            while ( v69 != v70 );
            v212 = (const char *)(v67 + 8LL * HIDWORD(v357));
          }
LABEL_114:
          while ( v212 != v70 )
          {
            if ( *(_QWORD *)v70 < 0xFFFFFFFFFFFFFFFELL )
              goto LABEL_104;
            v70 += 8;
          }
          v71 = *v68;
          if ( v69 != v70 )
            goto LABEL_105;
LABEL_116:
          v68 = (_QWORD **)v71;
          v66 = (_QWORD *)*v71;
        }
        while ( *v71 );
      }
LABEL_117:
      v73 = v355;
      v74 = v67;
    }
    if ( v73 != v74 )
      _libc_free(v74);
LABEL_120:
    if ( !v310 )
      goto LABEL_179;
    v75 = v339[2];
    v354 = 0;
    v355 = 0;
    v356 = 0;
    LODWORD(v357) = 0;
    if ( *(_DWORD *)(v75 + 56) )
    {
      v76 = 0;
      v77 = 0;
      v78 = 0;
      v79 = 0;
      while ( 1 )
      {
        v83 = (__int64 *)(*(_QWORD *)(v75 + 48) + 8 * v78);
        if ( !v76 )
          break;
        v80 = (v76 - 1) & (((unsigned int)*v83 >> 9) ^ ((unsigned int)*v83 >> 4));
        v81 = v77 + 16LL * v80;
        v82 = *(_QWORD *)v81;
        if ( *v83 == *(_QWORD *)v81 )
        {
LABEL_124:
          *(_DWORD *)(v81 + 8) = v79;
          v78 = (unsigned int)(v79 + 1);
          v79 = v78;
          if ( *(_DWORD *)(v75 + 56) <= (unsigned int)v78 )
            goto LABEL_133;
          goto LABEL_125;
        }
        v124 = 1;
        v125 = 0;
        while ( v82 != -8 )
        {
          if ( !v125 && v82 == -16 )
            v125 = v81;
          v80 = (v76 - 1) & (v124 + v80);
          v81 = v77 + 16LL * v80;
          v82 = *(_QWORD *)v81;
          if ( *v83 == *(_QWORD *)v81 )
            goto LABEL_124;
          ++v124;
        }
        if ( v125 )
          v81 = v125;
        v354 = (unsigned __int64 *)((char *)v354 + 1);
        v84 = (_DWORD)v356 + 1;
        if ( 4 * ((int)v356 + 1) >= 3 * v76 )
          goto LABEL_128;
        if ( v76 - (v84 + HIDWORD(v356)) <= v76 >> 3 )
        {
          sub_14672C0((__int64)&v354, v76);
          if ( !(_DWORD)v357 )
          {
LABEL_513:
            LODWORD(v356) = (_DWORD)v356 + 1;
            BUG();
          }
          v126 = 0;
          v127 = 1;
          v84 = (_DWORD)v356 + 1;
          v128 = (v357 - 1) & (((unsigned int)*v83 >> 9) ^ ((unsigned int)*v83 >> 4));
          v81 = v355 + 16LL * v128;
          v129 = *(_QWORD *)v81;
          if ( *v83 != *(_QWORD *)v81 )
          {
            while ( v129 != -8 )
            {
              if ( v129 == -16 && !v126 )
                v126 = v81;
              v128 = (v357 - 1) & (v127 + v128);
              v81 = v355 + 16LL * v128;
              v129 = *(_QWORD *)v81;
              if ( *v83 == *(_QWORD *)v81 )
                goto LABEL_130;
              ++v127;
            }
            goto LABEL_440;
          }
        }
LABEL_130:
        LODWORD(v356) = v84;
        if ( *(_QWORD *)v81 != -8 )
          --HIDWORD(v356);
        v87 = *v83;
        *(_DWORD *)(v81 + 8) = 0;
        *(_DWORD *)(v81 + 8) = v79;
        *(_QWORD *)v81 = v87;
        v78 = (unsigned int)(v79 + 1);
        v79 = v78;
        if ( *(_DWORD *)(v75 + 56) <= (unsigned int)v78 )
          goto LABEL_133;
LABEL_125:
        v77 = v355;
        v76 = v357;
      }
      v354 = (unsigned __int64 *)((char *)v354 + 1);
LABEL_128:
      sub_14672C0((__int64)&v354, 2 * v76);
      if ( !(_DWORD)v357 )
        goto LABEL_513;
      v84 = (_DWORD)v356 + 1;
      v85 = (v357 - 1) & (((unsigned int)*v83 >> 9) ^ ((unsigned int)*v83 >> 4));
      v81 = v355 + 16LL * v85;
      v86 = *(_QWORD *)v81;
      if ( *v83 != *(_QWORD *)v81 )
      {
        v264 = 1;
        v126 = 0;
        while ( v86 != -8 )
        {
          if ( v86 == -16 && !v126 )
            v126 = v81;
          v85 = (v357 - 1) & (v264 + v85);
          v81 = v355 + 16LL * v85;
          v86 = *(_QWORD *)v81;
          if ( *v83 == *(_QWORD *)v81 )
            goto LABEL_130;
          ++v264;
        }
LABEL_440:
        if ( v126 )
          v81 = v126;
        goto LABEL_130;
      }
      goto LABEL_130;
    }
LABEL_133:
    j___libc_free_0(v335);
    ++v334;
    v354 = (unsigned __int64 *)((char *)v354 + 1);
    v335 = v355;
    v355 = 0;
    v336 = (unsigned __int64)v356;
    v356 = 0;
    v337 = v357;
    LODWORD(v357) = 0;
    j___libc_free_0(0);
    sub_197DA30((__int64)&v333, &v310, a5, a6);
    v306 = v310;
    if ( !v310 )
      goto LABEL_179;
    v314[0] = (unsigned __int64)&v314[2];
    v314[1] = 0x400000000LL;
    do
    {
      v88 = v333;
      v89 = v340;
      a5 = _mm_loadu_si128((const __m128i *)(v306 + 1));
      v322 = a5;
      v90 = *(_QWORD *)(a5.m128i_i64[1] + 40);
      v354 = (unsigned __int64 *)&v356;
      v355 = 0x800000000LL;
      v91 = *(_QWORD *)(**(_QWORD **)(v333 + 32) + 8LL);
      if ( !v91 )
      {
        v100 = &v356;
        v107 = &v356;
        goto LABEL_168;
      }
      while ( 1 )
      {
        v92 = sub_1648700(v91);
        if ( (unsigned __int8)(*((_BYTE *)v92 + 16) - 25) <= 9u )
          break;
        v91 = *(_QWORD *)(v91 + 8);
        if ( !v91 )
        {
          v100 = &v356;
          goto LABEL_167;
        }
      }
      v93 = v88 + 56;
      while ( 1 )
      {
        v94 = v92[5];
        if ( !sub_1377F70(v93, v94) )
          goto LABEL_138;
        v97 = (unsigned int)v355;
        if ( (unsigned int)v355 >= HIDWORD(v355) )
        {
          sub_16CD150((__int64)&v354, &v356, 0, 8, v95, v96);
          v97 = (unsigned int)v355;
        }
        v354[v97] = v94;
        LODWORD(v355) = v355 + 1;
        v91 = *(_QWORD *)(v91 + 8);
        if ( !v91 )
          break;
        while ( 1 )
        {
          v92 = sub_1648700(v91);
          if ( (unsigned __int8)(*((_BYTE *)v92 + 16) - 25) <= 9u )
            break;
LABEL_138:
          v91 = *(_QWORD *)(v91 + 8);
          if ( !v91 )
            goto LABEL_144;
        }
      }
LABEL_144:
      v98 = (__int64 *)v354;
      v99 = 8LL * (unsigned int)v355;
      v100 = (const char **)&v354[(unsigned __int64)v99 / 8];
      v101 = v99 >> 3;
      v102 = v99 >> 5;
      if ( v102 )
      {
        while ( sub_15CC8F0(v89, v90, *v98) )
        {
          if ( !sub_15CC8F0(v89, v90, v98[1]) )
          {
            v103 = (unsigned __int64)v354;
            ++v98;
            if ( v354 != (unsigned __int64 *)&v356 )
              goto LABEL_170;
            goto LABEL_171;
          }
          if ( !sub_15CC8F0(v89, v90, v98[2]) )
          {
            v103 = (unsigned __int64)v354;
            v98 += 2;
            if ( v354 != (unsigned __int64 *)&v356 )
              goto LABEL_170;
            goto LABEL_171;
          }
          if ( !sub_15CC8F0(v89, v90, v98[3]) )
          {
            v103 = (unsigned __int64)v354;
            v98 += 3;
            if ( v354 != (unsigned __int64 *)&v356 )
              goto LABEL_170;
            goto LABEL_171;
          }
          v98 += 4;
          if ( !--v102 )
          {
            v101 = ((char *)v100 - (char *)v98) >> 3;
            goto LABEL_207;
          }
        }
        goto LABEL_151;
      }
LABEL_207:
      switch ( v101 )
      {
        case 2LL:
          goto LABEL_227;
        case 3LL:
          if ( sub_15CC8F0(v89, v90, *v98) )
          {
            ++v98;
LABEL_227:
            if ( sub_15CC8F0(v89, v90, *v98) )
            {
              ++v98;
LABEL_210:
              if ( sub_15CC8F0(v89, v90, *v98) )
              {
                v103 = (unsigned __int64)v354;
                v98 = (__int64 *)v100;
                if ( v354 == (unsigned __int64 *)&v356 )
                  goto LABEL_172;
LABEL_170:
                _libc_free(v103);
                goto LABEL_171;
              }
            }
          }
LABEL_151:
          v103 = (unsigned __int64)v354;
          if ( v354 != (unsigned __int64 *)&v356 )
            goto LABEL_170;
LABEL_171:
          if ( v98 == (__int64 *)v100 )
            goto LABEL_172;
          goto LABEL_173;
        case 1LL:
          goto LABEL_210;
      }
LABEL_167:
      v107 = v100;
LABEL_168:
      v103 = (unsigned __int64)v354;
      if ( v354 != (unsigned __int64 *)&v356 )
      {
        v98 = (__int64 *)v100;
        v100 = v107;
        goto LABEL_170;
      }
LABEL_172:
      if ( *(_QWORD *)(v322.m128i_i64[0] + 40) == **(_QWORD **)(v333 + 32)
        && sub_197D610(&v322, (__int64)v341, v333, a5, a6) )
      {
        v119 = LODWORD(v314[1]);
        if ( LODWORD(v314[1]) >= HIDWORD(v314[1]) )
        {
          sub_16CD150((__int64)v314, &v314[2], 0, 16, v117, v118);
          v119 = LODWORD(v314[1]);
        }
        a7 = (__m128)_mm_load_si128(&v322);
        *(__m128 *)(v314[0] + 16 * v119) = a7;
        ++LODWORD(v314[1]);
      }
LABEL_173:
      v306 = (_QWORD *)*v306;
    }
    while ( v306 );
    if ( !LODWORD(v314[1]) )
      goto LABEL_175;
    v130 = v314[0];
    v131 = v314[0] + 16;
    v132 = *(_QWORD *)v314[0];
    v308 = v314[0] + 16LL * LODWORD(v314[1]);
    if ( v308 != v314[0] + 16 )
    {
      v133 = (__int64 *)(v314[0] + 16);
      v134 = v337 - 1;
      v135 = v335 + 16LL * v337;
      do
      {
        if ( v337 )
        {
          v136 = v134 & (((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4));
          v137 = (__int64 *)(v335 + 16LL * v136);
          v138 = *v137;
          if ( v132 == *v137 )
          {
LABEL_260:
            v139 = *((_DWORD *)v137 + 2);
            v140 = *v133;
          }
          else
          {
            v211 = 1;
            while ( v138 != -8 )
            {
              v271 = v211 + 1;
              v136 = v134 & (v136 + v211);
              v137 = (__int64 *)(v335 + 16LL * v136);
              v138 = *v137;
              if ( *v137 == v132 )
                goto LABEL_260;
              v211 = v271;
            }
            v139 = *(_DWORD *)(v135 + 8);
            v140 = *v133;
          }
          v141 = v134 & (((unsigned int)v140 >> 9) ^ ((unsigned int)v140 >> 4));
          v142 = (__int64 *)(v335 + 16LL * v141);
          v143 = *v142;
          if ( v140 == *v142 )
          {
LABEL_262:
            v144 = *((_DWORD *)v142 + 2);
          }
          else
          {
            v210 = 1;
            while ( v143 != -8 )
            {
              v270 = v210 + 1;
              v141 = v134 & (v210 + v141);
              v142 = (__int64 *)(v335 + 16LL * v141);
              v143 = *v142;
              if ( v140 == *v142 )
                goto LABEL_262;
              v210 = v270;
            }
            v144 = *(_DWORD *)(v135 + 8);
          }
          if ( v139 < v144 )
            v132 = v140;
        }
        v133 += 2;
      }
      while ( (__int64 *)v308 != v133 );
      v145 = v337 - 1;
      do
      {
        if ( v337 )
        {
          v146 = *(_QWORD *)(v131 + 8);
          v147 = v145 & (((unsigned int)v146 >> 9) ^ ((unsigned int)v146 >> 4));
          v148 = (__int64 *)(v335 + 16LL * v147);
          v149 = *v148;
          if ( v146 == *v148 )
          {
LABEL_269:
            v150 = *((_DWORD *)v148 + 2);
            v151 = *(_QWORD *)(v130 + 8);
          }
          else
          {
            v207 = 1;
            while ( v149 != -8 )
            {
              v269 = v207 + 1;
              v147 = v145 & (v147 + v207);
              v148 = (__int64 *)(v335 + 16LL * v147);
              v149 = *v148;
              if ( v146 == *v148 )
                goto LABEL_269;
              v207 = v269;
            }
            v150 = *(_DWORD *)(v135 + 8);
            v151 = *(_QWORD *)(v130 + 8);
          }
          v152 = v145 & (((unsigned int)v151 >> 9) ^ ((unsigned int)v151 >> 4));
          v153 = (__int64 *)(v335 + 16LL * v152);
          v154 = *v153;
          if ( *v153 == v151 )
          {
LABEL_271:
            v155 = *((_DWORD *)v153 + 2);
          }
          else
          {
            v206 = 1;
            while ( v154 != -8 )
            {
              v268 = v206 + 1;
              v152 = v145 & (v206 + v152);
              v153 = (__int64 *)(v335 + 16LL * v152);
              v154 = *v153;
              if ( *v153 == v151 )
                goto LABEL_271;
              v206 = v268;
            }
            v155 = *(_DWORD *)(v135 + 8);
          }
          if ( v150 < v155 )
            v130 = v131;
        }
        v131 += 16LL;
      }
      while ( v308 != v131 );
    }
    v156 = *(_QWORD *)(v130 + 8);
    v354 = 0;
    v357 = 4;
    LODWORD(v358) = 0;
    v355 = (unsigned __int64)&v359;
    v356 = (const char *)&v359;
    v157 = v339[2];
    if ( v337 )
    {
      v158 = (v337 - 1) & (((unsigned int)v156 >> 4) ^ ((unsigned int)v156 >> 9));
      v159 = (__int64 *)(v335 + 16LL * v158);
      v160 = *v159;
      if ( v156 == *v159 )
        goto LABEL_277;
      v204 = 1;
      while ( v160 != -8 )
      {
        v287 = v204 + 1;
        v158 = (v337 - 1) & (v158 + v204);
        v159 = (__int64 *)(v335 + 16LL * v158);
        v160 = *v159;
        if ( v156 == *v159 )
          goto LABEL_277;
        v204 = v287;
      }
    }
    v159 = (__int64 *)(v335 + 16LL * v337);
LABEL_277:
    sub_197D540(
      *(_QWORD *)(v157 + 48) + 8LL * *((unsigned int *)v159 + 2) + 8,
      *(_QWORD *)(v157 + 48) + 8LL * *(unsigned int *)(v157 + 56),
      (__int64)&v354);
    if ( !v337 )
      goto LABEL_371;
    v161 = (v337 - 1) & (((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4));
    v162 = (__int64 *)(v335 + 16LL * v161);
    v163 = *v162;
    if ( *v162 != v132 )
    {
      v205 = 1;
      while ( v163 != -8 )
      {
        v286 = v205 + 1;
        v161 = (v337 - 1) & (v205 + v161);
        v162 = (__int64 *)(v335 + 16LL * v161);
        v163 = *v162;
        if ( v132 == *v162 )
          goto LABEL_279;
        v205 = v286;
      }
LABEL_371:
      v162 = (__int64 *)(v335 + 16LL * v337);
    }
LABEL_279:
    sub_197D540(*(_QWORD *)(v157 + 48), *(_QWORD *)(v157 + 48) + 8LL * *((unsigned int *)v162 + 2), (__int64)&v354);
    v165 = &v322.m128i_i64[1];
    v322.m128i_i32[2] = 0;
    v166 = (_QWORD *)v314[0];
    v167 = v314[0] + 16LL * LODWORD(v314[1]);
    v324 = (__int64)&v322.m128i_i64[1];
    dest = 0;
    v325 = &v322.m128i_i64[1];
    v326[0] = 0;
    if ( v314[0] != v167 )
    {
      do
      {
        v315.m128i_i64[0] = sub_197CF40(v166);
        v170 = v315.m128i_i64[0];
        v171 = sub_1819AD0(&v322, v165, (unsigned __int64 *)&v315);
        v173 = v172;
        v169 = (__int64)v171;
        if ( v172 )
        {
          v168 = v171 || &v322.m128i_u64[1] == (unsigned __int64 *)v172 || v170 < v172[2].m128i_i64[0];
          v169 = sub_22077B0(40);
          *(_QWORD *)(v169 + 32) = v315.m128i_i64[0];
          sub_220F040(v168, v169, v173, &v322.m128i_u64[1]);
          ++v326[0];
        }
        v166 += 2;
        v165 = (__int64 *)sub_220EF30(v169);
      }
      while ( (_QWORD *)v167 != v166 );
    }
    v174 = v339[1];
    v315.m128i_i64[0] = (__int64)v316;
    v315.m128i_i64[1] = 0x400000000LL;
    v175 = *(const __m128i **)(v174 + 272);
    v176 = *(unsigned int *)(v174 + 280);
    v299 = v175;
    v293 = &v175[v176];
    if ( v175 == &v175[v176] )
      goto LABEL_308;
    while ( 2 )
    {
      v298 = *(unsigned int **)(v299->m128i_i64[0] + 24);
      v292 = &v298[*(unsigned int *)(v299->m128i_i64[0] + 32)];
      if ( v298 == v292 )
        goto LABEL_307;
      while ( 1 )
      {
        v177 = *v298;
        v178 = v299->m128i_i64[1];
        v179 = *(unsigned int **)(v178 + 24);
        v305 = &v179[*(unsigned int *)(v178 + 32)];
        if ( v179 != v305 )
          break;
LABEL_329:
        if ( v292 == ++v298 )
          goto LABEL_307;
      }
      v180 = (unsigned __int64)v356;
      v181 = *(unsigned int **)(v178 + 24);
      while ( 2 )
      {
        v182 = *(_QWORD *)(v339[1] + 8);
        v183 = *(_QWORD *)(v182 + (v177 << 6) + 16);
        v184 = *(_QWORD *)(v182 + ((unsigned __int64)*v181 << 6) + 16);
        v185 = (const char *)v355;
        if ( v180 == v355 )
        {
          v186 = (const char *)(v180 + 8LL * HIDWORD(v357));
          if ( (const char *)v180 == v186 )
          {
            v188 = (const char *)v180;
            v187 = (const char *)v180;
          }
          else
          {
            v187 = (const char *)v180;
            do
            {
              if ( v183 == *(_QWORD *)v187 )
                break;
              v187 += 8;
            }
            while ( v186 != v187 );
            v188 = (const char *)(v180 + 8LL * HIDWORD(v357));
          }
          goto LABEL_338;
        }
        v186 = (const char *)(v180 + 8LL * (unsigned int)v357);
        v187 = (const char *)sub_16CC9F0((__int64)&v354, v183);
        if ( v183 == *(_QWORD *)v187 )
        {
          v180 = (unsigned __int64)v356;
          v185 = (const char *)v355;
          if ( v356 == (const char *)v355 )
            v188 = &v356[8 * HIDWORD(v357)];
          else
            v188 = &v356[8 * (unsigned int)v357];
LABEL_338:
          while ( v188 != v187 && *(_QWORD *)v187 >= 0xFFFFFFFFFFFFFFFELL )
            v187 += 8;
          goto LABEL_296;
        }
        v180 = (unsigned __int64)v356;
        v185 = (const char *)v355;
        if ( v356 == (const char *)v355 )
        {
          v188 = &v356[8 * HIDWORD(v357)];
          v187 = v188;
          goto LABEL_338;
        }
        v187 = &v356[8 * (unsigned int)v357];
        v188 = v187;
LABEL_296:
        if ( v187 == v186 )
          goto LABEL_316;
        v189 = (__m128i *)dest;
        if ( !dest )
          goto LABEL_316;
        v164 = (__m128i *)&v322.m128i_u64[1];
        do
        {
          while ( 1 )
          {
            v190 = v189[1].m128i_i64[0];
            v191 = v189[1].m128i_i64[1];
            if ( v189[2].m128i_i64[0] >= v184 )
              break;
            v189 = (__m128i *)v189[1].m128i_i64[1];
            if ( !v191 )
              goto LABEL_302;
          }
          v164 = v189;
          v189 = (__m128i *)v189[1].m128i_i64[0];
        }
        while ( v190 );
LABEL_302:
        if ( v164 == (__m128i *)&v322.m128i_u64[1] || v164[2].m128i_i64[0] > v184 )
        {
LABEL_316:
          if ( v185 == (const char *)v180 )
          {
            v199 = 8LL * HIDWORD(v357);
            v200 = &v185[v199];
            if ( &v185[v199] == v185 )
            {
LABEL_354:
              v185 = (const char *)(v180 + v199);
              v201 = (const char *)(v180 + v199);
            }
            else
            {
              while ( v184 != *(_QWORD *)v185 )
              {
                v185 += 8;
                if ( v200 == v185 )
                  goto LABEL_354;
              }
              v201 = (const char *)(v180 + v199);
            }
          }
          else
          {
            v302 = v188;
            v194 = (const char *)sub_16CC9F0((__int64)&v354, v184);
            v188 = v302;
            v185 = v194;
            if ( v184 == *(_QWORD *)v194 )
            {
              v180 = (unsigned __int64)v356;
              if ( v356 == (const char *)v355 )
                v201 = &v356[8 * HIDWORD(v357)];
              else
                v201 = &v356[8 * (unsigned int)v357];
            }
            else
            {
              v180 = (unsigned __int64)v356;
              if ( v356 != (const char *)v355 )
              {
                v185 = &v356[8 * (unsigned int)v357];
                goto LABEL_320;
              }
              v201 = &v356[8 * HIDWORD(v357)];
              v185 = v201;
            }
          }
          while ( v201 != v185 && *(_QWORD *)v185 >= 0xFFFFFFFFFFFFFFFELL )
            v185 += 8;
LABEL_320:
          if ( v188 != v185 )
          {
            v195 = (__m128i *)dest;
            if ( dest )
            {
              v196 = (__m128i *)&v322.m128i_u64[1];
              do
              {
                while ( 1 )
                {
                  v197 = v195[1].m128i_i64[0];
                  v198 = v195[1].m128i_i64[1];
                  if ( v195[2].m128i_i64[0] >= v183 )
                    break;
                  v195 = (__m128i *)v195[1].m128i_i64[1];
                  if ( !v198 )
                    goto LABEL_326;
                }
                v196 = v195;
                v195 = (__m128i *)v195[1].m128i_i64[0];
              }
              while ( v197 );
LABEL_326:
              if ( v196 != (__m128i *)&v322.m128i_u64[1] && v196[2].m128i_i64[0] <= v183 )
                break;
            }
          }
          if ( v305 == ++v181 )
            goto LABEL_329;
          continue;
        }
        break;
      }
      v192 = v315.m128i_u32[2];
      if ( v315.m128i_i32[2] >= (unsigned __int32)v315.m128i_i32[3] )
      {
        sub_16CD150((__int64)&v315, v316, 0, 16, (int)v164, (int)v179);
        v192 = v315.m128i_u32[2];
      }
      a8 = (__m128)_mm_loadu_si128(v299);
      *(__m128 *)(v315.m128i_i64[0] + 16 * v192) = a8;
      ++v315.m128i_i32[2];
LABEL_307:
      if ( v293 != ++v299 )
        continue;
      break;
    }
LABEL_308:
    sub_197D1C0(dest);
    if ( v356 != (const char *)v355 )
      _libc_free((unsigned __int64)v356);
    if ( v315.m128i_u32[2] > (unsigned int)dword_4FB09A0 * (unsigned __int64)LODWORD(v314[1])
      || *(_DWORD *)(sub_1458800(*v339) + 48) > (unsigned int)dword_4FB08C0 )
    {
      goto LABEL_314;
    }
    if ( !v315.m128i_i32[2] )
    {
      v213 = sub_1458800(*v339);
      if ( sub_1452CB0(v213) )
        goto LABEL_398;
    }
    v193 = *(_QWORD *)(**(_QWORD **)(v333 + 32) + 56LL) + 112LL;
    if ( (unsigned __int8)sub_1560180(v193, 34)
      || (unsigned __int8)sub_1560180(v193, 17)
      || !(unsigned __int8)sub_13FCBF0(v333) )
    {
      goto LABEL_314;
    }
    sub_1B1E040((unsigned int)&v354, (_DWORD)v339, v333, v338, v340, v348, 0);
    v274 = v315.m128i_i32[2];
    v322.m128i_i64[0] = (__int64)&dest;
    v322.m128i_i64[1] = 0x400000000LL;
    if ( v315.m128i_i32[2] )
    {
      if ( (unsigned __int64 *)v315.m128i_i64[0] != v316 )
      {
        v322 = v315;
        v315.m128i_i64[0] = (__int64)v316;
        v315.m128i_i64[1] = 0;
        goto LABEL_476;
      }
      if ( v315.m128i_i32[2] <= 4u )
      {
        v283 = v316;
        v284 = 16LL * v315.m128i_u32[2];
        p_dest = &dest;
        goto LABEL_495;
      }
      sub_16CD150((__int64)&v322, &dest, v315.m128i_u32[2], 16, v272, v273);
      p_dest = (unsigned __int64 *)v322.m128i_i64[0];
      v283 = (unsigned __int64 *)v315.m128i_i64[0];
      v284 = 16LL * v315.m128i_u32[2];
      if ( v284 )
LABEL_495:
        memcpy(p_dest, v283, v284);
      v322.m128i_i32[2] = v274;
      v315.m128i_i32[2] = 0;
    }
LABEL_476:
    sub_1B1DC30(&v354, &v322);
    if ( (unsigned __int64 *)v322.m128i_i64[0] != &dest )
      _libc_free(v322.m128i_u64[0]);
    v275 = (const __m128i *)sub_1458800(*v339);
    sub_197E390(&v322, v275, v276, v277, v278, v279);
    sub_1B1DDA0(&v354, &v322);
    v322.m128i_i64[0] = (__int64)&unk_49EC708;
    if ( v332 )
    {
      v280 = v331;
      v281 = &v331[7 * v332];
      do
      {
        if ( *v280 != -16 && *v280 != -8 )
        {
          v282 = v280[1];
          if ( (_QWORD *)v282 != v280 + 3 )
            _libc_free(v282);
        }
        v280 += 7;
      }
      while ( v281 != v280 );
    }
    j___libc_free_0(v331);
    if ( (char *)v326[0] != &v327 )
      _libc_free(v326[0]);
    sub_1B17630(&v322, v354);
    sub_1B1F0F0(&v354, &v322);
    if ( (unsigned __int64 *)v322.m128i_i64[0] != &dest )
      _libc_free(v322.m128i_u64[0]);
    sub_197E0D0((__int64)&v354);
LABEL_398:
    v214 = sub_157EB90(**(_QWORD **)(v333 + 32));
    v215 = sub_1632FA0(v214);
    v357 = 0;
    v216 = v215;
    v356 = "storeforward";
    v374 = v377;
    v354 = (unsigned __int64 *)v348;
    v375[0] = (unsigned __int64)v377;
    v384 = 1;
    v355 = v215;
    v358 = 0;
    v359 = 0;
    v360 = 0;
    v361 = 0;
    v362 = 0;
    v363 = 0;
    v364 = 0;
    v365 = 0;
    v366 = 0;
    v367 = 0;
    v368 = 0;
    v369 = 0;
    v370 = 0;
    v371 = 0;
    LODWORD(v372) = 0;
    v373 = 0;
    v375[1] = 2;
    v376 = 0;
    v378 = 0;
    v379 = 0;
    v380 = 0;
    v381 = 0;
    v382 = 0;
    v383 = 0;
    v217 = sub_15E0530(*(_QWORD *)(v348 + 24));
    v218 = (_QWORD *)v314[0];
    memset(v385, 0, 24);
    v385[3] = v217;
    v390 = v392;
    v391 = 0x800000000LL;
    v385[4] = 0;
    v386 = 0;
    v387 = 0;
    v388 = 0;
    v389 = v216;
    v309 = v314[0] + 16LL * LODWORD(v314[1]);
    if ( v314[0] != v309 )
    {
      do
      {
        v245 = *(_QWORD **)(*v218 - 24LL);
        v246 = sub_1494E70((__int64)v341, (__int64)v245, a5, a6);
        v247 = sub_13FC520(v333);
        v248 = sub_157EBA0(v247);
        v249 = sub_38767A0(&v354, *(_QWORD *)v246[4], *v245, v248);
        LOWORD(dest) = 259;
        v322.m128i_i64[0] = (__int64)"load_initial";
        v250 = 1 << (*(unsigned __int16 *)(*v218 + 18LL) >> 1) >> 1;
        v303 = sub_157EBA0(v247);
        v251 = sub_1648A60(64, 1u);
        if ( v251 )
          sub_15F90A0((__int64)v251, *(_QWORD *)(*(_QWORD *)v249 + 24LL), v249, (__int64)&v322, 0, v250, v303);
        v252 = *(_QWORD *)(**(_QWORD **)(v333 + 32) + 48LL);
        LOWORD(dest) = 259;
        if ( v252 )
          v252 -= 24;
        v322.m128i_i64[0] = (__int64)"store_forwarded";
        v300 = v252;
        v304 = *v251;
        v253 = sub_1648B60(64);
        v257 = v304;
        v258 = v300;
        v259 = v253;
        if ( v253 )
        {
          v260 = v253;
          sub_15F1EA0(v253, v304, 53, 0, 0, v300);
          *(_DWORD *)(v259 + 56) = 2;
          sub_164B780(v259, v322.m128i_i64);
          v257 = *(unsigned int *)(v259 + 56);
          sub_1648880(v259, v257, 1);
        }
        else
        {
          v260 = 0;
        }
        v261 = *(_DWORD *)(v259 + 20) & 0xFFFFFFF;
        if ( v261 == *(_DWORD *)(v259 + 56) )
        {
          sub_15F55D0(v259, v257, v254, v255, v256, v258);
          v261 = *(_DWORD *)(v259 + 20) & 0xFFFFFFF;
        }
        v262 = (v261 + 1) & 0xFFFFFFF;
        v263 = v262 | *(_DWORD *)(v259 + 20) & 0xF0000000;
        *(_DWORD *)(v259 + 20) = v263;
        if ( (v263 & 0x40000000) != 0 )
          v219 = *(_QWORD *)(v259 - 8);
        else
          v219 = v260 - 24 * v262;
        v220 = (__int64 **)(v219 + 24LL * (unsigned int)(v262 - 1));
        if ( *v220 )
        {
          v221 = v220[1];
          v222 = (unsigned __int64)v220[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v222 = v221;
          if ( v221 )
            v221[2] = v222 | v221[2] & 3;
        }
        *v220 = v251;
        v223 = v251[1];
        v220[1] = (__int64 *)v223;
        if ( v223 )
          *(_QWORD *)(v223 + 16) = (unsigned __int64)(v220 + 1) | *(_QWORD *)(v223 + 16) & 3LL;
        v220[2] = (__int64 *)((unsigned __int64)v220[2] & 3 | (unsigned __int64)(v251 + 1));
        v251[1] = (__int64)v220;
        v224 = *(_DWORD *)(v259 + 20) & 0xFFFFFFF;
        if ( (*(_BYTE *)(v259 + 23) & 0x40) != 0 )
          v225 = *(_QWORD *)(v259 - 8);
        else
          v225 = v260 - 24 * v224;
        v226 = *(unsigned int *)(v259 + 56);
        *(_QWORD *)(v225 + 8LL * (unsigned int)(v224 - 1) + 24 * v226 + 8) = v247;
        v231 = sub_13FCB50(v333);
        v234 = *(_QWORD *)(v218[1] - 48LL);
        v235 = *(_DWORD *)(v259 + 20) & 0xFFFFFFF;
        if ( v235 == *(_DWORD *)(v259 + 56) )
        {
          sub_15F55D0(v259, v226, v227, v228, v229, v230);
          v235 = *(_DWORD *)(v259 + 20) & 0xFFFFFFF;
        }
        v236 = (v235 + 1) & 0xFFFFFFF;
        v237 = v236 | *(_DWORD *)(v259 + 20) & 0xF0000000;
        *(_DWORD *)(v259 + 20) = v237;
        if ( (v237 & 0x40000000) != 0 )
          v238 = *(_QWORD *)(v259 - 8);
        else
          v238 = v260 - 24 * v236;
        v239 = (_QWORD *)(v238 + 24LL * (unsigned int)(v236 - 1));
        if ( *v239 )
        {
          v240 = v239[1];
          v241 = v239[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v241 = v240;
          if ( v240 )
            *(_QWORD *)(v240 + 16) = v241 | *(_QWORD *)(v240 + 16) & 3LL;
        }
        *v239 = v234;
        if ( v234 )
        {
          v242 = *(_QWORD *)(v234 + 8);
          v239[1] = v242;
          if ( v242 )
            *(_QWORD *)(v242 + 16) = (unsigned __int64)(v239 + 1) | *(_QWORD *)(v242 + 16) & 3LL;
          v239[2] = (v234 + 8) | v239[2] & 3LL;
          *(_QWORD *)(v234 + 8) = v239;
        }
        v243 = *(_DWORD *)(v259 + 20) & 0xFFFFFFF;
        if ( (*(_BYTE *)(v259 + 23) & 0x40) != 0 )
          v244 = *(_QWORD *)(v259 - 8);
        else
          v244 = v260 - 24 * v243;
        v218 += 2;
        *(_QWORD *)(v244 + 8LL * (unsigned int)(v243 - 1) + 24LL * *(unsigned int *)(v259 + 56) + 8) = v231;
        sub_164D160(
          *(v218 - 2),
          v259,
          (__m128)a5,
          *(double *)a6.m128i_i64,
          *(double *)a7.m128_u64,
          *(double *)a8.m128_u64,
          v232,
          v233,
          a11,
          a12);
      }
      while ( (_QWORD *)v309 != v218 );
      if ( v390 != v392 )
        _libc_free((unsigned __int64)v390);
      if ( v385[0] )
        sub_161E7C0((__int64)v385, v385[0]);
    }
    j___libc_free_0(v381);
    if ( (_BYTE *)v375[0] != v374 )
      _libc_free(v375[0]);
    j___libc_free_0(v370);
    j___libc_free_0(v366);
    j___libc_free_0(v362);
    if ( v360 )
    {
      v265 = v358;
      v266 = &v358[5 * v360];
      do
      {
        while ( *v265 == -8 )
        {
          if ( v265[1] != -8 )
            goto LABEL_455;
          v265 += 5;
          if ( v266 == v265 )
            goto LABEL_462;
        }
        if ( *v265 != -16 || v265[1] != -16 )
        {
LABEL_455:
          v267 = v265[4];
          if ( v267 != 0 && v267 != -8 && v267 != -16 )
            sub_1649B30(v265 + 2);
        }
        v265 += 5;
      }
      while ( v266 != v265 );
    }
LABEL_462:
    j___libc_free_0(v358);
    v296 = v294;
LABEL_314:
    if ( (unsigned __int64 *)v315.m128i_i64[0] != v316 )
      _libc_free(v315.m128i_u64[0]);
LABEL_175:
    if ( (unsigned __int64 *)v314[0] != &v314[2] )
      _libc_free(v314[0]);
    v108 = v310;
    while ( v108 )
    {
      v109 = v108;
      v108 = (_QWORD *)*v108;
      j_j___libc_free_0(v109, 24);
    }
LABEL_179:
    v349 = &unk_49EC708;
    if ( v353 )
    {
      v110 = v352;
      v111 = &v352[7 * v353];
      do
      {
        if ( *v110 != -8 && *v110 != -16 )
        {
          v112 = v110[1];
          if ( (_QWORD *)v112 != v110 + 3 )
            _libc_free(v112);
        }
        v110 += 7;
      }
      while ( v111 != v110 );
    }
    j___libc_free_0(v352);
    if ( v350 != &v351 )
      _libc_free((unsigned __int64)v350);
    if ( v347 )
    {
      if ( v346 )
      {
        v121 = v345;
        v122 = &v345[2 * v346];
        do
        {
          if ( *v121 != -8 && *v121 != -4 )
          {
            v123 = v121[1];
            if ( v123 )
              sub_161E7C0((__int64)(v121 + 1), v123);
          }
          v121 += 2;
        }
        while ( v122 != v121 );
      }
      j___libc_free_0(v345);
    }
    if ( (_DWORD)v344 )
    {
      v113 = v342;
      v322.m128i_i64[1] = 2;
      dest = 0;
      v114 = &v342[6 * (unsigned int)v344];
      v324 = -8;
      v322.m128i_i64[0] = (__int64)&unk_49EC740;
      v325 = 0;
      v355 = 2;
      v356 = 0;
      v357 = -16;
      v354 = (unsigned __int64 *)&unk_49EC740;
      v358 = 0;
      do
      {
        v115 = v113[3];
        *v113 = (__int64)&unk_49EE2B0;
        if ( v115 != 0 && v115 != -8 && v115 != -16 )
          sub_1649B30(v113 + 1);
        v113 += 6;
      }
      while ( v114 != v113 );
      v354 = (unsigned __int64 *)&unk_49EE2B0;
      if ( v357 != 0 && v357 != -8 && v357 != -16 )
        sub_1649B30(&v355);
      v322.m128i_i64[0] = (__int64)&unk_49EE2B0;
      if ( v324 != 0 && v324 != -8 && v324 != -16 )
        sub_1649B30(&v322.m128i_i64[1]);
    }
    j___libc_free_0(v342);
    j___libc_free_0(v341[0].m128i_i64[1]);
    j___libc_free_0(v335);
    ++v297;
  }
  while ( v295 != v297 );
  v295 = v311;
LABEL_203:
  if ( v295 != (__int64 *)v313 )
    _libc_free((unsigned __int64)v295);
  return v296;
}
