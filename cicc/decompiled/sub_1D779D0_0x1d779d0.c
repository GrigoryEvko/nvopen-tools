// Function: sub_1D779D0
// Address: 0x1d779d0
//
__int64 __fastcall sub_1D779D0(
        __int64 a1,
        __int64 a2,
        _BYTE *a3,
        __m128 a4,
        __m128 a5,
        __m128 a6,
        __m128 a7,
        double a8,
        double a9,
        __m128i a10,
        __m128 a11)
{
  double v11; // xmm4_8
  double v12; // xmm5_8
  __m128i *v13; // r15
  _BOOL4 v16; // eax
  __int64 v17; // r8
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 v20; // r14
  int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rsi
  double v25; // xmm4_8
  double v26; // xmm5_8
  unsigned int v27; // r14d
  __int64 *v29; // rsi
  __int64 v30; // r12
  unsigned int v31; // eax
  __int64 v32; // rdx
  unsigned int v33; // ebx
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rsi
  __int64 v37; // r9
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rax
  _WORD *v41; // rcx
  __m128i *v42; // rbx
  signed __int64 v43; // rdx
  __int64 *v44; // r12
  __int64 v45; // rcx
  __int64 v46; // rdx
  _WORD *v47; // r14
  __int64 v48; // r12
  unsigned __int8 v49; // bl
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // r10
  __int64 v53; // rax
  __int64 v54; // rcx
  _BYTE *v55; // r9
  __int64 v56; // rdx
  __int64 v57; // rsi
  __int8 *v58; // rax
  _QWORD *v59; // rbx
  __int64 v60; // rax
  __int64 v61; // rdi
  _QWORD *v62; // r12
  __int64 (*v63)(); // rax
  __int64 v64; // r13
  __int64 i; // r13
  _QWORD *v66; // rax
  char v67; // al
  _QWORD *v68; // r13
  int v69; // ebx
  __int64 v70; // r14
  __int64 v71; // rdx
  __int64 *v72; // rbx
  __int64 v73; // rcx
  __int64 v74; // rsi
  __int64 v75; // rax
  int v76; // edx
  __int64 v77; // r8
  __int64 v78; // rdx
  _QWORD *v79; // rax
  __int64 v80; // r10
  __int64 v81; // rdx
  bool v82; // al
  __int64 v83; // r12
  __int64 v84; // r12
  _QWORD *v85; // rax
  unsigned __int64 *j; // rdx
  int v87; // r8d
  int v88; // r9d
  __int64 v89; // rsi
  __int64 v90; // rdx
  __int64 v91; // rcx
  __int64 v92; // r8
  __int64 v93; // r9
  __int64 k; // r14
  __int64 v95; // rsi
  char v96; // dl
  int v97; // eax
  __int64 v98; // r15
  _QWORD *v99; // rax
  int v100; // r8d
  int v101; // r9d
  unsigned int v102; // ebx
  unsigned __int64 v103; // rax
  int v104; // eax
  int v105; // eax
  unsigned int v106; // r12d
  __int64 *v107; // rax
  __int64 v108; // rbx
  __int64 ****v109; // rax
  double v110; // xmm4_8
  double v111; // xmm5_8
  __int64 v112; // rax
  unsigned __int8 v113; // al
  unsigned __int8 v114; // r10
  __int64 v115; // rax
  __int64 (*v116)(); // rax
  char v117; // al
  unsigned __int8 *v118; // r15
  unsigned __int8 *v119; // r14
  __int64 v120; // rdi
  __int64 v121; // rsi
  __m128i *v122; // r12
  unsigned __int8 *v123; // rbx
  unsigned __int8 *v124; // r12
  __int64 v125; // rdi
  __int64 v126; // r13
  __int64 (*v127)(); // rax
  __int64 v128; // rbx
  char v129; // r14
  unsigned __int64 v130; // rcx
  _QWORD *v131; // r12
  _QWORD *v132; // rax
  __int64 v133; // r8
  unsigned int v134; // edi
  __int64 *v135; // rax
  __int64 v136; // rcx
  __int64 v137; // rdx
  __int64 v138; // rax
  __int64 v139; // rax
  double v140; // xmm4_8
  double v141; // xmm5_8
  __int64 v142; // r14
  unsigned __int64 v143; // r12
  unsigned int v144; // edx
  char v145; // cl
  unsigned __int64 v146; // rdx
  __int64 v147; // r12
  _WORD *v148; // rbx
  unsigned __int64 *v149; // rax
  unsigned __int64 v150; // rdx
  unsigned int v151; // esi
  __int64 v152; // r8
  unsigned int v153; // ecx
  unsigned __int64 *v154; // rax
  unsigned __int64 v155; // rdi
  __int64 v156; // rax
  __int64 v157; // r12
  __int64 v158; // r14
  __int64 v159; // rbx
  __int64 ***v160; // rdx
  __int64 **v161; // rax
  __int64 v162; // rcx
  int v163; // r8d
  unsigned __int8 *v164; // rbx
  unsigned __int8 *v165; // r12
  __int64 v166; // rdi
  int v167; // edx
  __int64 v168; // rax
  __int64 v169; // r15
  __int64 v170; // r14
  unsigned __int64 *v171; // rax
  unsigned __int64 v172; // rdx
  unsigned int v173; // esi
  __int64 v174; // r8
  unsigned int v175; // ecx
  unsigned __int64 *v176; // rax
  unsigned __int64 v177; // rdi
  unsigned int v178; // esi
  unsigned __int64 v179; // rdx
  int v180; // r8d
  __int64 v181; // rdi
  unsigned int v182; // eax
  __int64 v183; // r9
  __int64 v184; // rbx
  __int64 v185; // rcx
  __int64 v186; // rax
  _QWORD *v187; // rax
  __int64 v188; // r12
  unsigned __int64 v189; // rdi
  __int64 v190; // rdx
  __int64 v191; // rax
  __int64 *v192; // rbx
  __int64 *v193; // rax
  __int64 v194; // rsi
  __int64 *v195; // r12
  __int64 **v196; // rax
  int v197; // r11d
  __int64 *v198; // r14
  int v199; // ecx
  __int64 v200; // rax
  __int16 v201; // si
  int v202; // edi
  __int64 v203; // r9
  __int64 v204; // rax
  __int64 v205; // rcx
  __int64 v206; // rdx
  __int64 v207; // rax
  __int64 v208; // rcx
  __int64 v209; // rdi
  __int64 v210; // rsi
  unsigned __int8 *v211; // rsi
  unsigned int v212; // eax
  __int64 v213; // rdi
  int v214; // r10d
  __int64 *v215; // r9
  int v216; // r10d
  int v217; // eax
  int v218; // ecx
  int v219; // r11d
  unsigned __int64 *v220; // r10
  int v221; // ebx
  int v222; // edi
  int v223; // r10d
  unsigned int v224; // edx
  __int64 v225; // rsi
  int v226; // r11d
  unsigned __int64 *v227; // r10
  int v228; // ecx
  int v229; // edi
  __int64 v230; // rsi
  unsigned __int8 *v231; // rsi
  unsigned __int64 v232; // rbx
  _QWORD *v233; // rax
  __int64 v234; // r14
  __int64 v235; // rsi
  unsigned int v236; // eax
  __int64 *v237; // r12
  __int64 *v238; // rbx
  unsigned __int8 v239; // al
  unsigned int v240; // eax
  __int64 v241; // r9
  __int64 v242; // rdx
  unsigned __int64 *v243; // rax
  __int64 v244; // rdx
  __int64 v245; // rcx
  __int64 (*v246)(); // rax
  __int64 v247; // rax
  unsigned __int8 *v248; // rsi
  __int64 v249; // rax
  __int64 v250; // rax
  char v251; // al
  __int64 v252; // rsi
  unsigned __int64 v253; // rdx
  __int64 v254; // rcx
  __int64 v255; // r8
  __int64 v256; // r9
  __int64 v257; // rax
  __int64 v258; // rdx
  __m128i *v259; // rax
  __m128i *v260; // rbx
  __int64 v261; // rsi
  __int64 m128i_i64; // rdx
  __int64 v263; // rsi
  unsigned __int8 *v264; // rsi
  double v265; // xmm4_8
  double v266; // xmm5_8
  __int64 *v267; // rax
  __int64 v268; // r9
  char v269; // al
  __int64 v270; // r14
  __int64 v271; // r15
  __int64 v272; // rdi
  char v273; // cl
  __int64 *v274; // r14
  __int64 *v275; // rbx
  unsigned __int64 *v276; // rax
  unsigned __int64 v277; // rdx
  unsigned int v278; // esi
  __int64 v279; // r8
  unsigned int v280; // ecx
  unsigned __int64 *v281; // rax
  unsigned __int64 v282; // rdi
  unsigned int v283; // esi
  unsigned __int64 v284; // rdx
  __int64 v285; // r8
  unsigned int v286; // eax
  unsigned __int64 *v287; // rdi
  const void *v288; // rcx
  __int64 v289; // rax
  unsigned __int64 *v290; // r15
  __int64 v291; // r14
  __int64 v292; // rdi
  int v293; // r11d
  unsigned __int64 *v294; // r10
  int v295; // eax
  int v296; // ecx
  int v297; // r11d
  unsigned __int64 *v298; // r10
  int v299; // ecx
  int v300; // edi
  __int64 v301; // rdi
  _QWORD *v302; // rax
  int v303; // r11d
  __int64 *v304; // r8
  unsigned int v305; // ebx
  unsigned __int64 *v306; // rax
  __int64 *v307; // r14
  __int64 *v308; // rsi
  __int32 v309; // eax
  __int64 v310; // rdx
  __int64 v311; // rcx
  __int64 v312; // r8
  __int64 v313; // r9
  unsigned __int8 v314; // r12
  unsigned int v315; // ebx
  __int64 v316; // rdx
  __int64 v317; // rcx
  __int64 v318; // r8
  __int64 v319; // r9
  __int64 v320; // rdx
  __int64 v321; // rcx
  __int64 v322; // r8
  __int64 v323; // r9
  unsigned int v324; // eax
  _QWORD *v325; // rax
  __int64 v326; // rsi
  __int64 v327; // rbx
  unsigned __int8 v328; // al
  __int64 v329; // r12
  double v330; // xmm4_8
  double v331; // xmm5_8
  __m128i **v332; // rax
  __m128i *v333; // rsi
  unsigned __int64 v334; // rdi
  __m128i *v335; // rsi
  __int64 *v336; // rbx
  double v337; // xmm4_8
  double v338; // xmm5_8
  _QWORD *v339; // r14
  __int64 v340; // rax
  _QWORD *v341; // rax
  __int64 *v342; // rbx
  __int64 v343; // rax
  __int64 v344; // rsi
  unsigned int v345; // eax
  __int64 v346; // rdx
  unsigned int v347; // r14d
  int v349; // [rsp+0h] [rbp-360h]
  int v350; // [rsp+0h] [rbp-360h]
  int v351; // [rsp+0h] [rbp-360h]
  __int64 v352; // [rsp+8h] [rbp-358h]
  __int64 v353; // [rsp+8h] [rbp-358h]
  __int64 v354; // [rsp+8h] [rbp-358h]
  int v355; // [rsp+10h] [rbp-350h]
  __int64 *v356; // [rsp+10h] [rbp-350h]
  int v357; // [rsp+10h] [rbp-350h]
  int v358; // [rsp+10h] [rbp-350h]
  __int64 v359; // [rsp+18h] [rbp-348h]
  __int64 v360; // [rsp+18h] [rbp-348h]
  __int64 v361; // [rsp+18h] [rbp-348h]
  __int64 v362; // [rsp+28h] [rbp-338h]
  unsigned __int8 v363; // [rsp+28h] [rbp-338h]
  __int64 v364; // [rsp+28h] [rbp-338h]
  unsigned __int8 v365; // [rsp+30h] [rbp-330h]
  __int64 v366; // [rsp+30h] [rbp-330h]
  __int64 v367; // [rsp+30h] [rbp-330h]
  __int64 v368; // [rsp+30h] [rbp-330h]
  char v369; // [rsp+30h] [rbp-330h]
  __int64 v370; // [rsp+30h] [rbp-330h]
  unsigned __int8 v371; // [rsp+38h] [rbp-328h]
  __int64 v372; // [rsp+38h] [rbp-328h]
  __int64 v373; // [rsp+38h] [rbp-328h]
  __int64 v374; // [rsp+38h] [rbp-328h]
  __m128i *v375; // [rsp+58h] [rbp-308h]
  unsigned __int8 v376; // [rsp+68h] [rbp-2F8h]
  unsigned int v377; // [rsp+68h] [rbp-2F8h]
  unsigned __int8 v378; // [rsp+70h] [rbp-2F0h]
  __int64 *v379; // [rsp+70h] [rbp-2F0h]
  int v380; // [rsp+70h] [rbp-2F0h]
  _QWORD *v381; // [rsp+70h] [rbp-2F0h]
  _QWORD *v382; // [rsp+70h] [rbp-2F0h]
  __int64 v383; // [rsp+78h] [rbp-2E8h]
  unsigned __int8 v384; // [rsp+78h] [rbp-2E8h]
  __int64 v385; // [rsp+78h] [rbp-2E8h]
  unsigned __int8 v386; // [rsp+78h] [rbp-2E8h]
  __int64 v387; // [rsp+78h] [rbp-2E8h]
  __m128i *v388; // [rsp+78h] [rbp-2E8h]
  __m128i *v389; // [rsp+80h] [rbp-2E0h]
  __int64 v390; // [rsp+80h] [rbp-2E0h]
  _QWORD *v391; // [rsp+80h] [rbp-2E0h]
  unsigned __int8 v392; // [rsp+80h] [rbp-2E0h]
  unsigned __int8 v393; // [rsp+80h] [rbp-2E0h]
  __int64 v394; // [rsp+80h] [rbp-2E0h]
  __int64 v395; // [rsp+80h] [rbp-2E0h]
  unsigned int v396; // [rsp+80h] [rbp-2E0h]
  __int64 v397; // [rsp+80h] [rbp-2E0h]
  __int64 v398; // [rsp+80h] [rbp-2E0h]
  unsigned __int8 v399; // [rsp+80h] [rbp-2E0h]
  __int64 v400; // [rsp+88h] [rbp-2D8h]
  __int64 v401; // [rsp+88h] [rbp-2D8h]
  __int64 v402; // [rsp+88h] [rbp-2D8h]
  __int64 v403; // [rsp+88h] [rbp-2D8h]
  unsigned int v404; // [rsp+88h] [rbp-2D8h]
  __int64 v405; // [rsp+88h] [rbp-2D8h]
  __int64 *v406; // [rsp+88h] [rbp-2D8h]
  unsigned __int8 v407; // [rsp+97h] [rbp-2C9h] BYREF
  __int64 v408; // [rsp+98h] [rbp-2C8h] BYREF
  __int64 v409; // [rsp+A0h] [rbp-2C0h] BYREF
  __int64 v410; // [rsp+A8h] [rbp-2B8h]
  unsigned __int64 v411; // [rsp+B0h] [rbp-2B0h] BYREF
  unsigned int v412; // [rsp+B8h] [rbp-2A8h]
  unsigned __int64 *v413; // [rsp+C0h] [rbp-2A0h] BYREF
  unsigned int v414; // [rsp+C8h] [rbp-298h]
  __m128i v415; // [rsp+D0h] [rbp-290h] BYREF
  _QWORD v416[2]; // [rsp+E0h] [rbp-280h] BYREF
  __int64 v417[2]; // [rsp+F0h] [rbp-270h] BYREF
  _WORD v418[8]; // [rsp+100h] [rbp-260h] BYREF
  _WORD *v419; // [rsp+110h] [rbp-250h] BYREF
  __int64 v420; // [rsp+118h] [rbp-248h]
  _WORD v421[16]; // [rsp+120h] [rbp-240h] BYREF
  __m128 v422; // [rsp+140h] [rbp-220h] BYREF
  _QWORD v423[8]; // [rsp+150h] [rbp-210h] BYREF
  unsigned __int64 *v424; // [rsp+190h] [rbp-1D0h] BYREF
  __int64 v425; // [rsp+198h] [rbp-1C8h]
  __int64 *v426; // [rsp+1A0h] [rbp-1C0h] BYREF
  __int64 v427; // [rsp+1A8h] [rbp-1B8h]
  int v428; // [rsp+1B0h] [rbp-1B0h]
  _BYTE v429[40]; // [rsp+1B8h] [rbp-1A8h] BYREF
  __m128i *v430; // [rsp+1E0h] [rbp-180h] BYREF
  __int64 v431; // [rsp+1E8h] [rbp-178h]
  __m128i *v432; // [rsp+1F0h] [rbp-170h] BYREF
  __int64 *v433; // [rsp+1F8h] [rbp-168h]
  __int64 v434; // [rsp+270h] [rbp-F0h]
  __m128i v435; // [rsp+280h] [rbp-E0h] BYREF
  unsigned __int64 v436; // [rsp+290h] [rbp-D0h] BYREF
  __int64 v437; // [rsp+298h] [rbp-C8h]
  __int64 v438; // [rsp+2A0h] [rbp-C0h]
  int v439; // [rsp+2A8h] [rbp-B8h] BYREF
  __int64 v440; // [rsp+2B0h] [rbp-B0h]
  __int64 v441; // [rsp+2B8h] [rbp-A8h]
  __int64 v442; // [rsp+310h] [rbp-50h]

  v13 = (__m128i *)a2;
  v400 = a1 + 320;
  v16 = sub_13A0E30(a1 + 320, a2);
  v20 = v16;
  if ( v16 )
    return 0;
  v21 = *(unsigned __int8 *)(a2 + 16);
  if ( (_BYTE)v21 == 77 )
  {
    v22 = *(_QWORD *)(a1 + 904);
    v23 = *(_QWORD *)(a1 + 200);
    v436 = 0;
    v435.m128i_i64[0] = v22;
    v435.m128i_i64[1] = v23;
    v437 = 0;
    v438 = 0;
    v24 = sub_13E3350(a2, &v435, 0, 1, v17);
    if ( v24 )
    {
      v27 = 1;
      sub_164D160(
        (__int64)v13,
        v24,
        a4,
        *(double *)a5.m128_u64,
        *(double *)a6.m128_u64,
        *(double *)a7.m128_u64,
        v25,
        v26,
        *(double *)a10.m128i_i64,
        a11);
      sub_15F20C0(v13);
      return v27;
    }
    return 0;
  }
  if ( (unsigned int)(unsigned __int8)v21 - 60 <= 0xC )
  {
    if ( *(_BYTE *)(*(_QWORD *)(a2 - 24) + 16LL) <= 0x10u )
      return 0;
    v29 = *(__int64 **)(a1 + 176);
    if ( !v29 )
    {
      if ( (unsigned __int8)(v21 - 61) > 1u )
        return 0;
      v389 = v13;
      v27 = 0;
      goto LABEL_45;
    }
    v27 = sub_1D69880(v13, v29, *(_QWORD *)(a1 + 904));
    if ( (_BYTE)v27 )
      return 1;
    if ( (unsigned __int8)(v13[1].m128i_i8[0] - 61) > 1u )
      return 0;
    v30 = *(_QWORD *)(a1 + 176);
    v389 = v13;
    if ( !v30 )
      goto LABEL_45;
    v31 = sub_1D5D7E0(*(_QWORD *)(a1 + 904), (__int64 *)v13->m128i_i64[0], 0);
    v383 = v32;
    v33 = v31;
    v34 = sub_16498A0((__int64)v13);
    sub_1F40D10(&v435, v30, v34, v33, v383);
    if ( v435.m128i_i8[0] == 2 )
      return sub_1D69330((__int64)v13);
    if ( !*(_QWORD *)(a1 + 176) )
    {
LABEL_45:
      if ( (v13[1].m128i_i8[7] & 0x40) != 0 )
        v58 = (__int8 *)v13[-1].m128i_i64[1];
      else
        v58 = &v389->m128i_i8[-24 * (v13[1].m128i_i32[1] & 0xFFFFFFF)];
      v59 = *(_QWORD **)v58;
      v60 = *(_QWORD *)(*(_QWORD *)v58 + 8LL);
      if ( v60 && !*(_QWORD *)(v60 + 8) )
        return v27;
      v61 = *(_QWORD *)(a1 + 176);
      v62 = (_QWORD *)v13[2].m128i_i64[1];
      if ( v61 )
      {
        v63 = *(__int64 (**)())(*(_QWORD *)v61 + 784LL);
        if ( v63 == sub_1D5A3F0
          || !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))v63)(v61, v13->m128i_i64[0], *v59) )
        {
          return v27;
        }
      }
      if ( *((_BYTE *)v59 + 16) <= 0x17u )
        return v27;
      if ( v62 != (_QWORD *)v59[5] )
        return v27;
      v64 = v13->m128i_i64[1];
      if ( !v64 )
        return v27;
      while ( v62 == (_QWORD *)sub_1648700(v64)[5] )
      {
        v64 = *(_QWORD *)(v64 + 8);
        if ( !v64 )
          return v27;
      }
      for ( i = v59[1]; i; i = *(_QWORD *)(i + 8) )
      {
        v66 = sub_1648700(i);
        if ( v62 != (_QWORD *)v66[5] )
        {
          v67 = *((_BYTE *)v66 + 16);
          if ( (unsigned __int8)(v67 - 54) <= 1u || v67 == 77 )
            return v27;
        }
      }
      v430 = 0;
      v431 = 0;
      v432 = 0;
      LODWORD(v433) = 0;
      v68 = (_QWORD *)v59[1];
      if ( !v68 )
      {
        v301 = 0;
        v69 = 0;
LABEL_482:
        j___libc_free_0(v301);
        v27 |= v69;
        return v27;
      }
      v379 = v59;
      v376 = v27;
      v69 = 0;
      v70 = 0;
      while ( 1 )
      {
        v75 = sub_1648700((__int64)v68)[5];
        v424 = (unsigned __int64 *)v75;
        if ( v62 == (_QWORD *)v75 )
          goto LABEL_66;
        if ( !(_DWORD)v433 )
          break;
        LODWORD(v71) = ((_DWORD)v433 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
        v72 = (__int64 *)(v70 + 16LL * (unsigned int)v71);
        v73 = *v72;
        if ( v75 != *v72 )
        {
          v303 = 1;
          v304 = 0;
          while ( v73 != -8 )
          {
            if ( v73 == -16 && !v304 )
              v304 = v72;
            v71 = ((_DWORD)v433 - 1) & (unsigned int)(v71 + v303);
            v72 = (__int64 *)(v70 + 16 * v71);
            v73 = *v72;
            if ( v75 == *v72 )
              goto LABEL_64;
            ++v303;
          }
          if ( v304 )
            v72 = v304;
          v430 = (__m128i *)((char *)v430 + 1);
          v76 = (_DWORD)v432 + 1;
          if ( 4 * ((int)v432 + 1) < (unsigned int)(3 * (_DWORD)v433) )
          {
            if ( (int)v433 - HIDWORD(v432) - v76 <= (unsigned int)v433 >> 3 )
            {
              sub_1CD3B30((__int64)&v430, (int)v433);
LABEL_71:
              sub_1CD3040((__int64)&v430, (__int64 *)&v424, &v435);
              v72 = (__int64 *)v435.m128i_i64[0];
              v75 = (__int64)v424;
              v76 = (_DWORD)v432 + 1;
            }
            LODWORD(v432) = v76;
            if ( *v72 != -8 )
              --HIDWORD(v432);
            *v72 = v75;
            v72[1] = 0;
LABEL_75:
            v77 = sub_157EE30((__int64)v424);
            v78 = *v379;
            LOWORD(v436) = 257;
            if ( v77 )
              v77 -= 24;
            v385 = v77;
            v390 = v78;
            v79 = sub_1648A60(56, 1u);
            v80 = (__int64)v79;
            if ( v79 )
            {
              v81 = v390;
              v391 = v79;
              sub_15FC510((__int64)v79, (__int64)v13, v81, (__int64)&v435, v385);
              v80 = (__int64)v391;
            }
            v72[1] = v80;
            sub_165A590((__int64)&v435, v400, v80);
            v74 = v72[1];
            goto LABEL_65;
          }
LABEL_70:
          sub_1CD3B30((__int64)&v430, 2 * (_DWORD)v433);
          goto LABEL_71;
        }
LABEL_64:
        v74 = v72[1];
        if ( !v74 )
          goto LABEL_75;
LABEL_65:
        v69 = 1;
        sub_1593B40(v68, v74);
        v70 = v431;
LABEL_66:
        v68 = (_QWORD *)v68[1];
        if ( !v68 )
        {
          v301 = v70;
          v27 = v376;
          goto LABEL_482;
        }
      }
      v430 = (__m128i *)((char *)v430 + 1);
      goto LABEL_70;
    }
    v35 = *(_QWORD *)(a1 + 192);
    v407 = 0;
    v36 = (__int64)&v430;
    v384 = sub_14A3230(v35, (__int64)v13, &v407);
    v434 = a1 + 520;
    v415.m128i_i64[0] = (__int64)v416;
    v419 = v421;
    v430 = (__m128i *)&v432;
    v420 = 0x200000000LL;
    v431 = 0x1000000000LL;
    v416[0] = v13;
    v415.m128i_i64[1] = 0x100000001LL;
    v378 = sub_1D61850(a1, (__int64)&v430, (__int64)&v415, (__int64)&v419, 0, v37);
    v40 = (__int64)v419;
    v41 = &v419[4 * (unsigned int)v420];
    if ( v419 != v41 )
    {
      while ( 1 )
      {
        v42 = *(__m128i **)v40;
        if ( (*(_BYTE *)(*(_QWORD *)v40 + 23LL) & 0x40) != 0 )
        {
          v43 = v42[-1].m128i_i64[1];
        }
        else
        {
          v36 = 24LL * (v42[1].m128i_i32[1] & 0xFFFFFFF);
          v43 = (signed __int64)v42->m128i_i64 - v36;
        }
        v44 = *(__int64 **)v43;
        if ( *(_BYTE *)(*(_QWORD *)v43 + 16LL) == 54 )
          break;
        v40 += 8;
        if ( v41 == (_WORD *)v40 )
          goto LABEL_20;
      }
      if ( v378 || v44[5] != v42[2].m128i_i64[1] )
      {
        v362 = *(_QWORD *)(a1 + 176);
        v366 = *(_QWORD *)(a1 + 904);
        v112 = sub_1D5D7E0(v366, (__int64 *)v42->m128i_i64[0], 0);
        v36 = *v44;
        v372 = v112;
        v113 = sub_1D5D7E0(v366, (__int64 *)*v44, 0);
        v39 = v372;
        v38 = v362;
        v114 = v113;
        v115 = v44[1];
        if ( v115 )
        {
          if ( !*(_QWORD *)(v115 + 8) )
            goto LABEL_143;
        }
        if ( (!v114 || !*(_QWORD *)(v362 + 8LL * v114 + 120))
          && (_BYTE)v372
          && *(_QWORD *)(v362 + 8LL * (unsigned __int8)v372 + 120) )
        {
          goto LABEL_143;
        }
        v116 = *(__int64 (**)())(*(_QWORD *)v362 + 784LL);
        if ( v116 != sub_1D5A3F0 )
        {
          v363 = v114;
          v367 = v372;
          v36 = v42->m128i_i64[0];
          v373 = v38;
          v117 = ((__int64 (__fastcall *)(__int64, __int64, __int64))v116)(v38, v42->m128i_i64[0], *v44);
          v38 = v373;
          v39 = v367;
          v114 = v363;
          if ( v117 )
          {
LABEL_143:
            if ( v114
              && (_BYTE)v39
              && (((int)*(unsigned __int16 *)(v38 + 2 * (v114 + 115LL * (unsigned __int8)v39 + 16104)) >> (4 * ((v42[1].m128i_i8[0] == 61) + 2)))
                & 0xF) == 0 )
            {
              v118 = (unsigned __int8 *)v430;
              v119 = (unsigned __int8 *)v430 + 8 * (unsigned int)v431;
              while ( v118 != v119 )
              {
                while ( 1 )
                {
                  v120 = *((_QWORD *)v119 - 1);
                  v119 -= 8;
                  if ( !v120 )
                    break;
                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v120 + 8LL))(v120);
                  if ( v118 == v119 )
                    goto LABEL_148;
                }
              }
LABEL_148:
              LODWORD(v431) = 0;
              sub_15F2300(v42, (__int64)v44);
              v121 = v44[6];
              v122 = v42 + 3;
              v435.m128i_i64[0] = v121;
              if ( v121 )
              {
                sub_1623A60((__int64)&v435, v121, 2);
                if ( v122 == &v435 )
                {
                  if ( v435.m128i_i64[0] )
                    sub_161E7C0((__int64)&v435, v435.m128i_i64[0]);
                  goto LABEL_152;
                }
                v230 = v42[3].m128i_i64[0];
                if ( !v230 )
                {
LABEL_351:
                  v231 = (unsigned __int8 *)v435.m128i_i64[0];
                  v42[3].m128i_i64[0] = v435.m128i_i64[0];
                  if ( v231 )
                    sub_1623210((__int64)&v435, v231, (__int64)v42[3].m128i_i64);
                  goto LABEL_152;
                }
              }
              else if ( v122 == &v435 || (v230 = v42[3].m128i_i64[0]) == 0 )
              {
LABEL_152:
                v389 = v42;
                v13 = v42;
                v27 = 1;
                goto LABEL_153;
              }
              sub_161E7C0((__int64)v42[3].m128i_i64, v230);
              goto LABEL_351;
            }
          }
        }
      }
    }
LABEL_20:
    v45 = v384;
    if ( !v384 )
      goto LABEL_168;
    v424 = 0;
    v428 = 0;
    v46 = (unsigned int)v420;
    v365 = v407;
    v425 = (__int64)v429;
    v426 = (__int64 *)v429;
    v427 = 1;
    if ( v419 == &v419[4 * (unsigned int)v420] )
    {
      if ( !v407 )
        goto LABEL_168;
      v386 = v407;
      if ( (_DWORD)v420 != 1 )
        goto LABEL_166;
    }
    else
    {
      v371 = v27;
      v47 = &v419[4 * (unsigned int)v420];
      v48 = (__int64)v419;
      v49 = v384;
      do
      {
        v53 = *(_QWORD *)v48;
        if ( (*(_BYTE *)(*(_QWORD *)v48 + 23LL) & 0x40) != 0 )
        {
          v46 = *(_QWORD *)(v53 - 8);
        }
        else
        {
          v36 = 24LL * (*(_DWORD *)(v53 + 20) & 0xFFFFFFF);
          v46 = v53 - v36;
        }
        v50 = *(unsigned int *)(a1 + 712);
        if ( (_DWORD)v50 )
        {
          v51 = *(_QWORD *)v46;
          v39 = (unsigned int)(v50 - 1);
          v38 = *(_QWORD *)(a1 + 696);
          v36 = (unsigned int)v39 & (((unsigned int)*(_QWORD *)v46 >> 9) ^ ((unsigned int)*(_QWORD *)v46 >> 4));
          v46 = v38 + 16 * v36;
          v52 = *(_QWORD *)v46;
          if ( v51 == *(_QWORD *)v46 )
          {
LABEL_26:
            if ( v46 != v38 + 16 * v50 )
            {
              v36 = *(_QWORD *)(v46 + 8);
              v49 = 0;
              if ( v36 )
                sub_1412190((__int64)&v424, v36);
            }
          }
          else
          {
            v46 = 1;
            while ( v52 != -8 )
            {
              v45 = (unsigned int)(v46 + 1);
              v36 = (unsigned int)v39 & ((_DWORD)v46 + (_DWORD)v36);
              v46 = v38 + 16LL * (unsigned int)v36;
              v52 = *(_QWORD *)v46;
              if ( v51 == *(_QWORD *)v46 )
                goto LABEL_26;
              v46 = (unsigned int)v45;
            }
          }
        }
        v48 += 8;
      }
      while ( v47 != (_WORD *)v48 );
      v386 = v49;
      v27 = v371;
      if ( v49 )
      {
        if ( !v365 || (_DWORD)v420 != 1 )
        {
          v147 = (__int64)v419;
          if ( v419 == &v419[4 * (unsigned int)v420] )
            goto LABEL_166;
          v148 = &v419[4 * (unsigned int)v420];
          while ( 1 )
          {
            v156 = *(_QWORD *)v147;
            v149 = (*(_BYTE *)(*(_QWORD *)v147 + 23LL) & 0x40) != 0
                 ? *(unsigned __int64 **)(v156 - 8)
                 : (unsigned __int64 *)(v156 - 24LL * (*(_DWORD *)(v156 + 20) & 0xFFFFFFF));
            v150 = *v149;
            v151 = *(_DWORD *)(a1 + 712);
            v422.m128_u64[0] = *v149;
            if ( !v151 )
              break;
            v152 = *(_QWORD *)(a1 + 696);
            v153 = (v151 - 1) & (((unsigned int)v150 >> 9) ^ ((unsigned int)v150 >> 4));
            v154 = (unsigned __int64 *)(v152 + 16LL * v153);
            v155 = *v154;
            if ( v150 != *v154 )
            {
              v226 = 1;
              v227 = 0;
              while ( v155 != -8 )
              {
                if ( !v227 && v155 == -16 )
                  v227 = v154;
                v153 = (v151 - 1) & (v226 + v153);
                v154 = (unsigned __int64 *)(v152 + 16LL * v153);
                v155 = *v154;
                if ( v150 == *v154 )
                  goto LABEL_207;
                ++v226;
              }
              v228 = *(_DWORD *)(a1 + 704);
              if ( v227 )
                v154 = v227;
              ++*(_QWORD *)(a1 + 688);
              v229 = v228 + 1;
              if ( 4 * (v228 + 1) < 3 * v151 )
              {
                if ( v151 - *(_DWORD *)(a1 + 708) - v229 > v151 >> 3 )
                {
LABEL_342:
                  *(_DWORD *)(a1 + 704) = v229;
                  if ( *v154 != -8 )
                    --*(_DWORD *)(a1 + 708);
                  *v154 = v150;
                  v154[1] = 0;
                  goto LABEL_207;
                }
LABEL_347:
                sub_1C29540(a1 + 688, v151);
                sub_1CD2C70(a1 + 688, (__int64 *)&v422, &v435);
                v154 = (unsigned __int64 *)v435.m128i_i64[0];
                v150 = v422.m128_u64[0];
                v229 = *(_DWORD *)(a1 + 704) + 1;
                goto LABEL_342;
              }
LABEL_346:
              v151 *= 2;
              goto LABEL_347;
            }
LABEL_207:
            v147 += 8;
            v154[1] = (unsigned __int64)v13;
            if ( v148 == (_WORD *)v147 )
            {
              v27 = v371;
LABEL_166:
              if ( v426 != (__int64 *)v425 )
                _libc_free((unsigned __int64)v426);
LABEL_168:
              sub_1D5ABA0((__int64 *)&v430, 0);
              goto LABEL_153;
            }
          }
          ++*(_QWORD *)(a1 + 688);
          goto LABEL_346;
        }
        v386 = v365;
      }
    }
    v164 = (unsigned __int8 *)v430;
    v165 = (unsigned __int8 *)v430 + 8 * (unsigned int)v431;
    while ( v164 != v165 )
    {
      while ( 1 )
      {
        v166 = *((_QWORD *)v165 - 1);
        v165 -= 8;
        if ( !v166 )
          break;
        (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v166 + 8LL))(
          v166,
          v36,
          v46,
          v45,
          v38,
          v39);
        if ( v164 == v165 )
          goto LABEL_233;
      }
    }
LABEL_233:
    LODWORD(v431) = 0;
    v167 = v420;
    v168 = 4LL * (unsigned int)v420;
    v169 = (__int64)&v419[v168];
    if ( v419 != &v419[v168] )
    {
      v393 = v27;
      v170 = (__int64)v419;
      v374 = a1 + 864;
      v368 = a1 + 688;
      while ( 1 )
      {
        v188 = *(_QWORD *)v170;
        v171 = (*(_BYTE *)(*(_QWORD *)v170 + 23LL) & 0x40) != 0
             ? *(unsigned __int64 **)(v188 - 8)
             : (unsigned __int64 *)(v188 - 24LL * (*(_DWORD *)(v188 + 20) & 0xFFFFFFF));
        v172 = *v171;
        v173 = *(_DWORD *)(a1 + 712);
        v422.m128_u64[0] = *v171;
        if ( !v173 )
          break;
        v174 = *(_QWORD *)(a1 + 696);
        v175 = (v173 - 1) & (((unsigned int)v172 >> 9) ^ ((unsigned int)v172 >> 4));
        v176 = (unsigned __int64 *)(v174 + 16LL * v175);
        v177 = *v176;
        if ( v172 == *v176 )
          goto LABEL_238;
        v219 = 1;
        v220 = 0;
        while ( v177 != -8 )
        {
          if ( v177 == -16 && !v220 )
            v220 = v176;
          v175 = (v173 - 1) & (v219 + v175);
          v176 = (unsigned __int64 *)(v174 + 16LL * v175);
          v177 = *v176;
          if ( v172 == *v176 )
            goto LABEL_238;
          ++v219;
        }
        v221 = *(_DWORD *)(a1 + 704);
        if ( v220 )
          v176 = v220;
        ++*(_QWORD *)(a1 + 688);
        v222 = v221 + 1;
        if ( 4 * (v221 + 1) >= 3 * v173 )
          goto LABEL_320;
        if ( v173 - *(_DWORD *)(a1 + 708) - v222 <= v173 >> 3 )
          goto LABEL_321;
LABEL_316:
        *(_DWORD *)(a1 + 704) = v222;
        if ( *v176 != -8 )
          --*(_DWORD *)(a1 + 708);
        *v176 = v172;
        v176[1] = 0;
LABEL_238:
        v176[1] = 0;
        v178 = *(_DWORD *)(a1 + 888);
        if ( !v178 )
        {
          ++*(_QWORD *)(a1 + 864);
          goto LABEL_308;
        }
        v179 = v422.m128_u64[0];
        v180 = v178 - 1;
        v181 = *(_QWORD *)(a1 + 872);
        v182 = (v178 - 1) & (((unsigned __int32)v422.m128_i32[0] >> 9) ^ ((unsigned __int32)v422.m128_i32[0] >> 4));
        LODWORD(v183) = 9 * v182;
        v184 = v181 + 152LL * v182;
        v185 = *(_QWORD *)v184;
        if ( v422.m128_u64[0] != *(_QWORD *)v184 )
        {
          v216 = 1;
          v183 = 0;
          while ( v185 != -8 )
          {
            if ( !v183 && v185 == -16 )
              v183 = v184;
            v182 = v180 & (v216 + v182);
            v184 = v181 + 152LL * v182;
            v185 = *(_QWORD *)v184;
            if ( v422.m128_u64[0] == *(_QWORD *)v184 )
              goto LABEL_240;
            ++v216;
          }
          v217 = *(_DWORD *)(a1 + 880);
          if ( v183 )
            v184 = v183;
          ++*(_QWORD *)(a1 + 864);
          v218 = v217 + 1;
          if ( 4 * (v217 + 1) < 3 * v178 )
          {
            if ( v178 - *(_DWORD *)(a1 + 884) - v218 > v178 >> 3 )
            {
LABEL_304:
              *(_DWORD *)(a1 + 880) = v218;
              if ( *(_QWORD *)v184 != -8 )
                --*(_DWORD *)(a1 + 884);
              v187 = (_QWORD *)(v184 + 24);
              *(_QWORD *)v184 = v179;
              *(_QWORD *)(v184 + 8) = v184 + 24;
              *(_QWORD *)(v184 + 16) = 0x1000000000LL;
              goto LABEL_242;
            }
LABEL_309:
            sub_1D6F390(v374, v178);
            sub_1D680B0(v374, (__int64 *)&v422, &v435);
            v184 = v435.m128i_i64[0];
            v179 = v422.m128_u64[0];
            v218 = *(_DWORD *)(a1 + 880) + 1;
            goto LABEL_304;
          }
LABEL_308:
          v178 *= 2;
          goto LABEL_309;
        }
LABEL_240:
        v186 = *(unsigned int *)(v184 + 16);
        if ( (unsigned int)v186 >= *(_DWORD *)(v184 + 20) )
        {
          sub_16CD150(v184 + 8, (const void *)(v184 + 24), 0, 8, v180, v183);
          v187 = (_QWORD *)(*(_QWORD *)(v184 + 8) + 8LL * *(unsigned int *)(v184 + 16));
        }
        else
        {
          v187 = (_QWORD *)(*(_QWORD *)(v184 + 8) + 8 * v186);
        }
LABEL_242:
        v170 += 8;
        *v187 = v188;
        ++*(_DWORD *)(v184 + 16);
        if ( v169 == v170 )
        {
          v27 = v393;
          v169 = (__int64)v419;
          v167 = v420;
          v168 = 4LL * (unsigned int)v420;
          goto LABEL_246;
        }
      }
      ++*(_QWORD *)(a1 + 688);
LABEL_320:
      v173 *= 2;
LABEL_321:
      sub_1C29540(v368, v173);
      sub_1CD2C70(v368, (__int64 *)&v422, &v435);
      v176 = (unsigned __int64 *)v435.m128i_i64[0];
      v172 = v422.m128_u64[0];
      v222 = *(_DWORD *)(a1 + 704) + 1;
      goto LABEL_316;
    }
LABEL_246:
    v13 = *(__m128i **)(v169 + v168 * 2 - 8);
    v189 = (unsigned __int64)v426;
    LODWORD(v420) = v167 - 1;
    v190 = v425;
    if ( v386 )
      goto LABEL_254;
    v191 = HIDWORD(v427);
    if ( HIDWORD(v427) == v428 )
      goto LABEL_254;
    if ( v426 != (__int64 *)v425 )
      v191 = (unsigned int)v427;
    v192 = &v426[v191];
    if ( v192 == v426 )
      goto LABEL_254;
    v193 = v426;
    while ( 1 )
    {
      v194 = *v193;
      v195 = v193;
      if ( (unsigned __int64)*v193 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v192 == ++v193 )
        goto LABEL_254;
    }
    if ( v192 == v193 )
    {
LABEL_254:
      if ( v190 != v189 )
        _libc_free(v189);
      v389 = v13;
      if ( !v378 )
        goto LABEL_168;
      v27 = v378;
LABEL_153:
      if ( v419 != v421 )
        _libc_free((unsigned __int64)v419);
      if ( (_QWORD *)v415.m128i_i64[0] != v416 )
        _libc_free(v415.m128i_u64[0]);
      v123 = (unsigned __int8 *)v430;
      v124 = (unsigned __int8 *)v430 + 8 * (unsigned int)v431;
      if ( v430 != (__m128i *)v124 )
      {
        do
        {
          v125 = *((_QWORD *)v124 - 1);
          v124 -= 8;
          if ( v125 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v125 + 8LL))(v125);
        }
        while ( v123 != v124 );
        v124 = (unsigned __int8 *)v430;
      }
      if ( v124 != (unsigned __int8 *)&v432 )
        _libc_free((unsigned __int64)v124);
      goto LABEL_45;
    }
    v399 = v27;
    v388 = v13;
    while ( 1 )
    {
      v408 = v194;
      if ( sub_13A0E30(a1 + 520, v194) )
        goto LABEL_421;
      v435.m128i_i64[0] = (__int64)&v436;
      v435.m128i_i64[1] = 0x1000000000LL;
      v442 = a1 + 520;
      v417[0] = (__int64)v418;
      v417[1] = 0x100000000LL;
      v422.m128_u64[0] = (unsigned __int64)v423;
      v422.m128_u64[1] = 0x200000000LL;
      sub_14EF3D0((__int64)v417, &v408);
      v269 = sub_1D61850(a1, (__int64)&v435, (__int64)v417, (__int64)&v422, 0, v268);
      v270 = v435.m128i_i64[0];
      v369 = v269;
      v271 = v435.m128i_i64[0] + 8LL * v435.m128i_u32[2];
      while ( v270 != v271 )
      {
        while ( 1 )
        {
          v272 = *(_QWORD *)(v271 - 8);
          v271 -= 8;
          if ( !v272 )
            break;
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v272 + 8LL))(v272);
          if ( v270 == v271 )
            goto LABEL_431;
        }
      }
LABEL_431:
      v273 = v378;
      v435.m128i_i32[2] = 0;
      v274 = (__int64 *)v422.m128_u64[0];
      if ( v369 )
        v273 = v369;
      v378 = v273;
      if ( v422.m128_u64[0] != v422.m128_u64[0] + 8LL * v422.m128_u32[2] )
        break;
LABEL_444:
      if ( v274 != v423 )
        _libc_free((unsigned __int64)v274);
      if ( (_WORD *)v417[0] != v418 )
        _libc_free(v417[0]);
      v290 = (unsigned __int64 *)v435.m128i_i64[0];
      v291 = v435.m128i_i64[0] + 8LL * v435.m128i_u32[2];
      if ( v435.m128i_i64[0] != v291 )
      {
        do
        {
          v292 = *(_QWORD *)(v291 - 8);
          v291 -= 8;
          if ( v292 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v292 + 8LL))(v292);
        }
        while ( v290 != (unsigned __int64 *)v291 );
        v290 = (unsigned __int64 *)v435.m128i_i64[0];
      }
      if ( v290 != &v436 )
        _libc_free((unsigned __int64)v290);
LABEL_421:
      v267 = v195 + 1;
      if ( v195 + 1 != v192 )
      {
        while ( 1 )
        {
          v194 = *v267;
          v195 = v267;
          if ( (unsigned __int64)*v267 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v192 == ++v267 )
            goto LABEL_424;
        }
        if ( v192 != v267 )
          continue;
      }
LABEL_424:
      v27 = v399;
      v13 = v388;
      v189 = (unsigned __int64)v426;
      v190 = v425;
      goto LABEL_254;
    }
    v356 = v192;
    v275 = (__int64 *)(v422.m128_u64[0] + 8LL * v422.m128_u32[2]);
    v370 = a1 + 864;
    v364 = a1 + 688;
    while ( 1 )
    {
      v289 = *v274;
      v409 = v289;
      v276 = (*(_BYTE *)(v289 + 23) & 0x40) != 0
           ? *(unsigned __int64 **)(v289 - 8)
           : (unsigned __int64 *)(v289 - 24LL * (*(_DWORD *)(v289 + 20) & 0xFFFFFFF));
      v277 = *v276;
      v278 = *(_DWORD *)(a1 + 712);
      v411 = *v276;
      if ( !v278 )
        break;
      v279 = *(_QWORD *)(a1 + 696);
      v280 = (v278 - 1) & (((unsigned int)v277 >> 9) ^ ((unsigned int)v277 >> 4));
      v281 = (unsigned __int64 *)(v279 + 16LL * v280);
      v282 = *v281;
      if ( v277 == *v281 )
        goto LABEL_438;
      v297 = 1;
      v298 = 0;
      while ( v282 != -8 )
      {
        if ( !v298 && v282 == -16 )
          v298 = v281;
        v280 = (v278 - 1) & (v297 + v280);
        v281 = (unsigned __int64 *)(v279 + 16LL * v280);
        v282 = *v281;
        if ( v277 == *v281 )
          goto LABEL_438;
        ++v297;
      }
      v299 = *(_DWORD *)(a1 + 704);
      if ( v298 )
        v281 = v298;
      ++*(_QWORD *)(a1 + 688);
      v300 = v299 + 1;
      if ( 4 * (v299 + 1) >= 3 * v278 )
        goto LABEL_477;
      if ( v278 - *(_DWORD *)(a1 + 708) - v300 <= v278 >> 3 )
        goto LABEL_478;
LABEL_473:
      *(_DWORD *)(a1 + 704) = v300;
      if ( *v281 != -8 )
        --*(_DWORD *)(a1 + 708);
      *v281 = v277;
      v281[1] = 0;
LABEL_438:
      v281[1] = 0;
      v283 = *(_DWORD *)(a1 + 888);
      if ( !v283 )
      {
        ++*(_QWORD *)(a1 + 864);
        goto LABEL_465;
      }
      v284 = v411;
      v285 = *(_QWORD *)(a1 + 872);
      v286 = (v283 - 1) & (((unsigned int)v411 >> 9) ^ ((unsigned int)v411 >> 4));
      v287 = (unsigned __int64 *)(v285 + 152LL * v286);
      v288 = (const void *)*v287;
      if ( v411 != *v287 )
      {
        v293 = 1;
        v294 = 0;
        while ( v288 != (const void *)-8LL )
        {
          if ( v288 == (const void *)-16LL && !v294 )
            v294 = v287;
          v286 = (v283 - 1) & (v293 + v286);
          v287 = (unsigned __int64 *)(v285 + 152LL * v286);
          v288 = (const void *)*v287;
          if ( v411 == *v287 )
            goto LABEL_440;
          ++v293;
        }
        v295 = *(_DWORD *)(a1 + 880);
        if ( v294 )
          v287 = v294;
        ++*(_QWORD *)(a1 + 864);
        v296 = v295 + 1;
        if ( 4 * (v295 + 1) < 3 * v283 )
        {
          if ( v283 - *(_DWORD *)(a1 + 884) - v296 > v283 >> 3 )
          {
LABEL_461:
            *(_DWORD *)(a1 + 880) = v296;
            if ( *v287 != -8 )
              --*(_DWORD *)(a1 + 884);
            *v287 = v284;
            v287[1] = (unsigned __int64)(v287 + 3);
            v287[2] = 0x1000000000LL;
            goto LABEL_440;
          }
LABEL_466:
          sub_1D6F390(v370, v283);
          sub_1D680B0(v370, (__int64 *)&v411, &v413);
          v287 = v413;
          v284 = v411;
          v296 = *(_DWORD *)(a1 + 880) + 1;
          goto LABEL_461;
        }
LABEL_465:
        v283 *= 2;
        goto LABEL_466;
      }
LABEL_440:
      ++v274;
      sub_14EF3D0((__int64)(v287 + 1), &v409);
      if ( v275 == v274 )
      {
        v192 = v356;
        v274 = (__int64 *)v422.m128_u64[0];
        goto LABEL_444;
      }
    }
    ++*(_QWORD *)(a1 + 688);
LABEL_477:
    v278 *= 2;
LABEL_478:
    sub_1C29540(v364, v278);
    sub_1CD2C70(v364, (__int64 *)&v411, &v413);
    v281 = v413;
    v277 = v411;
    v300 = *(_DWORD *)(a1 + 704) + 1;
    goto LABEL_473;
  }
  v54 = (unsigned int)(v21 - 75);
  if ( (unsigned __int8)(v21 - 75) > 1u )
  {
    if ( (_BYTE)v21 == 54 )
    {
      sub_1625C10(a2, 16, 0);
      if ( !*(_QWORD *)(a1 + 176) )
        return 0;
      v82 = sub_15F32D0(a2);
      v83 = *(_QWORD *)a2;
      if ( v82 || (v392 = *(_BYTE *)(a2 + 18) & 1) != 0 )
      {
        v392 = 0;
        goto LABEL_127;
      }
      if ( (*(_BYTE *)(v83 + 8) & 0xFB) != 0xB )
      {
LABEL_127:
        v107 = (__int64 *)v13[-2].m128i_i64[1];
        v108 = *v107;
        if ( *(_BYTE *)(*v107 + 8) == 16 )
          v108 = **(_QWORD **)(v108 + 16);
        v109 = (__int64 ****)sub_13CF970((__int64)v13);
        return (unsigned int)sub_1D73760(
                               a1,
                               (__int64)v13,
                               *v109,
                               v83,
                               *(_DWORD *)(v108 + 8) >> 8,
                               *(double *)a4.m128_u64,
                               a5,
                               a6,
                               a7,
                               v110,
                               v111,
                               a10,
                               a11)
             | v392;
      }
      v84 = *(_QWORD *)(a2 + 8);
      if ( v84 && !*(_QWORD *)(v84 + 8) )
      {
        v302 = sub_1648700(*(_QWORD *)(a2 + 8));
        if ( sub_13A0E30(v400, (__int64)v302) )
        {
LABEL_126:
          v83 = v13->m128i_i64[0];
          goto LABEL_127;
        }
        v84 = *(_QWORD *)(a2 + 8);
      }
      v435.m128i_i64[0] = 0;
      v424 = (unsigned __int64 *)&v426;
      v425 = 0x800000000LL;
      v435.m128i_i64[1] = (__int64)&v439;
      v436 = (unsigned __int64)&v439;
      v437 = 16;
      LODWORD(v438) = 0;
      v430 = (__m128i *)&v432;
      v431 = 0x800000000LL;
      if ( v84 )
      {
        v85 = sub_1648700(v84);
        for ( j = (unsigned __int64 *)&v426; ; j = v424 )
        {
          j[v20] = (unsigned __int64)v85;
          v20 = (unsigned int)(v425 + 1);
          LODWORD(v425) = v425 + 1;
          v84 = *(_QWORD *)(v84 + 8);
          if ( !v84 )
            break;
          v85 = sub_1648700(v84);
          if ( HIDWORD(v425) <= (unsigned int)v20 )
          {
            v382 = v85;
            sub_16CD150((__int64)&v424, &v426, 0, 8, v87, v88);
            v20 = (unsigned int)v425;
            v85 = v382;
          }
        }
      }
      v89 = *(_QWORD *)a2;
      LODWORD(v409) = sub_1D5D7E0(*(_QWORD *)(a1 + 904), (__int64 *)v13->m128i_i64[0], 0);
      v410 = v90;
      if ( (_BYTE)v409 )
        v377 = sub_1D5A920(v409);
      else
        v377 = sub_1F58D40(&v409, v89, v90, v91, v92, v93);
      v412 = v377;
      if ( v377 > 0x40 )
      {
        sub_16A4EF0((__int64)&v411, 0, 0);
        v414 = v377;
        sub_16A4EF0((__int64)&v413, 0, 0);
      }
      else
      {
        v411 = 0;
        v414 = v377;
        v413 = 0;
      }
      LODWORD(k) = v425;
      if ( (_DWORD)v425 )
      {
        v375 = v13;
        while ( 1 )
        {
          v95 = v424[(unsigned int)k - 1];
          LODWORD(v425) = k - 1;
          v419 = (_WORD *)v95;
          sub_1412190((__int64)&v435, v95);
          if ( !v96 )
            goto LABEL_198;
          v97 = *((unsigned __int8 *)v419 + 16);
          if ( (_BYTE)v97 == 77 )
          {
            v98 = *((_QWORD *)v419 + 1);
            for ( k = (unsigned int)v425; v98; v98 = *(_QWORD *)(v98 + 8) )
            {
              v99 = sub_1648700(v98);
              if ( (unsigned int)k >= HIDWORD(v425) )
              {
                v381 = v99;
                sub_16CD150((__int64)&v424, &v426, 0, 8, v100, v101);
                k = (unsigned int)v425;
                v99 = v381;
              }
              v424[k] = (unsigned __int64)v99;
              k = (unsigned int)(v425 + 1);
              LODWORD(v425) = v425 + 1;
            }
            goto LABEL_105;
          }
          if ( v97 == 50 )
            break;
          if ( v97 == 60 )
          {
            v252 = *(_QWORD *)v419;
            v422.m128_i32[0] = sub_1D5D7E0(*(_QWORD *)(a1 + 904), *(__int64 **)v419, 0);
            v422.m128_u64[1] = v253;
            if ( v422.m128_i8[0] )
              v144 = sub_1D5A920(v422.m128_i8[0]);
            else
              v144 = sub_1F58D40(&v422, v252, v253, v254, v255, v256);
            if ( !v144 )
              goto LABEL_198;
            if ( v144 <= 0x40 )
            {
              v145 = 64 - v144;
LABEL_196:
              v146 = 0xFFFFFFFFFFFFFFFFLL >> v145;
              if ( v412 > 0x40 )
                *(_QWORD *)v411 |= v146;
              else
                v411 |= v146;
              goto LABEL_198;
            }
          }
          else
          {
            if ( v97 != 47 || (v142 = *(_QWORD *)(sub_13CF970((__int64)v419) + 24), *(_BYTE *)(v142 + 16) != 13) )
            {
LABEL_225:
              v13 = v375;
              goto LABEL_226;
            }
            v143 = v377 - 1;
            if ( *(_DWORD *)(v142 + 32) > 0x40u )
            {
              v380 = *(_DWORD *)(v142 + 32);
              if ( v380 - (unsigned int)sub_16A57B0(v142 + 24) <= 0x40 && v143 > **(_QWORD **)(v142 + 24) )
                v143 = **(_QWORD **)(v142 + 24);
            }
            else if ( v143 > *(_QWORD *)(v142 + 24) )
            {
              v143 = *(_QWORD *)(v142 + 24);
            }
            v144 = v377 - v143;
            if ( v377 == (_DWORD)v143 )
              goto LABEL_198;
            if ( v144 <= 0x40 )
            {
              v145 = v143 + 64 - v377;
              goto LABEL_196;
            }
          }
          sub_16A5260(&v411, 0, v144);
LABEL_198:
          LODWORD(k) = v425;
LABEL_105:
          if ( !(_DWORD)k )
          {
            v13 = v375;
            goto LABEL_107;
          }
        }
        v257 = *(_QWORD *)(sub_13CF970((__int64)v419) + 24);
        if ( *(_BYTE *)(v257 + 16) != 13 )
          goto LABEL_225;
        v422.m128_i32[2] = *(_DWORD *)(v257 + 32);
        if ( v422.m128_i32[2] > 0x40u )
          sub_16A4FD0((__int64)&v422, (const void **)(v257 + 24));
        else
          v422.m128_u64[0] = *(_QWORD *)(v257 + 24);
        if ( v412 > 0x40 )
          sub_16A89F0((__int64 *)&v411, (__int64 *)&v422);
        else
          v411 |= v422.m128_u64[0];
        if ( (int)sub_16A9900((__int64)&v422, (unsigned __int64 *)&v413) > 0 )
        {
          if ( v414 <= 0x40 && v422.m128_i32[2] <= 0x40u )
          {
            v414 = v422.m128_u32[2];
            v306 = (unsigned __int64 *)v422.m128_u64[0];
            v413 = (unsigned __int64 *)(v422.m128_u64[0] & (0xFFFFFFFFFFFFFFFFLL >> -v422.m128_i8[8]));
            goto LABEL_544;
          }
          sub_16A51C0((__int64)&v413, (__int64)&v422);
        }
        if ( v422.m128_i32[2] > 0x40u )
        {
          if ( !sub_16A5220((__int64)&v422, (const void **)&v413) || *(__m128i **)sub_13CF970((__int64)v419) != v375 )
          {
LABEL_402:
            if ( v422.m128_u64[0] )
              j_j___libc_free_0_0(v422.m128_u64[0]);
            goto LABEL_198;
          }
LABEL_401:
          sub_14EF3D0((__int64)&v430, &v419);
          if ( v422.m128_i32[2] > 0x40u )
            goto LABEL_402;
          goto LABEL_198;
        }
        v306 = (unsigned __int64 *)v422.m128_u64[0];
LABEL_544:
        if ( v306 != v413 || *(__m128i **)sub_13CF970((__int64)v419) != v375 )
          goto LABEL_198;
        goto LABEL_401;
      }
LABEL_107:
      v102 = v412;
      if ( v412 > 0x40 )
      {
        v305 = v102 - sub_16A57B0((__int64)&v411);
        v106 = v305;
        if ( v305 <= 1 || (unsigned int)sub_16A58F0((__int64)&v411) != v305 )
          goto LABEL_226;
      }
      else
      {
        if ( v411 )
        {
          _BitScanReverse64(&v103, v411);
          v104 = v103 ^ 0x3F;
        }
        else
        {
          v104 = 64;
        }
        v105 = v412 + v104 - 64;
        v106 = v412 - v105;
        if ( v412 - v105 <= 1 || v411 != 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v105 - (unsigned __int8)v412 + 64) )
          goto LABEL_226;
      }
      if ( v414 <= 0x40 )
      {
        if ( v413 != (unsigned __int64 *)v411 )
          goto LABEL_117;
      }
      else if ( !sub_16A5220((__int64)&v413, (const void **)&v411) )
      {
        v392 = 0;
        goto LABEL_115;
      }
      v307 = *(__int64 **)v13->m128i_i64[0];
      v308 = (__int64 *)sub_1644C60(v307, v106);
      v309 = sub_1D5D7E0(*(_QWORD *)(a1 + 904), v308, 0);
      v314 = v409;
      v415.m128i_i32[0] = v309;
      v415.m128i_i64[1] = v310;
      a4 = (__m128)_mm_loadu_si128(&v415);
      v422 = a4;
      if ( (_BYTE)v309 != (_BYTE)v409 || !(_BYTE)v409 && v422.m128_u64[1] != v410 )
      {
        v315 = sub_1D159A0((char *)&v409, (__int64)v308, v310, v311, v312, v313, v349, v352, v355, v359);
        if ( v315 > (unsigned int)sub_1D159A0(
                                    (char *)&v422,
                                    (__int64)v308,
                                    v316,
                                    v317,
                                    v318,
                                    v319,
                                    v350,
                                    v353,
                                    v357,
                                    v360) )
        {
          v324 = sub_1D159A0(v415.m128i_i8, (__int64)v308, v320, v321, v322, v323, v351, v354, v358, v361);
          if ( v324 > 7
            && (v324 & (v324 - 1)) == 0
            && v415.m128i_i8[0]
            && v314
            && (*(_BYTE *)(*(_QWORD *)(a1 + 176) + 2 * (v415.m128i_u8[0] + 115LL * v314 + 16104) + 1) & 0xF0) == 0 )
          {
            v325 = (_QWORD *)v13[2].m128i_i64[0];
            if ( v325 == (_QWORD *)(v13[2].m128i_i64[1] + 40) || !v325 )
              v326 = 0;
            else
              v326 = (__int64)(v325 - 3);
            sub_17CE510((__int64)&v422, v326, 0, 0, 0);
            v418[0] = 257;
            v327 = sub_159C0E0(v307, (__int64)&v411);
            v328 = *(_BYTE *)(v327 + 16);
            if ( v328 <= 0x10u )
            {
              if ( v328 == 13 )
              {
                v347 = *(_DWORD *)(v327 + 32);
                if ( v347 <= 0x40
                   ? 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v347) == *(_QWORD *)(v327 + 24)
                   : v347 == (unsigned int)sub_16A58F0(v327 + 24) )
                {
                  v329 = (__int64)v13;
                  goto LABEL_571;
                }
              }
              if ( v13[1].m128i_i8[0] <= 0x10u )
              {
                v329 = sub_15A2CF0(
                         v13->m128i_i64,
                         v327,
                         *(double *)a4.m128_u64,
                         *(double *)a5.m128_u64,
                         *(double *)a6.m128_u64);
LABEL_571:
                if ( *(_BYTE *)(v329 + 16) <= 0x17u )
                {
                  sub_165A590((__int64)&v419, v400, 0);
                  sub_164D160(
                    (__int64)v13,
                    0,
                    a4,
                    *(double *)a5.m128_u64,
                    *(double *)a6.m128_u64,
                    *(double *)a7.m128_u64,
                    v11,
                    v12,
                    *(double *)a10.m128i_i64,
                    a11);
                  BUG();
                }
                sub_165A590((__int64)&v419, v400, v329);
                sub_164D160(
                  (__int64)v13,
                  v329,
                  a4,
                  *(double *)a5.m128_u64,
                  *(double *)a6.m128_u64,
                  *(double *)a7.m128_u64,
                  v330,
                  v331,
                  *(double *)a10.m128i_i64,
                  a11);
                if ( (*(_BYTE *)(v329 + 23) & 0x40) != 0 )
                  v332 = *(__m128i ***)(v329 - 8);
                else
                  v332 = (__m128i **)(v329 - 24LL * (*(_DWORD *)(v329 + 20) & 0xFFFFFFF));
                if ( *v332 )
                {
                  v333 = v332[1];
                  v334 = (unsigned __int64)v332[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v334 = v333;
                  if ( v333 )
                    v333[1].m128i_i64[0] = v334 | v333[1].m128i_i64[0] & 3;
                }
                *v332 = v13;
                v335 = (__m128i *)v13->m128i_i64[1];
                v332[1] = v335;
                if ( v335 )
                  v335[1].m128i_i64[0] = (unsigned __int64)(v332 + 1) | v335[1].m128i_i64[0] & 3;
                v332[2] = (__m128i *)((unsigned __int64)v332[2] & 3 | (unsigned __int64)&v13->m128i_u64[1]);
                v336 = (__int64 *)v430;
                v13->m128i_i64[1] = (__int64)v332;
                v406 = &v336[(unsigned int)v431];
                while ( 1 )
                {
                  if ( v336 == v406 )
                  {
                    sub_17CD270((__int64 *)&v422);
                    v392 = 1;
                    goto LABEL_226;
                  }
                  v339 = (_QWORD *)*v336;
                  v340 = *(_QWORD *)(sub_13CF970(*v336) + 24);
                  if ( *(_DWORD *)(v340 + 32) > 0x40u )
                  {
                    if ( !sub_16A5220(v340 + 24, (const void **)&v411) )
                      goto LABEL_581;
                  }
                  else if ( *(_QWORD *)(v340 + 24) != v411 )
                  {
                    goto LABEL_581;
                  }
                  sub_164D160(
                    (__int64)v339,
                    v329,
                    a4,
                    *(double *)a5.m128_u64,
                    *(double *)a6.m128_u64,
                    *(double *)a7.m128_u64,
                    v337,
                    v338,
                    *(double *)a10.m128i_i64,
                    a11);
                  v341 = *(_QWORD **)(a1 + 232);
                  if ( v341 )
                    v341 -= 3;
                  if ( v339 == v341 )
                    *(_QWORD *)(a1 + 232) = v339[4];
                  sub_15F20C0(v339);
LABEL_581:
                  ++v336;
                }
              }
            }
            v421[0] = 257;
            v329 = sub_15FB440(26, v13->m128i_i64, v327, (__int64)&v419, 0);
            if ( v422.m128_u64[1] )
            {
              v342 = (__int64 *)v423[0];
              sub_157E9D0(v422.m128_u64[1] + 40, v329);
              v343 = *(_QWORD *)(v329 + 24);
              v344 = *v342;
              *(_QWORD *)(v329 + 32) = v342;
              v344 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v329 + 24) = v344 | v343 & 7;
              *(_QWORD *)(v344 + 8) = v329 + 24;
              *v342 = *v342 & 7 | (v329 + 24);
            }
            sub_164B780(v329, v417);
            sub_12A86E0((__int64 *)&v422, v329);
            goto LABEL_571;
          }
        }
      }
LABEL_226:
      if ( v414 > 0x40 )
      {
LABEL_115:
        if ( v413 )
          j_j___libc_free_0_0(v413);
      }
LABEL_117:
      if ( v412 > 0x40 && v411 )
        j_j___libc_free_0_0(v411);
      if ( v430 != (__m128i *)&v432 )
        _libc_free((unsigned __int64)v430);
      if ( v436 != v435.m128i_i64[1] )
        _libc_free(v436);
      if ( v424 != (unsigned __int64 *)&v426 )
        _libc_free((unsigned __int64)v424);
      goto LABEL_126;
    }
    v55 = *(_BYTE **)(a1 + 176);
    if ( (_BYTE)v21 == 55 )
    {
      v401 = *(_QWORD *)(a1 + 176);
      if ( v55 )
      {
        v157 = *(_QWORD *)(a1 + 904);
        v158 = **(_QWORD **)(a2 - 48);
        v159 = sub_127FA20(v157, v158);
        if ( ((v159 + 7) & 0xFFFFFFFFFFFFFFF8LL) == sub_127FA20(v157, v158) )
        {
          if ( sub_127FA20(v157, v158) )
          {
            v232 = (unsigned __int64)sub_127FA20(v157, v158) >> 1;
            v233 = (_QWORD *)sub_16498A0(a2);
            v419 = (_WORD *)sub_1644C60(v233, v232);
            v234 = sub_127FA20(v157, (__int64)v419);
            if ( ((v234 + 7) & 0xFFFFFFFFFFFFFFF8LL) == sub_127FA20(v157, (__int64)v419) )
            {
              v235 = *(_QWORD *)(a2 - 48);
              v435.m128i_i64[0] = (__int64)&v422;
              v435.m128i_i64[1] = (__int64)&v424;
              v436 = (unsigned int)v232;
              LOBYTE(v236) = sub_1D66F70(&v435, v235);
              v27 = v236;
              if ( (_BYTE)v236 )
              {
                if ( *(_BYTE *)(*(_QWORD *)v422.m128_u64[0] + 8LL) == 11
                  && (unsigned int)v232 >= (unsigned __int64)sub_127FA20(v157, *(_QWORD *)v422.m128_u64[0])
                  && *(_BYTE *)(*v424 + 8) == 11
                  && (unsigned int)v232 >= (unsigned __int64)sub_127FA20(v157, *v424) )
                {
                  v237 = (__int64 *)v422.m128_u64[0];
                  v238 = (__int64 *)v424;
                  v239 = *((_BYTE *)v424 + 16);
                  if ( *(_BYTE *)(v422.m128_u64[0] + 16) == 71 )
                  {
                    if ( v239 > 0x17u )
                    {
                      if ( v239 != 71 )
                        v238 = 0;
                    }
                    else
                    {
                      v238 = 0;
                    }
                    v345 = sub_1F59570(**(_QWORD **)(v422.m128_u64[0] - 24), 0);
                    v241 = v401;
                    v404 = v345;
                    v397 = v346;
                  }
                  else
                  {
                    if ( v239 <= 0x17u )
                    {
                      v238 = 0;
                    }
                    else if ( v239 != 71 )
                    {
                      v238 = 0;
                    }
                    v237 = 0;
                    v240 = sub_1F59570(*(_QWORD *)v422.m128_u64[0], 0);
                    v241 = v401;
                    v404 = v240;
                    v397 = v242;
                  }
                  v387 = v241;
                  v243 = v238 ? (unsigned __int64 *)*(v238 - 3) : v424;
                  v245 = sub_1F59570(*v243, 0);
                  if ( byte_4FC2880
                    || (v246 = *(__int64 (**)())(*(_QWORD *)v387 + 184LL), v246 != sub_1D5A390)
                    && ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, __int64, __int64))v246)(
                         v387,
                         v404,
                         v397,
                         v245,
                         v244) )
                  {
                    v247 = sub_16498A0((__int64)v13);
                    v248 = (unsigned __int8 *)v13[3].m128i_i64[0];
                    v435.m128i_i64[0] = 0;
                    v437 = v247;
                    v249 = v13[2].m128i_i64[1];
                    v438 = 0;
                    v435.m128i_i64[1] = v249;
                    v436 = (unsigned __int64)&v13[1].m128i_u64[1];
                    v439 = 0;
                    v440 = 0;
                    v441 = 0;
                    v430 = (__m128i *)v248;
                    if ( v248 )
                    {
                      sub_1623A60((__int64)&v430, (__int64)v248, 2);
                      if ( v435.m128i_i64[0] )
                        sub_161E7C0((__int64)&v435, v435.m128i_i64[0]);
                      v435.m128i_i64[0] = (__int64)v430;
                      if ( v430 )
                        sub_1623210((__int64)&v430, (unsigned __int8 *)v430, (__int64)&v435);
                    }
                    if ( v237 && v237[5] != v13[2].m128i_i64[1] )
                    {
                      LOWORD(v432) = 257;
                      v422.m128_u64[0] = sub_12AA3B0(v435.m128i_i64, 0x2Fu, *(v237 - 3), *v237, (__int64)&v430);
                    }
                    if ( v238 && v238[5] != v13[2].m128i_i64[1] )
                    {
                      LOWORD(v432) = 257;
                      v424 = (unsigned __int64 *)sub_12AA3B0(v435.m128i_i64, 0x2Fu, *(v238 - 3), *v238, (__int64)&v430);
                    }
                    v250 = sub_15F2050((__int64)v13);
                    v251 = *(_BYTE *)sub_1632FA0(v250);
                    v432 = v13;
                    v430 = &v435;
                    LOBYTE(v417[0]) = v251 ^ 1;
                    v431 = (__int64)&v419;
                    v433 = v417;
                    sub_1D634F0((__int64)&v430, v422.m128_i64[0], 0);
                    sub_1D634F0((__int64)&v430, (__int64)v424, 1);
                    sub_15F20C0(v13);
                    sub_17CD270(v435.m128i_i64);
                    return v27;
                  }
                }
              }
            }
          }
        }
      }
      sub_1625C10((__int64)v13, 16, 0);
      if ( !*(_QWORD *)(a1 + 176) )
        return 0;
      v160 = (__int64 ***)v13[-2].m128i_i64[1];
      v161 = *v160;
      if ( *((_BYTE *)*v160 + 8) != 16 )
        goto LABEL_215;
    }
    else
    {
      if ( !v55 )
        goto LABEL_35;
      if ( (_BYTE)v21 == 59 )
      {
        v160 = *(__int64 ****)(a2 - 48);
        v196 = *v160;
        if ( *((_BYTE *)*v160 + 8) == 16 )
          v196 = (__int64 **)*v196[2];
        v162 = *(_QWORD *)a2;
        v163 = *((_DWORD *)v196 + 2) >> 8;
        return sub_1D73760(a1, (__int64)v13, v160, v162, v163, *(double *)a4.m128_u64, a5, a6, a7, v18, v19, a10, a11);
      }
      if ( (_BYTE)v21 != 58 )
      {
LABEL_35:
        v56 = (unsigned int)(unsigned __int8)v21 - 35;
        if ( (unsigned int)v56 <= 0x11 )
        {
          if ( (_BYTE)v21 == 50 )
          {
            if ( byte_4FC2F80 && v55 )
              return sub_1D5C3D0(a2, (__int64)v55);
            return 0;
          }
          v56 = (unsigned int)(v21 - 48);
          if ( (unsigned __int8)(v21 - 48) <= 1u )
          {
            v57 = *(_QWORD *)(a2 - 24);
            if ( !v57 )
              BUG();
            if ( *(_BYTE *)(v57 + 16) == 13 && v55 && v55[17] )
              return sub_1D69EF0((__int64)v13, v57, (__int64)v55, *(_QWORD *)(a1 + 904));
            return 0;
          }
        }
        switch ( (_BYTE)v21 )
        {
          case 0x38:
            v27 = sub_15FA1F0(a2);
            if ( (_BYTE)v27 )
            {
              v405 = *(_QWORD *)a2;
              v398 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
              LOWORD(v436) = 261;
              v430 = (__m128i *)sub_1649960(a2);
              v431 = v258;
              v435.m128i_i64[0] = (__int64)&v430;
              v259 = (__m128i *)sub_1648A60(56, 1u);
              v260 = v259;
              if ( v259 )
                sub_15FD590((__int64)v259, v398, v405, (__int64)&v435, a2);
              v261 = *(_QWORD *)(a2 + 48);
              v435.m128i_i64[0] = v261;
              if ( v261 )
                sub_1623A60((__int64)&v435, v261, 2);
              m128i_i64 = (__int64)v260[3].m128i_i64;
              if ( &v260[3] != &v435 )
              {
                v263 = v260[3].m128i_i64[0];
                if ( v263 )
                {
                  sub_161E7C0((__int64)v260[3].m128i_i64, v263);
                  m128i_i64 = (__int64)v260[3].m128i_i64;
                }
                v264 = (unsigned __int8 *)v435.m128i_i64[0];
                v260[3].m128i_i64[0] = v435.m128i_i64[0];
                if ( v264 )
                {
                  sub_1623210((__int64)&v435, v264, m128i_i64);
                  v435.m128i_i64[0] = 0;
                }
              }
              sub_17CD270(v435.m128i_i64);
              sub_164D160(
                (__int64)v13,
                (__int64)v260,
                a4,
                *(double *)a5.m128_u64,
                *(double *)a6.m128_u64,
                *(double *)a7.m128_u64,
                v265,
                v266,
                *(double *)a10.m128i_i64,
                a11);
              sub_15F20C0(v13);
              sub_1D779D0(a1, v260, a3);
              return v27;
            }
            return sub_1D663F0(a2, *(_QWORD *)(a1 + 192));
          case 0x4E:
            return sub_1D765D0(a1, a2, a3, v54, a4, a5, a6, a7, v18, v19, a10, a11);
          case 0x4F:
            return sub_1D60900(
                     a1,
                     a2,
                     a4,
                     *(double *)a5.m128_u64,
                     *(double *)a6.m128_u64,
                     *(double *)a7.m128_u64,
                     v18,
                     v19,
                     *(double *)a10.m128i_i64,
                     a11,
                     v56,
                     v54,
                     v17,
                     (int)v55);
          case 0x55:
            return sub_1D6F6A0((__int64)v55, (_QWORD *)a2);
          case 0x1B:
            return sub_1D5EAC0(a1, a2);
          case 0x53:
            return sub_1D5D990(
                     (_QWORD *)a1,
                     (_QWORD *)a2,
                     a4,
                     *(double *)a5.m128_u64,
                     *(double *)a6.m128_u64,
                     *(double *)a7.m128_u64,
                     v18,
                     v19,
                     *(double *)a10.m128i_i64,
                     a11);
        }
        return 0;
      }
      v160 = *(__int64 ****)(a2 - 72);
      v161 = *v160;
      if ( *((_BYTE *)*v160 + 8) != 16 )
      {
LABEL_215:
        v162 = *(_QWORD *)v13[-3].m128i_i64[0];
        v163 = *((_DWORD *)v161 + 2) >> 8;
        return sub_1D73760(a1, (__int64)v13, v160, v162, v163, *(double *)a4.m128_u64, a5, a6, a7, v18, v19, a10, a11);
      }
    }
    v161 = (__int64 **)*v161[2];
    goto LABEL_215;
  }
  v55 = *(_BYTE **)(a1 + 176);
  if ( v55 )
  {
    if ( v55[16] )
      goto LABEL_35;
    v126 = *(_QWORD *)(a2 + 40);
    v127 = *(__int64 (**)())(*(_QWORD *)v55 + 24LL);
    if ( v127 != sub_1D5A350 && ((unsigned __int8 (__fastcall *)(_BYTE *))v127)(v55) && *(_BYTE *)(a2 + 16) == 76 )
      return sub_1D65F00(
               v13,
               a4,
               *(double *)a5.m128_u64,
               *(double *)a6.m128_u64,
               *(double *)a7.m128_u64,
               v140,
               v141,
               *(double *)a10.m128i_i64,
               a11);
  }
  else
  {
    v126 = *(_QWORD *)(a2 + 40);
  }
  v128 = *(_QWORD *)(a2 + 8);
  v129 = 0;
  v435 = 0u;
  v436 = 0;
  LODWORD(v437) = 0;
  if ( !v128 )
  {
LABEL_130:
    sub_15F20C0(v13);
    j___libc_free_0(v435.m128i_i64[1]);
    return 1;
  }
  do
  {
    v131 = (_QWORD *)v128;
    v132 = sub_1648700(v128);
    v128 = *(_QWORD *)(v128 + 8);
    if ( *((_BYTE *)v132 + 16) == 77 )
      continue;
    v133 = v132[5];
    if ( v133 == v126 )
      continue;
    if ( !(_DWORD)v437 )
    {
      ++v435.m128i_i64[0];
      goto LABEL_291;
    }
    v134 = (v437 - 1) & (((unsigned int)v133 >> 9) ^ ((unsigned int)v133 >> 4));
    v135 = (__int64 *)(v435.m128i_i64[1] + 16LL * v134);
    v136 = *v135;
    if ( v133 != *v135 )
    {
      v197 = 1;
      v198 = 0;
      while ( v136 != -8 )
      {
        if ( v136 == -16 && !v198 )
          v198 = v135;
        v134 = (v437 - 1) & (v197 + v134);
        v135 = (__int64 *)(v435.m128i_i64[1] + 16LL * v134);
        v136 = *v135;
        if ( v133 == *v135 )
          goto LABEL_180;
        ++v197;
      }
      if ( !v198 )
        v198 = v135;
      ++v435.m128i_i64[0];
      v199 = v436 + 1;
      if ( 4 * ((int)v436 + 1) < (unsigned int)(3 * v437) )
      {
        if ( (int)v437 - HIDWORD(v436) - v199 > (unsigned int)v437 >> 3 )
          goto LABEL_274;
        v396 = ((unsigned int)v133 >> 9) ^ ((unsigned int)v133 >> 4);
        v403 = v133;
        sub_1D69B70((__int64)&v435, v437);
        if ( !(_DWORD)v437 )
        {
LABEL_629:
          LODWORD(v436) = v436 + 1;
          BUG();
        }
        v215 = 0;
        v133 = v403;
        v223 = 1;
        v224 = (v437 - 1) & v396;
        v199 = v436 + 1;
        v198 = (__int64 *)(v435.m128i_i64[1] + 16LL * v224);
        v225 = *v198;
        if ( v403 == *v198 )
          goto LABEL_274;
        while ( v225 != -8 )
        {
          if ( !v215 && v225 == -16 )
            v215 = v198;
          v224 = (v437 - 1) & (v223 + v224);
          v198 = (__int64 *)(v435.m128i_i64[1] + 16LL * v224);
          v225 = *v198;
          if ( v403 == *v198 )
            goto LABEL_274;
          ++v223;
        }
        goto LABEL_295;
      }
LABEL_291:
      v402 = v133;
      sub_1D69B70((__int64)&v435, 2 * v437);
      if ( !(_DWORD)v437 )
        goto LABEL_629;
      v133 = v402;
      v212 = (v437 - 1) & (((unsigned int)v402 >> 9) ^ ((unsigned int)v402 >> 4));
      v199 = v436 + 1;
      v198 = (__int64 *)(v435.m128i_i64[1] + 16LL * v212);
      v213 = *v198;
      if ( v402 == *v198 )
        goto LABEL_274;
      v214 = 1;
      v215 = 0;
      while ( v213 != -8 )
      {
        if ( !v215 && v213 == -16 )
          v215 = v198;
        v212 = (v437 - 1) & (v214 + v212);
        v198 = (__int64 *)(v435.m128i_i64[1] + 16LL * v212);
        v213 = *v198;
        if ( v402 == *v198 )
          goto LABEL_274;
        ++v214;
      }
LABEL_295:
      if ( v215 )
        v198 = v215;
LABEL_274:
      LODWORD(v436) = v199;
      if ( *v198 != -8 )
        --HIDWORD(v436);
      *v198 = v133;
      v198[1] = 0;
LABEL_277:
      v200 = sub_157EE30(v133);
      v201 = v13[1].m128i_i16[1];
      v202 = v13[1].m128i_u8[0];
      v203 = v200;
      v204 = v200 - 24;
      v205 = v13[-2].m128i_i64[1];
      v206 = v13[-3].m128i_i64[0];
      if ( v203 )
        v203 = v204;
      LOWORD(v432) = 257;
      v207 = sub_15FEEB0(v202 - 24, v201 & 0x7FFF, v206, v205, (__int64)&v430, v203);
      v198[1] = v207;
      v208 = v207;
      v430 = (__m128i *)v13[3].m128i_i64[0];
      if ( v430 )
      {
        v394 = v207;
        sub_1623A60((__int64)&v430, (__int64)v430, 2);
        v208 = v394;
        v209 = v394 + 48;
        if ( (__m128i **)(v394 + 48) == &v430 )
        {
          if ( v430 )
            sub_161E7C0(v209, (__int64)v430);
          goto LABEL_283;
        }
        v210 = *(_QWORD *)(v394 + 48);
        if ( !v210 )
        {
LABEL_288:
          v211 = (unsigned __int8 *)v430;
          *(_QWORD *)(v208 + 48) = v430;
          if ( v211 )
            sub_1623210((__int64)&v430, v211, v209);
LABEL_283:
          v137 = v198[1];
          if ( *v131 )
          {
            v138 = v131[1];
LABEL_172:
            v130 = v131[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v130 = v138;
            if ( v138 )
              *(_QWORD *)(v138 + 16) = *(_QWORD *)(v138 + 16) & 3LL | v130;
          }
          *v131 = v137;
          v129 = 1;
          if ( !v137 )
            continue;
          goto LABEL_183;
        }
      }
      else
      {
        v209 = v207 + 48;
        if ( (__m128i **)(v207 + 48) == &v430 )
          goto LABEL_283;
        v210 = *(_QWORD *)(v207 + 48);
        if ( !v210 )
          goto LABEL_283;
      }
      v395 = v208;
      sub_161E7C0(v209, v210);
      v208 = v395;
      goto LABEL_288;
    }
LABEL_180:
    v137 = v135[1];
    if ( !v137 )
    {
      v198 = v135;
      goto LABEL_277;
    }
    v138 = v128;
    if ( *v131 )
      goto LABEL_172;
    *v131 = v137;
LABEL_183:
    v139 = *(_QWORD *)(v137 + 8);
    v131[1] = v139;
    if ( v139 )
      *(_QWORD *)(v139 + 16) = (unsigned __int64)(v131 + 1) | *(_QWORD *)(v139 + 16) & 3LL;
    v129 = 1;
    v131[2] = (v137 + 8) | v131[2] & 3LL;
    *(_QWORD *)(v137 + 8) = v131;
  }
  while ( v128 );
  if ( !v13->m128i_i64[1] )
    goto LABEL_130;
  j___libc_free_0(v435.m128i_i64[1]);
  if ( !v129 )
    return sub_1D65F00(
             v13,
             a4,
             *(double *)a5.m128_u64,
             *(double *)a6.m128_u64,
             *(double *)a7.m128_u64,
             v140,
             v141,
             *(double *)a10.m128i_i64,
             a11);
  return 1;
}
