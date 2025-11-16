// Function: sub_18D3D80
// Address: 0x18d3d80
//
__int64 __fastcall sub_18D3D80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  bool v5; // zf
  __int64 v6; // rax
  __int64 *v7; // rax
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __m128i *v15; // rax
  __int64 v16; // r8
  __int64 *v17; // rax
  __int64 v18; // rsi
  __int64 *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  int v22; // ebx
  unsigned int v23; // esi
  __int64 v24; // r15
  char v25; // dl
  __int64 *v26; // rax
  __int64 *v27; // r13
  __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 *v30; // rax
  __int64 *v31; // rdx
  __int64 *v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  __m128i *v37; // rax
  __int64 *v38; // rax
  int v39; // r8d
  int v40; // r9d
  __int64 v41; // r15
  __int64 *v42; // rbx
  __int64 v43; // rax
  __int64 *v44; // rax
  int v45; // r8d
  __int64 v46; // r15
  __int64 *v47; // rbx
  __int64 v48; // rax
  __int64 *v49; // rax
  __int64 *v50; // rax
  int v51; // r8d
  int v52; // r9d
  __int64 v53; // r13
  __int64 *v54; // r15
  __int64 v55; // rax
  __int64 *v56; // rax
  __int64 v57; // r13
  __int64 *v58; // r15
  __int64 v59; // rax
  __int64 v60; // rbx
  __int64 *v61; // rax
  __int64 v62; // rax
  unsigned int v63; // eax
  __int64 v64; // r14
  __int64 v65; // rbx
  __int64 *v66; // rax
  int v67; // r9d
  __int64 v68; // r8
  __int64 v69; // rax
  __int64 *v70; // rax
  __int64 *v71; // rdx
  __int64 v72; // rax
  __int64 *v73; // rax
  int v74; // r8d
  int v75; // r9d
  __int64 v76; // rbx
  __int64 *v77; // rdx
  __int64 *v78; // rax
  __int64 v79; // rsi
  __int64 *v80; // rax
  char v81; // dl
  __int64 *v82; // rax
  int v83; // r9d
  __int64 v84; // r8
  __int64 v85; // rbx
  __int64 v86; // rax
  __int64 *v87; // rax
  __int64 *v88; // rdx
  __int64 *v89; // rdi
  __int64 v90; // rbx
  __int64 v91; // rax
  int v92; // r8d
  int v93; // r9d
  __int64 *v94; // r14
  _QWORD *v95; // rbx
  __int64 v96; // rdx
  char *v97; // rdi
  __int64 v98; // rsi
  unsigned int v99; // eax
  int v100; // ecx
  char *v101; // rbx
  __int64 v102; // r10
  __int64 v103; // rax
  char *v104; // rax
  __int64 v105; // rdx
  __int64 *v106; // r15
  __int64 *v107; // rsi
  _QWORD *v108; // r13
  __int64 v109; // r12
  unsigned __int64 v110; // rdx
  __int64 v111; // rcx
  __int64 v112; // rdi
  __int64 v113; // rax
  __int64 v114; // rdx
  __int64 v115; // r13
  __int64 v116; // r12
  char *v117; // r15
  __int64 v118; // rbx
  char v119; // si
  _QWORD *i; // r15
  unsigned __int64 v121; // rdi
  unsigned __int64 v122; // rdi
  char *v123; // r12
  int v124; // eax
  __int64 v125; // rdx
  __int64 v126; // rdi
  int v127; // r11d
  unsigned int v128; // ecx
  __int64 *v129; // r12
  __int64 v130; // r8
  unsigned int v131; // eax
  __int64 v132; // rbx
  __int64 v133; // r8
  int v134; // r11d
  __int64 *v135; // rdx
  unsigned int v136; // edi
  __int64 *v137; // rax
  __int64 v138; // rcx
  __int64 v139; // r12
  unsigned int v140; // esi
  __int64 v141; // r12
  int v142; // r11d
  int v143; // r11d
  __int64 v144; // r9
  __int64 v145; // rcx
  int v146; // eax
  __int64 v147; // r8
  __int64 v148; // r12
  __int64 v149; // r15
  __int64 v150; // r12
  __int64 *v151; // rdx
  __int64 *v152; // rdi
  __int64 *v153; // rcx
  __int64 *v154; // rcx
  __int64 *v155; // r8
  __int64 *v156; // rcx
  __int64 v157; // r12
  _QWORD *v158; // rbx
  _QWORD *v159; // r13
  unsigned __int64 v160; // rdi
  unsigned __int64 v161; // rdi
  _QWORD *v162; // r13
  _QWORD *v163; // rbx
  __int64 v164; // r8
  unsigned int v165; // ecx
  __int64 *v166; // rdx
  __int64 v167; // r10
  __int64 v168; // rax
  int v169; // eax
  int v170; // r10d
  int v171; // r10d
  __int64 v172; // r8
  __int64 *v173; // rcx
  __int64 v174; // r15
  int v175; // esi
  __int64 v176; // rdi
  int v177; // edx
  int v178; // r11d
  int v179; // ebx
  _QWORD *v180; // r12
  _QWORD *v181; // r13
  unsigned __int64 v182; // rax
  __int64 v183; // r13
  __int64 v184; // r12
  int v185; // eax
  __int64 v186; // rbx
  int v187; // r13d
  unsigned __int64 v188; // rax
  __int64 *v189; // rax
  int v190; // r9d
  _QWORD *v191; // rbx
  _QWORD *v192; // r13
  __int64 v193; // rdx
  __int64 v194; // r8
  char *v195; // rsi
  unsigned int v196; // eax
  int v197; // ecx
  char **v198; // rbx
  __int64 v199; // rax
  __int64 v200; // rdx
  char *v201; // r8
  char *v202; // rcx
  __int64 v203; // r12
  __int64 v204; // r14
  __int64 v205; // rax
  unsigned __int64 v206; // r15
  __int64 v207; // rcx
  __int64 v208; // rbx
  unsigned __int64 v209; // r8
  __int64 v210; // r15
  __int64 v211; // r13
  unsigned __int64 v212; // r12
  __int64 v213; // rdx
  char v214; // dl
  _QWORD *v215; // r15
  unsigned __int64 v216; // rdi
  unsigned __int64 v217; // rdi
  __int64 v218; // r12
  _QWORD *v219; // r14
  unsigned int v220; // esi
  __int64 *v221; // rax
  __int64 v222; // r9
  __int64 v223; // rdx
  __int64 v224; // rcx
  __int64 v225; // rbx
  unsigned int v226; // r12d
  __int64 v227; // r15
  _QWORD *v228; // r13
  __int64 v229; // r14
  __int64 v230; // rbx
  unsigned __int64 v231; // r13
  __int64 v232; // r13
  unsigned int v233; // r12d
  int v234; // r9d
  __int64 v235; // rcx
  __int64 *v236; // r15
  __int64 v237; // r8
  unsigned int v238; // esi
  __int64 v239; // r10
  __int64 v240; // r8
  __int64 *v241; // rdx
  int v242; // r11d
  unsigned int v243; // edi
  __int64 *v244; // rax
  __int64 v245; // rcx
  __int64 v246; // rbx
  unsigned __int8 v247; // al
  char v248; // cl
  char v249; // dl
  __int64 v250; // rax
  __int64 v251; // rdx
  __int64 v252; // rsi
  int v253; // r11d
  int v254; // r11d
  __int64 v255; // r9
  __int64 v256; // r8
  int v257; // eax
  __int64 v258; // rdi
  int v259; // esi
  __int64 *v260; // rcx
  __int64 v261; // rdi
  int v262; // edi
  __int64 v263; // rax
  unsigned __int8 v264; // al
  int v265; // eax
  __int64 v266; // rax
  __int64 v267; // rdi
  int v268; // edi
  __int64 v269; // rax
  unsigned __int8 v270; // al
  int v271; // eax
  char v272; // al
  __int64 v273; // r8
  unsigned int v274; // esi
  __int64 v275; // r11
  int v276; // r9d
  __int64 v277; // rdx
  unsigned int v278; // edi
  __int64 *v279; // r13
  __int64 v280; // rcx
  int v281; // eax
  __int64 v282; // rdx
  _QWORD *v283; // rax
  _QWORD *j; // rdx
  _QWORD *v285; // r12
  _QWORD *v286; // rbx
  _QWORD *v287; // r13
  unsigned __int64 v288; // rdi
  unsigned __int64 v289; // rdi
  char v290; // bl
  char v291; // cl
  char v292; // al
  unsigned int v293; // r12d
  char v295; // bl
  char v296; // cl
  char v297; // al
  int v298; // eax
  __int64 v299; // rbx
  int v300; // r11d
  int v301; // r11d
  __int64 v302; // r9
  int v303; // esi
  __int64 v304; // r8
  __int64 v305; // rdi
  unsigned int v306; // ecx
  _QWORD *v307; // rdi
  unsigned int v308; // eax
  int v309; // eax
  unsigned __int64 v310; // rax
  unsigned __int64 v311; // rax
  int v312; // ebx
  __int64 v313; // r12
  _QWORD *v314; // rax
  __int64 v315; // rdx
  _QWORD *k; // rdx
  __int64 v317; // r12
  _QWORD *v318; // rbx
  _QWORD *v319; // r13
  unsigned __int64 v320; // rdi
  unsigned __int64 v321; // rdi
  int v322; // eax
  int v323; // r10d
  __int64 v324; // rax
  __int64 v325; // r14
  __int64 v326; // r15
  unsigned __int64 v327; // rbx
  char v328; // al
  char *v329; // r15
  char **v330; // r12
  char *v331; // rbx
  __int64 v332; // rax
  __int64 v333; // r15
  char *v334; // r14
  __int64 v335; // rax
  char *v336; // r12
  _QWORD *v337; // r15
  char **v338; // r13
  __int64 v339; // rbx
  _QWORD *v340; // r14
  unsigned __int64 v341; // rdi
  unsigned __int64 v342; // rdi
  __int64 v343; // rdx
  __int64 v344; // r15
  __int64 v345; // r13
  unsigned __int64 v346; // rbx
  char v347; // al
  __int64 *v348; // rax
  __int64 v349; // r13
  __int64 *v350; // rbx
  __int64 v351; // rax
  __int64 v352; // r13
  __int64 v353; // rax
  __int64 v354; // r13
  __int64 *v355; // r12
  __int64 v356; // rbx
  _QWORD *v357; // r15
  unsigned __int64 v358; // rdi
  unsigned __int64 v359; // rdi
  _QWORD *v360; // rax
  int v361; // edi
  __int64 *v362; // rsi
  __int64 *v363; // rdi
  __int64 *v364; // rcx
  __int64 *v365; // rdi
  __int64 *v366; // rcx
  int v367; // eax
  int v368; // edx
  __int64 *v369; // rax
  int v370; // eax
  int v371; // eax
  __int64 v372; // r11
  unsigned int v373; // edx
  __int64 v374; // rdi
  int v375; // esi
  __int64 *v376; // rcx
  int v377; // eax
  __int64 v378; // r11
  unsigned int v379; // edx
  int v380; // esi
  __int64 v381; // rdi
  int v382; // r10d
  __int64 v383; // [rsp+8h] [rbp-628h]
  unsigned __int64 v384; // [rsp+28h] [rbp-608h]
  __int64 v385; // [rsp+30h] [rbp-600h]
  unsigned __int8 v386; // [rsp+30h] [rbp-600h]
  __int64 v387; // [rsp+38h] [rbp-5F8h]
  char v388; // [rsp+47h] [rbp-5E9h]
  char v389; // [rsp+48h] [rbp-5E8h]
  _BYTE *v391; // [rsp+58h] [rbp-5D8h]
  unsigned __int64 v392; // [rsp+60h] [rbp-5D0h]
  __int64 v393; // [rsp+60h] [rbp-5D0h]
  __int64 v394; // [rsp+68h] [rbp-5C8h]
  __int64 v395; // [rsp+68h] [rbp-5C8h]
  __int64 v396; // [rsp+70h] [rbp-5C0h]
  __int64 v397; // [rsp+70h] [rbp-5C0h]
  char v399; // [rsp+78h] [rbp-5B8h]
  _BYTE *v401; // [rsp+88h] [rbp-5A8h]
  __int64 v402; // [rsp+90h] [rbp-5A0h]
  __int64 *v403; // [rsp+90h] [rbp-5A0h]
  __int64 v404; // [rsp+90h] [rbp-5A0h]
  __int64 v405; // [rsp+90h] [rbp-5A0h]
  unsigned int v406; // [rsp+90h] [rbp-5A0h]
  char *v407; // [rsp+90h] [rbp-5A0h]
  __int64 v408; // [rsp+90h] [rbp-5A0h]
  __int64 *v409; // [rsp+98h] [rbp-598h]
  char v410; // [rsp+98h] [rbp-598h]
  __int64 v412; // [rsp+A0h] [rbp-590h]
  _QWORD *v413; // [rsp+A0h] [rbp-590h]
  _QWORD *v414; // [rsp+A0h] [rbp-590h]
  __int64 v415; // [rsp+A0h] [rbp-590h]
  char v416; // [rsp+A0h] [rbp-590h]
  __int64 v418; // [rsp+A8h] [rbp-588h]
  __int64 v419; // [rsp+A8h] [rbp-588h]
  int v420; // [rsp+A8h] [rbp-588h]
  char *v421; // [rsp+A8h] [rbp-588h]
  __int64 v422; // [rsp+A8h] [rbp-588h]
  __int64 v423; // [rsp+B0h] [rbp-580h]
  _QWORD *v424; // [rsp+B0h] [rbp-580h]
  char **v425; // [rsp+B0h] [rbp-580h]
  __int64 v426; // [rsp+B0h] [rbp-580h]
  char **v427; // [rsp+B0h] [rbp-580h]
  __int64 v428; // [rsp+B0h] [rbp-580h]
  char *v429; // [rsp+B0h] [rbp-580h]
  __int64 v430; // [rsp+B0h] [rbp-580h]
  _QWORD *v431; // [rsp+B0h] [rbp-580h]
  char *v432; // [rsp+B0h] [rbp-580h]
  char *v433; // [rsp+B0h] [rbp-580h]
  __int64 *v434; // [rsp+B0h] [rbp-580h]
  __int64 v435; // [rsp+B0h] [rbp-580h]
  _QWORD *v436; // [rsp+B0h] [rbp-580h]
  __int64 v437; // [rsp+B0h] [rbp-580h]
  __int64 v438; // [rsp+B0h] [rbp-580h]
  _QWORD *v439; // [rsp+B8h] [rbp-578h]
  __int64 v440; // [rsp+B8h] [rbp-578h]
  _QWORD *v441; // [rsp+B8h] [rbp-578h]
  _QWORD *v442; // [rsp+B8h] [rbp-578h]
  __int64 v443; // [rsp+B8h] [rbp-578h]
  __int64 v444; // [rsp+B8h] [rbp-578h]
  int v445; // [rsp+B8h] [rbp-578h]
  __int64 v446; // [rsp+B8h] [rbp-578h]
  __int64 v447; // [rsp+B8h] [rbp-578h]
  __int64 v448; // [rsp+C8h] [rbp-568h] BYREF
  __int64 v449; // [rsp+D0h] [rbp-560h] BYREF
  __int64 v450; // [rsp+D8h] [rbp-558h] BYREF
  _BYTE *v451; // [rsp+E0h] [rbp-550h] BYREF
  __int64 v452; // [rsp+E8h] [rbp-548h]
  _BYTE v453[128]; // [rsp+F0h] [rbp-540h] BYREF
  _BYTE *v454; // [rsp+170h] [rbp-4C0h] BYREF
  __int64 v455; // [rsp+178h] [rbp-4B8h]
  _BYTE v456[128]; // [rsp+180h] [rbp-4B0h] BYREF
  __int64 v457; // [rsp+200h] [rbp-430h] BYREF
  __int64 *v458; // [rsp+208h] [rbp-428h]
  void *s; // [rsp+210h] [rbp-420h]
  _BYTE v460[12]; // [rsp+218h] [rbp-418h]
  _BYTE v461[136]; // [rsp+228h] [rbp-408h] BYREF
  __int64 v462; // [rsp+2B0h] [rbp-380h] BYREF
  __int64 *v463; // [rsp+2B8h] [rbp-378h]
  __int64 *v464; // [rsp+2C0h] [rbp-370h]
  __int64 v465; // [rsp+2C8h] [rbp-368h]
  int v466; // [rsp+2D0h] [rbp-360h]
  _BYTE v467[136]; // [rsp+2D8h] [rbp-358h] BYREF
  __m128i v468; // [rsp+360h] [rbp-2D0h] BYREF
  __int64 v469; // [rsp+370h] [rbp-2C0h] BYREF
  __int64 v470; // [rsp+378h] [rbp-2B8h] BYREF
  _BYTE *v471; // [rsp+380h] [rbp-2B0h]
  _BYTE *v472; // [rsp+388h] [rbp-2A8h]
  __int64 v473; // [rsp+390h] [rbp-2A0h]
  int v474; // [rsp+398h] [rbp-298h]
  _BYTE v475[16]; // [rsp+3A0h] [rbp-290h] BYREF
  __int64 v476; // [rsp+3B0h] [rbp-280h] BYREF
  _BYTE *v477; // [rsp+3B8h] [rbp-278h]
  _BYTE *v478; // [rsp+3C0h] [rbp-270h]
  __int64 v479; // [rsp+3C8h] [rbp-268h]
  int v480; // [rsp+3D0h] [rbp-260h]
  _BYTE v481[16]; // [rsp+3D8h] [rbp-258h] BYREF
  char v482; // [rsp+3E8h] [rbp-248h]
  __int64 *v483; // [rsp+470h] [rbp-1C0h] BYREF
  __int64 v484; // [rsp+478h] [rbp-1B8h]
  __int64 v485; // [rsp+480h] [rbp-1B0h] BYREF
  __int64 v486; // [rsp+488h] [rbp-1A8h] BYREF
  char *v487; // [rsp+490h] [rbp-1A0h] BYREF
  char *v488; // [rsp+498h] [rbp-198h]
  unsigned __int64 v489; // [rsp+4A0h] [rbp-190h]
  int v490; // [rsp+4A8h] [rbp-188h]
  char v491[8]; // [rsp+4B0h] [rbp-180h] BYREF
  char v492[8]; // [rsp+4B8h] [rbp-178h] BYREF
  __int64 v493; // [rsp+4C0h] [rbp-170h] BYREF
  char *v494; // [rsp+4C8h] [rbp-168h] BYREF
  char *v495; // [rsp+4D0h] [rbp-160h]
  unsigned __int64 v496; // [rsp+4D8h] [rbp-158h]
  int v497; // [rsp+4E0h] [rbp-150h]
  char v498[8]; // [rsp+4E8h] [rbp-148h] BYREF
  char v499[8]; // [rsp+4F0h] [rbp-140h] BYREF
  char v500; // [rsp+4F8h] [rbp-138h]
  char v501; // [rsp+500h] [rbp-130h]

  v5 = *(_BYTE *)(a1 + 340) == 0;
  v451 = v453;
  v452 = 0x1000000000LL;
  v455 = 0x1000000000LL;
  v454 = v456;
  v387 = a1 + 312;
  if ( v5 )
  {
    v367 = sub_1602B80(**(__int64 ***)(a1 + 312), "clang.arc.no_objc_arc_exceptions", 0x20u);
    if ( *(_BYTE *)(a1 + 340) )
    {
      *(_DWORD *)(a1 + 336) = v367;
    }
    else
    {
      *(_DWORD *)(a1 + 336) = v367;
      *(_BYTE *)(a1 + 340) = 1;
    }
  }
  v457 = 0;
  v458 = (__int64 *)v461;
  s = v461;
  v463 = (__int64 *)v467;
  v464 = (__int64 *)v467;
  v483 = &v485;
  v484 = 0x1000000000LL;
  *(_QWORD *)v460 = 16;
  *(_DWORD *)&v460[8] = 0;
  v6 = *(_QWORD *)(a2 + 80);
  v462 = 0;
  v465 = 16;
  v466 = 0;
  if ( v6 )
    v6 -= 24;
  v448 = v6;
  v7 = sub_18CDCF0(a3, &v448);
  v10 = v448;
  *((_DWORD *)v7 + 2) = 1;
  v11 = *(_QWORD *)(v10 + 40);
  v468.m128i_i64[0] = v10;
  LODWORD(v469) = 0;
  v11 &= 0xFFFFFFFFFFFFFFF8LL;
  v5 = v11 == 0;
  v12 = v11 - 24;
  v13 = 0;
  if ( !v5 )
    v13 = v12;
  v468.m128i_i64[1] = v13;
  v14 = (unsigned int)v484;
  if ( (unsigned int)v484 >= HIDWORD(v484) )
  {
    sub_16CD150((__int64)&v483, &v485, 0, 24, v8, v9);
    v14 = (unsigned int)v484;
  }
  v15 = (__m128i *)&v483[3 * v14];
  *v15 = _mm_load_si128(&v468);
  v16 = v448;
  v15[1].m128i_i64[0] = v469;
  v17 = v458;
  LODWORD(v484) = v484 + 1;
  if ( s != v458 )
    goto LABEL_9;
  v365 = &v458[*(unsigned int *)&v460[4]];
  if ( v458 != v365 )
  {
    v366 = 0;
    while ( 1 )
    {
      v18 = *v17;
      if ( v16 == *v17 )
        goto LABEL_10;
      if ( v18 == -2 )
        v366 = v17;
      if ( v365 == ++v17 )
      {
        if ( !v366 )
          break;
        *v366 = v16;
        v18 = v448;
        --*(_DWORD *)&v460[8];
        ++v457;
        goto LABEL_10;
      }
    }
  }
  if ( *(_DWORD *)&v460[4] < *(_DWORD *)v460 )
  {
    ++*(_DWORD *)&v460[4];
    *v365 = v16;
    v18 = v448;
    ++v457;
  }
  else
  {
LABEL_9:
    sub_16CCBA0((__int64)&v457, v16);
    v18 = v448;
  }
LABEL_10:
  v19 = v463;
  if ( v464 != v463 )
    goto LABEL_11;
  v363 = &v463[HIDWORD(v465)];
  if ( v463 != v363 )
  {
    v364 = 0;
    while ( 1 )
    {
      v16 = *v19;
      if ( *v19 == v18 )
        goto LABEL_12;
      if ( v16 == -2 )
        v364 = v19;
      if ( v363 == ++v19 )
      {
        if ( !v364 )
          break;
        *v364 = v18;
        --v466;
        ++v462;
        goto LABEL_12;
      }
    }
  }
  if ( HIDWORD(v465) < (unsigned int)v465 )
  {
    ++HIDWORD(v465);
    *v363 = v18;
    ++v462;
  }
  else
  {
LABEL_11:
    sub_16CCBA0((__int64)&v462, v18);
  }
LABEL_12:
  v20 = (unsigned int)v484;
  do
  {
    while ( 1 )
    {
      v21 = (__int64)&v483[3 * v20 - 3];
      v449 = *(_QWORD *)v21;
      if ( (*(_QWORD *)(v449 + 40) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v22 = sub_15F4D60((*(_QWORD *)(v449 + 40) & 0xFFFFFFFFFFFFFFF8LL) - 24);
        v21 = (__int64)&v483[3 * (unsigned int)v484 - 3];
      }
      else
      {
        v22 = 0;
      }
      v23 = *(_DWORD *)(v21 + 16);
      if ( v22 == v23 )
        break;
      while ( 1 )
      {
        v28 = *(_QWORD *)(v21 + 8);
        *(_DWORD *)(v21 + 16) = v23 + 1;
        v450 = sub_15F4DF0(v28, v23);
        v29 = v450;
        v30 = v458;
        if ( s != v458 )
          break;
        LODWORD(v16) = *(_DWORD *)&v460[4];
        v31 = &v458[*(unsigned int *)&v460[4]];
        if ( v458 == v31 )
        {
LABEL_61:
          if ( *(_DWORD *)&v460[4] < *(_DWORD *)v460 )
          {
            LODWORD(v16) = ++*(_DWORD *)&v460[4];
            *v31 = v450;
            v29 = v450;
            ++v457;
            goto LABEL_32;
          }
          break;
        }
        v32 = 0;
        while ( 1 )
        {
          v24 = *v30;
          if ( v450 == *v30 )
            break;
          if ( v24 == -2 )
          {
            v32 = v30;
            if ( v31 == v30 + 1 )
              goto LABEL_31;
            ++v30;
          }
          else if ( v31 == ++v30 )
          {
            if ( !v32 )
              goto LABEL_61;
LABEL_31:
            *v32 = v450;
            --*(_DWORD *)&v460[8];
            ++v457;
            goto LABEL_32;
          }
        }
LABEL_18:
        v26 = v463;
        if ( v464 == v463 )
        {
          v27 = &v463[HIDWORD(v465)];
          if ( v463 == v27 )
          {
            v154 = v463;
          }
          else
          {
            do
            {
              if ( *v26 == v24 )
                break;
              ++v26;
            }
            while ( v27 != v26 );
            v154 = &v463[HIDWORD(v465)];
          }
          goto LABEL_50;
        }
        v27 = &v464[(unsigned int)v465];
        v26 = sub_16CC9F0((__int64)&v462, v24);
        if ( *v26 == v24 )
        {
          if ( v464 == v463 )
            v154 = &v464[HIDWORD(v465)];
          else
            v154 = &v464[(unsigned int)v465];
LABEL_50:
          while ( v154 != v26 && (unsigned __int64)*v26 >= 0xFFFFFFFFFFFFFFFELL )
            ++v26;
          goto LABEL_22;
        }
        if ( v464 == v463 )
        {
          v26 = &v464[HIDWORD(v465)];
          v154 = v26;
          goto LABEL_50;
        }
        v26 = &v464[(unsigned int)v465];
LABEL_22:
        if ( v27 == v26 )
        {
          v50 = sub_18CDCF0(a3, &v449);
          v53 = v450;
          v54 = v50;
          v55 = *((unsigned int *)v50 + 42);
          if ( (unsigned int)v55 >= *((_DWORD *)v54 + 43) )
          {
            sub_16CD150((__int64)(v54 + 20), v54 + 22, 0, 8, v51, v52);
            v55 = *((unsigned int *)v54 + 42);
          }
          *(_QWORD *)(v54[20] + 8 * v55) = v53;
          ++*((_DWORD *)v54 + 42);
          v56 = sub_18CDCF0(a3, &v450);
          v57 = v449;
          v58 = v56;
          v59 = *((unsigned int *)v56 + 34);
          if ( (unsigned int)v59 >= *((_DWORD *)v58 + 35) )
          {
            sub_16CD150((__int64)(v58 + 16), v58 + 18, 0, 8, v16, v9);
            v59 = *((unsigned int *)v58 + 34);
          }
          *(_QWORD *)(v58[16] + 8 * v59) = v57;
          ++*((_DWORD *)v58 + 34);
        }
        v21 = (__int64)&v483[3 * (unsigned int)v484 - 3];
        v23 = *(_DWORD *)(v21 + 16);
        if ( v23 == v22 )
          goto LABEL_63;
      }
      sub_16CCBA0((__int64)&v457, v450);
      v24 = v450;
      if ( !v25 )
        goto LABEL_18;
      v29 = v450;
LABEL_32:
      v33 = *(_QWORD *)(v29 + 40);
      v468.m128i_i64[0] = v29;
      LODWORD(v469) = 0;
      v33 &= 0xFFFFFFFFFFFFFFF8LL;
      v5 = v33 == 0;
      v34 = v33 - 24;
      v35 = 0;
      if ( !v5 )
        v35 = v34;
      v468.m128i_i64[1] = v35;
      v36 = (unsigned int)v484;
      if ( (unsigned int)v484 >= HIDWORD(v484) )
      {
        sub_16CD150((__int64)&v483, &v485, 0, 24, v16, v9);
        v36 = (unsigned int)v484;
      }
      v37 = (__m128i *)&v483[3 * v36];
      *v37 = _mm_load_si128(&v468);
      v37[1].m128i_i64[0] = v469;
      LODWORD(v484) = v484 + 1;
      v38 = sub_18CDCF0(a3, &v449);
      v41 = v450;
      v42 = v38;
      v43 = *((unsigned int *)v38 + 42);
      if ( (unsigned int)v43 >= *((_DWORD *)v42 + 43) )
      {
        sub_16CD150((__int64)(v42 + 20), v42 + 22, 0, 8, v39, v40);
        v43 = *((unsigned int *)v42 + 42);
      }
      *(_QWORD *)(v42[20] + 8 * v43) = v41;
      ++*((_DWORD *)v42 + 42);
      v44 = sub_18CDCF0(a3, &v450);
      v46 = v449;
      v47 = v44;
      v48 = *((unsigned int *)v44 + 34);
      if ( (unsigned int)v48 >= *((_DWORD *)v47 + 35) )
      {
        sub_16CD150((__int64)(v47 + 16), v47 + 18, 0, 8, v45, v9);
        v48 = *((unsigned int *)v47 + 34);
      }
      *(_QWORD *)(v47[16] + 8 * v48) = v46;
      ++*((_DWORD *)v47 + 34);
      v49 = v463;
      if ( v464 == v463 )
      {
        v152 = &v463[HIDWORD(v465)];
        LODWORD(v16) = HIDWORD(v465);
        if ( v463 == v152 )
        {
LABEL_535:
          if ( HIDWORD(v465) >= (unsigned int)v465 )
            goto LABEL_41;
          LODWORD(v16) = ++HIDWORD(v465);
          *v152 = v450;
          ++v462;
        }
        else
        {
          v153 = 0;
          while ( v450 != *v49 )
          {
            if ( *v49 == -2 )
              v153 = v49;
            if ( v152 == ++v49 )
            {
              if ( !v153 )
                goto LABEL_535;
              *v153 = v450;
              --v466;
              ++v462;
              break;
            }
          }
        }
      }
      else
      {
LABEL_41:
        sub_16CCBA0((__int64)&v462, v450);
      }
      v20 = (unsigned int)v484;
    }
LABEL_63:
    v60 = v449;
    v61 = v463;
    if ( v464 == v463 )
    {
      v151 = &v463[HIDWORD(v465)];
      if ( v463 == v151 )
      {
LABEL_185:
        v61 = &v463[HIDWORD(v465)];
      }
      else
      {
        while ( v449 != *v61 )
        {
          if ( v151 == ++v61 )
            goto LABEL_185;
        }
      }
    }
    else
    {
      v61 = sub_16CC9F0((__int64)&v462, v449);
      if ( v60 == *v61 )
      {
        if ( v464 == v463 )
          v151 = &v464[HIDWORD(v465)];
        else
          v151 = &v464[(unsigned int)v465];
      }
      else
      {
        if ( v464 != v463 )
          goto LABEL_66;
        v61 = &v464[HIDWORD(v465)];
        v151 = v61;
      }
    }
    if ( v151 != v61 )
    {
      *v61 = -2;
      v62 = (unsigned int)v452;
      ++v466;
      if ( (unsigned int)v452 < HIDWORD(v452) )
        goto LABEL_67;
      goto LABEL_173;
    }
LABEL_66:
    v62 = (unsigned int)v452;
    if ( (unsigned int)v452 < HIDWORD(v452) )
      goto LABEL_67;
LABEL_173:
    sub_16CD150((__int64)&v451, v453, 0, 8, v16, v9);
    v62 = (unsigned int)v452;
LABEL_67:
    *(_QWORD *)&v451[8 * v62] = v449;
    LODWORD(v452) = v452 + 1;
    v5 = (_DWORD)v484 == 1;
    v20 = (unsigned int)(v484 - 1);
    LODWORD(v484) = v484 - 1;
  }
  while ( !v5 );
  ++v457;
  if ( s == v458 )
    goto LABEL_73;
  v63 = 4 * (*(_DWORD *)&v460[4] - *(_DWORD *)&v460[8]);
  if ( v63 < 0x20 )
    v63 = 32;
  if ( *(_DWORD *)v460 > v63 )
  {
    sub_16CC920((__int64)&v457);
  }
  else
  {
    memset(s, -1, 8LL * *(unsigned int *)v460);
LABEL_73:
    *(_QWORD *)&v460[4] = 0;
  }
  v468.m128i_i64[1] = 0x1000000000LL;
  v468.m128i_i64[0] = (__int64)&v469;
  v64 = *(_QWORD *)(a2 + 80);
  v412 = a2 + 72;
  if ( v64 == v412 )
    goto LABEL_109;
LABEL_77:
  while ( 2 )
  {
    v65 = v64 - 24;
    if ( !v64 )
      v65 = 0;
    v450 = v65;
    v66 = sub_18CDCF0(a3, &v450);
    if ( !*((_DWORD *)v66 + 42) )
    {
      *((_DWORD *)v66 + 3) = 1;
      v68 = v66[16];
      v69 = v468.m128i_u32[2];
      if ( v468.m128i_i32[2] >= (unsigned __int32)v468.m128i_i32[3] )
      {
        v408 = v68;
        sub_16CD150((__int64)&v468, &v469, 0, 16, v68, v67);
        v69 = v468.m128i_u32[2];
        v68 = v408;
      }
      v70 = (__int64 *)(v468.m128i_i64[0] + 16 * v69);
      *v70 = v65;
      v70[1] = v68;
      v71 = v458;
      v72 = (unsigned int)++v468.m128i_i32[2];
      if ( s == v458 )
      {
        v155 = &v458[*(unsigned int *)&v460[4]];
        if ( v458 == v155 )
        {
LABEL_598:
          if ( *(_DWORD *)&v460[4] >= *(_DWORD *)v460 )
            goto LABEL_83;
          ++*(_DWORD *)&v460[4];
          *v155 = v65;
          v72 = v468.m128i_u32[2];
          ++v457;
        }
        else
        {
          v156 = 0;
          while ( v65 != *v71 )
          {
            if ( *v71 == -2 )
              v156 = v71;
            if ( v155 == ++v71 )
            {
              if ( !v156 )
                goto LABEL_598;
              *v156 = v65;
              v72 = v468.m128i_u32[2];
              --*(_DWORD *)&v460[8];
              ++v457;
              break;
            }
          }
        }
      }
      else
      {
LABEL_83:
        sub_16CCBA0((__int64)&v457, v65);
        v72 = v468.m128i_u32[2];
      }
      if ( !(_DWORD)v72 )
        goto LABEL_76;
LABEL_85:
      v73 = sub_18CDCF0(a3, (__int64 *)(v468.m128i_i64[0] + 16 * v72 - 16));
      v76 = v73[16] + 8LL * *((unsigned int *)v73 + 34);
      while ( 1 )
      {
        while ( 1 )
        {
          v77 = (__int64 *)(v468.m128i_i64[0] + 16LL * v468.m128i_u32[2] - 16);
          v78 = (__int64 *)v77[1];
          if ( (__int64 *)v76 == v78 )
          {
            v90 = *v77;
            v91 = (unsigned int)v455;
            --v468.m128i_i32[2];
            if ( (unsigned int)v455 >= HIDWORD(v455) )
            {
              sub_16CD150((__int64)&v454, v456, 0, 8, v74, v75);
              v91 = (unsigned int)v455;
            }
            *(_QWORD *)&v454[8 * v91] = v90;
            v72 = v468.m128i_u32[2];
            LODWORD(v455) = v455 + 1;
            if ( !v468.m128i_i32[2] )
            {
              v64 = *(_QWORD *)(v64 + 8);
              if ( v412 == v64 )
                goto LABEL_107;
              goto LABEL_77;
            }
            goto LABEL_85;
          }
          v77[1] = (__int64)(v78 + 1);
          v79 = *v78;
          v80 = v458;
          v450 = v79;
          if ( s == v458 )
            break;
LABEL_88:
          sub_16CCBA0((__int64)&v457, v79);
          if ( v81 )
            goto LABEL_89;
        }
        v74 = *(_DWORD *)&v460[4];
        v88 = &v458[*(unsigned int *)&v460[4]];
        if ( v458 == v88 )
        {
LABEL_101:
          if ( *(_DWORD *)&v460[4] < *(_DWORD *)v460 )
          {
            ++*(_DWORD *)&v460[4];
            *v88 = v79;
            ++v457;
LABEL_89:
            v82 = sub_18CDCF0(a3, &v450);
            v84 = v450;
            v85 = v82[16];
            v86 = v468.m128i_u32[2];
            if ( v468.m128i_i32[2] >= (unsigned __int32)v468.m128i_i32[3] )
            {
              v404 = v450;
              sub_16CD150((__int64)&v468, &v469, 0, 16, v450, v83);
              v86 = v468.m128i_u32[2];
              v84 = v404;
            }
            v87 = (__int64 *)(v468.m128i_i64[0] + 16 * v86);
            *v87 = v84;
            v87[1] = v85;
            v72 = (unsigned int)++v468.m128i_i32[2];
            goto LABEL_85;
          }
          goto LABEL_88;
        }
        v89 = 0;
        while ( v79 != *v80 )
        {
          if ( *v80 == -2 )
          {
            v89 = v80;
            if ( v88 == v80 + 1 )
              goto LABEL_98;
            ++v80;
          }
          else if ( v88 == ++v80 )
          {
            if ( !v89 )
              goto LABEL_101;
LABEL_98:
            *v89 = v79;
            --*(_DWORD *)&v460[8];
            ++v457;
            goto LABEL_89;
          }
        }
      }
    }
LABEL_76:
    v64 = *(_QWORD *)(v64 + 8);
    if ( v412 != v64 )
      continue;
    break;
  }
LABEL_107:
  if ( (__int64 *)v468.m128i_i64[0] != &v469 )
    _libc_free(v468.m128i_u64[0]);
LABEL_109:
  if ( v483 != &v485 )
    _libc_free((unsigned __int64)v483);
  if ( v464 != v463 )
    _libc_free((unsigned __int64)v464);
  if ( s != v458 )
    _libc_free((unsigned __int64)s);
  v388 = 0;
  v392 = (unsigned __int64)v454;
  v401 = &v454[8 * (unsigned int)v455];
  if ( v454 == v401 )
    goto LABEL_252;
  while ( 2 )
  {
    v468.m128i_i64[0] = *((_QWORD *)v401 - 1);
    v94 = sub_18CDCF0(a3, v468.m128i_i64);
    v396 = (__int64)(v94 + 1);
    v95 = (_QWORD *)v94[20];
    v439 = v95;
    v413 = &v95[*((unsigned int *)v94 + 42)];
    if ( v95 == v413 )
      goto LABEL_238;
    v96 = *(unsigned int *)(a3 + 24);
    v97 = *(char **)(a3 + 8);
    if ( (_DWORD)v96 )
    {
      v98 = *v95;
      v92 = v96 - 1;
      v99 = (v96 - 1) & (((unsigned int)*v95 >> 9) ^ ((unsigned int)*v95 >> 4));
      v100 = 1;
      v101 = &v97[192 * v99];
      v102 = *(_QWORD *)v101;
      if ( *(_QWORD *)v101 == v98 )
        goto LABEL_119;
      while ( v102 != -8 )
      {
        v93 = v100 + 1;
        v99 = v92 & (v100 + v99);
        v101 = &v97[192 * v99];
        v102 = *(_QWORD *)v101;
        if ( v98 == *(_QWORD *)v101 )
          goto LABEL_119;
        ++v100;
      }
    }
    v101 = &v97[192 * v96];
LABEL_119:
    v394 = (__int64)(v94 + 9);
    if ( v94 + 9 != (__int64 *)(v101 + 72) )
    {
      v97 = (char *)v94[10];
      j___libc_free_0(v97);
      v103 = *((unsigned int *)v101 + 24);
      *((_DWORD *)v94 + 24) = v103;
      if ( (_DWORD)v103 )
      {
        v104 = (char *)sub_22077B0(16 * v103);
        v105 = *((unsigned int *)v94 + 24);
        v94[10] = (__int64)v104;
        v97 = v104;
        *((_DWORD *)v94 + 22) = *((_DWORD *)v101 + 22);
        *((_DWORD *)v94 + 23) = *((_DWORD *)v101 + 23);
        memcpy(v104, *((const void **)v101 + 10), 16 * v105);
      }
      else
      {
        v94[10] = 0;
        v94[11] = 0;
      }
    }
    if ( v94 + 13 == (__int64 *)(v101 + 104) )
      goto LABEL_141;
    v106 = (__int64 *)*((_QWORD *)v101 + 14);
    v107 = (__int64 *)*((_QWORD *)v101 + 13);
    v108 = (_QWORD *)v94[13];
    v109 = (char *)v106 - (char *)v107;
    v110 = v94[15] - (_QWORD)v108;
    if ( v110 >= (char *)v106 - (char *)v107 )
    {
      v111 = v94[14];
      v112 = v111 - (_QWORD)v108;
      v113 = v111 - (_QWORD)v108;
      if ( v109 > (unsigned __int64)(v111 - (_QWORD)v108) )
      {
        v343 = 0x86BCA1AF286BCA1BLL * (v112 >> 3);
        if ( v112 > 0 )
        {
          v432 = v101;
          v344 = (__int64)(v107 + 4);
          v345 = (__int64)(v108 + 4);
          v346 = 0x86BCA1AF286BCA1BLL * (v112 >> 3);
          do
          {
            *(_QWORD *)(v345 - 32) = *(_QWORD *)(v344 - 32);
            *(_BYTE *)(v345 - 24) = *(_BYTE *)(v344 - 24);
            *(_BYTE *)(v345 - 23) = *(_BYTE *)(v344 - 23);
            *(_BYTE *)(v345 - 22) = *(_BYTE *)(v344 - 22);
            *(_BYTE *)(v345 - 16) = *(_BYTE *)(v344 - 16);
            *(_BYTE *)(v345 - 15) = *(_BYTE *)(v344 - 15);
            *(_QWORD *)(v345 - 8) = *(_QWORD *)(v344 - 8);
            if ( v345 != v344 )
              sub_16CCD50(v345, v344, v343, v111, v92, v93);
            if ( v344 + 56 != v345 + 56 )
              sub_16CCD50(v345 + 56, v344 + 56, v343, v111, v92, v93);
            v347 = *(_BYTE *)(v344 + 112);
            v345 += 152;
            v344 += 152;
            *(_BYTE *)(v345 - 40) = v347;
            --v346;
          }
          while ( v346 );
          v101 = v432;
          v111 = v94[14];
          v108 = (_QWORD *)v94[13];
          v106 = (__int64 *)*((_QWORD *)v432 + 14);
          v107 = (__int64 *)*((_QWORD *)v432 + 13);
          v113 = v111 - (_QWORD)v108;
        }
        v348 = (__int64 *)((char *)v107 + v113);
        if ( v348 == v106 )
        {
          v123 = (char *)v108 + v109;
        }
        else
        {
          v433 = v101;
          v349 = v111;
          v350 = v348;
          do
          {
            if ( v349 )
            {
              *(_QWORD *)v349 = *v350;
              *(_BYTE *)(v349 + 8) = *((_BYTE *)v350 + 8);
              *(_BYTE *)(v349 + 9) = *((_BYTE *)v350 + 9);
              *(_BYTE *)(v349 + 10) = *((_BYTE *)v350 + 10);
              *(_BYTE *)(v349 + 16) = *((_BYTE *)v350 + 16);
              *(_BYTE *)(v349 + 17) = *((_BYTE *)v350 + 17);
              *(_QWORD *)(v349 + 24) = v350[3];
              sub_16CCCB0((_QWORD *)(v349 + 32), v349 + 72, (__int64)(v350 + 4));
              sub_16CCCB0((_QWORD *)(v349 + 88), v349 + 128, (__int64)(v350 + 11));
              *(_BYTE *)(v349 + 144) = *((_BYTE *)v350 + 144);
            }
            v350 += 19;
            v349 += 152;
          }
          while ( v350 != v106 );
          v101 = v433;
          v123 = (char *)(v94[13] + v109);
        }
      }
      else
      {
        if ( v109 > 0 )
        {
          v385 = v94[14];
          v423 = *((_QWORD *)v101 + 14) - (_QWORD)v107;
          v114 = 0x86BCA1AF286BCA1BLL * (v109 >> 3);
          v402 = v94[13];
          v115 = (__int64)(v108 + 4);
          v116 = (__int64)(v107 + 4);
          v117 = v101;
          v118 = v114;
          do
          {
            *(_QWORD *)(v115 - 32) = *(_QWORD *)(v116 - 32);
            *(_BYTE *)(v115 - 24) = *(_BYTE *)(v116 - 24);
            *(_BYTE *)(v115 - 23) = *(_BYTE *)(v116 - 23);
            *(_BYTE *)(v115 - 22) = *(_BYTE *)(v116 - 22);
            *(_BYTE *)(v115 - 16) = *(_BYTE *)(v116 - 16);
            *(_BYTE *)(v115 - 15) = *(_BYTE *)(v116 - 15);
            *(_QWORD *)(v115 - 8) = *(_QWORD *)(v116 - 8);
            if ( v115 != v116 )
              sub_16CCD50(v115, v116, v114, v111, v92, v93);
            if ( v116 + 56 != v115 + 56 )
              sub_16CCD50(v115 + 56, v116 + 56, v114, v111, v92, v93);
            v119 = *(_BYTE *)(v116 + 112);
            v115 += 152;
            v116 += 152;
            *(_BYTE *)(v115 - 40) = v119;
            --v118;
          }
          while ( v118 );
          v109 = v423;
          v101 = v117;
          v111 = v385;
          v108 = (_QWORD *)(v423 + v402);
        }
        for ( i = (_QWORD *)v111; i != v108; v108 += 19 )
        {
          v121 = v108[13];
          if ( v121 != v108[12] )
            _libc_free(v121);
          v122 = v108[6];
          if ( v122 != v108[5] )
            _libc_free(v122);
        }
        v123 = (char *)(v94[13] + v109);
      }
      goto LABEL_140;
    }
    if ( !v109 )
    {
      v352 = 0;
      goto LABEL_514;
    }
    if ( (unsigned __int64)v109 > 0x7FFFFFFFFFFFFFC8LL )
LABEL_613:
      sub_4261EA(v97, v107, v110);
    v434 = (__int64 *)*((_QWORD *)v101 + 13);
    v351 = sub_22077B0(*((_QWORD *)v101 + 14) - (_QWORD)v107);
    v107 = v434;
    v352 = v351;
LABEL_514:
    v353 = v352;
    if ( v106 != v107 )
    {
      v435 = v352;
      v354 = v109;
      v355 = v107;
      v407 = v101;
      v356 = v353;
      do
      {
        if ( v356 )
        {
          *(_QWORD *)v356 = *v355;
          *(_BYTE *)(v356 + 8) = *((_BYTE *)v355 + 8);
          *(_BYTE *)(v356 + 9) = *((_BYTE *)v355 + 9);
          *(_BYTE *)(v356 + 10) = *((_BYTE *)v355 + 10);
          *(_BYTE *)(v356 + 16) = *((_BYTE *)v355 + 16);
          *(_BYTE *)(v356 + 17) = *((_BYTE *)v355 + 17);
          *(_QWORD *)(v356 + 24) = v355[3];
          sub_16CCCB0((_QWORD *)(v356 + 32), v356 + 72, (__int64)(v355 + 4));
          sub_16CCCB0((_QWORD *)(v356 + 88), v356 + 128, (__int64)(v355 + 11));
          *(_BYTE *)(v356 + 144) = *((_BYTE *)v355 + 144);
        }
        v355 += 19;
        v356 += 152;
      }
      while ( v106 != v355 );
      v109 = v354;
      v101 = v407;
      v352 = v435;
    }
    v357 = (_QWORD *)v94[13];
    v436 = (_QWORD *)v94[14];
    if ( v436 != v357 )
    {
      do
      {
        v358 = v357[13];
        if ( v358 != v357[12] )
          _libc_free(v358);
        v359 = v357[6];
        if ( v359 != v357[5] )
          _libc_free(v359);
        v357 += 19;
      }
      while ( v436 != v357 );
      v357 = (_QWORD *)v94[13];
    }
    if ( v357 )
      j_j___libc_free_0(v357, v94[15] - (_QWORD)v357);
    v123 = (char *)(v352 + v109);
    v94[13] = v352;
    v94[15] = (__int64)v123;
LABEL_140:
    v94[14] = (__int64)v123;
LABEL_141:
    v124 = *((_DWORD *)v101 + 3);
    *((_DWORD *)v94 + 3) = v124;
    v424 = v439 + 1;
    if ( v413 != v439 + 1 )
    {
LABEL_142:
      v125 = *(unsigned int *)(a3 + 24);
      v126 = *(_QWORD *)(a3 + 8);
      if ( (_DWORD)v125 )
      {
        v127 = 1;
        v128 = (v125 - 1) & (((unsigned int)*v424 >> 9) ^ ((unsigned int)*v424 >> 4));
        v129 = (__int64 *)(v126 + 192LL * v128);
        v130 = *v129;
        if ( *v424 == *v129 )
          goto LABEL_144;
        while ( v130 != -8 )
        {
          v128 = (v125 - 1) & (v127 + v128);
          v129 = (__int64 *)(v126 + 192LL * v128);
          v130 = *v129;
          if ( *v424 == *v129 )
            goto LABEL_144;
          ++v127;
        }
      }
      v129 = (__int64 *)(v126 + 192 * v125);
LABEL_144:
      if ( v124 == -1 )
        goto LABEL_203;
      v131 = *((_DWORD *)v129 + 3) + v124;
      *((_DWORD *)v94 + 3) = v131;
      if ( v131 == -1 )
      {
        sub_18CEB70(v394);
        v317 = v94[13];
        v318 = (_QWORD *)v94[14];
        if ( (_QWORD *)v317 != v318 )
        {
          v319 = (_QWORD *)v94[13];
          do
          {
            v320 = v319[13];
            if ( v320 != v319[12] )
              _libc_free(v320);
            v321 = v319[6];
            if ( v321 != v319[5] )
              _libc_free(v321);
            v319 += 19;
          }
          while ( v318 != v319 );
          v94[14] = v317;
        }
        goto LABEL_203;
      }
      if ( v131 < *((_DWORD *)v129 + 3) )
      {
        *((_DWORD *)v94 + 3) = -1;
        sub_18CEB70(v394);
        v157 = v94[13];
        v158 = (_QWORD *)v94[14];
        if ( (_QWORD *)v157 != v158 )
        {
          v159 = (_QWORD *)v94[13];
          do
          {
            v160 = v159[13];
            if ( v160 != v159[12] )
              _libc_free(v160);
            v161 = v159[6];
            if ( v161 != v159[5] )
              _libc_free(v161);
            v159 += 19;
          }
          while ( v158 != v159 );
          v94[14] = v157;
        }
        goto LABEL_203;
      }
      v132 = v129[13];
      v440 = v129[14];
      if ( v132 == v440 )
        goto LABEL_206;
      v403 = v129;
LABEL_156:
      v140 = *((_DWORD *)v94 + 24);
      v141 = *(_QWORD *)v132;
      if ( !v140 )
      {
        ++v94[9];
        goto LABEL_158;
      }
      v133 = v94[10];
      v134 = 1;
      v135 = 0;
      v136 = (v140 - 1) & (((unsigned int)v141 >> 9) ^ ((unsigned int)v141 >> 4));
      v137 = (__int64 *)(v133 + 16LL * v136);
      v138 = *v137;
      if ( *v137 == v141 )
        goto LABEL_150;
      while ( v138 != -8 )
      {
        if ( v135 || v138 != -16 )
          v137 = v135;
        v136 = (v140 - 1) & (v134 + v136);
        v138 = *(_QWORD *)(v133 + 16LL * v136);
        if ( v141 == v138 )
        {
          v137 = (__int64 *)(v133 + 16LL * v136);
LABEL_150:
          v139 = v94[13] + 152 * v137[1] + 8;
          LOWORD(v483) = *(_WORD *)(v132 + 8);
          BYTE2(v483) = *(_BYTE *)(v132 + 10);
          LOWORD(v484) = *(_WORD *)(v132 + 16);
          v485 = *(_QWORD *)(v132 + 24);
          sub_16CCCB0(&v486, (__int64)v491, v132 + 32);
          sub_16CCCB0(&v493, (__int64)v498, v132 + 88);
          v500 = *(_BYTE *)(v132 + 144);
LABEL_151:
          sub_18DBA00(v139, &v483, 0);
          if ( v495 != v494 )
            _libc_free((unsigned __int64)v495);
          if ( v488 != v487 )
            _libc_free((unsigned __int64)v488);
          v132 += 152;
          if ( v440 != v132 )
            goto LABEL_156;
          v129 = v403;
LABEL_206:
          v162 = (_QWORD *)v94[14];
          v163 = (_QWORD *)v94[13];
          if ( v163 == v162 )
          {
LABEL_203:
            if ( v413 == ++v424 )
              goto LABEL_238;
            v124 = *((_DWORD *)v94 + 3);
            goto LABEL_142;
          }
          while ( 2 )
          {
            v168 = *((unsigned int *)v129 + 24);
            if ( (_DWORD)v168 )
            {
              v164 = v129[10];
              v165 = (v168 - 1) & (((unsigned int)*v163 >> 9) ^ ((unsigned int)*v163 >> 4));
              v166 = (__int64 *)(v164 + 16LL * v165);
              v167 = *v166;
              if ( *v163 != *v166 )
              {
                v177 = 1;
                while ( v167 != -8 )
                {
                  v178 = v177 + 1;
                  v165 = (v168 - 1) & (v177 + v165);
                  v166 = (__int64 *)(v164 + 16LL * v165);
                  v167 = *v166;
                  if ( *v163 == *v166 )
                    goto LABEL_209;
                  v177 = v178;
                }
                goto LABEL_213;
              }
LABEL_209:
              if ( v166 == (__int64 *)(v164 + 16 * v168) || v129[14] == v129[13] + 152 * v166[1] )
                goto LABEL_213;
            }
            else
            {
LABEL_213:
              v487 = v491;
              v488 = v491;
              v483 = 0;
              v484 = 0;
              v485 = 0;
              v486 = 0;
              v489 = 2;
              v490 = 0;
              v493 = 0;
              v494 = v498;
              v495 = v498;
              v496 = 2;
              v497 = 0;
              v500 = 0;
              sub_18DBA00(v163 + 1, &v483, 0);
              if ( v495 != v494 )
                _libc_free((unsigned __int64)v495);
              if ( v488 != v487 )
                _libc_free((unsigned __int64)v488);
            }
            v163 += 19;
            if ( v162 == v163 )
              goto LABEL_203;
            continue;
          }
        }
        ++v134;
        v135 = v137;
        v137 = (__int64 *)(v133 + 16LL * v136);
      }
      if ( !v135 )
        v135 = v137;
      v169 = *((_DWORD *)v94 + 22);
      ++v94[9];
      v146 = v169 + 1;
      if ( 4 * v146 >= 3 * v140 )
      {
LABEL_158:
        sub_18D2390(v394, 2 * v140);
        v142 = *((_DWORD *)v94 + 24);
        if ( !v142 )
          goto LABEL_635;
        v143 = v142 - 1;
        v144 = v94[10];
        LODWORD(v145) = v143 & (((unsigned int)v141 >> 9) ^ ((unsigned int)v141 >> 4));
        v146 = *((_DWORD *)v94 + 22) + 1;
        v135 = (__int64 *)(v144 + 16LL * (unsigned int)v145);
        v147 = *v135;
        if ( v141 != *v135 )
        {
          v361 = 1;
          v362 = 0;
          while ( v147 != -8 )
          {
            if ( v147 == -16 && !v362 )
              v362 = v135;
            v145 = v143 & (unsigned int)(v145 + v361);
            v135 = (__int64 *)(v144 + 16 * v145);
            v147 = *v135;
            if ( v141 == *v135 )
              goto LABEL_160;
            ++v361;
          }
          if ( v362 )
            v135 = v362;
        }
      }
      else if ( v140 - *((_DWORD *)v94 + 23) - v146 <= v140 >> 3 )
      {
        sub_18D2390(v394, v140);
        v170 = *((_DWORD *)v94 + 24);
        if ( v170 )
        {
          v171 = v170 - 1;
          v172 = v94[10];
          v173 = 0;
          LODWORD(v174) = v171 & (((unsigned int)v141 >> 9) ^ ((unsigned int)v141 >> 4));
          v175 = 1;
          v146 = *((_DWORD *)v94 + 22) + 1;
          v135 = (__int64 *)(v172 + 16LL * (unsigned int)v174);
          v176 = *v135;
          if ( v141 != *v135 )
          {
            while ( v176 != -8 )
            {
              if ( v176 == -16 && !v173 )
                v173 = v135;
              v174 = v171 & (unsigned int)(v174 + v175);
              v135 = (__int64 *)(v172 + 16 * v174);
              v176 = *v135;
              if ( v141 == *v135 )
                goto LABEL_160;
              ++v175;
            }
            if ( v173 )
              v135 = v173;
          }
          goto LABEL_160;
        }
LABEL_635:
        ++*((_DWORD *)v94 + 22);
        BUG();
      }
LABEL_160:
      *((_DWORD *)v94 + 22) = v146;
      if ( *v135 != -8 )
        --*((_DWORD *)v94 + 23);
      *v135 = v141;
      v135[1] = 0;
      v148 = v94[14] - v94[13];
      v135[1] = 0x86BCA1AF286BCA1BLL * (v148 >> 3);
      v149 = v94[14];
      if ( v149 == v94[15] )
      {
        sub_18D1390(v94 + 13, (char *)v94[14], (__int64 *)v132);
      }
      else
      {
        if ( v149 )
        {
          *(_QWORD *)v149 = *(_QWORD *)v132;
          *(_BYTE *)(v149 + 8) = *(_BYTE *)(v132 + 8);
          *(_BYTE *)(v149 + 9) = *(_BYTE *)(v132 + 9);
          *(_BYTE *)(v149 + 10) = *(_BYTE *)(v132 + 10);
          *(_BYTE *)(v149 + 16) = *(_BYTE *)(v132 + 16);
          *(_BYTE *)(v149 + 17) = *(_BYTE *)(v132 + 17);
          *(_QWORD *)(v149 + 24) = *(_QWORD *)(v132 + 24);
          sub_16CCCB0((_QWORD *)(v149 + 32), v149 + 72, v132 + 32);
          sub_16CCCB0((_QWORD *)(v149 + 88), v149 + 128, v132 + 88);
          *(_BYTE *)(v149 + 144) = *(_BYTE *)(v132 + 144);
          v149 = v94[14];
        }
        v94[14] = v149 + 152;
      }
      v150 = v94[13] + v148;
      v483 = 0;
      v487 = v491;
      v139 = v150 + 8;
      v488 = v491;
      v484 = 0;
      v485 = 0;
      v486 = 0;
      v489 = 2;
      v490 = 0;
      v493 = 0;
      v494 = v498;
      v495 = v498;
      v496 = 2;
      v497 = 0;
      v500 = 0;
      goto LABEL_151;
    }
LABEL_238:
    v179 = 0;
    v180 = *(_QWORD **)(v468.m128i_i64[0] + 48);
    v181 = (_QWORD *)(v468.m128i_i64[0] + 40);
    if ( (_QWORD *)(v468.m128i_i64[0] + 40) != v180 )
    {
      do
      {
        while ( 1 )
        {
          v182 = *v181 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v182 )
LABEL_637:
            BUG();
          if ( *(_BYTE *)(v182 - 8) == 29 )
            break;
          v179 |= sub_18D3860(a1, v182 - 24, v468.m128i_i64[0], a4, v396);
          v181 = (_QWORD *)(*v181 & 0xFFFFFFFFFFFFFFF8LL);
          if ( v181 == v180 )
            goto LABEL_244;
        }
        v181 = (_QWORD *)(*v181 & 0xFFFFFFFFFFFFFFF8LL);
      }
      while ( (_QWORD *)v182 != v180 );
    }
LABEL_244:
    v183 = v94[16];
    v184 = v183 + 8LL * *((unsigned int *)v94 + 34);
    if ( v183 != v184 )
    {
      v185 = v179;
      v186 = v94[16];
      v187 = v185;
      do
      {
        v188 = *(_QWORD *)(*(_QWORD *)v186 + 40LL) & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v188 )
          goto LABEL_637;
        if ( *(_BYTE *)(v188 - 8) == 29 )
          v187 |= sub_18D3860(a1, v188 - 24, v468.m128i_i64[0], a4, v396);
        v186 += 8;
      }
      while ( v184 != v186 );
      LOBYTE(v179) = v187;
    }
    v401 -= 8;
    v388 |= v179;
    if ( (_BYTE *)v392 != v401 )
      continue;
    break;
  }
LABEL_252:
  v384 = (unsigned __int64)v451;
  v391 = &v451[8 * (unsigned int)v452];
  if ( v451 != v391 )
  {
    v386 = 0;
    v383 = a1 + 160;
LABEL_254:
    v97 = (char *)a3;
    v462 = *((_QWORD *)v391 - 1);
    v189 = sub_18CDCF0(a3, &v462);
    v191 = (_QWORD *)v189[16];
    v409 = v189;
    v441 = v191;
    v192 = &v191[*((unsigned int *)v189 + 34)];
    if ( v191 == v192 )
      goto LABEL_285;
    v193 = *(unsigned int *)(a3 + 24);
    v194 = *(_QWORD *)(a3 + 8);
    if ( (_DWORD)v193 )
    {
      v195 = (char *)*v191;
      v190 = v193 - 1;
      v196 = (v193 - 1) & (((unsigned int)*v191 >> 9) ^ ((unsigned int)*v191 >> 4));
      v197 = 1;
      v198 = (char **)(v194 + 192LL * v196);
      v97 = *v198;
      if ( v195 == *v198 )
        goto LABEL_257;
      while ( v97 != (char *)-8LL )
      {
        v196 = v190 & (v197 + v196);
        v198 = (char **)(v194 + 192LL * v196);
        v97 = *v198;
        if ( v195 == *v198 )
          goto LABEL_257;
        ++v197;
      }
    }
    v198 = (char **)(v194 + 192 * v193);
LABEL_257:
    if ( v198 + 2 != (char **)(v409 + 2) )
    {
      v97 = (char *)v409[3];
      j___libc_free_0(v97);
      v199 = *((unsigned int *)v198 + 10);
      *((_DWORD *)v409 + 10) = v199;
      if ( (_DWORD)v199 )
      {
        v97 = (char *)sub_22077B0(16 * v199);
        v409[3] = (__int64)v97;
        v200 = *((unsigned int *)v409 + 10);
        *((_DWORD *)v409 + 8) = *((_DWORD *)v198 + 8);
        *((_DWORD *)v409 + 9) = *((_DWORD *)v198 + 9);
        memcpy(v97, v198[3], 16 * v200);
      }
      else
      {
        v409[3] = 0;
        v409[4] = 0;
      }
    }
    v107 = v409;
    if ( v198 + 6 != (char **)(v409 + 6) )
    {
      v201 = v198[7];
      v202 = v198[6];
      v110 = v409[6];
      v203 = v201 - v202;
      if ( v409[8] - v110 < v201 - v202 )
      {
        if ( v203 )
        {
          v421 = v198[6];
          v429 = v198[7];
          if ( (unsigned __int64)v203 > 0x7FFFFFFFFFFFFFC8LL )
            goto LABEL_613;
          v332 = sub_22077B0(v203);
          v201 = v429;
          v202 = v421;
          v333 = v332;
        }
        else
        {
          v333 = 0;
        }
        v334 = v202;
        v335 = v333;
        if ( v201 != v202 )
        {
          v430 = v203;
          v336 = v201;
          v422 = v333;
          v337 = v192;
          v338 = v198;
          v339 = v335;
          do
          {
            if ( v339 )
            {
              *(_QWORD *)v339 = *(_QWORD *)v334;
              *(_BYTE *)(v339 + 8) = v334[8];
              *(_BYTE *)(v339 + 9) = v334[9];
              *(_BYTE *)(v339 + 10) = v334[10];
              *(_BYTE *)(v339 + 16) = v334[16];
              *(_BYTE *)(v339 + 17) = v334[17];
              *(_QWORD *)(v339 + 24) = *((_QWORD *)v334 + 3);
              sub_16CCCB0((_QWORD *)(v339 + 32), v339 + 72, (__int64)(v334 + 32));
              sub_16CCCB0((_QWORD *)(v339 + 88), v339 + 128, (__int64)(v334 + 88));
              *(_BYTE *)(v339 + 144) = v334[144];
            }
            v334 += 152;
            v339 += 152;
          }
          while ( v336 != v334 );
          v198 = v338;
          v203 = v430;
          v192 = v337;
          v333 = v422;
        }
        v340 = (_QWORD *)v409[6];
        v431 = (_QWORD *)v409[7];
        if ( v431 != v340 )
        {
          do
          {
            v341 = v340[13];
            if ( v341 != v340[12] )
              _libc_free(v341);
            v342 = v340[6];
            if ( v342 != v340[5] )
              _libc_free(v342);
            v340 += 19;
          }
          while ( v431 != v340 );
          v340 = (_QWORD *)v409[6];
        }
        if ( v340 )
          j_j___libc_free_0(v340, v409[8] - (_QWORD)v340);
        v218 = v333 + v203;
        v409[6] = v333;
        v409[8] = v218;
      }
      else
      {
        v204 = v409[7];
        v205 = v204 - v110;
        v206 = v204 - v110;
        if ( v203 > v204 - v110 )
        {
          if ( v205 > 0 )
          {
            v427 = v198;
            v325 = (__int64)(v202 + 32);
            v326 = v110 + 32;
            v327 = 0x86BCA1AF286BCA1BLL * (v205 >> 3);
            do
            {
              *(_QWORD *)(v326 - 32) = *(_QWORD *)(v325 - 32);
              *(_BYTE *)(v326 - 24) = *(_BYTE *)(v325 - 24);
              *(_BYTE *)(v326 - 23) = *(_BYTE *)(v325 - 23);
              *(_BYTE *)(v326 - 22) = *(_BYTE *)(v325 - 22);
              *(_BYTE *)(v326 - 16) = *(_BYTE *)(v325 - 16);
              *(_BYTE *)(v326 - 15) = *(_BYTE *)(v325 - 15);
              *(_QWORD *)(v326 - 8) = *(_QWORD *)(v325 - 8);
              if ( v325 != v326 )
                sub_16CCD50(v326, v325, v110, (int)v202, (int)v201, v190);
              if ( v325 + 56 != v326 + 56 )
                sub_16CCD50(v326 + 56, v325 + 56, v110, (int)v202, (int)v201, v190);
              v328 = *(_BYTE *)(v325 + 112);
              v326 += 152;
              v325 += 152;
              *(_BYTE *)(v326 - 40) = v328;
              --v327;
            }
            while ( v327 );
            v198 = v427;
            v204 = v409[7];
            v110 = v409[6];
            v201 = v427[7];
            v202 = v427[6];
            v206 = v204 - v110;
          }
          v329 = &v202[v206];
          if ( v201 == v329 )
          {
            v218 = v110 + v203;
          }
          else
          {
            v428 = v203;
            v330 = v198;
            v331 = v201;
            do
            {
              if ( v204 )
              {
                *(_QWORD *)v204 = *(_QWORD *)v329;
                *(_BYTE *)(v204 + 8) = v329[8];
                *(_BYTE *)(v204 + 9) = v329[9];
                *(_BYTE *)(v204 + 10) = v329[10];
                *(_BYTE *)(v204 + 16) = v329[16];
                *(_BYTE *)(v204 + 17) = v329[17];
                *(_QWORD *)(v204 + 24) = *((_QWORD *)v329 + 3);
                sub_16CCCB0((_QWORD *)(v204 + 32), v204 + 72, (__int64)(v329 + 32));
                sub_16CCCB0((_QWORD *)(v204 + 88), v204 + 128, (__int64)(v329 + 88));
                *(_BYTE *)(v204 + 144) = v329[144];
              }
              v329 += 152;
              v204 += 152;
            }
            while ( v331 != v329 );
            v198 = v330;
            v218 = v409[6] + v428;
          }
        }
        else
        {
          if ( v203 > 0 )
          {
            v418 = v409[6];
            v207 = (__int64)(v202 + 32);
            v425 = v198;
            v208 = v110 + 32;
            v209 = 0x86BCA1AF286BCA1BLL * (v203 >> 3);
            v414 = v192;
            v210 = v203;
            v211 = v207;
            v212 = v209;
            do
            {
              *(_QWORD *)(v208 - 32) = *(_QWORD *)(v211 - 32);
              *(_BYTE *)(v208 - 24) = *(_BYTE *)(v211 - 24);
              *(_BYTE *)(v208 - 23) = *(_BYTE *)(v211 - 23);
              *(_BYTE *)(v208 - 22) = *(_BYTE *)(v211 - 22);
              *(_BYTE *)(v208 - 16) = *(_BYTE *)(v211 - 16);
              *(_BYTE *)(v208 - 15) = *(_BYTE *)(v211 - 15);
              v213 = *(_QWORD *)(v211 - 8);
              *(_QWORD *)(v208 - 8) = v213;
              if ( v211 != v208 )
                sub_16CCD50(v208, v211, v213, v207, v209, v190);
              if ( v208 + 56 != v211 + 56 )
                sub_16CCD50(v208 + 56, v211 + 56, v213, v207, v209, v190);
              v214 = *(_BYTE *)(v211 + 112);
              v208 += 152;
              v211 += 152;
              *(_BYTE *)(v208 - 40) = v214;
              --v212;
            }
            while ( v212 );
            v198 = v425;
            v192 = v414;
            v203 = v210;
            v110 = v210 + v418;
          }
          v215 = (_QWORD *)v110;
          if ( v110 != v204 )
          {
            do
            {
              v216 = v215[13];
              if ( v216 != v215[12] )
                _libc_free(v216);
              v217 = v215[6];
              if ( v217 != v215[5] )
                _libc_free(v217);
              v215 += 19;
            }
            while ( (_QWORD *)v204 != v215 );
          }
          v218 = v409[6] + v203;
        }
      }
      v409[7] = v218;
    }
    v219 = v441 + 1;
    *((_DWORD *)v409 + 2) = *((_DWORD *)v198 + 2);
    if ( v192 != v441 + 1 )
    {
      while ( 1 )
      {
        v223 = *(unsigned int *)(a3 + 24);
        v224 = *(_QWORD *)(a3 + 8);
        if ( !(_DWORD)v223 )
          goto LABEL_284;
        v220 = (v223 - 1) & (((unsigned int)*v219 >> 9) ^ ((unsigned int)*v219 >> 4));
        v221 = (__int64 *)(v224 + 192LL * v220);
        v222 = *v221;
        if ( *v219 != *v221 )
          break;
LABEL_282:
        ++v219;
        sub_18D2D20((__int64)(v409 + 1), (__int64)(v221 + 1));
        if ( v192 == v219 )
          goto LABEL_285;
      }
      v322 = 1;
      while ( v222 != -8 )
      {
        v323 = v322 + 1;
        v324 = ((_DWORD)v223 - 1) & (v220 + v322);
        v220 = v324;
        v221 = (__int64 *)(v224 + 192 * v324);
        v222 = *v221;
        if ( *v219 == *v221 )
          goto LABEL_282;
        v322 = v323;
      }
LABEL_284:
      v221 = (__int64 *)(v224 + 192 * v223);
      goto LABEL_282;
    }
LABEL_285:
    v389 = 0;
    v393 = v462;
    v405 = v462 + 40;
    v415 = *(_QWORD *)(v462 + 48);
    if ( v415 == v462 + 40 )
      goto LABEL_298;
    while ( 2 )
    {
      v225 = v415 - 24;
      if ( !v415 )
        v225 = 0;
      v226 = sub_14399D0(v225);
      switch ( v226 )
      {
        case 0u:
        case 1u:
          v261 = *(_QWORD *)(v225 - 24LL * (*(_DWORD *)(v225 + 20) & 0xFFFFFFF));
          while ( 2 )
          {
            v263 = sub_1649C60(v261);
            v262 = 23;
            v227 = v263;
            v264 = *(_BYTE *)(v263 + 16);
            if ( v264 <= 0x17u )
              goto LABEL_334;
            if ( v264 != 78 )
            {
              v262 = 2 * (v264 != 29) + 21;
LABEL_334:
              if ( (unsigned __int8)sub_1439C90(v262) )
                goto LABEL_335;
              break;
            }
            v262 = 21;
            if ( *(_BYTE *)(*(_QWORD *)(v227 - 24) + 16LL) )
              goto LABEL_334;
            v265 = sub_1438F00(*(_QWORD *)(v227 - 24));
            if ( (unsigned __int8)sub_1439C90(v265) )
            {
LABEL_335:
              v261 = *(_QWORD *)(v227 - 24LL * (*(_DWORD *)(v227 + 20) & 0xFFFFFFF));
              continue;
            }
            break;
          }
          v483 = (__int64 *)v227;
          v266 = sub_18D2800(v409 + 2, (__int64 *)&v483);
          v389 |= sub_18DBF60(v266, v226, v225);
          goto LABEL_290;
        case 4u:
          v267 = *(_QWORD *)(v225 - 24LL * (*(_DWORD *)(v225 + 20) & 0xFFFFFFF));
          while ( 2 )
          {
            v269 = sub_1649C60(v267);
            v268 = 23;
            v227 = v269;
            v270 = *(_BYTE *)(v269 + 16);
            if ( v270 <= 0x17u )
              goto LABEL_343;
            if ( v270 != 78 )
            {
              v268 = 2 * (v270 != 29) + 21;
LABEL_343:
              if ( (unsigned __int8)sub_1439C90(v268) )
                goto LABEL_344;
              break;
            }
            v268 = 21;
            if ( *(_BYTE *)(*(_QWORD *)(v227 - 24) + 16LL) )
              goto LABEL_343;
            v271 = sub_1438F00(*(_QWORD *)(v227 - 24));
            if ( (unsigned __int8)sub_1439C90(v271) )
            {
LABEL_344:
              v267 = *(_QWORD *)(v227 - 24LL * (*(_DWORD *)(v227 + 20) & 0xFFFFFFF));
              continue;
            }
            break;
          }
          v483 = (__int64 *)v227;
          v444 = sub_18D2800(v409 + 2, (__int64 *)&v483);
          v272 = sub_18DC040(v444, v387, v225);
          v273 = v444;
          if ( !v272 )
            goto LABEL_290;
          v274 = *(_DWORD *)(a5 + 24);
          if ( v274 )
          {
            v275 = *(_QWORD *)(a5 + 8);
            v276 = v274 - 1;
            v277 = (unsigned int)v225 >> 9;
            v445 = v277 ^ ((unsigned int)v225 >> 4);
            v278 = (v274 - 1) & v445;
            v279 = (__int64 *)(v275 + 144LL * v278);
            v280 = *v279;
            if ( v225 == *v279 )
              goto LABEL_352;
            v368 = 1;
            v369 = 0;
            while ( v280 != -8 )
            {
              if ( !v369 && v280 == -16 )
                v369 = v279;
              v382 = v368 + 1;
              v278 = v276 & (v368 + v278);
              v277 = v278;
              v279 = (__int64 *)(v275 + 144LL * v278);
              v280 = *v279;
              if ( v225 == *v279 )
                goto LABEL_352;
              v368 = v382;
            }
            if ( v369 )
              v279 = v369;
            ++*(_QWORD *)a5;
            v370 = *(_DWORD *)(a5 + 16) + 1;
            if ( 4 * v370 < 3 * v274 )
            {
              if ( v274 - *(_DWORD *)(a5 + 20) - v370 <= v274 >> 3 )
              {
                v438 = v273;
                sub_18D2120(a5, v274);
                v377 = *(_DWORD *)(a5 + 24);
                if ( !v377 )
                {
LABEL_636:
                  ++*(_DWORD *)(a5 + 16);
                  BUG();
                }
                v276 = v377 - 1;
                v378 = *(_QWORD *)(a5 + 8);
                v376 = 0;
                v273 = v438;
                v379 = (v377 - 1) & v445;
                v380 = 1;
                v279 = (__int64 *)(v378 + 144LL * v379);
                v370 = *(_DWORD *)(a5 + 16) + 1;
                v381 = *v279;
                if ( *v279 != v225 )
                {
                  while ( v381 != -8 )
                  {
                    if ( !v376 && v381 == -16 )
                      v376 = v279;
                    v379 = v276 & (v380 + v379);
                    v279 = (__int64 *)(v378 + 144LL * v379);
                    v381 = *v279;
                    if ( v225 == *v279 )
                      goto LABEL_573;
                    ++v380;
                  }
LABEL_592:
                  if ( v376 )
                    v279 = v376;
                  goto LABEL_573;
                }
              }
              goto LABEL_573;
            }
          }
          else
          {
            ++*(_QWORD *)a5;
          }
          v437 = v273;
          sub_18D2120(a5, 2 * v274);
          v371 = *(_DWORD *)(a5 + 24);
          if ( !v371 )
            goto LABEL_636;
          v276 = v371 - 1;
          v372 = *(_QWORD *)(a5 + 8);
          v273 = v437;
          v373 = (v371 - 1) & (((unsigned int)v225 >> 9) ^ ((unsigned int)v225 >> 4));
          v279 = (__int64 *)(v372 + 144LL * v373);
          v370 = *(_DWORD *)(a5 + 16) + 1;
          v374 = *v279;
          if ( v225 != *v279 )
          {
            v375 = 1;
            v376 = 0;
            while ( v374 != -8 )
            {
              if ( !v376 && v374 == -16 )
                v376 = v279;
              v373 = v276 & (v375 + v373);
              v279 = (__int64 *)(v372 + 144LL * v373);
              v374 = *v279;
              if ( v225 == *v279 )
                goto LABEL_573;
              ++v375;
            }
            goto LABEL_592;
          }
LABEL_573:
          *(_DWORD *)(a5 + 16) = v370;
          if ( *v279 != -8 )
            --*(_DWORD *)(a5 + 20);
          *v279 = v225;
          memset(v279 + 1, 0, 0x88u);
          LODWORD(v280) = 0;
          v277 = (__int64)(v279 + 8);
          v279[6] = 2;
          v279[4] = (__int64)(v279 + 8);
          v279[5] = (__int64)(v279 + 8);
          v279[11] = (__int64)(v279 + 15);
          v279[12] = (__int64)(v279 + 15);
          v279[13] = 2;
LABEL_352:
          *((_BYTE *)v279 + 8) = *(_BYTE *)(v273 + 8);
          *((_BYTE *)v279 + 9) = *(_BYTE *)(v273 + 9);
          v279[2] = *(_QWORD *)(v273 + 16);
          if ( (__int64 *)(v273 + 24) != v279 + 3 )
          {
            v446 = v273;
            sub_16CCD50((__int64)(v279 + 3), v273 + 24, v277, v280, v273, v276);
            v273 = v446;
          }
          if ( (__int64 *)(v273 + 80) != v279 + 10 )
          {
            v447 = v273;
            sub_16CCD50((__int64)(v279 + 10), v273 + 80, v277, v280, v273, v276);
            v273 = v447;
          }
          *((_BYTE *)v279 + 136) = *(_BYTE *)(v273 + 136);
          sub_18DB9D0(v273, 0);
LABEL_290:
          v442 = (_QWORD *)v409[7];
          if ( (_QWORD *)v409[6] != v442 )
          {
            v228 = (_QWORD *)v409[6];
            do
            {
              if ( *v228 != v227 )
              {
                v419 = *v228;
                if ( !(unsigned __int8)sub_18DC190(v228 + 1, v225, *v228, v383, v226) )
                  sub_18DC2A0(v228 + 1, v225, v419, v383, v226);
              }
              v228 += 19;
            }
            while ( v442 != v228 );
          }
LABEL_296:
          v415 = *(_QWORD *)(v415 + 8);
          if ( v405 != v415 )
            continue;
          v393 = v462;
LABEL_298:
          v229 = v409[6];
          v395 = v409[7];
          if ( v229 != v395 )
          {
LABEL_301:
            if ( !*(_BYTE *)(v229 + 10) )
              goto LABEL_300;
            v230 = *(_QWORD *)v229;
            v426 = *(_QWORD *)v229;
            v231 = *(_QWORD *)(v393 + 40) & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v231 )
              goto LABEL_300;
            v232 = v231 - 24;
            v420 = sub_15F4D60(v232);
            if ( !v420 )
              goto LABEL_300;
            v443 = v229;
            v233 = 0;
            v397 = v229 + 8;
            v416 = 0;
            v410 = 1;
            v406 = ((unsigned int)v230 >> 9) ^ ((unsigned int)v230 >> 4);
            v399 = 0;
            while ( 1 )
            {
              v250 = sub_15F4DF0(v232, v233);
              v251 = *(unsigned int *)(a3 + 24);
              v252 = *(_QWORD *)(a3 + 8);
              if ( !(_DWORD)v251 )
                break;
              v234 = 1;
              LODWORD(v235) = (v251 - 1) & (((unsigned int)v250 >> 9) ^ ((unsigned int)v250 >> 4));
              v236 = (__int64 *)(v252 + 192LL * (unsigned int)v235);
              v237 = *v236;
              if ( v250 != *v236 )
              {
                while ( v237 != -8 )
                {
                  v235 = ((_DWORD)v251 - 1) & (unsigned int)(v235 + v234);
                  v236 = (__int64 *)(v252 + 192 * v235);
                  v237 = *v236;
                  if ( v250 == *v236 )
                    goto LABEL_306;
                  ++v234;
                }
                v236 = (__int64 *)(v252 + 192 * v251);
LABEL_321:
                v238 = *((_DWORD *)v236 + 24);
                v239 = (__int64)(v236 + 9);
                if ( !v238 )
                {
LABEL_322:
                  ++v236[9];
                  goto LABEL_323;
                }
                goto LABEL_307;
              }
LABEL_306:
              v238 = *((_DWORD *)v236 + 24);
              v239 = (__int64)(v236 + 9);
              if ( !v238 )
                goto LABEL_322;
LABEL_307:
              v240 = v236[10];
              v241 = 0;
              v242 = 1;
              v243 = (v238 - 1) & v406;
              v244 = (__int64 *)(v240 + 16LL * v243);
              v245 = *v244;
              if ( *v244 == v426 )
                goto LABEL_308;
              while ( 1 )
              {
                if ( v245 == -8 )
                {
                  if ( !v241 )
                    v241 = v244;
                  v298 = *((_DWORD *)v236 + 22);
                  ++v236[9];
                  v257 = v298 + 1;
                  if ( 4 * v257 < 3 * v238 )
                  {
                    if ( v238 - *((_DWORD *)v236 + 23) - v257 > v238 >> 3 )
                      goto LABEL_403;
                    sub_18D2390(v239, v238);
                    v300 = *((_DWORD *)v236 + 24);
                    if ( v300 )
                    {
                      v301 = v300 - 1;
                      v302 = v236[10];
                      v260 = 0;
                      v303 = 1;
                      v257 = *((_DWORD *)v236 + 22) + 1;
                      LODWORD(v304) = v301 & v406;
                      v241 = (__int64 *)(v302 + 16LL * (v301 & v406));
                      v305 = *v241;
                      if ( *v241 == v426 )
                        goto LABEL_403;
                      while ( v305 != -8 )
                      {
                        if ( v305 == -16 && !v260 )
                          v260 = v241;
                        v304 = v301 & (unsigned int)(v304 + v303);
                        v241 = (__int64 *)(v302 + 16 * v304);
                        v305 = *v241;
                        if ( v426 == *v241 )
                          goto LABEL_403;
                        ++v303;
                      }
                      goto LABEL_421;
                    }
LABEL_638:
                    ++*((_DWORD *)v236 + 22);
                    BUG();
                  }
LABEL_323:
                  sub_18D2390(v239, 2 * v238);
                  v253 = *((_DWORD *)v236 + 24);
                  if ( v253 )
                  {
                    v254 = v253 - 1;
                    v255 = v236[10];
                    LODWORD(v256) = v254 & v406;
                    v257 = *((_DWORD *)v236 + 22) + 1;
                    v241 = (__int64 *)(v255 + 16LL * (v254 & v406));
                    v258 = *v241;
                    if ( *v241 == v426 )
                    {
LABEL_403:
                      *((_DWORD *)v236 + 22) = v257;
                      if ( *v241 != -8 )
                        --*((_DWORD *)v236 + 23);
                      v241[1] = 0;
                      *v241 = v426;
                      v299 = v236[14] - v236[13];
                      v241[1] = 0x86BCA1AF286BCA1BLL * (v299 >> 3);
                      v471 = v475;
                      v472 = v475;
                      v477 = v481;
                      v478 = v481;
                      v483 = (__int64 *)v426;
                      LOWORD(v484) = 0;
                      v468 = 0u;
                      v469 = 0;
                      v470 = 0;
                      v473 = 2;
                      v474 = 0;
                      v476 = 0;
                      v479 = 2;
                      v480 = 0;
                      v482 = 0;
                      BYTE2(v484) = 0;
                      v485 = 0;
                      v486 = 0;
                      sub_16CCEE0(&v487, (__int64)v492, 2, (__int64)&v470);
                      sub_16CCEE0(&v494, (__int64)v499, 2, (__int64)&v476);
                      v501 = v482;
                      sub_18D1D70((__int64)(v236 + 13), (__int64)&v483);
                      if ( (char *)v496 != v495 )
                        _libc_free(v496);
                      if ( (char *)v489 != v488 )
                        _libc_free(v489);
                      if ( v478 != v477 )
                        _libc_free((unsigned __int64)v478);
                      if ( v472 != v471 )
                        _libc_free((unsigned __int64)v472);
                      v246 = v236[13] + v299 + 8;
                      goto LABEL_309;
                    }
                    v259 = 1;
                    v260 = 0;
                    while ( v258 != -8 )
                    {
                      if ( !v260 && v258 == -16 )
                        v260 = v241;
                      v256 = v254 & (unsigned int)(v256 + v259);
                      v241 = (__int64 *)(v255 + 16 * v256);
                      v258 = *v241;
                      if ( v426 == *v241 )
                        goto LABEL_403;
                      ++v259;
                    }
LABEL_421:
                    if ( v260 )
                      v241 = v260;
                    goto LABEL_403;
                  }
                  goto LABEL_638;
                }
                if ( v245 != -16 || v241 )
                  v244 = v241;
                v243 = (v238 - 1) & (v242 + v243);
                v245 = *(_QWORD *)(v240 + 16LL * v243);
                if ( v426 == v245 )
                  break;
                ++v242;
                v241 = v244;
                v244 = (__int64 *)(v240 + 16LL * v243);
              }
              v244 = (__int64 *)(v240 + 16LL * v243);
LABEL_308:
              v246 = v236[13] + 152 * v244[1] + 8;
LABEL_309:
              v247 = *(_BYTE *)(v246 + 2);
              if ( !v247 )
                goto LABEL_395;
              v248 = *(_BYTE *)(v246 + 8);
              v249 = *(_BYTE *)(v229 + 10);
              if ( v249 == 2 )
              {
                if ( v247 != 2 )
                {
                  if ( (unsigned __int8)(v247 - 3) <= 3u )
                  {
                    v295 = v410;
                    v296 = *(_BYTE *)(v229 + 16) | v248;
                    v297 = v416;
                    if ( v296 )
                      v297 = v296;
                    v416 = v297;
                    if ( !v296 )
                      v295 = 0;
                    v410 = v295;
                  }
                  goto LABEL_318;
                }
LABEL_396:
                v399 = 1;
                goto LABEL_318;
              }
              if ( v249 != 3 )
                goto LABEL_318;
              if ( v247 == 3 )
                goto LABEL_396;
              if ( v247 > 3u )
              {
                if ( (unsigned __int8)(v247 - 4) > 2u )
                  goto LABEL_318;
                v290 = v410;
                v291 = *(_BYTE *)(v229 + 16) | v248;
                if ( !v291 )
                  v290 = 0;
                v292 = v416;
                if ( v291 )
                  v292 = v291;
                v410 = v290;
                ++v233;
                v416 = v292;
                if ( v420 == v233 )
                  goto LABEL_377;
              }
              else
              {
                if ( v247 != 2 )
                  goto LABEL_318;
                if ( *(_BYTE *)(v229 + 16) || v248 )
                {
                  *(_BYTE *)(v229 + 144) = 1;
                  goto LABEL_318;
                }
LABEL_395:
                sub_18DB9D0(v397, 0);
LABEL_318:
                if ( v420 == ++v233 )
                {
LABEL_377:
                  if ( v399 && !v410 )
                  {
                    sub_18DB9D0(v397, 0);
                  }
                  else if ( v416 )
                  {
                    *(_BYTE *)(v229 + 144) = 1;
                    v229 += 152;
                    if ( v395 == v443 + 152 )
                      goto LABEL_381;
                    goto LABEL_301;
                  }
LABEL_300:
                  v229 += 152;
                  if ( v395 == v229 )
                    goto LABEL_381;
                  goto LABEL_301;
                }
              }
            }
            v236 = (__int64 *)(v252 + 192 * v251);
            goto LABEL_321;
          }
LABEL_381:
          v391 -= 8;
          v386 |= v389;
          if ( (_BYTE *)v384 == v391 )
          {
            v293 = v386;
            LOBYTE(v293) = v388 & v386;
            goto LABEL_383;
          }
          break;
        case 7u:
        case 0x18u:
          goto LABEL_296;
        case 8u:
          v281 = *((_DWORD *)v409 + 8);
          ++v409[2];
          if ( !v281 )
          {
            if ( !*((_DWORD *)v409 + 9) )
              goto LABEL_363;
            v282 = *((unsigned int *)v409 + 10);
            if ( (unsigned int)v282 > 0x40 )
            {
              j___libc_free_0(v409[3]);
              v409[3] = 0;
              v409[4] = 0;
              *((_DWORD *)v409 + 10) = 0;
              goto LABEL_363;
            }
LABEL_360:
            v283 = (_QWORD *)v409[3];
            for ( j = &v283[2 * v282]; j != v283; v283 += 2 )
              *v283 = -8;
            v409[4] = 0;
            goto LABEL_363;
          }
          v306 = 4 * v281;
          v282 = *((unsigned int *)v409 + 10);
          if ( (unsigned int)(4 * v281) < 0x40 )
            v306 = 64;
          if ( v306 >= (unsigned int)v282 )
            goto LABEL_360;
          v307 = (_QWORD *)v409[3];
          v308 = v281 - 1;
          if ( !v308 )
          {
            v313 = 2048;
            v312 = 128;
LABEL_432:
            j___libc_free_0(v307);
            *((_DWORD *)v409 + 10) = v312;
            v314 = (_QWORD *)sub_22077B0(v313);
            v315 = *((unsigned int *)v409 + 10);
            v409[4] = 0;
            v409[3] = (__int64)v314;
            for ( k = &v314[2 * v315]; k != v314; v314 += 2 )
            {
              if ( v314 )
                *v314 = -8;
            }
            goto LABEL_363;
          }
          _BitScanReverse(&v308, v308);
          v309 = 1 << (33 - (v308 ^ 0x1F));
          if ( v309 < 64 )
            v309 = 64;
          if ( (_DWORD)v282 != v309 )
          {
            v310 = (4 * v309 / 3u + 1) | ((unsigned __int64)(4 * v309 / 3u + 1) >> 1);
            v311 = ((v310 | (v310 >> 2)) >> 4)
                 | v310
                 | (v310 >> 2)
                 | ((((v310 | (v310 >> 2)) >> 4) | v310 | (v310 >> 2)) >> 8);
            v312 = (v311 | (v311 >> 16)) + 1;
            v313 = 16 * ((v311 | (v311 >> 16)) + 1);
            goto LABEL_432;
          }
          v409[4] = 0;
          v360 = &v307[2 * (unsigned int)v282];
          do
          {
            if ( v307 )
              *v307 = -8;
            v307 += 2;
          }
          while ( v360 != v307 );
LABEL_363:
          v285 = (_QWORD *)v409[6];
          v286 = (_QWORD *)v409[7];
          if ( v285 != v286 )
          {
            v287 = (_QWORD *)v409[6];
            do
            {
              v288 = v287[13];
              if ( v288 != v287[12] )
                _libc_free(v288);
              v289 = v287[6];
              if ( v289 != v287[5] )
                _libc_free(v289);
              v287 += 19;
            }
            while ( v286 != v287 );
            v409[7] = (__int64)v285;
          }
          goto LABEL_296;
        default:
          v227 = 0;
          goto LABEL_290;
      }
      goto LABEL_254;
    }
  }
  v293 = 0;
LABEL_383:
  if ( v454 != v456 )
    _libc_free((unsigned __int64)v454);
  if ( v451 != v453 )
    _libc_free((unsigned __int64)v451);
  return v293;
}
