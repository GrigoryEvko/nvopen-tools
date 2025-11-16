// Function: sub_28342F0
// Address: 0x28342f0
//
__int64 __fastcall sub_28342F0(__int64 *a1, __int64 *a2, unsigned int a3, unsigned int a4, _QWORD *a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v9; // r11
  __int64 v10; // r10
  __int64 v11; // rcx
  __int64 *v12; // r14
  __int64 *i; // r12
  __m128i *v14; // rdi
  __int64 v15; // rdx
  __m128i v16; // xmm0
  __m128i v17; // xmm1
  __m128i v18; // xmm2
  __m128i v19; // xmm3
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r15
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 *v32; // r14
  __int64 v33; // r12
  __int64 v34; // rax
  __int64 v35; // r13
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __m128i v40; // xmm4
  __m128i v41; // xmm5
  __m128i v42; // xmm6
  unsigned __int64 *v43; // r15
  unsigned __int64 *v44; // r13
  unsigned __int64 v45; // rdi
  unsigned __int64 v46; // rbx
  unsigned __int64 *v47; // r13
  unsigned __int64 v48; // rdi
  __int64 v50; // r12
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // r14
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  __m128i v61; // xmm4
  __m128i v62; // xmm5
  __m128i v63; // xmm6
  unsigned __int64 *v64; // r15
  unsigned __int64 v65; // rbx
  unsigned __int64 *v66; // r13
  unsigned __int64 v67; // rdi
  __int64 v68; // rax
  __int64 v69; // rax
  unsigned __int64 *v70; // r14
  __int64 v71; // r8
  unsigned __int64 v72; // rdi
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // r12
  __int64 v76; // rax
  __int64 v77; // r12
  __int64 v78; // r12
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // r15
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  __m128i v89; // xmm7
  __m128i v90; // xmm4
  __m128i v91; // xmm5
  unsigned __int64 *v92; // r14
  unsigned __int64 *v93; // r13
  unsigned __int64 *v94; // rbx
  unsigned __int64 v95; // rdi
  __int64 v96; // rbx
  unsigned __int64 v97; // rax
  _QWORD *v98; // rdx
  unsigned __int64 v99; // rax
  __int64 v100; // r12
  __int64 *v101; // rax
  __int64 v102; // r14
  __int64 v103; // r12
  __int64 v104; // rbx
  __int64 v105; // r13
  __int64 v106; // r14
  unsigned __int64 v107; // rax
  int v108; // ecx
  int v109; // r8d
  unsigned int v110; // r15d
  __int64 v111; // rbx
  __int64 v112; // rax
  __int64 v113; // r12
  __int64 *v114; // r14
  __int64 v115; // r12
  __int64 v116; // rax
  __int64 v117; // rdx
  __int64 v118; // rcx
  __int64 v119; // r8
  __int64 v120; // r9
  __int64 v121; // rbx
  __int64 v122; // rdx
  __int64 v123; // rcx
  __int64 v124; // r8
  __int64 v125; // r9
  __m128i v126; // xmm0
  __m128i v127; // xmm1
  __m128i v128; // xmm2
  unsigned __int64 *v129; // rbx
  unsigned __int64 v130; // rdi
  __int64 v131; // rbx
  __int64 v132; // rax
  __int64 v133; // rdx
  __int64 v134; // rcx
  __int64 v135; // rdx
  __int64 v136; // r12
  __int64 v137; // rax
  __int64 v138; // rdx
  __int64 v139; // rcx
  __int64 v140; // r8
  __int64 v141; // r9
  __int64 v142; // r15
  __int64 v143; // rdx
  __int64 v144; // rcx
  __int64 v145; // r8
  __int64 v146; // r9
  __m128i v147; // xmm4
  __m128i v148; // xmm5
  __m128i v149; // xmm6
  unsigned __int64 *v150; // r14
  unsigned __int64 *v151; // rbx
  unsigned __int64 v152; // rdi
  unsigned __int64 *v153; // r15
  unsigned __int64 v154; // rdi
  __int64 v155; // rdx
  unsigned __int64 *v156; // r15
  unsigned __int64 v157; // rdi
  __int64 v158; // rax
  __int64 v159; // rax
  __int64 v160; // rax
  __int64 v161; // rax
  __int64 *v162; // r13
  __int64 v163; // r14
  __int64 v164; // rax
  __int64 v165; // rdx
  __int64 v166; // rcx
  __int64 v167; // r8
  __int64 v168; // r9
  __int64 v169; // r15
  __int64 v170; // rdx
  __int64 v171; // rcx
  __int64 v172; // r8
  __int64 v173; // r9
  __m128i v174; // xmm5
  __m128i v175; // xmm6
  __m128i v176; // xmm7
  unsigned __int64 *v177; // r15
  unsigned __int64 v178; // rdi
  unsigned __int64 v179; // r13
  unsigned __int64 *v180; // rbx
  unsigned __int64 v181; // rdi
  __int64 v182; // rax
  __int64 v183; // rax
  __int64 v184; // rax
  __int64 v185; // rax
  __int64 v186; // rbx
  __int64 v187; // rax
  __int64 v188; // r13
  __int64 v189; // rdx
  __int64 v190; // r15
  __int64 v191; // r12
  __int64 v192; // r14
  __int64 *v193; // r12
  __int64 v194; // r13
  __int64 v195; // rax
  __int64 v196; // rdx
  __int64 v197; // rcx
  __int64 v198; // r8
  __int64 v199; // r9
  __int64 v200; // rsi
  __int64 v201; // rbx
  __int64 v202; // rdx
  __int64 v203; // rcx
  __int64 v204; // r8
  __int64 v205; // r9
  char *v206; // rax
  char *v207; // rdx
  __int64 v208; // rax
  __int64 v209; // r12
  __int64 v210; // rax
  __int64 v211; // rax
  __int64 v212; // rdx
  __int64 v213; // r15
  __int64 v214; // r13
  int v215; // edx
  __int64 v216; // rbx
  __int64 v217; // rax
  __int64 v218; // r14
  __int64 v219; // rax
  __int64 v220; // r13
  __int64 v221; // rax
  __int64 v222; // rax
  __int64 v223; // r13
  __int64 v224; // rsi
  __int64 v225; // rax
  int v226; // edi
  unsigned int v227; // r9d
  __int64 *v228; // rdx
  __int64 v229; // rcx
  __int64 *v230; // rax
  unsigned int v231; // r8d
  __int64 *v232; // rcx
  __int64 v233; // r10
  __int64 v234; // r12
  __int64 v235; // rax
  __int64 v236; // rdx
  __int64 v237; // rcx
  __int64 v238; // r8
  __int64 v239; // r9
  __int64 v240; // rbx
  __int64 v241; // rdx
  __int64 v242; // rcx
  __int64 v243; // r8
  __int64 v244; // r9
  __m128i v245; // xmm7
  __m128i v246; // xmm7
  __m128i v247; // xmm7
  unsigned __int64 *v248; // r15
  unsigned __int64 v249; // rbx
  unsigned __int64 *v250; // r13
  unsigned __int64 v251; // rdi
  __int64 v252; // rax
  __int64 v253; // rdx
  __int64 v254; // rcx
  __int64 v255; // r12
  __int64 v256; // rax
  __int64 v257; // rcx
  __int64 v258; // rdx
  unsigned __int64 v259; // rax
  int v260; // ecx
  unsigned __int64 v261; // rax
  __int64 v262; // rbx
  __int64 v263; // r13
  char v264; // dh
  __int64 *v265; // rsi
  char v266; // al
  __int64 v267; // rdx
  __int64 v268; // rax
  __int64 v269; // r15
  __int64 v270; // r14
  __int64 v271; // rbx
  __int64 v272; // rdi
  _QWORD *v273; // rdi
  unsigned __int64 v274; // rax
  unsigned __int64 v275; // rsi
  __int64 v276; // r8
  __int64 v277; // r9
  __int64 *v278; // rdx
  __int64 *v279; // rax
  __int64 v280; // rcx
  _QWORD *v281; // rax
  unsigned __int64 v282; // rdx
  __int64 v283; // r8
  unsigned __int64 v284; // rdx
  __int64 v285; // r8
  __int64 v286; // rax
  char *v287; // rcx
  char *v288; // rax
  char v289; // si
  __int64 v290; // rax
  _QWORD *v291; // r10
  __int64 v292; // rsi
  _QWORD *v293; // r10
  _QWORD *v294; // rax
  __int64 v295; // r8
  int v296; // r9d
  __int64 v297; // rdx
  int v298; // eax
  __int64 v299; // r14
  unsigned int v300; // eax
  __int64 v301; // rbx
  __int64 v302; // r15
  __int64 k; // r14
  bool v304; // r12
  __int64 *v305; // rax
  __int64 v306; // rax
  _QWORD *v307; // rsi
  _QWORD *v308; // rax
  char v309; // dl
  char v310; // cl
  __int64 v311; // rax
  __int64 *v312; // rbx
  __int64 v313; // r14
  __int64 v314; // rax
  unsigned int v315; // edx
  __int64 v316; // r12
  unsigned __int64 v317; // rcx
  __int64 v318; // r8
  __int64 v319; // r13
  _BYTE *v320; // rax
  __int64 v321; // r9
  __int64 v322; // rax
  unsigned __int64 v323; // rdx
  __int64 v324; // rsi
  unsigned __int64 *v325; // r14
  __int64 v326; // rax
  unsigned __int64 v327; // rdi
  __int64 v328; // r12
  __int64 v329; // rax
  __int64 v330; // r8
  __int64 v331; // r9
  __int64 v332; // rbx
  __int64 v333; // r13
  unsigned __int64 v334; // rdx
  __int64 v335; // r14
  signed __int64 v336; // r15
  __int64 *v337; // rcx
  __int64 v338; // rax
  char *v339; // rbx
  char *v340; // r15
  _QWORD *v341; // rdi
  unsigned __int64 v342; // rsi
  unsigned __int64 v343; // rax
  int v344; // edx
  unsigned __int64 v345; // rax
  __int64 *v346; // r12
  __int64 v347; // rbx
  __int64 *v348; // r15
  __int64 v349; // r13
  _QWORD *v350; // rdi
  unsigned __int64 v351; // rax
  unsigned __int64 v352; // rsi
  __int64 v353; // r13
  __int64 v354; // r14
  _QWORD *v355; // rdx
  unsigned __int64 v356; // rax
  int v357; // edx
  unsigned __int64 v358; // r12
  unsigned __int64 v359; // rax
  __int64 v360; // rax
  __int64 v361; // rax
  _QWORD *v362; // rdx
  __int64 v363; // rcx
  unsigned __int64 v364; // rax
  __int64 v365; // rdx
  _BYTE *v366; // rax
  __int64 v367; // rcx
  __int64 v368; // r8
  __int64 v369; // r9
  __int64 v370; // rdx
  __int64 *v371; // rbx
  __int64 *j; // r15
  __int64 v373; // rax
  __int64 v374; // rax
  __int64 v375; // rax
  int v376; // r8d
  __int64 v377; // rsi
  _QWORD *v378; // rax
  _QWORD *v379; // rdx
  int v380; // ecx
  int v381; // r9d
  __int64 v382; // rax
  __int64 v383; // rax
  __int64 v384; // rax
  __int64 v385; // rax
  __int64 *v386; // r12
  __int64 v387; // r13
  __int64 v388; // rax
  __int64 v389; // rdx
  __int64 v390; // rcx
  __int64 v391; // r8
  __int64 v392; // r9
  __int64 v393; // r14
  __int64 v394; // rdx
  __int64 v395; // rcx
  __int64 v396; // r8
  __int64 v397; // r9
  __int64 v398; // rax
  __int64 v399; // rax
  __int64 *v400; // r13
  __int64 v401; // r12
  __int64 v402; // rax
  __int64 v403; // rdx
  __int64 v404; // rcx
  __int64 v405; // r8
  __int64 v406; // r9
  __int64 v407; // r15
  __int64 v408; // rdx
  __int64 v409; // rcx
  __int64 v410; // r8
  __int64 v411; // r9
  __m128i v412; // xmm6
  __m128i v413; // xmm7
  __m128i v414; // xmm4
  unsigned __int64 *v415; // r15
  unsigned __int64 v416; // rdi
  unsigned __int64 v417; // r13
  unsigned __int64 *v418; // rbx
  unsigned __int64 v419; // rdi
  __int64 v420; // rax
  __int64 v421; // rax
  __int64 v422; // [rsp-10h] [rbp-620h]
  __int64 v423; // [rsp-8h] [rbp-618h]
  __int64 v424; // [rsp+0h] [rbp-610h]
  __int64 v425; // [rsp+8h] [rbp-608h]
  int v426; // [rsp+1Ch] [rbp-5F4h]
  __int64 *v427; // [rsp+20h] [rbp-5F0h]
  __int64 v429; // [rsp+30h] [rbp-5E0h]
  __int64 v430; // [rsp+38h] [rbp-5D8h]
  __int64 v432; // [rsp+48h] [rbp-5C8h]
  __int64 v433; // [rsp+50h] [rbp-5C0h]
  __int64 v434; // [rsp+58h] [rbp-5B8h]
  __int64 v436; // [rsp+78h] [rbp-598h]
  int v437; // [rsp+80h] [rbp-590h]
  unsigned __int8 v438; // [rsp+86h] [rbp-58Ah]
  bool v439; // [rsp+87h] [rbp-589h]
  __int64 v440; // [rsp+88h] [rbp-588h]
  int v441; // [rsp+88h] [rbp-588h]
  __int64 v442; // [rsp+88h] [rbp-588h]
  bool v444; // [rsp+90h] [rbp-580h]
  __m128i v445; // [rsp+98h] [rbp-578h]
  unsigned __int64 *v446; // [rsp+A0h] [rbp-570h]
  __int64 v447; // [rsp+A0h] [rbp-570h]
  unsigned __int64 *v448; // [rsp+A0h] [rbp-570h]
  __int64 *v449; // [rsp+A8h] [rbp-568h]
  __int64 *v450; // [rsp+A8h] [rbp-568h]
  __int64 v451; // [rsp+A8h] [rbp-568h]
  __int64 *v452; // [rsp+A8h] [rbp-568h]
  unsigned __int64 v453; // [rsp+A8h] [rbp-568h]
  __int64 v454; // [rsp+A8h] [rbp-568h]
  __int64 v455; // [rsp+A8h] [rbp-568h]
  int v456; // [rsp+BCh] [rbp-554h] BYREF
  __m128i v457; // [rsp+C0h] [rbp-550h] BYREF
  __m128i v458; // [rsp+D0h] [rbp-540h] BYREF
  __m128i v459; // [rsp+E0h] [rbp-530h]
  _BYTE v460[16]; // [rsp+F0h] [rbp-520h] BYREF
  void (__fastcall *v461)(_BYTE *, _BYTE *, __int64); // [rsp+100h] [rbp-510h]
  unsigned __int8 (__fastcall *v462)(_BYTE *, __int64); // [rsp+108h] [rbp-508h]
  __m128i v463; // [rsp+110h] [rbp-500h] BYREF
  unsigned __int128 v464; // [rsp+120h] [rbp-4F0h]
  _QWORD v465[2]; // [rsp+130h] [rbp-4E0h] BYREF
  void (__fastcall *v466)(_QWORD *, _QWORD *, __int64); // [rsp+140h] [rbp-4D0h]
  __int64 v467; // [rsp+148h] [rbp-4C8h]
  __m128i v468; // [rsp+150h] [rbp-4C0h] BYREF
  __m128i v469; // [rsp+160h] [rbp-4B0h] BYREF
  __int64 v470; // [rsp+170h] [rbp-4A0h] BYREF
  __int64 *v471; // [rsp+178h] [rbp-498h]
  void (__fastcall *v472)(_BYTE *, __int64 *, __int64); // [rsp+180h] [rbp-490h]
  unsigned __int8 (__fastcall *v473)(_BYTE *, __int64); // [rsp+188h] [rbp-488h]
  __m128i v474; // [rsp+190h] [rbp-480h] BYREF
  __m128i v475; // [rsp+1A0h] [rbp-470h] BYREF
  _BYTE v476[16]; // [rsp+1B0h] [rbp-460h] BYREF
  void (__fastcall *v477)(_QWORD *, _BYTE *, __int64); // [rsp+1C0h] [rbp-450h]
  __int64 v478; // [rsp+1C8h] [rbp-448h]
  __int64 v479; // [rsp+1D0h] [rbp-440h] BYREF
  __int64 v480; // [rsp+1D8h] [rbp-438h]
  __int64 v481; // [rsp+1E0h] [rbp-430h]
  __int64 *v482; // [rsp+1E8h] [rbp-428h]
  __int64 v483; // [rsp+1F0h] [rbp-420h] BYREF
  char *v484; // [rsp+1F8h] [rbp-418h]
  __int64 v485; // [rsp+200h] [rbp-410h]
  int v486; // [rsp+208h] [rbp-408h]
  char v487; // [rsp+20Ch] [rbp-404h]
  char v488; // [rsp+210h] [rbp-400h] BYREF
  _BYTE *v489; // [rsp+230h] [rbp-3E0h] BYREF
  __int64 v490; // [rsp+238h] [rbp-3D8h]
  _BYTE v491[64]; // [rsp+240h] [rbp-3D0h] BYREF
  __int64 *v492; // [rsp+280h] [rbp-390h] BYREF
  __int64 v493; // [rsp+288h] [rbp-388h]
  __int64 v494; // [rsp+290h] [rbp-380h] BYREF
  __m128i v495; // [rsp+298h] [rbp-378h]
  __int64 v496; // [rsp+2A8h] [rbp-368h]
  __m128i v497; // [rsp+2B0h] [rbp-360h]
  __m128i v498; // [rsp+2C0h] [rbp-350h]
  unsigned __int64 *v499; // [rsp+2D0h] [rbp-340h] BYREF
  __int64 v500; // [rsp+2D8h] [rbp-338h]
  _BYTE v501[320]; // [rsp+2E0h] [rbp-330h] BYREF
  char v502; // [rsp+420h] [rbp-1F0h]
  int v503; // [rsp+424h] [rbp-1ECh]
  __int64 v504; // [rsp+428h] [rbp-1E8h]
  __int64 v505; // [rsp+430h] [rbp-1E0h] BYREF
  __int64 v506; // [rsp+438h] [rbp-1D8h]
  __int64 v507; // [rsp+440h] [rbp-1D0h] BYREF
  __m128i v508; // [rsp+448h] [rbp-1C8h] BYREF
  __int64 v509; // [rsp+458h] [rbp-1B8h]
  __m128i v510; // [rsp+460h] [rbp-1B0h] BYREF
  __m128i v511; // [rsp+470h] [rbp-1A0h] BYREF
  unsigned __int64 *v512; // [rsp+480h] [rbp-190h] BYREF
  unsigned int v513; // [rsp+488h] [rbp-188h]
  _BYTE v514[320]; // [rsp+490h] [rbp-180h] BYREF
  char v515; // [rsp+5D0h] [rbp-40h]
  int v516; // [rsp+5D4h] [rbp-3Ch]
  __int64 v517; // [rsp+5D8h] [rbp-38h]

  v6 = *a2;
  v433 = a4;
  v429 = 8LL * a4;
  v9 = *(_QWORD *)(v6 + v429);
  v10 = *(_QWORD *)(v6 + 8LL * a3);
  v432 = a3;
  v11 = *a1;
  v482 = (__int64 *)a1[5];
  v484 = &v488;
  v445.m128i_i64[0] = v9;
  v430 = 8LL * a3;
  v445.m128i_i64[1] = v10;
  v479 = v9;
  v480 = v10;
  v481 = v11;
  v483 = 0;
  v485 = 4;
  v486 = 0;
  v487 = 1;
  v489 = v491;
  v490 = 0x800000000LL;
  v438 = sub_282FC10((__int64)a5, a3, a4);
  if ( !v438 )
  {
    v50 = *v482;
    v449 = v482;
    v51 = sub_B2BE50(*v482);
    if ( sub_B6EA50(v51)
      || (v73 = sub_B2BE50(v50),
          v74 = sub_B6F970(v73),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v74 + 48LL))(v74)) )
    {
      v56 = **(_QWORD **)(v480 + 32);
      sub_D4BD20(&v463, v480, v52, v53, v54, v55);
      sub_B157E0((__int64)&v468, &v463);
      sub_B17640((__int64)&v505, (__int64)"loop-interchange", (__int64)"Dependence", 10, &v468, v56);
      sub_B18290((__int64)&v505, "Cannot interchange loops due to dependences.", 0x2Cu);
      v61 = _mm_loadu_si128(&v508);
      v62 = _mm_loadu_si128(&v510);
      v499 = (unsigned __int64 *)v501;
      LODWORD(v493) = v506;
      v63 = _mm_loadu_si128(&v511);
      v492 = (__int64 *)&unk_49D9D40;
      BYTE4(v493) = BYTE4(v506);
      v495 = v61;
      v494 = v507;
      v497 = v62;
      v496 = v509;
      v500 = 0x400000000LL;
      v498 = v63;
      if ( v513 )
      {
        sub_2830C60((__int64)&v499, (__int64)&v512, v57, v58, v59, v60);
        v505 = (__int64)&unk_49D9D40;
        v70 = v512;
        v502 = v515;
        v503 = v516;
        v504 = v517;
        v492 = (__int64 *)&unk_49D9DB0;
        v71 = 10LL * v513;
        v64 = &v512[v71];
        if ( v512 != &v512[v71] )
        {
          do
          {
            v64 -= 10;
            v72 = v64[4];
            if ( (unsigned __int64 *)v72 != v64 + 6 )
              j_j___libc_free_0(v72);
            if ( (unsigned __int64 *)*v64 != v64 + 2 )
              j_j___libc_free_0(*v64);
          }
          while ( v70 != v64 );
          v64 = v512;
        }
      }
      else
      {
        v64 = v512;
        v502 = v515;
        v503 = v516;
        v504 = v517;
        v492 = (__int64 *)&unk_49D9DB0;
      }
      if ( v64 != (unsigned __int64 *)v514 )
        _libc_free((unsigned __int64)v64);
      if ( v463.m128i_i64[0] )
        sub_B91220((__int64)&v463, v463.m128i_i64[0]);
      sub_1049740(v449, (__int64)&v492);
      v65 = (unsigned __int64)v499;
      v492 = (__int64 *)&unk_49D9D40;
      v66 = &v499[10 * (unsigned int)v500];
      if ( v499 != v66 )
      {
        do
        {
          v66 -= 10;
          v67 = v66[4];
          if ( (unsigned __int64 *)v67 != v66 + 6 )
            j_j___libc_free_0(v67);
          if ( (unsigned __int64 *)*v66 != v66 + 2 )
            j_j___libc_free_0(*v66);
        }
        while ( (unsigned __int64 *)v65 != v66 );
        v66 = v499;
      }
      if ( v66 != (unsigned __int64 *)v501 )
        _libc_free((unsigned __int64)v66);
    }
    goto LABEL_74;
  }
  v12 = *(__int64 **)(v479 + 40);
  for ( i = *(__int64 **)(v479 + 32); v12 != i; ++i )
  {
    v14 = &v468;
    sub_AA72C0(&v468, *i, 1);
    v16 = _mm_loadu_si128(&v468);
    v17 = _mm_loadu_si128(&v469);
    v461 = 0;
    v458 = v16;
    v459 = v17;
    if ( v472 )
    {
      v14 = (__m128i *)v460;
      v472(v460, &v470, 2);
      v462 = v473;
      v461 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))v472;
    }
    v18 = _mm_loadu_si128(&v474);
    v19 = _mm_loadu_si128(&v475);
    v466 = 0;
    v463 = v18;
    v464 = (unsigned __int128)v19;
    if ( v477 )
    {
      v14 = (__m128i *)v465;
      v477(v465, v476, 2);
      v467 = v478;
      v466 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v477;
    }
    while ( 1 )
    {
      v20 = v458.m128i_i64[0];
      v21 = v458.m128i_i64[0];
      if ( v458.m128i_i64[0] == v463.m128i_i64[0] )
        break;
      while ( 1 )
      {
        if ( !v21 )
          goto LABEL_47;
        if ( *(_BYTE *)(v21 - 24) == 85 )
        {
          v14 = (__m128i *)(v21 - 24);
          if ( !sub_B49E50(v21 - 24) )
          {
            v32 = v482;
            v438 = 0;
            v33 = *v482;
            v34 = sub_B2BE50(*v482);
            if ( sub_B6EA50(v34)
              || (v68 = sub_B2BE50(v33),
                  v69 = sub_B6F970(v68),
                  (*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v69 + 48LL))(v69, v21)) )
            {
              v35 = *(_QWORD *)(v21 + 16);
              sub_B157E0((__int64)&v457, (_QWORD *)(v21 + 24));
              sub_B17640((__int64)&v505, (__int64)"loop-interchange", (__int64)"CallInst", 8, &v457, v35);
              sub_B18290((__int64)&v505, "Cannot interchange loops due to call instruction.", 0x31u);
              v40 = _mm_loadu_si128(&v508);
              v41 = _mm_loadu_si128(&v510);
              v499 = (unsigned __int64 *)v501;
              LODWORD(v493) = v506;
              v42 = _mm_loadu_si128(&v511);
              v492 = (__int64 *)&unk_49D9D40;
              BYTE4(v493) = BYTE4(v506);
              v495 = v40;
              v494 = v507;
              v497 = v41;
              v496 = v509;
              v500 = 0x400000000LL;
              v498 = v42;
              if ( v513 )
              {
                sub_2830C60((__int64)&v499, (__int64)&v512, v36, v37, v38, v39);
                v505 = (__int64)&unk_49D9D40;
                v44 = v512;
                v502 = v515;
                v503 = v516;
                v504 = v517;
                v492 = (__int64 *)&unk_49D9DB0;
                v43 = &v512[10 * v513];
                if ( v512 != v43 )
                {
                  do
                  {
                    v43 -= 10;
                    v45 = v43[4];
                    if ( (unsigned __int64 *)v45 != v43 + 6 )
                      j_j___libc_free_0(v45);
                    if ( (unsigned __int64 *)*v43 != v43 + 2 )
                      j_j___libc_free_0(*v43);
                  }
                  while ( v44 != v43 );
                  v43 = v512;
                }
              }
              else
              {
                v43 = v512;
                v502 = v515;
                v503 = v516;
                v504 = v517;
                v492 = (__int64 *)&unk_49D9DB0;
              }
              if ( v43 != (unsigned __int64 *)v514 )
                _libc_free((unsigned __int64)v43);
              sub_1049740(v32, (__int64)&v492);
              v46 = (unsigned __int64)v499;
              v492 = (__int64 *)&unk_49D9D40;
              v47 = &v499[10 * (unsigned int)v500];
              if ( v499 != v47 )
              {
                do
                {
                  v47 -= 10;
                  v48 = v47[4];
                  if ( (unsigned __int64 *)v48 != v47 + 6 )
                    j_j___libc_free_0(v48);
                  if ( (unsigned __int64 *)*v47 != v47 + 2 )
                    j_j___libc_free_0(*v47);
                }
                while ( (unsigned __int64 *)v46 != v47 );
                v47 = v499;
              }
              if ( v47 != (unsigned __int64 *)v501 )
                _libc_free((unsigned __int64)v47);
            }
            if ( v466 )
              v466(v465, v465, 3);
            if ( v461 )
              v461(v460, v460, 3);
            if ( v477 )
              v477(v476, v476, 3);
            if ( v472 )
              v472(&v470, &v470, 3);
            goto LABEL_74;
          }
          v20 = v458.m128i_i64[0];
        }
        v20 = *(_QWORD *)(v20 + 8);
        v458.m128i_i16[4] = 0;
        v458.m128i_i64[0] = v20;
        v21 = v20;
        if ( v20 != v459.m128i_i64[0] )
          break;
LABEL_18:
        if ( v21 == v463.m128i_i64[0] )
          goto LABEL_19;
      }
      while ( 1 )
      {
        if ( v21 )
          v21 -= 24;
        if ( !v461 )
          sub_4263D6(v14, v21, v15);
        v14 = (__m128i *)v460;
        if ( v462(v460, v21) )
          break;
        v15 = 0;
        v21 = *(_QWORD *)(v458.m128i_i64[0] + 8);
        v458.m128i_i16[4] = 0;
        v458.m128i_i64[0] = v21;
        v20 = v21;
        if ( v459.m128i_i64[0] == v21 )
          goto LABEL_18;
      }
    }
LABEL_19:
    if ( v466 )
      v466(v465, v465, 3);
    if ( v461 )
      v461(v460, v460, 3);
    if ( v477 )
      v477(v476, v476, 3);
    if ( v472 )
      v472(&v470, &v470, 3);
  }
  v22 = v480;
  v23 = sub_AA5930(**(_QWORD **)(v480 + 32));
  v25 = v24;
  v26 = v23;
  if ( v23 != v24 )
  {
    while ( 1 )
    {
      v505 = 6;
      v506 = 0;
      v507 = 0;
      v508.m128i_i32[0] = 0;
      v508.m128i_i64[1] = 0;
      v509 = 0;
      v510.m128i_i64[0] = (__int64)&v511;
      v510.m128i_i64[1] = 0x200000000LL;
      if ( (unsigned __int8)sub_10238A0(v26, v22, v481, (__int64)&v505, 0, 0) )
      {
        v30 = (unsigned int)v490;
        v31 = (unsigned int)v490 + 1LL;
        if ( v31 > HIDWORD(v490) )
        {
          sub_C8D5F0((__int64)&v489, v491, v31, 8u, v28, v29);
          v30 = (unsigned int)v490;
        }
        *(_QWORD *)&v489[8 * v30] = v26;
        LODWORD(v490) = v490 + 1;
      }
      if ( (__m128i *)v510.m128i_i64[0] != &v511 )
        _libc_free(v510.m128i_u64[0]);
      if ( v507 != 0 && v507 != -4096 && v507 != -8192 )
        sub_BD60C0(&v505);
      if ( !v26 )
        BUG();
      v27 = *(_QWORD *)(v26 + 32);
      if ( !v27 )
        break;
      v26 = 0;
      if ( *(_BYTE *)(v27 - 24) == 84 )
        v26 = v27 - 24;
      if ( v25 == v26 )
        goto LABEL_97;
    }
LABEL_485:
    BUG();
  }
LABEL_97:
  if ( !(_DWORD)v490 )
    goto LABEL_98;
  v75 = v480;
  if ( *(_QWORD *)(v480 + 16) != *(_QWORD *)(v480 + 8) )
  {
    v76 = sub_D47930(v479);
    if ( !sub_AA5510(v76) )
    {
      v131 = sub_D47930(v75);
      v132 = sub_AA5930(v131);
      v134 = v133;
      if ( v132 != v133 )
      {
        while ( 1 )
        {
          v135 = *(_QWORD *)(v132 + 16);
          if ( v135 )
            break;
LABEL_201:
          v155 = *(_QWORD *)(v132 + 32);
          if ( !v155 )
            goto LABEL_485;
          v132 = 0;
          if ( *(_BYTE *)(v155 - 24) == 84 )
            v132 = v155 - 24;
          if ( v134 == v132 )
            goto LABEL_110;
        }
        while ( v131 != *(_QWORD *)(*(_QWORD *)(v135 + 24) + 40LL) )
        {
          v135 = *(_QWORD *)(v135 + 8);
          if ( !v135 )
            goto LABEL_201;
        }
        v136 = *v482;
        v452 = v482;
        v137 = sub_B2BE50(*v482);
        if ( !sub_B6EA50(v137) )
        {
          v160 = sub_B2BE50(v136);
          v161 = sub_B6F970(v160);
          if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v161 + 48LL))(v161) )
            goto LABEL_98;
        }
        v142 = **(_QWORD **)(v480 + 32);
        sub_D4BD20(&v463, v480, v138, v139, v140, v141);
        sub_B157E0((__int64)&v468, &v463);
        sub_B17640((__int64)&v505, (__int64)"loop-interchange", (__int64)"UnsupportedInnerLatchPHI", 24, &v468, v142);
        sub_B18290(
          (__int64)&v505,
          "Cannot interchange loops because unsupported PHI nodes found in inner loop latch.",
          0x51u);
        v147 = _mm_loadu_si128(&v508);
        v148 = _mm_loadu_si128(&v510);
        v499 = (unsigned __int64 *)v501;
        LODWORD(v493) = v506;
        v149 = _mm_loadu_si128(&v511);
        v492 = (__int64 *)&unk_49D9D40;
        BYTE4(v493) = BYTE4(v506);
        v495 = v147;
        v494 = v507;
        v497 = v148;
        v496 = v509;
        v500 = 0x400000000LL;
        v498 = v149;
        if ( v513 )
        {
          sub_2830C60((__int64)&v499, (__int64)&v512, v143, v144, v145, v146);
          v505 = (__int64)&unk_49D9D40;
          v150 = v512;
          v502 = v515;
          v503 = v516;
          v504 = v517;
          v492 = (__int64 *)&unk_49D9DB0;
          v153 = &v512[10 * v513];
          if ( v512 == v153 )
            goto LABEL_183;
          do
          {
            v153 -= 10;
            v154 = v153[4];
            if ( (unsigned __int64 *)v154 != v153 + 6 )
              j_j___libc_free_0(v154);
            if ( (unsigned __int64 *)*v153 != v153 + 2 )
              j_j___libc_free_0(*v153);
          }
          while ( v150 != v153 );
        }
        else
        {
          v502 = v515;
          v503 = v516;
          v504 = v517;
          v492 = (__int64 *)&unk_49D9DB0;
        }
        v150 = v512;
LABEL_183:
        if ( v150 != (unsigned __int64 *)v514 )
          _libc_free((unsigned __int64)v150);
        if ( v463.m128i_i64[0] )
          sub_B91220((__int64)&v463, v463.m128i_i64[0]);
        sub_1049740(v452, (__int64)&v492);
        v93 = v499;
        v492 = (__int64 *)&unk_49D9D40;
        v151 = &v499[10 * (unsigned int)v500];
        if ( v499 == v151 )
          goto LABEL_127;
        do
        {
          v151 -= 10;
          v152 = v151[4];
          if ( (unsigned __int64 *)v152 != v151 + 6 )
            j_j___libc_free_0(v152);
          if ( (unsigned __int64 *)*v151 != v151 + 2 )
            j_j___libc_free_0(*v151);
        }
        while ( v93 != v151 );
        goto LABEL_126;
      }
    }
LABEL_110:
    v75 = v480;
  }
  v77 = sub_D47930(v75);
  if ( v77 != sub_D46F00(v480) )
    goto LABEL_112;
  v96 = sub_D46F00(v479);
  if ( v96 != sub_D47930(v479) )
    goto LABEL_112;
  v97 = *(_QWORD *)(v77 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v97 == v77 + 48 || !v97 || (unsigned int)*(unsigned __int8 *)(v97 - 24) - 30 > 0xA )
    goto LABEL_485;
  if ( *(_BYTE *)(v97 - 24) != 31 )
    goto LABEL_112;
  v98 = (_QWORD *)(sub_D47930(v479) + 48);
  v99 = *v98 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v99 == v98 || !v99 || (unsigned int)*(unsigned __int8 *)(v99 - 24) - 30 > 0xA )
    goto LABEL_485;
  if ( *(_BYTE *)(v99 - 24) != 31 )
  {
LABEL_112:
    v78 = *v482;
    v450 = v482;
    v79 = sub_B2BE50(*v482);
    if ( !sub_B6EA50(v79) )
    {
      v158 = sub_B2BE50(v78);
      v159 = sub_B6F970(v158);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v159 + 48LL))(v159) )
        goto LABEL_98;
    }
    v84 = **(_QWORD **)(v479 + 32);
    sub_D4BD20(&v463, v479, v80, v81, v82, v83);
    sub_B157E0((__int64)&v468, &v463);
    sub_B17640((__int64)&v505, (__int64)"loop-interchange", (__int64)"ExitingNotLatch", 15, &v468, v84);
    sub_B18290((__int64)&v505, "Loops where the latch is not the exiting block cannot be interchange currently.", 0x4Fu);
    v89 = _mm_loadu_si128(&v508);
    v90 = _mm_loadu_si128(&v510);
    v499 = (unsigned __int64 *)v501;
    LODWORD(v493) = v506;
    v91 = _mm_loadu_si128(&v511);
    v492 = (__int64 *)&unk_49D9D40;
    BYTE4(v493) = BYTE4(v506);
    v495 = v89;
    v494 = v507;
    v497 = v90;
    v496 = v509;
    v500 = 0x400000000LL;
    v498 = v91;
    if ( v513 )
    {
      sub_2830C60((__int64)&v499, (__int64)&v512, v85, v86, v87, v88);
      v505 = (__int64)&unk_49D9D40;
      v92 = v512;
      v502 = v515;
      v503 = v516;
      v504 = v517;
      v492 = (__int64 *)&unk_49D9DB0;
      v156 = &v512[10 * v513];
      if ( v512 == v156 )
        goto LABEL_116;
      do
      {
        v156 -= 10;
        v157 = v156[4];
        if ( (unsigned __int64 *)v157 != v156 + 6 )
          j_j___libc_free_0(v157);
        if ( (unsigned __int64 *)*v156 != v156 + 2 )
          j_j___libc_free_0(*v156);
      }
      while ( v92 != v156 );
    }
    else
    {
      v502 = v515;
      v503 = v516;
      v504 = v517;
      v492 = (__int64 *)&unk_49D9DB0;
    }
    v92 = v512;
LABEL_116:
    if ( v92 != (unsigned __int64 *)v514 )
      _libc_free((unsigned __int64)v92);
    if ( v463.m128i_i64[0] )
      sub_B91220((__int64)&v463, v463.m128i_i64[0]);
    sub_1049740(v450, (__int64)&v492);
    v93 = v499;
    v492 = (__int64 *)&unk_49D9D40;
    v94 = &v499[10 * (unsigned int)v500];
    if ( v499 == v94 )
      goto LABEL_127;
    do
    {
      v94 -= 10;
      v95 = v94[4];
      if ( (unsigned __int64 *)v95 != v94 + 6 )
        j_j___libc_free_0(v95);
      if ( (unsigned __int64 *)*v94 != v94 + 2 )
        j_j___libc_free_0(*v94);
    }
    while ( v93 != v94 );
    goto LABEL_126;
  }
  v468.m128i_i64[0] = (__int64)&v469;
  v468.m128i_i64[1] = 0x800000000LL;
  if ( !(unsigned __int8)sub_282FEA0((__int64)&v479, v479, (__int64)&v468, v480) )
  {
    v400 = v482;
    v401 = *v482;
    v402 = sub_B2BE50(*v482);
    if ( !sub_B6EA50(v402) )
    {
      v420 = sub_B2BE50(v401);
      v421 = sub_B6F970(v420);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v421 + 48LL))(v421) )
      {
LABEL_238:
        if ( (__m128i *)v468.m128i_i64[0] != &v469 )
          _libc_free(v468.m128i_u64[0]);
        goto LABEL_98;
      }
    }
    v407 = **(_QWORD **)(v479 + 32);
    sub_D4BD20(&v458, v479, v403, v404, v405, v406);
    sub_B157E0((__int64)&v463, &v458);
    sub_B17640((__int64)&v505, (__int64)"loop-interchange", (__int64)"UnsupportedPHIOuter", 19, &v463, v407);
    sub_B18290(
      (__int64)&v505,
      "Only outer loops with induction or reduction PHI nodes can be interchanged currently.",
      0x55u);
    v412 = _mm_loadu_si128(&v508);
    v413 = _mm_loadu_si128(&v510);
    v499 = (unsigned __int64 *)v501;
    LODWORD(v493) = v506;
    v414 = _mm_loadu_si128(&v511);
    v495 = v412;
    BYTE4(v493) = BYTE4(v506);
    v497 = v413;
    v494 = v507;
    v492 = (__int64 *)&unk_49D9D40;
    v498 = v414;
    v496 = v509;
    v500 = 0x400000000LL;
    if ( v513 )
      sub_2830C60((__int64)&v499, (__int64)&v512, v408, v409, v410, v411);
    v502 = v515;
    v448 = v512;
    v503 = v516;
    v504 = v517;
    v492 = (__int64 *)&unk_49D9DB0;
    v505 = (__int64)&unk_49D9D40;
    v415 = &v512[10 * v513];
    while ( v448 != v415 )
    {
      v415 -= 10;
      v416 = v415[4];
      if ( (unsigned __int64 *)v416 != v415 + 6 )
        j_j___libc_free_0(v416);
      if ( (unsigned __int64 *)*v415 != v415 + 2 )
        j_j___libc_free_0(*v415);
    }
    if ( v512 != (unsigned __int64 *)v514 )
      _libc_free((unsigned __int64)v512);
    if ( v458.m128i_i64[0] )
      sub_B91220((__int64)&v458, v458.m128i_i64[0]);
    sub_1049740(v400, (__int64)&v492);
    v417 = (unsigned __int64)v499;
    v492 = (__int64 *)&unk_49D9D40;
    v418 = &v499[10 * (unsigned int)v500];
    while ( (unsigned __int64 *)v417 != v418 )
    {
      v418 -= 10;
      v419 = v418[4];
      if ( (unsigned __int64 *)v419 != v418 + 6 )
        j_j___libc_free_0(v419);
      if ( (unsigned __int64 *)*v418 != v418 + 2 )
        j_j___libc_free_0(*v418);
    }
LABEL_236:
    if ( v499 != (unsigned __int64 *)v501 )
      _libc_free((unsigned __int64)v499);
    goto LABEL_238;
  }
  v468.m128i_i32[2] = 0;
  v100 = v479;
  while ( 1 )
  {
    v101 = *(__int64 **)(v100 + 8);
    if ( *(__int64 **)(v100 + 16) == v101 )
      break;
    v100 = *v101;
    if ( !(unsigned __int8)sub_282FEA0((__int64)&v479, *v101, (__int64)&v468, 0) )
    {
      v162 = v482;
      v163 = *v482;
      v164 = sub_B2BE50(*v482);
      if ( !sub_B6EA50(v164) )
      {
        v182 = sub_B2BE50(v163);
        v183 = sub_B6F970(v182);
        if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v183 + 48LL))(v183) )
          goto LABEL_238;
      }
      v169 = **(_QWORD **)(v100 + 32);
      sub_D4BD20(&v458, v100, v165, v166, v167, v168);
      sub_B157E0((__int64)&v463, &v458);
      sub_B17640((__int64)&v505, (__int64)"loop-interchange", (__int64)"UnsupportedPHIInner", 19, &v463, v169);
      sub_B18290(
        (__int64)&v505,
        "Only inner loops with induction or reduction PHI nodes can be interchange currently.",
        0x54u);
      v174 = _mm_loadu_si128(&v508);
      v175 = _mm_loadu_si128(&v510);
      v499 = (unsigned __int64 *)v501;
      LODWORD(v493) = v506;
      v176 = _mm_loadu_si128(&v511);
      v495 = v174;
      BYTE4(v493) = BYTE4(v506);
      v497 = v175;
      v494 = v507;
      v492 = (__int64 *)&unk_49D9D40;
      v498 = v176;
      v496 = v509;
      v500 = 0x400000000LL;
      if ( v513 )
        sub_2830C60((__int64)&v499, (__int64)&v512, v170, v171, v172, v173);
      v502 = v515;
      v446 = v512;
      v503 = v516;
      v504 = v517;
      v492 = (__int64 *)&unk_49D9DB0;
      v505 = (__int64)&unk_49D9D40;
      v177 = &v512[10 * v513];
      while ( v446 != v177 )
      {
        v177 -= 10;
        v178 = v177[4];
        if ( (unsigned __int64 *)v178 != v177 + 6 )
          j_j___libc_free_0(v178);
        if ( (unsigned __int64 *)*v177 != v177 + 2 )
          j_j___libc_free_0(*v177);
      }
      if ( v512 != (unsigned __int64 *)v514 )
        _libc_free((unsigned __int64)v512);
      if ( v458.m128i_i64[0] )
        sub_B91220((__int64)&v458, v458.m128i_i64[0]);
      sub_1049740(v162, (__int64)&v492);
      v179 = (unsigned __int64)v499;
      v492 = (__int64 *)&unk_49D9D40;
      v180 = &v499[10 * (unsigned int)v500];
      while ( (unsigned __int64 *)v179 != v180 )
      {
        v180 -= 10;
        v181 = v180[4];
        if ( (unsigned __int64 *)v181 != v180 + 6 )
          j_j___libc_free_0(v181);
        if ( (unsigned __int64 *)*v180 != v180 + 2 )
          j_j___libc_free_0(*v180);
      }
      goto LABEL_236;
    }
  }
  v438 = sub_2830810((__int64)&v479);
  if ( !v438 )
  {
    v386 = v482;
    v387 = *v482;
    v388 = sub_B2BE50(*v482);
    if ( sub_B6EA50(v388)
      || (v398 = sub_B2BE50(v387),
          v399 = sub_B6F970(v398),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v399 + 48LL))(v399)) )
    {
      v393 = **(_QWORD **)(v480 + 32);
      sub_D4BD20(&v458, v480, v389, v390, v391, v392);
      sub_B157E0((__int64)&v463, &v458);
      sub_B17640((__int64)&v505, (__int64)"loop-interchange", (__int64)"UnsupportedStructureInner", 25, &v463, v393);
      sub_B18290((__int64)&v505, "Inner loop structure not understood currently.", 0x2Eu);
      sub_23FE290((__int64)&v492, (__int64)&v505, v394, v395, v396, v397);
      v504 = v517;
      v492 = (__int64 *)&unk_49D9DB0;
      v505 = (__int64)&unk_49D9D40;
      sub_23FD590((__int64)&v512);
      if ( v458.m128i_i64[0] )
        sub_B91220((__int64)&v458, v458.m128i_i64[0]);
      sub_1049740(v386, (__int64)&v492);
      v492 = (__int64 *)&unk_49D9D40;
      sub_23FD590((__int64)&v499);
    }
    goto LABEL_238;
  }
  if ( (__m128i *)v468.m128i_i64[0] != &v469 )
    _libc_free(v468.m128i_u64[0]);
  v102 = v479;
  v103 = v480;
  v104 = **(_QWORD **)(v479 + 32);
  v105 = sub_D4B130(v480);
  v106 = sub_D47930(v102);
  v107 = *(_QWORD *)(v104 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v107 == v104 + 48 )
    goto LABEL_47;
  if ( !v107 )
    goto LABEL_485;
  v108 = *(unsigned __int8 *)(v107 - 24);
  if ( (unsigned int)(v108 - 30) > 0xA )
LABEL_47:
    BUG();
  if ( (_BYTE)v108 != 31 )
    goto LABEL_162;
  v440 = v107 - 24;
  v109 = sub_B46E30(v107 - 24);
  if ( v109 )
  {
    v451 = v104;
    v110 = 0;
    v111 = v440;
    do
    {
      v441 = v109;
      v112 = sub_B46EC0(v111, v110);
      v109 = v441;
      if ( v105 != v112 && **(_QWORD **)(v103 + 32) != v112 && v106 != v112 )
        goto LABEL_162;
      ++v110;
    }
    while ( v441 != v110 );
    v104 = v451;
  }
  if ( sub_282F890(v104)
    || sub_282F890(v106)
    || v104 != v105 && sub_282F890(v105)
    || (v113 = sub_D47470(v103), v106 != sub_D52390(v113, v106, 0))
    || sub_282F890(v113) )
  {
LABEL_162:
    v114 = v482;
    v115 = *v482;
    v116 = sub_B2BE50(*v482);
    if ( !sub_B6EA50(v116) )
    {
      v184 = sub_B2BE50(v115);
      v185 = sub_B6F970(v184);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v185 + 48LL))(v185) )
        goto LABEL_98;
    }
    v121 = **(_QWORD **)(v480 + 32);
    sub_D4BD20(&v463, v480, v117, v118, v119, v120);
    sub_B157E0((__int64)&v468, &v463);
    sub_B17640((__int64)&v505, (__int64)"loop-interchange", (__int64)"NotTightlyNested", 16, &v468, v121);
    sub_B18290((__int64)&v505, "Cannot interchange loops because they are not tightly nested.", 0x3Du);
    v126 = _mm_loadu_si128(&v508);
    v127 = _mm_loadu_si128(&v510);
    v499 = (unsigned __int64 *)v501;
    LODWORD(v493) = v506;
    v128 = _mm_loadu_si128(&v511);
    v495 = v126;
    BYTE4(v493) = BYTE4(v506);
    v497 = v127;
    v494 = v507;
    v492 = (__int64 *)&unk_49D9D40;
    v498 = v128;
    v496 = v509;
    v500 = 0x400000000LL;
    if ( v513 )
      sub_2830C60((__int64)&v499, (__int64)&v512, v122, v123, v124, v125);
    v502 = v515;
    v503 = v516;
    v504 = v517;
    v492 = (__int64 *)&unk_49D9DB0;
    v505 = (__int64)&unk_49D9D40;
    sub_23FD590((__int64)&v512);
    if ( v463.m128i_i64[0] )
      sub_B91220((__int64)&v463, v463.m128i_i64[0]);
    sub_1049740(v114, (__int64)&v492);
    v93 = v499;
    v492 = (__int64 *)&unk_49D9D40;
    v129 = &v499[10 * (unsigned int)v500];
    if ( v499 == v129 )
    {
LABEL_127:
      if ( v93 != (unsigned __int64 *)v501 )
        _libc_free((unsigned __int64)v93);
      goto LABEL_98;
    }
    do
    {
      v129 -= 10;
      v130 = v129[4];
      if ( (unsigned __int64 *)v130 != v129 + 6 )
        j_j___libc_free_0(v130);
      if ( (unsigned __int64 *)*v129 != v129 + 2 )
        j_j___libc_free_0(*v129);
    }
    while ( v93 != v129 );
LABEL_126:
    v93 = v499;
    goto LABEL_127;
  }
  v186 = v480;
  v187 = sub_D47600(v480);
  v188 = sub_AA5930(v187);
  v190 = v189;
  if ( v188 != v189 )
  {
    while ( 1 )
    {
      if ( (*(_DWORD *)(v188 + 4) & 0x7FFFFFFu) > 1 )
        goto LABEL_249;
      v191 = *(_QWORD *)(v188 + 16);
      if ( v191 )
        break;
LABEL_260:
      v208 = *(_QWORD *)(v188 + 32);
      if ( !v208 )
        goto LABEL_485;
      v188 = 0;
      if ( *(_BYTE *)(v208 - 24) == 84 )
        v188 = v208 - 24;
      if ( v190 == v188 )
        goto LABEL_264;
    }
    while ( 1 )
    {
      v192 = *(_QWORD *)(v191 + 24);
      if ( *(_BYTE *)v192 != 84 )
        goto LABEL_249;
      if ( v487 )
      {
        v206 = v484;
        v207 = &v484[8 * HIDWORD(v485)];
        if ( v484 != v207 )
        {
          while ( v192 != *(_QWORD *)v206 )
          {
            v206 += 8;
            if ( v207 == v206 )
              goto LABEL_435;
          }
          goto LABEL_259;
        }
      }
      else if ( sub_C8CA60((__int64)&v483, v192) )
      {
        goto LABEL_259;
      }
LABEL_435:
      v377 = *(_QWORD *)(v192 + 40);
      if ( *(_BYTE *)(v186 + 84) )
      {
        v378 = *(_QWORD **)(v186 + 64);
        v379 = &v378[*(unsigned int *)(v186 + 76)];
        if ( v378 != v379 )
        {
          while ( v377 != *v378 )
          {
            if ( v379 == ++v378 )
              goto LABEL_259;
          }
LABEL_249:
          v193 = v482;
          v194 = *v482;
          v195 = sub_B2BE50(*v482);
          if ( sub_B6EA50(v195)
            || (v382 = sub_B2BE50(v194),
                v383 = sub_B6F970(v382),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v383 + 48LL))(v383)) )
          {
            v200 = v480;
            goto LABEL_251;
          }
LABEL_98:
          v438 = 0;
          goto LABEL_74;
        }
      }
      else if ( sub_C8CA60(v186 + 56, v377) )
      {
        goto LABEL_249;
      }
LABEL_259:
      v191 = *(_QWORD *)(v191 + 8);
      if ( !v191 )
        goto LABEL_260;
    }
  }
LABEL_264:
  v209 = v479;
  v210 = sub_D47600(v479);
  v211 = sub_AA5930(v210);
  v213 = v212;
  v214 = v211;
  while ( v213 != v214 )
  {
    v215 = *(_DWORD *)(v214 + 4);
    v216 = 0;
    if ( (v215 & 0x7FFFFFF) != 0 )
    {
      do
      {
        v217 = *(_QWORD *)(*(_QWORD *)(v214 - 8) + 32 * v216);
        if ( *(_BYTE *)v217 > 0x1Cu )
        {
          v218 = *(_QWORD *)(v217 + 40);
          if ( v218 == sub_D47930(v209) )
          {
            v219 = sub_D47930(v209);
            if ( !sub_AA5510(v219) )
            {
              v193 = v482;
              v220 = *v482;
              v221 = sub_B2BE50(*v482);
              if ( !sub_B6EA50(v221) )
              {
                v384 = sub_B2BE50(v220);
                v385 = sub_B6F970(v384);
                if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v385 + 48LL))(v385) )
                  goto LABEL_98;
              }
              v200 = v479;
LABEL_251:
              v201 = **(_QWORD **)(v200 + 32);
              sub_D4BD20(&v463, v200, v196, v197, v198, v199);
              sub_B157E0((__int64)&v468, &v463);
              sub_B17640((__int64)&v505, (__int64)"loop-interchange", (__int64)"UnsupportedExitPHI", 18, &v468, v201);
              sub_B18290((__int64)&v505, "Found unsupported PHI node in loop exit.", 0x28u);
              sub_23FE290((__int64)&v492, (__int64)&v505, v202, v203, v204, v205);
              v504 = v517;
              v492 = (__int64 *)&unk_49D9DB0;
              v505 = (__int64)&unk_49D9D40;
              sub_23FD590((__int64)&v512);
              if ( v463.m128i_i64[0] )
                sub_B91220((__int64)&v463, v463.m128i_i64[0]);
              sub_1049740(v193, (__int64)&v492);
              v492 = (__int64 *)&unk_49D9D40;
              sub_23FD590((__int64)&v499);
              goto LABEL_98;
            }
          }
          v215 = *(_DWORD *)(v214 + 4);
        }
        ++v216;
      }
      while ( (v215 & 0x7FFFFFFu) > (unsigned int)v216 );
    }
    v222 = *(_QWORD *)(v214 + 32);
    if ( !v222 )
      goto LABEL_485;
    v214 = 0;
    if ( *(_BYTE *)(v222 - 24) == 84 )
      v214 = v222 - 24;
  }
  v223 = *a1;
  v427 = (__int64 *)a1[5];
  v505 = v445.m128i_i64[1];
  v224 = *(_QWORD *)(a6 + 8);
  v225 = *(unsigned int *)(a6 + 24);
  if ( (_DWORD)v225 )
  {
    v226 = v225 - 1;
    v227 = (v225 - 1) & (((unsigned __int32)v445.m128i_i32[2] >> 9) ^ ((unsigned __int32)v445.m128i_i32[2] >> 4));
    v228 = (__int64 *)(v224 + 16LL * v227);
    v229 = *v228;
    if ( v445.m128i_i64[1] == *v228 )
    {
LABEL_280:
      v230 = (__int64 *)(v224 + 16 * v225);
      if ( v228 != v230 )
      {
        v231 = v226 & (((unsigned __int32)v445.m128i_i32[0] >> 9) ^ ((unsigned __int32)v445.m128i_i32[0] >> 4));
        v232 = (__int64 *)(v224 + 16LL * v231);
        v233 = *v232;
        if ( v445.m128i_i64[0] == *v232 )
        {
LABEL_282:
          if ( v230 != v232 )
          {
            if ( *((_DWORD *)v228 + 2) < *((_DWORD *)v232 + 2) )
            {
LABEL_284:
              v234 = *v427;
              v235 = sub_B2BE50(*v427);
              if ( sub_B6EA50(v235)
                || (v374 = sub_B2BE50(v234),
                    v375 = sub_B6F970(v374),
                    (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v375 + 48LL))(v375)) )
              {
                v240 = **(_QWORD **)(v445.m128i_i64[1] + 32);
                sub_D4BD20(&v463, v445.m128i_i64[1], v236, v237, v238, v239);
                sub_B157E0((__int64)&v468, &v463);
                sub_B17430((__int64)&v505, (__int64)"loop-interchange", (__int64)"Interchanged", 12, &v468, v240);
                sub_B18290((__int64)&v505, "Loop interchanged with enclosing loop.", 0x26u);
                v245 = _mm_loadu_si128(&v508);
                v499 = (unsigned __int64 *)v501;
                LODWORD(v493) = v506;
                v495 = v245;
                v246 = _mm_loadu_si128(&v510);
                BYTE4(v493) = BYTE4(v506);
                v497 = v246;
                v247 = _mm_loadu_si128(&v511);
                v494 = v507;
                v492 = (__int64 *)&unk_49D9D40;
                v496 = v509;
                v500 = 0x400000000LL;
                v498 = v247;
                if ( v513 )
                {
                  sub_2830C60((__int64)&v499, (__int64)&v512, v241, v242, v243, v244);
                  v505 = (__int64)&unk_49D9D40;
                  v325 = v512;
                  v502 = v515;
                  v503 = v516;
                  v504 = v517;
                  v492 = (__int64 *)&unk_49D9D78;
                  v326 = 10LL * v513;
                  v248 = &v512[v326];
                  if ( v512 != &v512[v326] )
                  {
                    do
                    {
                      v248 -= 10;
                      v327 = v248[4];
                      if ( (unsigned __int64 *)v327 != v248 + 6 )
                        j_j___libc_free_0(v327);
                      if ( (unsigned __int64 *)*v248 != v248 + 2 )
                        j_j___libc_free_0(*v248);
                    }
                    while ( v325 != v248 );
                    v248 = v512;
                  }
                }
                else
                {
                  v248 = v512;
                  v502 = v515;
                  v503 = v516;
                  v504 = v517;
                  v492 = (__int64 *)&unk_49D9D78;
                }
                if ( v248 != (unsigned __int64 *)v514 )
                  _libc_free((unsigned __int64)v248);
                if ( v463.m128i_i64[0] )
                  sub_B91220((__int64)&v463, v463.m128i_i64[0]);
                sub_1049740(v427, (__int64)&v492);
                v249 = (unsigned __int64)v499;
                v492 = (__int64 *)&unk_49D9D40;
                v250 = &v499[10 * (unsigned int)v500];
                if ( v499 != v250 )
                {
                  do
                  {
                    v250 -= 10;
                    v251 = v250[4];
                    if ( (unsigned __int64 *)v251 != v250 + 6 )
                      j_j___libc_free_0(v251);
                    if ( (unsigned __int64 *)*v250 != v250 + 2 )
                      j_j___libc_free_0(*v250);
                  }
                  while ( (unsigned __int64 *)v249 != v250 );
                  v250 = v499;
                }
                if ( v250 != (unsigned __int64 *)v501 )
                  _libc_free((unsigned __int64)v250);
              }
              v252 = a1[3];
              v253 = a1[1];
              v468 = v445;
              v254 = *a1;
              v470 = v252;
              v469.m128i_i64[0] = v254;
              v469.m128i_i64[1] = v253;
              v471 = &v479;
              if ( *(_QWORD *)(v445.m128i_i64[1] + 16) == *(_QWORD *)(v445.m128i_i64[1] + 8) )
              {
                v311 = sub_D4B130(v445.m128i_i64[1]);
                v312 = v471;
                v313 = v311;
                v314 = *((unsigned int *)v471 + 26);
                if ( !(_DWORD)v314 )
                  goto LABEL_319;
                v315 = 0;
                v492 = &v494;
                v493 = 0x800000000LL;
                v316 = v471[12];
                v317 = 8;
                v318 = v316 + 8 * v314;
                v319 = v318;
                while ( 1 )
                {
                  v324 = *(_QWORD *)(*(_QWORD *)v316 - 8LL);
                  if ( v313 == *(_QWORD *)(v324 + 32LL * *(unsigned int *)(*(_QWORD *)v316 + 72LL)) )
                    v320 = *(_BYTE **)(v324 + 32);
                  else
                    v320 = *(_BYTE **)v324;
                  if ( *v320 < 0x1Du )
                    v320 = 0;
                  v321 = (__int64)v320;
                  v322 = v315;
                  v323 = v315 + 1LL;
                  if ( v323 > v317 )
                  {
                    v447 = v321;
                    sub_C8D5F0((__int64)&v492, &v494, v323, 8u, v318, v321);
                    v322 = (unsigned int)v493;
                    v321 = v447;
                  }
                  v316 += 8;
                  v492[v322] = v321;
                  LODWORD(v493) = v493 + 1;
                  if ( v319 == v316 )
                    break;
                  v315 = v493;
                  v317 = HIDWORD(v493);
                }
                v353 = v469.m128i_i64[1];
                v508.m128i_i16[4] = 257;
                v354 = v470;
                v355 = (_QWORD *)(sub_D47930(v468.m128i_i64[1]) + 48);
                v356 = *v355 & 0xFFFFFFFFFFFFFFF8LL;
                if ( (_QWORD *)v356 == v355 )
                {
                  v358 = 0;
                }
                else
                {
                  if ( !v356 )
                    goto LABEL_485;
                  v357 = *(unsigned __int8 *)(v356 - 24);
                  v358 = 0;
                  v359 = v356 - 24;
                  if ( (unsigned int)(v357 - 30) < 0xB )
                    v358 = v359;
                }
                v360 = sub_D47930(v468.m128i_i64[1]);
                v361 = sub_F36960(v360, (__int64 *)(v358 + 24), 0, v354, v353, 0, (void **)&v505, 0);
                v508.m128i_i64[1] = (__int64)&v510;
                v509 = 0x400000000LL;
                v463.m128i_i64[0] = (__int64)&v456;
                v505 = 0;
                v506 = 0;
                v507 = 0;
                v508.m128i_i64[0] = 0;
                v456 = 0;
                v463.m128i_i64[1] = (__int64)&v505;
                v464 = __PAIR128__((unsigned __int64)(v312 + 12), &v468);
                v465[0] = v361;
                v362 = (_QWORD *)(sub_D47930(v468.m128i_i64[1]) + 48);
                v364 = *v362 & 0xFFFFFFFFFFFFFFF8LL;
                if ( (_QWORD *)v364 == v362 )
                  goto LABEL_487;
                if ( !v364 )
                  goto LABEL_485;
                v365 = (unsigned int)*(unsigned __int8 *)(v364 - 24) - 30;
                if ( (unsigned int)v365 > 0xA )
LABEL_487:
                  BUG();
                v366 = *(_BYTE **)(v364 - 120);
                if ( *v366 <= 0x1Cu )
                {
                  v457.m128i_i64[0] = 0;
                }
                else
                {
                  v457.m128i_i64[0] = (__int64)v366;
                  sub_28316E0((__int64)&v505, v457.m128i_i64, v365, v363, v422, v423);
                }
                sub_2831D20((unsigned int **)&v463);
                v370 = (unsigned int)v493;
                v371 = &v492[(unsigned int)v493];
                for ( j = v492; v371 != j; ++j )
                {
                  v373 = *j;
                  v458.m128i_i64[0] = v373;
                  sub_28316E0((__int64)&v505, v458.m128i_i64, v370, v367, v368, v369);
                }
                sub_2831D20((unsigned int **)&v463);
                if ( (__m128i *)v508.m128i_i64[1] != &v510 )
                  _libc_free(v508.m128i_u64[1]);
                sub_C7D6A0(v506, 8LL * v508.m128i_u32[0], 8);
                if ( v492 != &v494 )
                  _libc_free((unsigned __int64)v492);
                v445.m128i_i64[1] = v468.m128i_i64[1];
              }
              v255 = **(_QWORD **)(v445.m128i_i64[1] + 32);
              v256 = sub_AA4FF0(v255);
              v257 = v255 + 48;
              v258 = v256;
              if ( v256 )
              {
                v258 = v256 - 24;
                v259 = *(_QWORD *)(v255 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v257 == v259 )
                  goto LABEL_307;
              }
              else
              {
                v259 = *(_QWORD *)(v255 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v257 == v259 )
                  goto LABEL_310;
              }
              if ( !v259 )
                goto LABEL_485;
              v260 = *(unsigned __int8 *)(v259 - 24);
              v261 = v259 - 24;
              if ( (unsigned int)(v260 - 30) >= 0xB )
                v261 = 0;
              if ( v258 == v261 )
              {
LABEL_310:
                v268 = sub_D4B130(v468.m128i_i64[1]);
                v269 = **(_QWORD **)(v468.m128i_i64[0] + 32);
                if ( v268 != v269 )
                {
                  v505 = 0;
                  v506 = (__int64)&v508.m128i_i64[1];
                  v507 = 4;
                  v508.m128i_i32[0] = 0;
                  v508.m128i_i8[4] = 1;
                  v270 = *(_QWORD *)(v268 + 56);
                  if ( v270 != (*(_QWORD *)(v268 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
                  {
                    v453 = *(_QWORD *)(v268 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                    v271 = v424;
                    do
                    {
                      v272 = v270;
                      v270 = *(_QWORD *)(v270 + 8);
                      v273 = (_QWORD *)(v272 - 24);
                      v274 = *(_QWORD *)(v269 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                      if ( v274 == v269 + 48 )
                      {
                        v275 = 0;
                      }
                      else
                      {
                        if ( !v274 )
                          goto LABEL_485;
                        v275 = v274 - 24;
                        if ( (unsigned int)*(unsigned __int8 *)(v274 - 24) - 30 >= 0xB )
                          v275 = 0;
                      }
                      LOWORD(v271) = 0;
                      sub_B44500(v273, v275 + 24, v271);
                    }
                    while ( v270 != v453 );
                  }
                }
                if ( (unsigned __int8)sub_2832000(v468.m128i_i64) )
                {
                  v328 = sub_D4B130(v468.m128i_i64[0]);
                  v329 = sub_D4B130(v468.m128i_i64[1]);
                  v332 = *(_QWORD *)(v328 + 56);
                  v333 = v329;
                  v334 = *(_QWORD *)(v328 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  v505 = (__int64)&v507;
                  v506 = 0x400000000LL;
                  if ( v334 == v332 )
                  {
                    LODWORD(v506) = 0;
                  }
                  else
                  {
                    v335 = v332;
                    v336 = 0;
                    do
                    {
                      v335 = *(_QWORD *)(v335 + 8);
                      ++v336;
                    }
                    while ( v334 != v335 );
                    v337 = &v507;
                    if ( v336 > 4 )
                    {
                      sub_C8D5F0((__int64)&v505, &v507, v336, 8u, v330, v331);
                      v337 = (__int64 *)(v505 + 8LL * (unsigned int)v506);
                    }
                    do
                    {
                      v338 = v332 - 24;
                      if ( !v332 )
                        v338 = 0;
                      *v337++ = v338;
                      v332 = *(_QWORD *)(v332 + 8);
                    }
                    while ( v335 != v332 );
                    LODWORD(v506) = v506 + v336;
                    v339 = (char *)(v505 + 8LL * (unsigned int)v506);
                    if ( (char *)v505 != v339 )
                    {
                      v340 = (char *)v505;
                      do
                      {
                        v341 = *(_QWORD **)v340;
                        v340 += 8;
                        sub_B43D10(v341);
                      }
                      while ( v339 != v340 );
                    }
                  }
                  v342 = *(_QWORD *)(v328 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v328 + 48 == v342 )
                    goto LABEL_488;
                  if ( !v342 )
                    goto LABEL_485;
                  if ( (unsigned int)*(unsigned __int8 *)(v342 - 24) - 30 > 0xA )
LABEL_488:
                    BUG();
                  v343 = *(_QWORD *)(v333 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v333 + 48 == v343 )
                  {
                    v345 = 0;
                  }
                  else
                  {
                    if ( !v343 )
                      goto LABEL_485;
                    v344 = *(unsigned __int8 *)(v343 - 24);
                    v345 = v343 - 24;
                    if ( (unsigned int)(v344 - 30) >= 0xB )
                      v345 = 0;
                  }
                  sub_AA80F0(
                    *(_QWORD *)(v342 + 16),
                    (unsigned __int64 *)v342,
                    0,
                    v333,
                    *(__int64 **)(v333 + 56),
                    1,
                    (__int64 *)(v345 + 24),
                    0);
                  v346 = (__int64 *)(v505 + 8LL * (unsigned int)v506);
                  if ( (__int64 *)v505 != v346 )
                  {
                    v455 = v333 + 48;
                    v347 = v333;
                    v348 = (__int64 *)v505;
                    v349 = v425;
                    do
                    {
                      v350 = (_QWORD *)*v348;
                      v351 = *(_QWORD *)(v347 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                      if ( v455 == v351 )
                      {
                        v352 = 0;
                      }
                      else
                      {
                        if ( !v351 )
                          goto LABEL_485;
                        v352 = v351 - 24;
                        if ( (unsigned int)*(unsigned __int8 *)(v351 - 24) - 30 >= 0xB )
                          v352 = 0;
                      }
                      LOWORD(v349) = 0;
                      ++v348;
                      sub_B44220(v350, v352 + 24, v349);
                    }
                    while ( v346 != v348 );
                    v346 = (__int64 *)v505;
                  }
                  if ( v346 != &v507 )
                    _libc_free((unsigned __int64)v346);
                }
LABEL_319:
                sub_11D2180(v445.m128i_i64[0], a1[3], a1[1], *a1, v276, v277);
                v278 = (__int64 *)(*a2 + v430);
                v279 = (__int64 *)(v429 + *a2);
                v280 = *v279;
                *v279 = *v278;
                *v278 = v280;
                v281 = (_QWORD *)*a5;
                v282 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(a5[1] - *a5) >> 3);
                if ( (_DWORD)v282 )
                {
                  v283 = 3LL * (unsigned int)v282;
                  v284 = 0;
                  v285 = 8 * v283;
                  while ( 1 )
                  {
                    v286 = v281[v284 / 8];
                    v284 += 24LL;
                    v287 = (char *)(v286 + v432);
                    v288 = (char *)(v433 + v286);
                    v289 = *v288;
                    *v288 = *v287;
                    *v287 = v289;
                    if ( v285 == v284 )
                      break;
                    v281 = (_QWORD *)*a5;
                  }
                }
                goto LABEL_74;
              }
LABEL_307:
              v262 = v469.m128i_i64[1];
              v263 = v470;
              v508.m128i_i16[4] = 257;
              v265 = (__int64 *)sub_AA4FF0(v255);
              v266 = 0;
              if ( v265 )
                v266 = v264;
              v267 = 1;
              BYTE1(v267) = v266;
              sub_F36960(v255, v265, v267, v263, v262, 0, (void **)&v505, 0);
              goto LABEL_310;
            }
            v290 = a1[4];
            v291 = *(_QWORD **)(v290 + 144);
            v292 = (__int64)&v291[3 * *(unsigned int *)(v290 + 152)];
            sub_282FB40(v291, v292, v445.m128i_i64[1]);
            v294 = sub_282FB40(v293, v292, v445.m128i_i64[0]);
            if ( (_QWORD *)v292 == v294 )
            {
              v298 = 0;
              v297 = -1;
            }
            else
            {
              v297 = v294[1];
              v298 = *((_DWORD *)v294 + 4);
            }
            if ( v296 != v298 || v297 != v295 )
              goto LABEL_353;
          }
        }
        else
        {
          v380 = 1;
          while ( v233 != -4096 )
          {
            v381 = v380 + 1;
            v231 = v226 & (v380 + v231);
            v232 = (__int64 *)(v224 + 16LL * v231);
            v233 = *v232;
            if ( v445.m128i_i64[0] == *v232 )
              goto LABEL_282;
            v380 = v381;
          }
        }
      }
    }
    else
    {
      v376 = 1;
      while ( v229 != -4096 )
      {
        v227 = v226 & (v376 + v227);
        v228 = (__int64 *)(v224 + 16LL * v227);
        v229 = *v228;
        if ( v445.m128i_i64[1] == *v228 )
          goto LABEL_280;
        ++v376;
      }
    }
  }
  v434 = *(_QWORD *)(v445.m128i_i64[1] + 40);
  if ( *(_QWORD *)(v445.m128i_i64[1] + 32) == v434 )
    goto LABEL_346;
  v436 = *(_QWORD *)(v445.m128i_i64[1] + 32);
  v426 = 0;
  v437 = 0;
  do
  {
    v299 = *(_QWORD *)(*(_QWORD *)v436 + 56LL);
    v442 = *(_QWORD *)v436 + 48LL;
    if ( v299 != v442 )
    {
      while ( 1 )
      {
        if ( !v299 )
          goto LABEL_485;
        if ( *(_BYTE *)(v299 - 24) == 63 )
        {
          v300 = *(_DWORD *)(v299 - 20) & 0x7FFFFFF;
          if ( v300 )
            break;
        }
LABEL_343:
        v299 = *(_QWORD *)(v299 + 8);
        if ( v442 == v299 )
          goto LABEL_344;
      }
      v301 = v299 - 24;
      v439 = 0;
      v454 = v300 - 1;
      v302 = v299;
      v444 = 0;
      for ( k = 0; ; ++k )
      {
        v304 = sub_D97040(v223, *(_QWORD *)(*(_QWORD *)(v301 + 32 * (k - v300)) + 8LL));
        if ( !v304 )
          goto LABEL_336;
        v305 = sub_DD8400(v223, *(_QWORD *)(v301 + 32 * (k - (*(_DWORD *)(v302 - 20) & 0x7FFFFFF))));
        if ( *((_WORD *)v305 + 12) != 8 )
          goto LABEL_336;
        v306 = v305[6];
        if ( v445.m128i_i64[1] == v306 )
          break;
        if ( v445.m128i_i64[0] != v306 )
          goto LABEL_336;
        if ( v439 )
          goto LABEL_359;
        v444 = v304;
        if ( v454 == k )
        {
LABEL_356:
          v299 = v302;
          goto LABEL_343;
        }
LABEL_337:
        v300 = *(_DWORD *)(v302 - 20) & 0x7FFFFFF;
      }
      if ( v444 )
      {
        ++v437;
        v299 = v302;
        goto LABEL_343;
      }
      if ( v445.m128i_i64[0] == v445.m128i_i64[1] )
      {
LABEL_359:
        ++v426;
        v299 = v302;
        goto LABEL_343;
      }
      v439 = v304;
LABEL_336:
      if ( v454 == k )
        goto LABEL_356;
      goto LABEL_337;
    }
LABEL_344:
    v436 += 8;
  }
  while ( v434 != v436 );
  if ( v437 - v426 < 0 && (int)qword_4FFFD28 > v437 - v426 )
  {
LABEL_380:
    v427 = (__int64 *)a1[5];
    goto LABEL_284;
  }
LABEL_346:
  v307 = (_QWORD *)a5[1];
  v308 = (_QWORD *)*a5;
  if ( (_QWORD *)*a5 != v307 )
  {
    while ( 1 )
    {
      v310 = *(_BYTE *)(*v308 + v432);
      if ( v310 == 61 )
        break;
      if ( v310 == 73 )
        break;
      v309 = *(_BYTE *)(*v308 + v433);
      if ( v309 != 61 && v309 != 73 )
        break;
      v308 += 3;
      if ( v307 == v308 )
        goto LABEL_380;
    }
  }
LABEL_353:
  sub_2830EE0(v427, &v505);
  v438 = 0;
LABEL_74:
  if ( v489 != v491 )
    _libc_free((unsigned __int64)v489);
  if ( !v487 )
    _libc_free((unsigned __int64)v484);
  return v438;
}
