// Function: sub_19B15B0
// Address: 0x19b15b0
//
__int64 __fastcall sub_19B15B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char *a6,
        __m128i a7,
        __m128i a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15,
        __int64 a16)
{
  __int64 v17; // r13
  bool v18; // al
  _QWORD *v19; // rax
  unsigned int v20; // edx
  unsigned __int64 *v21; // rax
  __int64 v22; // r15
  int v23; // ebx
  __int64 v24; // rax
  __int64 v25; // r13
  int v26; // ecx
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 *v29; // r13
  __int64 *i; // r14
  unsigned __int64 *v31; // rbx
  unsigned __int64 *v32; // r12
  _QWORD *v33; // rbx
  _QWORD *v34; // r12
  unsigned __int64 *v35; // r13
  _BYTE *v36; // r13
  unsigned __int64 v37; // r12
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // r15
  unsigned __int64 v40; // rbx
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // r15
  unsigned __int64 v43; // rbx
  unsigned __int64 v44; // rdi
  __int64 v45; // rax
  unsigned __int64 *v46; // rbx
  unsigned __int64 *v47; // r15
  unsigned int v48; // r15d
  __int64 v50; // r14
  unsigned int v51; // r13d
  __int64 v52; // rax
  __int64 v53; // r13
  unsigned int v54; // eax
  unsigned int v55; // ebx
  __int64 v56; // rax
  __int64 v57; // r13
  __int64 v58; // rax
  _QWORD *v59; // rbx
  _QWORD *v60; // r13
  __int64 v61; // rax
  unsigned __int64 v62; // rbx
  unsigned __int64 *v63; // r13
  __int64 v64; // rax
  int v65; // r15d
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // r13
  __int64 v69; // rbx
  unsigned __int64 v70; // rax
  int v71; // eax
  double v72; // xmm4_8
  double v73; // xmm5_8
  int v74; // r15d
  double v75; // xmm4_8
  double v76; // xmm5_8
  __int64 v77; // rsi
  __int64 v78; // rdx
  __int64 v79; // rcx
  int v80; // r8d
  int v81; // r9d
  __int64 v82; // rdx
  _QWORD *v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rcx
  __int64 v86; // r8
  int v87; // r9d
  __int64 v88; // rdx
  __int64 v89; // rcx
  int v90; // r8d
  unsigned int v91; // r9d
  __int64 v92; // rdx
  __int64 v93; // rcx
  __int64 v94; // r8
  __int64 v95; // rdx
  __int64 v96; // rcx
  __int64 v97; // r8
  unsigned __int64 v98; // r9
  __int64 v99; // r8
  __int64 v100; // r9
  __int64 v101; // rax
  __int64 v102; // r13
  int v103; // edx
  __int64 v104; // rbx
  __int64 v105; // r8
  unsigned int v106; // eax
  __int64 v107; // rsi
  int v108; // r13d
  _QWORD *v109; // r12
  unsigned int v110; // eax
  _QWORD *v111; // rbx
  unsigned __int64 *v112; // r13
  __int64 v113; // rax
  __int64 v114; // rbx
  __int64 v115; // rax
  unsigned __int64 *v116; // r13
  unsigned __int64 *v117; // r15
  __int64 v118; // r11
  _QWORD *v119; // rax
  __int64 v120; // rdi
  __int64 v121; // rbx
  int v122; // edi
  _QWORD *v123; // rcx
  int k; // edx
  __int64 v125; // r8
  int v126; // edx
  _QWORD *v127; // r15
  unsigned int v128; // r14d
  _QWORD *v129; // rbx
  unsigned int v130; // eax
  unsigned __int64 *v131; // r14
  unsigned int v132; // eax
  int v133; // ebx
  _QWORD *v134; // rdi
  unsigned __int64 v135; // rdx
  _QWORD *v136; // rax
  _QWORD *v137; // rdx
  __int64 *v138; // rax
  __int64 *v139; // r12
  __int64 v140; // rcx
  __int64 *v141; // rbx
  unsigned __int64 v142; // r9
  __int64 *v143; // r8
  __int64 v144; // rsi
  __int64 *v145; // rdi
  __int64 *v146; // rax
  __int64 *v147; // rcx
  unsigned __int64 *v148; // r12
  unsigned __int64 *v149; // rbx
  __int64 v150; // rax
  unsigned int v151; // eax
  int v152; // r12d
  unsigned int v153; // eax
  __int64 *v154; // rsi
  __int64 *v155; // rcx
  _BYTE *v156; // rax
  __int64 *v157; // r10
  __int64 *v158; // r15
  __int64 j; // rbx
  unsigned int v160; // eax
  unsigned __int64 *v161; // rdi
  __int64 v162; // rax
  __int64 v163; // rsi
  __int64 v164; // r13
  __int64 v165; // r12
  __int64 *v166; // rax
  __int64 v167; // rdx
  int v168; // eax
  __int64 v169; // rdi
  __int64 *v170; // rax
  __int64 v171; // rcx
  unsigned __int64 v172; // rsi
  __int64 v173; // rcx
  __int64 v174; // rdx
  __int64 *v175; // rax
  __int64 *v176; // r14
  __int64 v177; // r12
  __int64 v178; // r13
  __int64 v179; // r15
  __int64 v180; // rax
  __int64 v181; // rdx
  __int64 v182; // r13
  __int64 *v183; // rdx
  __int64 v184; // r13
  __int64 v185; // rbx
  __int64 v186; // rax
  __int64 *v187; // rax
  __int64 v188; // rsi
  unsigned int v189; // edx
  __int64 *v190; // rax
  __int64 v191; // rdx
  unsigned int v192; // edi
  __int64 *v193; // rax
  __int64 v194; // rax
  __int64 v195; // rax
  unsigned int v196; // edx
  __int64 v197; // rsi
  __int64 v198; // rdx
  __int64 v199; // rdx
  __int64 v200; // rsi
  __int64 v201; // rsi
  __int64 *v202; // rsi
  __int64 v203; // rcx
  unsigned __int64 v204; // rdi
  __int64 v205; // rcx
  __int64 v206; // rax
  __int64 *v207; // rax
  __int64 v208; // rcx
  unsigned __int64 v209; // rsi
  __int64 v210; // rcx
  unsigned __int64 *v211; // rbx
  unsigned __int64 *v212; // r12
  __int64 v213; // rsi
  int v214; // r13d
  _QWORD *v215; // r12
  _QWORD *v216; // rbx
  unsigned int v217; // eax
  __int64 v218; // rax
  __int64 v219; // rdx
  _QWORD *v220; // rax
  _QWORD *v221; // rdx
  char v222; // al
  _QWORD *v223; // r12
  _QWORD *v224; // rbx
  __int64 v225; // rax
  unsigned int v226; // ecx
  _QWORD *v227; // rdi
  unsigned int v228; // eax
  __int64 v229; // rbx
  unsigned __int64 v230; // rdx
  _QWORD *v231; // rax
  _QWORD *v232; // rdx
  __int64 v233; // rax
  __int64 v234; // rcx
  unsigned __int64 v235; // rax
  __int64 v236; // r14
  __int64 v237; // rdi
  __int64 v238; // rdx
  _QWORD *v239; // rsi
  __int64 *v240; // rax
  __int64 v241; // r11
  unsigned __int64 v242; // r13
  int v243; // ebx
  __int64 *v244; // rdi
  unsigned int v245; // r12d
  unsigned int v246; // edi
  _QWORD *v247; // rax
  __int64 v248; // rdx
  unsigned int v249; // edx
  _QWORD *v250; // rax
  __int64 v251; // rbx
  __int64 v252; // r15
  __int64 v253; // r12
  unsigned __int64 v254; // rdx
  __int64 *v255; // rdi
  __int64 *v256; // rax
  __int64 v257; // r10
  _QWORD *v258; // rax
  unsigned __int64 v259; // r15
  int v260; // eax
  void *v261; // r12
  _QWORD *v262; // rsi
  int v263; // eax
  int v264; // eax
  _QWORD *v265; // rax
  unsigned __int64 v266; // r12
  int v267; // eax
  __int64 v268; // rbx
  void *v269; // rax
  _QWORD *v270; // rsi
  int v271; // eax
  __int64 v272; // rdx
  _QWORD *v273; // rax
  _QWORD *v274; // rdx
  __int64 *v275; // rbx
  __int64 *v276; // r13
  __int64 v277; // rax
  __int64 v278; // rcx
  __int64 v279; // rbx
  _QWORD *v280; // rax
  _QWORD *v281; // r12
  _QWORD *v282; // rax
  unsigned int v283; // ecx
  unsigned int v284; // eax
  int v285; // r12d
  unsigned int v286; // eax
  int v287; // ecx
  _QWORD *v288; // rdx
  int v289; // edx
  int m; // edi
  _QWORD *v291; // rcx
  __int64 v292; // r8
  _QWORD *v293; // rax
  int v294; // edx
  __int64 v295; // rax
  _QWORD *v296; // rax
  __int64 *v297; // rdx
  unsigned __int64 v298; // rax
  unsigned __int64 v299; // rax
  unsigned __int64 v300; // rbx
  __int64 *v301; // rdi
  _QWORD *v302; // rcx
  __int64 *v303; // rdx
  __int64 *v304; // rax
  unsigned int v305; // eax
  int v306; // ebx
  _QWORD *v307; // rdi
  __int64 v308; // rdx
  unsigned __int64 v309; // rdx
  _QWORD *v310; // rax
  _QWORD *v311; // rdx
  _QWORD *v312; // rax
  _QWORD *v313; // rdx
  int v314; // edx
  __int64 v315; // [rsp+8h] [rbp-8758h]
  __int64 v316; // [rsp+8h] [rbp-8758h]
  unsigned __int64 v317; // [rsp+10h] [rbp-8750h]
  _BYTE *v318; // [rsp+18h] [rbp-8748h]
  __int64 v319; // [rsp+20h] [rbp-8740h]
  unsigned __int64 v320; // [rsp+20h] [rbp-8740h]
  __int64 n; // [rsp+28h] [rbp-8738h]
  __int64 v322; // [rsp+30h] [rbp-8730h]
  unsigned __int64 v323; // [rsp+30h] [rbp-8730h]
  __int64 v324; // [rsp+30h] [rbp-8730h]
  __int64 v325; // [rsp+38h] [rbp-8728h]
  __int64 v326; // [rsp+40h] [rbp-8720h]
  __int64 v327; // [rsp+40h] [rbp-8720h]
  unsigned __int64 v328; // [rsp+40h] [rbp-8720h]
  __int64 v329; // [rsp+48h] [rbp-8718h]
  __int64 v330; // [rsp+48h] [rbp-8718h]
  __int64 v331; // [rsp+48h] [rbp-8718h]
  __int64 *v332; // [rsp+50h] [rbp-8710h]
  __int64 *v333; // [rsp+50h] [rbp-8710h]
  _BYTE *v334; // [rsp+50h] [rbp-8710h]
  __int64 v335; // [rsp+50h] [rbp-8710h]
  __int64 v336; // [rsp+50h] [rbp-8710h]
  int v337; // [rsp+58h] [rbp-8708h]
  __int64 v338; // [rsp+60h] [rbp-8700h]
  __int64 v339; // [rsp+60h] [rbp-8700h]
  __int64 *v340; // [rsp+60h] [rbp-8700h]
  __int64 v341; // [rsp+60h] [rbp-8700h]
  unsigned __int8 v343; // [rsp+A8h] [rbp-86B8h]
  unsigned __int64 v346; // [rsp+C8h] [rbp-8698h] BYREF
  __m128i v347; // [rsp+D0h] [rbp-8690h] BYREF
  __int64 v348; // [rsp+E0h] [rbp-8680h]
  __int64 v349; // [rsp+E8h] [rbp-8678h]
  __m128i v350; // [rsp+F0h] [rbp-8670h] BYREF
  __int64 v351; // [rsp+100h] [rbp-8660h]
  __int64 v352; // [rsp+108h] [rbp-8658h]
  __int64 v353; // [rsp+110h] [rbp-8650h] BYREF
  __int64 v354; // [rsp+118h] [rbp-8648h]
  __int64 v355; // [rsp+120h] [rbp-8640h]
  __int64 v356; // [rsp+128h] [rbp-8638h]
  _BYTE *v357; // [rsp+130h] [rbp-8630h] BYREF
  __int64 v358; // [rsp+138h] [rbp-8628h]
  _BYTE v359[64]; // [rsp+140h] [rbp-8620h] BYREF
  unsigned __int64 v360[2]; // [rsp+180h] [rbp-85E0h] BYREF
  _BYTE v361[64]; // [rsp+190h] [rbp-85D0h] BYREF
  unsigned __int64 *v362; // [rsp+1D0h] [rbp-8590h] BYREF
  __int64 v363; // [rsp+1D8h] [rbp-8588h]
  unsigned __int64 v364; // [rsp+1E0h] [rbp-8580h] BYREF
  __int64 v365; // [rsp+1E8h] [rbp-8578h]
  int v366; // [rsp+1F0h] [rbp-8570h]
  _BYTE v367[360]; // [rsp+1F8h] [rbp-8568h] BYREF
  __int64 v368; // [rsp+360h] [rbp-8400h] BYREF
  __int64 v369; // [rsp+368h] [rbp-83F8h]
  unsigned __int64 v370; // [rsp+370h] [rbp-83F0h] BYREF
  __int64 v371; // [rsp+378h] [rbp-83E8h]
  _QWORD *v372; // [rsp+380h] [rbp-83E0h]
  __int64 v373; // [rsp+388h] [rbp-83D8h] BYREF
  unsigned int v374; // [rsp+390h] [rbp-83D0h]
  __int64 v375; // [rsp+398h] [rbp-83C8h] BYREF
  __int64 v376; // [rsp+3A0h] [rbp-83C0h]
  __int64 v377; // [rsp+3A8h] [rbp-83B8h]
  __int64 v378; // [rsp+3B0h] [rbp-83B0h]
  __int64 v379; // [rsp+3B8h] [rbp-83A8h] BYREF
  __int64 v380; // [rsp+3C0h] [rbp-83A0h]
  __int64 v381; // [rsp+3C8h] [rbp-8398h]
  __int64 v382; // [rsp+3D0h] [rbp-8390h]
  __int64 v383; // [rsp+3D8h] [rbp-8388h]
  __int64 v384; // [rsp+3E0h] [rbp-8380h]
  __int64 v385; // [rsp+3E8h] [rbp-8378h]
  int v386; // [rsp+3F0h] [rbp-8370h]
  __int64 v387; // [rsp+3F8h] [rbp-8368h]
  _BYTE *v388; // [rsp+400h] [rbp-8360h]
  _BYTE *v389; // [rsp+408h] [rbp-8358h]
  __int64 v390; // [rsp+410h] [rbp-8350h]
  int v391; // [rsp+418h] [rbp-8348h]
  _BYTE v392[16]; // [rsp+420h] [rbp-8340h] BYREF
  __int64 v393; // [rsp+430h] [rbp-8330h]
  __int64 v394; // [rsp+438h] [rbp-8328h]
  __int64 v395; // [rsp+440h] [rbp-8320h] BYREF
  _QWORD *v396; // [rsp+448h] [rbp-8318h]
  __int64 v397; // [rsp+450h] [rbp-8310h]
  __int64 v398; // [rsp+458h] [rbp-8308h]
  __int16 v399; // [rsp+460h] [rbp-8300h]
  __int64 v400[5]; // [rsp+468h] [rbp-82F8h] BYREF
  int v401; // [rsp+490h] [rbp-82D0h]
  __int64 v402; // [rsp+498h] [rbp-82C8h]
  __int64 v403; // [rsp+4A0h] [rbp-82C0h]
  __int64 v404; // [rsp+4A8h] [rbp-82B8h]
  _BYTE *v405; // [rsp+4B0h] [rbp-82B0h]
  __int64 v406; // [rsp+4B8h] [rbp-82A8h]
  _BYTE v407[64]; // [rsp+4C0h] [rbp-82A0h] BYREF
  __int64 v408; // [rsp+500h] [rbp-8260h] BYREF
  unsigned __int64 v409; // [rsp+508h] [rbp-8258h]
  char *v410; // [rsp+510h] [rbp-8250h]
  __int64 v411; // [rsp+518h] [rbp-8248h]
  __int64 v412; // [rsp+520h] [rbp-8240h]
  __int64 v413; // [rsp+528h] [rbp-8238h]
  unsigned int v414; // [rsp+530h] [rbp-8230h]
  __int64 v415; // [rsp+538h] [rbp-8228h]
  _QWORD *v416; // [rsp+540h] [rbp-8220h]
  __int64 v417; // [rsp+548h] [rbp-8218h]
  _QWORD v418[2]; // [rsp+550h] [rbp-8210h] BYREF
  __int64 v419; // [rsp+560h] [rbp-8200h]
  __int64 v420; // [rsp+568h] [rbp-81F8h]
  __int64 v421; // [rsp+570h] [rbp-81F0h]
  __int64 v422; // [rsp+578h] [rbp-81E8h]
  __int64 v423; // [rsp+580h] [rbp-81E0h]
  __int64 v424; // [rsp+588h] [rbp-81D8h]
  int v425; // [rsp+590h] [rbp-81D0h]
  __int64 v426; // [rsp+598h] [rbp-81C8h] BYREF
  __int64 *v427; // [rsp+5A0h] [rbp-81C0h]
  __int64 *v428; // [rsp+5A8h] [rbp-81B8h]
  __int64 v429; // [rsp+5B0h] [rbp-81B0h]
  __int64 v430; // [rsp+5B8h] [rbp-81A8h]
  _QWORD *v431; // [rsp+5C0h] [rbp-81A0h] BYREF
  __int64 v432; // [rsp+5C8h] [rbp-8198h]
  _QWORD v433[3]; // [rsp+5D0h] [rbp-8190h] BYREF
  __int64 v434; // [rsp+5E8h] [rbp-8178h]
  __int64 v435; // [rsp+5F0h] [rbp-8170h]
  __int64 v436; // [rsp+5F8h] [rbp-8168h]
  __int16 v437; // [rsp+600h] [rbp-8160h]
  __int64 v438; // [rsp+608h] [rbp-8158h] BYREF
  __int64 v439; // [rsp+610h] [rbp-8150h] BYREF
  __int64 v440; // [rsp+618h] [rbp-8148h]
  _QWORD v441[2]; // [rsp+620h] [rbp-8140h] BYREF
  int v442; // [rsp+630h] [rbp-8130h]
  __int64 v443; // [rsp+638h] [rbp-8128h]
  unsigned __int64 *v444; // [rsp+640h] [rbp-8120h] BYREF
  __int64 v445; // [rsp+648h] [rbp-8118h]
  unsigned __int64 v446[2]; // [rsp+650h] [rbp-8110h] BYREF
  _BYTE v447[16]; // [rsp+660h] [rbp-8100h] BYREF
  _BYTE *v448; // [rsp+670h] [rbp-80F0h]
  __int64 v449; // [rsp+678h] [rbp-80E8h]
  _BYTE v450[31744]; // [rsp+680h] [rbp-80E0h] BYREF
  __int64 v451; // [rsp+8280h] [rbp-4E0h]
  _QWORD *v452; // [rsp+8288h] [rbp-4D8h]
  __int64 v453; // [rsp+8290h] [rbp-4D0h]
  unsigned int v454; // [rsp+8298h] [rbp-4C8h]
  __int64 *v455; // [rsp+82A0h] [rbp-4C0h]
  __int64 v456; // [rsp+82A8h] [rbp-4B8h]
  _BYTE v457[128]; // [rsp+82B0h] [rbp-4B0h] BYREF
  unsigned __int64 *v458; // [rsp+8330h] [rbp-430h]
  __int64 v459; // [rsp+8338h] [rbp-428h]
  _BYTE v460[384]; // [rsp+8340h] [rbp-420h] BYREF
  __int64 v461; // [rsp+84C0h] [rbp-2A0h]
  _BYTE *v462; // [rsp+84C8h] [rbp-298h]
  _BYTE *v463; // [rsp+84D0h] [rbp-290h]
  __int64 v464; // [rsp+84D8h] [rbp-288h]
  int v465; // [rsp+84E0h] [rbp-280h]
  _BYTE v466[64]; // [rsp+84E8h] [rbp-278h] BYREF
  __int64 v467; // [rsp+8528h] [rbp-238h]
  __int64 v468; // [rsp+8530h] [rbp-230h] BYREF
  __int64 v469; // [rsp+8538h] [rbp-228h]
  __int64 v470; // [rsp+8540h] [rbp-220h]
  __int64 v471; // [rsp+8548h] [rbp-218h]
  __int64 v472; // [rsp+8550h] [rbp-210h] BYREF
  __int64 v473; // [rsp+8558h] [rbp-208h]
  __int64 v474; // [rsp+8560h] [rbp-200h]
  unsigned int v475; // [rsp+8568h] [rbp-1F8h]
  int v476; // [rsp+8570h] [rbp-1F0h]
  __int64 *v477; // [rsp+8578h] [rbp-1E8h]
  __int64 v478; // [rsp+8580h] [rbp-1E0h]
  _BYTE v479[384]; // [rsp+8588h] [rbp-1D8h] BYREF
  __int64 v480; // [rsp+8708h] [rbp-58h]
  __int64 v481; // [rsp+8710h] [rbp-50h]
  __int64 v482; // [rsp+8718h] [rbp-48h]
  int v483; // [rsp+8720h] [rbp-40h]

  *(_DWORD *)(a5 + 16) = 0;
  if ( byte_4FB1A40 )
  {
    v17 = sub_1481F60((_QWORD *)a5, a3, a7, a8);
    v18 = sub_14562D0(v17);
    if ( v17 )
    {
      if ( v18 )
      {
        v19 = (_QWORD *)a3;
        v20 = 0;
        do
        {
          v19 = (_QWORD *)*v19;
          ++v20;
        }
        while ( v19 );
        if ( v20 <= 2 )
        {
          if ( !(unsigned __int8)sub_1BF9E60(a5, a3) )
            return 0;
          v52 = **(_QWORD **)(a3 + 32);
          v53 = *(_QWORD *)(v52 + 56);
          if ( *(_BYTE *)(a2 + 124) && v53 == *(_QWORD *)(a2 + 112) )
          {
            v54 = *(_DWORD *)(a2 + 120);
          }
          else
          {
            sub_1C007A0(a2, *(_QWORD *)(v52 + 56), 1, 1, 1, 0);
            *(_BYTE *)(a2 + 124) = 1;
            *(_QWORD *)(a2 + 112) = v53;
            v54 = sub_1C016B0(a2, v53);
            *(_DWORD *)(a2 + 120) = v54;
          }
          if ( v54 > 0x1E )
            return 0;
        }
      }
    }
  }
  if ( (unsigned int)sub_1998920(a3) > dword_4FB1420 )
    return 0;
  if ( byte_4FB1960 )
  {
    v50 = *(_QWORD *)(**(_QWORD **)(a3 + 32) + 56LL);
    v51 = sub_1BFBA30(a1, v50, 0);
    if ( v51 )
    {
      if ( v51 < (unsigned int)sub_1BF8310(**(_QWORD **)(a3 + 32), 0, a15) )
        return 0;
      if ( !*(_BYTE *)(a2 + 124) || v50 != *(_QWORD *)(a2 + 112) )
      {
        sub_1C007A0(a2, v50, 1, 1, 1, 0);
        *(_BYTE *)(a2 + 124) = 1;
        *(_QWORD *)(a2 + 112) = v50;
        *(_DWORD *)(a2 + 120) = sub_1C016B0(a2, v50);
      }
      v127 = *(_QWORD **)(a3 + 32);
      if ( v127 != *(_QWORD **)(a3 + 40) )
      {
        v128 = 0;
        v129 = *(_QWORD **)(a3 + 40);
        do
        {
          v130 = sub_1C014D0(a2, *v127, 0);
          if ( v128 < v130 )
            v128 = v130;
          ++v127;
        }
        while ( v129 != v127 );
        if ( dword_4FB1880 < v128 && 2 * v128 > v51 )
          return 0;
      }
    }
  }
  if ( byte_4FB1340 && *(_QWORD *)a3 )
  {
    *(_QWORD *)(a5 + 8) = a3;
    sub_145FF90(a5 + 144);
  }
  else
  {
    *(_QWORD *)(a5 + 8) = 0;
  }
  v408 = a4;
  v409 = a5;
  v415 = 0;
  v410 = a6;
  LODWORD(v426) = 0;
  v411 = a15;
  v427 = 0;
  v412 = a16;
  v428 = &v426;
  v413 = a3;
  LOWORD(v414) = 256;
  v429 = (__int64)&v426;
  v430 = 0;
  v431 = v433;
  v439 = 0;
  v440 = 1;
  v416 = v418;
  v417 = 0x800000000LL;
  v432 = 0x800000000LL;
  v21 = v441;
  do
    *v21++ = -8;
  while ( v21 != (unsigned __int64 *)&v444 );
  v451 = 0;
  v444 = v446;
  v445 = 0x400000000LL;
  v455 = (__int64 *)v457;
  v458 = (unsigned __int64 *)v460;
  v448 = v450;
  v459 = 0x800000000LL;
  v449 = 0x1000000000LL;
  v452 = 0;
  v453 = 0;
  v454 = 0;
  v456 = 0x1000000000LL;
  v461 = 0;
  v462 = v466;
  v463 = v466;
  v464 = 8;
  v465 = 0;
  v467 = a1;
  v468 = 0;
  v469 = 0;
  v470 = 0;
  v471 = 0;
  v472 = 0;
  v473 = 0;
  v474 = 0;
  v475 = 0;
  v477 = (__int64 *)v479;
  v478 = 0x1000000000LL;
  v480 = 0;
  v481 = 0;
  v482 = 0;
  v483 = 0;
  if ( !(unsigned __int8)sub_13FCBF0(a3) )
    goto LABEL_30;
  v338 = a4 + 208;
  if ( a4 + 208 == (*(_QWORD *)(a4 + 208) & 0xFFFFFFFFFFFFFFF8LL) )
    goto LABEL_30;
  v22 = *(_QWORD *)(a4 + 216);
  v23 = 201;
  if ( a4 + 208 != v22 )
  {
    do
    {
      v24 = v22 - 32;
      if ( !v22 )
        v24 = 0;
      if ( !--v23 )
        goto LABEL_30;
      v25 = *(_QWORD *)(v24 + 24);
      if ( *(_BYTE *)(v25 + 16) == 77 )
      {
        v26 = *(unsigned __int8 *)(sub_157ED20(*(_QWORD *)(v25 + 40)) + 16);
        if ( (unsigned int)(v26 - 73) <= 1 || (_BYTE)v26 == 34 )
        {
          v27 = *(_DWORD *)(v25 + 20) & 0xFFFFFFF;
          v28 = (*(_BYTE *)(v25 + 23) & 0x40) != 0 ? *(_QWORD *)(v25 - 8) : v25 - 24 * v27;
          v29 = (__int64 *)(v28 + 24LL * *(unsigned int *)(v25 + 56) + 8);
          for ( i = &v29[v27]; i != v29; ++v29 )
          {
            if ( *(_BYTE *)(sub_157ED20(*v29) + 16) == 34 )
              goto LABEL_30;
          }
        }
      }
      v22 = *(_QWORD *)(v22 + 8);
    }
    while ( v338 != v22 );
  }
  v65 = 0;
  v66 = sub_157EB90(**(_QWORD **)(a3 + 32));
  v329 = sub_1632FA0(v66);
  *(_DWORD *)(a5 + 16) = 0;
  v67 = **(_QWORD **)(a3 + 32);
  v68 = *(_QWORD *)(v67 + 48);
  v69 = v67 + 40;
  if ( v68 != v67 + 40 )
  {
    do
    {
      if ( !v68 )
        BUG();
      if ( *(_BYTE *)(v68 - 8) != 77 )
        break;
      v70 = sub_127FA20(v329, *(_QWORD *)(v68 - 24));
      v68 = *(_QWORD *)(v68 + 8);
      v65 += (v70 > 0x20) + 1;
    }
    while ( v69 != v68 );
    v65 *= 2;
    v67 = **(_QWORD **)(a3 + 32);
  }
  v71 = sub_1BFBA30(a1, *(_QWORD *)(v67 + 56), 0);
  if ( v71 <= 0 )
    v71 = 60;
  v74 = v65 - v71;
  BYTE1(v414) = v74 > 0;
  v476 = v74;
  sub_1997F10(&v408, a7, a8, a9, a10, v72, v73, a13, a14);
  sub_19996C0(&v408, a7, a8, a9, a10, v75, v76, a13, a14);
  if ( v338 == (*(_QWORD *)(a4 + 208) & 0xFFFFFFFFFFFFFFF8LL) )
    goto LABEL_30;
  v77 = *(_QWORD *)(a3 + 8);
  if ( *(_QWORD *)(a3 + 16) != v77 )
    goto LABEL_30;
  sub_199BED0(&v408);
  sub_19B0310((__int64)&v408, a7, a8);
  sub_19A4400((__int64)&v408, a7, a8);
  ++v472;
  if ( !(_DWORD)v474 )
  {
    if ( !HIDWORD(v474) )
      goto LABEL_141;
    v82 = v475;
    if ( v475 <= 0x40 )
      goto LABEL_138;
    j___libc_free_0(v473);
    v475 = 0;
LABEL_530:
    v473 = 0;
LABEL_140:
    v474 = 0;
    goto LABEL_141;
  }
  v79 = (unsigned int)(4 * v474);
  v77 = 64;
  v82 = v475;
  if ( (unsigned int)v79 < 0x40 )
    v79 = 64;
  if ( (unsigned int)v79 >= v475 )
  {
LABEL_138:
    v83 = (_QWORD *)v473;
    v78 = v473 + 16 * v82;
    if ( v473 != v78 )
    {
      do
      {
        *v83 = -8;
        v83 += 2;
      }
      while ( (_QWORD *)v78 != v83 );
    }
    goto LABEL_140;
  }
  if ( (_DWORD)v474 == 1 )
  {
    v152 = 64;
  }
  else
  {
    _BitScanReverse(&v151, v474 - 1);
    v152 = 1 << (33 - (v151 ^ 0x1F));
    if ( v152 < 64 )
      v152 = 64;
    if ( v475 == v152 )
      goto LABEL_266;
  }
  j___libc_free_0(v473);
  v153 = sub_19951E0(v152);
  v475 = v153;
  if ( !v153 )
    goto LABEL_530;
  v473 = sub_22077B0(16LL * v153);
LABEL_266:
  sub_19A5F10((__int64)&v472);
LABEL_141:
  sub_19A4EB0((__int64)&v408, a7, a8, v77, v78, v79, v80, v81);
  if ( !(_DWORD)v449 )
    goto LABEL_30;
  sub_19A87A0((__int64)&v408, a7, a8, v77, v84, v85, v86, v87);
  sub_19A6280((__int64)&v408);
  sub_19A09A0((__int64)&v408, v77, v88, v89, v90, v91);
  sub_19A0DF0((__int64)&v408, v77, v92, v93, v94);
  if ( (unsigned __int64)sub_1992970((__int64)&v408) > 0xFFFE )
    sub_19A6280((__int64)&v408);
  if ( byte_4FB1C00 )
    sub_19AC5F0((__int64)&v408);
  if ( byte_4FB1CE0 )
    sub_19AD670((__int64)&v408, v77, v95, v96, v97, v98);
  else
    sub_19971B0((__int64)&v408);
  if ( (unsigned __int64)sub_1992970((__int64)&v408) <= 0x3FFFB )
  {
LABEL_151:
    if ( !byte_4FB17A0 )
      goto LABEL_152;
    ++v468;
    if ( !(_DWORD)v470 )
    {
      if ( !HIDWORD(v470) )
        goto LABEL_501;
      v272 = (unsigned int)v471;
      if ( (unsigned int)v471 <= 0x40 )
        goto LABEL_494;
      j___libc_free_0(v469);
      LODWORD(v471) = 0;
LABEL_499:
      v469 = 0;
LABEL_500:
      v470 = 0;
      goto LABEL_501;
    }
    v283 = 4 * v470;
    v272 = (unsigned int)v471;
    if ( (unsigned int)(4 * v470) < 0x40 )
      v283 = 64;
    if ( (unsigned int)v471 <= v283 )
    {
LABEL_494:
      v273 = (_QWORD *)v469;
      v274 = (_QWORD *)(v469 + 8 * v272);
      if ( (_QWORD *)v469 != v274 )
      {
        do
          *v273++ = -8;
        while ( v274 != v273 );
      }
      goto LABEL_500;
    }
    if ( (_DWORD)v470 == 1 )
    {
      v285 = 64;
    }
    else
    {
      _BitScanReverse(&v284, v470 - 1);
      v285 = 1 << (33 - (v284 ^ 0x1F));
      if ( v285 < 64 )
        v285 = 64;
      if ( v285 == (_DWORD)v471 )
        goto LABEL_527;
    }
    j___libc_free_0(v469);
    v286 = sub_19951E0(v285);
    LODWORD(v471) = v286;
    if ( !v286 )
      goto LABEL_499;
    v469 = sub_22077B0(8LL * v286);
LABEL_527:
    sub_19A5F50((__int64)&v468);
LABEL_501:
    v275 = v455;
    v276 = &v455[(unsigned int)v456];
    if ( v455 == v276 )
    {
LABEL_152:
      v347.m128i_i64[0] = -1;
      v357 = v359;
      v358 = 0x800000000LL;
      v360[1] = 0x800000000LL;
      v363 = (__int64)v367;
      v364 = (unsigned __int64)v367;
      v101 = (unsigned int)v449;
      v360[0] = (unsigned __int64)v361;
      v347.m128i_i64[1] = -1;
      v348 = -1;
      v349 = -1;
      v350 = 0u;
      v351 = 0;
      v352 = 0;
      v362 = 0;
      v365 = 16;
      v366 = 0;
      v353 = 0;
      v354 = 0;
      v355 = 0;
      v356 = 0;
      if ( (unsigned int)v449 > 8 )
      {
        sub_16CD150((__int64)v360, v361, (unsigned int)v449, 8, v99, v100);
        v101 = (unsigned int)v449;
      }
      v368 = 0;
      v369 = (__int64)&v373;
      v370 = (unsigned __int64)&v373;
      v371 = 16;
      LODWORD(v372) = 0;
      if ( (_DWORD)v101 )
      {
        v102 = 0;
        v339 = 1984 * v101;
        do
        {
          v103 = *(_DWORD *)&v448[v102 + 752];
          if ( v103 != 1 )
            goto LABEL_156;
          v104 = *(_QWORD *)&v448[v102 + 744];
          v105 = *(_QWORD *)(v104 + 80);
          if ( (_DWORD)v470 )
          {
            if ( !v105 )
            {
              v278 = *(unsigned int *)(v104 + 40);
              v139 = *(__int64 **)(v104 + 32);
              v279 = (__int64)&v139[v278];
              v335 = v278;
              v280 = sub_19965D0(v139, v279, (__int64)&v468);
              v140 = v335;
              if ( (_QWORD *)v279 != v280 )
                goto LABEL_162;
              goto LABEL_231;
            }
            if ( (_DWORD)v471 )
            {
              v106 = (v471 - 1) & (((unsigned int)v105 >> 9) ^ ((unsigned int)v105 >> 4));
              v107 = *(_QWORD *)(v469 + 8LL * v106);
              if ( v105 == v107 )
                goto LABEL_162;
              while ( v107 != -8 )
              {
                v106 = (v471 - 1) & (v103 + v106);
                v107 = *(_QWORD *)(v469 + 8LL * v106);
                if ( v105 == v107 )
                  goto LABEL_162;
                ++v103;
              }
            }
            v281 = *(_QWORD **)(v104 + 32);
            v331 = *(_QWORD *)(v104 + 80);
            v336 = (__int64)&v281[*(unsigned int *)(v104 + 40)];
            v282 = sub_19965D0(v281, v336, (__int64)&v468);
            v105 = v331;
            if ( (_QWORD *)v336 != v282 )
              goto LABEL_162;
          }
          else if ( !v105 )
          {
            goto LABEL_230;
          }
          v138 = (__int64 *)v363;
          if ( v364 != v363 )
            goto LABEL_229;
          v154 = (__int64 *)(v363 + 8LL * HIDWORD(v365));
          if ( (__int64 *)v363 != v154 )
          {
            v155 = 0;
            while ( v105 != *v138 )
            {
              if ( *v138 == -2 )
                v155 = v138;
              if ( v154 == ++v138 )
              {
                if ( !v155 )
                  goto LABEL_622;
                *v155 = v105;
                --v366;
                v362 = (unsigned __int64 *)((char *)v362 + 1);
                v139 = *(__int64 **)(v104 + 32);
                v140 = *(unsigned int *)(v104 + 40);
                goto LABEL_231;
              }
            }
            goto LABEL_230;
          }
LABEL_622:
          if ( HIDWORD(v365) >= (unsigned int)v365 )
          {
LABEL_229:
            sub_16CCBA0((__int64)&v362, v105);
LABEL_230:
            v139 = *(__int64 **)(v104 + 32);
            v140 = *(unsigned int *)(v104 + 40);
            goto LABEL_231;
          }
          ++HIDWORD(v365);
          *v154 = v105;
          v362 = (unsigned __int64 *)((char *)v362 + 1);
          v139 = *(__int64 **)(v104 + 32);
          v140 = *(unsigned int *)(v104 + 40);
LABEL_231:
          v141 = &v139[v140];
          if ( v139 != v141 )
          {
            v142 = v364;
            v143 = (__int64 *)v363;
            do
            {
              v144 = *v139;
              if ( v143 != (__int64 *)v142 )
                goto LABEL_233;
              v145 = &v143[HIDWORD(v365)];
              if ( v143 != v145 )
              {
                v146 = v143;
                v147 = 0;
                while ( v144 != *v146 )
                {
                  if ( *v146 == -2 )
                    v147 = v146;
                  if ( v145 == ++v146 )
                  {
                    if ( !v147 )
                      goto LABEL_244;
                    *v147 = v144;
                    v142 = v364;
                    --v366;
                    v143 = (__int64 *)v363;
                    v362 = (unsigned __int64 *)((char *)v362 + 1);
                    goto LABEL_234;
                  }
                }
                goto LABEL_234;
              }
LABEL_244:
              if ( HIDWORD(v365) < (unsigned int)v365 )
              {
                ++HIDWORD(v365);
                *v145 = v144;
                v143 = (__int64 *)v363;
                v362 = (unsigned __int64 *)((char *)v362 + 1);
                v142 = v364;
              }
              else
              {
LABEL_233:
                sub_16CCBA0((__int64)&v362, v144);
                v142 = v364;
                v143 = (__int64 *)v363;
              }
LABEL_234:
              ++v139;
            }
            while ( v141 != v139 );
          }
LABEL_156:
          v102 += 1984;
        }
        while ( v102 != v339 );
      }
      v346 = 0;
      sub_19B0E20(
        (__int64)&v408,
        (__int64)&v357,
        &v347,
        (__int64)v360,
        &v350,
        (__int64)&v362,
        (__int64)&v368,
        &v346,
        (__int64)&v353);
LABEL_162:
      if ( v370 != v369 )
        _libc_free(v370);
      j___libc_free_0(v354);
      if ( v364 != v363 )
        _libc_free(v364);
      if ( (_BYTE *)v360[0] != v361 )
        _libc_free(v360[0]);
      LODWORD(v417) = 0;
      sub_1994270((__int64)v427);
      v427 = 0;
      v430 = 0;
      v428 = &v426;
      v429 = (__int64)&v426;
      LODWORD(v432) = 0;
      sub_19A59E0((__int64)&v439);
      v108 = v453;
      ++v451;
      LODWORD(v445) = 0;
      if ( v453 )
      {
        v109 = v452;
        v110 = 4 * v453;
        v111 = &v452[2 * v454];
        if ( (unsigned int)(4 * v453) < 0x40 )
          v110 = 64;
        if ( v454 <= v110 )
        {
          while ( v111 != v109 )
          {
            if ( *v109 != -8 )
            {
              if ( *v109 != -16 )
              {
                v112 = (unsigned __int64 *)v109[1];
                if ( ((unsigned __int8)v112 & 1) == 0 )
                {
                  if ( v112 )
                  {
                    _libc_free(*v112);
                    j_j___libc_free_0(v112, 24);
                  }
                }
              }
              *v109 = -8;
            }
            v109 += 2;
          }
        }
        else
        {
          do
          {
            if ( *v109 != -16 && *v109 != -8 )
            {
              v131 = (unsigned __int64 *)v109[1];
              if ( ((unsigned __int8)v131 & 1) == 0 )
              {
                if ( v131 )
                {
                  _libc_free(*v131);
                  j_j___libc_free_0(v131, 24);
                }
              }
            }
            v109 += 2;
          }
          while ( v111 != v109 );
          if ( v108 )
          {
            if ( v108 == 1 )
            {
              v133 = 64;
            }
            else
            {
              _BitScanReverse(&v132, v108 - 1);
              v133 = 1 << (33 - (v132 ^ 0x1F));
              if ( v133 < 64 )
                v133 = 64;
            }
            v134 = v452;
            if ( v454 == v133 )
            {
              v453 = 0;
              v293 = &v452[2 * v454];
              do
              {
                if ( v134 )
                  *v134 = -8;
                v134 += 2;
              }
              while ( v293 != v134 );
            }
            else
            {
              j___libc_free_0(v452);
              v135 = ((((((((4 * v133 / 3u + 1) | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 2)
                        | (4 * v133 / 3u + 1)
                        | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 4)
                      | (((4 * v133 / 3u + 1) | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 2)
                      | (4 * v133 / 3u + 1)
                      | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 8)
                    | (((((4 * v133 / 3u + 1) | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 2)
                      | (4 * v133 / 3u + 1)
                      | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 4)
                    | (((4 * v133 / 3u + 1) | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 2)
                    | (4 * v133 / 3u + 1)
                    | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 16;
              v454 = (v135
                    | (((((((4 * v133 / 3u + 1) | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 2)
                        | (4 * v133 / 3u + 1)
                        | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 4)
                      | (((4 * v133 / 3u + 1) | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 2)
                      | (4 * v133 / 3u + 1)
                      | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 8)
                    | (((((4 * v133 / 3u + 1) | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 2)
                      | (4 * v133 / 3u + 1)
                      | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 4)
                    | (((4 * v133 / 3u + 1) | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 2)
                    | (4 * v133 / 3u + 1)
                    | ((4 * v133 / 3u + 1) >> 1))
                   + 1;
              v136 = (_QWORD *)sub_22077B0(
                                 16
                               * ((v135
                                 | (((((((4 * v133 / 3u + 1) | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 2)
                                     | (4 * v133 / 3u + 1)
                                     | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 4)
                                   | (((4 * v133 / 3u + 1) | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 2)
                                   | (4 * v133 / 3u + 1)
                                   | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 8)
                                 | (((((4 * v133 / 3u + 1) | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 2)
                                   | (4 * v133 / 3u + 1)
                                   | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 4)
                                 | (((4 * v133 / 3u + 1) | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1)) >> 2)
                                 | (4 * v133 / 3u + 1)
                                 | ((unsigned __int64)(4 * v133 / 3u + 1) >> 1))
                                + 1));
              v453 = 0;
              v452 = v136;
              v137 = &v136[2 * v454];
              while ( v137 != v136 )
              {
                if ( v136 )
                  *v136 = -8;
                v136 += 2;
              }
            }
            goto LABEL_182;
          }
          if ( v454 )
          {
            j___libc_free_0(v452);
            v452 = 0;
            v453 = 0;
            v454 = 0;
            goto LABEL_182;
          }
        }
        v453 = 0;
      }
LABEL_182:
      LODWORD(v456) = 0;
      if ( !(_DWORD)v358 )
        goto LABEL_254;
      v362 = &v364;
      v363 = 0x1000000000LL;
      v113 = sub_157EB90(**(_QWORD **)(v413 + 32));
      v114 = sub_1632FA0(v113);
      v370 = (unsigned __int64)"lsr";
      v388 = v392;
      v368 = v409;
      v389 = v392;
      v399 = 1;
      v369 = v114;
      v371 = 0;
      v372 = 0;
      v373 = 0;
      v374 = 0;
      v375 = 0;
      v376 = 0;
      v377 = 0;
      v378 = 0;
      v379 = 0;
      v380 = 0;
      v381 = 0;
      v382 = 0;
      v383 = 0;
      v384 = 0;
      v385 = 0;
      v386 = 0;
      v387 = 0;
      v390 = 2;
      v391 = 0;
      v393 = 0;
      v394 = 0;
      v395 = 0;
      v396 = 0;
      v397 = 0;
      v398 = 0;
      v115 = sub_15E0530(*(_QWORD *)(v409 + 24));
      v116 = v458;
      memset(v400, 0, 24);
      v400[3] = v115;
      v405 = v407;
      v406 = 0x800000000LL;
      v393 = v413;
      v400[4] = 0;
      v394 = v415;
      v401 = 0;
      v402 = 0;
      v404 = v114;
      v117 = &v458[6 * (unsigned int)v459];
      v399 = 256;
      v403 = 0;
      if ( v458 == v117 )
      {
LABEL_276:
        v319 = (unsigned int)v449;
        if ( (_DWORD)v449 )
        {
          v326 = 0;
          do
          {
            v156 = &v448[1984 * v326];
            v157 = (__int64 *)*((_QWORD *)v156 + 7);
            v332 = &v157[10 * *((unsigned int *)v156 + 16)];
            if ( v157 != v332 )
            {
              v158 = (__int64 *)*((_QWORD *)v156 + 7);
              for ( j = (__int64)&v448[1984 * v326]; ; j = (__int64)&v448[1984 * v326] )
              {
                v163 = *v158;
                v164 = *(_QWORD *)&v357[8 * v326];
                if ( *(_BYTE *)(*v158 + 16) == 77 )
                {
                  sub_19AF1A0(&v408, v163, j, (__int64)v158, v164, (__int64)&v368, a7, a8, (__int64)&v362);
                }
                else
                {
                  v165 = sub_19A2ED0(
                           (__int64)&v408,
                           j,
                           (__int64)v158,
                           v164,
                           v163 + 24,
                           (__int64)&v368,
                           a7,
                           a8,
                           (__int64)&v362);
                  v166 = (__int64 *)v158[1];
                  v167 = *v166;
                  if ( *v166 != *(_QWORD *)v165 )
                  {
                    v322 = *v166;
                    v315 = *v158;
                    v361[1] = 1;
                    v360[0] = (unsigned __int64)"tmp";
                    v361[0] = 3;
                    v168 = sub_15FBEB0((_QWORD *)v165, 0, v167, 0);
                    v165 = sub_15FDBD0(v168, v165, v322, (__int64)v360, v315);
                  }
                  v169 = *v158;
                  if ( *(_DWORD *)(j + 32) == 3 )
                  {
                    if ( (*(_BYTE *)(v169 + 23) & 0x40) != 0 )
                      v170 = *(__int64 **)(v169 - 8);
                    else
                      v170 = (__int64 *)(v169 - 24LL * (*(_DWORD *)(v169 + 20) & 0xFFFFFFF));
                    if ( *v170 )
                    {
                      v171 = v170[1];
                      v172 = v170[2] & 0xFFFFFFFFFFFFFFFCLL;
                      *(_QWORD *)v172 = v171;
                      if ( v171 )
                        *(_QWORD *)(v171 + 16) = v172 | *(_QWORD *)(v171 + 16) & 3LL;
                    }
                    *v170 = v165;
                    if ( v165 )
                    {
                      v173 = *(_QWORD *)(v165 + 8);
                      v170[1] = v173;
                      if ( v173 )
                        *(_QWORD *)(v173 + 16) = (unsigned __int64)(v170 + 1) | *(_QWORD *)(v173 + 16) & 3LL;
                      v170[2] = (v165 + 8) | v170[2] & 3;
                      *(_QWORD *)(v165 + 8) = v170;
                    }
                    if ( *(_BYTE *)(v165 + 16) == 77 && byte_4FB1B20 )
                    {
                      v174 = (unsigned int)v478;
                      if ( (unsigned int)v478 >= HIDWORD(v478) )
                      {
                        v298 = (((HIDWORD(v478) + 2LL) | (((unsigned __int64)HIDWORD(v478) + 2) >> 1)) >> 2)
                             | (HIDWORD(v478) + 2LL)
                             | (((unsigned __int64)HIDWORD(v478) + 2) >> 1);
                        v299 = (v298 >> 4) | v298;
                        v300 = ((v299 >> 8) | v299 | (((v299 >> 8) | v299) >> 16) | (((v299 >> 8) | v299) >> 32)) + 1;
                        if ( v300 > 0xFFFFFFFF )
                          v300 = 0xFFFFFFFFLL;
                        v324 = malloc(24 * v300);
                        if ( !v324 )
                          sub_16BD1C0("Allocation failed", 1u);
                        v301 = v477;
                        v302 = (_QWORD *)v324;
                        v303 = v477;
                        v304 = &v477[3 * (unsigned int)v478];
                        if ( v477 != v304 )
                        {
                          do
                          {
                            if ( v302 )
                            {
                              *v302 = *v303;
                              v302[1] = v303[1];
                              v302[2] = v303[2];
                            }
                            v303 += 3;
                            v302 += 3;
                          }
                          while ( v304 != v303 );
                        }
                        if ( v301 != (__int64 *)v479 )
                          _libc_free((unsigned __int64)v301);
                        HIDWORD(v478) = v300;
                        v174 = (unsigned int)v478;
                        v477 = (__int64 *)v324;
                      }
                      v175 = &v477[3 * v174];
                      if ( v175 )
                      {
                        *v175 = v165;
                        v175[1] = v164;
                        v175[2] = (__int64)v158;
                      }
                      LODWORD(v478) = v478 + 1;
                    }
                  }
                  else
                  {
                    sub_1648780(v169, v158[1], v165);
                  }
                }
                v160 = v363;
                if ( (unsigned int)v363 >= HIDWORD(v363) )
                {
                  sub_170B450((__int64)&v362, 0);
                  v160 = v363;
                }
                v161 = &v362[3 * v160];
                if ( v161 )
                {
                  v162 = v158[1];
                  *v161 = 6;
                  v161[1] = 0;
                  v161[2] = v162;
                  if ( v162 != 0 && v162 != -8 && v162 != -16 )
                    sub_164C220((__int64)v161);
                  v160 = v363;
                }
                LOBYTE(v414) = 1;
                v158 += 10;
                LODWORD(v363) = v160 + 1;
                if ( v332 == v158 )
                  break;
              }
            }
            ++v326;
          }
          while ( v319 != v326 );
        }
        v333 = &v477[3 * (unsigned int)v478];
        if ( v477 != v333 )
        {
          v176 = v477;
          while ( 1 )
          {
            v177 = v413;
            v178 = (__int64)v410;
            v179 = *v176;
            v330 = v176[1];
            v340 = (__int64 *)v176[2];
            if ( sub_13FC520(v413) )
            {
              if ( sub_13F9E70(v177) )
              {
                if ( (*(_DWORD *)(v179 + 20) & 0xFFFFFFF) == 2 && (unsigned int)sub_1648EF0(v179) == 2 )
                {
                  v180 = sub_13FCB50(v177);
                  if ( sub_15CC8F0(v178, *(_QWORD *)(*v340 + 40), v180) )
                  {
                    v181 = (*(_BYTE *)(v179 + 23) & 0x40) != 0
                         ? *(_QWORD *)(v179 - 8)
                         : v179 - 24LL * (*(_DWORD *)(v179 + 20) & 0xFFFFFFF);
                    v182 = *(_QWORD *)(v181 + 24LL * *(unsigned int *)(v179 + 56) + 8);
                    if ( v182 == sub_13FC520(v177) )
                    {
                      v297 = (*(_BYTE *)(v179 + 23) & 0x40) != 0
                           ? *(__int64 **)(v179 - 8)
                           : (__int64 *)(v179 - 24LL * (*(_DWORD *)(v179 + 20) & 0xFFFFFFF));
                      v337 = 0;
                      v184 = *v297;
                      v185 = v297[3];
                    }
                    else
                    {
                      v183 = (*(_BYTE *)(v179 + 23) & 0x40) != 0
                           ? *(__int64 **)(v179 - 8)
                           : (__int64 *)(v179 - 24LL * (*(_DWORD *)(v179 + 20) & 0xFFFFFFF));
                      v337 = 1;
                      v184 = v183[3];
                      v185 = *v183;
                    }
                    if ( *(_BYTE *)(v184 + 16) == 13 && *(_BYTE *)(v185 + 16) > 0x17u )
                    {
                      v327 = *(_QWORD *)(v185 + 40);
                      if ( v327 != sub_13F9E70(v177) && *(_BYTE *)(v185 + 16) == 35 )
                      {
                        v186 = *(_QWORD *)(v185 + 8);
                        if ( v186 )
                        {
                          if ( !*(_QWORD *)(v186 + 8) )
                          {
                            v187 = (__int64 *)sub_13CF970(v185);
                            v188 = v187[3];
                            if ( v179 == *v187 )
                              goto LABEL_334;
                            if ( v179 == v188 )
                              break;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
LABEL_313:
            v176 += 3;
            if ( v333 == v176 )
              goto LABEL_363;
          }
          v188 = *v187;
LABEL_334:
          if ( *(_BYTE *)(v188 + 16) == 13 )
          {
            v189 = *(_DWORD *)(v184 + 32);
            v190 = *(__int64 **)(v184 + 24);
            if ( v189 > 0x40 )
              v191 = *v190;
            else
              v191 = (__int64)((_QWORD)v190 << (64 - (unsigned __int8)v189)) >> (64 - (unsigned __int8)v189);
            v192 = *(_DWORD *)(v188 + 32);
            v193 = *(__int64 **)(v188 + 24);
            if ( v192 > 0x40 )
              v194 = *v193;
            else
              v194 = (__int64)((_QWORD)v193 << (64 - (unsigned __int8)v192)) >> (64 - (unsigned __int8)v192);
            v195 = sub_159C470(*(_QWORD *)v184, v191 - v194, 0);
            v196 = *(_DWORD *)(v195 + 32);
            v197 = 1LL << ((unsigned __int8)v196 - 1);
            if ( v196 > 0x40 )
              v198 = *(_QWORD *)(*(_QWORD *)(v195 + 24) + 8LL * ((v196 - 1) >> 6));
            else
              v198 = *(_QWORD *)(v195 + 24);
            if ( (v198 & v197) != 0 )
            {
              v199 = *(_QWORD *)(v330 + 32);
              v200 = v199 + 8LL * *(unsigned int *)(v330 + 40);
              while ( v199 != v200 )
              {
                if ( *(_WORD *)(*(_QWORD *)v199 + 24LL) == 7 && (*(_BYTE *)(*(_QWORD *)v199 + 26LL) & 2) != 0 )
                  goto LABEL_313;
                v199 += 8;
              }
            }
            if ( (*(_BYTE *)(v179 + 23) & 0x40) != 0 )
              v201 = *(_QWORD *)(v179 - 8);
            else
              v201 = v179 - 24LL * (*(_DWORD *)(v179 + 20) & 0xFFFFFFF);
            v202 = (__int64 *)(24LL * v337 + v201);
            if ( *v202 )
            {
              v203 = v202[1];
              v204 = v202[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v204 = v203;
              if ( v203 )
                *(_QWORD *)(v203 + 16) = v204 | *(_QWORD *)(v203 + 16) & 3LL;
            }
            *v202 = v195;
            v205 = *(_QWORD *)(v195 + 8);
            v202[1] = v205;
            if ( v205 )
              *(_QWORD *)(v205 + 16) = (unsigned __int64)(v202 + 1) | *(_QWORD *)(v205 + 16) & 3LL;
            v202[2] = v202[2] & 3 | (v195 + 8);
            *(_QWORD *)(v195 + 8) = v202;
            sub_15F22F0((_QWORD *)v185, *v340);
            v206 = *v340;
            if ( (*(_BYTE *)(*v340 + 23) & 0x40) != 0 )
              v207 = *(__int64 **)(v206 - 8);
            else
              v207 = (__int64 *)(v206 - 24LL * (*(_DWORD *)(v206 + 20) & 0xFFFFFFF));
            if ( *v207 )
            {
              v208 = v207[1];
              v209 = v207[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v209 = v208;
              if ( v208 )
                *(_QWORD *)(v208 + 16) = v209 | *(_QWORD *)(v208 + 16) & 3LL;
            }
            *v207 = v185;
            v210 = *(_QWORD *)(v185 + 8);
            v207[1] = v210;
            if ( v210 )
              *(_QWORD *)(v210 + 16) = (unsigned __int64)(v207 + 1) | *(_QWORD *)(v210 + 16) & 3LL;
            v207[2] = v207[2] & 3 | (v185 + 8);
            *(_QWORD *)(v185 + 8) = v207;
          }
          goto LABEL_313;
        }
LABEL_363:
        v211 = v458;
        v212 = &v458[6 * (unsigned int)v459];
        if ( v458 != v212 )
        {
          do
          {
            v213 = (__int64)v211;
            v211 += 6;
            sub_199EAC0(&v408, v213, (__int64)&v368, (__int64)&v362, a7, a8);
            LOBYTE(v414) = 1;
          }
          while ( v212 != v211 );
        }
        v214 = v373;
        ++v371;
        if ( !v373 )
          goto LABEL_383;
        v215 = v372;
        v216 = &v372[5 * v374];
        v217 = 4 * v373;
        if ( (unsigned int)(4 * v373) < 0x40 )
          v217 = 64;
        if ( v374 <= v217 )
        {
          if ( v372 == v216 )
          {
LABEL_382:
            v373 = 0;
            goto LABEL_383;
          }
          while ( *v215 != -8 )
          {
            if ( *v215 != -16 || v215[1] != -16 )
              goto LABEL_372;
LABEL_375:
            *v215 = -8;
            v215[1] = -8;
LABEL_376:
            v215 += 5;
            if ( v215 == v216 )
              goto LABEL_382;
          }
          if ( v215[1] == -8 )
            goto LABEL_376;
LABEL_372:
          v218 = v215[4];
          if ( v218 != 0 && v218 != -8 && v218 != -16 )
            sub_1649B30(v215 + 2);
          goto LABEL_375;
        }
        while ( 1 )
        {
          if ( *v215 == -8 )
          {
            if ( v215[1] != -8 )
              goto LABEL_421;
          }
          else if ( *v215 != -16 || v215[1] != -16 )
          {
LABEL_421:
            v233 = v215[4];
            if ( v233 != -8 && v233 != 0 && v233 != -16 )
              sub_1649B30(v215 + 2);
          }
          v215 += 5;
          if ( v215 == v216 )
          {
            if ( v214 )
            {
              if ( v214 == 1 )
              {
                v306 = 64;
              }
              else
              {
                _BitScanReverse(&v305, v214 - 1);
                v306 = 1 << (33 - (v305 ^ 0x1F));
                if ( v306 < 64 )
                  v306 = 64;
              }
              v307 = v372;
              v308 = (unsigned int)v306;
              if ( v374 != v306 )
              {
                j___libc_free_0(v372);
                v309 = ((((((((4 * v306 / 3u + 1) | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 2)
                          | (4 * v306 / 3u + 1)
                          | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 4)
                        | (((4 * v306 / 3u + 1) | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 2)
                        | (4 * v306 / 3u + 1)
                        | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 8)
                      | (((((4 * v306 / 3u + 1) | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 2)
                        | (4 * v306 / 3u + 1)
                        | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 4)
                      | (((4 * v306 / 3u + 1) | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 2)
                      | (4 * v306 / 3u + 1)
                      | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 16;
                v374 = (v309
                      | (((((((4 * v306 / 3u + 1) | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 2)
                          | (4 * v306 / 3u + 1)
                          | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 4)
                        | (((4 * v306 / 3u + 1) | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 2)
                        | (4 * v306 / 3u + 1)
                        | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 8)
                      | (((((4 * v306 / 3u + 1) | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 2)
                        | (4 * v306 / 3u + 1)
                        | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 4)
                      | (((4 * v306 / 3u + 1) | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 2)
                      | (4 * v306 / 3u + 1)
                      | ((4 * v306 / 3u + 1) >> 1))
                     + 1;
                v310 = (_QWORD *)sub_22077B0(
                                   40
                                 * ((v309
                                   | (((((((4 * v306 / 3u + 1) | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 2)
                                       | (4 * v306 / 3u + 1)
                                       | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 4)
                                     | (((4 * v306 / 3u + 1) | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 2)
                                     | (4 * v306 / 3u + 1)
                                     | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 8)
                                   | (((((4 * v306 / 3u + 1) | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 2)
                                     | (4 * v306 / 3u + 1)
                                     | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 4)
                                   | (((4 * v306 / 3u + 1) | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1)) >> 2)
                                   | (4 * v306 / 3u + 1)
                                   | ((unsigned __int64)(4 * v306 / 3u + 1) >> 1))
                                  + 1));
                v373 = 0;
                v372 = v310;
                v311 = &v310[5 * v374];
                while ( v311 != v310 )
                {
                  if ( v310 )
                  {
                    *v310 = -8;
                    v310[1] = -8;
                  }
                  v310 += 5;
                }
                goto LABEL_383;
              }
            }
            else
            {
              if ( v374 )
              {
                j___libc_free_0(v372);
                v372 = 0;
                v373 = 0;
                v374 = 0;
                goto LABEL_383;
              }
              v307 = v372;
              v308 = 0;
            }
            v312 = v307;
            v373 = 0;
            v313 = &v307[5 * v308];
            while ( v313 != v312 )
            {
              if ( v312 )
              {
                *v312 = -8;
                v312[1] = -8;
              }
              v312 += 5;
            }
LABEL_383:
            sub_1940B30((__int64)&v375);
            sub_1940B30((__int64)&v379);
            ++v395;
            if ( (_DWORD)v397 )
            {
              v226 = 4 * v397;
              v219 = (unsigned int)v398;
              if ( (unsigned int)(4 * v397) < 0x40 )
                v226 = 64;
              if ( v226 >= (unsigned int)v398 )
              {
LABEL_386:
                v220 = v396;
                v221 = &v396[v219];
                if ( v396 != v221 )
                {
                  do
                    *v220++ = -8;
                  while ( v221 != v220 );
                }
                v397 = 0;
                goto LABEL_389;
              }
              v227 = v396;
              if ( (_DWORD)v397 == 1 )
              {
                LODWORD(v229) = 64;
              }
              else
              {
                _BitScanReverse(&v228, v397 - 1);
                v229 = (unsigned int)(1 << (33 - (v228 ^ 0x1F)));
                if ( (int)v229 < 64 )
                  v229 = 64;
                if ( (_DWORD)v229 == (_DWORD)v398 )
                {
                  v397 = 0;
                  v296 = &v396[v229];
                  while ( v296 != v227 )
                  {
                    if ( v227 )
                      *v227 = -8;
                    ++v227;
                  }
                  goto LABEL_389;
                }
              }
              j___libc_free_0(v396);
              v230 = ((((((((4 * (int)v229 / 3u + 1) | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 2)
                        | (4 * (int)v229 / 3u + 1)
                        | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 4)
                      | (((4 * (int)v229 / 3u + 1) | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 2)
                      | (4 * (int)v229 / 3u + 1)
                      | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 8)
                    | (((((4 * (int)v229 / 3u + 1) | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 2)
                      | (4 * (int)v229 / 3u + 1)
                      | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 4)
                    | (((4 * (int)v229 / 3u + 1) | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 2)
                    | (4 * (int)v229 / 3u + 1)
                    | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 16;
              LODWORD(v398) = (v230
                             | (((((((4 * (int)v229 / 3u + 1) | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 2)
                                 | (4 * (int)v229 / 3u + 1)
                                 | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 4)
                               | (((4 * (int)v229 / 3u + 1) | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 2)
                               | (4 * (int)v229 / 3u + 1)
                               | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 8)
                             | (((((4 * (int)v229 / 3u + 1) | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 2)
                               | (4 * (int)v229 / 3u + 1)
                               | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 4)
                             | (((4 * (int)v229 / 3u + 1) | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 2)
                             | (4 * (int)v229 / 3u + 1)
                             | ((4 * (int)v229 / 3u + 1) >> 1))
                            + 1;
              v231 = (_QWORD *)sub_22077B0(
                                 8
                               * ((v230
                                 | (((((((4 * (int)v229 / 3u + 1) | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 2)
                                     | (4 * (int)v229 / 3u + 1)
                                     | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 4)
                                   | (((4 * (int)v229 / 3u + 1) | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 2)
                                   | (4 * (int)v229 / 3u + 1)
                                   | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 8)
                                 | (((((4 * (int)v229 / 3u + 1) | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 2)
                                   | (4 * (int)v229 / 3u + 1)
                                   | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 4)
                                 | (((4 * (int)v229 / 3u + 1) | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1)) >> 2)
                                 | (4 * (int)v229 / 3u + 1)
                                 | ((unsigned __int64)(4 * (int)v229 / 3u + 1) >> 1))
                                + 1));
              v397 = 0;
              v396 = v231;
              v232 = &v231[(unsigned int)v398];
              while ( v232 != v231 )
              {
                if ( v231 )
                  *v231 = -8;
                ++v231;
              }
              goto LABEL_389;
            }
            if ( HIDWORD(v397) )
            {
              v219 = (unsigned int)v398;
              if ( (unsigned int)v398 <= 0x40 )
                goto LABEL_386;
              j___libc_free_0(v396);
              v396 = 0;
              v397 = 0;
              LODWORD(v398) = 0;
            }
LABEL_389:
            v222 = sub_199E860((__int64)&v362);
            LOBYTE(v414) = v222 | v414;
            if ( v405 != v407 )
              _libc_free((unsigned __int64)v405);
            if ( v400[0] )
              sub_161E7C0((__int64)v400, v400[0]);
            j___libc_free_0(v396);
            if ( v389 != v388 )
              _libc_free((unsigned __int64)v389);
            j___libc_free_0(v384);
            j___libc_free_0(v380);
            j___libc_free_0(v376);
            if ( v374 )
            {
              v223 = v372;
              v224 = &v372[5 * v374];
              do
              {
                if ( *v223 == -8 )
                {
                  if ( v223[1] != -8 )
                    goto LABEL_398;
                }
                else if ( *v223 != -16 || v223[1] != -16 )
                {
LABEL_398:
                  v225 = v223[4];
                  if ( v225 != -8 && v225 != 0 && v225 != -16 )
                    sub_1649B30(v223 + 2);
                }
                v223 += 5;
              }
              while ( v224 != v223 );
            }
            j___libc_free_0(v372);
            v148 = v362;
            v149 = &v362[3 * (unsigned int)v363];
            if ( v362 != v149 )
            {
              do
              {
                v150 = *(v149 - 1);
                v149 -= 3;
                if ( v150 != -8 && v150 != 0 && v150 != -16 )
                  sub_1649B30(v149);
              }
              while ( v148 != v149 );
              v148 = v362;
            }
            if ( v148 != &v364 )
              _libc_free((unsigned __int64)v148);
LABEL_254:
            if ( v357 != v359 )
              _libc_free((unsigned __int64)v357);
            goto LABEL_30;
          }
        }
      }
      while ( 1 )
      {
        v121 = *(_QWORD *)(*v116 + 24LL * *((unsigned int *)v116 + 2) - 24);
        if ( *(_BYTE *)(v121 + 16) == 77 )
        {
          if ( !(_DWORD)v398 )
          {
            ++v395;
            goto LABEL_190;
          }
          v118 = ((_DWORD)v398 - 1) & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4));
          v119 = &v396[v118];
          v120 = *v119;
          if ( *v119 != v121 )
          {
            v287 = 1;
            v288 = 0;
            while ( v120 != -8 )
            {
              if ( !v288 && v120 == -16 )
                v288 = v119;
              v118 = ((_DWORD)v398 - 1) & (unsigned int)(v118 + v287);
              v119 = &v396[v118];
              v120 = *v119;
              if ( v121 == *v119 )
                goto LABEL_186;
              ++v287;
            }
            if ( v288 )
              v119 = v288;
            ++v395;
            v126 = v397 + 1;
            if ( 4 * ((int)v397 + 1) >= (unsigned int)(3 * v398) )
            {
LABEL_190:
              sub_19A8110((__int64)&v395, 2 * v398);
              if ( !(_DWORD)v398 )
                goto LABEL_631;
              v122 = 1;
              v123 = 0;
              for ( k = (v398 - 1) & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4)); ; k = (v398 - 1) & v314 )
              {
                v119 = &v396[k];
                v125 = *v119;
                if ( v121 == *v119 )
                {
                  v126 = v397 + 1;
                  goto LABEL_194;
                }
                if ( v125 == -8 )
                  break;
                if ( v123 || v125 != -16 )
                  v119 = v123;
                v314 = v122 + k;
                v123 = v119;
                ++v122;
              }
              if ( v123 )
                v119 = v123;
              v126 = v397 + 1;
            }
            else if ( (int)v398 - HIDWORD(v397) - v126 <= (unsigned int)v398 >> 3 )
            {
              sub_19A8110((__int64)&v395, v398);
              if ( !(_DWORD)v398 )
              {
LABEL_631:
                LODWORD(v397) = v397 + 1;
                BUG();
              }
              v119 = 0;
              v289 = (v398 - 1) & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4));
              for ( m = 1; ; ++m )
              {
                v291 = &v396[v289];
                v292 = *v291;
                if ( v121 == *v291 )
                {
                  v126 = v397 + 1;
                  v119 = v291;
                  goto LABEL_194;
                }
                if ( v292 == -8 )
                  break;
                if ( v119 || v292 != -16 )
                  v291 = v119;
                v294 = m + v289;
                v119 = v291;
                v289 = (v398 - 1) & v294;
              }
              if ( !v119 )
                v119 = &v396[v289];
              v126 = v397 + 1;
            }
LABEL_194:
            LODWORD(v397) = v126;
            if ( *v119 != -8 )
              --HIDWORD(v397);
            *v119 = v121;
          }
        }
LABEL_186:
        v116 += 6;
        if ( v117 == v116 )
          goto LABEL_276;
      }
    }
    while ( 1 )
    {
      v277 = *v275;
      v362 = (unsigned __int64 *)*v275;
      if ( !BYTE1(v414) || !byte_4FB1260 )
        goto LABEL_506;
      if ( !sub_19936F0(v329, v277) )
        break;
LABEL_508:
      sub_19AF050((__int64)&v368, (__int64)&v468, (__int64 *)&v362);
LABEL_503:
      if ( v276 == ++v275 )
        goto LABEL_152;
    }
    v277 = (__int64)v362;
LABEL_506:
    if ( *(_WORD *)(v277 + 24) == 7 || !(unsigned __int8)sub_1993F10(v277) )
      goto LABEL_503;
    goto LABEL_508;
  }
  if ( byte_4FB1180 )
  {
    if ( (unsigned int)v449 <= 1 )
      goto LABEL_151;
    goto LABEL_30;
  }
  v234 = (unsigned int)v449;
  v320 = (unsigned int)v449;
  v334 = v448;
  v318 = &v448[1984 * (unsigned int)v449];
  if ( v448 == v318 )
    goto LABEL_151;
  v323 = 1;
  while ( 2 )
  {
    v235 = *((unsigned int *)v334 + 188);
    v317 = v235;
    if ( v235 <= 1 )
      goto LABEL_457;
    v341 = 0;
    v328 = 0;
    v325 = 96 * v235;
    do
    {
      v236 = *((_QWORD *)v334 + 93) + v341;
      v368 = 1;
      v237 = *(_QWORD *)(v236 + 80);
      if ( !v237 )
      {
        v243 = *(_DWORD *)(v236 + 40);
        if ( !v243 )
        {
          v242 = 1;
LABEL_460:
          v328 += (int)sub_39FAC40((v242 >> 1) & ~(-1LL << (v242 >> 58)));
          goto LABEL_453;
        }
        v238 = v454;
        v255 = *(__int64 **)(v236 + 32);
        v239 = v452;
        if ( v454 )
        {
          v100 = *v255;
          LODWORD(v99) = (v454 - 1) & (((unsigned int)*v255 >> 9) ^ ((unsigned int)*v255 >> 4));
          v256 = &v452[2 * (unsigned int)v99];
          v257 = *v256;
          if ( *v255 == *v256 )
          {
LABEL_464:
            v242 = v256[1];
            if ( (v242 & 1) == 0 )
              goto LABEL_468;
LABEL_465:
            v368 = v242;
            if ( v243 != 1 )
            {
              v244 = v255 + 1;
              v245 = 2;
              goto LABEL_442;
            }
            goto LABEL_460;
          }
          v263 = 1;
          while ( v257 != -8 )
          {
            v234 = (unsigned int)(v263 + 1);
            LODWORD(v99) = (v454 - 1) & (v263 + v99);
            v256 = &v452[2 * (unsigned int)v99];
            v257 = *v256;
            if ( v100 == *v256 )
              goto LABEL_464;
            v263 = v234;
          }
        }
        v242 = v452[2 * v454 + 1];
        if ( (v242 & 1) == 0 )
        {
LABEL_468:
          v258 = (_QWORD *)sub_22077B0(24);
          v259 = (unsigned __int64)v258;
          if ( v258 )
          {
            *v258 = 0;
            v258[1] = 0;
            v260 = *(_DWORD *)(v242 + 16);
            *(_DWORD *)(v259 + 16) = v260;
            if ( v260 )
            {
              v316 = (unsigned int)(v260 + 63) >> 6;
              n = 8 * v316;
              v261 = (void *)malloc(8 * v316);
              if ( !v261 )
              {
                if ( n || (v295 = malloc(1u)) == 0 )
                  sub_16BD1C0("Allocation failed", 1u);
                else
                  v261 = (void *)v295;
              }
              *(_QWORD *)v259 = v261;
              v262 = *(_QWORD **)v242;
              *(_QWORD *)(v259 + 8) = v316;
              memcpy(v261, v262, n);
            }
          }
          v368 = v259;
          v245 = 1;
          while ( 2 )
          {
            if ( v245 == v243 )
            {
              v242 = v368;
              goto LABEL_445;
            }
            v248 = v245;
            v239 = v452;
            ++v245;
            v244 = (__int64 *)(*(_QWORD *)(v236 + 32) + 8 * v248);
            v238 = v454;
LABEL_442:
            if ( (_DWORD)v238 )
            {
              v99 = *v244;
              v246 = (v238 - 1) & (((unsigned int)*v244 >> 9) ^ ((unsigned int)*v244 >> 4));
              v247 = &v239[2 * v246];
              v100 = *v247;
              if ( v99 != *v247 )
              {
                v264 = 1;
                while ( v100 != -8 )
                {
                  v234 = (unsigned int)(v264 + 1);
                  v246 = (v238 - 1) & (v264 + v246);
                  v247 = &v239[2 * v246];
                  v100 = *v247;
                  if ( v99 == *v247 )
                    goto LABEL_439;
                  v264 = v234;
                }
                goto LABEL_443;
              }
            }
            else
            {
LABEL_443:
              v238 *= 16;
              v247 = (_QWORD *)((char *)v239 + v238);
            }
LABEL_439:
            sub_1998630((unsigned __int64 *)&v368, v247 + 1, v238, v234, v99, v100);
            continue;
          }
        }
        goto LABEL_465;
      }
      v238 = v454;
      v239 = v452;
      if ( v454 )
      {
        LODWORD(v100) = v454 - 1;
        LODWORD(v99) = (v454 - 1) & (((unsigned int)v237 >> 9) ^ ((unsigned int)v237 >> 4));
        v240 = &v452[2 * (unsigned int)v99];
        v241 = *v240;
        if ( v237 == *v240 )
          goto LABEL_435;
        v271 = 1;
        while ( v241 != -8 )
        {
          v234 = (unsigned int)(v271 + 1);
          LODWORD(v99) = v100 & (v271 + v99);
          v240 = &v452[2 * (unsigned int)v99];
          v241 = *v240;
          if ( v237 == *v240 )
            goto LABEL_435;
          v271 = v234;
        }
      }
      v240 = &v452[2 * v454];
LABEL_435:
      v242 = v240[1];
      if ( (v242 & 1) != 0 )
      {
        v368 = v240[1];
        v243 = *(_DWORD *)(v236 + 40);
        if ( v243 )
        {
          v244 = *(__int64 **)(v236 + 32);
          v245 = 1;
          goto LABEL_442;
        }
        goto LABEL_460;
      }
      v265 = (_QWORD *)sub_22077B0(24);
      v266 = (unsigned __int64)v265;
      if ( v265 )
      {
        *v265 = 0;
        v265[1] = 0;
        v267 = *(_DWORD *)(v242 + 16);
        *(_DWORD *)(v266 + 16) = v267;
        if ( v267 )
        {
          v268 = (unsigned int)(v267 + 63) >> 6;
          v269 = (void *)sub_13A3880(8 * v268);
          *(_QWORD *)(v266 + 8) = v268;
          v270 = *(_QWORD **)v242;
          *(_QWORD *)v266 = v269;
          memcpy(v269, v270, 8 * v268);
        }
      }
      v368 = v266;
      v243 = *(_DWORD *)(v236 + 40);
      v242 = v266;
      if ( v243 )
      {
        v244 = *(__int64 **)(v236 + 32);
        v239 = v452;
        v245 = 1;
        v238 = v454;
        goto LABEL_442;
      }
LABEL_445:
      if ( (v242 & 1) != 0 )
        goto LABEL_460;
      v249 = (unsigned int)(*(_DWORD *)(v242 + 16) + 63) >> 6;
      if ( v249 )
      {
        v250 = *(_QWORD **)v242;
        LODWORD(v251) = 0;
        v252 = *(_QWORD *)v242 + 8LL;
        v253 = v252 + 8LL * (v249 - 1);
        while ( 1 )
        {
          v251 = (unsigned int)sub_39FAC40(*v250) + (unsigned int)v251;
          v250 = (_QWORD *)v252;
          if ( v252 == v253 )
            break;
          v252 += 8;
        }
        v328 += v251;
      }
      if ( v242 )
      {
        _libc_free(*(_QWORD *)v242);
        j_j___libc_free_0(v242, 24);
      }
LABEL_453:
      v341 += 96;
    }
    while ( v341 != v325 );
    v254 = v317 - v328 / v320;
    if ( v254 <= 1 || v254 <= 0x3FFFB && (v323 *= v254, v323 <= 0x3FFFB) )
    {
LABEL_457:
      v334 += 1984;
      if ( v318 == v334 )
        goto LABEL_151;
      continue;
    }
    break;
  }
LABEL_30:
  v343 = v414;
  j___libc_free_0(v481);
  if ( v477 != (__int64 *)v479 )
    _libc_free((unsigned __int64)v477);
  j___libc_free_0(v473);
  j___libc_free_0(v469);
  if ( v463 != v462 )
    _libc_free((unsigned __int64)v463);
  v31 = v458;
  v32 = &v458[6 * (unsigned int)v459];
  if ( v458 != v32 )
  {
    do
    {
      v32 -= 6;
      if ( (unsigned __int64 *)*v32 != v32 + 2 )
        _libc_free(*v32);
    }
    while ( v31 != v32 );
    v32 = v458;
  }
  if ( v32 != (unsigned __int64 *)v460 )
    _libc_free((unsigned __int64)v32);
  if ( v455 != (__int64 *)v457 )
    _libc_free((unsigned __int64)v455);
  if ( v454 )
  {
    v33 = v452;
    v34 = &v452[2 * v454];
    do
    {
      if ( *v33 != -16 && *v33 != -8 )
      {
        v35 = (unsigned __int64 *)v33[1];
        if ( ((unsigned __int8)v35 & 1) == 0 )
        {
          if ( v35 )
          {
            _libc_free(*v35);
            j_j___libc_free_0(v35, 24);
          }
        }
      }
      v33 += 2;
    }
    while ( v34 != v33 );
  }
  j___libc_free_0(v452);
  v36 = v448;
  v37 = (unsigned __int64)&v448[1984 * (unsigned int)v449];
  if ( v448 != (_BYTE *)v37 )
  {
    do
    {
      v37 -= 1984LL;
      v38 = *(_QWORD *)(v37 + 1928);
      if ( v38 != *(_QWORD *)(v37 + 1920) )
        _libc_free(v38);
      v39 = *(_QWORD *)(v37 + 744);
      v40 = v39 + 96LL * *(unsigned int *)(v37 + 752);
      if ( v39 != v40 )
      {
        do
        {
          v40 -= 96LL;
          v41 = *(_QWORD *)(v40 + 32);
          if ( v41 != v40 + 48 )
            _libc_free(v41);
        }
        while ( v39 != v40 );
        v39 = *(_QWORD *)(v37 + 744);
      }
      if ( v39 != v37 + 760 )
        _libc_free(v39);
      v42 = *(_QWORD *)(v37 + 56);
      v43 = v42 + 80LL * *(unsigned int *)(v37 + 64);
      if ( v42 != v43 )
      {
        do
        {
          v43 -= 80LL;
          v44 = *(_QWORD *)(v43 + 32);
          if ( v44 != *(_QWORD *)(v43 + 24) )
            _libc_free(v44);
        }
        while ( v42 != v43 );
        v42 = *(_QWORD *)(v37 + 56);
      }
      if ( v42 != v37 + 72 )
        _libc_free(v42);
      v45 = *(unsigned int *)(v37 + 24);
      if ( (_DWORD)v45 )
      {
        v363 = 0x400000001LL;
        v362 = &v364;
        v364 = -1;
        v368 = (__int64)&v370;
        v369 = 0x400000001LL;
        v370 = -2;
        v46 = *(unsigned __int64 **)(v37 + 8);
        v47 = &v46[6 * v45];
        do
        {
          if ( (unsigned __int64 *)*v46 != v46 + 2 )
            _libc_free(*v46);
          v46 += 6;
        }
        while ( v47 != v46 );
      }
      j___libc_free_0(*(_QWORD *)(v37 + 8));
    }
    while ( v36 != (_BYTE *)v37 );
    v37 = (unsigned __int64)v448;
  }
  if ( (_BYTE *)v37 != v450 )
    _libc_free(v37);
  if ( v444 != v446 )
    _libc_free((unsigned __int64)v444);
  if ( (v440 & 1) == 0 )
    j___libc_free_0(v441[0]);
  if ( v431 != v433 )
    _libc_free((unsigned __int64)v431);
  sub_1994270((__int64)v427);
  if ( v416 != v418 )
    _libc_free((unsigned __int64)v416);
  v48 = sub_1AA7010(**(_QWORD **)(a3 + 32), 0) | v343;
  if ( byte_4FB1EA0 )
  {
    v55 = sub_13FCBF0(a3);
    if ( (_BYTE)v55 )
    {
      v369 = 0x1000000000LL;
      v368 = (__int64)&v370;
      v56 = sub_157EB90(**(_QWORD **)(a3 + 32));
      v57 = sub_1632FA0(v56);
      v409 = v57;
      v427 = (__int64 *)&v431;
      v428 = (__int64 *)&v431;
      v408 = a5;
      v410 = "lsr";
      v411 = 0;
      v412 = 0;
      v413 = 0;
      v414 = 0;
      v415 = 0;
      v416 = 0;
      v417 = 0;
      v418[0] = 0;
      v418[1] = 0;
      v419 = 0;
      v420 = 0;
      v421 = 0;
      v422 = 0;
      v423 = 0;
      v424 = 0;
      v425 = 0;
      v426 = 0;
      v429 = 2;
      LODWORD(v430) = 0;
      memset(v433, 0, sizeof(v433));
      v434 = 0;
      v435 = 0;
      v436 = 0;
      v437 = 1;
      v58 = sub_15E0530(*(_QWORD *)(a5 + 24));
      v445 = v57;
      v446[1] = 0x800000000LL;
      v438 = 0;
      v440 = 0;
      v441[0] = v58;
      v441[1] = 0;
      v442 = 0;
      v443 = 0;
      v444 = 0;
      v439 = 0;
      v446[0] = (unsigned __int64)v447;
      if ( (unsigned int)sub_387CB50(&v408, a3, a6, &v368, a16) )
      {
        v48 = v55;
        sub_199E860((__int64)&v368);
        sub_1AA7010(**(_QWORD **)(a3 + 32), 0);
      }
      if ( (_BYTE *)v446[0] != v447 )
        _libc_free(v446[0]);
      if ( v438 )
        sub_161E7C0((__int64)&v438, v438);
      j___libc_free_0(v434);
      if ( v428 != v427 )
        _libc_free((unsigned __int64)v428);
      j___libc_free_0(v423);
      j___libc_free_0(v419);
      j___libc_free_0(v416);
      if ( v414 )
      {
        v59 = (_QWORD *)v412;
        v60 = (_QWORD *)(v412 + 40LL * v414);
        do
        {
          if ( *v59 == -8 )
          {
            if ( v59[1] != -8 )
              goto LABEL_111;
          }
          else if ( *v59 != -16 || v59[1] != -16 )
          {
LABEL_111:
            v61 = v59[4];
            if ( v61 != 0 && v61 != -8 && v61 != -16 )
              sub_1649B30(v59 + 2);
          }
          v59 += 5;
        }
        while ( v60 != v59 );
      }
      j___libc_free_0(v412);
      v62 = v368;
      v63 = (unsigned __int64 *)(v368 + 24LL * (unsigned int)v369);
      if ( (unsigned __int64 *)v368 != v63 )
      {
        do
        {
          v64 = *(v63 - 1);
          v63 -= 3;
          if ( v64 != 0 && v64 != -8 && v64 != -16 )
            sub_1649B30(v63);
        }
        while ( (unsigned __int64 *)v62 != v63 );
        v63 = (unsigned __int64 *)v368;
      }
      if ( v63 != &v370 )
        _libc_free((unsigned __int64)v63);
    }
  }
  return v48;
}
