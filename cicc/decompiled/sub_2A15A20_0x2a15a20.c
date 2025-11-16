// Function: sub_2A15A20
// Address: 0x2a15a20
//
__int64 __fastcall sub_2A15A20(
        __int64 *a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        __int64 a9,
        __int64 a10,
        __int64 *a11,
        unsigned __int8 a12,
        __int64 a13,
        unsigned __int64 a14)
{
  __int64 *v14; // r12
  unsigned int v15; // r13d
  __int64 *v17; // rsi
  const void *v18; // r13
  _QWORD *v19; // rbx
  _BYTE *v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  unsigned __int64 **v29; // r13
  __int64 v30; // r14
  __int64 v31; // rdx
  __int64 v32; // rcx
  unsigned int v33; // esi
  unsigned int v34; // eax
  _BYTE *v35; // rbx
  int v36; // esi
  char *v37; // rdx
  int v38; // eax
  unsigned int v39; // eax
  unsigned int i; // edi
  _QWORD *v41; // rsi
  char *v42; // r13
  unsigned __int64 v43; // rbx
  __int64 v44; // rsi
  _QWORD *v45; // r13
  _QWORD *v46; // rbx
  unsigned __int64 v47; // rcx
  _QWORD *v48; // rax
  _QWORD *v49; // rdx
  __int64 v50; // rax
  _QWORD *v51; // rdx
  __int64 v52; // rsi
  __int64 v53; // rax
  __int64 *v54; // r12
  __int64 *v55; // rbx
  unsigned __int64 v56; // rdi
  unsigned __int8 v57; // al
  char v58; // dl
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 v63; // rbx
  __int64 *v64; // rsi
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // rdx
  __int64 *v69; // rax
  const void *v70; // r13
  size_t v71; // r12
  _BYTE *v72; // rbx
  __int64 *v73; // rax
  void **v74; // r12
  void **v75; // rbx
  const char *v76; // rax
  __int64 v77; // rcx
  __int64 v78; // r8
  __int64 *v79; // rax
  __int64 v80; // r15
  __int64 v81; // rbx
  __m128i *v82; // r12
  __int64 v83; // rdi
  const char *v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rsi
  unsigned __int8 *v87; // rsi
  __int64 v88; // rdi
  __int64 v89; // rsi
  __int64 v90; // rsi
  __int64 v91; // r8
  __int64 v92; // r9
  void **v93; // rax
  __int64 *v94; // rbx
  unsigned int v95; // eax
  __int64 v96; // r12
  __int64 v97; // rbx
  __int64 v98; // rcx
  __int64 v99; // rax
  __int64 *v100; // rax
  __int64 v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // r8
  __int64 v104; // r9
  char *v105; // rsi
  __int64 v106; // rbx
  __int64 v107; // r8
  __int64 v108; // r9
  _QWORD *v109; // r12
  __int64 v110; // rax
  __int64 v111; // r12
  unsigned __int64 v112; // rax
  unsigned int v113; // r13d
  __int64 v114; // rax
  __int64 v115; // r12
  _QWORD *v116; // rax
  _QWORD *v117; // rdx
  unsigned __int64 v118; // r13
  unsigned int v119; // edx
  __int64 *v120; // rbx
  __int64 v121; // rax
  __int64 v122; // rax
  unsigned __int64 v123; // rdx
  _BYTE *v124; // rsi
  __int64 v125; // rax
  __int64 v126; // rdx
  unsigned int v127; // eax
  _QWORD *v128; // rax
  unsigned int v129; // eax
  _QWORD *v130; // r13
  _QWORD *v131; // rbx
  __int64 v132; // rdx
  __int64 v133; // rax
  _QWORD *v134; // rdi
  __int64 v135; // rdx
  __int64 v136; // r8
  __int64 *v137; // r9
  unsigned __int64 *v138; // rbx
  unsigned __int64 *v139; // r12
  __int64 v140; // r13
  __int64 v141; // r14
  __int64 v142; // rax
  __int64 v143; // rcx
  __int64 *v144; // rax
  __int64 v145; // r9
  __int64 *v146; // r12
  __int64 *j; // rbx
  __int64 v148; // r13
  __int64 v149; // rax
  __int64 v150; // rdx
  __int64 v151; // rbx
  __int64 v152; // rdx
  __int64 v153; // rdi
  unsigned int v154; // r9d
  int v155; // eax
  __int64 v156; // rdx
  __int64 v157; // r12
  unsigned int v158; // esi
  _QWORD *v159; // rcx
  __int64 v160; // r10
  __int64 v161; // r15
  int v162; // eax
  __int64 v163; // rdi
  __int64 v164; // rax
  __int64 v165; // rax
  __int64 v166; // rax
  int v167; // ecx
  int v168; // r15d
  __int64 v169; // r10
  __int64 v170; // r15
  unsigned __int64 v171; // rcx
  __int64 v172; // r14
  __int64 v173; // rax
  _QWORD *v174; // r12
  _QWORD *v175; // r15
  __int64 v176; // rax
  __int64 v177; // rax
  __int64 v178; // rdx
  __int64 v179; // rax
  __int64 v180; // rax
  int v181; // esi
  _QWORD *v182; // rdx
  int v183; // eax
  bool v184; // zf
  _QWORD *v185; // rdx
  __int64 v186; // rsi
  unsigned __int64 *v187; // r9
  __int64 *v188; // rsi
  unsigned int v189; // eax
  _QWORD *v190; // rbx
  _QWORD *v191; // r12
  __int64 v192; // rsi
  _BYTE *v193; // rsi
  _BYTE *v194; // rsi
  __int64 *v195; // rbx
  __int64 v196; // r12
  __int64 v197; // rcx
  __int64 v198; // r15
  __int64 v199; // rdi
  __int64 v200; // rdx
  __int64 v201; // rax
  __int64 v202; // rax
  __int64 v203; // r14
  _QWORD *v204; // rdi
  __int64 v205; // rax
  unsigned int v206; // eax
  __int64 v207; // rsi
  __int64 v208; // rdx
  unsigned __int64 v209; // rbx
  __int64 v210; // r8
  const char *v211; // r13
  unsigned __int64 v212; // r12
  __int64 v213; // rdx
  __int64 v214; // rcx
  __int64 v215; // r8
  __int64 v216; // r9
  __int64 v217; // rdx
  unsigned int v218; // esi
  __int64 v219; // rax
  __int64 v220; // rdx
  __int64 v221; // rcx
  __int64 v222; // r14
  int v223; // ecx
  unsigned __int64 v224; // rcx
  __int64 v225; // r13
  __int64 v226; // r14
  __int64 v227; // rbx
  unsigned __int8 *v228; // rax
  __int64 v229; // rbx
  __int64 v230; // rdx
  unsigned int v231; // eax
  __int64 v232; // rax
  __int64 v233; // rcx
  __int64 **v234; // r13
  __int64 **v235; // r14
  __int64 v236; // r15
  __int64 v237; // rax
  unsigned __int64 v238; // rdx
  const char *v239; // rdi
  __int64 v240; // r14
  unsigned __int64 v241; // rax
  __int64 v242; // rdx
  unsigned __int64 v243; // rdx
  const char *v244; // r15
  unsigned int v245; // eax
  unsigned int v246; // edx
  __int64 v247; // rbx
  __int64 v248; // rcx
  unsigned int v249; // eax
  __int64 v250; // r13
  _QWORD *v251; // rdi
  _QWORD *v252; // rax
  int v253; // r8d
  unsigned __int64 v254; // rcx
  size_t v255; // rdx
  __int64 v256; // rax
  __int64 v257; // rdx
  __int64 v258; // rcx
  void **v259; // rbx
  __int64 v260; // rdx
  __int64 *v261; // r13
  __int64 *v262; // r14
  __int64 v263; // r15
  unsigned __int64 v264; // rax
  __int64 v265; // r15
  char v266; // al
  __int64 *v267; // rdx
  __int64 *v268; // rax
  __int64 v269; // rsi
  _QWORD *v270; // r13
  __int64 v271; // rdx
  __int64 v272; // rcx
  __int64 v273; // r9
  __int64 v274; // r8
  __int64 v275; // rax
  __int64 v276; // r8
  __int64 v277; // r9
  _QWORD *v278; // rdx
  __int64 v279; // rdi
  __int64 v280; // rdx
  __int64 v281; // rcx
  __int64 v282; // r8
  __int64 v283; // r9
  _QWORD *v284; // rbx
  unsigned __int64 v285; // r15
  _QWORD *v286; // r13
  void (__fastcall *v287)(_QWORD *, _QWORD *, __int64); // rax
  _QWORD *v288; // rdi
  __int64 v289; // rax
  _QWORD *v290; // r14
  __int64 v291; // rax
  _QWORD *v292; // rbx
  __int64 v293; // rdx
  _QWORD *v294; // rdi
  __int64 v295; // rbx
  __int64 v296; // r13
  __int64 v297; // rax
  char *v298; // rbx
  __int64 v299; // rax
  __int64 v300; // rax
  char *v301; // rbx
  __int64 v302; // rdx
  __int64 v303; // rcx
  __int64 v304; // r8
  __int64 v305; // r9
  __int64 *v306; // r13
  __int64 v307; // rdx
  __int64 *v308; // rax
  unsigned int v309; // eax
  __int64 v310; // rax
  unsigned __int64 v311; // rbx
  const void *v312; // r14
  __int64 v313; // r12
  __int64 *v314; // rdi
  __int64 **v315; // rbx
  __int64 **v316; // r12
  __int64 v317; // rdx
  unsigned int v318; // eax
  __int64 *v319; // r15
  __int64 v320; // rax
  __int64 *v321; // rax
  __int64 *v322; // r15
  __int64 *v323; // rbx
  __int64 v324; // r12
  __int64 *v325; // r15
  __int64 v326; // rbx
  unsigned int v327; // r14d
  unsigned int v328; // edx
  int v329; // eax
  unsigned __int64 v330; // rax
  __int64 v331; // r8
  __int64 v332; // r9
  _QWORD *v333; // rdx
  __int64 v334; // rax
  _QWORD *v335; // r12
  _QWORD *v336; // rbx
  __int64 v337; // rsi
  unsigned __int64 v338; // rbx
  _BYTE *v339; // r13
  __int64 *v340; // rdi
  __int64 v341; // [rsp-10h] [rbp-7D0h]
  unsigned __int64 v342; // [rsp-10h] [rbp-7D0h]
  __int64 v343; // [rsp-10h] [rbp-7D0h]
  __int64 v344; // [rsp-10h] [rbp-7D0h]
  __int64 v345; // [rsp-8h] [rbp-7C8h]
  __int64 v346; // [rsp-8h] [rbp-7C8h]
  __int64 v347; // [rsp+38h] [rbp-788h]
  size_t n; // [rsp+50h] [rbp-770h]
  unsigned __int64 v349; // [rsp+58h] [rbp-768h]
  char *v350; // [rsp+60h] [rbp-760h]
  size_t v351; // [rsp+68h] [rbp-758h]
  char *v352; // [rsp+70h] [rbp-750h]
  char *v353; // [rsp+78h] [rbp-748h]
  unsigned __int8 v355; // [rsp+8Bh] [rbp-735h]
  char v356; // [rsp+8Ch] [rbp-734h]
  bool v357; // [rsp+8Dh] [rbp-733h]
  char v358; // [rsp+8Eh] [rbp-732h]
  char v359; // [rsp+8Fh] [rbp-731h]
  __int64 *v360; // [rsp+B0h] [rbp-710h]
  unsigned __int64 v362; // [rsp+C8h] [rbp-6F8h]
  unsigned __int8 v363; // [rsp+D8h] [rbp-6E8h]
  int v364; // [rsp+DCh] [rbp-6E4h]
  unsigned int v365; // [rsp+E0h] [rbp-6E0h]
  unsigned int v366; // [rsp+E4h] [rbp-6DCh]
  _QWORD *v370; // [rsp+108h] [rbp-6B8h]
  __int64 v371; // [rsp+118h] [rbp-6A8h]
  __int64 v372; // [rsp+118h] [rbp-6A8h]
  __int64 v373; // [rsp+118h] [rbp-6A8h]
  _QWORD *v374; // [rsp+118h] [rbp-6A8h]
  unsigned __int64 *v375; // [rsp+118h] [rbp-6A8h]
  _QWORD *v376; // [rsp+118h] [rbp-6A8h]
  int v377; // [rsp+120h] [rbp-6A0h]
  unsigned __int64 v378; // [rsp+120h] [rbp-6A0h]
  __int64 v379; // [rsp+128h] [rbp-698h]
  unsigned __int64 v380; // [rsp+128h] [rbp-698h]
  char *v381; // [rsp+128h] [rbp-698h]
  __int64 v382; // [rsp+128h] [rbp-698h]
  unsigned __int64 **v383; // [rsp+130h] [rbp-690h]
  __int64 k; // [rsp+130h] [rbp-690h]
  __int64 v385; // [rsp+130h] [rbp-690h]
  __int64 *v386; // [rsp+130h] [rbp-690h]
  const char *v387; // [rsp+130h] [rbp-690h]
  __int64 v388; // [rsp+130h] [rbp-690h]
  __int64 *v389; // [rsp+130h] [rbp-690h]
  __int64 *v390[2]; // [rsp+138h] [rbp-688h] BYREF
  unsigned int v391; // [rsp+14Ch] [rbp-674h] BYREF
  char *v392; // [rsp+150h] [rbp-670h] BYREF
  char *v393; // [rsp+158h] [rbp-668h] BYREF
  __int64 v394; // [rsp+160h] [rbp-660h] BYREF
  __int64 v395; // [rsp+168h] [rbp-658h] BYREF
  __int64 *v396; // [rsp+170h] [rbp-650h] BYREF
  _QWORD *v397; // [rsp+178h] [rbp-648h] BYREF
  _QWORD *v398; // [rsp+180h] [rbp-640h] BYREF
  __int64 v399; // [rsp+188h] [rbp-638h]
  __int64 *v400; // [rsp+190h] [rbp-630h] BYREF
  __int64 *v401; // [rsp+198h] [rbp-628h]
  __int64 *v402; // [rsp+1A0h] [rbp-620h]
  unsigned __int64 v403; // [rsp+1B0h] [rbp-610h] BYREF
  _BYTE *v404; // [rsp+1B8h] [rbp-608h]
  _BYTE *v405; // [rsp+1C0h] [rbp-600h]
  unsigned __int64 v406; // [rsp+1D0h] [rbp-5F0h] BYREF
  _BYTE *v407; // [rsp+1D8h] [rbp-5E8h]
  _BYTE *v408; // [rsp+1E0h] [rbp-5E0h]
  __int64 *v409; // [rsp+1F0h] [rbp-5D0h] BYREF
  _BYTE *v410; // [rsp+1F8h] [rbp-5C8h]
  _BYTE *v411; // [rsp+200h] [rbp-5C0h]
  __int64 v412; // [rsp+210h] [rbp-5B0h] BYREF
  __int64 *v413; // [rsp+218h] [rbp-5A8h]
  __int64 v414; // [rsp+220h] [rbp-5A0h]
  unsigned int v415; // [rsp+228h] [rbp-598h]
  void **v416; // [rsp+230h] [rbp-590h]
  const char *v417; // [rsp+238h] [rbp-588h]
  unsigned __int64 v418; // [rsp+240h] [rbp-580h]
  unsigned __int64 v419; // [rsp+248h] [rbp-578h]
  void *v420; // [rsp+250h] [rbp-570h] BYREF
  __int64 v421; // [rsp+258h] [rbp-568h] BYREF
  __int64 v422; // [rsp+260h] [rbp-560h]
  __int64 v423; // [rsp+268h] [rbp-558h]
  __int64 *v424; // [rsp+270h] [rbp-550h]
  char *v425; // [rsp+280h] [rbp-540h] BYREF
  __int64 v426; // [rsp+288h] [rbp-538h] BYREF
  __int64 v427; // [rsp+290h] [rbp-530h]
  __int64 v428; // [rsp+298h] [rbp-528h]
  __int64 v429; // [rsp+2A0h] [rbp-520h]
  _QWORD *v430; // [rsp+2B0h] [rbp-510h] BYREF
  __int64 v431; // [rsp+2B8h] [rbp-508h]
  _BYTE v432[32]; // [rsp+2C0h] [rbp-500h] BYREF
  unsigned __int64 **v433; // [rsp+2E0h] [rbp-4E0h] BYREF
  __int64 v434; // [rsp+2E8h] [rbp-4D8h]
  _BYTE v435[32]; // [rsp+2F0h] [rbp-4D0h] BYREF
  __int64 v436[4]; // [rsp+310h] [rbp-4B0h] BYREF
  unsigned int v437; // [rsp+330h] [rbp-490h]
  unsigned __int64 v438[3]; // [rsp+338h] [rbp-488h] BYREF
  __int64 *v439; // [rsp+350h] [rbp-470h] BYREF
  __int64 v440; // [rsp+358h] [rbp-468h]
  _BYTE v441[48]; // [rsp+360h] [rbp-460h] BYREF
  __int64 v442; // [rsp+390h] [rbp-430h] BYREF
  _QWORD *v443; // [rsp+398h] [rbp-428h]
  int v444; // [rsp+3A0h] [rbp-420h]
  int v445; // [rsp+3A4h] [rbp-41Ch]
  unsigned int v446; // [rsp+3A8h] [rbp-418h]
  _QWORD *v447; // [rsp+3B8h] [rbp-408h]
  unsigned int v448; // [rsp+3C8h] [rbp-3F8h]
  char v449; // [rsp+3D0h] [rbp-3F0h]
  __int64 v450; // [rsp+3E0h] [rbp-3E0h] BYREF
  __int64 v451; // [rsp+3E8h] [rbp-3D8h]
  __int64 v452; // [rsp+3F0h] [rbp-3D0h]
  __int64 v453; // [rsp+3F8h] [rbp-3C8h]
  _BYTE *v454; // [rsp+400h] [rbp-3C0h]
  __int64 v455; // [rsp+408h] [rbp-3B8h]
  _BYTE v456[32]; // [rsp+410h] [rbp-3B0h] BYREF
  unsigned __int64 *v457; // [rsp+430h] [rbp-390h] BYREF
  __int64 v458; // [rsp+438h] [rbp-388h]
  _BYTE v459[64]; // [rsp+440h] [rbp-380h] BYREF
  __m128i v460; // [rsp+480h] [rbp-340h] BYREF
  __int64 v461; // [rsp+490h] [rbp-330h] BYREF
  __int64 v462; // [rsp+498h] [rbp-328h]
  unsigned __int64 v463[6]; // [rsp+4A0h] [rbp-320h] BYREF
  const char *v464; // [rsp+4D0h] [rbp-2F0h] BYREF
  __int64 v465; // [rsp+4D8h] [rbp-2E8h] BYREF
  __int64 v466; // [rsp+4E0h] [rbp-2E0h] BYREF
  __int64 v467; // [rsp+4E8h] [rbp-2D8h]
  __int64 v468; // [rsp+4F0h] [rbp-2D0h]
  _QWORD *v469; // [rsp+4F8h] [rbp-2C8h]
  __int64 v470; // [rsp+500h] [rbp-2C0h]
  unsigned int v471; // [rsp+508h] [rbp-2B8h]
  char v472; // [rsp+510h] [rbp-2B0h]
  _BYTE v473[448]; // [rsp+520h] [rbp-2A0h] BYREF
  __int64 v474; // [rsp+6E0h] [rbp-E0h]
  __int64 v475; // [rsp+6E8h] [rbp-D8h]
  __int64 v476; // [rsp+6F0h] [rbp-D0h]
  __int64 v477; // [rsp+6F8h] [rbp-C8h]
  char v478; // [rsp+700h] [rbp-C0h]
  __int64 v479; // [rsp+708h] [rbp-B8h]
  char *v480; // [rsp+710h] [rbp-B0h]
  __int64 v481; // [rsp+718h] [rbp-A8h]
  int v482; // [rsp+720h] [rbp-A0h]
  char v483; // [rsp+724h] [rbp-9Ch]
  char v484; // [rsp+728h] [rbp-98h] BYREF
  __int16 v485; // [rsp+768h] [rbp-58h]
  _QWORD *v486; // [rsp+770h] [rbp-50h]
  _QWORD *v487; // [rsp+778h] [rbp-48h]
  __int64 v488; // [rsp+780h] [rbp-40h]

  v390[0] = a1;
  v14 = a11;
  v363 = a12;
  if ( !sub_D4B130((__int64)a1)
    || !sub_D47930((__int64)v390[0])
    || !(unsigned __int8)sub_D49210((__int64)v390[0])
    || (*(_WORD *)(**((_QWORD **)v390[0] + 4) + 2LL) & 0x7FFF) != 0 )
  {
    return 0;
  }
  v347 = sub_D4B130((__int64)v390[0]);
  v392 = (char *)**((_QWORD **)v390[0] + 4);
  v393 = (char *)sub_D47930((__int64)v390[0]);
  v430 = v432;
  v431 = 0x400000000LL;
  sub_D472F0((__int64)v390[0], (__int64)&v430);
  v17 = v390[0];
  v18 = (const void *)v390[0][4];
  n = v390[0][5] - (_QWORD)v18;
  if ( n > 0x7FFFFFFFFFFFFFF8LL )
    goto LABEL_106;
  if ( n )
  {
    v353 = (char *)sub_22077B0(n);
    v352 = &v353[n];
    memcpy(v353, v18, n);
    v17 = v390[0];
  }
  else
  {
    v352 = 0;
    v353 = 0;
  }
  v366 = sub_DBB070((__int64)a3, (__int64)v17, 0);
  v391 = 0;
  v358 = sub_DBA820((__int64)a3, (__int64)v390[0]);
  v399 = sub_F6EC60((__int64)v390[0], &v391);
  v19 = (_QWORD *)sub_AA4B30(**((_QWORD **)v390[0] + 4));
  v464 = (const char *)&v466;
  v20 = (_BYTE *)v19[29];
  sub_2A10890((__int64 *)&v464, v20, (__int64)&v20[v19[30]]);
  v468 = v19[33];
  v469 = (_QWORD *)v19[34];
  v24 = v19[35];
  v470 = v24;
  if ( (unsigned int)(v468 - 42) <= 1 && (unsigned int)sub_2A11A40((__int64)v390[0], (__int64)v20, v24, v21, v22, v23) )
  {
    sub_2240A30((unsigned __int64 *)&v464);
    sub_D4BD20(&v394, (__int64)v390[0], v302, v303, v304, v305);
    v366 = 0;
    if ( !v14 )
      goto LABEL_18;
    goto LABEL_13;
  }
  sub_2240A30((unsigned __int64 *)&v464);
  sub_D4BD20(&v394, (__int64)v390[0], v25, v26, v27, v28);
  if ( v14 )
  {
LABEL_13:
    if ( (unsigned __int8)sub_2A10940(*v14) )
    {
      v301 = v392;
      sub_B157E0((__int64)&v460, &v394);
      sub_B17430((__int64)&v464, (__int64)"loop-unroll", (__int64)"UnrollLoop", 10, &v460, (__int64)v301);
      sub_B18290((__int64)&v464, "  Applying unrolling strategy...", 0x20u);
      sub_1049740(v14, (__int64)&v464);
      v464 = (const char *)&unk_49D9D40;
      sub_23FD590((__int64)v473);
    }
  }
  if ( v366 && (unsigned int)a7 > v366 )
    LODWORD(a7) = v366;
LABEL_18:
  v412 = 0;
  v433 = (unsigned __int64 **)v435;
  v434 = 0x400000000LL;
  v413 = 0;
  v414 = 0;
  v415 = 0;
  sub_D46D90((__int64)v390[0], (__int64)&v433);
  v29 = v433;
  v383 = &v433[(unsigned int)v434];
  if ( v433 != v383 )
  {
    while ( 1 )
    {
      v457 = *v29;
      v35 = (_BYTE *)sub_986580((__int64)v457);
      if ( *v35 == 31 )
        break;
LABEL_24:
      if ( v383 == ++v29 )
        goto LABEL_41;
    }
    if ( (unsigned __int8)sub_2A105E0((__int64)&v412, (__int64 *)&v457, &v460) )
    {
      v30 = v460.m128i_i64[0] + 8;
LABEL_21:
      *(_DWORD *)v30 = sub_DBA790((__int64)a3, (__int64)v390[0], (__int64)v457);
      v33 = sub_DE5E70(a3, (__int64)v390[0], (__int64)v457);
      v34 = *(_DWORD *)v30;
      if ( *(_DWORD *)v30 )
      {
        HIDWORD(v31) = 0;
        *(_DWORD *)(v30 + 4) = 0;
        LODWORD(v31) = v34 % (unsigned int)a7;
        *(_DWORD *)(v30 + 8) = v34 % (unsigned int)a7;
      }
      else
      {
        v39 = a7;
        if ( (_DWORD)a7 )
        {
          if ( v33 )
          {
            v31 = (unsigned int)a7 % v33;
            v39 = v33;
            for ( i = (unsigned int)a7 % v33; i; i = v31 )
            {
              v31 = v39 % i;
              v39 = i;
            }
          }
        }
        else
        {
          v39 = v33;
        }
        *(_DWORD *)(v30 + 4) = v39;
        *(_DWORD *)(v30 + 8) = v39;
      }
      *(_BYTE *)(v30 + 12) = sub_B19060((__int64)(v390[0] + 7), *((_QWORD *)v35 - 4), v31, v32) ^ 1;
      sub_B1A4E0(v30 + 24, (__int64)v457);
      goto LABEL_24;
    }
    v36 = v415;
    v37 = (char *)v460.m128i_i64[0];
    ++v412;
    v38 = v414 + 1;
    v464 = (const char *)v460.m128i_i64[0];
    if ( 4 * ((int)v414 + 1) >= 3 * v415 )
    {
      v36 = 2 * v415;
    }
    else if ( v415 - HIDWORD(v414) - v38 > v415 >> 3 )
    {
LABEL_29:
      LODWORD(v414) = v38;
      if ( *(_QWORD *)v37 != -4096 )
        --HIDWORD(v414);
      v30 = (__int64)(v37 + 8);
      *(_QWORD *)v37 = v457;
      memset(v37 + 8, 0, 0x58u);
      *((_QWORD *)v37 + 4) = v37 + 48;
      *((_QWORD *)v37 + 5) = 0x600000000LL;
      goto LABEL_21;
    }
    sub_2A11010((__int64)&v412, v36);
    sub_2A105E0((__int64)&v412, (__int64 *)&v457, &v464);
    v37 = (char *)v464;
    v38 = v414 + 1;
    goto LABEL_29;
  }
LABEL_41:
  v357 = (_DWORD)a7 == v366;
  v364 = a7;
  if ( (_DWORD)a7 == v366 && (BYTE5(a7) = 0, v363) )
  {
    v41 = &v430[(unsigned int)v431];
    v359 = v41 != sub_2A106A0(v430, (__int64)v41);
  }
  else
  {
    v359 = 0;
  }
  v42 = v393;
  v43 = sub_986580((__int64)v393);
  if ( *(_BYTE *)v43 == 31 )
  {
    v44 = (__int64)v42;
    v356 = sub_D46CA0((__int64)v390[0], (__int64)v42);
    v355 = v356 | ((*(_DWORD *)(v43 + 4) & 0x7FFFFFF) != 3);
    if ( v355 )
    {
      v45 = sub_C52410();
      v46 = v45 + 1;
      v47 = sub_C959E0();
      v48 = (_QWORD *)v45[2];
      if ( v48 )
      {
        v49 = v45 + 1;
        do
        {
          v44 = v48[3];
          if ( v47 > v48[4] )
          {
            v48 = (_QWORD *)v48[3];
          }
          else
          {
            v49 = v48;
            v48 = (_QWORD *)v48[2];
          }
        }
        while ( v48 );
        if ( v46 != v49 && v47 >= v49[4] )
          v46 = v49;
      }
      if ( v46 == (_QWORD *)((char *)sub_C52410() + 8) )
        goto LABEL_502;
      v50 = v46[7];
      v44 = (__int64)(v46 + 6);
      if ( !v50 )
        goto LABEL_502;
      v51 = v46 + 6;
      do
      {
        if ( *(_DWORD *)(v50 + 32) < dword_500A368 )
        {
          v50 = *(_QWORD *)(v50 + 24);
        }
        else
        {
          v51 = (_QWORD *)v50;
          v50 = *(_QWORD *)(v50 + 16);
        }
      }
      while ( v50 );
      if ( (_QWORD *)v44 == v51 || dword_500A368 < *((_DWORD *)v51 + 8) || !*((_DWORD *)v51 + 9) )
LABEL_502:
        v57 = sub_2A10B40((__int64)v390[0]);
      else
        v57 = qword_500A3E8;
      v58 = BYTE1(a8);
      if ( (_BYTE)qword_500A148 )
        v57 |= BYTE1(a8);
      if ( BYTE2(a8) )
        v57 = BYTE2(a8);
      if ( !BYTE5(a7) )
        goto LABEL_89;
      v44 = (unsigned int)a7;
      if ( (unsigned __int8)sub_2A25260(
                              v390[0],
                              a7,
                              BYTE6(a7),
                              v57,
                              HIBYTE(a7),
                              (unsigned __int8)a8,
                              a2,
                              (__int64)a3,
                              a4,
                              a5,
                              a6,
                              v363,
                              a10,
                              BYTE4(a10),
                              a13,
                              BYTE1(a8)) )
      {
LABEL_501:
        v58 = BYTE1(a8);
LABEL_89:
        if ( v58 )
          sub_D4A9E0((__int64)v390[0]);
        if ( v364 == v366 )
        {
          if ( v14 )
          {
            v44 = (__int64)v390;
            sub_2A11DD0(v14, (__int64 *)v390, (unsigned int *)&a7);
          }
        }
        else if ( v14 && (unsigned __int8)sub_2A10940(*v14) )
        {
          v295 = **((_QWORD **)v390[0] + 4);
          sub_D4BD20(&v457, (__int64)v390[0], v59, v60, v61, v62);
          sub_B157E0((__int64)&v460, &v457);
          sub_B17430((__int64)&v464, (__int64)"loop-unroll", (__int64)"PartialUnrolled", 15, &v460, v295);
          if ( v457 )
            sub_B91220((__int64)&v457, (__int64)v457);
          sub_B18290((__int64)&v464, "unrolled loop by a factor of ", 0x1Du);
          sub_B169E0(v460.m128i_i64, "UnrollCount", 11, a7);
          sub_23FD640((__int64)&v464, (__int64)&v460);
          sub_2240A30(v463);
          sub_2240A30((unsigned __int64 *)&v460);
          if ( BYTE5(a7) )
            sub_B18290((__int64)&v464, " with run-time trip count", 0x19u);
          v44 = (__int64)&v464;
          sub_1049740(v14, (__int64)&v464);
          v464 = (const char *)&unk_49D9D40;
          sub_23FD590((__int64)v473);
        }
        if ( a3 )
        {
          if ( (_BYTE)a8 )
          {
            sub_D9FFA0((__int64)a3, v44);
          }
          else
          {
            sub_DAC8B0((__int64)a3, v390[0]);
            sub_D9D700((__int64)a3, 0);
          }
        }
        v442 = 0;
        v446 = 128;
        v443 = (_QWORD *)sub_C7D670(0x2000, 8);
        sub_23FE7B0((__int64)&v442);
        v449 = 0;
        v400 = 0;
        v401 = 0;
        v63 = *((_QWORD *)v392 + 7);
        v402 = 0;
        while ( 1 )
        {
          if ( !v63 )
            goto LABEL_609;
          if ( *(_BYTE *)(v63 - 24) != 84 )
            break;
          v464 = (const char *)(v63 - 24);
          v64 = v401;
          if ( v401 == v402 )
          {
            sub_2A12230((__int64)&v400, v401, &v464);
          }
          else
          {
            if ( v401 )
            {
              *v401 = v63 - 24;
              v64 = v401;
            }
            v401 = v64 + 1;
          }
          v63 = *(_QWORD *)(v63 + 8);
        }
        v403 = 0;
        v404 = 0;
        v405 = 0;
        v406 = 0;
        v407 = 0;
        v408 = 0;
        sub_2A11C40((__int64)&v403, &v392);
        sub_2A11C40((__int64)&v406, &v393);
        sub_D33BC0((__int64)v436, (__int64)v390[0]);
        sub_D4E470(v436, a2);
        v68 = 0x7FFFFFFFFFFFFFF8LL;
        v409 = 0;
        v410 = 0;
        v349 = v438[1];
        v411 = 0;
        v362 = v438[0];
        v69 = v390[0];
        v70 = (const void *)v390[0][4];
        v71 = v390[0][5] - (_QWORD)v70;
        if ( v71 <= 0x7FFFFFFFFFFFFFF8LL )
        {
          v72 = 0;
          if ( v71 )
          {
            v73 = (__int64 *)sub_22077B0(v390[0][5] - (_QWORD)v70);
            v72 = (char *)v73 + v71;
            v409 = v73;
            v411 = (char *)v73 + v71;
            memcpy(v73, v70, v71);
            v69 = v390[0];
          }
          v410 = v72;
          v454 = v456;
          v452 = 0;
          v453 = 0;
          v455 = 0x400000000LL;
          v74 = (void **)v69[2];
          v451 = 0;
          v75 = (void **)v69[1];
          v450 = 0;
          while ( v74 != v75 )
          {
            v76 = (const char *)*v75++;
            v464 = v76;
            sub_2A15120((__int64)&v450, (__int64 *)&v464, v68, v65, v66, v67);
          }
          if ( (unsigned __int8)sub_B921D0(*((_QWORD *)v392 + 9)) )
          {
            v79 = v390[0];
            if ( LOBYTE(qword_4F813A8[8]) || (v80 = v390[0][4], v379 = v390[0][5], v379 == v80) )
            {
LABEL_131:
              v88 = v79[4];
              v89 = v79[5];
              v439 = (__int64 *)v441;
              v90 = (v89 - v88) >> 3;
              v440 = 0x600000000LL;
              sub_F46230(v88, v90, (__int64)&v439, v77, v78);
              v360 = (__int64 *)*((_QWORD *)v393 + 4);
              if ( (_DWORD)a7 != 1 )
              {
                v365 = 1;
                while ( 1 )
                {
                  v460.m128i_i64[0] = 0;
                  v460.m128i_i64[1] = 1;
                  v457 = (unsigned __int64 *)v459;
                  v458 = 0x800000000LL;
                  v93 = (void **)&v461;
                  do
                  {
                    *v93 = (void *)-4096LL;
                    v93 += 2;
                  }
                  while ( v93 != (void **)&v464 );
                  v94 = v390[0];
                  *sub_2A127E0((__int64)&v460, (__int64 *)v390) = v94;
                  if ( v349 != v362 )
                    break;
LABEL_192:
                  sub_F45F60((__int64)v457, (unsigned int)v458, (__int64)&v442);
                  v138 = v457;
                  v139 = &v457[(unsigned int)v458];
                  if ( v139 != v457 )
                  {
                    while ( 1 )
                    {
                      v140 = *(_QWORD *)(*v138 + 56);
                      v141 = *v138 + 48;
                      if ( v141 != v140 )
                        break;
LABEL_204:
                      if ( v139 == ++v138 )
                        goto LABEL_205;
                    }
                    while ( v140 )
                    {
                      if ( *(_BYTE *)(v140 - 24) == 85
                        && (v142 = *(_QWORD *)(v140 - 56)) != 0
                        && !*(_BYTE *)v142
                        && (v143 = *(_QWORD *)(v140 + 56), *(_QWORD *)(v142 + 24) == v143)
                        && (*(_BYTE *)(v142 + 33) & 0x20) != 0
                        && *(_DWORD *)(v142 + 36) == 11 )
                      {
                        sub_CFEAE0(a5, v140 - 24, v135, v143, v136, v137);
                        v140 = *(_QWORD *)(v140 + 8);
                        if ( v141 == v140 )
                          goto LABEL_204;
                      }
                      else
                      {
                        v140 = *(_QWORD *)(v140 + 8);
                        if ( v141 == v140 )
                          goto LABEL_204;
                      }
                    }
LABEL_609:
                    BUG();
                  }
LABEL_205:
                  v464 = "It";
                  LOWORD(v468) = 2307;
                  LODWORD(v466) = v365;
                  sub_CA0F50((__int64 *)&v425, (void **)&v464);
                  v350 = v425;
                  v351 = v426;
                  v144 = (__int64 *)sub_AA48A0((__int64)v392);
                  v90 = (unsigned int)v440;
                  sub_F4CD20(v439, (unsigned int)v440, (__int64)v457, (unsigned int)v458, v144, v145, v350, v351);
                  sub_2240A30((unsigned __int64 *)&v425);
                  v92 = v341;
                  if ( (v460.m128i_i8[8] & 1) == 0 )
                  {
                    v90 = 16LL * (unsigned int)v462;
                    sub_C7D6A0(v461, v90, 8);
                  }
                  if ( v457 != (unsigned __int64 *)v459 )
                    _libc_free((unsigned __int64)v457);
                  if ( (_DWORD)a7 == ++v365 )
                    goto LABEL_210;
                }
                v380 = v349;
                while ( 1 )
                {
                  v464 = 0;
                  v95 = sub_AF1560(0x56u);
                  LODWORD(v467) = v95;
                  if ( v95 )
                  {
                    v465 = sub_C7D670((unsigned __int64)v95 << 6, 8);
                    sub_23FE7B0((__int64)&v464);
                  }
                  else
                  {
                    v465 = 0;
                    v466 = 0;
                  }
                  v425 = ".";
                  LOWORD(v429) = 2307;
                  v472 = 0;
                  LODWORD(v427) = v365;
                  v96 = sub_F4B360(*(_QWORD *)(v380 - 8), (__int64)&v464, (__int64 *)&v425, 0, 0);
                  v395 = v96;
                  v97 = *((_QWORD *)v392 + 9);
                  sub_B2B790(v97 + 72, v96);
                  v98 = *v360;
                  v99 = *(_QWORD *)(v96 + 24) & 7LL;
                  *(_QWORD *)(v96 + 32) = v360;
                  v98 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v96 + 24) = v98 | v99;
                  *(_QWORD *)(v98 + 8) = v96 + 24;
                  *v360 = *v360 & 7 | (v96 + 24);
                  sub_AA4C30(v96, *(_BYTE *)(v97 + 128));
                  v396 = sub_2A12AD0(*(_QWORD *)(v380 - 8), v395, a2, (__int64)&v460);
                  if ( v396 )
                  {
                    v100 = sub_2A127E0((__int64)&v460, (__int64 *)&v396);
                    sub_2A15120((__int64)&v450, v100, v101, v102, v103, v104);
                  }
                  v105 = *(char **)(v380 - 8);
                  if ( v105 != v392 )
                    goto LABEL_142;
                  v195 = v400;
                  if ( v401 != v400 )
                  {
                    v386 = v401;
                    do
                    {
                      v196 = *v195;
                      v198 = sub_2A14BC0((__int64)&v464, *v195)[2];
                      v199 = *(_QWORD *)(v198 - 8);
                      v200 = *(_DWORD *)(v198 + 4) & 0x7FFFFFF;
                      if ( (*(_DWORD *)(v198 + 4) & 0x7FFFFFF) != 0 )
                      {
                        v201 = 0;
                        v197 = v199 + 32LL * *(unsigned int *)(v198 + 72);
                        while ( v393 != *(char **)(v197 + 8 * v201) )
                        {
                          if ( (_DWORD)v200 == (_DWORD)++v201 )
                            goto LABEL_336;
                        }
                        v202 = 32 * v201;
                      }
                      else
                      {
LABEL_336:
                        v202 = 0x1FFFFFFFE0LL;
                      }
                      v203 = *(_QWORD *)(v199 + v202);
                      if ( *(_BYTE *)v203 > 0x1Cu
                        && v365 > 1
                        && (unsigned __int8)sub_B19060((__int64)(v390[0] + 7), *(_QWORD *)(v203 + 40), v200, v197) )
                      {
                        v203 = sub_2A14BC0((__int64)&v442, v203)[2];
                      }
                      v204 = sub_2A14BC0((__int64)&v464, v196);
                      v205 = v204[2];
                      if ( v203 != v205 )
                      {
                        if ( v205 != 0 && v205 != -4096 && v205 != -8192 )
                          sub_BD60C0(v204);
                        v204[2] = v203;
                        if ( v203 != 0 && v203 != -4096 && v203 != -8192 )
                          sub_BD73F0((__int64)v204);
                      }
                      ++v195;
                      sub_B43D60((_QWORD *)v198);
                    }
                    while ( v386 != v195 );
                  }
                  if ( !a9 )
                  {
                    v105 = *(char **)(v380 - 8);
                    goto LABEL_142;
                  }
                  v206 = v467;
                  v207 = v465;
                  if ( !(_DWORD)v467 )
                    goto LABEL_357;
                  v426 = 2;
                  v427 = 0;
                  v428 = -4096;
                  v429 = 0;
                  v208 = ((_DWORD)v467 - 1) & (((unsigned int)a9 >> 9) ^ ((unsigned int)a9 >> 4));
                  v209 = v465 + (v208 << 6);
                  v210 = *(_QWORD *)(v209 + 24);
                  if ( a9 != v210 )
                    break;
LABEL_334:
                  v425 = (char *)&unk_49DB368;
                  sub_D68D70(&v426);
                  v211 = v464;
                  v212 = v465 + ((unsigned __int64)(unsigned int)v467 << 6);
LABEL_335:
                  sub_B43D60(*(_QWORD **)(v209 + 56));
                  v417 = v211;
                  v418 = v209;
                  v416 = (void **)&v464;
                  v419 = v212;
                  sub_2A11C70((__int64)&v464, v207, v213, v214, v215, v216, (int)&v464, (int)v211, (_QWORD *)v209);
                  v105 = *(char **)(v380 - 8);
LABEL_142:
                  v106 = v395;
                  v109 = sub_2A14BC0((__int64)&v442, (__int64)v105);
                  v110 = v109[2];
                  if ( v106 != v110 )
                  {
                    if ( v110 != 0 && v110 != -4096 && v110 != -8192 )
                      sub_BD60C0(v109);
                    v109[2] = v106;
                    if ( v106 != 0 && v106 != -4096 && v106 != -8192 )
                      sub_BD73F0((__int64)v109);
                  }
                  if ( (_DWORD)v466 )
                  {
                    v170 = v465;
                    v171 = (unsigned __int64)(unsigned int)v467 << 6;
                    v172 = v465 + v171;
                    if ( v465 != v465 + v171 )
                    {
                      while ( 1 )
                      {
                        v173 = *(_QWORD *)(v170 + 24);
                        if ( v173 != -8192 && v173 != -4096 )
                          break;
                        v170 += 64;
                        if ( v172 == v170 )
                          goto LABEL_150;
                      }
                      if ( v172 != v170 )
                      {
                        v174 = (_QWORD *)v170;
                        v175 = (_QWORD *)(v465 + v171);
                        while ( 1 )
                        {
                          v176 = v174[3];
                          v421 = 2;
                          v422 = 0;
                          if ( v176 )
                          {
                            v423 = v176;
                            if ( v176 != -4096 && v176 != -8192 )
                              sub_BD73F0((__int64)&v421);
                          }
                          else
                          {
                            v423 = 0;
                          }
                          v424 = &v442;
                          v420 = &unk_49DD7B0;
                          if ( (unsigned __int8)sub_F9E960((__int64)&v442, (__int64)&v420, &v397) )
                          {
                            v108 = (__int64)(v397 + 5);
                            v177 = v423;
                            goto LABEL_260;
                          }
                          v181 = v446;
                          v182 = v397;
                          ++v442;
                          v183 = v444 + 1;
                          v398 = v397;
                          if ( 4 * (v444 + 1) >= 3 * v446 )
                            break;
                          if ( v446 - v445 - v183 <= v446 >> 3 )
                            goto LABEL_292;
LABEL_279:
                          v426 = 2;
                          v427 = 0;
                          v428 = -4096;
                          v429 = 0;
                          v184 = v182[3] == -4096;
                          v444 = v183;
                          if ( !v184 )
                            --v445;
                          v374 = v182;
                          v425 = (char *)&unk_49DB368;
                          sub_D68D70(&v426);
                          v185 = v374;
                          v177 = v423;
                          v186 = v374[3];
                          if ( v186 != v423 )
                          {
                            v187 = v374 + 1;
                            if ( v186 != 0 && v186 != -4096 && v186 != -8192 )
                            {
                              v370 = v374;
                              v375 = v374 + 1;
                              sub_BD60C0(v375);
                              v177 = v423;
                              v185 = v370;
                              v187 = v375;
                            }
                            v185[3] = v177;
                            if ( v177 == -4096 || v177 == 0 || v177 == -8192 )
                            {
                              v177 = v423;
                            }
                            else
                            {
                              v376 = v185;
                              sub_BD6050(v187, v421 & 0xFFFFFFFFFFFFFFF8LL);
                              v177 = v423;
                              v185 = v376;
                            }
                          }
                          v188 = v424;
                          v108 = (__int64)(v185 + 5);
                          v185[5] = 6;
                          v185[6] = 0;
                          v185[4] = v188;
                          v185[7] = 0;
LABEL_260:
                          v420 = &unk_49DB368;
                          if ( v177 != 0 && v177 != -4096 && v177 != -8192 )
                          {
                            v372 = v108;
                            sub_BD60C0(&v421);
                            v108 = v372;
                          }
                          v178 = *(_QWORD *)(v108 + 16);
                          v179 = v174[7];
                          if ( v178 != v179 )
                          {
                            if ( v178 != -4096 && v178 != 0 && v178 != -8192 )
                            {
                              v373 = v108;
                              sub_BD60C0((_QWORD *)v108);
                              v179 = v174[7];
                              v108 = v373;
                            }
                            *(_QWORD *)(v108 + 16) = v179;
                            if ( v179 != -4096 && v179 != 0 && v179 != -8192 )
                              sub_BD6050((unsigned __int64 *)v108, v174[5] & 0xFFFFFFFFFFFFFFF8LL);
                          }
                          v174 += 8;
                          if ( v174 != v175 )
                          {
                            while ( 1 )
                            {
                              v180 = v174[3];
                              if ( v180 != -4096 && v180 != -8192 )
                                break;
                              v174 += 8;
                              if ( v175 == v174 )
                                goto LABEL_150;
                            }
                            if ( v175 != v174 )
                              continue;
                          }
                          goto LABEL_150;
                        }
                        v181 = 2 * v446;
LABEL_292:
                        sub_CF32C0((__int64)&v442, v181);
                        sub_F9E960((__int64)&v442, (__int64)&v420, &v398);
                        v182 = v398;
                        v183 = v444 + 1;
                        goto LABEL_279;
                      }
                    }
                  }
LABEL_150:
                  v111 = *(_QWORD *)(v380 - 8);
                  v112 = *(_QWORD *)(v111 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v112 != v111 + 48 )
                  {
                    if ( !v112 )
                      BUG();
                    v371 = v112 - 24;
                    if ( (unsigned int)*(unsigned __int8 *)(v112 - 24) - 30 <= 0xA )
                    {
                      v377 = sub_B46E30(v112 - 24);
                      if ( v377 )
                      {
                        v113 = 0;
                        while ( 1 )
                        {
                          v114 = sub_B46EC0(v371, v113);
                          v115 = v114;
                          if ( *((_BYTE *)v390[0] + 84) )
                          {
                            v116 = (_QWORD *)v390[0][8];
                            v117 = &v116[*((unsigned int *)v390[0] + 19)];
                            if ( v116 == v117 )
                              goto LABEL_217;
                            while ( v115 != *v116 )
                            {
                              if ( v117 == ++v116 )
                                goto LABEL_217;
                            }
                          }
                          else if ( !sub_C8CA60((__int64)(v390[0] + 7), v114) )
                          {
LABEL_217:
                            v149 = sub_AA5930(v115);
                            v385 = v150;
                            v151 = v149;
                            while ( v385 != v151 )
                            {
                              v152 = 0x1FFFFFFFE0LL;
                              v153 = *(_QWORD *)(v151 - 8);
                              v154 = *(_DWORD *)(v151 + 72);
                              v155 = *(_DWORD *)(v151 + 4) & 0x7FFFFFF;
                              if ( v155 )
                              {
                                v156 = 0;
                                do
                                {
                                  if ( *(_QWORD *)(v380 - 8) == *(_QWORD *)(v153 + 32LL * v154 + 8 * v156) )
                                  {
                                    v152 = 32 * v156;
                                    goto LABEL_223;
                                  }
                                  ++v156;
                                }
                                while ( v155 != (_DWORD)v156 );
                                v152 = 0x1FFFFFFFE0LL;
                              }
LABEL_223:
                              v157 = *(_QWORD *)(v153 + v152);
                              if ( v446 )
                              {
                                v158 = (v446 - 1) & (((unsigned int)v157 >> 9) ^ ((unsigned int)v157 >> 4));
                                v159 = &v443[8 * (unsigned __int64)v158];
                                v160 = v159[3];
                                if ( v157 != v160 )
                                {
                                  v167 = 1;
                                  if ( v160 == -4096 )
                                    goto LABEL_227;
                                  while ( 1 )
                                  {
                                    v168 = v167 + 1;
                                    v158 = (v446 - 1) & (v167 + v158);
                                    v159 = &v443[8 * (unsigned __int64)v158];
                                    v169 = v159[3];
                                    if ( v157 == v169 )
                                      break;
                                    v167 = v168;
                                    if ( v169 == -4096 )
                                      goto LABEL_227;
                                  }
                                }
                                if ( v159 != &v443[8 * (unsigned __int64)v446] )
                                  v157 = v159[7];
                              }
LABEL_227:
                              v161 = v395;
                              if ( v154 == v155 )
                              {
                                sub_B48D90(v151);
                                v153 = *(_QWORD *)(v151 - 8);
                                v155 = *(_DWORD *)(v151 + 4) & 0x7FFFFFF;
                              }
                              v162 = (v155 + 1) & 0x7FFFFFF;
                              *(_DWORD *)(v151 + 4) = v162 | *(_DWORD *)(v151 + 4) & 0xF8000000;
                              v163 = 32LL * (unsigned int)(v162 - 1) + v153;
                              if ( *(_QWORD *)v163 )
                              {
                                v164 = *(_QWORD *)(v163 + 8);
                                **(_QWORD **)(v163 + 16) = v164;
                                if ( v164 )
                                  *(_QWORD *)(v164 + 16) = *(_QWORD *)(v163 + 16);
                              }
                              *(_QWORD *)v163 = v157;
                              if ( v157 )
                              {
                                v165 = *(_QWORD *)(v157 + 16);
                                *(_QWORD *)(v163 + 8) = v165;
                                if ( v165 )
                                  *(_QWORD *)(v165 + 16) = v163 + 8;
                                *(_QWORD *)(v163 + 16) = v157 + 16;
                                *(_QWORD *)(v157 + 16) = v163;
                              }
                              *(_QWORD *)(*(_QWORD *)(v151 - 8)
                                        + 32LL * *(unsigned int *)(v151 + 72)
                                        + 8LL * ((*(_DWORD *)(v151 + 4) & 0x7FFFFFFu) - 1)) = v161;
                              sub_DACA20((__int64)a3, (__int64)v390[0], v151);
                              v166 = *(_QWORD *)(v151 + 32);
                              if ( !v166 )
                                goto LABEL_609;
                              v151 = 0;
                              if ( *(_BYTE *)(v166 - 24) == 84 )
                                v151 = v166 - 24;
                            }
                          }
                          if ( v377 == ++v113 )
                          {
                            v111 = *(_QWORD *)(v380 - 8);
                            break;
                          }
                        }
                      }
                    }
                  }
                  if ( v392 != (char *)v111 )
                    goto LABEL_163;
                  v193 = v404;
                  if ( v404 == v405 )
                  {
                    sub_9319A0((__int64)&v403, v404, &v395);
                    v111 = *(_QWORD *)(v380 - 8);
LABEL_163:
                    if ( v393 != (char *)v111 )
                      goto LABEL_164;
                    goto LABEL_307;
                  }
                  if ( v404 )
                  {
                    *(_QWORD *)v404 = v395;
                    v193 = v404;
                  }
                  v404 = v193 + 8;
                  v111 = *(_QWORD *)(v380 - 8);
                  if ( v393 != (char *)v111 )
                    goto LABEL_164;
LABEL_307:
                  v194 = v407;
                  if ( v407 != v408 )
                  {
                    v118 = v395;
                    if ( v407 )
                    {
                      *(_QWORD *)v407 = v395;
                      v194 = v407;
                    }
                    v407 = v194 + 8;
                    v111 = *(_QWORD *)(v380 - 8);
                    goto LABEL_165;
                  }
                  sub_9319A0((__int64)&v406, v407, &v395);
                  v111 = *(_QWORD *)(v380 - 8);
LABEL_164:
                  v118 = v395;
LABEL_165:
                  if ( v415 )
                  {
                    v119 = (v415 - 1) & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
                    v120 = &v413[12 * v119];
                    if ( v111 == *v120 )
                      goto LABEL_167;
                    v108 = 1;
                    if ( *v120 != -4096 )
                    {
                      while ( 1 )
                      {
                        v107 = (unsigned int)(v108 + 1);
                        v119 = (v415 - 1) & (v108 + v119);
                        v120 = &v413[12 * v119];
                        if ( v111 == *v120 )
                          break;
                        v108 = (unsigned int)v107;
                        if ( *v120 == -4096 )
                          goto LABEL_171;
                      }
LABEL_167:
                      if ( v120 != &v413[12 * v415] )
                      {
                        v121 = *((unsigned int *)v120 + 10);
                        if ( v121 + 1 > (unsigned __int64)*((unsigned int *)v120 + 11) )
                        {
                          sub_C8D5F0((__int64)(v120 + 4), v120 + 6, v121 + 1, 8u, v107, v108);
                          v121 = *((unsigned int *)v120 + 10);
                        }
                        *(_QWORD *)(v120[4] + 8 * v121) = v118;
                        v118 = v395;
                        ++*((_DWORD *)v120 + 10);
                      }
                    }
                  }
LABEL_171:
                  v122 = (unsigned int)v458;
                  v123 = (unsigned int)v458 + 1LL;
                  if ( v123 > HIDWORD(v458) )
                  {
                    sub_C8D5F0((__int64)&v457, v459, v123, 8u, v107, v108);
                    v122 = (unsigned int)v458;
                  }
                  v457[v122] = v118;
                  v124 = v410;
                  LODWORD(v458) = v458 + 1;
                  if ( v410 == v411 )
                  {
                    sub_9319A0((__int64)&v409, v410, &v395);
                  }
                  else
                  {
                    if ( v410 )
                    {
                      *(_QWORD *)v410 = v395;
                      v124 = v410;
                    }
                    v410 = v124 + 8;
                  }
                  v125 = *(_QWORD *)(v380 - 8);
                  if ( (char *)v125 == v392 )
                  {
                    sub_2A15760(a4, v395, *(_QWORD *)(v406 + 8LL * (v365 - 1)));
                  }
                  else
                  {
                    if ( v125 )
                    {
                      v126 = (unsigned int)(*(_DWORD *)(v125 + 44) + 1);
                      v127 = *(_DWORD *)(v125 + 44) + 1;
                    }
                    else
                    {
                      v126 = 0;
                      v127 = 0;
                    }
                    if ( v127 >= *(_DWORD *)(a4 + 32) )
                      goto LABEL_607;
                    v128 = sub_2A14BC0(
                             (__int64)&v442,
                             **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a4 + 24) + 8 * v126) + 8LL));
                    sub_2A15760(a4, v395, v128[2]);
                  }
                  if ( v472 )
                  {
                    v189 = v471;
                    v472 = 0;
                    if ( v471 )
                    {
                      v190 = v469;
                      v191 = &v469[2 * v471];
                      do
                      {
                        if ( *v190 != -8192 && *v190 != -4096 )
                        {
                          v192 = v190[1];
                          if ( v192 )
                            sub_B91220((__int64)(v190 + 1), v192);
                        }
                        v190 += 2;
                      }
                      while ( v191 != v190 );
                      v189 = v471;
                    }
                    sub_C7D6A0((__int64)v469, 16LL * v189, 8);
                  }
                  v129 = v467;
                  if ( (_DWORD)v467 )
                  {
                    v130 = (_QWORD *)v465;
                    v421 = 2;
                    v422 = 0;
                    v423 = -4096;
                    v420 = &unk_49DD7B0;
                    v425 = (char *)&unk_49DD7B0;
                    v131 = (_QWORD *)(v465 + ((unsigned __int64)(unsigned int)v467 << 6));
                    v132 = -4096;
                    v424 = 0;
                    v426 = 2;
                    v427 = 0;
                    v428 = -8192;
                    v429 = 0;
                    while ( 1 )
                    {
                      v133 = v130[3];
                      if ( v132 != v133 && v133 != v428 )
                        sub_D68D70(v130 + 5);
                      *v130 = &unk_49DB368;
                      v134 = v130 + 1;
                      v130 += 8;
                      sub_D68D70(v134);
                      if ( v131 == v130 )
                        break;
                      v132 = v423;
                    }
                    v425 = (char *)&unk_49DB368;
                    sub_D68D70(&v426);
                    v420 = &unk_49DB368;
                    sub_D68D70(&v421);
                    v129 = v467;
                  }
                  sub_C7D6A0(v465, (unsigned __int64)v129 << 6, 8);
                  v380 -= 8LL;
                  if ( v362 == v380 )
                    goto LABEL_192;
                }
                v223 = 1;
                while ( v210 != -4096 )
                {
                  LODWORD(v208) = (v467 - 1) & (v223 + v208);
                  v209 = v465 + ((unsigned __int64)(unsigned int)v208 << 6);
                  v210 = *(_QWORD *)(v209 + 24);
                  if ( a9 == v210 )
                    goto LABEL_334;
                  ++v223;
                }
                v425 = (char *)&unk_49DB368;
                sub_D68D70(&v426);
                v207 = v465;
                v206 = v467;
LABEL_357:
                v211 = v464;
                v209 = v207 + ((unsigned __int64)v206 << 6);
                v212 = v209;
                goto LABEL_335;
              }
LABEL_210:
              v146 = v401;
              for ( j = v400; v146 != j; ++j )
              {
                v148 = *j;
                if ( v364 == v366 )
                {
                  v90 = sub_F0A930(*j, v347);
                  sub_BD84D0(v148, v90);
                  sub_B43D60((_QWORD *)v148);
                }
                else if ( (unsigned int)a7 > 1 )
                {
                  if ( (*(_DWORD *)(v148 + 4) & 0x7FFFFFF) != 0 )
                  {
                    v217 = 0;
                    while ( 1 )
                    {
                      v218 = v217;
                      if ( v393 == *(char **)(*(_QWORD *)(v148 - 8) + 32LL * *(unsigned int *)(v148 + 72) + 8 * v217) )
                        break;
                      if ( (*(_DWORD *)(v148 + 4) & 0x7FFFFFF) == (_DWORD)++v217 )
                        goto LABEL_484;
                    }
                  }
                  else
                  {
LABEL_484:
                    v218 = -1;
                  }
                  v219 = sub_B48BF0(*j, v218, 0);
                  v222 = v219;
                  if ( *(_BYTE *)v219 > 0x1Cu
                    && (unsigned __int8)sub_B19060((__int64)(v390[0] + 7), *(_QWORD *)(v219 + 40), v220, v221) )
                  {
                    v222 = sub_2A14BC0((__int64)&v442, v222)[2];
                  }
                  v90 = v222;
                  sub_F0A850(v148, v222, *((_QWORD *)v407 - 1));
                }
              }
              v224 = v406;
              v225 = (__int64)&v407[-v406] >> 3;
              if ( (_DWORD)v225 )
              {
                v226 = 0;
                while ( 1 )
                {
                  v227 = ((int)v226 + 1) % (unsigned int)v225;
                  v228 = (unsigned __int8 *)sub_986580(*(_QWORD *)(v224 + 8 * v226));
                  v90 = *(_QWORD *)(v403 + 8 * v226++);
                  sub_B47210(v228, v90, *(_QWORD *)(v403 + 8 * v227));
                  if ( (unsigned int)v225 == v226 )
                    break;
                  v224 = v406;
                }
              }
              if ( (unsigned int)a7 > 1 && v352 != v353 )
              {
                v381 = v353;
                while ( 1 )
                {
                  v229 = *(_QWORD *)v381;
                  if ( *(_QWORD *)v381 )
                  {
                    v230 = (unsigned int)(*(_DWORD *)(v229 + 44) + 1);
                    v231 = *(_DWORD *)(v229 + 44) + 1;
                  }
                  else
                  {
                    v230 = 0;
                    v231 = 0;
                  }
                  if ( v231 >= *(_DWORD *)(a4 + 32) )
                  {
                    v464 = (const char *)&v466;
                    v465 = 0x1000000000LL;
                    BUG();
                  }
                  v232 = *(_QWORD *)(*(_QWORD *)(a4 + 24) + 8 * v230);
                  v464 = (const char *)&v466;
                  v233 = 0x1000000000LL;
                  v465 = 0x1000000000LL;
                  v91 = *(_QWORD *)(v232 + 24);
                  v234 = (__int64 **)(v91 + 8LL * *(unsigned int *)(v232 + 32));
                  if ( (__int64 **)v91 == v234 )
                  {
                    v239 = (const char *)&v466;
                    v387 = (const char *)&v466;
                  }
                  else
                  {
                    v235 = *(__int64 ***)(v232 + 24);
                    do
                    {
                      while ( 1 )
                      {
                        v236 = **v235;
                        v90 = v236;
                        if ( !(unsigned __int8)sub_B19060((__int64)(v390[0] + 7), v236, v230, v233) )
                          break;
                        if ( v234 == ++v235 )
                          goto LABEL_375;
                      }
                      v237 = (unsigned int)v465;
                      v233 = HIDWORD(v465);
                      v238 = (unsigned int)v465 + 1LL;
                      if ( v238 > HIDWORD(v465) )
                      {
                        v90 = (__int64)&v466;
                        sub_C8D5F0((__int64)&v464, &v466, v238, 8u, v91, v92);
                        v237 = (unsigned int)v465;
                      }
                      v230 = (__int64)v464;
                      ++v235;
                      *(_QWORD *)&v464[8 * v237] = v236;
                      LODWORD(v465) = v465 + 1;
                    }
                    while ( v234 != v235 );
LABEL_375:
                    v239 = v464;
                    v387 = &v464[8 * (unsigned int)v465];
                  }
                  v224 = (unsigned __int64)v393;
                  v240 = *(_QWORD *)(*(_QWORD *)(v229 + 72) + 80LL);
                  if ( v240 )
                    v240 -= 24;
                  if ( v229 != v240 && v393 != (char *)v240 )
                  {
                    v91 = *(unsigned int *)(a4 + 32);
                    v241 = 0;
                    v242 = (unsigned int)(*(_DWORD *)(v229 + 44) + 1);
                    if ( (unsigned int)v242 < (unsigned int)v91 )
                      v241 = *(_QWORD *)(*(_QWORD *)(a4 + 24) + 8 * v242);
                    if ( v393 )
                    {
                      v90 = (unsigned int)(*((_DWORD *)v393 + 11) + 1);
                      v224 = v90;
                    }
                    else
                    {
                      v90 = 0;
                      v224 = 0;
                    }
                    v243 = 0;
                    if ( (unsigned int)v91 > (unsigned int)v224 )
                      v243 = *(_QWORD *)(*(_QWORD *)(a4 + 24) + 8 * v90);
                    for ( ; v241 != v243; v241 = *(_QWORD *)(v241 + 8) )
                    {
                      if ( *(_DWORD *)(v241 + 16) < *(_DWORD *)(v243 + 16) )
                      {
                        v224 = v241;
                        v241 = v243;
                        v243 = v224;
                      }
                    }
                    v240 = *(_QWORD *)v243;
                  }
                  if ( v387 != v239 )
                    break;
LABEL_411:
                  if ( v239 != (const char *)&v466 )
                    _libc_free((unsigned __int64)v239);
                  v381 += 8;
                  if ( v352 == v381 )
                    goto LABEL_414;
                }
                v244 = v239;
                while ( 1 )
                {
                  v258 = *(_QWORD *)v244;
                  if ( v240 )
                  {
                    v90 = (unsigned int)(*(_DWORD *)(v240 + 44) + 1);
                    v245 = *(_DWORD *)(v240 + 44) + 1;
                  }
                  else
                  {
                    v90 = 0;
                    v245 = 0;
                  }
                  v246 = *(_DWORD *)(a4 + 32);
                  v247 = 0;
                  if ( v245 < v246 )
                    v247 = *(_QWORD *)(*(_QWORD *)(a4 + 24) + 8 * v90);
                  if ( v258 )
                  {
                    v248 = (unsigned int)(*(_DWORD *)(v258 + 44) + 1);
                    v249 = v248;
                  }
                  else
                  {
                    v248 = 0;
                    v249 = 0;
                  }
                  if ( v246 <= v249 )
                    break;
                  v250 = *(_QWORD *)(*(_QWORD *)(a4 + 24) + 8 * v248);
                  *(_BYTE *)(a4 + 112) = 0;
                  v224 = *(_QWORD *)(v250 + 8);
                  if ( v247 != v224 )
                  {
                    v460.m128i_i64[0] = v250;
                    v251 = *(_QWORD **)(v224 + 24);
                    v378 = v224;
                    v90 = (__int64)&v251[*(unsigned int *)(v224 + 32)];
                    v252 = sub_2A107D0(v251, v90, v460.m128i_i64);
                    v254 = v378;
                    v92 = (__int64)(v252 + 1);
                    if ( v252 + 1 != (_QWORD *)v90 )
                    {
                      v255 = v90 - v92;
                      v90 = (__int64)(v252 + 1);
                      memmove(v252, v252 + 1, v255);
                      v254 = v378;
                      v253 = *(_DWORD *)(v378 + 32);
                    }
                    v91 = (unsigned int)(v253 - 1);
                    *(_DWORD *)(v254 + 32) = v91;
                    *(_QWORD *)(v250 + 8) = v247;
                    v256 = *(unsigned int *)(v247 + 32);
                    v224 = *(unsigned int *)(v247 + 36);
                    if ( v256 + 1 > v224 )
                    {
                      v90 = v247 + 40;
                      sub_C8D5F0(v247 + 24, (const void *)(v247 + 40), v256 + 1, 8u, v91, v92);
                      v256 = *(unsigned int *)(v247 + 32);
                    }
                    v257 = *(_QWORD *)(v247 + 24);
                    *(_QWORD *)(v257 + 8 * v256) = v250;
                    ++*(_DWORD *)(v247 + 32);
                    if ( *(_DWORD *)(v250 + 16) != *(_DWORD *)(*(_QWORD *)(v250 + 8) + 16LL) + 1 )
                      sub_2A10A30(v250, v90, v257, v224, v91, v92);
                  }
                  v244 += 8;
                  if ( v387 == v244 )
                  {
                    v239 = v464;
                    goto LABEL_411;
                  }
                }
                *(_BYTE *)(a4 + 112) = 0;
LABEL_607:
                BUG();
              }
LABEL_414:
              v457 = (unsigned __int64 *)v459;
              v458 = 0x300000000LL;
              v425 = (char *)&v457;
              if ( (_DWORD)v414 )
              {
                v321 = v413;
                v322 = &v413[12 * v415];
                if ( v413 != v322 )
                {
                  while ( 1 )
                  {
                    v323 = v321;
                    LOBYTE(v224) = *v321 == -8192 || *v321 == -4096;
                    if ( !(_BYTE)v224 )
                      break;
                    v321 += 12;
                    if ( v322 == v321 )
                      goto LABEL_415;
                  }
                  if ( v321 != v322 )
                  {
                    while ( 1 )
                    {
                      v91 = *((unsigned int *)v323 + 10);
                      if ( (_DWORD)v91 )
                        break;
LABEL_561:
                      v323 += 12;
                      if ( v323 != v322 )
                      {
                        while ( *v323 == -8192 || *v323 == -4096 )
                        {
                          v323 += 12;
                          if ( v322 == v323 )
                            goto LABEL_415;
                        }
                        if ( v322 != v323 )
                          continue;
                      }
                      goto LABEL_415;
                    }
                    v389 = v322;
                    v324 = 0;
                    v325 = v323;
                    v326 = (unsigned int)v91;
                    v327 = v91;
                    while ( 1 )
                    {
                      while ( 1 )
                      {
                        v90 = *v325;
                        v328 = ((int)v324 + 1) % v327;
                        if ( v364 == v366 )
                          break;
                        v224 = BYTE5(a7);
                        if ( BYTE5(a7) )
                        {
                          if ( (char *)v90 == v393 )
                          {
                            v224 = 0;
                            if ( v328 )
                              goto LABEL_559;
                          }
LABEL_553:
                          if ( !v325[3] )
                            goto LABEL_576;
                          goto LABEL_554;
                        }
                        if ( v328 == *((_DWORD *)v325 + 4) )
                          goto LABEL_553;
                        v90 = *((unsigned int *)v325 + 3);
                        if ( !(_DWORD)v90 || v328 % (unsigned int)v90 )
                          goto LABEL_559;
                        if ( !v325[3] )
LABEL_576:
                          v325[3] = *(_QWORD *)(v325[4] + 8 * v324);
LABEL_554:
                        if ( v326 == ++v324 )
                          goto LABEL_560;
                      }
                      if ( v358 )
                      {
                        if ( v324 )
                        {
                          LOBYTE(v224) = v328 == 0;
                          if ( (char *)v90 == v393 || v328 )
                            goto LABEL_559;
                        }
                        goto LABEL_553;
                      }
                      if ( v328 )
                      {
                        v329 = *((_DWORD *)v325 + 2);
                        if ( v328 == v329 || !v329 )
                          goto LABEL_553;
                        LOBYTE(v224) = 0;
                      }
                      else
                      {
                        v224 = v355;
                        if ( (char *)v90 != v393 )
                          goto LABEL_553;
                      }
LABEL_559:
                      v90 = *(_QWORD *)(v325[4] + 8 * v324++);
                      sub_2A10DD0((__int64 *)&v425, v90, v224, *((_BYTE *)v325 + 20));
                      if ( v326 == v324 )
                      {
LABEL_560:
                        v323 = v325;
                        v322 = v389;
                        goto LABEL_561;
                      }
                    }
                  }
                }
              }
LABEL_415:
              v474 = 0;
              v475 = 0;
              v464 = (const char *)&v466;
              v465 = 0x1000000000LL;
              v477 = 0;
              v476 = a4;
              v478 = 1;
              v479 = 0;
              v480 = &v484;
              v481 = 8;
              v482 = 0;
              v483 = 1;
              v485 = 0;
              v486 = 0;
              v487 = 0;
              v488 = 0;
              if ( (_DWORD)v434 != 1 || (_DWORD)v414 != 1 )
              {
                v259 = (void **)&v464;
                v90 = (__int64)v457;
                sub_FFB3D0((__int64)&v464, v457, (unsigned int)v458, v224, v91, v92);
                goto LABEL_418;
              }
              v306 = v413;
              v307 = *v413;
              v308 = &v413[12 * v415];
              if ( v413 != v308 )
              {
                while ( 1 )
                {
                  v307 = *v306;
                  if ( *v306 != -4096 && v307 != -8192 )
                    break;
                  if ( v308 == v306 + 12 )
                  {
                    v307 = v306[12];
                    v306 = &v413[12 * v415];
                    break;
                  }
                  v306 += 12;
                }
              }
              if ( !v306[3] )
              {
                v224 = *((unsigned int *)v306 + 10);
                v306[3] = *(_QWORD *)(v306[4] + 8 * v224 - 8);
              }
              if ( v307 )
              {
                v260 = (unsigned int)(*(_DWORD *)(v307 + 44) + 1);
                v309 = v260;
              }
              else
              {
                v260 = 0;
                v309 = 0;
              }
              if ( v309 >= *(_DWORD *)(a4 + 32) )
                BUG();
              v310 = *(_QWORD *)(*(_QWORD *)(a4 + 24) + 8 * v260);
              v311 = *(unsigned int *)(v310 + 32);
              v312 = *(const void **)(v310 + 24);
              v460.m128i_i64[0] = (__int64)&v461;
              v313 = 8 * v311;
              v460.m128i_i64[1] = 0x600000000LL;
              if ( v311 > 6 )
              {
                sub_C8D5F0((__int64)&v460, &v461, v311, 8u, v91, v92);
                v314 = (__int64 *)(v460.m128i_i64[0] + 8LL * v460.m128i_u32[2]);
              }
              else
              {
                v314 = &v461;
                if ( !v313 )
                {
LABEL_525:
                  v460.m128i_i32[2] = v311 + v313;
                  v315 = (__int64 **)&v314[(unsigned int)(v311 + v313)];
                  if ( v315 != (__int64 **)v314 )
                  {
                    v316 = (__int64 **)v314;
                    do
                    {
                      v319 = *v316;
                      v90 = **v316;
                      if ( !(unsigned __int8)sub_B19060((__int64)(v390[0] + 7), v90, v260, v224) )
                      {
                        v320 = v306[3];
                        if ( v320 )
                        {
                          v317 = (unsigned int)(*(_DWORD *)(v320 + 44) + 1);
                          v318 = *(_DWORD *)(v320 + 44) + 1;
                        }
                        else
                        {
                          v317 = 0;
                          v318 = 0;
                        }
                        v90 = 0;
                        if ( v318 < *(_DWORD *)(a4 + 32) )
                          v90 = *(_QWORD *)(*(_QWORD *)(a4 + 24) + 8 * v317);
                        sub_B1AE50(v319, v90);
                      }
                      ++v316;
                    }
                    while ( v315 != v316 );
                    v314 = (__int64 *)v460.m128i_i64[0];
                  }
                  if ( v314 != &v461 )
                    _libc_free((unsigned __int64)v314);
                  v259 = 0;
LABEL_418:
                  if ( v356 != 1 && v357 )
                  {
                    v330 = sub_986580(*((_QWORD *)v407 - 1));
                    v90 = v363;
                    sub_F55BE0(v330, v363, 0, 0, v331, v332);
                  }
                  v261 = (__int64 *)v406;
                  v262 = (__int64 *)v407;
                  if ( v407 != (_BYTE *)v406 )
                  {
                    v263 = 0;
                    if ( !v259 )
                      v263 = a4;
                    v388 = v263;
                    do
                    {
                      v264 = sub_986580(*v261);
                      if ( *(_BYTE *)v264 == 31 )
                      {
                        v260 = *(_DWORD *)(v264 + 4) & 0x7FFFFFF;
                        if ( (_DWORD)v260 == 1 )
                        {
                          v265 = *(_QWORD *)(v264 - 32);
                          v382 = sub_AA5510(v265);
                          v266 = sub_F39690(v265, (__int64)v259, a2, 0, 0, 0, v388);
                          v224 = v342;
                          v90 = v345;
                          if ( v266 )
                          {
                            v267 = (__int64 *)v407;
                            v268 = (__int64 *)v406;
                            if ( (_BYTE *)v406 != v407 )
                            {
                              do
                              {
                                if ( v265 == *v268 )
                                  *v268 = v382;
                                ++v268;
                              }
                              while ( v267 != v268 );
                            }
                            v90 = v265;
                            sub_2A11AF0((char **)&v409, v265);
                          }
                        }
                      }
                      ++v261;
                    }
                    while ( v262 != v261 );
                  }
                  if ( v259 )
                    a4 = sub_FFD350((__int64)&v464, v90, v260, v224, v91, v92);
                  if ( v364 == v366 )
                  {
                    sub_2A13F00((__int64)v390[0], 0, a2, (__int64)a3, a4, a5, a6, a14);
                    v269 = (__int64)v390[0];
                    v270 = (_QWORD *)*v390[0];
                    sub_D4F720(a2, v390[0]);
                    v274 = v346;
                    v390[0] = 0;
                  }
                  else
                  {
                    if ( (unsigned int)a7 <= 1 )
                    {
                      sub_2A13F00((__int64)v390[0], 0, a2, (__int64)a3, a4, a5, a6, a14);
                      v340 = v390[0];
                      v272 = v344;
                      v269 = v346;
                    }
                    else
                    {
                      v269 = 1;
                      sub_2A13F00((__int64)v390[0], 1, a2, (__int64)a3, a4, a5, a6, a14);
                      v340 = v390[0];
                      v273 = v343;
                    }
                    v270 = (_QWORD *)*v390[0];
                    if ( BYTE4(v399) )
                    {
                      v269 = (unsigned int)v399 / (unsigned int)a7;
                      sub_F6ED70((__int64)v340, v269, v391);
                    }
                  }
                  if ( (_BYTE)qword_500A228 )
                  {
                    v269 = a4;
                    sub_D50AF0(a2);
                  }
                  if ( !v270 )
                  {
                    v338 = (unsigned __int64)v454;
                    v339 = &v454[8 * (unsigned int)v455];
                    if ( v339 != v454 )
                    {
                      do
                      {
                        v338 += 8LL;
                        v269 = a4;
                        sub_F6AC10(*(char **)(v338 - 8), a4, a2, (__int64)a3, a5, 0, v363);
                      }
                      while ( v339 != (_BYTE *)v338 );
                    }
                    goto LABEL_452;
                  }
                  if ( v363 && v364 == v366 )
                  {
                    if ( v359 )
                      goto LABEL_444;
                    v359 = sub_2A10C00(v270, &v409, a2);
                  }
                  if ( !v359 )
                    goto LABEL_451;
LABEL_444:
                  v275 = sub_D48290(a2, *((_QWORD *)v407 - 1));
                  if ( v270 != (_QWORD *)v275 )
                  {
                    if ( !v275 )
                    {
LABEL_580:
                      v333 = v270;
                      do
                      {
                        v279 = (__int64)v333;
                        v333 = (_QWORD *)*v333;
                      }
                      while ( (_QWORD *)v275 != v333 );
                      goto LABEL_450;
                    }
                    v278 = (_QWORD *)v275;
                    while ( 1 )
                    {
                      v278 = (_QWORD *)*v278;
                      if ( v270 == v278 )
                        break;
                      if ( !v278 )
                        goto LABEL_580;
                    }
                  }
                  v279 = (__int64)v270;
LABEL_450:
                  sub_11D2180(v279, a4, a2, (__int64)a3, v276, v277);
LABEL_451:
                  v269 = a4;
                  sub_F6AC10((char *)v270, a4, a2, (__int64)a3, a5, 0, v363);
                  v271 = v346;
LABEL_452:
                  v15 = (v364 == v366) + 1;
                  sub_FFCE90((__int64)&v464, v269, v271, v272, v274, v273);
                  sub_FFD870((__int64)&v464, v269, v280, v281, v282, v283);
                  sub_FFBC40((__int64)&v464, v269);
                  v284 = v487;
                  v285 = (unsigned __int64)v486;
                  if ( v487 != v486 )
                  {
                    v286 = v486;
                    do
                    {
                      v287 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v286[7];
                      *v286 = &unk_49E5048;
                      if ( v287 )
                        v287(v286 + 5, v286 + 5, 3);
                      v288 = v286 + 1;
                      v286 += 9;
                      *(v286 - 9) = &unk_49DB368;
                      sub_D68D70(v288);
                    }
                    while ( v284 != v286 );
                    v15 = (v364 == v366) + 1;
                    v285 = (unsigned __int64)v486;
                  }
                  if ( v285 )
                    j_j___libc_free_0(v285);
                  if ( !v483 )
                    _libc_free((unsigned __int64)v480);
                  if ( v464 != (const char *)&v466 )
                    _libc_free((unsigned __int64)v464);
                  if ( v457 != (unsigned __int64 *)v459 )
                    _libc_free((unsigned __int64)v457);
                  if ( v439 != (__int64 *)v441 )
                    _libc_free((unsigned __int64)v439);
                  if ( v454 != v456 )
                    _libc_free((unsigned __int64)v454);
                  sub_C7D6A0(v451, 8LL * (unsigned int)v453, 8);
                  sub_2A11AC0((unsigned __int64 *)&v409);
                  sub_2A11AC0(v438);
                  sub_C7D6A0(v436[2], 16LL * v437, 8);
                  sub_2A11AC0(&v406);
                  sub_2A11AC0(&v403);
                  if ( v400 )
                    j_j___libc_free_0((unsigned __int64)v400);
                  if ( v449 )
                  {
                    v334 = v448;
                    v449 = 0;
                    if ( v448 )
                    {
                      v335 = v447;
                      v336 = &v447[2 * v448];
                      do
                      {
                        if ( *v335 != -8192 && *v335 != -4096 )
                        {
                          v337 = v335[1];
                          if ( v337 )
                            sub_B91220((__int64)(v335 + 1), v337);
                        }
                        v335 += 2;
                      }
                      while ( v336 != v335 );
                      v334 = v448;
                    }
                    sub_C7D6A0((__int64)v447, 16 * v334, 8);
                  }
                  v289 = v446;
                  if ( v446 )
                  {
                    v290 = v443;
                    v460.m128i_i64[1] = 2;
                    v461 = 0;
                    v291 = -4096;
                    v292 = &v443[8 * (unsigned __int64)v446];
                    v462 = -4096;
                    v460.m128i_i64[0] = (__int64)&unk_49DD7B0;
                    v463[0] = 0;
                    v465 = 2;
                    v466 = 0;
                    v467 = -8192;
                    v464 = (const char *)&unk_49DD7B0;
                    v468 = 0;
                    while ( 1 )
                    {
                      v293 = v290[3];
                      if ( v291 != v293 && v293 != v467 )
                        sub_D68D70(v290 + 5);
                      *v290 = &unk_49DB368;
                      v294 = v290 + 1;
                      v290 += 8;
                      sub_D68D70(v294);
                      if ( v292 == v290 )
                        break;
                      v291 = v462;
                    }
                    v464 = (const char *)&unk_49DB368;
                    sub_D68D70(&v465);
                    v460.m128i_i64[0] = (__int64)&unk_49DB368;
                    sub_D68D70(&v460.m128i_i64[1]);
                    v289 = v446;
                  }
                  sub_C7D6A0((__int64)v443, v289 << 6, 8);
                  goto LABEL_64;
                }
              }
              v90 = (__int64)v312;
              memcpy(v314, v312, 8 * v311);
              v314 = (__int64 *)v460.m128i_i64[0];
              LODWORD(v313) = v460.m128i_i32[2];
              goto LABEL_525;
            }
            do
            {
              v81 = *(_QWORD *)(*(_QWORD *)v80 + 56LL);
              for ( k = *(_QWORD *)v80 + 48LL; v81 != k; v81 = *(_QWORD *)(v81 + 8) )
              {
                v82 = (__m128i *)(v81 - 24);
                if ( !v81 )
                  v82 = 0;
                if ( !sub_B46AA0((__int64)v82) )
                {
                  v83 = sub_B10CD0((__int64)v82[3].m128i_i64);
                  if ( v83 )
                  {
                    v84 = (const char *)sub_2A114C0(v83, a7);
                    v465 = v85;
                    v464 = v84;
                    if ( (_BYTE)v85 )
                    {
                      sub_B10CB0(&v460, (__int64)v464);
                      if ( &v82[3] != &v460 )
                      {
                        v86 = v82[3].m128i_i64[0];
                        if ( v86 )
                          sub_B91220((__int64)v82[3].m128i_i64, v86);
                        v87 = (unsigned __int8 *)v460.m128i_i64[0];
                        v82[3].m128i_i64[0] = v460.m128i_i64[0];
                        if ( v87 )
                        {
                          sub_B976B0((__int64)&v460, v87, (__int64)v82[3].m128i_i64);
                          v460.m128i_i64[0] = 0;
                        }
                      }
                      sub_9C6650(&v460);
                    }
                  }
                }
              }
              v80 += 8;
            }
            while ( v379 != v80 );
          }
          v79 = v390[0];
          goto LABEL_131;
        }
LABEL_106:
        sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
      }
      if ( BYTE4(a7) )
      {
        BYTE5(a7) = 0;
        if ( v14 )
        {
          v296 = *v14;
          v297 = sub_B2BE50(*v14);
          if ( sub_B6EA50(v297)
            || (v299 = sub_B2BE50(v296),
                v300 = sub_B6F970(v299),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v300 + 48LL))(v300)) )
          {
            v298 = v392;
            sub_B157E0((__int64)&v460, &v394);
            sub_B17430((__int64)&v464, (__int64)"loop-unroll", (__int64)"UnrollLoop", 10, &v460, (__int64)v298);
            sub_B18290((__int64)&v464, "    Note : cannot generate the remainder loop. ", 0x2Fu);
            sub_B18290((__int64)&v464, "Will unroll the main loop with side-exits that may hurt performance", 0x43u);
            v44 = (__int64)&v464;
            sub_1049740(v14, (__int64)&v464);
            v464 = (const char *)&unk_49D9D40;
            sub_23FD590((__int64)v473);
          }
        }
        goto LABEL_501;
      }
      if ( v14 )
        sub_2A11340(v14, &v394, (__int64 *)&v392);
    }
    v15 = 0;
  }
  else
  {
    v52 = (__int64)v42;
    v15 = 0;
    sub_D46CA0((__int64)v390[0], v52);
  }
LABEL_64:
  if ( v433 != (unsigned __int64 **)v435 )
    _libc_free((unsigned __int64)v433);
  v53 = v415;
  if ( v415 )
  {
    v54 = v413;
    v55 = &v413[12 * v415];
    do
    {
      if ( *v54 != -8192 && *v54 != -4096 )
      {
        v56 = v54[4];
        if ( (__int64 *)v56 != v54 + 6 )
          _libc_free(v56);
      }
      v54 += 12;
    }
    while ( v55 != v54 );
    v53 = v415;
  }
  sub_C7D6A0((__int64)v413, 96 * v53, 8);
  if ( v394 )
    sub_B91220((__int64)&v394, v394);
  if ( v353 )
    j_j___libc_free_0((unsigned __int64)v353);
  if ( v430 != (_QWORD *)v432 )
    _libc_free((unsigned __int64)v430);
  return v15;
}
