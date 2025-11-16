// Function: sub_331F6A0
// Address: 0x331f6a0
//
unsigned __int64 __fastcall sub_331F6A0(
        __int64 *a1,
        size_t a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6)
{
  __int64 v6; // rbp
  __int64 v7; // r12
  __int64 v8; // r13
  unsigned int v9; // r10d
  __int64 v10; // r8
  __int64 v12; // r12
  unsigned __int64 result; // rax
  unsigned __int64 v14; // r14
  int v15; // eax
  const __m128i *v16; // roff
  __int64 v17; // rbx
  __int64 v18; // rax
  unsigned __int16 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned int v22; // r9d
  _DWORD *v23; // rcx
  unsigned __int16 *v24; // rax
  __int64 v25; // r13
  unsigned __int16 v26; // r9
  __int64 v27; // r8
  __int32 v28; // eax
  __int64 v29; // rsi
  __int64 v30; // r12
  int v31; // r14d
  __int64 *v32; // rax
  __int64 v33; // rdi
  __int64 v34; // r12
  __int64 v35; // rbx
  __int64 (*v36)(); // rax
  unsigned __int16 v37; // r8
  __int64 v38; // r9
  const __m128i *v39; // roff
  __int64 v40; // r14
  __int64 v41; // r15
  __int64 i; // rbx
  __int64 v43; // rsi
  __int64 v44; // rax
  __int64 *v45; // rax
  unsigned __int16 v46; // r9
  __int64 v47; // r14
  __int64 v48; // r15
  __int64 v49; // r10
  __int64 v50; // r8
  __int64 v51; // rax
  __int16 v52; // ax
  __int64 v53; // rdi
  __int64 v54; // r11
  __int64 v55; // r10
  __int64 v56; // rax
  __int64 v57; // r14
  __int64 v58; // r15
  unsigned __int16 v59; // bx
  __int64 v60; // rax
  unsigned __int16 v61; // dx
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rsi
  __int64 v65; // rcx
  __int64 v66; // rbx
  __int64 v67; // r11
  __int64 v68; // r14
  __int64 v69; // rsi
  int v70; // r9d
  __int64 v71; // rbx
  __int64 v72; // rax
  unsigned __int16 v73; // ax
  __int64 v74; // rcx
  __int64 v75; // rdi
  int v76; // edx
  __int64 v77; // r14
  __int64 v78; // r8
  __int64 v79; // r9
  unsigned __int16 *v80; // rdx
  __int64 *v81; // rcx
  __int64 v82; // rax
  __int64 v83; // r9
  __int64 v84; // r12
  __int64 v85; // rax
  unsigned __int64 v86; // rbx
  int v87; // ecx
  unsigned __int16 *v88; // rax
  __int64 v89; // rsi
  __int64 v90; // r8
  int v91; // ebx
  __int32 v92; // eax
  __int64 v93; // rdi
  int v94; // r9d
  __int64 v95; // rcx
  __int64 *v96; // rax
  __int64 v97; // rsi
  __int64 v98; // r14
  __int64 v99; // r15
  __int64 v100; // r10
  __int64 v101; // r11
  __int64 v102; // rax
  __int64 v103; // r13
  __int32 v104; // edx
  __int64 v105; // r8
  int v106; // ecx
  __int64 v107; // rsi
  __int64 v108; // rax
  __int64 v109; // r12
  __int64 v110; // rax
  __int64 v111; // r14
  unsigned __int64 v112; // r15
  __int32 v113; // r13d
  unsigned __int16 *v114; // rax
  __int64 v115; // rsi
  __int64 v116; // r8
  int v117; // ecx
  unsigned __int64 v118; // r14
  __int64 v119; // rax
  __int64 v120; // r13
  __int32 v121; // edx
  __int64 v122; // r8
  int v123; // ecx
  __int64 v124; // rsi
  _QWORD *v125; // rax
  __int64 v126; // r14
  __int64 v127; // r15
  int v128; // r9d
  __int64 v129; // r13
  __int64 v130; // rcx
  int v131; // r8d
  __int64 v132; // r12
  __int64 v133; // r14
  int v134; // edx
  __int64 v135; // r15
  __int64 v136; // r8
  __int64 v137; // r9
  int v138; // edx
  __int64 v139; // rax
  int v140; // eax
  unsigned __int64 v141; // rsi
  __int64 v142; // rdx
  char v143; // r8
  unsigned __int64 v144; // rax
  __int64 v145; // rdx
  __int64 v146; // r8
  unsigned __int16 v147; // bx
  unsigned int v148; // r14d
  __int64 v149; // rax
  __m128i v150; // rax
  unsigned __int64 v151; // rdx
  __int64 v152; // r11
  unsigned int *v153; // rdx
  __int64 v154; // rdi
  __int64 v155; // r14
  bool (__fastcall *v156)(__int64, unsigned int, __int64, __int64, unsigned __int16); // rbx
  unsigned __int16 *v157; // rcx
  unsigned int v158; // edx
  unsigned __int16 v159; // bx
  __int64 v160; // rdx
  __int64 v161; // rax
  __int64 v162; // rdx
  int v163; // r11d
  __int64 v164; // r8
  __int64 v165; // r9
  __int16 v166; // ax
  __int64 v167; // r10
  __int64 v168; // rsi
  __int64 v169; // r13
  __int64 v170; // rcx
  int v171; // r8d
  __int64 v172; // rcx
  __int64 v173; // rax
  unsigned int v174; // eax
  unsigned __int16 v175; // r15
  __int64 v176; // r14
  __int64 v177; // rax
  __int64 v178; // rax
  __int64 v179; // rdx
  int v180; // eax
  unsigned int *v181; // r14
  __int64 v182; // rdi
  __int64 v183; // rax
  unsigned __int16 v184; // dx
  __int64 v185; // r8
  __int64 v186; // r9
  __int64 v187; // rsi
  int v188; // ecx
  __int128 v189; // rax
  int v190; // r9d
  __int64 v191; // rsi
  __int64 v192; // r14
  __int64 v193; // r8
  int v194; // r13d
  int v195; // eax
  int v196; // eax
  bool v197; // cc
  bool v198; // cf
  __int64 *v199; // rdx
  __int64 v200; // rax
  __int64 v201; // rdi
  __int64 v202; // rax
  char v203; // al
  __int64 v204; // rsi
  __int64 v205; // r9
  __int128 *v206; // r15
  __int128 *v207; // r13
  __int64 v208; // rdi
  __int64 v209; // rsi
  __int64 v210; // r14
  __int64 v211; // rdi
  __int64 v212; // rdx
  __int64 v213; // rcx
  __int64 v214; // r8
  int v215; // r9d
  __int64 v216; // r10
  __int64 v217; // r11
  int v218; // r8d
  int v219; // eax
  int v220; // ecx
  int v221; // r9d
  unsigned __int16 *v222; // rcx
  __int64 v223; // rdx
  __int64 v224; // r15
  unsigned __int16 v225; // si
  __int64 v226; // r8
  __int64 v227; // rdx
  __int64 v228; // rdx
  __int64 v229; // rcx
  __int64 v230; // r8
  __int64 v231; // rdx
  __int64 v232; // rdx
  __int64 v233; // rcx
  __int64 v234; // r8
  __int64 v235; // rax
  unsigned __int64 v236; // rdx
  _QWORD *v237; // rax
  int v238; // edi
  _DWORD *v239; // r15
  __int64 v240; // rax
  bool v241; // al
  __int64 v242; // rdx
  __int64 v243; // rcx
  __int64 v244; // r8
  unsigned __int16 v245; // ax
  __int64 v246; // rax
  int v247; // edx
  char v248; // al
  __int64 v249; // rsi
  __int64 v250; // r13
  __int64 v251; // r14
  int v252; // ebx
  unsigned __int16 *v253; // rax
  __int64 v254; // rdi
  __int64 v255; // r8
  __int64 v256; // rcx
  __int64 v257; // rdx
  __int64 v258; // rdi
  __int64 v259; // r8
  __int64 v260; // rcx
  __int64 v261; // rax
  int v262; // edx
  int v263; // r14d
  __int64 v264; // rcx
  int v265; // edx
  __int64 v266; // rsi
  __int64 v267; // r13
  __int64 v268; // r14
  int v269; // ebx
  bool v270; // al
  __int64 v271; // rdx
  __int64 v272; // rcx
  unsigned __int16 v273; // ax
  __int64 v274; // rdx
  __int64 v275; // rax
  __int64 v276; // r13
  __int64 v277; // rcx
  __int64 v278; // rdx
  __int64 v279; // r12
  __int64 v280; // rdi
  __int64 v281; // r10
  __int64 v282; // r11
  __int64 v283; // rbx
  __int64 v284; // rax
  _QWORD *v285; // rdx
  __int64 v286; // r14
  __int64 v287; // rdx
  __int64 v288; // r12
  __int128 v289; // [rsp-188h] [rbp-188h]
  __int128 v290; // [rsp-178h] [rbp-178h]
  __int128 v291; // [rsp-178h] [rbp-178h]
  __int128 v292; // [rsp-168h] [rbp-168h]
  __int128 v293; // [rsp-168h] [rbp-168h]
  __int128 v294; // [rsp-168h] [rbp-168h]
  int v295; // [rsp-168h] [rbp-168h]
  __int128 v296; // [rsp-168h] [rbp-168h]
  __int128 v297; // [rsp-168h] [rbp-168h]
  int v298; // [rsp-144h] [rbp-144h]
  int v299; // [rsp-140h] [rbp-140h]
  __int64 v300; // [rsp-138h] [rbp-138h]
  __int64 v301; // [rsp-138h] [rbp-138h]
  __int64 v302; // [rsp-138h] [rbp-138h]
  __int64 v303; // [rsp-138h] [rbp-138h]
  __int64 v304; // [rsp-130h] [rbp-130h]
  __int64 v305; // [rsp-130h] [rbp-130h]
  unsigned int v306; // [rsp-130h] [rbp-130h]
  unsigned int v307; // [rsp-130h] [rbp-130h]
  __int64 v308; // [rsp-130h] [rbp-130h]
  __int16 v309; // [rsp-130h] [rbp-130h]
  __int64 v310; // [rsp-130h] [rbp-130h]
  __int16 v311; // [rsp-12Eh] [rbp-12Eh]
  __int16 v312; // [rsp-128h] [rbp-128h]
  __int64 (__fastcall *v313)(__int64, __int64, __int64, __int64, __int64); // [rsp-128h] [rbp-128h]
  __int64 v314; // [rsp-128h] [rbp-128h]
  __int64 v315; // [rsp-128h] [rbp-128h]
  unsigned int v316; // [rsp-128h] [rbp-128h]
  unsigned int v317; // [rsp-128h] [rbp-128h]
  int v318; // [rsp-128h] [rbp-128h]
  unsigned int v319; // [rsp-128h] [rbp-128h]
  __int64 v320; // [rsp-128h] [rbp-128h]
  int v321; // [rsp-128h] [rbp-128h]
  __int64 v322; // [rsp-120h] [rbp-120h]
  __int64 v323; // [rsp-120h] [rbp-120h]
  __int64 v324; // [rsp-120h] [rbp-120h]
  int v325; // [rsp-120h] [rbp-120h]
  int v326; // [rsp-120h] [rbp-120h]
  char v327; // [rsp-120h] [rbp-120h]
  int v328; // [rsp-120h] [rbp-120h]
  unsigned int v329; // [rsp-120h] [rbp-120h]
  __int64 v330; // [rsp-120h] [rbp-120h]
  int v331; // [rsp-120h] [rbp-120h]
  __int64 v332; // [rsp-120h] [rbp-120h]
  unsigned int v333; // [rsp-120h] [rbp-120h]
  int v334; // [rsp-120h] [rbp-120h]
  __int64 v335; // [rsp-120h] [rbp-120h]
  __int64 v336; // [rsp-120h] [rbp-120h]
  __int64 v337; // [rsp-120h] [rbp-120h]
  __m128i v338; // [rsp-118h] [rbp-118h]
  int v339; // [rsp-118h] [rbp-118h]
  __int64 v340; // [rsp-118h] [rbp-118h]
  unsigned int v341; // [rsp-118h] [rbp-118h]
  int v342; // [rsp-118h] [rbp-118h]
  int v343; // [rsp-118h] [rbp-118h]
  int v344; // [rsp-118h] [rbp-118h]
  int v345; // [rsp-118h] [rbp-118h]
  int v346; // [rsp-118h] [rbp-118h]
  int v347; // [rsp-118h] [rbp-118h]
  __int64 v348; // [rsp-118h] [rbp-118h]
  unsigned __int16 v349; // [rsp-118h] [rbp-118h]
  int v350; // [rsp-118h] [rbp-118h]
  int v351; // [rsp-118h] [rbp-118h]
  int v352; // [rsp-118h] [rbp-118h]
  unsigned __int64 v353; // [rsp-118h] [rbp-118h]
  __int128 v354; // [rsp-118h] [rbp-118h]
  unsigned __int64 v355; // [rsp-118h] [rbp-118h]
  __int64 v356; // [rsp-118h] [rbp-118h]
  __int64 v357; // [rsp-118h] [rbp-118h]
  __int64 v358; // [rsp-110h] [rbp-110h]
  unsigned __int16 v359; // [rsp-108h] [rbp-108h]
  __int64 v360; // [rsp-108h] [rbp-108h]
  __int64 v361; // [rsp-108h] [rbp-108h]
  __int64 v362; // [rsp-108h] [rbp-108h]
  unsigned __int64 v363; // [rsp-108h] [rbp-108h]
  unsigned __int64 v364; // [rsp-108h] [rbp-108h]
  int v365; // [rsp-108h] [rbp-108h]
  int v366; // [rsp-108h] [rbp-108h]
  unsigned __int64 v367; // [rsp-108h] [rbp-108h]
  int v368; // [rsp-108h] [rbp-108h]
  int v369; // [rsp-108h] [rbp-108h]
  __int64 v370; // [rsp-108h] [rbp-108h]
  unsigned __int64 v371; // [rsp-108h] [rbp-108h]
  __int64 v372; // [rsp-108h] [rbp-108h]
  __int64 v373; // [rsp-108h] [rbp-108h]
  __int64 v374; // [rsp-108h] [rbp-108h]
  unsigned int v375; // [rsp-108h] [rbp-108h]
  __int128 v376; // [rsp-108h] [rbp-108h]
  int v377; // [rsp-108h] [rbp-108h]
  __int64 v378; // [rsp-108h] [rbp-108h]
  unsigned int v379; // [rsp-108h] [rbp-108h]
  __int64 v380; // [rsp-108h] [rbp-108h]
  __int64 v381; // [rsp-100h] [rbp-100h]
  __int64 v382; // [rsp-100h] [rbp-100h]
  unsigned __int16 v383; // [rsp-E8h] [rbp-E8h] BYREF
  __int64 v384; // [rsp-E0h] [rbp-E0h]
  unsigned __int16 v385; // [rsp-D8h] [rbp-D8h] BYREF
  __int64 v386; // [rsp-D0h] [rbp-D0h]
  unsigned __int16 v387; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v388; // [rsp-C0h] [rbp-C0h]
  unsigned __int64 v389; // [rsp-B8h] [rbp-B8h]
  __int64 v390; // [rsp-B0h] [rbp-B0h]
  __int64 v391; // [rsp-A8h] [rbp-A8h]
  __int64 v392; // [rsp-A0h] [rbp-A0h]
  unsigned __int16 v393; // [rsp-98h] [rbp-98h] BYREF
  __int64 v394; // [rsp-90h] [rbp-90h]
  __int64 v395; // [rsp-88h] [rbp-88h]
  unsigned __int64 v396; // [rsp-80h] [rbp-80h]
  unsigned __int16 v397; // [rsp-78h] [rbp-78h] BYREF
  __int64 v398; // [rsp-70h] [rbp-70h]
  __m128i v399; // [rsp-68h] [rbp-68h] BYREF
  unsigned __int64 v400; // [rsp-58h] [rbp-58h] BYREF
  __int64 v401; // [rsp-50h] [rbp-50h]
  __int64 v402; // [rsp-48h] [rbp-48h]
  int v403; // [rsp-40h] [rbp-40h]
  __int64 v404; // [rsp-28h] [rbp-28h]
  __int64 v405; // [rsp-20h] [rbp-20h]
  __int64 v406; // [rsp-8h] [rbp-8h] BYREF

  v9 = *(_DWORD *)(a2 + 24);
  v406 = v6;
  v10 = v9;
  v405 = v8;
  v404 = v7;
  v12 = a2;
  switch ( v9 )
  {
    case 0u:
    case 1u:
    case 6u:
    case 7u:
    case 8u:
    case 9u:
    case 0xAu:
    case 0xBu:
    case 0xCu:
    case 0xDu:
    case 0xEu:
    case 0xFu:
    case 0x10u:
    case 0x11u:
    case 0x12u:
    case 0x13u:
    case 0x14u:
    case 0x15u:
    case 0x16u:
    case 0x17u:
    case 0x18u:
    case 0x19u:
    case 0x1Au:
    case 0x1Bu:
    case 0x1Cu:
    case 0x1Du:
    case 0x1Eu:
    case 0x1Fu:
    case 0x20u:
    case 0x21u:
    case 0x22u:
    case 0x23u:
    case 0x24u:
    case 0x25u:
    case 0x26u:
    case 0x27u:
    case 0x28u:
    case 0x29u:
    case 0x2Au:
    case 0x2Bu:
    case 0x2Cu:
    case 0x2Du:
    case 0x2Eu:
    case 0x2Fu:
    case 0x30u:
    case 0x31u:
    case 0x32u:
    case 0x33u:
    case 0x35u:
    case 0x41u:
    case 0x42u:
    case 0x43u:
    case 0x5Cu:
    case 0x5Du:
    case 0x5Eu:
    case 0x5Fu:
    case 0x66u:
    case 0x67u:
    case 0x68u:
    case 0x69u:
    case 0x6Au:
    case 0x6Bu:
    case 0x6Cu:
    case 0x6Du:
    case 0x6Eu:
    case 0x6Fu:
    case 0x70u:
    case 0x71u:
    case 0x72u:
    case 0x73u:
    case 0x74u:
    case 0x75u:
    case 0x76u:
    case 0x77u:
    case 0x78u:
    case 0x79u:
    case 0x7Au:
    case 0x7Bu:
    case 0x7Cu:
    case 0x7Du:
    case 0x7Eu:
    case 0x7Fu:
    case 0x80u:
    case 0x81u:
    case 0x82u:
    case 0x83u:
    case 0x84u:
    case 0x85u:
    case 0x86u:
    case 0x87u:
    case 0x88u:
    case 0x89u:
    case 0x8Au:
    case 0x8Bu:
    case 0x8Cu:
    case 0x8Du:
    case 0x8Eu:
    case 0x8Fu:
    case 0x90u:
    case 0x91u:
    case 0x92u:
    case 0x93u:
    case 0x94u:
    case 0x95u:
    case 0x99u:
    case 0x9Bu:
    case 0xA2u:
    case 0xA3u:
    case 0xA4u:
    case 0xA6u:
    case 0xA8u:
    case 0xA9u:
    case 0xAAu:
    case 0xB8u:
    case 0xB9u:
    case 0xCAu:
    case 0xD2u:
    case 0xD3u:
    case 0xD4u:
    case 0xD9u:
    case 0xDAu:
    case 0xE4u:
    case 0xE5u:
    case 0xE7u:
    case 0xE8u:
    case 0xEBu:
    case 0xEEu:
    case 0xEFu:
    case 0xF2u:
    case 0xF3u:
    case 0xF7u:
    case 0xF8u:
    case 0xF9u:
    case 0xFAu:
    case 0xFBu:
    case 0xFCu:
    case 0xFDu:
    case 0xFEu:
    case 0xFFu:
    case 0x100u:
    case 0x102u:
    case 0x103u:
    case 0x104u:
    case 0x106u:
    case 0x107u:
    case 0x108u:
    case 0x109u:
    case 0x10Au:
    case 0x10Bu:
    case 0x10Eu:
    case 0x10Fu:
    case 0x110u:
    case 0x111u:
    case 0x119u:
    case 0x11Au:
    case 0x11Fu:
    case 0x120u:
    case 0x121u:
    case 0x122u:
    case 0x123u:
    case 0x124u:
    case 0x127u:
    case 0x128u:
    case 0x129u:
    case 0x12Cu:
    case 0x12Du:
    case 0x12Eu:
    case 0x12Fu:
    case 0x130u:
    case 0x133u:
    case 0x134u:
    case 0x135u:
    case 0x136u:
    case 0x137u:
    case 0x138u:
    case 0x139u:
    case 0x13Au:
    case 0x13Bu:
    case 0x13Cu:
    case 0x13Du:
    case 0x13Eu:
    case 0x13Fu:
    case 0x140u:
    case 0x141u:
    case 0x142u:
    case 0x143u:
    case 0x144u:
    case 0x145u:
    case 0x146u:
    case 0x147u:
    case 0x148u:
    case 0x149u:
    case 0x14Au:
    case 0x14Bu:
    case 0x14Cu:
    case 0x14Du:
    case 0x14Eu:
    case 0x14Fu:
    case 0x150u:
    case 0x151u:
    case 0x152u:
    case 0x154u:
    case 0x155u:
    case 0x156u:
    case 0x157u:
    case 0x158u:
    case 0x159u:
    case 0x15Au:
    case 0x15Bu:
    case 0x15Cu:
    case 0x15Du:
    case 0x15Eu:
    case 0x15Fu:
    case 0x160u:
    case 0x161u:
    case 0x162u:
    case 0x163u:
    case 0x164u:
    case 0x165u:
    case 0x166u:
    case 0x167u:
    case 0x168u:
    case 0x169u:
    case 0x16Eu:
    case 0x170u:
    case 0x171u:
    case 0x172u:
    case 0x173u:
    case 0x174u:
    case 0x175u:
    case 0x176u:
    case 0x177u:
    case 0x189u:
    case 0x18Au:
    case 0x1EDu:
    case 0x1EEu:
    case 0x1EFu:
    case 0x1F0u:
      return 0;
    case 2u:
      return sub_32C64A0((__int64)a1, a2, a3, a4, v9, a6);
    case 3u:
    case 4u:
      return sub_3280C60(a1, a2);
    case 5u:
      return sub_3267C60(a1, a2);
    case 0x34u:
      return sub_32B1800(a1, a2);
    case 0x36u:
      return sub_328E660(a1, a2, **(unsigned __int16 **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
    case 0x37u:
      return sub_3311FA0((__int64)a1, a2, a3, a4, v9);
    case 0x38u:
      return sub_32E0540(a1, a2);
    case 0x39u:
      return sub_32A1EF0(a1, a2);
    case 0x3Au:
      return sub_32DCC60(a1, a2);
    case 0x3Bu:
      return sub_32F3470(a1, a2);
    case 0x3Cu:
      return sub_32F8D30(a1, a2);
    case 0x3Du:
    case 0x3Eu:
      return sub_32F3B40(a1, a2);
    case 0x3Fu:
      return sub_3324F40();
    case 0x40u:
      return sub_3324950();
    case 0x44u:
      return sub_330AAF0(a1, a2, a3, a4, v9, a6);
    case 0x45u:
      return sub_330A840(a1, a2, a3, a4, v9, a6);
    case 0x46u:
      return sub_3267170(a1, a2);
    case 0x47u:
      v96 = *(__int64 **)(a2 + 40);
      v97 = v96[10];
      v98 = *v96;
      v99 = v96[1];
      v100 = v96[5];
      v101 = v96[6];
      result = 0;
      if ( *(_DWORD *)(v97 + 24) == 67 )
      {
        v169 = *a1;
        v170 = *(_QWORD *)(v12 + 48);
        v171 = *(_DWORD *)(v12 + 68);
        v400 = *(_QWORD *)(v12 + 80);
        if ( v400 )
        {
          v331 = v170;
          v350 = v171;
          v374 = v100;
          v381 = v101;
          sub_325F5D0((__int64 *)&v400);
          LODWORD(v170) = v331;
          v171 = v350;
          v100 = v374;
          v101 = v381;
        }
        *((_QWORD *)&v294 + 1) = v101;
        *(_QWORD *)&v294 = v100;
        *((_QWORD *)&v291 + 1) = v99;
        *(_QWORD *)&v291 = v98;
        LODWORD(v401) = *(_DWORD *)(v12 + 72);
        v332 = sub_3411F20(v169, 69, (unsigned int)&v400, v170, v171, a6, v291, v294);
        sub_9C6650(&v400);
        return v332;
      }
      return result;
    case 0x48u:
      return sub_330AD20((__int64)a1, a2, a3, a4, v9, a6);
    case 0x49u:
      return sub_3268010(a1, a2);
    case 0x4Au:
      return sub_326C5C0((__int64)a1, a2);
    case 0x4Bu:
      return sub_3268160(a1, a2);
    case 0x4Cu:
    case 0x4Du:
      return sub_3310D30(a1, a2, a3, a4, v9, a6);
    case 0x4Eu:
    case 0x4Fu:
      return sub_330B250(a1, a2, a3, a4, v9, a6);
    case 0x50u:
    case 0x51u:
      return sub_3311880(a1, a2);
    case 0x52u:
    case 0x53u:
      return sub_3297000(a1, a2);
    case 0x54u:
    case 0x55u:
      return sub_32972C0(a1, a2);
    case 0x56u:
    case 0x57u:
      return sub_3279310(a1, a2);
    case 0x58u:
    case 0x59u:
    case 0x5Au:
    case 0x5Bu:
      return sub_3269A00(a1, a2);
    case 0x60u:
      return sub_32B6540((__int64 **)a1, a2);
    case 0x61u:
      return sub_32B8A20(a1, a2);
    case 0x62u:
      return sub_32BAF50(a1, a2);
    case 0x63u:
      return sub_3302520(a1, (_QWORD *)a2);
    case 0x64u:
      return sub_329CC30(a1, a2);
    case 0x65u:
      return sub_3269080(a1, a2);
    case 0x96u:
      return sub_32BC110((__int64 **)a1, a2);
    case 0x97u:
      return sub_3261CD0(a1, a2);
    case 0x98u:
      return sub_32D0A20((__int64)a1, a2);
    case 0x9Au:
      return sub_327EA70(a1, a2);
    case 0x9Cu:
      return sub_32BE8D0(a1, a2);
    case 0x9Du:
      return sub_32CBCB0(a1, a2, a3, a4, v9, a6);
    case 0x9Eu:
      return sub_32EC4F0(a1, (_QWORD *)a2);
    case 0x9Fu:
      return sub_32983B0((__int64)a1, a2);
    case 0xA0u:
      return sub_32E81A0(a1, a2);
    case 0xA1u:
      return sub_3318640(a1, (_QWORD *)a2, a3, a4, v9);
    case 0xA5u:
      return sub_32E3060(a1, a2, a3, a4, v9, a6);
    case 0xA7u:
      return sub_32ABE50((__int64)a1, a2, a3, a4, v9);
    case 0xABu:
      return sub_326E5E0(a1, a2);
    case 0xACu:
      return sub_32DC600(a1, a2);
    case 0xADu:
      return sub_3297590(a1, (_QWORD *)a2);
    case 0xAEu:
    case 0xAFu:
    case 0xB0u:
    case 0xB1u:
      return sub_32A6850(a1, a2);
    case 0xB2u:
    case 0xB3u:
      return sub_3297B40(a1, a2);
    case 0xB4u:
    case 0xB5u:
    case 0xB6u:
    case 0xB7u:
      return sub_32DBF90(a1, a2);
    case 0xBAu:
      return sub_33067C0(a1, a2);
    case 0xBBu:
      return sub_32E0F40(a1, a2);
    case 0xBCu:
      return sub_32FA5C0(a1, a2);
    case 0xBDu:
      return sub_328B360(a1, a2);
    case 0xBEu:
      return sub_32D9F40(a1, a2);
    case 0xBFu:
      return sub_32D8F30(a1, a2);
    case 0xC0u:
      return sub_32D7740(a1, a2);
    case 0xC1u:
    case 0xC2u:
      return sub_32D6B50(a1, a2);
    case 0xC3u:
    case 0xC4u:
      return sub_32D5A50(a1, a2);
    case 0xC5u:
      return sub_328B5C0(a1, a2);
    case 0xC6u:
      return sub_32648A0(a1, (_QWORD *)a2);
    case 0xC7u:
      return sub_3264A00(a1, (_QWORD *)a2);
    case 0xC8u:
      return sub_327D4E0(a1, a2);
    case 0xC9u:
      return sub_32AD270(a1, a2);
    case 0xCBu:
      return sub_3261930(a1, a2);
    case 0xCCu:
      return sub_3261850(a1, a2);
    case 0xCDu:
      return sub_32F4AC0(a1, a2, a3, a4, v9, a6);
    case 0xCEu:
      return sub_32F0F50(a1, (_QWORD *)a2, a3, a4, v9, a6);
    case 0xCFu:
      return sub_32F0A50(a1, a2);
    case 0xD0u:
      return sub_32FCEF0(a1, a2);
    case 0xD1u:
      return sub_3261F80(a1, a2);
    case 0xD5u:
      return sub_330BE50(a1, (_QWORD *)a2, a3, a4, v9, a6);
    case 0xD6u:
      return sub_330DD30(a1, a2, a3, a4, v9, a6);
    case 0xD7u:
      return sub_33100E0(a1, a2);
    case 0xD8u:
      return sub_32D2680(a1, (_QWORD *)a2);
    case 0xDBu:
      v80 = *(unsigned __int16 **)(a2 + 48);
      v81 = *(__int64 **)(a2 + 40);
      v82 = *v80;
      v83 = *((_QWORD *)v80 + 1);
      v399.m128i_i16[0] = *v80;
      v399.m128i_i64[1] = v83;
      v84 = *v81;
      if ( *(_DWORD *)(*v81 + 24) != 227 )
        return 0;
      v152 = *a1;
      v153 = *(unsigned int **)(v84 + 40);
      v154 = *(_QWORD *)(*a1 + 16);
      v155 = *(_QWORD *)v153;
      v156 = *(bool (__fastcall **)(__int64, unsigned int, __int64, __int64, unsigned __int16))(*(_QWORD *)v154 + 1752LL);
      v370 = v153[2];
      if ( v156 != sub_2FE3620 )
      {
        a2 = 229;
        v157 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v153 + 48LL) + 16 * v370);
        if ( v156(v154, 229u, *v157, *((_QWORD *)v157 + 1), v399.m128i_u32[0]) )
        {
          v152 = *a1;
          goto LABEL_263;
        }
        return 0;
      }
      v158 = 1;
      if ( (_WORD)v82 != 1 )
      {
        if ( !(_WORD)v82 )
          return 0;
        v158 = (unsigned __int16)v82;
        if ( !*(_QWORD *)(v154 + 8 * v82 + 112) )
          return 0;
      }
      if ( (*(_BYTE *)(v154 + 500LL * v158 + 6643) & 0xFB) != 0 )
        return 0;
LABEL_263:
      v159 = v399.m128i_i16[0];
      if ( v399.m128i_i16[0] )
      {
        if ( (unsigned __int16)(v399.m128i_i16[0] - 17) > 0xD3u )
        {
LABEL_265:
          v160 = v399.m128i_i64[1];
          goto LABEL_266;
        }
        v159 = word_4456580[v399.m128i_u16[0] - 1];
        v160 = 0;
      }
      else
      {
        v356 = v152;
        v241 = sub_30070B0((__int64)&v399);
        v152 = v356;
        if ( !v241 )
          goto LABEL_265;
        v245 = sub_3009970((__int64)&v399, a2, v242, v243, v244);
        v152 = v356;
        v159 = v245;
      }
LABEL_266:
      v347 = v152;
      v161 = sub_33F7D60(v152, v159, v160);
      v163 = v347;
      v164 = v161;
      v165 = v162;
      v400 = *(_QWORD *)(v84 + 80);
      if ( v400 )
      {
        v358 = v162;
        v328 = v347;
        v348 = v161;
        sub_B96E90((__int64)&v400, v400, 1);
        v163 = v328;
        v164 = v348;
        v165 = v358;
      }
      *((_QWORD *)&v293 + 1) = v165;
      *(_QWORD *)&v293 = v164;
      LODWORD(v401) = *(_DWORD *)(v84 + 72);
      *((_QWORD *)&v290 + 1) = v370;
      *(_QWORD *)&v290 = v155;
      result = sub_3406EB0(v163, 229, (unsigned int)&v400, v399.m128i_i32[0], v399.m128i_i32[2], v165, v290, v293);
      if ( v400 )
      {
LABEL_269:
        v371 = result;
        sub_B91220((__int64)&v400, v400);
        return v371;
      }
      return result;
    case 0xDCu:
      return sub_3276C70(a1, a2);
    case 0xDDu:
      return sub_326EF70(a1, a2);
    case 0xDEu:
      return sub_32F93A0(a1, a2);
    case 0xDFu:
    case 0xE0u:
    case 0xE1u:
      return sub_32EA290((__int64)a1, a2, a3, a4, v9, a6);
    case 0xE2u:
      return sub_3276A10(a1, a2);
    case 0xE3u:
      return sub_3276B40(a1, a2);
    case 0xE6u:
      return sub_32BD400((__int64)a1, a2);
    case 0xE9u:
      return sub_3311290(a1, a2, a3, a4, v9);
    case 0xEAu:
      return sub_32C3760(a1, a2, a3, a4, v9, a6);
    case 0xECu:
    case 0xF0u:
      return sub_326BC50(a1, a2);
    case 0xEDu:
      v95 = **(_QWORD **)(a2 + 40);
      result = 0;
      if ( *(_DWORD *)(v95 + 24) == 236 )
        return **(_QWORD **)(v95 + 40);
      return result;
    case 0xF1u:
      v95 = **(_QWORD **)(a2 + 40);
      result = 0;
      if ( *(_DWORD *)(v95 + 24) == 240 )
        return **(_QWORD **)(v95 + 40);
      return result;
    case 0xF4u:
      return sub_32CB720(a1, a2);
    case 0xF5u:
      v85 = *(_QWORD *)(a2 + 40);
      v86 = *(_QWORD *)v85;
      v87 = *(_DWORD *)(v85 + 8);
      v88 = *(unsigned __int16 **)(a2 + 48);
      v89 = *(_QWORD *)(a2 + 80);
      v363 = v86;
      v90 = *((_QWORD *)v88 + 1);
      v91 = *v88;
      v399.m128i_i64[0] = v89;
      if ( v89 )
      {
        v325 = v90;
        v342 = v87;
        sub_B96E90((__int64)&v399, v89, 1);
        LODWORD(v90) = v325;
        v87 = v342;
      }
      v92 = *(_DWORD *)(v12 + 72);
      LODWORD(v401) = v87;
      v93 = *a1;
      v399.m128i_i32[2] = v92;
      v343 = v90;
      v400 = v363;
      result = sub_3402EA0(v93, 245, (unsigned int)&v406 - 96, v91, v90, 0, (__int64)&v400, 1);
      if ( !result )
      {
        v195 = *(_DWORD *)(v363 + 24);
        if ( v195 == 245 )
        {
          result = **(_QWORD **)(v12 + 40);
        }
        else if ( v195 == 244 || v195 == 152 )
        {
          result = sub_33FAF80(*a1, 245, (unsigned int)&v406 - 96, v91, v343, v94, *(_OWORD *)*(_QWORD *)(v363 + 40));
        }
        else
        {
          result = sub_32CAE50(a1, v12);
        }
      }
      if ( v399.m128i_i64[0] )
      {
        v364 = result;
        sub_B91220((__int64)&v399, v399.m128i_i64[0]);
        return v364;
      }
      return result;
    case 0xF6u:
      v31 = *(_DWORD *)(a2 + 28);
      if ( (v31 & 0x400) != 0
        && ((*(_BYTE *)(*(_QWORD *)*a1 + 864LL) & 2) != 0 || (v31 & 0x40) != 0)
        && ((v32 = *(__int64 **)(a2 + 40),
             v33 = a1[1],
             v34 = *v32,
             v35 = v32[1],
             v36 = *(__int64 (**)())(*(_QWORD *)v33 + 216LL),
             v36 == sub_2FE2F50)
         || !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64))v36)(v33, v34, v35)) )
      {
        return sub_32C9810((__int64)a1, v34, v35, v31, 0);
      }
      else
      {
        return 0;
      }
    case 0x101u:
      return sub_327EC00(a1, a2);
    case 0x105u:
      v125 = *(_QWORD **)(a2 + 40);
      v126 = *v125;
      v127 = v125[1];
      if ( !(unsigned __int8)sub_33E2470(*a1, *v125, v127) )
        return 0;
      v129 = *a1;
      v130 = *(_QWORD *)(a2 + 48);
      v131 = *(_DWORD *)(a2 + 68);
      v400 = *(_QWORD *)(a2 + 80);
      if ( v400 )
      {
        v346 = v130;
        v369 = v131;
        sub_325F5D0((__int64 *)&v400);
        LODWORD(v130) = v346;
        v131 = v369;
      }
      *((_QWORD *)&v292 + 1) = v127;
      *(_QWORD *)&v292 = v126;
      LODWORD(v401) = *(_DWORD *)(a2 + 72);
      v132 = sub_3411EF0(v129, 261, (unsigned int)&v400, v130, v131, v128, v292);
      if ( v400 )
        sub_B91220((__int64)&v400, v400);
      return v132;
    case 0x10Cu:
      v119 = *(_QWORD *)(a2 + 40);
      v120 = *a1;
      v121 = *(_DWORD *)(v119 + 8);
      v122 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
      v123 = **(unsigned __int16 **)(a2 + 48);
      v399.m128i_i64[0] = *(_QWORD *)v119;
      v124 = *(_QWORD *)(a2 + 80);
      v399.m128i_i32[2] = v121;
      v400 = v124;
      if ( v124 )
      {
        v345 = v123;
        v368 = v122;
        sub_B96E90((__int64)&v400, v124, 1);
        v123 = v345;
        LODWORD(v122) = v368;
      }
      LODWORD(v401) = *(_DWORD *)(v12 + 72);
      v108 = sub_3402EA0(v120, 268, (unsigned int)&v400, v123, v122, 0, (__int64)&v399, 1);
      goto LABEL_196;
    case 0x10Du:
      v110 = *(_QWORD *)(a2 + 40);
      v111 = *a1;
      v112 = *(_QWORD *)v110;
      v113 = *(_DWORD *)(v110 + 8);
      v114 = *(unsigned __int16 **)(a2 + 48);
      v115 = *(_QWORD *)(a2 + 80);
      v116 = *((_QWORD *)v114 + 1);
      v117 = *v114;
      v399.m128i_i64[0] = v112;
      v399.m128i_i32[2] = v113;
      v400 = v115;
      if ( v115 )
      {
        v326 = v117;
        v366 = v116;
        sub_B96E90((__int64)&v400, v115, 1);
        v117 = v326;
        LODWORD(v116) = v366;
      }
      LODWORD(v401) = *(_DWORD *)(v12 + 72);
      result = sub_3402EA0(v111, 269, (unsigned int)&v400, v117, v116, 0, (__int64)&v399, 1);
      v118 = result;
      if ( v400 )
      {
        v367 = result;
        sub_B91220((__int64)&v400, v400);
        result = v367;
      }
      if ( !v118 )
      {
        v196 = *(_DWORD *)(v112 + 24);
        if ( v196 > 271 )
        {
          v198 = (unsigned int)(v196 - 273) < 2;
          result = 0;
          if ( v198 )
            return v112;
        }
        else
        {
          v197 = v196 < 268;
          result = v112;
          if ( v197 )
            return 0;
        }
      }
      return result;
    case 0x112u:
      v102 = *(_QWORD *)(a2 + 40);
      v103 = *a1;
      v104 = *(_DWORD *)(v102 + 8);
      v105 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
      v106 = **(unsigned __int16 **)(a2 + 48);
      v399.m128i_i64[0] = *(_QWORD *)v102;
      v107 = *(_QWORD *)(a2 + 80);
      v399.m128i_i32[2] = v104;
      v400 = v107;
      if ( v107 )
      {
        v344 = v106;
        v365 = v105;
        sub_B96E90((__int64)&v400, v107, 1);
        v106 = v344;
        LODWORD(v105) = v365;
      }
      LODWORD(v401) = *(_DWORD *)(v12 + 72);
      v108 = sub_3402EA0(v103, 274, (unsigned int)&v400, v106, v105, 0, (__int64)&v399, 1);
LABEL_196:
      v109 = v108;
      if ( v400 )
        sub_B91220((__int64)&v400, v400);
      return v109;
    case 0x113u:
    case 0x114u:
    case 0x115u:
    case 0x116u:
      v23 = *(_DWORD **)(a2 + 40);
      v24 = *(unsigned __int16 **)(a2 + 48);
      v25 = *a1;
      v26 = *v24;
      v27 = *((_QWORD *)v24 + 1);
      if ( *(_DWORD *)(*(_QWORD *)v23 + 24LL) == 51 )
      {
        v208 = *a1;
        v400 = 0;
        LODWORD(v401) = 0;
        v30 = sub_33F17F0(v208, 51, &v400, v26, v27);
        if ( v400 )
          sub_B91220((__int64)&v400, v400);
      }
      else
      {
        v28 = v23[2];
        v29 = *(_QWORD *)(a2 + 80);
        v399.m128i_i64[0] = *(_QWORD *)v23;
        v399.m128i_i32[2] = v28;
        v400 = v29;
        if ( v29 )
        {
          v339 = v27;
          v359 = v26;
          sub_B96E90((__int64)&v400, v29, 1);
          v9 = *(_DWORD *)(v12 + 24);
          LODWORD(v27) = v339;
          v26 = v359;
        }
        LODWORD(v401) = *(_DWORD *)(v12 + 72);
        v30 = sub_3402EA0(v25, v9, (unsigned int)&v400, v26, v27, 0, (__int64)&v399, 1);
        if ( v400 )
          sub_B91220((__int64)&v400, v400);
      }
      return v30;
    case 0x117u:
    case 0x118u:
    case 0x11Bu:
    case 0x11Cu:
    case 0x11Du:
    case 0x11Eu:
      return sub_328C2D0(a1, a2);
    case 0x125u:
      v45 = *(__int64 **)(a2 + 40);
      v46 = *(_WORD *)(a2 + 96);
      v47 = v46;
      v48 = *(_QWORD *)(a2 + 104);
      v360 = *v45;
      v340 = v45[1];
      v49 = *(_QWORD *)(v45[5] + 56);
      if ( !v49 )
        return 0;
      v50 = 0;
      do
      {
        v51 = *(_QWORD *)(v49 + 16);
        if ( a2 != v51 )
        {
          if ( *(_DWORD *)(v51 + 24) != 298 || v50 && v50 != v51 )
            return 0;
          v50 = *(_QWORD *)(v49 + 16);
        }
        v49 = *(_QWORD *)(v49 + 32);
      }
      while ( v49 );
      if ( !v50 )
        return 0;
      if ( (*(_BYTE *)(*(_QWORD *)(v50 + 112) + 37LL) & 0xF) != 0 )
        return 0;
      v52 = *(_WORD *)(v50 + 32);
      if ( (v52 & 8) != 0 )
        return 0;
      if ( (v52 & 0x380) != 0 )
        return 0;
      v53 = *(_QWORD *)(v50 + 40);
      if ( *(_DWORD *)(*(_QWORD *)(v53 + 80) + 24LL) != 51
        || v46 != *(_WORD *)(v50 + 96)
        || *(_QWORD *)(v50 + 104) != *(_QWORD *)(a2 + 104) && !v46 )
      {
        return 0;
      }
      v304 = *(_QWORD *)(a2 + 104);
      v312 = *(_WORD *)(a2 + 96);
      v322 = v50;
      if ( !(unsigned __int8)sub_33CFB90(v53, a2, 0, 2) )
        return 0;
      v54 = v304;
      v55 = 0;
      for ( i = *(_QWORD *)(v322 + 56); i; i = *(_QWORD *)(i + 32) )
      {
        if ( !*(_DWORD *)(i + 8) )
        {
          if ( *(_DWORD *)(*(_QWORD *)(i + 16) + 24LL) != 299 || v55 )
            return 0;
          v55 = *(_QWORD *)(i + 16);
        }
      }
      v303 = v322;
      v309 = v312;
      v320 = v54;
      if ( v55 )
      {
        v336 = v55;
        if ( (unsigned __int8)sub_3287C60(v55) )
        {
          if ( (*(_WORD *)(v336 + 32) & 0x380) == 0 )
          {
            v280 = *(_QWORD *)(v336 + 40);
            if ( *(_DWORD *)(*(_QWORD *)(v280 + 120) + 24LL) == 51
              && v309 == *(_WORD *)(v336 + 96)
              && (*(_QWORD *)(v336 + 104) == v320 || v309) )
            {
              if ( (unsigned __int8)sub_33CFB90(v280, v303, 1, 2) )
              {
                v281 = v336;
                v282 = *a1;
                v283 = *(_QWORD *)(v336 + 40);
                v284 = *(_QWORD *)(v336 + 112);
                v400 = *(_QWORD *)(a2 + 80);
                v285 = (_QWORD *)(v283 + 80);
                if ( v400 )
                {
                  v310 = v284;
                  v321 = v282;
                  sub_325F5D0((__int64 *)&v400);
                  v285 = (_QWORD *)(v283 + 80);
                  v284 = v310;
                  LODWORD(v282) = v321;
                  v281 = v336;
                }
                v337 = v281;
                LODWORD(v401) = *(_DWORD *)(a2 + 72);
                v286 = sub_33F6CE0(v282, v360, v340, (unsigned int)&v400, *v285, v285[1], v47, v48, v284);
                v288 = v287;
                sub_9C6650(&v400);
                v400 = v286;
                i = v286;
                v401 = v288;
                sub_32EB790((__int64)a1, v337, (__int64 *)&v400, 1, 0);
              }
            }
          }
        }
      }
      return i;
    case 0x126u:
      v37 = *(_WORD *)(a2 + 96);
      v38 = *(_QWORD *)(a2 + 104);
      v39 = *(const __m128i **)(a2 + 40);
      v40 = v37;
      v41 = v38;
      v399 = _mm_loadu_si128(v39);
      i = *(_QWORD *)(v39[2].m128i_i64[1] + 56);
      if ( !i )
        return 0;
      v43 = 0;
      do
      {
        v44 = *(_QWORD *)(i + 16);
        if ( v12 != v44 )
        {
          if ( *(_DWORD *)(v44 + 24) != 299 || v43 && v44 != v43 )
            return 0;
          v43 = *(_QWORD *)(i + 16);
        }
        i = *(_QWORD *)(i + 32);
      }
      while ( i );
      if ( !v43 )
        return 0;
      if ( (*(_BYTE *)(*(_QWORD *)(v43 + 112) + 37LL) & 0xF) != 0 )
        return 0;
      v166 = *(_WORD *)(v43 + 32);
      if ( (v166 & 8) != 0
        || (v166 & 0x380) != 0
        || *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v43 + 40) + 120LL) + 24LL) != 51
        || v37 != *(_WORD *)(v43 + 96)
        || *(_QWORD *)(v43 + 104) != v38 && !v37 )
      {
        return 0;
      }
      v330 = v38;
      v349 = v37;
      v372 = v43;
      if ( !(unsigned __int8)sub_33CFB90(&v399, v43, 0, 2) )
        return 0;
      v167 = *(_QWORD *)(v43 + 40);
      v168 = *(_QWORD *)(v167 + 40);
      if ( *(_DWORD *)(v168 + 24) == 298 )
      {
        v315 = *(_QWORD *)(v372 + 40);
        v373 = *(_QWORD *)(v167 + 40);
        if ( (unsigned __int8)sub_3287C60(v373) )
        {
          if ( (*(_WORD *)(v168 + 32) & 0x380) == 0
            && *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v168 + 40) + 80LL) + 24LL) == 51
            && v349 == *(_WORD *)(v168 + 96)
            && (*(_QWORD *)(v168 + 104) == v330 || v349) )
          {
            if ( (unsigned __int8)sub_33CFB90(v315, v168, 1, 2) )
            {
              v276 = *a1;
              v277 = *(_QWORD *)(v168 + 112);
              v278 = *(_QWORD *)(v168 + 40);
              v400 = *(_QWORD *)(v12 + 80);
              if ( v400 )
              {
                v335 = v278;
                v357 = v277;
                sub_325F5D0((__int64 *)&v400);
                v278 = v335;
                v277 = v357;
                v168 = v373;
              }
              LODWORD(v401) = *(_DWORD *)(v12 + 72);
              v279 = sub_33F7210(
                       v276,
                       **(_QWORD **)(v168 + 40),
                       *(_QWORD *)(*(_QWORD *)(v168 + 40) + 8LL),
                       (unsigned int)&v400,
                       *(_QWORD *)(v278 + 40),
                       *(_QWORD *)(v278 + 48),
                       v40,
                       v41,
                       v277);
              sub_9C6650(&v400);
              return v279;
            }
          }
        }
      }
      return i;
    case 0x12Au:
      return sub_3314670(a1, a2, a3, a4, v9, a6);
    case 0x12Bu:
      return sub_331C5B0(a1, a2);
    case 0x131u:
      return sub_32FDC00(a1, a2);
    case 0x132u:
      v63 = *(_QWORD *)(a2 + 40);
      v64 = *(_QWORD *)(a2 + 80);
      v65 = *(_QWORD *)(v63 + 120);
      v66 = *(_QWORD *)(v63 + 40);
      v67 = *(_QWORD *)(v63 + 80);
      v68 = *(unsigned int *)(v63 + 88);
      LODWORD(v63) = *(_DWORD *)(v63 + 128);
      v362 = v65;
      v399.m128i_i64[0] = v64;
      v341 = v63;
      if ( v64 )
      {
        v323 = v67;
        sub_B96E90((__int64)&v399, v64, 1);
        v67 = v323;
      }
      v69 = *a1;
      v299 = v67;
      v399.m128i_i32[2] = *(_DWORD *)(v12 + 72);
      v70 = *(_DWORD *)(v66 + 96);
      v71 = a1[1];
      v298 = v70;
      v300 = *(_QWORD *)(*(_QWORD *)(v67 + 48) + 16 * v68 + 8);
      v305 = *(unsigned __int16 *)(*(_QWORD *)(v67 + 48) + 16 * v68);
      v324 = *(_QWORD *)(v69 + 64);
      v313 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v71 + 528LL);
      v72 = sub_2E79000(*(__int64 **)(v69 + 40));
      v73 = v313(v71, v72, v324, v305, v300);
      v400 = (unsigned __int64)a1;
      BYTE4(v401) = 0;
      v74 = *a1;
      LODWORD(v401) = *((_DWORD *)a1 + 6);
      v75 = a1[1];
      *((_QWORD *)&v289 + 1) = v341;
      *(_QWORD *)&v289 = v362;
      v402 = v74;
      v77 = sub_348D3E0(v75, v73, v76, v299, v68, v298, v289, 0, (__int64)&v400, (__int64)&v399);
      if ( v399.m128i_i64[0] )
        sub_B91220((__int64)&v399, v399.m128i_i64[0]);
      if ( !v77 || *(_DWORD *)(v77 + 24) == 328 )
        return 0;
      v400 = v77;
      sub_32B3B20((__int64)(a1 + 71), (__int64 *)&v400);
      if ( *(int *)(v77 + 88) < 0 )
      {
        *(_DWORD *)(v77 + 88) = *((_DWORD *)a1 + 12);
        v240 = *((unsigned int *)a1 + 12);
        if ( v240 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
        {
          sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v240 + 1, 8u, v78, v79);
          v240 = *((unsigned int *)a1 + 12);
        }
        *(_QWORD *)(a1[5] + 8 * v240) = v77;
        ++*((_DWORD *)a1 + 12);
      }
      if ( *(_DWORD *)(v77 + 24) != 208 )
        return 0;
      v204 = *(_QWORD *)(v12 + 80);
      v205 = *a1;
      v206 = *(__int128 **)(v12 + 40);
      v207 = *(__int128 **)(v77 + 40);
      v400 = v204;
      if ( v204 )
      {
        v377 = v205;
        sub_B96E90((__int64)&v400, v204, 1);
        LODWORD(v205) = v377;
      }
      LODWORD(v401) = *(_DWORD *)(v12 + 72);
      result = sub_33FC1D0(
                 v205,
                 306,
                 (unsigned int)&v400,
                 1,
                 0,
                 v205,
                 *v206,
                 v207[5],
                 *v207,
                 *(__int128 *)((char *)v207 + 40),
                 v206[10]);
      if ( v400 )
        goto LABEL_269;
      return result;
    case 0x153u:
      v56 = *(_QWORD *)(a2 + 40);
      v57 = *(_QWORD *)(a2 + 104);
      v58 = *(_QWORD *)(v56 + 40);
      v361 = *(_QWORD *)(v56 + 48);
      v59 = *(_WORD *)(a2 + 96);
      v60 = *(_QWORD *)(v58 + 48) + 16LL * *(unsigned int *)(v56 + 48);
      v61 = *(_WORD *)v60;
      v62 = *(_QWORD *)(v60 + 8);
      v385 = v59;
      v386 = v57;
      v383 = v61;
      v384 = v62;
      if ( v61 == v59 )
      {
        if ( v59 || v62 == v57 )
          return 0;
        v387 = 0;
        v388 = v62;
      }
      else
      {
        v387 = v61;
        v388 = v62;
        if ( v61 )
        {
          if ( v61 == 1 || (unsigned __int16)(v61 - 504) <= 7u )
            goto LABEL_471;
          v141 = *(_QWORD *)&byte_444C4A0[16 * v61 - 16];
          v143 = byte_444C4A0[16 * v61 - 8];
LABEL_231:
          if ( v59 )
          {
            if ( v59 == 1 || (unsigned __int16)(v59 - 504) <= 7u )
              goto LABEL_471;
            v144 = *(_QWORD *)&byte_444C4A0[16 * v59 - 16];
            LOBYTE(v145) = byte_444C4A0[16 * v59 - 8];
          }
          else
          {
            v327 = v143;
            v144 = sub_3007260((__int64)&v385);
            v143 = v327;
            v389 = v144;
            v390 = v145;
          }
          if ( (_BYTE)v145 && !v143 || v141 <= v144 )
            return 0;
          if ( v59 )
          {
            if ( (unsigned __int16)(v59 - 17) > 0xD3u )
            {
              LOWORD(v400) = v59;
              v401 = v57;
              goto LABEL_239;
            }
            v59 = word_4456580[v59 - 1];
            v246 = 0;
          }
          else
          {
            if ( !sub_30070B0((__int64)&v385) )
            {
              v401 = v57;
              LOWORD(v400) = 0;
              goto LABEL_377;
            }
            v59 = sub_3009970((__int64)&v385, v141, v232, v233, v234);
            v246 = v257;
          }
          LOWORD(v400) = v59;
          v401 = v246;
          if ( v59 )
          {
LABEL_239:
            if ( v59 == 1 || (unsigned __int16)(v59 - 504) <= 7u )
              goto LABEL_471;
            v146 = *(_QWORD *)&byte_444C4A0[16 * v59 - 16];
LABEL_242:
            v147 = v383;
            v148 = v146;
            if ( v383 )
            {
              if ( (unsigned __int16)(v383 - 17) <= 0xD3u )
              {
                v147 = word_4456580[v383 - 1];
                v149 = 0;
                goto LABEL_245;
              }
            }
            else if ( sub_30070B0((__int64)&v383) )
            {
              v147 = sub_3009970((__int64)&v383, v141, v228, v229, v230);
              v149 = v231;
              goto LABEL_245;
            }
            v149 = v384;
LABEL_245:
            v393 = v147;
            v394 = v149;
            if ( !v147 )
            {
              v150.m128i_i64[0] = sub_3007260((__int64)&v393);
              v399 = v150;
LABEL_247:
              LODWORD(v401) = v150.m128i_i32[0];
              if ( v150.m128i_i32[0] > 0x40u )
                sub_C43690((__int64)&v400, 0, 0);
              else
                v400 = 0;
              if ( v148 )
              {
                if ( v148 > 0x40 )
                {
                  sub_C43C90(&v400, 0, v148);
                }
                else
                {
                  v151 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v148);
                  if ( (unsigned int)v401 > 0x40 )
                    *(_QWORD *)v400 |= v151;
                  else
                    v400 |= v151;
                }
              }
              if ( (unsigned __int8)sub_32D08B0((__int64)a1, v58, v361, (int)&v400) )
              {
                if ( (unsigned int)v401 > 0x40 && v400 )
                  j_j___libc_free_0_0(v400);
                return v12;
              }
              if ( (unsigned int)v401 > 0x40 && v400 )
                j_j___libc_free_0_0(v400);
              return 0;
            }
            if ( v147 != 1 && (unsigned __int16)(v147 - 504) > 7u )
            {
              v150.m128i_i64[0] = *(_QWORD *)&byte_444C4A0[16 * v147 - 16];
              goto LABEL_247;
            }
LABEL_471:
            BUG();
          }
LABEL_377:
          v235 = sub_3007260((__int64)&v400);
          v141 = v236;
          v395 = v235;
          LODWORD(v146) = v235;
          v396 = v236;
          goto LABEL_242;
        }
      }
      v391 = sub_3007260((__int64)&v387);
      v141 = v391;
      v392 = v142;
      v143 = v142;
      goto LABEL_231;
    case 0x16Au:
      return sub_3313A10((__int64)a1, a2);
    case 0x16Bu:
      return sub_3313C70((__int64)a1, a2);
    case 0x16Cu:
      return sub_3305550(a1, a2);
    case 0x16Du:
      return sub_3294B00(a1, a2);
    case 0x16Fu:
      return sub_32EF9B0(a1, (_QWORD *)a2, a3, a4, v9, a6);
    case 0x178u:
    case 0x179u:
    case 0x17Au:
    case 0x17Bu:
    case 0x17Cu:
    case 0x17Du:
    case 0x17Eu:
    case 0x17Fu:
    case 0x180u:
    case 0x181u:
    case 0x182u:
    case 0x183u:
    case 0x184u:
    case 0x185u:
    case 0x186u:
      v16 = *(const __m128i **)(a2 + 40);
      v17 = v16->m128i_i64[0];
      v338 = _mm_loadu_si128(v16);
      v18 = *(_QWORD *)(v16->m128i_i64[0] + 48) + 16LL * v16->m128i_u32[2];
      v19 = *(_WORD *)v18;
      v20 = *(_QWORD *)(v18 + 8);
      v397 = v19;
      v398 = v20;
      if ( v19 )
      {
        LODWORD(v21) = word_4456340[v19 - 1];
        if ( (unsigned __int16)(v19 - 176) <= 0x34u )
          goto LABEL_16;
      }
      else
      {
        v314 = v9;
        v329 = v9;
        v21 = sub_3007240((__int64)&v397);
        v9 = v329;
        v10 = v314;
        if ( BYTE4(v21) )
          goto LABEL_16;
      }
      if ( (_DWORD)v21 == 1 )
      {
        v209 = *(_QWORD *)(a2 + 80);
        v400 = v209;
        if ( v209 )
          sub_B96E90((__int64)&v400, v209, 1);
        v210 = *a1;
        v211 = *a1;
        LODWORD(v401) = *(_DWORD *)(v12 + 72);
        v216 = sub_3400EE0(v211, 0, &v400, 0, v10);
        v217 = v212;
        if ( v397 )
        {
          v218 = 0;
          LOWORD(v219) = word_4456580[v397 - 1];
        }
        else
        {
          v378 = v216;
          v382 = v212;
          v219 = sub_3009970((__int64)&v397, 0, v212, v213, v214);
          v216 = v378;
          v217 = v382;
          v311 = HIWORD(v219);
          v218 = v247;
        }
        *((_QWORD *)&v296 + 1) = v217;
        HIWORD(v220) = v311;
        *(_QWORD *)&v296 = v216;
        LOWORD(v220) = v219;
        result = sub_3406EB0(v210, 158, (unsigned int)&v400, v220, v218, v215, *(_OWORD *)&v338, v296);
        v222 = *(unsigned __int16 **)(v12 + 48);
        v224 = v223;
        v225 = *v222;
        v226 = *((_QWORD *)v222 + 1);
        v227 = *(_QWORD *)(result + 48) + 16LL * (unsigned int)v223;
        if ( *v222 != *(_WORD *)v227 || *(_QWORD *)(v227 + 8) != v226 && !v225 )
        {
          *((_QWORD *)&v297 + 1) = v224;
          *(_QWORD *)&v297 = result;
          result = sub_33FAF80(*a1, 215, (unsigned int)&v400, v225, v226, v221, v297);
        }
        if ( v400 )
        {
          v355 = result;
          sub_B91220((__int64)&v400, v400);
          return v355;
        }
        return result;
      }
LABEL_16:
      v22 = v9 - 384;
      if ( v9 - 384 > 1 )
      {
        if ( *(_DWORD *)(v17 + 24) != 160 )
        {
LABEL_18:
          if ( v22 > 2 )
            return 0;
          v180 = *(_DWORD *)(v17 + 24);
          goto LABEL_302;
        }
LABEL_328:
        v199 = *(__int64 **)(v17 + 40);
        v200 = *(unsigned __int16 *)(*(_QWORD *)(v199[5] + 48) + 16LL * *((unsigned int *)v199 + 12));
        if ( !(_WORD)v200 || !*(_QWORD *)(a1[1] + 8 * v200 + 112) )
          return 0;
        v201 = *v199;
        v202 = v199[1];
        v354 = (__int128)_mm_loadu_si128((const __m128i *)(v199 + 5));
        if ( v9 == 385 )
        {
          if ( *(_DWORD *)(v201 + 24) != 51 )
          {
            v308 = v10;
            v319 = v22;
            v248 = sub_33E0720(v201, v202, 0);
            v9 = 385;
            if ( !v248 )
              goto LABEL_334;
          }
        }
        else
        {
          if ( v9 != 384 )
            goto LABEL_18;
          v308 = v10;
          v319 = v22;
          if ( *(_DWORD *)(*v199 + 24) != 51 )
          {
            v203 = sub_33E07E0(v201, v202, 0);
            v9 = 384;
            if ( !v203 )
            {
LABEL_334:
              v22 = v319;
              v10 = v308;
              goto LABEL_18;
            }
          }
        }
        v249 = *(_QWORD *)(a2 + 80);
        v250 = *a1;
        v251 = *(_QWORD *)(*(_QWORD *)(v12 + 48) + 8LL);
        v252 = **(unsigned __int16 **)(v12 + 48);
        v399.m128i_i64[0] = v249;
        if ( v249 )
        {
          v379 = v9;
          sub_B96E90((__int64)&v399, v249, 1);
          v9 = v379;
        }
        v399.m128i_i32[2] = *(_DWORD *)(v12 + 72);
        result = sub_33FAF80(v250, v9, (unsigned int)&v399, v252, v251, v22, v354);
        goto LABEL_313;
      }
      v172 = a1[1];
      v333 = (v9 == 384) + 389;
      if ( v19 == 1 )
      {
        if ( (*(_BYTE *)(v172 + v10 + 6914) & 0xFB) == 0 )
          goto LABEL_301;
        v173 = 1;
      }
      else
      {
        if ( !v19 )
          goto LABEL_301;
        v173 = v19;
        if ( !*(_QWORD *)(v172 + 8LL * v19 + 112)
          || (*(_BYTE *)(v172 + 500LL * v19 + v10 + 6414) & 0xFB) == 0
          || !*(_QWORD *)(v172 + 8 * (v19 + 14LL)) )
        {
          goto LABEL_301;
        }
      }
      if ( (*(_BYTE *)(v333 + 500 * v173 + v172 + 6414) & 0xFB) != 0 )
        goto LABEL_301;
      v301 = v10;
      v306 = v9 - 384;
      v316 = v9;
      v174 = sub_33D4D80(*a1, v338.m128i_i64[0], v338.m128i_i64[1], 0);
      v175 = v397;
      v9 = v316;
      v22 = v306;
      v10 = v301;
      v176 = v174;
      if ( v397 )
      {
        if ( (unsigned __int16)(v397 - 17) > 0xD3u )
        {
LABEL_297:
          v177 = v398;
          goto LABEL_298;
        }
        v175 = word_4456580[v397 - 1];
        v177 = 0;
      }
      else
      {
        v270 = sub_30070B0((__int64)&v397);
        v9 = v316;
        v22 = v306;
        v10 = v301;
        if ( !v270 )
          goto LABEL_297;
        v273 = sub_3009970((__int64)&v397, v338.m128i_i64[0], v271, v272, v301);
        v9 = v316;
        v22 = v306;
        v175 = v273;
        v10 = v301;
        v177 = v274;
      }
LABEL_298:
      v399.m128i_i16[0] = v175;
      v399.m128i_i64[1] = v177;
      if ( v175 )
      {
        if ( v175 == 1 || (unsigned __int16)(v175 - 504) <= 7u )
          goto LABEL_471;
        v178 = *(_QWORD *)&byte_444C4A0[16 * v175 - 16];
      }
      else
      {
        v302 = v10;
        v307 = v22;
        v317 = v9;
        v178 = sub_3007260((__int64)&v399);
        v10 = v302;
        v22 = v307;
        v400 = v178;
        v9 = v317;
        v401 = v179;
      }
      if ( v178 == v176 )
      {
        v266 = *(_QWORD *)(a2 + 80);
        v267 = *a1;
        v268 = *(_QWORD *)(*(_QWORD *)(v12 + 48) + 8LL);
        v269 = **(unsigned __int16 **)(v12 + 48);
        v399.m128i_i64[0] = v266;
        if ( v266 )
          sub_B96E90((__int64)&v399, v266, 1);
        v399.m128i_i32[2] = *(_DWORD *)(v12 + 72);
        result = sub_33FAF80(v267, v333, (unsigned int)&v399, v269, v268, v22, *(_OWORD *)&v338);
        goto LABEL_313;
      }
LABEL_301:
      v180 = *(_DWORD *)(v17 + 24);
      if ( v180 != 160 )
      {
LABEL_302:
        if ( (unsigned int)(v180 - 213) > 2 )
          return 0;
        v181 = *(unsigned int **)(v17 + 40);
        v182 = a1[1];
        v183 = *(_QWORD *)(*(_QWORD *)v181 + 48LL) + 16LL * v181[2];
        v184 = *(_WORD *)v183;
        if ( *(_WORD *)v183 == 1 )
        {
          if ( (*(_BYTE *)(v182 + v10 + 6914) & 0xFB) != 0 )
            return 0;
          v185 = *(_QWORD *)(v183 + 8);
          v186 = *a1;
        }
        else
        {
          if ( !v184
            || !*(_QWORD *)(v182 + 8LL * v184 + 112)
            || (*(_BYTE *)(v182 + 500LL * v184 + v10 + 6414) & 0xFB) != 0 )
          {
            return 0;
          }
          v185 = *(_QWORD *)(v183 + 8);
          v186 = *a1;
          if ( (unsigned __int16)(v184 - 17) <= 0xD3u )
          {
            LODWORD(v185) = 0;
            v184 = word_4456580[v184 - 1];
          }
        }
        v187 = *(_QWORD *)(a2 + 80);
        v188 = v184;
        v399.m128i_i64[0] = v187;
        if ( v187 )
        {
          v318 = v184;
          v334 = v186;
          v351 = v185;
          v375 = v9;
          sub_B96E90((__int64)&v399, v187, 1);
          v188 = v318;
          LODWORD(v186) = v334;
          LODWORD(v185) = v351;
          v9 = v375;
        }
        v399.m128i_i32[2] = *(_DWORD *)(v12 + 72);
        *(_QWORD *)&v189 = sub_33FAF80(v186, v9, (unsigned int)&v399, v188, v185, v186, *(_OWORD *)v181);
        v190 = v295;
        v376 = v189;
        if ( v399.m128i_i64[0] )
          sub_B91220((__int64)&v399, v399.m128i_i64[0]);
        v191 = *(_QWORD *)(v12 + 80);
        v192 = *a1;
        v193 = *(_QWORD *)(*(_QWORD *)(v12 + 48) + 8LL);
        v194 = **(unsigned __int16 **)(v12 + 48);
        v399.m128i_i64[0] = v191;
        if ( v191 )
        {
          v352 = v193;
          sub_B96E90((__int64)&v399, v191, 1);
          LODWORD(v193) = v352;
        }
        v399.m128i_i32[2] = *(_DWORD *)(v12 + 72);
        result = sub_33FAF80(v192, *(_DWORD *)(v17 + 24), (unsigned int)&v399, v194, v193, v190, v376);
LABEL_313:
        if ( v399.m128i_i64[0] )
        {
          v353 = result;
          sub_B91220((__int64)&v399, v399.m128i_i64[0]);
          return v353;
        }
        return result;
      }
      goto LABEL_328;
    case 0x187u:
    case 0x188u:
      return sub_32811A0(a1, a2);
    case 0x18Bu:
    case 0x18Cu:
    case 0x18Du:
    case 0x18Eu:
    case 0x18Fu:
    case 0x190u:
    case 0x191u:
    case 0x192u:
    case 0x193u:
    case 0x194u:
    case 0x195u:
    case 0x196u:
    case 0x197u:
    case 0x198u:
    case 0x199u:
    case 0x19Au:
    case 0x19Bu:
    case 0x19Cu:
    case 0x19Du:
    case 0x19Eu:
    case 0x19Fu:
    case 0x1A0u:
    case 0x1A1u:
    case 0x1A2u:
    case 0x1A3u:
    case 0x1A4u:
    case 0x1A5u:
    case 0x1A6u:
    case 0x1A7u:
    case 0x1A8u:
    case 0x1A9u:
    case 0x1AAu:
    case 0x1ABu:
    case 0x1ACu:
    case 0x1ADu:
    case 0x1AEu:
    case 0x1AFu:
    case 0x1B0u:
    case 0x1B1u:
    case 0x1B2u:
    case 0x1B3u:
    case 0x1B4u:
    case 0x1B5u:
    case 0x1B6u:
    case 0x1B7u:
    case 0x1B8u:
    case 0x1B9u:
    case 0x1BAu:
    case 0x1BBu:
    case 0x1BCu:
    case 0x1BDu:
    case 0x1BEu:
    case 0x1BFu:
    case 0x1C0u:
    case 0x1C1u:
    case 0x1C2u:
    case 0x1C3u:
    case 0x1C4u:
    case 0x1C5u:
    case 0x1C6u:
    case 0x1C7u:
    case 0x1C8u:
    case 0x1C9u:
    case 0x1CAu:
    case 0x1CBu:
    case 0x1CCu:
    case 0x1CDu:
    case 0x1CEu:
    case 0x1CFu:
    case 0x1D0u:
    case 0x1D1u:
    case 0x1D2u:
    case 0x1D3u:
    case 0x1D4u:
    case 0x1D5u:
    case 0x1D6u:
    case 0x1D7u:
    case 0x1D8u:
    case 0x1D9u:
    case 0x1DAu:
    case 0x1DBu:
    case 0x1DCu:
    case 0x1DDu:
    case 0x1DEu:
    case 0x1DFu:
    case 0x1E0u:
    case 0x1E1u:
    case 0x1E2u:
    case 0x1E3u:
    case 0x1E4u:
    case 0x1E5u:
    case 0x1E6u:
    case 0x1E7u:
    case 0x1E8u:
    case 0x1E9u:
    case 0x1EAu:
    case 0x1EBu:
    case 0x1ECu:
      if ( v9 != 470 )
        goto LABEL_5;
      v133 = sub_3295160(a1, a2);
      if ( v133 )
        return v133;
      v9 = *(_DWORD *)(a2 + 24);
LABEL_5:
      if ( v9 != 467 )
        goto LABEL_6;
      v133 = sub_32954D0(a1, a2);
      if ( v133 )
        return v133;
      v9 = *(_DWORD *)(a2 + 24);
LABEL_6:
      if ( v9 != 469 )
        goto LABEL_7;
      v133 = sub_33058F0(a1, a2, a3, a4, v10);
      if ( v133 )
        return v133;
      v9 = *(_DWORD *)(a2 + 24);
LABEL_7:
      if ( v9 != 466 )
        goto LABEL_8;
      v133 = sub_32682B0(a1, a2, a3, a4, v10);
      if ( v133 )
        return v133;
      v9 = *(_DWORD *)(a2 + 24);
LABEL_8:
      v400 = sub_33CB1F0(v9);
      v14 = HIDWORD(v400);
      if ( BYTE4(v400) )
      {
        v237 = (_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL * (unsigned int)v400);
        LODWORD(v14) = sub_33CF170(*v237, v237[1]);
        v400 = sub_33CB160(*(unsigned int *)(a2 + 24));
        if ( !BYTE4(v400) )
          goto LABEL_382;
      }
      else
      {
        v400 = sub_33CB160(*(unsigned int *)(a2 + 24));
        if ( !BYTE4(v400) )
        {
LABEL_10:
          v15 = *(_DWORD *)(a2 + 24);
          if ( v15 <= 436 )
          {
            if ( v15 > 398 )
            {
              switch ( v15 )
              {
                case 399:
                  return sub_32D1280((const __m128i **)a1, a2);
                case 404:
                  return sub_32AF310(a2, *a1);
                case 428:
                  v138 = *(_DWORD *)(a2 + 28);
                  v400 = *a1;
                  LODWORD(v401) = v138;
                  v402 = *(_QWORD *)(v400 + 1024);
                  *(_QWORD *)(v400 + 1024) = &v400;
                  v139 = sub_32904E0((__int64 **)a1, a2);
                  v135 = v139;
                  v133 = v139;
                  if ( v139 )
                  {
                    v140 = *(_DWORD *)(v139 + 24);
                    if ( v140 )
                    {
                      if ( v140 != 328 )
                        goto LABEL_218;
                    }
                  }
                  goto LABEL_219;
                case 429:
                  v134 = *(_DWORD *)(a2 + 28);
                  v400 = *a1;
                  LODWORD(v401) = v134;
                  v402 = *(_QWORD *)(v400 + 1024);
                  *(_QWORD *)(v400 + 1024) = &v400;
                  v135 = sub_32924F0((__int64 **)a1, a2);
                  v133 = v135;
                  if ( v135 && *(_DWORD *)(v135 + 24) != 328 )
                  {
LABEL_218:
                    v399.m128i_i64[0] = v135;
                    sub_32B3B20((__int64)(a1 + 71), v399.m128i_i64);
                    if ( *(int *)(v135 + 88) < 0 )
                    {
                      *(_DWORD *)(v135 + 88) = *((_DWORD *)a1 + 12);
                      v275 = *((unsigned int *)a1 + 12);
                      if ( v275 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
                      {
                        sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v275 + 1, 8u, v136, v137);
                        v275 = *((unsigned int *)a1 + 12);
                      }
                      *(_QWORD *)(a1[5] + 8 * v275) = v135;
                      ++*((_DWORD *)a1 + 12);
                    }
                  }
LABEL_219:
                  *(_QWORD *)(v400 + 1024) = v402;
                  break;
                case 436:
                  v133 = sub_32C0EA0((__int64 **)a1, a2);
                  break;
                default:
                  return 0;
              }
              return v133;
            }
            return 0;
          }
          if ( v15 != 488 )
            return 0;
          return sub_328FC90(a1, a2);
        }
      }
      LODWORD(v14) = sub_33D1AE0(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL * (unsigned int)v400), 0) | v14;
LABEL_382:
      if ( !(_BYTE)v14 )
        goto LABEL_10;
      if ( (unsigned __int8)sub_33CB120(*(unsigned int *)(a2 + 24)) )
      {
        v253 = *(unsigned __int16 **)(a2 + 48);
        v254 = *a1;
        v255 = *((_QWORD *)v253 + 1);
        v256 = *v253;
        v400 = 0;
        LODWORD(v401) = 0;
        v133 = sub_33F17F0(v254, 51, &v400, v256, v255);
        if ( v400 )
          sub_B91220((__int64)&v400, v400);
        return v133;
      }
      v238 = *(_DWORD *)(a2 + 24);
      if ( v238 > 365 )
      {
        if ( v238 > 470 )
        {
          if ( v238 == 497 )
            goto LABEL_400;
        }
        else if ( v238 > 464 )
        {
          goto LABEL_400;
        }
      }
      else
      {
        if ( v238 > 337 )
          goto LABEL_400;
        if ( v238 > 294 )
        {
          if ( (unsigned int)(v238 - 298) <= 1 )
            goto LABEL_400;
        }
        else if ( v238 > 292 )
        {
          goto LABEL_400;
        }
      }
      if ( (*(_BYTE *)(a2 + 32) & 2) == 0 )
      {
        if ( !(unsigned __int8)sub_33CB150() )
          return 0;
        return **(_QWORD **)(a2 + 40);
      }
LABEL_400:
      v239 = *(_DWORD **)(a2 + 40);
      if ( (*(_BYTE *)(*(_QWORD *)(a2 + 112) + 32LL) & 2) != 0 )
        return *(_QWORD *)v239;
      v258 = *a1;
      v259 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
      v260 = **(unsigned __int16 **)(a2 + 48);
      v400 = 0;
      LODWORD(v401) = 0;
      v261 = sub_33F17F0(v258, 51, &v400, v260, v259);
      v263 = v262;
      if ( v400 )
      {
        v380 = v261;
        sub_B91220((__int64)&v400, v400);
        v261 = v380;
      }
      v264 = *(_QWORD *)v239;
      v265 = v239[2];
      LODWORD(v401) = v263;
      v402 = v264;
      v403 = v265;
      v400 = v261;
      return sub_32EB790((__int64)a1, a2, (__int64 *)&v400, 2, 1);
    case 0x1F1u:
      return sub_3294EC0(a1, a2);
    default:
      return 0;
  }
}
