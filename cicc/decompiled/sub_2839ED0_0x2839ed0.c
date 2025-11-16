// Function: sub_2839ED0
// Address: 0x2839ed0
//
__int64 __fastcall sub_2839ED0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  const __m128i *v7; // r15
  const __m128i *v8; // rax
  const char **v9; // r14
  __int64 *v10; // rdx
  _BYTE *v11; // rax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned __int64 v21; // rax
  const __m128i *v22; // rax
  void *v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  unsigned __int64 v26; // rsi
  __int64 *v27; // rdi
  unsigned __int64 v28; // rdx
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int64 v31; // rcx
  __m128i *v32; // rdx
  const __m128i *v33; // rax
  const __m128i *v34; // rcx
  __int64 v35; // rax
  __int64 v36; // rdx
  unsigned __int64 v37; // rdi
  __m128i *v38; // rdx
  const __m128i *v39; // rax
  unsigned __int64 v40; // rcx
  unsigned __int64 v41; // rax
  __int64 v42; // r12
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  unsigned __int64 v46; // r13
  __int64 v47; // r15
  __int64 *v48; // rax
  __int64 *v49; // rdx
  __int64 v50; // r12
  _QWORD *v51; // rax
  char v52; // dl
  unsigned __int64 v53; // rdx
  char v54; // si
  __int64 v55; // rbx
  __int64 v56; // rax
  unsigned __int64 v57; // rdx
  __int64 v58; // r13
  int v59; // r12d
  unsigned int v60; // r15d
  __int64 v61; // rsi
  _QWORD *v62; // rax
  _QWORD *v63; // rdx
  __int64 v65; // rax
  unsigned __int64 v66; // rdx
  __int64 *v67; // rax
  __int64 v68; // rsi
  __int64 *v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // r12
  unsigned int *v73; // rbx
  __int64 v74; // rax
  __int64 v75; // r13
  __int64 v76; // r15
  __int64 v77; // rdx
  __int64 v78; // rsi
  __int64 v79; // rdi
  __m128i *v80; // rax
  const __m128i *v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rdx
  _QWORD *v84; // rax
  char v85; // dl
  const __m128i *v86; // rax
  const __m128i **v87; // rbx
  __int64 v88; // rsi
  _QWORD *v89; // rax
  _QWORD *v90; // rcx
  unsigned __int64 v91; // rdi
  __int64 v92; // r15
  unsigned int v93; // esi
  unsigned __int64 v94; // r12
  __int64 v95; // rax
  int v96; // r13d
  int v97; // edi
  unsigned __int64 v98; // rcx
  unsigned int v99; // edx
  unsigned __int64 v100; // rax
  __int64 v101; // r8
  __int64 *v102; // rbx
  int v103; // edx
  unsigned int v104; // ecx
  __int64 v105; // r9
  __int64 v106; // rdx
  __int64 v107; // r9
  __int64 v108; // r13
  __int64 v109; // rbx
  __int64 v110; // rdx
  __int64 v111; // r8
  __int64 v112; // r13
  __int64 v113; // r12
  __int64 v114; // r15
  __int64 v115; // r14
  _QWORD *v116; // rax
  _QWORD *v117; // rdx
  __int64 v118; // rax
  unsigned __int64 v119; // rdx
  __int64 v120; // rdx
  __int64 *v121; // rax
  __int64 *v122; // r15
  __int64 v123; // r12
  __int64 v124; // r13
  __int64 v125; // rbx
  __int64 v126; // r14
  __int64 v127; // rax
  __int64 v128; // rbx
  unsigned __int64 v129; // rdi
  __int64 v130; // rdi
  char *v131; // rsi
  char *v132; // r8
  __int64 v133; // rbx
  __int64 *v134; // r13
  unsigned int v135; // r15d
  __int64 v136; // r12
  unsigned int v137; // edi
  __int64 *v138; // rdx
  __int64 v139; // r10
  unsigned int v140; // r11d
  __int64 v141; // r9
  unsigned int v142; // edi
  __int64 *v143; // rdx
  __int64 v144; // r10
  unsigned int v145; // edx
  unsigned int v146; // r13d
  unsigned int v147; // edi
  __int64 *v148; // rdx
  __int64 v149; // r11
  unsigned int v150; // r10d
  unsigned int v151; // edi
  __int64 *v152; // rdx
  __int64 v153; // r15
  unsigned int v154; // edx
  char *v155; // rdi
  __int64 v156; // r12
  unsigned int v157; // esi
  char **v158; // rdx
  __int64 v159; // r8
  __int64 v160; // r9
  unsigned int v161; // edx
  __int64 *v162; // rax
  __int64 v163; // rsi
  __int64 v164; // rdx
  __int64 *v165; // rcx
  __int64 v166; // r8
  __int64 v167; // r9
  char *v168; // rbx
  char v169; // di
  char *v170; // r12
  __int64 v171; // rsi
  _QWORD *v172; // rax
  __int64 v173; // rax
  const __m128i *v174; // r15
  __int64 v175; // r12
  __int64 v176; // rax
  __int64 v177; // rdx
  __int64 v178; // rax
  unsigned int *v179; // rcx
  __int64 v180; // rdx
  unsigned int *v181; // r12
  __int64 v182; // rbx
  __int64 v183; // rax
  __int64 v184; // r15
  __int64 v185; // r13
  _QWORD *v186; // rdx
  _QWORD *v187; // rcx
  _QWORD *v188; // rax
  _QWORD *v189; // rax
  _QWORD *v190; // rdx
  __int64 v191; // rax
  unsigned __int64 v192; // rdx
  __m128i v193; // xmm0
  __int64 *v194; // rbx
  unsigned __int64 v195; // rdi
  unsigned __int64 v196; // r12
  unsigned __int64 v197; // rdi
  __int64 v198; // rax
  _QWORD *v199; // r13
  _QWORD *v200; // rbx
  __int64 v201; // rax
  const __m128i *v202; // rbx
  __int64 v203; // r8
  __int64 v204; // rax
  __m128i v205; // xmm0
  unsigned __int64 v206; // rdx
  __int64 v207; // rax
  _QWORD *v208; // r12
  _QWORD *v209; // rbx
  __int64 v210; // rsi
  _QWORD *v211; // rax
  _QWORD *v212; // rax
  _QWORD *v213; // rdx
  unsigned __int64 v214; // rsi
  int v215; // edi
  unsigned int v216; // ecx
  __int64 v217; // r9
  int v218; // edx
  int v219; // edx
  unsigned int (__fastcall ***v220)(_QWORD); // rax
  const __m128i *v221; // rbx
  __int64 v222; // rdx
  __int64 v223; // rcx
  __int64 v224; // r8
  __int64 v225; // r9
  __m128i *v226; // r13
  __int64 v227; // rax
  __m128i *v228; // r12
  __int64 v229; // rax
  __int64 v230; // rbx
  const char **v231; // rax
  __m128i *v232; // r14
  const char **v233; // r15
  __m128i *v234; // r12
  __int64 v235; // rbx
  const __m128i *v236; // rbx
  __int64 v237; // r15
  int v238; // eax
  int v239; // edx
  int v240; // edx
  int v241; // edx
  int v242; // edi
  int v243; // r11d
  int v244; // r10d
  __int64 v245; // rax
  __int64 v246; // rax
  __m128i v247; // xmm4
  __m128i v248; // xmm0
  __int64 v249; // rax
  __m128i v250; // xmm5
  __m128i v251; // xmm6
  __m128i v252; // xmm7
  __m128i v253; // xmm3
  __int64 *v254; // rbx
  __int64 v255; // r15
  __int64 v256; // rax
  __int64 v257; // r12
  __int64 v258; // r13
  unsigned __int64 v259; // rax
  int v260; // edx
  unsigned __int64 v261; // rax
  __int64 v262; // rcx
  __int64 v263; // rax
  _QWORD *v264; // r15
  __int64 v265; // rax
  __int64 v266; // r10
  int v267; // edx
  __int64 v268; // rax
  unsigned __int64 v269; // rax
  int v270; // edx
  unsigned __int64 v271; // rax
  _QWORD *v272; // rax
  _QWORD *v273; // r12
  const __m128i *v274; // rax
  const __m128i *v275; // rsi
  __int64 v276; // r15
  __m128i *v277; // r10
  __int64 v278; // rax
  int v279; // eax
  int v280; // eax
  unsigned int v281; // edx
  __int64 v282; // rax
  __int64 v283; // rdx
  __int64 v284; // rdx
  __int64 v285; // rdi
  __int64 v286; // rax
  const __m128i *v287; // r12
  __int64 v288; // rax
  __int64 v289; // r13
  __int64 v290; // rdi
  const char *v291; // rsi
  __int64 v292; // r12
  __int64 v293; // r12
  int v294; // eax
  int v295; // eax
  unsigned int v296; // edx
  __int64 v297; // rax
  __int64 v298; // rdx
  __int64 v299; // rdx
  const char **v300; // r12
  const char *v301; // rsi
  __int64 v302; // rsi
  unsigned __int8 *v303; // rsi
  __int64 v304; // rsi
  unsigned __int8 *v305; // rsi
  signed __int64 v306; // rax
  char *v307; // rdx
  char *v308; // rbx
  int v309; // r9d
  unsigned __int64 v310; // [rsp+0h] [rbp-7C0h]
  __int64 v311; // [rsp+0h] [rbp-7C0h]
  __int64 v312; // [rsp+18h] [rbp-7A8h]
  __int64 v313; // [rsp+28h] [rbp-798h]
  __int64 v316; // [rsp+40h] [rbp-780h]
  __int64 *v317; // [rsp+48h] [rbp-778h]
  const __m128i *v318; // [rsp+48h] [rbp-778h]
  __int64 v319; // [rsp+48h] [rbp-778h]
  char v320; // [rsp+60h] [rbp-760h]
  const __m128i *v321; // [rsp+88h] [rbp-738h]
  __int64 *v322; // [rsp+88h] [rbp-738h]
  __m128i v323; // [rsp+90h] [rbp-730h] BYREF
  __int64 *v324; // [rsp+A0h] [rbp-720h]
  __m128i *v325; // [rsp+A8h] [rbp-718h]
  __int64 v326; // [rsp+B0h] [rbp-710h]
  __int64 v327; // [rsp+B8h] [rbp-708h]
  __m128i v328; // [rsp+C0h] [rbp-700h] BYREF
  const __m128i *v329; // [rsp+D0h] [rbp-6F0h]
  void *src; // [rsp+D8h] [rbp-6E8h]
  char *v331; // [rsp+E0h] [rbp-6E0h]
  const __m128i *v332; // [rsp+E8h] [rbp-6D8h]
  const __m128i *v333; // [rsp+F8h] [rbp-6C8h] BYREF
  __m128i v334; // [rsp+100h] [rbp-6C0h] BYREF
  void (__fastcall *v335)(__m128i *, __m128i *, __int64); // [rsp+110h] [rbp-6B0h]
  char (__fastcall *v336)(__int64 *, __int64 *); // [rsp+118h] [rbp-6A8h]
  void *v337; // [rsp+120h] [rbp-6A0h] BYREF
  __m128i v338; // [rsp+128h] [rbp-698h] BYREF
  __int64 (__fastcall *v339)(char *, __m128i *, int); // [rsp+138h] [rbp-688h]
  char (__fastcall *v340)(__int64 *, __int64 *); // [rsp+140h] [rbp-680h]
  __int64 *v341; // [rsp+150h] [rbp-670h] BYREF
  __int64 v342; // [rsp+158h] [rbp-668h]
  _BYTE v343[64]; // [rsp+160h] [rbp-660h] BYREF
  void *dest[16]; // [rsp+1A0h] [rbp-620h] BYREF
  unsigned __int64 v345; // [rsp+220h] [rbp-5A0h] BYREF
  unsigned __int64 v346; // [rsp+228h] [rbp-598h]
  __int64 v347; // [rsp+230h] [rbp-590h] BYREF
  int v348; // [rsp+238h] [rbp-588h]
  char v349; // [rsp+23Ch] [rbp-584h]
  _QWORD v350[8]; // [rsp+240h] [rbp-580h] BYREF
  unsigned __int64 v351; // [rsp+280h] [rbp-540h] BYREF
  unsigned __int64 v352; // [rsp+288h] [rbp-538h]
  unsigned __int64 v353; // [rsp+290h] [rbp-530h]
  const char *v354; // [rsp+2A0h] [rbp-520h] BYREF
  unsigned __int64 v355; // [rsp+2A8h] [rbp-518h] BYREF
  __int64 v356; // [rsp+2B0h] [rbp-510h] BYREF
  __m128i v357; // [rsp+2B8h] [rbp-508h] BYREF
  __m128i v358; // [rsp+2C8h] [rbp-4F8h] BYREF
  __m128i v359; // [rsp+2D8h] [rbp-4E8h] BYREF
  __m128i v360; // [rsp+2E8h] [rbp-4D8h] BYREF
  __int64 v361; // [rsp+2F8h] [rbp-4C8h]
  unsigned __int64 v362; // [rsp+300h] [rbp-4C0h] BYREF
  unsigned __int64 v363; // [rsp+308h] [rbp-4B8h]
  unsigned __int64 v364; // [rsp+310h] [rbp-4B0h]
  __int64 v365; // [rsp+320h] [rbp-4A0h] BYREF
  unsigned __int64 v366; // [rsp+328h] [rbp-498h]
  __int64 v367; // [rsp+330h] [rbp-490h]
  __int64 v368; // [rsp+338h] [rbp-488h]
  unsigned int v369; // [rsp+340h] [rbp-480h] BYREF
  __int64 v370; // [rsp+348h] [rbp-478h]
  __int64 *v371; // [rsp+350h] [rbp-470h]
  __int64 v372; // [rsp+358h] [rbp-468h]
  __int64 *v373; // [rsp+360h] [rbp-460h]
  __int64 v374; // [rsp+368h] [rbp-458h]
  char v375[8]; // [rsp+370h] [rbp-450h] BYREF
  __int64 v376; // [rsp+378h] [rbp-448h]
  _BYTE *v377; // [rsp+380h] [rbp-440h]
  const __m128i *v378; // [rsp+388h] [rbp-438h]
  void *v379; // [rsp+390h] [rbp-430h]
  _QWORD *v380; // [rsp+398h] [rbp-428h]
  unsigned int v381; // [rsp+3A8h] [rbp-418h]
  _QWORD *v382; // [rsp+3B8h] [rbp-408h]
  unsigned int v383; // [rsp+3C8h] [rbp-3F8h]
  char v384; // [rsp+3D0h] [rbp-3F0h]
  __int64 *v385; // [rsp+3E0h] [rbp-3E0h]
  unsigned __int64 v386; // [rsp+3F0h] [rbp-3D0h]
  __int64 *v387; // [rsp+420h] [rbp-3A0h] BYREF
  unsigned __int64 v388; // [rsp+428h] [rbp-398h] BYREF
  __int64 v389; // [rsp+430h] [rbp-390h] BYREF
  __int64 v390; // [rsp+438h] [rbp-388h]
  _QWORD v391[3]; // [rsp+440h] [rbp-380h] BYREF
  int v392; // [rsp+458h] [rbp-368h]
  __int64 v393; // [rsp+460h] [rbp-360h]
  __int64 v394; // [rsp+468h] [rbp-358h]
  __int64 v395; // [rsp+470h] [rbp-350h]
  __int64 v396; // [rsp+478h] [rbp-348h]
  __int64 *v397; // [rsp+480h] [rbp-340h]
  unsigned __int64 v398; // [rsp+488h] [rbp-338h]
  unsigned __int64 v399; // [rsp+490h] [rbp-330h]
  __int64 v400; // [rsp+498h] [rbp-328h] BYREF
  unsigned __int64 v401; // [rsp+4A0h] [rbp-320h]
  char *v402; // [rsp+4A8h] [rbp-318h]
  __int64 v403; // [rsp+4B0h] [rbp-310h]
  int v404; // [rsp+4B8h] [rbp-308h] BYREF
  char v405; // [rsp+4BCh] [rbp-304h]
  char v406; // [rsp+4C0h] [rbp-300h] BYREF
  const __m128i *v407; // [rsp+4F8h] [rbp-2C8h]
  const __m128i *v408; // [rsp+500h] [rbp-2C0h]
  void *v409; // [rsp+508h] [rbp-2B8h]
  __int64 v410; // [rsp+540h] [rbp-280h]
  __int64 v411; // [rsp+548h] [rbp-278h]
  __int64 v412; // [rsp+550h] [rbp-270h]
  int v413; // [rsp+558h] [rbp-268h]
  char *v414; // [rsp+560h] [rbp-260h]
  __int64 v415; // [rsp+568h] [rbp-258h]
  char v416; // [rsp+570h] [rbp-250h] BYREF
  __int64 v417; // [rsp+5A0h] [rbp-220h]
  __int64 v418; // [rsp+5A8h] [rbp-218h]
  __int64 v419; // [rsp+5B0h] [rbp-210h]
  int v420; // [rsp+5B8h] [rbp-208h]
  __int64 v421; // [rsp+5C0h] [rbp-200h]
  char *v422; // [rsp+5C8h] [rbp-1F8h]
  __int64 v423; // [rsp+5D0h] [rbp-1F0h]
  int v424; // [rsp+5D8h] [rbp-1E8h]
  char v425; // [rsp+5DCh] [rbp-1E4h]
  char v426; // [rsp+5E0h] [rbp-1E0h] BYREF
  __int64 v427; // [rsp+5F0h] [rbp-1D0h]
  __int64 v428; // [rsp+5F8h] [rbp-1C8h]
  __int64 v429; // [rsp+600h] [rbp-1C0h]
  __int64 v430; // [rsp+608h] [rbp-1B8h]
  __int64 v431; // [rsp+610h] [rbp-1B0h]
  __int64 v432; // [rsp+618h] [rbp-1A8h]
  __int16 v433; // [rsp+620h] [rbp-1A0h]
  char v434; // [rsp+622h] [rbp-19Eh]
  char *v435; // [rsp+628h] [rbp-198h]
  __int64 v436; // [rsp+630h] [rbp-190h]
  char v437; // [rsp+638h] [rbp-188h] BYREF
  __int64 v438; // [rsp+658h] [rbp-168h]
  __int64 v439; // [rsp+660h] [rbp-160h]
  __int16 v440; // [rsp+668h] [rbp-158h]
  __int64 v441; // [rsp+670h] [rbp-150h]
  _QWORD *v442; // [rsp+678h] [rbp-148h]
  void **v443; // [rsp+680h] [rbp-140h]
  __int64 v444; // [rsp+688h] [rbp-138h]
  int v445; // [rsp+690h] [rbp-130h]
  __int16 v446; // [rsp+694h] [rbp-12Ch]
  char v447; // [rsp+696h] [rbp-12Ah]
  __int64 v448; // [rsp+698h] [rbp-128h]
  __int64 v449; // [rsp+6A0h] [rbp-120h]
  _QWORD v450[3]; // [rsp+6A8h] [rbp-118h] BYREF
  __m128i v451; // [rsp+6C0h] [rbp-100h]
  __m128i v452; // [rsp+6D0h] [rbp-F0h]
  __m128i v453; // [rsp+6E0h] [rbp-E0h]
  __m128i v454; // [rsp+6F0h] [rbp-D0h]
  __int64 v455; // [rsp+700h] [rbp-C0h]
  void *v456; // [rsp+708h] [rbp-B8h] BYREF
  char v457[16]; // [rsp+710h] [rbp-B0h] BYREF
  __int64 (__fastcall *v458)(char *, __m128i *, int); // [rsp+720h] [rbp-A0h]
  char (__fastcall *v459)(__int64 *, __int64 *); // [rsp+728h] [rbp-98h]
  char *v460; // [rsp+730h] [rbp-90h]
  __int64 v461; // [rsp+738h] [rbp-88h]
  char v462; // [rsp+740h] [rbp-80h] BYREF
  const char *v463; // [rsp+780h] [rbp-40h]

  v7 = *(const __m128i **)(a1 + 32);
  v341 = (__int64 *)v343;
  v342 = 0x800000000LL;
  v8 = *(const __m128i **)(a1 + 40);
  v327 = a1;
  v326 = a2;
  v331 = (char *)a5;
  v328.m128i_i64[0] = a6;
  v321 = v8;
  if ( v7 == v8 )
  {
    LOBYTE(src) = 0;
    return (unsigned __int8)src;
  }
  v325 = (__m128i *)v7;
  v9 = &v354;
  v329 = (const __m128i *)&v345;
  v332 = (const __m128i *)&v387;
  v324 = (__int64 *)&v369;
  v323.m128i_i64[0] = (__int64)&v400;
  LOBYTE(src) = 0;
  do
  {
    v10 = (__int64 *)v325->m128i_i64[0];
    v351 = 0;
    memset(dest, 0, 0x78u);
    dest[1] = &dest[4];
    v347 = 0x100000008LL;
    v350[0] = v10;
    v387 = v10;
    LODWORD(dest[2]) = 8;
    BYTE4(dest[3]) = 1;
    v346 = (unsigned __int64)v350;
    v352 = 0;
    v353 = 0;
    v348 = 0;
    v349 = 1;
    v345 = 1;
    LOBYTE(v389) = 0;
    sub_2839E90((__int64)&v351, v332);
    sub_C8CF70((__int64)&v365, v324, 8, (__int64)&dest[4], (__int64)dest);
    v11 = dest[12];
    memset(&dest[12], 0, 24);
    v377 = v11;
    v378 = (const __m128i *)dest[13];
    v379 = dest[14];
    sub_C8CF70((__int64)&v354, &v357.m128i_u64[1], 8, (__int64)v350, (__int64)v329);
    v12 = v351;
    v351 = 0;
    v362 = v12;
    v13 = v352;
    v352 = 0;
    v363 = v13;
    v14 = v353;
    v353 = 0;
    v364 = v14;
    sub_C8CF70((__int64)v332, v391, 8, (__int64)&v357.m128i_i64[1], (__int64)&v354);
    v15 = v362;
    v362 = 0;
    v397 = (__int64 *)v15;
    v16 = v363;
    v363 = 0;
    v398 = v16;
    v17 = v364;
    v364 = 0;
    v399 = v17;
    sub_C8CF70(v323.m128i_i64[0], &v404, 8, (__int64)v324, (__int64)&v365);
    v21 = (unsigned __int64)v377;
    v377 = 0;
    v407 = (const __m128i *)v21;
    v22 = v378;
    v378 = 0;
    v408 = v22;
    v23 = v379;
    v379 = 0;
    v409 = v23;
    if ( v362 )
      j_j___libc_free_0(v362);
    if ( !v357.m128i_i8[4] )
      _libc_free(v355);
    if ( v377 )
      j_j___libc_free_0((unsigned __int64)v377);
    if ( !BYTE4(v368) )
      _libc_free(v366);
    if ( v351 )
      j_j___libc_free_0(v351);
    if ( !v349 )
      _libc_free(v346);
    if ( dest[12] )
      j_j___libc_free_0((unsigned __int64)dest[12]);
    if ( !BYTE4(dest[3]) )
      _libc_free((unsigned __int64)dest[1]);
    sub_C8CD80((__int64)&v354, (__int64)&v357.m128i_i64[1], (__int64)v332, v18, v19, v20);
    v26 = v398;
    v27 = v397;
    v362 = 0;
    v363 = 0;
    v364 = 0;
    v28 = v398 - (_QWORD)v397;
    if ( (__int64 *)v398 == v397 )
    {
      v30 = 0;
      v31 = 0;
    }
    else
    {
      if ( v28 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_398;
      v310 = v398 - (_QWORD)v397;
      v29 = sub_22077B0(v398 - (_QWORD)v397);
      v26 = v398;
      v27 = v397;
      v30 = v310;
      v31 = v29;
    }
    v362 = v31;
    v363 = v31;
    v364 = v31 + v30;
    if ( v27 != (__int64 *)v26 )
    {
      v32 = (__m128i *)v31;
      v33 = (const __m128i *)v27;
      do
      {
        if ( v32 )
        {
          *v32 = _mm_loadu_si128(v33);
          v24 = v33[1].m128i_i64[0];
          v32[1].m128i_i64[0] = v24;
        }
        v33 = (const __m128i *)((char *)v33 + 24);
        v32 = (__m128i *)((char *)v32 + 24);
      }
      while ( v33 != (const __m128i *)v26 );
      v31 += 8 * ((unsigned __int64)((char *)&v33[-2].m128i_u64[1] - (char *)v27) >> 3) + 24;
    }
    v363 = v31;
    v27 = &v365;
    sub_C8CD80((__int64)&v365, (__int64)v324, v323.m128i_i64[0], v31, v24, v25);
    v34 = v408;
    v26 = (unsigned __int64)v407;
    v377 = 0;
    v378 = 0;
    v379 = 0;
    v28 = (char *)v408 - (char *)v407;
    if ( v408 == v407 )
    {
      v36 = 0;
      v37 = 0;
    }
    else
    {
      if ( v28 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_398:
        sub_4261EA(v27, v26, v28);
      v311 = (char *)v408 - (char *)v407;
      v35 = sub_22077B0((char *)v408 - (char *)v407);
      v34 = v408;
      v26 = (unsigned __int64)v407;
      v36 = v311;
      v37 = v35;
    }
    v377 = (_BYTE *)v37;
    v379 = (void *)(v37 + v36);
    v38 = (__m128i *)v37;
    v378 = (const __m128i *)v37;
    if ( (const __m128i *)v26 != v34 )
    {
      v39 = (const __m128i *)v26;
      do
      {
        if ( v38 )
        {
          *v38 = _mm_loadu_si128(v39);
          v38[1].m128i_i64[0] = v39[1].m128i_i64[0];
        }
        v39 = (const __m128i *)((char *)v39 + 24);
        v38 = (__m128i *)((char *)v38 + 24);
      }
      while ( v39 != v34 );
      v38 = (__m128i *)(v37 + 8 * (((unsigned __int64)&v39[-2].m128i_u64[1] - v26) >> 3) + 24);
    }
    v40 = v363;
    v41 = v362;
    v378 = v38;
    if ( (__m128i *)(v363 - v362) == (__m128i *)((char *)v38 - v37) )
      goto LABEL_53;
    do
    {
LABEL_38:
      v42 = *(_QWORD *)(v40 - 24);
      LOBYTE(src) = sub_F6AC10((char *)v42, v326, v327, (__int64)v331, v328.m128i_i64[0], 0, 0) | (unsigned __int8)src;
      if ( *(_QWORD *)(v42 + 16) == *(_QWORD *)(v42 + 8) )
      {
        v65 = (unsigned int)v342;
        v43 = HIDWORD(v342);
        v66 = (unsigned int)v342 + 1LL;
        if ( v66 > HIDWORD(v342) )
        {
          sub_C8D5F0((__int64)&v341, v343, v66, 8u, v44, v45);
          v65 = (unsigned int)v342;
        }
        v341[v65] = v42;
        LODWORD(v342) = v342 + 1;
      }
      v46 = v363;
      while ( 1 )
      {
        v47 = *(_QWORD *)(v46 - 24);
        if ( *(_BYTE *)(v46 - 8) )
          break;
        v48 = *(__int64 **)(v47 + 8);
        *(_BYTE *)(v46 - 8) = 1;
        *(_QWORD *)(v46 - 16) = v48;
        if ( *(__int64 **)(v47 + 16) != v48 )
          goto LABEL_42;
LABEL_48:
        v363 -= 24LL;
        v41 = v362;
        v46 = v363;
        if ( v363 == v362 )
        {
          v40 = v362;
          goto LABEL_52;
        }
      }
      while ( 1 )
      {
        while ( 1 )
        {
          v48 = *(__int64 **)(v46 - 16);
          if ( *(__int64 **)(v47 + 16) == v48 )
            goto LABEL_48;
LABEL_42:
          v49 = v48 + 1;
          *(_QWORD *)(v46 - 16) = v48 + 1;
          v50 = *v48;
          if ( v357.m128i_i8[4] )
            break;
LABEL_50:
          sub_C8CC70((__int64)&v354, v50, (__int64)v49, v43, v44, v45);
          if ( v52 )
            goto LABEL_51;
        }
        v51 = (_QWORD *)v355;
        v49 = (__int64 *)(v355 + 8LL * HIDWORD(v356));
        if ( (__int64 *)v355 == v49 )
          break;
        while ( v50 != *v51 )
        {
          if ( v49 == ++v51 )
            goto LABEL_96;
        }
      }
LABEL_96:
      if ( HIDWORD(v356) >= (unsigned int)v356 )
        goto LABEL_50;
      ++HIDWORD(v356);
      *v49 = v50;
      ++v354;
LABEL_51:
      v345 = v50;
      LOBYTE(v347) = 0;
      sub_2839E90((__int64)&v362, v329);
      v41 = v362;
      v40 = v363;
LABEL_52:
      v37 = (unsigned __int64)v377;
    }
    while ( v40 - v41 != (char *)v378 - v377 );
LABEL_53:
    if ( v41 != v40 )
    {
      v53 = v37;
      while ( *(_QWORD *)v41 == *(_QWORD *)v53 )
      {
        v54 = *(_BYTE *)(v41 + 16);
        if ( v54 != *(_BYTE *)(v53 + 16) || v54 && *(_QWORD *)(v41 + 8) != *(_QWORD *)(v53 + 8) )
          break;
        v41 += 24LL;
        v53 += 24LL;
        if ( v41 == v40 )
          goto LABEL_60;
      }
      goto LABEL_38;
    }
LABEL_60:
    if ( v37 )
      j_j___libc_free_0(v37);
    if ( !BYTE4(v368) )
      _libc_free(v366);
    if ( v362 )
      j_j___libc_free_0(v362);
    if ( !v357.m128i_i8[4] )
      _libc_free(v355);
    if ( v407 )
      j_j___libc_free_0((unsigned __int64)v407);
    if ( !BYTE4(v403) )
      _libc_free(v401);
    if ( v397 )
      j_j___libc_free_0((unsigned __int64)v397);
    if ( !BYTE4(v390) )
      _libc_free(v388);
    v325 = (__m128i *)((char *)v325 + 8);
  }
  while ( v321 != v325 );
  v322 = &v341[(unsigned int)v342];
  if ( v341 != v322 )
  {
    v324 = v341;
LABEL_79:
    v55 = *v324;
    v56 = sub_D47930(*v324);
    if ( !v56 )
      goto LABEL_91;
    v57 = *(_QWORD *)(v56 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v57 == v56 + 48 )
      goto LABEL_91;
    if ( !v57 )
      BUG();
    v58 = v57 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v57 - 24) - 30 > 0xA )
      goto LABEL_91;
    v59 = sub_B46E30(v57 - 24);
    if ( !v59 )
      goto LABEL_91;
    v60 = 0;
    v331 = (char *)(v55 + 56);
    while ( 1 )
    {
      v61 = sub_B46EC0(v58, v60);
      if ( !*(_BYTE *)(v55 + 84) )
        break;
      v62 = *(_QWORD **)(v55 + 64);
      v63 = &v62[*(unsigned int *)(v55 + 76)];
      if ( v62 == v63 )
        goto LABEL_104;
      while ( v61 != *v62 )
      {
        if ( v63 == ++v62 )
          goto LABEL_104;
      }
LABEL_90:
      if ( v59 == ++v60 )
        goto LABEL_91;
    }
    if ( sub_C8CA60((__int64)v331, v61) )
      goto LABEL_90;
LABEL_104:
    if ( !sub_D46F00(v55) )
      goto LABEL_91;
    v67 = (__int64 *)sub_D440B0(a7, v55);
    v365 = v55;
    v371 = v67;
    v370 = v327;
    v366 = 0;
    v372 = v326;
    v367 = 0;
    v373 = a3;
    v368 = 0;
    v374 = a4;
    v369 = 0;
    v68 = *v67;
    v331 = v375;
    sub_DF2CB0((__int64)v375, v68);
    v333 = 0;
    v72 = v371[2];
    v320 = *(_BYTE *)(v72 + 232);
    if ( !v320 )
      goto LABEL_247;
    v387 = 0;
    v389 = 4;
    v388 = (unsigned __int64)v391;
    LODWORD(v390) = 0;
    BYTE4(v390) = 1;
    v73 = *(unsigned int **)(v72 + 240);
    v328.m128i_i64[0] = (__int64)&v73[3 * *(unsigned int *)(v72 + 248)];
    if ( v73 == (unsigned int *)v328.m128i_i64[0] )
    {
LABEL_140:
      if ( !v333 )
        goto LABEL_247;
      v92 = v371[2];
      v387 = 0;
      v388 = 0;
      v389 = 0;
      LODWORD(v390) = 0;
      if ( *(_DWORD *)(v92 + 64) )
      {
        v93 = 0;
        v94 = 0;
        v95 = 0;
        v96 = 0;
        while ( 1 )
        {
          v102 = (__int64 *)(*(_QWORD *)(v92 + 56) + 8 * v95);
          if ( v93 )
          {
            v97 = 1;
            v98 = 0;
            v99 = (v93 - 1) & (((unsigned int)*v102 >> 9) ^ ((unsigned int)*v102 >> 4));
            v100 = v94 + 16LL * v99;
            v101 = *(_QWORD *)v100;
            if ( *v102 == *(_QWORD *)v100 )
              goto LABEL_144;
            while ( v101 != -4096 )
            {
              if ( v101 == -8192 && !v98 )
                v98 = v100;
              v99 = (v93 - 1) & (v97 + v99);
              v100 = v94 + 16LL * v99;
              v101 = *(_QWORD *)v100;
              if ( *v102 == *(_QWORD *)v100 )
                goto LABEL_144;
              ++v97;
            }
            if ( v98 )
              v100 = v98;
            v387 = (__int64 *)((char *)v387 + 1);
            v103 = v389 + 1;
            if ( 4 * ((int)v389 + 1) < 3 * v93 )
            {
              if ( v93 - (v103 + HIDWORD(v389)) > v93 >> 3 )
                goto LABEL_150;
              sub_9BAAD0((__int64)v332, v93);
              if ( !(_DWORD)v390 )
              {
LABEL_506:
                LODWORD(v389) = v389 + 1;
                BUG();
              }
              v214 = 0;
              v103 = v389 + 1;
              v215 = 1;
              v216 = (v390 - 1) & (((unsigned int)*v102 >> 9) ^ ((unsigned int)*v102 >> 4));
              v100 = v388 + 16LL * v216;
              v217 = *(_QWORD *)v100;
              if ( *v102 == *(_QWORD *)v100 )
                goto LABEL_150;
              while ( v217 != -4096 )
              {
                if ( !v214 && v217 == -8192 )
                  v214 = v100;
                v216 = (v390 - 1) & (v215 + v216);
                v100 = v388 + 16LL * v216;
                v217 = *(_QWORD *)v100;
                if ( *v102 == *(_QWORD *)v100 )
                  goto LABEL_150;
                ++v215;
              }
              goto LABEL_395;
            }
          }
          else
          {
            v387 = (__int64 *)((char *)v387 + 1);
          }
          sub_9BAAD0((__int64)v332, 2 * v93);
          if ( !(_DWORD)v390 )
            goto LABEL_506;
          v103 = v389 + 1;
          v104 = (v390 - 1) & (((unsigned int)*v102 >> 9) ^ ((unsigned int)*v102 >> 4));
          v100 = v388 + 16LL * v104;
          v105 = *(_QWORD *)v100;
          if ( *v102 == *(_QWORD *)v100 )
            goto LABEL_150;
          v242 = 1;
          v214 = 0;
          while ( v105 != -4096 )
          {
            if ( v105 == -8192 && !v214 )
              v214 = v100;
            v104 = (v390 - 1) & (v242 + v104);
            v100 = v388 + 16LL * v104;
            v105 = *(_QWORD *)v100;
            if ( *v102 == *(_QWORD *)v100 )
              goto LABEL_150;
            ++v242;
          }
LABEL_395:
          if ( v214 )
            v100 = v214;
LABEL_150:
          LODWORD(v389) = v103;
          if ( *(_QWORD *)v100 != -4096 )
            --HIDWORD(v389);
          v106 = *v102;
          *(_DWORD *)(v100 + 8) = 0;
          *(_QWORD *)v100 = v106;
LABEL_144:
          *(_DWORD *)(v100 + 8) = v96;
          v95 = (unsigned int)(v96 + 1);
          v96 = v95;
          if ( *(_DWORD *)(v92 + 64) <= (unsigned int)v95 )
            break;
          v94 = v388;
          v93 = v390;
        }
      }
      sub_C7D6A0(v367, 16LL * v369, 8);
      ++v366;
      v367 = v388;
      v387 = (__int64 *)((char *)v387 + 1);
      v368 = v389;
      v388 = 0;
      v369 = v390;
      v389 = 0;
      LODWORD(v390) = 0;
      sub_C7D6A0(0, 0, 8);
      sub_28397C0((__int64)&v365, &v333);
      v325 = (__m128i *)v333;
      if ( !v333 )
        goto LABEL_247;
      v317 = (__int64 *)v9;
      dest[0] = &dest[2];
      dest[1] = (void *)0x400000000LL;
      v328.m128i_i64[0] = (__int64)&v389;
LABEL_155:
      v108 = *(_QWORD *)(v325[1].m128i_i64[0] + 40);
      v387 = (__int64 *)v328.m128i_i64[0];
      v388 = 0x800000000LL;
      v109 = *(_QWORD *)(**(_QWORD **)(v365 + 32) + 16LL);
      if ( !v109 )
        goto LABEL_180;
      while ( 1 )
      {
        v110 = *(_QWORD *)(v109 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v110 - 30) <= 0xAu )
          break;
        v109 = *(_QWORD *)(v109 + 8);
        if ( !v109 )
        {
LABEL_180:
          v122 = (__int64 *)v328.m128i_i64[0];
          v126 = v328.m128i_i64[0];
LABEL_181:
          v129 = (unsigned __int64)v387;
          if ( v387 == (__int64 *)v328.m128i_i64[0] )
            goto LABEL_184;
LABEL_182:
          _libc_free(v129);
          goto LABEL_183;
        }
      }
      v111 = v108;
      v112 = v372;
      v113 = v365 + 56;
      v114 = v365;
      v115 = *(_QWORD *)(v110 + 40);
      if ( !*(_BYTE *)(v365 + 84) )
        goto LABEL_168;
LABEL_158:
      v116 = *(_QWORD **)(v114 + 64);
      v117 = &v116[*(unsigned int *)(v114 + 76)];
      if ( v116 != v117 )
      {
        while ( v115 != *v116 )
        {
          if ( v117 == ++v116 )
            goto LABEL_165;
        }
LABEL_162:
        v118 = (unsigned int)v388;
        v119 = (unsigned int)v388 + 1LL;
        if ( v119 > HIDWORD(v388) )
        {
          v323.m128i_i64[0] = v111;
          sub_C8D5F0((__int64)v332, (const void *)v328.m128i_i64[0], v119, 8u, v111, v107);
          v118 = (unsigned int)v388;
          v111 = v323.m128i_i64[0];
        }
        v387[v118] = v115;
        LODWORD(v388) = v388 + 1;
      }
      while ( 1 )
      {
LABEL_165:
        v109 = *(_QWORD *)(v109 + 8);
        if ( !v109 )
        {
LABEL_170:
          v122 = v387;
          v123 = v112;
          v124 = v111;
          v125 = 8LL * (unsigned int)v388;
          v126 = (__int64)&v387[(unsigned __int64)v125 / 8];
          v127 = v125 >> 3;
          v128 = v125 >> 5;
          if ( !v128 )
          {
LABEL_273:
            switch ( v127 )
            {
              case 2LL:
LABEL_285:
                if ( (unsigned __int8)sub_B19720(v123, v124, *v122) )
                {
                  ++v122;
LABEL_287:
                  if ( (unsigned __int8)sub_B19720(v123, v124, *v122) )
                  {
                    v129 = (unsigned __int64)v387;
                    v122 = (__int64 *)v126;
                    if ( v387 == (__int64 *)v328.m128i_i64[0] )
                    {
LABEL_184:
                      v130 = v325->m128i_i64[1];
                      if ( *(_QWORD *)(v130 + 40) == **(_QWORD **)(v365 + 32) )
                      {
                        v202 = v325;
                        if ( (unsigned __int8)sub_2839560(v130, v325[1].m128i_i64[0], (__int64)v331, v365) )
                        {
                          v204 = LODWORD(dest[1]);
                          v205 = _mm_loadu_si128((const __m128i *)&v202->m128i_u64[1]);
                          v206 = LODWORD(dest[1]) + 1LL;
                          if ( v206 > HIDWORD(dest[1]) )
                          {
                            v323 = v205;
                            sub_C8D5F0((__int64)dest, &dest[2], v206, 0x10u, v203, v107);
                            v204 = LODWORD(dest[1]);
                            v205 = _mm_load_si128(&v323);
                          }
                          *((__m128i *)dest[0] + v204) = v205;
                          ++LODWORD(dest[1]);
                        }
                      }
                      goto LABEL_185;
                    }
                    goto LABEL_182;
                  }
                }
                break;
              case 3LL:
                if ( (unsigned __int8)sub_B19720(v123, v124, *v122) )
                {
                  ++v122;
                  goto LABEL_285;
                }
                break;
              case 1LL:
                goto LABEL_287;
              default:
                v122 = (__int64 *)v126;
                goto LABEL_181;
            }
LABEL_177:
            v129 = (unsigned __int64)v387;
            if ( v387 == (__int64 *)v328.m128i_i64[0] )
              goto LABEL_183;
            goto LABEL_182;
          }
          while ( 1 )
          {
            if ( !(unsigned __int8)sub_B19720(v123, v124, *v122) )
              goto LABEL_177;
            if ( !(unsigned __int8)sub_B19720(v123, v124, v122[1]) )
            {
              v129 = (unsigned __int64)v387;
              ++v122;
              if ( v387 == (__int64 *)v328.m128i_i64[0] )
                goto LABEL_183;
              goto LABEL_182;
            }
            if ( !(unsigned __int8)sub_B19720(v123, v124, v122[2]) )
            {
              v129 = (unsigned __int64)v387;
              v122 += 2;
              if ( v387 == (__int64 *)v328.m128i_i64[0] )
                goto LABEL_183;
              goto LABEL_182;
            }
            if ( !(unsigned __int8)sub_B19720(v123, v124, v122[3]) )
              break;
            v122 += 4;
            if ( !--v128 )
            {
              v127 = (v126 - (__int64)v122) >> 3;
              goto LABEL_273;
            }
          }
          v129 = (unsigned __int64)v387;
          v122 += 3;
          if ( v387 != (__int64 *)v328.m128i_i64[0] )
            goto LABEL_182;
LABEL_183:
          if ( v122 == (__int64 *)v126 )
            goto LABEL_184;
LABEL_185:
          v325 = (__m128i *)v325->m128i_i64[0];
          if ( !v325 )
          {
            v9 = (const char **)v317;
            if ( !LODWORD(dest[1]) )
              goto LABEL_243;
            v131 = (char *)dest[0];
            v132 = (char *)dest[0] + 16;
            v133 = *(_QWORD *)dest[0];
            v328.m128i_i64[0] = (__int64)dest[0] + 16 * LODWORD(dest[1]);
            if ( (char *)dest[0] + 16 != (void *)v328.m128i_i64[0] )
            {
              v134 = (__int64 *)((char *)dest[0] + 16);
              v135 = v369 - 1;
              v136 = v367 + 16LL * v369;
              do
              {
                if ( v369 )
                {
                  v137 = v135 & (((unsigned int)v133 >> 9) ^ ((unsigned int)v133 >> 4));
                  v138 = (__int64 *)(v367 + 16LL * v137);
                  v139 = *v138;
                  if ( v133 == *v138 )
                  {
LABEL_191:
                    v140 = *((_DWORD *)v138 + 2);
                    v141 = *v134;
                  }
                  else
                  {
                    v241 = 1;
                    while ( v139 != -4096 )
                    {
                      v309 = v241 + 1;
                      v137 = v135 & (v241 + v137);
                      v138 = (__int64 *)(v367 + 16LL * v137);
                      v139 = *v138;
                      if ( *v138 == v133 )
                        goto LABEL_191;
                      v241 = v309;
                    }
                    v140 = *(_DWORD *)(v136 + 8);
                    v141 = *v134;
                  }
                  v142 = v135 & (((unsigned int)v141 >> 9) ^ ((unsigned int)v141 >> 4));
                  v143 = (__int64 *)(v367 + 16LL * v142);
                  v144 = *v143;
                  if ( *v143 == v141 )
                  {
LABEL_193:
                    v145 = *((_DWORD *)v143 + 2);
                  }
                  else
                  {
                    v240 = 1;
                    while ( v144 != -4096 )
                    {
                      v142 = v135 & (v240 + v142);
                      LODWORD(v325) = v240 + 1;
                      v143 = (__int64 *)(v367 + 16LL * v142);
                      v144 = *v143;
                      if ( *v143 == v141 )
                        goto LABEL_193;
                      v240 = (int)v325;
                    }
                    v145 = *(_DWORD *)(v136 + 8);
                  }
                  if ( v145 > v140 )
                    v133 = v141;
                }
                v134 += 2;
              }
              while ( (__int64 *)v328.m128i_i64[0] != v134 );
              v146 = v369 - 1;
              do
              {
                v107 = *((_QWORD *)v132 + 1);
                if ( v369 )
                {
                  v147 = v146 & (((unsigned int)v107 >> 9) ^ ((unsigned int)v107 >> 4));
                  v148 = (__int64 *)(v367 + 16LL * v147);
                  v149 = *v148;
                  if ( *v148 == v107 )
                  {
LABEL_200:
                    v150 = *((_DWORD *)v148 + 2);
                    v107 = *((_QWORD *)v131 + 1);
                  }
                  else
                  {
                    v219 = 1;
                    while ( v149 != -4096 )
                    {
                      v244 = v219 + 1;
                      v147 = v146 & (v219 + v147);
                      v148 = (__int64 *)(v367 + 16LL * v147);
                      v149 = *v148;
                      if ( v107 == *v148 )
                        goto LABEL_200;
                      v219 = v244;
                    }
                    v150 = *(_DWORD *)(v136 + 8);
                    v107 = *((_QWORD *)v131 + 1);
                  }
                  v151 = v146 & (((unsigned int)v107 >> 9) ^ ((unsigned int)v107 >> 4));
                  v152 = (__int64 *)(v367 + 16LL * v151);
                  v153 = *v152;
                  if ( *v152 == v107 )
                  {
LABEL_202:
                    v154 = *((_DWORD *)v152 + 2);
                  }
                  else
                  {
                    v218 = 1;
                    while ( v153 != -4096 )
                    {
                      v243 = v218 + 1;
                      v151 = v146 & (v218 + v151);
                      v152 = (__int64 *)(v367 + 16LL * v151);
                      v153 = *v152;
                      if ( *v152 == v107 )
                        goto LABEL_202;
                      v218 = v243;
                    }
                    v154 = *(_DWORD *)(v136 + 8);
                  }
                  if ( v154 > v150 )
                    v131 = v132;
                }
                v132 += 16;
              }
              while ( (char *)v328.m128i_i64[0] != v132 );
            }
            v155 = (char *)*((_QWORD *)v131 + 1);
            v354 = 0;
            v357.m128i_i8[4] = 1;
            v356 = 4;
            v355 = (unsigned __int64)&v357.m128i_u64[1];
            v357.m128i_i32[0] = 0;
            v156 = v371[2];
            if ( v369 )
            {
              v157 = (v369 - 1) & (((unsigned int)v155 >> 4) ^ ((unsigned int)v155 >> 9));
              v158 = (char **)(v367 + 16LL * v157);
              v132 = *v158;
              if ( v155 == *v158 )
                goto LABEL_208;
              v239 = 1;
              while ( v132 != (char *)-4096LL )
              {
                v107 = (unsigned int)(v239 + 1);
                v157 = (v369 - 1) & (v239 + v157);
                v158 = (char **)(v367 + 16LL * v157);
                v132 = *v158;
                if ( v155 == *v158 )
                  goto LABEL_208;
                v239 = v107;
              }
            }
            v158 = (char **)(v367 + 16LL * v369);
LABEL_208:
            sub_28392C0(
              (_BYTE **)(*(_QWORD *)(v156 + 56) + 8LL * *((unsigned int *)v158 + 2) + 8),
              (_BYTE **)(*(_QWORD *)(v156 + 56) + 8LL * *(unsigned int *)(v156 + 64)),
              v317,
              v369,
              (__int64)v132,
              v107);
            if ( v369 )
            {
              v159 = v369 - 1;
              v161 = v159 & (((unsigned int)v133 >> 9) ^ ((unsigned int)v133 >> 4));
              v162 = (__int64 *)(v367 + 16LL * v161);
              v163 = *v162;
              if ( v133 == *v162 )
                goto LABEL_210;
              v238 = 1;
              while ( v163 != -4096 )
              {
                v160 = (unsigned int)(v238 + 1);
                v161 = v159 & (v238 + v161);
                v162 = (__int64 *)(v367 + 16LL * v161);
                v163 = *v162;
                if ( v133 == *v162 )
                  goto LABEL_210;
                v238 = v160;
              }
            }
            v162 = (__int64 *)(v367 + 16LL * v369);
LABEL_210:
            sub_28392C0(
              *(_BYTE ***)(v156 + 56),
              (_BYTE **)(*(_QWORD *)(v156 + 56) + 8LL * *((unsigned int *)v162 + 2)),
              v317,
              v369,
              v159,
              v160);
            v168 = (char *)dest[0];
            v387 = 0;
            BYTE4(v390) = 1;
            v169 = v320;
            v389 = 4;
            v170 = (char *)dest[0] + 16 * LODWORD(dest[1]);
            v388 = (unsigned __int64)v391;
            LODWORD(v390) = 0;
            if ( dest[0] != v170 )
            {
              while ( 2 )
              {
                while ( 1 )
                {
                  v171 = *(_QWORD *)(*(_QWORD *)v168 - 32LL);
                  if ( !v169 )
                    break;
                  v172 = (_QWORD *)v388;
                  v164 = HIDWORD(v389);
                  v165 = (__int64 *)(v388 + 8LL * HIDWORD(v389));
                  if ( (__int64 *)v388 == v165 )
                  {
LABEL_334:
                    if ( HIDWORD(v389) >= (unsigned int)v389 )
                      break;
                    v164 = (unsigned int)(HIDWORD(v389) + 1);
                    v168 += 16;
                    ++HIDWORD(v389);
                    *v165 = v171;
                    v169 = BYTE4(v390);
                    v387 = (__int64 *)((char *)v387 + 1);
                    if ( v170 == v168 )
                      goto LABEL_217;
                  }
                  else
                  {
                    while ( v171 != *v172 )
                    {
                      if ( v165 == ++v172 )
                        goto LABEL_334;
                    }
                    v168 += 16;
                    if ( v170 == v168 )
                      goto LABEL_217;
                  }
                }
                v168 += 16;
                sub_C8CC70((__int64)v332, v171, v164, (__int64)v165, v166, v167);
                v169 = BYTE4(v390);
                if ( v170 == v168 )
                  break;
                continue;
              }
            }
LABEL_217:
            v173 = v371[1];
            v345 = (unsigned __int64)&v347;
            v346 = 0x400000000LL;
            v174 = *(const __m128i **)(v173 + 296);
            v318 = &v174[*(unsigned int *)(v173 + 304)];
            if ( v174 == v318 )
              goto LABEL_236;
            while ( 2 )
            {
              v175 = *(_QWORD *)(v174->m128i_i64[0] + 16);
              v176 = v175 + 4LL * *(unsigned int *)(v174->m128i_i64[0] + 24);
              v323.m128i_i64[0] = v175;
              v316 = v176;
              if ( v175 == v176 )
                goto LABEL_235;
              v325 = (__m128i *)v174;
              while ( 1 )
              {
                v177 = v325->m128i_i64[1];
                v178 = *(unsigned int *)v323.m128i_i64[0];
                v179 = *(unsigned int **)(v177 + 16);
                v180 = *(unsigned int *)(v177 + 24);
                if ( v179 != &v179[v180] )
                  break;
LABEL_324:
                v323.m128i_i64[0] += 4;
                if ( v316 == v323.m128i_i64[0] )
                {
                  v174 = v325;
                  goto LABEL_235;
                }
              }
              v328.m128i_i64[0] = (__int64)&v179[v180];
              v181 = v179;
              v182 = 72 * v178;
              while ( 2 )
              {
                v183 = *(_QWORD *)(v371[1] + 8);
                v184 = *(_QWORD *)(v183 + v182 + 16);
                v185 = *(_QWORD *)(v183 + 72LL * *v181 + 16);
                if ( !v357.m128i_i8[4] )
                {
                  if ( sub_C8CA60((__int64)v9, v184) )
                    goto LABEL_227;
LABEL_321:
                  if ( v357.m128i_i8[4] )
                  {
                    v186 = (_QWORD *)v355;
                    v188 = (_QWORD *)(v355 + 8LL * HIDWORD(v356));
                    if ( v188 == (_QWORD *)v355 )
                    {
LABEL_323:
                      if ( (unsigned int *)v328.m128i_i64[0] == ++v181 )
                        goto LABEL_324;
                      continue;
                    }
LABEL_313:
                    while ( v185 != *v186 )
                    {
                      if ( v188 == ++v186 )
                        goto LABEL_323;
                    }
                  }
                  else if ( !sub_C8CA60((__int64)v9, v185) )
                  {
                    goto LABEL_323;
                  }
                  if ( BYTE4(v390) )
                  {
                    v212 = (_QWORD *)v388;
                    v213 = (_QWORD *)(v388 + 8LL * HIDWORD(v389));
                    if ( (_QWORD *)v388 != v213 )
                    {
                      while ( v184 != *v212 )
                      {
                        if ( v213 == ++v212 )
                          goto LABEL_323;
                      }
                      goto LABEL_232;
                    }
                  }
                  else if ( sub_C8CA60((__int64)v332, v184) )
                  {
                    goto LABEL_232;
                  }
                  goto LABEL_323;
                }
                break;
              }
              v186 = (_QWORD *)v355;
              v187 = (_QWORD *)(v355 + 8LL * HIDWORD(v356));
              if ( (_QWORD *)v355 == v187 )
                goto LABEL_323;
              v188 = (_QWORD *)v355;
              while ( v184 != *v188 )
              {
                if ( v187 == ++v188 )
                  goto LABEL_313;
              }
LABEL_227:
              if ( !BYTE4(v390) )
              {
                if ( sub_C8CA60((__int64)v332, v185) )
                  goto LABEL_232;
                goto LABEL_321;
              }
              v189 = (_QWORD *)v388;
              v190 = (_QWORD *)(v388 + 8LL * HIDWORD(v389));
              if ( (_QWORD *)v388 == v190 )
                goto LABEL_321;
              while ( v185 != *v189 )
              {
                if ( v190 == ++v189 )
                  goto LABEL_321;
              }
LABEL_232:
              v191 = (unsigned int)v346;
              v174 = v325;
              v192 = (unsigned int)v346 + 1LL;
              v193 = _mm_loadu_si128(v325);
              if ( v192 > HIDWORD(v346) )
              {
                v328 = v193;
                sub_C8D5F0((__int64)v329, &v347, v192, 0x10u, v166, v167);
                v191 = (unsigned int)v346;
                v193 = _mm_load_si128(&v328);
              }
              *(__m128i *)(v345 + 16 * v191) = v193;
              LODWORD(v346) = v346 + 1;
LABEL_235:
              if ( v318 != ++v174 )
                continue;
              break;
            }
LABEL_236:
            if ( !BYTE4(v390) )
              _libc_free(v388);
            if ( !v357.m128i_i8[4] )
              _libc_free(v355);
            if ( (unsigned int)v346 <= (unsigned int)qword_4FFFEE8 * (unsigned __int64)LODWORD(dest[1]) )
            {
              v220 = (unsigned int (__fastcall ***)(_QWORD))sub_D9B120(*v371);
              if ( (**v220)(v220) <= (unsigned int)qword_4FFFE08 )
              {
                v328.m128i_i8[0] = sub_D4B3D0(v365);
                if ( v328.m128i_i8[0] )
                {
                  if ( !(_DWORD)v346 )
                  {
                    v245 = sub_D9B120(*v371);
                    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v245 + 8LL))(v245) )
                      goto LABEL_404;
                  }
                  if ( !*((_BYTE *)v371 + 41) && !sub_11F3070(**(_QWORD **)(v365 + 32), v374, v373) )
                  {
                    v221 = v332;
                    sub_2A28870(v332, v371, v345, (unsigned int)v346, v365, v370, v372, v385);
                    sub_F6D5D0((__int64)v9, (__int64)v387, v222, v223, v224, v225);
                    sub_2A28FB0(v221, v9);
                    if ( v354 != (const char *)&v356 )
                      _libc_free((unsigned __int64)v354);
                    v226 = (__m128i *)dest[0];
                    v227 = 16LL * LODWORD(dest[1]);
                    v228 = (__m128i *)((char *)dest[0] + v227);
                    v229 = v227 >> 6;
                    if ( v229 )
                    {
                      v230 = v229 << 6;
                      v231 = v9;
                      src = v228;
                      v232 = (__m128i *)dest[0];
                      v233 = v231;
                      v234 = (__m128i *)((char *)dest[0] + v230);
                      v235 = (__int64)v331;
                      while ( 1 )
                      {
                        if ( *(_WORD *)(sub_DEEF40(v235, *(_QWORD *)(v232->m128i_i64[0] - 32)) + 24) != 8
                          || *(_WORD *)(sub_DEEF40(v235, *(_QWORD *)(v232->m128i_i64[1] - 32)) + 24) != 8 )
                        {
                          v228 = (__m128i *)src;
                          v226 = v232;
                          v9 = v233;
                          goto LABEL_373;
                        }
                        v226 = v232 + 1;
                        if ( *(_WORD *)(sub_DEEF40(v235, *(_QWORD *)(v232[1].m128i_i64[0] - 32)) + 24) != 8 )
                          break;
                        if ( *(_WORD *)(sub_DEEF40(v235, *(_QWORD *)(v232[1].m128i_i64[1] - 32)) + 24) != 8 )
                          break;
                        v226 = v232 + 2;
                        if ( *(_WORD *)(sub_DEEF40(v235, *(_QWORD *)(v232[2].m128i_i64[0] - 32)) + 24) != 8 )
                          break;
                        if ( *(_WORD *)(sub_DEEF40(v235, *(_QWORD *)(v232[2].m128i_i64[1] - 32)) + 24) != 8 )
                          break;
                        v226 = v232 + 3;
                        if ( *(_WORD *)(sub_DEEF40(v235, *(_QWORD *)(v232[3].m128i_i64[0] - 32)) + 24) != 8
                          || *(_WORD *)(sub_DEEF40(v235, *(_QWORD *)(v232[3].m128i_i64[1] - 32)) + 24) != 8 )
                        {
                          break;
                        }
                        v232 += 4;
                        if ( v232 == v234 )
                        {
                          v228 = (__m128i *)src;
                          v226 = v232;
                          v9 = v233;
                          goto LABEL_475;
                        }
                      }
                      v228 = (__m128i *)src;
                      v9 = v233;
LABEL_373:
                      if ( v228 != v226 )
                      {
                        v236 = v226 + 1;
                        if ( v228 != &v226[1] )
                        {
                          v237 = (__int64)v331;
                          do
                          {
                            if ( *(_WORD *)(sub_DEEF40(v237, *(_QWORD *)(v236->m128i_i64[0] - 32)) + 24) == 8
                              && *(_WORD *)(sub_DEEF40(v237, *(_QWORD *)(v236->m128i_i64[1] - 32)) + 24) == 8 )
                            {
                              ++v226;
                              v226[-1] = _mm_loadu_si128(v236);
                            }
                            ++v236;
                          }
                          while ( v228 != v236 );
                        }
                      }
LABEL_479:
                      v307 = (char *)dest[0];
                      v308 = (char *)((char *)dest[0] + 16 * LODWORD(dest[1]) - (char *)v228);
                      if ( v228 != (__m128i *)((char *)dest[0] + 16 * LODWORD(dest[1])) )
                      {
                        memmove(v226, v228, (char *)dest[0] + 16 * LODWORD(dest[1]) - (char *)v228);
                        v307 = (char *)dest[0];
                      }
                      LODWORD(dest[1]) = (&v308[(_QWORD)v226] - v307) >> 4;
                      sub_2808D60((__int64)v332);
LABEL_404:
                      v246 = sub_AA4E30(**(_QWORD **)(v365 + 32));
                      v402 = &v406;
                      v414 = &v416;
                      v387 = v385;
                      v388 = v246;
                      v389 = (__int64)"storeforward";
                      LOBYTE(v390) = 1;
                      memset(v391, 0, sizeof(v391));
                      v392 = 0;
                      v393 = 0;
                      v394 = 0;
                      v395 = 0;
                      v396 = 0;
                      v397 = 0;
                      v398 = 0;
                      v399 = 0;
                      v400 = 0;
                      v401 = 0;
                      v403 = 16;
                      v404 = 0;
                      v405 = 1;
                      v410 = 0;
                      v411 = 0;
                      v412 = 0;
                      v413 = 0;
                      v415 = 0x200000000LL;
                      v417 = 0;
                      v418 = 0;
                      v419 = 0;
                      v420 = 0;
                      v421 = 0;
                      v247 = _mm_loadu_si128(&v338);
                      v422 = &v426;
                      v433 = 1;
                      v356 = v246;
                      v334.m128i_i64[0] = (__int64)v332;
                      v339 = (__int64 (__fastcall *)(char *, __m128i *, int))sub_27BFDD0;
                      v248 = _mm_load_si128(&v334);
                      v423 = 2;
                      v336 = v340;
                      v334 = v247;
                      v340 = sub_27BFD20;
                      v338 = v248;
                      v424 = 0;
                      v425 = 1;
                      v427 = 0;
                      v428 = 0;
                      v429 = 0;
                      v430 = 0;
                      v431 = 0;
                      v432 = 0;
                      v434 = 0;
                      v337 = &unk_49DA0D8;
                      v335 = 0;
                      v354 = (const char *)&unk_49E5698;
                      v355 = (unsigned __int64)&unk_49D94D0;
                      v357 = (__m128i)(unsigned __int64)v246;
                      v358 = 0u;
                      v359 = 0u;
                      v360 = 0u;
                      LOWORD(v361) = 257;
                      v249 = sub_B2BE50(*v385);
                      v250 = _mm_loadu_si128(&v357);
                      v251 = _mm_loadu_si128(&v358);
                      v441 = v249;
                      v252 = _mm_loadu_si128(&v359);
                      v442 = v450;
                      v253 = _mm_loadu_si128(&v360);
                      v443 = &v456;
                      v435 = &v437;
                      v450[2] = v356;
                      v436 = 0x200000000LL;
                      v455 = v361;
                      v444 = 0;
                      v445 = 0;
                      v446 = 512;
                      v447 = 7;
                      v448 = 0;
                      v449 = 0;
                      v438 = 0;
                      v439 = 0;
                      v440 = 0;
                      v450[0] = &unk_49E5698;
                      v450[1] = &unk_49D94D0;
                      v456 = &unk_49DA0D8;
                      v458 = 0;
                      v451 = v250;
                      v452 = v251;
                      v453 = v252;
                      v454 = v253;
                      if ( v339 )
                      {
                        v339(v457, &v338, 2);
                        v459 = v340;
                        v458 = v339;
                      }
                      v354 = (const char *)&unk_49E5698;
                      v355 = (unsigned __int64)&unk_49D94D0;
                      nullsub_63();
                      nullsub_63();
                      sub_B32BF0(&v337);
                      if ( v335 )
                        v335(&v334, &v334, 3);
                      v460 = &v462;
                      v461 = 0x800000000LL;
                      v254 = (__int64 *)dest[0];
                      v463 = byte_3F871B3;
                      src = (char *)dest[0] + 16 * LODWORD(dest[1]);
                      if ( dest[0] == src )
                      {
LABEL_465:
                        sub_27C20B0((__int64)v332);
                        LOBYTE(src) = v328.m128i_i8[0];
                        goto LABEL_241;
                      }
                      while ( 2 )
                      {
                        v325 = *(__m128i **)(*v254 - 32);
                        v255 = sub_DEEF40((__int64)v331, (__int64)v325);
                        v256 = sub_D4B130(v365);
                        v257 = v256 + 48;
                        v258 = v256;
                        v259 = *(_QWORD *)(v256 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                        if ( v257 == v259 )
                        {
                          v261 = 0;
                        }
                        else
                        {
                          if ( !v259 )
                            goto LABEL_505;
                          v260 = *(unsigned __int8 *)(v259 - 24);
                          v261 = v259 - 24;
                          if ( (unsigned int)(v260 - 30) >= 0xB )
                            v261 = 0;
                        }
                        v262 = v261 + 24;
                        v263 = v312;
                        LOWORD(v263) = 0;
                        v312 = v263;
                        v264 = sub_F8DB90((__int64)v332, **(_QWORD **)(v255 + 32), v325->m128i_i64[1], v262, 0);
                        v265 = *v254;
                        v266 = *(_QWORD *)(*v254 + 8);
                        v354 = "load_initial";
                        v357.m128i_i16[4] = 259;
                        _BitScanReverse64((unsigned __int64 *)&v265, 1LL << (*(_WORD *)(v265 + 2) >> 1));
                        v267 = 63 - (v265 ^ 0x3F);
                        v268 = *(_QWORD *)(v258 + 48);
                        LODWORD(v325) = v267;
                        v269 = v268 & 0xFFFFFFFFFFFFFFF8LL;
                        if ( v269 == v257 )
                        {
                          v271 = 0;
                        }
                        else
                        {
                          if ( !v269 )
LABEL_505:
                            BUG();
                          v270 = *(unsigned __int8 *)(v269 - 24);
                          v271 = v269 - 24;
                          if ( (unsigned int)(v270 - 30) >= 0xB )
                            v271 = 0;
                        }
                        v319 = v266;
                        v323 = (__m128i)(v271 + 24);
                        v272 = sub_BD2C40(80, unk_3F10A14);
                        v273 = v272;
                        if ( v272 )
                          sub_B4D190(
                            (__int64)v272,
                            v319,
                            (__int64)v264,
                            (__int64)v9,
                            0,
                            (char)v325,
                            v323.m128i_i64[0],
                            v323.m128i_i64[1]);
                        v354 = "store_forwarded";
                        v357.m128i_i16[4] = 259;
                        v325 = (__m128i *)v273[1];
                        v274 = (const __m128i *)sub_BD2DA0(80);
                        v275 = v325;
                        v276 = (__int64)v274;
                        if ( v274 )
                        {
                          v325 = (__m128i *)v274;
                          sub_B44260((__int64)v274, (__int64)v275, 55, 0x8000000u, 0, 0);
                          *(_DWORD *)(v276 + 72) = 2;
                          sub_BD6B50((unsigned __int8 *)v276, v9);
                          sub_BD2A10(v276, *(_DWORD *)(v276 + 72), 1);
                          v277 = v325;
                        }
                        else
                        {
                          v277 = 0;
                        }
                        v278 = v313;
                        LOWORD(v278) = 1;
                        v313 = v278;
                        sub_B44220(v277, *(_QWORD *)(**(_QWORD **)(v365 + 32) + 56LL), v278);
                        v279 = *(_DWORD *)(v276 + 4) & 0x7FFFFFF;
                        if ( v279 == *(_DWORD *)(v276 + 72) )
                        {
                          sub_B48D90(v276);
                          v279 = *(_DWORD *)(v276 + 4) & 0x7FFFFFF;
                        }
                        v280 = (v279 + 1) & 0x7FFFFFF;
                        v281 = v280 | *(_DWORD *)(v276 + 4) & 0xF8000000;
                        v282 = *(_QWORD *)(v276 - 8) + 32LL * (unsigned int)(v280 - 1);
                        *(_DWORD *)(v276 + 4) = v281;
                        if ( *(_QWORD *)v282 )
                        {
                          v283 = *(_QWORD *)(v282 + 8);
                          **(_QWORD **)(v282 + 16) = v283;
                          if ( v283 )
                            *(_QWORD *)(v283 + 16) = *(_QWORD *)(v282 + 16);
                        }
                        *(_QWORD *)v282 = v273;
                        v284 = v273[2];
                        *(_QWORD *)(v282 + 8) = v284;
                        if ( v284 )
                          *(_QWORD *)(v284 + 16) = v282 + 8;
                        *(_QWORD *)(v282 + 16) = v273 + 2;
                        v273[2] = v282;
                        *(_QWORD *)(*(_QWORD *)(v276 - 8)
                                  + 32LL * *(unsigned int *)(v276 + 72)
                                  + 8LL * ((*(_DWORD *)(v276 + 4) & 0x7FFFFFFu) - 1)) = v258;
                        v285 = *v254;
                        v286 = *(_QWORD *)(v254[1] - 64);
                        v325 = (__m128i *)v273[1];
                        v287 = *(const __m128i **)(v286 + 8);
                        sub_B43CC0(v285);
                        v288 = v254[1];
                        v289 = *(_QWORD *)(v288 - 64);
                        if ( v325 != v287 )
                        {
                          v290 = *(_QWORD *)(v288 - 64);
                          v354 = "store_forward_cast";
                          v357.m128i_i16[4] = 259;
                          v289 = sub_B52260(v290, (__int64)v325, (__int64)v9, v288 + 24, 0);
                          v291 = *(const char **)(*v254 + 48);
                          v354 = v291;
                          if ( !v291 )
                          {
                            v292 = v289 + 48;
                            if ( (const char **)(v289 + 48) == v9 )
                              goto LABEL_438;
                            v304 = *(_QWORD *)(v289 + 48);
                            if ( !v304 )
                              goto LABEL_438;
                            goto LABEL_456;
                          }
                          v292 = v289 + 48;
                          sub_B96E90((__int64)v9, (__int64)v291, 1);
                          if ( (const char **)(v289 + 48) == v9 )
                          {
                            if ( v354 )
                              sub_B91220((__int64)v9, (__int64)v354);
                            goto LABEL_438;
                          }
                          v304 = *(_QWORD *)(v289 + 48);
                          if ( v304 )
LABEL_456:
                            sub_B91220(v292, v304);
                          v305 = (unsigned __int8 *)v354;
                          *(_QWORD *)(v289 + 48) = v354;
                          if ( v305 )
                            sub_B976B0((__int64)v9, v305, v292);
                        }
LABEL_438:
                        v293 = sub_D47930(v365);
                        v294 = *(_DWORD *)(v276 + 4) & 0x7FFFFFF;
                        if ( v294 == *(_DWORD *)(v276 + 72) )
                        {
                          sub_B48D90(v276);
                          v294 = *(_DWORD *)(v276 + 4) & 0x7FFFFFF;
                        }
                        v295 = (v294 + 1) & 0x7FFFFFF;
                        v296 = v295 | *(_DWORD *)(v276 + 4) & 0xF8000000;
                        v297 = *(_QWORD *)(v276 - 8) + 32LL * (unsigned int)(v295 - 1);
                        *(_DWORD *)(v276 + 4) = v296;
                        if ( *(_QWORD *)v297 )
                        {
                          v298 = *(_QWORD *)(v297 + 8);
                          **(_QWORD **)(v297 + 16) = v298;
                          if ( v298 )
                            *(_QWORD *)(v298 + 16) = *(_QWORD *)(v297 + 16);
                        }
                        *(_QWORD *)v297 = v289;
                        if ( v289 )
                        {
                          v299 = *(_QWORD *)(v289 + 16);
                          *(_QWORD *)(v297 + 8) = v299;
                          if ( v299 )
                            *(_QWORD *)(v299 + 16) = v297 + 8;
                          *(_QWORD *)(v297 + 16) = v289 + 16;
                          *(_QWORD *)(v289 + 16) = v297;
                        }
                        *(_QWORD *)(*(_QWORD *)(v276 - 8)
                                  + 32LL * *(unsigned int *)(v276 + 72)
                                  + 8LL * ((*(_DWORD *)(v276 + 4) & 0x7FFFFFFu) - 1)) = v293;
                        v300 = (const char **)(v276 + 48);
                        sub_BD84D0(*v254, v276);
                        v301 = *(const char **)(*v254 + 48);
                        v354 = v301;
                        if ( v301 )
                        {
                          sub_B96E90((__int64)v9, (__int64)v301, 1);
                          if ( v300 != v9 )
                          {
                            v302 = *(_QWORD *)(v276 + 48);
                            if ( v302 )
                              goto LABEL_452;
                            goto LABEL_453;
                          }
                          if ( v354 )
                            sub_B91220((__int64)v9, (__int64)v354);
                        }
                        else
                        {
                          if ( v300 == v9 )
                            goto LABEL_413;
                          v302 = *(_QWORD *)(v276 + 48);
                          if ( !v302 )
                            goto LABEL_413;
LABEL_452:
                          sub_B91220(v276 + 48, v302);
LABEL_453:
                          v303 = (unsigned __int8 *)v354;
                          *(_QWORD *)(v276 + 48) = v354;
                          if ( v303 )
                            sub_B976B0((__int64)v9, v303, v276 + 48);
                        }
LABEL_413:
                        v254 += 2;
                        if ( src == v254 )
                          goto LABEL_465;
                        continue;
                      }
                    }
LABEL_475:
                    v306 = (char *)v228 - (char *)v226;
                    if ( (char *)v228 - (char *)v226 != 32 )
                    {
                      if ( v306 != 48 )
                      {
                        if ( v306 != 16 )
                        {
LABEL_478:
                          v226 = v228;
                          goto LABEL_479;
                        }
LABEL_491:
                        if ( *(_WORD *)(sub_DEEF40((__int64)v331, *(_QWORD *)(v226->m128i_i64[0] - 32)) + 24) != 8
                          || *(_WORD *)(sub_DEEF40((__int64)v331, *(_QWORD *)(v226->m128i_i64[1] - 32)) + 24) != 8 )
                        {
                          goto LABEL_373;
                        }
                        goto LABEL_478;
                      }
                      if ( *(_WORD *)(sub_DEEF40((__int64)v331, *(_QWORD *)(v226->m128i_i64[0] - 32)) + 24) != 8
                        || *(_WORD *)(sub_DEEF40((__int64)v331, *(_QWORD *)(v226->m128i_i64[1] - 32)) + 24) != 8 )
                      {
                        goto LABEL_373;
                      }
                      ++v226;
                    }
                    if ( *(_WORD *)(sub_DEEF40((__int64)v331, *(_QWORD *)(v226->m128i_i64[0] - 32)) + 24) != 8
                      || *(_WORD *)(sub_DEEF40((__int64)v331, *(_QWORD *)(v226->m128i_i64[1] - 32)) + 24) != 8 )
                    {
                      goto LABEL_373;
                    }
                    ++v226;
                    goto LABEL_491;
                  }
                }
              }
            }
LABEL_241:
            if ( (__int64 *)v345 != &v347 )
              _libc_free(v345);
LABEL_243:
            if ( dest[0] != &dest[2] )
              _libc_free((unsigned __int64)dest[0]);
            v194 = (__int64 *)v333;
            if ( v333 )
            {
              do
              {
                v195 = (unsigned __int64)v194;
                v194 = (__int64 *)*v194;
                j_j___libc_free_0(v195);
              }
              while ( v194 );
            }
LABEL_247:
            if ( (_BYTE)src )
              sub_D37540(a7);
            v196 = v386;
            if ( v386 )
            {
              v197 = *(_QWORD *)(v386 + 40);
              if ( v197 != v386 + 56 )
                _libc_free(v197);
              j_j___libc_free_0(v196);
            }
            if ( v384 )
            {
              v207 = v383;
              v384 = 0;
              if ( v383 )
              {
                v208 = v382;
                v209 = &v382[2 * v383];
                do
                {
                  if ( *v208 != -4096 && *v208 != -8192 )
                  {
                    v210 = v208[1];
                    if ( v210 )
                      sub_B91220((__int64)(v208 + 1), v210);
                  }
                  v208 += 2;
                }
                while ( v209 != v208 );
                v207 = v383;
              }
              sub_C7D6A0((__int64)v382, 16 * v207, 8);
            }
            v198 = v381;
            if ( v381 )
            {
              v199 = v380;
              v355 = 2;
              v356 = 0;
              v200 = &v380[6 * v381];
              v357 = (__m128i)0xFFFFFFFFFFFFF000LL;
              v354 = (const char *)&unk_49DDFA0;
              v388 = 2;
              v389 = 0;
              v390 = -8192;
              v387 = (__int64 *)&unk_49DDFA0;
              v391[0] = 0;
              do
              {
                v201 = v199[3];
                *v199 = &unk_49DB368;
                if ( v201 != -4096 && v201 != 0 && v201 != -8192 )
                  sub_BD60C0(v199 + 1);
                v199 += 6;
              }
              while ( v200 != v199 );
              v387 = (__int64 *)&unk_49DB368;
              if ( v390 != -4096 && v390 != 0 && v390 != -8192 )
                sub_BD60C0(&v388);
              v354 = (const char *)&unk_49DB368;
              if ( v357.m128i_i64[0] != 0 && v357.m128i_i64[0] != -4096 && v357.m128i_i64[0] != -8192 )
                sub_BD60C0(&v355);
              v198 = v381;
            }
            sub_C7D6A0((__int64)v380, 48 * v198, 8);
            sub_C7D6A0(v376, 24LL * (unsigned int)v378, 8);
            sub_C7D6A0(v367, 16LL * v369, 8);
LABEL_91:
            if ( v322 == ++v324 )
            {
              v322 = v341;
              goto LABEL_93;
            }
            goto LABEL_79;
          }
          goto LABEL_155;
        }
        while ( 1 )
        {
          v120 = *(_QWORD *)(v109 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v120 - 30) > 0xAu )
            break;
          v115 = *(_QWORD *)(v120 + 40);
          if ( *(_BYTE *)(v114 + 84) )
            goto LABEL_158;
LABEL_168:
          v323.m128i_i64[0] = v111;
          v121 = sub_C8CA60(v113, v115);
          v111 = v323.m128i_i64[0];
          if ( v121 )
            goto LABEL_162;
          v109 = *(_QWORD *)(v109 + 8);
          if ( !v109 )
            goto LABEL_170;
        }
      }
    }
    while ( 1 )
    {
      v82 = *(_QWORD *)(v72 + 56);
      v75 = *(_QWORD *)(v82 + 8LL * *v73);
      v83 = v73[1];
      v76 = *(_QWORD *)(v82 + 8 * v83);
      if ( v73[2] - 1 > 1 )
      {
        if ( (unsigned __int8)sub_D354F0((__int64)v73) )
        {
          v74 = v75;
          v75 = v76;
          v76 = v74;
        }
        if ( *(_BYTE *)v75 == 62 && *(_BYTE *)v76 == 61 )
        {
          v77 = sub_B43CC0(v75);
          v78 = *(_BYTE *)v76 == 61 ? *(_QWORD *)(v76 + 8) : *(_QWORD *)(*(_QWORD *)(v76 - 64) + 8LL);
          v79 = *(_BYTE *)v75 == 61 ? *(_QWORD *)(v75 + 8) : *(_QWORD *)(*(_QWORD *)(v75 - 64) + 8LL);
          if ( (unsigned __int8)sub_B50C50(v79, v78, v77) )
          {
            v80 = (__m128i *)sub_22077B0(0x18u);
            v81 = v333;
            v80->m128i_i64[1] = v76;
            v80[1].m128i_i64[0] = v75;
            v80->m128i_i64[0] = (__int64)v81;
            v333 = v80;
          }
        }
        goto LABEL_118;
      }
      if ( *(_BYTE *)v75 == 61 )
      {
        if ( !BYTE4(v390) )
          goto LABEL_311;
        v211 = (_QWORD *)v388;
        v83 = HIDWORD(v389);
        v69 = (__int64 *)(v388 + 8LL * HIDWORD(v389));
        if ( (__int64 *)v388 == v69 )
        {
LABEL_304:
          if ( HIDWORD(v389) < (unsigned int)v389 )
          {
            v83 = (unsigned int)++HIDWORD(v389);
            *v69 = v75;
            v387 = (__int64 *)((char *)v387 + 1);
            goto LABEL_121;
          }
LABEL_311:
          sub_C8CC70((__int64)v332, v75, v83, (__int64)v69, v70, v71);
          goto LABEL_121;
        }
        while ( v75 != *v211 )
        {
          if ( v69 == ++v211 )
            goto LABEL_304;
        }
      }
LABEL_121:
      if ( *(_BYTE *)v76 != 61 )
        goto LABEL_118;
      if ( !BYTE4(v390) )
      {
LABEL_299:
        sub_C8CC70((__int64)v332, v76, v83, (__int64)v69, v70, v71);
        goto LABEL_118;
      }
      v84 = (_QWORD *)v388;
      v83 = HIDWORD(v389);
      v69 = (__int64 *)(v388 + 8LL * HIDWORD(v389));
      if ( (__int64 *)v388 != v69 )
      {
        while ( v76 != *v84 )
        {
          if ( v69 == ++v84 )
            goto LABEL_126;
        }
LABEL_118:
        v73 += 3;
        if ( (unsigned int *)v328.m128i_i64[0] == v73 )
          goto LABEL_128;
        continue;
      }
LABEL_126:
      if ( HIDWORD(v389) >= (unsigned int)v389 )
        goto LABEL_299;
      v73 += 3;
      ++HIDWORD(v389);
      *v69 = v76;
      v387 = (__int64 *)((char *)v387 + 1);
      if ( (unsigned int *)v328.m128i_i64[0] == v73 )
      {
LABEL_128:
        v85 = BYTE4(v390);
        if ( HIDWORD(v389) == (_DWORD)v390 || (v86 = v333) == 0 )
        {
LABEL_138:
          if ( !v85 )
            _libc_free(v388);
          goto LABEL_140;
        }
        v87 = &v333;
        while ( 2 )
        {
          v88 = v86->m128i_i64[1];
          if ( v85 )
          {
            v89 = (_QWORD *)v388;
            v90 = (_QWORD *)(v388 + 8LL * HIDWORD(v389));
            if ( (_QWORD *)v388 == v90 )
            {
LABEL_308:
              v87 = (const __m128i **)*v87;
LABEL_137:
              v86 = *v87;
              if ( !*v87 )
                goto LABEL_138;
              continue;
            }
            while ( v88 != *v89 )
            {
              if ( v90 == ++v89 )
                goto LABEL_308;
            }
          }
          else if ( !sub_C8CA60((__int64)v332, v88) )
          {
            v85 = BYTE4(v390);
            goto LABEL_308;
          }
          break;
        }
        v91 = (unsigned __int64)*v87;
        *v87 = (const __m128i *)(*v87)->m128i_i64[0];
        j_j___libc_free_0(v91);
        v85 = BYTE4(v390);
        goto LABEL_137;
      }
    }
  }
LABEL_93:
  if ( v322 != (__int64 *)v343 )
    _libc_free((unsigned __int64)v322);
  return (unsigned __int8)src;
}
