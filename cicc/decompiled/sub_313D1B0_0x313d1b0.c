// Function: sub_313D1B0
// Address: 0x313d1b0
//
__m128i *__fastcall sub_313D1B0(
        __m128i *a1,
        __int64 a2,
        __m128i *a3,
        void (__fastcall *a4)(__m128i *, __int64, char *, char *, __int64, __int64, char *, __int64, _QWORD, char *, unsigned __int64, __int64),
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        int a9,
        void (__fastcall *a10)(__m128i *, __int64, __int64, __int64, _QWORD **, __int64, __int64, __int64, const char *, __int64, __int64, __int64),
        __int64 a11,
        __int64 a12,
        unsigned __int64 a13,
        unsigned int a14,
        char a15)
{
  __int8 v17; // al
  __int64 *v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rsi
  __int64 v22; // rax
  __int64 *v23; // rsi
  __int64 v24; // r12
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r14
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // r12
  _QWORD *v32; // rax
  _QWORD *v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // r8
  __int64 v37; // r9
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rcx
  __m128i *v40; // r12
  __int64 v41; // rax
  unsigned __int64 v42; // rsi
  int v43; // edx
  __m128i *v44; // rax
  __m128i v45; // xmm2
  __int64 v46; // rcx
  __int64 v47; // rdx
  __m128i v48; // xmm0
  __int64 v49; // rdx
  void (__fastcall *v50)(char **, char **, __int64); // rax
  unsigned __int64 v51; // rax
  __int64 v52; // rax
  __int64 *v53; // rsi
  __int64 v54; // rax
  __int64 v55; // rdi
  __int64 v56; // r12
  __int64 v57; // r14
  __int64 v58; // rax
  char v59; // r15
  __int64 v60; // r15
  __int64 v61; // r12
  __int64 v62; // rdx
  unsigned int v63; // esi
  __int64 v64; // rdi
  __int64 v65; // r15
  __int64 v66; // rax
  char v67; // al
  char v68; // r14
  _QWORD *v69; // rax
  __int64 v70; // r12
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // r14
  __int64 v74; // r15
  __int64 v75; // rdx
  unsigned int v76; // esi
  __int64 v77; // rax
  unsigned __int64 v78; // rdx
  __int64 v79; // rdi
  __int64 v80; // r15
  __int64 v81; // rax
  char v82; // r14
  _QWORD *v83; // rax
  __int64 v84; // r12
  __int64 v85; // r8
  __int64 v86; // r9
  __int64 v87; // r14
  __int64 v88; // r15
  __int64 v89; // rdx
  unsigned int v90; // esi
  __int64 v91; // rax
  unsigned __int64 v92; // rdx
  char *v93; // rdx
  unsigned __int64 v94; // rax
  unsigned __int64 v95; // rax
  __int64 v96; // r8
  __int64 v97; // r9
  bool v98; // zf
  __int64 v99; // rcx
  _QWORD *v100; // rsi
  __int64 v101; // rdx
  __int64 v102; // rax
  _QWORD *v103; // rax
  __int64 v104; // r8
  __int64 v105; // r9
  __int64 v106; // rdx
  __m128i v107; // xmm0
  __m128i v108; // xmm3
  __int64 (__fastcall *v109)(__int64 *, unsigned __int64); // rax
  __int64 v110; // r8
  __int64 v111; // r9
  __int64 *v112; // rax
  __int64 v113; // rcx
  __int64 *v114; // r14
  __int64 v115; // rdx
  __int64 v116; // rcx
  __int64 *v117; // rcx
  __int64 v118; // rdx
  __int64 *v119; // rsi
  __int64 v120; // rdx
  __int64 v121; // rdx
  __int64 v122; // rdx
  unsigned int v123; // ecx
  __int64 *v124; // r8
  __int64 v125; // r9
  __int64 *j; // rdx
  __int64 v127; // rcx
  __int64 v128; // rsi
  __int64 v129; // rax
  __int64 v130; // rax
  __int16 v131; // dx
  __int64 v132; // r8
  __int64 v133; // r9
  __int64 *v134; // r15
  __int64 v135; // r13
  __int64 k; // r12
  __int64 v137; // rax
  __int64 v138; // rsi
  _QWORD *v139; // rax
  _QWORD *v140; // rdx
  __int64 v141; // r8
  int v142; // r10d
  unsigned int v143; // ecx
  __int64 *v144; // rax
  __int64 v145; // rdx
  __int64 v146; // r12
  __int64 v147; // rdx
  __int64 v148; // rdx
  __int64 v149; // r12
  unsigned __int64 v150; // rax
  unsigned __int64 v151; // rdx
  __int64 v152; // rax
  int v153; // eax
  __int64 v154; // rcx
  __int16 v155; // dx
  __int64 v156; // rax
  char v157; // dl
  __m128i *v158; // rsi
  __m128i *v159; // r12
  unsigned __int64 v160; // rax
  __int64 v161; // rax
  __int64 v162; // rbx
  __int64 v163; // r12
  __int64 v164; // rax
  __int64 v165; // rdi
  __int64 v166; // rax
  _BYTE *v167; // rax
  unsigned __int64 v168; // rsi
  __int64 **v169; // r14
  unsigned int v170; // r12d
  unsigned int v171; // eax
  _BYTE *v172; // rax
  unsigned __int64 v173; // rsi
  __int64 v174; // rax
  unsigned __int64 v175; // rdx
  __int64 v176; // rax
  __m128i v177; // xmm5
  __m128i *v178; // roff
  __m128i v179; // xmm0
  __int64 v180; // rdx
  __int64 v181; // rdx
  unsigned __int64 v182; // rax
  __int64 v183; // rdx
  unsigned __int64 v184; // rax
  unsigned int v185; // eax
  __int64 *v186; // r8
  __int64 v187; // r9
  unsigned int v188; // eax
  __int64 v189; // r9
  int v190; // r8d
  int v191; // r10d
  unsigned int v192; // eax
  __int64 v193; // r9
  int v194; // r8d
  int v195; // r10d
  unsigned int v196; // esi
  __int64 *v197; // r8
  __int64 v198; // r10
  _QWORD *v199; // rcx
  __int64 v200; // rdx
  __int64 v201; // rax
  _QWORD *v202; // rax
  __int64 v203; // r8
  __int64 v204; // r9
  int v205; // r10d
  __m128i v206; // xmm0
  __m128i v207; // xmm4
  unsigned __int64 v208; // rax
  int v209; // edx
  unsigned __int64 v210; // rax
  __m128i *v211; // rcx
  _QWORD *v212; // rax
  _QWORD *v213; // rdx
  __int64 v214; // rsi
  __int64 v215; // rsi
  __int64 v216; // rdx
  unsigned int v217; // ecx
  __int64 *v218; // rsi
  __int64 v219; // r10
  char *v220; // r12
  __int64 v221; // r8
  __int64 v222; // r9
  __int64 v223; // rcx
  __int64 v224; // r8
  __int64 v225; // r9
  __int64 v226; // rax
  int v227; // edx
  __int64 v228; // rax
  unsigned __int64 v229; // rdx
  __int64 v230; // rcx
  __int64 v231; // rdx
  int v232; // r8d
  int v233; // r10d
  int v234; // r8d
  int v235; // r10d
  __int64 v236; // rdx
  __int64 v237; // rcx
  __int64 v238; // r8
  __int64 v239; // r9
  __int64 v240; // rbx
  __int8 v241; // al
  unsigned int v242; // eax
  __int64 v243; // r11
  int v244; // esi
  __int64 v245; // rcx
  __int64 v246; // rsi
  unsigned int v247; // r14d
  __int64 v248; // rcx
  int v249; // eax
  int v250; // r8d
  int v251; // r11d
  unsigned int v252; // ecx
  __int64 v253; // r10
  int v254; // esi
  int v255; // r9d
  int v256; // edi
  unsigned int i; // edx
  _QWORD *v258; // r9
  int v259; // esi
  int v260; // r9d
  unsigned int v261; // edx
  __int64 v262; // [rsp+0h] [rbp-780h]
  unsigned __int16 v263; // [rsp+14h] [rbp-76Ch]
  unsigned __int8 v264; // [rsp+16h] [rbp-76Ah]
  _BYTE *v265; // [rsp+30h] [rbp-750h]
  _QWORD *v266; // [rsp+48h] [rbp-738h]
  __int16 v267; // [rsp+58h] [rbp-728h]
  char *v268; // [rsp+60h] [rbp-720h]
  char v269; // [rsp+60h] [rbp-720h]
  char *v270; // [rsp+68h] [rbp-718h]
  char v271; // [rsp+68h] [rbp-718h]
  __int64 v272; // [rsp+70h] [rbp-710h]
  __int64 v273; // [rsp+70h] [rbp-710h]
  _QWORD *v274; // [rsp+78h] [rbp-708h]
  __int64 v275; // [rsp+80h] [rbp-700h]
  unsigned __int64 v276; // [rsp+88h] [rbp-6F8h]
  __int64 *v277; // [rsp+88h] [rbp-6F8h]
  __int64 v278; // [rsp+90h] [rbp-6F0h]
  __int64 v279; // [rsp+98h] [rbp-6E8h]
  _QWORD *v280; // [rsp+A0h] [rbp-6E0h]
  __int64 v281; // [rsp+A0h] [rbp-6E0h]
  char *v282; // [rsp+A8h] [rbp-6D8h]
  _QWORD *v283; // [rsp+B8h] [rbp-6C8h]
  char v284; // [rsp+C0h] [rbp-6C0h]
  __int64 v285; // [rsp+C0h] [rbp-6C0h]
  _QWORD *v286; // [rsp+C8h] [rbp-6B8h]
  _QWORD *v287; // [rsp+D0h] [rbp-6B0h]
  __int64 v288; // [rsp+D0h] [rbp-6B0h]
  __int64 v290; // [rsp+E8h] [rbp-698h]
  _QWORD *v293; // [rsp+100h] [rbp-680h]
  _QWORD *v294; // [rsp+100h] [rbp-680h]
  __int64 *v296; // [rsp+118h] [rbp-668h]
  unsigned int v297; // [rsp+124h] [rbp-65Ch] BYREF
  __int64 v298; // [rsp+128h] [rbp-658h] BYREF
  __int64 v299; // [rsp+130h] [rbp-650h] BYREF
  _QWORD *v300; // [rsp+138h] [rbp-648h] BYREF
  __int64 v301[2]; // [rsp+140h] [rbp-640h] BYREF
  __int64 v302; // [rsp+150h] [rbp-630h]
  unsigned __int64 v303; // [rsp+160h] [rbp-620h] BYREF
  __int64 v304; // [rsp+168h] [rbp-618h]
  const char *v305; // [rsp+170h] [rbp-610h]
  __int16 v306; // [rsp+180h] [rbp-600h]
  char *v307; // [rsp+190h] [rbp-5F0h] BYREF
  __int64 v308; // [rsp+198h] [rbp-5E8h]
  _BYTE v309[32]; // [rsp+1A0h] [rbp-5E0h] BYREF
  __int64 v310; // [rsp+1C0h] [rbp-5C0h] BYREF
  __int64 v311; // [rsp+1C8h] [rbp-5B8h]
  __int64 v312; // [rsp+1D0h] [rbp-5B0h]
  __int64 v313; // [rsp+1D8h] [rbp-5A8h]
  __int64 *v314; // [rsp+1E0h] [rbp-5A0h]
  __int64 v315; // [rsp+1E8h] [rbp-598h]
  __int64 v316; // [rsp+1F0h] [rbp-590h] BYREF
  __int64 v317; // [rsp+1F8h] [rbp-588h]
  __int64 v318; // [rsp+200h] [rbp-580h]
  __int64 v319; // [rsp+208h] [rbp-578h]
  __int64 *v320; // [rsp+210h] [rbp-570h]
  __int64 v321; // [rsp+218h] [rbp-568h]
  __int64 v322; // [rsp+220h] [rbp-560h] BYREF
  __int64 v323; // [rsp+228h] [rbp-558h]
  __int64 v324; // [rsp+230h] [rbp-550h]
  __int64 v325; // [rsp+238h] [rbp-548h]
  __int64 *v326; // [rsp+240h] [rbp-540h]
  __int64 v327; // [rsp+248h] [rbp-538h]
  __int64 v328; // [rsp+250h] [rbp-530h] BYREF
  __int64 v329; // [rsp+258h] [rbp-528h]
  __int64 v330; // [rsp+260h] [rbp-520h]
  __int64 v331; // [rsp+268h] [rbp-518h]
  _QWORD *v332; // [rsp+270h] [rbp-510h]
  __int64 v333; // [rsp+278h] [rbp-508h]
  _QWORD *v334; // [rsp+280h] [rbp-500h] BYREF
  __int64 v335; // [rsp+288h] [rbp-4F8h]
  __int64 v336; // [rsp+290h] [rbp-4F0h]
  __int64 v337; // [rsp+298h] [rbp-4E8h]
  __m128i *v338; // [rsp+2A0h] [rbp-4E0h] BYREF
  __int64 v339; // [rsp+2A8h] [rbp-4D8h]
  __m128i v340; // [rsp+2B0h] [rbp-4D0h] BYREF
  __int64 v341; // [rsp+2C0h] [rbp-4C0h] BYREF
  __int64 v342; // [rsp+2C8h] [rbp-4B8h]
  __int64 v343; // [rsp+2D0h] [rbp-4B0h]
  __int16 v344; // [rsp+2D8h] [rbp-4A8h]
  __int64 v345; // [rsp+2E0h] [rbp-4A0h] BYREF
  __m128i v346; // [rsp+2F0h] [rbp-490h] BYREF
  __int64 (__fastcall *v347)(unsigned __int64 **, __int64 *, int); // [rsp+300h] [rbp-480h]
  __int64 (__fastcall *v348)(__int64 *, unsigned __int64); // [rsp+308h] [rbp-478h]
  _QWORD *v349; // [rsp+310h] [rbp-470h]
  __int64 v350; // [rsp+318h] [rbp-468h]
  __int64 v351; // [rsp+320h] [rbp-460h]
  _BYTE *v352; // [rsp+328h] [rbp-458h] BYREF
  __int64 v353; // [rsp+330h] [rbp-450h]
  _BYTE v354[24]; // [rsp+338h] [rbp-448h] BYREF
  unsigned __int64 v355[2]; // [rsp+350h] [rbp-430h] BYREF
  char v356; // [rsp+360h] [rbp-420h] BYREF
  __int64 v357; // [rsp+3E8h] [rbp-398h]
  unsigned int v358; // [rsp+3F8h] [rbp-388h]
  __int64 v359; // [rsp+408h] [rbp-378h]
  unsigned int v360; // [rsp+418h] [rbp-368h]
  __m128i v361; // [rsp+420h] [rbp-360h] BYREF
  __int64 (__fastcall *v362)(unsigned __int64 **, __int64 *, int); // [rsp+430h] [rbp-350h]
  __int64 (__fastcall *v363)(__int64 *, unsigned __int64); // [rsp+438h] [rbp-348h]
  __int64 v364; // [rsp+460h] [rbp-320h]
  unsigned int v365; // [rsp+470h] [rbp-310h]
  unsigned __int64 *v366; // [rsp+478h] [rbp-308h]
  char *v367; // [rsp+488h] [rbp-2F8h] BYREF
  char v368; // [rsp+498h] [rbp-2E8h] BYREF
  _QWORD *v369; // [rsp+4C8h] [rbp-2B8h]
  _QWORD v370[6]; // [rsp+4D8h] [rbp-2A8h] BYREF
  unsigned int v371; // [rsp+508h] [rbp-278h]
  char **v372; // [rsp+510h] [rbp-270h]
  char *v373; // [rsp+520h] [rbp-260h] BYREF
  __int64 v374; // [rsp+528h] [rbp-258h]
  _QWORD v375[2]; // [rsp+530h] [rbp-250h] BYREF
  char v376; // [rsp+540h] [rbp-240h]
  char v377; // [rsp+541h] [rbp-23Fh]
  __int64 v378; // [rsp+550h] [rbp-230h]
  __int64 v379; // [rsp+558h] [rbp-228h]
  __int64 v380; // [rsp+560h] [rbp-220h]
  __int64 v381; // [rsp+568h] [rbp-218h]
  __int64 v382; // [rsp+570h] [rbp-210h]
  unsigned __int64 v383; // [rsp+578h] [rbp-208h]
  _QWORD *v384; // [rsp+580h] [rbp-200h]
  unsigned __int64 v385; // [rsp+588h] [rbp-1F8h]
  __int64 v386; // [rsp+590h] [rbp-1F0h]
  char *v387; // [rsp+630h] [rbp-150h] BYREF
  unsigned __int64 v388; // [rsp+638h] [rbp-148h]
  __int64 v389; // [rsp+640h] [rbp-140h] BYREF
  __int64 *(__fastcall *v390)(__int64 *, __int64 **, __int64); // [rsp+648h] [rbp-138h]
  int v391; // [rsp+650h] [rbp-130h] BYREF
  char v392; // [rsp+654h] [rbp-12Ch]
  __int64 v393; // [rsp+660h] [rbp-120h]
  __int64 v394; // [rsp+668h] [rbp-118h]
  __int64 v395; // [rsp+670h] [rbp-110h]
  __int64 v396; // [rsp+678h] [rbp-108h]
  __int64 v397; // [rsp+680h] [rbp-100h]
  unsigned __int64 v398; // [rsp+688h] [rbp-F8h]
  _QWORD *v399; // [rsp+690h] [rbp-F0h]
  unsigned __int64 v400; // [rsp+698h] [rbp-E8h]
  __int64 v401; // [rsp+6A0h] [rbp-E0h]

  if ( !sub_31387E0(a2, (__int64)a3) )
  {
    v17 = a1[1].m128i_i8[8] & 0xFC | 2;
    *a1 = _mm_loadu_si128(a3);
    a1[1].m128i_i8[8] = v17;
    a1[1].m128i_i64[0] = a3[1].m128i_i64[0];
    return a1;
  }
  v19 = (__int64 *)sub_31376D0(a2, a3->m128i_i64, &v297);
  v282 = (char *)sub_313A9F0(a2, v19, v297, 0, 0);
  v278 = sub_3135D90(a2, (__int64)v282);
  v284 = *(_BYTE *)(a2 + 336);
  v296 = (__int64 *)(a2 + 512);
  if ( a13 && !v284 )
  {
    v169 = *(__int64 ***)(a2 + 2632);
    LOWORD(v391) = 257;
    v373 = v282;
    v374 = v278;
    v170 = sub_BCB060(*(_QWORD *)(a13 + 8));
    v171 = sub_BCB060((__int64)v169);
    v375[0] = sub_31223E0(v296, (unsigned int)(v170 <= v171) + 38, a13, v169, (__int64)&v387, 0, v361.m128i_i32[0], 0);
    LOWORD(v391) = 257;
    v172 = sub_3135910(a2, 11);
    v173 = 0;
    if ( v172 )
      v173 = *((_QWORD *)v172 + 3);
    sub_921880((unsigned int **)v296, v173, (int)v172, (int)&v373, 3, (__int64)&v387, 0);
  }
  if ( a14 != 6 )
  {
    v165 = *(_QWORD *)(a2 + 2632);
    v373 = v282;
    v374 = v278;
    v166 = sub_AD64C0(v165, a14, 1u);
    LOWORD(v391) = 257;
    v375[0] = v166;
    v167 = sub_3135910(a2, 12);
    v168 = 0;
    if ( v167 )
      v168 = *((_QWORD *)v167 + 3);
    sub_921880((unsigned int **)v296, v168, (int)v167, (int)&v373, 3, (__int64)&v387, 0);
  }
  v279 = *(_QWORD *)(a2 + 560);
  v20 = *(_QWORD *)(v279 + 72);
  v307 = v309;
  v275 = v20;
  v308 = 0x400000000LL;
  sub_3139A00((__int64)v296, a7, *(_QWORD *)(a7 + 56), 1, 0);
  v21 = *(__int64 **)(a2 + 2632);
  v387 = "tid.addr";
  LOWORD(v391) = 259;
  v22 = sub_23DEB90(v296, v21, 0, (__int64)&v387);
  v23 = *(__int64 **)(a2 + 2632);
  v24 = v22;
  v387 = "zero.addr";
  LOWORD(v391) = 259;
  v27 = sub_23DEB90(v296, v23, 0, (__int64)&v387);
  if ( v284 && (v164 = *(_QWORD *)(a2 + 504), *(_DWORD *)(v164 + 316)) )
  {
    v288 = sub_BCE3C0(*(__int64 **)v164, 0);
    v387 = "tid.addr.ascast";
    LOWORD(v391) = 259;
    v286 = sub_BD2C40(72, 1u);
    if ( v286 )
      sub_B51C90((__int64)v286, v24, v288, (__int64)&v387, 0, 0);
    sub_B43E90((__int64)v286, v24 + 24);
    sub_9C95B0((__int64)&v307, (__int64)v286);
    v281 = sub_BCE3C0(**(__int64 ***)(a2 + 504), 0);
    v387 = "zero.addr.ascast";
    LOWORD(v391) = 259;
    v283 = sub_BD2C40(72, 1u);
    if ( v283 )
      sub_B51C90((__int64)v283, v27, v281, (__int64)&v387, 0, 0);
    sub_B43E90((__int64)v283, v27 + 24);
    sub_9C95B0((__int64)&v307, (__int64)v283);
  }
  else
  {
    v283 = (_QWORD *)v27;
    v286 = (_QWORD *)v24;
  }
  v28 = (unsigned int)v308;
  v29 = (unsigned int)v308 + 1LL;
  if ( v29 > HIDWORD(v308) )
  {
    sub_C8D5F0((__int64)&v307, v309, v29, 8u, v25, v26);
    v28 = (unsigned int)v308;
  }
  *(_QWORD *)&v307[8 * v28] = v24;
  LODWORD(v308) = v308 + 1;
  v30 = (unsigned int)v308;
  if ( (unsigned __int64)(unsigned int)v308 + 1 > HIDWORD(v308) )
  {
    sub_C8D5F0((__int64)&v307, v309, (unsigned int)v308 + 1LL, 8u, v25, v26);
    v30 = (unsigned int)v308;
  }
  *(_QWORD *)&v307[8 * v30] = v27;
  v31 = *(_QWORD *)(a2 + 584);
  LODWORD(v308) = v308 + 1;
  sub_B43C20((__int64)&v387, v279);
  v32 = sub_BD2C40(72, unk_3F148B8);
  v274 = v32;
  if ( v32 )
    sub_B4C8A0((__int64)v32, v31, (__int64)v387, v388);
  v33 = (_QWORD *)v274[5];
  v387 = "omp.par.entry";
  LOWORD(v391) = 259;
  v280 = (_QWORD *)sub_AA8550(v33, v274 + 3, 0, (__int64)&v387, 0);
  v387 = "omp.par.region";
  LOWORD(v391) = 259;
  v268 = (char *)sub_AA8550(v280, v274 + 3, 0, (__int64)&v387, 0);
  v387 = "omp.par.pre_finalize";
  LOWORD(v391) = 259;
  v266 = (_QWORD *)sub_AA8550(v268, v274 + 3, 0, (__int64)&v387, 0);
  v387 = "omp.par.exit";
  LOWORD(v391) = 259;
  v34 = sub_AA8550(v266, v274 + 3, 0, (__int64)&v387, 0);
  v389 = 0;
  v298 = v34;
  v35 = sub_22077B0(0x18u);
  if ( v35 )
  {
    *(_QWORD *)v35 = a2;
    *(_QWORD *)(v35 + 8) = &v298;
    *(_QWORD *)(v35 + 16) = a6;
  }
  v38 = *(unsigned int *)(a2 + 12);
  v39 = *(_QWORD *)a2;
  v387 = (char *)v35;
  v40 = (__m128i *)&v387;
  v392 = a15;
  v390 = sub_31397B0;
  v389 = (__int64)sub_3120E50;
  v41 = *(unsigned int *)(a2 + 8);
  v391 = 48;
  v42 = v41 + 1;
  v43 = v41;
  if ( v41 + 1 > v38 )
  {
    if ( v39 > (unsigned __int64)&v387 || (unsigned __int64)&v387 >= v39 + 40 * v41 )
    {
      sub_313A0B0(a2, v42, v41, v39, v36, v37);
      v41 = *(unsigned int *)(a2 + 8);
      v39 = *(_QWORD *)a2;
      v43 = *(_DWORD *)(a2 + 8);
    }
    else
    {
      v220 = (char *)&v387 - v39;
      sub_313A0B0(a2, v42, v41, v39, v36, v37);
      v39 = *(_QWORD *)a2;
      v41 = *(unsigned int *)(a2 + 8);
      v40 = (__m128i *)&v220[*(_QWORD *)a2];
      v43 = *(_DWORD *)(a2 + 8);
    }
  }
  v44 = (__m128i *)(v39 + 40 * v41);
  if ( v44 )
  {
    v45 = _mm_loadu_si128(v44);
    v46 = v44[1].m128i_i64[1];
    v44[1].m128i_i64[0] = 0;
    v47 = v40[1].m128i_i64[0];
    v48 = _mm_loadu_si128(v40);
    v40[1].m128i_i64[0] = 0;
    *v40 = v45;
    v44[1].m128i_i64[0] = v47;
    v49 = v40[1].m128i_i64[1];
    *v44 = v48;
    v44[1].m128i_i64[1] = v49;
    LODWORD(v49) = v40[2].m128i_i32[0];
    v40[1].m128i_i64[1] = v46;
    v44[2].m128i_i32[0] = v49;
    v44[2].m128i_i8[4] = v40[2].m128i_i8[4];
    v43 = *(_DWORD *)(a2 + 8);
  }
  v50 = (void (__fastcall *)(char **, char **, __int64))v389;
  *(_DWORD *)(a2 + 8) = v43 + 1;
  if ( v50 )
    v50(&v387, &v387, 3);
  v51 = sub_986580((__int64)v280);
  sub_D5F1F0((__int64)v296, v51);
  v52 = *(_QWORD *)(a2 + 568);
  v53 = *(__int64 **)(a2 + 2632);
  LOWORD(v391) = 259;
  v272 = v52;
  v267 = *(_WORD *)(a2 + 576);
  v270 = *(char **)(a2 + 560);
  v387 = "tid.addr.local";
  v54 = sub_23DEB90(v296, v53, 0, (__int64)&v387);
  v55 = *(_QWORD *)(a2 + 560);
  v56 = *(_QWORD *)(a2 + 2632);
  v377 = 1;
  v276 = v54;
  v57 = v54;
  v373 = "tid";
  v376 = 3;
  v58 = sub_AA4E30(v55);
  v59 = sub_AE5020(v58, v56);
  LOWORD(v391) = 257;
  v287 = sub_BD2C40(80, 1u);
  if ( v287 )
    sub_B4D190((__int64)v287, v56, v57, (__int64)&v387, 0, v59, 0, 0);
  (*(void (__fastcall **)(_QWORD, _QWORD *, char **, __int64, __int64))(**(_QWORD **)(a2 + 600) + 16LL))(
    *(_QWORD *)(a2 + 600),
    v287,
    &v373,
    v296[7],
    v296[8]);
  v60 = *(_QWORD *)(a2 + 512);
  v61 = v60 + 16LL * *(unsigned int *)(a2 + 520);
  while ( v61 != v60 )
  {
    v62 = *(_QWORD *)(v60 + 8);
    v63 = *(_DWORD *)v60;
    v60 += 16;
    sub_B99FD0((__int64)v287, v63, v62);
  }
  v377 = 1;
  v64 = *(_QWORD *)(a2 + 560);
  v65 = *(_QWORD *)(a2 + 2632);
  v376 = 3;
  v373 = "tid.addr.use";
  v66 = sub_AA4E30(v64);
  v67 = sub_AE5020(v66, v65);
  LOWORD(v391) = 257;
  v68 = v67;
  v69 = sub_BD2C40(80, 1u);
  v70 = (__int64)v69;
  if ( v69 )
    sub_B4D190((__int64)v69, v65, (__int64)v286, (__int64)&v387, 0, v68, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, char **, __int64, __int64))(**(_QWORD **)(a2 + 600) + 16LL))(
    *(_QWORD *)(a2 + 600),
    v70,
    &v373,
    v296[7],
    v296[8]);
  v73 = *(_QWORD *)(a2 + 512);
  v74 = v73 + 16LL * *(unsigned int *)(a2 + 520);
  while ( v74 != v73 )
  {
    v75 = *(_QWORD *)(v73 + 8);
    v76 = *(_DWORD *)v73;
    v73 += 16;
    sub_B99FD0(v70, v76, v75);
  }
  v77 = (unsigned int)v308;
  v78 = (unsigned int)v308 + 1LL;
  if ( v78 > HIDWORD(v308) )
  {
    sub_C8D5F0((__int64)&v307, v309, v78, 8u, v71, v72);
    v77 = (unsigned int)v308;
  }
  *(_QWORD *)&v307[8 * v77] = v70;
  v79 = *(_QWORD *)(a2 + 560);
  v80 = *(_QWORD *)(a2 + 2632);
  LODWORD(v308) = v308 + 1;
  v377 = 1;
  v373 = "zero.addr.use";
  v376 = 3;
  v81 = sub_AA4E30(v79);
  v82 = sub_AE5020(v81, v80);
  LOWORD(v391) = 257;
  v83 = sub_BD2C40(80, 1u);
  v84 = (__int64)v83;
  if ( v83 )
    sub_B4D190((__int64)v83, v80, (__int64)v283, (__int64)&v387, 0, v82, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, char **, __int64, __int64))(**(_QWORD **)(a2 + 600) + 16LL))(
    *(_QWORD *)(a2 + 600),
    v84,
    &v373,
    v296[7],
    v296[8]);
  v87 = *(_QWORD *)(a2 + 512);
  v88 = v87 + 16LL * *(unsigned int *)(a2 + 520);
  while ( v88 != v87 )
  {
    v89 = *(_QWORD *)(v87 + 8);
    v90 = *(_DWORD *)v87;
    v87 += 16;
    sub_B99FD0(v84, v90, v89);
  }
  v91 = (unsigned int)v308;
  v92 = (unsigned int)v308 + 1LL;
  if ( v92 > HIDWORD(v308) )
  {
    sub_C8D5F0((__int64)&v307, v309, v92, 8u, v85, v86);
    v91 = (unsigned int)v308;
  }
  v93 = v307;
  *(_QWORD *)&v307[8 * v91] = v84;
  v94 = *((_QWORD *)v268 + 7);
  v374 = v272;
  v388 = v94;
  LOWORD(v375[0]) = v267;
  LOWORD(v389) = 1;
  v373 = v270;
  LODWORD(v308) = v308 + 1;
  v387 = v268;
  a4(&v361, a5, v93, v387, v85, v86, v270, v272, v375[0], v387, v94, v389);
  v95 = v361.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v361.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    a1[1].m128i_i8[8] |= 3u;
    a1->m128i_i64[0] = v95;
    goto LABEL_155;
  }
  v361.m128i_i64[0] = 0;
  sub_9C66B0(v361.m128i_i64);
  v98 = *(_BYTE *)(a2 + 336) == 0;
  v347 = 0;
  v352 = v354;
  v353 = 0x200000000LL;
  if ( v98 )
  {
    v373 = (char *)v375;
    v374 = 0x400000000LL;
    if ( (_DWORD)v308 )
    {
      sub_3120FA0((__int64)&v373, &v307, 0x400000000LL, (__int64)v375, v96, v97);
      v378 = a2;
      v379 = v275;
      v387 = (char *)&v389;
      v380 = (__int64)v282;
      v388 = 0x400000000LL;
      v381 = a12;
      v382 = (__int64)v287;
      v383 = v276;
      if ( (_DWORD)v374 )
      {
        sub_3120FA0((__int64)&v387, &v373, 0x400000000LL, v223, v224, v225);
        v201 = v378;
        v200 = v379;
        v282 = (char *)v380;
        v276 = v383;
        a12 = v381;
        v199 = (_QWORD *)v382;
      }
      else
      {
        v199 = v287;
        v200 = v275;
        v201 = a2;
      }
    }
    else
    {
      v378 = a2;
      v387 = (char *)&v389;
      v380 = (__int64)v282;
      v383 = v276;
      v381 = a12;
      v199 = v287;
      v379 = v275;
      v382 = (__int64)v287;
      v388 = 0x400000000LL;
      v200 = v275;
      v201 = a2;
    }
    v393 = v201;
    v394 = v200;
    v395 = (__int64)v282;
    v397 = (__int64)v199;
    v396 = a12;
    v362 = 0;
    v398 = v276;
    v202 = (_QWORD *)sub_22077B0(0x60u);
    if ( v202 )
    {
      v205 = v388;
      *v202 = v202 + 2;
      v202[1] = 0x400000000LL;
      if ( v205 )
      {
        v294 = v202;
        sub_3120FA0((__int64)v202, &v387, (__int64)(v202 + 2), 0x400000000LL, v203, v204);
        v202 = v294;
      }
      v202[6] = v393;
      v202[7] = v394;
      v202[8] = v395;
      v202[9] = v396;
      v202[10] = v397;
      v202[11] = v398;
    }
    v361.m128i_i64[0] = (__int64)v202;
    v206 = _mm_loadu_si128(&v361);
    v207 = _mm_loadu_si128(&v346);
    v362 = v347;
    v347 = sub_3121100;
    v361 = v207;
    v363 = v348;
    v109 = sub_31383F0;
    v346 = v206;
  }
  else
  {
    v373 = (char *)v375;
    v374 = 0x400000000LL;
    if ( (_DWORD)v308 )
    {
      sub_3120FA0((__int64)&v373, &v307, 0x400000000LL, (__int64)v375, v96, v97);
      v378 = a2;
      v379 = v275;
      v387 = (char *)&v389;
      v380 = a7;
      v388 = 0x400000000LL;
      v381 = (__int64)v282;
      v382 = a12;
      v383 = a13;
      v384 = v287;
      v385 = v276;
      v386 = v278;
      if ( (_DWORD)v374 )
      {
        sub_3120FA0((__int64)&v387, &v373, 0x400000000LL, (unsigned int)v374, v221, v222);
        v102 = v378;
        v101 = v379;
        v99 = v380;
        v282 = (char *)v381;
        v100 = v384;
        a12 = v382;
        a13 = v383;
        v276 = v385;
        v278 = v386;
      }
      else
      {
        v100 = v287;
        v99 = a7;
        v102 = a2;
        v101 = v275;
      }
    }
    else
    {
      v99 = a7;
      v378 = a2;
      v100 = v287;
      v387 = (char *)&v389;
      v381 = (__int64)v282;
      v380 = a7;
      v382 = a12;
      v384 = v287;
      v383 = a13;
      v379 = v275;
      v385 = v276;
      v388 = 0x400000000LL;
      v101 = v275;
      v102 = a2;
      v386 = v278;
    }
    v393 = v102;
    v394 = v101;
    v396 = (__int64)v282;
    v395 = v99;
    v397 = a12;
    v399 = v100;
    v398 = a13;
    v362 = 0;
    v400 = v276;
    v401 = v278;
    v103 = (_QWORD *)sub_22077B0(0x78u);
    if ( v103 )
    {
      *v103 = v103 + 2;
      v106 = (unsigned int)v388;
      v103[1] = 0x400000000LL;
      if ( (_DWORD)v106 )
      {
        v293 = v103;
        sub_3120FA0((__int64)v103, &v387, v106, 0x400000000LL, v104, v105);
        v103 = v293;
      }
      v103[6] = v393;
      v103[7] = v394;
      v103[8] = v395;
      v103[9] = v396;
      v103[10] = v397;
      v103[11] = v398;
      v103[12] = v399;
      v103[13] = v400;
      v103[14] = v401;
    }
    v361.m128i_i64[0] = (__int64)v103;
    v107 = _mm_loadu_si128(&v361);
    v108 = _mm_loadu_si128(&v346);
    v362 = v347;
    v347 = sub_3121200;
    v361 = v108;
    v363 = v348;
    v109 = sub_3139780;
    v346 = v107;
  }
  v348 = v109;
  sub_A17130((__int64)&v361);
  if ( v387 != (char *)&v389 )
    _libc_free((unsigned __int64)v387);
  if ( v373 != (char *)v375 )
    _libc_free((unsigned __int64)v373);
  v387 = 0;
  v349 = v280;
  v351 = a7;
  v373 = (char *)v375;
  v374 = 0x2000000000LL;
  v350 = v298;
  v388 = (unsigned __int64)&v391;
  v389 = 32;
  LODWORD(v390) = 0;
  BYTE4(v390) = 1;
  sub_3136910((__int64)&v346, (__int64)&v387, (__int64)&v373, (__int64)&v346, v110, v111);
  sub_29B4290((__int64)v355, v275);
  v340.m128i_i64[0] = (__int64)&v341;
  sub_3120C40(v340.m128i_i64, ".omp_par", (__int64)"");
  sub_29AFB10((__int64)&v361, (__int64 *)v373, (unsigned int)v374, 0, 0, 0, 0, 0, 1, 1, a7, (__int64)&v340, v284);
  sub_2240A30((unsigned __int64 *)&v340);
  v314 = &v316;
  v332 = &v334;
  v320 = &v322;
  v326 = &v328;
  v299 = 0;
  v310 = 0;
  v311 = 0;
  v312 = 0;
  v313 = 0;
  v315 = 0;
  v316 = 0;
  v317 = 0;
  v318 = 0;
  v319 = 0;
  v321 = 0;
  v322 = 0;
  v323 = 0;
  v324 = 0;
  v325 = 0;
  v327 = 0;
  v328 = 0;
  v329 = 0;
  v330 = 0;
  v331 = 0;
  v333 = 0;
  sub_29B2FB0((__int64)&v361, (__int64)v355, (__int64)&v322, (__int64)&v328, &v299);
  sub_29B2CD0((__int64)&v361, (__int64)&v310, (__int64)&v316, (__int64)&v322, 1);
  v112 = v314;
  v113 = 8LL * (unsigned int)v315;
  v114 = &v314[(unsigned __int64)v113 / 8];
  v115 = v113 >> 3;
  v116 = v113 >> 5;
  if ( v116 )
  {
    v117 = &v314[4 * v116];
    while ( 1 )
    {
      v122 = *v112;
      if ( *v112 )
      {
        if ( *(_BYTE *)v122 == 3 && *(_QWORD *)(v122 + 24) == *(_QWORD *)(a2 + 2776) )
          break;
      }
      v118 = v112[1];
      v119 = v112 + 1;
      if ( v118 && *(_BYTE *)v118 == 3 && *(_QWORD *)(v118 + 24) == *(_QWORD *)(a2 + 2776) )
      {
        if ( (_DWORD)v313 )
        {
          v185 = (v313 - 1) & (((unsigned int)v118 >> 9) ^ ((unsigned int)v118 >> 4));
          v186 = (__int64 *)(v311 + 8LL * v185);
          v187 = *v186;
          if ( v118 == *v186 )
          {
LABEL_175:
            *v186 = -8192;
            v112 = v119;
            LODWORD(v312) = v312 - 1;
            ++HIDWORD(v312);
            goto LABEL_69;
          }
          v234 = 1;
          while ( v187 != -4096 )
          {
            v235 = v234 + 1;
            v185 = (v313 - 1) & (v234 + v185);
            v186 = (__int64 *)(v311 + 8LL * v185);
            v187 = *v186;
            if ( v118 == *v186 )
              goto LABEL_175;
            v234 = v235;
          }
        }
        goto LABEL_188;
      }
      v120 = v112[2];
      v119 = v112 + 2;
      if ( v120 && *(_BYTE *)v120 == 3 && *(_QWORD *)(v120 + 24) == *(_QWORD *)(a2 + 2776) )
      {
        if ( (_DWORD)v313 )
        {
          v188 = (v313 - 1) & (((unsigned int)v120 >> 9) ^ ((unsigned int)v120 >> 4));
          v186 = (__int64 *)(v311 + 8LL * v188);
          v189 = *v186;
          if ( *v186 == v120 )
            goto LABEL_175;
          v190 = 1;
          while ( v189 != -4096 )
          {
            v191 = v190 + 1;
            v188 = (v313 - 1) & (v190 + v188);
            v186 = (__int64 *)(v311 + 8LL * v188);
            v189 = *v186;
            if ( v120 == *v186 )
              goto LABEL_175;
            v190 = v191;
          }
        }
        goto LABEL_188;
      }
      v121 = v112[3];
      v119 = v112 + 3;
      if ( v121 && *(_BYTE *)v121 == 3 && *(_QWORD *)(v121 + 24) == *(_QWORD *)(a2 + 2776) )
      {
        if ( (_DWORD)v313 )
        {
          v192 = (v313 - 1) & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4));
          v186 = (__int64 *)(v311 + 8LL * v192);
          v193 = *v186;
          if ( *v186 == v121 )
            goto LABEL_175;
          v194 = 1;
          while ( v193 != -4096 )
          {
            v195 = v194 + 1;
            v192 = (v313 - 1) & (v194 + v192);
            v186 = (__int64 *)(v311 + 8LL * v192);
            v193 = *v186;
            if ( v121 == *v186 )
              goto LABEL_175;
            v194 = v195;
          }
        }
LABEL_188:
        v112 = v119;
        goto LABEL_69;
      }
      v112 += 4;
      if ( v117 == v112 )
      {
        v115 = v114 - v112;
        goto LABEL_220;
      }
    }
    if ( (_DWORD)v313 )
    {
      v123 = (v313 - 1) & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4));
      v124 = (__int64 *)(v311 + 8LL * v123);
      v125 = *v124;
      if ( v122 == *v124 )
      {
LABEL_68:
        *v124 = -8192;
        LODWORD(v312) = v312 - 1;
        ++HIDWORD(v312);
      }
      else
      {
        v232 = 1;
        while ( v125 != -4096 )
        {
          v233 = v232 + 1;
          v123 = (v313 - 1) & (v232 + v123);
          v124 = (__int64 *)(v311 + 8LL * v123);
          v125 = *v124;
          if ( v122 == *v124 )
            goto LABEL_68;
          v232 = v233;
        }
      }
    }
    goto LABEL_69;
  }
LABEL_220:
  if ( v115 != 2 )
  {
    if ( v115 != 3 )
    {
      if ( v115 != 1 )
        goto LABEL_78;
      goto LABEL_223;
    }
    v230 = *v112;
    if ( *v112 && *(_BYTE *)v230 == 3 && *(_QWORD *)(v230 + 24) == *(_QWORD *)(a2 + 2776) )
    {
      if ( (_DWORD)v313 )
      {
        v256 = 1;
        for ( i = (v313 - 1) & (((unsigned int)v230 >> 9) ^ ((unsigned int)v230 >> 4)); ; i = (v313 - 1) & v261 )
        {
          v258 = (_QWORD *)(v311 + 8LL * i);
          if ( v230 == *v258 )
            break;
          if ( *v258 == -4096 )
            goto LABEL_69;
          v261 = v256 + i;
          ++v256;
        }
        *v258 = -8192;
        LODWORD(v312) = v312 - 1;
        ++HIDWORD(v312);
      }
      goto LABEL_69;
    }
    ++v112;
  }
  v231 = *v112;
  if ( *v112 && *(_BYTE *)v231 == 3 && *(_QWORD *)(v231 + 24) == *(_QWORD *)(a2 + 2776) )
  {
    if ( !(_DWORD)v313 )
    {
LABEL_69:
      if ( v114 != v112 )
      {
        for ( j = v112 + 1; v114 != j; ++j )
        {
          v127 = *j;
          if ( *j && *(_BYTE *)v127 == 3 && *(_QWORD *)(v127 + 24) == *(_QWORD *)(a2 + 2776) )
          {
            if ( (_DWORD)v313 )
            {
              v196 = (v313 - 1) & (((unsigned int)v127 >> 9) ^ ((unsigned int)v127 >> 4));
              v197 = (__int64 *)(v311 + 8LL * v196);
              v198 = *v197;
              if ( v127 == *v197 )
              {
LABEL_191:
                *v197 = -8192;
                LODWORD(v312) = v312 - 1;
                ++HIDWORD(v312);
              }
              else
              {
                v250 = 1;
                while ( v198 != -4096 )
                {
                  v251 = v250 + 1;
                  v196 = (v313 - 1) & (v196 + v250);
                  v197 = (__int64 *)(v311 + 8LL * v196);
                  v198 = *v197;
                  if ( v127 == *v197 )
                    goto LABEL_191;
                  v250 = v251;
                }
              }
            }
          }
          else
          {
            *v112++ = v127;
          }
        }
      }
      if ( v112 != &v314[(unsigned int)v315] )
        LODWORD(v315) = v112 - v314;
      goto LABEL_78;
    }
    v252 = (v313 - 1) & (((unsigned int)v231 >> 9) ^ ((unsigned int)v231 >> 4));
    v218 = (__int64 *)(v311 + 8LL * v252);
    v253 = *v218;
    if ( v231 != *v218 )
    {
      v254 = 1;
      while ( v253 != -4096 )
      {
        v255 = v254 + 1;
        v252 = (v313 - 1) & (v254 + v252);
        v218 = (__int64 *)(v311 + 8LL * v252);
        v253 = *v218;
        if ( v231 == *v218 )
          goto LABEL_228;
        v254 = v255;
      }
      goto LABEL_69;
    }
LABEL_228:
    *v218 = -8192;
    LODWORD(v312) = v312 - 1;
    ++HIDWORD(v312);
    goto LABEL_69;
  }
  ++v112;
LABEL_223:
  v216 = *v112;
  if ( *v112 && *(_BYTE *)v216 == 3 && *(_QWORD *)(v216 + 24) == *(_QWORD *)(a2 + 2776) )
  {
    if ( !(_DWORD)v313 )
      goto LABEL_69;
    v217 = (v313 - 1) & (((unsigned int)v216 >> 9) ^ ((unsigned int)v216 >> 4));
    v218 = (__int64 *)(v311 + 8LL * v217);
    v219 = *v218;
    if ( v216 != *v218 )
    {
      v259 = 1;
      while ( v219 != -4096 )
      {
        v260 = v259 + 1;
        v217 = (v313 - 1) & (v259 + v217);
        v218 = (__int64 *)(v311 + 8LL * v217);
        v219 = *v218;
        if ( v216 == *v218 )
          goto LABEL_228;
        v259 = v260;
      }
      goto LABEL_69;
    }
    goto LABEL_228;
  }
LABEL_78:
  v128 = 5;
  v265 = sub_3135910(a2, 5);
  v129 = *(_QWORD *)(v84 + 32);
  v285 = *(_QWORD *)(v84 + 40);
  if ( v285 + 48 == v129 || !v129 )
    v130 = 0;
  else
    v130 = v129 - 24;
  v290 = v130 + 24;
  v273 = sub_AA5190(a7);
  if ( v273 )
  {
    v269 = v131;
    v271 = HIBYTE(v131);
  }
  else
  {
    v271 = 0;
    v269 = 0;
  }
  v134 = v314;
  v277 = &v314[(unsigned int)v315];
  if ( v277 != v314 )
  {
    do
    {
      v135 = *v134;
      if ( (_QWORD *)*v134 == v286 || (_QWORD *)v135 == v283 )
      {
        v174 = (unsigned int)v353;
        v175 = (unsigned int)v353 + 1LL;
        if ( v175 > HIDWORD(v353) )
        {
          v128 = (__int64)v354;
          sub_C8D5F0((__int64)&v352, v354, v175, 8u, v132, v133);
          v174 = (unsigned int)v353;
        }
        *(_QWORD *)&v352[8 * v174] = v135;
        LODWORD(v353) = v353 + 1;
        goto LABEL_168;
      }
      v334 = 0;
      v335 = 0;
      v336 = 0;
      v337 = 0;
      v338 = &v340;
      v339 = 0;
      for ( k = *(_QWORD *)(v135 + 16); k; k = *(_QWORD *)(k + 8) )
      {
        v137 = *(_QWORD *)(k + 24);
        if ( *(_BYTE *)v137 > 0x1Cu )
        {
          v138 = *(_QWORD *)(v137 + 40);
          if ( BYTE4(v390) )
          {
            v139 = (_QWORD *)v388;
            v140 = (_QWORD *)(v388 + 8LL * HIDWORD(v389));
            if ( (_QWORD *)v388 == v140 )
              continue;
            while ( v138 != *v139 )
            {
              if ( v140 == ++v139 )
                goto LABEL_95;
            }
          }
          else if ( !sub_C8CA60((__int64)&v387, v138) )
          {
            continue;
          }
          if ( !(_DWORD)v337 )
          {
            v334 = (_QWORD *)((char *)v334 + 1);
            goto LABEL_277;
          }
          v141 = (unsigned int)(v337 - 1);
          v142 = 1;
          v133 = 0;
          v143 = v141 & (((unsigned int)k >> 9) ^ ((unsigned int)k >> 4));
          v144 = (__int64 *)(v335 + 8LL * v143);
          v145 = *v144;
          if ( k != *v144 )
          {
            while ( v145 != -4096 )
            {
              if ( v133 || v145 != -8192 )
                v144 = (__int64 *)v133;
              v133 = (unsigned int)(v142 + 1);
              v143 = v141 & (v142 + v143);
              v145 = *(_QWORD *)(v335 + 8LL * v143);
              if ( k == v145 )
                goto LABEL_95;
              ++v142;
              v133 = (__int64)v144;
              v144 = (__int64 *)(v335 + 8LL * v143);
            }
            if ( !v133 )
              v133 = (__int64)v144;
            v334 = (_QWORD *)((char *)v334 + 1);
            v227 = v336 + 1;
            if ( 4 * ((int)v336 + 1) < (unsigned int)(3 * v337) )
            {
              if ( (int)v337 - HIDWORD(v336) - v227 <= (unsigned int)v337 >> 3 )
              {
                sub_313CFE0((__int64)&v334, v337);
                if ( !(_DWORD)v337 )
                {
LABEL_329:
                  LODWORD(v336) = v336 + 1;
                  BUG();
                }
                v141 = v335;
                v246 = 0;
                v247 = (v337 - 1) & (((unsigned int)k >> 9) ^ ((unsigned int)k >> 4));
                v133 = v335 + 8LL * v247;
                v248 = *(_QWORD *)v133;
                v227 = v336 + 1;
                v249 = 1;
                if ( *(_QWORD *)v133 != k )
                {
                  while ( v248 != -4096 )
                  {
                    if ( !v246 && v248 == -8192 )
                      v246 = v133;
                    v247 = (v337 - 1) & (v249 + v247);
                    v133 = v335 + 8LL * v247;
                    v248 = *(_QWORD *)v133;
                    if ( k == *(_QWORD *)v133 )
                      goto LABEL_254;
                    ++v249;
                  }
                  if ( v246 )
                    v133 = v246;
                }
              }
              goto LABEL_254;
            }
LABEL_277:
            sub_313CFE0((__int64)&v334, 2 * v337);
            if ( !(_DWORD)v337 )
              goto LABEL_329;
            v141 = v335;
            v242 = (v337 - 1) & (((unsigned int)k >> 9) ^ ((unsigned int)k >> 4));
            v133 = v335 + 8LL * v242;
            v227 = v336 + 1;
            v243 = *(_QWORD *)v133;
            if ( *(_QWORD *)v133 != k )
            {
              v244 = 1;
              v245 = 0;
              while ( v243 != -4096 )
              {
                if ( v243 == -8192 && !v245 )
                  v245 = v133;
                v242 = (v337 - 1) & (v244 + v242);
                v133 = v335 + 8LL * v242;
                v243 = *(_QWORD *)v133;
                if ( k == *(_QWORD *)v133 )
                  goto LABEL_254;
                ++v244;
              }
              if ( v245 )
                v133 = v245;
            }
LABEL_254:
            LODWORD(v336) = v227;
            if ( *(_QWORD *)v133 != -4096 )
              --HIDWORD(v336);
            *(_QWORD *)v133 = k;
            v228 = (unsigned int)v339;
            v229 = (unsigned int)v339 + 1LL;
            if ( v229 > HIDWORD(v339) )
            {
              sub_C8D5F0((__int64)&v338, &v340, v229, 8u, v141, v133);
              v228 = (unsigned int)v339;
            }
            v338->m128i_i64[v228] = k;
            LODWORD(v339) = v339 + 1;
          }
        }
LABEL_95:
        ;
      }
      v146 = v135;
      if ( *(_BYTE *)(*(_QWORD *)(v135 + 8) + 8LL) != 14 )
      {
        v341 = 0;
        v340 = (__m128i)(unsigned __int64)v296;
        v342 = *(_QWORD *)(a2 + 560);
        if ( v342 != 0 && v342 != -4096 && v342 != -8192 )
          sub_BD73F0((__int64)&v340.m128i_i64[1]);
        v147 = *(_QWORD *)(a2 + 568);
        v344 = *(_WORD *)(a2 + 576);
        v343 = v147;
        sub_B33910(&v345, v296);
        sub_3139A00((__int64)v296, a7, v273, v269, v271);
        v303 = (unsigned __int64)sub_BD5D20(v135);
        v304 = v148;
        v306 = 773;
        v305 = ".reloaded";
        v149 = sub_23DEB90(v296, *(__int64 **)(v135 + 8), 0, (__int64)&v303);
        v150 = *(_QWORD *)(v279 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v150 == v279 + 48 )
        {
          v151 = 0;
        }
        else
        {
          if ( !v150 )
            goto LABEL_328;
          v151 = v150 - 24;
          if ( (unsigned int)*(unsigned __int8 *)(v150 - 24) - 30 >= 0xB )
            v151 = 0;
        }
        v152 = v262;
        LOWORD(v152) = 0;
        v262 = v152;
        sub_A88F30((__int64)v296, v279, v151 + 24, 0);
        sub_2463EC0(v296, v135, v149, v264, 0);
        sub_3139A00((__int64)v296, v285, v290, 0, 0);
        v153 = v263;
        v306 = 257;
        BYTE1(v153) = 0;
        v263 = (unsigned __int8)v263;
        v146 = sub_A82CA0((unsigned int **)v296, *(_QWORD *)(v135 + 8), v149, v153, 0, (__int64)&v303);
        sub_F11320((__int64)&v340);
      }
      v300 = 0;
      if ( *(_BYTE *)v135 != 85 )
        goto LABEL_106;
      v226 = *(_QWORD *)(v135 - 32);
      if ( v226 )
      {
        if ( *(_BYTE *)v226 )
        {
          v226 = 0;
        }
        else if ( *(_QWORD *)(v226 + 24) != *(_QWORD *)(v135 + 80) )
        {
          v226 = 0;
        }
      }
      if ( v265 == (_BYTE *)v226 )
      {
        v300 = v287;
      }
      else
      {
LABEL_106:
        v154 = *(_QWORD *)(a2 + 568);
        v155 = *(_WORD *)(a2 + 576);
        v156 = *(_QWORD *)(a2 + 560);
        v304 = v290;
        LOWORD(v305) = 0;
        v301[1] = v154;
        LOWORD(v302) = v155;
        v303 = v285;
        v301[0] = v156;
        a10(&v340, a11, v135, v146, &v300, v133, v285, v290, v305, v156, v154, v302);
        v157 = v342 & 1;
        LOBYTE(v342) = (2 * (v342 & 1)) | v342 & 0xFD;
        if ( v157 )
        {
          v158 = &v340;
          sub_3139B90(v301, v340.m128i_i64);
LABEL_108:
          if ( (v342 & 2) != 0 )
            sub_267DA20(&v340, (__int64)v158);
          if ( (v342 & 1) != 0 && v340.m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v340.m128i_i64[0] + 8LL))(v340.m128i_i64[0]);
          v159 = v338;
          goto LABEL_111;
        }
        v158 = (__m128i *)v340.m128i_i64[0];
        sub_3139A00((__int64)v296, v340.m128i_i64[0], v340.m128i_i64[1], v341, SBYTE1(v341));
        v208 = *(_QWORD *)(v285 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v285 + 48 == v208 )
        {
          v210 = 0;
        }
        else
        {
          if ( !v208 )
LABEL_328:
            BUG();
          v209 = *(unsigned __int8 *)(v208 - 24);
          v210 = v208 - 24;
          if ( (unsigned int)(v209 - 30) >= 0xB )
            v210 = 0;
        }
        v290 = v210 + 24;
        if ( (_QWORD *)v135 == v300 )
        {
          v301[0] = 1;
          v303 = 0;
          sub_9C66B0((__int64 *)&v303);
          goto LABEL_108;
        }
        sub_313A080(&v340, (__int64)v158);
      }
      v211 = v338;
      v159 = (__m128i *)((char *)v338 + 8 * (unsigned int)v339);
      if ( v338 != v159 )
      {
        do
        {
          v212 = (_QWORD *)v211->m128i_i64[0];
          v213 = v300;
          if ( *(_QWORD *)v211->m128i_i64[0] )
          {
            v214 = v212[1];
            *(_QWORD *)v212[2] = v214;
            if ( v214 )
              *(_QWORD *)(v214 + 16) = v212[2];
          }
          *v212 = v213;
          if ( v213 )
          {
            v215 = v213[2];
            v212[1] = v215;
            if ( v215 )
              *(_QWORD *)(v215 + 16) = v212 + 1;
            v212[2] = v213 + 2;
            v213[2] = v212;
          }
          v211 = (__m128i *)((char *)v211 + 8);
        }
        while ( v159 != v211 );
        v159 = v338;
      }
      v301[0] = 1;
      v340.m128i_i64[0] = 0;
      sub_9C66B0(v340.m128i_i64);
LABEL_111:
      if ( v159 != &v340 )
        _libc_free((unsigned __int64)v159);
      v128 = 8LL * (unsigned int)v337;
      sub_C7D6A0(v335, v128, 8);
      v160 = v301[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v301[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        a1[1].m128i_i8[8] |= 3u;
        a1->m128i_i64[0] = v160;
        goto LABEL_115;
      }
LABEL_168:
      ++v134;
    }
    while ( v277 != v134 );
  }
  v176 = *(unsigned int *)(a2 + 8);
  v177 = _mm_loadu_si128(&v340);
  v341 = 0;
  v178 = (__m128i *)(*(_QWORD *)a2 + 40 * v176 - 40);
  v179 = _mm_loadu_si128(v178);
  *v178 = v177;
  v340 = v179;
  v180 = v178[1].m128i_i64[0];
  v178[1].m128i_i64[0] = 0;
  v341 = v180;
  v181 = v178[1].m128i_i64[1];
  v178[1].m128i_i64[1] = v342;
  v342 = v181;
  LODWORD(v343) = v178[2].m128i_i32[0];
  BYTE4(v343) = v178[2].m128i_i8[4];
  LODWORD(v176) = *(_DWORD *)(a2 + 8) - 1;
  *(_DWORD *)(a2 + 8) = v176;
  sub_A17130(*(_QWORD *)a2 + 40LL * (unsigned int)v176);
  v182 = sub_986580((__int64)v266);
  v334 = v266;
  LOWORD(v336) = 0;
  v335 = v182 + 24;
  if ( !*(_QWORD *)(a6 + 16) )
    sub_4263D6(v266, v128, v183);
  (*(void (__fastcall **)(unsigned __int64 *, __int64, _QWORD **))(a6 + 24))(&v303, a6, &v334);
  v184 = v303 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v303 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    a1[1].m128i_i8[8] |= 3u;
    a1->m128i_i64[0] = v184;
  }
  else
  {
    v303 = 0;
    sub_9C66B0((__int64 *)&v303);
    sub_3139E10(a2 + 904, (__int64)&v346, v236, v237, v238, v239);
    v240 = v274[5];
    sub_B43D60(v274);
    v241 = a1[1].m128i_i8[8];
    a1->m128i_i64[0] = v240;
    a1->m128i_i64[1] = v240 + 48;
    a1[1].m128i_i8[8] = v241 & 0xFC | 2;
    a1[1].m128i_i16[0] = 0;
  }
  sub_A17130((__int64)&v340);
LABEL_115:
  if ( v332 != &v334 )
    _libc_free((unsigned __int64)v332);
  sub_C7D6A0(v329, 8LL * (unsigned int)v331, 8);
  if ( v326 != &v328 )
    _libc_free((unsigned __int64)v326);
  sub_C7D6A0(v323, 8LL * (unsigned int)v325, 8);
  if ( v320 != &v322 )
    _libc_free((unsigned __int64)v320);
  sub_C7D6A0(v317, 8LL * (unsigned int)v319, 8);
  if ( v314 != &v316 )
    _libc_free((unsigned __int64)v314);
  sub_C7D6A0(v311, 8LL * (unsigned int)v313, 8);
  if ( v372 != &v373 )
    _libc_free((unsigned __int64)v372);
  sub_C7D6A0(v370[4], 8LL * v371, 8);
  if ( v369 != v370 )
    j_j___libc_free_0((unsigned __int64)v369);
  if ( v367 != &v368 )
    _libc_free((unsigned __int64)v367);
  if ( v366 != (unsigned __int64 *)&v367 )
    _libc_free((unsigned __int64)v366);
  sub_C7D6A0(v364, 8LL * v365, 8);
  sub_C7D6A0(v359, 8LL * v360, 8);
  v161 = v358;
  if ( v358 )
  {
    v162 = v357;
    v163 = v357 + 40LL * v358;
    do
    {
      if ( *(_QWORD *)v162 != -8192 && *(_QWORD *)v162 != -4096 )
        sub_C7D6A0(*(_QWORD *)(v162 + 16), 8LL * *(unsigned int *)(v162 + 32), 8);
      v162 += 40;
    }
    while ( v163 != v162 );
    v161 = v358;
  }
  sub_C7D6A0(v357, 40 * v161, 8);
  if ( (char *)v355[0] != &v356 )
    _libc_free(v355[0]);
  if ( v373 != (char *)v375 )
    _libc_free((unsigned __int64)v373);
  if ( !BYTE4(v390) )
    _libc_free(v388);
  if ( v352 != v354 )
    _libc_free((unsigned __int64)v352);
  if ( v347 )
    v347((unsigned __int64 **)&v346, v346.m128i_i64, 3);
LABEL_155:
  if ( v307 != v309 )
    _libc_free((unsigned __int64)v307);
  return a1;
}
