// Function: sub_28A2A70
// Address: 0x28a2a70
//
void __fastcall sub_28A2A70(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v6; // rdx
  __int64 v7; // rsi
  char v8; // r12
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // rdx
  bool v16; // zf
  int v17; // ebx
  __int64 v18; // rax
  __int64 *v19; // rsi
  __int64 *v20; // rdx
  __int64 v21; // r14
  __int64 v22; // rbx
  __int64 v23; // rcx
  int v24; // r15d
  unsigned __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __m128i v30; // xmm1
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __m128i v35; // xmm2
  int v36; // eax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rax
  __m128i v42; // xmm0
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __m128i v47; // xmm3
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  __m128i v52; // xmm4
  __int64 v53; // r14
  __int64 v54; // r8
  unsigned int v55; // esi
  __int64 v56; // r15
  __int64 v57; // rcx
  __int64 v58; // r9
  unsigned int v59; // eax
  __int64 v60; // rbx
  __int64 v61; // rdx
  __int64 v62; // rax
  __int64 v63; // rbx
  __int64 v64; // rbx
  int v65; // edx
  unsigned int v66; // ecx
  unsigned __int8 v67; // al
  __int64 *v68; // rax
  int v69; // eax
  unsigned int v70; // eax
  __int64 v71; // rax
  _BYTE *v72; // rbx
  __int64 v73; // rax
  __int64 v74; // rax
  unsigned __int16 v75; // cx
  char v76; // r8
  __int64 v77; // rsi
  unsigned int v78; // eax
  __int64 *v79; // r15
  unsigned __int64 v80; // rbx
  unsigned __int64 v81; // rax
  __int64 *v82; // rdi
  __int64 *v83; // r15
  __int64 v84; // rbx
  __int64 *i; // r13
  __int64 v86; // rdx
  __int64 *v87; // r12
  unsigned int *v88; // r15
  __int64 v89; // r13
  unsigned int *v90; // r12
  _QWORD *v91; // rdi
  __int64 v92; // r15
  __int64 v93; // rax
  unsigned int v94; // ebx
  int v95; // edx
  _QWORD *v96; // r13
  __int64 *v97; // rdx
  __int64 *v98; // rax
  __int64 v99; // rcx
  __int64 v100; // rdx
  __int64 v101; // rcx
  __int64 v102; // rbx
  __int64 v103; // rcx
  _QWORD *v104; // rax
  unsigned int v105; // r14d
  __int64 v106; // rax
  _QWORD *v107; // rcx
  int v108; // r15d
  unsigned __int64 v109; // r12
  __int64 v110; // rdx
  unsigned int v111; // edi
  __int64 v112; // rdx
  __int64 *v113; // r12
  __int64 v114; // rcx
  unsigned int v115; // r15d
  unsigned int v116; // r12d
  __int64 v117; // rcx
  unsigned int v118; // r15d
  int v119; // esi
  unsigned int v120; // r14d
  int v121; // r13d
  __int64 **v122; // rbx
  __int64 v123; // r8
  __int64 v124; // r9
  __int64 v125; // r12
  __int64 v126; // rax
  unsigned __int64 v127; // rdx
  __int64 v128; // rax
  unsigned __int8 *v129; // r15
  __int64 v130; // rax
  unsigned int v131; // r12d
  __int64 v132; // rax
  __int16 v133; // ax
  __int64 v134; // rax
  unsigned __int8 *v135; // rbx
  __int64 (__fastcall *v136)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  unsigned __int8 *v137; // r13
  __int64 (__fastcall *v138)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  _BYTE *v139; // r15
  __int64 v140; // rbx
  __int64 v141; // r13
  __int64 v142; // rax
  __int64 v143; // rax
  __int64 v144; // rax
  unsigned __int8 *v145; // r13
  __int64 v146; // rax
  unsigned __int16 v147; // cx
  unsigned int v148; // r12d
  __int64 v149; // rax
  __int16 v150; // ax
  __int64 v151; // rax
  unsigned __int8 *v152; // r15
  __int64 (__fastcall *v153)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  unsigned __int8 *v154; // r12
  __int64 (__fastcall *v155)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  _BYTE *v156; // r15
  __int64 v157; // r15
  __int64 v158; // r13
  __int64 v159; // rax
  __int64 v160; // rax
  __int64 v161; // r13
  __int64 v162; // rbx
  __int64 v163; // rdx
  unsigned int v164; // esi
  __int64 v165; // r13
  __int64 v166; // rbx
  __int64 v167; // rdx
  unsigned int v168; // esi
  __int64 v169; // r15
  __int64 v170; // rbx
  __int64 v171; // rdx
  unsigned int v172; // esi
  __int64 v173; // r13
  __int64 v174; // rbx
  __int64 v175; // rdx
  unsigned int v176; // esi
  int v177; // eax
  int v178; // ecx
  _BYTE *v179; // rax
  __m128i *v180; // r15
  __int64 v181; // rcx
  unsigned __int64 v182; // rsi
  __m128i v183; // xmm6
  unsigned __int64 v184; // rdx
  __int64 v185; // rdx
  _QWORD *v186; // rdx
  _QWORD *v187; // r14
  __m128i v188; // xmm7
  char *v189; // rdi
  unsigned __int64 v190; // rax
  __int64 v191; // rdi
  __int64 v192; // r14
  __int64 v193; // r15
  __int64 v194; // rdx
  __int64 v195; // rcx
  __int64 v196; // r8
  __int64 v197; // r9
  int v198; // edx
  unsigned __int64 v199; // rdi
  __int64 v200; // r15
  __int64 v201; // rdx
  __int64 v202; // rcx
  __int64 v203; // r8
  __int64 v204; // r9
  unsigned __int64 v205; // rdi
  __int64 *v206; // rax
  __int64 v207; // rcx
  __int64 v208; // rbx
  __int64 v209; // r8
  __int64 v210; // r12
  __int64 v211; // rdi
  __int64 v212; // rax
  unsigned __int64 v213; // rax
  __int64 v214; // rsi
  __int64 v215; // rax
  __int64 v216; // r12
  __int64 v217; // rbx
  __int64 v218; // rcx
  int v219; // edx
  unsigned __int8 v220; // al
  unsigned __int8 v221; // al
  int v222; // r13d
  __int64 v223; // r13
  _BYTE *v224; // rbx
  __int64 v225; // rdx
  unsigned int v226; // esi
  __int64 v227; // rbx
  __int64 v228; // rdx
  __int64 v229; // r8
  __int64 v230; // r9
  int v231; // eax
  int v232; // eax
  unsigned int v233; // ecx
  __int64 v234; // rax
  __int64 v235; // rcx
  __int64 v236; // rcx
  __int64 v237; // rax
  unsigned __int64 v238; // rdx
  __int64 v239; // rax
  unsigned __int64 v240; // rdx
  unsigned __int64 v241; // rax
  __int64 v242; // rsi
  int v243; // eax
  unsigned __int64 v244; // rax
  __int64 v245; // rsi
  unsigned __int64 v246; // rax
  __int16 v247; // cx
  __int64 j; // r8
  __int64 v249; // r9
  int v250; // eax
  unsigned int v251; // edx
  __int64 v252; // rax
  __int64 v253; // rdx
  __int64 v254; // rdx
  __int64 v255; // r14
  __int64 v256; // rbx
  __int64 v257; // r12
  int v258; // eax
  unsigned int v259; // eax
  __int64 v260; // rdx
  __int64 v261; // rcx
  __int64 v262; // rdi
  _QWORD *v263; // rdx
  __int64 v264; // rdx
  __int64 v265; // rcx
  __int64 v266; // r8
  __int64 v267; // r9
  __int64 v268; // rdx
  __int64 v269; // rcx
  __int64 v270; // r8
  __int64 v271; // r9
  _QWORD *v272; // rbx
  _QWORD *k; // r14
  void (__fastcall *v274)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v275; // rax
  int v276; // r8d
  _BYTE *v277; // [rsp-20h] [rbp-920h]
  unsigned int v278; // [rsp+14h] [rbp-8ECh]
  __int64 v279; // [rsp+20h] [rbp-8E0h]
  unsigned int v280; // [rsp+28h] [rbp-8D8h]
  __int64 v281; // [rsp+30h] [rbp-8D0h]
  unsigned int v282; // [rsp+44h] [rbp-8BCh]
  __int64 v283; // [rsp+50h] [rbp-8B0h]
  __int64 v284; // [rsp+58h] [rbp-8A8h]
  __int64 v285; // [rsp+70h] [rbp-890h]
  int v286; // [rsp+80h] [rbp-880h]
  __int64 v287; // [rsp+88h] [rbp-878h]
  char v289; // [rsp+98h] [rbp-868h]
  unsigned int v290; // [rsp+A0h] [rbp-860h]
  __int64 v291; // [rsp+A8h] [rbp-858h]
  unsigned int v292; // [rsp+B0h] [rbp-850h]
  char v293; // [rsp+B7h] [rbp-849h]
  char v294; // [rsp+B7h] [rbp-849h]
  char v295; // [rsp+B7h] [rbp-849h]
  char v296; // [rsp+B8h] [rbp-848h]
  __int64 v297; // [rsp+B8h] [rbp-848h]
  char v298; // [rsp+C0h] [rbp-840h]
  int v299; // [rsp+C8h] [rbp-838h]
  __int64 **v300; // [rsp+C8h] [rbp-838h]
  __int64 v301; // [rsp+D0h] [rbp-830h]
  __int64 v302; // [rsp+D0h] [rbp-830h]
  __int64 v303; // [rsp+D8h] [rbp-828h]
  __int64 *v304; // [rsp+D8h] [rbp-828h]
  unsigned int v305; // [rsp+D8h] [rbp-828h]
  unsigned int v306; // [rsp+E0h] [rbp-820h]
  __int64 v307; // [rsp+E8h] [rbp-818h]
  unsigned __int8 *v308; // [rsp+E8h] [rbp-818h]
  unsigned __int8 *v309; // [rsp+E8h] [rbp-818h]
  unsigned int v310; // [rsp+E8h] [rbp-818h]
  int v311; // [rsp+F0h] [rbp-810h]
  __int16 v312; // [rsp+F8h] [rbp-808h]
  unsigned int v313; // [rsp+F8h] [rbp-808h]
  __int64 *v314; // [rsp+100h] [rbp-800h]
  __int16 v315; // [rsp+100h] [rbp-800h]
  __int64 *v316; // [rsp+100h] [rbp-800h]
  __int64 v317; // [rsp+108h] [rbp-7F8h]
  int v318; // [rsp+108h] [rbp-7F8h]
  int v319; // [rsp+108h] [rbp-7F8h]
  __int8 *v320; // [rsp+108h] [rbp-7F8h]
  int v321; // [rsp+108h] [rbp-7F8h]
  char v323; // [rsp+118h] [rbp-7E8h]
  unsigned int v324; // [rsp+118h] [rbp-7E8h]
  int v325; // [rsp+118h] [rbp-7E8h]
  int v327; // [rsp+128h] [rbp-7D8h]
  __int64 *v328; // [rsp+128h] [rbp-7D8h]
  char v329; // [rsp+128h] [rbp-7D8h]
  __int64 v330; // [rsp+128h] [rbp-7D8h]
  __int64 v331; // [rsp+128h] [rbp-7D8h]
  unsigned int v332; // [rsp+128h] [rbp-7D8h]
  __int64 v333; // [rsp+130h] [rbp-7D0h] BYREF
  unsigned int v334; // [rsp+138h] [rbp-7C8h] BYREF
  unsigned int v335; // [rsp+13Ch] [rbp-7C4h]
  char v336; // [rsp+140h] [rbp-7C0h]
  unsigned int v337; // [rsp+144h] [rbp-7BCh] BYREF
  unsigned int v338; // [rsp+148h] [rbp-7B8h]
  char v339; // [rsp+14Ch] [rbp-7B4h]
  unsigned __int64 v340; // [rsp+150h] [rbp-7B0h] BYREF
  __int64 v341; // [rsp+158h] [rbp-7A8h]
  __int64 v342; // [rsp+160h] [rbp-7A0h]
  __int64 v343; // [rsp+168h] [rbp-798h]
  __m128i *v344; // [rsp+170h] [rbp-790h] BYREF
  __int64 v345; // [rsp+178h] [rbp-788h]
  __m128i v346; // [rsp+180h] [rbp-780h] BYREF
  __m128i v347[3]; // [rsp+1B0h] [rbp-750h] BYREF
  _BYTE *v348; // [rsp+1E0h] [rbp-720h] BYREF
  __int64 v349; // [rsp+1E8h] [rbp-718h]
  _BYTE v350[32]; // [rsp+1F0h] [rbp-710h] BYREF
  unsigned int *v351; // [rsp+210h] [rbp-6F0h] BYREF
  __int64 v352; // [rsp+218h] [rbp-6E8h]
  _BYTE v353[128]; // [rsp+220h] [rbp-6E0h] BYREF
  _BYTE *v354; // [rsp+2A0h] [rbp-660h] BYREF
  unsigned __int64 v355; // [rsp+2A8h] [rbp-658h]
  _BYTE *v356; // [rsp+2B0h] [rbp-650h] BYREF
  __int64 v357; // [rsp+2B8h] [rbp-648h]
  __int64 v358; // [rsp+2C0h] [rbp-640h]
  _BYTE *v359; // [rsp+2C8h] [rbp-638h]
  __int64 v360; // [rsp+2D0h] [rbp-630h]
  __int64 v361; // [rsp+2D8h] [rbp-628h]
  _BYTE *v362; // [rsp+2E0h] [rbp-620h]
  __int64 v363; // [rsp+2E8h] [rbp-618h]
  __int64 v364; // [rsp+2F0h] [rbp-610h]
  __m128i v365; // [rsp+330h] [rbp-5D0h]
  char v366; // [rsp+340h] [rbp-5C0h]
  _BYTE *v367; // [rsp+350h] [rbp-5B0h] BYREF
  unsigned __int64 v368; // [rsp+358h] [rbp-5A8h]
  _BYTE v369[16]; // [rsp+360h] [rbp-5A0h] BYREF
  __int16 v370; // [rsp+370h] [rbp-590h]
  __int64 v371; // [rsp+388h] [rbp-578h]
  __int64 v372; // [rsp+390h] [rbp-570h]
  __int64 v373; // [rsp+3A8h] [rbp-558h]
  __int64 v374; // [rsp+3B0h] [rbp-550h]
  int v375; // [rsp+3B8h] [rbp-548h]
  void *v376; // [rsp+3D0h] [rbp-530h]
  __m128i v377; // [rsp+3E0h] [rbp-520h]
  char v378; // [rsp+3F0h] [rbp-510h]
  __m128i v379; // [rsp+400h] [rbp-500h] BYREF
  _BYTE v380[40]; // [rsp+410h] [rbp-4F0h] BYREF
  __int64 v381; // [rsp+438h] [rbp-4C8h]
  __int64 v382; // [rsp+440h] [rbp-4C0h]
  _QWORD *v383; // [rsp+448h] [rbp-4B8h]
  __int64 v384; // [rsp+450h] [rbp-4B0h]
  __int64 v385; // [rsp+458h] [rbp-4A8h]
  __m128i v386; // [rsp+490h] [rbp-470h] BYREF
  bool v387; // [rsp+4A0h] [rbp-460h]
  __m128i v388; // [rsp+4B0h] [rbp-450h] BYREF
  _DWORD v389[4]; // [rsp+4C0h] [rbp-440h] BYREF
  __int16 v390; // [rsp+4D0h] [rbp-430h]
  __m128i v391; // [rsp+540h] [rbp-3C0h] BYREF
  bool v392; // [rsp+550h] [rbp-3B0h]
  unsigned __int64 v393; // [rsp+560h] [rbp-3A0h] BYREF
  __int64 v394; // [rsp+568h] [rbp-398h]
  _BYTE v395[16]; // [rsp+570h] [rbp-390h] BYREF
  __int16 v396; // [rsp+580h] [rbp-380h]
  __m128i v397; // [rsp+5F0h] [rbp-310h] BYREF
  bool v398; // [rsp+600h] [rbp-300h]
  __m128i v399; // [rsp+610h] [rbp-2F0h] BYREF
  __int64 v400; // [rsp+620h] [rbp-2E0h] BYREF
  char v401; // [rsp+628h] [rbp-2D8h] BYREF
  __int16 v402; // [rsp+630h] [rbp-2D0h]
  _BYTE v403[24]; // [rsp+6A0h] [rbp-260h] BYREF
  bool v404; // [rsp+6B8h] [rbp-248h]
  __int64 v405; // [rsp+820h] [rbp-E0h]
  __int64 v406; // [rsp+828h] [rbp-D8h]
  __int64 v407; // [rsp+830h] [rbp-D0h]
  __int64 v408; // [rsp+838h] [rbp-C8h]
  char v409; // [rsp+840h] [rbp-C0h]
  __int64 v410; // [rsp+848h] [rbp-B8h]
  char *v411; // [rsp+850h] [rbp-B0h]
  __int64 v412; // [rsp+858h] [rbp-A8h]
  int v413; // [rsp+860h] [rbp-A0h]
  char v414; // [rsp+864h] [rbp-9Ch]
  char v415; // [rsp+868h] [rbp-98h] BYREF
  __int16 v416; // [rsp+8A8h] [rbp-58h]
  _QWORD *v417; // [rsp+8B0h] [rbp-50h]
  _QWORD *v418; // [rsp+8B8h] [rbp-48h]
  __int64 v419; // [rsp+8C0h] [rbp-40h]

  if ( !(_BYTE)qword_5004388 || !*(_QWORD *)(a1 + 40) )
    return;
  v4 = *(_DWORD *)(a2 + 4);
  v399.m128i_i32[0] = 234;
  v399.m128i_i32[2] = 0;
  v6 = v4 & 0x7FFFFFF;
  v301 = *(_QWORD *)(a2 - 32 * v6);
  v7 = *(_QWORD *)(a2 + 32 * (1 - v6));
  v291 = v7;
  v400 = (__int64)&v333;
  if ( !dword_5003CC8 )
  {
    v8 = sub_10E25C0((__int64)&v399, v7);
    if ( !v8 )
      goto LABEL_5;
LABEL_32:
    v22 = a2;
    sub_23D0AB0((__int64)&v351, a2, 0, 0, 0);
    v314 = *(__int64 **)(*(_QWORD *)(a2 + 8) + 24LL);
    sub_28940A0(
      (__int64)&v346,
      *(_QWORD *)(v22 + 32 * (2LL - (*(_DWORD *)(v22 + 4) & 0x7FFFFFF))),
      *(_QWORD *)(v22 + 32 * (3LL - (*(_DWORD *)(v22 + 4) & 0x7FFFFFF))));
    sub_28940A0(
      (__int64)v347,
      *(_QWORD *)(v22 + 32 * (v23 - (*(_DWORD *)(v22 + 4) & 0x7FFFFFF))),
      *(_QWORD *)(v22 + 32 * (4LL - (*(_DWORD *)(v22 + 4) & 0x7FFFFFF))));
    v24 = v347[0].m128i_i32[1];
    v354 = &v356;
    v318 = v346.m128i_i32[1];
    v327 = v346.m128i_i32[0];
    v355 = v25;
    v365 = 0u;
    v366 = dword_5003CC8 == 0;
    v367 = v369;
    v368 = v25;
    v377 = 0u;
    v378 = dword_5003CC8 == 0;
    if ( dword_5003CC8 )
    {
      LOBYTE(v394) = 0;
      v393 = v346.m128i_i64[0];
      sub_2895860((__int64)&v399, a1, v333, (__int64)&v393, (__int64)&v351);
      sub_2894810((__int64)&v354, (char **)&v399, v43, v44, v45, v46);
      v47 = _mm_loadu_si128((const __m128i *)v403);
      v366 = v403[16];
      v365 = v47;
      if ( (__int64 *)v399.m128i_i64[0] != &v400 )
        _libc_free(v399.m128i_u64[0]);
      v393 = __PAIR64__(v318, v24);
      LOBYTE(v394) = dword_5003CC8 == 0;
      sub_2895860((__int64)&v399, a1, v7, (__int64)&v393, (__int64)&v351);
      sub_2894810((__int64)&v367, (char **)&v399, v48, v49, v50, v51);
      v52 = _mm_loadu_si128((const __m128i *)v403);
      v378 = v403[16];
      v377 = v52;
      if ( (__int64 *)v399.m128i_i64[0] != &v400 )
        _libc_free(v399.m128i_u64[0]);
    }
    else
    {
      v393 = v346.m128i_i64[0];
      LOBYTE(v394) = 1;
      sub_2895860((__int64)&v399, a1, v301, (__int64)&v393, (__int64)&v351);
      sub_2894810((__int64)&v354, (char **)&v399, v26, v27, v28, v29);
      v30 = _mm_loadu_si128((const __m128i *)v403);
      v366 = v403[16];
      v365 = v30;
      if ( (__int64 *)v399.m128i_i64[0] != &v400 )
        _libc_free(v399.m128i_u64[0]);
      v393 = __PAIR64__(v318, v24);
      LOBYTE(v394) = dword_5003CC8 == 0;
      sub_2895860((__int64)&v399, a1, v333, (__int64)&v393, (__int64)&v351);
      sub_2894810((__int64)&v367, (char **)&v399, v31, v32, v33, v34);
      v35 = _mm_loadu_si128((const __m128i *)v403);
      v378 = v403[16];
      v377 = v35;
      if ( (__int64 *)v399.m128i_i64[0] != &v400 )
        _libc_free(v399.m128i_u64[0]);
      v301 = v7;
    }
    sub_2895340((__int64)&v379, v327, v24, v314);
    v36 = sub_28956B0(a2);
    sub_2899430(a1, v379.m128i_i64, (__int64 *)&v354, (__int64 *)&v367, (__int64)&v351, 0, 1, v36);
    sub_BED950((__int64)&v399, a3, a2);
    v41 = *(_QWORD *)(v301 + 16);
    if ( !v41 )
      goto LABEL_39;
    v53 = *(_QWORD *)(v41 + 8);
    if ( v53 )
      goto LABEL_39;
    sub_BED950((__int64)&v399, a3, v301);
    sub_9C95B0(a1 + 96, v301);
    sub_2895340((__int64)&v388, v318, v24, v314);
    v348 = (_BYTE *)v301;
    v55 = *(_DWORD *)(a1 + 264);
    v56 = a1 + 240;
    LODWORD(v349) = 0;
    if ( v55 )
    {
      v57 = *(_QWORD *)(a1 + 248);
      v58 = 1;
      v59 = (v55 - 1) & (((unsigned int)v301 >> 9) ^ ((unsigned int)v301 >> 4));
      v60 = v57 + 16LL * v59;
      v61 = *(_QWORD *)v60;
      if ( *(_QWORD *)v60 == v301 )
      {
LABEL_57:
        v62 = *(unsigned int *)(v60 + 8);
LABEL_58:
        v63 = *(_QWORD *)(a1 + 272) + 176 * v62;
        sub_2894810(v63 + 8, (char **)&v388, 5 * v62, v57, v54, v58);
        *(__m128i *)(v63 + 152) = _mm_loadu_si128(&v391);
        *(_BYTE *)(v63 + 168) = v392;
        if ( (_DWORD *)v388.m128i_i64[0] != v389 )
          _libc_free(v388.m128i_u64[0]);
LABEL_39:
        v399.m128i_i64[0] = (__int64)&v400;
        v399.m128i_i64[1] = 0x1000000000LL;
        if ( v379.m128i_i32[2] )
          sub_2894AD0((__int64)&v399, (__int64)&v379, v37, v38, v39, v40);
        v42 = _mm_loadu_si128(&v386);
        v403[16] = v387;
        *(__m128i *)v403 = v42;
        sub_289E450(a1, a2, &v399, &v351, v39, v40);
        if ( (__int64 *)v399.m128i_i64[0] != &v400 )
          _libc_free(v399.m128i_u64[0]);
        if ( (_BYTE *)v379.m128i_i64[0] != v380 )
          _libc_free(v379.m128i_u64[0]);
        if ( v367 != v369 )
          _libc_free((unsigned __int64)v367);
        if ( v354 != (_BYTE *)&v356 )
          _libc_free((unsigned __int64)v354);
        sub_F94A20(&v351, a2);
        return;
      }
      while ( v61 != -4096 )
      {
        if ( !v53 && v61 == -8192 )
          v53 = v60;
        v54 = (unsigned int)(v58 + 1);
        v59 = (v55 - 1) & (v58 + v59);
        v60 = v57 + 16LL * v59;
        v61 = *(_QWORD *)v60;
        if ( *(_QWORD *)v60 == v301 )
          goto LABEL_57;
        v58 = (unsigned int)v54;
      }
      if ( v53 )
        v60 = v53;
      ++*(_QWORD *)(a1 + 240);
      v177 = *(_DWORD *)(a1 + 256);
      v399.m128i_i64[0] = v60;
      v178 = v177 + 1;
      if ( 4 * (v177 + 1) < 3 * v55 )
      {
        v179 = (_BYTE *)v301;
        if ( v55 - *(_DWORD *)(a1 + 260) - v178 <= v55 >> 3 )
        {
          sub_D39D40(v56, v55);
          sub_22B1A50(v56, (__int64 *)&v348, &v399);
          v179 = v348;
          v60 = v399.m128i_i64[0];
          v178 = *(_DWORD *)(a1 + 256) + 1;
        }
        goto LABEL_222;
      }
    }
    else
    {
      v399.m128i_i64[0] = 0;
      ++*(_QWORD *)(a1 + 240);
    }
    sub_D39D40(v56, 2 * v55);
    sub_22B1A50(v56, (__int64 *)&v348, &v399);
    v179 = v348;
    v60 = v399.m128i_i64[0];
    v178 = *(_DWORD *)(a1 + 256) + 1;
LABEL_222:
    *(_DWORD *)(a1 + 256) = v178;
    if ( *(_QWORD *)v60 != -4096 )
      --*(_DWORD *)(a1 + 260);
    *(_QWORD *)v60 = v179;
    v180 = &v399;
    *(_DWORD *)(v60 + 8) = v349;
    v58 = (unsigned int)dword_5003CC8;
    v399.m128i_i64[0] = v301;
    v399.m128i_i64[1] = (__int64)&v401;
    v181 = *(unsigned int *)(a1 + 280);
    v182 = *(unsigned int *)(a1 + 284);
    v397 = 0u;
    v183 = _mm_loadu_si128(&v397);
    v393 = (unsigned __int64)v395;
    v394 = 0x1000000000LL;
    v400 = 0x1000000000LL;
    v184 = v181 + 1;
    v398 = dword_5003CC8 == 0;
    v404 = dword_5003CC8 == 0;
    v62 = v181;
    *(__m128i *)&v403[8] = v183;
    if ( v181 + 1 > v182 )
    {
      v190 = *(_QWORD *)(a1 + 272);
      v191 = a1 + 272;
      v192 = a1 + 288;
      if ( v190 > (unsigned __int64)&v399 || (unsigned __int64)&v399 >= v190 + 176 * v181 )
      {
        v200 = sub_C8D7D0(v191, a1 + 288, v184, 0xB0u, &v340, (unsigned int)dword_5003CC8);
        sub_2894F90((__int64 **)(a1 + 272), v200, v201, v202, v203, v204);
        v205 = *(_QWORD *)(a1 + 272);
        if ( v192 == v205 )
        {
          *(_DWORD *)(a1 + 284) = v340;
        }
        else
        {
          v321 = v340;
          _libc_free(v205);
          *(_DWORD *)(a1 + 284) = v321;
        }
        v62 = *(unsigned int *)(a1 + 280);
        *(_QWORD *)(a1 + 272) = v200;
        v180 = &v399;
        v181 = (unsigned int)v62;
      }
      else
      {
        v320 = &v399.m128i_i8[-v190];
        v193 = sub_C8D7D0(v191, a1 + 288, v184, 0xB0u, &v340, (unsigned int)dword_5003CC8);
        sub_2894F90((__int64 **)(a1 + 272), v193, v194, v195, v196, v197);
        v198 = v340;
        v199 = *(_QWORD *)(a1 + 272);
        if ( v192 == v199 )
        {
          *(_QWORD *)(a1 + 272) = v193;
          *(_DWORD *)(a1 + 284) = v198;
        }
        else
        {
          v311 = v340;
          _libc_free(v199);
          *(_QWORD *)(a1 + 272) = v193;
          *(_DWORD *)(a1 + 284) = v311;
        }
        v180 = (__m128i *)&v320[v193];
        v181 = *(unsigned int *)(a1 + 280);
        v62 = v181;
      }
    }
    v185 = 11 * v181;
    v57 = a1;
    v185 *= 16;
    v16 = *(_QWORD *)(a1 + 272) + v185 == 0;
    v186 = (_QWORD *)(*(_QWORD *)(a1 + 272) + v185);
    v187 = v186;
    if ( !v16 )
    {
      *v186 = v180->m128i_i64[0];
      v186[1] = v186 + 3;
      v186[2] = 0x1000000000LL;
      v54 = v180[1].m128i_u32[0];
      if ( (_DWORD)v54 )
        sub_2894810((__int64)(v186 + 1), (char **)&v180->m128i_i64[1], (__int64)v186, a1, v54, v58);
      v188 = _mm_loadu_si128((__m128i *)((char *)v180 + 152));
      *((_BYTE *)v187 + 168) = v180[10].m128i_i8[8];
      *(__m128i *)(v187 + 19) = v188;
      v62 = *(unsigned int *)(a1 + 280);
    }
    v189 = (char *)v399.m128i_i64[1];
    *(_DWORD *)(a1 + 280) = v62 + 1;
    if ( v189 != &v401 )
    {
      _libc_free((unsigned __int64)v189);
      v62 = (unsigned int)(*(_DWORD *)(a1 + 280) - 1);
    }
    *(_DWORD *)(v60 + 8) = v62;
    goto LABEL_58;
  }
  v8 = sub_10E25C0((__int64)&v399, v301);
  if ( v8 )
    goto LABEL_32;
LABEL_5:
  v9 = *(_QWORD *)(a2 + 16);
  if ( v9 )
  {
    v317 = *(_QWORD *)(v9 + 8);
    if ( !v317 && !dword_5003CC8 )
    {
      v10 = v301;
      v281 = *(_QWORD *)(v9 + 24);
      if ( *(_BYTE *)v301 != 61 )
        v10 = 0;
      v302 = v10;
      if ( *(_BYTE *)v291 == 61 && *(_BYTE *)v281 == 62 && v10 )
      {
        v340 = 0;
        v341 = 0;
        v342 = 0;
        v343 = 0;
        v344 = &v346;
        v345 = 0;
        v393 = *(_QWORD *)(v281 - 32);
        v399.m128i_i64[0] = 0;
        v340 = 1;
        sub_CE2A30((__int64)&v340, 0);
        sub_DA5B20((__int64)&v340, (__int64 *)&v393, &v399);
        LODWORD(v342) = v342 + 1;
        if ( *(_QWORD *)v399.m128i_i64[0] != -4096 )
          --HIDWORD(v342);
        *(_QWORD *)v399.m128i_i64[0] = v393;
        v13 = (unsigned int)v345;
        v14 = v393;
        v15 = (unsigned int)v345 + 1LL;
        if ( v15 > HIDWORD(v345) )
        {
          sub_C8D5F0((__int64)&v344, &v346, v15, 8u, v11, v12);
          v13 = (unsigned int)v345;
        }
        v344->m128i_i64[v13] = v14;
        v16 = (_DWORD)v345 == -1;
        LODWORD(v345) = v345 + 1;
        v351 = (unsigned int *)v353;
        v352 = 0x600000000LL;
        if ( !v16 )
        {
          v323 = v8;
          v17 = 0;
          v18 = 0;
          v307 = a4;
          do
          {
            v21 = v344->m128i_i64[v18];
            if ( *(_BYTE *)v21 > 0x1Cu )
            {
              if ( *(_BYTE *)v21 == 84 )
                goto LABEL_90;
              if ( !(unsigned __int8)sub_B19DB0(*(_QWORD *)(a1 + 40), v21, a2) )
              {
                if ( (unsigned __int8)sub_B46970((unsigned __int8 *)v21) || (unsigned __int8)sub_B46420(v21) )
                  goto LABEL_90;
                sub_9C95B0((__int64)&v351, v21);
                if ( (*(_BYTE *)(v21 + 7) & 0x40) != 0 )
                {
                  v19 = *(__int64 **)(v21 - 8);
                  v20 = &v19[4 * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF)];
                }
                else
                {
                  v20 = (__int64 *)v21;
                  v19 = (__int64 *)(v21 - 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF));
                }
                sub_289F980((__int64)&v340, v19, v20);
              }
            }
            v18 = (unsigned int)(v17 + 1);
            v17 = v18;
          }
          while ( (_DWORD)v18 != (_DWORD)v345 );
          v79 = (__int64 *)v351;
          a4 = v307;
          v80 = 2LL * (unsigned int)v352;
          if ( v351 != &v351[v80] )
          {
            v328 = (__int64 *)&v351[v80];
            _BitScanReverse64(&v81, (__int64)(v80 * 4) >> 3);
            sub_2894BE0((__int64 *)v351, (__int64 *)&v351[v80], 2LL * (int)(63 - (v81 ^ 0x3F)), a1);
            if ( v80 <= 32 )
            {
              sub_2895090(v79, v328, a1);
            }
            else
            {
              v82 = v79;
              v83 = v79 + 16;
              sub_2895090(v82, v83, a1);
              if ( v328 != v83 )
              {
                do
                {
                  v84 = *v83;
                  for ( i = v83; ; i[1] = *i )
                  {
                    v86 = *(i - 1);
                    v87 = i--;
                    if ( !(unsigned __int8)sub_B19DB0(*(_QWORD *)(a1 + 40), v84, v86) )
                      break;
                  }
                  *v87 = v84;
                  ++v83;
                }
                while ( v328 != v83 );
                v8 = v323;
                a4 = v307;
              }
            }
            v88 = &v351[2 * (unsigned int)v352];
            if ( v88 != v351 )
            {
              v89 = v303;
              v329 = v8;
              v90 = v351;
              do
              {
                v91 = *(_QWORD **)v90;
                LOWORD(v89) = 0;
                v90 += 2;
                sub_B444E0(v91, a2 + 24, v89);
              }
              while ( v88 != v90 );
              v8 = v329;
            }
          }
        }
        sub_D665A0(&v346, v302);
        sub_D665A0(v347, v291);
        v92 = *(_QWORD *)(v281 + 40);
        if ( *(_QWORD *)(v302 + 40) == v92 )
          v8 = *(_QWORD *)(v291 + 40) == v92;
        v93 = 0;
        v94 = 0;
        if ( *(_DWORD *)(a4 + 8) )
        {
          do
          {
            v330 = 8 * v93;
            v96 = *(_QWORD **)(*(_QWORD *)a4 + 8 * v93);
            if ( (!(unsigned __int8)sub_B19DB0(*(_QWORD *)(a1 + 40), (__int64)v96, v302)
               || !(unsigned __int8)sub_B19DB0(*(_QWORD *)(a1 + 40), (__int64)v96, v291))
              && !(unsigned __int8)sub_B19DB0(*(_QWORD *)(a1 + 40), v281, (__int64)v96)
              && (!v8 || v92 == v96[5]) )
            {
              sub_D669C0(&v399, (__int64)v96, 1u, 0);
              if ( v399.m128i_i64[0] )
              {
                if ( (unsigned __int8)sub_CF4E00(*(_QWORD *)(a1 + 32), (__int64)&v346, (__int64)&v399)
                  || (unsigned __int8)sub_CF4E00(*(_QWORD *)(a1 + 32), (__int64)v347, (__int64)&v399) )
                {
                  if ( v92 != v96[5] )
                  {
                    sub_9C95B0(a1 + 96, (__int64)v96);
                    v97 = (__int64 *)(*(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8) - 8);
                    v98 = (__int64 *)(v330 + *(_QWORD *)a4);
                    v99 = *v98;
                    *v98 = *v97;
                    *v97 = v99;
                    v95 = *(_DWORD *)(a4 + 8) - 1;
                    *(_DWORD *)(a4 + 8) = v95;
                    goto LABEL_113;
                  }
                  sub_B44530(v96, v281);
                }
              }
            }
            v95 = *(_DWORD *)(a4 + 8);
            ++v94;
LABEL_113:
            v93 = v94;
          }
          while ( v94 != v95 );
        }
        if ( (_BYTE)qword_50040E8 )
          goto LABEL_134;
        v100 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
        v101 = *(_QWORD *)(a2 + 32 * (3 - v100));
        v102 = *(_QWORD *)(v101 + 24);
        if ( *(_DWORD *)(v101 + 32) > 0x40u )
          v102 = **(_QWORD **)(v101 + 24);
        v103 = *(_QWORD *)(a2 + 32 * (2 - v100));
        v104 = *(_QWORD **)(v103 + 24);
        if ( *(_DWORD *)(v103 + 32) > 0x40u )
          v104 = (_QWORD *)*v104;
        v105 = (unsigned int)v104;
        v106 = *(_QWORD *)(a2 + 32 * (4 - v100));
        v107 = *(_QWORD **)(v106 + 24);
        if ( *(_DWORD *)(v106 + 32) > 0x40u )
          v107 = (_QWORD *)*v107;
        v325 = (int)v107;
        v108 = (int)v107;
        v331 = *(_QWORD *)(*(_QWORD *)(a2 + 8) + 24LL);
        v399.m128i_i64[0] = sub_DFB1B0(*(_QWORD *)(a1 + 16));
        v109 = v399.m128i_i64[0];
        v399.m128i_i64[1] = v110;
        v111 = 1;
        v393 = sub_BCAE30(v331);
        v394 = v112;
        if ( (unsigned int)(v109 / v393) )
          v111 = v109 / v393;
        if ( v105 > v111 || v325 != 1 )
        {
          v113 = *(__int64 **)(a1 + 16);
          sub_DFB180(v113, 1u);
          if ( (_DWORD)v102 * ((v105 + v111 - 1) / v111) + v108 * ((v111 - 1 + (unsigned int)v102) / v111) > (unsigned int)sub_DFB120((__int64)v113) )
          {
LABEL_134:
            sub_28940A0(
              (__int64)&v334,
              *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
              *(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
            sub_28940A0(
              (__int64)&v337,
              *(_QWORD *)(a2 + 32 * (v114 - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
              *(_QWORD *)(a2 + 32 * (4LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
            v282 = v334;
            v115 = v334;
            v278 = v338;
            v116 = v338;
            v306 = v335;
            v304 = *(__int64 **)(*(_QWORD *)(a2 + 8) + 24LL);
            v284 = sub_28A1B60((__int64 *)a1, v302, v281, v117);
            v283 = sub_28A1B60((__int64 *)a1, v291, v281, a2);
            v279 = *(_QWORD *)(v281 - 32);
            if ( !byte_50041C8 || v115 % (unsigned int)qword_50042A8 || (v118 = v116 % (unsigned int)qword_50042A8) != 0 )
            {
              v77 = v281;
              sub_23D0AB0((__int64)&v379, v281, 0, 0, 0);
              v280 = 0;
              v78 = qword_50042A8;
              if ( v278 )
              {
                while ( 1 )
                {
                  v77 = v282;
                  if ( v282 )
                    break;
LABEL_83:
                  v280 += v78;
                  if ( v278 <= v280 )
                    goto LABEL_84;
                }
                v292 = 0;
                while ( 1 )
                {
                  v391 = 0u;
                  v119 = v282 - v292;
                  if ( v282 - v292 > v78 )
                    v119 = v78;
                  if ( v278 - v280 <= v78 )
                    v78 = v278 - v280;
                  v299 = v119;
                  v286 = v78;
                  v120 = v78;
                  v392 = dword_5003CC8 == 0;
                  v121 = 0;
                  v388.m128i_i64[0] = (__int64)v389;
                  v388.m128i_i64[1] = 0x1000000000LL;
                  v122 = (__int64 **)sub_BCDA70(v304, v119);
                  if ( v120 )
                  {
                    do
                    {
                      v125 = sub_AC9350(v122);
                      v126 = v388.m128i_u32[2];
                      v127 = v388.m128i_u32[2] + 1LL;
                      if ( v127 > v388.m128i_u32[3] )
                      {
                        sub_C8D5F0((__int64)&v388, v389, v127, 8u, v123, v124);
                        v126 = v388.m128i_u32[2];
                      }
                      ++v121;
                      *(_QWORD *)(v388.m128i_i64[0] + 8 * v126) = v125;
                      ++v388.m128i_i32[2];
                    }
                    while ( v121 != v286 );
                  }
                  if ( v306 )
                    break;
LABEL_80:
                  v71 = sub_BCB2E0(v383);
                  v72 = (_BYTE *)sub_ACD640(v71, v280, 0);
                  v73 = sub_BCB2E0(v383);
                  v74 = sub_ACD640(v73, v292, 0);
                  v75 = *(_WORD *)(v281 + 2);
                  v76 = v75 & 1;
                  v277 = (_BYTE *)v74;
                  _BitScanReverse64((unsigned __int64 *)&v74, 1LL << (v75 >> 1));
                  v77 = (__int64)&v388;
                  LOBYTE(v75) = 63 - (v74 ^ 0x3F);
                  HIBYTE(v75) = 1;
                  sub_289AF80(
                    a1,
                    &v388,
                    v279,
                    v75,
                    v76,
                    v282,
                    v306,
                    dword_5003CC8 == 0,
                    v277,
                    v72,
                    v304,
                    (__int64)&v379);
                  if ( (_DWORD *)v388.m128i_i64[0] != v389 )
                    _libc_free(v388.m128i_u64[0]);
                  v78 = qword_50042A8;
                  v292 += qword_50042A8;
                  if ( v282 <= v292 )
                    goto LABEL_83;
                }
                v70 = qword_50042A8;
                v324 = 0;
                while ( 1 )
                {
                  if ( v306 - v324 <= v70 )
                    v70 = v306 - v324;
                  v319 = v70;
                  v293 = dword_5003CC8 == 0;
                  v128 = sub_BCB2E0(v383);
                  v129 = (unsigned __int8 *)sub_ACD640(v128, v324, 0);
                  v130 = sub_BCB2E0(v383);
                  v132 = sub_ACD640(v130, v292, 0);
                  v131 = v334;
                  v308 = (unsigned __int8 *)v132;
                  v296 = *(_WORD *)(v302 + 2) & 1;
                  _BitScanReverse64((unsigned __int64 *)&v132, 1LL << (*(_WORD *)(v302 + 2) >> 1));
                  if ( !v336 )
                    v131 = v335;
                  LOBYTE(v133) = 63 - (v132 ^ 0x3F);
                  HIBYTE(v133) = 1;
                  v315 = v133;
                  v396 = 257;
                  v370 = 257;
                  v134 = sub_BCB2E0(v383);
                  v135 = (unsigned __int8 *)sub_ACD640(v134, v131, 0);
                  v136 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v384 + 32LL);
                  if ( v136 != sub_9201A0 )
                    break;
                  if ( *v129 <= 0x15u && *v135 <= 0x15u )
                  {
                    if ( (unsigned __int8)sub_AC47B0(17) )
                      v137 = (unsigned __int8 *)sub_AD5570(17, (__int64)v129, v135, 0, 0);
                    else
                      v137 = (unsigned __int8 *)sub_AABE40(0x11u, v129, v135);
LABEL_159:
                    if ( v137 )
                      goto LABEL_160;
                  }
                  v402 = 257;
                  v137 = (unsigned __int8 *)sub_B504D0(17, (__int64)v129, (__int64)v135, (__int64)&v399, 0, 0);
                  (*(void (__fastcall **)(__int64, unsigned __int8 *, _BYTE **, __int64, __int64))(*(_QWORD *)v385 + 16LL))(
                    v385,
                    v137,
                    &v367,
                    v381,
                    v382);
                  v169 = v379.m128i_i64[0];
                  v170 = v379.m128i_i64[0] + 16LL * v379.m128i_u32[2];
                  if ( v379.m128i_i64[0] != v170 )
                  {
                    do
                    {
                      v171 = *(_QWORD *)(v169 + 8);
                      v172 = *(_DWORD *)v169;
                      v169 += 16;
                      sub_B99FD0((__int64)v137, v172, v171);
                    }
                    while ( v170 != v169 );
                  }
LABEL_160:
                  v138 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v384 + 32LL);
                  if ( v138 != sub_9201A0 )
                  {
                    v139 = (_BYTE *)v138(v384, 13u, v137, v308, 0, 0);
                    goto LABEL_165;
                  }
                  if ( *v137 <= 0x15u && *v308 <= 0x15u )
                  {
                    if ( (unsigned __int8)sub_AC47B0(13) )
                      v139 = (_BYTE *)sub_AD5570(13, (__int64)v137, v308, 0, 0);
                    else
                      v139 = (_BYTE *)sub_AABE40(0xDu, v137, v308);
LABEL_165:
                    if ( v139 )
                      goto LABEL_166;
                  }
                  v402 = 257;
                  v139 = (_BYTE *)sub_B504D0(13, (__int64)v137, (__int64)v308, (__int64)&v399, 0, 0);
                  (*(void (__fastcall **)(__int64, _BYTE *, unsigned __int64 *, __int64, __int64))(*(_QWORD *)v385 + 16LL))(
                    v385,
                    v139,
                    &v393,
                    v381,
                    v382);
                  v165 = v379.m128i_i64[0];
                  v166 = v379.m128i_i64[0] + 16LL * v379.m128i_u32[2];
                  if ( v379.m128i_i64[0] != v166 )
                  {
                    do
                    {
                      v167 = *(_QWORD *)(v165 + 8);
                      v168 = *(_DWORD *)v165;
                      v165 += 16;
                      sub_B99FD0((__int64)v139, v168, v167);
                    }
                    while ( v166 != v165 );
                  }
LABEL_166:
                  v402 = 257;
                  v354 = v139;
                  v140 = sub_921130((unsigned int **)&v379, (__int64)v304, v284, &v354, 1, (__int64)&v399, 0);
                  v141 = sub_BCDA70(v304, v319 * v299);
                  v142 = sub_BCB2E0(v383);
                  v143 = sub_ACD640(v142, v131, 0);
                  sub_289A9E0((__int64)&v393, a1, v141, v140, v315, v143, v296, v299, v319, v293, (__int64)&v379);
                  v294 = dword_5003CC8 == 0;
                  v144 = sub_BCB2E0(v383);
                  v145 = (unsigned __int8 *)sub_ACD640(v144, v280, 0);
                  v146 = sub_BCB2E0(v383);
                  v309 = (unsigned __int8 *)sub_ACD640(v146, v324, 0);
                  v147 = *(_WORD *)(v291 + 2);
                  LOWORD(v358) = 257;
                  v148 = v337;
                  v370 = 257;
                  v298 = v147 & 1;
                  _BitScanReverse64((unsigned __int64 *)&v149, 1LL << (v147 >> 1));
                  if ( !v339 )
                    v148 = v338;
                  LOBYTE(v150) = 63 - (v149 ^ 0x3F);
                  HIBYTE(v150) = 1;
                  v312 = v150;
                  v297 = v148;
                  v151 = sub_BCB2E0(v383);
                  v152 = (unsigned __int8 *)sub_ACD640(v151, v148, 0);
                  v153 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v384 + 32LL);
                  if ( v153 != sub_9201A0 )
                  {
                    v154 = (unsigned __int8 *)v153(v384, 17u, v145, v152, 0, 0);
                    goto LABEL_173;
                  }
                  if ( *v145 <= 0x15u && *v152 <= 0x15u )
                  {
                    if ( (unsigned __int8)sub_AC47B0(17) )
                      v154 = (unsigned __int8 *)sub_AD5570(17, (__int64)v145, v152, 0, 0);
                    else
                      v154 = (unsigned __int8 *)sub_AABE40(0x11u, v145, v152);
LABEL_173:
                    if ( v154 )
                      goto LABEL_174;
                  }
                  v402 = 257;
                  v154 = (unsigned __int8 *)sub_B504D0(17, (__int64)v145, (__int64)v152, (__int64)&v399, 0, 0);
                  (*(void (__fastcall **)(__int64, unsigned __int8 *, _BYTE **, __int64, __int64))(*(_QWORD *)v385 + 16LL))(
                    v385,
                    v154,
                    &v354,
                    v381,
                    v382);
                  v161 = v379.m128i_i64[0];
                  v162 = v379.m128i_i64[0] + 16LL * v379.m128i_u32[2];
                  if ( v379.m128i_i64[0] != v162 )
                  {
                    do
                    {
                      v163 = *(_QWORD *)(v161 + 8);
                      v164 = *(_DWORD *)v161;
                      v161 += 16;
                      sub_B99FD0((__int64)v154, v164, v163);
                    }
                    while ( v162 != v161 );
                  }
LABEL_174:
                  v155 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v384 + 32LL);
                  if ( v155 == sub_9201A0 )
                  {
                    if ( *v154 > 0x15u || *v309 > 0x15u )
                    {
LABEL_193:
                      v402 = 257;
                      v156 = (_BYTE *)sub_B504D0(13, (__int64)v154, (__int64)v309, (__int64)&v399, 0, 0);
                      (*(void (__fastcall **)(__int64, _BYTE *, _BYTE **, __int64, __int64))(*(_QWORD *)v385 + 16LL))(
                        v385,
                        v156,
                        &v367,
                        v381,
                        v382);
                      v173 = v379.m128i_i64[0];
                      v174 = v379.m128i_i64[0] + 16LL * v379.m128i_u32[2];
                      if ( v379.m128i_i64[0] != v174 )
                      {
                        do
                        {
                          v175 = *(_QWORD *)(v173 + 8);
                          v176 = *(_DWORD *)v173;
                          v173 += 16;
                          sub_B99FD0((__int64)v156, v176, v175);
                        }
                        while ( v174 != v173 );
                      }
                      goto LABEL_180;
                    }
                    if ( (unsigned __int8)sub_AC47B0(13) )
                      v156 = (_BYTE *)sub_AD5570(13, (__int64)v154, v309, 0, 0);
                    else
                      v156 = (_BYTE *)sub_AABE40(0xDu, v154, v309);
                  }
                  else
                  {
                    v156 = (_BYTE *)v155(v384, 13u, v154, v309, 0, 0);
                  }
                  if ( !v156 )
                    goto LABEL_193;
LABEL_180:
                  v402 = 257;
                  v348 = v156;
                  v157 = sub_921130((unsigned int **)&v379, (__int64)v304, v283, &v348, 1, (__int64)&v399, 0);
                  v158 = sub_BCDA70(v304, v286 * v319);
                  v159 = sub_BCB2E0(v383);
                  v160 = sub_ACD640(v159, v297, 0);
                  sub_289A9E0((__int64)&v399, a1, v158, v157, v312, v160, v298, v319, v286, v294, (__int64)&v379);
                  if ( *(_BYTE *)a2 <= 0x1Cu )
                  {
LABEL_74:
                    v69 = 32 * (unsigned __int8)byte_5004008;
                  }
                  else
                  {
                    switch ( *(_BYTE *)a2 )
                    {
                      case ')':
                      case '+':
                      case '-':
                      case '/':
                      case '2':
                      case '5':
                      case 'J':
                      case 'K':
                      case 'S':
                        goto LABEL_182;
                      case 'T':
                      case 'U':
                      case 'V':
                        v64 = *(_QWORD *)(a2 + 8);
                        v65 = *(unsigned __int8 *)(v64 + 8);
                        v66 = v65 - 17;
                        v67 = *(_BYTE *)(v64 + 8);
                        if ( (unsigned int)(v65 - 17) <= 1 )
                          v67 = *(_BYTE *)(**(_QWORD **)(v64 + 16) + 8LL);
                        if ( v67 <= 3u || v67 == 5 || (v67 & 0xFD) == 4 )
                          goto LABEL_182;
                        if ( (_BYTE)v65 == 15 )
                        {
                          if ( (*(_BYTE *)(v64 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(a2 + 8)) )
                            goto LABEL_74;
                          v68 = *(__int64 **)(v64 + 16);
                          v64 = *v68;
                          v65 = *(unsigned __int8 *)(*v68 + 8);
                          v66 = v65 - 17;
                        }
                        else if ( (_BYTE)v65 == 16 )
                        {
                          do
                          {
                            v64 = *(_QWORD *)(v64 + 24);
                            LOBYTE(v65) = *(_BYTE *)(v64 + 8);
                          }
                          while ( (_BYTE)v65 == 16 );
                          v66 = (unsigned __int8)v65 - 17;
                        }
                        if ( v66 <= 1 )
                          LOBYTE(v65) = *(_BYTE *)(**(_QWORD **)(v64 + 16) + 8LL);
                        if ( (unsigned __int8)v65 > 3u && (_BYTE)v65 != 5 && (v65 & 0xFD) != 4 )
                          goto LABEL_74;
LABEL_182:
                        v69 = sub_B45210(a2);
                        if ( byte_5004008 )
                          v69 |= 0x20u;
                        break;
                      default:
                        goto LABEL_74;
                    }
                  }
                  sub_2899430(a1, v388.m128i_i64, (__int64 *)&v393, v399.m128i_i64, (__int64)&v379, 1, 0, v69);
                  if ( (__int64 *)v399.m128i_i64[0] != &v400 )
                    _libc_free(v399.m128i_u64[0]);
                  if ( (_BYTE *)v393 != v395 )
                    _libc_free(v393);
                  v70 = qword_50042A8;
                  v324 += qword_50042A8;
                  if ( v306 <= v324 )
                    goto LABEL_80;
                }
                v137 = (unsigned __int8 *)v136(v384, 17u, v129, v135, 0, 0);
                goto LABEL_159;
              }
LABEL_84:
              sub_F94A20(&v379, v77);
            }
            else
            {
              v295 = v336;
              v310 = v335;
              v290 = v337;
              v313 = v334;
              v289 = v339;
              v305 = v338;
              v206 = *(__int64 **)(*(_QWORD *)(a2 + 8) + 24LL);
              v355 = __PAIR64__(qword_50042A8, v335);
              v354 = (_BYTE *)__PAIR64__(v338, v334);
              v316 = v206;
              v207 = *(_QWORD *)(a1 + 40);
              v356 = 0;
              v357 = 0;
              v358 = 0;
              v359 = 0;
              v360 = 0;
              v361 = 0;
              v362 = 0;
              v363 = 0;
              v364 = 0;
              v399.m128i_i64[0] = (__int64)&v400;
              v208 = *(_QWORD *)(a2 + 40);
              v209 = *(_QWORD *)(a1 + 48);
              v399.m128i_i64[1] = 0x1000000000LL;
              v411 = &v415;
              v407 = v207;
              v405 = 0;
              v406 = 0;
              v408 = 0;
              v409 = 1;
              v410 = 0;
              v412 = 8;
              v413 = 0;
              v414 = 1;
              v416 = 0;
              v417 = 0;
              v418 = 0;
              v419 = 0;
              v393 = (unsigned __int64)"continue";
              v396 = 259;
              v210 = sub_F36960(v208, (__int64 *)(a2 + 24), 0, v207, v209, 0, (void **)&v393, 0);
              sub_23D0AB0((__int64)&v367, a2, 0, 0, 0);
              v287 = sub_2A364A0(&v354, v208, v210, &v367, &v399, *(_QWORD *)(a1 + 48));
              v211 = *(_QWORD *)(a2 + 8);
              if ( (unsigned int)*(unsigned __int8 *)(v211 + 8) - 17 <= 1 )
                v211 = **(_QWORD **)(v211 + 16);
              v212 = sub_BCDA70((__int64 *)v211, qword_50042A8);
              v386 = 0u;
              v300 = (__int64 **)v212;
              v379.m128i_i64[0] = (__int64)v380;
              v387 = dword_5003CC8 == 0;
              v379.m128i_i64[1] = 0x1000000000LL;
              v213 = *(_QWORD *)(v363 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v213 == v363 + 48 )
              {
                v214 = 0;
              }
              else
              {
                if ( !v213 )
                  BUG();
                v214 = v213 - 24;
                if ( (unsigned int)*(unsigned __int8 *)(v213 - 24) - 30 >= 0xB )
                  v214 = 0;
              }
              sub_D5F1F0((__int64)&v367, v214);
              v348 = v350;
              v349 = 0x400000000LL;
              if ( (_DWORD)qword_50042A8 )
              {
                v332 = 0;
                do
                {
                  v390 = 2307;
                  v388.m128i_i64[0] = (__int64)"result.vec.";
                  v396 = 257;
                  v389[0] = v332;
                  v215 = sub_BD2DA0(80);
                  v216 = v215;
                  if ( v215 )
                  {
                    sub_B44260(v215, (__int64)v300, 55, 0x8000000u, 0, 0);
                    *(_DWORD *)(v216 + 72) = 2;
                    sub_BD6B50((unsigned __int8 *)v216, (const char **)&v393);
                    sub_BD2A10(v216, *(_DWORD *)(v216 + 72), 1);
                  }
                  if ( *(_BYTE *)v216 > 0x1Cu )
                  {
                    switch ( *(_BYTE *)v216 )
                    {
                      case ')':
                      case '+':
                      case '-':
                      case '/':
                      case '2':
                      case '5':
                      case 'J':
                      case 'K':
                      case 'S':
                        goto LABEL_268;
                      case 'T':
                      case 'U':
                      case 'V':
                        v217 = *(_QWORD *)(v216 + 8);
                        v218 = v217;
                        v219 = *(unsigned __int8 *)(v217 + 8);
                        if ( (unsigned int)(v219 - 17) <= 1 )
                          v218 = **(_QWORD **)(v217 + 16);
                        v220 = *(_BYTE *)(v218 + 8);
                        if ( v220 <= 3u || v220 == 5 || (v220 & 0xFD) == 4 )
                          goto LABEL_268;
                        if ( (_BYTE)v219 == 15 )
                        {
                          if ( (*(_BYTE *)(v217 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(v216 + 8)) )
                            break;
                          v217 = **(_QWORD **)(v217 + 16);
                        }
                        else if ( (_BYTE)v219 == 16 )
                        {
                          do
                            v217 = *(_QWORD *)(v217 + 24);
                          while ( *(_BYTE *)(v217 + 8) == 16 );
                        }
                        if ( (unsigned int)*(unsigned __int8 *)(v217 + 8) - 17 <= 1 )
                          v217 = **(_QWORD **)(v217 + 16);
                        v221 = *(_BYTE *)(v217 + 8);
                        if ( v221 <= 3u || v221 == 5 || (v221 & 0xFD) == 4 )
                        {
LABEL_268:
                          v222 = v375;
                          if ( v374 )
                            sub_B99FD0(v216, 3u, v374);
                          sub_B45150(v216, v222);
                        }
                        break;
                      default:
                        break;
                    }
                  }
                  (*(void (__fastcall **)(__int64, __int64, __m128i *, __int64, __int64))(*(_QWORD *)v373 + 16LL))(
                    v373,
                    v216,
                    &v388,
                    v371,
                    v372);
                  v223 = (__int64)v367;
                  v224 = &v367[16 * (unsigned int)v368];
                  if ( v367 != v224 )
                  {
                    do
                    {
                      v225 = *(_QWORD *)(v223 + 8);
                      v226 = *(_DWORD *)v223;
                      v223 += 16;
                      sub_B99FD0(v216, v226, v225);
                    }
                    while ( v224 != (_BYTE *)v223 );
                  }
                  v227 = sub_AA56F0(v357);
                  v228 = sub_AC9350(v300);
                  v231 = *(_DWORD *)(v216 + 4) & 0x7FFFFFF;
                  if ( v231 == *(_DWORD *)(v216 + 72) )
                  {
                    v285 = v228;
                    sub_B48D90(v216);
                    v228 = v285;
                    v231 = *(_DWORD *)(v216 + 4) & 0x7FFFFFF;
                  }
                  v232 = (v231 + 1) & 0x7FFFFFF;
                  v233 = v232 | *(_DWORD *)(v216 + 4) & 0xF8000000;
                  v234 = *(_QWORD *)(v216 - 8) + 32LL * (unsigned int)(v232 - 1);
                  *(_DWORD *)(v216 + 4) = v233;
                  if ( *(_QWORD *)v234 )
                  {
                    v235 = *(_QWORD *)(v234 + 8);
                    **(_QWORD **)(v234 + 16) = v235;
                    if ( v235 )
                      *(_QWORD *)(v235 + 16) = *(_QWORD *)(v234 + 16);
                  }
                  *(_QWORD *)v234 = v228;
                  if ( v228 )
                  {
                    v236 = *(_QWORD *)(v228 + 16);
                    *(_QWORD *)(v234 + 8) = v236;
                    if ( v236 )
                      *(_QWORD *)(v236 + 16) = v234 + 8;
                    *(_QWORD *)(v234 + 16) = v228 + 16;
                    *(_QWORD *)(v228 + 16) = v234;
                  }
                  *(_QWORD *)(*(_QWORD *)(v216 - 8)
                            + 32LL * *(unsigned int *)(v216 + 72)
                            + 8LL * ((*(_DWORD *)(v216 + 4) & 0x7FFFFFFu) - 1)) = v227;
                  v237 = v379.m128i_u32[2];
                  v238 = v379.m128i_u32[2] + 1LL;
                  if ( v238 > v379.m128i_u32[3] )
                  {
                    sub_C8D5F0((__int64)&v379, v380, v238, 8u, v229, v230);
                    v237 = v379.m128i_u32[2];
                  }
                  *(_QWORD *)(v379.m128i_i64[0] + 8 * v237) = v216;
                  v239 = (unsigned int)v349;
                  ++v379.m128i_i32[2];
                  v240 = (unsigned int)v349 + 1LL;
                  if ( v240 > HIDWORD(v349) )
                  {
                    sub_C8D5F0((__int64)&v348, v350, v240, 8u, v229, v230);
                    v239 = (unsigned int)v349;
                  }
                  ++v332;
                  *(_QWORD *)&v348[8 * v239] = v216;
                  LODWORD(v349) = v349 + 1;
                }
                while ( (unsigned int)qword_50042A8 > v332 );
              }
              v241 = *(_QWORD *)(v287 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v241 == v287 + 48 )
              {
                v242 = 0;
              }
              else
              {
                if ( !v241 )
                  BUG();
                v242 = v241 - 24;
                if ( (unsigned int)*(unsigned __int8 *)(v241 - 24) - 30 >= 0xB )
                  v242 = 0;
              }
              sub_D5F1F0((__int64)&v367, v242);
              sub_289B1E0(
                (__int64)&v388,
                a1,
                v284,
                0,
                0,
                v313,
                v310,
                v295,
                v356,
                v362,
                qword_50042A8,
                qword_50042A8,
                dword_5003CC8 == 0,
                v316,
                (__int64)&v367);
              sub_289B1E0(
                (__int64)&v393,
                a1,
                v283,
                0,
                0,
                v290,
                v305,
                v289,
                v362,
                v359,
                qword_50042A8,
                qword_50042A8,
                dword_5003CC8 == 0,
                v316,
                (__int64)&v367);
              v243 = sub_28956B0(a2);
              sub_2899430(a1, v379.m128i_i64, v388.m128i_i64, (__int64 *)&v393, (__int64)&v367, 1, 0, v243);
              v244 = *(_QWORD *)(v358 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v244 == v358 + 48 )
              {
                v245 = 0;
              }
              else
              {
                if ( !v244 )
                  BUG();
                v245 = v244 - 24;
                if ( (unsigned int)*(unsigned __int8 *)(v244 - 24) - 30 >= 0xB )
                  v245 = 0;
              }
              sub_D5F1F0((__int64)&v367, v245);
              _BitScanReverse64(&v246, 1LL << (*(_WORD *)(v281 + 2) >> 1));
              LOBYTE(v247) = 63 - (v246 ^ 0x3F);
              HIBYTE(v247) = 1;
              sub_289AF80(
                a1,
                &v379,
                *(_QWORD *)(v281 - 32),
                v247,
                *(_BYTE *)(v281 + 2) & 1,
                v313,
                v305,
                dword_5003CC8 == 0,
                v356,
                v359,
                v316,
                (__int64)&v367);
              while ( v379.m128i_i32[2] > v118 )
              {
                v255 = v364;
                v256 = *(_QWORD *)&v348[8 * v118];
                v257 = *(_QWORD *)(v379.m128i_i64[0] + 8LL * v118);
                v258 = *(_DWORD *)(v256 + 4) & 0x7FFFFFF;
                if ( v258 == *(_DWORD *)(v256 + 72) )
                {
                  sub_B48D90(*(_QWORD *)&v348[8 * v118]);
                  v258 = *(_DWORD *)(v256 + 4) & 0x7FFFFFF;
                }
                v250 = (v258 + 1) & 0x7FFFFFF;
                v251 = v250 | *(_DWORD *)(v256 + 4) & 0xF8000000;
                v252 = *(_QWORD *)(v256 - 8) + 32LL * (unsigned int)(v250 - 1);
                *(_DWORD *)(v256 + 4) = v251;
                if ( *(_QWORD *)v252 )
                {
                  v253 = *(_QWORD *)(v252 + 8);
                  **(_QWORD **)(v252 + 16) = v253;
                  if ( v253 )
                    *(_QWORD *)(v253 + 16) = *(_QWORD *)(v252 + 16);
                }
                *(_QWORD *)v252 = v257;
                if ( v257 )
                {
                  v254 = *(_QWORD *)(v257 + 16);
                  *(_QWORD *)(v252 + 8) = v254;
                  if ( v254 )
                    *(_QWORD *)(v254 + 16) = v252 + 8;
                  *(_QWORD *)(v252 + 16) = v257 + 16;
                  *(_QWORD *)(v257 + 16) = v252;
                }
                ++v118;
                *(_QWORD *)(*(_QWORD *)(v256 - 8)
                          + 32LL * *(unsigned int *)(v256 + 72)
                          + 8LL * ((*(_DWORD *)(v256 + 4) & 0x7FFFFFFu) - 1)) = v255;
              }
              v259 = v310 / (unsigned int)qword_50042A8;
              if ( v310 / (unsigned int)qword_50042A8 > 9 )
                v259 = 10;
              v260 = *(_QWORD *)(a1 + 48);
              v261 = *(unsigned int *)(v260 + 24);
              v262 = *(_QWORD *)(v260 + 8);
              if ( (_DWORD)v261 )
              {
                v261 = (unsigned int)(v261 - 1);
                v249 = 1;
                for ( j = (unsigned int)v261 & (((unsigned int)v363 >> 9) ^ ((unsigned int)v363 >> 4));
                      ;
                      j = (unsigned int)v261 & v276 )
                {
                  v263 = (_QWORD *)(v262 + 16LL * (unsigned int)j);
                  if ( v363 == *v263 )
                    break;
                  if ( *v263 == -4096 )
                    goto LABEL_314;
                  v276 = v249 + j;
                  v249 = (unsigned int)(v249 + 1);
                }
                v261 = v263[1];
                v317 = v261;
              }
LABEL_314:
              sub_F6DC70(v317, "llvm.loop.unroll.count", v259, v261, j, v249);
              if ( (_BYTE *)v393 != v395 )
                _libc_free(v393);
              if ( (_DWORD *)v388.m128i_i64[0] != v389 )
                _libc_free(v388.m128i_u64[0]);
              if ( v348 != v350 )
                _libc_free((unsigned __int64)v348);
              if ( (_BYTE *)v379.m128i_i64[0] != v380 )
                _libc_free(v379.m128i_u64[0]);
              nullsub_61();
              v376 = &unk_49DA100;
              nullsub_63();
              if ( v367 != v369 )
                _libc_free((unsigned __int64)v367);
              sub_FFCE90((__int64)&v399, (__int64)"llvm.loop.unroll.count", v264, v265, v266, v267);
              sub_FFD870((__int64)&v399, (__int64)"llvm.loop.unroll.count", v268, v269, v270, v271);
              sub_FFBC40((__int64)&v399, (__int64)"llvm.loop.unroll.count");
              v272 = v418;
              for ( k = v417; v272 != k; k += 9 )
              {
                v274 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))k[7];
                *k = &unk_49E5048;
                if ( v274 )
                  v274(k + 5, k + 5, 3);
                *k = &unk_49DB368;
                v275 = k[3];
                if ( v275 != -4096 && v275 != 0 && v275 != -8192 )
                  sub_BD60C0(k + 1);
              }
              if ( v417 )
                j_j___libc_free_0((unsigned __int64)v417);
              if ( !v414 )
                _libc_free((unsigned __int64)v411);
              if ( (__int64 *)v399.m128i_i64[0] != &v400 )
                _libc_free(v399.m128i_u64[0]);
            }
            sub_BED950((__int64)&v399, a3, v281);
            sub_BED950((__int64)&v399, a3, a2);
            sub_28957D0(a1, (_QWORD *)v281);
            sub_28957D0(a1, (_QWORD *)a2);
            if ( (unsigned __int8)sub_BD3610(v302, 0) )
            {
              sub_BED950((__int64)&v399, a3, v302);
              sub_28957D0(a1, (_QWORD *)v302);
            }
            if ( v291 != v302 && (unsigned __int8)sub_BD3610(v291, 0) )
            {
              sub_BED950((__int64)&v399, a3, v291);
              sub_28957D0(a1, (_QWORD *)v291);
            }
          }
        }
LABEL_90:
        if ( v351 != (unsigned int *)v353 )
          _libc_free((unsigned __int64)v351);
        if ( v344 != &v346 )
          _libc_free((unsigned __int64)v344);
        sub_C7D6A0(v341, 8LL * (unsigned int)v343, 8);
      }
    }
  }
}
