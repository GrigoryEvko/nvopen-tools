// Function: sub_EE4CB0
// Address: 0xee4cb0
//
__int64 __fastcall sub_EE4CB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbp
  __int64 v7; // r12
  __int64 v9; // r13
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rsi
  __int64 result; // rax
  __int64 v17; // r14
  unsigned __int8 v18; // bl
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // r15
  unsigned __int64 *v24; // rbx
  unsigned __int64 *v25; // r14
  __int64 v26; // r13
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // r15
  __int64 v34; // r14
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  unsigned __int8 v39; // bl
  __int64 v40; // r13
  __int64 v41; // rsi
  __int64 v42; // r14
  __int64 v43; // rsi
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  unsigned __int8 *v48; // rcx
  __int64 v49; // rdx
  unsigned __int8 v50; // si
  unsigned __int8 *v51; // r15
  __int64 v52; // rbx
  __int64 v53; // r14
  unsigned __int8 *v54; // r13
  unsigned __int64 v55; // rdx
  __int64 v56; // rcx
  unsigned __int8 v57; // si
  __int64 v58; // r8
  unsigned __int8 v59; // bl
  __int64 v60; // r8
  __int64 v61; // r13
  __int64 v62; // r14
  unsigned __int8 v63; // bl
  unsigned __int8 *v64; // r15
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // r9
  __int64 v77; // r15
  __int64 v78; // r13
  unsigned __int64 *v79; // r14
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 v86; // r9
  __int64 v87; // r8
  char v88; // al
  __int64 v89; // r15
  unsigned __int8 *v90; // r13
  __int64 v91; // r14
  char v92; // bl
  __int64 v93; // rdi
  char v94; // bl
  __int64 v95; // rsi
  __int64 v96; // rdx
  __int64 v97; // rcx
  __int64 v98; // r8
  __int64 v99; // r9
  __int64 v100; // rdx
  __int64 v101; // rcx
  __int64 v102; // r8
  __int64 v103; // r9
  char v104; // al
  __int64 v105; // r15
  __int64 v106; // r13
  unsigned __int64 *v107; // r14
  char v108; // bl
  __int64 v109; // rdx
  __int64 v110; // rcx
  __int64 v111; // r8
  __int64 v112; // r9
  __int64 v113; // rcx
  __int64 v114; // r8
  __int64 v115; // r9
  __int64 v116; // rdx
  __int64 v117; // rcx
  __int64 v118; // r8
  __int64 v119; // r9
  __int64 v120; // r15
  unsigned __int8 v121; // r13
  unsigned __int8 v122; // r14
  char v123; // bl
  __int64 v124; // rdx
  __int64 v125; // rcx
  __int64 v126; // r8
  __int64 v127; // r9
  __int64 v128; // rdx
  __int64 v129; // rcx
  __int64 v130; // r8
  __int64 v131; // r9
  __int64 v132; // rdx
  __int64 v133; // rcx
  __int64 v134; // r8
  __int64 v135; // r9
  char v136; // al
  __int64 v137; // r13
  __int64 v138; // r14
  unsigned __int64 *v139; // r15
  char v140; // bl
  __int64 v141; // rdx
  __int64 v142; // rcx
  __int64 v143; // r8
  __int64 v144; // r9
  __int64 v145; // rcx
  __int64 v146; // r8
  __int64 v147; // r9
  __int64 v148; // rdx
  __int64 v149; // rcx
  __int64 v150; // r8
  __int64 v151; // r9
  __int64 v152; // rsi
  __int64 v153; // rdx
  __int64 v154; // rcx
  __int64 v155; // r8
  __int64 v156; // r9
  char v157; // al
  __int64 v158; // r8
  __int64 v159; // r13
  __int64 v160; // r14
  unsigned __int8 *v161; // r15
  char v162; // bl
  __int64 v163; // rdx
  __int64 v164; // rcx
  __int64 v165; // r8
  __int64 v166; // r9
  __int64 v167; // r8
  __int64 v168; // r9
  unsigned __int8 v169; // bl
  unsigned __int8 *v170; // r15
  unsigned __int64 *v171; // r14
  __int64 v172; // r13
  __int64 v173; // rdx
  __int64 v174; // rcx
  __int64 v175; // r8
  __int64 v176; // r9
  __int64 v177; // rdx
  __int64 v178; // rcx
  __int64 v179; // r8
  __int64 v180; // rcx
  __int64 v181; // r8
  __int64 v182; // r9
  __int64 v183; // rdx
  __int64 v184; // rcx
  __int64 v185; // r8
  __int64 v186; // r9
  __int64 v187; // r15
  __int64 v188; // r13
  __int64 v189; // r14
  char v190; // bl
  char v191; // bl
  __int64 v192; // rsi
  __int64 v193; // rdx
  __int64 v194; // rcx
  __int64 v195; // r8
  __int64 v196; // r9
  __int64 v197; // rdx
  __int64 v198; // rcx
  __int64 v199; // r8
  __int64 v200; // r9
  char v201; // al
  __int64 v202; // rcx
  unsigned __int64 *v203; // rdx
  unsigned __int8 v204; // si
  __int64 v205; // r8
  __int64 v206; // r9
  __int64 v207; // r15
  __int64 v208; // rbx
  __int64 v209; // r14
  unsigned __int8 *v210; // r13
  __int64 v211; // rcx
  __int64 v212; // r8
  __int64 v213; // r9
  __int64 v214; // rdx
  __int64 v215; // rcx
  __int64 v216; // r8
  __int64 v217; // rcx
  __int64 v218; // r9
  __int64 v219; // rdx
  __int64 v220; // rcx
  __int64 v221; // r8
  __int64 v222; // r9
  __int64 v223; // r14
  unsigned __int8 v224; // r13
  int v225; // ebx
  __int64 v226; // rdx
  __int64 v227; // rcx
  __int64 v228; // r8
  __int64 v229; // r9
  __int64 v230; // rdx
  __int64 v231; // rcx
  __int64 v232; // r8
  __int64 v233; // r9
  int v234; // ebx
  int v235; // ebx
  unsigned __int64 v236; // rcx
  unsigned __int64 v237; // rdx
  unsigned __int8 v238; // si
  __int64 v239; // r13
  __int64 v240; // r14
  __int64 v241; // rsi
  unsigned __int64 *v242; // r15
  __int64 v243; // rdi
  __int64 v244; // rcx
  __int64 v245; // r8
  __int64 v246; // r9
  __int64 v247; // rdx
  __int64 v248; // rcx
  __int64 v249; // r8
  __int64 v250; // r9
  __int64 v251; // rbx
  int v252; // r13d
  unsigned int v253; // ebx
  __int64 v254; // rdx
  __int64 v255; // rcx
  __int64 v256; // r8
  __int64 v257; // r9
  __int64 v258; // r13
  int v259; // ebx
  __int64 v260; // rdx
  __int64 v261; // rcx
  __int64 v262; // r8
  __int64 v263; // r9
  unsigned __int8 v264; // si
  unsigned __int64 v265; // r8
  unsigned __int8 *v266; // rcx
  __int64 v267; // rdx
  __int64 v268; // r13
  unsigned int v269; // ebx
  __int64 v270; // rdx
  __int64 v271; // rcx
  __int64 v272; // r8
  __int64 v273; // r9
  __int64 v274; // r13
  __int64 v275; // rbx
  unsigned __int8 *v276; // r14
  __int64 v277; // r15
  __int64 v278; // rdx
  __int64 v279; // rcx
  __int64 v280; // r8
  __int64 v281; // r9
  __int64 v282; // rdx
  __int64 v283; // rcx
  __int64 v284; // r8
  __int64 v285; // r9
  __int64 v286; // [rsp-78h] [rbp-78h]
  __int64 v287; // [rsp-68h] [rbp-68h]
  __int64 v288; // [rsp-60h] [rbp-60h]
  __int64 v289; // [rsp-60h] [rbp-60h]
  __int64 v290; // [rsp-58h] [rbp-58h]
  unsigned __int64 *v291; // [rsp-58h] [rbp-58h]
  __int64 v292; // [rsp-50h] [rbp-50h]
  __int64 v293; // [rsp-50h] [rbp-50h]
  unsigned __int8 v294; // [rsp-50h] [rbp-50h]
  __int64 v295; // [rsp-50h] [rbp-50h]
  __int64 v296; // [rsp-50h] [rbp-50h]
  unsigned __int64 *v297; // [rsp-50h] [rbp-50h]
  __int64 v298[8]; // [rsp-40h] [rbp-40h] BYREF

  v298[7] = v6;
  v298[3] = v7;
  switch ( *(_BYTE *)(a1 + 8) )
  {
    case 0:
      v202 = *(_QWORD *)(a1 + 24);
      v203 = *(unsigned __int64 **)(a1 + 16);
      v204 = 0;
      return sub_EE4780(a2, v204, v203, v202, a5, a6);
    case 1:
      v55 = *(_QWORD *)(a1 + 16);
      v56 = *(_QWORD *)(a1 + 24);
      v57 = 1;
      v58 = *(_QWORD *)(a1 + 32);
      return sub_EE3E30(a2, v57, v55, v56, v58, a6);
    case 2:
      v274 = *(_QWORD *)(a1 + 40);
      v275 = *(_QWORD *)(a1 + 16);
      v298[0] = a2;
      v276 = *(unsigned __int8 **)(a1 + 32);
      v277 = *(_QWORD *)(a1 + 24);
      sub_D953B0(a2, 2, a3, a4, a5, a6);
      sub_D953B0(a2, v275, v278, v279, v280, v281);
      sub_EE3670(v298, v277, v276);
      return sub_D953B0(v298[0], v274, v282, v283, v284, v285);
    case 3:
      v268 = *(_QWORD *)(a1 + 16);
      v269 = *(_DWORD *)(a1 + 12);
      sub_D953B0(a2, 3, a3, a4, a5, a6);
      sub_D953B0(a2, v268, v270, v271, v272, v273);
      v15 = v269;
      goto LABEL_4;
    case 4:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 4;
      goto LABEL_3;
    case 5:
      v55 = *(_QWORD *)(a1 + 16);
      v56 = *(_QWORD *)(a1 + 24);
      v57 = 5;
      v58 = *(_QWORD *)(a1 + 32);
      return sub_EE3E30(a2, v57, v55, v56, v58, a6);
    case 6:
      v264 = 6;
      v265 = *(_QWORD *)(a1 + 32);
      v266 = *(unsigned __int8 **)(a1 + 24);
      v267 = *(_QWORD *)(a1 + 16);
      return sub_EE3F80(a2, v264, v267, v266, v265, a6);
    case 7:
      v264 = 7;
      v265 = *(_QWORD *)(a1 + 32);
      v266 = *(unsigned __int8 **)(a1 + 24);
      v267 = *(_QWORD *)(a1 + 16);
      return sub_EE3F80(a2, v264, v267, v266, v265, a6);
    case 8:
      v48 = *(unsigned __int8 **)(a1 + 24);
      v49 = *(_QWORD *)(a1 + 16);
      v50 = 8;
      return sub_EE3C10(a2, v50, v49, v48, a5, a6);
    case 9:
      return sub_EE3E30(a2, 9u, *(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 32), a6);
    case 0xA:
      v202 = *(_QWORD *)(a1 + 24);
      v203 = *(unsigned __int64 **)(a1 + 16);
      v204 = 10;
      return sub_EE4780(a2, v204, v203, v202, a5, a6);
    case 0xB:
      v55 = *(_QWORD *)(a1 + 16);
      v56 = *(_QWORD *)(a1 + 24);
      v57 = 11;
      v58 = *(_QWORD *)(a1 + 32);
      return sub_EE3E30(a2, v57, v55, v56, v58, a6);
    case 0xC:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 12;
      goto LABEL_3;
    case 0xD:
      v258 = *(_QWORD *)(a1 + 16);
      v259 = *(_DWORD *)(a1 + 24);
      sub_D953B0(a2, 13, a3, a4, a5, a6);
      sub_D953B0(a2, v258, v260, v261, v262, v263);
      v15 = v259;
      goto LABEL_4;
    case 0xE:
      v236 = *(_QWORD *)(a1 + 24);
      v237 = *(_QWORD *)(a1 + 16);
      v238 = 14;
      return sub_EE40D0(a2, v238, v237, v236, a5, a6);
    case 0xF:
      return sub_EE40D0(a2, 0xFu, *(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 24), a5, a6);
    case 0x10:
      return sub_EE45C0(
               a2,
               *(_QWORD *)(a1 + 16),
               *(_QWORD *)(a1 + 24),
               *(_QWORD *)(a1 + 32),
               *(unsigned int *)(a1 + 40),
               *(unsigned __int8 *)(a1 + 44),
               *(_QWORD *)(a1 + 48));
    case 0x11:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 17;
      goto LABEL_3;
    case 0x12:
      v202 = *(_QWORD *)(a1 + 24);
      v203 = *(unsigned __int64 **)(a1 + 16);
      v204 = 18;
      return sub_EE4780(a2, v204, v203, v202, a5, a6);
    case 0x13:
      return sub_EE48D0(
               a2,
               *(_QWORD *)(a1 + 16),
               *(_QWORD *)(a1 + 24),
               *(_QWORD *)(a1 + 32),
               *(_QWORD *)(a1 + 40),
               *(_QWORD *)(a1 + 48),
               *(_QWORD *)(a1 + 56),
               *(_DWORD *)(a1 + 64),
               *(_BYTE *)(a1 + 68));
    case 0x14:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 20;
      goto LABEL_3;
    case 0x15:
      return sub_EE3F80(a2, 0x15u, *(_QWORD *)(a1 + 16), *(unsigned __int8 **)(a1 + 24), *(_QWORD *)(a1 + 32), a6);
    case 0x16:
      v236 = *(_QWORD *)(a1 + 24);
      v237 = *(_QWORD *)(a1 + 16);
      v238 = 22;
      return sub_EE40D0(a2, v238, v237, v236, a5, a6);
    case 0x17:
      v236 = *(_QWORD *)(a1 + 24);
      v237 = *(_QWORD *)(a1 + 16);
      v238 = 23;
      return sub_EE40D0(a2, v238, v237, v236, a5, a6);
    case 0x18:
      v236 = *(_QWORD *)(a1 + 24);
      v237 = *(_QWORD *)(a1 + 16);
      v238 = 24;
      return sub_EE40D0(a2, v238, v237, v236, a5, a6);
    case 0x19:
      v236 = *(_QWORD *)(a1 + 24);
      v237 = *(_QWORD *)(a1 + 16);
      v238 = 25;
      return sub_EE40D0(a2, v238, v237, v236, a5, a6);
    case 0x1A:
      v236 = *(_QWORD *)(a1 + 24);
      v237 = *(_QWORD *)(a1 + 16);
      v238 = 26;
      return sub_EE40D0(a2, v238, v237, v236, a5, a6);
    case 0x1B:
      v39 = *(_BYTE *)(a1 + 32);
      v40 = *(_QWORD *)(a1 + 24);
      v41 = 27;
      v42 = *(_QWORD *)(a1 + 16);
      goto LABEL_11;
    case 0x1C:
      return sub_EE40D0(a2, 0x1Cu, *(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 24), a5, a6);
    case 0x1D:
      v236 = *(_QWORD *)(a1 + 24);
      v237 = *(_QWORD *)(a1 + 16);
      v238 = 29;
      return sub_EE40D0(a2, v238, v237, v236, a5, a6);
    case 0x1E:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 30;
      goto LABEL_3;
    case 0x1F:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 31;
      goto LABEL_3;
    case 0x20:
      v40 = *(_QWORD *)(a1 + 16);
      v39 = *(_BYTE *)(a1 + 24);
      v43 = 32;
      goto LABEL_12;
    case 0x21:
      v252 = *(_DWORD *)(a1 + 12);
      v253 = *(_DWORD *)(a1 + 16);
      sub_D953B0(a2, 33, a3, a4, a5, a6);
      sub_D953B0(a2, v252, v254, v255, v256, v257);
      v15 = v253;
      goto LABEL_4;
    case 0x22:
      v236 = *(_QWORD *)(a1 + 24);
      v237 = *(_QWORD *)(a1 + 16);
      v238 = 34;
      return sub_EE40D0(a2, v238, v237, v236, a5, a6);
    case 0x23:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 35;
      goto LABEL_3;
    case 0x24:
      v236 = *(_QWORD *)(a1 + 24);
      v237 = *(_QWORD *)(a1 + 16);
      v238 = 36;
      return sub_EE40D0(a2, v238, v237, v236, a5, a6);
    case 0x25:
      v236 = *(_QWORD *)(a1 + 24);
      v237 = *(_QWORD *)(a1 + 16);
      v238 = 37;
      return sub_EE40D0(a2, v238, v237, v236, a5, a6);
    case 0x26:
      v251 = *(_QWORD *)(a1 + 16);
      v239 = *(_QWORD *)(a1 + 40);
      v298[0] = a2;
      v240 = *(_QWORD *)(a1 + 32);
      v242 = *(unsigned __int64 **)(a1 + 24);
      sub_D953B0(a2, 38, a3, a4, a5, a6);
      v243 = a2;
      v241 = v251;
      goto LABEL_57;
    case 0x27:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 39;
      goto LABEL_3;
    case 0x28:
      v202 = *(_QWORD *)(a1 + 24);
      v203 = *(unsigned __int64 **)(a1 + 16);
      v204 = 40;
      return sub_EE4780(a2, v204, v203, v202, a5, a6);
    case 0x29:
      v202 = *(_QWORD *)(a1 + 24);
      v203 = *(unsigned __int64 **)(a1 + 16);
      v204 = 41;
      return sub_EE4780(a2, v204, v203, v202, a5, a6);
    case 0x2A:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 42;
      goto LABEL_3;
    case 0x2B:
      v239 = *(_QWORD *)(a1 + 32);
      v240 = *(_QWORD *)(a1 + 24);
      v298[0] = a2;
      v241 = 43;
      v242 = *(unsigned __int64 **)(a1 + 16);
      v243 = a2;
LABEL_57:
      sub_D953B0(v243, v241, a3, a4, a5, a6);
      sub_EE3CE0(v298, v242, v240, v244, v245, v246);
      return sub_D953B0(v298[0], v239, v247, v248, v249, v250);
    case 0x2C:
      BUG();
    case 0x2D:
      v236 = *(_QWORD *)(a1 + 24);
      v237 = *(_QWORD *)(a1 + 16);
      v238 = 45;
      return sub_EE40D0(a2, v238, v237, v236, a5, a6);
    case 0x2E:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 46;
      goto LABEL_3;
    case 0x2F:
      v235 = *(_DWORD *)(a1 + 12);
      sub_D953B0(a2, 47, a3, a4, a5, a6);
      v15 = v235;
      goto LABEL_4;
    case 0x30:
      v234 = *(_DWORD *)(a1 + 12);
      sub_D953B0(a2, 48, a3, a4, a5, a6);
      v15 = v234;
      goto LABEL_4;
    case 0x31:
      v223 = *(_QWORD *)(a1 + 16);
      v224 = *(_BYTE *)(a1 + 24);
      v225 = *(_DWORD *)(a1 + 28);
      sub_D953B0(a2, 49, a3, a4, a5, a6);
      sub_D953B0(a2, v223, v226, v227, v228, v229);
      sub_D953B0(a2, v224, v230, v231, v232, v233);
      v15 = v225;
      goto LABEL_4;
    case 0x32:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 50;
      goto LABEL_3;
    case 0x33:
      v48 = *(unsigned __int8 **)(a1 + 24);
      v49 = *(_QWORD *)(a1 + 16);
      v50 = 51;
      return sub_EE3C10(a2, v50, v49, v48, a5, a6);
    case 0x34:
      v205 = *(_QWORD *)(a1 + 40);
      v206 = *(_QWORD *)(a1 + 32);
      v298[0] = a2;
      v207 = *(_QWORD *)(a1 + 56);
      v208 = *(_QWORD *)(a1 + 48);
      v291 = (unsigned __int64 *)v205;
      v209 = *(_QWORD *)(a1 + 64);
      v210 = *(unsigned __int8 **)(a1 + 72);
      v287 = v206;
      v289 = *(_QWORD *)(a1 + 24);
      v297 = *(unsigned __int64 **)(a1 + 16);
      sub_D953B0(a2, 52, v289, a4, v205, v206);
      sub_EE3CE0(v298, v297, v289, v211, v212, v213);
      sub_D953B0(v298[0], v287, v214, v215, v216, v287);
      sub_EE3CE0(v298, v291, v208, v217, (__int64)v291, v218);
      sub_D953B0(v298[0], v207, v219, v220, v221, v222);
      return sub_EE3670(v298, v209, v210);
    case 0x35:
      v202 = *(_QWORD *)(a1 + 24);
      v203 = *(unsigned __int64 **)(a1 + 16);
      v204 = 53;
      return sub_EE4780(a2, v204, v203, v202, a5, a6);
    case 0x36:
      return sub_EE4280(
               a2,
               0x36u,
               *(_QWORD *)(a1 + 16),
               *(_QWORD *)(a1 + 24),
               *(unsigned __int8 **)(a1 + 32),
               *(_QWORD *)(a1 + 40),
               (char)(4 * *(_BYTE *)(a1 + 9)) >> 2);
    case 0x37:
      v188 = *(_QWORD *)(a1 + 24);
      v192 = 55;
      v189 = *(_QWORD *)(a1 + 16);
      v191 = (char)(4 * *(_BYTE *)(a1 + 9)) >> 2;
      goto LABEL_41;
    case 0x38:
      v201 = *(_BYTE *)(a1 + 9);
      v89 = *(_QWORD *)(a1 + 16);
      v298[0] = a2;
      v95 = 56;
      v90 = *(unsigned __int8 **)(a1 + 32);
      v91 = *(_QWORD *)(a1 + 24);
      v93 = a2;
      v94 = (char)(4 * v201) >> 2;
      goto LABEL_27;
    case 0x39:
      v187 = *(_QWORD *)(a1 + 16);
      v188 = *(_QWORD *)(a1 + 32);
      v189 = *(_QWORD *)(a1 + 24);
      v190 = 4 * *(_BYTE *)(a1 + 9);
      sub_D953B0(a2, 57, a3, a4, a5, a6);
      v191 = v190 >> 2;
      v192 = v187;
LABEL_41:
      sub_D953B0(a2, v192, a3, a4, a5, a6);
      sub_D953B0(a2, v189, v193, v194, v195, v196);
      sub_D953B0(a2, v188, v197, v198, v199, v200);
      v15 = v191;
      goto LABEL_4;
    case 0x3A:
      sub_EE4280(
        a2,
        0x3Au,
        *(_QWORD *)(a1 + 16),
        *(_QWORD *)(a1 + 24),
        *(unsigned __int8 **)(a1 + 32),
        *(_QWORD *)(a1 + 40),
        (char)(4 * *(_BYTE *)(a1 + 9)) >> 2);
      return v286;
    case 0x3B:
      v167 = *(_QWORD *)(a1 + 32);
      v168 = *(_QWORD *)(a1 + 24);
      v298[0] = a2;
      v169 = *(_BYTE *)(a1 + 64);
      v170 = *(unsigned __int8 **)(a1 + 40);
      v171 = *(unsigned __int64 **)(a1 + 48);
      v296 = v167;
      v172 = *(_QWORD *)(a1 + 56);
      v288 = v168;
      v290 = *(_QWORD *)(a1 + 16);
      sub_D953B0(a2, 59, a3, a4, v167, v168);
      sub_D953B0(a2, v290, v173, v174, v175, v176);
      sub_D953B0(a2, v288, v177, v178, v179, v288);
      sub_EE3670(v298, v296, v170);
      sub_EE3CE0(v298, v171, v172, v180, v181, v182);
      return sub_D953B0(v298[0], v169, v183, v184, v185, v186);
    case 0x3C:
      return sub_EE4440(
               a2,
               0x3Cu,
               *(_QWORD *)(a1 + 16),
               *(unsigned __int8 **)(a1 + 24),
               *(_QWORD *)(a1 + 32),
               (unsigned int)((char)(4 * *(_BYTE *)(a1 + 9)) >> 2));
    case 0x3D:
      v157 = *(_BYTE *)(a1 + 9);
      v158 = *(_QWORD *)(a1 + 16);
      v298[0] = a2;
      v159 = *(_QWORD *)(a1 + 40);
      v160 = *(_QWORD *)(a1 + 32);
      v161 = *(unsigned __int8 **)(a1 + 24);
      v162 = 4 * v157;
      v295 = v158;
      sub_D953B0(a2, 61, a3, a4, v158, a6);
      v140 = v162 >> 2;
      sub_EE3670(v298, v295, v161);
      sub_D953B0(v298[0], v160, v163, v164, v165, v166);
      v152 = v159;
      goto LABEL_34;
    case 0x3E:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 62;
      goto LABEL_3;
    case 0x3F:
      v136 = *(_BYTE *)(a1 + 9);
      v137 = *(_QWORD *)(a1 + 16);
      v298[0] = a2;
      v138 = *(_QWORD *)(a1 + 32);
      v139 = *(unsigned __int64 **)(a1 + 24);
      v140 = (char)(4 * v136) >> 2;
      v294 = *(_BYTE *)(a1 + 40);
      sub_D953B0(a2, 63, a3, a4, a5, a6);
      sub_D953B0(a2, v137, v141, v142, v143, v144);
      sub_EE3CE0(v298, v139, v138, v145, v146, v147);
      v152 = v294;
LABEL_34:
      sub_D953B0(v298[0], v152, v148, v149, v150, v151);
      return sub_D953B0(v298[0], v140, v153, v154, v155, v156);
    case 0x40:
      return sub_EE4AA0(
               a2,
               *(unsigned __int64 **)(a1 + 16),
               *(_QWORD *)(a1 + 24),
               *(_QWORD *)(a1 + 32),
               *(_QWORD *)(a1 + 40),
               *(_QWORD *)(a1 + 48),
               *(_BYTE *)(a1 + 56),
               *(_BYTE *)(a1 + 57),
               (char)(4 * *(_BYTE *)(a1 + 9)) >> 2);
    case 0x41:
      v120 = *(_QWORD *)(a1 + 16);
      v121 = *(_BYTE *)(a1 + 25);
      v122 = *(_BYTE *)(a1 + 24);
      v123 = 4 * *(_BYTE *)(a1 + 9);
      sub_D953B0(a2, 65, a3, a4, a5, a6);
      sub_D953B0(a2, v120, v124, v125, v126, v127);
      sub_D953B0(a2, v122, v128, v129, v130, v131);
      sub_D953B0(a2, v121, v132, v133, v134, v135);
      v15 = v123 >> 2;
      goto LABEL_4;
    case 0x42:
      return sub_EE4440(
               a2,
               0x42u,
               *(_QWORD *)(a1 + 16),
               *(unsigned __int8 **)(a1 + 24),
               *(_QWORD *)(a1 + 32),
               (unsigned int)((char)(4 * *(_BYTE *)(a1 + 9)) >> 2));
    case 0x43:
      v48 = *(unsigned __int8 **)(a1 + 24);
      v49 = *(_QWORD *)(a1 + 16);
      v50 = 67;
      return sub_EE3C10(a2, v50, v49, v48, a5, a6);
    case 0x44:
      v104 = *(_BYTE *)(a1 + 9);
      v105 = *(_QWORD *)(a1 + 16);
      v298[0] = a2;
      v106 = *(_QWORD *)(a1 + 32);
      v107 = *(unsigned __int64 **)(a1 + 24);
      v108 = 4 * v104;
      sub_D953B0(a2, 68, a3, a4, a5, a6);
      sub_D953B0(a2, v105, v109, v110, v111, v112);
      sub_EE3CE0(v298, v107, v106, v113, v114, v115);
      return sub_D953B0(v298[0], v108 >> 2, v116, v117, v118, v119);
    case 0x45:
      v87 = *(_QWORD *)(a1 + 16);
      v88 = *(_BYTE *)(a1 + 9);
      v298[0] = a2;
      v89 = *(_QWORD *)(a1 + 24);
      v90 = *(unsigned __int8 **)(a1 + 40);
      v91 = *(_QWORD *)(a1 + 32);
      v293 = v87;
      v92 = 4 * v88;
      sub_D953B0(a2, 69, a3, a4, v87, a6);
      a5 = v293;
      v93 = a2;
      v94 = v92 >> 2;
      v95 = v293;
LABEL_27:
      sub_D953B0(v93, v95, a3, a4, a5, a6);
      sub_D953B0(v298[0], v89, v96, v97, v98, v99);
      sub_EE3670(v298, v91, v90);
      return sub_D953B0(v298[0], v94, v100, v101, v102, v103);
    case 0x46:
      v77 = *(_QWORD *)(a1 + 16);
      v78 = *(_QWORD *)(a1 + 32);
      v298[0] = a2;
      v79 = *(unsigned __int64 **)(a1 + 24);
      sub_D953B0(a2, 70, a3, a4, a5, a6);
      sub_D953B0(a2, v77, v80, v81, v82, v83);
      return sub_EE3CE0(v298, v79, v78, v84, v85, v86);
    case 0x47:
      v60 = *(_QWORD *)(a1 + 32);
      v61 = *(_QWORD *)(a1 + 24);
      v298[0] = a2;
      v62 = *(_QWORD *)(a1 + 16);
      v63 = *(_BYTE *)(a1 + 48);
      v64 = *(unsigned __int8 **)(a1 + 40);
      v292 = v60;
      sub_D953B0(a2, 71, a3, a4, v60, a6);
      sub_D953B0(a2, v63, v65, v66, v67, v68);
      sub_EE3670(v298, v292, v64);
      sub_D953B0(v298[0], v62, v69, v70, v71, v72);
      return sub_D953B0(v298[0], v61, v73, v74, v75, v76);
    case 0x48:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 72;
      goto LABEL_3;
    case 0x49:
      v59 = *(_BYTE *)(a1 + 11);
      sub_D953B0(a2, 73, a3, a4, a5, a6);
      v15 = v59;
      goto LABEL_4;
    case 0x4A:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 74;
      goto LABEL_3;
    case 0x4B:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 75;
      goto LABEL_3;
    case 0x4C:
      v55 = *(_QWORD *)(a1 + 16);
      v56 = *(_QWORD *)(a1 + 24);
      v57 = 76;
      v58 = *(_QWORD *)(a1 + 32);
      return sub_EE3E30(a2, v57, v55, v56, v58, a6);
    case 0x4D:
      v51 = *(unsigned __int8 **)(a1 + 24);
      v52 = *(_QWORD *)(a1 + 16);
      v298[0] = a2;
      v53 = *(_QWORD *)(a1 + 32);
      v54 = *(unsigned __int8 **)(a1 + 40);
      sub_D953B0(a2, 77, a3, a4, a5, a6);
      sub_EE3670(v298, v52, v51);
      return sub_EE3670(v298, v53, v54);
    case 0x4E:
      v48 = *(unsigned __int8 **)(a1 + 24);
      v49 = *(_QWORD *)(a1 + 16);
      v50 = 78;
      return sub_EE3C10(a2, v50, v49, v48, a5, a6);
    case 0x4F:
      v48 = *(unsigned __int8 **)(a1 + 24);
      v49 = *(_QWORD *)(a1 + 16);
      v50 = 79;
      return sub_EE3C10(a2, v50, v49, v48, a5, a6);
    case 0x50:
      v48 = *(unsigned __int8 **)(a1 + 24);
      v49 = *(_QWORD *)(a1 + 16);
      v50 = 80;
      return sub_EE3C10(a2, v50, v49, v48, a5, a6);
    case 0x51:
      v39 = *(_BYTE *)(a1 + 32);
      v40 = *(_QWORD *)(a1 + 24);
      v41 = 81;
      v42 = *(_QWORD *)(a1 + 16);
LABEL_11:
      sub_D953B0(a2, v41, a3, a4, a5, a6);
      v43 = v42;
LABEL_12:
      sub_D953B0(a2, v43, a3, a4, a5, a6);
      sub_D953B0(a2, v40, v44, v45, v46, v47);
      v15 = v39;
      goto LABEL_4;
    case 0x52:
      v33 = *(_QWORD *)(a1 + 16);
      v34 = *(_QWORD *)(a1 + 24);
      v9 = *(_QWORD *)(a1 + 32);
      sub_D953B0(a2, 82, a3, a4, a5, a6);
      sub_D953B0(a2, v33, v35, v36, v37, v38);
      v10 = v34;
      goto LABEL_3;
    case 0x53:
      v23 = *(_QWORD *)(a1 + 24);
      v24 = *(unsigned __int64 **)(a1 + 16);
      v298[0] = a2;
      v25 = *(unsigned __int64 **)(a1 + 32);
      v26 = *(_QWORD *)(a1 + 40);
      sub_D953B0(a2, 83, a3, a4, a5, a6);
      sub_EE3CE0(v298, v24, v23, v27, v28, v29);
      return sub_EE3CE0(v298, v25, v26, v30, v31, v32);
    case 0x54:
      v17 = *(_QWORD *)(a1 + 16);
      v18 = *(_BYTE *)(a1 + 24);
      v9 = *(_QWORD *)(a1 + 32);
      sub_D953B0(a2, 84, a3, a4, a5, a6);
      sub_D953B0(a2, v17, v19, v20, v21, v22);
      v10 = v18;
      goto LABEL_3;
    case 0x55:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 85;
      goto LABEL_3;
    case 0x56:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 86;
      goto LABEL_3;
    case 0x57:
      v9 = *(_QWORD *)(a1 + 16);
      v10 = 87;
LABEL_3:
      sub_D953B0(a2, v10, a3, a4, a5, a6);
      v15 = v9;
LABEL_4:
      result = sub_D953B0(a2, v15, v11, v12, v13, v14);
      break;
    default:
      return result;
  }
  return result;
}
