// Function: sub_CC2400
// Address: 0xcc2400
//
__int64 __fastcall sub_CC2400(unsigned int *a1, int *a2, unsigned __int8 a3, __int64 a4, unsigned __int8 a5)
{
  int v7; // r8d
  int v8; // r9d
  int v9; // r12d
  unsigned int *v10; // r14
  unsigned int v11; // r13d
  unsigned int v12; // ecx
  unsigned int v13; // esi
  int v14; // r10d
  int v15; // edi
  int v16; // r11d
  int v17; // edi
  int v18; // r15d
  int v19; // r13d
  int v20; // r11d
  int v21; // edi
  int v22; // r13d
  unsigned int *v23; // r15
  int v24; // esi
  int v25; // r10d
  int v26; // r9d
  int v27; // edx
  int v28; // r12d
  int v29; // esi
  int v30; // ecx
  int v31; // r8d
  int v32; // r13d
  int v33; // r9d
  int v34; // r11d
  int v35; // edx
  int v36; // ebx
  int v37; // r12d
  int v38; // r8d
  int v39; // ecx
  int v40; // r13d
  int v41; // r13d
  int v42; // r12d
  int v43; // ebx
  int v44; // esi
  int v45; // r11d
  int v46; // r13d
  int v47; // r12d
  int v48; // r10d
  int v49; // esi
  int v50; // edi
  int v51; // r13d
  int v52; // ecx
  int v53; // r10d
  int v54; // edi
  int v55; // r9d
  int v56; // ecx
  int v57; // r8d
  int v58; // ebx
  int v59; // r9d
  int v60; // ebx
  int v61; // edx
  int v62; // r10d
  int v63; // r13d
  int v64; // r8d
  int v65; // edx
  unsigned int v66; // r13d
  int v67; // r13d
  int v68; // r11d
  int v69; // esi
  int v70; // edi
  int v71; // r13d
  int v72; // r11d
  int v73; // edi
  int v74; // r10d
  int v75; // r9d
  int v76; // r8d
  int v77; // edx
  int v78; // r12d
  int v79; // esi
  int v80; // r13d
  int v81; // ecx
  int v82; // r9d
  int v83; // esi
  int v84; // ebx
  int v85; // edx
  int v86; // r8d
  int v87; // r12d
  int v88; // r13d
  int v89; // ecx
  int v90; // r11d
  int v91; // r10d
  unsigned int v92; // ebx
  int v93; // r13d
  int v94; // edi
  int v95; // r12d
  int v96; // ebx
  int v97; // esi
  int v98; // ecx
  int v99; // r11d
  int v100; // r10d
  int v101; // r13d
  int v102; // edi
  int v103; // r12d
  int v104; // esi
  int v105; // ecx
  int v106; // r9d
  int v107; // r8d
  int v108; // edx
  int v109; // ebx
  int v110; // r13d
  int v111; // r9d
  int v112; // r8d
  int v113; // edx
  int v114; // ebx
  int v115; // r13d
  int v116; // r11d
  int v117; // edi
  int v118; // r13d
  int v119; // r11d
  int v120; // edi
  int v121; // r10d
  int v122; // r9d
  int v123; // edx
  int v124; // r12d
  int v125; // esi
  int v126; // r10d
  int v127; // ecx
  int v128; // r9d
  int v129; // esi
  int v130; // r8d
  int v131; // r11d
  int v132; // edx
  int v133; // r12d
  int v134; // r13d
  int v135; // ecx
  int v136; // ebx
  int v137; // r8d
  int v138; // r13d
  int v139; // r13d
  int v140; // r12d
  int v141; // ebx
  int v142; // esi
  int v143; // r11d
  int v144; // r13d
  int v145; // r12d
  int v146; // esi
  int v147; // r10d
  int v148; // r9d
  int v149; // edi
  int v150; // r8d
  int v151; // ecx
  int v152; // ebx
  int v153; // r10d
  int v154; // r9d
  int v155; // edi
  int v156; // ecx
  int v157; // ebx
  int v158; // edx
  int v159; // r10d
  int v160; // r13d
  int v161; // r8d
  int v162; // edx
  unsigned int v163; // r13d
  int v164; // r13d
  int v165; // r11d
  int v166; // esi
  int v167; // edi
  int v168; // r13d
  int v169; // r11d
  int v170; // edi
  int v171; // r10d
  int v172; // r9d
  int v173; // r8d
  int v174; // edx
  int v175; // r12d
  int v176; // esi
  int v177; // r13d
  int v178; // ecx
  int v179; // r9d
  int v180; // esi
  int v181; // ebx
  int v182; // edx
  int v183; // r8d
  int v184; // r12d
  int v185; // r13d
  int v186; // ecx
  int v187; // r11d
  int v188; // r10d
  unsigned int v189; // ebx
  int v190; // r13d
  int v191; // edi
  int v192; // r12d
  int v193; // ebx
  int v194; // esi
  int v195; // ecx
  int v196; // r11d
  int v197; // r10d
  int v198; // r13d
  int v199; // edi
  int v200; // r12d
  int v201; // esi
  int v202; // ecx
  int v203; // r9d
  int v204; // r8d
  int v205; // edx
  int v206; // ebx
  int v207; // r13d
  int v208; // r9d
  int v209; // r8d
  int v210; // edx
  int v211; // ebx
  int v212; // r13d
  int v213; // r11d
  int v214; // edi
  int v215; // r10d
  int v216; // r9d
  int v217; // r13d
  int v218; // r11d
  int v219; // edx
  int v220; // edi
  int v221; // r12d
  int v222; // esi
  int v223; // ecx
  int v224; // r10d
  int v225; // r9d
  int v226; // esi
  int v227; // r8d
  int v228; // r11d
  int v229; // edx
  int v230; // r12d
  int v231; // r13d
  int v232; // ecx
  int v233; // ebx
  int v234; // r8d
  int v235; // r13d
  int v236; // r13d
  int v237; // r12d
  int v238; // ebx
  int v239; // esi
  int v240; // r11d
  int v241; // r13d
  int v242; // r12d
  int v243; // esi
  int v244; // r10d
  int v245; // r9d
  int v246; // edi
  int v247; // ecx
  int v248; // ebx
  int v249; // r10d
  int v250; // edi
  int v251; // r9d
  int v252; // ecx
  int v253; // r8d
  int v254; // ebx
  int v255; // edx
  int v256; // r10d
  int v257; // r13d
  int v258; // r8d
  int v259; // edx
  int v260; // r13d
  int v261; // r11d
  int v262; // edi
  int v263; // r13d
  int v264; // r11d
  int v265; // edi
  int v266; // r13d
  int v267; // esi
  int v268; // r10d
  int v269; // r9d
  int v270; // r8d
  int v271; // edx
  int v272; // r13d
  int v273; // r12d
  int v274; // ecx
  int v275; // r9d
  int v276; // esi
  int v277; // r12d
  int v278; // ebx
  int v279; // ecx
  int v280; // r8d
  int v281; // r13d
  int v282; // edx
  int v283; // r11d
  int v284; // r10d
  int v285; // r13d
  int v286; // edi
  int v287; // r9d
  int v288; // r12d
  int v289; // ecx
  int v290; // r11d
  int v291; // r10d
  int v292; // edi
  int v293; // r13d
  int v294; // ecx
  int v295; // r12d
  int v296; // r8d
  int v297; // edx
  int v298; // r9d
  int v299; // esi
  int v300; // r12d
  int v301; // ebx
  int v302; // r8d
  int v303; // edx
  int v304; // esi
  int v305; // ebx
  int v306; // r11d
  int v307; // edi
  int v308; // r12d
  int v309; // ebx
  int v310; // r11d
  int v311; // r10d
  int v312; // r9d
  int v313; // edi
  int v314; // esi
  int v315; // r13d
  int v316; // ecx
  int v317; // r10d
  int v318; // r9d
  int v319; // esi
  int v320; // r13d
  int v321; // ecx
  int v322; // r8d
  int v323; // r11d
  int v324; // r10d
  int v325; // ebx
  int v326; // r12d
  int v327; // edi
  int v328; // edx
  int v329; // r8d
  int v330; // ebx
  int v331; // r12d
  int v332; // ebx
  int v333; // edx
  int v334; // r12d
  int v335; // r13d
  int v336; // edx
  int v337; // r11d
  int v338; // ebx
  int v339; // r13d
  int v340; // ecx
  int v341; // r8d
  int v342; // r9d
  int v343; // esi
  int v344; // r10d
  int v345; // edi
  int v346; // ebx
  int v347; // edx
  int v348; // r12d
  int v349; // r8d
  int v350; // r9d
  int v351; // esi
  __int64 result; // rax
  unsigned int *v353; // [rsp+0h] [rbp-88h]
  int v354; // [rsp+Ch] [rbp-7Ch]
  int v355; // [rsp+10h] [rbp-78h]
  int v356; // [rsp+10h] [rbp-78h]
  int v357; // [rsp+10h] [rbp-78h]
  int v358; // [rsp+10h] [rbp-78h]
  int v359; // [rsp+10h] [rbp-78h]
  int v360; // [rsp+10h] [rbp-78h]
  int v361; // [rsp+14h] [rbp-74h]
  int v362; // [rsp+14h] [rbp-74h]
  int v363; // [rsp+14h] [rbp-74h]
  int v364; // [rsp+14h] [rbp-74h]
  int v365; // [rsp+14h] [rbp-74h]
  int v366; // [rsp+14h] [rbp-74h]
  int v367; // [rsp+18h] [rbp-70h]
  int v368; // [rsp+18h] [rbp-70h]
  int v369; // [rsp+18h] [rbp-70h]
  int v370; // [rsp+18h] [rbp-70h]
  int v371; // [rsp+18h] [rbp-70h]
  int v372; // [rsp+18h] [rbp-70h]
  unsigned int v373; // [rsp+1Ch] [rbp-6Ch]
  int v374; // [rsp+1Ch] [rbp-6Ch]
  int v375; // [rsp+1Ch] [rbp-6Ch]
  int v376; // [rsp+1Ch] [rbp-6Ch]
  int v377; // [rsp+1Ch] [rbp-6Ch]
  int v378; // [rsp+1Ch] [rbp-6Ch]
  int v379; // [rsp+1Ch] [rbp-6Ch]
  int v380; // [rsp+20h] [rbp-68h]
  int v381; // [rsp+20h] [rbp-68h]
  int v382; // [rsp+24h] [rbp-64h]
  int v383; // [rsp+28h] [rbp-60h]
  int v384; // [rsp+2Ch] [rbp-5Ch]
  int v385; // [rsp+30h] [rbp-58h]
  int v386; // [rsp+30h] [rbp-58h]
  int v387; // [rsp+34h] [rbp-54h]
  int v388; // [rsp+38h] [rbp-50h]
  int v389; // [rsp+3Ch] [rbp-4Ch]
  int v390; // [rsp+40h] [rbp-48h]
  int v391; // [rsp+44h] [rbp-44h]
  int v392; // [rsp+48h] [rbp-40h]
  int v393; // [rsp+4Ch] [rbp-3Ch]
  int v394; // [rsp+50h] [rbp-38h]
  int v395; // [rsp+54h] [rbp-34h]
  int v396; // [rsp+58h] [rbp-30h]
  int v397; // [rsp+58h] [rbp-30h]
  int v398; // [rsp+5Ch] [rbp-2Ch]
  int v399; // [rsp+5Ch] [rbp-2Ch]

  v393 = a2[4];
  v7 = a2[1];
  v392 = a2[5];
  v8 = a2[2];
  v391 = a2[6];
  v9 = a2[3];
  v390 = a2[7];
  v389 = a2[8];
  v387 = a2[10];
  v10 = a1;
  v385 = a2[11];
  v384 = a2[12];
  v383 = a2[13];
  v388 = a2[9];
  v11 = a1[4];
  v398 = *a2;
  v382 = a2[14];
  v12 = a1[6];
  v380 = a2[15];
  v13 = a1[5];
  v395 = v8;
  v394 = v9;
  v373 = a1[7];
  v14 = v8 + v13 + a1[1];
  v396 = v7;
  v15 = v398 + v11 + *a1;
  v16 = v7 + v15;
  v17 = __ROL4__(a4 ^ v15, 16);
  v18 = v17 + 1779033703;
  LODWORD(a4) = __ROL4__(v14 ^ HIDWORD(a4), 16);
  v19 = __ROR4__((v17 + 1779033703) ^ v11, 12);
  v20 = v19 + v16;
  v21 = __ROR4__(v20 ^ v17, 8);
  v367 = v21 + v18;
  v22 = (v21 + v18) ^ v19;
  v23 = v10;
  LODWORD(v10) = a4 - 1150833019;
  v353 = v23;
  v24 = __ROR4__((a4 - 1150833019) ^ v13, 12);
  v355 = __ROR4__(v22, 7);
  v25 = v24 + v9 + v14;
  v26 = v393 + v12 + v23[2];
  LODWORD(a4) = __ROR4__(v25 ^ a4, 8);
  v27 = __ROL4__(v26 ^ a3, 16);
  LODWORD(v10) = a4 + (_DWORD)v10;
  v28 = v27 + 1013904242;
  v29 = __ROR4__((unsigned int)v10 ^ v24, 7);
  v30 = __ROR4__((v27 + 1013904242) ^ v12, 12);
  v31 = v391 + v23[3] + v373;
  v32 = __ROL4__(v31 ^ a5, 16);
  v33 = v30 + v392 + v26;
  LODWORD(v23) = v32 - 1521486534;
  v34 = v29 + v389 + v20;
  v35 = __ROR4__(v33 ^ v27, 8);
  v36 = __ROR4__((v32 - 1521486534) ^ v373, 12);
  v37 = v35 + v28;
  v38 = v36 + v390 + v31;
  v39 = __ROR4__(v37 ^ v30, 7);
  v40 = __ROR4__(v38 ^ v32, 8);
  LODWORD(v23) = v40 + (_DWORD)v23;
  v41 = __ROL4__(v34 ^ v40, 16);
  v42 = v41 + v37;
  v43 = __ROR4__((unsigned int)v23 ^ v36, 7);
  v44 = __ROR4__(v42 ^ v29, 12);
  v45 = v44 + v388 + v34;
  v46 = __ROR4__(v45 ^ v41, 8);
  v47 = v46 + v42;
  v374 = v46;
  v48 = v39 + v387 + v25;
  v49 = __ROR4__(v47 ^ v44, 7);
  v50 = __ROL4__(v48 ^ v21, 16);
  v51 = (_DWORD)v23 + v50;
  v52 = __ROR4__(((_DWORD)v23 + v50) ^ v39, 12);
  v53 = v52 + v385 + v48;
  v54 = __ROR4__(v53 ^ v50, 8);
  v361 = v51 + v54;
  v55 = v43 + v384 + v33;
  v56 = __ROR4__((v51 + v54) ^ v52, 7);
  LODWORD(a4) = __ROL4__(v55 ^ a4, 16);
  v57 = v355 + v382 + v38;
  LODWORD(v23) = a4 + v367;
  v58 = __ROR4__((a4 + v367) ^ v43, 12);
  v59 = v58 + v383 + v55;
  LODWORD(a4) = __ROR4__(v59 ^ a4, 8);
  LODWORD(v23) = a4 + (_DWORD)v23;
  v60 = __ROR4__((unsigned int)v23 ^ v58, 7);
  v61 = __ROL4__(v57 ^ v35, 16);
  LODWORD(v10) = v61 + (_DWORD)v10;
  v62 = v49 + v394 + v53;
  v63 = __ROR4__((unsigned int)v10 ^ v355, 12);
  v64 = v63 + v380 + v57;
  LODWORD(a4) = __ROL4__(v62 ^ a4, 16);
  v65 = __ROR4__(v64 ^ v61, 8);
  LODWORD(v10) = v65 + (_DWORD)v10;
  v66 = (unsigned int)v10 ^ v63;
  LODWORD(v10) = a4 + (_DWORD)v10;
  v67 = __ROR4__(v66, 7);
  v68 = v67 + v395 + v45;
  v69 = __ROR4__((unsigned int)v10 ^ v49, 12);
  v70 = __ROL4__(v68 ^ v54, 16);
  LODWORD(v23) = v70 + (_DWORD)v23;
  v71 = __ROR4__((unsigned int)v23 ^ v67, 12);
  v72 = v71 + v391 + v68;
  v73 = __ROR4__(v72 ^ v70, 8);
  LODWORD(v23) = v73 + (_DWORD)v23;
  v74 = v69 + v387 + v62;
  v75 = v56 + v390 + v59;
  v356 = __ROR4__((unsigned int)v23 ^ v71, 7);
  v76 = v60 + v393 + v64;
  LODWORD(a4) = __ROR4__(v74 ^ a4, 8);
  v77 = __ROL4__(v75 ^ v65, 16);
  v78 = v77 + v47;
  v368 = a4 + (_DWORD)v10;
  v79 = (a4 + (_DWORD)v10) ^ v69;
  v80 = __ROL4__(v76 ^ v374, 16);
  v81 = __ROR4__(v78 ^ v56, 12);
  LODWORD(v10) = v80 + v361;
  v82 = v81 + v398 + v75;
  v83 = __ROR4__(v79, 7);
  v84 = __ROR4__((v80 + v361) ^ v60, 12);
  v85 = __ROR4__(v82 ^ v77, 8);
  v86 = v84 + v383 + v76;
  v87 = v85 + v78;
  v88 = __ROR4__(v86 ^ v80, 8);
  v89 = __ROR4__(v87 ^ v81, 7);
  LODWORD(v10) = v88 + (_DWORD)v10;
  v90 = v83 + v396 + v72;
  v91 = v89 + v384 + v74;
  v92 = (unsigned int)v10 ^ v84;
  v93 = __ROL4__(v90 ^ v88, 16);
  v94 = __ROL4__(v91 ^ v73, 16);
  v95 = v93 + v87;
  LODWORD(v10) = v94 + (_DWORD)v10;
  v96 = __ROR4__(v92, 7);
  v97 = __ROR4__(v95 ^ v83, 12);
  v98 = __ROR4__((unsigned int)v10 ^ v89, 12);
  v99 = v97 + v385 + v90;
  v100 = v98 + v392 + v91;
  v101 = __ROR4__(v99 ^ v93, 8);
  v102 = __ROR4__(v100 ^ v94, 8);
  v103 = v101 + v95;
  v375 = v101;
  v362 = v102 + (_DWORD)v10;
  v104 = __ROR4__(v103 ^ v97, 7);
  v105 = __ROR4__((v102 + (_DWORD)v10) ^ v98, 7);
  v106 = v96 + v388 + v82;
  v107 = v356 + v380 + v86;
  LODWORD(a4) = __ROL4__(v106 ^ a4, 16);
  v108 = __ROL4__(v107 ^ v85, 16);
  LODWORD(v23) = a4 + (_DWORD)v23;
  LODWORD(v10) = v108 + v368;
  v109 = __ROR4__((unsigned int)v23 ^ v96, 12);
  v110 = __ROR4__((v108 + v368) ^ v356, 12);
  v111 = v109 + v382 + v106;
  v112 = v110 + v389 + v107;
  LODWORD(a4) = __ROR4__(v111 ^ a4, 8);
  v113 = __ROR4__(v112 ^ v108, 8);
  LODWORD(v23) = a4 + (_DWORD)v23;
  LODWORD(v10) = v113 + (_DWORD)v10;
  v114 = __ROR4__((unsigned int)v23 ^ v109, 7);
  v115 = __ROR4__((unsigned int)v10 ^ v110, 7);
  v116 = v115 + v394 + v99;
  v117 = __ROL4__(v116 ^ v102, 16);
  LODWORD(v23) = v117 + (_DWORD)v23;
  v118 = __ROR4__((unsigned int)v23 ^ v115, 12);
  v119 = v118 + v393 + v116;
  v120 = __ROR4__(v119 ^ v117, 8);
  LODWORD(v23) = v120 + (_DWORD)v23;
  v357 = __ROR4__((unsigned int)v23 ^ v118, 7);
  v121 = v104 + v387 + v100;
  v122 = v105 + v383 + v111;
  LODWORD(a4) = __ROL4__(v121 ^ a4, 16);
  LODWORD(v10) = a4 + (_DWORD)v10;
  v123 = __ROL4__(v122 ^ v113, 16);
  v124 = v123 + v103;
  v125 = __ROR4__((unsigned int)v10 ^ v104, 12);
  v126 = v125 + v384 + v121;
  v127 = __ROR4__(v124 ^ v105, 12);
  v128 = v127 + v395 + v122;
  LODWORD(a4) = __ROR4__(v126 ^ a4, 8);
  v369 = a4 + (_DWORD)v10;
  v129 = __ROR4__((a4 + (_DWORD)v10) ^ v125, 7);
  v130 = v114 + v390 + v112;
  v131 = v129 + v391 + v119;
  v132 = __ROR4__(v128 ^ v123, 8);
  v133 = v132 + v124;
  v134 = __ROL4__(v130 ^ v375, 16);
  LODWORD(v10) = v134 + v362;
  v135 = __ROR4__(v133 ^ v127, 7);
  v136 = __ROR4__((v134 + v362) ^ v114, 12);
  v137 = v136 + v382 + v130;
  v138 = __ROR4__(v137 ^ v134, 8);
  LODWORD(v10) = v138 + (_DWORD)v10;
  v139 = __ROL4__(v131 ^ v138, 16);
  v140 = v139 + v133;
  v141 = __ROR4__((unsigned int)v10 ^ v136, 7);
  v142 = __ROR4__(v140 ^ v129, 12);
  v143 = v142 + v392 + v131;
  v144 = __ROR4__(v143 ^ v139, 8);
  v145 = v144 + v140;
  v376 = v144;
  v146 = __ROR4__(v145 ^ v142, 7);
  v147 = v135 + v388 + v126;
  v148 = v141 + v385 + v128;
  v149 = __ROL4__(v147 ^ v120, 16);
  LODWORD(a4) = __ROL4__(v148 ^ a4, 16);
  v150 = v357 + v389 + v137;
  LODWORD(v10) = v149 + (_DWORD)v10;
  LODWORD(v23) = a4 + (_DWORD)v23;
  v151 = __ROR4__((unsigned int)v10 ^ v135, 12);
  v152 = __ROR4__((unsigned int)v23 ^ v141, 12);
  v153 = v151 + v398 + v147;
  v154 = v152 + v380 + v148;
  v155 = __ROR4__(v153 ^ v149, 8);
  LODWORD(a4) = __ROR4__(v154 ^ a4, 8);
  LODWORD(v23) = a4 + (_DWORD)v23;
  v363 = v155 + (_DWORD)v10;
  v156 = __ROR4__((v155 + (_DWORD)v10) ^ v151, 7);
  v157 = __ROR4__((unsigned int)v23 ^ v152, 7);
  v158 = __ROL4__(v150 ^ v132, 16);
  LODWORD(v10) = v158 + v369;
  v159 = v146 + v384 + v153;
  v160 = __ROR4__((v158 + v369) ^ v357, 12);
  v161 = v160 + v396 + v150;
  LODWORD(a4) = __ROL4__(v159 ^ a4, 16);
  v162 = __ROR4__(v161 ^ v158, 8);
  LODWORD(v10) = v162 + (_DWORD)v10;
  v163 = (unsigned int)v10 ^ v160;
  LODWORD(v10) = a4 + (_DWORD)v10;
  v164 = __ROR4__(v163, 7);
  v165 = v164 + v387 + v143;
  v166 = __ROR4__((unsigned int)v10 ^ v146, 12);
  v167 = __ROL4__(v165 ^ v155, 16);
  LODWORD(v23) = v167 + (_DWORD)v23;
  v168 = __ROR4__((unsigned int)v23 ^ v164, 12);
  v169 = v168 + v390 + v165;
  v170 = __ROR4__(v169 ^ v167, 8);
  LODWORD(v23) = v170 + (_DWORD)v23;
  v171 = v166 + v388 + v159;
  v172 = v156 + v382 + v154;
  v358 = __ROR4__((unsigned int)v23 ^ v168, 7);
  v173 = v157 + v383 + v161;
  LODWORD(a4) = __ROR4__(v171 ^ a4, 8);
  v174 = __ROL4__(v172 ^ v162, 16);
  v175 = v174 + v145;
  v370 = a4 + (_DWORD)v10;
  v176 = (a4 + (_DWORD)v10) ^ v166;
  v177 = __ROL4__(v173 ^ v376, 16);
  v178 = __ROR4__(v175 ^ v156, 12);
  LODWORD(v10) = v177 + v363;
  v179 = v178 + v394 + v172;
  v180 = __ROR4__(v176, 7);
  v181 = __ROR4__((v177 + v363) ^ v157, 12);
  v182 = __ROR4__(v179 ^ v174, 8);
  v183 = v181 + v380 + v173;
  v184 = v182 + v175;
  v185 = __ROR4__(v183 ^ v177, 8);
  v186 = __ROR4__(v184 ^ v178, 7);
  LODWORD(v10) = v185 + (_DWORD)v10;
  v187 = v180 + v393 + v169;
  v188 = v186 + v385 + v171;
  v189 = (unsigned int)v10 ^ v181;
  v190 = __ROL4__(v187 ^ v185, 16);
  v191 = __ROL4__(v188 ^ v170, 16);
  v192 = v190 + v184;
  LODWORD(v10) = v191 + (_DWORD)v10;
  v193 = __ROR4__(v189, 7);
  v194 = __ROR4__(v192 ^ v180, 12);
  v195 = __ROR4__((unsigned int)v10 ^ v186, 12);
  v196 = v194 + v398 + v187;
  v197 = v195 + v395 + v188;
  v198 = __ROR4__(v196 ^ v190, 8);
  v199 = __ROR4__(v197 ^ v191, 8);
  v200 = v198 + v192;
  v377 = v198;
  v364 = v199 + (_DWORD)v10;
  v201 = __ROR4__(v200 ^ v194, 7);
  v202 = __ROR4__((v199 + (_DWORD)v10) ^ v195, 7);
  v203 = v193 + v392 + v179;
  v204 = v358 + v396 + v183;
  LODWORD(a4) = __ROL4__(v203 ^ a4, 16);
  v205 = __ROL4__(v204 ^ v182, 16);
  LODWORD(v23) = a4 + (_DWORD)v23;
  LODWORD(v10) = v205 + v370;
  v206 = __ROR4__((unsigned int)v23 ^ v193, 12);
  v207 = __ROR4__((v205 + v370) ^ v358, 12);
  v208 = v206 + v389 + v203;
  v209 = v207 + v391 + v204;
  LODWORD(a4) = __ROR4__(v208 ^ a4, 8);
  v210 = __ROR4__(v209 ^ v205, 8);
  LODWORD(v23) = a4 + (_DWORD)v23;
  LODWORD(v10) = v210 + (_DWORD)v10;
  v211 = __ROR4__((unsigned int)v23 ^ v206, 7);
  v212 = __ROR4__((unsigned int)v10 ^ v207, 7);
  v213 = v212 + v384 + v196;
  v214 = __ROL4__(v213 ^ v199, 16);
  LODWORD(v23) = v214 + (_DWORD)v23;
  v215 = v201 + v388 + v197;
  v216 = v202 + v380 + v208;
  v217 = __ROR4__((unsigned int)v23 ^ v212, 12);
  v218 = v217 + v383 + v213;
  LODWORD(a4) = __ROL4__(v215 ^ a4, 16);
  v219 = __ROL4__(v216 ^ v210, 16);
  LODWORD(v10) = a4 + (_DWORD)v10;
  v220 = __ROR4__(v218 ^ v214, 8);
  v221 = v219 + v200;
  LODWORD(v23) = v220 + (_DWORD)v23;
  v222 = __ROR4__((unsigned int)v10 ^ v201, 12);
  v223 = __ROR4__(v221 ^ v202, 12);
  v224 = v222 + v385 + v215;
  v225 = v223 + v387 + v216;
  v359 = __ROR4__((unsigned int)v23 ^ v217, 7);
  LODWORD(a4) = __ROR4__(v224 ^ a4, 8);
  v371 = a4 + (_DWORD)v10;
  v226 = __ROR4__((a4 + (_DWORD)v10) ^ v222, 7);
  v227 = v211 + v382 + v209;
  v228 = v226 + v390 + v218;
  v229 = __ROR4__(v225 ^ v219, 8);
  v230 = v229 + v221;
  v231 = __ROL4__(v227 ^ v377, 16);
  LODWORD(v10) = v231 + v364;
  v232 = __ROR4__(v230 ^ v223, 7);
  v233 = __ROR4__((v231 + v364) ^ v211, 12);
  v234 = v233 + v389 + v227;
  v235 = __ROR4__(v234 ^ v231, 8);
  LODWORD(v10) = v235 + (_DWORD)v10;
  v236 = __ROL4__(v228 ^ v235, 16);
  v237 = v236 + v230;
  v238 = __ROR4__((unsigned int)v10 ^ v233, 7);
  v239 = __ROR4__(v237 ^ v226, 12);
  v240 = v239 + v395 + v228;
  v241 = __ROR4__(v240 ^ v236, 8);
  v242 = v241 + v237;
  v378 = v241;
  v243 = __ROR4__(v242 ^ v239, 7);
  v244 = v232 + v392 + v224;
  v245 = v238 + v398 + v225;
  v246 = __ROL4__(v244 ^ v220, 16);
  LODWORD(a4) = __ROL4__(v245 ^ a4, 16);
  LODWORD(v10) = v246 + (_DWORD)v10;
  LODWORD(v23) = a4 + (_DWORD)v23;
  v247 = __ROR4__((unsigned int)v10 ^ v232, 12);
  v248 = __ROR4__((unsigned int)v23 ^ v238, 12);
  v249 = v247 + v394 + v244;
  v250 = __ROR4__(v249 ^ v246, 8);
  v365 = v250 + (_DWORD)v10;
  v251 = v248 + v396 + v245;
  v252 = __ROR4__((v250 + (_DWORD)v10) ^ v247, 7);
  LODWORD(a4) = __ROR4__(v251 ^ a4, 8);
  v253 = v359 + v391 + v234;
  LODWORD(v23) = a4 + (_DWORD)v23;
  v254 = __ROR4__((unsigned int)v23 ^ v248, 7);
  v255 = __ROL4__(v253 ^ v229, 16);
  LODWORD(v10) = v255 + v371;
  v256 = v243 + v385 + v249;
  v257 = __ROR4__((v255 + v371) ^ v359, 12);
  v258 = v257 + v393 + v253;
  LODWORD(a4) = __ROL4__(v256 ^ a4, 16);
  v259 = __ROR4__(v258 ^ v255, 8);
  LODWORD(v10) = v259 + (_DWORD)v10;
  v260 = __ROR4__((unsigned int)v10 ^ v257, 7);
  v261 = v260 + v388 + v240;
  v262 = __ROL4__(v261 ^ v250, 16);
  LODWORD(v23) = v262 + (_DWORD)v23;
  v263 = __ROR4__((unsigned int)v23 ^ v260, 12);
  v264 = v263 + v382 + v261;
  v265 = __ROR4__(v264 ^ v262, 8);
  v372 = v265 + (_DWORD)v23;
  v266 = (v265 + (_DWORD)v23) ^ v263;
  LODWORD(v23) = (_DWORD)v10 + a4;
  v267 = __ROR4__(((_DWORD)v10 + a4) ^ v243, 12);
  v354 = __ROR4__(v266, 7);
  v268 = v267 + v392 + v256;
  v269 = v252 + v389 + v251;
  v270 = v254 + v380 + v258;
  LODWORD(a4) = __ROR4__(v268 ^ a4, 8);
  v271 = __ROL4__(v269 ^ v259, 16);
  v272 = __ROL4__(v270 ^ v378, 16);
  v273 = v271 + v242;
  v360 = (_DWORD)v23 + a4;
  v274 = __ROR4__(v273 ^ v252, 12);
  LODWORD(v10) = __ROR4__(((_DWORD)v23 + a4) ^ v267, 7);
  v275 = v274 + v384 + v269;
  v276 = __ROR4__(v275 ^ v271, 8);
  LODWORD(v23) = v365 + v272;
  v277 = v276 + v273;
  v278 = __ROR4__((v365 + v272) ^ v254, 12);
  v279 = __ROR4__(v277 ^ v274, 7);
  v280 = v278 + v396 + v270;
  v281 = __ROR4__(v280 ^ v272, 8);
  LODWORD(v23) = v281 + (_DWORD)v23;
  v282 = __ROR4__((unsigned int)v23 ^ v278, 7);
  v283 = (_DWORD)v10 + v383 + v264;
  v284 = v279 + v398 + v268;
  v285 = __ROL4__(v283 ^ v281, 16);
  v286 = __ROL4__(v284 ^ v265, 16);
  v287 = v282 + v395 + v275;
  v288 = v285 + v277;
  LODWORD(v23) = v286 + (_DWORD)v23;
  LODWORD(a4) = __ROL4__(v287 ^ a4, 16);
  LODWORD(v10) = __ROR4__(v288 ^ (unsigned int)v10, 12);
  v289 = __ROR4__((unsigned int)v23 ^ v279, 12);
  v290 = (_DWORD)v10 + v394 + v283;
  v291 = v289 + v387 + v284;
  v292 = __ROR4__(v291 ^ v286, 8);
  v366 = __ROR4__(v290 ^ v285, 8);
  v293 = v288 + v366;
  v379 = (_DWORD)v23 + v292;
  LODWORD(v10) = __ROR4__((v288 + v366) ^ (unsigned int)v10, 7);
  v294 = __ROR4__(((_DWORD)v23 + v292) ^ v289, 7);
  v295 = a4 + v372;
  v296 = v354 + v393 + v280;
  v297 = __ROR4__((a4 + v372) ^ v282, 12);
  v298 = v297 + v391 + v287;
  v299 = __ROL4__(v296 ^ v276, 16);
  LODWORD(v23) = v299 + v360;
  LODWORD(a4) = __ROR4__(v298 ^ a4, 8);
  v300 = a4 + v295;
  v301 = __ROR4__((v299 + v360) ^ v354, 12);
  v302 = v301 + v390 + v296;
  v303 = __ROR4__(v300 ^ v297, 7);
  v304 = __ROR4__(v302 ^ v299, 8);
  LODWORD(v23) = v304 + (_DWORD)v23;
  v305 = __ROR4__((unsigned int)v23 ^ v301, 7);
  v306 = v305 + v385 + v290;
  v307 = __ROL4__(v306 ^ v292, 16);
  v308 = v307 + v300;
  v309 = __ROR4__(v308 ^ v305, 12);
  v310 = v309 + v380 + v306;
  v311 = (_DWORD)v10 + v392 + v291;
  v312 = v294 + v396 + v298;
  v313 = __ROR4__(v310 ^ v307, 8);
  LODWORD(a4) = __ROL4__(v311 ^ a4, 16);
  v314 = __ROL4__(v312 ^ v304, 16);
  LODWORD(v23) = a4 + (_DWORD)v23;
  v315 = v314 + v293;
  v386 = v313 + v308;
  LODWORD(v10) = __ROR4__((unsigned int)v23 ^ (unsigned int)v10, 12);
  v316 = __ROR4__(v315 ^ v294, 12);
  v381 = __ROR4__((v313 + v308) ^ v309, 7);
  v317 = (_DWORD)v10 + v398 + v311;
  v318 = v316 + v388 + v312;
  LODWORD(a4) = __ROR4__(v317 ^ a4, 8);
  v319 = __ROR4__(v318 ^ v314, 8);
  LODWORD(v23) = a4 + (_DWORD)v23;
  v320 = v319 + v315;
  LODWORD(v10) = __ROR4__((unsigned int)v23 ^ (unsigned int)v10, 7);
  v321 = __ROR4__(v320 ^ v316, 7);
  v322 = v303 + v389 + v302;
  v323 = (_DWORD)v10 + v382 + v310;
  v324 = v321 + v395 + v317;
  v325 = __ROL4__(v322 ^ v366, 16);
  v326 = v325 + v379;
  v327 = __ROL4__(v324 ^ v313, 16);
  v328 = __ROR4__((v325 + v379) ^ v303, 12);
  v329 = v328 + v391 + v322;
  v330 = __ROR4__(v329 ^ v325, 8);
  v331 = v330 + v326;
  v332 = __ROL4__(v323 ^ v330, 16);
  v333 = v331 ^ v328;
  v334 = v327 + v331;
  v335 = v332 + v320;
  v336 = __ROR4__(v333, 7);
  LODWORD(v10) = __ROR4__(v335 ^ (unsigned int)v10, 12);
  v337 = (_DWORD)v10 + v387 + v323;
  v399 = (int)v10;
  v338 = __ROR4__(v337 ^ v332, 8);
  v339 = v338 + v335;
  v397 = v338;
  v340 = __ROR4__(v334 ^ v321, 12);
  v341 = v381 + v390 + v329;
  v342 = v336 + v394 + v318;
  v343 = __ROL4__(v341 ^ v319, 16);
  LODWORD(a4) = __ROL4__(v342 ^ a4, 16);
  v344 = v340 + v384 + v324;
  LODWORD(v23) = v343 + (_DWORD)v23;
  LODWORD(v10) = a4 + v386;
  v345 = __ROR4__(v344 ^ v327, 8);
  v346 = __ROR4__((unsigned int)v23 ^ v381, 12);
  v347 = __ROR4__((a4 + v386) ^ v336, 12);
  v348 = v345 + v334;
  v349 = v346 + v383 + v341;
  v350 = v347 + v393 + v342;
  v351 = __ROR4__(v349 ^ v343, 8);
  LODWORD(a4) = __ROR4__(v350 ^ a4, 8);
  LODWORD(v10) = a4 + (_DWORD)v10;
  *v353 = (unsigned int)v10 ^ v337;
  v353[1] = (v351 + (_DWORD)v23) ^ v344;
  v353[2] = v339 ^ v350;
  v353[3] = v348 ^ v349;
  v353[4] = __ROR4__(v346 ^ (v351 + (_DWORD)v23), 7) ^ v345;
  v353[6] = __ROR4__(v348 ^ v340, 7) ^ v351;
  result = __ROR4__(v399 ^ v339, 7) ^ (unsigned int)a4;
  v353[7] = v397 ^ __ROR4__((unsigned int)v10 ^ v347, 7);
  v353[5] = result;
  return result;
}
