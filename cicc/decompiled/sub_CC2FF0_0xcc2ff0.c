// Function: sub_CC2FF0
// Address: 0xcc2ff0
//
__int64 __fastcall sub_CC2FF0(
        _DWORD *a1,
        int *a2,
        unsigned __int8 a3,
        __int64 a4,
        unsigned __int8 a5,
        unsigned int *a6)
{
  int v8; // r9d
  int v9; // r10d
  int v10; // r12d
  _DWORD *v11; // r14
  int v12; // r13d
  int v13; // ecx
  int v14; // esi
  int v15; // r11d
  int v16; // r8d
  int v17; // edi
  int v18; // r15d
  int v19; // r13d
  int v20; // r11d
  int v21; // edi
  int v22; // r10d
  int v23; // r9d
  int v24; // r9d
  int v25; // esi
  int v26; // edx
  int v27; // r10d
  int v28; // r12d
  int v29; // ecx
  int v30; // esi
  int v31; // r8d
  int v32; // r9d
  int v33; // r11d
  int v34; // r13d
  int v35; // edx
  int v36; // r15d
  int v37; // r12d
  int v38; // ebx
  int v39; // ecx
  int v40; // r8d
  int v41; // r13d
  int v42; // r15d
  int v43; // r13d
  int v44; // r12d
  int v45; // ebx
  int v46; // esi
  int v47; // r11d
  int v48; // r13d
  int v49; // r12d
  int v50; // r10d
  int v51; // esi
  int v52; // edi
  int v53; // r13d
  int v54; // ecx
  int v55; // r10d
  int v56; // edi
  int v57; // r9d
  int v58; // ecx
  int v59; // r8d
  int v60; // r15d
  int v61; // edx
  int v62; // ebx
  int v63; // r9d
  int v64; // r15d
  int v65; // ebx
  int v66; // r13d
  int v67; // r10d
  int v68; // r8d
  int v69; // edx
  unsigned int v70; // r13d
  int v71; // r13d
  int v72; // r11d
  int v73; // esi
  int v74; // edi
  int v75; // r15d
  int v76; // r13d
  int v77; // r11d
  int v78; // edi
  int v79; // r15d
  int v80; // r10d
  int v81; // r9d
  int v82; // r8d
  int v83; // edx
  int v84; // r12d
  int v85; // r13d
  int v86; // esi
  int v87; // ecx
  int v88; // r9d
  int v89; // ebx
  int v90; // r8d
  int v91; // edx
  int v92; // r12d
  int v93; // r13d
  int v94; // ecx
  int v95; // r11d
  int v96; // r10d
  int v97; // ebx
  int v98; // r13d
  int v99; // edi
  int v100; // r9d
  int v101; // r12d
  int v102; // esi
  int v103; // ecx
  int v104; // r11d
  int v105; // r10d
  int v106; // r13d
  int v107; // edi
  int v108; // r12d
  int v109; // esi
  int v110; // ecx
  int v111; // r8d
  int v112; // r15d
  int v113; // edx
  int v114; // ebx
  int v115; // r9d
  int v116; // r13d
  int v117; // r8d
  int v118; // r15d
  int v119; // edx
  int v120; // ebx
  int v121; // r13d
  int v122; // r11d
  int v123; // edi
  int v124; // r15d
  int v125; // r13d
  int v126; // r11d
  int v127; // edi
  int v128; // r15d
  int v129; // r10d
  int v130; // r9d
  int v131; // edx
  int v132; // r12d
  int v133; // esi
  int v134; // r10d
  int v135; // ecx
  int v136; // r9d
  int v137; // esi
  int v138; // edx
  int v139; // r8d
  int v140; // r11d
  int v141; // r12d
  int v142; // r13d
  int v143; // ecx
  int v144; // ebx
  int v145; // r8d
  int v146; // r13d
  int v147; // r13d
  int v148; // r12d
  int v149; // ebx
  int v150; // esi
  int v151; // r11d
  int v152; // r13d
  int v153; // r12d
  int v154; // esi
  int v155; // r10d
  int v156; // r9d
  int v157; // edi
  int v158; // r8d
  int v159; // r15d
  int v160; // edx
  int v161; // ecx
  int v162; // ebx
  int v163; // r10d
  int v164; // r9d
  int v165; // edi
  int v166; // r15d
  int v167; // ecx
  int v168; // ebx
  int v169; // r13d
  int v170; // r10d
  int v171; // r8d
  int v172; // edx
  unsigned int v173; // r13d
  int v174; // r13d
  int v175; // r11d
  int v176; // esi
  int v177; // edi
  int v178; // r15d
  int v179; // r13d
  int v180; // r11d
  int v181; // edi
  int v182; // r15d
  int v183; // r10d
  int v184; // r9d
  int v185; // r8d
  int v186; // edx
  int v187; // r12d
  int v188; // r13d
  int v189; // esi
  int v190; // ecx
  int v191; // r9d
  int v192; // ebx
  int v193; // r8d
  int v194; // edx
  int v195; // r12d
  int v196; // r13d
  int v197; // ecx
  int v198; // r11d
  int v199; // r10d
  int v200; // ebx
  int v201; // r13d
  int v202; // edi
  int v203; // r9d
  int v204; // r12d
  int v205; // esi
  int v206; // ecx
  int v207; // r11d
  int v208; // r10d
  int v209; // r13d
  int v210; // edi
  int v211; // r12d
  int v212; // esi
  int v213; // ecx
  int v214; // r8d
  int v215; // r15d
  int v216; // edx
  int v217; // ebx
  int v218; // r9d
  int v219; // r13d
  int v220; // r8d
  int v221; // r15d
  int v222; // edx
  int v223; // ebx
  int v224; // r13d
  int v225; // r11d
  int v226; // edi
  int v227; // r15d
  int v228; // r13d
  int v229; // r10d
  int v230; // r9d
  int v231; // r11d
  int v232; // edx
  int v233; // r12d
  int v234; // edi
  int v235; // r15d
  int v236; // esi
  int v237; // ecx
  int v238; // r10d
  int v239; // r9d
  int v240; // esi
  int v241; // edx
  int v242; // r8d
  int v243; // r11d
  int v244; // r12d
  int v245; // r13d
  int v246; // ecx
  int v247; // ebx
  int v248; // r8d
  int v249; // r13d
  int v250; // r13d
  int v251; // r12d
  int v252; // ebx
  int v253; // esi
  int v254; // r11d
  int v255; // r13d
  int v256; // r12d
  int v257; // esi
  int v258; // r10d
  int v259; // r9d
  int v260; // edi
  int v261; // r15d
  int v262; // ecx
  int v263; // ebx
  int v264; // r10d
  int v265; // edi
  int v266; // r9d
  int v267; // ecx
  int v268; // r8d
  int v269; // r15d
  int v270; // edx
  int v271; // ebx
  int v272; // r13d
  int v273; // r10d
  int v274; // r8d
  int v275; // edx
  int v276; // r13d
  int v277; // r11d
  int v278; // edi
  int v279; // r15d
  int v280; // r13d
  int v281; // r11d
  int v282; // edi
  int v283; // r13d
  int v284; // esi
  int v285; // r9d
  int v286; // edx
  int v287; // r15d
  int v288; // r11d
  int v289; // ecx
  int v290; // r9d
  int v291; // edx
  int v292; // r15d
  int v293; // r12d
  int v294; // r10d
  int v295; // r13d
  int v296; // r8d
  int v297; // ebx
  int v298; // r10d
  int v299; // r13d
  int v300; // r8d
  int v301; // ecx
  int v302; // r13d
  int v303; // ebx
  int v304; // r15d
  int v305; // ebx
  int v306; // esi
  int v307; // r11d
  int v308; // r8d
  int v309; // r12d
  int v310; // ebx
  int v311; // esi
  int v312; // r12d
  int v313; // r8d
  int v314; // r12d
  int v315; // ecx
  int v316; // r11d
  int v317; // r10d
  int v318; // r8d
  int v319; // edx
  int v320; // ecx
  int v321; // r9d
  int v322; // ebx
  int v323; // edi
  int v324; // r10d
  int v325; // edx
  int v326; // edi
  int v327; // r15d
  int v328; // esi
  int v329; // r8d
  int v330; // edi
  int v331; // r15d
  int v332; // esi
  int v333; // r11d
  int v334; // r13d
  int v335; // edx
  int v336; // r10d
  int v337; // r9d
  int v338; // ebx
  int v339; // edi
  int v340; // r12d
  int v341; // r8d
  int v342; // r11d
  int v343; // r13d
  int v344; // ecx
  int v345; // edx
  int v346; // r9d
  int v347; // r12d
  int v348; // r10d
  int v349; // ebx
  int v350; // r15d
  int v351; // edi
  int v352; // esi
  int v353; // r8d
  int v354; // edi
  int v355; // ecx
  int v356; // r8d
  int v357; // r9d
  int v358; // ecx
  int v359; // r12d
  int v360; // ebx
  int v361; // r15d
  int v362; // esi
  int v363; // edi
  int v364; // r8d
  int v365; // r9d
  int v366; // r11d
  int v367; // ecx
  int v368; // r11d
  int v369; // r10d
  int v370; // edx
  int v371; // r13d
  int v372; // r12d
  int v373; // r10d
  int v374; // edx
  int v375; // r13d
  __int64 result; // rax
  _DWORD *v378; // [rsp+8h] [rbp-88h]
  int v379; // [rsp+10h] [rbp-80h]
  int v380; // [rsp+14h] [rbp-7Ch]
  int v381; // [rsp+18h] [rbp-78h]
  int v382; // [rsp+18h] [rbp-78h]
  int v383; // [rsp+18h] [rbp-78h]
  int v384; // [rsp+18h] [rbp-78h]
  int v385; // [rsp+18h] [rbp-78h]
  int v386; // [rsp+18h] [rbp-78h]
  int v387; // [rsp+1Ch] [rbp-74h]
  int v388; // [rsp+1Ch] [rbp-74h]
  int v389; // [rsp+1Ch] [rbp-74h]
  int v390; // [rsp+1Ch] [rbp-74h]
  int v391; // [rsp+1Ch] [rbp-74h]
  int v392; // [rsp+1Ch] [rbp-74h]
  int v393; // [rsp+20h] [rbp-70h]
  int v394; // [rsp+20h] [rbp-70h]
  int v395; // [rsp+20h] [rbp-70h]
  int v396; // [rsp+20h] [rbp-70h]
  int v397; // [rsp+20h] [rbp-70h]
  int v398; // [rsp+20h] [rbp-70h]
  int v399; // [rsp+24h] [rbp-6Ch]
  int v400; // [rsp+24h] [rbp-6Ch]
  int v401; // [rsp+24h] [rbp-6Ch]
  int v402; // [rsp+24h] [rbp-6Ch]
  int v403; // [rsp+24h] [rbp-6Ch]
  int v404; // [rsp+24h] [rbp-6Ch]
  int v405; // [rsp+24h] [rbp-6Ch]
  int v406; // [rsp+28h] [rbp-68h]
  int v407; // [rsp+28h] [rbp-68h]
  int v408; // [rsp+2Ch] [rbp-64h]
  int v409; // [rsp+30h] [rbp-60h]
  int v410; // [rsp+34h] [rbp-5Ch]
  int v411; // [rsp+38h] [rbp-58h]
  int v412; // [rsp+38h] [rbp-58h]
  int v413; // [rsp+3Ch] [rbp-54h]
  int v414; // [rsp+40h] [rbp-50h]
  int v415; // [rsp+44h] [rbp-4Ch]
  int v416; // [rsp+48h] [rbp-48h]
  int v417; // [rsp+4Ch] [rbp-44h]
  int v418; // [rsp+50h] [rbp-40h]
  int v419; // [rsp+54h] [rbp-3Ch]
  int v420; // [rsp+58h] [rbp-38h]
  int v421; // [rsp+5Ch] [rbp-34h]
  int v422; // [rsp+60h] [rbp-30h]
  int v423; // [rsp+60h] [rbp-30h]
  int v424; // [rsp+64h] [rbp-2Ch]
  int v425; // [rsp+64h] [rbp-2Ch]

  v8 = a2[3];
  v418 = a2[5];
  v417 = a2[6];
  v9 = a2[1];
  v419 = a2[4];
  v10 = a2[2];
  v416 = a2[7];
  v415 = a2[8];
  v414 = a2[9];
  v413 = a2[10];
  v11 = a1;
  v411 = a2[11];
  v410 = a2[12];
  v409 = a2[13];
  v12 = a1[4];
  v424 = *a2;
  v408 = a2[14];
  v13 = a1[6];
  v406 = a2[15];
  v14 = a1[5];
  v422 = v9;
  v420 = v8;
  v399 = a1[7];
  v421 = v10;
  v15 = v424 + v12 + *a1;
  v16 = v14 + a1[1];
  v17 = __ROL4__(a4 ^ v15, 16);
  v18 = v17 + 1779033703;
  v19 = __ROR4__((v17 + 1779033703) ^ v12, 12);
  v20 = v19 + v9 + v15;
  v21 = __ROR4__(v20 ^ v17, 8);
  v393 = v18 + v21;
  v378 = v11;
  v22 = v8 + v10 + v16;
  v23 = v11[2];
  v381 = __ROR4__((v18 + v21) ^ v19, 7);
  LODWORD(a4) = __ROL4__((v10 + v16) ^ HIDWORD(a4), 16);
  LODWORD(v11) = a4 - 1150833019;
  v24 = v419 + v13 + v23;
  v25 = __ROR4__((a4 - 1150833019) ^ v14, 12);
  v26 = __ROL4__(v24 ^ a3, 16);
  v27 = v25 + v22;
  v28 = v26 + 1013904242;
  LODWORD(a4) = __ROR4__(v27 ^ a4, 8);
  v29 = __ROR4__((v26 + 1013904242) ^ v13, 12);
  LODWORD(v11) = a4 + (_DWORD)v11;
  v30 = __ROR4__((unsigned int)v11 ^ v25, 7);
  v31 = v417 + v378[3] + v399;
  v32 = v29 + v418 + v24;
  v33 = v30 + v415 + v20;
  v34 = __ROL4__(v31 ^ a5, 16);
  v35 = __ROR4__(v32 ^ v26, 8);
  v36 = v34 - 1521486534;
  v37 = v35 + v28;
  v38 = __ROR4__((v34 - 1521486534) ^ v399, 12);
  v39 = __ROR4__(v37 ^ v29, 7);
  v40 = v38 + v416 + v31;
  v41 = __ROR4__(v40 ^ v34, 8);
  v42 = v41 + v36;
  v43 = __ROL4__(v33 ^ v41, 16);
  v44 = v43 + v37;
  v45 = __ROR4__(v42 ^ v38, 7);
  v46 = __ROR4__(v44 ^ v30, 12);
  v47 = v46 + v414 + v33;
  v48 = __ROR4__(v47 ^ v43, 8);
  v49 = v48 + v44;
  v400 = v48;
  v50 = v39 + v413 + v27;
  v51 = __ROR4__(v49 ^ v46, 7);
  v52 = __ROL4__(v50 ^ v21, 16);
  v53 = v42 + v52;
  v54 = __ROR4__((v42 + v52) ^ v39, 12);
  v55 = v54 + v411 + v50;
  v56 = __ROR4__(v55 ^ v52, 8);
  v387 = v53 + v56;
  v57 = v45 + v410 + v32;
  v58 = __ROR4__((v53 + v56) ^ v54, 7);
  LODWORD(a4) = __ROL4__(v57 ^ a4, 16);
  v59 = v381 + v408 + v40;
  v60 = a4 + v393;
  v61 = __ROL4__(v59 ^ v35, 16);
  v62 = __ROR4__((a4 + v393) ^ v45, 12);
  v63 = v62 + v409 + v57;
  LODWORD(a4) = __ROR4__(v63 ^ a4, 8);
  v64 = a4 + v60;
  v65 = __ROR4__(v64 ^ v62, 7);
  LODWORD(v11) = v61 + (_DWORD)v11;
  v66 = __ROR4__((unsigned int)v11 ^ v381, 12);
  v67 = v51 + v420 + v55;
  v68 = v66 + v406 + v59;
  LODWORD(a4) = __ROL4__(v67 ^ a4, 16);
  v69 = __ROR4__(v68 ^ v61, 8);
  LODWORD(v11) = v69 + (_DWORD)v11;
  v70 = (unsigned int)v11 ^ v66;
  LODWORD(v11) = a4 + (_DWORD)v11;
  v71 = __ROR4__(v70, 7);
  v72 = v71 + v421 + v47;
  v73 = __ROR4__((unsigned int)v11 ^ v51, 12);
  v74 = __ROL4__(v72 ^ v56, 16);
  v75 = v74 + v64;
  v76 = __ROR4__(v75 ^ v71, 12);
  v77 = v76 + v417 + v72;
  v78 = __ROR4__(v77 ^ v74, 8);
  v79 = v78 + v75;
  v80 = v73 + v413 + v67;
  v382 = __ROR4__(v79 ^ v76, 7);
  v81 = v58 + v416 + v63;
  LODWORD(a4) = __ROR4__(v80 ^ a4, 8);
  v82 = v65 + v419 + v68;
  v83 = __ROL4__(v81 ^ v69, 16);
  v394 = a4 + (_DWORD)v11;
  v84 = v83 + v49;
  v85 = __ROL4__(v82 ^ v400, 16);
  v86 = __ROR4__((a4 + (_DWORD)v11) ^ v73, 7);
  LODWORD(v11) = v85 + v387;
  v87 = __ROR4__(v84 ^ v58, 12);
  v88 = v87 + v424 + v81;
  v89 = __ROR4__((v85 + v387) ^ v65, 12);
  v90 = v89 + v409 + v82;
  v91 = __ROR4__(v88 ^ v83, 8);
  v92 = v91 + v84;
  v93 = __ROR4__(v90 ^ v85, 8);
  LODWORD(v11) = v93 + (_DWORD)v11;
  v94 = __ROR4__(v92 ^ v87, 7);
  v95 = v86 + v422 + v77;
  v96 = v94 + v410 + v80;
  v97 = __ROR4__((unsigned int)v11 ^ v89, 7);
  v98 = __ROL4__(v95 ^ v93, 16);
  v99 = __ROL4__(v96 ^ v78, 16);
  v100 = v97 + v414 + v88;
  v101 = v98 + v92;
  LODWORD(v11) = v99 + (_DWORD)v11;
  v102 = __ROR4__(v101 ^ v86, 12);
  v103 = __ROR4__((unsigned int)v11 ^ v94, 12);
  v104 = v102 + v411 + v95;
  v105 = v103 + v418 + v96;
  v106 = __ROR4__(v104 ^ v98, 8);
  v107 = __ROR4__(v105 ^ v99, 8);
  v108 = v106 + v101;
  v401 = v106;
  v388 = v107 + (_DWORD)v11;
  v109 = __ROR4__(v108 ^ v102, 7);
  v110 = __ROR4__((v107 + (_DWORD)v11) ^ v103, 7);
  v111 = v382 + v406 + v90;
  LODWORD(a4) = __ROL4__(v100 ^ a4, 16);
  v112 = a4 + v79;
  v113 = __ROL4__(v111 ^ v91, 16);
  LODWORD(v11) = v113 + v394;
  v114 = __ROR4__(v112 ^ v97, 12);
  v115 = v114 + v408 + v100;
  v116 = __ROR4__((v113 + v394) ^ v382, 12);
  v117 = v116 + v415 + v111;
  LODWORD(a4) = __ROR4__(v115 ^ a4, 8);
  v118 = a4 + v112;
  v119 = __ROR4__(v117 ^ v113, 8);
  LODWORD(v11) = v119 + (_DWORD)v11;
  v120 = __ROR4__(v118 ^ v114, 7);
  v121 = __ROR4__((unsigned int)v11 ^ v116, 7);
  v122 = v121 + v420 + v104;
  v123 = __ROL4__(v122 ^ v107, 16);
  v124 = v123 + v118;
  v125 = __ROR4__(v124 ^ v121, 12);
  v126 = v125 + v419 + v122;
  v127 = __ROR4__(v126 ^ v123, 8);
  v128 = v127 + v124;
  v383 = __ROR4__(v128 ^ v125, 7);
  v129 = v109 + v413 + v105;
  v130 = v110 + v409 + v115;
  LODWORD(a4) = __ROL4__(v129 ^ a4, 16);
  LODWORD(v11) = a4 + (_DWORD)v11;
  v131 = __ROL4__(v130 ^ v119, 16);
  v132 = v131 + v108;
  v133 = __ROR4__((unsigned int)v11 ^ v109, 12);
  v134 = v133 + v410 + v129;
  v135 = __ROR4__(v132 ^ v110, 12);
  v136 = v135 + v421 + v130;
  LODWORD(a4) = __ROR4__(v134 ^ a4, 8);
  v395 = a4 + (_DWORD)v11;
  v137 = __ROR4__((a4 + (_DWORD)v11) ^ v133, 7);
  v138 = __ROR4__(v136 ^ v131, 8);
  v139 = v120 + v416 + v117;
  v140 = v137 + v417 + v126;
  v141 = v138 + v132;
  v142 = __ROL4__(v139 ^ v401, 16);
  v143 = __ROR4__(v141 ^ v135, 7);
  LODWORD(v11) = v142 + v388;
  v144 = __ROR4__((v142 + v388) ^ v120, 12);
  v145 = v144 + v408 + v139;
  v146 = __ROR4__(v145 ^ v142, 8);
  LODWORD(v11) = v146 + (_DWORD)v11;
  v147 = __ROL4__(v140 ^ v146, 16);
  v148 = v147 + v141;
  v149 = __ROR4__((unsigned int)v11 ^ v144, 7);
  v150 = __ROR4__(v148 ^ v137, 12);
  v151 = v150 + v418 + v140;
  v152 = __ROR4__(v151 ^ v147, 8);
  v153 = v152 + v148;
  v402 = v152;
  v154 = __ROR4__(v153 ^ v150, 7);
  v155 = v143 + v414 + v134;
  v156 = v149 + v411 + v136;
  v157 = __ROL4__(v155 ^ v127, 16);
  LODWORD(a4) = __ROL4__(v156 ^ a4, 16);
  v158 = v383 + v415 + v145;
  LODWORD(v11) = v157 + (_DWORD)v11;
  v159 = a4 + v128;
  v160 = __ROL4__(v158 ^ v138, 16);
  v161 = __ROR4__((unsigned int)v11 ^ v143, 12);
  v162 = __ROR4__(v159 ^ v149, 12);
  v163 = v161 + v424 + v155;
  v164 = v162 + v406 + v156;
  v165 = __ROR4__(v163 ^ v157, 8);
  LODWORD(a4) = __ROR4__(v164 ^ a4, 8);
  v166 = a4 + v159;
  v389 = v165 + (_DWORD)v11;
  v167 = __ROR4__((v165 + (_DWORD)v11) ^ v161, 7);
  v168 = __ROR4__(v166 ^ v162, 7);
  LODWORD(v11) = v160 + v395;
  v169 = __ROR4__((v160 + v395) ^ v383, 12);
  v170 = v154 + v410 + v163;
  v171 = v169 + v422 + v158;
  LODWORD(a4) = __ROL4__(v170 ^ a4, 16);
  v172 = __ROR4__(v171 ^ v160, 8);
  LODWORD(v11) = v172 + (_DWORD)v11;
  v173 = (unsigned int)v11 ^ v169;
  LODWORD(v11) = a4 + (_DWORD)v11;
  v174 = __ROR4__(v173, 7);
  v175 = v174 + v413 + v151;
  v176 = __ROR4__((unsigned int)v11 ^ v154, 12);
  v177 = __ROL4__(v175 ^ v165, 16);
  v178 = v177 + v166;
  v179 = __ROR4__(v178 ^ v174, 12);
  v180 = v179 + v416 + v175;
  v181 = __ROR4__(v180 ^ v177, 8);
  v182 = v181 + v178;
  v183 = v176 + v414 + v170;
  v184 = v167 + v408 + v164;
  v384 = __ROR4__(v182 ^ v179, 7);
  LODWORD(a4) = __ROR4__(v183 ^ a4, 8);
  v185 = v168 + v409 + v171;
  v186 = __ROL4__(v184 ^ v172, 16);
  v396 = a4 + (_DWORD)v11;
  v187 = v186 + v153;
  v188 = __ROL4__(v185 ^ v402, 16);
  v189 = __ROR4__((a4 + (_DWORD)v11) ^ v176, 7);
  LODWORD(v11) = v188 + v389;
  v190 = __ROR4__(v187 ^ v167, 12);
  v191 = v190 + v420 + v184;
  v192 = __ROR4__((v188 + v389) ^ v168, 12);
  v193 = v192 + v406 + v185;
  v194 = __ROR4__(v191 ^ v186, 8);
  v195 = v194 + v187;
  v196 = __ROR4__(v193 ^ v188, 8);
  LODWORD(v11) = v196 + (_DWORD)v11;
  v197 = __ROR4__(v195 ^ v190, 7);
  v198 = v189 + v419 + v180;
  v199 = v197 + v411 + v183;
  v200 = __ROR4__((unsigned int)v11 ^ v192, 7);
  v201 = __ROL4__(v198 ^ v196, 16);
  v202 = __ROL4__(v199 ^ v181, 16);
  v203 = v200 + v418 + v191;
  v204 = v201 + v195;
  LODWORD(v11) = v202 + (_DWORD)v11;
  v205 = __ROR4__(v204 ^ v189, 12);
  v206 = __ROR4__((unsigned int)v11 ^ v197, 12);
  v207 = v205 + v424 + v198;
  v208 = v206 + v421 + v199;
  v209 = __ROR4__(v207 ^ v201, 8);
  v210 = __ROR4__(v208 ^ v202, 8);
  v211 = v209 + v204;
  v403 = v209;
  v390 = v210 + (_DWORD)v11;
  v212 = __ROR4__(v211 ^ v205, 7);
  v213 = __ROR4__((v210 + (_DWORD)v11) ^ v206, 7);
  v214 = v384 + v422 + v193;
  LODWORD(a4) = __ROL4__(v203 ^ a4, 16);
  v215 = a4 + v182;
  v216 = __ROL4__(v214 ^ v194, 16);
  LODWORD(v11) = v216 + v396;
  v217 = __ROR4__(v215 ^ v200, 12);
  v218 = v217 + v415 + v203;
  v219 = __ROR4__((v216 + v396) ^ v384, 12);
  v220 = v219 + v417 + v214;
  LODWORD(a4) = __ROR4__(v218 ^ a4, 8);
  v221 = a4 + v215;
  v222 = __ROR4__(v220 ^ v216, 8);
  LODWORD(v11) = v222 + (_DWORD)v11;
  v223 = __ROR4__(v221 ^ v217, 7);
  v224 = __ROR4__((unsigned int)v11 ^ v219, 7);
  v225 = v224 + v410 + v207;
  v226 = __ROL4__(v225 ^ v210, 16);
  v227 = v226 + v221;
  v228 = __ROR4__(v227 ^ v224, 12);
  v229 = v212 + v414 + v208;
  v230 = v213 + v406 + v218;
  v231 = v228 + v409 + v225;
  LODWORD(a4) = __ROL4__(v229 ^ a4, 16);
  v232 = __ROL4__(v230 ^ v222, 16);
  LODWORD(v11) = a4 + (_DWORD)v11;
  v233 = v232 + v211;
  v234 = __ROR4__(v231 ^ v226, 8);
  v235 = v234 + v227;
  v236 = __ROR4__((unsigned int)v11 ^ v212, 12);
  v237 = __ROR4__(v233 ^ v213, 12);
  v238 = v236 + v411 + v229;
  v239 = v237 + v413 + v230;
  v385 = __ROR4__(v235 ^ v228, 7);
  LODWORD(a4) = __ROR4__(v238 ^ a4, 8);
  v397 = a4 + (_DWORD)v11;
  v240 = __ROR4__((a4 + (_DWORD)v11) ^ v236, 7);
  v241 = __ROR4__(v239 ^ v232, 8);
  v242 = v223 + v408 + v220;
  v243 = v240 + v416 + v231;
  v244 = v241 + v233;
  v245 = __ROL4__(v242 ^ v403, 16);
  v246 = __ROR4__(v244 ^ v237, 7);
  LODWORD(v11) = v245 + v390;
  v247 = __ROR4__((v245 + v390) ^ v223, 12);
  v248 = v247 + v415 + v242;
  v249 = __ROR4__(v248 ^ v245, 8);
  LODWORD(v11) = v249 + (_DWORD)v11;
  v250 = __ROL4__(v243 ^ v249, 16);
  v251 = v250 + v244;
  v252 = __ROR4__((unsigned int)v11 ^ v247, 7);
  v253 = __ROR4__(v251 ^ v240, 12);
  v254 = v253 + v421 + v243;
  v255 = __ROR4__(v254 ^ v250, 8);
  v256 = v255 + v251;
  v404 = v255;
  v257 = __ROR4__(v256 ^ v253, 7);
  v258 = v246 + v418 + v238;
  v259 = v252 + v424 + v239;
  v260 = __ROL4__(v258 ^ v234, 16);
  LODWORD(a4) = __ROL4__(v259 ^ a4, 16);
  LODWORD(v11) = v260 + (_DWORD)v11;
  v261 = a4 + v235;
  v262 = __ROR4__((unsigned int)v11 ^ v246, 12);
  v263 = __ROR4__(v261 ^ v252, 12);
  v264 = v262 + v420 + v258;
  v265 = __ROR4__(v264 ^ v260, 8);
  v391 = v265 + (_DWORD)v11;
  v266 = v263 + v422 + v259;
  v267 = __ROR4__((v265 + (_DWORD)v11) ^ v262, 7);
  v268 = v385 + v417 + v248;
  LODWORD(a4) = __ROR4__(v266 ^ a4, 8);
  v269 = a4 + v261;
  v270 = __ROL4__(v268 ^ v241, 16);
  v271 = __ROR4__(v269 ^ v263, 7);
  LODWORD(v11) = v270 + v397;
  v272 = __ROR4__((v270 + v397) ^ v385, 12);
  v273 = v257 + v411 + v264;
  v274 = v272 + v419 + v268;
  LODWORD(a4) = __ROL4__(v273 ^ a4, 16);
  v275 = __ROR4__(v274 ^ v270, 8);
  LODWORD(v11) = v275 + (_DWORD)v11;
  v276 = __ROR4__((unsigned int)v11 ^ v272, 7);
  v277 = v276 + v414 + v254;
  v278 = __ROL4__(v277 ^ v265, 16);
  v279 = v278 + v269;
  v280 = __ROR4__(v279 ^ v276, 12);
  v281 = v280 + v408 + v277;
  v282 = __ROR4__(v281 ^ v278, 8);
  v398 = v282 + v279;
  v379 = __ROR4__((v282 + v279) ^ v280, 7);
  v283 = (_DWORD)v11 + a4;
  LODWORD(v11) = __ROR4__(((_DWORD)v11 + a4) ^ v257, 12);
  v284 = v418 + v273 + (_DWORD)v11;
  v285 = v267 + v415 + v266;
  LODWORD(a4) = __ROR4__(v284 ^ a4, 8);
  v286 = __ROL4__(v285 ^ v275, 16);
  v386 = v283 + a4;
  v287 = v256 + v286;
  LODWORD(v11) = __ROR4__((v283 + a4) ^ (unsigned int)v11, 7);
  v288 = (_DWORD)v11 + v409 + v281;
  v289 = __ROR4__((v256 + v286) ^ v267, 12);
  v290 = v289 + v410 + v285;
  v291 = __ROR4__(v290 ^ v286, 8);
  v292 = v291 + v287;
  v293 = __ROR4__(v292 ^ v289, 7);
  v294 = v271 + v406 + v274;
  v295 = __ROL4__(v294 ^ v404, 16);
  v296 = v295 + v391;
  v297 = __ROR4__((v295 + v391) ^ v271, 12);
  v298 = v297 + v422 + v294;
  v299 = __ROR4__(v298 ^ v295, 8);
  v300 = v299 + v296;
  v301 = __ROR4__(v300 ^ v297, 7);
  v302 = __ROL4__(v288 ^ v299, 16);
  v303 = v292 + v302;
  LODWORD(v11) = __ROR4__((v292 + v302) ^ (unsigned int)v11, 12);
  v304 = (_DWORD)v11 + v288 + v420;
  v405 = __ROR4__(v304 ^ v302, 8);
  v392 = v405 + v303;
  LODWORD(v11) = __ROR4__((v405 + v303) ^ (unsigned int)v11, 7);
  v305 = v293 + v424 + v284;
  v306 = __ROL4__(v305 ^ v282, 16);
  v307 = v301 + v421 + v290;
  v308 = v306 + v300;
  LODWORD(a4) = __ROL4__(v307 ^ a4, 16);
  v309 = __ROR4__(v308 ^ v293, 12);
  v310 = v309 + v413 + v305;
  v311 = __ROR4__(v310 ^ v306, 8);
  v380 = v311 + v308;
  v312 = (v311 + v308) ^ v309;
  v313 = v398 + a4;
  v314 = __ROR4__(v312, 7);
  v315 = __ROR4__((v398 + a4) ^ v301, 12);
  v316 = v315 + v417 + v307;
  LODWORD(a4) = __ROR4__(v316 ^ a4, 8);
  v317 = v379 + v419 + v298;
  v318 = a4 + v313;
  v319 = __ROL4__(v317 ^ v291, 16);
  v320 = __ROR4__(v318 ^ v315, 7);
  v321 = v319 + v386;
  v322 = (_DWORD)v11 + v418 + v310;
  v323 = __ROR4__((v319 + v386) ^ v379, 12);
  v324 = v323 + v416 + v317;
  v325 = __ROR4__(v324 ^ v319, 8);
  v326 = __ROR4__((v321 + v325) ^ v323, 7);
  v327 = v326 + v411 + v304;
  v328 = __ROL4__(v327 ^ v311, 16);
  v329 = v328 + v318;
  v330 = __ROR4__(v329 ^ v326, 12);
  v331 = v330 + v406 + v327;
  v332 = __ROR4__(v331 ^ v328, 8);
  v412 = v329 + v332;
  v333 = v314 + v422 + v316;
  v407 = __ROR4__((v329 + v332) ^ v330, 7);
  LODWORD(a4) = __ROL4__(v322 ^ a4, 16);
  v334 = a4 + v321 + v325;
  v335 = __ROL4__(v333 ^ v325, 16);
  v336 = v320 + v415 + v324;
  v337 = v392 + v335;
  LODWORD(v11) = __ROR4__(v334 ^ (unsigned int)v11, 12);
  v338 = (_DWORD)v11 + v424 + v322;
  v339 = __ROL4__(v336 ^ v405, 16);
  v340 = __ROR4__((v392 + v335) ^ v314, 12);
  v341 = v339 + v380;
  v342 = v340 + v414 + v333;
  LODWORD(a4) = __ROR4__(v338 ^ a4, 8);
  v343 = a4 + v334;
  v344 = __ROR4__((v339 + v380) ^ v320, 12);
  v345 = __ROR4__(v342 ^ v335, 8);
  v346 = v345 + v337;
  LODWORD(v11) = __ROR4__(v343 ^ (unsigned int)v11, 7);
  v347 = __ROR4__(v346 ^ v340, 7);
  v348 = v344 + v417 + v336;
  v349 = v347 + v421 + v338;
  v350 = (_DWORD)v11 + v408 + v331;
  v351 = __ROR4__(v348 ^ v339, 8);
  v352 = __ROL4__(v349 ^ v332, 16);
  v353 = v351 + v341;
  v354 = __ROL4__(v350 ^ v351, 16);
  v355 = v353 ^ v344;
  v356 = v352 + v353;
  v357 = v354 + v346;
  v358 = __ROR4__(v355, 7);
  v359 = __ROR4__(v356 ^ v347, 12);
  LODWORD(v11) = __ROR4__(v357 ^ (unsigned int)v11, 12);
  v360 = v359 + v410 + v349;
  v423 = v359;
  v361 = (_DWORD)v11 + v413 + v350;
  v425 = (int)v11;
  v362 = __ROR4__(v360 ^ v352, 8);
  v363 = __ROR4__(v361 ^ v354, 8);
  v364 = v362 + v356;
  v365 = v363 + v357;
  v366 = v358 + v420 + v342;
  LODWORD(a4) = __ROL4__(v366 ^ a4, 16);
  LODWORD(v11) = a4 + v412;
  v367 = __ROR4__((a4 + v412) ^ v358, 12);
  v368 = v367 + v419 + v366;
  v369 = v407 + v416 + v348;
  LODWORD(a4) = __ROR4__(v368 ^ a4, 8);
  LODWORD(v11) = a4 + (_DWORD)v11;
  v370 = __ROL4__(v369 ^ v345, 16);
  v371 = v370 + v343;
  v372 = __ROR4__(v371 ^ v407, 12);
  v373 = v372 + v409 + v369;
  *a6 = (unsigned int)v11 ^ v361;
  a6[2] = v365 ^ v368;
  v374 = __ROR4__(v373 ^ v370, 8);
  a6[3] = v364 ^ v373;
  v375 = v374 + v371;
  a6[1] = v375 ^ v360;
  a6[4] = v362 ^ __ROR4__(v375 ^ v372, 7);
  a6[5] = a4 ^ __ROR4__(v365 ^ v425, 7);
  a6[6] = v374 ^ __ROR4__(v364 ^ v423, 7);
  a6[7] = v363 ^ __ROR4__((unsigned int)v11 ^ v367, 7);
  a6[8] = *v378 ^ (unsigned int)v11;
  a6[9] = v378[1] ^ v375;
  a6[10] = v378[2] ^ v365;
  a6[11] = v378[3] ^ v364;
  a6[12] = v378[4] ^ v362;
  result = v378[5] ^ (unsigned int)a4;
  a6[13] = result;
  a6[14] = v378[6] ^ v374;
  a6[15] = v378[7] ^ v363;
  return result;
}
