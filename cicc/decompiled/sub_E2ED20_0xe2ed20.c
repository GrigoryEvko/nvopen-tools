// Function: sub_E2ED20
// Address: 0xe2ed20
//
void __fastcall sub_E2ED20(__int64 a1, __int64 *a2, unsigned int a3)
{
  __int64 v4; // rax
  unsigned __int64 v5; // rcx
  char *v6; // rdi
  unsigned __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  char *v11; // rdi
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  __m128i v14; // xmm0
  __m128i *v15; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // rcx
  char *v18; // rdi
  unsigned __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  char *v23; // rdi
  unsigned __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int64 v27; // rcx
  char *v28; // rdi
  unsigned __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned __int64 v32; // rcx
  char *v33; // rdi
  unsigned __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // rax
  unsigned __int64 v37; // rcx
  char *v38; // rdi
  unsigned __int64 v39; // rcx
  __int64 v40; // rax
  char *v41; // rax
  __int64 v42; // rax
  unsigned __int64 v43; // rcx
  char *v44; // rdi
  unsigned __int64 v45; // rcx
  __int64 v46; // rax
  char *v47; // rax
  __int64 v48; // rax
  unsigned __int64 v49; // rcx
  char *v50; // rdi
  unsigned __int64 v51; // rcx
  __int64 v52; // rax
  char *v53; // rax
  __int64 v54; // rax
  unsigned __int64 v55; // rcx
  char *v56; // rdi
  unsigned __int64 v57; // rcx
  __int64 v58; // rax
  __int64 v59; // rax
  unsigned __int64 v60; // rcx
  char *v61; // rdi
  unsigned __int64 v62; // rcx
  __int64 v63; // rax
  char *v64; // rax
  __int64 v65; // rax
  unsigned __int64 v66; // rcx
  char *v67; // rdi
  unsigned __int64 v68; // rcx
  __int64 v69; // rax
  __int64 v70; // rax
  unsigned __int64 v71; // rcx
  char *v72; // rdi
  unsigned __int64 v73; // rcx
  __int64 v74; // rax
  char *v75; // rax
  __int64 v76; // rax
  unsigned __int64 v77; // rcx
  char *v78; // rdi
  unsigned __int64 v79; // rcx
  __int64 v80; // rax
  __int64 v81; // rax
  unsigned __int64 v82; // rcx
  char *v83; // rdi
  unsigned __int64 v84; // rcx
  __int64 v85; // rax
  char *v86; // rax
  __int64 v87; // rax
  unsigned __int64 v88; // rcx
  char *v89; // rdi
  unsigned __int64 v90; // rcx
  __int64 v91; // rax
  char *v92; // rax
  __int64 v93; // rax
  unsigned __int64 v94; // rcx
  char *v95; // rdi
  unsigned __int64 v96; // rcx
  __int64 v97; // rax
  char *v98; // rax
  __int64 v99; // rax
  unsigned __int64 v100; // rcx
  char *v101; // rdi
  unsigned __int64 v102; // rcx
  __int64 v103; // rax
  __int64 v104; // rax
  unsigned __int64 v105; // rcx
  char *v106; // rdi
  unsigned __int64 v107; // rcx
  __int64 v108; // rax
  char *v109; // rax
  __int64 v110; // rax
  unsigned __int64 v111; // rcx
  char *v112; // rdi
  unsigned __int64 v113; // rcx
  __int64 v114; // rax
  char *v115; // rax
  __int64 v116; // rax
  unsigned __int64 v117; // rcx
  char *v118; // rdi
  unsigned __int64 v119; // rcx
  __int64 v120; // rax
  char *v121; // rax
  __int64 v122; // rax
  unsigned __int64 v123; // rcx
  char *v124; // rdi
  unsigned __int64 v125; // rcx
  __int64 v126; // rax
  __int64 v127; // rax
  unsigned __int64 v128; // rcx
  char *v129; // rdi
  unsigned __int64 v130; // rcx
  __int64 v131; // rax
  __int64 v132; // rax
  unsigned __int64 v133; // rcx
  char *v134; // rdi
  unsigned __int64 v135; // rcx
  __int64 v136; // rax
  char *v137; // rax
  __int64 v138; // rax
  unsigned __int64 v139; // rcx
  char *v140; // rdi
  unsigned __int64 v141; // rcx
  __int64 v142; // rax
  __int64 v143; // rax
  unsigned __int64 v144; // rcx
  char *v145; // rdi
  unsigned __int64 v146; // rcx
  __int64 v147; // rax
  __int64 v148; // rax
  unsigned __int64 v149; // rcx
  char *v150; // rdi
  unsigned __int64 v151; // rcx
  __int64 v152; // rax
  __int64 v153; // rax
  unsigned __int64 v154; // rcx
  char *v155; // rdi
  unsigned __int64 v156; // rcx
  __int64 v157; // rax
  __int64 v158; // rax
  unsigned __int64 v159; // rcx
  char *v160; // rdi
  unsigned __int64 v161; // rcx
  __int64 v162; // rax
  char *v163; // rax
  __int64 v164; // rax
  unsigned __int64 v165; // rcx
  char *v166; // rdi
  unsigned __int64 v167; // rcx
  __int64 v168; // rax
  __int64 v169; // rax
  unsigned __int64 v170; // rcx
  char *v171; // rdi
  unsigned __int64 v172; // rcx
  __int64 v173; // rax
  __int64 v174; // rax
  unsigned __int64 v175; // rcx
  char *v176; // rdi
  unsigned __int64 v177; // rcx
  __int64 v178; // rax
  char *v179; // rax
  __int64 v180; // rax
  unsigned __int64 v181; // rcx
  char *v182; // rdi
  unsigned __int64 v183; // rcx
  __int64 v184; // rax
  __int64 v185; // rax
  unsigned __int64 v186; // rcx
  char *v187; // rdi
  unsigned __int64 v188; // rcx
  __int64 v189; // rax
  __int64 v190; // rax
  unsigned __int64 v191; // rcx
  char *v192; // rdi
  unsigned __int64 v193; // rcx
  __int64 v194; // rax
  __m128i v195; // xmm0
  __m128i *v196; // rax
  __int64 v197; // rax
  unsigned __int64 v198; // rcx
  char *v199; // rdi
  unsigned __int64 v200; // rcx
  __int64 v201; // rax
  __m128i v202; // xmm0
  __m128i *v203; // rax
  __int64 v204; // rax
  unsigned __int64 v205; // rcx
  char *v206; // rdi
  unsigned __int64 v207; // rcx
  __int64 v208; // rax
  __m128i v209; // xmm0
  __m128i *v210; // rax
  __int64 v211; // rax
  unsigned __int64 v212; // rcx
  char *v213; // rdi
  unsigned __int64 v214; // rcx
  __int64 v215; // rax
  __m128i v216; // xmm0
  __m128i *v217; // rax
  __int64 v218; // rax
  unsigned __int64 v219; // rcx
  char *v220; // rdi
  unsigned __int64 v221; // rcx
  __int64 v222; // rax
  __m128i v223; // xmm0
  __m128i *v224; // rax
  __int64 v225; // rax
  unsigned __int64 v226; // rcx
  char *v227; // rdi
  unsigned __int64 v228; // rcx
  __int64 v229; // rax
  __m128i v230; // xmm0
  __m128i *v231; // rax
  __int64 v232; // rax
  unsigned __int64 v233; // rcx
  char *v234; // rdi
  unsigned __int64 v235; // rcx
  __int64 v236; // rax
  __m128i si128; // xmm0
  __m128i *v238; // rax
  __int64 v239; // rax
  unsigned __int64 v240; // rcx
  char *v241; // rdi
  unsigned __int64 v242; // rcx
  __int64 v243; // rax
  __int64 v244; // rax
  unsigned __int64 v245; // rcx
  char *v246; // rdi
  unsigned __int64 v247; // rcx
  __int64 v248; // rax
  __int64 v249; // rax
  unsigned __int64 v250; // rcx
  char *v251; // rdi
  unsigned __int64 v252; // rcx
  __int64 v253; // rax
  __int64 v254; // rax
  unsigned __int64 v255; // rcx
  char *v256; // rdi
  unsigned __int64 v257; // rcx
  __int64 v258; // rax
  __int64 v259; // rax
  unsigned __int64 v260; // rcx
  char *v261; // rdi
  unsigned __int64 v262; // rcx
  __int64 v263; // rax
  __int64 v264; // rax
  unsigned __int64 v265; // rcx
  char *v266; // rdi
  unsigned __int64 v267; // rcx
  __int64 v268; // rax
  __int64 v269; // rax
  unsigned __int64 v270; // rcx
  char *v271; // rdi
  unsigned __int64 v272; // rcx
  __int64 v273; // rax
  __int64 v274; // rax
  unsigned __int64 v275; // rcx
  char *v276; // rdi
  unsigned __int64 v277; // rcx
  __int64 v278; // rax
  __int64 v279; // rax
  unsigned __int64 v280; // rcx
  char *v281; // rdi
  unsigned __int64 v282; // rcx
  __int64 v283; // rax
  __int64 v284; // rax
  unsigned __int64 v285; // rcx
  char *v286; // rdi
  unsigned __int64 v287; // rcx
  __int64 v288; // rax
  __m128i v289; // xmm0
  __m128i *v290; // rax
  __int64 v291; // rax
  unsigned __int64 v292; // rcx
  char *v293; // rdi
  unsigned __int64 v294; // rcx
  __int64 v295; // rax
  __m128i v296; // xmm0
  __m128i *v297; // rax
  __int64 v298; // rax
  unsigned __int64 v299; // rcx
  char *v300; // rdi
  unsigned __int64 v301; // rcx
  __int64 v302; // rax
  __int64 v303; // rax
  unsigned __int64 v304; // rcx
  char *v305; // rdi
  unsigned __int64 v306; // rcx
  __int64 v307; // rax
  __m128i v308; // xmm0
  __m128i *v309; // rax
  __int64 v310; // rax
  unsigned __int64 v311; // rcx
  char *v312; // rdi
  unsigned __int64 v313; // rcx
  __int64 v314; // rax
  __m128i v315; // xmm0
  __m128i *v316; // rax
  __int64 v317; // rax
  unsigned __int64 v318; // rcx
  char *v319; // rdi
  unsigned __int64 v320; // rcx
  __int64 v321; // rax
  __m128i v322; // xmm0
  __m128i *v323; // rax
  __int64 v324; // rax
  unsigned __int64 v325; // rcx
  char *v326; // rdi
  unsigned __int64 v327; // rcx
  __int64 v328; // rax
  __m128i v329; // xmm0
  __m128i *v330; // rax
  __int64 v331; // rax
  unsigned __int64 v332; // rcx
  char *v333; // rdi
  unsigned __int64 v334; // rcx
  __int64 v335; // rax
  __m128i v336; // xmm0
  __m128i *v337; // rax
  __int64 v338; // rax
  unsigned __int64 v339; // rcx
  char *v340; // rdi
  unsigned __int64 v341; // rcx
  __int64 v342; // rax
  __m128i v343; // xmm0
  __m128i *v344; // rax
  __int64 v345; // rax
  unsigned __int64 v346; // rcx
  char *v347; // rdi
  unsigned __int64 v348; // rcx
  __int64 v349; // rax
  __m128i v350; // xmm0
  __m128i *v351; // rax
  __int64 v352; // rax
  unsigned __int64 v353; // rcx
  char *v354; // rdi
  unsigned __int64 v355; // rcx
  __int64 v356; // rax
  __m128i v357; // xmm0
  __m128i *v358; // rax
  __int64 v359; // rax
  unsigned __int64 v360; // rcx
  char *v361; // rdi
  unsigned __int64 v362; // rcx
  __int64 v363; // rax
  __m128i v364; // xmm0
  __m128i *v365; // rax
  __int64 v366; // rax
  unsigned __int64 v367; // rcx
  char *v368; // rdi
  unsigned __int64 v369; // rcx
  __int64 v370; // rax
  __m128i *v371; // rax
  __int64 v372; // rax
  unsigned __int64 v373; // rcx
  char *v374; // rdi
  unsigned __int64 v375; // rcx
  __int64 v376; // rax
  __m128i v377; // xmm0
  __m128i *v378; // rax
  unsigned int v379; // [rsp+Ch] [rbp-14h]
  unsigned int v380; // [rsp+Ch] [rbp-14h]
  unsigned int v381; // [rsp+Ch] [rbp-14h]
  unsigned int v382; // [rsp+Ch] [rbp-14h]
  unsigned int v383; // [rsp+Ch] [rbp-14h]
  unsigned int v384; // [rsp+Ch] [rbp-14h]
  unsigned int v385; // [rsp+Ch] [rbp-14h]
  unsigned int v386; // [rsp+Ch] [rbp-14h]
  unsigned int v387; // [rsp+Ch] [rbp-14h]
  unsigned int v388; // [rsp+Ch] [rbp-14h]
  unsigned int v389; // [rsp+Ch] [rbp-14h]
  unsigned int v390; // [rsp+Ch] [rbp-14h]
  unsigned int v391; // [rsp+Ch] [rbp-14h]
  unsigned int v392; // [rsp+Ch] [rbp-14h]
  unsigned int v393; // [rsp+Ch] [rbp-14h]
  unsigned int v394; // [rsp+Ch] [rbp-14h]
  unsigned int v395; // [rsp+Ch] [rbp-14h]
  unsigned int v396; // [rsp+Ch] [rbp-14h]
  unsigned int v397; // [rsp+Ch] [rbp-14h]
  unsigned int v398; // [rsp+Ch] [rbp-14h]
  unsigned int v399; // [rsp+Ch] [rbp-14h]
  unsigned int v400; // [rsp+Ch] [rbp-14h]
  unsigned int v401; // [rsp+Ch] [rbp-14h]
  unsigned int v402; // [rsp+Ch] [rbp-14h]
  unsigned int v403; // [rsp+Ch] [rbp-14h]
  unsigned int v404; // [rsp+Ch] [rbp-14h]
  unsigned int v405; // [rsp+Ch] [rbp-14h]
  unsigned int v406; // [rsp+Ch] [rbp-14h]
  unsigned int v407; // [rsp+Ch] [rbp-14h]
  unsigned int v408; // [rsp+Ch] [rbp-14h]
  unsigned int v409; // [rsp+Ch] [rbp-14h]
  unsigned int v410; // [rsp+Ch] [rbp-14h]
  unsigned int v411; // [rsp+Ch] [rbp-14h]
  unsigned int v412; // [rsp+Ch] [rbp-14h]
  unsigned int v413; // [rsp+Ch] [rbp-14h]
  unsigned int v414; // [rsp+Ch] [rbp-14h]
  unsigned int v415; // [rsp+Ch] [rbp-14h]
  unsigned int v416; // [rsp+Ch] [rbp-14h]
  unsigned int v417; // [rsp+Ch] [rbp-14h]
  unsigned int v418; // [rsp+Ch] [rbp-14h]
  unsigned int v419; // [rsp+Ch] [rbp-14h]
  unsigned int v420; // [rsp+Ch] [rbp-14h]
  unsigned int v421; // [rsp+Ch] [rbp-14h]
  unsigned int v422; // [rsp+Ch] [rbp-14h]
  unsigned int v423; // [rsp+Ch] [rbp-14h]
  unsigned int v424; // [rsp+Ch] [rbp-14h]
  unsigned int v425; // [rsp+Ch] [rbp-14h]
  unsigned int v426; // [rsp+Ch] [rbp-14h]
  unsigned int v427; // [rsp+Ch] [rbp-14h]
  unsigned int v428; // [rsp+Ch] [rbp-14h]
  unsigned int v429; // [rsp+Ch] [rbp-14h]
  unsigned int v430; // [rsp+Ch] [rbp-14h]
  unsigned int v431; // [rsp+Ch] [rbp-14h]
  unsigned int v432; // [rsp+Ch] [rbp-14h]
  unsigned int v433; // [rsp+Ch] [rbp-14h]
  unsigned int v434; // [rsp+Ch] [rbp-14h]
  unsigned int v435; // [rsp+Ch] [rbp-14h]
  unsigned int v436; // [rsp+Ch] [rbp-14h]
  unsigned int v437; // [rsp+Ch] [rbp-14h]
  unsigned int v438; // [rsp+Ch] [rbp-14h]
  unsigned int v439; // [rsp+Ch] [rbp-14h]
  unsigned int v440; // [rsp+Ch] [rbp-14h]
  unsigned int v441; // [rsp+Ch] [rbp-14h]
  unsigned int v442; // [rsp+Ch] [rbp-14h]

  switch ( *(_BYTE *)(a1 + 24) )
  {
    case 1:
      v185 = a2[1];
      v186 = a2[2];
      v187 = (char *)*a2;
      if ( v185 + 12 <= v186 )
        goto LABEL_206;
      v188 = 2 * v186;
      if ( v185 + 1004 > v188 )
        a2[2] = v185 + 1004;
      else
        a2[2] = v188;
      v412 = a3;
      v189 = realloc(v187);
      *a2 = v189;
      v187 = (char *)v189;
      if ( !v189 )
        goto LABEL_451;
      v185 = a2[1];
      a3 = v412;
LABEL_206:
      qmemcpy(&v187[v185], "operator new", 12);
      a2[1] += 12;
      break;
    case 2:
      v180 = a2[1];
      v181 = a2[2];
      v182 = (char *)*a2;
      if ( v180 + 15 <= v181 )
        goto LABEL_200;
      v183 = 2 * v181;
      if ( v180 + 1007 > v183 )
        a2[2] = v180 + 1007;
      else
        a2[2] = v183;
      v411 = a3;
      v184 = realloc(v182);
      *a2 = v184;
      v182 = (char *)v184;
      if ( !v184 )
        goto LABEL_451;
      v180 = a2[1];
      a3 = v411;
LABEL_200:
      qmemcpy(&v182[v180], "operator delete", 15);
      a2[1] += 15;
      break;
    case 3:
      v174 = a2[1];
      v175 = a2[2];
      v176 = (char *)*a2;
      if ( v174 + 9 <= v175 )
        goto LABEL_194;
      v177 = 2 * v175;
      if ( v174 + 1001 > v177 )
        a2[2] = v174 + 1001;
      else
        a2[2] = v177;
      v410 = a3;
      v178 = realloc(v176);
      *a2 = v178;
      v176 = (char *)v178;
      if ( !v178 )
        goto LABEL_451;
      v174 = a2[1];
      a3 = v410;
LABEL_194:
      v179 = &v176[v174];
      *(_QWORD *)v179 = 0x726F74617265706FLL;
      v179[8] = 61;
      a2[1] += 9;
      break;
    case 4:
      v169 = a2[1];
      v170 = a2[2];
      v171 = (char *)*a2;
      if ( v169 + 10 <= v170 )
        goto LABEL_188;
      v172 = 2 * v170;
      if ( v169 + 1002 > v172 )
        a2[2] = v169 + 1002;
      else
        a2[2] = v172;
      v409 = a3;
      v173 = realloc(v171);
      *a2 = v173;
      v171 = (char *)v173;
      if ( !v173 )
        goto LABEL_451;
      v169 = a2[1];
      a3 = v409;
LABEL_188:
      qmemcpy(&v171[v169], "operator>>", 10);
      a2[1] += 10;
      break;
    case 5:
      v164 = a2[1];
      v165 = a2[2];
      v166 = (char *)*a2;
      if ( v164 + 10 <= v165 )
        goto LABEL_182;
      v167 = 2 * v165;
      if ( v164 + 1002 > v167 )
        a2[2] = v164 + 1002;
      else
        a2[2] = v167;
      v408 = a3;
      v168 = realloc(v166);
      *a2 = v168;
      v166 = (char *)v168;
      if ( !v168 )
        goto LABEL_451;
      v164 = a2[1];
      a3 = v408;
LABEL_182:
      qmemcpy(&v166[v164], "operator<<", 10);
      a2[1] += 10;
      break;
    case 6:
      v158 = a2[1];
      v159 = a2[2];
      v160 = (char *)*a2;
      if ( v158 + 9 <= v159 )
        goto LABEL_176;
      v161 = 2 * v159;
      if ( v158 + 1001 > v161 )
        a2[2] = v158 + 1001;
      else
        a2[2] = v161;
      v407 = a3;
      v162 = realloc(v160);
      *a2 = v162;
      v160 = (char *)v162;
      if ( !v162 )
        goto LABEL_451;
      v158 = a2[1];
      a3 = v407;
LABEL_176:
      v163 = &v160[v158];
      *(_QWORD *)v163 = 0x726F74617265706FLL;
      v163[8] = 33;
      a2[1] += 9;
      break;
    case 7:
      v153 = a2[1];
      v154 = a2[2];
      v155 = (char *)*a2;
      if ( v153 + 10 <= v154 )
        goto LABEL_170;
      v156 = 2 * v154;
      if ( v153 + 1002 > v156 )
        a2[2] = v153 + 1002;
      else
        a2[2] = v156;
      v406 = a3;
      v157 = realloc(v155);
      *a2 = v157;
      v155 = (char *)v157;
      if ( !v157 )
        goto LABEL_451;
      v153 = a2[1];
      a3 = v406;
LABEL_170:
      qmemcpy(&v155[v153], "operator==", 10);
      a2[1] += 10;
      break;
    case 8:
      v148 = a2[1];
      v149 = a2[2];
      v150 = (char *)*a2;
      if ( v148 + 10 <= v149 )
        goto LABEL_164;
      v151 = 2 * v149;
      if ( v148 + 1002 > v151 )
        a2[2] = v148 + 1002;
      else
        a2[2] = v151;
      v405 = a3;
      v152 = realloc(v150);
      *a2 = v152;
      v150 = (char *)v152;
      if ( !v152 )
        goto LABEL_451;
      v148 = a2[1];
      a3 = v405;
LABEL_164:
      qmemcpy(&v150[v148], "operator!=", 10);
      a2[1] += 10;
      break;
    case 9:
      v143 = a2[1];
      v144 = a2[2];
      v145 = (char *)*a2;
      if ( v143 + 10 <= v144 )
        goto LABEL_158;
      v146 = 2 * v144;
      if ( v143 + 1002 > v146 )
        a2[2] = v143 + 1002;
      else
        a2[2] = v146;
      v404 = a3;
      v147 = realloc(v145);
      *a2 = v147;
      v145 = (char *)v147;
      if ( !v147 )
        goto LABEL_451;
      v143 = a2[1];
      a3 = v404;
LABEL_158:
      qmemcpy(&v145[v143], "operator[]", 10);
      a2[1] += 10;
      break;
    case 0xA:
      v138 = a2[1];
      v139 = a2[2];
      v140 = (char *)*a2;
      if ( v138 + 10 <= v139 )
        goto LABEL_152;
      v141 = 2 * v139;
      if ( v138 + 1002 > v141 )
        a2[2] = v138 + 1002;
      else
        a2[2] = v141;
      v403 = a3;
      v142 = realloc(v140);
      *a2 = v142;
      v140 = (char *)v142;
      if ( !v142 )
        goto LABEL_451;
      v138 = a2[1];
      a3 = v403;
LABEL_152:
      qmemcpy(&v140[v138], "operator->", 10);
      a2[1] += 10;
      break;
    case 0xB:
      v132 = a2[1];
      v133 = a2[2];
      v134 = (char *)*a2;
      if ( v132 + 9 <= v133 )
        goto LABEL_146;
      v135 = 2 * v133;
      if ( v132 + 1001 > v135 )
        a2[2] = v132 + 1001;
      else
        a2[2] = v135;
      v402 = a3;
      v136 = realloc(v134);
      *a2 = v136;
      v134 = (char *)v136;
      if ( !v136 )
        goto LABEL_451;
      v132 = a2[1];
      a3 = v402;
LABEL_146:
      v137 = &v134[v132];
      *(_QWORD *)v137 = 0x726F74617265706FLL;
      v137[8] = 42;
      a2[1] += 9;
      break;
    case 0xC:
      v127 = a2[1];
      v128 = a2[2];
      v129 = (char *)*a2;
      if ( v127 + 10 <= v128 )
        goto LABEL_140;
      v130 = 2 * v128;
      if ( v127 + 1002 > v130 )
        a2[2] = v127 + 1002;
      else
        a2[2] = v130;
      v401 = a3;
      v131 = realloc(v129);
      *a2 = v131;
      v129 = (char *)v131;
      if ( !v131 )
        goto LABEL_451;
      v127 = a2[1];
      a3 = v401;
LABEL_140:
      qmemcpy(&v129[v127], "operator++", 10);
      a2[1] += 10;
      break;
    case 0xD:
      v122 = a2[1];
      v123 = a2[2];
      v124 = (char *)*a2;
      if ( v122 + 10 <= v123 )
        goto LABEL_134;
      v125 = 2 * v123;
      if ( v122 + 1002 > v125 )
        a2[2] = v122 + 1002;
      else
        a2[2] = v125;
      v400 = a3;
      v126 = realloc(v124);
      *a2 = v126;
      v124 = (char *)v126;
      if ( !v126 )
        goto LABEL_451;
      v122 = a2[1];
      a3 = v400;
LABEL_134:
      qmemcpy(&v124[v122], "operator--", 10);
      a2[1] += 10;
      break;
    case 0xE:
      v116 = a2[1];
      v117 = a2[2];
      v118 = (char *)*a2;
      if ( v116 + 9 <= v117 )
        goto LABEL_128;
      v119 = 2 * v117;
      if ( v116 + 1001 > v119 )
        a2[2] = v116 + 1001;
      else
        a2[2] = v119;
      v399 = a3;
      v120 = realloc(v118);
      *a2 = v120;
      v118 = (char *)v120;
      if ( !v120 )
        goto LABEL_451;
      v116 = a2[1];
      a3 = v399;
LABEL_128:
      v121 = &v118[v116];
      *(_QWORD *)v121 = 0x726F74617265706FLL;
      v121[8] = 45;
      a2[1] += 9;
      break;
    case 0xF:
      v110 = a2[1];
      v111 = a2[2];
      v112 = (char *)*a2;
      if ( v110 + 9 <= v111 )
        goto LABEL_122;
      v113 = 2 * v111;
      if ( v110 + 1001 > v113 )
        a2[2] = v110 + 1001;
      else
        a2[2] = v113;
      v398 = a3;
      v114 = realloc(v112);
      *a2 = v114;
      v112 = (char *)v114;
      if ( !v114 )
        goto LABEL_451;
      v110 = a2[1];
      a3 = v398;
LABEL_122:
      v115 = &v112[v110];
      *(_QWORD *)v115 = 0x726F74617265706FLL;
      v115[8] = 43;
      a2[1] += 9;
      break;
    case 0x10:
      v104 = a2[1];
      v105 = a2[2];
      v106 = (char *)*a2;
      if ( v104 + 9 <= v105 )
        goto LABEL_116;
      v107 = 2 * v105;
      if ( v104 + 1001 > v107 )
        a2[2] = v104 + 1001;
      else
        a2[2] = v107;
      v397 = a3;
      v108 = realloc(v106);
      *a2 = v108;
      v106 = (char *)v108;
      if ( !v108 )
        goto LABEL_451;
      v104 = a2[1];
      a3 = v397;
LABEL_116:
      v109 = &v106[v104];
      *(_QWORD *)v109 = 0x726F74617265706FLL;
      v109[8] = 38;
      a2[1] += 9;
      break;
    case 0x11:
      v99 = a2[1];
      v100 = a2[2];
      v101 = (char *)*a2;
      if ( v99 + 11 <= v100 )
        goto LABEL_110;
      v102 = 2 * v100;
      if ( v99 + 1003 > v102 )
        a2[2] = v99 + 1003;
      else
        a2[2] = v102;
      v396 = a3;
      v103 = realloc(v101);
      *a2 = v103;
      v101 = (char *)v103;
      if ( !v103 )
        goto LABEL_451;
      v99 = a2[1];
      a3 = v396;
LABEL_110:
      qmemcpy(&v101[v99], "operator->*", 11);
      a2[1] += 11;
      break;
    case 0x12:
      v93 = a2[1];
      v94 = a2[2];
      v95 = (char *)*a2;
      if ( v93 + 9 <= v94 )
        goto LABEL_104;
      v96 = 2 * v94;
      if ( v93 + 1001 > v96 )
        a2[2] = v93 + 1001;
      else
        a2[2] = v96;
      v395 = a3;
      v97 = realloc(v95);
      *a2 = v97;
      v95 = (char *)v97;
      if ( !v97 )
        goto LABEL_451;
      v93 = a2[1];
      a3 = v395;
LABEL_104:
      v98 = &v95[v93];
      *(_QWORD *)v98 = 0x726F74617265706FLL;
      v98[8] = 47;
      a2[1] += 9;
      break;
    case 0x13:
      v87 = a2[1];
      v88 = a2[2];
      v89 = (char *)*a2;
      if ( v87 + 9 <= v88 )
        goto LABEL_98;
      v90 = 2 * v88;
      if ( v87 + 1001 > v90 )
        a2[2] = v87 + 1001;
      else
        a2[2] = v90;
      v394 = a3;
      v91 = realloc(v89);
      *a2 = v91;
      v89 = (char *)v91;
      if ( !v91 )
        goto LABEL_451;
      v87 = a2[1];
      a3 = v394;
LABEL_98:
      v92 = &v89[v87];
      *(_QWORD *)v92 = 0x726F74617265706FLL;
      v92[8] = 37;
      a2[1] += 9;
      break;
    case 0x14:
      v81 = a2[1];
      v82 = a2[2];
      v83 = (char *)*a2;
      if ( v81 + 9 <= v82 )
        goto LABEL_92;
      v84 = 2 * v82;
      if ( v81 + 1001 > v84 )
        a2[2] = v81 + 1001;
      else
        a2[2] = v84;
      v393 = a3;
      v85 = realloc(v83);
      *a2 = v85;
      v83 = (char *)v85;
      if ( !v85 )
        goto LABEL_451;
      v81 = a2[1];
      a3 = v393;
LABEL_92:
      v86 = &v83[v81];
      *(_QWORD *)v86 = 0x726F74617265706FLL;
      v86[8] = 60;
      a2[1] += 9;
      break;
    case 0x15:
      v76 = a2[1];
      v77 = a2[2];
      v78 = (char *)*a2;
      if ( v76 + 10 <= v77 )
        goto LABEL_86;
      v79 = 2 * v77;
      if ( v76 + 1002 > v79 )
        a2[2] = v76 + 1002;
      else
        a2[2] = v79;
      v392 = a3;
      v80 = realloc(v78);
      *a2 = v80;
      v78 = (char *)v80;
      if ( !v80 )
        goto LABEL_451;
      v76 = a2[1];
      a3 = v392;
LABEL_86:
      qmemcpy(&v78[v76], "operator<=", 10);
      a2[1] += 10;
      break;
    case 0x16:
      v70 = a2[1];
      v71 = a2[2];
      v72 = (char *)*a2;
      if ( v70 + 9 <= v71 )
        goto LABEL_80;
      v73 = 2 * v71;
      if ( v70 + 1001 > v73 )
        a2[2] = v70 + 1001;
      else
        a2[2] = v73;
      v391 = a3;
      v74 = realloc(v72);
      *a2 = v74;
      v72 = (char *)v74;
      if ( !v74 )
        goto LABEL_451;
      v70 = a2[1];
      a3 = v391;
LABEL_80:
      v75 = &v72[v70];
      *(_QWORD *)v75 = 0x726F74617265706FLL;
      v75[8] = 62;
      a2[1] += 9;
      break;
    case 0x17:
      v65 = a2[1];
      v66 = a2[2];
      v67 = (char *)*a2;
      if ( v65 + 10 <= v66 )
        goto LABEL_74;
      v68 = 2 * v66;
      if ( v65 + 1002 > v68 )
        a2[2] = v65 + 1002;
      else
        a2[2] = v68;
      v390 = a3;
      v69 = realloc(v67);
      *a2 = v69;
      v67 = (char *)v69;
      if ( !v69 )
        goto LABEL_451;
      v65 = a2[1];
      a3 = v390;
LABEL_74:
      qmemcpy(&v67[v65], "operator>=", 10);
      a2[1] += 10;
      break;
    case 0x18:
      v59 = a2[1];
      v60 = a2[2];
      v61 = (char *)*a2;
      if ( v59 + 9 <= v60 )
        goto LABEL_68;
      v62 = 2 * v60;
      if ( v59 + 1001 > v62 )
        a2[2] = v59 + 1001;
      else
        a2[2] = v62;
      v389 = a3;
      v63 = realloc(v61);
      *a2 = v63;
      v61 = (char *)v63;
      if ( !v63 )
        goto LABEL_451;
      v59 = a2[1];
      a3 = v389;
LABEL_68:
      v64 = &v61[v59];
      *(_QWORD *)v64 = 0x726F74617265706FLL;
      v64[8] = 44;
      a2[1] += 9;
      break;
    case 0x19:
      v54 = a2[1];
      v55 = a2[2];
      v56 = (char *)*a2;
      if ( v54 + 10 <= v55 )
        goto LABEL_62;
      v57 = 2 * v55;
      if ( v54 + 1002 > v57 )
        a2[2] = v54 + 1002;
      else
        a2[2] = v57;
      v388 = a3;
      v58 = realloc(v56);
      *a2 = v58;
      v56 = (char *)v58;
      if ( !v58 )
        goto LABEL_451;
      v54 = a2[1];
      a3 = v388;
LABEL_62:
      qmemcpy(&v56[v54], "operator()", 10);
      a2[1] += 10;
      break;
    case 0x1A:
      v48 = a2[1];
      v49 = a2[2];
      v50 = (char *)*a2;
      if ( v48 + 9 <= v49 )
        goto LABEL_56;
      v51 = 2 * v49;
      if ( v48 + 1001 > v51 )
        a2[2] = v48 + 1001;
      else
        a2[2] = v51;
      v387 = a3;
      v52 = realloc(v50);
      *a2 = v52;
      v50 = (char *)v52;
      if ( !v52 )
        goto LABEL_451;
      v48 = a2[1];
      a3 = v387;
LABEL_56:
      v53 = &v50[v48];
      *(_QWORD *)v53 = 0x726F74617265706FLL;
      v53[8] = 126;
      a2[1] += 9;
      break;
    case 0x1B:
      v42 = a2[1];
      v43 = a2[2];
      v44 = (char *)*a2;
      if ( v42 + 9 <= v43 )
        goto LABEL_50;
      v45 = 2 * v43;
      if ( v42 + 1001 > v45 )
        a2[2] = v42 + 1001;
      else
        a2[2] = v45;
      v386 = a3;
      v46 = realloc(v44);
      *a2 = v46;
      v44 = (char *)v46;
      if ( !v46 )
        goto LABEL_451;
      v42 = a2[1];
      a3 = v386;
LABEL_50:
      v47 = &v44[v42];
      *(_QWORD *)v47 = 0x726F74617265706FLL;
      v47[8] = 94;
      a2[1] += 9;
      break;
    case 0x1C:
      v36 = a2[1];
      v37 = a2[2];
      v38 = (char *)*a2;
      if ( v36 + 9 <= v37 )
        goto LABEL_44;
      v39 = 2 * v37;
      if ( v36 + 1001 > v39 )
        a2[2] = v36 + 1001;
      else
        a2[2] = v39;
      v385 = a3;
      v40 = realloc(v38);
      *a2 = v40;
      v38 = (char *)v40;
      if ( !v40 )
        goto LABEL_451;
      v36 = a2[1];
      a3 = v385;
LABEL_44:
      v41 = &v38[v36];
      *(_QWORD *)v41 = 0x726F74617265706FLL;
      v41[8] = 124;
      a2[1] += 9;
      break;
    case 0x1D:
      v31 = a2[1];
      v32 = a2[2];
      v33 = (char *)*a2;
      if ( v31 + 10 <= v32 )
        goto LABEL_38;
      v34 = 2 * v32;
      if ( v31 + 1002 > v34 )
        a2[2] = v31 + 1002;
      else
        a2[2] = v34;
      v384 = a3;
      v35 = realloc(v33);
      *a2 = v35;
      v33 = (char *)v35;
      if ( !v35 )
        goto LABEL_451;
      v31 = a2[1];
      a3 = v384;
LABEL_38:
      qmemcpy(&v33[v31], "operator&&", 10);
      a2[1] += 10;
      break;
    case 0x1E:
      v26 = a2[1];
      v27 = a2[2];
      v28 = (char *)*a2;
      if ( v26 + 10 <= v27 )
        goto LABEL_32;
      v29 = 2 * v27;
      if ( v26 + 1002 > v29 )
        a2[2] = v26 + 1002;
      else
        a2[2] = v29;
      v383 = a3;
      v30 = realloc(v28);
      *a2 = v30;
      v28 = (char *)v30;
      if ( !v30 )
        goto LABEL_451;
      v26 = a2[1];
      a3 = v383;
LABEL_32:
      qmemcpy(&v28[v26], "operator||", 10);
      a2[1] += 10;
      break;
    case 0x1F:
      v21 = a2[1];
      v22 = a2[2];
      v23 = (char *)*a2;
      if ( v21 + 10 <= v22 )
        goto LABEL_26;
      v24 = 2 * v22;
      if ( v21 + 1002 > v24 )
        a2[2] = v21 + 1002;
      else
        a2[2] = v24;
      v382 = a3;
      v25 = realloc(v23);
      *a2 = v25;
      v23 = (char *)v25;
      if ( !v25 )
        goto LABEL_451;
      v21 = a2[1];
      a3 = v382;
LABEL_26:
      qmemcpy(&v23[v21], "operator*=", 10);
      a2[1] += 10;
      break;
    case 0x20:
      v16 = a2[1];
      v17 = a2[2];
      v18 = (char *)*a2;
      if ( v16 + 10 <= v17 )
        goto LABEL_20;
      v19 = 2 * v17;
      if ( v16 + 1002 > v19 )
        a2[2] = v16 + 1002;
      else
        a2[2] = v19;
      v381 = a3;
      v20 = realloc(v18);
      *a2 = v20;
      v18 = (char *)v20;
      if ( !v20 )
        goto LABEL_451;
      v16 = a2[1];
      a3 = v381;
LABEL_20:
      qmemcpy(&v18[v16], "operator+=", 10);
      a2[1] += 10;
      break;
    case 0x21:
      v279 = a2[1];
      v280 = a2[2];
      v281 = (char *)*a2;
      if ( v279 + 10 <= v280 )
        goto LABEL_302;
      v282 = 2 * v280;
      if ( v279 + 1002 > v282 )
        a2[2] = v279 + 1002;
      else
        a2[2] = v282;
      v428 = a3;
      v283 = realloc(v281);
      *a2 = v283;
      v281 = (char *)v283;
      if ( !v283 )
        goto LABEL_451;
      v279 = a2[1];
      a3 = v428;
LABEL_302:
      qmemcpy(&v281[v279], "operator-=", 10);
      a2[1] += 10;
      break;
    case 0x22:
      v274 = a2[1];
      v275 = a2[2];
      v276 = (char *)*a2;
      if ( v274 + 10 <= v275 )
        goto LABEL_296;
      v277 = 2 * v275;
      if ( v274 + 1002 > v277 )
        a2[2] = v274 + 1002;
      else
        a2[2] = v277;
      v427 = a3;
      v278 = realloc(v276);
      *a2 = v278;
      v276 = (char *)v278;
      if ( !v278 )
        goto LABEL_451;
      v274 = a2[1];
      a3 = v427;
LABEL_296:
      qmemcpy(&v276[v274], "operator/=", 10);
      a2[1] += 10;
      break;
    case 0x23:
      v269 = a2[1];
      v270 = a2[2];
      v271 = (char *)*a2;
      if ( v269 + 10 <= v270 )
        goto LABEL_290;
      v272 = 2 * v270;
      if ( v269 + 1002 > v272 )
        a2[2] = v269 + 1002;
      else
        a2[2] = v272;
      v426 = a3;
      v273 = realloc(v271);
      *a2 = v273;
      v271 = (char *)v273;
      if ( !v273 )
        goto LABEL_451;
      v269 = a2[1];
      a3 = v426;
LABEL_290:
      qmemcpy(&v271[v269], "operator%=", 10);
      a2[1] += 10;
      break;
    case 0x24:
      v264 = a2[1];
      v265 = a2[2];
      v266 = (char *)*a2;
      if ( v264 + 11 <= v265 )
        goto LABEL_284;
      v267 = 2 * v265;
      if ( v264 + 1003 > v267 )
        a2[2] = v264 + 1003;
      else
        a2[2] = v267;
      v425 = a3;
      v268 = realloc(v266);
      *a2 = v268;
      v266 = (char *)v268;
      if ( !v268 )
        goto LABEL_451;
      v264 = a2[1];
      a3 = v425;
LABEL_284:
      qmemcpy(&v266[v264], "operator>>=", 11);
      a2[1] += 11;
      break;
    case 0x25:
      v259 = a2[1];
      v260 = a2[2];
      v261 = (char *)*a2;
      if ( v259 + 11 <= v260 )
        goto LABEL_278;
      v262 = 2 * v260;
      if ( v259 + 1003 > v262 )
        a2[2] = v259 + 1003;
      else
        a2[2] = v262;
      v424 = a3;
      v263 = realloc(v261);
      *a2 = v263;
      v261 = (char *)v263;
      if ( !v263 )
        goto LABEL_451;
      v259 = a2[1];
      a3 = v424;
LABEL_278:
      qmemcpy(&v261[v259], "operator<<=", 11);
      a2[1] += 11;
      break;
    case 0x26:
      v254 = a2[1];
      v255 = a2[2];
      v256 = (char *)*a2;
      if ( v254 + 10 <= v255 )
        goto LABEL_272;
      v257 = 2 * v255;
      if ( v254 + 1002 > v257 )
        a2[2] = v254 + 1002;
      else
        a2[2] = v257;
      v423 = a3;
      v258 = realloc(v256);
      *a2 = v258;
      v256 = (char *)v258;
      if ( !v258 )
        goto LABEL_451;
      v254 = a2[1];
      a3 = v423;
LABEL_272:
      qmemcpy(&v256[v254], "operator&=", 10);
      a2[1] += 10;
      break;
    case 0x27:
      v249 = a2[1];
      v250 = a2[2];
      v251 = (char *)*a2;
      if ( v249 + 10 <= v250 )
        goto LABEL_266;
      v252 = 2 * v250;
      if ( v249 + 1002 > v252 )
        a2[2] = v249 + 1002;
      else
        a2[2] = v252;
      v422 = a3;
      v253 = realloc(v251);
      *a2 = v253;
      v251 = (char *)v253;
      if ( !v253 )
        goto LABEL_451;
      v249 = a2[1];
      a3 = v422;
LABEL_266:
      qmemcpy(&v251[v249], "operator|=", 10);
      a2[1] += 10;
      break;
    case 0x28:
      v244 = a2[1];
      v245 = a2[2];
      v246 = (char *)*a2;
      if ( v244 + 10 <= v245 )
        goto LABEL_260;
      v247 = 2 * v245;
      if ( v244 + 1002 > v247 )
        a2[2] = v244 + 1002;
      else
        a2[2] = v247;
      v421 = a3;
      v248 = realloc(v246);
      *a2 = v248;
      v246 = (char *)v248;
      if ( !v248 )
        goto LABEL_451;
      v244 = a2[1];
      a3 = v421;
LABEL_260:
      qmemcpy(&v246[v244], "operator^=", 10);
      a2[1] += 10;
      break;
    case 0x29:
      v239 = a2[1];
      v240 = a2[2];
      v241 = (char *)*a2;
      if ( v239 + 12 <= v240 )
        goto LABEL_254;
      v242 = 2 * v240;
      if ( v239 + 1004 > v242 )
        a2[2] = v239 + 1004;
      else
        a2[2] = v242;
      v420 = a3;
      v243 = realloc(v241);
      *a2 = v243;
      v241 = (char *)v243;
      if ( !v243 )
        goto LABEL_451;
      v239 = a2[1];
      a3 = v420;
LABEL_254:
      qmemcpy(&v241[v239], "`vbase dtor'", 12);
      a2[1] += 12;
      break;
    case 0x2A:
      v232 = a2[1];
      v233 = a2[2];
      v234 = (char *)*a2;
      if ( v232 + 22 <= v233 )
        goto LABEL_248;
      v235 = 2 * v233;
      if ( v232 + 1014 > v235 )
        a2[2] = v232 + 1014;
      else
        a2[2] = v235;
      v419 = a3;
      v236 = realloc(v234);
      *a2 = v236;
      v234 = (char *)v236;
      if ( !v236 )
        goto LABEL_451;
      v232 = a2[1];
      a3 = v419;
LABEL_248:
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F7CA90);
      v238 = (__m128i *)&v234[v232];
      v238[1].m128i_i32[0] = 1869898784;
      v238[1].m128i_i16[2] = 10098;
      *v238 = si128;
      a2[1] += 22;
      break;
    case 0x2B:
      v225 = a2[1];
      v226 = a2[2];
      v227 = (char *)*a2;
      if ( v225 + 22 <= v226 )
        goto LABEL_242;
      v228 = 2 * v226;
      if ( v225 + 1014 > v228 )
        a2[2] = v225 + 1014;
      else
        a2[2] = v228;
      v418 = a3;
      v229 = realloc(v227);
      *a2 = v229;
      v227 = (char *)v229;
      if ( !v229 )
        goto LABEL_451;
      v225 = a2[1];
      a3 = v418;
LABEL_242:
      v230 = _mm_load_si128((const __m128i *)&xmmword_3F7CAA0);
      v231 = (__m128i *)&v227[v225];
      v231[1].m128i_i32[0] = 1920299887;
      v231[1].m128i_i16[2] = 10085;
      *v231 = v230;
      a2[1] += 22;
      break;
    case 0x2C:
      v218 = a2[1];
      v219 = a2[2];
      v220 = (char *)*a2;
      if ( v218 + 22 <= v219 )
        goto LABEL_236;
      v221 = 2 * v219;
      if ( v218 + 1014 > v221 )
        a2[2] = v218 + 1014;
      else
        a2[2] = v221;
      v417 = a3;
      v222 = realloc(v220);
      *a2 = v222;
      v220 = (char *)v222;
      if ( !v222 )
        goto LABEL_451;
      v218 = a2[1];
      a3 = v417;
LABEL_236:
      v223 = _mm_load_si128((const __m128i *)&xmmword_3F7CAB0);
      v224 = (__m128i *)&v220[v218];
      v224[1].m128i_i32[0] = 1869898784;
      v224[1].m128i_i16[2] = 10098;
      *v224 = v223;
      a2[1] += 22;
      break;
    case 0x2D:
      v211 = a2[1];
      v212 = a2[2];
      v213 = (char *)*a2;
      if ( v211 + 22 <= v212 )
        goto LABEL_230;
      v214 = 2 * v212;
      if ( v211 + 1014 > v214 )
        a2[2] = v211 + 1014;
      else
        a2[2] = v214;
      v416 = a3;
      v215 = realloc(v213);
      *a2 = v215;
      v213 = (char *)v215;
      if ( !v215 )
        goto LABEL_451;
      v211 = a2[1];
      a3 = v416;
LABEL_230:
      v216 = _mm_load_si128((const __m128i *)&xmmword_3F7CAC0);
      v217 = (__m128i *)&v213[v211];
      v217[1].m128i_i32[0] = 1869898098;
      v217[1].m128i_i16[2] = 10098;
      *v217 = v216;
      a2[1] += 22;
      break;
    case 0x2E:
      v204 = a2[1];
      v205 = a2[2];
      v206 = (char *)*a2;
      if ( v204 + 22 <= v205 )
        goto LABEL_224;
      v207 = 2 * v205;
      if ( v204 + 1014 > v207 )
        a2[2] = v204 + 1014;
      else
        a2[2] = v207;
      v415 = a3;
      v208 = realloc(v206);
      *a2 = v208;
      v206 = (char *)v208;
      if ( !v208 )
        goto LABEL_451;
      v204 = a2[1];
      a3 = v415;
LABEL_224:
      v209 = _mm_load_si128((const __m128i *)&xmmword_3F7CAD0);
      v210 = (__m128i *)&v206[v204];
      v210[1].m128i_i32[0] = 1869898098;
      v210[1].m128i_i16[2] = 10098;
      *v210 = v209;
      a2[1] += 22;
      break;
    case 0x2F:
      v197 = a2[1];
      v198 = a2[2];
      v199 = (char *)*a2;
      if ( v197 + 28 <= v198 )
        goto LABEL_218;
      v200 = 2 * v198;
      if ( v197 + 1020 > v200 )
        a2[2] = v197 + 1020;
      else
        a2[2] = v200;
      v414 = a3;
      v201 = realloc(v199);
      *a2 = v201;
      v199 = (char *)v201;
      if ( !v201 )
        goto LABEL_451;
      v197 = a2[1];
      a3 = v414;
LABEL_218:
      v202 = _mm_load_si128((const __m128i *)&xmmword_3F7CAE0);
      v203 = (__m128i *)&v199[v197];
      qmemcpy(&v203[1], "or iterator'", 12);
      *v203 = v202;
      a2[1] += 28;
      break;
    case 0x30:
      v190 = a2[1];
      v191 = a2[2];
      v192 = (char *)*a2;
      if ( v190 + 26 <= v191 )
        goto LABEL_212;
      v193 = 2 * v191;
      if ( v190 + 1018 > v193 )
        a2[2] = v190 + 1018;
      else
        a2[2] = v193;
      v413 = a3;
      v194 = realloc(v192);
      *a2 = v194;
      v192 = (char *)v194;
      if ( !v194 )
        goto LABEL_451;
      v190 = a2[1];
      a3 = v413;
LABEL_212:
      v195 = _mm_load_si128((const __m128i *)&xmmword_3F7CAF0);
      v196 = (__m128i *)&v192[v190];
      qmemcpy(&v196[1], "ement map'", 10);
      *v196 = v195;
      a2[1] += 26;
      break;
    case 0x31:
      v331 = a2[1];
      v332 = a2[2];
      v333 = (char *)*a2;
      if ( v331 + 25 <= v332 )
        goto LABEL_350;
      v334 = 2 * v332;
      if ( v331 + 1017 > v334 )
        a2[2] = v331 + 1017;
      else
        a2[2] = v334;
      v436 = a3;
      v335 = realloc(v333);
      *a2 = v335;
      v333 = (char *)v335;
      if ( !v335 )
        goto LABEL_451;
      v331 = a2[1];
      a3 = v436;
LABEL_350:
      v336 = _mm_load_si128((const __m128i *)&xmmword_3F7CB00);
      v337 = (__m128i *)&v333[v331];
      v337[1].m128i_i64[0] = 0x726F746172657469LL;
      v337[1].m128i_i8[8] = 39;
      *v337 = v336;
      a2[1] += 25;
      break;
    case 0x32:
      v324 = a2[1];
      v325 = a2[2];
      v326 = (char *)*a2;
      if ( v324 + 25 <= v325 )
        goto LABEL_344;
      v327 = 2 * v325;
      if ( v324 + 1017 > v327 )
        a2[2] = v324 + 1017;
      else
        a2[2] = v327;
      v435 = a3;
      v328 = realloc(v326);
      *a2 = v328;
      v326 = (char *)v328;
      if ( !v328 )
        goto LABEL_451;
      v324 = a2[1];
      a3 = v435;
LABEL_344:
      v329 = _mm_load_si128((const __m128i *)&xmmword_3F7CB10);
      v330 = (__m128i *)&v326[v324];
      v330[1].m128i_i64[0] = 0x726F746172657469LL;
      v330[1].m128i_i8[8] = 39;
      *v330 = v329;
      a2[1] += 25;
      break;
    case 0x33:
      v317 = a2[1];
      v318 = a2[2];
      v319 = (char *)*a2;
      if ( v317 + 31 <= v318 )
        goto LABEL_338;
      v320 = 2 * v318;
      if ( v317 + 1023 > v320 )
        a2[2] = v317 + 1023;
      else
        a2[2] = v320;
      v434 = a3;
      v321 = realloc(v319);
      *a2 = v321;
      v319 = (char *)v321;
      if ( !v321 )
        goto LABEL_451;
      v317 = a2[1];
      a3 = v434;
LABEL_338:
      v322 = _mm_load_si128((const __m128i *)&xmmword_3F7CB20);
      v323 = (__m128i *)&v319[v317];
      qmemcpy(&v323[1], " ctor iterator'", 15);
      *v323 = v322;
      a2[1] += 31;
      break;
    case 0x34:
      v310 = a2[1];
      v311 = a2[2];
      v312 = (char *)*a2;
      if ( v310 + 19 <= v311 )
        goto LABEL_332;
      v313 = 2 * v311;
      if ( v310 + 1011 > v313 )
        a2[2] = v310 + 1011;
      else
        a2[2] = v313;
      v433 = a3;
      v314 = realloc(v312);
      *a2 = v314;
      v312 = (char *)v314;
      if ( !v314 )
        goto LABEL_451;
      v310 = a2[1];
      a3 = v433;
LABEL_332:
      v315 = _mm_load_si128((const __m128i *)&xmmword_3F7CB30);
      v316 = (__m128i *)&v312[v310];
      v316[1].m128i_i16[0] = 25970;
      v316[1].m128i_i8[2] = 39;
      *v316 = v315;
      a2[1] += 19;
      break;
    case 0x35:
      v303 = a2[1];
      v304 = a2[2];
      v305 = (char *)*a2;
      if ( v303 + 28 <= v304 )
        goto LABEL_326;
      v306 = 2 * v304;
      if ( v303 + 1020 > v306 )
        a2[2] = v303 + 1020;
      else
        a2[2] = v306;
      v432 = a3;
      v307 = realloc(v305);
      *a2 = v307;
      v305 = (char *)v307;
      if ( !v307 )
        goto LABEL_451;
      v303 = a2[1];
      a3 = v432;
LABEL_326:
      v308 = _mm_load_si128((const __m128i *)&xmmword_3F7CB40);
      v309 = (__m128i *)&v305[v303];
      qmemcpy(&v309[1], "tor closure'", 12);
      *v309 = v308;
      a2[1] += 28;
      break;
    case 0x36:
      v298 = a2[1];
      v299 = a2[2];
      v300 = (char *)*a2;
      if ( v298 + 14 <= v299 )
        goto LABEL_320;
      v301 = 2 * v299;
      if ( v298 + 1006 > v301 )
        a2[2] = v298 + 1006;
      else
        a2[2] = v301;
      v431 = a3;
      v302 = realloc(v300);
      *a2 = v302;
      v300 = (char *)v302;
      if ( !v302 )
        goto LABEL_451;
      v298 = a2[1];
      a3 = v431;
LABEL_320:
      qmemcpy(&v300[v298], "operator new[]", 14);
      a2[1] += 14;
      break;
    case 0x37:
      v291 = a2[1];
      v292 = a2[2];
      v293 = (char *)*a2;
      if ( v291 + 17 <= v292 )
        goto LABEL_314;
      v294 = 2 * v292;
      if ( v291 + 1009 > v294 )
        a2[2] = v291 + 1009;
      else
        a2[2] = v294;
      v430 = a3;
      v295 = realloc(v293);
      *a2 = v295;
      v293 = (char *)v295;
      if ( !v295 )
        goto LABEL_451;
      v291 = a2[1];
      a3 = v430;
LABEL_314:
      v296 = _mm_load_si128((const __m128i *)&xmmword_3F7CB50);
      v297 = (__m128i *)&v293[v291];
      v297[1].m128i_i8[0] = 93;
      *v297 = v296;
      a2[1] += 17;
      break;
    case 0x38:
      v284 = a2[1];
      v285 = a2[2];
      v286 = (char *)*a2;
      if ( v284 + 30 <= v285 )
        goto LABEL_308;
      v287 = 2 * v285;
      if ( v284 + 1022 > v287 )
        a2[2] = v284 + 1022;
      else
        a2[2] = v287;
      v429 = a3;
      v288 = realloc(v286);
      *a2 = v288;
      v286 = (char *)v288;
      if ( !v288 )
        goto LABEL_451;
      v284 = a2[1];
      a3 = v429;
LABEL_308:
      v289 = _mm_load_si128((const __m128i *)&xmmword_3F7CB60);
      v290 = (__m128i *)&v286[v284];
      qmemcpy(&v290[1], "ctor iterator'", 14);
      *v290 = v289;
      a2[1] += 30;
      break;
    case 0x39:
      v359 = a2[1];
      v360 = a2[2];
      v361 = (char *)*a2;
      if ( v359 + 30 <= v360 )
        goto LABEL_374;
      v362 = 2 * v360;
      if ( v359 + 1022 > v362 )
        a2[2] = v359 + 1022;
      else
        a2[2] = v362;
      v440 = a3;
      v363 = realloc(v361);
      *a2 = v363;
      v361 = (char *)v363;
      if ( !v363 )
        goto LABEL_451;
      v359 = a2[1];
      a3 = v440;
LABEL_374:
      v364 = _mm_load_si128((const __m128i *)&xmmword_3F7CB60);
      v365 = (__m128i *)&v361[v359];
      qmemcpy(&v365[1], "dtor iterator'", 14);
      *v365 = v364;
      a2[1] += 30;
      break;
    case 0x3A:
      v352 = a2[1];
      v353 = a2[2];
      v354 = (char *)*a2;
      if ( v352 + 30 <= v353 )
        goto LABEL_368;
      v355 = 2 * v353;
      if ( v352 + 1022 > v355 )
        a2[2] = v352 + 1022;
      else
        a2[2] = v355;
      v439 = a3;
      v356 = realloc(v354);
      *a2 = v356;
      v354 = (char *)v356;
      if ( !v356 )
        goto LABEL_451;
      v352 = a2[1];
      a3 = v439;
LABEL_368:
      v357 = _mm_load_si128((const __m128i *)&xmmword_3F7CB70);
      v358 = (__m128i *)&v354[v352];
      qmemcpy(&v358[1], "ctor iterator'", 14);
      *v358 = v357;
      a2[1] += 30;
      break;
    case 0x3B:
      v345 = a2[1];
      v346 = a2[2];
      v347 = (char *)*a2;
      if ( v345 + 36 <= v346 )
        goto LABEL_362;
      v348 = 2 * v346;
      if ( v345 + 1028 > v348 )
        a2[2] = v345 + 1028;
      else
        a2[2] = v348;
      v438 = a3;
      v349 = realloc(v347);
      *a2 = v349;
      v347 = (char *)v349;
      if ( !v349 )
        goto LABEL_451;
      v345 = a2[1];
      a3 = v438;
LABEL_362:
      v350 = _mm_load_si128((const __m128i *)&xmmword_3F7CB80);
      v351 = (__m128i *)&v347[v345];
      v351[2].m128i_i32[0] = 661811060;
      *v351 = v350;
      v351[1] = _mm_load_si128((const __m128i *)&xmmword_3F7CB90);
      a2[1] += 36;
      break;
    case 0x3C:
      v338 = a2[1];
      v339 = a2[2];
      v340 = (char *)*a2;
      if ( v338 + 27 <= v339 )
        goto LABEL_356;
      v341 = 2 * v339;
      if ( v338 + 1019 > v341 )
        a2[2] = v338 + 1019;
      else
        a2[2] = v341;
      v437 = a3;
      v342 = realloc(v340);
      *a2 = v342;
      v340 = (char *)v342;
      if ( !v342 )
        goto LABEL_451;
      v338 = a2[1];
      a3 = v437;
LABEL_356:
      v343 = _mm_load_si128((const __m128i *)&xmmword_3F7CBA0);
      v344 = (__m128i *)&v340[v338];
      qmemcpy(&v344[1], "r iterator'", 11);
      *v344 = v343;
      a2[1] += 27;
      break;
    case 0x3D:
      v372 = a2[1];
      v373 = a2[2];
      v374 = (char *)*a2;
      if ( v372 + 40 <= v373 )
        goto LABEL_386;
      v375 = 2 * v373;
      if ( v372 + 1032 > v375 )
        a2[2] = v372 + 1032;
      else
        a2[2] = v375;
      v442 = a3;
      v376 = realloc(v374);
      *a2 = v376;
      v374 = (char *)v376;
      if ( !v376 )
        goto LABEL_451;
      v372 = a2[1];
      a3 = v442;
LABEL_386:
      v377 = _mm_load_si128((const __m128i *)&xmmword_3F7CBB0);
      v378 = (__m128i *)&v374[v372];
      v378[2].m128i_i64[0] = 0x27726F7461726574LL;
      *v378 = v377;
      v378[1] = _mm_load_si128((const __m128i *)&xmmword_3F7CBC0);
      a2[1] += 40;
      break;
    case 0x3E:
      v366 = a2[1];
      v367 = a2[2];
      v368 = (char *)*a2;
      if ( v366 + 48 <= v367 )
        goto LABEL_380;
      v369 = 2 * v367;
      if ( v366 + 1040 > v369 )
        a2[2] = v366 + 1040;
      else
        a2[2] = v369;
      v441 = a3;
      v370 = realloc(v368);
      *a2 = v370;
      v368 = (char *)v370;
      if ( !v370 )
        goto LABEL_451;
      v366 = a2[1];
      a3 = v441;
LABEL_380:
      v371 = (__m128i *)&v368[v366];
      *v371 = _mm_load_si128((const __m128i *)&xmmword_3F7CB60);
      v371[1] = _mm_load_si128((const __m128i *)&xmmword_3F7CBD0);
      v371[2] = _mm_load_si128((const __m128i *)&xmmword_3F7CBE0);
      a2[1] += 48;
      break;
    case 0x3F:
      v9 = a2[1];
      v10 = a2[2];
      v11 = (char *)*a2;
      if ( v9 + 17 <= v10 )
        goto LABEL_14;
      v12 = 2 * v10;
      if ( v9 + 1009 > v12 )
        a2[2] = v9 + 1009;
      else
        a2[2] = v12;
      v380 = a3;
      v13 = realloc(v11);
      *a2 = v13;
      v11 = (char *)v13;
      if ( !v13 )
        goto LABEL_451;
      v9 = a2[1];
      a3 = v380;
LABEL_14:
      v14 = _mm_load_si128((const __m128i *)&xmmword_3F7CBF0);
      v15 = (__m128i *)&v11[v9];
      v15[1].m128i_i8[0] = 116;
      *v15 = v14;
      a2[1] += 17;
      break;
    case 0x40:
      v4 = a2[1];
      v5 = a2[2];
      v6 = (char *)*a2;
      if ( v4 + 11 <= v5 )
        goto LABEL_7;
      v7 = 2 * v5;
      if ( v4 + 1003 > v7 )
        a2[2] = v4 + 1003;
      else
        a2[2] = v7;
      v379 = a3;
      v8 = realloc(v6);
      *a2 = v8;
      v6 = (char *)v8;
      if ( !v8 )
LABEL_451:
        abort();
      v4 = a2[1];
      a3 = v379;
LABEL_7:
      qmemcpy(&v6[v4], "operator<=>", 11);
      a2[1] += 11;
      break;
    default:
      break;
  }
  sub_E2EB40(a1, (__int64)a2, a3);
}
