// Function: sub_3122A50
// Address: 0x3122a50
//
void __fastcall sub_3122A50(__int64 *a1, int a2, __int64 a3)
{
  unsigned __int64 v3; // r15
  __int64 *v5; // r12
  __int64 v6; // rax
  bool v7; // zf
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r8
  int v11; // eax
  __int64 *v12; // rdx
  __int64 v13; // r15
  __int64 v14; // r14
  __int64 v15; // r8
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 *v19; // rdx
  __int64 *v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 *v26; // rsi
  __int64 v27; // rbx
  __int64 v28; // rax
  __int64 *v29; // rdx
  __int64 *v30; // rsi
  __int64 v31; // rax
  __int64 v32; // r14
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 *v35; // rdx
  __int64 *v36; // rsi
  __int64 v37; // r14
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 *v40; // rdx
  __int64 *v41; // rsi
  __int64 v42; // r14
  __int64 v43; // rax
  __int64 *v44; // rdx
  __int64 *v45; // rsi
  __int64 v46; // rax
  __int64 v47; // rbx
  __int64 v48; // r14
  __int64 v49; // rax
  __int64 *v50; // rdx
  __int64 *v51; // rsi
  __int64 v52; // rax
  __int64 v53; // rbx
  __int64 v54; // rax
  __int64 *v55; // rdx
  __int64 *v56; // rsi
  __int64 v57; // rax
  __int64 v58; // rbx
  __int64 v59; // rax
  __int64 *v60; // rdx
  __int64 *v61; // rsi
  __int64 v62; // rax
  __int64 v63; // rbx
  __int64 v64; // rax
  __int64 *v65; // rdx
  __int64 *v66; // rsi
  __int64 v67; // rax
  __int64 v68; // rbx
  __int64 v69; // rax
  __int64 *v70; // rdx
  __int64 *v71; // rsi
  __int64 v72; // rax
  __int64 v73; // rbx
  __int64 v74; // rax
  __int64 *v75; // rdx
  __int64 *v76; // rsi
  __int64 v77; // r14
  __int64 v78; // rax
  __int64 *v79; // rdx
  __int64 *v80; // rsi
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // r14
  __int64 v84; // r14
  __int64 v85; // rax
  __int64 *v86; // rdx
  __int64 *v87; // rsi
  __int64 v88; // r14
  __int64 v89; // r14
  __int64 v90; // rax
  __int64 *v91; // rdx
  __int64 *v92; // rsi
  __int64 v93; // r14
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 *v96; // rdx
  __int64 *v97; // rsi
  __int64 v98; // r14
  __int64 v99; // r14
  __int64 v100; // rax
  __int64 *v101; // rdx
  __int64 *v102; // rsi
  __int64 v103; // r14
  __int64 v104; // rax
  __int64 *v105; // rdx
  __int64 *v106; // rsi
  __int64 v107; // r14
  __int64 v108; // rax
  __int64 *v109; // rdx
  __int64 *v110; // rsi
  __int64 v111; // r14
  __int64 v112; // r14
  __int64 v113; // rax
  __int64 *v114; // rdx
  __int64 *v115; // rsi
  __int64 v116; // r14
  __int64 v117; // r14
  __int64 v118; // rax
  __int64 *v119; // rdx
  __int64 *v120; // rsi
  __int64 v121; // r14
  __int64 v122; // rax
  __int64 v123; // rax
  __int64 *v124; // rdx
  __int64 *v125; // rsi
  __int64 v126; // r14
  __int64 v127; // rax
  __int64 v128; // rax
  __int64 *v129; // rdx
  __int64 *v130; // rsi
  __int64 v131; // r14
  __int64 *v132; // rdx
  __int64 *v133; // rsi
  __int64 v134; // r14
  __int64 v135; // rax
  __int64 *v136; // rdx
  __int64 *v137; // rsi
  __int64 v138; // r14
  __int64 v139; // rax
  __int64 v140; // rax
  __int64 *v141; // rdx
  __int64 *v142; // rsi
  __int64 v143; // r14
  __int64 *v144; // rdx
  __int64 *v145; // rsi
  __int64 v146; // r14
  __int64 *v147; // rdx
  __int64 *v148; // rsi
  __int64 v149; // r14
  __int64 *v150; // rdx
  __int64 *v151; // rsi
  __int64 v152; // r14
  __int64 v153; // rax
  __int64 *v154; // rdx
  __int64 *v155; // rsi
  __int64 v156; // r14
  __int64 v157; // rax
  __int64 *v158; // rdx
  __int64 *v159; // rsi
  __int64 v160; // rbx
  __int64 *v161; // r14
  __int64 v162; // rax
  __int64 *v163; // rdx
  __int64 *v164; // rsi
  __int64 v165; // r14
  __int64 *v166; // rdx
  __int64 *v167; // rsi
  __int64 v168; // r14
  __int64 *v169; // rdx
  __int64 *v170; // rsi
  __int64 v171; // r14
  __int64 *v172; // rdx
  __int64 *v173; // rsi
  __int64 v174; // r14
  __int64 *v175; // rdx
  __int64 *v176; // rsi
  __int64 v177; // r14
  __int64 v178; // rax
  __int64 *v179; // rdx
  __int64 *v180; // rsi
  __int64 v181; // r14
  __int64 *v182; // rdx
  __int64 *v183; // rsi
  __int64 v184; // r14
  __int64 v185; // rax
  __int64 *v186; // rdx
  __int64 *v187; // rsi
  __int64 v188; // r14
  __int64 *v189; // rdx
  __int64 *v190; // rsi
  __int64 v191; // r14
  __int64 *v192; // rdx
  __int64 *v193; // rsi
  __int64 v194; // r14
  __int64 v195; // rax
  __int64 *v196; // rdx
  __int64 *v197; // rsi
  __int64 v198; // rbx
  __int64 v199; // rax
  __int64 *v200; // rdx
  __int64 *v201; // rsi
  __int64 v202; // r14
  __int64 v203; // rax
  __int64 *v204; // rdx
  __int64 *v205; // rsi
  __int64 v206; // r14
  __int64 v207; // rax
  __int64 *v208; // rdx
  __int64 *v209; // rsi
  __int64 v210; // r14
  __int64 v211; // r8
  __int64 v212; // r12
  __int64 *v213; // rdx
  __int64 *v214; // rsi
  __int64 v215; // r12
  __int64 v216; // rbx
  __int64 *v217; // rsi
  __int64 *v218; // rdx
  __int64 v219; // r12
  __int64 v220; // rbx
  __int64 *v221; // rsi
  __int64 *v222; // rdx
  __int64 v223; // r12
  __int64 v224; // rbx
  __int64 *v225; // rsi
  __int64 *v226; // rdx
  __int64 v227; // r12
  __int64 v228; // rbx
  __int64 *v229; // rsi
  __int64 *v230; // rdx
  __int64 v231; // r15
  __int64 v232; // rbx
  __int64 *v233; // rsi
  __int64 *v234; // rdx
  __int64 v235; // rbx
  __int64 *v236; // rsi
  __int64 *v237; // rdx
  __int64 v238; // r12
  __int64 v239; // rbx
  __int64 *v240; // rsi
  __int64 *v241; // rdx
  __int64 v242; // r12
  __int64 v243; // rbx
  __int64 *v244; // rsi
  __int64 *v245; // rdx
  __int64 v246; // rbx
  __int64 *v247; // rsi
  __int64 *v248; // rdx
  __int64 v249; // rbx
  __int64 *v250; // rsi
  __int64 *v251; // rdx
  __int64 v252; // rbx
  __int64 *v253; // rsi
  __int64 *v254; // rdx
  __int64 v255; // rbx
  __int64 *v256; // rsi
  __int64 *v257; // rdx
  __int64 v258; // r14
  __int64 *v259; // rdx
  __int64 *v260; // rsi
  __int64 v261; // r14
  __int64 *v262; // rdx
  __int64 *v263; // rsi
  __int64 v264; // r14
  __int64 *v265; // rsi
  __int64 *v266; // rdx
  __int64 v267; // r15
  __int64 *v268; // rdx
  __int64 *v269; // rsi
  __int64 v270; // r13
  __int64 v271; // rbx
  __int64 *v272; // rdx
  __int64 *v273; // rsi
  __int64 v274; // rbx
  __int64 *v275; // rdx
  __int64 *v276; // rsi
  __int64 j; // r12
  __int64 *v278; // rsi
  __int64 *v279; // rdx
  __int64 v280; // r12
  __int64 *v281; // rdx
  __int64 *v282; // rsi
  __int64 v283; // rbx
  __int64 v284; // r14
  __int64 v285; // rax
  __int64 *v286; // rdx
  __int64 *v287; // rsi
  __int64 v288; // rbx
  __int64 v289; // r14
  __int64 v290; // rax
  __int64 *v291; // rdx
  __int64 *v292; // rsi
  __int64 v293; // r12
  __int64 v294; // rbx
  __int64 *v295; // rdx
  __int64 *v296; // rsi
  __int64 v297; // r12
  __int64 v298; // rbx
  __int64 *v299; // rdx
  __int64 *v300; // rsi
  __int64 v301; // rbx
  __int64 v302; // r14
  __int64 v303; // rax
  __int64 *v304; // rdx
  __int64 *v305; // rsi
  __int64 v306; // rbx
  __int64 v307; // r14
  __int64 v308; // rax
  __int64 *v309; // rdx
  __int64 *v310; // rsi
  __int64 v311; // r12
  __int64 v312; // rbx
  __int64 *v313; // rdx
  __int64 *v314; // rsi
  __int64 v315; // r12
  __int64 v316; // rbx
  __int64 *v317; // rdx
  __int64 *v318; // rsi
  __int64 v319; // r14
  __int64 v320; // rax
  __int64 *v321; // rdx
  __int64 *v322; // rsi
  __int64 v323; // r14
  __int64 v324; // rax
  __int64 *v325; // rdx
  __int64 *v326; // rsi
  __int64 v327; // rbx
  __int64 v328; // rax
  __int64 *v329; // rdx
  __int64 *v330; // rsi
  __int64 v331; // rbx
  __int64 v332; // rax
  __int64 *v333; // rdx
  __int64 *v334; // rsi
  __int64 v335; // r14
  __int64 v336; // rax
  __int64 *v337; // rdx
  __int64 *v338; // rsi
  __int64 v339; // r14
  __int64 v340; // rax
  __int64 *v341; // rdx
  __int64 *v342; // rsi
  __int64 v343; // r14
  __int64 v344; // rax
  __int64 *v345; // rdx
  __int64 *v346; // rsi
  __int64 v347; // r14
  __int64 *v348; // rdx
  __int64 *v349; // rsi
  __int64 v350; // rbx
  __int64 v351; // rax
  __int64 *v352; // rdx
  __int64 *v353; // rsi
  __int64 i; // rbx
  __int64 v355; // rax
  __int64 *v356; // rdx
  __int64 *v357; // rsi
  __int64 v358; // r14
  __int64 *v359; // rdx
  __int64 *v360; // rsi
  __int64 v361; // r14
  __int64 *v362; // rdx
  __int64 *v363; // rsi
  __int64 v364; // r14
  __int64 *v365; // rdx
  __int64 *v366; // rsi
  __int64 v367; // r14
  __int64 *v368; // rdx
  __int64 *v369; // rsi
  __int64 v370; // r14
  __int64 v371; // rax
  __int64 *v372; // rdx
  __int64 *v373; // rsi
  __int64 v374; // r14
  __int64 *v375; // rdx
  __int64 *v376; // rsi
  __int64 v377; // r14
  __int64 *v378; // rdx
  __int64 *v379; // rsi
  __int64 v380; // [rsp+0h] [rbp-1B0h]
  __int64 v381; // [rsp+8h] [rbp-1A8h]
  __int64 v382; // [rsp+10h] [rbp-1A0h]
  __int64 v383; // [rsp+18h] [rbp-198h]
  __int64 v384; // [rsp+20h] [rbp-190h]
  __int64 v385; // [rsp+28h] [rbp-188h]
  __int64 v386; // [rsp+30h] [rbp-180h]
  __int64 v387; // [rsp+38h] [rbp-178h]
  __int64 v388; // [rsp+40h] [rbp-170h]
  __int64 v389; // [rsp+48h] [rbp-168h]
  __int64 v390; // [rsp+50h] [rbp-160h]
  __int64 v391; // [rsp+50h] [rbp-160h]
  __int64 v392; // [rsp+50h] [rbp-160h]
  __int64 v393; // [rsp+50h] [rbp-160h]
  __int64 v394; // [rsp+50h] [rbp-160h]
  __int64 v395; // [rsp+50h] [rbp-160h]
  __int64 v396; // [rsp+50h] [rbp-160h]
  __int64 v397; // [rsp+60h] [rbp-150h]
  __int64 v398; // [rsp+60h] [rbp-150h]
  __int64 v399; // [rsp+60h] [rbp-150h]
  __int64 v400; // [rsp+60h] [rbp-150h]
  __int64 v401; // [rsp+60h] [rbp-150h]
  __int64 v402; // [rsp+60h] [rbp-150h]
  __int64 v403; // [rsp+60h] [rbp-150h]
  __int64 v404; // [rsp+68h] [rbp-148h]
  __int64 v405; // [rsp+68h] [rbp-148h]
  __int64 v406; // [rsp+68h] [rbp-148h]
  __int64 v407; // [rsp+68h] [rbp-148h]
  __int64 v408; // [rsp+68h] [rbp-148h]
  __int64 *v409; // [rsp+68h] [rbp-148h]
  __int64 *v410; // [rsp+68h] [rbp-148h]
  __int64 v411; // [rsp+70h] [rbp-140h] BYREF
  unsigned __int64 v412; // [rsp+78h] [rbp-138h] BYREF
  unsigned __int64 v413; // [rsp+80h] [rbp-130h] BYREF
  __int64 v414; // [rsp+88h] [rbp-128h] BYREF
  __int64 v415; // [rsp+90h] [rbp-120h] BYREF
  __int64 v416; // [rsp+98h] [rbp-118h] BYREF
  __int64 *v417[2]; // [rsp+A0h] [rbp-110h] BYREF
  _QWORD v418[2]; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v419; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v420; // [rsp+C8h] [rbp-E8h]
  __int64 *v421; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v422; // [rsp+D8h] [rbp-D8h]
  _BYTE v423[32]; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v424; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v425; // [rsp+108h] [rbp-A8h] BYREF
  __int64 v426; // [rsp+110h] [rbp-A0h]
  __int64 v427; // [rsp+118h] [rbp-98h]
  __int64 v428; // [rsp+120h] [rbp-90h]
  __int64 v429; // [rsp+128h] [rbp-88h]
  __int64 v430; // [rsp+130h] [rbp-80h]
  __int64 v431; // [rsp+138h] [rbp-78h]
  __int64 v432; // [rsp+140h] [rbp-70h]
  __int64 v433; // [rsp+148h] [rbp-68h]
  __int64 v434; // [rsp+150h] [rbp-60h]
  __int64 v435; // [rsp+158h] [rbp-58h]
  __int64 v436; // [rsp+160h] [rbp-50h]
  __int64 v437; // [rsp+168h] [rbp-48h]
  __int64 v438; // [rsp+170h] [rbp-40h]

  v3 = 0;
  v5 = (__int64 *)sub_B2BE50(a3);
  v411 = *(_QWORD *)(a3 + 120);
  v412 = sub_A74680(&v411);
  v6 = sub_A74610(&v411);
  v7 = *(_QWORD *)(a3 + 104) == 0;
  v413 = v6;
  v421 = (__int64 *)v423;
  v422 = 0x400000000LL;
  if ( !v7 )
  {
    do
    {
      v8 = sub_A744E0(&v411, v3);
      v9 = (unsigned int)v422;
      v10 = v8;
      v11 = v422;
      if ( (unsigned int)v422 >= (unsigned __int64)HIDWORD(v422) )
      {
        if ( HIDWORD(v422) < (unsigned __int64)(unsigned int)v422 + 1 )
        {
          v398 = v10;
          sub_C8D5F0((__int64)&v421, v423, (unsigned int)v422 + 1LL, 8u, v10, (unsigned int)v422 + 1LL);
          v9 = (unsigned int)v422;
          v10 = v398;
        }
        v421[v9] = v10;
        LODWORD(v422) = v422 + 1;
      }
      else
      {
        v12 = &v421[(unsigned int)v422];
        if ( v12 )
        {
          *v12 = v10;
          v11 = v422;
        }
        LODWORD(v422) = v11 + 1;
      }
      ++v3;
    }
    while ( *(_QWORD *)(a3 + 104) > v3 );
  }
  v417[1] = v5;
  v417[0] = a1;
  if ( (_BYTE)qword_5032A88 )
  {
    v424 = sub_A778C0(v5, 41, 0);
    v425 = sub_A778C0(v5, 39, 0);
    v426 = sub_A778C0(v5, 29, 0);
    v427 = sub_A778C0(v5, 76, 0);
    v428 = sub_A77AB0(v5, 4u);
    v388 = sub_A79C90(v5, &v424, 5);
  }
  else
  {
    v424 = sub_A778C0(v5, 41, 0);
    v388 = sub_A79C90(v5, &v424, 1);
  }
  if ( (_BYTE)qword_5032A88 )
  {
    v424 = sub_A778C0(v5, 41, 0);
    v425 = sub_A778C0(v5, 39, 0);
    v426 = sub_A778C0(v5, 29, 0);
    v427 = sub_A778C0(v5, 76, 0);
    v428 = sub_A77AB0(v5, 5u);
    v381 = sub_A79C90(v5, &v424, 5);
  }
  else
  {
    v424 = sub_A778C0(v5, 41, 0);
    v381 = sub_A79C90(v5, &v424, 1);
  }
  if ( (_BYTE)qword_5032A88 )
  {
    v424 = sub_A778C0(v5, 41, 0);
    v425 = sub_A778C0(v5, 39, 0);
    v426 = sub_A778C0(v5, 29, 0);
    v427 = sub_A778C0(v5, 76, 0);
    v428 = sub_A77AB0(v5, 7u);
    v13 = sub_A79C90(v5, &v424, 5);
  }
  else
  {
    v424 = sub_A778C0(v5, 41, 0);
    v13 = sub_A79C90(v5, &v424, 1);
  }
  if ( (_BYTE)qword_5032A88 )
  {
    v424 = sub_A778C0(v5, 41, 0);
    v425 = sub_A778C0(v5, 39, 0);
    v426 = sub_A778C0(v5, 29, 0);
    v427 = sub_A778C0(v5, 76, 0);
    v428 = sub_A77AB0(v5, 8u);
    v383 = sub_A79C90(v5, &v424, 5);
  }
  else
  {
    v424 = sub_A778C0(v5, 41, 0);
    v383 = sub_A79C90(v5, &v424, 1);
  }
  if ( (_BYTE)qword_5032A88 )
  {
    v424 = sub_A778C0(v5, 41, 0);
    v425 = sub_A778C0(v5, 39, 0);
    v426 = sub_A778C0(v5, 76, 0);
    v427 = sub_A778C0(v5, 29, 0);
    v387 = sub_A79C90(v5, &v424, 4);
  }
  else
  {
    v424 = sub_A778C0(v5, 41, 0);
    v387 = sub_A79C90(v5, &v424, 1);
  }
  v424 = sub_A778C0(v5, 41, 0);
  v425 = sub_A778C0(v5, 6, 0);
  v389 = sub_A79C90(v5, &v424, 2);
  if ( (_BYTE)qword_5032A88 )
  {
    v424 = sub_A778C0(v5, 41, 0);
    v425 = sub_A778C0(v5, 39, 0);
    v426 = sub_A778C0(v5, 29, 0);
    v427 = sub_A778C0(v5, 76, 0);
    v428 = sub_A77AB0(v5, 0xFu);
    v386 = sub_A79C90(v5, &v424, 5);
  }
  else
  {
    v424 = sub_A778C0(v5, 41, 0);
    v386 = sub_A79C90(v5, &v424, 1);
  }
  v424 = sub_A778C0(v5, 3, 0);
  v384 = sub_A79C90(v5, &v424, 1);
  if ( (_BYTE)qword_5032A88 )
  {
    v424 = sub_A778C0(v5, 41, 0);
    v425 = sub_A778C0(v5, 39, 0);
    v426 = sub_A778C0(v5, 76, 0);
    v382 = sub_A79C90(v5, &v424, 3);
  }
  else
  {
    v424 = sub_A778C0(v5, 41, 0);
    v382 = sub_A79C90(v5, &v424, 1);
  }
  v424 = sub_A778C0(v5, 41, 0);
  v385 = sub_A79C90(v5, &v424, 1);
  if ( (_BYTE)qword_5032A88 )
  {
    v424 = sub_A778C0(v5, 51, 0);
    v425 = sub_A778C0(v5, 29, 0);
    v426 = sub_A77AD0(v5, 0);
    v397 = sub_A79C90(v5, &v424, 3);
  }
  else
  {
    v397 = sub_A79C90(v5, 0, 0);
  }
  v424 = sub_A778C0(v5, 41, 0);
  v425 = sub_A778C0(v5, 39, 0);
  v380 = sub_A79C90(v5, &v424, 2);
  if ( (_BYTE)qword_5032A88 )
  {
    v424 = sub_A77AD0(v5, 0);
    v425 = sub_A778C0(v5, 29, 0);
    v14 = sub_A79C90(v5, &v424, 2);
  }
  else
  {
    v14 = sub_A79C90(v5, 0, 0);
  }
  v424 = sub_A778C0(v5, 22, 0);
  v414 = sub_A79C90(v5, &v424, 1);
  v424 = sub_A778C0(v5, 79, 0);
  v415 = sub_A79C90(v5, &v424, 1);
  v424 = sub_A778C0(v5, 54, 0);
  v416 = sub_A79C90(v5, &v424, 1);
  if ( *(_DWORD *)(sub_AE4420(a1[63] + 312, (__int64)v5, 0) + 8) > 0x3FFFu )
  {
    v15 = sub_A79C90(v5, 0, 0);
  }
  else
  {
    v424 = sub_A778C0(v5, 79, 0);
    v15 = sub_A79C90(v5, &v424, 1);
  }
  switch ( a2 )
  {
    case 0:
    case 59:
    case 60:
    case 97:
    case 100:
    case 101:
    case 128:
    case 187:
    case 188:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v389);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v425 = v416;
      v424 = v397;
      sub_3121310(v417, v421, &v424, 1);
      v424 = v397;
      v425 = v416;
      sub_3121310(v417, v421 + 1, &v425, 1);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 1:
      v377 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v386);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      do
      {
        v378 = (__int64 *)((char *)&v424 + v377 * 8);
        v424 = v397;
        v379 = &v421[v377++];
        v425 = v416;
        v426 = v416;
        sub_3121310(v417, v379, v378, 1);
      }
      while ( v377 != 3 );
      goto LABEL_30;
    case 2:
    case 9:
    case 96:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v389);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      v425 = v416;
      v424 = v397;
      sub_3121310(v417, v421, &v424, 1);
      v424 = v397;
      v425 = v416;
      sub_3121310(v417, v421 + 1, &v425, 1);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 3:
      v16 = 0;
      v17 = sub_A79C90(v5, 0, 0);
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v17);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      sub_A79C90(v5, 0, 0);
      do
      {
        v18 = sub_A79C90(v5, 0, 0);
        v19 = (__int64 *)((char *)&v424 + v16 * 8);
        v424 = v18;
        v20 = &v421[v16++];
        v425 = v416;
        sub_3121310(v417, v20, v19, 1);
        sub_A79C90(v5, 0, 0);
      }
      while ( v16 != 2 );
      goto LABEL_30;
    case 4:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v389);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v424 = v397;
      sub_3121310(v417, v421, &v424, 1);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 5:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v381);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      v424 = v397;
      sub_3121310(v417, v421, &v424, 1);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 6:
    case 14:
    case 15:
    case 16:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v388);
      sub_3121310(v417, (__int64 *)&v413, &v415, 0);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 7:
      v374 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v385);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      do
      {
        v375 = (__int64 *)((char *)&v424 + v374 * 8);
        v424 = v397;
        v376 = &v421[v374++];
        v426 = v397;
        v425 = v416;
        sub_3121310(v417, v376, v375, 1);
      }
      while ( v374 != 3 );
      goto LABEL_30;
    case 8:
      v370 = 0;
      v371 = sub_A79C90(v5, 0, 0);
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v371);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      do
      {
        v372 = (__int64 *)((char *)&v424 + v370 * 8);
        v424 = v397;
        v373 = &v421[v370++];
        v426 = v397;
        v425 = v416;
        v427 = v416;
        sub_3121310(v417, v373, v372, 1);
      }
      while ( v370 != 4 );
      goto LABEL_30;
    case 10:
      v367 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v386);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      do
      {
        v368 = (__int64 *)((char *)&v424 + v367 * 8);
        v424 = v397;
        v369 = &v421[v367++];
        v425 = v416;
        v426 = v416;
        sub_3121310(v417, v369, v368, 1);
      }
      while ( v367 != 3 );
      goto LABEL_30;
    case 11:
      v364 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v386);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      do
      {
        v365 = (__int64 *)((char *)&v424 + v364 * 8);
        v424 = v397;
        v366 = &v421[v364++];
        v425 = v416;
        v426 = v416;
        sub_3121310(v417, v366, v365, 1);
      }
      while ( v364 != 3 );
      goto LABEL_30;
    case 12:
      v361 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v386);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      do
      {
        v362 = (__int64 *)((char *)&v424 + v361 * 8);
        v424 = v397;
        v363 = &v421[v361++];
        v425 = v416;
        v426 = v416;
        sub_3121310(v417, v363, v362, 1);
      }
      while ( v361 != 3 );
      goto LABEL_30;
    case 13:
      v358 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      do
      {
        v359 = (__int64 *)((char *)&v424 + v358 * 8);
        v424 = v397;
        v360 = &v421[v358++];
        v425 = v416;
        v426 = v397;
        v427 = v416;
        v428 = v397;
        sub_3121310(v417, v360, v359, 1);
      }
      while ( v358 != 5 );
      goto LABEL_30;
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
    case 22:
    case 23:
    case 25:
    case 26:
    case 27:
    case 28:
    case 31:
    case 32:
    case 33:
    case 34:
    case 35:
    case 37:
    case 38:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v388);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 24:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v400 = a3;
      for ( i = 0; ; ++i )
      {
        v419 = sub_A77AD0(v5, 0);
        v420 = sub_A778C0(v5, 78, 0);
        sub_A79C90(v5, &v419, 2);
        v424 = sub_A77AD0(v5, 0);
        v425 = sub_A778C0(v5, 78, 0);
        sub_A79C90(v5, &v424, 2);
        if ( i == 2 )
          break;
        v418[0] = sub_A77AD0(v5, 0);
        v418[1] = sub_A778C0(v5, 78, 0);
        v424 = sub_A79C90(v5, v418, 2);
        v419 = sub_A77AD0(v5, 0);
        v420 = sub_A778C0(v5, 78, 0);
        v355 = sub_A79C90(v5, &v419, 2);
        v356 = (__int64 *)((char *)&v424 + i * 8);
        v425 = v355;
        v357 = &v421[i];
        sub_3121310(v417, v357, v356, 1);
      }
      goto LABEL_54;
    case 29:
    case 30:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v388);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      v424 = v416;
      sub_3121310(v417, v421, &v424, 1);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 36:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v424 = sub_A77AD0(v5, 0);
      v425 = sub_A778C0(v5, 78, 0);
      sub_A79C90(v5, &v424, 2);
      v404 = a3;
      v350 = 0;
      do
      {
        v424 = v416;
        v419 = sub_A77AD0(v5, 0);
        v420 = sub_A778C0(v5, 78, 0);
        v351 = sub_A79C90(v5, &v419, 2);
        v352 = (__int64 *)((char *)&v424 + v350 * 8);
        v425 = v351;
        v353 = &v421[v350++];
        sub_3121310(v417, v353, v352, 1);
        v424 = sub_A77AD0(v5, 0);
        v425 = sub_A778C0(v5, 78, 0);
        sub_A79C90(v5, &v424, 2);
      }
      while ( v350 != 2 );
      goto LABEL_300;
    case 39:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 40:
    case 178:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v388);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 41:
    case 42:
    case 43:
    case 45:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v383);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v424 = v416;
      sub_3121310(v417, v421, &v424, 1);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 44:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v383);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v424 = v416;
      v425 = v416;
      sub_3121310(v417, v421, &v424, 1);
      v424 = v416;
      v425 = v416;
      sub_3121310(v417, v421 + 1, &v425, 1);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 46:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v386);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      v425 = v416;
      v424 = v397;
      sub_3121310(v417, v421, &v424, 1);
      v424 = v397;
      v425 = v416;
      sub_3121310(v417, v421 + 1, &v425, 1);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 47:
    case 49:
    case 65:
    case 70:
    case 83:
    case 84:
    case 85:
    case 86:
    case 87:
    case 173:
    case 174:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v386);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v425 = v416;
      v424 = v397;
      sub_3121310(v417, v421, &v424, 1);
      v424 = v397;
      v425 = v416;
      sub_3121310(v417, v421 + 1, &v425, 1);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 48:
      v347 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v386);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      do
      {
        v348 = (__int64 *)((char *)&v424 + v347 * 8);
        v424 = v397;
        v349 = &v421[v347++];
        v425 = v416;
        v426 = v416;
        sub_3121310(v417, v349, v348, 1);
      }
      while ( v347 != 3 );
      goto LABEL_30;
    case 50:
      v343 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v389);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v426 = sub_A79C90(v5, 0, 0);
      do
      {
        v424 = v397;
        v425 = v416;
        v344 = sub_A79C90(v5, 0, 0);
        v345 = (__int64 *)((char *)&v424 + v343 * 8);
        v426 = v344;
        v346 = &v421[v343++];
        sub_3121310(v417, v346, v345, 1);
        v426 = sub_A79C90(v5, 0, 0);
      }
      while ( v343 != 3 );
      goto LABEL_30;
    case 51:
      v339 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v389);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v426 = sub_A79C90(v5, 0, 0);
      do
      {
        v424 = v397;
        v425 = v416;
        v340 = sub_A79C90(v5, 0, 0);
        v341 = (__int64 *)((char *)&v424 + v339 * 8);
        v426 = v340;
        v342 = &v421[v339++];
        v427 = v415;
        sub_3121310(v417, v342, v341, 1);
        v426 = sub_A79C90(v5, 0, 0);
      }
      while ( v339 != 4 );
      goto LABEL_30;
    case 52:
      v335 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v389);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v426 = sub_A79C90(v5, 0, 0);
      do
      {
        v424 = v397;
        v425 = v416;
        v336 = sub_A79C90(v5, 0, 0);
        v337 = (__int64 *)((char *)&v424 + v335 * 8);
        v426 = v336;
        v338 = &v421[v335++];
        sub_3121310(v417, v338, v337, 1);
        v426 = sub_A79C90(v5, 0, 0);
      }
      while ( v335 != 3 );
      goto LABEL_30;
    case 53:
    case 102:
    case 103:
    case 154:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v425 = v416;
      v424 = v397;
      sub_3121310(v417, v421, &v424, 1);
      v424 = v397;
      v425 = v416;
      sub_3121310(v417, v421 + 1, &v425, 1);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 54:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v424 = v397;
      sub_3121310(v417, v421, &v424, 1);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 55:
      v396 = v15;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v389);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      v408 = a3;
      v331 = 0;
      v429 = sub_A79C90(v5, 0, 0);
      v161 = v5;
      do
      {
        v427 = v396;
        v424 = v397;
        v428 = v397;
        v425 = v416;
        v426 = v416;
        v332 = sub_A79C90(v5, 0, 0);
        v333 = (__int64 *)((char *)&v424 + v331 * 8);
        v429 = v332;
        v334 = &v421[v331++];
        sub_3121310(v417, v334, v333, 1);
        v429 = sub_A79C90(v5, 0, 0);
      }
      while ( v331 != 6 );
      goto LABEL_318;
    case 56:
      v395 = v15;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v389);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      v408 = a3;
      v327 = 0;
      v429 = sub_A79C90(v5, 0, 0);
      v161 = v5;
      do
      {
        v427 = v395;
        v424 = v397;
        v428 = v397;
        v425 = v416;
        v426 = v416;
        v328 = sub_A79C90(v5, 0, 0);
        v329 = (__int64 *)((char *)&v424 + v327 * 8);
        v429 = v328;
        v330 = &v421[v327++];
        sub_3121310(v417, v330, v329, 1);
        v429 = sub_A79C90(v5, 0, 0);
      }
      while ( v327 != 6 );
      goto LABEL_318;
    case 57:
      v323 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v389);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v426 = sub_A79C90(v5, 0, 0);
      do
      {
        v424 = v397;
        v425 = v416;
        v324 = sub_A79C90(v5, 0, 0);
        v325 = (__int64 *)((char *)&v424 + v323 * 8);
        v426 = v324;
        v326 = &v421[v323++];
        sub_3121310(v417, v326, v325, 1);
        v426 = sub_A79C90(v5, 0, 0);
      }
      while ( v323 != 3 );
      goto LABEL_30;
    case 58:
      v319 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v389);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v426 = sub_A79C90(v5, 0, 0);
      do
      {
        v424 = v397;
        v425 = v416;
        v320 = sub_A79C90(v5, 0, 0);
        v321 = (__int64 *)((char *)&v424 + v319 * 8);
        v426 = v320;
        v322 = &v421[v319++];
        sub_3121310(v417, v322, v321, 1);
        v426 = sub_A79C90(v5, 0, 0);
      }
      while ( v319 != 3 );
      goto LABEL_30;
    case 61:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v410 = v5;
      v315 = v397;
      v403 = a3;
      v316 = 0;
      do
      {
        v317 = (__int64 *)((char *)&v424 + v316 * 8);
        v424 = v315;
        v318 = &v421[v316++];
        v425 = v416;
        v426 = v416;
        v427 = v14;
        v428 = v14;
        v429 = v14;
        v430 = v14;
        v431 = v416;
        v432 = v416;
        sub_3121310(v417, v318, v317, 1);
      }
      while ( v316 != 9 );
      goto LABEL_306;
    case 62:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v410 = v5;
      v311 = v397;
      v403 = a3;
      v312 = 0;
      do
      {
        v313 = (__int64 *)((char *)&v424 + v312 * 8);
        v424 = v311;
        v314 = &v421[v312++];
        v425 = v416;
        v426 = v416;
        v427 = v14;
        v428 = v14;
        v429 = v14;
        v430 = v14;
        v431 = v416;
        v432 = v416;
        sub_3121310(v417, v314, v313, 1);
      }
      while ( v312 != 9 );
      goto LABEL_306;
    case 63:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v431 = sub_A79C90(v5, 0, 0);
      v404 = a3;
      v306 = v14;
      v432 = sub_A79C90(v5, 0, 0);
      v307 = 0;
      do
      {
        v427 = v306;
        v424 = v397;
        v428 = v306;
        v425 = v416;
        v426 = v416;
        v429 = v306;
        v430 = v306;
        v431 = sub_A79C90(v5, 0, 0);
        v308 = sub_A79C90(v5, 0, 0);
        v309 = (__int64 *)((char *)&v424 + v307 * 8);
        v432 = v308;
        v310 = &v421[v307++];
        sub_3121310(v417, v310, v309, 1);
        v431 = sub_A79C90(v5, 0, 0);
        v432 = sub_A79C90(v5, 0, 0);
      }
      while ( v307 != 9 );
      goto LABEL_300;
    case 64:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v431 = sub_A79C90(v5, 0, 0);
      v404 = a3;
      v301 = v14;
      v432 = sub_A79C90(v5, 0, 0);
      v302 = 0;
      do
      {
        v427 = v301;
        v424 = v397;
        v428 = v301;
        v425 = v416;
        v426 = v416;
        v429 = v301;
        v430 = v301;
        v431 = sub_A79C90(v5, 0, 0);
        v303 = sub_A79C90(v5, 0, 0);
        v304 = (__int64 *)((char *)&v424 + v302 * 8);
        v432 = v303;
        v305 = &v421[v302++];
        sub_3121310(v417, v305, v304, 1);
        v431 = sub_A79C90(v5, 0, 0);
        v432 = sub_A79C90(v5, 0, 0);
      }
      while ( v302 != 9 );
      goto LABEL_300;
    case 66:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v410 = v5;
      v297 = v397;
      v403 = a3;
      v298 = 0;
      do
      {
        v299 = (__int64 *)((char *)&v424 + v298 * 8);
        v424 = v297;
        v300 = &v421[v298++];
        v425 = v416;
        v426 = v416;
        v427 = v14;
        v428 = v14;
        v429 = v14;
        v430 = v14;
        v431 = v416;
        v432 = v416;
        sub_3121310(v417, v300, v299, 1);
      }
      while ( v298 != 9 );
      goto LABEL_306;
    case 67:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v410 = v5;
      v293 = v397;
      v403 = a3;
      v294 = 0;
      do
      {
        v295 = (__int64 *)((char *)&v424 + v294 * 8);
        v424 = v293;
        v296 = &v421[v294++];
        v425 = v416;
        v426 = v416;
        v427 = v14;
        v428 = v14;
        v429 = v14;
        v430 = v14;
        v431 = v416;
        v432 = v416;
        sub_3121310(v417, v296, v295, 1);
      }
      while ( v294 != 9 );
      goto LABEL_306;
    case 68:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v431 = sub_A79C90(v5, 0, 0);
      v404 = a3;
      v288 = v14;
      v432 = sub_A79C90(v5, 0, 0);
      v289 = 0;
      do
      {
        v427 = v288;
        v424 = v397;
        v428 = v288;
        v425 = v416;
        v426 = v416;
        v429 = v288;
        v430 = v288;
        v431 = sub_A79C90(v5, 0, 0);
        v290 = sub_A79C90(v5, 0, 0);
        v291 = (__int64 *)((char *)&v424 + v289 * 8);
        v432 = v290;
        v292 = &v421[v289++];
        sub_3121310(v417, v292, v291, 1);
        v431 = sub_A79C90(v5, 0, 0);
        v432 = sub_A79C90(v5, 0, 0);
      }
      while ( v289 != 9 );
      goto LABEL_300;
    case 69:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v431 = sub_A79C90(v5, 0, 0);
      v404 = a3;
      v283 = v14;
      v432 = sub_A79C90(v5, 0, 0);
      v284 = 0;
      do
      {
        v427 = v283;
        v424 = v397;
        v428 = v283;
        v425 = v416;
        v426 = v416;
        v429 = v283;
        v430 = v283;
        v431 = sub_A79C90(v5, 0, 0);
        v285 = sub_A79C90(v5, 0, 0);
        v286 = (__int64 *)((char *)&v424 + v284 * 8);
        v432 = v285;
        v287 = &v421[v284++];
        sub_3121310(v417, v287, v286, 1);
        v431 = sub_A79C90(v5, 0, 0);
        v432 = sub_A79C90(v5, 0, 0);
      }
      while ( v284 != 9 );
      goto LABEL_300;
    case 71:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v394 = a3;
      v409 = v5;
      v280 = 0;
      do
      {
        v281 = (__int64 *)((char *)&v424 + v280 * 8);
        v424 = v397;
        v282 = &v421[v280++];
        v425 = v416;
        v426 = v416;
        v427 = v14;
        v428 = v416;
        v429 = v416;
        v430 = v416;
        v431 = v416;
        sub_3121310(v417, v282, v281, 1);
      }
      while ( v280 != 8 );
      goto LABEL_282;
    case 72:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v394 = a3;
      v409 = v5;
      for ( j = 0; j != 8; ++j )
      {
        v424 = v397;
        v427 = v14;
        v425 = v416;
        v426 = v416;
        v278 = &v421[j];
        v428 = v415;
        v429 = v415;
        v430 = v416;
        v431 = v416;
        v279 = (__int64 *)((char *)&v424 + j * 8);
        sub_3121310(v417, v278, v279, 1);
      }
      goto LABEL_282;
    case 73:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v270 = a3;
      v274 = 0;
      do
      {
        v275 = (__int64 *)((char *)&v424 + v274 * 8);
        v427 = v14;
        v424 = v397;
        v276 = &v421[v274++];
        v425 = v416;
        v426 = v416;
        sub_3121310(v417, v276, v275, 1);
      }
      while ( v274 != 4 );
      goto LABEL_273;
    case 74:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v270 = a3;
      v271 = 0;
      do
      {
        v272 = (__int64 *)((char *)&v424 + v271 * 8);
        v427 = v14;
        v424 = v397;
        v273 = &v421[v271++];
        v425 = v416;
        v426 = v416;
        sub_3121310(v417, v273, v272, 1);
      }
      while ( v271 != 4 );
LABEL_273:
      *(_QWORD *)(v270 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 75:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v267 = 0;
      do
      {
        v268 = (__int64 *)((char *)&v424 + v267 * 8);
        v424 = v397;
        v269 = &v421[v267++];
        v425 = v416;
        v426 = v416;
        v427 = v416;
        v428 = v416;
        v429 = v416;
        v430 = v416;
        sub_3121310(v417, v269, v268, 1);
      }
      while ( v267 != 7 );
      goto LABEL_30;
    case 76:
      v264 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      do
      {
        v424 = v397;
        v427 = v415;
        v265 = &v421[v264];
        v428 = v415;
        v266 = (__int64 *)((char *)&v424 + v264 * 8);
        ++v264;
        v425 = v416;
        v426 = v416;
        v429 = v416;
        v430 = v416;
        sub_3121310(v417, v265, v266, 1);
      }
      while ( v264 != 7 );
      goto LABEL_30;
    case 77:
      v261 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      do
      {
        v262 = (__int64 *)((char *)&v424 + v261 * 8);
        v424 = v397;
        v263 = &v421[v261++];
        v425 = v416;
        v426 = v416;
        sub_3121310(v417, v263, v262, 1);
      }
      while ( v261 != 3 );
      goto LABEL_30;
    case 78:
      v258 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      do
      {
        v259 = (__int64 *)((char *)&v424 + v258 * 8);
        v424 = v397;
        v260 = &v421[v258++];
        v425 = v416;
        v426 = v416;
        sub_3121310(v417, v260, v259, 1);
      }
      while ( v258 != 3 );
      goto LABEL_30;
    case 79:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      v231 = a3;
      v255 = 0;
      do
      {
        v426 = v14;
        v425 = v416;
        v256 = &v421[v255];
        v257 = (__int64 *)((char *)&v424 + v255 * 8);
        ++v255;
        v424 = v397;
        v427 = v14;
        v428 = v14;
        v429 = v14;
        sub_3121310(v417, v256, v257, 1);
      }
      while ( v255 != 6 );
      goto LABEL_258;
    case 80:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      v231 = a3;
      v252 = 0;
      do
      {
        v426 = v14;
        v425 = v416;
        v253 = &v421[v252];
        v254 = (__int64 *)((char *)&v424 + v252 * 8);
        ++v252;
        v424 = v397;
        v427 = v14;
        v428 = v14;
        v429 = v14;
        sub_3121310(v417, v253, v254, 1);
      }
      while ( v252 != 6 );
      goto LABEL_258;
    case 81:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      v231 = a3;
      v249 = 0;
      do
      {
        v426 = v14;
        v425 = v416;
        v250 = &v421[v249];
        v251 = (__int64 *)((char *)&v424 + v249 * 8);
        ++v249;
        v424 = v397;
        v427 = v14;
        v428 = v14;
        v429 = v14;
        sub_3121310(v417, v250, v251, 1);
      }
      while ( v249 != 6 );
      goto LABEL_258;
    case 82:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      v231 = a3;
      v246 = 0;
      do
      {
        v426 = v14;
        v425 = v416;
        v247 = &v421[v246];
        v248 = (__int64 *)((char *)&v424 + v246 * 8);
        ++v246;
        v424 = v397;
        v427 = v14;
        v428 = v14;
        v429 = v14;
        sub_3121310(v417, v247, v248, 1);
      }
      while ( v246 != 6 );
      goto LABEL_258;
    case 88:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v410 = v5;
      v242 = v397;
      v403 = a3;
      v243 = 0;
      do
      {
        v424 = v242;
        v425 = v416;
        v244 = &v421[v243];
        v430 = v416;
        v431 = v416;
        v245 = (__int64 *)((char *)&v424 + v243 * 8);
        ++v243;
        v426 = v14;
        v427 = v14;
        v428 = v14;
        v429 = v14;
        sub_3121310(v417, v244, v245, 1);
      }
      while ( v243 != 8 );
      goto LABEL_306;
    case 89:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v410 = v5;
      v238 = v397;
      v403 = a3;
      v239 = 0;
      do
      {
        v424 = v238;
        v425 = v416;
        v240 = &v421[v239];
        v430 = v416;
        v431 = v416;
        v241 = (__int64 *)((char *)&v424 + v239 * 8);
        ++v239;
        v426 = v14;
        v427 = v14;
        v428 = v14;
        v429 = v14;
        sub_3121310(v417, v240, v241, 1);
      }
      while ( v239 != 8 );
      goto LABEL_306;
    case 90:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v231 = a3;
      v235 = 0;
      do
      {
        v426 = v14;
        v425 = v416;
        v236 = &v421[v235];
        v237 = (__int64 *)((char *)&v424 + v235 * 8);
        ++v235;
        v424 = v397;
        v427 = v14;
        v428 = v14;
        v429 = v14;
        sub_3121310(v417, v236, v237, 1);
      }
      while ( v235 != 6 );
      goto LABEL_258;
    case 91:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v231 = a3;
      v232 = 0;
      do
      {
        v426 = v14;
        v425 = v416;
        v233 = &v421[v232];
        v234 = (__int64 *)((char *)&v424 + v232 * 8);
        ++v232;
        v424 = v397;
        v427 = v14;
        v428 = v14;
        v429 = v14;
        sub_3121310(v417, v233, v234, 1);
      }
      while ( v232 != 6 );
LABEL_258:
      *(_QWORD *)(v231 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 92:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v410 = v5;
      v227 = v397;
      v403 = a3;
      v228 = 0;
      do
      {
        v430 = v14;
        v424 = v227;
        v425 = v416;
        v426 = v416;
        v229 = &v421[v228];
        v432 = v416;
        v433 = v416;
        v230 = (__int64 *)((char *)&v424 + v228 * 8);
        ++v228;
        v427 = v14;
        v428 = v14;
        v429 = v14;
        v431 = v14;
        sub_3121310(v417, v229, v230, 1);
      }
      while ( v228 != 10 );
      goto LABEL_306;
    case 93:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v410 = v5;
      v223 = v397;
      v403 = a3;
      v224 = 0;
      do
      {
        v430 = v14;
        v424 = v223;
        v425 = v416;
        v426 = v416;
        v225 = &v421[v224];
        v432 = v416;
        v433 = v416;
        v226 = (__int64 *)((char *)&v424 + v224 * 8);
        ++v224;
        v427 = v14;
        v428 = v14;
        v429 = v14;
        v431 = v14;
        sub_3121310(v417, v225, v226, 1);
      }
      while ( v224 != 10 );
      goto LABEL_306;
    case 94:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v410 = v5;
      v219 = v397;
      v403 = a3;
      v220 = 0;
      do
      {
        v430 = v14;
        v424 = v219;
        v425 = v416;
        v426 = v416;
        v221 = &v421[v220];
        v222 = (__int64 *)((char *)&v424 + v220 * 8);
        ++v220;
        v427 = v14;
        v428 = v14;
        v429 = v14;
        v431 = v14;
        sub_3121310(v417, v221, v222, 1);
      }
      while ( v220 != 8 );
      goto LABEL_306;
    case 95:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v13);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v410 = v5;
      v215 = v397;
      v403 = a3;
      v216 = 0;
      do
      {
        v430 = v14;
        v424 = v215;
        v425 = v416;
        v426 = v416;
        v217 = &v421[v216];
        v218 = (__int64 *)((char *)&v424 + v216 * 8);
        ++v216;
        v427 = v14;
        v428 = v14;
        v429 = v14;
        v431 = v14;
        sub_3121310(v417, v217, v218, 1);
      }
      while ( v216 != 8 );
LABEL_306:
      *(_QWORD *)(v403 + 120) = sub_A78180(v410, v412, v413, v421, (unsigned int)v422);
      break;
    case 98:
      v393 = v15;
      v210 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      sub_3121310(v417, (__int64 *)&v413, &v414, 0);
      v211 = v393;
      v394 = a3;
      v409 = v5;
      v212 = v211;
      do
      {
        v213 = (__int64 *)((char *)&v424 + v210 * 8);
        v424 = v397;
        v214 = &v421[v210++];
        v425 = v416;
        v426 = v416;
        v427 = v212;
        v428 = v212;
        v429 = v397;
        sub_3121310(v417, v214, v213, 1);
      }
      while ( v210 != 6 );
LABEL_282:
      *(_QWORD *)(v394 + 120) = sub_A78180(v409, v412, v413, v421, (unsigned int)v422);
      break;
    case 99:
      v206 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      v426 = sub_A79C90(v5, 0, 0);
      do
      {
        v424 = v397;
        v425 = v416;
        v207 = sub_A79C90(v5, 0, 0);
        v208 = (__int64 *)((char *)&v424 + v206 * 8);
        v426 = v207;
        v209 = &v421[v206++];
        sub_3121310(v417, v209, v208, 1);
        v426 = sub_A79C90(v5, 0, 0);
      }
      while ( v206 != 3 );
      goto LABEL_30;
    case 104:
      v202 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      v426 = sub_A79C90(v5, 0, 0);
      do
      {
        v424 = v397;
        v425 = v416;
        v203 = sub_A79C90(v5, 0, 0);
        v204 = (__int64 *)((char *)&v424 + v202 * 8);
        v426 = v203;
        v205 = &v421[v202];
        v428 = v397;
        ++v202;
        v427 = v416;
        v429 = v416;
        v430 = v397;
        sub_3121310(v417, v205, v204, 1);
        v426 = sub_A79C90(v5, 0, 0);
      }
      while ( v202 != 7 );
      goto LABEL_30;
    case 105:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v426 = sub_A79C90(v5, 0, 0);
      v404 = a3;
      v430 = sub_A79C90(v5, 0, 0);
      v198 = 0;
      do
      {
        v424 = v397;
        v425 = v416;
        v426 = sub_A79C90(v5, 0, 0);
        v428 = v14;
        v427 = v416;
        v429 = v14;
        v199 = sub_A79C90(v5, 0, 0);
        v200 = (__int64 *)((char *)&v424 + v198 * 8);
        v430 = v199;
        v201 = &v421[v198++];
        v431 = v416;
        v432 = v416;
        sub_3121310(v417, v201, v200, 1);
        v426 = sub_A79C90(v5, 0, 0);
        v430 = sub_A79C90(v5, 0, 0);
      }
      while ( v198 != 9 );
      goto LABEL_300;
    case 107:
      v392 = v15;
      v194 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      sub_3121310(v417, (__int64 *)&v413, &v414, 0);
      v430 = sub_A79C90(v5, 0, 0);
      do
      {
        v427 = v392;
        v424 = v397;
        v429 = v397;
        v425 = v416;
        v426 = v416;
        v428 = v392;
        v195 = sub_A79C90(v5, 0, 0);
        v196 = (__int64 *)((char *)&v424 + v194 * 8);
        v430 = v195;
        v197 = &v421[v194++];
        sub_3121310(v417, v197, v196, 1);
        v430 = sub_A79C90(v5, 0, 0);
      }
      while ( v194 != 7 );
      goto LABEL_30;
    case 108:
      v191 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      do
      {
        v192 = (__int64 *)((char *)&v424 + v191 * 8);
        v424 = v397;
        v193 = &v421[v191++];
        v425 = v416;
        v426 = v416;
        v427 = v416;
        sub_3121310(v417, v193, v192, 1);
      }
      while ( v191 != 4 );
      goto LABEL_30;
    case 109:
    case 112:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v424 = v416;
      v425 = v416;
      sub_3121310(v417, v421, &v424, 1);
      v424 = v416;
      v425 = v416;
      sub_3121310(v417, v421 + 1, &v425, 1);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 110:
      v188 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v389);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      do
      {
        v189 = (__int64 *)((char *)&v424 + v188 * 8);
        v424 = v397;
        v190 = &v421[v188++];
        v425 = v416;
        v426 = v416;
        sub_3121310(v417, v190, v189, 1);
      }
      while ( v188 != 3 );
      goto LABEL_30;
    case 111:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v424 = v416;
      sub_3121310(v417, v421, &v424, 1);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 113:
      v184 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v424 = sub_A79C90(v5, 0, 0);
      do
      {
        v185 = sub_A79C90(v5, 0, 0);
        v186 = (__int64 *)((char *)&v424 + v184 * 8);
        v424 = v185;
        v187 = &v421[v184++];
        v425 = v416;
        v426 = v416;
        v427 = v416;
        sub_3121310(v417, v187, v186, 1);
        v424 = sub_A79C90(v5, 0, 0);
      }
      while ( v184 != 4 );
      goto LABEL_30;
    case 114:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 115:
      v181 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v389);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      do
      {
        v182 = (__int64 *)((char *)&v424 + v181 * 8);
        v424 = v397;
        v183 = &v421[v181++];
        v425 = v416;
        v426 = v416;
        v427 = v397;
        v428 = v416;
        sub_3121310(v417, v183, v182, 1);
      }
      while ( v181 != 5 );
      goto LABEL_30;
    case 116:
      v177 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v389);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v429 = sub_A79C90(v5, 0, 0);
      do
      {
        v424 = v397;
        v425 = v416;
        v426 = v416;
        v427 = v397;
        v428 = v416;
        v178 = sub_A79C90(v5, 0, 0);
        v179 = (__int64 *)((char *)&v424 + v177 * 8);
        v429 = v178;
        v180 = &v421[v177++];
        v430 = v416;
        sub_3121310(v417, v180, v179, 1);
        v429 = sub_A79C90(v5, 0, 0);
      }
      while ( v177 != 7 );
      goto LABEL_30;
    case 117:
      v174 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      do
      {
        v175 = (__int64 *)((char *)&v424 + v174 * 8);
        v424 = v397;
        v176 = &v421[v174++];
        v425 = v416;
        v426 = v416;
        sub_3121310(v417, v176, v175, 1);
      }
      while ( v174 != 3 );
      goto LABEL_30;
    case 118:
      v171 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v385);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      do
      {
        v172 = (__int64 *)((char *)&v424 + v171 * 8);
        v424 = v397;
        v173 = &v421[v171++];
        v426 = v397;
        v425 = v416;
        sub_3121310(v417, v173, v172, 1);
      }
      while ( v171 != 3 );
      goto LABEL_30;
    case 119:
      v168 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v386);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      do
      {
        v169 = (__int64 *)((char *)&v424 + v168 * 8);
        v424 = v397;
        v170 = &v421[v168++];
        v425 = v416;
        v426 = v416;
        v427 = v416;
        sub_3121310(v417, v170, v169, 1);
      }
      while ( v168 != 4 );
      goto LABEL_30;
    case 121:
      v165 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v386);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      do
      {
        v166 = (__int64 *)((char *)&v424 + v165 * 8);
        v424 = v397;
        v167 = &v421[v165++];
        v425 = v416;
        v426 = v416;
        sub_3121310(v417, v167, v166, 1);
      }
      while ( v165 != 3 );
      goto LABEL_30;
    case 122:
      v391 = v15;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v408 = a3;
      v160 = 0;
      v428 = sub_A79C90(v5, 0, 0);
      v161 = v5;
      do
      {
        v426 = v391;
        v424 = v397;
        v427 = v397;
        v425 = v416;
        v162 = sub_A79C90(v5, 0, 0);
        v163 = (__int64 *)((char *)&v424 + v160 * 8);
        v428 = v162;
        v164 = &v421[v160++];
        v429 = v416;
        sub_3121310(v417, v164, v163, 1);
        v428 = sub_A79C90(v5, 0, 0);
      }
      while ( v160 != 6 );
LABEL_318:
      *(_QWORD *)(v408 + 120) = sub_A78180(v161, v412, v413, v421, (unsigned int)v422);
      break;
    case 123:
      v390 = v15;
      v156 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v426 = sub_A79C90(v5, 0, 0);
      do
      {
        v424 = v397;
        v425 = v416;
        v157 = sub_A79C90(v5, 0, 0);
        v158 = (__int64 *)((char *)&v424 + v156 * 8);
        v426 = v157;
        v159 = &v421[v156];
        v427 = v390;
        ++v156;
        sub_3121310(v417, v159, v158, 1);
        v426 = sub_A79C90(v5, 0, 0);
      }
      while ( v156 != 4 );
      goto LABEL_30;
    case 124:
      v152 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v425 = sub_A79C90(v5, 0, 0);
      do
      {
        v424 = v397;
        v153 = sub_A79C90(v5, 0, 0);
        v154 = (__int64 *)((char *)&v424 + v152 * 8);
        v425 = v153;
        v155 = &v421[v152];
        v426 = v397;
        ++v152;
        v427 = v397;
        v428 = v397;
        sub_3121310(v417, v155, v154, 1);
        v425 = sub_A79C90(v5, 0, 0);
      }
      while ( v152 != 5 );
      goto LABEL_30;
    case 125:
      v149 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v389);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      do
      {
        v150 = (__int64 *)((char *)&v424 + v149 * 8);
        v424 = v397;
        v151 = &v421[v149++];
        v425 = v416;
        v426 = v416;
        sub_3121310(v417, v151, v150, 1);
      }
      while ( v149 != 3 );
      goto LABEL_30;
    case 126:
      v146 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v389);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      do
      {
        v147 = (__int64 *)((char *)&v424 + v146 * 8);
        v424 = v397;
        v148 = &v421[v146++];
        v426 = v397;
        v425 = v416;
        sub_3121310(v417, v148, v147, 1);
      }
      while ( v146 != 3 );
      goto LABEL_30;
    case 127:
      v143 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v389);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      do
      {
        v144 = (__int64 *)((char *)&v424 + v143 * 8);
        v424 = v397;
        v145 = &v421[v143++];
        v426 = v397;
        v425 = v416;
        sub_3121310(v417, v145, v144, 1);
      }
      while ( v143 != 3 );
      goto LABEL_30;
    case 129:
      v407 = v15;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      sub_3121310(v417, (__int64 *)&v413, &v414, 0);
      v424 = v416;
      v425 = v407;
      sub_3121310(v417, v421, &v424, 1);
      v424 = v416;
      v425 = v407;
      sub_3121310(v417, v421 + 1, &v425, 1);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 130:
      v402 = v15;
      v131 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      sub_3121310(v417, (__int64 *)&v413, &v414, 0);
      do
      {
        v132 = (__int64 *)((char *)&v424 + v131 * 8);
        v425 = v402;
        v133 = &v421[v131++];
        v424 = v416;
        v426 = v402;
        sub_3121310(v417, v133, v132, 1);
      }
      while ( v131 != 3 );
      goto LABEL_30;
    case 131:
    case 136:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v382);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v424 = v416;
      sub_3121310(v417, v421, &v424, 1);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 132:
      v126 = 0;
      v127 = sub_A79C90(v5, 0, 0);
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v127);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      while ( 1 )
      {
        v424 = sub_A79C90(v5, 0, 0);
        v426 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        v430 = sub_A79C90(v5, 0, 0);
        if ( v126 == 8 )
          break;
        v424 = sub_A79C90(v5, 0, 0);
        v425 = v416;
        v426 = sub_A79C90(v5, 0, 0);
        v427 = v416;
        v428 = v416;
        v429 = sub_A79C90(v5, 0, 0);
        v128 = sub_A79C90(v5, 0, 0);
        v129 = (__int64 *)((char *)&v424 + v126 * 8);
        v430 = v128;
        v130 = &v421[v126++];
        v431 = v416;
        sub_3121310(v417, v130, v129, 1);
      }
      goto LABEL_30;
    case 133:
      v121 = 0;
      v122 = sub_A79C90(v5, 0, 0);
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v122);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      while ( 1 )
      {
        v424 = sub_A79C90(v5, 0, 0);
        v426 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        if ( v121 == 7 )
          break;
        v424 = sub_A79C90(v5, 0, 0);
        v425 = v416;
        v426 = sub_A79C90(v5, 0, 0);
        v427 = v416;
        v428 = v416;
        v123 = sub_A79C90(v5, 0, 0);
        v124 = (__int64 *)((char *)&v424 + v121 * 8);
        v429 = v123;
        v125 = &v421[v121++];
        v430 = v416;
        sub_3121310(v417, v125, v124, 1);
      }
      goto LABEL_30;
    case 134:
      v138 = 0;
      v139 = sub_A79C90(v5, 0, 0);
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v139);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      while ( 1 )
      {
        v424 = sub_A79C90(v5, 0, 0);
        v426 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        if ( v138 == 7 )
          break;
        v424 = sub_A79C90(v5, 0, 0);
        v425 = v416;
        v426 = sub_A79C90(v5, 0, 0);
        v427 = v416;
        v428 = v416;
        v140 = sub_A79C90(v5, 0, 0);
        v141 = (__int64 *)((char *)&v424 + v138 * 8);
        v429 = v140;
        v142 = &v421[v138++];
        v430 = v416;
        sub_3121310(v417, v142, v141, 1);
      }
      goto LABEL_30;
    case 135:
      v134 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v387);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v425 = sub_A79C90(v5, 0, 0);
      do
      {
        v424 = v416;
        v135 = sub_A79C90(v5, 0, 0);
        v136 = (__int64 *)((char *)&v424 + v134 * 8);
        v425 = v135;
        v137 = &v421[v134++];
        v426 = v416;
        sub_3121310(v417, v137, v136, 1);
        v425 = sub_A79C90(v5, 0, 0);
      }
      while ( v134 != 3 );
      goto LABEL_30;
    case 137:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v383);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 138:
      v117 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v385);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      while ( 1 )
      {
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
        v426 = sub_A79C90(v5, 0, 0);
        if ( v117 == 4 )
          break;
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
        v118 = sub_A79C90(v5, 0, 0);
        v119 = (__int64 *)((char *)&v424 + v117 * 8);
        v426 = v118;
        v120 = &v421[v117++];
        v427 = v416;
        sub_3121310(v417, v120, v119, 1);
      }
      goto LABEL_30;
    case 139:
      v116 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v385);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      while ( 1 )
      {
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
        v426 = sub_A79C90(v5, 0, 0);
        v428 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        v430 = sub_A79C90(v5, 0, 0);
        v431 = sub_A79C90(v5, 0, 0);
        v432 = sub_A79C90(v5, 0, 0);
        v433 = sub_A79C90(v5, 0, 0);
        v435 = sub_A79C90(v5, 0, 0);
        if ( v116 == 13 )
          break;
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
        v426 = sub_A79C90(v5, 0, 0);
        v427 = v416;
        v428 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        v430 = sub_A79C90(v5, 0, 0);
        v431 = sub_A79C90(v5, 0, 0);
        v432 = sub_A79C90(v5, 0, 0);
        v433 = sub_A79C90(v5, 0, 0);
        v434 = v416;
        v435 = sub_A79C90(v5, 0, 0);
        v436 = v416;
        sub_3121310(v417, &v421[v116], (__int64 *)((char *)&v424 + v116 * 8), 1);
        ++v116;
      }
      goto LABEL_30;
    case 140:
      v112 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v385);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      while ( 1 )
      {
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
        v426 = sub_A79C90(v5, 0, 0);
        v428 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        v430 = sub_A79C90(v5, 0, 0);
        v431 = sub_A79C90(v5, 0, 0);
        v432 = sub_A79C90(v5, 0, 0);
        v433 = sub_A79C90(v5, 0, 0);
        if ( v112 == 12 )
          break;
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
        v426 = sub_A79C90(v5, 0, 0);
        v427 = v416;
        v428 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        v430 = sub_A79C90(v5, 0, 0);
        v431 = sub_A79C90(v5, 0, 0);
        v432 = sub_A79C90(v5, 0, 0);
        v113 = sub_A79C90(v5, 0, 0);
        v114 = (__int64 *)((char *)&v424 + v112 * 8);
        v433 = v113;
        v115 = &v421[v112++];
        v434 = v416;
        v435 = v416;
        sub_3121310(v417, v115, v114, 1);
      }
      goto LABEL_30;
    case 141:
      v111 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v385);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      while ( 1 )
      {
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
        v426 = sub_A79C90(v5, 0, 0);
        v428 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        v430 = sub_A79C90(v5, 0, 0);
        v431 = sub_A79C90(v5, 0, 0);
        v432 = sub_A79C90(v5, 0, 0);
        v433 = sub_A79C90(v5, 0, 0);
        v437 = sub_A79C90(v5, 0, 0);
        if ( v111 == 15 )
          break;
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
        v426 = sub_A79C90(v5, 0, 0);
        v427 = v416;
        v428 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        v430 = sub_A79C90(v5, 0, 0);
        v431 = sub_A79C90(v5, 0, 0);
        v432 = sub_A79C90(v5, 0, 0);
        v433 = sub_A79C90(v5, 0, 0);
        v434 = v416;
        v435 = v416;
        v436 = v416;
        v437 = sub_A79C90(v5, 0, 0);
        v438 = v416;
        sub_3121310(v417, &v421[v111], (__int64 *)((char *)&v424 + v111 * 8), 1);
        ++v111;
      }
      goto LABEL_30;
    case 142:
      v107 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v385);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      v424 = sub_A79C90(v5, 0, 0);
      v425 = sub_A79C90(v5, 0, 0);
      do
      {
        v424 = sub_A79C90(v5, 0, 0);
        v108 = sub_A79C90(v5, 0, 0);
        v109 = (__int64 *)((char *)&v424 + v107 * 8);
        v425 = v108;
        v110 = &v421[v107++];
        v426 = v416;
        v427 = v416;
        sub_3121310(v417, v110, v109, 1);
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
      }
      while ( v107 != 4 );
      goto LABEL_30;
    case 143:
      v103 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v385);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      while ( 1 )
      {
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
        v428 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        v431 = sub_A79C90(v5, 0, 0);
        if ( v103 == 9 )
          break;
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
        v426 = v416;
        v427 = v416;
        v428 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        v430 = v416;
        v104 = sub_A79C90(v5, 0, 0);
        v105 = (__int64 *)((char *)&v424 + v103 * 8);
        v431 = v104;
        v106 = &v421[v103++];
        v432 = v416;
        sub_3121310(v417, v106, v105, 1);
      }
      goto LABEL_30;
    case 144:
      v99 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v385);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v424 = sub_A79C90(v5, 0, 0);
      v425 = sub_A79C90(v5, 0, 0);
      do
      {
        v424 = sub_A79C90(v5, 0, 0);
        v100 = sub_A79C90(v5, 0, 0);
        v101 = (__int64 *)((char *)&v424 + v99 * 8);
        v425 = v100;
        v102 = &v421[v99++];
        v426 = v416;
        sub_3121310(v417, v102, v101, 1);
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
      }
      while ( v99 != 3 );
      goto LABEL_30;
    case 145:
      v98 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v385);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      while ( 1 )
      {
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
        v427 = sub_A79C90(v5, 0, 0);
        v428 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        v430 = sub_A79C90(v5, 0, 0);
        v431 = sub_A79C90(v5, 0, 0);
        v432 = sub_A79C90(v5, 0, 0);
        v434 = sub_A79C90(v5, 0, 0);
        v436 = sub_A79C90(v5, 0, 0);
        if ( v98 == 13 )
          break;
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
        v426 = v416;
        v427 = sub_A79C90(v5, 0, 0);
        v428 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        v430 = sub_A79C90(v5, 0, 0);
        v431 = sub_A79C90(v5, 0, 0);
        v432 = sub_A79C90(v5, 0, 0);
        v433 = v416;
        v434 = sub_A79C90(v5, 0, 0);
        v435 = v416;
        v436 = sub_A79C90(v5, 0, 0);
        sub_3121310(v417, &v421[v98], (__int64 *)((char *)&v424 + v98 * 8), 1);
        ++v98;
      }
      goto LABEL_30;
    case 146:
      v93 = 0;
      v94 = sub_A79C90(v5, 0, 0);
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v94);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v424 = sub_A79C90(v5, 0, 0);
      v425 = sub_A79C90(v5, 0, 0);
      do
      {
        v424 = sub_A79C90(v5, 0, 0);
        v95 = sub_A79C90(v5, 0, 0);
        v96 = (__int64 *)((char *)&v424 + v93 * 8);
        v425 = v95;
        v97 = &v421[v93++];
        v426 = v416;
        sub_3121310(v417, v97, v96, 1);
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
      }
      while ( v93 != 3 );
      goto LABEL_30;
    case 148:
      v89 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v385);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v424 = sub_A79C90(v5, 0, 0);
      v425 = sub_A79C90(v5, 0, 0);
      do
      {
        v424 = sub_A79C90(v5, 0, 0);
        v90 = sub_A79C90(v5, 0, 0);
        v91 = (__int64 *)((char *)&v424 + v89 * 8);
        v425 = v90;
        v92 = &v421[v89++];
        v426 = v416;
        sub_3121310(v417, v92, v91, 1);
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
      }
      while ( v89 != 3 );
      goto LABEL_30;
    case 149:
      v88 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v385);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      while ( 1 )
      {
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
        v427 = sub_A79C90(v5, 0, 0);
        v428 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        v430 = sub_A79C90(v5, 0, 0);
        v431 = sub_A79C90(v5, 0, 0);
        v432 = sub_A79C90(v5, 0, 0);
        v434 = sub_A79C90(v5, 0, 0);
        v436 = sub_A79C90(v5, 0, 0);
        if ( v88 == 13 )
          break;
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
        v426 = v416;
        v427 = sub_A79C90(v5, 0, 0);
        v428 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        v430 = sub_A79C90(v5, 0, 0);
        v431 = sub_A79C90(v5, 0, 0);
        v432 = sub_A79C90(v5, 0, 0);
        v433 = v416;
        v434 = sub_A79C90(v5, 0, 0);
        v435 = v416;
        v436 = sub_A79C90(v5, 0, 0);
        sub_3121310(v417, &v421[v88], (__int64 *)((char *)&v424 + v88 * 8), 1);
        ++v88;
      }
      goto LABEL_30;
    case 150:
      v84 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v385);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v424 = sub_A79C90(v5, 0, 0);
      v425 = sub_A79C90(v5, 0, 0);
      do
      {
        v424 = sub_A79C90(v5, 0, 0);
        v85 = sub_A79C90(v5, 0, 0);
        v86 = (__int64 *)((char *)&v424 + v84 * 8);
        v425 = v85;
        v87 = &v421[v84++];
        v426 = v416;
        sub_3121310(v417, v87, v86, 1);
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
      }
      while ( v84 != 3 );
      goto LABEL_30;
    case 151:
      v83 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v385);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      while ( 1 )
      {
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
        v427 = sub_A79C90(v5, 0, 0);
        v428 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        v430 = sub_A79C90(v5, 0, 0);
        v431 = sub_A79C90(v5, 0, 0);
        v432 = sub_A79C90(v5, 0, 0);
        v434 = sub_A79C90(v5, 0, 0);
        v436 = sub_A79C90(v5, 0, 0);
        if ( v83 == 13 )
          break;
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
        v426 = v416;
        v427 = sub_A79C90(v5, 0, 0);
        v428 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        v430 = sub_A79C90(v5, 0, 0);
        v431 = sub_A79C90(v5, 0, 0);
        v432 = sub_A79C90(v5, 0, 0);
        v433 = v416;
        v434 = sub_A79C90(v5, 0, 0);
        v435 = v416;
        v436 = sub_A79C90(v5, 0, 0);
        sub_3121310(v417, &v421[v83], (__int64 *)((char *)&v424 + v83 * 8), 1);
        ++v83;
      }
      goto LABEL_30;
    case 152:
    case 153:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v385);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 155:
      v82 = sub_A79C90(v5, 0, 0);
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v82);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      sub_A79C90(v5, 0, 0);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, v421, &v424, 1);
      sub_A79C90(v5, 0, 0);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 156:
      v81 = sub_A79C90(v5, 0, 0);
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v81);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 158:
      v401 = v15;
      v77 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v384);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v404 = a3;
      while ( 1 )
      {
        v424 = sub_A79C90(v5, 0, 0);
        v429 = sub_A79C90(v5, 0, 0);
        v430 = sub_A79C90(v5, 0, 0);
        v431 = sub_A79C90(v5, 0, 0);
        if ( v77 == 9 )
          break;
        v424 = sub_A79C90(v5, 0, 0);
        v425 = v416;
        v426 = v416;
        v427 = v416;
        v428 = v416;
        v429 = sub_A79C90(v5, 0, 0);
        v430 = sub_A79C90(v5, 0, 0);
        v78 = sub_A79C90(v5, 0, 0);
        v79 = (__int64 *)((char *)&v424 + v77 * 8);
        v431 = v78;
        v80 = &v421[v77];
        v432 = v401;
        ++v77;
        sub_3121310(v417, v80, v79, 1);
      }
      goto LABEL_300;
    case 159:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v384);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v425 = sub_A79C90(v5, 0, 0);
      v426 = sub_A79C90(v5, 0, 0);
      v72 = a3;
      v73 = 0;
      v48 = v72;
      do
      {
        v424 = v397;
        v425 = sub_A79C90(v5, 0, 0);
        v74 = sub_A79C90(v5, 0, 0);
        v75 = (__int64 *)((char *)&v424 + v73 * 8);
        v426 = v74;
        v76 = &v421[v73++];
        v427 = v416;
        v428 = v416;
        v429 = v416;
        sub_3121310(v417, v76, v75, 1);
        v425 = sub_A79C90(v5, 0, 0);
        v426 = sub_A79C90(v5, 0, 0);
      }
      while ( v73 != 6 );
      goto LABEL_83;
    case 160:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v384);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v425 = sub_A79C90(v5, 0, 0);
      v426 = sub_A79C90(v5, 0, 0);
      v67 = a3;
      v68 = 0;
      v48 = v67;
      do
      {
        v424 = v397;
        v425 = sub_A79C90(v5, 0, 0);
        v69 = sub_A79C90(v5, 0, 0);
        v70 = (__int64 *)((char *)&v424 + v68 * 8);
        v426 = v69;
        v71 = &v421[v68++];
        v427 = v415;
        v428 = v415;
        v429 = v415;
        sub_3121310(v417, v71, v70, 1);
        v425 = sub_A79C90(v5, 0, 0);
        v426 = sub_A79C90(v5, 0, 0);
      }
      while ( v68 != 6 );
      goto LABEL_83;
    case 163:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v384);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v425 = sub_A79C90(v5, 0, 0);
      v426 = sub_A79C90(v5, 0, 0);
      v62 = a3;
      v63 = 0;
      v48 = v62;
      do
      {
        v424 = v397;
        v425 = sub_A79C90(v5, 0, 0);
        v64 = sub_A79C90(v5, 0, 0);
        v65 = (__int64 *)((char *)&v424 + v63 * 8);
        v426 = v64;
        v66 = &v421[v63++];
        v427 = v416;
        v428 = v416;
        sub_3121310(v417, v66, v65, 1);
        v425 = sub_A79C90(v5, 0, 0);
        v426 = sub_A79C90(v5, 0, 0);
      }
      while ( v63 != 5 );
      goto LABEL_83;
    case 164:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v384);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v425 = sub_A79C90(v5, 0, 0);
      v426 = sub_A79C90(v5, 0, 0);
      v57 = a3;
      v58 = 0;
      v48 = v57;
      do
      {
        v424 = v397;
        v425 = sub_A79C90(v5, 0, 0);
        v59 = sub_A79C90(v5, 0, 0);
        v60 = (__int64 *)((char *)&v424 + v58 * 8);
        v426 = v59;
        v61 = &v421[v58++];
        v427 = v415;
        v428 = v415;
        sub_3121310(v417, v61, v60, 1);
        v425 = sub_A79C90(v5, 0, 0);
        v426 = sub_A79C90(v5, 0, 0);
      }
      while ( v58 != 5 );
      goto LABEL_83;
    case 167:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v384);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v425 = sub_A79C90(v5, 0, 0);
      v426 = sub_A79C90(v5, 0, 0);
      v52 = a3;
      v53 = 0;
      v48 = v52;
      do
      {
        v424 = v397;
        v425 = sub_A79C90(v5, 0, 0);
        v54 = sub_A79C90(v5, 0, 0);
        v55 = (__int64 *)((char *)&v424 + v53 * 8);
        v426 = v54;
        v56 = &v421[v53++];
        v427 = v416;
        v428 = v416;
        v429 = v416;
        v430 = v416;
        sub_3121310(v417, v56, v55, 1);
        v425 = sub_A79C90(v5, 0, 0);
        v426 = sub_A79C90(v5, 0, 0);
      }
      while ( v53 != 7 );
      goto LABEL_83;
    case 168:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v384);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v425 = sub_A79C90(v5, 0, 0);
      v426 = sub_A79C90(v5, 0, 0);
      v46 = a3;
      v47 = 0;
      v48 = v46;
      do
      {
        v424 = v397;
        v425 = sub_A79C90(v5, 0, 0);
        v49 = sub_A79C90(v5, 0, 0);
        v50 = (__int64 *)((char *)&v424 + v47 * 8);
        v426 = v49;
        v51 = &v421[v47++];
        v427 = v415;
        v428 = v415;
        v429 = v415;
        v430 = v415;
        sub_3121310(v417, v51, v50, 1);
        v425 = sub_A79C90(v5, 0, 0);
        v426 = sub_A79C90(v5, 0, 0);
      }
      while ( v47 != 7 );
LABEL_83:
      *(_QWORD *)(v48 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 175:
      v42 = 0;
      v43 = sub_A79C90(v5, 0, 0);
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v43);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      do
      {
        v44 = (__int64 *)((char *)&v424 + v42 * 8);
        v45 = &v421[v42++];
        v424 = v416;
        v425 = v416;
        v426 = v416;
        sub_3121310(v417, v45, v44, 1);
      }
      while ( v42 != 3 );
      goto LABEL_30;
    case 176:
    case 186:
      v21 = sub_A79C90(v5, 0, 0);
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v21);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 177:
      v37 = 0;
      v38 = sub_A79C90(v5, 0, 0);
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v38);
      sub_3121310(v417, (__int64 *)&v413, &v416, 0);
      v424 = sub_A79C90(v5, 0, 0);
      v425 = sub_A79C90(v5, 0, 0);
      do
      {
        v424 = sub_A79C90(v5, 0, 0);
        v39 = sub_A79C90(v5, 0, 0);
        v40 = (__int64 *)((char *)&v424 + v37 * 8);
        v425 = v39;
        v41 = &v421[v37++];
        v426 = v415;
        sub_3121310(v417, v41, v40, 1);
        v424 = sub_A79C90(v5, 0, 0);
        v425 = sub_A79C90(v5, 0, 0);
      }
      while ( v37 != 3 );
      goto LABEL_30;
    case 179:
      v32 = 0;
      v33 = sub_A79C90(v5, 0, 0);
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v33);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v424 = sub_A79C90(v5, 0, 0);
      do
      {
        v34 = sub_A79C90(v5, 0, 0);
        v35 = (__int64 *)((char *)&v424 + v32 * 8);
        v424 = v34;
        v36 = &v421[v32++];
        v425 = v416;
        v426 = v416;
        sub_3121310(v417, v36, v35, 1);
        v424 = sub_A79C90(v5, 0, 0);
      }
      while ( v32 != 3 );
LABEL_30:
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 180:
      v406 = v15;
      v424 = sub_A778C0(v5, 41, 0);
      v425 = sub_A778C0(v5, 39, 0);
      BYTE4(v419) = 0;
      v426 = sub_A77AF0(v5, 0, (unsigned int *)&v419);
      v31 = sub_A79C90(v5, &v424, 3);
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v31);
      sub_3121310(v417, (__int64 *)&v413, &v414, 0);
      v424 = v406;
      sub_3121310(v417, v421, &v424, 1);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 181:
      v405 = v15;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v380);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      v424 = sub_A77AD0(v5, 0);
      v425 = sub_A778C0(v5, 2, 0);
      sub_A79C90(v5, &v424, 2);
      v400 = a3;
      v27 = 0;
      do
      {
        v419 = sub_A77AD0(v5, 0);
        v420 = sub_A778C0(v5, 2, 0);
        v28 = sub_A79C90(v5, &v419, 2);
        v29 = (__int64 *)((char *)&v424 + v27 * 8);
        v424 = v28;
        v30 = &v421[v27++];
        v425 = v405;
        sub_3121310(v417, v30, v29, 1);
        v424 = sub_A77AD0(v5, 0);
        v425 = sub_A778C0(v5, 2, 0);
        sub_A79C90(v5, &v424, 2);
      }
      while ( v27 != 2 );
LABEL_54:
      *(_QWORD *)(v400 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 182:
      v399 = v15;
      v22 = sub_A79C90(v5, 0, 0);
      v23 = 0;
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v22);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      sub_A79C90(v5, 0, 0);
      v404 = a3;
      do
      {
        v24 = sub_A79C90(v5, 0, 0);
        v25 = (__int64 *)((char *)&v424 + v23 * 8);
        v424 = v24;
        v26 = &v421[v23];
        v425 = v399;
        ++v23;
        sub_3121310(v417, v26, v25, 1);
        sub_A79C90(v5, 0, 0);
      }
      while ( v23 != 2 );
LABEL_300:
      *(_QWORD *)(v404 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    case 189:
    case 190:
      v412 = sub_A7A4C0((__int64 *)&v412, v5, v389);
      v424 = sub_A79C90(v5, 0, 0);
      sub_3121310(v417, (__int64 *)&v413, &v424, 0);
      *(_QWORD *)(a3 + 120) = sub_A78180(v5, v412, v413, v421, (unsigned int)v422);
      break;
    default:
      break;
  }
  if ( v421 != (__int64 *)v423 )
    _libc_free((unsigned __int64)v421);
}
