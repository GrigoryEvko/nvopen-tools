// Function: sub_40F894
// Address: 0x40f894
//
__int64 __fastcall sub_40F894(unsigned int *a1, char a2, unsigned int a3, unsigned __int64 a4)
{
  int v6; // edx
  int v7; // ecx
  int v8; // r8d
  int v9; // r9d
  int v10; // edx
  int v11; // ecx
  int v12; // r8d
  int v13; // r9d
  __int64 v14; // rcx
  __int64 v15; // rcx
  __int64 v16; // r9
  __int64 v17; // r11
  __int64 v18; // r11
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 v21; // rcx
  __int64 v22; // r9
  __int64 v23; // rcx
  __int64 v24; // rcx
  __int64 v25; // r9
  __int64 v26; // r8
  __int64 v27; // r8
  __int64 v28; // r10
  __int64 v29; // r9
  __int64 v30; // rcx
  __int64 v31; // rcx
  __int64 v32; // r9
  __int64 v33; // r8
  __int64 v34; // r8
  __int64 v35; // r10
  __int64 v36; // r9
  __int64 v37; // rcx
  __int64 v38; // rcx
  __int64 v39; // r9
  __int64 v40; // r10
  __int64 v41; // r9
  __int64 v42; // r9
  __int64 v43; // rcx
  __int64 v44; // rcx
  __int64 v45; // r9
  __int64 v46; // rcx
  __int64 v47; // rcx
  __int64 v48; // r9
  __int64 v49; // r9
  __int64 v50; // r9
  __int64 v51; // r11
  __int64 v52; // r11
  __int64 v53; // r9
  __int64 v54; // r11
  __int64 v55; // r11
  __int64 v56; // r9
  __int64 v57; // r11
  __int64 v58; // r11
  __int64 v59; // r9
  __int64 v60; // rcx
  __int64 v61; // rcx
  __int64 v62; // r9
  __int64 v63; // r8
  __int64 v64; // r8
  __int64 v65; // r10
  __int64 v66; // r9
  __int64 v67; // rcx
  __int64 v68; // rcx
  __int64 v69; // r9
  __int64 v70; // r8
  __int64 v71; // r8
  __int64 v72; // r10
  __int64 v73; // r9
  __int64 v74; // rcx
  __int64 v75; // rcx
  __int64 v76; // r9
  __int64 v77; // rcx
  __int64 v78; // rcx
  __int64 v79; // r9
  __int64 v80; // r8
  int v81; // edx
  int v82; // ecx
  __int64 v83; // r8
  __int64 v84; // r9
  __int64 v85; // r10
  __int64 v86; // r9
  __int64 v87; // rdx
  int v88; // ecx
  int v89; // r8d
  int v90; // r9d
  unsigned __int64 v91; // rdi
  unsigned __int64 v92; // rdi
  unsigned __int64 v93; // rdi
  __int64 *v94; // rsi
  int v95; // r8d
  int v96; // r9d
  int v97; // ecx
  bool v98; // r14
  unsigned __int64 v99; // rdx
  unsigned __int64 v100; // rdi
  unsigned __int64 v101; // rdi
  unsigned __int64 v102; // rdi
  unsigned __int64 v103; // rdi
  unsigned __int64 v104; // rdi
  unsigned __int64 v105; // rdi
  unsigned __int64 v106; // rdi
  unsigned __int64 v107; // rdi
  unsigned __int64 v108; // rdi
  unsigned __int64 v109; // rdi
  unsigned __int64 v110; // rdi
  unsigned __int64 v111; // rdi
  unsigned __int64 v112; // rdi
  unsigned __int64 v113; // rdi
  __int64 v114; // r8
  unsigned __int64 v115; // rdi
  unsigned __int64 v116; // rdi
  unsigned __int64 v117; // rax
  unsigned __int64 v118; // rdi
  unsigned __int64 v119; // rax
  unsigned __int64 v120; // rdi
  unsigned __int64 v121; // rax
  unsigned __int64 v122; // rdi
  unsigned __int64 v123; // rax
  unsigned __int64 v124; // rdi
  unsigned __int64 v125; // rax
  unsigned __int64 v126; // rdi
  unsigned __int64 v127; // rdi
  __int64 v128; // r8
  __int64 v129; // r8
  __int64 v130; // r8
  __int64 v131; // r8
  __int64 v132; // r8
  __int64 v133; // r8
  __int64 v134; // r8
  char *v135; // rsi
  __int64 v136; // r8
  int v137; // edx
  int v138; // ecx
  int v139; // r8d
  int v140; // r9d
  __int64 v141; // r8
  int v142; // r9d
  int v143; // edx
  int v144; // ecx
  int v145; // r8d
  int v146; // r9d
  unsigned __int64 v147; // rcx
  unsigned __int64 v148; // rax
  unsigned __int64 v149; // rax
  unsigned __int64 v150; // rax
  unsigned __int64 v151; // rax
  unsigned __int64 v152; // rax
  unsigned __int64 v153; // rax
  unsigned __int64 v154; // rax
  bool v155; // zf
  __int64 result; // rax
  int v157; // edx
  int v158; // ecx
  int v159; // r8d
  int v160; // r9d
  _BYTE *v161; // [rsp+0h] [rbp-DE0h]
  bool v162; // [rsp+12h] [rbp-DCEh]
  unsigned int v164; // [rsp+14h] [rbp-DCCh]
  unsigned __int64 v166; // [rsp+18h] [rbp-DC8h]
  unsigned int v167; // [rsp+2Ch] [rbp-DB4h] BYREF
  unsigned int v168; // [rsp+30h] [rbp-DB0h] BYREF
  int v169; // [rsp+34h] [rbp-DACh] BYREF
  unsigned __int64 v170; // [rsp+38h] [rbp-DA8h] BYREF
  __int64 v171; // [rsp+40h] [rbp-DA0h] BYREF
  __int64 v172; // [rsp+48h] [rbp-D98h] BYREF
  __int64 v173; // [rsp+50h] [rbp-D90h] BYREF
  __int64 v174; // [rsp+58h] [rbp-D88h] BYREF
  unsigned __int64 v175; // [rsp+60h] [rbp-D80h] BYREF
  unsigned __int64 v176; // [rsp+68h] [rbp-D78h] BYREF
  const char *v177; // [rsp+70h] [rbp-D70h] BYREF
  const char *v178; // [rsp+78h] [rbp-D68h] BYREF
  unsigned __int64 v179; // [rsp+80h] [rbp-D60h] BYREF
  unsigned __int64 v180; // [rsp+88h] [rbp-D58h] BYREF
  unsigned __int64 v181; // [rsp+90h] [rbp-D50h] BYREF
  unsigned __int64 v182; // [rsp+98h] [rbp-D48h] BYREF
  unsigned __int64 v183; // [rsp+A0h] [rbp-D40h] BYREF
  unsigned __int64 v184; // [rsp+A8h] [rbp-D38h] BYREF
  __int64 v185; // [rsp+B0h] [rbp-D30h] BYREF
  __int64 v186; // [rsp+B8h] [rbp-D28h] BYREF
  __int64 v187; // [rsp+C0h] [rbp-D20h] BYREF
  __int64 v188; // [rsp+C8h] [rbp-D18h] BYREF
  __int64 v189; // [rsp+D0h] [rbp-D10h] BYREF
  __int64 v190; // [rsp+D8h] [rbp-D08h] BYREF
  __int64 v191; // [rsp+E0h] [rbp-D00h] BYREF
  __int64 v192; // [rsp+E8h] [rbp-CF8h] BYREF
  __int64 v193; // [rsp+F0h] [rbp-CF0h] BYREF
  __int64 v194; // [rsp+F8h] [rbp-CE8h] BYREF
  __int64 v195; // [rsp+100h] [rbp-CE0h] BYREF
  __int64 v196; // [rsp+108h] [rbp-CD8h] BYREF
  __int64 v197; // [rsp+110h] [rbp-CD0h] BYREF
  __int64 v198; // [rsp+118h] [rbp-CC8h] BYREF
  __int64 v199; // [rsp+120h] [rbp-CC0h] BYREF
  __int64 v200; // [rsp+128h] [rbp-CB8h] BYREF
  __int64 v201; // [rsp+130h] [rbp-CB0h] BYREF
  __int64 v202; // [rsp+138h] [rbp-CA8h] BYREF
  __int64 v203; // [rsp+140h] [rbp-CA0h] BYREF
  __int64 v204; // [rsp+148h] [rbp-C98h] BYREF
  __int64 v205; // [rsp+150h] [rbp-C90h] BYREF
  __int64 v206; // [rsp+158h] [rbp-C88h] BYREF
  __int64 v207; // [rsp+160h] [rbp-C80h] BYREF
  __int64 v208; // [rsp+168h] [rbp-C78h] BYREF
  __int64 v209; // [rsp+170h] [rbp-C70h] BYREF
  __int64 v210; // [rsp+178h] [rbp-C68h] BYREF
  __int64 v211; // [rsp+180h] [rbp-C60h] BYREF
  __int64 v212; // [rsp+188h] [rbp-C58h] BYREF
  __int64 v213; // [rsp+190h] [rbp-C50h] BYREF
  __int64 v214; // [rsp+198h] [rbp-C48h] BYREF
  __int64 v215; // [rsp+1A0h] [rbp-C40h] BYREF
  __int64 v216; // [rsp+1A8h] [rbp-C38h] BYREF
  __int64 v217; // [rsp+1B0h] [rbp-C30h] BYREF
  __int64 v218; // [rsp+1B8h] [rbp-C28h] BYREF
  __int64 v219; // [rsp+1C0h] [rbp-C20h] BYREF
  __int64 v220; // [rsp+1C8h] [rbp-C18h] BYREF
  __int64 v221; // [rsp+1D0h] [rbp-C10h] BYREF
  __int64 v222; // [rsp+1D8h] [rbp-C08h] BYREF
  __int64 v223; // [rsp+1E0h] [rbp-C00h] BYREF
  __int64 v224; // [rsp+1E8h] [rbp-BF8h] BYREF
  __int64 v225; // [rsp+1F0h] [rbp-BF0h] BYREF
  __int64 v226; // [rsp+1F8h] [rbp-BE8h] BYREF
  __int64 v227; // [rsp+200h] [rbp-BE0h] BYREF
  int v228; // [rsp+208h] [rbp-BD8h]
  __int64 v229; // [rsp+210h] [rbp-BD0h]
  __int64 v230; // [rsp+228h] [rbp-BB8h] BYREF
  int v231; // [rsp+230h] [rbp-BB0h]
  char *v232; // [rsp+238h] [rbp-BA8h]
  __int64 v233; // [rsp+250h] [rbp-B90h] BYREF
  int v234; // [rsp+258h] [rbp-B88h]
  unsigned int v235; // [rsp+260h] [rbp-B80h]
  __int64 v236; // [rsp+278h] [rbp-B68h] BYREF
  int v237; // [rsp+280h] [rbp-B60h]
  char *v238; // [rsp+288h] [rbp-B58h]
  __int64 v239; // [rsp+2A0h] [rbp-B40h] BYREF
  int v240; // [rsp+2A8h] [rbp-B38h]
  unsigned __int64 v241; // [rsp+2B0h] [rbp-B30h]
  __int64 v242; // [rsp+2C8h] [rbp-B18h] BYREF
  int v243; // [rsp+2D0h] [rbp-B10h]
  char *v244; // [rsp+2D8h] [rbp-B08h]
  __int64 v245; // [rsp+2F0h] [rbp-AF0h] BYREF
  int v246; // [rsp+2F8h] [rbp-AE8h]
  unsigned __int64 v247; // [rsp+300h] [rbp-AE0h]
  __int64 v248; // [rsp+318h] [rbp-AC8h] BYREF
  int v249; // [rsp+320h] [rbp-AC0h]
  char *v250; // [rsp+328h] [rbp-AB8h]
  __int64 v251; // [rsp+340h] [rbp-AA0h] BYREF
  int v252; // [rsp+348h] [rbp-A98h]
  unsigned __int64 v253; // [rsp+350h] [rbp-A90h]
  __int64 v254; // [rsp+368h] [rbp-A78h] BYREF
  int v255; // [rsp+370h] [rbp-A70h]
  __int64 v256; // [rsp+378h] [rbp-A68h]
  __int64 v257; // [rsp+390h] [rbp-A50h] BYREF
  int v258; // [rsp+398h] [rbp-A48h]
  unsigned __int64 v259; // [rsp+3A0h] [rbp-A40h]
  __int64 v260; // [rsp+3B8h] [rbp-A28h] BYREF
  int v261; // [rsp+3C0h] [rbp-A20h]
  char *v262; // [rsp+3C8h] [rbp-A18h]
  __int64 v263; // [rsp+3E0h] [rbp-A00h] BYREF
  int v264; // [rsp+3E8h] [rbp-9F8h]
  unsigned __int64 v265; // [rsp+3F0h] [rbp-9F0h]
  __int64 v266; // [rsp+408h] [rbp-9D8h] BYREF
  int v267; // [rsp+410h] [rbp-9D0h]
  __int64 v268; // [rsp+418h] [rbp-9C8h]
  __int64 v269; // [rsp+430h] [rbp-9B0h] BYREF
  int v270; // [rsp+438h] [rbp-9A8h]
  unsigned __int64 v271; // [rsp+440h] [rbp-9A0h]
  __int64 v272; // [rsp+458h] [rbp-988h] BYREF
  int v273; // [rsp+460h] [rbp-980h]
  const char *v274; // [rsp+468h] [rbp-978h]
  __int64 v275; // [rsp+480h] [rbp-960h] BYREF
  int v276; // [rsp+488h] [rbp-958h]
  unsigned __int64 v277; // [rsp+490h] [rbp-950h]
  __int64 v278; // [rsp+4A8h] [rbp-938h] BYREF
  int v279; // [rsp+4B0h] [rbp-930h]
  __int64 v280; // [rsp+4B8h] [rbp-928h]
  __int64 v281; // [rsp+4D0h] [rbp-910h] BYREF
  int v282; // [rsp+4D8h] [rbp-908h]
  int v283; // [rsp+4E0h] [rbp-900h]
  __int64 v284; // [rsp+4F8h] [rbp-8E8h] BYREF
  int v285; // [rsp+500h] [rbp-8E0h]
  char *v286; // [rsp+508h] [rbp-8D8h]
  __int64 v287; // [rsp+520h] [rbp-8C0h] BYREF
  int v288; // [rsp+528h] [rbp-8B8h]
  unsigned __int64 v289; // [rsp+530h] [rbp-8B0h]
  __int64 v290; // [rsp+548h] [rbp-898h] BYREF
  int v291; // [rsp+550h] [rbp-890h]
  const char *v292; // [rsp+558h] [rbp-888h]
  __int64 v293; // [rsp+570h] [rbp-870h] BYREF
  int v294; // [rsp+578h] [rbp-868h]
  const char *v295; // [rsp+580h] [rbp-860h]
  __int64 v296; // [rsp+598h] [rbp-848h] BYREF
  int v297; // [rsp+5A0h] [rbp-840h]
  const char *v298; // [rsp+5A8h] [rbp-838h]
  __int64 v299; // [rsp+5C0h] [rbp-820h] BYREF
  int v300; // [rsp+5C8h] [rbp-818h]
  const char *v301; // [rsp+5D0h] [rbp-810h]
  __int64 v302; // [rsp+5E8h] [rbp-7F8h] BYREF
  int v303; // [rsp+5F0h] [rbp-7F0h]
  char *v304; // [rsp+5F8h] [rbp-7E8h]
  __int64 v305; // [rsp+610h] [rbp-7D0h] BYREF
  int v306; // [rsp+618h] [rbp-7C8h]
  unsigned int v307; // [rsp+620h] [rbp-7C0h]
  __int64 v308; // [rsp+638h] [rbp-7A8h] BYREF
  int v309; // [rsp+640h] [rbp-7A0h]
  char *v310; // [rsp+648h] [rbp-798h]
  __int64 v311; // [rsp+660h] [rbp-780h] BYREF
  int v312; // [rsp+668h] [rbp-778h]
  unsigned __int64 v313; // [rsp+670h] [rbp-770h]
  __int64 v314; // [rsp+688h] [rbp-758h] BYREF
  int v315; // [rsp+690h] [rbp-750h]
  const char *v316; // [rsp+698h] [rbp-748h]
  __int64 v317; // [rsp+6B0h] [rbp-730h] BYREF
  int v318; // [rsp+6B8h] [rbp-728h]
  char *v319; // [rsp+6C0h] [rbp-720h]
  __int64 v320; // [rsp+6D8h] [rbp-708h] BYREF
  int v321; // [rsp+6E0h] [rbp-700h]
  char *v322; // [rsp+6E8h] [rbp-6F8h]
  __int64 v323; // [rsp+700h] [rbp-6E0h] BYREF
  int v324; // [rsp+708h] [rbp-6D8h]
  __int64 *v325; // [rsp+710h] [rbp-6D0h]
  __int64 v326; // [rsp+728h] [rbp-6B8h] BYREF
  int v327; // [rsp+730h] [rbp-6B0h]
  const char *v328; // [rsp+738h] [rbp-6A8h]
  __int64 v329; // [rsp+750h] [rbp-690h] BYREF
  int v330; // [rsp+758h] [rbp-688h]
  unsigned __int64 v331; // [rsp+760h] [rbp-680h]
  __int64 v332; // [rsp+778h] [rbp-668h] BYREF
  int v333; // [rsp+780h] [rbp-660h]
  char *v334; // [rsp+788h] [rbp-658h]
  __int64 v335; // [rsp+7A0h] [rbp-640h] BYREF
  int v336; // [rsp+7A8h] [rbp-638h]
  unsigned __int64 v337; // [rsp+7B0h] [rbp-630h]
  __int64 v338; // [rsp+7C8h] [rbp-618h] BYREF
  int v339; // [rsp+7D0h] [rbp-610h]
  __int64 v340; // [rsp+7D8h] [rbp-608h]
  __int64 v341; // [rsp+7F0h] [rbp-5F0h] BYREF
  int v342; // [rsp+7F8h] [rbp-5E8h]
  unsigned __int64 v343; // [rsp+800h] [rbp-5E0h]
  __int64 v344; // [rsp+818h] [rbp-5C8h] BYREF
  int v345; // [rsp+820h] [rbp-5C0h]
  char *v346; // [rsp+828h] [rbp-5B8h]
  __int64 v347; // [rsp+840h] [rbp-5A0h] BYREF
  int v348; // [rsp+848h] [rbp-598h]
  unsigned __int64 v349; // [rsp+850h] [rbp-590h]
  __int64 v350; // [rsp+868h] [rbp-578h] BYREF
  int v351; // [rsp+870h] [rbp-570h]
  __int64 v352; // [rsp+878h] [rbp-568h]
  __int64 v353; // [rsp+890h] [rbp-550h] BYREF
  int v354; // [rsp+898h] [rbp-548h]
  __int64 v355; // [rsp+8A0h] [rbp-540h]
  __int64 v356; // [rsp+8B8h] [rbp-528h] BYREF
  int v357; // [rsp+8C0h] [rbp-520h]
  const char *v358; // [rsp+8C8h] [rbp-518h]
  __int64 v359; // [rsp+8E0h] [rbp-500h] BYREF
  int v360; // [rsp+8E8h] [rbp-4F8h]
  unsigned __int64 v361; // [rsp+8F0h] [rbp-4F0h]
  __int64 v362; // [rsp+908h] [rbp-4D8h] BYREF
  int v363; // [rsp+910h] [rbp-4D0h]
  const char *v364; // [rsp+918h] [rbp-4C8h]
  __int64 v365; // [rsp+930h] [rbp-4B0h] BYREF
  int v366; // [rsp+938h] [rbp-4A8h]
  unsigned __int64 v367; // [rsp+940h] [rbp-4A0h]
  __int64 v368; // [rsp+958h] [rbp-488h] BYREF
  int v369; // [rsp+960h] [rbp-480h]
  __int64 v370; // [rsp+968h] [rbp-478h]
  char v371[8]; // [rsp+980h] [rbp-460h] BYREF
  int v372; // [rsp+988h] [rbp-458h]
  char v373; // [rsp+990h] [rbp-450h] BYREF
  char v374[40]; // [rsp+9A8h] [rbp-438h] BYREF
  _BYTE v375[16]; // [rsp+9D0h] [rbp-410h] BYREF
  __int64 v376; // [rsp+9E0h] [rbp-400h]
  __int64 v377; // [rsp+9F0h] [rbp-3F0h]
  _BYTE v378[16]; // [rsp+A08h] [rbp-3D8h] BYREF
  __int64 v379; // [rsp+A18h] [rbp-3C8h]
  const char *v380; // [rsp+A40h] [rbp-3A0h] BYREF
  int v381; // [rsp+A48h] [rbp-398h]
  unsigned __int64 v382; // [rsp+A50h] [rbp-390h] BYREF
  int v383; // [rsp+A70h] [rbp-370h]
  unsigned __int64 v384; // [rsp+A78h] [rbp-368h]
  int v385; // [rsp+A98h] [rbp-348h]
  unsigned __int64 v386; // [rsp+AA0h] [rbp-340h] BYREF
  int v387; // [rsp+AC0h] [rbp-320h]
  unsigned __int64 v388; // [rsp+AC8h] [rbp-318h]
  int v389; // [rsp+AE8h] [rbp-2F8h]
  unsigned __int64 v390; // [rsp+AF0h] [rbp-2F0h] BYREF
  int v391; // [rsp+B10h] [rbp-2D0h]
  unsigned __int64 v392; // [rsp+B18h] [rbp-2C8h]
  int v393; // [rsp+B38h] [rbp-2A8h]
  unsigned __int64 v394; // [rsp+B40h] [rbp-2A0h] BYREF
  int v395; // [rsp+B60h] [rbp-280h]
  unsigned __int64 v396; // [rsp+B68h] [rbp-278h]
  int v397; // [rsp+B88h] [rbp-258h]
  unsigned __int64 v398; // [rsp+B90h] [rbp-250h] BYREF
  int v399; // [rsp+BB0h] [rbp-230h]
  unsigned __int64 v400; // [rsp+BB8h] [rbp-228h]
  int v401; // [rsp+BD8h] [rbp-208h]
  char v402; // [rsp+BE0h] [rbp-200h] BYREF
  _QWORD v403[61]; // [rsp+BF8h] [rbp-1E8h] BYREF

  v403[0] = 8;
  if ( (unsigned int)sub_1308610("arenas.page", &v170, v403, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"arenas.page",
      v6,
      v7,
      v8,
      v9);
    abort();
  }
  v403[0] = 4;
  if ( (unsigned int)sub_1308610("arenas.nbins", &v167, v403, 0, 0) )
  {
    sub_130ACF0(
      (unsigned int)"<jemalloc>: Failure in xmallctl(\"%s\", ...)\n",
      (unsigned int)"arenas.nbins",
      v10,
      v11,
      v12,
      v13);
    abort();
  }
  v171 = 0;
  v172 = 0;
  sub_40E2CD((__int64)&v227, (__int64)&v172);
  v227 = v14;
  v228 = 6;
  sub_40E2CD((__int64)&v230, (__int64)&v171);
  v232 = "size";
  v230 = v15;
  v231 = 9;
  sub_40E2CD((__int64)&v233, v16);
  v233 = v17;
  v234 = 3;
  sub_40E2CD((__int64)&v236, (__int64)&v171);
  v238 = "ind";
  v236 = v18;
  v237 = 9;
  sub_40E2CD((__int64)&v239, v19);
  v239 = v20;
  v240 = 5;
  sub_40E2CD((__int64)&v242, (__int64)&v171);
  v244 = "allocated";
  v242 = v21;
  v243 = 9;
  sub_40E2CD((__int64)&v245, v22);
  v245 = v23;
  v246 = 5;
  sub_40E2CD((__int64)&v248, (__int64)&v171);
  v250 = "nmalloc";
  v248 = v24;
  v249 = 9;
  sub_40E2CD((__int64)&v251, v25);
  v251 = v26;
  v252 = 5;
  sub_40E2CD((__int64)&v254, (__int64)&v171);
  v254 = v27;
  v255 = 9;
  v256 = v28;
  sub_40E2CD((__int64)&v257, v29);
  v257 = v30;
  v258 = 5;
  sub_40E2CD((__int64)&v260, (__int64)&v171);
  v262 = "ndalloc";
  v260 = v31;
  v261 = 9;
  sub_40E2CD((__int64)&v263, v32);
  v263 = v33;
  v264 = 5;
  sub_40E2CD((__int64)&v266, (__int64)&v171);
  v266 = v34;
  v267 = 9;
  v268 = v35;
  sub_40E2CD((__int64)&v269, v36);
  v269 = v37;
  v270 = 5;
  sub_40E2CD((__int64)&v272, (__int64)&v171);
  v274 = "nrequests";
  v272 = v38;
  v273 = 9;
  sub_40E2CD((__int64)&v275, v39);
  v275 = 0xA00000001LL;
  v276 = 5;
  sub_40E2CD((__int64)&v278, (__int64)&v171);
  v278 = 0xA00000001LL;
  v279 = 9;
  v280 = v40;
  sub_40E2CD((__int64)&v281, v41);
  v281 = 0x900000001LL;
  v282 = 3;
  sub_40E2CD((__int64)&v284, (__int64)&v171);
  v284 = 0x900000001LL;
  v286 = "nshards";
  v285 = 9;
  sub_40E2CD((__int64)&v287, v42);
  v287 = v43;
  v288 = 6;
  sub_40E2CD((__int64)&v290, (__int64)&v171);
  v292 = "curregs";
  v290 = v44;
  v291 = 9;
  sub_40E2CD((__int64)&v293, v45);
  v293 = v46;
  v294 = 6;
  sub_40E2CD((__int64)&v296, (__int64)&v171);
  v298 = "curslabs";
  v296 = v47;
  v297 = 9;
  sub_40E2CD((__int64)&v299, v48);
  v299 = 0xF00000001LL;
  v300 = 6;
  sub_40E2CD((__int64)&v302, (__int64)&v171);
  v302 = 0xF00000001LL;
  v304 = "nonfull_slabs";
  v303 = 9;
  sub_40E2CD((__int64)&v305, v49);
  v305 = 0x500000001LL;
  v306 = 3;
  sub_40E2CD((__int64)&v308, (__int64)&v171);
  v308 = 0x500000001LL;
  v310 = "regs";
  v309 = 9;
  sub_40E2CD((__int64)&v311, v50);
  v311 = v51;
  v312 = 6;
  sub_40E2CD((__int64)&v314, (__int64)&v171);
  v316 = "pgs";
  v314 = v52;
  v315 = 9;
  sub_40E2CD((__int64)&v317, v53);
  v317 = v54;
  v318 = 9;
  sub_40E2CD((__int64)&v320, (__int64)&v171);
  v322 = "justify_spacer";
  v320 = v55;
  v321 = 9;
  sub_40E2CD((__int64)&v323, v56);
  v323 = v57;
  v324 = 9;
  sub_40E2CD((__int64)&v326, (__int64)&v171);
  v328 = "util";
  v326 = v58;
  v327 = 9;
  sub_40E2CD((__int64)&v329, v59);
  v329 = v60;
  v330 = 5;
  sub_40E2CD((__int64)&v332, (__int64)&v171);
  v334 = "nfills";
  v332 = v61;
  v333 = 9;
  sub_40E2CD((__int64)&v335, v62);
  v335 = v63;
  v336 = 5;
  sub_40E2CD((__int64)&v338, (__int64)&v171);
  v338 = v64;
  v339 = 9;
  v340 = v65;
  sub_40E2CD((__int64)&v341, v66);
  v341 = v67;
  v342 = 5;
  sub_40E2CD((__int64)&v344, (__int64)&v171);
  v346 = "nflushes";
  v344 = v68;
  v345 = 9;
  sub_40E2CD((__int64)&v347, v69);
  v347 = v70;
  v348 = 5;
  sub_40E2CD((__int64)&v350, (__int64)&v171);
  v350 = v71;
  v351 = 9;
  v352 = v72;
  sub_40E2CD((__int64)&v353, v73);
  v353 = v74;
  v354 = 5;
  sub_40E2CD((__int64)&v356, (__int64)&v171);
  v358 = "nslabs";
  v356 = v75;
  v357 = 9;
  sub_40E2CD((__int64)&v359, v76);
  v359 = v77;
  v360 = 5;
  sub_40E2CD((__int64)&v362, (__int64)&v171);
  v364 = "nreslabs";
  v362 = v78;
  v363 = 9;
  sub_40E2CD((__int64)&v365, v79);
  v365 = v80;
  v366 = 5;
  sub_40E2CD((__int64)&v368, (__int64)&v171);
  v368 = v83;
  v369 = 9;
  v370 = v85;
  v322 = " ";
  v319 = " ";
  if ( a2 )
  {
    sub_40E313(v84, 0, 0, (__int64)&v380, (__int64)v371, v84);
    sub_40E313((__int64)&v171, 0, 0, (__int64)v403, (__int64)v374, v86);
  }
  HIDWORD(v230) -= 5;
  sub_130F1C0((_DWORD)a1, (unsigned int)"bins:", v81, v82, v83, v84);
  if ( *a1 == 2 )
    sub_40ECF5((int)a1, &v171, v87, v88, v89, v90, (char)v161);
  sub_40EDA0((__int64)a1, (__int64)"bins", v87);
  v223 = 7;
  v91 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
    v91 = sub_1313D30(__readfsqword(0) - 2664, 0);
  if ( (unsigned int)sub_133D570(v91, v375, 0, "stats.arenas", &v223) )
    goto LABEL_12;
  v224 = 7;
  v376 = a3;
  v92 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
    v92 = sub_1313D30(v92, 0);
  if ( (unsigned int)sub_133D570(v92, v375, 3, "bins", &v224) )
    goto LABEL_12;
  v225 = 7;
  v93 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
    v93 = sub_1313D30(v93, 0);
  v94 = (__int64 *)v378;
  if ( (unsigned int)sub_133D570(v93, v378, 0, "arenas.bin", &v225) )
  {
LABEL_12:
    sub_130AA40("<jemalloc>: Failure in ctl_mibnametomib()\n");
    abort();
  }
  v97 = 1000000000;
  v98 = 0;
  v99 = a4 % 0x3B9ACA00;
  v164 = 0;
  v166 = a4 / 0x3B9ACA00;
  while ( v167 > v164 )
  {
    v185 = 7;
    v186 = 8;
    v377 = v164;
    v100 = __readfsqword(0) - 2664;
    v379 = v164;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v100) = sub_1313D30(v100, 0);
    v94 = (__int64 *)v375;
    if ( (unsigned int)sub_133D620(
                         v100,
                         (unsigned int)v375,
                         5,
                         (unsigned int)"nslabs",
                         (unsigned int)&v185,
                         (unsigned int)&v173,
                         (__int64)&v186,
                         0,
                         0) )
    {
LABEL_24:
      sub_130AA40("<jemalloc>: Failure in ctl_bymibname()\n");
      abort();
    }
    v162 = v173 == 0;
    if ( v173 )
    {
      if ( v98 )
        sub_130F1C0((_DWORD)a1, (unsigned int)"                     ---\n", v99, v97, v95, v96);
    }
    else if ( *a1 > 1 )
    {
      goto LABEL_30;
    }
    v187 = 7;
    v188 = 8;
    v101 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v101) = sub_1313D30(v101, 0);
    if ( (unsigned int)sub_133D620(
                         v101,
                         (unsigned int)v378,
                         3,
                         (unsigned int)"size",
                         (unsigned int)&v187,
                         (unsigned int)&v174,
                         (__int64)&v188,
                         0,
                         0) )
      goto LABEL_24;
    v189 = 7;
    v190 = 4;
    v102 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v102) = sub_1313D30(v102, 0);
    if ( (unsigned int)sub_133D620(
                         v102,
                         (unsigned int)v378,
                         3,
                         (unsigned int)"nregs",
                         (unsigned int)&v189,
                         (unsigned int)&v168,
                         (__int64)&v190,
                         0,
                         0) )
      goto LABEL_24;
    v191 = 7;
    v192 = 8;
    v103 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v103) = sub_1313D30(v103, 0);
    if ( (unsigned int)sub_133D620(
                         v103,
                         (unsigned int)v378,
                         3,
                         (unsigned int)"slab_size",
                         (unsigned int)&v191,
                         (unsigned int)&v175,
                         (__int64)&v192,
                         0,
                         0) )
      goto LABEL_24;
    v193 = 7;
    v194 = 4;
    v104 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v104) = sub_1313D30(v104, 0);
    if ( (unsigned int)sub_133D620(
                         v104,
                         (unsigned int)v378,
                         3,
                         (unsigned int)"nshards",
                         (unsigned int)&v193,
                         (unsigned int)&v169,
                         (__int64)&v194,
                         0,
                         0) )
      goto LABEL_24;
    v195 = 7;
    v196 = 8;
    v105 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v105) = sub_1313D30(v105, 0);
    if ( (unsigned int)sub_133D620(
                         v105,
                         (unsigned int)v375,
                         5,
                         (unsigned int)"nmalloc",
                         (unsigned int)&v195,
                         (unsigned int)&v179,
                         (__int64)&v196,
                         0,
                         0) )
      goto LABEL_24;
    v197 = 7;
    v198 = 8;
    v106 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v106) = sub_1313D30(v106, 0);
    if ( (unsigned int)sub_133D620(
                         v106,
                         (unsigned int)v375,
                         5,
                         (unsigned int)"ndalloc",
                         (unsigned int)&v197,
                         (unsigned int)&v180,
                         (__int64)&v198,
                         0,
                         0) )
      goto LABEL_24;
    v199 = 7;
    v200 = 8;
    v107 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v107) = sub_1313D30(v107, 0);
    if ( (unsigned int)sub_133D620(
                         v107,
                         (unsigned int)v375,
                         5,
                         (unsigned int)"curregs",
                         (unsigned int)&v199,
                         (unsigned int)&v176,
                         (__int64)&v200,
                         0,
                         0) )
      goto LABEL_24;
    v201 = 7;
    v202 = 8;
    v108 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v108) = sub_1313D30(v108, 0);
    if ( (unsigned int)sub_133D620(
                         v108,
                         (unsigned int)v375,
                         5,
                         (unsigned int)"nrequests",
                         (unsigned int)&v201,
                         (unsigned int)&v181,
                         (__int64)&v202,
                         0,
                         0) )
      goto LABEL_24;
    v203 = 7;
    v204 = 8;
    v109 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v109) = sub_1313D30(v109, 0);
    if ( (unsigned int)sub_133D620(
                         v109,
                         (unsigned int)v375,
                         5,
                         (unsigned int)"nfills",
                         (unsigned int)&v203,
                         (unsigned int)&v182,
                         (__int64)&v204,
                         0,
                         0) )
      goto LABEL_24;
    v205 = 7;
    v206 = 8;
    v110 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v110) = sub_1313D30(v110, 0);
    if ( (unsigned int)sub_133D620(
                         v110,
                         (unsigned int)v375,
                         5,
                         (unsigned int)"nflushes",
                         (unsigned int)&v205,
                         (unsigned int)&v183,
                         (__int64)&v206,
                         0,
                         0) )
      goto LABEL_24;
    v207 = 7;
    v208 = 8;
    v111 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v111) = sub_1313D30(v111, 0);
    if ( (unsigned int)sub_133D620(
                         v111,
                         (unsigned int)v375,
                         5,
                         (unsigned int)"nreslabs",
                         (unsigned int)&v207,
                         (unsigned int)&v184,
                         (__int64)&v208,
                         0,
                         0) )
      goto LABEL_24;
    v209 = 7;
    v210 = 8;
    v112 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v112) = sub_1313D30(v112, 0);
    if ( (unsigned int)sub_133D620(
                         v112,
                         (unsigned int)v375,
                         5,
                         (unsigned int)"curslabs",
                         (unsigned int)&v209,
                         (unsigned int)&v177,
                         (__int64)&v210,
                         0,
                         0) )
      goto LABEL_24;
    v211 = 7;
    v212 = 8;
    v113 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      LODWORD(v113) = sub_1313D30(v113, 0);
    if ( (unsigned int)sub_133D620(
                         v113,
                         (unsigned int)v375,
                         5,
                         (unsigned int)"nonfull_slabs",
                         (unsigned int)&v211,
                         (unsigned int)&v178,
                         (__int64)&v212,
                         0,
                         0) )
      goto LABEL_24;
    if ( a2 )
    {
      v226 = 7;
      v115 = __readfsqword(0) - 2664;
      if ( __readfsbyte(0xFFFFF8C8) )
        v115 = sub_1313D30(v115, 0);
      if ( (unsigned int)sub_133D570(v115, v375, 5, "mutex", &v226) )
        goto LABEL_12;
      v381 = 5;
      v213 = 7;
      v214 = 8;
      v116 = __readfsqword(0) - 2664;
      if ( __readfsbyte(0xFFFFF8C8) )
        LODWORD(v116) = sub_1313D30(v116, 0);
      v161 = v375;
      if ( (unsigned int)sub_133D620(
                           v116,
                           (unsigned int)v375,
                           6,
                           (unsigned int)"num_ops",
                           (unsigned int)&v213,
                           (unsigned int)&v382,
                           (__int64)&v214,
                           0,
                           0) )
        goto LABEL_24;
      v117 = v382;
      v383 = 5;
      if ( v382 && a4 )
      {
        if ( a4 > 0x3B9AC9FF )
          v117 = v382 / v166;
      }
      else
      {
        v117 = 0;
      }
      v384 = v117;
      v385 = 5;
      v215 = 7;
      v118 = __readfsqword(0) - 2664;
      v216 = 8;
      if ( __readfsbyte(0xFFFFF8C8) )
        LODWORD(v118) = sub_1313D30(v118, 0);
      if ( (unsigned int)sub_133D620(
                           v118,
                           (unsigned int)v375,
                           6,
                           (unsigned int)"num_wait",
                           (unsigned int)&v215,
                           (unsigned int)&v386,
                           (__int64)&v216,
                           0,
                           0) )
        goto LABEL_24;
      v119 = v386;
      v387 = 5;
      if ( v386 && a4 )
      {
        if ( a4 > 0x3B9AC9FF )
          v119 = v386 / v166;
      }
      else
      {
        v119 = 0;
      }
      v388 = v119;
      v389 = 5;
      v217 = 7;
      v218 = 8;
      v120 = __readfsqword(0) - 2664;
      if ( __readfsbyte(0xFFFFF8C8) )
        LODWORD(v120) = sub_1313D30(v120, 0);
      if ( (unsigned int)sub_133D620(
                           v120,
                           (unsigned int)v375,
                           6,
                           (unsigned int)"num_spin_acq",
                           (unsigned int)&v217,
                           (unsigned int)&v390,
                           (__int64)&v218,
                           0,
                           0) )
        goto LABEL_24;
      v121 = v390;
      v391 = 5;
      if ( v390 && a4 )
      {
        if ( a4 > 0x3B9AC9FF )
          v121 = v390 / v166;
      }
      else
      {
        v121 = 0;
      }
      v392 = v121;
      v393 = 5;
      v219 = 7;
      v220 = 8;
      v122 = __readfsqword(0) - 2664;
      if ( __readfsbyte(0xFFFFF8C8) )
        LODWORD(v122) = sub_1313D30(v122, 0);
      if ( (unsigned int)sub_133D620(
                           v122,
                           (unsigned int)v375,
                           6,
                           (unsigned int)"num_owner_switch",
                           (unsigned int)&v219,
                           (unsigned int)&v394,
                           (__int64)&v220,
                           0,
                           0) )
        goto LABEL_24;
      v123 = v394;
      v395 = 5;
      if ( v394 && a4 )
      {
        if ( a4 > 0x3B9AC9FF )
          v123 = v394 / v166;
      }
      else
      {
        v123 = 0;
      }
      v396 = v123;
      v397 = 5;
      v221 = 7;
      v222 = 8;
      v124 = __readfsqword(0) - 2664;
      if ( __readfsbyte(0xFFFFF8C8) )
        LODWORD(v124) = sub_1313D30(v124, 0);
      if ( (unsigned int)sub_133D620(
                           v124,
                           (unsigned int)v375,
                           6,
                           (unsigned int)"total_wait_time",
                           (unsigned int)&v221,
                           (unsigned int)&v398,
                           (__int64)&v222,
                           0,
                           0) )
        goto LABEL_24;
      v125 = v398;
      v399 = 5;
      if ( v398 && a4 )
      {
        if ( a4 > 0x3B9AC9FF )
          v125 = v398 / v166;
      }
      else
      {
        v125 = 0;
      }
      v400 = v125;
      v401 = 5;
      v223 = 7;
      v224 = 8;
      v126 = __readfsqword(0) - 2664;
      if ( __readfsbyte(0xFFFFF8C8) )
        LODWORD(v126) = sub_1313D30(v126, 0);
      if ( (unsigned int)sub_133D620(
                           v126,
                           (unsigned int)v375,
                           6,
                           (unsigned int)"max_wait_time",
                           (unsigned int)&v223,
                           (unsigned int)&v402,
                           (__int64)&v224,
                           0,
                           0) )
        goto LABEL_24;
      v372 = 4;
      v225 = 7;
      v226 = 4;
      v127 = __readfsqword(0) - 2664;
      if ( __readfsbyte(0xFFFFF8C8) )
        LODWORD(v127) = sub_1313D30(v127, 0);
      if ( (unsigned int)sub_133D620(
                           v127,
                           (unsigned int)v375,
                           6,
                           (unsigned int)"max_num_thds",
                           (unsigned int)&v225,
                           (unsigned int)&v373,
                           (__int64)&v226,
                           0,
                           0) )
        goto LABEL_24;
    }
    if ( *a1 <= 1 )
      sub_130F360(a1);
    sub_40EDDD((__int64)a1, (__int64)"nmalloc", 5, (const char **)&v179, v114);
    sub_40EDDD((__int64)a1, (__int64)"ndalloc", 5, (const char **)&v180, v128);
    sub_40EDDD((__int64)a1, (__int64)"curregs", 6, (const char **)&v176, v129);
    sub_40EDDD((__int64)a1, (__int64)"nrequests", 5, (const char **)&v181, v130);
    sub_40EDDD((__int64)a1, (__int64)"nfills", 5, (const char **)&v182, v131);
    sub_40EDDD((__int64)a1, (__int64)"nflushes", 5, (const char **)&v183, v132);
    sub_40EDDD((__int64)a1, (__int64)"nreslabs", 5, (const char **)&v184, v133);
    sub_40EDDD((__int64)a1, (__int64)"curslabs", 6, &v177, v134);
    v135 = "nonfull_slabs";
    sub_40EDDD((__int64)a1, (__int64)"nonfull_slabs", 6, &v178, v136);
    if ( a2 )
    {
      sub_130F560(a1, "mutex");
      v135 = 0;
      sub_40EE2B(a1, 0, &v380, (__int64)v371, v141, v142);
      sub_40E56D(a1, 0, v143, v144, v145, v146, (char)v161);
    }
    sub_40E56D(a1, (__int64)v135, v137, v138, v139, v140, (char)v161);
    v147 = (_QWORD)v177 * v168;
    if ( !v147 )
      goto LABEL_133;
    if ( v176 > v147 )
    {
      sub_40E1DF((__int64)&v226, 6u, " race");
      goto LABEL_134;
    }
    v148 = 1000 * v176 / v147;
    if ( (unsigned int)v148 <= 9 )
    {
      sub_40E1DF((__int64)&v226, 6u, "0.00%u", v148);
      goto LABEL_134;
    }
    if ( (unsigned int)v148 <= 0x63 )
    {
      sub_40E1DF((__int64)&v226, 6u, "0.0%u", v148);
      goto LABEL_134;
    }
    if ( (unsigned int)v148 <= 0x3E7 )
      sub_40E1DF((__int64)&v226, 6u, "0.%u", v148);
    else
LABEL_133:
      sub_40E1DF((__int64)&v226, 6u, "1");
LABEL_134:
    v94 = (__int64 *)v176;
    v229 = v174;
    v235 = v164;
    LOBYTE(v97) = a4 == 0;
    v241 = v176 * v174;
    v149 = v179;
    v247 = v179;
    if ( v179 && a4 )
    {
      if ( a4 > 0x3B9AC9FF )
        v149 = v179 / v166;
    }
    else
    {
      v149 = 0;
    }
    v253 = v149;
    v150 = v180;
    v259 = v180;
    if ( v180 && a4 )
    {
      if ( a4 > 0x3B9AC9FF )
        v150 = v180 / v166;
    }
    else
    {
      v150 = 0;
    }
    v265 = v150;
    v151 = v181;
    v271 = v181;
    if ( v181 && a4 )
    {
      if ( a4 > 0x3B9AC9FF )
        v151 = v181 / v166;
    }
    else
    {
      v151 = 0;
    }
    v277 = v151;
    v289 = v176;
    v283 = v169;
    v325 = &v226;
    v295 = v177;
    v301 = v178;
    v307 = v168;
    v99 = v175 % v170;
    v313 = v175 / v170;
    v152 = v182;
    v331 = v182;
    if ( v182 && a4 )
    {
      if ( a4 > 0x3B9AC9FF )
      {
        v152 = v182 / v166;
        v99 = v182 % v166;
      }
    }
    else
    {
      v152 = 0;
    }
    v337 = v152;
    v153 = v183;
    v343 = v183;
    if ( v183 && a4 )
    {
      if ( a4 > 0x3B9AC9FF )
      {
        v153 = v183 / v166;
        v99 = v183 % v166;
      }
    }
    else
    {
      v153 = 0;
    }
    v349 = v153;
    v355 = v173;
    v154 = v184;
    v361 = v184;
    if ( v184 && a4 )
    {
      if ( a4 > 0x3B9AC9FF )
      {
        v154 = v184 / v166;
        v99 = v184 % v166;
      }
    }
    else
    {
      v154 = 0;
    }
    v155 = *a1 == 2;
    v367 = v154;
    if ( v155 )
    {
      v94 = &v172;
      sub_40ECF5((int)a1, &v172, v99, v97, v95, v96, (char)v161);
    }
LABEL_30:
    ++v164;
    v98 = v162;
  }
  result = sub_40E525(a1, (__int64)v94, v99, v97, v95, v96, (char)v161);
  if ( v98 )
    return sub_130F1C0((_DWORD)a1, (unsigned int)"                     ---\n", v157, v158, v159, v160);
  return result;
}
