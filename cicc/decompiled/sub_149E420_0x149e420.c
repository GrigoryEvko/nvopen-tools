// Function: sub_149E420
// Address: 0x149e420
//
void __fastcall sub_149E420(__int64 a1, int a2)
{
  __m128i v2; // [rsp+0h] [rbp-C30h] BYREF
  const char *v3; // [rsp+10h] [rbp-C20h]
  __int64 v4; // [rsp+18h] [rbp-C18h]
  int v5; // [rsp+20h] [rbp-C10h]
  char *v6; // [rsp+28h] [rbp-C08h]
  __int64 v7; // [rsp+30h] [rbp-C00h]
  const char *v8; // [rsp+38h] [rbp-BF8h]
  __int64 v9; // [rsp+40h] [rbp-BF0h]
  int v10; // [rsp+48h] [rbp-BE8h]
  char *v11; // [rsp+50h] [rbp-BE0h]
  __int64 v12; // [rsp+58h] [rbp-BD8h]
  const char *v13; // [rsp+60h] [rbp-BD0h]
  __int64 v14; // [rsp+68h] [rbp-BC8h]
  int v15; // [rsp+70h] [rbp-BC0h]
  char *v16; // [rsp+78h] [rbp-BB8h]
  __int64 v17; // [rsp+80h] [rbp-BB0h]
  const char *v18; // [rsp+88h] [rbp-BA8h]
  __int64 v19; // [rsp+90h] [rbp-BA0h]
  int v20; // [rsp+98h] [rbp-B98h]
  char *v21; // [rsp+A0h] [rbp-B90h]
  __int64 v22; // [rsp+A8h] [rbp-B88h]
  const char *v23; // [rsp+B0h] [rbp-B80h]
  __int64 v24; // [rsp+B8h] [rbp-B78h]
  int v25; // [rsp+C0h] [rbp-B70h]
  char *v26; // [rsp+C8h] [rbp-B68h]
  __int64 v27; // [rsp+D0h] [rbp-B60h]
  const char *v28; // [rsp+D8h] [rbp-B58h]
  __int64 v29; // [rsp+E0h] [rbp-B50h]
  int v30; // [rsp+E8h] [rbp-B48h]
  char *v31; // [rsp+F0h] [rbp-B40h]
  __int64 v32; // [rsp+F8h] [rbp-B38h]
  const char *v33; // [rsp+100h] [rbp-B30h]
  __int64 v34; // [rsp+108h] [rbp-B28h]
  int v35; // [rsp+110h] [rbp-B20h]
  const char *v36; // [rsp+118h] [rbp-B18h]
  __int64 v37; // [rsp+120h] [rbp-B10h]
  const char *v38; // [rsp+128h] [rbp-B08h]
  __int64 v39; // [rsp+130h] [rbp-B00h]
  int v40; // [rsp+138h] [rbp-AF8h]
  char *v41; // [rsp+140h] [rbp-AF0h]
  __int64 v42; // [rsp+148h] [rbp-AE8h]
  const char *v43; // [rsp+150h] [rbp-AE0h]
  __int64 v44; // [rsp+158h] [rbp-AD8h]
  int v45; // [rsp+160h] [rbp-AD0h]
  char *v46; // [rsp+168h] [rbp-AC8h]
  __int64 v47; // [rsp+170h] [rbp-AC0h]
  const char *v48; // [rsp+178h] [rbp-AB8h]
  __int64 v49; // [rsp+180h] [rbp-AB0h]
  int v50; // [rsp+188h] [rbp-AA8h]
  const char *v51; // [rsp+190h] [rbp-AA0h]
  __int64 v52; // [rsp+198h] [rbp-A98h]
  const char *v53; // [rsp+1A0h] [rbp-A90h]
  __int64 v54; // [rsp+1A8h] [rbp-A88h]
  int v55; // [rsp+1B0h] [rbp-A80h]
  char *v56; // [rsp+1B8h] [rbp-A78h]
  __int64 v57; // [rsp+1C0h] [rbp-A70h]
  const char *v58; // [rsp+1C8h] [rbp-A68h]
  __int64 v59; // [rsp+1D0h] [rbp-A60h]
  int v60; // [rsp+1D8h] [rbp-A58h]
  char *v61; // [rsp+1E0h] [rbp-A50h]
  __int64 v62; // [rsp+1E8h] [rbp-A48h]
  const char *v63; // [rsp+1F0h] [rbp-A40h]
  __int64 v64; // [rsp+1F8h] [rbp-A38h]
  int v65; // [rsp+200h] [rbp-A30h]
  char *v66; // [rsp+208h] [rbp-A28h]
  __int64 v67; // [rsp+210h] [rbp-A20h]
  const char *v68; // [rsp+218h] [rbp-A18h]
  __int64 v69; // [rsp+220h] [rbp-A10h]
  int v70; // [rsp+228h] [rbp-A08h]
  char *v71; // [rsp+230h] [rbp-A00h]
  __int64 v72; // [rsp+238h] [rbp-9F8h]
  const char *v73; // [rsp+240h] [rbp-9F0h]
  __int64 v74; // [rsp+248h] [rbp-9E8h]
  int v75; // [rsp+250h] [rbp-9E0h]
  char *v76; // [rsp+258h] [rbp-9D8h]
  __int64 v77; // [rsp+260h] [rbp-9D0h]
  const char *v78; // [rsp+268h] [rbp-9C8h]
  __int64 v79; // [rsp+270h] [rbp-9C0h]
  int v80; // [rsp+278h] [rbp-9B8h]
  char *v81; // [rsp+280h] [rbp-9B0h]
  __int64 v82; // [rsp+288h] [rbp-9A8h]
  const char *v83; // [rsp+290h] [rbp-9A0h]
  __int64 v84; // [rsp+298h] [rbp-998h]
  int v85; // [rsp+2A0h] [rbp-990h]
  char *v86; // [rsp+2A8h] [rbp-988h]
  __int64 v87; // [rsp+2B0h] [rbp-980h]
  const char *v88; // [rsp+2B8h] [rbp-978h]
  __int64 v89; // [rsp+2C0h] [rbp-970h]
  int v90; // [rsp+2C8h] [rbp-968h]
  const char *v91; // [rsp+2D0h] [rbp-960h]
  __int64 v92; // [rsp+2D8h] [rbp-958h]
  const char *v93; // [rsp+2E0h] [rbp-950h]
  __int64 v94; // [rsp+2E8h] [rbp-948h]
  int v95; // [rsp+2F0h] [rbp-940h]
  char *v96; // [rsp+2F8h] [rbp-938h]
  __int64 v97; // [rsp+300h] [rbp-930h]
  const char *v98; // [rsp+308h] [rbp-928h]
  __int64 v99; // [rsp+310h] [rbp-920h]
  int v100; // [rsp+318h] [rbp-918h]
  char *v101; // [rsp+320h] [rbp-910h]
  __int64 v102; // [rsp+328h] [rbp-908h]
  const char *v103; // [rsp+330h] [rbp-900h]
  __int64 v104; // [rsp+338h] [rbp-8F8h]
  int v105; // [rsp+340h] [rbp-8F0h]
  char *v106; // [rsp+348h] [rbp-8E8h]
  __int64 v107; // [rsp+350h] [rbp-8E0h]
  const char *v108; // [rsp+358h] [rbp-8D8h]
  __int64 v109; // [rsp+360h] [rbp-8D0h]
  int v110; // [rsp+368h] [rbp-8C8h]
  char *v111; // [rsp+370h] [rbp-8C0h]
  __int64 v112; // [rsp+378h] [rbp-8B8h]
  const char *v113; // [rsp+380h] [rbp-8B0h]
  __int64 v114; // [rsp+388h] [rbp-8A8h]
  int v115; // [rsp+390h] [rbp-8A0h]
  char *v116; // [rsp+398h] [rbp-898h]
  __int64 v117; // [rsp+3A0h] [rbp-890h]
  const char *v118; // [rsp+3A8h] [rbp-888h]
  __int64 v119; // [rsp+3B0h] [rbp-880h]
  int v120; // [rsp+3B8h] [rbp-878h]
  char *v121; // [rsp+3C0h] [rbp-870h]
  __int64 v122; // [rsp+3C8h] [rbp-868h]
  const char *v123; // [rsp+3D0h] [rbp-860h]
  __int64 v124; // [rsp+3D8h] [rbp-858h]
  int v125; // [rsp+3E0h] [rbp-850h]
  char *v126; // [rsp+3E8h] [rbp-848h]
  __int64 v127; // [rsp+3F0h] [rbp-840h]
  const char *v128; // [rsp+3F8h] [rbp-838h]
  __int64 v129; // [rsp+400h] [rbp-830h]
  int v130; // [rsp+408h] [rbp-828h]
  char *v131; // [rsp+410h] [rbp-820h]
  __int64 v132; // [rsp+418h] [rbp-818h]
  const char *v133; // [rsp+420h] [rbp-810h]
  __int64 v134; // [rsp+428h] [rbp-808h]
  int v135; // [rsp+430h] [rbp-800h]
  char *v136; // [rsp+438h] [rbp-7F8h]
  __int64 v137; // [rsp+440h] [rbp-7F0h]
  const char *v138; // [rsp+448h] [rbp-7E8h]
  __int64 v139; // [rsp+450h] [rbp-7E0h]
  int v140; // [rsp+458h] [rbp-7D8h]
  char *v141; // [rsp+460h] [rbp-7D0h]
  __int64 v142; // [rsp+468h] [rbp-7C8h]
  const char *v143; // [rsp+470h] [rbp-7C0h]
  __int64 v144; // [rsp+478h] [rbp-7B8h]
  int v145; // [rsp+480h] [rbp-7B0h]
  char *v146; // [rsp+488h] [rbp-7A8h]
  __int64 v147; // [rsp+490h] [rbp-7A0h]
  const char *v148; // [rsp+498h] [rbp-798h]
  __int64 v149; // [rsp+4A0h] [rbp-790h]
  int v150; // [rsp+4A8h] [rbp-788h]
  char *v151; // [rsp+4B0h] [rbp-780h]
  __int64 v152; // [rsp+4B8h] [rbp-778h]
  const char *v153; // [rsp+4C0h] [rbp-770h]
  __int64 v154; // [rsp+4C8h] [rbp-768h]
  int v155; // [rsp+4D0h] [rbp-760h]
  char *v156; // [rsp+4D8h] [rbp-758h]
  __int64 v157; // [rsp+4E0h] [rbp-750h]
  const char *v158; // [rsp+4E8h] [rbp-748h]
  __int64 v159; // [rsp+4F0h] [rbp-740h]
  int v160; // [rsp+4F8h] [rbp-738h]
  char *v161; // [rsp+500h] [rbp-730h]
  __int64 v162; // [rsp+508h] [rbp-728h]
  const char *v163; // [rsp+510h] [rbp-720h]
  __int64 v164; // [rsp+518h] [rbp-718h]
  int v165; // [rsp+520h] [rbp-710h]
  char *v166; // [rsp+528h] [rbp-708h]
  __int64 v167; // [rsp+530h] [rbp-700h]
  const char *v168; // [rsp+538h] [rbp-6F8h]
  __int64 v169; // [rsp+540h] [rbp-6F0h]
  int v170; // [rsp+548h] [rbp-6E8h]
  char *v171; // [rsp+550h] [rbp-6E0h]
  __int64 v172; // [rsp+558h] [rbp-6D8h]
  const char *v173; // [rsp+560h] [rbp-6D0h]
  __int64 v174; // [rsp+568h] [rbp-6C8h]
  int v175; // [rsp+570h] [rbp-6C0h]
  char *v176; // [rsp+578h] [rbp-6B8h]
  __int64 v177; // [rsp+580h] [rbp-6B0h]
  const char *v178; // [rsp+588h] [rbp-6A8h]
  __int64 v179; // [rsp+590h] [rbp-6A0h]
  int v180; // [rsp+598h] [rbp-698h]
  const char *v181; // [rsp+5A0h] [rbp-690h]
  __int64 v182; // [rsp+5A8h] [rbp-688h]
  const char *v183; // [rsp+5B0h] [rbp-680h]
  __int64 v184; // [rsp+5B8h] [rbp-678h]
  int v185; // [rsp+5C0h] [rbp-670h]
  const char *v186; // [rsp+5C8h] [rbp-668h]
  __int64 v187; // [rsp+5D0h] [rbp-660h]
  const char *v188; // [rsp+5D8h] [rbp-658h]
  __int64 v189; // [rsp+5E0h] [rbp-650h]
  int v190; // [rsp+5E8h] [rbp-648h]
  const char *v191; // [rsp+5F0h] [rbp-640h]
  __int64 v192; // [rsp+5F8h] [rbp-638h]
  const char *v193; // [rsp+600h] [rbp-630h]
  __int64 v194; // [rsp+608h] [rbp-628h]
  int v195; // [rsp+610h] [rbp-620h]
  const char *v196; // [rsp+618h] [rbp-618h]
  __int64 v197; // [rsp+620h] [rbp-610h]
  const char *v198; // [rsp+628h] [rbp-608h]
  __int64 v199; // [rsp+630h] [rbp-600h]
  int v200; // [rsp+638h] [rbp-5F8h]
  const char *v201; // [rsp+640h] [rbp-5F0h]
  __int64 v202; // [rsp+648h] [rbp-5E8h]
  const char *v203; // [rsp+650h] [rbp-5E0h]
  __int64 v204; // [rsp+658h] [rbp-5D8h]
  int v205; // [rsp+660h] [rbp-5D0h]
  const char *v206; // [rsp+668h] [rbp-5C8h]
  __int64 v207; // [rsp+670h] [rbp-5C0h]
  const char *v208; // [rsp+678h] [rbp-5B8h]
  __int64 v209; // [rsp+680h] [rbp-5B0h]
  int v210; // [rsp+688h] [rbp-5A8h]
  char *v211; // [rsp+690h] [rbp-5A0h]
  __int64 v212; // [rsp+698h] [rbp-598h]
  const char *v213; // [rsp+6A0h] [rbp-590h]
  __int64 v214; // [rsp+6A8h] [rbp-588h]
  int v215; // [rsp+6B0h] [rbp-580h]
  char *v216; // [rsp+6B8h] [rbp-578h]
  __int64 v217; // [rsp+6C0h] [rbp-570h]
  const char *v218; // [rsp+6C8h] [rbp-568h]
  __int64 v219; // [rsp+6D0h] [rbp-560h]
  int v220; // [rsp+6D8h] [rbp-558h]
  char *v221; // [rsp+6E0h] [rbp-550h]
  __int64 v222; // [rsp+6E8h] [rbp-548h]
  const char *v223; // [rsp+6F0h] [rbp-540h]
  __int64 v224; // [rsp+6F8h] [rbp-538h]
  int v225; // [rsp+700h] [rbp-530h]
  char *v226; // [rsp+708h] [rbp-528h]
  __int64 v227; // [rsp+710h] [rbp-520h]
  const char *v228; // [rsp+718h] [rbp-518h]
  __int64 v229; // [rsp+720h] [rbp-510h]
  int v230; // [rsp+728h] [rbp-508h]
  char *v231; // [rsp+730h] [rbp-500h]
  __int64 v232; // [rsp+738h] [rbp-4F8h]
  const char *v233; // [rsp+740h] [rbp-4F0h]
  __int64 v234; // [rsp+748h] [rbp-4E8h]
  int v235; // [rsp+750h] [rbp-4E0h]
  char *v236; // [rsp+758h] [rbp-4D8h]
  __int64 v237; // [rsp+760h] [rbp-4D0h]
  const char *v238; // [rsp+768h] [rbp-4C8h]
  __int64 v239; // [rsp+770h] [rbp-4C0h]
  int v240; // [rsp+778h] [rbp-4B8h]
  char *v241; // [rsp+780h] [rbp-4B0h]
  __int64 v242; // [rsp+788h] [rbp-4A8h]
  const char *v243; // [rsp+790h] [rbp-4A0h]
  __int64 v244; // [rsp+798h] [rbp-498h]
  int v245; // [rsp+7A0h] [rbp-490h]
  char *v246; // [rsp+7A8h] [rbp-488h]
  __int64 v247; // [rsp+7B0h] [rbp-480h]
  const char *v248; // [rsp+7B8h] [rbp-478h]
  __int64 v249; // [rsp+7C0h] [rbp-470h]
  int v250; // [rsp+7C8h] [rbp-468h]
  char *v251; // [rsp+7D0h] [rbp-460h]
  __int64 v252; // [rsp+7D8h] [rbp-458h]
  const char *v253; // [rsp+7E0h] [rbp-450h]
  __int64 v254; // [rsp+7E8h] [rbp-448h]
  int v255; // [rsp+7F0h] [rbp-440h]
  char *v256; // [rsp+7F8h] [rbp-438h]
  __int64 v257; // [rsp+800h] [rbp-430h]
  const char *v258; // [rsp+808h] [rbp-428h]
  __int64 v259; // [rsp+810h] [rbp-420h]
  int v260; // [rsp+818h] [rbp-418h]
  char *v261; // [rsp+820h] [rbp-410h]
  __int64 v262; // [rsp+828h] [rbp-408h]
  const char *v263; // [rsp+830h] [rbp-400h]
  __int64 v264; // [rsp+838h] [rbp-3F8h]
  int v265; // [rsp+840h] [rbp-3F0h]
  char *v266; // [rsp+848h] [rbp-3E8h]
  __int64 v267; // [rsp+850h] [rbp-3E0h]
  const char *v268; // [rsp+858h] [rbp-3D8h]
  __int64 v269; // [rsp+860h] [rbp-3D0h]
  int v270; // [rsp+868h] [rbp-3C8h]
  const char *v271; // [rsp+870h] [rbp-3C0h]
  __int64 v272; // [rsp+878h] [rbp-3B8h]
  const char *v273; // [rsp+880h] [rbp-3B0h]
  __int64 v274; // [rsp+888h] [rbp-3A8h]
  int v275; // [rsp+890h] [rbp-3A0h]
  const char *v276; // [rsp+898h] [rbp-398h]
  __int64 v277; // [rsp+8A0h] [rbp-390h]
  const char *v278; // [rsp+8A8h] [rbp-388h]
  __int64 v279; // [rsp+8B0h] [rbp-380h]
  int v280; // [rsp+8B8h] [rbp-378h]
  const char *v281; // [rsp+8C0h] [rbp-370h]
  __int64 v282; // [rsp+8C8h] [rbp-368h]
  const char *v283; // [rsp+8D0h] [rbp-360h]
  __int64 v284; // [rsp+8D8h] [rbp-358h]
  int v285; // [rsp+8E0h] [rbp-350h]
  const char *v286; // [rsp+8E8h] [rbp-348h]
  __int64 v287; // [rsp+8F0h] [rbp-340h]
  const char *v288; // [rsp+8F8h] [rbp-338h]
  __int64 v289; // [rsp+900h] [rbp-330h]
  int v290; // [rsp+908h] [rbp-328h]
  const char *v291; // [rsp+910h] [rbp-320h]
  __int64 v292; // [rsp+918h] [rbp-318h]
  const char *v293; // [rsp+920h] [rbp-310h]
  __int64 v294; // [rsp+928h] [rbp-308h]
  int v295; // [rsp+930h] [rbp-300h]
  const char *v296; // [rsp+938h] [rbp-2F8h]
  __int64 v297; // [rsp+940h] [rbp-2F0h]
  const char *v298; // [rsp+948h] [rbp-2E8h]
  __int64 v299; // [rsp+950h] [rbp-2E0h]
  int v300; // [rsp+958h] [rbp-2D8h]
  char *v301; // [rsp+960h] [rbp-2D0h]
  __int64 v302; // [rsp+968h] [rbp-2C8h]
  const char *v303; // [rsp+970h] [rbp-2C0h]
  __int64 v304; // [rsp+978h] [rbp-2B8h]
  int v305; // [rsp+980h] [rbp-2B0h]
  char *v306; // [rsp+988h] [rbp-2A8h]
  __int64 v307; // [rsp+990h] [rbp-2A0h]
  const char *v308; // [rsp+998h] [rbp-298h]
  __int64 v309; // [rsp+9A0h] [rbp-290h]
  int v310; // [rsp+9A8h] [rbp-288h]
  char *v311; // [rsp+9B0h] [rbp-280h]
  __int64 v312; // [rsp+9B8h] [rbp-278h]
  const char *v313; // [rsp+9C0h] [rbp-270h]
  __int64 v314; // [rsp+9C8h] [rbp-268h]
  int v315; // [rsp+9D0h] [rbp-260h]
  char *v316; // [rsp+9D8h] [rbp-258h]
  __int64 v317; // [rsp+9E0h] [rbp-250h]
  const char *v318; // [rsp+9E8h] [rbp-248h]
  __int64 v319; // [rsp+9F0h] [rbp-240h]
  int v320; // [rsp+9F8h] [rbp-238h]
  char *v321; // [rsp+A00h] [rbp-230h]
  __int64 v322; // [rsp+A08h] [rbp-228h]
  const char *v323; // [rsp+A10h] [rbp-220h]
  __int64 v324; // [rsp+A18h] [rbp-218h]
  int v325; // [rsp+A20h] [rbp-210h]
  char *v326; // [rsp+A28h] [rbp-208h]
  __int64 v327; // [rsp+A30h] [rbp-200h]
  const char *v328; // [rsp+A38h] [rbp-1F8h]
  __int64 v329; // [rsp+A40h] [rbp-1F0h]
  int v330; // [rsp+A48h] [rbp-1E8h]
  char *v331; // [rsp+A50h] [rbp-1E0h]
  __int64 v332; // [rsp+A58h] [rbp-1D8h]
  const char *v333; // [rsp+A60h] [rbp-1D0h]
  __int64 v334; // [rsp+A68h] [rbp-1C8h]
  int v335; // [rsp+A70h] [rbp-1C0h]
  char *v336; // [rsp+A78h] [rbp-1B8h]
  __int64 v337; // [rsp+A80h] [rbp-1B0h]
  const char *v338; // [rsp+A88h] [rbp-1A8h]
  __int64 v339; // [rsp+A90h] [rbp-1A0h]
  int v340; // [rsp+A98h] [rbp-198h]
  char *v341; // [rsp+AA0h] [rbp-190h]
  __int64 v342; // [rsp+AA8h] [rbp-188h]
  const char *v343; // [rsp+AB0h] [rbp-180h]
  __int64 v344; // [rsp+AB8h] [rbp-178h]
  int v345; // [rsp+AC0h] [rbp-170h]
  char *v346; // [rsp+AC8h] [rbp-168h]
  __int64 v347; // [rsp+AD0h] [rbp-160h]
  const char *v348; // [rsp+AD8h] [rbp-158h]
  __int64 v349; // [rsp+AE0h] [rbp-150h]
  int v350; // [rsp+AE8h] [rbp-148h]
  char *v351; // [rsp+AF0h] [rbp-140h]
  __int64 v352; // [rsp+AF8h] [rbp-138h]
  const char *v353; // [rsp+B00h] [rbp-130h]
  __int64 v354; // [rsp+B08h] [rbp-128h]
  int v355; // [rsp+B10h] [rbp-120h]
  char *v356; // [rsp+B18h] [rbp-118h]
  __int64 v357; // [rsp+B20h] [rbp-110h]
  const char *v358; // [rsp+B28h] [rbp-108h]
  __int64 v359; // [rsp+B30h] [rbp-100h]
  int v360; // [rsp+B38h] [rbp-F8h]
  const char *v361; // [rsp+B40h] [rbp-F0h]
  __int64 v362; // [rsp+B48h] [rbp-E8h]
  const char *v363; // [rsp+B50h] [rbp-E0h]
  __int64 v364; // [rsp+B58h] [rbp-D8h]
  int v365; // [rsp+B60h] [rbp-D0h]
  const char *v366; // [rsp+B68h] [rbp-C8h]
  __int64 v367; // [rsp+B70h] [rbp-C0h]
  const char *v368; // [rsp+B78h] [rbp-B8h]
  __int64 v369; // [rsp+B80h] [rbp-B0h]
  int v370; // [rsp+B88h] [rbp-A8h]
  const char *v371; // [rsp+B90h] [rbp-A0h]
  __int64 v372; // [rsp+B98h] [rbp-98h]
  const char *v373; // [rsp+BA0h] [rbp-90h]
  __int64 v374; // [rsp+BA8h] [rbp-88h]
  int v375; // [rsp+BB0h] [rbp-80h]
  const char *v376; // [rsp+BB8h] [rbp-78h]
  __int64 v377; // [rsp+BC0h] [rbp-70h]
  const char *v378; // [rsp+BC8h] [rbp-68h]
  __int64 v379; // [rsp+BD0h] [rbp-60h]
  int v380; // [rsp+BD8h] [rbp-58h]
  const char *v381; // [rsp+BE0h] [rbp-50h]
  __int64 v382; // [rsp+BE8h] [rbp-48h]
  const char *v383; // [rsp+BF0h] [rbp-40h]
  __int64 v384; // [rsp+BF8h] [rbp-38h]
  int v385; // [rsp+C00h] [rbp-30h]
  const char *v386; // [rsp+C08h] [rbp-28h]
  __int64 v387; // [rsp+C10h] [rbp-20h]
  const char *v388; // [rsp+C18h] [rbp-18h]
  __int64 v389; // [rsp+C20h] [rbp-10h]
  int v390; // [rsp+C28h] [rbp-8h]

  if ( a2 == 1 )
  {
    v2.m128i_i64[1] = 5;
    v2.m128i_i64[0] = (__int64)"ceilf";
    v3 = "vceilf";
    v6 = "fabsf";
    v8 = "vfabsf";
    v13 = "vfabsf";
    v16 = "floorf";
    v18 = "vfloorf";
    v21 = "sqrtf";
    v23 = "vsqrtf";
    v28 = "vsqrtf";
    v11 = "llvm.fabs.f32";
    v31 = "expf";
    v26 = "llvm.sqrt.f32";
    v4 = 6;
    v5 = 4;
    v7 = 5;
    v9 = 6;
    v10 = 4;
    v12 = 13;
    v14 = 6;
    v15 = 4;
    v17 = 6;
    v19 = 7;
    v20 = 4;
    v22 = 5;
    v24 = 6;
    v25 = 4;
    v27 = 13;
    v29 = 6;
    v30 = 4;
    v32 = 4;
    v33 = "vexpf";
    v38 = "vexpf";
    v41 = "expm1f";
    v43 = "vexpm1f";
    v46 = "logf";
    v48 = "vlogf";
    v53 = "vlogf";
    v56 = "log1pf";
    v36 = "llvm.exp.f32";
    v58 = "vlog1pf";
    v51 = "llvm.log.f32";
    v61 = "log10f";
    v63 = "vlog10f";
    v34 = 5;
    v35 = 4;
    v37 = 12;
    v39 = 5;
    v40 = 4;
    v42 = 6;
    v44 = 7;
    v45 = 4;
    v47 = 4;
    v49 = 5;
    v50 = 4;
    v52 = 12;
    v54 = 5;
    v55 = 4;
    v57 = 6;
    v59 = 7;
    v60 = 4;
    v62 = 6;
    v64 = 7;
    v65 = 4;
    v66 = "llvm.log10.f32";
    v68 = "vlog10f";
    v71 = "logbf";
    v73 = "vlogbf";
    v76 = "sinf";
    v78 = "vsinf";
    v83 = "vsinf";
    v86 = "cosf";
    v88 = "vcosf";
    v93 = "vcosf";
    v96 = "tanf";
    v81 = "llvm.sin.f32";
    v98 = "vtanf";
    v91 = "llvm.cos.f32";
    v67 = 14;
    v69 = 7;
    v70 = 4;
    v72 = 5;
    v74 = 6;
    v75 = 4;
    v77 = 4;
    v79 = 5;
    v80 = 4;
    v82 = 12;
    v84 = 5;
    v85 = 4;
    v87 = 4;
    v89 = 5;
    v90 = 4;
    v92 = 12;
    v94 = 5;
    v95 = 4;
    v97 = 4;
    v99 = 5;
    v101 = "asinf";
    v103 = "vasinf";
    v106 = "acosf";
    v108 = "vacosf";
    v111 = "atanf";
    v113 = "vatanf";
    v116 = "sinhf";
    v118 = "vsinhf";
    v121 = "coshf";
    v123 = "vcoshf";
    v126 = "tanhf";
    v128 = "vtanhf";
    v131 = "asinhf";
    v100 = 4;
    v102 = 5;
    v104 = 6;
    v105 = 4;
    v107 = 5;
    v109 = 6;
    v110 = 4;
    v112 = 5;
    v114 = 6;
    v115 = 4;
    v117 = 5;
    v119 = 6;
    v120 = 4;
    v122 = 5;
    v124 = 6;
    v125 = 4;
    v127 = 5;
    v129 = 6;
    v130 = 4;
    v132 = 6;
    v133 = "vasinhf";
    v136 = "acoshf";
    v138 = "vacoshf";
    v141 = "atanhf";
    v134 = 7;
    v135 = 4;
    v137 = 6;
    v139 = 7;
    v140 = 4;
    v142 = 6;
    v143 = "vatanhf";
    v144 = 7;
    v145 = 4;
    sub_149E2E0(a1, &v2, 29);
  }
  else if ( a2 == 2 )
  {
    v2.m128i_i64[1] = 3;
    v2.m128i_i64[0] = (__int64)"sin";
    v6 = "sin";
    v11 = "sin";
    v16 = "sinf";
    v21 = "sinf";
    v26 = "sinf";
    v3 = "__svml_sin2";
    v8 = "__svml_sin4";
    v13 = "__svml_sin8";
    v18 = "__svml_sinf4";
    v23 = "__svml_sinf8";
    v28 = "__svml_sinf16";
    v31 = "llvm.sin.f64";
    v4 = 11;
    v5 = 2;
    v7 = 3;
    v9 = 11;
    v10 = 4;
    v12 = 3;
    v14 = 11;
    v15 = 8;
    v17 = 4;
    v19 = 12;
    v20 = 4;
    v22 = 4;
    v24 = 12;
    v25 = 8;
    v27 = 4;
    v29 = 13;
    v30 = 16;
    v32 = 12;
    v33 = "__svml_sin2";
    v36 = "llvm.sin.f64";
    v41 = "llvm.sin.f64";
    v46 = "llvm.sin.f32";
    v51 = "llvm.sin.f32";
    v56 = "llvm.sin.f32";
    v38 = "__svml_sin4";
    v43 = "__svml_sin8";
    v48 = "__svml_sinf4";
    v53 = "__svml_sinf8";
    v58 = "__svml_sinf16";
    v61 = "cos";
    v63 = "__svml_cos2";
    v34 = 11;
    v35 = 2;
    v37 = 12;
    v39 = 11;
    v40 = 4;
    v42 = 12;
    v44 = 11;
    v45 = 8;
    v47 = 12;
    v49 = 12;
    v50 = 4;
    v52 = 12;
    v54 = 12;
    v55 = 8;
    v57 = 12;
    v59 = 13;
    v60 = 16;
    v62 = 3;
    v64 = 11;
    v65 = 2;
    v66 = "cos";
    v71 = "cos";
    v76 = "cosf";
    v81 = "cosf";
    v86 = "cosf";
    v68 = "__svml_cos4";
    v73 = "__svml_cos8";
    v78 = "__svml_cosf4";
    v83 = "__svml_cosf8";
    v88 = "__svml_cosf16";
    v91 = "llvm.cos.f64";
    v93 = "__svml_cos2";
    v96 = "llvm.cos.f64";
    v98 = "__svml_cos4";
    v67 = 3;
    v69 = 11;
    v70 = 4;
    v72 = 3;
    v74 = 11;
    v75 = 8;
    v77 = 4;
    v79 = 12;
    v80 = 4;
    v82 = 4;
    v84 = 12;
    v85 = 8;
    v87 = 4;
    v89 = 13;
    v90 = 16;
    v92 = 12;
    v94 = 11;
    v95 = 2;
    v97 = 12;
    v99 = 11;
    v101 = "llvm.cos.f64";
    v106 = "llvm.cos.f32";
    v111 = "llvm.cos.f32";
    v116 = "llvm.cos.f32";
    v103 = "__svml_cos8";
    v121 = "pow";
    v126 = "pow";
    v131 = "pow";
    v108 = "__svml_cosf4";
    v113 = "__svml_cosf8";
    v118 = "__svml_cosf16";
    v123 = "__svml_pow2";
    v128 = "__svml_pow4";
    v100 = 4;
    v102 = 12;
    v104 = 11;
    v105 = 8;
    v107 = 12;
    v109 = 12;
    v110 = 4;
    v112 = 12;
    v114 = 12;
    v115 = 8;
    v117 = 12;
    v119 = 13;
    v120 = 16;
    v122 = 3;
    v124 = 11;
    v125 = 2;
    v127 = 3;
    v129 = 11;
    v130 = 4;
    v132 = 3;
    v136 = "powf";
    v141 = "powf";
    v146 = "powf";
    v151 = "__pow_finite";
    v156 = "__pow_finite";
    v161 = "__pow_finite";
    v133 = "__svml_pow8";
    v138 = "__svml_powf4";
    v143 = "__svml_powf8";
    v148 = "__svml_powf16";
    v153 = "__svml_pow2";
    v158 = "__svml_pow4";
    v163 = "__svml_pow8";
    v134 = 11;
    v135 = 8;
    v137 = 4;
    v139 = 12;
    v140 = 4;
    v142 = 4;
    v144 = 12;
    v145 = 8;
    v147 = 4;
    v149 = 13;
    v150 = 16;
    v152 = 12;
    v154 = 11;
    v155 = 2;
    v157 = 12;
    v159 = 11;
    v160 = 4;
    v162 = 12;
    v164 = 11;
    v165 = 8;
    v166 = "__powf_finite";
    v171 = "__powf_finite";
    v176 = "__powf_finite";
    v193 = "__svml_pow8";
    v168 = "__svml_powf4";
    v173 = "__svml_powf8";
    v178 = "__svml_powf16";
    v181 = "llvm.pow.f64";
    v183 = "__svml_pow2";
    v186 = "llvm.pow.f64";
    v188 = "__svml_pow4";
    v191 = "llvm.pow.f64";
    v196 = "llvm.pow.f32";
    v167 = 13;
    v169 = 12;
    v170 = 4;
    v172 = 13;
    v174 = 12;
    v175 = 8;
    v177 = 13;
    v179 = 13;
    v180 = 16;
    v182 = 12;
    v184 = 11;
    v185 = 2;
    v187 = 12;
    v189 = 11;
    v190 = 4;
    v192 = 12;
    v194 = 11;
    v195 = 8;
    v197 = 12;
    v198 = "__svml_powf4";
    v208 = "__svml_powf16";
    v201 = "llvm.pow.f32";
    v206 = "llvm.pow.f32";
    v211 = "exp";
    v216 = "exp";
    v221 = "exp";
    v203 = "__svml_powf8";
    v213 = "__svml_exp2";
    v218 = "__svml_exp4";
    v223 = "__svml_exp8";
    v226 = "expf";
    v228 = "__svml_expf4";
    v199 = 12;
    v200 = 4;
    v202 = 12;
    v204 = 12;
    v205 = 8;
    v207 = 12;
    v209 = 13;
    v210 = 16;
    v212 = 3;
    v214 = 11;
    v215 = 2;
    v217 = 3;
    v219 = 11;
    v220 = 4;
    v222 = 3;
    v224 = 11;
    v225 = 8;
    v227 = 4;
    v229 = 12;
    v230 = 4;
    v231 = "expf";
    v236 = "expf";
    v241 = "__exp_finite";
    v246 = "__exp_finite";
    v251 = "__exp_finite";
    v233 = "__svml_expf8";
    v238 = "__svml_expf16";
    v243 = "__svml_exp2";
    v248 = "__svml_exp4";
    v253 = "__svml_exp8";
    v256 = "__expf_finite";
    v258 = "__svml_expf4";
    v261 = "__expf_finite";
    v263 = "__svml_expf8";
    v232 = 4;
    v234 = 12;
    v235 = 8;
    v237 = 4;
    v239 = 13;
    v240 = 16;
    v242 = 12;
    v244 = 11;
    v245 = 2;
    v247 = 12;
    v249 = 11;
    v250 = 4;
    v252 = 12;
    v254 = 11;
    v255 = 8;
    v257 = 13;
    v259 = 12;
    v260 = 4;
    v262 = 13;
    v264 = 12;
    v266 = "__expf_finite";
    v283 = "__svml_exp8";
    v268 = "__svml_expf16";
    v271 = "llvm.exp.f64";
    v273 = "__svml_exp2";
    v276 = "llvm.exp.f64";
    v278 = "__svml_exp4";
    v281 = "llvm.exp.f64";
    v286 = "llvm.exp.f32";
    v288 = "__svml_expf4";
    v291 = "llvm.exp.f32";
    v293 = "__svml_expf8";
    v296 = "llvm.exp.f32";
    v265 = 8;
    v267 = 13;
    v269 = 13;
    v270 = 16;
    v272 = 12;
    v274 = 11;
    v275 = 2;
    v277 = 12;
    v279 = 11;
    v280 = 4;
    v282 = 12;
    v284 = 11;
    v285 = 8;
    v287 = 12;
    v289 = 12;
    v290 = 4;
    v292 = 12;
    v294 = 12;
    v295 = 8;
    v297 = 12;
    v298 = "__svml_expf16";
    v301 = "log";
    v306 = "log";
    v311 = "log";
    v316 = "logf";
    v321 = "logf";
    v326 = "logf";
    v313 = "__svml_log8";
    v318 = "__svml_logf4";
    v299 = 13;
    v300 = 16;
    v302 = 3;
    v303 = "__svml_log2";
    v304 = 11;
    v305 = 2;
    v307 = 3;
    v308 = "__svml_log4";
    v309 = 11;
    v310 = 4;
    v312 = 3;
    v314 = 11;
    v315 = 8;
    v317 = 4;
    v319 = 12;
    v320 = 4;
    v322 = 4;
    v323 = "__svml_logf8";
    v324 = 12;
    v325 = 8;
    v327 = 4;
    v328 = "__svml_logf16";
    v329 = 13;
    v330 = 16;
    v331 = "__log_finite";
    v336 = "__log_finite";
    v341 = "__log_finite";
    v346 = "__logf_finite";
    v351 = "__logf_finite";
    v356 = "__logf_finite";
    v343 = "__svml_log8";
    v348 = "__svml_logf4";
    v332 = 12;
    v333 = "__svml_log2";
    v334 = 11;
    v335 = 2;
    v337 = 12;
    v338 = "__svml_log4";
    v339 = 11;
    v340 = 4;
    v342 = 12;
    v344 = 11;
    v345 = 8;
    v347 = 13;
    v349 = 12;
    v350 = 4;
    v352 = 13;
    v353 = "__svml_logf8";
    v354 = 12;
    v355 = 8;
    v357 = 13;
    v358 = "__svml_logf16";
    v359 = 13;
    v360 = 16;
    v361 = "llvm.log.f64";
    v362 = 12;
    v363 = "__svml_log2";
    v373 = "__svml_log8";
    v376 = "llvm.log.f32";
    v378 = "__svml_logf4";
    v381 = "llvm.log.f32";
    v386 = "llvm.log.f32";
    v364 = 11;
    v365 = 2;
    v366 = "llvm.log.f64";
    v367 = 12;
    v368 = "__svml_log4";
    v369 = 11;
    v370 = 4;
    v371 = "llvm.log.f64";
    v372 = 12;
    v374 = 11;
    v375 = 8;
    v377 = 12;
    v379 = 12;
    v380 = 4;
    v382 = 12;
    v383 = "__svml_logf8";
    v384 = 12;
    v385 = 8;
    v387 = 12;
    v388 = "__svml_logf16";
    v389 = 13;
    v390 = 16;
    sub_149E2E0(a1, &v2, 78);
  }
}
