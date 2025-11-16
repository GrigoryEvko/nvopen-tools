// Function: sub_1BB6740
// Address: 0x1bb6740
//
__int64 __fastcall sub_1BB6740(
        unsigned __int8 *a1,
        __int64 a2,
        __m128i a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13)
{
  __int64 v14; // rbx
  __int64 v15; // r14
  unsigned int v16; // r12d
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // r9
  __int64 v21; // rsi
  __int64 v22; // r8
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rax
  _BYTE *v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // r8
  __int64 v29; // rdi
  __int64 v30; // r9
  __int64 v31; // r10
  unsigned int v32; // r14d
  __int64 v33; // rsi
  __int64 *v34; // r15
  __int64 *v35; // r13
  __int64 v36; // r14
  __int64 v37; // rbx
  __int64 v38; // r13
  unsigned __int64 v39; // rdi
  __int64 v40; // rdi
  _QWORD *v41; // rbx
  _QWORD *v42; // r13
  unsigned __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rcx
  unsigned __int64 v46; // rax
  unsigned int v47; // eax
  char v48; // al
  char v49; // al
  __int64 v50; // rcx
  __int64 v51; // rax
  int v52; // eax
  char v53; // dl
  __int64 v54; // r9
  __int64 v55; // r8
  __int64 v56; // rdx
  __int64 v57; // rsi
  __int64 v58; // rcx
  _BYTE *v59; // rdi
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rcx
  char v63; // r9
  bool v64; // r14
  __int64 v65; // rax
  char v66; // r9
  __int64 *v67; // r14
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // rdx
  __int64 *v73; // rsi
  _QWORD *v74; // rdi
  __int64 v75; // rax
  __int64 v76; // rdx
  double v77; // xmm4_8
  double v78; // xmm5_8
  __int64 v79; // rax
  __int64 v80; // rax
  _BYTE *v81; // rbx
  _BYTE *v82; // r13
  __int64 v83; // r14
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 *v87; // rsi
  _QWORD *v88; // rdi
  __int64 v89; // r8
  __int64 v90; // rax
  __int64 v91; // rdx
  double v92; // xmm4_8
  double v93; // xmm5_8
  __int64 v94; // rax
  __int64 v95; // r14
  __int64 v96; // rax
  _QWORD *v97; // r12
  __int64 v98; // rax
  char *v99; // rsi
  size_t v100; // rdx
  _QWORD *v101; // rdi
  char v102; // r9
  __int64 v103; // rax
  __int64 *v104; // r12
  __int64 v105; // rax
  __int64 *v106; // r12
  __int64 v107; // rax
  __int64 v108; // r15
  __int64 *v109; // r14
  __int64 v110; // rax
  __int64 v111; // rax
  int v112; // r8d
  __int64 v113; // r9
  __int64 v114; // rax
  __int64 v115; // r12
  bool v116; // bl
  __int64 v117; // r13
  __int64 v118; // r14
  __int64 v119; // r11
  _BYTE *v120; // rdi
  _QWORD *v121; // rax
  unsigned __int64 v122; // rdx
  _BOOL4 v123; // eax
  __int64 v124; // rax
  bool v125; // al
  __int64 v126; // rax
  __int64 v127; // rax
  __int64 v128; // rax
  __int64 v129; // rax
  __int64 v130; // rax
  __int64 *v131; // r14
  __int64 v132; // r9
  __int64 v133; // rax
  __int64 v134; // r8
  int v135; // r9d
  __int64 v136; // rax
  unsigned __int8 *v137; // r14
  double v138; // xmm4_8
  double v139; // xmm5_8
  __int64 v140; // rax
  __int64 v141; // rax
  __int64 v142; // rax
  __int64 v143; // rax
  __int64 v144; // rax
  __int64 v145; // rax
  __int64 v146; // rax
  __int64 v147; // rax
  __int64 v148; // [rsp+8h] [rbp-FB8h]
  bool v149; // [rsp+10h] [rbp-FB0h]
  __int64 v150; // [rsp+10h] [rbp-FB0h]
  __int64 v151; // [rsp+28h] [rbp-F98h]
  __int64 v152; // [rsp+28h] [rbp-F98h]
  __int64 v153; // [rsp+28h] [rbp-F98h]
  unsigned int v154; // [rsp+30h] [rbp-F90h]
  __int64 v155; // [rsp+30h] [rbp-F90h]
  unsigned __int8 v156; // [rsp+30h] [rbp-F90h]
  unsigned __int64 v157; // [rsp+38h] [rbp-F88h]
  __int64 v158; // [rsp+38h] [rbp-F88h]
  __int64 v159; // [rsp+40h] [rbp-F80h]
  char v160; // [rsp+40h] [rbp-F80h]
  __int64 v161; // [rsp+48h] [rbp-F78h]
  char v162; // [rsp+48h] [rbp-F78h]
  __int64 v163; // [rsp+48h] [rbp-F78h]
  char v164; // [rsp+50h] [rbp-F70h]
  unsigned int v165; // [rsp+50h] [rbp-F70h]
  __int64 v166; // [rsp+50h] [rbp-F70h]
  __int64 v167; // [rsp+50h] [rbp-F70h]
  char v168; // [rsp+50h] [rbp-F70h]
  __int64 v169; // [rsp+50h] [rbp-F70h]
  __int64 *v170; // [rsp+58h] [rbp-F68h]
  __int64 *v171; // [rsp+58h] [rbp-F68h]
  int v172; // [rsp+60h] [rbp-F60h]
  __int64 v173; // [rsp+78h] [rbp-F48h] BYREF
  int v174; // [rsp+80h] [rbp-F40h] BYREF
  __int64 v175; // [rsp+88h] [rbp-F38h]
  __int64 v176; // [rsp+90h] [rbp-F30h]
  __m128i v177[2]; // [rsp+A0h] [rbp-F20h] BYREF
  const char *v178; // [rsp+C0h] [rbp-F00h]
  __int64 v179; // [rsp+C8h] [rbp-EF8h]
  char *v180; // [rsp+D0h] [rbp-EF0h] BYREF
  size_t v181; // [rsp+D8h] [rbp-EE8h]
  char v182; // [rsp+E0h] [rbp-EE0h] BYREF
  const char *v183; // [rsp+F0h] [rbp-ED0h]
  __int64 v184; // [rsp+F8h] [rbp-EC8h]
  char *v185; // [rsp+100h] [rbp-EC0h] BYREF
  size_t v186; // [rsp+108h] [rbp-EB8h]
  char v187; // [rsp+110h] [rbp-EB0h] BYREF
  char v188[8]; // [rsp+120h] [rbp-EA0h] BYREF
  unsigned int v189; // [rsp+128h] [rbp-E98h]
  unsigned int v190; // [rsp+138h] [rbp-E88h]
  int v191; // [rsp+148h] [rbp-E78h]
  __m128i v192; // [rsp+150h] [rbp-E70h] BYREF
  char v193; // [rsp+160h] [rbp-E60h]
  __m128i v194; // [rsp+180h] [rbp-E40h] BYREF
  __int64 v195; // [rsp+190h] [rbp-E30h] BYREF
  __int64 *v196; // [rsp+1A0h] [rbp-E20h]
  __int64 v197; // [rsp+1B0h] [rbp-E10h] BYREF
  __int64 v198[2]; // [rsp+1E0h] [rbp-DE0h] BYREF
  _QWORD v199[2]; // [rsp+1F0h] [rbp-DD0h] BYREF
  _QWORD *v200; // [rsp+200h] [rbp-DC0h]
  _QWORD v201[6]; // [rsp+210h] [rbp-DB0h] BYREF
  __int64 v202[5]; // [rsp+240h] [rbp-D80h] BYREF
  char v203; // [rsp+268h] [rbp-D58h]
  __int64 v204; // [rsp+270h] [rbp-D50h]
  __int64 v205; // [rsp+278h] [rbp-D48h]
  __int64 v206; // [rsp+280h] [rbp-D40h]
  int v207; // [rsp+288h] [rbp-D38h]
  __int64 v208; // [rsp+290h] [rbp-D30h]
  __int64 v209; // [rsp+298h] [rbp-D28h]
  __int64 v210; // [rsp+2A0h] [rbp-D20h]
  int v211; // [rsp+2A8h] [rbp-D18h]
  _QWORD v212[6]; // [rsp+2B0h] [rbp-D10h] BYREF
  _BYTE *v213; // [rsp+2E0h] [rbp-CE0h]
  __int64 v214; // [rsp+2E8h] [rbp-CD8h]
  _BYTE v215[32]; // [rsp+2F0h] [rbp-CD0h] BYREF
  __int64 v216; // [rsp+310h] [rbp-CB0h]
  __int64 v217; // [rsp+318h] [rbp-CA8h]
  __int64 v218; // [rsp+320h] [rbp-CA0h]
  _BYTE v219[112]; // [rsp+330h] [rbp-C90h] BYREF
  __int64 v220; // [rsp+3A0h] [rbp-C20h]
  __m128i v221; // [rsp+4A0h] [rbp-B20h] BYREF
  __int64 v222; // [rsp+4B0h] [rbp-B10h] BYREF
  __m128i v223; // [rsp+4B8h] [rbp-B08h]
  __int64 v224; // [rsp+4C8h] [rbp-AF8h]
  __int64 v225; // [rsp+4D0h] [rbp-AF0h]
  __m128i v226; // [rsp+4D8h] [rbp-AE8h]
  __int64 v227; // [rsp+4E8h] [rbp-AD8h]
  char v228; // [rsp+4F0h] [rbp-AD0h]
  _BYTE *v229; // [rsp+4F8h] [rbp-AC8h] BYREF
  __int64 v230; // [rsp+500h] [rbp-AC0h]
  _BYTE v231[352]; // [rsp+508h] [rbp-AB8h] BYREF
  char v232; // [rsp+668h] [rbp-958h]
  int v233; // [rsp+66Ch] [rbp-954h]
  __int64 v234; // [rsp+670h] [rbp-950h]
  __int64 *v235; // [rsp+680h] [rbp-940h] BYREF
  __int64 v236; // [rsp+688h] [rbp-938h]
  _QWORD v237[3]; // [rsp+690h] [rbp-930h] BYREF
  char v238; // [rsp+6A8h] [rbp-918h]
  __int64 v239; // [rsp+6B0h] [rbp-910h]
  __int64 v240; // [rsp+6B8h] [rbp-908h]
  __int64 v241; // [rsp+6C0h] [rbp-900h]
  int v242; // [rsp+6C8h] [rbp-8F8h]
  __int64 v243; // [rsp+6D0h] [rbp-8F0h]
  _QWORD v244[2]; // [rsp+6D8h] [rbp-8E8h] BYREF
  int v245; // [rsp+6E8h] [rbp-8D8h]
  __int64 v246; // [rsp+850h] [rbp-770h]
  __m128i v247; // [rsp+860h] [rbp-760h] BYREF
  _BYTE *v248; // [rsp+870h] [rbp-750h]
  __int64 v249; // [rsp+878h] [rbp-748h]
  _QWORD *v250; // [rsp+880h] [rbp-740h]
  void **v251; // [rsp+888h] [rbp-738h]
  __int64 *v252; // [rsp+890h] [rbp-730h]
  __int64 v253; // [rsp+898h] [rbp-728h]
  __int64 v254; // [rsp+8A0h] [rbp-720h] BYREF
  __int64 v255; // [rsp+8A8h] [rbp-718h]
  __int64 v256; // [rsp+8B0h] [rbp-710h]
  int v257; // [rsp+8B8h] [rbp-708h] BYREF
  unsigned int v258; // [rsp+8BCh] [rbp-704h]
  __int64 v259; // [rsp+8C0h] [rbp-700h]
  __int64 v260; // [rsp+8C8h] [rbp-6F8h]
  __int64 v261; // [rsp+8D0h] [rbp-6F0h]
  __int64 v262; // [rsp+8D8h] [rbp-6E8h]
  __int64 v263; // [rsp+8E0h] [rbp-6E0h]
  int v264; // [rsp+8E8h] [rbp-6D8h]
  __int64 v265; // [rsp+8F0h] [rbp-6D0h]
  __int64 v266; // [rsp+8F8h] [rbp-6C8h]
  _BYTE *v267; // [rsp+938h] [rbp-688h]
  __int64 v268; // [rsp+940h] [rbp-680h]
  _BYTE v269[32]; // [rsp+948h] [rbp-678h] BYREF
  __int64 v270; // [rsp+968h] [rbp-658h]
  __int64 v271; // [rsp+970h] [rbp-650h]
  unsigned int v272; // [rsp+978h] [rbp-648h]
  int v273; // [rsp+97Ch] [rbp-644h]
  int v274; // [rsp+988h] [rbp-638h] BYREF
  __int64 v275; // [rsp+990h] [rbp-630h]
  int *v276; // [rsp+998h] [rbp-628h]
  int *v277; // [rsp+9A0h] [rbp-620h]
  __int64 v278; // [rsp+9A8h] [rbp-618h]
  int v279; // [rsp+9B8h] [rbp-608h] BYREF
  __int64 v280; // [rsp+9C0h] [rbp-600h]
  int *v281; // [rsp+9C8h] [rbp-5F8h]
  int *v282; // [rsp+9D0h] [rbp-5F0h]
  __int64 v283; // [rsp+9D8h] [rbp-5E8h]
  _BYTE *v284; // [rsp+9E0h] [rbp-5E0h]
  __int64 v285; // [rsp+9E8h] [rbp-5D8h]
  _BYTE v286[32]; // [rsp+9F0h] [rbp-5D0h] BYREF
  __int64 v287; // [rsp+A10h] [rbp-5B0h]
  __int64 v288; // [rsp+A18h] [rbp-5A8h]
  _QWORD *v289; // [rsp+A20h] [rbp-5A0h]
  void **v290; // [rsp+A28h] [rbp-598h]
  __int64 v291; // [rsp+A30h] [rbp-590h]
  __int64 v292; // [rsp+A38h] [rbp-588h]
  __int64 v293; // [rsp+A40h] [rbp-580h]
  __int64 v294; // [rsp+A48h] [rbp-578h]
  int v295; // [rsp+A50h] [rbp-570h]
  _QWORD v296[6]; // [rsp+A60h] [rbp-560h] BYREF
  __int64 v297; // [rsp+A90h] [rbp-530h]
  __int64 v298; // [rsp+A98h] [rbp-528h]
  __int64 v299; // [rsp+AA0h] [rbp-520h]
  __int64 v300; // [rsp+AA8h] [rbp-518h]
  _QWORD *v301; // [rsp+AB0h] [rbp-510h]
  __int64 v302; // [rsp+AB8h] [rbp-508h]
  unsigned int v303; // [rsp+AC0h] [rbp-500h]
  __int64 v304; // [rsp+AC8h] [rbp-4F8h]
  __int64 v305; // [rsp+AD0h] [rbp-4F0h]
  __int64 v306; // [rsp+AD8h] [rbp-4E8h]
  int v307; // [rsp+AE0h] [rbp-4E0h]
  __int64 v308; // [rsp+AE8h] [rbp-4D8h]
  __int64 v309; // [rsp+AF0h] [rbp-4D0h]
  __int64 v310; // [rsp+AF8h] [rbp-4C8h]
  __int64 v311; // [rsp+B00h] [rbp-4C0h]
  _BYTE *v312; // [rsp+B08h] [rbp-4B8h]
  _BYTE *v313; // [rsp+B10h] [rbp-4B0h]
  __int64 v314; // [rsp+B18h] [rbp-4A8h]
  int v315; // [rsp+B20h] [rbp-4A0h]
  _BYTE v316[32]; // [rsp+B28h] [rbp-498h] BYREF
  __int64 v317; // [rsp+B48h] [rbp-478h]
  _BYTE *v318; // [rsp+B50h] [rbp-470h]
  _BYTE *v319; // [rsp+B58h] [rbp-468h]
  __int64 v320; // [rsp+B60h] [rbp-460h]
  int v321; // [rsp+B68h] [rbp-458h]
  _BYTE v322[64]; // [rsp+B70h] [rbp-450h] BYREF
  __int64 v323; // [rsp+BB0h] [rbp-410h]
  __int64 v324; // [rsp+BB8h] [rbp-408h]
  __int64 v325; // [rsp+BC0h] [rbp-400h]
  int v326; // [rsp+BC8h] [rbp-3F8h]
  __int64 v327; // [rsp+BD0h] [rbp-3F0h]
  __int64 v328; // [rsp+BD8h] [rbp-3E8h]
  _BYTE *v329; // [rsp+BE0h] [rbp-3E0h]
  _BYTE *v330; // [rsp+BE8h] [rbp-3D8h]
  __int64 v331; // [rsp+BF0h] [rbp-3D0h]
  int v332; // [rsp+BF8h] [rbp-3C8h]
  _BYTE v333[40]; // [rsp+C00h] [rbp-3C0h] BYREF
  int *v334; // [rsp+C28h] [rbp-398h]
  char *v335; // [rsp+C30h] [rbp-390h]
  __int64 v336; // [rsp+C38h] [rbp-388h]
  __int64 v337; // [rsp+C40h] [rbp-380h]
  __int64 v338; // [rsp+C48h] [rbp-378h]
  _BYTE *v339; // [rsp+C50h] [rbp-370h]
  _BYTE *v340; // [rsp+C58h] [rbp-368h]
  __int64 v341; // [rsp+C60h] [rbp-360h]
  int v342; // [rsp+C68h] [rbp-358h]
  _BYTE v343[64]; // [rsp+C70h] [rbp-350h] BYREF
  void *v344; // [rsp+CB0h] [rbp-310h] BYREF
  __int64 v345; // [rsp+CB8h] [rbp-308h]
  __int64 v346; // [rsp+CC0h] [rbp-300h]
  __int64 v347; // [rsp+CC8h] [rbp-2F8h]
  __int64 v348; // [rsp+CD0h] [rbp-2F0h]
  __int64 v349; // [rsp+CD8h] [rbp-2E8h]
  __int64 v350; // [rsp+CE0h] [rbp-2E0h]
  __int64 v351; // [rsp+CE8h] [rbp-2D8h]
  __int64 v352; // [rsp+CF0h] [rbp-2D0h]
  _BYTE *v353; // [rsp+CF8h] [rbp-2C8h]
  _BYTE *v354; // [rsp+D00h] [rbp-2C0h]
  __int64 v355; // [rsp+D08h] [rbp-2B8h] BYREF
  int v356; // [rsp+D10h] [rbp-2B0h]
  _BYTE v357[32]; // [rsp+D18h] [rbp-2A8h] BYREF
  __int64 v358; // [rsp+D38h] [rbp-288h]
  __int64 v359; // [rsp+D40h] [rbp-280h]
  __int64 v360; // [rsp+D48h] [rbp-278h]
  int v361; // [rsp+D50h] [rbp-270h]
  __int64 v362; // [rsp+D58h] [rbp-268h]
  __int64 v363; // [rsp+D60h] [rbp-260h]
  __int64 v364; // [rsp+D68h] [rbp-258h]
  int v365; // [rsp+D70h] [rbp-250h]
  __int64 v366; // [rsp+D78h] [rbp-248h]
  __int64 v367; // [rsp+D80h] [rbp-240h]
  __int64 v368; // [rsp+D88h] [rbp-238h]
  int v369; // [rsp+D90h] [rbp-230h]
  __int64 v370; // [rsp+D98h] [rbp-228h]
  __int64 v371; // [rsp+DA0h] [rbp-220h]
  __int64 v372; // [rsp+DA8h] [rbp-218h]
  int v373; // [rsp+DB0h] [rbp-210h]
  __int64 v374; // [rsp+DB8h] [rbp-208h]
  __int64 v375; // [rsp+DC0h] [rbp-200h]
  __int64 v376; // [rsp+DC8h] [rbp-1F8h]
  int v377; // [rsp+DD0h] [rbp-1F0h]
  __int64 v378; // [rsp+DD8h] [rbp-1E8h]
  _BYTE *v379; // [rsp+DE0h] [rbp-1E0h]
  __int64 v380; // [rsp+DE8h] [rbp-1D8h]
  _QWORD *v381; // [rsp+DF0h] [rbp-1D0h]
  __int64 v382; // [rsp+DF8h] [rbp-1C8h]
  _BYTE *v383; // [rsp+E00h] [rbp-1C0h]
  __int64 v384; // [rsp+E08h] [rbp-1B8h]
  __int64 v385; // [rsp+E10h] [rbp-1B0h]
  __int64 v386; // [rsp+E18h] [rbp-1A8h]
  __int64 v387; // [rsp+E20h] [rbp-1A0h]
  char *v388; // [rsp+E28h] [rbp-198h]
  __int64 *v389; // [rsp+E30h] [rbp-190h]
  __int64 v390; // [rsp+E38h] [rbp-188h]
  _BYTE *v391; // [rsp+E40h] [rbp-180h]
  _BYTE *v392; // [rsp+E48h] [rbp-178h]
  __int64 v393; // [rsp+E50h] [rbp-170h]
  int v394; // [rsp+E58h] [rbp-168h]
  _BYTE v395[128]; // [rsp+E60h] [rbp-160h] BYREF
  __int64 v396; // [rsp+EE0h] [rbp-E0h]
  _BYTE *v397; // [rsp+EE8h] [rbp-D8h]
  _BYTE *v398; // [rsp+EF0h] [rbp-D0h]
  __int64 v399; // [rsp+EF8h] [rbp-C8h]
  int v400; // [rsp+F00h] [rbp-C0h]
  _BYTE v401[184]; // [rsp+F08h] [rbp-B8h] BYREF

  v14 = a2;
  sub_1BF1BF0(v188, a2, *a1, *((_QWORD *)a1 + 11), a13);
  v15 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 56LL);
  v16 = sub_1BF5810(v188, v15, a2, a1[1]);
  if ( !(_BYTE)v16 )
    return v16;
  sub_14586F0((__int64)v219, *((_QWORD *)a1 + 1), a2);
  v18 = *((_QWORD *)a1 + 11);
  v19 = *((_QWORD *)a1 + 4);
  v296[2] = v219;
  v20 = *((_QWORD *)a1 + 2);
  v21 = *((_QWORD *)a1 + 10);
  v296[0] = v14;
  v22 = *((_QWORD *)a1 + 6);
  v176 = v18;
  v23 = *((_QWORD *)a1 + 9);
  v24 = *((_QWORD *)a1 + 7);
  v296[4] = v19;
  v298 = v18;
  v296[1] = v20;
  v296[3] = v22;
  v296[5] = v21;
  v174 = 0;
  v175 = 0;
  v297 = 0;
  v299 = 0;
  v300 = 0;
  v301 = 0;
  v302 = 0;
  v303 = 0;
  v304 = 0;
  v305 = 0;
  v306 = 0;
  v307 = 0;
  v308 = 0;
  v309 = 0;
  v310 = 0;
  v311 = 0;
  v312 = v316;
  v313 = v316;
  v318 = v322;
  v319 = v322;
  v329 = v333;
  v330 = v333;
  v334 = &v174;
  v314 = 4;
  v315 = 0;
  v317 = 0;
  v320 = 8;
  v321 = 0;
  v323 = 0;
  v324 = 0;
  v325 = 0;
  v326 = 0;
  v327 = 0;
  v328 = 0;
  v331 = 4;
  v332 = 0;
  v333[32] = 0;
  v335 = v188;
  v336 = v24;
  v337 = v23;
  v338 = 0;
  v339 = v343;
  v340 = v343;
  v341 = 8;
  v342 = 0;
  v16 = sub_1BF7B60(v296, (unsigned __int8)byte_4FB8040);
  if ( !(_BYTE)v16 )
  {
    sub_1B95750(v14, (__int64)v188, *((_QWORD **)a1 + 11));
    goto LABEL_15;
  }
  v164 = 0;
  if ( v191 != 1 )
  {
    v164 = sub_1560180(v15 + 112, 34);
    if ( !v164 )
      v164 = sub_1560180(v15 + 112, 17);
  }
  if ( *(_QWORD *)(v14 + 8) != *(_QWORD *)(v14 + 16) )
  {
    v25 = *((_QWORD *)a1 + 2);
    v26 = (_BYTE *)*((_QWORD *)a1 + 6);
    v27 = *((_QWORD *)a1 + 3);
    v28 = *((_QWORD *)a1 + 11);
    v29 = *(_QWORD *)(**(_QWORD **)(v14 + 32) + 56LL);
    v30 = *((_QWORD *)a1 + 9);
    v237[0] = *((_QWORD *)a1 + 4);
    v31 = *((_QWORD *)a1 + 7);
    v235 = (__int64 *)v219;
    v236 = v14;
    v237[1] = v25;
    v237[2] = v297;
    v238 = 0;
    v239 = 0;
    v240 = 0;
    v241 = 0;
    v242 = 0;
    v243 = 0;
    v244[0] = 0;
    v244[1] = 0;
    v245 = 0;
    v344 = 0;
    v345 = 0;
    v346 = 0;
    v347 = 0;
    v348 = 0;
    v349 = 0;
    v350 = 0;
    v379 = v219;
    v353 = v357;
    v354 = v357;
    v378 = v14;
    v380 = v25;
    v351 = 0;
    v352 = 0;
    v355 = 4;
    v356 = 0;
    v358 = 0;
    v359 = 0;
    v360 = 0;
    v361 = 0;
    v362 = 0;
    v363 = 0;
    v364 = 0;
    v365 = 0;
    v366 = 0;
    v367 = 0;
    v368 = 0;
    v369 = 0;
    v370 = 0;
    v371 = 0;
    v372 = 0;
    v373 = 0;
    v374 = 0;
    v375 = 0;
    v376 = 0;
    v377 = 0;
    v381 = v296;
    v382 = v27;
    v32 = v189;
    v391 = v395;
    v392 = v395;
    v397 = v401;
    v398 = v401;
    v33 = 0;
    v247.m128i_i64[0] = v14;
    v247.m128i_i64[1] = v25;
    v383 = v26;
    v384 = v31;
    v385 = v30;
    v386 = v28;
    v387 = v29;
    v388 = v188;
    v389 = (__int64 *)&v235;
    v390 = 0;
    v393 = 16;
    v394 = 0;
    v396 = 0;
    v399 = 16;
    v400 = 0;
    v248 = v26;
    v249 = v27;
    v250 = v296;
    v251 = &v344;
    v252 = &v254;
    v253 = 0x400000000LL;
    v259 = 0;
    v260 = 0;
    v261 = 0;
    if ( v191 != 1 )
    {
      v33 = 1;
      if ( !(unsigned __int8)sub_1560180(v29 + 112, 34) )
        v33 = (unsigned __int8)sub_1560180(v29 + 112, 17);
    }
    sub_1B98EF0((__int64)&v247, v33, v32);
    v34 = v252;
    v35 = &v252[(unsigned int)v253];
    if ( v252 != v35 )
    {
      do
      {
        v36 = *--v35;
        if ( v36 )
        {
          sub_1B949D0(v36);
          j_j___libc_free_0(v36, 472);
        }
      }
      while ( v34 != v35 );
      v35 = v252;
    }
    if ( v35 != &v254 )
      _libc_free((unsigned __int64)v35);
    sub_1B93BB0((__int64)&v344);
    v16 = 0;
    sub_1B901A0((__int64)&v235);
    goto LABEL_15;
  }
  v44 = sub_1481F60(*((_QWORD **)a1 + 1), v14, a3, a4);
  if ( !*(_WORD *)(v44 + 24) )
  {
    v45 = *(_QWORD *)(v44 + 32);
    if ( *(_DWORD *)(v45 + 32) <= 0x40u )
    {
      v46 = *(_QWORD *)(v45 + 24);
      if ( v46 <= 0xFFFFFFFE )
      {
LABEL_47:
        v47 = v46 + 1;
        goto LABEL_48;
      }
    }
    else
    {
      v159 = *(_QWORD *)(v44 + 32);
      v172 = *(_DWORD *)(v45 + 32);
      if ( v172 - (unsigned int)sub_16A57B0(v45 + 24) <= 0x40 )
      {
        v46 = **(_QWORD **)(v159 + 24);
        if ( v46 <= 0xFFFFFFFE )
          goto LABEL_47;
      }
    }
  }
  if ( byte_4FB8580 && (sub_1B18810((__int64)&v344, v14), v47 = (unsigned int)v344, BYTE4(v344))
    || (v47 = sub_1474290(*((_QWORD *)a1 + 1), v14)) != 0 )
  {
LABEL_48:
    if ( dword_4FB8E40 > v47 )
    {
      v48 = v164;
      if ( v191 != 1 )
        v48 = v16;
      v164 = v48;
    }
  }
  if ( (unsigned __int8)sub_1560180(v15 + 112, 25) )
  {
    v97 = (_QWORD *)*((_QWORD *)a1 + 11);
    v98 = sub_1BF18B0(v188, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64);
    sub_1BF1750(&v344, v98, "NoImplicitFloat", 15, v14, 0);
    v99 = "loop not vectorized due to NoImplicitFloat attribute";
    v100 = 52;
  }
  else
  {
    if ( v191 == 1 || !v193 || !(unsigned __int8)sub_14A2F60(*((_QWORD *)a1 + 3)) )
    {
      v49 = sub_14A2F30(*((_QWORD *)a1 + 3));
      v50 = *((_QWORD *)a1 + 4);
      v160 = v49;
      v51 = *((_QWORD *)a1 + 2);
      v202[0] = (__int64)v219;
      v202[3] = v51;
      v202[1] = v14;
      v202[2] = v50;
      v202[4] = v297;
      v203 = 0;
      v204 = 0;
      v205 = 0;
      v206 = 0;
      v207 = 0;
      v208 = 0;
      v209 = 0;
      v210 = 0;
      v211 = 0;
      v52 = sub_1B907A0(&dword_4FB8BE8);
      v53 = v160;
      if ( v52 > 0 )
        v53 = byte_4FB8C80;
      if ( v53 )
        sub_1BB4E00(v202);
      v54 = *((_QWORD *)a1 + 2);
      v55 = *((_QWORD *)a1 + 3);
      v344 = 0;
      v56 = *((_QWORD *)a1 + 11);
      v57 = *((_QWORD *)a1 + 7);
      v353 = v357;
      v58 = *((_QWORD *)a1 + 9);
      v59 = (_BYTE *)*((_QWORD *)a1 + 6);
      v354 = v357;
      v345 = 0;
      v346 = 0;
      v347 = 0;
      v348 = 0;
      v349 = 0;
      v350 = 0;
      v351 = 0;
      v352 = 0;
      v355 = 4;
      v356 = 0;
      v358 = 0;
      v359 = 0;
      v360 = 0;
      v361 = 0;
      v362 = 0;
      v363 = 0;
      v364 = 0;
      v365 = 0;
      v366 = 0;
      v367 = 0;
      v368 = 0;
      v369 = 0;
      v370 = 0;
      v371 = 0;
      v383 = v59;
      v379 = v219;
      v385 = v58;
      v381 = v296;
      v391 = v395;
      v392 = v395;
      v380 = v54;
      v382 = v55;
      v384 = v57;
      v386 = v56;
      v389 = v202;
      v397 = v401;
      v398 = v401;
      v372 = 0;
      v373 = 0;
      v374 = 0;
      v375 = 0;
      v376 = 0;
      v377 = 0;
      v378 = v14;
      v387 = v15;
      v388 = v188;
      v390 = 0;
      v393 = 16;
      v394 = 0;
      v396 = 0;
      v399 = 16;
      v400 = 0;
      sub_1B92F40((__int64)&v344);
      v60 = *((_QWORD *)a1 + 3);
      v61 = *((_QWORD *)a1 + 6);
      v212[0] = v14;
      v62 = *((_QWORD *)a1 + 2);
      v216 = 0;
      v212[2] = v61;
      v212[3] = v60;
      v213 = v215;
      v212[1] = v62;
      v212[4] = v296;
      v214 = 0x400000000LL;
      v212[5] = &v344;
      v217 = 0;
      v218 = 0;
      v157 = sub_1BB44E0((__int64)v212, v164, v189, a3, a4);
      v154 = sub_1BB46B0((__int64)&v344, v164, v157, HIDWORD(v157), a3, a4);
      v178 = 0;
      v165 = v190;
      v180 = &v182;
      v179 = 0;
      v181 = 0;
      v182 = 0;
      v183 = 0;
      v184 = 0;
      v185 = &v187;
      v186 = 0;
      v187 = 0;
      v63 = sub_1BF4A80(&v174, v15, v14, v188);
      if ( v63 )
      {
        v16 = 0;
        sub_1B95750(v14, (__int64)v188, *((_QWORD **)a1 + 11));
        sub_2240A30(&v185);
        goto LABEL_87;
      }
      v64 = v16;
      if ( (_DWORD)v157 == 1 )
      {
        v64 = 0;
        v178 = "VectorizationNotBeneficial";
        v179 = 26;
        sub_2241130(&v180, 0, v181, "the cost-model indicates that vectorization is not beneficial", 61);
        v63 = 0;
      }
      if ( v154 == 1 )
      {
        if ( v165 > 1 )
          goto LABEL_100;
        v183 = "InterleavingNotBeneficial";
        v162 = v63;
        v184 = 25;
        sub_2241130(&v185, 0, v186, "the cost-model indicates that interleaving is not beneficial", 60);
        v102 = v162;
        if ( v165 == 1 )
        {
          v184 = 36;
          v183 = "InterleavingNotBeneficialAndDisabled";
          if ( 0x3FFFFFFFFFFFFFFFLL - v186 <= 0x3A )
            sub_4262D8((__int64)"basic_string::append");
          sub_2241490(&v185, " and is explicitly disabled or interleave count is set to 1", 59);
          v102 = v162;
        }
        v168 = v102;
        v103 = sub_1BF18B0(v188, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64);
        v66 = v168;
        v161 = v103;
        if ( v64 )
        {
          v149 = v64;
          goto LABEL_67;
        }
      }
      else
      {
        v149 = v154 > 1 && v165 == 1;
        if ( !v149 )
        {
          if ( !v165 )
          {
            v65 = sub_1BF18B0(v188, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64);
            v66 = v16;
            v161 = v65;
LABEL_67:
            if ( v64 )
              goto LABEL_133;
            if ( v66 )
              goto LABEL_69;
            if ( v64 )
            {
LABEL_133:
              if ( v149 )
                goto LABEL_134;
            }
            v165 = v154;
            sub_1B95050((__int64)v212, v157, v154);
LABEL_102:
            if ( !v64 )
            {
LABEL_72:
              v69 = *((_QWORD *)a1 + 11);
              v70 = *((_QWORD *)a1 + 3);
              v247.m128i_i64[1] = v14;
              v71 = *((_QWORD *)a1 + 2);
              v256 = 0;
              v72 = *((_QWORD *)a1 + 9);
              v73 = (__int64 *)*((_QWORD *)a1 + 6);
              v255 = v69;
              v74 = (_QWORD *)*((_QWORD *)a1 + 4);
              v249 = v71;
              v247.m128i_i64[0] = (__int64)&unk_49F6E20;
              v248 = v219;
              v253 = v70;
              v252 = v73;
              v254 = v72;
              v250 = v74;
              v257 = 1;
              v258 = v165;
              v75 = sub_15E0530(*(_QWORD *)(v220 + 24));
              v259 = 0;
              v262 = v75;
              v267 = v269;
              v268 = 0x400000000LL;
              v285 = 0x400000000LL;
              v276 = &v274;
              v277 = &v274;
              v289 = v296;
              v281 = &v279;
              v282 = &v279;
              v284 = v286;
              v290 = &v344;
              v261 = 0;
              v263 = 0;
              v264 = 0;
              v265 = 0;
              v266 = 0;
              v260 = 0;
              v270 = 0;
              v271 = 0;
              v272 = v165;
              v273 = 1;
              v274 = 0;
              v275 = 0;
              v278 = 0;
              v279 = 0;
              v280 = 0;
              v283 = 0;
              v287 = 0;
              v288 = 0;
              LOBYTE(v291) = 0;
              v76 = *((_QWORD *)a1 + 4);
              v292 = 0;
              v293 = 0;
              v247.m128i_i64[0] = (__int64)&unk_49F6E58;
              v294 = 0;
              v295 = 0;
              sub_1BAFBA0((__int64)v212, (__int64)&v247, v76, a3, a4, a5, a6, v77, v78, a9, a10);
              v170 = (__int64 *)*((_QWORD *)a1 + 11);
              v79 = sub_15E0530(*v170);
              if ( sub_1602790(v79)
                || (v128 = sub_15E0530(*v170),
                    v129 = sub_16033E0(v128),
                    (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v129 + 48LL))(v129)) )
              {
                v158 = **(_QWORD **)(v14 + 32);
                sub_13FD840(v177, v14);
                sub_15C9090((__int64)&v194, v177);
                sub_15CA330((__int64)&v235, (__int64)"loop-vectorize", (__int64)"Interleaved", 11, &v194, v158);
                sub_15CAB20((__int64)&v235, "interleaved loop (interleaved count: ", 0x25u);
                sub_15C9C50((__int64)v198, "InterleaveCount", 15, v165);
                v166 = sub_17C2270((__int64)&v235, (__int64)v198);
                sub_15CAB20(v166, ")", 1u);
                v80 = v166;
                v221.m128i_i32[2] = *(_DWORD *)(v166 + 8);
                v221.m128i_i8[12] = *(_BYTE *)(v166 + 12);
                v222 = *(_QWORD *)(v166 + 16);
                v223 = _mm_loadu_si128((const __m128i *)(v166 + 24));
                v224 = *(_QWORD *)(v166 + 40);
                v221.m128i_i64[0] = (__int64)&unk_49ECF68;
                v225 = *(_QWORD *)(v166 + 48);
                v226 = _mm_loadu_si128((const __m128i *)(v166 + 56));
                v228 = *(_BYTE *)(v166 + 80);
                if ( v228 )
                  v227 = *(_QWORD *)(v166 + 72);
                v229 = v231;
                v230 = 0x400000000LL;
                if ( *(_DWORD *)(v166 + 96) )
                {
                  sub_1B99110((__int64)&v229, v166 + 88);
                  v80 = v166;
                }
                v232 = *(_BYTE *)(v80 + 456);
                v233 = *(_DWORD *)(v80 + 460);
                v234 = *(_QWORD *)(v80 + 464);
                v221.m128i_i64[0] = (__int64)&unk_49ECF98;
                if ( v200 != v201 )
                  j_j___libc_free_0(v200, v201[0] + 1LL);
                if ( (_QWORD *)v198[0] != v199 )
                  j_j___libc_free_0(v198[0], v199[0] + 1LL);
                v235 = (__int64 *)&unk_49ECF68;
                sub_1897B80((__int64)v244);
                if ( v177[0].m128i_i64[0] )
                  sub_161E7C0((__int64)v177, v177[0].m128i_i64[0]);
                sub_143AA50(v170, (__int64)&v221);
                v221.m128i_i64[0] = (__int64)&unk_49ECF68;
                sub_1897B80((__int64)&v229);
              }
              v247.m128i_i64[0] = (__int64)&unk_49F6E58;
              sub_1B90880((__int64)&v247);
LABEL_85:
              v192.m128i_i32[2] = 1;
              v247 = _mm_loadu_si128(&v192);
              sub_1BF1E00(v188, &v247, 1);
LABEL_86:
              sub_2240A30(&v185);
LABEL_87:
              sub_2240A30(&v180);
              v81 = v213;
              v82 = &v213[8 * (unsigned int)v214];
              if ( v213 != v82 )
              {
                do
                {
                  v83 = *((_QWORD *)v82 - 1);
                  v82 -= 8;
                  if ( v83 )
                  {
                    sub_1B949D0(v83);
                    j_j___libc_free_0(v83, 472);
                  }
                }
                while ( v81 != v82 );
                v82 = v213;
              }
              if ( v82 != v215 )
                _libc_free((unsigned __int64)v82);
              sub_1B93BB0((__int64)&v344);
              sub_1B901A0((__int64)v202);
              goto LABEL_15;
            }
LABEL_103:
            v84 = *((_QWORD *)a1 + 11);
            v85 = *((_QWORD *)a1 + 9);
            v247.m128i_i64[1] = v14;
            v86 = *((_QWORD *)a1 + 3);
            v256 = 0;
            v87 = (__int64 *)*((_QWORD *)a1 + 6);
            v88 = (_QWORD *)*((_QWORD *)a1 + 4);
            v255 = v84;
            v89 = *((_QWORD *)a1 + 2);
            v253 = v86;
            v247.m128i_i64[0] = (__int64)&unk_49F6E20;
            v257 = v157;
            v248 = v219;
            v249 = v89;
            v252 = v87;
            v254 = v85;
            v250 = v88;
            v258 = v165;
            v90 = sub_15E0530(*(_QWORD *)(v220 + 24));
            v259 = 0;
            v262 = v90;
            v267 = v269;
            v268 = 0x400000000LL;
            v285 = 0x400000000LL;
            v276 = &v274;
            v277 = &v274;
            v289 = v296;
            v281 = &v279;
            v282 = &v279;
            v273 = v157;
            v284 = v286;
            v261 = 0;
            v263 = 0;
            v264 = 0;
            v265 = 0;
            v266 = 0;
            v260 = 0;
            v270 = 0;
            v271 = 0;
            v272 = v165;
            v274 = 0;
            v275 = 0;
            v278 = 0;
            v279 = 0;
            v280 = 0;
            v283 = 0;
            v287 = 0;
            v288 = 0;
            v290 = &v344;
            v91 = *((_QWORD *)a1 + 4);
            LOBYTE(v291) = 0;
            v292 = 0;
            v293 = 0;
            v294 = 0;
            v295 = 0;
            sub_1BAFBA0((__int64)v212, (__int64)&v247, v91, a3, a4, a5, a6, v92, v93, a9, a10);
            if ( !(_BYTE)v291 )
            {
              v237[0] = 0;
              v235 = v237;
              v236 = 0x400000001LL;
              v111 = sub_13FD000(v14);
              v113 = v111;
              if ( !v111 )
                goto LABEL_166;
              v114 = *(unsigned int *)(v111 + 8);
              if ( (unsigned int)v114 <= 1 )
                goto LABEL_166;
              v156 = v16;
              v115 = v113;
              v151 = v14;
              v116 = 0;
              v117 = 1;
              v118 = v114;
              while ( 1 )
              {
                v119 = *(_QWORD *)(v115 + 8 * (v117 - v114));
                if ( (unsigned __int8)(*(_BYTE *)v119 - 4) <= 0x1Eu )
                {
                  v116 = 0;
                  v120 = *(_BYTE **)(v119 - 8LL * *(unsigned int *)(v119 + 8));
                  if ( !*v120 )
                  {
                    v121 = (_QWORD *)sub_161E970((__int64)v120);
                    if ( v122 > 0x17 )
                    {
                      v123 = *v121 ^ 0x6F6F6C2E6D766C6CLL | v121[1] ^ 0x6C6C6F726E752E70LL
                          || v121[2] != 0x656C62617369642ELL;
                      v116 = !v123;
                    }
                    v119 = *(_QWORD *)(v115 + 8 * (v117 - *(unsigned int *)(v115 + 8)));
                  }
                }
                v124 = (unsigned int)v236;
                if ( (unsigned int)v236 >= HIDWORD(v236) )
                {
                  v148 = v119;
                  sub_16CD150((__int64)&v235, v237, 0, 8, v112, v113);
                  v124 = (unsigned int)v236;
                  v119 = v148;
                }
                ++v117;
                v235[v124] = v119;
                LODWORD(v236) = v236 + 1;
                if ( v117 == v118 )
                  break;
                v114 = *(unsigned int *)(v115 + 8);
              }
              v125 = v116;
              v16 = v156;
              v14 = v151;
              if ( !v125 )
              {
LABEL_166:
                v131 = (__int64 *)sub_157E9C0(**(_QWORD **)(v14 + 32));
                v221.m128i_i64[0] = (__int64)&v222;
                v221.m128i_i64[1] = 0x100000000LL;
                v132 = sub_161FF10(v131, "llvm.loop.unroll.runtime.disable", 0x20u);
                v133 = v221.m128i_u32[2];
                if ( v221.m128i_i32[2] >= (unsigned __int32)v221.m128i_i32[3] )
                {
                  v153 = v132;
                  sub_16CD150((__int64)&v221, &v222, 0, 8, (int)&v221, v132);
                  v133 = v221.m128i_u32[2];
                  v132 = v153;
                }
                *(_QWORD *)(v221.m128i_i64[0] + 8 * v133) = v132;
                ++v221.m128i_i32[2];
                v134 = sub_1627350(v131, (__int64 *)v221.m128i_i64[0], (__int64 *)v221.m128i_u32[2], 0, 1);
                v136 = (unsigned int)v236;
                if ( (unsigned int)v236 >= HIDWORD(v236) )
                {
                  v152 = v134;
                  sub_16CD150((__int64)&v235, v237, 0, 8, v134, v135);
                  v136 = (unsigned int)v236;
                  v134 = v152;
                }
                v235[v136] = v134;
                LODWORD(v236) = v236 + 1;
                v137 = (unsigned __int8 *)sub_1627350(v131, v235, (__int64 *)(unsigned int)v236, 0, 1);
                sub_1630830(
                  (__int64)v137,
                  0,
                  v137,
                  *(double *)a3.m128i_i64,
                  *(double *)a4.m128i_i64,
                  *(double *)a5.m128i_i64,
                  a6,
                  v138,
                  v139,
                  a9,
                  a10);
                sub_13FCC30(v14, (__int64)v137);
                if ( (__int64 *)v221.m128i_i64[0] != &v222 )
                  _libc_free(v221.m128i_u64[0]);
              }
              if ( v235 != v237 )
                _libc_free((unsigned __int64)v235);
            }
            v171 = (__int64 *)*((_QWORD *)a1 + 11);
            v94 = sub_15E0530(*v171);
            if ( sub_1602790(v94)
              || (v126 = sub_15E0530(*v171),
                  v127 = sub_16033E0(v126),
                  (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v127 + 48LL))(v127)) )
            {
              v155 = **(_QWORD **)(v14 + 32);
              sub_13FD840(&v173, v14);
              sub_15C9090((__int64)v177, &v173);
              sub_15CA330((__int64)&v235, (__int64)"loop-vectorize", (__int64)"Vectorized", 10, v177, v155);
              sub_15CAB20((__int64)&v235, "vectorized loop (vectorization width: ", 0x26u);
              sub_15C9C50((__int64)v198, "VectorizationFactor", 19, v157);
              v95 = sub_17C2270((__int64)&v235, (__int64)v198);
              sub_15CAB20(v95, ", interleaved count: ", 0x15u);
              sub_15C9C50((__int64)&v194, "InterleaveCount", 15, v165);
              v167 = sub_17C2270(v95, (__int64)&v194);
              sub_15CAB20(v167, ")", 1u);
              v96 = v167;
              v221.m128i_i32[2] = *(_DWORD *)(v167 + 8);
              v221.m128i_i8[12] = *(_BYTE *)(v167 + 12);
              v222 = *(_QWORD *)(v167 + 16);
              v223 = _mm_loadu_si128((const __m128i *)(v167 + 24));
              v224 = *(_QWORD *)(v167 + 40);
              v221.m128i_i64[0] = (__int64)&unk_49ECF68;
              v225 = *(_QWORD *)(v167 + 48);
              v226 = _mm_loadu_si128((const __m128i *)(v167 + 56));
              v228 = *(_BYTE *)(v167 + 80);
              if ( v228 )
                v227 = *(_QWORD *)(v167 + 72);
              v229 = v231;
              v230 = 0x400000000LL;
              if ( *(_DWORD *)(v167 + 96) )
              {
                sub_1B99110((__int64)&v229, v167 + 88);
                v96 = v167;
              }
              v232 = *(_BYTE *)(v96 + 456);
              v233 = *(_DWORD *)(v96 + 460);
              v234 = *(_QWORD *)(v96 + 464);
              v221.m128i_i64[0] = (__int64)&unk_49ECF98;
              if ( v196 != &v197 )
                j_j___libc_free_0(v196, v197 + 1);
              if ( (__int64 *)v194.m128i_i64[0] != &v195 )
                j_j___libc_free_0(v194.m128i_i64[0], v195 + 1);
              if ( v200 != v201 )
                j_j___libc_free_0(v200, v201[0] + 1LL);
              if ( (_QWORD *)v198[0] != v199 )
                j_j___libc_free_0(v198[0], v199[0] + 1LL);
              v235 = (__int64 *)&unk_49ECF68;
              sub_1897B80((__int64)v244);
              if ( v173 )
                sub_161E7C0((__int64)&v173, v173);
              sub_143AA50(v171, (__int64)&v221);
              v221.m128i_i64[0] = (__int64)&unk_49ECF68;
              sub_1897B80((__int64)&v229);
            }
            sub_1B90880((__int64)&v247);
            goto LABEL_85;
          }
LABEL_100:
          v161 = sub_1BF18B0(v188, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64);
          if ( !v64 )
          {
            v154 = v165;
LABEL_69:
            v67 = (__int64 *)*((_QWORD *)a1 + 11);
            v68 = sub_15E0530(*v67);
            if ( sub_1602790(v68)
              || (v140 = sub_15E0530(*v67),
                  v141 = sub_16033E0(v140),
                  (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v141 + 48LL))(v141)) )
            {
              v150 = **(_QWORD **)(v14 + 32);
              sub_13FD840(v198, v14);
              sub_15C9090((__int64)&v221, v198);
              sub_15CA680((__int64)&v247, v161, (__int64)v178, v179, &v221, v150);
              sub_15CAB20((__int64)&v247, v180, v181);
              sub_18980B0((__int64)&v235, (__int64)&v247);
              v246 = v291;
              v235 = (__int64 *)&unk_49ECFF8;
              v247.m128i_i64[0] = (__int64)&unk_49ECF68;
              sub_1897B80((__int64)&v257);
              sub_17CD270(v198);
              sub_143AA50(v67, (__int64)&v235);
              v235 = (__int64 *)&unk_49ECF68;
              sub_1897B80((__int64)v244);
            }
            sub_1B95050((__int64)v212, v157, v154);
            v165 = v154;
            goto LABEL_72;
          }
          sub_1B95050((__int64)v212, v157, v165);
          goto LABEL_102;
        }
        v183 = "InterleavingBeneficialButDisabled";
        v184 = 33;
        sub_2241130(
          &v185,
          0,
          v186,
          "the cost-model indicates that interleaving is beneficial but is explicitly disabled or interleave count is set to 1",
          115);
        v154 = 1;
        v161 = sub_1BF18B0(v188, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64);
        if ( v64 )
        {
LABEL_134:
          v109 = (__int64 *)*((_QWORD *)a1 + 11);
          v110 = sub_15E0530(*v109);
          if ( sub_1602790(v110)
            || (v146 = sub_15E0530(*v109),
                v147 = sub_16033E0(v146),
                (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v147 + 48LL))(v147)) )
          {
            v163 = **(_QWORD **)(v14 + 32);
            sub_13FD840(v198, v14);
            sub_15C9090((__int64)&v221, v198);
            sub_15CA680((__int64)&v247, (__int64)"loop-vectorize", (__int64)v183, v184, &v221, v163);
            sub_15CAB20((__int64)&v247, v185, v186);
            sub_18980B0((__int64)&v235, (__int64)&v247);
            v246 = v291;
            v235 = (__int64 *)&unk_49ECFF8;
            v247.m128i_i64[0] = (__int64)&unk_49ECF68;
            sub_1897B80((__int64)&v257);
            sub_17CD270(v198);
            sub_143AA50(v109, (__int64)&v235);
            v235 = (__int64 *)&unk_49ECF68;
            sub_1897B80((__int64)v244);
          }
          sub_1B95050((__int64)v212, v157, v154);
          v165 = v154;
          goto LABEL_103;
        }
      }
      v104 = (__int64 *)*((_QWORD *)a1 + 11);
      v105 = sub_15E0530(*v104);
      if ( sub_1602790(v105)
        || (v144 = sub_15E0530(*v104),
            v145 = sub_16033E0(v144),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v145 + 48LL))(v145)) )
      {
        v169 = **(_QWORD **)(v14 + 32);
        sub_13FD840(v198, v14);
        sub_15C9090((__int64)&v221, v198);
        sub_15CA540((__int64)&v247, v161, (__int64)v178, v179, &v221, v169);
        sub_15CAB20((__int64)&v247, v180, v181);
        sub_18980B0((__int64)&v235, (__int64)&v247);
        v246 = v291;
        v235 = (__int64 *)&unk_49ECFC8;
        v247.m128i_i64[0] = (__int64)&unk_49ECF68;
        sub_1897B80((__int64)&v257);
        sub_17CD270(v198);
        sub_143AA50(v104, (__int64)&v235);
        v235 = (__int64 *)&unk_49ECF68;
        sub_1897B80((__int64)v244);
      }
      v106 = (__int64 *)*((_QWORD *)a1 + 11);
      v107 = sub_15E0530(*v106);
      if ( sub_1602790(v107)
        || (v142 = sub_15E0530(*v106),
            v143 = sub_16033E0(v142),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v143 + 48LL))(v143)) )
      {
        v108 = **(_QWORD **)(v14 + 32);
        sub_13FD840(v198, v14);
        sub_15C9090((__int64)&v221, v198);
        sub_15CA540((__int64)&v247, (__int64)"loop-vectorize", (__int64)v183, v184, &v221, v108);
        sub_15CAB20((__int64)&v247, v185, v186);
        sub_18980B0((__int64)&v235, (__int64)&v247);
        v246 = v291;
        v235 = (__int64 *)&unk_49ECFC8;
        v247.m128i_i64[0] = (__int64)&unk_49ECF68;
        sub_1897B80((__int64)&v257);
        sub_17CD270(v198);
        sub_143AA50(v106, (__int64)&v235);
        v235 = (__int64 *)&unk_49ECF68;
        sub_1897B80((__int64)v244);
      }
      v16 = 0;
      goto LABEL_86;
    }
    v97 = (_QWORD *)*((_QWORD *)a1 + 11);
    v130 = sub_1BF18B0(v188, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64);
    sub_1BF1750(&v344, v130, "UnsafeFP", 8, v14, 0);
    v99 = "loop not vectorized due to unsafe FP support.";
    v100 = 45;
  }
  sub_15CAB20((__int64)&v344, v99, v100);
  v101 = v97;
  v16 = 0;
  sub_143AA50(v101, (__int64)&v344);
  v344 = &unk_49ECF68;
  sub_1897B80((__int64)&v355);
  sub_1B95750(v14, (__int64)v188, *((_QWORD **)a1 + 11));
LABEL_15:
  if ( v340 != v339 )
    _libc_free((unsigned __int64)v340);
  if ( v330 != v329 )
    _libc_free((unsigned __int64)v330);
  j___libc_free_0(v324);
  if ( v319 != v318 )
    _libc_free((unsigned __int64)v319);
  if ( v313 != v312 )
    _libc_free((unsigned __int64)v313);
  v37 = v309;
  v38 = v308;
  if ( v309 != v308 )
  {
    do
    {
      v39 = *(_QWORD *)(v38 + 56);
      if ( v39 != v38 + 72 )
        _libc_free(v39);
      v40 = v38 + 8;
      v38 += 88;
      sub_1455FA0(v40);
    }
    while ( v37 != v38 );
    v38 = v308;
  }
  if ( v38 )
    j_j___libc_free_0(v38, v310 - v38);
  j___libc_free_0(v305);
  if ( v303 )
  {
    v41 = v301;
    v42 = &v301[22 * v303];
    do
    {
      if ( *v41 != -8 && *v41 != -16 )
      {
        v43 = v41[11];
        if ( v43 != v41[10] )
          _libc_free(v43);
        sub_1455FA0((__int64)(v41 + 1));
      }
      v41 += 22;
    }
    while ( v42 != v41 );
  }
  j___libc_free_0(v301);
  sub_1B93DB0((__int64)v219);
  return v16;
}
