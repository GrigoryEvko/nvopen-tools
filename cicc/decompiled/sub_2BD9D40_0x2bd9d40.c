// Function: sub_2BD9D40
// Address: 0x2bd9d40
//
__int64 __fastcall sub_2BD9D40(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12)
{
  __int64 v12; // r12
  __int64 v13; // rbx
  __int64 v14; // r14
  __int64 v15; // r13
  unsigned __int64 v16; // rdi
  __int64 v17; // r14
  __int64 v18; // r13
  unsigned __int64 v19; // rdi
  __int64 *v20; // r13
  unsigned int v21; // r15d
  unsigned __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // r13
  __int64 v26; // rdi
  __int64 v27; // r8
  __int64 v28; // r11
  __int64 v29; // r9
  __int64 v30; // r10
  char *v31; // rax
  __int64 *v32; // rdx
  char *v33; // rax
  char *v34; // rax
  char *v35; // rax
  char *v36; // rax
  _QWORD *v37; // r14
  _QWORD *v38; // r13
  unsigned __int64 v39; // rsi
  _QWORD *v40; // rax
  _QWORD *v41; // rdi
  __int64 v42; // rcx
  __int64 v43; // rdx
  __int64 v44; // rax
  _QWORD *v45; // rdi
  __int64 v46; // rcx
  __int64 v47; // rdx
  unsigned __int64 v48; // rdx
  _QWORD *v49; // r14
  _QWORD *v50; // r13
  unsigned __int64 v51; // rsi
  _QWORD *v52; // rax
  _QWORD *v53; // rdi
  __int64 v54; // rcx
  __int64 v55; // rdx
  __int64 v56; // rax
  _QWORD *v57; // rdi
  __int64 v58; // rcx
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 *v63; // rdi
  int v64; // r13d
  __int64 v65; // r8
  __int64 v66; // r9
  unsigned int v67; // eax
  __int64 v68; // r14
  __int64 v69; // rax
  __int64 v70; // r12
  unsigned __int64 v71; // rdx
  __int64 *v72; // rax
  int v73; // ebx
  __int64 *v74; // rcx
  __int64 *v75; // rdx
  __int64 v76; // r12
  __int64 v77; // rax
  unsigned __int64 v78; // rax
  __int64 v79; // r14
  int v80; // eax
  __int64 v81; // rcx
  __int64 v82; // rcx
  __int64 v83; // r8
  __int64 v84; // r9
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // rdx
  __int64 v90; // r8
  __int64 v91; // r9
  __int64 v92; // rcx
  __int64 v93; // rax
  __int64 v94; // rsi
  __int64 v95; // rdx
  __int64 v96; // r8
  unsigned __int64 v97; // rax
  __int64 v98; // r12
  __int64 v99; // rax
  unsigned __int64 v100; // rax
  __int64 v101; // rcx
  unsigned __int64 v102; // rax
  bool v103; // zf
  unsigned int v104; // eax
  __int64 v105; // rdx
  size_t v106; // rdx
  __int64 v107; // rdx
  _QWORD *v108; // rax
  _QWORD *v109; // rdx
  __int64 v110; // rcx
  __int64 v111; // r8
  __int64 v112; // r9
  unsigned int v113; // ecx
  _QWORD *v114; // rdi
  __int64 v115; // rsi
  unsigned int v116; // eax
  int v117; // eax
  unsigned __int64 v118; // rax
  __int64 v119; // rax
  int v120; // ecx
  __int64 v121; // r13
  _QWORD *v122; // rax
  _QWORD *j; // rdx
  unsigned int v124; // ecx
  char *v125; // rdi
  __int64 v126; // rsi
  unsigned int v127; // eax
  int v128; // r13d
  unsigned int v129; // eax
  _QWORD *v130; // rax
  _QWORD *i; // rdx
  _QWORD *v132; // rsi
  char *v133; // rax
  __int64 v134; // [rsp+8h] [rbp-1898h]
  __int64 v135; // [rsp+10h] [rbp-1890h]
  __int64 v136; // [rsp+20h] [rbp-1880h]
  int v137; // [rsp+28h] [rbp-1878h]
  unsigned __int64 v138[54]; // [rsp+30h] [rbp-1870h] BYREF
  __int64 v139; // [rsp+1E0h] [rbp-16C0h] BYREF
  __int64 *v140; // [rsp+1E8h] [rbp-16B8h]
  int v141; // [rsp+1F0h] [rbp-16B0h]
  int v142; // [rsp+1F4h] [rbp-16ACh]
  int v143; // [rsp+1F8h] [rbp-16A8h]
  char v144; // [rsp+1FCh] [rbp-16A4h]
  __int64 v145; // [rsp+200h] [rbp-16A0h] BYREF
  __int64 *v146; // [rsp+240h] [rbp-1660h]
  __int64 v147; // [rsp+248h] [rbp-1658h]
  __int64 v148; // [rsp+250h] [rbp-1650h] BYREF
  int v149; // [rsp+258h] [rbp-1648h]
  __int64 v150; // [rsp+260h] [rbp-1640h]
  int v151; // [rsp+268h] [rbp-1638h]
  __int64 v152; // [rsp+270h] [rbp-1630h]
  char v153[8]; // [rsp+390h] [rbp-1510h] BYREF
  unsigned __int64 v154; // [rsp+398h] [rbp-1508h]
  char v155; // [rsp+3ACh] [rbp-14F4h]
  char v156[64]; // [rsp+3B0h] [rbp-14F0h] BYREF
  _BYTE *v157; // [rsp+3F0h] [rbp-14B0h] BYREF
  __int64 v158; // [rsp+3F8h] [rbp-14A8h]
  _BYTE v159[320]; // [rsp+400h] [rbp-14A0h] BYREF
  char v160[8]; // [rsp+540h] [rbp-1360h] BYREF
  unsigned __int64 v161; // [rsp+548h] [rbp-1358h]
  char v162; // [rsp+55Ch] [rbp-1344h]
  char v163[64]; // [rsp+560h] [rbp-1340h] BYREF
  _BYTE *v164; // [rsp+5A0h] [rbp-1300h] BYREF
  __int64 v165; // [rsp+5A8h] [rbp-12F8h]
  _BYTE v166[320]; // [rsp+5B0h] [rbp-12F0h] BYREF
  __int64 *v167; // [rsp+6F0h] [rbp-11B0h] BYREF
  unsigned __int64 v168; // [rsp+6F8h] [rbp-11A8h]
  __int64 v169; // [rsp+700h] [rbp-11A0h] BYREF
  __int64 v170; // [rsp+708h] [rbp-1198h]
  char *v171; // [rsp+750h] [rbp-1150h] BYREF
  unsigned int v172; // [rsp+758h] [rbp-1148h]
  char v173; // [rsp+760h] [rbp-1140h] BYREF
  char v174[8]; // [rsp+8A0h] [rbp-1000h] BYREF
  unsigned __int64 v175; // [rsp+8A8h] [rbp-FF8h]
  char v176; // [rsp+8BCh] [rbp-FE4h]
  char *v177; // [rsp+900h] [rbp-FA0h] BYREF
  unsigned int v178; // [rsp+908h] [rbp-F98h]
  char v179; // [rsp+910h] [rbp-F90h] BYREF
  _QWORD v180[2]; // [rsp+A50h] [rbp-E50h] BYREF
  char v181; // [rsp+A60h] [rbp-E40h] BYREF
  __int64 v182; // [rsp+AA0h] [rbp-E00h]
  __int64 v183; // [rsp+AA8h] [rbp-DF8h]
  char v184; // [rsp+AB0h] [rbp-DF0h] BYREF
  _QWORD v185[2]; // [rsp+BD0h] [rbp-CD0h] BYREF
  char v186; // [rsp+BE0h] [rbp-CC0h] BYREF
  _QWORD v187[2]; // [rsp+D00h] [rbp-BA0h] BYREF
  char v188; // [rsp+D10h] [rbp-B90h] BYREF
  _QWORD v189[3]; // [rsp+D50h] [rbp-B50h] BYREF
  int v190; // [rsp+D68h] [rbp-B38h]
  char v191; // [rsp+D6Ch] [rbp-B34h]
  char v192; // [rsp+D70h] [rbp-B30h] BYREF
  __int64 v193; // [rsp+DF0h] [rbp-AB0h]
  char *v194; // [rsp+DF8h] [rbp-AA8h]
  __int64 v195; // [rsp+E00h] [rbp-AA0h]
  int v196; // [rsp+E08h] [rbp-A98h]
  char v197; // [rsp+E0Ch] [rbp-A94h]
  char v198; // [rsp+E10h] [rbp-A90h] BYREF
  __int64 v199; // [rsp+E90h] [rbp-A10h]
  __int64 v200; // [rsp+E98h] [rbp-A08h]
  __int64 v201; // [rsp+EA0h] [rbp-A00h]
  int v202; // [rsp+EA8h] [rbp-9F8h]
  __int64 v203; // [rsp+EB0h] [rbp-9F0h]
  __int64 v204; // [rsp+EB8h] [rbp-9E8h]
  __int64 v205; // [rsp+EC0h] [rbp-9E0h]
  __int64 v206; // [rsp+EC8h] [rbp-9D8h]
  _QWORD *v207; // [rsp+ED0h] [rbp-9D0h]
  __int64 v208; // [rsp+ED8h] [rbp-9C8h]
  _QWORD v209[3]; // [rsp+EE0h] [rbp-9C0h] BYREF
  int v210; // [rsp+EF8h] [rbp-9A8h]
  __int64 v211; // [rsp+F00h] [rbp-9A0h]
  __int64 v212; // [rsp+F08h] [rbp-998h]
  __int64 v213; // [rsp+F10h] [rbp-990h]
  __int64 v214; // [rsp+F18h] [rbp-988h]
  _BYTE *v215; // [rsp+F20h] [rbp-980h]
  __int64 v216; // [rsp+F28h] [rbp-978h]
  _BYTE v217[16]; // [rsp+F30h] [rbp-970h] BYREF
  __int64 v218; // [rsp+F40h] [rbp-960h]
  __int64 v219; // [rsp+F48h] [rbp-958h]
  __int64 v220; // [rsp+F50h] [rbp-950h]
  int v221; // [rsp+F58h] [rbp-948h]
  __int64 v222; // [rsp+F60h] [rbp-940h]
  __int64 v223; // [rsp+F68h] [rbp-938h]
  __int64 v224; // [rsp+F70h] [rbp-930h]
  __int64 v225; // [rsp+F78h] [rbp-928h]
  char v226; // [rsp+F80h] [rbp-920h] BYREF
  _QWORD v227[2]; // [rsp+10C0h] [rbp-7E0h] BYREF
  char v228; // [rsp+10D0h] [rbp-7D0h]
  char *v229; // [rsp+10D8h] [rbp-7C8h]
  __int64 v230; // [rsp+10E0h] [rbp-7C0h]
  char v231; // [rsp+10E8h] [rbp-7B8h] BYREF
  __int16 v232; // [rsp+1168h] [rbp-738h]
  _QWORD v233[3]; // [rsp+1170h] [rbp-730h] BYREF
  char v234; // [rsp+1188h] [rbp-718h] BYREF
  _QWORD v235[4]; // [rsp+1208h] [rbp-698h] BYREF
  __int64 v236; // [rsp+1228h] [rbp-678h] BYREF
  void *s; // [rsp+1230h] [rbp-670h]
  _BYTE v238[12]; // [rsp+1238h] [rbp-668h]
  char v239; // [rsp+1244h] [rbp-65Ch]
  char v240; // [rsp+1248h] [rbp-658h] BYREF
  __int64 v241; // [rsp+12C8h] [rbp-5D8h]
  void *v242; // [rsp+12D0h] [rbp-5D0h]
  __int64 v243; // [rsp+12D8h] [rbp-5C8h]
  __int64 v244; // [rsp+12E0h] [rbp-5C0h]
  __int64 v245; // [rsp+12E8h] [rbp-5B8h]
  _QWORD *v246; // [rsp+12F0h] [rbp-5B0h]
  __int64 v247; // [rsp+12F8h] [rbp-5A8h]
  __int64 v248; // [rsp+1300h] [rbp-5A0h]
  char *v249; // [rsp+1308h] [rbp-598h]
  __int64 v250; // [rsp+1310h] [rbp-590h]
  char v251; // [rsp+1318h] [rbp-588h] BYREF
  __int64 v252; // [rsp+1518h] [rbp-388h]
  char *v253; // [rsp+1520h] [rbp-380h]
  __int64 v254; // [rsp+1528h] [rbp-378h]
  int v255; // [rsp+1530h] [rbp-370h]
  char v256; // [rsp+1534h] [rbp-36Ch]
  char v257; // [rsp+1538h] [rbp-368h] BYREF
  _QWORD v258[3]; // [rsp+1558h] [rbp-348h] BYREF
  int v259; // [rsp+1570h] [rbp-330h]
  char v260; // [rsp+1574h] [rbp-32Ch]
  char v261; // [rsp+1578h] [rbp-328h] BYREF
  __int64 v262; // [rsp+1678h] [rbp-228h]
  __int64 v263; // [rsp+1680h] [rbp-220h]
  __int64 v264; // [rsp+1688h] [rbp-218h]
  __int64 v265; // [rsp+1690h] [rbp-210h]
  _QWORD *v266; // [rsp+1698h] [rbp-208h]
  __int64 v267; // [rsp+16A0h] [rbp-200h]
  _QWORD v268[11]; // [rsp+16A8h] [rbp-1F8h] BYREF
  int v269; // [rsp+1700h] [rbp-1A0h]
  __int64 *v270; // [rsp+1708h] [rbp-198h]
  __int64 v271; // [rsp+1710h] [rbp-190h]
  __int64 v272; // [rsp+1718h] [rbp-188h] BYREF
  __int64 v273; // [rsp+1720h] [rbp-180h]
  __int64 *v274; // [rsp+1728h] [rbp-178h]
  __int64 v275; // [rsp+1730h] [rbp-170h]
  __int64 v276; // [rsp+1738h] [rbp-168h]
  __int64 v277; // [rsp+1740h] [rbp-160h]
  __int64 v278; // [rsp+1748h] [rbp-158h]
  __int64 v279; // [rsp+1750h] [rbp-150h]
  __int64 v280; // [rsp+1758h] [rbp-148h]
  unsigned __int64 v281; // [rsp+1760h] [rbp-140h]
  __int64 v282; // [rsp+1768h] [rbp-138h]
  int v283; // [rsp+1770h] [rbp-130h]
  int v284; // [rsp+1774h] [rbp-12Ch]
  char *v285; // [rsp+1778h] [rbp-128h]
  __int64 v286; // [rsp+1780h] [rbp-120h]
  char v287; // [rsp+1788h] [rbp-118h] BYREF
  __int64 v288; // [rsp+17A8h] [rbp-F8h]
  __int64 v289; // [rsp+17B0h] [rbp-F0h]
  __int16 v290; // [rsp+17B8h] [rbp-E8h]
  __int64 v291; // [rsp+17C0h] [rbp-E0h]
  _QWORD *v292; // [rsp+17C8h] [rbp-D8h]
  _QWORD *v293; // [rsp+17D0h] [rbp-D0h]
  __int64 v294; // [rsp+17D8h] [rbp-C8h]
  int v295; // [rsp+17E0h] [rbp-C0h]
  __int16 v296; // [rsp+17E4h] [rbp-BCh]
  char v297; // [rsp+17E6h] [rbp-BAh]
  __int64 v298; // [rsp+17E8h] [rbp-B8h]
  __int64 v299; // [rsp+17F0h] [rbp-B0h]
  _QWORD v300[2]; // [rsp+17F8h] [rbp-A8h] BYREF
  _QWORD v301[4]; // [rsp+1808h] [rbp-98h] BYREF
  int v302; // [rsp+1828h] [rbp-78h]
  __int64 v303; // [rsp+1830h] [rbp-70h]
  char v304; // [rsp+1840h] [rbp-60h]
  __int64 v305; // [rsp+1848h] [rbp-58h]
  __int64 v306; // [rsp+1850h] [rbp-50h]
  __int64 v307; // [rsp+1858h] [rbp-48h]
  __int64 v308; // [rsp+1860h] [rbp-40h]

  v12 = a2;
  v13 = a1;
  *(_QWORD *)a1 = a3;
  *(_QWORD *)(a1 + 8) = a4;
  *(_QWORD *)(a1 + 32) = a8;
  *(_QWORD *)(a1 + 16) = a5;
  *(_QWORD *)(a1 + 40) = a9;
  *(_QWORD *)(a1 + 24) = a6;
  *(_QWORD *)(a1 + 48) = a10;
  *(_QWORD *)(a1 + 56) = a11;
  *(_QWORD *)(a1 + 64) = sub_B2BEC0(a2);
  sub_2B3FDA0(a1 + 72);
  v14 = *(_QWORD *)(a1 + 104);
  v15 = v14 + 88LL * *(unsigned int *)(a1 + 112);
  while ( v14 != v15 )
  {
    while ( 1 )
    {
      v15 -= 88;
      v16 = *(_QWORD *)(v15 + 8);
      if ( v16 == v15 + 24 )
        break;
      _libc_free(v16);
      if ( v14 == v15 )
        goto LABEL_5;
    }
  }
LABEL_5:
  *(_DWORD *)(v13 + 112) = 0;
  sub_2B3FDA0(v13 + 120);
  v17 = *(_QWORD *)(v13 + 152);
  v18 = v17 + 88LL * *(unsigned int *)(v13 + 160);
  while ( v17 != v18 )
  {
    v18 -= 88;
    v19 = *(_QWORD *)(v18 + 8);
    if ( v19 != v18 + 24 )
      _libc_free(v19);
  }
  v20 = *(__int64 **)(v13 + 8);
  *(_DWORD *)(v13 + 160) = 0;
  sub_DFB180(v20, 1u);
  if ( !(unsigned int)sub_DFB120((__int64)v20) )
    return 0;
  v21 = sub_B2D610(a2, 30);
  if ( (_BYTE)v21 )
    return 0;
  v23 = *(_QWORD *)(v13 + 64);
  v24 = *(_QWORD *)(v13 + 56);
  v182 = 0;
  v25 = *(_QWORD *)(v13 + 48);
  v26 = *(_QWORD *)(v13 + 40);
  v180[0] = &v181;
  v27 = *(_QWORD *)(v13 + 32);
  v28 = *(_QWORD *)(v13 + 24);
  v180[1] = 0x800000000LL;
  v29 = *(_QWORD *)(v13 + 16);
  v30 = *(_QWORD *)(v13 + 8);
  v183 = 1;
  v31 = &v184;
  v32 = *(__int64 **)v13;
  do
  {
    *(_QWORD *)v31 = -4096;
    v31 += 72;
  }
  while ( v31 != (char *)v185 );
  v33 = &v186;
  v185[0] = 0;
  v185[1] = 1;
  do
  {
    *(_QWORD *)v33 = -4096;
    v33 += 72;
  }
  while ( v33 != (char *)v187 );
  v34 = &v188;
  v187[0] = 0;
  v187[1] = 1;
  do
  {
    *(_QWORD *)v34 = -4096;
    v34 += 16;
  }
  while ( v34 != (char *)v189 );
  v189[0] = 0;
  v189[1] = &v192;
  v194 = &v198;
  v207 = v209;
  v215 = v217;
  v35 = &v226;
  v189[2] = 16;
  v190 = 0;
  v191 = 1;
  v193 = 0;
  v195 = 16;
  v196 = 0;
  v197 = 1;
  v199 = 0;
  v200 = 0;
  v201 = 0;
  v202 = 0;
  v203 = 0;
  v204 = 0;
  v205 = 0;
  v206 = 0;
  v208 = 0;
  memset(v209, 0, sizeof(v209));
  v210 = 0;
  v211 = 0;
  v212 = 0;
  v213 = 0;
  v214 = 0;
  v216 = 0;
  v217[0] = 0;
  v217[8] = 0;
  v218 = 0;
  v219 = 0;
  v220 = 0;
  v221 = 0;
  v224 = 0;
  v225 = 1;
  v222 = v28;
  v223 = v28;
  do
  {
    *(_QWORD *)v35 = -4;
    v35 += 40;
    *((_QWORD *)v35 - 4) = -3;
    *((_QWORD *)v35 - 3) = -4;
    *((_QWORD *)v35 - 2) = -3;
  }
  while ( v35 != (char *)v227 );
  v227[1] = 0;
  v227[0] = v233;
  v229 = &v231;
  v230 = 0x400000000LL;
  v228 = 0;
  v232 = 256;
  v233[1] = 0;
  v233[2] = 1;
  v233[0] = &unk_49DDBE8;
  v36 = &v234;
  do
  {
    *(_QWORD *)v36 = -4096;
    v36 += 16;
  }
  while ( v36 != (char *)v235 );
  memset(v235, 0, sizeof(v235));
  s = &v240;
  v249 = &v251;
  v250 = 0x1000000000LL;
  v253 = &v257;
  v258[1] = &v261;
  v236 = 0;
  *(_QWORD *)v238 = 16;
  *(_DWORD *)&v238[8] = 0;
  v239 = 1;
  v241 = 0;
  v242 = 0;
  v243 = 0;
  v244 = 0;
  v245 = 0;
  v246 = 0;
  v247 = 0;
  v248 = 0;
  v252 = 0;
  v254 = 4;
  v255 = 0;
  v256 = 1;
  v258[0] = 0;
  v258[2] = 32;
  v259 = 0;
  v260 = 1;
  v262 = 0;
  v263 = 0;
  v264 = 0;
  v265 = 0;
  v266 = v268;
  v270 = &v272;
  v276 = v29;
  v282 = a12;
  v277 = v27;
  v280 = v24;
  v274 = v32;
  v275 = v30;
  v279 = v25;
  v281 = v23;
  v167 = (__int64 *)&unk_49D94D0;
  v168 = v23;
  v267 = 0;
  memset(v268, 0, sizeof(v268));
  v269 = 0;
  v271 = 0;
  v272 = 0;
  v273 = v12;
  v278 = v26;
  v291 = sub_B2BE50(*v32);
  v292 = v300;
  v293 = v301;
  v286 = 0x200000000LL;
  v300[1] = v168;
  v296 = 512;
  v290 = 0;
  v301[0] = &unk_49DA0B0;
  v285 = &v287;
  v300[0] = &unk_49D94D0;
  v167 = (__int64 *)&unk_49D94D0;
  v294 = 0;
  v295 = 0;
  v297 = 7;
  v298 = 0;
  v299 = 0;
  v288 = 0;
  v289 = 0;
  nullsub_63();
  v303 = 0x100000000LL;
  memset(&v301[1], 0, 24);
  v302 = 0;
  v304 = 0;
  v305 = 0;
  v306 = 0;
  v307 = 0;
  v308 = 0;
  sub_30AB9D0(v273, v25, v258);
  v37 = sub_C52410();
  v38 = v37 + 1;
  v39 = sub_C959E0();
  v40 = (_QWORD *)v37[2];
  if ( v40 )
  {
    v41 = v37 + 1;
    do
    {
      while ( 1 )
      {
        v42 = v40[2];
        v43 = v40[3];
        if ( v39 <= v40[4] )
          break;
        v40 = (_QWORD *)v40[3];
        if ( !v43 )
          goto LABEL_28;
      }
      v41 = v40;
      v40 = (_QWORD *)v40[2];
    }
    while ( v42 );
LABEL_28:
    if ( v38 != v41 && v39 >= v41[4] )
      v38 = v41;
  }
  if ( v38 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_41;
  v44 = v38[7];
  if ( !v44 )
    goto LABEL_41;
  v45 = v38 + 6;
  do
  {
    while ( 1 )
    {
      v46 = *(_QWORD *)(v44 + 16);
      v47 = *(_QWORD *)(v44 + 24);
      if ( *(_DWORD *)(v44 + 32) >= dword_500FF48 )
        break;
      v44 = *(_QWORD *)(v44 + 24);
      if ( !v47 )
        goto LABEL_37;
    }
    v45 = (_QWORD *)v44;
    v44 = *(_QWORD *)(v44 + 16);
  }
  while ( v46 );
LABEL_37:
  if ( v38 + 6 == v45 || dword_500FF48 < *((_DWORD *)v45 + 8) || !*((_DWORD *)v45 + 9) )
  {
LABEL_41:
    v167 = (__int64 *)sub_DFB1B0(v275);
    v168 = v48;
    v283 = (int)v167;
  }
  else
  {
    v283 = dword_500FFC8;
  }
  v49 = sub_C52410();
  v50 = v49 + 1;
  v51 = sub_C959E0();
  v52 = (_QWORD *)v49[2];
  if ( v52 )
  {
    v53 = v49 + 1;
    do
    {
      while ( 1 )
      {
        v54 = v52[2];
        v55 = v52[3];
        if ( v51 <= v52[4] )
          break;
        v52 = (_QWORD *)v52[3];
        if ( !v55 )
          goto LABEL_47;
      }
      v53 = v52;
      v52 = (_QWORD *)v52[2];
    }
    while ( v54 );
LABEL_47:
    if ( v50 != v53 && v51 >= v53[4] )
      v50 = v53;
  }
  if ( v50 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_58;
  v56 = v50[7];
  if ( !v56 )
    goto LABEL_58;
  v57 = v50 + 6;
  do
  {
    while ( 1 )
    {
      v58 = *(_QWORD *)(v56 + 16);
      v59 = *(_QWORD *)(v56 + 24);
      if ( *(_DWORD *)(v56 + 32) >= dword_500FCA8 )
        break;
      v56 = *(_QWORD *)(v56 + 24);
      if ( !v59 )
        goto LABEL_56;
    }
    v57 = (_QWORD *)v56;
    v56 = *(_QWORD *)(v56 + 16);
  }
  while ( v58 );
LABEL_56:
  if ( v57 == v50 + 6 || dword_500FCA8 < *((_DWORD *)v57 + 8) || !*((_DWORD *)v57 + 9) )
LABEL_58:
    v284 = sub_DFB1F0(v275);
  else
    v284 = dword_500FD28;
  v60 = *(_QWORD *)(v13 + 40);
  v136 = v60;
  if ( *(_BYTE *)(v60 + 112) )
  {
    *(_DWORD *)(v60 + 116) = 0;
  }
  else
  {
    HIDWORD(v168) = 32;
    v167 = &v169;
    v61 = *(_QWORD *)(v60 + 96);
    if ( v61 )
    {
      v62 = *(_QWORD *)(v61 + 24);
      v63 = &v169;
      v64 = 1;
      v65 = v13;
      v169 = *(_QWORD *)(v136 + 96);
      v66 = v12;
      v170 = v62;
      LODWORD(v168) = 1;
      *(_DWORD *)(v61 + 72) = 0;
      v67 = 1;
      do
      {
        v73 = v64++;
        v74 = &v63[2 * v67 - 2];
        v75 = (__int64 *)v74[1];
        if ( v75 == (__int64 *)(*(_QWORD *)(*v74 + 24) + 8LL * *(unsigned int *)(*v74 + 32)) )
        {
          --v67;
          *(_DWORD *)(*v74 + 76) = v73;
          LODWORD(v168) = v67;
        }
        else
        {
          v68 = *v75;
          v74[1] = (__int64)(v75 + 1);
          v69 = (unsigned int)v168;
          v70 = *(_QWORD *)(v68 + 24);
          v71 = (unsigned int)v168 + 1LL;
          if ( v71 > HIDWORD(v168) )
          {
            v134 = v66;
            v135 = v65;
            sub_C8D5F0((__int64)&v167, &v169, v71, 0x10u, v65, v66);
            v63 = v167;
            v69 = (unsigned int)v168;
            v66 = v134;
            v65 = v135;
          }
          v72 = &v63[2 * v69];
          *v72 = v68;
          v72[1] = v70;
          v67 = v168 + 1;
          LODWORD(v168) = v168 + 1;
          *(_DWORD *)(v68 + 72) = v73;
          v63 = v167;
        }
      }
      while ( v67 );
      v13 = v65;
      v12 = v66;
      *(_DWORD *)(v136 + 116) = 0;
      *(_BYTE *)(v136 + 112) = 1;
      if ( v63 != &v169 )
        _libc_free((unsigned __int64)v63);
    }
  }
  v76 = *(_QWORD *)(v12 + 80);
  v141 = 8;
  v143 = 0;
  v144 = 1;
  v142 = 1;
  if ( v76 )
    v76 -= 24;
  memset(v138, 0, sizeof(v138));
  HIDWORD(v138[13]) = 8;
  v138[1] = (unsigned __int64)&v138[4];
  v138[12] = (unsigned __int64)&v138[14];
  v140 = &v145;
  v146 = &v148;
  v147 = 0x800000000LL;
  v77 = *(_QWORD *)(v76 + 48);
  LODWORD(v138[2]) = 8;
  v78 = v77 & 0xFFFFFFFFFFFFFFF8LL;
  BYTE4(v138[3]) = 1;
  v145 = v76;
  v139 = 1;
  if ( v78 == v76 + 48 )
    goto LABEL_186;
  if ( !v78 )
LABEL_179:
    BUG();
  v79 = v78 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v78 - 24) - 30 > 0xA )
  {
LABEL_186:
    v80 = 0;
    v81 = 0;
    v79 = 0;
  }
  else
  {
    v80 = sub_B46E30(v79);
    v81 = v79;
  }
  v152 = v76;
  v149 = v80;
  v148 = v81;
  v150 = v79;
  v151 = 0;
  LODWORD(v147) = 1;
  sub_D4D230((__int64)&v139);
  sub_F1FB80((__int64)v160, (__int64)v138);
  sub_F1FB80((__int64)v153, (__int64)&v139);
  sub_F1FB80((__int64)&v167, (__int64)v153);
  sub_F1FB80((__int64)v174, (__int64)v160);
  if ( v157 != v159 )
    _libc_free((unsigned __int64)v157);
  if ( !v155 )
    _libc_free(v154);
  if ( v164 != v166 )
    _libc_free((unsigned __int64)v164);
  if ( !v162 )
    _libc_free(v161);
  if ( v146 != &v148 )
    _libc_free((unsigned __int64)v146);
  if ( !v144 )
    _libc_free((unsigned __int64)v140);
  if ( (unsigned __int64 *)v138[12] != &v138[14] )
    _libc_free(v138[12]);
  if ( !BYTE4(v138[3]) )
    _libc_free(v138[1]);
  sub_C8CD80((__int64)v153, (__int64)v156, (__int64)&v167, v82, v83, v84);
  v88 = v172;
  v157 = v159;
  v158 = 0x800000000LL;
  if ( v172 )
    sub_2B48A20((__int64)&v157, (__int64 *)&v171, v85, v86, v87, v172);
  sub_C8CD80((__int64)v160, (__int64)v163, (__int64)v174, v86, v87, v88);
  v92 = v178;
  v164 = v166;
  v165 = 0x800000000LL;
  if ( v178 )
  {
    sub_2B48A20((__int64)&v164, (__int64 *)&v177, v89, v178, v90, v91);
    v92 = (unsigned int)v165;
  }
  v93 = (unsigned int)v158;
  while ( 1 )
  {
    v94 = (__int64)v157;
    v95 = 40 * v93;
    if ( v93 == v92 )
      break;
LABEL_101:
    v98 = *(_QWORD *)&v157[v95 - 8];
    v99 = sub_AA4FF0(v98);
    if ( !v99 )
      BUG();
    v100 = (unsigned int)*(unsigned __int8 *)(v99 - 24) - 39;
    if ( (unsigned int)v100 > 0x38 || (v101 = 0x100060000000001LL, !_bittest64(&v101, v100)) )
    {
      v102 = *(_QWORD *)(v98 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v102 == v98 + 48 )
        goto LABEL_109;
      if ( !v102 )
        goto LABEL_179;
      if ( *(_BYTE *)(v102 - 24) != 36 )
      {
LABEL_109:
        ++v236;
        if ( v239 )
          goto LABEL_114;
        v104 = 4 * (*(_DWORD *)&v238[4] - *(_DWORD *)&v238[8]);
        if ( v104 < 0x20 )
          v104 = 32;
        if ( v104 < *(_DWORD *)v238 )
        {
          sub_C8C990((__int64)&v236, v94);
        }
        else
        {
          memset(s, -1, 8LL * *(unsigned int *)v238);
LABEL_114:
          *(_QWORD *)&v238[4] = 0;
        }
        ++v241;
        if ( (_DWORD)v243 )
        {
          v124 = 4 * v243;
          v105 = (unsigned int)v244;
          if ( (unsigned int)(4 * v243) < 0x40 )
            v124 = 64;
          if ( (unsigned int)v244 <= v124 )
          {
LABEL_118:
            v106 = 8 * v105;
            if ( v106 )
              memset(v242, 255, v106);
            goto LABEL_120;
          }
          v125 = (char *)v242;
          v126 = 8LL * (unsigned int)v244;
          if ( (_DWORD)v243 == 1 )
          {
            v128 = 64;
          }
          else
          {
            _BitScanReverse(&v127, v243 - 1);
            v128 = 1 << (33 - (v127 ^ 0x1F));
            if ( v128 < 64 )
              v128 = 64;
            if ( (_DWORD)v244 == v128 )
            {
              v243 = 0;
              v133 = (char *)v242 + v126;
              do
              {
                if ( v125 )
                  *(_QWORD *)v125 = -1;
                v125 += 8;
              }
              while ( v133 != v125 );
              goto LABEL_121;
            }
          }
          sub_C7D6A0((__int64)v242, v126, 8);
          v129 = sub_2B149A0(v128);
          LODWORD(v244) = v129;
          if ( !v129 )
            goto LABEL_188;
          v130 = (_QWORD *)sub_C7D670(8LL * v129, 8);
          v243 = 0;
          v242 = v130;
          for ( i = &v130[(unsigned int)v244]; i != v130; ++v130 )
          {
            if ( v130 )
              *v130 = -1;
          }
        }
        else if ( HIDWORD(v243) )
        {
          v105 = (unsigned int)v244;
          if ( (unsigned int)v244 <= 0x40 )
            goto LABEL_118;
          sub_C7D6A0((__int64)v242, 8LL * (unsigned int)v244, 8);
          LODWORD(v244) = 0;
LABEL_188:
          v242 = 0;
LABEL_120:
          v243 = 0;
        }
LABEL_121:
        ++v245;
        if ( (_DWORD)v247 )
        {
          v113 = 4 * v247;
          v107 = (unsigned int)v248;
          if ( (unsigned int)(4 * v247) < 0x40 )
            v113 = 64;
          if ( v113 >= (unsigned int)v248 )
          {
LABEL_124:
            v108 = v246;
            v109 = &v246[v107];
            if ( v246 != v109 )
            {
              do
                *v108++ = -4096;
              while ( v109 != v108 );
            }
            v247 = 0;
          }
          else
          {
            v114 = v246;
            v115 = (unsigned int)v248;
            if ( (_DWORD)v247 == 1 )
            {
              v121 = 1024;
              v120 = 128;
            }
            else
            {
              _BitScanReverse(&v116, v247 - 1);
              v117 = 1 << (33 - (v116 ^ 0x1F));
              if ( v117 < 64 )
                v117 = 64;
              if ( v117 == (_DWORD)v248 )
              {
                v247 = 0;
                v132 = &v246[v115];
                do
                {
                  if ( v114 )
                    *v114 = -4096;
                  ++v114;
                }
                while ( v132 != v114 );
                goto LABEL_127;
              }
              v118 = (4 * v117 / 3u + 1) | ((unsigned __int64)(4 * v117 / 3u + 1) >> 1);
              v119 = ((((v118 >> 2) | v118 | (((v118 >> 2) | v118) >> 4)) >> 8)
                    | (v118 >> 2)
                    | v118
                    | (((v118 >> 2) | v118) >> 4)
                    | (((((v118 >> 2) | v118 | (((v118 >> 2) | v118) >> 4)) >> 8)
                      | (v118 >> 2)
                      | v118
                      | (((v118 >> 2) | v118) >> 4)) >> 16))
                   + 1;
              v120 = v119;
              v121 = 8 * v119;
            }
            v137 = v120;
            sub_C7D6A0((__int64)v246, v115 * 8, 8);
            LODWORD(v248) = v137;
            v122 = (_QWORD *)sub_C7D670(v121, 8);
            v247 = 0;
            v246 = v122;
            for ( j = &v122[(unsigned int)v248]; j != v122; ++v122 )
            {
              if ( v122 )
                *v122 = -4096;
            }
          }
        }
        else if ( HIDWORD(v247) )
        {
          v107 = (unsigned int)v248;
          if ( (unsigned int)v248 <= 0x40 )
            goto LABEL_124;
          sub_C7D6A0((__int64)v246, 8LL * (unsigned int)v248, 8);
          v246 = 0;
          v247 = 0;
          LODWORD(v248) = 0;
        }
LABEL_127:
        sub_2B5AE70(v13, v98);
        if ( *(_DWORD *)(v13 + 112) )
          v21 |= sub_2BCD660(v13, (__int64)v180, a7);
        v21 |= sub_2BD7F70(v13, v98, (unsigned __int64)v180, v110, v111, v112, a7);
        if ( *(_DWORD *)(v13 + 160) )
          v21 |= sub_2BCFB90(v13, a7, v98, (__int64)v180);
      }
    }
    v103 = (_DWORD)v158 == 1;
    v93 = (unsigned int)(v158 - 1);
    LODWORD(v158) = v158 - 1;
    if ( !v103 )
    {
      sub_D4D230((__int64)v153);
      v93 = (unsigned int)v158;
    }
    v92 = (unsigned int)v165;
  }
  v96 = (__int64)v164;
  if ( v157 != &v157[v95] )
  {
    v92 = (__int64)v164;
    v97 = (unsigned __int64)v157;
    while ( *(_QWORD *)(v97 + 32) == *(_QWORD *)(v92 + 32) )
    {
      if ( *(_DWORD *)(v97 + 24) != *(_DWORD *)(v92 + 24) )
        break;
      v91 = *(unsigned int *)(v92 + 8);
      if ( *(_DWORD *)(v97 + 8) != (_DWORD)v91 )
        break;
      v97 += 40LL;
      v92 += 40;
      if ( &v157[v95] == (_BYTE *)v97 )
        goto LABEL_160;
    }
    goto LABEL_101;
  }
LABEL_160:
  if ( v164 != v166 )
    _libc_free((unsigned __int64)v164);
  if ( !v162 )
    _libc_free(v161);
  if ( v157 != v159 )
    _libc_free((unsigned __int64)v157);
  if ( !v155 )
    _libc_free(v154);
  if ( v177 != &v179 )
    _libc_free((unsigned __int64)v177);
  if ( !v176 )
    _libc_free(v175);
  if ( v171 != &v173 )
    _libc_free((unsigned __int64)v171);
  if ( !BYTE4(v170) )
    _libc_free(v168);
  if ( (_BYTE)v21 )
    sub_2B78EC0((__int64)v180, v94, v95, v92, v96, v91);
  sub_2B41440((__int64)v180);
  return v21;
}
