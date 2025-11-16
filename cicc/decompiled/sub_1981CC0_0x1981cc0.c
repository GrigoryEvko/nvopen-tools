// Function: sub_1981CC0
// Address: 0x1981cc0
//
__int64 __fastcall sub_1981CC0(__int64 a1, __int64 a2, double a3, __m128i a4)
{
  __int64 v5; // r12
  char *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned __int64 v13; // rax
  unsigned int v14; // r13d
  __int64 v16; // rdx
  __int64 v17; // r12
  __int64 v18; // rbx
  unsigned int v19; // edx
  __int64 v20; // r12
  __int32 v21; // ebx
  int v22; // r8d
  int v23; // r9d
  __m128i v24; // xmm0
  const char *v25; // rax
  __int64 v26; // rdi
  __int64 v27; // r12
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  unsigned int v30; // eax
  unsigned int v31; // edx
  int v32; // esi
  __int64 v33; // rax
  unsigned int v34; // ebx
  __int64 *v35; // r14
  _QWORD *v36; // r12
  __int64 *v37; // rdi
  __int64 v38; // rax
  __int64 v39; // r13
  __int64 v40; // r14
  __int64 v41; // rbx
  __int64 i; // r12
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rbx
  __int64 v46; // rax
  __int64 v47; // rax
  _QWORD **v48; // r14
  __int64 v49; // r15
  unsigned __int64 v50; // rbx
  __int64 v51; // rax
  int v52; // r8d
  int v53; // r9d
  unsigned __int8 *v54; // rsi
  __int64 v55; // rax
  unsigned int v56; // edx
  __int64 v57; // r13
  __int64 v58; // r15
  __int64 *v59; // rax
  __int64 *v60; // rcx
  char v61; // dl
  char v62; // al
  __int64 v63; // rax
  __int64 *v64; // rsi
  __int64 *v65; // rcx
  unsigned __int8 *v66; // rsi
  unsigned __int8 *v67; // rsi
  __int64 *v68; // rdi
  __int64 *v69; // rbx
  __int64 v70; // rsi
  __int64 *j; // r13
  __int64 v72; // rdx
  unsigned int v73; // edx
  _QWORD *v74; // rbx
  __int64 v75; // r12
  __int64 v76; // rbx
  __int64 v77; // rsi
  __int64 v78; // rdi
  __int64 v79; // rbx
  __int64 v80; // r12
  __int64 v81; // rax
  __int64 v82; // rax
  _QWORD *v83; // rbx
  _QWORD *v84; // r12
  __int64 v85; // rax
  __int64 *v86; // rax
  __int64 v87; // rcx
  unsigned __int64 v88; // rdx
  __int64 v89; // rcx
  int v90; // eax
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rsi
  __int64 v94; // rbx
  _QWORD *v95; // r12
  __int64 v96; // r13
  __int64 v97; // rsi
  __int64 v98; // r12
  __int64 v99; // rax
  __int64 v100; // rax
  bool v101; // al
  bool v102; // al
  bool v103; // al
  bool v104; // al
  __int64 v105; // rbx
  __int64 v106; // rax
  __int64 v107; // rdx
  __int64 v108; // rsi
  __int64 v109; // rbx
  __int64 v110; // rax
  _QWORD *v111; // r13
  __int64 v112; // r12
  __int64 v113; // r13
  bool v114; // al
  bool v115; // al
  bool v116; // al
  __int64 v117; // r12
  __int64 v118; // rax
  __int64 v119; // rax
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // rsi
  __int64 v124; // rdi
  __int64 v125; // rsi
  __int64 v126; // r13
  __int64 v127; // rsi
  __int64 v128; // rax
  __int64 v129; // rax
  unsigned int v130; // r13d
  unsigned __int64 v131; // rax
  int v132; // eax
  __int64 v133; // rax
  unsigned int v134; // r13d
  unsigned __int64 v135; // rax
  int v136; // eax
  __int64 v137; // r13
  __int64 v138; // rsi
  __int64 v139; // rax
  __int64 v140; // rax
  __int64 v141; // rax
  __int64 v142; // rax
  __int64 v143; // rax
  __int64 v144; // r13
  __int64 v145; // rsi
  _QWORD *v146; // rax
  __int64 v147; // rax
  __int64 v148; // rax
  int v149; // eax
  __int64 v150; // [rsp+18h] [rbp-3F8h]
  __int64 v151; // [rsp+20h] [rbp-3F0h]
  unsigned __int64 v152; // [rsp+28h] [rbp-3E8h]
  __int64 v153; // [rsp+28h] [rbp-3E8h]
  __int64 v154; // [rsp+38h] [rbp-3D8h]
  unsigned __int64 v155; // [rsp+38h] [rbp-3D8h]
  __int64 v156; // [rsp+38h] [rbp-3D8h]
  __int64 v157; // [rsp+40h] [rbp-3D0h]
  __int64 v158; // [rsp+40h] [rbp-3D0h]
  __int64 v159; // [rsp+40h] [rbp-3D0h]
  __int64 v160; // [rsp+40h] [rbp-3D0h]
  unsigned int v161; // [rsp+40h] [rbp-3D0h]
  __int64 v162; // [rsp+40h] [rbp-3D0h]
  __int64 v163; // [rsp+40h] [rbp-3D0h]
  __int64 v164; // [rsp+40h] [rbp-3D0h]
  unsigned __int64 v165; // [rsp+40h] [rbp-3D0h]
  __int64 v166; // [rsp+40h] [rbp-3D0h]
  unsigned __int64 v167; // [rsp+40h] [rbp-3D0h]
  __int64 v168; // [rsp+48h] [rbp-3C8h]
  __int64 v169; // [rsp+48h] [rbp-3C8h]
  unsigned __int64 v170; // [rsp+48h] [rbp-3C8h]
  __int64 v171; // [rsp+48h] [rbp-3C8h]
  __int64 v172; // [rsp+48h] [rbp-3C8h]
  unsigned __int64 v173; // [rsp+50h] [rbp-3C0h]
  unsigned int v174; // [rsp+50h] [rbp-3C0h]
  _QWORD *v175; // [rsp+50h] [rbp-3C0h]
  __int64 v176; // [rsp+50h] [rbp-3C0h]
  unsigned __int64 v177; // [rsp+50h] [rbp-3C0h]
  int v178; // [rsp+58h] [rbp-3B8h]
  unsigned __int64 v179; // [rsp+58h] [rbp-3B8h]
  unsigned int v180; // [rsp+58h] [rbp-3B8h]
  _QWORD *v181; // [rsp+58h] [rbp-3B8h]
  __int64 v182; // [rsp+58h] [rbp-3B8h]
  __int64 v183; // [rsp+58h] [rbp-3B8h]
  __int64 v184; // [rsp+60h] [rbp-3B0h]
  __int64 v185; // [rsp+60h] [rbp-3B0h]
  __int64 v186; // [rsp+60h] [rbp-3B0h]
  __int64 v187; // [rsp+60h] [rbp-3B0h]
  __int64 v188; // [rsp+60h] [rbp-3B0h]
  __int64 *v189; // [rsp+60h] [rbp-3B0h]
  unsigned __int64 v190; // [rsp+60h] [rbp-3B0h]
  __int64 v191; // [rsp+60h] [rbp-3B0h]
  unsigned __int64 v192; // [rsp+60h] [rbp-3B0h]
  __int64 *v193; // [rsp+78h] [rbp-398h]
  __int64 v194; // [rsp+80h] [rbp-390h]
  unsigned __int8 v195; // [rsp+92h] [rbp-37Eh]
  unsigned __int8 v196; // [rsp+93h] [rbp-37Dh]
  int v197; // [rsp+94h] [rbp-37Ch]
  __int64 *v198; // [rsp+A8h] [rbp-368h]
  unsigned int v199; // [rsp+D0h] [rbp-340h] BYREF
  _QWORD *v200; // [rsp+D8h] [rbp-338h]
  __int64 v201; // [rsp+E0h] [rbp-330h]
  char v202; // [rsp+E8h] [rbp-328h]
  unsigned __int8 *v203[2]; // [rsp+F0h] [rbp-320h] BYREF
  _QWORD v204[2]; // [rsp+100h] [rbp-310h] BYREF
  __int64 *v205; // [rsp+110h] [rbp-300h] BYREF
  __int64 v206; // [rsp+118h] [rbp-2F8h]
  _BYTE v207[32]; // [rsp+120h] [rbp-2F0h] BYREF
  _QWORD *v208; // [rsp+140h] [rbp-2D0h] BYREF
  unsigned int v209; // [rsp+148h] [rbp-2C8h]
  unsigned int v210; // [rsp+14Ch] [rbp-2C4h]
  _QWORD v211[4]; // [rsp+150h] [rbp-2C0h] BYREF
  __int64 *v212; // [rsp+170h] [rbp-2A0h] BYREF
  __int64 v213; // [rsp+178h] [rbp-298h]
  _BYTE v214[32]; // [rsp+180h] [rbp-290h] BYREF
  unsigned __int8 *v215; // [rsp+1A0h] [rbp-270h] BYREF
  __int64 v216; // [rsp+1A8h] [rbp-268h]
  unsigned __int64 v217; // [rsp+1B0h] [rbp-260h]
  __int64 v218; // [rsp+1B8h] [rbp-258h]
  __int64 v219; // [rsp+1C0h] [rbp-250h]
  int v220; // [rsp+1C8h] [rbp-248h]
  __int64 v221; // [rsp+1D0h] [rbp-240h]
  __int64 v222; // [rsp+1D8h] [rbp-238h]
  unsigned __int8 *v223; // [rsp+1F0h] [rbp-220h] BYREF
  __int64 *v224; // [rsp+1F8h] [rbp-218h]
  __int64 *v225; // [rsp+200h] [rbp-210h]
  __int64 v226; // [rsp+208h] [rbp-208h]
  int v227; // [rsp+210h] [rbp-200h]
  _BYTE v228[40]; // [rsp+218h] [rbp-1F8h] BYREF
  __m128i v229; // [rsp+240h] [rbp-1D0h] BYREF
  const char *v230; // [rsp+250h] [rbp-1C0h] BYREF
  __int64 v231; // [rsp+258h] [rbp-1B8h]
  _QWORD *v232; // [rsp+260h] [rbp-1B0h]
  __int64 v233; // [rsp+268h] [rbp-1A8h]
  unsigned int v234; // [rsp+270h] [rbp-1A0h]
  __int64 v235; // [rsp+278h] [rbp-198h]
  __int64 v236; // [rsp+280h] [rbp-190h]
  __int64 v237; // [rsp+288h] [rbp-188h]
  __int64 v238; // [rsp+290h] [rbp-180h]
  __int64 v239; // [rsp+298h] [rbp-178h]
  __int64 v240; // [rsp+2A0h] [rbp-170h]
  __int64 v241; // [rsp+2A8h] [rbp-168h]
  __int64 v242; // [rsp+2B0h] [rbp-160h]
  __int64 v243; // [rsp+2B8h] [rbp-158h]
  __int64 v244; // [rsp+2C0h] [rbp-150h]
  __int64 v245; // [rsp+2C8h] [rbp-148h]
  int v246; // [rsp+2D0h] [rbp-140h]
  __int64 v247; // [rsp+2D8h] [rbp-138h]
  _BYTE *v248; // [rsp+2E0h] [rbp-130h]
  _BYTE *v249; // [rsp+2E8h] [rbp-128h]
  __int64 v250; // [rsp+2F0h] [rbp-120h]
  int v251; // [rsp+2F8h] [rbp-118h]
  _BYTE v252[16]; // [rsp+300h] [rbp-110h] BYREF
  __int64 v253; // [rsp+310h] [rbp-100h]
  __int64 v254; // [rsp+318h] [rbp-F8h]
  __int64 v255; // [rsp+320h] [rbp-F0h]
  __int64 v256; // [rsp+328h] [rbp-E8h]
  __int64 v257; // [rsp+330h] [rbp-E0h]
  __int64 v258; // [rsp+338h] [rbp-D8h]
  __int16 v259; // [rsp+340h] [rbp-D0h]
  __int64 v260[5]; // [rsp+348h] [rbp-C8h] BYREF
  int v261; // [rsp+370h] [rbp-A0h]
  __int64 v262; // [rsp+378h] [rbp-98h]
  __int64 v263; // [rsp+380h] [rbp-90h]
  __int64 v264; // [rsp+388h] [rbp-88h]
  _BYTE *v265; // [rsp+390h] [rbp-80h]
  __int64 v266; // [rsp+398h] [rbp-78h]
  _BYTE v267[112]; // [rsp+3A0h] [rbp-70h] BYREF

  *(_QWORD *)(a1 + 16) = a2;
  v5 = sub_157EB90(**(_QWORD **)(a2 + 32));
  v6 = sub_15E0FD0(79);
  v8 = sub_16321A0(v5, (__int64)v6, v7);
  if ( !v8 )
    return 0;
  if ( !*(_QWORD *)(v8 + 8) )
    return 0;
  v9 = sub_1632FA0(v5);
  v10 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 24) = v9;
  v11 = sub_13FC520(v10);
  *(_QWORD *)(a1 + 32) = v11;
  if ( !v11 )
    return 0;
  v12 = sub_13FCB50(*(_QWORD *)(a1 + 16));
  if ( !v12 )
    return 0;
  v13 = sub_157EBA0(v12);
  if ( *(_BYTE *)(v13 + 16) != 26 )
    return 0;
  if ( (*(_DWORD *)(v13 + 20) & 0xFFFFFFF) != 3 )
    return 0;
  v16 = *(_QWORD *)(v13 - 72);
  if ( *(_BYTE *)(v16 + 16) != 75 )
    return 0;
  v17 = *(_QWORD *)(v16 - 48);
  if ( !v17 )
    return 0;
  v18 = *(_QWORD *)(v16 - 24);
  if ( !v18 )
    return 0;
  v19 = *(unsigned __int16 *)(v16 + 18);
  BYTE1(v19) &= ~0x80u;
  if ( **(_QWORD **)(*(_QWORD *)(a1 + 16) + 32LL) != *(_QWORD *)(v13 - 24) )
    v19 = sub_15FF0F0(v19);
  sub_1981660((__int64)&v229, (__int64 *)a1, v19, v17, v18);
  if ( !(_BYTE)v231 )
    return 0;
  if ( *(_QWORD *)(v229.m128i_i64[1] + 40) != 2 )
    return 0;
  v20 = sub_13A5BC0((_QWORD *)v229.m128i_i64[1], *(_QWORD *)a1);
  if ( !sub_1456110(v20) && (!sub_1456170(v20) || !byte_4FB0C40) )
    return 0;
  v21 = v229.m128i_i32[0];
  if ( sub_1456110(v20) )
  {
    if ( ((v21 - 36) & 0xFFFFFFFA) != 0 )
      return 0;
  }
  else if ( (v21 & 0xFFFFFFFA) != 0x22 )
  {
    return 0;
  }
  v195 = v231;
  if ( !(_BYTE)v231 )
    return 0;
  v24 = _mm_load_si128(&v229);
  v25 = v230;
  *(__m128i *)(a1 + 40) = v24;
  *(_QWORD *)(a1 + 56) = v25;
  v14 = (unsigned __int8)byte_4FB0B60;
  if ( byte_4FB0B60 || !*(_QWORD *)(a1 + 8) )
  {
LABEL_36:
    v205 = (__int64 *)v207;
    v206 = 0x400000000LL;
    v38 = *(_QWORD *)(a1 + 16);
    v39 = *(_QWORD *)(v38 + 32);
    v40 = *(_QWORD *)(v38 + 40);
    if ( v39 != v40 )
    {
      do
      {
        v41 = *(_QWORD *)(*(_QWORD *)v39 + 48LL);
        for ( i = *(_QWORD *)v39 + 40LL; i != v41; v41 = *(_QWORD *)(v41 + 8) )
        {
          while ( 1 )
          {
            if ( !v41 )
              BUG();
            if ( *(_BYTE *)(v41 - 8) == 78 )
            {
              v43 = *(_QWORD *)(v41 - 48);
              if ( !*(_BYTE *)(v43 + 16) && (*(_BYTE *)(v43 + 33) & 0x20) != 0 && *(_DWORD *)(v43 + 36) == 79 )
                break;
            }
            v41 = *(_QWORD *)(v41 + 8);
            if ( i == v41 )
              goto LABEL_48;
          }
          v44 = (unsigned int)v206;
          if ( (unsigned int)v206 >= HIDWORD(v206) )
          {
            sub_16CD150((__int64)&v205, v207, 0, 8, v22, v23);
            v44 = (unsigned int)v206;
          }
          v205[v44] = v41 - 24;
          LODWORD(v206) = v206 + 1;
        }
LABEL_48:
        v39 += 8;
      }
      while ( v40 != v39 );
      if ( !(_DWORD)v206 )
      {
        v37 = v205;
        v14 = 0;
        goto LABEL_149;
      }
      v45 = *(_QWORD *)(a1 + 24);
      v46 = *(_QWORD *)a1;
      v14 = 0;
      v231 = 0;
      v248 = v252;
      v249 = v252;
      v229.m128i_i64[0] = v46;
      v229.m128i_i64[1] = v45;
      v230 = "loop-predication";
      v232 = 0;
      v233 = 0;
      v234 = 0;
      v235 = 0;
      v236 = 0;
      v237 = 0;
      v238 = 0;
      v239 = 0;
      v240 = 0;
      v241 = 0;
      v242 = 0;
      v243 = 0;
      v244 = 0;
      v245 = 0;
      v246 = 0;
      v247 = 0;
      v250 = 2;
      v251 = 0;
      v253 = 0;
      v254 = 0;
      v255 = 0;
      v256 = 0;
      v257 = 0;
      v258 = 0;
      v259 = 1;
      v47 = sub_15E0530(*(_QWORD *)(v46 + 24));
      v264 = v45;
      v260[3] = v47;
      v265 = v267;
      v266 = 0x800000000LL;
      memset(v260, 0, 24);
      v260[4] = 0;
      v261 = 0;
      v262 = 0;
      v263 = 0;
      v193 = &v205[(unsigned int)v206];
      if ( v205 == v193 )
        goto LABEL_120;
      v198 = v205;
      v48 = (_QWORD **)a1;
      v196 = 0;
      while ( 1 )
      {
        v49 = *v198;
        v50 = sub_157EBA0((__int64)v48[4]);
        v51 = sub_16498A0(v50);
        v221 = 0;
        v222 = 0;
        v54 = *(unsigned __int8 **)(v50 + 48);
        v218 = v51;
        v220 = 0;
        v55 = *(_QWORD *)(v50 + 40);
        v215 = 0;
        v216 = v55;
        v219 = 0;
        v217 = v50 + 24;
        v223 = v54;
        if ( v54 )
        {
          sub_1623A60((__int64)&v223, (__int64)v54, 2);
          if ( v215 )
            sub_161E7C0((__int64)&v215, (__int64)v215);
          v215 = v223;
          if ( v223 )
            sub_1623210((__int64)&v223, v223, (__int64)&v215);
        }
        v194 = v49;
        v197 = 0;
        v56 = 1;
        v57 = *(_QWORD *)(v49 - 24LL * (*(_DWORD *)(v49 + 20) & 0xFFFFFFF));
        v210 = 4;
        v223 = 0;
        v208 = v211;
        v58 = v57;
        v211[0] = v57;
        v224 = (__int64 *)v228;
        v225 = (__int64 *)v228;
        v226 = 4;
        v212 = (__int64 *)v214;
        v213 = 0x400000000LL;
        v59 = (__int64 *)v228;
        v227 = 0;
        v60 = (__int64 *)v228;
        while ( 1 )
        {
          v209 = v56 - 1;
          if ( v59 != v60 )
            goto LABEL_58;
          v64 = &v59[HIDWORD(v226)];
          if ( v64 != v59 )
            break;
LABEL_132:
          if ( HIDWORD(v226) < (unsigned int)v226 )
          {
            ++HIDWORD(v226);
            *v64 = v58;
            ++v223;
            goto LABEL_59;
          }
LABEL_58:
          sub_16CCBA0((__int64)&v223, v58);
          if ( !v61 )
            goto LABEL_74;
LABEL_59:
          v62 = *(_BYTE *)(v58 + 16);
          if ( v62 == 50 )
          {
LABEL_98:
            v79 = *(_QWORD *)(v58 - 48);
            if ( v79 )
            {
              v80 = *(_QWORD *)(v58 - 24);
              if ( v80 )
                goto LABEL_100;
            }
LABEL_62:
            v63 = (unsigned int)v213;
            if ( (unsigned int)v213 >= HIDWORD(v213) )
            {
              sub_16CD150((__int64)&v212, v214, 0, 8, v52, v53);
              v63 = (unsigned int)v213;
            }
            v212[v63] = v58;
            v56 = v209;
            LODWORD(v213) = v213 + 1;
            goto LABEL_65;
          }
LABEL_60:
          if ( v62 == 5 )
          {
            if ( *(_WORD *)(v58 + 18) == 26 )
            {
              v79 = *(_QWORD *)(v58 - 24LL * (*(_DWORD *)(v58 + 20) & 0xFFFFFFF));
              if ( v79 )
              {
                v80 = *(_QWORD *)(v58 + 24 * (1LL - (*(_DWORD *)(v58 + 20) & 0xFFFFFFF)));
                if ( v80 )
                {
LABEL_100:
                  v81 = v209;
                  if ( v209 >= v210 )
                  {
                    sub_16CD150((__int64)&v208, v211, 0, 8, v52, v53);
                    v81 = v209;
                  }
                  v208[v81] = v79;
                  v82 = v209 + 1;
                  v209 = v82;
                  if ( v210 <= (unsigned int)v82 )
                  {
                    sub_16CD150((__int64)&v208, v211, 0, 8, v52, v53);
                    v82 = v209;
                  }
                  v208[v82] = v80;
                  v56 = ++v209;
                  goto LABEL_65;
                }
              }
            }
            goto LABEL_62;
          }
          if ( v62 != 75 )
            goto LABEL_62;
          v73 = *(unsigned __int16 *)(v58 + 18);
          BYTE1(v73) &= ~0x80u;
          sub_1981660((__int64)&v199, (__int64 *)v48, v73, *(_QWORD *)(v58 - 48), *(_QWORD *)(v58 - 24));
          if ( !v202 )
            goto LABEL_62;
          if ( v199 != 36 )
            goto LABEL_62;
          v74 = v200;
          if ( v200[5] != 2 )
            goto LABEL_62;
          v75 = sub_13A5BC0(v200, (__int64)*v48);
          if ( !sub_1456110(v75) && (!sub_1456170(v75) || !byte_4FB0C40) )
            goto LABEL_62;
          v76 = sub_1456040(*(_QWORD *)v74[4]);
          v77 = sub_1456040(*(_QWORD *)v48[6][4]);
          if ( v76 == v77 )
          {
            v94 = (__int64)v48[7];
            v178 = *((_DWORD *)v48 + 10);
            v186 = (__int64)v48[6];
          }
          else
          {
            v184 = 1;
            v78 = (__int64)v48[3];
            while ( 2 )
            {
              switch ( *(_BYTE *)(v77 + 8) )
              {
                case 0:
                case 8:
                case 0xA:
                case 0xC:
                case 0x10:
                  v120 = v184 * *(_QWORD *)(v77 + 32);
                  v77 = *(_QWORD *)(v77 + 24);
                  v184 = v120;
                  continue;
                case 1:
                  v91 = 16;
                  goto LABEL_162;
                case 2:
                  v91 = 32;
                  goto LABEL_162;
                case 3:
                case 9:
                  v91 = 64;
                  goto LABEL_162;
                case 4:
                  v91 = 80;
                  goto LABEL_162;
                case 5:
                case 6:
                  v91 = 128;
                  goto LABEL_162;
                case 7:
                  v149 = sub_15A9520(v78, 0);
                  v78 = (__int64)v48[3];
                  v91 = (unsigned int)(8 * v149);
                  goto LABEL_162;
                case 0xB:
                  v91 = *(_DWORD *)(v77 + 8) >> 8;
                  goto LABEL_162;
                case 0xD:
                  v146 = (_QWORD *)sub_15A9930(v78, v77);
                  v78 = (__int64)v48[3];
                  v91 = 8LL * *v146;
                  goto LABEL_162;
                case 0xE:
                  v144 = 1;
                  v183 = *(_QWORD *)(v77 + 32);
                  v145 = *(_QWORD *)(v77 + 24);
                  v177 = (unsigned int)sub_15A9FE0(v78, v145);
                  while ( 2 )
                  {
                    switch ( *(_BYTE *)(v145 + 8) )
                    {
                      case 1:
                        v147 = 16;
                        break;
                      case 2:
                        v147 = 32;
                        break;
                      case 3:
                      case 9:
                        v147 = 64;
                        break;
                      case 4:
                        v147 = 80;
                        break;
                      case 5:
                      case 6:
                        v147 = 128;
                        break;
                      case 7:
                        v147 = 8 * (unsigned int)sub_15A9520(v78, 0);
                        break;
                      case 8:
                      case 0xA:
                      case 0xC:
                      case 0x10:
                        v148 = *(_QWORD *)(v145 + 32);
                        v145 = *(_QWORD *)(v145 + 24);
                        v144 *= v148;
                        continue;
                      case 0xB:
                        v147 = *(_DWORD *)(v145 + 8) >> 8;
                        break;
                      case 0xD:
                        v147 = 8LL * *(_QWORD *)sub_15A9930(v78, v145);
                        break;
                      case 0xE:
                        v153 = *(_QWORD *)(v145 + 24);
                        v172 = *(_QWORD *)(v145 + 32);
                        v167 = (unsigned int)sub_15A9FE0(v78, v153);
                        v147 = 8
                             * v172
                             * v167
                             * ((v167 + ((unsigned __int64)(sub_127FA20(v78, v153) + 7) >> 3) - 1)
                              / v167);
                        break;
                      case 0xF:
                        v147 = 8 * (unsigned int)sub_15A9520(v78, *(_DWORD *)(v145 + 8) >> 8);
                        break;
                    }
                    break;
                  }
                  v78 = (__int64)v48[3];
                  v91 = 8 * v177 * v183 * ((v177 + ((unsigned __int64)(v144 * v147 + 7) >> 3) - 1) / v177);
                  goto LABEL_162;
                case 0xF:
                  v90 = sub_15A9520(v78, *(_DWORD *)(v77 + 8) >> 8);
                  v78 = (__int64)v48[3];
                  v91 = (unsigned int)(8 * v90);
LABEL_162:
                  v92 = v184 * v91;
                  v93 = v76;
                  v185 = 1;
                  v173 = v92;
                  while ( 2 )
                  {
                    switch ( *(_BYTE *)(v93 + 8) )
                    {
                      case 1:
                        v122 = 16;
                        goto LABEL_195;
                      case 2:
                        v122 = 32;
                        goto LABEL_195;
                      case 3:
                      case 9:
                        v122 = 64;
                        goto LABEL_195;
                      case 4:
                        v122 = 80;
                        goto LABEL_195;
                      case 5:
                      case 6:
                        v122 = 128;
                        goto LABEL_195;
                      case 7:
                        v122 = 8 * (unsigned int)sub_15A9520(v78, 0);
                        goto LABEL_195;
                      case 0xB:
                        v122 = *(_DWORD *)(v93 + 8) >> 8;
                        goto LABEL_195;
                      case 0xD:
                        v122 = 8LL * *(_QWORD *)sub_15A9930(v78, v93);
                        goto LABEL_195;
                      case 0xE:
                        v126 = 1;
                        v182 = *(_QWORD *)(v93 + 32);
                        v127 = *(_QWORD *)(v93 + 24);
                        v170 = (unsigned int)sub_15A9FE0(v78, v127);
                        while ( 2 )
                        {
                          switch ( *(_BYTE *)(v127 + 8) )
                          {
                            case 1:
                              v141 = 16;
                              goto LABEL_226;
                            case 2:
                              v141 = 32;
                              goto LABEL_226;
                            case 3:
                            case 9:
                              v141 = 64;
                              goto LABEL_226;
                            case 4:
                              v141 = 80;
                              goto LABEL_226;
                            case 5:
                            case 6:
                              v141 = 128;
                              goto LABEL_226;
                            case 7:
                              v141 = 8 * (unsigned int)sub_15A9520(v78, 0);
                              goto LABEL_226;
                            case 0xB:
                              v141 = *(_DWORD *)(v127 + 8) >> 8;
                              goto LABEL_226;
                            case 0xD:
                              v141 = 8LL * *(_QWORD *)sub_15A9930(v78, v127);
                              goto LABEL_226;
                            case 0xE:
                              v151 = *(_QWORD *)(v127 + 24);
                              v166 = *(_QWORD *)(v127 + 32);
                              v155 = (unsigned int)sub_15A9FE0(v78, v151);
                              v141 = 8
                                   * v166
                                   * v155
                                   * ((v155 + ((unsigned __int64)(sub_127FA20(v78, v151) + 7) >> 3) - 1)
                                    / v155);
                              goto LABEL_226;
                            case 0xF:
                              v141 = 8 * (unsigned int)sub_15A9520(v78, *(_DWORD *)(v127 + 8) >> 8);
LABEL_226:
                              v122 = 8 * v170 * v182 * ((v170 + ((unsigned __int64)(v126 * v141 + 7) >> 3) - 1) / v170);
                              goto LABEL_195;
                            case 0x10:
                              v140 = *(_QWORD *)(v127 + 32);
                              v127 = *(_QWORD *)(v127 + 24);
                              v126 *= v140;
                              continue;
                            default:
                              goto LABEL_286;
                          }
                        }
                      case 0xF:
                        v122 = 8 * (unsigned int)sub_15A9520(v78, *(_DWORD *)(v93 + 8) >> 8);
LABEL_195:
                        if ( v185 * v122 > v173 )
                          goto LABEL_62;
                        if ( !byte_4FB0D20 )
                          goto LABEL_62;
                        v181 = v48[7];
                        if ( *((_WORD *)v181 + 12) )
                          goto LABEL_62;
                        v123 = (__int64)v48[6];
                        v176 = **(_QWORD **)(v123 + 32);
                        if ( *(_WORD *)(v176 + 24)
                          || !(unsigned __int8)sub_14798E0((__int64)*v48, v123, *((_DWORD *)v48 + 10), (bool *)v203) )
                        {
                          goto LABEL_62;
                        }
                        v124 = (__int64)v48[3];
                        v125 = v76;
                        v191 = 1;
                        while ( 2 )
                        {
                          switch ( *(_BYTE *)(v125 + 8) )
                          {
                            case 1:
                              v128 = 16;
                              goto LABEL_206;
                            case 2:
                              v128 = 32;
                              goto LABEL_206;
                            case 3:
                            case 9:
                              v128 = 64;
                              goto LABEL_206;
                            case 4:
                              v128 = 80;
                              goto LABEL_206;
                            case 5:
                            case 6:
                              v128 = 128;
                              goto LABEL_206;
                            case 7:
                              v128 = 8 * (unsigned int)sub_15A9520(v124, 0);
                              goto LABEL_206;
                            case 0xB:
                              v128 = *(_DWORD *)(v125 + 8) >> 8;
                              goto LABEL_206;
                            case 0xD:
                              v128 = 8LL * *(_QWORD *)sub_15A9930(v124, v125);
                              goto LABEL_206;
                            case 0xE:
                              v137 = 1;
                              v171 = *(_QWORD *)(v125 + 32);
                              v138 = *(_QWORD *)(v125 + 24);
                              v165 = (unsigned int)sub_15A9FE0(v124, v138);
                              while ( 2 )
                              {
                                switch ( *(_BYTE *)(v138 + 8) )
                                {
                                  case 1:
                                    v143 = 16;
                                    goto LABEL_231;
                                  case 2:
                                    v143 = 32;
                                    goto LABEL_231;
                                  case 3:
                                  case 9:
                                    v143 = 64;
                                    goto LABEL_231;
                                  case 4:
                                    v143 = 80;
                                    goto LABEL_231;
                                  case 5:
                                  case 6:
                                    v143 = 128;
                                    goto LABEL_231;
                                  case 7:
                                    v143 = 8 * (unsigned int)sub_15A9520(v124, 0);
                                    goto LABEL_231;
                                  case 0xB:
                                    v143 = *(_DWORD *)(v138 + 8) >> 8;
                                    goto LABEL_231;
                                  case 0xD:
                                    v143 = 8LL * *(_QWORD *)sub_15A9930(v124, v138);
                                    goto LABEL_231;
                                  case 0xE:
                                    v150 = *(_QWORD *)(v138 + 24);
                                    v156 = *(_QWORD *)(v138 + 32);
                                    v152 = (unsigned int)sub_15A9FE0(v124, v150);
                                    v143 = 8
                                         * v156
                                         * v152
                                         * ((v152 + ((unsigned __int64)(sub_127FA20(v124, v150) + 7) >> 3) - 1)
                                          / v152);
                                    goto LABEL_231;
                                  case 0xF:
                                    v143 = 8 * (unsigned int)sub_15A9520(v124, *(_DWORD *)(v138 + 8) >> 8);
LABEL_231:
                                    v128 = 8
                                         * v165
                                         * v171
                                         * ((v165 + ((unsigned __int64)(v137 * v143 + 7) >> 3) - 1)
                                          / v165);
                                    goto LABEL_206;
                                  case 0x10:
                                    v142 = *(_QWORD *)(v138 + 32);
                                    v138 = *(_QWORD *)(v138 + 24);
                                    v137 *= v142;
                                    continue;
                                  default:
                                    goto LABEL_286;
                                }
                              }
                            case 0xF:
                              v128 = 8 * (unsigned int)sub_15A9520(v124, *(_DWORD *)(v125 + 8) >> 8);
LABEL_206:
                              v192 = v191 * v128;
                              v129 = *(_QWORD *)(v176 + 32);
                              v130 = *(_DWORD *)(v129 + 32);
                              if ( v130 > 0x40 )
                              {
                                v132 = sub_16A57B0(v129 + 24);
                              }
                              else
                              {
                                v131 = *(_QWORD *)(v129 + 24);
                                if ( v131 )
                                {
                                  _BitScanReverse64(&v131, v131);
                                  LODWORD(v131) = v131 ^ 0x3F;
                                }
                                else
                                {
                                  LODWORD(v131) = 64;
                                }
                                v132 = v130 + v131 - 64;
                              }
                              if ( v130 - v132 >= v192 )
                                goto LABEL_62;
                              v133 = v181[4];
                              v134 = *(_DWORD *)(v133 + 32);
                              if ( v134 > 0x40 )
                              {
                                v136 = sub_16A57B0(v133 + 24);
                              }
                              else
                              {
                                v135 = *(_QWORD *)(v133 + 24);
                                if ( v135 )
                                {
                                  _BitScanReverse64(&v135, v135);
                                  LODWORD(v135) = v135 ^ 0x3F;
                                }
                                else
                                {
                                  LODWORD(v135) = 64;
                                }
                                v136 = v134 + v135 - 64;
                              }
                              if ( v134 - v136 >= v192 )
                                goto LABEL_62;
                              v178 = *((_DWORD *)v48 + 10);
                              v186 = sub_14835F0(*v48, (__int64)v48[6], v76, 0, v24, a4);
                              if ( *(_WORD *)(v186 + 24) != 7 )
                                goto LABEL_62;
                              v94 = sub_14835F0(*v48, (__int64)v48[7], v76, 0, v24, a4);
                              break;
                            case 0x10:
                              v139 = v191 * *(_QWORD *)(v125 + 32);
                              v125 = *(_QWORD *)(v125 + 24);
                              v191 = v139;
                              continue;
                            default:
                              goto LABEL_286;
                          }
                          break;
                        }
                        break;
                      case 0x10:
                        v121 = v185 * *(_QWORD *)(v93 + 32);
                        v93 = *(_QWORD *)(v93 + 24);
                        v185 = v121;
                        continue;
                      default:
                        goto LABEL_286;
                    }
                    break;
                  }
                  break;
                default:
LABEL_286:
                  BUG();
              }
              break;
            }
          }
          if ( v75 != sub_13A5BC0((_QWORD *)v186, (__int64)*v48) )
            goto LABEL_62;
          if ( sub_1456110(v75) )
          {
            v95 = v200;
            v96 = v201;
            v174 = v199;
            v97 = sub_1456040(*(_QWORD *)v200[4]);
            v98 = *(_QWORD *)v95[4];
            v99 = v186;
            v187 = (__int64)*v48;
            v168 = **(_QWORD **)(v99 + 32);
            v100 = sub_145CF80((__int64)*v48, v97, 1, 0);
            v154 = v187;
            v188 = sub_14806B0(v187, v168, v100, 0, 0);
            v204[0] = sub_14806B0((__int64)*v48, v96, v98, 0, 0);
            v203[0] = (unsigned __int8 *)v204;
            v204[1] = v188;
            v203[1] = (unsigned __int8 *)0x200000002LL;
            v189 = sub_147DD40(v154, (__int64 *)v203, 0, 0, v24, a4);
            if ( (_QWORD *)v203[0] != v204 )
              _libc_free((unsigned __int64)v203[0]);
            v157 = (__int64)*v48;
            v101 = sub_146CEE0((__int64)*v48, v98, (__int64)v48[2]);
            v52 = v157;
            if ( !v101 )
              goto LABEL_62;
            if ( !(unsigned __int8)sub_3870AF0(v98, v157) )
              goto LABEL_62;
            v158 = (__int64)*v48;
            v102 = sub_146CEE0((__int64)*v48, v96, (__int64)v48[2]);
            v52 = v158;
            if ( !v102 )
              goto LABEL_62;
            if ( !(unsigned __int8)sub_3870AF0(v96, v158) )
              goto LABEL_62;
            v159 = (__int64)*v48;
            v103 = sub_146CEE0((__int64)*v48, v94, (__int64)v48[2]);
            v52 = v159;
            if ( !v103 )
              goto LABEL_62;
            if ( !(unsigned __int8)sub_3870AF0(v94, v159) )
              goto LABEL_62;
            v160 = (__int64)*v48;
            v104 = sub_146CEE0((__int64)*v48, (__int64)v189, (__int64)v48[2]);
            v52 = v160;
            if ( !v104 || !(unsigned __int8)sub_3870AF0(v189, v160) )
              goto LABEL_62;
            v161 = sub_15FF4C0(v178);
            v179 = sub_157EBA0((__int64)v48[4]);
            v105 = sub_19817B0((__int64 *)v48, (__int64)&v229, (__int64)&v215, v161, v94, (__int64)v189, v179);
            v106 = sub_19817B0((__int64 *)v48, (__int64)&v229, (__int64)&v215, v174, v98, v96, v179);
            v107 = v105;
            LOWORD(v204[0]) = 257;
            v108 = v106;
          }
          else
          {
            v111 = v200;
            v112 = v201;
            v175 = v200;
            v169 = sub_1456040(*(_QWORD *)v200[4]);
            v162 = (__int64)*v48;
            v113 = *(_QWORD *)v111[4];
            v114 = sub_146CEE0((__int64)*v48, v113, (__int64)v48[2]);
            v52 = v162;
            if ( !v114 )
              goto LABEL_62;
            if ( !(unsigned __int8)sub_3870AF0(v113, v162) )
              goto LABEL_62;
            v163 = (__int64)*v48;
            v115 = sub_146CEE0((__int64)*v48, v112, (__int64)v48[2]);
            v52 = v163;
            if ( !v115 )
              goto LABEL_62;
            if ( !(unsigned __int8)sub_3870AF0(v112, v163) )
              goto LABEL_62;
            v164 = (__int64)*v48;
            v116 = sub_146CEE0((__int64)*v48, v94, (__int64)v48[2]);
            v52 = v164;
            if ( !v116
              || !(unsigned __int8)sub_3870AF0(v94, v164)
              || v175 != (_QWORD *)sub_1488A90(v186, (__int64)*v48, v24, a4) )
            {
              goto LABEL_62;
            }
            v190 = sub_157EBA0((__int64)v48[4]);
            v180 = sub_15FF4C0(v178);
            v117 = sub_19817B0((__int64 *)v48, (__int64)&v229, (__int64)&v215, 0x24u, v113, v112, v190);
            v118 = sub_145CF80((__int64)*v48, v169, 1, 0);
            v119 = sub_19817B0((__int64 *)v48, (__int64)&v229, (__int64)&v215, v180, v94, v118, v190);
            v108 = v117;
            LOWORD(v204[0]) = 257;
            v107 = v119;
          }
          v109 = sub_1281C00((__int64 *)&v215, v108, v107, (__int64)v203);
          v110 = (unsigned int)v213;
          if ( (unsigned int)v213 >= HIDWORD(v213) )
          {
            sub_16CD150((__int64)&v212, v214, 0, 8, v52, v53);
            v110 = (unsigned int)v213;
          }
          ++v197;
          v212[v110] = v109;
          v56 = v209;
          LODWORD(v213) = v213 + 1;
LABEL_65:
          if ( !v56 )
            goto LABEL_75;
LABEL_66:
          v58 = v208[v56 - 1];
          v60 = v225;
          v59 = v224;
        }
        v65 = 0;
        while ( *v59 != v58 )
        {
          if ( *v59 == -2 )
            v65 = v59;
          if ( v64 == ++v59 )
          {
            if ( !v65 )
              goto LABEL_132;
            *v65 = v58;
            --v227;
            ++v223;
            v62 = *(_BYTE *)(v58 + 16);
            if ( v62 != 50 )
              goto LABEL_60;
            goto LABEL_98;
          }
        }
LABEL_74:
        v56 = v209;
        if ( v209 )
          goto LABEL_66;
LABEL_75:
        if ( !v197 )
          goto LABEL_106;
        v66 = *(unsigned __int8 **)(v194 + 48);
        v216 = *(_QWORD *)(v194 + 40);
        v217 = v194 + 24;
        v203[0] = v66;
        if ( v66 )
        {
          sub_1623A60((__int64)v203, (__int64)v66, 2);
          v67 = v215;
          if ( v215 )
            goto LABEL_78;
LABEL_79:
          v215 = v203[0];
          if ( v203[0] )
            sub_1623210((__int64)v203, v203[0], (__int64)&v215);
        }
        else
        {
          v67 = v215;
          if ( v215 )
          {
LABEL_78:
            sub_161E7C0((__int64)&v215, (__int64)v67);
            goto LABEL_79;
          }
        }
        v68 = v212;
        v69 = &v212[(unsigned int)v213];
        if ( v212 != v69 )
        {
          v70 = *v212;
          for ( j = v212 + 1; v69 != j; ++j )
          {
            v72 = *j;
            if ( v70 )
            {
              LOWORD(v204[0]) = 257;
              v70 = sub_1281C00((__int64 *)&v215, v70, v72, (__int64)v203);
            }
            else
            {
              v70 = *j;
            }
          }
          v86 = (__int64 *)(v194 - 24LL * (*(_DWORD *)(v194 + 20) & 0xFFFFFFF));
          if ( *v86 )
            goto LABEL_138;
LABEL_140:
          *v86 = v70;
          if ( v70 )
          {
            v89 = *(_QWORD *)(v70 + 8);
            v86[1] = v89;
            if ( v89 )
              *(_QWORD *)(v89 + 16) = (unsigned __int64)(v86 + 1) | *(_QWORD *)(v89 + 16) & 3LL;
            v68 = v212;
            v86[2] = v86[2] & 3 | (v70 + 8);
            *(_QWORD *)(v70 + 8) = v86;
            v196 = v195;
            goto LABEL_107;
          }
          v196 = v195;
LABEL_106:
          v68 = v212;
          goto LABEL_107;
        }
        v70 = 0;
        v86 = (__int64 *)(v194 - 24LL * (*(_DWORD *)(v194 + 20) & 0xFFFFFFF));
        if ( *v86 )
        {
LABEL_138:
          v87 = v86[1];
          v88 = v86[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v88 = v87;
          if ( v87 )
            *(_QWORD *)(v87 + 16) = *(_QWORD *)(v87 + 16) & 3LL | v88;
          goto LABEL_140;
        }
        v196 = v195;
LABEL_107:
        if ( v68 != (__int64 *)v214 )
          _libc_free((unsigned __int64)v68);
        if ( v225 != v224 )
          _libc_free((unsigned __int64)v225);
        if ( v208 != v211 )
          _libc_free((unsigned __int64)v208);
        if ( v215 )
          sub_161E7C0((__int64)&v215, (__int64)v215);
        if ( v193 == ++v198 )
        {
          v14 = v196;
          if ( v265 != v267 )
            _libc_free((unsigned __int64)v265);
          if ( v260[0] )
            sub_161E7C0((__int64)v260, v260[0]);
LABEL_120:
          j___libc_free_0(v256);
          if ( v249 != v248 )
            _libc_free((unsigned __int64)v249);
          j___libc_free_0(v244);
          j___libc_free_0(v240);
          j___libc_free_0(v236);
          if ( v234 )
          {
            v83 = v232;
            v84 = &v232[5 * v234];
            do
            {
              if ( *v83 == -8 )
              {
                if ( v83[1] != -8 )
                  goto LABEL_125;
              }
              else if ( *v83 != -16 || v83[1] != -16 )
              {
LABEL_125:
                v85 = v83[4];
                if ( v85 != 0 && v85 != -8 && v85 != -16 )
                  sub_1649B30(v83 + 2);
              }
              v83 += 5;
            }
            while ( v84 != v83 );
          }
          j___libc_free_0(v232);
          v37 = v205;
LABEL_149:
          if ( v37 == (__int64 *)v207 )
            return v14;
LABEL_32:
          _libc_free((unsigned __int64)v37);
          return v14;
        }
      }
    }
    return 0;
  }
  v26 = *(_QWORD *)(a1 + 16);
  v229.m128i_i64[0] = (__int64)&v230;
  v229.m128i_i64[1] = 0x800000000LL;
  sub_13FA5B0(v26, (__int64)&v229);
  if ( v229.m128i_i32[2] == 1 )
  {
LABEL_33:
    v36 = (_QWORD *)v229.m128i_i64[0];
LABEL_34:
    if ( v36 != &v230 )
      _libc_free((unsigned __int64)v36);
    goto LABEL_36;
  }
  v27 = sub_13FCB50(*(_QWORD *)(a1 + 16));
  v28 = sub_157EBA0(v27);
  v29 = sub_15F4DF0(v28, 0);
  v30 = sub_1377370(*(_QWORD *)(a1 + 8), v27, **(_QWORD **)(*(_QWORD *)(a1 + 16) + 32LL) == v29);
  v24 = (__m128i)(unsigned int)dword_4FB0A80;
  a4 = (__m128i)0x3F800000u;
  v31 = v30;
  if ( *(float *)&dword_4FB0A80 < 1.0 )
  {
    v33 = 1;
    v32 = 1;
  }
  else
  {
    v32 = (int)*(float *)&dword_4FB0A80;
    v33 = (unsigned int)(int)*(float *)&dword_4FB0A80;
  }
  v34 = 0x80000000;
  if ( (unsigned __int64)v31 * v33 <= 0x80000000 )
    v34 = v32 * v31;
  v35 = (__int64 *)v229.m128i_i64[0];
  v36 = (_QWORD *)(v229.m128i_i64[0] + 16LL * v229.m128i_u32[2]);
  if ( (_QWORD *)v229.m128i_i64[0] == v36 )
    goto LABEL_34;
  while ( (unsigned int)sub_13774B0(*(_QWORD *)(a1 + 8), *v35, v35[1]) <= v34 )
  {
    v35 += 2;
    if ( v36 == v35 )
      goto LABEL_33;
  }
  v37 = (__int64 *)v229.m128i_i64[0];
  if ( (const char **)v229.m128i_i64[0] != &v230 )
    goto LABEL_32;
  return v14;
}
