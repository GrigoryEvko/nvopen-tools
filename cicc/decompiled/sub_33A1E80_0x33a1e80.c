// Function: sub_33A1E80
// Address: 0x33a1e80
//
void __fastcall sub_33A1E80(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned __int8 *v4; // rbx
  __int64 v5; // r14
  const __m128i *v6; // r12
  __int64 *v7; // rdi
  __int64 *v8; // rax
  __int64 v9; // rax
  __int64 (*v10)(); // r9
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int); // r14
  __int64 v12; // rax
  _DWORD *v13; // rax
  unsigned __int16 v14; // r8
  int v15; // eax
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r9
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 *v24; // rdx
  int v25; // edx
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r12
  int v30; // r12d
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r12
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdx
  bool v41; // zf
  __int64 v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // r9
  __int64 v46; // rdx
  __int64 v47; // r8
  unsigned __int64 v48; // r10
  __int64 *v49; // rdx
  __int64 v50; // r15
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r9
  __int64 v54; // rdx
  __int64 v55; // r8
  __int64 *v56; // rdx
  __int64 v57; // r12
  int v58; // eax
  __int64 v59; // r8
  __int64 v60; // r9
  int v61; // edx
  char v62; // r8
  int v63; // eax
  char v64; // dl
  int v65; // ecx
  int v66; // ecx
  int v67; // ecx
  int v68; // ecx
  int v69; // ecx
  __int64 v70; // rdx
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // r12
  __int64 v75; // r12
  __int64 v76; // r14
  __int64 v77; // r12
  __int64 v78; // rax
  unsigned int *v79; // rax
  __int64 v80; // rax
  __int64 v81; // rdi
  __int64 v82; // rdx
  int v83; // r9d
  __int64 v84; // rax
  unsigned int v85; // edx
  __int64 v86; // r10
  __int64 v87; // r8
  __int64 v88; // r9
  unsigned __int64 v89; // r11
  __int64 v90; // rax
  unsigned __int64 v91; // rdx
  __int64 *v92; // rax
  __int64 v93; // r14
  unsigned __int64 v94; // r9
  int v95; // edx
  unsigned __int64 v96; // r8
  __int64 v97; // rcx
  __int64 v98; // rax
  __int64 v99; // rsi
  __m128i v100; // xmm0
  int v101; // r15d
  __int16 v102; // kr00_2
  int v103; // eax
  __int64 v104; // r9
  unsigned __int64 v105; // r14
  unsigned int v106; // edx
  unsigned int v107; // r15d
  unsigned __int64 v108; // rax
  __int64 v109; // rdx
  unsigned __int64 *v110; // rdx
  int v111; // eax
  unsigned int v112; // edx
  unsigned int v113; // eax
  __int64 v114; // r8
  _QWORD *v115; // rax
  __int64 v116; // rax
  __int64 v117; // rdx
  __int64 v118; // r9
  __int64 v119; // rax
  unsigned __int64 v120; // r14
  __int64 v121; // r15
  __int64 v122; // rsi
  __int64 v123; // rax
  unsigned int v124; // edx
  __int64 v125; // rax
  unsigned __int64 v126; // rdx
  _QWORD *v127; // rax
  __int64 v128; // rdx
  __int64 v129; // rcx
  __int64 v130; // r8
  __int64 v131; // r9
  __int64 v132; // rax
  __int64 v133; // rdx
  __int64 v134; // r15
  unsigned __int64 v135; // rdx
  __int64 v136; // rax
  unsigned int v137; // eax
  int v138; // edx
  __int64 v139; // rax
  __int64 v140; // r9
  __int64 v141; // rsi
  unsigned int v142; // edx
  __int64 v143; // rdx
  unsigned __int64 v144; // r14
  __int64 v145; // r15
  __int64 v146; // rsi
  unsigned int v147; // edx
  __int64 v148; // rsi
  __int128 v149; // [rsp-10h] [rbp-270h]
  __int64 v150; // [rsp-10h] [rbp-270h]
  __int128 v151; // [rsp-10h] [rbp-270h]
  __int128 v152; // [rsp-10h] [rbp-270h]
  __int128 v153; // [rsp-10h] [rbp-270h]
  __int64 v154; // [rsp-8h] [rbp-268h]
  int v155; // [rsp+4h] [rbp-25Ch]
  __int64 v156; // [rsp+8h] [rbp-258h]
  __int64 v157; // [rsp+8h] [rbp-258h]
  unsigned __int64 v158; // [rsp+10h] [rbp-250h]
  __int64 v159; // [rsp+10h] [rbp-250h]
  __int64 v160; // [rsp+18h] [rbp-248h]
  __int64 v161; // [rsp+20h] [rbp-240h]
  __int64 v162; // [rsp+20h] [rbp-240h]
  __int64 v163; // [rsp+20h] [rbp-240h]
  __int64 v164; // [rsp+28h] [rbp-238h]
  unsigned __int64 v165; // [rsp+28h] [rbp-238h]
  unsigned __int64 v166; // [rsp+28h] [rbp-238h]
  int v167; // [rsp+30h] [rbp-230h]
  char v168; // [rsp+38h] [rbp-228h]
  unsigned __int64 v169; // [rsp+38h] [rbp-228h]
  bool v170; // [rsp+40h] [rbp-220h]
  int v171; // [rsp+40h] [rbp-220h]
  char v172; // [rsp+46h] [rbp-21Ah]
  bool v173; // [rsp+47h] [rbp-219h]
  int v174; // [rsp+48h] [rbp-218h]
  unsigned __int8 v175; // [rsp+48h] [rbp-218h]
  unsigned __int16 v176; // [rsp+58h] [rbp-208h]
  __int64 v177; // [rsp+58h] [rbp-208h]
  int v178; // [rsp+58h] [rbp-208h]
  __int64 v179; // [rsp+58h] [rbp-208h]
  __int64 v180; // [rsp+58h] [rbp-208h]
  unsigned __int64 v181; // [rsp+58h] [rbp-208h]
  __int64 v182; // [rsp+60h] [rbp-200h]
  unsigned __int64 v183; // [rsp+60h] [rbp-200h]
  unsigned __int64 v184; // [rsp+60h] [rbp-200h]
  unsigned __int64 v185; // [rsp+60h] [rbp-200h]
  int v186; // [rsp+60h] [rbp-200h]
  int v187; // [rsp+60h] [rbp-200h]
  int v188; // [rsp+60h] [rbp-200h]
  __int64 *v190; // [rsp+70h] [rbp-1F0h]
  __int64 v191; // [rsp+70h] [rbp-1F0h]
  __int64 v192; // [rsp+70h] [rbp-1F0h]
  __int64 v193; // [rsp+70h] [rbp-1F0h]
  __int64 v194; // [rsp+78h] [rbp-1E8h]
  __int64 v195; // [rsp+78h] [rbp-1E8h]
  __int64 v196; // [rsp+78h] [rbp-1E8h]
  __int64 v197; // [rsp+A0h] [rbp-1C0h] BYREF
  int v198; // [rsp+A8h] [rbp-1B8h]
  __int64 v199; // [rsp+B0h] [rbp-1B0h] BYREF
  int v200; // [rsp+B8h] [rbp-1A8h]
  __int64 v201; // [rsp+C0h] [rbp-1A0h]
  unsigned __int64 v202; // [rsp+D0h] [rbp-190h]
  __int64 v203; // [rsp+D8h] [rbp-188h]
  __int64 v204; // [rsp+E0h] [rbp-180h]
  unsigned __int8 *v205; // [rsp+F0h] [rbp-170h] BYREF
  __int64 v206; // [rsp+F8h] [rbp-168h]
  int v207; // [rsp+110h] [rbp-150h] BYREF
  __m128i v208; // [rsp+118h] [rbp-148h] BYREF
  unsigned __int64 v209; // [rsp+128h] [rbp-138h]
  int v210; // [rsp+130h] [rbp-130h]
  char v211; // [rsp+134h] [rbp-12Ch]
  int v212; // [rsp+138h] [rbp-128h]
  unsigned __int64 v213; // [rsp+140h] [rbp-120h]
  int v214; // [rsp+148h] [rbp-118h]
  _BYTE *v215; // [rsp+150h] [rbp-110h] BYREF
  __int64 v216; // [rsp+158h] [rbp-108h]
  _BYTE v217[64]; // [rsp+160h] [rbp-100h] BYREF
  _OWORD *v218; // [rsp+1A0h] [rbp-C0h] BYREF
  __int64 v219; // [rsp+1A8h] [rbp-B8h]
  _OWORD v220[11]; // [rsp+1B0h] [rbp-B0h] BYREF

  v4 = (unsigned __int8 *)a2;
  v5 = *(_QWORD *)(a2 - 32);
  if ( v5 )
  {
    if ( *(_BYTE *)v5 )
    {
      v5 = 0;
    }
    else if ( *(_QWORD *)(a2 + 80) != *(_QWORD *)(v5 + 24) )
    {
      v5 = 0;
    }
  }
  v170 = sub_B2DCC0(v5);
  v173 = !v170;
  if ( v170 )
  {
    v172 = 0;
    v6 = *(const __m128i **)(a1 + 864);
    v218 = v220;
    v219 = 0x800000000LL;
    goto LABEL_7;
  }
  if ( !(unsigned __int8)sub_B2DCE0(v5) || (a2 = 76, !(unsigned __int8)sub_B2D610(v5, 76)) )
  {
    v218 = v220;
LABEL_125:
    v219 = 0x800000000LL;
    v132 = sub_33738B0(a1, a2, v128, v129, v130, v131);
    v134 = v133;
    v135 = (unsigned __int64)v218;
    v172 = 0;
    *(_QWORD *)v218 = v132;
    *(_QWORD *)(v135 + 8) = v134;
    v6 = *(const __m128i **)(a1 + 864);
    LODWORD(v219) = v219 + 1;
    goto LABEL_7;
  }
  a2 = 41;
  v172 = sub_B2D610(v5, 41);
  v218 = v220;
  if ( !v172 )
    goto LABEL_125;
  v6 = *(const __m128i **)(a1 + 864);
  v219 = 0x800000001LL;
  v220[0] = _mm_loadu_si128(v6 + 24);
LABEL_7:
  v212 = 0;
  v213 = 0;
  v214 = 256;
  v7 = (__int64 *)v6[2].m128i_i64[1];
  v208.m128i_i64[1] = 0;
  v209 = 0;
  v211 = 0;
  v8 = (__int64 *)v6[1].m128i_i64[0];
  v207 = 0;
  v208.m128i_i16[0] = 0;
  v182 = (__int64)v8;
  v9 = *v8;
  v168 = 0;
  v10 = *(__int64 (**)())(v9 + 608);
  if ( v10 != sub_2FE3160 )
  {
    v168 = ((__int64 (__fastcall *)(__int64, int *, unsigned __int8 *, __int64 *, _QWORD))v10)(v182, &v207, v4, v7, a3);
    if ( v168 && (unsigned int)(v207 - 47) > 1 )
      goto LABEL_25;
    v6 = *(const __m128i **)(a1 + 864);
    v7 = (__int64 *)v6[2].m128i_i64[1];
    v9 = *(_QWORD *)v182;
  }
  v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(v9 + 32);
  v12 = sub_2E79000(v7);
  if ( v11 == sub_2D42F30 )
  {
    v13 = sub_AE2980(v12, 0);
    v14 = 2;
    v15 = v13[1];
    if ( v15 != 1 )
    {
      v14 = 3;
      if ( v15 != 2 )
      {
        v14 = 4;
        if ( v15 != 4 )
        {
          v14 = 5;
          if ( v15 != 8 )
          {
            v14 = 6;
            if ( v15 != 16 )
            {
              v14 = 7;
              if ( v15 != 32 )
              {
                v14 = 8;
                if ( v15 != 64 )
                  v14 = 9 * (v15 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v14 = v11(v182, v12, 0);
  }
  v16 = *(_DWORD *)(a1 + 848);
  v17 = *(_QWORD *)a1;
  v215 = 0;
  LODWORD(v216) = v16;
  if ( v17 )
  {
    if ( &v215 != (_BYTE **)(v17 + 48) )
    {
      v18 = *(_QWORD *)(v17 + 48);
      v215 = (_BYTE *)v18;
      if ( v18 )
      {
        v176 = v14;
        sub_B96E90((__int64)&v215, v18, 1);
        v14 = v176;
      }
    }
  }
  v19 = sub_3400BD0((_DWORD)v6, a3, (unsigned int)&v215, v14, 0, 1, 0);
  v21 = v20;
  v22 = (unsigned int)v219;
  v23 = v19;
  if ( (unsigned __int64)(unsigned int)v219 + 1 > HIDWORD(v219) )
  {
    v193 = v19;
    v196 = v21;
    sub_C8D5F0((__int64)&v218, v220, (unsigned int)v219 + 1LL, 0x10u, v19, v21);
    v22 = (unsigned int)v219;
    v23 = v193;
    v21 = v196;
  }
  v24 = (__int64 *)&v218[v22];
  *v24 = v23;
  v24[1] = v21;
  LODWORD(v219) = v219 + 1;
  if ( v215 )
    sub_B91220((__int64)&v215, (__int64)v215);
LABEL_25:
  v25 = *v4;
  if ( v25 == 40 )
  {
    v26 = 32LL * (unsigned int)sub_B491D0((__int64)v4);
  }
  else
  {
    v26 = 0;
    if ( v25 != 85 )
    {
      v26 = 64;
      if ( v25 != 34 )
        BUG();
    }
  }
  if ( (v4[7] & 0x80u) == 0 )
    goto LABEL_34;
  v27 = sub_BD2BC0((__int64)v4);
  v29 = v27 + v28;
  if ( (v4[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v29 >> 4) )
LABEL_168:
      BUG();
LABEL_34:
    v33 = 0;
    goto LABEL_35;
  }
  if ( !(unsigned int)((v29 - sub_BD2BC0((__int64)v4)) >> 4) )
    goto LABEL_34;
  if ( (v4[7] & 0x80u) == 0 )
    goto LABEL_168;
  v30 = *(_DWORD *)(sub_BD2BC0((__int64)v4) + 8);
  if ( (v4[7] & 0x80u) == 0 )
    BUG();
  v31 = sub_BD2BC0((__int64)v4);
  v33 = 32LL * (unsigned int)(*(_DWORD *)(v31 + v32 - 4) - v30);
LABEL_35:
  v34 = (32LL * (*((_DWORD *)v4 + 1) & 0x7FFFFFF) - 32 - v26 - v33) >> 5;
  if ( (_DWORD)v34 )
  {
    v35 = (unsigned int)v34;
    v36 = 0;
    v37 = *((_DWORD *)v4 + 1) & 0x7FFFFFF;
    v177 = v35;
    while ( 1 )
    {
      v50 = *(_QWORD *)&v4[32 * (v36 - v37)];
      if ( !(unsigned __int8)sub_B49B80((__int64)v4, v36, 14) )
      {
        v51 = sub_338B750(a1, v50);
        v53 = v52;
        v54 = (unsigned int)v219;
        v55 = v51;
        if ( (unsigned __int64)(unsigned int)v219 + 1 > HIDWORD(v219) )
        {
          v192 = v51;
          v195 = v53;
          sub_C8D5F0((__int64)&v218, v220, (unsigned int)v219 + 1LL, 0x10u, v51, v53);
          v54 = (unsigned int)v219;
          v55 = v192;
          v53 = v195;
        }
        v56 = (__int64 *)&v218[v54];
        ++v36;
        *v56 = v55;
        v56[1] = v53;
        LODWORD(v219) = v219 + 1;
        if ( v36 == v177 )
          break;
        goto LABEL_42;
      }
      v190 = *(__int64 **)(v50 + 8);
      v38 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
      v39 = sub_2D5BAE0(v182, v38, v190, 1);
      v41 = *(_BYTE *)v50 == 17;
      v42 = *(_QWORD *)(a1 + 864);
      v215 = 0;
      LODWORD(v216) = 0;
      if ( v41 )
      {
        v43 = sub_33FF780(v42, v50, (unsigned int)&v215, v39, v40, 1, 0);
        v45 = v44;
        v46 = (unsigned int)v219;
        v47 = v43;
        v48 = (unsigned int)v219 + 1LL;
        if ( v48 <= HIDWORD(v219) )
          goto LABEL_39;
      }
      else
      {
        v116 = sub_33FE020(v42, v50, &v215, v39, v40, 1);
        v45 = v117;
        v46 = (unsigned int)v219;
        v47 = v116;
        v48 = (unsigned int)v219 + 1LL;
        if ( v48 <= HIDWORD(v219) )
          goto LABEL_39;
      }
      v191 = v47;
      v194 = v45;
      sub_C8D5F0((__int64)&v218, v220, v48, 0x10u, v47, v45);
      v46 = (unsigned int)v219;
      v47 = v191;
      v45 = v194;
LABEL_39:
      v49 = (__int64 *)&v218[v46];
      *v49 = v47;
      v49[1] = v45;
      LODWORD(v219) = v219 + 1;
      if ( v215 )
        sub_B91220((__int64)&v215, (__int64)v215);
      if ( ++v36 == v177 )
        break;
LABEL_42:
      v37 = *((_DWORD *)v4 + 1) & 0x7FFFFFF;
    }
  }
  v57 = *((_QWORD *)v4 + 1);
  v215 = v217;
  v216 = 0x400000000LL;
  v58 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
  LOBYTE(v206) = 0;
  *((_QWORD *)&v149 + 1) = v206;
  v205 = 0;
  *(_QWORD *)&v149 = 0;
  sub_34B8C80(v182, v58, v57, (unsigned int)&v215, 0, 0, v149);
  if ( v173 )
  {
    v125 = (unsigned int)v216;
    v126 = (unsigned int)v216 + 1LL;
    if ( v126 > HIDWORD(v216) )
    {
      sub_C8D5F0((__int64)&v215, v217, v126, 0x10u, v59, v60);
      v125 = (unsigned int)v216;
    }
    v127 = &v215[16 * v125];
    *v127 = 1;
    v127[1] = 0;
    LODWORD(v216) = v216 + 1;
  }
  v178 = sub_33E5830(*(_QWORD *)(a1 + 864), v215);
  v167 = v61;
  v62 = sub_920620((__int64)v4);
  v63 = 0;
  if ( v62 )
  {
    v64 = v4[1] >> 1;
    v63 = (16 * v64) & 0x20;
    if ( (v64 & 4) != 0 )
      v63 = (16 * v64) & 0x20 | 0x40;
    v65 = v63;
    if ( (v64 & 8) != 0 )
    {
      LOBYTE(v65) = v63 | 0x80;
      v63 = v65;
    }
    v66 = v63;
    if ( (v64 & 0x10) != 0 )
    {
      BYTE1(v66) = BYTE1(v63) | 1;
      v63 = v66;
    }
    v67 = v63;
    if ( (v64 & 0x20) != 0 )
    {
      BYTE1(v67) = BYTE1(v63) | 2;
      v63 = v67;
    }
    v68 = v63;
    if ( (v64 & 0x40) != 0 )
    {
      BYTE1(v68) = BYTE1(v63) | 4;
      v63 = v68;
    }
    v69 = v63;
    if ( (v4[1] & 2) != 0 )
    {
      BYTE1(v69) = BYTE1(v63) | 8;
      v63 = v69;
    }
  }
  v70 = *(_QWORD *)(a1 + 864);
  v200 = v63;
  v71 = *(_QWORD *)(v70 + 1024);
  v199 = v70;
  v201 = v71;
  *(_QWORD *)(v70 + 1024) = &v199;
  if ( (v4[7] & 0x80u) != 0 )
  {
    v72 = sub_BD2BC0((__int64)v4);
    v74 = v72 + v73;
    if ( (v4[7] & 0x80u) != 0 )
      v74 -= sub_BD2BC0((__int64)v4);
    v75 = v74 >> 4;
    if ( (_DWORD)v75 )
    {
      v76 = 0;
      v77 = 16LL * (unsigned int)v75;
      while ( 1 )
      {
        v78 = 0;
        if ( (v4[7] & 0x80u) != 0 )
          v78 = sub_BD2BC0((__int64)v4);
        v79 = (unsigned int *)(v76 + v78);
        if ( *(_DWORD *)(*(_QWORD *)v79 + 8LL) == 9 )
          break;
        v76 += 16;
        if ( v76 == v77 )
          goto LABEL_75;
      }
      v80 = sub_338B750(a1, *(_QWORD *)&v4[32 * (v79[2] - (unsigned __int64)(*((_DWORD *)v4 + 1) & 0x7FFFFFF))]);
      v81 = *(_QWORD *)(a1 + 864);
      v154 = v82;
      v150 = v80;
      v164 = v82;
      v205 = 0;
      LODWORD(v206) = 0;
      v84 = sub_33FAF80(v81, 496, (unsigned int)&v205, 262, 0, v83);
      v86 = v84;
      v87 = v150;
      v88 = v154;
      v89 = v85 | v164 & 0xFFFFFFFF00000000LL;
      if ( v205 )
      {
        v161 = v84;
        v165 = v85 | v164 & 0xFFFFFFFF00000000LL;
        sub_B91220((__int64)&v205, (__int64)v205);
        v86 = v161;
        v89 = v165;
      }
      v90 = (unsigned int)v219;
      v91 = (unsigned int)v219 + 1LL;
      if ( v91 > HIDWORD(v219) )
      {
        v163 = v86;
        v166 = v89;
        sub_C8D5F0((__int64)&v218, v220, v91, 0x10u, v87, v88);
        v90 = (unsigned int)v219;
        v86 = v163;
        v89 = v166;
      }
      v92 = (__int64 *)&v218[v90];
      *v92 = v86;
      v92[1] = v89;
      LODWORD(v219) = v219 + 1;
    }
  }
LABEL_75:
  (*(void (__fastcall **)(__int64, unsigned __int8 *, _OWORD **, _QWORD))(*(_QWORD *)v182 + 2536LL))(
    v182,
    v4,
    &v218,
    *(_QWORD *)(a1 + 864));
  if ( v168 )
  {
    v169 = v209 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v209 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v162 = v212;
      if ( (v209 & 4) != 0 )
      {
        v155 = *(_DWORD *)(v169 + 12);
      }
      else
      {
        v143 = *(_QWORD *)(v169 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v143 + 8) - 17 <= 1 )
          v143 = **(_QWORD **)(v143 + 16);
        v155 = *(_DWORD *)(v143 + 8) >> 8;
      }
      v169 = v209;
    }
    else if ( v211 )
    {
      v162 = 0;
      v155 = v210;
    }
    else
    {
      v155 = 0;
      v162 = 0;
    }
    v93 = *(_QWORD *)(a1 + 864);
    sub_B91FC0((__int64 *)&v205, (__int64)v4);
    v94 = v213;
    v95 = *(_DWORD *)(a1 + 848);
    v96 = (unsigned __int64)v218;
    v97 = (unsigned int)v219;
    v197 = 0;
    v198 = v95;
    if ( v213 > 0x3FFFFFFFFFFFFFFBLL )
      v94 = 0xBFFFFFFFFFFFFFFELL;
    v171 = HIWORD(v214);
    v98 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      if ( &v197 != (__int64 *)(v98 + 48) )
      {
        v99 = *(_QWORD *)(v98 + 48);
        v197 = v99;
        if ( v99 )
        {
          v156 = (unsigned int)v219;
          v158 = (unsigned __int64)v218;
          v183 = v94;
          sub_B96E90((__int64)&v197, v99, 1);
          v97 = v156;
          v96 = v158;
          v94 = v183;
        }
      }
    }
    v100 = _mm_loadu_si128(&v208);
    v159 = v96;
    v160 = v97;
    v202 = v169;
    v203 = v162;
    v101 = v207;
    v157 = v94;
    LODWORD(v204) = v155;
    v102 = v214;
    BYTE4(v204) = 0;
    v103 = sub_33CC4A0(v93, v100.m128i_i64[0], v100.m128i_i64[1]);
    if ( HIBYTE(v102) )
      v103 = (unsigned __int8)v102;
    v105 = sub_33EB1C0(
             v93,
             v101,
             (unsigned int)&v197,
             v178,
             v167,
             v103,
             v159,
             v160,
             v100.m128i_i64[0],
             v100.m128i_i64[1],
             v202,
             v203,
             v204,
             v171,
             v157,
             (__int64)&v205);
    v107 = v106;
    v184 = v106;
    if ( v197 )
      sub_B91220((__int64)&v197, v197);
    if ( v173 )
      goto LABEL_90;
  }
  else
  {
    v118 = *(_QWORD *)(a1 + 864);
    v119 = *(_QWORD *)a1;
    if ( !v170 )
    {
      v120 = (unsigned __int64)v218;
      v121 = (unsigned int)v219;
      v41 = *(_BYTE *)(*((_QWORD *)v4 + 1) + 8LL) == 7;
      LODWORD(v206) = *(_DWORD *)(a1 + 848);
      v205 = 0;
      if ( v41 )
      {
        if ( v119 )
        {
          if ( &v205 != (unsigned __int8 **)(v119 + 48) )
          {
            v148 = *(_QWORD *)(v119 + 48);
            v205 = (unsigned __int8 *)v148;
            if ( v148 )
            {
              v188 = v118;
              sub_B96E90((__int64)&v205, v148, 1);
              LODWORD(v118) = v188;
            }
          }
        }
        *((_QWORD *)&v153 + 1) = v121;
        *(_QWORD *)&v153 = v120;
        v123 = sub_3411630(v118, 48, (unsigned int)&v205, v178, v167, v118, v153);
      }
      else
      {
        if ( v119 )
        {
          if ( &v205 != (unsigned __int8 **)(v119 + 48) )
          {
            v122 = *(_QWORD *)(v119 + 48);
            v205 = (unsigned __int8 *)v122;
            if ( v122 )
            {
              v186 = v118;
              sub_B96E90((__int64)&v205, v122, 1);
              LODWORD(v118) = v186;
            }
          }
        }
        *((_QWORD *)&v151 + 1) = v121;
        *(_QWORD *)&v151 = v120;
        v123 = sub_3411630(v118, 47, (unsigned int)&v205, v178, v167, v118, v151);
      }
      v105 = v123;
      v107 = v124;
      v184 = v124;
      if ( v205 )
      {
        sub_B91220((__int64)&v205, (__int64)v205);
        v108 = (unsigned int)(*(_DWORD *)(v105 + 68) - 1);
        if ( v172 )
          goto LABEL_91;
        goto LABEL_117;
      }
LABEL_90:
      v108 = (unsigned int)(*(_DWORD *)(v105 + 68) - 1);
      if ( v172 )
      {
LABEL_91:
        v109 = *(unsigned int *)(a1 + 136);
        if ( v109 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 140) )
        {
          v181 = v108;
          sub_C8D5F0(a1 + 128, (const void *)(a1 + 144), v109 + 1, 0x10u, v109 + 1, v104);
          v109 = *(unsigned int *)(a1 + 136);
          v108 = v181;
        }
        v110 = (unsigned __int64 *)(*(_QWORD *)(a1 + 128) + 16 * v109);
        *v110 = v105;
        v110[1] = v108;
        ++*(_DWORD *)(a1 + 136);
        goto LABEL_94;
      }
LABEL_117:
      v174 = v108;
      v179 = *(_QWORD *)(a1 + 864);
      nullsub_1875(v105, v179, 0);
      *(_QWORD *)(v179 + 384) = v105;
      *(_DWORD *)(v179 + 392) = v174;
      sub_33E2B60(v179, 0);
      goto LABEL_94;
    }
    LODWORD(v206) = *(_DWORD *)(a1 + 848);
    v144 = (unsigned __int64)v218;
    v145 = (unsigned int)v219;
    v205 = 0;
    if ( v119 )
    {
      if ( &v205 != (unsigned __int8 **)(v119 + 48) )
      {
        v146 = *(_QWORD *)(v119 + 48);
        v205 = (unsigned __int8 *)v146;
        if ( v146 )
        {
          v187 = v118;
          sub_B96E90((__int64)&v205, v146, 1);
          LODWORD(v118) = v187;
        }
      }
    }
    *((_QWORD *)&v152 + 1) = v145;
    *(_QWORD *)&v152 = v144;
    v105 = sub_3411630(v118, 46, (unsigned int)&v205, v178, v167, v118, v152);
    v107 = v147;
    v184 = v147;
    if ( v205 )
      sub_B91220((__int64)&v205, (__int64)v205);
  }
LABEL_94:
  v111 = *(unsigned __int8 *)(*((_QWORD *)v4 + 1) + 8LL);
  if ( (_BYTE)v111 == 7 )
    goto LABEL_99;
  if ( (unsigned int)(v111 - 17) > 1 )
  {
    v185 = v184 & 0xFFFFFFFF00000000LL | v107;
    v105 = sub_3375A10(a1, *(_QWORD *)(a1 + 864), v4, v105, v185);
    v107 = v112;
    v184 = v112 | v185 & 0xFFFFFFFF00000000LL;
  }
  LOWORD(v113) = sub_A74820((_QWORD *)v4 + 9);
  if ( BYTE1(v113) )
  {
    v114 = v113;
    if ( !(_BYTE)qword_50393C8 )
      goto LABEL_99;
  }
  else
  {
    v136 = *((_QWORD *)v4 - 4);
    if ( !v136 )
      goto LABEL_99;
    if ( *(_BYTE *)v136 )
      goto LABEL_99;
    if ( *(_QWORD *)(v136 + 24) != *((_QWORD *)v4 + 10) )
      goto LABEL_99;
    v205 = *(unsigned __int8 **)(v136 + 120);
    LOWORD(v137) = sub_A74820(&v205);
    if ( !(_BYTE)qword_50393C8 || !BYTE1(v137) )
      goto LABEL_99;
    v114 = v137;
  }
  v138 = *(_DWORD *)(a1 + 848);
  v139 = *(_QWORD *)a1;
  v205 = 0;
  v140 = *(_QWORD *)(a1 + 864);
  LODWORD(v206) = v138;
  if ( v139 )
  {
    if ( &v205 != (unsigned __int8 **)(v139 + 48) )
    {
      v141 = *(_QWORD *)(v139 + 48);
      v205 = (unsigned __int8 *)v141;
      if ( v141 )
      {
        v175 = v114;
        v180 = v140;
        sub_B96E90((__int64)&v205, v141, 1);
        v114 = v175;
        v140 = v180;
      }
    }
  }
  v105 = sub_33F3090(v140, &v205, v105, v184 & 0xFFFFFFFF00000000LL | v107, v114);
  v107 = v142;
  if ( v205 )
    sub_B91220((__int64)&v205, (__int64)v205);
LABEL_99:
  v205 = v4;
  v115 = sub_337DC20(a1 + 8, (__int64 *)&v205);
  *v115 = v105;
  *((_DWORD *)v115 + 2) = v107;
  *(_QWORD *)(v199 + 1024) = v201;
  if ( v215 != v217 )
    _libc_free((unsigned __int64)v215);
  if ( v218 != v220 )
    _libc_free((unsigned __int64)v218);
}
