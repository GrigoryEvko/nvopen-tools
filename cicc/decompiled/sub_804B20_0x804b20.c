// Function: sub_804B20
// Address: 0x804b20
//
_QWORD *__fastcall sub_804B20(__int64 *a1)
{
  __int64 v1; // r15
  __int64 v2; // r13
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 v5; // rdx
  _QWORD *v6; // rsi
  __int64 v7; // r14
  __int64 v8; // rax
  __m128i *v9; // rbx
  __int64 *v10; // r8
  char v11; // al
  __int64 v12; // rbx
  _BYTE *v13; // rax
  __m128i *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  _QWORD *v19; // rax
  __int64 *v20; // r13
  void *v21; // rax
  const __m128i *v22; // rsi
  __int64 v23; // r14
  _QWORD *v24; // rax
  __m128i *v26; // r14
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // rbx
  __int64 v30; // r15
  __int64 v31; // rdi
  __int64 k; // rax
  __int64 v33; // rsi
  __m128i *v34; // r13
  unsigned int v35; // esi
  __m128i *v36; // r15
  __m128i *v37; // r10
  _QWORD *v38; // rax
  _QWORD *v39; // r11
  __m128i *v40; // rax
  const __m128i *v41; // rsi
  _QWORD *v42; // rax
  __m128i *v43; // r14
  __int64 v44; // rbx
  __int64 v45; // r13
  __m128i *v46; // r12
  _QWORD *v47; // r12
  const __m128i *v48; // rsi
  __int64 v49; // r12
  __int64 v50; // rdi
  __int64 v51; // r15
  __int64 v52; // r13
  __int64 v53; // rax
  __m128i *v54; // r14
  char v55; // al
  __int64 v56; // rdx
  __int64 v57; // rbx
  int v58; // r11d
  __int64 v59; // r8
  __int64 v60; // rcx
  const __m128i *v61; // r14
  __int64 v62; // rax
  __int64 *v63; // rbx
  _QWORD *v64; // rax
  __int64 v65; // rsi
  _QWORD *v66; // rax
  __m128i *v67; // rbx
  __int64 v68; // rcx
  __int64 v69; // r8
  __int64 v70; // r14
  __int64 *v71; // r13
  const __m128i *v72; // rax
  _QWORD *m128i_i64; // r14
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 *v78; // r13
  __int8 v79; // cl
  __int64 v80; // rsi
  _QWORD *v81; // r14
  __int64 v82; // rax
  _QWORD *v83; // rax
  __int64 *v84; // r13
  __int64 v85; // r12
  __int64 v86; // rax
  __m128i *v87; // rax
  __int64 *v88; // rax
  _QWORD *v89; // r11
  __int64 *v90; // r14
  __m128i *v91; // rax
  __int64 *v92; // rax
  __int64 *v93; // rax
  __int64 *v94; // r12
  _QWORD *v95; // rax
  __int64 v96; // r13
  _BYTE *v97; // rax
  __int64 v98; // rsi
  int v99; // ebx
  __int64 *v100; // rax
  __int64 *v101; // r14
  __int64 v102; // rax
  __int64 v103; // rbx
  unsigned __int64 v104; // rax
  const __m128i *v105; // rax
  const __m128i *v106; // rax
  __m128i *v107; // r14
  _QWORD *v108; // rbx
  __int64 v109; // r14
  _BYTE *v110; // rsi
  _QWORD *v111; // rbx
  __int64 v112; // rdx
  __int64 v113; // rcx
  __int64 v114; // r8
  __int64 v115; // rax
  __int64 *v116; // rbx
  _QWORD *v117; // rax
  __int64 v118; // rsi
  _QWORD *v119; // rax
  _BYTE *v120; // rax
  __int64 v121; // rax
  __int64 v122; // r12
  __int64 v123; // rax
  __int64 i; // rax
  __m128i *v125; // r13
  _QWORD *v126; // rax
  __m128i *v127; // rax
  _QWORD *v128; // rbx
  _BYTE *v129; // rax
  __int64 v130; // rax
  __m128i *v131; // r13
  __int64 v132; // rax
  _BYTE *v133; // rax
  __int64 v134; // rax
  _QWORD *v135; // rax
  _QWORD *v136; // rax
  _QWORD *v137; // r14
  __int64 *v138; // rax
  _QWORD *v139; // rax
  const __m128i *v140; // rax
  __int64 *v141; // r13
  _QWORD *v142; // rbx
  __int64 v143; // rax
  __m128i *v144; // rax
  __int64 v145; // rax
  _QWORD *v146; // rbx
  __int64 v147; // rdx
  __int64 v148; // rcx
  __int64 v149; // r8
  __int64 v150; // rax
  __int64 *v151; // rax
  __int64 v152; // rsi
  _QWORD *v153; // rax
  __int64 v154; // rsi
  __int64 v155; // r12
  _QWORD *v156; // rax
  __int64 v157; // rsi
  _QWORD *v158; // rbx
  _QWORD *v159; // rax
  __int64 v160; // rax
  _QWORD *v161; // rax
  _BYTE *v162; // rax
  __int64 v163; // rcx
  __int64 v164; // r8
  __int64 v165; // r9
  __int64 v166; // rax
  __int64 *v167; // rbx
  _QWORD *v168; // rax
  __int64 v169; // rsi
  _QWORD *v170; // rax
  __m128i *v171; // rbx
  __int64 v172; // rcx
  __int64 v173; // r8
  __int64 *v174; // r14
  __int64 v175; // rax
  __int64 v176; // rax
  _BYTE *v177; // rax
  __int64 v178; // rax
  __int64 v179; // rax
  _QWORD *v180; // rax
  __int64 v181; // rax
  _QWORD *v182; // rax
  __int64 v183; // r9
  __int64 *v184; // rax
  __int64 *v185; // r12
  _BYTE *v186; // rax
  _QWORD *v187; // rax
  __int64 v188; // rbx
  __int64 v189; // r15
  const __m128i *v190; // rsi
  _QWORD *v191; // rbx
  _QWORD *v192; // rax
  __int64 *v193; // rbx
  _QWORD *v194; // rax
  __int64 v195; // r12
  _BYTE *v196; // rax
  __int64 v197; // rsi
  unsigned __int64 v198; // rdi
  __int64 *v199; // rax
  __int64 *v200; // rbx
  __int64 *v201; // rax
  __m128i *v202; // rax
  __int64 v203; // rax
  __int64 v204; // rdi
  __m128i *v205; // rax
  __int64 v206; // [rsp-10h] [rbp-1A0h]
  __int64 v207; // [rsp-8h] [rbp-198h]
  int v208; // [rsp+Ch] [rbp-184h]
  int v209; // [rsp+10h] [rbp-180h]
  __int64 v210; // [rsp+10h] [rbp-180h]
  __int64 v211; // [rsp+18h] [rbp-178h]
  __int64 v212; // [rsp+18h] [rbp-178h]
  __int64 v213; // [rsp+20h] [rbp-170h]
  const __m128i *v214; // [rsp+20h] [rbp-170h]
  const __m128i *v215; // [rsp+28h] [rbp-168h]
  _BYTE *v216; // [rsp+28h] [rbp-168h]
  int v217; // [rsp+28h] [rbp-168h]
  _BYTE *v218; // [rsp+30h] [rbp-160h]
  __int64 *v219; // [rsp+30h] [rbp-160h]
  __int64 v220; // [rsp+30h] [rbp-160h]
  _BOOL4 v221; // [rsp+30h] [rbp-160h]
  __int64 v222; // [rsp+30h] [rbp-160h]
  _QWORD *v223; // [rsp+38h] [rbp-158h]
  _BYTE *v224; // [rsp+40h] [rbp-150h]
  _QWORD *v225; // [rsp+48h] [rbp-148h]
  const __m128i *v226; // [rsp+50h] [rbp-140h]
  __int64 v227; // [rsp+58h] [rbp-138h]
  __m128i *v228; // [rsp+60h] [rbp-130h]
  __m128i *v229; // [rsp+60h] [rbp-130h]
  _BYTE *v230; // [rsp+68h] [rbp-128h]
  int v231; // [rsp+68h] [rbp-128h]
  bool v232; // [rsp+68h] [rbp-128h]
  _BOOL4 v233; // [rsp+68h] [rbp-128h]
  __int64 v234; // [rsp+68h] [rbp-128h]
  bool v235; // [rsp+70h] [rbp-120h]
  __int64 v236; // [rsp+70h] [rbp-120h]
  __m128i *v237; // [rsp+70h] [rbp-120h]
  _QWORD *v238; // [rsp+70h] [rbp-120h]
  _QWORD *v239; // [rsp+78h] [rbp-118h]
  __int64 v240; // [rsp+78h] [rbp-118h]
  __int64 j; // [rsp+78h] [rbp-118h]
  __int64 *v242; // [rsp+80h] [rbp-110h]
  __int64 *v243; // [rsp+80h] [rbp-110h]
  __m128i *v244; // [rsp+80h] [rbp-110h]
  __int64 v245; // [rsp+80h] [rbp-110h]
  __int64 v246; // [rsp+80h] [rbp-110h]
  __m128i *v247; // [rsp+80h] [rbp-110h]
  int v249; // [rsp+94h] [rbp-FCh] BYREF
  _QWORD *v250; // [rsp+98h] [rbp-F8h] BYREF
  const __m128i *v251; // [rsp+A0h] [rbp-F0h] BYREF
  __m128i *v252; // [rsp+A8h] [rbp-E8h] BYREF
  __m128i v253[2]; // [rsp+B0h] [rbp-E0h] BYREF
  __m128i *v254; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v255; // [rsp+D8h] [rbp-B8h]
  __m128i v256[2]; // [rsp+F0h] [rbp-A0h] BYREF
  __m128i v257; // [rsp+110h] [rbp-80h] BYREF
  __int64 v258; // [rsp+128h] [rbp-68h]
  _QWORD *v259; // [rsp+148h] [rbp-48h]

  v1 = a1[7];
  v2 = *(_QWORD *)(v1 + 32);
  if ( (*(_BYTE *)v1 & 1) != 0 )
  {
    v250 = 0;
    v251 = (const __m128i *)sub_724DC0();
    sub_7FAED0(v1);
    v3 = sub_691620(*(_QWORD *)(v1 + 8));
    v4 = sub_72D2E0(*(_QWORD **)(v1 + 8));
    if ( !(unsigned int)sub_8D3410(*(_QWORD *)(v1 + 8)) || !(unsigned int)sub_691630(v3, 1) )
    {
      v5 = *(_QWORD *)(v1 + 16);
      if ( !v5 )
        sub_721090();
      sub_7F1A60(*(__m128i **)(v1 + 24), *(_QWORD *)(v5 + 152), v5, 0, 0, 0, 0, 0);
      sub_7E1790((__int64)&v254);
      if ( v2 )
      {
        v6 = &v250;
        if ( !(unsigned int)sub_8D23E0(*(_QWORD *)(v1 + 8)) )
          v6 = 0;
        v242 = sub_7F6890(v1, v6, (int *)&v254);
        v7 = sub_7E7CB0(*v242);
        v8 = sub_7E2BE0(v7, (__int64)v242);
        sub_7E25D0(v8, (int *)&v254);
        v9 = (__m128i *)sub_73E830(v7);
        v9[1].m128i_i64[0] = *(_QWORD *)(v1 + 24);
        v239 = *(_QWORD **)(v1 + 40);
        if ( v239 )
          v239 = sub_7F6320(v1, v9, v4);
        v10 = sub_7F88E0(*(_QWORD *)(v1 + 16), v9);
        v11 = *(_BYTE *)(v2 + 48);
        if ( v11 )
        {
          v235 = 0;
          if ( HIDWORD(qword_4F0688C) )
            v235 = v11 == 5;
          v243 = v10;
          v12 = sub_7E7CB0(v4);
          v13 = sub_73E130(v243, v4);
          v230 = (_BYTE *)sub_7E2BE0(v12, (__int64)v13);
          sub_7E1790((__int64)v253);
          if ( !(unsigned int)sub_8D3410(*(_QWORD *)(v1 + 8)) || *(_BYTE *)(v2 + 48) != 1 )
            goto LABEL_14;
          for ( i = *(_QWORD *)(v1 + 8); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
            ;
          if ( !*(_QWORD *)(i + 128) )
          {
            v125 = (__m128i *)sub_73E830(v7);
            v126 = sub_73E830(v12);
            v14 = v125;
            sub_7FA9D0(v126, v125, v253[0].m128i_i32);
          }
          else
          {
LABEL_14:
            sub_7F90D0(v12, (__int64)&v257);
            v258 = *(_QWORD *)(v1 + 8);
            if ( (unsigned int)sub_8D23E0(v258) )
              v259 = v250;
            sub_7F73A0(v1, &v257, v253[0].m128i_i32);
            sub_7E1790((__int64)v256);
            sub_7FEC50(v2, &v257, 0, 0, 0, 0, v256, 0, 0);
            v14 = &v257;
            sub_7F98C0(v1, (__int64)&v257, (__int64)v239, 0, v256[0].m128i_i64[1], (__int64)v253);
          }
          if ( !HIDWORD(qword_4F0688C)
            || !v235
            || (v20 = (__int64 *)v253[0].m128i_i64[1], *(_BYTE *)(v253[0].m128i_i64[1] + 24) != 1)
            || *(_BYTE *)(v253[0].m128i_i64[1] + 56) != 105
            || (v123 = *(_QWORD *)(v253[0].m128i_i64[1] + 72), *(_BYTE *)(v123 + 24) != 20)
            || *(_BYTE *)(*(_QWORD *)(v123 + 56) + 174LL) != 1 )
          {
            v19 = sub_73E830(v12);
            v14 = v253;
            sub_7E25D0((__int64)v19, v253[0].m128i_i32);
            v20 = (__int64 *)v253[0].m128i_i64[1];
          }
          if ( (unsigned int)sub_7F6F10(v1, (__int64)v14, v15, v16, v17, v18) )
          {
            v21 = sub_7F0830(v230);
            v22 = v251;
            v23 = (__int64)v21;
            sub_72BB40(v4, v251);
            v24 = sub_73A720(v251, (__int64)v22);
            *(_QWORD *)(v23 + 16) = v20;
            v20[2] = (__int64)v24;
            v10 = (__int64 *)sub_73DBF0(0x67u, v4, v23);
          }
          else
          {
            v10 = (__int64 *)sub_73DF90((__int64)v230, v20);
          }
        }
      }
      else
      {
        v84 = sub_7F6890(v1, 0, (int *)&v254);
        v85 = sub_7E7CB0(*v84);
        v86 = sub_7E2BE0(v85, (__int64)v84);
        sub_7E25D0(v86, (int *)&v254);
        v87 = (__m128i *)sub_73E830(v85);
        v87[1].m128i_i64[0] = *(_QWORD *)(v1 + 24);
        v10 = sub_7F88E0(*(_QWORD *)(v1 + 16), v87);
      }
      if ( v255 )
        v10 = (__int64 *)sub_73DF90(v255, v10);
      sub_7E2300((__int64)a1, (__int64)v10, *a1);
      return sub_724E30((__int64)&v251);
    }
    v249 = 0;
    v49 = a1[7];
    v50 = *(_QWORD *)(v49 + 8);
    v236 = *(_QWORD *)(v49 + 32);
    v51 = v50;
    for ( j = *(_QWORD *)(v49 + 16); *(_BYTE *)(v51 + 140) == 12; v51 = *(_QWORD *)(v51 + 160) )
      ;
    v225 = (_QWORD *)sub_691620(v50);
    v52 = sub_72D2E0(v225);
    sub_7E1790((__int64)v253);
    sub_7E1790((__int64)&v254);
    v53 = *(_QWORD *)(v49 + 40);
    v231 = 0;
    v227 = v53;
    if ( v53 )
    {
      v227 = *(_QWORD *)(v53 + 16);
      v231 = sub_6013A0(v227, (__int64)&v249);
    }
    v54 = (__m128i *)sub_7F6890(v49, &v252, (int *)&v254);
    v54[1].m128i_i64[0] = *(_QWORD *)(v49 + 24);
    v55 = *(_BYTE *)v49;
    if ( (*(_BYTE *)v49 & 6) != 0 || v249 )
    {
      sub_7F1A60(v54, *(_QWORD *)(j + 152), j, 0, 0, 0, 0, 0);
      v103 = v54[1].m128i_i64[0];
      v104 = sub_7F5F50((__int64)v225, j);
      if ( v104 && (v105 = (const __m128i *)sub_73A830(v104, byte_4F06A51[0])) != 0 )
      {
        v54[1].m128i_i64[0] = (__int64)v105;
        v226 = v105;
        v106 = (const __m128i *)sub_73DBF0(0x27u, v54->m128i_i64[0], (__int64)v54);
        v106[1].m128i_i64[0] = v103;
        v107 = (__m128i *)v106;
        v223 = sub_7F6320(v49, v106, v52);
        v108 = sub_7F88E0(j, v107);
        v109 = sub_7E7CB0(v52);
        v110 = sub_73E130(v108, v52);
        v224 = (_BYTE *)sub_7E2BE0(v109, (__int64)v110);
        v111 = sub_73E830(v109);
        v115 = sub_7E1C30(v109, v110, v112, v113, v114);
        v116 = (__int64 *)sub_73E130(v111, v115);
        v117 = sub_7E8090(v226, 0);
        v118 = *v116;
        v116[2] = (__int64)v117;
        v119 = sub_73DBF0(0x32u, v118, (__int64)v116);
        v120 = sub_73E130(v119, v52);
        v121 = sub_7E2BE0(v109, (__int64)v120);
        sub_7E25D0(v121, v253[0].m128i_i32);
      }
      else
      {
        v223 = sub_7F6320(v49, v54, v52);
        v128 = sub_7F88E0(j, v54);
        v109 = sub_7E7CB0(v52);
        v129 = sub_73E130(v128, v52);
        v226 = 0;
        v224 = (_BYTE *)sub_7E2BE0(v109, (__int64)v129);
      }
      j = 0;
      v228 = (__m128i *)sub_73E830(v109);
    }
    else
    {
      if ( !v231 )
      {
        v224 = 0;
        if ( !v236 )
        {
          v228 = 0;
          v60 = 0;
          v59 = 0;
          v58 = 0;
          v223 = 0;
          goto LABEL_141;
        }
        v223 = 0;
        v226 = 0;
        v228 = 0;
        goto LABEL_60;
      }
      v223 = sub_7E8090(v54, 1u);
      v198 = sub_7F5F50((__int64)v225, j);
      if ( v198 )
      {
        v199 = sub_73A830(v198, byte_4F06A51[0]);
        v226 = (const __m128i *)v199;
        v200 = v199;
        if ( v199 )
        {
          v201 = (__int64 *)sub_73E130(v223, *v199);
          v201[2] = (__int64)v200;
          v224 = 0;
          v223 = sub_73DBF0(0x27u, *v201, (__int64)v201);
        }
        else
        {
          v224 = 0;
        }
        v228 = 0;
      }
      else
      {
        v226 = 0;
        v224 = 0;
        v228 = 0;
      }
    }
    v55 = *(_BYTE *)v49;
    if ( !v236 )
    {
      v231 = 0;
      v59 = 0;
      v60 = 0;
      v58 = 0;
      goto LABEL_64;
    }
LABEL_60:
    if ( *(_BYTE *)(v236 + 48) == 1 )
    {
      v231 = 0;
      v58 = 0;
      v59 = 0;
      v60 = 0;
    }
    else if ( (v55 & 0xC0) == 0
           && (v56 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v236 + 56) + 176LL) + 176LL),
               v57 = *(_QWORD *)(v56 + 176),
               *(_BYTE *)(v57 + 48) == 5) )
    {
      v233 = sub_7F9D00(*(_QWORD *)(v56 + 176));
      v202 = sub_7FDF40(*(_QWORD *)(v57 + 56), 1, 0);
      v203 = sub_7FE9A0((__int64)v202, *(__m128i **)(v57 + 64), 0);
      v204 = *(_QWORD *)(v57 + 40);
      v58 = v233;
      v60 = v203;
      if ( v204 )
      {
        v221 = v233;
        v234 = v203;
        sub_733650(v204);
        v58 = v221;
        v60 = v234;
      }
      v59 = *(_QWORD *)(v57 + 16);
      if ( v59 )
      {
        v217 = v58;
        v222 = v60;
        v205 = sub_7FDF40(*(_QWORD *)(v57 + 16), 1, 0);
        v60 = v222;
        v231 = 0;
        v59 = (__int64)v205;
        v58 = v217;
      }
      else
      {
        v231 = 0;
      }
      v55 = *(_BYTE *)v49;
    }
    else
    {
      v231 = 1;
      v58 = 0;
      v59 = 0;
      v60 = 0;
    }
LABEL_64:
    if ( (v55 & 6) != 0 || v249 )
    {
      v61 = v252;
      if ( v226 )
      {
        if ( dword_4F06880 )
        {
          v208 = v58;
          v210 = v59;
          v212 = v60;
          v158 = sub_7F5ED0(v52);
          v159 = sub_72BA30(byte_4F06A51[0]);
          v160 = sub_72D2E0(v159);
          v219 = (__int64 *)sub_73E130(v228, v160);
          v228 = (__m128i *)sub_7E8090(v228, 0);
          v219[2] = (__int64)sub_73A830(2, byte_4F06A51[0]);
          v161 = sub_73DBF0(0x33u, *v219, (__int64)v219);
          v162 = sub_73DCD0(v161);
          v220 = sub_698020(v162, 73, (__int64)v158, v163, v164, v165);
          v166 = sub_72D2E0(v226->m128i_i64[0]);
          v167 = (__int64 *)sub_73E130(v228, v166);
          v214 = (const __m128i *)sub_7E8090(v228, 0);
          v168 = sub_73A830(1, byte_4F06A51[0]);
          v169 = *v167;
          v167[2] = (__int64)v168;
          v170 = sub_73DBF0(0x33u, v169, (__int64)v167);
          v216 = sub_73DCD0(v170);
          v171 = (__m128i *)sub_7E8090(v61, 0);
          v174 = (__int64 *)sub_698020(v216, 73, (__int64)v61, v172, v173, (__int64)v216);
          v71 = (__int64 *)sub_7FC1E0(v214, v52, v171, v212, v210, 0, 0, v208);
          v70 = (__int64)sub_73DF90(v220, v174);
        }
        else
        {
          v209 = v58;
          v211 = v59;
          v213 = v60;
          v62 = sub_72D2E0(v226->m128i_i64[0]);
          v63 = (__int64 *)sub_73E130(v228, v62);
          v215 = (const __m128i *)sub_7E8090(v228, 0);
          v64 = sub_73A830(1, byte_4F06A51[0]);
          v65 = *v63;
          v63[2] = (__int64)v64;
          v66 = sub_73DBF0(0x33u, v65, (__int64)v63);
          v218 = sub_73DCD0(v66);
          v67 = (__m128i *)sub_7E8090(v61, 0);
          v70 = sub_698020(v218, 73, (__int64)v61, v68, v69, (__int64)v218);
          v71 = (__int64 *)sub_7FC1E0(v215, v52, v67, v213, v211, 0, 0, v209);
        }
        v72 = (const __m128i *)sub_73DF90(v70, v71);
        m128i_i64 = v72->m128i_i64;
        if ( !v227 )
        {
LABEL_70:
          if ( v231 )
          {
            v130 = sub_72D2E0((_QWORD *)v51);
            v131 = sub_7E7CA0(v130);
            v132 = sub_72D2E0((_QWORD *)v51);
            v133 = sub_73E130(m128i_i64, v132);
            v134 = sub_7E2BE0((__int64)v131, (__int64)v133);
            sub_7E25D0(v134, v253[0].m128i_i32);
            if ( !v236 || *(_BYTE *)(v236 + 48) != 1 )
            {
              sub_7F90D0((__int64)v131, (__int64)&v257);
              v258 = v51;
              if ( (unsigned int)sub_8D23E0(v51) )
                v259 = sub_7E8090(v252, 1u);
              sub_7F73A0(v49, &v257, v253[0].m128i_i32);
              sub_7E1790((__int64)v256);
              sub_7FEC50(v236, &v257, 0, 0, 0, 0, v256, 0, 0);
              sub_7F98C0(v49, (__int64)&v257, (__int64)v223, j, v256[0].m128i_i64[1], (__int64)v253);
              v135 = sub_73E830((__int64)v131);
              sub_7E25D0((__int64)v135, v253[0].m128i_i32);
              goto LABEL_74;
            }
            goto LABEL_133;
          }
          if ( v236 )
          {
            if ( *(_BYTE *)(v236 + 48) != 1 )
            {
              sub_7E25D0((__int64)m128i_i64, v253[0].m128i_i32);
              if ( *(_BYTE *)(v236 + 48) != 1 )
                goto LABEL_74;
              v131 = 0;
LABEL_133:
              v137 = 0;
              if ( !*(_QWORD *)(v51 + 128) )
              {
                v51 = (__int64)v225;
                v137 = sub_7E8090(v252, 1u);
              }
              v138 = sub_73E830((__int64)v131);
              sub_7FB7C0(v51, 1u, v138, v137, 0, 0, v253);
              v139 = sub_73E830((__int64)v131);
              sub_7E25D0((__int64)v139, v253[0].m128i_i32);
              v76 = v206;
              v77 = v207;
              goto LABEL_74;
            }
            v175 = sub_72D2E0((_QWORD *)v51);
            v131 = sub_7E7CA0(v175);
            v176 = sub_72D2E0((_QWORD *)v51);
            v177 = sub_73E130(m128i_i64, v176);
            v178 = sub_7E2BE0((__int64)v131, (__int64)v177);
            sub_7E25D0(v178, v253[0].m128i_i32);
            if ( *(_BYTE *)(v236 + 48) == 1 )
              goto LABEL_133;
          }
          else
          {
            sub_7E25D0((__int64)m128i_i64, v253[0].m128i_i32);
          }
LABEL_74:
          v78 = (__int64 *)v253[0].m128i_i64[1];
          if ( (*(_BYTE *)v49 & 6) != 0 || v249 )
          {
            if ( (unsigned int)sub_7F6F10(v49, (__int64)v253, v74, v75, v76, v77) )
            {
              v257.m128i_i64[0] = (__int64)sub_724DC0();
              v153 = sub_7F0830(v224);
              v153[2] = v78;
              v154 = v257.m128i_i64[0];
              v155 = (__int64)v153;
              sub_72BB40(*v78, (const __m128i *)v257.m128i_i64[0]);
              v156 = sub_73A720((const __m128i *)v257.m128i_i64[0], v154);
              v157 = *v78;
              v78[2] = (__int64)v156;
              v78 = (__int64 *)sub_73DBF0(0x67u, v157, v155);
              sub_724E30((__int64)&v257);
            }
            else if ( v224 )
            {
              v78 = (__int64 *)sub_73DF90((__int64)v224, v78);
            }
          }
          if ( v255 )
            v78 = (__int64 *)sub_73DF90(v255, v78);
          sub_7E2300((__int64)a1, (__int64)v78, *a1);
          return sub_724E30((__int64)&v251);
        }
        v141 = sub_7E88C0(v72);
        v146 = sub_7E8090(v228, 1u);
        v150 = sub_7E1C30(v228, 1, v147, v148, v149);
        v151 = (__int64 *)sub_73E130(v146, v150);
        v152 = *v151;
        v151[2] = (__int64)v226;
        v142 = sub_73DBF0(0x33u, v152, (__int64)v151);
      }
      else
      {
        v140 = (const __m128i *)sub_7FC1E0(v228, v52, v252, v60, v59, 0, 0, v58);
        m128i_i64 = v140->m128i_i64;
        if ( !v227 )
          goto LABEL_70;
        v141 = sub_7E88C0(v140);
        v142 = sub_7E8090(v228, 1u);
      }
      v143 = sub_7E1C10();
      v144 = (__m128i *)sub_73E130(v142, v143);
      v144[1].m128i_i64[0] = (__int64)v223;
      sub_7F88E0(v227, v144);
      v145 = sub_7DEB30((__int64)m128i_i64);
      m128i_i64 = sub_73DF90(v145, v141);
      sub_8255D0(v227);
      goto LABEL_70;
    }
LABEL_141:
    m128i_i64 = sub_7FC1E0(v228, v52, v252, v60, v59, j, v227, v58);
    goto LABEL_70;
  }
  v26 = *(__m128i **)(v1 + 24);
  v27 = *(_QWORD *)(v1 + 16);
  v28 = sub_691620(*(_QWORD *)(v1 + 8));
  v29 = v28;
  if ( (*(_BYTE *)v1 & 8) == 0 || !(unsigned int)sub_691630(v28, 0) )
  {
    if ( !v2 )
    {
      if ( !v27 )
        v27 = *(_QWORD *)(*(_QWORD *)(v29 + 168) + 184LL);
      sub_7EE560(v26, 0);
      v40 = (__m128i *)sub_7F6190(v27, *(_QWORD *)(v1 + 8), v26);
      v41 = (const __m128i *)sub_7F88E0(v27, v40);
      return (_QWORD *)sub_730620((__int64)a1, v41);
    }
    sub_7F3600(v26, 0, 0, 0);
    v26[1].m128i_i64[0] = 0;
    v30 = *(_QWORD *)(v2 + 16);
    v31 = *(_QWORD *)(v30 + 152);
    v240 = *(_QWORD *)(*(_QWORD *)(v30 + 40) + 32LL);
    for ( k = v31; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
      ;
    v33 = *(_QWORD *)(*(_QWORD *)(k + 168) + 40LL);
    if ( v33 )
      v33 = sub_8D71D0(v31);
    v34 = (__m128i *)sub_73E130(v26, v33);
    if ( (*(_BYTE *)(v30 + 192) & 2) != 0 && !v27 )
    {
      v36 = sub_7FDF40(v30, 3, 0);
      if ( (v36[12].m128i_i8[0] & 2) == 0 )
      {
        v37 = v34;
        v99 = 0;
        v34 = 0;
        goto LABEL_101;
      }
      v37 = (__m128i *)sub_7E8090(v34, 0);
      if ( !(_DWORD)qword_4F0688C )
      {
LABEL_100:
        v99 = 1;
LABEL_101:
        v100 = sub_7F87B0((__int64)v36, v37, 0, (v36[12].m128i_i8[0] & 2) != 0, 0);
        v101 = v100;
        if ( (v36[12].m128i_i8[0] & 2) != 0 )
          sub_7EBD50(v100);
        v102 = sub_72CBE0();
        v94 = (__int64 *)sub_73E130(v101, v102);
        goto LABEL_104;
      }
    }
    else
    {
      v35 = 0;
      v36 = sub_7FDF40(v30, 1, 0);
      if ( !(_DWORD)qword_4F0688C )
        v35 = v27 != 0;
      v37 = (__m128i *)sub_7E8090(v34, v35);
      if ( !(_DWORD)qword_4F0688C )
      {
        if ( v27 )
        {
          v244 = v37;
          v38 = sub_7E8090(v37, 1u);
          v37 = v244;
          v39 = v38;
LABEL_84:
          if ( (*(_BYTE *)(v27 + 89) & 4) != 0 )
          {
            v232 = 0;
            v79 = v36[12].m128i_i8[0];
            v246 = 0;
          }
          else
          {
            v79 = v36[12].m128i_i8[0];
            if ( (v79 & 2) != 0 )
            {
              if ( (_DWORD)qword_4F0688C )
              {
                v247 = v37;
                v136 = sub_7E8090(v34, 1u);
                v37 = v247;
                v39 = v136;
              }
              v229 = v37;
              v245 = (__int64)v39;
              v80 = sub_7E1C10();
              v81 = sub_73DBF0(0x12u, v80, v245);
              sub_7F08D0(v81, v80);
              v82 = sub_7E1C10();
              v237 = sub_7E7CA0(v82);
              v246 = sub_7E2BE0((__int64)v237, (__int64)v81);
              v83 = sub_73E830((__int64)v237);
              v79 = v36[12].m128i_i8[0];
              v37 = v229;
              v39 = v83;
              v232 = v246 != 0;
            }
            else
            {
              v232 = 0;
              v246 = 0;
            }
          }
          v238 = v39;
          v88 = sub_7F87B0((__int64)v36, v37, 0, (v79 & 2) != 0, 0);
          v89 = v238;
          v90 = v88;
          if ( (v36[12].m128i_i8[0] & 2) != 0 )
          {
            sub_7EBD50(v88);
            v89 = v238;
          }
          if ( !(_DWORD)qword_4F0688C || v232 )
          {
            v91 = (__m128i *)sub_7F6190(v27, v240, v89);
            v92 = sub_7F88E0(v27, v91);
            v93 = (__int64 *)sub_73DF90((__int64)v90, v92);
            v94 = v93;
            if ( v246 )
              v94 = (__int64 *)sub_73DF90(v246, v93);
LABEL_97:
            v95 = sub_7F0830(v34);
            v95[2] = v94;
            v96 = (__int64)v95;
            v97 = sub_7F8B70();
            v98 = *v94;
            v94[2] = (__int64)v97;
            v94 = (__int64 *)sub_73DBF0(0x67u, v98, v96);
LABEL_98:
            v41 = (const __m128i *)v94;
            return (_QWORD *)sub_730620((__int64)a1, v41);
          }
          v99 = 1;
          v127 = (__m128i *)sub_7F6190(v27, v240, v90);
          v94 = sub_7F88E0(v27, v127);
LABEL_104:
          if ( !v99 )
            goto LABEL_98;
          goto LABEL_97;
        }
        goto LABEL_100;
      }
    }
    v39 = 0;
    if ( v27 )
      goto LABEL_84;
    goto LABEL_100;
  }
  v42 = (_QWORD *)a1[7];
  v43 = (__m128i *)v42[3];
  v44 = v42[4];
  v45 = v42[2];
  v253[0].m128i_i64[0] = 0;
  sub_7EE560(v43, 0);
  if ( v44 )
  {
    v46 = sub_7FDF40(*(_QWORD *)(v44 + 16), 1, 0);
    if ( (v46[12].m128i_i8[0] & 2) != 0 && HIDWORD(qword_4F077B4) )
    {
      v254 = (__m128i *)sub_7E8090(v43, 1u);
      v179 = sub_8D46C0(v254->m128i_i64[0]);
      v180 = (_QWORD *)sub_691620(v179);
      v181 = sub_72D2E0(v180);
      v254 = (__m128i *)sub_73E130(v254, v181);
      v182 = sub_731330((__int64)v46);
      v184 = sub_7EBCC0((_QWORD **)v182, (__int64 *)&v254, 0, (__int64)v256, (const __m128i **)v253, v183);
      v185 = v184;
      if ( v253[0].m128i_i64[0] )
        v185 = (__int64 *)sub_73DF90(v253[0].m128i_i64[0], v184);
      v186 = sub_7F6050(v254);
      v187 = sub_7F0830(v186);
      v187[2] = v185;
      v188 = *v185;
      v189 = (__int64)v187;
      v190 = (const __m128i *)sub_724DC0();
      v257.m128i_i64[0] = (__int64)v190;
      sub_72BB40(v188, v190);
      v191 = sub_73A720((const __m128i *)v257.m128i_i64[0], (__int64)v190);
      sub_724E30((__int64)&v257);
      v185[2] = (__int64)v191;
      v192 = sub_73DBF0(0x67u, *v185, v189);
      v47 = v192;
      if ( v43 )
      {
        v193 = sub_7F9E20(v254, 0, (__int64)v192, v45);
        v194 = sub_7F0830(v43);
        v194[2] = v193;
        v195 = (__int64)v194;
        v196 = sub_7F8B70();
        v197 = *v193;
        v193[2] = (__int64)v196;
        v48 = (const __m128i *)sub_73DBF0(0x67u, v197, v195);
        return (_QWORD *)sub_730620((__int64)a1, v48);
      }
    }
    else
    {
      sub_7F9D60();
      v257.m128i_i64[0] = (__int64)sub_724DC0();
      v47 = sub_731330((__int64)v46);
      sub_724E30((__int64)&v257);
    }
  }
  else
  {
    v122 = sub_7F9D60();
    v257.m128i_i64[0] = (__int64)sub_724DC0();
    sub_72BB40(v122, (const __m128i *)v257.m128i_i64[0]);
    v47 = sub_73A720((const __m128i *)v257.m128i_i64[0], v257.m128i_i64[0]);
    sub_724E30((__int64)&v257);
  }
  v48 = (const __m128i *)sub_7F9E20(v43, 0, (__int64)v47, v45);
  return (_QWORD *)sub_730620((__int64)a1, v48);
}
