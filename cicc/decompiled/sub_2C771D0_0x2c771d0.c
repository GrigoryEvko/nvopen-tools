// Function: sub_2C771D0
// Address: 0x2c771d0
//
void __fastcall sub_2C771D0(__int64 a1, const char *a2)
{
  __int64 v3; // r12
  unsigned __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 *v7; // rax
  __m128i v8; // xmm0
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 i; // rbx
  __int64 v12; // rdi
  __int64 *v13; // rax
  __m128i *v14; // rdx
  __m128i v15; // xmm0
  int v16; // eax
  __int64 *v17; // rax
  __m128i *v18; // rdx
  __m128i si128; // xmm0
  __int16 v20; // dx
  const char *v21; // rsi
  __int64 *v22; // rax
  char *v23; // rcx
  __int64 v24; // rdx
  __m128i v25; // xmm0
  _BYTE *v26; // rax
  __int64 v27; // rdi
  __int16 v28; // ax
  const char *v29; // rsi
  __int64 *v30; // rax
  char *v31; // rcx
  __m128i *v32; // rdx
  __m128i v33; // xmm0
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // rax
  char *v37; // rsi
  __int64 v38; // rdi
  __int64 v39; // rdx
  __m128i v40; // xmm0
  __int64 v41; // rcx
  __m128i v42; // xmm0
  __int64 *v43; // rax
  __m128i *v44; // rdx
  __int64 v45; // rdi
  __m128i v46; // xmm0
  __m128i v47; // xmm0
  __int64 *v48; // rax
  __m128i *v49; // rdx
  __m128i v50; // xmm0
  __int64 v51; // rdx
  __int64 *v52; // rax
  __int64 v53; // rax
  char *v54; // rsi
  __int64 *v55; // rax
  char *v56; // rcx
  size_t v57; // rdx
  __int64 v58; // r15
  __m128i v59; // xmm0
  __m128i *v60; // rdi
  __int64 *v61; // rax
  char *v62; // rcx
  __m128i *v63; // rdx
  __m128i v64; // xmm0
  const char *v65; // rsi
  __int64 *v66; // rax
  char *v67; // rcx
  __m128i *v68; // rdx
  __m128i v69; // xmm0
  _BYTE *v70; // rax
  __int64 v71; // rdi
  __int64 *v72; // rax
  char *v73; // rcx
  __int64 v74; // rdx
  __m128i v75; // xmm0
  __int64 *v76; // rax
  char *v77; // rcx
  __m128i *v78; // rdx
  __m128i v79; // xmm0
  __int64 *v80; // rax
  char *v81; // rcx
  __int64 v82; // rdx
  __m128i v83; // xmm0
  __int64 *v84; // rax
  char *v85; // rcx
  __int64 v86; // rdx
  __m128i v87; // xmm0
  __int64 *v88; // rax
  char *v89; // rcx
  __m128i *v90; // rdx
  __int64 *v91; // rax
  __int64 *v92; // rax
  char *v93; // rcx
  __int64 v94; // rdx
  __m128i v95; // xmm0
  __int64 *v96; // rax
  char *v97; // rcx
  __m128i *v98; // rdx
  __int64 *v99; // rax
  char *v100; // rcx
  __int64 v101; // rdx
  __m128i v102; // xmm0
  __int64 *v103; // rax
  char *v104; // rcx
  __int64 v105; // rdx
  __m128i v106; // xmm0
  __int64 *v107; // rax
  char *v108; // rcx
  __int64 v109; // rdx
  __m128i v110; // xmm0
  __int64 *v111; // rax
  char *v112; // rcx
  __int64 v113; // rdx
  __m128i v114; // xmm0
  __int64 *v115; // rax
  char *v116; // rcx
  __int64 v117; // rdx
  __m128i v118; // xmm0
  __int64 *v119; // rax
  char *v120; // rcx
  __int64 v121; // rdx
  __m128i v122; // xmm0
  __int64 *v123; // rax
  char *v124; // rcx
  __int64 v125; // rdx
  __m128i v126; // xmm0
  __int64 *v127; // rax
  char *v128; // rcx
  __int64 v129; // rdx
  __m128i v130; // xmm0
  __int64 *v131; // rax
  char *v132; // rcx
  __int64 v133; // rdx
  __m128i v134; // xmm0
  __int64 *v135; // rax
  char *v136; // rcx
  __int64 v137; // rdx
  __m128i v138; // xmm0
  __int64 *v139; // rax
  char *v140; // rcx
  __int64 v141; // rdx
  __m128i v142; // xmm0
  __int64 *v143; // rax
  char *v144; // rcx
  __int64 v145; // rdx
  __m128i v146; // xmm0
  __int64 *v147; // rax
  char *v148; // rcx
  __int64 v149; // rdx
  __m128i v150; // xmm0
  __int64 *v151; // rax
  __m128i v152; // xmm0
  __int64 v153; // rdx
  __int64 v154; // rcx
  const char *v155; // rsi
  __int64 *v156; // rax
  char *v157; // rcx
  __int64 v158; // rdx
  __m128i v159; // xmm0
  char *v160; // rsi
  char *v161; // rcx
  __int64 v162; // rdi
  __m128i *v163; // rdx
  __int64 v164; // rdx
  __int64 *v165; // rax
  __int64 v166; // rax
  char *v167; // rax
  __int64 v168; // rax
  __int64 v169; // rdx
  char *v170; // rcx
  __int64 v171; // rdx
  char *v172; // rcx
  __int64 v173; // rdx
  char *v174; // rcx
  __int64 v175; // rdx
  char *v176; // rcx
  __int64 v177; // rdx
  char *v178; // rcx
  __int64 v179; // rdx
  char *v180; // rcx
  __int64 v181; // rdx
  char *v182; // rcx
  __int64 v183; // rdx
  char *v184; // rcx
  __int64 v185; // rdx
  char *v186; // rcx
  __int64 v187; // rdx
  char *v188; // rcx
  __int64 v189; // rdx
  char *v190; // rcx
  __int64 v191; // rdx
  char *v192; // rcx
  __int64 v193; // rdx
  char *v194; // rcx
  __int64 v195; // rdx
  char *v196; // rcx
  __int64 v197; // rdx
  char *v198; // rcx
  __int64 v199; // rdx
  char *v200; // rcx
  __int64 v201; // rdx
  char *v202; // rcx
  __int64 v203; // rdx
  char *v204; // rcx
  __int64 v205; // rdx
  char *v206; // rcx
  __int64 v207; // rdx
  char *v208; // rcx
  __int64 v209; // rdx
  char *v210; // rcx
  __int64 v211; // rdx
  char *v212; // rcx
  size_t v213; // [rsp+10h] [rbp-90h]
  __int64 v214; // [rsp+18h] [rbp-88h]
  __int64 v215; // [rsp+28h] [rbp-78h] BYREF
  __int64 v216; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v217; // [rsp+38h] [rbp-68h]
  __int64 v218; // [rsp+40h] [rbp-60h] BYREF
  __int64 v219; // [rsp+48h] [rbp-58h]
  char *v220; // [rsp+50h] [rbp-50h] BYREF
  int v221; // [rsp+58h] [rbp-48h]
  char v222; // [rsp+60h] [rbp-40h] BYREF

  v3 = (__int64)a2;
  sub_CE8EA0((__int64)&v220, (__int64)a2);
  v217 = sub_CE9030((__int64)a2);
  v4 = HIDWORD(v217);
  if ( v221 || BYTE4(v217) )
  {
    v16 = *(_DWORD *)(a1 + 8);
    if ( v16 )
    {
      if ( v16 <= 899 )
      {
        v17 = sub_2C764C0(a1, (__int64)a2, 2u);
        v18 = (__m128i *)v17[4];
        if ( (unsigned __int64)(v17[3] - (_QWORD)v18) <= 0x5B )
        {
          a2 = "Cluster dimensions and cluster maximum blocks are not supported on pre-Hopper Architectures\n";
          sub_CB6200(
            (__int64)v17,
            "Cluster dimensions and cluster maximum blocks are not supported on pre-Hopper Architectures\n",
            0x5Cu);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_42D0540);
          qmemcpy(&v18[5], "chitectures\n", 12);
          *v18 = si128;
          v18[1] = _mm_load_si128((const __m128i *)&xmmword_42D0550);
          v18[2] = _mm_load_si128((const __m128i *)&xmmword_42D0560);
          v18[3] = _mm_load_si128((const __m128i *)&xmmword_42D0570);
          v18[4] = _mm_load_si128((const __m128i *)&xmmword_42D0580);
          v17[4] += 92;
        }
      }
    }
  }
  if ( !(unsigned __int8)sub_CE9220(v3) )
  {
    if ( !v221 && !(_BYTE)v4 )
      goto LABEL_16;
    a2 = (const char *)v3;
    v7 = sub_2C764C0(a1, v3, 2u);
    v5 = v7[4];
    if ( (unsigned __int64)(v7[3] - v5) <= 0x53 )
    {
      a2 = "Cluster dimensions and cluster maximum blocks are only allowed for kernel functions\n";
      sub_CB6200(
        (__int64)v7,
        "Cluster dimensions and cluster maximum blocks are only allowed for kernel functions\n",
        0x54u);
    }
    else
    {
      v8 = _mm_load_si128((const __m128i *)&xmmword_42D0540);
      *(_DWORD *)(v5 + 80) = 175337071;
      *(__m128i *)v5 = v8;
      *(__m128i *)(v5 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D0550);
      *(__m128i *)(v5 + 32) = _mm_load_si128((const __m128i *)&xmmword_42D0560);
      *(__m128i *)(v5 + 48) = _mm_load_si128((const __m128i *)&xmmword_42D0590);
      *(__m128i *)(v5 + 64) = _mm_load_si128((const __m128i *)&xmmword_42D05A0);
      v7[4] += 84;
    }
  }
  if ( v221 )
  {
    v5 = (__int64)v220;
    v6 = *(unsigned int *)v220;
    if ( v221 != 1 )
    {
      if ( !*((_DWORD *)v220 + 1) )
      {
        if ( v221 != 2 && !*((_DWORD *)v220 + 2) && !(_DWORD)v6 )
          goto LABEL_14;
        goto LABEL_82;
      }
      if ( v221 != 2 )
      {
        a2 = (const char *)*((unsigned int *)v220 + 2);
        if ( (_DWORD)a2 && (_DWORD)v6 )
          goto LABEL_14;
        goto LABEL_82;
      }
    }
    if ( (_DWORD)v6 )
      goto LABEL_14;
LABEL_82:
    a2 = (const char *)v3;
    v43 = sub_2C764C0(a1, v3, 2u);
    v44 = (__m128i *)v43[4];
    v45 = (__int64)v43;
    if ( (unsigned __int64)(v43[3] - (_QWORD)v44) <= 0x2A )
    {
      a2 = "If any cluster dimension is specified as 0 ";
      v168 = sub_CB6200((__int64)v43, "If any cluster dimension is specified as 0 ", 0x2Bu);
      v5 = *(_QWORD *)(v168 + 32);
      v45 = v168;
    }
    else
    {
      v46 = _mm_load_si128((const __m128i *)&xmmword_42D05B0);
      v6 = 12320;
      qmemcpy(&v44[2], "ified as 0 ", 11);
      *v44 = v46;
      v44[1] = _mm_load_si128((const __m128i *)&xmmword_42D05C0);
      v5 = v43[4] + 43;
      v43[4] = v5;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v45 + 24) - v5) <= 0x30 )
    {
      a2 = "then all other dimensions must be specified as 0\n";
      sub_CB6200(v45, "then all other dimensions must be specified as 0\n", 0x31u);
    }
    else
    {
      v47 = _mm_load_si128((const __m128i *)&xmmword_42D05D0);
      *(_BYTE *)(v5 + 48) = 10;
      *(__m128i *)v5 = v47;
      *(__m128i *)(v5 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D05E0);
      *(__m128i *)(v5 + 32) = _mm_load_si128((const __m128i *)&xmmword_42D05F0);
      *(_QWORD *)(v45 + 32) += 49LL;
    }
  }
LABEL_14:
  if ( (_BYTE)v4 )
  {
    v5 = (unsigned int)v217;
    if ( !(_DWORD)v217 )
    {
      a2 = (const char *)v3;
      v151 = sub_2C764C0(a1, v3, 2u);
      v5 = v151[4];
      if ( (unsigned __int64)(v151[3] - v5) <= 0x27 )
      {
        a2 = "Cluster maximum blocks must be non-zero\n";
        sub_CB6200((__int64)v151, "Cluster maximum blocks must be non-zero\n", 0x28u);
      }
      else
      {
        v152 = _mm_load_si128((const __m128i *)&xmmword_42D0600);
        *(_QWORD *)(v5 + 32) = 0xA6F72657A2D6E6FLL;
        *(__m128i *)v5 = v152;
        *(__m128i *)(v5 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D0610);
        v151[4] += 40;
      }
    }
  }
LABEL_16:
  v214 = **(_QWORD **)(*(_QWORD *)(v3 + 24) + 16LL);
  if ( (*(_BYTE *)(v3 + 33) & 0x20) == 0 )
  {
    if ( (*(_BYTE *)(v3 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v3, (__int64)a2, v5, v6);
      v9 = *(_QWORD *)(v3 + 96);
      if ( (*(_BYTE *)(v3 + 2) & 1) != 0 )
        sub_B2C6D0(v3, (__int64)a2, v153, v154);
      v10 = *(_QWORD *)(v3 + 96);
    }
    else
    {
      v9 = *(_QWORD *)(v3 + 96);
      v10 = v9;
    }
    for ( i = v10 + 40LL * *(_QWORD *)(v3 + 104); v9 != i; v9 += 40 )
    {
      v12 = *(_QWORD *)(v9 + 8);
      if ( *(_BYTE *)(v12 + 8) == 12 )
      {
        v218 = sub_BCAE30(v12);
        v219 = v51;
        if ( (unsigned __int64)sub_CA1930(&v218) <= 0x1F )
        {
          v215 = *(_QWORD *)(v3 + 120);
          if ( !(unsigned __int8)sub_A74710(&v215, *(_DWORD *)(v9 + 32) + 1, 54) )
          {
            v216 = *(_QWORD *)(v3 + 120);
            if ( !(unsigned __int8)sub_A74710(&v216, *(_DWORD *)(v9 + 32) + 1, 79) )
            {
              v52 = sub_2C764C0(a1, v3, 2u);
              v53 = sub_904010((__int64)v52, "Integer parameter less than 32-bits without ");
              sub_904010(v53, "sext/zext flag\n");
            }
          }
        }
      }
      v218 = *(_QWORD *)(v3 + 120);
      if ( (unsigned __int8)sub_A74710(&v218, *(_DWORD *)(v9 + 32) + 1, 15) )
      {
        v13 = sub_2C764C0(a1, v3, 1u);
        v14 = (__m128i *)v13[4];
        if ( (unsigned __int64)(v13[3] - (_QWORD)v14) <= 0x2C )
        {
          sub_CB6200((__int64)v13, "InReg attribute on parameter will be ignored\n", 0x2Du);
        }
        else
        {
          v15 = _mm_load_si128((const __m128i *)&xmmword_42D0640);
          qmemcpy(&v14[2], "l be ignored\n", 13);
          *v14 = v15;
          v14[1] = _mm_load_si128((const __m128i *)&xmmword_42D0650);
          v13[4] += 45;
        }
      }
      v218 = *(_QWORD *)(v3 + 120);
      if ( (unsigned __int8)sub_A74710(&v218, *(_DWORD *)(v9 + 32) + 1, 21) )
      {
        v48 = sub_2C764C0(a1, v3, 1u);
        v49 = (__m128i *)v48[4];
        if ( (unsigned __int64)(v48[3] - (_QWORD)v49) <= 0x2B )
        {
          sub_CB6200((__int64)v48, "Nest attribute on parameter will be ignored\n", 0x2Cu);
        }
        else
        {
          v50 = _mm_load_si128((const __m128i *)&xmmword_42D0660);
          qmemcpy(&v49[2], " be ignored\n", 12);
          *v49 = v50;
          v49[1] = _mm_load_si128((const __m128i *)&xmmword_42D0670);
          v48[4] += 44;
        }
      }
    }
    if ( *(_BYTE *)(v214 + 8) == 12 )
    {
      v218 = sub_BCAE30(v214);
      v219 = v164;
      if ( (unsigned __int64)sub_CA1930(&v218) <= 0x1F )
      {
        v215 = *(_QWORD *)(v3 + 120);
        if ( !(unsigned __int8)sub_A74710(&v215, 0, 54) )
        {
          v216 = *(_QWORD *)(v3 + 120);
          if ( !(unsigned __int8)sub_A74710(&v216, 0, 79) )
          {
            v165 = sub_2C764C0(a1, v3, 2u);
            v166 = sub_904010((__int64)v165, "Integer return less than 32-bits without ");
            sub_904010(v166, "sext/zext flag\n");
          }
        }
      }
    }
  }
  v20 = *(_WORD *)(v3 + 34) >> 1;
  if ( (*(_WORD *)(v3 + 34) & 0x400) != 0 )
  {
    v54 = (char *)v3;
    v55 = sub_2C764C0(a1, v3, 0);
    v57 = v55[4];
    v58 = (__int64)v55;
    if ( v55[3] - v57 <= 0x17 )
    {
      v54 = "Explicit section marker ";
      v58 = sub_CB6200((__int64)v55, "Explicit section marker ", 0x18u);
    }
    else
    {
      v59 = _mm_load_si128((const __m128i *)&xmmword_42D0680);
      *(_QWORD *)(v57 + 16) = 0x2072656B72616D20LL;
      *(__m128i *)v57 = v59;
      v55[4] += 24;
    }
    if ( (*(_BYTE *)(v3 + 35) & 4) != 0 )
    {
      v167 = (char *)sub_B31D10(v3, (__int64)v54, v57);
      v60 = *(__m128i **)(v58 + 32);
      v54 = v167;
      if ( *(_QWORD *)(v58 + 24) - (_QWORD)v60 >= v57 )
      {
        if ( v57 )
        {
          v213 = v57;
          memcpy(v60, v167, v57);
          v57 = *(_QWORD *)(v58 + 32) + v213;
          *(_QWORD *)(v58 + 32) = v57;
          v60 = (__m128i *)v57;
        }
        goto LABEL_98;
      }
      v58 = sub_CB6200(v58, (unsigned __int8 *)v167, v57);
    }
    v60 = *(__m128i **)(v58 + 32);
LABEL_98:
    if ( *(_QWORD *)(v58 + 24) - (_QWORD)v60 <= 0xFu )
    {
      v54 = "is not allowed.\n";
      sub_CB6200(v58, (unsigned __int8 *)"is not allowed.\n", 0x10u);
    }
    else
    {
      *v60 = _mm_load_si128((const __m128i *)&xmmword_42D0690);
      *(_QWORD *)(v58 + 32) += 16LL;
    }
    sub_2C76240(a1, (__int64)v54, v57, v56);
    v20 = *(_WORD *)(v3 + 34) >> 1;
  }
  if ( (v20 & 0x3F) != 0 )
  {
    v21 = (const char *)v3;
    v22 = sub_2C764C0(a1, v3, 0);
    v24 = v22[4];
    if ( (unsigned __int64)(v22[3] - v24) <= 0x22 )
    {
      v21 = "Explicit alignment is not allowed.\n";
      sub_CB6200((__int64)v22, "Explicit alignment is not allowed.\n", 0x23u);
    }
    else
    {
      v25 = _mm_load_si128((const __m128i *)&xmmword_42D06A0);
      *(_BYTE *)(v24 + 34) = 10;
      *(_WORD *)(v24 + 32) = 11876;
      *(__m128i *)v24 = v25;
      *(__m128i *)(v24 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D06B0);
      v22[4] += 35;
    }
    v26 = *(_BYTE **)(a1 + 16);
    if ( v26 )
      *v26 = 0;
    if ( !*(_DWORD *)(a1 + 4) )
    {
      v27 = *(_QWORD *)(a1 + 24);
      if ( *(_QWORD *)(v27 + 32) != *(_QWORD *)(v27 + 16) )
      {
        sub_CB5AE0((__int64 *)v27);
        v27 = *(_QWORD *)(a1 + 24);
      }
      sub_CEB520(*(_QWORD **)(v27 + 48), (__int64)v21, v24, v23);
    }
  }
  v28 = *(_WORD *)(v3 + 2);
  if ( (v28 & 2) != 0 )
  {
    v61 = sub_2C764C0(a1, v3, 0);
    v63 = (__m128i *)v61[4];
    if ( (unsigned __int64)(v61[3] - (_QWORD)v63) <= 0x1B )
    {
      sub_CB6200((__int64)v61, "Prefix data is not allowed.\n", 0x1Cu);
      sub_2C76240(a1, (__int64)"Prefix data is not allowed.\n", v185, v186);
    }
    else
    {
      v64 = _mm_load_si128((const __m128i *)&xmmword_42D06C0);
      qmemcpy(&v63[1], "ot allowed.\n", 12);
      *v63 = v64;
      v61[4] += 28;
      sub_2C76240(a1, v3, (__int64)v63, v62);
    }
    v28 = *(_WORD *)(v3 + 2);
    if ( (v28 & 4) == 0 )
    {
LABEL_44:
      if ( (v28 & 8) == 0 )
        goto LABEL_45;
      goto LABEL_113;
    }
  }
  else if ( (v28 & 4) == 0 )
  {
    goto LABEL_44;
  }
  v65 = (const char *)v3;
  v66 = sub_2C764C0(a1, v3, 0);
  v68 = (__m128i *)v66[4];
  if ( (unsigned __int64)(v66[3] - (_QWORD)v68) <= 0x1D )
  {
    v65 = "Prologue data is not allowed.\n";
    sub_CB6200((__int64)v66, "Prologue data is not allowed.\n", 0x1Eu);
  }
  else
  {
    v69 = _mm_load_si128((const __m128i *)&xmmword_42D06D0);
    qmemcpy(&v68[1], " not allowed.\n", 14);
    *v68 = v69;
    v66[4] += 30;
  }
  v70 = *(_BYTE **)(a1 + 16);
  if ( v70 )
    *v70 = 0;
  if ( !*(_DWORD *)(a1 + 4) )
  {
    v71 = *(_QWORD *)(a1 + 24);
    if ( *(_QWORD *)(v71 + 32) != *(_QWORD *)(v71 + 16) )
    {
      sub_CB5AE0((__int64 *)v71);
      v71 = *(_QWORD *)(a1 + 24);
    }
    sub_CEB520(*(_QWORD **)(v71 + 48), (__int64)v65, (__int64)v68, v67);
  }
  v28 = *(_WORD *)(v3 + 2);
  if ( (v28 & 8) == 0 )
  {
LABEL_45:
    if ( (v28 & 0x4000) == 0 )
      goto LABEL_46;
    goto LABEL_116;
  }
LABEL_113:
  v72 = sub_2C764C0(a1, v3, 0);
  v74 = v72[4];
  if ( (unsigned __int64)(v72[3] - v74) <= 0x24 )
  {
    sub_CB6200((__int64)v72, "Personality function is not allowed.\n", 0x25u);
    sub_2C76240(a1, (__int64)"Personality function is not allowed.\n", v183, v184);
  }
  else
  {
    v75 = _mm_load_si128((const __m128i *)&xmmword_42D06E0);
    *(_DWORD *)(v74 + 32) = 778331511;
    *(_BYTE *)(v74 + 36) = 10;
    *(__m128i *)v74 = v75;
    *(__m128i *)(v74 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D06F0);
    v72[4] += 37;
    sub_2C76240(a1, v3, v74, v73);
  }
  if ( (*(_WORD *)(v3 + 2) & 0x4000) != 0 )
  {
LABEL_116:
    v76 = sub_2C764C0(a1, v3, 0);
    v78 = (__m128i *)v76[4];
    if ( (unsigned __int64)(v76[3] - (_QWORD)v78) <= 0x1B )
    {
      sub_CB6200((__int64)v76, "GC names are not supported.\n", 0x1Cu);
      sub_2C76240(a1, (__int64)"GC names are not supported.\n", v187, v188);
    }
    else
    {
      v79 = _mm_load_si128((const __m128i *)&xmmword_42D0700);
      qmemcpy(&v78[1], " supported.\n", 12);
      *v78 = v79;
      v76[4] += 28;
      sub_2C76240(a1, v3, (__int64)v78, v77);
    }
  }
LABEL_46:
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 86) )
  {
    v88 = sub_2C764C0(a1, v3, 0);
    v90 = (__m128i *)v88[4];
    if ( (unsigned __int64)(v88[3] - (_QWORD)v90) <= 0x2F )
    {
      sub_CB6200((__int64)v88, "alignstack function attribute is not supported.\n", 0x30u);
      sub_2C76240(a1, (__int64)"alignstack function attribute is not supported.\n", v189, v190);
    }
    else
    {
      *v90 = _mm_load_si128((const __m128i *)&xmmword_42D0710);
      v90[1] = _mm_load_si128((const __m128i *)&xmmword_42D0720);
      v90[2] = _mm_load_si128((const __m128i *)&xmmword_42D0730);
      v88[4] += 48;
      sub_2C76240(a1, v3, (__int64)v90, v89);
    }
  }
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 42) )
  {
    v84 = sub_2C764C0(a1, v3, 0);
    v86 = v84[4];
    if ( (unsigned __int64)(v84[3] - v86) <= 0x30 )
    {
      sub_CB6200((__int64)v84, "nonlazybind function attribute is not supported.\n", 0x31u);
      sub_2C76240(a1, (__int64)"nonlazybind function attribute is not supported.\n", v191, v192);
    }
    else
    {
      v87 = _mm_load_si128((const __m128i *)&xmmword_42D0740);
      *(_BYTE *)(v86 + 48) = 10;
      *(__m128i *)v86 = v87;
      *(__m128i *)(v86 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D0750);
      *(__m128i *)(v86 + 32) = _mm_load_si128((const __m128i *)&xmmword_42D0760);
      v84[4] += 49;
      sub_2C76240(a1, v3, v86, v85);
    }
  }
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 20) )
  {
    v80 = sub_2C764C0(a1, v3, 0);
    v82 = v80[4];
    if ( (unsigned __int64)(v80[3] - v82) <= 0x2A )
    {
      sub_CB6200((__int64)v80, "naked function attribute is not supported.\n", 0x2Bu);
      sub_2C76240(a1, (__int64)"naked function attribute is not supported.\n", v193, v194);
    }
    else
    {
      v83 = _mm_load_si128((const __m128i *)&xmmword_42D0770);
      qmemcpy((void *)(v82 + 32), "supported.\n", 11);
      *(__m128i *)v82 = v83;
      *(__m128i *)(v82 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D0780);
      v80[4] += 43;
      sub_2C76240(a1, v3, v82, v81);
    }
  }
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 30) )
  {
    v147 = sub_2C764C0(a1, v3, 0);
    v149 = v147[4];
    if ( (unsigned __int64)(v147[3] - v149) <= 0x34 )
    {
      sub_CB6200((__int64)v147, "noimplicitfloat function attribute is not supported.\n", 0x35u);
      sub_2C76240(a1, (__int64)"noimplicitfloat function attribute is not supported.\n", v169, v170);
    }
    else
    {
      v150 = _mm_load_si128((const __m128i *)&xmmword_42D0790);
      *(_DWORD *)(v149 + 48) = 778331508;
      *(_BYTE *)(v149 + 52) = 10;
      *(__m128i *)v149 = v150;
      *(__m128i *)(v149 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D07A0);
      *(__m128i *)(v149 + 32) = _mm_load_si128((const __m128i *)&xmmword_42D07B0);
      v147[4] += 53;
      sub_2C76240(a1, v3, v149, v148);
    }
  }
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 35) )
  {
    v143 = sub_2C764C0(a1, v3, 0);
    v145 = v143[4];
    if ( (unsigned __int64)(v143[3] - v145) <= 0x2E )
    {
      sub_CB6200((__int64)v143, "noredzone function attribute is not supported.\n", 0x2Fu);
      sub_2C76240(a1, (__int64)"noredzone function attribute is not supported.\n", v171, v172);
    }
    else
    {
      v146 = _mm_load_si128((const __m128i *)&xmmword_42D07C0);
      qmemcpy((void *)(v145 + 32), "not supported.\n", 15);
      *(__m128i *)v145 = v146;
      *(__m128i *)(v145 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D07D0);
      v143[4] += 47;
      sub_2C76240(a1, v3, v145, v144);
    }
  }
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 53) )
  {
    v139 = sub_2C764C0(a1, v3, 0);
    v141 = v139[4];
    if ( (unsigned __int64)(v139[3] - v141) <= 0x32 )
    {
      sub_CB6200((__int64)v139, "returns_twice function attribute is not supported.\n", 0x33u);
      sub_2C76240(a1, (__int64)"returns_twice function attribute is not supported.\n", v195, v196);
    }
    else
    {
      v142 = _mm_load_si128((const __m128i *)&xmmword_42D07E0);
      *(_BYTE *)(v141 + 50) = 10;
      *(_WORD *)(v141 + 48) = 11876;
      *(__m128i *)v141 = v142;
      *(__m128i *)(v141 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D07F0);
      *(__m128i *)(v141 + 32) = _mm_load_si128((const __m128i *)&xmmword_42D0800);
      v139[4] += 51;
      sub_2C76240(a1, v3, v141, v140);
    }
  }
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 69) )
  {
    v135 = sub_2C764C0(a1, v3, 0);
    v137 = v135[4];
    if ( (unsigned __int64)(v135[3] - v137) <= 0x28 )
    {
      sub_CB6200((__int64)v135, "ssp function attribute is not supported.\n", 0x29u);
      sub_2C76240(a1, (__int64)"ssp function attribute is not supported.\n", v197, v198);
    }
    else
    {
      v138 = _mm_load_si128((const __m128i *)&xmmword_42D0810);
      *(_BYTE *)(v137 + 40) = 10;
      *(_QWORD *)(v137 + 32) = 0x2E646574726F7070LL;
      *(__m128i *)v137 = v138;
      *(__m128i *)(v137 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D0820);
      v135[4] += 41;
      sub_2C76240(a1, v3, v137, v136);
    }
  }
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 70) )
  {
    v131 = sub_2C764C0(a1, v3, 0);
    v133 = v131[4];
    if ( (unsigned __int64)(v131[3] - v133) <= 0x2B )
    {
      sub_CB6200((__int64)v131, "sspreq function attribute is not supported.\n", 0x2Cu);
      sub_2C76240(a1, (__int64)"sspreq function attribute is not supported.\n", v199, v200);
    }
    else
    {
      v134 = _mm_load_si128((const __m128i *)&xmmword_42D0830);
      qmemcpy((void *)(v133 + 32), " supported.\n", 12);
      *(__m128i *)v133 = v134;
      *(__m128i *)(v133 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D0840);
      v131[4] += 44;
      sub_2C76240(a1, v3, v133, v132);
    }
  }
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 71) )
  {
    v127 = sub_2C764C0(a1, v3, 0);
    v129 = v127[4];
    if ( (unsigned __int64)(v127[3] - v129) <= 0x2E )
    {
      sub_CB6200((__int64)v127, "sspstrong function attribute is not supported.\n", 0x2Fu);
      sub_2C76240(a1, (__int64)"sspstrong function attribute is not supported.\n", v201, v202);
    }
    else
    {
      v130 = _mm_load_si128((const __m128i *)&xmmword_42D0850);
      qmemcpy((void *)(v129 + 32), "not supported.\n", 15);
      *(__m128i *)v129 = v130;
      *(__m128i *)(v129 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D07D0);
      v127[4] += 47;
      sub_2C76240(a1, v3, v129, v128);
    }
  }
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 95) )
  {
    v123 = sub_2C764C0(a1, v3, 0);
    v125 = v123[4];
    if ( (unsigned __int64)(v123[3] - v125) <= 0x2C )
    {
      sub_CB6200((__int64)v123, "uwtable function attribute is not supported.\n", 0x2Du);
      sub_2C76240(a1, (__int64)"uwtable function attribute is not supported.\n", v203, v204);
    }
    else
    {
      v126 = _mm_load_si128((const __m128i *)&xmmword_42D0860);
      qmemcpy((void *)(v125 + 32), "t supported.\n", 13);
      *(__m128i *)v125 = v126;
      *(__m128i *)(v125 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D0870);
      v123[4] += 45;
      sub_2C76240(a1, v3, v125, v124);
    }
  }
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 17) )
  {
    v119 = sub_2C764C0(a1, v3, 0);
    v121 = v119[4];
    if ( (unsigned __int64)(v119[3] - v121) <= 0x2E )
    {
      sub_CB6200((__int64)v119, "jumptable function attribute is not supported.\n", 0x2Fu);
      sub_2C76240(a1, (__int64)"jumptable function attribute is not supported.\n", v205, v206);
    }
    else
    {
      v122 = _mm_load_si128((const __m128i *)&xmmword_42D0880);
      qmemcpy((void *)(v121 + 32), "not supported.\n", 15);
      *(__m128i *)v121 = v122;
      *(__m128i *)(v121 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D07D0);
      v119[4] += 47;
      sub_2C76240(a1, v3, v121, v120);
    }
  }
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 4) )
  {
    v115 = sub_2C764C0(a1, v3, 0);
    v117 = v115[4];
    if ( (unsigned __int64)(v115[3] - v117) <= 0x2C )
    {
      sub_CB6200((__int64)v115, "builtin function attribute is not supported.\n", 0x2Du);
      sub_2C76240(a1, (__int64)"builtin function attribute is not supported.\n", v207, v208);
    }
    else
    {
      v118 = _mm_load_si128((const __m128i *)&xmmword_42D0890);
      qmemcpy((void *)(v117 + 32), "t supported.\n", 13);
      *(__m128i *)v117 = v118;
      *(__m128i *)(v117 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D0870);
      v115[4] += 45;
      sub_2C76240(a1, v3, v117, v116);
    }
  }
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 23) )
  {
    v111 = sub_2C764C0(a1, v3, 0);
    v113 = v111[4];
    if ( (unsigned __int64)(v111[3] - v113) <= 0x2E )
    {
      sub_CB6200((__int64)v111, "nobuiltin function attribute is not supported.\n", 0x2Fu);
      sub_2C76240(a1, (__int64)"nobuiltin function attribute is not supported.\n", v209, v210);
    }
    else
    {
      v114 = _mm_load_si128((const __m128i *)&xmmword_42D08A0);
      qmemcpy((void *)(v113 + 32), "not supported.\n", 15);
      *(__m128i *)v113 = v114;
      *(__m128i *)(v113 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D07D0);
      v111[4] += 47;
      sub_2C76240(a1, v3, v113, v112);
    }
  }
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 56) )
  {
    v107 = sub_2C764C0(a1, v3, 0);
    v109 = v107[4];
    if ( (unsigned __int64)(v107[3] - v109) <= 0x35 )
    {
      sub_CB6200((__int64)v107, "sanitize_address function attribute is not supported.\n", 0x36u);
      sub_2C76240(a1, (__int64)"sanitize_address function attribute is not supported.\n", v175, v176);
    }
    else
    {
      v110 = _mm_load_si128((const __m128i *)&xmmword_4293160);
      *(_DWORD *)(v109 + 48) = 1684370546;
      *(_WORD *)(v109 + 52) = 2606;
      *(__m128i *)v109 = v110;
      *(__m128i *)(v109 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D08B0);
      *(__m128i *)(v109 + 32) = _mm_load_si128((const __m128i *)&xmmword_42D08C0);
      v107[4] += 54;
      sub_2C76240(a1, v3, v109, v108);
    }
  }
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 59) )
  {
    v103 = sub_2C764C0(a1, v3, 0);
    v105 = v103[4];
    if ( (unsigned __int64)(v103[3] - v105) <= 0x34 )
    {
      sub_CB6200((__int64)v103, "sanitize_memory function attribute is not supported.\n", 0x35u);
      sub_2C76240(a1, (__int64)"sanitize_memory function attribute is not supported.\n", v177, v178);
    }
    else
    {
      v106 = _mm_load_si128((const __m128i *)&xmmword_42D08D0);
      *(_DWORD *)(v105 + 48) = 778331508;
      *(_BYTE *)(v105 + 52) = 10;
      *(__m128i *)v105 = v106;
      *(__m128i *)(v105 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D07A0);
      *(__m128i *)(v105 + 32) = _mm_load_si128((const __m128i *)&xmmword_42D07B0);
      v103[4] += 53;
      sub_2C76240(a1, v3, v105, v104);
    }
  }
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 63) )
  {
    v99 = sub_2C764C0(a1, v3, 0);
    v101 = v99[4];
    if ( (unsigned __int64)(v99[3] - v101) <= 0x34 )
    {
      sub_CB6200((__int64)v99, "sanitize_thread function attribute is not supported.\n", 0x35u);
      sub_2C76240(a1, (__int64)"sanitize_thread function attribute is not supported.\n", v179, v180);
    }
    else
    {
      v102 = _mm_load_si128((const __m128i *)&xmmword_42D08E0);
      *(_DWORD *)(v101 + 48) = 778331508;
      *(_BYTE *)(v101 + 52) = 10;
      *(__m128i *)v101 = v102;
      *(__m128i *)(v101 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D07A0);
      *(__m128i *)(v101 + 32) = _mm_load_si128((const __m128i *)&xmmword_42D07B0);
      v99[4] += 53;
      sub_2C76240(a1, v3, v101, v100);
    }
  }
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 94) )
  {
    v96 = sub_2C764C0(a1, v3, 0);
    v98 = (__m128i *)v96[4];
    if ( (unsigned __int64)(v96[3] - (_QWORD)v98) <= 0x2F )
    {
      sub_CB6200((__int64)v96, "alignstack function attribute is not supported.\n", 0x30u);
      sub_2C76240(a1, (__int64)"alignstack function attribute is not supported.\n", v181, v182);
    }
    else
    {
      *v98 = _mm_load_si128((const __m128i *)&xmmword_42D0710);
      v98[1] = _mm_load_si128((const __m128i *)&xmmword_42D0720);
      v98[2] = _mm_load_si128((const __m128i *)&xmmword_42D0730);
      v96[4] += 48;
      sub_2C76240(a1, v3, (__int64)v98, v97);
    }
  }
  v218 = *(_QWORD *)(v3 + 120);
  if ( (unsigned __int8)sub_A73ED0(&v218, 55) )
  {
    v92 = sub_2C764C0(a1, v3, 0);
    v94 = v92[4];
    if ( (unsigned __int64)(v92[3] - v94) <= 0x2E )
    {
      sub_CB6200((__int64)v92, "safestack function attribute is not supported.\n", 0x2Fu);
      sub_2C76240(a1, (__int64)"safestack function attribute is not supported.\n", v173, v174);
    }
    else
    {
      v95 = _mm_load_si128((const __m128i *)&xmmword_42D08F0);
      qmemcpy((void *)(v94 + 32), "not supported.\n", 15);
      *(__m128i *)v94 = v95;
      *(__m128i *)(v94 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D07D0);
      v92[4] += 47;
      sub_2C76240(a1, 11876, v94, v93);
    }
  }
  if ( !(unsigned __int8)sub_CE9220(v3) )
  {
    if ( !(unsigned __int8)sub_CE80C0(v3)
      && !(unsigned __int8)sub_CE80F0(v3)
      && !(unsigned __int8)sub_CE8120(v3)
      && !(unsigned __int8)sub_CE8150(v3)
      && !(unsigned __int8)sub_CE8180(v3) )
    {
      goto LABEL_66;
    }
    if ( *(_BYTE *)(v214 + 8) == 7 )
    {
      if ( *(_DWORD *)(*(_QWORD *)(v3 + 24) + 12LL) == 1 )
        goto LABEL_66;
    }
    else
    {
      v155 = (const char *)v3;
      v156 = sub_2C764C0(a1, v3, 0);
      v158 = v156[4];
      if ( (unsigned __int64)(v156[3] - v158) <= 0x18 )
      {
        v155 = "non-void entry function.\n";
        sub_CB6200((__int64)v156, "non-void entry function.\n", 0x19u);
      }
      else
      {
        v159 = _mm_load_si128((const __m128i *)&xmmword_42D0900);
        *(_BYTE *)(v158 + 24) = 10;
        *(_QWORD *)(v158 + 16) = 0x2E6E6F6974636E75LL;
        *(__m128i *)v158 = v159;
        v156[4] += 25;
      }
      sub_2C76240(a1, (__int64)v155, v158, v157);
      if ( *(_DWORD *)(*(_QWORD *)(v3 + 24) + 12LL) == 1 )
      {
LABEL_66:
        if ( !(unsigned __int8)sub_CE92C0(v3) )
          goto LABEL_67;
        goto LABEL_74;
      }
    }
    v29 = (const char *)v3;
    v91 = sub_2C764C0(a1, v3, 0);
    v32 = (__m128i *)v91[4];
    if ( (unsigned __int64)(v91[3] - (_QWORD)v32) <= 0x1F )
    {
      v29 = "entry function with parameters.\n";
      sub_CB6200((__int64)v91, "entry function with parameters.\n", 0x20u);
    }
    else
    {
      *v32 = _mm_load_si128((const __m128i *)&xmmword_42D0910);
      v32[1] = _mm_load_si128((const __m128i *)&xmmword_42D0920);
      v91[4] += 32;
    }
    goto LABEL_72;
  }
  if ( *(_BYTE *)(v214 + 8) == 7 )
    goto LABEL_66;
  v29 = (const char *)v3;
  v30 = sub_2C764C0(a1, v3, 0);
  v32 = (__m128i *)v30[4];
  if ( (unsigned __int64)(v30[3] - (_QWORD)v32) > 0x18 )
  {
    v33 = _mm_load_si128((const __m128i *)&xmmword_42D0900);
    v32[1].m128i_i8[8] = 10;
    v32[1].m128i_i64[0] = 0x2E6E6F6974636E75LL;
    *v32 = v33;
    v30[4] += 25;
LABEL_72:
    sub_2C76240(a1, (__int64)v29, (__int64)v32, v31);
    goto LABEL_73;
  }
  sub_CB6200((__int64)v30, "non-void entry function.\n", 0x19u);
  sub_2C76240(a1, (__int64)"non-void entry function.\n", v211, v212);
LABEL_73:
  if ( !(unsigned __int8)sub_CE92C0(v3) )
    goto LABEL_67;
LABEL_74:
  v34 = sub_CE8210(v3);
  v35 = v34;
  if ( !v34 )
    goto LABEL_67;
  v36 = *(_QWORD *)(v34 + 24);
  if ( *(_BYTE *)(**(_QWORD **)(v36 + 16) + 8LL) == 7 )
  {
    if ( *(_DWORD *)(v36 + 12) == 1 )
      goto LABEL_67;
    goto LABEL_77;
  }
  v160 = "Error: ";
  sub_904010(*(_QWORD *)(a1 + 24), "Error: ");
  v162 = *(_QWORD *)(a1 + 24);
  v163 = *(__m128i **)(v162 + 32);
  if ( *(_QWORD *)(v162 + 24) - (_QWORD)v163 <= 0x1Fu )
  {
    v160 = "non-void exit handler function.\n";
    sub_CB6200(v162, "non-void exit handler function.\n", 0x20u);
  }
  else
  {
    *v163 = _mm_load_si128((const __m128i *)&xmmword_42D0930);
    v163[1] = _mm_load_si128((const __m128i *)&xmmword_42D0940);
    *(_QWORD *)(v162 + 32) += 32LL;
  }
  sub_2C76240(a1, (__int64)v160, (__int64)v163, v161);
  if ( *(_DWORD *)(*(_QWORD *)(v35 + 24) + 12LL) != 1 )
  {
LABEL_77:
    v37 = "Error: ";
    sub_904010(*(_QWORD *)(a1 + 24), "Error: ");
    v38 = *(_QWORD *)(a1 + 24);
    v39 = *(_QWORD *)(v38 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v38 + 24) - v39) <= 0x26 )
    {
      v37 = "exit handler function with parameters.\n";
      sub_CB6200(v38, "exit handler function with parameters.\n", 0x27u);
    }
    else
    {
      v40 = _mm_load_si128((const __m128i *)&xmmword_42D0950);
      v41 = 11891;
      *(_DWORD *)(v39 + 32) = 1919251557;
      *(_WORD *)(v39 + 36) = 11891;
      *(__m128i *)v39 = v40;
      v42 = _mm_load_si128((const __m128i *)&xmmword_42D0960);
      *(_BYTE *)(v39 + 38) = 10;
      *(__m128i *)(v39 + 16) = v42;
      *(_QWORD *)(v38 + 32) += 39LL;
    }
    sub_2C76240(a1, (__int64)v37, v39, (char *)v41);
  }
LABEL_67:
  if ( v220 != &v222 )
    _libc_free((unsigned __int64)v220);
}
