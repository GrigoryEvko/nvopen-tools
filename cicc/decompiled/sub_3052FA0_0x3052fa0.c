// Function: sub_3052FA0
// Address: 0x3052fa0
//
__int64 __fastcall sub_3052FA0(__int64 a1, __int64 a2, _DWORD *a3)
{
  _DWORD *v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // r8
  _DWORD *v8; // r11
  unsigned int v9; // edx
  _DWORD *v10; // rax
  int v11; // ecx
  char v12; // al
  bool v13; // cc
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  char v17; // al
  __int64 v18; // rax
  __int64 v19; // rsi
  unsigned int v20; // edx
  __int64 v21; // rdx
  __int64 *v22; // rax
  unsigned __int16 *v23; // rdx
  __int64 i; // rax
  __int64 v25; // rax
  __int64 *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 *v29; // rax
  __int16 v30; // ax
  __int16 v31; // ax
  __int16 v32; // ax
  __int16 v33; // ax
  __int16 v34; // ax
  __int16 v35; // ax
  __int16 v36; // ax
  __int16 v37; // ax
  __int16 v38; // ax
  __int16 v39; // ax
  __int16 v40; // ax
  __int16 v41; // ax
  __int16 v42; // ax
  __int16 v43; // ax
  __int16 v44; // ax
  __int16 v45; // ax
  __int16 v46; // ax
  __int16 v47; // ax
  __int16 v48; // ax
  __int64 j; // rax
  __int64 *v50; // rcx
  __int64 k; // rax
  __int64 v52; // rax
  int v53; // edx
  int v54; // ecx
  __int64 *v55; // rsi
  int v56; // eax
  _BYTE *v57; // rsi
  __int64 v58; // rcx
  __int64 v59; // rcx
  __int64 *v60; // rdx
  __int64 m; // rax
  __int64 *v62; // rdx
  __int64 n; // rax
  __int64 **v64; // rdi
  __int64 v65; // rcx
  __int64 v66; // rdx
  __int64 *v67; // rax
  __int64 v68; // rcx
  unsigned int v69; // esi
  char v70; // al
  __int64 v71; // rdx
  __int64 *v72; // rax
  __int64 *v73; // rdx
  int v74; // ecx
  __int64 v75; // rsi
  int v76; // eax
  __int64 v77; // rcx
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // r8
  __int64 v81; // r15
  unsigned int *v82; // r13
  __int64 v83; // r14
  char v84; // al
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r8
  char v88; // al
  char v89; // dl
  __int64 v90; // r14
  __int64 v91; // r14
  char v92; // al
  __int64 v93; // rcx
  __int64 v94; // r8
  __int64 v95; // rdx
  unsigned int *v96; // r13
  __int64 ii; // rsi
  unsigned __int16 *v98; // r15
  __int64 v99; // rdx
  __int64 v100; // rcx
  __int64 v101; // r8
  __int64 v102; // rax
  char v103; // al
  unsigned int *v104; // r15
  __int64 v105; // rsi
  char v106; // al
  __int64 v107; // r8
  unsigned int v108; // edx
  unsigned int v109; // ecx
  __int64 v110; // rax
  __int64 v111; // rcx
  __int64 v112; // rsi
  unsigned int v113; // edx
  unsigned int *v114; // r12
  __int64 v115; // rdx
  __int64 v116; // rsi
  __int64 v117; // rax
  __int64 v118; // rax
  unsigned __int16 *jj; // rdx
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 v122; // rsi
  _DWORD *v123; // r12
  __int64 v124; // rsi
  __int64 v125; // rdx
  __int64 v126; // rcx
  __int64 v127; // r8
  __int64 v128; // rax
  unsigned __int16 *v129; // rdi
  __int64 kk; // rcx
  __int64 v131; // rdx
  __int64 *v132; // rax
  __int64 v133; // rcx
  __int64 v134; // rdx
  __int64 *mm; // rax
  __int64 v136; // rax
  unsigned int *v137; // r13
  unsigned int v138; // r15d
  unsigned int v139; // eax
  __int64 v140; // rcx
  __int64 v141; // r8
  __int64 v142; // rdx
  char v143; // al
  __int64 v144; // r14
  char v145; // al
  char v146; // dl
  __int64 v147; // r14
  __int64 v148; // r14
  char v149; // al
  bool v150; // al
  unsigned int *v151; // r13
  unsigned int v152; // r14d
  char v153; // al
  __int64 v154; // rdx
  __int64 v155; // rcx
  __int64 v156; // r8
  __int64 v157; // r14
  char v158; // al
  char v159; // dl
  __int64 v160; // r14
  __int64 v161; // r14
  int v162; // eax
  __int64 v163; // rcx
  __int64 v164; // r8
  __int64 v165; // rdx
  char v166; // dl
  char v167; // al
  char v168; // al
  bool v169; // zf
  __int64 v170; // rax
  __int64 (__fastcall *v171)(__int64); // rax
  __int64 v172; // rsi
  __int64 *v173; // rdx
  __int64 v174; // rax
  __int64 *v175; // rdx
  __int64 v176; // rax
  __int64 result; // rax
  __int64 v178; // rdx
  __int64 v179; // rcx
  __int64 v180; // r8
  __int64 v181; // rax
  __int64 v182; // rsi
  unsigned int v183; // edx
  __int64 v184; // rax
  __int64 v185; // rsi
  unsigned int v186; // edx
  __int64 v187; // rax
  int v188; // edi
  int v189; // edx
  int v190; // edi
  int v191; // edi
  __int64 v192; // r11
  unsigned int v193; // ecx
  int v194; // r9d
  int v195; // r10d
  int v196; // r9d
  int v197; // r9d
  __int64 v198; // r11
  int v199; // r10d
  unsigned int v200; // ecx
  int v201; // edi
  _DWORD *v202; // [rsp+8h] [rbp-158h]
  char v203; // [rsp+8h] [rbp-158h]
  _QWORD *v204; // [rsp+10h] [rbp-150h]
  __int64 v205; // [rsp+20h] [rbp-140h]
  __int64 *v206; // [rsp+38h] [rbp-128h] BYREF
  __int64 v207; // [rsp+40h] [rbp-120h] BYREF
  __int64 v208; // [rsp+48h] [rbp-118h] BYREF
  __int64 v209; // [rsp+50h] [rbp-110h] BYREF
  __int64 v210; // [rsp+58h] [rbp-108h]
  __int64 v211; // [rsp+60h] [rbp-100h]
  __int64 v212; // [rsp+68h] [rbp-F8h] BYREF
  __int64 v213; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v214; // [rsp+78h] [rbp-E8h]
  __int64 v215; // [rsp+80h] [rbp-E0h]
  __int64 v216; // [rsp+88h] [rbp-D8h]
  __int64 v217; // [rsp+90h] [rbp-D0h]
  __int64 v218; // [rsp+98h] [rbp-C8h]
  _QWORD v219[16]; // [rsp+A0h] [rbp-C0h] BYREF
  int v220; // [rsp+120h] [rbp-40h]
  char v221; // [rsp+124h] [rbp-3Ch] BYREF

  sub_3446ED0();
  *(_QWORD *)(a1 + 537008) = a2;
  *(_QWORD *)(a1 + 537016) = a3;
  *(_DWORD *)(a1 + 537024) = 0;
  *(_QWORD *)a1 = &unk_4A2E630;
  *(_QWORD *)(a1 + 536968) = -1;
  *(_QWORD *)(a1 + 536976) = -1;
  *(_QWORD *)(a1 + 536996) = -1;
  *(_DWORD *)(a1 + 60) = 2;
  *(_QWORD *)(a1 + 64) = 0x200000002LL;
  sub_2FE6BE0(a1, 1);
  v5 = (_DWORD *)*(unsigned int *)(a1 + 48);
  if ( !(_DWORD)v5 )
  {
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_313;
  }
  v6 = *(_QWORD *)(a1 + 32);
  v7 = 1;
  v8 = 0;
  v9 = ((_WORD)v5 - 1) & 0x940;
  v10 = (_DWORD *)(v6 + 8LL * (((_WORD)v5 - 1) & 0x940));
  v11 = *v10;
  if ( *v10 != 64 )
  {
    while ( v11 != -1 )
    {
      if ( v11 == -2 && !v8 )
        v8 = v10;
      v9 = ((_DWORD)v5 - 1) & (v7 + v9);
      v10 = (_DWORD *)(v6 + 8LL * v9);
      v11 = *v10;
      if ( *v10 == 64 )
        goto LABEL_3;
      v7 = (unsigned int)(v7 + 1);
    }
    v188 = *(_DWORD *)(a1 + 40);
    if ( v8 )
      v10 = v8;
    ++*(_QWORD *)(a1 + 24);
    v189 = v188 + 1;
    if ( 4 * (v188 + 1) < (unsigned int)(3 * (_DWORD)v5) )
    {
      if ( (int)v5 - *(_DWORD *)(a1 + 44) - v189 > (unsigned int)v5 >> 3 )
      {
LABEL_309:
        *(_DWORD *)(a1 + 40) = v189;
        if ( *v10 != -1 )
          --*(_DWORD *)(a1 + 44);
        *(_QWORD *)v10 = 64;
        goto LABEL_3;
      }
      sub_A09770(a1 + 24, (int)v5);
      v196 = *(_DWORD *)(a1 + 48);
      if ( v196 )
      {
        v197 = v196 - 1;
        v198 = *(_QWORD *)(a1 + 32);
        v199 = 1;
        v200 = v197 & 0x940;
        v189 = *(_DWORD *)(a1 + 40) + 1;
        v5 = 0;
        v10 = (_DWORD *)(v198 + 8LL * (v197 & 0x940));
        v201 = *v10;
        if ( *v10 == 64 )
          goto LABEL_309;
        while ( v201 != -1 )
        {
          if ( v201 == -2 && !v5 )
            v5 = v10;
          v7 = (unsigned int)(v199 + 1);
          v200 = v197 & (v199 + v200);
          v10 = (_DWORD *)(v198 + 8LL * v200);
          v201 = *v10;
          if ( *v10 == 64 )
            goto LABEL_309;
          ++v199;
        }
        goto LABEL_317;
      }
      goto LABEL_333;
    }
LABEL_313:
    sub_A09770(a1 + 24, 2 * (_DWORD)v5);
    v190 = *(_DWORD *)(a1 + 48);
    if ( v190 )
    {
      v191 = v190 - 1;
      v192 = *(_QWORD *)(a1 + 32);
      v5 = (_DWORD *)*(unsigned int *)(a1 + 40);
      v193 = v191 & 0x940;
      v189 = (_DWORD)v5 + 1;
      v10 = (_DWORD *)(v192 + 8LL * (v191 & 0x940));
      v194 = *v10;
      if ( *v10 == 64 )
        goto LABEL_309;
      v195 = 1;
      v5 = 0;
      while ( v194 != -1 )
      {
        if ( !v5 && v194 == -2 )
          v5 = v10;
        v7 = (unsigned int)(v195 + 1);
        v193 = v191 & (v195 + v193);
        v10 = (_DWORD *)(v192 + 8LL * v193);
        v194 = *v10;
        if ( *v10 == 64 )
          goto LABEL_309;
        ++v195;
      }
LABEL_317:
      if ( v5 )
        v10 = v5;
      goto LABEL_309;
    }
LABEL_333:
    ++*(_DWORD *)(a1 + 40);
    BUG();
  }
LABEL_3:
  v10[1] = 32;
  v12 = 1 - (((_BYTE)qword_502B588 == 0) - 1);
  *(_BYTE *)(a1 + 70072) = 4;
  v13 = a3[85] <= 0x12Bu;
  *(_BYTE *)(a1 + 72) = v12;
  *(_QWORD *)(a1 + 176) = &off_4A2FA40;
  *(_QWORD *)(a1 + 160) = &off_4A2FCE0;
  *(_QWORD *)(a1 + 128) = &off_4A2FD40;
  *(_QWORD *)(a1 + 208) = &off_4A2FB60;
  *(_QWORD *)(a1 + 488) = &off_4A2FC20;
  *(_QWORD *)(a1 + 408) = &off_4A2FC20;
  *(_QWORD *)(a1 + 168) = &off_4A2FC20;
  *(_QWORD *)(a1 + 1128) = &off_4A2FC20;
  *(_QWORD *)(a1 + 1216) = &off_4A2FC20;
  *(_QWORD *)(a1 + 216) = &off_4A2F980;
  *(_QWORD *)(a1 + 200) = &off_4A2FCE0;
  *(_QWORD *)(a1 + 192) = &off_4A2FCE0;
  *(_WORD *)(a1 + 70070) = 516;
  *(_BYTE *)(a1 + 70079) = 2;
  *(_BYTE *)(a1 + 10740) = 0;
  if ( !v13 && a3[84] > 0x1Fu )
    *(_BYTE *)(a1 + 10741) = 0;
  *(_BYTE *)(a1 + 12122) = sub_305B500(a3, v5, &off_4A2FCE0, &off_4A2F980, v7) ^ 1;
  v17 = sub_305B500(a3, v5, v14, v15, v16);
  *(_BYTE *)(a1 + 36462) = 4;
  *(_BYTE *)(a1 + 38462) = 4;
  *(_BYTE *)(a1 + 75572) = 4;
  *(_BYTE *)(a1 + 75579) = 2;
  *(_BYTE *)(a1 + 70122) = 2 * (v17 == 0);
  *(_WORD *)(a1 + 75570) = 516;
  *(_BYTE *)(a1 + 75622) = 2 * ((unsigned __int8)sub_305B520(a3, 208) == 0);
  v204 = (_QWORD *)(a1 + 525240);
  v205 = a1 + 525248;
  if ( (unsigned __int8)sub_305B520(a3, 208) )
  {
    *(_BYTE *)(a1 + 11622) = 0;
  }
  else
  {
    v18 = *(_QWORD *)(a1 + 525256);
    *(_BYTE *)(a1 + 11622) = 1;
    WORD2(v207) = 10;
    v19 = a1 + 525248;
    LODWORD(v207) = 208;
    if ( !v18 )
      goto LABEL_298;
    do
    {
      v20 = *(_DWORD *)(v18 + 32);
      if ( v20 <= 0xCF || v20 == 208 && *(_WORD *)(v18 + 36) <= 9u )
      {
        v18 = *(_QWORD *)(v18 + 24);
      }
      else
      {
        v19 = v18;
        v18 = *(_QWORD *)(v18 + 16);
      }
    }
    while ( v18 );
    if ( v19 == v205 || *(_DWORD *)(v19 + 32) > 0xD0u || *(_DWORD *)(v19 + 32) == 208 && *(_WORD *)(v19 + 36) > 0xAu )
    {
LABEL_298:
      v206 = &v207;
      v19 = sub_3052EE0(v204, v19, &v206);
    }
    *(_WORD *)(v19 + 40) = 12;
  }
  *(_BYTE *)(a1 + 30072) = 4;
  v21 = 189;
  *(_WORD *)(a1 + 30070) = 516;
  *(_WORD *)(a1 + 25070) = 1028;
  v207 = 0x38000000BDLL;
  v208 = 0x4600000044LL;
  v209 = 0xC7000000C9LL;
  v210 = 0xC6000000C8LL;
  v211 = 0xE3000000E2LL;
  v212 = 0xC4000000C3LL;
  v213 = 0xAD0000003ALL;
  v214 = 0xCA000000ACLL;
  v215 = 0xC2000000C1LL;
  v216 = 0x4A0000004CLL;
  v217 = 0x3B00000052LL;
  v218 = 0xCF00000041LL;
  v219[0] = 0xBE000000D0LL;
  v219[1] = 0xB5000000DCLL;
  v219[2] = 0x50000000B4LL;
  v219[3] = 0xBF0000003FLL;
  v219[4] = 0xC00000003DLL;
  v219[5] = 0x4E00000056LL;
  v219[6] = 0x540000004BLL;
  v219[7] = 0x4500000039LL;
  v219[8] = 0x4D00000047LL;
  v219[9] = 0x5300000048LL;
  v219[10] = 0x420000003CLL;
  v219[11] = 0xB7000000DDLL;
  v219[12] = 0x51000000B6LL;
  *(_BYTE *)(a1 + 30079) = 2;
  *(_BYTE *)(a1 + 25072) = 4;
  *(_BYTE *)(a1 + 25079) = 4;
  *(_BYTE *)(a1 + 24148) = 4;
  v219[13] = 0x3E00000040LL;
  v219[14] = 0x4F00000057LL;
  v219[15] = 0xCE00000049LL;
  v22 = &v207;
  v220 = 85;
  while ( 1 )
  {
    v22 = (__int64 *)((char *)v22 + 4);
    *(_BYTE *)(a1 + v21 + 24914) = 2;
    if ( v22 == (__int64 *)&v221 )
      break;
    v21 = *(unsigned int *)v22;
  }
  v23 = (unsigned __int16 *)&unk_44C7B60;
  for ( i = 10; ; i = *v23 )
  {
    ++v23;
    v25 = a1 + 500 * i;
    *(_BYTE *)(v25 + 6621) = 2;
    *(_BYTE *)(v25 + 6720) = 2;
    if ( v23 == (unsigned __int16 *)&unk_44C7B7A )
      break;
  }
  v207 = 0x7002F00060005LL;
  v26 = &v207;
  v27 = 5;
  *(_BYTE *)(a1 + 10636) = 0;
  *(_BYTE *)(a1 + 10136) = 0;
  *(_BYTE *)(a1 + 9636) = 0;
  *(_BYTE *)(a1 + 9136) = 0;
  *(_BYTE *)(a1 + 7636) = 2;
  *(_BYTE *)(a1 + 30136) = 2;
  *(_WORD *)(a1 + 10124) = 1028;
  *(_BYTE *)(a1 + 10126) = 4;
  *(_WORD *)(a1 + 10624) = 1028;
  *(_BYTE *)(a1 + 10626) = 4;
  *(_BYTE *)(a1 + 10115) = 0;
  *(_BYTE *)(a1 + 10615) = 0;
  LOWORD(v208) = 8;
  while ( 1 )
  {
    v26 = (__int64 *)((char *)v26 + 2);
    *(_WORD *)(a1 + 500 * v27 + 6607) = 514;
    if ( v26 == (__int64 *)((char *)&v208 + 2) )
      break;
    v27 = *(unsigned __int16 *)v26;
  }
  if ( a3[86] > 0x1Fu )
  {
    v28 = 193;
    v207 = 0xC2000000C1LL;
    v208 = 0xC4000000C3LL;
    v29 = &v207;
    *(_WORD *)(a1 + 10109) = 0;
    while ( 1 )
    {
      v29 = (__int64 *)((char *)v29 + 4);
      *(_BYTE *)(a1 + v28 + 10414) = 4;
      if ( v29 == &v209 )
        break;
      v28 = *(unsigned int *)v29;
    }
  }
  v30 = *(_WORD *)(a1 + 150012);
  *(_BYTE *)(a1 + 9611) = 2;
  *(_WORD *)(a1 + 7216) = 1026;
  LOBYTE(v30) = v30 & 0xF;
  *(_BYTE *)(a1 + 9927) = 4;
  *(_BYTE *)(a1 + 10427) = 4;
  *(_WORD *)(a1 + 150012) = v30 | 0x20;
  v31 = *(_WORD *)(a1 + 150560);
  *(_WORD *)(a1 + 6960) = 1028;
  LOBYTE(v31) = v31 & 0xF;
  *(_BYTE *)(a1 + 6962) = 4;
  *(_WORD *)(a1 + 150560) = v31 | 0x20;
  v32 = *(_WORD *)(a1 + 150010);
  LOBYTE(v32) = v32 & 0xF;
  *(_WORD *)(a1 + 150010) = v32 | 0x20;
  v33 = *(_WORD *)(a1 + 150558);
  LOBYTE(v33) = v33 & 0xF;
  *(_WORD *)(a1 + 150558) = v33 | 0x20;
  v34 = *(_WORD *)(a1 + 150562);
  LOBYTE(v34) = v34 & 0xF;
  *(_WORD *)(a1 + 150562) = v34 | 0x20;
  v35 = *(_WORD *)(a1 + 224224);
  LOBYTE(v35) = v35 & 0xF;
  *(_WORD *)(a1 + 224224) = v35 | 0x20;
  v36 = *(_WORD *)(a1 + 235184);
  LOBYTE(v36) = v36 & 0xF;
  *(_WORD *)(a1 + 235184) = v36 | 0x20;
  v37 = *(_WORD *)(a1 + 224246);
  LOBYTE(v37) = v37 & 0xF;
  *(_WORD *)(a1 + 224246) = v37 | 0x20;
  v38 = *(_WORD *)(a1 + 235206);
  LOBYTE(v38) = v38 & 0xF;
  *(_WORD *)(a1 + 235206) = v38 | 0x20;
  v39 = *(_WORD *)(a1 + 235224);
  LOBYTE(v39) = v39 & 0xF;
  *(_WORD *)(a1 + 235224) = v39 | 0x20;
  v40 = *(_WORD *)(a1 + 225324);
  LOBYTE(v40) = v40 & 0xF;
  *(_WORD *)(a1 + 225324) = v40 | 0x20;
  v41 = *(_WORD *)(a1 + 236284);
  LOBYTE(v41) = v41 & 0xF;
  *(_WORD *)(a1 + 236284) = v41 | 0x20;
  v42 = *(_WORD *)(a1 + 225346);
  LOBYTE(v42) = v42 & 0xF;
  *(_WORD *)(a1 + 225346) = v42 | 0x20;
  v43 = *(_WORD *)(a1 + 236306);
  *(_WORD *)(a1 + 447016) = 514;
  LOBYTE(v43) = v43 & 0xF;
  *(_WORD *)(a1 + 447290) = 514;
  *(_BYTE *)(a1 + 447292) = 2;
  *(_WORD *)(a1 + 236306) = v43 | 0x20;
  v44 = *(_WORD *)(a1 + 236324);
  *(_WORD *)(a1 + 7712) = 1028;
  LOBYTE(v44) = v44 & 0xF;
  *(_WORD *)(a1 + 236324) = v44 | 0x20;
  v45 = *(_WORD *)(a1 + 227518);
  LOBYTE(v45) = v45 & 0xF;
  *(_WORD *)(a1 + 227518) = v45 | 0x20;
  v46 = *(_WORD *)(a1 + 236834);
  LOBYTE(v46) = v46 & 0xF;
  *(_WORD *)(a1 + 236834) = v46 | 0x20;
  v47 = *(_WORD *)(a1 + 227540);
  LOBYTE(v47) = v47 & 0xF;
  *(_WORD *)(a1 + 227540) = v47 | 0x20;
  v48 = *(_WORD *)(a1 + 236856);
  LOBYTE(v48) = v48 & 0xF;
  *(_WORD *)(a1 + 236856) = v48 | 0x20;
  for ( j = 0; j != 2192; j += 274 )
  {
    *(_WORD *)(a1 + 2 * j + 144514) = *(_WORD *)(a1 + 2 * j + 144514) & 0xF | 0x1110;
    *(_BYTE *)(a1 + j + 444268) = 2;
  }
  v50 = &v207;
  v207 = 0x1100000016LL;
  v208 = 0xD0000000BLL;
  v209 = 0xC0000000ALL;
  v210 = 0x1400000012LL;
  v211 = 0x1500000013LL;
  for ( k = 22; ; k = *(int *)v50 )
  {
    v50 = (__int64 *)((char *)v50 + 4);
    v52 = a1 + 140 * k;
    v53 = *(_DWORD *)(v52 + 521536);
    BYTE1(v53) = BYTE1(v53) & 0xF0 | 2;
    *(_DWORD *)(v52 + 521536) = v53;
    if ( v50 == &v212 )
      break;
  }
  LOBYTE(v54) = 1;
  v207 = 0x200000001LL;
  v55 = &v207;
  LODWORD(v208) = 3;
  v56 = *(unsigned __int16 *)(a1 + 169240);
  while ( 1 )
  {
    v55 = (__int64 *)((char *)v55 + 4);
    v56 = (2 << (4 * v54)) | ~(15 << (4 * v54)) & v56;
    if ( v55 == (__int64 *)((char *)&v208 + 4) )
      break;
    v54 = *(_DWORD *)v55;
  }
  *(_WORD *)(a1 + 169240) = v56;
  v57 = (_BYTE *)(a1 + 14960);
  *(_BYTE *)(a1 + 456631) = 2;
  LOWORD(v58) = 17;
  *(_BYTE *)(a1 + 12926) = 0;
  *(_BYTE *)(a1 + 12426) = 0;
  *(_BYTE *)(a1 + 11926) = 0;
  *(_BYTE *)(a1 + 11426) = 0;
  *(_BYTE *)(a1 + 10214) = 4;
  *(_BYTE *)(a1 + 10714) = 4;
  *(_WORD *)(a1 + 7227) = 1028;
  *(_WORD *)(a1 + 7245) = 0;
  do
  {
    if ( (unsigned __int8)sub_302E500(v58) )
    {
      v57[252] = 4;
      v57[253] = 4;
      v57[1] = 4;
      *v57 = 4;
    }
    v58 = v59 + 1;
    v57 += 500;
  }
  while ( v58 != 176 );
  v60 = &v207;
  v207 = 0x40003E003C003ALL;
  v208 = 0x46004500440042LL;
  v209 = 0x4A004900480047LL;
  v210 = 0x4E004D004C004BLL;
  v211 = 0x5200510050004FLL;
  v212 = 0x56005500540053LL;
  v213 = 0x5A005900580057LL;
  v214 = 0x5E005D005C005BLL;
  v215 = 0x6200610060005FLL;
  v216 = 0x66006500640063LL;
  v217 = 0x6A006900680067LL;
  v218 = 0x6E006D006C006BLL;
  for ( m = 58; ; m = *(unsigned __int16 *)v60 )
  {
    v60 = (__int64 *)((char *)v60 + 2);
    *(_BYTE *)(a1 + 500 * m + 6461) = 4;
    if ( v60 == v219 )
      break;
  }
  v62 = &v207;
  LODWORD(v208) = 10485919;
  v207 = 0x9E009900950093LL;
  WORD2(v208) = 161;
  for ( n = 147; ; n = *(unsigned __int16 *)v62 )
  {
    v62 = (__int64 *)((char *)v62 + 2);
    *(_BYTE *)(a1 + 500 * n + 6461) = 4;
    if ( v62 == (__int64 *)((char *)&v208 + 6) )
      break;
  }
  *(_BYTE *)(a1 + 9231) = 4;
  v64 = &v206;
  WORD2(v206) = 8;
  v65 = 6;
  v207 = 0xB4000000BDLL;
  *(_DWORD *)(a1 + 7231) = (_DWORD)&unk_4020204;
  *(_BYTE *)(a1 + 8961) = 4;
  *(_BYTE *)(a1 + 10961) = 4;
  *(_BYTE *)(a1 + 11254) = 4;
  *(_BYTE *)(a1 + 11256) = 4;
  LODWORD(v206) = 458758;
  v208 = 0xB6000000B5LL;
  LODWORD(v209) = 183;
  while ( 1 )
  {
    v66 = 189;
    v67 = &v207;
    v68 = a1 + 500 * v65;
    while ( 1 )
    {
      v67 = (__int64 *)((char *)v67 + 4);
      *(_BYTE *)(v66 + v68 + 6414) = 0;
      if ( v67 == (__int64 *)((char *)&v209 + 4) )
        break;
      v66 = *(unsigned int *)v67;
    }
    v64 = (__int64 **)((char *)v64 + 2);
    if ( (__int64 **)((char *)&v206 + 6) == v64 )
      break;
    v65 = *(unsigned __int16 *)v64;
  }
  v13 = a3[85] <= 0x383u;
  *(_BYTE *)(a1 + 9618) = 1;
  *(_WORD *)(a1 + 10113) = 0;
  v69 = a3[84];
  *(_WORD *)(a1 + 9613) = 257;
  *(_WORD *)(a1 + 10613) = 1028;
  *(_BYTE *)(a1 + 30103) = 4;
  if ( v13 || v69 <= 0x4F )
  {
    *(_DWORD *)(a1 + 30094) = (_DWORD)&unk_4040404;
    *(_WORD *)(a1 + 30113) = 514;
    v70 = 4;
  }
  else
  {
    v70 = 0;
    *(_DWORD *)(a1 + 30094) = 0;
    *(_WORD *)(a1 + 30113) = 514;
  }
  *(_BYTE *)(a1 + 29970) = v70;
  v207 = 0x3C0000003BLL;
  v71 = 59;
  v208 = 0xC0000000BFLL;
  v209 = 0xAC000000ADLL;
  v210 = 0xE3000000E2LL;
  v211 = 0xDD000000DCLL;
  v72 = &v207;
  *(_WORD *)(a1 + 29971) = 1028;
  *(_BYTE *)(a1 + 30104) = 4;
  *(_WORD *)(a1 + 29975) = 1028;
  LODWORD(v212) = 208;
  while ( 1 )
  {
    v72 = (__int64 *)((char *)v72 + 4);
    *(_BYTE *)(a1 + v71 + 29914) = 2;
    if ( v72 == (__int64 *)((char *)&v212 + 4) )
      break;
    v71 = *(unsigned int *)v72;
  }
  *(_DWORD *)(a1 + 9982) = 0;
  if ( v69 > 0x2A )
    *(_DWORD *)(a1 + 10482) = 0;
  *(_BYTE *)(a1 + 9612) = 2;
  v73 = &v207;
  v207 = 0xBA00000038LL;
  v74 = 56;
  v208 = 0xD8000000EALL;
  v75 = 1;
  v209 = 0x3A00000060LL;
  v210 = 0x3D000000BELL;
  v211 = 0xCE0000003ELL;
  *(_BYTE *)(a1 + 30112) = 2;
  *(_BYTE *)(a1 + 10112) = 2;
  *(_BYTE *)(a1 + 10612) = 2;
  *(_BYTE *)(a1 + 7619) = 4;
  *(_WORD *)(a1 + 10477) = 514;
  v212 = 0xEB0000009CLL;
  while ( 1 )
  {
    v76 = v74;
    v77 = v74 & 7;
    v73 = (__int64 *)((char *)v73 + 4);
    *(_BYTE *)(a1 + (v76 >> 3) + 525170) |= 1 << v77;
    if ( v73 == &v213 )
      break;
    v74 = *(_DWORD *)v73;
  }
  if ( (unsigned __int8)sub_305B500(a3, 1, v73, v77, &v207) || a3[86] > 0x4Fu )
    *(_BYTE *)(a1 + 525196) |= 1u;
  v81 = 96;
  v82 = (unsigned int *)&unk_44C7B50;
  v83 = 96;
  v84 = sub_305B500(a3, 1, v78, v79, v80);
  while ( 1 )
  {
    *(_BYTE *)(a1 + v83 + 11914) = v84 ^ 1;
    v88 = sub_305B500(a3, v75, v85, v86, v87);
    if ( (_DWORD)v81 == 266 )
    {
      v89 = 2;
      if ( a3[85] <= 0x2EDu )
        goto LABEL_75;
LABEL_73:
      v88 &= a3[84] > 0x45u;
LABEL_74:
      v89 = 2 * (v88 == 0);
      goto LABEL_75;
    }
    if ( (unsigned int)(v81 - 279) > 5 )
      goto LABEL_74;
    v89 = 2;
    if ( a3[85] > 0x31Fu )
      goto LABEL_73;
LABEL_75:
    v90 = *v82;
    *(_BYTE *)(a1 + v81 + 69914) = v89;
    *(_BYTE *)(a1 + v90 + 75414) = 2 * ((unsigned __int8)sub_305B520(a3, (unsigned int)v90) == 0);
    v91 = *v82;
    v75 = v91;
    v92 = sub_305B520(a3, v91);
    v95 = *v82;
    *(_BYTE *)(a1 + v91 + 11414) = v92 ^ 1;
    if ( (unsigned int)v95 <= 0x1F3 && *(_BYTE *)(a1 + (unsigned int)v95 + 11414) == 1 )
    {
      v136 = *(_QWORD *)(a1 + 525256);
      LODWORD(v207) = v95;
      WORD2(v207) = 10;
      v75 = a1 + 525248;
      if ( !v136 )
        goto LABEL_184;
      do
      {
        while ( (unsigned int)v95 <= *(_DWORD *)(v136 + 32)
             && ((_DWORD)v95 != *(_DWORD *)(v136 + 32) || *(_WORD *)(v136 + 36) > 9u) )
        {
          v75 = v136;
          v136 = *(_QWORD *)(v136 + 16);
          if ( !v136 )
            goto LABEL_182;
        }
        v136 = *(_QWORD *)(v136 + 24);
      }
      while ( v136 );
LABEL_182:
      if ( v75 == v205
        || (unsigned int)v95 < *(_DWORD *)(v75 + 32)
        || (_DWORD)v95 == *(_DWORD *)(v75 + 32) && *(_WORD *)(v75 + 36) > 0xAu )
      {
LABEL_184:
        v206 = &v207;
        v75 = sub_3052EE0(v204, v75, &v206);
      }
      *(_WORD *)(v75 + 40) = 12;
    }
    if ( ++v82 == (unsigned int *)&unk_44C7B60 )
      break;
    v83 = *v82;
    v84 = sub_305B500(a3, v75, v95, v93, v94);
    if ( (_DWORD)v83 == 266 )
    {
      if ( a3[85] > 0x2EDu )
        goto LABEL_81;
    }
    else
    {
      v85 = (unsigned int)(v83 - 279);
      if ( (unsigned int)v85 > 5 )
        goto LABEL_82;
      if ( a3[85] > 0x31Fu )
      {
LABEL_81:
        LOBYTE(v85) = a3[84] > 0x45u;
        v84 &= v85;
        goto LABEL_82;
      }
    }
    v84 = 0;
LABEL_82:
    v81 = *v82;
  }
  v96 = (unsigned int *)&unk_44C7B40;
  for ( ii = 96; ; ii = *v96 )
  {
    v98 = (unsigned __int16 *)&unk_44C7B38;
    if ( (unsigned __int8)sub_305B520(a3, ii) )
    {
LABEL_90:
      if ( ++v98 == (unsigned __int16 *)&unk_44C7B3C )
        goto LABEL_94;
      goto LABEL_91;
    }
    while ( 1 )
    {
      ii = 150;
      if ( !(unsigned __int8)sub_305B520(a3, 150) )
        goto LABEL_90;
      v102 = *v98;
      v99 = *v96;
      ++v98;
      *(_BYTE *)(v99 + a1 + 500 * v102 + 6414) = 4;
      if ( v98 == (unsigned __int16 *)&unk_44C7B3C )
        break;
LABEL_91:
      ii = *v96;
      if ( (unsigned __int8)sub_305B520(a3, ii) )
        goto LABEL_90;
    }
LABEL_94:
    if ( &unk_44C7B4C == (_UNKNOWN *)++v96 )
      break;
  }
  if ( a3[85] > 0x211u && a3[84] > 0x3Bu && (unsigned __int8)sub_305B500(a3, ii, v99, v100, v101) )
    v103 = 0;
  else
    v103 = 2;
  *(_BYTE *)(a1 + 12158) = v103;
  v104 = (unsigned int *)&unk_44C7B20;
  *(_BYTE *)(a1 + 70158) = v103;
  *(_BYTE *)(a1 + 11658) = 2 * ((unsigned __int8)sub_305B520(a3, 244) == 0);
  v105 = 268;
  *(_BYTE *)(a1 + 75658) = 2 * ((unsigned __int8)sub_305B520(a3, 244) == 0);
  while ( 1 )
  {
    *(_BYTE *)(a1 + (unsigned int)v105 + 11914) = 0;
    *(_BYTE *)(a1 + (unsigned int)v105 + 12414) = 0;
    *(_BYTE *)(a1 + (unsigned int)v105 + 12914) = 0;
    *(_BYTE *)(a1 + (unsigned int)v105 + 69914) = 2;
    *(_BYTE *)(a1 + (unsigned int)v105 + 75414) = 2;
    v106 = sub_305B520(a3, v105);
    v108 = *v104;
    *(_BYTE *)(a1 + (unsigned int)v105 + 11414) = v106 ^ 1;
    if ( v108 <= 0x1F3 && *(_BYTE *)(a1 + v108 + 11414) == 1 )
    {
      v121 = *(_QWORD *)(a1 + 525256);
      v107 = 10;
      LODWORD(v207) = v108;
      WORD2(v207) = 10;
      v122 = a1 + 525248;
      if ( !v121 )
        goto LABEL_146;
      do
      {
        while ( v108 <= *(_DWORD *)(v121 + 32) && (v108 != *(_DWORD *)(v121 + 32) || *(_WORD *)(v121 + 36) > 9u) )
        {
          v122 = v121;
          v121 = *(_QWORD *)(v121 + 16);
          if ( !v121 )
            goto LABEL_144;
        }
        v121 = *(_QWORD *)(v121 + 24);
      }
      while ( v121 );
LABEL_144:
      if ( v122 == v205
        || v108 < *(_DWORD *)(v122 + 32)
        || v108 == *(_DWORD *)(v122 + 32) && *(_WORD *)(v122 + 36) > 0xAu )
      {
LABEL_146:
        v206 = &v207;
        v122 = sub_3052EE0(v204, v122, &v206);
      }
      *(_WORD *)(v122 + 40) = 12;
    }
    if ( &unk_44C7B38 == (_UNKNOWN *)++v104 )
      break;
    v105 = *v104;
  }
  v109 = a3[85];
  if ( v109 <= 0x31F || a3[84] <= 0x46u )
    *(_BYTE *)(a1 + 12654) = 2;
  if ( v109 <= 0x383 || a3[84] <= 0x4Du )
  {
    v118 = 10;
    for ( jj = (unsigned __int16 *)&unk_44C7B10; ; v118 = *jj )
    {
      ++jj;
      v120 = a1 + 500 * v118;
      *(_BYTE *)(v120 + 6647) = 4;
      *(_BYTE *)(v120 + 6644) = 4;
      if ( jj == (unsigned __int16 *)&unk_44C7B16 )
        break;
    }
    if ( v109 <= 0x383 || a3[84] <= 0x4Du )
    {
      v129 = (unsigned __int16 *)&unk_44C7B08;
      v107 = (__int64)&v207;
      for ( kk = 2; ; kk = *v129 )
      {
        v207 = 0xDD000000DCLL;
        v131 = 220;
        v132 = &v207;
        v208 = 0xE3000000E2LL;
        v133 = a1 + 500 * kk;
        while ( 1 )
        {
          v132 = (__int64 *)((char *)v132 + 4);
          *(_BYTE *)(v131 + v133 + 6414) = 4;
          if ( v132 == &v209 )
            break;
          v131 = *(unsigned int *)v132;
        }
        if ( &unk_44C7B10 == (_UNKNOWN *)++v129 )
          break;
      }
      v134 = 220;
      v207 = 0xDD000000DCLL;
      v208 = 0xE3000000E2LL;
      for ( mm = &v207; ; v134 = *(unsigned int *)mm )
      {
        mm = (__int64 *)((char *)mm + 4);
        *(_BYTE *)(a1 + v134 + 11414) = 4;
        if ( mm == &v209 )
          break;
      }
    }
  }
  v110 = *(_QWORD *)(a1 + 525256);
  v111 = 10;
  *(_BYTE *)(a1 + 12186) = 1;
  *(_BYTE *)(a1 + 70186) = 2;
  v112 = a1 + 525248;
  *(_BYTE *)(a1 + 75686) = 2;
  *(_BYTE *)(a1 + 12686) = 4;
  *(_BYTE *)(a1 + 13186) = 4;
  *(_BYTE *)(a1 + 11686) = 1;
  LODWORD(v207) = 272;
  WORD2(v207) = 10;
  if ( !v110 )
    goto LABEL_117;
  do
  {
    while ( 1 )
    {
      v113 = *(_DWORD *)(v110 + 32);
      if ( v113 <= 0x10F || v113 == 272 && *(_WORD *)(v110 + 36) <= 9u )
        break;
      v112 = v110;
      v110 = *(_QWORD *)(v110 + 16);
      if ( !v110 )
        goto LABEL_115;
    }
    v110 = *(_QWORD *)(v110 + 24);
  }
  while ( v110 );
LABEL_115:
  if ( v112 == v205 || *(_DWORD *)(v112 + 32) > 0x110u || *(_DWORD *)(v112 + 32) == 272 && *(_WORD *)(v112 + 36) > 0xAu )
  {
LABEL_117:
    v206 = &v207;
    v112 = sub_3052EE0(v204, v112, &v206);
  }
  v202 = a3;
  *(_WORD *)(v112 + 40) = 12;
  v114 = (unsigned int *)&unk_44C7AF0;
  v115 = 99;
  *(_BYTE *)(a1 + 12066) = 2;
  *(_BYTE *)(a1 + 70066) = 2;
  *(_BYTE *)(a1 + 11566) = 2;
  *(_BYTE *)(a1 + 75566) = 2;
  *(_BYTE *)(a1 + 12566) = 4;
  *(_BYTE *)(a1 + 13066) = 4;
  while ( 1 )
  {
    LODWORD(v207) = v115;
    v116 = a1 + 525248;
    *(_BYTE *)(a1 + (unsigned int)v115 + 11914) = 1;
    *(_BYTE *)(a1 + (unsigned int)v115 + 12414) = 0;
    *(_BYTE *)(a1 + (unsigned int)v115 + 12914) = 0;
    *(_BYTE *)(a1 + (unsigned int)v115 + 69914) = 2;
    *(_BYTE *)(a1 + (unsigned int)v115 + 75414) = 2;
    *(_BYTE *)(a1 + (unsigned int)v115 + 11414) = 1;
    WORD2(v207) = 10;
    v117 = *(_QWORD *)(a1 + 525256);
    if ( !v117 )
      goto LABEL_130;
    do
    {
      while ( *(_DWORD *)(v117 + 32) >= (unsigned int)v115
           && (*(_DWORD *)(v117 + 32) != (_DWORD)v115 || *(_WORD *)(v117 + 36) > 9u) )
      {
        v116 = v117;
        v117 = *(_QWORD *)(v117 + 16);
        if ( !v117 )
          goto LABEL_126;
      }
      v117 = *(_QWORD *)(v117 + 24);
    }
    while ( v117 );
LABEL_126:
    if ( v116 == v205
      || *(_DWORD *)(v116 + 32) > (unsigned int)v115
      || *(_DWORD *)(v116 + 32) == (_DWORD)v115 && *(_WORD *)(v116 + 36) > 0xAu )
    {
LABEL_130:
      v206 = &v207;
      v116 = sub_3052EE0(v204, v116, &v206);
    }
    ++v114;
    *(_WORD *)(v116 + 40) = 12;
    if ( v114 == (unsigned int *)&unk_44C7B04 )
      break;
    v115 = *v114;
  }
  v123 = v202;
  *(_BYTE *)(a1 + 12659) = 0;
  *(_BYTE *)(a1 + 13159) = 0;
  if ( v202[84] > 0x40u )
  {
    *(_BYTE *)(a1 + 12159) = sub_305B500(v202, v116, v115, v111, v107) ^ 1;
    *(_BYTE *)(a1 + 70159) = 2 * ((unsigned __int8)sub_305B500(v202, v116, v178, v179, v180) == 0);
  }
  else
  {
    *(_BYTE *)(a1 + 12159) = 1;
    *(_BYTE *)(a1 + 70159) = 2;
  }
  v124 = 245;
  *(_BYTE *)(a1 + 75659) = 2 * ((unsigned __int8)sub_305B520(v202, 245) == 0);
  if ( (unsigned __int8)sub_305B520(v202, 245) )
  {
    *(_BYTE *)(a1 + 11659) = 0;
  }
  else
  {
    v128 = *(_QWORD *)(a1 + 525256);
    *(_BYTE *)(a1 + 11659) = 1;
    WORD2(v207) = 10;
    v124 = a1 + 525248;
    LODWORD(v207) = 245;
    if ( !v128 )
      goto LABEL_160;
    do
    {
      v125 = *(unsigned int *)(v128 + 32);
      if ( (unsigned int)v125 <= 0xF4 || (_DWORD)v125 == 245 && *(_WORD *)(v128 + 36) <= 9u )
      {
        v128 = *(_QWORD *)(v128 + 24);
      }
      else
      {
        v124 = v128;
        v128 = *(_QWORD *)(v128 + 16);
      }
    }
    while ( v128 );
    if ( v124 == v205 || *(_DWORD *)(v124 + 32) > 0xF5u || *(_DWORD *)(v124 + 32) == 245 && *(_WORD *)(v124 + 36) > 0xAu )
    {
LABEL_160:
      v206 = &v207;
      v124 = sub_3052EE0(v204, v124, &v206);
    }
    *(_WORD *)(v124 + 40) = 12;
  }
  v137 = (unsigned int *)&unk_44C7AE8;
  v138 = 279;
  while ( 2 )
  {
    *(_BYTE *)(a1 + v138 + 12414) = 0;
    *(_BYTE *)(a1 + v138 + 12914) = 0;
    v139 = sub_305B500(v202, v124, v125, v126, v127);
    v142 = v139;
    if ( v138 == 266 )
    {
      v143 = 1;
      if ( v202[85] > 0x2EDu )
        goto LABEL_191;
    }
    else
    {
      v143 = v139 ^ 1;
      if ( v138 - 279 <= 5 )
      {
        v143 = 1;
        if ( v202[85] > 0x31Fu )
LABEL_191:
          v143 = !((unsigned __int8)v142 & (v202[84] > 0x45u));
      }
    }
    *(_BYTE *)(a1 + v138 + 11914) = v143;
    v144 = *v137;
    v145 = sub_305B500(v202, v124, v142, v140, v141);
    if ( (_DWORD)v144 == 266 )
    {
      v146 = 2;
      if ( v202[85] <= 0x2EDu )
        goto LABEL_197;
LABEL_195:
      v145 &= v202[84] > 0x45u;
      goto LABEL_196;
    }
    if ( (unsigned int)(v144 - 279) > 5 )
    {
LABEL_196:
      v146 = 2 * (v145 == 0);
      goto LABEL_197;
    }
    v146 = 2;
    if ( v202[85] > 0x31Fu )
      goto LABEL_195;
LABEL_197:
    *(_BYTE *)(a1 + v144 + 69914) = v146;
    v147 = *v137;
    *(_BYTE *)(a1 + v147 + 75414) = 2 * ((unsigned __int8)sub_305B520(v202, v147) == 0);
    v148 = *v137;
    v124 = v148;
    v149 = sub_305B520(v202, v148);
    v125 = *v137;
    *(_BYTE *)(a1 + v148 + 11414) = v149 ^ 1;
    if ( (unsigned int)v125 <= 0x1F3 && *(_BYTE *)(a1 + (unsigned int)v125 + 11414) == 1 )
    {
      LODWORD(v207) = v125;
      v124 = a1 + 525248;
      WORD2(v207) = 10;
      v187 = *(_QWORD *)(a1 + 525256);
      if ( !v187 )
        goto LABEL_281;
      do
      {
        while ( (unsigned int)v125 <= *(_DWORD *)(v187 + 32)
             && ((_DWORD)v125 != *(_DWORD *)(v187 + 32) || *(_WORD *)(v187 + 36) > 9u) )
        {
          v124 = v187;
          v187 = *(_QWORD *)(v187 + 16);
          if ( !v187 )
            goto LABEL_279;
        }
        v187 = *(_QWORD *)(v187 + 24);
      }
      while ( v187 );
LABEL_279:
      if ( v124 == v205
        || (unsigned int)v125 < *(_DWORD *)(v124 + 32)
        || (_DWORD)v125 == *(_DWORD *)(v124 + 32) && *(_WORD *)(v124 + 36) > 0xAu )
      {
LABEL_281:
        v206 = &v207;
        v124 = sub_3052EE0(v204, v124, &v206);
      }
      *(_WORD *)(v124 + 40) = 12;
    }
    if ( &unk_44C7AF0 != (_UNKNOWN *)++v137 )
    {
      v138 = *v137;
      continue;
    }
    break;
  }
  v150 = 0;
  if ( v202[85] > 0x31Fu )
    v150 = v202[84] > 0x45u;
  v151 = (unsigned int *)&unk_44C7AE0;
  v152 = 283;
  v203 = !v150 ? 2 : 0;
  while ( 2 )
  {
    *(_BYTE *)(a1 + v152 + 12414) = v203;
    v153 = sub_305B500(v123, v124, v125, v126, v127);
    if ( v152 == 266 )
    {
      v154 = 2;
      if ( v123[85] <= 0x2EDu )
        goto LABEL_209;
LABEL_207:
      v153 &= v123[84] > 0x45u;
LABEL_208:
      LOBYTE(v154) = v153 == 0;
      v154 = (unsigned int)(2 * v154);
      goto LABEL_209;
    }
    if ( v152 - 279 > 5 )
      goto LABEL_208;
    v154 = 2;
    if ( v123[85] > 0x31Fu )
      goto LABEL_207;
LABEL_209:
    *(_BYTE *)(a1 + v152 + 11914) = v154;
    v157 = *v151;
    v158 = sub_305B500(v123, v124, v154, v155, v156);
    if ( (_DWORD)v157 == 266 )
    {
      v159 = 2;
      if ( v123[85] <= 0x2EDu )
        goto LABEL_214;
LABEL_212:
      v158 &= v123[84] > 0x45u;
LABEL_213:
      v159 = 2 * (v158 == 0);
      goto LABEL_214;
    }
    if ( (unsigned int)(v157 - 279) > 5 )
      goto LABEL_213;
    v159 = 2;
    if ( v123[85] > 0x31Fu )
      goto LABEL_212;
LABEL_214:
    *(_BYTE *)(a1 + v157 + 69914) = v159;
    v160 = *v151;
    *(_BYTE *)(a1 + v160 + 11414) = 2 * ((unsigned __int8)sub_305B520(v123, v160) == 0);
    v161 = *v151;
    v124 = v161;
    ++v151;
    *(_BYTE *)(a1 + v161 + 75414) = 2 * ((unsigned __int8)sub_305B520(v123, v161) == 0);
    if ( v151 != (unsigned int *)&unk_44C7AE8 )
    {
      v152 = *v151;
      continue;
    }
    break;
  }
  *(_BYTE *)(a1 + 10468) = 4;
  *(_BYTE *)(a1 + 9467) = 4;
  *(_WORD *)(a1 + 9967) = 1028;
  *(_WORD *)(a1 + 10963) = 1028;
  *(_BYTE *)(a1 + 12680) = 0;
  v162 = sub_305B500(v123, v161, v125, v126, v127);
  v165 = 1;
  if ( v123[85] > 0x2EDu && v123[84] > 0x45u )
    v165 = v162 ^ 1u;
  *(_BYTE *)(a1 + 12180) = v165;
  v166 = sub_305B500(v123, v161, v165, v163, v164);
  v167 = 2;
  if ( v123[85] > 0x2EDu && v123[84] > 0x45u )
    v167 = 2 * (v166 == 0);
  *(_BYTE *)(a1 + 70180) = v167;
  *(_BYTE *)(a1 + 11680) = sub_305B520(v123, 266) ^ 1;
  v168 = sub_305B520(v123, 266);
  v169 = (_BYTE)qword_502B048 == 0;
  *(_BYTE *)(a1 + 75680) = 2 * (v168 == 0);
  if ( !v169 )
  {
    v181 = *(_QWORD *)(a1 + 525256);
    *(_BYTE *)(a1 + 12677) = 0;
    *(_BYTE *)(a1 + 12177) = 1;
    v182 = a1 + 525248;
    LODWORD(v207) = 263;
    WORD2(v207) = 11;
    if ( !v181 )
      goto LABEL_260;
    do
    {
      while ( 1 )
      {
        v183 = *(_DWORD *)(v181 + 32);
        if ( v183 <= 0x106 || v183 == 263 && *(_WORD *)(v181 + 36) <= 0xAu )
          break;
        v182 = v181;
        v181 = *(_QWORD *)(v181 + 16);
        if ( !v181 )
          goto LABEL_258;
      }
      v181 = *(_QWORD *)(v181 + 24);
    }
    while ( v181 );
LABEL_258:
    if ( v182 == v205
      || *(_DWORD *)(v182 + 32) > 0x107u
      || *(_DWORD *)(v182 + 32) == 263 && *(_WORD *)(v182 + 36) > 0xBu )
    {
LABEL_260:
      v206 = &v207;
      v182 = sub_3052EE0(v204, v182, &v206);
    }
    *(_WORD *)(v182 + 40) = 12;
    v184 = *(_QWORD *)(a1 + 525256);
    *(_BYTE *)(a1 + 11677) = 1;
    v185 = a1 + 525248;
    LODWORD(v207) = 263;
    WORD2(v207) = 10;
    if ( !v184 )
      goto LABEL_270;
    do
    {
      while ( 1 )
      {
        v186 = *(_DWORD *)(v184 + 32);
        if ( v186 <= 0x106 || v186 == 263 && *(_WORD *)(v184 + 36) <= 9u )
          break;
        v185 = v184;
        v184 = *(_QWORD *)(v184 + 16);
        if ( !v184 )
          goto LABEL_268;
      }
      v184 = *(_QWORD *)(v184 + 24);
    }
    while ( v184 );
LABEL_268:
    if ( v185 == v205
      || *(_DWORD *)(v185 + 32) > 0x107u
      || *(_DWORD *)(v185 + 32) == 263 && *(_WORD *)(v185 + 36) > 0xAu )
    {
LABEL_270:
      v206 = &v207;
      v185 = sub_3052EE0(v204, v185, &v206);
    }
    *(_WORD *)(v185 + 40) = 12;
    *(_BYTE *)(a1 + 70177) = 2;
    *(_BYTE *)(a1 + 75677) = 2;
  }
  v170 = *(_QWORD *)v123;
  *(_BYTE *)(a1 + 10149) = 4;
  *(_BYTE *)(a1 + 10649) = 4;
  *(_BYTE *)(a1 + 10258) = 2;
  *(_BYTE *)(a1 + 10758) = 2;
  v171 = *(__int64 (__fastcall **)(__int64))(v170 + 200);
  if ( v171 == sub_3020000 )
    v172 = (__int64)(v123 + 114);
  else
    v172 = v171((__int64)v123);
  sub_2FE8000(a1, v172);
  *(_DWORD *)(a1 + 96) = 32;
  *(_QWORD *)(a1 + 84) = 0x4000000080LL;
  v173 = &v207;
  v207 = 0x460040003C003ALL;
  v174 = 58;
  LODWORD(v208) = 6160462;
  WORD2(v208) = 110;
  while ( 1 )
  {
    v173 = (__int64 *)((char *)v173 + 2);
    *(_BYTE *)(a1 + 500 * v174 + 6461) = 4;
    if ( v173 == (__int64 *)((char *)&v208 + 6) )
      break;
    v174 = *(unsigned __int16 *)v173;
  }
  LODWORD(v208) = 6160462;
  v175 = &v207;
  v207 = 0x460040003C003ALL;
  v176 = 58;
  WORD2(v208) = 110;
  while ( 1 )
  {
    result = 500 * v176;
    v175 = (__int64 *)((char *)v175 + 2);
    *(_BYTE *)(a1 + result + 6462) = 4;
    if ( v175 == (__int64 *)((char *)&v208 + 6) )
      break;
    v176 = *(unsigned __int16 *)v175;
  }
  return result;
}
