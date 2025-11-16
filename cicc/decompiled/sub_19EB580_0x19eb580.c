// Function: sub_19EB580
// Address: 0x19eb580
//
unsigned __int64 __fastcall sub_19EB580(__int64 a1, __int64 a2)
{
  _DWORD *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rbx
  char v6; // al
  __int64 *v7; // r12
  _DWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  _QWORD *v11; // rax
  _BYTE *v12; // rdx
  __int64 v13; // rdx
  __int64 *v14; // rsi
  __int64 *v15; // rdi
  unsigned __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 *v20; // rax
  char v21; // r8
  __int64 *v22; // r8
  unsigned __int64 v23; // rbx
  __int64 v24; // rax
  unsigned __int64 v25; // rcx
  unsigned __int64 v26; // rdx
  __int64 *v27; // rax
  char v28; // di
  unsigned __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rsi
  unsigned int v36; // ecx
  __int64 *v37; // rdx
  __int64 v38; // r8
  __int64 v39; // r13
  __int64 v40; // r12
  unsigned int v41; // esi
  __int64 v42; // r8
  unsigned int v43; // edx
  __int64 v44; // rax
  __int64 v45; // rcx
  __int64 v46; // rbx
  unsigned int v47; // esi
  __int64 v48; // rdi
  unsigned int v49; // edx
  __int64 *v50; // rax
  __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 i; // rbx
  __int64 v54; // rdi
  unsigned int v55; // edx
  __int64 *v56; // rax
  __int64 v57; // rcx
  int v58; // eax
  __int64 v59; // r12
  unsigned int v60; // esi
  int v61; // r8d
  int v62; // r8d
  __int64 v63; // r9
  int v64; // ecx
  __int64 v65; // rdx
  __int64 v66; // r11
  int v67; // ecx
  __int64 v68; // r12
  _QWORD *v69; // rax
  int v70; // edi
  int v71; // esi
  int v72; // esi
  __int64 v73; // r8
  unsigned int v74; // ecx
  __int64 v75; // rdx
  _QWORD *v76; // r9
  unsigned __int64 v77; // rdi
  __int64 v78; // rcx
  __int64 *v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // r12
  __int64 v83; // r13
  __int64 *v84; // rax
  char v85; // dl
  __int64 v86; // rbx
  __int64 *v87; // rax
  __int64 *v88; // rcx
  __int64 *v89; // rsi
  char v90; // di
  char v91; // al
  bool v92; // al
  __int64 v93; // rbx
  unsigned __int64 result; // rax
  __int64 v95; // r14
  __int64 v96; // rcx
  __int64 v97; // r8
  unsigned int v98; // edx
  __int64 v99; // rdi
  _DWORD *v100; // r12
  unsigned int v101; // esi
  int v102; // edx
  int v103; // edx
  int v104; // r10d
  int v105; // r10d
  __int64 *v106; // r9
  int v107; // edi
  int v108; // r10d
  __int64 *v109; // r9
  int v110; // edi
  int v111; // ecx
  __int64 v112; // rdi
  int v113; // r11d
  __int64 v114; // r10
  int v115; // ecx
  int v116; // edx
  int v117; // edx
  int v118; // r9d
  unsigned int v119; // esi
  int v120; // eax
  int v121; // eax
  __int64 v122; // rax
  int v123; // edi
  __int64 *v124; // rsi
  int v125; // r11d
  unsigned __int64 v126; // r10
  int v127; // edi
  __int64 v129; // [rsp+20h] [rbp-280h]
  __int64 v130; // [rsp+38h] [rbp-268h]
  __int64 v131; // [rsp+48h] [rbp-258h] BYREF
  __int64 *v132; // [rsp+50h] [rbp-250h] BYREF
  __int64 v133; // [rsp+58h] [rbp-248h]
  char v134; // [rsp+60h] [rbp-240h]
  _QWORD v135[16]; // [rsp+70h] [rbp-230h] BYREF
  __int64 v136; // [rsp+F0h] [rbp-1B0h] BYREF
  _BYTE *v137; // [rsp+F8h] [rbp-1A8h]
  _BYTE *v138; // [rsp+100h] [rbp-1A0h]
  __int64 v139; // [rsp+108h] [rbp-198h]
  int v140; // [rsp+110h] [rbp-190h]
  _BYTE v141[64]; // [rsp+118h] [rbp-188h] BYREF
  unsigned __int64 v142; // [rsp+158h] [rbp-148h] BYREF
  unsigned __int64 v143; // [rsp+160h] [rbp-140h]
  unsigned __int64 v144; // [rsp+168h] [rbp-138h]
  _QWORD v145[2]; // [rsp+170h] [rbp-130h] BYREF
  char v146; // [rsp+180h] [rbp-120h]
  __int64 *v147; // [rsp+1D8h] [rbp-C8h]
  __int64 *v148; // [rsp+1E0h] [rbp-C0h]
  _QWORD v149[13]; // [rsp+1F0h] [rbp-B0h] BYREF
  __int64 *v150; // [rsp+258h] [rbp-48h]
  __int64 *v151; // [rsp+260h] [rbp-40h]

  *(_DWORD *)(a1 + 1464) = 0;
  v3 = sub_19E13F0(a1, 0, 0);
  v4 = *(_QWORD *)(a1 + 32);
  *(_QWORD *)(a1 + 1432) = v3;
  v129 = a1 + 1960;
  *((_QWORD *)v3 + 5) = *(_QWORD *)(v4 + 120);
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 120LL);
  v136 = v5;
  v6 = sub_19E11D0(a1 + 1960, &v136, v145);
  v7 = (__int64 *)v145[0];
  if ( v6 )
    goto LABEL_2;
  v119 = *(_DWORD *)(a1 + 1984);
  v120 = *(_DWORD *)(a1 + 1976);
  ++*(_QWORD *)(a1 + 1960);
  v121 = v120 + 1;
  if ( 4 * v121 >= 3 * v119 )
  {
    v119 *= 2;
    goto LABEL_188;
  }
  if ( v119 - *(_DWORD *)(a1 + 1980) - v121 <= v119 >> 3 )
  {
LABEL_188:
    sub_19E42B0(v129, v119);
    sub_19E11D0(v129, &v136, v145);
    v7 = (__int64 *)v145[0];
    v121 = *(_DWORD *)(a1 + 1976) + 1;
  }
  *(_DWORD *)(a1 + 1976) = v121;
  if ( *v7 != -8 )
    --*(_DWORD *)(a1 + 1980);
  v122 = v136;
  v7[1] = 0;
  *v7 = v122;
LABEL_2:
  v8 = sub_19E13F0(a1, 0, 0);
  *((_QWORD *)v8 + 5) = v5;
  v7[1] = (__int64)v8;
  v9 = *(_QWORD *)(a1 + 8);
  memset(v135, 0, sizeof(v135));
  v135[1] = &v135[5];
  v135[2] = &v135[5];
  LODWORD(v135[3]) = 8;
  v10 = *(_QWORD *)(v9 + 56);
  v136 = 0;
  v137 = v141;
  v138 = v141;
  v139 = 8;
  v140 = 0;
  v142 = 0;
  v143 = 0;
  v144 = 0;
  v11 = sub_1412190((__int64)&v136, v10);
  if ( v138 == v137 )
    v12 = &v138[8 * HIDWORD(v139)];
  else
    v12 = &v138[8 * (unsigned int)v139];
  v145[0] = v11;
  v145[1] = v12;
  sub_19E4730((__int64)v145);
  v145[0] = v10;
  v146 = 0;
  sub_13B8390(&v142, (__int64)v145);
  sub_13BA6D0(v145, &v136, v135);
  sub_19E4F00(&v136);
  sub_19E4F00(v135);
  sub_16CCCB0(v135, (__int64)&v135[5], (__int64)v145);
  v14 = v148;
  v15 = v147;
  memset(&v135[13], 0, 24);
  v16 = (char *)v148 - (char *)v147;
  if ( v148 == v147 )
  {
    v16 = 0;
    v18 = 0;
  }
  else
  {
    if ( v16 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_204;
    v17 = sub_22077B0((char *)v148 - (char *)v147);
    v14 = v148;
    v15 = v147;
    v18 = v17;
  }
  v135[13] = v18;
  v135[14] = v18;
  v135[15] = v18 + v16;
  if ( v14 != v15 )
  {
    v19 = v18;
    v20 = v15;
    do
    {
      if ( v19 )
      {
        *(_QWORD *)v19 = *v20;
        v21 = *((_BYTE *)v20 + 16);
        *(_BYTE *)(v19 + 16) = v21;
        if ( v21 )
          *(_QWORD *)(v19 + 8) = v20[1];
      }
      v20 += 3;
      v19 += 24;
    }
    while ( v20 != v14 );
    v18 += 8 * ((unsigned __int64)((char *)(v20 - 3) - (char *)v15) >> 3) + 24;
  }
  v15 = &v136;
  v135[14] = v18;
  sub_16CCCB0(&v136, (__int64)v141, (__int64)v149);
  v14 = v151;
  v22 = v150;
  v142 = 0;
  v143 = 0;
  v144 = 0;
  v23 = (char *)v151 - (char *)v150;
  if ( v151 != v150 )
  {
    if ( v23 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v24 = sub_22077B0((char *)v151 - (char *)v150);
      v14 = v151;
      v22 = v150;
      v25 = v24;
      goto LABEL_17;
    }
LABEL_204:
    sub_4261EA(v15, v14, v13);
  }
  v25 = 0;
LABEL_17:
  v142 = v25;
  v143 = v25;
  v144 = v25 + v23;
  if ( v14 == v22 )
  {
    v29 = v25;
  }
  else
  {
    v26 = v25;
    v27 = v22;
    do
    {
      if ( v26 )
      {
        *(_QWORD *)v26 = *v27;
        v28 = *((_BYTE *)v27 + 16);
        *(_BYTE *)(v26 + 16) = v28;
        if ( v28 )
          *(_QWORD *)(v26 + 8) = v27[1];
      }
      v27 += 3;
      v26 += 24LL;
    }
    while ( v14 != v27 );
    v29 = v25 + 8 * ((unsigned __int64)((char *)(v14 - 3) - (char *)v22) >> 3) + 24;
  }
  v30 = v135[14];
  v31 = v135[13];
  v143 = v29;
  if ( v135[14] - v135[13] == v29 - v25 )
    goto LABEL_87;
  do
  {
LABEL_25:
    v32 = *(_QWORD *)(a1 + 32);
    v33 = **(_QWORD **)(v30 - 24);
    v34 = *(unsigned int *)(v32 + 112);
    v130 = v33;
    if ( (_DWORD)v34 )
    {
      v35 = *(_QWORD *)(v32 + 96);
      v36 = (v34 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
      v37 = (__int64 *)(v35 + 16LL * v36);
      v38 = *v37;
      if ( v33 != *v37 )
      {
        v117 = 1;
        while ( v38 != -8 )
        {
          v118 = v117 + 1;
          v36 = (v34 - 1) & (v117 + v36);
          v37 = (__int64 *)(v35 + 16LL * v36);
          v38 = *v37;
          if ( v33 == *v37 )
            goto LABEL_27;
          v117 = v118;
        }
        goto LABEL_41;
      }
LABEL_27:
      if ( v37 != (__int64 *)(v35 + 16 * v34) )
      {
        v39 = v37[1];
        if ( v39 )
        {
          v40 = *(_QWORD *)(v39 + 8);
          if ( v40 != v39 )
          {
            while ( 1 )
            {
              v46 = v40 - 48;
              v47 = *(_DWORD *)(a1 + 1984);
              if ( !v40 )
                v46 = 0;
              v131 = v46;
              if ( !v47 )
                break;
              v48 = *(_QWORD *)(a1 + 1968);
              v49 = (v47 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
              v50 = (__int64 *)(v48 + 16LL * v49);
              v51 = *v50;
              if ( v46 != *v50 )
              {
                v108 = 1;
                v109 = 0;
                while ( v51 != -8 )
                {
                  if ( !v109 && v51 == -16 )
                    v109 = v50;
                  v49 = (v47 - 1) & (v108 + v49);
                  v50 = (__int64 *)(v48 + 16LL * v49);
                  v51 = *v50;
                  if ( v46 == *v50 )
                    goto LABEL_38;
                  ++v108;
                }
                v110 = *(_DWORD *)(a1 + 1976);
                if ( v109 )
                  v50 = v109;
                ++*(_QWORD *)(a1 + 1960);
                v111 = v110 + 1;
                if ( 4 * (v110 + 1) < 3 * v47 )
                {
                  v112 = v46;
                  if ( v47 - *(_DWORD *)(a1 + 1980) - v111 > v47 >> 3 )
                  {
LABEL_132:
                    *(_DWORD *)(a1 + 1976) = v111;
                    if ( *v50 != -8 )
                      --*(_DWORD *)(a1 + 1980);
                    *v50 = v112;
                    v50[1] = 0;
                    goto LABEL_38;
                  }
LABEL_137:
                  sub_19E42B0(v129, v47);
                  sub_19E11D0(v129, &v131, &v132);
                  v50 = v132;
                  v112 = v131;
                  v111 = *(_DWORD *)(a1 + 1976) + 1;
                  goto LABEL_132;
                }
LABEL_136:
                v47 *= 2;
                goto LABEL_137;
              }
LABEL_38:
              v52 = *(_QWORD *)(a1 + 1432);
              v50[1] = v52;
              if ( *(_BYTE *)(v46 + 16) == 22 )
              {
                if ( *(_BYTE *)(*(_QWORD *)(v46 + 72) + 16LL) != 55 )
                  goto LABEL_33;
                ++*(_DWORD *)(v52 + 184);
                v40 = *(_QWORD *)(v40 + 8);
                if ( v39 == v40 )
                  goto LABEL_41;
              }
              else
              {
                sub_1412190(v52 + 128, v46);
                v41 = *(_DWORD *)(a1 + 2016);
                v132 = (__int64 *)v46;
                LODWORD(v133) = 1;
                if ( !v41 )
                {
                  ++*(_QWORD *)(a1 + 1992);
                  goto LABEL_148;
                }
                v42 = *(_QWORD *)(a1 + 2000);
                v43 = (v41 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
                v44 = v42 + 16LL * v43;
                v45 = *(_QWORD *)v44;
                if ( v46 != *(_QWORD *)v44 )
                {
                  v113 = 1;
                  v114 = 0;
                  while ( v45 != -8 )
                  {
                    if ( v45 == -16 && !v114 )
                      v114 = v44;
                    v43 = (v41 - 1) & (v113 + v43);
                    v44 = v42 + 16LL * v43;
                    v45 = *(_QWORD *)v44;
                    if ( v46 == *(_QWORD *)v44 )
                      goto LABEL_33;
                    ++v113;
                  }
                  v115 = *(_DWORD *)(a1 + 2008);
                  if ( v114 )
                    v44 = v114;
                  ++*(_QWORD *)(a1 + 1992);
                  v116 = v115 + 1;
                  if ( 4 * (v115 + 1) < 3 * v41 )
                  {
                    if ( v41 - *(_DWORD *)(a1 + 2012) - v116 > v41 >> 3 )
                    {
LABEL_144:
                      *(_DWORD *)(a1 + 2008) = v116;
                      if ( *(_QWORD *)v44 != -8 )
                        --*(_DWORD *)(a1 + 2012);
                      *(_QWORD *)v44 = v46;
                      *(_DWORD *)(v44 + 8) = v133;
                      goto LABEL_33;
                    }
LABEL_149:
                    sub_19E4430(a1 + 1992, v41);
                    sub_19E1280(a1 + 1992, (__int64 *)&v132, &v131);
                    v44 = v131;
                    v46 = (__int64)v132;
                    v116 = *(_DWORD *)(a1 + 2008) + 1;
                    goto LABEL_144;
                  }
LABEL_148:
                  v41 *= 2;
                  goto LABEL_149;
                }
LABEL_33:
                v40 = *(_QWORD *)(v40 + 8);
                if ( v39 == v40 )
                  goto LABEL_41;
              }
            }
            ++*(_QWORD *)(a1 + 1960);
            goto LABEL_136;
          }
        }
      }
    }
LABEL_41:
    for ( i = *(_QWORD *)(v130 + 48); v130 + 40 != i; i = *(_QWORD *)(i + 8) )
    {
      if ( !i )
        BUG();
      v58 = *(unsigned __int8 *)(i - 8);
      if ( (_BYTE)v58 == 77 )
      {
        v68 = *(_QWORD *)(i - 16);
        if ( !v68 )
          goto LABEL_50;
        do
        {
          v69 = sub_1648700(v68);
          v70 = *((unsigned __int8 *)v69 + 16);
          if ( (unsigned __int8)v70 > 0x17u )
          {
            v71 = *(_DWORD *)(a1 + 2416);
            if ( v71 )
            {
              v72 = v71 - 1;
              v73 = *(_QWORD *)(a1 + 2400);
              v74 = v72 & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
              v75 = v73 + 16LL * v74;
              v76 = *(_QWORD **)v75;
              if ( v69 == *(_QWORD **)v75 )
              {
LABEL_62:
                if ( *(_DWORD *)(v75 + 8) )
                {
                  if ( byte_4FB3BA0 )
                  {
                    v77 = (unsigned int)(v70 - 35);
                    if ( (unsigned __int8)v77 <= 0x2Cu )
                    {
                      v78 = 0x1300000BFFFFLL;
                      if ( _bittest64(&v78, v77) )
                      {
                        v79 = sub_1412190(a1 + 1536, (__int64)v69);
                        v80 = *(_QWORD *)(a1 + 1552);
                        if ( v80 == *(_QWORD *)(a1 + 1544) )
                          v81 = *(unsigned int *)(a1 + 1564);
                        else
                          v81 = *(unsigned int *)(a1 + 1560);
                        v132 = v79;
                        v133 = v80 + 8 * v81;
                        sub_19E4730((__int64)&v132);
                      }
                    }
                  }
                }
              }
              else
              {
                v103 = 1;
                while ( v76 != (_QWORD *)-8LL )
                {
                  v104 = v103 + 1;
                  v74 = v72 & (v74 + v103);
                  v75 = v73 + 16LL * v74;
                  v76 = *(_QWORD **)v75;
                  if ( v69 == *(_QWORD **)v75 )
                    goto LABEL_62;
                  v103 = v104;
                }
              }
            }
          }
          v68 = *(_QWORD *)(v68 + 8);
        }
        while ( v68 );
        v58 = *(unsigned __int8 *)(i - 8);
      }
      if ( (unsigned int)(v58 - 25) <= 9 && !*(_BYTE *)(*(_QWORD *)(i - 24) + 8LL) )
        continue;
LABEL_50:
      v59 = i - 24;
      sub_1412190(*(_QWORD *)(a1 + 1432) + 56LL, i - 24);
      v60 = *(_DWORD *)(a1 + 1496);
      v131 = i - 24;
      if ( !v60 )
      {
        ++*(_QWORD *)(a1 + 1472);
        goto LABEL_52;
      }
      v54 = *(_QWORD *)(a1 + 1480);
      v55 = (v60 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
      v56 = (__int64 *)(v54 + 16LL * v55);
      v57 = *v56;
      if ( v59 != *v56 )
      {
        v105 = 1;
        v106 = 0;
        while ( v57 != -8 )
        {
          if ( v57 == -16 && !v106 )
            v106 = v56;
          v55 = (v60 - 1) & (v105 + v55);
          v56 = (__int64 *)(v54 + 16LL * v55);
          v57 = *v56;
          if ( v59 == *v56 )
            goto LABEL_44;
          ++v105;
        }
        v107 = *(_DWORD *)(a1 + 1488);
        if ( v106 )
          v56 = v106;
        ++*(_QWORD *)(a1 + 1472);
        v67 = v107 + 1;
        if ( 4 * (v107 + 1) >= 3 * v60 )
        {
LABEL_52:
          sub_19E4130(a1 + 1472, 2 * v60);
          v61 = *(_DWORD *)(a1 + 1496);
          if ( !v61 )
          {
            ++*(_DWORD *)(a1 + 1488);
            BUG();
          }
          v59 = v131;
          v62 = v61 - 1;
          v63 = *(_QWORD *)(a1 + 1480);
          v64 = *(_DWORD *)(a1 + 1488);
          LODWORD(v65) = v62 & (((unsigned int)v131 >> 9) ^ ((unsigned int)v131 >> 4));
          v56 = (__int64 *)(v63 + 16LL * (unsigned int)v65);
          v66 = *v56;
          if ( v131 == *v56 )
          {
LABEL_54:
            v67 = v64 + 1;
          }
          else
          {
            v123 = 1;
            v124 = 0;
            while ( v66 != -8 )
            {
              if ( v66 == -16 && !v124 )
                v124 = v56;
              v65 = v62 & (unsigned int)(v65 + v123);
              v56 = (__int64 *)(v63 + 16 * v65);
              v66 = *v56;
              if ( v131 == *v56 )
                goto LABEL_54;
              ++v123;
            }
            v67 = v64 + 1;
            if ( v124 )
              v56 = v124;
          }
        }
        else if ( v60 - *(_DWORD *)(a1 + 1492) - v67 <= v60 >> 3 )
        {
          sub_19E4130(a1 + 1472, v60);
          sub_19E1120(a1 + 1472, &v131, &v132);
          v56 = v132;
          v59 = v131;
          v67 = *(_DWORD *)(a1 + 1488) + 1;
        }
        *(_DWORD *)(a1 + 1488) = v67;
        if ( *v56 != -8 )
          --*(_DWORD *)(a1 + 1492);
        *v56 = v59;
        v56[1] = 0;
      }
LABEL_44:
      v56[1] = *(_QWORD *)(a1 + 1432);
    }
    v82 = v135[14];
    do
    {
      v83 = *(_QWORD *)(v82 - 24);
      if ( !*(_BYTE *)(v82 - 8) )
      {
        v84 = *(__int64 **)(v83 + 24);
        *(_BYTE *)(v82 - 8) = 1;
        *(_QWORD *)(v82 - 16) = v84;
        goto LABEL_76;
      }
      while ( 1 )
      {
        v84 = *(__int64 **)(v82 - 16);
LABEL_76:
        if ( v84 == *(__int64 **)(v83 + 32) )
          break;
        *(_QWORD *)(v82 - 16) = v84 + 1;
        v86 = *v84;
        v87 = (__int64 *)v135[1];
        if ( v135[2] != v135[1] )
          goto LABEL_74;
        v88 = (__int64 *)(v135[1] + 8LL * HIDWORD(v135[3]));
        if ( (__int64 *)v135[1] == v88 )
        {
LABEL_110:
          if ( HIDWORD(v135[3]) < LODWORD(v135[3]) )
          {
            ++HIDWORD(v135[3]);
            *v88 = v86;
            ++v135[0];
LABEL_85:
            v132 = (__int64 *)v86;
            v134 = 0;
            sub_13B8390(&v135[13], (__int64)&v132);
            v31 = v135[13];
            v30 = v135[14];
            goto LABEL_86;
          }
LABEL_74:
          sub_16CCBA0((__int64)v135, v86);
          if ( v85 )
            goto LABEL_85;
        }
        else
        {
          v89 = 0;
          while ( v86 != *v87 )
          {
            if ( *v87 == -2 )
            {
              v89 = v87;
              if ( v88 == v87 + 1 )
                goto LABEL_84;
              ++v87;
            }
            else if ( v88 == ++v87 )
            {
              if ( !v89 )
                goto LABEL_110;
LABEL_84:
              *v89 = v86;
              --LODWORD(v135[4]);
              ++v135[0];
              goto LABEL_85;
            }
          }
        }
      }
      v135[14] -= 24LL;
      v31 = v135[13];
      v82 = v135[14];
    }
    while ( v135[14] != v135[13] );
    v30 = v135[13];
LABEL_86:
    v25 = v142;
  }
  while ( v30 - v31 != v143 - v142 );
LABEL_87:
  if ( v30 != v31 )
  {
    while ( *(_QWORD *)v31 == *(_QWORD *)v25 )
    {
      v90 = *(_BYTE *)(v31 + 16);
      v91 = *(_BYTE *)(v25 + 16);
      if ( v90 && v91 )
        v92 = *(_QWORD *)(v31 + 8) == *(_QWORD *)(v25 + 8);
      else
        v92 = v90 == v91;
      if ( !v92 )
        break;
      v31 += 24;
      v25 += 24LL;
      if ( v31 == v30 )
        goto LABEL_94;
    }
    goto LABEL_25;
  }
LABEL_94:
  sub_19E4F00(&v136);
  sub_19E4F00(v135);
  sub_19E4F00(v149);
  sub_19E4F00(v145);
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    sub_15E08E0(a2, v30);
    v93 = *(_QWORD *)(a2 + 88);
    result = 5LL * *(_QWORD *)(a2 + 96);
    v95 = v93 + 40LL * *(_QWORD *)(a2 + 96);
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    {
      result = sub_15E08E0(a2, v30);
      v93 = *(_QWORD *)(a2 + 88);
    }
  }
  else
  {
    v93 = *(_QWORD *)(a2 + 88);
    result = 5LL * *(_QWORD *)(a2 + 96);
    v95 = v93 + 40LL * *(_QWORD *)(a2 + 96);
  }
  if ( v93 != v95 )
  {
    while ( 1 )
    {
      v136 = v93;
      v100 = sub_19E13F0(a1, v93, 0);
      sub_1412190((__int64)(v100 + 14), v136);
      v101 = *(_DWORD *)(a1 + 1496);
      if ( !v101 )
        break;
      v96 = v136;
      v97 = *(_QWORD *)(a1 + 1480);
      v98 = (v101 - 1) & (((unsigned int)v136 >> 9) ^ ((unsigned int)v136 >> 4));
      result = v97 + 16LL * v98;
      v99 = *(_QWORD *)result;
      if ( v136 == *(_QWORD *)result )
      {
LABEL_99:
        v93 += 40;
        *(_QWORD *)(result + 8) = v100;
        if ( v95 == v93 )
          return result;
      }
      else
      {
        v125 = 1;
        v126 = 0;
        while ( v99 != -8 )
        {
          if ( v99 == -16 && !v126 )
            v126 = result;
          v98 = (v101 - 1) & (v125 + v98);
          result = v97 + 16LL * v98;
          v99 = *(_QWORD *)result;
          if ( v136 == *(_QWORD *)result )
            goto LABEL_99;
          ++v125;
        }
        v127 = *(_DWORD *)(a1 + 1488);
        if ( v126 )
          result = v126;
        ++*(_QWORD *)(a1 + 1472);
        v102 = v127 + 1;
        if ( 4 * (v127 + 1) < 3 * v101 )
        {
          if ( v101 - *(_DWORD *)(a1 + 1492) - v102 > v101 >> 3 )
            goto LABEL_104;
          goto LABEL_103;
        }
LABEL_102:
        v101 *= 2;
LABEL_103:
        sub_19E4130(a1 + 1472, v101);
        sub_19E1120(a1 + 1472, &v136, v145);
        result = v145[0];
        v96 = v136;
        v102 = *(_DWORD *)(a1 + 1488) + 1;
LABEL_104:
        *(_DWORD *)(a1 + 1488) = v102;
        if ( *(_QWORD *)result != -8 )
          --*(_DWORD *)(a1 + 1492);
        v93 += 40;
        *(_QWORD *)(result + 8) = 0;
        *(_QWORD *)result = v96;
        *(_QWORD *)(result + 8) = v100;
        if ( v95 == v93 )
          return result;
      }
    }
    ++*(_QWORD *)(a1 + 1472);
    goto LABEL_102;
  }
  return result;
}
