// Function: sub_30760B0
// Address: 0x30760b0
//
__int64 *__fastcall sub_30760B0(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  unsigned int v7; // eax
  __int64 v9; // r13
  char *v10; // rsi
  __int64 v11; // rdx
  int v12; // edx
  char v13; // dl
  unsigned int v15; // r13d
  int v16; // edx
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // rdx
  int v22; // r14d
  __int64 v23; // rax
  __int64 v24; // rdx
  int v25; // edx
  __int64 v26; // r9
  __int64 v27; // rdx
  unsigned __int8 *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r15
  char *v31; // rdx
  int v32; // ecx
  int v33; // eax
  __int64 *v34; // rax
  __int64 v35; // r15
  __int64 v36; // rax
  unsigned __int64 v37; // rdi
  char v38; // r15
  __int64 v39; // r14
  void *v40; // rsi
  char v41; // al
  int v42; // eax
  __int64 v43; // rdx
  int v44; // eax
  __int64 v45; // rdx
  bool v46; // si
  __int64 v47; // r14
  int v48; // eax
  __int64 v49; // rdx
  __int64 v50; // r13
  unsigned __int8 *v51; // rax
  int v52; // r13d
  int v53; // eax
  __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rdi
  __int64 v57; // r12
  unsigned int v58; // edx
  _QWORD *v59; // rax
  int v60; // esi
  __int64 v61; // rax
  int v62; // esi
  unsigned int v63; // edx
  __int64 *v64; // rbx
  __int64 v65; // rbx
  int v66; // edx
  int v67; // ebx
  __int64 v68; // r13
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rdi
  __int64 v72; // rcx
  _QWORD *v73; // rsi
  __int64 v74; // rax
  __int64 v75; // rdi
  _BYTE *v76; // r14
  _BYTE *v77; // rdi
  char v78; // al
  __int64 v79; // rdi
  __int64 v80; // rcx
  __int64 v81; // rax
  _QWORD *v82; // rsi
  __int64 v83; // rsi
  _QWORD *v84; // rbx
  __int64 v85; // r12
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // rdx
  _BYTE *v89; // rax
  int v90; // edx
  unsigned int v91; // r13d
  __int64 v92; // r15
  __int64 v93; // rax
  __int64 v94; // rdx
  __int64 v95; // r14
  __int64 v96; // rdx
  int v97; // r14d
  __int64 v98; // rax
  __int64 v99; // rdx
  int v100; // edx
  __int64 v101; // r9
  __int64 v102; // rdx
  unsigned __int8 *v103; // rax
  __int64 v104; // rdx
  __int64 v105; // r15
  __int64 *v106; // rdx
  int v107; // ecx
  __int64 *v108; // rax
  __int64 v109; // r15
  __int64 v110; // rax
  __int64 v111; // rcx
  _QWORD *v112; // rax
  const char *v113; // rax
  int v114; // ecx
  __int64 v115; // rdx
  int v116; // esi
  unsigned int v117; // edx
  __int64 v118; // rbx
  __int64 v119; // rbx
  __int64 v120; // [rsp+0h] [rbp-D0h]
  __int64 v121; // [rsp+0h] [rbp-D0h]
  int v122; // [rsp+8h] [rbp-C8h]
  unsigned int v123; // [rsp+8h] [rbp-C8h]
  __int64 v124; // [rsp+8h] [rbp-C8h]
  int v125; // [rsp+8h] [rbp-C8h]
  unsigned int v126; // [rsp+8h] [rbp-C8h]
  __int64 v127; // [rsp+8h] [rbp-C8h]
  __int64 v128; // [rsp+10h] [rbp-C0h]
  unsigned __int8 *v129; // [rsp+10h] [rbp-C0h]
  __int64 v130; // [rsp+10h] [rbp-C0h]
  unsigned __int8 *v131; // [rsp+10h] [rbp-C0h]
  __int64 *v132; // [rsp+18h] [rbp-B8h]
  __int64 *v133; // [rsp+18h] [rbp-B8h]
  __int64 v134; // [rsp+38h] [rbp-98h] BYREF
  __int64 *v135; // [rsp+40h] [rbp-90h] BYREF
  __int64 v136; // [rsp+48h] [rbp-88h]
  _BYTE v137[16]; // [rsp+50h] [rbp-80h] BYREF
  __int16 v138; // [rsp+60h] [rbp-70h]
  const char *v139; // [rsp+70h] [rbp-60h] BYREF
  __int64 v140; // [rsp+78h] [rbp-58h]
  _BYTE v141[16]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v142; // [rsp+90h] [rbp-40h]

  v6 = *((_QWORD *)a3 - 4);
  if ( !v6 || *(_BYTE *)v6 || *(_QWORD *)(v6 + 24) != *((_QWORD *)a3 + 10) )
    BUG();
  v7 = *(_DWORD *)(v6 + 36);
  if ( v7 > 0x22E0 )
  {
    if ( v7 - 8930 > 2 )
    {
      if ( v7 <= 0x230C )
        goto LABEL_7;
      if ( v7 > 0x298C )
        goto LABEL_51;
      if ( v7 > 0x296D )
      {
        switch ( v7 )
        {
          case 0x296Eu:
            v15 = 355;
            goto LABEL_23;
          case 0x296Fu:
            v38 = 0;
            v15 = 355;
            v39 = sub_B43CB0((__int64)a3);
            goto LABEL_48;
          case 0x2971u:
            v38 = 1;
            v15 = 355;
            v39 = sub_B43CB0((__int64)a3);
            goto LABEL_48;
          case 0x2980u:
          case 0x2984u:
          case 0x2988u:
          case 0x298Cu:
            v52 = 43;
LABEL_86:
            v139 = sub_BD5D20((__int64)a3);
            v53 = *((_DWORD *)a3 + 1);
            v140 = v54;
            v55 = *((_QWORD *)a3 + 1);
            v142 = 261;
            v9 = sub_B51D30(v52, *(_QWORD *)&a3[-32 * (v53 & 0x7FFFFFF)], v55, (__int64)&v139, 0, 0);
            goto LABEL_66;
          default:
            goto LABEL_51;
        }
      }
      if ( v7 == 9534 || v7 == 9540 )
      {
        v15 = 335;
        goto LABEL_23;
      }
      if ( v7 != 9270 )
        goto LABEL_51;
      v139 = sub_BD5D20((__int64)a3);
      v48 = *((_DWORD *)a3 + 1);
      v142 = 261;
      v140 = v49;
      v50 = *(_QWORD *)&a3[-32 * (v48 & 0x7FFFFFF)];
      v51 = sub_AD8DD0(*(_QWORD *)(v50 + 8), 1.0);
      v9 = sub_B504D0(21, (__int64)v51, v50, (__int64)&v139, 0, 0);
      goto LABEL_66;
    }
LABEL_9:
    v9 = (__int64)a3;
    v10 = *(char **)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    v11 = *((_QWORD *)v10 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17 <= 1 )
      v11 = **(_QWORD **)(v11 + 16);
    v12 = *(_DWORD *)(v11 + 8) >> 8;
    if ( !v12 )
    {
      v13 = *v10;
      if ( (unsigned __int8)*v10 > 0x1Cu )
      {
        if ( v13 != 79 )
          goto LABEL_15;
      }
      else if ( v13 != 5 || *((_WORD *)v10 + 1) != 50 )
      {
        goto LABEL_15;
      }
      v43 = *(_QWORD *)(*((_QWORD *)v10 - 4) + 8LL);
      if ( (unsigned int)*(unsigned __int8 *)(v43 + 8) - 17 <= 1 )
        v43 = **(_QWORD **)(v43 + 16);
      v12 = *(_DWORD *)(v43 + 8) >> 8;
      if ( !v12 )
        goto LABEL_15;
    }
    if ( v12 != 101 )
    {
      switch ( v7 )
      {
        case 0x22E0u:
          v46 = v12 == 1;
          break;
        case 0x22E1u:
          goto LABEL_15;
        case 0x22E2u:
          v46 = v12 == 5;
          break;
        case 0x22E3u:
        case 0x22E4u:
          v46 = v12 == 3;
          break;
        default:
          v46 = v12 == 4;
          break;
      }
      v47 = sub_AD64C0(*((_QWORD *)a3 + 1), v46, 0);
      if ( *((_QWORD *)a3 + 2) )
        goto LABEL_74;
    }
LABEL_15:
    v9 = 0;
LABEL_16:
    v135 = (__int64 *)v9;
    LOBYTE(v136) = 1;
    return v135;
  }
  if ( v7 > 0x22DE )
    goto LABEL_9;
  if ( v7 > 0x21C2 )
  {
LABEL_7:
    switch ( v7 )
    {
      case 0x21C3u:
        v15 = 172;
        goto LABEL_23;
      case 0x21C4u:
        v38 = 0;
        v15 = 172;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_48;
      case 0x21C6u:
        v38 = 1;
        v15 = 172;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_48;
      case 0x21D3u:
      case 0x21D5u:
      case 0x21D8u:
      case 0x21DAu:
        v38 = 0;
        v15 = 173;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_93;
      case 0x21D6u:
        v15 = 173;
        goto LABEL_23;
      case 0x21D7u:
        v38 = 0;
        v15 = 173;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_48;
      case 0x21DBu:
      case 0x21DCu:
      case 0x21DEu:
      case 0x21DFu:
        v38 = 1;
        v15 = 173;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_93;
      case 0x21DDu:
        v38 = 1;
        v15 = 173;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_48;
      case 0x2205u:
        v15 = 237;
        goto LABEL_23;
      case 0x2207u:
        v38 = 0;
        v15 = 237;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_48;
      case 0x2209u:
      case 0x220Au:
        v38 = 0;
        v15 = 237;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_93;
      case 0x220Eu:
        v38 = 1;
        v15 = 237;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_48;
      case 0x220Fu:
      case 0x2210u:
        v38 = 1;
        v15 = 237;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_93;
      case 0x2213u:
        v38 = 1;
        v15 = 235;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_48;
      case 0x2214u:
      case 0x2215u:
        v38 = 1;
        v15 = 235;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_93;
      case 0x2222u:
        v38 = 0;
        v15 = 235;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_48;
      case 0x2223u:
      case 0x2224u:
        v38 = 0;
        v15 = 235;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_93;
      case 0x223Cu:
        v15 = 248;
        goto LABEL_23;
      case 0x223Eu:
        v38 = 0;
        v15 = 248;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_48;
      case 0x2240u:
      case 0x2241u:
        v38 = 0;
        v15 = 248;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_93;
      case 0x2245u:
        v38 = 1;
        v15 = 248;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_48;
      case 0x2246u:
      case 0x2247u:
        v38 = 1;
        v15 = 248;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_93;
      case 0x224Au:
        v38 = 1;
        v15 = 246;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_48;
      case 0x224Bu:
      case 0x224Cu:
        v38 = 1;
        v15 = 246;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_93;
      case 0x2259u:
        v38 = 0;
        v15 = 246;
        v39 = sub_B43CB0((__int64)a3);
        goto LABEL_48;
      case 0x225Au:
      case 0x225Bu:
        v38 = 0;
        v15 = 246;
        v39 = sub_B43CB0((__int64)a3);
LABEL_93:
        v40 = sub_C332F0();
        goto LABEL_49;
      case 0x2272u:
      case 0x2273u:
        v9 = (__int64)a3;
        v71 = *((_DWORD *)a3 + 1) & 0x7FFFFFF;
        v72 = *(_QWORD *)&a3[32 * (2 - v71)];
        if ( *(_BYTE *)v72 != 17 )
          goto LABEL_51;
        v73 = *(_QWORD **)(v72 + 24);
        if ( *(_DWORD *)(v72 + 32) > 0x40u )
          v73 = (_QWORD *)*v73;
        if ( *(_DWORD *)(*((_QWORD *)a3 + 1) + 8LL) >> 8 <= (unsigned __int64)v73 )
        {
          v47 = *(_QWORD *)&a3[32 * ((v7 == 8818) - v71)];
          v74 = *((_QWORD *)a3 + 2);
          if ( v47 )
          {
            if ( v74 )
              goto LABEL_74;
          }
          else if ( v74 )
          {
            sub_10A5FE0(*(_QWORD *)(a2 + 40), (__int64)a3);
            BUG();
          }
          goto LABEL_51;
        }
        v90 = *a3;
        v142 = 257;
        v91 = (v7 != 8818) + 180;
        if ( v90 == 40 )
        {
          v92 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a3);
        }
        else
        {
          v92 = -32;
          if ( v90 != 85 )
          {
            v92 = -96;
            if ( v90 != 34 )
LABEL_211:
              BUG();
          }
        }
        if ( (a3[7] & 0x80u) != 0 )
        {
          v93 = sub_BD2BC0((__int64)a3);
          v95 = v93 + v94;
          v96 = 0;
          if ( (a3[7] & 0x80u) != 0 )
            v96 = sub_BD2BC0((__int64)a3);
          if ( (unsigned int)((v95 - v96) >> 4) )
          {
            if ( (a3[7] & 0x80u) == 0 )
              BUG();
            v97 = *(_DWORD *)(sub_BD2BC0((__int64)a3) + 8);
            if ( (a3[7] & 0x80u) == 0 )
              BUG();
            v98 = sub_BD2BC0((__int64)a3);
            v92 -= 32LL * (unsigned int)(*(_DWORD *)(v98 + v99 - 4) - v97);
          }
        }
        v100 = *((_DWORD *)a3 + 1);
        v101 = (__int64)&a3[v92];
        v135 = (__int64 *)v137;
        v136 = 0x300000000LL;
        v102 = 32LL * (v100 & 0x7FFFFFF);
        v103 = &a3[-v102];
        v104 = v92 + v102;
        v105 = v104 >> 5;
        if ( (unsigned __int64)v104 > 0x60 )
        {
          v127 = v101;
          v131 = v103;
          sub_C8D5F0((__int64)&v135, v137, v104 >> 5, 8u, a5, v101);
          v107 = v136;
          v101 = v127;
          v133 = v135;
          v106 = &v135[(unsigned int)v136];
          v103 = v131;
        }
        else
        {
          v133 = (__int64 *)v137;
          v106 = (__int64 *)v137;
          v107 = 0;
        }
        if ( v103 != (unsigned __int8 *)v101 )
        {
          do
          {
            if ( v106 )
              *v106 = *(_QWORD *)v103;
            v103 += 32;
            ++v106;
          }
          while ( (unsigned __int8 *)v101 != v103 );
          v107 = v136;
          v133 = v135;
        }
        LODWORD(v136) = v107 + v105;
        v130 = (unsigned int)(v107 + v105);
        v125 = v107 + v105;
        v134 = *((_QWORD *)a3 + 1);
        v108 = (__int64 *)sub_B43CA0((__int64)a3);
        v109 = 0;
        v110 = sub_B6E160(v108, v91, (__int64)&v134, 1);
        if ( v110 )
          v109 = *(_QWORD *)(v110 + 24);
        v121 = v110;
        v126 = v125 + 1;
        v9 = (__int64)sub_BD2CC0(88, v126);
        if ( v9 )
        {
          sub_B44260(v9, **(_QWORD **)(v109 + 16), 56, v126 & 0x7FFFFFF, 0, 0);
          *(_QWORD *)(v9 + 72) = 0;
          sub_B4A290(v9, v109, v121, v133, v130, (__int64)&v139, 0, 0);
        }
        v37 = (unsigned __int64)v135;
        if ( v135 == (__int64 *)v137 )
          goto LABEL_66;
        goto LABEL_44;
      case 0x22A1u:
      case 0x22A5u:
      case 0x2308u:
      case 0x230Cu:
        v52 = 44;
        goto LABEL_86;
      default:
        goto LABEL_51;
    }
  }
  if ( v7 == 8490 )
  {
    v111 = *(_QWORD *)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    v112 = *(_QWORD **)(v111 + 24);
    if ( *(_DWORD *)(v111 + 32) > 0x40u )
      v112 = (_QWORD *)*v112;
    if ( ((unsigned __int8)v112 & 7) != 1 || *(_BYTE *)(*((_QWORD *)a3 + 1) + 8LL) != 3 )
      goto LABEL_51;
    v113 = sub_BD5D20((__int64)a3);
    v114 = *((_DWORD *)a3 + 1);
    v139 = v113;
    v140 = v115;
    v142 = 261;
    v9 = sub_B504D0(
           21,
           *(_QWORD *)&a3[32 * (1LL - (v114 & 0x7FFFFFF))],
           *(_QWORD *)&a3[32 * (2LL - (v114 & 0x7FFFFFF))],
           (__int64)&v139,
           0,
           0);
  }
  else
  {
    if ( v7 <= 0x212A )
    {
      switch ( v7 )
      {
        case 0x2062u:
          v38 = 0;
          v15 = 21;
          v39 = sub_B43CB0((__int64)a3);
          break;
        case 0x2064u:
          v38 = 1;
          v15 = 21;
          v39 = sub_B43CB0((__int64)a3);
          break;
        case 0x2061u:
          v15 = 21;
          goto LABEL_23;
        default:
          goto LABEL_51;
      }
LABEL_48:
      v40 = sub_C33310();
LABEL_49:
      v41 = sub_B2DB90(v39, (__int64)v40);
      if ( sub_CEA700(v41) != v38 )
        goto LABEL_50;
      goto LABEL_23;
    }
    if ( v7 != 8504 )
    {
      if ( v7 != 8589 )
        goto LABEL_51;
      v15 = 170;
LABEL_23:
      v16 = *a3;
      if ( v16 == 40 )
      {
        v17 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a3);
      }
      else
      {
        v17 = -32;
        if ( v16 != 85 )
        {
          v17 = -96;
          if ( v16 != 34 )
            goto LABEL_211;
        }
      }
      if ( (a3[7] & 0x80u) != 0 )
      {
        v18 = sub_BD2BC0((__int64)a3);
        v20 = v18 + v19;
        v21 = 0;
        if ( (a3[7] & 0x80u) != 0 )
          v21 = sub_BD2BC0((__int64)a3);
        if ( (unsigned int)((v20 - v21) >> 4) )
        {
          if ( (a3[7] & 0x80u) == 0 )
            BUG();
          v22 = *(_DWORD *)(sub_BD2BC0((__int64)a3) + 8);
          if ( (a3[7] & 0x80u) == 0 )
            BUG();
          v23 = sub_BD2BC0((__int64)a3);
          v17 -= 32LL * (unsigned int)(*(_DWORD *)(v23 + v24 - 4) - v22);
        }
      }
      v25 = *((_DWORD *)a3 + 1);
      v26 = (__int64)&a3[v17];
      v139 = v141;
      v140 = 0x400000000LL;
      v27 = 32LL * (v25 & 0x7FFFFFF);
      v28 = &a3[-v27];
      v29 = v17 + v27;
      v30 = v29 >> 5;
      if ( (unsigned __int64)v29 > 0x80 )
      {
        v124 = v26;
        v129 = v28;
        sub_C8D5F0((__int64)&v139, v141, v29 >> 5, 8u, a5, v26);
        v32 = v140;
        v26 = v124;
        v132 = (__int64 *)v139;
        v31 = (char *)&v139[8 * (unsigned int)v140];
        v28 = v129;
      }
      else
      {
        v132 = (__int64 *)v141;
        v31 = v141;
        v32 = 0;
      }
      if ( v28 != (unsigned __int8 *)v26 )
      {
        do
        {
          if ( v31 )
            *(_QWORD *)v31 = *(_QWORD *)v28;
          v28 += 32;
          v31 += 8;
        }
        while ( (unsigned __int8 *)v26 != v28 );
        v32 = v140;
        v132 = (__int64 *)v139;
      }
      v33 = *((_DWORD *)a3 + 1);
      LODWORD(v140) = v30 + v32;
      v122 = v30 + v32;
      v134 = *(_QWORD *)(*(_QWORD *)&a3[-32 * (v33 & 0x7FFFFFF)] + 8LL);
      v138 = 257;
      v128 = (unsigned int)(v30 + v32);
      v34 = (__int64 *)sub_B43CA0((__int64)a3);
      v35 = 0;
      v36 = sub_B6E160(v34, v15, (__int64)&v134, 1);
      if ( v36 )
        v35 = *(_QWORD *)(v36 + 24);
      v120 = v36;
      v123 = v122 + 1;
      v9 = (__int64)sub_BD2CC0(88, v123);
      if ( v9 )
      {
        sub_B44260(v9, **(_QWORD **)(v35 + 16), 56, v123 & 0x7FFFFFF, 0, 0);
        *(_QWORD *)(v9 + 72) = 0;
        sub_B4A290(v9, v35, v120, v132, v128, (__int64)&v135, 0, 0);
      }
      v37 = (unsigned __int64)v139;
      if ( v139 != v141 )
LABEL_44:
        _libc_free(v37);
      goto LABEL_66;
    }
    v139 = sub_BD5D20((__int64)a3);
    v44 = *((_DWORD *)a3 + 1);
    v140 = v45;
    v142 = 261;
    v9 = sub_B504D0(
           21,
           *(_QWORD *)&a3[-32 * (v44 & 0x7FFFFFF)],
           *(_QWORD *)&a3[32 * (1LL - (v44 & 0x7FFFFFF))],
           (__int64)&v139,
           0,
           0);
  }
LABEL_66:
  if ( v9 )
    goto LABEL_16;
LABEL_50:
  v6 = *((_QWORD *)a3 - 4);
  if ( !v6 )
    goto LABEL_210;
LABEL_51:
  if ( *(_BYTE *)v6 || *(_QWORD *)(v6 + 24) != *((_QWORD *)a3 + 10) )
LABEL_210:
    BUG();
  v42 = *(_DWORD *)(v6 + 36);
  if ( v42 == 9176 )
  {
    v56 = *(_QWORD *)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    if ( *(_BYTE *)v56 != 17 )
      goto LABEL_56;
    v57 = *(_QWORD *)&a3[32 * (1LL - (*((_DWORD *)a3 + 1) & 0x7FFFFFF))];
    if ( *(_BYTE *)v57 != 17 )
      goto LABEL_56;
    v58 = *(_DWORD *)(v56 + 32);
    v59 = *(_QWORD **)(v56 + 24);
    if ( v58 > 0x40 )
    {
      v116 = *v59;
      v61 = *v59 & 0xFFFFFFLL;
      v62 = v116 & 0x800000;
    }
    else
    {
      if ( !v58 )
      {
        v117 = *(_DWORD *)(v57 + 32);
        v61 = *(_QWORD *)(v57 + 24);
        if ( v117 > 0x40 )
        {
          v119 = *(_QWORD *)v61;
          LODWORD(v61) = 0;
          v66 = v119;
          v67 = v119 & 0xFFFFFF;
        }
        else
        {
          if ( !v117 )
          {
            LODWORD(v61) = 0;
            v67 = 0;
            goto LABEL_108;
          }
          v118 = v61 << (64 - (unsigned __int8)v117);
          LODWORD(v61) = 0;
          v66 = v118 >> (64 - (unsigned __int8)v117);
          v67 = v66 & 0xFFFFFF;
        }
LABEL_106:
        if ( (v66 & 0x800000) != 0 )
          v67 |= 0xFF000000;
        goto LABEL_108;
      }
      v60 = (__int64)((_QWORD)v59 << (64 - (unsigned __int8)v58)) >> (64 - (unsigned __int8)v58);
      v61 = ((__int64)((_QWORD)v59 << (64 - (unsigned __int8)v58)) >> (64 - (unsigned __int8)v58)) & 0xFFFFFF;
      v62 = v60 & 0x800000;
    }
    v63 = *(_DWORD *)(v57 + 32);
    v64 = *(__int64 **)(v57 + 24);
    if ( v63 > 0x40 )
    {
      v65 = *v64;
LABEL_104:
      v66 = v65;
      v67 = v65 & 0xFFFFFF;
      if ( !v62 )
        goto LABEL_106;
      goto LABEL_105;
    }
    if ( v63 )
    {
      v65 = (__int64)((_QWORD)v64 << (64 - (unsigned __int8)v63)) >> (64 - (unsigned __int8)v63);
      goto LABEL_104;
    }
    v67 = 0;
    v66 = 0;
    if ( v62 )
    {
LABEL_105:
      LODWORD(v61) = v61 | 0xFF000000;
      goto LABEL_106;
    }
LABEL_108:
    v68 = sub_ACD640(*(_QWORD *)(v56 + 8), (int)v61, 1u);
    v69 = sub_ACD640(*(_QWORD *)(v57 + 8), v67, 1u);
    v142 = 257;
    v70 = sub_B504D0(17, v68, v69, (__int64)&v139, 0, 0);
    LOBYTE(v136) = 1;
    return (__int64 *)v70;
  }
  if ( v42 != 9177 )
  {
    if ( v42 != 8534 )
    {
LABEL_56:
      LOBYTE(v136) = 0;
      return v135;
    }
    v9 = (__int64)a3;
    v75 = *(_QWORD *)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    v76 = (_BYTE *)(v75 + 24);
    if ( *(_BYTE *)v75 != 18 )
    {
      v88 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v75 + 8) + 8LL) - 17;
      if ( (unsigned int)v88 > 1 )
        goto LABEL_56;
      if ( *(_BYTE *)v75 > 0x15u )
        goto LABEL_56;
      v89 = sub_AD7630(v75, 0, v88);
      if ( !v89 || *v89 != 18 )
        goto LABEL_56;
      v76 = v89 + 24;
    }
    v77 = v76;
    if ( *(void **)v76 == sub_C33340() )
      v77 = (_BYTE *)*((_QWORD *)v76 + 1);
    v78 = v77[20] & 7;
    v79 = *((_QWORD *)a3 + 1);
    if ( v78 == 1 )
      v47 = sub_AD6400(v79);
    else
      v47 = sub_AD6450(v79);
    if ( !*((_QWORD *)a3 + 2) )
      goto LABEL_15;
LABEL_74:
    sub_10A5FE0(*(_QWORD *)(a2 + 40), (__int64)a3);
    if ( a3 == (unsigned __int8 *)v47 )
      v47 = sub_ACADE0(*((__int64 ***)a3 + 1));
    if ( !*(_QWORD *)(v47 + 16) && *(_BYTE *)v47 > 0x1Cu && (*(_BYTE *)(v47 + 7) & 0x10) == 0 && (a3[7] & 0x10) != 0 )
      sub_BD6B90((unsigned __int8 *)v47, a3);
    sub_BD84D0((__int64)a3, v47);
    goto LABEL_16;
  }
  v80 = *(_QWORD *)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
  if ( *(_BYTE *)v80 != 17 )
    goto LABEL_56;
  v81 = *(_QWORD *)&a3[32 * (1LL - (*((_DWORD *)a3 + 1) & 0x7FFFFFF))];
  if ( *(_BYTE *)v81 != 17 )
    goto LABEL_56;
  v82 = *(_QWORD **)(v80 + 24);
  if ( *(_DWORD *)(v80 + 32) > 0x40u )
    v82 = (_QWORD *)*v82;
  v83 = (unsigned int)v82 & 0xFFFFFF;
  v84 = *(_QWORD **)(v81 + 24);
  if ( *(_DWORD *)(v81 + 32) > 0x40u )
    v84 = (_QWORD *)*v84;
  v85 = sub_ACD640(*(_QWORD *)(v80 + 8), v83, 0);
  v86 = sub_ACD640(*(_QWORD *)(v85 + 8), (unsigned int)v84 & 0xFFFFFF, 0);
  v142 = 257;
  v87 = sub_B504D0(17, v85, v86, (__int64)&v139, 0, 0);
  LOBYTE(v136) = 1;
  return (__int64 *)v87;
}
