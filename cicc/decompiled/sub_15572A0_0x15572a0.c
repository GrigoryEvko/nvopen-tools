// Function: sub_15572A0
// Address: 0x15572a0
//
void __fastcall sub_15572A0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rbx
  __int64 v5; // r8
  __int64 (__fastcall *v6)(__int64); // rax
  char v7; // al
  const char *v8; // r15
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r8
  void (*v13)(); // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // edi
  __int64 v17; // r15
  unsigned __int16 v18; // ax
  const char *v19; // rax
  char v20; // al
  unsigned __int8 v21; // al
  __int64 v22; // r15
  const char *v23; // rax
  size_t v24; // rdx
  __int64 v25; // rdi
  unsigned int v26; // ebx
  __int16 v27; // ax
  __int64 *v28; // r15
  unsigned int v29; // edx
  __int64 **v30; // rcx
  __int64 v31; // rbx
  __int64 v32; // rcx
  __int64 v33; // rdx
  unsigned int v34; // esi
  int v35; // eax
  int v36; // eax
  __int64 v37; // rcx
  const char *v38; // rdi
  __int64 v39; // rcx
  __int64 **v40; // rax
  __int64 v41; // rbx
  __int64 v42; // rax
  bool v43; // zf
  int v44; // ebx
  __int64 v45; // rax
  __int64 v46; // rax
  unsigned __int16 v47; // r15
  __int64 v48; // rax
  __int64 v49; // rbx
  __int64 v50; // r15
  __int64 v51; // rax
  __int64 v52; // rax
  int v53; // edx
  __int64 v54; // rdi
  __int64 v55; // rsi
  __int64 v56; // rdx
  __int64 v57; // rdi
  unsigned __int8 v58; // al
  __int64 v59; // r15
  unsigned int v60; // ebx
  _QWORD *v61; // rax
  char v62; // bl
  __int64 v63; // r15
  __int64 v64; // r8
  __int64 v65; // rax
  __int64 v66; // rax
  unsigned int v67; // r15d
  unsigned __int8 v68; // bl
  __int64 v69; // rsi
  int v70; // r15d
  __int64 **v71; // rax
  __int64 v72; // rax
  __int64 v73; // rbx
  __int64 v74; // rsi
  __int64 v75; // rax
  __int64 v76; // rdx
  unsigned int v77; // r15d
  __int64 **v78; // rax
  __int64 v79; // rax
  __int64 v80; // rbx
  __int64 v81; // rax
  unsigned int v82; // ebx
  __int64 v83; // rsi
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // rbx
  __int64 j; // r15
  unsigned __int8 v88; // bl
  int v89; // r15d
  __int64 v90; // rsi
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // r15
  int i; // ebx
  __int64 v95; // rax
  __int64 v96; // rdi
  __int64 v97; // rax
  __int64 *v98; // rsi
  __int64 **v99; // rax
  __int64 v100; // rax
  __int64 *v101; // rbx
  __int64 *v102; // rcx
  __int64 *v103; // rax
  int v104; // r15d
  __int64 *v105; // r8
  __int64 v106; // rax
  __int64 *v107; // rsi
  const char *v108; // rsi
  __int64 v109; // rbx
  int v110; // ebx
  int v111; // eax
  __int64 v112; // rbx
  __int64 v113; // rax
  __int64 v114; // rbx
  int v115; // ebx
  int v116; // eax
  __int64 v117; // rbx
  __int64 v118; // rax
  __int64 v119; // rbx
  __int64 v120; // rax
  __int64 v121; // rdx
  __int64 v122; // rcx
  int v123; // eax
  __int16 v124; // ax
  unsigned int v125; // ebx
  __int64 v126; // rax
  __int64 v127; // rax
  __int64 v128; // rax
  const char *v129; // rsi
  __int64 v130; // rbx
  __int64 v131; // rax
  __int64 v132; // rdx
  __int64 v133; // rcx
  int v134; // eax
  __int64 v135; // rax
  __int64 v136; // rax
  __int64 v137; // [rsp+8h] [rbp-C8h]
  __int64 v138; // [rsp+8h] [rbp-C8h]
  __int64 v139; // [rsp+8h] [rbp-C8h]
  __int64 v140; // [rsp+8h] [rbp-C8h]
  __int64 v141; // [rsp+10h] [rbp-C0h]
  __int64 v142; // [rsp+10h] [rbp-C0h]
  __int64 v143; // [rsp+10h] [rbp-C0h]
  __int64 v144; // [rsp+10h] [rbp-C0h]
  __int64 v145; // [rsp+10h] [rbp-C0h]
  __int64 v146; // [rsp+10h] [rbp-C0h]
  __int64 *v147; // [rsp+10h] [rbp-C0h]
  __int64 *v148; // [rsp+10h] [rbp-C0h]
  __int64 *v149; // [rsp+10h] [rbp-C0h]
  __int64 (__fastcall *v150)(__int64); // [rsp+18h] [rbp-B8h]
  __int64 v151; // [rsp+18h] [rbp-B8h]
  __int64 v152; // [rsp+18h] [rbp-B8h]
  unsigned __int8 v153; // [rsp+18h] [rbp-B8h]
  int v154; // [rsp+18h] [rbp-B8h]
  int v155; // [rsp+18h] [rbp-B8h]
  __int64 v156; // [rsp+18h] [rbp-B8h]
  int v157; // [rsp+18h] [rbp-B8h]
  int v158; // [rsp+18h] [rbp-B8h]
  __int64 *v159; // [rsp+18h] [rbp-B8h]
  __int64 v160; // [rsp+18h] [rbp-B8h]
  __int64 v161; // [rsp+18h] [rbp-B8h]
  __int64 v162; // [rsp+18h] [rbp-B8h]
  __int64 v163; // [rsp+18h] [rbp-B8h]
  __int64 v164; // [rsp+18h] [rbp-B8h]
  __int64 v165; // [rsp+18h] [rbp-B8h]
  void *v166; // [rsp+20h] [rbp-B0h] BYREF
  _QWORD v167[2]; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v168; // [rsp+38h] [rbp-98h]
  __int64 v169; // [rsp+40h] [rbp-90h]
  const char *v170; // [rsp+50h] [rbp-80h] BYREF
  __int64 v171; // [rsp+58h] [rbp-78h] BYREF
  __int64 v172; // [rsp+60h] [rbp-70h] BYREF
  __int64 v173; // [rsp+68h] [rbp-68h]
  __int64 v174; // [rsp+70h] [rbp-60h]

  v4 = a1[77];
  if ( !v4 )
    goto LABEL_11;
  v5 = *a1;
  v6 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)*a1 + 88LL);
  v168 = a2;
  v167[0] = 2;
  v150 = v6;
  v167[1] = 0;
  if ( a2 != -8 && a2 != -16 )
  {
    v141 = v5;
    sub_164C220(v167);
    v5 = v141;
  }
  v137 = v5;
  v166 = &unk_49ECE00;
  v169 = v4;
  v7 = sub_154CE90(v4, (__int64)&v166, &v170);
  v8 = v170;
  v9 = v137;
  if ( !v7 )
  {
    v34 = *(_DWORD *)(v4 + 24);
    v35 = *(_DWORD *)(v4 + 16);
    ++*(_QWORD *)v4;
    v36 = v35 + 1;
    if ( 4 * v36 >= 3 * v34 )
    {
      v34 *= 2;
      v146 = v137;
    }
    else
    {
      if ( v34 - *(_DWORD *)(v4 + 20) - v36 > v34 >> 3 )
        goto LABEL_53;
      v146 = v137;
    }
    sub_1556EF0(v4, v34);
    sub_154CE90(v4, (__int64)&v166, &v170);
    v8 = v170;
    v9 = v146;
    v36 = *(_DWORD *)(v4 + 16) + 1;
LABEL_53:
    *(_DWORD *)(v4 + 16) = v36;
    v171 = 2;
    v172 = 0;
    v173 = -8;
    v174 = 0;
    if ( *((_QWORD *)v8 + 3) != -8 )
    {
      --*(_DWORD *)(v4 + 20);
      v170 = (const char *)&unk_49EE2B0;
      if ( v173 != 0 && v173 != -16 && v173 != -8 )
      {
        v143 = v9;
        sub_1649B30(&v171);
        v9 = v143;
      }
    }
    v37 = *((_QWORD *)v8 + 3);
    v10 = v168;
    if ( v37 != v168 )
    {
      v38 = v8 + 8;
      if ( v37 != -8 && v37 != 0 && v37 != -16 )
      {
        v138 = v9;
        sub_1649B30(v38);
        v10 = v168;
        v9 = v138;
      }
      *((_QWORD *)v8 + 3) = v10;
      if ( v10 == 0 || v10 == -8 || v10 == -16 )
      {
        v10 = v168;
      }
      else
      {
        v144 = v9;
        sub_1649AC0(v38, v167[0] & 0xFFFFFFFFFFFFFFF8LL);
        v10 = v168;
        v9 = v144;
      }
    }
    v39 = v169;
    *((_DWORD *)v8 + 10) = 0;
    *((_QWORD *)v8 + 4) = v39;
    goto LABEL_7;
  }
  v10 = v168;
LABEL_7:
  v166 = &unk_49EE2B0;
  if ( v10 != 0 && v10 != -8 && v10 != -16 )
  {
    v142 = v9;
    sub_1649B30(v167);
    v9 = v142;
  }
  *((_DWORD *)v8 + 10) = v150(v9);
LABEL_11:
  v11 = a1[29];
  v12 = *a1;
  if ( v11 )
  {
    v13 = *(void (**)())(*(_QWORD *)v11 + 40LL);
    if ( v13 != nullsub_537 )
    {
      ((void (__fastcall *)(__int64, __int64, __int64))v13)(v11, a2, *a1);
      v12 = *a1;
    }
  }
  sub_1263B40(v12, "  ");
  if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
  {
    sub_154B790(*a1, a2);
    sub_1263B40(*a1, " = ");
  }
  else if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) )
  {
    v44 = sub_154F3B0(a1[4], a2, v14, v15);
    if ( v44 == -1 )
    {
      sub_1263B40(*a1, "<badref> = ");
    }
    else
    {
      v45 = sub_1549FC0(*a1, 0x25u);
      v46 = sub_16E7AB0(v45, v44);
      sub_1263B40(v46, " = ");
    }
  }
  v16 = *(unsigned __int8 *)(a2 + 16);
  v17 = *a1;
  if ( (_BYTE)v16 == 78 )
  {
    v18 = *(_WORD *)(a2 + 18) & 3;
    if ( v18 == 2 )
    {
      sub_1263B40(*a1, "musttail ");
      v16 = *(unsigned __int8 *)(a2 + 16);
      v17 = *a1;
    }
    else if ( (unsigned int)v18 - 1 <= 1 )
    {
      sub_1263B40(*a1, "tail ");
      v16 = *(unsigned __int8 *)(a2 + 16);
      v17 = *a1;
    }
    else if ( v18 == 3 )
    {
      sub_1263B40(*a1, "notail ");
      v16 = *(unsigned __int8 *)(a2 + 16);
      v17 = *a1;
    }
  }
  v19 = (const char *)sub_15F29F0((unsigned int)(v16 - 24));
  sub_1263B40(v17, v19);
  v20 = *(_BYTE *)(a2 + 16);
  if ( v20 != 54 )
  {
    if ( v20 != 55 )
      goto LABEL_23;
    goto LABEL_145;
  }
  if ( (unsigned __int8)sub_15F32D0(a2) )
  {
LABEL_116:
    sub_1263B40(*a1, " atomic");
LABEL_117:
    v20 = *(_BYTE *)(a2 + 16);
    goto LABEL_23;
  }
  v20 = *(_BYTE *)(a2 + 16);
  if ( v20 == 55 )
  {
LABEL_145:
    if ( !(unsigned __int8)sub_15F32D0(a2) )
      goto LABEL_117;
    goto LABEL_116;
  }
LABEL_23:
  if ( v20 != 58 )
  {
    if ( (unsigned __int8)(v20 - 54) <= 1u )
      goto LABEL_25;
    goto LABEL_120;
  }
  v27 = *(_WORD *)(a2 + 18);
  if ( (v27 & 0x100) == 0 )
  {
LABEL_31:
    if ( (v27 & 1) == 0 )
      goto LABEL_26;
    goto LABEL_32;
  }
  sub_1263B40(*a1, " weak");
  v20 = *(_BYTE *)(a2 + 16);
  if ( (unsigned __int8)(v20 - 54) <= 1u )
  {
LABEL_25:
    if ( (*(_BYTE *)(a2 + 18) & 1) == 0 )
      goto LABEL_26;
    goto LABEL_32;
  }
  if ( v20 == 58 )
  {
    v27 = *(_WORD *)(a2 + 18);
    goto LABEL_31;
  }
LABEL_120:
  if ( v20 == 59 && (*(_BYTE *)(a2 + 18) & 1) != 0 )
LABEL_32:
    sub_1263B40(*a1, " volatile");
LABEL_26:
  sub_154A7B0(*a1, a2);
  v21 = *(_BYTE *)(a2 + 16);
  if ( (unsigned __int8)(v21 - 75) <= 1u )
  {
    v22 = sub_1549FC0(*a1, 0x20u);
    v23 = (const char *)sub_15FF290(*(_WORD *)(a2 + 18) & 0x7FFF);
    sub_1549FF0(v22, v23, v24);
    v21 = *(_BYTE *)(a2 + 16);
  }
  if ( v21 == 59 )
  {
    v25 = *a1;
    v26 = (*(unsigned __int16 *)(a2 + 18) >> 5) & 0x7FFFBFF;
    switch ( v26 )
    {
      case 0u:
        sub_1263B40(v25, " xchg");
        v21 = *(_BYTE *)(a2 + 16);
        break;
      case 1u:
        sub_1263B40(v25, " add");
        v21 = *(_BYTE *)(a2 + 16);
        break;
      case 2u:
        sub_1263B40(v25, " sub");
        v21 = *(_BYTE *)(a2 + 16);
        break;
      case 3u:
        sub_1263B40(v25, " and");
        v21 = *(_BYTE *)(a2 + 16);
        break;
      case 4u:
        sub_1263B40(v25, " nand");
        v21 = *(_BYTE *)(a2 + 16);
        break;
      case 5u:
        sub_1263B40(v25, " or");
        v21 = *(_BYTE *)(a2 + 16);
        break;
      case 6u:
        sub_1263B40(v25, " xor");
        v21 = *(_BYTE *)(a2 + 16);
        break;
      case 7u:
        sub_1263B40(v25, " max");
        v21 = *(_BYTE *)(a2 + 16);
        break;
      case 8u:
        sub_1263B40(v25, " min");
        v21 = *(_BYTE *)(a2 + 16);
        break;
      case 9u:
        sub_1263B40(v25, " umax");
        v21 = *(_BYTE *)(a2 + 16);
        break;
      case 0xAu:
        sub_1263B40(v25, " umin");
        v21 = *(_BYTE *)(a2 + 16);
        break;
      default:
        v91 = sub_1263B40(v25, " <unknown operation ");
        v92 = sub_16E7AB0(v91, v26);
        sub_1263B40(v92, ">");
        v21 = *(_BYTE *)(a2 + 16);
        break;
    }
  }
  v28 = 0;
  v29 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( v29 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v30 = *(__int64 ***)(a2 - 8);
    else
      v30 = (__int64 **)(a2 - 24LL * v29);
    v28 = *v30;
  }
  if ( v21 == 26 )
  {
    if ( v29 == 3 )
    {
      sub_1549FC0(*a1, 0x20u);
      sub_15520E0(a1, *(__int64 **)(a2 - 72), 1);
      sub_1263B40(*a1, ", ");
      sub_15520E0(a1, *(__int64 **)(a2 - 24), 1);
      sub_1263B40(*a1, ", ");
      sub_15520E0(a1, *(__int64 **)(a2 - 48), 1);
      v21 = *(_BYTE *)(a2 + 16);
      goto LABEL_71;
    }
    v53 = 26;
    goto LABEL_91;
  }
  if ( v21 != 27 )
  {
    switch ( v21 )
    {
      case 0x1Cu:
        v82 = 1;
        sub_1549FC0(*a1, 0x20u);
        sub_15520E0(a1, v28, 1);
        sub_1263B40(*a1, ", [");
        v157 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        if ( v157 != 1 )
        {
LABEL_152:
          if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          {
LABEL_153:
            v83 = *(_QWORD *)(a2 - 8);
            goto LABEL_154;
          }
          while ( 1 )
          {
            v83 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
LABEL_154:
            v84 = v82++;
            sub_15520E0(a1, *(__int64 **)(v83 + 24 * v84), 1);
            if ( v157 == v82 )
              break;
            if ( v82 == 1 )
              goto LABEL_152;
            sub_1263B40(*a1, ", ");
            if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
              goto LABEL_153;
          }
        }
        goto LABEL_165;
      case 0x4Du:
        sub_1549FC0(*a1, 0x20u);
        sub_154DAA0((__int64)(a1 + 5), *(_QWORD *)a2, *a1);
        sub_1549FC0(*a1, 0x20u);
        if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
        {
          v31 = 0;
          v151 = 8LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
          while ( 1 )
          {
            sub_1263B40(*a1, "[ ");
            v32 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
                ? *(_QWORD *)(a2 - 8)
                : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
            sub_15520E0(a1, *(__int64 **)(v32 + 3 * v31), 0);
            sub_1263B40(*a1, ", ");
            v33 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
                ? *(_QWORD *)(a2 - 8)
                : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
            v31 += 8;
            sub_15520E0(a1, *(__int64 **)(v31 + v33 + 24LL * *(unsigned int *)(a2 + 56)), 0);
            sub_1263B40(*a1, " ]");
            if ( v151 == v31 )
              break;
            sub_1263B40(*a1, ", ");
          }
        }
        goto LABEL_70;
      case 0x56u:
        sub_1549FC0(*a1, 0x20u);
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v40 = *(__int64 ***)(a2 - 8);
        else
          v40 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        sub_15520E0(a1, *v40, 1);
        v41 = *(_QWORD *)(a2 + 56);
        v152 = v41 + 4LL * *(unsigned int *)(a2 + 64);
        while ( v152 != v41 )
        {
          v41 += 4;
          v42 = sub_1263B40(*a1, ", ");
          sub_16E7A90(v42, *(unsigned int *)(v41 - 4));
        }
        goto LABEL_70;
      case 0x57u:
        sub_1549FC0(*a1, 0x20u);
        v78 = (__int64 **)sub_13CF970(a2);
        sub_15520E0(a1, *v78, 1);
        sub_1263B40(*a1, ", ");
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v79 = *(_QWORD *)(a2 - 8);
        else
          v79 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        sub_15520E0(a1, *(__int64 **)(v79 + 24), 1);
        v80 = *(_QWORD *)(a2 + 56);
        v156 = v80 + 4LL * *(unsigned int *)(a2 + 64);
        while ( v156 != v80 )
        {
          v80 += 4;
          v81 = sub_1263B40(*a1, ", ");
          sub_16E7A90(v81, *(unsigned int *)(v80 - 4));
        }
        goto LABEL_70;
      case 0x58u:
        sub_1549FC0(*a1, 0x20u);
        sub_154DAA0((__int64)(a1 + 5), *(_QWORD *)a2, *a1);
        if ( (*(_BYTE *)(a2 + 18) & 1) != 0 || (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
        {
          sub_1549FC0(*a1, 0xAu);
          if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
            sub_1263B40(*a1, "          cleanup");
          v158 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
          if ( v158 )
          {
            v93 = 0;
            for ( i = 0; i != v158; ++i )
            {
              if ( i || (*(_BYTE *)(a2 + 18) & 1) != 0 )
                sub_1263B40(*a1, "\n");
              if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
                v95 = *(_QWORD *)(a2 - 8);
              else
                v95 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
              v96 = *a1;
              if ( *(_BYTE *)(**(_QWORD **)(v95 + v93) + 8LL) == 14 )
                sub_1263B40(v96, "          filter ");
              else
                sub_1263B40(v96, "          catch ");
              if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
                v97 = *(_QWORD *)(a2 - 8);
              else
                v97 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
              v98 = *(__int64 **)(v97 + v93);
              v93 += 24;
              sub_15520E0(a1, v98, 1);
            }
          }
        }
        goto LABEL_70;
      case 0x22u:
        sub_1263B40(*a1, " within ");
        v99 = (__int64 **)sub_13CF970(a2);
        sub_15520E0(a1, *v99, 0);
        sub_1263B40(*a1, " [");
        v100 = sub_13CF970(a2);
        v101 = (__int64 *)(v100 + 24);
        v102 = (__int64 *)(v100 + 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        v103 = (__int64 *)(v100 + 48);
        if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
          v101 = v103;
        v159 = v102;
        if ( v101 != v102 )
        {
          v104 = 0;
          v105 = (__int64 *)sub_1523720(*v101);
          while ( 1 )
          {
            ++v104;
            sub_15520E0(a1, v105, 1);
            v101 += 3;
            if ( v101 == v159 )
              break;
            v106 = sub_1523720(*v101);
            v105 = (__int64 *)v106;
            if ( v104 )
            {
              v147 = (__int64 *)v106;
              sub_1263B40(*a1, ", ");
              v105 = v147;
            }
          }
        }
        sub_1263B40(*a1, "] unwind ");
        if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
        {
          v107 = *(__int64 **)(sub_13CF970(a2) + 24);
          if ( v107 )
            goto LABEL_224;
        }
LABEL_225:
        sub_1263B40(*a1, "to caller");
        v21 = *(_BYTE *)(a2 + 16);
        goto LABEL_71;
    }
    v53 = v21;
    if ( (unsigned int)v21 - 73 <= 1 )
    {
      sub_1263B40(*a1, " within ");
      sub_15520E0(a1, *(__int64 **)(a2 - 24), 0);
      sub_1263B40(*a1, " [");
      v85 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      if ( (_DWORD)v85 != 1 )
      {
        v86 = (unsigned int)(v85 - 2);
        for ( j = 0; ; ++j )
        {
          sub_15520E0(a1, *(__int64 **)(a2 + 24 * (j - v85)), 1);
          if ( v86 == j )
            break;
          if ( (_DWORD)j != -1 )
            sub_1263B40(*a1, ", ");
          v85 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        }
      }
LABEL_165:
      sub_1549FC0(*a1, 0x5Du);
      v21 = *(_BYTE *)(a2 + 16);
      goto LABEL_71;
    }
    if ( v21 == 25 && !v28 )
    {
      sub_1263B40(*a1, " void");
      v21 = *(_BYTE *)(a2 + 16);
      goto LABEL_71;
    }
    switch ( v21 )
    {
      case 0x21u:
        sub_1263B40(*a1, " from ");
        sub_15520E0(a1, *(__int64 **)(a2 - 48), 0);
        v108 = " to ";
LABEL_231:
        sub_1263B40(*a1, v108);
        v107 = *(__int64 **)(a2 - 24);
LABEL_224:
        sub_15520E0(a1, v107, 1);
LABEL_70:
        v21 = *(_BYTE *)(a2 + 16);
        goto LABEL_71;
      case 0x20u:
        sub_1263B40(*a1, " from ");
        sub_15520E0(a1, *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), 0);
        sub_1263B40(*a1, " unwind ");
        if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
        {
          sub_15520E0(a1, *(__int64 **)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))), 1);
          v21 = *(_BYTE *)(a2 + 16);
          goto LABEL_71;
        }
        goto LABEL_225;
      case 0x4Eu:
        if ( (*(_WORD *)(a2 + 18) & 0x7FFC) != 0 )
        {
          sub_1263B40(*a1, " ");
          sub_154A100((*(unsigned __int16 *)(a2 + 18) >> 2) & 0x3FFFDFFF, *a1);
        }
        v109 = *(_QWORD *)(a2 + 64);
        v148 = *(__int64 **)(a2 - 24);
        v160 = **(_QWORD **)(v109 + 16);
        v166 = *(void **)(a2 + 56);
        if ( (unsigned __int8)sub_15602F0(&v166, 0) )
        {
          v140 = sub_1549FC0(*a1, 0x20u);
          sub_1560450(&v170, &v166, 0, 0);
          sub_16E7EE0(v140, v170, v171);
          sub_2240A30(&v170);
        }
        sub_1549FC0(*a1, 0x20u);
        if ( !(*(_DWORD *)(v109 + 8) >> 8) )
          v109 = v160;
        sub_154DAA0((__int64)(a1 + 5), v109, *a1);
        sub_1549FC0(*a1, 0x20u);
        sub_15520E0(a1, v148, 0);
        sub_1549FC0(*a1, 0x28u);
        v110 = *(_DWORD *)(a2 + 20);
        v111 = (v110 & 0xFFFFFFF) - 1 - sub_154CB40(a2);
        if ( v111 )
        {
          v112 = 0;
          v161 = (unsigned int)(v111 - 1);
          while ( 1 )
          {
            v113 = sub_1560230(&v166, (unsigned int)v112);
            sub_1552790(a1, *(__int64 **)(a2 + 24 * (v112 - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))), v113);
            if ( v112 == v161 )
              break;
            ++v112;
            sub_1263B40(*a1, ", ");
          }
        }
        if ( (*(_WORD *)(a2 + 18) & 3) == 2 )
        {
          v135 = *(_QWORD *)(a2 + 40);
          if ( v135 )
          {
            v136 = *(_QWORD *)(v135 + 56);
            if ( v136 )
            {
              if ( *(_DWORD *)(*(_QWORD *)(v136 + 24) + 8LL) >> 8 )
                sub_1263B40(*a1, ", ...");
            }
          }
        }
        sub_1549FC0(*a1, 0x29u);
        if ( (unsigned __int8)sub_15602F0(&v166, 0xFFFFFFFFLL) )
        {
          v130 = sub_1263B40(*a1, " #");
          v165 = a1[4];
          v131 = sub_1560250(&v166);
          v134 = sub_154F2E0(v165, v131, v132, v133);
          sub_16E7AB0(v130, v134);
        }
        sub_15528B0(a1, a2 | 4);
        v21 = *(_BYTE *)(a2 + 16);
        goto LABEL_71;
      case 0x1Du:
        v114 = *(_QWORD *)(a2 + 64);
        v149 = *(__int64 **)(a2 - 72);
        v162 = **(_QWORD **)(v114 + 16);
        v166 = *(void **)(a2 + 56);
        if ( (*(_WORD *)(a2 + 18) & 0x7FFC) != 0 )
        {
          sub_1263B40(*a1, " ");
          sub_154A100((*(unsigned __int16 *)(a2 + 18) >> 2) & 0x3FFFDFFF, *a1);
        }
        if ( (unsigned __int8)sub_15602F0(&v166, 0) )
        {
          v139 = sub_1549FC0(*a1, 0x20u);
          sub_1560450(&v170, &v166, 0, 0);
          sub_16E7EE0(v139, v170, v171);
          sub_2240A30(&v170);
        }
        sub_1549FC0(*a1, 0x20u);
        if ( !(*(_DWORD *)(v114 + 8) >> 8) )
          v114 = v162;
        sub_154DAA0((__int64)(a1 + 5), v114, *a1);
        sub_1549FC0(*a1, 0x20u);
        sub_15520E0(a1, v149, 0);
        sub_1549FC0(*a1, 0x28u);
        v115 = *(_DWORD *)(a2 + 20);
        v116 = (v115 & 0xFFFFFFF) - 3 - sub_154CBE0(a2);
        if ( v116 )
        {
          v117 = 0;
          v163 = (unsigned int)(v116 - 1);
          while ( 1 )
          {
            v118 = sub_1560230(&v166, (unsigned int)v117);
            sub_1552790(a1, *(__int64 **)(a2 + 24 * (v117 - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))), v118);
            if ( v163 == v117 )
              break;
            ++v117;
            sub_1263B40(*a1, ", ");
          }
        }
        sub_1549FC0(*a1, 0x29u);
        if ( (unsigned __int8)sub_15602F0(&v166, 0xFFFFFFFFLL) )
        {
          v119 = sub_1263B40(*a1, " #");
          v164 = a1[4];
          v120 = sub_1560250(&v166);
          v123 = sub_154F2E0(v164, v120, v121, v122);
          sub_16E7AB0(v119, v123);
        }
        sub_15528B0(a1, a2 & 0xFFFFFFFFFFFFFFFBLL);
        sub_1263B40(*a1, "\n          to ");
        sub_15520E0(a1, *(__int64 **)(a2 - 48), 1);
        v108 = " unwind ";
        goto LABEL_231;
      case 0x35u:
        sub_1549FC0(*a1, 0x20u);
        v124 = *(_WORD *)(a2 + 18);
        if ( (v124 & 0x20) != 0 )
        {
          sub_1263B40(*a1, "inalloca ");
          v124 = *(_WORD *)(a2 + 18);
        }
        if ( (v124 & 0x40) != 0 )
          sub_1263B40(*a1, "swifterror ");
        sub_154DAA0((__int64)(a1 + 5), *(_QWORD *)(a2 + 56), *a1);
        if ( !*(_QWORD *)(a2 - 24)
          || (unsigned __int8)sub_15F8BF0(a2)
          || !(unsigned __int8)sub_1642F90(**(_QWORD **)(a2 - 24), 32) )
        {
          sub_1263B40(*a1, ", ");
          sub_15520E0(a1, *(__int64 **)(a2 - 24), 1);
        }
        if ( (unsigned int)(1 << *(_WORD *)(a2 + 18)) >> 1 )
        {
          v128 = sub_1263B40(*a1, ", align ");
          sub_16E7A90(v128, (unsigned int)(1 << *(_WORD *)(a2 + 18)) >> 1);
        }
        v125 = *(_DWORD *)(*(_QWORD *)a2 + 8LL) >> 8;
        if ( v125 )
        {
          v126 = sub_1263B40(*a1, ", addrspace(");
          v127 = sub_16E7A90(v126, v125);
          sub_1549FC0(v127, 0x29u);
          v21 = *(_BYTE *)(a2 + 16);
          goto LABEL_71;
        }
        goto LABEL_70;
    }
LABEL_91:
    if ( (unsigned int)(v53 - 60) <= 0xC )
    {
      if ( v28 )
      {
        sub_1549FC0(*a1, 0x20u);
        sub_15520E0(a1, v28, 1);
      }
      v129 = " to ";
    }
    else
    {
      if ( v21 != 82 )
      {
        if ( !v28 )
          goto LABEL_71;
        v54 = *a1;
        if ( v21 == 56 )
        {
          sub_1549FC0(v54, 0x20u);
          v55 = *(_QWORD *)(a2 + 56);
          v56 = *a1;
          v57 = (__int64)(a1 + 5);
        }
        else
        {
          if ( v21 != 54 )
          {
LABEL_97:
            v58 = v21 - 25;
            if ( v58 > 0x3Cu || (v62 = 1, ((0x1040000040000001uLL >> v58) & 1) == 0) )
            {
              v59 = *v28;
              v60 = 1;
              v154 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
              while ( 1 )
              {
                if ( v60 == v154 )
                {
                  v62 = 0;
                  sub_1549FC0(*a1, 0x20u);
                  sub_154DAA0((__int64)(a1 + 5), v59, *a1);
                  goto LABEL_104;
                }
                v61 = *(_QWORD **)(sub_13CF970(a2) + 24LL * v60);
                if ( v61 )
                {
                  if ( v59 != *v61 )
                    break;
                }
                ++v60;
              }
              v62 = 1;
            }
LABEL_104:
            v63 = 0;
            sub_1549FC0(*a1, 0x20u);
            v155 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
            while ( v155 != (_DWORD)v63 )
            {
              if ( (_DWORD)v63 )
                sub_1263B40(*a1, ", ");
              v64 = sub_13CF970(a2);
              v65 = 24 * v63++;
              sub_15520E0(a1, *(__int64 **)(v64 + v65), v62);
            }
            goto LABEL_70;
          }
          sub_1549FC0(v54, 0x20u);
          v56 = *a1;
          v55 = *(_QWORD *)a2;
          v57 = (__int64)(a1 + 5);
        }
        sub_154DAA0(v57, v55, v56);
        sub_1549FC0(*a1, 0x2Cu);
        v21 = *(_BYTE *)(a2 + 16);
        goto LABEL_97;
      }
      if ( v28 )
      {
        sub_1549FC0(*a1, 0x20u);
        sub_15520E0(a1, v28, 1);
      }
      v129 = ", ";
    }
    sub_1263B40(*a1, v129);
    sub_154DAA0((__int64)(a1 + 5), *(_QWORD *)a2, *a1);
    goto LABEL_70;
  }
  sub_1549FC0(*a1, 0x20u);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v71 = *(__int64 ***)(a2 - 8);
  else
    v71 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  sub_15520E0(a1, *v71, 1);
  sub_1263B40(*a1, ", ");
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v72 = *(_QWORD *)(a2 - 8);
  else
    v72 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v73 = 0;
  sub_15520E0(a1, *(__int64 **)(v72 + 24), 1);
  sub_1263B40(*a1, " [");
  v145 = ((*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1) - 1;
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1 != 1 )
  {
    do
    {
      v77 = 2 * ++v73;
      sub_1263B40(*a1, "\n    ");
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v74 = *(_QWORD *)(a2 - 8);
      else
        v74 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      sub_15520E0(a1, *(__int64 **)(v74 + 24LL * v77), 1);
      sub_1263B40(*a1, ", ");
      v75 = 24;
      if ( (_DWORD)v73 != -1 )
        v75 = 24LL * (v77 + 1);
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v76 = *(_QWORD *)(a2 - 8);
      else
        v76 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      sub_15520E0(a1, *(__int64 **)(v76 + v75), 1);
    }
    while ( v145 != v73 );
  }
  sub_1263B40(*a1, "\n  ]");
  v21 = *(_BYTE *)(a2 + 16);
LABEL_71:
  if ( v21 == 54 || v21 == 55 )
  {
    if ( (unsigned __int8)sub_15F32D0(a2) )
    {
      v67 = *(unsigned __int16 *)(a2 + 18);
      v68 = *(_BYTE *)(a2 + 56);
      v69 = sub_16498A0(a2);
      v70 = (v67 >> 7) & 7;
      if ( v70 )
        sub_1549590(a1, v69, v70, v68);
    }
    if ( (unsigned int)(1 << (*(unsigned __int16 *)(a2 + 18) >> 1)) >> 1 )
    {
      v66 = sub_1263B40(*a1, ", align ");
      sub_16E7A90(v66, (unsigned int)(1 << (*(unsigned __int16 *)(a2 + 18) >> 1) >> 1));
    }
  }
  else if ( v21 == 58 )
  {
    v153 = *(_BYTE *)(a2 + 56);
    v47 = *(_WORD *)(a2 + 18);
    v48 = sub_16498A0(a2);
    v49 = (v47 >> 2) & 7;
    v50 = (unsigned __int8)v47 >> 5;
    if ( v153 != 1 )
      sub_15494B0(a1, v48, v153);
    v51 = sub_1263B40(*a1, " ");
    sub_1263B40(v51, (&off_4C6F320)[v49]);
    v52 = sub_1263B40(*a1, " ");
    sub_1263B40(v52, (&off_4C6F320)[v50]);
  }
  else
  {
    if ( v21 == 59 )
    {
      v88 = *(_BYTE *)(a2 + 56);
      v89 = (*(unsigned __int16 *)(a2 + 18) >> 2) & 7;
    }
    else
    {
      if ( v21 != 57 )
        goto LABEL_75;
      v88 = *(_BYTE *)(a2 + 56);
      v89 = (*(unsigned __int16 *)(a2 + 18) >> 1) & 0x7FFFBFFF;
    }
    v90 = sub_16498A0(a2);
    if ( v89 )
      sub_1549590(a1, v90, v89, v88);
  }
LABEL_75:
  v43 = *(_QWORD *)(a2 + 48) == 0;
  v170 = (const char *)&v172;
  v171 = 0x400000000LL;
  if ( !v43 || *(__int16 *)(a2 + 18) < 0 )
    sub_161F840(a2, &v170);
  sub_1550BA0(a1, (unsigned int *)&v170, ", ", 2u);
  sub_1552170(a1, (const char *)a2);
  if ( v170 != (const char *)&v172 )
    _libc_free((unsigned __int64)v170);
}
