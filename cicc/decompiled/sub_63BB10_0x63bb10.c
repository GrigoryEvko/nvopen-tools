// Function: sub_63BB10
// Address: 0x63bb10
//
__int64 __fastcall sub_63BB10(__int64 a1, _BOOL8 a2)
{
  unsigned int v2; // r15d
  char v4; // al
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // r12
  int i; // r15d
  char j; // al
  __int64 v11; // r10
  __int64 v12; // r11
  bool v13; // r9
  __int64 v14; // rax
  char v15; // al
  int v16; // eax
  int v17; // eax
  _QWORD *v18; // rdx
  __int64 v19; // rcx
  bool v20; // r9
  __int64 v21; // r11
  __int64 v22; // rdi
  int v23; // eax
  __int64 v24; // rsi
  _QWORD *v25; // rdi
  __int64 v26; // rax
  _BOOL8 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // r11
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // r13
  _BOOL4 v34; // r9d
  __int64 v35; // rdx
  __int64 v36; // r9
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // r10
  int v41; // ecx
  __int64 v42; // rdi
  __int64 v43; // rax
  __int64 v44; // r10
  __int64 v45; // r11
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r11
  __int64 v49; // rax
  __int64 v50; // r11
  __int64 v51; // rax
  __int64 v52; // r10
  __int64 v53; // r11
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // r8
  __int64 v57; // rdx
  bool v58; // r12
  __int64 v59; // rax
  __int64 v60; // rax
  int v61; // eax
  __int64 v62; // rax
  __int64 v63; // r11
  __int64 v64; // r10
  __int64 v65; // r8
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // r8
  __int64 v71; // r10
  __int64 v72; // rdx
  __int64 v73; // rdi
  __int64 v74; // rcx
  int v75; // eax
  int v76; // eax
  unsigned __int8 v77; // al
  char v78; // dl
  int v79; // eax
  int v80; // eax
  __int64 v81; // rax
  __int64 v82; // r11
  __int64 v83; // rcx
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // [rsp-8h] [rbp-98h]
  __int64 v89; // [rsp+8h] [rbp-88h]
  __int64 v90; // [rsp+8h] [rbp-88h]
  __int64 v91; // [rsp+8h] [rbp-88h]
  __int64 v92; // [rsp+8h] [rbp-88h]
  __int64 v93; // [rsp+8h] [rbp-88h]
  __int64 v94; // [rsp+8h] [rbp-88h]
  __int64 v95; // [rsp+10h] [rbp-80h]
  __int64 v96; // [rsp+10h] [rbp-80h]
  __int64 v97; // [rsp+10h] [rbp-80h]
  __int64 v98; // [rsp+10h] [rbp-80h]
  __int64 v99; // [rsp+10h] [rbp-80h]
  __int64 v100; // [rsp+10h] [rbp-80h]
  __int64 v101; // [rsp+10h] [rbp-80h]
  __int64 v102; // [rsp+10h] [rbp-80h]
  __int64 v103; // [rsp+10h] [rbp-80h]
  __int64 v104; // [rsp+10h] [rbp-80h]
  __int64 v105; // [rsp+10h] [rbp-80h]
  __int64 v106; // [rsp+10h] [rbp-80h]
  _BOOL4 v107; // [rsp+18h] [rbp-78h]
  __int64 v108; // [rsp+18h] [rbp-78h]
  __int64 v109; // [rsp+18h] [rbp-78h]
  __int64 v110; // [rsp+18h] [rbp-78h]
  __int64 v111; // [rsp+20h] [rbp-70h]
  __int64 v112; // [rsp+20h] [rbp-70h]
  __int64 v113; // [rsp+20h] [rbp-70h]
  __int64 v114; // [rsp+28h] [rbp-68h]
  __int64 v115; // [rsp+28h] [rbp-68h]
  __int64 v116; // [rsp+28h] [rbp-68h]
  __int64 v117; // [rsp+28h] [rbp-68h]
  __int64 v118; // [rsp+28h] [rbp-68h]
  __int64 v119; // [rsp+28h] [rbp-68h]
  __int64 v120; // [rsp+28h] [rbp-68h]
  __int64 v121; // [rsp+28h] [rbp-68h]
  __int64 v122; // [rsp+28h] [rbp-68h]
  __int64 v123; // [rsp+28h] [rbp-68h]
  __int64 v124; // [rsp+28h] [rbp-68h]
  bool v125; // [rsp+28h] [rbp-68h]
  bool v126; // [rsp+30h] [rbp-60h]
  bool v127; // [rsp+30h] [rbp-60h]
  __int64 v128; // [rsp+30h] [rbp-60h]
  __int64 v129; // [rsp+30h] [rbp-60h]
  __int64 v130; // [rsp+30h] [rbp-60h]
  __int64 v131; // [rsp+30h] [rbp-60h]
  __int64 v132; // [rsp+30h] [rbp-60h]
  __int64 v133; // [rsp+30h] [rbp-60h]
  __int64 v134; // [rsp+30h] [rbp-60h]
  __int64 v135; // [rsp+30h] [rbp-60h]
  __int64 v136; // [rsp+30h] [rbp-60h]
  __int64 v137; // [rsp+30h] [rbp-60h]
  __int64 v138; // [rsp+30h] [rbp-60h]
  __int64 v139; // [rsp+30h] [rbp-60h]
  __int64 v140; // [rsp+30h] [rbp-60h]
  unsigned __int8 v141; // [rsp+38h] [rbp-58h]
  __int64 v142; // [rsp+38h] [rbp-58h]
  bool v143; // [rsp+38h] [rbp-58h]
  __int64 v144; // [rsp+38h] [rbp-58h]
  __int64 v145; // [rsp+38h] [rbp-58h]
  __int64 v146; // [rsp+38h] [rbp-58h]
  __int64 v147; // [rsp+38h] [rbp-58h]
  __int64 v148; // [rsp+40h] [rbp-50h] BYREF
  __int64 v149; // [rsp+48h] [rbp-48h] BYREF
  _QWORD v150[8]; // [rsp+50h] [rbp-40h] BYREF

  v2 = 0;
  v148 = 0;
  if ( dword_4F077C4 != 2 )
    return v2;
  v4 = *(_BYTE *)(a1 + 80);
  v6 = a2;
  if ( v4 != 7 && v4 != 9 )
    return v2;
  v7 = *(_QWORD *)(a1 + 88);
  v2 = 0;
  if ( !v7 )
    return v2;
  if ( (unsigned int)sub_8D3F60(*(_QWORD *)(v7 + 120)) )
  {
    LODWORD(v150[0]) = 0;
    v34 = a2;
    a2 = 1;
    sub_84CF20(*(_QWORD *)(v7 + 120), 1, 0, 0, 0, v34, v7 + 120, (__int64)v150);
  }
  v8 = *(_QWORD *)(v7 + 120);
  i = 0;
  v141 = *(_BYTE *)(v7 + 136);
  if ( (*(_BYTE *)(v8 + 140) & 0xFB) == 8 )
  {
    a2 = dword_4F077C4 != 2;
    v15 = sub_8D4C10(v8, a2);
    v8 = *(_QWORD *)(v7 + 120);
    for ( i = v15 & 1; *(_BYTE *)(v8 + 140) == 12; v8 = *(_QWORD *)(v8 + 160) )
      ;
  }
  if ( (unsigned int)sub_8D3410(v8) )
  {
    v11 = sub_8D40F0(v8);
    for ( j = *(_BYTE *)(v11 + 140); j == 12; j = *(_BYTE *)(v11 + 140) )
      v11 = *(_QWORD *)(v11 + 160);
  }
  else
  {
    j = *(_BYTE *)(v8 + 140);
    v11 = v8;
  }
  if ( (unsigned __int8)(j - 9) > 2u )
  {
    if ( dword_4F077C4 == 2 )
    {
      v142 = v11;
      v16 = sub_8D23B0(v11);
      v11 = v142;
      if ( v16 )
      {
        sub_8AE000(v142);
        v11 = v142;
      }
    }
    goto LABEL_16;
  }
  v12 = *(_QWORD *)(*(_QWORD *)v11 + 96LL);
  v13 = (*(_BYTE *)(v11 + 177) & 0x20) != 0;
  if ( dword_4F077C4 == 2 )
  {
    v113 = *(_QWORD *)(*(_QWORD *)v11 + 96LL);
    v125 = (*(_BYTE *)(v11 + 177) & 0x20) != 0;
    v140 = v11;
    v79 = sub_8D23B0(v11);
    v11 = v140;
    v13 = v125;
    v12 = v113;
    if ( v79 )
    {
      sub_8AE000(v140);
      v12 = v113;
      v13 = v125;
      v11 = v140;
    }
  }
  if ( !v12
    || *(char *)(v12 + 178) < 0
    && !*(_QWORD *)(v12 + 8)
    && !*(_QWORD *)(v12 + 24)
    && ((*(_BYTE *)(v7 + 172) & 8) == 0 || (*(_BYTE *)(v11 + 179) & 4) != 0)
    || (v114 = v12, v126 = v13, *(_BYTE *)(v7 + 136) == 1)
    || (v111 = v11, v17 = sub_8D23B0(v8), v11 = v111, v107 = v17) )
  {
LABEL_16:
    if ( (*(_BYTE *)(v7 + 173) & 4) != 0 )
    {
      v2 = 0;
      v14 = sub_725A70(0);
      sub_630370(v7, v14, &v148, v6, 0, 0);
    }
    else
    {
      return (unsigned int)sub_8DD3B0(v11) != 0;
    }
    return v2;
  }
  v20 = v126;
  v21 = v114;
  if ( *(_BYTE *)(a1 + 80) == 9 )
  {
    v112 = 0;
    if ( dword_4F04C44 == -1 )
    {
      v18 = qword_4F04C68;
      v37 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( (*(_BYTE *)(v37 + 6) & 6) == 0 && *(_BYTE *)(v37 + 4) != 12 )
      {
        a2 = 1;
        v97 = v11;
        sub_8646E0(*(_QWORD *)(a1 + 64), 1);
        v20 = v126;
        v21 = v114;
        v11 = v97;
      }
    }
  }
  else
  {
    if ( dword_4D048B8 && v141 <= 2u )
    {
      v112 = 0;
      if ( dword_4F04C58 != -1 )
      {
        a2 = 0;
        v102 = v11;
        sub_733780(0, 0, 0, 1, 0);
        v20 = v126;
        v21 = v114;
        v11 = v102;
        v112 = qword_4F06BC0;
      }
    }
    else
    {
      v112 = 0;
    }
    if ( (*(_BYTE *)(a1 + 81) & 0x10) == 0 )
    {
      v22 = *(_QWORD *)(a1 + 64);
      if ( v22 )
      {
        a2 = 0;
        v95 = v11;
        v115 = v21;
        v127 = v20;
        sub_864360(v22, 0);
        v11 = v95;
        v21 = v115;
        v20 = v127;
      }
    }
  }
  if ( v20 )
  {
    v24 = v11;
    v25 = (_QWORD *)v11;
    v130 = v11;
    v117 = sub_87CF10(v11, v11, v6);
    if ( !v117 )
    {
      if ( (*(_BYTE *)(v7 + 172) & 0x18) != 0 )
      {
        v2 = 1;
        goto LABEL_50;
      }
      v2 = 1;
      v86 = sub_725A70(5);
      *(_BYTE *)(v86 + 72) &= ~1u;
      v48 = v86;
      *(_QWORD *)(v86 + 56) = 0;
      goto LABEL_87;
    }
    if ( (*(_BYTE *)(v7 + 172) & 0x18) == 0 )
    {
      v2 = 1;
      v60 = sub_725A70(5);
      *(_BYTE *)(v60 + 72) &= ~1u;
      v48 = v60;
      *(_QWORD *)(v60 + 56) = 0;
LABEL_119:
      *(_QWORD *)(v48 + 16) = v117;
      *(_BYTE *)(v117 + 193) |= 0x40u;
      goto LABEL_87;
    }
    v39 = sub_724DC0(v25, v24, v35, v27, v28, v36);
    v40 = v130;
    v149 = v39;
    if ( !unk_4F04C50 || (*(_BYTE *)(*(_QWORD *)(unk_4F04C50 + 32LL) + 193LL) & 4) != 0 )
    {
      v2 = 1;
      goto LABEL_139;
    }
    v129 = 0;
    v2 = 1;
    goto LABEL_79;
  }
  if ( (*(_BYTE *)(v21 + 176) & 0x4A) != 0
    || *(_QWORD *)(v21 + 8)
    && (v116 = v11, v128 = v21, v23 = sub_879360(v21, a2, v18, v19), v21 = v128, v11 = v116, v23) )
  {
    v101 = v21;
    v119 = v11;
    v59 = sub_87CD20(v11, v6, v11, v150);
    v11 = v119;
    v129 = v59;
    v21 = v101;
    if ( !LODWORD(v150[0]) && (!v59 || (*(_BYTE *)(v59 + 193) & 0x10) != 0) )
    {
      if ( *(_QWORD *)(v101 + 16) )
      {
        sub_876D90(v119, v119, v6, 1, 0);
        v21 = v101;
        v11 = v119;
      }
      if ( i )
      {
        if ( (*(_BYTE *)(v11 + 179) & 1) == 0 )
        {
          v105 = v21;
          v124 = v11;
          v76 = sub_630090(v11);
          v11 = v124;
          v21 = v105;
          if ( !(v76 | (unsigned int)qword_4D0495C | HIDWORD(qword_4D0495C)) )
          {
            v77 = 7;
            if ( !dword_4D04964 )
            {
              v78 = *(_BYTE *)(v105 + 176);
              if ( (v78 & 1) != 0 || !*(_QWORD *)(v105 + 16) && *(_QWORD *)(v105 + 8) )
                v77 = (v78 & 4) == 0 ? 7 : 5;
            }
            sub_686750(v77, 811, v6, a1, v124);
            v21 = v105;
            v11 = v124;
          }
        }
      }
    }
    v2 = 1;
  }
  else if ( !i || (*(_BYTE *)(v11 + 179) & 1) != 0 )
  {
    v120 = v21;
    v135 = v11;
    v61 = sub_876D90(v11, v11, v6, 1, 0);
    v11 = v135;
    v21 = v120;
    if ( !v61 || (v129 = 0, v2 = 1, *(char *)(v120 + 178) < 0) )
    {
      v129 = 0;
      v2 = *(_BYTE *)(v11 + 179) & 1;
    }
  }
  else
  {
    v129 = 0;
    v2 = 0;
  }
  v24 = v11;
  v25 = (_QWORD *)v11;
  v89 = v21;
  v96 = v11;
  v26 = sub_87CF10(v11, v11, v6);
  v117 = v26;
  v30 = v89;
  v31 = v26 | v129;
  if ( __PAIR128__(v129, v26) == 0 )
  {
    if ( (*(_BYTE *)(v7 + 173) & 4) == 0 )
    {
      if ( unk_4F04C50 && *(char *)(v89 + 178) >= 0 )
      {
        v25 = (_QWORD *)v7;
        sub_86F660(v7);
      }
      if ( (*(_BYTE *)(v7 + 172) & 8) == 0 )
        goto LABEL_50;
      if ( v141 <= 2u )
      {
        v32 = sub_724D80(10);
        *(_BYTE *)(v7 + 177) = 1;
        *(_QWORD *)(v7 + 184) = v32;
        v33 = v32;
        if ( !(unsigned int)sub_72FDF0(v8, v32) )
          sub_72C970(v33);
        *(_BYTE *)(v33 + 171) |= 2u;
        sub_7296C0(v150);
        v24 = sub_725A70(1);
        sub_71AAB0(v33, v24);
        v25 = (_QWORD *)LODWORD(v150[0]);
        sub_729730(LODWORD(v150[0]));
        goto LABEL_50;
      }
      v146 = sub_724D50(10);
      v139 = sub_725A70(2);
      *(_QWORD *)(v139 + 56) = v146;
      v75 = sub_72FDF0(v8, v146);
      v48 = v139;
      if ( !v75 )
      {
        sub_72C970(v146);
        v48 = v139;
      }
      goto LABEL_87;
    }
    if ( (*(_BYTE *)(v7 + 172) & 0x18) != 0 )
      goto LABEL_50;
    goto LABEL_126;
  }
  if ( (*(_BYTE *)(v7 + 172) & 0x18) != 0 )
  {
    v87 = sub_724DC0(v25, v24, v31, v27, v28, v29);
    v40 = v96;
    v149 = v87;
    if ( unk_4F04C50 )
    {
      v41 = 1;
      if ( (*(_BYTE *)(*(_QWORD *)(unk_4F04C50 + 32LL) + 193LL) & 4) != 0 )
      {
LABEL_80:
        v42 = v129;
        if ( v129 )
        {
          v131 = v40;
          v43 = sub_62FD00(v42, 0, 1, v41);
          v44 = v131;
          v45 = v43;
          if ( *(_BYTE *)(v8 + 140) == 8 && *(char *)(v8 + 168) >= 0 )
          {
            v109 = v131;
            v137 = v43;
            v69 = sub_725A70(6);
            v70 = 0;
            v71 = v109;
            if ( (*(_BYTE *)(v8 + 169) & 1) == 0 )
            {
              v70 = 1;
              if ( *(_QWORD *)(v8 + 128) )
                v70 = *(_QWORD *)(v8 + 128) / *(_QWORD *)(v109 + 128);
            }
            v72 = v109;
            v73 = v137;
            v110 = v69;
            v138 = v71;
            sub_63BA50(v73, v8, v72, v69, v70);
            v74 = v110;
            v44 = v138;
            v107 = 1;
            v45 = v74;
          }
          goto LABEL_82;
        }
LABEL_139:
        v136 = v40;
        v67 = sub_725A70(1);
        v44 = v136;
        v45 = v67;
LABEL_82:
        if ( v117 )
        {
          *(_QWORD *)(v45 + 16) = v117;
          *(_BYTE *)(v117 + 193) |= 0x40u;
        }
        *(_QWORD *)(v45 + 8) = v7;
        v24 = v6;
        v98 = v44;
        v118 = v45;
        v150[0] = 0;
        v150[1] = 0;
        if ( (unsigned int)sub_7A1C60(v45, v6, v8, 1, v149, (unsigned int)v150, 0) )
        {
          v62 = sub_724E50(&v149, v6, v46, v47, v88);
          v63 = v118;
          v64 = v98;
          v65 = v62;
        }
        else
        {
          if ( dword_4F04C44 != -1
            || (v68 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v68 + 6) & 6) != 0)
            || *(_BYTE *)(v68 + 4) == 12
            || (*(_BYTE *)(v7 + 172) & 0x18) == 0x10 && *(_BYTE *)(v118 + 48) <= 2u )
          {
            sub_724E30(&v149);
            sub_67E3D0(v150);
            v48 = v118;
            *(_QWORD *)(v118 + 8) = 0;
LABEL_87:
            v24 = v48;
            v25 = (_QWORD *)v7;
            sub_630370(v7, v48, &v148, v6, 0, 0);
            goto LABEL_50;
          }
          v93 = v98;
          v104 = v118;
          v24 = (__int64)v150;
          v122 = sub_67D9D0(2807, v6);
          sub_67E370(v122, v150);
          sub_685910(v122);
          v123 = sub_72C9A0();
          sub_724E30(&v149);
          v64 = v93;
          v63 = v104;
          v65 = v123;
        }
        v25 = v150;
        v103 = v65;
        v121 = v63;
        v92 = v64;
        sub_67E3D0(v150);
        v48 = v121;
        v28 = v103;
        *(_QWORD *)(v121 + 8) = 0;
        if ( !v103 )
          goto LABEL_87;
        if ( v92 != v8 )
        {
          v24 = dword_4F07588;
          if ( !dword_4F07588 || (v66 = *(_QWORD *)(v8 + 32), *(_QWORD *)(v92 + 32) != v66) || !v66 )
          {
            v27 = v107;
            if ( !v107 )
            {
              v25 = (_QWORD *)v103;
              v24 = v8;
              v28 = sub_62FF50(v103, v8);
            }
          }
        }
        if ( v141 > 2u || dword_4F04C58 != -1 )
        {
          v145 = v28;
          v48 = sub_725A70(2);
          *(_QWORD *)(v48 + 56) = v145;
          *(_BYTE *)(v48 + 50) = (*(_BYTE *)(v145 + 170) >> 6 << 7) | *(_BYTE *)(v48 + 50) & 0x7F;
          goto LABEL_87;
        }
        *(_BYTE *)(v7 + 177) = 1;
        *(_QWORD *)(v7 + 184) = v28;
        goto LABEL_50;
      }
    }
LABEL_79:
    v41 = 0;
    goto LABEL_80;
  }
  v90 = v96;
  v99 = v30;
  if ( v129 )
  {
    v49 = sub_724DC0(v25, v24, v31, v27, v28, v29);
    v50 = v99;
    v150[0] = v49;
    if ( unk_4F04C50 )
      v107 = (*(_BYTE *)(*(_QWORD *)(unk_4F04C50 + 32LL) + 193LL) & 4) != 0;
    v24 = 0;
    v100 = v90;
    v91 = v50;
    v51 = sub_62FD00(v129, 0, 1, v107);
    v52 = v100;
    v108 = v51;
    if ( (*(_BYTE *)(v129 + 193) & 2) != 0
      && (*(_BYTE *)(v7 + 173) & 4) == 0
      && *(_BYTE *)(v51 + 48) == 5
      && (*(_QWORD *)(v51 + 8) = v7,
          v24 = 1,
          v106 = v91,
          v94 = v52,
          v80 = sub_71AAF0(v51, 1, 0, (*(_BYTE *)(v129 + 193) & 4) != 0, v6, v150[0]),
          v52 = v94,
          *(_QWORD *)(v108 + 8) = 0,
          v80) )
    {
      v81 = sub_740630(v150[0]);
      v82 = v106;
      v83 = v81;
      if ( v94 != v8 )
      {
        if ( !dword_4F07588 || (v84 = *(_QWORD *)(v8 + 32), *(_QWORD *)(v94 + 32) != v84) || !v84 )
        {
          v24 = v8;
          v85 = sub_62FF50(v83, v8);
          v82 = v106;
          v83 = v85;
        }
      }
      if ( v141 <= 2u && (!*(_QWORD *)(v82 + 24) || (*(_BYTE *)(v82 + 177) & 2) != 0) )
      {
        *(_BYTE *)(v7 + 177) = 1;
        if ( dword_4F04C58 == -1 )
        {
          *(_QWORD *)(v7 + 184) = v83;
        }
        else
        {
          v24 = 0;
          sub_7333B0(v7, 0, 1, v83, 0);
        }
        v25 = v150;
        sub_724E30(v150);
        goto LABEL_50;
      }
      v147 = v83;
      v53 = sub_725A70(2);
      *(_QWORD *)(v53 + 56) = v147;
      *(_BYTE *)(v53 + 50) = (*(_BYTE *)(v147 + 170) >> 6 << 7) | *(_BYTE *)(v53 + 50) & 0x7F;
      v143 = v117 != 0;
    }
    else
    {
      v53 = v108;
      v143 = v117 != 0;
      if ( v52 != v8 )
      {
        if ( !dword_4F07588 || (v54 = *(_QWORD *)(v8 + 32), *(_QWORD *)(v52 + 32) != v54) || !v54 )
        {
          v132 = v52;
          v55 = sub_725A70(6);
          v56 = 0;
          if ( (*(_BYTE *)(v8 + 169) & 1) == 0 )
          {
            v56 = 1;
            if ( *(_QWORD *)(v8 + 128) )
              v56 = *(_QWORD *)(v8 + 128) / *(_QWORD *)(v132 + 128);
          }
          v57 = v132;
          v24 = v8;
          v133 = v55;
          sub_63BA50(v108, v8, v57, v55, v56);
          v53 = v133;
          if ( (*(_BYTE *)(v8 + 169) & 1) != 0 )
            *(_BYTE *)(v133 + 50) |= 0x80u;
          v58 = v143 && dword_4D048B8 != 0;
          if ( v58 )
          {
            v24 = 1;
            *(_QWORD *)(v108 + 16) = v117;
            *(_BYTE *)(v117 + 193) |= 0x40u;
            sub_734250(v108, 1);
            v143 = v58;
            v53 = v133;
          }
        }
      }
    }
    v25 = v150;
    v134 = v53;
    sub_724E30(v150);
    v48 = v134;
    goto LABEL_107;
  }
  v31 = v26;
LABEL_126:
  v25 = 0;
  v144 = v31;
  v48 = sub_725A70(0);
  v117 = v144;
  v143 = v144 != 0;
LABEL_107:
  if ( v48 )
  {
    if ( !v143 )
      goto LABEL_87;
    goto LABEL_119;
  }
LABEL_50:
  if ( *(_BYTE *)(a1 + 80) == 9 )
  {
    if ( dword_4F04C44 == -1 )
    {
      v38 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( (*(_BYTE *)(v38 + 6) & 6) == 0 && *(_BYTE *)(v38 + 4) != 12 )
        sub_866010(v25, v24, qword_4F04C68, v27, v28);
    }
  }
  else
  {
    if ( v112 )
      sub_630710(v112, v148, 0);
    if ( (*(_BYTE *)(a1 + 81) & 0x10) == 0 && *(_QWORD *)(a1 + 64) )
      sub_8645D0();
  }
  return v2;
}
