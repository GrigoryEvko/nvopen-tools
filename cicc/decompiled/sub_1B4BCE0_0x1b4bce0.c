// Function: sub_1B4BCE0
// Address: 0x1b4bce0
//
_BOOL8 __fastcall sub_1B4BCE0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // r15
  __int64 v12; // r12
  __int64 v13; // r13
  bool v14; // zf
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r13
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // r14
  __int64 v22; // r13
  __int64 v23; // r13
  __int64 *v24; // r14
  __int64 **i; // r12
  __int64 v26; // rbx
  __int64 *v27; // r14
  __int64 v28; // rsi
  __int64 *v29; // rax
  __int64 *v30; // rdi
  __int64 *v31; // rcx
  __int64 **v32; // rbx
  __int64 *j; // r8
  __int64 v34; // r14
  __int64 *v35; // r12
  __int64 v36; // rsi
  __int64 *v37; // rax
  __int64 *v38; // rdi
  __int64 *v39; // rcx
  __int64 *v40; // rax
  __int64 *v41; // r14
  __int64 v42; // r13
  __int64 *v43; // rbx
  __int64 v44; // rax
  __int64 *v45; // r12
  __int64 v46; // r14
  __int64 v47; // rax
  __int64 v48; // r15
  __int64 v49; // r10
  unsigned int v50; // eax
  unsigned __int16 v51; // ax
  __int64 v52; // rax
  __int64 v53; // r15
  __int64 v54; // r13
  __int64 *v55; // rax
  __int64 v56; // rbx
  __int64 v57; // r12
  __int64 v58; // rbx
  __int64 v59; // r12
  __int64 v60; // rbx
  __int64 v61; // r13
  __int64 v62; // r12
  double v63; // xmm4_8
  double v64; // xmm5_8
  __int64 v65; // r8
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rsi
  __int64 v71; // r9
  __int64 v72; // r11
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rax
  double v76; // xmm4_8
  double v77; // xmm5_8
  __int64 *v78; // rsi
  _QWORD *v79; // rax
  _QWORD *v80; // rax
  __int64 v81; // r9
  __int64 v82; // rsi
  unsigned int v83; // r13d
  unsigned int v84; // eax
  unsigned int v85; // r8d
  unsigned int v86; // esi
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // [rsp+10h] [rbp-1A0h]
  __int64 v90; // [rsp+10h] [rbp-1A0h]
  __int64 v91; // [rsp+10h] [rbp-1A0h]
  __int64 v92; // [rsp+18h] [rbp-198h]
  __int64 *v93; // [rsp+18h] [rbp-198h]
  __int64 v94; // [rsp+18h] [rbp-198h]
  __int64 v95; // [rsp+18h] [rbp-198h]
  __int64 v96; // [rsp+18h] [rbp-198h]
  __int64 v97; // [rsp+18h] [rbp-198h]
  __int64 v98; // [rsp+28h] [rbp-188h]
  __int64 *v99; // [rsp+28h] [rbp-188h]
  __int64 v100; // [rsp+28h] [rbp-188h]
  _QWORD *v101; // [rsp+28h] [rbp-188h]
  unsigned int v102; // [rsp+28h] [rbp-188h]
  bool v103; // [rsp+35h] [rbp-17Bh]
  char v104; // [rsp+36h] [rbp-17Ah]
  char v105; // [rsp+37h] [rbp-179h]
  __int64 v107; // [rsp+40h] [rbp-170h]
  __int64 v108; // [rsp+48h] [rbp-168h]
  __int64 v109; // [rsp+50h] [rbp-160h]
  __int64 v110; // [rsp+50h] [rbp-160h]
  __int64 v111; // [rsp+58h] [rbp-158h]
  bool v112; // [rsp+58h] [rbp-158h]
  __int64 v113; // [rsp+58h] [rbp-158h]
  __int64 v114; // [rsp+58h] [rbp-158h]
  __int64 v115; // [rsp+60h] [rbp-150h]
  __int64 v116; // [rsp+68h] [rbp-148h]
  __int64 v117[2]; // [rsp+70h] [rbp-140h] BYREF
  __int64 v118; // [rsp+80h] [rbp-130h]
  __int64 v119; // [rsp+90h] [rbp-120h] BYREF
  __int64 *v120; // [rsp+98h] [rbp-118h]
  __int64 *v121; // [rsp+A0h] [rbp-110h]
  __int64 v122; // [rsp+A8h] [rbp-108h]
  int v123; // [rsp+B0h] [rbp-100h]
  _BYTE v124[40]; // [rsp+B8h] [rbp-F8h] BYREF
  __int64 v125; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 *v126; // [rsp+E8h] [rbp-C8h]
  __int64 *v127; // [rsp+F0h] [rbp-C0h]
  __int64 v128; // [rsp+F8h] [rbp-B8h]
  int v129; // [rsp+100h] [rbp-B0h]
  _BYTE v130[40]; // [rsp+108h] [rbp-A8h] BYREF
  __int64 v131; // [rsp+130h] [rbp-80h] BYREF
  __int64 v132; // [rsp+138h] [rbp-78h]
  __int64 *v133[14]; // [rsp+140h] [rbp-70h] BYREF

  v11 = *(_QWORD *)(a2 - 48);
  v12 = *(_QWORD *)(a2 - 24);
  v109 = *(_QWORD *)(a1 - 24);
  v116 = *(_QWORD *)(a1 - 48);
  v13 = sub_157F1C0(v11);
  v14 = v11 == sub_157F1C0(v12);
  v15 = v13;
  if ( v14 )
    v15 = v11;
  v115 = v15;
  if ( !v15 )
    return 0;
  v16 = *(_QWORD *)(a2 + 40);
  if ( v16 == v116 )
  {
    if ( v15 == v11 )
    {
      v44 = v109;
      v11 = v12;
      v109 = 0;
      v12 = 0;
      v104 = 1;
      v116 = v44;
      v105 = 1;
      goto LABEL_10;
    }
    v104 = 0;
    v105 = 1;
    v116 = v109;
    v109 = 0;
    goto LABEL_8;
  }
  if ( v15 == v11 )
  {
    if ( v16 == v109 )
    {
      v11 = v12;
      v104 = 1;
      v12 = 0;
      v109 = 0;
      v105 = 0;
      goto LABEL_10;
    }
    v19 = v12;
    v104 = 1;
    v12 = v115;
    v11 = v19;
    goto LABEL_7;
  }
  v104 = 0;
  if ( v16 != v109 )
  {
LABEL_7:
    v105 = 0;
    goto LABEL_8;
  }
  v105 = 0;
  v109 = 0;
LABEL_8:
  if ( v115 == v12 )
    v12 = 0;
LABEL_10:
  v17 = *(_QWORD *)(a1 + 40);
  v111 = *(_QWORD *)(a2 + 40);
  if ( v17 != sub_157F0B0(v116) )
    return 0;
  if ( v111 != sub_157F1C0(v116) )
    return 0;
  v20 = *(_QWORD *)(a2 + 40);
  if ( v20 != sub_157F0B0(v11) )
    return 0;
  if ( v115 != sub_157F1C0(v11) )
    return 0;
  if ( v109 )
  {
    v21 = *(_QWORD *)(a1 + 40);
    v22 = *(_QWORD *)(a2 + 40);
    if ( v21 != sub_157F0B0(v109) || v22 != sub_157F1C0(v109) )
      return 0;
  }
  if ( v12 )
  {
    v23 = *(_QWORD *)(a2 + 40);
    if ( v23 != sub_157F0B0(v12) || v115 != sub_157F1C0(v12) )
      return 0;
  }
  if ( !(v103 = sub_1648CD0(*(_QWORD *)(a2 + 40), 2)) )
    return 0;
  v24 = (__int64 *)v109;
  v119 = 0;
  v120 = (__int64 *)v124;
  v121 = (__int64 *)v124;
  v126 = (__int64 *)v130;
  v127 = (__int64 *)v130;
  v122 = 4;
  v132 = v116;
  v131 = v109;
  v123 = 0;
  v125 = 0;
  v128 = 4;
  v129 = 0;
  v108 = v12;
  for ( i = (__int64 **)&v131; ; v24 = *i )
  {
    if ( v24 )
    {
      v26 = v24[6];
      v27 = v24 + 5;
      while ( v27 != (__int64 *)v26 )
      {
LABEL_33:
        if ( !v26 )
          BUG();
        if ( *(_BYTE *)(v26 - 8) == 55 )
        {
          v28 = *(_QWORD *)(v26 - 48);
          v29 = v120;
          if ( v121 != v120 )
            goto LABEL_31;
          v30 = &v120[HIDWORD(v122)];
          if ( v120 != v30 )
          {
            v31 = 0;
            while ( v28 != *v29 )
            {
              if ( *v29 == -2 )
                v31 = v29;
              if ( v30 == ++v29 )
              {
                if ( !v31 )
                  goto LABEL_108;
                *v31 = v28;
                --v123;
                v26 = *(_QWORD *)(v26 + 8);
                ++v119;
                if ( v27 != (__int64 *)v26 )
                  goto LABEL_33;
                goto LABEL_44;
              }
            }
            goto LABEL_32;
          }
LABEL_108:
          if ( HIDWORD(v122) < (unsigned int)v122 )
          {
            ++HIDWORD(v122);
            *v30 = v28;
            ++v119;
          }
          else
          {
LABEL_31:
            sub_16CCBA0((__int64)&v119, v28);
          }
        }
LABEL_32:
        v26 = *(_QWORD *)(v26 + 8);
      }
    }
LABEL_44:
    if ( v133 == ++i )
      break;
  }
  v132 = v11;
  v32 = (__int64 **)&v131;
  v131 = v108;
  for ( j = (__int64 *)v108; ; j = *v32 )
  {
    if ( j )
    {
      v34 = j[6];
      v35 = j + 5;
      while ( v35 != (__int64 *)v34 )
      {
LABEL_51:
        if ( !v34 )
          BUG();
        if ( *(_BYTE *)(v34 - 8) == 55 )
        {
          v36 = *(_QWORD *)(v34 - 48);
          v37 = v126;
          if ( v127 != v126 )
            goto LABEL_49;
          v38 = &v126[HIDWORD(v128)];
          if ( v126 != v38 )
          {
            v39 = 0;
            while ( v36 != *v37 )
            {
              if ( *v37 == -2 )
                v39 = v37;
              if ( v38 == ++v37 )
              {
                if ( !v39 )
                  goto LABEL_106;
                *v39 = v36;
                --v129;
                v34 = *(_QWORD *)(v34 + 8);
                ++v125;
                if ( v35 != (__int64 *)v34 )
                  goto LABEL_51;
                goto LABEL_62;
              }
            }
            goto LABEL_50;
          }
LABEL_106:
          if ( HIDWORD(v128) < (unsigned int)v128 )
          {
            ++HIDWORD(v128);
            *v38 = v36;
            ++v125;
          }
          else
          {
LABEL_49:
            sub_16CCBA0((__int64)&v125, v36);
          }
        }
LABEL_50:
        v34 = *(_QWORD *)(v34 + 8);
      }
    }
LABEL_62:
    if ( ++v32 == v133 )
      break;
  }
  sub_1B48980((__int64)&v119, (__int64)&v125);
  v40 = v121;
  if ( v121 == v120 )
    v41 = &v121[HIDWORD(v122)];
  else
    v41 = &v121[(unsigned int)v122];
  if ( v121 == v41 )
    goto LABEL_68;
  while ( 1 )
  {
    v42 = *v40;
    v43 = v40;
    if ( (unsigned __int64)*v40 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v41 == ++v40 )
      goto LABEL_68;
  }
  if ( v40 == v41 )
  {
LABEL_68:
    v112 = 0;
    goto LABEL_69;
  }
  v112 = 0;
  v107 = v11;
  v45 = v41;
  v46 = v109;
  do
  {
    if ( !byte_4FB7220
      && (v46 && !(unsigned __int8)sub_1B4BA30(v46)
       || v116 && !(unsigned __int8)sub_1B4BA30(v116)
       || v108 && !(unsigned __int8)sub_1B4BA30(v108)
       || v107 && !(unsigned __int8)sub_1B4BA30(v107)) )
    {
      goto LABEL_100;
    }
    v110 = sub_1B42310(v46, v116);
    v47 = sub_1B42310(v108, v107);
    v48 = v47;
    if ( !v49 )
      goto LABEL_100;
    if ( !v47 )
      goto LABEL_100;
    v50 = *(unsigned __int16 *)(v47 + 18);
    if ( ((v50 >> 7) & 6) != 0 )
      goto LABEL_100;
    if ( (v50 & 1) != 0 )
      goto LABEL_100;
    v51 = *(_WORD *)(v110 + 18);
    if ( ((v51 >> 7) & 6) != 0 || (v51 & 1) != 0 )
      goto LABEL_100;
    v52 = sub_157F0B0(v107);
    v98 = v52 + 40;
    if ( *(_QWORD *)(v52 + 48) != v52 + 40 )
    {
      v89 = v48;
      v53 = *(_QWORD *)(v52 + 48);
      v92 = v42;
      do
      {
        v54 = 0;
        if ( v53 )
          v54 = v53 - 24;
        if ( (unsigned __int8)sub_15F2ED0(v54) || (unsigned __int8)sub_15F3040(v54) )
          goto LABEL_100;
        v53 = *(_QWORD *)(v53 + 8);
      }
      while ( v98 != v53 );
      v42 = v92;
      v48 = v89;
    }
    if ( *(_QWORD *)(v107 + 48) != v107 + 40 )
    {
      v99 = v43;
      v56 = *(_QWORD *)(v107 + 48);
      v93 = v45;
      while ( v56 )
      {
        v57 = v56 - 24;
        if ( v48 != v56 - 24 )
          goto LABEL_117;
LABEL_114:
        v56 = *(_QWORD *)(v56 + 8);
        if ( v107 + 40 == v56 )
        {
          v43 = v99;
          v45 = v93;
          goto LABEL_123;
        }
      }
      v57 = 0;
LABEL_117:
      if ( (unsigned __int8)sub_15F2ED0(v57) || (unsigned __int8)sub_15F3040(v57) )
      {
LABEL_118:
        v43 = v99;
        v45 = v93;
        goto LABEL_100;
      }
      goto LABEL_114;
    }
LABEL_123:
    if ( v108 && *(_QWORD *)(v108 + 48) != v108 + 40 )
    {
      v99 = v43;
      v58 = *(_QWORD *)(v108 + 48);
      v93 = v45;
      while ( v58 )
      {
        v59 = v58 - 24;
        if ( v48 != v58 - 24 )
          goto LABEL_129;
LABEL_126:
        v58 = *(_QWORD *)(v58 + 8);
        if ( v108 + 40 == v58 )
        {
          v43 = v99;
          v45 = v93;
          goto LABEL_134;
        }
      }
      v59 = 0;
LABEL_129:
      if ( (unsigned __int8)sub_15F2ED0(v59) || (unsigned __int8)sub_15F3040(v59) )
        goto LABEL_118;
      goto LABEL_126;
    }
LABEL_134:
    if ( v110 + 24 == *(_QWORD *)(v110 + 40) + 40LL )
      goto LABEL_144;
    v99 = v43;
    v60 = v110 + 24;
    v90 = v42;
    v61 = *(_QWORD *)(v110 + 40) + 40LL;
    v93 = v45;
    do
    {
      if ( v60 )
      {
        v62 = v60 - 24;
        if ( v110 == v60 - 24 )
          goto LABEL_136;
      }
      else
      {
        v62 = 0;
      }
      if ( (unsigned __int8)sub_15F2ED0(v62) || (unsigned __int8)sub_15F3040(v62) )
        goto LABEL_118;
LABEL_136:
      v60 = *(_QWORD *)(v60 + 8);
    }
    while ( v61 != v60 );
    v43 = v99;
    v45 = v93;
    v42 = v90;
LABEL_144:
    v131 = *(_QWORD *)(v115 + 8);
    sub_15CDD40(&v131);
    v131 = *(_QWORD *)(v131 + 8);
    sub_15CDD40(&v131);
    v131 = *(_QWORD *)(v131 + 8);
    sub_15CDD40(&v131);
    v65 = v115;
    if ( !v131 )
      goto LABEL_148;
    v66 = v108;
    if ( !v108 )
      v66 = sub_157F0B0(v107);
    v132 = v66;
    v131 = v107;
    v65 = sub_1AAB350(v115, &v131, 2, "condstore.split", 0, 0, a4, a5, a6, a7, v63, v64, a10, a11, 0);
    if ( v65 )
    {
LABEL_148:
      v113 = v65;
      v67 = sub_157F0B0(v116);
      v91 = *(_QWORD *)(sub_157EBA0(v67) - 72);
      v68 = sub_157F0B0(v107);
      v94 = *(_QWORD *)(sub_157EBA0(v68) - 72);
      v69 = sub_1B48D60(*(_QWORD *)(v110 - 48), *(_QWORD *)(v110 + 40), 0);
      v100 = sub_1B48D60(*(_QWORD *)(v48 - 48), *(_QWORD *)(v48 + 40), v69);
      v70 = sub_157EE30(v113);
      if ( v70 )
        v70 -= 24;
      sub_17CE510((__int64)&v131, v70, 0, 0, 0);
      v71 = v94;
      v72 = v91;
      if ( *(_QWORD *)(v110 + 40) != v46 )
      {
        LOWORD(v118) = 257;
        v73 = sub_156D290(&v131, v91, (__int64)v117);
        v71 = v94;
        v72 = v73;
      }
      if ( v108 != *(_QWORD *)(v48 + 40) )
      {
        v95 = v72;
        LOWORD(v118) = 257;
        v74 = sub_156D290(&v131, v71, (__int64)v117);
        v72 = v95;
        v71 = v74;
      }
      if ( v105 )
      {
        v97 = v71;
        LOWORD(v118) = 257;
        v88 = sub_156D290(&v131, v72, (__int64)v117);
        v71 = v97;
        v72 = v88;
      }
      if ( v104 )
      {
        v96 = v72;
        LOWORD(v118) = 257;
        v87 = sub_156D290(&v131, v71, (__int64)v117);
        v72 = v96;
        v71 = v87;
      }
      LOWORD(v118) = 257;
      v75 = sub_156D390(&v131, v72, v71, (__int64)v117);
      v78 = v133[0];
      if ( v133[0] )
        v78 = v133[0] - 3;
      v79 = sub_1AA92B0(v75, (__int64)v78, 0, 0, 0, 0, a4, a5, a6, a7, v76, v77, a10, a11);
      sub_17050D0(&v131, (__int64)v79);
      LOWORD(v118) = 257;
      v80 = sub_1648A60(64, 2u);
      v81 = (__int64)v80;
      if ( v80 )
      {
        v82 = v100;
        v101 = v80;
        sub_15F9650((__int64)v80, v82, v42, 0, 0);
        v81 = (__int64)v101;
      }
      v114 = v81;
      sub_1B43510(v81, v117, v132, v133[0]);
      sub_12A86E0(&v131, v114);
      v117[0] = 0;
      v117[1] = 0;
      v118 = 0;
      sub_14A8180(v110, v117, 0);
      sub_14A8180(v110, v117, 1);
      sub_1626170(v114, v117);
      v102 = 1 << (*(unsigned __int16 *)(v110 + 18) >> 1) >> 1;
      v83 = 1 << (*(unsigned __int16 *)(v48 + 18) >> 1) >> 1;
      v84 = sub_15A9FE0(a3, **(_QWORD **)(v114 - 48));
      v85 = v102;
      v86 = v84;
      if ( v102 > v83 )
      {
        v85 = v83;
        v83 = v102;
      }
      if ( v85 )
      {
        sub_15F9450(v114, v85);
      }
      else if ( v83 )
      {
        if ( v84 > v83 )
          v86 = v83;
        sub_15F9450(v114, v86);
      }
      else
      {
        sub_15F9450(v114, v84);
      }
      sub_15F20C0((_QWORD *)v48);
      sub_15F20C0((_QWORD *)v110);
      sub_17CD270(&v131);
      v112 = v103;
    }
LABEL_100:
    v55 = v43 + 1;
    if ( v43 + 1 == v45 )
      break;
    while ( 1 )
    {
      v42 = *v55;
      v43 = v55;
      if ( (unsigned __int64)*v55 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v45 == ++v55 )
        goto LABEL_69;
    }
  }
  while ( v45 != v55 );
LABEL_69:
  if ( v127 != v126 )
    _libc_free((unsigned __int64)v127);
  if ( v121 != v120 )
    _libc_free((unsigned __int64)v121);
  return v112;
}
