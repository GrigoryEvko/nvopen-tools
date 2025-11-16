// Function: sub_329BF20
// Address: 0x329bf20
//
__int64 __fastcall sub_329BF20(__int64 *a1, __int64 a2)
{
  unsigned int v4; // r12d
  __int64 v5; // r15
  unsigned __int16 *v6; // rax
  unsigned __int16 v7; // cx
  __int64 v8; // r13
  __int64 (*v9)(); // rax
  __int64 v10; // rdi
  __int64 v11; // r11
  __int64 v12; // rsi
  unsigned int v13; // r15d
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  int v16; // ecx
  __int64 v17; // rax
  __int64 result; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  int v21; // ecx
  __int64 v22; // rdx
  __int64 v23; // rsi
  char (__fastcall *v24)(__int64, unsigned int); // rax
  int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // r15
  char v29; // al
  __int64 v30; // r11
  int v31; // ecx
  char v32; // al
  __int64 v33; // rax
  __int64 v34; // r15
  char v35; // al
  __int64 v36; // r11
  int v37; // ecx
  __int64 *v38; // rax
  __int64 v39; // r15
  char v40; // al
  char v41; // r8
  __int64 v42; // rsi
  __int64 v43; // r12
  int v44; // ecx
  unsigned __int64 v45; // r8
  __int64 v46; // rax
  __int64 v47; // r9
  __int16 v48; // dx
  __int64 v49; // rax
  int v50; // esi
  __int64 v51; // rdi
  char v52; // al
  char v53; // al
  char v54; // al
  char v55; // al
  unsigned __int64 *v56; // rax
  unsigned int v57; // edx
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rax
  unsigned __int16 *v60; // rdx
  int v61; // eax
  __int64 v62; // rdx
  unsigned __int64 v63; // rax
  __int64 v64; // rdi
  int v65; // ecx
  unsigned __int64 v66; // r15
  unsigned int v67; // edx
  __int64 v68; // rdi
  unsigned __int64 *v69; // rax
  __int64 v70; // rax
  unsigned int v71; // edx
  bool v72; // al
  int v73; // eax
  bool v74; // al
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // rdx
  int v78; // esi
  unsigned __int64 v79; // r15
  unsigned int v80; // edx
  char v81; // al
  char v82; // al
  unsigned __int64 v83; // rdx
  bool v84; // zf
  int v85; // eax
  __int64 v86; // rax
  char v87; // al
  bool v88; // zf
  int v89; // eax
  __int64 v90; // rax
  char v91; // al
  __int128 v92; // [rsp-20h] [rbp-150h]
  __int128 v93; // [rsp-10h] [rbp-140h]
  int v94; // [rsp+Ch] [rbp-124h]
  __int64 v95; // [rsp+10h] [rbp-120h]
  __int64 v96; // [rsp+20h] [rbp-110h]
  char v97; // [rsp+20h] [rbp-110h]
  __int64 v98; // [rsp+20h] [rbp-110h]
  __int64 v99; // [rsp+28h] [rbp-108h]
  __int64 v100; // [rsp+30h] [rbp-100h]
  unsigned __int64 v101; // [rsp+38h] [rbp-F8h]
  unsigned int v102; // [rsp+40h] [rbp-F0h]
  unsigned int v103; // [rsp+44h] [rbp-ECh]
  __int64 v104; // [rsp+48h] [rbp-E8h]
  __int64 v105; // [rsp+48h] [rbp-E8h]
  int v106; // [rsp+50h] [rbp-E0h]
  int v107; // [rsp+50h] [rbp-E0h]
  int v108; // [rsp+50h] [rbp-E0h]
  int v109; // [rsp+50h] [rbp-E0h]
  int v110; // [rsp+50h] [rbp-E0h]
  __int64 v111; // [rsp+58h] [rbp-D8h]
  __int64 v112; // [rsp+58h] [rbp-D8h]
  int v113; // [rsp+60h] [rbp-D0h]
  unsigned int v114; // [rsp+60h] [rbp-D0h]
  int v115; // [rsp+60h] [rbp-D0h]
  __int64 v116; // [rsp+60h] [rbp-D0h]
  __int64 v117; // [rsp+60h] [rbp-D0h]
  __int64 v118; // [rsp+68h] [rbp-C8h]
  __int64 v119; // [rsp+68h] [rbp-C8h]
  unsigned __int64 v120; // [rsp+68h] [rbp-C8h]
  __int64 v121; // [rsp+68h] [rbp-C8h]
  __int64 v122; // [rsp+68h] [rbp-C8h]
  __int64 v123; // [rsp+68h] [rbp-C8h]
  unsigned __int64 v124; // [rsp+68h] [rbp-C8h]
  int v125; // [rsp+68h] [rbp-C8h]
  __int64 v126; // [rsp+68h] [rbp-C8h]
  int v127; // [rsp+70h] [rbp-C0h]
  int v128; // [rsp+70h] [rbp-C0h]
  unsigned __int64 v129; // [rsp+70h] [rbp-C0h]
  __int64 v130; // [rsp+70h] [rbp-C0h]
  __int64 v131; // [rsp+70h] [rbp-C0h]
  unsigned __int16 v132; // [rsp+80h] [rbp-B0h]
  __m128i v133; // [rsp+80h] [rbp-B0h]
  __int64 v134; // [rsp+80h] [rbp-B0h]
  __int64 v135; // [rsp+90h] [rbp-A0h] BYREF
  unsigned int v136; // [rsp+98h] [rbp-98h]
  __int64 v137; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v138; // [rsp+A8h] [rbp-88h]
  unsigned __int64 v139; // [rsp+B0h] [rbp-80h]
  __int64 v140; // [rsp+B8h] [rbp-78h]
  unsigned __int64 v141; // [rsp+C0h] [rbp-70h] BYREF
  unsigned __int64 v142; // [rsp+C8h] [rbp-68h]
  __int64 v143; // [rsp+D0h] [rbp-60h]
  __int64 v144; // [rsp+D8h] [rbp-58h]
  unsigned __int64 v145; // [rsp+E0h] [rbp-50h] BYREF
  __int64 v146; // [rsp+E8h] [rbp-48h]
  unsigned __int64 v147; // [rsp+F0h] [rbp-40h]
  unsigned __int64 v148; // [rsp+F8h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 24);
  v5 = *(_QWORD *)(*a1 + 16);
  v6 = *(unsigned __int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v132 = *v6;
  v9 = *(__int64 (**)())(*(_QWORD *)v5 + 1664LL);
  if ( v9 == sub_2FE3590 )
    goto LABEL_2;
  v127 = v7;
  if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, __int64))v9)(v5, v4, v7, v8) )
    goto LABEL_2;
  result = sub_3275DB0(a2, *a1, 0);
  if ( result )
    return result;
  v23 = *(unsigned int *)(a2 + 24);
  v24 = *(char (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v5 + 1360LL);
  if ( v24 == sub_2FE3400 )
  {
    if ( (int)v23 <= 98 )
    {
      if ( (int)v23 > 55 )
      {
        switch ( (int)v23 )
        {
          case '8':
          case ':':
          case '?':
          case '@':
          case 'D':
          case 'F':
          case 'L':
          case 'M':
          case 'R':
          case 'S':
          case '`':
          case 'b':
            goto LABEL_62;
          default:
            goto LABEL_2;
        }
      }
      goto LABEL_2;
    }
    if ( (int)v23 > 188 )
    {
      if ( (unsigned int)(v23 - 279) > 7 )
        goto LABEL_2;
    }
    else if ( (int)v23 <= 185 && (unsigned int)(v23 - 172) > 0xB )
    {
      goto LABEL_2;
    }
  }
  else if ( !v24(v5, v23) )
  {
    goto LABEL_2;
  }
LABEL_62:
  result = sub_3275DB0(a2, *a1, 1);
  if ( result )
    return result;
LABEL_2:
  v10 = *(_QWORD *)(a2 + 40);
  v11 = *(_QWORD *)v10;
  if ( *(_DWORD *)(*(_QWORD *)v10 + 24LL) == 205 )
  {
    v19 = *(_QWORD *)(v11 + 56);
    if ( v19 )
    {
      v13 = *(_DWORD *)(v10 + 8);
      v20 = *(_QWORD *)(v11 + 56);
      v21 = 1;
      do
      {
        while ( v13 != *(_DWORD *)(v20 + 8) )
        {
          v20 = *(_QWORD *)(v20 + 32);
          if ( !v20 )
            goto LABEL_24;
        }
        if ( !v21 )
          goto LABEL_3;
        v22 = *(_QWORD *)(v20 + 32);
        if ( !v22 )
        {
          v21 = 0;
          goto LABEL_34;
        }
        if ( v13 == *(_DWORD *)(v22 + 8) )
          goto LABEL_3;
        v20 = *(_QWORD *)(v22 + 32);
        v21 = 0;
      }
      while ( v20 );
LABEL_24:
      if ( v21 != 1 )
        goto LABEL_34;
    }
  }
LABEL_3:
  v12 = *(_QWORD *)(v10 + 40);
  v13 = *(_DWORD *)(v10 + 48);
  v14 = v13 | *(_QWORD *)(v10 + 8) & 0xFFFFFFFF00000000LL;
  v11 = v12;
  if ( v4 - 190 <= 2 )
  {
    v15 = *(_QWORD *)(v12 + 56);
    if ( v15 )
    {
      v16 = 1;
      do
      {
        while ( v13 != *(_DWORD *)(v15 + 8) )
        {
          v15 = *(_QWORD *)(v15 + 32);
          if ( !v15 )
            goto LABEL_12;
        }
        if ( !v16 )
          goto LABEL_13;
        v17 = *(_QWORD *)(v15 + 32);
        if ( !v17 )
          goto LABEL_66;
        if ( *(_DWORD *)(v17 + 8) == v13 )
          goto LABEL_13;
        v15 = *(_QWORD *)(v17 + 32);
        v16 = 0;
      }
      while ( v15 );
LABEL_12:
      if ( v16 == 1 )
        goto LABEL_13;
LABEL_66:
      v51 = *a1;
      v135 = 0;
      v136 = 0;
      LODWORD(v146) = 1;
      v145 = 0;
      LODWORD(v148) = 1;
      v147 = 0;
      v52 = sub_329BA40(v51, v12, v14, &v135, (__int64)&v145);
      v11 = v12;
      if ( !v52 )
        goto LABEL_67;
      v57 = v146;
      if ( (unsigned int)v146 > 0x40 )
      {
        v115 = v146;
        v73 = sub_C44500((__int64)&v145);
        v11 = v12;
        v57 = v115 - v73;
      }
      else if ( (_DWORD)v146 )
      {
        if ( v145 << (64 - (unsigned __int8)v146) == -1 )
        {
          v57 = v146 - 64;
        }
        else
        {
          _BitScanReverse64(&v58, ~(v145 << (64 - (unsigned __int8)v146)));
          v57 = v146 - (v58 ^ 0x3F);
        }
      }
      v59 = v57;
      v60 = (unsigned __int16 *)(*(_QWORD *)(v12 + 48) + 16LL * v13);
      v124 = v59;
      v61 = *v60;
      v62 = *((_QWORD *)v60 + 1);
      LOWORD(v137) = v61;
      v138 = v62;
      if ( (_WORD)v61 )
      {
        if ( (unsigned __int16)(v61 - 17) > 0xD3u )
        {
          LOWORD(v141) = v61;
          v142 = v62;
          goto LABEL_86;
        }
        LOWORD(v61) = word_4456580[v61 - 1];
        v83 = 0;
      }
      else
      {
        v112 = v62;
        v116 = v11;
        v74 = sub_30070B0((__int64)&v137);
        v11 = v116;
        if ( !v74 )
        {
          v142 = v112;
          LOWORD(v141) = 0;
          goto LABEL_102;
        }
        LOWORD(v61) = sub_3009970((__int64)&v137, v12, v112, v75, v76);
        v11 = v116;
      }
      LOWORD(v141) = v61;
      v142 = v83;
      if ( (_WORD)v61 )
      {
LABEL_86:
        if ( (_WORD)v61 == 1 || (unsigned __int16)(v61 - 504) <= 7u )
          BUG();
        v63 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v61 - 16];
        goto LABEL_89;
      }
LABEL_102:
      v117 = v11;
      v63 = sub_3007260((__int64)&v141);
      v11 = v117;
      v139 = v63;
      v140 = v77;
LABEL_89:
      if ( v124 < v63 )
      {
        v11 = v135;
        v13 = v136;
      }
LABEL_67:
      if ( (unsigned int)v148 > 0x40 && v147 )
      {
        v121 = v11;
        j_j___libc_free_0_0(v147);
        v11 = v121;
      }
      if ( (unsigned int)v146 > 0x40 && v145 )
      {
        v122 = v11;
        j_j___libc_free_0_0(v145);
        v11 = v122;
      }
    }
  }
LABEL_13:
  if ( *(_DWORD *)(v11 + 24) != 205 )
    return 0;
  v19 = *(_QWORD *)(v11 + 56);
  if ( !v19 )
    return 0;
  v21 = 1;
LABEL_34:
  v25 = 1;
  do
  {
    while ( v13 != *(_DWORD *)(v19 + 8) )
    {
      v19 = *(_QWORD *)(v19 + 32);
      if ( !v19 )
        goto LABEL_41;
    }
    if ( !v25 )
      return 0;
    v26 = *(_QWORD *)(v19 + 32);
    if ( !v26 )
      goto LABEL_42;
    if ( *(_DWORD *)(v26 + 8) == v13 )
      return 0;
    v19 = *(_QWORD *)(v26 + 32);
    v25 = 0;
  }
  while ( v19 );
LABEL_41:
  if ( v25 == 1 )
    return 0;
LABEL_42:
  v27 = *(_QWORD *)(v11 + 40);
  v106 = v21;
  v118 = v11;
  v28 = *(_QWORD *)(v27 + 48);
  v104 = *(_QWORD *)(v27 + 40);
  v101 = v28;
  v113 = *(_DWORD *)(v27 + 48);
  v29 = sub_326A930(v104, v28, 1u);
  v30 = v118;
  v31 = v106;
  if ( !v29 )
  {
    v32 = sub_33E2470(*a1, v104, v28);
    v30 = v118;
    v31 = v106;
    if ( !v32 )
      return 0;
  }
  v33 = *(_QWORD *)(v30 + 40);
  v107 = v31;
  v119 = v30;
  v34 = *(_QWORD *)(v33 + 88);
  v100 = *(_QWORD *)(v33 + 80);
  v99 = v34;
  v102 = *(_DWORD *)(v33 + 88);
  v35 = sub_326A930(v100, v34, 1u);
  v36 = v119;
  v37 = v107;
  if ( !v35 )
  {
    v53 = sub_33E2470(*a1, v100, v34);
    v36 = v119;
    v37 = v107;
    if ( !v53 )
      return 0;
  }
  if ( v4 - 186 > 1 )
    goto LABEL_46;
  v109 = v37;
  v123 = v36;
  v54 = sub_33E0720(v104, v101, 0);
  v36 = v123;
  v37 = v109;
  if ( v54 )
  {
    v55 = sub_33E07E0(v100, v34, 0);
    v36 = v123;
    v37 = v109;
    if ( v55 )
      goto LABEL_78;
  }
  v110 = v37;
  v126 = v36;
  v81 = sub_33E0720(v100, v34, 0);
  v36 = v126;
  v37 = v110;
  if ( !v81 )
    goto LABEL_46;
  v82 = sub_33E07E0(v104, v101, 0);
  v36 = v126;
  v37 = v110;
  if ( v82 )
  {
LABEL_78:
    v41 = 1;
    v56 = (unsigned __int64 *)(*(_QWORD *)(a2 + 40) + 40LL * (v37 ^ 1u));
    v39 = v56[1];
    v120 = *v56;
    v103 = *((_DWORD *)v56 + 2);
  }
  else
  {
LABEL_46:
    v96 = v36;
    v108 = v37;
    v38 = (__int64 *)(*(_QWORD *)(a2 + 40) + 40LL * (v37 ^ 1u));
    v39 = v38[1];
    v120 = *v38;
    v103 = *((_DWORD *)v38 + 2);
    v40 = sub_326A930(*v38, v39, 1u);
    v37 = v108;
    v36 = v96;
    if ( v40 )
    {
      v41 = 0;
    }
    else
    {
      if ( !(unsigned __int8)sub_33E2470(*a1, v120, v39) )
        return 0;
      v41 = 0;
      v36 = v96;
      v37 = v108;
    }
  }
  v42 = *(_QWORD *)(v36 + 80);
  v137 = v42;
  if ( v42 )
  {
    v94 = v37;
    v95 = v36;
    v97 = v41;
    sub_B96E90((__int64)&v137, v42, 1);
    v37 = v94;
    v36 = v95;
    v41 = v97;
  }
  LODWORD(v138) = *(_DWORD *)(v36 + 72);
  if ( v41 )
  {
    if ( v4 == 186 )
    {
      v130 = v36;
      v84 = (unsigned __int8)sub_33E0720(v104, v101, 0) == 0;
      v85 = v113;
      if ( v84 )
        v85 = v103;
      v114 = v85;
      v86 = v104;
      if ( v84 )
        v86 = v120;
      v111 = v86;
      v87 = sub_33E0720(v100, v99, 0);
      v36 = v130;
      if ( !v87 )
        goto LABEL_54;
    }
    else
    {
      if ( v4 != 187 )
      {
        v114 = v103;
        v111 = v120;
LABEL_54:
        v43 = *a1;
        v44 = v132;
        v45 = v120;
        v133 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v36 + 40));
        v46 = *(_QWORD *)(**(_QWORD **)(v36 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v36 + 40) + 8LL);
        v47 = v103;
        v48 = *(_WORD *)v46;
        v49 = *(_QWORD *)(v46 + 8);
        LOWORD(v145) = v48;
        v146 = v49;
        if ( v48 )
        {
          v50 = ((unsigned __int16)(v48 - 17) < 0xD4u) + 205;
        }
        else
        {
          v125 = v44;
          v129 = v45;
          v72 = sub_30070B0((__int64)&v145);
          v44 = v125;
          v45 = v129;
          v47 = v103;
          v50 = 205 - (!v72 - 1);
        }
        *((_QWORD *)&v93 + 1) = v47;
        *(_QWORD *)&v93 = v45;
        *((_QWORD *)&v92 + 1) = v114;
        *(_QWORD *)&v92 = v111;
        result = sub_340EC60(v43, v50, (unsigned int)&v137, v44, v8, 0, v133.m128i_i64[0], v133.m128i_i64[1], v92, v93);
        *(_DWORD *)(result + 28) = *(_DWORD *)(a2 + 28);
        goto LABEL_57;
      }
      v131 = v36;
      v88 = (unsigned __int8)sub_33E07E0(v104, v101, 0) == 0;
      v89 = v113;
      if ( v88 )
        v89 = v103;
      v114 = v89;
      v90 = v104;
      if ( v88 )
        v90 = v120;
      v111 = v90;
      v91 = sub_33E07E0(v100, v99, 0);
      v36 = v131;
      if ( !v91 )
        goto LABEL_54;
    }
    v103 = v102;
    v120 = v100;
    goto LABEL_54;
  }
  v98 = v36;
  v64 = *a1;
  if ( v37 )
  {
    HIWORD(v65) = HIWORD(v127);
    LOWORD(v65) = v132;
    v66 = v103 | v39 & 0xFFFFFFFF00000000LL;
    v128 = v65;
    v145 = v120;
    v147 = v104;
    v146 = v66;
    v148 = v101;
    v111 = sub_3402EA0(v64, v4, (unsigned int)&v137, v65, v8, 0, (__int64)&v145, 2);
    v114 = v67;
    if ( !v111 )
      goto LABEL_95;
    v68 = *a1;
    v105 = v98;
    v142 = v66;
    v141 = v120;
    v143 = v100;
    v144 = v99;
    v69 = &v141;
  }
  else
  {
    HIWORD(v78) = HIWORD(v127);
    LOWORD(v78) = v132;
    v145 = v104;
    v128 = v78;
    v146 = v101;
    v79 = v103 | v39 & 0xFFFFFFFF00000000LL;
    v147 = v120;
    v148 = v79;
    v111 = sub_3402EA0(v64, v4, (unsigned int)&v137, v78, v8, 0, (__int64)&v145, 2);
    v114 = v80;
    if ( !v111 )
      goto LABEL_95;
    v68 = *a1;
    v105 = v98;
    v148 = v79;
    v145 = v100;
    v146 = v99;
    v147 = v120;
    v69 = &v145;
  }
  v70 = sub_3402EA0(v68, v4, (unsigned int)&v137, v128, v8, 0, (__int64)v69, 2);
  v36 = v105;
  v120 = v70;
  v103 = v71;
  if ( v70 )
    goto LABEL_54;
LABEL_95:
  result = 0;
LABEL_57:
  if ( v137 )
  {
    v134 = result;
    sub_B91220((__int64)&v137, v137);
    return v134;
  }
  return result;
}
