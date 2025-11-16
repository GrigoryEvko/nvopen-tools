// Function: sub_2B5F980
// Address: 0x2b5f980
//
__int64 __fastcall sub_2B5F980(__int64 *a1, unsigned __int64 a2, __int64 *a3)
{
  __int64 v3; // rcx
  __int64 *v4; // r12
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 *v7; // rax
  unsigned __int8 v8; // cl
  unsigned __int8 v9; // cl
  unsigned __int8 v10; // cl
  unsigned __int8 v11; // cl
  unsigned __int8 v12; // al
  __int64 *v13; // rsi
  __int64 *v14; // rax
  unsigned int v15; // r8d
  __int64 v16; // r13
  __int64 v17; // rbx
  __int64 i; // rdx
  __int16 v19; // ax
  __int64 v20; // rcx
  __int64 *v21; // r15
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 *v27; // r14
  __int64 v28; // r8
  __int64 *v29; // r15
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  unsigned __int64 *v33; // r14
  unsigned __int64 *v34; // r15
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rdi
  __int64 v38; // r15
  __int64 *v39; // r14
  _BYTE *v40; // r13
  unsigned __int8 v41; // bl
  __int64 v42; // r13
  _BYTE *v43; // r14
  _QWORD *v44; // rbx
  unsigned __int64 v45; // rdi
  unsigned __int64 v46; // rdi
  unsigned __int64 v47; // rdi
  int v49; // r12d
  unsigned int v50; // ebx
  int v51; // r12d
  __int64 v52; // rax
  __int64 v53; // rax
  unsigned int v54; // r14d
  int v55; // eax
  int *v56; // r8
  int v57; // eax
  unsigned int v58; // edi
  int *v59; // r9
  int v60; // r10d
  int v61; // eax
  unsigned int v62; // edi
  int *v63; // r9
  int v64; // r10d
  unsigned __int8 v65; // al
  unsigned __int8 v66; // al
  __int64 v67; // rdx
  __int64 v68; // rax
  int v69; // eax
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // r9
  _BYTE *v75; // r13
  char *v76; // rbx
  size_t v77; // rdx
  int v78; // r9d
  int v79; // esi
  int v80; // ebx
  __int64 v81; // rax
  _QWORD *v82; // rbx
  __int64 v83; // rax
  __int64 v84; // rdx
  _QWORD *v85; // r12
  _QWORD *v86; // rax
  int v87; // r9d
  int v88; // esi
  size_t v89; // rdx
  _BYTE *v90; // [rsp+10h] [rbp-10A0h]
  unsigned int v91; // [rsp+18h] [rbp-1098h]
  __int64 v92; // [rsp+18h] [rbp-1098h]
  unsigned int v93; // [rsp+20h] [rbp-1090h]
  int v94; // [rsp+20h] [rbp-1090h]
  int v95; // [rsp+24h] [rbp-108Ch]
  int v96; // [rsp+30h] [rbp-1080h]
  int v97; // [rsp+38h] [rbp-1078h]
  unsigned int v98; // [rsp+3Ch] [rbp-1074h]
  __int64 *v99; // [rsp+40h] [rbp-1070h]
  __int64 *v100; // [rsp+48h] [rbp-1068h]
  __int64 v101; // [rsp+48h] [rbp-1068h]
  __int64 v102; // [rsp+48h] [rbp-1068h]
  __int64 v103; // [rsp+48h] [rbp-1068h]
  __int64 v104; // [rsp+50h] [rbp-1060h]
  unsigned int v105; // [rsp+58h] [rbp-1058h]
  bool v106; // [rsp+5Dh] [rbp-1053h]
  char v107; // [rsp+5Eh] [rbp-1052h]
  unsigned __int8 v108; // [rsp+5Fh] [rbp-1051h]
  __int64 *v109; // [rsp+60h] [rbp-1050h]
  unsigned __int64 v110; // [rsp+68h] [rbp-1048h]
  _BYTE *v111; // [rsp+70h] [rbp-1040h] BYREF
  __int64 v112; // [rsp+78h] [rbp-1038h]
  _BYTE v113[224]; // [rsp+80h] [rbp-1030h] BYREF
  char *v114; // [rsp+160h] [rbp-F50h] BYREF
  __int64 v115; // [rsp+168h] [rbp-F48h]
  char v116; // [rsp+170h] [rbp-F40h] BYREF
  __int64 *v117; // [rsp+250h] [rbp-E60h] BYREF
  __int64 v118; // [rsp+258h] [rbp-E58h]
  __int64 v119; // [rsp+260h] [rbp-E50h] BYREF
  __int64 v120; // [rsp+268h] [rbp-E48h]
  _BYTE *v121; // [rsp+270h] [rbp-E40h]
  __int64 v122; // [rsp+278h] [rbp-E38h]
  _BYTE v123[1760]; // [rsp+280h] [rbp-E30h] BYREF
  __int64 v124; // [rsp+960h] [rbp-750h] BYREF
  _BYTE *v125; // [rsp+968h] [rbp-748h]
  unsigned __int64 *v126; // [rsp+970h] [rbp-740h] BYREF
  __int64 v127; // [rsp+978h] [rbp-738h]
  _BYTE *v128; // [rsp+980h] [rbp-730h] BYREF
  __int64 v129; // [rsp+988h] [rbp-728h]
  _BYTE v130[1824]; // [rsp+990h] [rbp-720h] BYREF

  v3 = (__int64)(8 * a2) >> 3;
  v4 = &a1[a2];
  v5 = (__int64)(8 * a2) >> 5;
  v110 = a2;
  v109 = a3;
  if ( v5 <= 0 )
  {
    v6 = a1;
LABEL_13:
    if ( v3 != 2 )
    {
      if ( v3 != 3 )
      {
        if ( v3 != 1 )
          goto LABEL_18;
        goto LABEL_16;
      }
      v65 = *(_BYTE *)*v6;
      if ( v65 != 13 && v65 <= 0x1Cu )
        goto LABEL_79;
      ++v6;
    }
    v66 = *(_BYTE *)*v6;
    if ( v66 <= 0x1Cu && v66 != 13 )
      goto LABEL_79;
    ++v6;
LABEL_16:
    v12 = *(_BYTE *)*v6;
    if ( v12 > 0x1Cu || v12 == 13 )
      goto LABEL_18;
    goto LABEL_79;
  }
  v6 = a1;
  v7 = &a1[4 * v5];
  while ( 1 )
  {
    v8 = *(_BYTE *)*v6;
    if ( v8 != 13 && v8 <= 0x1Cu )
      goto LABEL_79;
    v9 = *(_BYTE *)v6[1];
    if ( v9 <= 0x1Cu && v9 != 13 )
    {
      ++v6;
      goto LABEL_79;
    }
    v10 = *(_BYTE *)v6[2];
    if ( v10 <= 0x1Cu && v10 != 13 )
    {
      v6 += 2;
      goto LABEL_79;
    }
    v11 = *(_BYTE *)v6[3];
    if ( v11 != 13 && v11 <= 0x1Cu )
      break;
    v6 += 4;
    if ( v7 == v6 )
    {
      v3 = v4 - v6;
      goto LABEL_13;
    }
  }
  v6 += 3;
LABEL_79:
  if ( v4 != v6 )
    return 0;
LABEL_18:
  v13 = &a1[a2];
  v14 = (__int64 *)sub_2B0CB90((_BYTE **)a1, (__int64)v13);
  if ( v4 == v14 )
    return 0;
  v16 = *v14;
  v100 = v14;
  v17 = 0;
  for ( i = *v14; ; i = *v14 )
  {
    v17 -= (*(_BYTE *)i < 0x1Du) - 1LL;
    if ( ++v14 == v4 )
      break;
  }
  if ( v110 > 2 )
  {
    v108 = *(_BYTE *)v16;
    if ( v108 == 84 )
    {
      v95 = 84;
      v98 = 17;
      v105 = 42;
      goto LABEL_121;
    }
    if ( (unsigned int)v17 < v110 >> 1 )
      return 0;
  }
  else
  {
    if ( v110 == 2 && (unsigned int)v17 <= 1 )
      return 0;
    v108 = *(_BYTE *)v16;
  }
  i = (unsigned int)v108 - 67;
  v95 = v108;
  v105 = v108 - 42;
  v98 = v108 - 67;
  if ( (unsigned __int8)(v108 - 82) > 1u )
  {
LABEL_121:
    v107 = 0;
    v97 = 42;
    v106 = 0;
    v96 = v95 - 29;
    goto LABEL_37;
  }
  v19 = *(_WORD *)(v16 + 2);
  v117 = 0;
  v121 = v123;
  v128 = v130;
  v97 = v19 & 0x3F;
  LODWORD(v114) = v97;
  v118 = 0;
  v119 = 0;
  v120 = 0;
  v122 = 0;
  v124 = 0;
  v125 = 0;
  v126 = 0;
  v127 = 0;
  v129 = 0;
  sub_2B5D710((__int64)&v117, (int *)&v114);
  LODWORD(v114) = v97;
  sub_2B5D710((__int64)&v124, (int *)&v114);
  if ( a1 == v4 )
    goto LABEL_132;
  v20 = v17;
  v21 = a1;
  do
  {
    if ( (unsigned __int8)(*(_BYTE *)*v21 - 82) > 1u )
    {
      LODWORD(v17) = v20;
      goto LABEL_31;
    }
    v92 = v20;
    v54 = *(_WORD *)(*v21 + 2) & 0x3F;
    v55 = sub_B52F50(v54);
    LODWORD(v114) = v54;
    v94 = v55;
    sub_2B5D710((__int64)&v124, (int *)&v114);
    v20 = v92;
    v56 = (int *)(v118 + 4LL * (unsigned int)v120);
    if ( (_DWORD)v120 )
    {
      v57 = v120 - 1;
      v58 = (v120 - 1) & (37 * v54);
      v59 = (int *)(v118 + 4LL * v58);
      v60 = *v59;
      if ( v54 == *v59 )
      {
LABEL_129:
        if ( v56 != v59 )
          goto LABEL_130;
      }
      else
      {
        v78 = 1;
        while ( v60 != -1 )
        {
          v79 = v78 + 1;
          v58 = v57 & (v58 + v78);
          v59 = (int *)(v118 + 4LL * v58);
          v60 = *v59;
          if ( v54 == *v59 )
            goto LABEL_129;
          v78 = v79;
        }
      }
      v62 = v57 & (37 * v94);
      v63 = (int *)(v118 + 4LL * v62);
      v64 = *v63;
      if ( v94 == *v63 )
      {
LABEL_142:
        if ( v56 != v63 )
          goto LABEL_130;
      }
      else
      {
        v87 = 1;
        while ( v64 != -1 )
        {
          v88 = v87 + 1;
          v62 = v57 & (v62 + v87);
          v63 = (int *)(v118 + 4LL * v62);
          v64 = *v63;
          if ( v94 == *v63 )
            goto LABEL_142;
          v87 = v88;
        }
      }
    }
    LODWORD(v114) = v54;
    sub_2B5D710((__int64)&v117, (int *)&v114);
    v20 = v92;
LABEL_130:
    ++v21;
  }
  while ( v4 != v21 );
  LODWORD(v17) = v20;
LABEL_132:
  if ( (unsigned int)v129 <= 2 )
LABEL_31:
    v106 = 0;
  else
    v106 = (_DWORD)v122 == 2;
  if ( v128 != v130 )
    _libc_free((unsigned __int64)v128);
  sub_C7D6A0((__int64)v125, 4LL * (unsigned int)v127, 4);
  if ( v121 != v123 )
    _libc_free((unsigned __int64)v121);
  v13 = (__int64 *)(4LL * (unsigned int)v120);
  v96 = v108 - 29;
  sub_C7D6A0(v118, (__int64)v13, 4);
  v107 = 1;
LABEL_37:
  v111 = v113;
  v112 = 0x100000000LL;
  if ( *(_BYTE *)v16 == 85 )
  {
    v91 = sub_9B78C0(v16, v109);
    v22 = sub_B43CA0(v16);
    v125 = (_BYTE *)v16;
    v124 = v22;
    v126 = (unsigned __int64 *)&v128;
    v127 = 0x800000000LL;
    sub_D39570(v16, (unsigned int *)&v126);
    v117 = &v119;
    v118 = 0x800000000LL;
    sub_D39570(v16, (unsigned int *)&v117);
    v13 = (__int64 *)&v117;
    sub_2B467C0((unsigned int *)&v111, (unsigned int *)&v117, v23, v24, v25, v26);
    v27 = v117;
    v28 = 28LL * (unsigned int)v118;
    v29 = &v117[v28];
    if ( v117 != &v117[v28] )
    {
      do
      {
        v29 -= 28;
        v30 = v29[23];
        if ( (__int64 *)v30 != v29 + 25 )
        {
          v13 = (__int64 *)(v29[25] + 1);
          j_j___libc_free_0(v30);
        }
        v31 = v29[19];
        if ( (__int64 *)v31 != v29 + 21 )
        {
          v13 = (__int64 *)(v29[21] + 1);
          j_j___libc_free_0(v31);
        }
        v32 = v29[1];
        if ( (__int64 *)v32 != v29 + 3 )
          _libc_free(v32);
      }
      while ( v27 != v29 );
      v29 = v117;
    }
    if ( v29 != &v119 )
      _libc_free((unsigned __int64)v29);
    v33 = v126;
    v34 = &v126[28 * (unsigned int)v127];
    if ( v126 != v34 )
    {
      do
      {
        v34 -= 28;
        v35 = v34[23];
        if ( (unsigned __int64 *)v35 != v34 + 25 )
        {
          v13 = (__int64 *)(v34[25] + 1);
          j_j___libc_free_0(v35);
        }
        v36 = v34[19];
        if ( (unsigned __int64 *)v36 != v34 + 21 )
        {
          v13 = (__int64 *)(v34[21] + 1);
          j_j___libc_free_0(v36);
        }
        v37 = v34[1];
        if ( (unsigned __int64 *)v37 != v34 + 3 )
          _libc_free(v37);
      }
      while ( v33 != v34 );
      v34 = v126;
    }
    if ( v34 != (unsigned __int64 *)&v128 )
      _libc_free((unsigned __int64)v34);
    if ( !(unsigned __int8)sub_9B7470(v91) )
    {
      i = (unsigned int)v112;
      if ( !(_DWORD)v112 )
      {
        v43 = v111;
        v42 = 0;
        goto LABEL_76;
      }
    }
  }
  else
  {
    v91 = 0;
  }
  v90 = (_BYTE *)v16;
  v38 = v16;
  v39 = v100;
  v104 = (unsigned int)v17;
  v99 = v4;
  v93 = v96;
  while ( 2 )
  {
    v40 = (_BYTE *)*v39;
    v41 = *(_BYTE *)*v39;
    if ( v41 <= 0x1Cu )
      goto LABEL_110;
    if ( v104 != v110 && ((unsigned __int8)(v41 - 48) <= 5u || v41 == 85) )
      goto LABEL_66;
    v49 = v41 - 29;
    if ( v105 <= 0x11 )
    {
      i = (unsigned int)v41 - 42;
      if ( (unsigned int)i <= 0x11 )
      {
        if ( v41 != v108 )
        {
          i = v93;
          if ( v93 != v49 )
          {
            if ( v93 != v96
              || (unsigned int)v41 - 48 <= 1
              || (unsigned __int8)(v41 - 51) <= 1u
              || (unsigned int)(v95 - 48) <= 1
              || v93 - 22 <= 1 )
            {
              goto LABEL_66;
            }
LABEL_118:
            v93 = v41 - 29;
            v90 = v40;
          }
        }
        goto LABEL_110;
      }
    }
    if ( v98 <= 0xC && (unsigned int)v41 - 67 <= 0xC )
    {
      v101 = sub_986520(v38);
      v52 = sub_986520((__int64)v40);
      i = v101;
      v13 = *(__int64 **)v52;
      if ( *(_QWORD *)(*(_QWORD *)v52 + 8LL) != *(_QWORD *)(*(_QWORD *)v101 + 8LL) )
        goto LABEL_66;
      if ( v41 != v108 && v93 != v49 )
      {
        if ( v93 != v96 )
          goto LABEL_66;
        goto LABEL_118;
      }
LABEL_110:
      if ( v99 == ++v39 )
      {
        v42 = v38;
        goto LABEL_67;
      }
      continue;
    }
    break;
  }
  if ( (unsigned __int8)(v41 - 82) <= 1u )
  {
    if ( v107 )
    {
      if ( *(_QWORD *)(*((_QWORD *)v40 - 8) + 8LL) != *(_QWORD *)(*(_QWORD *)(v38 - 64) + 8LL) )
        goto LABEL_66;
      v50 = *((_WORD *)v40 + 1) & 0x3F;
      v51 = sub_B52F50(v50);
      if ( v110 != 2 && !v106 || v50 != v97 && v51 != v97 )
      {
        v13 = (__int64 *)v40;
        if ( !(unsigned __int8)sub_2B64E30(v38, v40, v109) )
        {
          if ( (_BYTE *)v38 == v90 )
          {
            if ( v50 == v97 )
              v40 = (_BYTE *)v38;
            v90 = v40;
          }
          else
          {
            v13 = (__int64 *)v40;
            if ( !(unsigned __int8)sub_2B64E30(v90, v40, v109) )
            {
              v61 = *((_WORD *)v90 + 1) & 0x3F;
              if ( v50 != v97 && v51 != v97 && v50 != v61 && v51 != v61 )
                goto LABEL_66;
            }
          }
        }
      }
    }
    else if ( v41 != v108 )
    {
      goto LABEL_66;
    }
    goto LABEL_110;
  }
  if ( v41 != v108 )
    goto LABEL_66;
  switch ( v108 )
  {
    case '?':
      if ( (*((_DWORD *)v40 + 1) & 0x7FFFFFF) != 2 )
        goto LABEL_66;
      v53 = sub_986520(v38);
      i = *(_QWORD *)v53;
      if ( *(_QWORD *)(*(_QWORD *)v53 + 8LL) != *(_QWORD *)(*((_QWORD *)v40 - 8) + 8LL) )
        goto LABEL_66;
      goto LABEL_110;
    case 'Z':
      if ( !(unsigned __int8)sub_2B15E10(v40, (__int64)v13, i, v110, v15) )
        goto LABEL_66;
      goto LABEL_110;
    case '=':
      if ( sub_B46500(v40) || (v40[2] & 1) != 0 || sub_B46500((unsigned __int8 *)v38) || (*(_BYTE *)(v38 + 2) & 1) != 0 )
        goto LABEL_66;
      goto LABEL_110;
  }
  if ( v108 != 85 )
    goto LABEL_110;
  v67 = *((_QWORD *)v40 - 4);
  v68 = *(_QWORD *)(v38 - 32);
  if ( v67 && !*(_BYTE *)v67 && *(_QWORD *)(v67 + 24) == *((_QWORD *)v40 + 10) )
  {
    if ( !v68 || *(_BYTE *)v68 || *(_QWORD *)(v38 + 80) != *(_QWORD *)(v68 + 24) )
      goto LABEL_66;
LABEL_170:
    if ( v67 != v68 )
      goto LABEL_66;
  }
  else if ( v68 && !*(_BYTE *)v68 )
  {
    if ( *(_QWORD *)(v68 + 24) == *(_QWORD *)(v38 + 80) )
      goto LABEL_66;
    v67 = 0;
    v68 = 0;
    goto LABEL_170;
  }
  if ( (unsigned int)sub_A172A0(*v39) )
  {
    if ( !(unsigned int)sub_A172A0(v38) )
      goto LABEL_66;
    if ( *(char *)(v38 + 7) >= 0 )
      BUG();
    v80 = *(_DWORD *)(v38 + 4);
    v81 = sub_BD2BC0(v38);
    v102 = *((_DWORD *)v40 + 1) & 0x7FFFFFF;
    v82 = (_QWORD *)(v38 + 32 * (*(unsigned int *)(v81 + 8) - (unsigned __int64)(v80 & 0x7FFFFFF)));
    if ( (char)v40[7] >= 0 )
      BUG();
    v83 = sub_BD2BC0((__int64)v40);
    v85 = &v40[32 * (*(unsigned int *)(v83 + v84 - 4) - v102)];
    if ( (char)v40[7] >= 0 )
      BUG();
    v103 = *((_DWORD *)v40 + 1) & 0x7FFFFFF;
    v86 = &v40[32 * (*(unsigned int *)(sub_BD2BC0((__int64)v40) + 8) - v103)];
    if ( v85 != v86 )
    {
      while ( *v86 == *v82 )
      {
        v86 += 4;
        v82 += 4;
        if ( v85 == v86 )
          goto LABEL_172;
      }
LABEL_66:
      v42 = 0;
      goto LABEL_67;
    }
  }
LABEL_172:
  v13 = v109;
  v69 = sub_9B78C0((__int64)v40, v109);
  i = v91;
  if ( v91 != v69 )
    goto LABEL_66;
  if ( v91 )
    goto LABEL_110;
  v70 = sub_B43CA0((__int64)v40);
  v127 = 0x800000000LL;
  v124 = v70;
  v125 = v40;
  v126 = (unsigned __int64 *)&v128;
  sub_D39570((__int64)v40, (unsigned int *)&v126);
  v117 = &v119;
  v118 = 0x800000000LL;
  sub_D39570((__int64)v40, (unsigned int *)&v117);
  v114 = &v116;
  v115 = 0x100000000LL;
  if ( (_DWORD)v118 )
    sub_2B467C0((unsigned int *)&v114, (unsigned int *)&v117, v71, v72, v73, v74);
  sub_2B30DE0((__int64)&v117);
  sub_2B30DE0((__int64)&v126);
  if ( (_DWORD)v112 == (_DWORD)v115 )
  {
    v75 = v111;
    v76 = v114;
    if ( *((_DWORD *)v114 + 54) == *((_DWORD *)v111 + 54) )
    {
      v77 = *((_QWORD *)v114 + 20);
      if ( v77 == *((_QWORD *)v111 + 20)
        && (!v77 || !memcmp(*((const void **)v114 + 19), *((const void **)v111 + 19), v77)) )
      {
        v89 = *((_QWORD *)v76 + 24);
        if ( v89 == *((_QWORD *)v75 + 24)
          && (!v89 || !memcmp(*((const void **)v76 + 23), *((const void **)v75 + 23), v89))
          && *(_DWORD *)v76 == *(_DWORD *)v75
          && v76[4] == v75[4] )
        {
          v13 = (__int64 *)(v75 + 8);
          if ( (unsigned __int8)sub_2B3C690((__int64 *)v76 + 1, (__int64 *)v75 + 1) )
          {
            sub_2B35260((__int64)&v114);
            goto LABEL_110;
          }
        }
      }
    }
  }
  v42 = 0;
  sub_2B35260((__int64)&v114);
LABEL_67:
  v43 = v111;
  v44 = &v111[224 * (unsigned int)v112];
  if ( v111 != (_BYTE *)v44 )
  {
    do
    {
      v44 -= 28;
      v45 = v44[23];
      if ( (_QWORD *)v45 != v44 + 25 )
        j_j___libc_free_0(v45);
      v46 = v44[19];
      if ( (_QWORD *)v46 != v44 + 21 )
        j_j___libc_free_0(v46);
      v47 = v44[1];
      if ( (_QWORD *)v47 != v44 + 3 )
        _libc_free(v47);
    }
    while ( v43 != (_BYTE *)v44 );
    v43 = v111;
  }
LABEL_76:
  if ( v43 != v113 )
    _libc_free((unsigned __int64)v43);
  return v42;
}
