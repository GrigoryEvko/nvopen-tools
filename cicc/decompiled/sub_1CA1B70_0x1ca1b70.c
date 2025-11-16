// Function: sub_1CA1B70
// Address: 0x1ca1b70
//
_BYTE *__fastcall sub_1CA1B70(_QWORD *a1, __int64 a2, _BYTE *a3, __int64 a4, _QWORD *a5, int a6, unsigned __int8 a7)
{
  int v7; // r10d
  _QWORD *v11; // rax
  _QWORD *v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rdx
  char v16; // al
  _BYTE *v17; // r12
  unsigned __int8 v18; // al
  _BYTE *v19; // rax
  _QWORD *v21; // rax
  _QWORD *v22; // rax
  _QWORD *v23; // r14
  __int64 v24; // rsi
  __int64 *v25; // r12
  __int64 v26; // rax
  __int64 v27; // r8
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r15
  _QWORD *v31; // rdx
  _QWORD *v32; // r8
  _QWORD *v33; // rdi
  _QWORD *v34; // rax
  __int64 v35; // rax
  int v36; // r10d
  _QWORD *v37; // rax
  int v38; // r10d
  __int64 *v39; // rcx
  unsigned int v40; // eax
  __int64 v41; // rbx
  __int64 *v42; // rdx
  __int64 v43; // rax
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rdx
  __int64 *v47; // rsi
  __int64 v48; // rcx
  int v49; // eax
  __int64 v50; // rax
  int v51; // esi
  _BYTE *v52; // rsi
  __int64 *v53; // rax
  __int64 v54; // rdi
  unsigned __int64 v55; // rsi
  __int64 v56; // rdi
  __int64 v57; // rdx
  _BYTE *v58; // rsi
  __int64 *v59; // rax
  __int64 v60; // r14
  __int64 v61; // r10
  _QWORD *v62; // r13
  __int64 v63; // rax
  __int64 *v64; // r15
  _QWORD *v65; // rsi
  _QWORD *v66; // rax
  _QWORD *v67; // rdx
  _BOOL8 v68; // rdi
  __int64 v69; // rax
  _BYTE *i; // r14
  __int64 v71; // rdx
  bool v72; // al
  size_t v73; // rdx
  char *v74; // r14
  __int64 v75; // rcx
  size_t v76; // rdx
  char *v77; // r14
  size_t v78; // rax
  size_t v79; // rdx
  char *v80; // r14
  __int64 v81; // rcx
  size_t v82; // rdx
  char *v83; // r14
  size_t v84; // rax
  __int64 v85; // rsi
  __int64 *v86; // r12
  __int64 v87; // rsi
  unsigned __int8 *v88; // rsi
  _QWORD *v89; // rdi
  __int64 v90; // rsi
  unsigned __int8 *v91; // rsi
  _QWORD *v92; // rax
  int v93; // [rsp+1Ch] [rbp-B4h]
  int v94; // [rsp+1Ch] [rbp-B4h]
  __int64 v95; // [rsp+20h] [rbp-B0h]
  __int64 v96; // [rsp+20h] [rbp-B0h]
  __int64 v97; // [rsp+28h] [rbp-A8h]
  int v98; // [rsp+28h] [rbp-A8h]
  __int64 v99; // [rsp+28h] [rbp-A8h]
  __int64 v100; // [rsp+28h] [rbp-A8h]
  __int64 v101; // [rsp+28h] [rbp-A8h]
  __int64 v102; // [rsp+28h] [rbp-A8h]
  __int64 v103; // [rsp+28h] [rbp-A8h]
  int v105; // [rsp+30h] [rbp-A0h]
  __int64 v106; // [rsp+30h] [rbp-A0h]
  __int64 *v107; // [rsp+30h] [rbp-A0h]
  int v108; // [rsp+30h] [rbp-A0h]
  int v109; // [rsp+30h] [rbp-A0h]
  int v110; // [rsp+30h] [rbp-A0h]
  __int64 *v111; // [rsp+30h] [rbp-A0h]
  __int64 v112; // [rsp+30h] [rbp-A0h]
  __int64 v113; // [rsp+30h] [rbp-A0h]
  __int64 v114; // [rsp+30h] [rbp-A0h]
  _QWORD *v115; // [rsp+30h] [rbp-A0h]
  _QWORD *v116; // [rsp+30h] [rbp-A0h]
  _BYTE *v117; // [rsp+38h] [rbp-98h] BYREF
  __int64 *v118; // [rsp+40h] [rbp-90h] BYREF
  __int64 v119; // [rsp+48h] [rbp-88h]
  __int64 v120; // [rsp+50h] [rbp-80h]
  __int64 v121[2]; // [rsp+60h] [rbp-70h] BYREF
  __int16 v122; // [rsp+70h] [rbp-60h]
  char *v123; // [rsp+80h] [rbp-50h]
  __int64 v124; // [rsp+88h] [rbp-48h]
  char v125; // [rsp+90h] [rbp-40h] BYREF

  v7 = a6;
  v117 = a3;
  if ( !a7 )
  {
    v11 = (_QWORD *)a5[2];
    if ( v11 )
    {
      v13 = a5 + 1;
      do
      {
        while ( 1 )
        {
          v14 = v11[2];
          v15 = v11[3];
          if ( v11[4] >= (unsigned __int64)a3 )
            break;
          v11 = (_QWORD *)v11[3];
          if ( !v15 )
            goto LABEL_7;
        }
        v13 = v11;
        v11 = (_QWORD *)v11[2];
      }
      while ( v14 );
LABEL_7:
      if ( v13 != a5 + 1 && v13[4] <= (unsigned __int64)a3 )
        return (_BYTE *)*sub_1C9EA30(a5, (unsigned __int64 *)&v117);
    }
    v16 = sub_1C9F660((__int64)a1, a2, (unsigned __int64)a3);
    v7 = a6;
    if ( !v16 )
      goto LABEL_26;
  }
  v17 = v117;
  v123 = &v125;
  v124 = 0x200000000LL;
  v18 = v117[16];
  switch ( v18 )
  {
    case 3u:
LABEL_26:
      v21 = sub_1C9EA30(a5, (unsigned __int64 *)&v117);
      v17 = v117;
      *v21 = v117;
      return v17;
    case 0x35u:
    case 0x46u:
    case 0x11u:
    case 0x48u:
      goto LABEL_14;
    case 0x4Eu:
      v26 = *((_QWORD *)v117 - 3);
      if ( !*(_BYTE *)(v26 + 16) && (*(_BYTE *)(v26 + 33) & 0x20) != 0 )
      {
        if ( *(_DWORD *)(v26 + 36) != 3660 )
          return v17;
        v27 = sub_1CA1B70(
                (_DWORD)a1,
                a2,
                *(_QWORD *)&v117[-24 * (*((_DWORD *)v117 + 5) & 0xFFFFFFF)],
                (_DWORD)v117,
                (_DWORD)a5,
                v7,
                a7);
        v28 = *(_QWORD *)v27;
        v29 = *(_QWORD *)v27;
        if ( *(_BYTE *)(*(_QWORD *)v27 + 8LL) == 16 )
          v29 = **(_QWORD **)(v28 + 16);
        if ( *(_DWORD *)(v29 + 8) > 0x1FFu )
        {
          v17 = (_BYTE *)v27;
LABEL_16:
          if ( a7 )
            return v17;
LABEL_34:
          *sub_1C9EA30(a5, (unsigned __int64 *)&v117) = v17;
          return v17;
        }
        v111 = (__int64 *)v27;
        v121[0] = *(_QWORD *)v27;
        v121[1] = v28;
        v59 = (__int64 *)sub_15F2050((__int64)v117);
        v60 = sub_15E26F0(v59, 3660, v121, 2);
        v122 = 257;
        v118 = v111;
        v119 = *(_QWORD *)&v117[24 * (1LL - (*((_DWORD *)v117 + 5) & 0xFFFFFFF))];
        v112 = *(_QWORD *)(*(_QWORD *)v60 + 24LL);
        v19 = sub_1648AB0(72, 3u, 0);
        if ( v19 )
        {
          v61 = v112;
          v113 = (__int64)v19;
          v100 = v61;
          sub_15F1EA0((__int64)v19, **(_QWORD **)(v61 + 16), 54, (__int64)(v19 - 72), 3, (__int64)v117);
          *(_QWORD *)(v113 + 56) = 0;
          sub_15F5B40(v113, v100, v60, (__int64 *)&v118, 2, (__int64)v121, 0, 0);
          v19 = (_BYTE *)v113;
        }
LABEL_15:
        v17 = v19;
        goto LABEL_16;
      }
      goto LABEL_14;
    case 0x47u:
      v105 = v7;
      v97 = sub_1CA1B70((_DWORD)a1, a2, *((_QWORD *)v117 - 3), (_DWORD)v117, (_DWORD)a5, v7, a7);
      v106 = sub_1646BA0(*(__int64 **)(*(_QWORD *)v117 + 24LL), v105);
      v121[0] = (__int64)"bitCast";
      v122 = 259;
      v22 = sub_1648A60(56, 1u);
      v23 = v22;
      if ( v22 )
        sub_15FD590((__int64)v22, v97, v106, (__int64)v121, (__int64)v117);
LABEL_29:
      v24 = *((_QWORD *)v117 + 6);
      v25 = v23 + 6;
      v121[0] = v24;
      if ( v24 )
      {
        sub_1623A60((__int64)v121, v24, 2);
        if ( v25 == v121 )
        {
          if ( v121[0] )
            sub_161E7C0((__int64)v121, v121[0]);
          goto LABEL_33;
        }
        v87 = v23[6];
        if ( !v87 )
        {
LABEL_125:
          v88 = (unsigned __int8 *)v121[0];
          v23[6] = v121[0];
          if ( v88 )
            sub_1623210((__int64)v121, v88, (__int64)(v23 + 6));
LABEL_33:
          v17 = v23;
          if ( a7 )
            return v17;
          goto LABEL_34;
        }
      }
      else
      {
        if ( v25 == v121 )
          goto LABEL_33;
        v87 = v23[6];
        if ( !v87 )
          goto LABEL_33;
      }
      sub_161E7C0((__int64)(v23 + 6), v87);
      goto LABEL_125;
    case 0x56u:
LABEL_14:
      v19 = sub_1CA18D0((__int64)a1, v117, v7, a2);
      goto LABEL_15;
    case 0x38u:
      v69 = sub_1CA1B70(
              (_DWORD)a1,
              a2,
              *(_QWORD *)&v117[-24 * (*((_DWORD *)v117 + 5) & 0xFFFFFFF)],
              (_DWORD)v117,
              (_DWORD)a5,
              v7,
              a7);
      v120 = 0;
      v118 = 0;
      v119 = 0;
      v101 = v69;
      for ( i = &v117[24 * (1LL - (*((_DWORD *)v117 + 5) & 0xFFFFFFF))]; v117 != i; i += 24 )
      {
        v71 = *(_QWORD *)i;
        v121[0] = v71;
        sub_15E88C0((__int64)&v118, v121);
      }
      v121[0] = (__int64)"getElem";
      v122 = 259;
      v116 = sub_1704C00(*((_QWORD *)v117 + 7), v101, v118, (v119 - (__int64)v118) >> 3, (__int64)v121, (__int64)v117);
      v72 = sub_15FA300((__int64)v117);
      sub_15FA2E0((__int64)v116, v72);
      v73 = 0;
      v74 = off_4CD4970[0];
      if ( off_4CD4970[0] )
        v73 = strlen(off_4CD4970[0]);
      if ( *((_QWORD *)v117 + 6) || *((__int16 *)v117 + 9) < 0 )
      {
        v75 = sub_1625940((__int64)v117, v74, v73);
        if ( v75 )
        {
          v76 = 0;
          v77 = off_4CD4970[0];
          if ( off_4CD4970[0] )
          {
            v102 = v75;
            v78 = strlen(off_4CD4970[0]);
            v75 = v102;
            v76 = v78;
          }
          sub_1626100((__int64)v116, v77, v76, v75);
        }
      }
      v79 = 0;
      v80 = off_4CD4978[0];
      if ( off_4CD4978[0] )
        v79 = strlen(off_4CD4978[0]);
      if ( *((_QWORD *)v117 + 6) || *((__int16 *)v117 + 9) < 0 )
      {
        v81 = sub_1625940((__int64)v117, v80, v79);
        if ( v81 )
        {
          v82 = 0;
          v83 = off_4CD4978[0];
          if ( off_4CD4978[0] )
          {
            v103 = v81;
            v84 = strlen(off_4CD4978[0]);
            v81 = v103;
            v82 = v84;
          }
          sub_1626100((__int64)v116, v83, v82, v81);
        }
      }
      v85 = *((_QWORD *)v117 + 6);
      v121[0] = v85;
      v86 = v116 + 6;
      if ( v85 )
      {
        sub_1623A60((__int64)v121, v85, 2);
        if ( v86 == v121 )
        {
          if ( v121[0] )
            sub_161E7C0((__int64)v121, v121[0]);
          goto LABEL_117;
        }
        v90 = v116[6];
        if ( !v90 )
        {
LABEL_133:
          v91 = (unsigned __int8 *)v121[0];
          v116[6] = v121[0];
          if ( v91 )
            sub_1623210((__int64)v121, v91, (__int64)v86);
          goto LABEL_117;
        }
      }
      else if ( v86 == v121 || (v90 = v116[6]) == 0 )
      {
LABEL_117:
        if ( !a7 )
          *sub_1C9EA30(a5, (unsigned __int64 *)&v117) = v116;
        if ( v118 )
          j_j___libc_free_0(v118, v120 - (_QWORD)v118);
        return v116;
      }
      sub_161E7C0((__int64)v86, v90);
      goto LABEL_133;
  }
  if ( v18 <= 0x17u )
  {
    if ( v18 != 5 )
      goto LABEL_26;
    goto LABEL_14;
  }
  if ( v18 != 77 )
  {
    if ( v18 == 79 )
    {
      v98 = v7;
      v107 = (__int64 *)sub_1CA2860((_DWORD)a1, a2, *((_QWORD *)v117 - 6), (_DWORD)v117, (_DWORD)a5, v7, a7);
      v99 = sub_1CA2860((_DWORD)a1, a2, *((_QWORD *)v117 - 3), (_DWORD)v117, (_DWORD)a5, v98, a7);
      v121[0] = (__int64)"selectInst";
      v122 = 259;
      v30 = *((_QWORD *)v117 - 9);
      v23 = sub_1648A60(56, 3u);
      if ( v23 )
      {
        sub_15F1EA0((__int64)v23, *v107, 55, (__int64)(v23 - 9), 3, (__int64)v117);
        sub_1593B40(v23 - 9, v30);
        sub_1593B40(v23 - 6, (__int64)v107);
        sub_1593B40(v23 - 3, v99);
        sub_164B780((__int64)v23, v121);
      }
      goto LABEL_29;
    }
    if ( v18 != 54 )
      goto LABEL_26;
    goto LABEL_14;
  }
  v31 = (_QWORD *)a1[24];
  v118 = (__int64 *)v117;
  v32 = a1 + 23;
  if ( !v31 )
    goto LABEL_54;
  v33 = a1 + 23;
  v34 = v31;
  do
  {
    if ( v34[4] < (unsigned __int64)v117 )
    {
      v34 = (_QWORD *)v34[3];
    }
    else
    {
      v33 = v34;
      v34 = (_QWORD *)v34[2];
    }
  }
  while ( v34 );
  if ( v33 != v32 && v33[4] <= (unsigned __int64)v117 )
  {
    v62 = a1 + 23;
    do
    {
      if ( v31[4] < (unsigned __int64)v117 )
      {
        v31 = (_QWORD *)v31[3];
      }
      else
      {
        v62 = v31;
        v31 = (_QWORD *)v31[2];
      }
    }
    while ( v31 );
    if ( v62 == v32 || v62[4] > (unsigned __int64)v117 )
    {
      v115 = a1 + 23;
      v63 = sub_22077B0(48);
      v64 = v118;
      v65 = v62;
      *(_QWORD *)(v63 + 40) = 0;
      v62 = (_QWORD *)v63;
      *(_QWORD *)(v63 + 32) = v64;
      v66 = sub_1C9EBD0(a1 + 22, v65, (unsigned __int64 *)(v63 + 32));
      if ( v67 )
      {
        v68 = v66 || v115 == v67 || (unsigned __int64)v64 < v67[4];
        sub_220F040(v68, v62, v67, v115);
        ++a1[27];
      }
      else
      {
        v89 = v62;
        v62 = v66;
        j_j___libc_free_0(v89, 48);
      }
    }
    return (_BYTE *)v62[5];
  }
  else
  {
LABEL_54:
    v93 = v7;
    v95 = sub_1646BA0(*(__int64 **)(*(_QWORD *)v117 + 24LL), v7);
    v121[0] = (__int64)"phiNode";
    v122 = 259;
    v108 = *((_DWORD *)v118 + 5) & 0xFFFFFFF;
    v35 = sub_1648B60(64);
    v36 = v93;
    v17 = (_BYTE *)v35;
    if ( v35 )
    {
      sub_15F1EA0(v35, v95, 53, 0, 0, (__int64)v118);
      *((_DWORD *)v17 + 14) = v108;
      sub_164B780((__int64)v17, v121);
      sub_1648880((__int64)v17, *((_DWORD *)v17 + 14), 1);
      v36 = v93;
    }
    v109 = v36;
    v37 = sub_1C9ECD0(a1 + 22, (unsigned __int64 *)&v118);
    v38 = v109;
    *v37 = v17;
    if ( !a7 )
    {
      v92 = sub_1C9EA30(a5, (unsigned __int64 *)&v117);
      v38 = v109;
      *v92 = v17;
    }
    v39 = v118;
    v40 = *((_DWORD *)v118 + 5) & 0xFFFFFFF;
    if ( v40 )
    {
      v41 = 0;
      do
      {
        if ( (*((_BYTE *)v39 + 23) & 0x40) != 0 )
          v42 = (__int64 *)*(v39 - 1);
        else
          v42 = &v39[-3 * v40];
        v110 = v38;
        v43 = sub_1CA2860((_DWORD)a1, a2, v42[3 * v41], (_DWORD)v39, (_DWORD)a5, v38, a7);
        v38 = v110;
        v46 = v43;
        if ( (*((_BYTE *)v118 + 23) & 0x40) != 0 )
          v47 = (__int64 *)*(v118 - 1);
        else
          v47 = &v118[-3 * (*((_DWORD *)v118 + 5) & 0xFFFFFFF)];
        v48 = v47[3 * *((unsigned int *)v118 + 14) + 1 + v41];
        v49 = *((_DWORD *)v17 + 5) & 0xFFFFFFF;
        if ( v49 == *((_DWORD *)v17 + 14) )
        {
          v94 = v110;
          v96 = v47[3 * *((unsigned int *)v118 + 14) + 1 + v41];
          v114 = v46;
          sub_15F55D0((__int64)v17, (__int64)v47, v46, v48, v44, v45);
          v38 = v94;
          v48 = v96;
          v46 = v114;
          v49 = *((_DWORD *)v17 + 5) & 0xFFFFFFF;
        }
        v50 = (v49 + 1) & 0xFFFFFFF;
        v51 = v50 | *((_DWORD *)v17 + 5) & 0xF0000000;
        *((_DWORD *)v17 + 5) = v51;
        if ( (v51 & 0x40000000) != 0 )
          v52 = (_BYTE *)*((_QWORD *)v17 - 1);
        else
          v52 = &v17[-24 * v50];
        v53 = (__int64 *)&v52[24 * (unsigned int)(v50 - 1)];
        if ( *v53 )
        {
          v54 = v53[1];
          v55 = v53[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v55 = v54;
          if ( v54 )
            *(_QWORD *)(v54 + 16) = *(_QWORD *)(v54 + 16) & 3LL | v55;
        }
        *v53 = v46;
        if ( v46 )
        {
          v56 = *(_QWORD *)(v46 + 8);
          v53[1] = v56;
          if ( v56 )
            *(_QWORD *)(v56 + 16) = (unsigned __int64)(v53 + 1) | *(_QWORD *)(v56 + 16) & 3LL;
          v53[2] = v53[2] & 3 | (v46 + 8);
          *(_QWORD *)(v46 + 8) = v53;
        }
        v57 = *((_DWORD *)v17 + 5) & 0xFFFFFFF;
        if ( (v17[23] & 0x40) != 0 )
          v58 = (_BYTE *)*((_QWORD *)v17 - 1);
        else
          v58 = &v17[-24 * v57];
        ++v41;
        *(_QWORD *)&v58[24 * *((unsigned int *)v17 + 14) + 8 + 8 * (unsigned int)(v57 - 1)] = v48;
        v39 = v118;
        v40 = *((_DWORD *)v118 + 5) & 0xFFFFFFF;
      }
      while ( v40 > (unsigned int)v41 );
    }
  }
  return v17;
}
