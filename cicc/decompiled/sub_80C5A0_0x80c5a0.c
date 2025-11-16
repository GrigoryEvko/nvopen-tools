// Function: sub_80C5A0
// Address: 0x80c5a0
//
__int64 __fastcall sub_80C5A0(__int64 a1, char a2, int a3, int a4, _DWORD *a5, _QWORD *a6)
{
  int v6; // r10d
  __int64 v10; // r12
  __int64 v11; // r8
  __int64 i; // rax
  _QWORD *v14; // rdi
  __int64 v15; // rdx
  unsigned __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r9
  const char *v20; // rsi
  _BOOL4 v21; // eax
  char v22; // r9
  __int64 j; // rcx
  bool v24; // bl
  char v25; // dl
  __int64 *v26; // rbx
  __int64 v27; // r11
  __int64 v28; // rax
  char v29; // dl
  __int64 k; // rax
  int v31; // eax
  int v32; // eax
  __int64 v33; // rax
  unsigned __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  char v38; // r9
  __int64 v39; // rcx
  bool v40; // dl
  int v41; // eax
  _BOOL4 v42; // eax
  _BOOL4 v43; // eax
  _BOOL4 v44; // eax
  _BOOL4 v45; // eax
  __int64 v46; // rax
  __int64 v47; // rax
  int v48; // eax
  _BOOL4 v49; // eax
  _BOOL4 v50; // eax
  const char *v51; // rdi
  const char *v52; // rsi
  int v53; // eax
  int v54; // eax
  _BOOL4 v55; // eax
  int v56; // [rsp-6Ch] [rbp-6Ch]
  int v57; // [rsp-68h] [rbp-68h]
  __int64 v58; // [rsp-68h] [rbp-68h]
  __int64 v59; // [rsp-60h] [rbp-60h]
  _DWORD *v60; // [rsp-60h] [rbp-60h]
  _DWORD *v61; // [rsp-60h] [rbp-60h]
  char v62; // [rsp-60h] [rbp-60h]
  char v63; // [rsp-58h] [rbp-58h]
  __int64 v64; // [rsp-58h] [rbp-58h]
  int v65; // [rsp-58h] [rbp-58h]
  int v66; // [rsp-58h] [rbp-58h]
  int v67; // [rsp-58h] [rbp-58h]
  int v68; // [rsp-58h] [rbp-58h]
  int v69; // [rsp-58h] [rbp-58h]
  int v70; // [rsp-58h] [rbp-58h]
  __int64 v71; // [rsp-58h] [rbp-58h]
  __int64 v72; // [rsp-50h] [rbp-50h]
  __int64 v73; // [rsp-50h] [rbp-50h]
  bool v74; // [rsp-50h] [rbp-50h]
  char v75; // [rsp-50h] [rbp-50h]
  char v76; // [rsp-50h] [rbp-50h]
  char v77; // [rsp-50h] [rbp-50h]
  char v78; // [rsp-50h] [rbp-50h]
  __int64 v79; // [rsp-50h] [rbp-50h]
  _DWORD *v80; // [rsp-48h] [rbp-48h]
  char v81; // [rsp-48h] [rbp-48h]
  __int64 v82; // [rsp-48h] [rbp-48h]
  __int64 v83; // [rsp-48h] [rbp-48h]
  __int64 v84; // [rsp-48h] [rbp-48h]
  __int64 v85; // [rsp-48h] [rbp-48h]
  bool v87; // [rsp-40h] [rbp-40h]
  char v88; // [rsp-40h] [rbp-40h]
  char v89; // [rsp-40h] [rbp-40h]
  char v90; // [rsp-40h] [rbp-40h]
  _DWORD *v91; // [rsp-40h] [rbp-40h]
  __int64 v92; // [rsp-40h] [rbp-40h]
  __int64 v93; // [rsp-40h] [rbp-40h]

  if ( a6[5] )
    return 0;
  v6 = a4;
  v10 = a1;
  if ( a2 == 6 )
  {
    v80 = a5;
    v10 = sub_809390(a1);
    v21 = sub_809440(v10);
    v6 = a4;
    a5 = v80;
    if ( !v21 )
      return 0;
    if ( (*(_BYTE *)(v10 + 91) & 2) == 0 )
      goto LABEL_38;
    i = qword_4F18C00[BYTE1(v10)];
    if ( !i )
      goto LABEL_38;
    goto LABEL_12;
  }
  if ( (*(_BYTE *)(a1 + 91) & 2) != 0 )
  {
    for ( i = qword_4F18C00[BYTE1(a1)]; i; i = *(_QWORD *)(i + 8) )
    {
LABEL_12:
      if ( *(_QWORD *)(i + 16) == v10 && (*(_BYTE *)(i + 40) & 1) == a3 )
      {
        if ( v6 )
          return 1;
        v14 = (_QWORD *)qword_4F18BE0;
        ++*a6;
        v15 = v14[2];
        if ( (unsigned __int64)(v15 + 1) > v14[1] )
        {
          v92 = i;
          sub_823810(v14);
          v14 = (_QWORD *)qword_4F18BE0;
          i = v92;
          v15 = *(_QWORD *)(qword_4F18BE0 + 16);
        }
        *(_BYTE *)(v14[4] + v15) = 83;
        ++v14[2];
        v16 = *(_QWORD *)(i + 32);
        if ( v16 > 0x24 )
        {
          sub_809040(v16 - 1, a6);
          v14 = (_QWORD *)qword_4F18BE0;
        }
        else if ( v16 )
        {
          ++*a6;
          v17 = v14[2];
          if ( (unsigned __int64)(v17 + 1) > v14[1] )
          {
            v93 = i;
            sub_823810(v14);
            v14 = (_QWORD *)qword_4F18BE0;
            i = v93;
            v17 = *(_QWORD *)(qword_4F18BE0 + 16);
          }
          *(_BYTE *)(v14[4] + v17) = a0123456789abcd_3[*(_QWORD *)(i + 32) - 1];
          ++v14[2];
        }
LABEL_22:
        ++*a6;
        v18 = v14[2];
        if ( (unsigned __int64)(v18 + 1) > v14[1] )
        {
          sub_823810(v14);
          v14 = (_QWORD *)qword_4F18BE0;
          v18 = *(_QWORD *)(qword_4F18BE0 + 16);
        }
        *(_BYTE *)(v14[4] + v18) = 95;
        ++v14[2];
        return 1;
      }
    }
  }
  if ( a2 == 28 )
  {
    if ( (*(_BYTE *)(v10 + 124) & 0x10) == 0 )
      goto LABEL_7;
    v20 = "St";
    goto LABEL_33;
  }
  if ( a2 == 59 )
  {
    v19 = *(_QWORD *)(v10 + 40);
    if ( !v19 || *(_BYTE *)(v19 + 28) != 3 || (*(_BYTE *)(*(_QWORD *)(v19 + 32) + 124LL) & 0x10) == 0 )
      goto LABEL_7;
    if ( *(_QWORD *)(v10 + 8) && !strcmp(*(const char **)(v10 + 8), "allocator") )
    {
      v20 = "Sa";
    }
    else
    {
      v91 = a5;
      v54 = sub_80A940(*(const char **)(v10 + 8), *(_QWORD *)(v10 + 40));
      a5 = v91;
      if ( !v54 )
        goto LABEL_7;
      v20 = "Sb";
    }
    goto LABEL_33;
  }
  if ( a2 != 6 )
  {
LABEL_7:
    v11 = *qword_4D03FD0;
    if ( !*qword_4D03FD0 )
      return 0;
    v87 = 0;
    v25 = 0;
    j = 0;
    v22 = 0;
    goto LABEL_42;
  }
LABEL_38:
  v22 = *(_BYTE *)(v10 + 140);
  for ( j = v10; v22 == 12; v22 = *(_BYTE *)(j + 140) )
    j = *(_QWORD *)(j + 160);
  v24 = (unsigned __int8)(v22 - 9) <= 1u;
  v25 = v24;
  if ( (unsigned __int8)(v22 - 9) > 1u )
  {
    v11 = *qword_4D03FD0;
    v87 = *qword_4D03FD0 == 0;
    goto LABEL_42;
  }
  v36 = *(_QWORD *)(v10 + 40);
  if ( v36 && *(_BYTE *)(v36 + 28) == 3 && (*(_BYTE *)(*(_QWORD *)(v36 + 32) + 124LL) & 0x10) != 0 )
  {
    v60 = a5;
    v65 = v6;
    v74 = (unsigned __int8)(v22 - 9) <= 1u;
    v82 = j;
    v88 = v22;
    v37 = sub_809820(v10);
    v38 = v88;
    v39 = v82;
    v40 = v74;
    v6 = v65;
    a5 = v60;
    if ( !v37 )
      goto LABEL_123;
    v41 = sub_80A940(*(const char **)(v37 + 8), *(_QWORD *)(v37 + 40));
    v39 = v82;
    v40 = v74;
    a5 = v60;
    if ( !v41 )
      goto LABEL_123;
    v66 = v6;
    v89 = v38;
    v42 = sub_80B1D0(v10);
    v38 = v89;
    v39 = v82;
    v20 = "Ss";
    v40 = v74;
    v6 = v66;
    a5 = v60;
    if ( !v42 )
    {
LABEL_123:
      v61 = a5;
      v67 = v6;
      v75 = v40;
      v83 = v39;
      v90 = v38;
      v43 = sub_80B840(v10, "basic_istream");
      v20 = "Si";
      v6 = v67;
      a5 = v61;
      if ( !v43 )
      {
        v44 = sub_80B840(v10, "basic_ostream");
        v6 = v67;
        a5 = v61;
        v20 = "So";
        if ( !v44 )
        {
          v45 = sub_80B840(v10, "basic_iostream");
          v6 = v67;
          a5 = v61;
          if ( !v45 )
          {
            v22 = v90;
            j = v83;
            v25 = v75;
            goto LABEL_80;
          }
          v20 = "Sd";
        }
      }
    }
LABEL_33:
    *a5 = 1;
    if ( !v6 )
    {
      *a6 += 2LL;
      sub_8238B0(qword_4F18BE0, v20, 2);
    }
    return 1;
  }
LABEL_80:
  v11 = *qword_4D03FD0;
  v87 = *qword_4D03FD0 == 0 && v24;
  if ( v87 )
  {
    if ( (*(_BYTE *)(v10 + 91) & 2) == 0 )
    {
      v25 = 1;
      v11 = 0;
      if ( (*(_BYTE *)(j + 177) & 0x20) != 0 || (*(_BYTE *)(j + 91) & 2) != 0 )
        goto LABEL_42;
      v68 = v6;
      v76 = v22;
      v84 = j;
      if ( !(unsigned int)sub_8D3AA0(j) )
        return 0;
      v46 = *(_QWORD *)(v84 + 40);
      if ( !v46 || *(_BYTE *)(v46 + 28) != 3 || (*(_BYTE *)(*(_QWORD *)(v46 + 32) + 124LL) & 0x10) == 0 )
        return 0;
      v47 = sub_809820(v84);
      j = v84;
      v22 = v76;
      v6 = v68;
      if ( v47 )
      {
        v48 = sub_80A940(*(const char **)(v47 + 8), *(_QWORD *)(v47 + 40));
        j = v84;
        if ( v48 )
        {
          v69 = v6;
          v77 = v22;
          v49 = sub_80B1D0(v84);
          v11 = 0;
          j = v84;
          v6 = v69;
          v22 = v77;
          v25 = 1;
          if ( v49 )
            goto LABEL_42;
        }
      }
      v70 = v6;
      v78 = v22;
      v85 = j;
      v50 = sub_80B840(j, "basic_istream");
      j = v85;
      v22 = v78;
      v6 = v70;
      if ( !v50 )
      {
        v55 = sub_80B840(v85, "basic_ostream");
        j = v85;
        v22 = v78;
        v6 = v70;
        if ( !v55 )
        {
          if ( !sub_80B840(v85, "basic_iostream") )
            return 0;
          j = v85;
          v6 = v70;
          v25 = 1;
          v11 = 0;
          v22 = v78;
          goto LABEL_42;
        }
      }
    }
    v25 = 1;
    v11 = 0;
  }
  else
  {
    v87 = *qword_4D03FD0 == 0;
  }
LABEL_42:
  v26 = (__int64 *)a6[2];
  if ( !v26 )
    return 0;
  v81 = v25 & 1;
  while ( 1 )
  {
    if ( *((_BYTE *)v26 + 24) != a2 )
      goto LABEL_44;
    v27 = v26[2];
    if ( a2 == 6 )
      break;
    if ( v10 == v27 )
      goto LABEL_62;
    if ( v11 )
    {
      v28 = *(_QWORD *)(v10 + 32);
      if ( *(_QWORD *)(v27 + 32) == v28 )
      {
        if ( v28 )
          goto LABEL_62;
      }
    }
LABEL_44:
    v26 = (__int64 *)*v26;
    if ( !v26 )
      return 0;
  }
  v29 = *(_BYTE *)(v27 + 140);
  for ( k = v26[2]; v29 == 12; v29 = *(_BYTE *)(k + 140) )
    k = *(_QWORD *)(k + 160);
  if ( v29 != v22 )
    goto LABEL_44;
  if ( v87 && k != j && v81 )
  {
    if ( (*(_BYTE *)(j + 177) & 0x20) == 0 || (*(_BYTE *)(k + 177) & 0x20) == 0 )
      goto LABEL_44;
    v51 = 0;
    if ( (*(_BYTE *)(j + 89) & 0x40) == 0 )
    {
      if ( (*(_BYTE *)(j + 89) & 8) != 0 )
        v51 = *(const char **)(j + 24);
      else
        v51 = *(const char **)(j + 8);
    }
    if ( (*(_BYTE *)(k + 89) & 0x40) == 0 )
    {
      v52 = (*(_BYTE *)(k + 89) & 8) != 0 ? *(const char **)(k + 24) : *(const char **)(k + 8);
      if ( v51 )
      {
        if ( v52 )
        {
          v56 = v6;
          v58 = j;
          v62 = v22;
          v71 = v11;
          v79 = v26[2];
          v53 = strcmp(v51, v52);
          v27 = v79;
          v11 = v71;
          v22 = v62;
          j = v58;
          v6 = v56;
          if ( v53 )
            goto LABEL_44;
        }
      }
    }
  }
  if ( (v26[5] & 1) != a3 )
    goto LABEL_44;
  if ( v10 != v27 )
  {
    v57 = v6;
    v59 = j;
    v63 = v22;
    v72 = v11;
    v31 = sub_8D97D0(v10, v27, 2304, j, v11);
    v11 = v72;
    v22 = v63;
    j = v59;
    v6 = v57;
    if ( !v31 )
      goto LABEL_44;
  }
  if ( dword_4F077BC )
  {
    v64 = j;
    v73 = v11;
    v32 = sub_808F30(v10, v26[2]);
    v11 = v73;
    j = v64;
    if ( v32 )
      goto LABEL_44;
  }
LABEL_62:
  if ( !v6 )
  {
    v14 = (_QWORD *)qword_4F18BE0;
    ++*a6;
    v33 = v14[2];
    if ( (unsigned __int64)(v33 + 1) > v14[1] )
    {
      sub_823810(v14);
      v14 = (_QWORD *)qword_4F18BE0;
      v33 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v14[4] + v33) = 83;
    ++v14[2];
    v34 = v26[4];
    if ( v34 > 0x24 )
    {
      sub_809040(v34 - 1, a6);
      v14 = (_QWORD *)qword_4F18BE0;
    }
    else if ( v34 )
    {
      ++*a6;
      v35 = v14[2];
      if ( (unsigned __int64)(v35 + 1) > v14[1] )
      {
        sub_823810(v14);
        v14 = (_QWORD *)qword_4F18BE0;
        v35 = *(_QWORD *)(qword_4F18BE0 + 16);
      }
      *(_BYTE *)(v14[4] + v35) = a0123456789abcd_3[v26[4] - 1];
      ++v14[2];
    }
    goto LABEL_22;
  }
  return 1;
}
