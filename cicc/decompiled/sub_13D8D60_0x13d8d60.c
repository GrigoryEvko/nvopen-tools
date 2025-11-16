// Function: sub_13D8D60
// Address: 0x13d8d60
//
_QWORD *__fastcall sub_13D8D60(__int64 a1, __int64 a2, __int64 a3, char a4, _QWORD *a5, unsigned int a6)
{
  __int64 v7; // r14
  __int64 v8; // r13
  unsigned int v9; // r12d
  _QWORD *result; // rax
  __int64 v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rbx
  unsigned __int8 v16; // al
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned __int8 v19; // al
  unsigned __int8 v20; // al
  __int64 v21; // rcx
  __int64 v22; // rdx
  int v23; // eax
  char v24; // al
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rcx
  char v33; // al
  __int64 v34; // [rsp+8h] [rbp-48h]
  int v35; // [rsp+10h] [rbp-40h]
  __int64 v38; // [rsp+18h] [rbp-38h]
  __int64 v39; // [rsp+18h] [rbp-38h]

  v7 = a3;
  v8 = a2;
  v9 = a1;
  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
  {
    if ( *(_BYTE *)(a3 + 16) <= 0x10u )
      return (_QWORD *)sub_14D7760(a1, a2, a3, *a5, a5[1]);
    v9 = sub_15FF5D0(a1);
    v8 = v7;
    v7 = a2;
  }
  v11 = **(_QWORD **)v8;
  if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) == 16 )
  {
    v12 = *(_QWORD *)(*(_QWORD *)v8 + 32LL);
    v11 = sub_1643320(v11);
    a2 = (unsigned int)v12;
    v15 = sub_16463B0(v11, (unsigned int)v12);
    if ( !v9 )
      return (_QWORD *)sub_15A0640(v15);
  }
  else
  {
    v15 = sub_1643320(v11);
    if ( !v9 )
      return (_QWORD *)sub_15A0640(v15);
  }
  if ( v9 == 15 )
    return (_QWORD *)sub_15A0600(v15);
  if ( (a4 & 2) != 0 )
  {
    if ( v9 == 8 )
      return (_QWORD *)sub_15A0640(v15);
    if ( v9 == 7 )
      return (_QWORD *)sub_15A0600(v15);
  }
  v16 = *(_BYTE *)(v7 + 16);
  if ( v16 == 14 )
  {
    if ( *(_QWORD *)(v7 + 32) == sub_16982C0(v11, a2, v13, v14) )
      v18 = *(_QWORD *)(v7 + 40) + 8LL;
    else
      v18 = v7 + 32;
  }
  else
  {
    v17 = *(_QWORD *)v7;
    if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) != 16 || v16 > 0x10u )
      goto LABEL_20;
    v11 = v7;
    v26 = sub_15A1020(v7);
    if ( !v26 || (v39 = v26, *(_BYTE *)(v26 + 16) != 14) )
    {
      a2 = 0;
      v35 = *(_QWORD *)(*(_QWORD *)v7 + 32LL);
      if ( !v35 )
        goto LABEL_15;
      while ( 1 )
      {
        v11 = v7;
        v31 = sub_15A0A60(v7, a2);
        v17 = v31;
        if ( !v31 )
          goto LABEL_20;
        v33 = *(_BYTE *)(v31 + 16);
        a2 = (unsigned int)a2;
        v34 = v17;
        if ( v33 != 9 )
        {
          if ( v33 != 14 )
            goto LABEL_20;
          a2 = (unsigned int)a2;
          v17 = *(_QWORD *)(v17 + 32) == sub_16982C0(v7, (unsigned int)a2, v17, v32)
              ? *(_QWORD *)(v34 + 40) + 8LL
              : v34 + 32;
          if ( (*(_BYTE *)(v17 + 18) & 7) != 1 )
            goto LABEL_20;
        }
        a2 = (unsigned int)(a2 + 1);
        if ( v35 == (_DWORD)a2 )
          goto LABEL_15;
      }
    }
    v29 = sub_16982C0(v7, a2, v27, v28);
    v17 = v39;
    if ( *(_QWORD *)(v39 + 32) == v29 )
      v18 = *(_QWORD *)(v39 + 40) + 8LL;
    else
      v18 = v39 + 32;
  }
  if ( (*(_BYTE *)(v18 + 18) & 7) == 1 )
  {
LABEL_15:
    v19 = sub_15FF810(v9);
    return (_QWORD *)sub_15A0680(v15, v19, 0);
  }
LABEL_20:
  if ( *(_BYTE *)(v8 + 16) == 9 )
    goto LABEL_15;
  v20 = *(_BYTE *)(v7 + 16);
  if ( v20 == 9 )
    goto LABEL_15;
  if ( v8 == v7 )
  {
    if ( (unsigned __int8)sub_15FF820(v9) )
      return (_QWORD *)sub_15A0600(v15);
    v11 = v9;
    if ( (unsigned __int8)sub_15FF850(v9) )
      return (_QWORD *)sub_15A0640(v15);
    v20 = *(_BYTE *)(v8 + 16);
  }
  v21 = v7 + 24;
  if ( v20 != 14 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) != 16 )
      goto LABEL_35;
    if ( v20 > 0x10u )
      goto LABEL_35;
    v11 = v7;
    v25 = sub_15A1020(v7);
    if ( !v25 || *(_BYTE *)(v25 + 16) != 14 )
      goto LABEL_35;
    v21 = v25 + 24;
  }
  v38 = v21;
  if ( *(_QWORD *)(v21 + 8) == sub_16982C0(v11, a2, v17, v21) )
  {
    v30 = *(_QWORD *)(v38 + 16);
    v22 = v30 + 8;
    v23 = *(_BYTE *)(v30 + 26) & 7;
    if ( !v23 )
    {
LABEL_26:
      if ( (*(_BYTE *)(v22 + 18) & 8) != 0 )
      {
        if ( v9 == 4 )
          return (_QWORD *)sub_15A0640(v15);
        if ( v9 != 11 )
          goto LABEL_29;
      }
      else
      {
        if ( v9 == 2 )
          return (_QWORD *)sub_15A0640(v15);
        if ( v9 != 13 )
          goto LABEL_29;
      }
      return (_QWORD *)sub_15A0600(v15);
    }
  }
  else
  {
    v22 = v38 + 8;
    LOBYTE(v23) = *(_BYTE *)(v38 + 26) & 7;
    if ( !(_BYTE)v23 )
      goto LABEL_26;
  }
  if ( (_BYTE)v23 == 3 )
  {
    if ( v9 == 4 )
      goto LABEL_34;
    if ( v9 != 11 )
      goto LABEL_35;
    goto LABEL_60;
  }
LABEL_29:
  if ( (*(_BYTE *)(v22 + 18) & 8) == 0 )
    goto LABEL_35;
  if ( v9 > 0xB )
  {
    if ( v9 != 14 )
      goto LABEL_35;
LABEL_60:
    if ( !(unsigned __int8)sub_14ABE10(v8, a5[1]) )
      goto LABEL_35;
    return (_QWORD *)sub_15A0600(v15);
  }
  if ( v9 > 9 )
    goto LABEL_60;
  if ( v9 == 1 || v9 - 4 <= 1 )
  {
LABEL_34:
    if ( !(unsigned __int8)sub_14ABE10(v8, a5[1]) )
      goto LABEL_35;
    return (_QWORD *)sub_15A0640(v15);
  }
LABEL_35:
  v24 = *(_BYTE *)(v8 + 16);
  if ( v24 == 79 || *(_BYTE *)(v7 + 16) == 79 )
  {
    result = sub_13D86C0(v9, v8, v7, a5, a6);
    if ( result )
      return result;
    v24 = *(_BYTE *)(v8 + 16);
  }
  if ( v24 == 77 )
    return (_QWORD *)sub_13D91D0(v9, v8, v7, a5, a6);
  result = 0;
  if ( *(_BYTE *)(v7 + 16) == 77 )
    return (_QWORD *)sub_13D91D0(v9, v8, v7, a5, a6);
  return result;
}
