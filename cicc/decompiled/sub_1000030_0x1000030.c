// Function: sub_1000030
// Address: 0x1000030
//
__int64 __fastcall sub_1000030(__int64 a1, __int64 a2, unsigned __int8 a3, const __m128i *a4)
{
  _BYTE *v4; // r14
  int v6; // r15d
  __int64 v7; // r13
  _BYTE *v8; // rax
  unsigned int v9; // r14d
  int v10; // edx
  _BYTE *v11; // rdx
  _BYTE *v12; // r9
  _BYTE *v13; // rax
  int v14; // eax
  __int64 v15; // r9
  bool v16; // al
  int v17; // eax
  _BYTE *v18; // rdx
  _BYTE *v19; // r9
  int v20; // r13d
  bool v21; // al
  int v22; // eax
  _BYTE *v23; // [rsp+0h] [rbp-50h]
  _BYTE *v24; // [rsp+0h] [rbp-50h]
  _BYTE *v25; // [rsp+0h] [rbp-50h]
  __int64 v26; // [rsp+8h] [rbp-48h]
  int v27; // [rsp+8h] [rbp-48h]
  _BYTE *v28; // [rsp+8h] [rbp-48h]
  _BYTE *v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+8h] [rbp-48h]
  _BYTE *v31; // [rsp+8h] [rbp-48h]

  if ( !a1 )
    return 0;
  v4 = *(_BYTE **)(a1 - 64);
  if ( !v4 )
    return 0;
  if ( !(unsigned __int8)sub_FFFE90(*(_QWORD *)(a1 - 32)) )
    return 0;
  v6 = sub_B53900(a1);
  if ( (unsigned int)(v6 - 32) > 1 )
    return 0;
  if ( *v4 != 44 || (v11 = (_BYTE *)*((_QWORD *)v4 - 8)) == 0 || (v12 = (_BYTE *)*((_QWORD *)v4 - 4)) == 0 )
  {
    if ( !a2 )
      return 0;
    goto LABEL_8;
  }
  if ( !a2 )
    return 0;
  v7 = *(_QWORD *)(a2 - 64);
  v13 = *(_BYTE **)(a2 - 32);
  if ( v11 == (_BYTE *)v7 && v12 == v13 )
  {
    v23 = (_BYTE *)*((_QWORD *)v4 - 4);
    v28 = (_BYTE *)*((_QWORD *)v4 - 8);
    v17 = sub_B53900(a2);
    v18 = v28;
    v19 = v23;
    v20 = v17;
    goto LABEL_54;
  }
  if ( v12 == (_BYTE *)v7 && v11 == v13 )
  {
    v25 = (_BYTE *)*((_QWORD *)v4 - 4);
    v31 = (_BYTE *)*((_QWORD *)v4 - 8);
    v22 = sub_B53960(a2);
    v19 = v25;
    v18 = v31;
    v20 = v22;
LABEL_54:
    v24 = v19;
    v29 = v18;
    v21 = sub_B532A0(v20);
    v11 = v29;
    v12 = v24;
    if ( v21 )
    {
      if ( ((v20 - 35) & 0xFFFFFFFD) == 0 && v6 == 33 && !a3 )
        return sub_AD6400(*(_QWORD *)(a2 + 8));
      if ( ((v20 - 34) & 0xFFFFFFFD) == 0 )
      {
        if ( v6 == 32 && a3 )
          return sub_AD6450(*(_QWORD *)(a2 + 8));
        if ( v6 == 33 )
          goto LABEL_21;
LABEL_62:
        if ( ((v20 - 35) & 0xFFFFFFFD) == 0 )
          goto LABEL_50;
        goto LABEL_63;
      }
      if ( v6 != 33 )
        goto LABEL_62;
    }
LABEL_63:
    v7 = *(_QWORD *)(a2 - 64);
    v13 = *(_BYTE **)(a2 - 32);
  }
  if ( !v7 )
    return 0;
  if ( v4 == (_BYTE *)v7 && v11 == v13 )
  {
    v30 = (__int64)v12;
    v14 = sub_B53900(a2);
    v15 = v30;
LABEL_35:
    if ( v14 == 35 )
    {
      if ( v6 == 33 && a3 )
      {
LABEL_38:
        if ( (unsigned __int8)sub_9B6260(v15, a4, 0) )
          return a2;
      }
    }
    else if ( (a3 | (v14 != 36)) != 1 && v6 == 32 )
    {
      goto LABEL_38;
    }
LABEL_8:
    v7 = *(_QWORD *)(a2 - 64);
    goto LABEL_9;
  }
  if ( v4 == v13 && v13 && v11 == (_BYTE *)v7 )
  {
    v26 = (__int64)v12;
    v14 = sub_B53960(a2);
    v15 = v26;
    goto LABEL_35;
  }
LABEL_9:
  if ( !v7 )
    return 0;
  v8 = *(_BYTE **)(a2 - 32);
  if ( v4 == v8 && v8 )
  {
    v27 = sub_B53900(a2);
    v16 = sub_B532A0(v27);
    v10 = v27;
    if ( v16 )
      goto LABEL_17;
    v7 = *(_QWORD *)(a2 - 64);
    if ( (_BYTE *)v7 != v4 )
      return 0;
  }
  else if ( (_BYTE *)v7 != v4 )
  {
    return 0;
  }
  if ( !v7 )
    return 0;
  v7 = *(_QWORD *)(a2 - 32);
  if ( !v7 )
    return 0;
  v9 = sub_B53900(a2);
  if ( !sub_B532A0(v9) )
    return 0;
  v10 = sub_B52F50(v9);
LABEL_17:
  switch ( v10 )
  {
    case '"':
      if ( v6 != 32 || !(unsigned __int8)sub_9B6260(v7, a4, 0) )
        return 0;
      goto LABEL_50;
    case '%':
      if ( v6 == 33 && (unsigned __int8)sub_9B6260(v7, a4, 0) )
      {
LABEL_21:
        if ( a3 )
          return a2;
        return a1;
      }
      return 0;
    case '$':
      if ( v6 == 33 )
        goto LABEL_21;
      if ( v6 != 32 || !a3 )
        return 0;
      return sub_AD6450(*(_QWORD *)(a2 + 8));
  }
  if ( v10 != 35 )
    return 0;
  if ( v6 == 32 )
  {
LABEL_50:
    if ( !a3 )
      return a2;
    return a1;
  }
  if ( a3 )
    return 0;
  return sub_AD6400(*(_QWORD *)(a2 + 8));
}
