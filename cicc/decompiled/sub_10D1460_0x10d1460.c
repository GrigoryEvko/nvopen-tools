// Function: sub_10D1460
// Address: 0x10d1460
//
bool __fastcall sub_10D1460(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  _BYTE *v4; // r12
  _BYTE *v6; // r12
  _BYTE *v7; // r13
  _BYTE *v8; // r13
  char v9; // al
  __int64 v10; // rsi
  __int64 v11; // rax
  _BYTE *v12; // r13
  _BYTE *v13; // r13
  char v14; // al
  __int64 v15; // rsi
  __int64 v16; // rax
  char v17; // al
  __int64 v18; // rsi
  char v19; // al
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v4 != 57 )
    goto LABEL_4;
  v12 = (_BYTE *)*((_QWORD *)v4 - 8);
  if ( *v12 != 59 )
    goto LABEL_18;
  v17 = sub_995B10(a1, *((_QWORD *)v12 - 8));
  v18 = *((_QWORD *)v12 - 4);
  if ( v17 && v18 )
  {
    *a1[1] = v18;
  }
  else
  {
    if ( !(unsigned __int8)sub_995B10(a1, v18) || (v21 = *((_QWORD *)v12 - 8)) == 0 )
    {
LABEL_18:
      v13 = (_BYTE *)*((_QWORD *)v4 - 4);
      goto LABEL_19;
    }
    *a1[1] = v21;
  }
  v13 = (_BYTE *)*((_QWORD *)v4 - 4);
  if ( !v13 )
  {
LABEL_19:
    if ( *v13 == 59 )
    {
      v14 = sub_995B10(a1, *((_QWORD *)v13 - 8));
      v15 = *((_QWORD *)v13 - 4);
      if ( v14 && v15 )
      {
        *a1[1] = v15;
      }
      else
      {
        if ( !(unsigned __int8)sub_995B10(a1, v15) )
          goto LABEL_4;
        v23 = *((_QWORD *)v13 - 8);
        if ( !v23 )
          goto LABEL_4;
        *a1[1] = v23;
      }
      v16 = *((_QWORD *)v4 - 8);
      if ( v16 )
      {
        *a1[2] = v16;
        goto LABEL_25;
      }
    }
LABEL_4:
    v6 = (_BYTE *)*((_QWORD *)a3 - 4);
    goto LABEL_5;
  }
  *a1[2] = v13;
LABEL_25:
  v6 = (_BYTE *)*((_QWORD *)a3 - 4);
  result = 1;
  if ( (_BYTE *)*a1[3] != v6 )
  {
LABEL_5:
    if ( *v6 != 57 )
      return 0;
    v7 = (_BYTE *)*((_QWORD *)v6 - 8);
    if ( *v7 != 59 )
      goto LABEL_9;
    v19 = sub_995B10(a1, *((_QWORD *)v7 - 8));
    v20 = *((_QWORD *)v7 - 4);
    if ( v19 && v20 )
    {
      *a1[1] = v20;
    }
    else
    {
      if ( !(unsigned __int8)sub_995B10(a1, v20) || (v22 = *((_QWORD *)v7 - 8)) == 0 )
      {
LABEL_9:
        v8 = (_BYTE *)*((_QWORD *)v6 - 4);
        goto LABEL_10;
      }
      *a1[1] = v22;
    }
    v8 = (_BYTE *)*((_QWORD *)v6 - 4);
    if ( v8 )
    {
      *a1[2] = v8;
      return *a1[3] == *((_QWORD *)a3 - 8);
    }
LABEL_10:
    if ( *v8 != 59 )
      return 0;
    v9 = sub_995B10(a1, *((_QWORD *)v8 - 8));
    v10 = *((_QWORD *)v8 - 4);
    if ( v9 && v10 )
    {
      *a1[1] = v10;
    }
    else
    {
      if ( !(unsigned __int8)sub_995B10(a1, v10) )
        return 0;
      v24 = *((_QWORD *)v8 - 8);
      if ( !v24 )
        return 0;
      *a1[1] = v24;
    }
    v11 = *((_QWORD *)v6 - 8);
    if ( !v11 )
      return 0;
    *a1[2] = v11;
    return *a1[3] == *((_QWORD *)a3 - 8);
  }
  return result;
}
