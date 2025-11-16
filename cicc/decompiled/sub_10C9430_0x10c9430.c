// Function: sub_10C9430
// Address: 0x10c9430
//
__int64 __fastcall sub_10C9430(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  _BYTE *v5; // r13
  _BYTE *v6; // r13
  char v7; // al
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rax
  char v11; // al
  __int64 v12; // rsi
  __int64 v13; // rax

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v5 != 59 )
    goto LABEL_4;
  v11 = sub_995B10(a1, *((_QWORD *)v5 - 8));
  v12 = *((_QWORD *)v5 - 4);
  if ( v11 && v12 )
  {
    *a1[1] = v12;
  }
  else
  {
    if ( !(unsigned __int8)sub_995B10(a1, v12) || (v13 = *((_QWORD *)v5 - 8)) == 0 )
    {
LABEL_4:
      v6 = (_BYTE *)*((_QWORD *)a3 - 4);
      goto LABEL_5;
    }
    *a1[1] = v13;
  }
  v6 = (_BYTE *)*((_QWORD *)a3 - 4);
  if ( v6 )
  {
    *a1[2] = v6;
    return 1;
  }
LABEL_5:
  if ( *v6 == 59 )
  {
    v7 = sub_995B10(a1, *((_QWORD *)v6 - 8));
    v8 = *((_QWORD *)v6 - 4);
    if ( v7 && v8 )
    {
      *a1[1] = v8;
    }
    else
    {
      if ( !(unsigned __int8)sub_995B10(a1, v8) )
        return 0;
      v9 = *((_QWORD *)v6 - 8);
      if ( !v9 )
        return 0;
      *a1[1] = v9;
    }
    v10 = *((_QWORD *)a3 - 8);
    if ( v10 )
    {
      *a1[2] = v10;
      return 1;
    }
  }
  return 0;
}
