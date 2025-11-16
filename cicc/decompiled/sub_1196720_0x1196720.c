// Function: sub_1196720
// Address: 0x1196720
//
bool __fastcall sub_1196720(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  _BYTE *v5; // rax
  _BYTE *v6; // r13
  char v7; // al
  __int64 v8; // rax
  _BYTE *v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // rdx
  char v12; // al
  bool v13; // zf
  __int64 v14; // rsi

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  if ( *(_BYTE *)a2 != 57 )
    return 0;
  v5 = *(_BYTE **)(a2 - 64);
  if ( *v5 != 42 || (v11 = *((_QWORD *)v5 - 8)) == 0 )
  {
LABEL_8:
    v6 = *(_BYTE **)(a2 - 32);
    v7 = *v6;
    goto LABEL_9;
  }
  **a1 = v11;
  v12 = sub_995B10(a1 + 1, *((_QWORD *)v5 - 4));
  v6 = *(_BYTE **)(a2 - 32);
  v13 = v12 == 0;
  v7 = *v6;
  if ( !v13 && v7 == 59 )
  {
    if ( (unsigned __int8)sub_995B10(a1 + 2, *((_QWORD *)v6 - 8)) )
    {
      v14 = *((_QWORD *)v6 - 4);
      if ( v14 == *a1[3] )
        return 1;
    }
    else
    {
      v14 = *((_QWORD *)v6 - 4);
    }
    if ( (unsigned __int8)sub_995B10(a1 + 2, v14) && *((_QWORD *)v6 - 8) == *a1[3] )
      return 1;
    goto LABEL_8;
  }
LABEL_9:
  if ( v7 == 42 )
  {
    v8 = *((_QWORD *)v6 - 8);
    if ( v8 )
    {
      **a1 = v8;
      if ( (unsigned __int8)sub_995B10(a1 + 1, *((_QWORD *)v6 - 4)) )
      {
        v9 = *(_BYTE **)(a2 - 64);
        if ( *v9 == 59 )
        {
          if ( !(unsigned __int8)sub_995B10(a1 + 2, *((_QWORD *)v9 - 8)) )
          {
            v10 = *((_QWORD *)v9 - 4);
            goto LABEL_15;
          }
          v10 = *((_QWORD *)v9 - 4);
          if ( v10 != *a1[3] )
          {
LABEL_15:
            if ( (unsigned __int8)sub_995B10(a1 + 2, v10) )
              return *a1[3] == *((_QWORD *)v9 - 8);
            return 0;
          }
          return 1;
        }
      }
    }
  }
  return 0;
}
