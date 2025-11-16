// Function: sub_10CFD60
// Address: 0x10cfd60
//
bool __fastcall sub_10CFD60(__int64 a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  _BYTE *v4; // rax
  _BYTE *v5; // rcx
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rsi
  __int64 v9; // rax

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v4 != 59 )
    goto LABEL_4;
  v8 = *((_QWORD *)v4 - 8);
  v9 = *((_QWORD *)v4 - 4);
  if ( v8 == *(_QWORD *)a1 && v9 )
  {
    **(_QWORD **)(a1 + 8) = v9;
  }
  else
  {
    if ( *(_QWORD *)a1 != v9 || !v8 )
    {
LABEL_4:
      v5 = (_BYTE *)*((_QWORD *)a3 - 4);
      goto LABEL_5;
    }
    **(_QWORD **)(a1 + 8) = v8;
  }
  v5 = (_BYTE *)*((_QWORD *)a3 - 4);
  result = 1;
  if ( *(_BYTE **)(a1 + 16) != v5 )
  {
LABEL_5:
    if ( *v5 == 59 )
    {
      v6 = *((_QWORD *)v5 - 8);
      v7 = *((_QWORD *)v5 - 4);
      if ( v6 == *(_QWORD *)a1 && v7 )
      {
        **(_QWORD **)(a1 + 8) = v7;
        return *((_QWORD *)a3 - 8) == *(_QWORD *)(a1 + 16);
      }
      if ( v6 && *(_QWORD *)a1 == v7 )
      {
        **(_QWORD **)(a1 + 8) = v6;
        return *((_QWORD *)a3 - 8) == *(_QWORD *)(a1 + 16);
      }
    }
    return 0;
  }
  return result;
}
