// Function: sub_10CFEC0
// Address: 0x10cfec0
//
__int64 __fastcall sub_10CFEC0(__int64 a1, int a2, unsigned __int8 *a3)
{
  _BYTE *v5; // r13
  _BYTE *v6; // r13
  char v7; // al
  __int64 v8; // rsi
  __int64 v9; // rax
  char v10; // al
  __int64 v11; // rsi

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v5 == 59
    && ((v10 = sub_995B10((_QWORD **)a1, *((_QWORD *)v5 - 8)), v11 = *((_QWORD *)v5 - 4), v10)
     && v11 == *(_QWORD *)(a1 + 8)
     || (unsigned __int8)sub_995B10((_QWORD **)a1, v11) && *((_QWORD *)v5 - 8) == *(_QWORD *)(a1 + 8)) )
  {
    v6 = (_BYTE *)*((_QWORD *)a3 - 4);
    if ( v6 )
    {
      **(_QWORD **)(a1 + 16) = v6;
      return 1;
    }
  }
  else
  {
    v6 = (_BYTE *)*((_QWORD *)a3 - 4);
  }
  if ( *v6 != 59 )
    return 0;
  v7 = sub_995B10((_QWORD **)a1, *((_QWORD *)v6 - 8));
  v8 = *((_QWORD *)v6 - 4);
  if ( (!v7 || v8 != *(_QWORD *)(a1 + 8))
    && (!(unsigned __int8)sub_995B10((_QWORD **)a1, v8) || *((_QWORD *)v6 - 8) != *(_QWORD *)(a1 + 8)) )
  {
    return 0;
  }
  v9 = *((_QWORD *)a3 - 8);
  if ( !v9 )
    return 0;
  **(_QWORD **)(a1 + 16) = v9;
  return 1;
}
