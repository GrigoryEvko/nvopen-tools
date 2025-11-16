// Function: sub_10CFE10
// Address: 0x10cfe10
//
_BOOL8 __fastcall sub_10CFE10(__int64 a1, int a2, unsigned __int8 *a3)
{
  _BOOL8 result; // rax
  char v5; // al
  _BYTE *v6; // rsi
  _BYTE *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = sub_995B10((_QWORD **)a1, *((_QWORD *)a3 - 8));
  v6 = (_BYTE *)*((_QWORD *)a3 - 4);
  result = 1;
  if ( !v5
    || *v6 <= 0x1Cu
    || (**(_QWORD **)(a1 + 8) = v6, *v6 != 57)
    || (v8 = *(_QWORD *)(a1 + 16), *((_QWORD *)v6 - 8) != v8) && v8 != *((_QWORD *)v6 - 4) )
  {
    if ( !(unsigned __int8)sub_995B10((_QWORD **)a1, (__int64)v6) )
      return 0;
    v7 = (_BYTE *)*((_QWORD *)a3 - 8);
    if ( *v7 <= 0x1Cu )
      return 0;
    **(_QWORD **)(a1 + 8) = v7;
    if ( *v7 != 57 )
      return 0;
    v9 = *(_QWORD *)(a1 + 16);
    if ( *((_QWORD *)v7 - 8) != v9 && v9 != *((_QWORD *)v7 - 4) )
      return 0;
  }
  return result;
}
