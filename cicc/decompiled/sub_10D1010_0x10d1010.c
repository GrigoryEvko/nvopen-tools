// Function: sub_10D1010
// Address: 0x10d1010
//
_BOOL8 __fastcall sub_10D1010(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  _BOOL8 result; // rax
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rsi

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  v5 = *(_QWORD *)(v4 + 16);
  if ( v5
    && !*(_QWORD *)(v5 + 8)
    && *(_BYTE *)v4 == 58
    && (result = (*(_BYTE *)(v4 + 1) & 2) != 0, (*(_BYTE *)(v4 + 1) & 2) != 0)
    && (v11 = *(_QWORD *)(v4 - 64)) != 0
    && (**a1 = v11, (v12 = *(_QWORD *)(v4 - 32)) != 0) )
  {
    *a1[1] = v12;
    v6 = *((_QWORD *)a3 - 4);
    if ( v6 )
    {
      *a1[2] = v6;
      return result;
    }
  }
  else
  {
    v6 = *((_QWORD *)a3 - 4);
  }
  v7 = *(_QWORD *)(v6 + 16);
  if ( !v7 )
    return 0;
  if ( *(_QWORD *)(v7 + 8) )
    return 0;
  if ( *(_BYTE *)v6 != 58 )
    return 0;
  result = (*(_BYTE *)(v6 + 1) & 2) != 0;
  if ( (*(_BYTE *)(v6 + 1) & 2) == 0 )
    return 0;
  v8 = *(_QWORD *)(v6 - 64);
  if ( !v8 )
    return 0;
  **a1 = v8;
  v9 = *(_QWORD *)(v6 - 32);
  if ( !v9 )
    return 0;
  *a1[1] = v9;
  v10 = *((_QWORD *)a3 - 8);
  if ( !v10 )
    return 0;
  *a1[2] = v10;
  return result;
}
