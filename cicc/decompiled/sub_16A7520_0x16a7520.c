// Function: sub_16A7520
// Address: 0x16a7520
//
_BOOL8 __fastcall sub_16A7520(__int64 a1, __int64 a2, _BOOL8 a3, unsigned int a4)
{
  __int64 v5; // r9
  _BOOL8 v6; // r8
  __int64 i; // rax
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rsi

  if ( !a4 )
    return a3;
  v5 = a4;
  v6 = a3;
  for ( i = 0; i != v5; ++i )
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(a1 + 8 * i);
      v10 = *(_QWORD *)(a2 + 8 * i);
      if ( !v6 )
        break;
      v8 = v9 - 1 - v10;
      *(_QWORD *)(a1 + 8 * i) = v8;
      v6 = v8 >= v9;
      if ( ++i == v5 )
        return v6;
    }
    *(_QWORD *)(a1 + 8 * i) = v9 - v10;
    v6 = v9 < v10;
  }
  return v6;
}
