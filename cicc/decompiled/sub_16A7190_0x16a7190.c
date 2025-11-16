// Function: sub_16A7190
// Address: 0x16a7190
//
_BOOL8 __fastcall sub_16A7190(__int64 a1, __int64 a2, _BOOL8 a3, unsigned int a4)
{
  __int64 v4; // r9
  _BOOL8 v5; // r8
  __int64 i; // rax
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rcx
  bool v9; // cf
  unsigned __int64 v10; // rdx

  if ( !a4 )
    return a3;
  v4 = a4;
  v5 = a3;
  for ( i = 0; i != v4; ++i )
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(a1 + 8 * i);
      v9 = __CFADD__(*(_QWORD *)(a2 + 8 * i), v8);
      v10 = *(_QWORD *)(a2 + 8 * i) + v8;
      if ( !v5 )
        break;
      v7 = v10 + 1;
      *(_QWORD *)(a1 + 8 * i) = v7;
      v5 = v7 <= v8;
      if ( v4 == ++i )
        return v5;
    }
    *(_QWORD *)(a1 + 8 * i) = v10;
    v5 = v9;
  }
  return v5;
}
