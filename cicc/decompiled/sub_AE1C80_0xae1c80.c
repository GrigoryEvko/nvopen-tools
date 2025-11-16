// Function: sub_AE1C80
// Address: 0xae1c80
//
__int64 __fastcall sub_AE1C80(__int64 a1, unsigned __int64 a2)
{
  __int64 v4; // r8
  _QWORD *v5; // rsi
  __int64 i; // rax
  __int64 v7; // rcx
  unsigned __int64 *v8; // rdx

  v4 = a1 + 24;
  v5 = (_QWORD *)(a1 + 24);
  for ( i = *(_DWORD *)(a1 + 20) & 0x7FFFFFFF; i > 0; i = i - v7 - 1 )
  {
    while ( 1 )
    {
      v7 = i >> 1;
      v8 = &v5[2 * (i >> 1)];
      if ( a2 >= *v8 )
        break;
      i >>= 1;
      if ( v7 <= 0 )
        return ((__int64)v5 - v4 - 16) >> 4;
    }
    v5 = v8 + 2;
  }
  return ((__int64)v5 - v4 - 16) >> 4;
}
