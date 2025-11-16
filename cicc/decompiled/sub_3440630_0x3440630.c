// Function: sub_3440630
// Address: 0x3440630
//
__int64 __fastcall sub_3440630(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 i; // rsi
  __int64 v6; // r9
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // edx

  for ( i = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 3); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v6 = a1 + 8 * ((i >> 1) + (i & 0xFFFFFFFFFFFFFFFELL));
      v7 = *(unsigned int *)(a3 + 16);
      v8 = *(unsigned int *)(v6 + 16);
      if ( (unsigned int)v8 > 6 || (v9 = dword_44E2140[v8], (unsigned int)v7 > 6) )
        BUG();
      if ( v9 <= dword_44E2140[v7] )
        break;
      a1 = v6 + 24;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return a1;
    }
  }
  return a1;
}
