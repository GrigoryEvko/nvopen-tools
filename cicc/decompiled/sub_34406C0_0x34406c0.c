// Function: sub_34406C0
// Address: 0x34406c0
//
__int64 __fastcall sub_34406C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 i; // rsi
  __int64 v6; // rdx
  __int64 v7; // r9
  __int64 v8; // rax
  unsigned int v9; // edx

  for ( i = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 3); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v6 = *(unsigned int *)(a3 + 16);
      v7 = a1 + 8 * ((i >> 1) + (i & 0xFFFFFFFFFFFFFFFELL));
      v8 = *(unsigned int *)(v7 + 16);
      if ( (unsigned int)v6 > 6 || (v9 = dword_44E2140[v6], (unsigned int)v8 > 6) )
        BUG();
      if ( v9 > dword_44E2140[v8] )
        break;
      a1 = v7 + 24;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return a1;
    }
  }
  return a1;
}
