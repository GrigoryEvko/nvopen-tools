// Function: sub_F8E430
// Address: 0xf8e430
//
void __fastcall sub_F8E430(unsigned __int64 *a1, __int64 a2)
{
  unsigned __int64 *v2; // rsi
  unsigned __int64 *v3; // rax
  unsigned __int64 i; // rdx

  v2 = &a1[a2];
  if ( a1 != v2 )
  {
    v3 = a1 + 1;
    for ( i = *a1; v2 != v3; ++v3 )
    {
      if ( i < *v3 )
        i = *v3;
    }
    if ( i > 0xFFFFFFFF )
    {
      _BitScanReverse64(&i, i);
      do
        *a1++ >>= 32 - ((unsigned __int8)i ^ 0x3Fu);
      while ( v2 != a1 );
    }
  }
}
