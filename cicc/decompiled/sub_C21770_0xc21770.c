// Function: sub_C21770
// Address: 0xc21770
//
bool __fastcall sub_C21770(__int64 a1)
{
  char *v1; // rax
  unsigned int v2; // ecx
  __int64 v3; // rdi
  char v4; // dl
  __int64 v5; // rsi
  __int64 v6; // rsi

  v1 = *(char **)(a1 + 8);
  v2 = 0;
  v3 = 0;
  do
  {
    if ( !v1 )
      return 0;
    v4 = *v1;
    v5 = *v1 & 0x7F;
    if ( v2 > 0x3E )
    {
      if ( v2 == 63 )
      {
        if ( v5 != (v4 & 1) )
          return 0;
      }
      else if ( (*v1 & 0x7F) != 0 )
      {
        return 0;
      }
    }
    v6 = v5 << v2;
    ++v1;
    v2 += 7;
    v3 += v6;
  }
  while ( v4 < 0 );
  return v3 == 0x5350524F46343204LL;
}
