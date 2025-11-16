// Function: sub_2241820
// Address: 0x2241820
//
__int64 __fastcall sub_2241820(__int64 *a1, const void *a2, unsigned __int64 a3, size_t a4)
{
  __int64 v4; // r12
  size_t v5; // rax
  size_t v6; // rax
  __int64 v8; // r13

  v4 = -1;
  v5 = a1[1];
  if ( a4 <= v5 )
  {
    v6 = v5 - a4;
    v8 = *a1;
    if ( v6 <= a3 )
      a3 = v6;
    v4 = a3;
    if ( a4 )
    {
      while ( memcmp((const void *)(v8 + v4), a2, a4) )
      {
        if ( !v4 )
          return -1;
        --v4;
        if ( !a4 )
          return v4;
      }
    }
  }
  return v4;
}
