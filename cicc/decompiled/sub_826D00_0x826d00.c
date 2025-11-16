// Function: sub_826D00
// Address: 0x826d00
//
__int64 __fastcall sub_826D00(__int64 a1)
{
  __int64 *v1; // rax
  unsigned int v2; // r8d
  __int64 v4; // rax
  __m128i *v5; // rax
  __m128i *i; // rdx

  v1 = *(__int64 **)(a1 + 120);
  v2 = 0;
  if ( !v1 )
    return 0;
  if ( !*((_BYTE *)v1 + 15) || (v1 = (__int64 *)*v1) != 0 )
  {
    v2 = 0;
    if ( !*v1 && *((_DWORD *)v1 + 2) == 4 )
    {
      v4 = v1[6];
      if ( v4 )
      {
        v5 = sub_73D7F0(*(_QWORD *)(v4 + 152));
        for ( i = *(__m128i **)(*(_QWORD *)(a1 + 8) + 64LL); v5[8].m128i_i8[12] == 12; v5 = (__m128i *)v5[10].m128i_i64[0] )
          ;
        while ( i[8].m128i_i8[12] == 12 )
          i = (__m128i *)i[10].m128i_i64[0];
        return v5 == i;
      }
    }
  }
  return v2;
}
