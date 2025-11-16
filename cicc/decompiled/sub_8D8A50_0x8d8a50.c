// Function: sub_8D8A50
// Address: 0x8d8a50
//
__int64 __fastcall sub_8D8A50(__m128i *a1, const __m128i *a2)
{
  __int8 v2; // dl
  __m128i *i; // rax
  __int8 v4; // cl
  const __m128i *v5; // rax
  int v6; // edx
  __int8 v8; // dl
  const __m128i *v9; // rax
  int v10; // r14d
  int v11; // eax
  __m128i *v12; // r13
  __m128i *v13; // rax
  __int8 v14; // dl
  __m128i *v15; // rax

  if ( dword_4F077C0 )
  {
    v2 = a1[8].m128i_i8[12];
    for ( i = a1; v2 == 12; v2 = i[8].m128i_i8[12] )
      i = (__m128i *)i[10].m128i_i64[0];
    v4 = a2[8].m128i_i8[12];
    if ( v4 == 12 )
    {
      v5 = a2;
      do
      {
        v5 = (const __m128i *)v5[10].m128i_i64[0];
        v4 = v5[8].m128i_i8[12];
      }
      while ( v4 == 12 );
    }
    if ( v4 != v2 )
    {
      if ( (unsigned int)sub_8D3B40((__int64)a1) )
      {
        v8 = a2[8].m128i_i8[12];
        if ( v8 == 12 )
        {
          v9 = a2;
          do
          {
            v9 = (const __m128i *)v9[10].m128i_i64[0];
            v8 = v9[8].m128i_i8[12];
          }
          while ( v8 == 12 );
        }
        if ( v8 )
          return (__int64)a1;
      }
      if ( (unsigned int)sub_8D3B40((__int64)a2) )
      {
        v14 = a1[8].m128i_i8[12];
        if ( v14 == 12 )
        {
          v15 = a1;
          do
          {
            v15 = (__m128i *)v15[10].m128i_i64[0];
            v14 = v15[8].m128i_i8[12];
          }
          while ( v14 == 12 );
        }
        if ( v14 )
          return (__int64)a1;
      }
    }
  }
  v6 = dword_4F077C4;
  if ( dword_4F077C4 == 2 )
    return sub_8D79B0(a1, (__int64)a2);
  if ( (a1[8].m128i_i8[12] & 0xFB) != 8 )
  {
    if ( (a2[8].m128i_i8[12] & 0xFB) != 8 )
      return sub_8D79B0(a1, (__int64)a2);
    v10 = 0;
    goto LABEL_22;
  }
  v10 = sub_8D4C10((__int64)a1, 1) & 0xFFFFFF8F;
  v11 = 0;
  if ( (a2[8].m128i_i8[12] & 0xFB) == 8 )
  {
    v6 = dword_4F077C4;
LABEL_22:
    v11 = sub_8D4C10((__int64)a2, v6 != 2) & 0xFFFFFF8F;
  }
  if ( v11 == v10 )
    return sub_8D79B0(a1, (__int64)a2);
  v12 = sub_73D4C0(a2, dword_4F077C4 == 2);
  v13 = sub_73D4C0(a1, dword_4F077C4 == 2);
  return sub_8D79B0(v13, (__int64)v12);
}
