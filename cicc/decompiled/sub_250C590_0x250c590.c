// Function: sub_250C590
// Address: 0x250c590
//
__int64 __fastcall sub_250C590(const __m128i *a1, unsigned __int8 **a2, __int64 a3)
{
  __int8 v4; // cl
  char v5; // al
  unsigned __int8 *v7; // rdi
  unsigned __int8 *v8; // r12

  v4 = a1->m128i_i8[8];
  v5 = *((_BYTE *)a2 + 8);
  if ( v5 == v4 )
  {
    if ( !v5 )
      return _mm_loadu_si128(a1).m128i_u64[0];
    v7 = *a2;
    if ( *a2 == (unsigned __int8 *)a1->m128i_i64[0] )
      return _mm_loadu_si128(a1).m128i_u64[0];
    if ( v7 )
      goto LABEL_13;
    return 0;
  }
  if ( !v5 )
    return _mm_loadu_si128(a1).m128i_u64[0];
  v7 = *a2;
  if ( !*a2 )
    return 0;
  if ( v4 )
  {
LABEL_13:
    v8 = (unsigned __int8 *)a1->m128i_i64[0];
    if ( a1->m128i_i64[0] )
    {
      if ( !a3 )
        a3 = *((_QWORD *)v8 + 1);
      if ( (unsigned int)*v8 - 12 <= 1 )
        return sub_250C3F0((unsigned __int64)v7, a3);
      if ( (unsigned int)*v7 - 12 <= 1 || v8 == (unsigned __int8 *)sub_250C3F0((unsigned __int64)v7, a3) )
        return _mm_loadu_si128(a1).m128i_u64[0];
    }
    return 0;
  }
  if ( a3 )
    return sub_250C3F0((unsigned __int64)v7, a3);
  return a3;
}
