// Function: sub_6671B0
// Address: 0x6671b0
//
__int64 __fastcall sub_6671B0(const __m128i *a1, __int64 a2)
{
  __int64 result; // rax
  const __m128i *i; // rbx
  unsigned __int8 v4; // r13
  bool v5; // zf
  __m128i v6[3]; // [rsp+0h] [rbp-30h] BYREF

  result = a2;
  for ( i = a1; *(_BYTE *)(result + 140) == 12; result = *(_QWORD *)(result + 160) )
    ;
  v4 = *(_BYTE *)(result + 160);
  if ( a1 )
  {
    do
    {
      while ( 1 )
      {
        v5 = i[10].m128i_i8[13] == 1;
        i[8].m128i_i64[0] = a2;
        if ( v5 )
        {
          result = sub_621140((__int64)i, (__int64)i, v4);
          if ( !(_DWORD)result )
            break;
        }
        i = (const __m128i *)i[7].m128i_i64[1];
        if ( !i )
          return result;
      }
      v6[0] = _mm_loadu_si128(i + 11);
      result = ((__int64 (__fastcall *)(__m128i *, const __m128i *, _QWORD, _QWORD, _QWORD, _QWORD))sub_70FF50)(
                 v6,
                 i,
                 0,
                 0,
                 0,
                 0);
      i = (const __m128i *)i[7].m128i_i64[1];
    }
    while ( i );
  }
  return result;
}
