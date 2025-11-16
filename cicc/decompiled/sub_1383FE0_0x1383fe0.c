// Function: sub_1383FE0
// Address: 0x1383fe0
//
__m128i *__fastcall sub_1383FE0(__m128i *a1, __m128i *a2, int a3)
{
  int v3; // ebx
  __m128i *result; // rax
  __m128i *v6; // rdi
  __int32 v7; // esi
  __int32 v8; // ecx
  __int32 v9; // r11d
  __int32 v10; // r8d
  int v11; // edx
  __int64 v12; // r9
  const __m128i *v13; // rdx

  if ( a1 == a2 )
    return a2;
  result = a1;
  v6 = (__m128i *)((char *)a1 + 24);
  if ( a2 == v6 )
    return a2;
  while ( 1 )
  {
    v7 = v6[-2].m128i_i32[2];
    LOBYTE(a3) = result[1].m128i_i32[2] == v7;
    v8 = v6[-2].m128i_i32[3];
    LOBYTE(v3) = v6->m128i_i32[1] == v8;
    v9 = v6[-1].m128i_i32[0];
    v10 = v6[-1].m128i_i32[1];
    v11 = v3 & a3;
    v12 = v6[-1].m128i_i64[1];
    LOBYTE(v3) = v6->m128i_i32[2] == v9;
    a3 = v3 & v11;
    if ( ((v6->m128i_i32[3] == v10) & (unsigned __int8)a3) != 0 && v6[1].m128i_i64[0] == v12 )
      break;
    result = v6;
    v6 = (__m128i *)((char *)v6 + 24);
    if ( a2 == v6 )
      return a2;
  }
  if ( a2 != result )
  {
    v13 = result + 3;
    if ( a2 == &result[3] )
    {
      return v6;
    }
    else
    {
      while ( 1 )
      {
        if ( v13->m128i_i32[3] != v10
          || v13->m128i_i32[2] != v9
          || v13->m128i_i32[0] != v7
          || v13->m128i_i32[1] != v8
          || v13[1].m128i_i64[0] != v12 )
        {
          result = (__m128i *)((char *)result + 24);
          *result = _mm_loadu_si128(v13);
          result[1].m128i_i64[0] = v13[1].m128i_i64[0];
        }
        v13 = (const __m128i *)((char *)v13 + 24);
        if ( a2 == v13 )
          break;
        v7 = result->m128i_i32[0];
        v8 = result->m128i_i32[1];
        v9 = result->m128i_i32[2];
        v10 = result->m128i_i32[3];
        v12 = result[1].m128i_i64[0];
      }
      return (__m128i *)((char *)result + 24);
    }
  }
  return result;
}
