// Function: sub_39859D0
// Address: 0x39859d0
//
const __m128i *__fastcall sub_39859D0(__m128i *a1, const __m128i *a2)
{
  __int32 v2; // r15d
  __int64 v3; // r14
  const __m128i *result; // rax
  unsigned __int64 v5; // r14
  __int64 v6; // r15
  const __m128i *v7; // r14
  unsigned __int64 v8; // rbx
  __int32 v9; // [rsp+Ch] [rbp-94h]
  const __m128i *v10; // [rsp+20h] [rbp-80h]
  __m128i *v11; // [rsp+28h] [rbp-78h]
  char v12[8]; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v13; // [rsp+38h] [rbp-68h]
  char v14[8]; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int64 v15; // [rsp+58h] [rbp-48h]

  if ( a1 != a2 && a2 != &a1[1] )
  {
    v10 = a1 + 1;
    do
    {
      sub_15B1350(
        (__int64)v14,
        *(unsigned __int64 **)(v10->m128i_i64[1] + 24),
        *(unsigned __int64 **)(v10->m128i_i64[1] + 32));
      v5 = v15;
      sub_15B1350(
        (__int64)v12,
        *(unsigned __int64 **)(a1->m128i_i64[1] + 24),
        *(unsigned __int64 **)(a1->m128i_i64[1] + 32));
      if ( v5 < v13 )
      {
        v2 = v10->m128i_i32[0];
        v3 = v10->m128i_i64[1];
        if ( a1 != v10 )
          memmove(&a1[1], a1, (char *)v10 - (char *)a1);
        a1->m128i_i32[0] = v2;
        a1->m128i_i64[1] = v3;
      }
      else
      {
        v6 = v10->m128i_i64[1];
        v7 = v10;
        v9 = v10->m128i_i32[0];
        while ( 1 )
        {
          v11 = (__m128i *)v7--;
          sub_15B1350((__int64)v14, *(unsigned __int64 **)(v6 + 24), *(unsigned __int64 **)(v6 + 32));
          v8 = v15;
          sub_15B1350(
            (__int64)v12,
            *(unsigned __int64 **)(v7->m128i_i64[1] + 24),
            *(unsigned __int64 **)(v7->m128i_i64[1] + 32));
          if ( v8 >= v13 )
            break;
          v7[1] = _mm_loadu_si128(v7);
        }
        v11->m128i_i64[1] = v6;
        v11->m128i_i32[0] = v9;
      }
      result = ++v10;
    }
    while ( a2 != v10 );
  }
  return result;
}
