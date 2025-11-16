// Function: sub_B89740
// Address: 0xb89740
//
__m128i *__fastcall sub_B89740(__int64 a1, __int64 *a2)
{
  __int64 v4; // rax
  __int64 *v5; // r8
  __int64 v6; // r12
  __m128i *result; // rax
  __int64 *v8; // r14
  __int64 v9; // r15
  __int64 v10; // rcx
  __m128i **v11; // rdi
  __m128i *v12; // rdx
  __m128i *v13; // rsi
  __m128i v14; // [rsp+10h] [rbp-40h] BYREF

  v4 = sub_B873F0(*(_QWORD *)(a1 + 8), a2);
  v5 = *(__int64 **)v4;
  v6 = *(_QWORD *)v4 + 8LL * *(unsigned int *)(v4 + 8);
  result = &v14;
  if ( (__int64 *)v6 != v5 )
  {
    v8 = v5;
    do
    {
      v9 = *v8;
      result = (__m128i *)sub_B81110(a1, *v8, 1);
      v10 = (__int64)result;
      if ( result )
      {
        v11 = (__m128i **)a2[1];
        result = *v11;
        v12 = v11[1];
        if ( *v11 == v12 )
          goto LABEL_11;
        while ( v9 != result->m128i_i64[0] )
        {
          if ( v12 == ++result )
            goto LABEL_11;
        }
        if ( v10 != result->m128i_i64[1] )
        {
LABEL_11:
          v14.m128i_i64[1] = v10;
          v13 = v11[1];
          v14.m128i_i64[0] = v9;
          if ( v13 == v11[2] )
          {
            result = (__m128i *)sub_B83480((const __m128i **)v11, v13, &v14);
          }
          else
          {
            if ( v13 )
            {
              *v13 = _mm_loadu_si128(&v14);
              v13 = v11[1];
            }
            v11[1] = v13 + 1;
          }
        }
      }
      ++v8;
    }
    while ( (__int64 *)v6 != v8 );
  }
  return result;
}
