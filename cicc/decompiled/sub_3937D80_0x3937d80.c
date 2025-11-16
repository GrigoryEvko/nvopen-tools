// Function: sub_3937D80
// Address: 0x3937d80
//
void __fastcall sub_3937D80(__m128i **a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v4; // rbx
  __m128i *v5; // r12
  __m128i *v6; // rax
  __int64 *v7; // rbx
  __int64 v8; // rax
  unsigned __int64 v9; // rdi
  __int64 *v10; // r14
  __int64 v11[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v12; // [rsp+10h] [rbp-30h]

  v4 = a2;
  v5 = *a1;
  if ( *a1 != (__m128i *)a1 )
  {
    if ( a2 == a3 )
    {
      do
      {
LABEL_13:
        v10 = (__int64 *)v5;
        v5 = (__m128i *)v5->m128i_i64[0];
        a1[2] = (__m128i *)((char *)a1[2] - 1);
        sub_2208CA0(v10);
        j_j___libc_free_0((unsigned __int64)v10);
      }
      while ( a1 != (__m128i **)v5 );
      return;
    }
    while ( 1 )
    {
      v5[1] = _mm_loadu_si128(v4 + 1);
      v5 = (__m128i *)v5->m128i_i64[0];
      v4 = (const __m128i *)v4->m128i_i64[0];
      if ( a1 == (__m128i **)v5 )
        break;
      if ( a3 == v4 )
        goto LABEL_13;
    }
  }
  if ( a3 != v4 )
  {
    v12 = 0;
    v11[1] = (__int64)v11;
    v11[0] = (__int64)v11;
    do
    {
      v6 = (__m128i *)sub_22077B0(0x20u);
      v6[1] = _mm_loadu_si128(v4 + 1);
      sub_2208C80(v6, (__int64)v11);
      ++v12;
      v4 = (const __m128i *)v4->m128i_i64[0];
    }
    while ( a3 != v4 );
    if ( (__int64 *)v11[0] != v11 )
    {
      sub_2208C50((__int64)a1, v11[0], (__int64)v11);
      v7 = (__int64 *)v11[0];
      v8 = v12;
      v12 = 0;
      a1[2] = (__m128i *)((char *)a1[2] + v8);
      while ( v7 != v11 )
      {
        v9 = (unsigned __int64)v7;
        v7 = (__int64 *)*v7;
        j_j___libc_free_0(v9);
      }
    }
  }
}
