// Function: sub_2366A40
// Address: 0x2366a40
//
__int64 *__fastcall sub_2366A40(__int64 *a1, const __m128i *a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // rdi
  const __m128i *v5; // r13
  __int64 v6; // rdx
  __int64 v7; // r14
  __int64 v8; // r12
  __m128i v9; // xmm1
  __int64 v10; // rbx
  unsigned __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r15
  unsigned __int64 v16; // [rsp+10h] [rbp-40h]
  const __m128i *v17; // [rsp+18h] [rbp-38h]

  v3 = 40 * a3;
  v17 = (const __m128i *)((char *)a2 + 40 * a3);
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( (unsigned __int64)(40 * a3) > 0x7FFFFFFFFFFFFFF8LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  if ( v3 )
  {
    v4 = 40 * a3;
    v5 = a2;
    v7 = sub_22077B0(40 * a3);
    *a1 = v7;
    for ( a1[2] = v7 + v3; v17 != v5; v7 += 40 )
    {
      if ( v7 )
      {
        v8 = v5[1].m128i_i64[1];
        v9 = _mm_loadu_si128(v5);
        *(_QWORD *)(v7 + 16) = 0;
        v10 = v5[1].m128i_i64[0];
        *(_QWORD *)(v7 + 24) = 0;
        *(_QWORD *)(v7 + 32) = 0;
        *(__m128i *)v7 = v9;
        v11 = v8 - v10;
        if ( v8 == v10 )
        {
          v13 = 0;
        }
        else
        {
          if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
            sub_4261EA(v4, v11, v6);
          v4 = v8 - v10;
          v16 = v8 - v10;
          v12 = sub_22077B0(v8 - v10);
          v8 = v5[1].m128i_i64[1];
          v10 = v5[1].m128i_i64[0];
          v11 = v16;
          v13 = v12;
        }
        *(_QWORD *)(v7 + 16) = v13;
        *(_QWORD *)(v7 + 24) = v13;
        for ( *(_QWORD *)(v7 + 32) = v13 + v11; v8 != v10; v13 += 40 )
        {
          if ( v13 )
          {
            v4 = v13 + 16;
            *(__m128i *)v13 = _mm_loadu_si128((const __m128i *)v10);
            sub_23667F0((__m128i **)(v13 + 16), (const __m128i **)(v10 + 16), v6);
          }
          v10 += 40;
        }
        *(_QWORD *)(v7 + 24) = v13;
      }
      v5 = (const __m128i *)((char *)v5 + 40);
    }
  }
  else
  {
    v7 = 0;
  }
  a1[1] = v7;
  return a1;
}
