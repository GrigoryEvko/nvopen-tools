// Function: sub_F86830
// Address: 0xf86830
//
__int64 __fastcall sub_F86830(__int64 a1, __int64 a2)
{
  __m128i *v2; // r13
  __int64 v3; // rax
  __int64 *v4; // r15
  __int64 v5; // r9
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 *v8; // r12
  __int64 v9; // rcx
  __int64 v10; // rax
  __m128i v11; // xmm0
  __int64 *v12; // rax
  __int64 v14; // [rsp+8h] [rbp-38h]

  v2 = *(__m128i **)a1;
  v3 = 16LL * *(unsigned int *)(a1 + 8);
  v4 = (__int64 *)(*(_QWORD *)a1 + v3);
  v5 = v3 >> 4;
  if ( v3 )
  {
    while ( 1 )
    {
      v14 = v5;
      v6 = 2 * v5;
      v7 = sub_2207800(16 * v5, &unk_435FF63);
      v8 = (__int64 *)v7;
      if ( v7 )
        break;
      v5 = v14 >> 1;
      if ( !(v14 >> 1) )
        goto LABEL_9;
    }
    v9 = v7 + v6 * 8;
    v10 = v7 + 16;
    *(__m128i *)(v10 - 16) = _mm_loadu_si128(v2);
    if ( v9 == v10 )
    {
      v12 = v8;
    }
    else
    {
      do
      {
        v11 = _mm_loadu_si128((const __m128i *)(v10 - 16));
        v10 += 16;
        *(__m128i *)(v10 - 16) = v11;
      }
      while ( v9 != v10 );
      v12 = &v8[v6 - 2];
    }
    v2->m128i_i64[0] = *v12;
    v2->m128i_i64[1] = v12[1];
    sub_F86740(v2->m128i_i64, v4, v8, v14, a2);
  }
  else
  {
LABEL_9:
    v6 = 0;
    sub_F7B580(v2->m128i_i64, v4, a2);
    v8 = 0;
  }
  return j_j___libc_free_0(v8, v6 * 8);
}
