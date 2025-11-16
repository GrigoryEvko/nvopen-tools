// Function: sub_28ED9A0
// Address: 0x28ed9a0
//
void __fastcall sub_28ED9A0(__int64 a1)
{
  __m128i *v1; // r13
  __int64 v2; // rax
  __m128i *v3; // rbx
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // rax
  __m128i *v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rax
  __m128i v10; // xmm0
  const __m128i *v11; // rax

  v1 = *(__m128i **)a1;
  v2 = 16LL * *(unsigned int *)(a1 + 8);
  v3 = (__m128i *)(*(_QWORD *)a1 + v2);
  v4 = v2 >> 4;
  if ( v2 )
  {
    while ( 1 )
    {
      v5 = v4;
      v6 = sub_2207800(16 * v4);
      v7 = (__m128i *)v6;
      if ( v6 )
        break;
      v4 >>= 1;
      if ( !v4 )
        goto LABEL_9;
    }
    v8 = v6 + v5 * 16;
    v9 = v6 + 16;
    *(__m128i *)(v9 - 16) = _mm_loadu_si128(v1);
    if ( v8 == v9 )
    {
      v11 = v7;
    }
    else
    {
      do
      {
        v10 = _mm_loadu_si128((const __m128i *)(v9 - 16));
        v9 += 16;
        *(__m128i *)(v9 - 16) = v10;
      }
      while ( v8 != v9 );
      v11 = &v7[v5 - 1];
    }
    *v1 = _mm_loadu_si128(v11);
    sub_28EC570(v1, v3, v7, v4);
  }
  else
  {
LABEL_9:
    v7 = 0;
    sub_28EB030(v1, v3);
  }
  j_j___libc_free_0((unsigned __int64)v7);
}
