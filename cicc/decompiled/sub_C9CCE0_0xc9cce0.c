// Function: sub_C9CCE0
// Address: 0xc9cce0
//
__int64 __fastcall sub_C9CCE0(__int64 a1, double *a2)
{
  double *v2; // r12
  double v3; // xmm0_8
  const __m128i *v4; // rbx
  _BYTE *v5; // rsi
  __m128i *v6; // r14
  __int64 v7; // rdx
  __m128i v8; // xmm4
  __m128i v9; // xmm5
  _BYTE *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rdx
  __m128i v15; // xmm3
  __m128i *v16; // r15
  __m128i v17; // xmm2
  __m128i v18; // xmm6
  __m128i v19; // xmm7
  __int64 result; // rax
  __m128i v22; // [rsp+30h] [rbp-A0h] BYREF
  __m128i v23; // [rsp+40h] [rbp-90h] BYREF
  __int64 v24; // [rsp+50h] [rbp-80h]
  __int64 v25[2]; // [rsp+58h] [rbp-78h] BYREF
  _QWORD v26[2]; // [rsp+68h] [rbp-68h] BYREF
  __int64 v27[2]; // [rsp+78h] [rbp-58h] BYREF
  _QWORD v28[9]; // [rsp+88h] [rbp-48h] BYREF

  if ( (double *)a1 != a2 )
  {
    v2 = (double *)(a1 + 104);
    while ( a2 != v2 )
    {
      v3 = *v2;
      v4 = (const __m128i *)v2;
      v2 += 13;
      if ( *(double *)a1 <= v3 )
      {
        result = sub_C9CB60(v4);
      }
      else
      {
        v5 = (_BYTE *)*((_QWORD *)v2 - 8);
        v6 = (__m128i *)(v2 - 8);
        v7 = *((_QWORD *)v2 - 7);
        v8 = _mm_loadu_si128((const __m128i *)(v2 - 13));
        v24 = *((_QWORD *)v2 - 9);
        v9 = _mm_loadu_si128((const __m128i *)(v2 - 11));
        v25[0] = (__int64)v26;
        v22 = v8;
        v23 = v9;
        sub_C9CAB0(v25, v5, (__int64)&v5[v7]);
        v10 = (_BYTE *)*((_QWORD *)v2 - 4);
        v11 = *((_QWORD *)v2 - 3);
        v27[0] = (__int64)v28;
        sub_C9CAB0(v27, v10, (__int64)&v10[v11]);
        v12 = (__int64)v4->m128i_i64 - a1;
        v13 = 0x4EC4EC4EC4EC4EC5LL * (((__int64)v4->m128i_i64 - a1) >> 3);
        if ( v12 > 0 )
        {
          do
          {
            v14 = v6[-7].m128i_i64[0];
            v15 = _mm_loadu_si128(v6 - 8);
            v16 = v6;
            v6 = (__m128i *)((char *)v6 - 104);
            v17 = _mm_loadu_si128((__m128i *)((char *)v6 - 40));
            v6[6].m128i_i64[0] = v14;
            v6[4] = v17;
            v6[5] = v15;
            sub_2240AE0(v16, v6);
            sub_2240AE0(&v16[2], &v16[-5].m128i_u64[1]);
            --v13;
          }
          while ( v13 );
        }
        v18 = _mm_loadu_si128(&v22);
        v19 = _mm_loadu_si128(&v23);
        *(_QWORD *)(a1 + 32) = v24;
        *(__m128i *)a1 = v18;
        *(__m128i *)(a1 + 16) = v19;
        sub_2240AE0(a1 + 40, v25);
        result = sub_2240AE0(a1 + 72, v27);
        if ( (_QWORD *)v27[0] != v28 )
          result = j_j___libc_free_0(v27[0], v28[0] + 1LL);
        if ( (_QWORD *)v25[0] != v26 )
          result = j_j___libc_free_0(v25[0], v26[0] + 1LL);
      }
    }
  }
  return result;
}
