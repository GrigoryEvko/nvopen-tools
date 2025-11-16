// Function: sub_16D63F0
// Address: 0x16d63f0
//
__int64 __fastcall sub_16D63F0(__int64 a1, double *a2)
{
  double *v2; // r12
  double v3; // xmm0_8
  const __m128i *v4; // rbx
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  __m128i *v7; // r14
  __m128i v8; // xmm4
  __m128i v9; // xmm5
  _BYTE *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rdx
  unsigned __int64 v13; // rbx
  __m128i v14; // xmm2
  __m128i v15; // xmm3
  __m128i *v16; // r15
  __m128i v17; // xmm7
  __int64 result; // rax
  __m128i v20; // [rsp+30h] [rbp-90h] BYREF
  __m128i v21; // [rsp+40h] [rbp-80h] BYREF
  __int64 v22[2]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v23[2]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v24[2]; // [rsp+70h] [rbp-50h] BYREF
  _QWORD v25[8]; // [rsp+80h] [rbp-40h] BYREF

  if ( (double *)a1 != a2 )
  {
    v2 = (double *)(a1 + 96);
    while ( a2 != v2 )
    {
      v3 = *v2;
      v4 = (const __m128i *)v2;
      v2 += 12;
      if ( *(double *)a1 <= v3 )
      {
        result = sub_16D62A0(v4);
      }
      else
      {
        v5 = (_BYTE *)*((_QWORD *)v2 - 8);
        v6 = *((_QWORD *)v2 - 7);
        v7 = (__m128i *)(v2 - 8);
        v8 = _mm_loadu_si128((const __m128i *)v2 - 6);
        v9 = _mm_loadu_si128((const __m128i *)v2 - 5);
        v22[0] = (__int64)v23;
        v20 = v8;
        v21 = v9;
        sub_16D5EB0(v22, v5, (__int64)&v5[v6]);
        v10 = (_BYTE *)*((_QWORD *)v2 - 4);
        v11 = *((_QWORD *)v2 - 3);
        v24[0] = (__int64)v25;
        sub_16D5EB0(v24, v10, (__int64)&v10[v11]);
        v12 = (__int64)v4->m128i_i64 - a1;
        v13 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v4->m128i_i64 - a1) >> 5);
        if ( v12 > 0 )
        {
          do
          {
            v14 = _mm_loadu_si128(v7 - 8);
            v15 = _mm_loadu_si128(v7 - 7);
            v16 = v7;
            v7 -= 6;
            v7[4] = v14;
            v7[5] = v15;
            sub_2240AE0(v16, v7);
            sub_2240AE0(&v16[2], &v16[-4]);
            --v13;
          }
          while ( v13 );
        }
        v17 = _mm_loadu_si128(&v21);
        *(__m128i *)a1 = _mm_loadu_si128(&v20);
        *(__m128i *)(a1 + 16) = v17;
        sub_2240AE0(a1 + 32, v22);
        result = sub_2240AE0(a1 + 64, v24);
        if ( (_QWORD *)v24[0] != v25 )
          result = j_j___libc_free_0(v24[0], v25[0] + 1LL);
        if ( (_QWORD *)v22[0] != v23 )
          result = j_j___libc_free_0(v22[0], v23[0] + 1LL);
      }
    }
  }
  return result;
}
