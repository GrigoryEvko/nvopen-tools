// Function: sub_102D580
// Address: 0x102d580
//
void __fastcall sub_102D580(__m128i **a1, unsigned int a2)
{
  const __m128i *v3; // r13
  __m128i *v4; // r14
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  const __m128i *i; // rsi
  __m128i v9; // xmm0
  unsigned __int64 v10; // rcx
  __m128i *v11; // rdx
  const __m128i *j; // rax
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __m128i *v15; // rax
  const __m128i *v16; // r9
  __m128i v17; // xmm3
  __m128i *v18; // rdi
  __m128i *v19; // rax
  const __m128i *v20; // r9
  _OWORD v21[3]; // [rsp+0h] [rbp-30h] BYREF

  v3 = a1[1];
  v4 = *a1;
  v5 = (char *)v3 - (char *)*a1;
  v6 = (v5 >> 4) - a2;
  if ( v6 != 1 )
  {
    if ( v6 != 2 )
    {
      if ( v6 && v4 != v3 )
      {
        _BitScanReverse64(&v7, a1[1] - *a1);
        sub_1029840(*a1, a1[1], 2LL * (int)(63 - (v7 ^ 0x3F)));
        if ( v5 <= 256 )
        {
          sub_1029CF0(v4, v3);
        }
        else
        {
          sub_1029CF0(v4, v4 + 16);
          for ( i = v4 + 16; i != v3; *v11 = v9 )
          {
            v9 = _mm_loadu_si128(i);
            v10 = i->m128i_i64[0];
            v11 = (__m128i *)i;
            for ( j = i - 1; v10 < j->m128i_i64[0]; j[2] = v13 )
            {
              v13 = _mm_loadu_si128(j);
              v11 = (__m128i *)j--;
            }
            ++i;
          }
        }
      }
      return;
    }
    v14 = _mm_loadu_si128(v3 - 1);
    a1[1] = (__m128i *)&v3[-1];
    v21[0] = v14;
    v15 = (__m128i *)sub_10297E0(v4, (__int64)v3[-2].m128i_i64, (unsigned __int64 *)v21);
    sub_102D4A0((__int64)a1, v15, v16);
    v3 = a1[1];
    v5 = (char *)v3 - (char *)*a1;
  }
  if ( v5 != 16 )
  {
    v17 = _mm_loadu_si128(v3 - 1);
    v18 = *a1;
    a1[1] = (__m128i *)&v3[-1];
    v21[0] = v17;
    v19 = (__m128i *)sub_10297E0(v18, (__int64)v3[-1].m128i_i64, (unsigned __int64 *)v21);
    sub_102D4A0((__int64)a1, v19, v20);
  }
}
