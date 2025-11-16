// Function: sub_1414480
// Address: 0x1414480
//
void __fastcall sub_1414480(__m128i **a1, unsigned int a2)
{
  unsigned __int64 *v3; // r13
  __m128i *v4; // r14
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __m128i *i; // rsi
  unsigned __int64 v9; // rcx
  __int64 v10; // rdi
  __int64 *v11; // rdx
  const __m128i *j; // rax
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  __m128i *v15; // rax
  const __m128i *v16; // r9
  __m128i v17; // xmm2
  __m128i *v18; // rdi
  __m128i *v19; // rax
  const __m128i *v20; // r9
  _OWORD v21[3]; // [rsp+0h] [rbp-30h] BYREF

  v3 = (unsigned __int64 *)a1[1];
  v4 = *a1;
  v5 = (char *)v3 - (char *)*a1;
  v6 = (v5 >> 4) - a2;
  if ( v6 != 1 )
  {
    if ( v6 != 2 )
    {
      if ( v6 && v4 != (__m128i *)v3 )
      {
        _BitScanReverse64(&v7, a1[1] - *a1);
        sub_1411980(*a1, a1[1], 2LL * (int)(63 - (v7 ^ 0x3F)));
        if ( v5 <= 256 )
        {
          sub_1411BF0((unsigned __int64 *)v4, v3);
        }
        else
        {
          sub_1411BF0((unsigned __int64 *)v4, (unsigned __int64 *)&v4[16]);
          for ( i = v4 + 16; i != (__m128i *)v3; v11[1] = v10 )
          {
            v9 = i->m128i_i64[0];
            v10 = i->m128i_i64[1];
            v11 = (__int64 *)i;
            for ( j = i - 1; v9 < j->m128i_i64[0]; j[2] = v13 )
            {
              v13 = _mm_loadu_si128(j);
              v11 = (__int64 *)j--;
            }
            ++i;
            *v11 = v9;
          }
        }
      }
      return;
    }
    v14 = _mm_loadu_si128((const __m128i *)v3 - 1);
    a1[1] = (__m128i *)(v3 - 2);
    v21[0] = v14;
    v15 = (__m128i *)sub_1411920(v4, (__int64)(v3 - 4), (unsigned __int64 *)v21);
    sub_14143A0((__int64)a1, v15, v16);
    v3 = (unsigned __int64 *)a1[1];
    v5 = (char *)v3 - (char *)*a1;
  }
  if ( v5 != 16 )
  {
    v17 = _mm_loadu_si128((const __m128i *)v3 - 1);
    v18 = *a1;
    a1[1] = (__m128i *)(v3 - 2);
    v21[0] = v17;
    v19 = (__m128i *)sub_1411920(v18, (__int64)(v3 - 2), (unsigned __int64 *)v21);
    sub_14143A0((__int64)a1, v19, v20);
  }
}
