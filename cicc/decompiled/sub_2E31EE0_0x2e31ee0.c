// Function: sub_2E31EE0
// Address: 0x2e31ee0
//
void __fastcall sub_2E31EE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int *v6; // r12
  __int64 v7; // r13
  unsigned __int64 v9; // rax
  int *v10; // rdx
  int *v11; // rdi
  int *v12; // r9
  int *v13; // rax
  int v14; // r8d
  __int64 v15; // rsi
  __int64 v16; // rcx
  const __m128i *v17; // r10
  __m128i v18; // xmm1
  unsigned __int32 v19; // ecx
  __int8 *v20; // rax
  __m128i v21; // xmm0
  __m128i *v22; // rdx
  __m128i v23; // xmm2
  __m128i v24; // [rsp+0h] [rbp-40h] BYREF
  __int64 v25; // [rsp+10h] [rbp-30h]

  v6 = *(unsigned int **)(a1 + 192);
  v7 = *(_QWORD *)(a1 + 184);
  if ( v6 != (unsigned int *)v7 )
  {
    _BitScanReverse64(&v9, 0xAAAAAAAAAAAAAAABLL * (((__int64)v6 - v7) >> 3));
    sub_2E30560(*(unsigned int **)(a1 + 184), *(unsigned int **)(a1 + 192), 2LL * (int)(63 - (v9 ^ 0x3F)), a4, a5, a6);
    if ( (__int64)v6 - v7 > 384 )
    {
      sub_2E2FFA0(v7, (unsigned int *)(v7 + 384));
      for ( ; v6 != (unsigned int *)v17; *(__m128i *)((char *)v22 + 8) = v23 )
      {
        v18 = _mm_loadu_si128(v17);
        v19 = v17->m128i_i32[0];
        v25 = v17[1].m128i_i64[0];
        v20 = &v17[-2].m128i_i8[8];
        v24.m128i_i64[1] = v18.m128i_i64[1];
        if ( v19 >= v17[-2].m128i_i32[2] )
        {
          v22 = (__m128i *)v17;
        }
        else
        {
          do
          {
            v21 = _mm_loadu_si128((const __m128i *)(v20 + 8));
            *((_DWORD *)v20 + 6) = *(_DWORD *)v20;
            v22 = (__m128i *)v20;
            v20 -= 24;
            *(__m128i *)(v20 + 56) = v21;
          }
          while ( v19 < *(_DWORD *)v20 );
        }
        v23 = _mm_loadu_si128((const __m128i *)&v24.m128i_u64[1]);
        v17 = (const __m128i *)((char *)v17 + 24);
        v22->m128i_i32[0] = v19;
      }
    }
    else
    {
      sub_2E2FFA0(v7, v6);
    }
    v10 = *(int **)(a1 + 184);
    v11 = *(int **)(a1 + 192);
    if ( v10 != v11 )
    {
      v12 = *(int **)(a1 + 184);
      do
      {
        v13 = v10 + 6;
        v14 = *v10;
        v15 = *((_QWORD *)v10 + 1);
        v16 = *((_QWORD *)v10 + 2);
        if ( v10 + 6 == v11 )
        {
LABEL_19:
          v10 = v11;
        }
        else
        {
          while ( 1 )
          {
            v10 = v13;
            if ( v14 != *v13 )
              break;
            v15 |= *((_QWORD *)v13 + 1);
            v16 |= *((_QWORD *)v13 + 2);
            v13 += 6;
            if ( v11 == v13 )
              goto LABEL_19;
          }
        }
        *v12 = v14;
        v12 += 6;
        *((_QWORD *)v12 - 2) = v15;
        *((_QWORD *)v12 - 1) = v16;
        v11 = *(int **)(a1 + 192);
      }
      while ( v11 != v10 );
      if ( v12 != v10 )
        *(_QWORD *)(a1 + 192) = v12;
    }
  }
}
