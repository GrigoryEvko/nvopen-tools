// Function: sub_1A1C330
// Address: 0x1a1c330
//
__int64 __fastcall sub_1A1C330(__int64 a1, const void *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const void *v6; // r8
  size_t v7; // r15
  unsigned __int64 v9; // rbx
  unsigned int v10; // r13d
  __int64 v11; // rdx
  __int64 v12; // r14
  unsigned int v13; // ebx
  __int64 result; // rax
  __int64 v15; // rdx
  __m128i *v16; // r13
  __m128i *v17; // r14
  __int64 v18; // rbx
  __m128i *v19; // r14
  __int64 v20; // rdx
  __int64 v21; // rcx
  unsigned __int64 v22; // r15
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // rbx
  __m128i *v26; // rax
  __m128i *v27; // r12
  __m128i *v28; // rsi
  __m128i *v29; // rax
  __m128i v30; // xmm0
  __int64 v31; // rdi
  const __m128i *v32; // rax
  __m128i *v33; // rbx
  unsigned __int64 *v34; // rdi
  __int64 v35; // [rsp+0h] [rbp-50h]
  __int64 v36; // [rsp+8h] [rbp-48h]
  __int64 v37; // [rsp+10h] [rbp-40h]
  __int64 v38; // [rsp+18h] [rbp-38h]

  v6 = a2;
  v7 = 24 * a3;
  v9 = 0xAAAAAAAAAAAAAAABLL * ((24 * a3) >> 3);
  v10 = *(_DWORD *)(a1 + 16);
  v11 = v10;
  if ( v9 > *(unsigned int *)(a1 + 20) - (unsigned __int64)v10 )
  {
    sub_16CD150(a1 + 8, (const void *)(a1 + 24), v9 + v10, 24, (int)a2, a6);
    v11 = *(unsigned int *)(a1 + 16);
    v6 = a2;
  }
  v12 = *(_QWORD *)(a1 + 8);
  if ( v7 )
  {
    memcpy((void *)(v12 + 24 * v11), v6, v7);
    v12 = *(_QWORD *)(a1 + 8);
    LODWORD(v11) = *(_DWORD *)(a1 + 16);
  }
  v13 = v11 + v9;
  *(_DWORD *)(a1 + 16) = v13;
  result = 24LL * (int)v10;
  v15 = 24LL * v13;
  v16 = (__m128i *)(v12 + result);
  v17 = (__m128i *)(v15 + v12);
  if ( v16 != v17 )
  {
    v18 = v15 - result;
    _BitScanReverse64((unsigned __int64 *)&result, 0xAAAAAAAAAAAAAAABLL * ((v15 - result) >> 3));
    sub_1A1B940((__int64)v16, v17, 2LL * (int)(63 - (result ^ 0x3F)), a4, (__int64)v6, a6);
    if ( v18 > 384 )
    {
      v33 = v16 + 24;
      sub_1A1AE20(v16->m128i_i64, v16[24].m128i_i64);
      if ( v17 != &v16[24] )
      {
        do
        {
          v34 = (unsigned __int64 *)v33;
          v33 = (__m128i *)((char *)v33 + 24);
          sub_1A1AC50(v34);
        }
        while ( v17 != v33 );
      }
    }
    else
    {
      sub_1A1AE20(v16->m128i_i64, v17->m128i_i64);
    }
    v19 = *(__m128i **)(a1 + 8);
    result = 3LL * *(unsigned int *)(a1 + 16);
    v20 = (__int64)&v19->m128i_i64[3 * *(unsigned int *)(a1 + 16)];
    if ( v19 != v16 && v16 != (__m128i *)v20 )
    {
      v21 = 0x555555555555555LL;
      v22 = 0xAAAAAAAAAAAAAAABLL * (((char *)v16 - (char *)v19) >> 3);
      v23 = 0xAAAAAAAAAAAAAAABLL * ((v20 - (__int64)v16) >> 3);
      v24 = v22 - 0x5555555555555555LL * ((v20 - (__int64)v16) >> 3);
      if ( v24 <= 0x555555555555555LL )
        v21 = v22 - 0x5555555555555555LL * ((v20 - (__int64)v16) >> 3);
      if ( v24 <= 0 )
      {
LABEL_24:
        v25 = 0;
        sub_1A1B460(v19->m128i_i8, v16, v20, 0xAAAAAAAAAAAAAAABLL * (((char *)v16 - (char *)v19) >> 3), v23);
        v27 = 0;
      }
      else
      {
        while ( 1 )
        {
          v35 = v23;
          v36 = v20;
          v37 = v21;
          v25 = 24 * v21;
          v38 = 24 * v21;
          v26 = (__m128i *)sub_2207800(24 * v21, &unk_435FF63);
          v20 = v36;
          v23 = v35;
          v27 = v26;
          if ( v26 )
            break;
          v21 = v37 >> 1;
          if ( !(v37 >> 1) )
            goto LABEL_24;
        }
        v28 = (__m128i *)((char *)v26 + v25);
        *v26 = _mm_loadu_si128(v19);
        v26[1].m128i_i64[0] = v19[1].m128i_i64[0];
        v29 = (__m128i *)((char *)v26 + 24);
        if ( v28 == (__m128i *)&v27[1].m128i_u64[1] )
        {
          v32 = v27;
        }
        else
        {
          do
          {
            v30 = _mm_loadu_si128((__m128i *)((char *)v29 - 24));
            v31 = v29[-1].m128i_i64[1];
            v29 = (__m128i *)((char *)v29 + 24);
            *(__m128i *)((char *)v29 - 24) = v30;
            v29[-1].m128i_i64[1] = v31;
          }
          while ( v28 != v29 );
          v32 = (__m128i *)((char *)v27 + v38 - 24);
        }
        *v19 = _mm_loadu_si128(v32);
        v19[1].m128i_i64[0] = v32[1].m128i_i64[0];
        sub_1A1BE10(
          v19,
          v16,
          v36,
          0xAAAAAAAAAAAAAAABLL * (((char *)v16 - (char *)v19) >> 3),
          v35,
          v27,
          (const __m128i *)v37);
      }
      return j_j___libc_free_0(v27, v25);
    }
  }
  return result;
}
