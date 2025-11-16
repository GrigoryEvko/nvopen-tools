// Function: sub_3990CF0
// Address: 0x3990cf0
//
__m128i *__fastcall sub_3990CF0(__int64 a1, const void *a2, __int64 a3, __int64 a4, int a5, int a6)
{
  size_t v6; // r14
  unsigned __int64 v8; // r12
  __int64 v9; // rdi
  __m128i *v10; // r13
  unsigned int v11; // r12d
  __m128i *v12; // rsi
  __int64 v13; // r12
  __m128i *v14; // r14
  unsigned __int64 v15; // rax
  const __m128i *v16; // r12
  const __m128i *v17; // rdi
  size_t v18; // r12
  __m128i *v19; // rax
  __int64 v20; // rdx
  __m128i *v21; // rcx
  __m128i *result; // rax

  v6 = 32 * a3;
  v8 = (32 * a3) >> 5;
  v9 = *(unsigned int *)(a1 + 24);
  if ( v8 > (unsigned __int64)*(unsigned int *)(a1 + 28) - v9 )
  {
    sub_16CD150(a1 + 16, (const void *)(a1 + 32), v8 + v9, 32, a5, a6);
    v9 = *(unsigned int *)(a1 + 24);
  }
  v10 = *(__m128i **)(a1 + 16);
  if ( v6 )
  {
    memcpy(&v10[2 * v9], a2, v6);
    v10 = *(__m128i **)(a1 + 16);
    LODWORD(v9) = *(_DWORD *)(a1 + 24);
  }
  v11 = v9 + v8;
  v12 = v10;
  *(_DWORD *)(a1 + 24) = v11;
  v13 = 2LL * v11;
  v14 = &v10[v13];
  if ( &v10[v13] != v10 )
  {
    _BitScanReverse64(&v15, (v13 * 16) >> 5);
    sub_39908E0(v10, &v10[v13], 2LL * (int)(63 - (v15 ^ 0x3F)), a4);
    if ( (unsigned __int64)v13 <= 32 )
    {
      sub_3985DD0(v10, &v10[v13]);
    }
    else
    {
      v16 = v10 + 32;
      sub_3985DD0(v10, v10 + 32);
      if ( v14 != &v10[32] )
      {
        do
        {
          v17 = v16;
          v16 += 2;
          sub_39856E0(v17);
        }
        while ( v14 != v16 );
        v10 = *(__m128i **)(a1 + 16);
        v12 = &v10[2 * *(unsigned int *)(a1 + 24)];
        goto LABEL_10;
      }
    }
    v10 = *(__m128i **)(a1 + 16);
    v12 = &v10[2 * *(unsigned int *)(a1 + 24)];
  }
LABEL_10:
  v18 = 0;
  v19 = sub_3984B10(v10, v12);
  v20 = *(_QWORD *)(a1 + 16);
  v21 = v19;
  result = (__m128i *)(v20 + 32LL * *(unsigned int *)(a1 + 24));
  if ( result != v12 )
  {
    v18 = v20 + 32LL * *(unsigned int *)(a1 + 24) - (_QWORD)v12;
    result = (__m128i *)memmove(v21, v12, v18);
    v20 = *(_QWORD *)(a1 + 16);
    v21 = result;
  }
  *(_DWORD *)(a1 + 24) = (__int64)((__int64)v21->m128i_i64 + v18 - v20) >> 5;
  return result;
}
