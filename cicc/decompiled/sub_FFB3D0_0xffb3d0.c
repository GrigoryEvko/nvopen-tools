// Function: sub_FFB3D0
// Address: 0xffb3d0
//
void __fastcall sub_FFB3D0(__int64 a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  __int64 v9; // rdi
  unsigned __int64 v10; // rdx
  const __m128i *v11; // r13
  const __m128i *v12; // rbx
  __int64 v13; // rax
  __m128i v14; // xmm0
  const __m128i *v15; // r14
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // r8
  __m128i *v19; // rax
  const void *v20; // rsi
  _BYTE *v21; // r14
  _BYTE v22[8]; // [rsp+0h] [rbp-50h] BYREF
  __m128i v23; // [rsp+8h] [rbp-48h]

  v8 = *(_QWORD *)(a1 + 544);
  if ( v8 )
  {
    if ( *(_BYTE *)(a1 + 560) != 1 )
    {
      sub_B26780(v8, a2, a3);
      v9 = *(_QWORD *)(a1 + 552);
      if ( !v9 )
        return;
LABEL_7:
      sub_B2A430(v9, a2, a3);
      return;
    }
  }
  else
  {
    v9 = *(_QWORD *)(a1 + 552);
    if ( !v9 )
      return;
    if ( *(_BYTE *)(a1 + 560) != 1 )
      goto LABEL_7;
  }
  v10 = a3 + *(unsigned int *)(a1 + 8);
  if ( v10 > *(unsigned int *)(a1 + 12) )
    sub_C8D5F0(a1, (const void *)(a1 + 16), v10, 0x20u, a5, a6);
  v11 = (const __m128i *)&a2[2 * a3];
  if ( v11 != (const __m128i *)a2 )
  {
    v12 = (const __m128i *)a2;
    do
    {
      if ( v12->m128i_i64[0] != (v12->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v13 = *(unsigned int *)(a1 + 8);
        v14 = _mm_loadu_si128(v12);
        v22[0] = 0;
        v15 = (const __m128i *)v22;
        v16 = *(unsigned int *)(a1 + 12);
        v17 = *(_QWORD *)a1;
        v18 = v13 + 1;
        v23 = v14;
        if ( v13 + 1 > v16 )
        {
          v20 = (const void *)(a1 + 16);
          if ( v17 > (unsigned __int64)v22 || (unsigned __int64)v22 >= v17 + 32 * v13 )
          {
            sub_C8D5F0(a1, v20, v18, 0x20u, v18, a6);
            v17 = *(_QWORD *)a1;
            v13 = *(unsigned int *)(a1 + 8);
          }
          else
          {
            v21 = &v22[-v17];
            sub_C8D5F0(a1, v20, v18, 0x20u, v18, a6);
            v17 = *(_QWORD *)a1;
            v13 = *(unsigned int *)(a1 + 8);
            v15 = (const __m128i *)&v21[*(_QWORD *)a1];
          }
        }
        v19 = (__m128i *)(v17 + 32 * v13);
        *v19 = _mm_loadu_si128(v15);
        v19[1] = _mm_loadu_si128(v15 + 1);
        ++*(_DWORD *)(a1 + 8);
      }
      ++v12;
    }
    while ( v11 != v12 );
  }
}
