// Function: sub_34118E0
// Address: 0x34118e0
//
unsigned __int8 *__fastcall sub_34118E0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        __m128i a7,
        __int128 a8,
        __int128 a9,
        unsigned int a10,
        const __m128i *a11)
{
  int v13; // r14d
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  __int64 v16; // rdx
  const __m128i *v17; // r8
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r9
  unsigned __int64 v21; // rdx
  const __m128i **v22; // rax
  __int64 v23; // rax
  __int64 m128i_i64; // r14
  __m128i v25; // xmm0
  __int64 v26; // rax
  _QWORD *v27; // rax
  unsigned __int64 v28; // r14
  __int64 v29; // r15
  unsigned __int16 *v30; // rsi
  __int64 v31; // rdx
  unsigned int *v32; // rax
  __int64 v33; // rdx
  __int64 v34; // r9
  unsigned __int8 *v35; // r12
  __int128 v37; // [rsp-8h] [rbp-F0h]
  __m128i v38; // [rsp+8h] [rbp-E0h] BYREF
  const __m128i *v39; // [rsp+18h] [rbp-D0h]
  _QWORD *v40; // [rsp+20h] [rbp-C8h]
  _QWORD *v41; // [rsp+28h] [rbp-C0h] BYREF
  __int64 v42; // [rsp+30h] [rbp-B8h]
  _QWORD v43[2]; // [rsp+38h] [rbp-B0h] BYREF
  __m128i v44; // [rsp+48h] [rbp-A0h]
  __m128i v45; // [rsp+58h] [rbp-90h]
  unsigned __int8 *v46; // [rsp+68h] [rbp-80h]
  __int64 v47; // [rsp+70h] [rbp-78h]

  v13 = *(_DWORD *)(a2 + 8);
  v14 = _mm_loadu_si128((const __m128i *)&a8);
  v40 = v43;
  v15 = _mm_loadu_si128((const __m128i *)&a9);
  v41 = v43;
  v43[0] = a4;
  v43[1] = a5;
  v42 = 0x800000004LL;
  v44 = v14;
  v45 = v15;
  v46 = sub_3400BD0((__int64)a1, a6, a3, 7, 0, 1u, a7, 0);
  v47 = v16;
  v17 = (const __m128i *)sub_3400BD0((__int64)a1, a10, a3, 7, 0, 1u, a7, 0);
  v18 = (unsigned int)v42;
  v20 = v19;
  v21 = (unsigned int)v42 + 1LL;
  if ( v21 > HIDWORD(v42) )
  {
    v38.m128i_i64[0] = (__int64)v17;
    v38.m128i_i64[1] = v20;
    sub_C8D5F0((__int64)&v41, v40, v21, 0x10u, (__int64)v17, v20);
    v18 = (unsigned int)v42;
    v20 = v38.m128i_i64[1];
    v17 = (const __m128i *)v38.m128i_i64[0];
  }
  v22 = (const __m128i **)&v41[2 * v18];
  *v22 = v17;
  v22[1] = (const __m128i *)v20;
  v23 = (unsigned int)(v42 + 1);
  LODWORD(v42) = v42 + 1;
  if ( v13 )
  {
    v17 = a11;
    m128i_i64 = (__int64)a11[(unsigned int)(v13 - 1) + 1].m128i_i64;
    do
    {
      v25 = _mm_loadu_si128(v17);
      if ( v23 + 1 > (unsigned __int64)HIDWORD(v42) )
      {
        v39 = v17;
        v38 = v25;
        sub_C8D5F0((__int64)&v41, v40, v23 + 1, 0x10u, (__int64)v17, v20);
        v23 = (unsigned int)v42;
        v17 = v39;
        v25 = _mm_load_si128(&v38);
      }
      ++v17;
      *(__m128i *)&v41[2 * v23] = v25;
      v23 = (unsigned int)(v42 + 1);
      LODWORD(v42) = v42 + 1;
    }
    while ( v17 != (const __m128i *)m128i_i64 );
  }
  v26 = *(unsigned int *)(a2 + 8);
  if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), v26 + 1, 0x10u, (__int64)v17, v20);
    v26 = *(unsigned int *)(a2 + 8);
  }
  v27 = (_QWORD *)(*(_QWORD *)a2 + 16 * v26);
  *v27 = 1;
  v28 = (unsigned __int64)v41;
  v27[1] = 0;
  v29 = (unsigned int)v42;
  v30 = *(unsigned __int16 **)a2;
  v31 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  *(_DWORD *)(a2 + 8) = v31;
  v32 = (unsigned int *)sub_33E5830(a1, v30, v31);
  *((_QWORD *)&v37 + 1) = v29;
  *(_QWORD *)&v37 = v28;
  v35 = sub_3411630(a1, 317, a3, v32, v33, v34, v37);
  if ( v41 != v40 )
    _libc_free((unsigned __int64)v41);
  return v35;
}
