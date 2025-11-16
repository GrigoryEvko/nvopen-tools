// Function: sub_1D38FD0
// Address: 0x1d38fd0
//
__int64 *__fastcall sub_1D38FD0(
        __int64 *a1,
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
  unsigned int v16; // edx
  unsigned int v17; // edx
  const __m128i *v18; // r8
  const __m128i *v19; // r9
  __int64 v20; // rax
  const __m128i **v21; // rax
  __int64 v22; // rax
  __int64 m128i_i64; // r14
  __int64 v24; // rax
  _QWORD *v25; // rax
  unsigned __int64 v26; // r14
  __int64 v27; // r15
  unsigned __int8 *v28; // rsi
  __int64 v29; // rdx
  const void ***v30; // rax
  int v31; // edx
  __int64 v32; // r9
  __int64 *v33; // r12
  __int128 v35; // [rsp-8h] [rbp-F0h]
  const __m128i *v36; // [rsp+10h] [rbp-D8h]
  const __m128i *v37; // [rsp+18h] [rbp-D0h]
  const __m128i *v38; // [rsp+18h] [rbp-D0h]
  _QWORD *v39; // [rsp+28h] [rbp-C0h] BYREF
  __int64 v40; // [rsp+30h] [rbp-B8h]
  _QWORD v41[2]; // [rsp+38h] [rbp-B0h] BYREF
  __m128i v42; // [rsp+48h] [rbp-A0h]
  __m128i v43; // [rsp+58h] [rbp-90h]
  __int64 v44; // [rsp+68h] [rbp-80h]
  __int64 v45; // [rsp+70h] [rbp-78h]

  v13 = *(_DWORD *)(a2 + 8);
  v14 = _mm_loadu_si128((const __m128i *)&a8);
  v15 = _mm_loadu_si128((const __m128i *)&a9);
  v39 = v41;
  v41[0] = a4;
  v41[1] = a5;
  v40 = 0x800000004LL;
  v42 = v14;
  v43 = v15;
  v44 = sub_1D38BB0((__int64)a1, a6, a3, 5, 0, 1, a7, *(double *)v14.m128i_i64, v15, 0);
  v45 = v16;
  v18 = (const __m128i *)sub_1D38BB0((__int64)a1, a10, a3, 5, 0, 1, a7, *(double *)v14.m128i_i64, v15, 0);
  v19 = (const __m128i *)v17;
  v20 = (unsigned int)v40;
  if ( (unsigned int)v40 >= HIDWORD(v40) )
  {
    v36 = v18;
    v38 = (const __m128i *)v17;
    sub_16CD150((__int64)&v39, v41, 0, 16, (int)v18, v17);
    v20 = (unsigned int)v40;
    v18 = v36;
    v19 = v38;
  }
  v21 = (const __m128i **)&v39[2 * v20];
  *v21 = v18;
  v21[1] = v19;
  v22 = (unsigned int)(v40 + 1);
  LODWORD(v40) = v40 + 1;
  if ( v13 )
  {
    v18 = a11;
    m128i_i64 = (__int64)a11[(unsigned int)(v13 - 1) + 1].m128i_i64;
    do
    {
      if ( (unsigned int)v22 >= HIDWORD(v40) )
      {
        v37 = v18;
        sub_16CD150((__int64)&v39, v41, 0, 16, (int)v18, (int)v19);
        v22 = (unsigned int)v40;
        v18 = v37;
      }
      a7 = _mm_loadu_si128(v18++);
      *(__m128i *)&v39[2 * v22] = a7;
      v22 = (unsigned int)(v40 + 1);
      LODWORD(v40) = v40 + 1;
    }
    while ( (const __m128i *)m128i_i64 != v18 );
  }
  v24 = *(unsigned int *)(a2 + 8);
  if ( (unsigned int)v24 >= *(_DWORD *)(a2 + 12) )
  {
    sub_16CD150(a2, (const void *)(a2 + 16), 0, 16, (int)v18, (int)v19);
    v24 = *(unsigned int *)(a2 + 8);
  }
  v25 = (_QWORD *)(*(_QWORD *)a2 + 16 * v24);
  *v25 = 1;
  v26 = (unsigned __int64)v39;
  v25[1] = 0;
  v27 = (unsigned int)v40;
  v28 = *(unsigned __int8 **)a2;
  v29 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  *(_DWORD *)(a2 + 8) = v29;
  v30 = (const void ***)sub_1D25C30((__int64)a1, v28, v29);
  *((_QWORD *)&v35 + 1) = v27;
  *(_QWORD *)&v35 = v26;
  v33 = sub_1D36D80(a1, 204, a3, v30, v31, *(double *)a7.m128i_i64, *(double *)v14.m128i_i64, v15, v32, v35);
  if ( v39 != v41 )
    _libc_free((unsigned __int64)v39);
  return v33;
}
