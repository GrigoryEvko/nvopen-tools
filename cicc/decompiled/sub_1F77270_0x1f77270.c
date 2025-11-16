// Function: sub_1F77270
// Address: 0x1f77270
//
__int64 *__fastcall sub_1F77270(__int64 **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v5; // rax
  bool v6; // zf
  unsigned __int16 v7; // r14
  unsigned __int8 v8; // r13
  const void **v9; // rax
  __int64 *v10; // rcx
  __int64 v12; // rax
  __int64 *result; // rax
  __int64 *v14; // rax
  __int64 v15; // r15
  const __m128i *v16; // rdx
  __m128 v17; // xmm0
  unsigned __int8 *v18; // rax
  __int64 v19; // rax
  char v20; // al
  __int64 v21; // rdx
  int v22; // edx
  __int64 v23; // rcx
  __m128i v24; // xmm1
  __int128 v25; // rax
  __int64 *v26; // rdi
  unsigned int v27; // edx
  unsigned int v28; // edx
  unsigned int v29; // edx
  __int64 v30; // [rsp+18h] [rbp-E8h]
  __int64 v31; // [rsp+20h] [rbp-E0h]
  __int128 v32; // [rsp+20h] [rbp-E0h]
  __int64 *v33; // [rsp+30h] [rbp-D0h]
  __int64 (__fastcall *v34)(__int64 *, __int64, __int64, __int64, __int64); // [rsp+40h] [rbp-C0h]
  __int64 v35; // [rsp+40h] [rbp-C0h]
  __int64 v36; // [rsp+50h] [rbp-B0h]
  __int128 v37; // [rsp+50h] [rbp-B0h]
  __int64 *v38; // [rsp+50h] [rbp-B0h]
  unsigned int v39; // [rsp+A0h] [rbp-60h]
  const void **v40; // [rsp+A8h] [rbp-58h]
  char v41[8]; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v42; // [rsp+B8h] [rbp-48h]
  __int64 v43; // [rsp+C0h] [rbp-40h] BYREF
  int v44; // [rsp+C8h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 40);
  v6 = *((_BYTE *)a1 + 24) == 0;
  v7 = *(_WORD *)(a2 + 24);
  v8 = *(_BYTE *)v5;
  v9 = *(const void ***)(v5 + 8);
  LOBYTE(v39) = v8;
  v40 = v9;
  if ( !v6 )
    return 0;
  v10 = a1[1];
  v12 = 1;
  if ( v8 != 1 )
  {
    if ( !v8 )
      return 0;
    v12 = v8;
    if ( !v10[v8 + 15] )
      return 0;
  }
  if ( (*((_BYTE *)v10 + 259 * v12 + 2557) & 0xFB) != 0 )
    return 0;
  v14 = *(__int64 **)(a2 + 32);
  v15 = *v14;
  if ( *(_WORD *)(*v14 + 24) != 135 )
    return 0;
  if ( !sub_1D18C00(*v14, 1, *((_DWORD *)v14 + 2)) )
    return 0;
  v16 = *(const __m128i **)(v15 + 32);
  if ( *(_WORD *)(v16->m128i_i64[0] + 24) != 137 )
    return 0;
  v17 = (__m128)_mm_loadu_si128(v16);
  v33 = a1[1];
  v18 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(v16->m128i_i64[0] + 32) + 40LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(v16->m128i_i64[0] + 32) + 8LL));
  v36 = (*a1)[6];
  v30 = *((_QWORD *)v18 + 1);
  v31 = *v18;
  v34 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64))(*v33 + 264);
  v19 = sub_1E0A0C0((*a1)[4]);
  v20 = v34(v33, v19, v36, v31, v30);
  v41[0] = v20;
  v42 = v21;
  v22 = v20 ? sub_1F6C8D0(v20) : sub_1F58D40((__int64)v41);
  if ( (unsigned int)sub_1F6C8D0(v8) != v22 )
    return 0;
  v23 = *(_QWORD *)(v15 + 32);
  v24 = _mm_loadu_si128((const __m128i *)(v23 + 80));
  v25 = *(_OWORD *)(v23 + 40);
  v43 = *(_QWORD *)(a2 + 72);
  if ( v43 )
  {
    v32 = v25;
    sub_1F6CA20(&v43);
    v25 = v32;
  }
  v26 = *a1;
  v44 = *(_DWORD *)(a2 + 64);
  if ( v7 == 154 )
  {
    *(_QWORD *)&v37 = sub_1D332F0(
                        v26,
                        154,
                        (__int64)&v43,
                        v39,
                        v40,
                        0,
                        *(double *)v17.m128_u64,
                        *(double *)v24.m128i_i64,
                        a5,
                        v25,
                        *((unsigned __int64 *)&v25 + 1),
                        *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
    *((_QWORD *)&v37 + 1) = v29;
    v35 = (__int64)sub_1D332F0(
                     *a1,
                     154,
                     (__int64)&v43,
                     v39,
                     v40,
                     0,
                     *(double *)v17.m128_u64,
                     *(double *)v24.m128i_i64,
                     a5,
                     v24.m128i_i64[0],
                     v24.m128i_u64[1],
                     *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
  }
  else
  {
    *(_QWORD *)&v37 = sub_1D309E0(
                        v26,
                        v7,
                        (__int64)&v43,
                        v39,
                        v40,
                        0,
                        *(double *)v17.m128_u64,
                        *(double *)v24.m128i_i64,
                        *(double *)a5.m128i_i64,
                        v25);
    *((_QWORD *)&v37 + 1) = v27;
    v35 = sub_1D309E0(
            *a1,
            v7,
            (__int64)&v43,
            v39,
            v40,
            0,
            *(double *)v17.m128_u64,
            *(double *)v24.m128i_i64,
            *(double *)a5.m128i_i64,
            *(_OWORD *)&v24);
  }
  result = sub_1D3A900(
             *a1,
             0x87u,
             (__int64)&v43,
             v39,
             v40,
             0,
             v17,
             *(double *)v24.m128i_i64,
             a5,
             v17.m128_u64[0],
             (__int16 *)v17.m128_u64[1],
             v37,
             v35,
             v28);
  if ( v43 )
  {
    v38 = result;
    sub_161E7C0((__int64)&v43, v43);
    return v38;
  }
  return result;
}
