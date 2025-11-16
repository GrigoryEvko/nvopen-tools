// Function: sub_15A2780
// Address: 0x15a2780
//
__int64 __fastcall sub_15A2780(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9)
{
  __m128i v9; // xmm4
  __m128i v10; // xmm5
  __m128i v11; // xmm7
  __m128i v12; // xmm6
  __int64 v13; // r13
  char v14; // al
  __int64 *v15; // rdx
  unsigned int v17; // esi
  int v18; // eax
  int v19; // eax
  __int64 *v20; // [rsp+8h] [rbp-E8h] BYREF
  _OWORD v21[3]; // [rsp+10h] [rbp-E0h] BYREF
  _BYTE v22[24]; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v23; // [rsp+58h] [rbp-98h] BYREF
  __m128i v24; // [rsp+68h] [rbp-88h]
  unsigned __int64 v25; // [rsp+80h] [rbp-70h] BYREF
  __m128i v26; // [rsp+88h] [rbp-68h]
  __m128i v27; // [rsp+98h] [rbp-58h]
  __m128i v28; // [rsp+A8h] [rbp-48h]
  __int64 v29; // [rsp+B8h] [rbp-38h]

  *(_QWORD *)v22 = a2;
  v23 = _mm_loadu_si128((const __m128i *)&a8);
  *(__m128i *)&v22[8] = _mm_loadu_si128((const __m128i *)&a7);
  v24 = _mm_loadu_si128((const __m128i *)&a9);
  v25 = sub_1597510((__int64 *)v23.m128i_i64[1], v23.m128i_i64[1] + 4 * a9);
  *(_QWORD *)&v21[0] = sub_1597240(*(__int64 **)&v22[16], *(_QWORD *)&v22[16] + 8 * v23.m128i_i64[0]);
  LODWORD(v25) = sub_1597150(&v22[8], &v22[9], (__int16 *)&v22[10], (__int64 *)v21, (__int64 *)&v25);
  LODWORD(v25) = sub_15981B0((__int64 *)v22, (int *)&v25);
  v9 = _mm_loadu_si128((const __m128i *)&v22[16]);
  v10 = _mm_loadu_si128((const __m128i *)&v23.m128i_u64[1]);
  v26 = _mm_loadu_si128((const __m128i *)v22);
  v27 = v9;
  v29 = v24.m128i_i64[1];
  v28 = v10;
  if ( (unsigned __int8)sub_1598AB0(a1, (__int64)&v25, (__int64 **)v21)
    && *(_QWORD *)&v21[0] != *(_QWORD *)(a1 + 8) + 8LL * *(unsigned int *)(a1 + 24) )
  {
    return **(_QWORD **)&v21[0];
  }
  v11 = _mm_loadu_si128((const __m128i *)&a8);
  v21[0] = _mm_loadu_si128((const __m128i *)&a7);
  v12 = _mm_loadu_si128((const __m128i *)&a9);
  v21[1] = v11;
  v21[2] = v12;
  v13 = sub_1595240((unsigned __int8 *)v21, a2);
  v14 = sub_1598AB0(a1, (__int64)&v25, &v20);
  v15 = v20;
  if ( !v14 )
  {
    v17 = *(_DWORD *)(a1 + 24);
    v18 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v19 = v18 + 1;
    if ( 4 * v19 >= 3 * v17 )
    {
      v17 *= 2;
    }
    else if ( v17 - *(_DWORD *)(a1 + 20) - v19 > v17 >> 3 )
    {
LABEL_8:
      *(_DWORD *)(a1 + 16) = v19;
      if ( *v15 != -8 )
        --*(_DWORD *)(a1 + 20);
      *v15 = v13;
      return v13;
    }
    sub_15A25C0(a1, v17);
    sub_1598AB0(a1, (__int64)&v25, &v20);
    v15 = v20;
    v19 = *(_DWORD *)(a1 + 16) + 1;
    goto LABEL_8;
  }
  return v13;
}
