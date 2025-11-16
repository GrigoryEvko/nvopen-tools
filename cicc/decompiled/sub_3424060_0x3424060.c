// Function: sub_3424060
// Address: 0x3424060
//
__int64 __fastcall sub_3424060(__int64 a1, __int64 a2, const __m128i *a3, const __m128i *a4)
{
  const __m128i *v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v8; // rax
  __m128i v9; // xmm0
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 *v13; // rbx
  __int64 v14; // rax
  unsigned __int64 v15; // r13
  __m128i v18; // [rsp+20h] [rbp-70h] BYREF
  __m128i v19; // [rsp+30h] [rbp-60h]
  __int64 v20[2]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v21; // [rsp+50h] [rbp-40h]

  v20[1] = (__int64)v20;
  v20[0] = (__int64)v20;
  v21 = 0;
  if ( a4 == a3 )
    return a2;
  v4 = a3;
  do
  {
    v5 = sub_22077B0(0x98u);
    v6 = v4->m128i_i64[0];
    v7 = v5;
    v18 = _mm_loadu_si128(v4);
    v8 = sub_33ECD10(1u);
    v9 = _mm_load_si128(&v18);
    *(_QWORD *)(v7 + 80) = 0x100000000LL;
    *(_QWORD *)(v7 + 64) = v8;
    *(_QWORD *)(v7 + 104) = 0xFFFFFFFFLL;
    *(_WORD *)(v7 + 48) = 0;
    v10 = v7 + 112;
    *(_QWORD *)(v7 + 144) = 0;
    *(_QWORD *)(v7 + 16) = 0;
    *(_QWORD *)(v7 + 24) = 0;
    *(_QWORD *)(v7 + 32) = 0;
    *(_QWORD *)(v7 + 40) = 328;
    *(_WORD *)(v7 + 50) = -1;
    *(_DWORD *)(v7 + 52) = -1;
    *(_QWORD *)(v7 + 56) = 0;
    *(_QWORD *)(v7 + 72) = 0;
    *(_DWORD *)(v7 + 88) = 0;
    *(_QWORD *)(v7 + 96) = 0;
    *(_QWORD *)(v7 + 136) = 0;
    *(_QWORD *)(v7 + 128) = v7 + 16;
    v19 = v9;
    *(_QWORD *)(v7 + 112) = v9.m128i_i64[0];
    *(_DWORD *)(v7 + 120) = v19.m128i_i32[2];
    v11 = *(_QWORD *)(v6 + 56);
    *(_QWORD *)(v7 + 144) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 24) = v7 + 144;
    *(_QWORD *)(v7 + 136) = v6 + 56;
    ++v4;
    *(_QWORD *)(v6 + 56) = v10;
    *(_DWORD *)(v7 + 80) = 1;
    *(_QWORD *)(v7 + 56) = v10;
    sub_2208C80((_QWORD *)v7, (__int64)v20);
    ++v21;
  }
  while ( a4 != v4 );
  if ( (__int64 *)v20[0] == v20 )
    return a2;
  v12 = v20[0];
  sub_2208C50(a2, v20[0], (__int64)v20);
  v13 = (__int64 *)v20[0];
  v14 = v21;
  v21 = 0;
  *(_QWORD *)(a1 + 16) += v14;
  while ( v13 != v20 )
  {
    v15 = (unsigned __int64)v13;
    v13 = (__int64 *)*v13;
    sub_33CF710(v15 + 16);
    j_j___libc_free_0(v15);
  }
  return v12;
}
