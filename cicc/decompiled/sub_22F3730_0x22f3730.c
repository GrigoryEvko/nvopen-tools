// Function: sub_22F3730
// Address: 0x22f3730
//
__int64 __fastcall sub_22F3730(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v3; // r14
  unsigned int v5; // r13d
  __m128i *v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __m128i *v10; // rdi
  __int64 v11; // r12
  __int64 v12; // rax
  __m128i *v14; // [rsp+0h] [rbp-40h] BYREF
  __int64 v15; // [rsp+8h] [rbp-38h]
  __m128i v16[3]; // [rsp+10h] [rbp-30h] BYREF

  v3 = a1 + 328;
  v5 = *(_DWORD *)(a1 + 192);
  v14 = v16;
  sub_22F3320((__int64 *)&v14, a2, (__int64)&a2[a3]);
  v6 = (__m128i *)sub_22077B0(0x30u);
  v6[1].m128i_i64[0] = (__int64)v6[2].m128i_i64;
  if ( v14 == v16 )
  {
    v6[2] = _mm_load_si128(v16);
  }
  else
  {
    v6[1].m128i_i64[0] = (__int64)v14;
    v6[2].m128i_i64[0] = v16[0].m128i_i64[0];
  }
  v7 = v15;
  v14 = v16;
  v15 = 0;
  v6[1].m128i_i64[1] = v7;
  v16[0].m128i_i8[0] = 0;
  sub_2208C80(v6, v3);
  v10 = v14;
  ++*(_QWORD *)(a1 + 344);
  if ( v10 != v16 )
    j_j___libc_free_0((unsigned __int64)v10);
  v11 = *(_QWORD *)(*(_QWORD *)(a1 + 336) + 16LL);
  v12 = *(unsigned int *)(a1 + 192);
  if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 196) )
  {
    sub_C8D5F0(a1 + 184, (const void *)(a1 + 200), v12 + 1, 8u, v8, v9);
    v12 = *(unsigned int *)(a1 + 192);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8 * v12) = v11;
  ++*(_DWORD *)(a1 + 192);
  return v5;
}
