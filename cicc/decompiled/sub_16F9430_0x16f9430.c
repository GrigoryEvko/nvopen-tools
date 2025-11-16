// Function: sub_16F9430
// Address: 0x16f9430
//
__int64 __fastcall sub_16F9430(__int64 a1, char a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __m128i v5; // xmm0
  unsigned __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rdx
  _QWORD *v9; // rdi
  __m128i v11; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v12; // [rsp+18h] [rbp-48h]
  __int64 v13; // [rsp+20h] [rbp-40h]
  _QWORD v14[7]; // [rsp+28h] [rbp-38h] BYREF

  sub_16F91E0(a1, -1);
  *(_BYTE *)(a1 + 73) = 0;
  *(_DWORD *)(a1 + 240) = 0;
  v3 = *(_QWORD *)(a1 + 40);
  v12 = v14;
  v11.m128i_i64[0] = v3;
  v13 = 0;
  LOBYTE(v14[0]) = 0;
  v11.m128i_i64[1] = 3;
  sub_16F7930(a1, 3u);
  v4 = sub_145CBF0((__int64 *)(a1 + 80), 72, 16);
  v5 = _mm_loadu_si128(&v11);
  v6 = v4;
  *(_QWORD *)v4 = 0;
  v7 = v13;
  *(_QWORD *)(v4 + 8) = 0;
  *(__m128i *)(v4 + 24) = v5;
  *(_DWORD *)(v4 + 16) = (a2 == 0) + 5;
  *(_QWORD *)(v4 + 40) = v4 + 56;
  sub_16F6740((__int64 *)(v4 + 40), v14, (__int64)v14 + v7);
  v8 = *(_QWORD *)(a1 + 184);
  *(_QWORD *)(v6 + 8) = a1 + 184;
  v8 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v6 = v8 | *(_QWORD *)v6 & 7LL;
  *(_QWORD *)(v8 + 8) = v6;
  v9 = v12;
  *(_QWORD *)(a1 + 184) = *(_QWORD *)(a1 + 184) & 7LL | v6;
  if ( v9 != v14 )
    j_j___libc_free_0(v9, v14[0] + 1LL);
  return 1;
}
