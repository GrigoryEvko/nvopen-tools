// Function: sub_16F99F0
// Address: 0x16f99f0
//
__int64 __fastcall sub_16F99F0(__int64 a1, char a2)
{
  __int64 v3; // rax
  __m128i v4; // xmm0
  unsigned __int64 v5; // rbx
  __int64 v6; // rdx
  unsigned __int64 v7; // rdx
  __int64 v8; // rbx
  int v9; // r8d
  int v10; // r9d
  _QWORD *v11; // rdi
  __m128i v13; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v14; // [rsp+18h] [rbp-48h]
  __int64 v15; // [rsp+20h] [rbp-40h]
  _QWORD v16[7]; // [rsp+28h] [rbp-38h] BYREF

  v14 = v16;
  v15 = 0;
  LOBYTE(v16[0]) = 0;
  v13.m128i_i64[1] = 1;
  v13.m128i_i64[0] = *(_QWORD *)(a1 + 40);
  sub_16F7930(a1, 1u);
  v3 = sub_145CBF0((__int64 *)(a1 + 80), 72, 16);
  v4 = _mm_loadu_si128(&v13);
  v5 = v3;
  *(_QWORD *)v3 = 0;
  v6 = v15;
  *(_QWORD *)(v3 + 8) = 0;
  *(__m128i *)(v3 + 24) = v4;
  *(_DWORD *)(v3 + 16) = a2 == 0 ? 14 : 12;
  *(_QWORD *)(v3 + 40) = v3 + 56;
  sub_16F6740((__int64 *)(v3 + 40), v16, (__int64)v16 + v6);
  *(_QWORD *)(v5 + 8) = a1 + 184;
  v7 = *(_QWORD *)(a1 + 184) & 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v5 = v7 | *(_QWORD *)v5 & 7LL;
  *(_QWORD *)(v7 + 8) = v5;
  v8 = *(_QWORD *)(a1 + 184) & 7LL | v5;
  LODWORD(v7) = *(_DWORD *)(a1 + 60) - 1;
  *(_QWORD *)(a1 + 184) = v8;
  sub_16F79B0(a1, v8 & 0xFFFFFFFFFFFFFFF8LL, v7, 0, v9, v10);
  v11 = v14;
  ++*(_DWORD *)(a1 + 68);
  *(_BYTE *)(a1 + 73) = 1;
  if ( v11 != v16 )
    j_j___libc_free_0(v11, v16[0] + 1LL);
  return 1;
}
