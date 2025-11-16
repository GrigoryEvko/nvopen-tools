// Function: sub_16F98C0
// Address: 0x16f98c0
//
__int64 __fastcall sub_16F98C0(__int64 a1, char a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __m128i v4; // xmm0
  unsigned __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rdx
  int v8; // eax
  __m128i v10; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v11; // [rsp+18h] [rbp-48h]
  __int64 v12; // [rsp+20h] [rbp-40h]
  _QWORD v13[7]; // [rsp+28h] [rbp-38h] BYREF

  sub_16F7BE0(a1, *(_DWORD *)(a1 + 68));
  *(_BYTE *)(a1 + 73) = 0;
  v11 = v13;
  v12 = 0;
  v2 = *(_QWORD *)(a1 + 40);
  LOBYTE(v13[0]) = 0;
  v10.m128i_i64[0] = v2;
  v10.m128i_i64[1] = 1;
  sub_16F7930(a1, 1u);
  v3 = sub_145CBF0((__int64 *)(a1 + 80), 72, 16);
  v4 = _mm_loadu_si128(&v10);
  v5 = v3;
  *(_QWORD *)v3 = 0;
  v6 = v12;
  *(_QWORD *)(v3 + 8) = 0;
  *(__m128i *)(v3 + 24) = v4;
  *(_DWORD *)(v3 + 16) = a2 == 0 ? 15 : 13;
  *(_QWORD *)(v3 + 40) = v3 + 56;
  sub_16F6740((__int64 *)(v3 + 40), v13, (__int64)v13 + v6);
  v7 = *(_QWORD *)(a1 + 184);
  *(_QWORD *)(v5 + 8) = a1 + 184;
  v7 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v5 = v7 | *(_QWORD *)v5 & 7LL;
  *(_QWORD *)(v7 + 8) = v5;
  v8 = *(_DWORD *)(a1 + 68);
  *(_QWORD *)(a1 + 184) = *(_QWORD *)(a1 + 184) & 7LL | v5;
  if ( v8 )
    *(_DWORD *)(a1 + 68) = v8 - 1;
  if ( v11 != v13 )
    j_j___libc_free_0(v11, v13[0] + 1LL);
  return 1;
}
