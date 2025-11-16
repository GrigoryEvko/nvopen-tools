// Function: sub_16F9560
// Address: 0x16f9560
//
__int64 __fastcall sub_16F9560(__int64 a1)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rax
  __m128i v4; // xmm0
  unsigned __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rdx
  _QWORD *v8; // rdi
  __m128i v10; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v11; // [rsp+18h] [rbp-48h]
  __int64 v12; // [rsp+20h] [rbp-40h]
  _QWORD v13[7]; // [rsp+28h] [rbp-38h] BYREF

  if ( *(_DWORD *)(a1 + 60) )
  {
    ++*(_DWORD *)(a1 + 64);
    *(_DWORD *)(a1 + 60) = 0;
  }
  sub_16F91E0(a1, -1);
  *(_BYTE *)(a1 + 73) = 0;
  v2 = *(_QWORD *)(a1 + 40);
  *(_DWORD *)(a1 + 240) = 0;
  v11 = v13;
  v12 = 0;
  LOBYTE(v13[0]) = 0;
  v10 = (__m128i)v2;
  v3 = sub_145CBF0((__int64 *)(a1 + 80), 72, 16);
  v4 = _mm_loadu_si128(&v10);
  v5 = v3;
  *(_QWORD *)v3 = 0;
  v6 = v12;
  *(_QWORD *)(v3 + 8) = 0;
  *(__m128i *)(v3 + 24) = v4;
  *(_DWORD *)(v3 + 16) = 2;
  *(_QWORD *)(v3 + 40) = v3 + 56;
  sub_16F6740((__int64 *)(v3 + 40), v13, (__int64)v13 + v6);
  v7 = *(_QWORD *)(a1 + 184);
  *(_QWORD *)(v5 + 8) = a1 + 184;
  v7 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v5 = v7 | *(_QWORD *)v5 & 7LL;
  *(_QWORD *)(v7 + 8) = v5;
  v8 = v11;
  *(_QWORD *)(a1 + 184) = *(_QWORD *)(a1 + 184) & 7LL | v5;
  if ( v8 != v13 )
    j_j___libc_free_0(v8, v13[0] + 1LL);
  return 1;
}
