// Function: sub_CAA9B0
// Address: 0xcaa9b0
//
__int64 __fastcall sub_CAA9B0(__int64 a1, char a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rbx
  __m128i v5; // xmm0
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // rdi
  __m128i v13; // [rsp+8h] [rbp-58h] BYREF
  _BYTE *v14; // [rsp+18h] [rbp-48h]
  __int64 v15; // [rsp+20h] [rbp-40h]
  _QWORD v16[7]; // [rsp+28h] [rbp-38h] BYREF

  v14 = v16;
  v15 = 0;
  LOBYTE(v16[0]) = 0;
  v13.m128i_i64[1] = 1;
  v13.m128i_i64[0] = *(_QWORD *)(a1 + 40);
  sub_CA7F70(a1, 1u);
  v3 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 160) += 72LL;
  v4 = (v3 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( *(_QWORD *)(a1 + 88) >= v4 + 72 && v3 )
  {
    *(_QWORD *)(a1 + 80) = v4 + 72;
    if ( !v4 )
    {
      MEMORY[8] = a1 + 176;
      BUG();
    }
  }
  else
  {
    v4 = sub_9D1E70(a1 + 80, 72, 72, 4);
  }
  *(_QWORD *)v4 = 0;
  *(_QWORD *)(v4 + 8) = 0;
  *(_DWORD *)(v4 + 16) = a2 == 0 ? 14 : 12;
  v5 = _mm_loadu_si128(&v13);
  *(_QWORD *)(v4 + 40) = v4 + 56;
  *(__m128i *)(v4 + 24) = v5;
  sub_CA64F0((__int64 *)(v4 + 40), v14, (__int64)&v14[v15]);
  v6 = *(_QWORD *)v4;
  v7 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(v4 + 8) = a1 + 176;
  v7 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v4 = v7 | v6 & 7;
  *(_QWORD *)(v7 + 8) = v4;
  v8 = *(_QWORD *)(a1 + 176) & 7LL | v4;
  LODWORD(v7) = *(_DWORD *)(a1 + 60) - 1;
  *(_QWORD *)(a1 + 176) = v8;
  sub_CA80E0(a1, v8 & 0xFFFFFFFFFFFFFFF8LL, v7, 0, v9, v10);
  v11 = v14;
  ++*(_DWORD *)(a1 + 68);
  *(_WORD *)(a1 + 73) = 1;
  if ( v11 != v16 )
    j_j___libc_free_0(v11, v16[0] + 1LL);
  return 1;
}
