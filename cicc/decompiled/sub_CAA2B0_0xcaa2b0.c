// Function: sub_CAA2B0
// Address: 0xcaa2b0
//
__int64 __fastcall sub_CAA2B0(__int64 a1, char a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  unsigned __int64 v5; // rbx
  __m128i v6; // xmm0
  __int64 v7; // rdx
  __int64 v8; // rax
  _QWORD *v9; // rdi
  __m128i v11; // [rsp+8h] [rbp-58h] BYREF
  _BYTE *v12; // [rsp+18h] [rbp-48h]
  __int64 v13; // [rsp+20h] [rbp-40h]
  _QWORD v14[7]; // [rsp+28h] [rbp-38h] BYREF

  sub_CA9FC0(a1, -1);
  *(_WORD *)(a1 + 73) = 0;
  v12 = v14;
  *(_DWORD *)(a1 + 232) = 0;
  v3 = *(_QWORD *)(a1 + 40);
  v13 = 0;
  v11.m128i_i64[0] = v3;
  LOBYTE(v14[0]) = 0;
  v11.m128i_i64[1] = 3;
  sub_CA7F70(a1, 3u);
  v4 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 160) += 72LL;
  v5 = (v4 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( *(_QWORD *)(a1 + 88) >= v5 + 72 && v4 )
  {
    *(_QWORD *)(a1 + 80) = v5 + 72;
    if ( !v5 )
    {
      MEMORY[8] = a1 + 176;
      BUG();
    }
  }
  else
  {
    v5 = sub_9D1E70(a1 + 80, 72, 72, 4);
  }
  *(_QWORD *)v5 = 0;
  *(_QWORD *)(v5 + 8) = 0;
  *(_DWORD *)(v5 + 16) = (a2 == 0) + 5;
  v6 = _mm_loadu_si128(&v11);
  *(_QWORD *)(v5 + 40) = v5 + 56;
  *(__m128i *)(v5 + 24) = v6;
  sub_CA64F0((__int64 *)(v5 + 40), v12, (__int64)&v12[v13]);
  v7 = *(_QWORD *)(a1 + 176);
  v8 = *(_QWORD *)v5;
  *(_QWORD *)(v5 + 8) = a1 + 176;
  v7 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v5 = v7 | v8 & 7;
  *(_QWORD *)(v7 + 8) = v5;
  v9 = v12;
  *(_QWORD *)(a1 + 176) = *(_QWORD *)(a1 + 176) & 7LL | v5;
  if ( v9 != v14 )
    j_j___libc_free_0(v9, v14[0] + 1LL);
  return 1;
}
