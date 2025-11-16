// Function: sub_CAA150
// Address: 0xcaa150
//
__int64 __fastcall sub_CAA150(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  unsigned __int64 v4; // rbx
  __m128i v5; // xmm0
  __int64 v6; // rdx
  __int64 v7; // rax
  _QWORD *v8; // rdi
  __m128i v10; // [rsp+8h] [rbp-58h] BYREF
  _BYTE *v11; // [rsp+18h] [rbp-48h]
  __int64 v12; // [rsp+20h] [rbp-40h]
  _QWORD v13[7]; // [rsp+28h] [rbp-38h] BYREF

  sub_CA8360(a1, *(_DWORD *)(a1 + 68));
  v11 = v13;
  *(_WORD *)(a1 + 73) = 1;
  v2 = *(_QWORD *)(a1 + 40);
  v12 = 0;
  v10.m128i_i64[0] = v2;
  LOBYTE(v13[0]) = 0;
  v10.m128i_i64[1] = 1;
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
  *(_DWORD *)(v4 + 16) = 11;
  v5 = _mm_loadu_si128(&v10);
  *(_QWORD *)(v4 + 40) = v4 + 56;
  *(__m128i *)(v4 + 24) = v5;
  sub_CA64F0((__int64 *)(v4 + 40), v11, (__int64)&v11[v12]);
  v6 = *(_QWORD *)(a1 + 176);
  v7 = *(_QWORD *)v4;
  *(_QWORD *)(v4 + 8) = a1 + 176;
  v6 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v4 = v6 | v7 & 7;
  *(_QWORD *)(v6 + 8) = v4;
  v8 = v11;
  *(_QWORD *)(a1 + 176) = *(_QWORD *)(a1 + 176) & 7LL | v4;
  if ( v8 != v13 )
    j_j___libc_free_0(v8, v13[0] + 1LL);
  return 1;
}
