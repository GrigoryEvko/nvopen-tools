// Function: sub_CAB480
// Address: 0xcab480
//
__int64 __fastcall sub_CAB480(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rbx
  __m128i v10; // xmm0
  __int64 v11; // rdx
  __int64 v12; // rax
  _QWORD *v13; // rdi
  __m128i v15; // [rsp+8h] [rbp-58h] BYREF
  _BYTE *v16; // [rsp+18h] [rbp-48h]
  __int64 v17; // [rsp+20h] [rbp-40h]
  _QWORD v18[7]; // [rsp+28h] [rbp-38h] BYREF

  sub_CAB2C0(a1, *(_DWORD *)(a1 + 60), 9, (unsigned __int64 *)(a1 + 176), a5, a6);
  sub_CA8360(a1, *(_DWORD *)(a1 + 68));
  v16 = v18;
  *(_WORD *)(a1 + 73) = 1;
  v7 = *(_QWORD *)(a1 + 40);
  v17 = 0;
  v15.m128i_i64[0] = v7;
  LOBYTE(v18[0]) = 0;
  v15.m128i_i64[1] = 1;
  sub_CA7F70(a1, 1u);
  v8 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 160) += 72LL;
  v9 = (v8 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( *(_QWORD *)(a1 + 88) >= v9 + 72 && v8 )
  {
    *(_QWORD *)(a1 + 80) = v9 + 72;
    if ( !v9 )
    {
      MEMORY[8] = a1 + 176;
      BUG();
    }
  }
  else
  {
    v9 = sub_9D1E70(a1 + 80, 72, 72, 4);
  }
  *(_QWORD *)v9 = 0;
  *(_QWORD *)(v9 + 8) = 0;
  *(_DWORD *)(v9 + 16) = 7;
  v10 = _mm_loadu_si128(&v15);
  *(_QWORD *)(v9 + 40) = v9 + 56;
  *(__m128i *)(v9 + 24) = v10;
  sub_CA64F0((__int64 *)(v9 + 40), v16, (__int64)&v16[v17]);
  v11 = *(_QWORD *)(a1 + 176);
  v12 = *(_QWORD *)v9;
  *(_QWORD *)(v9 + 8) = a1 + 176;
  v11 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v9 = v11 | v12 & 7;
  *(_QWORD *)(v11 + 8) = v9;
  v13 = v16;
  *(_QWORD *)(a1 + 176) = *(_QWORD *)(a1 + 176) & 7LL | v9;
  if ( v13 != v18 )
    j_j___libc_free_0(v13, v18[0] + 1LL);
  return 1;
}
