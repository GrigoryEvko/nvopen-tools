// Function: sub_CAB5F0
// Address: 0xcab5f0
//
__int64 __fastcall sub_CAB5F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // esi
  int v8; // eax
  bool v9; // zf
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rbx
  __m128i v13; // xmm0
  __int64 v14; // rdx
  __int64 v15; // rax
  _QWORD *v16; // rdi
  __m128i v18; // [rsp+8h] [rbp-58h] BYREF
  _BYTE *v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h]
  _QWORD v21[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = *(_DWORD *)(a1 + 68);
  if ( !v7 )
  {
    sub_CAB2C0(a1, *(_DWORD *)(a1 + 60), 10, (unsigned __int64 *)(a1 + 176), a5, a6);
    v7 = *(_DWORD *)(a1 + 68);
  }
  sub_CA8360(a1, v7);
  v8 = *(_DWORD *)(a1 + 68);
  v19 = v21;
  *(_BYTE *)(a1 + 74) = 0;
  v9 = v8 == 0;
  v10 = *(_QWORD *)(a1 + 40);
  v20 = 0;
  *(_BYTE *)(a1 + 73) = v9;
  v18.m128i_i64[0] = v10;
  LOBYTE(v21[0]) = 0;
  v18.m128i_i64[1] = 1;
  sub_CA7F70(a1, 1u);
  v11 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 160) += 72LL;
  v12 = (v11 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( *(_QWORD *)(a1 + 88) >= v12 + 72 && v11 )
  {
    *(_QWORD *)(a1 + 80) = v12 + 72;
    if ( !v12 )
    {
      MEMORY[8] = a1 + 176;
      BUG();
    }
  }
  else
  {
    v12 = sub_9D1E70(a1 + 80, 72, 72, 4);
  }
  *(_QWORD *)v12 = 0;
  *(_QWORD *)(v12 + 8) = 0;
  *(_DWORD *)(v12 + 16) = 16;
  v13 = _mm_loadu_si128(&v18);
  *(_QWORD *)(v12 + 40) = v12 + 56;
  *(__m128i *)(v12 + 24) = v13;
  sub_CA64F0((__int64 *)(v12 + 40), v19, (__int64)&v19[v20]);
  v14 = *(_QWORD *)(a1 + 176);
  v15 = *(_QWORD *)v12;
  *(_QWORD *)(v12 + 8) = a1 + 176;
  v14 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v12 = v14 | v15 & 7;
  *(_QWORD *)(v14 + 8) = v12;
  v16 = v19;
  *(_QWORD *)(a1 + 176) = *(_QWORD *)(a1 + 176) & 7LL | v12;
  if ( v16 != v21 )
    j_j___libc_free_0(v16, v21[0] + 1LL);
  return 1;
}
