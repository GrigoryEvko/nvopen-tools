// Function: sub_16FA440
// Address: 0x16fa440
//
__int64 __fastcall sub_16FA440(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  int v7; // esi
  int v8; // eax
  bool v9; // zf
  __int64 v10; // rax
  __int64 v11; // rax
  __m128i v12; // xmm0
  _BYTE *v13; // rsi
  unsigned __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  _QWORD *v18; // rdi
  __m128i v20; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h]
  _QWORD v23[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = *(_DWORD *)(a1 + 68);
  if ( !v7 )
  {
    sub_16FA1A0(a1, *(_DWORD *)(a1 + 60), 10, (unsigned __int64 *)(a1 + 184), a5, a6);
    v7 = *(_DWORD *)(a1 + 68);
  }
  sub_16F7BE0(a1, v7);
  v8 = *(_DWORD *)(a1 + 68);
  v21 = v23;
  v22 = 0;
  v9 = v8 == 0;
  v10 = *(_QWORD *)(a1 + 40);
  LOBYTE(v23[0]) = 0;
  *(_BYTE *)(a1 + 73) = v9;
  v20.m128i_i64[0] = v10;
  v20.m128i_i64[1] = 1;
  sub_16F7930(a1, 1u);
  v11 = sub_145CBF0((__int64 *)(a1 + 80), 72, 16);
  v12 = _mm_loadu_si128(&v20);
  v13 = v21;
  v14 = v11;
  *(_QWORD *)v11 = 0;
  v15 = v22;
  *(_QWORD *)(v11 + 8) = 0;
  *(__m128i *)(v11 + 24) = v12;
  *(_DWORD *)(v11 + 16) = 16;
  *(_QWORD *)(v11 + 40) = v11 + 56;
  sub_16F6740((__int64 *)(v11 + 40), v13, (__int64)&v13[v15]);
  v16 = *(_QWORD *)v14;
  *(_QWORD *)(v14 + 8) = a1 + 184;
  v17 = *(_QWORD *)(a1 + 184) & 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v14 = v17 | v16 & 7;
  *(_QWORD *)(v17 + 8) = v14;
  v18 = v21;
  *(_QWORD *)(a1 + 184) = *(_QWORD *)(a1 + 184) & 7LL | v14;
  if ( v18 != v23 )
    j_j___libc_free_0(v18, v23[0] + 1LL);
  return 1;
}
