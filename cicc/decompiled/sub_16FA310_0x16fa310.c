// Function: sub_16FA310
// Address: 0x16fa310
//
__int64 __fastcall sub_16FA310(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rax
  __int64 v8; // rax
  __m128i v9; // xmm0
  unsigned __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  _QWORD *v14; // rdi
  __m128i v16; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v17; // [rsp+18h] [rbp-48h]
  __int64 v18; // [rsp+20h] [rbp-40h]
  _QWORD v19[7]; // [rsp+28h] [rbp-38h] BYREF

  sub_16FA1A0(a1, *(_DWORD *)(a1 + 60), 9, (unsigned __int64 *)(a1 + 184), a5, a6);
  sub_16F7BE0(a1, *(_DWORD *)(a1 + 68));
  *(_BYTE *)(a1 + 73) = 1;
  v7 = *(_QWORD *)(a1 + 40);
  v17 = v19;
  v16.m128i_i64[0] = v7;
  v18 = 0;
  LOBYTE(v19[0]) = 0;
  v16.m128i_i64[1] = 1;
  sub_16F7930(a1, 1u);
  v8 = sub_145CBF0((__int64 *)(a1 + 80), 72, 16);
  v9 = _mm_loadu_si128(&v16);
  v10 = v8;
  *(_QWORD *)v8 = 0;
  v11 = v18;
  *(_QWORD *)(v8 + 8) = 0;
  *(__m128i *)(v8 + 24) = v9;
  *(_DWORD *)(v8 + 16) = 7;
  *(_QWORD *)(v8 + 40) = v8 + 56;
  sub_16F6740((__int64 *)(v8 + 40), v19, (__int64)v19 + v11);
  v12 = *(_QWORD *)v10;
  *(_QWORD *)(v10 + 8) = a1 + 184;
  v13 = *(_QWORD *)(a1 + 184) & 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v10 = v13 | v12 & 7;
  *(_QWORD *)(v13 + 8) = v10;
  v14 = v17;
  *(_QWORD *)(a1 + 184) = *(_QWORD *)(a1 + 184) & 7LL | v10;
  if ( v14 != v19 )
    j_j___libc_free_0(v14, v19[0] + 1LL);
  return 1;
}
