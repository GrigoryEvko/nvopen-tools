// Function: sub_16F91E0
// Address: 0x16f91e0
//
__int64 __fastcall sub_16F91E0(__int64 a1, int a2)
{
  int v2; // eax
  __int64 v4; // rax
  _BYTE *v5; // rsi
  __m128i v6; // xmm0
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rdx
  __m128i v14; // [rsp+18h] [rbp-68h] BYREF
  _QWORD *v15; // [rsp+28h] [rbp-58h]
  __int64 v16; // [rsp+30h] [rbp-50h]
  _QWORD v17[9]; // [rsp+38h] [rbp-48h] BYREF

  v15 = v17;
  v2 = *(_DWORD *)(a1 + 68);
  v16 = 0;
  LOBYTE(v17[0]) = 0;
  if ( !v2 && a2 < *(_DWORD *)(a1 + 56) )
  {
    do
    {
      v14.m128i_i64[0] = *(_QWORD *)(a1 + 40);
      v14.m128i_i64[1] = 1;
      v4 = sub_145CBF0((__int64 *)(a1 + 80), 72, 16);
      v5 = v15;
      v6 = _mm_loadu_si128(&v14);
      v7 = v4;
      *(_QWORD *)v4 = 0;
      v8 = v16;
      *(_QWORD *)(v4 + 8) = 0;
      *(__m128i *)(v4 + 24) = v6;
      *(_DWORD *)(v4 + 16) = 8;
      *(_QWORD *)(v4 + 40) = v4 + 56;
      sub_16F6740((__int64 *)(v4 + 40), v5, (__int64)&v5[v8]);
      v9 = *(_QWORD *)v7;
      v10 = *(_QWORD *)(a1 + 184);
      *(_QWORD *)(v7 + 8) = a1 + 184;
      v10 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v7 = v10 | v9 & 7;
      *(_QWORD *)(v10 + 8) = v7;
      v11 = *(unsigned int *)(a1 + 208);
      v12 = *(_QWORD *)(a1 + 200);
      LODWORD(v9) = *(_DWORD *)(a1 + 208);
      *(_QWORD *)(a1 + 184) = *(_QWORD *)(a1 + 184) & 7LL | v7;
      LODWORD(v12) = *(_DWORD *)(v12 + 4 * v11 - 4);
      *(_DWORD *)(a1 + 208) = v9 - 1;
      *(_DWORD *)(a1 + 56) = v12;
    }
    while ( a2 < (int)v12 );
    if ( v15 != v17 )
      j_j___libc_free_0(v15, v17[0] + 1LL);
  }
  return 1;
}
