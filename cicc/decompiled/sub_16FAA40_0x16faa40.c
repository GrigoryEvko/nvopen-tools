// Function: sub_16FAA40
// Address: 0x16faa40
//
__int64 __fastcall sub_16FAA40(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 result; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __m128i v11; // xmm0
  unsigned __int64 v12; // r12
  __int64 v13; // rdx
  _QWORD *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __m128i v19; // xmm1
  __m128i v20; // [rsp+8h] [rbp-68h] BYREF
  _BYTE *v21; // [rsp+18h] [rbp-58h]
  __int64 v22; // [rsp+20h] [rbp-50h]
  _QWORD v23[9]; // [rsp+28h] [rbp-48h] BYREF

  sub_16F91E0(a1, -1);
  *(_BYTE *)(a1 + 73) = 0;
  v2 = *(_QWORD *)(a1 + 40);
  *(_DWORD *)(a1 + 240) = 0;
  sub_16F78D0(a1, 0x25u);
  v3 = *(_QWORD *)(a1 + 40);
  v4 = sub_16F7770(a1, (char *)sub_16F6460, 0, *(_QWORD *)(a1 + 40));
  *(_QWORD *)(a1 + 40) = v4;
  v5 = v4 - v3;
  v6 = sub_16F7770(a1, (char *)sub_16F6430, 0, v4);
  *(_QWORD *)(a1 + 40) = v6;
  v7 = v6;
  result = 0;
  v20 = 0u;
  v21 = v23;
  v22 = 0;
  LOBYTE(v23[0]) = 0;
  if ( v5 == 4 )
  {
    if ( *(_DWORD *)v3 != 1280131417 )
      return result;
    v9 = sub_16F7770(a1, (char *)sub_16F6460, 0, v7);
    *(_QWORD *)(a1 + 40) = v9;
    v20.m128i_i64[0] = v2;
    v20.m128i_i64[1] = v9 - v2;
    v10 = sub_145CBF0((__int64 *)(a1 + 80), 72, 16);
    v11 = _mm_loadu_si128(&v20);
    *(_QWORD *)v10 = 0;
    v12 = v10;
    *(_QWORD *)(v10 + 8) = 0;
    *(__m128i *)(v10 + 24) = v11;
    *(_DWORD *)(v10 + 16) = 3;
  }
  else
  {
    if ( v5 != 3 )
      return result;
    if ( *(_WORD *)v3 != 16724 || *(_BYTE *)(v3 + 2) != 71 )
      return 0;
    v15 = sub_16F7770(a1, (char *)sub_16F6460, 0, v7);
    *(_QWORD *)(a1 + 40) = v15;
    v16 = sub_16F7770(a1, (char *)sub_16F6430, 0, v15);
    *(_QWORD *)(a1 + 40) = v16;
    v17 = sub_16F7770(a1, (char *)sub_16F6460, 0, v16);
    *(_QWORD *)(a1 + 40) = v17;
    v20.m128i_i64[0] = v2;
    v20.m128i_i64[1] = v17 - v2;
    v18 = sub_145CBF0((__int64 *)(a1 + 80), 72, 16);
    v19 = _mm_loadu_si128(&v20);
    *(_QWORD *)v18 = 0;
    v12 = v18;
    *(_QWORD *)(v18 + 8) = 0;
    *(__m128i *)(v18 + 24) = v19;
    *(_DWORD *)(v18 + 16) = 4;
  }
  *(_QWORD *)(v12 + 40) = v12 + 56;
  sub_16F6740((__int64 *)(v12 + 40), v21, (__int64)&v21[v22]);
  v13 = *(_QWORD *)(a1 + 184);
  *(_QWORD *)(v12 + 8) = a1 + 184;
  v13 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v12 = v13 | *(_QWORD *)v12 & 7LL;
  *(_QWORD *)(v13 + 8) = v12;
  v14 = v21;
  *(_QWORD *)(a1 + 184) = *(_QWORD *)(a1 + 184) & 7LL | v12;
  if ( v14 != v23 )
    j_j___libc_free_0(v14, v23[0] + 1LL);
  return 1;
}
