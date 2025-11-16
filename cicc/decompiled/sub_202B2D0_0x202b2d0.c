// Function: sub_202B2D0
// Address: 0x202b2d0
//
__int64 *__fastcall sub_202B2D0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 *v8; // r13
  const void **v9; // r15
  unsigned int v10; // r12d
  __int64 *v11; // r12
  __int64 v13; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int64 v14; // [rsp+8h] [rbp-58h]
  __int128 v15; // [rsp+10h] [rbp-50h] BYREF
  __int64 v16; // [rsp+20h] [rbp-40h] BYREF
  int v17; // [rsp+28h] [rbp-38h]

  v13 = 0;
  LODWORD(v14) = 0;
  *(_QWORD *)&v15 = 0;
  DWORD2(v15) = 0;
  sub_2025380(a1, a2, (__int64)&v13, (__int64)&v15, a3, a4, a5);
  v6 = *(_QWORD *)(a2 + 40);
  v7 = *(_QWORD *)(a2 + 72);
  v8 = *(__int64 **)(a1 + 8);
  v9 = *(const void ***)(v6 + 8);
  v10 = **(unsigned __int8 **)(a2 + 40);
  v16 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v16, v7, 2);
  v17 = *(_DWORD *)(a2 + 64);
  v11 = sub_1D332F0(
          v8,
          107,
          (__int64)&v16,
          v10,
          v9,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          a5,
          v13,
          v14,
          v15);
  if ( v16 )
    sub_161E7C0((__int64)&v16, v16);
  return v11;
}
