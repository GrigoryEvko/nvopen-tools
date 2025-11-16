// Function: sub_21246C0
// Address: 0x21246c0
//
__int64 __fastcall sub_21246C0(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // r13
  __int128 v10; // rax
  __int64 *v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // edx
  __int64 v14; // r12
  __int64 v15; // r12
  __int64 *v17; // [rsp+8h] [rbp-68h]
  __int64 v18; // [rsp+30h] [rbp-40h] BYREF
  int v19; // [rsp+38h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 32);
  v7 = *(_QWORD *)(a2 + 72);
  v8 = *(_QWORD *)(v6 + 40);
  v9 = *(_QWORD *)(v6 + 48);
  v18 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v18, v7, 2);
  v19 = *(_DWORD *)(a2 + 64);
  if ( (*(_BYTE *)(a2 + 27) & 4) != 0 )
  {
    v17 = *(__int64 **)(a1 + 8);
    *(_QWORD *)&v10 = sub_1D38E70((__int64)v17, 0, (__int64)&v18, 0, a3, a4, a5);
    v11 = sub_1D332F0(
            v17,
            154,
            (__int64)&v18,
            *(unsigned __int8 *)(a2 + 88),
            *(const void ***)(a2 + 96),
            0,
            *(double *)a3.m128i_i64,
            a4,
            a5,
            v8,
            v9,
            v10);
    v14 = sub_200D2A0(a1, (__int64)v11, v12, *(double *)a3.m128i_i64, a4, *(double *)a5.m128i_i64);
  }
  else
  {
    v14 = sub_2120330(a1, v8, v9);
  }
  v15 = sub_1D2BB40(
          *(_QWORD **)(a1 + 8),
          **(_QWORD **)(a2 + 32),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
          (__int64)&v18,
          v14,
          v13 | v9 & 0xFFFFFFFF00000000LL,
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL),
          *(_QWORD *)(a2 + 104));
  if ( v18 )
    sub_161E7C0((__int64)&v18, v18);
  return v15;
}
