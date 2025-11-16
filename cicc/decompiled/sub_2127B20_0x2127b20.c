// Function: sub_2127B20
// Address: 0x2127b20
//
__int64 __fastcall sub_2127B20(__int64 *a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 *v6; // r15
  __int64 *v7; // r12
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 v10; // rsi
  const void **v11; // r8
  __int64 v12; // rcx
  __int64 v13; // r12
  __int128 v15; // [rsp-10h] [rbp-70h]
  __int64 v16; // [rsp+0h] [rbp-60h]
  const void **v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+10h] [rbp-50h] BYREF
  int v19; // [rsp+18h] [rbp-48h]
  const void **v20; // [rsp+20h] [rbp-40h]

  v6 = (__int64 *)a1[1];
  v7 = sub_200DAC0(
         (__int64)a1,
         **(_QWORD **)(a2 + 32),
         *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
         *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
         *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
         a3,
         a4,
         a5);
  v9 = v8;
  sub_1F40D10(
    (__int64)&v18,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v10 = *(_QWORD *)(a2 + 72);
  v11 = v20;
  v12 = (unsigned __int8)v19;
  v18 = v10;
  if ( v10 )
  {
    v16 = (unsigned __int8)v19;
    v17 = v20;
    sub_1623A60((__int64)&v18, v10, 2);
    v12 = v16;
    v11 = v17;
  }
  *((_QWORD *)&v15 + 1) = v9;
  *(_QWORD *)&v15 = v7;
  v19 = *(_DWORD *)(a2 + 64);
  v13 = sub_1D309E0(v6, 144, (__int64)&v18, v12, v11, 0, *(double *)a3.m128i_i64, a4, *(double *)a5.m128i_i64, v15);
  if ( v18 )
    sub_161E7C0((__int64)&v18, v18);
  return v13;
}
