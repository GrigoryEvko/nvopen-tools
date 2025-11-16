// Function: sub_211A930
// Address: 0x211a930
//
__int64 *__fastcall sub_211A930(__int64 *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 *v6; // r13
  __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // r15
  unsigned __int64 v10; // rdx
  __int64 v11; // rsi
  const void **v12; // r8
  __int64 v13; // rcx
  __int64 *v14; // r12
  __int128 v16; // [rsp-10h] [rbp-80h]
  __int64 v17; // [rsp+0h] [rbp-70h]
  const void **v18; // [rsp+8h] [rbp-68h]
  __int64 v19; // [rsp+10h] [rbp-60h]
  unsigned __int64 v20; // [rsp+18h] [rbp-58h]
  __int64 v21; // [rsp+20h] [rbp-50h] BYREF
  int v22; // [rsp+28h] [rbp-48h]
  const void **v23; // [rsp+30h] [rbp-40h]

  v6 = (__int64 *)a1[1];
  v7 = sub_200D2A0(
         (__int64)a1,
         *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
         *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
         a3,
         a4,
         *(double *)a5.m128i_i64);
  v9 = v8;
  v19 = sub_200D2A0(
          (__int64)a1,
          **(_QWORD **)(a2 + 32),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
          a3,
          a4,
          *(double *)a5.m128i_i64);
  v20 = v10;
  sub_1F40D10(
    (__int64)&v21,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v11 = *(_QWORD *)(a2 + 72);
  v12 = v23;
  v13 = (unsigned __int8)v22;
  v21 = v11;
  if ( v11 )
  {
    v17 = (unsigned __int8)v22;
    v18 = v23;
    sub_1623A60((__int64)&v21, v11, 2);
    v13 = v17;
    v12 = v18;
  }
  *((_QWORD *)&v16 + 1) = v9;
  *(_QWORD *)&v16 = v7;
  v22 = *(_DWORD *)(a2 + 64);
  v14 = sub_1D332F0(v6, 50, (__int64)&v21, v13, v12, 0, a3, a4, a5, v19, v20, v16);
  if ( v21 )
    sub_161E7C0((__int64)&v21, v21);
  return v14;
}
