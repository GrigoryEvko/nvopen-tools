// Function: sub_21BE640
// Address: 0x21be640
//
__int64 __fastcall sub_21BE640(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rsi
  __int64 v7; // r14
  _QWORD *v8; // r14
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // rsi
  _QWORD *v12; // r9
  __int64 v13; // r14
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int128 v19; // [rsp-68h] [rbp-68h]
  _QWORD *v20; // [rsp-50h] [rbp-50h]
  __int64 v21; // [rsp-48h] [rbp-48h] BYREF
  int v22; // [rsp-40h] [rbp-40h]

  if ( **(_BYTE **)(a2 + 40) != 8 )
    return 0;
  v6 = *(_QWORD *)(a2 + 72);
  v7 = *(_QWORD *)(a1 + 272);
  v21 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v21, v6, 2);
  v22 = *(_DWORD *)(a2 + 64);
  v8 = sub_1D36490(v7, *(_QWORD *)(a2 + 88) + 24LL, (__int64)&v21, 8u, 0, 1u, a3, a4, a5);
  v10 = v9;
  if ( v21 )
    sub_161E7C0((__int64)&v21, v21);
  v11 = *(_QWORD *)(a2 + 72);
  v12 = *(_QWORD **)(a1 + 272);
  v21 = v11;
  if ( v11 )
  {
    v20 = v12;
    sub_1623A60((__int64)&v21, v11, 2);
    v12 = v20;
  }
  *((_QWORD *)&v19 + 1) = v10;
  *(_QWORD *)&v19 = v8;
  v22 = *(_DWORD *)(a2 + 64);
  v13 = sub_1D2CC80(v12, 3075, (__int64)&v21, 8, 0, (__int64)v12, v19);
  if ( v21 )
    sub_161E7C0((__int64)&v21, v21);
  sub_1D444E0(*(_QWORD *)(a1 + 272), a2, v13);
  sub_1D49010(v13);
  sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v14, v15, v16, v17);
  return 1;
}
