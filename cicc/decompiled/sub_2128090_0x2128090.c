// Function: sub_2128090
// Address: 0x2128090
//
__int64 __fastcall sub_2128090(__int64 *a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v6; // rsi
  const void **v7; // r15
  unsigned int v8; // r14d
  __int64 v9; // rsi
  __int64 *v10; // rdi
  __int64 v11; // r14
  __int64 v13; // [rsp+0h] [rbp-50h] BYREF
  int v14; // [rsp+8h] [rbp-48h]
  const void **v15; // [rsp+10h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v13,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = *(_QWORD *)(a2 + 72);
  v7 = v15;
  v8 = (unsigned __int8)v14;
  v13 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v13, v6, 2);
  v9 = *(unsigned __int16 *)(a2 + 24);
  v10 = (__int64 *)a1[1];
  v14 = *(_DWORD *)(a2 + 64);
  v11 = sub_1D309E0(v10, v9, (__int64)&v13, v8, v7, 0, a3, a4, a5, *(_OWORD *)*(_QWORD *)(a2 + 32));
  if ( v13 )
    sub_161E7C0((__int64)&v13, v13);
  return v11;
}
