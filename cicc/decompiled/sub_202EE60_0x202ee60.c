// Function: sub_202EE60
// Address: 0x202ee60
//
__int64 __fastcall sub_202EE60(__int64 *a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v6; // rsi
  const void **v7; // r15
  unsigned int v8; // r14d
  __int64 *v9; // r12
  __int128 *v10; // rcx
  __int64 v11; // r14
  __int128 *v13; // [rsp+8h] [rbp-58h]
  __int64 v14; // [rsp+10h] [rbp-50h] BYREF
  int v15; // [rsp+18h] [rbp-48h]
  const void **v16; // [rsp+20h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v14,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = *(_QWORD *)(a2 + 72);
  v7 = v16;
  v8 = (unsigned __int8)v15;
  v9 = (__int64 *)a1[1];
  v14 = v6;
  v10 = *(__int128 **)(a2 + 32);
  if ( v6 )
  {
    v13 = *(__int128 **)(a2 + 32);
    sub_1623A60((__int64)&v14, v6, 2);
    v10 = v13;
  }
  v15 = *(_DWORD *)(a2 + 64);
  v11 = sub_1D309E0(v9, 111, (__int64)&v14, v8, v7, 0, a3, a4, a5, *v10);
  if ( v14 )
    sub_161E7C0((__int64)&v14, v14);
  return v11;
}
