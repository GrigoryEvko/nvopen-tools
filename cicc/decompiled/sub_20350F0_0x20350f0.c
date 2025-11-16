// Function: sub_20350F0
// Address: 0x20350f0
//
__int64 __fastcall sub_20350F0(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 *v9; // r13
  __int64 v10; // r10
  __int64 v11; // r11
  unsigned int v12; // r12d
  const void **v13; // r15
  __int64 v14; // r12
  __int128 v16; // [rsp-10h] [rbp-60h]
  __int64 v17; // [rsp+0h] [rbp-50h]
  __int64 v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h] BYREF
  int v20; // [rsp+18h] [rbp-38h]

  v6 = sub_2032580(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v8 = *(_QWORD *)(a2 + 72);
  v9 = *(__int64 **)(a1 + 8);
  v10 = v6;
  v11 = v7;
  v12 = **(unsigned __int8 **)(a2 + 40);
  v13 = *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL);
  v19 = v8;
  if ( v8 )
  {
    v18 = v7;
    v17 = v6;
    sub_1623A60((__int64)&v19, v8, 2);
    v10 = v17;
    v11 = v18;
  }
  *((_QWORD *)&v16 + 1) = v11;
  *(_QWORD *)&v16 = v10;
  v20 = *(_DWORD *)(a2 + 64);
  v14 = sub_1D309E0(v9, 158, (__int64)&v19, v12, v13, 0, a3, a4, a5, v16);
  if ( v19 )
    sub_161E7C0((__int64)&v19, v19);
  return v14;
}
