// Function: sub_2126900
// Address: 0x2126900
//
__int64 __fastcall sub_2126900(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 *v9; // r13
  __int64 v10; // r10
  __int64 v11; // r11
  unsigned int v12; // r12d
  const void **v13; // r15
  __int64 v14; // rsi
  __int64 v15; // r12
  __int128 v17; // [rsp-10h] [rbp-60h]
  __int64 v18; // [rsp+0h] [rbp-50h]
  __int64 v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h] BYREF
  int v21; // [rsp+18h] [rbp-38h]

  v6 = sub_2125740(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v8 = *(_QWORD *)(a2 + 72);
  v9 = *(__int64 **)(a1 + 8);
  v10 = v6;
  v11 = v7;
  v12 = **(unsigned __int8 **)(a2 + 40);
  v13 = *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL);
  v20 = v8;
  if ( v8 )
  {
    v19 = v7;
    v18 = v6;
    sub_1623A60((__int64)&v20, v8, 2);
    v10 = v18;
    v11 = v19;
  }
  *((_QWORD *)&v17 + 1) = v11;
  *(_QWORD *)&v17 = v10;
  v14 = *(unsigned __int16 *)(a2 + 24);
  v21 = *(_DWORD *)(a2 + 64);
  v15 = sub_1D309E0(v9, v14, (__int64)&v20, v12, v13, 0, a3, a4, a5, v17);
  if ( v20 )
    sub_161E7C0((__int64)&v20, v20);
  return v15;
}
