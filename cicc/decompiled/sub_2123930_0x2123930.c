// Function: sub_2123930
// Address: 0x2123930
//
__int64 __fastcall sub_2123930(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 *v6; // r12
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  unsigned __int64 v10; // r10
  unsigned __int8 *v11; // rax
  __int64 v12; // r11
  unsigned int v13; // r13d
  const void **v14; // r15
  __int64 v15; // r12
  __int128 v17; // [rsp-10h] [rbp-60h]
  unsigned __int64 v18; // [rsp+0h] [rbp-50h]
  __int64 v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h] BYREF
  int v21; // [rsp+18h] [rbp-38h]

  v6 = *(__int64 **)(a1 + 8);
  v7 = sub_2120330(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v9 = *(_QWORD *)(a2 + 72);
  v10 = v7;
  v11 = *(unsigned __int8 **)(a2 + 40);
  v12 = v8;
  v13 = *v11;
  v14 = (const void **)*((_QWORD *)v11 + 1);
  v20 = v9;
  if ( v9 )
  {
    v19 = v8;
    v18 = v10;
    sub_1623A60((__int64)&v20, v9, 2);
    v10 = v18;
    v12 = v19;
  }
  *((_QWORD *)&v17 + 1) = v12;
  *(_QWORD *)&v17 = v10;
  v21 = *(_DWORD *)(a2 + 64);
  v15 = sub_1D309E0(v6, 158, (__int64)&v20, v13, v14, 0, a3, a4, a5, v17);
  if ( v20 )
    sub_161E7C0((__int64)&v20, v20);
  return v15;
}
