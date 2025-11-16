// Function: sub_1802450
// Address: 0x1802450
//
__int64 __fastcall sub_1802450(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rdi
  __int64 *v6; // rax
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r15
  _QWORD *v11; // rax
  __int64 v12; // r14
  __int64 v13; // r13
  _QWORD *v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rsi
  unsigned __int8 *v18; // rsi
  _QWORD v20[2]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v21; // [rsp+10h] [rbp-40h]

  v5 = *(_QWORD **)(a2 + 216);
  v20[0] = "asan.module_dtor";
  v21 = 259;
  v6 = (__int64 *)sub_1643270(v5);
  v7 = sub_16453E0(v6, 0);
  v8 = sub_1648B60(120);
  v9 = v8;
  if ( v8 )
    sub_15E2490(v8, v7, 7, (__int64)v20, a3);
  *(_QWORD *)(a2 + 376) = v9;
  v10 = *(_QWORD *)(a2 + 216);
  v21 = 257;
  v11 = (_QWORD *)sub_22077B0(64);
  v12 = (__int64)v11;
  if ( v11 )
    sub_157FB60(v11, v10, (__int64)v20, v9, 0);
  v13 = *(_QWORD *)(a2 + 216);
  v14 = sub_1648A60(56, 0);
  v15 = (__int64)v14;
  if ( v14 )
    sub_15F7190((__int64)v14, v13, v12);
  v16 = sub_16498A0(v15);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 24) = v16;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(v15 + 40);
  *(_QWORD *)(a1 + 16) = v15 + 24;
  v17 = *(_QWORD *)(v15 + 48);
  v20[0] = v17;
  if ( v17 )
  {
    sub_1623A60((__int64)v20, v17, 2);
    if ( *(_QWORD *)a1 )
      sub_161E7C0(a1, *(_QWORD *)a1);
    v18 = (unsigned __int8 *)v20[0];
    *(_QWORD *)a1 = v20[0];
    if ( v18 )
      sub_1623210((__int64)v20, v18, a1);
  }
  return a1;
}
