// Function: sub_17B70C0
// Address: 0x17b70c0
//
__int64 __fastcall sub_17B70C0(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rdi
  __int64 v4; // rax
  _QWORD *v5; // rdi
  __int64 v6; // rax
  _QWORD *v7; // rdi
  __int64 v8; // rax
  _QWORD *v9; // rdi
  __int64 v10; // rax
  _QWORD *v11; // rdi
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rax
  char v16; // r13
  _QWORD v18[10]; // [rsp+0h] [rbp-50h] BYREF

  v2 = sub_1643350(*(_QWORD **)(a1 + 64));
  v3 = *(_QWORD **)(a1 + 64);
  v18[0] = v2;
  v4 = sub_16471D0(v3, 0);
  v5 = *(_QWORD **)(a1 + 64);
  v18[1] = v4;
  v6 = sub_1643350(v5);
  v7 = *(_QWORD **)(a1 + 64);
  v18[2] = v6;
  v8 = sub_1643330(v7);
  v9 = *(_QWORD **)(a1 + 64);
  v18[3] = v8;
  v10 = sub_1643350(v9);
  v11 = *(_QWORD **)(a1 + 64);
  v18[4] = v10;
  v12 = (__int64 *)sub_1643270(v11);
  v13 = sub_1644EA0(v12, v18, 5, 0);
  v14 = sub_1632190(*(_QWORD *)(a1 + 48), (__int64)"llvm_gcda_emit_function", 23, v13);
  if ( *(_BYTE *)(v14 + 16) )
    return v14;
  v15 = **(_QWORD **)(a1 + 56);
  if ( *(_BYTE *)(v15 + 144) )
  {
    v16 = 58;
  }
  else
  {
    v16 = 40;
    if ( !*(_BYTE *)(v15 + 146) )
      return v14;
  }
  sub_15E0DF0(v14, 0, v16);
  sub_15E0DF0(v14, 2, v16);
  sub_15E0DF0(v14, 3, v16);
  sub_15E0DF0(v14, 4, v16);
  return v14;
}
