// Function: sub_17B7000
// Address: 0x17b7000
//
__int64 __fastcall sub_17B7000(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rdi
  __int64 v4; // rax
  _QWORD *v5; // rdi
  __int64 *v6; // rax
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rax
  char v10; // dl
  _QWORD v12[4]; // [rsp+0h] [rbp-20h] BYREF

  v2 = sub_1643350(*(_QWORD **)(a1 + 64));
  v3 = *(_QWORD **)(a1 + 64);
  v12[0] = v2;
  v4 = sub_1647230(v3, 0);
  v5 = *(_QWORD **)(a1 + 64);
  v12[1] = v4;
  v6 = (__int64 *)sub_1643270(v5);
  v7 = sub_1644EA0(v6, v12, 2, 0);
  v8 = sub_1632190(*(_QWORD *)(a1 + 48), (__int64)"llvm_gcda_emit_arcs", 19, v7);
  if ( *(_BYTE *)(v8 + 16) )
    return v8;
  v9 = **(_QWORD **)(a1 + 56);
  if ( *(_BYTE *)(v9 + 144) )
  {
    v10 = 58;
  }
  else
  {
    v10 = 40;
    if ( !*(_BYTE *)(v9 + 146) )
      return v8;
  }
  sub_15E0DF0(v8, 0, v10);
  return v8;
}
