// Function: sub_17B6EA0
// Address: 0x17b6ea0
//
__int64 __fastcall sub_17B6EA0(__int64 a1)
{
  __int64 *v2; // r13
  __int64 *v3; // r12
  __int64 *v4; // rax
  __int64 v5; // rax
  _QWORD *v6; // rdi
  __int64 *v7; // rax
  __int64 v8; // rax
  _QWORD v10[6]; // [rsp+0h] [rbp-30h] BYREF

  v2 = (__int64 *)sub_1643350(*(_QWORD **)(a1 + 64));
  v3 = (__int64 *)sub_1643360(*(_QWORD **)(a1 + 64));
  v10[0] = sub_1647190(v2, 0);
  v4 = (__int64 *)sub_1647190(v3, 0);
  v5 = sub_1647190(v4, 0);
  v6 = *(_QWORD **)(a1 + 64);
  v10[1] = v5;
  v7 = (__int64 *)sub_1643270(v6);
  v8 = sub_1644EA0(v7, v10, 2, 0);
  return sub_1632190(*(_QWORD *)(a1 + 48), (__int64)"__llvm_gcov_indirect_counter_increment", 38, v8);
}
