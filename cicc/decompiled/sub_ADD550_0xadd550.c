// Function: sub_ADD550
// Address: 0xadd550
//
__int64 __fastcall sub_ADD550(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  int v8; // r12d
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  int v13; // eax

  v4 = sub_BCB2E0(*(_QWORD *)(a1 + 8));
  v5 = sub_ACD640(v4, a2, 1u);
  v8 = sub_B98A20(v5, a2, v6, v7);
  v9 = sub_BCB2E0(*(_QWORD *)(a1 + 8));
  v10 = sub_ACD640(v9, a3, 1u);
  v13 = sub_B98A20(v10, a3, v11, v12);
  return sub_B02F70(*(_QWORD *)(a1 + 8), v13, v8, 0, 0, 0, 1);
}
