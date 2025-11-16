// Function: sub_39ACC30
// Address: 0x39acc30
//
unsigned __int64 __fastcall sub_39ACC30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 v5; // rax

  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 248LL);
  v4 = sub_38CF310(a3, 0, v3, 0);
  v5 = sub_38CF310(a2, 0, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 248LL), 0);
  return sub_38CB1F0(17, v5, v4, v3, 0);
}
