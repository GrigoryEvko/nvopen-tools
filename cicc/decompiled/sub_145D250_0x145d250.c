// Function: sub_145D250
// Address: 0x145d250
//
__int64 __fastcall sub_145D250(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rax

  v5 = a4;
  v6 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 40LL));
  v7 = sub_15A9930(v6, a3);
  return sub_145CF80(a1, a2, *(_QWORD *)(v7 + 8 * v5 + 16), 0);
}
