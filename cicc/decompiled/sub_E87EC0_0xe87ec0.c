// Function: sub_E87EC0
// Address: 0xe87ec0
//
unsigned __int64 __fastcall sub_E87EC0(__int64 a1)
{
  unsigned __int64 v1; // rax
  __int64 v2; // r8
  unsigned __int64 v3; // rax
  __int64 v4; // r8
  unsigned __int64 result; // rax

  *(_QWORD *)(a1 + 24) = sub_E6D310(*(_QWORD **)(a1 + 920), ".text", 5, 2, 0, 0);
  v1 = sub_E6D310(*(_QWORD **)(a1 + 920), ".bss", 4, 15, 0, 0);
  v2 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 40) = v1;
  v3 = sub_E6D310(*(_QWORD **)(a1 + 920), ".ppa1", 5, 0, v2, 2);
  v4 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 752) = v3;
  *(_QWORD *)(a1 + 760) = sub_E6D310(*(_QWORD **)(a1 + 920), ".ppa2", 5, 0, v4, 4);
  *(_QWORD *)(a1 + 768) = sub_E6D310(*(_QWORD **)(a1 + 920), ".ppa2list", 9, 19, 0, 0);
  *(_QWORD *)(a1 + 776) = sub_E6D310(*(_QWORD **)(a1 + 920), ".ada", 4, 19, 0, 0);
  result = sub_E6D310(*(_QWORD **)(a1 + 920), "B_IDRL", 6, 19, 0, 0);
  *(_QWORD *)(a1 + 784) = result;
  return result;
}
