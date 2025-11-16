// Function: sub_3737610
// Address: 0x3737610
//
__int64 __fastcall sub_3737610(__int64 *a1)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 v4; // r13
  __int64 v5; // rbx
  unsigned __int16 v6; // ax

  v2 = sub_31DA6B0(a1[23]);
  v3 = a1[26];
  v4 = *(_QWORD *)(v3 + 4880);
  v5 = *(_QWORD *)(*(_QWORD *)(v2 + 312) + 16LL);
  v6 = sub_3220AA0(v3);
  return sub_324AC60(a1, (__int64)(a1 + 1), v6 < 5u ? 8499 : 115, v4, v5);
}
