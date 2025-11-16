// Function: sub_160F490
// Address: 0x160f490
//
__int64 __fastcall sub_160F490(__int64 a1)
{
  _QWORD *v2; // rdi

  v2 = (_QWORD *)(a1 + 160);
  *(v2 - 20) = &unk_49EDC10;
  *v2 = &unk_49EDCC8;
  sub_160F3F0((__int64)v2);
  return sub_1636790(a1);
}
