// Function: sub_B83C20
// Address: 0xb83c20
//
__int64 __fastcall sub_B83C20(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  _QWORD *v4; // rdi

  v2 = a1 - 176;
  v4 = (_QWORD *)(a1 + 392);
  *(v4 - 71) = &unk_49DA898;
  *(v4 - 49) = &unk_49DA950;
  *v4 = &unk_49DA990;
  sub_B83890((__int64)v4);
  sub_B81E70(a1, a2);
  return sub_BB9100(v2);
}
