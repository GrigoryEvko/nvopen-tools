// Function: sub_B83C80
// Address: 0xb83c80
//
__int64 __fastcall sub_B83C80(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi

  v3 = (_QWORD *)(a1 + 568);
  *(v3 - 71) = &unk_49DA770;
  *(v3 - 49) = &unk_49DA828;
  *v3 = &unk_49DA868;
  sub_B83890((__int64)v3);
  sub_B81E70(a1 + 176, a2);
  return sub_BB9100(a1);
}
