// Function: sub_B83DA0
// Address: 0xb83da0
//
__int64 __fastcall sub_B83DA0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi

  v3 = (_QWORD *)(a1 + 568);
  *(v3 - 71) = &unk_49DA898;
  *(v3 - 49) = &unk_49DA950;
  *v3 = &unk_49DA990;
  sub_B83890((__int64)v3);
  sub_B81E70(a1 + 176, a2);
  sub_BB9100(a1);
  return j_j___libc_free_0(a1, 1304);
}
