// Function: sub_B83E10
// Address: 0xb83e10
//
__int64 __fastcall sub_B83E10(_QWORD *a1, __int64 a2)
{
  *(a1 - 71) = &unk_49DA898;
  *(a1 - 49) = &unk_49DA950;
  *a1 = &unk_49DA990;
  sub_B83890((__int64)a1);
  sub_B81E70((__int64)(a1 - 49), a2);
  sub_BB9100(a1 - 71);
  return j_j___libc_free_0(a1 - 71, 1304);
}
