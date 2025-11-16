// Function: sub_B83FD0
// Address: 0xb83fd0
//
__int64 __fastcall sub_B83FD0(_QWORD *a1, __int64 a2)
{
  *(a1 - 71) = &unk_49DA770;
  *(a1 - 49) = &unk_49DA828;
  *a1 = &unk_49DA868;
  sub_B83890((__int64)a1);
  sub_B81E70((__int64)(a1 - 49), a2);
  sub_BB9100(a1 - 71);
  return j_j___libc_free_0(a1 - 71, 1296);
}
