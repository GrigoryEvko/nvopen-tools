// Function: sub_B83F60
// Address: 0xb83f60
//
__int64 __fastcall sub_B83F60(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  _QWORD *v4; // rdi

  v2 = a1 - 176;
  v4 = (_QWORD *)(a1 + 392);
  *(v4 - 71) = &unk_49DA770;
  *(v4 - 49) = &unk_49DA828;
  *v4 = &unk_49DA868;
  sub_B83890((__int64)v4);
  sub_B81E70(a1, a2);
  sub_BB9100(v2);
  return j_j___libc_free_0(v2, 1296);
}
