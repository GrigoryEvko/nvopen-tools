// Function: sub_D89B80
// Address: 0xd89b80
//
__int64 __fastcall sub_D89B80(_QWORD *a1)
{
  __int64 v2; // rdi

  v2 = (__int64)(a1 + 22);
  *(_QWORD *)(v2 - 176) = &unk_49DE740;
  sub_D89A50(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
