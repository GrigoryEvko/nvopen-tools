// Function: sub_D89BD0
// Address: 0xd89bd0
//
__int64 __fastcall sub_D89BD0(_QWORD *a1)
{
  __int64 v2; // rdi

  v2 = (__int64)(a1 + 22);
  *(_QWORD *)(v2 - 176) = &unk_49DE740;
  sub_D89A50(v2);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  return j_j___libc_free_0(a1, 224);
}
