// Function: sub_FDC130
// Address: 0xfdc130
//
__int64 __fastcall sub_FDC130(_QWORD *a1)
{
  __int64 *v2; // rdi

  v2 = a1 + 22;
  *(v2 - 22) = (__int64)&unk_49E54A0;
  sub_FDC110(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
