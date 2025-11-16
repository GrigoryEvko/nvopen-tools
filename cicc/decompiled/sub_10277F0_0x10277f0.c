// Function: sub_10277F0
// Address: 0x10277f0
//
__int64 __fastcall sub_10277F0(_QWORD *a1)
{
  __int64 *v2; // rdi

  v2 = a1 + 22;
  *(v2 - 22) = (__int64)&unk_49E5748;
  sub_FDC110(v2);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  return j_j___libc_free_0(a1, 216);
}
