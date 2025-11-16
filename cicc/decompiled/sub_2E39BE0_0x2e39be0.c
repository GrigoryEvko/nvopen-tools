// Function: sub_2E39BE0
// Address: 0x2e39be0
//
__int64 __fastcall sub_2E39BE0(_QWORD *a1)
{
  __int64 *v2; // rdi

  v2 = a1 + 25;
  *(v2 - 25) = (__int64)&unk_4A288F0;
  sub_2E39BC0(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
