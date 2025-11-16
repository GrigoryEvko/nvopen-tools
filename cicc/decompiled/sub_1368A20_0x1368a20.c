// Function: sub_1368A20
// Address: 0x1368a20
//
__int64 __fastcall sub_1368A20(_QWORD *a1)
{
  __int64 *v2; // rdi

  v2 = a1 + 20;
  *(v2 - 20) = (__int64)&unk_49E89A8;
  sub_1368A00(v2);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
