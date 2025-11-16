// Function: sub_13E6580
// Address: 0x13e6580
//
__int64 __fastcall sub_13E6580(_QWORD *a1)
{
  __int64 *v2; // rdi

  v2 = a1 + 20;
  *(v2 - 20) = (__int64)&unk_49EA7A0;
  sub_1368A00(v2);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
