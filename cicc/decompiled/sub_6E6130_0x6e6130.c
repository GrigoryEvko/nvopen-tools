// Function: sub_6E6130
// Address: 0x6e6130
//
__int64 __fastcall sub_6E6130(int a1, int a2, int a3, int a4)
{
  return sub_6E6080(
           a1,
           a2,
           a3,
           a4,
           (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0,
           ((*(_BYTE *)(qword_4D03C50 + 17LL) >> 6) ^ 1) & 1);
}
