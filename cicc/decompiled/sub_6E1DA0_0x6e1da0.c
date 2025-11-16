// Function: sub_6E1DA0
// Address: 0x6e1da0
//
__int64 __fastcall sub_6E1DA0(__int64 a1, __int64 a2)
{
  return sub_73F570(
           a1,
           a2,
           (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x10) != 0,
           (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0,
           *(_BYTE *)(qword_4D03C50 + 17LL) & 1);
}
