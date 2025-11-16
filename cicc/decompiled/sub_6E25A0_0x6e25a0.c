// Function: sub_6E25A0
// Address: 0x6e25a0
//
__int64 __fastcall sub_6E25A0(_BYTE *a1, _DWORD *a2)
{
  __int64 result; // rax

  *a1 = *(_BYTE *)(qword_4D03C50 + 16LL);
  *a2 = (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0;
  result = *(_DWORD *)(qword_4D03C50 + 16LL) & 0xBFFFFF00 | 4;
  *(_DWORD *)(qword_4D03C50 + 16LL) = result;
  return result;
}
