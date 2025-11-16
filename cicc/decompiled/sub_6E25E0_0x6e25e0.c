// Function: sub_6E25E0
// Address: 0x6e25e0
//
__int64 __fastcall sub_6E25E0(char a1, char a2)
{
  __int64 v2; // rdx
  __int64 result; // rax
  char v4; // si

  v2 = qword_4D03C50;
  result = (unsigned __int8)(a2 & 1) << 6;
  v4 = *(_BYTE *)(qword_4D03C50 + 19LL);
  *(_BYTE *)(qword_4D03C50 + 16LL) = a1;
  *(_BYTE *)(v2 + 19) = result | v4 & 0xBF;
  return result;
}
