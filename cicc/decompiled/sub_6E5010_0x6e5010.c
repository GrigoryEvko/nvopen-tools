// Function: sub_6E5010
// Address: 0x6e5010
//
__int64 __fastcall sub_6E5010(_BYTE *a1, _BYTE *a2)
{
  char v2; // al
  char v3; // al
  __int64 result; // rax

  v2 = a2[18] & 8 | a1[18] & 0xF7;
  a1[18] = v2;
  v3 = a2[18] & 0x10 | v2 & 0xEF;
  a1[18] = v3;
  a1[18] = a2[18] & 0x20 | v3 & 0xDF;
  a1[20] = a2[20] & 2 | a1[20] & 0xFD;
  result = a2[19] & 0x80 | a1[19] & 0x7Fu;
  a1[19] = a2[19] & 0x80 | a1[19] & 0x7F;
  return result;
}
