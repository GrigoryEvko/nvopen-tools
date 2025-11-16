// Function: sub_730580
// Address: 0x730580
//
__int64 __fastcall sub_730580(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 result; // rax

  v2 = *(_BYTE *)(a1 + 25) & 1 | *(_BYTE *)(a2 + 25) & 0xFE;
  *(_BYTE *)(a2 + 25) = v2;
  result = *(_BYTE *)(a1 + 25) & 2 | v2 & 0xFFFFFFFD;
  *(_BYTE *)(a2 + 25) = result;
  return result;
}
