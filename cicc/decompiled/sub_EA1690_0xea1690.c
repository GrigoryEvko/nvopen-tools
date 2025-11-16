// Function: sub_EA1690
// Address: 0xea1690
//
__int64 __fastcall sub_EA1690(__int64 a1, unsigned int a2)
{
  __int64 result; // rax

  result = *(_WORD *)(a1 + 12) & 0xFC7F;
  *(_WORD *)(a1 + 12) = *(_WORD *)(a1 + 12) & 0xFC7F | ((unsigned __int16)(a2 >> 5) << 7);
  return result;
}
