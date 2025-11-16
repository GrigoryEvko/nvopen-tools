// Function: sub_154A4E0
// Address: 0x154a4e0
//
__int64 __fastcall sub_154A4E0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = (*(_BYTE *)(a1 + 32) & 0xFu) - 7;
  if ( (unsigned int)result > 1
    && ((*(_BYTE *)(a1 + 32) & 0xF) == 9 || (*(_BYTE *)(a1 + 32) & 0x30) == 0)
    && (*(_BYTE *)(a1 + 33) & 0x40) != 0 )
  {
    return sub_1263B40(a2, "dso_local ");
  }
  return result;
}
