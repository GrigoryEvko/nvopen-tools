// Function: sub_15A5120
// Address: 0x15a5120
//
__int64 __fastcall sub_15A5120(__int64 a1)
{
  __int64 result; // rax

  result = (*(_BYTE *)(a1 + 32) & 0xFu) - 7;
  if ( (unsigned int)result <= 1 || (*(_BYTE *)(a1 + 32) & 0x30) != 0 && (*(_BYTE *)(a1 + 32) & 0xF) != 9 )
    *(_BYTE *)(a1 + 33) |= 0x40u;
  return result;
}
