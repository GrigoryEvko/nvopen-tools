// Function: sub_A518A0
// Address: 0xa518a0
//
__int64 __fastcall sub_A518A0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( (*(_BYTE *)(a1 + 33) & 0x40) != 0 )
  {
    result = (*(_BYTE *)(a1 + 32) & 0xFu) - 7;
    if ( (unsigned int)result > 1 && ((*(_BYTE *)(a1 + 32) & 0x30) == 0 || (*(_BYTE *)(a1 + 32) & 0xF) == 9) )
      return sub_904010(a2, "dso_local ");
  }
  return result;
}
