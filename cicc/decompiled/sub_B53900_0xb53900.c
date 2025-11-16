// Function: sub_B53900
// Address: 0xb53900
//
unsigned __int64 __fastcall sub_B53900(__int64 a1)
{
  unsigned __int64 result; // rax

  result = *(_WORD *)(a1 + 2) & 0x3F;
  if ( *(_BYTE *)a1 == 82 )
    return *(_WORD *)(a1 + 2) & 0x3F | ((unsigned __int64)((*(_BYTE *)(a1 + 1) & 2) != 0) << 32);
  return result;
}
