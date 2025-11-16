// Function: sub_6ED0A0
// Address: 0x6ed0a0
//
_BOOL8 __fastcall sub_6ED0A0(__int64 a1)
{
  _BOOL8 result; // rax

  result = 0;
  if ( *(_BYTE *)(a1 + 16) == 1 )
    return (*(_BYTE *)(*(_QWORD *)(a1 + 144) + 25LL) & 2) != 0;
  return result;
}
