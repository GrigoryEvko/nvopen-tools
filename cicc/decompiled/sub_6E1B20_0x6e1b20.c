// Function: sub_6E1B20
// Address: 0x6e1b20
//
_BOOL8 __fastcall sub_6E1B20(__int64 a1)
{
  _BOOL8 result; // rax

  result = 0;
  if ( !*(_BYTE *)(a1 + 8) )
    return (*(_BYTE *)(*(_QWORD *)(a1 + 24) + 28LL) & 2) != 0;
  return result;
}
