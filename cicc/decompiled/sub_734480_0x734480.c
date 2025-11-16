// Function: sub_734480
// Address: 0x734480
//
_BOOL8 __fastcall sub_734480(__int64 a1)
{
  _BOOL8 result; // rax
  __int64 v2; // rdx
  char v3; // al

  result = 0;
  if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
  {
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
    v3 = *(_BYTE *)(v2 + 140);
    if ( (unsigned __int8)(v3 - 9) > 2u )
      return v3 == 14;
    else
      return (*(_BYTE *)(v2 + 177) & 0x20) != 0;
  }
  return result;
}
