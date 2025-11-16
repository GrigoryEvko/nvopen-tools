// Function: sub_732AF0
// Address: 0x732af0
//
__int64 __fastcall sub_732AF0(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a1 + 193);
  if ( (result & 0x20) == 0 )
  {
    *(_BYTE *)(a1 + 193) = result | 0x20;
    return sub_760730();
  }
  return result;
}
