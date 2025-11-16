// Function: sub_254E6E0
// Address: 0x254e6e0
//
__int64 __fastcall sub_254E6E0(__int64 a1)
{
  __int64 result; // rax

  result = (unsigned int)*(unsigned __int8 *)sub_250D070((_QWORD *)(a1 + 72)) - 12;
  if ( (unsigned int)result <= 1 )
  {
    result = *(unsigned __int8 *)(a1 + 96);
    *(_BYTE *)(a1 + 97) = result;
  }
  return result;
}
