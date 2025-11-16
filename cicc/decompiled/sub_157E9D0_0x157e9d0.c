// Function: sub_157E9D0
// Address: 0x157e9d0
//
__int64 __fastcall sub_157E9D0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_15F2040(a2, a1 - 40);
  if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
  {
    result = sub_157E9B0(a1 - 40);
    if ( result )
      return sub_164D6D0(result, a2);
  }
  return result;
}
