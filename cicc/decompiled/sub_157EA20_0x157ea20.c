// Function: sub_157EA20
// Address: 0x157ea20
//
__int64 __fastcall sub_157EA20(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v4; // rax

  result = sub_15F2040(a2, 0);
  if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
  {
    result = sub_157E9B0(a1 - 40);
    v3 = result;
    if ( result )
    {
      v4 = sub_16498B0(a2);
      return sub_164D860(v3, v4);
    }
  }
  return result;
}
