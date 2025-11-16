// Function: sub_A6DEB0
// Address: 0xa6deb0
//
__int64 __fastcall sub_A6DEB0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rax

  result = sub_B2D620(a1, "probe-stack", 11);
  if ( !(_BYTE)result )
  {
    result = sub_B2D620(a2, "probe-stack", 11);
    if ( (_BYTE)result )
    {
      v3 = sub_B2D7E0(a2, "probe-stack", 11);
      return sub_B2CDC0(a1, v3);
    }
  }
  return result;
}
