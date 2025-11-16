// Function: sub_CE8040
// Address: 0xce8040
//
unsigned __int64 __fastcall sub_CE8040(__int64 a1, char a2)
{
  unsigned __int64 result; // rax

  if ( a2 )
  {
    result = sub_B2D620(a1, "nvvm.kernel", 0xBu);
    if ( !(_BYTE)result )
      return sub_B2CD60(a1, "nvvm.kernel", 0xBu, 0, 0);
  }
  else
  {
    result = sub_B2D620(a1, "nvvm.kernel", 0xBu);
    if ( (_BYTE)result )
      return sub_B2D4A0(a1, "nvvm.kernel", 0xBu);
  }
  return result;
}
