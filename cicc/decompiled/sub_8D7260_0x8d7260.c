// Function: sub_8D7260
// Address: 0x8d7260
//
_BOOL8 __fastcall sub_8D7260(char a1, char a2, int a3)
{
  _BOOL8 result; // rax

  if ( !a3 || (result = 1, !unk_4D04334) )
  {
    result = 1;
    if ( unk_4F06904 )
      return a1 == a2;
  }
  return result;
}
