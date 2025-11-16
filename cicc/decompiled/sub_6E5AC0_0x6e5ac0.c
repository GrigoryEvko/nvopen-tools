// Function: sub_6E5AC0
// Address: 0x6e5ac0
//
_BOOL8 sub_6E5AC0()
{
  _BOOL8 result; // rax

  result = 0;
  if ( qword_4D03C50 )
    return (*(_BYTE *)(qword_4D03C50 + 24LL) & 3) != 0;
  return result;
}
