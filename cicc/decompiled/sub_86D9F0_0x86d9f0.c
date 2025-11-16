// Function: sub_86D9F0
// Address: 0x86d9f0
//
_BOOL8 sub_86D9F0()
{
  _BOOL8 result; // rax

  result = 0;
  if ( unk_4D03B90 != -1 )
    return (*(_BYTE *)(qword_4D03B98 + 176LL * unk_4D03B90 + 5) & 4) != 0;
  return result;
}
