// Function: sub_7B80B0
// Address: 0x7b80b0
//
_BOOL8 __fastcall sub_7B80B0(unsigned __int16 a1)
{
  _BOOL8 result; // rax

  result = 0;
  if ( a1 > 0x18u )
    return dword_4F05DC0[*(char *)*(&off_4B6DFA0 + a1) + 128] != 0;
  return result;
}
