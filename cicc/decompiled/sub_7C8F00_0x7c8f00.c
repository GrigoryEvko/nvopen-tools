// Function: sub_7C8F00
// Address: 0x7c8f00
//
_BOOL8 __fastcall sub_7C8F00(char *s2)
{
  _BOOL8 result; // rax
  const char *v2; // r8

  result = 0;
  if ( word_4F06418[0] == 1 && qword_4D04A00 )
  {
    v2 = *(const char **)(qword_4D04A00 + 8);
    if ( *v2 == *s2 )
      return strcmp(v2, s2) == 0;
  }
  return result;
}
