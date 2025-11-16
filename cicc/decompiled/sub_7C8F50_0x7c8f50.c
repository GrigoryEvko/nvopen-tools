// Function: sub_7C8F50
// Address: 0x7c8f50
//
_BOOL8 __fastcall sub_7C8F50(unsigned __int16 a1, char *a2)
{
  if ( !sub_7C8F00(a2) )
    return word_4F06418[0] == a1;
  word_4F06418[0] = a1;
  return 1;
}
