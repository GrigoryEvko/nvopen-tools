// Function: sub_1341200
// Address: 0x1341200
//
unsigned __int8 __fastcall sub_1341200(void *s, size_t n)
{
  unsigned __int8 result; // al

  if ( unk_4F969BC == 1 )
    return (unsigned __int8)memset(s, 0, n);
  result = sub_130CD80(s, n);
  if ( result )
    return (unsigned __int8)memset(s, 0, n);
  return result;
}
