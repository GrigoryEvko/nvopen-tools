// Function: sub_67D850
// Address: 0x67d850
//
_BOOL8 __fastcall sub_67D850(int a1, char a2, int a3)
{
  _BOOL8 result; // rax

  result = (unsigned int)(a1 - 1) > 0xED1;
  if ( (unsigned int)(a1 - 1) <= 0xED1 )
  {
    if ( a2 )
    {
      if ( a2 == 1 )
      {
        byte_4CFFE80[4 * a1 + 2] |= 1u;
      }
      else
      {
        byte_4CFFE80[4 * a1 + 1] = a2;
        if ( a3 )
          byte_4CFFE80[4 * a1] = a2;
      }
    }
    else
    {
      byte_4CFFE80[4 * a1 + 1] = byte_4CFFE80[4 * a1];
    }
  }
  return result;
}
