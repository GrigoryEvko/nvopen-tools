// Function: sub_16C6880
// Address: 0x16c6880
//
__int64 sub_16C6880()
{
  if ( byte_4FA0550 )
    return (unsigned int)dword_4FA0558;
  if ( (unsigned int)sub_2207590(&byte_4FA0550) )
  {
    dword_4FA0558 = getpagesize();
    sub_2207640(&byte_4FA0550);
  }
  return (unsigned int)dword_4FA0558;
}
