// Function: sub_23C67A0
// Address: 0x23c67a0
//
__int128 *sub_23C67A0()
{
  if ( byte_4FDF328 )
    return &xmmword_4FDF340;
  if ( (unsigned int)sub_2207590((__int64)&byte_4FDF328) )
  {
    qword_4FDF360 = 0;
    xmmword_4FDF350 = 0;
    xmmword_4FDF340 = 0;
    LODWORD(xmmword_4FDF350) = 1;
    dword_4FDF368 = 0;
    qword_4FDF370 = 0;
    qword_4FDF378 = 0;
    qword_4FDF380 = 0;
    __cxa_atexit(sub_23C6910, &xmmword_4FDF340, &qword_4A427C0);
    sub_2207640((__int64)&byte_4FDF328);
  }
  return &xmmword_4FDF340;
}
