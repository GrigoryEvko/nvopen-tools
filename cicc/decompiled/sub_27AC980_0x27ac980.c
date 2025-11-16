// Function: sub_27AC980
// Address: 0x27ac980
//
__int64 *sub_27AC980()
{
  if ( byte_4FFC580 )
    return &qword_4FFC5A0;
  if ( (unsigned int)sub_2207590((__int64)&byte_4FFC580) )
  {
    qword_4FFC5B0 = 0;
    qword_4FFC5A0 = (__int64)&qword_4FFC5B0;
    qword_4FFC5D0 = (__int64)algn_4FFC5E0;
    qword_4FFC5D8 = 0x400000000LL;
    qword_4FFC5A8 = 0x400000001LL;
    __cxa_atexit((void (*)(void *))sub_27ABC80, &qword_4FFC5A0, &qword_4A427C0);
    sub_2207640((__int64)&byte_4FFC580);
  }
  return &qword_4FFC5A0;
}
