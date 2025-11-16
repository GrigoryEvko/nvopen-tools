// Function: sub_27ACA30
// Address: 0x27aca30
//
__int64 *sub_27ACA30()
{
  if ( byte_4FFC508 )
    return &qword_4FFC520;
  if ( (unsigned int)sub_2207590((__int64)&byte_4FFC508) )
  {
    qword_4FFC530 = 1;
    qword_4FFC520 = (__int64)&qword_4FFC530;
    qword_4FFC550 = (__int64)algn_4FFC560;
    qword_4FFC558 = 0x400000000LL;
    qword_4FFC528 = 0x400000001LL;
    __cxa_atexit((void (*)(void *))sub_27ABC80, &qword_4FFC520, &qword_4A427C0);
    sub_2207640((__int64)&byte_4FFC508);
  }
  return &qword_4FFC520;
}
