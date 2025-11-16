// Function: sub_C95C80
// Address: 0xc95c80
//
__int128 *sub_C95C80()
{
  if ( byte_4F84F18 )
    return &xmmword_4F84F20;
  if ( (unsigned int)sub_2207590(&byte_4F84F18) )
  {
    qword_4F84F40 = 0;
    xmmword_4F84F20 = 0;
    xmmword_4F84F30 = 0;
    qword_4F84F48 = 0;
    qword_4F84F50 = 0;
    qword_4F84F58 = 0;
    __cxa_atexit((void (*)(void *))sub_C95B90, &xmmword_4F84F20, &qword_4A427C0);
    sub_2207640(&byte_4F84F18);
  }
  return &xmmword_4F84F20;
}
