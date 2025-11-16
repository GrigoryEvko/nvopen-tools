// Function: sub_BC2B00
// Address: 0xbc2b00
//
__int128 *sub_BC2B00()
{
  if ( byte_4F82430 )
    return &xmmword_4F82440;
  if ( (unsigned int)sub_2207590(&byte_4F82430) )
  {
    xmmword_4F82440 = 0;
    xmmword_4F82450 = 0;
    xmmword_4F82460 = 0;
    qword_4F82470 = 0;
    qword_4F82478 = 0;
    qword_4F82480 = 0;
    qword_4F82488 = 0;
    qword_4F82490 = 0;
    dword_4F82498 = 0;
    qword_4F824A0 = 0;
    qword_4F824A8 = 0;
    qword_4F824B0 = 0x1000000000LL;
    qword_4F824B8 = 0;
    qword_4F824C0 = 0;
    qword_4F824C8 = 0;
    qword_4F824D0 = 0;
    qword_4F824D8 = 0;
    qword_4F824E0 = 0;
    __cxa_atexit((void (*)(void *))sub_BC2A10, &xmmword_4F82440, &qword_4A427C0);
    sub_2207640(&byte_4F82430);
  }
  return &xmmword_4F82440;
}
