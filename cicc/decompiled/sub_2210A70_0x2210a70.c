// Function: sub_2210A70
// Address: 0x2210a70
//
void sub_2210A70(void)
{
  if ( !byte_4FD65F0 && (unsigned int)sub_2207590((__int64)&byte_4FD65F0) )
  {
    pthread_key_create(&dword_4FD65FC, (void (*)(void *))sub_22109F0);
    __cxa_atexit(sub_2210A60, &unk_4FD65F8, &qword_4A427C0);
    sub_2207640((__int64)&byte_4FD65F0);
  }
  sub_39FAD40(sub_2210A20);
}
