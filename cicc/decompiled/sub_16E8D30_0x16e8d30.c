// Function: sub_16E8D30
// Address: 0x16e8d30
//
__int64 *sub_16E8D30()
{
  if ( byte_4FA16C8 )
    return &qword_4FA16E0;
  if ( (unsigned int)sub_2207590(&byte_4FA16C8) )
  {
    dword_4FA1700 = 0;
    qword_4FA16F8 = 0;
    qword_4FA16F0 = 0;
    qword_4FA16E8 = 0;
    qword_4FA16E0 = (__int64)&unk_49EFCB8;
    __cxa_atexit((void (*)(void *))sub_16E79F0, &qword_4FA16E0, &qword_4A427C0);
    sub_2207640(&byte_4FA16C8);
  }
  return &qword_4FA16E0;
}
