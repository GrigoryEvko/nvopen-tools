// Function: sub_16E8CB0
// Address: 0x16e8cb0
//
void *sub_16E8CB0()
{
  __int64 v1; // r8

  if ( byte_4FA1708 )
    return &unk_4FA1720;
  if ( (unsigned int)sub_2207590(&byte_4FA1708) )
  {
    sub_16E8970((__int64)&unk_4FA1720, 2, 0, 1u, v1);
    __cxa_atexit((void (*)(void *))sub_16E7C30, &unk_4FA1720, &qword_4A427C0);
    sub_2207640(&byte_4FA1708);
  }
  return &unk_4FA1720;
}
