// Function: sub_CB72A0
// Address: 0xcb72a0
//
void *sub_CB72A0()
{
  if ( byte_4F85050 )
    return &unk_4F85060;
  if ( (unsigned int)sub_2207590(&byte_4F85050) )
  {
    sub_CB6EE0((__int64)&unk_4F85060, 2, 0, 1u, 0);
    __cxa_atexit((void (*)(void *))sub_CB5B00, &unk_4F85060, &qword_4A427C0);
    sub_2207640(&byte_4F85050);
  }
  return &unk_4F85060;
}
