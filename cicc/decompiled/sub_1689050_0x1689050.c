// Function: sub_1689050
// Address: 0x1689050
//
void *sub_1689050()
{
  void *result; // rax
  const void *v1; // r12

  if ( !unk_4F9F820 || (result = pthread_getspecific(dword_4F9F868)) == 0 )
  {
    v1 = (const void *)sub_1688F80();
    pthread_setspecific(dword_4F9F868, v1);
    return (void *)v1;
  }
  return result;
}
