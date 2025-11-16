// Function: sub_987100
// Address: 0x987100
//
void *__fastcall sub_987100(__int64 a1)
{
  void *result; // rax

  result = (void *)*(unsigned int *)(a1 + 8);
  if ( (unsigned int)result > 0x40 )
    return memset(*(void **)a1, 0, 8 * (((unsigned __int64)(unsigned int)result + 63) >> 6));
  *(_QWORD *)a1 = 0;
  return result;
}
