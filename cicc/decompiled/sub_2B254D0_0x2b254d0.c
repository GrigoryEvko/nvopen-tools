// Function: sub_2B254D0
// Address: 0x2b254d0
//
unsigned __int64 *__fastcall sub_2B254D0(unsigned __int64 *a1)
{
  unsigned __int64 v1; // rax
  size_t v3; // rdx

  v1 = *a1;
  if ( (*a1 & 1) != 0 )
  {
    *a1 = v1 & 0xFC00000000000000LL | 1;
    return a1;
  }
  v3 = 8LL * *(unsigned int *)(v1 + 8);
  if ( !v3 )
    return a1;
  memset(*(void **)v1, 0, v3);
  return a1;
}
