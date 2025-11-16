// Function: sub_130C9C0
// Address: 0x130c9c0
//
void *__fastcall sub_130C9C0(void *addr, size_t len, _BYTE *a3)
{
  int v3; // edx
  void *result; // rax

  if ( byte_4F969C0 )
  {
    *a3 = 1;
    v3 = 3;
  }
  else
  {
    v3 = *a3 != 0 ? 3 : 0;
  }
  result = mmap(addr, len, v3, flags, -1, 0);
  if ( result == (void *)-1LL )
    return 0;
  if ( addr )
  {
    if ( addr != result )
    {
      sub_130C960(result, len);
      return 0;
    }
  }
  return result;
}
