// Function: sub_721990
// Address: 0x721990
//
int __fastcall sub_721990(void *a1, size_t a2)
{
  int result; // eax

  result = munmap(a1, a2);
  if ( result )
    sub_721090();
  return result;
}
