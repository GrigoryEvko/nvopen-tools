// Function: sub_851CB0
// Address: 0x851cb0
//
__int64 __fastcall sub_851CB0(const char *ptr)
{
  FILE *v1; // r13
  __int64 result; // rax
  size_t ptra[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = qword_4F5FB40;
  if ( ptr )
  {
    ptra[0] = strlen(ptr) + 1;
    fwrite(ptra, 8u, 1u, v1);
    result = fwrite(ptr, ptra[0], 1u, qword_4F5FB40);
    if ( result != 1 )
      sub_851C80();
  }
  else
  {
    ptra[0] = 0;
    return fwrite(ptra, 8u, 1u, qword_4F5FB40);
  }
  return result;
}
