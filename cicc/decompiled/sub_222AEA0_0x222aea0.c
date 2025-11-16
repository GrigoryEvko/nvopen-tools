// Function: sub_222AEA0
// Address: 0x222aea0
//
__off64_t __fastcall sub_222AEA0(FILE **a1, __off64_t a2)
{
  char *IO_write_base; // rax
  int v3; // r8d
  __off64_t result; // rax

  IO_write_base = (*a1)->_IO_write_base;
  if ( IO_write_base != (char *)sub_222ABE0 )
    return ((__int64 (__fastcall *)(FILE **, __off64_t, _QWORD))IO_write_base)(a1, a2, 0);
  v3 = fseeko64(a1[8], a2, 0);
  result = -1;
  if ( !v3 )
    return ftello64(a1[8]);
  return result;
}
