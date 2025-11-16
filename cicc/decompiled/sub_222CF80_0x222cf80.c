// Function: sub_222CF80
// Address: 0x222cf80
//
void __noreturn sub_222CF80(char *s, ...)
{
  void *v1; // rsp
  __int64 v2; // rax
  __int64 v3; // [rsp+0h] [rbp-E0h] BYREF
  gcc_va_list va; // [rsp+8h] [rbp-D8h] BYREF

  va_start(va, s);
  v1 = alloca(strlen(s) + 520);
  sub_223EFE0(&v3);
  v2 = sub_2252770(16);
  sub_2257230(v2, &v3);
  JUMPOUT(0x426368);
}
