// Function: ctor_051
// Address: 0x490930
//
int ctor_051()
{
  char v1; // [rsp+3h] [rbp-Dh] BYREF
  int v2; // [rsp+4h] [rbp-Ch] BYREF
  char *v3; // [rsp+8h] [rbp-8h] BYREF

  v1 = 0;
  v2 = 1;
  v3 = &v1;
  sub_D1DA90(&unk_4F86B80, "enable-unsafe-globalsmodref-alias-results", &v3, &v2);
  return __cxa_atexit(sub_984900, &unk_4F86B80, &qword_4A427C0);
}
