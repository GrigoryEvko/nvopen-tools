// Function: ctor_060
// Address: 0x494cc0
//
int ctor_060()
{
  char v1; // [rsp+3h] [rbp-Dh] BYREF
  int v2; // [rsp+4h] [rbp-Ch] BYREF
  char *v3; // [rsp+8h] [rbp-8h] BYREF

  v1 = 1;
  v2 = 1;
  v3 = &v1;
  sub_DF51E0(&unk_4F89B60, "enable-scoped-noalias", &v3, &v2);
  return __cxa_atexit(sub_984900, &unk_4F89B60, &qword_4A427C0);
}
