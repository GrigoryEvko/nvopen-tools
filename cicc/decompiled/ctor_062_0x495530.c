// Function: ctor_062
// Address: 0x495530
//
int ctor_062()
{
  char v1; // [rsp+3h] [rbp-Dh] BYREF
  int v2; // [rsp+4h] [rbp-Ch] BYREF
  char *v3; // [rsp+8h] [rbp-8h] BYREF

  v1 = 1;
  v2 = 1;
  v3 = &v1;
  sub_E01060(&unk_4F89FC0, "enable-tbaa", &v3, &v2);
  return __cxa_atexit(sub_984900, &unk_4F89FC0, &qword_4A427C0);
}
