// Function: ctor_045
// Address: 0x48fc30
//
int ctor_045()
{
  char v1; // [rsp+3h] [rbp-Dh] BYREF
  int v2; // [rsp+4h] [rbp-Ch] BYREF
  char *v3; // [rsp+8h] [rbp-8h] BYREF

  v1 = 0;
  v3 = &v1;
  v2 = 1;
  sub_CF7670(&unk_4F86560, "disable-basic-aa", &v2, &v3);
  return __cxa_atexit(sub_984900, &unk_4F86560, &qword_4A427C0);
}
