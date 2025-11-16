// Function: ctor_646
// Address: 0x5970e0
//
int ctor_646()
{
  char v1; // [rsp+3h] [rbp-Dh] BYREF
  int v2; // [rsp+4h] [rbp-Ch] BYREF
  char *v3; // [rsp+8h] [rbp-8h] BYREF

  v1 = 1;
  v3 = &v1;
  v2 = 1;
  sub_3212250(&unk_50362A0, "trim-var-locs", &v2, &v3);
  return __cxa_atexit(sub_984900, &unk_50362A0, &qword_4A427C0);
}
