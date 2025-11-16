// Function: ctor_677
// Address: 0x5a44b0
//
int ctor_677()
{
  char v1; // [rsp+7h] [rbp-19h] BYREF
  char *v2; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v3[2]; // [rsp+10h] [rbp-10h] BYREF

  v1 = 1;
  v2 = &v1;
  v3[0] = "Should mir-strip-debug only strip debug info from debugified modules by default";
  v3[1] = 79;
  sub_3573C00(&unk_503EF00, "mir-strip-debugify-only", v3, &v2);
  return __cxa_atexit(sub_984900, &unk_503EF00, &qword_4A427C0);
}
