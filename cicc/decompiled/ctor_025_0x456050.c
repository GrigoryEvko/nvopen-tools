// Function: ctor_025
// Address: 0x456050
//
int ctor_025()
{
  char v1; // [rsp+7h] [rbp-29h] BYREF
  char *v2; // [rsp+8h] [rbp-28h] BYREF
  _QWORD v3[4]; // [rsp+10h] [rbp-20h] BYREF

  v2 = &v1;
  v1 = 1;
  v3[0] = "Write debug info in the new non-intrinsic format. Has no effect if --preserve-input-debuginfo-format=true.";
  v3[1] = 106;
  sub_B3AC20(&unk_4F81700, "write-experimental-debuginfo", v3, &v2);
  return __cxa_atexit(sub_984900, &unk_4F81700, &qword_4A427C0);
}
