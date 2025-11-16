// Function: ctor_054
// Address: 0x4920c0
//
int ctor_054()
{
  int v1; // [rsp+4h] [rbp-1Ch] BYREF
  void *v2; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v3[2]; // [rsp+10h] [rbp-10h] BYREF

  v3[0] = "Verify loop info (time consuming)";
  v3[1] = 33;
  v1 = 1;
  v2 = &unk_4F876C8;
  sub_D4ADB0(&unk_4F87600, "verify-loop-info", &v2, &v1, v3);
  return __cxa_atexit(sub_AA4490, &unk_4F87600, &qword_4A427C0);
}
