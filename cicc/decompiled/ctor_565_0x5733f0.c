// Function: ctor_565
// Address: 0x5733f0
//
int ctor_565()
{
  int v1; // [rsp+4h] [rbp-1Ch] BYREF
  void *v2; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v3[2]; // [rsp+10h] [rbp-10h] BYREF

  v3[0] = "Verify machine dominator info (time consuming)";
  v3[1] = 46;
  v1 = 1;
  v2 = &unk_501FF28;
  sub_2E6D5F0(&unk_501FE60, "verify-machine-dom-info", &v2, &v1, v3);
  return __cxa_atexit(sub_AA4490, &unk_501FE60, &qword_4A427C0);
}
