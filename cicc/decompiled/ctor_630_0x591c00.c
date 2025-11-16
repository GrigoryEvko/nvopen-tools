// Function: ctor_630
// Address: 0x591c00
//
int ctor_630()
{
  char v1; // [rsp+Bh] [rbp-25h] BYREF
  int v2; // [rsp+Ch] [rbp-24h] BYREF
  char *v3; // [rsp+10h] [rbp-20h] BYREF
  void *v4; // [rsp+18h] [rbp-18h] BYREF
  _QWORD v5[2]; // [rsp+20h] [rbp-10h] BYREF

  v2 = 1;
  v3 = &v1;
  v1 = 1;
  v4 = &unk_5031DC8;
  v5[0] = "enable/disable all ARC Optimizations";
  v5[1] = 36;
  sub_3108540(&unk_5031D00, "enable-objc-arc-opts", v5, &v4, &v3, &v2);
  return __cxa_atexit(sub_AA4490, &unk_5031D00, &qword_4A427C0);
}
