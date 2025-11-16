// Function: ctor_612
// Address: 0x589370
//
int ctor_612()
{
  int v1; // [rsp+4h] [rbp-1Ch] BYREF
  void *v2; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v3[2]; // [rsp+10h] [rbp-10h] BYREF

  v3[1] = 76;
  v2 = &unk_502D428;
  v3[0] = "NVPTX Specific: Programmer asserts that any kernel ptr parameter is restrict";
  v1 = 1;
  sub_3085AC0(&unk_502D360, "nvptx-kernel-params-restrict", &v1, v3, &v2);
  return __cxa_atexit(sub_AA4490, &unk_502D360, &qword_4A427C0);
}
