// Function: ctor_685
// Address: 0x5a6170
//
int ctor_685()
{
  int v1; // [rsp+4h] [rbp-1Ch] BYREF
  void *v2; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v3[2]; // [rsp+10h] [rbp-10h] BYREF

  v3[0] = "Verify during register allocation";
  v3[1] = 33;
  v1 = 1;
  v2 = &unk_503FCFD;
  sub_B1A660(&unk_503FD00, "verify-regalloc", &v2, &v1, v3);
  return __cxa_atexit(sub_AA4490, &unk_503FD00, &qword_4A427C0);
}
