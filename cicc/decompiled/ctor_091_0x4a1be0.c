// Function: ctor_091
// Address: 0x4a1be0
//
int ctor_091()
{
  int v1; // [rsp+4h] [rbp-1Ch] BYREF
  int *v2; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v3[2]; // [rsp+10h] [rbp-10h] BYREF

  v3[0] = "Maximum number phis to handle in intptr/ptrint folding";
  v3[1] = 54;
  v1 = 512;
  v2 = &v1;
  sub_116DF60(&unk_4F909E0, "instcombine-max-num-phis", &v2, v3);
  return __cxa_atexit(sub_984970, &unk_4F909E0, &qword_4A427C0);
}
