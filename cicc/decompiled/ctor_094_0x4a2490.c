// Function: ctor_094
// Address: 0x4a2490
//
int ctor_094()
{
  int v1; // [rsp+4h] [rbp-1Ch] BYREF
  void *v2; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v3[2]; // [rsp+10h] [rbp-10h] BYREF

  v3[0] = "Verify loop lcssa form (time consuming)";
  v3[1] = 39;
  v1 = 1;
  v2 = &unk_4F90F08;
  sub_11CE220(&unk_4F90E40, "verify-loop-lcssa", &v2, &v1, v3);
  return __cxa_atexit(sub_AA4490, &unk_4F90E40, &qword_4A427C0);
}
