// Function: ctor_026
// Address: 0x4560c0
//
int ctor_026()
{
  char v1; // [rsp+7h] [rbp-19h] BYREF
  char *v2; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v3[2]; // [rsp+10h] [rbp-10h] BYREF

  v3[0] = "Disables inttoptr/ptrtoint roundtrip optimization";
  v3[1] = 49;
  v1 = 0;
  v2 = &v1;
  sub_B56270(&unk_4F81820, "disable-i2p-p2i-opt", &v2, v3);
  return __cxa_atexit(sub_984900, &unk_4F81820, &qword_4A427C0);
}
