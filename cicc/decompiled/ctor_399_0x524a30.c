// Function: ctor_399
// Address: 0x524a30
//
int ctor_399()
{
  int v1; // [rsp+4h] [rbp-1Ch] BYREF
  const char *v2; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v3[2]; // [rsp+10h] [rbp-10h] BYREF

  v3[0] = "Dump functions and their MD5 hash to deobfuscate profile data";
  v1 = 1;
  v3[1] = 61;
  v2 = byte_3F871B3;
  sub_244E2F0(&unk_4FE6320, "orderfile-write-mapping", &v2, v3, &v1);
  return __cxa_atexit(sub_BC5A40, &unk_4FE6320, &qword_4A427C0);
}
