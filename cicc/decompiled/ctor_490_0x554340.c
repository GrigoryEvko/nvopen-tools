// Function: ctor_490
// Address: 0x554340
//
int ctor_490()
{
  char v1; // [rsp+7h] [rbp-19h] BYREF
  char *v2; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v3[2]; // [rsp+10h] [rbp-10h] BYREF

  v3[0] = "Disable generation of discriminator information.";
  v3[1] = 48;
  v1 = 0;
  v2 = &v1;
  sub_299FD60(&unk_5007960, "no-discriminators", &v2, v3);
  return __cxa_atexit(sub_984900, &unk_5007960, &qword_4A427C0);
}
