// Function: ctor_627
// Address: 0x58f520
//
int ctor_627()
{
  char v1; // [rsp+7h] [rbp-19h] BYREF
  char *v2; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v3[2]; // [rsp+10h] [rbp-10h] BYREF

  v3[0] = "In the Lint pass, abort on errors.";
  v3[1] = 34;
  v1 = 0;
  v2 = &v1;
  sub_B56270(&unk_5031240, "lint-abort-on-error", &v2, v3);
  return __cxa_atexit(sub_984900, &unk_5031240, &qword_4A427C0);
}
