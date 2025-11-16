// Function: ctor_393
// Address: 0x520980
//
int ctor_393()
{
  _QWORD v1[2]; // [rsp+0h] [rbp-10h] BYREF

  v1[0] = "Use one trap block per function";
  v1[1] = 31;
  sub_23F72B0(&unk_4FE26C0, "bounds-checking-single-trap", v1);
  return __cxa_atexit(sub_984900, &unk_4FE26C0, &qword_4A427C0);
}
