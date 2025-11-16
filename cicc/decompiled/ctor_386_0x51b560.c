// Function: ctor_386
// Address: 0x51b560
//
int ctor_386()
{
  char v1; // [rsp+7h] [rbp-9h] BYREF
  char *v2; // [rsp+8h] [rbp-8h] BYREF

  v1 = 0;
  v2 = &v1;
  sub_22ED340(&unk_4FDC1A0, "safepoint-ir-verifier-print-only", &v2);
  return __cxa_atexit(sub_984900, &unk_4FDC1A0, &qword_4A427C0);
}
