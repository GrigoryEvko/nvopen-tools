// Function: ctor_592
// Address: 0x57cca0
//
int ctor_592()
{
  char v1; // [rsp+7h] [rbp-19h] BYREF
  char *v2; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v3[2]; // [rsp+10h] [rbp-10h] BYREF

  v1 = 1;
  v2 = &v1;
  v3[0] = "Enable loop iv regalloc heuristic";
  v3[1] = 33;
  sub_2FB2D00(&unk_5025C40, "enable-split-loopiv-heuristic", v3, &v2);
  return __cxa_atexit(sub_984900, &unk_5025C40, &qword_4A427C0);
}
