// Function: ctor_022
// Address: 0x455bb0
//
int ctor_022()
{
  int v1; // [rsp+4h] [rbp-1Ch] BYREF
  void *v2; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v3[2]; // [rsp+10h] [rbp-10h] BYREF

  v3[0] = "Verify dominator info (time consuming)";
  v3[1] = 38;
  v1 = 1;
  v2 = &unk_4F81528;
  sub_B1A660(&unk_4F81460, "verify-dom-info", &v2, &v1, v3);
  return __cxa_atexit(sub_AA4490, &unk_4F81460, &qword_4A427C0);
}
