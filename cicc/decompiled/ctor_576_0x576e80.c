// Function: ctor_576
// Address: 0x576e80
//
int ctor_576()
{
  int v1; // [rsp+4h] [rbp-1Ch] BYREF
  int *v2; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v3[2]; // [rsp+10h] [rbp-10h] BYREF

  v1 = 1;
  v2 = &v1;
  v3[0] = "Hoist consts in phis";
  v3[1] = 20;
  sub_2F26540(&unk_5022620, "hoistphiconsts", v3, &v2);
  return __cxa_atexit(sub_B2B680, &unk_5022620, &qword_4A427C0);
}
