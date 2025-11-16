// Function: sub_16C1C80
// Address: 0x16c1c80
//
_QWORD *__fastcall sub_16C1C80(_QWORD *a1, char *a2)
{
  char *v2; // r14
  char v3; // al
  _QWORD v5[2]; // [rsp+10h] [rbp-80h] BYREF
  char v6; // [rsp+20h] [rbp-70h]
  _QWORD v7[4]; // [rsp+30h] [rbp-60h] BYREF
  int v8; // [rsp+50h] [rbp-40h]
  _QWORD *v9; // [rsp+58h] [rbp-38h]

  v2 = a2;
  *a1 = a1 + 2;
  a1[1] = 0x2000000000LL;
  v9 = a1;
  v7[0] = &unk_49EFC48;
  v8 = 1;
  memset(&v7[1], 0, 24);
  sub_16E7A40(v7, 0, 0, 0);
  do
  {
    v3 = *v2;
    v5[1] = "%.2x";
    v5[0] = &unk_49EF3B0;
    ++v2;
    v6 = v3;
    sub_16E8450(v7, v5);
  }
  while ( v2 != a2 + 16 );
  v7[0] = &unk_49EFD28;
  sub_16E7960(v7);
  return a1;
}
