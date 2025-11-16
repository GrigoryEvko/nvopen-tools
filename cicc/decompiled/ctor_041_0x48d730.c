// Function: ctor_041
// Address: 0x48d730
//
int ctor_041()
{
  char v1; // [rsp+7h] [rbp-29h] BYREF
  char *v2; // [rsp+8h] [rbp-28h] BYREF
  _QWORD v3[4]; // [rsp+10h] [rbp-20h] BYREF

  sub_2208040(&unk_4F85228);
  __cxa_atexit(sub_2208810, &unk_4F85228, &qword_4A427C0);
  v3[0] = "Do not transplant metadata onto functions";
  v3[1] = 41;
  v1 = 0;
  v2 = &v1;
  sub_CE9FB0(&unk_4F85160, "disable-attrib-transplant", &v2, v3);
  return __cxa_atexit(sub_984900, &unk_4F85160, &qword_4A427C0);
}
