// Function: ctor_468
// Address: 0x549c10
//
int ctor_468()
{
  char v1; // [rsp+7h] [rbp-9h] BYREF
  char *v2; // [rsp+8h] [rbp-8h] BYREF

  v1 = 1;
  v2 = &v1;
  sub_2845D90(&unk_5000640, "enable-loop-simplifycfg-term-folding", &v2);
  return __cxa_atexit(sub_984900, &unk_5000640, &qword_4A427C0);
}
