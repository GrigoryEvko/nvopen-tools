// Function: ctor_661
// Address: 0x59d5d0
//
int ctor_661()
{
  int v1; // [rsp+10h] [rbp-50h] BYREF
  int v2; // [rsp+14h] [rbp-4Ch] BYREF
  int *v3; // [rsp+18h] [rbp-48h] BYREF
  const char *v4; // [rsp+20h] [rbp-40h] BYREF
  __int64 v5; // [rsp+28h] [rbp-38h]

  v3 = &v2;
  v4 = "Debug control for aggressive anti-dep breaker";
  v1 = 1;
  v2 = 0;
  v5 = 45;
  sub_34B3EA0(&unk_503A560, "agg-antidep-debugdiv", &v4, &v3, &v1);
  __cxa_atexit(sub_B2B680, &unk_503A560, &qword_4A427C0);
  v4 = "Debug control for aggressive anti-dep breaker";
  v1 = 1;
  v2 = 0;
  v3 = &v2;
  v5 = 45;
  sub_34B3EA0(&unk_503A480, "agg-antidep-debugmod", &v4, &v3, &v1);
  return __cxa_atexit(sub_B2B680, &unk_503A480, &qword_4A427C0);
}
