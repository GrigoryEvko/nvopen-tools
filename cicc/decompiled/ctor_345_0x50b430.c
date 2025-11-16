// Function: ctor_345
// Address: 0x50b430
//
int ctor_345()
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
  sub_20C2E60(&unk_4FCF160, "agg-antidep-debugdiv", &v4, &v3, &v1);
  __cxa_atexit(sub_12EDEA0, &unk_4FCF160, &qword_4A427C0);
  v4 = "Debug control for aggressive anti-dep breaker";
  v1 = 1;
  v2 = 0;
  v3 = &v2;
  v5 = 45;
  sub_20C2E60(&unk_4FCF080, "agg-antidep-debugmod", &v4, &v3, &v1);
  return __cxa_atexit(sub_12EDEA0, &unk_4FCF080, &qword_4A427C0);
}
