// Function: ctor_063
// Address: 0x4957d0
//
int ctor_063()
{
  int v1; // [rsp+10h] [rbp-60h] BYREF
  int v2; // [rsp+14h] [rbp-5Ch] BYREF
  __int64 *v3; // [rsp+18h] [rbp-58h] BYREF
  const char *v4; // [rsp+20h] [rbp-50h] BYREF
  __int64 v5; // [rsp+28h] [rbp-48h]
  char *v6; // [rsp+30h] [rbp-40h] BYREF
  __int64 v7; // [rsp+38h] [rbp-38h]

  __cxa_atexit(sub_E3EBA0, &qword_4F8A300, &qword_4A427C0);
  __cxa_atexit(sub_E3EBA0, &qword_4F8A2F0, &qword_4A427C0);
  __cxa_atexit(sub_E3EBA0, &qword_4F8A2E0, &qword_4A427C0);
  v3 = &qword_4F8A300;
  v6 = "pattern";
  v4 = "Enable optimization remarks from passes whose name match the given regular expression";
  v1 = 2;
  v2 = 1;
  v5 = 85;
  v7 = 7;
  sub_495590((__int64)&unk_4F8A220, "pass-remarks", (__int64 *)&v6, (__int64 *)&v4, &v2, &v3, &v1);
  __cxa_atexit(sub_E3EAB0, &unk_4F8A220, &qword_4A427C0);
  v3 = &qword_4F8A2F0;
  v6 = "pattern";
  v4 = "Enable missed optimization remarks from passes whose name match the given regular expression";
  v1 = 2;
  v2 = 1;
  v5 = 92;
  v7 = 7;
  sub_495590((__int64)&unk_4F8A160, "pass-remarks-missed", (__int64 *)&v6, (__int64 *)&v4, &v2, &v3, &v1);
  __cxa_atexit(sub_E3EAB0, &unk_4F8A160, &qword_4A427C0);
  v3 = &qword_4F8A2E0;
  v4 = "Enable optimization analysis remarks from passes whose name match the given regular expression";
  v1 = 2;
  v2 = 1;
  v5 = 94;
  v6 = "pattern";
  v7 = 7;
  sub_495590((__int64)&unk_4F8A0A0, "pass-remarks-analysis", (__int64 *)&v6, (__int64 *)&v4, &v2, &v3, &v1);
  return __cxa_atexit(sub_E3EAB0, &unk_4F8A0A0, &qword_4A427C0);
}
