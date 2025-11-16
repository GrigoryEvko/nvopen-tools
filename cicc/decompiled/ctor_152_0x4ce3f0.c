// Function: ctor_152
// Address: 0x4ce3f0
//
int ctor_152()
{
  int v1; // [rsp+1Ch] [rbp-64h] BYREF
  int v2; // [rsp+20h] [rbp-60h] BYREF
  int v3; // [rsp+24h] [rbp-5Ch] BYREF
  __int64 *v4; // [rsp+28h] [rbp-58h] BYREF
  const char *v5; // [rsp+30h] [rbp-50h] BYREF
  __int64 v6; // [rsp+38h] [rbp-48h]
  char *v7; // [rsp+40h] [rbp-40h] BYREF
  __int64 v8; // [rsp+48h] [rbp-38h]

  __cxa_atexit(sub_166D230, &qword_4F9F310, &qword_4A427C0);
  __cxa_atexit(sub_166D230, &qword_4F9F300, &qword_4A427C0);
  __cxa_atexit(sub_166D230, &qword_4F9F2F0, &qword_4A427C0);
  v4 = &qword_4F9F310;
  v7 = "pattern";
  v5 = "Enable optimization remarks from passes whose name match the given regular expression";
  v1 = 1;
  v2 = 2;
  v3 = 1;
  v6 = 85;
  v8 = 7;
  sub_4CE200((__int64)&unk_4F9F220, "pass-remarks", (__int64 *)&v7, (__int64 *)&v5, &v3, &v4, &v2, &v1);
  __cxa_atexit(sub_166D2C0, &unk_4F9F220, &qword_4A427C0);
  v4 = &qword_4F9F300;
  v7 = "pattern";
  v5 = "Enable missed optimization remarks from passes whose name match the given regular expression";
  v1 = 1;
  v2 = 2;
  v3 = 1;
  v6 = 92;
  v8 = 7;
  sub_4CE200((__int64)&unk_4F9F140, "pass-remarks-missed", (__int64 *)&v7, (__int64 *)&v5, &v3, &v4, &v2, &v1);
  __cxa_atexit(sub_166D2C0, &unk_4F9F140, &qword_4A427C0);
  v4 = &qword_4F9F2F0;
  v5 = "Enable optimization analysis remarks from passes whose name match the given regular expression";
  v1 = 1;
  v2 = 2;
  v3 = 1;
  v6 = 94;
  v7 = "pattern";
  v8 = 7;
  sub_4CE200((__int64)&unk_4F9F060, "pass-remarks-analysis", (__int64 *)&v7, (__int64 *)&v5, &v3, &v4, &v2, &v1);
  return __cxa_atexit(sub_166D2C0, &unk_4F9F060, &qword_4A427C0);
}
