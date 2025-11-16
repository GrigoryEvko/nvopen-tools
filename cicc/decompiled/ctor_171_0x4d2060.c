// Function: ctor_171
// Address: 0x4d2060
//
int ctor_171()
{
  int v0; // eax
  int v1; // esi
  int v2; // esi
  int v4; // [rsp+3Ch] [rbp-54h] BYREF
  int v5; // [rsp+40h] [rbp-50h] BYREF
  int v6; // [rsp+44h] [rbp-4Ch] BYREF
  int *v7; // [rsp+48h] [rbp-48h] BYREF
  const char *v8; // [rsp+50h] [rbp-40h] BYREF
  __int64 v9; // [rsp+58h] [rbp-38h]

  qword_4FA31E0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA31EC &= 0xF000u;
  qword_4FA31F0 = 0;
  qword_4FA31F8 = 0;
  qword_4FA3200 = 0;
  qword_4FA3208 = 0;
  qword_4FA3210 = 0;
  dword_4FA31E8 = v0;
  qword_4FA3218 = 0;
  qword_4FA3228 = (__int64)qword_4FA01C0;
  qword_4FA3238 = (__int64)&unk_4FA3258;
  qword_4FA3240 = (__int64)&unk_4FA3258;
  qword_4FA3220 = 0;
  qword_4FA3230 = 0;
  qword_4FA3288 = (__int64)&unk_49E74E8;
  word_4FA3290 = 256;
  qword_4FA3248 = 4;
  dword_4FA3250 = 0;
  qword_4FA31E0 = (__int64)&unk_49EEC70;
  qword_4FA3298 = (__int64)&unk_49EEDB0;
  byte_4FA3278 = 0;
  byte_4FA3280 = 0;
  sub_16B8280(&qword_4FA31E0, "disable-icp", 11);
  qword_4FA3208 = (__int64)"Disable indirect call promotion";
  word_4FA3290 = 256;
  byte_4FA3280 = 0;
  qword_4FA3210 = 31;
  LOBYTE(word_4FA31EC) = word_4FA31EC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FA31E0);
  __cxa_atexit(sub_12EDEC0, &qword_4FA31E0, &qword_4A427C0);
  v8 = "Max number of promotions for this compilation";
  v9 = 45;
  v4 = 1;
  v5 = 1;
  v6 = 0;
  v7 = &v6;
  sub_17C1EA0(&unk_4FA3100, "icp-cutoff", &v7, &v5, &v4, &v8);
  __cxa_atexit(sub_12EDE60, &unk_4FA3100, &qword_4A427C0);
  v9 = 52;
  v8 = "Skip Callsite up to this number for this compilation";
  v4 = 1;
  v5 = 1;
  v6 = 0;
  v7 = &v6;
  sub_17C1EA0(&unk_4FA3020, "icp-csskip", &v7, &v5, &v4, &v8);
  __cxa_atexit(sub_12EDE60, &unk_4FA3020, &qword_4A427C0);
  qword_4FA2F40 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  byte_4FA2FD8 = 0;
  word_4FA2FF0 = 256;
  qword_4FA2F50 = 0;
  word_4FA2F4C &= 0xF000u;
  qword_4FA2FF8 = (__int64)&unk_49EEDB0;
  qword_4FA2F40 = (__int64)&unk_49EEC70;
  dword_4FA2F48 = v1;
  qword_4FA2F98 = (__int64)&unk_4FA2FB8;
  qword_4FA2FA0 = (__int64)&unk_4FA2FB8;
  qword_4FA2F88 = (__int64)qword_4FA01C0;
  qword_4FA2FE8 = (__int64)&unk_49E74E8;
  qword_4FA2F58 = 0;
  qword_4FA2F60 = 0;
  qword_4FA2F68 = 0;
  qword_4FA2F70 = 0;
  qword_4FA2F78 = 0;
  qword_4FA2F80 = 0;
  qword_4FA2F90 = 0;
  qword_4FA2FA8 = 4;
  dword_4FA2FB0 = 0;
  byte_4FA2FE0 = 0;
  sub_16B8280(&qword_4FA2F40, "icp-lto", 7);
  word_4FA2FF0 = 256;
  byte_4FA2FE0 = 0;
  qword_4FA2F70 = 39;
  qword_4FA2F68 = (__int64)"Run indirect-call promotion in LTO mode";
  LOBYTE(word_4FA2F4C) = word_4FA2F4C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FA2F40);
  __cxa_atexit(sub_12EDEC0, &qword_4FA2F40, &qword_4A427C0);
  LOBYTE(v5) = 0;
  v8 = "Run indirect-call promotion in SamplePGO mode";
  v9 = 45;
  v6 = 1;
  v7 = &v5;
  sub_17C2030(&unk_4FA2E60, "icp-samplepgo", &v7, &v6, &v8);
  __cxa_atexit(sub_12EDEC0, &unk_4FA2E60, &qword_4A427C0);
  LOBYTE(v5) = 0;
  v8 = "Run indirect-call promotion for call instructions only";
  v9 = 54;
  v6 = 1;
  v7 = &v5;
  sub_17C2030(&unk_4FA2D80, "icp-call-only", &v7, &v6, &v8);
  __cxa_atexit(sub_12EDEC0, &unk_4FA2D80, &qword_4A427C0);
  qword_4FA2CA0 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  byte_4FA2D38 = 0;
  word_4FA2D50 = 256;
  qword_4FA2CB0 = 0;
  word_4FA2CAC &= 0xF000u;
  qword_4FA2D58 = (__int64)&unk_49EEDB0;
  qword_4FA2CA0 = (__int64)&unk_49EEC70;
  dword_4FA2CA8 = v2;
  qword_4FA2CF8 = (__int64)&unk_4FA2D18;
  qword_4FA2D00 = (__int64)&unk_4FA2D18;
  qword_4FA2CE8 = (__int64)qword_4FA01C0;
  qword_4FA2D48 = (__int64)&unk_49E74E8;
  qword_4FA2CB8 = 0;
  qword_4FA2CC0 = 0;
  qword_4FA2CC8 = 0;
  qword_4FA2CD0 = 0;
  qword_4FA2CD8 = 0;
  qword_4FA2CE0 = 0;
  qword_4FA2CF0 = 0;
  qword_4FA2D08 = 4;
  dword_4FA2D10 = 0;
  byte_4FA2D40 = 0;
  sub_16B8280(&qword_4FA2CA0, "icp-invoke-only", 15);
  byte_4FA2D40 = 0;
  word_4FA2D50 = 256;
  qword_4FA2CD0 = 55;
  LOBYTE(word_4FA2CAC) = word_4FA2CAC & 0x9F | 0x20;
  qword_4FA2CC8 = (__int64)"Run indirect-call promotion for invoke instruction only";
  sub_16B88A0(&qword_4FA2CA0);
  __cxa_atexit(sub_12EDEC0, &qword_4FA2CA0, &qword_4A427C0);
  v7 = &v5;
  v8 = "Dump IR after transformation happens";
  v9 = 36;
  v6 = 1;
  LOBYTE(v5) = 0;
  sub_17C2030(&unk_4FA2BC0, "icp-dumpafter", &v7, &v6, &v8);
  return __cxa_atexit(sub_12EDEC0, &unk_4FA2BC0, &qword_4A427C0);
}
