// Function: ctor_252
// Address: 0x4f04e0
//
int ctor_252()
{
  int v0; // edx
  char v2; // [rsp+3h] [rbp-4Dh] BYREF
  int v3; // [rsp+4h] [rbp-4Ch] BYREF
  char *v4; // [rsp+8h] [rbp-48h] BYREF
  const char *v5; // [rsp+10h] [rbp-40h] BYREF
  __int64 v6; // [rsp+18h] [rbp-38h]

  v5 = "Dump the function under Convergency Analysis";
  v6 = 44;
  v3 = 1;
  v2 = 0;
  v4 = &v2;
  sub_1C04CB0(&unk_4FBA000, "dump-conv-func", &v4, &v3, &v5);
  __cxa_atexit(sub_12EDEC0, &unk_4FBA000, &qword_4A427C0);
  v6 = 44;
  v5 = "Dump text format of the convergency analysis";
  v3 = 1;
  v2 = 0;
  v4 = &v2;
  sub_1C04CB0(&unk_4FB9F20, "dump-conv-text", &v4, &v3, &v5);
  __cxa_atexit(sub_12EDEC0, &unk_4FB9F20, &qword_4A427C0);
  qword_4FB9E40 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB9E4C &= 0xF000u;
  qword_4FB9E50 = 0;
  qword_4FB9E88 = (__int64)qword_4FA01C0;
  qword_4FB9E58 = 0;
  qword_4FB9E60 = 0;
  qword_4FB9E68 = 0;
  dword_4FB9E48 = v0;
  qword_4FB9E98 = (__int64)&unk_4FB9EB8;
  qword_4FB9EA0 = (__int64)&unk_4FB9EB8;
  qword_4FB9E70 = 0;
  qword_4FB9E78 = 0;
  qword_4FB9EE8 = (__int64)&unk_49E74E8;
  word_4FB9EF0 = 256;
  qword_4FB9E80 = 0;
  qword_4FB9E90 = 0;
  qword_4FB9E40 = (__int64)&unk_49EEC70;
  qword_4FB9EA8 = 4;
  byte_4FB9ED8 = 0;
  qword_4FB9EF8 = (__int64)&unk_49EEDB0;
  dword_4FB9EB0 = 0;
  byte_4FB9EE0 = 0;
  sub_16B8280(&qword_4FB9E40, "dump-conv-dot", 13);
  word_4FB9EF0 = 256;
  byte_4FB9EE0 = 0;
  qword_4FB9E70 = 43;
  LOBYTE(word_4FB9E4C) = word_4FB9E4C & 0x9F | 0x20;
  qword_4FB9E68 = (__int64)"Dump dot format of the convergency analysis";
  sub_16B88A0(&qword_4FB9E40);
  return __cxa_atexit(sub_12EDEC0, &qword_4FB9E40, &qword_4A427C0);
}
