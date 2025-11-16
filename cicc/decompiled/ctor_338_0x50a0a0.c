// Function: ctor_338
// Address: 0x50a0a0
//
int ctor_338()
{
  int v0; // edx
  char v2; // [rsp+3h] [rbp-4Dh] BYREF
  int v3; // [rsp+4h] [rbp-4Ch] BYREF
  char *v4; // [rsp+8h] [rbp-48h] BYREF
  const char *v5; // [rsp+10h] [rbp-40h] BYREF
  __int64 v6; // [rsp+18h] [rbp-38h]

  v5 = "Clone multicolor basic blocks but do not demote cross scopes";
  v2 = 0;
  v4 = &v2;
  v6 = 60;
  v3 = 1;
  sub_1F60950(&unk_4FCE600, "disable-demotion", &v3, &v5, &v4);
  __cxa_atexit(sub_12EDEC0, &unk_4FCE600, &qword_4A427C0);
  v2 = 0;
  v5 = "Do not remove implausible terminators or other similar cleanups";
  v4 = &v2;
  v6 = 63;
  v3 = 1;
  sub_1F60950(&unk_4FCE520, "disable-cleanups", &v3, &v5, &v4);
  __cxa_atexit(sub_12EDEC0, &unk_4FCE520, &qword_4A427C0);
  qword_4FCE440 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCE44C &= 0xF000u;
  qword_4FCE450 = 0;
  qword_4FCE488 = (__int64)qword_4FA01C0;
  qword_4FCE458 = 0;
  qword_4FCE460 = 0;
  qword_4FCE468 = 0;
  dword_4FCE448 = v0;
  qword_4FCE498 = (__int64)&unk_4FCE4B8;
  qword_4FCE4A0 = (__int64)&unk_4FCE4B8;
  qword_4FCE470 = 0;
  qword_4FCE478 = 0;
  qword_4FCE4E8 = (__int64)&unk_49E74E8;
  word_4FCE4F0 = 256;
  qword_4FCE480 = 0;
  qword_4FCE490 = 0;
  qword_4FCE440 = (__int64)&unk_49EEC70;
  qword_4FCE4A8 = 4;
  byte_4FCE4D8 = 0;
  qword_4FCE4F8 = (__int64)&unk_49EEDB0;
  dword_4FCE4B0 = 0;
  byte_4FCE4E0 = 0;
  sub_16B8280(&qword_4FCE440, "demote-catchswitch-only", 23);
  word_4FCE4F0 = 256;
  byte_4FCE4E0 = 0;
  qword_4FCE470 = 41;
  LOBYTE(word_4FCE44C) = word_4FCE44C & 0x9F | 0x20;
  qword_4FCE468 = (__int64)"Demote catchswitch BBs only (for wasm EH)";
  sub_16B88A0(&qword_4FCE440);
  return __cxa_atexit(sub_12EDEC0, &qword_4FCE440, &qword_4A427C0);
}
