// Function: ctor_722
// Address: 0x5c0f80
//
int ctor_722()
{
  int v0; // edx

  qword_5052C00 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_5052C0C &= 0xF000u;
  qword_5052C10 = 0;
  qword_5052C48 = (__int64)qword_4FA01C0;
  qword_5052C18 = 0;
  qword_5052C20 = 0;
  qword_5052C28 = 0;
  dword_5052C08 = v0;
  qword_5052C58 = (__int64)&unk_5052C78;
  qword_5052C60 = (__int64)&unk_5052C78;
  qword_5052C30 = 0;
  qword_5052C38 = 0;
  qword_5052CA8 = (__int64)&unk_49E74A8;
  qword_5052C40 = 0;
  qword_5052C50 = 0;
  qword_5052C00 = (__int64)&unk_49EEAF0;
  qword_5052C68 = 4;
  dword_5052C70 = 0;
  qword_5052CB8 = (__int64)&unk_49EEE10;
  byte_5052C98 = 0;
  dword_5052CA0 = 0;
  byte_5052CB4 = 1;
  dword_5052CB0 = 0;
  sub_16B8280(&qword_5052C00, "asm-macro-max-nesting-depth", 27);
  dword_5052CA0 = 20;
  byte_5052CB4 = 1;
  dword_5052CB0 = 20;
  qword_5052C30 = 54;
  LOBYTE(word_5052C0C) = word_5052C0C & 0x9F | 0x20;
  qword_5052C28 = (__int64)"The maximum nesting depth allowed for assembly macros.";
  sub_16B88A0(&qword_5052C00);
  return __cxa_atexit(sub_12EDE60, &qword_5052C00, &qword_4A427C0);
}
