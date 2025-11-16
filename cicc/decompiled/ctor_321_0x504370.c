// Function: ctor_321
// Address: 0x504370
//
int ctor_321()
{
  int v0; // edx

  qword_4FC9C88 = (__int64)"pbqp";
  qword_4FC9C98 = (__int64)"PBQP register allocator";
  qword_4FC9C80 = 0;
  qword_4FC9CA8 = (__int64)sub_1ECC880;
  qword_4FC9C90 = 4;
  qword_4FC9CA0 = 23;
  sub_1E40390(&unk_4FCB760, &qword_4FC9C80);
  __cxa_atexit(sub_1EB3C00, &qword_4FC9C80, &qword_4A427C0);
  qword_4FC9BA0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC9BAC &= 0xF000u;
  qword_4FC9BB0 = 0;
  qword_4FC9BE8 = (__int64)qword_4FA01C0;
  qword_4FC9BB8 = 0;
  qword_4FC9BC0 = 0;
  qword_4FC9BC8 = 0;
  dword_4FC9BA8 = v0;
  qword_4FC9BF8 = (__int64)&unk_4FC9C18;
  qword_4FC9C00 = (__int64)&unk_4FC9C18;
  qword_4FC9BD0 = 0;
  qword_4FC9BD8 = 0;
  qword_4FC9C48 = (__int64)&unk_49E74E8;
  word_4FC9C50 = 256;
  qword_4FC9BE0 = 0;
  qword_4FC9BF0 = 0;
  qword_4FC9BA0 = (__int64)&unk_49EEC70;
  qword_4FC9C08 = 4;
  byte_4FC9C38 = 0;
  qword_4FC9C58 = (__int64)&unk_49EEDB0;
  dword_4FC9C10 = 0;
  byte_4FC9C40 = 0;
  sub_16B8280(&qword_4FC9BA0, "pbqp-coalescing", 15);
  qword_4FC9BC8 = (__int64)"Attempt coalescing during PBQP register allocation.";
  word_4FC9C50 = 256;
  byte_4FC9C40 = 0;
  qword_4FC9BD0 = 51;
  LOBYTE(word_4FC9BAC) = word_4FC9BAC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC9BA0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FC9BA0, &qword_4A427C0);
}
