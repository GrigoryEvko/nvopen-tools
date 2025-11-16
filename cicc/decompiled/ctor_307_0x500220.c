// Function: ctor_307
// Address: 0x500220
//
int ctor_307()
{
  int v0; // edx

  qword_4FC6B00 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC6B0C &= 0xF000u;
  qword_4FC6B10 = 0;
  qword_4FC6B48 = (__int64)qword_4FA01C0;
  qword_4FC6B18 = 0;
  qword_4FC6B20 = 0;
  qword_4FC6B28 = 0;
  dword_4FC6B08 = v0;
  qword_4FC6B58 = (__int64)&unk_4FC6B78;
  qword_4FC6B60 = (__int64)&unk_4FC6B78;
  qword_4FC6B30 = 0;
  qword_4FC6B38 = 0;
  qword_4FC6BA8 = (__int64)&unk_49E74E8;
  word_4FC6BB0 = 256;
  qword_4FC6B40 = 0;
  qword_4FC6B50 = 0;
  qword_4FC6B00 = (__int64)&unk_49EEC70;
  qword_4FC6B68 = 4;
  byte_4FC6B98 = 0;
  qword_4FC6BB8 = (__int64)&unk_49EEDB0;
  dword_4FC6B70 = 0;
  byte_4FC6BA0 = 0;
  sub_16B8280(&qword_4FC6B00, "enable-linkonceodr-outlining", 28);
  word_4FC6BB0 = 256;
  byte_4FC6BA0 = 0;
  qword_4FC6B30 = 52;
  LOBYTE(word_4FC6B0C) = word_4FC6B0C & 0x9F | 0x20;
  qword_4FC6B28 = (__int64)"Enable the machine outliner on linkonceodr functions";
  sub_16B88A0(&qword_4FC6B00);
  return __cxa_atexit(sub_12EDEC0, &qword_4FC6B00, &qword_4A427C0);
}
