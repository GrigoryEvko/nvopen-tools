// Function: ctor_130
// Address: 0x4af0f0
//
int ctor_130()
{
  int v0; // edx

  qword_4F9A3C0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9A3CC &= 0xF000u;
  qword_4F9A3D0 = 0;
  qword_4F9A408 = (__int64)&unk_4FA01C0;
  qword_4F9A3D8 = 0;
  qword_4F9A3E0 = 0;
  qword_4F9A3E8 = 0;
  dword_4F9A3C8 = v0;
  qword_4F9A418 = (__int64)&unk_4F9A438;
  qword_4F9A420 = (__int64)&unk_4F9A438;
  qword_4F9A3F0 = 0;
  qword_4F9A3F8 = 0;
  qword_4F9A468 = (__int64)&unk_49E74E8;
  word_4F9A470 = 256;
  qword_4F9A400 = 0;
  qword_4F9A410 = 0;
  qword_4F9A3C0 = (__int64)&unk_49EEC70;
  qword_4F9A428 = 4;
  byte_4F9A458 = 0;
  qword_4F9A478 = (__int64)&unk_49EEDB0;
  dword_4F9A430 = 0;
  byte_4F9A460 = 0;
  sub_16B8280(&qword_4F9A3C0, "only-simple-regions", 19);
  qword_4F9A3E8 = (__int64)"Show only simple regions in the graphviz viewer";
  word_4F9A470 = 256;
  byte_4F9A460 = 0;
  qword_4F9A3F0 = 47;
  LOBYTE(word_4F9A3CC) = word_4F9A3CC & 0x9F | 0x20;
  sub_16B88A0(&qword_4F9A3C0);
  return __cxa_atexit(sub_12EDEC0, &qword_4F9A3C0, &qword_4A427C0);
}
