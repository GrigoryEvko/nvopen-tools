// Function: ctor_363
// Address: 0x511bb0
//
int ctor_363()
{
  int v0; // edx

  qword_4FD3DC0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FD3DCC &= 0xF000u;
  qword_4FD3DD0 = 0;
  qword_4FD3E08 = (__int64)qword_4FA01C0;
  qword_4FD3DD8 = 0;
  qword_4FD3DE0 = 0;
  qword_4FD3DE8 = 0;
  dword_4FD3DC8 = v0;
  qword_4FD3E18 = (__int64)&unk_4FD3E38;
  qword_4FD3E20 = (__int64)&unk_4FD3E38;
  qword_4FD3DF0 = 0;
  qword_4FD3DF8 = 0;
  qword_4FD3E68 = (__int64)&unk_49E74E8;
  word_4FD3E70 = 256;
  qword_4FD3E00 = 0;
  qword_4FD3E10 = 0;
  qword_4FD3DC0 = (__int64)&unk_49EEC70;
  qword_4FD3E28 = 4;
  byte_4FD3E58 = 0;
  qword_4FD3E78 = (__int64)&unk_49EEDB0;
  dword_4FD3E30 = 0;
  byte_4FD3E60 = 0;
  sub_16B8280(&qword_4FD3DC0, "vasp-fix2", 9);
  word_4FD3E70 = 256;
  byte_4FD3E60 = 0;
  qword_4FD3DF0 = 0;
  LOBYTE(word_4FD3DCC) = word_4FD3DCC & 0x9F | 0x20;
  qword_4FD3DE8 = (__int64)byte_3F871B3;
  sub_16B88A0(&qword_4FD3DC0);
  return __cxa_atexit(sub_12EDEC0, &qword_4FD3DC0, &qword_4A427C0);
}
