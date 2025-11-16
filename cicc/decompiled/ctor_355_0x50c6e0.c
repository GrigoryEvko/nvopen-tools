// Function: ctor_355
// Address: 0x50c6e0
//
int ctor_355()
{
  int v0; // edx

  qword_4FCFC20 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCFC2C &= 0xF000u;
  qword_4FCFC30 = 0;
  qword_4FCFC68 = (__int64)qword_4FA01C0;
  qword_4FCFC38 = 0;
  qword_4FCFC40 = 0;
  qword_4FCFC48 = 0;
  dword_4FCFC28 = v0;
  qword_4FCFC78 = (__int64)&unk_4FCFC98;
  qword_4FCFC80 = (__int64)&unk_4FCFC98;
  qword_4FCFC50 = 0;
  qword_4FCFC58 = 0;
  qword_4FCFCC8 = (__int64)&unk_49E74C8;
  qword_4FCFC60 = 0;
  qword_4FCFC70 = 0;
  qword_4FCFC20 = (__int64)&unk_49EEB70;
  qword_4FCFC88 = 4;
  dword_4FCFC90 = 0;
  qword_4FCFCD8 = (__int64)&unk_49EEDF0;
  byte_4FCFCB8 = 0;
  dword_4FCFCC0 = 0;
  byte_4FCFCD4 = 1;
  dword_4FCFCD0 = 0;
  sub_16B8280(&qword_4FCFC20, "stackmap-version", 16);
  dword_4FCFCC0 = 3;
  byte_4FCFCD4 = 1;
  dword_4FCFCD0 = 3;
  qword_4FCFC50 = 51;
  LOBYTE(word_4FCFC2C) = word_4FCFC2C & 0x9F | 0x20;
  qword_4FCFC48 = (__int64)"Specify the stackmap encoding version (default = 3)";
  sub_16B88A0(&qword_4FCFC20);
  return __cxa_atexit(sub_12EDEA0, &qword_4FCFC20, &qword_4A427C0);
}
