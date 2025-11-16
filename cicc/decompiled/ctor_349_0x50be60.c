// Function: ctor_349
// Address: 0x50be60
//
int ctor_349()
{
  int v0; // edx

  qword_4FCF780 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FCF78C &= 0xF000u;
  qword_4FCF790 = 0;
  qword_4FCF7C8 = (__int64)qword_4FA01C0;
  qword_4FCF798 = 0;
  qword_4FCF7A0 = 0;
  qword_4FCF7A8 = 0;
  dword_4FCF788 = v0;
  qword_4FCF7D8 = (__int64)&unk_4FCF7F8;
  qword_4FCF7E0 = (__int64)&unk_4FCF7F8;
  qword_4FCF7B0 = 0;
  qword_4FCF7B8 = 0;
  qword_4FCF828 = (__int64)&unk_49E74E8;
  word_4FCF830 = 256;
  qword_4FCF7C0 = 0;
  qword_4FCF7D0 = 0;
  qword_4FCF780 = (__int64)&unk_49EEC70;
  qword_4FCF7E8 = 4;
  byte_4FCF818 = 0;
  qword_4FCF838 = (__int64)&unk_49EEDB0;
  dword_4FCF7F0 = 0;
  byte_4FCF820 = 0;
  sub_16B8280(&qword_4FCF780, "view-edge-bundles", 17);
  qword_4FCF7B0 = 42;
  LOBYTE(word_4FCF78C) = word_4FCF78C & 0x9F | 0x20;
  qword_4FCF7A8 = (__int64)"Pop up a window to show edge bundle graphs";
  sub_16B88A0(&qword_4FCF780);
  return __cxa_atexit(sub_12EDEC0, &qword_4FCF780, &qword_4A427C0);
}
