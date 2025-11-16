// Function: ctor_114
// Address: 0x4ab4c0
//
int ctor_114()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax

  qword_4F97D80 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F97D8C &= 0xF000u;
  qword_4F97D90 = 0;
  qword_4F97D98 = 0;
  qword_4F97DA0 = 0;
  qword_4F97DA8 = 0;
  qword_4F97DB0 = 0;
  dword_4F97D88 = v0;
  qword_4F97DB8 = 0;
  qword_4F97DC8 = (__int64)&unk_4FA01C0;
  qword_4F97DD8 = (__int64)&unk_4F97DF8;
  qword_4F97DE0 = (__int64)&unk_4F97DF8;
  qword_4F97DC0 = 0;
  qword_4F97DD0 = 0;
  word_4F97E30 = 256;
  qword_4F97E28 = (__int64)&unk_49E74E8;
  qword_4F97DE8 = 4;
  qword_4F97D80 = (__int64)&unk_49EEC70;
  byte_4F97E18 = 0;
  qword_4F97E38 = (__int64)&unk_49EEDB0;
  dword_4F97DF0 = 0;
  byte_4F97E20 = 0;
  sub_16B8280(&qword_4F97D80, "strict-aliasing", 15);
  word_4F97E30 = 257;
  byte_4F97E20 = 1;
  qword_4F97DB0 = 27;
  LOBYTE(word_4F97D8C) = word_4F97D8C & 0x9F | 0x20;
  qword_4F97DA8 = (__int64)"Datatype based strict alias";
  sub_16B88A0(&qword_4F97D80);
  __cxa_atexit(sub_12EDEC0, &qword_4F97D80, &qword_4A427C0);
  qword_4F97CA0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F97D50 = 256;
  word_4F97CAC &= 0xF000u;
  qword_4F97CB0 = 0;
  qword_4F97CB8 = 0;
  qword_4F97CC0 = 0;
  dword_4F97CA8 = v1;
  qword_4F97D48 = (__int64)&unk_49E74E8;
  qword_4F97CE8 = (__int64)&unk_4FA01C0;
  qword_4F97CF8 = (__int64)&unk_4F97D18;
  qword_4F97D00 = (__int64)&unk_4F97D18;
  qword_4F97CA0 = (__int64)&unk_49EEC70;
  qword_4F97D58 = (__int64)&unk_49EEDB0;
  qword_4F97CC8 = 0;
  qword_4F97CD0 = 0;
  qword_4F97CD8 = 0;
  qword_4F97CE0 = 0;
  qword_4F97CF0 = 0;
  qword_4F97D08 = 4;
  dword_4F97D10 = 0;
  byte_4F97D38 = 0;
  byte_4F97D40 = 0;
  sub_16B8280(&qword_4F97CA0, "traverse-address-aliasing", 25);
  word_4F97D50 = 257;
  byte_4F97D40 = 1;
  qword_4F97CD0 = 36;
  LOBYTE(word_4F97CAC) = word_4F97CAC & 0x9F | 0x20;
  qword_4F97CC8 = (__int64)"Find address space through traversal";
  sub_16B88A0(&qword_4F97CA0);
  __cxa_atexit(sub_12EDEC0, &qword_4F97CA0, &qword_4A427C0);
  qword_4F97BC0 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F97C70 = 256;
  word_4F97BCC &= 0xF000u;
  qword_4F97BD0 = 0;
  qword_4F97BD8 = 0;
  qword_4F97BE0 = 0;
  dword_4F97BC8 = v2;
  qword_4F97C68 = (__int64)&unk_49E74E8;
  qword_4F97C08 = (__int64)&unk_4FA01C0;
  qword_4F97C18 = (__int64)&unk_4F97C38;
  qword_4F97C20 = (__int64)&unk_4F97C38;
  qword_4F97BC0 = (__int64)&unk_49EEC70;
  qword_4F97C78 = (__int64)&unk_49EEDB0;
  qword_4F97BE8 = 0;
  qword_4F97BF0 = 0;
  qword_4F97BF8 = 0;
  qword_4F97C00 = 0;
  qword_4F97C10 = 0;
  qword_4F97C28 = 4;
  dword_4F97C30 = 0;
  byte_4F97C58 = 0;
  byte_4F97C60 = 0;
  sub_16B8280(&qword_4F97BC0, "basicaa-recphi", 14);
  byte_4F97C60 = 0;
  word_4F97C70 = 256;
  LOBYTE(word_4F97BCC) = word_4F97BCC & 0x9F | 0x20;
  sub_16B88A0(&qword_4F97BC0);
  return __cxa_atexit(sub_12EDEC0, &qword_4F97BC0, &qword_4A427C0);
}
