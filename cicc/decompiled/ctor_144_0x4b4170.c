// Function: ctor_144
// Address: 0x4b4170
//
int ctor_144()
{
  int v0; // edx
  __int64 v1; // rax
  const char *v3; // [rsp+0h] [rbp-20h] BYREF
  char v4; // [rsp+10h] [rbp-10h]
  char v5; // [rsp+11h] [rbp-Fh]

  qword_4F9E080 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9E08C &= 0xF000u;
  qword_4F9E0C8 = (__int64)&unk_4FA01C0;
  qword_4F9E0D8 = (__int64)&unk_4F9E0F8;
  qword_4F9E0E0 = (__int64)&unk_4F9E0F8;
  qword_4F9E090 = 0;
  dword_4F9E088 = v0;
  qword_4F9E128 = (__int64)&unk_49E74E8;
  qword_4F9E098 = 0;
  qword_4F9E0A0 = 0;
  qword_4F9E080 = (__int64)&unk_49EAB58;
  qword_4F9E0A8 = 0;
  qword_4F9E0B0 = 0;
  qword_4F9E0B8 = 0;
  qword_4F9E0C0 = 0;
  qword_4F9E0D0 = 0;
  qword_4F9E0E8 = 4;
  dword_4F9E0F0 = 0;
  byte_4F9E118 = 0;
  qword_4F9E120 = 0;
  byte_4F9E131 = 0;
  qword_4F9E138 = (__int64)&unk_49EEDB0;
  sub_16B8280(&qword_4F9E080, "verify-dom-info", 15);
  if ( qword_4F9E120 )
  {
    v1 = sub_16E8CB0();
    v5 = 1;
    v3 = "cl::location(x) specified more than once!";
    v4 = 3;
    sub_16B1F90(&qword_4F9E080, &v3, 0, 0, v1);
  }
  else
  {
    byte_4F9E131 = 1;
    qword_4F9E120 = (__int64)byte_4F9E148;
    byte_4F9E130 = byte_4F9E148[0];
  }
  qword_4F9E0B0 = 38;
  LOBYTE(word_4F9E08C) = word_4F9E08C & 0x9F | 0x20;
  qword_4F9E0A8 = (__int64)"Verify dominator info (time consuming)";
  sub_16B88A0(&qword_4F9E080);
  return __cxa_atexit(sub_13F9A70, &qword_4F9E080, &qword_4A427C0);
}
