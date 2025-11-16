// Function: ctor_194
// Address: 0x4dfc90
//
int ctor_194()
{
  int v0; // edx

  qword_4FADEC0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_4FADED0 = 0;
  qword_4FADED8 = 0;
  qword_4FADEE0 = 0;
  qword_4FADEE8 = 0;
  word_4FADECC = word_4FADECC & 0xF000 | 1;
  qword_4FADEF0 = 0;
  qword_4FADF08 = (__int64)qword_4FA01C0;
  qword_4FADF18 = (__int64)&unk_4FADF38;
  qword_4FADF20 = (__int64)&unk_4FADF38;
  dword_4FADEC8 = v0;
  qword_4FADEF8 = 0;
  qword_4FADEC0 = (__int64)&unk_49E75F8;
  qword_4FADF00 = 0;
  qword_4FADF10 = 0;
  qword_4FADF90 = (__int64)&unk_49EEE90;
  qword_4FADF28 = 4;
  dword_4FADF30 = 0;
  byte_4FADF58 = 0;
  qword_4FADF60 = 0;
  qword_4FADF68 = 0;
  qword_4FADF70 = 0;
  qword_4FADF78 = 0;
  qword_4FADF80 = 0;
  qword_4FADF88 = 0;
  sub_16B8280(&qword_4FADEC0, "force-attribute", 15);
  qword_4FADEF0 = 176;
  LOBYTE(word_4FADECC) = word_4FADECC & 0x9F | 0x20;
  qword_4FADEE8 = (__int64)"Add an attribute to a function. This should be a pair of 'function-name:attribute-name', for "
                           "example -force-attribute=foo:noinline. This option can be specified multiple times.";
  sub_16B88A0(&qword_4FADEC0);
  return __cxa_atexit(sub_12F08D0, &qword_4FADEC0, &qword_4A427C0);
}
