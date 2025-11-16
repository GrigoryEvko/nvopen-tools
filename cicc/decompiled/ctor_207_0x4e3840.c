// Function: ctor_207
// Address: 0x4e3840
//
int ctor_207()
{
  int v0; // edx

  qword_4FB0660 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB066C &= 0xF000u;
  qword_4FB0670 = 0;
  qword_4FB06A8 = (__int64)qword_4FA01C0;
  qword_4FB0678 = 0;
  qword_4FB0680 = 0;
  qword_4FB0688 = 0;
  dword_4FB0668 = v0;
  qword_4FB06B8 = (__int64)&unk_4FB06D8;
  qword_4FB06C0 = (__int64)&unk_4FB06D8;
  qword_4FB0690 = 0;
  qword_4FB0698 = 0;
  qword_4FB0708 = (__int64)&unk_49E74E8;
  word_4FB0710 = 256;
  qword_4FB06A0 = 0;
  qword_4FB06B0 = 0;
  qword_4FB0660 = (__int64)&unk_49EEC70;
  qword_4FB06C8 = 4;
  byte_4FB06F8 = 0;
  qword_4FB0718 = (__int64)&unk_49EEDB0;
  dword_4FB06D0 = 0;
  byte_4FB0700 = 0;
  sub_16B8280(&qword_4FB0660, "use-lir-code-size-heurs", 23);
  qword_4FB0688 = (__int64)"Use loop idiom recognition code size heuristics when compilingwith -Os/-Oz";
  word_4FB0710 = 257;
  byte_4FB0700 = 1;
  qword_4FB0690 = 74;
  LOBYTE(word_4FB066C) = word_4FB066C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FB0660);
  return __cxa_atexit(sub_12EDEC0, &qword_4FB0660, &qword_4A427C0);
}
