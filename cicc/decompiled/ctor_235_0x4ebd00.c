// Function: ctor_235
// Address: 0x4ebd00
//
int ctor_235()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax

  qword_4FB6520 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB652C &= 0xF000u;
  qword_4FB6530 = 0;
  qword_4FB6538 = 0;
  qword_4FB6540 = 0;
  qword_4FB6548 = 0;
  qword_4FB6550 = 0;
  dword_4FB6528 = v0;
  qword_4FB6558 = 0;
  qword_4FB6568 = (__int64)qword_4FA01C0;
  qword_4FB6578 = (__int64)&unk_4FB6598;
  qword_4FB6580 = (__int64)&unk_4FB6598;
  qword_4FB6560 = 0;
  qword_4FB6570 = 0;
  word_4FB65D0 = 256;
  qword_4FB65C8 = (__int64)&unk_49E74E8;
  qword_4FB6588 = 4;
  qword_4FB6520 = (__int64)&unk_49EEC70;
  byte_4FB65B8 = 0;
  qword_4FB65D8 = (__int64)&unk_49EEDB0;
  dword_4FB6590 = 0;
  byte_4FB65C0 = 0;
  sub_16B8280(&qword_4FB6520, "initlocals", 10);
  word_4FB65D0 = 256;
  byte_4FB65C0 = 0;
  qword_4FB6550 = 55;
  LOBYTE(word_4FB652C) = word_4FB652C & 0x98 | 0x21;
  qword_4FB6548 = (__int64)"After inlinig callee, initialize locals at the callsite";
  sub_16B88A0(&qword_4FB6520);
  __cxa_atexit(sub_12EDEC0, &qword_4FB6520, &qword_4A427C0);
  qword_4FB6440 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB64F0 = 256;
  word_4FB644C &= 0xF000u;
  qword_4FB6450 = 0;
  qword_4FB6458 = 0;
  qword_4FB6460 = 0;
  dword_4FB6448 = v1;
  qword_4FB64E8 = (__int64)&unk_49E74E8;
  qword_4FB6488 = (__int64)qword_4FA01C0;
  qword_4FB6498 = (__int64)&unk_4FB64B8;
  qword_4FB64A0 = (__int64)&unk_4FB64B8;
  qword_4FB6440 = (__int64)&unk_49EEC70;
  qword_4FB64F8 = (__int64)&unk_49EEDB0;
  qword_4FB6468 = 0;
  qword_4FB6470 = 0;
  qword_4FB6478 = 0;
  qword_4FB6480 = 0;
  qword_4FB6490 = 0;
  qword_4FB64A8 = 4;
  dword_4FB64B0 = 0;
  byte_4FB64D8 = 0;
  byte_4FB64E0 = 0;
  sub_16B8280(&qword_4FB6440, "enable-noalias-to-md-conversion", 31);
  word_4FB64F0 = 257;
  byte_4FB64E0 = 1;
  qword_4FB6470 = 55;
  LOBYTE(word_4FB644C) = word_4FB644C & 0x9F | 0x20;
  qword_4FB6468 = (__int64)"Convert noalias attributes to metadata during inlining.";
  sub_16B88A0(&qword_4FB6440);
  __cxa_atexit(sub_12EDEC0, &qword_4FB6440, &qword_4A427C0);
  qword_4FB6360 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB6410 = 256;
  word_4FB636C &= 0xF000u;
  qword_4FB6370 = 0;
  qword_4FB6378 = 0;
  qword_4FB6380 = 0;
  dword_4FB6368 = v2;
  qword_4FB6408 = (__int64)&unk_49E74E8;
  qword_4FB63A8 = (__int64)qword_4FA01C0;
  qword_4FB63B8 = (__int64)&unk_4FB63D8;
  qword_4FB63C0 = (__int64)&unk_4FB63D8;
  qword_4FB6360 = (__int64)&unk_49EEC70;
  qword_4FB6418 = (__int64)&unk_49EEDB0;
  qword_4FB6388 = 0;
  qword_4FB6390 = 0;
  qword_4FB6398 = 0;
  qword_4FB63A0 = 0;
  qword_4FB63B0 = 0;
  qword_4FB63C8 = 4;
  dword_4FB63D0 = 0;
  byte_4FB63F8 = 0;
  byte_4FB6400 = 0;
  sub_16B8280(&qword_4FB6360, "preserve-alignment-assumptions-during-inlining", 46);
  byte_4FB6400 = 1;
  word_4FB6410 = 257;
  qword_4FB6390 = 56;
  LOBYTE(word_4FB636C) = word_4FB636C & 0x9F | 0x20;
  qword_4FB6388 = (__int64)"Convert align attributes to assumptions during inlining.";
  sub_16B88A0(&qword_4FB6360);
  return __cxa_atexit(sub_12EDEC0, &qword_4FB6360, &qword_4A427C0);
}
