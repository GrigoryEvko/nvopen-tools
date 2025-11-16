// Function: ctor_261
// Address: 0x4f2690
//
int ctor_261()
{
  int v0; // edx

  qword_4FBB620 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FBB62C &= 0xF000u;
  qword_4FBB630 = 0;
  qword_4FBB668 = (__int64)qword_4FA01C0;
  qword_4FBB638 = 0;
  qword_4FBB640 = 0;
  qword_4FBB648 = 0;
  dword_4FBB628 = v0;
  qword_4FBB678 = (__int64)&unk_4FBB698;
  qword_4FBB680 = (__int64)&unk_4FBB698;
  qword_4FBB650 = 0;
  qword_4FBB658 = 0;
  qword_4FBB6C8 = (__int64)&unk_49E74E8;
  word_4FBB6D0 = 256;
  qword_4FBB660 = 0;
  qword_4FBB670 = 0;
  qword_4FBB620 = (__int64)&unk_49EEC70;
  qword_4FBB688 = 4;
  byte_4FBB6B8 = 0;
  qword_4FBB6D8 = (__int64)&unk_49EEDB0;
  dword_4FBB690 = 0;
  byte_4FBB6C0 = 0;
  sub_16B8280(&qword_4FBB620, "basic-dbe", 9);
  word_4FBB6D0 = 256;
  byte_4FBB6C0 = 0;
  qword_4FBB650 = 100;
  LOBYTE(word_4FBB62C) = word_4FBB62C & 0x9F | 0x20;
  qword_4FBB648 = (__int64)"Run the basic dead barrier elimination which uses one bit for the liveness of memory at the barrier.";
  sub_16B88A0(&qword_4FBB620);
  return __cxa_atexit(sub_12EDEC0, &qword_4FBB620, &qword_4A427C0);
}
