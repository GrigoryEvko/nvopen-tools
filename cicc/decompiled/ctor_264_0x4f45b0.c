// Function: ctor_264
// Address: 0x4f45b0
//
int ctor_264()
{
  int v0; // eax
  int v1; // eax

  qword_4FBD3E0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FBD3EC &= 0xF000u;
  qword_4FBD3F0 = 0;
  qword_4FBD3F8 = 0;
  qword_4FBD400 = 0;
  qword_4FBD408 = 0;
  qword_4FBD410 = 0;
  dword_4FBD3E8 = v0;
  qword_4FBD418 = 0;
  qword_4FBD428 = (__int64)qword_4FA01C0;
  qword_4FBD438 = (__int64)&unk_4FBD458;
  qword_4FBD440 = (__int64)&unk_4FBD458;
  qword_4FBD420 = 0;
  qword_4FBD430 = 0;
  qword_4FBD488 = (__int64)&unk_49E74C8;
  qword_4FBD448 = 4;
  qword_4FBD3E0 = (__int64)&unk_49EEB70;
  dword_4FBD450 = 0;
  qword_4FBD498 = (__int64)&unk_49EEDF0;
  byte_4FBD478 = 0;
  dword_4FBD480 = 0;
  byte_4FBD494 = 1;
  dword_4FBD490 = 0;
  sub_16B8280(&qword_4FBD3E0, "dump-ip-msp", 11);
  dword_4FBD480 = 0;
  byte_4FBD494 = 1;
  dword_4FBD490 = 0;
  qword_4FBD410 = 63;
  LOBYTE(word_4FBD3EC) = word_4FBD3EC & 0x9F | 0x20;
  qword_4FBD408 = (__int64)"Dump information from Inter-Procedural Memory Space Propagation";
  sub_16B88A0(&qword_4FBD3E0);
  __cxa_atexit(sub_12EDEA0, &qword_4FBD3E0, &qword_4A427C0);
  qword_4FBD300 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FBD30C &= 0xF000u;
  qword_4FBD310 = 0;
  qword_4FBD318 = 0;
  qword_4FBD320 = 0;
  qword_4FBD328 = 0;
  qword_4FBD330 = 0;
  dword_4FBD308 = v1;
  qword_4FBD3A8 = (__int64)&unk_49E74C8;
  qword_4FBD348 = (__int64)qword_4FA01C0;
  qword_4FBD358 = (__int64)&unk_4FBD378;
  qword_4FBD360 = (__int64)&unk_4FBD378;
  qword_4FBD300 = (__int64)&unk_49EEB70;
  qword_4FBD3B8 = (__int64)&unk_49EEDF0;
  qword_4FBD338 = 0;
  qword_4FBD340 = 0;
  qword_4FBD350 = 0;
  qword_4FBD368 = 4;
  dword_4FBD370 = 0;
  byte_4FBD398 = 0;
  dword_4FBD3A0 = 0;
  byte_4FBD3B4 = 1;
  dword_4FBD3B0 = 0;
  sub_16B8280(&qword_4FBD300, "do-clone-for-ip-msp", 19);
  dword_4FBD3A0 = -1;
  byte_4FBD3B4 = 1;
  dword_4FBD3B0 = -1;
  qword_4FBD330 = 70;
  LOBYTE(word_4FBD30C) = word_4FBD30C & 0x9F | 0x20;
  qword_4FBD328 = (__int64)"Control number of clones for inter-procedural Memory Space Propagation";
  sub_16B88A0(&qword_4FBD300);
  return __cxa_atexit(sub_12EDEA0, &qword_4FBD300, &qword_4A427C0);
}
