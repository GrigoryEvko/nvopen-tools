// Function: ctor_020
// Address: 0x455980
//
_QWORD *ctor_020()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax

  qword_4F81360 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8136C = word_4F8136C & 0x8000;
  unk_4F81370 = 0;
  qword_4F813A8[1] = 0x100000000LL;
  unk_4F81368 = v0;
  unk_4F81378 = 0;
  unk_4F81380 = 0;
  unk_4F81388 = 0;
  unk_4F81390 = 0;
  unk_4F81398 = 0;
  unk_4F813A0 = 0;
  qword_4F813A8[0] = &qword_4F813A8[2];
  qword_4F813A8[3] = 0;
  qword_4F813A8[4] = &qword_4F813A8[7];
  qword_4F813A8[5] = 1;
  LODWORD(qword_4F813A8[6]) = 0;
  BYTE4(qword_4F813A8[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_4F813A8[1]);
  if ( (unsigned __int64)LODWORD(qword_4F813A8[1]) + 1 > HIDWORD(qword_4F813A8[1]) )
  {
    sub_C8D5F0(qword_4F813A8, &qword_4F813A8[2], LODWORD(qword_4F813A8[1]) + 1LL, 8);
    v2 = LODWORD(qword_4F813A8[1]);
  }
  *(_QWORD *)(qword_4F813A8[0] + 8 * v2) = v1;
  ++LODWORD(qword_4F813A8[1]);
  qword_4F813A8[8] = 0;
  qword_4F813A8[9] = &unk_49D9748;
  qword_4F813A8[10] = 0;
  qword_4F81360 = &unk_49DC090;
  qword_4F813A8[11] = &unk_49DC1D0;
  qword_4F813A8[15] = nullsub_23;
  qword_4F813A8[14] = sub_984030;
  sub_C53080(&qword_4F81360, "enable-fs-discriminator", 23);
  unk_4F81390 = 43;
  LOBYTE(word_4F8136C) = word_4F8136C & 0x9F | 0x20;
  unk_4F81388 = "Enable adding flow sensitive discriminators";
  sub_C53130(&qword_4F81360);
  __cxa_atexit(sub_984900, &qword_4F81360, &qword_4A427C0);
  qword_4F81350[0] = -1;
  qword_4F81350[1] = 0;
  return qword_4F81350;
}
