// Function: ctor_390
// Address: 0x51ddf0
//
int ctor_390()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax

  qword_4FDF3A0 = &unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4FDF3AC = word_4FDF3AC & 0x8000;
  unk_4FDF3A8 = v0;
  qword_4FDF3E8[1] = 0x100000000LL;
  unk_4FDF3B0 = 0;
  unk_4FDF3B8 = 0;
  unk_4FDF3C0 = 0;
  unk_4FDF3C8 = 0;
  unk_4FDF3D0 = 0;
  unk_4FDF3D8 = 0;
  unk_4FDF3E0 = 0;
  qword_4FDF3E8[0] = &qword_4FDF3E8[2];
  qword_4FDF3E8[3] = 0;
  qword_4FDF3E8[4] = &qword_4FDF3E8[7];
  qword_4FDF3E8[5] = 1;
  LODWORD(qword_4FDF3E8[6]) = 0;
  BYTE4(qword_4FDF3E8[6]) = 1;
  v1 = sub_C57470();
  v2 = LODWORD(qword_4FDF3E8[1]);
  if ( (unsigned __int64)LODWORD(qword_4FDF3E8[1]) + 1 > HIDWORD(qword_4FDF3E8[1]) )
  {
    sub_C8D5F0(qword_4FDF3E8, &qword_4FDF3E8[2], LODWORD(qword_4FDF3E8[1]) + 1LL, 8);
    v2 = LODWORD(qword_4FDF3E8[1]);
  }
  *(_QWORD *)(qword_4FDF3E8[0] + 8 * v2) = v1;
  ++LODWORD(qword_4FDF3E8[1]);
  qword_4FDF3E8[8] = 0;
  qword_4FDF3E8[9] = &unk_49D9748;
  qword_4FDF3E8[10] = 0;
  qword_4FDF3A0 = &unk_49DC090;
  qword_4FDF3E8[11] = &unk_49DC1D0;
  qword_4FDF3E8[15] = nullsub_23;
  qword_4FDF3E8[14] = sub_984030;
  sub_C53080(&qword_4FDF3A0, "no-kernel-info-end-lto", 22);
  unk_4FDF3D0 = 63;
  unk_4FDF3C8 = "remove the kernel-info pass at the end of the full LTO pipeline";
  LOWORD(qword_4FDF3E8[10]) = 256;
  LOBYTE(qword_4FDF3E8[8]) = 0;
  LOBYTE(word_4FDF3AC) = word_4FDF3AC & 0x9F | 0x20;
  sub_C53130(&qword_4FDF3A0);
  return __cxa_atexit(sub_984900, &qword_4FDF3A0, &qword_4A427C0);
}
