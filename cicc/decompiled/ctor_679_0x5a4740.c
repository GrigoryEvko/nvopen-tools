// Function: ctor_679
// Address: 0x5a4740
//
int __fastcall ctor_679(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax

  qword_503F180 = &unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  *(_DWORD *)&word_503F18C = word_503F18C & 0x8000;
  unk_503F188 = v4;
  qword_503F1C8[1] = 0x100000000LL;
  unk_503F190 = 0;
  unk_503F198 = 0;
  unk_503F1A0 = 0;
  unk_503F1A8 = 0;
  unk_503F1B0 = 0;
  unk_503F1B8 = 0;
  unk_503F1C0 = 0;
  qword_503F1C8[0] = &qword_503F1C8[2];
  qword_503F1C8[3] = 0;
  qword_503F1C8[4] = &qword_503F1C8[7];
  qword_503F1C8[5] = 1;
  LODWORD(qword_503F1C8[6]) = 0;
  BYTE4(qword_503F1C8[6]) = 1;
  v5 = sub_C57470();
  v6 = LODWORD(qword_503F1C8[1]);
  if ( (unsigned __int64)LODWORD(qword_503F1C8[1]) + 1 > HIDWORD(qword_503F1C8[1]) )
  {
    sub_C8D5F0(qword_503F1C8, &qword_503F1C8[2], LODWORD(qword_503F1C8[1]) + 1LL, 8);
    v6 = LODWORD(qword_503F1C8[1]);
  }
  *(_QWORD *)(qword_503F1C8[0] + 8 * v6) = v5;
  ++LODWORD(qword_503F1C8[1]);
  qword_503F1C8[8] = 0;
  qword_503F1C8[9] = &unk_49D9748;
  qword_503F1C8[10] = 0;
  qword_503F180 = &unk_49DC090;
  qword_503F1C8[11] = &unk_49DC1D0;
  qword_503F1C8[15] = nullsub_23;
  qword_503F1C8[14] = sub_984030;
  sub_C53080(&qword_503F180, "improved-fs-discriminator", 25);
  LOBYTE(qword_503F1C8[8]) = 0;
  unk_503F1B0 = 72;
  LOBYTE(word_503F18C) = word_503F18C & 0x9F | 0x20;
  LOWORD(qword_503F1C8[10]) = 256;
  unk_503F1A8 = "New FS discriminators encoding (incompatible with the original encoding)";
  sub_C53130(&qword_503F180);
  return __cxa_atexit(sub_984900, &qword_503F180, &qword_4A427C0);
}
