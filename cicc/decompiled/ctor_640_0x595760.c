// Function: ctor_640
// Address: 0x595760
//
int __fastcall ctor_640(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // rdx

  qword_5035720 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_503579C = 1;
  qword_5035770 = 0x100000000LL;
  dword_503572C &= 0x8000u;
  qword_5035738 = 0;
  qword_5035740 = 0;
  qword_5035748 = 0;
  dword_5035728 = v4;
  word_5035730 = 0;
  qword_5035750 = 0;
  qword_5035758 = 0;
  qword_5035760 = 0;
  qword_5035768 = (__int64)&unk_5035778;
  qword_5035780 = 0;
  qword_5035788 = (__int64)&unk_50357A0;
  qword_5035790 = 1;
  dword_5035798 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5035770;
  v7 = (unsigned int)qword_5035770 + 1LL;
  if ( v7 > HIDWORD(qword_5035770) )
  {
    sub_C8D5F0((char *)&unk_5035778 - 16, &unk_5035778, v7, 8);
    v6 = (unsigned int)qword_5035770;
  }
  *(_QWORD *)(qword_5035768 + 8 * v6) = v5;
  qword_50357B0 = (__int64)&unk_49DB998;
  LODWORD(qword_5035770) = qword_5035770 + 1;
  byte_50357C0 = 0;
  qword_5035720 = (__int64)&unk_49DB9B8;
  qword_50357C8 = (__int64)&unk_49DC2C0;
  qword_50357A8 = 0;
  qword_50357E8 = (__int64)nullsub_121;
  qword_50357B8 = 0;
  qword_50357E0 = (__int64)sub_C1A370;
  sub_C53080(&qword_5035720, "sbvec-stop-at", 13);
  qword_50357A8 = -1;
  byte_50357C0 = 1;
  qword_50357B8 = -1;
  qword_5035750 = 75;
  LOBYTE(dword_503572C) = dword_503572C & 0x9F | 0x20;
  qword_5035748 = (__int64)"Vectorize if the invocation count is < than this. 0 disables vectorization.";
  sub_C53130(&qword_5035720);
  __cxa_atexit(sub_C1A610, &qword_5035720, &qword_4A427C0);
  qword_5035640 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_C1A610, &qword_5035720, v8, v9), 1u);
  dword_503564C &= 0x8000u;
  word_5035650 = 0;
  qword_5035690 = 0x100000000LL;
  qword_5035658 = 0;
  qword_5035660 = 0;
  qword_5035668 = 0;
  dword_5035648 = v10;
  qword_5035670 = 0;
  qword_5035678 = 0;
  qword_5035680 = 0;
  qword_5035688 = (__int64)&unk_5035698;
  qword_50356A0 = 0;
  qword_50356A8 = (__int64)&unk_50356C0;
  qword_50356B0 = 1;
  dword_50356B8 = 0;
  byte_50356BC = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5035690;
  v13 = (unsigned int)qword_5035690 + 1LL;
  if ( v13 > HIDWORD(qword_5035690) )
  {
    sub_C8D5F0((char *)&unk_5035698 - 16, &unk_5035698, v13, 8);
    v12 = (unsigned int)qword_5035690;
  }
  *(_QWORD *)(qword_5035688 + 8 * v12) = v11;
  qword_50356D0 = (__int64)&unk_49DB998;
  LODWORD(qword_5035690) = qword_5035690 + 1;
  byte_50356E0 = 0;
  qword_5035640 = (__int64)&unk_49DB9B8;
  qword_50356E8 = (__int64)&unk_49DC2C0;
  qword_50356C8 = 0;
  qword_5035708 = (__int64)nullsub_121;
  qword_50356D8 = 0;
  qword_5035700 = (__int64)sub_C1A370;
  sub_C53080(&qword_5035640, "sbvec-stop-bndl", 15);
  qword_50356C8 = -1;
  byte_50356E0 = 1;
  qword_50356D8 = -1;
  qword_5035670 = 34;
  LOBYTE(dword_503564C) = dword_503564C & 0x9F | 0x20;
  qword_5035668 = (__int64)"Vectorize up to this many bundles.";
  sub_C53130(&qword_5035640);
  return __cxa_atexit(sub_C1A610, &qword_5035640, &qword_4A427C0);
}
