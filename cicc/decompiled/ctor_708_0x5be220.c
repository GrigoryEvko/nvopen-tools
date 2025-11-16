// Function: ctor_708
// Address: 0x5be220
//
int __fastcall ctor_708(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx

  qword_50510A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_505111C = 1;
  qword_50510F0 = 0x100000000LL;
  dword_50510AC &= 0x8000u;
  qword_50510B8 = 0;
  qword_50510C0 = 0;
  qword_50510C8 = 0;
  dword_50510A8 = v4;
  word_50510B0 = 0;
  qword_50510D0 = 0;
  qword_50510D8 = 0;
  qword_50510E0 = 0;
  qword_50510E8 = (__int64)&unk_50510F8;
  qword_5051100 = 0;
  qword_5051108 = (__int64)&unk_5051120;
  qword_5051110 = 1;
  dword_5051118 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50510F0;
  v7 = (unsigned int)qword_50510F0 + 1LL;
  if ( v7 > HIDWORD(qword_50510F0) )
  {
    sub_C8D5F0((char *)&unk_50510F8 - 16, &unk_50510F8, v7, 8);
    v6 = (unsigned int)qword_50510F0;
  }
  *(_QWORD *)(qword_50510E8 + 8 * v6) = v5;
  LODWORD(qword_50510F0) = qword_50510F0 + 1;
  qword_5051128 = 0;
  qword_5051130 = (__int64)&unk_49D9748;
  qword_5051138 = 0;
  qword_50510A0 = (__int64)&unk_49DC090;
  qword_5051140 = (__int64)&unk_49DC1D0;
  qword_5051160 = (__int64)nullsub_23;
  qword_5051158 = (__int64)sub_984030;
  sub_C53080(&qword_50510A0, "disable-dfa-sched", 17);
  qword_50510D0 = 36;
  LOBYTE(dword_50510AC) = dword_50510AC & 0x9F | 0x20;
  qword_50510C8 = (__int64)"Disable use of DFA during scheduling";
  sub_C53130(&qword_50510A0);
  __cxa_atexit(sub_984900, &qword_50510A0, &qword_4A427C0);
  qword_5050FC0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_50510A0, v8, v9), 1u);
  dword_5050FCC &= 0x8000u;
  word_5050FD0 = 0;
  qword_5051010 = 0x100000000LL;
  qword_5050FD8 = 0;
  qword_5050FE0 = 0;
  qword_5050FE8 = 0;
  dword_5050FC8 = v10;
  qword_5050FF0 = 0;
  qword_5050FF8 = 0;
  qword_5051000 = 0;
  qword_5051008 = (__int64)&unk_5051018;
  qword_5051020 = 0;
  qword_5051028 = (__int64)&unk_5051040;
  qword_5051030 = 1;
  dword_5051038 = 0;
  byte_505103C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5051010;
  v13 = (unsigned int)qword_5051010 + 1LL;
  if ( v13 > HIDWORD(qword_5051010) )
  {
    sub_C8D5F0((char *)&unk_5051018 - 16, &unk_5051018, v13, 8);
    v12 = (unsigned int)qword_5051010;
  }
  *(_QWORD *)(qword_5051008 + 8 * v12) = v11;
  LODWORD(qword_5051010) = qword_5051010 + 1;
  qword_5051048 = 0;
  qword_5051050 = (__int64)&unk_49DA090;
  qword_5051058 = 0;
  qword_5050FC0 = (__int64)&unk_49DBF90;
  qword_5051060 = (__int64)&unk_49DC230;
  qword_5051080 = (__int64)nullsub_58;
  qword_5051078 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_5050FC0, "dfa-sched-reg-pressure-threshold", 32);
  LODWORD(qword_5051048) = 5;
  BYTE4(qword_5051058) = 1;
  LODWORD(qword_5051058) = 5;
  qword_5050FF0 = 50;
  LOBYTE(dword_5050FCC) = dword_5050FCC & 0x9F | 0x20;
  qword_5050FE8 = (__int64)"Track reg pressure and switch priority to in-depth";
  sub_C53130(&qword_5050FC0);
  return __cxa_atexit(sub_B2B680, &qword_5050FC0, &qword_4A427C0);
}
