// Function: ctor_657
// Address: 0x59bd50
//
int __fastcall ctor_657(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_50396E0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  dword_50396EC &= 0x8000u;
  word_50396F0 = 0;
  qword_5039730 = 0x100000000LL;
  qword_50396F8 = 0;
  qword_5039700 = 0;
  qword_5039708 = 0;
  dword_50396E8 = v4;
  qword_5039710 = 0;
  qword_5039718 = 0;
  qword_5039720 = 0;
  qword_5039728 = (__int64)&unk_5039738;
  qword_5039740 = 0;
  qword_5039748 = (__int64)&unk_5039760;
  qword_5039750 = 1;
  dword_5039758 = 0;
  byte_503975C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5039730;
  v7 = (unsigned int)qword_5039730 + 1LL;
  if ( v7 > HIDWORD(qword_5039730) )
  {
    sub_C8D5F0((char *)&unk_5039738 - 16, &unk_5039738, v7, 8);
    v6 = (unsigned int)qword_5039730;
  }
  *(_QWORD *)(qword_5039728 + 8 * v6) = v5;
  LODWORD(qword_5039730) = qword_5039730 + 1;
  qword_5039768 = 0;
  qword_5039770 = (__int64)&unk_49D9748;
  qword_5039778 = 0;
  qword_50396E0 = (__int64)&unk_49DC090;
  qword_5039780 = (__int64)&unk_49DC1D0;
  qword_50397A0 = (__int64)nullsub_23;
  qword_5039798 = (__int64)sub_984030;
  sub_C53080(&qword_50396E0, "dag-dump-verbose", 16);
  qword_5039710 = 58;
  LOBYTE(dword_50396EC) = dword_50396EC & 0x9F | 0x20;
  qword_5039708 = (__int64)"Display more information when dumping selection DAG nodes.";
  sub_C53130(&qword_50396E0);
  return __cxa_atexit(sub_984900, &qword_50396E0, &qword_4A427C0);
}
