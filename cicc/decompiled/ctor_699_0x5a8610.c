// Function: ctor_699
// Address: 0x5a8610
//
int __fastcall ctor_699(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5041580 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_50415FC = 1;
  qword_50415D0 = 0x100000000LL;
  dword_504158C &= 0x8000u;
  qword_5041598 = 0;
  qword_50415A0 = 0;
  qword_50415A8 = 0;
  dword_5041588 = v4;
  word_5041590 = 0;
  qword_50415B0 = 0;
  qword_50415B8 = 0;
  qword_50415C0 = 0;
  qword_50415C8 = (__int64)&unk_50415D8;
  qword_50415E0 = 0;
  qword_50415E8 = (__int64)&unk_5041600;
  qword_50415F0 = 1;
  dword_50415F8 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50415D0;
  v7 = (unsigned int)qword_50415D0 + 1LL;
  if ( v7 > HIDWORD(qword_50415D0) )
  {
    sub_C8D5F0((char *)&unk_50415D8 - 16, &unk_50415D8, v7, 8);
    v6 = (unsigned int)qword_50415D0;
  }
  *(_QWORD *)(qword_50415C8 + 8 * v6) = v5;
  LODWORD(qword_50415D0) = qword_50415D0 + 1;
  qword_5041608 = 0;
  qword_5041610 = (__int64)&unk_49D9748;
  qword_5041618 = 0;
  qword_5041580 = (__int64)&unk_49DC090;
  qword_5041620 = (__int64)&unk_49DC1D0;
  qword_5041640 = (__int64)nullsub_23;
  qword_5041638 = (__int64)sub_984030;
  sub_C53080(&qword_5041580, "interactive-model-runner-echo-reply", 35);
  LOBYTE(qword_5041608) = 0;
  LOWORD(qword_5041618) = 256;
  qword_50415B0 = 109;
  LOBYTE(dword_504158C) = dword_504158C & 0x9F | 0x20;
  qword_50415A8 = (__int64)"The InteractiveModelRunner will echo back to stderr the data received from the host (for debu"
                           "gging purposes).";
  sub_C53130(&qword_5041580);
  return __cxa_atexit(sub_984900, &qword_5041580, &qword_4A427C0);
}
