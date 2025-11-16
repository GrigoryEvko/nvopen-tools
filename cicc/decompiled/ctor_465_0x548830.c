// Function: ctor_465
// Address: 0x548830
//
int ctor_465()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_4FFFE60 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FFFEDC = 1;
  qword_4FFFEB0 = 0x100000000LL;
  dword_4FFFE6C &= 0x8000u;
  qword_4FFFE78 = 0;
  qword_4FFFE80 = 0;
  qword_4FFFE88 = 0;
  dword_4FFFE68 = v0;
  word_4FFFE70 = 0;
  qword_4FFFE90 = 0;
  qword_4FFFE98 = 0;
  qword_4FFFEA0 = 0;
  qword_4FFFEA8 = (__int64)&unk_4FFFEB8;
  qword_4FFFEC0 = 0;
  qword_4FFFEC8 = (__int64)&unk_4FFFEE0;
  qword_4FFFED0 = 1;
  dword_4FFFED8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FFFEB0;
  v3 = (unsigned int)qword_4FFFEB0 + 1LL;
  if ( v3 > HIDWORD(qword_4FFFEB0) )
  {
    sub_C8D5F0((char *)&unk_4FFFEB8 - 16, &unk_4FFFEB8, v3, 8);
    v2 = (unsigned int)qword_4FFFEB0;
  }
  *(_QWORD *)(qword_4FFFEA8 + 8 * v2) = v1;
  qword_4FFFEF0 = (__int64)&unk_49D9728;
  LODWORD(qword_4FFFEB0) = qword_4FFFEB0 + 1;
  qword_4FFFEE8 = 0;
  qword_4FFFE60 = (__int64)&unk_49DBF10;
  qword_4FFFF00 = (__int64)&unk_49DC290;
  qword_4FFFEF8 = 0;
  qword_4FFFF20 = (__int64)nullsub_24;
  qword_4FFFF18 = (__int64)sub_984050;
  sub_C53080(&qword_4FFFE60, "runtime-check-per-loop-load-elim", 32);
  qword_4FFFE90 = 62;
  LODWORD(qword_4FFFEE8) = 1;
  BYTE4(qword_4FFFEF8) = 1;
  LODWORD(qword_4FFFEF8) = 1;
  LOBYTE(dword_4FFFE6C) = dword_4FFFE6C & 0x9F | 0x20;
  qword_4FFFE88 = (__int64)"Max number of memchecks allowed per eliminated load on average";
  sub_C53130(&qword_4FFFE60);
  __cxa_atexit(sub_984970, &qword_4FFFE60, &qword_4A427C0);
  qword_4FFFD80 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FFFD8C &= 0x8000u;
  word_4FFFD90 = 0;
  qword_4FFFDD0 = 0x100000000LL;
  qword_4FFFD98 = 0;
  qword_4FFFDA0 = 0;
  qword_4FFFDA8 = 0;
  dword_4FFFD88 = v4;
  qword_4FFFDB0 = 0;
  qword_4FFFDB8 = 0;
  qword_4FFFDC0 = 0;
  qword_4FFFDC8 = (__int64)&unk_4FFFDD8;
  qword_4FFFDE0 = 0;
  qword_4FFFDE8 = (__int64)&unk_4FFFE00;
  qword_4FFFDF0 = 1;
  dword_4FFFDF8 = 0;
  byte_4FFFDFC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FFFDD0;
  v7 = (unsigned int)qword_4FFFDD0 + 1LL;
  if ( v7 > HIDWORD(qword_4FFFDD0) )
  {
    sub_C8D5F0((char *)&unk_4FFFDD8 - 16, &unk_4FFFDD8, v7, 8);
    v6 = (unsigned int)qword_4FFFDD0;
  }
  *(_QWORD *)(qword_4FFFDC8 + 8 * v6) = v5;
  qword_4FFFE10 = (__int64)&unk_49D9728;
  LODWORD(qword_4FFFDD0) = qword_4FFFDD0 + 1;
  qword_4FFFE08 = 0;
  qword_4FFFD80 = (__int64)&unk_49DBF10;
  qword_4FFFE20 = (__int64)&unk_49DC290;
  qword_4FFFE18 = 0;
  qword_4FFFE40 = (__int64)nullsub_24;
  qword_4FFFE38 = (__int64)sub_984050;
  sub_C53080(&qword_4FFFD80, "loop-load-elimination-scev-check-threshold", 42);
  LODWORD(qword_4FFFE08) = 8;
  BYTE4(qword_4FFFE18) = 1;
  LODWORD(qword_4FFFE18) = 8;
  qword_4FFFDB0 = 67;
  LOBYTE(dword_4FFFD8C) = dword_4FFFD8C & 0x9F | 0x20;
  qword_4FFFDA8 = (__int64)"The maximum number of SCEV checks allowed for Loop Load Elimination";
  sub_C53130(&qword_4FFFD80);
  return __cxa_atexit(sub_984970, &qword_4FFFD80, &qword_4A427C0);
}
