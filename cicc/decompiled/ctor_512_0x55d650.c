// Function: ctor_512
// Address: 0x55d650
//
int ctor_512()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_500C120 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_500C12C &= 0x8000u;
  word_500C130 = 0;
  qword_500C170 = 0x100000000LL;
  qword_500C138 = 0;
  qword_500C140 = 0;
  qword_500C148 = 0;
  dword_500C128 = v0;
  qword_500C150 = 0;
  qword_500C158 = 0;
  qword_500C160 = 0;
  qword_500C168 = (__int64)&unk_500C178;
  qword_500C180 = 0;
  qword_500C188 = (__int64)&unk_500C1A0;
  qword_500C190 = 1;
  dword_500C198 = 0;
  byte_500C19C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_500C170;
  v3 = (unsigned int)qword_500C170 + 1LL;
  if ( v3 > HIDWORD(qword_500C170) )
  {
    sub_C8D5F0((char *)&unk_500C178 - 16, &unk_500C178, v3, 8);
    v2 = (unsigned int)qword_500C170;
  }
  *(_QWORD *)(qword_500C168 + 8 * v2) = v1;
  LODWORD(qword_500C170) = qword_500C170 + 1;
  qword_500C1A8 = 0;
  qword_500C1B0 = (__int64)&unk_49D9728;
  qword_500C1B8 = 0;
  qword_500C120 = (__int64)&unk_49DBF10;
  qword_500C1C0 = (__int64)&unk_49DC290;
  qword_500C1E0 = (__int64)nullsub_24;
  qword_500C1D8 = (__int64)sub_984050;
  sub_C53080(&qword_500C120, "max-booleans-in-control-flow-hub", 32);
  LODWORD(qword_500C1A8) = 32;
  BYTE4(qword_500C1B8) = 1;
  LODWORD(qword_500C1B8) = 32;
  qword_500C150 = 118;
  LOBYTE(dword_500C12C) = dword_500C12C & 0x9F | 0x20;
  qword_500C148 = (__int64)"Set the maximum number of outgoing blocks for using a boolean value to record the exiting blo"
                           "ck in the ControlFlowHub.";
  sub_C53130(&qword_500C120);
  return __cxa_atexit(sub_984970, &qword_500C120, &qword_4A427C0);
}
