// Function: ctor_539
// Address: 0x56b5a0
//
int ctor_539()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_5016140 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_50161BC = 1;
  qword_5016190 = 0x100000000LL;
  dword_501614C &= 0x8000u;
  qword_5016158 = 0;
  qword_5016160 = 0;
  qword_5016168 = 0;
  dword_5016148 = v0;
  word_5016150 = 0;
  qword_5016170 = 0;
  qword_5016178 = 0;
  qword_5016180 = 0;
  qword_5016188 = (__int64)&unk_5016198;
  qword_50161A0 = 0;
  qword_50161A8 = (__int64)&unk_50161C0;
  qword_50161B0 = 1;
  dword_50161B8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5016190;
  v3 = (unsigned int)qword_5016190 + 1LL;
  if ( v3 > HIDWORD(qword_5016190) )
  {
    sub_C8D5F0((char *)&unk_5016198 - 16, &unk_5016198, v3, 8);
    v2 = (unsigned int)qword_5016190;
  }
  *(_QWORD *)(qword_5016188 + 8 * v2) = v1;
  LODWORD(qword_5016190) = qword_5016190 + 1;
  qword_50161C8 = 0;
  qword_50161D0 = (__int64)&unk_49D9748;
  qword_50161D8 = 0;
  qword_5016140 = (__int64)&unk_49DC090;
  qword_50161E0 = (__int64)&unk_49DC1D0;
  qword_5016200 = (__int64)nullsub_23;
  qword_50161F8 = (__int64)sub_984030;
  sub_C53080(&qword_5016140, "use-max-local-array-alignment", 29);
  LOBYTE(qword_50161C8) = 0;
  LOWORD(qword_50161D8) = 256;
  qword_5016170 = 39;
  LOBYTE(dword_501614C) = dword_501614C & 0x9F | 0x20;
  qword_5016168 = (__int64)"Use mmaximum alignment for local memory";
  sub_C53130(&qword_5016140);
  return __cxa_atexit(sub_984900, &qword_5016140, &qword_4A427C0);
}
