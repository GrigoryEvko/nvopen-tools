// Function: ctor_080
// Address: 0x49df70
//
int ctor_080()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4F8ED80 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F8EDFC = 1;
  qword_4F8EDD0 = 0x100000000LL;
  dword_4F8ED8C &= 0x8000u;
  qword_4F8ED98 = 0;
  qword_4F8EDA0 = 0;
  qword_4F8EDA8 = 0;
  dword_4F8ED88 = v0;
  word_4F8ED90 = 0;
  qword_4F8EDB0 = 0;
  qword_4F8EDB8 = 0;
  qword_4F8EDC0 = 0;
  qword_4F8EDC8 = (__int64)&unk_4F8EDD8;
  qword_4F8EDE0 = 0;
  qword_4F8EDE8 = (__int64)&unk_4F8EE00;
  qword_4F8EDF0 = 1;
  dword_4F8EDF8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F8EDD0;
  v3 = (unsigned int)qword_4F8EDD0 + 1LL;
  if ( v3 > HIDWORD(qword_4F8EDD0) )
  {
    sub_C8D5F0((char *)&unk_4F8EDD8 - 16, &unk_4F8EDD8, v3, 8);
    v2 = (unsigned int)qword_4F8EDD0;
  }
  *(_QWORD *)(qword_4F8EDC8 + 8 * v2) = v1;
  LODWORD(qword_4F8EDD0) = qword_4F8EDD0 + 1;
  qword_4F8EE08 = 0;
  qword_4F8EE10 = (__int64)&unk_49D9748;
  qword_4F8EE18 = 0;
  qword_4F8ED80 = (__int64)&unk_49DC090;
  qword_4F8EE20 = (__int64)&unk_49DC1D0;
  qword_4F8EE40 = (__int64)nullsub_23;
  qword_4F8EE38 = (__int64)sub_984030;
  sub_C53080(&qword_4F8ED80, "disable-last-run-tracking", 25);
  qword_4F8EDB0 = 25;
  LOBYTE(qword_4F8EE08) = 0;
  LOBYTE(dword_4F8ED8C) = dword_4F8ED8C & 0x9F | 0x20;
  qword_4F8EDA8 = (__int64)"Disable last run tracking";
  LOWORD(qword_4F8EE18) = 256;
  sub_C53130(&qword_4F8ED80);
  return __cxa_atexit(sub_984900, &qword_4F8ED80, &qword_4A427C0);
}
