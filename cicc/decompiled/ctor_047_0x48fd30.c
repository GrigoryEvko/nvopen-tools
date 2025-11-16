// Function: ctor_047
// Address: 0x48fd30
//
int ctor_047()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4F86640 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F866BC = 1;
  qword_4F86690 = 0x100000000LL;
  dword_4F8664C &= 0x8000u;
  qword_4F86658 = 0;
  qword_4F86660 = 0;
  qword_4F86668 = 0;
  dword_4F86648 = v0;
  word_4F86650 = 0;
  qword_4F86670 = 0;
  qword_4F86678 = 0;
  qword_4F86680 = 0;
  qword_4F86688 = (__int64)&unk_4F86698;
  qword_4F866A0 = 0;
  qword_4F866A8 = (__int64)&unk_4F866C0;
  qword_4F866B0 = 1;
  dword_4F866B8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F86690;
  v3 = (unsigned int)qword_4F86690 + 1LL;
  if ( v3 > HIDWORD(qword_4F86690) )
  {
    sub_C8D5F0((char *)&unk_4F86698 - 16, &unk_4F86698, v3, 8);
    v2 = (unsigned int)qword_4F86690;
  }
  *(_QWORD *)(qword_4F86688 + 8 * v2) = v1;
  LODWORD(qword_4F86690) = qword_4F86690 + 1;
  qword_4F866C8 = 0;
  qword_4F866D0 = (__int64)&unk_49D9748;
  qword_4F866D8 = 0;
  qword_4F86640 = (__int64)&unk_49DC090;
  qword_4F866E0 = (__int64)&unk_49DC1D0;
  qword_4F86700 = (__int64)nullsub_23;
  qword_4F866F8 = (__int64)sub_984030;
  sub_C53080(&qword_4F86640, "verify-assumption-cache", 23);
  qword_4F86670 = 39;
  LOBYTE(qword_4F866C8) = 0;
  LOBYTE(dword_4F8664C) = dword_4F8664C & 0x9F | 0x20;
  qword_4F86668 = (__int64)"Enable verification of assumption cache";
  LOWORD(qword_4F866D8) = 256;
  sub_C53130(&qword_4F86640);
  return __cxa_atexit(sub_984900, &qword_4F86640, &qword_4A427C0);
}
