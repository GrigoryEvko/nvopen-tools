// Function: ctor_416
// Address: 0x530c50
//
int ctor_416()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4FEFD00 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FEFD0C &= 0x8000u;
  word_4FEFD10 = 0;
  qword_4FEFD50 = 0x100000000LL;
  qword_4FEFD18 = 0;
  qword_4FEFD20 = 0;
  qword_4FEFD28 = 0;
  dword_4FEFD08 = v0;
  qword_4FEFD30 = 0;
  qword_4FEFD38 = 0;
  qword_4FEFD40 = 0;
  qword_4FEFD48 = (__int64)&unk_4FEFD58;
  qword_4FEFD60 = 0;
  qword_4FEFD68 = (__int64)&unk_4FEFD80;
  qword_4FEFD70 = 1;
  dword_4FEFD78 = 0;
  byte_4FEFD7C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FEFD50;
  v3 = (unsigned int)qword_4FEFD50 + 1LL;
  if ( v3 > HIDWORD(qword_4FEFD50) )
  {
    sub_C8D5F0((char *)&unk_4FEFD58 - 16, &unk_4FEFD58, v3, 8);
    v2 = (unsigned int)qword_4FEFD50;
  }
  *(_QWORD *)(qword_4FEFD48 + 8 * v2) = v1;
  LODWORD(qword_4FEFD50) = qword_4FEFD50 + 1;
  qword_4FEFD88 = 0;
  qword_4FEFD90 = (__int64)&unk_49D9748;
  qword_4FEFD98 = 0;
  qword_4FEFD00 = (__int64)&unk_49DC090;
  qword_4FEFDA0 = (__int64)&unk_49DC1D0;
  qword_4FEFDC0 = (__int64)nullsub_23;
  qword_4FEFDB8 = (__int64)sub_984030;
  sub_C53080(&qword_4FEFD00, "avail-extern-to-local", 21);
  qword_4FEFD30 = 83;
  LOBYTE(dword_4FEFD0C) = dword_4FEFD0C & 0x9F | 0x20;
  qword_4FEFD28 = (__int64)"Convert available_externally into locals, renaming them to avoid link-time clashes.";
  sub_C53130(&qword_4FEFD00);
  return __cxa_atexit(sub_984900, &qword_4FEFD00, &qword_4A427C0);
}
