// Function: ctor_415
// Address: 0x530a30
//
int ctor_415()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4FEFC20 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FEFC2C &= 0x8000u;
  word_4FEFC30 = 0;
  qword_4FEFC70 = 0x100000000LL;
  qword_4FEFC38 = 0;
  qword_4FEFC40 = 0;
  qword_4FEFC48 = 0;
  dword_4FEFC28 = v0;
  qword_4FEFC50 = 0;
  qword_4FEFC58 = 0;
  qword_4FEFC60 = 0;
  qword_4FEFC68 = (__int64)&unk_4FEFC78;
  qword_4FEFC80 = 0;
  qword_4FEFC88 = (__int64)&unk_4FEFCA0;
  qword_4FEFC90 = 1;
  dword_4FEFC98 = 0;
  byte_4FEFC9C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FEFC70;
  v3 = (unsigned int)qword_4FEFC70 + 1LL;
  if ( v3 > HIDWORD(qword_4FEFC70) )
  {
    sub_C8D5F0((char *)&unk_4FEFC78 - 16, &unk_4FEFC78, v3, 8);
    v2 = (unsigned int)qword_4FEFC70;
  }
  *(_QWORD *)(qword_4FEFC68 + 8 * v2) = v1;
  LODWORD(qword_4FEFC70) = qword_4FEFC70 + 1;
  qword_4FEFCA8 = 0;
  qword_4FEFCB0 = (__int64)&unk_49D9728;
  qword_4FEFCB8 = 0;
  qword_4FEFC20 = (__int64)&unk_49DBF10;
  qword_4FEFCC0 = (__int64)&unk_49DC290;
  qword_4FEFCE0 = (__int64)nullsub_24;
  qword_4FEFCD8 = (__int64)sub_984050;
  sub_C53080(&qword_4FEFC20, "cvp-max-functions-per-value", 27);
  LODWORD(qword_4FEFCA8) = 4;
  BYTE4(qword_4FEFCB8) = 1;
  LODWORD(qword_4FEFCB8) = 4;
  qword_4FEFC50 = 58;
  LOBYTE(dword_4FEFC2C) = dword_4FEFC2C & 0x9F | 0x20;
  qword_4FEFC48 = (__int64)"The maximum number of functions to track per lattice value";
  sub_C53130(&qword_4FEFC20);
  return __cxa_atexit(sub_984970, &qword_4FEFC20, &qword_4A427C0);
}
