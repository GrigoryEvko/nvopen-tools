// Function: ctor_084
// Address: 0x49fd60
//
int ctor_084()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4F8FB00 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F8FB7C = 1;
  qword_4F8FB50 = 0x100000000LL;
  dword_4F8FB0C &= 0x8000u;
  qword_4F8FB18 = 0;
  qword_4F8FB20 = 0;
  qword_4F8FB28 = 0;
  dword_4F8FB08 = v0;
  word_4F8FB10 = 0;
  qword_4F8FB30 = 0;
  qword_4F8FB38 = 0;
  qword_4F8FB40 = 0;
  qword_4F8FB48 = (__int64)&unk_4F8FB58;
  qword_4F8FB60 = 0;
  qword_4F8FB68 = (__int64)&unk_4F8FB80;
  qword_4F8FB70 = 1;
  dword_4F8FB78 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F8FB50;
  v3 = (unsigned int)qword_4F8FB50 + 1LL;
  if ( v3 > HIDWORD(qword_4F8FB50) )
  {
    sub_C8D5F0((char *)&unk_4F8FB58 - 16, &unk_4F8FB58, v3, 8);
    v2 = (unsigned int)qword_4F8FB50;
  }
  *(_QWORD *)(qword_4F8FB48 + 8 * v2) = v1;
  LODWORD(qword_4F8FB50) = qword_4F8FB50 + 1;
  qword_4F8FB88 = 0;
  qword_4F8FB90 = (__int64)&unk_49D9748;
  qword_4F8FB98 = 0;
  qword_4F8FB00 = (__int64)&unk_49DC090;
  qword_4F8FBA0 = (__int64)&unk_49DC1D0;
  qword_4F8FBC0 = (__int64)nullsub_23;
  qword_4F8FBB8 = (__int64)sub_984030;
  sub_C53080(&qword_4F8FB00, "gvn-add-phi-translation", 23);
  LOBYTE(qword_4F8FB88) = 0;
  LOWORD(qword_4F8FB98) = 256;
  qword_4F8FB30 = 42;
  LOBYTE(dword_4F8FB0C) = dword_4F8FB0C & 0x9F | 0x20;
  qword_4F8FB28 = (__int64)"Enable phi-translation of add instructions";
  sub_C53130(&qword_4F8FB00);
  return __cxa_atexit(sub_984900, &qword_4F8FB00, &qword_4A427C0);
}
