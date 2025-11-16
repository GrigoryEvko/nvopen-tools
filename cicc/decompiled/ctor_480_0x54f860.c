// Function: ctor_480
// Address: 0x54f860
//
int ctor_480()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5004CA0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5004CF0 = 0x100000000LL;
  dword_5004CAC &= 0x8000u;
  word_5004CB0 = 0;
  qword_5004CB8 = 0;
  qword_5004CC0 = 0;
  dword_5004CA8 = v0;
  qword_5004CC8 = 0;
  qword_5004CD0 = 0;
  qword_5004CD8 = 0;
  qword_5004CE0 = 0;
  qword_5004CE8 = (__int64)&unk_5004CF8;
  qword_5004D00 = 0;
  qword_5004D08 = (__int64)&unk_5004D20;
  qword_5004D10 = 1;
  dword_5004D18 = 0;
  byte_5004D1C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5004CF0;
  v3 = (unsigned int)qword_5004CF0 + 1LL;
  if ( v3 > HIDWORD(qword_5004CF0) )
  {
    sub_C8D5F0((char *)&unk_5004CF8 - 16, &unk_5004CF8, v3, 8);
    v2 = (unsigned int)qword_5004CF0;
  }
  *(_QWORD *)(qword_5004CE8 + 8 * v2) = v1;
  qword_5004D30 = (__int64)&unk_49D9748;
  LODWORD(qword_5004CF0) = qword_5004CF0 + 1;
  qword_5004D28 = 0;
  qword_5004CA0 = (__int64)&unk_49DC090;
  qword_5004D40 = (__int64)&unk_49DC1D0;
  qword_5004D38 = 0;
  qword_5004D60 = (__int64)nullsub_23;
  qword_5004D58 = (__int64)sub_984030;
  sub_C53080(&qword_5004CA0, "special-reassociate-for-threadid", 32);
  qword_5004CC8 = (__int64)"Reassociate - do not move back expressions that use threadid";
  LOWORD(qword_5004D38) = 257;
  LOBYTE(qword_5004D28) = 1;
  qword_5004CD0 = 60;
  LOBYTE(dword_5004CAC) = dword_5004CAC & 0x9F | 0x20;
  sub_C53130(&qword_5004CA0);
  __cxa_atexit(sub_984900, &qword_5004CA0, &qword_4A427C0);
  qword_5004BC0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5004C10 = 0x100000000LL;
  word_5004BD0 = 0;
  dword_5004BCC &= 0x8000u;
  qword_5004BD8 = 0;
  qword_5004BE0 = 0;
  dword_5004BC8 = v4;
  qword_5004BE8 = 0;
  qword_5004BF0 = 0;
  qword_5004BF8 = 0;
  qword_5004C00 = 0;
  qword_5004C08 = (__int64)&unk_5004C18;
  qword_5004C20 = 0;
  qword_5004C28 = (__int64)&unk_5004C40;
  qword_5004C30 = 1;
  dword_5004C38 = 0;
  byte_5004C3C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5004C10;
  v7 = (unsigned int)qword_5004C10 + 1LL;
  if ( v7 > HIDWORD(qword_5004C10) )
  {
    sub_C8D5F0((char *)&unk_5004C18 - 16, &unk_5004C18, v7, 8);
    v6 = (unsigned int)qword_5004C10;
  }
  *(_QWORD *)(qword_5004C08 + 8 * v6) = v5;
  qword_5004C50 = (__int64)&unk_49D9748;
  LODWORD(qword_5004C10) = qword_5004C10 + 1;
  qword_5004C48 = 0;
  qword_5004BC0 = (__int64)&unk_49DC090;
  qword_5004C60 = (__int64)&unk_49DC1D0;
  qword_5004C58 = 0;
  qword_5004C80 = (__int64)nullsub_23;
  qword_5004C78 = (__int64)sub_984030;
  sub_C53080(&qword_5004BC0, "reassociate-use-cse-local", 25);
  qword_5004BF0 = 77;
  qword_5004BE8 = (__int64)"Only reorder expressions within a basic block when exposing CSE opportunities";
  LOWORD(qword_5004C58) = 257;
  LOBYTE(qword_5004C48) = 1;
  LOBYTE(dword_5004BCC) = dword_5004BCC & 0x9F | 0x20;
  sub_C53130(&qword_5004BC0);
  return __cxa_atexit(sub_984900, &qword_5004BC0, &qword_4A427C0);
}
