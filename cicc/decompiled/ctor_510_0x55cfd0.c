// Function: ctor_510
// Address: 0x55cfd0
//
int ctor_510()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_500BF20 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500BF70 = 0x100000000LL;
  word_500BF30 = 0;
  dword_500BF2C &= 0x8000u;
  qword_500BF38 = 0;
  qword_500BF40 = 0;
  dword_500BF28 = v0;
  qword_500BF48 = 0;
  qword_500BF50 = 0;
  qword_500BF58 = 0;
  qword_500BF60 = 0;
  qword_500BF68 = (__int64)&unk_500BF78;
  qword_500BF80 = 0;
  qword_500BF88 = (__int64)&unk_500BFA0;
  qword_500BF90 = 1;
  dword_500BF98 = 0;
  byte_500BF9C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_500BF70;
  v3 = (unsigned int)qword_500BF70 + 1LL;
  if ( v3 > HIDWORD(qword_500BF70) )
  {
    sub_C8D5F0((char *)&unk_500BF78 - 16, &unk_500BF78, v3, 8);
    v2 = (unsigned int)qword_500BF70;
  }
  *(_QWORD *)(qword_500BF68 + 8 * v2) = v1;
  LODWORD(qword_500BF70) = qword_500BF70 + 1;
  qword_500BFA8 = 0;
  qword_500BFB0 = (__int64)&unk_49D9748;
  qword_500BFB8 = 0;
  qword_500BF20 = (__int64)&unk_49DC090;
  qword_500BFC0 = (__int64)&unk_49DC1D0;
  qword_500BFE0 = (__int64)nullsub_23;
  qword_500BFD8 = (__int64)sub_984030;
  sub_C53080(&qword_500BF20, "sccp-use-bfs", 12);
  LOBYTE(qword_500BFA8) = 1;
  qword_500BF50 = 63;
  LOBYTE(dword_500BF2C) = dword_500BF2C & 0x9F | 0x20;
  LOWORD(qword_500BFB8) = 257;
  qword_500BF48 = (__int64)"Use breadth-first traversal for worklist instead of depth-first";
  sub_C53130(&qword_500BF20);
  __cxa_atexit(sub_984900, &qword_500BF20, &qword_4A427C0);
  qword_500BE40 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_500BEBC = 1;
  qword_500BE90 = 0x100000000LL;
  dword_500BE4C &= 0x8000u;
  qword_500BE58 = 0;
  qword_500BE60 = 0;
  qword_500BE68 = 0;
  dword_500BE48 = v4;
  word_500BE50 = 0;
  qword_500BE70 = 0;
  qword_500BE78 = 0;
  qword_500BE80 = 0;
  qword_500BE88 = (__int64)&unk_500BE98;
  qword_500BEA0 = 0;
  qword_500BEA8 = (__int64)&unk_500BEC0;
  qword_500BEB0 = 1;
  dword_500BEB8 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_500BE90;
  v7 = (unsigned int)qword_500BE90 + 1LL;
  if ( v7 > HIDWORD(qword_500BE90) )
  {
    sub_C8D5F0((char *)&unk_500BE98 - 16, &unk_500BE98, v7, 8);
    v6 = (unsigned int)qword_500BE90;
  }
  *(_QWORD *)(qword_500BE88 + 8 * v6) = v5;
  LODWORD(qword_500BE90) = qword_500BE90 + 1;
  qword_500BEC8 = 0;
  qword_500BED0 = (__int64)&unk_49DA090;
  qword_500BED8 = 0;
  qword_500BE40 = (__int64)&unk_49DBF90;
  qword_500BEE0 = (__int64)&unk_49DC230;
  qword_500BF00 = (__int64)nullsub_58;
  qword_500BEF8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_500BE40, "sccp-max-range-ext", 18);
  LODWORD(qword_500BEC8) = 10;
  BYTE4(qword_500BED8) = 1;
  LODWORD(qword_500BED8) = 10;
  qword_500BE70 = 53;
  LOBYTE(dword_500BE4C) = dword_500BE4C & 0x9F | 0x20;
  qword_500BE68 = (__int64)"Maximum number of range extensions requiring widening";
  sub_C53130(&qword_500BE40);
  return __cxa_atexit(sub_B2B680, &qword_500BE40, &qword_4A427C0);
}
