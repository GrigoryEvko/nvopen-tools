// Function: ctor_049
// Address: 0x4904f0
//
int ctor_049()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4F869C0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F869CC &= 0x8000u;
  word_4F869D0 = 0;
  qword_4F86A10 = 0x100000000LL;
  qword_4F869D8 = 0;
  qword_4F869E0 = 0;
  qword_4F869E8 = 0;
  dword_4F869C8 = v0;
  qword_4F869F0 = 0;
  qword_4F869F8 = 0;
  qword_4F86A00 = 0;
  qword_4F86A08 = (__int64)&unk_4F86A18;
  qword_4F86A20 = 0;
  qword_4F86A28 = (__int64)&unk_4F86A40;
  qword_4F86A30 = 1;
  dword_4F86A38 = 0;
  byte_4F86A3C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F86A10;
  v3 = (unsigned int)qword_4F86A10 + 1LL;
  if ( v3 > HIDWORD(qword_4F86A10) )
  {
    sub_C8D5F0((char *)&unk_4F86A18 - 16, &unk_4F86A18, v3, 8);
    v2 = (unsigned int)qword_4F86A10;
  }
  *(_QWORD *)(qword_4F86A08 + 8 * v2) = v1;
  LODWORD(qword_4F86A10) = qword_4F86A10 + 1;
  qword_4F86A48 = 0;
  qword_4F86A50 = (__int64)&unk_49D9728;
  qword_4F86A58 = 0;
  qword_4F869C0 = (__int64)&unk_49DBF10;
  qword_4F86A60 = (__int64)&unk_49DC290;
  qword_4F86A80 = (__int64)nullsub_24;
  qword_4F86A78 = (__int64)sub_984050;
  sub_C53080(&qword_4F869C0, "dom-tree-reachability-max-bbs-to-explore", 40);
  qword_4F869F0 = 54;
  LODWORD(qword_4F86A48) = 32;
  BYTE4(qword_4F86A58) = 1;
  LODWORD(qword_4F86A58) = 32;
  LOBYTE(dword_4F869CC) = dword_4F869CC & 0x9F | 0x20;
  qword_4F869E8 = (__int64)"Max number of BBs to explore for reachability analysis";
  sub_C53130(&qword_4F869C0);
  return __cxa_atexit(sub_984970, &qword_4F869C0, &qword_4A427C0);
}
