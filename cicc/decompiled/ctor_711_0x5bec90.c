// Function: ctor_711
// Address: 0x5bec90
//
int __fastcall ctor_711(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5051440 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  dword_505144C &= 0x8000u;
  word_5051450 = 0;
  qword_5051490 = 0x100000000LL;
  qword_5051458 = 0;
  qword_5051460 = 0;
  qword_5051468 = 0;
  dword_5051448 = v4;
  qword_5051470 = 0;
  qword_5051478 = 0;
  qword_5051480 = 0;
  qword_5051488 = (__int64)&unk_5051498;
  qword_50514A0 = 0;
  qword_50514A8 = (__int64)&unk_50514C0;
  qword_50514B0 = 1;
  dword_50514B8 = 0;
  byte_50514BC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5051490;
  v7 = (unsigned int)qword_5051490 + 1LL;
  if ( v7 > HIDWORD(qword_5051490) )
  {
    sub_C8D5F0((char *)&unk_5051498 - 16, &unk_5051498, v7, 8);
    v6 = (unsigned int)qword_5051490;
  }
  *(_QWORD *)(qword_5051488 + 8 * v6) = v5;
  LODWORD(qword_5051490) = qword_5051490 + 1;
  qword_50514C8 = 0;
  qword_50514D0 = (__int64)&unk_49D9728;
  qword_50514D8 = 0;
  qword_5051440 = (__int64)&unk_49DBF10;
  qword_50514E0 = (__int64)&unk_49DC290;
  qword_5051500 = (__int64)nullsub_24;
  qword_50514F8 = (__int64)sub_984050;
  sub_C53080(&qword_5051440, "dfa-instr-limit", 15);
  LODWORD(qword_50514C8) = 0;
  BYTE4(qword_50514D8) = 1;
  LODWORD(qword_50514D8) = 0;
  qword_5051470 = 50;
  LOBYTE(dword_505144C) = dword_505144C & 0x9F | 0x20;
  qword_5051468 = (__int64)"If present, stops packetizing after N instructions";
  sub_C53130(&qword_5051440);
  return __cxa_atexit(sub_984970, &qword_5051440, &qword_4A427C0);
}
