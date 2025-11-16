// Function: ctor_550
// Address: 0x56df70
//
int ctor_550()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // r15
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-38h]

  qword_501D5A0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_501D5F0 = 0x100000000LL;
  word_501D5B0 = 0;
  dword_501D5AC &= 0x8000u;
  qword_501D5B8 = 0;
  qword_501D5C0 = 0;
  dword_501D5A8 = v0;
  qword_501D5C8 = 0;
  qword_501D5D0 = 0;
  qword_501D5D8 = 0;
  qword_501D5E0 = 0;
  qword_501D5E8 = (__int64)&unk_501D5F8;
  qword_501D600 = 0;
  qword_501D608 = (__int64)&unk_501D620;
  qword_501D610 = 1;
  dword_501D618 = 0;
  byte_501D61C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_501D5F0;
  v3 = (unsigned int)qword_501D5F0 + 1LL;
  if ( v3 > HIDWORD(qword_501D5F0) )
  {
    sub_C8D5F0((char *)&unk_501D5F8 - 16, &unk_501D5F8, v3, 8);
    v2 = (unsigned int)qword_501D5F0;
  }
  *(_QWORD *)(qword_501D5E8 + 8 * v2) = v1;
  qword_501D630 = (__int64)&unk_49D9728;
  qword_501D5A0 = (__int64)&unk_49DBF10;
  qword_501D640 = (__int64)&unk_49DC290;
  LODWORD(qword_501D5F0) = qword_501D5F0 + 1;
  qword_501D660 = (__int64)nullsub_24;
  qword_501D628 = 0;
  qword_501D658 = (__int64)sub_984050;
  qword_501D638 = 0;
  sub_C53080(&qword_501D5A0, "memcmp-num-loads-per-block", 26);
  LODWORD(qword_501D628) = 1;
  BYTE4(qword_501D638) = 1;
  LODWORD(qword_501D638) = 1;
  qword_501D5D0 = 108;
  LOBYTE(dword_501D5AC) = dword_501D5AC & 0x9F | 0x20;
  qword_501D5C8 = (__int64)"The number of loads per basic block for inline expansion of memcmp that is only being compared against zero.";
  sub_C53130(&qword_501D5A0);
  __cxa_atexit(sub_984970, &qword_501D5A0, &qword_4A427C0);
  qword_501D4C0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501D53C = 1;
  qword_501D510 = 0x100000000LL;
  dword_501D4CC &= 0x8000u;
  qword_501D508 = (__int64)&unk_501D518;
  qword_501D4D8 = 0;
  qword_501D4E0 = 0;
  dword_501D4C8 = v4;
  word_501D4D0 = 0;
  qword_501D4E8 = 0;
  qword_501D4F0 = 0;
  qword_501D4F8 = 0;
  qword_501D500 = 0;
  qword_501D520 = 0;
  qword_501D528 = (__int64)&unk_501D540;
  qword_501D530 = 1;
  dword_501D538 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_501D510;
  if ( (unsigned __int64)(unsigned int)qword_501D510 + 1 > HIDWORD(qword_501D510) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_501D518 - 16, &unk_501D518, (unsigned int)qword_501D510 + 1LL, 8);
    v6 = (unsigned int)qword_501D510;
    v5 = v12;
  }
  *(_QWORD *)(qword_501D508 + 8 * v6) = v5;
  qword_501D550 = (__int64)&unk_49D9728;
  qword_501D4C0 = (__int64)&unk_49DBF10;
  qword_501D560 = (__int64)&unk_49DC290;
  LODWORD(qword_501D510) = qword_501D510 + 1;
  qword_501D580 = (__int64)nullsub_24;
  qword_501D548 = 0;
  qword_501D578 = (__int64)sub_984050;
  qword_501D558 = 0;
  sub_C53080(&qword_501D4C0, "max-loads-per-memcmp", 20);
  qword_501D4F0 = 51;
  LOBYTE(dword_501D4CC) = dword_501D4CC & 0x9F | 0x20;
  qword_501D4E8 = (__int64)"Set maximum number of loads used in expanded memcmp";
  sub_C53130(&qword_501D4C0);
  __cxa_atexit(sub_984970, &qword_501D4C0, &qword_4A427C0);
  qword_501D3E0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_501D3EC &= 0x8000u;
  word_501D3F0 = 0;
  qword_501D430 = 0x100000000LL;
  qword_501D428 = (__int64)&unk_501D438;
  qword_501D3F8 = 0;
  qword_501D400 = 0;
  dword_501D3E8 = v7;
  qword_501D408 = 0;
  qword_501D410 = 0;
  qword_501D418 = 0;
  qword_501D420 = 0;
  qword_501D440 = 0;
  qword_501D448 = (__int64)&unk_501D460;
  qword_501D450 = 1;
  dword_501D458 = 0;
  byte_501D45C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_501D430;
  v10 = (unsigned int)qword_501D430 + 1LL;
  if ( v10 > HIDWORD(qword_501D430) )
  {
    sub_C8D5F0((char *)&unk_501D438 - 16, &unk_501D438, v10, 8);
    v9 = (unsigned int)qword_501D430;
  }
  *(_QWORD *)(qword_501D428 + 8 * v9) = v8;
  qword_501D470 = (__int64)&unk_49D9728;
  qword_501D3E0 = (__int64)&unk_49DBF10;
  qword_501D480 = (__int64)&unk_49DC290;
  LODWORD(qword_501D430) = qword_501D430 + 1;
  qword_501D4A0 = (__int64)nullsub_24;
  qword_501D468 = 0;
  qword_501D498 = (__int64)sub_984050;
  qword_501D478 = 0;
  sub_C53080(&qword_501D3E0, "max-loads-per-memcmp-opt-size", 29);
  qword_501D410 = 62;
  LOBYTE(dword_501D3EC) = dword_501D3EC & 0x9F | 0x20;
  qword_501D408 = (__int64)"Set maximum number of loads used in expanded memcmp for -Os/Oz";
  sub_C53130(&qword_501D3E0);
  return __cxa_atexit(sub_984970, &qword_501D3E0, &qword_4A427C0);
}
