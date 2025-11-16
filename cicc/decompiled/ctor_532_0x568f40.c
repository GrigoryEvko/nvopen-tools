// Function: ctor_532
// Address: 0x568f40
//
int ctor_532()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5014720 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5014770 = 0x100000000LL;
  word_5014730 = 0;
  dword_501472C &= 0x8000u;
  qword_5014738 = 0;
  qword_5014740 = 0;
  dword_5014728 = v0;
  qword_5014748 = 0;
  qword_5014750 = 0;
  qword_5014758 = 0;
  qword_5014760 = 0;
  qword_5014768 = (__int64)&unk_5014778;
  qword_5014780 = 0;
  qword_5014788 = (__int64)&unk_50147A0;
  qword_5014790 = 1;
  dword_5014798 = 0;
  byte_501479C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5014770;
  v3 = (unsigned int)qword_5014770 + 1LL;
  if ( v3 > HIDWORD(qword_5014770) )
  {
    sub_C8D5F0((char *)&unk_5014778 - 16, &unk_5014778, v3, 8);
    v2 = (unsigned int)qword_5014770;
  }
  *(_QWORD *)(qword_5014768 + 8 * v2) = v1;
  LODWORD(qword_5014770) = qword_5014770 + 1;
  qword_50147A8 = 0;
  qword_50147B0 = (__int64)&unk_49DA090;
  qword_50147B8 = 0;
  qword_5014720 = (__int64)&unk_49DBF90;
  qword_50147C0 = (__int64)&unk_49DC230;
  qword_50147E0 = (__int64)nullsub_58;
  qword_50147D8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_5014720, "normalize-gep", 13);
  LODWORD(qword_50147A8) = 1;
  BYTE4(qword_50147B8) = 1;
  LODWORD(qword_50147B8) = 1;
  qword_5014750 = 31;
  LOBYTE(dword_501472C) = dword_501472C & 0x9F | 0x20;
  qword_5014748 = (__int64)"Normalize 64-bit GEP subscripts";
  sub_C53130(&qword_5014720);
  __cxa_atexit(sub_B2B680, &qword_5014720, &qword_4A427C0);
  qword_5014640 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_50146BC = 1;
  qword_5014690 = 0x100000000LL;
  dword_501464C &= 0x8000u;
  qword_5014658 = 0;
  qword_5014660 = 0;
  qword_5014668 = 0;
  dword_5014648 = v4;
  word_5014650 = 0;
  qword_5014670 = 0;
  qword_5014678 = 0;
  qword_5014680 = 0;
  qword_5014688 = (__int64)&unk_5014698;
  qword_50146A0 = 0;
  qword_50146A8 = (__int64)&unk_50146C0;
  qword_50146B0 = 1;
  dword_50146B8 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5014690;
  v7 = (unsigned int)qword_5014690 + 1LL;
  if ( v7 > HIDWORD(qword_5014690) )
  {
    sub_C8D5F0((char *)&unk_5014698 - 16, &unk_5014698, v7, 8);
    v6 = (unsigned int)qword_5014690;
  }
  *(_QWORD *)(qword_5014688 + 8 * v6) = v5;
  LODWORD(qword_5014690) = qword_5014690 + 1;
  qword_50146C8 = 0;
  qword_50146D0 = (__int64)&unk_49D9748;
  qword_50146D8 = 0;
  qword_5014640 = (__int64)&unk_49DC090;
  qword_50146E0 = (__int64)&unk_49DC1D0;
  qword_5014700 = (__int64)nullsub_23;
  qword_50146F8 = (__int64)sub_984030;
  sub_C53080(&qword_5014640, "dump-normalize-gep", 18);
  LOBYTE(qword_50146C8) = 0;
  LOWORD(qword_50146D8) = 256;
  qword_5014670 = 57;
  LOBYTE(dword_501464C) = dword_501464C & 0x9F | 0x20;
  qword_5014668 = (__int64)"Dump Debug Message during Normalize 64-bit GEP subscripts";
  sub_C53130(&qword_5014640);
  return __cxa_atexit(sub_984900, &qword_5014640, &qword_4A427C0);
}
