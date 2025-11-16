// Function: ctor_668
// Address: 0x59fe00
//
int __fastcall ctor_668(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx

  qword_503BBE0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_503BC5C = 1;
  qword_503BC30 = 0x100000000LL;
  dword_503BBEC &= 0x8000u;
  qword_503BBF8 = 0;
  qword_503BC00 = 0;
  qword_503BC08 = 0;
  dword_503BBE8 = v4;
  word_503BBF0 = 0;
  qword_503BC10 = 0;
  qword_503BC18 = 0;
  qword_503BC20 = 0;
  qword_503BC28 = (__int64)&unk_503BC38;
  qword_503BC40 = 0;
  qword_503BC48 = (__int64)&unk_503BC60;
  qword_503BC50 = 1;
  dword_503BC58 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_503BC30;
  v7 = (unsigned int)qword_503BC30 + 1LL;
  if ( v7 > HIDWORD(qword_503BC30) )
  {
    sub_C8D5F0((char *)&unk_503BC38 - 16, &unk_503BC38, v7, 8);
    v6 = (unsigned int)qword_503BC30;
  }
  *(_QWORD *)(qword_503BC28 + 8 * v6) = v5;
  LODWORD(qword_503BC30) = qword_503BC30 + 1;
  qword_503BC68 = 0;
  qword_503BC70 = (__int64)&unk_49DA090;
  qword_503BC78 = 0;
  qword_503BBE0 = (__int64)&unk_49DBF90;
  qword_503BC80 = (__int64)&unk_49DC230;
  qword_503BCA0 = (__int64)nullsub_58;
  qword_503BC98 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_503BBE0, "imp-null-check-page-size", 24);
  qword_503BC10 = 36;
  qword_503BC08 = (__int64)"The page size of the target in bytes";
  LODWORD(qword_503BC68) = 4096;
  BYTE4(qword_503BC78) = 1;
  LODWORD(qword_503BC78) = 4096;
  LOBYTE(dword_503BBEC) = dword_503BBEC & 0x9F | 0x20;
  sub_C53130(&qword_503BBE0);
  __cxa_atexit(sub_B2B680, &qword_503BBE0, &qword_4A427C0);
  qword_503BB00 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_B2B680, &qword_503BBE0, v8, v9), 1u);
  dword_503BB0C &= 0x8000u;
  word_503BB10 = 0;
  qword_503BB50 = 0x100000000LL;
  qword_503BB18 = 0;
  qword_503BB20 = 0;
  qword_503BB28 = 0;
  dword_503BB08 = v10;
  qword_503BB30 = 0;
  qword_503BB38 = 0;
  qword_503BB40 = 0;
  qword_503BB48 = (__int64)&unk_503BB58;
  qword_503BB60 = 0;
  qword_503BB68 = (__int64)&unk_503BB80;
  qword_503BB70 = 1;
  dword_503BB78 = 0;
  byte_503BB7C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_503BB50;
  v13 = (unsigned int)qword_503BB50 + 1LL;
  if ( v13 > HIDWORD(qword_503BB50) )
  {
    sub_C8D5F0((char *)&unk_503BB58 - 16, &unk_503BB58, v13, 8);
    v12 = (unsigned int)qword_503BB50;
  }
  *(_QWORD *)(qword_503BB48 + 8 * v12) = v11;
  LODWORD(qword_503BB50) = qword_503BB50 + 1;
  qword_503BB88 = 0;
  qword_503BB90 = (__int64)&unk_49D9728;
  qword_503BB98 = 0;
  qword_503BB00 = (__int64)&unk_49DBF10;
  qword_503BBA0 = (__int64)&unk_49DC290;
  qword_503BBC0 = (__int64)nullsub_24;
  qword_503BBB8 = (__int64)sub_984050;
  sub_C53080(&qword_503BB00, "imp-null-max-insts-to-consider", 30);
  qword_503BB30 = 108;
  qword_503BB28 = (__int64)"The max number of instructions to consider hoisting loads over (the algorithm is quadratic over this number)";
  LODWORD(qword_503BB88) = 8;
  BYTE4(qword_503BB98) = 1;
  LODWORD(qword_503BB98) = 8;
  LOBYTE(dword_503BB0C) = dword_503BB0C & 0x9F | 0x20;
  sub_C53130(&qword_503BB00);
  return __cxa_atexit(sub_984970, &qword_503BB00, &qword_4A427C0);
}
