// Function: ctor_662
// Address: 0x59d6c0
//
int __fastcall ctor_662(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // edx
  __int64 v10; // r12
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  int v14; // [rsp+4h] [rbp-3Ch] BYREF
  const char *v15; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v16[6]; // [rsp+10h] [rbp-30h] BYREF

  v15 = ".text.split.";
  v16[0] = "The text prefix to use for cold basic block clusters";
  v14 = 1;
  qword_503A720 = &unk_49DC150;
  v16[1] = 52;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  *(_DWORD *)&word_503A72C = word_503A72C & 0x8000;
  qword_503A768[1] = 0x100000000LL;
  unk_503A728 = v4;
  unk_503A730 = 0;
  unk_503A738 = 0;
  unk_503A740 = 0;
  unk_503A748 = 0;
  unk_503A750 = 0;
  unk_503A758 = 0;
  unk_503A760 = 0;
  qword_503A768[0] = &qword_503A768[2];
  qword_503A768[3] = 0;
  qword_503A768[4] = &qword_503A768[7];
  qword_503A768[5] = 1;
  LODWORD(qword_503A768[6]) = 0;
  BYTE4(qword_503A768[6]) = 1;
  v5 = sub_C57470();
  v6 = LODWORD(qword_503A768[1]);
  if ( (unsigned __int64)LODWORD(qword_503A768[1]) + 1 > HIDWORD(qword_503A768[1]) )
  {
    sub_C8D5F0(qword_503A768, &qword_503A768[2], LODWORD(qword_503A768[1]) + 1LL, 8);
    v6 = LODWORD(qword_503A768[1]);
  }
  *(_QWORD *)(qword_503A768[0] + 8 * v6) = v5;
  qword_503A768[8] = &qword_503A768[10];
  qword_503A768[13] = &qword_503A768[15];
  ++LODWORD(qword_503A768[1]);
  qword_503A768[9] = 0;
  qword_503A768[12] = &unk_49DC130;
  LOBYTE(qword_503A768[10]) = 0;
  qword_503A768[14] = 0;
  qword_503A720 = &unk_49DC010;
  LOBYTE(qword_503A768[15]) = 0;
  LOBYTE(qword_503A768[17]) = 0;
  qword_503A768[18] = &unk_49DC350;
  qword_503A768[22] = nullsub_92;
  qword_503A768[21] = sub_BC4D70;
  sub_34BC9B0(&qword_503A720, "bbsections-cold-text-prefix", v16, &v15, &v14);
  sub_C53130(&qword_503A720);
  __cxa_atexit(sub_BC5A40, &qword_503A720, &qword_4A427C0);
  qword_503A640 = (__int64)&unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_BC5A40, &qword_503A720, v7, v8), 1u);
  byte_503A6BC = 1;
  qword_503A690 = 0x100000000LL;
  dword_503A64C &= 0x8000u;
  qword_503A658 = 0;
  qword_503A660 = 0;
  qword_503A668 = 0;
  dword_503A648 = v9;
  word_503A650 = 0;
  qword_503A670 = 0;
  qword_503A678 = 0;
  qword_503A680 = 0;
  qword_503A688 = (__int64)&unk_503A698;
  qword_503A6A0 = 0;
  qword_503A6A8 = (__int64)&unk_503A6C0;
  qword_503A6B0 = 1;
  dword_503A6B8 = 0;
  v10 = sub_C57470();
  v11 = (unsigned int)qword_503A690;
  v12 = (unsigned int)qword_503A690 + 1LL;
  if ( v12 > HIDWORD(qword_503A690) )
  {
    sub_C8D5F0((char *)&unk_503A698 - 16, &unk_503A698, v12, 8);
    v11 = (unsigned int)qword_503A690;
  }
  *(_QWORD *)(qword_503A688 + 8 * v11) = v10;
  LODWORD(qword_503A690) = qword_503A690 + 1;
  qword_503A6C8 = 0;
  qword_503A6D0 = (__int64)&unk_49D9748;
  qword_503A6D8 = 0;
  qword_503A640 = (__int64)&unk_49DC090;
  qword_503A6E0 = (__int64)&unk_49DC1D0;
  qword_503A700 = (__int64)nullsub_23;
  qword_503A6F8 = (__int64)sub_984030;
  sub_C53080(&qword_503A640, "bbsections-detect-source-drift", 30);
  qword_503A670 = 76;
  qword_503A668 = (__int64)"This checks if there is a fdo instr. profile hash mismatch for this function";
  LOWORD(qword_503A6D8) = 257;
  LOBYTE(qword_503A6C8) = 1;
  LOBYTE(dword_503A64C) = dword_503A64C & 0x9F | 0x20;
  sub_C53130(&qword_503A640);
  return __cxa_atexit(sub_984900, &qword_503A640, &qword_4A427C0);
}
