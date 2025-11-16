// Function: ctor_578
// Address: 0x577ac0
//
int ctor_578()
{
  int v0; // edx
  __int64 v1; // r8
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r14
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v9; // [rsp+0h] [rbp-60h]
  char v10; // [rsp+13h] [rbp-4Dh] BYREF
  int v11; // [rsp+14h] [rbp-4Ch] BYREF
  char *v12; // [rsp+18h] [rbp-48h] BYREF
  const char *v13; // [rsp+20h] [rbp-40h] BYREF
  __int64 v14; // [rsp+28h] [rbp-38h]

  v13 = "Disable critical edge splitting during PHI elimination";
  v12 = &v10;
  v14 = 54;
  v11 = 1;
  v10 = 0;
  sub_2ABC310(&unk_5022EE0, "disable-phi-elim-edge-splitting", &v12, &v11, &v13);
  __cxa_atexit(sub_984900, &unk_5022EE0, &qword_4A427C0);
  qword_5022E00 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5022E50 = 0x100000000LL;
  dword_5022E0C &= 0x8000u;
  word_5022E10 = 0;
  qword_5022E18 = 0;
  qword_5022E20 = 0;
  dword_5022E08 = v0;
  qword_5022E28 = 0;
  qword_5022E30 = 0;
  qword_5022E38 = 0;
  qword_5022E40 = 0;
  qword_5022E48 = (__int64)&unk_5022E58;
  qword_5022E60 = 0;
  qword_5022E68 = (__int64)&unk_5022E80;
  qword_5022E70 = 1;
  dword_5022E78 = 0;
  byte_5022E7C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5022E50;
  v3 = (unsigned int)qword_5022E50 + 1LL;
  if ( v3 > HIDWORD(qword_5022E50) )
  {
    v9 = v1;
    sub_C8D5F0((char *)&unk_5022E58 - 16, &unk_5022E58, v3, 8);
    v2 = (unsigned int)qword_5022E50;
    v1 = v9;
  }
  *(_QWORD *)(qword_5022E48 + 8 * v2) = v1;
  qword_5022E90 = (__int64)&unk_49D9748;
  LODWORD(qword_5022E50) = qword_5022E50 + 1;
  qword_5022E88 = 0;
  qword_5022E00 = (__int64)&unk_49DC090;
  qword_5022E98 = 0;
  qword_5022EA0 = (__int64)&unk_49DC1D0;
  qword_5022EC0 = (__int64)nullsub_23;
  qword_5022EB8 = (__int64)sub_984030;
  sub_C53080(&qword_5022E00, "phi-elim-split-all-critical-edges", 33);
  LOWORD(qword_5022E98) = 256;
  LOBYTE(qword_5022E88) = 0;
  qword_5022E30 = 47;
  LOBYTE(dword_5022E0C) = dword_5022E0C & 0x9F | 0x20;
  qword_5022E28 = (__int64)"Split all critical edges during PHI elimination";
  sub_C53130(&qword_5022E00);
  __cxa_atexit(sub_984900, &qword_5022E00, &qword_4A427C0);
  v14 = 59;
  v13 = "Do not use an early exit if isLiveOutPastPHIs returns true.";
  v11 = 1;
  v12 = &v10;
  v10 = 0;
  sub_2ABC310(&unk_5022D20, "no-phi-elim-live-out-early-exit", &v12, &v11, &v13);
  __cxa_atexit(sub_984900, &unk_5022D20, &qword_4A427C0);
  qword_5022C40 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5022C90 = 0x100000000LL;
  word_5022C50 = 0;
  dword_5022C4C &= 0x8000u;
  qword_5022C58 = 0;
  qword_5022C60 = 0;
  dword_5022C48 = v4;
  qword_5022C68 = 0;
  qword_5022C70 = 0;
  qword_5022C78 = 0;
  qword_5022C80 = 0;
  qword_5022C88 = (__int64)&unk_5022C98;
  qword_5022CA0 = 0;
  qword_5022CA8 = (__int64)&unk_5022CC0;
  qword_5022CB0 = 1;
  dword_5022CB8 = 0;
  byte_5022CBC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5022C90;
  v7 = (unsigned int)qword_5022C90 + 1LL;
  if ( v7 > HIDWORD(qword_5022C90) )
  {
    sub_C8D5F0((char *)&unk_5022C98 - 16, &unk_5022C98, v7, 8);
    v6 = (unsigned int)qword_5022C90;
  }
  *(_QWORD *)(qword_5022C88 + 8 * v6) = v5;
  qword_5022CD0 = (__int64)&unk_49D9748;
  LODWORD(qword_5022C90) = qword_5022C90 + 1;
  qword_5022CC8 = 0;
  qword_5022C40 = (__int64)&unk_49DC090;
  qword_5022CD8 = 0;
  qword_5022CE0 = (__int64)&unk_49DC1D0;
  qword_5022D00 = (__int64)nullsub_23;
  qword_5022CF8 = (__int64)sub_984030;
  sub_C53080(&qword_5022C40, "donot-insert-dup-copies", 23);
  LOBYTE(qword_5022CC8) = 1;
  LOWORD(qword_5022CD8) = 257;
  qword_5022C70 = 101;
  LOBYTE(dword_5022C4C) = dword_5022C4C & 0x9F | 0x20;
  qword_5022C68 = (__int64)"Do not insert duplicate copies to a predecessor bb, if the copy is already dominated by another copy.";
  sub_C53130(&qword_5022C40);
  return __cxa_atexit(sub_984900, &qword_5022C40, &qword_4A427C0);
}
