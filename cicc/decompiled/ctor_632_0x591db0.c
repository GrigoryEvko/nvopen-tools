// Function: ctor_632
// Address: 0x591db0
//
int __fastcall ctor_632(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  int v15; // edx
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v19; // [rsp+8h] [rbp-58h]
  _QWORD v20[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v21[8]; // [rsp+20h] [rbp-40h] BYREF

  qword_5032140 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5032190 = 0x100000000LL;
  dword_503214C &= 0x8000u;
  word_5032150 = 0;
  qword_5032158 = 0;
  qword_5032160 = 0;
  dword_5032148 = v4;
  qword_5032168 = 0;
  qword_5032170 = 0;
  qword_5032178 = 0;
  qword_5032180 = 0;
  qword_5032188 = (__int64)&unk_5032198;
  qword_50321A0 = 0;
  qword_50321A8 = (__int64)&unk_50321C0;
  qword_50321B0 = 1;
  dword_50321B8 = 0;
  byte_50321BC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5032190;
  v7 = (unsigned int)qword_5032190 + 1LL;
  if ( v7 > HIDWORD(qword_5032190) )
  {
    sub_C8D5F0((char *)&unk_5032198 - 16, &unk_5032198, v7, 8);
    v6 = (unsigned int)qword_5032190;
  }
  *(_QWORD *)(qword_5032188 + 8 * v6) = v5;
  qword_50321D0 = (__int64)&unk_49D9748;
  LODWORD(qword_5032190) = qword_5032190 + 1;
  qword_50321C8 = 0;
  qword_5032140 = (__int64)&unk_49DC090;
  qword_50321D8 = 0;
  qword_50321E0 = (__int64)&unk_49DC1D0;
  qword_5032200 = (__int64)nullsub_23;
  qword_50321F8 = (__int64)sub_984030;
  sub_C53080(&qword_5032140, "codegen-data-generate", 21);
  LOWORD(qword_50321D8) = 256;
  LOBYTE(qword_50321C8) = 0;
  qword_5032170 = 38;
  LOBYTE(dword_503214C) = dword_503214C & 0x9F | 0x20;
  qword_5032168 = (__int64)"Emit CodeGen Data into custom sections";
  sub_C53130(&qword_5032140);
  __cxa_atexit(sub_984900, &qword_5032140, &qword_4A427C0);
  qword_5032040 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5032140, v8, v9), 1u);
  qword_5032090 = 0x100000000LL;
  dword_503204C &= 0x8000u;
  word_5032050 = 0;
  qword_5032058 = 0;
  qword_5032060 = 0;
  dword_5032048 = v10;
  qword_5032068 = 0;
  qword_5032070 = 0;
  qword_5032078 = 0;
  qword_5032080 = 0;
  qword_5032088 = (__int64)&unk_5032098;
  qword_50320A0 = 0;
  qword_50320A8 = (__int64)&unk_50320C0;
  qword_50320B0 = 1;
  dword_50320B8 = 0;
  byte_50320BC = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5032090;
  if ( (unsigned __int64)(unsigned int)qword_5032090 + 1 > HIDWORD(qword_5032090) )
  {
    v19 = v11;
    sub_C8D5F0((char *)&unk_5032098 - 16, &unk_5032098, (unsigned int)qword_5032090 + 1LL, 8);
    v12 = (unsigned int)qword_5032090;
    v11 = v19;
  }
  *(_QWORD *)(qword_5032088 + 8 * v12) = v11;
  qword_50320C8 = (__int64)&byte_50320D8;
  qword_50320F0 = (__int64)&byte_5032100;
  LODWORD(qword_5032090) = qword_5032090 + 1;
  qword_50320D0 = 0;
  qword_50320E8 = (__int64)&unk_49DC130;
  byte_50320D8 = 0;
  byte_5032100 = 0;
  qword_5032040 = (__int64)&unk_49DC010;
  qword_50320F8 = 0;
  byte_5032110 = 0;
  qword_5032118 = (__int64)&unk_49DC350;
  qword_5032138 = (__int64)nullsub_92;
  qword_5032130 = (__int64)sub_BC4D70;
  sub_C53080(&qword_5032040, "codegen-data-use-path", 21);
  v20[0] = v21;
  sub_31112A0(v20, byte_3F871B3);
  sub_2240AE0(&qword_50320C8, v20);
  byte_5032110 = 1;
  sub_2240AE0(&qword_50320F0, v20);
  if ( (_QWORD *)v20[0] != v21 )
    j_j___libc_free_0(v20[0], v21[0] + 1LL);
  qword_5032070 = 39;
  LOBYTE(dword_503204C) = dword_503204C & 0x9F | 0x20;
  qword_5032068 = (__int64)"File path to where .cgdata file is read";
  sub_C53130(&qword_5032040);
  __cxa_atexit(sub_BC5A40, &qword_5032040, &qword_4A427C0);
  qword_5031F60 = &unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_BC5A40, &qword_5032040, v13, v14), 1u);
  *(_DWORD *)&word_5031F6C = word_5031F6C & 0x8000;
  unk_5031F68 = v15;
  qword_5031FA8[1] = 0x100000000LL;
  unk_5031F70 = 0;
  unk_5031F78 = 0;
  unk_5031F80 = 0;
  unk_5031F88 = 0;
  unk_5031F90 = 0;
  unk_5031F98 = 0;
  unk_5031FA0 = 0;
  qword_5031FA8[0] = &qword_5031FA8[2];
  qword_5031FA8[3] = 0;
  qword_5031FA8[4] = &qword_5031FA8[7];
  qword_5031FA8[5] = 1;
  LODWORD(qword_5031FA8[6]) = 0;
  BYTE4(qword_5031FA8[6]) = 1;
  v16 = sub_C57470();
  v17 = LODWORD(qword_5031FA8[1]);
  if ( (unsigned __int64)LODWORD(qword_5031FA8[1]) + 1 > HIDWORD(qword_5031FA8[1]) )
  {
    sub_C8D5F0(qword_5031FA8, &qword_5031FA8[2], LODWORD(qword_5031FA8[1]) + 1LL, 8);
    v17 = LODWORD(qword_5031FA8[1]);
  }
  *(_QWORD *)(qword_5031FA8[0] + 8 * v17) = v16;
  qword_5031FA8[9] = &unk_49D9748;
  ++LODWORD(qword_5031FA8[1]);
  qword_5031FA8[8] = 0;
  qword_5031F60 = &unk_49DC090;
  qword_5031FA8[10] = 0;
  qword_5031FA8[11] = &unk_49DC1D0;
  qword_5031FA8[15] = nullsub_23;
  qword_5031FA8[14] = sub_984030;
  sub_C53080(&qword_5031F60, "codegen-data-thinlto-two-rounds", 31);
  LOBYTE(qword_5031FA8[8]) = 0;
  LOWORD(qword_5031FA8[10]) = 256;
  unk_5031F90 = 157;
  LOBYTE(word_5031F6C) = word_5031F6C & 0x9F | 0x20;
  unk_5031F88 = "Enable two-round ThinLTO code generation. The first round emits codegen data, while the second round use"
                "s the emitted codegen data for further optimizations.";
  sub_C53130(&qword_5031F60);
  __cxa_atexit(sub_984900, &qword_5031F60, &qword_4A427C0);
  return __cxa_atexit(sub_31121F0, &unk_5031F48, &qword_4A427C0);
}
