// Function: ctor_672
// Address: 0x5a15a0
//
int __fastcall ctor_672(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
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
  unsigned __int64 v18; // rdx
  __int64 v20; // [rsp+8h] [rbp-38h]

  qword_503D140 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_503D190 = 0x100000000LL;
  dword_503D14C &= 0x8000u;
  word_503D150 = 0;
  qword_503D158 = 0;
  qword_503D160 = 0;
  dword_503D148 = v4;
  qword_503D168 = 0;
  qword_503D170 = 0;
  qword_503D178 = 0;
  qword_503D180 = 0;
  qword_503D188 = (__int64)&unk_503D198;
  qword_503D1A0 = 0;
  qword_503D1A8 = (__int64)&unk_503D1C0;
  qword_503D1B0 = 1;
  dword_503D1B8 = 0;
  byte_503D1BC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_503D190;
  v7 = (unsigned int)qword_503D190 + 1LL;
  if ( v7 > HIDWORD(qword_503D190) )
  {
    sub_C8D5F0((char *)&unk_503D198 - 16, &unk_503D198, v7, 8);
    v6 = (unsigned int)qword_503D190;
  }
  *(_QWORD *)(qword_503D188 + 8 * v6) = v5;
  qword_503D1C8 = (__int64)&byte_503D1D8;
  qword_503D1F0 = (__int64)&byte_503D200;
  qword_503D1E8 = (__int64)&unk_49DC130;
  qword_503D140 = (__int64)&unk_49DC010;
  LODWORD(qword_503D190) = qword_503D190 + 1;
  qword_503D1D0 = 0;
  qword_503D218 = (__int64)&unk_49DC350;
  byte_503D1D8 = 0;
  qword_503D238 = (__int64)nullsub_92;
  qword_503D1F8 = 0;
  qword_503D230 = (__int64)sub_BC4D70;
  byte_503D200 = 0;
  byte_503D210 = 0;
  sub_C53080(&qword_503D140, "mcfg-func-name", 14);
  qword_503D170 = 70;
  LOBYTE(dword_503D14C) = dword_503D14C & 0x9F | 0x20;
  qword_503D168 = (__int64)"The name of a function (or its substring) whose CFG is viewed/printed.";
  sub_C53130(&qword_503D140);
  __cxa_atexit(sub_BC5A40, &qword_503D140, &qword_4A427C0);
  qword_503D040 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_BC5A40, &qword_503D140, v8, v9), 1u);
  qword_503D090 = 0x100000000LL;
  word_503D050 = 0;
  dword_503D04C &= 0x8000u;
  qword_503D058 = 0;
  qword_503D060 = 0;
  dword_503D048 = v10;
  qword_503D068 = 0;
  qword_503D070 = 0;
  qword_503D078 = 0;
  qword_503D080 = 0;
  qword_503D088 = (__int64)&unk_503D098;
  qword_503D0A0 = 0;
  qword_503D0A8 = (__int64)&unk_503D0C0;
  qword_503D0B0 = 1;
  dword_503D0B8 = 0;
  byte_503D0BC = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_503D090;
  if ( (unsigned __int64)(unsigned int)qword_503D090 + 1 > HIDWORD(qword_503D090) )
  {
    v20 = v11;
    sub_C8D5F0((char *)&unk_503D098 - 16, &unk_503D098, (unsigned int)qword_503D090 + 1LL, 8);
    v12 = (unsigned int)qword_503D090;
    v11 = v20;
  }
  *(_QWORD *)(qword_503D088 + 8 * v12) = v11;
  qword_503D0C8 = &byte_503D0D8;
  qword_503D0F0 = (__int64)&byte_503D100;
  qword_503D0E8 = (__int64)&unk_49DC130;
  qword_503D040 = (__int64)&unk_49DC010;
  LODWORD(qword_503D090) = qword_503D090 + 1;
  qword_503D0D0 = 0;
  qword_503D118 = (__int64)&unk_49DC350;
  byte_503D0D8 = 0;
  qword_503D138 = (__int64)nullsub_92;
  qword_503D0F8 = 0;
  qword_503D130 = (__int64)sub_BC4D70;
  byte_503D100 = 0;
  byte_503D110 = 0;
  sub_C53080(&qword_503D040, "mcfg-dot-filename-prefix", 24);
  qword_503D070 = 51;
  LOBYTE(dword_503D04C) = dword_503D04C & 0x9F | 0x20;
  qword_503D068 = (__int64)"The prefix used for the Machine CFG dot file names.";
  sub_C53130(&qword_503D040);
  __cxa_atexit(sub_BC5A40, &qword_503D040, &qword_4A427C0);
  qword_503CF60 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_BC5A40, &qword_503D040, v13, v14), 1u);
  byte_503CFDC = 1;
  qword_503CFB0 = 0x100000000LL;
  dword_503CF6C &= 0x8000u;
  qword_503CF78 = 0;
  qword_503CF80 = 0;
  qword_503CF88 = 0;
  dword_503CF68 = v15;
  word_503CF70 = 0;
  qword_503CF90 = 0;
  qword_503CF98 = 0;
  qword_503CFA0 = 0;
  qword_503CFA8 = (__int64)&unk_503CFB8;
  qword_503CFC0 = 0;
  qword_503CFC8 = (__int64)&unk_503CFE0;
  qword_503CFD0 = 1;
  dword_503CFD8 = 0;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_503CFB0;
  v18 = (unsigned int)qword_503CFB0 + 1LL;
  if ( v18 > HIDWORD(qword_503CFB0) )
  {
    sub_C8D5F0((char *)&unk_503CFB8 - 16, &unk_503CFB8, v18, 8);
    v17 = (unsigned int)qword_503CFB0;
  }
  *(_QWORD *)(qword_503CFA8 + 8 * v17) = v16;
  LODWORD(qword_503CFB0) = qword_503CFB0 + 1;
  qword_503CFE8 = 0;
  qword_503CFF0 = (__int64)&unk_49D9748;
  qword_503CFF8 = 0;
  qword_503CF60 = (__int64)&unk_49DC090;
  qword_503D000 = (__int64)&unk_49DC1D0;
  qword_503D020 = (__int64)nullsub_23;
  qword_503D018 = (__int64)sub_984030;
  sub_C53080(&qword_503CF60, "dot-mcfg-only", 13);
  LOBYTE(qword_503CFE8) = 0;
  LOWORD(qword_503CFF8) = 256;
  qword_503CF90 = 38;
  LOBYTE(dword_503CF6C) = dword_503CF6C & 0x9F | 0x20;
  qword_503CF88 = (__int64)"Print only the CFG without blocks body";
  sub_C53130(&qword_503CF60);
  return __cxa_atexit(sub_984900, &qword_503CF60, &qword_4A427C0);
}
