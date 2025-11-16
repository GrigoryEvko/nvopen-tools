// Function: ctor_622
// Address: 0x58bed0
//
int __fastcall ctor_622(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
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
  _QWORD v15[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v16[6]; // [rsp+10h] [rbp-30h] BYREF

  qword_502ECC0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_502ED10 = 0x100000000LL;
  word_502ECD0 = 0;
  dword_502ECCC &= 0x8000u;
  qword_502ECD8 = 0;
  qword_502ECE0 = 0;
  dword_502ECC8 = v4;
  qword_502ECE8 = 0;
  qword_502ECF0 = 0;
  qword_502ECF8 = 0;
  qword_502ED00 = 0;
  qword_502ED08 = (__int64)&unk_502ED18;
  qword_502ED20 = 0;
  qword_502ED28 = (__int64)&unk_502ED40;
  qword_502ED30 = 1;
  dword_502ED38 = 0;
  byte_502ED3C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_502ED10;
  v7 = (unsigned int)qword_502ED10 + 1LL;
  if ( v7 > HIDWORD(qword_502ED10) )
  {
    sub_C8D5F0((char *)&unk_502ED18 - 16, &unk_502ED18, v7, 8);
    v6 = (unsigned int)qword_502ED10;
  }
  *(_QWORD *)(qword_502ED08 + 8 * v6) = v5;
  LODWORD(qword_502ED10) = qword_502ED10 + 1;
  qword_502ED48 = 0;
  qword_502ED50 = (__int64)&unk_49D9748;
  qword_502ED58 = 0;
  qword_502ECC0 = (__int64)&unk_49DC090;
  qword_502ED60 = (__int64)&unk_49DC1D0;
  qword_502ED80 = (__int64)nullsub_23;
  qword_502ED78 = (__int64)sub_984030;
  sub_C53080(&qword_502ECC0, "dot-ddg-only", 12);
  qword_502ECF0 = 20;
  LOBYTE(dword_502ECCC) = dword_502ECCC & 0x9F | 0x20;
  qword_502ECE8 = (__int64)"simple ddg dot graph";
  sub_C53130(&qword_502ECC0);
  __cxa_atexit(sub_984900, &qword_502ECC0, &qword_4A427C0);
  qword_502EBC0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502ECC0, v8, v9), 1u);
  byte_502EC3C = 1;
  qword_502EC10 = 0x100000000LL;
  dword_502EBCC &= 0x8000u;
  qword_502EBD8 = 0;
  qword_502EBE0 = 0;
  qword_502EBE8 = 0;
  dword_502EBC8 = v10;
  word_502EBD0 = 0;
  qword_502EBF0 = 0;
  qword_502EBF8 = 0;
  qword_502EC00 = 0;
  qword_502EC08 = (__int64)&unk_502EC18;
  qword_502EC20 = 0;
  qword_502EC28 = (__int64)&unk_502EC40;
  qword_502EC30 = 1;
  dword_502EC38 = 0;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_502EC10;
  v13 = (unsigned int)qword_502EC10 + 1LL;
  if ( v13 > HIDWORD(qword_502EC10) )
  {
    sub_C8D5F0((char *)&unk_502EC18 - 16, &unk_502EC18, v13, 8);
    v12 = (unsigned int)qword_502EC10;
  }
  *(_QWORD *)(qword_502EC08 + 8 * v12) = v11;
  qword_502EC48 = (__int64)&byte_502EC58;
  qword_502EC70 = (__int64)&byte_502EC80;
  LODWORD(qword_502EC10) = qword_502EC10 + 1;
  qword_502EC50 = 0;
  qword_502EC68 = (__int64)&unk_49DC130;
  byte_502EC58 = 0;
  byte_502EC80 = 0;
  qword_502EBC0 = (__int64)&unk_49DC010;
  qword_502EC78 = 0;
  byte_502EC90 = 0;
  qword_502EC98 = (__int64)&unk_49DC350;
  qword_502ECB8 = (__int64)nullsub_92;
  qword_502ECB0 = (__int64)sub_BC4D70;
  sub_C53080(&qword_502EBC0, "dot-ddg-filename-prefix", 23);
  v15[0] = v16;
  LODWORD(v16[0]) = 6775908;
  v15[1] = 3;
  sub_2240AE0(&qword_502EC48, v15);
  byte_502EC90 = 1;
  sub_2240AE0(&qword_502EC70, v15);
  if ( (_QWORD *)v15[0] != v16 )
    j_j___libc_free_0(v15[0], v16[0] + 1LL);
  qword_502EBF0 = 43;
  LOBYTE(dword_502EBCC) = dword_502EBCC & 0x9F | 0x20;
  qword_502EBE8 = (__int64)"The prefix used for the DDG dot file names.";
  sub_C53130(&qword_502EBC0);
  return __cxa_atexit(sub_BC5A40, &qword_502EBC0, &qword_4A427C0);
}
