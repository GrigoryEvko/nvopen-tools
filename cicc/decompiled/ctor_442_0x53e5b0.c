// Function: ctor_442
// Address: 0x53e5b0
//
int ctor_442()
{
  __int64 v0; // rax
  __int64 v1; // r12
  int v2; // edx
  __int64 v3; // r12
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  int v6; // edx
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  _QWORD v11[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v12[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v13[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v14[8]; // [rsp+30h] [rbp-40h] BYREF

  v0 = sub_C60B10();
  v13[0] = v14;
  v1 = v0;
  sub_2739300(v13, "Controls which conditions are eliminated");
  v11[0] = v12;
  sub_2739300(v11, "conds-eliminated");
  sub_CF9810(v1, v11, v13);
  if ( (_QWORD *)v11[0] != v12 )
    j_j___libc_free_0(v11[0], v12[0] + 1LL);
  if ( (_QWORD *)v13[0] != v14 )
    j_j___libc_free_0(v13[0], v14[0] + 1LL);
  qword_4FFA140 = (__int64)&unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFA190 = 0x100000000LL;
  word_4FFA150 = 0;
  dword_4FFA14C &= 0x8000u;
  qword_4FFA158 = 0;
  qword_4FFA160 = 0;
  dword_4FFA148 = v2;
  qword_4FFA168 = 0;
  qword_4FFA170 = 0;
  qword_4FFA178 = 0;
  qword_4FFA180 = 0;
  qword_4FFA188 = (__int64)&unk_4FFA198;
  qword_4FFA1A0 = 0;
  qword_4FFA1A8 = (__int64)&unk_4FFA1C0;
  qword_4FFA1B0 = 1;
  dword_4FFA1B8 = 0;
  byte_4FFA1BC = 1;
  v3 = sub_C57470();
  v4 = (unsigned int)qword_4FFA190;
  v5 = (unsigned int)qword_4FFA190 + 1LL;
  if ( v5 > HIDWORD(qword_4FFA190) )
  {
    sub_C8D5F0((char *)&unk_4FFA198 - 16, &unk_4FFA198, v5, 8);
    v4 = (unsigned int)qword_4FFA190;
  }
  *(_QWORD *)(qword_4FFA188 + 8 * v4) = v3;
  LODWORD(qword_4FFA190) = qword_4FFA190 + 1;
  qword_4FFA1C8 = 0;
  qword_4FFA1D0 = (__int64)&unk_49D9728;
  qword_4FFA1D8 = 0;
  qword_4FFA140 = (__int64)&unk_49DBF10;
  qword_4FFA1E0 = (__int64)&unk_49DC290;
  qword_4FFA200 = (__int64)nullsub_24;
  qword_4FFA1F8 = (__int64)sub_984050;
  sub_C53080(&qword_4FFA140, "constraint-elimination-max-rows", 31);
  LODWORD(qword_4FFA1C8) = 500;
  BYTE4(qword_4FFA1D8) = 1;
  LODWORD(qword_4FFA1D8) = 500;
  qword_4FFA170 = 51;
  LOBYTE(dword_4FFA14C) = dword_4FFA14C & 0x9F | 0x20;
  qword_4FFA168 = (__int64)"Maximum number of rows to keep in constraint system";
  sub_C53130(&qword_4FFA140);
  __cxa_atexit(sub_984970, &qword_4FFA140, &qword_4A427C0);
  qword_4FFA060 = (__int64)&unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FFA0DC = 1;
  qword_4FFA0B0 = 0x100000000LL;
  dword_4FFA06C &= 0x8000u;
  qword_4FFA078 = 0;
  qword_4FFA080 = 0;
  qword_4FFA088 = 0;
  dword_4FFA068 = v6;
  word_4FFA070 = 0;
  qword_4FFA090 = 0;
  qword_4FFA098 = 0;
  qword_4FFA0A0 = 0;
  qword_4FFA0A8 = (__int64)&unk_4FFA0B8;
  qword_4FFA0C0 = 0;
  qword_4FFA0C8 = (__int64)&unk_4FFA0E0;
  qword_4FFA0D0 = 1;
  dword_4FFA0D8 = 0;
  v7 = sub_C57470();
  v8 = (unsigned int)qword_4FFA0B0;
  v9 = (unsigned int)qword_4FFA0B0 + 1LL;
  if ( v9 > HIDWORD(qword_4FFA0B0) )
  {
    sub_C8D5F0((char *)&unk_4FFA0B8 - 16, &unk_4FFA0B8, v9, 8);
    v8 = (unsigned int)qword_4FFA0B0;
  }
  *(_QWORD *)(qword_4FFA0A8 + 8 * v8) = v7;
  LODWORD(qword_4FFA0B0) = qword_4FFA0B0 + 1;
  qword_4FFA0E8 = 0;
  qword_4FFA0F0 = (__int64)&unk_49D9748;
  qword_4FFA0F8 = 0;
  qword_4FFA060 = (__int64)&unk_49DC090;
  qword_4FFA100 = (__int64)&unk_49DC1D0;
  qword_4FFA120 = (__int64)nullsub_23;
  qword_4FFA118 = (__int64)sub_984030;
  sub_C53080(&qword_4FFA060, "constraint-elimination-dump-reproducers", 39);
  LOBYTE(qword_4FFA0E8) = 0;
  LOWORD(qword_4FFA0F8) = 256;
  qword_4FFA090 = 48;
  LOBYTE(dword_4FFA06C) = dword_4FFA06C & 0x9F | 0x20;
  qword_4FFA088 = (__int64)"Dump IR to reproduce successful transformations.";
  sub_C53130(&qword_4FFA060);
  return __cxa_atexit(sub_984900, &qword_4FFA060, &qword_4A427C0);
}
