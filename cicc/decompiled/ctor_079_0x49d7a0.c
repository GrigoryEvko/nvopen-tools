// Function: ctor_079
// Address: 0x49d7a0
//
int ctor_079()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v14; // [rsp+8h] [rbp-38h]
  __int64 v15; // [rsp+8h] [rbp-38h]

  qword_4F8ECA0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8ECF0 = 0x100000000LL;
  dword_4F8ECAC &= 0x8000u;
  word_4F8ECB0 = 0;
  qword_4F8ECB8 = 0;
  qword_4F8ECC0 = 0;
  dword_4F8ECA8 = v0;
  qword_4F8ECC8 = 0;
  qword_4F8ECD0 = 0;
  qword_4F8ECD8 = 0;
  qword_4F8ECE0 = 0;
  qword_4F8ECE8 = (__int64)&unk_4F8ECF8;
  qword_4F8ED00 = 0;
  qword_4F8ED08 = (__int64)&unk_4F8ED20;
  qword_4F8ED10 = 1;
  dword_4F8ED18 = 0;
  byte_4F8ED1C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F8ECF0;
  v3 = (unsigned int)qword_4F8ECF0 + 1LL;
  if ( v3 > HIDWORD(qword_4F8ECF0) )
  {
    sub_C8D5F0((char *)&unk_4F8ECF8 - 16, &unk_4F8ECF8, v3, 8);
    v2 = (unsigned int)qword_4F8ECF0;
  }
  *(_QWORD *)(qword_4F8ECE8 + 8 * v2) = v1;
  qword_4F8ED30 = (__int64)&unk_49D9728;
  LODWORD(qword_4F8ECF0) = qword_4F8ECF0 + 1;
  qword_4F8ED28 = 0;
  qword_4F8ECA0 = (__int64)&unk_49DBF10;
  qword_4F8ED40 = (__int64)&unk_49DC290;
  qword_4F8ED38 = 0;
  qword_4F8ED60 = (__int64)nullsub_24;
  qword_4F8ED58 = (__int64)sub_984050;
  sub_C53080(&qword_4F8ECA0, "icp-remaining-percent-threshold", 31);
  LODWORD(qword_4F8ED28) = 30;
  BYTE4(qword_4F8ED38) = 1;
  LODWORD(qword_4F8ED38) = 30;
  qword_4F8ECD0 = 91;
  LOBYTE(dword_4F8ECAC) = dword_4F8ECAC & 0x9F | 0x20;
  qword_4F8ECC8 = (__int64)"The percentage threshold against remaining unpromoted indirect call count for the promotion";
  sub_C53130(&qword_4F8ECA0);
  __cxa_atexit(sub_984970, &qword_4F8ECA0, &qword_4A427C0);
  qword_4F8EBC0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8EC10 = 0x100000000LL;
  word_4F8EBD0 = 0;
  dword_4F8EBCC &= 0x8000u;
  qword_4F8EBD8 = 0;
  qword_4F8EBE0 = 0;
  dword_4F8EBC8 = v4;
  qword_4F8EBE8 = 0;
  qword_4F8EBF0 = 0;
  qword_4F8EBF8 = 0;
  qword_4F8EC00 = 0;
  qword_4F8EC08 = (__int64)&unk_4F8EC18;
  qword_4F8EC20 = 0;
  qword_4F8EC28 = (__int64)&unk_4F8EC40;
  qword_4F8EC30 = 1;
  dword_4F8EC38 = 0;
  byte_4F8EC3C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F8EC10;
  if ( (unsigned __int64)(unsigned int)qword_4F8EC10 + 1 > HIDWORD(qword_4F8EC10) )
  {
    v14 = v5;
    sub_C8D5F0((char *)&unk_4F8EC18 - 16, &unk_4F8EC18, (unsigned int)qword_4F8EC10 + 1LL, 8);
    v6 = (unsigned int)qword_4F8EC10;
    v5 = v14;
  }
  *(_QWORD *)(qword_4F8EC08 + 8 * v6) = v5;
  qword_4F8EC50 = (__int64)&unk_49D9728;
  LODWORD(qword_4F8EC10) = qword_4F8EC10 + 1;
  qword_4F8EC48 = 0;
  qword_4F8EBC0 = (__int64)&unk_49DBF10;
  qword_4F8EC60 = (__int64)&unk_49DC290;
  qword_4F8EC58 = 0;
  qword_4F8EC80 = (__int64)nullsub_24;
  qword_4F8EC78 = (__int64)sub_984050;
  sub_C53080(&qword_4F8EBC0, "icp-total-percent-threshold", 27);
  LODWORD(qword_4F8EC48) = 5;
  BYTE4(qword_4F8EC58) = 1;
  LODWORD(qword_4F8EC58) = 5;
  qword_4F8EBF0 = 62;
  LOBYTE(dword_4F8EBCC) = dword_4F8EBCC & 0x9F | 0x20;
  qword_4F8EBE8 = (__int64)"The percentage threshold against total count for the promotion";
  sub_C53130(&qword_4F8EBC0);
  __cxa_atexit(sub_984970, &qword_4F8EBC0, &qword_4A427C0);
  qword_4F8EAE0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F8EB5C = 1;
  qword_4F8EB30 = 0x100000000LL;
  dword_4F8EAEC &= 0x8000u;
  qword_4F8EAF8 = 0;
  qword_4F8EB00 = 0;
  qword_4F8EB08 = 0;
  dword_4F8EAE8 = v7;
  word_4F8EAF0 = 0;
  qword_4F8EB10 = 0;
  qword_4F8EB18 = 0;
  qword_4F8EB20 = 0;
  qword_4F8EB28 = (__int64)&unk_4F8EB38;
  qword_4F8EB40 = 0;
  qword_4F8EB48 = (__int64)&unk_4F8EB60;
  qword_4F8EB50 = 1;
  dword_4F8EB58 = 0;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4F8EB30;
  if ( (unsigned __int64)(unsigned int)qword_4F8EB30 + 1 > HIDWORD(qword_4F8EB30) )
  {
    v15 = v8;
    sub_C8D5F0((char *)&unk_4F8EB38 - 16, &unk_4F8EB38, (unsigned int)qword_4F8EB30 + 1LL, 8);
    v9 = (unsigned int)qword_4F8EB30;
    v8 = v15;
  }
  *(_QWORD *)(qword_4F8EB28 + 8 * v9) = v8;
  qword_4F8EB70 = (__int64)&unk_49D9728;
  LODWORD(qword_4F8EB30) = qword_4F8EB30 + 1;
  qword_4F8EB68 = 0;
  qword_4F8EAE0 = (__int64)&unk_49DBF10;
  qword_4F8EB80 = (__int64)&unk_49DC290;
  qword_4F8EB78 = 0;
  qword_4F8EBA0 = (__int64)nullsub_24;
  qword_4F8EB98 = (__int64)sub_984050;
  sub_C53080(&qword_4F8EAE0, "icp-max-prom", 12);
  LODWORD(qword_4F8EB68) = 3;
  BYTE4(qword_4F8EB78) = 1;
  LODWORD(qword_4F8EB78) = 3;
  qword_4F8EB10 = 60;
  LOBYTE(dword_4F8EAEC) = dword_4F8EAEC & 0x9F | 0x20;
  qword_4F8EB08 = (__int64)"Max number of promotions for a single indirect call callsite";
  sub_C53130(&qword_4F8EAE0);
  __cxa_atexit(sub_984970, &qword_4F8EAE0, &qword_4A427C0);
  qword_4F8EA00 = &unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)&word_4F8EA0C = word_4F8EA0C & 0x8000;
  unk_4F8EA10 = 0;
  qword_4F8EA48[1] = 0x100000000LL;
  unk_4F8EA08 = v10;
  qword_4F8EA48[0] = &qword_4F8EA48[2];
  unk_4F8EA18 = 0;
  unk_4F8EA20 = 0;
  unk_4F8EA28 = 0;
  unk_4F8EA30 = 0;
  unk_4F8EA38 = 0;
  unk_4F8EA40 = 0;
  qword_4F8EA48[3] = 0;
  qword_4F8EA48[4] = &qword_4F8EA48[7];
  qword_4F8EA48[5] = 1;
  LODWORD(qword_4F8EA48[6]) = 0;
  BYTE4(qword_4F8EA48[6]) = 1;
  v11 = sub_C57470();
  v12 = LODWORD(qword_4F8EA48[1]);
  if ( (unsigned __int64)LODWORD(qword_4F8EA48[1]) + 1 > HIDWORD(qword_4F8EA48[1]) )
  {
    sub_C8D5F0(qword_4F8EA48, &qword_4F8EA48[2], LODWORD(qword_4F8EA48[1]) + 1LL, 8);
    v12 = LODWORD(qword_4F8EA48[1]);
  }
  *(_QWORD *)(qword_4F8EA48[0] + 8 * v12) = v11;
  qword_4F8EA48[9] = &unk_49D9728;
  ++LODWORD(qword_4F8EA48[1]);
  qword_4F8EA48[8] = 0;
  qword_4F8EA00 = &unk_49DBF10;
  qword_4F8EA48[11] = &unk_49DC290;
  qword_4F8EA48[10] = 0;
  qword_4F8EA48[15] = nullsub_24;
  qword_4F8EA48[14] = sub_984050;
  sub_C53080(&qword_4F8EA00, "icp-max-num-vtables", 19);
  BYTE4(qword_4F8EA48[10]) = 1;
  LODWORD(qword_4F8EA48[8]) = 6;
  unk_4F8EA30 = 62;
  LODWORD(qword_4F8EA48[10]) = 6;
  LOBYTE(word_4F8EA0C) = word_4F8EA0C & 0x9F | 0x20;
  unk_4F8EA28 = "Max number of vtables annotated for a vtable load instruction.";
  sub_C53130(&qword_4F8EA00);
  return __cxa_atexit(sub_984970, &qword_4F8EA00, &qword_4A427C0);
}
