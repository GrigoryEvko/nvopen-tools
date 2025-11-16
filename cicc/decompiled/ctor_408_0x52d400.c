// Function: ctor_408
// Address: 0x52d400
//
int ctor_408()
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
  __int64 v12; // [rsp+18h] [rbp-58h]
  char v13; // [rsp+23h] [rbp-4Dh] BYREF
  int v14; // [rsp+24h] [rbp-4Ch] BYREF
  char *v15; // [rsp+28h] [rbp-48h] BYREF
  const char *v16; // [rsp+30h] [rbp-40h] BYREF
  __int64 v17; // [rsp+38h] [rbp-38h]

  qword_4FECFA0 = (__int64)"__sanitizer_metadata_covered";
  qword_4FECFA8 = 28;
  qword_4FECFB0 = (__int64)"sanmd_covered";
  qword_4FECF80 = (__int64)"__sanitizer_metadata_atomics";
  qword_4FECFB8 = 13;
  qword_4FECF90 = (__int64)"sanmd_atomics";
  qword_4FECF88 = 28;
  qword_4FECF98 = 13;
  qword_4FECEA0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FECEF0 = 0x100000000LL;
  dword_4FECEAC &= 0x8000u;
  word_4FECEB0 = 0;
  qword_4FECEB8 = 0;
  qword_4FECEC0 = 0;
  dword_4FECEA8 = v0;
  qword_4FECEC8 = 0;
  qword_4FECED0 = 0;
  qword_4FECED8 = 0;
  qword_4FECEE0 = 0;
  qword_4FECEE8 = (__int64)&unk_4FECEF8;
  qword_4FECF00 = 0;
  qword_4FECF08 = (__int64)&unk_4FECF20;
  qword_4FECF10 = 1;
  dword_4FECF18 = 0;
  byte_4FECF1C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FECEF0;
  v3 = (unsigned int)qword_4FECEF0 + 1LL;
  if ( v3 > HIDWORD(qword_4FECEF0) )
  {
    sub_C8D5F0((char *)&unk_4FECEF8 - 16, &unk_4FECEF8, v3, 8);
    v2 = (unsigned int)qword_4FECEF0;
  }
  *(_QWORD *)(qword_4FECEE8 + 8 * v2) = v1;
  qword_4FECF30 = (__int64)&unk_49D9748;
  qword_4FECEA0 = (__int64)&unk_49DC090;
  qword_4FECF40 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FECEF0) = qword_4FECEF0 + 1;
  qword_4FECF60 = (__int64)nullsub_23;
  qword_4FECF28 = 0;
  qword_4FECF58 = (__int64)sub_984030;
  qword_4FECF38 = 0;
  sub_C53080(&qword_4FECEA0, "sanitizer-metadata-weak-callbacks", 33);
  qword_4FECEC8 = (__int64)"Declare callbacks extern weak, and only call if non-null.";
  LOWORD(qword_4FECF38) = 257;
  LOBYTE(qword_4FECF28) = 1;
  qword_4FECED0 = 57;
  LOBYTE(dword_4FECEAC) = dword_4FECEAC & 0x9F | 0x20;
  sub_C53130(&qword_4FECEA0);
  __cxa_atexit(sub_984900, &qword_4FECEA0, &qword_4A427C0);
  qword_4FECDC0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FECE10 = 0x100000000LL;
  dword_4FECDCC &= 0x8000u;
  qword_4FECE08 = (__int64)&unk_4FECE18;
  word_4FECDD0 = 0;
  qword_4FECDD8 = 0;
  dword_4FECDC8 = v4;
  qword_4FECDE0 = 0;
  qword_4FECDE8 = 0;
  qword_4FECDF0 = 0;
  qword_4FECDF8 = 0;
  qword_4FECE00 = 0;
  qword_4FECE20 = 0;
  qword_4FECE28 = (__int64)&unk_4FECE40;
  qword_4FECE30 = 1;
  dword_4FECE38 = 0;
  byte_4FECE3C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FECE10;
  if ( (unsigned __int64)(unsigned int)qword_4FECE10 + 1 > HIDWORD(qword_4FECE10) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_4FECE18 - 16, &unk_4FECE18, (unsigned int)qword_4FECE10 + 1LL, 8);
    v6 = (unsigned int)qword_4FECE10;
    v5 = v12;
  }
  *(_QWORD *)(qword_4FECE08 + 8 * v6) = v5;
  qword_4FECE50 = (__int64)&unk_49D9748;
  qword_4FECDC0 = (__int64)&unk_49DC090;
  qword_4FECE60 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FECE10) = qword_4FECE10 + 1;
  qword_4FECE80 = (__int64)nullsub_23;
  qword_4FECE48 = 0;
  qword_4FECE78 = (__int64)sub_984030;
  qword_4FECE58 = 0;
  sub_C53080(&qword_4FECDC0, "sanitizer-metadata-nosanitize-attr", 34);
  qword_4FECDE8 = (__int64)"Mark some metadata features uncovered in functions with associated no_sanitize attributes.";
  LOWORD(qword_4FECE58) = 257;
  LOBYTE(qword_4FECE48) = 1;
  qword_4FECDF0 = 90;
  LOBYTE(dword_4FECDCC) = dword_4FECDCC & 0x9F | 0x20;
  sub_C53130(&qword_4FECDC0);
  __cxa_atexit(sub_984900, &qword_4FECDC0, &qword_4A427C0);
  v13 = 0;
  v15 = &v13;
  v16 = "Emit PCs for covered functions.";
  v14 = 1;
  v17 = 31;
  sub_248A750(&unk_4FECCE0, "sanitizer-metadata-covered", &v16, &v14, &v15);
  __cxa_atexit(sub_984900, &unk_4FECCE0, &qword_4A427C0);
  v16 = "Emit PCs for atomic operations.";
  v15 = &v13;
  v13 = 0;
  v14 = 1;
  v17 = 31;
  sub_248A750(&unk_4FECC00, "sanitizer-metadata-atomics", &v16, &v14, &v15);
  __cxa_atexit(sub_984900, &unk_4FECC00, &qword_4A427C0);
  qword_4FECB20 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FECB70 = 0x100000000LL;
  dword_4FECB2C &= 0x8000u;
  word_4FECB30 = 0;
  qword_4FECB68 = (__int64)&unk_4FECB78;
  qword_4FECB38 = 0;
  dword_4FECB28 = v7;
  qword_4FECB40 = 0;
  qword_4FECB48 = 0;
  qword_4FECB50 = 0;
  qword_4FECB58 = 0;
  qword_4FECB60 = 0;
  qword_4FECB80 = 0;
  qword_4FECB88 = (__int64)&unk_4FECBA0;
  qword_4FECB90 = 1;
  dword_4FECB98 = 0;
  byte_4FECB9C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4FECB70;
  v10 = (unsigned int)qword_4FECB70 + 1LL;
  if ( v10 > HIDWORD(qword_4FECB70) )
  {
    sub_C8D5F0((char *)&unk_4FECB78 - 16, &unk_4FECB78, v10, 8);
    v9 = (unsigned int)qword_4FECB70;
  }
  *(_QWORD *)(qword_4FECB68 + 8 * v9) = v8;
  qword_4FECBB0 = (__int64)&unk_49D9748;
  qword_4FECB20 = (__int64)&unk_49DC090;
  qword_4FECBC0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FECB70) = qword_4FECB70 + 1;
  qword_4FECBE0 = (__int64)nullsub_23;
  qword_4FECBA8 = 0;
  qword_4FECBD8 = (__int64)sub_984030;
  qword_4FECBB8 = 0;
  sub_C53080(&qword_4FECB20, "sanitizer-metadata-uar", 22);
  qword_4FECB50 = 78;
  qword_4FECB48 = (__int64)"Emit PCs for start of functions that are subject for use-after-return checking";
  LOBYTE(qword_4FECBA8) = 0;
  LOBYTE(dword_4FECB2C) = dword_4FECB2C & 0x9F | 0x20;
  LOWORD(qword_4FECBB8) = 256;
  sub_C53130(&qword_4FECB20);
  return __cxa_atexit(sub_984900, &qword_4FECB20, &qword_4A427C0);
}
