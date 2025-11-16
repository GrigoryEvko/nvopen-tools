// Function: ctor_441
// Address: 0x53df90
//
int ctor_441()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-38h]

  qword_4FF9F80 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF9FD0 = 0x100000000LL;
  dword_4FF9F8C &= 0x8000u;
  word_4FF9F90 = 0;
  qword_4FF9F98 = 0;
  qword_4FF9FA0 = 0;
  dword_4FF9F88 = v0;
  qword_4FF9FA8 = 0;
  qword_4FF9FB0 = 0;
  qword_4FF9FB8 = 0;
  qword_4FF9FC0 = 0;
  qword_4FF9FC8 = (__int64)&unk_4FF9FD8;
  qword_4FF9FE0 = 0;
  qword_4FF9FE8 = (__int64)&unk_4FFA000;
  qword_4FF9FF0 = 1;
  dword_4FF9FF8 = 0;
  byte_4FF9FFC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF9FD0;
  v3 = (unsigned int)qword_4FF9FD0 + 1LL;
  if ( v3 > HIDWORD(qword_4FF9FD0) )
  {
    sub_C8D5F0((char *)&unk_4FF9FD8 - 16, &unk_4FF9FD8, v3, 8);
    v2 = (unsigned int)qword_4FF9FD0;
  }
  *(_QWORD *)(qword_4FF9FC8 + 8 * v2) = v1;
  qword_4FFA010 = (__int64)&unk_49D9748;
  qword_4FF9F80 = (__int64)&unk_49DC090;
  LODWORD(qword_4FF9FD0) = qword_4FF9FD0 + 1;
  qword_4FFA008 = 0;
  qword_4FFA020 = (__int64)&unk_49DC1D0;
  qword_4FFA018 = 0;
  qword_4FFA040 = (__int64)nullsub_23;
  qword_4FFA038 = (__int64)sub_984030;
  sub_C53080(&qword_4FF9F80, "consthoist-with-block-frequency", 31);
  LOWORD(qword_4FFA018) = 257;
  LOBYTE(qword_4FFA008) = 1;
  qword_4FF9FB0 = 139;
  LOBYTE(dword_4FF9F8C) = dword_4FF9F8C & 0x9F | 0x20;
  qword_4FF9FA8 = (__int64)"Enable the use of the block frequency analysis to reduce the chance to execute const material"
                           "ization more frequently than without hoisting.";
  sub_C53130(&qword_4FF9F80);
  __cxa_atexit(sub_984900, &qword_4FF9F80, &qword_4A427C0);
  qword_4FF9EA0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF9EF0 = 0x100000000LL;
  dword_4FF9EAC &= 0x8000u;
  word_4FF9EB0 = 0;
  qword_4FF9EB8 = 0;
  qword_4FF9EC0 = 0;
  dword_4FF9EA8 = v4;
  qword_4FF9EC8 = 0;
  qword_4FF9ED0 = 0;
  qword_4FF9ED8 = 0;
  qword_4FF9EE0 = 0;
  qword_4FF9EE8 = (__int64)&unk_4FF9EF8;
  qword_4FF9F00 = 0;
  qword_4FF9F08 = (__int64)&unk_4FF9F20;
  qword_4FF9F10 = 1;
  dword_4FF9F18 = 0;
  byte_4FF9F1C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FF9EF0;
  if ( (unsigned __int64)(unsigned int)qword_4FF9EF0 + 1 > HIDWORD(qword_4FF9EF0) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_4FF9EF8 - 16, &unk_4FF9EF8, (unsigned int)qword_4FF9EF0 + 1LL, 8);
    v6 = (unsigned int)qword_4FF9EF0;
    v5 = v12;
  }
  *(_QWORD *)(qword_4FF9EE8 + 8 * v6) = v5;
  qword_4FF9F30 = (__int64)&unk_49D9748;
  qword_4FF9EA0 = (__int64)&unk_49DC090;
  LODWORD(qword_4FF9EF0) = qword_4FF9EF0 + 1;
  qword_4FF9F28 = 0;
  qword_4FF9F40 = (__int64)&unk_49DC1D0;
  qword_4FF9F38 = 0;
  qword_4FF9F60 = (__int64)nullsub_23;
  qword_4FF9F58 = (__int64)sub_984030;
  sub_C53080(&qword_4FF9EA0, "consthoist-gep", 14);
  LOBYTE(qword_4FF9F28) = 0;
  LOWORD(qword_4FF9F38) = 256;
  qword_4FF9ED0 = 37;
  LOBYTE(dword_4FF9EAC) = dword_4FF9EAC & 0x9F | 0x20;
  qword_4FF9EC8 = (__int64)"Try hoisting constant gep expressions";
  sub_C53130(&qword_4FF9EA0);
  __cxa_atexit(sub_984900, &qword_4FF9EA0, &qword_4A427C0);
  qword_4FF9DC0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FF9E3C = 1;
  qword_4FF9E10 = 0x100000000LL;
  dword_4FF9DCC &= 0x8000u;
  qword_4FF9DD8 = 0;
  qword_4FF9DE0 = 0;
  qword_4FF9DE8 = 0;
  dword_4FF9DC8 = v7;
  word_4FF9DD0 = 0;
  qword_4FF9DF0 = 0;
  qword_4FF9DF8 = 0;
  qword_4FF9E00 = 0;
  qword_4FF9E08 = (__int64)&unk_4FF9E18;
  qword_4FF9E20 = 0;
  qword_4FF9E28 = (__int64)&unk_4FF9E40;
  qword_4FF9E30 = 1;
  dword_4FF9E38 = 0;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4FF9E10;
  v10 = (unsigned int)qword_4FF9E10 + 1LL;
  if ( v10 > HIDWORD(qword_4FF9E10) )
  {
    sub_C8D5F0((char *)&unk_4FF9E18 - 16, &unk_4FF9E18, v10, 8);
    v9 = (unsigned int)qword_4FF9E10;
  }
  *(_QWORD *)(qword_4FF9E08 + 8 * v9) = v8;
  LODWORD(qword_4FF9E10) = qword_4FF9E10 + 1;
  qword_4FF9E48 = 0;
  qword_4FF9E50 = (__int64)&unk_49D9728;
  qword_4FF9E58 = 0;
  qword_4FF9DC0 = (__int64)&unk_49DBF10;
  qword_4FF9E60 = (__int64)&unk_49DC290;
  qword_4FF9E80 = (__int64)nullsub_24;
  qword_4FF9E78 = (__int64)sub_984050;
  sub_C53080(&qword_4FF9DC0, "consthoist-min-num-to-rebase", 28);
  qword_4FF9DF0 = 82;
  qword_4FF9DE8 = (__int64)"Do not rebase if number of dependent constants of a Base is less than this number.";
  LODWORD(qword_4FF9E48) = 0;
  BYTE4(qword_4FF9E58) = 1;
  LODWORD(qword_4FF9E58) = 0;
  LOBYTE(dword_4FF9DCC) = dword_4FF9DCC & 0x9F | 0x20;
  sub_C53130(&qword_4FF9DC0);
  return __cxa_atexit(sub_984970, &qword_4FF9DC0, &qword_4A427C0);
}
