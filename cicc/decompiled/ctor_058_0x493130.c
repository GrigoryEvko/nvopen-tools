// Function: ctor_058
// Address: 0x493130
//
int ctor_058()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rdx

  qword_4F88100 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F88150 = 0x100000000LL;
  dword_4F8810C &= 0x8000u;
  word_4F88110 = 0;
  qword_4F88118 = 0;
  qword_4F88120 = 0;
  dword_4F88108 = v0;
  qword_4F88128 = 0;
  qword_4F88130 = 0;
  qword_4F88138 = 0;
  qword_4F88140 = 0;
  qword_4F88148 = (__int64)&unk_4F88158;
  qword_4F88160 = 0;
  qword_4F88168 = (__int64)&unk_4F88180;
  qword_4F88170 = 1;
  dword_4F88178 = 0;
  byte_4F8817C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F88150;
  v3 = (unsigned int)qword_4F88150 + 1LL;
  if ( v3 > HIDWORD(qword_4F88150) )
  {
    sub_C8D5F0((char *)&unk_4F88158 - 16, &unk_4F88158, v3, 8);
    v2 = (unsigned int)qword_4F88150;
  }
  *(_QWORD *)(qword_4F88148 + 8 * v2) = v1;
  LODWORD(qword_4F88150) = qword_4F88150 + 1;
  qword_4F88188 = 0;
  qword_4F88190 = (__int64)&unk_49DA090;
  qword_4F88198 = 0;
  qword_4F88100 = (__int64)&unk_49DBF90;
  qword_4F881A0 = (__int64)&unk_49DC230;
  qword_4F881C0 = (__int64)nullsub_58;
  qword_4F881B8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4F88100, "stack-safety-max-iterations", 27);
  LODWORD(qword_4F88188) = 20;
  BYTE4(qword_4F88198) = 1;
  LODWORD(qword_4F88198) = 20;
  LOBYTE(dword_4F8810C) = dword_4F8810C & 0x9F | 0x20;
  sub_C53130(&qword_4F88100);
  __cxa_atexit(sub_B2B680, &qword_4F88100, &qword_4A427C0);
  qword_4F88020 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F88070 = 0x100000000LL;
  dword_4F8802C &= 0x8000u;
  word_4F88030 = 0;
  qword_4F88038 = 0;
  qword_4F88040 = 0;
  dword_4F88028 = v4;
  qword_4F88048 = 0;
  qword_4F88050 = 0;
  qword_4F88058 = 0;
  qword_4F88060 = 0;
  qword_4F88068 = (__int64)&unk_4F88078;
  qword_4F88080 = 0;
  qword_4F88088 = (__int64)&unk_4F880A0;
  qword_4F88090 = 1;
  dword_4F88098 = 0;
  byte_4F8809C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F88070;
  v7 = (unsigned int)qword_4F88070 + 1LL;
  if ( v7 > HIDWORD(qword_4F88070) )
  {
    sub_C8D5F0((char *)&unk_4F88078 - 16, &unk_4F88078, v7, 8);
    v6 = (unsigned int)qword_4F88070;
  }
  *(_QWORD *)(qword_4F88068 + 8 * v6) = v5;
  qword_4F880B0 = (__int64)&unk_49D9748;
  qword_4F88020 = (__int64)&unk_49DC090;
  LODWORD(qword_4F88070) = qword_4F88070 + 1;
  qword_4F880A8 = 0;
  qword_4F880C0 = (__int64)&unk_49DC1D0;
  qword_4F880B8 = 0;
  qword_4F880E0 = (__int64)nullsub_23;
  qword_4F880D8 = (__int64)sub_984030;
  sub_C53080(&qword_4F88020, "stack-safety-print", 18);
  LOWORD(qword_4F880B8) = 256;
  LOBYTE(qword_4F880A8) = 0;
  LOBYTE(dword_4F8802C) = dword_4F8802C & 0x9F | 0x20;
  sub_C53130(&qword_4F88020);
  __cxa_atexit(sub_984900, &qword_4F88020, &qword_4A427C0);
  qword_4F87F40 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F87F90 = 0x100000000LL;
  word_4F87F50 = 0;
  dword_4F87F4C &= 0x8000u;
  qword_4F87F58 = 0;
  qword_4F87F60 = 0;
  dword_4F87F48 = v8;
  qword_4F87F68 = 0;
  qword_4F87F70 = 0;
  qword_4F87F78 = 0;
  qword_4F87F80 = 0;
  qword_4F87F88 = (__int64)&unk_4F87F98;
  qword_4F87FA0 = 0;
  qword_4F87FA8 = (__int64)&unk_4F87FC0;
  qword_4F87FB0 = 1;
  dword_4F87FB8 = 0;
  byte_4F87FBC = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4F87F90;
  v11 = (unsigned int)qword_4F87F90 + 1LL;
  if ( v11 > HIDWORD(qword_4F87F90) )
  {
    sub_C8D5F0((char *)&unk_4F87F98 - 16, &unk_4F87F98, v11, 8);
    v10 = (unsigned int)qword_4F87F90;
  }
  *(_QWORD *)(qword_4F87F88 + 8 * v10) = v9;
  qword_4F87FD0 = (__int64)&unk_49D9748;
  qword_4F87F40 = (__int64)&unk_49DC090;
  LODWORD(qword_4F87F90) = qword_4F87F90 + 1;
  qword_4F87FC8 = 0;
  qword_4F87FE0 = (__int64)&unk_49DC1D0;
  qword_4F87FD8 = 0;
  qword_4F88000 = (__int64)nullsub_23;
  qword_4F87FF8 = (__int64)sub_984030;
  sub_C53080(&qword_4F87F40, "stack-safety-run", 16);
  LOBYTE(qword_4F87FC8) = 0;
  LOWORD(qword_4F87FD8) = 256;
  LOBYTE(dword_4F87F4C) = dword_4F87F4C & 0x9F | 0x20;
  sub_C53130(&qword_4F87F40);
  return __cxa_atexit(sub_984900, &qword_4F87F40, &qword_4A427C0);
}
