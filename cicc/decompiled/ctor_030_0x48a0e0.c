// Function: ctor_030
// Address: 0x48a0e0
//
int ctor_030()
{
  int v0; // edx
  __int64 v1; // r14
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-98h]
  int v13; // [rsp+1Ch] [rbp-84h] BYREF
  const char *v14; // [rsp+20h] [rbp-80h] BYREF
  __int64 v15; // [rsp+28h] [rbp-78h]
  _BYTE v16[32]; // [rsp+30h] [rbp-70h] BYREF
  _BYTE v17[16]; // [rsp+50h] [rbp-50h] BYREF
  __int64 (__fastcall *v18)(); // [rsp+60h] [rbp-40h]
  __int64 (__fastcall *v19)(); // [rsp+68h] [rbp-38h]

  v14 = "Disable pass number (in execution order)";
  v19 = sub_BB8310;
  v18 = sub_BB74C0;
  v15 = 40;
  sub_BB8070(v16, v17);
  v13 = 0;
  qword_4F82160 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F821B0 = 0x100000000LL;
  dword_4F8216C &= 0x8000u;
  word_4F82170 = 0;
  qword_4F82178 = 0;
  qword_4F82180 = 0;
  dword_4F82168 = v0;
  qword_4F82188 = 0;
  qword_4F82190 = 0;
  qword_4F82198 = 0;
  qword_4F821A0 = 0;
  qword_4F821A8 = (__int64)&unk_4F821B8;
  qword_4F821C0 = 0;
  qword_4F821C8 = (__int64)&unk_4F821E0;
  qword_4F821D0 = 1;
  dword_4F821D8 = 0;
  byte_4F821DC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F821B0;
  v3 = (unsigned int)qword_4F821B0 + 1LL;
  if ( v3 > HIDWORD(qword_4F821B0) )
  {
    sub_C8D5F0((char *)&unk_4F821B8 - 16, &unk_4F821B8, v3, 8);
    v2 = (unsigned int)qword_4F821B0;
  }
  *(_QWORD *)(qword_4F821A8 + 8 * v2) = v1;
  LODWORD(qword_4F821B0) = qword_4F821B0 + 1;
  qword_4F821E8 = 0;
  qword_4F821F0 = (__int64)&unk_49DA090;
  qword_4F821F8 = 0;
  qword_4F82160 = (__int64)&unk_49DBF90;
  qword_4F82200 = (__int64)&unk_49DC230;
  qword_4F82220 = (__int64)nullsub_58;
  qword_4F82218 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4F82160, "opt-bisect-skip", 15);
  LOBYTE(dword_4F8216C) = dword_4F8216C & 0x9F | 0x20;
  sub_BB7D00(&qword_4F82160, &unk_3F55F70, &v13, v16, &v14);
  sub_C53130(&qword_4F82160);
  sub_A17130(v16);
  sub_A17130(v17);
  __cxa_atexit(sub_B2B680, &qword_4F82160, &qword_4A427C0);
  v14 = "Maximum optimization to perform";
  v19 = sub_BB77E0;
  v18 = sub_BB74D0;
  v15 = 31;
  sub_BB8070(v16, v17);
  v13 = 0;
  qword_4F82080 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F820D0 = 0x100000000LL;
  dword_4F8208C &= 0x8000u;
  word_4F82090 = 0;
  qword_4F820C8 = (__int64)&unk_4F820D8;
  qword_4F82098 = 0;
  dword_4F82088 = v4;
  qword_4F820A0 = 0;
  qword_4F820A8 = 0;
  qword_4F820B0 = 0;
  qword_4F820B8 = 0;
  qword_4F820C0 = 0;
  qword_4F820E0 = 0;
  qword_4F820E8 = (__int64)&unk_4F82100;
  qword_4F820F0 = 1;
  dword_4F820F8 = 0;
  byte_4F820FC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F820D0;
  if ( (unsigned __int64)(unsigned int)qword_4F820D0 + 1 > HIDWORD(qword_4F820D0) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_4F820D8 - 16, &unk_4F820D8, (unsigned int)qword_4F820D0 + 1LL, 8);
    v6 = (unsigned int)qword_4F820D0;
    v5 = v12;
  }
  *(_QWORD *)(qword_4F820C8 + 8 * v6) = v5;
  LODWORD(qword_4F820D0) = qword_4F820D0 + 1;
  qword_4F82108 = 0;
  qword_4F82110 = (__int64)&unk_49DA090;
  qword_4F82118 = 0;
  qword_4F82080 = (__int64)&unk_49DBF90;
  qword_4F82120 = (__int64)&unk_49DC230;
  qword_4F82140 = (__int64)nullsub_58;
  qword_4F82138 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4F82080, "opt-bisect-limit", 16);
  LOBYTE(dword_4F8208C) = dword_4F8208C & 0x9F | 0x20;
  sub_BB7D00(&qword_4F82080, &unk_3F55F70, &v13, v16, &v14);
  sub_C53130(&qword_4F82080);
  sub_A17130(v16);
  sub_A17130(v17);
  __cxa_atexit(sub_B2B680, &qword_4F82080, &qword_4A427C0);
  qword_4F81FA0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F8201C = 1;
  qword_4F81FF0 = 0x100000000LL;
  dword_4F81FAC &= 0x8000u;
  qword_4F81FB8 = 0;
  qword_4F81FC0 = 0;
  qword_4F81FC8 = 0;
  dword_4F81FA8 = v7;
  word_4F81FB0 = 0;
  qword_4F81FD0 = 0;
  qword_4F81FD8 = 0;
  qword_4F81FE0 = 0;
  qword_4F81FE8 = (__int64)&unk_4F81FF8;
  qword_4F82000 = 0;
  qword_4F82008 = (__int64)&unk_4F82020;
  qword_4F82010 = 1;
  dword_4F82018 = 0;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4F81FF0;
  v10 = (unsigned int)qword_4F81FF0 + 1LL;
  if ( v10 > HIDWORD(qword_4F81FF0) )
  {
    sub_C8D5F0((char *)&unk_4F81FF8 - 16, &unk_4F81FF8, v10, 8);
    v9 = (unsigned int)qword_4F81FF0;
  }
  *(_QWORD *)(qword_4F81FE8 + 8 * v9) = v8;
  LODWORD(qword_4F81FF0) = qword_4F81FF0 + 1;
  qword_4F82028 = 0;
  qword_4F82030 = (__int64)&unk_49D9748;
  qword_4F82038 = 0;
  qword_4F81FA0 = (__int64)&unk_49DC090;
  qword_4F82040 = (__int64)&unk_49DC1D0;
  qword_4F82060 = (__int64)nullsub_23;
  qword_4F82058 = (__int64)sub_984030;
  sub_C53080(&qword_4F81FA0, "opt-bisect-verbose", 18);
  qword_4F81FD0 = 48;
  qword_4F81FC8 = (__int64)"Show verbose output when opt-bisect-limit is set";
  LOWORD(qword_4F82038) = 257;
  LOBYTE(qword_4F82028) = 1;
  LOBYTE(dword_4F81FAC) = dword_4F81FAC & 0x98 | 0x20;
  sub_C53130(&qword_4F81FA0);
  return __cxa_atexit(sub_984900, &qword_4F81FA0, &qword_4A427C0);
}
