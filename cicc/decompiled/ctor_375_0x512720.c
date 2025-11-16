// Function: ctor_375
// Address: 0x512720
//
int ctor_375()
{
  int v0; // edx
  __int64 v1; // r14
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r13
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  char v13; // [rsp+7h] [rbp-59h] BYREF
  char *v14; // [rsp+8h] [rbp-58h] BYREF
  const char *v15; // [rsp+10h] [rbp-50h] BYREF
  __int64 v16; // [rsp+18h] [rbp-48h]
  _QWORD v17[8]; // [rsp+20h] [rbp-40h] BYREF

  v15 = "profgen";
  v14 = &v13;
  v16 = 7;
  v13 = 0;
  sub_2262040(&unk_4FD6F20, "profgen", &v14, &v15);
  __cxa_atexit(sub_984900, &unk_4FD6F20, &qword_4A427C0);
  v14 = &v13;
  v16 = 7;
  v15 = "profuse";
  v13 = 0;
  sub_2262040(&unk_4FD6E40, "profuse", &v14, &v15);
  __cxa_atexit(sub_984900, &unk_4FD6E40, &qword_4A427C0);
  qword_4FD6D40 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FD6D90 = 0x100000000LL;
  dword_4FD6D4C &= 0x8000u;
  word_4FD6D50 = 0;
  qword_4FD6D58 = 0;
  qword_4FD6D60 = 0;
  dword_4FD6D48 = v0;
  qword_4FD6D68 = 0;
  qword_4FD6D70 = 0;
  qword_4FD6D78 = 0;
  qword_4FD6D80 = 0;
  qword_4FD6D88 = (__int64)&unk_4FD6D98;
  qword_4FD6DA0 = 0;
  qword_4FD6DA8 = (__int64)&unk_4FD6DC0;
  qword_4FD6DB0 = 1;
  dword_4FD6DB8 = 0;
  byte_4FD6DBC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FD6D90;
  v3 = (unsigned int)qword_4FD6D90 + 1LL;
  if ( v3 > HIDWORD(qword_4FD6D90) )
  {
    sub_C8D5F0((char *)&unk_4FD6D98 - 16, &unk_4FD6D98, v3, 8);
    v2 = (unsigned int)qword_4FD6D90;
  }
  *(_QWORD *)(qword_4FD6D88 + 8 * v2) = v1;
  qword_4FD6DC8 = (__int64)&byte_4FD6DD8;
  qword_4FD6DF0 = (__int64)&byte_4FD6E00;
  LODWORD(qword_4FD6D90) = qword_4FD6D90 + 1;
  qword_4FD6DD0 = 0;
  qword_4FD6DE8 = (__int64)&unk_49DC130;
  byte_4FD6DD8 = 0;
  byte_4FD6E00 = 0;
  qword_4FD6D40 = (__int64)&unk_49DC010;
  qword_4FD6DF8 = 0;
  byte_4FD6E10 = 0;
  qword_4FD6E18 = (__int64)&unk_49DC350;
  qword_4FD6E38 = (__int64)nullsub_92;
  qword_4FD6E30 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4FD6D40, "proffile", 8);
  v15 = (const char *)v17;
  v16 = 0;
  LOBYTE(v17[0]) = 0;
  sub_2240AE0(&qword_4FD6DC8, &v15);
  byte_4FD6E10 = 1;
  sub_2240AE0(&qword_4FD6DF0, &v15);
  if ( v15 != (const char *)v17 )
    j_j___libc_free_0(v15, v17[0] + 1LL);
  qword_4FD6D70 = 47;
  qword_4FD6D68 = (__int64)"Name for file that contains profile information";
  sub_C53130(&qword_4FD6D40);
  __cxa_atexit(sub_BC5A40, &qword_4FD6D40, &qword_4A427C0);
  qword_4FD6C60 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FD6CB0 = 0x100000000LL;
  dword_4FD6C6C &= 0x8000u;
  word_4FD6C70 = 0;
  qword_4FD6C78 = 0;
  qword_4FD6C80 = 0;
  dword_4FD6C68 = v4;
  qword_4FD6C88 = 0;
  qword_4FD6C90 = 0;
  qword_4FD6C98 = 0;
  qword_4FD6CA0 = 0;
  qword_4FD6CA8 = (__int64)&unk_4FD6CB8;
  qword_4FD6CC0 = 0;
  qword_4FD6CC8 = (__int64)&unk_4FD6CE0;
  qword_4FD6CD0 = 1;
  dword_4FD6CD8 = 0;
  byte_4FD6CDC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FD6CB0;
  v7 = (unsigned int)qword_4FD6CB0 + 1LL;
  if ( v7 > HIDWORD(qword_4FD6CB0) )
  {
    sub_C8D5F0((char *)&unk_4FD6CB8 - 16, &unk_4FD6CB8, v7, 8);
    v6 = (unsigned int)qword_4FD6CB0;
  }
  *(_QWORD *)(qword_4FD6CA8 + 8 * v6) = v5;
  qword_4FD6CF0 = (__int64)&unk_49D9748;
  qword_4FD6C60 = (__int64)&unk_49DC090;
  LODWORD(qword_4FD6CB0) = qword_4FD6CB0 + 1;
  qword_4FD6CE8 = 0;
  qword_4FD6D00 = (__int64)&unk_49DC1D0;
  qword_4FD6CF8 = 0;
  qword_4FD6D20 = (__int64)nullsub_23;
  qword_4FD6D18 = (__int64)sub_984030;
  sub_C53080(&qword_4FD6C60, "sep-comp", 8);
  LOWORD(qword_4FD6CF8) = 256;
  qword_4FD6C88 = (__int64)"The file is being compiled under separate compilation mode.";
  LOBYTE(qword_4FD6CE8) = 0;
  qword_4FD6C90 = 59;
  sub_C53130(&qword_4FD6C60);
  __cxa_atexit(sub_984900, &qword_4FD6C60, &qword_4A427C0);
  qword_4FD6B80 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FD6BD0 = 0x100000000LL;
  word_4FD6B90 = 0;
  dword_4FD6B8C &= 0x8000u;
  qword_4FD6B98 = 0;
  qword_4FD6BA0 = 0;
  dword_4FD6B88 = v8;
  qword_4FD6BA8 = 0;
  qword_4FD6BB0 = 0;
  qword_4FD6BB8 = 0;
  qword_4FD6BC0 = 0;
  qword_4FD6BC8 = (__int64)&unk_4FD6BD8;
  qword_4FD6BE0 = 0;
  qword_4FD6BE8 = (__int64)&unk_4FD6C00;
  qword_4FD6BF0 = 1;
  dword_4FD6BF8 = 0;
  byte_4FD6BFC = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_4FD6BD0;
  v11 = (unsigned int)qword_4FD6BD0 + 1LL;
  if ( v11 > HIDWORD(qword_4FD6BD0) )
  {
    sub_C8D5F0((char *)&unk_4FD6BD8 - 16, &unk_4FD6BD8, v11, 8);
    v10 = (unsigned int)qword_4FD6BD0;
  }
  *(_QWORD *)(qword_4FD6BC8 + 8 * v10) = v9;
  qword_4FD6C10 = (__int64)&unk_49D9748;
  qword_4FD6B80 = (__int64)&unk_49DC090;
  LODWORD(qword_4FD6BD0) = qword_4FD6BD0 + 1;
  qword_4FD6C08 = 0;
  qword_4FD6C20 = (__int64)&unk_49DC1D0;
  qword_4FD6C18 = 0;
  qword_4FD6C40 = (__int64)nullsub_23;
  qword_4FD6C38 = (__int64)sub_984030;
  sub_C53080(&qword_4FD6B80, "merge-pgo-atomics", 17);
  LOBYTE(qword_4FD6C08) = 1;
  LOWORD(qword_4FD6C18) = 257;
  qword_4FD6BA8 = (__int64)"merge pgo atomic counters";
  qword_4FD6BB0 = 25;
  sub_C53130(&qword_4FD6B80);
  return __cxa_atexit(sub_984900, &qword_4FD6B80, &qword_4A427C0);
}
