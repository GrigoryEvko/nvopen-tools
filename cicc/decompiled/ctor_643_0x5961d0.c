// Function: ctor_643
// Address: 0x5961d0
//
int __fastcall ctor_643(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
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
  __int64 v20; // [rsp+8h] [rbp-58h]
  int v21; // [rsp+14h] [rbp-4Ch] BYREF
  const char *v22; // [rsp+18h] [rbp-48h] BYREF
  _QWORD v23[8]; // [rsp+20h] [rbp-40h] BYREF

  qword_5035C80 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5035CD0 = 0x100000000LL;
  word_5035C90 = 0;
  dword_5035C8C &= 0x8000u;
  qword_5035C98 = 0;
  qword_5035CA0 = 0;
  dword_5035C88 = v4;
  qword_5035CA8 = 0;
  qword_5035CB0 = 0;
  qword_5035CB8 = 0;
  qword_5035CC0 = 0;
  qword_5035CC8 = (__int64)&unk_5035CD8;
  qword_5035CE0 = 0;
  qword_5035CE8 = (__int64)&unk_5035D00;
  qword_5035CF0 = 1;
  dword_5035CF8 = 0;
  byte_5035CFC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5035CD0;
  v7 = (unsigned int)qword_5035CD0 + 1LL;
  if ( v7 > HIDWORD(qword_5035CD0) )
  {
    sub_C8D5F0((char *)&unk_5035CD8 - 16, &unk_5035CD8, v7, 8);
    v6 = (unsigned int)qword_5035CD0;
  }
  *(_QWORD *)(qword_5035CC8 + 8 * v6) = v5;
  qword_5035D10 = (__int64)&unk_49D9728;
  qword_5035C80 = (__int64)&unk_49DBF10;
  LODWORD(qword_5035CD0) = qword_5035CD0 + 1;
  qword_5035D08 = 0;
  qword_5035D20 = (__int64)&unk_49DC290;
  qword_5035D18 = 0;
  qword_5035D40 = (__int64)nullsub_24;
  qword_5035D38 = (__int64)sub_984050;
  sub_C53080(&qword_5035C80, "sbvec-seed-bundle-size-limit", 28);
  LODWORD(qword_5035D08) = 32;
  BYTE4(qword_5035D18) = 1;
  LODWORD(qword_5035D18) = 32;
  qword_5035CB0 = 58;
  LOBYTE(dword_5035C8C) = dword_5035C8C & 0x9F | 0x20;
  qword_5035CA8 = (__int64)"Limit the size of the seed bundle to cap compilation time.";
  sub_C53130(&qword_5035C80);
  __cxa_atexit(sub_984970, &qword_5035C80, &qword_4A427C0);
  v23[1] = 90;
  v23[0] = "Collect these seeds. Use empty for none or a comma-separated list of 'loads' and 'stores'.";
  v22 = "loads,stores";
  v21 = 1;
  qword_5035B80 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_5035C80, v8, v9), 1u);
  byte_5035BFC = 1;
  qword_5035BD0 = 0x100000000LL;
  dword_5035B8C &= 0x8000u;
  qword_5035B98 = 0;
  qword_5035BA0 = 0;
  qword_5035BA8 = 0;
  dword_5035B88 = v10;
  word_5035B90 = 0;
  qword_5035BB0 = 0;
  qword_5035BB8 = 0;
  qword_5035BC0 = 0;
  qword_5035BC8 = (__int64)&unk_5035BD8;
  qword_5035BE0 = 0;
  qword_5035BE8 = (__int64)&unk_5035C00;
  qword_5035BF0 = 1;
  dword_5035BF8 = 0;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5035BD0;
  if ( (unsigned __int64)(unsigned int)qword_5035BD0 + 1 > HIDWORD(qword_5035BD0) )
  {
    v20 = v11;
    sub_C8D5F0((char *)&unk_5035BD8 - 16, &unk_5035BD8, (unsigned int)qword_5035BD0 + 1LL, 8);
    v12 = (unsigned int)qword_5035BD0;
    v11 = v20;
  }
  *(_QWORD *)(qword_5035BC8 + 8 * v12) = v11;
  qword_5035C08 = (__int64)&byte_5035C18;
  qword_5035C30 = (__int64)&byte_5035C40;
  LODWORD(qword_5035BD0) = qword_5035BD0 + 1;
  qword_5035C10 = 0;
  qword_5035C28 = (__int64)&unk_49DC130;
  byte_5035C18 = 0;
  byte_5035C40 = 0;
  qword_5035B80 = (__int64)&unk_49DC010;
  qword_5035C38 = 0;
  byte_5035C50 = 0;
  qword_5035C58 = (__int64)&unk_49DC350;
  qword_5035C78 = (__int64)nullsub_92;
  qword_5035C70 = (__int64)sub_BC4D70;
  sub_31C3330(&qword_5035B80, "sbvec-collect-seeds", &v22, &v21, v23);
  sub_C53130(&qword_5035B80);
  __cxa_atexit(sub_BC5A40, &qword_5035B80, &qword_4A427C0);
  qword_5035AA0 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_BC5A40, &qword_5035B80, v13, v14), 1u);
  dword_5035AAC &= 0x8000u;
  word_5035AB0 = 0;
  qword_5035AF0 = 0x100000000LL;
  qword_5035AB8 = 0;
  qword_5035AC0 = 0;
  qword_5035AC8 = 0;
  dword_5035AA8 = v15;
  qword_5035AD0 = 0;
  qword_5035AD8 = 0;
  qword_5035AE0 = 0;
  qword_5035AE8 = (__int64)&unk_5035AF8;
  qword_5035B00 = 0;
  qword_5035B08 = (__int64)&unk_5035B20;
  qword_5035B10 = 1;
  dword_5035B18 = 0;
  byte_5035B1C = 1;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_5035AF0;
  v18 = (unsigned int)qword_5035AF0 + 1LL;
  if ( v18 > HIDWORD(qword_5035AF0) )
  {
    sub_C8D5F0((char *)&unk_5035AF8 - 16, &unk_5035AF8, v18, 8);
    v17 = (unsigned int)qword_5035AF0;
  }
  *(_QWORD *)(qword_5035AE8 + 8 * v17) = v16;
  qword_5035B30 = (__int64)&unk_49D9728;
  qword_5035AA0 = (__int64)&unk_49DBF10;
  LODWORD(qword_5035AF0) = qword_5035AF0 + 1;
  qword_5035B28 = 0;
  qword_5035B40 = (__int64)&unk_49DC290;
  qword_5035B38 = 0;
  qword_5035B60 = (__int64)nullsub_24;
  qword_5035B58 = (__int64)sub_984050;
  sub_C53080(&qword_5035AA0, "sbvec-seed-groups-limit", 23);
  LODWORD(qword_5035B28) = 256;
  BYTE4(qword_5035B38) = 1;
  LODWORD(qword_5035B38) = 256;
  qword_5035AD0 = 75;
  LOBYTE(dword_5035AAC) = dword_5035AAC & 0x9F | 0x20;
  qword_5035AC8 = (__int64)"Limit the number of collected seeds groups in a BB to cap compilation time.";
  sub_C53130(&qword_5035AA0);
  return __cxa_atexit(sub_984970, &qword_5035AA0, &qword_4A427C0);
}
