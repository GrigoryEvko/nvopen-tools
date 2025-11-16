// Function: ctor_379
// Address: 0x5191f0
//
int ctor_379()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // edx
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rbx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+8h] [rbp-58h]
  _QWORD v21[2]; // [rsp+10h] [rbp-50h] BYREF
  char v22[64]; // [rsp+20h] [rbp-40h] BYREF

  qword_4FDB280 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FDB2D0 = 0x100000000LL;
  dword_4FDB28C &= 0x8000u;
  word_4FDB290 = 0;
  qword_4FDB298 = 0;
  qword_4FDB2A0 = 0;
  dword_4FDB288 = v0;
  qword_4FDB2A8 = 0;
  qword_4FDB2B0 = 0;
  qword_4FDB2B8 = 0;
  qword_4FDB2C0 = 0;
  qword_4FDB2C8 = (__int64)&unk_4FDB2D8;
  qword_4FDB2E0 = 0;
  qword_4FDB2E8 = (__int64)&unk_4FDB300;
  qword_4FDB2F0 = 1;
  dword_4FDB2F8 = 0;
  byte_4FDB2FC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FDB2D0;
  v3 = (unsigned int)qword_4FDB2D0 + 1LL;
  if ( v3 > HIDWORD(qword_4FDB2D0) )
  {
    sub_C8D5F0((char *)&unk_4FDB2D8 - 16, &unk_4FDB2D8, v3, 8);
    v2 = (unsigned int)qword_4FDB2D0;
  }
  *(_QWORD *)(qword_4FDB2C8 + 8 * v2) = v1;
  qword_4FDB310 = (__int64)&unk_49D9748;
  qword_4FDB280 = (__int64)&unk_49DC090;
  qword_4FDB320 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FDB2D0) = qword_4FDB2D0 + 1;
  qword_4FDB340 = (__int64)nullsub_23;
  qword_4FDB308 = 0;
  qword_4FDB338 = (__int64)sub_984030;
  qword_4FDB318 = 0;
  sub_C53080(&qword_4FDB280, "callgraph-heat-colors", 21);
  LOWORD(qword_4FDB318) = 256;
  LOBYTE(qword_4FDB308) = 0;
  qword_4FDB2B0 = 30;
  LOBYTE(dword_4FDB28C) = dword_4FDB28C & 0x9F | 0x20;
  qword_4FDB2A8 = (__int64)"Show heat colors in call-graph";
  sub_C53130(&qword_4FDB280);
  __cxa_atexit(sub_984900, &qword_4FDB280, &qword_4A427C0);
  qword_4FDB1A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FDB1F0 = 0x100000000LL;
  dword_4FDB1AC &= 0x8000u;
  qword_4FDB1E8 = (__int64)&unk_4FDB1F8;
  word_4FDB1B0 = 0;
  qword_4FDB1B8 = 0;
  dword_4FDB1A8 = v4;
  qword_4FDB1C0 = 0;
  qword_4FDB1C8 = 0;
  qword_4FDB1D0 = 0;
  qword_4FDB1D8 = 0;
  qword_4FDB1E0 = 0;
  qword_4FDB200 = 0;
  qword_4FDB208 = (__int64)&unk_4FDB220;
  qword_4FDB210 = 1;
  dword_4FDB218 = 0;
  byte_4FDB21C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FDB1F0;
  if ( (unsigned __int64)(unsigned int)qword_4FDB1F0 + 1 > HIDWORD(qword_4FDB1F0) )
  {
    v19 = v5;
    sub_C8D5F0((char *)&unk_4FDB1F8 - 16, &unk_4FDB1F8, (unsigned int)qword_4FDB1F0 + 1LL, 8);
    v6 = (unsigned int)qword_4FDB1F0;
    v5 = v19;
  }
  *(_QWORD *)(qword_4FDB1E8 + 8 * v6) = v5;
  qword_4FDB230 = (__int64)&unk_49D9748;
  qword_4FDB1A0 = (__int64)&unk_49DC090;
  qword_4FDB240 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FDB1F0) = qword_4FDB1F0 + 1;
  qword_4FDB260 = (__int64)nullsub_23;
  qword_4FDB228 = 0;
  qword_4FDB258 = (__int64)sub_984030;
  qword_4FDB238 = 0;
  sub_C53080(&qword_4FDB1A0, "callgraph-show-weights", 22);
  LOWORD(qword_4FDB238) = 256;
  LOBYTE(qword_4FDB228) = 0;
  qword_4FDB1D0 = 31;
  LOBYTE(dword_4FDB1AC) = dword_4FDB1AC & 0x9F | 0x20;
  qword_4FDB1C8 = (__int64)"Show edges labeled with weights";
  sub_C53130(&qword_4FDB1A0);
  __cxa_atexit(sub_984900, &qword_4FDB1A0, &qword_4A427C0);
  qword_4FDB0C0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FDB110 = 0x100000000LL;
  dword_4FDB0CC &= 0x8000u;
  qword_4FDB108 = (__int64)&unk_4FDB118;
  word_4FDB0D0 = 0;
  qword_4FDB0D8 = 0;
  dword_4FDB0C8 = v7;
  qword_4FDB0E0 = 0;
  qword_4FDB0E8 = 0;
  qword_4FDB0F0 = 0;
  qword_4FDB0F8 = 0;
  qword_4FDB100 = 0;
  qword_4FDB120 = 0;
  qword_4FDB128 = (__int64)&unk_4FDB140;
  qword_4FDB130 = 1;
  dword_4FDB138 = 0;
  byte_4FDB13C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4FDB110;
  if ( (unsigned __int64)(unsigned int)qword_4FDB110 + 1 > HIDWORD(qword_4FDB110) )
  {
    v20 = v8;
    sub_C8D5F0((char *)&unk_4FDB118 - 16, &unk_4FDB118, (unsigned int)qword_4FDB110 + 1LL, 8);
    v9 = (unsigned int)qword_4FDB110;
    v8 = v20;
  }
  *(_QWORD *)(qword_4FDB108 + 8 * v9) = v8;
  qword_4FDB150 = (__int64)&unk_49D9748;
  qword_4FDB0C0 = (__int64)&unk_49DC090;
  qword_4FDB160 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FDB110) = qword_4FDB110 + 1;
  qword_4FDB180 = (__int64)nullsub_23;
  qword_4FDB148 = 0;
  qword_4FDB178 = (__int64)sub_984030;
  qword_4FDB158 = 0;
  sub_C53080(&qword_4FDB0C0, "callgraph-multigraph", 20);
  LOWORD(qword_4FDB158) = 256;
  LOBYTE(qword_4FDB148) = 0;
  qword_4FDB0F0 = 51;
  LOBYTE(dword_4FDB0CC) = dword_4FDB0CC & 0x9F | 0x20;
  qword_4FDB0E8 = (__int64)"Show call-multigraph (do not remove parallel edges)";
  sub_C53130(&qword_4FDB0C0);
  __cxa_atexit(sub_984900, &qword_4FDB0C0, &qword_4A427C0);
  qword_4FDAFC0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FDB010 = 0x100000000LL;
  word_4FDAFD0 = 0;
  dword_4FDAFCC &= 0x8000u;
  qword_4FDAFD8 = 0;
  qword_4FDAFE0 = 0;
  dword_4FDAFC8 = v10;
  qword_4FDAFE8 = 0;
  qword_4FDAFF0 = 0;
  qword_4FDAFF8 = 0;
  qword_4FDB000 = 0;
  qword_4FDB008 = (__int64)&unk_4FDB018;
  qword_4FDB020 = 0;
  qword_4FDB028 = (__int64)&unk_4FDB040;
  qword_4FDB030 = 1;
  dword_4FDB038 = 0;
  byte_4FDB03C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_4FDB010;
  v13 = (unsigned int)qword_4FDB010 + 1LL;
  if ( v13 > HIDWORD(qword_4FDB010) )
  {
    sub_C8D5F0((char *)&unk_4FDB018 - 16, &unk_4FDB018, v13, 8);
    v12 = (unsigned int)qword_4FDB010;
  }
  *(_QWORD *)(qword_4FDB008 + 8 * v12) = v11;
  qword_4FDB048 = (__int64)&byte_4FDB058;
  qword_4FDB070 = (__int64)&byte_4FDB080;
  qword_4FDB068 = (__int64)&unk_49DC130;
  qword_4FDAFC0 = (__int64)&unk_49DC010;
  LODWORD(qword_4FDB010) = qword_4FDB010 + 1;
  qword_4FDB050 = 0;
  qword_4FDB098 = (__int64)&unk_49DC350;
  byte_4FDB058 = 0;
  qword_4FDB0B8 = (__int64)nullsub_92;
  qword_4FDB078 = 0;
  qword_4FDB0B0 = (__int64)sub_BC4D70;
  byte_4FDB080 = 0;
  byte_4FDB090 = 0;
  sub_C53080(&qword_4FDAFC0, "callgraph-dot-filename-prefix", 29);
  qword_4FDAFF0 = 49;
  LOBYTE(dword_4FDAFCC) = dword_4FDAFCC & 0x9F | 0x20;
  qword_4FDAFE8 = (__int64)"The prefix used for the CallGraph dot file names.";
  sub_C53130(&qword_4FDAFC0);
  __cxa_atexit(sub_BC5A40, &qword_4FDAFC0, &qword_4A427C0);
  qword_4FDAEC0 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FDAECC &= 0x8000u;
  word_4FDAED0 = 0;
  qword_4FDAF10 = 0x100000000LL;
  qword_4FDAED8 = 0;
  qword_4FDAEE0 = 0;
  qword_4FDAEE8 = 0;
  dword_4FDAEC8 = v14;
  qword_4FDAEF0 = 0;
  qword_4FDAEF8 = 0;
  qword_4FDAF00 = 0;
  qword_4FDAF08 = (__int64)&unk_4FDAF18;
  qword_4FDAF20 = 0;
  qword_4FDAF28 = (__int64)&unk_4FDAF40;
  qword_4FDAF30 = 1;
  dword_4FDAF38 = 0;
  byte_4FDAF3C = 1;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_4FDAF10;
  v17 = (unsigned int)qword_4FDAF10 + 1LL;
  if ( v17 > HIDWORD(qword_4FDAF10) )
  {
    sub_C8D5F0((char *)&unk_4FDAF18 - 16, &unk_4FDAF18, v17, 8);
    v16 = (unsigned int)qword_4FDAF10;
  }
  *(_QWORD *)(qword_4FDAF08 + 8 * v16) = v15;
  qword_4FDAF48 = (__int64)&byte_4FDAF58;
  qword_4FDAF70 = (__int64)&byte_4FDAF80;
  qword_4FDAF68 = (__int64)&unk_49DC130;
  qword_4FDAEC0 = (__int64)&unk_49DC010;
  LODWORD(qword_4FDAF10) = qword_4FDAF10 + 1;
  qword_4FDAF50 = 0;
  qword_4FDAF98 = (__int64)&unk_49DC350;
  byte_4FDAF58 = 0;
  qword_4FDAFB8 = (__int64)nullsub_92;
  qword_4FDAF78 = 0;
  qword_4FDAFB0 = (__int64)sub_BC4D70;
  byte_4FDAF80 = 0;
  byte_4FDAF90 = 0;
  sub_C53080(&qword_4FDAEC0, "callgraph-dot-filename-basename", 31);
  strcpy(v22, "callgraph");
  v21[0] = v22;
  v21[1] = 9;
  sub_2240AE0(&qword_4FDAF48, v21);
  byte_4FDAF90 = 1;
  sub_2240AE0(&qword_4FDAF70, v21);
  if ( (char *)v21[0] != v22 )
    j_j___libc_free_0(v21[0], *(_QWORD *)v22 + 1LL);
  qword_4FDAEF0 = 52;
  LOBYTE(dword_4FDAECC) = dword_4FDAECC & 0x9F | 0x20;
  qword_4FDAEE8 = (__int64)"The base name used for the CallGraph dot file names.";
  sub_C53130(&qword_4FDAEC0);
  return __cxa_atexit(sub_BC5A40, &qword_4FDAEC0, &qword_4A427C0);
}
