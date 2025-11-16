// Function: ctor_682
// Address: 0x5a5540
//
int ctor_682()
{
  __int64 v0; // rdx
  __int64 v1; // rcx
  int v2; // edx
  __int64 v3; // r15
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rcx
  int v8; // edx
  __int64 v9; // r14
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned int v12; // eax
  char v14; // [rsp+7h] [rbp-79h] BYREF
  __int64 v15; // [rsp+8h] [rbp-78h] BYREF
  _QWORD v16[4]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v17; // [rsp+30h] [rbp-50h] BYREF
  __int64 v18; // [rsp+38h] [rbp-48h]
  __int64 v19; // [rsp+40h] [rbp-40h]

  v17 = 1;
  v18 = 300;
  sub_3595F90(&unk_503FA40, &v17, 2, v16);
  __cxa_atexit(sub_3593040, &unk_503FA40, &qword_4A427C0);
  v17 = 1;
  v18 = 33;
  v19 = 300;
  sub_3595F90(&unk_503FA20, &v17, 3, v16);
  __cxa_atexit(sub_3593040, &unk_503FA20, &qword_4A427C0);
  v17 = 1;
  v18 = 100;
  sub_3595F90(&unk_503FA00, &v17, 2, v16);
  __cxa_atexit(sub_3593040, &unk_503FA00, &qword_4A427C0);
  qword_503F900 = (__int64)&unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_3593040, &unk_503FA00, v0, v1), 1u);
  byte_503F97C = 1;
  qword_503F950 = 0x100000000LL;
  dword_503F90C &= 0x8000u;
  qword_503F918 = 0;
  qword_503F920 = 0;
  qword_503F928 = 0;
  dword_503F908 = v2;
  word_503F910 = 0;
  qword_503F930 = 0;
  qword_503F938 = 0;
  qword_503F940 = 0;
  qword_503F948 = (__int64)&unk_503F958;
  qword_503F960 = 0;
  qword_503F968 = (__int64)&unk_503F980;
  qword_503F970 = 1;
  dword_503F978 = 0;
  v3 = sub_C57470();
  v4 = (unsigned int)qword_503F950;
  v5 = (unsigned int)qword_503F950 + 1LL;
  if ( v5 > HIDWORD(qword_503F950) )
  {
    sub_C8D5F0((char *)&unk_503F958 - 16, &unk_503F958, v5, 8);
    v4 = (unsigned int)qword_503F950;
  }
  *(_QWORD *)(qword_503F948 + 8 * v4) = v3;
  qword_503F988 = (__int64)&byte_503F998;
  qword_503F9B0 = (__int64)&byte_503F9C0;
  LODWORD(qword_503F950) = qword_503F950 + 1;
  qword_503F990 = 0;
  qword_503F9A8 = (__int64)&unk_49DC130;
  byte_503F998 = 0;
  byte_503F9C0 = 0;
  qword_503F900 = (__int64)&unk_49DC010;
  qword_503F9B8 = 0;
  byte_503F9D0 = 0;
  qword_503F9D8 = (__int64)&unk_49DC350;
  qword_503F9F8 = (__int64)nullsub_92;
  qword_503F9F0 = (__int64)sub_BC4D70;
  sub_C53080(&qword_503F900, "regalloc-evict-interactive-channel-base", 39);
  qword_503F930 = 209;
  LOBYTE(dword_503F90C) = dword_503F90C & 0x9F | 0x20;
  qword_503F928 = (__int64)"Base file path for the interactive mode. The incoming filename should have the name <regalloc"
                           "-evict-interactive-channel-base>.in, while the outgoing name should be <regalloc-evict-intera"
                           "ctive-channel-base>.out";
  sub_C53130(&qword_503F900);
  __cxa_atexit(sub_BC5A40, &qword_503F900, &qword_4A427C0);
  qword_503F820 = (__int64)&unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_BC5A40, &qword_503F900, v6, v7), 1u);
  dword_503F82C &= 0x8000u;
  word_503F830 = 0;
  qword_503F870 = 0x100000000LL;
  qword_503F838 = 0;
  qword_503F840 = 0;
  qword_503F848 = 0;
  dword_503F828 = v8;
  qword_503F850 = 0;
  qword_503F858 = 0;
  qword_503F860 = 0;
  qword_503F868 = (__int64)&unk_503F878;
  qword_503F880 = 0;
  qword_503F888 = (__int64)&unk_503F8A0;
  qword_503F890 = 1;
  dword_503F898 = 0;
  byte_503F89C = 1;
  v9 = sub_C57470();
  v10 = (unsigned int)qword_503F870;
  v11 = (unsigned int)qword_503F870 + 1LL;
  if ( v11 > HIDWORD(qword_503F870) )
  {
    sub_C8D5F0((char *)&unk_503F878 - 16, &unk_503F878, v11, 8);
    v10 = (unsigned int)qword_503F870;
  }
  *(_QWORD *)(qword_503F868 + 8 * v10) = v9;
  LODWORD(qword_503F870) = qword_503F870 + 1;
  qword_503F8A8 = 0;
  qword_503F8B0 = (__int64)&unk_49D9728;
  qword_503F8B8 = 0;
  qword_503F820 = (__int64)&unk_49DBF10;
  qword_503F8C0 = (__int64)&unk_49DC290;
  qword_503F8E0 = (__int64)nullsub_24;
  qword_503F8D8 = (__int64)sub_984050;
  sub_C53080(&qword_503F820, "mlregalloc-max-eviction-count", 29);
  qword_503F850 = 95;
  LODWORD(qword_503F8A8) = 100;
  BYTE4(qword_503F8B8) = 1;
  LODWORD(qword_503F8B8) = 100;
  LOBYTE(dword_503F82C) = dword_503F82C & 0x9F | 0x20;
  qword_503F848 = (__int64)"The maximum number of times a live range can be evicted before preventing it from being evicted";
  sub_C53130(&qword_503F820);
  __cxa_atexit(sub_984970, &qword_503F820, &qword_4A427C0);
  v17 = 1;
  v18 = 33;
  sub_3595F90(&qword_503F7F0, &v17, 2, v16);
  __cxa_atexit(sub_3593040, &qword_503F7F0, &qword_4A427C0);
  v15 = 1;
  sub_3595F90(v16, &v15, 1, &v14);
  sub_35929F0(&v17, "index_to_evict");
  v12 = sub_310D010();
  sub_310F6F0(&unk_503F7A0, &v17, 0, v12, 8, v16);
  sub_2240A30(&v17);
  if ( v16[0] )
    j_j___libc_free_0(v16[0], v16[2] - v16[0]);
  return __cxa_atexit(sub_30FB2C0, &unk_503F7A0, &qword_4A427C0);
}
