// Function: ctor_586
// Address: 0x57a500
//
int __fastcall ctor_586(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
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
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rcx
  int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rcx
  int v30; // edx
  __int64 v31; // rbx
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v35; // [rsp+8h] [rbp-58h]
  __int64 v36; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+8h] [rbp-58h]
  __int64 v38; // [rsp+8h] [rbp-58h]
  int v39; // [rsp+10h] [rbp-50h] BYREF
  int v40; // [rsp+14h] [rbp-4Ch] BYREF
  int *v41; // [rsp+18h] [rbp-48h] BYREF
  const char *v42; // [rsp+20h] [rbp-40h] BYREF
  __int64 v43; // [rsp+28h] [rbp-38h]

  qword_5024DA0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5024DF0 = 0x100000000LL;
  dword_5024DAC &= 0x8000u;
  word_5024DB0 = 0;
  qword_5024DB8 = 0;
  qword_5024DC0 = 0;
  dword_5024DA8 = v4;
  qword_5024DC8 = 0;
  qword_5024DD0 = 0;
  qword_5024DD8 = 0;
  qword_5024DE0 = 0;
  qword_5024DE8 = (__int64)&unk_5024DF8;
  qword_5024E00 = 0;
  qword_5024E08 = (__int64)&unk_5024E20;
  qword_5024E10 = 1;
  dword_5024E18 = 0;
  byte_5024E1C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5024DF0;
  v7 = (unsigned int)qword_5024DF0 + 1LL;
  if ( v7 > HIDWORD(qword_5024DF0) )
  {
    sub_C8D5F0((char *)&unk_5024DF8 - 16, &unk_5024DF8, v7, 8);
    v6 = (unsigned int)qword_5024DF0;
  }
  *(_QWORD *)(qword_5024DE8 + 8 * v6) = v5;
  qword_5024E30 = (__int64)&unk_49D9748;
  qword_5024DA0 = (__int64)&unk_49DC090;
  qword_5024E40 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5024DF0) = qword_5024DF0 + 1;
  qword_5024E60 = (__int64)nullsub_23;
  qword_5024E28 = 0;
  qword_5024E58 = (__int64)sub_984030;
  qword_5024E38 = 0;
  sub_C53080(&qword_5024DA0, "join-liveintervals", 18);
  qword_5024DC8 = (__int64)"Coalesce copies (default=true)";
  LOWORD(qword_5024E38) = 257;
  LOBYTE(qword_5024E28) = 1;
  qword_5024DD0 = 30;
  LOBYTE(dword_5024DAC) = dword_5024DAC & 0x9F | 0x20;
  sub_C53130(&qword_5024DA0);
  __cxa_atexit(sub_984900, &qword_5024DA0, &qword_4A427C0);
  qword_5024CC0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5024DA0, v8, v9), 1u);
  qword_5024D10 = 0x100000000LL;
  dword_5024CCC &= 0x8000u;
  qword_5024D08 = (__int64)&unk_5024D18;
  word_5024CD0 = 0;
  qword_5024CD8 = 0;
  dword_5024CC8 = v10;
  qword_5024CE0 = 0;
  qword_5024CE8 = 0;
  qword_5024CF0 = 0;
  qword_5024CF8 = 0;
  qword_5024D00 = 0;
  qword_5024D20 = 0;
  qword_5024D28 = (__int64)&unk_5024D40;
  qword_5024D30 = 1;
  dword_5024D38 = 0;
  byte_5024D3C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5024D10;
  if ( (unsigned __int64)(unsigned int)qword_5024D10 + 1 > HIDWORD(qword_5024D10) )
  {
    v35 = v11;
    sub_C8D5F0((char *)&unk_5024D18 - 16, &unk_5024D18, (unsigned int)qword_5024D10 + 1LL, 8);
    v12 = (unsigned int)qword_5024D10;
    v11 = v35;
  }
  *(_QWORD *)(qword_5024D08 + 8 * v12) = v11;
  qword_5024D50 = (__int64)&unk_49D9748;
  qword_5024CC0 = (__int64)&unk_49DC090;
  qword_5024D60 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5024D10) = qword_5024D10 + 1;
  qword_5024D80 = (__int64)nullsub_23;
  qword_5024D48 = 0;
  qword_5024D78 = (__int64)sub_984030;
  qword_5024D58 = 0;
  sub_C53080(&qword_5024CC0, "terminal-rule", 13);
  qword_5024CE8 = (__int64)"Apply the terminal rule";
  LOWORD(qword_5024D58) = 256;
  LOBYTE(qword_5024D48) = 0;
  qword_5024CF0 = 23;
  LOBYTE(dword_5024CCC) = dword_5024CCC & 0x9F | 0x20;
  sub_C53130(&qword_5024CC0);
  __cxa_atexit(sub_984900, &qword_5024CC0, &qword_4A427C0);
  qword_5024BE0 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5024CC0, v13, v14), 1u);
  qword_5024C30 = 0x100000000LL;
  dword_5024BEC &= 0x8000u;
  qword_5024C28 = (__int64)&unk_5024C38;
  word_5024BF0 = 0;
  qword_5024BF8 = 0;
  dword_5024BE8 = v15;
  qword_5024C00 = 0;
  qword_5024C08 = 0;
  qword_5024C10 = 0;
  qword_5024C18 = 0;
  qword_5024C20 = 0;
  qword_5024C40 = 0;
  qword_5024C48 = (__int64)&unk_5024C60;
  qword_5024C50 = 1;
  dword_5024C58 = 0;
  byte_5024C5C = 1;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_5024C30;
  if ( (unsigned __int64)(unsigned int)qword_5024C30 + 1 > HIDWORD(qword_5024C30) )
  {
    v36 = v16;
    sub_C8D5F0((char *)&unk_5024C38 - 16, &unk_5024C38, (unsigned int)qword_5024C30 + 1LL, 8);
    v17 = (unsigned int)qword_5024C30;
    v16 = v36;
  }
  *(_QWORD *)(qword_5024C28 + 8 * v17) = v16;
  qword_5024C70 = (__int64)&unk_49D9748;
  qword_5024BE0 = (__int64)&unk_49DC090;
  qword_5024C80 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5024C30) = qword_5024C30 + 1;
  qword_5024CA0 = (__int64)nullsub_23;
  qword_5024C68 = 0;
  qword_5024C98 = (__int64)sub_984030;
  qword_5024C78 = 0;
  sub_C53080(&qword_5024BE0, "join-splitedges", 15);
  qword_5024C10 = 50;
  qword_5024C08 = (__int64)"Coalesce copies on split edges (default=subtarget)";
  LOBYTE(dword_5024BEC) = dword_5024BEC & 0x9F | 0x20;
  sub_C53130(&qword_5024BE0);
  __cxa_atexit(sub_984900, &qword_5024BE0, &qword_4A427C0);
  qword_5024B00 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5024BE0, v18, v19), 1u);
  qword_5024B50 = 0x100000000LL;
  dword_5024B0C &= 0x8000u;
  word_5024B10 = 0;
  qword_5024B48 = (__int64)&unk_5024B58;
  qword_5024B18 = 0;
  dword_5024B08 = v20;
  qword_5024B20 = 0;
  qword_5024B28 = 0;
  qword_5024B30 = 0;
  qword_5024B38 = 0;
  qword_5024B40 = 0;
  qword_5024B60 = 0;
  qword_5024B68 = (__int64)&unk_5024B80;
  qword_5024B70 = 1;
  dword_5024B78 = 0;
  byte_5024B7C = 1;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_5024B50;
  if ( (unsigned __int64)(unsigned int)qword_5024B50 + 1 > HIDWORD(qword_5024B50) )
  {
    v37 = v21;
    sub_C8D5F0((char *)&unk_5024B58 - 16, &unk_5024B58, (unsigned int)qword_5024B50 + 1LL, 8);
    v22 = (unsigned int)qword_5024B50;
    v21 = v37;
  }
  *(_QWORD *)(qword_5024B48 + 8 * v22) = v21;
  LODWORD(qword_5024B50) = qword_5024B50 + 1;
  qword_5024B88 = 0;
  qword_5024B90 = (__int64)&unk_49DC110;
  qword_5024B98 = 0;
  qword_5024B00 = (__int64)&unk_49D97F0;
  qword_5024BA0 = (__int64)&unk_49DC200;
  qword_5024BC0 = (__int64)nullsub_26;
  qword_5024BB8 = (__int64)sub_9C26D0;
  sub_C53080(&qword_5024B00, "join-globalcopies", 17);
  qword_5024B30 = 52;
  qword_5024B28 = (__int64)"Coalesce copies that span blocks (default=subtarget)";
  LODWORD(qword_5024B88) = 0;
  BYTE4(qword_5024B98) = 1;
  LODWORD(qword_5024B98) = 0;
  LOBYTE(dword_5024B0C) = dword_5024B0C & 0x9F | 0x20;
  sub_C53130(&qword_5024B00);
  __cxa_atexit(sub_9C44F0, &qword_5024B00, &qword_4A427C0);
  qword_5024A20 = (__int64)&unk_49DC150;
  v25 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_9C44F0, &qword_5024B00, v23, v24), 1u);
  byte_5024A9C = 1;
  qword_5024A70 = 0x100000000LL;
  dword_5024A2C &= 0x8000u;
  qword_5024A68 = (__int64)&unk_5024A78;
  qword_5024A38 = 0;
  qword_5024A40 = 0;
  dword_5024A28 = v25;
  word_5024A30 = 0;
  qword_5024A48 = 0;
  qword_5024A50 = 0;
  qword_5024A58 = 0;
  qword_5024A60 = 0;
  qword_5024A80 = 0;
  qword_5024A88 = (__int64)&unk_5024AA0;
  qword_5024A90 = 1;
  dword_5024A98 = 0;
  v26 = sub_C57470();
  v27 = (unsigned int)qword_5024A70;
  if ( (unsigned __int64)(unsigned int)qword_5024A70 + 1 > HIDWORD(qword_5024A70) )
  {
    v38 = v26;
    sub_C8D5F0((char *)&unk_5024A78 - 16, &unk_5024A78, (unsigned int)qword_5024A70 + 1LL, 8);
    v27 = (unsigned int)qword_5024A70;
    v26 = v38;
  }
  *(_QWORD *)(qword_5024A68 + 8 * v27) = v26;
  qword_5024AB0 = (__int64)&unk_49D9748;
  qword_5024A20 = (__int64)&unk_49DC090;
  qword_5024AC0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5024A70) = qword_5024A70 + 1;
  qword_5024AE0 = (__int64)nullsub_23;
  qword_5024AA8 = 0;
  qword_5024AD8 = (__int64)sub_984030;
  qword_5024AB8 = 0;
  sub_C53080(&qword_5024A20, "verify-coalescing", 17);
  qword_5024A50 = 58;
  qword_5024A48 = (__int64)"Verify machine instrs before and after register coalescing";
  LOBYTE(dword_5024A2C) = dword_5024A2C & 0x9F | 0x20;
  sub_C53130(&qword_5024A20);
  __cxa_atexit(sub_984900, &qword_5024A20, &qword_4A427C0);
  qword_5024940 = (__int64)&unk_49DC150;
  v30 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5024A20, v28, v29), 1u);
  dword_502494C &= 0x8000u;
  word_5024950 = 0;
  qword_5024990 = 0x100000000LL;
  qword_5024958 = 0;
  qword_5024960 = 0;
  qword_5024968 = 0;
  dword_5024948 = v30;
  qword_5024970 = 0;
  qword_5024978 = 0;
  qword_5024980 = 0;
  qword_5024988 = (__int64)&unk_5024998;
  qword_50249A0 = 0;
  qword_50249A8 = (__int64)&unk_50249C0;
  qword_50249B0 = 1;
  dword_50249B8 = 0;
  byte_50249BC = 1;
  v31 = sub_C57470();
  v32 = (unsigned int)qword_5024990;
  v33 = (unsigned int)qword_5024990 + 1LL;
  if ( v33 > HIDWORD(qword_5024990) )
  {
    sub_C8D5F0((char *)&unk_5024998 - 16, &unk_5024998, v33, 8);
    v32 = (unsigned int)qword_5024990;
  }
  *(_QWORD *)(qword_5024988 + 8 * v32) = v31;
  LODWORD(qword_5024990) = qword_5024990 + 1;
  qword_50249C8 = 0;
  qword_50249D0 = (__int64)&unk_49D9728;
  qword_50249D8 = 0;
  qword_5024940 = (__int64)&unk_49DBF10;
  qword_50249E0 = (__int64)&unk_49DC290;
  qword_5024A00 = (__int64)nullsub_24;
  qword_50249F8 = (__int64)sub_984050;
  sub_C53080(&qword_5024940, "late-remat-update-threshold", 27);
  qword_5024970 = 266;
  LODWORD(qword_50249C8) = 100;
  BYTE4(qword_50249D8) = 1;
  LODWORD(qword_50249D8) = 100;
  LOBYTE(dword_502494C) = dword_502494C & 0x9F | 0x20;
  qword_5024968 = (__int64)"During rematerialization for a copy, if the def instruction has many other copy uses to be re"
                           "materialized, delay the multiple separate live interval update work and do them all at once a"
                           "fter all those rematerialization are done. It will save a lot of repeated work. ";
  sub_C53130(&qword_5024940);
  __cxa_atexit(sub_984970, &qword_5024940, &qword_4A427C0);
  v41 = &v39;
  v42 = "If the valnos size of an interval is larger than the threshold, it is regarded as a large interval. ";
  v39 = 100;
  v43 = 100;
  v40 = 1;
  sub_2F680B0(&unk_5024860, "large-interval-size-threshold", &v40, &v42, &v41);
  __cxa_atexit(sub_984970, &unk_5024860, &qword_4A427C0);
  v41 = &v39;
  v39 = 256;
  v42 = "For a large interval, if it is coalesced with other live intervals many times more than the threshold, stop its "
        "coalescing to control the compile time. ";
  v43 = 152;
  v40 = 1;
  sub_2F680B0(&unk_5024780, "large-interval-freq-threshold", &v40, &v42, &v41);
  return __cxa_atexit(sub_984970, &unk_5024780, &qword_4A427C0);
}
