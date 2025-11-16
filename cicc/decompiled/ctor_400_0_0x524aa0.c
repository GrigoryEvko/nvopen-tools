// Function: ctor_400_0
// Address: 0x524aa0
//
int ctor_400_0()
{
  int v0; // edx
  __int64 v1; // rax
  __int64 v2; // rdx
  int v3; // edx
  __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // edx
  __int64 v16; // r13
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  int v19; // edx
  __int64 v20; // rax
  __int64 v21; // rdx
  int v22; // edx
  __int64 v23; // rax
  __int64 v24; // rdx
  int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rdx
  int v28; // edx
  __int64 v29; // rax
  __int64 v30; // rdx
  int v31; // edx
  __int64 v32; // rax
  __int64 v33; // rdx
  int v34; // edx
  __int64 v35; // rax
  __int64 v36; // rdx
  int v37; // edx
  __int64 v38; // r12
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  int v41; // edx
  __int64 v42; // rbx
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  __int64 v46; // [rsp+8h] [rbp-108h]
  __int64 v47; // [rsp+8h] [rbp-108h]
  __int64 v48; // [rsp+8h] [rbp-108h]
  __int64 v49; // [rsp+8h] [rbp-108h]
  __int64 v50; // [rsp+8h] [rbp-108h]
  __int64 v51; // [rsp+8h] [rbp-108h]
  __int64 v52; // [rsp+8h] [rbp-108h]
  __int64 v53; // [rsp+8h] [rbp-108h]
  __int64 v54; // [rsp+8h] [rbp-108h]
  __int64 v55; // [rsp+8h] [rbp-108h]
  __int64 v56; // [rsp+8h] [rbp-108h]
  int v57; // [rsp+14h] [rbp-FCh] BYREF
  int *v58; // [rsp+18h] [rbp-F8h] BYREF
  _QWORD v59[2]; // [rsp+20h] [rbp-F0h] BYREF
  const char *v60; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v61; // [rsp+38h] [rbp-D8h]
  _QWORD v62[2]; // [rsp+40h] [rbp-D0h] BYREF
  int v63; // [rsp+50h] [rbp-C0h]
  const char *v64; // [rsp+58h] [rbp-B8h]
  __int64 v65; // [rsp+60h] [rbp-B0h]
  char *v66; // [rsp+68h] [rbp-A8h]
  __int64 v67; // [rsp+70h] [rbp-A0h]
  int v68; // [rsp+78h] [rbp-98h]
  const char *v69; // [rsp+80h] [rbp-90h]
  __int64 v70; // [rsp+88h] [rbp-88h]
  const char *v71; // [rsp+90h] [rbp-80h]
  __int64 v72; // [rsp+98h] [rbp-78h]
  int v73; // [rsp+A0h] [rbp-70h]
  const char *v74; // [rsp+A8h] [rbp-68h]
  __int64 v75; // [rsp+B0h] [rbp-60h]

  v60 = "Use debug info to correlate profiles. (Deprecated, use -profile-correlate=debug-info)";
  LOBYTE(v58) = 0;
  v59[0] = &v58;
  v61 = 85;
  sub_24531A0(&unk_4FE7640, "debug-info-correlate", &v60, v59);
  __cxa_atexit(sub_984900, &unk_4FE7640, &qword_4A427C0);
  v62[0] = byte_3F871B3;
  v64 = "No profile correlation";
  v66 = "debug-info";
  v69 = "Use debug info to correlate";
  v71 = "binary";
  v74 = "Use binary to correlate";
  v61 = 0x400000003LL;
  v58 = &v57;
  v60 = (const char *)v62;
  v62[1] = 0;
  v63 = 0;
  v65 = 22;
  v67 = 10;
  v68 = 1;
  v70 = 27;
  v72 = 6;
  v73 = 2;
  v75 = 23;
  v57 = 0;
  v59[0] = "Use debug info or binary file to correlate profiles.";
  v59[1] = 52;
  sub_245EA50(&unk_4FE73E0, "profile-correlate", v59, &v58, &v60);
  if ( v60 != (const char *)v62 )
    _libc_free(v60, "profile-correlate");
  __cxa_atexit(sub_24506A0, &unk_4FE73E0, &qword_4A427C0);
  qword_4FE7300 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE7350 = 0x100000000LL;
  dword_4FE730C &= 0x8000u;
  qword_4FE7348 = (__int64)&unk_4FE7358;
  word_4FE7310 = 0;
  qword_4FE7318 = 0;
  dword_4FE7308 = v0;
  qword_4FE7320 = 0;
  qword_4FE7328 = 0;
  qword_4FE7330 = 0;
  qword_4FE7338 = 0;
  qword_4FE7340 = 0;
  qword_4FE7360 = 0;
  qword_4FE7368 = (__int64)&unk_4FE7380;
  qword_4FE7370 = 1;
  dword_4FE7378 = 0;
  byte_4FE737C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FE7350;
  if ( (unsigned __int64)(unsigned int)qword_4FE7350 + 1 > HIDWORD(qword_4FE7350) )
  {
    v46 = v1;
    sub_C8D5F0((char *)&unk_4FE7358 - 16, &unk_4FE7358, (unsigned int)qword_4FE7350 + 1LL, 8);
    v2 = (unsigned int)qword_4FE7350;
    v1 = v46;
  }
  *(_QWORD *)(qword_4FE7348 + 8 * v2) = v1;
  LODWORD(qword_4FE7350) = qword_4FE7350 + 1;
  qword_4FE7388 = 0;
  qword_4FE7390 = (__int64)&unk_49D9748;
  qword_4FE7398 = 0;
  qword_4FE7300 = (__int64)&unk_49DC090;
  qword_4FE73A0 = (__int64)&unk_49DC1D0;
  qword_4FE73C0 = (__int64)nullsub_23;
  qword_4FE73B8 = (__int64)sub_984030;
  sub_C53080(&qword_4FE7300, "hash-based-counter-split", 24);
  LOWORD(qword_4FE7398) = 257;
  qword_4FE7328 = (__int64)"Rename counter variable of a comdat function based on cfg hash";
  qword_4FE7330 = 62;
  LOBYTE(qword_4FE7388) = 1;
  sub_C53130(&qword_4FE7300);
  __cxa_atexit(sub_984900, &qword_4FE7300, &qword_4A427C0);
  v60 = "Enable relocating counters at runtime.";
  LOBYTE(v58) = 0;
  v59[0] = &v58;
  v61 = 38;
  sub_2453390(&unk_4FE7220, "runtime-counter-relocation", &v60, v59);
  __cxa_atexit(sub_984900, &unk_4FE7220, &qword_4A427C0);
  qword_4FE7140 = (__int64)&unk_49DC150;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE7190 = 0x100000000LL;
  dword_4FE714C &= 0x8000u;
  qword_4FE7188 = (__int64)&unk_4FE7198;
  word_4FE7150 = 0;
  qword_4FE7158 = 0;
  dword_4FE7148 = v3;
  qword_4FE7160 = 0;
  qword_4FE7168 = 0;
  qword_4FE7170 = 0;
  qword_4FE7178 = 0;
  qword_4FE7180 = 0;
  qword_4FE71A0 = 0;
  qword_4FE71A8 = (__int64)&unk_4FE71C0;
  qword_4FE71B0 = 1;
  dword_4FE71B8 = 0;
  byte_4FE71BC = 1;
  v4 = sub_C57470();
  v5 = (unsigned int)qword_4FE7190;
  if ( (unsigned __int64)(unsigned int)qword_4FE7190 + 1 > HIDWORD(qword_4FE7190) )
  {
    v47 = v4;
    sub_C8D5F0((char *)&unk_4FE7198 - 16, &unk_4FE7198, (unsigned int)qword_4FE7190 + 1LL, 8);
    v5 = (unsigned int)qword_4FE7190;
    v4 = v47;
  }
  *(_QWORD *)(qword_4FE7188 + 8 * v5) = v4;
  LODWORD(qword_4FE7190) = qword_4FE7190 + 1;
  qword_4FE71C8 = 0;
  qword_4FE71D0 = (__int64)&unk_49D9748;
  qword_4FE71D8 = 0;
  qword_4FE7140 = (__int64)&unk_49DC090;
  qword_4FE71E0 = (__int64)&unk_49DC1D0;
  qword_4FE7200 = (__int64)nullsub_23;
  qword_4FE71F8 = (__int64)sub_984030;
  sub_C53080(&qword_4FE7140, "vp-static-alloc", 15);
  qword_4FE7168 = (__int64)"Do static counter allocation for value profiler";
  LOWORD(qword_4FE71D8) = 257;
  qword_4FE7170 = 47;
  LOBYTE(qword_4FE71C8) = 1;
  sub_C53130(&qword_4FE7140);
  __cxa_atexit(sub_984900, &qword_4FE7140, &qword_4A427C0);
  qword_4FE7060 = (__int64)&unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE70B0 = 0x100000000LL;
  dword_4FE706C &= 0x8000u;
  qword_4FE70A8 = (__int64)&unk_4FE70B8;
  word_4FE7070 = 0;
  qword_4FE7078 = 0;
  dword_4FE7068 = v6;
  qword_4FE7080 = 0;
  qword_4FE7088 = 0;
  qword_4FE7090 = 0;
  qword_4FE7098 = 0;
  qword_4FE70A0 = 0;
  qword_4FE70C0 = 0;
  qword_4FE70C8 = (__int64)&unk_4FE70E0;
  qword_4FE70D0 = 1;
  dword_4FE70D8 = 0;
  byte_4FE70DC = 1;
  v7 = sub_C57470();
  v8 = (unsigned int)qword_4FE70B0;
  if ( (unsigned __int64)(unsigned int)qword_4FE70B0 + 1 > HIDWORD(qword_4FE70B0) )
  {
    v48 = v7;
    sub_C8D5F0((char *)&unk_4FE70B8 - 16, &unk_4FE70B8, (unsigned int)qword_4FE70B0 + 1LL, 8);
    v8 = (unsigned int)qword_4FE70B0;
    v7 = v48;
  }
  *(_QWORD *)(qword_4FE70A8 + 8 * v8) = v7;
  LODWORD(qword_4FE70B0) = qword_4FE70B0 + 1;
  byte_4FE7100 = 0;
  qword_4FE70F0 = (__int64)&unk_49DE5F0;
  qword_4FE70E8 = 0;
  qword_4FE70F8 = 0;
  qword_4FE7060 = (__int64)&unk_49DE610;
  qword_4FE7108 = (__int64)&unk_49DC2F0;
  qword_4FE7128 = (__int64)nullsub_190;
  qword_4FE7120 = (__int64)sub_D83E80;
  sub_C53080(&qword_4FE7060, "vp-counters-per-site", 20);
  qword_4FE7088 = (__int64)"The average number of profile counters allocated per value profiling site.";
  qword_4FE70E8 = 0x3FF0000000000000LL;
  qword_4FE70F8 = 0x3FF0000000000000LL;
  qword_4FE7090 = 74;
  byte_4FE7100 = 1;
  sub_C53130(&qword_4FE7060);
  __cxa_atexit(sub_D84280, &qword_4FE7060, &qword_4A427C0);
  qword_4FE6F80 = (__int64)&unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE6FD0 = 0x100000000LL;
  dword_4FE6F8C &= 0x8000u;
  word_4FE6F90 = 0;
  qword_4FE6FC8 = (__int64)&unk_4FE6FD8;
  qword_4FE6F98 = 0;
  dword_4FE6F88 = v9;
  qword_4FE6FA0 = 0;
  qword_4FE6FA8 = 0;
  qword_4FE6FB0 = 0;
  qword_4FE6FB8 = 0;
  qword_4FE6FC0 = 0;
  qword_4FE6FE0 = 0;
  qword_4FE6FE8 = (__int64)&unk_4FE7000;
  qword_4FE6FF0 = 1;
  dword_4FE6FF8 = 0;
  byte_4FE6FFC = 1;
  v10 = sub_C57470();
  v11 = (unsigned int)qword_4FE6FD0;
  if ( (unsigned __int64)(unsigned int)qword_4FE6FD0 + 1 > HIDWORD(qword_4FE6FD0) )
  {
    v49 = v10;
    sub_C8D5F0((char *)&unk_4FE6FD8 - 16, &unk_4FE6FD8, (unsigned int)qword_4FE6FD0 + 1LL, 8);
    v11 = (unsigned int)qword_4FE6FD0;
    v10 = v49;
  }
  *(_QWORD *)(qword_4FE6FC8 + 8 * v11) = v10;
  LODWORD(qword_4FE6FD0) = qword_4FE6FD0 + 1;
  qword_4FE7008 = 0;
  qword_4FE7010 = (__int64)&unk_49D9748;
  qword_4FE7018 = 0;
  qword_4FE6F80 = (__int64)&unk_49DC090;
  qword_4FE7020 = (__int64)&unk_49DC1D0;
  qword_4FE7040 = (__int64)nullsub_23;
  qword_4FE7038 = (__int64)sub_984030;
  sub_C53080(&qword_4FE6F80, "instrprof-atomic-counter-update-all", 35);
  qword_4FE6FB0 = 58;
  qword_4FE6FA8 = (__int64)"Make all profile counter updates atomic (for testing only)";
  LOWORD(qword_4FE7018) = 256;
  LOBYTE(qword_4FE7008) = 0;
  sub_C53130(&qword_4FE6F80);
  __cxa_atexit(sub_984900, &qword_4FE6F80, &qword_4A427C0);
  qword_4FE6EA0 = (__int64)&unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FE6F1C = 1;
  qword_4FE6EF0 = 0x100000000LL;
  dword_4FE6EAC &= 0x8000u;
  qword_4FE6EE8 = (__int64)&unk_4FE6EF8;
  qword_4FE6EB8 = 0;
  qword_4FE6EC0 = 0;
  dword_4FE6EA8 = v12;
  word_4FE6EB0 = 0;
  qword_4FE6EC8 = 0;
  qword_4FE6ED0 = 0;
  qword_4FE6ED8 = 0;
  qword_4FE6EE0 = 0;
  qword_4FE6F00 = 0;
  qword_4FE6F08 = (__int64)&unk_4FE6F20;
  qword_4FE6F10 = 1;
  dword_4FE6F18 = 0;
  v13 = sub_C57470();
  v14 = (unsigned int)qword_4FE6EF0;
  if ( (unsigned __int64)(unsigned int)qword_4FE6EF0 + 1 > HIDWORD(qword_4FE6EF0) )
  {
    v50 = v13;
    sub_C8D5F0((char *)&unk_4FE6EF8 - 16, &unk_4FE6EF8, (unsigned int)qword_4FE6EF0 + 1LL, 8);
    v14 = (unsigned int)qword_4FE6EF0;
    v13 = v50;
  }
  *(_QWORD *)(qword_4FE6EE8 + 8 * v14) = v13;
  LODWORD(qword_4FE6EF0) = qword_4FE6EF0 + 1;
  qword_4FE6F28 = 0;
  qword_4FE6F30 = (__int64)&unk_49D9748;
  qword_4FE6F38 = 0;
  qword_4FE6EA0 = (__int64)&unk_49DC090;
  qword_4FE6F40 = (__int64)&unk_49DC1D0;
  qword_4FE6F60 = (__int64)nullsub_23;
  qword_4FE6F58 = (__int64)sub_984030;
  sub_C53080(&qword_4FE6EA0, "atomic-counter-update-promoted", 30);
  qword_4FE6ED0 = 68;
  qword_4FE6EC8 = (__int64)"Do counter update using atomic fetch add  for promoted counters only";
  LOWORD(qword_4FE6F38) = 256;
  LOBYTE(qword_4FE6F28) = 0;
  sub_C53130(&qword_4FE6EA0);
  __cxa_atexit(sub_984900, &qword_4FE6EA0, &qword_4A427C0);
  v60 = "Use atomic fetch add for first counter in a function (usually the entry counter)";
  LOBYTE(v58) = 0;
  v59[0] = &v58;
  v61 = 80;
  sub_24531A0(&unk_4FE6DC0, "atomic-first-counter", &v60, v59);
  __cxa_atexit(sub_984900, &unk_4FE6DC0, &qword_4A427C0);
  v60 = "Do conditional counter updates in single byte counters mode)";
  LOBYTE(v58) = 0;
  v59[0] = &v58;
  v61 = 60;
  sub_2453390(&unk_4FE6CE0, "conditional-counter-update", &v60, v59);
  __cxa_atexit(sub_984900, &unk_4FE6CE0, &qword_4A427C0);
  v60 = "Do counter register promotion";
  LOBYTE(v58) = 0;
  v59[0] = &v58;
  v61 = 29;
  sub_24531A0(&unk_4FE6C00, "do-counter-promotion", &v60, v59);
  __cxa_atexit(sub_984900, &unk_4FE6C00, &qword_4A427C0);
  qword_4FE6B20 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FE6B2C &= 0x8000u;
  word_4FE6B30 = 0;
  qword_4FE6B70 = 0x100000000LL;
  qword_4FE6B38 = 0;
  qword_4FE6B40 = 0;
  qword_4FE6B48 = 0;
  dword_4FE6B28 = v15;
  qword_4FE6B50 = 0;
  qword_4FE6B58 = 0;
  qword_4FE6B60 = 0;
  qword_4FE6B68 = (__int64)&unk_4FE6B78;
  qword_4FE6B80 = 0;
  qword_4FE6B88 = (__int64)&unk_4FE6BA0;
  qword_4FE6B90 = 1;
  dword_4FE6B98 = 0;
  byte_4FE6B9C = 1;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_4FE6B70;
  v18 = (unsigned int)qword_4FE6B70 + 1LL;
  if ( v18 > HIDWORD(qword_4FE6B70) )
  {
    sub_C8D5F0((char *)&unk_4FE6B78 - 16, &unk_4FE6B78, v18, 8);
    v17 = (unsigned int)qword_4FE6B70;
  }
  *(_QWORD *)(qword_4FE6B68 + 8 * v17) = v16;
  qword_4FE6BB0 = (__int64)&unk_49D9728;
  qword_4FE6B20 = (__int64)&unk_49DBF10;
  LODWORD(qword_4FE6B70) = qword_4FE6B70 + 1;
  qword_4FE6BA8 = 0;
  qword_4FE6BC0 = (__int64)&unk_49DC290;
  qword_4FE6BB8 = 0;
  qword_4FE6BE0 = (__int64)nullsub_24;
  qword_4FE6BD8 = (__int64)sub_984050;
  sub_C53080(&qword_4FE6B20, "max-counter-promotions-per-loop", 31);
  LODWORD(qword_4FE6BA8) = 20;
  qword_4FE6B48 = (__int64)"Max number counter promotions per loop to avoid increasing register pressure too much";
  BYTE4(qword_4FE6BB8) = 1;
  LODWORD(qword_4FE6BB8) = 20;
  qword_4FE6B50 = 85;
  sub_C53130(&qword_4FE6B20);
  __cxa_atexit(sub_984970, &qword_4FE6B20, &qword_4A427C0);
  qword_4FE6A40 = (__int64)&unk_49DC150;
  v19 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FE6A4C &= 0x8000u;
  word_4FE6A50 = 0;
  qword_4FE6A90 = 0x100000000LL;
  qword_4FE6A88 = (__int64)&unk_4FE6A98;
  qword_4FE6A58 = 0;
  qword_4FE6A60 = 0;
  dword_4FE6A48 = v19;
  qword_4FE6A68 = 0;
  qword_4FE6A70 = 0;
  qword_4FE6A78 = 0;
  qword_4FE6A80 = 0;
  qword_4FE6AA0 = 0;
  qword_4FE6AA8 = (__int64)&unk_4FE6AC0;
  qword_4FE6AB0 = 1;
  dword_4FE6AB8 = 0;
  byte_4FE6ABC = 1;
  v20 = sub_C57470();
  v21 = (unsigned int)qword_4FE6A90;
  if ( (unsigned __int64)(unsigned int)qword_4FE6A90 + 1 > HIDWORD(qword_4FE6A90) )
  {
    v51 = v20;
    sub_C8D5F0((char *)&unk_4FE6A98 - 16, &unk_4FE6A98, (unsigned int)qword_4FE6A90 + 1LL, 8);
    v21 = (unsigned int)qword_4FE6A90;
    v20 = v51;
  }
  *(_QWORD *)(qword_4FE6A88 + 8 * v21) = v20;
  LODWORD(qword_4FE6A90) = qword_4FE6A90 + 1;
  qword_4FE6AC8 = 0;
  qword_4FE6AD0 = (__int64)&unk_49DA090;
  qword_4FE6AD8 = 0;
  qword_4FE6A40 = (__int64)&unk_49DBF90;
  qword_4FE6AE0 = (__int64)&unk_49DC230;
  qword_4FE6B00 = (__int64)nullsub_58;
  qword_4FE6AF8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FE6A40, "max-counter-promotions", 22);
  LODWORD(qword_4FE6AC8) = -1;
  qword_4FE6A68 = (__int64)"Max number of allowed counter promotions";
  BYTE4(qword_4FE6AD8) = 1;
  LODWORD(qword_4FE6AD8) = -1;
  qword_4FE6A70 = 40;
  sub_C53130(&qword_4FE6A40);
  __cxa_atexit(sub_B2B680, &qword_4FE6A40, &qword_4A427C0);
  qword_4FE6960 = (__int64)&unk_49DC150;
  v22 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FE696C &= 0x8000u;
  word_4FE6970 = 0;
  qword_4FE69B0 = 0x100000000LL;
  qword_4FE69A8 = (__int64)&unk_4FE69B8;
  qword_4FE6978 = 0;
  qword_4FE6980 = 0;
  dword_4FE6968 = v22;
  qword_4FE6988 = 0;
  qword_4FE6990 = 0;
  qword_4FE6998 = 0;
  qword_4FE69A0 = 0;
  qword_4FE69C0 = 0;
  qword_4FE69C8 = (__int64)&unk_4FE69E0;
  qword_4FE69D0 = 1;
  dword_4FE69D8 = 0;
  byte_4FE69DC = 1;
  v23 = sub_C57470();
  v24 = (unsigned int)qword_4FE69B0;
  if ( (unsigned __int64)(unsigned int)qword_4FE69B0 + 1 > HIDWORD(qword_4FE69B0) )
  {
    v52 = v23;
    sub_C8D5F0((char *)&unk_4FE69B8 - 16, &unk_4FE69B8, (unsigned int)qword_4FE69B0 + 1LL, 8);
    v24 = (unsigned int)qword_4FE69B0;
    v23 = v52;
  }
  *(_QWORD *)(qword_4FE69A8 + 8 * v24) = v23;
  qword_4FE69F0 = (__int64)&unk_49D9728;
  qword_4FE6960 = (__int64)&unk_49DBF10;
  LODWORD(qword_4FE69B0) = qword_4FE69B0 + 1;
  qword_4FE69E8 = 0;
  qword_4FE6A00 = (__int64)&unk_49DC290;
  qword_4FE69F8 = 0;
  qword_4FE6A20 = (__int64)nullsub_24;
  qword_4FE6A18 = (__int64)sub_984050;
  sub_C53080(&qword_4FE6960, "speculative-counter-promotion-max-exiting", 41);
  LODWORD(qword_4FE69E8) = 3;
  qword_4FE6988 = (__int64)"The max number of exiting blocks of a loop to allow  speculative counter promotion";
  BYTE4(qword_4FE69F8) = 1;
  LODWORD(qword_4FE69F8) = 3;
  qword_4FE6990 = 82;
  sub_C53130(&qword_4FE6960);
  __cxa_atexit(sub_984970, &qword_4FE6960, &qword_4A427C0);
  qword_4FE6880 = (__int64)&unk_49DC150;
  v25 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE68D0 = 0x100000000LL;
  dword_4FE688C &= 0x8000u;
  qword_4FE68C8 = (__int64)&unk_4FE68D8;
  word_4FE6890 = 0;
  qword_4FE6898 = 0;
  dword_4FE6888 = v25;
  qword_4FE68A0 = 0;
  qword_4FE68A8 = 0;
  qword_4FE68B0 = 0;
  qword_4FE68B8 = 0;
  qword_4FE68C0 = 0;
  qword_4FE68E0 = 0;
  qword_4FE68E8 = (__int64)&unk_4FE6900;
  qword_4FE68F0 = 1;
  dword_4FE68F8 = 0;
  byte_4FE68FC = 1;
  v26 = sub_C57470();
  v27 = (unsigned int)qword_4FE68D0;
  if ( (unsigned __int64)(unsigned int)qword_4FE68D0 + 1 > HIDWORD(qword_4FE68D0) )
  {
    v53 = v26;
    sub_C8D5F0((char *)&unk_4FE68D8 - 16, &unk_4FE68D8, (unsigned int)qword_4FE68D0 + 1LL, 8);
    v27 = (unsigned int)qword_4FE68D0;
    v26 = v53;
  }
  *(_QWORD *)(qword_4FE68C8 + 8 * v27) = v26;
  LODWORD(qword_4FE68D0) = qword_4FE68D0 + 1;
  qword_4FE6908 = 0;
  qword_4FE6910 = (__int64)&unk_49D9748;
  qword_4FE6918 = 0;
  qword_4FE6880 = (__int64)&unk_49DC090;
  qword_4FE6920 = (__int64)&unk_49DC1D0;
  qword_4FE6940 = (__int64)nullsub_23;
  qword_4FE6938 = (__int64)sub_984030;
  sub_C53080(&qword_4FE6880, "speculative-counter-promotion-to-loop", 37);
  qword_4FE68B0 = 189;
  qword_4FE68A8 = (__int64)"When the option is false, if the target block is in a loop, the promotion will be disallowed "
                           "unless the promoted counter  update can be further/iteratively promoted into an acyclic  region.";
  sub_C53130(&qword_4FE6880);
  __cxa_atexit(sub_984900, &qword_4FE6880, &qword_4A427C0);
  qword_4FE67A0 = (__int64)&unk_49DC150;
  v28 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE67F0 = 0x100000000LL;
  dword_4FE67AC &= 0x8000u;
  qword_4FE67E8 = (__int64)&unk_4FE67F8;
  word_4FE67B0 = 0;
  qword_4FE67B8 = 0;
  dword_4FE67A8 = v28;
  qword_4FE67C0 = 0;
  qword_4FE67C8 = 0;
  qword_4FE67D0 = 0;
  qword_4FE67D8 = 0;
  qword_4FE67E0 = 0;
  qword_4FE6800 = 0;
  qword_4FE6808 = (__int64)&unk_4FE6820;
  qword_4FE6810 = 1;
  dword_4FE6818 = 0;
  byte_4FE681C = 1;
  v29 = sub_C57470();
  v30 = (unsigned int)qword_4FE67F0;
  if ( (unsigned __int64)(unsigned int)qword_4FE67F0 + 1 > HIDWORD(qword_4FE67F0) )
  {
    v54 = v29;
    sub_C8D5F0((char *)&unk_4FE67F8 - 16, &unk_4FE67F8, (unsigned int)qword_4FE67F0 + 1LL, 8);
    v30 = (unsigned int)qword_4FE67F0;
    v29 = v54;
  }
  *(_QWORD *)(qword_4FE67E8 + 8 * v30) = v29;
  LODWORD(qword_4FE67F0) = qword_4FE67F0 + 1;
  qword_4FE6828 = 0;
  qword_4FE6830 = (__int64)&unk_49D9748;
  qword_4FE6838 = 0;
  qword_4FE67A0 = (__int64)&unk_49DC090;
  qword_4FE6840 = (__int64)&unk_49DC1D0;
  qword_4FE6860 = (__int64)nullsub_23;
  qword_4FE6858 = (__int64)sub_984030;
  sub_C53080(&qword_4FE67A0, "iterative-counter-promotion", 27);
  LOWORD(qword_4FE6838) = 257;
  qword_4FE67C8 = (__int64)"Allow counter promotion across the whole loop nest.";
  LOBYTE(qword_4FE6828) = 1;
  qword_4FE67D0 = 51;
  sub_C53130(&qword_4FE67A0);
  __cxa_atexit(sub_984900, &qword_4FE67A0, &qword_4A427C0);
  qword_4FE66C0 = (__int64)&unk_49DC150;
  v31 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE6710 = 0x100000000LL;
  dword_4FE66CC &= 0x8000u;
  qword_4FE6708 = (__int64)&unk_4FE6718;
  word_4FE66D0 = 0;
  qword_4FE66D8 = 0;
  dword_4FE66C8 = v31;
  qword_4FE66E0 = 0;
  qword_4FE66E8 = 0;
  qword_4FE66F0 = 0;
  qword_4FE66F8 = 0;
  qword_4FE6700 = 0;
  qword_4FE6720 = 0;
  qword_4FE6728 = (__int64)&unk_4FE6740;
  qword_4FE6730 = 1;
  dword_4FE6738 = 0;
  byte_4FE673C = 1;
  v32 = sub_C57470();
  v33 = (unsigned int)qword_4FE6710;
  if ( (unsigned __int64)(unsigned int)qword_4FE6710 + 1 > HIDWORD(qword_4FE6710) )
  {
    v55 = v32;
    sub_C8D5F0((char *)&unk_4FE6718 - 16, &unk_4FE6718, (unsigned int)qword_4FE6710 + 1LL, 8);
    v33 = (unsigned int)qword_4FE6710;
    v32 = v55;
  }
  *(_QWORD *)(qword_4FE6708 + 8 * v33) = v32;
  LODWORD(qword_4FE6710) = qword_4FE6710 + 1;
  qword_4FE6748 = 0;
  qword_4FE6750 = (__int64)&unk_49D9748;
  qword_4FE6758 = 0;
  qword_4FE66C0 = (__int64)&unk_49DC090;
  qword_4FE6760 = (__int64)&unk_49DC1D0;
  qword_4FE6780 = (__int64)nullsub_23;
  qword_4FE6778 = (__int64)sub_984030;
  sub_C53080(&qword_4FE66C0, "skip-ret-exit-block", 19);
  LOWORD(qword_4FE6758) = 257;
  qword_4FE66E8 = (__int64)"Suppress counter promotion if exit blocks contain ret.";
  LOBYTE(qword_4FE6748) = 1;
  qword_4FE66F0 = 54;
  sub_C53130(&qword_4FE66C0);
  __cxa_atexit(sub_984900, &qword_4FE66C0, &qword_4A427C0);
  qword_4FE65E0 = (__int64)&unk_49DC150;
  v34 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE6630 = 0x100000000LL;
  dword_4FE65EC &= 0x8000u;
  qword_4FE6628 = (__int64)&unk_4FE6638;
  word_4FE65F0 = 0;
  qword_4FE65F8 = 0;
  dword_4FE65E8 = v34;
  qword_4FE6600 = 0;
  qword_4FE6608 = 0;
  qword_4FE6610 = 0;
  qword_4FE6618 = 0;
  qword_4FE6620 = 0;
  qword_4FE6640 = 0;
  qword_4FE6648 = (__int64)&unk_4FE6660;
  qword_4FE6650 = 1;
  dword_4FE6658 = 0;
  byte_4FE665C = 1;
  v35 = sub_C57470();
  v36 = (unsigned int)qword_4FE6630;
  if ( (unsigned __int64)(unsigned int)qword_4FE6630 + 1 > HIDWORD(qword_4FE6630) )
  {
    v56 = v35;
    sub_C8D5F0((char *)&unk_4FE6638 - 16, &unk_4FE6638, (unsigned int)qword_4FE6630 + 1LL, 8);
    v36 = (unsigned int)qword_4FE6630;
    v35 = v56;
  }
  *(_QWORD *)(qword_4FE6628 + 8 * v36) = v35;
  LODWORD(qword_4FE6630) = qword_4FE6630 + 1;
  qword_4FE6668 = 0;
  qword_4FE6670 = (__int64)&unk_49D9748;
  qword_4FE6678 = 0;
  qword_4FE65E0 = (__int64)&unk_49DC090;
  qword_4FE6680 = (__int64)&unk_49DC1D0;
  qword_4FE66A0 = (__int64)nullsub_23;
  qword_4FE6698 = (__int64)sub_984030;
  sub_C53080(&qword_4FE65E0, "sampled-instrumentation", 23);
  LOWORD(qword_4FE6678) = 256;
  LOBYTE(qword_4FE6668) = 0;
  qword_4FE6610 = 31;
  LOBYTE(dword_4FE65EC) = dword_4FE65EC & 0xF8 | 1;
  qword_4FE6608 = (__int64)"Do PGO instrumentation sampling";
  sub_C53130(&qword_4FE65E0);
  __cxa_atexit(sub_984900, &qword_4FE65E0, &qword_4A427C0);
  qword_4FE6500 = (__int64)&unk_49DC150;
  v37 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FE6550 = 0x100000000LL;
  dword_4FE650C &= 0x8000u;
  word_4FE6510 = 0;
  qword_4FE6548 = (__int64)&unk_4FE6558;
  qword_4FE6518 = 0;
  dword_4FE6508 = v37;
  qword_4FE6520 = 0;
  qword_4FE6528 = 0;
  qword_4FE6530 = 0;
  qword_4FE6538 = 0;
  qword_4FE6540 = 0;
  qword_4FE6560 = 0;
  qword_4FE6568 = (__int64)&unk_4FE6580;
  qword_4FE6570 = 1;
  dword_4FE6578 = 0;
  byte_4FE657C = 1;
  v38 = sub_C57470();
  v39 = (unsigned int)qword_4FE6550;
  v40 = (unsigned int)qword_4FE6550 + 1LL;
  if ( v40 > HIDWORD(qword_4FE6550) )
  {
    sub_C8D5F0((char *)&unk_4FE6558 - 16, &unk_4FE6558, v40, 8);
    v39 = (unsigned int)qword_4FE6550;
  }
  *(_QWORD *)(qword_4FE6548 + 8 * v39) = v38;
  qword_4FE6590 = (__int64)&unk_49D9728;
  qword_4FE6500 = (__int64)&unk_49DBF10;
  LODWORD(qword_4FE6550) = qword_4FE6550 + 1;
  qword_4FE6588 = 0;
  qword_4FE65A0 = (__int64)&unk_49DC290;
  qword_4FE6598 = 0;
  qword_4FE65C0 = (__int64)nullsub_24;
  qword_4FE65B8 = (__int64)sub_984050;
  sub_C53080(&qword_4FE6500, "sampled-instr-period", 20);
  qword_4FE6530 = 427;
  qword_4FE6528 = (__int64)"Set the profile instrumentation sample period. A sample period of 0 is invalid. For each samp"
                           "le period, a fixed number of consecutive samples will be recorded. The number is controlled b"
                           "y 'sampled-instr-burst-duration' flag. The default sample period of 65536 is optimized for ge"
                           "nerating efficient code that leverages unsigned short integer wrapping in overflow, but this "
                           "is disabled under simple sampling (burst duration = 1).";
  LODWORD(qword_4FE6588) = 0x10000;
  BYTE4(qword_4FE6598) = 1;
  LODWORD(qword_4FE6598) = 0x10000;
  sub_C53130(&qword_4FE6500);
  __cxa_atexit(sub_984970, &qword_4FE6500, &qword_4A427C0);
  qword_4FE6420 = (__int64)&unk_49DC150;
  v41 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FE642C &= 0x8000u;
  word_4FE6430 = 0;
  qword_4FE6470 = 0x100000000LL;
  qword_4FE6438 = 0;
  qword_4FE6440 = 0;
  qword_4FE6448 = 0;
  dword_4FE6428 = v41;
  qword_4FE6450 = 0;
  qword_4FE6458 = 0;
  qword_4FE6460 = 0;
  qword_4FE6468 = (__int64)&unk_4FE6478;
  qword_4FE6480 = 0;
  qword_4FE6488 = (__int64)&unk_4FE64A0;
  qword_4FE6490 = 1;
  dword_4FE6498 = 0;
  byte_4FE649C = 1;
  v42 = sub_C57470();
  v43 = (unsigned int)qword_4FE6470;
  v44 = (unsigned int)qword_4FE6470 + 1LL;
  if ( v44 > HIDWORD(qword_4FE6470) )
  {
    sub_C8D5F0((char *)&unk_4FE6478 - 16, &unk_4FE6478, v44, 8);
    v43 = (unsigned int)qword_4FE6470;
  }
  *(_QWORD *)(qword_4FE6468 + 8 * v43) = v42;
  qword_4FE64B0 = (__int64)&unk_49D9728;
  qword_4FE6420 = (__int64)&unk_49DBF10;
  LODWORD(qword_4FE6470) = qword_4FE6470 + 1;
  qword_4FE64A8 = 0;
  qword_4FE64C0 = (__int64)&unk_49DC290;
  qword_4FE64B8 = 0;
  qword_4FE64E0 = (__int64)nullsub_24;
  qword_4FE64D8 = (__int64)sub_984050;
  sub_C53080(&qword_4FE6420, "sampled-instr-burst-duration", 28);
  qword_4FE6450 = 330;
  qword_4FE6448 = (__int64)"Set the profile instrumentation burst duration, which can range from 1 to the value of 'sampl"
                           "ed-instr-period' (0 is invalid). This number of samples will be recorded for each 'sampled-in"
                           "str-period' count update. Setting to 1 enables simple sampling, in which case it is recommend"
                           "ed to set 'sampled-instr-period' to a prime number.";
  LODWORD(qword_4FE64A8) = 200;
  BYTE4(qword_4FE64B8) = 1;
  LODWORD(qword_4FE64B8) = 200;
  sub_C53130(&qword_4FE6420);
  return __cxa_atexit(sub_984970, &qword_4FE6420, &qword_4A427C0);
}
