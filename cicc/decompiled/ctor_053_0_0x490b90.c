// Function: ctor_053_0
// Address: 0x490b90
//
int ctor_053_0()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  __int64 v4; // rax
  int v5; // edx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  int v13; // edx
  __int64 v14; // r12
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  int v17; // edx
  __int64 v18; // r12
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  int v21; // edx
  __int64 v22; // r12
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rdx
  int v28; // edx
  __int64 v29; // rax
  __int64 v30; // rdx
  int v31; // edx
  __int64 v32; // r13
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  int v35; // edx
  __int64 v36; // rbx
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v41; // [rsp+8h] [rbp-68h]
  __int64 v42; // [rsp+8h] [rbp-68h]
  __int64 v43; // [rsp+8h] [rbp-68h]
  __int64 v44; // [rsp+8h] [rbp-68h]
  _QWORD v45[4]; // [rsp+10h] [rbp-60h] BYREF
  char v46; // [rsp+30h] [rbp-40h]
  char v47; // [rsp+31h] [rbp-3Fh]

  qword_4F87520 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F8752C &= 0x8000u;
  word_4F87530 = 0;
  qword_4F87570 = 0x100000000LL;
  qword_4F87538 = 0;
  qword_4F87540 = 0;
  qword_4F87548 = 0;
  dword_4F87528 = v0;
  qword_4F87550 = 0;
  qword_4F87558 = 0;
  qword_4F87560 = 0;
  qword_4F87568 = (__int64)&unk_4F87578;
  qword_4F87580 = 0;
  qword_4F87588 = (__int64)&unk_4F875A0;
  qword_4F87590 = 1;
  dword_4F87598 = 0;
  byte_4F8759C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F87570;
  v3 = (unsigned int)qword_4F87570 + 1LL;
  if ( v3 > HIDWORD(qword_4F87570) )
  {
    sub_C8D5F0((char *)&unk_4F87578 - 16, &unk_4F87578, v3, 8);
    v2 = (unsigned int)qword_4F87570;
  }
  *(_QWORD *)(qword_4F87568 + 8 * v2) = v1;
  LODWORD(qword_4F87570) = qword_4F87570 + 1;
  byte_4F875BC = 0;
  qword_4F875B0 = (__int64)&unk_49D9728;
  qword_4F87520 = (__int64)&unk_49DDF20;
  qword_4F875A8 = 0;
  qword_4F875E0 = (__int64)nullsub_186;
  qword_4F875C0 = (__int64)&unk_49DC290;
  qword_4F875D8 = (__int64)sub_D320E0;
  sub_C53080(&qword_4F87520, "force-vector-width", 18);
  qword_4F87550 = 40;
  LOBYTE(dword_4F8752C) = dword_4F8752C & 0x9F | 0x20;
  qword_4F87548 = (__int64)"Sets the SIMD width. Zero is autoselect.";
  if ( qword_4F875A8 )
  {
    v4 = sub_CEADF0();
    v47 = 1;
    v45[0] = "cl::location(x) specified more than once!";
    v46 = 3;
    sub_C53280(&qword_4F87520, v45, 0, 0, v4);
  }
  else
  {
    byte_4F875BC = 1;
    qword_4F875A8 = (__int64)dword_4F87508;
    dword_4F875B8 = dword_4F87508[0];
  }
  sub_C53130(&qword_4F87520);
  __cxa_atexit(sub_D32600, &qword_4F87520, &qword_4A427C0);
  qword_4F87440 = (__int64)&unk_49DC150;
  v5 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F8744C &= 0x8000u;
  word_4F87450 = 0;
  qword_4F87490 = 0x100000000LL;
  qword_4F87488 = (__int64)&unk_4F87498;
  qword_4F87458 = 0;
  qword_4F87460 = 0;
  dword_4F87448 = v5;
  qword_4F87468 = 0;
  qword_4F87470 = 0;
  qword_4F87478 = 0;
  qword_4F87480 = 0;
  qword_4F874A0 = 0;
  qword_4F874A8 = (__int64)&unk_4F874C0;
  qword_4F874B0 = 1;
  dword_4F874B8 = 0;
  byte_4F874BC = 1;
  v6 = sub_C57470();
  v7 = (unsigned int)qword_4F87490;
  if ( (unsigned __int64)(unsigned int)qword_4F87490 + 1 > HIDWORD(qword_4F87490) )
  {
    v44 = v6;
    sub_C8D5F0((char *)&unk_4F87498 - 16, &unk_4F87498, (unsigned int)qword_4F87490 + 1LL, 8);
    v7 = (unsigned int)qword_4F87490;
    v6 = v44;
  }
  *(_QWORD *)(qword_4F87488 + 8 * v7) = v6;
  LODWORD(qword_4F87490) = qword_4F87490 + 1;
  byte_4F874DC = 0;
  qword_4F874D0 = (__int64)&unk_49D9728;
  qword_4F87440 = (__int64)&unk_49DDF20;
  qword_4F874C8 = 0;
  qword_4F87500 = (__int64)nullsub_186;
  qword_4F874E0 = (__int64)&unk_49DC290;
  qword_4F874F8 = (__int64)sub_D320E0;
  sub_C53080(&qword_4F87440, "force-vector-interleave", 23);
  qword_4F87470 = 60;
  LOBYTE(dword_4F8744C) = dword_4F8744C & 0x9F | 0x20;
  qword_4F87468 = (__int64)"Sets the vectorization interleave count. Zero is autoselect.";
  if ( qword_4F874C8 )
  {
    v8 = sub_CEADF0();
    v47 = 1;
    v45[0] = "cl::location(x) specified more than once!";
    v46 = 3;
    sub_C53280(&qword_4F87440, v45, 0, 0, v8);
  }
  else
  {
    byte_4F874DC = 1;
    qword_4F874C8 = (__int64)dword_4F87428;
    dword_4F874D8 = dword_4F87428[0];
  }
  sub_C53130(&qword_4F87440);
  __cxa_atexit(sub_D32600, &qword_4F87440, &qword_4A427C0);
  qword_4F87360 = (__int64)&unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F8736C &= 0x8000u;
  word_4F87370 = 0;
  qword_4F873B0 = 0x100000000LL;
  qword_4F873A8 = (__int64)&unk_4F873B8;
  qword_4F87378 = 0;
  qword_4F87380 = 0;
  dword_4F87368 = v9;
  qword_4F87388 = 0;
  qword_4F87390 = 0;
  qword_4F87398 = 0;
  qword_4F873A0 = 0;
  qword_4F873C0 = 0;
  qword_4F873C8 = (__int64)&unk_4F873E0;
  qword_4F873D0 = 1;
  dword_4F873D8 = 0;
  byte_4F873DC = 1;
  v10 = sub_C57470();
  v11 = (unsigned int)qword_4F873B0;
  if ( (unsigned __int64)(unsigned int)qword_4F873B0 + 1 > HIDWORD(qword_4F873B0) )
  {
    v41 = v10;
    sub_C8D5F0((char *)&unk_4F873B8 - 16, &unk_4F873B8, (unsigned int)qword_4F873B0 + 1LL, 8);
    v11 = (unsigned int)qword_4F873B0;
    v10 = v41;
  }
  *(_QWORD *)(qword_4F873A8 + 8 * v11) = v10;
  LODWORD(qword_4F873B0) = qword_4F873B0 + 1;
  byte_4F873FC = 0;
  qword_4F873F0 = (__int64)&unk_49D9728;
  qword_4F87360 = (__int64)&unk_49DDF20;
  qword_4F873E8 = 0;
  qword_4F87420 = (__int64)nullsub_186;
  qword_4F87400 = (__int64)&unk_49DC290;
  qword_4F87418 = (__int64)sub_D320E0;
  sub_C53080(&qword_4F87360, "runtime-memory-check-threshold", 30);
  qword_4F87390 = 123;
  LOBYTE(dword_4F8736C) = dword_4F8736C & 0x9F | 0x20;
  qword_4F87388 = (__int64)"When performing memory disambiguation checks at runtime do not generate more than this number"
                           " of comparisons (default = 8).";
  if ( qword_4F873E8 )
  {
    v12 = sub_CEADF0();
    v47 = 1;
    v45[0] = "cl::location(x) specified more than once!";
    v46 = 3;
    sub_C53280(&qword_4F87360, v45, 0, 0, v12);
  }
  else
  {
    qword_4F873E8 = (__int64)&unk_4F87348;
  }
  *(_DWORD *)qword_4F873E8 = 8;
  byte_4F873FC = 1;
  dword_4F873F8 = 8;
  sub_C53130(&qword_4F87360);
  __cxa_atexit(sub_D32600, &qword_4F87360, &qword_4A427C0);
  qword_4F87280 = (__int64)&unk_49DC150;
  v13 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F872FC = 1;
  word_4F87290 = 0;
  qword_4F872D0 = 0x100000000LL;
  dword_4F8728C &= 0x8000u;
  qword_4F87298 = 0;
  qword_4F872A0 = 0;
  dword_4F87288 = v13;
  qword_4F872A8 = 0;
  qword_4F872B0 = 0;
  qword_4F872B8 = 0;
  qword_4F872C0 = 0;
  qword_4F872C8 = (__int64)&unk_4F872D8;
  qword_4F872E0 = 0;
  qword_4F872E8 = (__int64)&unk_4F87300;
  qword_4F872F0 = 1;
  dword_4F872F8 = 0;
  v14 = sub_C57470();
  v15 = (unsigned int)qword_4F872D0;
  v16 = (unsigned int)qword_4F872D0 + 1LL;
  if ( v16 > HIDWORD(qword_4F872D0) )
  {
    sub_C8D5F0((char *)&unk_4F872D8 - 16, &unk_4F872D8, v16, 8);
    v15 = (unsigned int)qword_4F872D0;
  }
  *(_QWORD *)(qword_4F872C8 + 8 * v15) = v14;
  LODWORD(qword_4F872D0) = qword_4F872D0 + 1;
  qword_4F87308 = 0;
  qword_4F87310 = (__int64)&unk_49D9728;
  qword_4F87318 = 0;
  qword_4F87280 = (__int64)&unk_49DBF10;
  qword_4F87320 = (__int64)&unk_49DC290;
  qword_4F87340 = (__int64)nullsub_24;
  qword_4F87338 = (__int64)sub_984050;
  sub_C53080(&qword_4F87280, "memory-check-merge-threshold", 28);
  qword_4F872B0 = 94;
  LODWORD(qword_4F87308) = 100;
  BYTE4(qword_4F87318) = 1;
  LODWORD(qword_4F87318) = 100;
  LOBYTE(dword_4F8728C) = dword_4F8728C & 0x9F | 0x20;
  qword_4F872A8 = (__int64)"Maximum number of comparisons done when trying to merge runtime memory checks. (default = 100)";
  sub_C53130(&qword_4F87280);
  __cxa_atexit(sub_984970, &qword_4F87280, &qword_4A427C0);
  qword_4F871A0 = (__int64)&unk_49DC150;
  v17 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F871F0 = 0x100000000LL;
  word_4F871B0 = 0;
  dword_4F871AC &= 0x8000u;
  qword_4F871B8 = 0;
  qword_4F871C0 = 0;
  dword_4F871A8 = v17;
  qword_4F871C8 = 0;
  qword_4F871D0 = 0;
  qword_4F871D8 = 0;
  qword_4F871E0 = 0;
  qword_4F871E8 = (__int64)&unk_4F871F8;
  qword_4F87200 = 0;
  qword_4F87208 = (__int64)&unk_4F87220;
  qword_4F87210 = 1;
  dword_4F87218 = 0;
  byte_4F8721C = 1;
  v18 = sub_C57470();
  v19 = (unsigned int)qword_4F871F0;
  v20 = (unsigned int)qword_4F871F0 + 1LL;
  if ( v20 > HIDWORD(qword_4F871F0) )
  {
    sub_C8D5F0((char *)&unk_4F871F8 - 16, &unk_4F871F8, v20, 8);
    v19 = (unsigned int)qword_4F871F0;
  }
  *(_QWORD *)(qword_4F871E8 + 8 * v19) = v18;
  LODWORD(qword_4F871F0) = qword_4F871F0 + 1;
  qword_4F87228 = 0;
  qword_4F87230 = (__int64)&unk_49D9728;
  qword_4F87238 = 0;
  qword_4F871A0 = (__int64)&unk_49DBF10;
  qword_4F87240 = (__int64)&unk_49DC290;
  qword_4F87260 = (__int64)nullsub_24;
  qword_4F87258 = (__int64)sub_984050;
  sub_C53080(&qword_4F871A0, "max-dependences", 15);
  qword_4F871D0 = 79;
  LODWORD(qword_4F87228) = 100;
  BYTE4(qword_4F87238) = 1;
  LODWORD(qword_4F87238) = 100;
  LOBYTE(dword_4F871AC) = dword_4F871AC & 0x9F | 0x20;
  qword_4F871C8 = (__int64)"Maximum number of dependences collected by loop-access analysis (default = 100)";
  sub_C53130(&qword_4F871A0);
  __cxa_atexit(sub_984970, &qword_4F871A0, &qword_4A427C0);
  qword_4F870C0 = (__int64)&unk_49DC150;
  v21 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F87110 = 0x100000000LL;
  dword_4F870CC &= 0x8000u;
  word_4F870D0 = 0;
  qword_4F870D8 = 0;
  qword_4F870E0 = 0;
  dword_4F870C8 = v21;
  qword_4F870E8 = 0;
  qword_4F870F0 = 0;
  qword_4F870F8 = 0;
  qword_4F87100 = 0;
  qword_4F87108 = (__int64)&unk_4F87118;
  qword_4F87120 = 0;
  qword_4F87128 = (__int64)&unk_4F87140;
  qword_4F87130 = 1;
  dword_4F87138 = 0;
  byte_4F8713C = 1;
  v22 = sub_C57470();
  v23 = (unsigned int)qword_4F87110;
  v24 = (unsigned int)qword_4F87110 + 1LL;
  if ( v24 > HIDWORD(qword_4F87110) )
  {
    sub_C8D5F0((char *)&unk_4F87118 - 16, &unk_4F87118, v24, 8);
    v23 = (unsigned int)qword_4F87110;
  }
  *(_QWORD *)(qword_4F87108 + 8 * v23) = v22;
  qword_4F87150 = (__int64)&unk_49D9748;
  LODWORD(qword_4F87110) = qword_4F87110 + 1;
  qword_4F87148 = 0;
  qword_4F870C0 = (__int64)&unk_49DC090;
  qword_4F87160 = (__int64)&unk_49DC1D0;
  qword_4F87158 = 0;
  qword_4F87180 = (__int64)nullsub_23;
  qword_4F87178 = (__int64)sub_984030;
  sub_C53080(&qword_4F870C0, "enable-mem-access-versioning", 28);
  LOWORD(qword_4F87158) = 257;
  LOBYTE(qword_4F87148) = 1;
  qword_4F870F0 = 47;
  LOBYTE(dword_4F870CC) = dword_4F870CC & 0x9F | 0x20;
  qword_4F870E8 = (__int64)"Enable symbolic stride memory access versioning";
  sub_C53130(&qword_4F870C0);
  __cxa_atexit(sub_984900, &qword_4F870C0, &qword_4A427C0);
  qword_4F86FE0 = (__int64)&unk_49DC150;
  v25 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F87030 = 0x100000000LL;
  dword_4F86FEC &= 0x8000u;
  qword_4F87028 = (__int64)&unk_4F87038;
  word_4F86FF0 = 0;
  qword_4F86FF8 = 0;
  dword_4F86FE8 = v25;
  qword_4F87000 = 0;
  qword_4F87008 = 0;
  qword_4F87010 = 0;
  qword_4F87018 = 0;
  qword_4F87020 = 0;
  qword_4F87040 = 0;
  qword_4F87048 = (__int64)&unk_4F87060;
  qword_4F87050 = 1;
  dword_4F87058 = 0;
  byte_4F8705C = 1;
  v26 = sub_C57470();
  v27 = (unsigned int)qword_4F87030;
  if ( (unsigned __int64)(unsigned int)qword_4F87030 + 1 > HIDWORD(qword_4F87030) )
  {
    v42 = v26;
    sub_C8D5F0((char *)&unk_4F87038 - 16, &unk_4F87038, (unsigned int)qword_4F87030 + 1LL, 8);
    v27 = (unsigned int)qword_4F87030;
    v26 = v42;
  }
  *(_QWORD *)(qword_4F87028 + 8 * v27) = v26;
  qword_4F87070 = (__int64)&unk_49D9748;
  LODWORD(qword_4F87030) = qword_4F87030 + 1;
  qword_4F87068 = 0;
  qword_4F86FE0 = (__int64)&unk_49DC090;
  qword_4F87080 = (__int64)&unk_49DC1D0;
  qword_4F87078 = 0;
  qword_4F870A0 = (__int64)nullsub_23;
  qword_4F87098 = (__int64)sub_984030;
  sub_C53080(&qword_4F86FE0, "store-to-load-forwarding-conflict-detection", 43);
  qword_4F87010 = 49;
  LOWORD(qword_4F87078) = 257;
  LOBYTE(qword_4F87068) = 1;
  LOBYTE(dword_4F86FEC) = dword_4F86FEC & 0x9F | 0x20;
  qword_4F87008 = (__int64)"Enable conflict detection in loop-access analysis";
  sub_C53130(&qword_4F86FE0);
  __cxa_atexit(sub_984900, &qword_4F86FE0, &qword_4A427C0);
  qword_4F86F00 = (__int64)&unk_49DC150;
  v28 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F86F50 = 0x100000000LL;
  dword_4F86F0C &= 0x8000u;
  word_4F86F10 = 0;
  qword_4F86F48 = (__int64)&unk_4F86F58;
  qword_4F86F18 = 0;
  dword_4F86F08 = v28;
  qword_4F86F20 = 0;
  qword_4F86F28 = 0;
  qword_4F86F30 = 0;
  qword_4F86F38 = 0;
  qword_4F86F40 = 0;
  qword_4F86F60 = 0;
  qword_4F86F68 = (__int64)&unk_4F86F80;
  qword_4F86F70 = 1;
  dword_4F86F78 = 0;
  byte_4F86F7C = 1;
  v29 = sub_C57470();
  v30 = (unsigned int)qword_4F86F50;
  if ( (unsigned __int64)(unsigned int)qword_4F86F50 + 1 > HIDWORD(qword_4F86F50) )
  {
    v43 = v29;
    sub_C8D5F0((char *)&unk_4F86F58 - 16, &unk_4F86F58, (unsigned int)qword_4F86F50 + 1LL, 8);
    v30 = (unsigned int)qword_4F86F50;
    v29 = v43;
  }
  *(_QWORD *)(qword_4F86F48 + 8 * v30) = v29;
  LODWORD(qword_4F86F50) = qword_4F86F50 + 1;
  qword_4F86F88 = 0;
  qword_4F86F90 = (__int64)&unk_49D9728;
  qword_4F86F98 = 0;
  qword_4F86F00 = (__int64)&unk_49DBF10;
  qword_4F86FA0 = (__int64)&unk_49DC290;
  qword_4F86FC0 = (__int64)nullsub_24;
  qword_4F86FB8 = (__int64)sub_984050;
  sub_C53080(&qword_4F86F00, "max-forked-scev-depth", 21);
  qword_4F86F30 = 63;
  LODWORD(qword_4F86F88) = 5;
  BYTE4(qword_4F86F98) = 1;
  LODWORD(qword_4F86F98) = 5;
  LOBYTE(dword_4F86F0C) = dword_4F86F0C & 0x9F | 0x20;
  qword_4F86F28 = (__int64)"Maximum recursion depth when finding forked SCEVs (default = 5)";
  sub_C53130(&qword_4F86F00);
  __cxa_atexit(sub_984970, &qword_4F86F00, &qword_4A427C0);
  qword_4F86E20 = (__int64)&unk_49DC150;
  v31 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F86E9C = 1;
  word_4F86E30 = 0;
  qword_4F86E70 = 0x100000000LL;
  dword_4F86E2C &= 0x8000u;
  qword_4F86E68 = (__int64)&unk_4F86E78;
  qword_4F86E38 = 0;
  dword_4F86E28 = v31;
  qword_4F86E40 = 0;
  qword_4F86E48 = 0;
  qword_4F86E50 = 0;
  qword_4F86E58 = 0;
  qword_4F86E60 = 0;
  qword_4F86E80 = 0;
  qword_4F86E88 = (__int64)&unk_4F86EA0;
  qword_4F86E90 = 1;
  dword_4F86E98 = 0;
  v32 = sub_C57470();
  v33 = (unsigned int)qword_4F86E70;
  v34 = (unsigned int)qword_4F86E70 + 1LL;
  if ( v34 > HIDWORD(qword_4F86E70) )
  {
    sub_C8D5F0((char *)&unk_4F86E78 - 16, &unk_4F86E78, v34, 8);
    v33 = (unsigned int)qword_4F86E70;
  }
  *(_QWORD *)(qword_4F86E68 + 8 * v33) = v32;
  qword_4F86EB0 = (__int64)&unk_49D9748;
  LODWORD(qword_4F86E70) = qword_4F86E70 + 1;
  qword_4F86EA8 = 0;
  qword_4F86E20 = (__int64)&unk_49DC090;
  qword_4F86EC0 = (__int64)&unk_49DC1D0;
  qword_4F86EB8 = 0;
  qword_4F86EE0 = (__int64)nullsub_23;
  qword_4F86ED8 = (__int64)sub_984030;
  sub_C53080(&qword_4F86E20, "laa-speculate-unit-stride", 25);
  LOWORD(qword_4F86EB8) = 257;
  LOBYTE(qword_4F86EA8) = 1;
  qword_4F86E50 = 51;
  LOBYTE(dword_4F86E2C) = dword_4F86E2C & 0x9F | 0x20;
  qword_4F86E48 = (__int64)"Speculate that non-constant strides are unit in LAA";
  sub_C53130(&qword_4F86E20);
  __cxa_atexit(sub_984900, &qword_4F86E20, &qword_4A427C0);
  qword_4F86D40 = (__int64)&unk_49DC150;
  v35 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F86D90 = 0x100000000LL;
  word_4F86D50 = 0;
  dword_4F86D4C &= 0x8000u;
  qword_4F86D58 = 0;
  qword_4F86D60 = 0;
  dword_4F86D48 = v35;
  qword_4F86D68 = 0;
  qword_4F86D70 = 0;
  qword_4F86D78 = 0;
  qword_4F86D80 = 0;
  qword_4F86D88 = (__int64)&unk_4F86D98;
  qword_4F86DA0 = 0;
  qword_4F86DA8 = (__int64)&unk_4F86DC0;
  qword_4F86DB0 = 1;
  dword_4F86DB8 = 0;
  byte_4F86DBC = 1;
  v36 = sub_C57470();
  v37 = (unsigned int)qword_4F86D90;
  v38 = (unsigned int)qword_4F86D90 + 1LL;
  if ( v38 > HIDWORD(qword_4F86D90) )
  {
    sub_C8D5F0((char *)&unk_4F86D98 - 16, &unk_4F86D98, v38, 8);
    v37 = (unsigned int)qword_4F86D90;
  }
  *(_QWORD *)(qword_4F86D88 + 8 * v37) = v36;
  qword_4F86DD0 = (__int64)&unk_49D9748;
  LODWORD(qword_4F86D90) = qword_4F86D90 + 1;
  byte_4F86DD9 = 0;
  qword_4F86D40 = (__int64)&unk_49D9AD8;
  qword_4F86DE0 = (__int64)&unk_49DC1D0;
  qword_4F86DC8 = 0;
  qword_4F86E00 = (__int64)nullsub_39;
  qword_4F86DF8 = (__int64)sub_AA4180;
  sub_C53080(&qword_4F86D40, "hoist-runtime-checks", 20);
  qword_4F86D70 = 64;
  LOBYTE(dword_4F86D4C) = dword_4F86D4C & 0x9F | 0x20;
  qword_4F86D68 = (__int64)"Hoist inner loop runtime memory checks to outer loop if possible";
  if ( qword_4F86DC8 )
  {
    v39 = sub_CEADF0();
    v47 = 1;
    v45[0] = "cl::location(x) specified more than once!";
    v46 = 3;
    sub_C53280(&qword_4F86D40, v45, 0, 0, v39);
  }
  else
  {
    qword_4F86DC8 = (__int64)&unk_4F86D30;
  }
  *(_BYTE *)qword_4F86DC8 = 1;
  unk_4F86DD8 = 257;
  sub_C53130(&qword_4F86D40);
  return __cxa_atexit(sub_AA4490, &qword_4F86D40, &qword_4A427C0);
}
