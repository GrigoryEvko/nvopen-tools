// Function: ctor_652_0
// Address: 0x599ef0
//
int ctor_652_0()
{
  __int64 v0; // rdx
  __int64 v1; // rcx
  int v2; // edx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // rcx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rcx
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rcx
  int v17; // edx
  __int64 v18; // rbx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rcx
  int v23; // edx
  __int64 v24; // rbx
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rcx
  int v29; // edx
  __int64 v30; // rbx
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v34; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+8h] [rbp-58h]
  __int64 v36; // [rsp+8h] [rbp-58h]
  char v37; // [rsp+13h] [rbp-4Dh] BYREF
  int v38; // [rsp+14h] [rbp-4Ch] BYREF
  char *v39; // [rsp+18h] [rbp-48h] BYREF
  const char *v40; // [rsp+20h] [rbp-40h] BYREF
  __int64 v41; // [rsp+28h] [rbp-38h]

  sub_334D620(&unk_5039020, "list-burr", "Bottom-up register reduction list scheduling", sub_3355700);
  __cxa_atexit(sub_334CAC0, &unk_5039020, &qword_4A427C0);
  sub_334D620(&unk_5038FE0, "source", "Similar to list-burr but schedules in source order when possible", sub_33553F0);
  __cxa_atexit(sub_334CAC0, &unk_5038FE0, &qword_4A427C0);
  sub_334D620(
    &unk_5038FA0,
    "list-hybrid",
    "Bottom-up register pressure aware list scheduling which tries to balance latency and register pressure",
    sub_3354FA0);
  __cxa_atexit(sub_334CAC0, &unk_5038FA0, &qword_4A427C0);
  sub_334D620(
    &unk_5038F60,
    "list-ilp",
    "Bottom-up register pressure aware list scheduling which tries to balance ILP and register pressure",
    sub_3355A10);
  __cxa_atexit(sub_334CAC0, &unk_5038F60, &qword_4A427C0);
  v37 = 0;
  v40 = "Disable cycle-level precision during preRA scheduling";
  v41 = 53;
  v39 = &v37;
  v38 = 1;
  sub_3354790(&unk_5038E80, "disable-sched-cycles", &v38, &v39, &v40);
  __cxa_atexit(sub_984900, &unk_5038E80, &qword_4A427C0);
  v37 = 0;
  v40 = "Disable regpressure priority in sched=list-ilp";
  v41 = 46;
  v39 = &v37;
  v38 = 1;
  sub_26C1EA0(&unk_5038DA0, "disable-sched-reg-pressure", &v38, &v39, &v40);
  __cxa_atexit(sub_984900, &unk_5038DA0, &qword_4A427C0);
  qword_5038CC0 = (__int64)&unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &unk_5038DA0, v0, v1), 1u);
  dword_5038CCC &= 0x8000u;
  word_5038CD0 = 0;
  qword_5038D10 = 0x100000000LL;
  qword_5038CD8 = 0;
  qword_5038CE0 = 0;
  qword_5038CE8 = 0;
  dword_5038CC8 = v2;
  qword_5038CF0 = 0;
  qword_5038CF8 = 0;
  qword_5038D00 = 0;
  qword_5038D08 = (__int64)&unk_5038D18;
  qword_5038D20 = 0;
  qword_5038D28 = (__int64)&unk_5038D40;
  qword_5038D30 = 1;
  dword_5038D38 = 0;
  byte_5038D3C = 1;
  v3 = sub_C57470();
  v4 = (unsigned int)qword_5038D10;
  if ( (unsigned __int64)(unsigned int)qword_5038D10 + 1 > HIDWORD(qword_5038D10) )
  {
    v34 = v3;
    sub_C8D5F0((char *)&unk_5038D18 - 16, &unk_5038D18, (unsigned int)qword_5038D10 + 1LL, 8);
    v4 = (unsigned int)qword_5038D10;
    v3 = v34;
  }
  *(_QWORD *)(qword_5038D08 + 8 * v4) = v3;
  LODWORD(qword_5038D10) = qword_5038D10 + 1;
  qword_5038D48 = 0;
  qword_5038D50 = (__int64)&unk_49D9748;
  qword_5038D58 = 0;
  qword_5038CC0 = (__int64)&unk_49DC090;
  qword_5038D60 = (__int64)&unk_49DC1D0;
  qword_5038D80 = (__int64)nullsub_23;
  qword_5038D78 = (__int64)sub_984030;
  sub_C53080(&qword_5038CC0, "disable-sched-live-uses", 23);
  LOWORD(qword_5038D58) = 257;
  LOBYTE(qword_5038D48) = 1;
  qword_5038CF0 = 43;
  LOBYTE(dword_5038CCC) = dword_5038CCC & 0x9F | 0x20;
  qword_5038CE8 = (__int64)"Disable live use priority in sched=list-ilp";
  sub_C53130(&qword_5038CC0);
  __cxa_atexit(sub_984900, &qword_5038CC0, &qword_4A427C0);
  qword_5038BE0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5038CC0, v5, v6), 1u);
  qword_5038C30 = 0x100000000LL;
  dword_5038BEC &= 0x8000u;
  qword_5038C28 = (__int64)&unk_5038C38;
  word_5038BF0 = 0;
  qword_5038BF8 = 0;
  dword_5038BE8 = v7;
  qword_5038C00 = 0;
  qword_5038C08 = 0;
  qword_5038C10 = 0;
  qword_5038C18 = 0;
  qword_5038C20 = 0;
  qword_5038C40 = 0;
  qword_5038C48 = (__int64)&unk_5038C60;
  qword_5038C50 = 1;
  dword_5038C58 = 0;
  byte_5038C5C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_5038C30;
  if ( (unsigned __int64)(unsigned int)qword_5038C30 + 1 > HIDWORD(qword_5038C30) )
  {
    v35 = v8;
    sub_C8D5F0((char *)&unk_5038C38 - 16, &unk_5038C38, (unsigned int)qword_5038C30 + 1LL, 8);
    v9 = (unsigned int)qword_5038C30;
    v8 = v35;
  }
  *(_QWORD *)(qword_5038C28 + 8 * v9) = v8;
  LODWORD(qword_5038C30) = qword_5038C30 + 1;
  qword_5038C68 = 0;
  qword_5038C70 = (__int64)&unk_49D9748;
  qword_5038C78 = 0;
  qword_5038BE0 = (__int64)&unk_49DC090;
  qword_5038C80 = (__int64)&unk_49DC1D0;
  qword_5038CA0 = (__int64)nullsub_23;
  qword_5038C98 = (__int64)sub_984030;
  sub_C53080(&qword_5038BE0, "disable-sched-vrcycle", 21);
  LOWORD(qword_5038C78) = 256;
  LOBYTE(qword_5038C68) = 0;
  qword_5038C10 = 50;
  LOBYTE(dword_5038BEC) = dword_5038BEC & 0x9F | 0x20;
  qword_5038C08 = (__int64)"Disable virtual register cycle interference checks";
  sub_C53130(&qword_5038BE0);
  __cxa_atexit(sub_984900, &qword_5038BE0, &qword_4A427C0);
  v37 = 0;
  v40 = "Disable physreg def-use affinity";
  v41 = 32;
  v39 = &v37;
  v38 = 1;
  sub_26C1EA0(&unk_5038B00, "disable-sched-physreg-join", &v38, &v39, &v40);
  __cxa_atexit(sub_984900, &unk_5038B00, &qword_4A427C0);
  v37 = 1;
  v40 = "Disable no-stall priority in sched=list-ilp";
  v41 = 43;
  v39 = &v37;
  v38 = 1;
  sub_3354790(&unk_5038A20, "disable-sched-stalls", &v38, &v39, &v40);
  __cxa_atexit(sub_984900, &unk_5038A20, &qword_4A427C0);
  qword_5038940 = (__int64)&unk_49DC150;
  v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &unk_5038A20, v10, v11), 1u);
  qword_5038990 = 0x100000000LL;
  dword_503894C &= 0x8000u;
  qword_5038988 = (__int64)&unk_5038998;
  word_5038950 = 0;
  qword_5038958 = 0;
  dword_5038948 = v12;
  qword_5038960 = 0;
  qword_5038968 = 0;
  qword_5038970 = 0;
  qword_5038978 = 0;
  qword_5038980 = 0;
  qword_50389A0 = 0;
  qword_50389A8 = (__int64)&unk_50389C0;
  qword_50389B0 = 1;
  dword_50389B8 = 0;
  byte_50389BC = 1;
  v13 = sub_C57470();
  v14 = (unsigned int)qword_5038990;
  if ( (unsigned __int64)(unsigned int)qword_5038990 + 1 > HIDWORD(qword_5038990) )
  {
    v36 = v13;
    sub_C8D5F0((char *)&unk_5038998 - 16, &unk_5038998, (unsigned int)qword_5038990 + 1LL, 8);
    v14 = (unsigned int)qword_5038990;
    v13 = v36;
  }
  *(_QWORD *)(qword_5038988 + 8 * v14) = v13;
  LODWORD(qword_5038990) = qword_5038990 + 1;
  qword_50389C8 = 0;
  qword_50389D0 = (__int64)&unk_49D9748;
  qword_50389D8 = 0;
  qword_5038940 = (__int64)&unk_49DC090;
  qword_50389E0 = (__int64)&unk_49DC1D0;
  qword_5038A00 = (__int64)nullsub_23;
  qword_50389F8 = (__int64)sub_984030;
  sub_C53080(&qword_5038940, "disable-sched-critical-path", 27);
  LOWORD(qword_50389D8) = 256;
  LOBYTE(qword_50389C8) = 0;
  qword_5038970 = 48;
  LOBYTE(dword_503894C) = dword_503894C & 0x9F | 0x20;
  qword_5038968 = (__int64)"Disable critical path priority in sched=list-ilp";
  sub_C53130(&qword_5038940);
  __cxa_atexit(sub_984900, &qword_5038940, &qword_4A427C0);
  v37 = 0;
  v40 = "Disable scheduled-height priority in sched=list-ilp";
  v41 = 51;
  v39 = &v37;
  v38 = 1;
  sub_3354790(&unk_5038860, "disable-sched-height", &v38, &v39, &v40);
  __cxa_atexit(sub_984900, &unk_5038860, &qword_4A427C0);
  qword_5038780 = (__int64)&unk_49DC150;
  v17 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &unk_5038860, v15, v16), 1u);
  qword_50387D0 = 0x100000000LL;
  dword_503878C &= 0x8000u;
  word_5038790 = 0;
  qword_5038798 = 0;
  qword_50387A0 = 0;
  dword_5038788 = v17;
  qword_50387A8 = 0;
  qword_50387B0 = 0;
  qword_50387B8 = 0;
  qword_50387C0 = 0;
  qword_50387C8 = (__int64)&unk_50387D8;
  qword_50387E0 = 0;
  qword_50387E8 = (__int64)&unk_5038800;
  qword_50387F0 = 1;
  dword_50387F8 = 0;
  byte_50387FC = 1;
  v18 = sub_C57470();
  v19 = (unsigned int)qword_50387D0;
  v20 = (unsigned int)qword_50387D0 + 1LL;
  if ( v20 > HIDWORD(qword_50387D0) )
  {
    sub_C8D5F0((char *)&unk_50387D8 - 16, &unk_50387D8, v20, 8);
    v19 = (unsigned int)qword_50387D0;
  }
  *(_QWORD *)(qword_50387C8 + 8 * v19) = v18;
  LODWORD(qword_50387D0) = qword_50387D0 + 1;
  qword_5038808 = 0;
  qword_5038810 = (__int64)&unk_49D9748;
  qword_5038818 = 0;
  qword_5038780 = (__int64)&unk_49DC090;
  qword_5038820 = (__int64)&unk_49DC1D0;
  qword_5038840 = (__int64)nullsub_23;
  qword_5038838 = (__int64)sub_984030;
  sub_C53080(&qword_5038780, "disable-2addr-hack", 18);
  LOWORD(qword_5038818) = 257;
  LOBYTE(qword_5038808) = 1;
  qword_50387B0 = 36;
  LOBYTE(dword_503878C) = dword_503878C & 0x9F | 0x20;
  qword_50387A8 = (__int64)"Disable scheduler's two-address hack";
  sub_C53130(&qword_5038780);
  __cxa_atexit(sub_984900, &qword_5038780, &qword_4A427C0);
  qword_50386A0 = (__int64)&unk_49DC150;
  v23 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5038780, v21, v22), 1u);
  qword_50386F0 = 0x100000000LL;
  word_50386B0 = 0;
  dword_50386AC &= 0x8000u;
  qword_50386B8 = 0;
  qword_50386C0 = 0;
  dword_50386A8 = v23;
  qword_50386C8 = 0;
  qword_50386D0 = 0;
  qword_50386D8 = 0;
  qword_50386E0 = 0;
  qword_50386E8 = (__int64)&unk_50386F8;
  qword_5038700 = 0;
  qword_5038708 = (__int64)&unk_5038720;
  qword_5038710 = 1;
  dword_5038718 = 0;
  byte_503871C = 1;
  v24 = sub_C57470();
  v25 = (unsigned int)qword_50386F0;
  v26 = (unsigned int)qword_50386F0 + 1LL;
  if ( v26 > HIDWORD(qword_50386F0) )
  {
    sub_C8D5F0((char *)&unk_50386F8 - 16, &unk_50386F8, v26, 8);
    v25 = (unsigned int)qword_50386F0;
  }
  *(_QWORD *)(qword_50386E8 + 8 * v25) = v24;
  LODWORD(qword_50386F0) = qword_50386F0 + 1;
  qword_5038728 = 0;
  qword_5038730 = (__int64)&unk_49DA090;
  qword_5038738 = 0;
  qword_50386A0 = (__int64)&unk_49DBF90;
  qword_5038740 = (__int64)&unk_49DC230;
  qword_5038760 = (__int64)nullsub_58;
  qword_5038758 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_50386A0, "max-sched-reorder", 17);
  LODWORD(qword_5038728) = 6;
  BYTE4(qword_5038738) = 1;
  LODWORD(qword_5038738) = 6;
  qword_50386D0 = 76;
  LOBYTE(dword_50386AC) = dword_50386AC & 0x9F | 0x20;
  qword_50386C8 = (__int64)"Number of instructions to allow ahead of the critical path in sched=list-ilp";
  sub_C53130(&qword_50386A0);
  __cxa_atexit(sub_B2B680, &qword_50386A0, &qword_4A427C0);
  qword_50385C0 = (__int64)&unk_49DC150;
  v29 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_B2B680, &qword_50386A0, v27, v28), 1u);
  dword_50385CC &= 0x8000u;
  word_50385D0 = 0;
  qword_5038610 = 0x100000000LL;
  qword_50385D8 = 0;
  qword_50385E0 = 0;
  qword_50385E8 = 0;
  dword_50385C8 = v29;
  qword_50385F0 = 0;
  qword_50385F8 = 0;
  qword_5038600 = 0;
  qword_5038608 = (__int64)&unk_5038618;
  qword_5038620 = 0;
  qword_5038628 = (__int64)&unk_5038640;
  qword_5038630 = 1;
  dword_5038638 = 0;
  byte_503863C = 1;
  v30 = sub_C57470();
  v31 = (unsigned int)qword_5038610;
  v32 = (unsigned int)qword_5038610 + 1LL;
  if ( v32 > HIDWORD(qword_5038610) )
  {
    sub_C8D5F0((char *)&unk_5038618 - 16, &unk_5038618, v32, 8);
    v31 = (unsigned int)qword_5038610;
  }
  *(_QWORD *)(qword_5038608 + 8 * v31) = v30;
  LODWORD(qword_5038610) = qword_5038610 + 1;
  qword_5038648 = 0;
  qword_5038650 = (__int64)&unk_49D9728;
  qword_5038658 = 0;
  qword_50385C0 = (__int64)&unk_49DBF10;
  qword_5038660 = (__int64)&unk_49DC290;
  qword_5038680 = (__int64)nullsub_24;
  qword_5038678 = (__int64)sub_984050;
  sub_C53080(&qword_50385C0, "sched-avg-ipc", 13);
  LODWORD(qword_5038648) = 1;
  BYTE4(qword_5038658) = 1;
  LODWORD(qword_5038658) = 1;
  qword_50385F0 = 51;
  LOBYTE(dword_50385CC) = dword_50385CC & 0x9F | 0x20;
  qword_50385E8 = (__int64)"Average inst/cycle when no target itinerary exists.";
  sub_C53130(&qword_50385C0);
  return __cxa_atexit(sub_984970, &qword_50385C0, &qword_4A427C0);
}
