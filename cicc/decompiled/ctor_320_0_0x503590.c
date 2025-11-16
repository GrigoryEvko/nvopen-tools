// Function: ctor_320_0
// Address: 0x503590
//
int ctor_320_0()
{
  int v0; // eax
  int v1; // edx
  int v2; // edx
  int v3; // r8d
  int v4; // r11d
  int v5; // r9d
  int v6; // r8d
  int v7; // eax
  int v9; // [rsp+30h] [rbp-100h] BYREF
  int v10; // [rsp+34h] [rbp-FCh] BYREF
  int *v11; // [rsp+38h] [rbp-F8h] BYREF
  _QWORD v12[2]; // [rsp+40h] [rbp-F0h] BYREF
  _QWORD v13[2]; // [rsp+50h] [rbp-E0h] BYREF
  _QWORD v14[2]; // [rsp+60h] [rbp-D0h] BYREF
  int v15; // [rsp+70h] [rbp-C0h]
  char *v16; // [rsp+78h] [rbp-B8h]
  __int64 v17; // [rsp+80h] [rbp-B0h]
  char *v18; // [rsp+88h] [rbp-A8h]
  __int64 v19; // [rsp+90h] [rbp-A0h]
  int v20; // [rsp+98h] [rbp-98h]
  const char *v21; // [rsp+A0h] [rbp-90h]
  __int64 v22; // [rsp+A8h] [rbp-88h]
  char *v23; // [rsp+B0h] [rbp-80h]
  __int64 v24; // [rsp+B8h] [rbp-78h]
  int v25; // [rsp+C0h] [rbp-70h]
  const char *v26; // [rsp+C8h] [rbp-68h]
  __int64 v27; // [rsp+D0h] [rbp-60h]

  v11 = &v10;
  v14[0] = "default";
  v16 = "Default";
  v18 = "size";
  v21 = "Optimize for size";
  v23 = "speed";
  v26 = "Optimize for speed";
  v13[1] = 0x400000003LL;
  v10 = 2;
  v13[0] = v14;
  v14[1] = 7;
  v15 = 0;
  v17 = 7;
  v19 = 4;
  v20 = 1;
  v22 = 17;
  v24 = 5;
  v25 = 2;
  v27 = 18;
  v12[0] = "Spill mode for splitting live ranges";
  v12[1] = 36;
  v9 = 1;
  sub_1EC0540(&unk_4FC9920, "split-spill-mode", &v9, v12, v13, &v11);
  if ( (_QWORD *)v13[0] != v14 )
    _libc_free(v13[0], "split-spill-mode");
  __cxa_atexit(sub_1EBB0F0, &unk_4FC9920, &qword_4A427C0);
  qword_4FC9840 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC984C &= 0xF000u;
  qword_4FC9888 = (__int64)qword_4FA01C0;
  qword_4FC9850 = 0;
  qword_4FC9858 = 0;
  qword_4FC9860 = 0;
  dword_4FC9848 = v0;
  qword_4FC9898 = (__int64)&unk_4FC98B8;
  qword_4FC98A0 = (__int64)&unk_4FC98B8;
  qword_4FC9868 = 0;
  qword_4FC9870 = 0;
  qword_4FC98E8 = (__int64)&unk_49E74A8;
  qword_4FC9878 = 0;
  qword_4FC9840 = (__int64)&unk_49EEAF0;
  qword_4FC9880 = 0;
  qword_4FC9890 = 0;
  qword_4FC98F8 = (__int64)&unk_49EEE10;
  qword_4FC98A8 = 4;
  dword_4FC98B0 = 0;
  byte_4FC98D8 = 0;
  dword_4FC98E0 = 0;
  byte_4FC98F4 = 1;
  dword_4FC98F0 = 0;
  sub_16B8280(&qword_4FC9840, "lcr-max-depth", 13);
  qword_4FC9870 = 32;
  qword_4FC9868 = (__int64)"Last chance recoloring max depth";
  dword_4FC98E0 = 5;
  byte_4FC98F4 = 1;
  dword_4FC98F0 = 5;
  LOBYTE(word_4FC984C) = word_4FC984C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC9840);
  __cxa_atexit(sub_12EDE60, &qword_4FC9840, &qword_4A427C0);
  qword_4FC9760 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  byte_4FC97F8 = 0;
  qword_4FC97B8 = (__int64)&unk_4FC97D8;
  qword_4FC97C0 = (__int64)&unk_4FC97D8;
  word_4FC976C &= 0xF000u;
  qword_4FC9818 = (__int64)&unk_49EEE10;
  qword_4FC9760 = (__int64)&unk_49EEAF0;
  dword_4FC9768 = v1;
  qword_4FC97A8 = (__int64)qword_4FA01C0;
  qword_4FC9808 = (__int64)&unk_49E74A8;
  qword_4FC9770 = 0;
  qword_4FC9778 = 0;
  qword_4FC9780 = 0;
  qword_4FC9788 = 0;
  qword_4FC9790 = 0;
  qword_4FC9798 = 0;
  qword_4FC97A0 = 0;
  qword_4FC97B0 = 0;
  qword_4FC97C8 = 4;
  dword_4FC97D0 = 0;
  dword_4FC9800 = 0;
  byte_4FC9814 = 1;
  dword_4FC9810 = 0;
  sub_16B8280((char *)&unk_4FC97D8 - 120, "lcr-max-interf", 14);
  qword_4FC9790 = 74;
  dword_4FC9800 = 8;
  byte_4FC9814 = 1;
  dword_4FC9810 = 8;
  qword_4FC9788 = (__int64)"Last chance recoloring maximum number of considered interference at a time";
  LOBYTE(word_4FC976C) = word_4FC976C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC9760);
  __cxa_atexit(sub_12EDE60, &qword_4FC9760, &qword_4A427C0);
  qword_4FC9680 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC968C &= 0xF000u;
  qword_4FC96D8 = (__int64)&unk_4FC96F8;
  qword_4FC96E0 = (__int64)&unk_4FC96F8;
  qword_4FC9690 = 0;
  qword_4FC9698 = 0;
  qword_4FC96A0 = 0;
  dword_4FC9688 = v2;
  word_4FC9730 = 256;
  qword_4FC9728 = (__int64)&unk_49E74E8;
  qword_4FC9680 = (__int64)&unk_49EEC70;
  qword_4FC9738 = (__int64)&unk_49EEDB0;
  qword_4FC96C8 = (__int64)qword_4FA01C0;
  qword_4FC96A8 = 0;
  qword_4FC96B0 = 0;
  qword_4FC96B8 = 0;
  qword_4FC96C0 = 0;
  qword_4FC96D0 = 0;
  qword_4FC96E8 = 4;
  dword_4FC96F0 = 0;
  byte_4FC9718 = 0;
  byte_4FC9720 = 0;
  sub_16B8280(&qword_4FC9680, "exhaustive-register-search", 26);
  qword_4FC96A8 = (__int64)"Exhaustive Search for registers bypassing the depth and interference cutoffs of last chance recoloring";
  qword_4FC96B0 = 102;
  LOBYTE(word_4FC968C) = word_4FC968C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC9680);
  __cxa_atexit(sub_12EDEC0, &qword_4FC9680, &qword_4A427C0);
  qword_4FC95A0 = (__int64)&unk_49EED30;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC95AC &= 0xF000u;
  qword_4FC95F8 = (__int64)&unk_4FC9618;
  qword_4FC9600 = (__int64)&unk_4FC9618;
  qword_4FC95B0 = 0;
  qword_4FC9648 = (__int64)&unk_49E74E8;
  word_4FC9650 = 256;
  qword_4FC95A0 = (__int64)&unk_49EEC70;
  dword_4FC95A8 = v3;
  qword_4FC9658 = (__int64)&unk_49EEDB0;
  qword_4FC95E8 = (__int64)qword_4FA01C0;
  qword_4FC95B8 = 0;
  qword_4FC95C0 = 0;
  qword_4FC95C8 = 0;
  qword_4FC95D0 = 0;
  qword_4FC95D8 = 0;
  qword_4FC95E0 = 0;
  qword_4FC95F0 = 0;
  qword_4FC9608 = 4;
  dword_4FC9610 = 0;
  byte_4FC9638 = 0;
  byte_4FC9640 = 0;
  sub_16B8280((char *)&unk_4FC9618 - 120, "enable-local-reassign", 21);
  word_4FC9650 = 256;
  byte_4FC9640 = 0;
  qword_4FC95D0 = 91;
  qword_4FC95C8 = (__int64)"Local reassignment can yield better allocation decisions, but may be compile time intensive";
  LOBYTE(word_4FC95AC) = word_4FC95AC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC95A0);
  __cxa_atexit(sub_12EDEC0, &qword_4FC95A0, &qword_4A427C0);
  qword_4FC94C0 = (__int64)&unk_49EED30;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC94CC &= 0xF000u;
  word_4FC9570 = 256;
  qword_4FC9518 = (__int64)&unk_4FC9538;
  qword_4FC9520 = (__int64)&unk_4FC9538;
  qword_4FC9568 = (__int64)&unk_49E74E8;
  qword_4FC94C0 = (__int64)&unk_49EEC70;
  qword_4FC9578 = (__int64)&unk_49EEDB0;
  dword_4FC94C8 = v4;
  qword_4FC9508 = (__int64)qword_4FA01C0;
  qword_4FC94D0 = 0;
  qword_4FC94D8 = 0;
  qword_4FC94E0 = 0;
  qword_4FC94E8 = 0;
  qword_4FC94F0 = 0;
  qword_4FC94F8 = 0;
  qword_4FC9500 = 0;
  qword_4FC9510 = 0;
  qword_4FC9528 = 4;
  dword_4FC9530 = 0;
  byte_4FC9558 = 0;
  byte_4FC9560 = 0;
  sub_16B8280((char *)&unk_4FC9538 - 120, "enable-deferred-spilling", 24);
  word_4FC9570 = 256;
  qword_4FC94E8 = (__int64)"Instead of spilling a variable right away, defer the actual code insertion to the end of the "
                           "allocation. That way the allocator might still find a suitable coloring for this variable bec"
                           "ause of other evicted variables.";
  byte_4FC9560 = 0;
  qword_4FC94F0 = 218;
  LOBYTE(word_4FC94CC) = word_4FC94CC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC94C0);
  __cxa_atexit(sub_12EDEC0, &qword_4FC94C0, &qword_4A427C0);
  qword_4FC93E0 = (__int64)&unk_49EED30;
  v5 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_4FC9438 = (__int64)&unk_4FC9458;
  qword_4FC9440 = (__int64)&unk_4FC9458;
  word_4FC93EC &= 0xF000u;
  qword_4FC9498 = (__int64)&unk_49EEE10;
  qword_4FC93E0 = (__int64)&unk_49EEAF0;
  dword_4FC93E8 = v5;
  qword_4FC9428 = (__int64)qword_4FA01C0;
  qword_4FC9488 = (__int64)&unk_49E74A8;
  qword_4FC93F0 = 0;
  qword_4FC93F8 = 0;
  qword_4FC9400 = 0;
  qword_4FC9408 = 0;
  qword_4FC9410 = 0;
  qword_4FC9418 = 0;
  qword_4FC9420 = 0;
  qword_4FC9430 = 0;
  qword_4FC9448 = 4;
  dword_4FC9450 = 0;
  byte_4FC9478 = 0;
  dword_4FC9480 = 0;
  byte_4FC9494 = 1;
  dword_4FC9490 = 0;
  sub_16B8280(&qword_4FC93E0, "huge-size-for-split", 19);
  qword_4FC9408 = (__int64)"A threshold of live range size which may cause high compile time cost in global splitting.";
  byte_4FC9494 = 1;
  qword_4FC9410 = 90;
  dword_4FC9480 = 5000;
  LOBYTE(word_4FC93EC) = word_4FC93EC & 0x9F | 0x20;
  dword_4FC9490 = 5000;
  sub_16B88A0(&qword_4FC93E0);
  __cxa_atexit(sub_12EDE60, &qword_4FC93E0, &qword_4A427C0);
  qword_4FC9300 = (__int64)&unk_49EED30;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_4FC9358 = (__int64)&unk_4FC9378;
  qword_4FC9360 = (__int64)&unk_4FC9378;
  word_4FC930C &= 0xF000u;
  qword_4FC93B8 = (__int64)&unk_49EEE10;
  qword_4FC9300 = (__int64)&unk_49EEAF0;
  dword_4FC9308 = v6;
  qword_4FC9348 = (__int64)qword_4FA01C0;
  qword_4FC93A8 = (__int64)&unk_49E74A8;
  qword_4FC9310 = 0;
  qword_4FC9318 = 0;
  qword_4FC9320 = 0;
  qword_4FC9328 = 0;
  qword_4FC9330 = 0;
  qword_4FC9338 = 0;
  qword_4FC9340 = 0;
  qword_4FC9350 = 0;
  qword_4FC9368 = 4;
  dword_4FC9370 = 0;
  byte_4FC9398 = 0;
  dword_4FC93A0 = 0;
  byte_4FC93B4 = 1;
  dword_4FC93B0 = 0;
  sub_16B8280(&qword_4FC9300, "regalloc-csr-first-time-cost", 28);
  qword_4FC9330 = 49;
  qword_4FC9328 = (__int64)"Cost for first time use of callee-saved register.";
  dword_4FC93A0 = 0;
  byte_4FC93B4 = 1;
  dword_4FC93B0 = 0;
  LOBYTE(word_4FC930C) = word_4FC930C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC9300);
  __cxa_atexit(sub_12EDE60, &qword_4FC9300, &qword_4A427C0);
  qword_4FC9220 = (__int64)&unk_49EED30;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC922C &= 0xF000u;
  word_4FC92D0 = 256;
  qword_4FC9230 = 0;
  qword_4FC9238 = 0;
  qword_4FC92C8 = (__int64)&unk_49E74E8;
  qword_4FC9220 = (__int64)&unk_49EEC70;
  dword_4FC9228 = v7;
  qword_4FC92D8 = (__int64)&unk_49EEDB0;
  qword_4FC9268 = (__int64)qword_4FA01C0;
  qword_4FC9278 = (__int64)&unk_4FC9298;
  qword_4FC9280 = (__int64)&unk_4FC9298;
  qword_4FC9240 = 0;
  qword_4FC9248 = 0;
  qword_4FC9250 = 0;
  qword_4FC9258 = 0;
  qword_4FC9260 = 0;
  qword_4FC9270 = 0;
  qword_4FC9288 = 4;
  dword_4FC9290 = 0;
  byte_4FC92B8 = 0;
  byte_4FC92C0 = 0;
  sub_16B8280((char *)&unk_4FC9298 - 120, "condsider-local-interval-cost", 29);
  word_4FC92D0 = 256;
  byte_4FC92C0 = 0;
  qword_4FC9250 = 105;
  LOBYTE(word_4FC922C) = word_4FC922C & 0x9F | 0x20;
  qword_4FC9248 = (__int64)"Consider the cost of local intervals created by a split candidate when choosing the best split candidate.";
  sub_16B88A0(&qword_4FC9220);
  __cxa_atexit(sub_12EDEC0, &qword_4FC9220, &qword_4A427C0);
  qword_4FC91E8 = (__int64)"greedy";
  qword_4FC91F8 = (__int64)"greedy register allocator";
  qword_4FC91E0 = 0;
  qword_4FC91F0 = 6;
  qword_4FC9200 = 25;
  qword_4FC9208 = (__int64)sub_1EBDCD0;
  sub_1E40390(&unk_4FCB760, &qword_4FC91E0);
  return __cxa_atexit(sub_1EB3C00, &qword_4FC91E0, &qword_4A427C0);
}
