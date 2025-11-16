// Function: ctor_392_0
// Address: 0x51e600
//
int ctor_392_0()
{
  int v0; // edx
  int v1; // r8d
  int v2; // r9d
  int v3; // edx
  int v4; // r8d
  int v5; // r9d
  __int128 v7; // [rsp-50h] [rbp-240h]
  __int128 v8; // [rsp-50h] [rbp-240h]
  __int128 v9; // [rsp-40h] [rbp-230h]
  __int128 v10; // [rsp-40h] [rbp-230h]
  __int128 v11; // [rsp-28h] [rbp-218h]
  __int128 v12; // [rsp-28h] [rbp-218h]
  __int128 v13; // [rsp-18h] [rbp-208h]
  __int128 v14; // [rsp-18h] [rbp-208h]
  int v15; // [rsp+30h] [rbp-1C0h] BYREF
  int v16; // [rsp+34h] [rbp-1BCh] BYREF
  int *v17; // [rsp+38h] [rbp-1B8h] BYREF
  _QWORD v18[2]; // [rsp+40h] [rbp-1B0h] BYREF
  _QWORD v19[2]; // [rsp+50h] [rbp-1A0h] BYREF
  __int64 v20; // [rsp+60h] [rbp-190h]
  const char *v21; // [rsp+68h] [rbp-188h]
  __int64 v22; // [rsp+70h] [rbp-180h]
  char *v23; // [rsp+80h] [rbp-170h] BYREF
  __int64 v24; // [rsp+88h] [rbp-168h]
  __int64 v25; // [rsp+90h] [rbp-160h]
  const char *v26; // [rsp+98h] [rbp-158h]
  __int64 v27; // [rsp+A0h] [rbp-150h]
  _QWORD v28[2]; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v29; // [rsp+C0h] [rbp-130h]
  const char *v30; // [rsp+C8h] [rbp-128h]
  __int64 v31; // [rsp+D0h] [rbp-120h]
  char *v32; // [rsp+E0h] [rbp-110h] BYREF
  __int64 v33; // [rsp+E8h] [rbp-108h]
  __int64 v34; // [rsp+F0h] [rbp-100h]
  const char *v35; // [rsp+F8h] [rbp-F8h]
  __int64 v36; // [rsp+100h] [rbp-F0h]
  const char *v37; // [rsp+110h] [rbp-E0h] BYREF
  __int64 v38; // [rsp+118h] [rbp-D8h]
  _QWORD v39[2]; // [rsp+120h] [rbp-D0h] BYREF
  int v40; // [rsp+130h] [rbp-C0h]
  const char *v41; // [rsp+138h] [rbp-B8h]
  __int64 v42; // [rsp+140h] [rbp-B0h]
  char *v43; // [rsp+148h] [rbp-A8h]
  __int64 v44; // [rsp+150h] [rbp-A0h]
  int v45; // [rsp+158h] [rbp-98h]
  const char *v46; // [rsp+160h] [rbp-90h]
  __int64 v47; // [rsp+168h] [rbp-88h]
  char *v48; // [rsp+170h] [rbp-80h]
  __int64 v49; // [rsp+178h] [rbp-78h]
  int v50; // [rsp+180h] [rbp-70h]
  const char *v51; // [rsp+188h] [rbp-68h]
  __int64 v52; // [rsp+190h] [rbp-60h]

  sub_D95050(&qword_4FE25E0, 0, 0);
  qword_4FE2668 = 0;
  qword_4FE26A0 = (__int64)nullsub_23;
  qword_4FE2680 = (__int64)&unk_49DC1D0;
  qword_4FE2698 = (__int64)sub_984030;
  qword_4FE2670 = (__int64)&unk_49D9748;
  qword_4FE25E0 = (__int64)&unk_49DC090;
  qword_4FE2678 = 0;
  sub_C53080(&qword_4FE25E0, "asan-kernel", 11);
  qword_4FE2608 = (__int64)"Enable KernelAddressSanitizer instrumentation";
  LOWORD(qword_4FE2678) = 256;
  LOBYTE(qword_4FE2668) = 0;
  qword_4FE2610 = 45;
  byte_4FE25EC = byte_4FE25EC & 0x9F | 0x20;
  sub_C53130(&qword_4FE25E0);
  __cxa_atexit(sub_984900, &qword_4FE25E0, &qword_4A427C0);
  LOBYTE(v23) = 0;
  v37 = "Enable recovery mode (continue-after-error).";
  v32 = (char *)&v23;
  LODWORD(v28[0]) = 1;
  v38 = 44;
  sub_23E6020(&unk_4FE2500, "asan-recover", &v37, v28, &v32);
  __cxa_atexit(sub_984900, &unk_4FE2500, &qword_4A427C0);
  sub_D95050(&qword_4FE2420, 0, 0);
  qword_4FE24A8 = 0;
  qword_4FE24E0 = (__int64)nullsub_23;
  qword_4FE24C0 = (__int64)&unk_49DC1D0;
  qword_4FE24D8 = (__int64)sub_984030;
  qword_4FE2420 = (__int64)&unk_49DC090;
  qword_4FE24B0 = (__int64)&unk_49D9748;
  qword_4FE24B8 = 0;
  sub_C53080(&qword_4FE2420, "asan-guard-against-version-mismatch", 35);
  qword_4FE2448 = (__int64)"Guard against compiler/runtime version mismatch.";
  LOWORD(qword_4FE24B8) = 257;
  LOBYTE(qword_4FE24A8) = 1;
  qword_4FE2450 = 48;
  byte_4FE242C = byte_4FE242C & 0x9F | 0x20;
  sub_C53130(&qword_4FE2420);
  __cxa_atexit(sub_984900, &qword_4FE2420, &qword_4A427C0);
  LOBYTE(v23) = 1;
  v37 = "instrument read instructions";
  v32 = (char *)&v23;
  LODWORD(v28[0]) = 1;
  v38 = 28;
  sub_23E6230(&unk_4FE2340, "asan-instrument-reads", &v37, v28, &v32);
  __cxa_atexit(sub_984900, &unk_4FE2340, &qword_4A427C0);
  LOBYTE(v23) = 1;
  v37 = "instrument write instructions";
  v32 = (char *)&v23;
  LODWORD(v28[0]) = 1;
  v38 = 29;
  sub_23E6440(&unk_4FE2260, "asan-instrument-writes", &v37, v28, &v32);
  __cxa_atexit(sub_984900, &unk_4FE2260, &qword_4A427C0);
  sub_D95050(&qword_4FE2180, 0, 0);
  qword_4FE2208 = 0;
  qword_4FE2218 = 0;
  qword_4FE2220 = (__int64)&unk_49DC1D0;
  qword_4FE2240 = (__int64)nullsub_23;
  qword_4FE2238 = (__int64)sub_984030;
  qword_4FE2180 = (__int64)&unk_49DC090;
  qword_4FE2210 = (__int64)&unk_49D9748;
  sub_C53080(&qword_4FE2180, "asan-use-stack-safety", 21);
  qword_4FE21A8 = (__int64)"Use Stack Safety analysis results";
  LOWORD(qword_4FE2218) = 257;
  LOBYTE(qword_4FE2208) = 1;
  qword_4FE21B0 = 33;
  byte_4FE218C = byte_4FE218C & 0x98 | 0x20;
  sub_C53130(&qword_4FE2180);
  __cxa_atexit(sub_984900, &qword_4FE2180, &qword_4A427C0);
  LOBYTE(v23) = 1;
  v37 = "instrument atomic instructions (rmw, cmpxchg)";
  v32 = (char *)&v23;
  LODWORD(v28[0]) = 1;
  v38 = 45;
  sub_23E6650(&unk_4FE20A0, "asan-instrument-atomics", &v37, v28, &v32);
  __cxa_atexit(sub_984900, &unk_4FE20A0, &qword_4A427C0);
  LOBYTE(v23) = 1;
  v37 = "instrument byval call arguments";
  v32 = (char *)&v23;
  LODWORD(v28[0]) = 1;
  v38 = 31;
  sub_23E6230(&unk_4FE1FC0, "asan-instrument-byval", &v37, v28, &v32);
  __cxa_atexit(sub_984900, &unk_4FE1FC0, &qword_4A427C0);
  LOBYTE(v23) = 0;
  v37 = "use instrumentation with slow path for all accesses";
  v32 = (char *)&v23;
  LODWORD(v28[0]) = 1;
  v38 = 51;
  sub_23E6230(&unk_4FE1EE0, "asan-always-slow-path", &v37, v28, &v32);
  __cxa_atexit(sub_984900, &unk_4FE1EE0, &qword_4A427C0);
  LOBYTE(v23) = 0;
  v37 = "Load shadow address into a local variable for each function";
  v32 = (char *)&v23;
  LODWORD(v28[0]) = 1;
  v38 = 59;
  sub_23E6860(&unk_4FE1E00, "asan-force-dynamic-shadow", &v37, v28, &v32);
  __cxa_atexit(sub_984900, &unk_4FE1E00, &qword_4A427C0);
  sub_D95050(&qword_4FE1D20, 0, 0);
  qword_4FE1DA8 = 0;
  qword_4FE1DB8 = 0;
  qword_4FE1DC0 = (__int64)&unk_49DC1D0;
  qword_4FE1DE0 = (__int64)nullsub_23;
  qword_4FE1DD8 = (__int64)sub_984030;
  qword_4FE1D20 = (__int64)&unk_49DC090;
  qword_4FE1DB0 = (__int64)&unk_49D9748;
  sub_C53080(&qword_4FE1D20, "asan-with-ifunc", 15);
  qword_4FE1D50 = 76;
  qword_4FE1D48 = (__int64)"Access dynamic shadow through an ifunc global on platforms that support this";
  LOBYTE(qword_4FE1DA8) = 1;
  byte_4FE1D2C = byte_4FE1D2C & 0x9F | 0x20;
  LOWORD(qword_4FE1DB8) = 257;
  sub_C53130(&qword_4FE1D20);
  __cxa_atexit(sub_984900, &qword_4FE1D20, &qword_4A427C0);
  sub_D95050(&qword_4FE1C40, 0, 0);
  qword_4FE1CC8 = 0;
  qword_4FE1CD8 = 0;
  qword_4FE1CE0 = (__int64)&unk_49DC1D0;
  qword_4FE1D00 = (__int64)nullsub_23;
  qword_4FE1CF8 = (__int64)sub_984030;
  qword_4FE1C40 = (__int64)&unk_49DC090;
  qword_4FE1CD0 = (__int64)&unk_49D9748;
  sub_C53080(&qword_4FE1C40, "asan-with-ifunc-suppress-remat", 30);
  qword_4FE1C70 = 98;
  qword_4FE1C68 = (__int64)"Suppress rematerialization of dynamic shadow address by passing it through inline asm in prologue.";
  LOBYTE(qword_4FE1CC8) = 1;
  byte_4FE1C4C = byte_4FE1C4C & 0x9F | 0x20;
  LOWORD(qword_4FE1CD8) = 257;
  sub_C53130(&qword_4FE1C40);
  __cxa_atexit(sub_984900, &qword_4FE1C40, &qword_4A427C0);
  sub_D95050(&qword_4FE1B60, 0, 0);
  qword_4FE1BE8 = 0;
  qword_4FE1BF8 = 0;
  qword_4FE1BF0 = (__int64)&unk_49DA090;
  qword_4FE1B60 = (__int64)&unk_49DBF90;
  qword_4FE1C20 = (__int64)nullsub_58;
  qword_4FE1C18 = (__int64)sub_B2B5F0;
  qword_4FE1C00 = (__int64)&unk_49DC230;
  sub_C53080(&qword_4FE1B60, "asan-max-ins-per-bb", 19);
  LODWORD(qword_4FE1BE8) = 10000;
  qword_4FE1B88 = (__int64)"maximal number of instructions to instrument in any given BB";
  BYTE4(qword_4FE1BF8) = 1;
  LODWORD(qword_4FE1BF8) = 10000;
  qword_4FE1B90 = 60;
  byte_4FE1B6C = byte_4FE1B6C & 0x9F | 0x20;
  sub_C53130(&qword_4FE1B60);
  __cxa_atexit(sub_B2B680, &qword_4FE1B60, &qword_4A427C0);
  sub_D95050(&qword_4FE1A80, 0, 0);
  qword_4FE1B08 = 0;
  qword_4FE1B10 = (__int64)&unk_49D9748;
  qword_4FE1B20 = (__int64)&unk_49DC1D0;
  qword_4FE1B40 = (__int64)nullsub_23;
  qword_4FE1A80 = (__int64)&unk_49DC090;
  qword_4FE1B38 = (__int64)sub_984030;
  qword_4FE1B18 = 0;
  sub_C53080(&qword_4FE1A80, "asan-stack", 10);
  qword_4FE1AB0 = 19;
  qword_4FE1AA8 = (__int64)"Handle stack memory";
  LOBYTE(qword_4FE1B08) = 1;
  byte_4FE1A8C = byte_4FE1A8C & 0x9F | 0x20;
  LOWORD(qword_4FE1B18) = 257;
  sub_C53130(&qword_4FE1A80);
  __cxa_atexit(sub_984900, &qword_4FE1A80, &qword_4A427C0);
  sub_D95050(&qword_4FE19A0, 0, 0);
  qword_4FE1A28 = 0;
  qword_4FE1A38 = 0;
  qword_4FE1A30 = (__int64)&unk_49D9728;
  qword_4FE19A0 = (__int64)&unk_49DBF10;
  qword_4FE1A40 = (__int64)&unk_49DC290;
  qword_4FE1A60 = (__int64)nullsub_24;
  qword_4FE1A58 = (__int64)sub_984050;
  sub_C53080(&qword_4FE19A0, "asan-max-inline-poisoning-size", 30);
  qword_4FE19D0 = 65;
  qword_4FE19C8 = (__int64)"Inline shadow poisoning for blocks up to the given size in bytes.";
  LODWORD(qword_4FE1A28) = 64;
  BYTE4(qword_4FE1A38) = 1;
  LODWORD(qword_4FE1A38) = 64;
  byte_4FE19AC = byte_4FE19AC & 0x9F | 0x20;
  sub_C53130(&qword_4FE19A0);
  __cxa_atexit(sub_984970, &qword_4FE19A0, &qword_4A427C0);
  v39[0] = "never";
  v41 = "Never detect stack use after return.";
  v43 = "runtime";
  v46 = "Detect stack use after return if binary flag 'ASAN_OPTIONS=detect_stack_use_after_return' is set.";
  v48 = "always";
  v51 = "Always detect stack use after return.";
  v38 = 0x400000003LL;
  v32 = "Sets the mode of detection for stack-use-after-return.";
  LODWORD(v23) = 1;
  v28[0] = &v23;
  LODWORD(v19[0]) = 1;
  v37 = (const char *)v39;
  v39[1] = 5;
  v40 = 0;
  v42 = 36;
  v44 = 7;
  v45 = 1;
  v47 = 97;
  v49 = 6;
  v50 = 2;
  v52 = 37;
  v33 = 54;
  sub_23F56C0(&unk_4FE1740, "asan-use-after-return", &v32, &v37, v19, v28);
  if ( v37 != (const char *)v39 )
    _libc_free(v37, "asan-use-after-return");
  __cxa_atexit(sub_23DCA30, &unk_4FE1740, &qword_4A427C0);
  LOBYTE(v23) = 1;
  v37 = "Create redzones for byval arguments (extra copy required)";
  v32 = (char *)&v23;
  LODWORD(v28[0]) = 1;
  v38 = 57;
  sub_23E6650(&unk_4FE1660, "asan-redzone-byval-args", &v37, v28, &v32);
  __cxa_atexit(sub_984900, &unk_4FE1660, &qword_4A427C0);
  sub_D95050(&qword_4FE1580, 0, 0);
  qword_4FE1608 = 0;
  qword_4FE1618 = 0;
  qword_4FE1610 = (__int64)&unk_49D9748;
  qword_4FE1580 = (__int64)&unk_49DC090;
  qword_4FE1620 = (__int64)&unk_49DC1D0;
  qword_4FE1640 = (__int64)nullsub_23;
  qword_4FE1638 = (__int64)sub_984030;
  sub_C53080(&qword_4FE1580, "asan-use-after-scope", 20);
  qword_4FE15A8 = (__int64)"Check stack-use-after-scope";
  LOWORD(qword_4FE1618) = 256;
  LOBYTE(qword_4FE1608) = 0;
  qword_4FE15B0 = 27;
  byte_4FE158C = byte_4FE158C & 0x9F | 0x20;
  sub_C53130(&qword_4FE1580);
  __cxa_atexit(sub_984900, &qword_4FE1580, &qword_4A427C0);
  LOBYTE(v23) = 1;
  v37 = "Handle global objects";
  v32 = (char *)&v23;
  LODWORD(v28[0]) = 1;
  v38 = 21;
  sub_23E6020(&unk_4FE14A0, "asan-globals", &v37, v28, &v32);
  __cxa_atexit(sub_984900, &unk_4FE14A0, &qword_4A427C0);
  LOBYTE(v23) = 1;
  v37 = "Handle C++ initializer order";
  v32 = (char *)&v23;
  LODWORD(v28[0]) = 1;
  v38 = 28;
  sub_23E6860(&unk_4FE13C0, "asan-initialization-order", &v37, v28, &v32);
  __cxa_atexit(sub_984900, &unk_4FE13C0, &qword_4A427C0);
  LOBYTE(v23) = 0;
  v37 = "Instrument <, <=, >, >=, - with pointer operands";
  v32 = (char *)&v23;
  LODWORD(v28[0]) = 1;
  v38 = 48;
  sub_23E6A70(&unk_4FE12E0, "asan-detect-invalid-pointer-pair", &v37, v28, &v32);
  __cxa_atexit(sub_984900, &unk_4FE12E0, &qword_4A427C0);
  LOBYTE(v23) = 0;
  v37 = "Instrument <, <=, >, >= with pointer operands";
  v32 = (char *)&v23;
  LODWORD(v28[0]) = 1;
  v38 = 45;
  sub_23E6C80(&unk_4FE1200, "asan-detect-invalid-pointer-cmp", &v37, v28, &v32);
  __cxa_atexit(sub_984900, &unk_4FE1200, &qword_4A427C0);
  LOBYTE(v23) = 0;
  v37 = "Instrument - operations with pointer operands";
  v32 = (char *)&v23;
  LODWORD(v28[0]) = 1;
  v38 = 45;
  sub_23E6C80(&unk_4FE1120, "asan-detect-invalid-pointer-sub", &v37, v28, &v32);
  __cxa_atexit(sub_984900, &unk_4FE1120, &qword_4A427C0);
  sub_D95050(&qword_4FE1040, 0, 0);
  qword_4FE10C8 = 0;
  qword_4FE10D8 = 0;
  qword_4FE10D0 = (__int64)&unk_49D9728;
  qword_4FE1040 = (__int64)&unk_49DBF10;
  qword_4FE10E0 = (__int64)&unk_49DC290;
  qword_4FE1100 = (__int64)nullsub_24;
  qword_4FE10F8 = (__int64)sub_984050;
  sub_C53080(&qword_4FE1040, "asan-realign-stack", 18);
  qword_4FE1070 = 54;
  qword_4FE1068 = (__int64)"Realign stack to the value of this flag (power of two)";
  LODWORD(qword_4FE10C8) = 32;
  BYTE4(qword_4FE10D8) = 1;
  LODWORD(qword_4FE10D8) = 32;
  byte_4FE104C = byte_4FE104C & 0x9F | 0x20;
  sub_C53130(&qword_4FE1040);
  __cxa_atexit(sub_984970, &qword_4FE1040, &qword_4A427C0);
  sub_D95050(&qword_4FE0F60, 0, 0);
  qword_4FE0FE8 = 0;
  qword_4FE0FF8 = 0;
  qword_4FE0FF0 = (__int64)&unk_49DA090;
  qword_4FE0F60 = (__int64)&unk_49DBF90;
  qword_4FE1000 = (__int64)&unk_49DC230;
  qword_4FE1020 = (__int64)nullsub_58;
  qword_4FE1018 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FE0F60, "asan-instrumentation-with-call-threshold", 40);
  qword_4FE0F90 = 156;
  qword_4FE0F88 = (__int64)"If the function being instrumented contains more than this number of memory accesses, use cal"
                           "lbacks instead of inline checks (-1 means never use callbacks).";
  LODWORD(qword_4FE0FE8) = 7000;
  BYTE4(qword_4FE0FF8) = 1;
  LODWORD(qword_4FE0FF8) = 7000;
  byte_4FE0F6C = byte_4FE0F6C & 0x9F | 0x20;
  sub_C53130(&qword_4FE0F60);
  __cxa_atexit(sub_B2B680, &qword_4FE0F60, &qword_4A427C0);
  sub_D95050(&qword_4FE0E60, 0, 0);
  qword_4FE0EE8 = (__int64)&byte_4FE0EF8;
  qword_4FE0F10 = (__int64)&byte_4FE0F20;
  qword_4FE0EF0 = 0;
  byte_4FE0EF8 = 0;
  qword_4FE0F08 = (__int64)&unk_49DC130;
  qword_4FE0F18 = 0;
  byte_4FE0F20 = 0;
  qword_4FE0E60 = (__int64)&unk_49DC010;
  byte_4FE0F30 = 0;
  qword_4FE0F38 = (__int64)&unk_49DC350;
  qword_4FE0F58 = (__int64)nullsub_92;
  qword_4FE0F50 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4FE0E60, "asan-memory-access-callback-prefix", 34);
  qword_4FE0E88 = (__int64)"Prefix for memory access callbacks";
  qword_4FE0E90 = 34;
  byte_4FE0E6C = byte_4FE0E6C & 0x9F | 0x20;
  v39[0] = 0x5F6E6173615F5FLL;
  v37 = (const char *)v39;
  v38 = 7;
  sub_2240AE0(&qword_4FE0EE8, &v37);
  byte_4FE0F30 = 1;
  sub_2240AE0(&qword_4FE0F10, &v37);
  sub_2240A30(&v37);
  sub_C53130(&qword_4FE0E60);
  __cxa_atexit(sub_BC5A40, &qword_4FE0E60, &qword_4A427C0);
  v37 = "Use prefix for memory intrinsics in KASAN mode";
  LOBYTE(v23) = 0;
  v32 = (char *)&v23;
  LODWORD(v28[0]) = 1;
  v38 = 46;
  sub_23E6A70(&unk_4FE0D80, "asan-kernel-mem-intrinsic-prefix", &v37, v28, &v32);
  __cxa_atexit(sub_984900, &unk_4FE0D80, &qword_4A427C0);
  v37 = "instrument dynamic allocas";
  LOBYTE(v23) = 1;
  v32 = (char *)&v23;
  LODWORD(v28[0]) = 1;
  v38 = 26;
  sub_23E6C80(&unk_4FE0CA0, "asan-instrument-dynamic-allocas", &v37, v28, &v32);
  __cxa_atexit(sub_984900, &unk_4FE0CA0, &qword_4A427C0);
  sub_D95050(&qword_4FE0BC0, 0, 0);
  qword_4FE0C80 = (__int64)nullsub_23;
  qword_4FE0C50 = (__int64)&unk_49D9748;
  qword_4FE0C78 = (__int64)sub_984030;
  qword_4FE0BC0 = (__int64)&unk_49DC090;
  qword_4FE0C60 = (__int64)&unk_49DC1D0;
  qword_4FE0C48 = 0;
  qword_4FE0C58 = 0;
  sub_C53080(&qword_4FE0BC0, "asan-skip-promotable-allocas", 28);
  qword_4FE0BE8 = (__int64)"Do not instrument promotable allocas";
  LOWORD(qword_4FE0C58) = 257;
  LOBYTE(qword_4FE0C48) = 1;
  qword_4FE0BF0 = 36;
  byte_4FE0BCC = byte_4FE0BCC & 0x9F | 0x20;
  sub_C53130(&qword_4FE0BC0);
  __cxa_atexit(sub_984900, &qword_4FE0BC0, &qword_4A427C0);
  v35 = "Use global constructors";
  v19[0] = &v17;
  v30 = "No constructors";
  v33 = 6;
  LODWORD(v34) = 1;
  v36 = 23;
  v28[1] = 4;
  LODWORD(v29) = 0;
  *((_QWORD *)&v13 + 1) = "Use global constructors";
  v31 = 15;
  *(_QWORD *)&v13 = v34;
  *((_QWORD *)&v11 + 1) = 6;
  *(_QWORD *)&v11 = "global";
  *((_QWORD *)&v9 + 1) = "No constructors";
  *(_QWORD *)&v9 = v29;
  *((_QWORD *)&v7 + 1) = 4;
  *(_QWORD *)&v7 = "none";
  v32 = "global";
  LODWORD(v18[0]) = 1;
  LODWORD(v17) = 1;
  v28[0] = "none";
  sub_23E6E90(
    (unsigned int)&v37,
    (unsigned int)&qword_4FE0BC0,
    v0,
    (unsigned int)"No constructors",
    v1,
    v2,
    v7,
    v9,
    15,
    v11,
    v13,
    23);
  v23 = "Sets the ASan constructor kind";
  v24 = 30;
  sub_23F5B40(&unk_4FE0960, "asan-constructor-kind", &v23, &v37, v19, v18);
  if ( v37 != (const char *)v39 )
    _libc_free(v37, "asan-constructor-kind");
  __cxa_atexit(sub_23DC9A0, &unk_4FE0960, &qword_4A427C0);
  sub_D95050(&qword_4FE0880, 0, 0);
  qword_4FE0908 = 0;
  qword_4FE0918 = 0;
  qword_4FE0910 = (__int64)&unk_49DA090;
  qword_4FE0880 = (__int64)&unk_49DBF90;
  qword_4FE0920 = (__int64)&unk_49DC230;
  qword_4FE0940 = (__int64)nullsub_58;
  qword_4FE0938 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FE0880, "asan-mapping-scale", 18);
  qword_4FE08B0 = 28;
  qword_4FE08A8 = (__int64)"scale of asan shadow mapping";
  LODWORD(qword_4FE0908) = 0;
  BYTE4(qword_4FE0918) = 1;
  LODWORD(qword_4FE0918) = 0;
  byte_4FE088C = byte_4FE088C & 0x9F | 0x20;
  sub_C53130(&qword_4FE0880);
  __cxa_atexit(sub_B2B680, &qword_4FE0880, &qword_4A427C0);
  sub_D95050(&qword_4FE07A0, 0, 0);
  byte_4FE0840 = 0;
  qword_4FE0828 = 0;
  qword_4FE0830 = (__int64)&unk_49DB998;
  qword_4FE0838 = 0;
  qword_4FE07A0 = (__int64)&unk_49DB9B8;
  qword_4FE0848 = (__int64)&unk_49DC2C0;
  qword_4FE0868 = (__int64)nullsub_121;
  qword_4FE0860 = (__int64)sub_C1A370;
  sub_C53080(&qword_4FE07A0, "asan-mapping-offset", 19);
  qword_4FE07D0 = 44;
  qword_4FE07C8 = (__int64)"offset of asan shadow mapping [EXPERIMENTAL]";
  qword_4FE0828 = 0;
  byte_4FE0840 = 1;
  qword_4FE0838 = 0;
  byte_4FE07AC = byte_4FE07AC & 0x9F | 0x20;
  sub_C53130(&qword_4FE07A0);
  __cxa_atexit(sub_C1A610, &qword_4FE07A0, &qword_4A427C0);
  sub_D95050(&qword_4FE06C0, 0, 0);
  qword_4FE0748 = 0;
  qword_4FE0758 = 0;
  qword_4FE0750 = (__int64)&unk_49D9748;
  qword_4FE06C0 = (__int64)&unk_49DC090;
  qword_4FE0760 = (__int64)&unk_49DC1D0;
  qword_4FE0780 = (__int64)nullsub_23;
  qword_4FE0778 = (__int64)sub_984030;
  sub_C53080(&qword_4FE06C0, "asan-opt", 8);
  qword_4FE06F0 = 24;
  qword_4FE06E8 = (__int64)"Optimize instrumentation";
  LOBYTE(qword_4FE0748) = 1;
  byte_4FE06CC = byte_4FE06CC & 0x9F | 0x20;
  LOWORD(qword_4FE0758) = 257;
  sub_C53130(&qword_4FE06C0);
  __cxa_atexit(sub_984900, &qword_4FE06C0, &qword_4A427C0);
  LOBYTE(v18[0]) = 0;
  v37 = "Optimize callbacks";
  v23 = (char *)v18;
  LODWORD(v19[0]) = 1;
  v38 = 18;
  sub_23E6650(&unk_4FE05E0, "asan-optimize-callbacks", &v37, v19, &v23);
  __cxa_atexit(sub_984900, &unk_4FE05E0, &qword_4A427C0);
  sub_D95050(&qword_4FE0500, 0, 0);
  qword_4FE05C0 = (__int64)nullsub_23;
  qword_4FE0590 = (__int64)&unk_49D9748;
  qword_4FE0500 = (__int64)&unk_49DC090;
  qword_4FE05A0 = (__int64)&unk_49DC1D0;
  qword_4FE05B8 = (__int64)sub_984030;
  qword_4FE0588 = 0;
  qword_4FE0598 = 0;
  sub_C53080(&qword_4FE0500, "asan-opt-same-temp", 18);
  qword_4FE0528 = (__int64)"Instrument the same temp just once";
  LOWORD(qword_4FE0598) = 257;
  LOBYTE(qword_4FE0588) = 1;
  qword_4FE0530 = 34;
  byte_4FE050C = byte_4FE050C & 0x9F | 0x20;
  sub_C53130(&qword_4FE0500);
  __cxa_atexit(sub_984900, &qword_4FE0500, &qword_4A427C0);
  LOBYTE(v18[0]) = 1;
  v37 = "Don't instrument scalar globals";
  v23 = (char *)v18;
  LODWORD(v19[0]) = 1;
  v38 = 31;
  sub_23E6F00(&unk_4FE0420, "asan-opt-globals", &v37, v19, &v23);
  __cxa_atexit(sub_984900, &unk_4FE0420, &qword_4A427C0);
  sub_D95050(&qword_4FE0340, 0, 0);
  qword_4FE0400 = (__int64)nullsub_23;
  qword_4FE03D0 = (__int64)&unk_49D9748;
  qword_4FE0340 = (__int64)&unk_49DC090;
  qword_4FE03E0 = (__int64)&unk_49DC1D0;
  qword_4FE03F8 = (__int64)sub_984030;
  qword_4FE03C8 = 0;
  qword_4FE03D8 = 0;
  sub_C53080(&qword_4FE0340, "asan-opt-stack", 14);
  qword_4FE0368 = (__int64)"Don't instrument scalar stack variables";
  LOWORD(qword_4FE03D8) = 256;
  LOBYTE(qword_4FE03C8) = 0;
  qword_4FE0370 = 39;
  byte_4FE034C = byte_4FE034C & 0x9F | 0x20;
  sub_C53130(&qword_4FE0340);
  __cxa_atexit(sub_984900, &qword_4FE0340, &qword_4A427C0);
  LOBYTE(v18[0]) = 1;
  v37 = "Use dynamic alloca to represent stack variables";
  v23 = (char *)v18;
  LODWORD(v19[0]) = 1;
  v38 = 47;
  sub_23E6860(&unk_4FE0260, "asan-stack-dynamic-alloca", &v37, v19, &v23);
  __cxa_atexit(sub_984900, &unk_4FE0260, &qword_4A427C0);
  sub_D95050(&qword_4FE0180, 0, 0);
  qword_4FE0208 = 0;
  qword_4FE0218 = 0;
  qword_4FE0210 = (__int64)&unk_49D9728;
  qword_4FE0180 = (__int64)&unk_49DBF10;
  qword_4FE0220 = (__int64)&unk_49DC290;
  qword_4FE0240 = (__int64)nullsub_24;
  qword_4FE0238 = (__int64)sub_984050;
  sub_C53080(&qword_4FE0180, "asan-force-experiment", 21);
  qword_4FE01B0 = 43;
  qword_4FE01A8 = (__int64)"Force optimization experiment (for testing)";
  LODWORD(qword_4FE0208) = 0;
  BYTE4(qword_4FE0218) = 1;
  LODWORD(qword_4FE0218) = 0;
  byte_4FE018C = byte_4FE018C & 0x9F | 0x20;
  sub_C53130(&qword_4FE0180);
  __cxa_atexit(sub_984970, &qword_4FE0180, &qword_4A427C0);
  LOBYTE(v18[0]) = 1;
  v37 = "Use private aliases for global variables";
  v23 = (char *)v18;
  LODWORD(v19[0]) = 1;
  v38 = 40;
  sub_23E6440(&unk_4FE00A0, "asan-use-private-alias", &v37, v19, &v23);
  __cxa_atexit(sub_984900, &unk_4FE00A0, &qword_4A427C0);
  LOBYTE(v18[0]) = 1;
  v37 = "Use odr indicators to improve ODR reporting";
  v23 = (char *)v18;
  LODWORD(v19[0]) = 1;
  v38 = 43;
  sub_23E6440(&unk_4FDFFC0, "asan-use-odr-indicator", &v37, v19, &v23);
  __cxa_atexit(sub_984900, &unk_4FDFFC0, &qword_4A427C0);
  LOBYTE(v18[0]) = 1;
  v37 = "Use linker features to support dead code stripping of globals";
  v23 = (char *)v18;
  LODWORD(v19[0]) = 1;
  v38 = 61;
  sub_23E6860(&unk_4FDFEE0, "asan-globals-live-support", &v37, v19, &v23);
  __cxa_atexit(sub_984900, &unk_4FDFEE0, &qword_4A427C0);
  LOBYTE(v18[0]) = 1;
  v37 = "Place ASan constructors in comdat sections";
  v23 = (char *)v18;
  LODWORD(v19[0]) = 1;
  v38 = 42;
  sub_23E6F00(&unk_4FDFE00, "asan-with-comdat", &v37, v19, &v23);
  __cxa_atexit(sub_984900, &unk_4FDFE00, &qword_4A427C0);
  v26 = "Use global destructors";
  v17 = &v15;
  v21 = "No destructors";
  v24 = 6;
  LODWORD(v25) = 1;
  v27 = 22;
  v19[1] = 4;
  LODWORD(v20) = 0;
  *((_QWORD *)&v14 + 1) = "Use global destructors";
  v22 = 14;
  *(_QWORD *)&v14 = v25;
  *((_QWORD *)&v12 + 1) = 6;
  *(_QWORD *)&v12 = "global";
  *((_QWORD *)&v10 + 1) = "No destructors";
  *(_QWORD *)&v10 = v20;
  *((_QWORD *)&v8 + 1) = 4;
  *(_QWORD *)&v8 = "none";
  v23 = "global";
  v16 = 1;
  v15 = 2;
  v19[0] = "none";
  sub_23E6E90(
    (unsigned int)&v37,
    (unsigned int)&unk_4FDFE00,
    v3,
    (unsigned int)"No destructors",
    v4,
    v5,
    v8,
    v10,
    14,
    v12,
    v14,
    22);
  v18[0] = "Sets the ASan destructor kind. The default is to use the value provided to the pass constructor";
  v18[1] = 95;
  sub_23F5FC0(&unk_4FDFBA0, "asan-destructor-kind", v18, &v37, &v17, &v16);
  if ( v37 != (const char *)v39 )
    _libc_free(v37, "asan-destructor-kind");
  __cxa_atexit(sub_23DCAC0, &unk_4FDFBA0, &qword_4A427C0);
  sub_D95050(&qword_4FDFAC0, 0, 0);
  qword_4FDFB48 = 0;
  qword_4FDFB58 = 0;
  qword_4FDFB50 = (__int64)&unk_49DA090;
  qword_4FDFAC0 = (__int64)&unk_49DBF90;
  qword_4FDFB60 = (__int64)&unk_49DC230;
  qword_4FDFB80 = (__int64)nullsub_58;
  qword_4FDFB78 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FDFAC0, "asan-debug", 10);
  qword_4FDFAF0 = 5;
  qword_4FDFAE8 = (__int64)"debug";
  LODWORD(qword_4FDFB48) = 0;
  BYTE4(qword_4FDFB58) = 1;
  LODWORD(qword_4FDFB58) = 0;
  byte_4FDFACC = byte_4FDFACC & 0x9F | 0x20;
  sub_C53130(&qword_4FDFAC0);
  __cxa_atexit(sub_B2B680, &qword_4FDFAC0, &qword_4A427C0);
  sub_D95050(&qword_4FDF9E0, 0, 0);
  qword_4FDFA70 = (__int64)&unk_49DA090;
  qword_4FDFAA0 = (__int64)nullsub_58;
  qword_4FDF9E0 = (__int64)&unk_49DBF90;
  qword_4FDFA80 = (__int64)&unk_49DC230;
  qword_4FDFA98 = (__int64)sub_B2B5F0;
  qword_4FDFA68 = 0;
  qword_4FDFA78 = 0;
  sub_C53080(&qword_4FDF9E0, "asan-debug-stack", 16);
  qword_4FDFA10 = 11;
  qword_4FDFA08 = (__int64)"debug stack";
  LODWORD(qword_4FDFA68) = 0;
  BYTE4(qword_4FDFA78) = 1;
  LODWORD(qword_4FDFA78) = 0;
  byte_4FDF9EC = byte_4FDF9EC & 0x9F | 0x20;
  sub_C53130(&qword_4FDF9E0);
  __cxa_atexit(sub_B2B680, &qword_4FDF9E0, &qword_4A427C0);
  sub_D95050(&qword_4FDF8E0, 0, 0);
  qword_4FDF968 = &byte_4FDF978;
  qword_4FDF990 = (__int64)&byte_4FDF9A0;
  qword_4FDF970 = 0;
  byte_4FDF978 = 0;
  qword_4FDF988 = (__int64)&unk_49DC130;
  qword_4FDF998 = 0;
  byte_4FDF9A0 = 0;
  qword_4FDF8E0 = (__int64)&unk_49DC010;
  byte_4FDF9B0 = 0;
  qword_4FDF9B8 = (__int64)&unk_49DC350;
  qword_4FDF9D8 = (__int64)nullsub_92;
  qword_4FDF9D0 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4FDF8E0, "asan-debug-func", 15);
  qword_4FDF910 = 10;
  byte_4FDF8EC = byte_4FDF8EC & 0x9F | 0x20;
  qword_4FDF908 = (__int64)"Debug func";
  sub_C53130(&qword_4FDF8E0);
  __cxa_atexit(sub_BC5A40, &qword_4FDF8E0, &qword_4A427C0);
  LODWORD(v17) = -1;
  v37 = "Debug min inst";
  v18[0] = &v17;
  v16 = 1;
  v38 = 14;
  sub_23E7110(&unk_4FDF800, "asan-debug-min", &v37, &v16, v18);
  __cxa_atexit(sub_B2B680, &unk_4FDF800, &qword_4A427C0);
  v18[0] = &v17;
  v37 = "Debug max inst";
  LODWORD(v17) = -1;
  v16 = 1;
  v38 = 14;
  sub_23E7110(&unk_4FDF720, "asan-debug-max", &v37, &v16, v18);
  return __cxa_atexit(sub_B2B680, &unk_4FDF720, &qword_4A427C0);
}
