// Function: ctor_397_0
// Address: 0x5221b0
//
int ctor_397_0()
{
  int v0; // r8d
  int v1; // r9d
  const char *v2; // rsi
  char *v3; // r14
  char *v4; // r15
  __int64 v5; // rcx
  const char *v6; // rdx
  const char *v7; // r11
  __int64 v8; // r10
  __int64 v9; // rax
  const __m128i *v10; // rcx
  int v11; // edx
  __int64 v12; // rax
  __m128i v13; // xmm1
  __int8 v14; // dl
  int v15; // ecx
  int v16; // r8d
  int v17; // r9d
  __m128i *v19; // rax
  void *v20; // rdi
  __int64 v21; // rsi
  const __m128i *v22; // rdx
  __int64 v23; // rcx
  __int8 v24; // di
  int v25; // eax
  __int128 v26; // [rsp-80h] [rbp-2E0h]
  __int128 v27; // [rsp-80h] [rbp-2E0h]
  __int128 v28; // [rsp-70h] [rbp-2D0h]
  __int128 v29; // [rsp-70h] [rbp-2D0h]
  __int128 v30; // [rsp-58h] [rbp-2B8h]
  __int128 v31; // [rsp-58h] [rbp-2B8h]
  __int128 v32; // [rsp-48h] [rbp-2A8h]
  __int128 v33; // [rsp-48h] [rbp-2A8h]
  __int128 v34; // [rsp-30h] [rbp-290h]
  __int128 v35; // [rsp-30h] [rbp-290h]
  __int128 v36; // [rsp-20h] [rbp-280h]
  __int128 v37; // [rsp-20h] [rbp-280h]
  __int64 v38; // [rsp+0h] [rbp-260h]
  unsigned __int64 v39; // [rsp+10h] [rbp-250h]
  char v40; // [rsp+18h] [rbp-248h]
  __int64 v41; // [rsp+20h] [rbp-240h]
  int v42; // [rsp+20h] [rbp-240h]
  const char *v43; // [rsp+30h] [rbp-230h]
  int v44; // [rsp+40h] [rbp-220h] BYREF
  int v45; // [rsp+44h] [rbp-21Ch] BYREF
  int *v46; // [rsp+48h] [rbp-218h] BYREF
  _QWORD v47[4]; // [rsp+50h] [rbp-210h] BYREF
  __int64 v48; // [rsp+70h] [rbp-1F0h]
  const char *v49; // [rsp+78h] [rbp-1E8h]
  __int64 v50; // [rsp+80h] [rbp-1E0h]
  _QWORD v51[2]; // [rsp+90h] [rbp-1D0h] BYREF
  __int64 v52; // [rsp+A0h] [rbp-1C0h]
  const char *v53; // [rsp+A8h] [rbp-1B8h]
  __int64 v54; // [rsp+B0h] [rbp-1B0h]
  char *v55; // [rsp+C0h] [rbp-1A0h]
  __int64 v56; // [rsp+C8h] [rbp-198h]
  __int64 v57; // [rsp+D0h] [rbp-190h]
  const char *v58; // [rsp+D8h] [rbp-188h]
  __int64 v59; // [rsp+E0h] [rbp-180h]
  _QWORD v60[2]; // [rsp+F0h] [rbp-170h] BYREF
  __int64 v61; // [rsp+100h] [rbp-160h]
  const char *v62; // [rsp+108h] [rbp-158h]
  __int64 v63; // [rsp+110h] [rbp-150h]
  _QWORD v64[2]; // [rsp+120h] [rbp-140h] BYREF
  __int64 v65; // [rsp+130h] [rbp-130h]
  const char *v66; // [rsp+138h] [rbp-128h]
  __int64 v67; // [rsp+140h] [rbp-120h]
  const char *v68; // [rsp+150h] [rbp-110h] BYREF
  __int64 v69; // [rsp+158h] [rbp-108h]
  __int64 v70; // [rsp+160h] [rbp-100h]
  const char *v71; // [rsp+168h] [rbp-F8h]
  __int64 v72; // [rsp+170h] [rbp-F0h]
  int v73; // [rsp+178h] [rbp-E8h]
  char v74; // [rsp+17Ch] [rbp-E4h]
  char *v75; // [rsp+180h] [rbp-E0h] BYREF
  __int64 v76; // [rsp+188h] [rbp-D8h]
  char v77[208]; // [rsp+190h] [rbp-D0h] BYREF

  sub_D95050(&qword_4FE5820, 0, 0);
  qword_4FE58A8 = (__int64)&byte_4FE58B8;
  qword_4FE58D0 = (__int64)&byte_4FE58E0;
  qword_4FE58B0 = 0;
  byte_4FE58B8 = 0;
  qword_4FE58C8 = (__int64)&unk_49DC130;
  qword_4FE58D8 = 0;
  byte_4FE58E0 = 0;
  qword_4FE5820 = (__int64)&unk_49DC010;
  byte_4FE58F0 = 0;
  qword_4FE58F8 = (__int64)&unk_49DC350;
  qword_4FE5918 = (__int64)nullsub_92;
  qword_4FE5910 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4FE5820, "hwasan-memory-access-callback-prefix", 36);
  qword_4FE5848 = (__int64)"Prefix for memory access callbacks";
  qword_4FE5850 = 34;
  byte_4FE582C = byte_4FE582C & 0x9F | 0x20;
  v75 = v77;
  strcpy(v77, "__hwasan_");
  v76 = 9;
  sub_2240AE0(&qword_4FE58A8, &v75);
  byte_4FE58F0 = 1;
  sub_2240AE0(&qword_4FE58D0, &v75);
  sub_2240A30(&v75);
  sub_C53130(&qword_4FE5820);
  __cxa_atexit(sub_BC5A40, &qword_4FE5820, &qword_4A427C0);
  sub_D95050(&qword_4FE5740, 0, 0);
  qword_4FE57C8 = 0;
  qword_4FE57D8 = 0;
  qword_4FE57D0 = (__int64)&unk_49D9748;
  qword_4FE5740 = (__int64)&unk_49DC090;
  qword_4FE57E0 = (__int64)&unk_49DC1D0;
  qword_4FE5800 = (__int64)nullsub_23;
  qword_4FE57F8 = (__int64)sub_984030;
  sub_C53080(&qword_4FE5740, "hwasan-kernel-mem-intrinsic-prefix", 34);
  qword_4FE5768 = (__int64)"Use prefix for memory intrinsics in KASAN mode";
  LOWORD(qword_4FE57D8) = 256;
  LOBYTE(qword_4FE57C8) = 0;
  qword_4FE5770 = 46;
  byte_4FE574C = byte_4FE574C & 0x9F | 0x20;
  sub_C53130(&qword_4FE5740);
  __cxa_atexit(sub_984900, &qword_4FE5740, &qword_4A427C0);
  sub_D95050(&qword_4FE5660, 0, 0);
  qword_4FE5720 = (__int64)nullsub_23;
  qword_4FE56F0 = (__int64)&unk_49D9748;
  qword_4FE5718 = (__int64)sub_984030;
  qword_4FE5700 = (__int64)&unk_49DC1D0;
  qword_4FE56E8 = 0;
  qword_4FE56F8 = 0;
  qword_4FE5660 = (__int64)&unk_49DC090;
  sub_C53080(&qword_4FE5660, "hwasan-instrument-with-calls", 28);
  qword_4FE5688 = (__int64)"instrument reads and writes with callbacks";
  LOWORD(qword_4FE56F8) = 256;
  LOBYTE(qword_4FE56E8) = 0;
  byte_4FE566C = byte_4FE566C & 0x9F | 0x20;
  qword_4FE5690 = 42;
  sub_C53130(&qword_4FE5660);
  __cxa_atexit(sub_984900, &qword_4FE5660, &qword_4A427C0);
  v75 = "instrument read instructions";
  v68 = (const char *)v60;
  LOBYTE(v60[0]) = 1;
  LODWORD(v64[0]) = 1;
  v76 = 28;
  sub_23E6650(&unk_4FE5580, "hwasan-instrument-reads", &v75, v64, &v68);
  __cxa_atexit(sub_984900, &unk_4FE5580, &qword_4A427C0);
  v68 = (const char *)v60;
  v75 = "instrument write instructions";
  LOBYTE(v60[0]) = 1;
  LODWORD(v64[0]) = 1;
  v76 = 29;
  sub_243B8A0(&unk_4FE54A0, "hwasan-instrument-writes", &v75, v64, &v68);
  __cxa_atexit(sub_984900, &unk_4FE54A0, &qword_4A427C0);
  v68 = (const char *)v60;
  v75 = "instrument atomic instructions (rmw, cmpxchg)";
  LOBYTE(v60[0]) = 1;
  LODWORD(v64[0]) = 1;
  v76 = 45;
  sub_23E6860(&unk_4FE53C0, "hwasan-instrument-atomics", &v75, v64, &v68);
  __cxa_atexit(sub_984900, &unk_4FE53C0, &qword_4A427C0);
  v68 = (const char *)v60;
  v75 = "instrument byval arguments";
  LOBYTE(v60[0]) = 1;
  LODWORD(v64[0]) = 1;
  v76 = 26;
  sub_23E6650(&unk_4FE52E0, "hwasan-instrument-byval", &v75, v64, &v68);
  __cxa_atexit(sub_984900, &unk_4FE52E0, &qword_4A427C0);
  v68 = (const char *)v60;
  v75 = "Enable recovery mode (continue-after-error).";
  LOBYTE(v60[0]) = 0;
  LODWORD(v64[0]) = 1;
  v76 = 44;
  sub_243BAB0(&unk_4FE5200, "hwasan-recover", &v75, v64, &v68);
  __cxa_atexit(sub_984900, &unk_4FE5200, &qword_4A427C0);
  v68 = (const char *)v60;
  v75 = "instrument stack (allocas)";
  LOBYTE(v60[0]) = 1;
  LODWORD(v64[0]) = 1;
  v76 = 26;
  sub_23E6650(&unk_4FE5120, "hwasan-instrument-stack", &v75, v64, &v68);
  __cxa_atexit(sub_984900, &unk_4FE5120, &qword_4A427C0);
  sub_D95050(&qword_4FE5040, 0, 0);
  qword_4FE50C8 = 0;
  qword_4FE5100 = (__int64)nullsub_23;
  qword_4FE50E0 = (__int64)&unk_49DC1D0;
  qword_4FE50F8 = (__int64)sub_984030;
  qword_4FE50D0 = (__int64)&unk_49D9748;
  qword_4FE50D8 = 0;
  qword_4FE5040 = (__int64)&unk_49DC090;
  sub_C53080(&qword_4FE5040, "hwasan-use-stack-safety", 23);
  LOBYTE(qword_4FE50C8) = 1;
  qword_4FE5068 = (__int64)"Use Stack Safety analysis results";
  LOWORD(qword_4FE50D8) = 257;
  qword_4FE5070 = 33;
  byte_4FE504C = byte_4FE504C & 0x98 | 0x20;
  sub_C53130(&qword_4FE5040);
  __cxa_atexit(sub_984900, &qword_4FE5040, &qword_4A427C0);
  sub_D95050(&qword_4FE4F60, 0, 0);
  byte_4FE5000 = 0;
  qword_4FE5028 = (__int64)nullsub_121;
  qword_4FE4FF0 = (__int64)&unk_49DB998;
  qword_4FE5020 = (__int64)sub_C1A370;
  qword_4FE4F60 = (__int64)&unk_49DB9B8;
  qword_4FE5008 = (__int64)&unk_49DC2C0;
  qword_4FE4FE8 = 0;
  qword_4FE4FF8 = 0;
  sub_C53080(&qword_4FE4F60, "hwasan-max-lifetimes-for-alloca", 31);
  qword_4FE4FE8 = 3;
  qword_4FE4F88 = (__int64)"How many lifetime ends to handle for a single alloca.";
  byte_4FE5000 = 1;
  qword_4FE4FF8 = 3;
  qword_4FE4F90 = 53;
  byte_4FE4F6C = byte_4FE4F6C & 0x98 | 0x40;
  sub_C53130(&qword_4FE4F60);
  __cxa_atexit(sub_C1A610, &qword_4FE4F60, &qword_4A427C0);
  sub_D95050(&qword_4FE4E80, 0, 0);
  qword_4FE4F08 = 0;
  qword_4FE4F40 = (__int64)nullsub_23;
  qword_4FE4F20 = (__int64)&unk_49DC1D0;
  qword_4FE4F38 = (__int64)sub_984030;
  qword_4FE4F10 = (__int64)&unk_49D9748;
  qword_4FE4F18 = 0;
  qword_4FE4E80 = (__int64)&unk_49DC090;
  sub_C53080(&qword_4FE4E80, "hwasan-use-after-scope", 22);
  qword_4FE4EA8 = (__int64)"detect use after scope within function";
  LOWORD(qword_4FE4F18) = 257;
  LOBYTE(qword_4FE4F08) = 1;
  qword_4FE4EB0 = 38;
  byte_4FE4E8C = byte_4FE4E8C & 0x9F | 0x20;
  sub_C53130(&qword_4FE4E80);
  __cxa_atexit(sub_984900, &qword_4FE4E80, &qword_4A427C0);
  sub_D95050(&qword_4FE4DA0, 0, 0);
  qword_4FE4E28 = 0;
  qword_4FE4E60 = (__int64)nullsub_23;
  qword_4FE4E40 = (__int64)&unk_49DC1D0;
  qword_4FE4E58 = (__int64)sub_984030;
  qword_4FE4E30 = (__int64)&unk_49D9748;
  qword_4FE4E38 = 0;
  qword_4FE4DA0 = (__int64)&unk_49DC090;
  sub_C53080(&qword_4FE4DA0, "hwasan-generate-tags-with-calls", 31);
  qword_4FE4DC8 = (__int64)"generate new tags with runtime library calls";
  LOWORD(qword_4FE4E38) = 256;
  LOBYTE(qword_4FE4E28) = 0;
  qword_4FE4DD0 = 44;
  byte_4FE4DAC = byte_4FE4DAC & 0x9F | 0x20;
  sub_C53130(&qword_4FE4DA0);
  __cxa_atexit(sub_984900, &qword_4FE4DA0, &qword_4A427C0);
  v68 = (const char *)v60;
  v75 = "Instrument globals";
  LOBYTE(v60[0]) = 0;
  LODWORD(v64[0]) = 1;
  v76 = 18;
  sub_243BAB0(&unk_4FE4CC0, "hwasan-globals", &v75, v64, &v68);
  __cxa_atexit(sub_984900, &unk_4FE4CC0, &qword_4A427C0);
  sub_D95050(&qword_4FE4BE0, 0, 0);
  qword_4FE4C68 = 0;
  qword_4FE4C78 = 0;
  qword_4FE4C70 = (__int64)&unk_49DA090;
  qword_4FE4BE0 = (__int64)&unk_49DBF90;
  qword_4FE4CA0 = (__int64)nullsub_58;
  qword_4FE4C98 = (__int64)sub_B2B5F0;
  qword_4FE4C80 = (__int64)&unk_49DC230;
  sub_C53080(&qword_4FE4BE0, "hwasan-match-all-tag", 20);
  qword_4FE4C10 = 52;
  qword_4FE4C08 = (__int64)"don't report bad accesses via pointers with this tag";
  LODWORD(qword_4FE4C68) = -1;
  BYTE4(qword_4FE4C78) = 1;
  LODWORD(qword_4FE4C78) = -1;
  byte_4FE4BEC = byte_4FE4BEC & 0x9F | 0x20;
  sub_C53130(&qword_4FE4BE0);
  __cxa_atexit(sub_B2B680, &qword_4FE4BE0, &qword_4A427C0);
  sub_D95050(&qword_4FE4B00, 0, 0);
  qword_4FE4B88 = 0;
  qword_4FE4B90 = (__int64)&unk_49D9748;
  qword_4FE4BA0 = (__int64)&unk_49DC1D0;
  qword_4FE4BC0 = (__int64)nullsub_23;
  qword_4FE4B98 = 0;
  qword_4FE4BB8 = (__int64)sub_984030;
  qword_4FE4B00 = (__int64)&unk_49DC090;
  sub_C53080(&qword_4FE4B00, "hwasan-kernel", 13);
  qword_4FE4B28 = (__int64)"Enable KernelHWAddressSanitizer instrumentation";
  LOWORD(qword_4FE4B98) = 256;
  LOBYTE(qword_4FE4B88) = 0;
  qword_4FE4B30 = 47;
  byte_4FE4B0C = byte_4FE4B0C & 0x9F | 0x20;
  sub_C53130(&qword_4FE4B00);
  __cxa_atexit(sub_984900, &qword_4FE4B00, &qword_4A427C0);
  sub_D95050(&qword_4FE4A20, 0, 0);
  qword_4FE4AE8 = (__int64)nullsub_121;
  qword_4FE4AC8 = (__int64)&unk_49DC2C0;
  qword_4FE4AB0 = (__int64)&unk_49DB998;
  qword_4FE4A20 = (__int64)&unk_49DB9B8;
  qword_4FE4AE0 = (__int64)sub_C1A370;
  qword_4FE4AA8 = 0;
  qword_4FE4AB8 = 0;
  byte_4FE4AC0 = 0;
  sub_C53080(&qword_4FE4A20, "hwasan-mapping-offset", 21);
  qword_4FE4A50 = 43;
  qword_4FE4A48 = (__int64)"HWASan shadow mapping offset [EXPERIMENTAL]";
  byte_4FE4A2C = byte_4FE4A2C & 0x9F | 0x20;
  sub_C53130(&qword_4FE4A20);
  __cxa_atexit(sub_C1A610, &qword_4FE4A20, &qword_4A427C0);
  v66 = "Use TLS";
  v62 = "Use ifunc global";
  v64[0] = "tls";
  v55 = "global";
  v60[0] = "ifunc";
  v64[1] = 3;
  LODWORD(v65) = 3;
  v67 = 7;
  v60[1] = 5;
  LODWORD(v61) = 2;
  *((_QWORD *)&v36 + 1) = "Use TLS";
  v63 = 16;
  *(_QWORD *)&v36 = v65;
  v58 = "Use global";
  *((_QWORD *)&v34 + 1) = 3;
  v56 = 6;
  *(_QWORD *)&v34 = "tls";
  LODWORD(v57) = 1;
  v59 = 10;
  *((_QWORD *)&v32 + 1) = "Use ifunc global";
  *(_QWORD *)&v32 = v61;
  *((_QWORD *)&v30 + 1) = 5;
  *(_QWORD *)&v30 = "ifunc";
  *((_QWORD *)&v28 + 1) = "Use global";
  *(_QWORD *)&v28 = v57;
  *((_QWORD *)&v26 + 1) = 6;
  *(_QWORD *)&v26 = "global";
  sub_22735E0(
    (unsigned int)&v75,
    (unsigned int)&qword_4FE4A20,
    (unsigned int)"ifunc",
    (unsigned int)"tls",
    v0,
    v1,
    v26,
    v28,
    10,
    v30,
    v32,
    16,
    v34,
    v36,
    7);
  sub_D95050(&qword_4FE47C0, 0, 0);
  qword_4FE47C0 = (__int64)&off_4A16758;
  v2 = "hwasan-mapping-offset-dynamic";
  qword_4FE4860 = (__int64)off_4A16708;
  qword_4FE4870 = (__int64)&unk_4FE4880;
  qword_4FE4878 = 0x800000000LL;
  qword_4FE4A18 = (__int64)nullsub_1501;
  qword_4FE4A10 = (__int64)sub_24339A0;
  qword_4FE4848 = 0;
  qword_4FE4858 = 0;
  qword_4FE4850 = (__int64)&off_4A166E8;
  qword_4FE4868 = (__int64)&qword_4FE47C0;
  sub_C53080(&qword_4FE47C0, "hwasan-mapping-offset-dynamic", 29);
  qword_4FE47F0 = 45;
  qword_4FE47E8 = (__int64)"HWASan shadow mapping dynamic offset location";
  byte_4FE47CC = byte_4FE47CC & 0x9F | 0x20;
  if ( v75 != &v75[40 * (unsigned int)v76] )
  {
    v3 = &v75[40 * (unsigned int)v76];
    v4 = v75;
    do
    {
      v5 = *((_QWORD *)v4 + 3);
      v6 = (const char *)*((_QWORD *)v4 + 4);
      v7 = *(const char **)v4;
      v8 = *((_QWORD *)v4 + 1);
      v73 = *((_DWORD *)v4 + 4);
      v9 = (unsigned int)qword_4FE4878;
      v70 = v5;
      v10 = (const __m128i *)&v68;
      v71 = v6;
      v68 = v7;
      v11 = qword_4FE4878;
      v69 = v8;
      v72 = (__int64)&off_4A166E8;
      v74 = 1;
      if ( (unsigned __int64)(unsigned int)qword_4FE4878 + 1 > HIDWORD(qword_4FE4878) )
      {
        if ( qword_4FE4870 > (unsigned __int64)&v68
          || (unsigned __int64)&v68 >= qword_4FE4870 + 48 * (unsigned __int64)(unsigned int)qword_4FE4878 )
        {
          v39 = -1;
          v40 = 0;
        }
        else
        {
          v40 = 1;
          v39 = 0xAAAAAAAAAAAAAAABLL * (((__int64)&v68 - qword_4FE4870) >> 4);
        }
        v41 = v8;
        v43 = v7;
        v19 = (__m128i *)sub_C8D7D0((char *)&unk_4FE4880 - 16, &unk_4FE4880, (unsigned int)qword_4FE4878 + 1LL, 48, v51);
        v20 = (void *)qword_4FE4870;
        v7 = v43;
        v21 = (__int64)v19;
        v8 = v41;
        v22 = (const __m128i *)qword_4FE4870;
        v23 = qword_4FE4870 + 48LL * (unsigned int)qword_4FE4878;
        if ( qword_4FE4870 != v23 )
        {
          do
          {
            if ( v19 )
            {
              *v19 = _mm_loadu_si128(v22);
              v19[1] = _mm_loadu_si128(v22 + 1);
              v19[2].m128i_i32[2] = v22[2].m128i_i32[2];
              v24 = v22[2].m128i_i8[12];
              v19[2].m128i_i64[0] = (__int64)&off_4A166E8;
              v19[2].m128i_i8[12] = v24;
            }
            v22 += 3;
            v19 += 3;
          }
          while ( (const __m128i *)v23 != v22 );
          v20 = (void *)qword_4FE4870;
        }
        v25 = v51[0];
        if ( v20 != &unk_4FE4880 )
        {
          v38 = v41;
          v42 = v51[0];
          _libc_free(v20, v21);
          v8 = v38;
          v7 = v43;
          v25 = v42;
        }
        HIDWORD(qword_4FE4878) = v25;
        v9 = (unsigned int)qword_4FE4878;
        v10 = (const __m128i *)&v68;
        qword_4FE4870 = v21;
        v11 = qword_4FE4878;
        if ( v40 )
          v10 = (const __m128i *)(v21 + 48 * v39);
      }
      v12 = qword_4FE4870 + 48 * v9;
      if ( v12 )
      {
        v13 = _mm_loadu_si128(v10 + 1);
        *(__m128i *)v12 = _mm_loadu_si128(v10);
        *(__m128i *)(v12 + 16) = v13;
        *(_DWORD *)(v12 + 40) = v10[2].m128i_i32[2];
        v14 = v10[2].m128i_i8[12];
        *(_QWORD *)(v12 + 32) = &off_4A166E8;
        *(_BYTE *)(v12 + 44) = v14;
        v11 = qword_4FE4878;
      }
      v2 = v7;
      v4 += 40;
      LODWORD(qword_4FE4878) = v11 + 1;
      sub_C52F90(qword_4FE4868, v7, v8, v10);
    }
    while ( v3 != v4 );
  }
  sub_C53130(&qword_4FE47C0);
  if ( v75 != v77 )
    _libc_free(v75, v2);
  __cxa_atexit(sub_2434740, &qword_4FE47C0, &qword_4A427C0);
  sub_D95050(&qword_4FE46E0, 0, 0);
  qword_4FE4768 = 0;
  qword_4FE4778 = 0;
  qword_4FE4770 = (__int64)&unk_49D9748;
  qword_4FE46E0 = (__int64)&unk_49DC090;
  qword_4FE4780 = (__int64)&unk_49DC1D0;
  qword_4FE47A0 = (__int64)nullsub_23;
  qword_4FE4798 = (__int64)sub_984030;
  sub_C53080(&qword_4FE46E0, "hwasan-with-frame-record", 24);
  qword_4FE4710 = 37;
  qword_4FE4708 = (__int64)"Use ring buffer for stack allocations";
  byte_4FE46EC = byte_4FE46EC & 0x9F | 0x20;
  sub_C53130(&qword_4FE46E0);
  __cxa_atexit(sub_984900, &qword_4FE46E0, &qword_4A427C0);
  sub_D95050(&qword_4FE4600, 0, 0);
  qword_4FE4688 = 0;
  qword_4FE4698 = 0;
  qword_4FE4690 = (__int64)&unk_49DA090;
  qword_4FE4600 = (__int64)&unk_49DBF90;
  qword_4FE46A0 = (__int64)&unk_49DC230;
  qword_4FE46C0 = (__int64)nullsub_58;
  qword_4FE46B8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_4FE4600, "hwasan-percentile-cutoff-hot", 28);
  qword_4FE4630 = 22;
  qword_4FE4628 = (__int64)"Hot percentile cutoff.";
  sub_C53130(&qword_4FE4600);
  __cxa_atexit(sub_B2B680, &qword_4FE4600, &qword_4A427C0);
  sub_D95050(&qword_4FE4520, 0, 0);
  qword_4FE45A8 = 0;
  qword_4FE45B8 = 0;
  qword_4FE45B0 = (__int64)&unk_49E5940;
  qword_4FE4520 = (__int64)&unk_49E5960;
  qword_4FE45C0 = (__int64)&unk_49DC320;
  qword_4FE45E0 = (__int64)nullsub_385;
  qword_4FE45D8 = (__int64)sub_1038930;
  sub_C53080(&qword_4FE4520, "hwasan-random-rate", 18);
  qword_4FE4550 = 189;
  qword_4FE4548 = (__int64)"Probability value in the range [0.0, 1.0] to keep instrumentation of a function. Note: instru"
                           "mentation can be skipped randomly OR because of the hot percentile cutoff, if both are supplied.";
  sub_C53130(&qword_4FE4520);
  __cxa_atexit(sub_1038DB0, &qword_4FE4520, &qword_4A427C0);
  v71 = "Add a call to __hwasan_add_frame_record for storing into the stack ring buffer";
  v53 = "Insert instructions into the prologue for storing into the stack ring buffer directly";
  v68 = "libcall";
  v69 = 7;
  LODWORD(v70) = 2;
  v72 = 78;
  v51[0] = "instr";
  LODWORD(v52) = 1;
  *((_QWORD *)&v37 + 1) = "Add a call to __hwasan_add_frame_record for storing into the stack ring buffer";
  v54 = 85;
  *(_QWORD *)&v37 = v70;
  v49 = "Do not record stack ring history";
  *((_QWORD *)&v35 + 1) = 7;
  v45 = 1;
  *(_QWORD *)&v35 = "libcall";
  v46 = &v45;
  v44 = 1;
  *((_QWORD *)&v33 + 1) = "Insert instructions into the prologue for storing into the stack ring buffer directly";
  v51[1] = 5;
  *(_QWORD *)&v33 = v52;
  v47[3] = 4;
  LODWORD(v48) = 0;
  v50 = 32;
  *((_QWORD *)&v31 + 1) = 5;
  *(_QWORD *)&v31 = "instr";
  *((_QWORD *)&v29 + 1) = "Do not record stack ring history";
  *(_QWORD *)&v29 = v48;
  *((_QWORD *)&v27 + 1) = 4;
  *(_QWORD *)&v27 = "none";
  sub_22735E0(
    (unsigned int)&v75,
    (unsigned int)&qword_4FE4520,
    (unsigned int)"libcall",
    v15,
    v16,
    v17,
    v27,
    v29,
    32,
    v31,
    v33,
    85,
    v35,
    v37,
    78);
  v47[1] = 73;
  v47[0] = "Record stack frames with tagged allocations in a thread-local ring buffer";
  sub_24431B0(&unk_4FE42C0, "hwasan-record-stack-history", v47, &v75, &v44, &v46);
  if ( v75 != v77 )
    _libc_free(v75, "hwasan-record-stack-history");
  __cxa_atexit(sub_24347D0, &unk_4FE42C0, &qword_4A427C0);
  sub_D95050(&qword_4FE41E0, 0, 0);
  qword_4FE4268 = 0;
  qword_4FE42A0 = (__int64)nullsub_23;
  qword_4FE4270 = (__int64)&unk_49D9748;
  qword_4FE41E0 = (__int64)&unk_49DC090;
  qword_4FE4280 = (__int64)&unk_49DC1D0;
  qword_4FE4298 = (__int64)sub_984030;
  qword_4FE4278 = 0;
  sub_C53080(&qword_4FE41E0, "hwasan-instrument-mem-intrinsics", 32);
  qword_4FE4210 = 28;
  qword_4FE4208 = (__int64)"instrument memory intrinsics";
  LOBYTE(qword_4FE4268) = 1;
  byte_4FE41EC = byte_4FE41EC & 0x9F | 0x20;
  LOWORD(qword_4FE4278) = 257;
  sub_C53130(&qword_4FE41E0);
  __cxa_atexit(sub_984900, &qword_4FE41E0, &qword_4A427C0);
  v75 = "instrument landing pads";
  v47[0] = &v45;
  LOBYTE(v45) = 0;
  LODWORD(v46) = 1;
  v76 = 23;
  sub_243BCC0(&unk_4FE4100, "hwasan-instrument-landing-pads", &v75, &v46, v47);
  __cxa_atexit(sub_984900, &unk_4FE4100, &qword_4A427C0);
  v75 = "use short granules in allocas and outlined checks";
  v47[0] = &v45;
  LOBYTE(v45) = 0;
  LODWORD(v46) = 1;
  v76 = 49;
  sub_23E6860(&unk_4FE4020, "hwasan-use-short-granules", &v75, &v46, v47);
  __cxa_atexit(sub_984900, &unk_4FE4020, &qword_4A427C0);
  sub_D95050(&qword_4FE3F40, 0, 0);
  qword_4FE4000 = (__int64)nullsub_23;
  qword_4FE3FD0 = (__int64)&unk_49D9748;
  qword_4FE3F40 = (__int64)&unk_49DC090;
  qword_4FE3FE0 = (__int64)&unk_49DC1D0;
  qword_4FE3FF8 = (__int64)sub_984030;
  qword_4FE3FC8 = 0;
  qword_4FE3FD8 = 0;
  sub_C53080(&qword_4FE3F40, "hwasan-instrument-personality-functions", 39);
  qword_4FE3F70 = 32;
  qword_4FE3F68 = (__int64)"instrument personality functions";
  byte_4FE3F4C = byte_4FE3F4C & 0x9F | 0x20;
  sub_C53130(&qword_4FE3F40);
  __cxa_atexit(sub_984900, &qword_4FE3F40, &qword_4A427C0);
  v47[0] = &v45;
  v75 = "inline all checks";
  LOBYTE(v45) = 0;
  LODWORD(v46) = 1;
  v76 = 17;
  sub_243B8A0(&unk_4FE3E60, "hwasan-inline-all-checks", &v75, &v46, v47);
  __cxa_atexit(sub_984900, &unk_4FE3E60, &qword_4A427C0);
  v47[0] = &v45;
  v75 = "inline all checks";
  LOBYTE(v45) = 0;
  LODWORD(v46) = 1;
  v76 = 17;
  sub_243BCC0(&unk_4FE3D80, "hwasan-inline-fast-path-checks", &v75, &v46, v47);
  __cxa_atexit(sub_984900, &unk_4FE3D80, &qword_4A427C0);
  sub_D95050(&qword_4FE3CA0, 0, 0);
  qword_4FE3D28 = 0;
  qword_4FE3D40 = (__int64)&unk_49DC1D0;
  qword_4FE3D30 = (__int64)&unk_49D9748;
  qword_4FE3D60 = (__int64)nullsub_23;
  qword_4FE3CA0 = (__int64)&unk_49DC090;
  qword_4FE3D58 = (__int64)sub_984030;
  qword_4FE3D38 = 0;
  sub_C53080(&qword_4FE3CA0, "hwasan-experimental-use-page-aliases", 36);
  qword_4FE3CC8 = (__int64)"Use page aliasing in HWASan";
  LOWORD(qword_4FE3D38) = 256;
  LOBYTE(qword_4FE3D28) = 0;
  qword_4FE3CD0 = 27;
  byte_4FE3CAC = byte_4FE3CAC & 0x9F | 0x20;
  sub_C53130(&qword_4FE3CA0);
  return __cxa_atexit(sub_984900, &qword_4FE3CA0, &qword_4A427C0);
}
