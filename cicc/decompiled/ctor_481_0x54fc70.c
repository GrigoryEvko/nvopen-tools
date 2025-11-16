// Function: ctor_481
// Address: 0x54fc70
//
int ctor_481()
{
  __int64 v0; // rax
  const char *v2; // [rsp+10h] [rbp-60h] BYREF
  char v3; // [rsp+30h] [rbp-40h]
  char v4; // [rsp+31h] [rbp-3Fh]

  sub_D95050(&qword_50052C0, 0, 0);
  qword_5005348 = 0;
  qword_5005358 = 0;
  qword_5005350 = (__int64)&unk_49D9748;
  qword_5005380 = (__int64)nullsub_23;
  qword_50052C0 = (__int64)&unk_49DC090;
  qword_5005360 = (__int64)&unk_49DC1D0;
  qword_5005378 = (__int64)sub_984030;
  sub_C53080(&qword_50052C0, "spp-print-liveset", 17);
  LOWORD(qword_5005358) = 256;
  LOBYTE(qword_5005348) = 0;
  byte_50052CC = byte_50052CC & 0x9F | 0x20;
  sub_C53130(&qword_50052C0);
  __cxa_atexit(sub_984900, &qword_50052C0, &qword_4A427C0);
  sub_D95050(&qword_50051E0, 0, 0);
  qword_5005270 = (__int64)&unk_49D9748;
  qword_50051E0 = (__int64)&unk_49DC090;
  qword_5005268 = 0;
  qword_5005278 = 0;
  qword_5005280 = (__int64)&unk_49DC1D0;
  qword_50052A0 = (__int64)nullsub_23;
  qword_5005298 = (__int64)sub_984030;
  sub_C53080(&qword_50051E0, "spp-print-liveset-size", 22);
  LOWORD(qword_5005278) = 256;
  LOBYTE(qword_5005268) = 0;
  byte_50051EC = byte_50051EC & 0x9F | 0x20;
  sub_C53130(&qword_50051E0);
  __cxa_atexit(sub_984900, &qword_50051E0, &qword_4A427C0);
  sub_D95050(&qword_5005100, 0, 0);
  qword_5005190 = (__int64)&unk_49D9748;
  qword_5005100 = (__int64)&unk_49DC090;
  qword_5005188 = 0;
  qword_5005198 = 0;
  qword_50051A0 = (__int64)&unk_49DC1D0;
  qword_50051C0 = (__int64)nullsub_23;
  qword_50051B8 = (__int64)sub_984030;
  sub_C53080(&qword_5005100, "spp-print-base-pointers", 23);
  LOBYTE(qword_5005188) = 0;
  LOWORD(qword_5005198) = 256;
  byte_500510C = byte_500510C & 0x9F | 0x20;
  sub_C53130(&qword_5005100);
  __cxa_atexit(sub_984900, &qword_5005100, &qword_4A427C0);
  sub_D95050(&qword_5005020, 0, 0);
  qword_50050A8 = 0;
  qword_50050B8 = 0;
  qword_50050B0 = (__int64)&unk_49D9728;
  qword_5005020 = (__int64)&unk_49DBF10;
  qword_50050C0 = (__int64)&unk_49DC290;
  qword_50050E0 = (__int64)nullsub_24;
  qword_50050D8 = (__int64)sub_984050;
  sub_C53080(&qword_5005020, "spp-rematerialization-threshold", 31);
  LODWORD(qword_50050A8) = 6;
  BYTE4(qword_50050B8) = 1;
  LODWORD(qword_50050B8) = 6;
  byte_500502C = byte_500502C & 0x9F | 0x20;
  sub_C53130(&qword_5005020);
  __cxa_atexit(sub_984970, &qword_5005020, &qword_4A427C0);
  sub_D95050(&qword_5004F40, 0, 0);
  byte_5004FD9 = 0;
  qword_5004FD0 = (__int64)&unk_49D9748;
  qword_5004F40 = (__int64)&unk_49D9AD8;
  qword_5004FC8 = 0;
  qword_5005000 = (__int64)nullsub_39;
  qword_5004FE0 = (__int64)&unk_49DC1D0;
  qword_5004FF8 = (__int64)sub_AA4180;
  sub_C53080(&qword_5004F40, "rs4gc-clobber-non-live", 22);
  if ( qword_5004FC8 )
  {
    v0 = sub_CEADF0();
    v4 = 1;
    v2 = "cl::location(x) specified more than once!";
    v3 = 3;
    sub_C53280(&qword_5004F40, &v2, 0, 0, v0);
  }
  else
  {
    byte_5004FD9 = 1;
    qword_5004FC8 = (__int64)&byte_5005008;
    byte_5004FD8 = byte_5005008;
  }
  byte_5004F4C = byte_5004F4C & 0x9F | 0x20;
  sub_C53130(&qword_5004F40);
  __cxa_atexit(sub_AA4490, &qword_5004F40, &qword_4A427C0);
  sub_D95050(&qword_5004E60, 0, 0);
  qword_5004F20 = (__int64)nullsub_23;
  qword_5004F18 = (__int64)sub_984030;
  qword_5004EF0 = (__int64)&unk_49D9748;
  qword_5004E60 = (__int64)&unk_49DC090;
  qword_5004EE8 = 0;
  qword_5004F00 = (__int64)&unk_49DC1D0;
  qword_5004EF8 = 0;
  sub_C53080(&qword_5004E60, "rs4gc-allow-statepoint-with-no-deopt-info", 41);
  LOBYTE(qword_5004EE8) = 1;
  byte_5004E6C = byte_5004E6C & 0x9F | 0x20;
  LOWORD(qword_5004EF8) = 257;
  sub_C53130(&qword_5004E60);
  __cxa_atexit(sub_984900, &qword_5004E60, &qword_4A427C0);
  sub_D95050(&qword_5004D80, 0, 0);
  qword_5004D80 = (__int64)&unk_49DC090;
  qword_5004E10 = (__int64)&unk_49D9748;
  qword_5004E20 = (__int64)&unk_49DC1D0;
  qword_5004E40 = (__int64)nullsub_23;
  qword_5004E38 = (__int64)sub_984030;
  qword_5004E08 = 0;
  qword_5004E18 = 0;
  sub_C53080(&qword_5004D80, "rs4gc-remat-derived-at-uses", 27);
  LOWORD(qword_5004E18) = 257;
  LOBYTE(qword_5004E08) = 1;
  byte_5004D8C = byte_5004D8C & 0x9F | 0x20;
  sub_C53130(&qword_5004D80);
  return __cxa_atexit(sub_984900, &qword_5004D80, &qword_4A427C0);
}
