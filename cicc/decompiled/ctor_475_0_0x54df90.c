// Function: ctor_475_0
// Address: 0x54df90
//
int ctor_475_0()
{
  int v1; // [rsp+24h] [rbp-FCh] BYREF
  int *v2; // [rsp+28h] [rbp-F8h] BYREF
  _QWORD v3[2]; // [rsp+30h] [rbp-F0h] BYREF
  const char *v4; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v5; // [rsp+48h] [rbp-D8h]
  _QWORD v6[2]; // [rsp+50h] [rbp-D0h] BYREF
  int v7; // [rsp+60h] [rbp-C0h]
  const char *v8; // [rsp+68h] [rbp-B8h]
  __int64 v9; // [rsp+70h] [rbp-B0h]
  const char *v10; // [rsp+78h] [rbp-A8h]
  __int64 v11; // [rsp+80h] [rbp-A0h]
  int v12; // [rsp+88h] [rbp-98h]
  const char *v13; // [rsp+90h] [rbp-90h]
  __int64 v14; // [rsp+98h] [rbp-88h]

  sub_D95050(&qword_5004300, 0, 0);
  qword_5004388 = 0;
  qword_5004398 = 0;
  qword_5004390 = (__int64)&unk_49D9748;
  qword_5004300 = (__int64)&unk_49DC090;
  qword_50043A0 = (__int64)&unk_49DC1D0;
  qword_50043C0 = (__int64)nullsub_23;
  qword_50043B8 = (__int64)sub_984030;
  sub_C53080(&qword_5004300, "fuse-matrix", 11);
  LOWORD(qword_5004398) = 257;
  LOBYTE(qword_5004388) = 1;
  qword_5004330 = 42;
  byte_500430C = byte_500430C & 0x9F | 0x20;
  qword_5004328 = (__int64)"Enable/disable fusing matrix instructions.";
  sub_C53130(&qword_5004300);
  __cxa_atexit(sub_984900, &qword_5004300, &qword_4A427C0);
  sub_D95050(&qword_5004220, 0, 0);
  qword_50042A8 = 0;
  qword_50042B8 = 0;
  qword_50042B0 = (__int64)&unk_49D9728;
  qword_5004220 = (__int64)&unk_49DBF10;
  qword_50042C0 = (__int64)&unk_49DC290;
  qword_50042E0 = (__int64)nullsub_24;
  qword_50042D8 = (__int64)sub_984050;
  sub_C53080(&qword_5004220, "fuse-matrix-tile-size", 21);
  LODWORD(qword_50042A8) = 4;
  BYTE4(qword_50042B8) = 1;
  LODWORD(qword_50042B8) = 4;
  qword_5004250 = 66;
  byte_500422C = byte_500422C & 0x9F | 0x20;
  qword_5004248 = (__int64)"Tile size for matrix instruction fusion using square-shaped tiles.";
  sub_C53130(&qword_5004220);
  __cxa_atexit(sub_984970, &qword_5004220, &qword_4A427C0);
  v4 = "Generate loop nest for tiling.";
  v5 = 30;
  LODWORD(v2) = 1;
  LOBYTE(v1) = 0;
  v3[0] = &v1;
  sub_23A1680(&unk_5004140, "fuse-matrix-use-loops", v3, &v2, &v4);
  __cxa_atexit(sub_984900, &unk_5004140, &qword_4A427C0);
  sub_D95050(&qword_5004060, 0, 0);
  qword_50040E8 = 0;
  qword_5004120 = (__int64)nullsub_23;
  qword_50040F0 = (__int64)&unk_49D9748;
  qword_5004060 = (__int64)&unk_49DC090;
  qword_5004118 = (__int64)sub_984030;
  qword_50040F8 = 0;
  qword_5004100 = (__int64)&unk_49DC1D0;
  sub_C53080(&qword_5004060, "force-fuse-matrix", 17);
  LOWORD(qword_50040F8) = 256;
  LOBYTE(qword_50040E8) = 0;
  qword_5004090 = 55;
  byte_500406C = byte_500406C & 0x9F | 0x20;
  qword_5004088 = (__int64)"Force matrix instruction fusion even if not profitable.";
  sub_C53130(&qword_5004060);
  __cxa_atexit(sub_984900, &qword_5004060, &qword_4A427C0);
  v5 = 116;
  v4 = "Allow the use of FMAs if available and profitable. This may result in different results, due to less rounding error.";
  LODWORD(v2) = 1;
  LOBYTE(v1) = 0;
  v3[0] = &v1;
  sub_23A1680(&unk_5003F80, "matrix-allow-contract", v3, &v2, &v4);
  __cxa_atexit(sub_984900, &unk_5003F80, &qword_4A427C0);
  sub_D95050(&qword_5003EA0, 0, 0);
  qword_5003F28 = 0;
  qword_5003F60 = (__int64)nullsub_23;
  qword_5003F30 = (__int64)&unk_49D9748;
  qword_5003EA0 = (__int64)&unk_49DC090;
  qword_5003F58 = (__int64)sub_984030;
  qword_5003F38 = 0;
  qword_5003F40 = (__int64)&unk_49DC1D0;
  sub_C53080(&qword_5003EA0, "verify-matrix-shapes", 20);
  LOWORD(qword_5003F38) = 256;
  LOBYTE(qword_5003F28) = 0;
  qword_5003ED0 = 41;
  byte_5003EAC = byte_5003EAC & 0x9F | 0x20;
  qword_5003EC8 = (__int64)"Enable/disable matrix shape verification.";
  sub_C53130(&qword_5003EA0);
  __cxa_atexit(sub_984900, &qword_5003EA0, &qword_4A427C0);
  v4 = (const char *)v6;
  v6[0] = "column-major";
  v8 = "Use column-major layout";
  v10 = "row-major";
  v13 = "Use row-major layout";
  v5 = 0x400000002LL;
  v3[0] = "Sets the default matrix layout";
  v6[1] = 12;
  v7 = 0;
  v9 = 23;
  v11 = 9;
  v12 = 1;
  v14 = 20;
  v3[1] = 30;
  v1 = 0;
  v2 = &v1;
  sub_28A17F0(&unk_5003C40, "matrix-default-layout", &v2, v3, &v4);
  if ( v4 != (const char *)v6 )
    _libc_free(v4, "matrix-default-layout");
  __cxa_atexit(sub_2895160, &unk_5003C40, &qword_4A427C0);
  sub_D95050(&qword_5003B60, 0, 0);
  qword_5003BE8 = 0;
  qword_5003BF8 = 0;
  qword_5003BF0 = (__int64)&unk_49D9748;
  qword_5003B60 = (__int64)&unk_49DC090;
  qword_5003C00 = (__int64)&unk_49DC1D0;
  qword_5003C20 = (__int64)nullsub_23;
  qword_5003C18 = (__int64)sub_984030;
  sub_C53080(&qword_5003B60, "matrix-print-after-transpose-opt", 32);
  LOBYTE(qword_5003BE8) = 0;
  LOWORD(qword_5003BF8) = 256;
  sub_C53130(&qword_5003B60);
  return __cxa_atexit(sub_984900, &qword_5003B60, &qword_4A427C0);
}
