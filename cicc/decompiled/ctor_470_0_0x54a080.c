// Function: ctor_470_0
// Address: 0x54a080
//
int ctor_470_0()
{
  int v1; // [rsp+30h] [rbp-100h] BYREF
  int v2; // [rsp+34h] [rbp-FCh] BYREF
  int *v3; // [rsp+38h] [rbp-F8h] BYREF
  _QWORD v4[2]; // [rsp+40h] [rbp-F0h] BYREF
  const char *v5; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v6; // [rsp+58h] [rbp-D8h]
  _QWORD v7[2]; // [rsp+60h] [rbp-D0h] BYREF
  int v8; // [rsp+70h] [rbp-C0h]
  const char *v9; // [rsp+78h] [rbp-B8h]
  __int64 v10; // [rsp+80h] [rbp-B0h]
  const char *v11; // [rsp+88h] [rbp-A8h]
  __int64 v12; // [rsp+90h] [rbp-A0h]
  int v13; // [rsp+98h] [rbp-98h]
  const char *v14; // [rsp+A0h] [rbp-90h]
  __int64 v15; // [rsp+A8h] [rbp-88h]
  const char *v16; // [rsp+B0h] [rbp-80h]
  __int64 v17; // [rsp+B8h] [rbp-78h]
  int v18; // [rsp+C0h] [rbp-70h]
  const char *v19; // [rsp+C8h] [rbp-68h]
  __int64 v20; // [rsp+D0h] [rbp-60h]

  sub_D95050(&qword_5001860, 0, 0);
  qword_50018E8 = 0;
  qword_50018F8 = 0;
  qword_50018F0 = (__int64)&unk_49D9748;
  qword_5001860 = (__int64)&unk_49DC090;
  qword_5001900 = (__int64)&unk_49DC1D0;
  qword_5001920 = (__int64)nullsub_23;
  qword_5001918 = (__int64)sub_984030;
  sub_C53080(&qword_5001860, "enable-lsr-phielim", 18);
  LOWORD(qword_50018F8) = 257;
  LOBYTE(qword_50018E8) = 1;
  qword_5001890 = 26;
  byte_500186C = byte_500186C & 0x9F | 0x20;
  qword_5001888 = (__int64)"Enable LSR phi elimination";
  sub_C53130(&qword_5001860);
  __cxa_atexit(sub_984900, &qword_5001860, &qword_4A427C0);
  v5 = "Add instruction count to a LSR cost model";
  v4[0] = &v2;
  v6 = 41;
  LOBYTE(v2) = 1;
  LODWORD(v3) = 1;
  sub_285D7F0(&unk_5001780, "lsr-insns-cost", &v3, v4, &v5);
  __cxa_atexit(sub_984900, &unk_5001780, &qword_4A427C0);
  v5 = "Narrow LSR complex solution using expectation of registers number";
  v4[0] = &v2;
  v6 = 65;
  LOBYTE(v2) = 0;
  LODWORD(v3) = 1;
  sub_285D7F0(&unk_50016A0, "lsr-exp-narrow", &v3, v4, &v5);
  __cxa_atexit(sub_984900, &unk_50016A0, &qword_4A427C0);
  sub_D95050(&qword_50015C0, 0, 0);
  qword_50015C0 = (__int64)&unk_49DC090;
  qword_5001660 = (__int64)&unk_49DC1D0;
  qword_5001680 = (__int64)nullsub_23;
  qword_5001650 = (__int64)&unk_49D9748;
  qword_5001678 = (__int64)sub_984030;
  qword_5001648 = 0;
  qword_5001658 = 0;
  sub_C53080(&qword_50015C0, "lsr-filter-same-scaled-reg", 26);
  LOWORD(qword_5001658) = 257;
  LOBYTE(qword_5001648) = 1;
  qword_50015F0 = 91;
  qword_50015E8 = (__int64)"Narrow LSR search space by filtering non-optimal formulae with the same ScaledReg and Scale";
  byte_50015CC = byte_50015CC & 0x9F | 0x20;
  sub_C53130(&qword_50015C0);
  __cxa_atexit(sub_984900, &qword_50015C0, &qword_4A427C0);
  v7[0] = "none";
  v9 = "Don't prefer any addressing mode";
  v14 = "Prefer pre-indexed addressing mode";
  v11 = "preindexed";
  v19 = "Prefer post-indexed addressing mode";
  v16 = "postindexed";
  v4[0] = "A flag that overrides the target's preferred addressing mode.";
  v5 = (const char *)v7;
  v7[1] = 4;
  v8 = 2;
  v10 = 32;
  v12 = 10;
  v13 = 0;
  v15 = 34;
  v17 = 11;
  v18 = 1;
  v20 = 35;
  v6 = 0x400000003LL;
  v4[1] = 61;
  v2 = 2;
  v3 = &v2;
  v1 = 1;
  sub_287B610(&unk_5001360, "lsr-preferred-addressing-mode", &v1, &v3, v4, &v5);
  if ( v5 != (const char *)v7 )
    _libc_free(v5, "lsr-preferred-addressing-mode");
  __cxa_atexit(sub_2851A70, &unk_5001360, &qword_4A427C0);
  sub_D95050(&qword_5001280, 0, 0);
  qword_5001308 = 0;
  qword_5001318 = 0;
  qword_5001310 = (__int64)&unk_49D9728;
  qword_5001280 = (__int64)&unk_49DBF10;
  qword_5001320 = (__int64)&unk_49DC290;
  qword_5001340 = (__int64)nullsub_24;
  qword_5001338 = (__int64)sub_984050;
  sub_C53080(&qword_5001280, "lsr-complexity-limit", 20);
  LODWORD(qword_5001308) = 0xFFFF;
  BYTE4(qword_5001318) = 1;
  LODWORD(qword_5001318) = 0xFFFF;
  qword_50012B0 = 33;
  byte_500128C = byte_500128C & 0x9F | 0x20;
  qword_50012A8 = (__int64)"LSR search space complexity limit";
  sub_C53130(&qword_5001280);
  __cxa_atexit(sub_984970, &qword_5001280, &qword_4A427C0);
  sub_D95050(&qword_50011A0, 0, 0);
  qword_5001260 = (__int64)nullsub_24;
  qword_5001230 = (__int64)&unk_49D9728;
  qword_50011A0 = (__int64)&unk_49DBF10;
  qword_5001240 = (__int64)&unk_49DC290;
  qword_5001258 = (__int64)sub_984050;
  qword_5001228 = 0;
  qword_5001238 = 0;
  sub_C53080(&qword_50011A0, "lsr-setupcost-depth-limit", 25);
  LODWORD(qword_5001228) = 7;
  BYTE4(qword_5001238) = 1;
  LODWORD(qword_5001238) = 7;
  qword_50011D0 = 48;
  byte_50011AC = byte_50011AC & 0x9F | 0x20;
  qword_50011C8 = (__int64)"The limit on recursion depth for LSRs setup cost";
  sub_C53130(&qword_50011A0);
  __cxa_atexit(sub_984970, &qword_50011A0, &qword_4A427C0);
  sub_D95050(&qword_50010C0, 0, 0);
  qword_5001148 = 0;
  qword_5001158 = 0;
  qword_5001150 = (__int64)&unk_49DC110;
  qword_50010C0 = (__int64)&unk_49D97F0;
  qword_5001160 = (__int64)&unk_49DC200;
  qword_5001180 = (__int64)nullsub_26;
  qword_5001178 = (__int64)sub_9C26D0;
  sub_C53080(&qword_50010C0, "lsr-drop-solution", 17);
  qword_50010F0 = 49;
  byte_50010CC = byte_50010CC & 0x9F | 0x20;
  qword_50010E8 = (__int64)"Attempt to drop solution if it is less profitable";
  sub_C53130(&qword_50010C0);
  __cxa_atexit(sub_9C44F0, &qword_50010C0, &qword_4A427C0);
  sub_D95050(&qword_5000FE0, 0, 0);
  qword_5001068 = 0;
  qword_5001070 = (__int64)&unk_49D9748;
  qword_50010A0 = (__int64)nullsub_23;
  qword_5000FE0 = (__int64)&unk_49DC090;
  qword_5001080 = (__int64)&unk_49DC1D0;
  qword_5001098 = (__int64)sub_984030;
  qword_5001078 = 0;
  sub_C53080(&qword_5000FE0, "lsr-enable-vscale-immediates", 28);
  LOBYTE(qword_5001068) = 1;
  qword_5001010 = 52;
  qword_5001008 = (__int64)"Enable analysis of vscale-relative immediates in LSR";
  byte_5000FEC = byte_5000FEC & 0x9F | 0x20;
  LOWORD(qword_5001078) = 257;
  sub_C53130(&qword_5000FE0);
  __cxa_atexit(sub_984900, &qword_5000FE0, &qword_4A427C0);
  sub_D95050(&qword_5000F00, 0, 0);
  qword_5000F90 = (__int64)&unk_49D9748;
  qword_5000FC0 = (__int64)nullsub_23;
  qword_5000F00 = (__int64)&unk_49DC090;
  qword_5000FA0 = (__int64)&unk_49DC1D0;
  qword_5000FB8 = (__int64)sub_984030;
  qword_5000F88 = 0;
  qword_5000F98 = 0;
  sub_C53080(&qword_5000F00, "lsr-drop-scaled-reg-for-vscale", 30);
  LOWORD(qword_5000F98) = 257;
  LOBYTE(qword_5000F88) = 1;
  qword_5000F30 = 60;
  qword_5000F28 = (__int64)"Avoid using scaled registers with vscale-relative addressing";
  byte_5000F0C = byte_5000F0C & 0x9F | 0x20;
  sub_C53130(&qword_5000F00);
  __cxa_atexit(sub_984900, &qword_5000F00, &qword_4A427C0);
  sub_D95050(&qword_5000E20, 0, 0);
  qword_5000EB0 = (__int64)&unk_49D9748;
  qword_5000EE0 = (__int64)nullsub_23;
  qword_5000E20 = (__int64)&unk_49DC090;
  qword_5000EC0 = (__int64)&unk_49DC1D0;
  qword_5000ED8 = (__int64)sub_984030;
  qword_5000EA8 = 0;
  qword_5000EB8 = 0;
  sub_C53080(&qword_5000E20, "lsr-fix-iv-inc", 14);
  LOWORD(qword_5000EB8) = 257;
  qword_5000E48 = (__int64)"Try to make loop IV increment staying inside loop exiting block";
  LOBYTE(qword_5000EA8) = 1;
  qword_5000E50 = 63;
  byte_5000E2C = byte_5000E2C & 0x9F | 0x20;
  sub_C53130(&qword_5000E20);
  __cxa_atexit(sub_984900, &qword_5000E20, &qword_4A427C0);
  v5 = "Disable loop strength reduce for unknown trip loop ";
  LODWORD(v3) = 1;
  v4[0] = &v2;
  v6 = 51;
  LOBYTE(v2) = 1;
  sub_285DA00(&unk_5000D40, "disable-unknown-trip-lsr", v4, &v5, &v3);
  __cxa_atexit(sub_984900, &unk_5000D40, &qword_4A427C0);
  sub_D95050(&qword_5000C60, 0, 0);
  qword_5000CF0 = (__int64)&unk_49D9748;
  qword_5000D20 = (__int64)nullsub_23;
  qword_5000C60 = (__int64)&unk_49DC090;
  qword_5000D18 = (__int64)sub_984030;
  qword_5000D00 = (__int64)&unk_49DC1D0;
  qword_5000CE8 = 0;
  qword_5000CF8 = 0;
  sub_C53080(&qword_5000C60, "lsr-skip-outer-loop", 19);
  LOBYTE(qword_5000CE8) = 1;
  qword_5000C88 = (__int64)"Ignore outer loop IV in LSR";
  LOWORD(qword_5000CF8) = 257;
  qword_5000C90 = 27;
  byte_5000C6C = byte_5000C6C & 0x9F | 0x20;
  sub_C53130(&qword_5000C60);
  __cxa_atexit(sub_984900, &qword_5000C60, &qword_4A427C0);
  sub_D95050(&qword_5000B80, 0, 0);
  qword_5000C08 = 0;
  qword_5000C20 = (__int64)&unk_49DC290;
  qword_5000C10 = (__int64)&unk_49D9728;
  qword_5000C40 = (__int64)nullsub_24;
  qword_5000B80 = (__int64)&unk_49DBF10;
  qword_5000C38 = (__int64)sub_984050;
  qword_5000C18 = 0;
  sub_C53080(&qword_5000B80, "lsr-loop-level", 14);
  LODWORD(qword_5000C08) = 1;
  qword_5000BA8 = (__int64)"loop strength reduce on loop levels";
  BYTE4(qword_5000C18) = 1;
  LODWORD(qword_5000C18) = 1;
  qword_5000BB0 = 35;
  byte_5000B8C = byte_5000B8C & 0x9F | 0x20;
  sub_C53130(&qword_5000B80);
  __cxa_atexit(sub_984970, &qword_5000B80, &qword_4A427C0);
  v5 = "Consider outer-loop IV as loop-invariant in LSR";
  v4[0] = &v2;
  LODWORD(v3) = 1;
  v6 = 47;
  LOBYTE(v2) = 1;
  sub_285DA00(&unk_5000AA0, "lsr-outer-loop-invariant", v4, &v5, &v3);
  __cxa_atexit(sub_984900, &unk_5000AA0, &qword_4A427C0);
  sub_D95050(&qword_50009C0, 0, 0);
  qword_5000A50 = (__int64)&unk_49D9748;
  qword_5000A80 = (__int64)nullsub_23;
  qword_50009C0 = (__int64)&unk_49DC090;
  qword_5000A60 = (__int64)&unk_49DC1D0;
  qword_5000A78 = (__int64)sub_984030;
  qword_5000A48 = 0;
  qword_5000A58 = 0;
  sub_C53080(&qword_50009C0, "lsr-simplify-code", 17);
  LOBYTE(qword_5000A48) = 1;
  qword_50009E8 = (__int64)"Ignore the Factor -1 for simplifying code in LSR";
  LOWORD(qword_5000A58) = 257;
  qword_50009F0 = 48;
  byte_50009CC = byte_50009CC & 0x9F | 0x20;
  sub_C53130(&qword_50009C0);
  __cxa_atexit(sub_984900, &qword_50009C0, &qword_4A427C0);
  sub_D95050(&qword_50008E0, 0, 0);
  qword_5000970 = (__int64)&unk_49D9748;
  qword_50009A0 = (__int64)nullsub_23;
  qword_50008E0 = (__int64)&unk_49DC090;
  qword_5000980 = (__int64)&unk_49DC1D0;
  qword_5000998 = (__int64)sub_984030;
  qword_5000968 = 0;
  qword_5000978 = 0;
  sub_C53080(&qword_50008E0, "do-lsr-64-bit", 13);
  qword_5000908 = (__int64)"loop strength reduce for 64-bit";
  LOBYTE(qword_5000968) = 1;
  LOWORD(qword_5000978) = 257;
  qword_5000910 = 31;
  byte_50008EC = byte_50008EC & 0x9F | 0x20;
  sub_C53130(&qword_50008E0);
  return __cxa_atexit(sub_984900, &qword_50008E0, &qword_4A427C0);
}
