// Function: ctor_427_0
// Address: 0x5358c0
//
int ctor_427_0()
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
  char *v11; // [rsp+88h] [rbp-A8h]
  __int64 v12; // [rsp+90h] [rbp-A0h]
  int v13; // [rsp+98h] [rbp-98h]
  const char *v14; // [rsp+A0h] [rbp-90h]
  __int64 v15; // [rsp+A8h] [rbp-88h]
  char *v16; // [rsp+B0h] [rbp-80h]
  __int64 v17; // [rsp+B8h] [rbp-78h]
  int v18; // [rsp+C0h] [rbp-70h]
  const char *v19; // [rsp+C8h] [rbp-68h]
  __int64 v20; // [rsp+D0h] [rbp-60h]

  sub_D95050(&qword_4FF40E0, 0, 0);
  qword_4FF4168 = (__int64)&byte_4FF4178;
  qword_4FF4190 = (__int64)&byte_4FF41A0;
  qword_4FF4170 = 0;
  byte_4FF4178 = 0;
  qword_4FF4188 = (__int64)&unk_49DC130;
  qword_4FF4198 = 0;
  byte_4FF41A0 = 0;
  qword_4FF40E0 = (__int64)&unk_49DC010;
  byte_4FF41B0 = 0;
  qword_4FF41B8 = (__int64)&unk_49DC350;
  qword_4FF41D8 = (__int64)nullsub_92;
  qword_4FF41D0 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4FF40E0, "memprof-dot-file-path-prefix", 28);
  sub_263F570(&v5, byte_3F871B3);
  sub_2240AE0(&qword_4FF4168, &v5);
  byte_4FF41B0 = 1;
  sub_2240AE0(&qword_4FF4190, &v5);
  sub_2240A30(&v5);
  qword_4FF4120 = 8;
  qword_4FF4110 = 49;
  byte_4FF40EC = byte_4FF40EC & 0x9F | 0x20;
  qword_4FF4118 = (__int64)"filename";
  qword_4FF4108 = (__int64)"Specify the path prefix of the MemProf dot files.";
  sub_C53130(&qword_4FF40E0);
  __cxa_atexit(sub_BC5A40, &qword_4FF40E0, &qword_4A427C0);
  v6 = 26;
  v5 = "Export graph to dot files.";
  v4[0] = &v2;
  LODWORD(v3) = 1;
  LOBYTE(v2) = 0;
  sub_23A1680(&unk_4FF4000, "memprof-export-to-dot", v4, &v3, &v5);
  __cxa_atexit(sub_984900, &unk_4FF4000, &qword_4A427C0);
  v5 = (const char *)v7;
  v7[0] = "all";
  v9 = "Export full callsite graph";
  v11 = "alloc";
  v14 = "Export only nodes with contexts feeding given -memprof-dot-alloc-id";
  v16 = "context";
  v19 = "Export only nodes with given -memprof-dot-context-id";
  v6 = 0x400000003LL;
  v3 = &v2;
  v7[1] = 3;
  v8 = 0;
  v10 = 26;
  v12 = 5;
  v13 = 1;
  v15 = 67;
  v17 = 7;
  v18 = 2;
  v20 = 52;
  v2 = 0;
  v1 = 1;
  v4[0] = "Scope of graph to export to dot";
  v4[1] = 31;
  sub_265ECC0(&unk_4FF3DA0, "memprof-dot-scope", v4, &v1, &v3, &v5);
  if ( v5 != (const char *)v7 )
    _libc_free(v5, "memprof-dot-scope");
  __cxa_atexit(sub_26405D0, &unk_4FF3DA0, &qword_4A427C0);
  sub_D95050(&qword_4FF3CC0, 0, 0);
  qword_4FF3D48 = 0;
  qword_4FF3D58 = 0;
  qword_4FF3D50 = (__int64)&unk_49D9728;
  qword_4FF3CC0 = (__int64)&unk_49DBF10;
  qword_4FF3D80 = (__int64)nullsub_24;
  qword_4FF3D60 = (__int64)&unk_49DC290;
  qword_4FF3D78 = (__int64)sub_984050;
  sub_C53080(&qword_4FF3CC0, "memprof-dot-alloc-id", 20);
  LODWORD(qword_4FF3D48) = 0;
  BYTE4(qword_4FF3D58) = 1;
  LODWORD(qword_4FF3D58) = 0;
  qword_4FF3CF0 = 91;
  byte_4FF3CCC = byte_4FF3CCC & 0x9F | 0x20;
  qword_4FF3CE8 = (__int64)"Id of alloc to export if -memprof-dot-scope=alloc or to highlight if -memprof-dot-scope=all";
  sub_C53130(&qword_4FF3CC0);
  __cxa_atexit(sub_984970, &qword_4FF3CC0, &qword_4A427C0);
  sub_D95050(&qword_4FF3BE0, 0, 0);
  qword_4FF3CA0 = (__int64)nullsub_24;
  qword_4FF3C70 = (__int64)&unk_49D9728;
  qword_4FF3BE0 = (__int64)&unk_49DBF10;
  qword_4FF3C80 = (__int64)&unk_49DC290;
  qword_4FF3C98 = (__int64)sub_984050;
  qword_4FF3C68 = 0;
  qword_4FF3C78 = 0;
  sub_C53080(&qword_4FF3BE0, "memprof-dot-context-id", 22);
  LODWORD(qword_4FF3C68) = 0;
  BYTE4(qword_4FF3C78) = 1;
  LODWORD(qword_4FF3C78) = 0;
  qword_4FF3C10 = 79;
  byte_4FF3BEC = byte_4FF3BEC & 0x9F | 0x20;
  qword_4FF3C08 = (__int64)"Id of context to export if -memprof-dot-scope=context or to highlight otherwise";
  sub_C53130(&qword_4FF3BE0);
  __cxa_atexit(sub_984970, &qword_4FF3BE0, &qword_4A427C0);
  sub_D95050(&qword_4FF3B00, 0, 0);
  qword_4FF3B88 = 0;
  qword_4FF3BC0 = (__int64)nullsub_23;
  qword_4FF3B90 = (__int64)&unk_49D9748;
  qword_4FF3BB8 = (__int64)sub_984030;
  qword_4FF3B98 = 0;
  qword_4FF3B00 = (__int64)&unk_49DC090;
  qword_4FF3BA0 = (__int64)&unk_49DC1D0;
  sub_C53080(&qword_4FF3B00, "memprof-dump-ccg", 16);
  LOWORD(qword_4FF3B98) = 256;
  qword_4FF3B28 = (__int64)"Dump CallingContextGraph to stdout after each stage.";
  LOBYTE(qword_4FF3B88) = 0;
  qword_4FF3B30 = 52;
  byte_4FF3B0C = byte_4FF3B0C & 0x9F | 0x20;
  sub_C53130(&qword_4FF3B00);
  __cxa_atexit(sub_984900, &qword_4FF3B00, &qword_4A427C0);
  sub_D95050(&qword_4FF3A20, 0, 0);
  qword_4FF3AC0 = (__int64)&unk_49DC1D0;
  qword_4FF3AE0 = (__int64)nullsub_23;
  qword_4FF3AD8 = (__int64)sub_984030;
  qword_4FF3A20 = (__int64)&unk_49DC090;
  qword_4FF3AB0 = (__int64)&unk_49D9748;
  qword_4FF3AA8 = 0;
  qword_4FF3AB8 = 0;
  sub_C53080(&qword_4FF3A20, "memprof-verify-ccg", 18);
  LOBYTE(qword_4FF3AA8) = 0;
  LOWORD(qword_4FF3AB8) = 256;
  qword_4FF3A48 = (__int64)"Perform verification checks on CallingContextGraph.";
  qword_4FF3A50 = 51;
  byte_4FF3A2C = byte_4FF3A2C & 0x9F | 0x20;
  sub_C53130(&qword_4FF3A20);
  __cxa_atexit(sub_984900, &qword_4FF3A20, &qword_4A427C0);
  sub_D95050(&qword_4FF3940, 0, 0);
  qword_4FF39E0 = (__int64)&unk_49DC1D0;
  qword_4FF3A00 = (__int64)nullsub_23;
  qword_4FF39F8 = (__int64)sub_984030;
  qword_4FF39D0 = (__int64)&unk_49D9748;
  qword_4FF3940 = (__int64)&unk_49DC090;
  qword_4FF39C8 = 0;
  qword_4FF39D8 = 0;
  sub_C53080(&qword_4FF3940, "memprof-verify-nodes", 20);
  qword_4FF3968 = (__int64)"Perform frequent verification checks on nodes.";
  LOWORD(qword_4FF39D8) = 256;
  LOBYTE(qword_4FF39C8) = 0;
  qword_4FF3970 = 46;
  byte_4FF394C = byte_4FF394C & 0x9F | 0x20;
  sub_C53130(&qword_4FF3940);
  __cxa_atexit(sub_984900, &qword_4FF3940, &qword_4A427C0);
  sub_D95050(&qword_4FF3840, 0, 0);
  qword_4FF38D0 = 0;
  qword_4FF38C8 = (__int64)&byte_4FF38D8;
  qword_4FF38F0 = (__int64)&byte_4FF3900;
  qword_4FF38E8 = (__int64)&unk_49DC130;
  qword_4FF3840 = (__int64)&unk_49DC010;
  qword_4FF3938 = (__int64)nullsub_92;
  qword_4FF3930 = (__int64)sub_BC4D70;
  qword_4FF3918 = (__int64)&unk_49DC350;
  byte_4FF38D8 = 0;
  qword_4FF38F8 = 0;
  byte_4FF3900 = 0;
  byte_4FF3910 = 0;
  sub_C53080(&qword_4FF3840, "memprof-import-summary", 22);
  qword_4FF3870 = 61;
  qword_4FF3868 = (__int64)"Import summary to use for testing the ThinLTO backend via opt";
  byte_4FF384C = byte_4FF384C & 0x9F | 0x20;
  sub_C53130(&qword_4FF3840);
  __cxa_atexit(sub_BC5A40, &qword_4FF3840, &qword_4A427C0);
  sub_D95050(&qword_4FF3760, 0, 0);
  qword_4FF37E8 = 0;
  qword_4FF3820 = (__int64)nullsub_24;
  qword_4FF3818 = (__int64)sub_984050;
  qword_4FF37F0 = (__int64)&unk_49D9728;
  qword_4FF3760 = (__int64)&unk_49DBF10;
  qword_4FF3800 = (__int64)&unk_49DC290;
  qword_4FF37F8 = 0;
  sub_C53080(&qword_4FF3760, "memprof-tail-call-search-depth", 30);
  qword_4FF3788 = (__int64)"Max depth to recursively search for missing frames through tail calls.";
  LODWORD(qword_4FF37E8) = 5;
  BYTE4(qword_4FF37F8) = 1;
  LODWORD(qword_4FF37F8) = 5;
  byte_4FF376C = byte_4FF376C & 0x9F | 0x20;
  qword_4FF3790 = 70;
  sub_C53130(&qword_4FF3760);
  __cxa_atexit(sub_984970, &qword_4FF3760, &qword_4A427C0);
  sub_D95050(&qword_4FF3680, 0, 0);
  qword_4FF3720 = (__int64)&unk_49DC1D0;
  qword_4FF3740 = (__int64)nullsub_23;
  qword_4FF3738 = (__int64)sub_984030;
  qword_4FF3680 = (__int64)&unk_49DC090;
  qword_4FF3710 = (__int64)&unk_49D9748;
  qword_4FF3708 = 0;
  qword_4FF3718 = 0;
  sub_C53080(&qword_4FF3680, "memprof-allow-recursive-callsites", 33);
  LOWORD(qword_4FF3718) = 257;
  LOBYTE(qword_4FF3708) = 1;
  qword_4FF36B0 = 55;
  qword_4FF36A8 = (__int64)"Allow cloning of callsites involved in recursive cycles";
  byte_4FF368C = byte_4FF368C & 0x9F | 0x20;
  sub_C53130(&qword_4FF3680);
  __cxa_atexit(sub_984900, &qword_4FF3680, &qword_4A427C0);
  v6 = 50;
  v5 = "Allow cloning of contexts through recursive cycles";
  v4[0] = &v2;
  LODWORD(v3) = 1;
  LOBYTE(v2) = 1;
  sub_2646830(&unk_4FF35A0, "memprof-clone-recursive-contexts", v4, &v3, &v5);
  __cxa_atexit(sub_984900, &unk_4FF35A0, &qword_4A427C0);
  v6 = 49;
  v5 = "Allow cloning of contexts having recursive cycles";
  v4[0] = &v2;
  LODWORD(v3) = 1;
  LOBYTE(v2) = 1;
  sub_2646830(&unk_4FF34C0, "memprof-allow-recursive-contexts", v4, &v3, &v5);
  __cxa_atexit(sub_984900, &unk_4FF34C0, &qword_4A427C0);
  sub_D95050(qword_4FF33E0, 0, 0);
  qword_4FF33E0[0] = &unk_49DC090;
  qword_4FF33E0[18] = &unk_49D9748;
  qword_4FF33E0[20] = &unk_49DC1D0;
  qword_4FF33E0[24] = nullsub_23;
  qword_4FF33E0[23] = sub_984030;
  qword_4FF33E0[17] = 0;
  qword_4FF33E0[19] = 0;
  sub_C53080(qword_4FF33E0, "enable-memprof-context-disambiguation", 37);
  LOWORD(qword_4FF33E0[19]) = 256;
  qword_4FF33E0[5] = "Enable MemProf context disambiguation";
  LOBYTE(qword_4FF33E0[17]) = 0;
  qword_4FF33E0[6] = 37;
  BYTE4(qword_4FF33E0[1]) = BYTE4(qword_4FF33E0[1]) & 0x98 | 0x21;
  sub_C53130(qword_4FF33E0);
  __cxa_atexit(sub_984900, qword_4FF33E0, &qword_4A427C0);
  v6 = 45;
  v5 = "Linking with hot/cold operator new interfaces";
  v4[0] = &v2;
  LODWORD(v3) = 1;
  LOBYTE(v2) = 0;
  sub_23A1680(&unk_4FF3300, "supports-hot-cold-new", v4, &v3, &v5);
  __cxa_atexit(sub_984900, &unk_4FF3300, &qword_4A427C0);
  sub_D95050(&qword_4FF3220, 0, 0);
  qword_4FF32C0 = (__int64)&unk_49DC1D0;
  qword_4FF32B0 = (__int64)&unk_49D9748;
  qword_4FF3220 = (__int64)&unk_49DC090;
  qword_4FF32E0 = (__int64)nullsub_23;
  qword_4FF32A8 = 0;
  qword_4FF32D8 = (__int64)sub_984030;
  qword_4FF32B8 = 0;
  sub_C53080(&qword_4FF3220, "memprof-require-definition-for-promotion", 40);
  LOWORD(qword_4FF32B8) = 256;
  LOBYTE(qword_4FF32A8) = 0;
  qword_4FF3250 = 64;
  byte_4FF322C = byte_4FF322C & 0x9F | 0x20;
  qword_4FF3248 = (__int64)"Require target function definition when promoting indirect calls";
  sub_C53130(&qword_4FF3220);
  __cxa_atexit(sub_984900, &qword_4FF3220, &qword_4A427C0);
  sub_263F570(&qword_4FF3200, ".memprof.");
  return __cxa_atexit(sub_2240D50, &qword_4FF3200, &qword_4A427C0);
}
