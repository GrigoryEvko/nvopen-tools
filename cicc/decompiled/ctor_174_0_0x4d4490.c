// Function: ctor_174_0
// Address: 0x4d4490
//
int ctor_174_0()
{
  int v0; // eax
  int v1; // eax
  int v2; // ecx
  int v3; // ecx
  int v4; // ecx
  int v5; // ecx
  int v6; // esi
  int v7; // ebx
  int v8; // eax
  char v10; // [rsp+3Bh] [rbp-F5h] BYREF
  int v11; // [rsp+3Ch] [rbp-F4h] BYREF
  _QWORD v12[2]; // [rsp+40h] [rbp-F0h] BYREF
  const char *v13; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v14; // [rsp+58h] [rbp-D8h]
  _QWORD v15[2]; // [rsp+60h] [rbp-D0h] BYREF
  int v16; // [rsp+70h] [rbp-C0h]
  const char *v17; // [rsp+78h] [rbp-B8h]
  __int64 v18; // [rsp+80h] [rbp-B0h]
  char *v19; // [rsp+88h] [rbp-A8h]
  __int64 v20; // [rsp+90h] [rbp-A0h]
  int v21; // [rsp+98h] [rbp-98h]
  const char *v22; // [rsp+A0h] [rbp-90h]
  __int64 v23; // [rsp+A8h] [rbp-88h]
  char *v24; // [rsp+B0h] [rbp-80h]
  __int64 v25; // [rsp+B8h] [rbp-78h]
  int v26; // [rsp+C0h] [rbp-70h]
  const char *v27; // [rsp+C8h] [rbp-68h]
  __int64 v28; // [rsp+D0h] [rbp-60h]

  qword_4FA5940 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA594C &= 0xF000u;
  qword_4FA5950 = 0;
  qword_4FA5958 = 0;
  qword_4FA5960 = 0;
  qword_4FA5968 = 0;
  qword_4FA5970 = 0;
  dword_4FA5948 = v0;
  qword_4FA5978 = 0;
  qword_4FA5988 = (__int64)qword_4FA01C0;
  qword_4FA5998 = (__int64)&unk_4FA59B8;
  qword_4FA59A0 = (__int64)&unk_4FA59B8;
  qword_4FA59E0 = (__int64)&byte_4FA59F0;
  qword_4FA5A08 = (__int64)&byte_4FA5A18;
  qword_4FA5980 = 0;
  qword_4FA5990 = 0;
  qword_4FA5A00 = (__int64)&unk_49EED10;
  qword_4FA59A8 = 4;
  dword_4FA59B0 = 0;
  qword_4FA5940 = (__int64)&unk_49EEBF0;
  byte_4FA59D8 = 0;
  qword_4FA59E8 = 0;
  qword_4FA5A30 = (__int64)&unk_49EEE90;
  qword_4FA5A38 = (__int64)&byte_4FA5A48;
  byte_4FA59F0 = 0;
  qword_4FA5A10 = 0;
  byte_4FA5A18 = 0;
  byte_4FA5A28 = 0;
  qword_4FA5A40 = 0;
  byte_4FA5A48 = 0;
  sub_16B8280(&byte_4FA5A48 - 264, "pgo-test-profile-file", 21);
  sub_17E3D90(&v13, byte_3F871B3);
  sub_2240AE0(&qword_4FA59E0, &v13);
  byte_4FA5A28 = 1;
  sub_2240AE0(&qword_4FA5A08, &v13);
  sub_2240A30(&v13);
  qword_4FA5980 = 8;
  qword_4FA5970 = 70;
  LOBYTE(word_4FA594C) = word_4FA594C & 0x9F | 0x20;
  qword_4FA5978 = (__int64)"filename";
  qword_4FA5968 = (__int64)"Specify the path of profile data file. This ismainly for test purpose.";
  sub_16B88A0(&qword_4FA5940);
  __cxa_atexit(sub_12F0C20, &qword_4FA5940, &qword_4A427C0);
  qword_4FA5860 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA586C &= 0xF000u;
  word_4FA5910 = 256;
  qword_4FA5870 = 0;
  qword_4FA5878 = 0;
  qword_4FA5880 = 0;
  dword_4FA5868 = v1;
  qword_4FA5888 = 0;
  qword_4FA58A8 = (__int64)qword_4FA01C0;
  qword_4FA58B8 = (__int64)&unk_4FA58D8;
  qword_4FA58C0 = (__int64)&unk_4FA58D8;
  qword_4FA5890 = 0;
  qword_4FA5898 = 0;
  qword_4FA5908 = (__int64)&unk_49E74E8;
  qword_4FA5860 = (__int64)&unk_49EEC70;
  qword_4FA5918 = (__int64)&unk_49EEDB0;
  qword_4FA58A0 = 0;
  qword_4FA58B0 = 0;
  qword_4FA58C8 = 4;
  dword_4FA58D0 = 0;
  byte_4FA58F8 = 0;
  byte_4FA5900 = 0;
  sub_16B8280(&qword_4FA5860, "disable-vp", 10);
  word_4FA5910 = 257;
  byte_4FA5900 = 1;
  qword_4FA5890 = 23;
  qword_4FA5888 = (__int64)"Disable Value Profiling";
  LOBYTE(word_4FA586C) = word_4FA586C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FA5860);
  __cxa_atexit(sub_12EDEC0, &qword_4FA5860, &qword_4A427C0);
  qword_4FA5780 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_4FA57D8 = (__int64)&unk_4FA57F8;
  qword_4FA57E0 = (__int64)&unk_4FA57F8;
  word_4FA578C &= 0xF000u;
  qword_4FA5790 = 0;
  qword_4FA5828 = (__int64)&unk_49E74A8;
  dword_4FA5788 = v2;
  qword_4FA57C8 = (__int64)qword_4FA01C0;
  qword_4FA5780 = (__int64)&unk_49EEAF0;
  qword_4FA5838 = (__int64)&unk_49EEE10;
  qword_4FA5798 = 0;
  qword_4FA57A0 = 0;
  qword_4FA57A8 = 0;
  qword_4FA57B0 = 0;
  qword_4FA57B8 = 0;
  qword_4FA57C0 = 0;
  qword_4FA57D0 = 0;
  qword_4FA57E8 = 4;
  dword_4FA57F0 = 0;
  byte_4FA5818 = 0;
  dword_4FA5820 = 0;
  byte_4FA5834 = 1;
  dword_4FA5830 = 0;
  sub_16B8280(&qword_4FA5780, "icp-max-annotations", 19);
  dword_4FA5820 = 3;
  byte_4FA5834 = 1;
  dword_4FA5830 = 3;
  qword_4FA57B0 = 61;
  qword_4FA57A8 = (__int64)"Max number of annotations for a single indirect call callsite";
  LOBYTE(word_4FA578C) = word_4FA578C & 0x98 | 0x21;
  sub_16B88A0(&qword_4FA5780);
  __cxa_atexit(sub_12EDE60, &qword_4FA5780, &qword_4A427C0);
  qword_4FA56A0 = (__int64)&unk_49EED30;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_4FA56F8 = (__int64)&unk_4FA5718;
  word_4FA56AC &= 0xF000u;
  qword_4FA5700 = (__int64)&unk_4FA5718;
  qword_4FA5748 = (__int64)&unk_49E74A8;
  qword_4FA56B0 = 0;
  dword_4FA56A8 = v3;
  qword_4FA5758 = (__int64)&unk_49EEE10;
  qword_4FA56E8 = (__int64)qword_4FA01C0;
  qword_4FA56B8 = 0;
  qword_4FA56A0 = (__int64)&unk_49EEAF0;
  qword_4FA56C0 = 0;
  qword_4FA56C8 = 0;
  qword_4FA56D0 = 0;
  qword_4FA56D8 = 0;
  qword_4FA56E0 = 0;
  qword_4FA56F0 = 0;
  qword_4FA5708 = 4;
  dword_4FA5710 = 0;
  byte_4FA5738 = 0;
  dword_4FA5740 = 0;
  byte_4FA5754 = 1;
  dword_4FA5750 = 0;
  sub_16B8280((char *)&unk_4FA5718 - 120, "memop-max-annotations", 21);
  dword_4FA5740 = 4;
  byte_4FA5754 = 1;
  dword_4FA5750 = 4;
  qword_4FA56D0 = 68;
  qword_4FA56C8 = (__int64)"Max number of preicise value annotations for a single memopintrinsic";
  LOBYTE(word_4FA56AC) = word_4FA56AC & 0x98 | 0x21;
  sub_16B88A0(&qword_4FA56A0);
  __cxa_atexit(sub_12EDE60, &qword_4FA56A0, &qword_4A427C0);
  qword_4FA55C0 = (__int64)&unk_49EED30;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_4FA5618 = (__int64)&unk_4FA5638;
  qword_4FA5620 = (__int64)&unk_4FA5638;
  word_4FA5670 = 256;
  word_4FA55CC &= 0xF000u;
  qword_4FA5668 = (__int64)&unk_49E74E8;
  dword_4FA55C8 = v4;
  qword_4FA55C0 = (__int64)&unk_49EEC70;
  qword_4FA5608 = (__int64)qword_4FA01C0;
  qword_4FA5678 = (__int64)&unk_49EEDB0;
  qword_4FA55D0 = 0;
  qword_4FA55D8 = 0;
  qword_4FA55E0 = 0;
  qword_4FA55E8 = 0;
  qword_4FA55F0 = 0;
  qword_4FA55F8 = 0;
  qword_4FA5600 = 0;
  qword_4FA5610 = 0;
  qword_4FA5628 = 4;
  dword_4FA5630 = 0;
  byte_4FA5658 = 0;
  byte_4FA5660 = 0;
  sub_16B8280(&qword_4FA55C0, "do-comdat-renaming", 18);
  word_4FA5670 = 256;
  qword_4FA55E8 = (__int64)"Append function hash to the name of COMDAT function to avoid function hash mismatch due to the preinliner";
  byte_4FA5660 = 0;
  qword_4FA55F0 = 105;
  LOBYTE(word_4FA55CC) = word_4FA55CC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FA55C0);
  __cxa_atexit(sub_12EDEC0, &qword_4FA55C0, &qword_4A427C0);
  qword_4FA54E0 = (__int64)&unk_49EED30;
  v5 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA5590 = 256;
  qword_4FA5538 = (__int64)&unk_4FA5558;
  word_4FA54EC &= 0xF000u;
  qword_4FA5598 = (__int64)&unk_49EEDB0;
  qword_4FA5540 = (__int64)&unk_4FA5558;
  dword_4FA54E8 = v5;
  qword_4FA5528 = (__int64)qword_4FA01C0;
  qword_4FA5588 = (__int64)&unk_49E74E8;
  qword_4FA54E0 = (__int64)&unk_49EEC70;
  qword_4FA54F0 = 0;
  qword_4FA54F8 = 0;
  qword_4FA5500 = 0;
  qword_4FA5508 = 0;
  qword_4FA5510 = 0;
  qword_4FA5518 = 0;
  qword_4FA5520 = 0;
  qword_4FA5530 = 0;
  qword_4FA5548 = 4;
  dword_4FA5550 = 0;
  byte_4FA5578 = 0;
  byte_4FA5580 = 0;
  sub_16B8280((char *)&unk_4FA5558 - 120, "pgo-warn-missing-function", 25);
  word_4FA5590 = 256;
  byte_4FA5580 = 0;
  qword_4FA5510 = 81;
  qword_4FA5508 = (__int64)"Use this option to turn on/off warnings about missing profile data for functions.";
  LOBYTE(word_4FA54EC) = word_4FA54EC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FA54E0);
  __cxa_atexit(sub_12EDEC0, &qword_4FA54E0, &qword_4A427C0);
  v14 = 67;
  v13 = "Use this option to turn off/on warnings about profile cfg mismatch.";
  v12[0] = &v10;
  v11 = 1;
  v10 = 0;
  sub_17E7FC0(&unk_4FA5400, "no-pgo-warn-mismatch", v12, &v11, &v13);
  __cxa_atexit(sub_12EDEC0, &unk_4FA5400, &qword_4A427C0);
  qword_4FA5320 = (__int64)&unk_49EED30;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_4FA5378 = (__int64)&unk_4FA5398;
  qword_4FA5380 = (__int64)&unk_4FA5398;
  word_4FA53D0 = 256;
  qword_4FA53D8 = (__int64)&unk_49EEDB0;
  qword_4FA5320 = (__int64)&unk_49EEC70;
  dword_4FA5328 = v6;
  word_4FA532C &= 0xF000u;
  qword_4FA5368 = (__int64)qword_4FA01C0;
  qword_4FA53C8 = (__int64)&unk_49E74E8;
  qword_4FA5330 = 0;
  qword_4FA5338 = 0;
  qword_4FA5340 = 0;
  qword_4FA5348 = 0;
  qword_4FA5350 = 0;
  qword_4FA5358 = 0;
  qword_4FA5360 = 0;
  qword_4FA5370 = 0;
  qword_4FA5388 = 4;
  dword_4FA5390 = 0;
  byte_4FA53B8 = 0;
  byte_4FA53C0 = 0;
  sub_16B8280(&qword_4FA5320, "no-pgo-warn-mismatch-comdat", 27);
  word_4FA53D0 = 257;
  qword_4FA5348 = (__int64)"The option is used to turn on/off warnings about hash mismatch for comdat functions.";
  byte_4FA53C0 = 1;
  qword_4FA5350 = 84;
  LOBYTE(word_4FA532C) = word_4FA532C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FA5320);
  __cxa_atexit(sub_12EDEC0, &qword_4FA5320, &qword_4A427C0);
  qword_4FA5240 = (__int64)&unk_49EED30;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA524C &= 0xF000u;
  qword_4FA5298 = (__int64)&unk_4FA52B8;
  qword_4FA52A0 = (__int64)&unk_4FA52B8;
  qword_4FA5250 = 0;
  qword_4FA52F8 = (__int64)&unk_49EEDB0;
  qword_4FA52E8 = (__int64)&unk_49E74E8;
  dword_4FA5248 = v7;
  qword_4FA5240 = (__int64)&unk_49EEC70;
  qword_4FA5288 = (__int64)qword_4FA01C0;
  word_4FA52F0 = 256;
  qword_4FA5258 = 0;
  qword_4FA5260 = 0;
  qword_4FA5268 = 0;
  qword_4FA5270 = 0;
  qword_4FA5278 = 0;
  qword_4FA5280 = 0;
  qword_4FA5290 = 0;
  qword_4FA52A8 = 4;
  dword_4FA52B0 = 0;
  byte_4FA52D8 = 0;
  byte_4FA52E0 = 0;
  sub_16B8280(&qword_4FA5240, "pgo-instr-select", 16);
  byte_4FA52E0 = 0;
  word_4FA52F0 = 256;
  qword_4FA5270 = 67;
  LOBYTE(word_4FA524C) = word_4FA524C & 0x9F | 0x20;
  qword_4FA5268 = (__int64)"Use this option to turn on/off SELECT instruction instrumentation. ";
  sub_16B88A0(&qword_4FA5240);
  __cxa_atexit(sub_12EDEC0, &qword_4FA5240, &qword_4A427C0);
  v13 = (const char *)v15;
  v15[0] = "none";
  v17 = "do not show.";
  v19 = "graph";
  v22 = "show a graph.";
  v24 = "text";
  v27 = "show in text.";
  v14 = 0x400000003LL;
  v15[1] = 4;
  v16 = 0;
  v18 = 12;
  v20 = 5;
  v21 = 1;
  v23 = 13;
  v25 = 4;
  v26 = 2;
  v28 = 13;
  v12[0] = "A boolean option to show CFG dag or text with raw profile counts from profile data. See also option -pgo-view"
           "-counts. To limit graph display to only one function, use filtering option -view-bfi-func-name.";
  v12[1] = 204;
  v11 = 1;
  sub_17E8140(&unk_4FA4FE0, "pgo-view-raw-counts", &v11, v12, &v13);
  if ( v13 != (const char *)v15 )
    _libc_free(v13, "pgo-view-raw-counts");
  __cxa_atexit(sub_1367990, &unk_4FA4FE0, &qword_4A427C0);
  qword_4FA4F00 = (__int64)&unk_49EED30;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA4F0C &= 0xF000u;
  qword_4FA4F10 = 0;
  qword_4FA4F18 = 0;
  qword_4FA4F20 = 0;
  qword_4FA4F28 = 0;
  dword_4FA4F08 = v8;
  qword_4FA4F30 = 0;
  qword_4FA4F48 = (__int64)qword_4FA01C0;
  qword_4FA4F58 = (__int64)&unk_4FA4F78;
  qword_4FA4F60 = (__int64)&unk_4FA4F78;
  qword_4FA4F38 = 0;
  qword_4FA4F40 = 0;
  qword_4FA4FA8 = (__int64)&unk_49E74E8;
  word_4FA4FB0 = 256;
  qword_4FA4F50 = 0;
  byte_4FA4F98 = 0;
  qword_4FA4F00 = (__int64)&unk_49EEC70;
  qword_4FA4F68 = 4;
  byte_4FA4FA0 = 0;
  qword_4FA4FB8 = (__int64)&unk_49EEDB0;
  dword_4FA4F70 = 0;
  sub_16B8280(&qword_4FA4F00, "pgo-instr-memop", 15);
  word_4FA4FB0 = 257;
  byte_4FA4FA0 = 1;
  qword_4FA4F30 = 63;
  LOBYTE(word_4FA4F0C) = word_4FA4F0C & 0x9F | 0x20;
  qword_4FA4F28 = (__int64)"Use this option to turn on/off memory intrinsic size profiling.";
  sub_16B88A0(&qword_4FA4F00);
  __cxa_atexit(sub_12EDEC0, &qword_4FA4F00, &qword_4A427C0);
  v14 = 139;
  v13 = "When this option is on, the annotated branch probability will be emitted as optimization remarks: -{Rpass|pass-r"
        "emarks}=pgo-instrumentation";
  v11 = 1;
  v10 = 0;
  v12[0] = &v10;
  sub_17E7FC0(&unk_4FA4E20, "pgo-emit-branch-prob", v12, &v11, &v13);
  return __cxa_atexit(sub_12EDEC0, &unk_4FA4E20, &qword_4A427C0);
}
