// Function: ctor_387
// Address: 0x51b5b0
//
int ctor_387()
{
  __int64 v0; // r14
  _QWORD v2[2]; // [rsp+0h] [rbp-60h] BYREF
  char v3; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v4[2]; // [rsp+20h] [rbp-40h] BYREF
  char v5; // [rsp+30h] [rbp-30h] BYREF

  qword_4FDC380 = (__int64)&qword_4FDC3B0;
  qword_4FDC388 = 1;
  qword_4FDC390 = 0;
  qword_4FDC398 = 0;
  dword_4FDC3A0 = 1065353216;
  qword_4FDC3A8 = 0;
  qword_4FDC3B0 = 0;
  __cxa_atexit(sub_8565C0, &qword_4FDC3B0 - 6, &qword_4A427C0);
  v0 = sub_C60B10();
  v4[0] = &v5;
  sub_2305260(v4, "How many AAs should be initialized");
  v2[0] = &v3;
  sub_2305260(v2, "num-abstract-attributes");
  sub_CF9810(v0, v2, v4);
  sub_2240A30(v2);
  sub_2240A30(v4);
  sub_C88F40(&unk_4FDC370, "^(default|thinlto-pre-link|thinlto|lto-pre-link|lto)<(O[0123sz])>$", 66, 0);
  __cxa_atexit(sub_C88FF0, &unk_4FDC370, &qword_4A427C0);
  v4[0] = "Print a '-passes' compatible string describing the pipeline (best-effort only).";
  v4[1] = 79;
  sub_2342660(&unk_4FDC2A0, "print-pipeline-passes", v4);
  return __cxa_atexit(sub_984900, &unk_4FDC2A0, &qword_4A427C0);
}
