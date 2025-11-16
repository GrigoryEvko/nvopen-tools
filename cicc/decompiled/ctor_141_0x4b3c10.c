// Function: ctor_141
// Address: 0x4b3c10
//
int ctor_141()
{
  char v1; // [rsp+3h] [rbp-4Dh] BYREF
  int v2; // [rsp+4h] [rbp-4Ch] BYREF
  char *v3; // [rsp+8h] [rbp-48h] BYREF
  const char *v4; // [rsp+10h] [rbp-40h] BYREF
  __int64 v5; // [rsp+18h] [rbp-38h]

  v4 = "Import full type definitions for ThinLTO.";
  v3 = &v1;
  v5 = 41;
  v2 = 1;
  v1 = 0;
  sub_1516190(&unk_4F9DCE0, "import-full-type-definitions", &v3, &v2, &v4);
  __cxa_atexit(sub_12EDEC0, &unk_4F9DCE0, &qword_4A427C0);
  v3 = &v1;
  v4 = "Force disable the lazy-loading on-demand of metadata when loading bitcode for importing.";
  v5 = 88;
  v2 = 1;
  v1 = 0;
  sub_1516190(&unk_4F9DC00, "disable-ondemand-mds-loading", &v3, &v2, &v4);
  return __cxa_atexit(sub_12EDEC0, &unk_4F9DC00, &qword_4A427C0);
}
