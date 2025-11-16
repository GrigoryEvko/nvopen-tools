// Function: ctor_014
// Address: 0x453ed0
//
int ctor_014()
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
  sub_A03D00(&unk_4F805A0, "import-full-type-definitions", &v3, &v2, &v4);
  __cxa_atexit(sub_984900, &unk_4F805A0, &qword_4A427C0);
  v3 = &v1;
  v4 = "Force disable the lazy-loading on-demand of metadata when loading bitcode for importing.";
  v5 = 88;
  v2 = 1;
  v1 = 0;
  sub_A03D00(&unk_4F804C0, "disable-ondemand-mds-loading", &v3, &v2, &v4);
  return __cxa_atexit(sub_984900, &unk_4F804C0, &qword_4A427C0);
}
