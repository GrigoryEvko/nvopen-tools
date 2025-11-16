// Function: ctor_336
// Address: 0x509ca0
//
int ctor_336()
{
  char v1; // [rsp+3h] [rbp-4Dh] BYREF
  int v2; // [rsp+4h] [rbp-4Ch] BYREF
  char *v3; // [rsp+8h] [rbp-48h] BYREF
  const char *v4; // [rsp+10h] [rbp-40h] BYREF
  __int64 v5; // [rsp+18h] [rbp-38h]

  v4 = "Use TargetSchedModel for latency lookup";
  v3 = &v1;
  v5 = 39;
  v1 = 1;
  v2 = 1;
  sub_1F4C2E0(&unk_4FCE180, "schedmodel", &v2, &v3, &v4);
  __cxa_atexit(sub_12EDEC0, &unk_4FCE180, &qword_4A427C0);
  v3 = &v1;
  v4 = "Use InstrItineraryData for latency lookup";
  v5 = 41;
  v1 = 1;
  v2 = 1;
  sub_1F4C2E0(&unk_4FCE0A0, "scheditins", &v2, &v3, &v4);
  return __cxa_atexit(sub_12EDEC0, &unk_4FCE0A0, &qword_4A427C0);
}
