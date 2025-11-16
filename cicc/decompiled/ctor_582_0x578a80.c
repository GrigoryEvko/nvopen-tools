// Function: ctor_582
// Address: 0x578a80
//
int ctor_582()
{
  __int64 v0; // rdx
  int v2; // [rsp+Ch] [rbp-4h] BYREF

  v2 = 1;
  sub_2F44EC0(&unk_50238C0, "rafast-ignore-missing-defs", &v2);
  __cxa_atexit(sub_984900, &unk_50238C0, &qword_4A427C0);
  qword_5023890 = 4;
  v0 = unk_5023860;
  unk_5023860 = &qword_5023880;
  qword_5023888 = (__int64)"fast";
  qword_5023898 = (__int64)"fast register allocator";
  qword_50238A0 = 23;
  qword_50238A8 = (__int64)sub_2F42900;
  qword_5023880 = v0;
  if ( qword_5023870 )
    (*(void (__fastcall **)(_QWORD *, char *, __int64, __int64 (__fastcall *)(), const char *, __int64))(*qword_5023870 + 24LL))(
      qword_5023870,
      "fast",
      4,
      sub_2F42900,
      "fast register allocator",
      23);
  return __cxa_atexit(sub_2F41140, &qword_5023880, &qword_4A427C0);
}
